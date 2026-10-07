//! macOS VideoToolbox VP9 profile 0 backend.
//!
//! VideoToolbox decodes VP9 through a supplemental decoder that has to be registered before a
//! session can use it, and only on Macs whose media engine decodes VP9. A sample is handed over
//! as the whole chunk, so a superframe's hidden frames are decoded on the way to the frame it
//! shows, and each sample is waited for, since every VP9 sample shows exactly one frame.
//!
//! A `show_existing_frame` chunk decodes nothing, so it is not handed over at all: the backend
//! keeps the picture each reference slot holds, as the slots' frames are output, and shows the
//! named slot's picture again. A hidden frame is never output, so for a slot holding one the
//! chunks since the last key frame, which the backend keeps, are replayed through the software
//! decoder, which gives the same picture since VP9 decoding is exact. Encoders show a hidden
//! frame again rarely, so the replay is rare too.
//!
//! On a virtual Mac, VideoToolbox's VP9 decoder fails a decode with OSStatus -12909 or -19092
//! while another session in the process is decoding, so there every session's creation, decode
//! and teardown take turns (#580). Physical Macs decode in every session at once.

use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock, PoisonError};

use apple_cf::cf::{AsCFType, CFData, CFDictionary, CFNumber, CFString, CFType};
use apple_cf::cm::{CMBlockBuffer, CMFormatDescription, CMSampleBuffer};
use apple_cf::raw;
use videotoolbox::DecompressionSession;

use crate::vp9_dec::{ChunkInspector, DecodedPicture, Decoder, chunk_frames};
use crate::{
    CancellationToken, DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, Limits, Result,
    VideoDecoder, VideoDecoderConfig, VideoDimensions, Vp9CodecConfig,
};

const CODEC_TYPE_VP9: videotoolbox::ffi::CMVideoCodecType = u32::from_be_bytes(*b"vp09");

/// A decoded picture at the size VideoToolbox returned it, as three tightly packed 4:2:0 planes.
struct RawPicture {
    width: usize,
    height: usize,
    planes: [Vec<u8>; 3],
}

type OutputQueue = Arc<Mutex<Vec<Result<RawPicture>>>>;

fn register_decoder() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        videotoolbox::ffi::VTRegisterSupplementalVideoDecoderIfAvailable(CODEC_TYPE_VP9);
    });
}

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        old: *mut c_void,
        old_length: *mut usize,
        new: *mut c_void,
        new_length: usize,
    ) -> c_int;
}

/// Whether this Mac runs under a hypervisor, as the `macos-latest` GitHub runner does.
fn is_virtual_machine() -> bool {
    static VIRTUAL: OnceLock<bool> = OnceLock::new();
    *VIRTUAL.get_or_init(|| {
        let mut present: c_int = 0;
        let mut length = size_of::<c_int>();
        let status = unsafe {
            sysctlbyname(
                c"kern.hv_vmm_present".as_ptr(),
                (&raw mut present).cast(),
                &mut length,
                ptr::null_mut(),
                0,
            )
        };
        status == 0 && present != 0
    })
}

/// Holds every other VP9 session's VideoToolbox calls off on a virtual Mac, whose VP9 decoder
/// fails a decode while another session decodes; `None` on a physical Mac, which needs no turns.
fn exclusive() -> Option<MutexGuard<'static, ()>> {
    static SESSIONS: Mutex<()> = Mutex::new(());
    is_virtual_machine().then(|| SESSIONS.lock().unwrap_or_else(PoisonError::into_inner))
}

/// Whether this Mac advertises a hardware VP9 decoder.
pub(crate) fn is_vp9_available(_dimensions: VideoDimensions) -> bool {
    register_decoder();
    unsafe { videotoolbox::ffi::VTIsHardwareDecodeSupported(CODEC_TYPE_VP9) != 0 }
}

pub(crate) fn create_vp9(
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<Box<dyn VideoDecoder>> {
    if !is_vp9_available(configuration.coded_dimensions) {
        return Err(unsupported(
            "VideoToolbox does not advertise hardware VP9 decoding",
        ));
    }
    let format = create_format_description(configuration)?;
    let full_range = vp9_record(&configuration.configuration)?.video_full_range;
    let output = Arc::new(Mutex::new(Vec::new()));
    let session = create_session(&format, full_range, *limits, Arc::clone(&output))?;
    Ok(Box::new(Vp9Decoder {
        limits: *limits,
        full_range,
        format,
        output,
        session: Some(session),
        inspector: ChunkInspector::default(),
        slots: Default::default(),
        history: Vec::new(),
        output_wanted: true,
    }))
}

struct Vp9Decoder {
    limits: Limits,
    /// Whether the track declares full-range samples, which are then asked for in a full-range
    /// pixel buffer so that VideoToolbox does not rescale them.
    full_range: bool,
    format: CMFormatDescription,
    output: OutputQueue,
    session: Option<DecompressionSession>,
    inspector: ChunkInspector,
    /// The picture each reference slot holds, or `None` when it holds a frame that was never
    /// output.
    slots: [Option<Arc<RawPicture>>; 8],
    /// Every chunk decoded since the last key frame, for showing a hidden frame again.
    history: Vec<Vec<u8>>,
    output_wanted: bool,
}

impl Vp9Decoder {
    fn session(&self) -> Result<&DecompressionSession> {
        self.session
            .as_ref()
            .ok_or_else(|| codec("VideoToolbox decoder session is not initialized"))
    }

    /// Decodes a chunk and returns the one picture it outputs.
    fn decode(&mut self, sample: &EncodedVideoSample) -> Result<RawPicture> {
        let sample_buffer = create_sample_buffer(sample, &self.format)?;
        let session = self.session()?;
        let _exclusive = exclusive();
        session
            .decode_with_options(&sample_buffer, 0, None)
            .map_err(|error| codec(format!("VideoToolbox rejected VP9 input: {error}")))?;
        session
            .wait_for_async_frames()
            .map_err(|error| codec(format!("VideoToolbox did not finish VP9 input: {error}")))?;
        let outputs = std::mem::take(
            &mut *self
                .output
                .lock()
                .map_err(|_| codec("VideoToolbox output queue is unavailable"))?,
        );
        let mut pictures = outputs.into_iter().collect::<Result<Vec<_>>>()?;
        let picture = pictures
            .pop()
            .ok_or_else(|| codec("VideoToolbox output no picture for a VP9 sample"))?;
        if !pictures.is_empty() {
            return Err(codec(
                "VideoToolbox output several pictures for a VP9 sample",
            ));
        }
        Ok(picture)
    }

    /// Shows a hidden frame again by replaying the chunks since the last key frame through the
    /// software decoder, then `chunk`, the `show_existing_frame` that names it.
    fn replay(&self, chunk: &[u8]) -> Result<RawPicture> {
        let mut decoder = Decoder::new(self.limits);
        decoder.set_output_wanted(false);
        for earlier in &self.history {
            decoder.decode_chunk(earlier)?;
        }
        decoder.set_output_wanted(true);
        let picture = decoder
            .decode_chunk(chunk)?
            .ok_or_else(|| codec("the VP9 frame shown again was not decoded"))?;
        Ok(RawPicture {
            width: picture.width,
            height: picture.height,
            planes: picture.planes,
        })
    }
}

impl VideoDecoder for Vp9Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(limit("VP9 sample exceeds the allocation limit"));
        }
        if sample.data.is_empty() {
            return Err(malformed("VP9 sample is empty"));
        }
        // Every header is read before anything reaches VideoToolbox, so a sample the software
        // decoder would refuse is refused here too, with the same error.
        let frames = chunk_frames(&sample.data)?;
        let mut infos = Vec::with_capacity(frames.len());
        for frame in &frames {
            infos.push(self.inspector.inspect_frame(frame)?);
        }
        let last = *infos.last().expect("a chunk has a frame");
        if !last.shown {
            return Err(malformed("VP9 sample does not show a frame"));
        }
        let picture = match last.existing {
            Some(slot) => {
                if infos.len() > 1 {
                    return Err(unsupported(
                        "VideoToolbox VP9 decoding does not support a superframe that ends by \
                         showing an existing frame",
                    ));
                }
                match self.slots[slot].clone() {
                    Some(picture) => picture,
                    None => {
                        let picture = Arc::new(self.replay(&sample.data)?);
                        self.slots[slot] = Some(Arc::clone(&picture));
                        picture
                    }
                }
            }
            None => {
                let picture = Arc::new(self.decode(sample)?);
                if infos[0].key_frame {
                    self.history.clear();
                }
                self.history.push(sample.data.clone());
                // Only the chunk's last frame is output; the hidden ones before it are not.
                let count = infos.len();
                for (position, info) in infos.iter().enumerate() {
                    let held = (position + 1 == count).then(|| Arc::clone(&picture));
                    for (index, slot) in self.slots.iter_mut().enumerate() {
                        if info.refresh_frame_flags & (1 << index) != 0 {
                            *slot = held.clone();
                        }
                    }
                }
                picture
            }
        };
        check_cancelled(cancellation)?;
        if !self.output_wanted {
            return Ok(Vec::new());
        }
        let shape = last.shape;
        if picture.width < shape.width || picture.height < shape.height {
            return Err(codec(
                "VideoToolbox returned a VP9 picture smaller than the frame",
            ));
        }
        let picture = DecodedPicture {
            width: shape.width,
            height: shape.height,
            planes: crop(&picture, shape.width, shape.height),
            color_space: shape.color_space,
            full_range: shape.full_range,
        };
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame: crate::vp9_decoder::picture_to_rgba(&picture, &self.limits)?,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        // Every sample is waited for as it is submitted; nothing is held back.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        drop_session(self.session.take());
        self.output
            .lock()
            .map_err(|_| codec("VideoToolbox output queue is unavailable"))?
            .clear();
        self.inspector.reset();
        self.slots = Default::default();
        self.history.clear();
        self.session = Some(create_session(
            &self.format,
            self.full_range,
            self.limits,
            Arc::clone(&self.output),
        )?);
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        // Every picture is still read back, because a later `show_existing_frame` may show it
        // again; only the RGBA conversion is skipped.
        self.output_wanted = wanted;
    }
}

impl Drop for Vp9Decoder {
    fn drop(&mut self) {
        drop_session(self.session.take());
    }
}

/// Invalidates a session, in turn with the other sessions on a virtual Mac.
fn drop_session(session: Option<DecompressionSession>) {
    let _exclusive = exclusive();
    drop(session);
}

/// Crops a picture's planes to `width` x `height`.
fn crop(picture: &RawPicture, width: usize, height: usize) -> [Vec<u8>; 3] {
    let plane = |index: usize, plane_width: usize, plane_height: usize, stride: usize| {
        picture.planes[index]
            .chunks(stride)
            .take(plane_height)
            .flat_map(|row| &row[..plane_width])
            .copied()
            .collect::<Vec<u8>>()
    };
    let chroma_stride = picture.width.div_ceil(2);
    [
        plane(0, width, height, picture.width),
        plane(1, width.div_ceil(2), height.div_ceil(2), chroma_stride),
        plane(2, width.div_ceil(2), height.div_ceil(2), chroma_stride),
    ]
}

fn create_session(
    format: &CMFormatDescription,
    full_range: bool,
    limits: Limits,
    output: OutputQueue,
) -> Result<DecompressionSession> {
    let pixel_format = CFNumber::from_u64(u64::from(if full_range {
        raw::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    } else {
        raw::kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    }));
    let pixel_format_key = unsafe {
        CFType::from_raw_retained(raw::kCVPixelBufferPixelFormatTypeKey.cast_mut().cast())
    }
    .ok_or_else(|| codec("CoreVideo pixel-format key is unavailable"))?;
    let attributes = CFDictionary::from_pairs(&[(
        &pixel_format_key as &dyn AsCFType,
        &pixel_format as &dyn AsCFType,
    )]);
    let _exclusive = exclusive();
    let session = DecompressionSession::new_with_image_buffer_attributes(
        format,
        Some(&attributes),
        move |decoded| {
            let result = read_picture(decoded, &limits);
            if let Ok(mut queue) = output.lock() {
                queue.push(result);
            }
        },
    )
    .map_err(|error| {
        unsupported(format!(
            "could not create VideoToolbox VP9 decoder: {error}"
        ))
    })?;

    let using_hardware = unsafe {
        session.copy_property(
            videotoolbox::ffi::kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
        )
    }
    .map_err(|error| {
        codec(format!(
            "could not inspect VideoToolbox VP9 decoder: {error}"
        ))
    })?
    .is_some_and(|value| {
        value.as_ptr().cast_const() == unsafe { videotoolbox::ffi::kCFBooleanTrue }.cast()
    });
    if !using_hardware {
        return Err(unsupported(
            "VideoToolbox created a software decoder instead of a hardware decoder",
        ));
    }
    Ok(session)
}

/// The `vpcC` sample description extension VideoToolbox needs: the track's own box without its
/// size and type, or one describing a profile 0, 8-bit 4:2:0 stream when the track has none.
fn vpcc_atom(configuration: &[u8]) -> Result<Vec<u8>> {
    if configuration.len() >= 8 && &configuration[4..8] == b"vpcC" {
        return Ok(configuration[8..].to_vec());
    }
    let record = vp9_record(configuration)?;
    Ok(vec![
        1,
        0,
        0,
        0,
        record.profile,
        record.level,
        (record.bit_depth << 4) | (record.chroma_subsampling << 1),
        record.colour_primaries,
        record.transfer_characteristics,
        record.matrix_coefficients,
        0,
        0,
    ])
}

fn vp9_record(configuration: &[u8]) -> Result<Vp9CodecConfig> {
    Vp9CodecConfig::parse(configuration)
        .map_err(|error| malformed(format!("invalid VP9 configuration: {error}")))
}

fn create_format_description(configuration: &VideoDecoderConfig) -> Result<CMFormatDescription> {
    let atom_name = CFString::new("vpcC");
    let atom = CFData::from_bytes(vpcc_atom(&configuration.configuration)?);
    let atoms = CFDictionary::from_pairs(&[(&atom_name as &dyn AsCFType, &atom as &dyn AsCFType)]);
    let atoms_key = unsafe {
        CFType::from_raw_retained(
            raw::kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms
                .cast_mut()
                .cast(),
        )
    }
    .ok_or_else(|| codec("CoreMedia sample description extension key is unavailable"))?;
    let extensions =
        CFDictionary::from_pairs(&[(&atoms_key as &dyn AsCFType, &atoms as &dyn AsCFType)]);
    let dimensions = configuration.coded_dimensions;
    let mut format: raw::CMVideoFormatDescriptionRef = ptr::null_mut();
    let status = unsafe {
        raw::CMVideoFormatDescriptionCreate(
            raw::kCFAllocatorDefault,
            CODEC_TYPE_VP9,
            i32::try_from(dimensions.width).map_err(|_| limit("VP9 width overflows"))?,
            i32::try_from(dimensions.height).map_err(|_| limit("VP9 height overflows"))?,
            extensions.as_ptr().cast_const().cast(),
            &mut format,
        )
    };
    if status != 0 || format.is_null() {
        return Err(unsupported(format!(
            "VideoToolbox rejected the VP9 format description (OSStatus {status})"
        )));
    }
    CMFormatDescription::from_raw(format.cast_mut().cast())
        .ok_or_else(|| codec("VideoToolbox did not return a VP9 format description"))
}

fn create_sample_buffer(
    sample: &EncodedVideoSample,
    format: &CMFormatDescription,
) -> Result<CMSampleBuffer> {
    let presentation = i64::try_from(sample.presentation_index.0)
        .map_err(|_| limit("VP9 presentation index exceeds VideoToolbox timestamp range"))?;
    let block = CMBlockBuffer::create(&sample.data)
        .ok_or_else(|| codec("could not allocate a CoreMedia VP9 block buffer"))?;
    let valid_time = |value| raw::CMTime {
        value,
        timescale: 1,
        flags: raw::kCMTimeFlags_Valid,
        epoch: 0,
    };
    let timing = raw::CMSampleTimingInfo {
        duration: valid_time(1),
        presentationTimeStamp: valid_time(presentation),
        decodeTimeStamp: raw::CMTime {
            value: 0,
            timescale: 0,
            flags: 0,
            epoch: 0,
        },
    };
    let sample_size = sample.data.len();
    let mut sample_buffer: raw::CMSampleBufferRef = ptr::null_mut();
    let status = unsafe {
        raw::CMSampleBufferCreateReady(
            raw::kCFAllocatorDefault,
            block.as_ptr().cast(),
            format.as_ptr().cast(),
            1,
            1,
            &timing,
            1,
            &sample_size,
            &mut sample_buffer,
        )
    };
    if status != 0 || sample_buffer.is_null() {
        return Err(codec(format!(
            "could not create a CoreMedia VP9 sample (OSStatus {status})"
        )));
    }
    CMSampleBuffer::from_raw(sample_buffer.cast())
        .ok_or_else(|| codec("CoreMedia did not return a VP9 sample buffer"))
}

/// Copies a decoded NV12 picture out of its pixel buffer as three 4:2:0 planes.
fn read_picture(decoded: videotoolbox::DecodedFrame, limits: &Limits) -> Result<RawPicture> {
    if decoded.status != 0 {
        return Err(codec(format!(
            "VideoToolbox VP9 decode failed with OSStatus {}",
            decoded.status
        )));
    }
    let buffer = decoded
        .image_buffer
        .ok_or_else(|| codec("VideoToolbox completed VP9 decode without an image"))?;
    if buffer.pixel_format() != raw::kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        && buffer.pixel_format() != raw::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    {
        return Err(codec(format!(
            "VideoToolbox returned unexpected pixel format 0x{:08x}",
            buffer.pixel_format()
        )));
    }
    if buffer.plane_count() != 2 {
        return Err(codec("VideoToolbox NV12 output does not have two planes"));
    }
    let width = buffer.width();
    let height = buffer.height();
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let bytes = width
        .checked_mul(height)
        .and_then(|luma| luma.checked_add(2 * chroma_width * chroma_height))
        .ok_or_else(|| limit("VideoToolbox VP9 picture size overflows"))?;
    if bytes as u64 > limits.max_allocation_bytes {
        return Err(limit(
            "VideoToolbox VP9 picture exceeds the allocation limit",
        ));
    }
    let guard = buffer
        .lock_read_only()
        .map_err(|status| codec(format!("could not lock VideoToolbox output ({status})")))?;
    let mut luma = Vec::with_capacity(width * height);
    for y in 0..height {
        let row = guard
            .plane_row(0, y)
            .filter(|row| row.len() >= width)
            .ok_or_else(|| codec("VideoToolbox luma row is unavailable"))?;
        luma.extend_from_slice(&row[..width]);
    }
    let mut u = Vec::with_capacity(chroma_width * chroma_height);
    let mut v = Vec::with_capacity(chroma_width * chroma_height);
    for y in 0..chroma_height {
        let row = guard
            .plane_row(1, y)
            .filter(|row| row.len() >= chroma_width * 2)
            .ok_or_else(|| codec("VideoToolbox chroma row is unavailable"))?;
        for pair in row[..chroma_width * 2].chunks_exact(2) {
            u.push(pair[0]);
            v.push(pair[1]);
        }
    }
    Ok(RawPicture {
        width,
        height,
        planes: [luma, u, v],
    })
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(Error::new(
            ErrorKind::Cancelled,
            "codec operation cancelled",
        ))
    } else {
        Ok(())
    }
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}
