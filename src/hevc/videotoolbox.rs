//! macOS VideoToolbox HEVC backend.

use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use apple_cf::cf::{AsCFType, CFDictionary, CFNumber, CFType};
use apple_cf::cm::{CMBlockBuffer, CMFormatDescription, CMSampleBuffer};
use apple_cf::raw;
use videotoolbox::{Codec as VtCodec, DecompressionSession};

use super::engine::hvcc::HvccRecord;
use super::readback;
use crate::{
    CancellationToken, DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex, Limits,
    PixelFormat, Plane, Result, VideoDecoder, VideoDecoderConfig, VideoDimensions, VideoFrame,
};

type OutputQueue = Arc<Mutex<Vec<Result<DecodedVideoFrame>>>>;

/// Whether the pictures being decoded are wanted, shared with the decompression callback that
/// would otherwise convert each one to RGBA.
type OutputWanted = Arc<AtomicBool>;

pub(super) fn is_available(_dimensions: VideoDimensions) -> bool {
    DecompressionSession::is_hardware_decode_supported(VtCodec::HEVC)
}

pub(super) fn create(
    configuration: &VideoDecoderConfig,
    limits: &Limits,
    record: &HvccRecord,
) -> Result<Box<dyn VideoDecoder>> {
    if !is_available(configuration.coded_dimensions) {
        return Err(unsupported(
            "VideoToolbox does not advertise hardware HEVC decoding",
        ));
    }
    VideoToolboxDecoder::new(configuration.clone(), *limits, record)
        .map(|decoder| Box::new(decoder) as Box<dyn VideoDecoder>)
}

struct VideoToolboxDecoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
    format: CMFormatDescription,
    output: OutputQueue,
    output_wanted: OutputWanted,
    session: Option<DecompressionSession>,
}

impl VideoToolboxDecoder {
    fn new(configuration: VideoDecoderConfig, limits: Limits, record: &HvccRecord) -> Result<Self> {
        let format = create_format_description(record)?;
        let output = Arc::new(Mutex::new(Vec::new()));
        let output_wanted = Arc::new(AtomicBool::new(true));
        let session = create_session(
            &format,
            &configuration,
            limits,
            Arc::clone(&output),
            Arc::clone(&output_wanted),
        )?;
        Ok(Self {
            configuration,
            limits,
            format,
            output,
            output_wanted,
            session: Some(session),
        })
    }

    fn session(&self) -> Result<&DecompressionSession> {
        self.session
            .as_ref()
            .ok_or_else(|| codec("VideoToolbox decoder session is not initialized"))
    }

    fn take_output(&self) -> Result<Vec<DecodedVideoFrame>> {
        let mut queue = self
            .output
            .lock()
            .map_err(|_| codec("VideoToolbox output queue is unavailable"))?;
        std::mem::take(&mut *queue).into_iter().collect()
    }

    fn clear_output(&self) -> Result<()> {
        self.output
            .lock()
            .map_err(|_| codec("VideoToolbox output queue is unavailable"))?
            .clear();
        Ok(())
    }
}

impl VideoDecoder for VideoToolboxDecoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(limit("HEVC access unit exceeds the allocation limit"));
        }
        let sample_buffer = create_sample_buffer(sample, &self.format)?;
        // `kVTDecodeFrame_DoNotOutputFrame` is VideoToolbox's own name for this: the sample is
        // decoded, and stays a reference for the frames after it, but no picture comes back and
        // the callback has nothing to lock or convert.
        let flags = if self.output_wanted.load(Ordering::Acquire) {
            0
        } else {
            videotoolbox::ffi::kVTDecodeFrame_DoNotOutputFrame
        };
        self.session().and_then(|session| {
            session
                .decode_with_options(&sample_buffer, flags, None)
                .map_err(|error| codec(format!("VideoToolbox rejected HEVC input: {error}")))?;
            session
                .wait_for_async_frames()
                .map_err(|error| codec(format!("VideoToolbox did not finish HEVC input: {error}")))
        })?;
        check_cancelled(cancellation)?;
        self.take_output()
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        self.session().and_then(|session| {
            session.finish_delayed_frames().map_err(|error| {
                codec(format!("VideoToolbox could not drain HEVC output: {error}"))
            })?;
            session.wait_for_async_frames().map_err(|error| {
                codec(format!(
                    "VideoToolbox did not finish draining HEVC output: {error}"
                ))
            })
        })?;
        check_cancelled(cancellation)?;
        self.take_output()
    }

    fn reset(&mut self) -> Result<()> {
        self.session.take();
        self.clear_output()?;
        self.session = Some(create_session(
            &self.format,
            &self.configuration,
            self.limits,
            Arc::clone(&self.output),
            Arc::clone(&self.output_wanted),
        )?);
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted.store(wanted, Ordering::Release);
    }
}

fn create_session(
    format: &CMFormatDescription,
    configuration: &VideoDecoderConfig,
    limits: Limits,
    output: OutputQueue,
    output_wanted: OutputWanted,
) -> Result<DecompressionSession> {
    let pixel_format = CFNumber::from_u64(u64::from(
        raw::kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    ));
    let pixel_format_key = unsafe {
        CFType::from_raw_retained(raw::kCVPixelBufferPixelFormatTypeKey.cast_mut().cast())
    }
    .ok_or_else(|| codec("CoreVideo pixel-format key is unavailable"))?;
    let attributes = CFDictionary::from_pairs(&[(
        &pixel_format_key as &dyn AsCFType,
        &pixel_format as &dyn AsCFType,
    )]);
    let callback_configuration = configuration.clone();
    let session = DecompressionSession::new_with_image_buffer_attributes(
        format,
        Some(&attributes),
        move |decoded| {
            // A frame decoded with `kVTDecodeFrame_DoNotOutputFrame` still reaches the callback,
            // with no image buffer to convert. Dropping it here is the whole saving.
            if !output_wanted.load(Ordering::Acquire) {
                return;
            }
            let result = decoded_frame(decoded, &callback_configuration, &limits);
            if let Ok(mut queue) = output.lock() {
                queue.push(result);
            }
        },
    )
    .map_err(|error| {
        unsupported(format!(
            "could not create VideoToolbox HEVC decoder: {error}"
        ))
    })?;

    let using_hardware = unsafe {
        session.copy_property(
            videotoolbox::ffi::kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
        )
    }
    .map_err(|error| {
        codec(format!(
            "could not inspect VideoToolbox HEVC decoder: {error}"
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

fn create_format_description(record: &HvccRecord) -> Result<CMFormatDescription> {
    let parameter_sets: Vec<Vec<u8>> = record
        .nal_units
        .iter()
        .filter(|unit| matches!(unit.header.nal_unit_type, 32..=34))
        .map(coded_nal)
        .collect();
    if !(32..=34).all(|kind| {
        record
            .nal_units
            .iter()
            .any(|unit| unit.header.nal_unit_type == kind)
    }) {
        return Err(codec(
            "VideoToolbox HEVC configuration requires VPS, SPS, and PPS parameter sets",
        ));
    }
    let pointers: Vec<*const u8> = parameter_sets.iter().map(|set| set.as_ptr()).collect();
    let sizes: Vec<usize> = parameter_sets.iter().map(Vec::len).collect();
    let mut format: raw::CMFormatDescriptionRef = ptr::null_mut();
    let status = unsafe {
        raw::CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            raw::kCFAllocatorDefault,
            parameter_sets.len(),
            pointers.as_ptr(),
            sizes.as_ptr(),
            i32::try_from(record.length_size)
                .map_err(|_| codec("HEVC NAL length size overflows"))?,
            ptr::null(),
            &mut format,
        )
    };
    if status != 0 || format.is_null() {
        return Err(unsupported(format!(
            "VideoToolbox rejected the HEVC format description (OSStatus {status})"
        )));
    }
    CMFormatDescription::from_raw(format.cast_mut().cast())
        .ok_or_else(|| codec("VideoToolbox did not return an HEVC format description"))
}

fn create_sample_buffer(
    sample: &EncodedVideoSample,
    format: &CMFormatDescription,
) -> Result<CMSampleBuffer> {
    let presentation = i64::try_from(sample.presentation_index.0)
        .map_err(|_| limit("HEVC presentation index exceeds VideoToolbox timestamp range"))?;
    let block = CMBlockBuffer::create(&sample.data)
        .ok_or_else(|| codec("could not allocate a CoreMedia HEVC block buffer"))?;
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
            "could not create a CoreMedia HEVC sample (OSStatus {status})"
        )));
    }
    CMSampleBuffer::from_raw(sample_buffer.cast())
        .ok_or_else(|| codec("CoreMedia did not return an HEVC sample buffer"))
}

fn decoded_frame(
    decoded: videotoolbox::DecodedFrame,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<DecodedVideoFrame> {
    if decoded.status != 0 {
        return Err(codec(format!(
            "VideoToolbox HEVC decode failed with OSStatus {}",
            decoded.status
        )));
    }
    let buffer = decoded
        .image_buffer
        .ok_or_else(|| codec("VideoToolbox completed HEVC decode without an image"))?;
    let (presentation, timescale) = decoded.presentation_time;
    if presentation < 0 || timescale != 1 {
        return Err(codec("VideoToolbox returned an unexpected HEVC timestamp"));
    }
    if buffer.pixel_format() != raw::kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
        return Err(codec(format!(
            "VideoToolbox returned unexpected pixel format 0x{:08x}",
            buffer.pixel_format()
        )));
    }
    let width = configuration.coded_dimensions.width as usize;
    let height = configuration.coded_dimensions.height as usize;
    if buffer.width() != width || buffer.height() != height {
        return Err(codec("VideoToolbox returned unexpected HEVC dimensions"));
    }
    if buffer.plane_count() != 2 {
        return Err(codec("VideoToolbox NV12 output does not have two planes"));
    }
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| limit("VideoToolbox RGBA stride overflows"))?;
    let length = stride
        .checked_mul(height)
        .ok_or_else(|| limit("VideoToolbox RGBA allocation overflows"))?;
    if length as u64 > limits.max_allocation_bytes {
        return Err(limit(
            "VideoToolbox RGBA output exceeds the allocation limit",
        ));
    }
    // The surface-copy phase for VideoToolbox is the base-address lock: the
    // decoded `CVPixelBuffer` is already in memory the CPU can address on a
    // unified-memory host, so this is a lock rather than a transfer, and the
    // seam is what says so rather than assuming it.
    let surface_copy = readback::Timer::start();
    let guard = buffer
        .lock_read_only()
        .map_err(|status| codec(format!("could not lock VideoToolbox output ({status})")))?;
    surface_copy.record(readback::Phase::SurfaceCopy);
    let color_convert = readback::Timer::start();
    let mut rgba = vec![0_u8; length];
    for y in 0..height {
        let luma = guard
            .plane_row(0, y)
            .ok_or_else(|| codec("VideoToolbox luma row is unavailable"))?;
        let chroma = guard
            .plane_row(1, y / 2)
            .ok_or_else(|| codec("VideoToolbox chroma row is unavailable"))?;
        if luma.len() < width || chroma.len() < width {
            return Err(codec("VideoToolbox NV12 row is shorter than expected"));
        }
        for x in 0..width {
            let y_scaled = i32::from(luma[x]) * 8 - 128;
            let u_scaled = i32::from(chroma[x & !1]) * 8 - 1024;
            let v_scaled = i32::from(chroma[(x & !1) + 1]) * 8 - 1024;
            let y_term = multiply_high(y_scaled, 9_539);
            let target = y * stride + x * 4;
            rgba[target] = clip_u8(y_term.saturating_add(multiply_high(v_scaled, 13_075)));
            rgba[target + 1] = clip_u8(
                y_term
                    .saturating_add(multiply_high(u_scaled, -3_209))
                    .saturating_add(multiply_high(v_scaled, -6_660)),
            );
            rgba[target + 2] = clip_u8(y_term.saturating_add(multiply_high(u_scaled, 16_525)));
            rgba[target + 3] = 255;
        }
    }
    color_convert.record(readback::Phase::ColorConvert);
    readback::count_frame();
    Ok(DecodedVideoFrame {
        presentation_index: FrameIndex(presentation as u64),
        frame: VideoFrame::new(
            configuration.coded_dimensions,
            PixelFormat::Rgba8,
            configuration.color_range,
            vec![Plane { data: rgba, stride }],
            limits,
        )?,
    })
}

fn multiply_high(left: i32, right: i32) -> i32 {
    left.saturating_mul(right) >> 16
}

fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn coded_nal(unit: &super::engine::nal::NalUnit) -> Vec<u8> {
    let mut output = Vec::with_capacity(unit.escaped.len() + 2);
    output.push((unit.header.nal_unit_type << 1) | (unit.header.nuh_layer_id >> 5));
    output.push(((unit.header.nuh_layer_id & 0x1f) << 3) | (unit.header.temporal_id + 1));
    output.extend_from_slice(&unit.escaped);
    output
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

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}
