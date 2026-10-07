//! macOS VideoToolbox hardware HEVC Main and VP9 profile 0 encoder.
//!
//! A `VTCompressionSession` that is required to be hardware-backed, in real-time mode, with frame
//! reordering off so decode order is presentation order and every sample's DTS equals its PTS.
//! Frames are copied into BGRA pixel buffers from the session's own pool - a straight row copy for
//! `Bgra8` input and a byte swizzle for `Rgba8` - and VideoToolbox does the conversion to YCbCr on
//! the media engine, so no CPU colour conversion runs.
//!
//! # The `hvcC` has to exist before the first frame
//!
//! [`VideoEncoder::config`] is read before any frame is written, and VideoToolbox only reveals the
//! parameter sets it chose once it has encoded something. So creation encodes one black priming
//! frame, one frame duration *before* the stream starts, and builds the `hvcC` from the parameter
//! sets that frame came out with; its sample is discarded. Every caller frame is then submitted
//! one frame duration later than its own timestamp, so the priming frame never collides with
//! frame zero, and frame zero is forced to be a keyframe so the stream still opens on one. Any
//! later sample that references a parameter set the `hvcC` does not declare is an error rather
//! than a stream that would not decode.
//!
//! A VP9 stream's `vpcC` is built the same way, from the colour space and range the priming key
//! frame signals, and every later key frame has to signal the same. VP9 is rate controlled by
//! quality rather than to a bitrate, and its samples are already one chunk each.
//!
//! # Dropped frames
//!
//! Real-time mode lets VideoToolbox drop a frame it cannot keep up with. An MP4 track's decode
//! timestamps must be contiguous, so a sample is held back until the next one arrives and its
//! duration is stretched over any gap, and the last one is stretched to the end of the input at
//! [`VideoEncoder::finish`]. The stream's length is always exactly what was submitted.
//!
//! # Cancellation
//!
//! `finish` completes every submitted frame. Dropping the encoder without finishing it - which is
//! what an abandoned [`crate::MediaOutput`] or a dropped encode future amounts to - cancels
//! instead: the session is invalidated at once, and VideoToolbox discards what it had not yet
//! emitted rather than being waited on.

use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};

use super::annexb::{ParameterSets, reframe_length_prefixed};
use crate::{
    Codec, CodecImplementation, ColorRange, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, FrameSource, Limits, Orientation, PixelFormat, Result, SampleDependency,
    VideoDimensions, VideoEncoder, VideoEncoderConfig, VideoEncoderFormat, VideoFrame,
};

type OSStatus = i32;
type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFArrayRef = *const c_void;
type CFBooleanRef = *const c_void;
type CMSampleBufferRef = *mut c_void;
type CMBlockBufferRef = *mut c_void;
type CMFormatDescriptionRef = *mut c_void;
type CVPixelBufferRef = *mut c_void;
type CVPixelBufferPoolRef = *mut c_void;
type VTCompressionSessionRef = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

const CM_TIME_VALID: u32 = 1;
const CM_TIME_INVALID: CMTime = CMTime {
    value: 0,
    timescale: 0,
    flags: 0,
    epoch: 0,
};
/// `kCMVideoCodecType_HEVC`.
const HEVC: u32 = u32::from_be_bytes(*b"hvc1");
/// `kCMVideoCodecType_VP9`.
const VP9: u32 = u32::from_be_bytes(*b"vp09");
/// `kCVPixelFormatType_32BGRA`.
const BGRA: u32 = u32::from_be_bytes(*b"BGRA");
/// `kCFNumberSInt32Type`.
const CF_NUMBER_SINT32: isize = 3;
/// `kCFNumberFloat64Type`.
const CF_NUMBER_FLOAT64: isize = 6;
/// `kVTEncodeInfo_FrameDropped`.
const FRAME_DROPPED: u32 = 1 << 1;

type OutputCallback = unsafe extern "C" fn(
    refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    flags: u32,
    sample: CMSampleBufferRef,
);

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    // Only their addresses are used.
    static kCFTypeDictionaryKeyCallBacks: u8;
    static kCFTypeDictionaryValueCallBacks: u8;
    static kCFBooleanTrue: CFBooleanRef;
    static kCFBooleanFalse: CFBooleanRef;
    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFMutableDictionaryRef;
    fn CFDictionarySetValue(
        dictionary: CFMutableDictionaryRef,
        key: *const c_void,
        value: *const c_void,
    );
    fn CFDictionaryGetValue(dictionary: CFDictionaryRef, key: *const c_void) -> *const c_void;
    fn CFNumberCreate(allocator: *const c_void, kind: isize, value: *const c_void) -> CFTypeRef;
    fn CFArrayGetCount(array: CFArrayRef) -> isize;
    fn CFArrayGetValueAtIndex(array: CFArrayRef, index: isize) -> *const c_void;
    fn CFBooleanGetValue(boolean: CFBooleanRef) -> u8;
    fn CFRelease(value: CFTypeRef);
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    static kCVPixelBufferWidthKey: CFStringRef;
    static kCVPixelBufferHeightKey: CFStringRef;
    static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
    static kCVImageBufferYCbCrMatrix_ITU_R_601_4: CFStringRef;
    fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: *const c_void,
        pool: CVPixelBufferPoolRef,
        out: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(buffer: CVPixelBufferRef) -> *mut u8;
    fn CVPixelBufferGetBytesPerRow(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidth(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(buffer: CVPixelBufferRef) -> usize;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    static kCMSampleAttachmentKey_NotSync: CFStringRef;
    fn CMSampleBufferGetDataBuffer(sample: CMSampleBufferRef) -> CMBlockBufferRef;
    fn CMSampleBufferGetFormatDescription(sample: CMSampleBufferRef) -> CMFormatDescriptionRef;
    fn CMSampleBufferGetPresentationTimeStamp(sample: CMSampleBufferRef) -> CMTime;
    fn CMSampleBufferGetSampleAttachmentsArray(sample: CMSampleBufferRef, create: u8)
    -> CFArrayRef;
    fn CMBlockBufferGetDataLength(buffer: CMBlockBufferRef) -> usize;
    fn CMBlockBufferCopyDataBytes(
        buffer: CMBlockBufferRef,
        offset: usize,
        length: usize,
        destination: *mut c_void,
    ) -> OSStatus;
    fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        description: CMFormatDescriptionRef,
        index: usize,
        set_out: *mut *const u8,
        size_out: *mut usize,
        count_out: *mut usize,
        header_length_out: *mut i32,
    ) -> OSStatus;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    static kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: CFStringRef;
    static kVTCompressionPropertyKey_RealTime: CFStringRef;
    static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
    static kVTProfileLevel_HEVC_Main_AutoLevel: CFStringRef;
    static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
    static kVTCompressionPropertyKey_Quality: CFStringRef;
    static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
    static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
    static kVTCompressionPropertyKey_YCbCrMatrix: CFStringRef;
    static kVTCompressionPropertyKey_PixelTransferProperties: CFStringRef;
    static kVTPixelTransferPropertyKey_DestinationYCbCrMatrix: CFStringRef;
    static kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder: CFStringRef;
    static kVTEncodeFrameOptionKey_ForceKeyFrame: CFStringRef;
    fn VTCopySupportedPropertyDictionaryForEncoder(
        width: i32,
        height: i32,
        codec: u32,
        encoder_specification: CFDictionaryRef,
        encoder_id_out: *mut CFStringRef,
        supported_properties_out: *mut CFDictionaryRef,
    ) -> OSStatus;
    fn VTCompressionSessionCreate(
        allocator: *const c_void,
        width: i32,
        height: i32,
        codec: u32,
        encoder_specification: CFDictionaryRef,
        source_attributes: CFDictionaryRef,
        compressed_allocator: *const c_void,
        callback: OutputCallback,
        refcon: *mut c_void,
        out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    fn VTSessionSetProperty(
        session: VTCompressionSessionRef,
        key: CFStringRef,
        value: CFTypeRef,
    ) -> OSStatus;
    fn VTSessionCopyProperty(
        session: VTCompressionSessionRef,
        key: CFStringRef,
        allocator: *const c_void,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;
    fn VTCompressionSessionGetPixelBufferPool(
        session: VTCompressionSessionRef,
    ) -> CVPixelBufferPoolRef;
    fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image: CVPixelBufferRef,
        pts: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        frame_refcon: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        until: CMTime,
    ) -> OSStatus;
    fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);
}

/// What the HEVC or VP9 factory resolved from a configuration for the hardware path.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct Settings {
    /// The target bitrate, or for a stream rate controlled by quality a nominal one.
    pub bits_per_second: u32,
    /// `kVTCompressionPropertyKey_Quality` in `0.0..=1.0`, for a stream rate controlled by
    /// quality rather than to [`Self::bits_per_second`].
    pub quality: Option<f64>,
    pub keyframe_interval: u32,
}

/// The `CMVideoCodecType` VideoToolbox encodes `codec` as.
fn codec_type(codec: Codec) -> Option<u32> {
    match codec {
        Codec::Hevc => Some(HEVC),
        Codec::Vp9 => Some(VP9),
        _ => None,
    }
}

fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::Vp9 => "VP9",
        _ => "HEVC",
    }
}

/// Whether VideoToolbox has a hardware `codec` encoder for pictures of this size.
///
/// Asks for the encoder's supported properties under the same "hardware required" specification
/// the session is created with, which answers without creating a session.
#[doc(hidden)]
pub fn is_available(codec: Codec, dimensions: VideoDimensions) -> bool {
    let Some(codec) = codec_type(codec) else {
        return false;
    };
    let (Ok(width), Ok(height)) = (
        i32::try_from(dimensions.width),
        i32::try_from(dimensions.height),
    ) else {
        return false;
    };
    // SAFETY: the specification outlives the call, and both outputs are released when set.
    unsafe {
        let specification = hardware_specification();
        let mut encoder_id: CFStringRef = ptr::null();
        let mut properties: CFDictionaryRef = ptr::null();
        let status = VTCopySupportedPropertyDictionaryForEncoder(
            width,
            height,
            codec,
            specification.0,
            &mut encoder_id,
            &mut properties,
        );
        drop(Owned(encoder_id));
        drop(Owned(properties));
        status == 0
    }
}

#[doc(hidden)]
pub fn create(
    configuration: &VideoEncoderConfig,
    limits: &Limits,
    settings: Settings,
) -> Result<Box<dyn VideoEncoder>> {
    VideoToolboxEncoder::new(configuration.clone(), *limits, settings)
        .map(|encoder| Box::new(encoder) as Box<dyn VideoEncoder>)
}

/// One encoded picture as the output callback saw it.
struct Picture {
    /// The sample as VideoToolbox emitted it: an HEVC access unit, four-byte length-prefixed, or
    /// a VP9 chunk.
    data: Vec<u8>,
    pts: CMTime,
    is_sync: bool,
    /// The parameter sets the sample's format description declares. Empty for VP9.
    parameter_sets: ParameterSets,
}

/// What the output callback hands back, in submission order.
enum Output {
    Picture(Picture),
    Dropped,
    Failed(String),
}

/// What the output callback shares with the encoder.
struct Shared {
    /// Whether the session encodes HEVC, whose format descriptions carry parameter sets.
    hevc: bool,
    outputs: Mutex<Vec<Output>>,
}

/// What the track's decoder configuration declares, which every sample is held to.
enum Declared {
    /// The parameter sets the `hvcC` carries.
    Hevc(ParameterSets),
    /// The `vpcC`, and the level it names, as the priming key frame signalled them.
    Vp9 { vpcc: Vec<u8>, level: u8 },
}

struct VideoToolboxEncoder {
    configuration: VideoEncoderConfig,
    limits: Limits,
    session: VTCompressionSessionRef,
    /// The callback's queue. The session holds a strong reference through its refcon, released
    /// in `Drop` once the session can no longer call back.
    shared: Arc<Shared>,
    config: EncoderConfig,
    /// What the decoder configuration in [`Self::config`] declares.
    declared: Declared,
    next_index: u64,
    /// The last sample out of the session, held until the next one says how long it lasts.
    pending: Option<EncodedSample>,
    /// Whether any sample has been returned to the caller yet.
    emitted: bool,
    finished: bool,
}

impl VideoToolboxEncoder {
    fn new(configuration: VideoEncoderConfig, limits: Limits, settings: Settings) -> Result<Self> {
        let dimensions = configuration.coded_dimensions;
        let width = i32::try_from(dimensions.width)
            .map_err(|_| limit("frame width exceeds VideoToolbox's range"))?;
        let height = i32::try_from(dimensions.height)
            .map_err(|_| limit("frame height exceeds VideoToolbox's range"))?;
        let codec_type = codec_type(configuration.codec)
            .ok_or_else(|| unsupported("VideoToolbox encodes HEVC or VP9 only"))?;
        let name = codec_name(configuration.codec);
        let shared = Arc::new(Shared {
            hevc: configuration.codec == Codec::Hevc,
            outputs: Mutex::default(),
        });
        let refcon = Arc::into_raw(Arc::clone(&shared))
            .cast_mut()
            .cast::<c_void>();
        let mut session: VTCompressionSessionRef = ptr::null_mut();
        // SAFETY: every object passed is valid for the call. The refcon is a strong reference to
        // `shared` that the encoder releases only after invalidating the session.
        let status = unsafe {
            let specification = hardware_specification();
            let format = number_i32(BGRA as i32);
            let pixel_width = number_i32(width);
            let pixel_height = number_i32(height);
            let surface = dictionary(&[]);
            let attributes = dictionary(&[
                (kCVPixelBufferPixelFormatTypeKey, format.0),
                (kCVPixelBufferWidthKey, pixel_width.0),
                (kCVPixelBufferHeightKey, pixel_height.0),
                (kCVPixelBufferIOSurfacePropertiesKey, surface.0),
            ]);
            VTCompressionSessionCreate(
                ptr::null(),
                width,
                height,
                codec_type,
                specification.0,
                attributes.0,
                ptr::null(),
                on_output,
                refcon,
                &mut session,
            )
        };
        if status != 0 || session.is_null() {
            // SAFETY: the session was never created, so nothing else holds this reference.
            drop(unsafe { Arc::from_raw(refcon.cast_const().cast::<Shared>()) });
            return Err(unsupported(format!(
                "could not create a hardware VideoToolbox {name} encoder (OSStatus {status})"
            )));
        }
        // From here on `Drop` owns the session and the refcon.
        let mut encoder = Self {
            config: EncoderConfig {
                codec: configuration.codec,
                timescale: 0,
                decoder_config: Vec::new(),
            },
            configuration,
            limits,
            session,
            shared,
            declared: Declared::Hevc(ParameterSets::default()),
            next_index: 0,
            pending: None,
            emitted: false,
            finished: false,
        };
        encoder.configure(settings)?;
        encoder.prime()?;
        Ok(encoder)
    }

    fn name(&self) -> &'static str {
        codec_name(self.configuration.codec)
    }

    fn configure(&mut self, settings: Settings) -> Result<()> {
        let label = self.name();
        let bits_per_second = i32::try_from(settings.bits_per_second)
            .map_err(|_| invalid_input("target bitrate exceeds VideoToolbox's range"))?;
        let keyframe_interval = i32::try_from(settings.keyframe_interval)
            .map_err(|_| invalid_input("keyframe interval exceeds VideoToolbox's range"))?;
        let frame_rate =
            f64::from(self.configuration.timescale) / f64::from(self.configuration.frame_duration);
        // SAFETY: the session is valid and every value is a live CF object for the call.
        unsafe {
            let average = number_i32(bits_per_second);
            let interval = number_i32(keyframe_interval);
            let rate = number_f64(frame_rate);
            let quality = settings.quality.map(number_f64);
            // The BGRA-to-YCbCr conversion VideoToolbox runs on the way in takes its matrix from
            // here, not from `YCbCrMatrix`, and defaults to BT.709. The crate's decoders convert
            // back with BT.601, so anything else shifts colour on a round trip.
            let transfer = dictionary(&[(
                kVTPixelTransferPropertyKey_DestinationYCbCrMatrix,
                kCVImageBufferYCbCrMatrix_ITU_R_601_4,
            )]);
            let mut required = vec![
                (
                    kVTCompressionPropertyKey_AllowFrameReordering,
                    kCFBooleanFalse,
                    "frame reordering",
                ),
                (
                    kVTCompressionPropertyKey_RealTime,
                    kCFBooleanTrue,
                    "real-time mode",
                ),
                (
                    kVTCompressionPropertyKey_MaxKeyFrameInterval,
                    interval.0,
                    "keyframe interval",
                ),
                (
                    kVTCompressionPropertyKey_PixelTransferProperties,
                    transfer.0,
                    "BT.601 colour conversion",
                ),
            ];
            // Hints the encoder may decline. The matrix signalled in the stream matches the one
            // the conversion above used.
            let mut hints = vec![
                (kVTCompressionPropertyKey_ExpectedFrameRate, rate.0),
                (
                    kVTCompressionPropertyKey_YCbCrMatrix,
                    kCVImageBufferYCbCrMatrix_ITU_R_601_4,
                ),
            ];
            if self.configuration.codec == Codec::Hevc {
                required.push((
                    kVTCompressionPropertyKey_ProfileLevel,
                    kVTProfileLevel_HEVC_Main_AutoLevel,
                    "profile",
                ));
            }
            match &quality {
                // A stream rate controlled by quality still declares its nominal bitrate, but
                // only as a hint.
                Some(quality) => hints.extend([
                    (kVTCompressionPropertyKey_Quality, quality.0),
                    (kVTCompressionPropertyKey_AverageBitRate, average.0),
                ]),
                None => required.push((
                    kVTCompressionPropertyKey_AverageBitRate,
                    average.0,
                    "target bitrate",
                )),
            }
            for (key, value, name) in required {
                let status = VTSessionSetProperty(self.session, key, value);
                if status != 0 {
                    return Err(unsupported(format!(
                        "VideoToolbox rejected the {label} {name} (OSStatus {status})"
                    )));
                }
            }
            for (key, value) in hints {
                VTSessionSetProperty(self.session, key, value);
            }
            let status = VTCompressionSessionPrepareToEncodeFrames(self.session);
            if status != 0 {
                return Err(unsupported(format!(
                    "VideoToolbox could not prepare the {label} encoder (OSStatus {status})"
                )));
            }
            let mut using: CFTypeRef = ptr::null();
            let status = VTSessionCopyProperty(
                self.session,
                kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                ptr::null(),
                &mut using,
            );
            let using = Owned(using);
            if status != 0 || using.0.is_null() || CFBooleanGetValue(using.0) == 0 {
                return Err(unsupported(format!(
                    "VideoToolbox created a software {label} encoder instead of a hardware one"
                )));
            }
        }
        Ok(())
    }

    /// Encodes the black priming frame and declares the parameter sets, or for VP9 the colour
    /// signalling, it came out with.
    fn prime(&mut self) -> Result<()> {
        let dimensions = self.configuration.coded_dimensions;
        let black = VideoFrame::new(
            dimensions,
            PixelFormat::Bgra8,
            ColorRange::Limited,
            vec![crate::Plane {
                data: [0, 0, 0, 255].repeat(dimensions.width as usize * dimensions.height as usize),
                stride: dimensions.width as usize * 4,
            }],
            &self.limits,
        )?;
        let buffer = self.pixel_buffer(&black, Orientation::TopLeft)?;
        self.submit(&buffer, 0, true)?;
        self.complete()?;
        if self.configuration.codec == Codec::Vp9 {
            return self.prime_vp9();
        }
        let mut declared = ParameterSets::default();
        for output in self.take_outputs() {
            match output {
                Output::Picture(picture) => {
                    reframe_length_prefixed(&picture.data, &mut declared)
                        .ok_or_else(|| codec("VideoToolbox primed with no HEVC picture"))?;
                    for set in [
                        &picture.parameter_sets.vps,
                        &picture.parameter_sets.sps,
                        &picture.parameter_sets.pps,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        declared.collect(set);
                    }
                }
                Output::Dropped => {}
                Output::Failed(error) => return Err(codec(error)),
            }
        }
        self.config = EncoderConfig {
            codec: Codec::Hevc,
            timescale: self.configuration.timescale,
            decoder_config: declared.hvcc().ok_or_else(|| {
                codec("VideoToolbox did not produce a complete set of HEVC parameter sets")
            })?,
        };
        self.declared = Declared::Hevc(declared);
        Ok(())
    }

    /// Declares the `vpcC` the priming key frame signals.
    fn prime_vp9(&mut self) -> Result<()> {
        let level = zvidlib_vp9_syntax::pick_level(
            self.configuration.coded_dimensions,
            self.configuration.timescale,
            self.configuration.frame_duration,
        )
        .ok_or_else(|| invalid_input("VP9 dimensions and frame rate exceed level 6.2 limits"))?;
        let mut vpcc = None;
        for output in self.take_outputs() {
            match output {
                Output::Picture(picture) if vpcc.is_none() => {
                    vpcc = Some(
                        zvidlib_vp9_syntax::key_frame_vpcc(&picture.data, level).ok_or_else(
                            || codec("VideoToolbox primed with no VP9 profile 0 key frame"),
                        )?,
                    );
                }
                Output::Picture(_) | Output::Dropped => {}
                Output::Failed(error) => return Err(codec(error)),
            }
        }
        let vpcc = vpcc.ok_or_else(|| codec("VideoToolbox dropped its VP9 priming frame"))?;
        self.config = EncoderConfig {
            codec: Codec::Vp9,
            timescale: self.configuration.timescale,
            decoder_config: vpcc.clone(),
        };
        self.declared = Declared::Vp9 { vpcc, level };
        Ok(())
    }

    /// A pixel buffer from the session's pool holding `frame` as BGRA, top row first.
    fn pixel_buffer(&self, frame: &VideoFrame, orientation: Orientation) -> Result<Owned> {
        let width = frame.dimensions.width as usize;
        let height = frame.dimensions.height as usize;
        let plane = frame
            .planes
            .first()
            .ok_or_else(|| invalid_input("input frame has no pixel plane"))?;
        let row_bytes = width * 4;
        if plane.stride < row_bytes || plane.data.len() < plane.stride * (height - 1) + row_bytes {
            return Err(invalid_input("input frame plane is shorter than its size"));
        }
        let swizzle = match frame.pixel_format {
            PixelFormat::Bgra8 => false,
            PixelFormat::Rgba8 => true,
            _ => {
                return Err(invalid_input(
                    "VideoToolbox encoding accepts Rgba8 or Bgra8",
                ));
            }
        };
        // SAFETY: the buffer is locked while it is written, each row write stays inside the row
        // the buffer reports, and the buffer's size is checked against the frame's first.
        unsafe {
            let pool = VTCompressionSessionGetPixelBufferPool(self.session);
            if pool.is_null() {
                return Err(codec("the VideoToolbox encoder has no pixel buffer pool"));
            }
            let mut buffer: CVPixelBufferRef = ptr::null_mut();
            let status = CVPixelBufferPoolCreatePixelBuffer(ptr::null(), pool, &mut buffer);
            if status != 0 || buffer.is_null() {
                return Err(codec(format!(
                    "could not allocate a VideoToolbox pixel buffer (CVReturn {status})"
                )));
            }
            let owned = Owned(buffer.cast_const());
            if CVPixelBufferGetWidth(buffer) != width || CVPixelBufferGetHeight(buffer) != height {
                return Err(codec("the VideoToolbox pixel buffer has the wrong size"));
            }
            if CVPixelBufferLockBaseAddress(buffer, 0) != 0 {
                return Err(codec("could not lock the VideoToolbox pixel buffer"));
            }
            let base = CVPixelBufferGetBaseAddress(buffer);
            let stride = CVPixelBufferGetBytesPerRow(buffer);
            if base.is_null() || stride < row_bytes {
                CVPixelBufferUnlockBaseAddress(buffer, 0);
                return Err(codec("the VideoToolbox pixel buffer is not addressable"));
            }
            for row in 0..height {
                let source_row = match orientation {
                    Orientation::TopLeft => row,
                    Orientation::BottomLeft => height - 1 - row,
                };
                let source = &plane.data[source_row * plane.stride..][..row_bytes];
                let target = std::slice::from_raw_parts_mut(base.add(row * stride), row_bytes);
                if swizzle {
                    for (to, from) in target.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
                        to.copy_from_slice(&[from[2], from[1], from[0], from[3]]);
                    }
                } else {
                    target.copy_from_slice(source);
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, 0);
            Ok(owned)
        }
    }

    /// Hands one pixel buffer to the session at `tick`, in the configured timescale.
    fn submit(&self, buffer: &Owned, tick: i64, key_frame: bool) -> Result<()> {
        let timescale = i32::try_from(self.configuration.timescale)
            .map_err(|_| limit("timescale exceeds VideoToolbox's range"))?;
        let pts = CMTime {
            value: tick,
            timescale,
            flags: CM_TIME_VALID,
            epoch: 0,
        };
        let duration = CMTime {
            value: i64::from(self.configuration.frame_duration),
            ..pts
        };
        // SAFETY: the session and buffer are valid; the session retains the buffer for as long
        // as it needs it, and the frame options only for the call.
        let status = unsafe {
            let options = key_frame
                .then(|| dictionary(&[(kVTEncodeFrameOptionKey_ForceKeyFrame, kCFBooleanTrue)]));
            VTCompressionSessionEncodeFrame(
                self.session,
                buffer.0.cast_mut(),
                pts,
                duration,
                options.as_ref().map_or(ptr::null(), |options| options.0),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(codec(format!(
                "VideoToolbox rejected a frame (OSStatus {status})"
            )));
        }
        Ok(())
    }

    /// Blocks until every submitted frame has been through the output callback.
    fn complete(&self) -> Result<()> {
        // SAFETY: the session is valid.
        let status = unsafe { VTCompressionSessionCompleteFrames(self.session, CM_TIME_INVALID) };
        if status != 0 {
            return Err(codec(format!(
                "VideoToolbox could not finish frames (OSStatus {status})"
            )));
        }
        Ok(())
    }

    fn take_outputs(&self) -> Vec<Output> {
        std::mem::take(
            &mut *self
                .shared
                .outputs
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// Turns whatever the callback has queued into samples, returning every one whose duration
    /// is now known.
    fn collect(&mut self) -> Result<Vec<EncodedSample>> {
        let mut samples = Vec::new();
        for output in self.take_outputs() {
            let picture = match output {
                Output::Picture(picture) => picture,
                // Its time is covered by the sample before it; see the module notes.
                Output::Dropped => continue,
                Output::Failed(error) => return Err(codec(error)),
            };
            let (data, is_sync) = match &self.declared {
                Declared::Hevc(declared) => {
                    let mut sets = picture.parameter_sets;
                    let data = reframe_length_prefixed(&picture.data, &mut sets)
                        .ok_or_else(|| codec("VideoToolbox returned a malformed HEVC access unit"))?
                        .data;
                    if !declared.contains_all(&sets) {
                        return Err(codec(
                            "VideoToolbox changed its HEVC parameter sets after the stream was declared",
                        ));
                    }
                    (data, picture.is_sync)
                }
                Declared::Vp9 { vpcc, level } => {
                    // Only a key frame makes a VP9 sample a sync sample.
                    let key = zvidlib_vp9_syntax::key_frame_vpcc(&picture.data, *level);
                    if key.as_ref().is_some_and(|key| key != vpcc) {
                        return Err(codec(
                            "VideoToolbox changed its VP9 colour signalling after the stream was declared",
                        ));
                    }
                    (picture.data, key.is_some())
                }
            };
            if data.len() as u64 > self.limits.max_allocation_bytes {
                return Err(limit(format!(
                    "{} sample exceeds configured allocation limit",
                    self.name()
                )));
            }
            // The stream timestamp, undoing the one-frame offset every frame was submitted at.
            let mut tick = self.ticks(picture.pts)? - i64::from(self.configuration.frame_duration);
            if tick < 0 {
                // The priming frame's own sample, already accounted for.
                continue;
            }
            if !self.emitted && self.pending.is_none() {
                // The stream opens at zero even if its first frame was dropped; this picture
                // covers the gap.
                tick = 0;
            }
            if let Some(mut previous) = self.pending.take() {
                previous.duration = u32::try_from(tick - previous.dts)
                    .ok()
                    .filter(|duration| *duration > 0)
                    .ok_or_else(|| codec("VideoToolbox returned frames out of order"))?;
                samples.push(previous);
            }
            self.pending = Some(EncodedSample {
                data,
                dts: tick,
                pts: tick,
                duration: self.configuration.frame_duration,
                is_sync,
                dependency: if is_sync {
                    SampleDependency::INDEPENDENT
                } else {
                    SampleDependency::DEPENDENT
                },
            });
        }
        self.emitted |= !samples.is_empty();
        Ok(samples)
    }

    /// `time` in the configured timescale.
    fn ticks(&self, time: CMTime) -> Result<i64> {
        let timescale = i64::from(self.configuration.timescale);
        if time.flags & CM_TIME_VALID == 0 || time.timescale <= 0 {
            return Err(codec("VideoToolbox returned a frame without a timestamp"));
        }
        if i64::from(time.timescale) == timescale {
            return Ok(time.value);
        }
        let scaled = i128::from(time.value) * i128::from(timescale) / i128::from(time.timescale);
        i64::try_from(scaled).map_err(|_| limit("timeline overflows"))
    }

    fn check_frame(&self, frame: &VideoFrame) -> Result<()> {
        if frame.dimensions != self.configuration.coded_dimensions
            || frame.pixel_format != self.configuration.input_format
            || frame.color_range != self.configuration.color_range
        {
            return Err(invalid_input(
                "input frame does not match the configured size and format",
            ));
        }
        Ok(())
    }
}

impl VideoEncoder for VideoToolboxEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn implementation(&self) -> CodecImplementation {
        // Session creation required a hardware encoder and checked it got one.
        CodecImplementation::Hardware
    }

    fn backend_name(&self) -> &str {
        match self.configuration.codec {
            Codec::Vp9 => "VideoToolbox VP9",
            _ => "VideoToolbox HEVC",
        }
    }

    fn format(&self) -> VideoEncoderFormat {
        VideoEncoderFormat {
            dimensions: self.configuration.coded_dimensions,
            pixel_format: self.configuration.input_format,
        }
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        source: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            if self.finished {
                return Err(Error::new(
                    ErrorKind::InvalidState,
                    format!("{} encoder has already been finished", self.name()),
                ));
            }
            if index.0 != self.next_index {
                return Err(invalid_input(format!(
                    "{} encoder frame indexes must be consecutive and start at zero",
                    self.name()
                )));
            }
            let source = match source {
                FrameSource::Cpu(source) => source,
                FrameSource::Graphics(_) => {
                    return Err(unsupported(format!(
                        "the VideoToolbox {} encoder requires a CPU frame source",
                        self.name()
                    )));
                }
            };
            self.check_frame(source.frame)?;
            let buffer = self.pixel_buffer(source.frame, source.orientation)?;
            // One frame duration later than the stream timestamp; see the module notes.
            let tick = index
                .0
                .checked_add(1)
                .and_then(|frames| frames.checked_mul(u64::from(self.configuration.frame_duration)))
                .and_then(|tick| i64::try_from(tick).ok())
                .ok_or_else(|| limit("timeline overflows"))?;
            self.submit(&buffer, tick, index.0 == 0)?;
            self.next_index += 1;
            self.collect()
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            if self.finished {
                return Ok(Vec::new());
            }
            self.complete()?;
            let mut samples = self.collect()?;
            self.finished = true;
            if let Some(mut last) = self.pending.take() {
                // Stretched to the end of the input, over any frames dropped after it.
                let end = self
                    .next_index
                    .checked_mul(u64::from(self.configuration.frame_duration))
                    .and_then(|end| i64::try_from(end).ok())
                    .ok_or_else(|| limit("timeline overflows"))?;
                last.duration = u32::try_from(end - last.dts)
                    .ok()
                    .filter(|duration| *duration > 0)
                    .ok_or_else(|| codec("VideoToolbox returned a frame past the end"))?;
                samples.push(last);
            }
            if self.next_index > 0 && !self.emitted && samples.is_empty() {
                return Err(codec("VideoToolbox dropped every frame"));
            }
            self.emitted |= !samples.is_empty();
            Ok(samples)
        })
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        // SAFETY: invalidating tears the session down without waiting for frames it has not
        // emitted - the cancellation path - and no callback runs after it returns, so the refcon's
        // reference to `shared` can be released.
        unsafe {
            VTCompressionSessionInvalidate(self.session);
            CFRelease(self.session.cast_const());
            drop(Arc::from_raw(Arc::as_ptr(&self.shared)));
        }
    }
}

unsafe extern "C" fn on_output(
    refcon: *mut c_void,
    _frame_refcon: *mut c_void,
    status: OSStatus,
    flags: u32,
    sample: CMSampleBufferRef,
) {
    // SAFETY: `refcon` is the encoder's strong reference to its queue, alive while the session is.
    let shared = unsafe { &*refcon.cast_const().cast::<Shared>() };
    let output = if status != 0 {
        Output::Failed(format!("VideoToolbox encode failed (OSStatus {status})"))
    } else if sample.is_null() || flags & FRAME_DROPPED != 0 {
        Output::Dropped
    } else {
        // SAFETY: the sample buffer is valid for the duration of the callback.
        unsafe { read_sample(sample, shared.hevc) }
    };
    shared
        .outputs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(output);
}

/// Copies an encoded sample, and for HEVC the parameter sets it declares, out of its buffer.
unsafe fn read_sample(sample: CMSampleBufferRef, hevc: bool) -> Output {
    // SAFETY: the caller passes a valid sample buffer, and everything read from it is copied
    // before the callback returns.
    unsafe {
        let parameter_sets = if hevc {
            match read_parameter_sets(sample) {
                Ok(sets) => sets,
                Err(error) => return Output::Failed(error),
            }
        } else {
            ParameterSets::default()
        };
        let block = CMSampleBufferGetDataBuffer(sample);
        if block.is_null() {
            return Output::Failed("VideoToolbox returned a sample without data".into());
        }
        let length = CMBlockBufferGetDataLength(block);
        let mut data = vec![0_u8; length];
        let status = CMBlockBufferCopyDataBytes(block, 0, length, data.as_mut_ptr().cast());
        if status != 0 {
            return Output::Failed(format!(
                "could not copy VideoToolbox output (OSStatus {status})"
            ));
        }
        let mut is_sync = true;
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, 0);
        if !attachments.is_null() && CFArrayGetCount(attachments) > 0 {
            let first = CFArrayGetValueAtIndex(attachments, 0);
            let not_sync = CFDictionaryGetValue(first, kCMSampleAttachmentKey_NotSync);
            is_sync = not_sync.is_null() || CFBooleanGetValue(not_sync) == 0;
        }
        Output::Picture(Picture {
            data,
            pts: CMSampleBufferGetPresentationTimeStamp(sample),
            is_sync,
            parameter_sets,
        })
    }
}

/// The HEVC parameter sets an encoded sample's format description declares.
unsafe fn read_parameter_sets(
    sample: CMSampleBufferRef,
) -> std::result::Result<ParameterSets, String> {
    // SAFETY: the caller passes a valid sample buffer, and every set is copied out of it.
    unsafe {
        let mut parameter_sets = ParameterSets::default();
        let description = CMSampleBufferGetFormatDescription(sample);
        if description.is_null() {
            return Err("VideoToolbox returned HEVC without a format".into());
        }
        let mut count = 0_usize;
        let mut header_length = 0_i32;
        let status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
            description,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut count,
            &mut header_length,
        );
        if status != 0 {
            return Err(format!(
                "VideoToolbox HEVC format has no parameter sets (OSStatus {status})"
            ));
        }
        if header_length != 4 {
            return Err(format!(
                "VideoToolbox HEVC uses {header_length}-byte NAL lengths instead of 4"
            ));
        }
        for index in 0..count {
            let mut set = ptr::null();
            let mut size = 0_usize;
            if CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                description,
                index,
                &mut set,
                &mut size,
                ptr::null_mut(),
                ptr::null_mut(),
            ) == 0
                && !set.is_null()
            {
                parameter_sets.collect(std::slice::from_raw_parts(set, size));
            }
        }
        Ok(parameter_sets)
    }
}

/// An owned Core Foundation object, released on drop.
struct Owned(CFTypeRef);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: this holds the only reference it was created with.
            unsafe { CFRelease(self.0) };
        }
    }
}

fn number_i32(value: i32) -> Owned {
    // SAFETY: CFNumberCreate copies the value.
    Owned(unsafe { CFNumberCreate(ptr::null(), CF_NUMBER_SINT32, (&raw const value).cast()) })
}

fn number_f64(value: f64) -> Owned {
    // SAFETY: CFNumberCreate copies the value.
    Owned(unsafe { CFNumberCreate(ptr::null(), CF_NUMBER_FLOAT64, (&raw const value).cast()) })
}

/// # Safety
///
/// Every key and value must be a valid CF object.
unsafe fn dictionary(entries: &[(CFStringRef, CFTypeRef)]) -> Owned {
    // SAFETY: the dictionary retains its keys and values; the caller vouches for them.
    unsafe {
        let dictionary = CFDictionaryCreateMutable(
            ptr::null(),
            0,
            (&raw const kCFTypeDictionaryKeyCallBacks).cast(),
            (&raw const kCFTypeDictionaryValueCallBacks).cast(),
        );
        for &(key, value) in entries {
            CFDictionarySetValue(dictionary, key, value);
        }
        Owned(dictionary.cast_const())
    }
}

/// `{ RequireHardwareAcceleratedVideoEncoder: true }`, so VideoToolbox never substitutes its
/// software encoder for a caller that was told it would get hardware.
unsafe fn hardware_specification() -> Owned {
    // SAFETY: both are framework constants.
    unsafe {
        dictionary(&[(
            kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
            kCFBooleanTrue,
        )])
    }
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}
