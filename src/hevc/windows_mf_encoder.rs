//! Windows Media Foundation HEVC Main encode: the GPU vendor's hardware MFT
//! when one is registered (NVENC, Quick Sync, AMF), else Microsoft's software
//! HEVC MFT from the HEVC Video Extensions.
//!
//! Each encoder runs on its own worker thread, as the Media Foundation decoder
//! in [`super::windows_mf`] does, so the MFT lives in a multithreaded COM
//! apartment of its own whatever apartment the caller's thread is in.
//!
//! The MFT is set up for real-time capture: constant frame rate, a target
//! bitrate, a caller-chosen keyframe interval, no B-frames (so output order is
//! input order and every sample's DTS equals its PTS) and low latency. Hardware
//! MFTs are asynchronous and are driven through their event queue; software
//! ones are synchronous. The output is Annex B, so each access unit is reframed
//! with four-byte lengths and its parameter sets are moved into the `hvcC`,
//! which has to exist before the first frame is submitted and is therefore
//! taken from the MFT's sequence header or, for an MFT that does not publish
//! one, from a one-frame probe encode.
//!
//! RGBA and BGRA input is handed to a hardware MFT as ARGB32 when it accepts
//! that, so the colour conversion runs on the GPU; everything else is
//! converted to NV12 on the CPU with the same BT.601 studio-swing kernels the
//! native encoder uses.

use std::ffi::c_void;
use std::future::Future;
use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use windows::Win32::Media::MediaFoundation::{
    CODECAPI_AVEncCommonMaxBitRate, CODECAPI_AVEncCommonMeanBitRate,
    CODECAPI_AVEncCommonRateControlMode, CODECAPI_AVEncMPVDefaultBPictureCount,
    CODECAPI_AVEncMPVGOPSize, CODECAPI_AVLowLatencyMode, ICodecAPI, IMFActivate, IMFMediaEvent,
    IMFMediaEventGenerator, IMFMediaType, IMFSample, IMFShutdown, IMFTransform, MEError,
    METransformDrainComplete, METransformHaveOutput, METransformNeedInput,
    MF_E_NO_EVENTS_AVAILABLE, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_EVENT_FLAG_NO_WAIT, MF_LOW_LATENCY, MF_MT_AVG_BITRATE,
    MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE,
    MF_MT_MAJOR_TYPE, MF_MT_MPEG_SEQUENCE_HEADER, MF_MT_MPEG2_PROFILE, MF_MT_PIXEL_ASPECT_RATIO,
    MF_MT_SUBTYPE, MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_YUV_MATRIX, MF_TRANSFORM_ASYNC,
    MF_TRANSFORM_ASYNC_UNLOCK, MF_VERSION, MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample,
    MFMediaType_Video, MFNominalRange_16_235, MFSampleExtension_CleanPoint, MFShutdown, MFStartup,
    MFT_CATEGORY_VIDEO_ENCODER, MFT_ENUM_FLAG, MFT_ENUM_FLAG_HARDWARE, MFT_ENUM_FLAG_SORTANDFILTER,
    MFT_ENUM_FLAG_SYNCMFT, MFT_FRIENDLY_NAME_Attribute, MFT_MESSAGE_COMMAND_DRAIN,
    MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
    MFT_MESSAGE_NOTIFY_END_OF_STREAM, MFT_MESSAGE_NOTIFY_END_STREAMING,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER,
    MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES, MFT_OUTPUT_STREAM_PROVIDES_SAMPLES,
    MFT_REGISTER_TYPE_INFO, MFTEnumEx, MFVideoFormat_ARGB32, MFVideoFormat_HEVC,
    MFVideoFormat_NV12, MFVideoInterlace_Progressive, MFVideoTransferMatrix_BT601,
    eAVEncCommonRateControlMode_PeakConstrainedVBR, eAVEncH265VProfile_Main_420_8,
};
use windows::Win32::System::Com::{
    COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree, CoUninitialize,
};
use windows::Win32::System::Variant::VARIANT;
use windows::core::{GUID, HRESULT, Interface, PWSTR};

use super::annexb::{self, ParameterSets};
use super::engine::encoder::colorconv;
use crate::{
    CodecImplementation, ColorRange, EncodedSample, EncoderConfig, EncoderFuture, Error, ErrorKind,
    FrameIndex, FrameSource, Limits, Orientation, PixelFormat, Result, SampleDependency,
    VideoEncoder, VideoEncoderFormat,
};

/// How long an asynchronous MFT may go without raising an event while this
/// backend is waiting on one before the encoder is reported as lost. A
/// healthy hardware encoder answers within a frame; the allowance is for the
/// first frame of a session, which can include driver initialization.
const STALL_TIMEOUT: Duration = Duration::from_secs(10);

/// How often a wait on an asynchronous MFT looks for its next event and for
/// cancellation.
const POLL_INTERVAL: Duration = Duration::from_micros(500);

/// Media Foundation sample times are in 100-nanosecond units.
const HNS_PER_SECOND: i128 = 10_000_000;

/// Which class of registered MFT to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MftClass {
    /// The GPU vendor's asynchronous hardware encoder.
    Hardware,
    /// Microsoft's synchronous software encoder.
    Software,
}

impl MftClass {
    fn enum_flags(self) -> MFT_ENUM_FLAG {
        match self {
            Self::Hardware => MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            Self::Software => MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_SORTANDFILTER,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
        }
    }
}

/// The encode an MFT is asked for, resolved from a
/// [`crate::VideoEncoderConfig`] by the factory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Settings {
    pub width: u32,
    pub height: u32,
    pub input_format: PixelFormat,
    pub timescale: u32,
    pub frame_duration: u32,
    pub bits_per_second: u32,
    /// Frames from one keyframe to the next. Nonzero.
    pub keyframe_interval: u32,
}

impl Settings {
    /// Why this backend cannot take `self`, if it cannot.
    pub(super) fn unsupported_reason(&self) -> Option<&'static str> {
        if !matches!(
            self.input_format,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Yuv420p8
        ) {
            return Some("Media Foundation HEVC encoding accepts Rgba8, Bgra8 or Yuv420p8 input");
        }
        if self.width % 2 != 0 || self.height % 2 != 0 {
            return Some("Media Foundation HEVC encoding requires even dimensions");
        }
        if self.timescale == 0 || self.frame_duration == 0 {
            return Some("HEVC encoding requires a nonzero timescale and frame duration");
        }
        // A frame shorter than two 100 ns ticks cannot be told apart from its
        // neighbour once Media Foundation rounds its timestamp.
        if i128::from(self.frame_duration) * HNS_PER_SECOND < 2 * i128::from(self.timescale) {
            return Some("Media Foundation HEVC encoding requires frames of at least 200 ns");
        }
        None
    }

    /// `MF_MT_FRAME_RATE`: the frame rate in lowest terms.
    fn frame_rate(&self) -> u64 {
        let divisor = gcd(self.timescale, self.frame_duration);
        (u64::from(self.timescale / divisor) << 32) | u64::from(self.frame_duration / divisor)
    }

    fn frame_size(&self) -> u64 {
        (u64::from(self.width) << 32) | u64::from(self.height)
    }

    /// The Media Foundation time of frame `index`.
    fn sample_time(&self, index: u64) -> i64 {
        let ticks = i128::from(index) * i128::from(self.frame_duration) * HNS_PER_SECOND;
        i64::try_from(ticks / i128::from(self.timescale)).unwrap_or(i64::MAX)
    }

    /// The frame a Media Foundation time belongs to, rounding to the nearest.
    fn frame_at(&self, time: i64) -> u64 {
        let per_frame = i128::from(self.frame_duration) * HNS_PER_SECOND;
        let scaled = i128::from(time.max(0)) * i128::from(self.timescale);
        u64::try_from((scaled + per_frame / 2) / per_frame).unwrap_or(u64::MAX)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// The layout the MFT was given for its input, which decides the conversion
/// every frame takes on the caller's thread before it is handed over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Feed {
    /// 8-bit 4:2:0 with interleaved chroma, converted on the CPU.
    Nv12,
    /// 32-bit BGRA in memory, converted to YUV by the MFT on the GPU.
    Argb32,
}

impl Feed {
    fn subtype(self) -> GUID {
        match self {
            Self::Nv12 => MFVideoFormat_NV12,
            Self::Argb32 => MFVideoFormat_ARGB32,
        }
    }

    fn stride(self, width: u32) -> u32 {
        match self {
            Self::Nv12 => width,
            Self::Argb32 => width * 4,
        }
    }

    fn frame_len(self, width: u32, height: u32) -> usize {
        let (width, height) = (width as usize, height as usize);
        match self {
            Self::Nv12 => width * height * 3 / 2,
            Self::Argb32 => width * height * 4,
        }
    }
}

/// Whether an MFT of `class` can take `settings`, by configuring one.
pub(super) fn probe(settings: Settings, class: MftClass) -> Result<()> {
    run_on_mf_thread("zvidlib-mf-hevc-probe", move || {
        let mut reasons = Vec::new();
        for activate in candidates(class)? {
            let name = friendly_name(&activate);
            match Mft::open(activate, class, &settings) {
                Ok(_) => return Ok(()),
                Err(error) => reasons.push(format!("{name}: {}", error.message())),
            }
        }
        Err(unavailable(class, reasons))
    })
}

/// Creates an encoder on the first MFT of `class` that accepts `settings`.
pub(super) fn create(
    settings: Settings,
    class: MftClass,
    limits: &Limits,
) -> Result<Box<dyn VideoEncoder>> {
    let payload = u64::from(settings.width) * u64::from(settings.height) * 4;
    if payload > limits.max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "HEVC frame exceeds configured allocation limit",
        ));
    }
    MfHevcEncoder::spawn(settings, class).map(|encoder| Box::new(encoder) as Box<dyn VideoEncoder>)
}

/// Runs `work` on a fresh thread with COM and Media Foundation started.
fn run_on_mf_thread<T: Send + 'static>(
    name: &str,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let (done_tx, done_rx) = sync_channel(1);
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let result = MfRuntime::start().and_then(|_runtime| work());
            let _ = done_tx.send(result);
        })
        .map_err(|error| codec(format!("could not start Media Foundation thread: {error}")))?;
    done_rx
        .recv()
        .map_err(|_| codec("Media Foundation thread stopped unexpectedly"))?
}

fn unavailable(class: MftClass, reasons: Vec<String>) -> Error {
    let detail = if reasons.is_empty() {
        "none is installed".to_owned()
    } else {
        reasons.join("; ")
    };
    Error::new(
        ErrorKind::Unsupported,
        format!(
            "no {} Media Foundation HEVC encoder accepts this configuration ({detail})",
            class.label()
        ),
    )
}

/// Registered HEVC encoders of `class`, best first.
fn candidates(class: MftClass) -> Result<Vec<IMFActivate>> {
    let output = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Video,
        guidSubtype: MFVideoFormat_HEVC,
    };
    let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
    let mut count = 0u32;
    // SAFETY: the out pointers are valid, and the array MFTEnumEx allocates is
    // read within its reported length and then freed.
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_ENCODER,
            class.enum_flags(),
            None,
            Some(&output),
            &mut list,
            &mut count,
        )
        .map_err(|error| windows_error("could not enumerate HEVC encoders", error))?;
        if list.is_null() {
            return Ok(Vec::new());
        }
        let found = (0..count as usize)
            .filter_map(|index| (*list.add(index)).take())
            .collect();
        CoTaskMemFree(Some(list as *const c_void));
        Ok(found)
    }
}

fn friendly_name(activate: &IMFActivate) -> String {
    let mut value = PWSTR::null();
    let mut len = 0;
    // SAFETY: the out pointers are valid; the string is freed after copying.
    unsafe {
        if activate
            .GetAllocatedString(&MFT_FRIENDLY_NAME_Attribute, &mut value, &mut len)
            .is_err()
        {
            return "unnamed HEVC encoder".to_owned();
        }
        let name = value.to_string().unwrap_or_default();
        CoTaskMemFree(Some(value.0 as *const c_void));
        name
    }
}

/// One configured encoder MFT.
struct Mft {
    transform: IMFTransform,
    activate: IMFActivate,
    /// The event queue of an asynchronous MFT.
    events: Option<IMFMediaEventGenerator>,
    /// `METransformNeedInput` events not yet answered.
    need_input: u32,
    provides_samples: bool,
    output_size: u32,
    feed: Feed,
    class: MftClass,
    name: String,
    streaming: bool,
}

/// One compressed output buffer as the MFT produced it.
struct RawOutput {
    stream: Vec<u8>,
    time: Option<i64>,
    clean_point: bool,
}

impl Mft {
    /// Activates and configures `activate`, without starting to stream.
    fn open(activate: IMFActivate, class: MftClass, settings: &Settings) -> Result<Self> {
        let name = friendly_name(&activate);
        // SAFETY: plain Media Foundation calls on objects this thread owns.
        let transform: IMFTransform = unsafe { activate.ActivateObject() }
            .map_err(|error| windows_error("could not activate the encoder", error))?;
        let mut mft = Self {
            transform,
            activate,
            events: None,
            need_input: 0,
            provides_samples: false,
            output_size: 0,
            feed: Feed::Nv12,
            class,
            name,
            streaming: false,
        };
        mft.configure(settings)?;
        Ok(mft)
    }

    fn configure(&mut self, settings: &Settings) -> Result<()> {
        // SAFETY: plain Media Foundation calls on objects this thread owns.
        unsafe {
            let attributes = self.transform.GetAttributes().ok();
            let is_async = attributes
                .as_ref()
                .and_then(|attributes| attributes.GetUINT32(&MF_TRANSFORM_ASYNC).ok())
                .unwrap_or(0)
                != 0;
            if is_async {
                let attributes = attributes
                    .as_ref()
                    .ok_or_else(|| codec("asynchronous encoder has no attributes"))?;
                attributes
                    .SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1)
                    .map_err(|error| windows_error("could not unlock the encoder", error))?;
                self.events = Some(
                    self.transform
                        .cast::<IMFMediaEventGenerator>()
                        .map_err(|error| windows_error("encoder has no event queue", error))?,
                );
            }
            if let Some(attributes) = &attributes {
                let _ = attributes.SetUINT32(&MF_LOW_LATENCY, 1);
            }

            // Rate control and GOP structure go in before the media types:
            // several encoders only read them when the output type is set.
            self.set_codec_values(settings);

            let output = video_type(settings, MFVideoFormat_HEVC)?;
            set(
                output.SetUINT32(&MF_MT_AVG_BITRATE, settings.bits_per_second),
                "bitrate",
            )?;
            set(
                output.SetUINT32(&MF_MT_MPEG2_PROFILE, eAVEncH265VProfile_Main_420_8.0 as u32),
                "profile",
            )?;
            self.transform
                .SetOutputType(0, &output, 0)
                .map_err(|error| windows_error("encoder rejected the HEVC Main output", error))?;

            self.feed = self.set_input_type(settings)?;
            // Some encoders settle their final values only once both types
            // are known, so the settings are applied again.
            self.set_codec_values(settings);

            let info = self
                .transform
                .GetOutputStreamInfo(0)
                .map_err(|error| windows_error("could not read the output stream", error))?;
            self.provides_samples = info.dwFlags
                & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32
                    | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0 as u32)
                != 0;
            self.output_size = info.cbSize.max(
                settings
                    .width
                    .saturating_mul(settings.height)
                    .saturating_mul(3)
                    / 2,
            );
        }
        Ok(())
    }

    /// Picks the input layout: ARGB32 where a hardware encoder will convert
    /// RGB itself, else NV12.
    fn set_input_type(&self, settings: &Settings) -> Result<Feed> {
        let rgb = matches!(
            settings.input_format,
            PixelFormat::Rgba8 | PixelFormat::Bgra8
        );
        let feeds: &[Feed] = if rgb && self.class == MftClass::Hardware {
            &[Feed::Argb32, Feed::Nv12]
        } else {
            &[Feed::Nv12]
        };
        let mut last = None;
        for &feed in feeds {
            let input = video_type(settings, feed.subtype())?;
            // SAFETY: plain Media Foundation calls on objects this thread owns.
            let result = unsafe {
                // Positive: top-down rows, packed with no padding.
                set(
                    input.SetUINT32(&MF_MT_DEFAULT_STRIDE, feed.stride(settings.width)),
                    "input stride",
                )?;
                // The crate's decoder, like its encoder, is BT.601 studio
                // swing throughout, so an encoder converting RGB itself has
                // to be asked for the same matrix.
                set(
                    input.SetUINT32(&MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT601.0 as u32),
                    "colour matrix",
                )?;
                set(
                    input.SetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235.0 as u32),
                    "nominal range",
                )?;
                self.transform.SetInputType(0, &input, 0)
            };
            match result {
                Ok(()) => return Ok(feed),
                Err(error) => last = Some(error),
            }
        }
        Err(windows_error(
            "encoder rejected the input format",
            last.expect("at least one input format is tried"),
        ))
    }

    /// Applies the encoder settings that are not media type attributes. Each
    /// is best effort, since encoders differ in which they expose; the one
    /// that matters for correctness, no B-frames, is checked on the output
    /// instead, where a reordered picture is reported as an error.
    fn set_codec_values(&self, settings: &Settings) {
        let Ok(api) = self.transform.cast::<ICodecAPI>() else {
            return;
        };
        let values = [
            (CODECAPI_AVLowLatencyMode, VARIANT::from(true)),
            (
                CODECAPI_AVEncCommonRateControlMode,
                VARIANT::from(eAVEncCommonRateControlMode_PeakConstrainedVBR.0 as u32),
            ),
            (
                CODECAPI_AVEncCommonMeanBitRate,
                VARIANT::from(settings.bits_per_second),
            ),
            (
                CODECAPI_AVEncCommonMaxBitRate,
                VARIANT::from(settings.bits_per_second.saturating_mul(2)),
            ),
            (
                CODECAPI_AVEncMPVGOPSize,
                VARIANT::from(settings.keyframe_interval),
            ),
            (CODECAPI_AVEncMPVDefaultBPictureCount, VARIANT::from(0u32)),
        ];
        for (property, value) in &values {
            // SAFETY: both pointers are valid for the call.
            let _ = unsafe { api.SetValue(property, value) };
        }
    }

    /// Parameter sets the MFT publishes on its output type, if it does.
    fn sequence_header(&self) -> ParameterSets {
        let mut sets = ParameterSets::default();
        // SAFETY: the blob is copied and then freed.
        unsafe {
            let Ok(kind) = self.transform.GetOutputCurrentType(0) else {
                return sets;
            };
            let mut data = std::ptr::null_mut();
            let mut len = 0;
            if kind
                .GetAllocatedBlob(&MF_MT_MPEG_SEQUENCE_HEADER, &mut data, &mut len)
                .is_ok()
            {
                let header = std::slice::from_raw_parts(data, len as usize).to_vec();
                CoTaskMemFree(Some(data as *const c_void));
                for nal in annexb::split_annex_b(&header) {
                    sets.collect(nal);
                }
            }
        }
        sets
    }

    fn start(&mut self) -> Result<()> {
        // SAFETY: plain Media Foundation calls on objects this thread owns.
        unsafe {
            self.transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
                .map_err(|error| mf_error("could not start the encoder", error))?;
            self.streaming = true;
            self.transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
                .map_err(|error| mf_error("could not start the encoder", error))?;
        }
        Ok(())
    }

    /// Submits one frame and collects whatever output is ready.
    fn submit(
        &mut self,
        sample: &IMFSample,
        cancelled: &AtomicBool,
        out: &mut Vec<RawOutput>,
    ) -> Result<()> {
        if self.events.is_some() {
            while self.need_input == 0 {
                self.next_event(true, cancelled, out)?;
            }
            // SAFETY: plain Media Foundation call.
            unsafe { self.transform.ProcessInput(0, sample, 0) }
                .map_err(|error| mf_error("encoder rejected a frame", error))?;
            self.need_input -= 1;
            while self.next_event(false, cancelled, out)?.is_some() {}
        } else {
            // SAFETY: plain Media Foundation call.
            match unsafe { self.transform.ProcessInput(0, sample, 0) } {
                Ok(()) => {}
                Err(error) if error.code() == MF_E_NOTACCEPTING => {
                    self.drain_sync(out)?;
                    // SAFETY: plain Media Foundation call.
                    unsafe { self.transform.ProcessInput(0, sample, 0) }
                        .map_err(|error| mf_error("encoder rejected a frame", error))?;
                }
                Err(error) => return Err(mf_error("encoder rejected a frame", error)),
            }
            self.drain_sync(out)?;
        }
        Ok(())
    }

    /// Ends the stream and collects every output still inside the MFT.
    fn drain(&mut self, cancelled: &AtomicBool, out: &mut Vec<RawOutput>) -> Result<()> {
        // SAFETY: plain Media Foundation calls.
        unsafe {
            let _ = self
                .transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
            self.transform
                .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                .map_err(|error| mf_error("could not drain the encoder", error))?;
        }
        if self.events.is_some() {
            while self.next_event(true, cancelled, out)? != Some(METransformDrainComplete.0 as u32)
            {
            }
        } else {
            self.drain_sync(out)?;
        }
        Ok(())
    }

    /// Discards everything inside the MFT, for a cancelled stream.
    fn flush(&mut self) {
        // SAFETY: plain Media Foundation call.
        let _ = unsafe { self.transform.ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0) };
    }

    /// Pulls every output a synchronous MFT has ready.
    fn drain_sync(&mut self, out: &mut Vec<RawOutput>) -> Result<()> {
        while let Some(output) = self.process_output()? {
            out.extend(output);
        }
        Ok(())
    }

    /// One `ProcessOutput` call: `None` when the MFT needs more input.
    fn process_output(&mut self) -> Result<Option<Option<RawOutput>>> {
        // SAFETY: the output buffer array lives across the call, and the
        // sample and event collection it may hold are released below.
        unsafe {
            let sample = if self.provides_samples {
                None
            } else {
                let sample = MFCreateSample().map_err(|error| mf_error("output sample", error))?;
                let buffer = MFCreateMemoryBuffer(self.output_size)
                    .map_err(|error| mf_error("output buffer", error))?;
                sample
                    .AddBuffer(&buffer)
                    .map_err(|error| mf_error("output buffer", error))?;
                Some(sample)
            };
            let mut buffers = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(sample),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0;
            let result = self.transform.ProcessOutput(0, &mut buffers, &mut status);
            let sample = ManuallyDrop::take(&mut buffers[0].pSample);
            drop(ManuallyDrop::take(&mut buffers[0].pEvents));
            match result {
                Ok(()) => {}
                Err(error) if error.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(None),
                Err(error) if error.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    let kind = self
                        .transform
                        .GetOutputAvailableType(0, 0)
                        .map_err(|error| mf_error("encoder output changed", error))?;
                    self.transform
                        .SetOutputType(0, &kind, 0)
                        .map_err(|error| mf_error("encoder output changed", error))?;
                    let info = self
                        .transform
                        .GetOutputStreamInfo(0)
                        .map_err(|error| mf_error("encoder output changed", error))?;
                    self.output_size = info.cbSize.max(self.output_size);
                    return Ok(Some(None));
                }
                Err(error) => return Err(mf_error("encoder failed to produce output", error)),
            }
            let Some(sample) = sample else {
                return Ok(Some(None));
            };
            let buffer = sample
                .ConvertToContiguousBuffer()
                .map_err(|error| mf_error("encoder output", error))?;
            let mut data = std::ptr::null_mut();
            let mut len = 0;
            buffer
                .Lock(&mut data, None, Some(&mut len))
                .map_err(|error| mf_error("encoder output", error))?;
            let stream = std::slice::from_raw_parts(data, len as usize).to_vec();
            buffer
                .Unlock()
                .map_err(|error| mf_error("encoder output", error))?;
            Ok(Some(Some(RawOutput {
                stream,
                time: sample.GetSampleTime().ok(),
                clean_point: sample.GetUINT32(&MFSampleExtension_CleanPoint).unwrap_or(0) != 0,
            })))
        }
    }

    /// Waits for (or, with `wait` false, polls for) the next event of an
    /// asynchronous MFT and handles it, returning its type.
    fn next_event(
        &mut self,
        wait: bool,
        cancelled: &AtomicBool,
        out: &mut Vec<RawOutput>,
    ) -> Result<Option<u32>> {
        let events = self
            .events
            .clone()
            .expect("only asynchronous MFTs have events");
        let deadline = Instant::now() + STALL_TIMEOUT;
        loop {
            // The blocking form of `GetEvent` cannot be interrupted, so a
            // wait polls instead, which is what lets it notice cancellation
            // and an encoder that has stopped answering.
            // SAFETY: plain Media Foundation call.
            match unsafe { events.GetEvent(MF_EVENT_FLAG_NO_WAIT) } {
                Ok(event) => return self.on_event(&event, out).map(Some),
                Err(error) if error.code() == MF_E_NO_EVENTS_AVAILABLE => {
                    if !wait {
                        return Ok(None);
                    }
                    if cancelled.load(Ordering::Acquire) {
                        return Err(cancelled_error());
                    }
                    if Instant::now() >= deadline {
                        return Err(device_lost(format!(
                            "{} stopped responding for {} s",
                            self.name,
                            STALL_TIMEOUT.as_secs()
                        )));
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                Err(error) => return Err(mf_error("encoder event queue failed", error)),
            }
        }
    }

    fn on_event(&mut self, event: &IMFMediaEvent, out: &mut Vec<RawOutput>) -> Result<u32> {
        // SAFETY: plain Media Foundation calls.
        let (kind, status) = unsafe {
            (
                event
                    .GetType()
                    .map_err(|error| mf_error("encoder event", error))?,
                event
                    .GetStatus()
                    .map_err(|error| mf_error("encoder event", error))?,
            )
        };
        if status.is_err() || kind == MEError.0 as u32 {
            let status = if status.is_err() {
                status
            } else {
                HRESULT(0x8000_4005_u32 as i32) // E_FAIL
            };
            return Err(mf_error(
                "encoder reported an error",
                windows::core::Error::from_hresult(status),
            ));
        }
        if kind == METransformNeedInput.0 as u32 {
            self.need_input += 1;
        } else if kind == METransformHaveOutput.0 as u32
            && let Some(output) = self.process_output()?
        {
            out.extend(output);
        }
        Ok(kind)
    }
}

impl Drop for Mft {
    fn drop(&mut self) {
        // SAFETY: plain Media Foundation calls; the MFT is released afterwards.
        unsafe {
            if self.streaming {
                let _ = self
                    .transform
                    .ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
            }
            if let Ok(shutdown) = self.transform.cast::<IMFShutdown>() {
                let _ = shutdown.Shutdown();
            }
            let _ = self.activate.ShutdownObject();
        }
    }
}

/// A progressive, square-pixel video media type of `subtype` at the
/// configured size and rate.
fn video_type(settings: &Settings, subtype: GUID) -> Result<IMFMediaType> {
    // SAFETY: plain Media Foundation calls on a type this thread owns.
    unsafe {
        let kind = MFCreateMediaType().map_err(|error| windows_error("media type", error))?;
        set(
            kind.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video),
            "major type",
        )?;
        set(kind.SetGUID(&MF_MT_SUBTYPE, &subtype), "subtype")?;
        set(
            kind.SetUINT64(&MF_MT_FRAME_SIZE, settings.frame_size()),
            "frame size",
        )?;
        set(
            kind.SetUINT64(&MF_MT_FRAME_RATE, settings.frame_rate()),
            "frame rate",
        )?;
        set(
            kind.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1 << 32) | 1),
            "pixel aspect ratio",
        )?;
        set(
            kind.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32),
            "interlace mode",
        )?;
        Ok(kind)
    }
}

fn set(result: windows::core::Result<()>, what: &str) -> Result<()> {
    result.map_err(|error| windows_error(&format!("could not set the {what}"), error))
}

/// The encoder as the worker thread holds it.
struct Core {
    mft: Mft,
    settings: Settings,
    /// The parameter sets the `hvcC` declares.
    declared: ParameterSets,
    /// Frames submitted, in order, not yet matched to an output.
    pending: std::collections::VecDeque<u64>,
    /// The last frame an output was emitted for.
    last_emitted: Option<u64>,
    /// Set once the stream has failed, been cancelled or been finished; every
    /// later request answers with it.
    failure: Option<Error>,
    // Declared last so COM and Media Foundation outlive the MFT.
    _runtime: MfRuntime,
}

impl Core {
    fn open(settings: Settings, class: MftClass) -> Result<Self> {
        let runtime = MfRuntime::start()?;
        let mut reasons = Vec::new();
        for activate in candidates(class)? {
            let name = friendly_name(&activate);
            match Self::open_with(activate, class, settings) {
                Ok((mft, declared)) => {
                    return Ok(Self {
                        mft,
                        settings,
                        declared,
                        pending: Default::default(),
                        last_emitted: None,
                        failure: None,
                        _runtime: runtime,
                    });
                }
                Err(error) => reasons.push(format!("{name}: {}", error.message())),
            }
        }
        Err(unavailable(class, reasons))
    }

    /// Opens and starts one candidate, with the parameter sets its stream
    /// will reference.
    fn open_with(
        activate: IMFActivate,
        class: MftClass,
        settings: Settings,
    ) -> Result<(Mft, ParameterSets)> {
        let mut mft = Mft::open(activate.clone(), class, &settings)?;
        let mut declared = mft.sequence_header();
        if !declared.is_complete() {
            // This MFT only writes its parameter sets in-band, so encode one
            // throwaway frame to learn them. The probe has to be a separate
            // instance: the real stream must start with its own first frame.
            drop(mft);
            declared = probe_parameter_sets(activate.clone(), class, &settings)?;
            mft = Mft::open(activate, class, &settings)?;
        }
        if declared.hvcc().is_none() {
            return Err(codec(
                "encoder parameter sets do not describe an HEVC stream",
            ));
        }
        mft.start()?;
        Ok((mft, declared))
    }

    fn encode(
        &mut self,
        index: u64,
        payload: &[u8],
        cancelled: &AtomicBool,
    ) -> Result<Vec<EncodedSample>> {
        self.guard(cancelled, |core| {
            let sample = input_sample(payload, &core.settings, index)?;
            let mut out = Vec::new();
            core.mft.submit(&sample, cancelled, &mut out)?;
            core.pending.push_back(index);
            core.samples(out)
        })
    }

    fn finish(&mut self, cancelled: &AtomicBool) -> Result<Vec<EncodedSample>> {
        let samples = self.guard(cancelled, |core| {
            let mut out = Vec::new();
            core.mft.drain(cancelled, &mut out)?;
            core.samples(out)
        })?;
        self.failure = Some(Error::new(
            ErrorKind::InvalidState,
            "the HEVC encoder has already been finished",
        ));
        Ok(samples)
    }

    /// Runs `work` unless the stream has already stopped, and stops it if
    /// `work` fails or is cancelled.
    fn guard(
        &mut self,
        cancelled: &AtomicBool,
        work: impl FnOnce(&mut Self) -> Result<Vec<EncodedSample>>,
    ) -> Result<Vec<EncodedSample>> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone());
        }
        let result = if cancelled.load(Ordering::Acquire) {
            Err(cancelled_error())
        } else {
            work(self)
        };
        if let Err(error) = &result {
            if error.kind() == ErrorKind::Cancelled {
                self.mft.flush();
            }
            self.failure = Some(error.clone());
        }
        result
    }

    fn samples(&mut self, outputs: Vec<RawOutput>) -> Result<Vec<EncodedSample>> {
        let mut samples = Vec::with_capacity(outputs.len());
        for output in outputs {
            let mut seen = ParameterSets::default();
            let Some(unit) = annexb::reframe_access_unit(&output.stream, &mut seen) else {
                continue;
            };
            if !self.declared.contains_all(&seen) {
                return Err(codec(
                    "encoder changed its parameter sets mid-stream, which an hvc1 track cannot carry",
                ));
            }
            let index = self.output_index(output.time)?;
            let is_sync = unit.is_irap || output.clean_point;
            let tick = index
                .checked_mul(u64::from(self.settings.frame_duration))
                .and_then(|tick| i64::try_from(tick).ok())
                .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC timeline overflows"))?;
            samples.push(EncodedSample {
                data: unit.data,
                dts: tick,
                pts: tick,
                duration: self.settings.frame_duration,
                is_sync,
                dependency: if is_sync {
                    SampleDependency::INDEPENDENT
                } else {
                    SampleDependency::DEPENDENT
                },
            });
        }
        Ok(samples)
    }

    /// Which submitted frame an output is. The MFT's timestamp says, where
    /// it gives one, which keeps a frame the encoder chose to drop from
    /// shifting every later one; without a timestamp outputs are matched to
    /// inputs in order.
    fn output_index(&mut self, time: Option<i64>) -> Result<u64> {
        let index = match time {
            Some(time) => {
                let index = self.settings.frame_at(time);
                while self.pending.front().is_some_and(|&pending| pending < index) {
                    self.pending.pop_front();
                }
                if self.pending.front() != Some(&index) {
                    return Err(codec(format!(
                        "encoder emitted a picture at {time} (frame {index}) that was not submitted"
                    )));
                }
                self.pending.pop_front();
                index
            }
            None => self
                .pending
                .pop_front()
                .ok_or_else(|| codec("encoder emitted more pictures than it was given"))?,
        };
        if self.last_emitted.is_some_and(|last| index <= last) {
            return Err(codec(
                "encoder reordered its output; B-frames could not be disabled",
            ));
        }
        self.last_emitted = Some(index);
        Ok(index)
    }
}

/// Encodes one black frame on a throwaway instance to learn the parameter
/// sets an MFT writes only in-band.
fn probe_parameter_sets(
    activate: IMFActivate,
    class: MftClass,
    settings: &Settings,
) -> Result<ParameterSets> {
    let mut mft = Mft::open(activate, class, settings)?;
    mft.start()?;
    let payload = match mft.feed {
        Feed::Nv12 => {
            let luma = settings.width as usize * settings.height as usize;
            let mut frame = vec![16; luma];
            frame.resize(luma * 3 / 2, 128);
            frame
        }
        Feed::Argb32 => [0, 0, 0, 255].repeat(settings.width as usize * settings.height as usize),
    };
    let sample = input_sample(&payload, settings, 0)?;
    let never = AtomicBool::new(false);
    let mut out = Vec::new();
    mft.submit(&sample, &never, &mut out)?;
    mft.drain(&never, &mut out)?;
    let mut sets = ParameterSets::default();
    for output in &out {
        for nal in annexb::split_annex_b(&output.stream) {
            sets.collect(nal);
        }
    }
    if sets.is_complete() {
        Ok(sets)
    } else {
        Err(codec("encoder did not emit its parameter sets"))
    }
}

fn input_sample(payload: &[u8], settings: &Settings, index: u64) -> Result<IMFSample> {
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "HEVC input frame is too large"))?;
    // SAFETY: the buffer is locked for the copy and holds `len` bytes.
    unsafe {
        let buffer = MFCreateMemoryBuffer(len).map_err(|error| mf_error("input buffer", error))?;
        let mut data = std::ptr::null_mut();
        buffer
            .Lock(&mut data, None, None)
            .map_err(|error| mf_error("input buffer", error))?;
        std::ptr::copy_nonoverlapping(payload.as_ptr(), data, payload.len());
        buffer
            .Unlock()
            .map_err(|error| mf_error("input buffer", error))?;
        buffer
            .SetCurrentLength(len)
            .map_err(|error| mf_error("input buffer", error))?;
        let sample = MFCreateSample().map_err(|error| mf_error("input sample", error))?;
        sample
            .AddBuffer(&buffer)
            .map_err(|error| mf_error("input sample", error))?;
        let time = settings.sample_time(index);
        sample
            .SetSampleTime(time)
            .map_err(|error| mf_error("input sample", error))?;
        sample
            .SetSampleDuration(settings.sample_time(index + 1) - time)
            .map_err(|error| mf_error("input sample", error))?;
        Ok(sample)
    }
}

enum Command {
    Encode {
        index: u64,
        payload: Vec<u8>,
        reply: Arc<Reply>,
    },
    Finish {
        reply: Arc<Reply>,
    },
}

/// Where the worker leaves one request's result for the future awaiting it.
#[derive(Default)]
struct Reply {
    state: Mutex<ReplyState>,
}

#[derive(Default)]
struct ReplyState {
    result: Option<Result<Vec<EncodedSample>>>,
    waker: Option<Waker>,
}

impl Reply {
    fn complete(&self, result: Result<Vec<EncodedSample>>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// Awaits one request. Dropping it before it resolves cancels the stream:
/// the worker stops waiting on the MFT, flushes it, and every later call on
/// the encoder reports [`ErrorKind::Cancelled`], since the caller can no
/// longer know which frames were encoded.
struct Pending {
    reply: Arc<Reply>,
    cancelled: Arc<AtomicBool>,
    done: bool,
}

impl Future for Pending {
    type Output = Result<Vec<EncodedSample>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .reply
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match state.result.take() {
            Some(result) => {
                drop(state);
                self.done = true;
                Poll::Ready(result)
            }
            None => {
                state.waker = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if !self.done {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

struct MfHevcEncoder {
    commands: Option<Sender<Command>>,
    worker: Option<JoinHandle<()>>,
    cancelled: Arc<AtomicBool>,
    config: EncoderConfig,
    settings: Settings,
    feed: Feed,
    class: MftClass,
    name: String,
    next_index: u64,
    finished: bool,
}

impl MfHevcEncoder {
    fn spawn(settings: Settings, class: MftClass) -> Result<Self> {
        let (command_tx, command_rx) = channel();
        let (ready_tx, ready_rx) = sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = thread::Builder::new()
            .name("zvidlib-mf-hevc-encode".into())
            .spawn(move || match Core::open(settings, class) {
                Ok(core) => {
                    let ready = (core.declared.hvcc(), core.mft.feed, core.mft.name.clone());
                    if ready_tx.send(Ok(ready)).is_ok() {
                        run_worker(core, &command_rx, &worker_cancelled);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            })
            .map_err(|error| codec(format!("could not start Media Foundation worker: {error}")))?;
        let ready = ready_rx.recv().unwrap_or_else(|_| {
            Err(codec(
                "Media Foundation worker stopped during initialization",
            ))
        });
        let (hvcc, feed, name) = match ready {
            Ok(ready) => ready,
            Err(error) => {
                let _ = worker.join();
                return Err(error);
            }
        };
        Ok(Self {
            commands: Some(command_tx),
            worker: Some(worker),
            cancelled,
            config: EncoderConfig {
                codec: crate::Codec::Hevc,
                timescale: settings.timescale,
                decoder_config: hvcc.expect("checked when the encoder was opened"),
            },
            settings,
            feed,
            class,
            name,
            next_index: 0,
            finished: false,
        })
    }

    fn request(&self, command: impl FnOnce(Arc<Reply>) -> Command) -> Pending {
        let reply = Arc::new(Reply::default());
        let sent = self
            .commands
            .as_ref()
            .is_some_and(|commands| commands.send(command(Arc::clone(&reply))).is_ok());
        if !sent {
            reply.complete(Err(codec("Media Foundation encoder worker is not running")));
        }
        Pending {
            reply,
            cancelled: Arc::clone(&self.cancelled),
            done: false,
        }
    }

    fn ready(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(cancelled_error());
        }
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "the HEVC encoder has already been finished",
            ));
        }
        Ok(())
    }

    /// Converts a source frame to the layout the MFT was given.
    fn payload(&self, source: FrameSource<'_>) -> Result<Vec<u8>> {
        let FrameSource::Cpu(source) = source else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Media Foundation HEVC encoder requires a CPU frame source",
            ));
        };
        let frame = source.frame;
        let (width, height) = (self.settings.width, self.settings.height);
        if frame.dimensions.width != width
            || frame.dimensions.height != height
            || frame.pixel_format != self.settings.input_format
            || frame.color_range != ColorRange::Limited
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "HEVC input frame does not match the configured format",
            ));
        }
        let (width, height) = (width as usize, height as usize);
        let row = |plane: usize, y: usize, bytes: usize| {
            let y = match source.orientation {
                Orientation::TopLeft => y,
                Orientation::BottomLeft => {
                    let rows = if plane == 0 { height } else { height / 2 };
                    rows - 1 - y
                }
            };
            let plane = &frame.planes[plane];
            &plane.data[y * plane.stride..][..bytes]
        };
        let mut out = vec![
            0;
            self.feed
                .frame_len(self.settings.width, self.settings.height)
        ];
        match (self.feed, frame.pixel_format) {
            (Feed::Argb32, PixelFormat::Bgra8) => {
                for (y, dst) in out.chunks_exact_mut(width * 4).enumerate() {
                    dst.copy_from_slice(row(0, y, width * 4));
                }
            }
            (Feed::Argb32, PixelFormat::Rgba8) => {
                for (y, dst) in out.chunks_exact_mut(width * 4).enumerate() {
                    swap_red_blue(row(0, y, width * 4), dst);
                }
            }
            (Feed::Nv12, PixelFormat::Yuv420p8) => {
                let (luma, chroma) = out.split_at_mut(width * height);
                for (y, dst) in luma.chunks_exact_mut(width).enumerate() {
                    dst.copy_from_slice(row(0, y, width));
                }
                for (y, dst) in chroma.chunks_exact_mut(width).enumerate() {
                    let (cb, cr) = (row(1, y, width / 2), row(2, y, width / 2));
                    for ((pair, &cb), &cr) in dst.chunks_exact_mut(2).zip(cb).zip(cr) {
                        pair[0] = cb;
                        pair[1] = cr;
                    }
                }
            }
            (Feed::Nv12, format @ (PixelFormat::Rgba8 | PixelFormat::Bgra8)) => {
                let bgra = format == PixelFormat::Bgra8;
                let mut top = vec![0; if bgra { width * 4 } else { 0 }];
                let mut bottom = top.clone();
                let mut cb = vec![0; width / 2];
                let mut cr = vec![0; width / 2];
                let (luma, chroma) = out.split_at_mut(width * height);
                for (pair, dst) in chroma.chunks_exact_mut(width).enumerate() {
                    let (upper, lower) = if bgra {
                        swap_red_blue(row(0, pair * 2, width * 4), &mut top);
                        swap_red_blue(row(0, pair * 2 + 1, width * 4), &mut bottom);
                        (&top[..], &bottom[..])
                    } else {
                        (row(0, pair * 2, width * 4), row(0, pair * 2 + 1, width * 4))
                    };
                    colorconv::luma_row(upper, &mut luma[pair * 2 * width..][..width]);
                    colorconv::luma_row(lower, &mut luma[(pair * 2 + 1) * width..][..width]);
                    colorconv::chroma_row_pair(upper, lower, &mut cb, &mut cr);
                    for ((pair, &cb), &cr) in dst.chunks_exact_mut(2).zip(&cb).zip(&cr) {
                        pair[0] = cb;
                        pair[1] = cr;
                    }
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "Media Foundation HEVC encoder cannot take this pixel format",
                ));
            }
        }
        Ok(out)
    }
}

/// Copies RGBA pixels to BGRA or back; the swap is its own inverse.
fn swap_red_blue(src: &[u8], dst: &mut [u8]) {
    for (from, to) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        to.copy_from_slice(&[from[2], from[1], from[0], from[3]]);
    }
}

impl VideoEncoder for MfHevcEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> VideoEncoderFormat {
        VideoEncoderFormat {
            dimensions: crate::VideoDimensions {
                width: self.settings.width,
                height: self.settings.height,
            },
            pixel_format: self.settings.input_format,
        }
    }

    fn implementation(&self) -> CodecImplementation {
        match self.class {
            MftClass::Hardware => CodecImplementation::Hardware,
            MftClass::Software => CodecImplementation::Software,
        }
    }

    fn backend_name(&self) -> &str {
        &self.name
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        source: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        let prepared = self.ready().and_then(|()| {
            if index.0 != self.next_index {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "HEVC encoder frame indexes must be consecutive and start at zero",
                ));
            }
            self.payload(source)
        });
        match prepared {
            Ok(payload) => {
                self.next_index += 1;
                Box::pin(self.request(|reply| Command::Encode {
                    index: index.0,
                    payload,
                    reply,
                }))
            }
            Err(error) => Box::pin(async move { Err(error) }),
        }
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        if let Err(error) = self.ready() {
            return Box::pin(async move { Err(error) });
        }
        self.finished = true;
        Box::pin(self.request(|reply| Command::Finish { reply }))
    }
}

impl Drop for MfHevcEncoder {
    fn drop(&mut self) {
        // Closing the channel stops the worker once any request it is still
        // on returns, which a cancelled one does within a poll interval.
        self.commands.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_worker(mut core: Core, commands: &Receiver<Command>, cancelled: &AtomicBool) {
    while let Ok(command) = commands.recv() {
        match command {
            Command::Encode {
                index,
                payload,
                reply,
            } => reply.complete(core.encode(index, &payload, cancelled)),
            Command::Finish { reply } => reply.complete(core.finish(cancelled)),
        }
    }
}

struct MfRuntime;

impl MfRuntime {
    fn start() -> Result<Self> {
        // SAFETY: balanced by `Drop`, on the same thread.
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED)
                .ok()
                .map_err(|error| windows_error("could not initialize COM", error))?;
            if let Err(error) = MFStartup(MF_VERSION, 0) {
                CoUninitialize();
                return Err(windows_error(
                    "could not initialize Media Foundation",
                    error,
                ));
            }
        }
        Ok(Self)
    }
}

impl Drop for MfRuntime {
    fn drop(&mut self) {
        // SAFETY: balances `start`, on the same thread.
        unsafe {
            let _ = MFShutdown();
            CoUninitialize();
        }
    }
}

/// HRESULTs that mean the GPU behind a hardware encoder is gone, rather than
/// that the encoder rejected something.
const DEVICE_LOST: [(u32, &str); 6] = [
    (0x887A_0005, "DXGI_ERROR_DEVICE_REMOVED"),
    (0x887A_0006, "DXGI_ERROR_DEVICE_HUNG"),
    (0x887A_0007, "DXGI_ERROR_DEVICE_RESET"),
    (0x887A_0020, "DXGI_ERROR_DRIVER_INTERNAL_ERROR"),
    (0x8876_0868, "D3DERR_DEVICELOST"),
    (0x8876_0870, "D3DDDIERR_DEVICEREMOVED"),
];

/// Classifies a Media Foundation failure while streaming: a lost device is
/// [`ErrorKind::Graphics`], so a caller can tell it apart from a bad frame and
/// knows to create a new encoder; anything else is [`ErrorKind::Codec`].
fn mf_error(context: &str, error: windows::core::Error) -> Error {
    let code = error.code().0 as u32;
    match DEVICE_LOST.iter().find(|(lost, _)| *lost == code) {
        Some((_, name)) => device_lost(format!("{context}: {name} ({error})")),
        None => windows_error(context, error),
    }
}

fn device_lost(detail: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorKind::Graphics,
        format!(
            "the Media Foundation HEVC encoder lost its device ({detail}); \
             create a new encoder to continue"
        ),
    )
}

fn cancelled_error() -> Error {
    Error::new(ErrorKind::Cancelled, "HEVC encoding was cancelled")
}

fn windows_error(context: &str, error: windows::core::Error) -> Error {
    codec(format!("{context}: {error}"))
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CancellationToken, CpuFrameSource, EncodedVideoSample, ExactFrameReader, Plane,
        VideoDecoderConfig, VideoDimensions, VideoFrame, native_hevc_video_decoder_factory,
    };

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut context = Context::from_waker(Waker::noop());
        let mut future = std::pin::pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
            thread::yield_now();
        }
    }

    /// Frame `index` of a moving diagonal gradient with a flat red block, in
    /// `format`. The YUV form is converted with the native encoder's kernels,
    /// so every format describes the same picture.
    fn frame(width: u32, height: u32, index: u64, format: PixelFormat) -> VideoFrame {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits).unwrap();
        let (width, height) = (width as usize, height as usize);
        let shift = index as usize * 4;
        let rgba = |x: usize, y: usize| -> [u8; 4] {
            if (width / 4..width / 2).contains(&x) && (height / 4..height / 2).contains(&y) {
                [200, 60, 40, 255]
            } else {
                [
                    ((x + shift) * 255 / width) as u8,
                    (y * 255 / height) as u8,
                    ((x + y + shift) * 255 / (width + height)) as u8,
                    255,
                ]
            }
        };
        let row = |y: usize| -> Vec<u8> { (0..width).flat_map(|x| rgba(x, y)).collect() };
        let planes = match format {
            PixelFormat::Rgba8 => vec![Plane {
                data: (0..height).flat_map(row).collect(),
                stride: width * 4,
            }],
            PixelFormat::Bgra8 => {
                let mut data: Vec<u8> = (0..height).flat_map(row).collect();
                data.chunks_exact_mut(4).for_each(|pixel| pixel.swap(0, 2));
                vec![Plane {
                    data,
                    stride: width * 4,
                }]
            }
            PixelFormat::Yuv420p8 => {
                let mut luma = vec![0; width * height];
                for (y, out) in luma.chunks_exact_mut(width).enumerate() {
                    colorconv::luma_row(&row(y), out);
                }
                let mut cb = vec![0; width / 2 * height / 2];
                let mut cr = cb.clone();
                for pair in 0..height / 2 {
                    colorconv::chroma_row_pair(
                        &row(pair * 2),
                        &row(pair * 2 + 1),
                        &mut cb[pair * width / 2..][..width / 2],
                        &mut cr[pair * width / 2..][..width / 2],
                    );
                }
                vec![
                    Plane {
                        data: luma,
                        stride: width,
                    },
                    Plane {
                        data: cb,
                        stride: width / 2,
                    },
                    Plane {
                        data: cr,
                        stride: width / 2,
                    },
                ]
            }
            _ => unreachable!("not an input this backend takes"),
        };
        VideoFrame::new(dimensions, format, ColorRange::Limited, planes, &limits).unwrap()
    }

    fn psnr(reference: &VideoFrame, decoded: &VideoFrame) -> f64 {
        let (a, b) = (&reference.planes[0].data, &decoded.planes[0].data);
        let mut squared = 0.0;
        let mut count = 0.0;
        for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            for channel in 0..3 {
                squared += (f64::from(a[channel]) - f64::from(b[channel])).powi(2);
                count += 1.0;
            }
        }
        10.0 * (255.0_f64.powi(2) / (squared / count).max(1e-9)).log10()
    }

    fn round_trip_settings(input_format: PixelFormat) -> Settings {
        Settings {
            width: 320,
            height: 240,
            input_format,
            timescale: 30_000,
            frame_duration: 1_001,
            bits_per_second: 4_000_000,
            keyframe_interval: 5,
        }
    }

    /// Every class this host has, fed every input format, encodes a stream
    /// the crate's own software decoder reads back close to the source, with
    /// a keyframe exactly where the interval puts one and the constant-rate
    /// clock on every sample. A class the host does not have is skipped,
    /// which on a CI runner without a GPU is the hardware one and, without
    /// the HEVC Video Extensions, the software one too.
    #[test]
    fn each_media_foundation_class_round_trips_through_the_native_decoder() {
        const FRAMES: u64 = 12;
        let limits = Limits::default();
        for class in [MftClass::Hardware, MftClass::Software] {
            for input_format in [
                PixelFormat::Rgba8,
                PixelFormat::Bgra8,
                PixelFormat::Yuv420p8,
            ] {
                let settings = round_trip_settings(input_format);
                let mut encoder = match create(settings, class, &limits) {
                    Ok(encoder) => encoder,
                    Err(error) => {
                        eprintln!("skipping {class:?} {input_format:?}: {error}");
                        continue;
                    }
                };
                let label = format!("{class:?} {input_format:?} via {}", encoder.backend_name());
                assert_eq!(
                    encoder.implementation(),
                    match class {
                        MftClass::Hardware => CodecImplementation::Hardware,
                        MftClass::Software => CodecImplementation::Software,
                    }
                );
                let mut samples = Vec::new();
                for index in 0..FRAMES {
                    let source = frame(320, 240, index, input_format);
                    samples.extend(
                        block_on(encoder.encode(
                            FrameIndex(index),
                            FrameSource::Cpu(CpuFrameSource {
                                frame: &source,
                                orientation: Orientation::TopLeft,
                            }),
                        ))
                        .unwrap(),
                    );
                }
                samples.extend(block_on(encoder.finish()).unwrap());
                assert_eq!(samples.len() as u64, FRAMES, "{label}");
                for (index, sample) in samples.iter().enumerate() {
                    let tick = index as i64 * 1_001;
                    assert_eq!((sample.dts, sample.pts), (tick, tick), "{label}");
                    assert_eq!(sample.duration, 1_001, "{label}");
                    assert_eq!(sample.is_sync, index % 5 == 0, "{label} frame {index}");
                }

                let decoder_config = VideoDecoderConfig {
                    codec: crate::Codec::Hevc,
                    profile: crate::CodecProfile::HevcMain,
                    coded_dimensions: VideoDimensions::new(320, 240, &limits).unwrap(),
                    output_format: PixelFormat::Rgba8,
                    color_range: ColorRange::Limited,
                    hardware: crate::HardwarePreference::Avoid,
                    configuration: encoder.config().decoder_config.clone(),
                };
                let samples = samples
                    .into_iter()
                    .enumerate()
                    .map(|(index, sample)| EncodedVideoSample {
                        presentation_index: FrameIndex(index as u64),
                        random_access: sample.is_sync,
                        data: sample.data,
                    })
                    .collect();
                let mut reader = ExactFrameReader::new(
                    &native_hevc_video_decoder_factory(),
                    decoder_config,
                    samples,
                    limits,
                )
                .unwrap();
                let mut worst = f64::INFINITY;
                for index in 0..FRAMES {
                    let decoded = reader
                        .get(FrameIndex(index), &CancellationToken::new())
                        .unwrap();
                    let reference = frame(320, 240, index, PixelFormat::Rgba8);
                    worst = worst.min(psnr(&reference, &decoded));
                }
                eprintln!("{label}: worst PSNR {worst:.1} dB");
                assert!(worst > 32.0, "{label}: worst PSNR {worst:.1} dB");
            }
        }
    }

    fn settings(timescale: u32, frame_duration: u32) -> Settings {
        Settings {
            width: 64,
            height: 64,
            input_format: PixelFormat::Rgba8,
            timescale,
            frame_duration,
            bits_per_second: 1_000_000,
            keyframe_interval: 30,
        }
    }

    #[test]
    fn frame_times_round_trip_through_media_foundation_units() {
        for (timescale, frame_duration) in [(30, 1), (30_000, 1_001), (90_000, 3_003), (600, 25)] {
            let settings = settings(timescale, frame_duration);
            for index in [0, 1, 2, 29, 30, 1_000, 123_456] {
                assert_eq!(
                    settings.frame_at(settings.sample_time(index)),
                    index,
                    "{timescale}/{frame_duration} frame {index}"
                );
            }
        }
        assert_eq!(settings(30_000, 1_001).frame_rate(), (30_000 << 32) | 1_001);
        assert_eq!(settings(60, 2).frame_rate(), (30 << 32) | 1);
    }

    #[test]
    fn settings_reject_what_media_foundation_cannot_take() {
        assert_eq!(settings(30, 1).unsupported_reason(), None);
        let mut odd = settings(30, 1);
        odd.width = 63;
        assert!(odd.unsupported_reason().is_some());
        let mut gray = settings(30, 1);
        gray.input_format = PixelFormat::Gray8;
        assert!(gray.unsupported_reason().is_some());
        assert!(settings(10_000_000, 1).unsupported_reason().is_some());
    }

    #[test]
    fn a_lost_device_is_a_graphics_error_and_anything_else_is_a_codec_error() {
        for (code, name) in DEVICE_LOST {
            let error = mf_error(
                "encoder rejected a frame",
                windows::core::Error::from_hresult(HRESULT(code as i32)),
            );
            assert_eq!(error.kind(), ErrorKind::Graphics, "{name}");
            assert!(error.message().contains(name), "{}", error.message());
            assert!(error.message().contains("create a new encoder"));
        }
        let error = mf_error(
            "encoder rejected a frame",
            windows::core::Error::from_hresult(MF_E_NOTACCEPTING),
        );
        assert_eq!(error.kind(), ErrorKind::Codec);
    }

    #[test]
    fn rgb_input_is_swizzled_and_nv12_interleaves_chroma() {
        let mut dst = [0; 8];
        swap_red_blue(&[1, 2, 3, 4, 5, 6, 7, 8], &mut dst);
        assert_eq!(dst, [3, 2, 1, 4, 7, 6, 5, 8]);
        assert_eq!(Feed::Nv12.frame_len(64, 32), 64 * 32 * 3 / 2);
        assert_eq!(Feed::Argb32.stride(64), 256);
    }
}
