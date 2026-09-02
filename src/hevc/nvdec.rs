//! NVIDIA NVDEC HEVC backend for Windows and Linux, loaded from the display driver at runtime.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use libloading::Library;

use super::engine::hvcc::{HvccRecord, split_length_prefixed};
use super::readback;
use crate::{
    CancellationToken, DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex, Limits,
    PixelFormat, Plane, Result, VideoDecoder, VideoDecoderConfig, VideoDimensions, VideoFrame,
};

const CUDA_SUCCESS: i32 = 0;
const CUDA_VIDEO_CODEC_HEVC: u32 = 8;
const CUDA_VIDEO_CHROMA_420: u32 = 1;
const CUDA_VIDEO_SURFACE_NV12: u32 = 0;
const CUDA_VIDEO_DEINTERLACE_WEAVE: u32 = 0;
const CUDA_VIDEO_CREATE_PREFER_CUVID: u32 = 4;
const CUVID_PKT_ENDOFSTREAM: u32 = 0x01;
const CUVID_PKT_TIMESTAMP: u32 = 0x02;
const CUVID_PKT_ENDOFPICTURE: u32 = 0x08;

#[cfg(windows)]
const CUDA_DRIVER_LIBRARY: &str = "nvcuda.dll";
#[cfg(windows)]
const NVDEC_DRIVER_LIBRARY: &str = "nvcuvid.dll";
#[cfg(target_os = "linux")]
const CUDA_DRIVER_LIBRARY: &str = "libcuda.so.1";
#[cfg(target_os = "linux")]
const NVDEC_DRIVER_LIBRARY: &str = "libnvcuvid.so.1";

type CuInit = unsafe extern "system" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> i32;
type CuCtxCreate = unsafe extern "system" fn(*mut *mut c_void, u32, c_int) -> i32;
type CuCtxDestroy = unsafe extern "system" fn(*mut c_void) -> i32;
type CuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> i32;
type CuGetErrorName = unsafe extern "system" fn(i32, *mut *const c_char) -> i32;
type CuvidGetDecoderCaps = unsafe extern "system" fn(*mut CuvidDecodeCaps) -> i32;
type CuvidCreateDecoder =
    unsafe extern "system" fn(*mut *mut c_void, *mut CuvidDecodeCreateInfo) -> i32;
type CuvidDestroyDecoder = unsafe extern "system" fn(*mut c_void) -> i32;
type CuvidDecodePicture = unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32;
type CuvidMapVideoFrame =
    unsafe extern "system" fn(*mut c_void, c_int, *mut u64, *mut u32, *mut CuvidProcParams) -> i32;
type CuvidUnmapVideoFrame = unsafe extern "system" fn(*mut c_void, u64) -> i32;
type CuvidCreateVideoParser =
    unsafe extern "system" fn(*mut *mut c_void, *mut CuvidParserParams) -> i32;
type CuvidParseVideoData =
    unsafe extern "system" fn(*mut c_void, *mut CuvidSourceDataPacket) -> i32;
type CuvidDestroyVideoParser = unsafe extern "system" fn(*mut c_void) -> i32;

pub(super) fn is_available(dimensions: VideoDimensions) -> bool {
    let (ready_tx, ready_rx) = sync_channel(1);
    if thread::Builder::new()
        .name("zvidlib-nvdec-probe".into())
        .spawn(move || {
            let result = NvRuntime::start().and_then(|runtime| runtime.require_hevc(dimensions));
            let _ = ready_tx.send(result);
        })
        .is_err()
    {
        return false;
    }
    matches!(ready_rx.recv(), Ok(Ok(())))
}

pub(super) fn create(
    configuration: &VideoDecoderConfig,
    limits: &Limits,
    record: &HvccRecord,
) -> Result<Box<dyn VideoDecoder>> {
    NvDecoder::spawn(
        configuration.clone(),
        *limits,
        record.length_size,
        annex_b_parameter_sets(record),
    )
    .map(|decoder| Box::new(decoder) as Box<dyn VideoDecoder>)
}

enum Command {
    Submit {
        sample: EncodedVideoSample,
        cancellation: CancellationToken,
        response: SyncSender<Result<Vec<DecodedVideoFrame>>>,
    },
    Drain {
        cancellation: CancellationToken,
        response: SyncSender<Result<Vec<DecodedVideoFrame>>>,
    },
    Reset {
        response: SyncSender<Result<Vec<DecodedVideoFrame>>>,
    },
    SetOutputWanted {
        wanted: bool,
        response: SyncSender<Result<Vec<DecodedVideoFrame>>>,
    },
    Stop,
}

struct NvDecoder {
    commands: SyncSender<Command>,
    worker: Option<JoinHandle<()>>,
}

impl NvDecoder {
    fn spawn(
        configuration: VideoDecoderConfig,
        limits: Limits,
        nal_length_size: usize,
        parameter_sets: Vec<u8>,
    ) -> Result<Self> {
        let (command_tx, command_rx) = sync_channel(1);
        let (ready_tx, ready_rx) = sync_channel(1);
        let worker = thread::Builder::new()
            .name("zvidlib-nvdec-hevc".into())
            .spawn(move || {
                match NvDecoderCore::new(configuration, limits, nal_length_size, parameter_sets) {
                    Ok(core) => {
                        if ready_tx.send(Ok(())).is_ok() {
                            run_worker(core, command_rx);
                        }
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })
            .map_err(|error| codec(format!("could not start NVDEC worker: {error}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                commands: command_tx,
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(codec("NVDEC worker stopped during initialization"))
            }
        }
    }

    fn request(
        &self,
        make: impl FnOnce(SyncSender<Result<Vec<DecodedVideoFrame>>>) -> Command,
    ) -> Result<Vec<DecodedVideoFrame>> {
        let (response_tx, response_rx) = sync_channel(1);
        self.commands
            .send(make(response_tx))
            .map_err(|_| codec("NVDEC worker is not running"))?;
        response_rx
            .recv()
            .map_err(|_| codec("NVDEC worker stopped unexpectedly"))?
    }
}

impl VideoDecoder for NvDecoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        self.request(|response| Command::Submit {
            sample: sample.clone(),
            cancellation: cancellation.clone(),
            response,
        })
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        self.request(|response| Command::Drain {
            cancellation: cancellation.clone(),
            response,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.request(|response| Command::Reset { response })
            .map(|_| ())
    }

    /// Forwards the hint to the worker and waits for it, so that a sample submitted after this
    /// returns is decoded under the setting the caller asked for. The command channel is FIFO,
    /// so the wait is not what orders it; it is what keeps a dropped worker visible here rather
    /// than silently leaving the decoder converting frames nobody wants.
    fn set_output_wanted(&mut self, wanted: bool) {
        let _ = self.request(|response| Command::SetOutputWanted { wanted, response });
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_worker(mut core: NvDecoderCore, commands: Receiver<Command>) {
    while let Ok(command) = commands.recv() {
        match command {
            Command::Submit {
                sample,
                cancellation,
                response,
            } => {
                let _ = response.send(core.submit(&sample, &cancellation));
            }
            Command::Drain {
                cancellation,
                response,
            } => {
                let _ = response.send(core.drain(&cancellation));
            }
            Command::Reset { response } => {
                let _ = response.send(core.reset().map(|()| Vec::new()));
            }
            Command::SetOutputWanted { wanted, response } => {
                core.set_output_wanted(wanted);
                let _ = response.send(Ok(Vec::new()));
            }
            Command::Stop => return,
        }
    }
}

struct NvDecoderCore {
    runtime: NvRuntime,
    parser: *mut c_void,
    callback_state: Box<CallbackState>,
    configuration: VideoDecoderConfig,
    limits: Limits,
    nal_length_size: usize,
    parameter_sets: Vec<u8>,
    presentation_indexes: BinaryHeap<Reverse<FrameIndex>>,
    next_timestamp: i64,
}

impl NvDecoderCore {
    fn new(
        configuration: VideoDecoderConfig,
        limits: Limits,
        nal_length_size: usize,
        parameter_sets: Vec<u8>,
    ) -> Result<Self> {
        let runtime = NvRuntime::start()?;
        runtime.require_hevc(configuration.coded_dimensions)?;
        let callback_api = CallbackApi::from_api(&runtime.api);
        let mut callback_state = Box::new(CallbackState {
            api: callback_api,
            decoder: ptr::null_mut(),
            dimensions: configuration.coded_dimensions,
            max_allocation_bytes: limits.max_allocation_bytes,
            output_wanted: true,
            frames: Vec::new(),
            error: None,
        });
        let parser = create_parser(&runtime.api, &mut callback_state)?;
        Ok(Self {
            runtime,
            parser,
            callback_state,
            configuration,
            limits,
            nal_length_size,
            parameter_sets,
            presentation_indexes: BinaryHeap::new(),
            next_timestamp: 1,
        })
    }

    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        let data = annex_b_sample(
            sample,
            self.nal_length_size,
            &self.parameter_sets,
            self.limits.max_allocation_bytes,
        )?;
        let timestamp = self.next_timestamp;
        self.next_timestamp = self
            .next_timestamp
            .checked_add(1)
            .ok_or_else(|| codec("NVDEC timestamp identity overflow"))?;
        self.presentation_indexes
            .push(Reverse(sample.presentation_index));
        let mut packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE,
            payload_size: u32::try_from(data.len()).map_err(|_| {
                Error::new(ErrorKind::ResourceLimit, "HEVC access unit is too large")
            })?,
            payload: data.as_ptr(),
            timestamp,
        };
        self.parse(&mut packet)?;
        check_cancelled(cancellation)?;
        self.take_frames()
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        let mut packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_ENDOFSTREAM,
            ..CuvidSourceDataPacket::zeroed()
        };
        self.parse(&mut packet)?;
        check_cancelled(cancellation)?;
        self.take_frames()
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        self.callback_state.output_wanted = wanted;
    }

    fn reset(&mut self) -> Result<()> {
        self.destroy_parser_and_decoder();
        self.callback_state.frames.clear();
        self.callback_state.error = None;
        self.parser = create_parser(&self.runtime.api, &mut self.callback_state)?;
        self.presentation_indexes.clear();
        self.next_timestamp = 1;
        Ok(())
    }

    fn parse(&mut self, packet: &mut CuvidSourceDataPacket) -> Result<()> {
        let result = unsafe { (self.runtime.api.cuvid_parse_video_data)(self.parser, packet) };
        if let Some(message) = self.callback_state.error.take() {
            return Err(codec(message));
        }
        self.runtime
            .api
            .check(result, "NVDEC could not parse HEVC data")
    }

    fn take_frames(&mut self) -> Result<Vec<DecodedVideoFrame>> {
        let raw_frames = std::mem::take(&mut self.callback_state.frames);
        raw_frames
            .into_iter()
            .filter_map(|raw| {
                // Popped for a suppressed picture too: it was displayed, so it owns the identity
                // at the front of the heap whether or not a frame is built from it.
                let presentation_index = match self.presentation_indexes.pop() {
                    Some(Reverse(index)) => index,
                    None => {
                        return Some(Err(codec(
                            "NVDEC produced a frame without a submitted identity",
                        )));
                    }
                };
                let raw = raw?;
                Some(
                    nv12_to_rgba(raw, &self.configuration, &self.limits).map(|frame| {
                        DecodedVideoFrame {
                            presentation_index,
                            frame,
                        }
                    }),
                )
            })
            .collect()
    }

    fn destroy_parser_and_decoder(&mut self) {
        if !self.parser.is_null() {
            unsafe { (self.runtime.api.cuvid_destroy_video_parser)(self.parser) };
            self.parser = ptr::null_mut();
        }
        if !self.callback_state.decoder.is_null() {
            unsafe { (self.runtime.api.cuvid_destroy_decoder)(self.callback_state.decoder) };
            self.callback_state.decoder = ptr::null_mut();
        }
    }
}

impl Drop for NvDecoderCore {
    fn drop(&mut self) {
        self.destroy_parser_and_decoder();
    }
}

struct NvRuntime {
    api: NvApi,
    context: *mut c_void,
}

impl NvRuntime {
    fn start() -> Result<Self> {
        let api = NvApi::load()?;
        api.check(unsafe { (api.cu_init)(0) }, "could not initialize CUDA")?;
        let mut device = 0;
        api.check(
            unsafe { (api.cu_device_get)(&mut device, 0) },
            "could not select CUDA device",
        )?;
        let mut context = ptr::null_mut();
        api.check(
            unsafe { (api.cu_ctx_create)(&mut context, 0, device) },
            "could not create CUDA context",
        )?;
        Ok(Self { api, context })
    }

    fn require_hevc(&self, dimensions: VideoDimensions) -> Result<()> {
        let mut caps = CuvidDecodeCaps {
            codec_type: CUDA_VIDEO_CODEC_HEVC,
            chroma_format: CUDA_VIDEO_CHROMA_420,
            bit_depth_minus8: 0,
            ..CuvidDecodeCaps::zeroed()
        };
        self.api.check(
            unsafe { (self.api.cuvid_get_decoder_caps)(&mut caps) },
            "could not query NVDEC HEVC capabilities",
        )?;
        let macroblocks =
            u64::from(dimensions.width.div_ceil(16)) * u64::from(dimensions.height.div_ceil(16));
        if caps.is_supported == 0
            || dimensions.width < u32::from(caps.min_width)
            || dimensions.height < u32::from(caps.min_height)
            || dimensions.width > caps.max_width
            || dimensions.height > caps.max_height
            || macroblocks > u64::from(caps.max_mb_count)
        {
            return Err(unsupported(
                "NVIDIA adapter does not support this HEVC Main configuration",
            ));
        }
        Ok(())
    }
}

impl Drop for NvRuntime {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe { (self.api.cu_ctx_destroy)(self.context) };
        }
    }
}

struct NvApi {
    _cuda: Library,
    _cuvid: Library,
    cu_init: CuInit,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_get_error_name: CuGetErrorName,
    cuvid_get_decoder_caps: CuvidGetDecoderCaps,
    cuvid_create_decoder: CuvidCreateDecoder,
    cuvid_destroy_decoder: CuvidDestroyDecoder,
    cuvid_decode_picture: CuvidDecodePicture,
    cuvid_map_video_frame: CuvidMapVideoFrame,
    cuvid_unmap_video_frame: CuvidUnmapVideoFrame,
    cuvid_create_video_parser: CuvidCreateVideoParser,
    cuvid_parse_video_data: CuvidParseVideoData,
    cuvid_destroy_video_parser: CuvidDestroyVideoParser,
}

impl NvApi {
    fn load() -> Result<Self> {
        let cuda = unsafe { Library::new(CUDA_DRIVER_LIBRARY) }
            .map_err(|error| unsupported(format!("NVIDIA CUDA driver is unavailable: {error}")))?;
        let cuvid = unsafe { Library::new(NVDEC_DRIVER_LIBRARY) }
            .map_err(|error| unsupported(format!("NVIDIA NVDEC driver is unavailable: {error}")))?;
        unsafe {
            Ok(Self {
                cu_init: load_symbol(&cuda, b"cuInit\0")?,
                cu_device_get: load_symbol(&cuda, b"cuDeviceGet\0")?,
                cu_ctx_create: load_symbol(&cuda, b"cuCtxCreate_v2\0")?,
                cu_ctx_destroy: load_symbol(&cuda, b"cuCtxDestroy_v2\0")?,
                cu_memcpy_dtoh: load_symbol(&cuda, b"cuMemcpyDtoH_v2\0")?,
                cu_get_error_name: load_symbol(&cuda, b"cuGetErrorName\0")?,
                cuvid_get_decoder_caps: load_symbol(&cuvid, b"cuvidGetDecoderCaps\0")?,
                cuvid_create_decoder: load_symbol(&cuvid, b"cuvidCreateDecoder\0")?,
                cuvid_destroy_decoder: load_symbol(&cuvid, b"cuvidDestroyDecoder\0")?,
                cuvid_decode_picture: load_symbol(&cuvid, b"cuvidDecodePicture\0")?,
                cuvid_map_video_frame: load_symbol(&cuvid, b"cuvidMapVideoFrame64\0")?,
                cuvid_unmap_video_frame: load_symbol(&cuvid, b"cuvidUnmapVideoFrame64\0")?,
                cuvid_create_video_parser: load_symbol(&cuvid, b"cuvidCreateVideoParser\0")?,
                cuvid_parse_video_data: load_symbol(&cuvid, b"cuvidParseVideoData\0")?,
                cuvid_destroy_video_parser: load_symbol(&cuvid, b"cuvidDestroyVideoParser\0")?,
                _cuda: cuda,
                _cuvid: cuvid,
            })
        }
    }

    fn check(&self, result: i32, context: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name = ptr::null();
        let named = unsafe { (self.cu_get_error_name)(result, &mut name) } == CUDA_SUCCESS
            && !name.is_null();
        let detail = if named {
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA error {result}")
        };
        Err(codec(format!("{context}: {detail}")))
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Result<T> {
    unsafe { library.get::<T>(symbol) }
        .map(|value| *value)
        .map_err(|error| unsupported(format!("NVIDIA driver symbol is unavailable: {error}")))
}

#[derive(Clone, Copy)]
struct CallbackApi {
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cuvid_create_decoder: CuvidCreateDecoder,
    cuvid_decode_picture: CuvidDecodePicture,
    cuvid_map_video_frame: CuvidMapVideoFrame,
    cuvid_unmap_video_frame: CuvidUnmapVideoFrame,
}

impl CallbackApi {
    fn from_api(api: &NvApi) -> Self {
        Self {
            cu_memcpy_dtoh: api.cu_memcpy_dtoh,
            cuvid_create_decoder: api.cuvid_create_decoder,
            cuvid_decode_picture: api.cuvid_decode_picture,
            cuvid_map_video_frame: api.cuvid_map_video_frame,
            cuvid_unmap_video_frame: api.cuvid_unmap_video_frame,
        }
    }
}

struct CallbackState {
    api: CallbackApi,
    decoder: *mut c_void,
    dimensions: VideoDimensions,
    max_allocation_bytes: u64,
    /// Whether the pictures arriving at the display callback are wanted as frames. A suppressed
    /// picture is still decoded, and still reaches the callback; what it skips is the readback.
    output_wanted: bool,
    /// One entry per displayed picture, in display order. A suppressed picture leaves `None`,
    /// which `take_frames` needs in order to consume that picture's presentation identity: the
    /// identities are popped one per displayed picture, so dropping an entry outright would
    /// hand every later frame the index of the one before it.
    frames: Vec<Option<RawNv12Frame>>,
    error: Option<String>,
}

struct RawNv12Frame {
    pitch: usize,
    data: Vec<u8>,
}

fn create_parser(api: &NvApi, state: &mut Box<CallbackState>) -> Result<*mut c_void> {
    let mut parser = ptr::null_mut();
    let mut params = CuvidParserParams {
        codec_type: CUDA_VIDEO_CODEC_HEVC,
        max_num_decode_surfaces: 20,
        clock_rate: 10_000_000,
        max_display_delay: 4,
        user_data: (&mut **state as *mut CallbackState).cast(),
        sequence_callback: Some(sequence_callback),
        decode_callback: Some(decode_callback),
        display_callback: Some(display_callback),
        ..CuvidParserParams::zeroed()
    };
    api.check(
        unsafe { (api.cuvid_create_video_parser)(&mut parser, &mut params) },
        "could not create NVDEC HEVC parser",
    )?;
    if parser.is_null() {
        return Err(codec("NVDEC HEVC parser was not returned"));
    }
    Ok(parser)
}

unsafe extern "system" fn sequence_callback(
    user: *mut c_void,
    format: *mut CuVideoFormat,
) -> c_int {
    let state = unsafe { &mut *user.cast::<CallbackState>() };
    let result = (|| {
        let format = unsafe { format.as_ref() }.ok_or("NVDEC sequence format was null")?;
        if format.codec != CUDA_VIDEO_CODEC_HEVC
            || format.chroma_format != CUDA_VIDEO_CHROMA_420
            || format.bit_depth_luma_minus8 != 0
            || format.bit_depth_chroma_minus8 != 0
        {
            return Err("NVDEC sequence is not 8-bit HEVC Main 4:2:0".to_owned());
        }
        if format.display_area.right - format.display_area.left != state.dimensions.width as i32
            || format.display_area.bottom - format.display_area.top
                != state.dimensions.height as i32
        {
            return Err(
                "NVDEC sequence dimensions differ from the decoder configuration".to_owned(),
            );
        }
        if !state.decoder.is_null() {
            return Ok(());
        }
        let mut info = CuvidDecodeCreateInfo {
            width: format.coded_width,
            height: format.coded_height,
            num_decode_surfaces: 20,
            codec_type: CUDA_VIDEO_CODEC_HEVC,
            chroma_format: CUDA_VIDEO_CHROMA_420,
            creation_flags: CUDA_VIDEO_CREATE_PREFER_CUVID,
            max_width: format.coded_width,
            max_height: format.coded_height,
            display_area: CuvidRect {
                left: i16::try_from(format.display_area.left)
                    .map_err(|_| "display left overflows")?,
                top: i16::try_from(format.display_area.top).map_err(|_| "display top overflows")?,
                right: i16::try_from(format.display_area.right)
                    .map_err(|_| "display right overflows")?,
                bottom: i16::try_from(format.display_area.bottom)
                    .map_err(|_| "display bottom overflows")?,
            },
            output_format: CUDA_VIDEO_SURFACE_NV12,
            deinterlace_mode: CUDA_VIDEO_DEINTERLACE_WEAVE,
            target_width: state.dimensions.width,
            target_height: state.dimensions.height,
            num_output_surfaces: 2,
            target_rect: CuvidRect {
                left: 0,
                top: 0,
                right: i16::try_from(state.dimensions.width)
                    .map_err(|_| "target width overflows")?,
                bottom: i16::try_from(state.dimensions.height)
                    .map_err(|_| "target height overflows")?,
            },
            ..CuvidDecodeCreateInfo::zeroed()
        };
        let result = unsafe { (state.api.cuvid_create_decoder)(&mut state.decoder, &mut info) };
        if result != CUDA_SUCCESS || state.decoder.is_null() {
            return Err(format!(
                "NVDEC could not create HEVC decoder: CUDA error {result}"
            ));
        }
        Ok(())
    })();
    match result {
        Ok(()) => 1,
        Err(message) => {
            state.error = Some(message.to_string());
            0
        }
    }
}

unsafe extern "system" fn decode_callback(user: *mut c_void, picture: *mut c_void) -> c_int {
    let state = unsafe { &mut *user.cast::<CallbackState>() };
    if state.decoder.is_null() || picture.is_null() {
        state.error = Some("NVDEC decode callback received invalid state".into());
        return 0;
    }
    let result = unsafe { (state.api.cuvid_decode_picture)(state.decoder, picture) };
    if result == CUDA_SUCCESS {
        1
    } else {
        state.error = Some(format!(
            "NVDEC could not decode HEVC picture: CUDA error {result}"
        ));
        0
    }
}

unsafe extern "system" fn display_callback(
    user: *mut c_void,
    display: *mut CuvidParserDispInfo,
) -> c_int {
    let state = unsafe { &mut *user.cast::<CallbackState>() };
    let result = (|| {
        let display = unsafe { display.as_ref() }.ok_or("NVDEC display info was null")?;
        if state.decoder.is_null() {
            return Err("NVDEC display callback has no decoder".to_owned());
        }
        // The picture was decoded and remains a reference for the frames after it; skipping the
        // map/copy-back/convert below is the whole saving. The placeholder still records that a
        // picture was displayed, which is what keeps the presentation identities aligned.
        if !state.output_wanted {
            state.frames.push(None);
            return Ok(());
        }
        let mut processing = CuvidProcParams {
            progressive_frame: display.progressive_frame,
            second_field: 0,
            top_field_first: display.top_field_first,
            unpaired_field: i32::from(display.repeat_first_field < 0),
            ..CuvidProcParams::zeroed()
        };
        let mut device_pointer = 0_u64;
        let mut pitch = 0_u32;
        // NVDEC's surface-copy phase is the whole map/`cuMemcpyDtoH`/unmap
        // sequence: on a discrete GPU this is the PCIe transfer, and it is the
        // half of readback the colour conversion below cannot account for.
        let surface_copy = readback::Timer::start();
        let mapped = unsafe {
            (state.api.cuvid_map_video_frame)(
                state.decoder,
                display.picture_index,
                &mut device_pointer,
                &mut pitch,
                &mut processing,
            )
        };
        if mapped != CUDA_SUCCESS {
            return Err(format!(
                "NVDEC could not map HEVC frame: CUDA error {mapped}"
            ));
        }
        let byte_count = usize::try_from(pitch)
            .ok()
            .and_then(|pitch| pitch.checked_mul(state.dimensions.height as usize))
            .and_then(|luma| luma.checked_add(luma / 2))
            .ok_or_else(|| "NVDEC frame size overflows".to_owned())?;
        if byte_count as u64 > state.max_allocation_bytes {
            unsafe { (state.api.cuvid_unmap_video_frame)(state.decoder, device_pointer) };
            return Err("NVDEC frame exceeds the allocation limit".to_owned());
        }
        let mut data = vec![0_u8; byte_count];
        let copied = unsafe {
            (state.api.cu_memcpy_dtoh)(data.as_mut_ptr().cast(), device_pointer, byte_count)
        };
        let unmapped =
            unsafe { (state.api.cuvid_unmap_video_frame)(state.decoder, device_pointer) };
        if copied != CUDA_SUCCESS {
            return Err(format!("NVDEC frame readback failed: CUDA error {copied}"));
        }
        if unmapped != CUDA_SUCCESS {
            return Err(format!("NVDEC frame unmap failed: CUDA error {unmapped}"));
        }
        surface_copy.record(readback::Phase::SurfaceCopy);
        state.frames.push(Some(RawNv12Frame {
            pitch: pitch as usize,
            data,
        }));
        Ok(())
    })();
    match result {
        Ok(()) => 1,
        Err(message) => {
            state.error = Some(message.to_string());
            0
        }
    }
}

fn nv12_to_rgba(
    raw: RawNv12Frame,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    let width = configuration.coded_dimensions.width as usize;
    let height = configuration.coded_dimensions.height as usize;
    if raw.pitch < width {
        return Err(codec("NVDEC output pitch is smaller than the frame width"));
    }
    let required = raw
        .pitch
        .checked_mul(height)
        .and_then(|luma| luma.checked_add(luma / 2))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "NVDEC output size overflows"))?;
    if raw.data.len() < required {
        return Err(codec("NVDEC output buffer is truncated"));
    }
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC RGBA stride overflows"))?;
    let length = stride
        .checked_mul(height)
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC RGBA allocation overflows"))?;
    if length as u64 > limits.max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "HEVC RGBA output exceeds the allocation limit",
        ));
    }
    let color_convert = readback::Timer::start();
    let mut rgba = vec![0_u8; length];
    for y in 0..height {
        let luma = &raw.data[y * raw.pitch..y * raw.pitch + width];
        let chroma_start = (height + y / 2) * raw.pitch;
        let chroma = &raw.data[chroma_start..chroma_start + width];
        for x in 0..width {
            let y_scaled = i32::from(luma[x]) * 8 - 128;
            let u_scaled = i32::from(chroma[x & !1]) * 8 - 1024;
            let v_scaled = i32::from(chroma[(x & !1) + 1]) * 8 - 1024;
            let y_term = multiply_high(y_scaled, 9_539);
            let at = y * stride + x * 4;
            rgba[at] = clip_u8(y_term.saturating_add(multiply_high(v_scaled, 13_075)));
            rgba[at + 1] = clip_u8(
                y_term
                    .saturating_add(multiply_high(u_scaled, -3_209))
                    .saturating_add(multiply_high(v_scaled, -6_660)),
            );
            rgba[at + 2] = clip_u8(y_term.saturating_add(multiply_high(u_scaled, 16_525)));
            rgba[at + 3] = 255;
        }
    }
    color_convert.record(readback::Phase::ColorConvert);
    readback::count_frame();
    VideoFrame::new(
        configuration.coded_dimensions,
        PixelFormat::Rgba8,
        configuration.color_range,
        vec![Plane { data: rgba, stride }],
        limits,
    )
}

fn annex_b_parameter_sets(record: &HvccRecord) -> Vec<u8> {
    let mut output = Vec::new();
    for unit in &record.nal_units {
        append_annex_b_unit(&mut output, unit);
    }
    output
}

fn annex_b_sample(
    sample: &EncodedVideoSample,
    nal_length_size: usize,
    parameter_sets: &[u8],
    max_allocation_bytes: u64,
) -> Result<Vec<u8>> {
    let units = split_length_prefixed(&sample.data, nal_length_size).map_err(|error| {
        Error::new(
            ErrorKind::MalformedMedia,
            format!("invalid HEVC access unit: {error}"),
        )
    })?;
    let estimated = sample
        .data
        .len()
        .checked_add(if sample.random_access {
            parameter_sets.len()
        } else {
            0
        })
        .and_then(|size| size.checked_add(units.len() * 4))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC access unit size overflows"))?;
    if estimated as u64 > max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "HEVC access unit exceeds the allocation limit",
        ));
    }
    let mut output = Vec::with_capacity(estimated);
    if sample.random_access {
        output.extend_from_slice(parameter_sets);
    }
    for unit in &units {
        append_annex_b_unit(&mut output, unit);
    }
    Ok(output)
}

fn append_annex_b_unit(output: &mut Vec<u8>, unit: &super::engine::nal::NalUnit) {
    output.extend_from_slice(&[0, 0, 0, 1]);
    output.push((unit.header.nal_unit_type << 1) | (unit.header.nuh_layer_id >> 5));
    output.push(((unit.header.nuh_layer_id & 0x1f) << 3) | (unit.header.temporal_id + 1));
    output.extend_from_slice(&unit.escaped);
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

fn multiply_high(left: i32, right: i32) -> i32 {
    left.saturating_mul(right) >> 16
}

fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

#[repr(C)]
struct CuvidDecodeCaps {
    codec_type: u32,
    chroma_format: u32,
    bit_depth_minus8: u32,
    reserved1: [u32; 3],
    is_supported: u8,
    reserved2: [u8; 3],
    max_width: u32,
    max_height: u32,
    max_mb_count: u32,
    min_width: u16,
    min_height: u16,
    reserved3: [u32; 11],
}

impl CuvidDecodeCaps {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CuvidRect {
    left: i16,
    top: i16,
    right: i16,
    bottom: i16,
}

#[repr(C)]
struct CuvidDecodeCreateInfo {
    width: u32,
    height: u32,
    num_decode_surfaces: u32,
    codec_type: u32,
    chroma_format: u32,
    creation_flags: u32,
    bit_depth_minus8: u32,
    intra_decode_only: u32,
    max_width: u32,
    max_height: u32,
    reserved1: u32,
    display_area: CuvidRect,
    output_format: u32,
    deinterlace_mode: u32,
    target_width: u32,
    target_height: u32,
    num_output_surfaces: u32,
    video_context_lock: *mut c_void,
    target_rect: CuvidRect,
    reserved2: [u32; 5],
}

impl CuvidDecodeCreateInfo {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
struct CuVideoFormat {
    codec: u32,
    frame_rate_numerator: u32,
    frame_rate_denominator: u32,
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    reserved1: u8,
    coded_width: u32,
    coded_height: u32,
    display_area: CuvidDisplayRect,
    chroma_format: u32,
    bitrate: u32,
    aspect_ratio_x: i32,
    aspect_ratio_y: i32,
    video_signal_description: [u8; 4],
    sequence_header_data_length: u32,
}

#[repr(C)]
struct CuvidDisplayRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

type SequenceCallback = Option<unsafe extern "system" fn(*mut c_void, *mut CuVideoFormat) -> c_int>;
type DecodeCallback = Option<unsafe extern "system" fn(*mut c_void, *mut c_void) -> c_int>;
type DisplayCallback =
    Option<unsafe extern "system" fn(*mut c_void, *mut CuvidParserDispInfo) -> c_int>;

#[repr(C)]
struct CuvidParserParams {
    codec_type: u32,
    max_num_decode_surfaces: u32,
    clock_rate: u32,
    error_threshold: u32,
    max_display_delay: u32,
    reserved1: [u32; 5],
    user_data: *mut c_void,
    sequence_callback: SequenceCallback,
    decode_callback: DecodeCallback,
    display_callback: DisplayCallback,
    reserved2: [*mut c_void; 7],
    extended_video_info: *mut c_void,
}

impl CuvidParserParams {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
struct CuvidSourceDataPacket {
    flags: u32,
    payload_size: u32,
    payload: *const u8,
    timestamp: i64,
}

impl CuvidSourceDataPacket {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
struct CuvidParserDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: i64,
}

#[repr(C)]
struct CuvidProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: u32,
    reserved_zero: u32,
    raw_input_device_pointer: u64,
    raw_input_pitch: u32,
    raw_input_format: u32,
    raw_output_device_pointer: u64,
    raw_output_pitch: u32,
    reserved1: u32,
    output_stream: *mut c_void,
    reserved: [u32; 46],
    reserved2: [*mut c_void; 2],
}

impl CuvidProcParams {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
