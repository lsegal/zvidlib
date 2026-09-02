//! Windows Media Foundation HEVC decode backed by the active D3D11 video device.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_DECODER_PROFILE_HEVC_VLD_MAIN, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device,
    ID3D11DeviceContext, ID3D11Texture2D, ID3D11VideoDevice,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};
use windows::Win32::Media::MediaFoundation::{
    CLSID_MSH265DecoderMFT, IMFAttributes, IMFDXGIBuffer, IMFDXGIDeviceManager, IMFSample,
    IMFTransform, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT, MF_E_TRANSFORM_STREAM_CHANGE,
    MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_SA_D3D11_AWARE,
    MF_TRANSFORM_ASYNC, MF_VERSION, MFCreateDXGIDeviceManager, MFCreateMediaType,
    MFCreateMemoryBuffer, MFCreateSample, MFMediaType_Video, MFShutdown, MFStartup,
    MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
    MFT_MESSAGE_NOTIFY_END_OF_STREAM, MFT_MESSAGE_NOTIFY_START_OF_STREAM,
    MFT_MESSAGE_SET_D3D_MANAGER, MFT_OUTPUT_DATA_BUFFER, MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES,
    MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MFVideoFormat_HEVC, MFVideoFormat_NV12,
    MFVideoInterlace_Progressive,
};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};
use windows::core::Interface;

use super::engine::hvcc::{HvccRecord, split_length_prefixed};
use super::readback;
use crate::{
    CancellationToken, DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex, Limits,
    PixelFormat, Plane, Result, VideoDecoder, VideoDecoderConfig, VideoDimensions, VideoFrame,
};

pub(super) fn is_available(dimensions: VideoDimensions) -> bool {
    let (ready_tx, ready_rx) = sync_channel(1);
    if thread::Builder::new()
        .name("zvidlib-mf-probe".into())
        .spawn(move || {
            let result = MfRuntime::start().and_then(|_runtime| {
                let (device, _) = create_d3d_device()?;
                require_hevc_hardware(&device, dimensions)?;
                let transform = create_transform()?;
                require_d3d_aware(&transform)
            });
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
    HardwareDecoder::spawn(
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

struct HardwareDecoder {
    commands: SyncSender<Command>,
    worker: Option<JoinHandle<()>>,
}

impl HardwareDecoder {
    fn spawn(
        configuration: VideoDecoderConfig,
        limits: Limits,
        nal_length_size: usize,
        parameter_sets: Vec<u8>,
    ) -> Result<Self> {
        let (command_tx, command_rx) = sync_channel(1);
        let (ready_tx, ready_rx) = sync_channel(1);
        let worker = thread::Builder::new()
            .name("zvidlib-mf-hevc".into())
            .spawn(move || {
                let core = DecoderCore::new(configuration, limits, nal_length_size, parameter_sets);
                match core {
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
            .map_err(|error| codec(format!("could not start Media Foundation worker: {error}")))?;
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
                Err(codec(
                    "Media Foundation worker stopped during initialization",
                ))
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
            .map_err(|_| codec("Media Foundation decoder worker is not running"))?;
        response_rx
            .recv()
            .map_err(|_| codec("Media Foundation decoder worker stopped unexpectedly"))?
    }
}

impl VideoDecoder for HardwareDecoder {
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
    /// returns is drained under the setting the caller asked for. The command channel is FIFO,
    /// so the wait is not what orders it; it is what keeps a dropped worker visible here rather
    /// than silently leaving the decoder converting frames nobody wants.
    fn set_output_wanted(&mut self, wanted: bool) {
        let _ = self.request(|response| Command::SetOutputWanted { wanted, response });
    }
}

impl Drop for HardwareDecoder {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_worker(mut core: DecoderCore, commands: Receiver<Command>) {
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

struct DecoderCore {
    _runtime: MfRuntime,
    transform: IMFTransform,
    _device: ID3D11Device,
    context: ID3D11DeviceContext,
    _device_manager: IMFDXGIDeviceManager,
    configuration: VideoDecoderConfig,
    limits: Limits,
    nal_length_size: usize,
    parameter_sets: Vec<u8>,
    identities: HashMap<i64, FrameIndex>,
    next_identity: i64,
    /// Whether the samples `ProcessOutput` hands back are wanted as frames. A suppressed sample
    /// is still decoded and still collected - the transform will not proceed until its output is
    /// taken - but it never reaches the staging copy or the colour conversion.
    output_wanted: bool,
}

impl DecoderCore {
    fn new(
        configuration: VideoDecoderConfig,
        limits: Limits,
        nal_length_size: usize,
        parameter_sets: Vec<u8>,
    ) -> Result<Self> {
        let runtime = MfRuntime::start()?;
        let (device, context) = create_d3d_device()?;
        require_hevc_hardware(&device, configuration.coded_dimensions)?;
        let transform = create_transform()?;
        require_d3d_aware(&transform)?;
        let mut reset_token = 0;
        let mut manager = None;
        unsafe {
            MFCreateDXGIDeviceManager(&mut reset_token, &mut manager)
                .map_err(|error| windows_error("could not create D3D device manager", error))?;
        }
        let manager = manager.ok_or_else(|| codec("D3D device manager was not returned"))?;
        unsafe {
            manager
                .ResetDevice(&device, reset_token)
                .map_err(|error| windows_error("could not attach D3D device", error))?;
            transform
                .ProcessMessage(
                    MFT_MESSAGE_SET_D3D_MANAGER,
                    Interface::as_raw(&manager) as usize,
                )
                .map_err(|error| windows_error("HEVC decoder rejected D3D manager", error))?;
        }
        set_input_type(&transform, configuration.coded_dimensions)?;
        set_nv12_output_type(&transform)?;
        let stream_info = unsafe { transform.GetOutputStreamInfo(0) }
            .map_err(|error| windows_error("could not query HEVC output stream", error))?;
        let provides = MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32;
        let can_provide = MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0 as u32;
        if stream_info.dwFlags & (provides | can_provide) == 0 {
            return Err(codec(
                "D3D HEVC decoder requires caller-allocated output samples",
            ));
        }
        unsafe {
            transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
                .map_err(|error| windows_error("could not begin HEVC streaming", error))?;
            transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
                .map_err(|error| windows_error("could not start HEVC stream", error))?;
        }
        Ok(Self {
            _runtime: runtime,
            transform,
            _device: device,
            context,
            _device_manager: manager,
            configuration,
            limits,
            nal_length_size,
            parameter_sets,
            identities: HashMap::new(),
            next_identity: 1,
            output_wanted: true,
        })
    }

    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "HEVC access unit exceeds the allocation limit",
            ));
        }
        let annex_b = annex_b_sample(
            sample,
            self.nal_length_size,
            &self.parameter_sets,
            self.limits.max_allocation_bytes,
        )?;
        let token = self.next_identity;
        self.next_identity = self
            .next_identity
            .checked_add(1)
            .ok_or_else(|| codec("Media Foundation timestamp identity overflow"))?;
        self.identities.insert(token, sample.presentation_index);
        let input = make_input_sample(&annex_b, token)?;
        let mut output = self.collect_output(cancellation)?;
        match unsafe { self.transform.ProcessInput(0, &input, 0) } {
            Ok(()) => {}
            Err(error) if error.code() == MF_E_NOTACCEPTING => {
                output.extend(self.collect_output(cancellation)?);
                unsafe { self.transform.ProcessInput(0, &input, 0) }.map_err(|error| {
                    windows_error("Media Foundation rejected HEVC input", error)
                })?;
            }
            Err(error) => {
                return Err(windows_error("Media Foundation rejected HEVC input", error));
            }
        }
        output.extend(self.collect_output(cancellation)?);
        Ok(output)
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        unsafe {
            self.transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)
                .map_err(|error| windows_error("could not end HEVC stream", error))?;
            self.transform
                .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                .map_err(|error| windows_error("could not drain HEVC decoder", error))?;
        }
        self.collect_output(cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        unsafe {
            self.transform
                .ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0)
                .map_err(|error| windows_error("could not reset HEVC decoder", error))?;
            self.transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
                .map_err(|error| windows_error("could not restart HEVC decoder", error))?;
        }
        self.identities.clear();
        self.next_identity = 1;
        Ok(())
    }

    fn collect_output(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        let mut frames = Vec::new();
        loop {
            check_cancelled(cancellation)?;
            let mut data = MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(None),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            };
            let mut status = 0;
            let result = unsafe {
                self.transform
                    .ProcessOutput(0, std::slice::from_mut(&mut data), &mut status)
            };
            let sample = unsafe { ManuallyDrop::take(&mut data.pSample) };
            let _events = unsafe { ManuallyDrop::take(&mut data.pEvents) };
            match result {
                Ok(()) => {
                    let sample = sample.ok_or_else(|| {
                        codec("Media Foundation reported output without a video sample")
                    })?;
                    if self.output_wanted {
                        frames.push(self.convert_output(sample)?);
                    } else {
                        self.discard_output(&sample)?;
                    }
                }
                Err(error) if error.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(frames),
                Err(error) if error.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    set_nv12_output_type(&self.transform)?;
                }
                Err(error) => {
                    return Err(windows_error(
                        "Media Foundation could not produce HEVC output",
                        error,
                    ));
                }
            }
        }
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
    }

    /// Retires a decoded sample without reading its picture.
    ///
    /// Only the identity bookkeeping of [`DecoderCore::convert_output`] is kept: the sample's
    /// timestamp is still matched and consumed, so the frames that follow keep their own, and
    /// dropping the sample releases the decoder's surface. What is skipped is the D3D11 staging
    /// copy and the NV12-to-RGBA pass over a picture nothing will draw.
    fn discard_output(&mut self, sample: &IMFSample) -> Result<()> {
        let token = unsafe { sample.GetSampleTime() }
            .map_err(|error| windows_error("decoded HEVC frame has no timestamp", error))?;
        self.identities.remove(&token).ok_or_else(|| {
            codec("decoded HEVC frame timestamp does not match a submitted sample")
        })?;
        Ok(())
    }

    fn convert_output(&mut self, sample: IMFSample) -> Result<DecodedVideoFrame> {
        let token = unsafe { sample.GetSampleTime() }
            .map_err(|error| windows_error("decoded HEVC frame has no timestamp", error))?;
        let presentation_index = self.identities.remove(&token).ok_or_else(|| {
            codec("decoded HEVC frame timestamp does not match a submitted sample")
        })?;
        let buffer = unsafe { sample.GetBufferByIndex(0) }
            .map_err(|error| windows_error("decoded HEVC frame has no buffer", error))?;
        let dxgi: IMFDXGIBuffer = buffer
            .cast()
            .map_err(|error| windows_error("decoded HEVC frame is not D3D11-backed", error))?;
        let mut raw_texture = ptr::null_mut::<c_void>();
        unsafe {
            dxgi.GetResource(&ID3D11Texture2D::IID, &mut raw_texture)
                .map_err(|error| windows_error("could not access decoded D3D11 texture", error))?;
        }
        if raw_texture.is_null() {
            return Err(codec("decoded D3D11 texture was null"));
        }
        let source = unsafe { ID3D11Texture2D::from_raw(raw_texture) };
        let source_subresource = unsafe { dxgi.GetSubresourceIndex() }
            .map_err(|error| windows_error("could not query decoded texture index", error))?;
        let frame = copy_nv12_to_rgba(
            &self._device,
            &self.context,
            &source,
            source_subresource,
            &self.configuration,
            &self.limits,
        )?;
        Ok(DecodedVideoFrame {
            presentation_index,
            frame,
        })
    }
}

struct MfRuntime;

impl MfRuntime {
    fn start() -> Result<Self> {
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
        unsafe {
            let _ = MFShutdown();
            CoUninitialize();
        }
    }
}

fn create_d3d_device() -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(|error| windows_error("could not create D3D11 video device", error))?;
    }
    Ok((
        device.ok_or_else(|| codec("D3D11 device was not returned"))?,
        context.ok_or_else(|| codec("D3D11 context was not returned"))?,
    ))
}

fn require_hevc_hardware(device: &ID3D11Device, dimensions: VideoDimensions) -> Result<()> {
    let video: ID3D11VideoDevice = device
        .cast()
        .map_err(|error| windows_error("D3D11 video decode is unavailable", error))?;
    let profile_count = unsafe { video.GetVideoDecoderProfileCount() };
    let mut found = false;
    for index in 0..profile_count {
        if unsafe { video.GetVideoDecoderProfile(index) }
            .is_ok_and(|profile| profile == D3D11_DECODER_PROFILE_HEVC_VLD_MAIN)
        {
            found = true;
            break;
        }
    }
    if !found {
        return Err(unsupported(
            "D3D11 adapter does not support HEVC Main decode",
        ));
    }
    let supported = unsafe {
        video.CheckVideoDecoderFormat(&D3D11_DECODER_PROFILE_HEVC_VLD_MAIN, DXGI_FORMAT_NV12)
    }
    .map_err(|error| windows_error("could not query D3D11 HEVC output support", error))?;
    if !supported.as_bool() {
        return Err(unsupported("D3D11 adapter cannot decode HEVC Main to NV12"));
    }
    let pixels = u64::from(dimensions.width) * u64::from(dimensions.height);
    if pixels == 0 {
        return Err(unsupported("D3D11 HEVC dimensions must be nonzero"));
    }
    Ok(())
}

fn create_transform() -> Result<IMFTransform> {
    unsafe { CoCreateInstance(&CLSID_MSH265DecoderMFT, None, CLSCTX_INPROC_SERVER) }
        .map_err(|error| windows_error("Windows HEVC decoder is unavailable", error))
}

fn require_d3d_aware(transform: &IMFTransform) -> Result<()> {
    let attributes: IMFAttributes = unsafe { transform.GetAttributes() }
        .map_err(|error| windows_error("could not query HEVC decoder attributes", error))?;
    if unsafe { attributes.GetUINT32(&MF_TRANSFORM_ASYNC) }.unwrap_or(0) != 0 {
        return Err(unsupported(
            "the installed HEVC decoder requires unsupported asynchronous MFT processing",
        ));
    }
    if unsafe { attributes.GetUINT32(&MF_SA_D3D11_AWARE) }.unwrap_or(0) == 0 {
        return Err(unsupported(
            "the installed HEVC decoder is not D3D11 accelerated",
        ));
    }
    Ok(())
}

fn set_input_type(transform: &IMFTransform, dimensions: VideoDimensions) -> Result<()> {
    let media_type = unsafe { MFCreateMediaType() }
        .map_err(|error| windows_error("could not create HEVC input type", error))?;
    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .and_then(|()| media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_HEVC))
            .and_then(|()| media_type.SetUINT64(&MF_MT_FRAME_SIZE, frame_size(dimensions)))
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            })
            .and_then(|()| transform.SetInputType(0, &media_type, 0))
            .map_err(|error| windows_error("could not configure HEVC input type", error))
    }
}

fn set_nv12_output_type(transform: &IMFTransform) -> Result<()> {
    for index in 0..64 {
        let media_type = match unsafe { transform.GetOutputAvailableType(0, index) } {
            Ok(media_type) => media_type,
            Err(_) => break,
        };
        let subtype = unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) };
        if subtype.is_ok_and(|subtype| subtype == MFVideoFormat_NV12)
            && unsafe { transform.SetOutputType(0, &media_type, 0) }.is_ok()
        {
            return Ok(());
        }
    }
    Err(unsupported(
        "Windows HEVC decoder does not expose D3D11 NV12 output",
    ))
}

fn make_input_sample(data: &[u8], timestamp: i64) -> Result<IMFSample> {
    let size = u32::try_from(data.len())
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "HEVC access unit is too large"))?;
    let buffer = unsafe { MFCreateMemoryBuffer(size) }
        .map_err(|error| windows_error("could not allocate HEVC input buffer", error))?;
    let mut destination = ptr::null_mut();
    unsafe {
        buffer
            .Lock(&mut destination, None, None)
            .map_err(|error| windows_error("could not lock HEVC input buffer", error))?;
        ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
        let unlock = buffer.Unlock();
        unlock.map_err(|error| windows_error("could not unlock HEVC input buffer", error))?;
        buffer
            .SetCurrentLength(size)
            .map_err(|error| windows_error("could not size HEVC input buffer", error))?;
    }
    let sample = unsafe { MFCreateSample() }
        .map_err(|error| windows_error("could not allocate HEVC input sample", error))?;
    unsafe {
        sample
            .AddBuffer(&buffer)
            .and_then(|()| sample.SetSampleTime(timestamp))
            .and_then(|()| sample.SetSampleDuration(1))
            .map_err(|error| windows_error("could not populate HEVC input sample", error))?;
    }
    Ok(sample)
}

fn copy_nv12_to_rgba(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    source: &ID3D11Texture2D,
    source_subresource: u32,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    let dimensions = configuration.coded_dimensions;
    let stride = usize::try_from(dimensions.width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC RGBA stride overflows"))?;
    let length = stride
        .checked_mul(dimensions.height as usize)
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "HEVC RGBA allocation overflows"))?;
    if length as u64 > limits.max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "HEVC RGBA output exceeds the allocation limit",
        ));
    }
    let description = D3D11_TEXTURE2D_DESC {
        Width: dimensions.width,
        Height: dimensions.height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_NV12,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
    };
    // Media Foundation's surface-copy phase is the staging round trip: a
    // CPU-readable texture, the copy into it, and the map that makes it
    // addressable. The decoded texture itself is never CPU-readable.
    let surface_copy = readback::Timer::start();
    let mut staging = None;
    unsafe {
        device
            .CreateTexture2D(&description, None, Some(&mut staging))
            .map_err(|error| windows_error("could not allocate HEVC readback texture", error))?;
    }
    let staging = staging.ok_or_else(|| codec("HEVC readback texture was not returned"))?;
    unsafe {
        context.CopySubresourceRegion(&staging, 0, 0, 0, 0, source, source_subresource, None);
    }
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        context
            .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|error| windows_error("could not map HEVC readback texture", error))?;
    }
    if mapped.pData.is_null() {
        unsafe { context.Unmap(&staging, 0) };
        return Err(codec("mapped HEVC readback texture was null"));
    }
    surface_copy.record(readback::Phase::SurfaceCopy);
    let color_convert = readback::Timer::start();
    let row_pitch = mapped.RowPitch as usize;
    let width = dimensions.width as usize;
    let height = dimensions.height as usize;
    let base = mapped.pData.cast::<u8>();
    let mut rgba = vec![0_u8; length];
    for y in 0..height {
        let luma = unsafe { std::slice::from_raw_parts(base.add(y * row_pitch), width) };
        let chroma =
            unsafe { std::slice::from_raw_parts(base.add((height + y / 2) * row_pitch), width) };
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
    unsafe { context.Unmap(&staging, 0) };
    VideoFrame::new(
        dimensions,
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

fn frame_size(dimensions: VideoDimensions) -> u64 {
    (u64::from(dimensions.width) << 32) | u64::from(dimensions.height)
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

fn windows_error(context: &str, error: windows::core::Error) -> Error {
    codec(format!("{context}: {error}"))
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}
