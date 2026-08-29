//! Native GL rendering example.
//!
//! Demuxes an MP4 (by default the Big Buck Bunny sample checked into
//! `examples/media/BigBuckBunny.mp4`) and drives the real
//! native HEVC decoder and the CPU -> native OpenGL [`execute_transfer`] path with a
//! [`GraphicsAdapter`] backed by a real
//! `winit` window and `glutin` OpenGL context, so the uploaded frames are drawn to an actual
//! window on all platforms. The example prefers an available accelerated backend and falls back
//! to the dependency-free software decoder. Decoding runs on a background thread, and the render
//! loop displays the newest completed frame. Playback loops back to the first frame once the last
//! one is shown, and the decoded-frame playback rate is drawn in the top-left corner.
//!
//! Run with:
//!
//! ```console
//! cargo run --example native_gl --features native
//! ```

use std::env;
use std::fs;
use std::future::Future;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::DisplayBuilder;
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, CodecProfile, ColorRange, CpuFrameSource, Error, ErrorKind,
    ExactFrameReader, FrameDestination, FrameIndex, FrameSource, GraphicsAdapter, GraphicsApi,
    GraphicsResource, HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions, Orientation,
    PixelFormat, ResourceKind, ResourceOwnership, Result, TrackKind, TransferPolicy,
    VideoDecoderConfig, VideoDecoderFactory, VideoDimensions, VideoFrame, execute_transfer,
    native_hevc_video_decoder_factory,
};

mod gl_window;

use gl_window::{FpsCounter, GlWindowAdapter};

const TEXTURE_HANDLE: u64 = 1;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {}", error.message());
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let input_path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/media/BigBuckBunny.mp4")
    });

    let bytes = fs::read(&input_path).map_err(|error| {
        invalid(format!(
            "could not read {} ({error}); pass an MP4 path as the first argument to use a different sample",
            input_path.display()
        ))
    })?;
    let source = MemorySource::new(bytes);

    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))?;
    let video = demuxer
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .ok_or_else(|| invalid("the MP4 does not contain a video track"))?;
    let dimensions = video
        .dimensions
        .ok_or_else(|| invalid("the video track does not report dimensions"))?;

    println!(
        "Opened {}: {}x{} {:?}, {} samples ({} in presentation order)",
        input_path.display(),
        dimensions.width,
        dimensions.height,
        video.codec,
        video.samples.len(),
        video.presentation_order.len(),
    );
    let frame_durations = video
        .presentation_order
        .iter()
        .map(|&decode_index| {
            let sample = video
                .samples
                .get(decode_index)
                .ok_or_else(|| invalid("video presentation index references a missing sample"))?;
            sample_duration(sample.duration, video.timescale)
        })
        .collect::<Result<Vec<_>>>()?;
    let limits = Limits::default();
    let samples = block_on(video.to_encoded_video_samples(&source, &limits))?;
    let factory = native_hevc_video_decoder_factory();
    let configuration = VideoDecoderConfig {
        codec: video.codec,
        profile: CodecProfile::HevcMain,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: video.decoder_config.clone(),
    };
    let support = factory.capability(&configuration);
    let reader = ExactFrameReader::new(&factory, configuration, samples, limits)?;
    println!("Selected {support:?} HEVC decoding.");
    println!(
        "Rendering decoded {:?} pixels through native OpenGL.",
        video.codec
    );

    let decoder = DecodeThread::spawn(reader, frame_durations);
    let mut app = App::new(dimensions, decoder);
    let event_loop = EventLoop::new()
        .map_err(|error| invalid(format!("could not create the event loop: {error}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut app)
        .map_err(|error| invalid(format!("windowed event loop failed: {error}")))?;
    app.into_result()
}

/// One decoded frame handed from the background decode thread to the render loop.
struct DecodedFrame {
    frame: VideoFrame,
    display_duration: Duration,
}

/// Decodes the (looping) video on a dedicated background thread and hands finished frames to
/// the render loop through a bounded channel.
///
/// The accelerated decoder exceeds the bundled sample's source rate on supported hardware. The
/// software fallback remains slower at 1080p; keeping either backend off the render callback lets
/// the window redraw at the display's own pace while showing the newest completed frame.
struct DecodeThread {
    frames: Receiver<DecodedFrame>,
    cancellation: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

impl DecodeThread {
    fn spawn(mut reader: ExactFrameReader, frame_durations: Vec<Duration>) -> Self {
        let (sender, frames) = sync_channel(1);
        let cancellation = CancellationToken::new();
        let thread_cancellation = cancellation.clone();
        let handle = thread::spawn(move || {
            let mut index = 0usize;
            loop {
                let frame = match reader.get(FrameIndex(index as u64), &thread_cancellation) {
                    Ok(frame) => frame,
                    Err(_) => return,
                };
                let display_duration = frame_durations[index];
                if sender
                    .send(DecodedFrame {
                        frame,
                        display_duration,
                    })
                    .is_err()
                {
                    return;
                }
                index = (index + 1) % frame_durations.len();
            }
        });
        Self {
            frames,
            cancellation,
            handle: Some(handle),
        }
    }

    /// Returns the next decoded frame without draining later frames from the bounded channel.
    fn next(&mut self) -> std::result::Result<Option<DecodedFrame>, ()> {
        match self.frames.try_recv() {
            Ok(frame) => Ok(Some(frame)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(()),
        }
    }
}

impl Drop for DecodeThread {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Drives the `winit` window, owns the `glutin` GL context, and renders looping frames.
struct App {
    dimensions: VideoDimensions,
    decoder: DecodeThread,
    fps: FpsCounter,
    pacer: FramePacer,
    state: Option<WindowState>,
    error: Option<Error>,
}

struct WindowState {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    adapter: GlWindowAdapter,
}

impl App {
    fn new(dimensions: VideoDimensions, decoder: DecodeThread) -> Self {
        Self {
            dimensions,
            decoder,
            fps: FpsCounter::new(),
            pacer: FramePacer::new(),
            state: None,
            error: None,
        }
    }

    fn into_result(self) -> Result<()> {
        match self.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
        eprintln!("error: {}", error.message());
        self.error = Some(error);
        event_loop.exit();
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        let now = Instant::now();
        let frame_presented = if !self.pacer.ready(now) {
            false
        } else {
            match self.decoder.next() {
                Ok(Some(decoded)) => {
                    let resource = GraphicsResource::new(
                        GraphicsApi::NativeOpenGl,
                        state.adapter.context_identity(),
                        state.adapter.execution_owner(),
                        ResourceKind::Texture2d,
                        TEXTURE_HANDLE,
                        self.dimensions,
                        PixelFormat::Rgba8,
                        ColorRange::Limited,
                        Orientation::TopLeft,
                        ResourceOwnership::Caller,
                    );
                    if let Err(error) = execute_transfer(
                        Some(&mut state.adapter),
                        FrameSource::Cpu(CpuFrameSource {
                            frame: &decoded.frame,
                            orientation: Orientation::TopLeft,
                        }),
                        FrameDestination::Graphics(resource),
                        TransferPolicy::any(),
                    ) {
                        return self.fail(event_loop, error);
                    }
                    self.pacer.presented(decoded.display_duration, now);
                    true
                }
                Ok(None) => false,
                Err(()) => {
                    return self.fail(
                        event_loop,
                        invalid("the decode thread stopped unexpectedly"),
                    );
                }
            }
        };

        let fps = self.fps.update(frame_presented, now);
        state.adapter.draw(TEXTURE_HANDLE, self.dimensions, fps);
        if let Err(error) = state.surface.swap_buffers(&state.context) {
            return self.fail(
                event_loop,
                invalid(format!("could not swap GL buffers: {error}")),
            );
        }

        state.window.request_redraw();
    }
}

/// Prevents decoded frames from being displayed faster than their source presentation durations.
struct FramePacer {
    next_frame_at: Option<Instant>,
}

impl FramePacer {
    fn new() -> Self {
        Self {
            next_frame_at: None,
        }
    }

    fn ready(&self, now: Instant) -> bool {
        self.next_frame_at.is_none_or(|deadline| now >= deadline)
    }

    fn presented(&mut self, duration: Duration, now: Instant) {
        self.next_frame_at = Some(now + duration);
    }
}

fn sample_duration(ticks: u32, timescale: u32) -> Result<Duration> {
    if ticks == 0 || timescale == 0 {
        return Err(invalid("video sample timing must be nonzero"));
    }
    let nanos = u64::from(ticks)
        .checked_mul(1_000_000_000)
        .ok_or_else(|| invalid("video sample duration overflow"))?;
    Ok(Duration::from_nanos(
        (nanos + u64::from(timescale) / 2) / u64::from(timescale),
    ))
}

#[cfg(test)]
mod pacing_tests {
    use super::*;

    #[test]
    fn converts_mp4_timing_to_a_frame_duration() {
        assert_eq!(
            sample_duration(512, 12_288).unwrap(),
            Duration::from_nanos(41_666_667)
        );
    }

    #[test]
    fn waits_a_full_source_duration_after_each_presented_frame() {
        let start = Instant::now();
        let duration = Duration::from_millis(40);
        let mut pacer = FramePacer::new();

        assert!(pacer.ready(start));
        pacer.presented(duration, start);
        assert!(!pacer.ready(start + Duration::from_millis(39)));
        assert!(pacer.ready(start + duration));

        let late = start + Duration::from_millis(100);
        pacer.presented(duration, late);
        assert!(!pacer.ready(late + Duration::from_millis(39)));
        assert!(pacer.ready(late + duration));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("zvidlib native GL example")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.dimensions.width.max(1),
                self.dimensions.height.max(1),
            ));
        let template = ConfigTemplateBuilder::new();
        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));

        let (window, gl_config) = match display_builder.build(event_loop, template, |configs| {
            configs
                .reduce(|accum, config| {
                    if config.num_samples() > accum.num_samples() {
                        config
                    } else {
                        accum
                    }
                })
                .expect("glutin reported no usable GL configs")
        }) {
            Ok((Some(window), config)) => (window, config),
            Ok((None, _)) => {
                return self.fail(event_loop, invalid("glutin did not create a window"));
            }
            Err(error) => {
                return self.fail(
                    event_loop,
                    invalid(format!("could not create a GL window: {error}")),
                );
            }
        };

        let raw_window_handle = window.window_handle().ok().map(|handle| handle.as_raw());
        let gl_display = gl_config.display();
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(None))
            .build(raw_window_handle);
        let not_current_context =
            match unsafe { gl_display.create_context(&gl_config, &context_attributes) } {
                Ok(context) => context,
                Err(error) => {
                    return self.fail(
                        event_loop,
                        invalid(format!("could not create a GL context: {error}")),
                    );
                }
            };

        let size = window.inner_size();
        let width = NonZeroU32::new(size.width.max(1)).expect("non-zero window width");
        let height = NonZeroU32::new(size.height.max(1)).expect("non-zero window height");
        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle.expect("window must provide a raw window handle"),
            width,
            height,
        );
        let surface =
            match unsafe { gl_display.create_window_surface(&gl_config, &surface_attributes) } {
                Ok(surface) => surface,
                Err(error) => {
                    return self.fail(
                        event_loop,
                        invalid(format!("could not create a GL surface: {error}")),
                    );
                }
            };

        let context = match not_current_context.make_current(&surface) {
            Ok(context) => context,
            Err(error) => {
                return self.fail(
                    event_loop,
                    invalid(format!("could not activate the GL context: {error}")),
                );
            }
        };

        if let Err(error) =
            surface.set_swap_interval(&context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
        {
            eprintln!("warning: could not enable vsync: {error}");
        }

        let gl = unsafe {
            glow::Context::from_loader_function(|symbol| {
                let symbol = std::ffi::CString::new(symbol).unwrap();
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            })
        };
        let mut adapter = GlWindowAdapter::new(gl);
        adapter.resize(size.width, size.height);

        window.request_redraw();
        self.state = Some(WindowState {
            window,
            surface,
            context,
            adapter,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let matches_window = self
            .state
            .as_ref()
            .is_some_and(|state| state.window.id() == id);
        if !matches_window {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    if let (Some(width), Some(height)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        state.surface.resize(&state.context, width, height);
                        state.adapter.resize(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.render(event_loop),
            _ => {}
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

/// Drives a future to completion. `Mp4Demuxer::open` never actually suspends against a
/// `MemorySource`, so a single poll always resolves.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut boxed = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match boxed.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("unexpected pending future in a synchronous example"),
    }
}
