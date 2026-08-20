//! Native GL rendering example.
//!
//! Demuxes an MP4 (by default the Big Buck Bunny sample checked into
//! `examples/media/BigBuckBunny.mp4`) and drives the real
//! CPU -> native OpenGL [`execute_transfer`] path with a [`GraphicsAdapter`] backed by a real
//! `winit` window and `glutin` OpenGL context, so the uploaded frames are drawn to an actual
//! window on all platforms. Playback loops back to the first frame once the last one is shown,
//! and an FPS counter is drawn in the top-left corner.
//!
//! zvidlib does not ship a compressed (HEVC/AV1) video decoder backend yet, so real decoded
//! pixels are not available; this example demuxes the sample's real track metadata but renders
//! synthetic gradient frames sized to the demuxed video track in place of
//! `ExactFrameReader::get` output.
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
use std::task::{Context, Poll, Waker};
use std::time::Instant;

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
    ColorRange, CpuFrameSource, Error, ErrorKind, FrameDestination, FrameSource, GraphicsAdapter,
    GraphicsApi, GraphicsResource, Limits, Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat,
    Plane, ResourceKind, ResourceOwnership, Result, TrackKind, TransferPolicy, VideoDimensions,
    VideoFrame, execute_transfer,
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
    let input_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/media/BigBuckBunny.mp4"));

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
    println!(
        "zvidlib has no registered compressed decoder backend yet, so this example renders \
         synthetic frames through the real CPU -> native GL upload path instead of decoded \
         {:?} pixels.",
        video.codec
    );

    let frame_count = video.presentation_order.len().max(1);
    let limits = Limits::default();
    let mut app = App::new(dimensions, frame_count, limits);
    let event_loop = EventLoop::new()
        .map_err(|error| invalid(format!("could not create the event loop: {error}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut app)
        .map_err(|error| invalid(format!("windowed event loop failed: {error}")))?;
    app.into_result()
}

/// Drives the `winit` window, owns the `glutin` GL context, and renders looping frames.
struct App {
    dimensions: VideoDimensions,
    frame_count: usize,
    limits: Limits,
    frame_index: usize,
    fps: FpsCounter,
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
    fn new(dimensions: VideoDimensions, frame_count: usize, limits: Limits) -> Self {
        Self {
            dimensions,
            frame_count,
            limits,
            frame_index: 0,
            fps: FpsCounter::new(Instant::now()),
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

        let frame = match synthetic_frame(
            self.dimensions,
            self.frame_index,
            self.frame_count,
            &self.limits,
        ) {
            Ok(frame) => frame,
            Err(error) => return self.fail(event_loop, error),
        };
        let resource = GraphicsResource::new(
            GraphicsApi::NativeOpenGl,
            state.adapter.context_identity(),
            state.adapter.execution_owner(),
            ResourceKind::Texture2d,
            TEXTURE_HANDLE,
            self.dimensions,
            PixelFormat::Rgba8,
            ColorRange::Full,
            Orientation::TopLeft,
            ResourceOwnership::Caller,
        );
        if let Err(error) = execute_transfer(
            Some(&mut state.adapter),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Graphics(resource),
            TransferPolicy::any(),
        ) {
            return self.fail(event_loop, error);
        }

        let fps = self.fps.tick(Instant::now());
        state.adapter.draw(TEXTURE_HANDLE, self.dimensions, fps);
        if let Err(error) = state.surface.swap_buffers(&state.context) {
            return self.fail(
                event_loop,
                invalid(format!("could not swap GL buffers: {error}")),
            );
        }

        // Loop back to the first frame once the last one has been shown.
        self.frame_index = (self.frame_index + 1) % self.frame_count.max(1);
        state.window.request_redraw();
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
        let adapter = GlWindowAdapter::new(gl);

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

/// Builds a synthetic RGBA gradient frame sized to `dimensions`. Stands in for a decoded frame
/// until zvidlib registers a compressed video decoder backend.
fn synthetic_frame(
    dimensions: VideoDimensions,
    index: usize,
    frame_count: usize,
    limits: &Limits,
) -> Result<VideoFrame> {
    let width = dimensions.width as usize;
    let height = dimensions.height as usize;
    let stride = width * 4;
    let phase = (255 * index / frame_count.max(1)) as u8;
    let mut data = vec![0_u8; stride * height];
    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * 4;
            data[offset] = (255 * x / width.max(1)) as u8;
            data[offset + 1] = (255 * y / height.max(1)) as u8;
            data[offset + 2] = phase;
            data[offset + 3] = 255;
        }
    }
    VideoFrame::new(
        dimensions,
        PixelFormat::Rgba8,
        ColorRange::Full,
        vec![Plane { data, stride }],
        limits,
    )
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
