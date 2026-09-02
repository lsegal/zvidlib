//! Native GL rendering example.
//!
//! Demuxes an MP4 (by default the Big Buck Bunny sample checked into
//! `examples/media/BigBuckBunny.mp4`) and drives the real
//! native HEVC decoder and the CPU -> native OpenGL [`execute_transfer`] path with a
//! [`GraphicsAdapter`] backed by a real
//! `winit` window and `glutin` OpenGL context, so the uploaded frames are drawn to an actual
//! window on all platforms. The example prefers an available accelerated backend and falls back
//! to the dependency-free software decoder. Playback loops back to the first frame once the last
//! one is shown, the decoded-frame playback rate is drawn in the top-left corner, and a control
//! legend is drawn above the timeline bar until it fades out (press `H` to show or hide it).
//!
//! Dragging the timeline bar decodes its previews on a background thread (see [`scrub`]), so the
//! window keeps drawing and keeps its audio scheduled while the pointer moves.
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
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use zvidlib::io::MemorySource;
use zvidlib::{
    AacSampleReader, Codec, CodecProfile, ColorRange, CpuFrameSource, DefaultAudioOutput, Error,
    ErrorKind, FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource,
    HardwarePreference, IndexedPresentationTimeline, Limits, Mp4Demuxer, Mp4DemuxerOptions,
    NativeAacDecoder, NativeAudioOutput, Orientation, PixelFormat, PlaybackController,
    PlaybackOptions, ResourceKind, ResourceOwnership, Result, TrackKind, TransferPolicy,
    VideoDecoderConfig, VideoDecoderFactory, VideoDimensions, execute_transfer,
    native_hevc_video_decoder_factory,
};

mod gl_window;
mod scrub;

use gl_window::{CONTROL_LEGEND, FpsCounter, GlWindowAdapter, LegendVisibility};
use scrub::{FrameService, target_frame};

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
    let audio = demuxer
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Audio && track.codec == Codec::Aac)
        .ok_or_else(|| invalid("the MP4 does not contain an AAC audio track"))?;
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
    let limits = Limits::default();
    let video_samples = block_on(video.to_encoded_video_samples(&source, &limits))?;
    let audio_config = audio.aac_config()?;
    let audio_packets = block_on(audio.to_encoded_audio_samples(&source, &limits))?;
    let audio_timing = audio.audio_timing(demuxer.movie_timescale)?;
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
    // Every frame the window draws comes off this one worker thread, playback's included: a
    // decode on the event-loop thread is what froze the window for seconds after a scrub.
    let frames = FrameService::new(&factory, configuration, video_samples, limits)?;
    let video_reader = frames.source();
    let audio_decoder = NativeAacDecoder::new(&audio_config, Limits::default())?;
    let audio_reader = AacSampleReader::new(
        audio_decoder,
        audio_packets,
        audio_config.sample_rate,
        audio_config.channels,
        audio_timing,
        2,
        Limits::default(),
    )?;
    let output = NativeAudioOutput(DefaultAudioOutput::open(
        audio_config.sample_rate,
        audio_config.channels,
    )?);
    let frame_count = video.presentation_order.len().max(1) as u64;
    let frames_per_five_seconds = ((u128::from(frame_count) * u128::from(video.timescale) * 5)
        / u128::from(video.duration.max(1)))
    .max(1)
    .min(u128::from(frame_count)) as u64;
    let timeline =
        IndexedPresentationTimeline::from_mp4_track(video, audio_config.sample_rate, &limits)?;
    let playback = PlaybackController::new_with_indexed_timeline(
        video_reader,
        audio_reader,
        output,
        timeline,
        PlaybackOptions::for_sample_rate(audio_config.sample_rate),
    )?;
    println!("Selected {support:?} HEVC decoding.");
    println!(
        "Playing {:?} video and AAC audio through zvidlib's synchronized pipeline.",
        video.codec
    );

    println!(
        "Controls: SPACE play/pause, LEFT/RIGHT previous/next frame, J/L rewind/fast-forward 5s, \
         click or drag the timeline bar along the bottom edge to scrub, H show/hide the \
         on-screen legend ({} lines).",
        CONTROL_LEGEND.len()
    );

    let mut app = App::new(
        dimensions,
        playback,
        frame_count,
        frames_per_five_seconds,
        frames,
    );
    let event_loop = EventLoop::new()
        .map_err(|error| invalid(format!("could not create the event loop: {error}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut app)
        .map_err(|error| invalid(format!("windowed event loop failed: {error}")))?;
    app.into_result()
}

/// Drives the `winit` window, owns the `glutin` GL context, and renders looping frames.
struct App<P> {
    dimensions: VideoDimensions,
    playback: P,
    frame_count: u64,
    frames_per_five_seconds: u64,
    /// A frame is wanted that playback is not going to ask for on its own: the still the window
    /// shows while paused, at the frame a seek or a committed scrub left it on.
    needs_static_frame: bool,
    timeline_hover: Option<f32>,
    frames: FrameService,
    /// The pointer is down on the timeline bar, so previews replace normal presentation.
    dragging: bool,
    /// The exact frame the drag is pointing at, committed when the pointer is released.
    scrub_target: Option<u64>,
    legend: LegendVisibility,
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

impl<V, A, O> App<PlaybackController<V, A, O>>
where
    V: zvidlib::PlaybackVideoSource,
    A: zvidlib::PlaybackAudioSource,
    O: zvidlib::PlaybackAudioOutput,
{
    fn new(
        dimensions: VideoDimensions,
        playback: PlaybackController<V, A, O>,
        frame_count: u64,
        frames_per_five_seconds: u64,
        frames: FrameService,
    ) -> Self {
        Self {
            dimensions,
            playback,
            frame_count,
            frames_per_five_seconds,
            needs_static_frame: false,
            timeline_hover: None,
            frames,
            dragging: false,
            scrub_target: None,
            legend: LegendVisibility::new(Instant::now()),
            fps: FpsCounter::new(),
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

    fn toggle_playback(&mut self, event_loop: &ActiveEventLoop) {
        let result = if self.playback.is_playing() {
            self.needs_static_frame = true;
            self.playback.pause()
        } else {
            self.needs_static_frame = false;
            self.playback.play()
        };
        if let Err(error) = result {
            self.fail(event_loop, error);
        }
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn seek_by_frames(&mut self, event_loop: &ActiveEventLoop, delta: i64) {
        let Ok(current) = self.playback.current_frame_index() else {
            return;
        };
        let maximum = self.frame_count.saturating_sub(1);
        let target = if delta < 0 {
            current.0.saturating_sub(delta.unsigned_abs())
        } else {
            current.0.saturating_add(delta as u64).min(maximum)
        };
        self.seek_to_frame(event_loop, target);
    }

    fn seek_to_frame(&mut self, event_loop: &ActiveEventLoop, frame: u64) {
        if let Err(error) = self.playback.seek(zvidlib::FrameIndex(frame)) {
            return self.fail(event_loop, error);
        }
        if !self.playback.is_playing() {
            self.needs_static_frame = true;
        }
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    fn timeline_fraction(&self, x: f64, y: f64) -> Option<f32> {
        let state = self.state.as_ref()?;
        let size = state.window.inner_size();
        if y < f64::from(size.height.saturating_sub(40)) {
            return None;
        }
        Some(self.horizontal_fraction(x))
    }

    /// Where `x` falls across the window, ignoring how far the pointer has drifted off the bar
    /// vertically: a drag that started on the timeline keeps scrubbing until it is released.
    fn horizontal_fraction(&self, x: f64) -> f32 {
        let width = self
            .state
            .as_ref()
            .map_or(1, |state| state.window.inner_size().width);
        (x / f64::from(width.max(1))).clamp(0.0, 1.0) as f32
    }

    /// Records where the drag is pointing and points the worker thread at it.
    ///
    /// Nothing here decodes. The worker walks to the frame under the pointer from its
    /// random-access point, drawing what it passes, so the picture follows the drag from the
    /// first intra picture onwards instead of waiting for the whole group of pictures.
    fn scrub_preview(&mut self, fraction: f32) {
        let target = target_frame(fraction, self.frame_count);
        self.scrub_target = Some(target);
        self.frames.request(target);
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    /// Ends a drag by moving playback to the frame the pointer was left on. Only the clock moves
    /// here: the picture arrives from the worker, which the drag has already walked to that
    /// frame, so committing decodes nothing.
    fn commit_scrub(&mut self, event_loop: &ActiveEventLoop) {
        self.dragging = false;
        let Some(target) = self.scrub_target.take() else {
            return;
        };
        // Unconditional, even when the position has not moved: a preview replaced whatever was
        // on screen, so the committed frame still has to be drawn.
        self.seek_to_frame(event_loop, target);
    }

    /// Draws the newest frame the worker has decoded, if one has arrived, without waiting for it.
    fn present_latest_frame(&mut self, event_loop: &ActiveEventLoop) -> bool {
        // `present` would decode the frame the audio clock calls for, which is exactly the work a
        // drag must not do on this thread; the audio queue still needs topping up.
        if let Err(error) = self.playback.pump_audio() {
            self.fail(event_loop, error);
            return false;
        }
        match self.frames.take_latest() {
            Some(frame) => {
                self.upload_frame(event_loop, &frame);
                true
            }
            None => false,
        }
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            return;
        }

        let now = Instant::now();
        let frame_presented = if self.dragging {
            self.present_latest_frame(event_loop)
        } else if self.playback.is_playing() {
            match self.playback.present() {
                Ok((_, Some(frame))) => {
                    self.upload_frame(event_loop, &frame);
                    true
                }
                Ok((presentation, None)) => {
                    if presentation.finished {
                        if let Err(error) = self.playback.seek(zvidlib::FrameIndex(0)) {
                            return self.fail(event_loop, error);
                        }
                    }
                    false
                }
                // The frame the clock calls for is still decoding. The window keeps the picture
                // it has and asks again on the next redraw rather than waiting for it, which is
                // what lets playback resume immediately after a seek.
                Err(error) if error.kind() == ErrorKind::WouldBlock => false,
                Err(error) => {
                    return self.fail(event_loop, error);
                }
            }
        } else if self.needs_static_frame {
            match self.playback.current_frame() {
                Ok(frame) => {
                    self.needs_static_frame = false;
                    self.upload_frame(event_loop, &frame);
                    true
                }
                // Same again while paused: the still is drawn when the worker has decoded it, and
                // until then whatever the walk to it has passed keeps the window moving.
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    self.present_latest_frame(event_loop)
                }
                Err(error) => return self.fail(event_loop, error),
            }
        } else {
            false
        };

        let last_frame = self.frame_count.saturating_sub(1).max(1) as f32;
        // While dragging, the bar follows the pointer rather than the audio clock, which has not
        // moved there yet.
        let progress = match self.scrub_target {
            Some(target) if self.dragging => target as f32 / last_frame,
            _ => self
                .playback
                .current_frame_index()
                .map(|frame| frame.0 as f32 / last_frame)
                .unwrap_or(0.0),
        };
        let fps = self.fps.update(frame_presented, now);
        let legend_opacity = self.legend.opacity(now);
        let state = self.state.as_mut().expect("state exists");
        state.adapter.draw(
            TEXTURE_HANDLE,
            self.dimensions,
            fps,
            progress,
            self.timeline_hover,
            legend_opacity,
        );
        if let Err(error) = state.surface.swap_buffers(&state.context) {
            return self.fail(
                event_loop,
                invalid(format!("could not swap GL buffers: {error}")),
            );
        }

        // Keep redrawing while the legend is fading so the fade animates even when paused, while
        // a drag is open so a preview that finishes decoding is picked up, and while a still is
        // outstanding so the frame a seek asked for is drawn when the worker delivers it.
        if self.playback.is_playing()
            || legend_opacity > 0.0
            || self.dragging
            || self.needs_static_frame
        {
            state.window.request_redraw();
        }
    }

    fn upload_frame(&mut self, event_loop: &ActiveEventLoop, frame: &zvidlib::VideoFrame) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
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
                frame,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Graphics(resource),
            TransferPolicy::any(),
        ) {
            self.fail(event_loop, error);
        }
    }
}

impl<V, A, O> ApplicationHandler for App<PlaybackController<V, A, O>>
where
    V: zvidlib::PlaybackVideoSource,
    A: zvidlib::PlaybackAudioSource,
    O: zvidlib::PlaybackAudioOutput,
{
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
        if let Err(error) = self.playback.play() {
            self.fail(event_loop, error);
        }
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
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Space) => self.toggle_playback(event_loop),
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.seek_by_frames(event_loop, -1),
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.seek_by_frames(event_loop, 1),
                    PhysicalKey::Code(KeyCode::KeyJ) => {
                        self.seek_by_frames(event_loop, -(self.frames_per_five_seconds as i64))
                    }
                    PhysicalKey::Code(KeyCode::KeyL) => {
                        self.seek_by_frames(event_loop, self.frames_per_five_seconds as i64)
                    }
                    PhysicalKey::Code(KeyCode::KeyH) => {
                        self.legend.toggle(Instant::now());
                        if let Some(state) = self.state.as_ref() {
                            state.window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.timeline_hover = self.timeline_fraction(position.x, position.y);
                // Hovering only moves the bar's marker. Seeking on a bare pointer move is what
                // made a single pass across the bar queue hundreds of decodes (issue #319).
                if self.dragging {
                    let fraction = self.horizontal_fraction(position.x);
                    // A drag keeps the marker under the pointer even once it has slid off the
                    // bar vertically, since it is still scrubbing.
                    self.timeline_hover = Some(fraction);
                    self.scrub_preview(fraction);
                } else if let Some(state) = self.state.as_ref() {
                    state.window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(fraction) = self.timeline_hover {
                    self.dragging = true;
                    self.scrub_preview(fraction);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if self.dragging {
                    self.commit_scrub(event_loop);
                }
            }
            // A drag that leaves the window has no release to wait for.
            WindowEvent::CursorLeft { .. } => {
                self.timeline_hover = None;
                if self.dragging {
                    self.commit_scrub(event_loop);
                }
            }
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
