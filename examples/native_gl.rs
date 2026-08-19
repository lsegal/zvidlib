//! Native GL rendering example.
//!
//! Demuxes an MP4 (by default the Big Buck Bunny sample, see
//! `examples/README.md` for how to download it) and drives the real
//! CPU -> native OpenGL [`execute_transfer`] path with a minimal in-process
//! [`GraphicsAdapter`] that stands in for a real GL context. Each uploaded
//! "texture" is written out as a PPM image so the transfer is observable
//! without a window.
//!
//! zvidlib does not ship a compressed (HEVC/AV1) video decoder backend yet,
//! so real decoded pixels are not available; this example demuxes the
//! sample's real track metadata but renders synthetic gradient frames sized
//! to the demuxed video track in place of `ExactFrameReader::get` output.
//!
//! Run with:
//!
//! ```console
//! cargo run --example native_gl --features native
//! ```

use std::collections::HashMap;
use std::env;
use std::fs;
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySource;
use zvidlib::{
    ColorRange, ContextIdentity, CpuFrameSource, Error, ErrorKind, ExecutionOwner,
    FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource, Limits,
    Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Plane, ResourceKind,
    ResourceOwnership, Result, TrackKind, TransferMode, TransferPolicy, TransferStage,
    VideoDimensions, VideoFrame, execute_transfer,
};

const FRAME_COUNT: usize = 5;
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
    let output_dir = PathBuf::from("examples/output/native_gl");

    let bytes = fs::read(&input_path).map_err(|error| {
        invalid(format!(
            "could not read {} ({error}); see examples/README.md for how to download the Big Buck Bunny sample",
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

    fs::create_dir_all(&output_dir).map_err(|error| {
        invalid(format!(
            "could not create {}: {error}",
            output_dir.display()
        ))
    })?;

    let mut adapter = RecordingGraphicsAdapter::new();
    let limits = Limits::default();
    let frame_count = FRAME_COUNT.min(video.presentation_order.len().max(1));
    for index in 0..frame_count {
        let frame = synthetic_frame(dimensions, index, frame_count, &limits)?;
        let resource = GraphicsResource::new(
            GraphicsApi::NativeOpenGl,
            adapter.context_identity(),
            adapter.execution_owner(),
            ResourceKind::Texture2d,
            TEXTURE_HANDLE,
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Full,
            Orientation::TopLeft,
            ResourceOwnership::Caller,
        );
        let capability = execute_transfer(
            Some(&mut adapter),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Graphics(resource),
            TransferPolicy::any(),
        )?;

        let path = output_dir.join(format!("frame_{index:04}.ppm"));
        adapter.write_texture_ppm(TEXTURE_HANDLE, dimensions, &path)?;
        println!(
            "Uploaded frame {index} via {:?} ({} stage(s)) -> {}",
            capability.mode,
            capability.stages.len(),
            path.display()
        );
    }

    Ok(())
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

/// A minimal [`GraphicsAdapter`] that records uploaded textures in memory instead of issuing
/// real OpenGL calls, so this example runs without windowing or GL dependencies.
struct RecordingGraphicsAdapter {
    context: ContextIdentity,
    owner: ExecutionOwner,
    textures: HashMap<u64, (VideoDimensions, Vec<u8>)>,
}

impl RecordingGraphicsAdapter {
    fn new() -> Self {
        Self {
            context: ContextIdentity(1),
            owner: ExecutionOwner(1),
            textures: HashMap::new(),
        }
    }

    fn write_texture_ppm(
        &self,
        handle: u64,
        dimensions: VideoDimensions,
        path: &std::path::Path,
    ) -> Result<()> {
        let (stored_dimensions, data) = self
            .textures
            .get(&handle)
            .ok_or_else(|| invalid("no texture was uploaded for this handle"))?;
        if *stored_dimensions != dimensions {
            return Err(invalid(
                "stored texture dimensions do not match the request",
            ));
        }
        let mut file = fs::File::create(path)
            .map_err(|error| invalid(format!("could not create {}: {error}", path.display())))?;
        write!(
            file,
            "P6\n{} {}\n255\n",
            dimensions.width, dimensions.height
        )
        .map_err(|error| invalid(format!("could not write {}: {error}", path.display())))?;
        for pixel in data.chunks_exact(4) {
            file.write_all(&pixel[..3])
                .map_err(|error| invalid(format!("could not write {}: {error}", path.display())))?;
        }
        Ok(())
    }
}

impl GraphicsAdapter for RecordingGraphicsAdapter {
    fn api(&self) -> GraphicsApi {
        GraphicsApi::NativeOpenGl
    }

    fn context_identity(&self) -> ContextIdentity {
        self.context
    }

    fn execution_owner(&self) -> ExecutionOwner {
        self.owner
    }

    fn is_current(&self) -> bool {
        true
    }

    fn is_context_lost(&self) -> bool {
        false
    }

    fn capability(
        &self,
        _source: FrameSource<'_>,
        _destination: &FrameDestination<'_>,
    ) -> TransferMode {
        TransferMode::GpuCopy
    }

    fn upload(
        &mut self,
        source: CpuFrameSource<'_>,
        destination: GraphicsResource,
        _stages: &[TransferStage],
    ) -> Result<()> {
        let plane = source
            .frame
            .planes
            .first()
            .ok_or_else(|| invalid("the source frame has no planes"))?;
        self.textures.insert(
            destination.handle(),
            (destination.dimensions(), plane.data.clone()),
        );
        Ok(())
    }

    fn readback(
        &mut self,
        _source: GraphicsResource,
        _destination: zvidlib::CpuFrameDestination<'_>,
        _stages: &[TransferStage],
    ) -> Result<()> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "RecordingGraphicsAdapter does not implement readback",
        ))
    }

    fn copy(
        &mut self,
        _source: GraphicsResource,
        _destination: GraphicsResource,
        _stages: &[TransferStage],
    ) -> Result<()> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "RecordingGraphicsAdapter does not implement copy",
        ))
    }

    fn delete(&mut self, resource: GraphicsResource) -> Result<()> {
        self.textures.remove(&resource.handle());
        Ok(())
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
