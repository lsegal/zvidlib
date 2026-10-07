//! Frame transfer contracts shared by CPU, native OpenGL, and browser WebGL backends.
//!
//! Graphics objects remain opaque and caller-owned unless explicitly marked otherwise. Backends
//! perform the actual GL/WebGL calls through [`GraphicsAdapter`], after this module validates the
//! context identity, execution owner, current-context state, context-loss state, and transfer
//! policy.

use crate::{
    ColorRange, Error, ErrorKind, PixelFormat, Plane, Result, TransferMode, VideoDimensions,
    VideoFrame,
};

/// The graphics API that created a resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphicsApi {
    NativeOpenGl,
    WebGl,
}

/// Stable identity supplied by a context adapter. It is never inferred from a GL object name.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextIdentity(pub u64);

/// Identity of the thread, worker, or executor allowed to use a context.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionOwner(pub u64);

/// Kind of caller- or library-supplied graphics object.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    Texture2d,
    Framebuffer,
}

/// Determines whether zvidlib may ask the adapter to delete a resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceOwnership {
    Caller,
    Library,
}

/// An opaque native GL or browser WebGL image resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphicsResource {
    api: GraphicsApi,
    context: ContextIdentity,
    owner: ExecutionOwner,
    kind: ResourceKind,
    handle: u64,
    dimensions: VideoDimensions,
    pixel_format: PixelFormat,
    color_range: ColorRange,
    orientation: Orientation,
    ownership: ResourceOwnership,
}

impl GraphicsResource {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        api: GraphicsApi,
        context: ContextIdentity,
        owner: ExecutionOwner,
        kind: ResourceKind,
        handle: u64,
        dimensions: VideoDimensions,
        pixel_format: PixelFormat,
        color_range: ColorRange,
        orientation: Orientation,
        ownership: ResourceOwnership,
    ) -> Self {
        Self {
            api,
            context,
            owner,
            kind,
            handle,
            dimensions,
            pixel_format,
            color_range,
            orientation,
            ownership,
        }
    }

    pub const fn api(self) -> GraphicsApi {
        self.api
    }
    pub const fn context(self) -> ContextIdentity {
        self.context
    }
    pub const fn owner(self) -> ExecutionOwner {
        self.owner
    }
    pub const fn kind(self) -> ResourceKind {
        self.kind
    }
    pub const fn handle(self) -> u64 {
        self.handle
    }
    pub const fn dimensions(self) -> VideoDimensions {
        self.dimensions
    }
    pub const fn pixel_format(self) -> PixelFormat {
        self.pixel_format
    }
    pub const fn color_range(self) -> ColorRange {
        self.color_range
    }
    pub const fn orientation(self) -> Orientation {
        self.orientation
    }
    pub const fn ownership(self) -> ResourceOwnership {
        self.ownership
    }

    /// Releases a library-owned resource. Caller-owned handles are deliberately left untouched.
    pub fn release(self, adapter: &mut dyn GraphicsAdapter) -> Result<()> {
        validate_resource(adapter, self)?;
        if self.ownership == ResourceOwnership::Library {
            adapter.delete(self)?;
        }
        Ok(())
    }
}

/// Location of the first logical row in memory or a graphics resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Orientation {
    TopLeft,
    BottomLeft,
}

/// Scaling filter requested by a transfer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScaleFilter {
    Nearest,
    Linear,
}

/// Whether numeric color values must be converted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ColorConversion {
    None,
    FullToLimited,
    LimitedToFull,
}

/// An explicit, inspectable operation in a transfer plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TransferStage {
    RepackStride,
    Scale { filter: ScaleFilter },
    FlipVertical,
    ConvertPixelFormat { from: PixelFormat, to: PixelFormat },
    ConvertColor { conversion: ColorConversion },
}

/// Runtime support and the cost/stages of a proposed transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferCapability {
    pub mode: TransferMode,
    pub stages: Vec<TransferStage>,
    pub reason: Option<String>,
}

impl TransferCapability {
    fn supported(mode: TransferMode, stages: Vec<TransferStage>) -> Self {
        Self {
            mode,
            stages,
            reason: None,
        }
    }

    fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            mode: TransferMode::Unsupported,
            stages: Vec::new(),
            reason: Some(reason.into()),
        }
    }

    pub fn is_supported(&self) -> bool {
        self.mode != TransferMode::Unsupported
    }
}

/// Cost policy for a frame transfer. `require` is strict and never silently falls back.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransferPolicy {
    required: Option<TransferMode>,
    scale_filter: ScaleFilter,
}

impl TransferPolicy {
    pub const fn any() -> Self {
        Self {
            required: None,
            scale_filter: ScaleFilter::Nearest,
        }
    }
    pub const fn require(mode: TransferMode) -> Self {
        Self {
            required: Some(mode),
            scale_filter: ScaleFilter::Nearest,
        }
    }
    pub const fn with_scale_filter(mut self, filter: ScaleFilter) -> Self {
        self.scale_filter = filter;
        self
    }
    pub const fn required(self) -> Option<TransferMode> {
        self.required
    }
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self::any()
    }
}

/// A borrowed CPU source with explicit orientation. Plane strides live on [`Plane`].
#[derive(Clone, Copy, Debug)]
pub struct CpuFrameSource<'a> {
    pub frame: &'a VideoFrame,
    pub orientation: Orientation,
}

/// A borrowed mutable destination plane with an explicit row stride.
#[derive(Debug)]
pub struct CpuPlaneDestination<'a> {
    pub data: &'a mut [u8],
    pub stride: usize,
}

/// A borrowed CPU destination. It does not take ownership of caller memory.
#[derive(Debug)]
pub struct CpuFrameDestination<'a> {
    pub dimensions: VideoDimensions,
    pub pixel_format: PixelFormat,
    pub color_range: ColorRange,
    pub orientation: Orientation,
    pub planes: Vec<CpuPlaneDestination<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub enum FrameSource<'a> {
    Cpu(CpuFrameSource<'a>),
    Graphics(GraphicsResource),
}

#[derive(Debug)]
pub enum FrameDestination<'a> {
    Cpu(CpuFrameDestination<'a>),
    Graphics(GraphicsResource),
}

/// Platform adapter that performs native GL or WebGL operations.
///
/// Implementations must not retain borrowed CPU memory after a call returns. The portable layer
/// invokes these methods only after checking context identity, execution owner, current state,
/// context loss, API identity, and policy.
pub trait GraphicsAdapter {
    fn api(&self) -> GraphicsApi;
    fn context_identity(&self) -> ContextIdentity;
    fn execution_owner(&self) -> ExecutionOwner;
    fn is_current(&self) -> bool;
    fn is_context_lost(&self) -> bool;
    fn capability(
        &self,
        source: FrameSource<'_>,
        destination: &FrameDestination<'_>,
    ) -> TransferMode;
    fn upload(
        &mut self,
        source: CpuFrameSource<'_>,
        destination: GraphicsResource,
        stages: &[TransferStage],
    ) -> Result<()>;
    fn readback(
        &mut self,
        source: GraphicsResource,
        destination: CpuFrameDestination<'_>,
        stages: &[TransferStage],
    ) -> Result<()>;
    fn copy(
        &mut self,
        source: GraphicsResource,
        destination: GraphicsResource,
        stages: &[TransferStage],
    ) -> Result<()>;
    fn delete(&mut self, resource: GraphicsResource) -> Result<()>;
}

/// Inspects a transfer without executing it.
pub fn inspect_transfer(
    adapter: Option<&dyn GraphicsAdapter>,
    source: FrameSource<'_>,
    destination: &FrameDestination<'_>,
    policy: TransferPolicy,
) -> TransferCapability {
    let mut stages = stages_for(source, destination, policy.scale_filter);
    let mode = match (source, destination) {
        (FrameSource::Cpu(_), FrameDestination::Cpu(_)) => TransferMode::CpuCopy,
        _ => {
            let Some(adapter) = adapter else {
                return TransferCapability::unsupported(
                    "a graphics transfer requires a context adapter",
                );
            };
            if let Err(error) = validate_endpoints(adapter, source, destination) {
                return TransferCapability::unsupported(error.message());
            }
            adapter.capability(source, destination)
        }
    };
    if mode == TransferMode::Unsupported {
        return TransferCapability::unsupported("the backend does not support this transfer");
    }
    if let Some(required) = policy.required {
        if required == TransferMode::Unsupported || required != mode {
            return TransferCapability::unsupported(format!(
                "strict policy requires {required:?}, but the backend reports {mode:?}"
            ));
        }
    }
    if mode == TransferMode::Shared && !stages.is_empty() {
        return TransferCapability::unsupported(
            "the backend reported a shared transfer that requires conversion stages",
        );
    }
    if mode == TransferMode::Shared {
        stages.clear();
    }
    TransferCapability::supported(mode, stages)
}

/// Validates, plans, and executes a CPU/GL/WebGL transfer.
pub fn execute_transfer(
    mut adapter: Option<&mut dyn GraphicsAdapter>,
    source: FrameSource<'_>,
    destination: FrameDestination<'_>,
    policy: TransferPolicy,
) -> Result<TransferCapability> {
    let capability = inspect_transfer(adapter.as_deref(), source, &destination, policy);
    if !capability.is_supported() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            capability
                .reason
                .clone()
                .unwrap_or_else(|| "unsupported frame transfer".into()),
        ));
    }
    match (source, destination) {
        (FrameSource::Cpu(source), FrameDestination::Cpu(destination)) => {
            transfer_cpu(source, destination, policy.scale_filter)?
        }
        (FrameSource::Cpu(source), FrameDestination::Graphics(destination)) => adapter
            .as_mut()
            .expect("validated adapter")
            .upload(source, destination, &capability.stages)?,
        (FrameSource::Graphics(source), FrameDestination::Cpu(destination)) => adapter
            .as_mut()
            .expect("validated adapter")
            .readback(source, destination, &capability.stages)?,
        (FrameSource::Graphics(source), FrameDestination::Graphics(destination)) => adapter
            .as_mut()
            .expect("validated adapter")
            .copy(source, destination, &capability.stages)?,
    }
    Ok(capability)
}

fn validate_endpoints(
    adapter: &dyn GraphicsAdapter,
    source: FrameSource<'_>,
    destination: &FrameDestination<'_>,
) -> Result<()> {
    if adapter.is_context_lost() {
        return Err(graphics("the graphics context is lost"));
    }
    if !adapter.is_current() {
        return Err(graphics("the graphics context is not current"));
    }
    if let FrameSource::Graphics(resource) = source {
        validate_resource(adapter, resource)?;
    }
    if let FrameDestination::Graphics(resource) = destination {
        validate_resource(adapter, *resource)?;
    }
    Ok(())
}

fn validate_resource(adapter: &dyn GraphicsAdapter, resource: GraphicsResource) -> Result<()> {
    if adapter.is_context_lost() {
        return Err(graphics("the graphics context is lost"));
    }
    if !adapter.is_current() {
        return Err(graphics("the graphics context is not current"));
    }
    if resource.api != adapter.api() {
        return Err(graphics("graphics resource API does not match the adapter"));
    }
    if resource.context != adapter.context_identity() {
        return Err(graphics("graphics resource belongs to a different context"));
    }
    if resource.owner != adapter.execution_owner() {
        return Err(graphics(
            "graphics resource belongs to a different execution owner",
        ));
    }
    Ok(())
}

fn stages_for(
    source: FrameSource<'_>,
    destination: &FrameDestination<'_>,
    filter: ScaleFilter,
) -> Vec<TransferStage> {
    let (source_dimensions, source_format, source_range, source_orientation, source_stride) =
        source_metadata(source);
    let (
        destination_dimensions,
        destination_format,
        destination_range,
        destination_orientation,
        destination_stride,
    ) = destination_metadata(destination);
    let mut stages = Vec::new();
    if source_stride != destination_stride {
        stages.push(TransferStage::RepackStride);
    }
    if source_dimensions != destination_dimensions {
        stages.push(TransferStage::Scale { filter });
    }
    if source_orientation != destination_orientation {
        stages.push(TransferStage::FlipVertical);
    }
    if source_format != destination_format {
        stages.push(TransferStage::ConvertPixelFormat {
            from: source_format,
            to: destination_format,
        });
    }
    if source_range != destination_range {
        let conversion = match (source_range, destination_range) {
            (ColorRange::Full, ColorRange::Limited) => ColorConversion::FullToLimited,
            (ColorRange::Limited, ColorRange::Full) => ColorConversion::LimitedToFull,
            _ => ColorConversion::None,
        };
        stages.push(TransferStage::ConvertColor { conversion });
    }
    stages
}

fn source_metadata(
    source: FrameSource<'_>,
) -> (
    VideoDimensions,
    PixelFormat,
    ColorRange,
    Orientation,
    Option<usize>,
) {
    match source {
        FrameSource::Cpu(cpu) => (
            cpu.frame.dimensions,
            cpu.frame.pixel_format,
            cpu.frame.color_range,
            cpu.orientation,
            cpu.frame.planes.first().map(|plane| plane.stride),
        ),
        FrameSource::Graphics(resource) => (
            resource.dimensions,
            resource.pixel_format,
            resource.color_range,
            resource.orientation,
            None,
        ),
    }
}

fn destination_metadata(
    destination: &FrameDestination<'_>,
) -> (
    VideoDimensions,
    PixelFormat,
    ColorRange,
    Orientation,
    Option<usize>,
) {
    match destination {
        FrameDestination::Cpu(cpu) => (
            cpu.dimensions,
            cpu.pixel_format,
            cpu.color_range,
            cpu.orientation,
            cpu.planes.first().map(|plane| plane.stride),
        ),
        FrameDestination::Graphics(resource) => (
            resource.dimensions,
            resource.pixel_format,
            resource.color_range,
            resource.orientation,
            None,
        ),
    }
}

fn transfer_cpu(
    source: CpuFrameSource<'_>,
    mut destination: CpuFrameDestination<'_>,
    filter: ScaleFilter,
) -> Result<()> {
    if filter == ScaleFilter::Linear && source.frame.dimensions != destination.dimensions {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "linear CPU scaling is not implemented",
        ));
    }
    validate_cpu_destination(&destination)?;
    if !is_packed(source.frame.pixel_format) || !is_packed(destination.pixel_format) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "CPU conversion currently supports packed RGBA, BGRA, RGB, and gray frames",
        ));
    }
    let source_width = usize::try_from(source.frame.dimensions.width)
        .map_err(|_| limit("source width cannot be represented"))?;
    let source_height = usize::try_from(source.frame.dimensions.height)
        .map_err(|_| limit("source height cannot be represented"))?;
    let destination_width = usize::try_from(destination.dimensions.width)
        .map_err(|_| limit("destination width cannot be represented"))?;
    let destination_height = usize::try_from(destination.dimensions.height)
        .map_err(|_| limit("destination height cannot be represented"))?;
    let source_plane = &source.frame.planes[0];
    let destination_plane = &mut destination.planes[0];
    for destination_y in 0..destination_height {
        let logical_source_y = destination_y
            .checked_mul(source_height)
            .ok_or_else(|| limit("vertical scale overflow"))?
            / destination_height;
        let source_y = orient_row(logical_source_y, source_height, source.orientation);
        let output_y = orient_row(destination_y, destination_height, destination.orientation);
        for destination_x in 0..destination_width {
            let source_x = destination_x
                .checked_mul(source_width)
                .ok_or_else(|| limit("horizontal scale overflow"))?
                / destination_width;
            let mut rgba = read_pixel(source_plane, source.frame.pixel_format, source_x, source_y)?;
            convert_range(&mut rgba, source.frame.color_range, destination.color_range);
            write_pixel(
                destination_plane,
                destination.pixel_format,
                destination_x,
                output_y,
                rgba,
            )?;
        }
    }
    Ok(())
}

fn validate_cpu_destination(destination: &CpuFrameDestination<'_>) -> Result<()> {
    if destination.planes.len() != 1 || !is_packed(destination.pixel_format) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "CPU destination must contain one supported packed plane",
        ));
    }
    let width = usize::try_from(destination.dimensions.width)
        .map_err(|_| limit("destination width cannot be represented"))?;
    let height = usize::try_from(destination.dimensions.height)
        .map_err(|_| limit("destination height cannot be represented"))?;
    let row_bytes = width
        .checked_mul(bytes_per_pixel(destination.pixel_format)?)
        .ok_or_else(|| limit("destination row size overflow"))?;
    let plane = &destination.planes[0];
    if plane.stride < row_bytes {
        return Err(invalid("destination stride is too small"));
    }
    let length = plane
        .stride
        .checked_mul(height)
        .ok_or_else(|| limit("destination plane size overflow"))?;
    if plane.data.len() < length {
        return Err(invalid(
            "destination plane is shorter than its stride and height",
        ));
    }
    Ok(())
}

fn is_packed(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Rgb8 | PixelFormat::Gray8
    )
}

fn bytes_per_pixel(format: PixelFormat) -> Result<usize> {
    match format {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => Ok(4),
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Gray8 => Ok(1),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "pixel format is not a packed CPU format",
        )),
    }
}

fn orient_row(row: usize, height: usize, orientation: Orientation) -> usize {
    match orientation {
        Orientation::TopLeft => row,
        Orientation::BottomLeft => height - 1 - row,
    }
}

fn read_pixel(plane: &Plane, format: PixelFormat, x: usize, y: usize) -> Result<[u8; 4]> {
    let offset = y
        .checked_mul(plane.stride)
        .and_then(|row| row.checked_add(x.checked_mul(bytes_per_pixel(format).ok()?)?))
        .ok_or_else(|| limit("source pixel offset overflow"))?;
    let data = &plane.data;
    match format {
        PixelFormat::Rgba8 => Ok([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
        PixelFormat::Bgra8 => Ok([
            data[offset + 2],
            data[offset + 1],
            data[offset],
            data[offset + 3],
        ]),
        PixelFormat::Rgb8 => Ok([data[offset], data[offset + 1], data[offset + 2], 255]),
        PixelFormat::Gray8 => Ok([data[offset], data[offset], data[offset], 255]),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "pixel format is not a packed CPU format",
        )),
    }
}

fn write_pixel(
    plane: &mut CpuPlaneDestination<'_>,
    format: PixelFormat,
    x: usize,
    y: usize,
    rgba: [u8; 4],
) -> Result<()> {
    let offset = y
        .checked_mul(plane.stride)
        .and_then(|row| row.checked_add(x.checked_mul(bytes_per_pixel(format).ok()?)?))
        .ok_or_else(|| limit("destination pixel offset overflow"))?;
    match format {
        PixelFormat::Rgba8 => plane.data[offset..offset + 4].copy_from_slice(&rgba),
        PixelFormat::Bgra8 => {
            plane.data[offset..offset + 4].copy_from_slice(&[rgba[2], rgba[1], rgba[0], rgba[3]])
        }
        PixelFormat::Rgb8 => plane.data[offset..offset + 3].copy_from_slice(&rgba[..3]),
        PixelFormat::Gray8 => {
            plane.data[offset] = ((u16::from(rgba[0]) * 77
                + u16::from(rgba[1]) * 150
                + u16::from(rgba[2]) * 29
                + 128)
                >> 8) as u8
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "pixel format is not a packed CPU format",
            ));
        }
    }
    Ok(())
}

fn convert_range(rgba: &mut [u8; 4], source: ColorRange, destination: ColorRange) {
    if source == destination {
        return;
    }
    for component in &mut rgba[..3] {
        *component = match (source, destination) {
            (ColorRange::Full, ColorRange::Limited) => {
                (16 + (u32::from(*component) * 219 + 127) / 255) as u8
            }
            (ColorRange::Limited, ColorRange::Full) => {
                ((u32::from(component.saturating_sub(16)) * 255 + 109) / 219).min(255) as u8
            }
            _ => *component,
        };
    }
}

fn graphics(message: &str) -> Error {
    Error::new(ErrorKind::Graphics, message)
}
fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
fn limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;

    fn dimensions(width: u32, height: u32) -> VideoDimensions {
        VideoDimensions::new(width, height, &Limits::default()).unwrap()
    }

    fn rgba_frame(width: u32, height: u32, stride: usize, data: Vec<u8>) -> VideoFrame {
        VideoFrame::new(
            dimensions(width, height),
            PixelFormat::Rgba8,
            ColorRange::Full,
            vec![Plane { data, stride }],
            &Limits::default(),
        )
        .unwrap()
    }

    fn resource(api: GraphicsApi, ownership: ResourceOwnership) -> GraphicsResource {
        GraphicsResource::new(
            api,
            ContextIdentity(7),
            ExecutionOwner(9),
            ResourceKind::Texture2d,
            42,
            dimensions(2, 2),
            PixelFormat::Rgba8,
            ColorRange::Full,
            Orientation::BottomLeft,
            ownership,
        )
    }

    #[derive(Debug)]
    struct FakeAdapter {
        api: GraphicsApi,
        context: ContextIdentity,
        owner: ExecutionOwner,
        current: bool,
        lost: bool,
        mode: TransferMode,
        calls: Vec<&'static str>,
    }

    impl FakeAdapter {
        fn new(api: GraphicsApi, mode: TransferMode) -> Self {
            Self {
                api,
                context: ContextIdentity(7),
                owner: ExecutionOwner(9),
                current: true,
                lost: false,
                mode,
                calls: Vec::new(),
            }
        }
    }

    impl GraphicsAdapter for FakeAdapter {
        fn api(&self) -> GraphicsApi {
            self.api
        }
        fn context_identity(&self) -> ContextIdentity {
            self.context
        }
        fn execution_owner(&self) -> ExecutionOwner {
            self.owner
        }
        fn is_current(&self) -> bool {
            self.current
        }
        fn is_context_lost(&self) -> bool {
            self.lost
        }
        fn capability(
            &self,
            _source: FrameSource<'_>,
            _destination: &FrameDestination<'_>,
        ) -> TransferMode {
            self.mode
        }
        fn upload(
            &mut self,
            _source: CpuFrameSource<'_>,
            _destination: GraphicsResource,
            _stages: &[TransferStage],
        ) -> Result<()> {
            self.calls.push("upload");
            Ok(())
        }
        fn readback(
            &mut self,
            _source: GraphicsResource,
            mut destination: CpuFrameDestination<'_>,
            _stages: &[TransferStage],
        ) -> Result<()> {
            destination.planes[0].data.fill(23);
            self.calls.push("readback");
            Ok(())
        }
        fn copy(
            &mut self,
            _source: GraphicsResource,
            _destination: GraphicsResource,
            _stages: &[TransferStage],
        ) -> Result<()> {
            self.calls.push("copy");
            Ok(())
        }
        fn delete(&mut self, _resource: GraphicsResource) -> Result<()> {
            self.calls.push("delete");
            Ok(())
        }
    }

    #[test]
    fn cpu_transfer_handles_stride_scaling_orientation_and_format() {
        let source = rgba_frame(
            2,
            2,
            12,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 9, 9, 9, 9, 0, 0, 255, 255, 255, 255, 255, 255, 9,
                9, 9, 9,
            ],
        );
        let mut output = vec![0; 4];
        let destination = CpuFrameDestination {
            dimensions: dimensions(1, 1),
            pixel_format: PixelFormat::Bgra8,
            color_range: ColorRange::Full,
            orientation: Orientation::BottomLeft,
            planes: vec![CpuPlaneDestination {
                data: &mut output,
                stride: 4,
            }],
        };
        let capability = execute_transfer(
            None,
            FrameSource::Cpu(CpuFrameSource {
                frame: &source,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Cpu(destination),
            TransferPolicy::any(),
        )
        .unwrap();
        assert_eq!(output, [0, 0, 255, 255]);
        assert_eq!(capability.mode, TransferMode::CpuCopy);
        assert!(capability.stages.contains(&TransferStage::RepackStride));
        assert!(capability.stages.contains(&TransferStage::Scale {
            filter: ScaleFilter::Nearest
        }));
        assert!(capability.stages.contains(&TransferStage::FlipVertical));
    }

    #[test]
    fn cpu_transfer_converts_color_range() {
        let source = rgba_frame(1, 1, 4, vec![0, 255, 128, 255]);
        let mut output = vec![0; 4];
        let destination = CpuFrameDestination {
            dimensions: dimensions(1, 1),
            pixel_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            orientation: Orientation::TopLeft,
            planes: vec![CpuPlaneDestination {
                data: &mut output,
                stride: 4,
            }],
        };
        execute_transfer(
            None,
            FrameSource::Cpu(CpuFrameSource {
                frame: &source,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Cpu(destination),
            TransferPolicy::any(),
        )
        .unwrap();
        assert_eq!(output, [16, 235, 126, 255]);
    }

    #[test]
    fn native_gl_upload_and_webgl_readback_use_the_adapter() {
        let frame = rgba_frame(2, 2, 8, vec![0; 16]);
        let mut native = FakeAdapter::new(GraphicsApi::NativeOpenGl, TransferMode::GpuCopy);
        execute_transfer(
            Some(&mut native),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Graphics(resource(
                GraphicsApi::NativeOpenGl,
                ResourceOwnership::Caller,
            )),
            TransferPolicy::require(TransferMode::GpuCopy),
        )
        .unwrap();
        assert_eq!(native.calls, ["upload"]);

        let mut output = vec![0; 16];
        let destination = CpuFrameDestination {
            dimensions: dimensions(2, 2),
            pixel_format: PixelFormat::Rgba8,
            color_range: ColorRange::Full,
            orientation: Orientation::TopLeft,
            planes: vec![CpuPlaneDestination {
                data: &mut output,
                stride: 8,
            }],
        };
        let mut web = FakeAdapter::new(GraphicsApi::WebGl, TransferMode::CpuCopy);
        execute_transfer(
            Some(&mut web),
            FrameSource::Graphics(resource(GraphicsApi::WebGl, ResourceOwnership::Caller)),
            FrameDestination::Cpu(destination),
            TransferPolicy::require(TransferMode::CpuCopy),
        )
        .unwrap();
        assert_eq!(output, vec![23; 16]);
        assert_eq!(web.calls, ["readback"]);
    }

    #[test]
    fn strict_policy_rejects_fallback_without_calling_backend() {
        let frame = rgba_frame(2, 2, 8, vec![0; 16]);
        let mut adapter = FakeAdapter::new(GraphicsApi::NativeOpenGl, TransferMode::CpuCopy);
        let error = execute_transfer(
            Some(&mut adapter),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
            FrameDestination::Graphics(resource(
                GraphicsApi::NativeOpenGl,
                ResourceOwnership::Caller,
            )),
            TransferPolicy::require(TransferMode::Shared),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(adapter.calls.is_empty());
    }

    #[test]
    fn shared_graphics_transfer_is_reported_when_no_conversion_is_needed() {
        let mut adapter = FakeAdapter::new(GraphicsApi::NativeOpenGl, TransferMode::Shared);
        let graphics = resource(GraphicsApi::NativeOpenGl, ResourceOwnership::Caller);
        let capability = execute_transfer(
            Some(&mut adapter),
            FrameSource::Graphics(graphics),
            FrameDestination::Graphics(graphics),
            TransferPolicy::require(TransferMode::Shared),
        )
        .unwrap();
        assert_eq!(capability.mode, TransferMode::Shared);
        assert!(capability.stages.is_empty());
        assert_eq!(adapter.calls, ["copy"]);
    }

    #[test]
    fn context_loss_identity_owner_and_current_state_are_validated() {
        let destination = FrameDestination::Graphics(resource(
            GraphicsApi::NativeOpenGl,
            ResourceOwnership::Caller,
        ));
        let frame = rgba_frame(2, 2, 8, vec![0; 16]);
        let source = FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        });
        let mut adapter = FakeAdapter::new(GraphicsApi::NativeOpenGl, TransferMode::GpuCopy);
        adapter.lost = true;
        assert_eq!(
            inspect_transfer(Some(&adapter), source, &destination, TransferPolicy::any()).mode,
            TransferMode::Unsupported
        );
        adapter.lost = false;
        adapter.current = false;
        assert_eq!(
            inspect_transfer(Some(&adapter), source, &destination, TransferPolicy::any()).mode,
            TransferMode::Unsupported
        );
        adapter.current = true;
        adapter.context = ContextIdentity(99);
        assert_eq!(
            inspect_transfer(Some(&adapter), source, &destination, TransferPolicy::any()).mode,
            TransferMode::Unsupported
        );
        adapter.context = ContextIdentity(7);
        adapter.owner = ExecutionOwner(99);
        assert_eq!(
            inspect_transfer(Some(&adapter), source, &destination, TransferPolicy::any()).mode,
            TransferMode::Unsupported
        );
    }

    #[test]
    fn cleanup_deletes_only_library_owned_resources() {
        let mut adapter = FakeAdapter::new(GraphicsApi::WebGl, TransferMode::GpuCopy);
        resource(GraphicsApi::WebGl, ResourceOwnership::Caller)
            .release(&mut adapter)
            .unwrap();
        assert!(adapter.calls.is_empty());
        resource(GraphicsApi::WebGl, ResourceOwnership::Library)
            .release(&mut adapter)
            .unwrap();
        assert_eq!(adapter.calls, ["delete"]);
    }
}
