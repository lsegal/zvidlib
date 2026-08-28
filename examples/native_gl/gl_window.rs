//! Real GL backend for the `native_gl` example: a [`zvidlib::GraphicsAdapter`] that uploads
//! frames as a real OpenGL texture, draws them to the window each frame via `glow`, and overlays
//! a small bitmap-font FPS counter in the top-left corner.

use std::time::{Duration, Instant};

use glow::HasContext;
use zvidlib::{
    ContextIdentity, CpuFrameSource, Error, ErrorKind, ExecutionOwner, FrameSource,
    GraphicsAdapter, GraphicsApi, GraphicsResource, Result, TransferMode, TransferStage,
    VideoDimensions,
};

const VIDEO_VERTEX_SHADER: &str = r#"#version 330 core
in vec2 aCorner;
uniform vec4 uRectNdc;
out vec2 vUv;
void main() {
    gl_Position = vec4(mix(uRectNdc.x, uRectNdc.z, aCorner.x),
                        mix(uRectNdc.y, uRectNdc.w, aCorner.y), 0.0, 1.0);
    vUv = vec2(aCorner.x, 1.0 - aCorner.y);
}
"#;

const VIDEO_FRAGMENT_SHADER: &str = r#"#version 330 core
in vec2 vUv;
uniform sampler2D uTexture;
out vec4 fragColor;
void main() {
    fragColor = texture(uTexture, vUv);
}
"#;

/// Rows of a 5x7 bitmap font, MSB (bit 4) is the leftmost column of each row.
const FONT_WIDTH: usize = 5;
const FONT_HEIGHT: usize = 7;

fn glyph(ch: char) -> [u8; FONT_HEIGHT] {
    match ch {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110,
        ],
        ':' => [
            0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000,
        ],
        _ => [0; FONT_HEIGHT],
    }
}

/// Rasterizes `text` into an RGBA buffer, `scale` pixels per font pixel, with a 1-font-pixel
/// margin and a translucent background so the counter stays legible over any frame content.
fn rasterize_text(text: &str, scale: usize) -> (usize, usize, Vec<u8>) {
    let margin = 1usize;
    let cols = text.chars().count();
    let glyph_w = FONT_WIDTH + 1;
    let width = (cols * glyph_w + margin * 2) * scale;
    let height = (FONT_HEIGHT + margin * 2) * scale;
    let mut data = vec![0_u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            data[offset] = 0;
            data[offset + 1] = 0;
            data[offset + 2] = 0;
            data[offset + 3] = 140;
        }
    }

    for (index, ch) in text.chars().enumerate() {
        let rows = glyph(ch);
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..FONT_WIDTH {
                if bits & (1 << (FONT_WIDTH - 1 - col)) == 0 {
                    continue;
                }
                let base_x = (margin + index * glyph_w + col) * scale;
                let base_y = (margin + row) * scale;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = base_x + dx;
                        let py = base_y + dy;
                        let offset = (py * width + px) * 4;
                        data[offset] = 255;
                        data[offset + 1] = 255;
                        data[offset + 2] = 255;
                        data[offset + 3] = 255;
                    }
                }
            }
        }
    }

    (width, height, data)
}

/// Tracks presented video-frame timing and reports a smoothed frames-per-second value.
pub struct FpsCounter {
    last: Option<Instant>,
    smoothed: f32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            last: None,
            smoothed: 0.0,
        }
    }

    /// Updates the measurement for a render pass, sampling time only when a new video frame was
    /// presented. Redraw-only passes retain the last reported playback rate.
    pub fn update(&mut self, frame_presented: bool, now: Instant) -> f32 {
        if !frame_presented {
            return self.smoothed;
        }
        let Some(last) = self.last.replace(now) else {
            return self.smoothed;
        };
        let delta = now
            .saturating_duration_since(last)
            .max(Duration::from_micros(1));
        let instantaneous = 1.0 / delta.as_secs_f32();
        // Exponential moving average so the on-screen counter doesn't flicker every frame.
        self.smoothed = if self.smoothed == 0.0 {
            instantaneous
        } else {
            self.smoothed * 0.9 + instantaneous * 0.1
        };
        self.smoothed
    }
}

#[cfg(test)]
mod fps_counter_tests {
    use super::*;

    #[test]
    fn reports_the_rate_of_presented_video_frames() {
        let start = Instant::now();
        let mut counter = FpsCounter::new();

        assert_eq!(counter.update(true, start), 0.0);

        let measured = counter.update(true, start + Duration::from_millis(100));
        assert!((measured - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn retains_the_last_rate_between_presented_frames() {
        let start = Instant::now();
        let mut counter = FpsCounter::new();
        counter.update(true, start);
        let measured = counter.update(true, start + Duration::from_millis(100));

        assert_eq!(
            counter.update(false, start + Duration::from_millis(150)),
            measured
        );

        let next = counter.update(true, start + Duration::from_millis(200));
        assert!((next - 10.0).abs() < f32::EPSILON);
    }
}

fn compile_shader(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::Shader> {
    unsafe {
        let shader = gl
            .create_shader(kind)
            .map_err(|error| gl_error(format!("could not create shader: {error}")))?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(gl_error(format!("shader compile failed: {log}")));
        }
        Ok(shader)
    }
}

fn link_program(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<glow::Program> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|error| gl_error(format!("could not create program: {error}")))?;
        let vertex_shader = compile_shader(gl, glow::VERTEX_SHADER, vertex)?;
        let fragment_shader = compile_shader(gl, glow::FRAGMENT_SHADER, fragment)?;
        gl.attach_shader(program, vertex_shader);
        gl.attach_shader(program, fragment_shader);
        gl.link_program(program);
        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(gl_error(format!("program link failed: {log}")));
        }
        Ok(program)
    }
}

fn gl_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Internal, message)
}

/// A [`GraphicsAdapter`] that uploads frames as a real OpenGL texture via `glow` and draws them
/// to the current window each frame, with an FPS counter overlay in the top-left corner.
pub struct GlWindowAdapter {
    gl: glow::Context,
    context: ContextIdentity,
    owner: ExecutionOwner,
    program: glow::Program,
    rect_ndc_location: Option<glow::UniformLocation>,
    texture_location: Option<glow::UniformLocation>,
    vao: glow::VertexArray,
    video_texture: Option<(u64, VideoDimensions, glow::Texture)>,
    text_texture: glow::Texture,
    text_texture_size: (usize, usize),
    window_size: (u32, u32),
}

impl GlWindowAdapter {
    pub fn new(gl: glow::Context) -> Self {
        let program = link_program(&gl, VIDEO_VERTEX_SHADER, VIDEO_FRAGMENT_SHADER)
            .expect("built-in shaders must compile");
        let rect_ndc_location = unsafe { gl.get_uniform_location(program, "uRectNdc") };
        let texture_location = unsafe { gl.get_uniform_location(program, "uTexture") };

        let (vao, text_texture) = unsafe {
            let vao = gl.create_vertex_array().expect("create vertex array");
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().expect("create buffer");
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let corners: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast(&corners),
                glow::STATIC_DRAW,
            );
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);

            let text_texture = gl.create_texture().expect("create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(text_texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            (vao, text_texture)
        };

        Self {
            gl,
            context: ContextIdentity(1),
            owner: ExecutionOwner(1),
            program,
            rect_ndc_location,
            texture_location,
            vao,
            video_texture: None,
            text_texture,
            text_texture_size: (0, 0),
            window_size: (1, 1),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_size = (width.max(1), height.max(1));
        unsafe {
            self.gl.viewport(0, 0, width as i32, height as i32);
        }
    }

    /// Draws the texture uploaded for `handle` letterboxed/pillarboxed so it is fully contained
    /// within the window (preserving its aspect ratio, matching CSS `object-fit: contain`), then
    /// overlays the FPS counter text in the top-left corner.
    pub fn draw(&mut self, handle: u64, dimensions: VideoDimensions, fps: f32) {
        let gl = &self.gl;
        unsafe {
            gl.clear_color(0.05, 0.05, 0.05, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));

            if let Some((stored_handle, stored_dimensions, texture)) = self.video_texture {
                if stored_handle == handle && stored_dimensions == dimensions {
                    let (x0, y0, x1, y1) = self.contained_rect(dimensions);
                    self.draw_rect(texture, x0, y0, x1, y1);
                }
            }

            let (width, height, data) = rasterize_text(&format!("FPS: {fps:.1}"), 3);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.text_texture));
            if self.text_texture_size == (width, height) {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&data)),
                );
            } else {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&data)),
                );
                self.text_texture_size = (width, height);
            }

            let (window_width, window_height) = self.window_size;
            let ndc_width = 2.0 * width as f32 / window_width as f32;
            let ndc_height = 2.0 * height as f32 / window_height as f32;
            self.draw_rect(
                self.text_texture,
                -1.0,
                1.0 - ndc_height,
                -1.0 + ndc_width,
                1.0,
            );
        }
    }

    /// Returns the NDC rect for `dimensions` scaled to fit entirely within the current window
    /// while preserving its aspect ratio, centered on both axes.
    fn contained_rect(&self, dimensions: VideoDimensions) -> (f32, f32, f32, f32) {
        let (window_width, window_height) = self.window_size;
        let scale = (window_width as f32 / dimensions.width.max(1) as f32)
            .min(window_height as f32 / dimensions.height.max(1) as f32);
        let ndc_width = (dimensions.width as f32 * scale / window_width as f32).min(1.0);
        let ndc_height = (dimensions.height as f32 * scale / window_height as f32).min(1.0);
        (-ndc_width, -ndc_height, ndc_width, ndc_height)
    }

    unsafe fn draw_rect(&self, texture: glow::Texture, x0: f32, y0: f32, x1: f32, y1: f32) {
        let gl = &self.gl;
        unsafe {
            gl.uniform_4_f32(self.rect_ndc_location.as_ref(), x0, y0, x1, y1);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.uniform_1_i32(self.texture_location.as_ref(), 0);
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }
}

impl GraphicsAdapter for GlWindowAdapter {
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
        _destination: &zvidlib::FrameDestination<'_>,
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
            .ok_or_else(|| gl_error("the source frame has no planes"))?;
        let dimensions = destination.dimensions();

        let (texture, reuse_storage) = match self.video_texture {
            Some((_, stored_dimensions, texture)) if stored_dimensions == dimensions => {
                (texture, true)
            }
            Some((_, _, texture)) => {
                unsafe { self.gl.delete_texture(texture) };
                (self.create_video_texture(), false)
            }
            None => (self.create_video_texture(), false),
        };

        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            if reuse_storage {
                // Reusing the previous frame's texture storage avoids a driver-side
                // reallocation on every frame, which mattered for keeping up with the source
                // video's frame rate.
                self.gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    dimensions.width as i32,
                    dimensions.height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&plane.data)),
                );
            } else {
                self.gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA as i32,
                    dimensions.width as i32,
                    dimensions.height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&plane.data)),
                );
            }
        }
        self.video_texture = Some((destination.handle(), dimensions, texture));
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
            "GlWindowAdapter does not implement readback",
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
            "GlWindowAdapter does not implement copy",
        ))
    }

    fn delete(&mut self, resource: GraphicsResource) -> Result<()> {
        if let Some((handle, _, texture)) = self.video_texture {
            if handle == resource.handle() {
                unsafe { self.gl.delete_texture(texture) };
                self.video_texture = None;
            }
        }
        Ok(())
    }
}

impl GlWindowAdapter {
    fn create_video_texture(&self) -> glow::Texture {
        unsafe {
            let texture = self.gl.create_texture().expect("create texture");
            self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            texture
        }
    }
}

fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and any bit pattern is valid, so reinterpreting a `&[f32]`
    // as `&[u8]` with the corresponding length is sound.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
