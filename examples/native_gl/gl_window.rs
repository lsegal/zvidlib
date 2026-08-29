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
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        _ => [0; FONT_HEIGHT],
    }
}

/// The keyboard and pointer controls drawn as an on-screen legend, one line each. Every
/// character used here must be covered by [`glyph`].
pub const CONTROL_LEGEND: &[&str] = &[
    "SPACE  PLAY/PAUSE",
    "LEFT/RIGHT  STEP ONE FRAME",
    "J/L  SEEK -/+ 5 SECONDS",
    "CLICK OR DRAG THE BAR TO SCRUB",
    "H  SHOW/HIDE THIS LEGEND",
];

/// Rasterizes a single line of `text` into a fully opaque RGBA buffer.
fn rasterize_text(text: &str, scale: usize) -> (usize, usize, Vec<u8>) {
    rasterize_lines(&[text], scale, 1.0)
}

/// Rasterizes `lines` into an RGBA buffer, `scale` pixels per font pixel, with a 1-font-pixel
/// margin and a translucent background so the text stays legible over any frame content.
/// `opacity` scales the background and glyph alpha alike so callers can fade the overlay out.
fn rasterize_lines(lines: &[&str], scale: usize, opacity: f32) -> (usize, usize, Vec<u8>) {
    let opacity = opacity.clamp(0.0, 1.0);
    let margin = 1usize;
    let glyph_w = FONT_WIDTH + 1;
    let glyph_h = FONT_HEIGHT + 1;
    let cols = lines
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0);
    let rows = lines.len().max(1);
    let width = (cols * glyph_w + margin * 2) * scale;
    let height = ((rows - 1) * glyph_h + FONT_HEIGHT + margin * 2) * scale;
    let mut data = vec![0_u8; width * height * 4];
    let background_alpha = (140.0 * opacity) as u8;
    let glyph_alpha = (255.0 * opacity) as u8;

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            data[offset] = 0;
            data[offset + 1] = 0;
            data[offset + 2] = 0;
            data[offset + 3] = background_alpha;
        }
    }

    for (line_index, line) in lines.iter().enumerate() {
        for (index, ch) in line.chars().enumerate() {
            let bitmap = glyph(ch);
            for (row, bits) in bitmap.iter().enumerate() {
                for col in 0..FONT_WIDTH {
                    if bits & (1 << (FONT_WIDTH - 1 - col)) == 0 {
                        continue;
                    }
                    let base_x = (margin + index * glyph_w + col) * scale;
                    let base_y = (margin + line_index * glyph_h + row) * scale;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = base_x + dx;
                            let py = base_y + dy;
                            let offset = (py * width + px) * 4;
                            data[offset] = 255;
                            data[offset + 1] = 255;
                            data[offset + 2] = 255;
                            data[offset + 3] = glyph_alpha;
                        }
                    }
                }
            }
        }
    }

    (width, height, data)
}

/// How long the control legend stays fully visible before it starts fading out.
const LEGEND_VISIBLE: Duration = Duration::from_secs(5);
/// How long the control legend takes to fade from fully visible to fully transparent.
const LEGEND_FADE: Duration = Duration::from_millis(1500);

/// Tracks the control legend's visibility: it is shown when playback starts, fades out on its own
/// so it does not permanently cover the video, and can be toggled back on from the keyboard.
pub struct LegendVisibility {
    shown_at: Option<Instant>,
    pinned: bool,
}

impl LegendVisibility {
    /// Starts visible, fading out `LEGEND_VISIBLE` after `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            shown_at: Some(now),
            pinned: false,
        }
    }

    /// Hides the legend when any of it is still visible, otherwise shows it until toggled again.
    pub fn toggle(&mut self, now: Instant) {
        if self.opacity(now) > 0.0 {
            self.shown_at = None;
            self.pinned = false;
        } else {
            self.shown_at = Some(now);
            self.pinned = true;
        }
    }

    /// Returns the legend's opacity at `now`, in `0.0..=1.0`.
    pub fn opacity(&self, now: Instant) -> f32 {
        let Some(shown_at) = self.shown_at else {
            return 0.0;
        };
        if self.pinned {
            return 1.0;
        }
        let elapsed = now.saturating_duration_since(shown_at);
        if elapsed < LEGEND_VISIBLE {
            return 1.0;
        }
        let faded = (elapsed - LEGEND_VISIBLE).as_secs_f32() / LEGEND_FADE.as_secs_f32();
        (1.0 - faded).clamp(0.0, 1.0)
    }
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
    legend_texture: glow::Texture,
    legend_texture_size: (usize, usize),
    legend_opacity: Option<u8>,
    timeline_background: glow::Texture,
    timeline_progress: glow::Texture,
    timeline_hover: glow::Texture,
    window_size: (u32, u32),
}

impl GlWindowAdapter {
    pub fn new(gl: glow::Context) -> Self {
        let program = link_program(&gl, VIDEO_VERTEX_SHADER, VIDEO_FRAGMENT_SHADER)
            .expect("built-in shaders must compile");
        let rect_ndc_location = unsafe { gl.get_uniform_location(program, "uRectNdc") };
        let texture_location = unsafe { gl.get_uniform_location(program, "uTexture") };

        let (
            vao,
            text_texture,
            legend_texture,
            timeline_background,
            timeline_progress,
            timeline_hover,
        ) = unsafe {
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

            let text_texture = overlay_texture(&gl);
            let legend_texture = overlay_texture(&gl);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            (
                vao,
                text_texture,
                legend_texture,
                solid_texture(&gl, [25, 25, 25, 220]),
                solid_texture(&gl, [85, 185, 235, 255]),
                solid_texture(&gl, [255, 255, 255, 235]),
            )
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
            legend_texture,
            legend_texture_size: (0, 0),
            legend_opacity: None,
            timeline_background,
            timeline_progress,
            timeline_hover,
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
    /// overlays the FPS counter text in the top-left corner and, while `legend_opacity` is above
    /// zero, the control legend just above the timeline bar.
    pub fn draw(
        &mut self,
        handle: u64,
        dimensions: VideoDimensions,
        fps: f32,
        progress: f32,
        hover: Option<f32>,
        legend_opacity: f32,
    ) {
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
            self.upload_overlay(
                self.text_texture,
                self.text_texture_size,
                width,
                height,
                &data,
            );
            self.text_texture_size = (width, height);

            let (window_width, window_height) = self.window_size;
            // 14px tall, inset 14px from the bottom edge so the bar and its hover marker stay
            // fully on-screen instead of being clipped by the window border.
            let pixel = 2.0 / window_height as f32;
            let timeline_bottom = -1.0 + pixel * 14.0;
            let timeline_top = timeline_bottom + pixel * 14.0;
            let progress_right = -1.0 + progress.clamp(0.0, 1.0) * 2.0;
            self.draw_rect(
                self.timeline_background,
                -1.0,
                timeline_bottom,
                1.0,
                timeline_top,
            );
            self.draw_rect(
                self.timeline_progress,
                -1.0,
                timeline_bottom,
                progress_right,
                timeline_top,
            );
            if let Some(hover) = hover {
                let x = -1.0 + hover.clamp(0.0, 1.0) * 2.0;
                let half_width = 3.0 * 2.0 / window_width as f32;
                self.draw_rect(
                    self.timeline_hover,
                    x - half_width,
                    timeline_bottom - pixel * 5.0,
                    x + half_width,
                    timeline_top + pixel * 5.0,
                );
            }

            if legend_opacity > 0.0 {
                let quantized = (legend_opacity.clamp(0.0, 1.0) * 255.0) as u8;
                if self.legend_opacity != Some(quantized) {
                    let (legend_width, legend_height, legend_data) =
                        rasterize_lines(CONTROL_LEGEND, 3, legend_opacity);
                    self.upload_overlay(
                        self.legend_texture,
                        self.legend_texture_size,
                        legend_width,
                        legend_height,
                        &legend_data,
                    );
                    self.legend_texture_size = (legend_width, legend_height);
                    self.legend_opacity = Some(quantized);
                }
                let (legend_width, legend_height) = self.legend_texture_size;
                let horizontal_pixel = 2.0 / window_width as f32;
                let left = -1.0 + horizontal_pixel * 14.0;
                let bottom = timeline_top + pixel * 10.0;
                self.draw_rect(
                    self.legend_texture,
                    left,
                    bottom,
                    left + horizontal_pixel * legend_width as f32,
                    bottom + pixel * legend_height as f32,
                );
            } else {
                self.legend_opacity = None;
            }

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

    /// Uploads `data` into `texture`, reusing its storage when `stored_size` already matches the
    /// new size so repeated overlay updates avoid a driver-side reallocation.
    unsafe fn upload_overlay(
        &self,
        texture: glow::Texture,
        stored_size: (usize, usize),
        width: usize,
        height: usize,
        data: &[u8],
    ) {
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            if stored_size == (width, height) {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(data)),
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
                    glow::PixelUnpackData::Slice(Some(data)),
                );
            }
        }
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

/// Creates a NEAREST-filtered, clamped texture for the bitmap-font overlays.
unsafe fn overlay_texture(gl: &glow::Context) -> glow::Texture {
    unsafe {
        let texture = gl.create_texture().expect("create texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
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
        texture
    }
}

unsafe fn solid_texture(gl: &glow::Context, color: [u8; 4]) -> glow::Texture {
    unsafe {
        let texture = gl.create_texture().expect("create texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
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
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            1,
            1,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&color)),
        );
        texture
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

#[cfg(test)]
mod tests {
    use super::*;

    fn is_blank(ch: char) -> bool {
        glyph(ch) == [0; FONT_HEIGHT]
    }

    #[test]
    fn font_covers_the_characters_the_legend_needs() {
        for ch in ('A'..='Z').chain("/-+0123456789.:".chars()) {
            assert!(!is_blank(ch), "{ch:?} must have a bitmap");
        }
        assert!(is_blank(' '), "space must render blank");
    }

    #[test]
    fn every_legend_character_is_drawable() {
        for line in CONTROL_LEGEND {
            for ch in line.chars() {
                assert!(
                    ch == ' ' || !is_blank(ch),
                    "legend line {line:?} uses undrawable {ch:?}"
                );
            }
        }
    }

    #[test]
    fn rasterized_lines_are_sized_for_the_widest_line() {
        let (width, height, data) = rasterize_lines(&["AB", "CDE"], 2, 1.0);
        assert_eq!(width, (3 * (FONT_WIDTH + 1) + 2) * 2);
        assert_eq!(height, ((FONT_HEIGHT + 1) + FONT_HEIGHT + 2) * 2);
        assert_eq!(data.len(), width * height * 4);
    }

    #[test]
    fn opacity_scales_background_and_glyph_alpha() {
        let (width, _, opaque) = rasterize_lines(&["A"], 1, 1.0);
        let (_, _, faded) = rasterize_lines(&["A"], 1, 0.5);
        let (_, _, hidden) = rasterize_lines(&["A"], 1, 0.0);
        // The top-left pixel is background; 'A' lights its top row's middle columns, so the
        // pixel one column past the margin on the first glyph row is lit.
        let glyph_pixel = (width + 2) * 4 + 3;
        assert_eq!(opaque[3], 140);
        assert_eq!(opaque[glyph_pixel], 255);
        assert_eq!(faded[3], 70);
        assert_eq!(faded[glyph_pixel], 127);
        assert_eq!(hidden[3], 0);
        assert_eq!(hidden[glyph_pixel], 0);
    }

    #[test]
    fn fps_overlay_rasterization_is_unchanged() {
        let text = "FPS: 12.3";
        let (width, height, data) = rasterize_text(text, 3);
        assert_eq!(width, (text.chars().count() * (FONT_WIDTH + 1) + 2) * 3);
        assert_eq!(height, (FONT_HEIGHT + 2) * 3);
        assert!(data.chunks_exact(4).any(|pixel| pixel == [255; 4]));
    }

    #[test]
    fn legend_starts_visible_and_fades_out() {
        let start = Instant::now();
        let legend = LegendVisibility::new(start);
        assert_eq!(legend.opacity(start), 1.0);
        assert_eq!(legend.opacity(start + LEGEND_VISIBLE), 1.0);
        let midway = legend.opacity(start + LEGEND_VISIBLE + LEGEND_FADE / 2);
        assert!((0.4..0.6).contains(&midway), "midway opacity was {midway}");
        assert_eq!(legend.opacity(start + LEGEND_VISIBLE + LEGEND_FADE), 0.0);
        assert_eq!(
            legend.opacity(start + LEGEND_VISIBLE + LEGEND_FADE * 10),
            0.0
        );
    }

    #[test]
    fn toggling_hides_a_visible_legend_and_pins_a_hidden_one() {
        let start = Instant::now();
        let mut legend = LegendVisibility::new(start);

        legend.toggle(start);
        assert_eq!(legend.opacity(start), 0.0);

        let shown = start + LEGEND_VISIBLE;
        legend.toggle(shown);
        assert_eq!(legend.opacity(shown), 1.0);
        // Pinned visibility never fades on its own.
        assert_eq!(
            legend.opacity(shown + LEGEND_VISIBLE + LEGEND_FADE * 10),
            1.0
        );

        legend.toggle(shown + LEGEND_VISIBLE);
        assert_eq!(legend.opacity(shown + LEGEND_VISIBLE), 0.0);
    }
}
