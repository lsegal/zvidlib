//! Per-stage access to the VP8 decoder for `crates/zvidlib-vp8/benches/vp8_decode.rs` (issue
//! #568).
//!
//! Internal and unstable. The counterpart to [`super::bench`], which times the
//! kernels the encoder and the decoder share: these are the stages only the
//! decoder takes - subblock intra prediction, the DC-only inverse DCT, the
//! bilinear filters of bitstream versions 1 to 3 and the simple loop filter -
//! plus whole-frame decoding. Each stage runs the decoder's own entry point,
//! on whichever instruction set [`zvidlib_core::simd::set_override`] selects, over a
//! frame's worth of deterministic synthetic input, and returns the bytes it
//! produced so the benchmark can hold every instruction set to the scalar
//! result before timing it.

use super::decoder::Decoder;
use super::frame_encoder::FrameEncoder;
use super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::predict::{Plane, idct_dc_add_row, predict_inter, predict_subblock};
use super::tables::BILINEAR_FILTERS;
use crate::{Limits, Result};

/// A small deterministic generator, so every instruction set and every run
/// sees the same input.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// A smooth, textured plane, so that loop-filter thresholds pass on some
/// edges and fail on others as they do in real content.
fn textured_plane(width: usize, height: usize, rng: &mut Rng) -> Plane {
    let mut plane = Plane::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let wave = ((x * 7 + y * 3) % 64) as u64 + ((x / 16 + y / 16) % 4) as u64 * 24;
            plane.data[y * width + x] = (wave + rng.below(6) + 64) as u8;
        }
    }
    plane
}

/// The synthetic inputs of the per-stage groups, for one frame of
/// `width` x `height` (multiples of 16).
pub struct Vp8StageInputs {
    width: usize,
    height: usize,
    luma: Plane,
    chroma: Plane,
    /// One DC coefficient per 4x4 block of the frame, luma then chroma.
    dc: Vec<i16>,
    /// One luma motion vector per macroblock, in eighth samples.
    motion: Vec<(i32, i32)>,
}

impl Vp8StageInputs {
    /// Builds the inputs for a `width` x `height` frame.
    ///
    /// # Panics
    ///
    /// If either dimension is not a positive multiple of 16.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0 && width % 16 == 0 && height % 16 == 0);
        let mut rng = Rng(0x568);
        let luma = textured_plane(width, height, &mut rng);
        let chroma = textured_plane(width / 2, height / 2, &mut rng);
        let blocks = (width / 4) * (height / 4) + 2 * (width / 8) * (height / 8);
        let dc = (0..blocks).map(|_| rng.below(640) as i16 - 320).collect();
        let motion = (0..(width / 16) * (height / 16))
            .map(|_| (rng.below(257) as i32 - 128, rng.below(257) as i32 - 128))
            .collect();
        Self {
            width,
            height,
            luma,
            chroma,
            dc,
            motion,
        }
    }

    fn macroblocks(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        let columns = self.width / 16;
        (0..columns * (self.height / 16))
            .map(move |index| (index, index % columns, index / columns))
    }

    /// Every 4x4 luma subblock of the frame predicted from the source's
    /// samples around it, cycling through the ten subblock modes. Returns
    /// the predicted plane.
    #[must_use]
    pub fn subblock_prediction(&self) -> Vec<u8> {
        let plane = &self.luma;
        let mut out = Plane::new(self.width, self.height);
        let sample = |x: usize, y: usize| {
            let x = x.saturating_sub(1).min(plane.width - 1);
            let y = y.saturating_sub(1).min(plane.height - 1);
            plane.data[y * plane.width + x]
        };
        for y in (0..self.height).step_by(4) {
            for x in (0..self.width).step_by(4) {
                let above: [u8; 9] = std::array::from_fn(|i| sample(x + i, y));
                let left: [u8; 4] = std::array::from_fn(|i| sample(x, y + i + 1));
                let mode = ((x / 4 + y / 4) % 10) as u8;
                predict_subblock(
                    mode,
                    &above,
                    &left,
                    &mut out.data,
                    y * self.width + x,
                    self.width,
                );
            }
        }
        out.data
    }

    /// The DC-only inverse DCT added to every 4x4 block of the luma and the
    /// two chroma planes, a row of a macroblock's blocks at a time as the
    /// decoder adds them (four luma or two chroma). Returns the planes.
    #[must_use]
    pub fn dc_inverse_dct(&self) -> Vec<u8> {
        let mut planes = [self.luma.clone(), self.chroma.clone(), self.chroma.clone()];
        let mut dc = self.dc.iter();
        for (index, plane) in planes.iter_mut().enumerate() {
            let stride = plane.width;
            let per_row = if index == 0 { 4 } else { 2 };
            for y in (0..plane.height).step_by(4) {
                for x in (0..plane.width).step_by(4 * per_row) {
                    let dcs: [i16; 4] = std::array::from_fn(|_| *dc.next().unwrap_or(&0));
                    idct_dc_add_row(&dcs[..per_row], &mut plane.data, y * stride + x, stride);
                }
            }
        }
        let [luma, u, v] = planes;
        let mut out = luma.data;
        out.extend(u.data);
        out.extend(v.data);
        out
    }

    /// Bilinear sub-pixel prediction of every macroblock - a 16x16 luma block
    /// and an 8x8 chroma block, with every fourth macroblock split into 4x4
    /// blocks as `SPLITMV` codes them. Returns the predicted planes.
    #[must_use]
    pub fn bilinear_prediction(&self) -> Vec<u8> {
        let filters = &BILINEAR_FILTERS;
        let mut luma = Plane::new(self.width, self.height);
        let mut chroma = Plane::new(self.width / 2, self.height / 2);
        for (index, mb_x, mb_y) in self.macroblocks() {
            let (mv_x, mv_y) = self.motion[index];
            if index % 4 == 3 {
                for block in 0..16 {
                    let (x, y) = (mb_x * 16 + (block & 3) * 4, mb_y * 16 + (block >> 2) * 4);
                    let (dx, dy) = ((block as i32 * 5) % 7, (block as i32 * 3) % 5);
                    predict_inter(
                        &self.luma,
                        &mut luma,
                        x,
                        y,
                        4,
                        4,
                        mv_x + dx,
                        mv_y - dy,
                        filters,
                    );
                }
                for block in 0..4 {
                    let (x, y) = (mb_x * 8 + (block & 1) * 4, mb_y * 8 + (block >> 1) * 4);
                    let shift = block as i32;
                    let (cx, cy) = (mv_x / 2 + shift, mv_y / 2 - shift);
                    predict_inter(&self.chroma, &mut chroma, x, y, 4, 4, cx, cy, filters);
                }
            } else {
                let (x, y) = (mb_x * 16, mb_y * 16);
                predict_inter(&self.luma, &mut luma, x, y, 16, 16, mv_x, mv_y, filters);
                let (x, y) = (mb_x * 8, mb_y * 8);
                predict_inter(
                    &self.chroma,
                    &mut chroma,
                    x,
                    y,
                    8,
                    8,
                    mv_x / 2,
                    mv_y / 2,
                    filters,
                );
            }
        }
        let mut out = luma.data;
        out.extend(chroma.data);
        out
    }

    /// The simple loop filter over the whole frame's luma, every macroblock
    /// with inner edges and a level that varies across the frame. Returns the
    /// filtered luma.
    #[must_use]
    pub fn simple_loop_filter(&self) -> Vec<u8> {
        let mut planes = [self.luma.clone(), self.chroma.clone(), self.chroma.clone()];
        let macroblocks: Vec<MacroblockFilter> = self
            .macroblocks()
            .map(|(index, _, _)| MacroblockFilter {
                level: 8 + (index % 48) as u8,
                inner_edges: true,
            })
            .collect();
        let frame = FrameFilter {
            simple: true,
            sharpness: 0,
            key_frame: false,
        };
        filter_frame(&mut planes, &macroblocks, self.width / 16, frame);
        let [luma, _, _] = planes;
        luma.data
    }
}

/// Encodes `frames` frames of `width` x `height` synthetic content - a
/// textured picture panning by a fraction of a sample each frame, so the
/// inter frames carry sub-sample motion - with the native VP8 encoder: one
/// key frame, then inter frames.
///
/// # Errors
///
/// If the encoder rejects a frame.
pub fn synthetic_stream(width: usize, height: usize, frames: usize) -> Result<Vec<Vec<u8>>> {
    let mut encoder = FrameEncoder::new(width, height);
    let (padded_width, padded_height) = encoder.padded_dimensions();
    let texture = |x: f64, y: f64| {
        let value = 128.0
            + 60.0 * (x * 0.07).sin() * (y * 0.05).cos()
            + 30.0 * ((x + y) * 0.21).sin()
            + 20.0 * ((x * 0.013).floor() + (y * 0.017).floor()).rem_euclid(2.0);
        value.clamp(0.0, 255.0) as u8
    };
    (0..frames)
        .map(|frame| {
            let shift_x = frame as f64 * 1.25;
            let shift_y = frame as f64 * 0.75;
            let plane = |width: usize, height: usize, scale: f64, bias: f64| {
                let mut plane = Plane::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        plane.data[y * width + x] = texture(
                            x as f64 * scale + shift_x + bias,
                            y as f64 * scale + shift_y + bias,
                        );
                    }
                }
                plane
            };
            let source = [
                plane(padded_width, padded_height, 1.0, 0.0),
                plane(padded_width / 2, padded_height / 2, 2.0, 37.0),
                plane(padded_width / 2, padded_height / 2, 2.0, 91.0),
            ];
            encoder.encode(&source, frame == 0, 40)
        })
        .collect()
}

/// Decodes `frames`, VP8 frames of one stream in order, with the software
/// decoder, and returns every shown picture's planes back to back.
///
/// # Errors
///
/// If a frame does not decode.
pub fn decode_pictures(frames: &[&[u8]]) -> Result<Vec<u8>> {
    let mut decoder = Decoder::new(Limits::default());
    let mut out = Vec::new();
    for frame in frames {
        if let Some(picture) = decoder.decode(frame)? {
            for plane in &picture.planes {
                out.extend_from_slice(plane);
            }
        }
    }
    Ok(out)
}
