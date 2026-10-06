//! Per-stage access to the VP8 decoder for the criterion benchmark suite
//! (issue #568). Internal and unstable; see `benches/vp8_decode.rs`.
//!
//! Each stage runs the decoder's own kernels - the same functions
//! reconstruction calls, on whichever instruction set
//! [`crate::simd::set_override`] selects - over a frame's worth of
//! deterministic synthetic input, and returns the bytes it produced so the
//! benchmark can hold every instruction set to the scalar result before timing
//! it.

use super::decoder::Decoder;
use super::frame_encoder::FrameEncoder;
use super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::predict::{
    Edges, Plane, idct_add, idct_dc_add, inverse_walsh, predict_block, predict_inter,
    predict_subblock,
};
use super::tables::{BILINEAR_FILTERS, SIXTAP_FILTERS, TM_PRED};
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

/// Dequantized coefficients of one block, mostly small, with the high
/// frequencies usually zero as after real quantization.
fn coefficients(rng: &mut Rng) -> [i16; 16] {
    let mut block = [0i16; 16];
    let used = 1 + rng.below(16) as usize;
    for coefficient in &mut block[..used] {
        *coefficient = rng.below(160) as i16 - 80;
    }
    block
}

/// The synthetic inputs of the per-stage groups, for one frame of
/// `width` x `height` (multiples of 16).
pub struct Vp8StageInputs {
    width: usize,
    height: usize,
    luma: Plane,
    chroma: Plane,
    /// One Y2 block and 24 residual blocks per macroblock; every third
    /// residual block has only a DC coefficient.
    residuals: Vec<[i16; 16]>,
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
        let macroblocks = (width / 16) * (height / 16);
        let residuals = (0..macroblocks * 25)
            .map(|index| {
                let mut block = coefficients(&mut rng);
                if index % 3 == 0 {
                    block[1..].fill(0);
                }
                block
            })
            .collect();
        let motion = (0..macroblocks)
            .map(|_| (rng.below(257) as i32 - 128, rng.below(257) as i32 - 128))
            .collect();
        Self {
            width,
            height,
            luma,
            chroma,
            residuals,
            motion,
        }
    }

    fn macroblocks(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        let columns = self.width / 16;
        (0..columns * (self.height / 16))
            .map(move |index| (index, index % columns, index / columns))
    }

    /// Inverse Walsh-Hadamard of each macroblock's Y2 block, then the inverse
    /// DCT (or its DC-only shortcut) of its 16 luma and 8 chroma blocks added
    /// to the planes, as reconstruction does. Returns the planes.
    #[must_use]
    pub fn inverse_transforms(&self) -> Vec<u8> {
        let mut luma = self.luma.clone();
        let mut chroma = self.chroma.clone();
        let mut dc_sum = 0i32;
        for (index, mb_x, mb_y) in self.macroblocks() {
            let blocks = &self.residuals[index * 25..(index + 1) * 25];
            let dc = inverse_walsh(&blocks[24]);
            dc_sum = dc_sum.wrapping_add(dc.iter().map(|&value| i32::from(value)).sum::<i32>());
            for (block, coefficients) in blocks[..24].iter().enumerate() {
                let (plane, x, y) = if block < 16 {
                    (
                        &mut luma,
                        mb_x * 16 + (block & 3) * 4,
                        mb_y * 16 + (block >> 2) * 4,
                    )
                } else {
                    let within = (block - 16) & 3;
                    (
                        &mut chroma,
                        mb_x * 8 + (within & 1) * 4,
                        mb_y * 8 + (within >> 1) * 4,
                    )
                };
                let stride = plane.width;
                let offset = y * stride + x;
                if coefficients[1..].iter().all(|&value| value == 0) {
                    idct_dc_add(coefficients[0], &mut plane.data, offset, stride);
                } else {
                    idct_add(coefficients, &mut plane.data, offset, stride);
                }
            }
        }
        let mut out = luma.data;
        out.extend(chroma.data);
        out.extend(dc_sum.to_le_bytes());
        out
    }

    /// Motion-compensated prediction of every macroblock - a 16x16 luma block
    /// and two 8x8 chroma blocks, with every fourth macroblock split into
    /// 4x4 blocks as `SPLITMV` codes them - through the six-tap or the
    /// bilinear filters. Returns the predicted planes.
    #[must_use]
    pub fn inter_prediction(&self, bilinear: bool) -> Vec<u8> {
        let filters = if bilinear {
            &BILINEAR_FILTERS
        } else {
            &SIXTAP_FILTERS
        };
        let mut luma = Plane::new(self.width, self.height);
        let mut chroma = Plane::new(self.width / 2, self.height / 2);
        for (index, mb_x, mb_y) in self.macroblocks() {
            let (mv_x, mv_y) = self.motion[index];
            if index % 4 == 3 {
                for block in 0..16 {
                    let (bx, by) = (block & 3, block >> 2);
                    predict_inter(
                        &self.luma,
                        &mut luma,
                        mb_x * 16 + bx * 4,
                        mb_y * 16 + by * 4,
                        4,
                        4,
                        mv_x + (block as i32 * 5) % 7,
                        mv_y - (block as i32 * 3) % 5,
                        filters,
                    );
                }
                for block in 0..4 {
                    predict_inter(
                        &self.chroma,
                        &mut chroma,
                        mb_x * 8 + (block & 1) * 4,
                        mb_y * 8 + (block >> 1) * 4,
                        4,
                        4,
                        mv_x / 2 + block as i32,
                        mv_y / 2 - block as i32,
                        filters,
                    );
                }
            } else {
                predict_inter(
                    &self.luma,
                    &mut luma,
                    mb_x * 16,
                    mb_y * 16,
                    16,
                    16,
                    mv_x,
                    mv_y,
                    filters,
                );
                predict_inter(
                    &self.chroma,
                    &mut chroma,
                    mb_x * 8,
                    mb_y * 8,
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

    /// Intra prediction of every macroblock: TM over the 16x16 luma block and
    /// an 8x8 chroma block, and the 16 luma subblocks in each of the ten
    /// subblock modes in turn. Returns the predicted planes.
    #[must_use]
    pub fn intra_prediction(&self) -> Vec<u8> {
        let mut luma = self.luma.clone();
        let mut chroma = self.chroma.clone();
        let mut subblocks = Plane::new(self.width, self.height);
        for (index, mb_x, mb_y) in self.macroblocks() {
            let stride = self.luma.width;
            let origin = mb_y * 16 * stride + mb_x * 16;
            let luma_edges = macroblock_edges::<16, 17>(&self.luma, mb_x, mb_y);
            predict_block(
                TM_PRED,
                &luma_edges,
                true,
                true,
                &mut luma.data,
                origin,
                stride,
            );
            let chroma_edges = macroblock_edges::<8, 9>(&self.chroma, mb_x, mb_y);
            let chroma_stride = self.chroma.width;
            predict_block(
                TM_PRED,
                &chroma_edges,
                true,
                true,
                &mut chroma.data,
                mb_y * 8 * chroma_stride + mb_x * 8,
                chroma_stride,
            );
            for block in 0..16 {
                let offset = origin + (block >> 2) * 4 * stride + (block & 3) * 4;
                let x = mb_x * 16 + (block & 3) * 4;
                let y = mb_y * 16 + (block >> 2) * 4;
                let (above, left) = edges::<4, 9>(&self.luma, x, y);
                let mode = ((index + block) % 10) as u8;
                predict_subblock(mode, &above, &left, &mut subblocks.data, offset, stride);
            }
        }
        let mut out = luma.data;
        out.extend(chroma.data);
        out.extend(subblocks.data);
        out
    }

    /// The normal or the simple loop filter over the whole frame, every
    /// macroblock with inner edges and a level that varies across the frame.
    /// Returns the filtered planes.
    #[must_use]
    pub fn loop_filter(&self, simple: bool) -> Vec<u8> {
        let mut planes = [self.luma.clone(), self.chroma.clone(), self.chroma.clone()];
        let macroblocks: Vec<MacroblockFilter> = self
            .macroblocks()
            .map(|(index, _, _)| MacroblockFilter {
                level: 8 + (index % 48) as u8,
                inner_edges: true,
            })
            .collect();
        filter_frame(
            &mut planes,
            &macroblocks,
            self.width / 16,
            FrameFilter {
                simple,
                sharpness: 0,
                key_frame: false,
            },
        );
        let [luma, u, v] = planes;
        let mut out = luma.data;
        out.extend(u.data);
        out.extend(v.data);
        out
    }
}

/// The `A` samples above the `N`x`N` block at `(x, y)`, from the one above-left
/// on, and the `N` to its left, with coordinates clamped to the plane. That is
/// enough of a prediction edge for a timing input.
fn edges<const N: usize, const A: usize>(plane: &Plane, x: usize, y: usize) -> ([u8; A], [u8; N]) {
    let sample = |x: usize, y: usize| {
        let x = x.saturating_sub(1).min(plane.width - 1);
        let y = y.saturating_sub(1).min(plane.height - 1);
        plane.data[y * plane.width + x]
    };
    (
        std::array::from_fn(|i| sample(x + i, y)),
        std::array::from_fn(|i| sample(x, y + i + 1)),
    )
}

fn macroblock_edges<const N: usize, const A: usize>(
    plane: &Plane,
    mb_x: usize,
    mb_y: usize,
) -> Edges<N, A> {
    let (above, left) = edges::<N, A>(plane, mb_x * N, mb_y * N);
    Edges { above, left }
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
