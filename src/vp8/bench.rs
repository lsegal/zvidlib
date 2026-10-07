//! Per-stage access to the VP8 encoder for `benches/vp8_encode.rs`.
//!
//! Internal and unstable. The encoder's kernels are private to the crate, and a
//! benchmark is a separate crate, so each function here runs one kernel over a
//! whole frame through exactly the entry point the encoder calls, which is what
//! makes its time comparable with the whole-frame groups. Every function
//! returns, or leaves in its output plane, something that depends on every
//! value the kernel produced, so the bench's bit-exactness guard compares the
//! kernel's real output across instruction sets.

use super::frame_encoder;
use super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::predict::{self, Plane};
use super::tables::{AC_Q_LOOKUP, DC_Q_LOOKUP, SIXTAP_FILTERS, TM_PRED};

/// An 8-bit plane: luma whose dimensions are whole macroblocks, or the
/// half-size chroma of one.
#[derive(Clone, Debug)]
pub struct BenchPlane(Plane);

impl BenchPlane {
    /// Wraps `data`, `width` x `height` samples with no row padding.
    ///
    /// # Panics
    ///
    /// Panics unless both dimensions are nonzero multiples of 8 and `data`
    /// holds exactly `width * height` samples. The functions below that work a
    /// macroblock at a time cover the whole 16x16 macroblocks of a plane.
    #[must_use]
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert!(
            width > 0 && height > 0 && width % 8 == 0 && height % 8 == 0,
            "a bench plane is whole 8x8 blocks, not {width}x{height}"
        );
        assert_eq!(data.len(), width * height, "{width}x{height} plane");
        Self(Plane {
            data,
            width,
            height,
        })
    }

    /// The plane's samples.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.0.data
    }

    fn macroblocks(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (columns, rows) = (self.0.width / 16, self.0.height / 16);
        (0..rows).flat_map(move |y| (0..columns).map(move |x| (x, y)))
    }

    fn blocks(&self) -> impl Iterator<Item = usize> + use<> {
        let (width, height) = (self.0.width, self.0.height);
        (0..height / 4).flat_map(move |y| (0..width / 4).map(move |x| y * 4 * width + x * 4))
    }
}

/// Folds `values` into `digest` order-sensitively.
fn fold(digest: u64, values: &[i16]) -> u64 {
    values.iter().fold(digest, |digest, &value| {
        digest.rotate_left(5) ^ u64::from(value as u16)
    })
}

/// The SAD of every macroblock of `source` against `reference` displaced by
/// whole samples, as the motion search's whole-sample steps measure it;
/// macroblocks the displacement takes past the reference's edge take the
/// clamped scalar path, as they do in a search.
#[must_use]
pub fn sad(source: &BenchPlane, reference: &BenchPlane, dx: i32, dy: i32) -> u64 {
    source
        .macroblocks()
        .map(|(x, y)| {
            u64::from(frame_encoder::sad16_full(
                &source.0,
                &reference.0,
                x * 16,
                y * 16,
                dx,
                dy,
            ))
        })
        .sum()
}

/// The 16x16 SATD of every macroblock of `source` against `prediction`, as
/// mode decision measures each candidate.
#[must_use]
pub fn satd(source: &BenchPlane, prediction: &BenchPlane) -> u64 {
    source
        .macroblocks()
        .map(|(x, y)| {
            let origin = y * 16 * source.0.width + x * 16;
            u64::from(frame_encoder::satd::<16>(&source.0, &prediction.0, origin))
        })
        .sum()
}

/// The residual and forward DCT of every 4x4 block.
#[must_use]
pub fn forward_dct(source: &BenchPlane, prediction: &BenchPlane) -> u64 {
    source.blocks().fold(0, |digest, offset| {
        fold(
            digest,
            &frame_encoder::residual_dct(&source.0, &prediction.0, offset),
        )
    })
}

/// The forward Walsh-Hadamard transform of each block of luma DC
/// coefficients.
#[must_use]
pub fn forward_walsh(blocks: &[[i16; 16]]) -> u64 {
    blocks.iter().fold(0, |digest, block| {
        fold(digest, &frame_encoder::forward_walsh(block))
    })
}

/// Quantizes each block at quantizer index `q` with the luma factors.
#[must_use]
pub fn quantize(blocks: &[[i16; 16]], q: u8) -> u64 {
    let q = usize::from(q.min(127));
    let factors = [DC_Q_LOOKUP[q], AC_Q_LOOKUP[q]];
    blocks.iter().fold(0, |digest, block| {
        let (levels, dequantized) = frame_encoder::quantize(block, factors, 0);
        fold(fold(digest, &levels), &dequantized)
    })
}

/// Adds the inverse DCT of `blocks[i % blocks.len()]` to the `i`th 4x4 block
/// of `plane`.
pub fn inverse_dct(plane: &mut BenchPlane, blocks: &[[i16; 16]]) {
    let stride = plane.0.width;
    for (index, offset) in plane.blocks().enumerate() {
        predict::idct_add(
            &blocks[index % blocks.len()],
            &mut plane.0.data,
            offset,
            stride,
        );
    }
}

/// The inverse Walsh-Hadamard transform of each Y2 block.
#[must_use]
pub fn inverse_walsh(blocks: &[[i16; 16]]) -> u64 {
    blocks.iter().fold(0, |digest, block| {
        fold(digest, &predict::inverse_walsh(block))
    })
}

/// Predicts every macroblock of `output` from `reference` displaced by
/// `(mv_x, mv_y)` in eighth samples, as luma vectors reach the predictor,
/// with the six-tap filters.
pub fn sixtap(reference: &BenchPlane, output: &mut BenchPlane, mv_x: i32, mv_y: i32) {
    for (x, y) in reference.macroblocks() {
        predict::predict_inter(
            &reference.0,
            &mut output.0,
            x * 16,
            y * 16,
            16,
            16,
            mv_x,
            mv_y,
            &SIXTAP_FILTERS,
        );
    }
}

/// Predicts every macroblock of `plane` with `TM_PRED` from the edges of the
/// macroblocks before it.
pub fn tm_predict(plane: &mut BenchPlane) {
    let stride = plane.0.width;
    for (x, y) in plane.macroblocks() {
        let edges = predict::macroblock_edges::<16, 21>(&plane.0, x, y);
        predict::predict_block(
            TM_PRED,
            &edges,
            y > 0,
            x > 0,
            &mut plane.0.data,
            y * 16 * stride + x * 16,
            stride,
        );
    }
}

/// Loop-filters the frame `planes` (luma, then two half-size chroma planes)
/// with every macroblock at `level`, with its inner edges, under the normal
/// filter or the `simple` one.
pub fn loop_filter(planes: &mut [BenchPlane; 3], level: u8, simple: bool) {
    let mb_cols = planes[0].0.width / 16;
    let count = mb_cols * (planes[0].0.height / 16);
    let macroblocks = vec![
        MacroblockFilter {
            level,
            inner_edges: true,
        };
        count
    ];
    let frame = FrameFilter {
        simple,
        sharpness: 0,
        key_frame: false,
    };
    // Moved out and back rather than cloned, so the group times the filter
    // and not a frame-sized copy.
    let mut frame_planes =
        [0, 1, 2].map(|index| std::mem::replace(&mut planes[index].0, Plane::new(0, 0)));
    filter_frame(&mut frame_planes, &macroblocks, mb_cols, frame);
    for (plane, filtered) in planes.iter_mut().zip(frame_planes) {
        plane.0 = filtered;
    }
}
