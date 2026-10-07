//! Benchmark-only access to the VP9 decoder's individual stages.
//!
//! `crate::vp9_dec` is a private module and criterion benchmarks are a
//! separate crate, so `crates/zvidlib-vp9-decoder/benches/vp9_decode.rs` cannot otherwise reach the
//! inverse transforms, the convolution, the intra predictors or the loop
//! filter on their own, nor a decode that stops at the YUV picture rather
//! than paying for the RGBA conversion the public decoder ends with, which is
//! a dispatch site of its own (`yuv_to_rgba`).
//!
//! Each [`Stage`] owns deterministic inputs built once, and
//! [`Stage::run`] does the stage's work over a whole plane through the same
//! entry points the decoder calls, so it follows [`crate::simd::set_override`]
//! like a real decode. `run` returns the bytes that identify its result,
//! because the benchmark compares those across instruction sets before timing
//! anything. This module is `#[doc(hidden)]` and not part of the stable API.

use crate::Limits;
use crate::vp9_dec::Decoder;
use crate::vp9_dec::loopfilter::{Pixels, Thresholds, thresholds};
use crate::vp9_dec::recon::{
    self, D45_PRED, D63_PRED, D117_PRED, D135_PRED, D207_PRED, DC_PRED, DCT_DCT, IntraEdges,
    TM_PRED,
};
use crate::vp9_dec::tables::FILTER_KERNELS;

/// `tx_type` with the ADST in both directions.
const ADST_ADST: u8 = 3;
/// The D153 intra mode, which the reconstruction handles as its default arm.
const D153_PRED: u8 = 6;

/// An inverse transform family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformKind {
    /// The DCT in both directions, at every size from 4x4 to 32x32.
    Dct,
    /// The ADST in both directions, at 4x4, 8x8 or 16x16.
    Adst,
    /// The 4x4 Walsh-Hadamard transform of lossless frames.
    Wht,
}

/// A sub-pixel interpolation filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpFilter {
    Regular,
    Smooth,
    Sharp,
    Bilinear,
}

/// A family of intra predictors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntraKind {
    /// DC prediction from both edges.
    Dc,
    /// TrueMotion.
    Tm,
    /// The six directional predictors, in turn.
    Directional,
}

/// A loop filter width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopFilterTaps {
    /// `filter4`, on every 8x8 edge.
    Four,
    /// `filter8`, on every 8x8 edge.
    Eight,
    /// The 16-wide filter, on every 16x16 edge.
    Sixteen,
}

/// Deterministic bytes and values for the generated inputs.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn signed(&mut self, span: i32) -> i32 {
        (self.next() % (2 * span as u32 + 1)) as i32 - span
    }
}

/// A plane of smooth gradients with a little noise and some flat regions,
/// so prediction and filtering see something like picture content.
fn picture(rng: &mut Lcg, width: usize, height: usize) -> Vec<u8> {
    let mut plane = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let block = (x / 16 + y / 16) % 5;
            let base = ((x * 3 + y * 2) % 256) as i32;
            let noise = if block == 0 {
                0
            } else {
                rng.signed(block as i32)
            };
            plane[y * width + x] = (base + noise).clamp(0, 255) as u8;
        }
    }
    plane
}

/// A cheap 64-bit digest, so a whole decode can be compared across
/// instruction sets without returning megabytes per iteration.
fn digest(bytes: &[u8]) -> [u8; 8] {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        hash = (hash ^ u64::from_le_bytes(word))
            .wrapping_mul(0x100_0000_01b3)
            .rotate_left(29);
    }
    hash.to_le_bytes()
}

struct TransformBlock {
    offset: usize,
    coefficients: Vec<i32>,
}

struct InterBlock {
    origin: usize,
    dest: usize,
    x_frac: i32,
    y_frac: i32,
}

struct IntraBlock {
    offset: usize,
    bs: usize,
    mode: u8,
    edges: IntraEdges,
}

struct Edge {
    start: isize,
    vertical: bool,
    thresholds: Thresholds,
}

enum Work {
    Transform {
        tx_size: u8,
        tx_type: u8,
        lossless: bool,
        blocks: Vec<TransformBlock>,
    },
    Inter {
        reference: Vec<u8>,
        reference_stride: usize,
        filter: usize,
        block: usize,
        compound: bool,
        blocks: Vec<InterBlock>,
    },
    Intra {
        blocks: Vec<IntraBlock>,
    },
    LoopFilter {
        taps: LoopFilterTaps,
        edges: Vec<Edge>,
    },
    Decode {
        chunks: Vec<Vec<u8>>,
    },
}

/// One stage's work over a whole plane, with its inputs built once.
pub struct Stage {
    width: usize,
    height: usize,
    plane: Vec<u8>,
    work: Work,
}

impl Stage {
    /// Every `size`x`size` block of a `width`x`height` plane inverse
    /// transformed and added to its prediction, with every coefficient
    /// present (the full transform rather than an end-of-block shortcut).
    ///
    /// # Panics
    ///
    /// If `size` is not one `kind` supports.
    #[must_use]
    pub fn inverse_transform(
        kind: TransformKind,
        size: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let tx_size = match size {
            4 => 0,
            8 => 1,
            16 => 2,
            32 if kind == TransformKind::Dct => 3,
            _ => panic!("no {size}x{size} {kind:?}"),
        };
        assert!(kind != TransformKind::Wht || size == 4, "the WHT is 4x4");
        let mut rng = Lcg(0x7470_0000 + size as u64);
        let plane = picture(&mut rng, width, height);
        let mut blocks = Vec::new();
        for y in (0..height - size + 1).step_by(size) {
            for x in (0..width - size + 1).step_by(size) {
                // Coefficients that fall off with frequency, the shape a
                // quantized residual has.
                let coefficients = (0..size * size)
                    .map(|i| {
                        let (row, column) = (i / size, i % size);
                        rng.signed(256 / (1 + row + column) as i32)
                    })
                    .collect();
                blocks.push(TransformBlock {
                    offset: y * width + x,
                    coefficients,
                });
            }
        }
        let tx_type = match kind {
            TransformKind::Adst => ADST_ADST,
            _ => DCT_DCT,
        };
        Self {
            width,
            height,
            plane,
            work: Work::Transform {
                tx_size,
                tx_type,
                lossless: kind == TransformKind::Wht,
                blocks,
            },
        }
    }

    /// Every `block`x`block` block of a `width`x`height` plane predicted
    /// from a reference at a sub-pixel position, cycling through every
    /// phase (whole-pixel, horizontal-only, vertical-only and both). With
    /// `compound`, each block is predicted twice and averaged, as a
    /// two-reference block is.
    #[must_use]
    pub fn inter_prediction(
        filter: InterpFilter,
        compound: bool,
        block: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let mut rng = Lcg(0x6d63_0000 + block as u64);
        let border = 16;
        let reference_stride = width + 2 * border;
        let reference = picture(&mut rng, reference_stride, height + 2 * border);
        let plane = vec![0u8; width * height];
        let mut blocks = Vec::new();
        let mut phase = 0;
        for y in (0..height - block + 1).step_by(block) {
            for x in (0..width - block + 1).step_by(block) {
                phase += 7;
                let dx = rng.signed(4);
                let dy = rng.signed(4);
                blocks.push(InterBlock {
                    origin: (border as i32 + y as i32 + dy) as usize * reference_stride
                        + (border as i32 + x as i32 + dx) as usize,
                    dest: y * width + x,
                    x_frac: phase % 16,
                    y_frac: (phase / 16) % 16,
                });
            }
        }
        Self {
            width,
            height,
            plane,
            work: Work::Inter {
                reference,
                reference_stride,
                filter: filter as usize,
                block,
                compound,
                blocks,
            },
        }
    }

    /// A `width`x`height` plane covered by intra predictions of `kind`, in
    /// 32x32 tiles that cycle through every block size from 4x4 to 32x32.
    #[must_use]
    pub fn intra_prediction(kind: IntraKind, width: usize, height: usize) -> Self {
        let mut rng = Lcg(0x6970_0000);
        let plane = picture(&mut rng, width, height);
        let directional = [
            D45_PRED, D135_PRED, D117_PRED, D153_PRED, D207_PRED, D63_PRED,
        ];
        let mut blocks = Vec::new();
        let mut tile = 0usize;
        for ty in (0..height - 31).step_by(32) {
            for tx in (0..width - 31).step_by(32) {
                let bs = 4 << (tile % 4);
                tile += 1;
                for y in (ty..ty + 32).step_by(bs) {
                    for x in (tx..tx + 32).step_by(bs) {
                        let mode = match kind {
                            IntraKind::Dc => DC_PRED,
                            IntraKind::Tm => TM_PRED,
                            IntraKind::Directional => directional[blocks.len() % 6],
                        };
                        let edges = IntraEdges {
                            above: core::array::from_fn(|i| {
                                plane[(x + i) % width + (y + 7) % height * width]
                            }),
                            left: core::array::from_fn(|i| {
                                plane[(y + i) % height * width + (x + 3) % width]
                            }),
                        };
                        blocks.push(IntraBlock {
                            offset: y * width + x,
                            bs,
                            mode,
                            edges,
                        });
                    }
                }
            }
        }
        Self {
            width,
            height,
            plane,
            work: Work::Intra { blocks },
        }
    }

    /// Every vertical edge and then every horizontal edge of a
    /// `width`x`height` plane filtered with `taps`, 8x8 edges for the
    /// narrow filters and 16x16 ones for the wide filter, at filter levels
    /// that vary from edge to edge.
    #[must_use]
    pub fn loop_filter(taps: LoopFilterTaps, width: usize, height: usize) -> Self {
        let mut rng = Lcg(0x6c66_0000);
        let plane = picture(&mut rng, width, height);
        let table = thresholds(0);
        let spacing = if taps == LoopFilterTaps::Sixteen {
            16
        } else {
            8
        };
        let mut edges = Vec::new();
        for vertical in [true, false] {
            for y in (0..height).step_by(spacing) {
                for x in (0..width).step_by(spacing) {
                    if (vertical && x < spacing) || (!vertical && y < spacing) {
                        continue;
                    }
                    edges.push(Edge {
                        start: (y * width + x) as isize,
                        vertical,
                        thresholds: table[16 + rng.next() as usize % 48],
                    });
                }
            }
        }
        Self {
            width,
            height,
            plane,
            work: Work::LoopFilter { taps, edges },
        }
    }

    /// Whole-frame decode of `chunks`, the samples of one VP9 stream, to YUV
    /// pictures. Returns a digest of every shown picture.
    #[must_use]
    pub fn decode(chunks: Vec<Vec<u8>>) -> Self {
        Self {
            width: 0,
            height: 0,
            plane: Vec::new(),
            work: Work::Decode { chunks },
        }
    }

    /// Runs the stage once and returns the bytes it produced: the filtered
    /// or reconstructed plane, or a digest of each decoded picture.
    ///
    /// # Panics
    ///
    /// If a [`Stage::decode`] stream fails to decode.
    #[must_use]
    pub fn run(&self) -> Vec<u8> {
        let mut plane = self.plane.clone();
        let stride = self.width;
        match &self.work {
            Work::Transform {
                tx_size,
                tx_type,
                lossless,
                blocks,
            } => {
                for block in blocks {
                    recon::inverse_transform_add(
                        &block.coefficients,
                        &mut plane[block.offset..],
                        stride,
                        *tx_size,
                        *tx_type,
                        block.coefficients.len(),
                        *lossless,
                    );
                }
            }
            Work::Inter {
                reference,
                reference_stride,
                filter,
                block,
                compound,
                blocks,
            } => {
                let mut temp = Box::new([0u8; 64 * 135]);
                for b in blocks {
                    for (pass, x_frac, y_frac) in [(0, b.x_frac, b.y_frac), (1, b.y_frac, b.x_frac)]
                    {
                        if pass == 1 && !compound {
                            break;
                        }
                        recon::convolve(
                            reference,
                            b.origin,
                            *reference_stride,
                            &mut plane[b.dest..],
                            stride,
                            *block,
                            *block,
                            &FILTER_KERNELS[*filter],
                            x_frac,
                            16,
                            y_frac,
                            16,
                            pass == 1,
                            &mut temp,
                        );
                    }
                }
            }
            Work::Intra { blocks } => {
                for block in blocks {
                    recon::predict_intra(
                        &mut plane[block.offset..],
                        stride,
                        block.bs,
                        block.mode,
                        &block.edges,
                        true,
                        true,
                    );
                }
            }
            Work::LoopFilter { taps, edges } => {
                let mut pixels = Pixels {
                    data: &mut plane,
                    stride,
                    isa: super::active_isa(),
                };
                let count = if *taps == LoopFilterTaps::Sixteen {
                    16
                } else {
                    8
                };
                for edge in edges {
                    let (step, along) = if edge.vertical {
                        (1, stride as isize)
                    } else {
                        (stride as isize, 1)
                    };
                    let count = count.min(if edge.vertical {
                        self.height - edge.start as usize / stride
                    } else {
                        self.width - edge.start as usize % stride
                    });
                    match taps {
                        LoopFilterTaps::Four => {
                            pixels.lpf4(edge.start, step, along, count, edge.thresholds);
                        }
                        LoopFilterTaps::Eight => {
                            pixels.lpf8(edge.start, step, along, count, edge.thresholds);
                        }
                        LoopFilterTaps::Sixteen => {
                            pixels.lpf16(edge.start, step, along, count, edge.thresholds);
                        }
                    }
                }
            }
            Work::Decode { chunks } => {
                let mut decoder = Decoder::new(Limits::default());
                let mut digests = Vec::new();
                for chunk in chunks {
                    if let Some(picture) = decoder.decode_chunk(chunk).expect("the stream decodes")
                    {
                        for plane in &picture.planes {
                            digests.extend_from_slice(&digest(plane));
                        }
                    }
                }
                return digests;
            }
        }
        plane
    }
}
