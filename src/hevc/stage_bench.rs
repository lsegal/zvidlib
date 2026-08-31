//! Prepared per-stage workloads for the HEVC decoder benchmark.
//!
//! `benches/codec.rs` is an external crate and the HEVC engine
//! (`crate::hevc::engine`) is crate-private, so a benchmark cannot reach the
//! individual decode stages the way it reaches the public AV1 kernels. This
//! module is the narrow surface that closes that gap: one prepared workload per
//! hot stage, each with its input construction in [`HevcStageInputs::new`] and
//! only the kernel under test inside the `run_*` method.
//!
//! Every `run_*` method returns the eight bytes of an FNV-1a fold over all of
//! the samples the stage produced. The bytes exist for the benchmark harness's
//! bit-exactness guard, which compares one instruction set's arm against the
//! scalar arm; folding rather than returning the samples themselves keeps a
//! multi-megabyte allocation out of the timed loop, and the fold still depends
//! on every output sample, so a backend that diverged anywhere changes it.
//!
//! None of these methods pin an instruction set. They call the same entry
//! points the decoder calls, which resolve through
//! [`crate::simd::set_override`], so the caller selects the arm exactly as it
//! does for a whole-frame decode.

use super::engine::deblock::{EdgePos, EdgeQp, EdgeType, SamplePlane, filter_luma_block_edge};
use super::engine::inter_pred::{RefPlane, default_weighted_pred, interp_luma_block};
use super::engine::intra_pred::{
    Component as IntraComponent, IntraPredParams, ReferenceSamples, intra_predict,
};
use super::engine::picture::{Picture, Plane};
use super::engine::sao::{ResolvedSao, ResolvedSaoComponent, apply_sao_picture};
use super::engine::transform::{BlockParams, Component as TxComponent, PredMode, residual_block};
use super::engine::{BitReader, CabacEngine, ContextModel};

/// Luma side of one inter-prediction block.
const INTER_BLOCK: usize = 16;
/// The CTB side the SAO stage is driven at, as `log2`.
const SAO_CTB_LOG2: u32 = 6;
/// Bit depth every stage runs at; the bundled sample is 8-bit Main.
const BIT_DEPTH: u8 = 8;
/// Transform-block sides the inverse-transform stage covers.
const TX_SIZES: [usize; 4] = [4, 8, 16, 32];
/// CABAC bins one iteration of the entropy stage decodes.
const CABAC_BINS: usize = 1 << 18;

/// An FNV-1a fold, so a stage's whole output identifies it in eight bytes.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }

    fn push_i32(&mut self, value: i32) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }

    fn push_all(&mut self, values: &[i32]) {
        for &value in values {
            self.push_i32(value);
        }
    }

    fn finish(self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

/// A deterministic, dependency-free generator for benchmark content.
///
/// Reproducibility is the point: the bit-exactness guard runs the same workload
/// once per instruction set and compares the results, so the inputs have to be
/// identical across those runs and across processes.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }
}

/// Synthetic 8-bit luma content mixing textured and flat regions.
///
/// The flat regions are what make this usable for deblocking: the §8.7.2.5.3
/// wide-filter decision is gated on a flatness check across the edge, so a
/// purely textured plane never reaches the strong filter and would measure only
/// half the kernel. Every other stage sees a mixture, which is what real content
/// looks like.
fn synthetic_luma(width: usize, height: usize, seed: u64) -> Vec<i32> {
    let mut rng = Lcg::new(seed);
    let mut plane = vec![0i32; width * height];
    for y in 0..height {
        // Alternating 32-row bands: textured, then flat with a small step every
        // 8 columns so the deblocking edges have something to smooth.
        let flat_band = (y / 32) % 2 == 1;
        for x in 0..width {
            let value = if flat_band {
                let level = 40 + 12 * ((x / 8) % 5) as i32;
                level + i32::from((x % 8) == 0)
            } else {
                let base = ((x * 3 + y * 5) % 200) as i32;
                base + (rng.below(24) as i32) - 12
            };
            plane[y * width + x] = value.clamp(0, 255);
        }
    }
    plane
}

/// Prepared inputs for every HEVC stage the benchmark times.
///
/// Construction is the setup half — plane allocation, content generation,
/// reference-sample and coefficient-block construction, CABAC bitstream
/// synthesis — and never appears inside a timed loop.
pub struct HevcStageInputs {
    width: usize,
    height: usize,
    luma: Vec<i32>,
    intra_blocks: Vec<(ReferenceSamples, IntraPredParams)>,
    intra_samples: u64,
    tx_blocks: Vec<(Vec<i32>, BlockParams)>,
    tx_samples: u64,
    sao_picture: Picture,
    sao_ctbs: Vec<ResolvedSao>,
    cabac_bytes: Vec<u8>,
}

impl HevcStageInputs {
    /// Builds every stage's inputs for a `width` x `height` luma frame.
    ///
    /// # Panics
    /// Panics if the dimensions are not a multiple of the 64-sample CTB the SAO
    /// stage is driven at, or if either is zero.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "the frame has to have samples");
        let luma = synthetic_luma(width, height, 0x5EED_1234);
        let intra_blocks = build_intra_blocks();
        let intra_samples = intra_blocks
            .iter()
            .map(|(p, _)| (p.n_tbs() * p.n_tbs()) as u64)
            .sum();
        let tx_blocks = build_tx_blocks();
        let tx_samples = tx_blocks.iter().map(|(l, _)| l.len() as u64).sum();
        let (sao_picture, sao_ctbs) = build_sao_inputs(width, height, &luma);
        Self {
            width,
            height,
            luma,
            intra_blocks,
            intra_samples,
            tx_blocks,
            tx_samples,
            sao_picture,
            sao_ctbs,
            cabac_bytes: build_cabac_bitstream(),
        }
    }

    /// Luma samples the inter-prediction stage writes per run.
    #[must_use]
    pub fn inter_pred_samples(&self) -> u64 {
        let (bx, by) = self.inter_grid();
        (bx * by * INTER_BLOCK * INTER_BLOCK) as u64
    }

    /// Samples the intra-prediction stage writes per run.
    #[must_use]
    pub fn intra_pred_samples(&self) -> u64 {
        self.intra_samples
    }

    /// Luma samples the deblocking stage rewrites per run.
    #[must_use]
    pub fn deblock_samples(&self) -> u64 {
        (self.luma_edges().len() * 4 * 8) as u64
    }

    /// Samples the SAO stage reads and writes per run, luma plus chroma.
    #[must_use]
    pub fn sao_samples(&self) -> u64 {
        (self.width * self.height) as u64 * 3 / 2
    }

    /// Residual samples the inverse-transform stage produces per run.
    #[must_use]
    pub fn inverse_transform_samples(&self) -> u64 {
        self.tx_samples
    }

    /// Bins the entropy stage decodes per run.
    #[must_use]
    pub fn cabac_bins(&self) -> u64 {
        CABAC_BINS as u64
    }

    /// §8.5.3.3 — fractional-sample interpolation and the weighted combine.
    ///
    /// Bi-predicts a grid of 16x16 luma blocks: two 8-tap interpolations at
    /// different fractional phases, combined through the §8.5.3.3.4.2 default
    /// weighted prediction. Both are the vectorized `filter_taps` /
    /// `combine_weighted` primitives.
    ///
    /// # Panics
    /// Panics if the prepared plane no longer matches its dimensions.
    #[must_use]
    pub fn run_inter_pred(&self) -> Vec<u8> {
        let plane = RefPlane::new(&self.luma, self.width, self.height)
            .expect("the prepared plane matches its dimensions");
        let (blocks_x, blocks_y) = self.inter_grid();
        let mut digest = Digest::new();
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x = (bx * INTER_BLOCK) as i32;
                let y = (by * INTER_BLOCK) as i32;
                // Sweep every fractional phase so both the one- and the
                // two-dimensional filter paths run.
                let l0 = interp_luma_block(
                    &plane,
                    x,
                    y,
                    (bx % 4) as i32,
                    (by % 4) as i32,
                    INTER_BLOCK,
                    INTER_BLOCK,
                    BIT_DEPTH,
                )
                .expect("the block lies inside the plane");
                let l1 = interp_luma_block(
                    &plane,
                    x + 1,
                    y + 1,
                    ((bx + 1) % 4) as i32,
                    ((by + 2) % 4) as i32,
                    INTER_BLOCK,
                    INTER_BLOCK,
                    BIT_DEPTH,
                )
                .expect("the block lies inside the plane");
                let combined = default_weighted_pred(
                    &l0,
                    &l1,
                    true,
                    true,
                    INTER_BLOCK,
                    INTER_BLOCK,
                    BIT_DEPTH,
                )
                .expect("both lists are the block's size");
                digest.push_all(&combined);
            }
        }
        digest.finish()
    }

    /// §8.4.4.2 — reference-sample smoothing and the planar / DC / angular
    /// predictors, over every mode at every transform-block size.
    #[must_use]
    pub fn run_intra_pred(&self) -> Vec<u8> {
        let mut digest = Digest::new();
        for (samples, params) in &self.intra_blocks {
            let pred = intra_predict(samples, params).expect("the prepared block is well-formed");
            digest.push_all(&pred);
        }
        digest.finish()
    }

    /// §8.7.2.5.4 — the luma block-edge deblocking filter over a whole frame's
    /// worth of vertical edge segments.
    ///
    /// `bS == 2` and a high QP put the §8.7.2.5.3 thresholds where the strong
    /// filter is reachable; the plane's flat bands are what actually reach it.
    #[must_use]
    pub fn run_deblock(&self) -> Vec<u8> {
        let mut samples = self.luma.clone();
        let mut plane = SamplePlane {
            samples: &mut samples,
            width: self.width,
            stride: self.width,
        };
        let qp = EdgeQp {
            qp_q: 37,
            qp_p: 37,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            bit_depth: BIT_DEPTH,
        };
        for pos in self.luma_edges() {
            filter_luma_block_edge(&mut plane, pos, 2, qp);
        }
        let mut digest = Digest::new();
        digest.push_all(&samples);
        digest.finish()
    }

    /// §8.7.3 — sample adaptive offset over every CTB of a full picture, with
    /// both the band-offset and the edge-offset classifiers in the grid.
    #[must_use]
    pub fn run_sao(&self) -> Vec<u8> {
        let filtered = apply_sao_picture(
            self.sao_picture.clone(),
            &self.sao_ctbs,
            SAO_CTB_LOG2,
            1,
            true,
            true,
        );
        let mut digest = Digest::new();
        for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
            digest.push_all(filtered.plane(plane));
        }
        digest.finish()
    }

    /// §8.6.2 / §8.6.3 / §8.6.4 — dequantization and the inverse transform, at
    /// every transform-block size and for both the DCT and the 4x4 intra DST.
    #[must_use]
    pub fn run_inverse_transform(&self) -> Vec<u8> {
        let mut digest = Digest::new();
        for (levels, params) in &self.tx_blocks {
            let residual =
                residual_block(levels, None, *params).expect("the prepared block is well-formed");
            digest.push_all(&residual);
        }
        digest.finish()
    }

    /// §9.3.4 — the CABAC arithmetic decoder's bin loop.
    ///
    /// This stage has no vector path and is not expected to grow one: the
    /// arithmetic decoder is inherently serial, each bin's range update
    /// depending on the previous one's. It is measured because it is the serial
    /// fraction that bounds what vectorizing everything else can buy — the
    /// residual parser in `engine::residual` spends essentially all of its time
    /// in these two calls, so their throughput is the Amdahl ceiling for the
    /// whole-frame arms.
    ///
    /// The mixture is roughly the context-coded / bypass split of residual
    /// parsing: three context-coded decisions per bypass bin.
    #[must_use]
    pub fn run_cabac(&self) -> Vec<u8> {
        let mut contexts: Vec<ContextModel> = (0..64)
            .map(|i| ContextModel::init(90 + (i as u8 % 60), 30))
            .collect();
        let mut engine =
            CabacEngine::new(BitReader::new(&self.cabac_bytes)).expect("the buffer has bits");
        let mut digest = Digest::new();
        let mut acc = 0i32;
        let context_count = contexts.len();
        for i in 0..CABAC_BINS {
            let bin = if i % 4 == 3 {
                engine.decode_bypass()
            } else {
                engine.decode_decision(&mut contexts[i % context_count])
            };
            match bin {
                Ok(bin) => acc = acc.wrapping_mul(3).wrapping_add(i32::from(bin)),
                // A fixed buffer decoded to exhaustion stops here; the bin count
                // reached is the same on every arm because the input is fixed.
                Err(_) => break,
            }
            digest.push_i32(acc);
        }
        digest.finish()
    }

    /// Inter-prediction block grid, leaving a block of margin on each axis so
    /// the `+1` L1 offset stays addressable.
    fn inter_grid(&self) -> (usize, usize) {
        (
            (self.width / INTER_BLOCK).saturating_sub(1),
            (self.height / INTER_BLOCK).saturating_sub(1),
        )
    }

    /// Every vertical luma edge segment of the frame, on the §8.7.2.4 grid:
    /// 8 samples apart across the edge, 4 rows apart along it.
    fn luma_edges(&self) -> Vec<EdgePos> {
        let mut edges = Vec::new();
        let mut ey = 0;
        while ey + 4 <= self.height {
            let mut ex = 8;
            while ex + 8 <= self.width {
                edges.push(EdgePos {
                    ex,
                    ey,
                    edge: EdgeType::Vertical,
                });
                ex += 8;
            }
            ey += 4;
        }
        edges
    }
}

/// One reference array and parameter set per (mode, size) pair, repeated so a
/// run covers a frame-scale number of samples rather than a handful of blocks.
fn build_intra_blocks() -> Vec<(ReferenceSamples, IntraPredParams)> {
    const REPEATS: usize = 24;
    let mut rng = Lcg::new(0x17AB_5EED);
    let mut blocks = Vec::new();
    for _ in 0..REPEATS {
        for n_tbs in TX_SIZES {
            let want = 2 * n_tbs;
            let left: Vec<i32> = (0..want).map(|_| rng.below(256) as i32).collect();
            let top: Vec<i32> = (0..want).map(|_| rng.below(256) as i32).collect();
            let corner = rng.below(256) as i32;
            let samples = ReferenceSamples::new(n_tbs, corner, left, top)
                .expect("the arrays are 2*nTbS long and nTbS is legal");
            for mode in 0..=34u8 {
                blocks.push((
                    samples.clone(),
                    IntraPredParams {
                        pred_mode_intra: mode,
                        cidx: IntraComponent::Luma,
                        bit_depth: BIT_DEPTH,
                        bit_depth_luma: BIT_DEPTH,
                        intra_smoothing_disabled: false,
                        strong_intra_smoothing_enabled: true,
                        chroma_array_type_3: false,
                        disable_boundary_filter: false,
                    },
                ));
            }
        }
    }
    blocks
}

/// Coefficient blocks at every transform size, sparse the way real residual is:
/// energy concentrated in the low-frequency corner, most positions zero.
fn build_tx_blocks() -> Vec<(Vec<i32>, BlockParams)> {
    const BLOCKS_PER_SIZE: usize = 512;
    let mut rng = Lcg::new(0x7A_5F0);
    let mut blocks = Vec::new();
    for n_tbs in TX_SIZES {
        for i in 0..BLOCKS_PER_SIZE {
            let mut levels = vec![0i32; n_tbs * n_tbs];
            // A low-frequency sub-block of non-zero levels, as a coded
            // sub-block group would leave.
            let coded = (n_tbs / 2).max(4);
            for y in 0..coded {
                for x in 0..coded {
                    let magnitude = (rng.below(64) as i32) - 32;
                    levels[y * n_tbs + x] = magnitude;
                }
            }
            let pred_mode = if i % 2 == 0 {
                PredMode::Intra
            } else {
                PredMode::Inter
            };
            blocks.push((
                levels,
                BlockParams {
                    n_tbs,
                    q_p: 26 + (i as u32 % 12),
                    component: TxComponent::Luma,
                    pred_mode,
                    bit_depth: BIT_DEPTH,
                    extended_precision: false,
                    transquant_bypass: false,
                    transform_skip: false,
                    transform_skip_rotation_enabled: false,
                },
            ));
        }
    }
    blocks
}

/// A full 4:2:0 picture of synthetic content plus a per-CTB SAO parameter grid
/// that alternates band offset and the four edge-offset classes.
fn build_sao_inputs(width: usize, height: usize, luma: &[i32]) -> (Picture, Vec<ResolvedSao>) {
    let mut picture = Picture::new(width, height, 1, BIT_DEPTH, BIT_DEPTH);
    let (y_plane, _) = picture.plane_mut(Plane::Luma);
    y_plane.copy_from_slice(luma);
    let chroma = synthetic_luma(width / 2, height / 2, 0xC4B0);
    for plane in [Plane::Cb, Plane::Cr] {
        let (samples, _) = picture.plane_mut(plane);
        samples.copy_from_slice(&chroma);
    }

    let ctb = 1usize << SAO_CTB_LOG2;
    let ctbs_x = width.div_ceil(ctb);
    let ctbs_y = height.div_ceil(ctb);
    let mut grid = Vec::with_capacity(ctbs_x * ctbs_y);
    for i in 0..ctbs_x * ctbs_y {
        let mut components = [ResolvedSaoComponent::off(); 3];
        for (cidx, component) in components.iter_mut().enumerate() {
            // Alternate band offset (type 1) and edge offset (type 2), cycling
            // the four edge classes, so both classifiers and all four
            // directions run inside one picture.
            let band = (i + cidx) % 5 == 0;
            *component = if band {
                ResolvedSaoComponent {
                    sao_type_idx: 1,
                    offset_val: [0, 3, -2, 1, -3],
                    band_position: ((i * 7 + cidx) % 28) as u8,
                    eo_class: 0,
                }
            } else {
                ResolvedSaoComponent {
                    sao_type_idx: 2,
                    offset_val: [0, 2, 1, -1, -2],
                    band_position: 0,
                    eo_class: ((i + cidx) % 4) as u8,
                }
            };
        }
        grid.push(ResolvedSao { components });
    }
    (picture, grid)
}

/// A fixed pseudo-random byte buffer for the CABAC stage.
///
/// The arithmetic decoder's cost is in its renormalization and range updates,
/// which depend on the bin values it produces, not on those bins being a valid
/// slice segment. A fixed buffer makes the bin sequence reproducible, which is
/// what the cross-instruction-set comparison needs.
fn build_cabac_bitstream() -> Vec<u8> {
    let mut rng = Lcg::new(0xCABA_C0DE);
    (0..CABAC_BINS).map(|_| rng.below(256) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// A frame small enough to build and run quickly, but still a whole number
    /// of the 64-sample CTBs the SAO stage is driven at.
    fn small_inputs() -> HevcStageInputs {
        HevcStageInputs::new(256, 128)
    }

    /// Every stage must produce identical output on every instruction set the
    /// host can run.
    ///
    /// This is the same guarantee the benchmark's own guard checks, asserted
    /// here so it fails in `cargo test` rather than only when someone runs the
    /// benchmarks: a stage whose vector path diverged would otherwise report a
    /// speedup that is really just a different answer.
    #[test]
    fn every_stage_is_bit_exact_across_instruction_sets() {
        let _guard = simd::test_lock();
        let inputs = small_inputs();
        let stages: [(&str, fn(&HevcStageInputs) -> Vec<u8>); 6] = [
            ("inter_pred", HevcStageInputs::run_inter_pred),
            ("intra_pred", HevcStageInputs::run_intra_pred),
            ("deblock", HevcStageInputs::run_deblock),
            ("sao", HevcStageInputs::run_sao),
            ("inverse_transform", HevcStageInputs::run_inverse_transform),
            ("cabac", HevcStageInputs::run_cabac),
        ];

        for (name, run) in stages {
            simd::set_override(Some(SimdIsa::Scalar));
            let reference = run(&inputs);
            for isa in simd::available() {
                simd::set_override(Some(isa));
                assert_eq!(
                    run(&inputs),
                    reference,
                    "{name} on {} diverged from the scalar reference",
                    isa.name()
                );
            }
        }
        simd::set_override(None);
    }

    /// The digests have to depend on the stages' samples.
    ///
    /// A workload that returned a constant would pass the bit-exactness guard
    /// above vacuously, so check that different stages of the same inputs
    /// produce different digests and that none of them is the empty fold.
    #[test]
    fn stage_digests_depend_on_the_stage_output() {
        let _guard = simd::test_lock();
        simd::set_override(Some(SimdIsa::Scalar));
        let inputs = small_inputs();
        let empty = Digest::new().finish();
        let digests = [
            inputs.run_inter_pred(),
            inputs.run_intra_pred(),
            inputs.run_deblock(),
            inputs.run_sao(),
            inputs.run_inverse_transform(),
            inputs.run_cabac(),
        ];
        simd::set_override(None);

        for digest in &digests {
            assert_eq!(digest.len(), 8, "a digest is eight bytes");
            assert_ne!(*digest, empty, "a stage folded nothing");
        }
        for (i, a) in digests.iter().enumerate() {
            for b in &digests[i + 1..] {
                assert_ne!(a, b, "two stages produced the same digest");
            }
        }
    }

    /// The deblocking input has to actually reach the wide filter.
    ///
    /// The §8.7.2.5.3 decision takes the strong filter only where the samples
    /// on both sides of an edge are flat enough. Content that never satisfies
    /// it would leave the wide path untimed while still producing a plausible
    /// number, so assert the plane contains flat runs across its edges rather
    /// than trusting the generator.
    #[test]
    fn the_deblocking_content_contains_flat_edges() {
        let inputs = small_inputs();
        let flat_edges = inputs
            .luma_edges()
            .iter()
            .filter(|pos| {
                // The §8.7.2.5.3 dEp/dEq flatness test reads p3..p0 / q0..q3 of
                // the segment's first row; approximate it by requiring both
                // sides to span no more than a couple of levels.
                let row = pos.ey * inputs.width;
                let span = |from: usize| {
                    let window = &inputs.luma[row + from..row + from + 4];
                    window.iter().max().unwrap() - window.iter().min().unwrap()
                };
                span(pos.ex - 4) <= 2 && span(pos.ex) <= 2
            })
            .count();
        assert!(
            flat_edges > inputs.luma_edges().len() / 10,
            "only {flat_edges} of {} edges are flat; the wide filter would go untimed",
            inputs.luma_edges().len()
        );
    }

    /// The stages' reported work has to match what they actually touch, since
    /// it is the denominator of every throughput number the benchmark prints.
    #[test]
    fn reported_sample_counts_match_the_workloads() {
        let inputs = small_inputs();
        assert_eq!(inputs.inter_pred_samples(), 15 * 7 * 256);
        assert_eq!(inputs.sao_samples(), 256 * 128 * 3 / 2);
        assert_eq!(
            inputs.deblock_samples(),
            inputs.luma_edges().len() as u64 * 32
        );
        assert_eq!(inputs.cabac_bins(), CABAC_BINS as u64);
        assert!(inputs.intra_pred_samples() > 0);
        assert!(inputs.inverse_transform_samples() > 0);
    }
}
