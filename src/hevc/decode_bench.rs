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
use super::engine::inter_pred::{
    RefPlane, default_weighted_pred, interp_chroma_block, interp_luma_block,
};
use super::engine::intra_pred::{
    Component as IntraComponent, IntraPredParams, ReferenceSamples, intra_predict,
};
use super::engine::picture::{Picture, Plane};
use super::engine::sao::{
    ResolvedSao, ResolvedSaoComponent, SaoBoundaries, apply_sao_picture_with_boundaries,
};
use super::engine::transform::{BlockParams, Component as TxComponent, PredMode, residual_block};
use super::engine::{BitReader, CabacEngine, ContextModel};
use super::{HevcDecoder, ParsedConfiguration, picture_to_rgba};
use crate::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, HardwarePreference,
    Limits, PixelFormat, VideoDecoder, VideoDecoderConfig, VideoDimensions,
};

/// The largest inter prediction unit, and the cell the §8.5.3.3 workload's
/// prediction-unit schedule is laid out on: every cell is filled by prediction
/// units of one class, so a cell is either one 64x64 unit or the 4, 16 or 64
/// smaller ones that tile it.
const INTER_CELL: usize = 64;

/// The §8.5.3.3 prediction-unit mix a real decode runs, as issue #280 measured
/// it: 48 frames of `examples/media/BigBuckBunny.mp4` at 1920x1080, every
/// prediction unit the decoder reconstructed, weighted by luma sample.
///
/// Each entry is `(luma side, bi-predicted, share of luma samples)`. The
/// workload this drives replaced a uniform grid of 16x16 bi-predicted
/// *luma-only* blocks, which that measurement showed was not what the stage
/// runs on real content in any of the three ways that matter to a vector
/// kernel:
///
/// * **Size.** 62% of predicted luma samples are in 64x64 units and 31% in
///   32x32; 16x16 is 5% and 8x8 is 1.4%. The old grid's block was the third
///   most common size by sample and carried 5% of the real work.
/// * **List utilisation.** 38% of predicted luma samples are uni-predicted,
///   which runs one interpolation rather than two and takes the
///   §8.5.3.3.4.2 single-list combine rather than the two-list one. The old
///   grid was bi-predicted throughout.
/// * **Component.** At 4:2:0 every luma sample brings half a chroma sample,
///   through the §8.5.3.3.3.3 4-tap filter rather than the 8-tap one — a
///   third of the stage's samples, on a kernel that measures materially lower
///   than the luma one. The old grid predicted no chroma at all.
///
/// The clip codes 2Nx2N prediction units throughout, so every size here is
/// square; a stream using the §7.3.8.5 asymmetric partitions would add
/// rectangular units, and this table would be re-measured rather than guessed.
const INTER_PU_MIX: [(usize, bool, f64); 8] = [
    (64, true, 0.432),
    (64, false, 0.189),
    (32, true, 0.162),
    (32, false, 0.149),
    (16, false, 0.036),
    (16, true, 0.017),
    (8, false, 0.009),
    (8, true, 0.005),
];
/// The §7.3.8.3 SAO parameter mix a real decode runs, as issue #310 measured
/// it: 48 frames of `examples/media/BigBuckBunny.mp4` at 1920x1080, all 26,520
/// coding tree blocks the decoder resolved parameters for, counted per colour
/// component.
///
/// Row 0 is luma, row 1 chroma — Cb and Cr came out identical CTB for CTB, as
/// §7.3.8.3 signals one `sao_type_idx_chroma` and one `SaoEoClass` for the
/// pair. Each row is the share of CTBs taking, in order: no SAO at all
/// (`SaoTypeIdx == 0`), band offset, and the four edge-offset classes
/// (0-degree, 90-degree, 135-degree, 45-degree).
///
/// The workload this drives replaced a grid that put band or edge offset on
/// *every* CTB of *every* component, which is not what the stage runs on real
/// content in the way that matters most to it: **SAO is off on 86.7% of luma
/// CTBs and 94.4% of chroma CTBs**, and an off CTB costs nothing at all. The
/// old grid therefore filtered about eight times the samples a 1080p frame
/// actually puts through the classifiers, so its ratio was the kernels'
/// throughput on a saturated picture rather than their worth to a decode.
const SAO_CTB_MIX: [[f64; 6]; 2] = [
    // off, band, EO 0-deg, EO 90-deg, EO 135-deg, EO 45-deg
    [0.866780, 0.020362, 0.020777, 0.030656, 0.026697, 0.034729],
    [0.944155, 0.009540, 0.001357, 0.004450, 0.015875, 0.024623],
];
/// The CTB side the SAO stage is driven at, as `log2`.
const SAO_CTB_LOG2: u32 = 6;
/// Bit depth every stage runs at; the bundled sample is 8-bit Main.
const BIT_DEPTH: u8 = 8;
/// Transform-block sides the inverse-transform stage covers.
const TX_SIZES: [usize; 4] = [4, 8, 16, 32];
/// CABAC bins one iteration of the entropy stage decodes.
const CABAC_BINS: usize = 1 << 18;

/// One scheduled §8.5.3.3 prediction unit of the inter-prediction workload.
#[derive(Debug, Clone, Copy)]
struct InterPu {
    /// Top-left luma position, in samples.
    x: i32,
    y: i32,
    /// Luma side; the unit is square, as every unit the measured clip codes is.
    n: usize,
    /// `predFlagL1` — whether this unit runs two interpolations and the
    /// two-list §8.5.3.3.4.2 combine, or one and the single-list one.
    bi: bool,
    /// Which fractional phase pair this unit filters at, so the schedule
    /// sweeps the one- and two-dimensional filter paths and the copy path.
    phase: i32,
}

/// Lay out one frame's worth of prediction units matching [`INTER_PU_MIX`].
///
/// The frame is walked in [`INTER_CELL`]-sized cells, and each cell is filled
/// entirely by units of one class. The class is chosen greedily — whichever is
/// furthest below its target share of luma samples so far — so the emitted mix
/// converges on the measured one exactly rather than approximately, and does so
/// deterministically, which is what the benchmark's bit-exactness guard needs.
///
/// A margin of one cell is left on the right and bottom edges because the
/// workload offsets its second reference list by one sample; every unit and
/// both of its lists therefore lie inside the plane.
fn schedule_inter_pus(width: usize, height: usize) -> Vec<InterPu> {
    let cells_x = (width / INTER_CELL).saturating_sub(1);
    let cells_y = (height / INTER_CELL).saturating_sub(1);
    let mut emitted = [0u64; INTER_PU_MIX.len()];
    let mut total = 0u64;
    let mut pus = Vec::new();
    let mut phase = 0i32;
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            // Furthest below target, measured as a sample deficit so the very
            // first cell (where every share is zero) still picks the largest.
            let class = (0..INTER_PU_MIX.len())
                .max_by(|&a, &b| {
                    let deficit = |i: usize| {
                        INTER_PU_MIX[i]
                            .2
                            .mul_add(total as f64, -(emitted[i] as f64))
                    };
                    deficit(a)
                        .partial_cmp(&deficit(b))
                        .expect("the shares are finite")
                })
                .expect("the mix is not empty");
            let (n, bi, _) = INTER_PU_MIX[class];
            for by in (0..INTER_CELL).step_by(n) {
                for bx in (0..INTER_CELL).step_by(n) {
                    pus.push(InterPu {
                        x: (cx * INTER_CELL + bx) as i32,
                        y: (cy * INTER_CELL + by) as i32,
                        n,
                        bi,
                        phase,
                    });
                    phase = (phase + 1) % 16;
                }
            }
            let cell_samples = (INTER_CELL * INTER_CELL) as u64;
            emitted[class] += cell_samples;
            total += cell_samples;
        }
    }
    pus
}

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

    /// Folds `value` in one step.
    fn push_u64(&mut self, value: u64) {
        self.0 = (self.0 ^ value).wrapping_mul(0x1000_0000_01b3);
    }

    /// Folds a whole plane of decoded samples, two at a time.
    ///
    /// [`push_all`] costs a multiply per *byte*, which is nothing next to a
    /// per-stage workload of a few thousand samples but is real work next to a
    /// whole-frame group that folds several million per iteration, inside the
    /// timed loop. This is the wide step for those; every sample still reaches
    /// the accumulator, which is what the harness's bit-exactness guard needs.
    ///
    /// [`push_all`]: Self::push_all
    fn push_samples(&mut self, values: &[i32]) {
        let mut chunks = values.chunks_exact(2);
        for pair in &mut chunks {
            self.push_u64(u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32));
        }
        for &value in chunks.remainder() {
            self.push_u64(u64::from(value as u32));
        }
    }

    /// Folds a byte buffer eight bytes at a time, for the same reason as
    /// [`push_samples`].
    ///
    /// [`push_samples`]: Self::push_samples
    fn push_bytes(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.push_u64(u64::from_le_bytes(
                chunk.try_into().expect("the chunk is eight bytes"),
            ));
        }
        for &byte in chunks.remainder() {
            self.push_u64(u64::from(byte));
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
    /// A full 8-bit 4:2:0 picture of synthetic content, driven by both the
    /// §8.7.3 SAO stage and the output-conversion stage — the second needs
    /// exactly what the first already builds, a whole decoded picture.
    picture: Picture,
    sao_ctbs: Vec<ResolvedSao>,
    /// The single-slice, single-tile §8.7.3.2 boundary grids the stage is
    /// driven with. A decode always has these — whether a stream is
    /// single-slice and single-tile is not known before it is parsed — so
    /// running the stage without them (as this group did before issue #310)
    /// measures a dispatch the decoder never takes.
    sao_boundaries: SaoBoundaries,
    /// Samples the SAO classifiers actually run over per run, luma plus
    /// chroma: [`SAO_CTB_MIX`] leaves most CTBs with `SaoTypeIdx == 0`, and
    /// those cost nothing.
    sao_filtered_samples: u64,
    /// The §8.5.3.3 prediction units one run of the stage reconstructs, in
    /// raster order, built to [`INTER_PU_MIX`]. Scheduling them is setup, so
    /// it happens here rather than inside the timed loop.
    inter_pus: Vec<InterPu>,
    inter_luma_samples: u64,
    convert_config: VideoDecoderConfig,
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
        let (picture, sao_ctbs) = build_sao_inputs(width, height, &luma);
        let (sao_boundaries, sao_filtered_samples) = build_sao_boundaries(width, height, &sao_ctbs);
        let inter_pus = schedule_inter_pus(width, height);
        let inter_luma_samples = inter_pus.iter().map(|pu| (pu.n * pu.n) as u64).sum();
        Self {
            width,
            height,
            luma,
            intra_blocks,
            intra_samples,
            tx_blocks,
            tx_samples,
            picture,
            sao_ctbs,
            sao_boundaries,
            sao_filtered_samples,
            inter_pus,
            inter_luma_samples,
            convert_config: convert_config(width, height),
            cabac_bytes: build_cabac_bitstream(),
        }
    }

    /// Luma samples the inter-prediction stage writes per run.
    ///
    /// Luma only, so the figure stays comparable with the one this group
    /// reported before it predicted chroma; each of these samples brings half
    /// a chroma sample of §8.5.3.3.3.3 work with it at 4:2:0.
    #[must_use]
    pub fn inter_pred_samples(&self) -> u64 {
        self.inter_luma_samples
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

    /// Samples the SAO stage classifies per run, luma plus chroma.
    ///
    /// Counts only the CTBs [`SAO_CTB_MIX`] leaves switched on, since a CTB
    /// with `SaoTypeIdx == 0` is returned from before it reads a sample. On the
    /// measured mix that is a small minority of the picture, which is the
    /// point: the throughput this divides is the classifiers', not the frame's.
    #[must_use]
    pub fn sao_samples(&self) -> u64 {
        self.sao_filtered_samples
    }

    /// Residual samples the inverse-transform stage produces per run.
    #[must_use]
    pub fn inverse_transform_samples(&self) -> u64 {
        self.tx_samples
    }

    /// Samples the output-conversion stage writes per run, one RGBA pixel each.
    #[must_use]
    pub fn color_convert_samples(&self) -> u64 {
        (self.width * self.height) as u64
    }

    /// Bins the entropy stage decodes per run.
    #[must_use]
    pub fn cabac_bins(&self) -> u64 {
        CABAC_BINS as u64
    }

    /// §8.5.3.3 — fractional-sample interpolation and the weighted combine,
    /// over the prediction-unit mix a real decode runs.
    ///
    /// Each scheduled unit interpolates its luma through the §8.5.3.3.3.2
    /// 8-tap filter and, at 4:2:0, its Cb and Cr through the §8.5.3.3.3.3
    /// 4-tap filter, once per used reference list, and combines the result
    /// through the §8.5.3.3.4.2 default weighted prediction. Every one of
    /// those is a vectorized `filter_taps` / `combine_weighted` primitive.
    ///
    /// The unit sizes, the uni/bi split and the presence of chroma all come
    /// from [`INTER_PU_MIX`], which is measured rather than chosen; see that
    /// constant for what the uniform 16x16 bi-predicted luma-only grid this
    /// replaced was getting wrong.
    ///
    /// # Panics
    /// Panics if the prepared planes no longer match their dimensions.
    #[must_use]
    pub fn run_inter_pred(&self) -> Vec<u8> {
        let luma = RefPlane::new(&self.luma, self.width, self.height)
            .expect("the prepared plane matches its dimensions");
        let (cw, ch) = (self.width / 2, self.height / 2);
        let cb = RefPlane::new(self.picture.plane(Plane::Cb), cw, ch)
            .expect("the prepared picture's Cb plane matches its dimensions");
        let cr = RefPlane::new(self.picture.plane(Plane::Cr), cw, ch)
            .expect("the prepared picture's Cr plane matches its dimensions");
        let mut digest = Digest::new();
        for pu in &self.inter_pus {
            let n = pu.n;
            // Sweep the fractional phases so both the one-dimensional and the
            // two-dimensional filter paths run, and so a unit that lands on a
            // full-sample position (the copy path) is reached too.
            let (fx0, fy0) = (pu.phase % 4, (pu.phase / 4) % 4);
            let (fx1, fy1) = ((pu.phase + 1) % 4, (pu.phase + 3) % 4);
            let l0 = interp_luma_block(&luma, pu.x, pu.y, fx0, fy0, n, n, BIT_DEPTH)
                .expect("the unit lies inside the plane");
            let l1 = if pu.bi {
                interp_luma_block(&luma, pu.x + 1, pu.y + 1, fx1, fy1, n, n, BIT_DEPTH)
                    .expect("the unit lies inside the plane")
            } else {
                Vec::new()
            };
            let combined = default_weighted_pred(&l0, &l1, true, pu.bi, n, n, BIT_DEPTH)
                .expect("both lists are the unit's size");
            digest.push_all(&combined);

            // §8.5.3.3.3.3: the chroma units are half the luma unit's side at
            // 4:2:0, at eighth-sample phases rather than quarter-sample ones.
            let (cn, cx, cy) = (n / 2, pu.x / 2, pu.y / 2);
            let (cfx0, cfy0) = (pu.phase % 8, (pu.phase / 8) % 8);
            let (cfx1, cfy1) = ((pu.phase + 3) % 8, (pu.phase + 5) % 8);
            for plane in [&cb, &cr] {
                let c0 = interp_chroma_block(plane, cx, cy, cfx0, cfy0, cn, cn, BIT_DEPTH)
                    .expect("the unit lies inside the plane");
                let c1 = if pu.bi {
                    interp_chroma_block(plane, cx + 1, cy + 1, cfx1, cfy1, cn, cn, BIT_DEPTH)
                        .expect("the unit lies inside the plane")
                } else {
                    Vec::new()
                };
                let combined = default_weighted_pred(&c0, &c1, true, pu.bi, cn, cn, BIT_DEPTH)
                    .expect("both lists are the unit's size");
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
        let filtered = apply_sao_picture_with_boundaries(
            self.picture.clone(),
            &self.sao_ctbs,
            SAO_CTB_LOG2,
            1,
            true,
            true,
            Some(&self.sao_boundaries),
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

    /// The YUV420-to-RGBA output conversion every whole-frame decode ends in.
    ///
    /// Issue #220: this stage is not decoding, but it was a third of what the
    /// whole-frame groups measure (see `benches/README.md`), so it is timed
    /// directly rather than inferred by subtracting one whole-frame group from
    /// another. Issue #219 gave it the [`color_convert`] kernel family, so its
    /// arms now separate by instruction set like the decoding stages do, and
    /// this group is what reports that.
    ///
    /// # Panics
    /// Panics if the prepared picture no longer converts.
    ///
    /// [`color_convert`]: super::color_convert
    #[must_use]
    pub fn run_color_convert(&self) -> Vec<u8> {
        let frame = picture_to_rgba(&self.picture, &self.convert_config, &Limits::default())
            .expect("the prepared picture is Main-profile 8-bit 4:2:0");
        let mut digest = Digest::new();
        for plane in &frame.planes {
            digest.push_bytes(&plane.data);
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

/// Pick the class furthest below its target share of the CTBs emitted so far.
///
/// Greedy on the sample deficit, as [`schedule_inter_pus`] is, so the emitted
/// mix converges on the measured one exactly rather than approximately and does
/// so deterministically — which is what the benchmark's bit-exactness guard
/// needs.
fn sao_class_for(shares: &[f64; 6], emitted: &[u64; 6], total: u64) -> usize {
    (0..6)
        .max_by(|&a, &b| {
            let deficit = |i: usize| shares[i].mul_add(total as f64, -(emitted[i] as f64));
            deficit(a)
                .partial_cmp(&deficit(b))
                .expect("the shares are finite")
        })
        .expect("the mix is not empty")
}

/// The resolved SAO parameters of one [`SAO_CTB_MIX`] class.
fn sao_component_for(class: usize, i: usize, cidx: usize) -> ResolvedSaoComponent {
    match class {
        0 => ResolvedSaoComponent::off(),
        1 => ResolvedSaoComponent {
            sao_type_idx: 1,
            offset_val: [0, 3, -2, 1, -3],
            band_position: ((i * 7 + cidx) % 28) as u8,
            eo_class: 0,
        },
        _ => ResolvedSaoComponent {
            sao_type_idx: 2,
            offset_val: [0, 2, 1, -1, -2],
            band_position: 0,
            eo_class: (class - 2) as u8,
        },
    }
}

/// A full 4:2:0 picture of synthetic content plus a per-CTB SAO parameter grid
/// built to the measured [`SAO_CTB_MIX`].
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
    // One deficit accumulator per row of the mix: luma is scheduled on its own
    // shares, and Cb / Cr share the chroma row's — and take the same class as
    // each other, as §7.3.8.3 signals them as one.
    let mut emitted = [[0u64; 6]; 2];
    let mut total = 0u64;
    for i in 0..ctbs_x * ctbs_y {
        let luma_class = sao_class_for(&SAO_CTB_MIX[0], &emitted[0], total);
        let chroma_class = sao_class_for(&SAO_CTB_MIX[1], &emitted[1], total);
        emitted[0][luma_class] += 1;
        emitted[1][chroma_class] += 1;
        total += 1;
        grid.push(ResolvedSao {
            components: [
                sao_component_for(luma_class, i, 0),
                sao_component_for(chroma_class, i, 1),
                sao_component_for(chroma_class, i, 2),
            ],
        });
    }
    (picture, grid)
}

/// The §8.7.3.2 boundary grids for a single-slice, single-tile picture, plus
/// the number of samples the grid's switched-on CTBs classify.
///
/// Every CTB is in slice 0 and tile 0, so nothing here denies a neighbour read
/// — but the decoder still carries the grids, and the per-CTB
/// [`SaoBoundaries::ctb_neighbourhood_unconstrained`] test that clears the
/// vector path has to run over them. Passing them is what makes this group time
/// the dispatch a decode takes.
fn build_sao_boundaries(width: usize, height: usize, grid: &[ResolvedSao]) -> (SaoBoundaries, u64) {
    let ctb = 1usize << SAO_CTB_LOG2;
    let ctbs_x = width.div_ceil(ctb);
    let ctbs_y = height.div_ceil(ctb);
    let mut samples = 0u64;
    for (i, resolved) in grid.iter().enumerate() {
        let (rx, ry) = (i % ctbs_x, i / ctbs_x);
        let w = ctb.min(width.saturating_sub(rx * ctb));
        let h = ctb.min(height.saturating_sub(ry * ctb));
        if resolved.components[0].sao_type_idx != 0 {
            samples += (w * h) as u64;
        }
        for cidx in 1..3 {
            if resolved.components[cidx].sao_type_idx != 0 {
                samples += ((w / 2) * (h / 2)) as u64;
            }
        }
    }
    let n = ctbs_x * ctbs_y;
    let boundaries = SaoBoundaries {
        slice_addr_of_ctb: vec![0; n],
        tile_id_of_ctb: vec![0; n],
        pic_w_ctbs: ctbs_x,
        ctb_log2_size_y: SAO_CTB_LOG2,
        across_slices: true,
        across_tiles: true,
        filter_across_of_ctb: Some(vec![true; n]),
        ctb_ts_of_rs: Some((0..n as u32).collect()),
    };
    (boundaries, samples)
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

/// The decoder configuration [`HevcStageInputs::run_color_convert`] converts
/// under: the same RGBA8 limited-range output every whole-frame group asks for,
/// at the prepared picture's dimensions.
fn convert_config(width: usize, height: usize) -> VideoDecoderConfig {
    let dimensions = VideoDimensions::new(width as u32, height as u32, &Limits::default())
        .expect("the benchmark frame is within the default limits");
    VideoDecoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: Vec::new(),
    }
}

/// Decodes `frames` frames of `samples` and stops at the decoded `Picture`.
///
/// Issue #220: the public decoder's `submit` returns RGBA, so a whole-frame
/// benchmark through it measures decoding *plus* the colour conversion of
/// [`HevcStageInputs::run_color_convert`] — a third of the interval, which
/// until issue #219 gave it a kernel had no vector path at all and diluted
/// every scalar-versus-SIMD ratio taken off those groups. This is the same
/// decode without that tail: it drives the same
/// `HevcDecoder` over the same access units and collects its pictures instead
/// of converting them, so the difference between this and the end-to-end group
/// is the conversion and nothing else about how the bitstream is handled.
///
/// Returns a fold over every decoded sample, for the harness's bit-exactness
/// guard — the same fold [`decode_frames`] applies to the converted bytes, so
/// neither group pays for identifying its output more than the other does. Pins no instruction set; the caller selects the arm through
/// [`crate::simd::set_override`] exactly as for an end-to-end decode.
///
/// # Panics
/// Panics if the configuration is not decodable or if `samples` does not yield
/// `frames` frames — a benchmark measuring less work than it reports would be
/// worse than a failure.
/// Decodes `frames` frames of `samples` all the way out to RGBA.
///
/// The end-to-end half of the issue #220 split, and the counterpart to
/// [`decode_pictures`]: same decoder, same access units, same frame count, same
/// output fold — the only difference is the `picture_to_rgba` pass on each
/// decoded picture, which is what makes the gap between the two groups the
/// conversion rather than an artefact of how each arm identifies its output.
///
/// # Panics
/// Panics under the same conditions as [`decode_pictures`].
#[must_use]
pub fn decode_frames(
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    limits: &Limits,
    frames: u64,
) -> Vec<u8> {
    let mut decoder = bench_decoder(configuration, limits);
    let cancellation = CancellationToken::new();
    let mut digest = Digest::new();
    let mut decoded = 0_u64;
    for sample in samples {
        for frame in decoder
            .submit(sample, &cancellation)
            .expect("the sample decodes")
        {
            for plane in &frame.frame.planes {
                digest.push_bytes(&plane.data);
            }
            decoded += 1;
        }
        if decoded >= frames {
            break;
        }
    }
    assert_frame_count(decoded, frames);
    digest.finish()
}

/// The software decoder the whole-frame benchmark surface drives.
///
/// # Panics
/// Panics if the configuration is not decodable.
fn bench_decoder(configuration: &VideoDecoderConfig, limits: &Limits) -> HevcDecoder {
    let parsed = ParsedConfiguration::parse(configuration, limits)
        .expect("the sample's configuration record parses");
    HevcDecoder::new(configuration.clone(), *limits, parsed)
        .expect("the software HEVC decoder is constructible")
}

/// A benchmark measuring less work than it reports would be worse than a
/// failure.
fn assert_frame_count(decoded: u64, frames: u64) {
    assert!(
        decoded >= frames,
        "the sample yielded {decoded} decoded pictures, not the {frames} the group reports"
    );
}

#[must_use]
pub fn decode_pictures(
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    limits: &Limits,
    frames: u64,
) -> Vec<u8> {
    let mut decoder = bench_decoder(configuration, limits);
    let cancellation = CancellationToken::new();
    let mut digest = Digest::new();
    let mut decoded = 0_u64;
    for sample in samples {
        for picture in decoder
            .submit_pictures(sample, &cancellation)
            .expect("the sample decodes")
        {
            for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
                digest.push_samples(picture.plane(plane));
            }
            decoded += 1;
        }
        if decoded >= frames {
            break;
        }
    }
    assert_frame_count(decoded, frames);
    digest.finish()
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

    /// The §8.5.3.3 workload has to run the prediction-unit mix issue #280
    /// measured, not a mix that happens to be close to it.
    ///
    /// The uniform 16x16 bi-predicted luma-only grid this replaced would fail
    /// every assertion here: one size instead of four, no uni-predicted units,
    /// and no chroma. A schedule that drifted back towards one of those — say
    /// by rounding every cell to the largest class — would look like a faster
    /// kernel rather than a narrower workload, which is the failure this
    /// benchmark exists to avoid.
    #[test]
    fn the_inter_pred_schedule_runs_the_measured_prediction_unit_mix() {
        // A frame large enough that each class gets whole cells to land in;
        // the greedy schedule converges on the shares as cells accumulate.
        let pus = schedule_inter_pus(1920, 1088);
        let total: u64 = pus.iter().map(|pu| (pu.n * pu.n) as u64).sum();
        assert!(total > 0, "the schedule emitted no prediction units");

        for &(n, bi, share) in &INTER_PU_MIX {
            let got: u64 = pus
                .iter()
                .filter(|pu| pu.n == n && pu.bi == bi)
                .map(|pu| (pu.n * pu.n) as u64)
                .sum();
            let got = got as f64 / total as f64;
            assert!(
                (got - share).abs() < 0.01,
                "{n}x{n} {} is {:.1}% of luma samples, measured {:.1}%",
                if bi { "bi" } else { "uni" },
                got * 100.0,
                share * 100.0,
            );
        }

        // Bi-prediction is the majority but not the whole of it, and every
        // measured size is present — the three things the old grid collapsed.
        let bi: u64 = pus
            .iter()
            .filter(|pu| pu.bi)
            .map(|pu| (pu.n * pu.n) as u64)
            .sum();
        let bi = bi as f64 / total as f64;
        assert!((0.55..0.70).contains(&bi), "bi-predicted share is {bi}");
        for side in [8usize, 16, 32, 64] {
            assert!(
                pus.iter().any(|pu| pu.n == side),
                "no {side}x{side} prediction unit was scheduled"
            );
        }

        // And the workload predicts chroma: every unit is at least 8x8 luma,
        // so every unit has a 4:2:0 chroma unit of at least 4x4 to filter.
        assert!(
            pus.iter()
                .all(|pu| pu.n >= 8 && pu.x % 2 == 0 && pu.y % 2 == 0),
            "a unit has no whole 4:2:0 chroma unit"
        );
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
        /// One stage's name and the entry point that runs it.
        type Stage = (&'static str, fn(&HevcStageInputs) -> Vec<u8>);
        let stages: [Stage; 7] = [
            ("inter_pred", HevcStageInputs::run_inter_pred),
            ("intra_pred", HevcStageInputs::run_intra_pred),
            ("deblock", HevcStageInputs::run_deblock),
            ("sao", HevcStageInputs::run_sao),
            ("inverse_transform", HevcStageInputs::run_inverse_transform),
            ("color_convert", HevcStageInputs::run_color_convert),
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

    /// The SAO grid has to run the §7.3.8.3 parameter mix issue #310 measured,
    /// not a saturated picture.
    ///
    /// The share that matters most is the one that used to be zero: on real
    /// content SAO is switched *off* on most CTBs, and an off CTB is returned
    /// from before it reads a sample. A grid drifting back towards filtering
    /// everything would read as a faster kernel when it is really a wider
    /// workload, so the emitted mix is pinned rather than trusted.
    #[test]
    fn the_sao_grid_runs_the_measured_parameter_mix() {
        let inputs = HevcStageInputs::new(1920, 1088);
        let total = inputs.sao_ctbs.len() as f64;
        assert!(total > 0.0, "the grid has CTBs");
        for (row, cidx) in [(0usize, 0usize), (1, 1), (1, 2)] {
            for class in 0..6 {
                let want = SAO_CTB_MIX[row][class];
                let got = inputs
                    .sao_ctbs
                    .iter()
                    .filter(|r| {
                        let c = &r.components[cidx];
                        let seen = match c.sao_type_idx {
                            0 => 0,
                            1 => 1,
                            _ => 2 + usize::from(c.eo_class),
                        };
                        seen == class
                    })
                    .count() as f64
                    / total;
                assert!(
                    (got - want).abs() < 0.01,
                    "component {cidx} class {class}: {got:.4} against the measured {want:.4}"
                );
            }
        }
        // Cb and Cr are signalled as one in §7.3.8.3, so they resolve alike.
        assert!(
            inputs.sao_ctbs.iter().all(|r| r.components[1].sao_type_idx
                == r.components[2].sao_type_idx
                && r.components[1].eo_class == r.components[2].eo_class),
            "Cb and Cr take the same SAO type and class"
        );
        // Most of the picture is switched off, so the classifiers run over a
        // fraction of it rather than all of it.
        let all = (1920u64 * 1088) * 3 / 2;
        assert!(
            inputs.sao_samples() < all / 4,
            "{} filtered samples against {all} in the picture",
            inputs.sao_samples()
        );
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
            inputs.run_color_convert(),
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
        // Three 64x64 cells fit a 256x128 frame once the one-cell margin the
        // second reference list needs is taken off each edge, and every cell
        // is filled by prediction units of one class whatever that class is.
        assert_eq!(inputs.inter_pred_samples(), 3 * 64 * 64);
        // Only the CTBs the measured mix leaves switched on are counted, so
        // this is a fraction of the 256x128 4:2:0 picture rather than all of it.
        assert!(
            inputs.sao_samples() > 0 && inputs.sao_samples() < 256 * 128 * 3 / 2,
            "sao_samples was {}",
            inputs.sao_samples()
        );
        assert_eq!(
            inputs.deblock_samples(),
            inputs.luma_edges().len() as u64 * 32
        );
        assert_eq!(inputs.cabac_bins(), CABAC_BINS as u64);
        assert!(inputs.intra_pred_samples() > 0);
        assert!(inputs.inverse_transform_samples() > 0);
    }

    /// The picture-only path exists so a whole-frame benchmark can leave the
    /// output conversion out of the measured interval — which is only sound if
    /// it is otherwise the same decode. Converting what it collects has to
    /// reproduce, frame for frame and byte for byte, what the public
    /// `submit` returns.
    #[test]
    fn the_picture_only_path_decodes_what_the_rgba_path_decodes() {
        let (configuration, samples) = pcm_stream();
        let limits = Limits::default();
        let cancellation = CancellationToken::new();

        let mut rgba_decoder = new_decoder(&configuration, &limits);
        let mut expected = Vec::new();
        for sample in &samples {
            for frame in rgba_decoder.submit(sample, &cancellation).unwrap() {
                expected.push(frame.frame);
            }
        }
        assert!(
            !expected.is_empty(),
            "the generated stream has to produce output frames for this to test anything"
        );

        let mut picture_decoder = new_decoder(&configuration, &limits);
        let mut actual = Vec::new();
        for sample in &samples {
            for picture in picture_decoder
                .submit_pictures(sample, &cancellation)
                .unwrap()
            {
                actual.push(picture_to_rgba(&picture, &configuration, &limits).unwrap());
            }
        }
        assert_eq!(actual, expected);
    }

    /// The digest the picture-only whole-frame group returns is what the
    /// harness compares across instruction sets, so it has to depend on the
    /// decoded samples and be reproducible.
    #[test]
    fn decode_pictures_folds_the_decoded_samples_reproducibly() {
        let (configuration, samples) = pcm_stream();
        let limits = Limits::default();
        let one = decode_pictures(&configuration, &samples, &limits, 1);
        assert_eq!(one, decode_pictures(&configuration, &samples, &limits, 1));
        assert_ne!(one, decode_pictures(&configuration, &samples, &limits, 2));
        assert_ne!(one, Digest::new().finish());
    }

    /// The two whole-frame surfaces have to decode the same frames, so the
    /// interval between them is the conversion and not a different workload.
    #[test]
    fn the_two_whole_frame_paths_fold_the_same_number_of_frames() {
        let (configuration, samples) = pcm_stream();
        let limits = Limits::default();
        let frames = decode_frames(&configuration, &samples, &limits, 2);
        assert_eq!(frames, decode_frames(&configuration, &samples, &limits, 2));
        assert_ne!(
            frames,
            decode_pictures(&configuration, &samples, &limits, 2)
        );
        assert_ne!(frames, Digest::new().finish());
    }

    /// The conversion stage folds a real converted frame, and folds a
    /// different one for different content.
    #[test]
    fn color_convert_folds_the_converted_frame() {
        let inputs = small_inputs();
        let converted = inputs.run_color_convert();
        assert_eq!(converted, inputs.run_color_convert());
        assert_ne!(converted, HevcStageInputs::new(128, 64).run_color_convert());
        assert_eq!(inputs.color_convert_samples(), 256 * 128);
    }

    /// A short PCM-coded stream, built with the crate's own HEVC encoder so
    /// the decode-side tests run on a real bitstream without a fixture.
    fn pcm_stream() -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
        use super::super::encoder::{hvcc_box, length_prefixed_vcl};
        use super::super::engine::encoder::pcm::encode_idr_pcm_au;
        use crate::FrameIndex;

        const SIDE: usize = 16;
        const FRAMES: u64 = 4;
        let mut rng = Lcg::new(0x8EEF_0220);
        let y: Vec<u8> = (0..SIDE * SIDE).map(|_| rng.below(256) as u8).collect();
        let chroma: Vec<u8> = (0..SIDE * SIDE / 4).map(|_| rng.below(256) as u8).collect();
        let au = encode_idr_pcm_au(&y, &chroma, &chroma, SIDE, SIDE).unwrap();
        let data = length_prefixed_vcl(&au).unwrap();
        let mut configuration = convert_config(SIDE, SIDE);
        configuration.configuration = hvcc_box(&au).unwrap();
        let samples = (0..FRAMES)
            .map(|index| EncodedVideoSample {
                presentation_index: FrameIndex(index),
                random_access: true,
                data: data.clone(),
            })
            .collect();
        (configuration, samples)
    }

    fn new_decoder(configuration: &VideoDecoderConfig, limits: &Limits) -> HevcDecoder {
        let parsed = ParsedConfiguration::parse(configuration, limits).unwrap();
        HevcDecoder::new(configuration.clone(), *limits, parsed).unwrap()
    }
}
