//! The single-tile, all-intra encoder: superblock/partition iteration (§5.11.2/.4), DC intra
//! prediction (§7.11.2.5), the forward transforms, and coefficient coding with full context
//! derivation (§5.11.39, §8.3.2).
//!
//! Two quantization profiles are coded, selected by the frame's `base_q_idx` exactly as
//! [`crate::av1_intra_decoder`] selects its two reconstruction paths:
//!
//! - `base_q_idx == 0` (`CodedLossless`): every transform block is a 4x4 forward WHT
//!   ([`super::wht::fwht4x4`]) of the source residual, coded as-is. Prediction neighbours are the
//!   source samples, which under lossless coding *are* the reconstruction.
//! - `base_q_idx != 0` (non-lossless): each coding block picks a square transform size through
//!   §5.11.16 `read_tx_size` under `TX_MODE_SELECT`, and each transform block picks a `tx_type`
//!   from the reduced intra set the decoder reads back (§5.11.47). Coefficients come from
//!   [`super::transform::forward_transform`], are quantized against the same `dc_q`/`ac_q` steps
//!   [`crate::av1_intra::inverse_transform`] dequantizes with, and are then dequantized and
//!   inverse-transformed here so prediction runs off the same reconstruction the decoder builds.
//!
//! All coded blocks are square `DC_PRED` intra blocks. Lossless frames code one 64x64
//! `PARTITION_NONE` block per superblock except at the right/bottom frame edges, where the spec's
//! forced splitting applies; non-lossless frames additionally split by rate-distortion cost down
//! to [`MIN_PARTITION_WIDTH`], and always split a block that would not fit whole inside the coded
//! frame so that every transform block the decoder reconstructs is in bounds. The frame is coded
//! on the MI-unit grid (`mi_cols*4 x mi_rows*4`, i.e. dimensions rounded up to a multiple of 8);
//! the out-of-frame padding is edge-replicated and cropped away on decode.

// Adapted and modified from gamut, Copyright (c) 2026 Justin Chung, MIT licensed.
use super::cdf;
use super::symbol::SymbolEncoder;
use super::transform::forward_transform;
use super::wht::fwht4x4;
use crate::av1_intra::{Av1TxType, dq_denom, get_ac_quant, get_dc_quant, inverse_transform};
use crate::av1_intra_pred::add_residual_row;
use crate::av1_simd::coeff;

/// `NUM_BASE_LEVELS` (§3).
const NUM_BASE_LEVELS: i32 = 2;
/// `NUM_BASE_LEVELS + COEFF_BASE_RANGE`, the golomb threshold (§5.11.39).
const COEFF_BASE_PLUS_RANGE: i32 = 14;
/// Largest square transform [`super::transform::forward_transform`] implements. `TX_64X64` has an
/// inverse kernel but no forward one, so a 64x64 coding block always signals a `tx_depth` of at
/// least 1.
const MAX_FORWARD_TX: usize = 32;
/// Sentinel for an unfilled memo slot; every real answer encodes as a smaller byte.
const MEMO_UNSET: u8 = u8::MAX;
/// Partition-tree levels the memos cover, one per `Mi_Width_Log2` a coding block can have
/// (`bw` of 8, 16, 32 and 64, so `bsl` 1 through 4).
const MEMO_LEVELS: usize = 5;

/// A 64x64 superblock's side in MI (4-sample) units, which is the grid
/// [`FrameEncoder::encode_superblocks`] walks and the one the per-column transform-gain
/// accumulators are indexed on.
const SB4: usize = 16;

/// Bits [`estimate_rate`] charges one CDF-coded symbol.
///
/// The estimator ranks candidates rather than predicting the arithmetic coder's fractional
/// output, so every symbol §5.11.39 writes is charged the same nominal cost and the *shape* of
/// the estimate comes from how many symbols a level costs rather than from any one symbol's
/// probability. `2` keeps that nominal cost on the scale the raw bits in the same expression are
/// already on, so a symbol and a literal are comparable.
const SYMBOL_BITS: i64 = 2;
/// Bits [`estimate_rate`] charges a block whose levels are all zero: the `all_zero` symbol alone.
const ZERO_BLOCK_BITS: i64 = SYMBOL_BITS;
/// Bits [`estimate_rate`] charges the cheapest block it can charge that is *not* all zero: the
/// `all_zero` symbol, a one-position end-of-block, and one magnitude-1 coefficient with its sign.
const MIN_CODED_BLOCK_BITS: i64 = ZERO_BLOCK_BITS + SYMBOL_BITS + 3;

/// Transform blocks per probing size trial that [`FrameEncoder::choose_tx_size`] searches with the
/// whole transform-type set instead of the set's DCT alone, to measure what the type search is
/// worth at that size before extrapolating it over the trial's remaining blocks.
const TYPE_GAIN_PROBES: usize = 1;

/// [`TYPE_GAIN_PROBES`] for a size search that can reach a transform whose `Dq_Denom` is not 1.
const LARGE_TYPE_GAIN_PROBES: usize = 4;

/// Coding blocks between two whose size search probes.
///
/// What a probe measures is a ratio between the whole type set's cost and DCT's on the same
/// block, and that ratio is a property of the frame's content and its quantizer far more than of
/// the individual block - so it is measured on a sample of the size searches and reused across
/// the rest, rather than re-measured on every one. The sample is taken per *coding block* rather
/// than per size trial: a block ranks its sizes against each other, so all of its trials have to
/// be corrected by the same kind of estimate or the ranking compares a freshly measured size
/// against a remembered one. The frame's first size search probes, so a correction is available
/// from the first block that needs one.
///
/// The ratio is only stable frame-wide while the frame's content is, and the reuse is what costs
/// accuracy when it is not: a block corrected by a ratio measured in a region unlike its own can
/// have two close sizes ranked the wrong way round. That used to be what bounded this constant
/// from above. It is not any more, and `8` is chosen from a different column than the one that
/// first set it.
///
/// Re-measured with [`TYPE_GAIN_TRUST`] in force, on the eight-frame set in
/// `measure_type_gain_sampling_intervals` - a hard scene edge, a four-quadrant frame, full-range
/// noise, a smooth surface, directional edges, bands, a mosaic, and the encoder's own
/// `test_pattern` - at 192x160 and at the 128x96 the ceilings below are set from, against the
/// same estimator probing every size search. Cost is the encoder's own `sse + lambda * bits`,
/// summed over the frame and compared at equal quantizer; times are the minimum of five
/// interleaved rounds per arm in `measure_type_gain_sampling_cost`, and candidates are the
/// 192x160 set's:
///
/// | interval | worst penalty vs unsampled | mean vs exhaustive | candidates | time |
/// |---------:|---------------------------:|-------------------:|-----------:|--------:|
/// | 1        | 0.0%                       | +0.19%             | 243,694    | 0.640 s |
/// | 2        | +1.3%                      | -0.36%             | 203,477    | 0.587 s |
/// | 4        | +3.5%                      | -0.42%             | 183,677    | 0.563 s |
/// | 8        | +3.4%                      | -0.41%             | 174,638    | 0.549 s |
/// | 16       | +1.5%                      | -0.55%             | 167,546    | 0.542 s |
///
/// The penalty column above was measured under the rate model #299 replaced, and it is what
/// moved this constant to `8`: with `estimate_rate` charging a level `2 + 2 * bit_length(level)`,
/// the sampled estimator's own error swamped the sampling rate, every interval from `1` to `64`
/// landed within +3.5% of the unsampled estimator, and nothing on the rate-distortion side
/// disqualified a longer stride. #278 took the candidate saving that was left, #323 and #329
/// established that neither `TX_4X4` coverage nor rate-distortion bounded the value from above,
/// and #332 replaced the assertion that used to pin it with one holding the swept range to a
/// penalty bound and the remaining candidate saving to under a tenth.
///
/// #299 priced a level the way §5.11.39 codes one, and the ordinary upper bound came back. The
/// interval is once again bounded by the distortion the shortcuts are allowed to cost, and
/// sharply: on the 96x80 `test_pattern` the shipped `2` reconstructs within 0.033 dB of the
/// exhaustive search at its worst quantizer, while `3`, `4`, `6`, `8`, `16`, `32` and `64` all
/// sit at 0.203 dB - four times the 0.05 dB
/// `the_search_shortcuts_stay_within_their_rate_and_distortion_bound` allows, flat across the
/// whole range rather than drifting into it, so this is a property of the sampling and not of
/// where a stride's phase happens to land. `1` is not the value either: it probes every size
/// search and reconstructs *worse* than `2` at `qindex` 1, because a trial that probes is
/// corrected in full and an over-large correction moves the ranking away from what the
/// exhaustive search would have chosen.
///
/// So the value returns to `2` and the upper bound is asserted as what it now is, by
/// `a_longer_type_gain_sampling_interval_costs_more_distortion_than_the_bound_allows`, which
/// replaces #332's. The per-frame ceilings in
/// `the_type_gain_per_frame_penalties_are_pinned_at_the_shipped_sampling_interval` are re-measured
/// at `2` as that test's own rule requires; they come out *tighter* than they were at `8`, the
/// whole set fitting under 1% except `scene_edge` at +1.99%, where `bands` had needed 4%. The
/// candidate saving `8` was taken for is given back - it is not available at a distortion the
/// shortcuts are allowed to spend - and the search still clears the four-fold reduction those
/// bounds assert, at 4.26x.
pub(super) const TYPE_GAIN_SAMPLE_INTERVAL: usize = 2;

/// Probes a transform size's accumulated gain ratio remembers, as the window of an exponential
/// recency weighting.
///
/// The ratio a probe measures is a property of the *content around the block it was measured
/// on*, not of the frame: on a frame whose statistics change across it, a block corrected by a
/// ratio measured on the other side of that change can have two close sizes ranked the wrong way
/// round. Accumulating every probe of a size equally over the whole frame is what made the
/// correction frame-wide. So each accumulator is aged by `(n-1)/n` before a new probe joins it,
/// and a probe's weight decays geometrically over the following `n` probes - which, since the
/// size searches that probe are visited in the decoder's superblock raster order, makes the
/// ratio a block reads back the one its own neighbourhood measured.
///
/// `1` is the shortest window there is: a non-probing trial is corrected by the single most
/// recent probe at that size and by nothing older. On its own that is too noisy an estimate to
/// rank a size on - it costs `smooth` +12.97% against the unsampled estimator un-shrunk - which
/// is why it is paired with [`TYPE_GAIN_TRUST`] at half, and why the two are documented as one
/// choice there rather than as two independent ones. #308 removed this window entirely on the
/// grounds that the shrinkage subsumed it; that has not held since #299's rate model.
///
/// #343 recorded that `measure_type_gain_memory_windows` had gone flat on the tuning set of the
/// day - every window from one probe to frame-wide accumulation measuring the identical penalty -
/// so the window was no longer chosen by a measurement either. It separates again on the three
/// `gain_*` frames that issue added, which vary the *type gain* across a region boundary rather
/// than the residual energy: `measure_type_gain_separation` reads a spread of up to 1.7% across
/// the windows on `gain_bands` and up to 0.77% on `gain_edge`, against 0.000% on all but two of
/// the twelve frame-and-quantizer pairs the old set offers.
///
/// What still fixes the value is the distortion bound, and sharply. On the extended set at the
/// shipped [`TYPE_GAIN_PROBE_TRUST`], `measure_type_gain_calibration` puts every window from `3`
/// upwards - and frame-wide accumulation with them - at 0.159 dB below the exhaustive search at
/// `qindex` 1, three times the 0.05 dB
/// `the_search_shortcuts_stay_within_their_rate_and_distortion_bound` allows, at every shrinkage.
/// Only `1` holds it, at the shrinkages [`TYPE_GAIN_TRUST`] is chosen from. The longer windows
/// are marginally cheaper on the tuning set - +1.22% worst against `1`'s +1.44% - and that is not
/// a trade the bound permits.
///
/// The decay is one multiply and one divide per accumulator, paid only on the sampled trials.
pub(super) const TYPE_GAIN_MEMORY: usize = 1;

/// Transform sizes [`FrameEncoder::type_gain`] accumulates over: `TX_4X4` through `TX_32X32`,
/// which is every size [`super::transform::forward_transform`] implements.
const TYPE_GAIN_SIZES: usize = 4;

/// What a probed transform block is identified by: its position on the coded grid, its transform
/// size, and the DC prediction it was measured against - the four values that determine its
/// residual and so everything the search derives from it. Only the coverage measurement in
/// `measure_probe_reuse_coverage` reads these back; see [`FrameEncoder::probe_keys`].
#[cfg(test)]
type ProbeKey = (usize, usize, usize, u8);

/// One transform size's accumulated probe measurement for the frame.
#[derive(Clone, Copy, Default)]
struct TypeGain {
    /// Summed DCT-only cost of every block probed at this size.
    dct_cost: i64,
    /// Summed best-of-set cost of the same blocks, so `dct_cost - best_cost` is what the type
    /// search has been measured to be worth at this size.
    best_cost: i64,
    /// Probes accumulated, which is how many transform blocks [`Self::dct_cost`] and
    /// [`Self::best_cost`] are summed over. It is the divisor of a *per-block* measured gain,
    /// which is what every [`GainModel`] but [`GainModel::Linear`] extrapolates from, and it is
    /// also what [`Self::ratio`]'s mean divides by.
    probes: i64,
    /// The same measurement as the mean of the probes' *own* ratios, in [`GAIN_RATIO_ONE`]
    /// units, so an expensive probe does not outweigh a cheap one. Measured and rejected; kept
    /// for [`GainRatio`].
    #[cfg(test)]
    ratio: i64,
}

/// Where a trial that did not probe reads its gain ratio back from.
///
/// Shipped encodes always use [`GainLocality::Blended`]; the other arms exist so
/// `measure_type_gain_locality` can measure it against them on the same frames.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GainLocality {
    /// The running accumulator alone, which is what #272 shipped: probe order, so the probes
    /// nearest a block are a horizontal run of coding blocks.
    Running,
    /// The superblock column's accumulator alone, whose most recent probes before the current
    /// superblock are the ones directly above it.
    Column,
    /// Both, summed, so the ratio a block reads back is measured on its own neighbourhood in
    /// both axes.
    Blended,
}

/// How a remembered gain ratio is averaged over the probes it remembers.
///
/// Shipped encodes always use [`GainRatio::Mean`]; [`GainRatio::Weighted`] is what #272 shipped
/// and is kept so `measure_type_gain_ratio` can measure the two against each other.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GainRatio {
    /// Ratio of the summed costs, which weights each probe by how expensive its block was.
    Weighted,
    /// Mean of the probes' own ratios, which weights every probe equally.
    Mean,
}

/// How much of a *remembered* gain a trial that did not probe is corrected by, in
/// [`TYPE_GAIN_TRUST_ONE`] sixteenths. A trial that probed is corrected in full by what it
/// measured on its own block; only an estimate carried from other blocks is shrunk.
///
/// #281 chose `2` against a surface whose worst case was `scene_edge` at +9.32%, and #308 then
/// removed the recency window on the grounds that this shrinkage subsumed it - both measured
/// while [`estimate_rate`] charged a level `2 + 2 * bit_length(level)`. #299 replaced that with
/// the symbol-counting model §5.11.39 actually codes, which removed most of the penalty both
/// mechanisms existed to absorb, and #343 recorded what was left: `measure_type_gain_trust` and
/// `measure_type_gain_memory_windows` read *flat* on the whole tuning set, every frame within
/// 0.63% at every quantizer and at every value of either knob. Neither constant was chosen by a
/// measurement of its own any more, and what actually selected the pair was
/// `measure_type_gain_memory_against_trust`: a 66-cell grid of which `(memory 1, trust 8)` was
/// the only cell holding both the 0.05 dB distortion bound and the per-frame ceilings at once.
///
/// Two things changed that, and both are measurements rather than a re-search of the grid.
///
/// **The tuning set separates the type gain now.** Every frame `content_frames` had varied the
/// residual *energy* across a boundary, which is the right axis for [`TYPE_GAIN_SAMPLE_INTERVAL`]
/// (whose error is a stale cost) and the wrong one for this shrinkage, which carries the ratio
/// `(dct - best) / dct` across a boundary. A boundary between two regions that both have a near
/// zero type gain does not move that ratio however sharply the energy jumps, which is why the
/// sweeps read flat once the rate model stopped mispricing the levels a boundary produces. The
/// three `gain_*` frames put a region the non-DCT types win on - a sawtooth ramp, hard diagonal
/// edges - next to full-range noise, where they cannot, so the ratio genuinely changes across the
/// boundary at every quantizer. `measure_type_gain_separation` is the check: on `gain_edge` and
/// `gain_bands` the shrinkage moves the frame's penalty by 0.03% to 2.2% at almost every
/// quantizer and both sizes, against the 0.000% the old set reads at all but two of twelve.
///
/// **The correction a trial measures on its own blocks is shrunk too.** That was the last
/// unmeasured assumption in the mechanism, and [`TYPE_GAIN_PROBE_TRUST`] is where it is measured;
/// with it in force this shrinkage is no longer carrying the whole calibration alone. On the
/// extended set, `measure_type_gain_calibration` puts the worst frame of the grid at +1.2% to
/// +1.8% for every cell but the un-shrunk corner, against +1.99% to +12.97% before, and the
/// 0.05 dB bound holds over a `13..=16` by `8..=10` region of the `(probe_trust, trust)` plane
/// rather than at one point of it. `8` sits inside that region on both axes.
///
/// What the value says, read as a policy, is what it always said: correct a trial by a ratio it
/// measured on its own blocks at seven eighths, and one carried from elsewhere in the frame at
/// half of that, a remembered ratio being worth less than a measured one. The difference is that
/// both halves of it are now measured. `measure_unified_type_gain` is the check on the policy
/// itself - correcting *every* trial from the accumulator, so the two are not distinguished at
/// all, costs the tuning set's worst frame between +6.4% and +37.9% at every window and
/// shrinkage, and misses the distortion bound at 0.203 dB almost everywhere.
///
/// The residual 0.2 dB the DCT-only ranking carries at `qindex` 1 is not this constant's to fix,
/// and `measure_type_gain_probes_against_trust` is what establishes that: measuring the ratio on
/// *more* blocks makes it monotonically worse - -0.344 dB at two probes, -0.460 dB at six - while
/// the candidate reduction falls under the 4x the bound requires. A sharper estimate of the ratio
/// produces a worse size ranking. #349 read that as the extrapolation model - the correction
/// assumes the type search's gain scales with a trial's searched cost, which over-credits the
/// smallest size at a quantizer where every block is coded - and built every replacement that
/// reading asks for. The over-credit is not the error: it is what the correction is *worth*. A
/// credit that does not depend on how much of the trial was probed is flat in the probe count,
/// which is the monotonicity a model error would have to lose, and it reconstructs exactly where
/// crediting nothing does. See [`GainModel`], where the sweep and its four corners are recorded.
/// So no number of probes and no setting of these constants reaches the residual, and they are
/// biases rather than a sharper estimator.
///
/// No bound, ceiling or test was relaxed to reach this pair. The per-frame ceilings were
/// re-measured at the new constants as that test's own rule requires: `scene_edge` *tightens*
/// from 2.5% to 1%, the whole set fits under 1% except `mosaic` at +1.22% and `gain_bands` at
/// +1.44%, and the worst frame of the tuning set falls from +1.99% to +1.44%.
///
/// # This is a bias, not an estimator's shrinkage
///
/// A shrinkage presumes there is a quantity being estimated and that the estimate is noisy. That
/// is not the situation here, and #356 measured why. What a probe measures is the type set's gain
/// against a DCT-only trial's reconstruction and coefficient contexts, and the emitting pass
/// produces neither: it writes the type its own search picks and reconstructs from it. A size
/// trial's cost is therefore a counterfactual the encoder never emits, and every ranking built on
/// it - corrected or not - inherits that, which is why #349 found no shape of credit that moved a
/// single size decision.
///
/// `a_context_consistent_size_trial_ranks_like_the_exhaustive_search_and_costs_like_it` shows the
/// counterfactual is the whole of the residual: a trial that codes each of its blocks with the
/// type the emitting pass would pick, and keeps it, reproduces the exhaustive search's size
/// decisions *exactly* at every quantizer measured, byte for byte and to 0.000 dB - not three
/// decisions better, but all of them. It also costs a candidate reduction of only 1.69x-1.93x
/// against the exhaustive search, where `the_search_shortcuts_stay_within_their_rate_and_
/// distortion_bound` requires 4x, so it cannot ship at any setting of anything here.
///
/// So this constant is not shrinking a noisy measurement of the right quantity towards its mean.
/// It is damping a measurement of the *wrong* quantity - one taken in contexts the frame will not
/// have - because damping it happens to cost less than believing it. That makes it a bias term,
/// permanently and by construction rather than until a better estimator arrives, and the grid
/// search above is the honest way to pick a bias: sweep it against the assertions it has to hold
/// and take the cell that holds them. Read `8` as "believe half of a measurement known to be of
/// the wrong thing", and re-derive it by re-running that grid whenever the rate model, the
/// quantizer or the type set moves - not by reasoning about how noisy a probe is.
///
/// None of this is particular to a *remembered* ratio. A probe measures the same wrong quantity
/// whichever trial reads it back, so [`TYPE_GAIN_PROBE_TRUST`] is a bias on the same grounds and
/// is documented there in the same terms; the two differ only in whose blocks the measurement
/// came from, which is a matter of how much of it to damp rather than of what it is a
/// measurement of.
pub(super) const TYPE_GAIN_TRUST: i64 = 8;

/// How much of a gain a trial *measured on its own blocks* is corrected by, in
/// [`TYPE_GAIN_TRUST_ONE`] sixteenths.
///
/// The correction extrapolates a ratio measured on [`TYPE_GAIN_PROBES`] of a trial's blocks over
/// all of them, and until #343 a trial that probed was corrected by that ratio in *full*. Nothing
/// measured that it should be. It was the trial's own measurement, so it was believed entirely,
/// and only a ratio carried from other blocks was shrunk - by [`TYPE_GAIN_TRUST`], which was then
/// left carrying the whole calibration and, on the tuning set of the day, could not be chosen by
/// a sweep of its own.
///
/// A full-strength correction is not what the measurements support. `measure_dct_only_ranking_error`
/// is the sharpest form of it: an estimator probing *every* size search, where every trial is
/// corrected in full and no ratio is ever remembered, reconstructs 0.344 dB *below* the exhaustive
/// search at `qindex` 1 on the 96x80 `test_pattern`, against the sampled estimator's +0.023 dB.
/// The extrapolation over-credits, and it over-credits most where a trial has the most blocks to
/// extrapolate over, which is the smallest size - and #349 measured that the over-credit is the
/// correction's whole value rather than its error, which is why this constant damps it rather
/// than the model being replaced. [`GainModel`] records that sweep.
///
/// `measure_type_gain_probe_trust` sweeps this against [`TYPE_GAIN_TRUST`] over the whole tuning
/// set - `content_frames` including the three `gain_*` frames added for #343, plus `test_pattern`,
/// at 128x96 and 192x160 over six quantizers - and `14` is where it lands. At the shipped
/// shrinkage the tuning set's worst frame falls from +1.99% at the un-shrunk `16` to +1.67% at
/// `14` and `15`, back to +1.86% at `13` and to +3.62% at `12`; the mean improves from -0.085% to
/// -0.145%; and the 0.05 dB distortion bound holds over the whole `13..=16` by `trust` `8..=10`
/// region rather than at the single cell of a 66-cell grid #343 recorded. `14` is the interior of
/// that region on both axes, and it is chosen by the tuning set rather than by the bound.
///
/// This costs nothing: the shrinkage is the multiply and divide that was already there.
///
/// # This is a bias, not an estimator's shrinkage
///
/// Read as a shrinkage, `14` would say a trial's own probe is a noisy estimate of the gain its
/// blocks would see and should be pulled towards the mean by an eighth. It is not, and #356
/// measured why - the same measurement that made [`TYPE_GAIN_TRUST`] a bias, and it applies here
/// in full. A probe measures the type set's gain against a *DCT-only* trial's reconstruction and
/// coefficient contexts, and the emitting pass has neither: it writes the type its own search
/// picks and reconstructs from it. That a probe was taken on the trial's own blocks changes whose
/// blocks the wrong quantity was measured on, not that it is the wrong quantity. Probing more of
/// them does not help either, which is `measure_type_gain_probes_against_trust`'s result: the
/// reconstruction goes monotonically further below the exhaustive search, -0.344 dB at two probes
/// and -0.460 dB at six, while the candidate reduction falls under the 4x
/// `the_search_shortcuts_stay_within_their_rate_and_distortion_bound` requires.
///
/// `a_context_consistent_size_trial_ranks_like_the_exhaustive_search_and_costs_like_it` is where
/// that counterfactual is shown to be the whole of the residual rather than part of it, and it is
/// also why no setting of this constant can be the fix: a trial that codes each of its blocks
/// with the type the emitting pass would pick, and keeps it, reproduces the exhaustive search's
/// size decisions exactly, byte for byte and to 0.000 dB, at a candidate reduction of 1.69x-1.93x
/// that cannot ship. So `14` damps a measurement of the wrong quantity because damping it costs
/// less than believing it - a bias term, permanently and by construction rather than until a
/// better estimator arrives.
///
/// Re-derive it the way a bias is derived, by re-running `measure_type_gain_probe_trust`'s sweep
/// against [`TYPE_GAIN_TRUST`] and taking the interior of the region that holds the assertions,
/// whenever the rate model, the quantizer or the type set moves. Do not reason about it as a
/// confidence in a probe against noise.
pub(super) const TYPE_GAIN_PROBE_TRUST: i64 = 14;

/// [`TYPE_GAIN_TRUST`] denominator: the un-shrunk correction.
const TYPE_GAIN_TRUST_ONE: i64 = 16;

/// The shape the transform-gain correction extrapolates a probe's measurement in.
///
/// A probe measures what the whole transform-type set is worth against the set's `DCT_DCT` on
/// `p` of a trial's blocks, and the correction has to turn that into what the set is worth on
/// the trial's other `b - p`. How it does that is the *model*, and it is separable from how much
/// of the result is believed ([`TYPE_GAIN_TRUST`], [`TYPE_GAIN_PROBE_TRUST`]) and from which
/// blocks the measurement came off ([`TYPE_GAIN_SAMPLE_INTERVAL`], [`TYPE_GAIN_MEMORY`]).
///
/// Only the shipped [`TYPE_GAIN_MODEL`] is constructed outside tests; the others are reachable
/// from [`FrameEncoder::with_type_gain_model`], which a non-test build does not have.
///
/// # Shipped encodes use [`GainModel::Linear`], and #349 is why
///
/// #349 read the residual at `base_q_idx` 1 as this extrapolation: the credit grows linearly in
/// the trial's searched cost, so it over-credits the size with the most blocks to extrapolate
/// over, and a sharper probe ranks monotonically worse, which is what a model error looks like.
/// The three replacements that reading asks for are the other variants here, and
/// `measure_type_gain_models` crosses them with the probe count on the 96x80 `test_pattern`.
/// What it measures is that the credit has no useful freedom in its shape, only in its
/// magnitude, and that the two bounds fix that magnitude from opposite sides:
///
/// | credit | `qindex` 1 at 1 probe | at 64 probes | worst reduction |
/// |:-------|----------------------:|-------------:|----------------:|
/// | none (the DCT-only ranking) | -0.203 dB | -0.203 dB | 4.77x |
/// | [`Linear`](GainModel::Linear), shipped | +0.023 dB | -0.125 dB | 4.26x |
/// | [`PerBlock`](GainModel::PerBlock) | +0.023 dB | -0.125 dB | 4.26x |
/// | [`Saturating`](GainModel::Saturating) at one block | -0.203 dB | -0.203 dB | 4.77x |
/// | [`Amplified`](GainModel::Amplified) at 48 sixteenths | +0.324 dB | +0.105 dB | 2.46x |
///
/// Read across, that is one number in four disguises. Counting the gain per block rather than
/// per unit of searched cost moves nothing at any probe count, so the cost weighting is not the
/// axis the residual lives on. Making the credit independent of how much of the trial was
/// probed, which is what a saturation does and what the monotonicity #349 asks for requires,
/// satisfies that monotonicity exactly - and satisfies it by crediting nothing: it lands on the
/// bare DCT-only ranking at every probe count. And the value the shipped credit does buy comes
/// from *over*-crediting: a trial's first block measures a larger gain than the trial's mean,
/// and the linear extrapolation multiplies that one-block sample over every block of the trial.
/// Probing more blocks removes the over-credit, which is why a sharper estimate ranks worse.
///
/// [`Amplified`](GainModel::Amplified) is that over-credit stated rather than sampled into
/// existence: about three times the trial's own measured per-block gain, credited whatever the
/// probe count, so the probe count only sharpens the measurement. It restores `base_q_idx` 1
/// with every block of every trial probed - and takes the candidate reduction to 2.46x, against
/// the 4x `the_search_shortcuts_stay_within_their_rate_and_distortion_bound` requires, because
/// the credit that keeps the smallest size reachable is also what makes it expensive.
///
/// So the ranking wants more credit at `base_q_idx` 1, the candidate bound wants less, and the
/// shipped configuration is the cell where the two meet.
/// `the_type_gain_credit_is_a_magnitude_rather_than_a_shape` asserts all four corners of that,
/// so a future change to the shape has to move those numbers rather than restate the reading.
/// It is also what #356 predicts: the correction is a scalar credit against a cost measured in
/// contexts the emitting pass will not have, and the only degree of freedom such a credit has is
/// how much of it to believe. [`TYPE_GAIN_TRUST`] and [`TYPE_GAIN_PROBE_TRUST`] therefore stay,
/// and stay biases.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GainModel {
    /// The gain scales with the trial's searched cost: `(dct_p - best_p) * dct_b / dct_p`, where
    /// `dct_b` is [`FrameEncoder::trial_searched_cost`]. This is what #272 shipped and what #349
    /// replaced, and it is kept so a sweep can measure the others against it.
    Linear,
    /// The gain scales with the trial's searched *block count*: `(dct_p - best_p) * b / p`. An
    /// expensive block is credited the same as a cheap one at the same size, which is the axis
    /// the type set's win is actually counted on - each block is one more chance for a non-DCT
    /// type to beat `DCT_DCT`.
    PerBlock,
    /// The same per-block gain, damped by a credit that saturates in the trial's block count:
    /// `(dct_p - best_p) * e(b) / p` with `e(b) = b * s / (b + s - 1)` and `s`
    /// [`TYPE_GAIN_SATURATION`]. `e(1)` is `1` and `e` approaches `s` from below, so a trial of
    /// one block is credited what it measured and a trial of sixty-four is credited far less than
    /// sixty-four times it.
    Saturating,
    /// The per-block gain over the trial's blocks, scaled by [`TYPE_GAIN_AMPLIFICATION`]
    /// sixteenths: `(dct_p - best_p) * b * a / (p * 16)`.
    ///
    /// This is the only shape that can put the credit *above* what the type set was measured to
    /// win, which is where the shipped estimator's one-block sample already puts it: a trial's
    /// first block measures a larger gain than its mean, so the shipped correction over-credits
    /// by an amount nothing states and every sharper measurement removes. `a` states it, so the
    /// probe count only sharpens the per-block gain instead of also changing how much of it is
    /// credited. `16` is [`PerBlock`](GainModel::PerBlock) exactly.
    Amplified,
}

/// Blocks the [`GainModel::Saturating`] credit saturates at: the `s` of `e(b) = b * s / (b+s-1)`.
///
/// This is the one number in the model that says how far a probe's measurement carries, and it
/// is what makes the credit stop depending on how much of a trial was probed - the monotonicity
/// #349 asks for, which [`GainModel`] records is bought by crediting nothing. Nothing ships it,
/// so `4` is only the value `measure_type_gain_models` sweeps around.
pub(super) const TYPE_GAIN_SATURATION: usize = 4;

/// Sixteenths [`GainModel::Amplified`] credits the trial's measured per-block gain at.
///
/// `16` is the trial's own measurement, un-amplified. `48` is where the shipped estimator's
/// one-block sample already puts the credit, and is the value [`GainModel`]'s table and
/// `the_type_gain_credit_is_a_magnitude_rather_than_a_shape` read the model at; nothing ships
/// either, so this is only the value `measure_type_gain_models` sweeps around.
pub(super) const TYPE_GAIN_AMPLIFICATION: i64 = 16;

/// The extrapolation shipped encodes use, which is the one #349 measured the alternatives
/// against and did not displace. Every other [`GainModel`] is reachable only from
/// [`FrameEncoder::with_type_gain_model`], which a non-test build does not have.
pub(super) const TYPE_GAIN_MODEL: GainModel = GainModel::Linear;

/// Fixed-point scale of [`TypeGain::ratio`], which is a fraction in `0..=1` and needs the
/// resolution an integer ratio of costs would otherwise lose.
#[cfg(test)]
const GAIN_RATIO_ONE: i64 = 1 << 20;

/// Slot in [`FrameEncoder::type_gain`] for a transform of `tx_width` samples a side.
fn type_gain_slot(tx_width: usize) -> usize {
    (tx_width / 4).trailing_zeros() as usize
}

/// Smallest coding block the non-lossless partition search will produce. A 16x16 block can still
/// signal `TX_16X16`, `TX_8X8`, or `TX_4X4`, so every transform size this encoder emits stays
/// reachable without searching partitions all the way down to 8x8.
const MIN_PARTITION_WIDTH: usize = 16;

/// Encoder for the single tile that spans the whole frame.
pub(crate) struct FrameEncoder<'a> {
    plane: &'a [u8],
    width: usize,
    height: usize,
    mi_cols: usize,
    mi_rows: usize,
    coded_w: usize,
    coded_h: usize,
    sym: SymbolEncoder,
    above_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_level: Vec<u8>,
    left_dc: Vec<u8>,
    /// `Mi_Width_Log2` of the block covering each MI cell (for the partition context).
    mi_bsl: Vec<u8>,
    /// `AboveTxWidth`/`LeftTxHeight` (§9.3's `tx_depth` context): the transform width and height
    /// the last coded block left on each MI column and row.
    above_tx_width: Vec<u8>,
    left_tx_height: Vec<u8>,
    /// `base_q_idx`; `0` is the lossless WHT profile.
    qindex: u8,
    /// `get_dc_quant(qindex)` / `get_ac_quant(qindex)`, the dequantization steps the decoder
    /// applies, so forward quantization is their exact inverse.
    dc_quant: i32,
    ac_quant: i32,
    /// Lagrange multiplier of the `sse + lambda * bits` cost the transform-size, transform-type,
    /// and partition searches minimize.
    lambda: i64,
    /// The non-lossless reconstruction, `coded_w x coded_h`, which prediction reads exactly as
    /// the decoder reads its own. Empty for lossless frames, which predict from the source.
    recon: Vec<u8>,
    /// Every `(size, tx_type)` pair actually written to the bitstream, in coding order, so tests
    /// can assert what the round trip covered rather than assume it.
    #[cfg(test)]
    emitted: Vec<(usize, Av1TxType)>,
    /// Memoized `decide_split` answers, one slot per `(block size, MI position)`.
    ///
    /// A block at `(r, c, bw)` is only ever searched from one encoder state per frame: every
    /// speculative trial rolls the state back with [`Snapshot`] before the next one, and the
    /// emitting pass walks the same decision tree in the same order, so it reaches each block
    /// with exactly the state its trial saw. Without the memo the emitting pass re-runs the
    /// whole subtree search that the trial just ran, once per level of the partition tree.
    split_memo: Vec<u8>,
    /// Memoized `choose_tx_size` answers, in the same slot layout as [`Self::split_memo`].
    tx_size_memo: Vec<u8>,
    /// Transform blocks the current size trial may still probe with the whole type set. Zero
    /// outside a [`FrameEncoder::choose_tx_size`] trial, which is what keeps every other
    /// speculative pass on the set's DCT alone.
    probe_budget: usize,
    /// What the type search has measured out to be worth at each transform size so far this
    /// frame, one slot per [`type_gain_slot`]. Probes accumulate into it and a trial that did not
    /// probe reads it back, so a size's gain is measured on a sample of its trials rather than on
    /// all of them.
    type_gain: [TypeGain; TYPE_GAIN_SIZES],
    /// The same measurement kept per superblock column, one [`TYPE_GAIN_SIZES`] block per column
    /// of the frame. A probe joins its own column's accumulator as well as the running one, and
    /// because a column is only revisited a superblock row later, the probes it still remembers
    /// when a superblock starts are the ones directly above it. Measured and rejected - it does
    /// not move the frame #272's recency weighting left behind - so only [`GainLocality`] reads
    /// it and nothing outside tests maintains it.
    #[cfg(test)]
    column_gain: Vec<TypeGain>,
    /// Superblock column of the coding block whose size search is running, which is the
    /// [`Self::column_gain`] block that search probes into and reads back.
    #[cfg(test)]
    gain_column: usize,
    /// The current trial's own probe measurement, or a zeroed pair when it did not probe. A
    /// trial that probed is corrected by what it measured itself, exactly as every trial was
    /// before the measurement was sampled; only the trials that skip the probe fall back to
    /// [`Self::type_gain`].
    probe_dct_cost: i64,
    probe_best_cost: i64,
    /// Transform blocks the current trial probed, which is the divisor of the per-block gain
    /// [`GainModel::PerBlock`] and [`GainModel::Saturating`] extrapolate from. It is not always
    /// [`Self::type_gain_probes`]: a trial with fewer blocks than that probes all of them.
    probe_blocks: i64,
    /// Size searches run so far this frame, which is what [`TYPE_GAIN_SAMPLE_INTERVAL`] samples.
    size_searches: usize,
    /// Summed DCT-only cost of every block the current trial actually searched, which is the base
    /// the measured gain is extrapolated over. Zero-skipped blocks are excluded: no transform type
    /// can improve a block that codes no coefficients.
    trial_searched_cost: i64,
    /// How many blocks that cost is summed over, which is the base a per-block gain is
    /// extrapolated over. Zero-skipped blocks are excluded from it for the same reason.
    trial_searched_blocks: i64,
    /// Cost a running size trial may reach before it is abandoned, or [`i64::MAX`] when nothing
    /// is being trialled. A trial's cost is a sum of squared errors and `lambda * bits`, so it
    /// only grows: once the partial sum passes what the incumbent size already costs, no
    /// remaining block can bring it back and the rest of the trial cannot change the answer.
    /// [`FrameEncoder::choose_tx_size`] sets it from the incumbent, for any trial whose ranking
    /// cost [`Self::trial_rank_bound`] can bound from below by the sum this accumulates.
    trial_ceiling: i64,
    /// The credit the running trial's ranking will subtract from that sum, as the
    /// `(measured, dct)` pair [`GainModel::Linear`] scales by [`Self::trial_searched_cost`], or
    /// `(0, 0)` when the trial is ranked on the raw sum. Fixed for the whole trial:
    /// [`FrameEncoder::shipped_trial_credit`] only accepts a trial that does not probe, so the
    /// accumulator it is read from cannot move while the trial runs.
    trial_credit: (i64, i64),
    /// Whether the pass now running is a [`FrameEncoder::choose_tx_size`] size trial, as opposed
    /// to one of the partition search's own measurement passes. Context-consistency is a
    /// property of the size trial alone: it is the size ranking the counterfactual distorts, and
    /// making every other speculative pass code the full type set as well is what made it cost
    /// the exhaustive search.
    in_size_trial: bool,
    /// Set when the trial [`Self::trial_ceiling`] bounds gave up part-way, so
    /// [`FrameEncoder::choose_tx_size`] knows the cost it got back is a partial sum that must not
    /// be ranked.
    trial_abandoned: bool,
    /// Set by [`Self::without_search_shortcuts`] to restore the original exhaustive search, so a
    /// test can compare the shortcuts against the search they stand in for.
    #[cfg(test)]
    exhaustive: bool,
    /// Set by [`Self::with_reversed_candidate_order`] to walk the transform-type and
    /// transform-size candidates backwards, so a test can prove no decision depends on the order
    /// they are evaluated in.
    #[cfg(test)]
    reversed_candidates: bool,
    /// Set by [`Self::with_context_consistent_trials`] to make a size trial code each of its
    /// blocks with the type the emitting pass would pick, keeping that type's reconstruction and
    /// contexts, instead of ranking on the set's DCT and correcting the ranking afterwards. This
    /// is the trial whose ranking is the emitting pass's own; nothing ships it, because
    /// `a_context_consistent_size_trial_costs_more_candidates_than_the_shortcuts_allow` measures
    /// what it costs.
    #[cfg(test)]
    context_consistent_trials: bool,
    /// Set by [`Self::with_context_consistent_size_trials`] to make context-consistency a
    /// property of the *size trial* alone, leaving every other speculative pass on the set's
    /// `DCT_DCT` exactly as the shipped search leaves it. This is the affordable half of the
    /// arm above, and `a_bounded_context_consistent_size_trial_keeps_the_ranking_and_still_costs_too_much`
    /// is what it is measured by.
    #[cfg(test)]
    consistent_size_trials: bool,
    /// Set by [`Self::with_unbounded_size_trials`] to stop bounding the shipped size trial by
    /// the incumbent size's cost, so a test can measure what that bound buys and show that
    /// turning it off changes nothing but the candidate count.
    #[cfg(test)]
    unbounded_size_trials: bool,
    /// Set by [`Self::without_type_gain_correction`] to rank sizes on the type set's `DCT_DCT`
    /// alone - no probe, no accumulator, no correction - which is the bare ranking every
    /// correction is applied on top of, and the baseline any of them has to beat.
    #[cfg(test)]
    no_type_gain_correction: bool,
    /// The sampling interval in force, so a test can sweep it and measure what
    /// [`TYPE_GAIN_SAMPLE_INTERVAL`] costs at each value instead of asserting the shipped one is
    /// right. Outside tests the constant is read directly.
    #[cfg(test)]
    type_gain_interval: usize,
    /// Whether every size search that can reach the smallest transform probes, whatever the
    /// stride says. This is the guarantee #323 weighed against the phase dependence it would
    /// remove; `measure_type_gain_phase_aliasing` prices it and nothing ships it, so it is off
    /// outside that measurement.
    #[cfg(test)]
    force_smallest_size_probes: bool,
    /// One entry per size search this frame, in the order they ran: the smallest transform width
    /// the search could have chosen, and whether it probed. Together they say which strides ever
    /// probe a trial that carries a given size, which is what `measure_type_gain_phase_aliasing`
    /// reports.
    #[cfg(test)]
    size_search_probes: Vec<(usize, bool)>,
    /// The locality arm in force, so a test can measure the shipped one against the accumulators
    /// it blends. [`GainLocality::Blended`] outside tests.
    #[cfg(test)]
    type_gain_locality: GainLocality,
    /// Transform blocks a probing size trial measures with the whole type set, so a test can
    /// sweep it. [`TYPE_GAIN_PROBES`] outside tests.
    #[cfg(test)]
    type_gain_probes: usize,
    /// How a remembered ratio is averaged, so a test can sweep it. [`GainRatio::Mean`] outside
    /// tests.
    #[cfg(test)]
    type_gain_ratio: GainRatio,
    /// The recency window in force, so a test can sweep it. [`TYPE_GAIN_MEMORY`] outside tests.
    #[cfg(test)]
    type_gain_memory: usize,
    /// The shrinkage in force, in the units of [`TYPE_GAIN_TRUST`], so a test can sweep it.
    #[cfg(test)]
    type_gain_trust: i64,
    /// The shrinkage a trial's own probe measurement is corrected by, so a test can sweep it.
    /// [`TYPE_GAIN_PROBE_TRUST`] outside tests.
    #[cfg(test)]
    type_gain_probe_trust: i64,
    /// Whether every trial reads its ratio back from the accumulator, including one that probed,
    /// so a test can measure the probing asymmetry away. `false` outside tests.
    #[cfg(test)]
    unified_type_gain: bool,
    /// The extrapolation in force, so a test can sweep the credit's shape against the probe
    /// count. [`TYPE_GAIN_MODEL`] outside tests.
    #[cfg(test)]
    type_gain_model: GainModel,
    /// The block count [`GainModel::Saturating`]'s credit saturates at, so a test can sweep it.
    /// [`TYPE_GAIN_SATURATION`] outside tests.
    #[cfg(test)]
    type_gain_saturation: usize,
    /// The sixteenths [`GainModel::Amplified`] credits at, so a test can sweep it.
    /// [`TYPE_GAIN_AMPLIFICATION`] outside tests.
    #[cfg(test)]
    type_gain_amplification: i64,
    /// Transform-type candidates actually transformed, quantized and reconstructed, which is the
    /// work the shortcuts exist to remove.
    #[cfg(test)]
    candidates_evaluated: u64,
    /// Every transform-size decision this frame's searches made, for [`SearchReport`].
    #[cfg(test)]
    size_choices: Vec<(usize, usize, usize, usize)>,
    /// Reused buffers for the per-block coefficient context pass.
    coeff_ctx: CoeffScratch,
    /// Exact `sse + lambda * bits` ties the transform-type and transform-size searches had to
    /// settle, so a test can show the tie-break is reached on real content rather than assert an
    /// order-independence that holds only because nothing ever tied.
    #[cfg(test)]
    cost_ties: u64,
    /// Size trials [`Self::trial_ceiling`] abandoned part-way, so a test can show the bound is
    /// reached on real content rather than assert a saving that never fires.
    #[cfg(test)]
    abandoned_trials: u64,
    /// Transform blocks a context-consistent size trial searched with the whole type set, which
    /// is the base the candidate budget in
    /// `measure_context_consistent_trial_candidate_budget` is divided over.
    #[cfg(test)]
    consistent_trial_blocks: u64,
    /// Every [`ProbeKey`] a size trial probed, in the order they were measured, against the key
    /// the emitting pass actually reached at each transform block's position. How far the two
    /// sets overlap is what any reuse of a probe's result could ever have covered, which
    /// `measure_probe_reuse_coverage` reports and #291 closed on.
    #[cfg(test)]
    probe_keys: Vec<ProbeKey>,
    /// `(size, prediction)` of every transform block the emitting pass wrote, by its position.
    #[cfg(test)]
    emitted_blocks: std::collections::HashMap<(usize, usize), (usize, u8)>,
    /// Coding blocks the emitting pass wrote, which bounds how many probes could ever be read
    /// back: a size trial probes one block, so no emitted coding block can consume more than one.
    #[cfg(test)]
    emitted_coding_blocks: u64,
    /// Size searches that probed, which bounds it further under the sampling interval.
    #[cfg(test)]
    probing_size_searches: u64,
    /// Emitted blocks the zero-block shortcut decided, which no probe ever ran on.
    #[cfg(test)]
    zero_skipped_emitted: u64,
    /// Emitted blocks that ran a transform-type search over more than one candidate, which is
    /// every emitted block a probe could ever have stood in for: a zero-skipped block never
    /// searches, and a size whose derived set names one type has nothing to choose between.
    #[cfg(test)]
    reusable_emitted: u64,
}

/// What one encode of a tile did, for the tests that compare two searches against each other.
#[cfg(test)]
pub(crate) struct SearchReport {
    pub(crate) tile: Vec<u8>,
    pub(crate) reconstruction: Vec<u8>,
    pub(crate) coded_width: usize,
    pub(crate) trace: Vec<(usize, Av1TxType)>,
    pub(crate) candidates_evaluated: u64,
    /// Every transform-size decision the frame's searches made, as `(row, column, block width,
    /// chosen transform width)` in MI units, so a measurement can attribute a difference between
    /// two searches to a place in the frame rather than to a position in the trace.
    pub(crate) size_choices: Vec<(usize, usize, usize, usize)>,
    pub(crate) cost_ties: u64,
    /// Size trials the incumbent size's cost bound abandoned part-way, whether the trial was a
    /// shipped one or a context-consistent one.
    pub(crate) abandoned_trials: u64,
    /// Transform blocks a context-consistent size trial searched with the whole type set.
    pub(crate) consistent_trial_blocks: u64,
    /// The probe and emit key sets and the counts that bound their overlap, for
    /// `measure_probe_reuse_coverage`.
    pub(crate) probe_keys: Vec<(usize, usize, usize, u8)>,
    pub(crate) emitted_blocks: std::collections::HashMap<(usize, usize), (usize, u8)>,
    pub(crate) emitted_coding_blocks: u64,
    pub(crate) probing_size_searches: u64,
    /// Every size search's smallest reachable transform width and whether it probed, in search
    /// order, for `measure_type_gain_phase_aliasing`.
    pub(crate) size_search_probes: Vec<(usize, bool)>,
    pub(crate) zero_skipped_emitted: u64,
    pub(crate) reusable_emitted: u64,
}

/// One transform type considered for a block, with everything the winner needs to be written:
/// the `tx_type` symbol's index in its set, the quantized levels, the reconstructed residual,
/// and the `sse + lambda * bits` cost the search minimizes.
struct TxCandidate {
    symbol: usize,
    /// Only the trace the tests assert on reads this back; the bitstream carries `symbol`.
    #[cfg_attr(not(test), allow(dead_code))]
    tx_type: Av1TxType,
    levels: Vec<i32>,
    reconstructed: Vec<i16>,
    cost: i64,
}

/// Reusable buffers for one transform block's §8.3.2 coefficient contexts.
///
/// The contexts are derived for the whole block in a single pass before the serial symbol loop
/// runs, which is legal because every neighbour `coeff_base` and `coeff_br` consult lies later in
/// the up-right diagonal scan than the position consulting it, and the loop walks that scan
/// backwards — so those neighbours are already final (or, past the end-of-block, zero) before the
/// first symbol is written. See [`crate::av1_simd::coeff`] for the vector kernel that pass
/// dispatches to.
///
/// The buffers live on the encoder rather than the block so a frame's hundreds of thousands of
/// transform blocks share one allocation each. The padded plane's zero border is written once per
/// size and never again, so a block only overwrites its own `size * size` interior.
#[derive(Default)]
pub(crate) struct CoeffScratch {
    /// The size the buffers below are currently shaped for; `0` before the first block.
    size: usize,
    /// Clamped magnitudes in the zero-padded layout `crate::av1_simd::coeff` reads.
    plane: Vec<i32>,
    /// `coeff_base` context per raster position.
    pub(crate) base: Vec<i32>,
    /// `coeff_br` context per raster position.
    pub(crate) br: Vec<i32>,
}

impl CoeffScratch {
    /// Reshapes the buffers for a `size x size` block, rewriting the padded plane's zero border
    /// only when the size actually changed.
    fn resize(&mut self, size: usize) {
        if self.size == size {
            return;
        }
        self.size = size;
        coeff::reset_padded_plane(&mut self.plane, size);
        self.base.clear();
        self.base.resize(size * size, 0);
        self.br.clear();
        self.br.resize(size * size, 0);
    }

    /// Fills [`Self::base`] and [`Self::br`] for every position of a block whose quantized
    /// coefficients are `quant`, through the vector kernel when the active instruction set has
    /// one and through `tile.rs`'s scalar reference otherwise.
    pub(crate) fn derive(&mut self, quant: &[i32], size: usize) {
        self.resize(size);
        let isa = coeff::active_isa();
        if coeff::has_vector_kernel(isa) {
            coeff::fill_padded_levels(&mut self.plane, quant, size);
            if crate::av1_simd::coeff_contexts(isa, &self.plane, size, &mut self.base, &mut self.br)
            {
                return;
            }
        }
        for pos in 0..size * size {
            self.base[pos] = coeff_base_ctx(pos, quant, size) as i32;
            self.br[pos] = coeff_br_ctx(pos, quant, size) as i32;
        }
    }
}

/// The block-local encoder state a speculative (non-emitting) trial mutates, saved so the trial
/// can be rolled back and the winning candidate replayed from the same starting point.
struct Snapshot {
    x0: usize,
    y0: usize,
    recon_width: usize,
    recon: Vec<u8>,
    above_start: usize,
    above_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_start: usize,
    left_level: Vec<u8>,
    left_dc: Vec<u8>,
}

impl<'a> FrameEncoder<'a> {
    /// Creates an encoder over one tightly packed 8-bit monochrome plane at `qindex`
    /// (`base_q_idx`); `0` selects the lossless profile.
    pub(crate) fn new(plane: &'a [u8], width: usize, height: usize, qindex: u8) -> Self {
        let mi_cols = 2 * ((width + 7) >> 3);
        let mi_rows = 2 * ((height + 7) >> 3);
        let coded_w = mi_cols * 4;
        let coded_h = mi_rows * 4;
        let ac_quant = get_ac_quant(qindex);
        Self {
            plane,
            width,
            height,
            mi_cols,
            mi_rows,
            coded_w,
            coded_h,
            sym: SymbolEncoder::new(),
            above_level: vec![0; mi_cols],
            above_dc: vec![0; mi_cols],
            left_level: vec![0; mi_rows],
            left_dc: vec![0; mi_rows],
            mi_bsl: vec![0; mi_cols * mi_rows],
            above_tx_width: vec![0; mi_cols],
            left_tx_height: vec![0; mi_rows],
            qindex,
            dc_quant: get_dc_quant(qindex),
            ac_quant,
            // Distortion is summed squared error in the sample domain and rate is counted in
            // bits, so the multiplier scales with the square of the quantization step, the
            // standard `lambda ~ q^2` relation. The divisor is a plain tuning constant.
            lambda: (i64::from(ac_quant) * i64::from(ac_quant) / 256).max(1),
            recon: if qindex == 0 {
                Vec::new()
            } else {
                vec![0; coded_w * coded_h]
            },
            #[cfg(test)]
            emitted: Vec::new(),
            split_memo: vec![MEMO_UNSET; MEMO_LEVELS * mi_cols * mi_rows],
            tx_size_memo: vec![MEMO_UNSET; MEMO_LEVELS * mi_cols * mi_rows],
            probe_budget: 0,
            type_gain: [TypeGain::default(); TYPE_GAIN_SIZES],
            #[cfg(test)]
            column_gain: vec![TypeGain::default(); mi_cols.div_ceil(SB4) * TYPE_GAIN_SIZES],
            #[cfg(test)]
            gain_column: 0,
            probe_dct_cost: 0,
            probe_best_cost: 0,
            probe_blocks: 0,
            size_searches: 0,
            trial_searched_cost: 0,
            trial_searched_blocks: 0,
            trial_ceiling: i64::MAX,
            trial_credit: (0, 0),
            in_size_trial: false,
            trial_abandoned: false,
            #[cfg(test)]
            exhaustive: false,
            #[cfg(test)]
            reversed_candidates: false,
            #[cfg(test)]
            context_consistent_trials: false,
            #[cfg(test)]
            consistent_size_trials: false,
            #[cfg(test)]
            unbounded_size_trials: false,
            #[cfg(test)]
            no_type_gain_correction: false,
            #[cfg(test)]
            type_gain_interval: TYPE_GAIN_SAMPLE_INTERVAL,
            #[cfg(test)]
            force_smallest_size_probes: false,
            #[cfg(test)]
            size_search_probes: Vec::new(),
            #[cfg(test)]
            type_gain_locality: GainLocality::Running,
            #[cfg(test)]
            type_gain_probes: TYPE_GAIN_PROBES,
            #[cfg(test)]
            type_gain_ratio: GainRatio::Weighted,
            #[cfg(test)]
            type_gain_trust: TYPE_GAIN_TRUST,
            #[cfg(test)]
            type_gain_probe_trust: TYPE_GAIN_PROBE_TRUST,
            #[cfg(test)]
            unified_type_gain: false,
            #[cfg(test)]
            type_gain_model: TYPE_GAIN_MODEL,
            #[cfg(test)]
            type_gain_saturation: TYPE_GAIN_SATURATION,
            #[cfg(test)]
            type_gain_amplification: TYPE_GAIN_AMPLIFICATION,
            #[cfg(test)]
            type_gain_memory: TYPE_GAIN_MEMORY,
            #[cfg(test)]
            candidates_evaluated: 0,
            #[cfg(test)]
            size_choices: Vec::new(),
            coeff_ctx: CoeffScratch::default(),
            #[cfg(test)]
            cost_ties: 0,
            #[cfg(test)]
            abandoned_trials: 0,
            #[cfg(test)]
            consistent_trial_blocks: 0,
            #[cfg(test)]
            probe_keys: Vec::new(),
            #[cfg(test)]
            emitted_blocks: std::collections::HashMap::new(),
            #[cfg(test)]
            emitted_coding_blocks: 0,
            #[cfg(test)]
            probing_size_searches: 0,
            #[cfg(test)]
            zero_skipped_emitted: 0,
            #[cfg(test)]
            reusable_emitted: 0,
        }
    }

    /// Turns off every search shortcut, leaving the exhaustive `sse + lambda * bits` search over
    /// all partitions, sizes and types. Only the memoization stays on, because it answers from
    /// the search rather than in place of it.
    #[cfg(test)]
    pub(crate) fn without_search_shortcuts(mut self) -> Self {
        self.exhaustive = true;
        self
    }

    /// Evaluates every transform-type and transform-size candidate in the reverse of the order
    /// the searches normally walk. A search whose ties are settled by a total order over the
    /// candidates encodes identically either way; one that keeps whichever equal-cost candidate
    /// it happened to see first does not.
    #[cfg(test)]
    pub(crate) fn with_reversed_candidate_order(mut self) -> Self {
        self.reversed_candidates = true;
        self
    }

    /// Makes every transform-size trial code its blocks with the type the emitting pass would
    /// pick, and keep it.
    ///
    /// The shipped trial ranks a size on the type set's `DCT_DCT` alone and probes a sample of
    /// its blocks with the rest of the set purely to *measure* - the block keeps DCT's result, so
    /// the trial's reconstruction and coefficient contexts are a DCT-only trial's and the
    /// measurement is credited back to the trial's total in [`Self::corrected_trial_cost`]. That
    /// is a counterfactual: the emitting pass writes the type its own search picks and
    /// reconstructs from it, so neither the cost the trial ranked on nor the contexts it built
    /// are the ones the frame goes on to produce. Under this arm the trial *is* the emitting
    /// pass, block for block, and no correction is applied on top of it, which is the only trial
    /// whose ranking is by construction the emitting pass's own. It exists to price that, not to
    /// ship: it costs the full type set on every searched block of every size candidate.
    #[cfg(test)]
    pub(crate) fn with_context_consistent_trials(mut self) -> Self {
        self.context_consistent_trials = true;
        self
    }

    /// Makes only the transform-*size* trial context-consistent, and leaves every other
    /// speculative pass - the partition search's whole-block and split-subtree measurements -
    /// ranking on the set's `DCT_DCT` as the shipped search does.
    ///
    /// [`Self::with_context_consistent_trials`] makes every non-emitting pass code the full type
    /// set, which is why it costs the exhaustive search. But the counterfactual #356 identified
    /// is a property of the *size* ranking: it is the size trial whose cost and contexts the
    /// emitting pass never reproduces. Restricting the consistency to that trial is the
    /// reduction #388 asks for under "a consistency applied only to the blocks whose ranking it
    /// changes", and it is exact where it matters - the size decisions this arm makes are still
    /// the exhaustive search's, on every coding block at every quantizer measured. What it gives
    /// up is the byte-for-byte identity with the exhaustive search, which was being carried by
    /// the *partition* passes rather than by the size trial.
    #[cfg(test)]
    pub(crate) fn with_context_consistent_size_trials(mut self) -> Self {
        self.consistent_size_trials = true;
        self
    }

    /// Whether only the size trial is context-consistent. Never, outside tests.
    #[cfg(test)]
    fn consistent_size_trials(&self) -> bool {
        self.consistent_size_trials
    }

    /// Whether only the size trial is context-consistent. Never, outside tests.
    #[cfg(not(test))]
    fn consistent_size_trials(&self) -> bool {
        false
    }

    /// Whether a size trial codes its blocks with the type the emitting pass would pick. Never,
    /// outside tests.
    #[cfg(test)]
    fn context_consistent_trials(&self) -> bool {
        self.context_consistent_trials
    }

    /// Whether a size trial codes its blocks with the type the emitting pass would pick. Never,
    /// outside tests.
    #[cfg(not(test))]
    fn context_consistent_trials(&self) -> bool {
        false
    }

    /// Stops bounding a shipped size trial by the incumbent size's cost, leaving every trial to
    /// run to its last block as it did before #398.
    ///
    /// The bound is exact, so this arm is the same encode: it exists so a test can measure the
    /// candidates the bound removes, and assert that removing them changed nothing else.
    #[cfg(test)]
    pub(crate) fn with_unbounded_size_trials(mut self) -> Self {
        self.unbounded_size_trials = true;
        self
    }

    /// Whether a shipped size trial may be abandoned at the incumbent's cost. Always, outside
    /// tests.
    #[cfg(test)]
    fn bounded_size_trials(&self) -> bool {
        !self.unbounded_size_trials
    }

    /// Whether a shipped size trial may be abandoned at the incumbent's cost. Always, outside
    /// tests.
    #[cfg(not(test))]
    fn bounded_size_trials(&self) -> bool {
        true
    }

    /// Ranks transform sizes on the set's `DCT_DCT` alone, with no probe and no correction, which
    /// is the ranking [`TYPE_GAIN_TRUST`] exists to correct.
    #[cfg(test)]
    pub(crate) fn without_type_gain_correction(mut self) -> Self {
        self.no_type_gain_correction = true;
        self
    }

    /// Whether the transform-gain correction runs at all. Always, outside tests.
    #[cfg(test)]
    fn type_gain_correction(&self) -> bool {
        !self.no_type_gain_correction
    }

    /// Whether the transform-gain correction runs at all. Always, outside tests.
    #[cfg(not(test))]
    fn type_gain_correction(&self) -> bool {
        true
    }

    /// Whether the search shortcuts are on. Always, outside tests.
    #[cfg(test)]
    fn shortcuts(&self) -> bool {
        !self.exhaustive
    }

    /// Whether the search shortcuts are on. Always, outside tests.
    #[cfg(not(test))]
    fn shortcuts(&self) -> bool {
        true
    }

    /// Overrides the probe sampling interval, so a test can measure the estimator at intervals
    /// other than the shipped one. `1` probes every size search, which is the unsampled search
    /// the shipped interval approximates.
    #[cfg(test)]
    pub(crate) fn with_type_gain_interval(mut self, interval: usize) -> Self {
        assert!(interval >= 1, "a sampling interval of 0 samples nothing");
        self.type_gain_interval = interval;
        self
    }

    /// Coding blocks between two whose size search probes. [`TYPE_GAIN_SAMPLE_INTERVAL`] outside
    /// tests, where nothing can override it.
    #[cfg(test)]
    fn type_gain_interval(&self) -> usize {
        self.type_gain_interval
    }

    /// Coding blocks between two whose size search probes. [`TYPE_GAIN_SAMPLE_INTERVAL`] outside
    /// tests, where nothing can override it.
    #[cfg(not(test))]
    fn type_gain_interval(&self) -> usize {
        TYPE_GAIN_SAMPLE_INTERVAL
    }

    /// Probes every size search that can reach the smallest transform, on top of the ones the
    /// stride samples. This is the structural guarantee #323 asked for, kept only so
    /// `measure_type_gain_phase_aliasing` can price what it costs; a shipped encode samples on
    /// the stride alone.
    #[cfg(test)]
    pub(crate) fn with_forced_smallest_size_probes(mut self) -> Self {
        self.force_smallest_size_probes = true;
        self
    }

    /// Whether a size search that can reach the smallest transform probes whatever the stride
    /// says. Never, outside the measurement that priced it.
    #[cfg(test)]
    fn force_smallest_size_probes(&self) -> bool {
        self.force_smallest_size_probes
    }

    /// Whether a size search that can reach the smallest transform probes whatever the stride
    /// says. Never, outside the measurement that priced it.
    #[cfg(not(test))]
    fn force_smallest_size_probes(&self) -> bool {
        false
    }

    /// Overrides where a trial reads its gain ratio back from, so a test can measure the shipped
    /// blend against each accumulator on its own.
    #[cfg(test)]
    pub(crate) fn with_type_gain_locality(mut self, locality: GainLocality) -> Self {
        self.type_gain_locality = locality;
        self
    }

    /// The locality arm in force. Only a test can select anything but
    /// [`GainLocality::Running`], which is what the encoder does.
    #[cfg(test)]
    fn type_gain_locality(&self) -> GainLocality {
        self.type_gain_locality
    }

    /// Overrides how many of a probing trial's blocks are measured with the whole type set, so a
    /// test can measure what a noisier or steadier probe is worth.
    #[cfg(test)]
    pub(crate) fn with_type_gain_probes(mut self, probes: usize) -> Self {
        assert!(
            probes >= 1,
            "a trial that probes measures at least one block"
        );
        self.type_gain_probes = probes;
        self
    }

    /// Blocks a probing trial measures. [`TYPE_GAIN_PROBES`] outside tests.
    #[cfg(test)]
    fn type_gain_probes(&self) -> usize {
        self.type_gain_probes
    }

    /// Blocks a probing trial measures. [`TYPE_GAIN_PROBES`] outside tests.
    #[cfg(not(test))]
    fn type_gain_probes(&self) -> usize {
        TYPE_GAIN_PROBES
    }

    /// Overrides how a remembered ratio is averaged, so a test can measure the shipped mean
    /// against the cost-weighted ratio of sums it replaced.
    #[cfg(test)]
    pub(crate) fn with_type_gain_ratio(mut self, ratio: GainRatio) -> Self {
        self.type_gain_ratio = ratio;
        self
    }

    /// How a remembered ratio is averaged. Only a test can select anything but
    /// [`GainRatio::Weighted`], which is what the encoder does.
    #[cfg(test)]
    fn type_gain_ratio(&self) -> GainRatio {
        self.type_gain_ratio
    }

    /// Overrides the recency window, so a test can measure the estimator between remembering one
    /// probe and remembering the whole frame. `usize::MAX` disables the decay, which is the
    /// frame-wide accumulation this replaced.
    #[cfg(test)]
    pub(crate) fn with_type_gain_memory(mut self, memory: usize) -> Self {
        assert!(
            memory >= 1,
            "a window of 0 would divide the accumulator by zero"
        );
        self.type_gain_memory = memory;
        self
    }

    /// Probes a size's gain ratio remembers. [`TYPE_GAIN_MEMORY`] outside tests, where nothing
    /// can override it.
    #[cfg(test)]
    fn type_gain_memory(&self) -> usize {
        self.type_gain_memory
    }

    /// Probes a size's gain ratio remembers. [`TYPE_GAIN_MEMORY`] outside tests, where nothing
    /// can override it.
    #[cfg(not(test))]
    fn type_gain_memory(&self) -> usize {
        TYPE_GAIN_MEMORY
    }

    /// Overrides how far a remembered gain is shrunk toward no correction at all, so a test can
    /// sweep it. `16` is the un-shrunk correction and `0` no correction.
    #[cfg(test)]
    pub(crate) fn with_type_gain_trust(mut self, trust: i64) -> Self {
        assert!(
            (0..=TYPE_GAIN_TRUST_ONE).contains(&trust),
            "shrinkage is a fraction"
        );
        self.type_gain_trust = trust;
        self
    }

    /// The shrinkage in force. [`TYPE_GAIN_TRUST`] outside tests.
    #[cfg(test)]
    fn type_gain_trust(&self) -> i64 {
        self.type_gain_trust
    }

    /// The shrinkage in force. [`TYPE_GAIN_TRUST`] outside tests.
    #[cfg(not(test))]
    fn type_gain_trust(&self) -> i64 {
        TYPE_GAIN_TRUST
    }

    /// Overrides how far a trial's *own* probe measurement is shrunk, so a test can sweep it.
    #[cfg(test)]
    pub(crate) fn with_type_gain_probe_trust(mut self, trust: i64) -> Self {
        assert!(
            (0..=TYPE_GAIN_TRUST_ONE).contains(&trust),
            "shrinkage is a fraction"
        );
        self.type_gain_probe_trust = trust;
        self
    }

    /// The measured-gain shrinkage in force. [`TYPE_GAIN_PROBE_TRUST`] outside tests.
    #[cfg(test)]
    fn type_gain_probe_trust(&self) -> i64 {
        self.type_gain_probe_trust
    }

    /// The measured-gain shrinkage in force. [`TYPE_GAIN_PROBE_TRUST`] outside tests.
    #[cfg(not(test))]
    fn type_gain_probe_trust(&self) -> i64 {
        TYPE_GAIN_PROBE_TRUST
    }

    /// Corrects every trial from the accumulator, including one that probed, so the correction no
    /// longer depends on which coding blocks the stride happened to sample.
    #[cfg(test)]
    pub(crate) fn with_unified_type_gain(mut self) -> Self {
        self.unified_type_gain = true;
        self
    }

    /// Whether a trial that probed still reads its own measurement back. `false` outside tests.
    #[cfg(test)]
    fn unified_type_gain(&self) -> bool {
        self.unified_type_gain
    }

    /// Whether a trial that probed still reads its own measurement back. `false` outside tests.
    #[cfg(not(test))]
    fn unified_type_gain(&self) -> bool {
        false
    }

    /// Overrides the extrapolation the correction uses, so a sweep can cross the credit's shape
    /// with the probe count instead of asserting the shipped shape is right.
    #[cfg(test)]
    pub(crate) fn with_type_gain_model(mut self, model: GainModel) -> Self {
        self.type_gain_model = model;
        self
    }

    /// The extrapolation in force. [`TYPE_GAIN_MODEL`] outside tests.
    #[cfg(test)]
    fn type_gain_model(&self) -> GainModel {
        self.type_gain_model
    }

    /// The extrapolation in force. [`TYPE_GAIN_MODEL`] outside tests.
    #[cfg(not(test))]
    fn type_gain_model(&self) -> GainModel {
        TYPE_GAIN_MODEL
    }

    /// Overrides where [`GainModel::Saturating`]'s credit saturates, so a sweep can choose it.
    #[cfg(test)]
    pub(crate) fn with_type_gain_saturation(mut self, blocks: usize) -> Self {
        assert!(
            blocks >= 1,
            "a credit saturating at no blocks credits nothing"
        );
        self.type_gain_saturation = blocks;
        self
    }

    /// Where [`GainModel::Saturating`]'s credit saturates. [`TYPE_GAIN_SATURATION`] outside
    /// tests.
    #[cfg(test)]
    fn type_gain_saturation(&self) -> i64 {
        self.type_gain_saturation as i64
    }

    /// Where [`GainModel::Saturating`]'s credit saturates. [`TYPE_GAIN_SATURATION`] outside
    /// tests.
    #[cfg(not(test))]
    fn type_gain_saturation(&self) -> i64 {
        TYPE_GAIN_SATURATION as i64
    }

    /// Overrides how far [`GainModel::Amplified`] credits a measured per-block gain, so a sweep
    /// can choose it.
    #[cfg(test)]
    pub(crate) fn with_type_gain_amplification(mut self, sixteenths: i64) -> Self {
        self.type_gain_amplification = sixteenths;
        self
    }

    /// The sixteenths [`GainModel::Amplified`] credits at. [`TYPE_GAIN_AMPLIFICATION`] outside
    /// tests.
    #[cfg(test)]
    fn type_gain_amplification(&self) -> i64 {
        self.type_gain_amplification
    }

    /// The sixteenths [`GainModel::Amplified`] credits at. [`TYPE_GAIN_AMPLIFICATION`] outside
    /// tests.
    #[cfg(not(test))]
    fn type_gain_amplification(&self) -> i64 {
        TYPE_GAIN_AMPLIFICATION
    }

    /// Encodes the tile and returns the symbol-coded bytes (`decode_tile`, §5.11.2).
    pub(crate) fn encode(mut self) -> Vec<u8> {
        self.encode_superblocks();
        self.sym.finish()
    }

    /// Encodes the tile and returns its bytes, the reconstruction a decoder rebuilds from them
    /// (`coded_w x coded_h`), the `(size, tx_type)` trace, and the number of transform-type
    /// candidates the searches evaluated - everything a test needs to compare one search against
    /// another on rate, distortion, coverage and cost at once.
    #[cfg(test)]
    pub(crate) fn encode_with_report(mut self) -> SearchReport {
        self.encode_superblocks();
        SearchReport {
            reconstruction: std::mem::take(&mut self.recon),
            coded_width: self.coded_w,
            trace: std::mem::take(&mut self.emitted),
            candidates_evaluated: self.candidates_evaluated,
            size_choices: std::mem::take(&mut self.size_choices),
            cost_ties: self.cost_ties,
            abandoned_trials: self.abandoned_trials,
            consistent_trial_blocks: self.consistent_trial_blocks,
            probe_keys: std::mem::take(&mut self.probe_keys),
            emitted_blocks: std::mem::take(&mut self.emitted_blocks),
            emitted_coding_blocks: self.emitted_coding_blocks,
            probing_size_searches: self.probing_size_searches,
            size_search_probes: std::mem::take(&mut self.size_search_probes),
            zero_skipped_emitted: self.zero_skipped_emitted,
            reusable_emitted: self.reusable_emitted,
            tile: self.sym.finish(),
        }
    }

    /// [`Self::encode`] plus the `(size, tx_type)` of every transform block it wrote.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn encode_with_trace(mut self) -> (Vec<u8>, Vec<(usize, Av1TxType)>) {
        self.encode_superblocks();
        let emitted = std::mem::take(&mut self.emitted);
        (self.sym.finish(), emitted)
    }

    fn encode_superblocks(&mut self) {
        let mut r = 0;
        while r < self.mi_rows {
            self.left_level.fill(0);
            self.left_dc.fill(0);
            self.left_tx_height.fill(0);
            let mut c = 0;
            while c < self.mi_cols {
                self.encode_partition(r, c, 64, true);
                c += SB4;
            }
            r += SB4;
        }
    }

    /// Padded (edge-replicated) source sample of `plane` at coded-grid position `(x, y)`.
    fn sample(&self, x: usize, y: usize) -> i32 {
        let xx = x.min(self.width - 1);
        let yy = y.min(self.height - 1);
        i32::from(self.plane[yy * self.width + xx])
    }

    /// Whether the whole `bw x bw` block at MI position `(r, c)` lies inside the coded frame.
    /// The decoder rejects a transform block that hangs off the coded frame, so a block that
    /// does not fit is always split further.
    fn fits(&self, r: usize, c: usize, bw: usize) -> bool {
        c * 4 + bw <= self.coded_w && r * 4 + bw <= self.coded_h
    }

    /// Codes (or, when `emit` is false, only measures and reconstructs) the partition subtree
    /// rooted at MI position `(r, c)` for a `bw x bw` block, returning its rate-distortion cost.
    /// Lossless frames never split by cost and their cost is unused.
    fn encode_partition(&mut self, r: usize, c: usize, bw: usize, emit: bool) -> i64 {
        if r >= self.mi_rows || c >= self.mi_cols {
            return 0;
        }
        let num4x4 = bw / 4;
        let half = num4x4 >> 1;
        let has_rows = r + half < self.mi_rows;
        let has_cols = c + half < self.mi_cols;
        let bsl = num4x4.trailing_zeros() as usize; // Mi_Width_Log2

        let split = if bw < 8 {
            false // PARTITION_NONE forced, no symbol
        } else if has_rows && has_cols {
            let chosen = self.decide_split(r, c, bw);
            if emit {
                let ctx = self.partition_ctx(r, c, bsl);
                // PARTITION_NONE is 0 and PARTITION_SPLIT is 3 in every partition alphabet.
                self.sym
                    .encode_symbol(usize::from(chosen) * 3, partition_cdf(bsl, ctx));
            }
            chosen
        } else if has_cols {
            if emit {
                let ctx = self.partition_ctx(r, c, bsl);
                let cdf2 = split_or_horz_cdf(partition_cdf(bsl, ctx));
                self.sym.encode_symbol(1, &cdf2); // split
            }
            true
        } else if has_rows {
            if emit {
                let ctx = self.partition_ctx(r, c, bsl);
                let cdf2 = split_or_vert_cdf(partition_cdf(bsl, ctx));
                self.sym.encode_symbol(1, &cdf2); // split
            }
            true
        } else {
            true // forced PARTITION_SPLIT, no symbol
        };

        if !split {
            return self.encode_block(r, c, bw, emit);
        }
        let h = bw / 2;
        self.encode_partition(r, c, h, emit)
            + self.encode_partition(r, c + half, h, emit)
            + self.encode_partition(r + half, c, h, emit)
            + self.encode_partition(r + half, c + half, h, emit)
    }

    /// Whether a `bw x bw` block that *could* be coded whole is split anyway: always when it
    /// would not fit inside the coded frame, and otherwise when four half-width subtrees cost
    /// less than one block does.
    fn decide_split(&mut self, r: usize, c: usize, bw: usize) -> bool {
        if self.qindex == 0 {
            return false;
        }
        if !self.fits(r, c, bw) {
            return true;
        }
        if bw <= MIN_PARTITION_WIDTH {
            return false;
        }
        let slot = self.memo_slot(r, c, bw);
        if self.split_memo[slot] != MEMO_UNSET {
            return self.split_memo[slot] != 0;
        }
        let snapshot = self.snapshot(r, c, bw);
        let whole = self.encode_block(r, c, bw, false);
        self.restore(snapshot);

        // Four sub-blocks each pay their own partition, skip, mode, and tx_size symbols; charging
        // a flat header cost keeps the search from splitting for a negligible distortion win.
        const SPLIT_HEADER_BITS: i64 = 24;
        // A split subtree's cost is a sum of squared errors and positive bit counts, so it is
        // never negative: once the whole block costs no more than the header the split would
        // pay, the comparison below cannot come out in the split's favour and the four subtree
        // searches are pure waste. This is the search's own arithmetic, not an approximation.
        if self.shortcuts() && whole <= self.lambda * SPLIT_HEADER_BITS {
            self.split_memo[slot] = 0;
            return false;
        }

        let half = (bw / 4) >> 1;
        let h = bw / 2;
        let snapshot = self.snapshot(r, c, bw);
        let split = self.encode_partition(r, c, h, false)
            + self.encode_partition(r, c + half, h, false)
            + self.encode_partition(r + half, c, h, false)
            + self.encode_partition(r + half, c + half, h, false);
        self.restore(snapshot);

        // Strictly less, so a split that only ties the whole block does not happen: the tie goes
        // to `PARTITION_NONE`, the cheaper of the two to signal (one partition symbol against
        // five, plus four sub-blocks' own headers). Unlike the type and size searches this
        // comparison is between two named alternatives rather than a scan, so the preference is
        // the whole tie-break; there is no candidate order it could otherwise fall back on.
        let chosen = split + self.lambda * SPLIT_HEADER_BITS < whole;
        self.split_memo[slot] = u8::from(chosen);
        chosen
    }

    /// Slot in [`Self::split_memo`] and [`Self::tx_size_memo`] for the `bw x bw` block at
    /// `(r, c)`.
    fn memo_slot(&self, r: usize, c: usize, bw: usize) -> usize {
        let bsl = (bw / 4).trailing_zeros() as usize;
        (bsl.min(MEMO_LEVELS - 1) * self.mi_rows + r) * self.mi_cols + c
    }

    /// The `tx_depth` context (§9.3): how many of the above and left neighbours already carry a
    /// transform at least as large as this block's `Max_Tx_Size_Rect`. Every block this encoder
    /// codes is intra and unskipped, so the neighbour value is always the neighbour's own
    /// transform extent rather than its block size.
    fn tx_depth_ctx(&self, r: usize, c: usize, max_tx: usize) -> usize {
        let above = r > 0 && usize::from(self.above_tx_width[c]) >= max_tx;
        let left = c > 0 && usize::from(self.left_tx_height[r]) >= max_tx;
        usize::from(above) + usize::from(left)
    }

    /// `set_txfm_ctxs`: a coding block leaves its transform extent on every MI column and row it
    /// covers, for the next block's [`Self::tx_depth_ctx`].
    fn set_tx_ctx(&mut self, r: usize, c: usize, units: usize, tx_width: usize) {
        let extent = u8::try_from(tx_width).unwrap_or(u8::MAX);
        for column in c..(c + units).min(self.mi_cols) {
            self.above_tx_width[column] = extent;
        }
        for row in r..(r + units).min(self.mi_rows) {
            self.left_tx_height[row] = extent;
        }
    }

    fn partition_ctx(&self, r: usize, c: usize, bsl: usize) -> usize {
        let above = r > 0 && usize::from(self.mi_bsl[(r - 1) * self.mi_cols + c]) < bsl;
        let left = c > 0 && usize::from(self.mi_bsl[r * self.mi_cols + (c - 1)]) < bsl;
        usize::from(left) * 2 + usize::from(above)
    }

    /// Codes one `PARTITION_NONE` coding block, returning its rate-distortion cost.
    fn encode_block(&mut self, r: usize, c: usize, bw: usize, emit: bool) -> i64 {
        let n4 = bw / 4;
        let bsl = n4.trailing_zeros() as u8;

        if emit {
            // intra_frame_mode_info: skip=0 (ctx 0), y_mode=DC_PRED (ctx [0][0]).
            // Monochrome streams have no UV mode syntax.
            self.sym.encode_symbol(0, &cdf::SKIP[0]);
            self.sym.encode_symbol(0, &cdf::INTRA_FRAME_Y_MODE_DC_DC);

            for y in 0..n4 {
                for x in 0..n4 {
                    let (rr, cc) = (r + y, c + x);
                    if rr < self.mi_rows && cc < self.mi_cols {
                        self.mi_bsl[rr * self.mi_cols + cc] = bsl;
                    }
                }
            }
        }

        if self.qindex == 0 {
            // residual(): raster of 4×4 luma transform blocks (Lossless ⇒ TX_4X4).
            for ty in 0..n4 {
                for tx in 0..n4 {
                    let sx = c * 4 + tx * 4;
                    let sy = r * 4 + ty * 4;
                    if sx >= self.coded_w || sy >= self.coded_h {
                        continue; // transform block entirely outside the frame
                    }
                    self.lossless_transform_block(sx, sy, bw);
                }
            }
            return 0;
        }

        let tx_width = self.choose_tx_size(r, c, bw);
        if emit {
            // read_tx_size (§5.11.16) under TX_MODE_SELECT: the depth halving
            // `Max_Tx_Size_Rect[MiSize]` down to the chosen size.
            let largest = bw.min(MAX_TX_WIDTH);
            if largest > 4 {
                let ctx = self.tx_depth_ctx(r, c, largest);
                let (depth_cdf, _) = cdf::tx_depth_cdf(bw, ctx);
                let depth = (largest / tx_width).trailing_zeros() as usize;
                self.sym.encode_symbol(depth, depth_cdf);
            }
            self.set_tx_ctx(r, c, n4, tx_width);
        }
        self.code_block_transforms(r, c, bw, tx_width, emit)
    }

    /// Picks the transform size for a `bw x bw` coding block by trial-coding every size
    /// `read_tx_size` can signal for it and keeping the cheapest.
    ///
    /// The trials themselves rank on the set's DCT alone, which on its own is not a fair
    /// comparison *between sizes*: coding the block as sixteen 4x4 transforms gives the emitting
    /// pass's type search sixteen chances to beat DCT against one for a single 16x16 transform,
    /// and a DCT-only ranking throws that advantage away. A sampled subset of the trials at each
    /// size - every [`TYPE_GAIN_SAMPLE_INTERVAL`]-th, counting the first - therefore probes
    /// [`TYPE_GAIN_PROBES`] of its blocks with the whole type set, and the gain that measures out
    /// is extrapolated over every trial of that size in [`Self::corrected_trial_cost`] - so the
    /// correction grows with the trial's block count, as the type search's real advantage does,
    /// at a cost that does not, and that is paid on a sample of the trials rather than all of
    /// them.
    fn choose_tx_size(&mut self, r: usize, c: usize, bw: usize) -> usize {
        let slot = self.memo_slot(r, c, bw);
        if self.tx_size_memo[slot] != MEMO_UNSET {
            return 4 << self.tx_size_memo[slot];
        }
        let largest = bw.min(MAX_TX_WIDTH);
        // Only the depth cap is needed here; it does not vary with the neighbour context.
        let (_, max_depth) = cdf::tx_depth_cdf(bw, 0);
        #[cfg(test)]
        {
            self.gain_column = c / SB4;
        }
        // The sizes `read_tx_size` can signal for this block, in increasing depth order, which is
        // increasing symbol order: depth `d` is the symbol the decoder reads.
        let mut widths = Vec::new();
        for depth in 0..=max_depth {
            let tx_width = (largest >> depth).max(4);
            if tx_width <= MAX_FORWARD_TX {
                widths.push(tx_width);
            }
            if tx_width == 4 {
                break;
            }
        }
        #[cfg(test)]
        if self.reversed_candidates {
            widths.reverse();
        }
        // §7.12.3's `Dq_Denom` makes a 32x32 transform's coefficients twice the magnitude of a
        // smaller one's for the same reconstruction, so its DCT-vs-set gain has a different
        // scale from theirs and a sampled measurement of it does not carry. A size search that
        // can reach 32x32 therefore always probes, and probes several of the trial's blocks
        // rather than one.
        //
        // The stride otherwise samples coding blocks, not transform sizes, and which sizes a
        // search can even reach is a property of the coding block's width - only a 16x16 or
        // smaller block trials `TX_4X4` at all. Whether the smallest transform is *selected*
        // somewhere in a frame therefore depends on whether the particular block it wins at was
        // itself sampled, because a trial that probed is corrected by its own measurement at full
        // strength while every other trial's is shrunk to [`TYPE_GAIN_TRUST`] sixteenths. That is
        // the phase dependence #323 recorded, and it is left standing deliberately: the only
        // guarantee that removes it - `with_forced_smallest_size_probes` - is measured there to
        // cost 28-70% more transform-type candidates and up to +12.4% rate-distortion, for an
        // outcome the sampled estimator is not worse than.
        let large = largest.min(MAX_FORWARD_TX) >= 32;
        // A context-consistent trial has nothing to probe: it already codes every block with the
        // type the emitting pass would pick, so the gain a probe measures is inside its own cost.
        let consistent =
            self.shortcuts() && (self.context_consistent_trials() || self.consistent_size_trials());
        let probing = self.shortcuts()
            && !consistent
            && self.type_gain_correction()
            && (self.sample_type_gain()
                || large
                || (self.force_smallest_size_probes() && widths.last() == Some(&4)));
        #[cfg(test)]
        {
            let smallest = widths.iter().copied().min().unwrap_or(0);
            self.size_search_probes.push((smallest, probing));
            if probing {
                self.probing_size_searches += 1;
            }
        }
        let mut best = (0usize, i64::MAX);
        for &tx_width in &widths {
            let snapshot = self.snapshot(r, c, bw);
            self.probe_budget = match (probing, large) {
                (true, true) => LARGE_TYPE_GAIN_PROBES,
                (true, false) => self.type_gain_probes(),
                (false, _) => 0,
            };
            self.probe_dct_cost = 0;
            self.probe_best_cost = 0;
            self.probe_blocks = 0;
            self.trial_searched_cost = 0;
            self.trial_searched_blocks = 0;
            // A trial may be abandoned part-way whenever `trial_rank_bound` can turn the sum
            // `code_block_transforms` accumulates into a lower bound on the cost the trial will
            // finally be *ranked* on. A context-consistent trial is ranked on that sum directly,
            // so the sum is its own bound. A shipped trial is ranked on
            // `corrected_trial_cost(cost, ...)`, and #398 is where that case is settled: a trial
            // that does not probe reads its credit from an accumulator fixed before it starts,
            // and under `GainModel::Linear` the credit is `measured * trial_searched_cost / dct`
            // - a fixed fraction of a quantity that grows only as fast as the sum itself. When
            // that fraction is at most one the corrected cost is monotone in the sum too, and
            // `shipped_trial_credit` hands back the `(measured, dct)` that makes it so.
            self.trial_credit = (0, 0);
            let bounded = if consistent {
                true
            } else {
                match self.shipped_trial_credit(tx_width, probing) {
                    Some(credit) => {
                        self.trial_credit = credit;
                        true
                    }
                    None => false,
                }
            };
            self.trial_ceiling = if bounded {
                // The incumbent's cost is reachable only by a size that also wins the tie, which
                // is the larger transform; a smaller one has to come in strictly under it. The
                // tie-break is over the ranking cost, so it reads the same either side of the
                // correction: `trial_rank_bound` bounds that same ranking cost from below.
                if tx_width > best.0 {
                    best.1
                } else {
                    best.1 - 1
                }
            } else {
                i64::MAX
            };
            self.trial_abandoned = false;
            self.in_size_trial = true;
            let cost = self.code_block_transforms(r, c, bw, tx_width, false);
            self.in_size_trial = false;
            let abandoned = self.trial_abandoned;
            self.trial_ceiling = i64::MAX;
            self.trial_credit = (0, 0);
            self.trial_abandoned = false;
            self.restore(snapshot);
            if abandoned {
                #[cfg(test)]
                {
                    self.abandoned_trials += 1;
                }
                continue;
            }
            // Correcting a context-consistent trial would double-count: the type search's gain is
            // already in the cost it just measured, on every block rather than on a sample.
            let cost = if consistent || !self.type_gain_correction() {
                cost
            } else {
                self.corrected_trial_cost(cost, tx_width)
            };
            // Sizes are ranked by the same total order the type search uses: cost first, and an
            // exact tie broken towards the cheaper size to signal, which is the largest one -
            // its depth, and so its `read_tx_size` symbol, is the smallest. Deciding a tie by the
            // loop's order instead would make the answer a property of how this search happens to
            // enumerate, not of the block;
            // `the_search_is_independent_of_the_order_candidates_are_evaluated_in` pins that down.
            #[cfg(test)]
            if cost == best.1 && best.0 != 0 {
                self.cost_ties += 1;
            }
            if cost < best.1 || (cost == best.1 && tx_width > best.0) {
                best = (tx_width, cost);
            }
        }
        // Nothing outside a size trial may probe: every other speculative pass stays on the
        // set's DCT alone, which is where the shortcut's speedup comes from.
        self.probe_budget = 0;
        debug_assert_ne!(best.0, 0, "every coding block has one legal transform size");
        #[cfg(test)]
        self.size_choices.push((r, c, bw, best.0));
        self.tx_size_memo[slot] = (best.0 / 4).trailing_zeros() as u8;
        best.0
    }

    /// Whether this coding block's size search probes, which every
    /// [`TYPE_GAIN_SAMPLE_INTERVAL`]-th one does, counting the frame's first.
    fn sample_type_gain(&mut self) -> bool {
        let sample = self.size_searches % self.type_gain_interval() == 0;
        self.size_searches += 1;
        sample
    }

    /// Joins a probe to one accumulator, which is the frame's running sum at that size.
    ///
    /// #272 aged each accumulator by `(n-1)/n` first, so a probe's weight decayed over the
    /// following `TYPE_GAIN_MEMORY` probes and a block read back the ratio its own neighbourhood
    /// measured rather than the frame's. [`TYPE_GAIN_TRUST`] then shrank a *remembered* ratio to
    /// an eighth, and under that shrinkage the window stopped being measurable: issue #308 swept
    /// every window from `1` to the un-decayed sum on `scene_edge`, on three frames built to
    /// separate them - statistics alternating every 16 rows, the same two statistics on a 32x32
    /// checkerboard so boundaries run in both axes, and a spatial frequency rising continuously
    /// across the frame - at 128x96, 192x160 and 320x256, and every window from `2` upwards
    /// measured the identical penalty on all fifteen frame-and-size pairs. Only `1`, which
    /// remembers a single probe, moved anything at all, and only on one of them. So the decay is
    /// gone and this is a sum again; what it was carrying is carried by the shrinkage.
    /// Ages an accumulator by `(n-1)/n` before a new probe joins it.
    ///
    /// A probe's influence then decays away over the following `n` probes, so the ratio a block
    /// reads back is the one its own neighbourhood measured rather than the whole frame's, and it
    /// follows the content across a region boundary within a few coding blocks instead of never.
    /// One multiply and one divide per accumulator, on the sampled trials only.
    ///
    /// `usize::MAX` is the sentinel for no decay at all, which is the frame-wide accumulation
    /// this replaced; `measure_type_gain_memory_windows` sweeps it as the far end of the window.
    fn decay(gain: &mut TypeGain, memory: usize) {
        if memory == usize::MAX {
            return;
        }
        let (num, den) = (memory as i64 - 1, memory as i64);
        gain.dct_cost = gain.dct_cost * num / den;
        gain.best_cost = gain.best_cost * num / den;
        gain.probes = gain.probes * num / den;
        #[cfg(test)]
        {
            gain.ratio = gain.ratio * num / den;
        }
    }

    fn accumulate(gain: &mut TypeGain, dct: i64, best_of_set: i64) {
        #[cfg(test)]
        {
            let probe_ratio = if dct > 0 {
                (dct - best_of_set) * GAIN_RATIO_ONE / dct
            } else {
                0
            };
            gain.ratio = (gain.ratio * gain.probes + probe_ratio) / (gain.probes + 1);
        }
        gain.probes += 1;
        gain.dct_cost += dct;
        gain.best_cost += best_of_set;
    }

    /// What a trial that did not probe reads back, under the arm [`Self::type_gain_locality`]
    /// and [`Self::type_gain_ratio`] select. Outside tests there is only the running
    /// accumulator's ratio of sums, which is what the sweeps measured every arm against.
    #[cfg(test)]
    fn remembered_gain(&self, running: TypeGain, slot: usize) -> (i64, i64, i64) {
        let column = self.column_gain[self.gain_column * TYPE_GAIN_SIZES + slot];
        let (dct, best, probes) = match self.type_gain_locality() {
            GainLocality::Running => (running.dct_cost, running.best_cost, running.probes),
            GainLocality::Column => (column.dct_cost, column.best_cost, column.probes),
            GainLocality::Blended => (
                running.dct_cost + column.dct_cost,
                running.best_cost + column.best_cost,
                running.probes + column.probes,
            ),
        };
        match self.type_gain_ratio() {
            GainRatio::Weighted => (dct, best, probes),
            // The mean is already a ratio, so it is handed back over `GAIN_RATIO_ONE` and the
            // subtraction in `corrected_trial_cost` is the same either way.
            GainRatio::Mean => {
                // The mean is a ratio of costs and carries no block count with it, so it can
                // only be read by the model that consumes a ratio of costs.
                debug_assert_eq!(
                    self.type_gain_model(),
                    GainModel::Linear,
                    "a per-block model cannot read a gain expressed as a ratio of costs"
                );
                let ratio = match self.type_gain_locality() {
                    GainLocality::Running => running.ratio,
                    GainLocality::Column => column.ratio,
                    GainLocality::Blended => (running.ratio + column.ratio) / 2,
                };
                if dct <= 0 {
                    (0, 0, 0)
                } else {
                    (GAIN_RATIO_ONE, GAIN_RATIO_ONE - ratio, probes)
                }
            }
        }
    }

    /// The credit the shipped ranking will subtract from a size trial of `tx_width`, as the
    /// `(measured, dct)` pair [`GainModel::Linear`] scales by [`Self::trial_searched_cost`], or
    /// `None` when the trial's ranking cost is not monotone in the sum it accumulates.
    ///
    /// This is #398's answer to the reason #388 recorded for gating the trial cost bound to the
    /// context-consistent arm. The shipped trial ranks on [`Self::corrected_trial_cost`], which
    /// subtracts a credit, so a partial sum is a proof of nothing *in general*. It is a proof
    /// under the shipped estimator, on the majority of trials, and this is the exact statement
    /// of when:
    ///
    /// - **The trial must not probe.** A probing trial measures its own `(dct_p, best_p)` as it
    ///   goes, so the credit is not a function of the trial's progress that anything is known
    ///   about. Only every [`TYPE_GAIN_SAMPLE_INTERVAL`]-th size search probes, so this is the
    ///   majority of trials rather than a corner of them. A non-probing trial leaves
    ///   `probe_dct_cost` at zero and touches no accumulator, so the `(dct, best, probes)` read
    ///   here is what [`Self::corrected_trial_cost`] will read back at the end.
    /// - **The model must be [`GainModel::Linear`].** It alone scales the measurement by the
    ///   trial's searched *cost*, which grows exactly as fast as the sum. Every other model
    ///   scales by the searched *block count*, and a block that costs nothing still adds a whole
    ///   block's credit, so the corrected cost can fall as the trial runs. Only `Linear` ships.
    /// - **The credit fraction must be at most one.** With `measured = (dct - best) * trust /
    ///   TYPE_GAIN_TRUST_ONE` fixed, the credit is `measured * s / dct` for a searched cost `s`.
    ///   A block adds `c >= 0` to the sum and at most `c` to `s`, so the corrected cost moves by
    ///   at least `c - ceil(measured * c / dct)`, which is non-negative exactly when `measured <=
    ///   dct`. At the fraction's ceiling of one - reachable only when the whole of a probe's DCT
    ///   cost was beaten away and `trust` is not shrinking it - that step is zero rather than
    ///   negative: the corrected cost stops rising but never falls, so the bound stays exact and
    ///   only stops abandoning anything. The shrinkage is asserted into `0..=TYPE_GAIN_TRUST_ONE`
    ///   and `best` is `cheapest.min(dct)`, so the fraction cannot exceed one under any arm this
    ///   crate can construct; it is still checked, because the bound is exact only if it holds.
    ///
    /// A trial that will not be corrected at all - the [`Self::type_gain_correction`] arm off, or
    /// an accumulator with nothing in it - ranks on the raw sum, which is the same statement with
    /// a zero credit.
    fn shipped_trial_credit(&self, tx_width: usize, probing: bool) -> Option<(i64, i64)> {
        if !self.shortcuts() || probing || !self.bounded_size_trials() {
            return None;
        }
        if !self.type_gain_correction() {
            return Some((0, 0));
        }
        if self.type_gain_model() != GainModel::Linear {
            return None;
        }
        let slot = type_gain_slot(tx_width);
        let running = self.type_gain[slot];
        #[cfg(not(test))]
        let (dct, best, probes) = (running.dct_cost, running.best_cost, running.probes);
        #[cfg(test)]
        let (dct, best, probes) = self.remembered_gain(running, slot);
        // The same two guards `corrected_trial_cost` returns the uncorrected cost under, and
        // fixed for the trial for the same reason: nothing a non-probing trial does moves them.
        if dct <= 0 || probes <= 0 {
            return Some((0, 0));
        }
        let measured = (dct - best) * self.type_gain_trust() / TYPE_GAIN_TRUST_ONE;
        (0..=dct).contains(&measured).then_some((measured, dct))
    }

    /// The smallest cost the running trial can still be *ranked* at, given the sum it has
    /// accumulated so far.
    ///
    /// [`Self::shipped_trial_credit`] establishes that the ranking cost only rises from here, so
    /// this value is a lower bound on it: the trial's own final ranking is this same expression
    /// evaluated on the sums it finishes with.
    ///
    /// The credit is taken in `i128` where [`Self::corrected_trial_cost`] takes it in `i64` with
    /// a `saturating_mul`. That does not make the two disagree in the direction that matters: a
    /// saturated product is smaller than the real one, so the shipped credit is never larger than
    /// this one and the shipped ranking cost is never smaller than this bound. The bound stays
    /// valid - a little loose - in the range where the shipped correction saturates.
    fn trial_rank_bound(&self, cost: i64) -> i64 {
        let (measured, dct) = self.trial_credit;
        if measured <= 0 || dct <= 0 {
            return cost;
        }
        // `measured <= dct` and `trial_searched_cost <= cost`, so the credit is at most `cost`
        // and the narrowing cannot overflow.
        let credit = i128::from(measured) * i128::from(self.trial_searched_cost) / i128::from(dct);
        cost - credit as i64
    }

    /// Discounts a size trial's DCT-only cost by what the type search has been measured to be
    /// worth at this transform size.
    ///
    /// A probe measures `dct_p - best_p` over `p` blocks and the trial has `b` searched blocks
    /// costing `dct_b`; the model is what turns the first into a credit against the second.
    /// Blocks the zero-block shortcut decided are in neither: no transform type improves a block
    /// that codes no coefficients. A trial that probed uses its own `p`; one that skipped the
    /// probe uses every block sampled at that size so far this frame instead, so it is still
    /// corrected - by the measurement a probe of its own would have been taking.
    ///
    /// [`GainModel::Linear`] scales the measurement by `dct_b / dct_p`, [`GainModel::PerBlock`]
    /// by `b / p`, and [`GainModel::Saturating`] by `e(b) / p`. Only the last of the three is
    /// still a model when `p` reaches `b`: the other two collapse to the measurement itself, so
    /// how much a trial is credited depends on whether the probe count happened to cover it, and
    /// raising the probe count moves the *sizes* against each other rather than only sharpening
    /// the estimate. That is the dependence [`TYPE_GAIN_SATURATION`] removes.
    fn corrected_trial_cost(&self, cost: i64, tx_width: usize) -> i64 {
        let measured_here = self.probe_dct_cost > 0 && !self.unified_type_gain();
        let (dct, best, probes) = if measured_here {
            (self.probe_dct_cost, self.probe_best_cost, self.probe_blocks)
        } else {
            let slot = type_gain_slot(tx_width);
            let running = self.type_gain[slot];
            #[cfg(not(test))]
            {
                (running.dct_cost, running.best_cost, running.probes)
            }
            #[cfg(test)]
            {
                self.remembered_gain(running, slot)
            }
        };
        // A trial every block of which the zero-block shortcut decided has nothing to credit,
        // and no probe or accumulator to credit it from.
        if dct <= 0 || probes <= 0 || self.trial_searched_blocks <= 0 {
            return cost;
        }
        let mut measured = dct - best;
        let trust = if measured_here {
            self.type_gain_probe_trust()
        } else {
            self.type_gain_trust()
        };
        measured = measured * trust / TYPE_GAIN_TRUST_ONE;
        let blocks = self.trial_searched_blocks;
        let credit = match self.type_gain_model() {
            GainModel::Linear => measured.saturating_mul(self.trial_searched_cost) / dct,
            GainModel::PerBlock => measured.saturating_mul(blocks) / probes,
            GainModel::Amplified => {
                let numerator = i128::from(measured)
                    * i128::from(blocks)
                    * i128::from(self.type_gain_amplification());
                let denominator = i128::from(probes) * i128::from(TYPE_GAIN_TRUST_ONE);
                (numerator / denominator) as i64
            }
            GainModel::Saturating => {
                // `e(b) = b * s / (b + s - 1)`, evaluated together with the `/ p` so the
                // saturation is not rounded away on a trial of one or two blocks.
                let saturation = self.type_gain_saturation();
                let numerator = i128::from(measured) * i128::from(blocks) * i128::from(saturation);
                let denominator = i128::from(probes) * i128::from(blocks + saturation - 1);
                (numerator / denominator) as i64
            }
        };
        cost - credit
    }

    /// Walks a coding block's transform blocks in the decoder's raster order, reconstructing
    /// each one and summing their rate-distortion costs.
    fn code_block_transforms(
        &mut self,
        r: usize,
        c: usize,
        bw: usize,
        tx_width: usize,
        emit: bool,
    ) -> i64 {
        let units = bw / 4;
        let step = tx_width / 4;
        #[cfg(test)]
        if emit {
            self.emitted_coding_blocks += 1;
        }
        let mut cost = 0;
        let mut ty = 0;
        while ty < units {
            let mut tx = 0;
            while tx < units {
                let x = c * 4 + tx * 4;
                let y = r * 4 + ty * 4;
                if x < self.coded_w && y < self.coded_h {
                    cost += self.transform_block(x, y, bw, tx_width, emit);
                }
                // A transform block's cost is `sse + lambda * bits`, both non-negative, so this
                // sum is monotone, and `trial_rank_bound` turns it into a lower bound on the
                // cost the trial will be ranked on. A trial already above its ceiling on that
                // bound can only end above it. Abandoning it here skips the type search on every
                // block it has not reached yet; the caller discards the partial sum rather than
                // ranking on it. The raw comparison comes first because the bound never exceeds
                // the sum, so a sum inside the ceiling needs no arithmetic at all.
                if cost > self.trial_ceiling && self.trial_rank_bound(cost) > self.trial_ceiling {
                    self.trial_abandoned = true;
                    return cost;
                }
                tx += step;
            }
            ty += step;
        }
        cost
    }

    /// Lossless 4x4 transform block: the WHT of the source residual against a DC prediction
    /// taken from the (padded) source, which is the reconstruction under lossless coding.
    fn lossless_transform_block(&mut self, sx: usize, sy: usize, block_width: usize) {
        let avg = self.lossless_dc_avg(sx, sy);
        let mut res = [0i32; 16];
        for i in 0..4 {
            for j in 0..4 {
                res[i * 4 + j] = self.sample(sx + j, sy + i) - avg;
            }
        }
        let quant = fwht4x4(&res);
        self.code_coefficients(
            sx >> 2,
            sy >> 2,
            block_width,
            4,
            &quant,
            &cdf::DEFAULT_SCAN_4X4,
            None,
            true,
        );
    }

    /// Non-lossless transform block: DC prediction from the reconstruction, then the cheapest
    /// `tx_type` of the reduced set the decoder reads back. Returns the block's cost and leaves
    /// the reconstruction and coefficient contexts updated whether or not symbols were emitted,
    /// because later blocks in the same trial depend on both.
    fn transform_block(
        &mut self,
        x: usize,
        y: usize,
        block_width: usize,
        size: usize,
        emit: bool,
    ) -> i64 {
        let prediction = self.dc_prediction(x, y, size);
        #[cfg(test)]
        if emit {
            self.emitted_blocks.insert((x, y), (size, prediction));
        }

        let mut residual = vec![0i32; size * size];
        for row in 0..size {
            for column in 0..size {
                residual[row * size + column] =
                    self.sample(x + column, y + row) - i32::from(prediction);
            }
        }

        // read_tx_type (§5.11.47/§5.11.48): the set the decoder derives for an intra block of
        // this size under the frame header's `reduced_tx_set = 1`. Every type the set names that
        // this crate has a kernel for is a candidate, and its symbol is its index in the set.
        let set = cdf::get_tx_set(size, false, true);
        let inverse = cdf::tx_type_inverse_set(set);
        let mut candidates: Vec<(usize, Av1TxType)> = inverse
            .iter()
            .enumerate()
            .filter_map(|(symbol, &(_, tx_type))| Some((symbol, tx_type?)))
            .collect();
        #[cfg(test)]
        if self.reversed_candidates {
            candidates.reverse();
        }
        let scan = cdf::up_right_diagonal_scan(size);

        // Coding nothing costs the residual's own energy plus the single `all_zero` bit, and any
        // block that codes even one coefficient costs at least `MIN_CODED_BLOCK_BITS`. When the
        // former is already cheaper, no transform type can win, whatever it does to the
        // coefficients — so the whole search is skipped and the all-zero block written directly.
        // This is an exact consequence of the cost function, not an approximation of it: it is
        // the flat regions of a picture, where most of the exhaustive search was being spent.
        let energy: i64 = residual
            .iter()
            .map(|&sample| i64::from(sample) * i64::from(sample))
            .sum();
        if self.shortcuts()
            && energy + self.lambda * ZERO_BLOCK_BITS < self.lambda * MIN_CODED_BLOCK_BITS
        {
            #[cfg(test)]
            if emit {
                self.zero_skipped_emitted += 1;
            }
            return self.write_zero_block(x, y, block_width, size, prediction, &scan, energy, emit);
        }

        // Searching every type on every trial multiplies the transform-size and partition
        // searches by the size of the type set, and those trials only rank *sizes* and
        // *partitions*: the type each block finally writes is picked by the full search below on
        // the emitting pass. Ranking the trials on the set's DCT alone is what keeps the search
        // linear in the number of candidates rather than a product of them.
        //
        // A size trial gets to probe its first [`TYPE_GAIN_PROBES`] searched blocks with the
        // whole set anyway, to measure what the type search is worth at that size. The probe only
        // measures: the block still keeps DCT's result, so the trial's reconstruction, contexts
        // and cost are exactly the ones a DCT-only trial produces, and the measurement is applied
        // to the trial as a whole in `corrected_trial_cost`.
        #[cfg(test)]
        if emit && candidates.len() > 1 {
            self.reusable_emitted += 1;
        }
        let trial = self.shortcuts() && !emit;
        let mut probing = false;
        let consistent_here = self.context_consistent_trials()
            || (self.consistent_size_trials() && self.in_size_trial);
        #[cfg(test)]
        if consistent_here && trial && candidates.len() > 1 {
            self.consistent_trial_blocks += 1;
        }
        if trial && candidates.len() > 1 && !consistent_here {
            if self.probe_budget > 0 {
                self.probe_budget -= 1;
                probing = true;
            } else {
                let representative = candidates
                    .iter()
                    .position(|&(_, tx_type)| tx_type == Av1TxType::DctDct)
                    .unwrap_or(0);
                candidates = vec![candidates[representative]];
            }
        }

        let mut best: Option<TxCandidate> = None;
        let mut cheapest = i64::MAX;
        for &(symbol, tx_type) in &candidates {
            #[cfg(test)]
            {
                self.candidates_evaluated += 1;
            }
            let coefficients = forward_transform(&residual, size, tx_type);
            let levels = self.quantize(&coefficients, size);
            let reconstructed =
                inverse_transform(&levels, size, tx_type, self.dc_quant, self.ac_quant);
            let mut distortion = 0i64;
            for (sample, &value) in residual.iter().zip(reconstructed.iter()) {
                let error = i64::from(*sample) - i64::from(value);
                distortion += error * error;
            }
            let cost = distortion + self.lambda * estimate_rate(&levels, &scan);
            cheapest = cheapest.min(cost);
            // A probe ranks the block on DCT like any other trial block; every other pass keeps
            // the cheapest type, which on the emitting pass is the type actually written.
            //
            // Ties are not hypothetical: on the 96x80 test pattern of `nonlossless_tests` the
            // smallest emitting-pass margins are 4 (a 4x4 block between `DCT_DCT` and `IDTX`) and
            // 0 (a 4x4 block whose `ADST_ADST` and `DCT_DCT` costs are exactly equal). A strict
            // `<` would settle that block by whichever type `tx_type_inverse_set` lists first,
            // making the winner a property of a table's order and of this loop's direction rather
            // than of the block. The comparison is therefore a total order over
            // `(cost, symbol)`: equal cost is broken towards the smaller `tx_type` symbol, the
            // one the decoder reads earliest in the derived set and the cheaper of the two to
            // signal under `tx_type_cdf`, whose probabilities are ordered by symbol. That leaves
            // no dependence on evaluation order at all, which
            // `the_search_is_independent_of_the_order_candidates_are_evaluated_in` asserts
            // directly and `equal_cost_ties_are_reached_by_a_normal_encode` shows is not vacuous.
            #[cfg(test)]
            if !probing && best.as_ref().is_some_and(|best| best.cost == cost) {
                self.cost_ties += 1;
            }
            let candidate = TxCandidate {
                symbol,
                tx_type,
                levels,
                reconstructed,
                cost,
            };
            if probing {
                // A probe measures and nothing more: the block keeps DCT's result so the trial's
                // reconstruction, contexts and cost are exactly a DCT-only trial's, and what the
                // whole set was worth is carried out of the loop by `cheapest` alone.
                if tx_type == Av1TxType::DctDct || best.is_none() {
                    best = Some(candidate);
                }
            } else if best
                .as_ref()
                .is_none_or(|best| (cost, symbol) < (best.cost, best.symbol))
            {
                best = Some(candidate);
            }
        }
        if probing {
            let dct = best.as_ref().map_or(0, |best| best.cost);
            let best_of_set = cheapest.min(dct);
            self.probe_dct_cost += dct;
            self.probe_best_cost += best_of_set;
            self.probe_blocks += 1;
            let slot = type_gain_slot(size);
            let memory = self.type_gain_memory();
            Self::decay(&mut self.type_gain[slot], memory);
            Self::accumulate(&mut self.type_gain[slot], dct, best_of_set);
            #[cfg(test)]
            {
                let column = self.gain_column * TYPE_GAIN_SIZES + slot;
                Self::decay(&mut self.column_gain[column], memory);
                Self::accumulate(&mut self.column_gain[column], dct, best_of_set);
            }
            #[cfg(test)]
            self.probe_keys.push((x, y, size, prediction));
        }
        let winner = best.expect("every transform size has at least one candidate type");
        let cost = self.write_candidate(x, y, block_width, size, prediction, &winner, &scan, emit);
        if trial {
            self.trial_searched_cost += cost;
            self.trial_searched_blocks += 1;
        }
        cost
    }

    /// Reconstructs a transform block from the type the search picked and codes its coefficients,
    /// returning the candidate's cost.
    ///
    /// This is the whole of a transform block's output, shared by the searched winner and by the
    /// zero-block shortcut's forced all-zero block, so neither can drift from the other.
    #[allow(clippy::too_many_arguments)]
    fn write_candidate(
        &mut self,
        x: usize,
        y: usize,
        block_width: usize,
        size: usize,
        prediction: u8,
        candidate: &TxCandidate,
        scan: &[usize],
        emit: bool,
    ) -> i64 {
        for row in 0..size {
            let start = (y + row) * self.coded_w + x;
            let destination = &mut self.recon[start..start + size];
            destination.fill(prediction);
            add_residual_row(
                &candidate.reconstructed[row * size..(row + 1) * size],
                destination,
            );
        }

        // §5.11.39 `coeffs` reads `transform_type()` immediately after `all_zero` and before
        // `eob_pt`, so the symbol goes to the coefficient coder rather than after it. DC_PRED is
        // the only y_mode this encoder signals, so the CDF's intra direction is 0.
        let set = cdf::get_tx_set(size, false, true);
        let tx_type_symbol =
            cdf::tx_type_cdf(set, size, 0).map(|tx_cdf| (candidate.symbol, tx_cdf));
        self.code_coefficients(
            x >> 2,
            y >> 2,
            block_width,
            size,
            &candidate.levels,
            scan,
            tx_type_symbol,
            emit,
        );
        #[cfg(test)]
        if emit {
            self.emitted.push((size, candidate.tx_type));
        }
        candidate.cost
    }

    /// Writes the all-zero coefficient block whose cost the type search cannot beat, updating
    /// the reconstruction (which is the prediction, the residual being dropped entirely) and the
    /// coefficient contexts exactly as a searched block does.
    #[allow(clippy::too_many_arguments)]
    fn write_zero_block(
        &mut self,
        x: usize,
        y: usize,
        block_width: usize,
        size: usize,
        prediction: u8,
        scan: &[usize],
        energy: i64,
        emit: bool,
    ) -> i64 {
        for row in 0..size {
            let start = (y + row) * self.coded_w + x;
            self.recon[start..start + size].fill(prediction);
        }
        let levels = vec![0i32; size * size];
        // §5.11.39 `coeffs` reads `transform_type()` only when `all_zero` is 0, so a block whose
        // every coefficient is zero carries no `tx_type` symbol at all.
        self.code_coefficients(x >> 2, y >> 2, block_width, size, &levels, scan, None, emit);
        #[cfg(test)]
        if emit {
            // No `tx_type` symbol follows a block the decoder reads as fully skipped, so the
            // pair the trace records is the one the search would have defaulted to.
            self.emitted.push((size, Av1TxType::DctDct));
        }
        energy + self.lambda * ZERO_BLOCK_BITS
    }

    /// Forward quantization: the exact inverse of the `level * q / Dq_Denom[txSz]`
    /// dequantization [`inverse_transform`] applies, rounded to nearest.
    fn quantize(&self, coefficients: &[i32], size: usize) -> Vec<i32> {
        let denominator = i64::from(dq_denom(size));
        coefficients
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let step = i64::from(if index == 0 {
                    self.dc_quant
                } else {
                    self.ac_quant
                });
                let magnitude = i64::from(value).abs() * denominator;
                let level = (magnitude + step / 2) / step;
                let level = i32::try_from(level).unwrap_or(i32::MAX);
                if value < 0 { -level } else { level }
            })
            .collect()
    }

    /// Spec §7.11.2 DC intra prediction over the reconstruction, for any `size x size` transform
    /// block: the rounded average of the `size` samples immediately above and/or to the left, or
    /// 128 when neither neighbour is available. Mirrors the decoder's `dc_prediction_sized`.
    fn dc_prediction(&self, x: usize, y: usize, size: usize) -> u8 {
        let above = |offset: usize| u32::from(self.recon[(y - 1) * self.coded_w + x + offset]);
        let left = |offset: usize| u32::from(self.recon[(y + offset) * self.coded_w + x - 1]);
        match (y > 0, x > 0) {
            (true, true) => {
                let sum = (0..size).map(above).sum::<u32>() + (0..size).map(left).sum::<u32>();
                let count = 2 * size as u32;
                ((sum + count / 2) / count) as u8
            }
            (false, true) => {
                let sum = (0..size).map(left).sum::<u32>();
                let count = size as u32;
                ((sum + count / 2) / count) as u8
            }
            (true, false) => {
                let sum = (0..size).map(above).sum::<u32>();
                let count = size as u32;
                ((sum + count / 2) / count) as u8
            }
            (false, false) => 128,
        }
    }

    /// DC intra prediction for the lossless path, whose neighbours are the (padded) source
    /// samples because lossless reconstruction reproduces them exactly (§7.11.2.5).
    fn lossless_dc_avg(&self, sx: usize, sy: usize) -> i32 {
        let have_above = sy > 0;
        let have_left = sx > 0;
        match (have_above, have_left) {
            (true, true) => {
                let mut s = 0;
                for k in 0..4 {
                    s += self.sample(sx + k, sy - 1);
                    s += self.sample(sx - 1, sy + k);
                }
                (s + 4) >> 3
            }
            (false, true) => {
                let mut s = 0;
                for k in 0..4 {
                    s += self.sample(sx - 1, sy + k);
                }
                (s + 2) >> 2
            }
            (true, false) => {
                let mut s = 0;
                for k in 0..4 {
                    s += self.sample(sx + k, sy - 1);
                }
                (s + 2) >> 2
            }
            (false, false) => 128, // 1 << (BitDepth - 1)
        }
    }

    /// Codes one transform block's quantized coefficients (§5.11.39), returning whether the
    /// block carried any (`false` means `all_zero` was signalled). The coefficient contexts are
    /// updated either way; only the symbol writes are suppressed when `emit` is false, so a
    /// speculative trial and the replay that follows it derive identical contexts.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn code_coefficients(
        &mut self,
        x4: usize,
        y4: usize,
        block_width: usize,
        size: usize,
        quant: &[i32],
        scan: &[usize],
        tx_type_symbol: Option<(usize, &'static [u16])>,
        emit: bool,
    ) -> bool {
        let ptype = 0;
        // The specification selects every coefficient CDF below by a quantizer context derived
        // from `base_q_idx` and (except for `dc_sign`) by a transform-size context, exactly as
        // the decoders do. A lossless TX_4X4 frame lands on `qctx = 0`, `txSzCtx = 0`.
        let qctx = cdf::coeff_qctx(self.qindex);
        let tx_ctx = cdf::coeff_tx_size_ctx(size);
        let units = size / 4;
        let count = size * size;
        debug_assert_eq!(quant.len(), count);
        debug_assert_eq!(scan.len(), count);

        let mut eob = 0usize;
        for c in 0..count {
            if quant[scan[c]] != 0 {
                eob = c + 1;
            }
        }

        let txb_ctx = self.txb_skip_ctx(x4, y4, units, block_width, size);
        if emit {
            self.sym.encode_symbol(
                usize::from(eob == 0),
                cdf::txb_skip_cdf(qctx, tx_ctx, txb_ctx),
            );
        }
        if eob == 0 {
            self.set_ctx(x4, y4, units, 0, 0);
            return false;
        }

        // §5.11.39 reads `transform_type()` here: after `all_zero`, before `eob_pt`.
        if emit {
            if let Some((symbol, tx_cdf)) = tx_type_symbol {
                self.sym.encode_symbol(symbol, tx_cdf);
            }
        }

        // eob position (TX_CLASS_2D ⇒ eob_pt context 0).
        let eobpt = eobpt_from_eob(eob);
        if emit {
            self.sym
                .encode_symbol(eobpt - 1, cdf::eob_pt_cdf(qctx, size, ptype));
        }
        if eobpt >= 3 {
            let nbits = eobpt - 2;
            let base = (1usize << (eobpt - 2)) + 1;
            let extra = eob - base;
            if emit {
                self.sym.encode_symbol(
                    (extra >> (nbits - 1)) & 1,
                    cdf::eob_extra_cdf(qctx, tx_ctx, ptype, eobpt - 3),
                );
                // The remaining `nbits - 1` bits are equiprobable literals in MSB-first order,
                // which is one literal run rather than one call per bit.
                self.sym.encode_literal(extra as u32, nbits as u32 - 1);
            }
        }

        // Base levels + base range, scanned from the last coefficient back to DC. Every context
        // the loop needs is derived for the whole block up front (see `CoeffScratch`), and only
        // the emitting pass consults them, so a speculative trial skips the derivation outright.
        if emit {
            let mut scratch = std::mem::take(&mut self.coeff_ctx);
            scratch.derive(quant, size);
            for c in (0..eob).rev() {
                let pos = scan[c];
                let level = quant[pos].abs();
                if c == eob - 1 {
                    let ctx = coeff_base_eob_ctx(c, count);
                    self.sym.encode_symbol(
                        (level.min(3) - 1) as usize,
                        cdf::coeff_base_eob_cdf(qctx, tx_ctx, ptype, ctx),
                    );
                } else {
                    let ctx = scratch.base[pos] as usize;
                    self.sym.encode_symbol(
                        level.min(3) as usize,
                        cdf::coeff_base_cdf(qctx, tx_ctx, ptype, ctx),
                    );
                }
                if level > NUM_BASE_LEVELS {
                    let br_ctx = scratch.br[pos] as usize;
                    let mut rem = level - 3;
                    for _ in 0..4 {
                        let brv = rem.min(3);
                        self.sym.encode_symbol(
                            brv as usize,
                            cdf::coeff_br_cdf(qctx, tx_ctx, ptype, br_ctx),
                        );
                        rem -= brv;
                        if brv < 3 {
                            break;
                        }
                    }
                }
            }
            self.coeff_ctx = scratch;
        }

        // Signs (DC sign is CDF-coded; the rest are raw bits) and golomb tails.
        for (c, &pos) in scan.iter().enumerate().take(eob) {
            let level = quant[pos].abs();
            if level != 0 {
                let neg = quant[pos] < 0;
                if emit {
                    if c == 0 {
                        let ctx = self.dc_sign_ctx(x4, y4, units);
                        self.sym
                            .encode_symbol(usize::from(neg), cdf::dc_sign_cdf(qctx, ptype, ctx));
                    } else {
                        self.sym.encode_literal(u32::from(neg), 1);
                    }
                    if level > COEFF_BASE_PLUS_RANGE {
                        golomb(&mut self.sym, (level - COEFF_BASE_PLUS_RANGE) as u32);
                    }
                }
            }
        }

        // Every coefficient at or past the end-of-block is zero by the definition of `eob`, so
        // summing the whole block's magnitudes is the same `culLevel` the scanned levels gave.
        let cul = quant.iter().map(|value| value.abs()).sum::<i32>().min(63) as u8;
        let dc_cat = if quant[0] == 0 {
            0
        } else if quant[0] < 0 {
            1
        } else {
            2
        };
        self.set_ctx(x4, y4, units, cul, dc_cat);
        true
    }

    /// §5.11.39's trailing context update: a transform block leaves its cumulative level and DC
    /// sign category on every 4x4 column and row it covers.
    fn set_ctx(&mut self, x4: usize, y4: usize, units: usize, cul: u8, dc: u8) {
        for column in x4..(x4 + units).min(self.mi_cols) {
            self.above_level[column] = cul;
            self.above_dc[column] = dc;
        }
        for row in y4..(y4 + units).min(self.mi_rows) {
            self.left_level[row] = cul;
            self.left_dc[row] = dc;
        }
    }

    /// `getTXBSkipCtx` (§8.3.2), over the transform block's own width and height in 4x4 units.
    ///
    /// The specification's first case returns context 0 outright when the transform covers the
    /// whole coding block, without consulting a neighbour. Every coding block here is square, so
    /// that is exactly `tx_width == block_width`. It cannot fire on a lossless frame, whose
    /// transforms are all 4x4 while no coding block is narrower than
    /// [`MIN_PARTITION_WIDTH`], but it fires on most non-lossless blocks. The decoders derive
    /// the same context, so encoder and decoder stay in step.
    fn txb_skip_ctx(
        &self,
        x4: usize,
        y4: usize,
        units: usize,
        block_width: usize,
        tx_width: usize,
    ) -> usize {
        if tx_width >= block_width {
            return 0;
        }
        let top = self.above_level[x4..(x4 + units).min(self.mi_cols)]
            .iter()
            .copied()
            .max()
            .map_or(0, i32::from);
        let left = self.left_level[y4..(y4 + units).min(self.mi_rows)]
            .iter()
            .copied()
            .max()
            .map_or(0, i32::from);
        if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + usize::from(top.max(left) > 3)
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        }
    }

    /// `getDcSignCtx` (§8.3.2), summed over every 4x4 column and row the block covers.
    fn dc_sign_ctx(&self, x4: usize, y4: usize, units: usize) -> usize {
        let mut s = 0i32;
        let above = &self.above_dc[x4..(x4 + units).min(self.mi_cols)];
        let left = &self.left_dc[y4..(y4 + units).min(self.mi_rows)];
        for &cat in above.iter().chain(left.iter()) {
            if cat == 1 {
                s -= 1;
            } else if cat == 2 {
                s += 1;
            }
        }
        if s < 0 {
            1
        } else if s > 0 {
            2
        } else {
            0
        }
    }

    /// Captures the reconstruction and coefficient-context state a `bw x bw` block at `(r, c)`
    /// can touch, so a speculative trial can be rolled back.
    fn snapshot(&self, r: usize, c: usize, bw: usize) -> Snapshot {
        let (x0, y0) = (c * 4, r * 4);
        let recon_width = bw.min(self.coded_w - x0);
        let recon_height = bw.min(self.coded_h - y0);
        let mut recon = Vec::with_capacity(recon_width * recon_height);
        for row in 0..recon_height {
            let start = (y0 + row) * self.coded_w + x0;
            recon.extend_from_slice(&self.recon[start..start + recon_width]);
        }
        let units = bw / 4;
        let above_end = (c + units).min(self.mi_cols);
        let left_end = (r + units).min(self.mi_rows);
        Snapshot {
            x0,
            y0,
            recon_width,
            recon,
            above_start: c,
            above_level: self.above_level[c..above_end].to_vec(),
            above_dc: self.above_dc[c..above_end].to_vec(),
            left_start: r,
            left_level: self.left_level[r..left_end].to_vec(),
            left_dc: self.left_dc[r..left_end].to_vec(),
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        for (row, chunk) in snapshot
            .recon
            .chunks(snapshot.recon_width.max(1))
            .enumerate()
        {
            let start = (snapshot.y0 + row) * self.coded_w + snapshot.x0;
            self.recon[start..start + chunk.len()].copy_from_slice(chunk);
        }
        let above = snapshot.above_start;
        self.above_level[above..above + snapshot.above_level.len()]
            .copy_from_slice(&snapshot.above_level);
        self.above_dc[above..above + snapshot.above_dc.len()].copy_from_slice(&snapshot.above_dc);
        let left = snapshot.left_start;
        self.left_level[left..left + snapshot.left_level.len()]
            .copy_from_slice(&snapshot.left_level);
        self.left_dc[left..left + snapshot.left_dc.len()].copy_from_slice(&snapshot.left_dc);
    }
}

/// `Max_Tx_Size_Rect` cap the decoder's `read_tx_size` applies (`TX_64X64`).
const MAX_TX_WIDTH: usize = 64;

/// Bits §5.11.39's coefficient coder spends on one quantized magnitude, counted as the symbols
/// and raw bits it actually writes.
///
/// `coeff_base` (or `coeff_base_eob` at the last position) carries the level up to
/// [`NUM_BASE_LEVELS`]; above that, up to four `coeff_br` symbols carry three more each, so the
/// cost grows *linearly* with the level through the base range; only past
/// [`COEFF_BASE_PLUS_RANGE`] does it flatten to the exp-Golomb tail's `2 * bit_length(x) - 1`
/// raw bits. A nonzero level also carries a sign.
///
/// This shape is the whole point of the function. A closed `2 + 2 * bit_length(level)` charge is
/// logarithmic everywhere and prices a doubled level at two more bits at every magnitude, which
/// is only true in the golomb tail. §7.12.3's `Dq_Denom` makes every `TX_32X32` level twice a
/// smaller transform's for the same reconstruction, so a logarithmic charge under-prices a
/// 32x32 trial by the whole width of the base range - which is where most coefficients on
/// ordinary content sit - and the size search buys 32x32's lower distortion with bits the
/// estimate never charged it for.
fn coefficient_bits(level: u32) -> i64 {
    // The base symbol is written for every coefficient below the end-of-block, zero or not.
    let mut bits = SYMBOL_BITS;
    if level == 0 {
        return bits;
    }
    bits += 1; // sign
    if level as i32 > NUM_BASE_LEVELS {
        // `rem = level - 3` in steps of 3, at most four symbols, exactly as `code_coefficients`
        // writes them.
        let remainder = level - NUM_BASE_LEVELS as u32 - 1;
        bits += SYMBOL_BITS * i64::from((remainder / 3 + 1).min(4));
    }
    if level as i32 > COEFF_BASE_PLUS_RANGE {
        let tail = level - COEFF_BASE_PLUS_RANGE as u32;
        bits += 2 * i64::from(bit_length(tail)) - 1;
    }
    bits
}

/// Bits a coefficient block is estimated to cost: the `all_zero` symbol plus, when the block is
/// coded, the end-of-block position and [`coefficient_bits`] per coefficient up to it. Only the
/// relative ordering of candidates matters, so this stays a closed form over the quantized levels
/// rather than a trial arithmetic encode.
fn estimate_rate(levels: &[i32], scan: &[usize]) -> i64 {
    let mut eob = 0usize;
    for (index, &position) in scan.iter().enumerate() {
        if levels[position] != 0 {
            eob = index + 1;
        }
    }
    if eob == 0 {
        return ZERO_BLOCK_BITS;
    }
    // `eob_pt` is one symbol; `eob_extra` above it is one symbol and `eobPt - 3` raw bits.
    let eobpt = eobpt_from_eob(eob);
    let mut bits = ZERO_BLOCK_BITS + SYMBOL_BITS;
    if eobpt >= 3 {
        bits += SYMBOL_BITS + eobpt as i64 - 3;
    }
    for &position in scan.iter().take(eob) {
        bits += coefficient_bits(levels[position].unsigned_abs());
    }
    bits
}

fn bit_length(value: u32) -> u32 {
    32 - value.leading_zeros()
}

/// Selects the partition CDF by `bsl` (`Mi_Width_Log2`); M0 never uses 128×128 superblocks.
fn partition_cdf(bsl: usize, ctx: usize) -> &'static [u16] {
    match bsl {
        1 => &cdf::PARTITION_W8[ctx],
        2 => &cdf::PARTITION_W16[ctx],
        3 => &cdf::PARTITION_W32[ctx],
        _ => &cdf::PARTITION_W64[ctx],
    }
}

/// Derives the 2-symbol `split_or_horz` CDF from the partition CDF (§8.3.2): the vertical-ish
/// partition probabilities are folded into the "split" outcome.
fn split_or_horz_cdf(p: &[u16]) -> [u16; 2] {
    let psum = (p[2] - p[1])
        + (p[3] - p[2])
        + (p[4] - p[3])
        + (p[6] - p[5])
        + (p[7] - p[6])
        + (p[9] - p[8]);
    [32768 - psum, 32768]
}

/// Derives the 2-symbol `split_or_vert` CDF from the partition CDF (§8.3.2).
fn split_or_vert_cdf(p: &[u16]) -> [u16; 2] {
    let psum = (p[1] - p[0])
        + (p[3] - p[2])
        + (p[4] - p[3])
        + (p[5] - p[4])
        + (p[6] - p[5])
        + (p[8] - p[7]);
    [32768 - psum, 32768]
}

/// `eobPt` from `eob` (inverts `eob = (eobPt < 2) ? eobPt : (1 << (eobPt-2)) + 1`, §5.11.39).
fn eobpt_from_eob(eob: usize) -> usize {
    if eob <= 1 {
        eob
    } else {
        (32 - ((eob - 1) as u32).leading_zeros()) as usize + 1
    }
}

fn coeff_base_eob_ctx(c: usize, count: usize) -> usize {
    if c == 0 {
        0
    } else if c <= count / 8 {
        1
    } else if c <= count / 4 {
        2
    } else {
        3
    }
}

/// `getCoeffBaseCtx` (§8.3.2) for `TX_CLASS_2D`: the scalar reference
/// [`crate::av1_simd::coeff::block_contexts`] is a lane-by-lane transliteration of. `levels` holds
/// the block's quantized coefficients; only their magnitudes are read.
fn coeff_base_ctx(pos: usize, levels: &[i32], size: usize) -> usize {
    let (row, col) = (pos / size, pos % size);
    let mut mag = 0i32;
    for &(dr, dc) in &cdf::SIG_REF_DIFF_OFFSET_2D {
        let (rr, cc) = (row + dr, col + dc);
        if rr < size && cc < size {
            mag += levels[rr * size + cc].abs().min(3);
        }
    }
    let ctx = (((mag + 1) >> 1).min(4)) as usize;
    if row == 0 && col == 0 {
        return 0;
    }
    ctx + cdf::coeff_base_ctx_offset(row, col)
}

/// `getCoeffBrCtx` (§8.3.2) for `TX_CLASS_2D`, the other half of the scalar reference described
/// on [`coeff_base_ctx`].
fn coeff_br_ctx(pos: usize, levels: &[i32], size: usize) -> usize {
    let (row, col) = (pos / size, pos % size);
    let mut mag = 0i32;
    for &(dr, dc) in &cdf::MAG_REF_OFFSET_2D {
        let (rr, cc) = (row + dr, col + dc);
        if rr < size && cc < size {
            mag += levels[rr * size + cc].abs().min(15);
        }
    }
    let mag = (((mag + 1) >> 1).min(6)) as usize;
    if pos == 0 {
        mag
    } else if row < 2 && col < 2 {
        mag + 7
    } else {
        mag + 14
    }
}

/// Exp-Golomb tail used for coefficient magnitudes above the base-range cap (§5.11.39).
fn golomb(sym: &mut SymbolEncoder, x: u32) {
    let len = 32 - x.leading_zeros(); // bit length, x >= 1
    // The code is `len - 1` zeros followed by `x` itself in `len` bits, and `x`'s top bit is set,
    // so the whole thing is `x` written as one `2 * len - 1`-bit literal: the leading zeros are
    // just the field's own padding. That is one literal run instead of `2 * len - 1` single-bit
    // calls whenever the field fits in a `u32`.
    let bits = 2 * len - 1;
    if bits <= 32 {
        sym.encode_literal(x, bits);
    } else {
        sym.encode_literal(0, len - 1);
        sym.encode_literal(x, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// Small deterministic LCG, matching the style used elsewhere in the crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    /// Quantized coefficients spanning the ranges the contexts distinguish: zeros, the base
    /// levels, the base-range cap, and magnitudes past the golomb threshold, in both signs.
    fn coefficients(size: usize, seed: u64) -> Vec<i32> {
        let mut rng = Lcg(seed);
        (0..size * size)
            .map(|_| {
                let magnitude = match rng.next() % 5 {
                    0 => 0,
                    1 => (rng.next() % 3) as i32,
                    2 => (rng.next() % 16) as i32,
                    3 => (rng.next() % 64) as i32,
                    _ => (rng.next() % 4096) as i32,
                };
                if rng.next() % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect()
    }

    /// The widths worth covering: every one from a single coefficient up past two full AVX2
    /// vectors, so a partial trailing vector is exercised at both lane counts, plus the large
    /// transform sizes the non-lossless encoder actually codes.
    const WIDTHS: [usize; 20] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 32, 64,
    ];

    /// The whole point of the vector kernel: it must agree with `tile.rs`'s scalar §8.3.2
    /// reference lane for lane, at every width and on every instruction set the host has.
    #[test]
    fn the_context_pass_matches_the_scalar_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        for &size in &WIDTHS {
            for seed in 0..4u64 {
                let quant = coefficients(size, seed * 977 + size as u64);
                let mut expected_base = vec![0i32; size * size];
                let mut expected_br = vec![0i32; size * size];
                for pos in 0..size * size {
                    expected_base[pos] = coeff_base_ctx(pos, &quant, size) as i32;
                    expected_br[pos] = coeff_br_ctx(pos, &quant, size) as i32;
                }
                for isa in simd::available() {
                    simd::set_override(Some(isa));
                    assert_eq!(
                        coeff::active_isa(),
                        isa,
                        "the coefficient context site did not follow the override"
                    );
                    let mut scratch = CoeffScratch::default();
                    scratch.derive(&quant, size);
                    assert_eq!(
                        scratch.base,
                        expected_base,
                        "coeff_base at size {size}, seed {seed}, isa {}",
                        isa.name()
                    );
                    assert_eq!(
                        scratch.br,
                        expected_br,
                        "coeff_br at size {size}, seed {seed}, isa {}",
                        isa.name()
                    );
                }
            }
        }
        simd::set_override(None);
    }

    /// Deriving the contexts up front is only legal because the backwards scan never consults a
    /// neighbour it has not already coded, and never consults a non-zero one past the
    /// end-of-block. Replay the incremental derivation the coding loop used to do and check it
    /// against the one-pass answer, for every end-of-block a block can have.
    #[test]
    fn the_one_pass_derivation_matches_the_incremental_scan_order_one() {
        let _guard = simd::test_lock();
        simd::set_override(Some(SimdIsa::Scalar));
        for &size in &[4usize, 8, 16, 32] {
            let count = size * size;
            let scan = cdf::up_right_diagonal_scan(size);
            let dense = coefficients(size, 31 + size as u64);
            for eob in 1..=count {
                // Past the end-of-block every coefficient is zero by the definition of `eob`,
                // which is the property the one-pass derivation leans on.
                let mut quant = vec![0i32; count];
                for &pos in scan.iter().take(eob) {
                    quant[pos] = dense[pos];
                }
                let mut scratch = CoeffScratch::default();
                scratch.derive(&quant, size);

                let mut levels = vec![0i32; count];
                for c in (0..eob).rev() {
                    let pos = scan[c];
                    assert_eq!(
                        coeff_base_ctx(pos, &levels, size) as i32,
                        scratch.base[pos],
                        "coeff_base at size {size}, eob {eob}, scan index {c}"
                    );
                    assert_eq!(
                        coeff_br_ctx(pos, &levels, size) as i32,
                        scratch.br[pos],
                        "coeff_br at size {size}, eob {eob}, scan index {c}"
                    );
                    levels[pos] = quant[pos].abs();
                }
            }
        }
        simd::set_override(None);
    }

    /// The encoded bitstream is the contract: a vector context pass that changed a single symbol
    /// would produce a different tile, so every instruction set must emit the same bytes.
    #[test]
    fn the_encoded_tile_is_byte_identical_on_every_instruction_set() {
        let _guard = simd::test_lock();
        let (width, height) = (61usize, 37usize);
        let mut rng = Lcg(0x5eed);
        let plane: Vec<u8> = (0..width * height)
            .map(|_| (rng.next() & 0xff) as u8)
            .collect();
        for qindex in [0u8, 40, 160] {
            simd::set_override(Some(SimdIsa::Scalar));
            let reference = FrameEncoder::new(&plane, width, height, qindex).encode();
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let coded = FrameEncoder::new(&plane, width, height, qindex).encode();
                assert_eq!(
                    coded,
                    reference,
                    "tile bytes differ at qindex {qindex} on {}",
                    isa.name()
                );
            }
        }
        simd::set_override(None);
    }
}
