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
use crate::av1_intra::{Av1TxType, get_ac_quant, get_dc_quant, inverse_transform};
use crate::av1_intra_pred::add_residual_row;

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

/// Bits [`estimate_rate`] charges a block whose levels are all zero: the `all_zero` flag alone.
const ZERO_BLOCK_BITS: i64 = 1;
/// Bits [`estimate_rate`] charges the cheapest block it can charge that is *not* all zero: the
/// `all_zero` flag, a one-position end-of-block, and one magnitude-1 coefficient with its sign.
const MIN_CODED_BLOCK_BITS: i64 = 7;

/// Transform blocks per probing size trial that [`FrameEncoder::choose_tx_size`] searches with the
/// whole transform-type set instead of the set's DCT alone, to measure what the type search is
/// worth at that size before extrapolating it over the trial's remaining blocks.
const TYPE_GAIN_PROBES: usize = 1;

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
/// have two close sizes ranked the wrong way round. `2` is where that trade was measured out, on
/// the six-frame set in `measure_type_gain_sampling_intervals` at 192x160 - a hard scene edge, a
/// four-quadrant frame, full-range noise, a smooth surface, directional edges, and the encoder's
/// own `test_pattern` - against the same estimator probing every size search, which is the
/// unsampled search this interval approximates. Cost is the encoder's own `sse + lambda * bits`,
/// summed over the frame and compared at equal quantizer:
///
/// | interval | worst penalty vs unsampled | mean vs exhaustive | candidates | time |
/// |---------:|---------------------------:|-------------------:|-----------:|--------:|
/// | 1        | 0.0%                       | +0.25%             | 181,557    | 0.644 s |
/// | 2        | +44.1%                     | +1.05%             | 155,143    | 0.607 s |
/// | 4        | +64.6%                     | +1.70%             | 142,372    | 0.574 s |
/// | 8        | +78.7%                     | +1.97%             | 136,250    | 0.565 s |
/// | 16       | +85.8%                     | +2.29%             | 128,035    | 0.557 s |
///
/// Every worst case is the same frame and quantizer - the hard scene edge at `qindex` 160, whose
/// two halves have unrelated statistics - and the original value of `8` carried nearly twice
/// the error there against `2` while saving 7% of the encode. `1` is not the value because it
/// evaluates 181,557 transform-type candidates against the exhaustive search's 700,004, which
/// no longer clears the
/// four-fold reduction the shortcuts exist for and
/// `the_search_shortcuts_stay_within_their_rate_and_distortion_bound` asserts; `2` clears it with
/// 155,143. What remains at `2` is the estimator mixing statistics across regions rather than the
/// sampling rate, which no interval fixes.
pub(super) const TYPE_GAIN_SAMPLE_INTERVAL: usize = 2;

/// Probes a transform size's accumulated gain ratio remembers, as the window of an exponential
/// recency weighting.
///
/// PLACEHOLDER_DOC
pub(super) const TYPE_GAIN_MEMORY: usize = 8;

/// Transform sizes [`FrameEncoder::type_gain`] accumulates over: `TX_4X4` through `TX_32X32`,
/// which is every size [`super::transform::forward_transform`] implements.
const TYPE_GAIN_SIZES: usize = 4;

/// One transform size's accumulated probe measurement for the frame.
#[derive(Clone, Copy, Default)]
struct TypeGain {
    /// Summed DCT-only cost of every block probed at this size.
    dct_cost: i64,
    /// Summed best-of-set cost of the same blocks, so `dct_cost - best_cost` is what the type
    /// search has been measured to be worth at this size.
    best_cost: i64,
}

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
    /// The current trial's own probe measurement, or a zeroed pair when it did not probe. A
    /// trial that probed is corrected by what it measured itself, exactly as every trial was
    /// before the measurement was sampled; only the trials that skip the probe fall back to
    /// [`Self::type_gain`].
    probe_dct_cost: i64,
    probe_best_cost: i64,
    /// Size searches run so far this frame, which is what [`TYPE_GAIN_SAMPLE_INTERVAL`] samples.
    size_searches: usize,
    /// Summed DCT-only cost of every block the current trial actually searched, which is the base
    /// the measured gain is extrapolated over. Zero-skipped blocks are excluded: no transform type
    /// can improve a block that codes no coefficients.
    trial_searched_cost: i64,
    /// Set by [`Self::without_search_shortcuts`] to restore the original exhaustive search, so a
    /// test can compare the shortcuts against the search they stand in for.
    #[cfg(test)]
    exhaustive: bool,
    /// The sampling interval in force, so a test can sweep it and measure what
    /// [`TYPE_GAIN_SAMPLE_INTERVAL`] costs at each value instead of asserting the shipped one is
    /// right. Outside tests the constant is read directly.
    #[cfg(test)]
    type_gain_interval: usize,
    /// The recency window in force, so a test can sweep it the same way. Outside tests
    /// [`TYPE_GAIN_MEMORY`] is read directly.
    #[cfg(test)]
    type_gain_memory: usize,
    /// Transform-type candidates actually transformed, quantized and reconstructed, which is the
    /// work the shortcuts exist to remove.
    #[cfg(test)]
    candidates_evaluated: u64,
}

/// What one encode of a tile did, for the tests that compare two searches against each other.
#[cfg(test)]
pub(crate) struct SearchReport {
    pub(crate) tile: Vec<u8>,
    pub(crate) reconstruction: Vec<u8>,
    pub(crate) coded_width: usize,
    pub(crate) trace: Vec<(usize, Av1TxType)>,
    pub(crate) candidates_evaluated: u64,
}

/// One transform type considered for a block, with everything the winner needs to be written:
/// the `tx_type` symbol's index in its set, the quantized levels, the reconstructed residual,
/// and the `sse + lambda * bits` cost the search minimizes.
struct TxCandidate {
    symbol: usize,
    tx_type: Av1TxType,
    levels: Vec<i32>,
    reconstructed: Vec<i16>,
    cost: i64,
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
            probe_dct_cost: 0,
            probe_best_cost: 0,
            size_searches: 0,
            trial_searched_cost: 0,
            #[cfg(test)]
            exhaustive: false,
            #[cfg(test)]
            type_gain_interval: TYPE_GAIN_SAMPLE_INTERVAL,
            #[cfg(test)]
            type_gain_memory: TYPE_GAIN_MEMORY,
            #[cfg(test)]
            candidates_evaluated: 0,
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

    /// Overrides the recency window, so a test can measure the estimator between remembering one
    /// probe and remembering the whole frame. `usize::MAX` disables the decay, which is the
    /// frame-wide accumulation this replaced.
    #[cfg(test)]
    pub(crate) fn with_type_gain_memory(mut self, memory: usize) -> Self {
        assert!(memory >= 1, "a window of 0 remembers nothing, not even the probe just taken");
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
        const SB4: usize = 16; // 64×64 superblock in MI units
        let mut r = 0;
        while r < self.mi_rows {
            self.left_level.fill(0);
            self.left_dc.fill(0);
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
                let (depth_cdf, _) = cdf::tx_depth_cdf(bw);
                let depth = (largest / tx_width).trailing_zeros() as usize;
                self.sym.encode_symbol(depth, depth_cdf);
            }
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
        let (_, max_depth) = cdf::tx_depth_cdf(bw);
        let probing = self.shortcuts() && self.sample_type_gain();
        let mut best = (0usize, i64::MAX);
        for depth in 0..=max_depth {
            let tx_width = (largest >> depth).max(4);
            if tx_width > MAX_FORWARD_TX {
                continue;
            }
            let snapshot = self.snapshot(r, c, bw);
            self.probe_budget = if probing { TYPE_GAIN_PROBES } else { 0 };
            self.probe_dct_cost = 0;
            self.probe_best_cost = 0;
            self.trial_searched_cost = 0;
            let cost = self.code_block_transforms(r, c, bw, tx_width, false);
            self.restore(snapshot);
            let cost = self.corrected_trial_cost(cost, tx_width);
            if cost < best.1 {
                best = (tx_width, cost);
            }
            if tx_width == 4 {
                break;
            }
        }
        // Nothing outside a size trial may probe: every other speculative pass stays on the
        // set's DCT alone, which is where the shortcut's speedup comes from.
        self.probe_budget = 0;
        debug_assert_ne!(best.0, 0, "every coding block has one legal transform size");
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

    /// Discounts a size trial's DCT-only cost by what the type search has been measured to be
    /// worth at this transform size.
    ///
    /// With `p` this size's probed blocks and `s` every block this trial searched, the estimate
    /// is `sum(best_p)/sum(dct_p)` applied to `sum(dct_s)`, which reduces to subtracting
    /// `(dct_p - best_p) * dct_s / dct_p`. Blocks the zero-block shortcut decided are not in `s`:
    /// no transform type improves a block that codes no coefficients. A trial that probed uses
    /// its own `p`; one that skipped the probe uses every block sampled at that size so far this
    /// frame instead, so it is still corrected - by the ratio a probe of its own would have been
    /// measuring.
    fn corrected_trial_cost(&self, cost: i64, tx_width: usize) -> i64 {
        let (dct, best) = if self.probe_dct_cost > 0 {
            (self.probe_dct_cost, self.probe_best_cost)
        } else {
            let gain = self.type_gain[type_gain_slot(tx_width)];
            (gain.dct_cost, gain.best_cost)
        };
        if dct <= 0 {
            return cost;
        }
        let measured = dct - best;
        cost - measured.saturating_mul(self.trial_searched_cost) / dct
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
        let trial = self.shortcuts() && !emit;
        let mut probing = false;
        if trial && candidates.len() > 1 {
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
            let levels = self.quantize(&coefficients);
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
            // The keep test is strictly less, so an exact tie keeps the earlier candidate and the
            // winner is a function of `candidates` order alone. Ties are not hypothetical: on the
            // 96x80 test pattern of `nonlossless_tests` the smallest emitting-pass margins are 4
            // (a 4x4 block between `DCT_DCT` and `IDTX`) and 0 (a 4x4 block between `ADST_ADST`
            // and `DCT_DCT`), where the set order alone decides it. That is the margin issue #231
            // was about: a one-unit cost difference flips which type such a block writes, so the
            // comparison has to be a total order over a fixed candidate order rather than
            // anything that depends on evaluation order or on the host. Which type wins there is
            // therefore a property of the pattern, not of the encoder, and is not asserted; that
            // the *bitstream* comes out the same everywhere is, in `nonlossless_tests`.
            let keep = if probing {
                tx_type == Av1TxType::DctDct || best.is_none()
            } else {
                best.as_ref().is_none_or(|best| cost < best.cost)
            };
            if keep {
                best = Some(TxCandidate {
                    symbol,
                    tx_type,
                    levels,
                    reconstructed,
                    cost,
                });
            }
        }
        if probing {
            let dct = best.as_ref().map_or(0, |best| best.cost);
            let best_of_set = cheapest.min(dct);
            self.probe_dct_cost += dct;
            self.probe_best_cost += best_of_set;
            let memory = self.type_gain_memory();
            let gain = &mut self.type_gain[type_gain_slot(size)];
            // Recency weighting: what the accumulator already holds is aged by `(n-1)/n` before
            // the new probe joins it, so a probe's influence decays away over the following `n`
            // and the ratio a block reads back is the one its own neighbourhood measured. One
            // multiply and one divide per accumulator, on the sampled trials only.
            //
            // `usize::MAX` is the sentinel for no decay at all, which is the frame-wide
            // accumulation this replaced; a test sweeps it as the far end of the window.
            if memory != usize::MAX {
                let (num, den) = (memory as i64 - 1, memory as i64);
                gain.dct_cost = gain.dct_cost * num / den;
                gain.best_cost = gain.best_cost * num / den;
            }
            gain.dct_cost += dct;
            gain.best_cost += best_of_set;
        }
        // `tx_type` itself is only read by the trace the tests assert on; the bitstream carries
        // its `symbol` index instead.
        #[cfg_attr(not(test), allow(unused_variables))]
        let TxCandidate {
            symbol,
            tx_type,
            levels,
            reconstructed,
            cost,
        } = best.expect("every transform size has at least one candidate type");

        for row in 0..size {
            let start = (y + row) * self.coded_w + x;
            let destination = &mut self.recon[start..start + size];
            destination.fill(prediction);
            add_residual_row(&reconstructed[row * size..(row + 1) * size], destination);
        }

        let coded = self.code_coefficients(x >> 2, y >> 2, block_width, size, &levels, &scan, emit);
        #[cfg(test)]
        if emit {
            self.emitted.push((size, tx_type));
        }
        // The decoder reads `tx_type` after the coefficients, and only for a block that was not
        // fully skipped; a skipped block's type is irrelevant because its residual is zero.
        if coded && emit {
            // DC_PRED is the only y_mode this encoder signals, so the CDF's intra direction is 0.
            if let Some(tx_cdf) = cdf::tx_type_cdf(set, size, 0) {
                self.sym.encode_symbol(symbol, tx_cdf);
            }
        }
        if trial {
            self.trial_searched_cost += cost;
        }
        cost
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
        self.code_coefficients(x >> 2, y >> 2, block_width, size, &levels, scan, emit);
        #[cfg(test)]
        if emit {
            // No `tx_type` symbol follows a block the decoder reads as fully skipped, so the
            // pair the trace records is the one the search would have defaulted to.
            self.emitted.push((size, Av1TxType::DctDct));
        }
        energy + self.lambda * ZERO_BLOCK_BITS
    }

    /// Forward quantization: the exact inverse of the `level * q` dequantization
    /// [`inverse_transform`] applies, rounded to nearest.
    fn quantize(&self, coefficients: &[i32]) -> Vec<i32> {
        coefficients
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let step = i64::from(if index == 0 {
                    self.dc_quant
                } else {
                    self.ac_quant
                });
                let magnitude = i64::from(value).abs();
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
                let mut i = nbits as isize - 2;
                while i >= 0 {
                    self.sym.encode_literal(((extra >> i) & 1) as u32, 1);
                    i -= 1;
                }
            }
        }

        // Base levels + base range, scanned from the last coefficient back to DC.
        let mut levels = vec![0i32; count];
        for c in (0..eob).rev() {
            let pos = scan[c];
            let level = quant[pos].abs();
            if c == eob - 1 {
                let ctx = coeff_base_eob_ctx(c, count);
                if emit {
                    self.sym.encode_symbol(
                        (level.min(3) - 1) as usize,
                        cdf::coeff_base_eob_cdf(qctx, tx_ctx, ptype, ctx),
                    );
                }
            } else {
                let ctx = coeff_base_ctx(pos, &levels, size);
                if emit {
                    self.sym.encode_symbol(
                        level.min(3) as usize,
                        cdf::coeff_base_cdf(qctx, tx_ctx, ptype, ctx),
                    );
                }
            }
            if level > NUM_BASE_LEVELS {
                let br_ctx = coeff_br_ctx(pos, &levels, size);
                let mut rem = level - 3;
                for _ in 0..4 {
                    let brv = rem.min(3);
                    if emit {
                        self.sym.encode_symbol(
                            brv as usize,
                            cdf::coeff_br_cdf(qctx, tx_ctx, ptype, br_ctx),
                        );
                    }
                    rem -= brv;
                    if brv < 3 {
                        break;
                    }
                }
            }
            levels[pos] = level;
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

        let cul = levels.iter().sum::<i32>().min(63) as u8;
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

/// Bits a coefficient block is estimated to cost, as the `all_zero` flag plus, when the block is
/// coded, the end-of-block position and an exp-Golomb-shaped magnitude and sign per coefficient.
/// Only the relative ordering of candidates matters, so this stays a closed form over the
/// quantized levels rather than a trial arithmetic encode.
fn estimate_rate(levels: &[i32], scan: &[usize]) -> i64 {
    let mut eob = 0usize;
    for (index, &position) in scan.iter().enumerate() {
        if levels[position] != 0 {
            eob = index + 1;
        }
    }
    if eob == 0 {
        return 1;
    }
    let mut bits = 1 + 2 * i64::from(bit_length(eob as u32));
    for &position in scan.iter().take(eob) {
        let level = levels[position].unsigned_abs();
        bits += if level == 0 {
            1
        } else {
            2 + 2 * i64::from(bit_length(level))
        };
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
    for _ in 0..(len - 1) {
        sym.encode_literal(0, 1);
    }
    sym.encode_literal(1, 1);
    let mut i = len as isize - 2;
    while i >= 0 {
        sym.encode_literal((x >> i) & 1, 1);
        i -= 1;
    }
}
