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

/// `NUM_BASE_LEVELS` (§3).
const NUM_BASE_LEVELS: i32 = 2;
/// `NUM_BASE_LEVELS + COEFF_BASE_RANGE`, the golomb threshold (§5.11.39).
const COEFF_BASE_PLUS_RANGE: i32 = 14;
/// Largest square transform [`super::transform::forward_transform`] implements. `TX_64X64` has an
/// inverse kernel but no forward one, so a 64x64 coding block always signals a `tx_depth` of at
/// least 1.
const MAX_FORWARD_TX: usize = 32;
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
        }
    }

    /// Encodes the tile and returns the symbol-coded bytes (`decode_tile`, §5.11.2).
    pub(crate) fn encode(mut self) -> Vec<u8> {
        self.encode_superblocks();
        self.sym.finish()
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
        let snapshot = self.snapshot(r, c, bw);
        let whole = self.encode_block(r, c, bw, false);
        self.restore(snapshot);

        let half = (bw / 4) >> 1;
        let h = bw / 2;
        let snapshot = self.snapshot(r, c, bw);
        let split = self.encode_partition(r, c, h, false)
            + self.encode_partition(r, c + half, h, false)
            + self.encode_partition(r + half, c, h, false)
            + self.encode_partition(r + half, c + half, h, false);
        self.restore(snapshot);

        // Four sub-blocks each pay their own partition, skip, mode, and tx_size symbols; charging
        // a flat header cost keeps the search from splitting for a negligible distortion win.
        const SPLIT_HEADER_BITS: i64 = 24;
        split + self.lambda * SPLIT_HEADER_BITS < whole
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
    fn choose_tx_size(&mut self, r: usize, c: usize, bw: usize) -> usize {
        let largest = bw.min(MAX_TX_WIDTH);
        // Only the depth cap is needed here; it does not vary with the neighbour context.
        let (_, max_depth) = cdf::tx_depth_cdf(bw, 0);
        let mut best = (0usize, i64::MAX);
        for depth in 0..=max_depth {
            let tx_width = (largest >> depth).max(4);
            if tx_width > MAX_FORWARD_TX {
                continue;
            }
            let snapshot = self.snapshot(r, c, bw);
            let cost = self.code_block_transforms(r, c, bw, tx_width, false);
            self.restore(snapshot);
            if cost < best.1 {
                best = (tx_width, cost);
            }
            if tx_width == 4 {
                break;
            }
        }
        debug_assert_ne!(best.0, 0, "every coding block has one legal transform size");
        best.0
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
        let candidates: Vec<(usize, Av1TxType)> = inverse
            .iter()
            .enumerate()
            .filter_map(|(symbol, &(_, tx_type))| Some((symbol, tx_type?)))
            .collect();
        let scan = cdf::up_right_diagonal_scan(size);
        let mut best: Option<TxCandidate> = None;
        for &(symbol, tx_type) in &candidates {
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
            if best.as_ref().is_none_or(|best| cost < best.cost) {
                best = Some(TxCandidate {
                    symbol,
                    tx_type,
                    levels,
                    reconstructed,
                    cost,
                });
            }
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

        // §5.11.39 `coeffs` reads `transform_type()` immediately after `all_zero` and before
        // `eob_pt`, so the symbol goes to the coefficient coder rather than after it. DC_PRED is
        // the only y_mode this encoder signals, so the CDF's intra direction is 0.
        let tx_type_symbol = cdf::tx_type_cdf(set, size, 0).map(|tx_cdf| (symbol, tx_cdf));
        self.code_coefficients(
            x >> 2,
            y >> 2,
            block_width,
            size,
            &levels,
            &scan,
            tx_type_symbol,
            emit,
        );
        #[cfg(test)]
        if emit {
            self.emitted.push((size, tx_type));
        }
        cost
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
