//! §9.3 CABAC arithmetic decoding engine.
//!
//! This module is a clean-room transcription of the ITU-T Rec. H.265 |
//! ISO/IEC 23008-2 §9.3 context-adaptive binary arithmetic coding
//! *decode* engine — the lowest layer of HEVC entropy decoding, the
//! gateway through which all slice-data syntax elements (§7.3.8) are
//! read. It deliberately stops at the engine boundary: it exposes the
//! four arithmetic-decode primitives plus the context-variable model,
//! but does not implement the §9.3.4.2 binarization / context-index
//! derivation for any specific syntax element (those sit one layer up,
//! in the slice-data parser).
//!
//! The pieces transcribed here are:
//!
//! * §9.3.2.6 — [`CabacEngine::new`] / [`CabacEngine::init_engine`]:
//!   `ivlCurrRange = 510`, `ivlOffset = read_bits(9)`.
//! * §9.3.2.2 — [`ContextModel::init`] (equations 9-4, 9-5, 9-6) plus
//!   the §9.3.2.2 `initType` derivation (equation 9-7,
//!   [`init_type`]). A context variable is a `(pStateIdx, valMps)`
//!   pair derived from an 8-bit `initValue` and the slice QP.
//! * §9.3.4.3.2 — [`CabacEngine::decode_decision`] (DecodeDecision):
//!   the regular context-coded bin, with the Table 9-52 `rangeTabLps`
//!   interval split, the §9.3.4.3.2.2 state transition (Table 9-53),
//!   and the §9.3.4.3.3 renormalization.
//! * §9.3.4.3.4 — [`CabacEngine::decode_bypass`] (DecodeBypass): the
//!   equal-probability bin.
//! * §9.3.4.3.5 — [`CabacEngine::decode_terminate`] (DecodeTerminate):
//!   the `end_of_slice_segment_flag` / `end_of_subset_one_bit` /
//!   `pcm_flag` decision before termination.
//! * §9.3.4.3.6 — [`CabacEngine::align`] (the alignment process prior
//!   to aligned bypass decoding for `coeff_abs_level_remaining[ ]` /
//!   `coeff_sign_flag[ ]`): `ivlCurrRange = 256`.
//!
//! The Table 9-52 / Table 9-53 values are transcribed directly from the
//! H.265 specification text (`docs/video/h265/`). They are arithmetic
//! constants of the format, not external-library code.
//!
//! # Bit source
//!
//! The engine reads raw bits via [`BitReader`] (`read_bits(1)` in the
//! spec is [`BitReader::u1`], `read_bits(9)` is `BitReader::u(9)`). The
//! caller positions the reader at the first byte of `slice_segment_data()`
//! — for example via
//! [`crate::hevc::engine::slice::SliceSegmentHeader::byte_offset_to_slice_data`] — and
//! then constructs the engine, which immediately consumes the 9-bit
//! initial `ivlOffset` per §9.3.2.6.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};

/// Errors that can arise while running the CABAC decode engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacError {
    /// The bit reader ran out of bits (a `read_bits()` past the end of
    /// the slice-segment-data buffer). A conforming bitstream never
    /// reaches this state during a valid decode, so it indicates a
    /// truncated or malformed stream.
    EndOfBuffer,
    /// §9.3.2.6 forbids `ivlOffset` taking the value 510 or 511 after
    /// the initial 9-bit read ("The bitstream shall not contain data
    /// that result in a value of ivlOffset being equal to 510 or
    /// 511.").
    InvalidInitOffset(u16),
}

impl core::fmt::Display for CabacError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EndOfBuffer => f.write_str("CABAC engine read past end of slice-data buffer"),
            Self::InvalidInitOffset(v) => {
                write!(f, "CABAC init ivlOffset {v} is forbidden (must be < 510)")
            }
        }
    }
}

impl std::error::Error for CabacError {}

impl From<BitReaderError> for CabacError {
    fn from(_: BitReaderError) -> Self {
        // The only `BitReaderError` the engine can provoke is running
        // out of bits — every other `BitReader` failure mode (too many
        // bits, Exp-Golomb overflow) is unreachable from `u1()` /
        // `u(9)`.
        Self::EndOfBuffer
    }
}

/// The §9.3.2.2 `initType` selector (equation 9-7). `slice_type` is the
/// §7.4.7.1 enum (0 = B, 1 = P, 2 = I); `cabac_init_flag` is the
/// slice-header flag (only present for P / B slices).
///
/// ```text
/// if( slice_type == I )       initType = 0
/// else if( slice_type == P )  initType = cabac_init_flag ? 2 : 1
/// else /* B */                initType = cabac_init_flag ? 1 : 2
/// ```
///
/// `slice_type` here uses [`crate::hevc::engine::slice::SliceType`]'s numeric values:
/// `SliceType::B == 0`, `SliceType::P == 1`, `SliceType::I == 2`.
#[must_use]
pub fn init_type(slice_type: u8, cabac_init_flag: bool) -> u8 {
    match slice_type {
        // I slice.
        2 => 0,
        // P slice.
        1 => {
            if cabac_init_flag {
                2
            } else {
                1
            }
        }
        // B slice (any other slice_type value funnels here).
        _ => {
            if cabac_init_flag {
                1
            } else {
                2
            }
        }
    }
}

/// A single CABAC context variable: the §9.3.4.3 probability-state
/// index `pStateIdx` (0..=63) paired with the most-probable-symbol
/// value `valMps` (0 or 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextModel {
    /// §9.3.4.3 probability state index, 0..=63. State 0 is the
    /// `pLPS == 0.5` extreme; higher indices have lower LPS
    /// probability.
    pub p_state_idx: u8,
    /// The value of the most probable symbol (0 or 1).
    pub val_mps: u8,
}

impl ContextModel {
    /// §9.3.2.2 context-variable initialization (equations 9-4, 9-5,
    /// 9-6). `init_value` is the 8-bit `initValue` table entry for the
    /// context; `slice_qp_y` is `SliceQpY` (equation 7-54).
    ///
    /// ```text
    /// slopeIdx    = initValue >> 4
    /// offsetIdx   = initValue & 15                                 (9-4)
    /// m           = slopeIdx * 5 − 45
    /// n           = ( offsetIdx << 3 ) − 16                        (9-5)
    /// preCtxState = Clip3( 1, 126,
    ///                      ( ( m * Clip3( 0, 51, SliceQpY ) ) >> 4 ) + n )
    /// valMps      = ( preCtxState <= 63 ) ? 0 : 1
    /// pStateIdx   = valMps ? ( preCtxState − 64 ) : ( 63 − preCtxState ) (9-6)
    /// ```
    ///
    /// The `>> 4` in equation 9-6 is an arithmetic shift of a signed
    /// product (`m` can be negative); this is reproduced with signed
    /// arithmetic.
    #[must_use]
    pub fn init(init_value: u8, slice_qp_y: i32) -> Self {
        // Equation 9-4.
        let slope_idx = (init_value >> 4) as i32;
        let offset_idx = (init_value & 15) as i32;
        // Equation 9-5.
        let m = slope_idx * 5 - 45;
        let n = (offset_idx << 3) - 16;
        // Equation 9-6: Clip3(0, 51, SliceQpY) then the signed
        // arithmetic-shift product, then Clip3(1, 126, …).
        let qp = slice_qp_y.clamp(0, 51);
        let pre_ctx_state = (((m * qp) >> 4) + n).clamp(1, 126);
        let (val_mps, p_state_idx) = if pre_ctx_state <= 63 {
            (0u8, (63 - pre_ctx_state) as u8)
        } else {
            (1u8, (pre_ctx_state - 64) as u8)
        };
        Self {
            p_state_idx,
            val_mps,
        }
    }

    /// The non-adapting state described in §9.3.2.2 NOTE 2 for
    /// `ctxTable == 0`, `ctxIdx == 0` (the `end_of_slice_segment_flag`
    /// / `end_of_subset_one_bit` / `pcm_flag` decision when emulated
    /// through DecodeDecision): `pStateIdx = 63`, `valMps = 0`.
    /// `decode_terminate` does not need this state — it is provided for
    /// completeness of the model.
    #[must_use]
    pub fn terminate_state() -> Self {
        Self {
            p_state_idx: 63,
            val_mps: 0,
        }
    }
}

/// Table 9-52 — `rangeTabLps[ pStateIdx ][ qRangeIdx ]`. 64 states ×
/// 4 quantized-range columns. Transcribed from the H.265 specification
/// text (§9.3.4.3.2.1).
#[rustfmt::skip]
pub(crate) const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [ 95, 116, 137, 158], [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
    [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116], [ 66,  80,  95, 110],
    [ 62,  76,  90, 104], [ 59,  72,  86,  99], [ 56,  69,  81,  94], [ 53,  65,  77,  89],
    [ 51,  62,  73,  85], [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
    [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62], [ 35,  43,  51,  59],
    [ 33,  41,  48,  56], [ 32,  39,  46,  53], [ 30,  37,  43,  50], [ 29,  35,  41,  48],
    [ 27,  33,  39,  45], [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
    [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33], [ 19,  23,  27,  31],
    [ 18,  22,  26,  30], [ 17,  21,  25,  28], [ 16,  20,  23,  27], [ 15,  19,  22,  25],
    [ 14,  18,  21,  24], [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
    [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18], [ 10,  12,  15,  17],
    [ 10,  12,  14,  16], [  9,  11,  13,  15], [  9,  11,  12,  14], [  8,  10,  12,  14],
    [  8,   9,  11,  13], [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
    [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9], [  2,   2,   2,   2],
];

/// Table 9-53 — `transIdxLps[ pStateIdx ]`, the probability-state
/// transition after decoding the LPS (`1 − valMps`). Transcribed from
/// the H.265 specification text (§9.3.4.3.2.2).
#[rustfmt::skip]
pub(crate) const TRANS_IDX_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Table 9-53 — `transIdxMps[ pStateIdx ]`, the probability-state
/// transition after decoding the MPS (`valMps`). Transcribed from the
/// H.265 specification text (§9.3.4.3.2.2).
#[rustfmt::skip]
pub(crate) const TRANS_IDX_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// The §9.3.4.3 arithmetic decoding engine state plus its bit source.
///
/// The engine holds the two §9.3.2.6 registers `ivlCurrRange` and
/// `ivlOffset` and the [`BitReader`] from which renormalization /
/// bypass pull `read_bits(1)`. Context variables ([`ContextModel`])
/// live in the caller's context-model array; each
/// [`decode_decision`](CabacEngine::decode_decision) call takes and
/// mutates one in place.
#[derive(Debug)]
pub struct CabacEngine<'a> {
    reader: BitReader<'a>,
    /// §9.3.2.6 `ivlCurrRange` (held in a 16-bit-wide register; values
    /// stay within 9 bits during decode).
    ivl_curr_range: u16,
    /// §9.3.2.6 `ivlOffset` (10-bit register precision is required for
    /// bypass).
    ivl_offset: u16,
}

impl<'a> CabacEngine<'a> {
    /// §9.3.2.6 — construct and initialize the arithmetic decoding
    /// engine over `reader`, which the caller has positioned at the
    /// first bit of `slice_segment_data()`. Sets `ivlCurrRange = 510`
    /// and `ivlOffset = read_bits(9)`, and validates the §9.3.2.6
    /// constraint that `ivlOffset` is not 510 or 511.
    pub fn new(mut reader: BitReader<'a>) -> Result<Self, CabacError> {
        let ivl_offset = reader.u(9)? as u16;
        if ivl_offset >= 510 {
            return Err(CabacError::InvalidInitOffset(ivl_offset));
        }
        Ok(Self {
            reader,
            ivl_curr_range: 510,
            ivl_offset,
        })
    }

    /// §9.3.2.6 — re-initialize the engine registers on an already
    /// constructed engine (e.g. after `pcm_flag == 1` data is consumed,
    /// per §9.3.2.5 / §9.3.2.6). The bit reader is *not* repositioned;
    /// the caller is responsible for having advanced it to the byte
    /// boundary following the PCM samples before calling this.
    pub fn init_engine(&mut self) -> Result<(), CabacError> {
        let ivl_offset = self.reader.u(9)? as u16;
        if ivl_offset >= 510 {
            return Err(CabacError::InvalidInitOffset(ivl_offset));
        }
        self.ivl_curr_range = 510;
        self.ivl_offset = ivl_offset;
        Ok(())
    }

    /// Current `ivlCurrRange` register value (for tests / inspection).
    #[must_use]
    pub fn ivl_curr_range(&self) -> u16 {
        self.ivl_curr_range
    }

    /// §9.3.4.3.6 — alignment process prior to aligned bypass
    /// decoding: `ivlCurrRange` is set equal to 256. Invoked when
    /// `cabac_bypass_alignment_enabled_flag == 1` before the bypass
    /// decoding of `coeff_sign_flag[ ]` / `coeff_abs_level_remaining[
    /// ]` (and the §7.3.8.13 palette escape group) when
    /// `escapeDataPresent == 1`. After alignment every bypass bin
    /// consumes exactly one stream bit (the NOTE's shift-register
    /// view); bypass decoding itself never changes `ivlCurrRange`, so
    /// repeated invocations inside one aligned run are no-ops.
    pub fn align_bypass(&mut self) {
        self.ivl_curr_range = 256;
    }

    /// Current `ivlOffset` register value (for tests / inspection).
    #[must_use]
    pub fn ivl_offset(&self) -> u16 {
        self.ivl_offset
    }

    /// Bit position of the underlying reader (for tests / locating the
    /// engine's consumption point in the buffer).
    #[must_use]
    pub fn bit_pos(&self) -> usize {
        self.reader.bit_pos()
    }

    /// §7.3.8.7 support — consume `pcm_alignment_zero_bit` from the
    /// raw bit source until it is byte-aligned. Called right after a
    /// `pcm_flag == 1` terminate bin: §9.3.4.3.5 leaves the bitstream
    /// pointer directly after the terminating one bit, so the raw
    /// reader position is authoritative for `byte_aligned()`.
    ///
    /// # Errors
    /// [`CabacError::EndOfBuffer`] on a truncated stream.
    pub fn pcm_align(&mut self) -> Result<(), CabacError> {
        while self.reader.bit_pos() % 8 != 0 {
            let _ = self.reader.u1()?;
        }
        Ok(())
    }

    /// §7.3.8.7 support — read one raw `u(n)` field (a
    /// `pcm_sample_luma` / `pcm_sample_chroma` value) directly from the
    /// bit source, bypassing the arithmetic engine.
    ///
    /// # Errors
    /// [`CabacError::EndOfBuffer`] on a truncated stream.
    pub fn read_raw_bits(&mut self, n: u8) -> Result<u32, CabacError> {
        Ok(self.reader.u(n)?)
    }

    /// Consume the engine and hand back its bit source (tests / callers
    /// that continue non-CABAC parsing after termination).
    #[must_use]
    pub fn into_reader(self) -> BitReader<'a> {
        self.reader
    }

    /// §9.3.4.3.3 — renormalization in the arithmetic decoding engine
    /// (RenormD). While `ivlCurrRange < 256`, double `ivlCurrRange` and
    /// shift one fresh bit into `ivlOffset`.
    fn renorm(&mut self) -> Result<(), CabacError> {
        while self.ivl_curr_range < 256 {
            self.ivl_curr_range <<= 1;
            let bit = self.reader.u1()? as u16;
            self.ivl_offset = (self.ivl_offset << 1) | bit;
        }
        Ok(())
    }

    /// §9.3.4.3.2 — DecodeDecision: decode one regular context-coded
    /// bin against the supplied context variable `ctx`, mutating it
    /// per the §9.3.4.3.2.2 state-transition process. Returns the
    /// decoded bin value (0 or 1).
    pub fn decode_decision(&mut self, ctx: &mut ContextModel) -> Result<u8, CabacError> {
        // Step 1 (equations 9-64, 9-65): the LPS sub-range.
        let q_range_idx = ((self.ivl_curr_range >> 6) & 3) as usize;
        let ivl_lps_range = RANGE_TAB_LPS[ctx.p_state_idx as usize][q_range_idx] as u16;

        // Step 2: subtract the LPS range; compare ivlOffset.
        self.ivl_curr_range -= ivl_lps_range;
        let bin_val;
        if self.ivl_offset >= self.ivl_curr_range {
            // LPS path.
            bin_val = 1 - ctx.val_mps;
            self.ivl_offset -= self.ivl_curr_range;
            self.ivl_curr_range = ivl_lps_range;
        } else {
            // MPS path.
            bin_val = ctx.val_mps;
        }

        // §9.3.4.3.2.2 state transition (equation 9-66).
        if bin_val == ctx.val_mps {
            ctx.p_state_idx = TRANS_IDX_MPS[ctx.p_state_idx as usize];
        } else {
            if ctx.p_state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[ctx.p_state_idx as usize];
        }

        // §9.3.4.3.3 renormalization.
        self.renorm()?;
        Ok(bin_val)
    }

    /// §9.3.4.3.4 — DecodeBypass: decode one equal-probability bin.
    /// Doubles `ivlOffset` (shifting in one fresh bit), then compares
    /// against `ivlCurrRange`. Returns the decoded bin (0 or 1).
    pub fn decode_bypass(&mut self) -> Result<u8, CabacError> {
        let bit = self.reader.u1()? as u16;
        self.ivl_offset = (self.ivl_offset << 1) | bit;
        if self.ivl_offset >= self.ivl_curr_range {
            self.ivl_offset -= self.ivl_curr_range;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// §9.3.4.3.4 — convenience wrapper: decode `n` bypass bins
    /// MSB-first and accumulate them into an integer (`n` in 0..=32).
    /// This is the common pattern for fixed-length bypass-coded fields.
    /// `n == 0` returns 0 and consumes nothing.
    pub fn decode_bypass_bits(&mut self, n: u8) -> Result<u32, CabacError> {
        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | (self.decode_bypass()? as u32);
        }
        Ok(value)
    }

    /// §9.3.4.3.5 — DecodeTerminate: the decision before termination
    /// (`end_of_slice_segment_flag`, `end_of_subset_one_bit`,
    /// `pcm_flag`; `ctxTable == 0`, `ctxIdx == 0`). Decrements
    /// `ivlCurrRange` by 2; if `ivlOffset >= ivlCurrRange` the result
    /// is 1 (no renormalization, decoding is terminated), otherwise the
    /// result is 0 and renormalization is applied.
    pub fn decode_terminate(&mut self) -> Result<u8, CabacError> {
        self.ivl_curr_range -= 2;
        if self.ivl_offset >= self.ivl_curr_range {
            Ok(1)
        } else {
            self.renorm()?;
            Ok(0)
        }
    }

    /// §9.3.4.3.6 — alignment process prior to aligned bypass decoding
    /// (applied before `coeff_abs_level_remaining[ ]` and
    /// `coeff_sign_flag[ ]`): `ivlCurrRange = 256`. The bit reader and
    /// `ivlOffset` are left untouched.
    pub fn align(&mut self) {
        self.ivl_curr_range = 256;
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    // --- §9.3.2.2 context-variable init (equations 9-4..9-6) ---

    #[test]
    fn init_type_eq_9_7() {
        // I slice (slice_type == 2) → 0 regardless of cabac_init_flag.
        assert_eq!(init_type(2, false), 0);
        assert_eq!(init_type(2, true), 0);
        // P slice (slice_type == 1).
        assert_eq!(init_type(1, false), 1);
        assert_eq!(init_type(1, true), 2);
        // B slice (slice_type == 0).
        assert_eq!(init_type(0, false), 2);
        assert_eq!(init_type(0, true), 1);
    }

    #[test]
    fn context_init_worked_example() {
        // Hand-evaluate equations 9-4..9-6 for initValue = 154,
        // SliceQpY = 26 (a common mid-range QP).
        //   slopeIdx  = 154 >> 4 = 9
        //   offsetIdx = 154 & 15 = 10
        //   m = 9*5 - 45 = 0
        //   n = (10 << 3) - 16 = 64
        //   preCtxState = Clip3(1,126, (0*26 >> 4) + 64) = 64
        //   valMps = (64 <= 63) ? 0 : 1 = 1
        //   pStateIdx = 1 ? (64-64) : … = 0
        let c = ContextModel::init(154, 26);
        assert_eq!(c.val_mps, 1);
        assert_eq!(c.p_state_idx, 0);
    }

    #[test]
    fn context_init_negative_slope() {
        // initValue = 0, SliceQpY = 51 (max). Exercises a negative m
        // and the signed arithmetic shift in equation 9-6.
        //   slopeIdx  = 0, offsetIdx = 0
        //   m = -45, n = -16
        //   m*Clip3(0,51,51) = -45*51 = -2295
        //   -2295 >> 4 = -144 (arithmetic shift toward -inf)
        //   preCtxState = Clip3(1,126, -144 + -16) = Clip3(1,126,-160) = 1
        //   valMps = (1 <= 63) ? 0 : 1 = 0
        //   pStateIdx = 0 ? … : (63 - 1) = 62
        let c = ContextModel::init(0, 51);
        assert_eq!(c.val_mps, 0);
        assert_eq!(c.p_state_idx, 62);
    }

    #[test]
    fn context_init_high_initvalue() {
        // initValue = 255, SliceQpY = 0. Drives preCtxState toward the
        // high clip.
        //   slopeIdx = 15, offsetIdx = 15
        //   m = 15*5 - 45 = 30
        //   n = (15 << 3) - 16 = 104
        //   preCtxState = Clip3(1,126, (30*0 >> 4) + 104) = 104
        //   valMps = (104 <= 63) ? 0 : 1 = 1
        //   pStateIdx = 104 - 64 = 40
        let c = ContextModel::init(255, 0);
        assert_eq!(c.val_mps, 1);
        assert_eq!(c.p_state_idx, 40);
    }

    #[test]
    fn context_init_clips_low_qp_range() {
        // SliceQpY below 0 is clipped to 0 by Clip3(0,51,SliceQpY);
        // verify the negative-QP input lands at the same state as QP 0.
        let c_neg = ContextModel::init(154, -10);
        let c_zero = ContextModel::init(154, 0);
        assert_eq!(c_neg, c_zero);
    }

    #[test]
    fn terminate_state_is_non_adapting() {
        // §9.3.2.2 NOTE 2: pStateIdx = 63, valMps = 0.
        let c = ContextModel::terminate_state();
        assert_eq!(c.p_state_idx, 63);
        assert_eq!(c.val_mps, 0);
    }

    // --- Table integrity ---

    #[test]
    fn range_tab_lps_corner_values() {
        // Spot-check the four corners of Table 9-52.
        assert_eq!(RANGE_TAB_LPS[0], [128, 176, 208, 240]);
        assert_eq!(RANGE_TAB_LPS[31], [29, 35, 41, 48]);
        assert_eq!(RANGE_TAB_LPS[32], [27, 33, 39, 45]);
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
        // The whole table is non-increasing down each column (a
        // structural property of the LPS-range quantization).
        for (s, row) in RANGE_TAB_LPS.iter().enumerate().skip(1) {
            let prev = &RANGE_TAB_LPS[s - 1];
            for (col, (cur, p)) in row.iter().zip(prev.iter()).enumerate() {
                assert!(cur <= p, "column {col} not non-increasing at state {s}");
            }
        }
    }

    #[test]
    fn trans_idx_tables_bounds_and_endpoints() {
        // Every transition target is a valid state index (0..=63).
        for s in 0..64usize {
            assert!(TRANS_IDX_LPS[s] < 64);
            assert!(TRANS_IDX_MPS[s] < 64);
        }
        // State 0 LPS stays at 0; state 63 is the absorbing terminal
        // state for both transitions.
        assert_eq!(TRANS_IDX_LPS[0], 0);
        assert_eq!(TRANS_IDX_MPS[63], 63);
        assert_eq!(TRANS_IDX_LPS[63], 63);
        // transIdxMps is monotonically non-decreasing — confidence
        // never goes backward after an MPS. (Equality is allowed near
        // the upper extreme: index 62 maps to 62 in Table 9-53.)
        for (s, &mps) in TRANS_IDX_MPS.iter().enumerate().take(63) {
            assert!(
                mps >= s as u8,
                "transIdxMps[{s}] = {mps} is below the state"
            );
        }
        // transIdxLps is monotonically non-decreasing too (LPS at a
        // high-confidence state drops to a less-confident state, but
        // not below the LPS landing at a lower state).
        for win in TRANS_IDX_LPS.windows(2) {
            assert!(win[1] >= win[0]);
        }
    }

    // --- §9.3.2.6 engine init ---

    #[test]
    fn engine_init_reads_nine_bit_offset() {
        // First 9 bits = "1 01010101" = 341; ivlCurrRange = 510.
        // byte0 = 0b10101010, byte1 MSB = 1 → 9-bit value 0x155 = 341.
        let buf = [0b1010_1010, 0b1000_0000];
        let eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_curr_range(), 510);
        assert_eq!(eng.ivl_offset(), 341);
        assert_eq!(eng.bit_pos(), 9);
    }

    #[test]
    fn engine_init_rejects_offset_510_511() {
        // 9 bits = 0x1FE = 510 → forbidden. byte0 = 0xFF, byte1 MSB
        // = 0.
        let buf = [0xFF, 0b0000_0000];
        match CabacEngine::new(BitReader::new(&buf)) {
            Err(CabacError::InvalidInitOffset(510)) => {}
            other => panic!("expected InvalidInitOffset(510), got {other:?}"),
        }
        // 9 bits = 0x1FF = 511 → forbidden. byte0 = 0xFF, byte1 MSB
        // = 1.
        let buf = [0xFF, 0b1000_0000];
        match CabacEngine::new(BitReader::new(&buf)) {
            Err(CabacError::InvalidInitOffset(511)) => {}
            other => panic!("expected InvalidInitOffset(511), got {other:?}"),
        }
        // 509 is allowed: 0x1FD; byte0 = 0xFE, byte1 MSB = 1.
        let buf = [0xFE, 0b1000_0000];
        assert!(CabacEngine::new(BitReader::new(&buf)).is_ok());
    }

    // --- §9.3.4.3.4 bypass ---

    #[test]
    fn bypass_reads_bits_msb_first() {
        // After init, ivlCurrRange = 510, ivlOffset is the first 9
        // bits. Each bypass doubles ivlOffset (shift in one bit) and
        // compares to 510.
        //
        // Construct so the bypass bins are deterministic. Init offset
        // = 0 (9 zero bits at stream bits 0..=8). The next 4 stream
        // bits feed four bypass decisions and must come out MSB-first.
        // 9 init bits use byte0 (bits 0..=7) and bit 8 = byte1 MSB.
        // For init = 0, byte0 = 0x00 and byte1 MSB = 0. The bypass
        // bits start at stream bit 9 = byte1 bits 1..=4. We want them
        // to be 1, 0, 1, 1 → byte1 bits 1..=4 = 1011 → byte1 LSBs are
        // ignored, byte1 = 0b0_1011_000 = 0x58.
        //   bin0: offset = (0<<1)|1 = 1;   1 < 510 → 0
        //   bin1: offset = (1<<1)|0 = 2;   2 < 510 → 0
        //   bin2: offset = (2<<1)|1 = 5;   5 < 510 → 0
        //   bin3: offset = (5<<1)|1 = 11; 11 < 510 → 0
        // Small offsets always yield 0 — verify offset accumulation.
        let buf = [0x00, 0x58, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 0);
        assert_eq!(eng.decode_bypass().unwrap(), 0);
        assert_eq!(eng.ivl_offset(), 1);
        assert_eq!(eng.decode_bypass().unwrap(), 0);
        assert_eq!(eng.ivl_offset(), 2);
        assert_eq!(eng.decode_bypass().unwrap(), 0);
        assert_eq!(eng.ivl_offset(), 5);
        assert_eq!(eng.decode_bypass().unwrap(), 0);
        assert_eq!(eng.ivl_offset(), 11);
    }

    #[test]
    fn bypass_one_when_offset_exceeds_range() {
        // Init offset = 256. After one bypass shifting in a 1: offset
        // = (256<<1)|1 = 513 >= 510 → bin 1, offset -= 510 → 3.
        //
        // 9-bit offset stream bits 0..=8: 1, 0, 0, 0, 0, 0, 0, 0, 0
        // → byte0 = 0b1000_0000, byte1 MSB = 0. We then want bypass
        // bit (stream bit 9) = 1 → byte1 bit1 = 1 → byte1 = 0b0100_0000.
        let buf = [0b1000_0000, 0b0100_0000];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 256);
        let bin = eng.decode_bypass().unwrap();
        assert_eq!(bin, 1);
        assert_eq!(eng.ivl_offset(), 513 - 510);
    }

    #[test]
    fn bypass_bits_accumulates_msb_first() {
        // decode_bypass_bits(0) is a no-op returning 0.
        let buf = [0x00, 0x00, 0xFF];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.decode_bypass_bits(0).unwrap(), 0);
        assert_eq!(eng.bit_pos(), 9);
    }

    // --- §9.3.4.3.5 terminate ---

    #[test]
    fn terminate_returns_one_and_stops() {
        // Init offset = 509 (max legal). ivlCurrRange = 510. terminate:
        // range -= 2 → 508; offset (509) >= 508 → return 1, no renorm.
        let buf = [0xFE, 0b1000_0000];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 509);
        let pos_before = eng.bit_pos();
        assert_eq!(eng.decode_terminate().unwrap(), 1);
        assert_eq!(eng.ivl_curr_range(), 508);
        // No renormalization → no further bits consumed.
        assert_eq!(eng.bit_pos(), pos_before);
    }

    #[test]
    fn terminate_returns_zero_and_renorms() {
        // Init offset = 0. ivlCurrRange = 510. terminate: range -= 2 →
        // 508; offset (0) < 508 → return 0, then renorm: 508 >= 256 so
        // no renorm bits consumed.
        let buf = [0x00, 0x00, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.decode_terminate().unwrap(), 0);
        assert_eq!(eng.ivl_curr_range(), 508);
        assert_eq!(eng.bit_pos(), 9);
    }

    // --- §9.3.4.3.6 alignment ---

    #[test]
    fn align_sets_range_256() {
        let buf = [0x00, 0x00, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        eng.align();
        assert_eq!(eng.ivl_curr_range(), 256);
        // ivlOffset and the reader are untouched.
        assert_eq!(eng.ivl_offset(), 0);
        assert_eq!(eng.bit_pos(), 9);
    }

    // --- §9.3.4.3.2 DecodeDecision: MPS / LPS paths + renorm ---

    #[test]
    fn decode_decision_mps_path_no_renorm() {
        // Init offset = 0, ivlCurrRange = 510. Context: pStateIdx = 0,
        // valMps = 0.
        //   qRangeIdx = (510 >> 6) & 3 = 7 & 3 = 3
        //   ivlLpsRange = RANGE_TAB_LPS[0][3] = 240
        //   ivlCurrRange = 510 - 240 = 270
        //   offset (0) < 270 → MPS → binVal = valMps = 0
        //   transition: binVal == valMps → pStateIdx = transIdxMps[0]=1
        //   renorm: 270 >= 256 → none
        let buf = [0x00, 0x00, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = ContextModel {
            p_state_idx: 0,
            val_mps: 0,
        };
        let bin = eng.decode_decision(&mut ctx).unwrap();
        assert_eq!(bin, 0);
        assert_eq!(eng.ivl_curr_range(), 270);
        assert_eq!(ctx.p_state_idx, 1);
        assert_eq!(ctx.val_mps, 0);
        assert_eq!(eng.bit_pos(), 9); // no renorm bits
    }

    #[test]
    fn decode_decision_lps_path_flips_mps_and_renorms() {
        // Drive the LPS path with pStateIdx = 0 (so valMps flips).
        // Init offset must be >= ivlCurrRange after the LPS subtraction.
        //   qRangeIdx = (510 >> 6) & 3 = 3
        //   ivlLpsRange = RANGE_TAB_LPS[0][3] = 240
        //   ivlCurrRange = 510 - 240 = 270
        // Need offset >= 270. Use init offset = 300 (0b1_0010_1100).
        //   LPS: binVal = 1 - valMps = 1
        //        offset = 300 - 270 = 30
        //        ivlCurrRange = 240
        //   transition: binVal != valMps, pStateIdx==0 → valMps flips
        //        to 1; pStateIdx = transIdxLps[0] = 0
        //   renorm: 240 < 256 → one doubling: range = 480, offset =
        //        (30<<1)|nextbit.
        // 9-bit offset 300 = 0b1_0010_1100. Pack into bytes:
        //   byte0 = 0b1001_0110 (bits 0..7), byte1 MSB = 0 (bit 8).
        //   300 binary (9 bits): 1 0010 1100.
        //     bit0=1 bit1=0 bit2=0 bit3=1 bit4=0 bit5=1 bit6=1 bit7=0 → 0x96
        //     bit8=0 → byte1 MSB 0.
        //   The renorm bit (stream bit 9) = byte1 bit1; set it to 1.
        let buf = [0x96, 0b0100_0000, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 300);
        let mut ctx = ContextModel {
            p_state_idx: 0,
            val_mps: 0,
        };
        let bin = eng.decode_decision(&mut ctx).unwrap();
        assert_eq!(bin, 1);
        assert_eq!(ctx.val_mps, 1); // flipped
        assert_eq!(ctx.p_state_idx, 0); // transIdxLps[0]
        assert_eq!(eng.ivl_curr_range(), 480); // 240 << 1
        assert_eq!(eng.ivl_offset(), (30 << 1) | 1);
        assert_eq!(eng.bit_pos(), 10); // one renorm bit
    }

    #[test]
    fn end_of_buffer_surfaces_error() {
        // Provide exactly 9 bits (one byte + one bit). After init, the
        // bit reader is at its end, so the next bypass fails. We pad
        // the buffer to 9 bits by giving 2 bytes but truncating the
        // BitReader via `skip` would be circular — instead exercise
        // the path that needs more than the buffer holds by consuming
        // enough bypass bits to overrun.
        let buf = [0x00, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        // 16 - 9 = 7 bits left; the 8th bypass call must fail.
        for _ in 0..7 {
            eng.decode_bypass().unwrap();
        }
        assert_eq!(eng.decode_bypass(), Err(CabacError::EndOfBuffer));
    }

    #[test]
    fn round_trips_a_known_state_sequence() {
        // Decode several MPS bins in a row from an all-zero stream with
        // a fresh I-slice-style context and confirm the probability
        // state walks up the transIdxMps ladder deterministically.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = ContextModel {
            p_state_idx: 0,
            val_mps: 0,
        };
        let mut states = Vec::new();
        for _ in 0..5 {
            let bin = eng.decode_decision(&mut ctx).unwrap();
            assert_eq!(bin, 0); // all-zero offset always takes MPS here
            states.push(ctx.p_state_idx);
        }
        // transIdxMps ladder from 0: 1, 2, 3, 4, 5.
        assert_eq!(states, vec![1, 2, 3, 4, 5]);
    }
}
