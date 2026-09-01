//! §9.3.5 CABAC arithmetic *encoding* engine (the spec's informative
//! encoder, which "matches the arithmetic decoding engine described in
//! clause 9.3.4.3").
//!
//! The engine state is `ivlLow` / `ivlCurrRange` plus the
//! `firstBitFlag` / `bitsOutstanding` carry-control of the §9.3.5.4
//! renormalization (Figure 9-11) and its PutBit procedure
//! (Figure 9-12). The four entry points mirror the decoder:
//!
//! * §9.3.5.2 [`CabacEncoder::new`] — `ivlLow = 0`,
//!   `ivlCurrRange = 510`, `firstBitFlag = 1`, `bitsOutstanding = 0`.
//!   The same initialization re-arms the engine after PCM sample data
//!   ([`CabacEncoder::reinit`]).
//! * §9.3.5.3 [`CabacEncoder::encode_decision`] (EncodeDecision) — the
//!   Table 9-52 `rangeTabLps` interval split, the §9.3.4.3.2.2 state
//!   transition (Table 9-53), then renormalization.
//! * §9.3.5.5 [`CabacEncoder::encode_bypass`] (EncodeBypass).
//! * §9.3.5.6 [`CabacEncoder::encode_terminate`] (EncodeTerminate) —
//!   `end_of_slice_segment_flag` / `end_of_subset_one_bit` /
//!   `pcm_flag`; a `binVal == 1` runs the Figure 9-15 flush whose last
//!   written bit is the `rbsp_stop_one_bit` /
//!   `alignment_bit_equal_to_one`.
//!
//! Bits go into the caller's [`BitWriter`]; the encoder is constructed
//! at a byte-aligned position (the start of `slice_segment_data()`)
//! and, via the first-bit suppression, produces exactly the byte
//! sequence whose 9-bit preload the §9.3.2.6 decoder init consumes.
//!
//! Correctness contract: any bin sequence encoded through this engine
//! and decoded through [`crate::hevc::engine::cabac::CabacEngine`] (with identically
//! initialized context models) reproduces the original bins — pinned
//! by the roundtrip tests below.

use crate::hevc::engine::cabac::{ContextModel, RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};
use crate::hevc::engine::encoder::bitwriter::BitWriter;

/// Bins coded per unrolled [`CabacEncoder::encode_bypass_run`] step.
///
/// `ivlLow < 512` and `ivlCurrRange <= 510`, so `( ivlLow << n ) +
/// ivlCurrRange * value` needs `n + 10` bits; 16 leaves ample headroom
/// inside the `u32` register and covers a full 4x4 sub-block's
/// `coeff_sign_flag` run in one step.
const BYPASS_RUN_CHUNK_BINS: u8 = 16;

/// §9.3.5 arithmetic encoding engine over a borrowed [`BitWriter`].
#[derive(Debug)]
pub struct CabacEncoder {
    /// `ivlLow` — low end of the current sub-interval (10-bit register
    /// per the §9.3.5.2 NOTE; `u32` gives headroom).
    low: u32,
    /// `ivlCurrRange` — width of the current sub-interval (9 bits).
    range: u32,
    /// `firstBitFlag` — suppresses the very first PutBit output (the
    /// decoder's 9-bit preload supplies it implicitly).
    first_bit: bool,
    /// `bitsOutstanding` — pending carry-propagation bits.
    outstanding: u64,
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacEncoder {
    /// §9.3.5.2 InitEncoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            first_bit: true,
            outstanding: 0,
        }
    }

    /// §9.3.5.2 — re-initialize after PCM sample data (`pcm_flag == 1`
    /// coding units re-arm the engine exactly like a slice start).
    pub fn reinit(&mut self) {
        *self = Self::new();
    }

    /// Figure 9-12 PutBit(B): carry-over control. The first bit of a
    /// slice segment's arithmetic codeword is not written.
    fn put_bit(&mut self, w: &mut BitWriter, b: u8) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            w.put_bit(b);
        }
        while self.outstanding > 0 {
            w.put_bit(1 - b);
            self.outstanding -= 1;
        }
    }

    /// Figure 9-11 RenormE.
    fn renorm(&mut self, w: &mut BitWriter) {
        while self.range < 256 {
            if self.low >= 512 {
                self.low -= 512;
                self.put_bit(w, 1);
            } else if self.low < 256 {
                self.put_bit(w, 0);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// §9.3.5.3 EncodeDecision: one context-coded bin. Updates the
    /// context model exactly as the decoder's §9.3.4.3.2.2 transition.
    pub fn encode_decision(&mut self, w: &mut BitWriter, ctx: &mut ContextModel, bin: u8) {
        // Equation 9-64: qRangeIdx = ( ivlCurrRange >> 6 ) & 3.
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps = u32::from(RANGE_TAB_LPS[ctx.p_state_idx as usize][q_range_idx]);
        self.range -= lps;
        if bin != ctx.val_mps {
            // LPS path: the LPS sub-interval sits above the MPS one.
            self.low += self.range;
            self.range = lps;
            if ctx.p_state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[ctx.p_state_idx as usize];
        } else {
            ctx.p_state_idx = TRANS_IDX_MPS[ctx.p_state_idx as usize];
        }
        self.renorm(w);
    }

    /// §9.3.5.5 EncodeBypass: one equal-probability bin (renorm
    /// inlined per Figure 9-13).
    pub fn encode_bypass(&mut self, w: &mut BitWriter, bin: u8) {
        self.low <<= 1;
        if bin != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.low -= 1024;
            self.put_bit(w, 1);
        } else if self.low < 512 {
            self.put_bit(w, 0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
    }

    /// §9.3.5.5 over a *run* of `n` bypass bins at once, MSB-first.
    ///
    /// The per-bin loop is `ivlLow = ivlLow * 2 + binVal * ivlCurrRange`
    /// with a reduction after each bin, so unrolled over `n` bins it is
    /// the single step
    ///
    /// ```text
    /// ivlLow = ( ivlLow << n ) + ivlCurrRange * value
    /// ```
    ///
    /// followed by the same Figure 9-12 PutBit emission of the top `n`
    /// bits, each tested against a window scaled by the `i` bins still
    /// below it. That is an identity and not an approximation: the two
    /// paths agree bit for bit and leave identical `ivlLow` /
    /// `bitsOutstanding`, because the only difference is *when* a carry
    /// out of the lower bins is resolved — arithmetically here, through
    /// the outstanding-bit deferral in the bin-at-a-time engine — and
    /// both resolutions write the same bits. Pinned by
    /// `bypass_run_matches_bin_at_a_time`.
    ///
    /// This is the entry point the residual writer's bypass runs use:
    /// `coeff_sign_flag` (up to 16 in a row per sub-block), the
    /// Golomb-Rice and Exp-Golomb parts of `coeff_abs_level_remaining`,
    /// and the `last_sig_coeff_*` suffixes.
    ///
    /// `n <= 32`; the run is coded in [`BYPASS_RUN_CHUNK_BINS`]-bin
    /// chunks so the unrolled `ivlLow` stays inside `u32`.
    pub fn encode_bypass_run(&mut self, w: &mut BitWriter, value: u32, n: u8) {
        debug_assert!(n <= 32, "a bypass run is at most one u32 wide");
        // A one-bin "run" is the ordinary §9.3.5.5 step; going through the
        // unrolled form would only add a multiply. The residual writer
        // reaches this on every `cRiceParam == 0` remainder.
        if n <= 1 {
            if n == 1 {
                self.encode_bypass(w, (value & 1) as u8);
            }
            return;
        }
        if n <= BYPASS_RUN_CHUNK_BINS {
            self.encode_bypass_chunk(w, value & ((1u32 << n) - 1), n);
            return;
        }
        let mut left = n;
        while left > 0 {
            let take = left.min(BYPASS_RUN_CHUNK_BINS);
            left -= take;
            let chunk = (value >> left) & ((1u32 << take) - 1);
            self.encode_bypass_chunk(w, chunk, take);
        }
    }

    /// One unrolled §9.3.5.5 step over `n <= BYPASS_RUN_CHUNK_BINS` bins.
    fn encode_bypass_chunk(&mut self, w: &mut BitWriter, value: u32, n: u8) {
        // `ivlLow < 512` and `ivlCurrRange <= 510` on entry, so this stays
        // below `2 ** (n + 10)` — inside `u32` for `n <= 22`.
        self.low = (self.low << n) + self.range * value;
        for i in (0..n).rev() {
            // The `i` bins still below this one scale the Figure 9-13
            // window: `512 << i` is this bin's half, `1024 << i` its top.
            let half = 512u32 << i;
            if self.low >= half << 1 {
                self.low -= half << 1;
                self.put_bit(w, 1);
            } else if self.low < half {
                self.put_bit(w, 0);
            } else {
                self.low -= half;
                self.outstanding += 1;
            }
        }
    }

    /// MSB-first multi-bit bypass helper (the dual of
    /// [`crate::hevc::engine::cabac::CabacEngine::decode_bypass_bits`]).
    /// Routed through [`CabacEncoder::encode_bypass_run`].
    pub fn encode_bypass_bits(&mut self, w: &mut BitWriter, value: u32, n: u8) {
        self.encode_bypass_run(w, value, n);
    }

    /// §9.3.5.6 EncodeTerminate: `end_of_slice_segment_flag` /
    /// `end_of_subset_one_bit` / `pcm_flag`. `bin == 1` terminates the
    /// arithmetic codeword and flushes (Figure 9-15) — the last written
    /// bit is the `rbsp_stop_one_bit` / `alignment_bit_equal_to_one`;
    /// the writer is then ready for byte-aligned raw data (PCM) or the
    /// slice end.
    pub fn encode_terminate(&mut self, w: &mut BitWriter, bin: u8) {
        self.range -= 2;
        if bin != 0 {
            self.low += self.range;
            self.flush(w);
        } else {
            self.renorm(w);
        }
    }

    /// Figure 9-15 EncodeFlush.
    fn flush(&mut self, w: &mut BitWriter) {
        self.range = 2;
        self.renorm(w);
        self.put_bit(w, ((self.low >> 9) & 1) as u8);
        // WriteBits( ( ivlLow >> 7 ) & 3 | 1, 2 ) — the trailing 1 is
        // the terminating one bit the decoder sees as the last bit
        // inserted into ivlOffset.
        w.put_bits(((self.low >> 7) & 3) | 1, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;
    use crate::hevc::engine::cabac::CabacEngine;

    /// Encode `bins` as context-coded decisions with a single shared
    /// context (encoder and decoder start from the same init), append a
    /// terminate-1, and decode back.
    fn roundtrip_decisions(init_value: u8, qp: i32, bins: &[u8]) {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ectx = ContextModel::init(init_value, qp);
        for &b in bins {
            enc.encode_decision(&mut w, &mut ectx, b);
        }
        enc.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();

        let mut dec = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dctx = ContextModel::init(init_value, qp);
        for (i, &b) in bins.iter().enumerate() {
            assert_eq!(dec.decode_decision(&mut dctx).unwrap(), b, "bin {i}");
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1, "terminator");
        // The context models must have walked the same state path.
        assert_eq!(ectx, dctx, "context state after roundtrip");
    }

    #[test]
    fn decision_roundtrip_simple_patterns() {
        roundtrip_decisions(154, 26, &[0]);
        roundtrip_decisions(154, 26, &[1]);
        roundtrip_decisions(154, 26, &[1, 0, 1, 1, 0, 0, 1, 0]);
        roundtrip_decisions(200, 37, &[1; 64]);
        roundtrip_decisions(63, 0, &[0; 64]);
    }

    #[test]
    fn decision_roundtrip_pseudorandom_long() {
        // Deterministic LCG bins over several init values / QPs.
        let mut x = 0x1234_5678u32;
        let mut bins = Vec::new();
        for _ in 0..4096 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            bins.push(((x >> 16) & 1) as u8);
        }
        for (iv, qp) in [(154u8, 26), (79, 12), (227, 51), (16, 0)] {
            roundtrip_decisions(iv, qp, &bins);
        }
    }

    #[test]
    fn bypass_roundtrip() {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let pattern = 0xDEAD_BEEFu32;
        enc.encode_bypass_bits(&mut w, pattern, 32);
        enc.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();

        let mut dec = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        assert_eq!(dec.decode_bypass_bits(32).unwrap(), pattern);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    /// The run-at-a-time §9.3.5.5 path must be an exact identity for the
    /// bin-at-a-time one: same bytes, same `ivlLow`, same
    /// `bitsOutstanding`, same `firstBitFlag`, from every reachable
    /// engine state and for every run length the writer can emit.
    #[test]
    fn bypass_run_matches_bin_at_a_time() {
        // Prime the engine into a spread of distinct states first, so the
        // comparison is not made from `InitEncoder` alone: the deferral
        // that the two paths resolve differently only shows up when
        // `ivlCurrRange` and `bitsOutstanding` are non-trivial.
        let primers: [&[u8]; 6] = [
            &[],
            &[1],
            &[0, 1, 1, 0, 1],
            &[1; 17],
            &[0; 17],
            &[1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        ];
        let mut x = 0x9E37_79B9u32;
        for primer in primers {
            for n in 1..=32u8 {
                let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
                let mut values = vec![0u32, mask, mask >> 1, 1, mask ^ 1];
                for _ in 0..16 {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    values.push(x & mask);
                }
                for value in values {
                    let (run_bytes, run_enc) = {
                        let mut w = BitWriter::new();
                        let mut enc = CabacEncoder::new();
                        let mut ctx = ContextModel::init(154, 26);
                        for &b in primer {
                            enc.encode_decision(&mut w, &mut ctx, b);
                        }
                        enc.encode_bypass_run(&mut w, value, n);
                        (w, enc)
                    };
                    let (bin_bytes, bin_enc) = {
                        let mut w = BitWriter::new();
                        let mut enc = CabacEncoder::new();
                        let mut ctx = ContextModel::init(154, 26);
                        for &b in primer {
                            enc.encode_decision(&mut w, &mut ctx, b);
                        }
                        for i in (0..n).rev() {
                            enc.encode_bypass(&mut w, ((value >> i) & 1) as u8);
                        }
                        (w, enc)
                    };
                    let label = format!("primer {} n {n} value {value:#x}", primer.len());
                    assert_eq!(run_bytes.bit_len(), bin_bytes.bit_len(), "bit length, {label}");
                    assert_eq!(run_bytes.finish(), bin_bytes.finish(), "bytes, {label}");
                    assert_eq!(run_enc.low, bin_enc.low, "ivlLow, {label}");
                    assert_eq!(run_enc.range, bin_enc.range, "ivlCurrRange, {label}");
                    assert_eq!(
                        run_enc.outstanding, bin_enc.outstanding,
                        "bitsOutstanding, {label}"
                    );
                    assert_eq!(run_enc.first_bit, bin_enc.first_bit, "firstBitFlag, {label}");
                }
            }
        }
    }

    /// A run wider than one unrolled chunk still decodes back, so the
    /// chunking that keeps `ivlLow` inside `u32` is transparent.
    #[test]
    fn long_bypass_runs_roundtrip() {
        for n in [17u8, 24, 31, 32] {
            let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            let value = 0xC3A5_5A3Cu32 & mask;
            let mut w = BitWriter::new();
            let mut enc = CabacEncoder::new();
            enc.encode_bypass_run(&mut w, value, n);
            enc.encode_terminate(&mut w, 1);
            w.align_zero();
            let bytes = w.finish();

            let mut dec = CabacEngine::new(BitReader::new(&bytes)).expect("init");
            assert_eq!(dec.decode_bypass_bits(n).unwrap(), value, "n {n}");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn mixed_decision_bypass_terminate_roundtrip() {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ectx = [ContextModel::init(154, 26), ContextModel::init(31, 40)];
        // Interleave: ctx0, bypass, ctx1, terminate-0 (continue), repeat.
        for round in 0..50u32 {
            enc.encode_decision(&mut w, &mut ectx[0], (round & 1) as u8);
            enc.encode_bypass(&mut w, ((round >> 1) & 1) as u8);
            enc.encode_decision(&mut w, &mut ectx[1], ((round >> 2) & 1) as u8);
            enc.encode_terminate(&mut w, 0);
        }
        enc.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();

        let mut dec = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dctx = [ContextModel::init(154, 26), ContextModel::init(31, 40)];
        for round in 0..50u32 {
            assert_eq!(
                dec.decode_decision(&mut dctx[0]).unwrap(),
                (round & 1) as u8
            );
            assert_eq!(dec.decode_bypass().unwrap(), ((round >> 1) & 1) as u8);
            assert_eq!(
                dec.decode_decision(&mut dctx[1]).unwrap(),
                ((round >> 2) & 1) as u8
            );
            assert_eq!(dec.decode_terminate().unwrap(), 0, "round {round}");
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
        assert_eq!(ectx[0], dctx[0]);
        assert_eq!(ectx[1], dctx[1]);
    }

    /// The PCM shape: terminate-1 mid-stream, byte-align, raw bits,
    /// engine re-init on both sides, then more coded bins.
    #[test]
    fn pcm_style_flush_align_raw_reinit_roundtrip() {
        let raw: [u8; 5] = [0x11, 0x22, 0x33, 0xFE, 0x00];

        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ectx = ContextModel::init(154, 26);
        enc.encode_decision(&mut w, &mut ectx, 1);
        enc.encode_decision(&mut w, &mut ectx, 0);
        // pcm_flag == 1: terminate + flush, then alignment zeros + raw.
        enc.encode_terminate(&mut w, 1);
        w.align_zero();
        for &b in &raw {
            w.put_bits(u32::from(b), 8);
        }
        // §9.3.5.2: re-init after the PCM data, keep encoding.
        enc.reinit();
        enc.encode_decision(&mut w, &mut ectx, 1);
        enc.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();

        let mut dec = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dctx = ContextModel::init(154, 26);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 0);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
        // pcm_alignment_zero_bit run, raw PCM reads, §9.3.2.6 re-init.
        dec.pcm_align().unwrap();
        for &b in &raw {
            assert_eq!(dec.read_raw_bits(8).unwrap(), u32::from(b));
        }
        dec.init_engine().expect("reinit");
        assert_eq!(dec.decode_decision(&mut dctx).unwrap(), 1);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }
}
