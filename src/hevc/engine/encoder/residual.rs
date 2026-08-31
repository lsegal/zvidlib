//! §7.3.8.11 `residual_coding( )` on the write side.
//!
//! The mirror of [`crate::hevc::engine::residual::decode_residual_coding_with`]:
//! same walk, same order, same §9.3.4.2 `ctxInc` derivations — every one of
//! which is imported from [`crate::hevc::engine::binarization`] rather than
//! restated here, so the two directions cannot drift apart. The only
//! difference is the direction of each bin: where the decoder asks its source
//! for a bin and stores the result, this driver computes the bin from the
//! quantized `TransCoeffLevel` array it was handed and writes it.
//!
//! ## What is and is not supported
//!
//! This writer covers the shape the encoder actually emits: no transform skip,
//! no transquant bypass, no RDPCM, no sign-data hiding, no persistent Rice
//! adaptation, no bypass alignment, no extended precision. Each of those is a
//! range-extension or PPS flag the encoder's own parameter sets clear, and
//! leaving them out keeps the driver readable — the decoder still parses all
//! of them, so nothing is lost on the read side. [`ResidualWriteParams`]
//! carries only what the emitted streams vary: the block size, the component,
//! and the §7.4.9.11 scan order.
//!
//! ## Deriving what the decoder infers
//!
//! Several elements are *inferred* rather than coded — `coded_sub_block_flag`
//! for the DC and last sub-blocks, and the DC `sig_coeff_flag` of a sub-block
//! whose other fifteen flags were all zero. The writer has to reproduce those
//! inferences exactly, because a bin written where the decoder infers one (or
//! the reverse) desynchronizes the whole slice. The walk below mirrors the
//! decoder's `infer_sb_dc_sig` state for that reason rather than deriving the
//! same thing a second, differently-shaped way.

use crate::hevc::engine::binarization::{
    COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN, Greater1State,
    coeff_abs_level_greater2_flag_ctx_inc, coeff_abs_level_remaining_c_max_eq_9_26,
    coeff_abs_level_remaining_c_rice_param_eq_9_24, coded_sub_block_flag_ctx_inc_with_edge,
    last_sig_coeff_position, last_sig_coeff_prefix_cmax, last_sig_coeff_prefix_ctx_inc,
    last_sig_coeff_prefix_ctx_offset_shift, last_sig_coeff_suffix_n_bits,
    sig_coeff_flag_ctx_inc_from_sig_ctx, sig_coeff_flag_sig_ctx_dc,
    sig_coeff_flag_sig_ctx_general, sig_coeff_flag_sig_ctx_log2_2,
};
use crate::hevc::engine::cabac::ContextModel;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::residual::{ResidualContexts, ResidualElement};
use crate::hevc::engine::scan::{ScanIdx, scan_order};

/// A sink for the bins one `residual_coding( )` invocation produces — the
/// write-side counterpart of
/// [`crate::hevc::engine::residual::ResidualBinSource`].
///
/// Tests script a recording sink to compare the `(element, ctxInc)` request
/// sequence against the decoder's; production uses [`EngineResidualBinSink`].
pub(crate) trait ResidualBinSink {
    /// Encode one context-coded bin for `element` at the bank-relative
    /// `ctx_inc` derived per §9.3.4.2.
    fn decision(&mut self, element: ResidualElement, ctx_inc: u32, bin: u8);

    /// Encode one bypass-coded bin (§9.3.4.3.4).
    fn bypass(&mut self, bin: u8);

    /// Encode `n` bypass-coded bins of `value`, MSB-first.
    fn bypass_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.bypass(((value >> i) & 1) as u8);
        }
    }
}

/// The production [`ResidualBinSink`]: context-coded bins go through
/// [`CabacEncoder::encode_decision`] against the matching
/// [`ResidualContexts`] bank slot, bypass bins through
/// [`CabacEncoder::encode_bypass`].
pub(crate) struct EngineResidualBinSink<'w, 'e, 'c> {
    /// The bit writer the arithmetic codeword accumulates into.
    pub writer: &'w mut BitWriter,
    /// The §9.3.5 arithmetic encoding engine.
    pub cabac: &'e mut CabacEncoder,
    /// The residual context banks for the active `initType`.
    pub contexts: &'c mut ResidualContexts,
}

impl ResidualBinSink for EngineResidualBinSink<'_, '_, '_> {
    fn decision(&mut self, element: ResidualElement, ctx_inc: u32, bin: u8) {
        let bank: &mut [ContextModel] = match element {
            ResidualElement::LastSigCoeffXPrefix => &mut self.contexts.last_sig_coeff_x_prefix,
            ResidualElement::LastSigCoeffYPrefix => &mut self.contexts.last_sig_coeff_y_prefix,
            ResidualElement::CodedSubBlockFlag => &mut self.contexts.coded_sub_block_flag,
            ResidualElement::SigCoeffFlag => &mut self.contexts.sig_coeff_flag,
            ResidualElement::CoeffAbsLevelGreater1Flag => {
                &mut self.contexts.coeff_abs_level_greater1_flag
            }
            ResidualElement::CoeffAbsLevelGreater2Flag => {
                &mut self.contexts.coeff_abs_level_greater2_flag
            }
        };
        self.cabac
            .encode_decision(self.writer, &mut bank[ctx_inc as usize], bin);
    }

    fn bypass(&mut self, bin: u8) {
        self.cabac.encode_bypass(self.writer, bin);
    }
}

/// The caller-derived inputs to one `residual_coding( )` invocation the writer
/// needs. The counterpart of
/// [`crate::hevc::engine::residual::ResidualCodingParams`], minus every field
/// that names a feature this writer does not emit (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidualWriteParams {
    /// `log2TrafoSize` of the transform block, 2..=5.
    pub log2_trafo_size: u32,
    /// `cIdx > 0`.
    pub is_chroma: bool,
    /// The §7.4.9.11 scan order
    /// ([`crate::hevc::engine::residual::residual_coding_scan_idx`]).
    pub scan_idx: ScanIdx,
}

/// `true` when `levels` has at least one non-zero coefficient — the `cbf`
/// value the transform-unit syntax codes for this block.
#[must_use]
pub(crate) fn has_coded_levels(levels: &[i32]) -> bool {
    levels.iter().any(|&l| l != 0)
}

/// Write one §7.3.8.11 `residual_coding( )` body for the quantized
/// `TransCoeffLevel` array `levels`, row-major by `y`.
///
/// The caller must already have written the block's `cbf`, and `levels` must
/// contain at least one non-zero coefficient — a `cbf == 0` block codes no
/// `residual_coding( )` at all.
///
/// # Panics
/// Panics if `log2_trafo_size` is outside 2..=5, if `levels` is not
/// `(1 << log2_trafo_size)²` long, or if every level is zero. All three are
/// encoder-internal invariants the transform-unit writer establishes.
pub(crate) fn write_residual_coding<S: ResidualBinSink>(
    sink: &mut S,
    params: &ResidualWriteParams,
    levels: &[i32],
) {
    let log2 = params.log2_trafo_size;
    assert!((2..=5).contains(&log2), "log2TrafoSize outside 2..=5");
    let size = 1usize << log2;
    assert_eq!(levels.len(), size * size, "level block size mismatch");
    let is_chroma = params.is_chroma;
    let scan_idx_num = u32::from(params.scan_idx.index());

    // §6.5 scan tables — the same two the decoder walks.
    let pos_scan = scan_order(2, params.scan_idx).expect("4x4 scan order");
    let sub_scan = scan_order((log2 - 2) as u8, params.scan_idx).expect("sub-block scan order");
    let num_sb_1d = 1usize << (log2 - 2);

    // Locate the last significant coefficient in scan order. The decoder
    // derives (lastSubBlock, lastScanPos) by walking backwards from the coded
    // position until it matches; the writer walks the same scan forwards and
    // keeps the last significant hit, which is the same point.
    let mut last_sub_block: i32 = -1;
    let mut last_scan_pos: i32 = -1;
    for (i, sb) in sub_scan.iter().enumerate() {
        for (n, p) in pos_scan.iter().enumerate() {
            let xc = (usize::from(sb.x) << 2) + usize::from(p.x);
            let yc = (usize::from(sb.y) << 2) + usize::from(p.y);
            if levels[yc * size + xc] != 0 {
                last_sub_block = i as i32;
                last_scan_pos = n as i32;
            }
        }
    }
    assert!(
        last_sub_block >= 0,
        "residual_coding( ) called for an all-zero block"
    );
    let last_sb = sub_scan[last_sub_block as usize];
    let last_pos = pos_scan[last_scan_pos as usize];
    let last_x = ((u32::from(last_sb.x)) << 2) + u32::from(last_pos.x);
    let last_y = ((u32::from(last_sb.y)) << 2) + u32::from(last_pos.y);

    // §7.4.9.11 eq. 7-78: the vertical scan swaps the coded pair.
    let (wire_x, wire_y) = if params.scan_idx == ScanIdx::Vertical {
        (last_y, last_x)
    } else {
        (last_x, last_y)
    };
    let (prefix_x, suffix_x) = last_sig_coeff_binarization(wire_x, log2);
    let (prefix_y, suffix_y) = last_sig_coeff_binarization(wire_y, log2);
    // §7.3.8.11 bin order: both context-coded prefixes, then both bypass
    // suffixes.
    write_last_sig_prefix(
        sink,
        log2,
        is_chroma,
        ResidualElement::LastSigCoeffXPrefix,
        prefix_x,
    );
    write_last_sig_prefix(
        sink,
        log2,
        is_chroma,
        ResidualElement::LastSigCoeffYPrefix,
        prefix_y,
    );
    for (prefix, suffix) in [(prefix_x, suffix_x), (prefix_y, suffix_y)] {
        let n_bits = last_sig_coeff_suffix_n_bits(prefix);
        if n_bits > 0 {
            sink.bypass_bits(suffix, n_bits as u8);
        }
    }

    // coded_sub_block_flag[ xS ][ yS ] grid, row-major by yS.
    let mut csbf = vec![0u8; num_sb_1d * num_sb_1d];
    let csbf_at = |grid: &[u8], xs: usize, ys: usize| -> u8 {
        if xs < num_sb_1d && ys < num_sb_1d {
            grid[ys * num_sb_1d + xs]
        } else {
            0
        }
    };

    let mut g1_state = Greater1State::new();
    let mut last_g1_bin: u8 = 0;

    for i in (0..=last_sub_block).rev() {
        let sb = sub_scan[i as usize];
        let (xs, ys) = (u32::from(sb.x), u32::from(sb.y));
        let is_last_sb = i == last_sub_block;

        // The sub-block's 16 levels in scan order.
        let level_at = |n: usize| -> i32 {
            let xc = (xs as usize) * 4 + usize::from(pos_scan[n].x);
            let yc = (ys as usize) * 4 + usize::from(pos_scan[n].y);
            levels[yc * size + xc]
        };
        let any_nonzero = (0..16).any(|n| level_at(n) != 0);

        // coded_sub_block_flag: coded for 0 < i < lastSubBlock, inferred 1
        // for the DC sub-block and the last significant one.
        let mut infer_sb_dc_sig = false;
        let sb_coded: u8 = if i < last_sub_block && i > 0 {
            let bin = u8::from(any_nonzero);
            let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
            let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
            let ctx_inc =
                coded_sub_block_flag_ctx_inc_with_edge(is_chroma, xs, ys, log2, right, below);
            sink.decision(ResidualElement::CodedSubBlockFlag, ctx_inc, bin);
            infer_sb_dc_sig = true;
            bin
        } else {
            1
        };
        csbf[ys as usize * num_sb_1d + xs as usize] = sb_coded;

        // sig_coeff_flag pass, indexed by in-sub-block scan position.
        let mut sig = [0u8; 16];
        for n in 0..16 {
            sig[n] = u8::from(level_at(n) != 0);
        }
        if is_last_sb {
            // Significant by definition, and not coded.
            debug_assert_eq!(sig[last_scan_pos as usize], 1);
        }
        let start_n: i32 = if is_last_sb { last_scan_pos - 1 } else { 15 };
        for n in (0..=start_n).rev() {
            let n = n as usize;
            let xc = (xs << 2) + u32::from(pos_scan[n].x);
            let yc = (ys << 2) + u32::from(pos_scan[n].y);
            if sb_coded == 1 && (n > 0 || !infer_sb_dc_sig) {
                let sig_ctx = if log2 == 2 {
                    sig_coeff_flag_sig_ctx_log2_2(xc & 3, yc & 3)
                } else if xc + yc == 0 {
                    sig_coeff_flag_sig_ctx_dc(is_chroma, log2, scan_idx_num)
                } else {
                    let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
                    let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
                    sig_coeff_flag_sig_ctx_general(
                        is_chroma,
                        log2,
                        xc,
                        yc,
                        xs,
                        ys,
                        right,
                        below,
                        scan_idx_num,
                    )
                };
                let ctx_inc = sig_coeff_flag_ctx_inc_from_sig_ctx(sig_ctx, is_chroma);
                sink.decision(ResidualElement::SigCoeffFlag, ctx_inc, sig[n]);
                if sig[n] == 1 {
                    infer_sb_dc_sig = false;
                }
            }
            // The remaining branch is the decoder's inference of the DC cell
            // of an explicitly-coded sub-block whose other fifteen flags were
            // zero: nothing is written, and `sig[0]` already holds 1 because
            // the level is non-zero (a coded-1 sub-block has one).
        }
        debug_assert!(
            sb_coded == 0 || !infer_sb_dc_sig || sig[0] == 1,
            "an explicitly coded sub-block with no signalled significance must \
             carry its level at the DC cell"
        );
        if sb_coded == 0 {
            continue;
        }

        // coeff_abs_level_greater1_flag pass, with the per-sub-block
        // numGreater1Flag < 8 cap and the lazy §9.3.4.2.6 entry.
        let mut num_greater1: u32 = 0;
        let mut last_greater1_scan_pos: i32 = -1;
        let mut g1 = [0u8; 16];
        let mut entered_subblock = false;
        for n in (0..16).rev() {
            if sig[n] != 1 {
                continue;
            }
            if num_greater1 < 8 {
                if !entered_subblock {
                    g1_state.on_subblock_entry(i as u32, is_chroma, last_g1_bin);
                    entered_subblock = true;
                }
                let bin = u8::from(level_at(n).unsigned_abs() > 1);
                let ctx_inc = g1_state.current_ctx_inc(is_chroma);
                sink.decision(ResidualElement::CoeffAbsLevelGreater1Flag, ctx_inc, bin);
                g1_state.on_coeff_abs_level_greater1_flag(bin);
                last_g1_bin = bin;
                g1[n] = bin;
                num_greater1 += 1;
                if bin == 1 && last_greater1_scan_pos == -1 {
                    last_greater1_scan_pos = n as i32;
                }
            }
        }

        // coeff_abs_level_greater2_flag — at most once per sub-block.
        let mut g2 = [0u8; 16];
        if last_greater1_scan_pos != -1 {
            let n = last_greater1_scan_pos as usize;
            let bin = u8::from(level_at(n).unsigned_abs() > 2);
            let ctx_inc = coeff_abs_level_greater2_flag_ctx_inc(g1_state.ctx_set(), is_chroma);
            sink.decision(ResidualElement::CoeffAbsLevelGreater2Flag, ctx_inc, bin);
            g2[n] = bin;
        }

        // coeff_sign_flag pass (bypass): one per significant position, since
        // sign data hiding is off in every stream this writer emits.
        for n in (0..16).rev() {
            if sig[n] == 1 {
                sink.bypass(u8::from(level_at(n) < 0));
            }
        }

        // Level pass: coeff_abs_level_remaining with the §9.3.3.11 eq.-9-24
        // Rice adaptation.
        let mut num_sig_coeff: u32 = 0;
        let mut c_last_abs_level: u32 = 0;
        let mut c_last_rice_param: u32 = 0;
        for n in (0..16).rev() {
            if sig[n] != 1 {
                continue;
            }
            let abs_level = level_at(n).unsigned_abs();
            let base_level = 1 + u32::from(g1[n]) + u32::from(g2[n]);
            let threshold = if num_sig_coeff < 8 {
                if n as i32 == last_greater1_scan_pos {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            if base_level == threshold {
                let c_rice_param = coeff_abs_level_remaining_c_rice_param_eq_9_24(
                    c_last_abs_level,
                    c_last_rice_param,
                );
                let remaining = abs_level - base_level;
                write_coeff_abs_level_remaining(sink, remaining, c_rice_param);
                c_last_abs_level = base_level + remaining;
                c_last_rice_param = c_rice_param;
            } else {
                debug_assert_eq!(
                    abs_level, base_level,
                    "a level above its threshold must carry a remainder"
                );
            }
            num_sig_coeff += 1;
        }
    }
}

/// The §7.4.9.11 `last_sig_coeff_*` binarization of one position: the
/// `(prefix, suffix)` pair whose [`last_sig_coeff_position`] is `position`.
fn last_sig_coeff_binarization(position: u32, log2_trafo_size: u32) -> (u32, u32) {
    if position <= 3 {
        return (position, 0);
    }
    // Prefixes above 3 each cover a power-of-two window; take the last one
    // whose base does not exceed the position.
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let mut prefix = 4;
    while prefix < c_max && last_sig_coeff_position(prefix + 1, Some(0)) <= position {
        prefix += 1;
    }
    let base = last_sig_coeff_position(prefix, Some(0));
    debug_assert!(position >= base);
    (prefix, position - base)
}

/// The §9.3.3.10 truncated-Rice prefix bins of one `last_sig_coeff_*_prefix`,
/// each with its §9.3.4.2.3 `ctxInc`.
fn write_last_sig_prefix<S: ResidualBinSink>(
    sink: &mut S,
    log2_trafo_size: u32,
    is_chroma: bool,
    element: ResidualElement,
    prefix: u32,
) {
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let (ctx_offset, ctx_shift) =
        last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size, is_chroma);
    for bin_idx in 0..prefix {
        let ctx_inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_offset, ctx_shift);
        sink.decision(element, ctx_inc, 1);
    }
    // The terminating 0 is absent when the value saturates cMax.
    if prefix < c_max {
        let ctx_inc = last_sig_coeff_prefix_ctx_inc(prefix, ctx_offset, ctx_shift);
        sink.decision(element, ctx_inc, 0);
    }
}

/// §9.3.3.11 — the bypass bin string for one `coeff_abs_level_remaining[ n ]`
/// value at the derived `cRiceParam`: a truncated-Rice prefix, and on the
/// all-ones escape the EGk suffix with `k = cRiceParam + 1`.
fn write_coeff_abs_level_remaining<S: ResidualBinSink>(
    sink: &mut S,
    value: u32,
    c_rice_param: u32,
) {
    let c_max = coeff_abs_level_remaining_c_max_eq_9_26(c_rice_param);
    if value < c_max {
        // §9.3.3.2 TR: a unary prefix of `value >> cRiceParam` ones, the
        // terminating zero, then the `cRiceParam`-bit remainder.
        let prefix_len = value >> c_rice_param;
        debug_assert!(prefix_len < COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN);
        for _ in 0..prefix_len {
            sink.bypass(1);
        }
        sink.bypass(0);
        if c_rice_param > 0 {
            sink.bypass_bits(value & ((1 << c_rice_param) - 1), c_rice_param as u8);
        }
        return;
    }
    // Escape: the all-ones TR prefix, then §9.3.3.3 EGk.
    for _ in 0..COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN {
        sink.bypass(1);
    }
    let k = c_rice_param + 1;
    let rest = u64::from(value - c_max);
    let mut prefix_ones: u32 = 0;
    while rest >= (((1u64 << (prefix_ones + 1)) - 1) << k) {
        prefix_ones += 1;
    }
    for _ in 0..prefix_ones {
        sink.bypass(1);
    }
    sink.bypass(0);
    let suffix = rest - (((1u64 << prefix_ones) - 1) << k);
    sink.bypass_bits(suffix as u32, (prefix_ones + k) as u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;
    use crate::hevc::engine::cabac::CabacEngine;
    use crate::hevc::engine::residual::{
        EngineResidualBinSource, ResidualCodingParams, decode_residual_coding_with,
    };

    /// A quantized level block with the awkward shapes: sign changes, levels
    /// that escape past greater1 and greater2, isolated sub-blocks, and a
    /// sub-block whose only significant coefficient sits at its DC cell (the
    /// decoder's `inferSbDcSigCoeffFlag` inference).
    fn levels(log2: u32) -> Vec<i32> {
        let size = 1usize << log2;
        let mut v = vec![0i32; size * size];
        v[0] = 7;
        v[1] = -1;
        v[size] = 2;
        v[size + 1] = -3;
        if size >= 8 {
            // A sub-block reached only through its DC cell.
            v[4 * size + 4] = -1;
            // A level well past the escape into the EGk suffix.
            v[2 * size + 3] = 900;
            v[size + 5] = -46;
        }
        if size >= 16 {
            v[9 * size + 10] = 4;
            v[15 * size + 15] = -2;
        }
        v
    }

    /// Encode a level block, then decode it back through the decoder's own
    /// §7.3.8.11 driver and require the levels — and the derived
    /// last-significant position — to come back identical.
    fn roundtrip(log2: u32, is_chroma: bool, scan_idx: ScanIdx, levels: &[i32]) {
        let write = ResidualWriteParams {
            log2_trafo_size: log2,
            is_chroma,
            scan_idx,
        };
        let mut writer = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut enc_ctx = ResidualContexts::init(0, 26);
        write_residual_coding(
            &mut EngineResidualBinSink {
                writer: &mut writer,
                cabac: &mut cabac,
                contexts: &mut enc_ctx,
            },
            &write,
            levels,
        );
        cabac.encode_terminate(&mut writer, 1);
        writer.align_zero();
        let bytes = writer.finish();

        let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("engine init");
        let mut dec_ctx = ResidualContexts::init(0, 26);
        let block = decode_residual_coding_with(
            &ResidualCodingParams {
                log2_trafo_size: log2,
                is_chroma,
                scan_idx,
                sign_data_hiding_enabled_flag: false,
                sign_hidden_suppressed: false,
                transform_skip_sig_ctx: false,
                persistent_rice_adaptation_enabled_flag: false,
                cabac_bypass_alignment_enabled_flag: false,
                extended_precision_processing_flag: false,
                bit_depth: 8,
                rice_stat_transform_skip: false,
            },
            &mut EngineResidualBinSource {
                engine: &mut engine,
                contexts: &mut dec_ctx,
            },
        )
        .expect("decode");
        assert_eq!(block.levels, levels, "log2 {log2} chroma {is_chroma}");
        // The two sides must also have walked the context banks identically,
        // or the next transform block in the slice would desynchronize.
        assert_eq!(enc_ctx, dec_ctx, "context state diverged");
        assert_eq!(engine.decode_terminate().unwrap(), 1, "terminator");
    }

    #[test]
    fn every_block_size_roundtrips_through_the_decoder() {
        for log2 in 2..=5u32 {
            roundtrip(log2, false, ScanIdx::Diagonal, &levels(log2));
        }
    }

    #[test]
    fn chroma_blocks_roundtrip_through_the_decoder() {
        for log2 in 2..=4u32 {
            roundtrip(log2, true, ScanIdx::Diagonal, &levels(log2));
        }
    }

    #[test]
    fn the_mode_dependent_scans_roundtrip_through_the_decoder() {
        // The horizontal and vertical scans reorder the whole walk and, for
        // the vertical one, swap the coded last-significant pair (eq. 7-78).
        for scan in [ScanIdx::Horizontal, ScanIdx::Vertical] {
            roundtrip(2, false, scan, &levels(2));
            roundtrip(3, true, scan, &levels(3));
        }
    }

    #[test]
    fn a_single_dc_coefficient_roundtrips() {
        for log2 in 2..=5u32 {
            let size = 1usize << log2;
            let mut v = vec![0i32; size * size];
            v[0] = -1;
            roundtrip(log2, false, ScanIdx::Diagonal, &v);
        }
    }

    #[test]
    fn a_single_coefficient_in_the_far_corner_roundtrips() {
        // The widest last_sig_coeff prefix + suffix binarization.
        for log2 in 2..=5u32 {
            let size = 1usize << log2;
            let mut v = vec![0i32; size * size];
            v[size * size - 1] = 3;
            roundtrip(log2, false, ScanIdx::Diagonal, &v);
        }
    }

    #[test]
    fn a_dense_block_of_large_levels_roundtrips() {
        // Every position significant drives the numGreater1Flag < 8 cap, the
        // per-sub-block Rice adaptation, and long escape suffixes.
        let log2 = 4u32;
        let size = 1usize << log2;
        let v: Vec<i32> = (0..size * size)
            .map(|i| {
                let magnitude = 1 + (i as i32 * 37) % 400;
                if i % 3 == 0 { -magnitude } else { magnitude }
            })
            .collect();
        roundtrip(log2, false, ScanIdx::Diagonal, &v);
    }

    #[test]
    fn every_last_significant_position_binarizes_back_to_itself() {
        for log2 in 2..=5u32 {
            for position in 0..(1u32 << log2) {
                let (prefix, suffix) = last_sig_coeff_binarization(position, log2);
                assert!(prefix <= last_sig_coeff_prefix_cmax(log2));
                let suffix = (last_sig_coeff_suffix_n_bits(prefix) > 0).then_some(suffix);
                assert_eq!(
                    last_sig_coeff_position(prefix, suffix),
                    position,
                    "log2 {log2} position {position}"
                );
            }
        }
    }
}
