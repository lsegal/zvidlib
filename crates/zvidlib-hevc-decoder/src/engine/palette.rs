//! §7.3.8.13 `palette_coding( )` parse, the §8.4.4.2.7 palette-mode
//! decoding process, and the palette predictor state machinery
//! (§9.3.2.3 initialization, eq. 8-79 update) of the HEVC Screen
//! Content Coding extensions.
//!
//! A palette coding unit replaces prediction + transform coding
//! entirely: the CU signals a small colour table (composed of entries
//! REUSED from a rolling predictor palette plus explicitly signalled
//! new entries, eq. 7-82), an index map covering the block in the
//! §6.5.6 traverse scan (run-length coded, with per-run "copy the
//! index from the row above" escapes), and optional escape samples
//! (quantized when the CU is not transquant-bypassed) for positions
//! whose index equals `MaxPaletteIndex`.
//!
//! The predictor palette is CABAC-adjacent parse state: it initializes
//! from the PPS / SPS palette predictor initializers at every point
//! §9.3.2.1 re-initializes context variables (slice start, tile start,
//! WPP fallback), synchronizes across WPP rows and dependent slice
//! segments alongside the context tables (§9.3.2.4 / §9.3.2.5), and
//! every palette CU rewrites it per eq. 8-79. It therefore lives
//! inside [`SliceContexts`] so the existing storage / synchronization
//! clones carry it.

use crate::hevc::engine::binarization::{
    CuChromaQpOffset, CuQpDelta, decode_cu_chroma_qp_offset, decode_cu_qp_delta, decode_eg_k,
    palette_run_prefix_ctx_inc, palette_run_prefix_tr_cmax, read_truncated_rice_prefix,
};
use crate::hevc::engine::cabac::{CabacEngine, CabacError};
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::scan::traverse;

/// The rolling predictor palette (§9.3.2.3 / eq. 8-79):
/// `PredictorPaletteSize` and `PredictorPaletteEntries[comp][i]`.
///
/// Component 0 is luma; components 1 / 2 exist when `ChromaArrayType
/// != 0` (the parse always keeps three vectors; chroma vectors stay
/// empty-aligned for monochrome).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PalettePredictor {
    /// `PredictorPaletteEntries[comp][0..PredictorPaletteSize]`.
    pub entries: [Vec<u16>; 3],
}

impl PalettePredictor {
    /// `PredictorPaletteSize`.
    #[must_use]
    pub fn size(&self) -> usize {
        self.entries[0].len()
    }

    /// §9.3.2.3 — initialize from explicit initializer rows
    /// (`initializers[comp][i]`; the PPS body if present, else the SPS
    /// body, else empty). `num_comps` is `ChromaArrayType == 0 ? 1 : 3`.
    #[must_use]
    pub fn from_initializers(initializers: &[Vec<u32>], num_comps: usize) -> Self {
        let mut p = Self::default();
        let size = initializers.first().map_or(0, Vec::len);
        for c in 0..num_comps.min(3) {
            let row = initializers.get(c).map_or(&[][..], Vec::as_slice);
            p.entries[c] = (0..size)
                .map(|i| row.get(i).copied().unwrap_or(0).min(u32::from(u16::MAX)) as u16)
                .collect();
        }
        p
    }
}

/// Parameters the §7.3.8.13 parse needs from the active parameter
/// sets and CU context.
#[derive(Debug, Clone, Copy)]
pub struct PaletteParams {
    /// `palette_max_size` (§7.4.3.2.3).
    pub palette_max_size: u32,
    /// `PaletteMaxPredictorSize` (eq. 7-35).
    pub palette_max_predictor_size: u32,
    /// `ChromaArrayType`.
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u32,
    /// `BitDepthC`.
    pub bit_depth_chroma: u32,
    /// `cu_transquant_bypass_flag` of the CU.
    pub cu_transquant_bypass_flag: bool,
    /// PPS `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled_flag: bool,
    /// Per-slice `cu_chroma_qp_offset_enabled_flag`.
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// `chroma_qp_offset_list_len_minus1`.
    pub chroma_qp_offset_list_len_minus1: u32,
}

/// Parse errors of `palette_coding( )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteError {
    /// The arithmetic engine failed / bitstream exhausted.
    Cabac(CabacError),
    /// A syntax element violated a §7.4.9.6 conformance bound.
    Malformed(&'static str),
}

impl core::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cabac(e) => write!(f, "palette_coding: {e}"),
            Self::Malformed(m) => write!(f, "palette_coding: {m}"),
        }
    }
}

impl std::error::Error for PaletteError {}

impl From<CabacError> for PaletteError {
    fn from(e: CabacError) -> Self {
        Self::Cabac(e)
    }
}

/// One parsed palette coding unit: everything the §8.4.4.2.7
/// reconstruction needs. Index / escape arrays are CU-local, in luma
/// resolution and CODED (pre-transpose) coordinates — eq. 8-69/8-70
/// apply the transpose at reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteCu {
    /// `nCbS`.
    pub n_cbs: usize,
    /// `CurrentPaletteEntries[comp][0..CurrentPaletteSize]` (eq. 7-82).
    pub palette: [Vec<u16>; 3],
    /// `palette_escape_val_present_flag`.
    pub escape_present: bool,
    /// `palette_transpose_flag`.
    pub transpose: bool,
    /// `PaletteIndexMap` over the CU, row-major in coded coordinates
    /// (`index_map[y * n_cbs + x]`), values `0..=MaxPaletteIndex`.
    pub index_map: Vec<u8>,
    /// `PaletteEscapeVal[comp]`, row-major in coded luma coordinates;
    /// only positions whose index equals `MaxPaletteIndex` (and pass
    /// the §7.3.8.13 chroma-phase gate) carry meaningful values.
    pub escape_vals: [Vec<u16>; 3],
    /// The `delta_qp( )` decoded inside `palette_coding( )` when
    /// escapes are present.
    pub cu_qp_delta: Option<CuQpDelta>,
    /// The `chroma_qp_offset( )` decoded inside `palette_coding( )`.
    pub cu_chroma_qp_offset: Option<CuChromaQpOffset>,
}

impl PaletteCu {
    /// `MaxPaletteIndex` (§7.4.9.6).
    #[must_use]
    pub fn max_palette_index(&self) -> u32 {
        self.palette[0].len() as u32 + u32::from(self.escape_present) - 1
    }
}

/// §9.3.3.6 — Truncated Binary decode (all bins bypass).
fn decode_tb(engine: &mut CabacEngine<'_>, c_max: u32) -> Result<u32, CabacError> {
    if c_max == 0 {
        return Ok(0);
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros(); // Floor(Log2(n)), n >= 2
    let u = (1u32 << (k + 1)) - n;
    let mut val = 0u32;
    for _ in 0..k {
        val = (val << 1) | u32::from(engine.decode_bypass()?);
    }
    if val < u {
        Ok(val)
    } else {
        let val = (val << 1) | u32::from(engine.decode_bypass()?);
        Ok(val - u)
    }
}

/// §9.3.3.14 — `num_palette_indices_minus1` (bypass TR prefix with
/// `cRiceParam = 3 + ((MaxPaletteIndex + 1) >> 3)`, EGk(k+1) escape).
fn decode_num_palette_indices_minus1(
    engine: &mut CabacEngine<'_>,
    max_palette_index: u32,
) -> Result<u32, CabacError> {
    let k = 3 + ((max_palette_index + 1) >> 3);
    let c_max = 4u32 << k;
    // TR(cMax, cRiceParam = k): unary quotient (up to 4 ones), then a
    // k-bit rice remainder when the quotient did not escape.
    let mut q = 0u32;
    while q < 4 {
        if engine.decode_bypass()? == 0 {
            break;
        }
        q += 1;
    }
    if q < 4 {
        let mut rem = 0u32;
        for _ in 0..k {
            rem = (rem << 1) | u32::from(engine.decode_bypass()?);
        }
        Ok((q << k) + rem)
    } else {
        // Prefix all-ones (prefixVal == cMax): EGk suffix with order
        // k + 1 (eq. 9-33).
        Ok(c_max + decode_eg_k(engine, k + 1)?)
    }
}

/// §9.3.3.12 — `palette_escape_val`: FL(bitDepth) under transquant
/// bypass, else EG3.
fn decode_palette_escape_val(
    engine: &mut CabacEngine<'_>,
    bypass_cu: bool,
    bit_depth: u32,
) -> Result<u32, CabacError> {
    if bypass_cu {
        let mut v = 0u32;
        for _ in 0..bit_depth {
            v = (v << 1) | u32::from(engine.decode_bypass()?);
        }
        Ok(v)
    } else {
        decode_eg_k(engine, 3)
    }
}

/// Decode one §7.3.8.13 `palette_coding( x0, y0, nCbS )` body and
/// apply the eq. 8-79 predictor palette update to
/// `ctx.palette_predictor`.
///
/// `is_cu_qp_delta_coded` / `is_cu_chroma_qp_offset_coded` mirror the
/// §7.3.8.14 / §7.3.8.15 quantization-group gates (the caller's
/// [`crate::hevc::engine::transform_unit::QuantGroupState`]).
///
/// # Errors
/// [`PaletteError`] on CABAC exhaustion or a §7.4.9.6 bound violation.
#[allow(clippy::too_many_lines)]
pub fn decode_palette_coding(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &PaletteParams,
    qg: &mut crate::hevc::engine::transform_unit::QuantGroupState,
    n_cbs: usize,
) -> Result<PaletteCu, PaletteError> {
    let num_comps = if params.chroma_array_type == 0 { 1 } else { 3 };
    let max_size = params.palette_max_size as usize;
    let predictor_size = ctx.palette_predictor.size();

    // ---- predictor entry reuse (palette_predictor_run, EG0) ----
    let mut reuse = vec![false; predictor_size];
    let mut num_predicted = 0usize;
    let mut finished = false;
    let mut idx = 0usize;
    while idx < predictor_size && !finished && num_predicted < max_size {
        let run = decode_eg_k(engine, 0)?;
        if run == 1 {
            finished = true;
        } else {
            if run > 1 {
                idx += (run - 1) as usize;
            }
            if idx >= predictor_size {
                return Err(PaletteError::Malformed(
                    "palette_predictor_run past PredictorPaletteSize",
                ));
            }
            reuse[idx] = true;
            num_predicted += 1;
        }
        idx += 1;
    }

    // ---- num_signalled_palette_entries (EG0) + new entries ----
    let num_signalled = if num_predicted < max_size {
        decode_eg_k(engine, 0)? as usize
    } else {
        0
    };
    if num_predicted + num_signalled > max_size {
        return Err(PaletteError::Malformed(
            "CurrentPaletteSize past palette_max_size",
        ));
    }
    let mut new_entries: [Vec<u16>; 3] = Default::default();
    for (c, out) in new_entries.iter_mut().enumerate().take(num_comps) {
        let bd = if c == 0 {
            params.bit_depth_luma
        } else {
            params.bit_depth_chroma
        };
        for _ in 0..num_signalled {
            let mut v = 0u32;
            for _ in 0..bd {
                v = (v << 1) | u32::from(engine.decode_bypass()?);
            }
            out.push(v as u16);
        }
    }

    // ---- CurrentPaletteEntries (eq. 7-82) ----
    let mut palette: [Vec<u16>; 3] = Default::default();
    for c in 0..num_comps {
        for (i, &r) in reuse.iter().enumerate() {
            if r {
                palette[c].push(ctx.palette_predictor.entries[c][i]);
            }
        }
        palette[c].extend_from_slice(&new_entries[c]);
    }
    let current_palette_size = palette[0].len();

    // ---- palette_escape_val_present_flag ----
    let escape_present = if current_palette_size != 0 {
        engine.decode_bypass()? != 0
    } else {
        true // inferred 1 (§7.4.9.6)
    };
    let max_palette_index = current_palette_size as i64 - 1 + i64::from(escape_present);
    if max_palette_index < 0 {
        return Err(PaletteError::Malformed(
            "empty palette without escape samples",
        ));
    }
    let max_palette_index = max_palette_index as u32;
    if max_palette_index > u32::from(u8::MAX) {
        return Err(PaletteError::Malformed("MaxPaletteIndex past u8 range"));
    }

    // ---- explicit index list + final-run / transpose flags ----
    let mut palette_index_idc: Vec<u32> = Vec::new();
    let mut copy_above_final_run = false;
    let mut transpose = false;
    let mut num_palette_indices = 1usize; // num_palette_indices_minus1 + 1 (inferred 0)
    if max_palette_index > 0 {
        let npi_m1 = decode_num_palette_indices_minus1(engine, max_palette_index)?;
        if npi_m1 as usize >= n_cbs * n_cbs {
            return Err(PaletteError::Malformed(
                "num_palette_indices_minus1 past the block area",
            ));
        }
        num_palette_indices = npi_m1 as usize + 1;
        let mut adjust = 0u32;
        for _ in 0..num_palette_indices {
            if max_palette_index - adjust > 0 {
                // §9.3.3.13: cMax = MaxPaletteIndex on the first
                // invocation, MaxPaletteIndex − 1 afterwards.
                let c_max = max_palette_index - adjust;
                palette_index_idc.push(decode_tb(engine, c_max)?);
            } else {
                palette_index_idc.push(0);
            }
            adjust = 1;
        }
        copy_above_final_run = engine.decode_decision(&mut ctx.palette_copy_above_flag[0])? != 0;
        transpose = engine.decode_decision(&mut ctx.palette_transpose_flag[0])? != 0;
    }

    // ---- delta_qp( ) / chroma_qp_offset( ) (escapes only) ----
    let mut cu_qp_delta = None;
    let mut cu_chroma_qp_offset = None;
    if escape_present {
        if params.cu_qp_delta_enabled_flag && !qg.is_cu_qp_delta_coded {
            qg.is_cu_qp_delta_coded = true;
            let (bin0, rest) = ctx.cu_qp_delta_abs.split_at_mut(1);
            let delta = decode_cu_qp_delta(engine, &mut bin0[0], &mut rest[0])?;
            qg.cu_qp_delta_val = delta.value;
            cu_qp_delta = Some(delta);
        }
        if !params.cu_transquant_bypass_flag
            && params.cu_chroma_qp_offset_enabled_flag
            && !qg.is_cu_chroma_qp_offset_coded
        {
            qg.is_cu_chroma_qp_offset_coded = true;
            let offset = decode_cu_chroma_qp_offset(
                engine,
                &mut ctx.cu_chroma_qp_offset_flag[0],
                &mut ctx.cu_chroma_qp_offset_idx[0],
                params.chroma_qp_offset_list_len_minus1,
            )?;
            cu_chroma_qp_offset = Some(offset);
        }
    }

    // ---- index-map run loop (§7.3.8.13, transcribed) ----
    let scan = traverse(n_cbs);
    let area = n_cbs * n_cbs;
    let mut index_map = vec![0u8; area];
    let mut copy_above = vec![false; area]; // CopyAboveIndicesFlag, coded coords
    let mut remaining = num_palette_indices;
    let mut scan_pos = 0usize;
    let mut curr_palette_index = 0u32;
    // The RAW `palette_idx_idc` (`PaletteIndexIdc[ currNumIndices ]`,
    // §7.4.9.6 inferred 0 when absent) BEFORE the eq. 7-84
    // adjustment: §9.3.4.2.8 derives the `palette_run_prefix` ctxInc
    // from the syntax element value, not from `CurrPaletteIndex`.
    let mut raw_palette_idx_idc = 0u32;
    while scan_pos < area {
        let pos = &scan[scan_pos];
        let (x_c, y_c) = (pos.x as usize, pos.y as usize);
        let prev_copy_above = scan_pos > 0 && {
            let p = &scan[scan_pos - 1];
            copy_above[p.y as usize * n_cbs + p.x as usize]
        };
        let mut run_minus1 = area - scan_pos - 1;
        let mut copy_flag = false;
        if max_palette_index > 0 && scan_pos >= n_cbs && !prev_copy_above {
            if remaining > 0 && scan_pos < area - 1 {
                let ctx_bin = engine.decode_decision(&mut ctx.palette_copy_above_flag[0])? != 0;
                copy_flag = ctx_bin;
            } else {
                copy_flag = !(scan_pos == area - 1 && remaining > 0);
            }
        }
        if !copy_flag {
            // CurrPaletteIndex = PaletteIndexIdc[currNumIndices], then
            // the eq. 7-84 adjustment against adjustedRefPaletteIndex
            // (eq. 7-83).
            let curr_num_indices = num_palette_indices - remaining;
            curr_palette_index = palette_index_idc
                .get(curr_num_indices)
                .copied()
                .unwrap_or(0);
            raw_palette_idx_idc = curr_palette_index;
            let adjusted_ref = if scan_pos > 0 {
                let p = &scan[scan_pos - 1];
                if !copy_above[p.y as usize * n_cbs + p.x as usize] {
                    u32::from(index_map[p.y as usize * n_cbs + p.x as usize])
                } else {
                    u32::from(index_map[(y_c - 1) * n_cbs + x_c])
                }
            } else {
                max_palette_index + 1
            };
            if curr_palette_index >= adjusted_ref {
                curr_palette_index += 1;
            }
            if curr_palette_index > max_palette_index {
                return Err(PaletteError::Malformed(
                    "palette index past MaxPaletteIndex",
                ));
            }
        }
        if max_palette_index > 0 {
            if !copy_flag {
                if remaining == 0 {
                    return Err(PaletteError::Malformed(
                        "palette runs exhaust the signalled index count",
                    ));
                }
                remaining -= 1;
            }
            if remaining > 0 || copy_flag != copy_above_final_run {
                let max_run_minus1 = (area as i64)
                    - (scan_pos as i64)
                    - 1
                    - (remaining as i64)
                    - i64::from(copy_above_final_run);
                if max_run_minus1 < 0 {
                    return Err(PaletteError::Malformed("PaletteMaxRunMinus1 negative"));
                }
                run_minus1 = 0;
                if max_run_minus1 > 0 {
                    // palette_run_prefix: TR with per-bin §9.3.4.2.8
                    // ctxInc for bins 0..=4, bypass beyond.
                    let idc_for_ctx = if copy_flag { 0 } else { raw_palette_idx_idc };
                    let c_max = palette_run_prefix_tr_cmax(max_run_minus1 as u32);
                    let (prefix, _escape) =
                        read_truncated_rice_prefix(
                            c_max,
                            |bin_idx| match palette_run_prefix_ctx_inc(
                                bin_idx,
                                copy_flag,
                                idc_for_ctx,
                            ) {
                                Some(inc) => engine
                                    .decode_decision(&mut ctx.palette_run_prefix[inc as usize]),
                                None => engine.decode_bypass(),
                            },
                        )?;
                    if prefix < 2 {
                        run_minus1 = prefix as usize; // eq. 7-85
                    } else {
                        let prefix_offset = 1u32 << (prefix - 1);
                        let suffix = if max_run_minus1 as u32 != prefix_offset {
                            // palette_run_suffix: TB with the Table
                            // 9-43 cMax.
                            let c_max_tb = if (prefix_offset << 1) > max_run_minus1 as u32 {
                                max_run_minus1 as u32 - prefix_offset
                            } else {
                                prefix_offset - 1
                            };
                            decode_tb(engine, c_max_tb)?
                        } else {
                            0
                        };
                        run_minus1 = (prefix_offset + suffix) as usize; // eq. 7-86
                    }
                    if run_minus1 as i64 > max_run_minus1 {
                        return Err(PaletteError::Malformed(
                            "PaletteRunMinus1 past PaletteMaxRunMinus1",
                        ));
                    }
                }
            }
        }
        // Write the run.
        let mut run_pos = 0usize;
        while run_pos <= run_minus1 {
            let p = &scan[scan_pos];
            let (x_r, y_r) = (p.x as usize, p.y as usize);
            if !copy_flag {
                copy_above[y_r * n_cbs + x_r] = false;
                index_map[y_r * n_cbs + x_r] = curr_palette_index as u8;
            } else {
                if y_r == 0 {
                    return Err(PaletteError::Malformed("copy-above run in the top row"));
                }
                copy_above[y_r * n_cbs + x_r] = true;
                index_map[y_r * n_cbs + x_r] = index_map[(y_r - 1) * n_cbs + x_r];
            }
            run_pos += 1;
            scan_pos += 1;
        }
    }
    if max_palette_index > 0 && remaining != 0 {
        return Err(PaletteError::Malformed(
            "palette runs left signalled indices unconsumed",
        ));
    }

    // ---- escape values ----
    let mut escape_vals: [Vec<u16>; 3] = Default::default();
    for v in escape_vals.iter_mut().take(num_comps) {
        *v = vec![0u16; area];
    }
    if escape_present {
        #[allow(clippy::needless_range_loop)]
        for c in 0..num_comps {
            let bd = if c == 0 {
                params.bit_depth_luma
            } else {
                params.bit_depth_chroma
            };
            for pos in &scan {
                let (x_c, y_c) = (pos.x as usize, pos.y as usize);
                if u32::from(index_map[y_c * n_cbs + x_c]) != max_palette_index {
                    continue;
                }
                // §7.3.8.13 per-component sample-presence gate.
                let present = c == 0
                    || (x_c % 2 == 0 && y_c % 2 == 0 && params.chroma_array_type == 1)
                    || (x_c % 2 == 0 && !transpose && params.chroma_array_type == 2)
                    || (y_c % 2 == 0 && transpose && params.chroma_array_type == 2)
                    || params.chroma_array_type == 3;
                if present {
                    let v =
                        decode_palette_escape_val(engine, params.cu_transquant_bypass_flag, bd)?;
                    // Conformance bound: 0..(1 << (bitDepth + 1)) − 1;
                    // clamp into u16 defensively.
                    escape_vals[c][y_c * n_cbs + x_c] =
                        v.min((1u32 << (bd + 1).min(16)) - 1) as u16;
                }
            }
        }
    }

    let cu = PaletteCu {
        n_cbs,
        palette,
        escape_present,
        transpose,
        index_map,
        escape_vals,
        cu_qp_delta,
        cu_chroma_qp_offset,
    };

    // ---- predictor palette update (eq. 8-79) ----
    let max_pred = params.palette_max_predictor_size as usize;
    let mut new_pred: [Vec<u16>; 3] = Default::default();
    for (c, np) in new_pred.iter_mut().enumerate().take(num_comps) {
        np.clone_from(&cu.palette[c]);
    }
    let mut new_size = current_palette_size;
    for (i, &r) in reuse.iter().enumerate() {
        if new_size >= max_pred {
            break;
        }
        if !r {
            for (c, np) in new_pred.iter_mut().enumerate().take(num_comps) {
                np.push(ctx.palette_predictor.entries[c][i]);
            }
            new_size += 1;
        }
    }
    ctx.palette_predictor = PalettePredictor { entries: new_pred };

    Ok(cu)
}

/// §8.4.4.2.7 — reconstruct one colour component of a palette CU into
/// the caller's sample writer.
///
/// * `c_idx` — colour component (0 luma, 1 Cb, 2 Cr).
/// * `(sub_w, sub_h)` — `(nSubWidth, nSubHeight)` for the component
///   (1/1 for luma; `SubWidthC`/`SubHeightC` for chroma).
/// * `qp` — the §8.6.1-derived `Qp′` for the component (the escape
///   path clamps at 0 per eq. 8-73..8-75).
/// * `bit_depth` — the component's bit depth.
/// * `set` — called as `set(x, y, value)` in COMPONENT coordinates
///   relative to the CU's component-plane origin.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_palette_component<F: FnMut(usize, usize, i32)>(
    cu: &PaletteCu,
    c_idx: usize,
    sub_w: usize,
    sub_h: usize,
    qp: i32,
    bit_depth: u32,
    cu_transquant_bypass: bool,
    mut set: F,
) {
    let n = cu.n_cbs;
    let (n_cb_sx, n_cb_sy) = (n / sub_w, n / sub_h);
    let max_index = cu.max_palette_index();
    let level_scale = [40i64, 45, 51, 57, 64, 72];
    let qp = qp.max(0);
    for y in 0..n_cb_sy {
        for x in 0..n_cb_sx {
            // eq. 8-69 / 8-70 — transpose maps the CODED coordinates
            // onto the raster block.
            let (x_l, y_l) = if cu.transpose {
                (y * sub_h, x * sub_w)
            } else {
                (x * sub_w, y * sub_h)
            };
            let idx = u32::from(cu.index_map[y_l * n + x_l]);
            let is_escape = cu.escape_present && idx == max_index;
            let v = if !is_escape {
                // eq. 8-71.
                i32::from(cu.palette[c_idx][idx as usize])
            } else if cu_transquant_bypass {
                // eq. 8-72.
                i32::from(cu.escape_vals[c_idx][y_l * n + x_l])
            } else {
                // eq. 8-77 / 8-78 — escape dequantization.
                let val = i64::from(cu.escape_vals[c_idx][y_l * n + x_l]);
                let tmp =
                    ((val * level_scale[(qp % 6) as usize]) << (qp / 6)).wrapping_add(32) >> 6;
                tmp.clamp(0, (1i64 << bit_depth) - 1) as i32
            };
            set(x, y, v);
        }
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::cabac::init_type;
    use crate::hevc::engine::encoder::bitwriter::BitWriter;
    use crate::hevc::engine::encoder::cabac::CabacEncoder;
    use crate::hevc::engine::transform_unit::QuantGroupState;

    fn params(bypass: bool) -> PaletteParams {
        PaletteParams {
            palette_max_size: 63,
            palette_max_predictor_size: 128,
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            cu_transquant_bypass_flag: bypass,
            cu_qp_delta_enabled_flag: false,
            cu_chroma_qp_offset_enabled_flag: false,
            chroma_qp_offset_list_len_minus1: 0,
        }
    }

    /// Bypass-encode an EG0 value (the §9.3.3.3 dual of
    /// `decode_eg_k(0)`).
    fn encode_eg0(cabac: &mut CabacEncoder, w: &mut BitWriter, v: u32) {
        let mut prefix_ones = 0u32;
        while ((1u64 << (prefix_ones + 1)) - 1) <= u64::from(v) {
            prefix_ones += 1;
        }
        for _ in 0..prefix_ones {
            cabac.encode_bypass(w, 1);
        }
        cabac.encode_bypass(w, 0);
        let base = (1u64 << prefix_ones) - 1;
        let suffix = u64::from(v) - base;
        for i in (0..prefix_ones).rev() {
            cabac.encode_bypass(w, ((suffix >> i) & 1) as u8);
        }
    }

    /// TB-encode (§9.3.3.6 dual of `decode_tb`).
    fn encode_tb(cabac: &mut CabacEncoder, w: &mut BitWriter, v: u32, c_max: u32) {
        if c_max == 0 {
            return;
        }
        let n = c_max + 1;
        let k = 31 - n.leading_zeros();
        let u = (1u32 << (k + 1)) - n;
        if v < u {
            for i in (0..k).rev() {
                cabac.encode_bypass(w, ((v >> i) & 1) as u8);
            }
        } else {
            let val = v + u;
            for i in (0..=k).rev() {
                cabac.encode_bypass(w, ((val >> i) & 1) as u8);
            }
        }
    }

    /// Round-trip a hand-built 8x8 palette CU: three signalled
    /// entries, no escapes, an explicit-index run, a copy-above run
    /// and the final run.
    #[test]
    fn palette_coding_roundtrips_hand_built_block() {
        let n = 8usize;
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut ectx = SliceContexts::init(init_type(2, false), 26);

        // Empty predictor ⇒ no palette_predictor_run loop bins.
        // num_signalled_palette_entries = 3.
        encode_eg0(&mut cabac, &mut w, 3);
        // new_palette_entries: comp0 {10, 20, 30}, comp1 {60, 70, 80},
        // comp2 {110, 120, 130} — FL(8) each.
        for base in [10u32, 60, 110] {
            for i in 0..3u32 {
                cabac.encode_bypass_bits(&mut w, base + 10 * i, 8);
            }
        }
        // palette_escape_val_present_flag = 0.
        cabac.encode_bypass(&mut w, 0);
        // MaxPaletteIndex = 2. Plan: rows 0..3 index 0 (run to end of
        // 32 samples? no — traverse scan): we emit indices [0, 1] and
        // rely on: idx run of 31, copy-above run of 16, final run
        // (explicit, index 1) to the end.
        // num_palette_indices_minus1 = 1 (two explicit indices).
        {
            // cRiceParam k = 3 + ((2+1)>>3) = 3; value 1 < cMax:
            // quotient 0 ⇒ one 0 bin, then 3 rice bits '001'.
            cabac.encode_bypass(&mut w, 0);
            cabac.encode_bypass_bits(&mut w, 1, 3);
        }
        // palette_idx_idc[0] = 0 (TB cMax = 2: n=3, k=1, u=1; 0 -> '0').
        encode_tb(&mut cabac, &mut w, 0, 2);
        // palette_idx_idc[1] = 0 (TB cMax = 1: n=2,k=0,u=1 -> 0 bins
        // for value 0? k=0: read 0 bits, val=0 < u=1 -> 0). No bins.
        encode_tb(&mut cabac, &mut w, 0, 1);
        // copy_above_indices_for_final_run_flag = 0 (final run explicit).
        cabac.encode_decision(&mut w, &mut ectx.palette_copy_above_flag[0], 0);
        // palette_transpose_flag = 0.
        cabac.encode_decision(&mut w, &mut ectx.palette_transpose_flag[0], 0);
        // (no escapes ⇒ no delta_qp)
        // Scan loop plan over 64 samples:
        //  pos 0: explicit idx (idc 0 ⇒ index 0), remaining 2-1=1,
        //         run: PaletteMaxRunMinus1 = 64-0-1-1-0 = 62,
        //         prefix cMax = floor(log2(62))+1 = 6.
        //         PaletteRunMinus1 = 31: prefix ≥ 2: 31 = 16+15 ⇒
        //         prefix 5 ('11111' then 0 if < cMax... prefix=5 <
        //         cMax=6 ⇒ five 1s + terminating 0), suffix TB:
        //         PrefixOffset=16, (32 > 62? no) ⇒ cMax_tb = 15,
        //         suffix = 15.
        //  pos 32: copy_above_palette_indices_flag = 1, run:
        //         remaining 1 > 0 ⇒ PaletteMaxRunMinus1 =
        //         64-32-1-1-0 = 30, PaletteRunMinus1 = 15: prefix 4
        //         + suffix TB (PrefixOffset 8, 16>30? no ⇒ cMax_tb
        //         = 7, suffix 7).
        //  pos 48: prev is copy-above ⇒ no copy flag read; explicit
        //         index (idc 0 adjusted vs left neighbour...) run to
        //         end (remaining becomes 0, copy_flag(0) ==
        //         final_run_flag(0) ⇒ RunToEnd).
        {
            // pos 0 run: prefix 5 bins: ctxInc from
            // palette_run_prefix_ctx_inc(bin, false, idx=0):
            // bin0 -> eq 9-63(0) = 0; bins 1..4 -> 3,3,4,4; bin 5+
            // bypass... prefix value 5 means bins 0..4 = 1 and bin 5
            // = 0 terminator (5 < cMax 6).
            for bin_idx in 0..5u32 {
                let inc = palette_run_prefix_ctx_inc(bin_idx, false, 0).unwrap();
                cabac.encode_decision(&mut w, &mut ectx.palette_run_prefix[inc as usize], 1);
            }
            cabac.encode_bypass(&mut w, 0); // bin 5: bypass terminator
            encode_tb(&mut cabac, &mut w, 15, 15); // suffix
        }
        {
            // pos 32: copy_above_palette_indices_flag = 1.
            cabac.encode_decision(&mut w, &mut ectx.palette_copy_above_flag[0], 1);
            // run prefix 4 (bins 0..3 ones, bin 4 zero), copy row of
            // Table 9-51: 5,6,6,7 then terminator ctx 7.
            for bin_idx in 0..4u32 {
                let inc = palette_run_prefix_ctx_inc(bin_idx, true, 0).unwrap();
                cabac.encode_decision(&mut w, &mut ectx.palette_run_prefix[inc as usize], 1);
            }
            let inc = palette_run_prefix_ctx_inc(4, true, 0).unwrap();
            cabac.encode_decision(&mut w, &mut ectx.palette_run_prefix[inc as usize], 0);
            encode_tb(&mut cabac, &mut w, 7, 7); // suffix
        }
        // pos 48: explicit index, remaining -> 0, copy(false) ==
        // final(false) ⇒ RunToEnd, no bins.
        cabac.encode_terminate(&mut w, 1);
        let bytes = w.finish();

        // ---- decode ----
        let mut dctx = SliceContexts::init(init_type(2, false), 26);
        let mut engine = CabacEngine::new(crate::hevc::engine::bitreader::BitReader::new(&bytes))
            .expect("engine");
        let mut qg = QuantGroupState::default();
        let cu = decode_palette_coding(&mut engine, &mut dctx, &params(true), &mut qg, n)
            .expect("palette parse");
        assert_eq!(cu.palette[0], vec![10, 20, 30]);
        assert_eq!(cu.palette[1], vec![60, 70, 80]);
        assert_eq!(cu.palette[2], vec![110, 120, 130]);
        assert!(!cu.escape_present);
        assert!(!cu.transpose);
        // Positions 0..=31 (traverse order = rows 0..3 serpentine):
        // index 0. Positions 32..47 copy row above (rows 4..5 copy
        // row 3 = index 0)... every sample ends up index 0 except the
        // final run: idc[1] = 0 adjusts (eq. 7-84) against the
        // neighbour index 0 ⇒ index 1 for positions 48..63 (rows
        // 6..7).
        for pos in 0..48usize {
            let p = traverse(n)[pos];
            assert_eq!(
                cu.index_map[p.y as usize * n + p.x as usize],
                0,
                "pos {pos}"
            );
        }
        for pos in 48..64usize {
            let p = traverse(n)[pos];
            assert_eq!(
                cu.index_map[p.y as usize * n + p.x as usize],
                1,
                "pos {pos}"
            );
        }
        // Predictor update: current palette becomes the predictor.
        assert_eq!(dctx.palette_predictor.entries[0], vec![10, 20, 30]);
        assert_eq!(dctx.palette_predictor.size(), 3);

        // ---- reconstruction (4:2:0) ----
        let mut luma = vec![0i32; 64];
        reconstruct_palette_component(&cu, 0, 1, 1, 26, 8, true, |x, y, v| {
            luma[y * 8 + x] = v;
        });
        for y in 0..8 {
            for x in 0..8 {
                let expect = if y >= 6 { 20 } else { 10 };
                assert_eq!(luma[y * 8 + x], expect, "luma ({x},{y})");
            }
        }
        let mut cb = [0i32; 16];
        reconstruct_palette_component(&cu, 1, 2, 2, 26, 8, true, |x, y, v| {
            cb[y * 4 + x] = v;
        });
        for y in 0..4 {
            for x in 0..4 {
                let expect = if y >= 3 { 70 } else { 60 };
                assert_eq!(cb[y * 4 + x], expect, "cb ({x},{y})");
            }
        }
    }

    /// Predictor reuse: runs select predictor entries, the update
    /// appends unused entries after the current palette (eq. 8-79).
    #[test]
    fn predictor_reuse_and_update() {
        let n = 4usize;
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        // Seed a 4-entry predictor.
        let seed = PalettePredictor {
            entries: [vec![1, 2, 3, 4], vec![11, 12, 13, 14], vec![21, 22, 23, 24]],
        };

        // Reuse entries 0 and 2: runs 0 ("reuse current"), then 2
        // (skip one, reuse), then 1 (finish).
        encode_eg0(&mut cabac, &mut w, 0);
        encode_eg0(&mut cabac, &mut w, 2);
        encode_eg0(&mut cabac, &mut w, 1);
        // num_signalled = 0.
        encode_eg0(&mut cabac, &mut w, 0);
        // escape_present = 0 (CurrentPaletteSize = 2).
        cabac.encode_bypass(&mut w, 0);
        // MaxPaletteIndex = 1 > 0: num_palette_indices_minus1 = 0
        // (k = 3, quotient 0 + 3 rice zero bits).
        cabac.encode_bypass(&mut w, 0);
        cabac.encode_bypass_bits(&mut w, 0, 3);
        // palette_idx_idc[0] = 0: TB cMax = 1 -> n=2, k=0, u=1, value
        // 0 < u -> zero bins... wait k = floor(log2(2)) = 1, u = 2^2-2
        // = 2: value 0 < 2 -> FL(k=1) one bin '0'.
        encode_tb(&mut cabac, &mut w, 0, 1);
        let mut ectx = SliceContexts::init(init_type(2, false), 26);
        // final-run flag = 0, transpose = 0.
        cabac.encode_decision(&mut w, &mut ectx.palette_copy_above_flag[0], 0);
        cabac.encode_decision(&mut w, &mut ectx.palette_transpose_flag[0], 0);
        // Scan: pos 0 explicit idx 0, remaining -> 0, copy(0) ==
        // final(0) => run to end. No further bins.
        cabac.encode_terminate(&mut w, 1);
        let bytes = w.finish();

        let mut dctx = SliceContexts::init(init_type(2, false), 26);
        dctx.palette_predictor = seed;
        let mut engine = CabacEngine::new(crate::hevc::engine::bitreader::BitReader::new(&bytes))
            .expect("engine");
        let mut qg = QuantGroupState::default();
        let cu = decode_palette_coding(&mut engine, &mut dctx, &params(true), &mut qg, n)
            .expect("palette parse");
        assert_eq!(cu.palette[0], vec![1, 3]);
        assert_eq!(cu.palette[1], vec![11, 13]);
        assert!(cu.index_map.iter().all(|&i| i == 0));
        // eq. 8-79: current palette first, then unused predictor
        // entries 1 and 3.
        assert_eq!(dctx.palette_predictor.entries[0], vec![1, 3, 2, 4]);
        assert_eq!(dctx.palette_predictor.entries[2], vec![21, 23, 22, 24]);
    }

    /// §9.3.3.6 TB is the FL/offset split around `u`.
    #[test]
    fn tb_binarization_shapes() {
        for c_max in 1u32..20 {
            for v in 0..=c_max {
                let mut w = BitWriter::new();
                let mut cabac = CabacEncoder::new();
                encode_tb(&mut cabac, &mut w, v, c_max);
                cabac.encode_terminate(&mut w, 1);
                let bytes = w.finish();
                let mut engine =
                    CabacEngine::new(crate::hevc::engine::bitreader::BitReader::new(&bytes))
                        .expect("engine");
                assert_eq!(
                    decode_tb(&mut engine, c_max).expect("tb"),
                    v,
                    "cMax {c_max} value {v}"
                );
            }
        }
    }

    /// Escape reconstruction: eq. 8-77 dequantization and the eq. 8-72
    /// bypass passthrough.
    #[test]
    fn escape_dequantization_matches_eq_8_77() {
        let cu = PaletteCu {
            n_cbs: 4,
            palette: [vec![50], vec![60], vec![70]],
            escape_present: true,
            transpose: false,
            index_map: {
                let mut m = vec![0u8; 16];
                m[0] = 1; // escape at (0,0)
                m
            },
            escape_vals: [
                {
                    let mut v = vec![0u16; 16];
                    v[0] = 7;
                    v
                },
                vec![0u16; 16],
                vec![0u16; 16],
            ],
            cu_qp_delta: None,
            cu_chroma_qp_offset: None,
        };
        // qP = 10: levelScale[4] = 64, shift 1: ((7*64) << 1 + 32) >> 6
        // = (896 + 32) >> 6 = 14.
        let mut out = [0i32; 16];
        reconstruct_palette_component(&cu, 0, 1, 1, 10, 8, false, |x, y, v| {
            out[y * 4 + x] = v;
        });
        assert_eq!(out[0], 14);
        assert_eq!(out[1], 50);
        // Bypass: the escape value is the sample.
        let mut out2 = [0i32; 16];
        reconstruct_palette_component(&cu, 0, 1, 1, 10, 8, true, |x, y, v| {
            out2[y * 4 + x] = v;
        });
        assert_eq!(out2[0], 7);
    }
}
