//! PCM-only IDR encoder — the crate's encoder bootstrap.
//!
//! Emits fully conformant Main-profile Annex B access units in which
//! every coding unit is a `pcm_flag == 1` PCM block (§7.3.8.7): the
//! samples ride uncompressed inside the slice data, so the decode is
//! exactly lossless. The geometry is fixed at the simplest legal
//! shape: `CtbSizeY == MinCbSizeY == 16` with the PCM size range
//! pinned to 16 (`Log2MinIpcmCbSizeY == Log2MaxIpcmCbSizeY == 4`), so
//! every CTB is one unsplit coding unit and the whole §7.3.8 syntax
//! walk per CTB is: `part_mode` (one context-coded bin, `PART_2Nx2N`),
//! `pcm_flag` (a §9.3.5.6 terminate bin), the `pcm_alignment_zero_bit`
//! run, the raw §7.3.8.7 sample payload, the §9.3.5.2 engine re-init,
//! and `end_of_slice_segment_flag`.
//!
//! Every picture is an IDR (`IDR_N_LP`) with in-band VPS/SPS/PPS, so
//! the stream is seekable anywhere and decoder-stateless. 4:2:0 8-bit
//! only; picture dimensions must be multiples of 16 (the CTB size —
//! this bootstrap writes no conformance-window cropping).
//!
//! In-loop filters are neutralized the conformant way: SAO off in the
//! SPS, deblocking disabled in the PPS, and
//! `pcm_loop_filter_disabled_flag == 1` — a PCM-only picture
//! reconstructs to the raw samples bit for bit.

use crate::hevc::engine::availability::{PictureTiling, TilingParams};
use crate::hevc::engine::cabac::init_type;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::nal::{annexb, nal_unit};

/// The fixed CTB / coding-block / PCM-block log2 size of this encoder.
const CTB_LOG2: u32 = 4;
/// The fixed CTB size (16).
const CTB: usize = 1 << CTB_LOG2;
/// `SliceQpY` written in every slice header (`slice_qp_delta == 0`
/// over `init_qp_minus26 == 0`). PCM blocks carry no residual, so the
/// QP only seeds context initialization.
const SLICE_QP: i32 = 26;

/// Per-AU shape options for the PCM encoder.
#[derive(Debug, Clone)]
pub struct PcmAuOptions {
    /// Number of slice segments (1 = a single independent segment;
    /// more = one independent + N−1 dependent segments). Ignored when
    /// [`Self::independent_slices`] is non-empty.
    pub segments: usize,
    /// Independent slices as `(start_ctb, slice_loop_filter_across_
    /// slices_enabled_flag)` pairs (sorted, first start 0). Empty =
    /// the [`Self::segments`] split. Heterogeneous per-slice flags
    /// exercise the §8.7.2.1 / §8.7.3.2 per-slice boundary gates.
    pub independent_slices: Vec<(usize, bool)>,
    /// Enable the §8.7.2 deblocking filter (PPS
    /// `pps_deblocking_filter_disabled_flag == 0`). With every CU a
    /// PCM block and `pcm_loop_filter_disabled_flag == 1`, a
    /// conforming decoder must leave all samples untouched — this
    /// exercises the §8.7.2.5.4 `nDp`/`nDq` PCM suppression.
    pub deblocking: bool,
    /// Enable luma band-offset SAO (SPS SAO on,
    /// `slice_sao_luma_flag == 1`, per-CTB §7.3.8.3 band parameters
    /// with non-zero offsets).
    pub sao_luma_band: bool,
    /// Enable luma vertical-class edge-offset SAO (`sao_type_idx_luma
    /// == 2`, `sao_eo_class_luma == 1`): the classification reads
    /// above / below neighbours, exercising the §8.7.3.2 cross-slice
    /// availability at horizontal slice boundaries. Mutually exclusive
    /// with [`Self::sao_luma_band`].
    pub sao_luma_eo_vertical: bool,
    /// `pcm_loop_filter_disabled_flag` (§7.4.3.2.1). `false` lets the
    /// enabled loop filters modify the PCM samples like any other CU's.
    pub pcm_loop_filter_disabled: bool,
    /// Tile grid as `(columns, rows)` — `tiles_enabled_flag == 1` with
    /// `uniform_spacing_flag == 1` and
    /// `loop_filter_across_tiles_enabled_flag == 0`. The picture is
    /// coded as ONE slice segment whose CTBs walk the §6.5.1 tile scan:
    /// every tile is its own §7.3.8.1 subset (`end_of_subset_one_bit` +
    /// byte alignment between tiles, §9.3.2.2 context re-initialization
    /// at each tile start) and the slice header carries the §7.4.7.1
    /// `entry_point_offset_minus1[]` block. Requires `columns * rows >=
    /// 2`; incompatible with multi-segment / SAO options.
    pub tiles: Option<(u32, u32)>,
}

impl Default for PcmAuOptions {
    fn default() -> Self {
        Self {
            segments: 1,
            independent_slices: Vec::new(),
            deblocking: false,
            sao_luma_band: false,
            sao_luma_eo_vertical: false,
            pcm_loop_filter_disabled: true,
            tiles: None,
        }
    }
}

/// Errors from the PCM encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEncodeError {
    /// Width or height is zero or not a multiple of the 16-sample CTB.
    BadDimensions {
        /// Requested luma width.
        width: usize,
        /// Requested luma height.
        height: usize,
    },
    /// A supplied plane's length does not match the 4:2:0 geometry.
    PlaneSize {
        /// Which plane (`"y"`, `"cb"`, `"cr"`).
        plane: &'static str,
        /// Required sample count.
        expected: usize,
        /// Supplied sample count.
        got: usize,
    },
}

impl core::fmt::Display for PcmEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadDimensions { width, height } => write!(
                f,
                "PCM encoder requires nonzero dimensions that are multiples of 16, got {width}x{height}"
            ),
            Self::PlaneSize {
                plane,
                expected,
                got,
            } => write!(f, "{plane} plane has {got} samples, expected {expected}"),
        }
    }
}

impl std::error::Error for PcmEncodeError {}

/// Table A.8 — pick the smallest `general_level_idc` whose `MaxLumaPs`
/// covers the picture. (A PCM payload can exceed a level's bit-rate
/// ceilings for large / high-rate content; this bootstrap levels by
/// luma picture size, the binding constraint for its intended
/// small-picture use.)
pub(crate) fn level_idc_for(luma_ps: usize) -> u8 {
    const LEVELS: [(usize, u8); 9] = [
        (36_864, 30),      // 1
        (122_880, 60),     // 2
        (245_760, 63),     // 2.1
        (552_960, 90),     // 3
        (983_040, 93),     // 3.1
        (2_228_224, 120),  // 4
        (8_912_896, 150),  // 5
        (35_651_584, 180), // 6
        (usize::MAX, 186), // 6.2
    ];
    LEVELS
        .iter()
        .find(|(max_ps, _)| luma_ps <= *max_ps)
        .map(|&(_, idc)| idc)
        .unwrap_or(186)
}

/// Table A.8 — the smallest `general_level_idc` whose `MaxTileCols` /
/// `MaxTileRows` cover the requested tile grid (§A.4.2 item f:
/// `num_tile_columns_minus1 < MaxTileCols`, `num_tile_rows_minus1 <
/// MaxTileRows`). This bootstrap's tiny tiles can undercut the
/// 256-sample minimum tile-column width; the level is chosen by tile
/// count, the constraint a decoder actually dispatches on.
fn level_idc_for_tiles(cols: u32, rows: u32) -> u8 {
    const LEVELS: [(u32, u32, u8); 6] = [
        (1, 1, 30),    // 1 .. 2.1
        (2, 2, 90),    // 3
        (3, 3, 93),    // 3.1
        (5, 5, 120),   // 4 / 4.1
        (10, 11, 150), // 5 .. 5.2
        (20, 22, 180), // 6 .. 6.2
    ];
    LEVELS
        .iter()
        .find(|&&(c, r, _)| cols <= c && rows <= r)
        .map(|&(_, _, idc)| idc)
        .unwrap_or(180)
}

/// §7.3.3 `profile_tier_level( 1, 0 )` — Main profile, Main tier.
pub(crate) fn write_ptl(w: &mut BitWriter, level_idc: u8) {
    w.put_bits(0, 2); // general_profile_space
    w.put_bit(0); // general_tier_flag
    w.put_bits(1, 5); // general_profile_idc = 1 (Main)
    // general_profile_compatibility_flag[0..32]: Main (1) is also
    // decodable by Main 10 (2) decoders.
    let mut compat: u32 = 0;
    compat |= 1 << (31 - 1); // flag[1] — Main
    compat |= 1 << (31 - 2); // flag[2] — Main 10
    w.put_bits(compat, 32);
    w.put_bit(1); // general_progressive_source_flag
    w.put_bit(0); // general_interlaced_source_flag
    w.put_bit(1); // general_non_packed_constraint_flag
    w.put_bit(1); // general_frame_only_constraint_flag
    // Main profile: general_reserved_zero_43bits.
    w.put_bits(0, 32);
    w.put_bits(0, 11);
    // Profile 1: general_inbld_flag = 0.
    w.put_bit(0);
    w.put_bits(u32::from(level_idc), 8); // general_level_idc
    // max_sub_layers_minus1 == 0: no sub-layer PTL syntax.
}

/// §7.3.2.1 — the minimal single-layer VPS.
pub(crate) fn write_vps(level_idc: u8) -> Vec<u8> {
    write_vps_cfg(level_idc, 1, 0)
}

/// [`write_vps`] with explicit `vps_max_dec_pic_buffering_minus1[0]` /
/// `vps_max_num_reorder_pics[0]` (the hierarchical-B encoder holds
/// more references and reorders output).
pub(crate) fn write_vps_cfg(
    level_idc: u8,
    max_dec_pic_buffering_minus1: u32,
    max_num_reorder_pics: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl(&mut w, level_idc);
    w.put_bit(1); // vps_sub_layer_ordering_info_present_flag
    w.ue(max_dec_pic_buffering_minus1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(max_num_reorder_pics); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.put_bits(0, 6); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.put_bit(0); // vps_timing_info_present_flag
    w.put_bit(0); // vps_extension_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.2 — the fixed-geometry SPS (4:2:0, 8-bit, CTB 16, PCM on).
pub(crate) fn write_sps(
    width: usize,
    height: usize,
    level_idc: u8,
    sao_enabled: bool,
    pcm_loop_filter_disabled: bool,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc = 4:2:0
    w.ue(width as u32); // pic_width_in_luma_samples
    w.ue(height as u32); // pic_height_in_luma_samples
    w.put_bit(0); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4
    w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(CTB_LOG2 - 3); // log2_min_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_luma_coding_block_size (CTB 16)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(2); // log2_diff_max_min_luma_transform_block_size (16)
    w.ue(0); // max_transform_hierarchy_depth_inter
    w.ue(0); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(0); // amp_enabled_flag
    w.put_bit(u8::from(sao_enabled)); // sample_adaptive_offset_enabled_flag
    w.put_bit(1); // pcm_enabled_flag
    w.put_bits(7, 4); // pcm_sample_bit_depth_luma_minus1 (8-bit)
    w.put_bits(7, 4); // pcm_sample_bit_depth_chroma_minus1
    w.ue(CTB_LOG2 - 3); // log2_min_pcm_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_pcm_luma_coding_block_size
    w.put_bit(u8::from(pcm_loop_filter_disabled)); // pcm_loop_filter_disabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    w.put_bit(0); // vui_parameters_present_flag
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.3 — the all-defaults PPS (deblocking disabled), optionally
/// with a uniform tile grid (`tiles_enabled_flag == 1`,
/// `uniform_spacing_flag == 1`,
/// `loop_filter_across_tiles_enabled_flag == 0`).
pub(crate) fn write_pps(
    dependent_slice_segments_enabled: bool,
    deblocking_enabled: bool,
    tiles: Option<(u32, u32)>,
) -> Vec<u8> {
    write_pps_lf(
        dependent_slice_segments_enabled,
        deblocking_enabled,
        false,
        tiles,
    )
}

/// [`write_pps`] with `deblocking_filter_override_enabled_flag`
/// control: the loop-filter encoders write the PPS with
/// `pps_deblocking_filter_disabled_flag == 1` +
/// `deblocking_filter_override_enabled_flag == 1` so every slice
/// elects deblocking on/off itself (§7.3.6.1 override group).
pub(crate) fn write_pps_lf(
    dependent_slice_segments_enabled: bool,
    deblocking_enabled: bool,
    deblocking_override_enabled: bool,
    tiles: Option<(u32, u32)>,
) -> Vec<u8> {
    write_pps_full(
        dependent_slice_segments_enabled,
        deblocking_enabled,
        deblocking_override_enabled,
        tiles,
        false,
    )
}

/// [`write_pps_lf`] with `cu_qp_delta_enabled_flag` control: the
/// adaptive-quantization encoders signal per-CTB QP through §7.3.8.10
/// `cu_qp_delta` (`diff_cu_qp_delta_depth == 0`, one quantization
/// group per CTB).
pub(crate) fn write_pps_full(
    dependent_slice_segments_enabled: bool,
    deblocking_enabled: bool,
    deblocking_override_enabled: bool,
    tiles: Option<(u32, u32)>,
    cu_qp_delta: bool,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.put_bit(u8::from(dependent_slice_segments_enabled)); // dependent_slice_segments_enabled_flag
    w.put_bit(0); // output_flag_present_flag
    w.put_bits(0, 3); // num_extra_slice_header_bits
    w.put_bit(0); // sign_data_hiding_enabled_flag
    w.put_bit(0); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(SLICE_QP - 26); // init_qp_minus26
    w.put_bit(0); // constrained_intra_pred_flag
    w.put_bit(0); // transform_skip_enabled_flag
    w.put_bit(u8::from(cu_qp_delta)); // cu_qp_delta_enabled_flag
    if cu_qp_delta {
        w.ue(0); // diff_cu_qp_delta_depth (QG == CTB)
    }
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.put_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(0); // weighted_pred_flag
    w.put_bit(0); // weighted_bipred_flag
    w.put_bit(0); // transquant_bypass_enabled_flag
    w.put_bit(u8::from(tiles.is_some())); // tiles_enabled_flag
    w.put_bit(0); // entropy_coding_sync_enabled_flag
    if let Some((cols, rows)) = tiles {
        w.ue(cols - 1); // num_tile_columns_minus1
        w.ue(rows - 1); // num_tile_rows_minus1
        w.put_bit(1); // uniform_spacing_flag
        w.put_bit(0); // loop_filter_across_tiles_enabled_flag
    }
    w.put_bit(1); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(1); // deblocking_filter_control_present_flag
    w.put_bit(u8::from(deblocking_override_enabled)); // deblocking_filter_override_enabled_flag
    w.put_bit(u8::from(!deblocking_enabled)); // pps_deblocking_filter_disabled_flag
    if deblocking_enabled {
        w.se(0); // pps_beta_offset_div2
        w.se(0); // pps_tc_offset_div2
    }
    w.put_bit(0); // pps_scaling_list_data_present_flag
    w.put_bit(0); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(0); // slice_segment_header_extension_present_flag
    w.put_bit(0); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.6.1 + §7.3.8.1 — the picture's slice segments: one
/// independent I-slice segment followed by `segments − 1` dependent
/// slice segments (§7.4.7.1 inheritance), every CTB a PCM coding
/// unit. Returns one RBSP per segment.
///
/// The CABAC context variables carry across the segment boundary per
/// §9.3.1 / §9.3.2.4 (`TableStateIdxDs` storage at each segment's
/// `end_of_slice_segment_flag == 1`, §9.3.2.5 synchronization at the
/// next dependent segment's start) — each segment gets a fresh
/// §9.3.5.2 arithmetic engine over the shared context state.
fn write_idr_slice_segments(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    opts: &PcmAuOptions,
) -> Vec<Vec<u8>> {
    let sao = opts.sao_luma_band || opts.sao_luma_eo_vertical;
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let total = ctbs_x * ctbs_y;
    // Ceil(Log2(PicSizeInCtbsY)) — the slice_segment_address width.
    let addr_bits = (usize::BITS - (total - 1).leading_zeros()).max(1) as u8;

    // Segment plan: (start_ctb, Some(across_flag) for an independent
    // slice header / None for a dependent segment).
    let plan: Vec<(usize, Option<bool>)> = if opts.independent_slices.is_empty() {
        // Even split; segment 0 independent, the rest dependent.
        let seg_len = total.div_ceil(opts.segments);
        (0..opts.segments)
            .map(|i| (i * seg_len, (i == 0).then_some(true)))
            .collect()
    } else {
        opts.independent_slices
            .iter()
            .map(|&(start, across)| (start, Some(across)))
            .collect()
    };
    let dependent_mode = opts.independent_slices.is_empty() && opts.segments > 1;

    // initType 0 (I slice, equation 9-7), SliceQpY = 26. Carried across
    // dependent segments (§9.3.2.4 / §9.3.2.5); re-initialized at each
    // independent slice.
    let mut ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);
    // Start CTB of the current independent slice (SAO merge / left
    // availability is denied across independent slices).
    let mut cur_slice_start = 0usize;

    let mut out = Vec::with_capacity(plan.len());
    for (seg_idx, &(start, indep)) in plan.iter().enumerate() {
        let end = plan.get(seg_idx + 1).map_or(total, |&(next, _)| next);
        let mut w = BitWriter::new();

        // ---- slice_segment_header() ----
        let first = seg_idx == 0;
        w.put_bit(u8::from(first)); // first_slice_segment_in_pic_flag
        w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
        w.ue(0); // slice_pic_parameter_set_id
        if !first {
            if dependent_mode {
                w.put_bit(1); // dependent_slice_segment_flag
            }
            w.put_bits(start as u32, addr_bits); // slice_segment_address
        }
        if let Some(across) = indep {
            // Independent slice: fresh CABAC contexts (§9.3.2.2) and a
            // full slice-level header.
            ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);
            cur_slice_start = start;
            w.ue(2); // slice_type = I
            if sao {
                // SPS SAO enabled: the slice SAO flags are present.
                w.put_bit(1); // slice_sao_luma_flag
                w.put_bit(0); // slice_sao_chroma_flag (4:2:0 present)
            }
            w.se(SLICE_QP - 26); // slice_qp_delta
            if opts.deblocking || sao {
                // §7.3.6.1 gate: pps_loop_filter_across_slices == 1 and
                // (SAO on or deblocking not disabled).
                w.put_bit(u8::from(across)); // slice_loop_filter_across_slices_enabled_flag
            }
        }
        // No tiles / WPP: no entry points. No header extension.
        w.rbsp_trailing_bits(); // byte_alignment() before slice data

        // ---- slice_segment_data() ----
        let mut cabac = CabacEncoder::new();
        for addr in start..end {
            let x0 = (addr % ctbs_x) * CTB;
            let y0 = (addr / ctbs_x) * CTB;
            if sao {
                // §7.3.8.3 sao( rx, ry ): decline the available merges
                // (left / above must lie in the same slice + tile),
                // then explicit luma parameters.
                let (rx, ry) = (addr % ctbs_x, addr / ctbs_x);
                if rx > 0 && addr > cur_slice_start {
                    cabac.encode_decision(&mut w, &mut ctxs.sao_merge_flag[0], 0);
                }
                if ry > 0 && addr - ctbs_x >= cur_slice_start {
                    cabac.encode_decision(&mut w, &mut ctxs.sao_merge_flag[0], 0);
                }
                if opts.sao_luma_band {
                    // sao_type_idx_luma = 1 (band): TR bins "1" (ctx),
                    // "0" (bypass).
                    cabac.encode_decision(&mut w, &mut ctxs.sao_type_idx[0], 1);
                    cabac.encode_bypass(&mut w, 0);
                    // sao_offset_abs[0][i] — TR(cMax 7, bypass): values
                    // 1, 2, 3, 4.
                    for v in 1..=4u32 {
                        for _ in 0..v {
                            cabac.encode_bypass(&mut w, 1);
                        }
                        cabac.encode_bypass(&mut w, 0); // v < cMax terminator
                    }
                    // sao_offset_sign[0][i]: +, −, +, −.
                    for sign in [0u8, 1, 0, 1] {
                        cabac.encode_bypass(&mut w, sign);
                    }
                    // sao_band_position[0]: FL(5) = 8.
                    cabac.encode_bypass_bits(&mut w, 8, 5);
                } else {
                    // sao_type_idx_luma = 2 (edge): TR bins "1" (ctx),
                    // "1" (bypass).
                    cabac.encode_decision(&mut w, &mut ctxs.sao_type_idx[0], 1);
                    cabac.encode_bypass(&mut w, 1);
                    // sao_offset_abs[0][i]: 1, 2, 1, 2 (edge offsets
                    // carry implicit signs per §7.4.9.3).
                    for v in [1u32, 2, 1, 2] {
                        for _ in 0..v {
                            cabac.encode_bypass(&mut w, 1);
                        }
                        cabac.encode_bypass(&mut w, 0);
                    }
                    // sao_eo_class_luma: FL(2) = 1 (vertical — reads
                    // the above / below neighbours).
                    cabac.encode_bypass_bits(&mut w, 1, 2);
                }
            }
            write_pcm_ctu(&mut w, &mut cabac, &mut ctxs, y, cb, cr, width, x0, y0);
            // end_of_slice_segment_flag: 1 only at the segment's last CTB.
            cabac.encode_terminate(&mut w, u8::from(addr == end - 1));
        }
        // The final terminate-1 flush wrote the rbsp_stop_one_bit;
        // rbsp_slice_segment_trailing_bits() is alignment zeros from here.
        w.align_zero();
        out.push(w.finish());
    }
    out
}

/// §7.3.8.2 + §7.3.8.5 — one all-PCM CTB (`CtbLog2SizeY ==
/// MinCbLog2SizeY`, so the coding quadtree is the single unsplit CU):
/// `part_mode` (§9.3.3.7 bin "1" = `PART_2Nx2N`), `pcm_flag` (a
/// §9.3.5.6 terminate bin — value 1 flushes the codeword), the
/// `pcm_alignment_zero_bit` run, the raw §7.3.8.7 samples (luma then
/// Cb, Cr), and the §9.3.5.2 engine re-initialization.
#[allow(clippy::too_many_arguments)]
fn write_pcm_ctu(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
) {
    let cw = width / 2;
    cabac.encode_decision(w, &mut ctxs.part_mode[0], 1);
    cabac.encode_terminate(w, 1);
    w.align_zero();
    for j in 0..CTB {
        for i in 0..CTB {
            w.put_bits(u32::from(y[(y0 + j) * width + x0 + i]), 8);
        }
    }
    let (cx, cy) = (x0 / 2, y0 / 2);
    for plane in [cb, cr] {
        for j in 0..CTB / 2 {
            for i in 0..CTB / 2 {
                w.put_bits(u32::from(plane[(cy + j) * cw + cx + i]), 8);
            }
        }
    }
    cabac.reinit();
}

/// §7.4.1.1 — the coded (emulation-prevention-escaped) byte length of
/// `bytes`, given the zero-run carried in from the preceding coded
/// bytes. Returns the length and the carry-out zero run. The dual of
/// the counting the §7.4.7.1 `entry_point_offset_minus1[i]` values
/// perform over the coded slice-segment data.
fn escaped_len(bytes: &[u8], mut zero_run: u32) -> (usize, u32) {
    let mut len = 0usize;
    for &b in bytes {
        if zero_run >= 2 && b <= 0x03 {
            len += 1; // emulation_prevention_three_byte
            zero_run = 0;
        }
        len += 1;
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
    (len, zero_run)
}

/// §7.3.6.1 + §7.3.8.1 — the tiled picture as ONE independent I-slice
/// segment whose CTBs walk the §6.5.1 tile scan. Every tile is its own
/// subset: fresh §9.3.2.2 contexts and a fresh §9.3.5.2 arithmetic
/// engine at each tile start, `end_of_subset_one_bit` + byte alignment
/// after every tile but the last, and the §7.4.7.1
/// `entry_point_offset_minus1[]` block (offsets in CODED bytes,
/// emulation-prevention included) in the slice header.
fn write_tiled_idr_slice(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    opts: &PcmAuOptions,
) -> Vec<u8> {
    let (cols, rows) = opts.tiles.expect("caller checked the tile grid");
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let total = ctbs_x * ctbs_y;
    let tiling = PictureTiling::new(
        ctbs_x as u32,
        ctbs_y as u32,
        width as u32,
        height as u32,
        CTB_LOG2,
        2, // MinTbLog2SizeY (SPS: log2_min_luma_transform_block_size = 4)
        &TilingParams {
            num_tile_columns_minus1: cols - 1,
            num_tile_rows_minus1: rows - 1,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
        },
    )
    .expect("validated tile geometry");

    // ---- slice_segment_data( ), one coded subset per tile ----
    let mut subsets: Vec<Vec<u8>> = Vec::new();
    let mut w = BitWriter::new();
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);
    for ts in 0..total as u32 {
        let rs = tiling.ctb_addr_ts_to_rs(ts) as usize;
        let x0 = (rs % ctbs_x) * CTB;
        let y0 = (rs / ctbs_x) * CTB;
        write_pcm_ctu(&mut w, &mut cabac, &mut ctxs, y, cb, cr, width, x0, y0);
        let last_of_pic = ts == total as u32 - 1;
        // end_of_slice_segment_flag: 1 only at the picture's last CTB.
        cabac.encode_terminate(&mut w, u8::from(last_of_pic));
        let last_of_tile = last_of_pic || tiling.tile_id(ts + 1) != tiling.tile_id(ts);
        if last_of_tile {
            if !last_of_pic {
                // §7.3.8.1 end_of_subset_one_bit (terminate-1 flush).
                cabac.encode_terminate(&mut w, 1);
            }
            // byte_alignment( ) — the flush wrote the trailing one bit.
            w.align_zero();
            subsets.push(std::mem::take(&mut w).finish());
            // §9.3.2.2 — fresh contexts + engine at the next tile start.
            cabac = CabacEncoder::new();
            ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);
        }
    }

    // §7.4.7.1 — entry_point_offset_minus1[i] counts CODED bytes; the
    // emulation-prevention zero-run threads across subset boundaries
    // (the slice header's byte_alignment one-bit makes its last byte
    // nonzero, so the first subset starts with a zero run of 0).
    let mut offsets: Vec<u32> = Vec::new();
    let mut zero_run = 0u32;
    for s in &subsets[..subsets.len() - 1] {
        let (len, zr) = escaped_len(s, zero_run);
        zero_run = zr;
        offsets.push(len as u32 - 1);
    }
    let offset_len = offsets
        .iter()
        .map(|&o| 32 - o.leading_zeros())
        .max()
        .unwrap_or(1)
        .max(1) as u8;

    // ---- slice_segment_header( ) ----
    let mut h = BitWriter::new();
    h.put_bit(1); // first_slice_segment_in_pic_flag
    h.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    h.ue(0); // slice_pic_parameter_set_id
    h.ue(2); // slice_type = I
    h.se(SLICE_QP - 26); // slice_qp_delta
    if opts.deblocking {
        // §7.3.6.1 gate: pps_loop_filter_across_slices == 1 and
        // deblocking not disabled.
        h.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    // tiles_enabled_flag == 1: the entry-point block is present.
    h.ue(offsets.len() as u32); // num_entry_point_offsets
    if !offsets.is_empty() {
        h.ue(u32::from(offset_len) - 1); // offset_len_minus1
        for &o in &offsets {
            h.put_bits(o, offset_len); // entry_point_offset_minus1[i]
        }
    }
    h.rbsp_trailing_bits(); // byte_alignment() before slice data
    debug_assert_ne!(h.as_bytes().last(), Some(&0), "aligned header byte");
    for s in &subsets {
        for &b in s {
            h.put_bits(u32::from(b), 8);
        }
    }
    h.finish()
}

/// Encode one 4:2:0 8-bit frame as a self-contained IDR access unit
/// (`VPS + SPS + PPS + IDR_N_LP` in Annex B form). The decode is
/// bit-exact lossless: every coding unit is a §7.3.8.7 PCM block.
///
/// # Errors
/// [`PcmEncodeError`] when the dimensions are not nonzero multiples of
/// 16 or a plane buffer has the wrong length.
pub fn encode_idr_pcm_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, PcmEncodeError> {
    encode_au(y, cb, cr, width, height, PcmAuOptions::default())
}

/// As [`encode_idr_pcm_au`], with the picture split into `segments`
/// slice segments: the first independent, the rest §7.3.6.1
/// *dependent* slice segments (`dependent_slice_segment_flag == 1`,
/// header inherited per §7.4.7.1, CABAC contexts carried across the
/// boundary per §9.3.2.4 / §9.3.2.5).
///
/// # Errors
/// [`PcmEncodeError`] as for [`encode_idr_pcm_au`], or
/// [`PcmEncodeError::BadDimensions`] when `segments` is 0 or exceeds
/// the picture's CTB count.
pub fn encode_idr_pcm_au_segmented(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    segments: usize,
) -> Result<Vec<u8>, PcmEncodeError> {
    encode_au(
        y,
        cb,
        cr,
        width,
        height,
        PcmAuOptions {
            segments,
            ..PcmAuOptions::default()
        },
    )
}

/// As [`encode_idr_pcm_au`], with the full [`PcmAuOptions`] shape
/// control (slice segmentation, deblocking on, luma band-offset SAO).
///
/// # Errors
/// [`PcmEncodeError`] as for [`encode_idr_pcm_au_segmented`].
pub fn encode_idr_pcm_au_opts(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    opts: PcmAuOptions,
) -> Result<Vec<u8>, PcmEncodeError> {
    encode_au(y, cb, cr, width, height, opts)
}

fn encode_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    opts: PcmAuOptions,
) -> Result<Vec<u8>, PcmEncodeError> {
    let segments = opts.segments;
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    let total_ctbs = (width / CTB) * (height / CTB);
    if segments == 0 || segments > total_ctbs {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    // Independent-slice plans must be sorted, start at CTB 0 and stay
    // in range; band and edge SAO are mutually exclusive.
    if !opts.independent_slices.is_empty() {
        let starts: Vec<usize> = opts.independent_slices.iter().map(|&(s, _)| s).collect();
        if starts[0] != 0
            || starts.windows(2).any(|w| w[1] <= w[0])
            || starts.iter().any(|&s| s >= total_ctbs)
        {
            return Err(PcmEncodeError::BadDimensions { width, height });
        }
    }
    if opts.sao_luma_band && opts.sao_luma_eo_vertical {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    // Tile grid: at least two tiles, each column/row at least one CTB
    // wide/tall; the tiled picture is a single slice segment with SAO
    // off (the tile-scan SAO merge availability is out of this
    // bootstrap's scope).
    if let Some((cols, rows)) = opts.tiles {
        let sao = opts.sao_luma_band || opts.sao_luma_eo_vertical;
        if cols == 0
            || rows == 0
            || cols * rows < 2
            || cols as usize > width / CTB
            || rows as usize > height / CTB
            || segments != 1
            || !opts.independent_slices.is_empty()
            || sao
        {
            return Err(PcmEncodeError::BadDimensions { width, height });
        }
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
        if buf.len() != expected {
            Err(PcmEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        } else {
            Ok(())
        }
    };
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let level_idc = opts.tiles.map_or_else(
        || level_idc_for(width * height),
        |(c, r)| level_idc_for(width * height).max(level_idc_for_tiles(c, r)),
    );
    let dependent_mode = opts.independent_slices.is_empty() && segments > 1;
    let sao = opts.sao_luma_band || opts.sao_luma_eo_vertical;
    let mut units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(
            33,
            0,
            0,
            &write_sps(width, height, level_idc, sao, opts.pcm_loop_filter_disabled),
        ), // SPS_NUT
        nal_unit(
            34,
            0,
            0,
            &write_pps(dependent_mode, opts.deblocking, opts.tiles),
        ), // PPS_NUT
    ];
    if opts.tiles.is_some() {
        let rbsp = write_tiled_idr_slice(y, cb, cr, width, height, &opts);
        units.push(nal_unit(20, 0, 0, &rbsp)); // IDR_N_LP
    } else {
        for rbsp in write_idr_slice_segments(y, cb, cr, width, height, &opts) {
            units.push(nal_unit(20, 0, 0, &rbsp)); // IDR_N_LP
        }
    }
    Ok(annexb(&units))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::pps::PicParameterSet;
    use crate::hevc::engine::sps::SeqParameterSet;

    fn gradient_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        let cb: Vec<u8> = (0..w * h / 4).map(|i| (i * 3 % 240 + 8) as u8).collect();
        let cr: Vec<u8> = (0..w * h / 4).map(|i| (255 - i * 5 % 250) as u8).collect();
        (y, cb, cr)
    }

    #[test]
    fn rejects_bad_geometry() {
        let (y, cb, cr) = gradient_planes(16, 16);
        assert!(matches!(
            encode_idr_pcm_au(&y, &cb, &cr, 20, 16),
            Err(PcmEncodeError::BadDimensions { .. })
        ));
        assert!(matches!(
            encode_idr_pcm_au(&y, &cb, &cr, 32, 16),
            Err(PcmEncodeError::PlaneSize { .. })
        ));
    }

    /// The whole bootstrap contract: encode → decode with the crate's
    /// own end-to-end driver → bit-exact planes (PCM is lossless).
    #[test]
    fn pcm_au_roundtrips_losslessly_through_the_decoder() {
        for (w, h) in [(16usize, 16usize), (48, 32), (32, 64)] {
            let (y, cb, cr) = gradient_planes(w, h);
            let au = encode_idr_pcm_au(&y, &cb, &cr, w, h).expect("encode");
            let frames =
                crate::hevc::engine::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1, "{w}x{h}: one IDR frame");
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "{w}x{h}: lossless PCM roundtrip"
            );
        }
    }

    /// Two frames form two self-contained IDR AUs; both decode losslessly
    /// and in order.
    #[test]
    fn multi_frame_pcm_stream_decodes_in_order() {
        let (y0, cb0, cr0) = gradient_planes(32, 32);
        let y1: Vec<u8> = y0.iter().map(|&v| v ^ 0x5A).collect();
        let mut stream = encode_idr_pcm_au(&y0, &cb0, &cr0, 32, 32).expect("au0");
        stream.extend(encode_idr_pcm_au(&y1, &cb0, &cr0, 32, 32).expect("au1"));
        let frames =
            crate::hevc::engine::sequence::decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 2);
        let p0 = frames[0].picture.to_planar_u8().unwrap();
        let p1 = frames[1].picture.to_planar_u8().unwrap();
        assert_eq!(&p0[..y0.len()], &y0[..]);
        assert_eq!(&p1[..y1.len()], &y1[..]);
        assert_ne!(p0, p1);
    }

    /// Dependent slice segments: the picture splits into an independent
    /// segment + dependent segments whose CABAC contexts continue
    /// across the NAL boundary (§9.3.2.4 / §9.3.2.5) and whose header
    /// values inherit per §7.4.7.1 — still bit-exact lossless.
    #[test]
    fn dependent_slice_segments_roundtrip_losslessly() {
        let (w, h) = (48usize, 48usize); // 9 CTBs
        let (y, cb, cr) = gradient_planes(w, h);
        for segments in [2usize, 3, 9] {
            let au = encode_idr_pcm_au_segmented(&y, &cb, &cr, w, h, segments).expect("encode");
            // One IDR NAL per segment behind the parameter sets.
            let units = crate::hevc::engine::nal::collect_nal_units(&au).expect("walk");
            assert_eq!(units.len(), 3 + segments, "{segments} segments");
            assert!(units[4..].iter().all(|u| u.header.nal_unit_type == 20));
            let frames =
                crate::hevc::engine::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1);
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "{segments}-segment lossless roundtrip"
            );
        }
    }

    /// A dependent segment decoded WITHOUT the preceding segment's
    /// stored context state is rejected, not misdecoded.
    #[test]
    fn dependent_segment_without_predecessor_is_rejected() {
        let (y, cb, cr) = gradient_planes(32, 32);
        let au = encode_idr_pcm_au_segmented(&y, &cb, &cr, 32, 32, 2).expect("encode");
        let units = crate::hevc::engine::nal::collect_nal_units(&au).expect("walk");
        // Drop the independent slice segment (unit 3), keep the
        // dependent one: the driver must refuse.
        let mut broken = Vec::new();
        for (i, u) in units.iter().enumerate() {
            if i == 3 {
                continue;
            }
            let mut coded = vec![0, 0, 0, 1];
            coded.extend(crate::hevc::engine::encoder::nal::nal_unit(
                u.header.nal_unit_type,
                u.header.nuh_layer_id,
                u.header.temporal_id,
                &u.rbsp,
            ));
            broken.extend(coded);
        }
        assert!(crate::hevc::engine::sequence::decode_annexb_sequence(&broken).is_err());
    }

    /// §8.7.2.5.4 / §8.7.3.1 — with deblocking and luma band-offset
    /// SAO ENABLED over an all-PCM picture and
    /// `pcm_loop_filter_disabled_flag == 1`, a conforming decoder must
    /// leave every reconstructed sample untouched. This pins the
    /// per-CU loop-filter suppression map end to end (a decoder that
    /// filtered the PCM samples would corrupt the smooth gradient).
    #[test]
    fn pcm_loop_filter_suppression_keeps_samples_exact() {
        let (w, h) = (48usize, 32usize);
        let (y, cb, cr) = gradient_planes(w, h);
        for (deblocking, sao_luma_band) in [(true, false), (false, true), (true, true)] {
            let opts = PcmAuOptions {
                segments: 1,
                deblocking,
                sao_luma_band,
                ..PcmAuOptions::default()
            };
            let au = encode_idr_pcm_au_opts(&y, &cb, &cr, w, h, opts).expect("encode");
            let frames =
                crate::hevc::engine::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1);
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "deblocking={deblocking} sao={sao_luma_band}: PCM samples survive the loop filters"
            );
        }
    }

    /// Independent multi-slice pictures with heterogeneous per-slice
    /// `slice_loop_filter_across_slices_enabled_flag` values roundtrip
    /// through the crate's own decoder (which implements the §8.7.2.1
    /// per-slice gate), and the opposite flag layouts produce
    /// different pictures (the boundary edge filters on exactly one
    /// side of the pair).
    #[test]
    fn independent_slices_with_heterogeneous_across_flags() {
        let (w, h) = (48usize, 32usize);
        let (y, cb, cr) = gradient_planes(w, h);
        let mut decoded = Vec::new();
        for flags in [(true, false), (false, true)] {
            let opts = PcmAuOptions {
                independent_slices: vec![(0, flags.0), (3, flags.1)],
                deblocking: true,
                pcm_loop_filter_disabled: false,
                ..PcmAuOptions::default()
            };
            let au = encode_idr_pcm_au_opts(&y, &cb, &cr, w, h, opts).expect("encode");
            let frames =
                crate::hevc::engine::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1);
            decoded.push(frames[0].picture.to_planar_u8().expect("8-bit"));
        }
        assert_ne!(
            decoded[0], decoded[1],
            "opposite per-slice flags gate the boundary deblocking differently"
        );
    }

    /// True multi-tile single-slice streams: the picture walks the
    /// §6.5.1 tile scan inside ONE slice segment, every tile its own
    /// §7.3.8.1 subset behind a §7.4.7.1 entry-point offset, with the
    /// §9.3.2.2 per-tile CABAC context re-initialization on both the
    /// encode and decode sides — bit-exact lossless through the
    /// crate's own decoder.
    #[test]
    fn tiled_pcm_au_roundtrips_losslessly() {
        for (w, h, cols, rows) in [
            (64usize, 48usize, 2u32, 2u32), // uneven rows: 4x3 CTBs / 2x2
            (96, 64, 3, 2),                 // 6x4 CTBs / 3x2
            (32, 16, 2, 1),                 // single tile row
            (80, 80, 5, 5),                 // one CTB per tile
        ] {
            let (y, cb, cr) = gradient_planes(w, h);
            let opts = PcmAuOptions {
                tiles: Some((cols, rows)),
                ..PcmAuOptions::default()
            };
            let au = encode_idr_pcm_au_opts(&y, &cb, &cr, w, h, opts).expect("encode");
            let frames =
                crate::hevc::engine::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1, "{cols}x{rows}: one IDR frame");
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "{w}x{h} {cols}x{rows} tiles: lossless roundtrip"
            );
        }
    }

    /// The tiled stream's parameter sets and slice header really carry
    /// the tile syntax: `tiles_enabled_flag == 1`, the uniform 2x2
    /// grid, and `num_entry_point_offsets == 3` with offsets that
    /// partition the coded slice data.
    #[test]
    fn tiled_pcm_au_signals_tiles_and_entry_points() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = gradient_planes(w, h);
        let opts = PcmAuOptions {
            tiles: Some((2, 2)),
            ..PcmAuOptions::default()
        };
        let au = encode_idr_pcm_au_opts(&y, &cb, &cr, w, h, opts).expect("encode");
        let units = crate::hevc::engine::nal::collect_nal_units(&au).expect("walk");
        assert_eq!(units.len(), 4);
        let sps = SeqParameterSet::parse(&units[1].rbsp).expect("sps");
        let pps = PicParameterSet::parse(&units[2].rbsp).expect("pps");
        assert!(pps.tiles_enabled_flag);
        assert_eq!(pps.tiles.num_tile_columns_minus1, 1);
        assert_eq!(pps.tiles.num_tile_rows_minus1, 1);
        assert!(pps.tiles.uniform_spacing_flag);
        assert!(!pps.loop_filter_across_tiles_enabled_flag);
        let header = crate::hevc::engine::slice::SliceSegmentHeader::parse(
            &units[3].rbsp,
            units[3].header.nal_unit_type,
            &sps,
            &pps,
        )
        .expect("slice header");
        let eps = header.entry_point_offsets.expect("entry points present");
        assert_eq!(eps.num_entry_point_offsets, 3);
        assert_eq!(eps.entry_point_offset_minus1.len(), 3);
    }

    /// Tile-option validation: zero-sized grids, single-tile grids,
    /// grids finer than the CTB grid, and combinations with
    /// multi-segment or SAO options are rejected.
    #[test]
    fn tiled_pcm_au_rejects_bad_grids() {
        let (y, cb, cr) = gradient_planes(32, 32);
        for opts in [
            PcmAuOptions {
                tiles: Some((1, 1)),
                ..PcmAuOptions::default()
            },
            PcmAuOptions {
                tiles: Some((0, 2)),
                ..PcmAuOptions::default()
            },
            PcmAuOptions {
                tiles: Some((3, 1)), // 2x2 CTBs: only 2 columns exist
                ..PcmAuOptions::default()
            },
            PcmAuOptions {
                tiles: Some((2, 2)),
                segments: 2,
                ..PcmAuOptions::default()
            },
            PcmAuOptions {
                tiles: Some((2, 2)),
                sao_luma_band: true,
                ..PcmAuOptions::default()
            },
        ] {
            assert!(
                matches!(
                    encode_idr_pcm_au_opts(&y, &cb, &cr, 32, 32, opts.clone()),
                    Err(PcmEncodeError::BadDimensions { .. })
                ),
                "{opts:?}"
            );
        }
    }

    #[test]
    fn written_sps_pps_parse_back() {
        let (y, cb, cr) = gradient_planes(48, 32);
        let au = encode_idr_pcm_au(&y, &cb, &cr, 48, 32).expect("encode");
        let units = crate::hevc::engine::nal::collect_nal_units(&au).expect("walk");
        assert_eq!(units.len(), 4);
        let sps = SeqParameterSet::parse(&units[1].rbsp).expect("sps parses");
        assert_eq!(sps.pic_width_in_luma_samples, 48);
        assert_eq!(sps.pic_height_in_luma_samples, 32);
        assert_eq!(sps.chroma_format_idc, 1);
        let pcm = sps.pcm.as_ref().expect("pcm block present");
        assert_eq!(pcm.bit_depth_luma_minus1, 7);
        assert!(pcm.loop_filter_disabled_flag);
        let pps = PicParameterSet::parse(&units[2].rbsp).expect("pps parses");
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(pps.deblocking.disabled_flag);
    }
}
