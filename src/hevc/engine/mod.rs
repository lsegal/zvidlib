//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** the decoder is end-to-end. [`decode_annexb_sequence`] /
//! [`SequenceDecoder`] decode whole Annex B byte streams to
//! output-order pictures, and [`make_decoder`] exposes the same driver
//! through the [`oxideav_core::Decoder`] registry contract (registered
//! by [`register`] under `"h265"` / `"hevc"`, accepting Annex B and
//! `hvcC` / length-prefixed transport). Every Annex B bitstream in the
//! staged 16-fixture conformance corpus decodes byte-exact, plus
//! self-built pins for explicit weighted prediction, PCM, dependent
//! slice segments and per-slice loop-filter flags. [`make_encoder`] is
//! the PCM-only IDR encoder bootstrap (lossless, conformant, every
//! packet a random access point) over the write-side stack in
//! [`encoder`] (bit writer, NAL encapsulation, the §9.3.5 CABAC
//! encoding engine, parameter-set / slice writers). See `README.md`
//! for coverage and the remaining gaps (true multi-tile fixtures,
//! encoder beyond the PCM bootstrap).
//!
//! The sections below record the per-round rebuild history of the
//! subsystems the driver composes.
//!
//! **History:** clean-room rebuild (post 2026-05-18 audit).
//! The latest round adds the §8.6.2 / §8.6.3 / §8.6.4 scaling,
//! transformation and residual-array construction step — the new
//! [`transform`] module. [`transform::scale_coefficients`] implements
//! the §8.6.3 dequantization (the `levelScale` / `m[x][y]` /
//! `1 << (qP/6)` product, `bdShift` offset-round and
//! `[coeffMin, coeffMax]` clip of equations 8-300..8-309);
//! [`transform::inverse_transform`] implements the §8.6.4 separable
//! inverse transform (the equation-8-316 4x4 DST-VII for `MODE_INTRA`
//! 4x4 luma and the equations-8-318..8-321 32x32 DCT-II with the
//! equation-8-317 column subsampling for every other block, plus the
//! equation-8-314 intermediate offset-round); and
//! [`transform::residual_block`] orchestrates the §8.6.2 dispatch over
//! `cu_transquant_bypass_flag` (the equation-8-297 `rotateCoeffs`
//! pass-through), `transform_skip_flag` (the equation-8-298 `tsShift`
//! left-shift), and the full scale-then-transform path, applying the
//! equation-8-299 final `bdShift` offset-round.
//!
//! Round 12 finishes the §7.3.2.1 VPS tail through the optional VPS
//! timing-info block ([`vps::HevcVps`] now carries `max_layer_id`,
//! `num_layer_sets_minus1`, the `layer_id_included_flag[][]`
//! inclusion matrix as [`vps::LayerIdInclusionRow`] rows, the
//! `vps_timing_info_present_flag` block as [`vps::VpsTimingInfo`] —
//! `u(32)` `num_units_in_tick` / `time_scale`,
//! `poc_proportional_to_timing_flag` +
//! `num_ticks_poc_diff_one_minus1`, and `num_hrd_parameters` — plus
//! `vps_extension_flag`); per-HRD `hrd_parameters()` bodies and the
//! extension-data payload are surfaced as
//! [`vps::HevcVps::opaque_tail`].
//!
//! Round 11 landed the §9.3 CABAC arithmetic decoding engine
//! ([`cabac::CabacEngine`] / [`cabac::ContextModel`] / [`cabac::init_type`]):
//! the §9.3.2.6 engine-register init, the §9.3.2.2 context-variable init
//! (equations 9-4..9-7), the §9.3.4.3.2 DecodeDecision primitive (with
//! the Table 9-52 / Table 9-53 LPS-range / state-transition tables), the
//! §9.3.4.3.3 RenormD loop, the §9.3.4.3.4 DecodeBypass primitive (with
//! an MSB-first `decode_bypass_bits(n)` helper), the §9.3.4.3.5
//! DecodeTerminate primitive, and the §9.3.4.3.6 aligned-bypass
//! alignment hook. The engine ships standalone — independent of the
//! §9.3.4.2 per-syntax-element binarization / context-index derivation
//! that the slice-data parser still needs.
//!
//! Rounds 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 land the Annex B NAL-unit
//! byte-stream walker, the §7.3.1.2 NAL header parse, the §7.3.2.1
//! VPS structural parse (with a §7.3.3 profile_tier_level walk), the
//! full §7.3.2.2 SPS parse (through the `vui_parameters_present_flag`
//! / `sps_extension_present_flag` gates, with the VUI body and any
//! extension payload surfaced as an opaque-bytes tail), the
//! §7.3.2.3.1 PPS parse (full general body through
//! `pps_extension_present_flag`, including the tiles and
//! deblocking-control blocks; the PPS extension bodies are surfaced as
//! an opaque tail), the §7.3.6.1 slice-segment-header parse —
//! independent I-slice IDR segments end to end (round 6), and
//! independent **non-IDR I-slice** segments through the §7.3.6.1 POC +
//! short-term-RPS + long-term-RPS block end to end (round 7) — and
//! now (round 8) the §7.3.4 `scaling_list_data()` parse with the
//! §7.4.5 `ScalingList[sizeId][matrixId][i]` derivation, wired into
//! both the SPS (`sps_scaling_list_data_present_flag`) and PPS
//! (`pps_scaling_list_data_present_flag`) paths. The P/B
//! reference-list / weighted-prediction sub-structures are still
//! surfaced as an opaque tail.
//!
//! ## What works today
//!
//! * Annex B byte-stream splitting (3- and 4-byte start codes,
//!   trailing-zero padding tolerance).
//! * §7.3.1.2 NAL header parse: `forbidden_zero_bit`,
//!   `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived
//!   from `nuh_temporal_id_plus1`).
//! * §7.4.1.1 emulation-prevention byte strip (`0x00 0x00 0x03` →
//!   `0x00 0x00`).
//! * MSB-first bit reader with `u(n)` and 0-th-order
//!   unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors.
//! * §7.3.2.1 [`vps::HevcVps`] — vps_id, base-layer / max-layers /
//!   sub-layers / temporal-nesting flags, reserved-0xFFFF validation,
//!   the §7.3.3 profile_tier_level walk (general profile + level +
//!   per-sub-layer present-flag gates and `sub_layer_level_idc`), and
//!   the per-sub-layer DPB / reorder / latency triple loop.
//! * §7.3.2.2 [`sps::SeqParameterSet`] — vps-id back-reference,
//!   max-sub-layers / nesting flag, the §7.3.3 PTL re-walk,
//!   `chroma_format_idc` / `separate_colour_plane_flag`,
//!   `pic_width_in_luma_samples` / `pic_height_in_luma_samples`,
//!   conformance-window quad, `bit_depth_{luma,chroma}_minus8`,
//!   `log2_max_pic_order_cnt_lsb_minus4`, the per-sub-layer
//!   DPB / reorder / latency triple loop, the four
//!   `log2_*_block_size{_minus_2,_minus_3,_diff_max_min}` fields,
//!   `max_transform_hierarchy_depth_{inter,intra}`,
//!   `scaling_list_enabled_flag` (with the nested
//!   `sps_scaling_list_data_present_flag` / [`scaling_list::ScalingListData`]
//!   §7.3.4 block), `amp_enabled_flag`,
//!   `sample_adaptive_offset_enabled_flag`, the [`sps::PcmInfo`] block
//!   gated by `pcm_enabled_flag`, the
//!   `num_short_term_ref_pic_sets` ue(v) + per-set
//!   [`sps::ShortTermRefPicSet`] (§7.3.7, both explicit and
//!   inter-RPS-prediction forms), the
//!   `long_term_ref_pics_present_flag` block plus
//!   [`sps::LongTermRefPicEntry`] table, the
//!   `sps_temporal_mvp_enabled_flag` /
//!   `strong_intra_smoothing_enabled_flag` pair, the
//!   `vui_parameters_present_flag` gate whose §E.2.1
//!   `vui_parameters()` body is decoded into [`vui::VuiParameters`]
//!   (aspect-ratio / EXTENDED_SAR, overscan, video-signal-type +
//!   colour-description, chroma-loc, default-display-window, the
//!   `vui_timing_info` block — `u(32)` num_units_in_tick / time_scale
//!   plus the nested §E.2.3 `hrd_parameters()` call — and
//!   bitstream-restriction), and the `sps_extension_present_flag`
//!   gate whose extension body is surfaced as [`sps::OpaqueTail`].
//!   The §7.3.4 `scaling_list_data()` block — when
//!   `sps_scaling_list_data_present_flag == 1` — is parsed and the
//!   §7.4.5 `ScalingList[sizeId][matrixId][i]` coefficient arrays are
//!   derived (default tables + prediction inference); see
//!   [`scaling_list::ScalingListData`].
//! * §6.5 [`scan`] — all four scan-order initialization processes plus
//!   the §7.4.2 [`scan::scan_order`] `ScanOrder[log2BlockSize][scanIdx]`
//!   accessor: [`scan::up_right_diagonal`] (§6.5.3, equation 6-11),
//!   [`scan::horizontal`] (§6.5.4, equation 6-12),
//!   [`scan::vertical`] (§6.5.5, equation 6-13), and
//!   [`scan::traverse`] (§6.5.6, equation 6-14, the boustrophedon
//!   raster). [`scan::scan_order`] enforces §7.4.2's populated ranges
//!   (`log2BlockSize` 0..=3 for diagonal / horizontal / vertical, 2..=5
//!   for traverse). §7.4.5
//!   [`scaling_list::ScalingListData::scaling_factors`] expands the
//!   flat scaling lists into the two-dimensional
//!   `ScalingFactor[sizeId][matrixId][x][y]` quantization matrices
//!   (equations 7-44..7-51: the diagonal scatter, the 2x / 4x block
//!   replication, the DC `[0][0]` override, and the
//!   `ChromaArrayType == 3` 32x32-chroma derivation).
//! * §7.3.2.3.1 [`pps::PicParameterSet`] — the full general
//!   `pic_parameter_set_rbsp()` body: the `pps_*_id` pair, the
//!   slice-header gates, `init_qp_minus26` (`se(v)`), the chroma QP
//!   offsets, the tiles block ([`pps::TileInfo`] — column/row counts
//!   plus the explicit `column_width_minus1[]` / `row_height_minus1[]`
//!   arrays when `uniform_spacing_flag == 0`), the
//!   deblocking-filter-control block ([`pps::DeblockingFilterControl`]),
//!   `lists_modification_present_flag`,
//!   `log2_parallel_merge_level_minus2`, and the
//!   `pps_extension_present_flag` gate. When
//!   `pps_extension_present_flag == 1` the eight bits of typed
//!   extension flags are decoded into [`pps::PpsExtensionFlags`]
//!   (`pps_range_extension_flag`, `pps_multilayer_extension_flag`,
//!   `pps_3d_extension_flag`, `pps_scc_extension_flag`, and the
//!   reserved `pps_extension_4bits`); any extension body whose flag
//!   is set is surfaced as a shared [`sps::OpaqueTail`] starting at
//!   the first body's bit position. When
//!   `pps_scaling_list_data_present_flag == 1` the §7.3.4
//!   `scaling_list_data()` block is parsed into
//!   [`scaling_list::ScalingListData`]. The §7.4.3.3.1 inference rules
//!   are applied so absent conditional fields carry their effective
//!   value.
//! * §7.3.6.1 [`slice::SliceSegmentHeader`] — the
//!   `slice_segment_header()` parse for an independent slice segment,
//!   taking the activated SPS + PPS as context (the
//!   `slice_segment_address` and `slice_pic_order_cnt_lsb` widths plus
//!   the SAO / MVP / tiles gates are SPS/PPS-derived). Independent
//!   **I-slice** segments — both IDR and non-IDR — parse end to end
//!   through `byte_alignment()`, including the §7.3.6.1 non-IDR POC
//!   (`slice_pic_order_cnt_lsb`) + short-term-RPS
//!   (`short_term_ref_pic_set_sps_flag` /
//!   in-line `st_ref_pic_set(num_short_term_ref_pic_sets)` via
//!   [`sps::ShortTermRefPicSet::parse_slice_inline`] /
//!   `short_term_ref_pic_set_idx`) + long-term-RPS block (per-entry
//!   SPS-indexed vs in-slice + `delta_poc_msb_present_flag` /
//!   `delta_poc_msb_cycle_lt`, surfaced as
//!   [`slice::SliceLongTermRefPic`]). The P/B reference-list /
//!   weighted-prediction sub-structures are still surfaced as an
//!   [`sps::OpaqueTail`]. The §7.4.7.1 inference rules are applied to
//!   absent fields.
//! * §7.3.6.2 [`slice::RefPicListsModification`] — the
//!   `ref_pic_lists_modification()` syntax structure as a standalone
//!   parser. The parser walks the
//!   `ref_pic_list_modification_flag_lX` `u(1)` gates and the
//!   `list_entry_lX[]` `u(v)` loops (each entry
//!   `Ceil( Log2( NumPicTotalCurr ) )` bits wide and range-checked
//!   per §7.4.7.2); the implicit `RefPicListTempX` derivation of
//!   §8.3.4 stays the consumer's responsibility.
//! * §7.4.7.2 [`slice::NumPicTotalCurrInputs`] — the
//!   `NumPicTotalCurr` derivation (equation 7-57) as a small typed
//!   builder taking the per-position `UsedByCurrPicS0` /
//!   `UsedByCurrPicS1` / `UsedByCurrPicLt` flags from the active
//!   short-term RPS + the slice's long-term ref list and the
//!   `pps_curr_pic_ref_enabled_flag` closing-clause flag, returning
//!   the typed `NumPicTotalCurr: u32`. A
//!   [`slice::NumPicTotalCurrInputs::from_explicit_short_term_rps`]
//!   convenience constructor sources `S0` / `S1` straight off an
//!   explicit-form [`sps::ShortTermRefPicSet`]; the
//!   inter-RPS-prediction form needs the §7.4.8 derivation to run
//!   first.
//!   [`slice::SliceLongTermRefPic::used_by_curr_pic_lt`] resolves
//!   each long-term entry's `UsedByCurrPicLt[i]` per §7.4.7.1
//!   (SPS-table lookup for SPS-resident entries, direct flag for
//!   in-slice entries). The F.7.4.7.2 multilayer-extension form
//!   (equation F-56) is reachable through
//!   [`slice::NumPicTotalCurrInputs::with_multilayer_extension`].
//! * §7.3.6.3 [`slice::PredWeightTable`] — the
//!   `pred_weight_table()` syntax structure as a standalone parser.
//!   The parser walks the `luma_log2_weight_denom` /
//!   `delta_chroma_log2_weight_denom` denominators, the two flag passes
//!   (`luma_weight_lX_flag[i]` + `chroma_weight_lX_flag[i]`), and the
//!   per-reference delta block (`delta_luma_weight_lX[i]` /
//!   `luma_offset_lX[i]` / `delta_chroma_weight_lX[i][j]` /
//!   `delta_chroma_offset_lX[i][j]`), applying the §7.4.7.3 range
//!   bounds + the per-i §7.3.6.3 outer-gate (`pic_layer_id !=
//!   nuh_layer_id || PicOrderCnt(RefPicListX[i]) !=
//!   PicOrderCnt(CurrPic)`) decision supplied by the caller, the
//!   `ChromaLog2WeightDenom ∈ 0..=7` derived range, and the
//!   `sumWeightLXFlags ≤ 24` conformance cap.
//!   [`slice::PredWeightTable::luma_weight_l0`] /
//!   [`slice::PredWeightTable::chroma_weight_l0`] (mirrored for L1)
//!   resolve each derived `LumaWeightLX[i]` /
//!   `ChromaWeightLX[i][j]`; [`slice::PredWeightTable::chroma_offset_l0`]
//!   (mirrored) applies equation 7-58 for `ChromaOffsetLX[i][j]`.
//!
//! See [`nal`] for the byte-stream walker entry points, [`vps`] for
//! the parsed VPS structure, [`sps`] for the parsed SPS, [`pps`]
//! for the parsed PPS, and [`crate::hevc::engine::slice`] for the parsed slice
//! header.

#![warn(missing_debug_implementations)]
#![allow(dead_code, unused_imports)]

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod availability;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod binarization;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod bitreader;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod cabac;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod ctx_init;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod deblock;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod decode;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod dpb;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod hrd;
pub mod hvcc;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod inter_pred;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod inter_recon;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod intra_mode_field;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod intra_pred;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod motion;
pub mod nal;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod palette;
pub mod picture;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod poc;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod pps;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod pu_mv;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod recon;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod residual;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod sao;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod scaling_list;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod scan;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod sei;
pub mod sequence;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod slice;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod slice_data;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod sps;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod transform;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod transform_tree;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod transform_unit;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod vps;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod vui;

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use availability::{AvailabilityError, PictureTiling, TilingParams};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use bitreader::{BitReader, BitReaderError};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use cabac::{CabacEngine, CabacError, ContextModel, init_type};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use ctx_init::SliceContexts;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use deblock::{
    BoundaryStrength, DeblockCu, DeblockCuDesc, DeblockCuParams, EdgeFlags, EdgeType, NoFilterMap,
    TransformSplit, deblock_picture, deblock_picture_full, derive_boundary_strength,
    derive_edge_flags, filter_cu_edges, filter_cu_edges_full,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use decode::{PictureHeaderInfo, PictureRefState, PictureSequenceState, SliceRefParams};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use dpb::{
    Dpb, DpbEntry, LongTermEntry, Marking, RefPicListParams, RefPicLists, ResolvedRps, RpsPocLists,
    build_rps_poc_lists, no_backward_pred_flag, select_col_pic,
};
pub use hrd::HrdError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use hrd::{
    CpbEntry, HEVC_MAX_CPB_CNT, HEVC_MAX_ELEMENTAL_DURATION_IN_TC_MINUS1, HrdCommonInfo,
    HrdParameters, SubLayerHrd, SubLayerHrdParameters, VpsHrdEntry,
};
pub use hvcc::{
    HvccError, HvccRecord, extradata_is_hvcc, nal_unit_from_coded, parse_hvcc,
    split_length_prefixed,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use inter_pred::{
    InterPredError, InterPredGeometry, InterPrediction, ListPrediction, MotionVector, PuWeights,
    RefPlane, WpListWeights, default_weighted_pred, explicit_weighted_pred, interp_chroma_block,
    interp_luma_block, predict_inter_pu, predict_inter_pu_weighted,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use inter_recon::SliceWpTables;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use intra_mode_field::{IntraModeField, MIN_BLOCK_LOG2, MIN_BLOCK_SIZE, Neighbour};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use motion::{
    MergeCandidate, MergeListParams, MotionCell, MotionField, Mv, MvpContext, NeighbourPu,
    PartitionContext, RefPicId, SpatialMergeCandidates, SpatialMergeNeighbours, TemporalMvContext,
    append_combined_bi_candidates, append_zero_merge_candidates, build_merge_candidate,
    derive_chroma_mv, derive_mvp_candidate, derive_spatial_merge_candidates, derive_temporal_mv,
    reconstruct_mv,
};
pub use nal::{NalError, NalHeader, NalIter, NalUnit, collect_nal_units};
pub use picture::{Picture, Plane, clip1, sub_wh_c};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use poc::{NalKind, PicOrderCnt, PocState, diff_pic_order_cnt};
pub use pps::PpsError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use pps::{
    ChromaQpOffsetListEntry, DeblockingFilterControl, PicParameterSet, PpsRangeExtension, TileInfo,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use pu_mv::{
    InterCuDesc, PartMode as PuPartMode, PuGeometry, PuMotion, PuMvContext, PuRect, pu_partitions,
    resolve_cu_motion, resolve_pu_motion,
};
pub use recon::ReconError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use recon::{
    IntraPictureParams, PlacedCtu, ReconCtx, ReconParams, ResolvedList, SliceSegmentBoundary,
    build_slice_addr_map, reconstruct_inter_pu, reconstruct_inter_pu_weighted,
    reconstruct_intra_ctu, reconstruct_intra_ctu_ctx, reconstruct_intra_picture,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use sao::{
    ResolvedSao, ResolvedSaoComponent, apply_sao_ctb, apply_sao_picture, apply_sao_picture_full,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use scaling_list::{
    MAX_COEF_NUM, NUM_MATRIX_IDS, NUM_SIZE_IDS, ScalingFactorMatrix, ScalingFactors,
    ScalingListData, ScalingListError, ScalingListMatrix,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use scan::{
    ScanIdx, ScanOrderError, ScanPos, horizontal, scan_order, traverse, up_right_diagonal, vertical,
};
pub use sequence::{DecodedFrame, SequenceDecoder, SequenceError, decode_annexb_sequence};
pub use slice::SliceError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use slice::{
    BLA_W_LP, EntryPointOffsets, IDR_N_LP, IDR_W_RADL, NumPicTotalCurrInputs, PredWeightEntry,
    PredWeightTable, PredWeightTableInputs, RSV_IRAP_VCL23, RefPicListsModification,
    SliceDeblocking, SliceLongTermRefPic, SliceLongTermRefPicSource, SliceSegmentHeader, SliceType,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use slice_data::{
    CodingQuadtree, CodingTreeUnit, CodingUnit, IntraLumaMode, PcmSamples, PictureParseState,
    PredictionUnit, SaoComponent, SaoCtbParams, SliceDataParams, decode_coding_quadtree,
    decode_coding_tree_unit, decode_coding_tree_unit_in_picture, decode_sao,
};
pub use sps::{ShortTermRefPicSetMaterializeError, SpsError};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use sps::{
    ConformanceWindow, HEVC_MAX_NUM_LONG_TERM_RPS, HEVC_MAX_NUM_SHORT_TERM_RPS, HEVC_MAX_RPS_PICS,
    LongTermRefPicEntry, MaterializedShortTermRefPicSet, OpaqueTail, PcmInfo, SeqParameterSet,
    ShortTermRefPicSet, SpsExtensionFlags, SpsRangeExtension,
};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use transform_tree::{TransformTree, TransformTreeParams, decode_transform_tree};
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use transform_unit::{
    CuPredMode, QuantGroupState, TransformUnit, TransformUnitParams, decode_transform_unit,
};
pub use vps::VpsError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use vps::{
    HEVC_MAX_SUB_LAYERS, HEVC_VPS_MAX_NUM_LAYER_SETS, HEVC_VPS_MAX_NUM_LAYERS, HevcVps,
    LayerIdInclusionRow, ProfileTierLevel, SubLayerOrderingInfo, VpsTimingInfo,
};
pub use vui::VuiError;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use vui::{
    BitstreamRestriction, ColourDescription, DefaultDisplayWindow, EXTENDED_SAR, VideoSignalType,
    VuiParameters, VuiTimingInfo,
};

/// Crate-local error type for the structural utilities (the NAL
/// walker and parameter-set parsers surface their own [`NalError`] /
/// [`VpsError`] types directly; the decode drivers use
/// [`sequence::SequenceError`] and the registry decoder maps into
/// [`oxideav_core::Error`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The requested functionality is not implemented (the encoder,
    /// and the remaining decoder gaps listed in `README.md`).
    NotImplemented,
    /// A NAL-walker error surfaced through the top-level entry
    /// points.
    Nal(NalError),
    /// A VPS-parser error surfaced through the top-level entry
    /// points.
    Vps(VpsError),
    /// An SPS-parser error surfaced through the top-level entry
    /// points.
    Sps(SpsError),
    /// A PPS-parser error surfaced through the top-level entry
    /// points.
    Pps(PpsError),
    /// A slice-segment-header-parser error surfaced through the
    /// top-level entry points.
    Slice(SliceError),
    /// An `hrd_parameters()` parser error surfaced through the
    /// top-level entry points.
    Hrd(HrdError),
    /// A `vui_parameters()` parser error surfaced through the
    /// top-level entry points.
    Vui(VuiError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-h265: decoder/encoder not wired up yet"),
            Self::Nal(e) => write!(f, "oxideav-h265 NAL error: {e}"),
            Self::Vps(e) => write!(f, "oxideav-h265 VPS error: {e}"),
            Self::Sps(e) => write!(f, "oxideav-h265 SPS error: {e}"),
            Self::Pps(e) => write!(f, "oxideav-h265 PPS error: {e}"),
            Self::Slice(e) => write!(f, "oxideav-h265 slice header error: {e}"),
            Self::Hrd(e) => write!(f, "oxideav-h265 hrd error: {e}"),
            Self::Vui(e) => write!(f, "oxideav-h265 vui error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<NalError> for Error {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}

impl From<VpsError> for Error {
    fn from(e: VpsError) -> Self {
        Self::Vps(e)
    }
}

impl From<SpsError> for Error {
    fn from(e: SpsError) -> Self {
        Self::Sps(e)
    }
}

impl From<PpsError> for Error {
    fn from(e: PpsError) -> Self {
        Self::Pps(e)
    }
}

impl From<SliceError> for Error {
    fn from(e: SliceError) -> Self {
        Self::Slice(e)
    }
}

impl From<HrdError> for Error {
    fn from(e: HrdError) -> Self {
        Self::Hrd(e)
    }
}

impl From<VuiError> for Error {
    fn from(e: VuiError) -> Self {
        Self::Vui(e)
    }
}
