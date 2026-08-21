//! Sequence Parameter Set (SPS) parser per ITU-T Rec. H.265 §7.3.2.2.
//!
//! Scope: parse the full SPS RBSP body through the
//! `strong_intra_smoothing_enabled_flag` field, the
//! `vui_parameters_present_flag` gate (whose §E.2.1
//! `vui_parameters()` body is decoded into [`crate::hevc::engine::vui::VuiParameters`]),
//! and the `sps_extension_present_flag` gate. When the extension gate
//! is set, the trailing bytes (extension payload + RBSP trailing
//! bits) are surfaced as an **opaque tail**: a copy of the
//! still-unparsed RBSP bytes plus the bit offset within the first
//! byte at which the opaque tail begins. The per-extension
//! `sps_*_extension( )` syntax structures are not materialised yet.
//!
//! When `scaling_list_enabled_flag == 1` and
//! `sps_scaling_list_data_present_flag == 1`, the §7.3.4
//! `scaling_list_data()` block is parsed via the shared
//! [`crate::hevc::engine::scaling_list`] module; otherwise the §7.4.5 default lists
//! apply.
//!
//! ## Layout summary
//!
//! ```text
//! sps_video_parameter_set_id                       u(4)
//! sps_max_sub_layers_minus1                        u(3)
//! sps_temporal_id_nesting_flag                     u(1)
//! profile_tier_level( 1, sps_max_sub_layers_minus1 )
//! sps_seq_parameter_set_id                        ue(v)
//! chroma_format_idc                               ue(v)
//! if( chroma_format_idc == 3 )
//!   separate_colour_plane_flag                     u(1)
//! pic_width_in_luma_samples                       ue(v)
//! pic_height_in_luma_samples                      ue(v)
//! conformance_window_flag                          u(1)
//! if( conformance_window_flag ) {
//!   conf_win_left_offset                          ue(v)
//!   conf_win_right_offset                         ue(v)
//!   conf_win_top_offset                           ue(v)
//!   conf_win_bottom_offset                        ue(v)
//! }
//! bit_depth_luma_minus8                           ue(v)
//! bit_depth_chroma_minus8                         ue(v)
//! log2_max_pic_order_cnt_lsb_minus4               ue(v)
//! sps_sub_layer_ordering_info_present_flag         u(1)
//! for( i = (...) ; i <= sps_max_sub_layers_minus1; i++ ) {
//!   sps_max_dec_pic_buffering_minus1[i]           ue(v)
//!   sps_max_num_reorder_pics[i]                   ue(v)
//!   sps_max_latency_increase_plus1[i]             ue(v)
//! }
//! log2_min_luma_coding_block_size_minus3          ue(v)
//! log2_diff_max_min_luma_coding_block_size        ue(v)
//! log2_min_luma_transform_block_size_minus2       ue(v)
//! log2_diff_max_min_luma_transform_block_size     ue(v)
//! max_transform_hierarchy_depth_inter             ue(v)
//! max_transform_hierarchy_depth_intra             ue(v)
//! scaling_list_enabled_flag                        u(1)
//!   if( scaling_list_enabled_flag ) {
//!     sps_scaling_list_data_present_flag            u(1)
//!     if( sps_scaling_list_data_present_flag )
//!       scaling_list_data( )                        /* §7.3.4 */
//!   }
//! amp_enabled_flag                                 u(1)
//! sample_adaptive_offset_enabled_flag              u(1)
//! pcm_enabled_flag                                 u(1)
//! if( pcm_enabled_flag ) {
//!   pcm_sample_bit_depth_luma_minus1               u(4)
//!   pcm_sample_bit_depth_chroma_minus1             u(4)
//!   log2_min_pcm_luma_coding_block_size_minus3    ue(v)
//!   log2_diff_max_min_pcm_luma_coding_block_size  ue(v)
//!   pcm_loop_filter_disabled_flag                  u(1)
//! }
//! num_short_term_ref_pic_sets                     ue(v)
//! for( i = 0; i < num_short_term_ref_pic_sets; i++ )
//!   st_ref_pic_set( i )
//! long_term_ref_pics_present_flag                  u(1)
//! if( long_term_ref_pics_present_flag ) {
//!   num_long_term_ref_pics_sps                    ue(v)
//!   for( i = 0; i < num_long_term_ref_pics_sps; i++ ) {
//!     lt_ref_pic_poc_lsb_sps[i]                    u(v)  /* log2_max_poc_lsb+4 */
//!     used_by_curr_pic_lt_sps_flag[i]              u(1)
//!   }
//! }
//! sps_temporal_mvp_enabled_flag                    u(1)
//! strong_intra_smoothing_enabled_flag              u(1)
//! vui_parameters_present_flag                      u(1)
//! if( vui_parameters_present_flag )
//!   vui_parameters( )                              /* §E.2.1 */
//! sps_extension_present_flag                       u(1)
//! if( sps_extension_present_flag ) {
//!   sps_range_extension_flag                        u(1)
//!   sps_multilayer_extension_flag                   u(1)
//!   sps_3d_extension_flag                           u(1)
//!   sps_scc_extension_flag                          u(1)
//!   sps_extension_4bits                             u(4)
//!   /* opaque tail begins at the first set body, or stays
//!      empty when every flag is 0 */
//! }
//! ```
//!
//! When `sps_extension_present_flag == 1` the eight bits of typed
//! extension flags (`sps_range_extension_flag`,
//! `sps_multilayer_extension_flag`, `sps_3d_extension_flag`,
//! `sps_scc_extension_flag`, `sps_extension_4bits`) are decoded into
//! [`SpsExtensionFlags`]. When all five sub-fields are zero only the
//! RBSP trailing byte remains and no opaque tail is captured; otherwise
//! the extension bodies (`sps_range_extension()`,
//! `sps_multilayer_extension()`, `sps_3d_extension()`,
//! `sps_scc_extension()`, and the `sps_extension_data_flag` while-loop
//! gated by `sps_extension_4bits`) are surfaced as a single
//! [`OpaqueTail`] starting at the first body's bit position.
//!
//! Validity checks performed here, sourced from §7.4.3.2:
//!
//! * `sps_max_sub_layers_minus1` range 0..=6.
//! * `sps_video_parameter_set_id` and `sps_seq_parameter_set_id`
//!   range 0..=15.
//! * `chroma_format_idc` range 0..=3.
//! * `pic_width_in_luma_samples` and `pic_height_in_luma_samples` not
//!   equal to zero (the modulo-`MinCbSizeY` check is deferred — it
//!   depends on the not-yet-validated `log2_min_cb_size`).
//! * `bit_depth_luma_minus8` and `bit_depth_chroma_minus8` range
//!   0..=8 (§7.4.3.2 caps the encoded bit depth at 16).
//! * `log2_max_pic_order_cnt_lsb_minus4` range 0..=12.
//! * `pcm_sample_bit_depth_luma_minus1` / `pcm_sample_bit_depth_chroma_minus1`
//!   must satisfy `PcmBitDepthY <= BitDepthY` /
//!   `PcmBitDepthC <= BitDepthC` per §7.4.3.2 / equation (7-25 / 7-26).
//! * `num_short_term_ref_pic_sets` range 0..=64.
//! * `num_long_term_ref_pics_sps` range 0..=32.
//! * `num_negative_pics` / `num_positive_pics` capped at 16 (the
//!   §7.4.8 bound is
//!   `sps_max_dec_pic_buffering_minus1[sps_max_sub_layers_minus1]`,
//!   itself bounded at 16 because `MaxDpbSize` per §A.4.2 is 16).
//! * `delta_poc_s0_minus1` / `delta_poc_s1_minus1` /
//!   `abs_delta_rps_minus1` range 0..=2^15-1.
//! * `sps_scaling_list_data_present_flag == 1` triggers a §7.3.4
//!   `scaling_list_data()` parse into [`ScalingListData`] via the
//!   shared [`crate::hevc::engine::scaling_list`] module.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::scaling_list::{ScalingListData, ScalingListError};
use crate::hevc::engine::vps::{
    HEVC_MAX_SUB_LAYERS, ProfileTierLevel, SubLayerOrderingInfo, VpsError,
};
use crate::hevc::engine::vui::{VuiError, VuiParameters};

/// Maximum number of short-term reference picture sets an SPS may
/// carry. Per §7.4.3.2 the field is bounded at 64 inclusive.
pub const HEVC_MAX_NUM_SHORT_TERM_RPS: usize = 64;

/// Maximum number of long-term reference pictures an SPS may carry.
/// Per §7.4.3.2 the field is bounded at 32 inclusive.
pub const HEVC_MAX_NUM_LONG_TERM_RPS: usize = 32;

/// Maximum number of negative / positive entries permitted in one
/// short-term RPS. Per §7.4.8 the bound is
/// `sps_max_dec_pic_buffering_minus1[sps_max_sub_layers_minus1]`
/// which §A.4.2 caps at `MaxDpbSize - 1 = 15` (for typical levels);
/// 16 is a defensive upper bound for the parser.
pub const HEVC_MAX_RPS_PICS: usize = 16;

/// Errors that can arise while parsing an SPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpsError {
    /// The RBSP ran out of bits before the SPS was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.3.2.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as a `u32` (signed elements are
        /// re-cast at the call site).
        got: u32,
    },
    /// A `scaling_list_data()` parse (§7.3.4) from the SPS failed.
    ScalingList(ScalingListError),
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
    /// A `profile_tier_level()` parse from the §7.3.3 walk failed.
    Ptl(VpsError),
    /// A `vui_parameters()` parse (§E.2.1) from the SPS failed.
    Vui(VuiError),
}

impl core::fmt::Display for SpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("SPS RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "SPS syntax element {field} out of range: {got}")
            }
            Self::ScalingList(e) => write!(f, "scaling-list error during SPS parse: {e}"),
            Self::Bitstream(e) => write!(f, "bitstream error during SPS parse: {e}"),
            Self::Ptl(e) => write!(f, "profile_tier_level error during SPS parse: {e}"),
            Self::Vui(e) => write!(f, "vui_parameters error during SPS parse: {e}"),
        }
    }
}

impl std::error::Error for SpsError {}

impl From<BitReaderError> for SpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<VpsError> for SpsError {
    fn from(e: VpsError) -> Self {
        // Surface the inner reader-truncation directly so callers can
        // distinguish "ran off the end mid-PTL" from other PTL faults.
        if matches!(e, VpsError::Truncated) {
            Self::Truncated
        } else {
            Self::Ptl(e)
        }
    }
}

impl From<ScalingListError> for SpsError {
    fn from(e: ScalingListError) -> Self {
        // Flatten truncation / raw-reader faults to the SPS-level
        // equivalents so the public surface stays predictable; carry
        // the structured scaling-list faults through as-is.
        match e {
            ScalingListError::Truncated => Self::Truncated,
            ScalingListError::Bitstream(b) => Self::Bitstream(b),
            other => Self::ScalingList(other),
        }
    }
}

impl From<VuiError> for SpsError {
    fn from(e: VuiError) -> Self {
        // Flatten truncation / raw-reader faults to the SPS-level
        // equivalents so the public surface stays predictable; carry
        // the structured VUI faults through as-is.
        match e {
            VuiError::Truncated => Self::Truncated,
            VuiError::Bitstream(b) => Self::Bitstream(b),
            other => Self::Vui(other),
        }
    }
}

/// Conformance-window offsets, in §6.5.5 SubWidthC/SubHeightC units
/// (not pixel units — the multiplier depends on
/// [`SeqParameterSet::chroma_format_idc`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConformanceWindow {
    /// `conf_win_left_offset` (`ue(v)`).
    pub left_offset: u32,
    /// `conf_win_right_offset` (`ue(v)`).
    pub right_offset: u32,
    /// `conf_win_top_offset` (`ue(v)`).
    pub top_offset: u32,
    /// `conf_win_bottom_offset` (`ue(v)`).
    pub bottom_offset: u32,
}

/// PCM-related SPS block per §7.3.2.2 (`pcm_enabled_flag == 1` only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmInfo {
    /// `pcm_sample_bit_depth_luma_minus1` (`u(4)`). The §7.4.3.2
    /// `PcmBitDepthY` derivation is `value + 1`.
    pub bit_depth_luma_minus1: u8,
    /// `pcm_sample_bit_depth_chroma_minus1` (`u(4)`). `PcmBitDepthC = value + 1`.
    pub bit_depth_chroma_minus1: u8,
    /// `log2_min_pcm_luma_coding_block_size_minus3` (`ue(v)`).
    /// `Log2MinIpcmCbSizeY = value + 3`.
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_pcm_luma_coding_block_size` (`ue(v)`).
    /// `Log2MaxIpcmCbSizeY = Log2MinIpcmCbSizeY + value`.
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    /// `pcm_loop_filter_disabled_flag` (`u(1)`).
    pub loop_filter_disabled_flag: bool,
}

/// Short-term reference picture set per §7.3.7. Only the explicit
/// (non-inter-predicted) form materialises the per-entry POC and
/// `used_by_curr_pic` arrays; the inter-RPS-prediction form materialises
/// only the inputs to the §7.4.8 derivation (`delta_idx_minus1`,
/// `delta_rps_sign`, `abs_delta_rps_minus1`, and the
/// `used_by_curr_pic_flag` / `use_delta_flag` arrays of length
/// `NumDeltaPocs[RefRpsIdx] + 1`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortTermRefPicSet {
    /// `inter_ref_pic_set_prediction_flag` — inferred to 0 when not
    /// signalled (i.e. always 0 for `stRpsIdx == 0`).
    pub inter_ref_pic_set_prediction_flag: bool,
    /// `delta_idx_minus1` (only meaningful when
    /// `inter_ref_pic_set_prediction_flag == 1` and the index this RPS
    /// is being constructed at equals `num_short_term_ref_pic_sets`;
    /// otherwise inferred to 0). The §7.4.8 derivation is
    /// `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)`.
    pub delta_idx_minus1: u32,
    /// `delta_rps_sign` (1-bit).
    pub delta_rps_sign: bool,
    /// `abs_delta_rps_minus1` (`ue(v)`, range 0..=2^15-1).
    pub abs_delta_rps_minus1: u32,
    /// Per-entry `used_by_curr_pic_flag[j]` for the inter-RPS-prediction
    /// case. Length is `NumDeltaPocs[RefRpsIdx] + 1` when the form
    /// applies, zero otherwise.
    pub used_by_curr_pic_flag: Vec<bool>,
    /// Per-entry `use_delta_flag[j]` for the inter-RPS-prediction case.
    /// Same length / population rule as
    /// [`used_by_curr_pic_flag`](Self::used_by_curr_pic_flag); per
    /// §7.4.8 the value is inferred to 1 when the
    /// `used_by_curr_pic_flag[j]` was 1.
    pub use_delta_flag: Vec<bool>,
    /// `num_negative_pics` (`ue(v)`), only meaningful in the explicit
    /// form.
    pub num_negative_pics: u32,
    /// `num_positive_pics` (`ue(v)`), only meaningful in the explicit
    /// form.
    pub num_positive_pics: u32,
    /// `delta_poc_s0_minus1[i]` (`ue(v)`), explicit form. Length
    /// `num_negative_pics`.
    pub delta_poc_s0_minus1: Vec<u32>,
    /// `used_by_curr_pic_s0_flag[i]` (`u(1)`), explicit form. Length
    /// `num_negative_pics`.
    pub used_by_curr_pic_s0_flag: Vec<bool>,
    /// `delta_poc_s1_minus1[i]` (`ue(v)`), explicit form. Length
    /// `num_positive_pics`.
    pub delta_poc_s1_minus1: Vec<u32>,
    /// `used_by_curr_pic_s1_flag[i]` (`u(1)`), explicit form. Length
    /// `num_positive_pics`.
    pub used_by_curr_pic_s1_flag: Vec<bool>,
}

impl ShortTermRefPicSet {
    /// `NumDeltaPocs[stRpsIdx]` per §7.4.8: in the explicit form this
    /// is `num_negative_pics + num_positive_pics`; in the inter-RPS
    /// form the exact count requires the §7.4.8 derivation
    /// (equations 7-61 / 7-62 / 7-71) against a materialised source
    /// RPS. See [`Self::materialize`] / [`MaterializedShortTermRefPicSet`]
    /// for the full derivation.
    ///
    /// For the inter-RPS form this helper returns the count of
    /// `use_delta_flag[j] == 1` entries, which is an upper bound that
    /// happens to be exact when none of the source POCs flip sign
    /// across `deltaRps`. Callers that need the exact wire-conformant
    /// count must materialise the RPS chain.
    pub fn num_delta_pocs(&self) -> u32 {
        if self.inter_ref_pic_set_prediction_flag {
            self.use_delta_flag.iter().filter(|&&v| v).count() as u32
        } else {
            self.num_negative_pics + self.num_positive_pics
        }
    }

    /// Materialise this `st_ref_pic_set(stRpsIdx)` into the post-§7.4.8
    /// per-position arrays `(NumNegativePics, DeltaPocS0[],
    /// UsedByCurrPicS0[], NumPositivePics, DeltaPocS1[],
    /// UsedByCurrPicS1[])` consumed by §7.4.7.2 and downstream paths.
    ///
    /// * For the explicit form (`inter_ref_pic_set_prediction_flag ==
    ///   0`) this is equations 7-63..7-70: `NumNegativePics =
    ///   num_negative_pics`, `NumPositivePics = num_positive_pics`,
    ///   `UsedByCurrPicSk[i] = used_by_curr_pic_sk_flag[i]`, and the
    ///   cumulative `DeltaPocS0[i] = DeltaPocS0[i-1] -
    ///   (delta_poc_s0_minus1[i] + 1)` recurrence with the
    ///   first-element seeds (`DeltaPocS0` at i=0 is
    ///   `-(delta_poc_s0_minus1 at 0 + 1)` and `DeltaPocS1` at i=0 is
    ///   `delta_poc_s1_minus1 at 0 + 1`). `source` is ignored in this
    ///   branch.
    /// * For the inter-RPS-prediction form
    ///   (`inter_ref_pic_set_prediction_flag == 1`) this runs
    ///   equations 7-61 (negative side, source-S1-reverse +
    ///   `deltaRps`-self + source-S0-forward) and 7-62 (positive side,
    ///   source-S0-reverse + `deltaRps`-self + source-S1-forward) over
    ///   the already-materialised source RPS supplied via `source`,
    ///   with `deltaRps = (1 - 2*delta_rps_sign) *
    ///   (abs_delta_rps_minus1 + 1)`. The output's
    ///   `NumNegativePics` / `NumPositivePics` reflect the surviving
    ///   `dPoc < 0` / `dPoc > 0` entries with their `use_delta_flag[j]
    ///   == 1` gates, per equation 7-71's `NumDeltaPocs`
    ///   summation. The `source` argument must be `Some(_)` for the
    ///   inter form; the function returns
    ///   [`ShortTermRefPicSetMaterializeError::MissingSource`] if
    ///   absent. Bounds on `used_by_curr_pic_flag` /
    ///   `use_delta_flag` length are checked against the source's
    ///   `NumDeltaPocs + 1`; a mismatch returns
    ///   [`ShortTermRefPicSetMaterializeError::SourceLengthMismatch`].
    pub fn materialize(
        &self,
        source: Option<&MaterializedShortTermRefPicSet>,
    ) -> Result<MaterializedShortTermRefPicSet, ShortTermRefPicSetMaterializeError> {
        if self.inter_ref_pic_set_prediction_flag {
            let source = source.ok_or(ShortTermRefPicSetMaterializeError::MissingSource)?;
            // `deltaRps` per equation 7-60. The bit-width of
            // `abs_delta_rps_minus1` is bounded by `ue(v)` plus the
            // §7.4.8 range check (<= 2^15-1 -> deltaRps fits in i32).
            let abs = self.abs_delta_rps_minus1 as i64 + 1;
            let delta_rps = if self.delta_rps_sign { -abs } else { abs } as i32;
            let num_neg_src = source.num_negative_pics();
            let num_pos_src = source.num_positive_pics();
            let num_delta_src = source.num_delta_pocs(); // == num_neg_src + num_pos_src
            // Per §7.4.8 the `used_by_curr_pic_flag` / `use_delta_flag`
            // arrays have length `NumDeltaPocs[RefRpsIdx] + 1`.
            let expected = (num_delta_src + 1) as usize;
            if self.used_by_curr_pic_flag.len() != expected || self.use_delta_flag.len() != expected
            {
                return Err(ShortTermRefPicSetMaterializeError::SourceLengthMismatch {
                    expected: expected as u32,
                    got_used: self.used_by_curr_pic_flag.len(),
                    got_delta: self.use_delta_flag.len(),
                });
            }
            // Negative side (equation 7-61): walk source's positive
            // POCs in reverse (their `dPoc = DeltaPocS1[RefRpsIdx][j]
            // + deltaRps` may have crossed zero), then optionally
            // `deltaRps` itself (only if negative), then source's
            // negative POCs in forward order.
            let mut delta_poc_s0 = Vec::new();
            let mut used_by_curr_pic_s0 = Vec::new();
            for j in (0..num_pos_src).rev() {
                let d_poc = source.delta_poc_s1[j as usize] + delta_rps;
                if d_poc < 0 && self.use_delta_flag[(num_neg_src + j) as usize] {
                    delta_poc_s0.push(d_poc);
                    used_by_curr_pic_s0
                        .push(self.used_by_curr_pic_flag[(num_neg_src + j) as usize]);
                }
            }
            if delta_rps < 0 && self.use_delta_flag[num_delta_src as usize] {
                delta_poc_s0.push(delta_rps);
                used_by_curr_pic_s0.push(self.used_by_curr_pic_flag[num_delta_src as usize]);
            }
            for j in 0..num_neg_src {
                let d_poc = source.delta_poc_s0[j as usize] + delta_rps;
                if d_poc < 0 && self.use_delta_flag[j as usize] {
                    delta_poc_s0.push(d_poc);
                    used_by_curr_pic_s0.push(self.used_by_curr_pic_flag[j as usize]);
                }
            }
            // Positive side (equation 7-62): walk source's negative
            // POCs in reverse (their `dPoc = DeltaPocS0[RefRpsIdx][j]
            // + deltaRps` may have crossed zero), then optionally
            // `deltaRps` itself (only if positive), then source's
            // positive POCs in forward order.
            let mut delta_poc_s1 = Vec::new();
            let mut used_by_curr_pic_s1 = Vec::new();
            for j in (0..num_neg_src).rev() {
                let d_poc = source.delta_poc_s0[j as usize] + delta_rps;
                if d_poc > 0 && self.use_delta_flag[j as usize] {
                    delta_poc_s1.push(d_poc);
                    used_by_curr_pic_s1.push(self.used_by_curr_pic_flag[j as usize]);
                }
            }
            if delta_rps > 0 && self.use_delta_flag[num_delta_src as usize] {
                delta_poc_s1.push(delta_rps);
                used_by_curr_pic_s1.push(self.used_by_curr_pic_flag[num_delta_src as usize]);
            }
            for j in 0..num_pos_src {
                let d_poc = source.delta_poc_s1[j as usize] + delta_rps;
                if d_poc > 0 && self.use_delta_flag[(num_neg_src + j) as usize] {
                    delta_poc_s1.push(d_poc);
                    used_by_curr_pic_s1
                        .push(self.used_by_curr_pic_flag[(num_neg_src + j) as usize]);
                }
            }
            Ok(MaterializedShortTermRefPicSet {
                delta_poc_s0,
                used_by_curr_pic_s0,
                delta_poc_s1,
                used_by_curr_pic_s1,
            })
        } else {
            // Explicit form, equations 7-63..7-70.
            let mut delta_poc_s0 = Vec::with_capacity(self.num_negative_pics as usize);
            let mut prev: i32 = 0;
            for (i, &delta_minus1) in self.delta_poc_s0_minus1.iter().enumerate() {
                // delta_minus1 has been range-checked to 0..=2^15-1 on
                // parse; the cumulative sum cannot exceed i32 capacity
                // given num_negative_pics <= 16.
                let step = delta_minus1 as i32 + 1;
                let d = if i == 0 { -step } else { prev - step };
                delta_poc_s0.push(d);
                prev = d;
            }
            let mut delta_poc_s1 = Vec::with_capacity(self.num_positive_pics as usize);
            let mut prev: i32 = 0;
            for (i, &delta_minus1) in self.delta_poc_s1_minus1.iter().enumerate() {
                let step = delta_minus1 as i32 + 1;
                let d = if i == 0 { step } else { prev + step };
                delta_poc_s1.push(d);
                prev = d;
            }
            Ok(MaterializedShortTermRefPicSet {
                delta_poc_s0,
                used_by_curr_pic_s0: self.used_by_curr_pic_s0_flag.clone(),
                delta_poc_s1,
                used_by_curr_pic_s1: self.used_by_curr_pic_s1_flag.clone(),
            })
        }
    }
}

/// Errors that can arise while running the §7.4.8 inter-RPS-prediction
/// derivation via [`ShortTermRefPicSet::materialize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortTermRefPicSetMaterializeError {
    /// The RPS uses inter-RPS prediction
    /// (`inter_ref_pic_set_prediction_flag == 1`) but the caller did
    /// not supply a materialised source RPS.
    MissingSource,
    /// The on-wire `used_by_curr_pic_flag` / `use_delta_flag` arrays
    /// did not match the source RPS's `NumDeltaPocs[RefRpsIdx] + 1`
    /// length. This indicates the parser and the materialiser saw
    /// different source RPSes (most likely a `RefRpsIdx` mismatch).
    SourceLengthMismatch {
        /// Expected length: `NumDeltaPocs[RefRpsIdx] + 1`.
        expected: u32,
        /// Actual length of `used_by_curr_pic_flag`.
        got_used: usize,
        /// Actual length of `use_delta_flag`.
        got_delta: usize,
    },
}

impl core::fmt::Display for ShortTermRefPicSetMaterializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingSource => f.write_str(
                "short-term RPS materialise: inter_ref_pic_set_prediction_flag is set but no source RPS supplied",
            ),
            Self::SourceLengthMismatch {
                expected,
                got_used,
                got_delta,
            } => write!(
                f,
                "short-term RPS materialise: per-position array length mismatch: expected {expected}, got used={got_used} delta={got_delta}"
            ),
        }
    }
}

impl std::error::Error for ShortTermRefPicSetMaterializeError {}

/// Materialised short-term reference-picture-set per §7.4.8 — the
/// post-derivation form that exposes the per-position `DeltaPocS0[]`,
/// `UsedByCurrPicS0[]`, `DeltaPocS1[]`, `UsedByCurrPicS1[]` arrays as
/// signed POC deltas. `NumNegativePics[stRpsIdx]` and
/// `NumPositivePics[stRpsIdx]` are exposed as the array lengths.
///
/// The DeltaPocS0 array is in source order — i.e. the negative POCs
/// in the order §7.4.8 produces them — which for the explicit form
/// (equations 7-67 / 7-69) is descending (each element is the next
/// further-in-the-past POC) and for the inter-RPS-prediction form
/// (equation 7-61) walks source positives in reverse, then optionally
/// `deltaRps`, then source negatives in forward order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaterializedShortTermRefPicSet {
    /// `DeltaPocS0[stRpsIdx][i]` for `i` in
    /// `0..NumNegativePics[stRpsIdx]`. All entries are strictly
    /// negative.
    pub delta_poc_s0: Vec<i32>,
    /// `UsedByCurrPicS0[stRpsIdx][i]` for `i` in
    /// `0..NumNegativePics[stRpsIdx]`.
    pub used_by_curr_pic_s0: Vec<bool>,
    /// `DeltaPocS1[stRpsIdx][i]` for `i` in
    /// `0..NumPositivePics[stRpsIdx]`. All entries are strictly
    /// positive.
    pub delta_poc_s1: Vec<i32>,
    /// `UsedByCurrPicS1[stRpsIdx][i]` for `i` in
    /// `0..NumPositivePics[stRpsIdx]`.
    pub used_by_curr_pic_s1: Vec<bool>,
}

impl MaterializedShortTermRefPicSet {
    /// `NumNegativePics[stRpsIdx]` per §7.4.8.
    pub fn num_negative_pics(&self) -> u32 {
        self.delta_poc_s0.len() as u32
    }
    /// `NumPositivePics[stRpsIdx]` per §7.4.8.
    pub fn num_positive_pics(&self) -> u32 {
        self.delta_poc_s1.len() as u32
    }
    /// `NumDeltaPocs[stRpsIdx]` per equation 7-71.
    pub fn num_delta_pocs(&self) -> u32 {
        self.num_negative_pics() + self.num_positive_pics()
    }
}

/// Long-term reference picture entry per the `long_term_ref_pics_present_flag`
/// block of §7.3.2.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTermRefPicEntry {
    /// `lt_ref_pic_poc_lsb_sps[i]` — `u(v)` whose width is
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` bits.
    pub poc_lsb: u32,
    /// `used_by_curr_pic_lt_sps_flag[i]`.
    pub used_by_curr_pic: bool,
}

/// Opaque suffix surfaced when the parser hits an extension body it
/// does not yet decode (any of `sps_range_extension()`,
/// `sps_multilayer_extension()`, `sps_3d_extension()`,
/// `sps_scc_extension()`, or the `sps_extension_data_flag` while-loop
/// gated by `sps_extension_4bits != 0`; also reused by the
/// [`crate::hevc::engine::pps::PicParameterSet`] extension tail). The bytes captured
/// are the still-unparsed RBSP body, starting at the byte that
/// contains the next un-read bit. `start_bit_in_first_byte` is the
/// bit offset of that next bit within `bytes[0]` (0 = MSB).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueTail {
    /// Raw RBSP bytes from the first byte containing the next
    /// un-read bit onwards (including the `rbsp_trailing_bits()` byte
    /// at the end).
    pub bytes: Vec<u8>,
    /// Bit offset within `bytes[0]` where the opaque tail begins,
    /// in MSB-first order (0..=7).
    pub start_bit_in_first_byte: u8,
}

/// SPS extension-flag block per §7.3.2.2.1
/// (`sps_extension_present_flag == 1`), holding the four typed
/// extension-present flags and the reserved-for-future-use
/// `sps_extension_4bits` group.
///
/// Per §7.4.3.2.1, when `sps_extension_present_flag == 0` every flag
/// in this block is inferred to 0 and `sps_extension_4bits` is
/// inferred to 0; the parser surfaces that case as
/// [`SeqParameterSet::extension_flags`] = `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpsExtensionFlags {
    /// `sps_range_extension_flag` (§7.3.2.2.1). When true, a
    /// `sps_range_extension()` body (§7.3.2.2.2) follows in the bit
    /// stream and is currently surfaced inside the SPS
    /// [`SeqParameterSet::opaque_tail`]. This flag selects the §A.3.5
    /// Format Range Extensions (RExt) profiles family.
    pub sps_range_extension_flag: bool,
    /// `sps_multilayer_extension_flag` (§7.3.2.2.1, Annex F /
    /// scalable & multi-view extensions). When true, a
    /// `sps_multilayer_extension()` body follows and is surfaced
    /// inside the opaque tail.
    pub sps_multilayer_extension_flag: bool,
    /// `sps_3d_extension_flag` (§7.3.2.2.1, Annex I). When true, a
    /// `sps_3d_extension()` body follows and is surfaced inside the
    /// opaque tail.
    pub sps_3d_extension_flag: bool,
    /// `sps_scc_extension_flag` (§7.3.2.2.1). When true, a
    /// `sps_scc_extension()` body follows and is surfaced inside the
    /// opaque tail. This flag selects the §A.3.7 Screen Content
    /// Coding (SCC) profiles family.
    pub sps_scc_extension_flag: bool,
    /// `sps_extension_4bits` (`u(4)`). For bitstreams conforming to
    /// the current version of the specification this value shall be
    /// 0; non-zero values are reserved for future use. The §7.4.3.2.1
    /// decoder-side rule is to allow any value and (if it is non-zero)
    /// consume but ignore the `sps_extension_data_flag` while-loop it
    /// gates, so the parser surfaces the value verbatim. The trailing
    /// `while( more_rbsp_data() ) sps_extension_data_flag` block (only
    /// signalled when this field is non-zero) is surfaced inside the
    /// opaque tail.
    pub sps_extension_4bits: u8,
}

impl SpsExtensionFlags {
    /// True when at least one of the four extension flags is set or
    /// `sps_extension_4bits` is non-zero — i.e. when at least one
    /// downstream extension body follows in the bit stream and the
    /// SPS therefore carries an opaque tail starting at the first
    /// body's bit position.
    pub fn has_body(&self) -> bool {
        self.sps_range_extension_flag
            || self.sps_multilayer_extension_flag
            || self.sps_3d_extension_flag
            || self.sps_scc_extension_flag
            || self.sps_extension_4bits != 0
    }

    /// True when the `sps_scc_extension()` body can be decoded in
    /// place — it is signalled and no still-opaque body
    /// (`sps_multilayer_extension()` / `sps_3d_extension()`) precedes
    /// it in the bit stream. When a multilayer / 3D body precedes it,
    /// the SCC body stays inside the opaque tail.
    fn scc_decodable_in_place(&self) -> bool {
        self.sps_scc_extension_flag
            && !self.sps_multilayer_extension_flag
            && !self.sps_3d_extension_flag
    }

    /// True when an extension body still follows the (range +
    /// optionally SCC) bodies decoded in place — i.e. a multilayer /
    /// 3D body, the `sps_extension_data_flag` while-loop, or an
    /// SCC body whose multilayer/3D predecessor kept it opaque.
    fn has_opaque_body_after_decoded(&self) -> bool {
        if self.sps_multilayer_extension_flag || self.sps_3d_extension_flag {
            // The first un-decoded body is the multilayer / 3D one;
            // everything from there (incl. any SCC body) is opaque.
            return true;
        }
        // No multilayer / 3D body: SCC (if present) was decoded in
        // place, so only the sps_extension_data_flag while-loop may
        // remain.
        self.sps_extension_4bits != 0
    }
}

/// Decoded `sps_scc_extension()` body per §7.3.2.2.3, present when
/// [`SpsExtensionFlags::sps_scc_extension_flag`] is set and no opaque
/// multilayer / 3D body precedes it. Per §7.4.3.2.3 the absent fields
/// are inferred to 0 / empty when this struct is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpsSccExtension {
    /// `sps_curr_pic_ref_enabled_flag` — when 1, a picture referring
    /// to the SPS may be in a reference picture list of one of its own
    /// slices (intra block copy).
    pub sps_curr_pic_ref_enabled_flag: bool,
    /// `palette_mode_enabled_flag` — when 1, the palette-mode decoding
    /// process may be used for intra blocks.
    pub palette_mode_enabled_flag: bool,
    /// `palette_max_size` (`ue(v)`), present only when
    /// `palette_mode_enabled_flag`; the maximum allowed palette size
    /// (inferred 0 otherwise).
    pub palette_max_size: u32,
    /// `delta_palette_max_predictor_size` (`ue(v)`), present only when
    /// `palette_mode_enabled_flag`. `PaletteMaxPredictorSize =
    /// palette_max_size + value` (eq. 7-35).
    pub delta_palette_max_predictor_size: u32,
    /// `sps_palette_predictor_initializers_present_flag` — when 1, the
    /// sequence palette predictor is initialised from
    /// [`Self::sps_palette_predictor_initializer`].
    pub sps_palette_predictor_initializers_present_flag: bool,
    /// `sps_num_palette_predictor_initializers_minus1` (`ue(v)`),
    /// present only when the initializers-present flag is set; the
    /// initializer table then holds `value + 1` entries per component.
    pub sps_num_palette_predictor_initializers_minus1: u32,
    /// `sps_palette_predictor_initializer[comp][i]` (§7.3.2.2.3),
    /// indexed `[comp][i]`. `comp` runs over `numComps` (1 when
    /// `chroma_format_idc == 0`, else 3). Each value is `u(v)` —
    /// `BitDepthY` bits for `comp == 0`, `BitDepthC` bits otherwise.
    /// Empty when no initializers are signalled.
    pub sps_palette_predictor_initializer: Vec<Vec<u32>>,
    /// `motion_vector_resolution_control_idc` (`u(2)`) — controls the
    /// presence / inference of `use_integer_mv_flag`.
    pub motion_vector_resolution_control_idc: u8,
    /// `intra_boundary_filtering_disabled_flag` — when 1, the intra
    /// boundary filtering process is unconditionally disabled.
    pub intra_boundary_filtering_disabled_flag: bool,
}

impl SpsSccExtension {
    /// Decode `sps_scc_extension()` (§7.3.2.2.3). `chroma_format_idc`
    /// selects `numComps` (1 if 0, else 3) and `bit_depth_luma` /
    /// `bit_depth_chroma` give the `u(v)` width of each palette
    /// predictor initializer component.
    fn parse(
        br: &mut BitReader,
        chroma_format_idc: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
    ) -> Result<Self, SpsError> {
        let sps_curr_pic_ref_enabled_flag = br.u1()? != 0;
        let palette_mode_enabled_flag = br.u1()? != 0;
        let mut palette_max_size = 0u32;
        let mut delta_palette_max_predictor_size = 0u32;
        let mut sps_palette_predictor_initializers_present_flag = false;
        let mut sps_num_palette_predictor_initializers_minus1 = 0u32;
        let mut sps_palette_predictor_initializer = Vec::new();
        if palette_mode_enabled_flag {
            palette_max_size = br.ue()?;
            delta_palette_max_predictor_size = br.ue()?;
            // §7.4.3.2.3: when palette_max_size == 0 the
            // delta_palette_max_predictor_size must be 0 (bitstream
            // conformance — a zero-size palette cannot grow the
            // predictor).
            if palette_max_size == 0 && delta_palette_max_predictor_size != 0 {
                return Err(SpsError::ValueOutOfRange {
                    field: "delta_palette_max_predictor_size",
                    got: delta_palette_max_predictor_size,
                });
            }
            sps_palette_predictor_initializers_present_flag = br.u1()? != 0;
            // §7.4.3.2.3: likewise the initializers-present flag must be
            // 0 when palette_max_size == 0.
            if palette_max_size == 0 && sps_palette_predictor_initializers_present_flag {
                return Err(SpsError::ValueOutOfRange {
                    field: "sps_palette_predictor_initializers_present_flag",
                    got: 1,
                });
            }
            if sps_palette_predictor_initializers_present_flag {
                sps_num_palette_predictor_initializers_minus1 = br.ue()?;
                // §7.4.3.2.3: bounded by PaletteMaxPredictorSize − 1,
                // itself capped by the profile limits (§A.3.7:
                // PaletteMaxPredictorSize <= 128); reject anything past
                // the largest representable predictor so a malformed
                // count cannot drive the initializer allocation.
                if sps_num_palette_predictor_initializers_minus1 >= 128 {
                    return Err(SpsError::ValueOutOfRange {
                        field: "sps_num_palette_predictor_initializers_minus1",
                        got: sps_num_palette_predictor_initializers_minus1,
                    });
                }
                let num_comps = if chroma_format_idc == 0 { 1 } else { 3 };
                let num_entries = sps_num_palette_predictor_initializers_minus1 as usize + 1;
                sps_palette_predictor_initializer.reserve(num_comps);
                for comp in 0..num_comps {
                    let width = if comp == 0 {
                        bit_depth_luma
                    } else {
                        bit_depth_chroma
                    };
                    let mut row = Vec::with_capacity(num_entries);
                    for _ in 0..num_entries {
                        row.push(br.u(width)?);
                    }
                    sps_palette_predictor_initializer.push(row);
                }
            }
        }
        let motion_vector_resolution_control_idc = br.u(2)? as u8;
        // §7.4.3.2.3: the value 3 is reserved for future use and must
        // not appear in a conforming bitstream of this version.
        if motion_vector_resolution_control_idc == 3 {
            return Err(SpsError::ValueOutOfRange {
                field: "motion_vector_resolution_control_idc",
                got: 3,
            });
        }
        let intra_boundary_filtering_disabled_flag = br.u1()? != 0;
        Ok(Self {
            sps_curr_pic_ref_enabled_flag,
            palette_mode_enabled_flag,
            palette_max_size,
            delta_palette_max_predictor_size,
            sps_palette_predictor_initializers_present_flag,
            sps_num_palette_predictor_initializers_minus1,
            sps_palette_predictor_initializer,
            motion_vector_resolution_control_idc,
            intra_boundary_filtering_disabled_flag,
        })
    }

    /// `PaletteMaxPredictorSize` (eq. 7-35) — the maximum allowed
    /// palette predictor size, `palette_max_size +
    /// delta_palette_max_predictor_size`.
    pub fn palette_max_predictor_size(&self) -> u32 {
        self.palette_max_size + self.delta_palette_max_predictor_size
    }
}

/// Decoded `sps_range_extension()` body per §7.3.2.2.2, present when
/// [`SpsExtensionFlags::sps_range_extension_flag`] is set. Every field
/// is a single `u(1)` flag; per §7.4.3.2.2 each is inferred to 0 when
/// the `sps_range_extension()` body is absent (i.e. when
/// [`SeqParameterSet::sps_range_extension`] is `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpsRangeExtension {
    /// `transform_skip_rotation_enabled_flag` — when 1, a rotation is
    /// applied to the residual of intra 4×4 transform-skip /
    /// transform-bypass blocks.
    pub transform_skip_rotation_enabled_flag: bool,
    /// `transform_skip_context_enabled_flag` — when 1, a particular
    /// context is used for parsing `sig_coeff_flag` / coefficient
    /// magnitudes of transform-skip / transform-bypass blocks.
    pub transform_skip_context_enabled_flag: bool,
    /// `implicit_rdpcm_enabled_flag` — when 1, residual modification
    /// for blocks using a transform-bypass may be used for intra
    /// blocks referring to the SPS.
    pub implicit_rdpcm_enabled_flag: bool,
    /// `explicit_rdpcm_enabled_flag` — when 1, residual modification
    /// for blocks using a transform-bypass may be used for inter
    /// blocks referring to the SPS.
    pub explicit_rdpcm_enabled_flag: bool,
    /// `extended_precision_processing_flag` — when 1, an extended
    /// dynamic range is used for coefficient parsing and inverse
    /// transform processing.
    pub extended_precision_processing_flag: bool,
    /// `intra_smoothing_disabled_flag` — when 1, the filtering process
    /// of neighbouring samples is unconditionally disabled for intra
    /// prediction.
    pub intra_smoothing_disabled_flag: bool,
    /// `high_precision_offsets_enabled_flag` — when 1, weighted
    /// prediction offsets and SAO offsets use a bit depth of
    /// `BitDepth` rather than the default precision.
    pub high_precision_offsets_enabled_flag: bool,
    /// `persistent_rice_adaptation_enabled_flag` — when 1, the Rice
    /// parameter derivation for the binarization of
    /// `coeff_abs_level_remaining[]` is initialised at the start of
    /// each sub-block using mode-dependent statistics accumulated from
    /// previous sub-blocks.
    pub persistent_rice_adaptation_enabled_flag: bool,
    /// `cabac_bypass_alignment_enabled_flag` — when 1, a CABAC
    /// alignment process is used prior to bypass decoding of the
    /// syntax elements `coeff_sign_flag[]` and
    /// `coeff_abs_level_remaining[]`.
    pub cabac_bypass_alignment_enabled_flag: bool,
}

impl SpsRangeExtension {
    /// Decode the nine `u(1)` flags of `sps_range_extension()`
    /// (§7.3.2.2.2) in bit-stream order.
    fn parse(br: &mut BitReader) -> Result<Self, SpsError> {
        Ok(Self {
            transform_skip_rotation_enabled_flag: br.u1()? != 0,
            transform_skip_context_enabled_flag: br.u1()? != 0,
            implicit_rdpcm_enabled_flag: br.u1()? != 0,
            explicit_rdpcm_enabled_flag: br.u1()? != 0,
            extended_precision_processing_flag: br.u1()? != 0,
            intra_smoothing_disabled_flag: br.u1()? != 0,
            high_precision_offsets_enabled_flag: br.u1()? != 0,
            persistent_rice_adaptation_enabled_flag: br.u1()? != 0,
            cabac_bypass_alignment_enabled_flag: br.u1()? != 0,
        })
    }
}

/// Parsed Sequence Parameter Set per §7.3.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqParameterSet {
    /// `sps_video_parameter_set_id` (`u(4)`, range 0..=15).
    pub vps_id: u8,
    /// `sps_max_sub_layers_minus1` (`u(3)`, range 0..=6).
    pub max_sub_layers_minus1: u8,
    /// `sps_temporal_id_nesting_flag`.
    pub temporal_id_nesting_flag: bool,
    /// Parsed `profile_tier_level()` subroutine.
    pub ptl: ProfileTierLevel,
    /// `sps_seq_parameter_set_id` (`ue(v)`, range 0..=15).
    pub sps_id: u8,
    /// `chroma_format_idc` (`ue(v)`, range 0..=3). 0 monochrome, 1
    /// 4:2:0, 2 4:2:2, 3 4:4:4.
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`. Inferred to false when not
    /// signalled (which is whenever `chroma_format_idc != 3`).
    pub separate_colour_plane_flag: bool,
    /// `pic_width_in_luma_samples` (`ue(v)`).
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples` (`ue(v)`).
    pub pic_height_in_luma_samples: u32,
    /// `conformance_window_flag`.
    pub conformance_window_flag: bool,
    /// `conf_win_*_offset` triple plus bottom — zeroed when
    /// `conformance_window_flag` is false (§7.4.3.2.1).
    pub conformance_window: ConformanceWindow,
    /// `bit_depth_luma_minus8` (`ue(v)`, range 0..=8). The decoded
    /// `BitDepthY` is `8 + value`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8` (`ue(v)`, range 0..=8).
    pub bit_depth_chroma_minus8: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4` (`ue(v)`, range 0..=12).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// `sps_sub_layer_ordering_info_present_flag`.
    pub sub_layer_ordering_info_present_flag: bool,
    /// Per-sub-layer DPB / reorder / latency triples. Indices outside
    /// `0..=max_sub_layers_minus1` are zero-initialised; when the
    /// present flag was 0 every lower-indexed entry is copied from
    /// the `[max_sub_layers_minus1]` slot (§7.4.3.2.1).
    pub sub_layer_ordering_info: [SubLayerOrderingInfo; HEVC_MAX_SUB_LAYERS],
    /// `log2_min_luma_coding_block_size_minus3` (`ue(v)`).
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_luma_coding_block_size` (`ue(v)`). Per the
    /// `log2_ctb_size` derivation in §7.4.3.2.1, the CTU size is
    /// `1 << (3 + log2_min_cb_minus3 + log2_diff_max_min_luma_cb)`.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// `log2_min_luma_transform_block_size_minus2` (`ue(v)`).
    pub log2_min_luma_transform_block_size_minus2: u8,
    /// `log2_diff_max_min_luma_transform_block_size` (`ue(v)`).
    pub log2_diff_max_min_luma_transform_block_size: u8,
    /// `max_transform_hierarchy_depth_inter` (`ue(v)`).
    pub max_transform_hierarchy_depth_inter: u8,
    /// `max_transform_hierarchy_depth_intra` (`ue(v)`).
    pub max_transform_hierarchy_depth_intra: u8,
    /// `scaling_list_enabled_flag` (§7.3.2.2). When set, the SPS
    /// either carries an explicit [`Self::scaling_list_data`] (when
    /// `sps_scaling_list_data_present_flag == 1`) or the §7.4.5 default
    /// scaling lists apply.
    pub scaling_list_enabled_flag: bool,
    /// `sps_scaling_list_data_present_flag` (§7.3.2.2). Inferred to
    /// `false` (the §7.4.5 default lists apply) when
    /// `scaling_list_enabled_flag == 0`.
    pub sps_scaling_list_data_present_flag: bool,
    /// The parsed §7.3.4 `scaling_list_data()` structure when
    /// `sps_scaling_list_data_present_flag == 1`; `None` otherwise (the
    /// §7.4.5 default lists apply).
    pub scaling_list_data: Option<ScalingListData>,
    /// `amp_enabled_flag` — asymmetric motion partitions.
    pub amp_enabled_flag: bool,
    /// `sample_adaptive_offset_enabled_flag` — SPS-level gate for
    /// SAO; the per-slice gates are in the slice header.
    pub sample_adaptive_offset_enabled_flag: bool,
    /// `pcm_enabled_flag` — when set, [`Self::pcm`] is populated.
    pub pcm_enabled_flag: bool,
    /// `pcm_*` block per §7.3.2.2 (only meaningful when
    /// [`Self::pcm_enabled_flag`] is true).
    pub pcm: Option<PcmInfo>,
    /// `num_short_term_ref_pic_sets` (`ue(v)`, range 0..=64).
    pub num_short_term_ref_pic_sets: u32,
    /// Parsed `st_ref_pic_set()` entries, length
    /// [`Self::num_short_term_ref_pic_sets`].
    pub short_term_ref_pic_sets: Vec<ShortTermRefPicSet>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present_flag: bool,
    /// `num_long_term_ref_pics_sps` (`ue(v)`, range 0..=32). Zero
    /// when [`Self::long_term_ref_pics_present_flag`] is false.
    pub num_long_term_ref_pics_sps: u32,
    /// Per-entry long-term ref pic POC + used-by-curr-pic flag.
    /// Empty when [`Self::long_term_ref_pics_present_flag`] is false.
    pub long_term_ref_pics: Vec<LongTermRefPicEntry>,
    /// `sps_temporal_mvp_enabled_flag`.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing_enabled_flag: bool,
    /// `vui_parameters_present_flag`. When true, the §E.2.1
    /// `vui_parameters( )` body is decoded into
    /// [`Self::vui_parameters`] and parsing continues to
    /// `sps_extension_present_flag`.
    pub vui_parameters_present_flag: bool,
    /// Parsed §E.2.1 `vui_parameters()` body when
    /// [`Self::vui_parameters_present_flag`] is true; `None`
    /// otherwise.
    pub vui_parameters: Option<VuiParameters>,
    /// `sps_extension_present_flag`. Read in both the VUI-present and
    /// VUI-absent paths now that the VUI body is fully decoded. When
    /// true, the typed extension flag block is decoded into
    /// [`Self::extension_flags`]; any extension body that follows
    /// (plus the RBSP trailing bits) is surfaced as
    /// [`Self::opaque_tail`].
    pub sps_extension_present_flag: bool,
    /// Typed extension-flag block, decoded when
    /// `sps_extension_present_flag == 1` per §7.3.2.2.1. `None` when
    /// the gate is 0; every flag is then inferred to 0 per §7.4.3.2.1.
    pub extension_flags: Option<SpsExtensionFlags>,
    /// Decoded `sps_range_extension()` body (§7.3.2.2.2), present when
    /// `extension_flags.sps_range_extension_flag` is set. `None`
    /// otherwise; per §7.4.3.2.2 every field is then inferred to 0.
    pub sps_range_extension: Option<SpsRangeExtension>,
    /// Decoded `sps_scc_extension()` body (§7.3.2.2.3), present when
    /// `extension_flags.sps_scc_extension_flag` is set **and** no
    /// opaque multilayer / 3D body precedes it. `None` otherwise; per
    /// §7.4.3.2.3 every field is then inferred to 0 / empty.
    pub sps_scc_extension: Option<SpsSccExtension>,
    /// Opaque suffix of the SPS RBSP. Populated when
    /// `sps_extension_present_flag == 1` **and**
    /// [`SpsExtensionFlags::has_body`] is true on the decoded flags —
    /// the captured bytes start at the first set body
    /// (`sps_range_extension()` if `sps_range_extension_flag`,
    /// otherwise the next set flag's body) and run through
    /// `rbsp_trailing_bits()`. `None` when the SPS ended cleanly
    /// after `sps_extension_present_flag == 0` or after a typed flag
    /// block in which every flag is 0.
    pub opaque_tail: Option<OpaqueTail>,
}

impl SeqParameterSet {
    /// Parse `seq_parameter_set_rbsp()` starting from the first bit
    /// of the (already-unescaped) RBSP body — i.e. after the two-byte
    /// NAL header has been removed (see [`crate::hevc::engine::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, SpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br, rbsp)
    }

    /// Materialise the full SPS-level `short_term_ref_pic_sets[]`
    /// list into the post-§7.4.8 form, chaining inter-RPS-prediction
    /// entries through their `RefRpsIdx = stRpsIdx -
    /// (delta_idx_minus1 + 1)` source.
    ///
    /// The returned vector is the same length as
    /// [`Self::short_term_ref_pic_sets`], and the `idx`-th element is
    /// the materialisation of
    /// `self.short_term_ref_pic_sets[idx]`. Returns an error if any
    /// in-chain materialisation fails (e.g. `RefRpsIdx` underflow,
    /// `used_by_curr_pic_flag` / `use_delta_flag` length mismatch).
    pub fn materialize_short_term_ref_pic_sets(
        &self,
    ) -> Result<Vec<MaterializedShortTermRefPicSet>, ShortTermRefPicSetMaterializeError> {
        let mut out: Vec<MaterializedShortTermRefPicSet> =
            Vec::with_capacity(self.short_term_ref_pic_sets.len());
        for (st_rps_idx, rps) in self.short_term_ref_pic_sets.iter().enumerate() {
            // `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)` per
            // equation 7-59. For the explicit form `source` is unused
            // and `RefRpsIdx` is not derived. For the inter form
            // `delta_idx_minus1` was parsed as 0 for any SPS-resident
            // entry that did not signal it explicitly (the wire signal
            // is only present at the slice-inline call site), which
            // maps to the immediately-preceding entry.
            let source = if rps.inter_ref_pic_set_prediction_flag {
                let ref_rps_idx = (st_rps_idx as i64) - (rps.delta_idx_minus1 as i64 + 1);
                if ref_rps_idx < 0 {
                    return Err(ShortTermRefPicSetMaterializeError::MissingSource);
                }
                out.get(ref_rps_idx as usize)
            } else {
                None
            };
            out.push(rps.materialize(source)?);
        }
        Ok(out)
    }

    fn parse_inner(br: &mut BitReader<'_>, rbsp: &[u8]) -> Result<Self, SpsError> {
        let vps_id = br.u(4)? as u8;
        let max_sub_layers_minus1 = br.u(3)? as u8;
        if max_sub_layers_minus1 > 6 {
            return Err(SpsError::ValueOutOfRange {
                field: "sps_max_sub_layers_minus1",
                got: max_sub_layers_minus1 as u32,
            });
        }
        let temporal_id_nesting_flag = br.u1()? != 0;

        // profile_tier_level( 1, sps_max_sub_layers_minus1 )
        let ptl = ProfileTierLevel::parse(br, true, max_sub_layers_minus1)?;

        let sps_id_raw = br.ue()?;
        if sps_id_raw > 15 {
            return Err(SpsError::ValueOutOfRange {
                field: "sps_seq_parameter_set_id",
                got: sps_id_raw,
            });
        }
        let sps_id = sps_id_raw as u8;

        let chroma_format_idc_raw = br.ue()?;
        if chroma_format_idc_raw > 3 {
            return Err(SpsError::ValueOutOfRange {
                field: "chroma_format_idc",
                got: chroma_format_idc_raw,
            });
        }
        let chroma_format_idc = chroma_format_idc_raw as u8;

        let separate_colour_plane_flag = if chroma_format_idc == 3 {
            br.u1()? != 0
        } else {
            false
        };

        // §A.4.1 items b) / c): each dimension "shall be less than or
        // equal to Sqrt( MaxLumaPs * 8 )". The largest Table A.8
        // MaxLumaPs is 142 606 336 (levels 7 .. 7.2), giving
        // Sqrt( 142 606 336 * 8 ) = 33 776 (integer part). Enforcing
        // the ceiling here keeps every downstream PicWidthInCtbsY /
        // PicSizeInCtbsY derivation (eqs. 7-15 .. 7-19) inside u32.
        const MAX_LUMA_DIMENSION: u32 = 33_776;
        let pic_width_in_luma_samples = br.ue()?;
        if pic_width_in_luma_samples == 0 || pic_width_in_luma_samples > MAX_LUMA_DIMENSION {
            return Err(SpsError::ValueOutOfRange {
                field: "pic_width_in_luma_samples",
                got: pic_width_in_luma_samples,
            });
        }
        let pic_height_in_luma_samples = br.ue()?;
        if pic_height_in_luma_samples == 0 || pic_height_in_luma_samples > MAX_LUMA_DIMENSION {
            return Err(SpsError::ValueOutOfRange {
                field: "pic_height_in_luma_samples",
                got: pic_height_in_luma_samples,
            });
        }

        let conformance_window_flag = br.u1()? != 0;
        let conformance_window = if conformance_window_flag {
            ConformanceWindow {
                left_offset: br.ue()?,
                right_offset: br.ue()?,
                top_offset: br.ue()?,
                bottom_offset: br.ue()?,
            }
        } else {
            ConformanceWindow::default()
        };

        let bit_depth_luma_minus8_raw = br.ue()?;
        if bit_depth_luma_minus8_raw > 8 {
            return Err(SpsError::ValueOutOfRange {
                field: "bit_depth_luma_minus8",
                got: bit_depth_luma_minus8_raw,
            });
        }
        let bit_depth_chroma_minus8_raw = br.ue()?;
        if bit_depth_chroma_minus8_raw > 8 {
            return Err(SpsError::ValueOutOfRange {
                field: "bit_depth_chroma_minus8",
                got: bit_depth_chroma_minus8_raw,
            });
        }

        let log2_max_pic_order_cnt_lsb_minus4_raw = br.ue()?;
        if log2_max_pic_order_cnt_lsb_minus4_raw > 12 {
            return Err(SpsError::ValueOutOfRange {
                field: "log2_max_pic_order_cnt_lsb_minus4",
                got: log2_max_pic_order_cnt_lsb_minus4_raw,
            });
        }

        let sub_layer_ordering_info_present_flag = br.u1()? != 0;
        let last = max_sub_layers_minus1 as usize;
        let start = if sub_layer_ordering_info_present_flag {
            0usize
        } else {
            last
        };
        let mut sub_layer_ordering_info = [SubLayerOrderingInfo::default(); HEVC_MAX_SUB_LAYERS];
        for entry in sub_layer_ordering_info
            .iter_mut()
            .take(last + 1)
            .skip(start)
        {
            let max_dpb = br.ue()?;
            let max_reorder = br.ue()?;
            let max_lat = br.ue()?;
            *entry = SubLayerOrderingInfo {
                max_dec_pic_buffering_minus1: max_dpb,
                max_num_reorder_pics: max_reorder,
                max_latency_increase_plus1: max_lat,
            };
        }
        if !sub_layer_ordering_info_present_flag {
            // §7.4.3.2.1: when the present flag is 0, every lower-indexed
            // sub-layer inherits the [max_sub_layers_minus1] triple.
            let copy = sub_layer_ordering_info[last];
            for entry in sub_layer_ordering_info.iter_mut().take(last) {
                *entry = copy;
            }
        }

        let log2_min_luma_coding_block_size_minus3_raw = br.ue()?;
        let log2_diff_max_min_luma_coding_block_size_raw = br.ue()?;
        // §7.4.3.2.1 eqs. 7-10 / 7-11: MinCbLog2SizeY =
        // log2_min_luma_coding_block_size_minus3 + 3 and CtbLog2SizeY =
        // MinCbLog2SizeY + log2_diff_max_min_luma_coding_block_size.
        // Every Annex A profile requires "CtbLog2SizeY derived
        // according to active SPSs ... shall be in the range of 4 to 6,
        // inclusive" (e.g. the §A.3.2 Main-profile item), and the
        // eq.-7-13 `CtbSizeY = 1 << CtbLog2SizeY` shift (re-derived all
        // over the slice/CTB layers) is only meaningful under that
        // bound — reject out-of-range values here.
        let ctb_log2_size_y = log2_min_luma_coding_block_size_minus3_raw
            .saturating_add(3)
            .saturating_add(log2_diff_max_min_luma_coding_block_size_raw);
        if !(4..=6).contains(&ctb_log2_size_y) {
            return Err(SpsError::ValueOutOfRange {
                field: "CtbLog2SizeY",
                got: ctb_log2_size_y,
            });
        }
        let log2_min_luma_coding_block_size_minus3 =
            log2_min_luma_coding_block_size_minus3_raw as u8;
        let log2_diff_max_min_luma_coding_block_size =
            log2_diff_max_min_luma_coding_block_size_raw as u8;
        let min_cb_log2_size_y = u32::from(log2_min_luma_coding_block_size_minus3) + 3;

        let log2_min_luma_transform_block_size_minus2_raw = br.ue()?;
        // §7.4.3.2.1: "The CVS shall not contain data that result in
        // MinTbLog2SizeY greater than or equal to MinCbLog2SizeY"
        // (MinTbLog2SizeY = log2_min_luma_transform_block_size_minus2
        // + 2).
        let min_tb_log2_size_y = log2_min_luma_transform_block_size_minus2_raw.saturating_add(2);
        if min_tb_log2_size_y >= min_cb_log2_size_y {
            return Err(SpsError::ValueOutOfRange {
                field: "log2_min_luma_transform_block_size_minus2",
                got: log2_min_luma_transform_block_size_minus2_raw,
            });
        }
        let log2_min_luma_transform_block_size_minus2 =
            log2_min_luma_transform_block_size_minus2_raw as u8;

        let log2_diff_max_min_luma_transform_block_size_raw = br.ue()?;
        // §7.4.3.2.1: "The CVS shall not contain data that result in
        // MaxTbLog2SizeY greater than Min( CtbLog2SizeY, 5 )".
        let max_tb_log2_size_y =
            min_tb_log2_size_y.saturating_add(log2_diff_max_min_luma_transform_block_size_raw);
        if max_tb_log2_size_y > ctb_log2_size_y.min(5) {
            return Err(SpsError::ValueOutOfRange {
                field: "log2_diff_max_min_luma_transform_block_size",
                got: log2_diff_max_min_luma_transform_block_size_raw,
            });
        }
        let log2_diff_max_min_luma_transform_block_size =
            log2_diff_max_min_luma_transform_block_size_raw as u8;

        // §7.4.3.2.1: both hierarchy depths "shall be in the range of
        // 0 to CtbLog2SizeY − MinTbLog2SizeY, inclusive".
        let max_hierarchy_depth = ctb_log2_size_y - min_tb_log2_size_y;
        let max_transform_hierarchy_depth_inter_raw = br.ue()?;
        if max_transform_hierarchy_depth_inter_raw > max_hierarchy_depth {
            return Err(SpsError::ValueOutOfRange {
                field: "max_transform_hierarchy_depth_inter",
                got: max_transform_hierarchy_depth_inter_raw,
            });
        }
        let max_transform_hierarchy_depth_inter = max_transform_hierarchy_depth_inter_raw as u8;
        let max_transform_hierarchy_depth_intra_raw = br.ue()?;
        if max_transform_hierarchy_depth_intra_raw > max_hierarchy_depth {
            return Err(SpsError::ValueOutOfRange {
                field: "max_transform_hierarchy_depth_intra",
                got: max_transform_hierarchy_depth_intra_raw,
            });
        }
        let max_transform_hierarchy_depth_intra = max_transform_hierarchy_depth_intra_raw as u8;

        let scaling_list_enabled_flag = br.u1()? != 0;
        let mut sps_scaling_list_data_present_flag = false;
        let mut scaling_list_data = None;
        if scaling_list_enabled_flag {
            // §7.3.2.2: when scaling_list_enabled_flag == 1, an inner
            // sps_scaling_list_data_present_flag gates the explicit
            // scaling_list_data() structure (§7.3.4). When the inner
            // flag is 0 the default scaling lists (§7.4.5 Tables 7-5 /
            // 7-6) apply, so the SPS still parses.
            sps_scaling_list_data_present_flag = br.u1()? != 0;
            if sps_scaling_list_data_present_flag {
                scaling_list_data = Some(ScalingListData::parse(br)?);
            }
        }

        let amp_enabled_flag = br.u1()? != 0;
        let sample_adaptive_offset_enabled_flag = br.u1()? != 0;

        let pcm_enabled_flag = br.u1()? != 0;
        let pcm = if pcm_enabled_flag {
            let bit_depth_luma_minus1 = br.u(4)? as u8;
            let pcm_bit_depth_y = bit_depth_luma_minus1 as u32 + 1;
            let bit_depth_y = 8 + bit_depth_luma_minus8_raw;
            if pcm_bit_depth_y > bit_depth_y {
                return Err(SpsError::ValueOutOfRange {
                    field: "pcm_sample_bit_depth_luma_minus1",
                    got: bit_depth_luma_minus1 as u32,
                });
            }
            let bit_depth_chroma_minus1 = br.u(4)? as u8;
            let pcm_bit_depth_c = bit_depth_chroma_minus1 as u32 + 1;
            let bit_depth_c = 8 + bit_depth_chroma_minus8_raw;
            if pcm_bit_depth_c > bit_depth_c {
                return Err(SpsError::ValueOutOfRange {
                    field: "pcm_sample_bit_depth_chroma_minus1",
                    got: bit_depth_chroma_minus1 as u32,
                });
            }
            let log2_min_pcm_luma_coding_block_size_minus3 = br.ue()? as u8;
            let log2_diff_max_min_pcm_luma_coding_block_size = br.ue()? as u8;
            let loop_filter_disabled_flag = br.u1()? != 0;
            Some(PcmInfo {
                bit_depth_luma_minus1,
                bit_depth_chroma_minus1,
                log2_min_pcm_luma_coding_block_size_minus3,
                log2_diff_max_min_pcm_luma_coding_block_size,
                loop_filter_disabled_flag,
            })
        } else {
            None
        };

        let num_short_term_ref_pic_sets_raw = br.ue()?;
        if num_short_term_ref_pic_sets_raw > HEVC_MAX_NUM_SHORT_TERM_RPS as u32 {
            return Err(SpsError::ValueOutOfRange {
                field: "num_short_term_ref_pic_sets",
                got: num_short_term_ref_pic_sets_raw,
            });
        }
        let num_short_term_ref_pic_sets = num_short_term_ref_pic_sets_raw;
        let mut short_term_ref_pic_sets = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
        for st_rps_idx in 0..num_short_term_ref_pic_sets as usize {
            let prev = if st_rps_idx == 0 {
                None
            } else {
                short_term_ref_pic_sets.last()
            };
            let rps = ShortTermRefPicSet::parse(
                br,
                st_rps_idx as u32,
                num_short_term_ref_pic_sets,
                prev,
                &short_term_ref_pic_sets,
            )?;
            short_term_ref_pic_sets.push(rps);
        }

        let long_term_ref_pics_present_flag = br.u1()? != 0;
        let mut num_long_term_ref_pics_sps = 0u32;
        let mut long_term_ref_pics = Vec::new();
        if long_term_ref_pics_present_flag {
            let raw = br.ue()?;
            if raw > HEVC_MAX_NUM_LONG_TERM_RPS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_long_term_ref_pics_sps",
                    got: raw,
                });
            }
            num_long_term_ref_pics_sps = raw;
            let poc_lsb_bits = log2_max_pic_order_cnt_lsb_minus4_raw as u8 + 4;
            long_term_ref_pics.reserve(num_long_term_ref_pics_sps as usize);
            for _ in 0..num_long_term_ref_pics_sps {
                let poc_lsb = br.u(poc_lsb_bits)?;
                let used = br.u1()? != 0;
                long_term_ref_pics.push(LongTermRefPicEntry {
                    poc_lsb,
                    used_by_curr_pic: used,
                });
            }
        }

        let sps_temporal_mvp_enabled_flag = br.u1()? != 0;
        let strong_intra_smoothing_enabled_flag = br.u1()? != 0;

        let vui_parameters_present_flag = br.u1()? != 0;
        // §E.2.1: the vui_parameters() body is decoded in full when
        // signalled, with the nested hrd_parameters( 1,
        // sps_max_sub_layers_minus1 ) call taking the SPS-level
        // maxNumSubLayersMinus1. Parsing then continues to
        // sps_extension_present_flag in both paths.
        let vui_parameters = if vui_parameters_present_flag {
            Some(VuiParameters::parse(br, max_sub_layers_minus1)?)
        } else {
            None
        };

        let (
            sps_extension_present_flag,
            extension_flags,
            sps_range_extension,
            sps_scc_extension,
            opaque_tail,
        ) = if br.bits_left() == 0 {
            // The fixture corpus encoders sometimes elide the
            // sps_extension_present_flag if no extension is signalled
            // and the rbsp_trailing_bits happens to land on a byte
            // boundary; the field is still required, so a buffer with
            // no bits left here is a truncation.
            return Err(SpsError::Truncated);
        } else {
            let gate = br.u1()? != 0;
            if gate {
                // §7.3.2.2.1: when the gate is open, decode the eight
                // bits of typed extension flags first.
                let sps_range_extension_flag = br.u1()? != 0;
                let sps_multilayer_extension_flag = br.u1()? != 0;
                let sps_3d_extension_flag = br.u1()? != 0;
                let sps_scc_extension_flag = br.u1()? != 0;
                let sps_extension_4bits = br.u(4)? as u8;
                let flags = SpsExtensionFlags {
                    sps_range_extension_flag,
                    sps_multilayer_extension_flag,
                    sps_3d_extension_flag,
                    sps_scc_extension_flag,
                    sps_extension_4bits,
                };
                // §7.3.2.2.1: the range extension body (if signalled)
                // is the first to follow the eight typed flag bits, so
                // decode it in full.
                let range_ext = if flags.sps_range_extension_flag {
                    Some(SpsRangeExtension::parse(br)?)
                } else {
                    None
                };
                // §7.3.2.2.1 body order is range, multilayer, 3d, scc.
                // The SCC body can be decoded in place only when no
                // (still-opaque) multilayer / 3D body precedes it;
                // otherwise it stays inside the opaque tail.
                let scc_ext = if flags.scc_decodable_in_place() {
                    Some(SpsSccExtension::parse(
                        br,
                        chroma_format_idc,
                        8 + bit_depth_luma_minus8_raw as u8,
                        8 + bit_depth_chroma_minus8_raw as u8,
                    )?)
                } else {
                    None
                };
                // If any still-opaque body (a multilayer / 3D body, an
                // SCC body kept opaque by such a predecessor, or the
                // sps_extension_data_flag while-loop) follows, capture
                // the rest of the RBSP as an opaque tail starting at
                // the first un-decoded body's bit position. Otherwise
                // only rbsp_trailing_bits remains, consumed implicitly.
                let tail = if flags.has_opaque_body_after_decoded() {
                    Some(OpaqueTail::capture_at(br.bit_pos(), rbsp))
                } else {
                    None
                };
                (true, Some(flags), range_ext, scc_ext, tail)
            } else {
                // No extension present. Only the rbsp_trailing_bits
                // remain — a single `1` bit followed by zero-padding
                // to a byte boundary. We do not require the caller to
                // have validated it; surface nothing for the opaque tail.
                (false, None, None, None, None)
            }
        };

        Ok(Self {
            vps_id,
            max_sub_layers_minus1,
            temporal_id_nesting_flag,
            ptl,
            sps_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            conformance_window_flag,
            conformance_window,
            bit_depth_luma_minus8: bit_depth_luma_minus8_raw as u8,
            bit_depth_chroma_minus8: bit_depth_chroma_minus8_raw as u8,
            log2_max_pic_order_cnt_lsb_minus4: log2_max_pic_order_cnt_lsb_minus4_raw as u8,
            sub_layer_ordering_info_present_flag,
            sub_layer_ordering_info,
            log2_min_luma_coding_block_size_minus3,
            log2_diff_max_min_luma_coding_block_size,
            log2_min_luma_transform_block_size_minus2,
            log2_diff_max_min_luma_transform_block_size,
            max_transform_hierarchy_depth_inter,
            max_transform_hierarchy_depth_intra,
            scaling_list_enabled_flag,
            sps_scaling_list_data_present_flag,
            scaling_list_data,
            amp_enabled_flag,
            sample_adaptive_offset_enabled_flag,
            pcm_enabled_flag,
            pcm,
            num_short_term_ref_pic_sets,
            short_term_ref_pic_sets,
            long_term_ref_pics_present_flag,
            num_long_term_ref_pics_sps,
            long_term_ref_pics,
            sps_temporal_mvp_enabled_flag,
            strong_intra_smoothing_enabled_flag,
            vui_parameters_present_flag,
            vui_parameters,
            sps_extension_present_flag,
            extension_flags,
            sps_range_extension,
            sps_scc_extension,
            opaque_tail,
        })
    }

    /// Convenience: derive the SPS-level `BitDepthY` (luma bit depth).
    pub fn bit_depth_luma(&self) -> u8 {
        8 + self.bit_depth_luma_minus8
    }

    /// Derive the SPS-level `BitDepthC` (chroma bit depth).
    pub fn bit_depth_chroma(&self) -> u8 {
        8 + self.bit_depth_chroma_minus8
    }

    /// Derive `MinCbLog2SizeY = log2_min_luma_coding_block_size_minus3 + 3`
    /// (§7.4.3.2.1).
    pub fn log2_min_cb_size(&self) -> u8 {
        self.log2_min_luma_coding_block_size_minus3 + 3
    }

    /// Derive `CtbLog2SizeY = MinCbLog2SizeY
    /// + log2_diff_max_min_luma_coding_block_size` (§7.4.3.2.1).
    pub fn log2_ctb_size(&self) -> u8 {
        self.log2_min_cb_size() + self.log2_diff_max_min_luma_coding_block_size
    }

    /// Derive `MinTbLog2SizeY = log2_min_luma_transform_block_size_minus2 + 2`
    /// (§7.4.3.2.1).
    pub fn log2_min_tb_size(&self) -> u8 {
        self.log2_min_luma_transform_block_size_minus2 + 2
    }

    /// Derive `MaxPicOrderCntLsb = 1 << (log2_max_pic_order_cnt_lsb_minus4 + 4)`
    /// (§7.4.3.2.1).
    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        1u32 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
}

impl OpaqueTail {
    /// Capture all RBSP bytes from the byte holding the bit at
    /// `bit_pos` (counted MSB-first from the start of `rbsp`) through
    /// end-of-buffer. Used by both the SPS extension tail and the
    /// [`crate::hevc::engine::pps::PicParameterSet`] extension tail.
    pub fn capture_at(bit_pos: usize, rbsp: &[u8]) -> Self {
        let byte_index = bit_pos / 8;
        let bit_in_byte = (bit_pos % 8) as u8;
        Self {
            bytes: rbsp[byte_index..].to_vec(),
            start_bit_in_first_byte: bit_in_byte,
        }
    }
}

impl ShortTermRefPicSet {
    /// Parse the in-line slice-header `st_ref_pic_set( num_short_term_ref_pic_sets )`
    /// per §7.3.6.1 / §7.3.7.
    ///
    /// This is the entry point the slice-header parser uses when
    /// `short_term_ref_pic_set_sps_flag == 0`: the picture's short-term
    /// RPS is constructed *inline* in the slice header at index
    /// `stRpsIdx == num_short_term_ref_pic_sets`, with the SPS's
    /// pre-existing list ([`SeqParameterSet::short_term_ref_pic_sets`])
    /// supplying `all_rps` for the §7.4.8 `RefRpsIdx` derivation.
    ///
    /// `br` must be positioned at the first bit of `st_ref_pic_set()`.
    /// On success the reader is advanced past the structure; on a
    /// truncation or range-check failure the reader state is undefined.
    pub fn parse_slice_inline(
        br: &mut BitReader<'_>,
        sps: &SeqParameterSet,
    ) -> Result<Self, SpsError> {
        Self::parse(
            br,
            sps.num_short_term_ref_pic_sets,
            sps.num_short_term_ref_pic_sets,
            sps.short_term_ref_pic_sets.last(),
            &sps.short_term_ref_pic_sets,
        )
    }

    /// Parse one `st_ref_pic_set( stRpsIdx )` per §7.3.7.
    ///
    /// * `st_rps_idx` is `stRpsIdx`.
    /// * `num_short_term_ref_pic_sets` is the SPS-level count being
    ///   constructed (used to detect when `delta_idx_minus1` is signalled).
    /// * `prev` is the previously-parsed RPS, used when the
    ///   inter-RPS-prediction form is invoked without explicit
    ///   `delta_idx_minus1` (i.e. `stRpsIdx < num_short_term_ref_pic_sets`).
    /// * `all_rps` is the full list of RPSes parsed so far; the
    ///   §7.4.8 `RefRpsIdx` computation indexes into it.
    fn parse(
        br: &mut BitReader<'_>,
        st_rps_idx: u32,
        num_short_term_ref_pic_sets: u32,
        prev: Option<&ShortTermRefPicSet>,
        all_rps: &[ShortTermRefPicSet],
    ) -> Result<Self, SpsError> {
        let inter_ref_pic_set_prediction_flag = if st_rps_idx != 0 {
            br.u1()? != 0
        } else {
            false
        };
        if inter_ref_pic_set_prediction_flag {
            // delta_idx_minus1 is only signalled when the RPS being
            // constructed is the slice-header in-line RPS, i.e.
            // stRpsIdx == num_short_term_ref_pic_sets. For SPS-resident
            // entries the value is inferred to 0 per §7.4.8.
            let delta_idx_minus1 = if st_rps_idx == num_short_term_ref_pic_sets {
                br.ue()?
            } else {
                0
            };
            if delta_idx_minus1 >= st_rps_idx {
                return Err(SpsError::ValueOutOfRange {
                    field: "delta_idx_minus1",
                    got: delta_idx_minus1,
                });
            }
            let delta_rps_sign = br.u1()? != 0;
            let abs_delta_rps_minus1 = br.ue()?;
            if abs_delta_rps_minus1 > (1 << 15) - 1 {
                return Err(SpsError::ValueOutOfRange {
                    field: "abs_delta_rps_minus1",
                    got: abs_delta_rps_minus1,
                });
            }
            // RefRpsIdx = stRpsIdx − (delta_idx_minus1 + 1)
            let ref_rps_idx = (st_rps_idx as i64) - (delta_idx_minus1 as i64 + 1);
            let ref_rps = if ref_rps_idx >= 0 && (ref_rps_idx as usize) < all_rps.len() {
                Some(&all_rps[ref_rps_idx as usize])
            } else {
                // For SPS entries we expect ref_rps_idx in-range; the
                // only legal use of an out-of-range RefRpsIdx is when
                // st_rps_idx == num_short_term_ref_pic_sets, which is
                // the slice-header in-line case (handled elsewhere).
                prev
            };
            let num_delta_pocs = ref_rps.map(|r| r.num_delta_pocs()).unwrap_or(0);
            let entries = num_delta_pocs as usize + 1;
            let mut used_by_curr_pic_flag = Vec::with_capacity(entries);
            let mut use_delta_flag = Vec::with_capacity(entries);
            for _ in 0..entries {
                let used = br.u1()? != 0;
                used_by_curr_pic_flag.push(used);
                if !used {
                    let ud = br.u1()? != 0;
                    use_delta_flag.push(ud);
                } else {
                    // Per §7.4.8: when used_by_curr_pic_flag[j] is 1,
                    // use_delta_flag[j] is inferred to be 1.
                    use_delta_flag.push(true);
                }
            }
            Ok(Self {
                inter_ref_pic_set_prediction_flag,
                delta_idx_minus1,
                delta_rps_sign,
                abs_delta_rps_minus1,
                used_by_curr_pic_flag,
                use_delta_flag,
                num_negative_pics: 0,
                num_positive_pics: 0,
                delta_poc_s0_minus1: Vec::new(),
                used_by_curr_pic_s0_flag: Vec::new(),
                delta_poc_s1_minus1: Vec::new(),
                used_by_curr_pic_s1_flag: Vec::new(),
            })
        } else {
            let num_negative_pics = br.ue()?;
            if num_negative_pics > HEVC_MAX_RPS_PICS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_negative_pics",
                    got: num_negative_pics,
                });
            }
            let num_positive_pics = br.ue()?;
            if num_positive_pics > HEVC_MAX_RPS_PICS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_positive_pics",
                    got: num_positive_pics,
                });
            }
            let mut delta_poc_s0_minus1 = Vec::with_capacity(num_negative_pics as usize);
            let mut used_by_curr_pic_s0_flag = Vec::with_capacity(num_negative_pics as usize);
            for _ in 0..num_negative_pics {
                let dp = br.ue()?;
                if dp > (1 << 15) - 1 {
                    return Err(SpsError::ValueOutOfRange {
                        field: "delta_poc_s0_minus1",
                        got: dp,
                    });
                }
                delta_poc_s0_minus1.push(dp);
                used_by_curr_pic_s0_flag.push(br.u1()? != 0);
            }
            let mut delta_poc_s1_minus1 = Vec::with_capacity(num_positive_pics as usize);
            let mut used_by_curr_pic_s1_flag = Vec::with_capacity(num_positive_pics as usize);
            for _ in 0..num_positive_pics {
                let dp = br.ue()?;
                if dp > (1 << 15) - 1 {
                    return Err(SpsError::ValueOutOfRange {
                        field: "delta_poc_s1_minus1",
                        got: dp,
                    });
                }
                delta_poc_s1_minus1.push(dp);
                used_by_curr_pic_s1_flag.push(br.u1()? != 0);
            }
            Ok(Self {
                inter_ref_pic_set_prediction_flag,
                delta_idx_minus1: 0,
                delta_rps_sign: false,
                abs_delta_rps_minus1: 0,
                used_by_curr_pic_flag: Vec::new(),
                use_delta_flag: Vec::new(),
                num_negative_pics,
                num_positive_pics,
                delta_poc_s0_minus1,
                used_by_curr_pic_s0_flag,
                delta_poc_s1_minus1,
                used_by_curr_pic_s1_flag,
            })
        }
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::nal::{collect_nal_units, strip_emulation_prevention};

    /// Helper: convert a bit string (any non-`0`/`1` characters are
    /// ignored, useful for visual spacing) into a packed MSB-first
    /// byte vector with zero-padding up to the next byte boundary.
    fn bits_to_bytes(s: &str) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for c in s.chars() {
            if c == '0' || c == '1' {
                bits.push((c as u8) - b'0');
            }
        }
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut out = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            out.push(b);
        }
        out
    }

    /// Header bits through the per-sub-layer ordering triple, with the
    /// `pic_width_in_luma_samples` / `pic_height_in_luma_samples`
    /// `ue(v)` bit strings supplied by the caller (the fixture default
    /// is 16 × 16 → `000010001` each). Everything else is fixed:
    /// `vps_id=0, max_sub_layers_minus1=0, nesting=1, a §7.3.3 PTL
    /// walk (profile_idc=1, level=30), sps_id=0, chroma_format_idc=1,
    /// conf_win=0, bit_depths=0, log2_max_poc_lsb_minus4=4,
    /// ordering_info present with single triple {0,0,0}`.
    fn synthesised_header_through_ordering(width_ue: &str, height_ue: &str) -> String {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting flag
        // profile_tier_level(1, 0)
        s += "00"; // profile_space
        s += "0"; // tier
        s += "00001"; // profile_idc
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
        // sps_id ue(v)=0
        s += "1";
        // chroma_format_idc ue(v)=1 → '010'
        s += "010";
        // width ue(v)
        s += width_ue;
        // height ue(v)
        s += height_ue;
        // conf_win = 0
        s += "0";
        // bd_luma=0, bd_chroma=0
        s += "1";
        s += "1";
        // log2_max_poc_lsb_minus4 = 4 → '00101'
        s += "00101";
        // ordering present = 1, single triple {0,0,0}
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        s
    }

    /// Prefix of bits that get every SPS hand-assembled fixture through
    /// the structural header up to (but not including) the round-4 tail.
    /// `chroma_format_idc=1, width=16, height=16, conf_win=0, bit_depths=0,
    /// log2_max_poc_lsb_minus4=4, max_sub_layers_minus1=0, ordering_info
    /// present with single triple {dpb=0, reorder=0, latency=0},
    /// log2_min_cb=0, log2_diff=1, log2_min_tb=0, log2_diff_tb=2,
    /// max_transform_depth_*=0, scaling_list=0, amp=0, sao=1`.
    ///
    /// This is the EXACT same prefix the round-3 tests used; the round-4
    /// tail tests then concatenate the tail bits they want to exercise.
    fn synthesised_prefix_bits() -> String {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        // log2_min_cb_minus3 = 0
        s += "1";
        // log2_diff = 1 → '010'
        s += "010";
        // log2_min_tb_minus2 = 0
        s += "1";
        // log2_diff_tb = 2 → '011'
        s += "011";
        // max_transform_depth_{inter,intra} = 0
        s += "1";
        s += "1";
        // scaling_list_enabled = 0
        s += "0";
        // amp_enabled = 0
        s += "0";
        // sao_enabled = 1
        s += "1";
        s
    }

    /// SPS RBSP body extracted from
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed and emulation-prevention bytes stripped. The wire
    /// SPS (NAL idx 1, type 33) was 38 bytes including the two-byte
    /// header; after §7.4.1.1 strip (3 escape bytes removed) the body
    /// is 32 bytes.
    const TINY_SPS_RBSP: &[u8] = &[
        0x01, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00, 0x1E, 0xA0, 0x88,
        0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x32, 0x10,
    ];

    #[test]
    fn parses_tiny_fixture_sps() {
        let sps = SeqParameterSet::parse(TINY_SPS_RBSP).expect("SPS parse");
        // Trace cross-check (docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt):
        //   SPS sps_id=0 vps_id=0 max_sub_layers=1 profile_idc=4 level_idc=30
        //       chroma_format_idc=1 bit_depth=8 bit_depth_chroma=8
        //       width=16 height=16 log2_ctb_size=4 log2_min_cb_size=3
        //       log2_min_tb_size=2 sao_enabled=1 amp_enabled=0 pcm_enabled=0
        //       scaling_list_enabled=0 long_term_ref_pics=0 temporal_mvp=1
        //       strong_intra_smoothing=1
        assert_eq!(sps.vps_id, 0);
        assert_eq!(sps.max_sub_layers_minus1, 0); // max_sub_layers = 1
        assert!(sps.temporal_id_nesting_flag);
        assert_eq!(sps.ptl.general_profile_idc, 4);
        assert_eq!(sps.ptl.general_level_idc, 30);
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert!(!sps.conformance_window_flag);
        assert_eq!(sps.conformance_window, ConformanceWindow::default());
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert_eq!(sps.bit_depth_luma(), 8);
        assert_eq!(sps.bit_depth_chroma(), 8);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 4); // MaxPicOrderCntLsb = 256
        assert_eq!(sps.max_pic_order_cnt_lsb(), 256);
        assert!(sps.sub_layer_ordering_info_present_flag);
        assert_eq!(
            sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(sps.sub_layer_ordering_info[0].max_num_reorder_pics, 0);
        assert_eq!(sps.sub_layer_ordering_info[0].max_latency_increase_plus1, 1);
        assert_eq!(sps.log2_min_cb_size(), 3); // 8x8 minimum CU
        assert_eq!(sps.log2_ctb_size(), 4); // 16x16 CTU
        assert_eq!(sps.log2_min_tb_size(), 2); // 4x4 minimum TU
        assert_eq!(sps.max_transform_hierarchy_depth_inter, 0);
        assert_eq!(sps.max_transform_hierarchy_depth_intra, 0);
        assert!(!sps.scaling_list_enabled_flag);
        assert!(!sps.amp_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        // Round-4 tail.
        assert!(!sps.pcm_enabled_flag);
        assert!(sps.pcm.is_none());
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(sps.short_term_ref_pic_sets.is_empty());
        assert!(!sps.long_term_ref_pics_present_flag);
        assert_eq!(sps.num_long_term_ref_pics_sps, 0);
        assert!(sps.long_term_ref_pics.is_empty());
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
        // The fixture's x265 CLI encode signals a §E.2.1 VUI body. It
        // decodes to a square (1:1) sample aspect ratio, an
        // unspecified-but-present video_signal_type, and a timing-info
        // block of vui_num_units_in_tick = 1 / vui_time_scale = 25
        // (25 fps) with neither HRD nor bitstream restriction. After
        // the VUI the SPS ends cleanly: sps_extension_present_flag == 0
        // and only the rbsp_trailing_bits() remain, so no opaque tail
        // is captured.
        assert!(sps.vui_parameters_present_flag);
        let vui = sps.vui_parameters.as_ref().expect("VUI body");
        assert!(vui.aspect_ratio_info_present_flag);
        assert_eq!(vui.aspect_ratio_idc, 1); // 1:1 square
        assert!(vui.sar_width.is_none());
        assert!(!vui.overscan_info_present_flag);
        assert!(vui.video_signal_type_present_flag);
        let vst = vui.video_signal_type.as_ref().expect("video signal type");
        assert_eq!(vst.video_format, 5); // unspecified
        assert!(!vst.video_full_range_flag);
        assert!(vst.colour_description.is_none());
        assert!(!vui.chroma_loc_info_present_flag);
        assert!(!vui.neutral_chroma_indication_flag);
        assert!(!vui.field_seq_flag);
        assert!(!vui.frame_field_info_present_flag);
        assert!(!vui.default_display_window_flag);
        assert!(vui.vui_timing_info_present_flag);
        let ti = vui.timing_info.as_ref().expect("timing info");
        assert_eq!(ti.num_units_in_tick, 1);
        assert_eq!(ti.time_scale, 25);
        assert!(!ti.poc_proportional_to_timing_flag);
        assert!(!ti.hrd_parameters_present_flag);
        assert!(!vui.bitstream_restriction_flag);
        // No extension, no opaque tail.
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// End-to-end: pull the SPS NAL out of the raw Annex B stream
    /// through the walker, then parse it. The fixture's full byte
    /// sequence is captured inline so the test does not depend on the
    /// `docs/` tree at run time.
    #[test]
    fn parses_tiny_fixture_sps_via_nal_walker() {
        // VPS NAL + SPS NAL only — the rest of the fixture (PPS / SEI
        // / slice) is unrelated to this test.
        let raw = &[
            // VPS: 00 00 00 01 40 01 ...
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00,
            0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
            // SPS: 00 00 00 01 42 01 ...
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F,
            0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xA0, 0x88, 0x45, 0x96, 0xEA, 0xAF, 0x2B,
            0xC0, 0x5A, 0x02, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x32, 0x10,
        ];
        let units = collect_nal_units(raw).expect("walker");
        assert_eq!(units.len(), 2);
        assert_eq!(units[1].header.nal_unit_type, 33); // SPS_NUT
        let sps = SeqParameterSet::parse(&units[1].rbsp).expect("SPS parse");
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert_eq!(sps.log2_ctb_size(), 4);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
        assert!(sps.vui_parameters_present_flag);
    }

    #[test]
    fn strip_emulation_prevention_then_parse_matches_inline_decode() {
        // The wire SPS body (post NAL header, with the 3 emulation-
        // prevention escapes left in) must, after a §7.4.1.1 strip,
        // match TINY_SPS_RBSP exactly.
        let wire = &[
            0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00,
            0x1E, 0xA0, 0x88, 0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x03,
            0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x32, 0x10,
        ];
        let unesc = strip_emulation_prevention(wire);
        assert_eq!(unesc, TINY_SPS_RBSP);
        let a = SeqParameterSet::parse(&unesc).expect("SPS parse");
        let b = SeqParameterSet::parse(TINY_SPS_RBSP).expect("SPS parse");
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_truncated_rbsp() {
        // Cut the buffer just past the leading `vps_id / max_sub /
        // nesting` byte — well before profile_tier_level finishes.
        let err = SeqParameterSet::parse(&TINY_SPS_RBSP[..3]).unwrap_err();
        assert_eq!(err, SpsError::Truncated);
    }

    /// Hand-assembled SPS exercising the `chroma_format_idc == 3`
    /// path (so `separate_colour_plane_flag` is signalled) plus the
    /// `conformance_window_flag == 1` four-`ue(v)` block. The remaining
    /// fields are kept at minimal values to make the bit string
    /// hand-traceable.
    #[test]
    fn parses_444_with_conformance_window() {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting
        // PTL(1,0)
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110";
        // sps_id=0
        s += "1";
        // chroma_format_idc=3 → '00100'
        s += "00100";
        // separate_colour_plane_flag=1
        s += "1";
        // width=16, height=16
        s += "000010001";
        s += "000010001";
        // conf_win=1, four ue(v)=0
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        // bd_luma=2 → '011'
        s += "011";
        s += "011";
        // log2_max_poc_lsb_minus4=4
        s += "00101";
        // ordering present, single triple of zeros
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        // log2_min_cb=0
        s += "1";
        // log2_diff=1
        s += "010";
        s += "1";
        s += "011";
        s += "1";
        s += "1";
        // scaling=0
        s += "0";
        // amp=1
        s += "1";
        // sao=1
        s += "1";
        // pcm_enabled=0
        s += "0";
        // num_short_term_ref_pic_sets=0
        s += "1";
        // long_term_ref_pics_present=0
        s += "0";
        // temporal_mvp=1
        s += "1";
        // strong_intra_smoothing=0
        s += "0";
        // vui=0
        s += "0";
        // sps_extension_present=0
        s += "0";
        // rbsp trailing bits (stop bit then zero pad)
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.chroma_format_idc, 3);
        assert!(sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma(), 10);
        assert!(sps.amp_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(!sps.strong_intra_smoothing_enabled_flag);
        assert!(!sps.vui_parameters_present_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// Hand-assembled SPS with two sub-layers and the ordering-info
    /// present flag set to 0 — the [0] sub-layer must inherit the
    /// [1] triple per §7.4.3.2.1.
    #[test]
    fn ordering_info_present_flag_zero_propagates() {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "001"; // max_sub_layers_minus1 = 1
        s += "1";
        // PTL(1, 1):
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
        // sub_layer_profile_present[0]/level_present[0] = 0,0
        s += "00";
        for _ in 0..14 {
            s += "0";
        }
        // sps_id=0
        s += "1";
        // chroma=1
        s += "010";
        // width=16, height=16
        s += "000010001";
        s += "000010001";
        // conf_win=0
        s += "0";
        // bd_luma=0
        s += "1";
        // bd_chroma=0
        s += "1";
        // log2_max_poc_lsb_minus4=4
        s += "00101";
        // sub_layer_ordering_info_present_flag = 0
        s += "0";
        // single triple at i=max_sub_layers_minus1 (=1): dpb=2 ('011'), reorder=0 ('1'), latency=0 ('1')
        s += "011";
        s += "1";
        s += "1";
        s += "1";
        s += "010";
        s += "1";
        s += "011";
        s += "1";
        s += "1";
        s += "0";
        s += "0";
        s += "1";
        // pcm_enabled=0
        s += "0";
        // num_short_term_ref_pic_sets=0
        s += "1";
        // long_term=0
        s += "0";
        // temporal_mvp=1
        s += "1";
        // strong_intra_smoothing=1
        s += "1";
        // vui=0
        s += "0";
        // sps_extension_present=0
        s += "0";
        // stop bit
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.sub_layer_ordering_info_present_flag);
        assert_eq!(sps.max_sub_layers_minus1, 1);
        assert_eq!(
            sps.sub_layer_ordering_info[1].max_dec_pic_buffering_minus1,
            2
        );
        // [0] inherited from [1]
        assert_eq!(
            sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert!(sps.strong_intra_smoothing_enabled_flag);
    }

    /// SPS prefix bits identical to `synthesised_prefix_bits()` but
    /// stopping just *before* `scaling_list_enabled_flag` (i.e. after
    /// `max_transform_hierarchy_depth_intra`). Round 8 scaling-list
    /// tests append their own scaling_list block + the amp/sao bits +
    /// the SPS tail.
    fn synthesised_prefix_before_scaling_list() -> String {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting flag
        s += "00"; // profile_space
        s += "0"; // tier
        s += "00001"; // profile_idc
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
        s += "1"; // sps_id=0
        s += "010"; // chroma_format_idc=1
        s += "000010001"; // width=16
        s += "000010001"; // height=16
        s += "0"; // conf_win=0
        s += "1"; // bd_luma=0
        s += "1"; // bd_chroma=0
        s += "00101"; // log2_max_poc_lsb_minus4=4
        s += "1"; // ordering present=1
        s += "1"; // dpb=0
        s += "1"; // reorder=0
        s += "1"; // latency=0
        s += "1"; // log2_min_cb_minus3=0
        s += "010"; // log2_diff=1
        s += "1"; // log2_min_tb_minus2=0
        s += "011"; // log2_diff_tb=2
        s += "1"; // max_transform_depth_inter=0
        s += "1"; // max_transform_depth_intra=0
        s
    }

    /// SPS tail bits from `amp_enabled` through the stop bit, matching
    /// the round-4 minimal tail (sao=1, pcm=0, num_short_term_rps=0,
    /// long_term=0, temporal_mvp=1, strong_intra_smoothing=1, vui=0,
    /// extension=0, stop=1).
    fn synthesised_tail_after_scaling_list() -> String {
        let mut s = String::new();
        s += "0"; // amp_enabled=0
        s += "1"; // sao_enabled=1
        s += "0"; // pcm_enabled=0
        s += "1"; // num_short_term_ref_pic_sets ue=0
        s += "0"; // long_term_ref_pics=0
        s += "1"; // temporal_mvp=1
        s += "1"; // strong_intra_smoothing=1
        s += "0"; // vui=0
        s += "0"; // sps_extension_present=0
        s += "1"; // stop bit
        s
    }

    /// `scaling_list_enabled_flag == 1` with
    /// `sps_scaling_list_data_present_flag == 0`: per §7.4.5 the
    /// default scaling lists apply and the SPS parses end to end with
    /// no explicit [`ScalingListData`].
    #[test]
    fn scaling_list_enabled_default_lists() {
        let mut s = synthesised_prefix_before_scaling_list();
        s += "1"; // scaling_list_enabled = 1
        s += "0"; // sps_scaling_list_data_present_flag = 0
        s += &synthesised_tail_after_scaling_list();
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.scaling_list_enabled_flag);
        assert!(!sps.sps_scaling_list_data_present_flag);
        assert!(sps.scaling_list_data.is_none());
    }

    /// `scaling_list_enabled_flag == 1` with
    /// `sps_scaling_list_data_present_flag == 1`: the SPS carries an
    /// explicit `scaling_list_data()` (§7.3.4). Here every one of the
    /// 24 slots signals pred_mode=0, delta=0 (use the default list), so
    /// the parsed lists equal the §7.4.5 default tables.
    #[test]
    fn scaling_list_enabled_explicit_all_default() {
        let mut s = synthesised_prefix_before_scaling_list();
        s += "1"; // scaling_list_enabled = 1
        s += "1"; // sps_scaling_list_data_present_flag = 1
        // scaling_list_data(): 24 slots, each pred_mode=0
        // ('0') + scaling_list_pred_matrix_id_delta ue=0 ('1').
        // sizeId 0/1/2 each 6 slots; sizeId 3 only 2 slots.
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                s += "0"; // scaling_list_pred_mode_flag = 0
                s += "1"; // scaling_list_pred_matrix_id_delta ue = 0
                m += step;
            }
        }
        s += &synthesised_tail_after_scaling_list();
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.scaling_list_enabled_flag);
        assert!(sps.sps_scaling_list_data_present_flag);
        let data = sps.scaling_list_data.expect("scaling_list_data present");
        // 4x4 default = all 16.
        assert_eq!(data.lists[0][0].coef, vec![16u16; 16]);
        // 8x8 inter default for matrixId 3.
        assert_eq!(data.lists[1][3].coef[63], 91);
    }

    /// `chroma_format_idc` is u-Exp-Golomb so on-wire codeNum 4 would
    /// decode to a value of 4 — outside the legal 0..=3 range. The
    /// parser must reject it.
    #[test]
    fn rejects_chroma_format_idc_out_of_range() {
        let mut s = String::new();
        s += "0000";
        s += "000";
        s += "1";
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110";
        // sps_id=0
        s += "1";
        // chroma_format_idc ue=4 → '00101'
        s += "00101";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "chroma_format_idc",
                got: 4
            }
        );
    }

    /// Fuzz regression (r282 `parse_annexb`): an SPS whose
    /// coding-block-size pair drives `CtbLog2SizeY` (§7.4.3.2.1
    /// eqs. 7-10 / 7-11) past the Annex A profile bound of 4..=6
    /// previously survived the parse and panicked downstream in every
    /// `CtbSizeY = 1 << CtbLog2SizeY` (eq. 7-13) re-derivation.
    #[test]
    fn rejects_ctb_log2_size_above_6() {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        // log2_min_luma_coding_block_size_minus3 = 0 → MinCbLog2SizeY = 3
        s += "1";
        // log2_diff_max_min_luma_coding_block_size ue = 4 → CtbLog2SizeY = 7
        s += "00101";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "CtbLog2SizeY",
                got: 7
            }
        );
    }

    /// Annex A also bounds `CtbLog2SizeY` from below (4): a
    /// 8×8-CTB SPS must be rejected.
    #[test]
    fn rejects_ctb_log2_size_below_4() {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        // log2_min_cb_minus3 = 0, log2_diff = 0 → CtbLog2SizeY = 3
        s += "1";
        s += "1";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "CtbLog2SizeY",
                got: 3
            }
        );
    }

    /// §7.4.3.2.1: "The CVS shall not contain data that result in
    /// MinTbLog2SizeY greater than or equal to MinCbLog2SizeY".
    #[test]
    fn rejects_min_tb_log2_size_reaching_min_cb() {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        s += "1"; // MinCbLog2SizeY = 3
        s += "010"; // CtbLog2SizeY = 4
        s += "010"; // log2_min_tb_minus2 = 1 → MinTbLog2SizeY = 3 == MinCb
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "log2_min_luma_transform_block_size_minus2",
                got: 1
            }
        );
    }

    /// §7.4.3.2.1: "The CVS shall not contain data that result in
    /// MaxTbLog2SizeY greater than Min( CtbLog2SizeY, 5 )".
    #[test]
    fn rejects_max_tb_log2_size_above_cap() {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        s += "1"; // MinCbLog2SizeY = 3
        s += "010"; // CtbLog2SizeY = 4 → cap = Min(4, 5) = 4
        s += "1"; // MinTbLog2SizeY = 2
        s += "00100"; // log2_diff_tb = 3 → MaxTbLog2SizeY = 5 > 4
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "log2_diff_max_min_luma_transform_block_size",
                got: 3
            }
        );
    }

    /// §7.4.3.2.1: both hierarchy depths "shall be in the range of 0
    /// to CtbLog2SizeY − MinTbLog2SizeY, inclusive" (= 2 here).
    #[test]
    fn rejects_transform_hierarchy_depth_above_cap() {
        let mut s = synthesised_header_through_ordering("000010001", "000010001");
        s += "1"; // MinCbLog2SizeY = 3
        s += "010"; // CtbLog2SizeY = 4
        s += "1"; // MinTbLog2SizeY = 2
        s += "011"; // log2_diff_tb = 2 → MaxTbLog2SizeY = 4 (legal)
        s += "00100"; // max_transform_hierarchy_depth_inter = 3 > 2
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "max_transform_hierarchy_depth_inter",
                got: 3
            }
        );
    }

    /// §A.4.1 item b): `pic_width_in_luma_samples` ≤
    /// `Sqrt( MaxLumaPs * 8 )` = 33 776 at the largest Table A.8
    /// level. An unbounded width previously let the
    /// PicWidthInCtbsY × PicHeightInCtbsY product (eq. 7-19 territory)
    /// overflow u32 downstream.
    #[test]
    fn rejects_oversized_pic_width() {
        // ue(33 777): codeNum + 1 = 33 778 needs 16 bits → 15-zero
        // prefix followed by the 16-bit value.
        let code: u32 = 33_778;
        let len = 32 - code.leading_zeros();
        let mut ue = "0".repeat(len as usize - 1);
        for i in (0..len).rev() {
            ue.push(if (code >> i) & 1 == 1 { '1' } else { '0' });
        }
        let s = synthesised_header_through_ordering(&ue, "000010001");
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "pic_width_in_luma_samples",
                got: 33_777
            }
        );
    }

    /// Hand-assembled SPS exercising the `pcm_enabled_flag == 1`
    /// branch (§7.3.2.2 PCM block).
    #[test]
    fn parses_pcm_enabled() {
        let mut s = synthesised_prefix_bits();
        // pcm_enabled_flag = 1
        s += "1";
        // pcm_sample_bit_depth_luma_minus1 = 7 (PcmBitDepthY = 8)
        s += "0111";
        // pcm_sample_bit_depth_chroma_minus1 = 7
        s += "0111";
        // log2_min_pcm_luma_coding_block_size_minus3 = 0 → '1'
        s += "1";
        // log2_diff_max_min_pcm_luma_coding_block_size = 0 → '1'
        s += "1";
        // pcm_loop_filter_disabled_flag = 1
        s += "1";
        // num_short_term_ref_pic_sets = 0
        s += "1";
        // long_term_ref_pics_present_flag = 0
        s += "0";
        // sps_temporal_mvp_enabled_flag = 1
        s += "1";
        // strong_intra_smoothing_enabled_flag = 1
        s += "1";
        // vui_parameters_present_flag = 0
        s += "0";
        // sps_extension_present_flag = 0
        s += "0";
        // stop bit
        s += "1";
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.pcm_enabled_flag);
        let pcm = sps.pcm.expect("pcm block");
        assert_eq!(pcm.bit_depth_luma_minus1, 7);
        assert_eq!(pcm.bit_depth_chroma_minus1, 7);
        assert_eq!(pcm.log2_min_pcm_luma_coding_block_size_minus3, 0);
        assert_eq!(pcm.log2_diff_max_min_pcm_luma_coding_block_size, 0);
        assert!(pcm.loop_filter_disabled_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
    }

    /// PCM bit depth above luma bit depth must be rejected per
    /// §7.4.3.2 / equation (7-25).
    #[test]
    fn rejects_pcm_bit_depth_exceeding_luma() {
        let mut s = synthesised_prefix_bits();
        s += "1"; // pcm_enabled_flag
        // pcm_sample_bit_depth_luma_minus1 = 15 (PcmBitDepthY = 16)
        // BitDepthY in the synthesised prefix is 8.
        s += "1111";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "pcm_sample_bit_depth_luma_minus1",
                got: 15
            }
        );
    }

    /// Hand-assembled SPS with one explicit short-term RPS (no
    /// inter-RPS-prediction): 1 negative pic, 0 positive pics.
    #[test]
    fn parses_one_short_term_rps_explicit() {
        let mut s = synthesised_prefix_bits();
        // pcm_enabled_flag = 0
        s += "0";
        // num_short_term_ref_pic_sets = 1 → ue(v) codeNum 1 → '010'
        s += "010";
        // st_ref_pic_set(0): inter_ref_pic_set_prediction_flag is NOT
        // signalled (st_rps_idx == 0); implicit 0.
        //   num_negative_pics = 1 → '010'
        s += "010";
        //   num_positive_pics = 0 → '1'
        s += "1";
        //   delta_poc_s0_minus1[0] = 0 → '1'
        s += "1";
        //   used_by_curr_pic_s0_flag[0] = 1
        s += "1";
        // long_term_ref_pics_present_flag = 0
        s += "0";
        // sps_temporal_mvp_enabled_flag = 1
        s += "1";
        // strong_intra_smoothing_enabled_flag = 0
        s += "0";
        // vui_parameters_present_flag = 0
        s += "0";
        // sps_extension_present_flag = 0
        s += "0";
        s += "1"; // stop bit

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.num_short_term_ref_pic_sets, 1);
        let rps = &sps.short_term_ref_pic_sets[0];
        assert!(!rps.inter_ref_pic_set_prediction_flag);
        assert_eq!(rps.num_negative_pics, 1);
        assert_eq!(rps.num_positive_pics, 0);
        assert_eq!(rps.delta_poc_s0_minus1, vec![0]);
        assert_eq!(rps.used_by_curr_pic_s0_flag, vec![true]);
        assert!(rps.delta_poc_s1_minus1.is_empty());
        assert_eq!(rps.num_delta_pocs(), 1);
    }

    /// Hand-assembled SPS with two short-term RPSes, the second one
    /// using inter-RPS-prediction relative to the first. Exercises
    /// the `for( j = 0; j <= NumDeltaPocs[RefRpsIdx]; j++ )` loop in
    /// §7.3.7 with the `use_delta_flag` inference of §7.4.8.
    #[test]
    fn parses_inter_rps_prediction() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag
        // num_short_term_ref_pic_sets = 2 → codeNum 2 → '011'
        s += "011";
        // st_ref_pic_set(0): explicit, num_negative_pics=1, num_positive=0,
        // delta_poc_s0_minus1[0]=0, used_by_curr_pic_s0_flag[0]=1
        s += "010"; // num_neg=1
        s += "1"; // num_pos=0
        s += "1"; // dp0=0
        s += "1"; // used=1
        // st_ref_pic_set(1): inter_ref_pic_set_prediction_flag = 1
        s += "1";
        //   delta_idx_minus1 is NOT signalled (st_rps_idx=1, num=2; only
        //   signalled when st_rps_idx == num). Inferred to 0 →
        //   RefRpsIdx = 1 - (0+1) = 0.
        //   delta_rps_sign = 0
        s += "0";
        //   abs_delta_rps_minus1 = 0 → codeNum 0 → '1'
        s += "1";
        //   NumDeltaPocs[0] = 1, so loop runs j=0..1 (2 entries):
        //     j=0: used_by_curr_pic_flag = 1  → use_delta_flag inferred 1
        s += "1";
        //     j=1: used_by_curr_pic_flag = 0, use_delta_flag = 1
        s += "0";
        s += "1";
        // long_term=0, temporal_mvp=1, strong_intra_smoothing=1, vui=0, ext=0, stop
        s += "0";
        s += "1";
        s += "1";
        s += "0";
        s += "0";
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.num_short_term_ref_pic_sets, 2);
        let r1 = &sps.short_term_ref_pic_sets[1];
        assert!(r1.inter_ref_pic_set_prediction_flag);
        assert_eq!(r1.delta_idx_minus1, 0);
        assert!(!r1.delta_rps_sign);
        assert_eq!(r1.abs_delta_rps_minus1, 0);
        assert_eq!(r1.used_by_curr_pic_flag, vec![true, false]);
        assert_eq!(r1.use_delta_flag, vec![true, true]);
    }

    /// Hand-assembled SPS exercising the long-term-ref-pic block.
    #[test]
    fn parses_long_term_ref_pics() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag = 0
        s += "1"; // num_short_term_ref_pic_sets = 0
        s += "1"; // long_term_ref_pics_present_flag = 1
        // num_long_term_ref_pics_sps = 2 → codeNum 2 → '011'
        s += "011";
        // log2_max_pic_order_cnt_lsb_minus4 = 4 in the synthesised
        // prefix, so lt_ref_pic_poc_lsb_sps[i] is 8 bits wide.
        //   i=0: poc_lsb = 0x10, used_by_curr_pic_lt_sps_flag = 1
        s += "00010000";
        s += "1";
        //   i=1: poc_lsb = 0x20, used = 0
        s += "00100000";
        s += "0";
        // temporal_mvp=1, strong_intra=0, vui=0, ext=0, stop
        s += "1";
        s += "0";
        s += "0";
        s += "0";
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.long_term_ref_pics_present_flag);
        assert_eq!(sps.num_long_term_ref_pics_sps, 2);
        assert_eq!(sps.long_term_ref_pics.len(), 2);
        assert_eq!(sps.long_term_ref_pics[0].poc_lsb, 0x10);
        assert!(sps.long_term_ref_pics[0].used_by_curr_pic);
        assert_eq!(sps.long_term_ref_pics[1].poc_lsb, 0x20);
        assert!(!sps.long_term_ref_pics[1].used_by_curr_pic);
    }

    /// SPS with `vui_parameters_present_flag == 1`: the parser now
    /// decodes the §E.2.1 `vui_parameters()` body in full and
    /// continues to `sps_extension_present_flag`. Here the VUI is the
    /// minimal all-flags-off body (ten `u(1)` flags), so no opaque
    /// tail is captured.
    #[test]
    fn decodes_vui_then_continues_to_extension_flag() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag = 0
        s += "1"; // num_short_term_ref_pic_sets = 0
        s += "0"; // long_term = 0
        s += "1"; // temporal_mvp = 1
        s += "1"; // strong_intra_smoothing = 1
        s += "1"; // vui_parameters_present_flag = 1
        // vui_parameters(): ten flags all 0 (minimal body)
        s += "0000000000";
        s += "0"; // sps_extension_present_flag = 0
        s += "1"; // rbsp stop bit

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.vui_parameters_present_flag);
        let vui = sps.vui_parameters.as_ref().expect("VUI body");
        assert!(!vui.aspect_ratio_info_present_flag);
        assert!(!vui.overscan_info_present_flag);
        assert!(!vui.video_signal_type_present_flag);
        assert!(!vui.chroma_loc_info_present_flag);
        assert!(!vui.vui_timing_info_present_flag);
        assert!(!vui.bitstream_restriction_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// SPS with `vui_parameters_present_flag == 1` whose VUI signals a
    /// timing-info block, followed by `sps_extension_present_flag == 1`
    /// — the extension body after the (now fully decoded) VUI is
    /// surfaced as an opaque tail.
    #[test]
    fn decodes_vui_then_captures_extension_tail() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag = 0
        s += "1"; // num_short_term_ref_pic_sets = 0
        s += "0"; // long_term = 0
        s += "1"; // temporal_mvp = 1
        s += "1"; // strong_intra_smoothing = 1
        s += "1"; // vui_parameters_present_flag = 1
        // vui_parameters():
        s += "0"; // aspect_ratio_info_present_flag = 0
        s += "0"; // overscan_info_present_flag = 0
        s += "0"; // video_signal_type_present_flag = 0
        s += "0"; // chroma_loc_info_present_flag = 0
        s += "0"; // neutral_chroma_indication_flag = 0
        s += "0"; // field_seq_flag = 0
        s += "0"; // frame_field_info_present_flag = 0
        s += "0"; // default_display_window_flag = 0
        s += "1"; // vui_timing_info_present_flag = 1
        s += "00000000000000000000000000000001"; // vui_num_units_in_tick = 1
        s += "00000000000000000000000000011001"; // vui_time_scale = 25
        s += "0"; // vui_poc_proportional_to_timing_flag = 0
        s += "0"; // vui_hrd_parameters_present_flag = 0
        s += "0"; // bitstream_restriction_flag = 0
        // sps_extension_present_flag = 1, followed by the
        // typed extension flag block. Set the range-extension
        // flag so the nine `sps_range_extension()` flags are
        // decoded; with no further body, no opaque tail follows.
        s += "1"; // sps_extension_present_flag
        s += "1"; // sps_range_extension_flag = 1
        s += "000"; // sps_multilayer / sps_3d / sps_scc = 0
        s += "0000"; // sps_extension_4bits = 0
        s += "101010101"; // sps_range_extension() nine flags
        s += "1"; // rbsp_trailing_bits stop bit

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let vui = sps.vui_parameters.as_ref().expect("VUI body");
        let ti = vui.timing_info.as_ref().expect("timing info");
        assert_eq!(ti.num_units_in_tick, 1);
        assert_eq!(ti.time_scale, 25);
        assert!(sps.sps_extension_present_flag);
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(flags.sps_range_extension_flag);
        assert!(!flags.sps_multilayer_extension_flag);
        assert!(!flags.sps_3d_extension_flag);
        assert!(!flags.sps_scc_extension_flag);
        assert_eq!(flags.sps_extension_4bits, 0);
        assert!(flags.has_body());
        // The range-extension body is now decoded, not opaque: with no
        // multilayer/3d/scc/4bits body after it, no opaque tail remains.
        assert!(sps.opaque_tail.is_none());
        let re = sps.sps_range_extension.expect("range extension body");
        assert!(re.transform_skip_rotation_enabled_flag); // 1
        assert!(!re.transform_skip_context_enabled_flag); // 0
        assert!(re.implicit_rdpcm_enabled_flag); // 1
        assert!(!re.explicit_rdpcm_enabled_flag); // 0
        assert!(re.extended_precision_processing_flag); // 1
        assert!(!re.intra_smoothing_disabled_flag); // 0
        assert!(re.high_precision_offsets_enabled_flag); // 1
        assert!(!re.persistent_rice_adaptation_enabled_flag); // 0
        assert!(re.cabac_bypass_alignment_enabled_flag); // 1
    }

    /// `sps_extension_present_flag == 1` with `sps_range_extension_flag
    /// == 1` decodes the typed flag block then the nine
    /// `sps_range_extension()` flags (§7.3.2.2.2) in bit-stream order.
    /// This is the RExt-profile entry point (§A.3.5).
    #[test]
    fn decodes_sps_range_extension_body() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "1"; // sps_range_extension_flag = 1
        s += "000"; // sps_multilayer / sps_3d / sps_scc = 0
        s += "0000"; // sps_extension_4bits = 0
        // sps_range_extension() nine `u(1)` flags, all set —
        // every RExt tool enabled.
        s += "111111111";
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.vui_parameters_present_flag);
        assert!(sps.sps_extension_present_flag);
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(flags.sps_range_extension_flag);
        assert_eq!(flags.sps_extension_4bits, 0);
        // No body after the range extension → no opaque tail.
        assert!(sps.opaque_tail.is_none());
        let re = sps.sps_range_extension.expect("range extension body");
        assert!(re.transform_skip_rotation_enabled_flag);
        assert!(re.transform_skip_context_enabled_flag);
        assert!(re.implicit_rdpcm_enabled_flag);
        assert!(re.explicit_rdpcm_enabled_flag);
        assert!(re.extended_precision_processing_flag);
        assert!(re.intra_smoothing_disabled_flag);
        assert!(re.high_precision_offsets_enabled_flag);
        assert!(re.persistent_rice_adaptation_enabled_flag);
        assert!(re.cabac_bypass_alignment_enabled_flag);
    }

    /// `sps_range_extension_flag == 1` followed by `sps_scc_extension_flag
    /// == 1` (no multilayer/3D body between them): the nine
    /// range-extension flags AND the `sps_scc_extension()` body
    /// (§7.3.2.2.3) are both decoded in place — no opaque tail remains.
    /// Palette mode disabled, so only the three leading/trailing fields
    /// are present.
    #[test]
    fn decodes_range_extension_then_scc_body_no_palette() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "1"; // sps_range_extension_flag = 1
        s += "00"; // sps_multilayer / sps_3d = 0
        s += "1"; // sps_scc_extension_flag = 1
        s += "0000"; // sps_extension_4bits = 0
        s += "000000000"; // sps_range_extension() nine flags, all 0
        // sps_scc_extension():
        s += "1"; // sps_curr_pic_ref_enabled_flag = 1
        s += "0"; // palette_mode_enabled_flag = 0 (palette block absent)
        s += "10"; // motion_vector_resolution_control_idc = 2 (u(2))
        s += "1"; // intra_boundary_filtering_disabled_flag = 1
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(flags.sps_range_extension_flag);
        assert!(flags.sps_scc_extension_flag);
        let re = sps.sps_range_extension.expect("range extension body");
        assert_eq!(re, SpsRangeExtension::default());
        // Both bodies are decoded in place; nothing is left opaque.
        assert!(sps.opaque_tail.is_none());
        let scc = sps.sps_scc_extension.expect("scc extension body");
        assert!(scc.sps_curr_pic_ref_enabled_flag);
        assert!(!scc.palette_mode_enabled_flag);
        assert_eq!(scc.palette_max_size, 0);
        assert_eq!(scc.delta_palette_max_predictor_size, 0);
        assert!(!scc.sps_palette_predictor_initializers_present_flag);
        assert!(scc.sps_palette_predictor_initializer.is_empty());
        assert_eq!(scc.motion_vector_resolution_control_idc, 2);
        assert!(scc.intra_boundary_filtering_disabled_flag);
    }

    /// `sps_scc_extension()` with `palette_mode_enabled_flag == 1` and
    /// palette predictor initializers present: the `palette_max_size`,
    /// `delta_palette_max_predictor_size`, and per-component initializer
    /// table (§7.3.2.2.3) are decoded, each `u(v)` sized by `BitDepthY` /
    /// `BitDepthC`. `chroma_format_idc == 1` here gives `numComps == 3`.
    #[test]
    fn decodes_scc_extension_palette_initializers() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0001"; // range/multilayer/3d = 0, scc = 1
        s += "0000"; // sps_extension_4bits = 0
        // sps_scc_extension():
        s += "0"; // sps_curr_pic_ref_enabled_flag = 0
        s += "1"; // palette_mode_enabled_flag = 1
        s += "00100"; // palette_max_size = ue(3) → "00100"
        s += "010"; // delta_palette_max_predictor_size = ue(1) → "010"
        s += "1"; // sps_palette_predictor_initializers_present_flag = 1
        s += "010"; // sps_num_palette_predictor_initializers_minus1 = ue(1) → 1
        // numComps = 3, num_entries = 2; default BitDepthY/C = 8.
        // comp 0 (8-bit): 0x01, 0x02
        s += "00000001";
        s += "00000010";
        // comp 1 (8-bit): 0x03, 0x04
        s += "00000011";
        s += "00000100";
        // comp 2 (8-bit): 0x05, 0x06
        s += "00000101";
        s += "00000110";
        s += "10"; // motion_vector_resolution_control_idc = 2 (u(2))
        s += "0"; // intra_boundary_filtering_disabled_flag = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.opaque_tail.is_none());
        let scc = sps.sps_scc_extension.expect("scc extension body");
        assert!(!scc.sps_curr_pic_ref_enabled_flag);
        assert!(scc.palette_mode_enabled_flag);
        assert_eq!(scc.palette_max_size, 3);
        assert_eq!(scc.delta_palette_max_predictor_size, 1);
        // PaletteMaxPredictorSize = 3 + 1 (eq. 7-35).
        assert_eq!(scc.palette_max_predictor_size(), 4);
        assert!(scc.sps_palette_predictor_initializers_present_flag);
        assert_eq!(scc.sps_num_palette_predictor_initializers_minus1, 1);
        assert_eq!(
            scc.sps_palette_predictor_initializer,
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
        );
        assert_eq!(scc.motion_vector_resolution_control_idc, 2);
        assert!(!scc.intra_boundary_filtering_disabled_flag);
    }

    /// §7.4.3.2.3 / §A.3.7: `sps_num_palette_predictor_initializers_minus1`
    /// past the largest representable predictor (PaletteMaxPredictorSize
    /// <= 128) is rejected BEFORE any initializer allocation — a fuzzed
    /// count once drove a multi-GiB `Vec::with_capacity`.
    #[test]
    fn rejects_oversized_palette_predictor_initializer_count() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0001"; // range/multilayer/3d = 0, scc = 1
        s += "0000"; // sps_extension_4bits = 0
        // sps_scc_extension():
        s += "0"; // sps_curr_pic_ref_enabled_flag = 0
        s += "1"; // palette_mode_enabled_flag = 1
        s += "00100"; // palette_max_size = 3
        s += "010"; // delta_palette_max_predictor_size = 1
        s += "1"; // sps_palette_predictor_initializers_present_flag = 1
        // sps_num_palette_predictor_initializers_minus1 = ue(128):
        // 129 in 8 bits = 10000001, prefix of 7 zeros.
        s += "0000000";
        s += "10000001";
        s += "1"; // (whatever follows is unreached)
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).expect_err("oversized initializer count");
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "sps_num_palette_predictor_initializers_minus1",
                got: 128
            }
        ));
    }

    /// §7.4.3.2.3: `motion_vector_resolution_control_idc == 3` is
    /// reserved and rejected as out-of-range.
    #[test]
    fn rejects_reserved_motion_vector_resolution_control_idc() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0001"; // range/multilayer/3d = 0, scc = 1
        s += "0000"; // sps_extension_4bits = 0
        // sps_scc_extension():
        s += "0"; // sps_curr_pic_ref_enabled_flag = 0
        s += "0"; // palette_mode_enabled_flag = 0
        s += "11"; // motion_vector_resolution_control_idc = 3 (reserved)
        s += "0"; // intra_boundary_filtering_disabled_flag = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).expect_err("reserved mvr idc");
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "motion_vector_resolution_control_idc",
                got: 3
            }
        ));
    }

    /// §7.4.3.2.3: when `palette_max_size == 0`, a non-zero
    /// `delta_palette_max_predictor_size` violates bitstream
    /// conformance and is rejected.
    #[test]
    fn rejects_delta_palette_when_max_size_zero() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0001"; // range/multilayer/3d = 0, scc = 1
        s += "0000"; // sps_extension_4bits = 0
        // sps_scc_extension():
        s += "0"; // sps_curr_pic_ref_enabled_flag = 0
        s += "1"; // palette_mode_enabled_flag = 1
        s += "1"; // palette_max_size = 0 (ue)
        s += "010"; // delta_palette_max_predictor_size = 1 (ue) → illegal
        s += "0"; // sps_palette_predictor_initializers_present_flag = 0
        s += "00"; // motion_vector_resolution_control_idc = 0
        s += "0"; // intra_boundary_filtering_disabled_flag = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).expect_err("delta vs max_size 0");
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "delta_palette_max_predictor_size",
                got: 1
            }
        ));
    }

    /// `sps_extension_present_flag == 1` with every typed extension
    /// flag (and `sps_extension_4bits`) equal to 0 decodes the
    /// flag block but consumes only `rbsp_trailing_bits()` afterwards
    /// — no opaque tail is surfaced because no extension body follows.
    #[test]
    fn decodes_extension_flag_block_without_bodies() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0000"; // four typed flags all 0
        s += "0000"; // sps_extension_4bits = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.sps_extension_present_flag);
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(!flags.sps_range_extension_flag);
        assert!(!flags.sps_multilayer_extension_flag);
        assert!(!flags.sps_3d_extension_flag);
        assert!(!flags.sps_scc_extension_flag);
        assert_eq!(flags.sps_extension_4bits, 0);
        assert!(!flags.has_body());
        assert!(sps.opaque_tail.is_none());
    }

    /// `sps_extension_present_flag == 1` with `sps_scc_extension_flag
    /// == 1` (and no preceding range/multilayer/3D body) selects the
    /// §A.3.7 Screen Content Coding profile family; the typed block
    /// decodes cleanly and the `sps_scc_extension()` body (§7.3.2.2.3)
    /// is decoded in place with no opaque tail. Palette mode disabled.
    #[test]
    fn decodes_scc_extension_body_no_range() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0"; // sps_range_extension_flag = 0
        s += "0"; // sps_multilayer_extension_flag = 0
        s += "0"; // sps_3d_extension_flag = 0
        s += "1"; // sps_scc_extension_flag = 1
        s += "0000"; // sps_extension_4bits = 0
        // sps_scc_extension():
        s += "1"; // sps_curr_pic_ref_enabled_flag = 1
        s += "0"; // palette_mode_enabled_flag = 0
        s += "01"; // motion_vector_resolution_control_idc = 1 (u(2))
        s += "0"; // intra_boundary_filtering_disabled_flag = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(!flags.sps_range_extension_flag);
        assert!(!flags.sps_multilayer_extension_flag);
        assert!(!flags.sps_3d_extension_flag);
        assert!(flags.sps_scc_extension_flag);
        assert_eq!(flags.sps_extension_4bits, 0);
        assert!(flags.has_body());
        assert!(sps.opaque_tail.is_none());
        let scc = sps.sps_scc_extension.expect("scc extension body");
        assert!(scc.sps_curr_pic_ref_enabled_flag);
        assert!(!scc.palette_mode_enabled_flag);
        assert_eq!(scc.motion_vector_resolution_control_idc, 1);
        assert!(!scc.intra_boundary_filtering_disabled_flag);
    }

    /// When `sps_scc_extension_flag == 1` but a `sps_multilayer_extension()`
    /// body precedes it (§7.3.2.2.1 body order), the SCC body cannot be
    /// decoded in place and the whole multilayer-onward span — including
    /// the SCC body — stays in the opaque tail.
    #[test]
    fn scc_stays_opaque_behind_multilayer_body() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0"; // sps_range_extension_flag = 0
        s += "1"; // sps_multilayer_extension_flag = 1
        s += "0"; // sps_3d_extension_flag = 0
        s += "1"; // sps_scc_extension_flag = 1
        s += "0000"; // sps_extension_4bits = 0
        s += "11001100"; // opaque multilayer + scc span sentinel
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let flags = sps.extension_flags.expect("extension flag block");
        assert!(flags.sps_multilayer_extension_flag);
        assert!(flags.sps_scc_extension_flag);
        // Multilayer body is still opaque, so the SCC body cannot be
        // decoded in place; both stay in the captured tail.
        assert!(sps.sps_scc_extension.is_none());
        assert!(sps.opaque_tail.is_some());
    }

    /// `sps_extension_present_flag == 1` with the four typed
    /// extension flags all 0 but `sps_extension_4bits != 0` still
    /// surfaces an opaque tail — the §7.3.2.2.1
    /// `while( more_rbsp_data() ) sps_extension_data_flag` block is
    /// gated by `sps_extension_4bits` (the §7.4.3.2.1 decoder rule is
    /// to ignore the data flags but they must be skipped past
    /// rbsp_trailing_bits, so the bytes are surfaced as opaque).
    #[test]
    fn captures_extension_data_flag_tail_when_4bits_nonzero() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
        s += "0000"; // four typed flags = 0
        s += "0001"; // sps_extension_4bits = 1 (reserved, non-zero)
        s += "0"; // a single sps_extension_data_flag value
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let flags = sps.extension_flags.expect("extension flag block");
        assert_eq!(flags.sps_extension_4bits, 1);
        assert!(flags.has_body());
        assert!(sps.opaque_tail.is_some());
    }

    /// When `sps_extension_present_flag == 0` the typed
    /// extension-flag block is absent and every flag is inferred to
    /// 0 per §7.4.3.2.1.
    #[test]
    fn extension_flags_absent_when_gate_zero() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "0"; // sps_extension_present_flag = 0
        s += "1"; // rbsp_trailing_bits stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.extension_flags.is_none());
        assert!(sps.opaque_tail.is_none());
    }

    /// SPS with both `vui` and `extension` flags off — no opaque
    /// tail is captured (only the rbsp_trailing_bits remain).
    #[test]
    fn no_opaque_tail_when_flags_clear() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "0"; // sps_extension_present=0
        s += "1"; // stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.vui_parameters_present_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// `num_short_term_ref_pic_sets > 64` is illegal per §7.4.3.2.
    #[test]
    fn rejects_too_many_short_term_rps() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        // num_short_term_ref_pic_sets = 65 → codeNum 65
        // 65 in 0-th order Exp-Golomb: leadingZeroBits=6,
        //   suffix = 65 - (2^6 - 1) = 2 = '000010'
        // so the bit string is '000000 1 000010' = 13 bits.
        s += "0000001000010";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "num_short_term_ref_pic_sets",
                ..
            }
        ));
    }

    /// §7.4.8 explicit-form materialisation: the cumulative recurrence
    /// for `DeltaPocS0[i]` (equation 7-69) and `DeltaPocS1[i]`
    /// (equation 7-70) starting from the equation-7-67 / 7-68 seeds.
    #[test]
    fn materialize_explicit_form_recurrence() {
        // Hand-assembled RPS: num_negative_pics = 3,
        // delta_poc_s0_minus1 = [0, 1, 0], used_by_curr_pic_s0_flag =
        // [true, false, true]; num_positive_pics = 2,
        // delta_poc_s1_minus1 = [1, 0], used_by_curr_pic_s1_flag =
        // [true, true]. Per §7.4.8:
        //   DeltaPocS0 = [-1, -3, -4]
        //   DeltaPocS1 = [ 2,  3]
        let rps = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            delta_idx_minus1: 0,
            delta_rps_sign: false,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: Vec::new(),
            use_delta_flag: Vec::new(),
            num_negative_pics: 3,
            num_positive_pics: 2,
            delta_poc_s0_minus1: vec![0, 1, 0],
            used_by_curr_pic_s0_flag: vec![true, false, true],
            delta_poc_s1_minus1: vec![1, 0],
            used_by_curr_pic_s1_flag: vec![true, true],
        };
        let m = rps.materialize(None).expect("explicit materialise");
        assert_eq!(m.delta_poc_s0, vec![-1, -3, -4]);
        assert_eq!(m.used_by_curr_pic_s0, vec![true, false, true]);
        assert_eq!(m.delta_poc_s1, vec![2, 3]);
        assert_eq!(m.used_by_curr_pic_s1, vec![true, true]);
        assert_eq!(m.num_negative_pics(), 3);
        assert_eq!(m.num_positive_pics(), 2);
        assert_eq!(m.num_delta_pocs(), 5);
    }

    /// §7.4.8 inter-RPS-prediction (equations 7-61 / 7-62): a tiny
    /// chain where the source has one negative POC at -1 and the
    /// derived RPS uses `deltaRps = +1` to shift it past zero, so the
    /// source's negative drops out and `deltaRps` itself lands as a
    /// positive entry gated by the wire flags. Matches the
    /// `parses_inter_rps_prediction` fixture above.
    #[test]
    fn materialize_inter_rps_prediction_matches_fixture() {
        let src = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        // The inter-form RPS from the existing
        // `parses_inter_rps_prediction` test: deltaRps = +1,
        // used_by_curr_pic_flag = [true, false], use_delta_flag = [true,
        // true].
        let inter = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1: 0,
            delta_rps_sign: false,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: vec![true, false],
            use_delta_flag: vec![true, true],
            num_negative_pics: 0,
            num_positive_pics: 0,
            delta_poc_s0_minus1: Vec::new(),
            used_by_curr_pic_s0_flag: Vec::new(),
            delta_poc_s1_minus1: Vec::new(),
            used_by_curr_pic_s1_flag: Vec::new(),
        };
        let m = inter
            .materialize(Some(&src))
            .expect("inter-RPS materialise");
        // Negative side (equation 7-61):
        //   * No source positives.
        //   * deltaRps = +1 ≥ 0, skip the self-term.
        //   * Source negative j=0: dPoc = -1 + 1 = 0, not < 0, skip.
        //   ⇒ NumNegativePics = 0.
        assert!(m.delta_poc_s0.is_empty());
        assert!(m.used_by_curr_pic_s0.is_empty());
        // Positive side (equation 7-62):
        //   * Source negative j=0 reverse: dPoc = -1 + 1 = 0, not > 0,
        //     skip.
        //   * deltaRps = +1 > 0 and use_delta_flag[NumDeltaPocs=1] =
        //     true ⇒ DeltaPocS1[0] = +1, UsedByCurrPicS1[0] =
        //     used_by_curr_pic_flag[1] = false.
        //   * No source positives.
        //   ⇒ NumPositivePics = 1.
        assert_eq!(m.delta_poc_s1, vec![1]);
        assert_eq!(m.used_by_curr_pic_s1, vec![false]);
    }

    /// §7.4.8 inter-RPS-prediction with `deltaRps < 0`: a source
    /// positive POC at +1 with `deltaRps = -2` falls onto the negative
    /// side via the source-positives-reverse step of equation 7-61.
    #[test]
    fn materialize_inter_rps_prediction_negative_delta_rps() {
        // Source: one positive POC at +1 (no negatives).
        let src = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![],
            used_by_curr_pic_s0: vec![],
            delta_poc_s1: vec![1],
            used_by_curr_pic_s1: vec![true],
        };
        // Inter-form: deltaRps = -(1+1) = -2; arrays sized
        // NumDeltaPocs[src]+1 = 2.
        // used_by_curr_pic_flag indexed:
        //   * [0..NumNegativePics[src]) = none
        //   * [NumNegativePics, NumNegativePics + NumPositivePics) = [0..1) = j_src=0
        //   * [NumDeltaPocs] = trailing slot for `deltaRps` self-term
        // So used_by_curr_pic_flag = [true_for_src_pos0, true_for_self_term].
        let inter = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1: 0,
            delta_rps_sign: true, // negative
            abs_delta_rps_minus1: 1,
            used_by_curr_pic_flag: vec![true, true],
            use_delta_flag: vec![true, true],
            num_negative_pics: 0,
            num_positive_pics: 0,
            delta_poc_s0_minus1: Vec::new(),
            used_by_curr_pic_s0_flag: Vec::new(),
            delta_poc_s1_minus1: Vec::new(),
            used_by_curr_pic_s1_flag: Vec::new(),
        };
        let m = inter
            .materialize(Some(&src))
            .expect("inter-RPS negative deltaRps");
        // Negative side (equation 7-61):
        //   * Source positive j=0 reverse: dPoc = 1 + (-2) = -1 < 0 and
        //     use_delta_flag[NumNeg=0 + 0] = use_delta_flag[0] = true
        //     ⇒ DeltaPocS0[0] = -1, UsedByCurrPicS0[0] =
        //       used_by_curr_pic_flag[0] = true.
        //   * deltaRps = -2 < 0 and use_delta_flag[NumDeltaPocs=1] =
        //     true ⇒ DeltaPocS0[1] = -2, UsedByCurrPicS0[1] =
        //       used_by_curr_pic_flag[1] = true.
        //   * No source negatives.
        assert_eq!(m.delta_poc_s0, vec![-1, -2]);
        assert_eq!(m.used_by_curr_pic_s0, vec![true, true]);
        // Positive side: source negatives reverse: none. deltaRps < 0,
        // skip self-term. Source positive j=0: dPoc = +1 + (-2) = -1,
        // not > 0, skip. ⇒ empty.
        assert!(m.delta_poc_s1.is_empty());
        assert!(m.used_by_curr_pic_s1.is_empty());
    }

    /// [`ShortTermRefPicSet::materialize`] rejects the inter form
    /// without a source RPS.
    #[test]
    fn materialize_inter_rps_rejects_missing_source() {
        let inter = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1: 0,
            delta_rps_sign: false,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: vec![true],
            use_delta_flag: vec![true],
            ..Default::default()
        };
        assert_eq!(
            inter.materialize(None),
            Err(ShortTermRefPicSetMaterializeError::MissingSource),
        );
    }

    /// [`ShortTermRefPicSet::materialize`] surfaces a per-position
    /// array-length mismatch against the source RPS's `NumDeltaPocs +
    /// 1`.
    #[test]
    fn materialize_inter_rps_rejects_length_mismatch() {
        let src = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1, -2],
            used_by_curr_pic_s0: vec![true, true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        // NumDeltaPocs[src] = 2, so the inter form expects arrays of
        // length 3. Pass 2-element arrays to provoke the mismatch.
        let inter = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1: 0,
            delta_rps_sign: false,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: vec![true, true],
            use_delta_flag: vec![true, true],
            ..Default::default()
        };
        assert!(matches!(
            inter.materialize(Some(&src)),
            Err(ShortTermRefPicSetMaterializeError::SourceLengthMismatch {
                expected: 3,
                got_used: 2,
                got_delta: 2,
            }),
        ));
    }

    /// [`SeqParameterSet::materialize_short_term_ref_pic_sets`]
    /// chains explicit and inter-RPS-predicted entries through their
    /// `RefRpsIdx` lookups using the same fixture as
    /// `parses_inter_rps_prediction`.
    #[test]
    fn sps_materialize_chains_inter_rps_prediction() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag
        s += "011"; // num_short_term_ref_pic_sets = 2
        // st_ref_pic_set(0): explicit num_neg=1, num_pos=0,
        // delta_poc_s0_minus1[0]=0, used_by_curr_pic_s0_flag=1
        s += "010";
        s += "1";
        s += "1";
        s += "1";
        // st_ref_pic_set(1): inter_ref_pic_set_prediction_flag = 1,
        // delta_rps_sign=0, abs_delta_rps_minus1=0 (deltaRps=+1),
        // used_by_curr_pic_flag = [1, 0], use_delta_flag[1] = 1.
        s += "1";
        s += "0";
        s += "1";
        s += "1"; // used_by_curr_pic_flag[0] = 1 (use_delta inferred)
        s += "0"; // used_by_curr_pic_flag[1] = 0
        s += "1"; // use_delta_flag[1] = 1
        // long_term=0, temporal_mvp=1, strong_intra_smoothing=1,
        // vui=0, ext=0, stop bit
        s += "0";
        s += "1";
        s += "1";
        s += "0";
        s += "0";
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        let m = sps
            .materialize_short_term_ref_pic_sets()
            .expect("materialize");
        assert_eq!(m.len(), 2);
        // Explicit source (idx 0): DeltaPocS0=[-1], UsedByCurrPicS0=[true].
        assert_eq!(m[0].delta_poc_s0, vec![-1]);
        assert_eq!(m[0].used_by_curr_pic_s0, vec![true]);
        assert_eq!(m[0].num_delta_pocs(), 1);
        // Inter-predicted (idx 1): NumNegativePics=0,
        // NumPositivePics=1 with DeltaPocS1=[+1] from the deltaRps
        // self-term (see `materialize_inter_rps_prediction_matches_fixture`).
        assert!(m[1].delta_poc_s0.is_empty());
        assert_eq!(m[1].delta_poc_s1, vec![1]);
        assert_eq!(m[1].used_by_curr_pic_s1, vec![false]);
    }

    /// `num_long_term_ref_pics_sps > 32` is illegal per §7.4.3.2.
    #[test]
    fn rejects_too_many_long_term_rps() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "1"; // long_term_ref_pics_present
        // num_long_term_ref_pics_sps = 33: codeNum 33
        // leadingZeroBits=5, suffix = 33 - (2^5 - 1) = 2 = '00010'
        // bit string: '00000 1 00010' = 11 bits.
        s += "00000100010";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "num_long_term_ref_pics_sps",
                ..
            }
        ));
    }
}
