//! Picture Parameter Set (PPS) parser per ITU-T Rec. H.265 §7.3.2.3.1.
//!
//! Round-5 scope: parse the general `pic_parameter_set_rbsp()` body
//! through `pps_extension_present_flag`, materialising every field of
//! the §7.3.2.3.1 syntax table including the tiles block
//! (`num_tile_columns_minus1` / `num_tile_rows_minus1` /
//! `uniform_spacing_flag` and, when `uniform_spacing_flag == 0`, the
//! `column_width_minus1[]` / `row_height_minus1[]` arrays) and the
//! deblocking-filter-control block. The §7.4.3.3.1 inference rules for
//! the conditionally-signalled fields are applied so the parsed struct
//! always carries the effective value.
//!
//! One body is parsed via the shared `scaling_list` module; one is
//! surfaced rather than decoded:
//!
//! * `pps_scaling_list_data_present_flag == 1` triggers a §7.3.4
//!   `scaling_list_data()` parse into [`ScalingListData`] (shared with
//!   the SPS path).
//! * When `pps_extension_present_flag == 1` the eight bits of typed
//!   extension flags (`pps_range_extension_flag`,
//!   `pps_multilayer_extension_flag`, `pps_3d_extension_flag`,
//!   `pps_scc_extension_flag`, `pps_extension_4bits`) are decoded into
//!   [`PpsExtensionFlags`]. When all five sub-fields are zero only the
//!   `rbsp_trailing_bits()` remain and they are consumed implicitly;
//!   otherwise the extension bodies (`pps_range_extension()`,
//!   `pps_multilayer_extension()`, `pps_3d_extension()`,
//!   `pps_scc_extension()`, and the
//!   `pps_extension_data_flag` while-loop gated by
//!   `pps_extension_4bits`) are surfaced as a single [`OpaqueTail`]
//!   starting at the bit position of the first body. The §7.4.3.3.1
//!   inference rules apply for the absent case (every flag inferred
//!   to 0).
//!
//! ## Layout summary
//!
//! ```text
//! pps_pic_parameter_set_id                          ue(v)
//! pps_seq_parameter_set_id                          ue(v)
//! dependent_slice_segments_enabled_flag              u(1)
//! output_flag_present_flag                           u(1)
//! num_extra_slice_header_bits                        u(3)
//! sign_data_hiding_enabled_flag                      u(1)
//! cabac_init_present_flag                            u(1)
//! num_ref_idx_l0_default_active_minus1              ue(v)
//! num_ref_idx_l1_default_active_minus1              ue(v)
//! init_qp_minus26                                   se(v)
//! constrained_intra_pred_flag                        u(1)
//! transform_skip_enabled_flag                        u(1)
//! cu_qp_delta_enabled_flag                           u(1)
//! if( cu_qp_delta_enabled_flag )
//!   diff_cu_qp_delta_depth                          ue(v)
//! pps_cb_qp_offset                                  se(v)
//! pps_cr_qp_offset                                  se(v)
//! pps_slice_chroma_qp_offsets_present_flag           u(1)
//! weighted_pred_flag                                 u(1)
//! weighted_bipred_flag                               u(1)
//! transquant_bypass_enabled_flag                     u(1)
//! tiles_enabled_flag                                 u(1)
//! entropy_coding_sync_enabled_flag                   u(1)
//! if( tiles_enabled_flag ) {
//!   num_tile_columns_minus1                         ue(v)
//!   num_tile_rows_minus1                            ue(v)
//!   uniform_spacing_flag                             u(1)
//!   if( !uniform_spacing_flag ) {
//!     for( i = 0; i < num_tile_columns_minus1; i++ )
//!       column_width_minus1[i]                      ue(v)
//!     for( i = 0; i < num_tile_rows_minus1; i++ )
//!       row_height_minus1[i]                        ue(v)
//!   }
//!   loop_filter_across_tiles_enabled_flag            u(1)
//! }
//! pps_loop_filter_across_slices_enabled_flag         u(1)
//! deblocking_filter_control_present_flag             u(1)
//! if( deblocking_filter_control_present_flag ) {
//!   deblocking_filter_override_enabled_flag          u(1)
//!   pps_deblocking_filter_disabled_flag             u(1)
//!   if( !pps_deblocking_filter_disabled_flag ) {
//!     pps_beta_offset_div2                          se(v)
//!     pps_tc_offset_div2                            se(v)
//!   }
//! }
//! pps_scaling_list_data_present_flag                 u(1)
//!   if( pps_scaling_list_data_present_flag )
//!     scaling_list_data( )                           /* §7.3.4 */
//! lists_modification_present_flag                    u(1)
//! log2_parallel_merge_level_minus2                  ue(v)
//! slice_segment_header_extension_present_flag        u(1)
//! pps_extension_present_flag                         u(1)
//! if( pps_extension_present_flag ) {
//!   pps_range_extension_flag                          u(1)
//!   pps_multilayer_extension_flag                     u(1)
//!   pps_3d_extension_flag                             u(1)
//!   pps_scc_extension_flag                            u(1)
//!   pps_extension_4bits                               u(4)
//! }
//!   /* opaque tail begins here when any of the four extension
//!      flags is 1, or when pps_extension_4bits != 0 */
//! ```
//!
//! Validity checks performed here, sourced from §7.4.3.3.1:
//!
//! * `pps_pic_parameter_set_id` range 0..=63.
//! * `pps_seq_parameter_set_id` range 0..=15.
//! * `num_ref_idx_l{0,1}_default_active_minus1` range 0..=14.
//! * `init_qp_minus26` range −( 26 + QpBdOffsetY ) to +25. The
//!   lower bound depends on the bit depth from the active SPS
//!   (`QpBdOffsetY = 6 * bit_depth_luma_minus8`); without an SPS the
//!   parser uses the loosest legal bound, −( 26 + 6*8 ) = −74, and a
//!   caller that has resolved the SPS can re-check against the exact
//!   bound via [`PicParameterSet::init_qp_in_range`].
//! * `pps_cb_qp_offset` / `pps_cr_qp_offset` range −12..=12.
//! * `pps_beta_offset_div2` / `pps_tc_offset_div2` range −6..=6.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::scaling_list::{ScalingListData, ScalingListError};
use crate::hevc::engine::sps::OpaqueTail;

/// Errors that can arise while parsing a PPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpsError {
    /// The RBSP ran out of bits before the PPS was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.3.3.1.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as an `i64` (covers both `ue(v)` and
        /// `se(v)` elements).
        got: i64,
    },
    /// A `scaling_list_data()` parse (§7.3.4) from the PPS failed.
    ScalingList(ScalingListError),
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for PpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("PPS RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "PPS syntax element {field} out of range: {got}")
            }
            Self::ScalingList(e) => write!(f, "scaling-list error during PPS parse: {e}"),
            Self::Bitstream(e) => write!(f, "bitstream error during PPS parse: {e}"),
        }
    }
}

impl std::error::Error for PpsError {}

impl From<BitReaderError> for PpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<ScalingListError> for PpsError {
    fn from(e: ScalingListError) -> Self {
        // Flatten truncation / raw-reader faults to the PPS-level
        // equivalents; carry structured scaling-list faults as-is.
        match e {
            ScalingListError::Truncated => Self::Truncated,
            ScalingListError::Bitstream(b) => Self::Bitstream(b),
            other => Self::ScalingList(other),
        }
    }
}

/// Tiles partitioning block per §7.3.2.3.1 (`tiles_enabled_flag == 1`).
///
/// When `tiles_enabled_flag` is 0 this block is absent and the
/// derived values are inferred per §7.4.3.3.1 (one column, one row,
/// uniform spacing, loop filtering across tiles enabled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    /// `num_tile_columns_minus1` (`ue(v)`). The number of tile columns
    /// is `value + 1`.
    pub num_tile_columns_minus1: u32,
    /// `num_tile_rows_minus1` (`ue(v)`). The number of tile rows is
    /// `value + 1`.
    pub num_tile_rows_minus1: u32,
    /// `uniform_spacing_flag`. When true the explicit
    /// `column_width_minus1[]` / `row_height_minus1[]` arrays are
    /// absent and the partitioning is uniform.
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[i]` for `i` in
    /// `0..num_tile_columns_minus1`. Empty when `uniform_spacing_flag`.
    pub column_width_minus1: Vec<u32>,
    /// `row_height_minus1[i]` for `i` in `0..num_tile_rows_minus1`.
    /// Empty when `uniform_spacing_flag`.
    pub row_height_minus1: Vec<u32>,
}

/// Deblocking-filter-control block per §7.3.2.3.1
/// (`deblocking_filter_control_present_flag == 1`).
///
/// The [`Default`] value matches the §7.4.3.3.1 inference for the
/// absent-control case: every flag false and both offsets 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeblockingFilterControl {
    /// `deblocking_filter_override_enabled_flag`. Inferred to false
    /// when the control block is absent (§7.4.3.3.1).
    pub override_enabled_flag: bool,
    /// `pps_deblocking_filter_disabled_flag`. Inferred to false when
    /// the control block is absent (§7.4.3.3.1).
    pub disabled_flag: bool,
    /// `pps_beta_offset_div2` (`se(v)`, range −6..=6). Inferred to 0
    /// when absent (the disabled / no-control case).
    pub beta_offset_div2: i8,
    /// `pps_tc_offset_div2` (`se(v)`, range −6..=6). Inferred to 0
    /// when absent.
    pub tc_offset_div2: i8,
}

/// PPS extension-flag block per §7.3.2.3.1
/// (`pps_extension_present_flag == 1`), holding the four typed
/// extension-present flags and the reserved-for-future-use
/// `pps_extension_4bits` group as a single `u8` (low 4 bits carry the
/// value; high 4 bits are zero).
///
/// Per §7.4.3.3.1, when `pps_extension_present_flag == 0` every flag
/// in this block is inferred to 0 and `pps_extension_4bits` is
/// inferred to 0; the parser surfaces that case as
/// [`PicParameterSet::extension_flags`] = `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PpsExtensionFlags {
    /// `pps_range_extension_flag` (§7.3.2.3.1). When true, a
    /// `pps_range_extension()` body (§7.3.2.3.2) follows in the bit
    /// stream and is currently surfaced inside the PPS
    /// [`PicParameterSet::opaque_tail`].
    pub pps_range_extension_flag: bool,
    /// `pps_multilayer_extension_flag` (§7.3.2.3.1, Annex F). When
    /// true, a `pps_multilayer_extension()` body follows and is
    /// surfaced inside the opaque tail.
    pub pps_multilayer_extension_flag: bool,
    /// `pps_3d_extension_flag` (§7.3.2.3.1, Annex I). When true, a
    /// `pps_3d_extension()` body follows and is surfaced inside the
    /// opaque tail.
    pub pps_3d_extension_flag: bool,
    /// `pps_scc_extension_flag` (§7.3.2.3.1). When true, a
    /// `pps_scc_extension()` body follows and is surfaced inside the
    /// opaque tail.
    pub pps_scc_extension_flag: bool,
    /// `pps_extension_4bits` (`u(4)`). For bitstreams conforming to
    /// the current version of the specification this value shall be
    /// 0; non-zero values are reserved for future use. The §7.4.3.3.1
    /// decoder-side rule is to allow any value and ignore the
    /// `pps_extension_data_flag` while-loop it gates, so the parser
    /// surfaces the value verbatim. The trailing
    /// `while( more_rbsp_data() ) pps_extension_data_flag` block (only
    /// signalled when this field is non-zero) is surfaced inside the
    /// opaque tail.
    pub pps_extension_4bits: u8,
}

impl PpsExtensionFlags {
    /// True when at least one of the four extension flags is set or
    /// `pps_extension_4bits` is non-zero — i.e. when at least one
    /// downstream extension body follows in the bit stream and the
    /// PPS therefore carries an opaque tail starting at the first
    /// body's bit position.
    pub fn has_body(&self) -> bool {
        self.pps_range_extension_flag
            || self.pps_multilayer_extension_flag
            || self.pps_3d_extension_flag
            || self.pps_scc_extension_flag
            || self.pps_extension_4bits != 0
    }

    /// True when the `pps_scc_extension()` body can be decoded in
    /// place — it is signalled and no still-opaque body
    /// (`pps_multilayer_extension()` / `pps_3d_extension()`) precedes it
    /// in the bit stream. When a multilayer / 3D body precedes it, the
    /// SCC body stays inside the opaque tail.
    fn scc_decodable_in_place(&self) -> bool {
        self.pps_scc_extension_flag
            && !self.pps_multilayer_extension_flag
            && !self.pps_3d_extension_flag
    }

    /// True when an extension body still follows the (range + optionally
    /// SCC) bodies decoded in place — i.e. a multilayer / 3D body, the
    /// `pps_extension_data_flag` while-loop, or an SCC body whose
    /// multilayer / 3D predecessor kept it opaque.
    fn has_opaque_body_after_decoded(&self) -> bool {
        if self.pps_multilayer_extension_flag || self.pps_3d_extension_flag {
            // The first un-decoded body is the multilayer / 3D one;
            // everything from there (incl. any SCC body) is opaque.
            return true;
        }
        // No multilayer / 3D body: SCC (if present) was decoded in
        // place, so only the pps_extension_data_flag while-loop may
        // remain.
        self.pps_extension_4bits != 0
    }
}

/// One `( cb_qp_offset_list[i], cr_qp_offset_list[i] )` pair from the
/// `pps_range_extension()` chroma-QP-offset list (§7.3.2.3.2). Both
/// offsets are `se(v)` values constrained to −12..=+12 (§7.4.3.3.2),
/// used in the §8.6.1 derivation of `Qp′Cb` / `Qp′Cr` when
/// `cu_chroma_qp_offset_flag` selects entry `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChromaQpOffsetListEntry {
    /// `cb_qp_offset_list[i]` (`se(v)`, −12..=+12).
    pub cb_qp_offset: i8,
    /// `cr_qp_offset_list[i]` (`se(v)`, −12..=+12).
    pub cr_qp_offset: i8,
}

/// Decoded `pps_range_extension()` body per §7.3.2.3.2, present when
/// [`PpsExtensionFlags::pps_range_extension_flag`] is set. Per
/// §7.4.3.3.2 the absent fields are inferred to 0 / empty when this
/// struct is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PpsRangeExtension {
    /// `log2_max_transform_skip_block_size_minus2` (`ue(v)`), present
    /// only when `transform_skip_enabled_flag` is set; inferred to 0
    /// otherwise. `Log2MaxTransformSkipSize = value + 2` (eq. 7-38).
    pub log2_max_transform_skip_block_size_minus2: u32,
    /// `cross_component_prediction_enabled_flag` — when 1,
    /// `log2_res_scale_abs_plus1` / `res_scale_sign_flag` may be
    /// present in the transform-unit syntax.
    pub cross_component_prediction_enabled_flag: bool,
    /// `chroma_qp_offset_list_enabled_flag` — when 1,
    /// `cu_chroma_qp_offset_flag` may be present in the transform-unit
    /// syntax and [`Self::chroma_qp_offset_list`] is populated.
    pub chroma_qp_offset_list_enabled_flag: bool,
    /// `diff_cu_chroma_qp_offset_depth` (`ue(v)`), present only when
    /// `chroma_qp_offset_list_enabled_flag`. `Log2MinCuChromaQpOffsetSize
    /// = CtbLog2SizeY − value` (eq. 7-39).
    pub diff_cu_chroma_qp_offset_depth: u32,
    /// `chroma_qp_offset_list_len_minus1` (`ue(v)`, 0..=5), present
    /// only when `chroma_qp_offset_list_enabled_flag`. The
    /// [`Self::chroma_qp_offset_list`] then holds `value + 1` entries.
    pub chroma_qp_offset_list_len_minus1: u32,
    /// The `cb_qp_offset_list[] / cr_qp_offset_list[]` pairs
    /// (§7.3.2.3.2). Empty when `chroma_qp_offset_list_enabled_flag`
    /// is 0; otherwise holds `chroma_qp_offset_list_len_minus1 + 1`
    /// entries.
    pub chroma_qp_offset_list: Vec<ChromaQpOffsetListEntry>,
    /// `log2_sao_offset_scale_luma` (`ue(v)`) — base-2 log of the SAO
    /// luma-offset scaling parameter (range 0..=Max(0, BitDepthY−10)).
    pub log2_sao_offset_scale_luma: u32,
    /// `log2_sao_offset_scale_chroma` (`ue(v)`) — base-2 log of the
    /// SAO chroma-offset scaling parameter (0..=Max(0, BitDepthC−10)).
    pub log2_sao_offset_scale_chroma: u32,
}

impl PpsRangeExtension {
    /// Maximum permitted `chroma_qp_offset_list_len_minus1` (§7.4.3.3.2).
    const MAX_CHROMA_QP_OFFSET_LIST_LEN_MINUS1: u32 = 5;

    /// Decode `pps_range_extension()` (§7.3.2.3.2). The leading
    /// `log2_max_transform_skip_block_size_minus2` is present only when
    /// `transform_skip_enabled_flag` (a general-body field) is set.
    fn parse(br: &mut BitReader, transform_skip_enabled_flag: bool) -> Result<Self, PpsError> {
        let log2_max_transform_skip_block_size_minus2 = if transform_skip_enabled_flag {
            br.ue()?
        } else {
            0
        };
        let cross_component_prediction_enabled_flag = br.u1()? != 0;
        let chroma_qp_offset_list_enabled_flag = br.u1()? != 0;
        let mut diff_cu_chroma_qp_offset_depth = 0u32;
        let mut chroma_qp_offset_list_len_minus1 = 0u32;
        let mut chroma_qp_offset_list = Vec::new();
        if chroma_qp_offset_list_enabled_flag {
            diff_cu_chroma_qp_offset_depth = br.ue()?;
            chroma_qp_offset_list_len_minus1 = br.ue()?;
            if chroma_qp_offset_list_len_minus1 > Self::MAX_CHROMA_QP_OFFSET_LIST_LEN_MINUS1 {
                return Err(PpsError::ValueOutOfRange {
                    field: "chroma_qp_offset_list_len_minus1",
                    got: chroma_qp_offset_list_len_minus1 as i64,
                });
            }
            let len = chroma_qp_offset_list_len_minus1 as usize + 1;
            chroma_qp_offset_list.reserve(len);
            for _ in 0..len {
                let cb = br.se()?;
                let cr = br.se()?;
                for (field, v) in [("cb_qp_offset_list", cb), ("cr_qp_offset_list", cr)] {
                    if !(-12..=12).contains(&v) {
                        return Err(PpsError::ValueOutOfRange {
                            field,
                            got: v as i64,
                        });
                    }
                }
                chroma_qp_offset_list.push(ChromaQpOffsetListEntry {
                    cb_qp_offset: cb as i8,
                    cr_qp_offset: cr as i8,
                });
            }
        }
        let log2_sao_offset_scale_luma = br.ue()?;
        let log2_sao_offset_scale_chroma = br.ue()?;
        Ok(Self {
            log2_max_transform_skip_block_size_minus2,
            cross_component_prediction_enabled_flag,
            chroma_qp_offset_list_enabled_flag,
            diff_cu_chroma_qp_offset_depth,
            chroma_qp_offset_list_len_minus1,
            chroma_qp_offset_list,
            log2_sao_offset_scale_luma,
            log2_sao_offset_scale_chroma,
        })
    }
}

/// Decoded `pps_scc_extension()` body per §7.3.2.3.3, present when
/// [`PpsExtensionFlags::pps_scc_extension_flag`] is set and no opaque
/// multilayer / 3D body precedes it. Per §7.4.3.3.3 the absent fields
/// are inferred to 0 / empty when this struct is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PpsSccExtension {
    /// `pps_curr_pic_ref_enabled_flag` — when 1, a picture referring to
    /// the PPS may appear in one of its own slices' reference picture
    /// lists (intra block copy at the picture level).
    pub pps_curr_pic_ref_enabled_flag: bool,
    /// `residual_adaptive_colour_transform_enabled_flag` — when 1, the
    /// adaptive colour transform may be applied to residuals.
    pub residual_adaptive_colour_transform_enabled_flag: bool,
    /// `pps_slice_act_qp_offsets_present_flag` — present only when
    /// `residual_adaptive_colour_transform_enabled_flag`; when 1, the
    /// `slice_act_*_qp_offset` fields are present in slice headers.
    pub pps_slice_act_qp_offsets_present_flag: bool,
    /// `pps_act_y_qp_offset_plus5` (`se(v)`), present only when
    /// `residual_adaptive_colour_transform_enabled_flag`.
    /// `PpsActQpOffsetY = value − 5` (eq. 7-39).
    pub pps_act_y_qp_offset_plus5: i32,
    /// `pps_act_cb_qp_offset_plus5` (`se(v)`), present only when
    /// `residual_adaptive_colour_transform_enabled_flag`.
    /// `PpsActQpOffsetCb = value − 5` (eq. 7-40).
    pub pps_act_cb_qp_offset_plus5: i32,
    /// `pps_act_cr_qp_offset_plus3` (`se(v)`), present only when
    /// `residual_adaptive_colour_transform_enabled_flag`.
    /// `PpsActQpOffsetCr = value − 3` (eq. 7-41).
    pub pps_act_cr_qp_offset_plus3: i32,
    /// `pps_palette_predictor_initializers_present_flag` — when 1, the
    /// picture palette predictor is initialised from
    /// [`Self::pps_palette_predictor_initializer`].
    pub pps_palette_predictor_initializers_present_flag: bool,
    /// `pps_num_palette_predictor_initializers` (`ue(v)`), present only
    /// when the initializers-present flag is set; the initializer table
    /// then holds this many entries per component.
    pub pps_num_palette_predictor_initializers: u32,
    /// `monochrome_palette_flag` — present only when there is at least
    /// one initializer; selects `numComps` (1 when set, else 3).
    pub monochrome_palette_flag: bool,
    /// `luma_bit_depth_entry_minus8` (`ue(v)`); the initializer's luma
    /// component is `u(v)` of `luma_bit_depth_entry_minus8 + 8` bits.
    pub luma_bit_depth_entry_minus8: u32,
    /// `chroma_bit_depth_entry_minus8` (`ue(v)`), present only when not
    /// `monochrome_palette_flag`; chroma components are `u(v)` of
    /// `chroma_bit_depth_entry_minus8 + 8` bits.
    pub chroma_bit_depth_entry_minus8: u32,
    /// `pps_palette_predictor_initializer[comp][i]` (§7.3.2.3.3),
    /// indexed `[comp][i]`. `comp` runs over `numComps`. Empty when no
    /// initializers are signalled.
    pub pps_palette_predictor_initializer: Vec<Vec<u32>>,
}

impl PpsSccExtension {
    /// Decode `pps_scc_extension()` (§7.3.2.3.3). Unlike the SPS variant,
    /// the palette predictor initializer `u(v)` widths come from this
    /// body's own `luma_bit_depth_entry_minus8` /
    /// `chroma_bit_depth_entry_minus8`, so no SPS context is needed.
    fn parse(br: &mut BitReader) -> Result<Self, PpsError> {
        let pps_curr_pic_ref_enabled_flag = br.u1()? != 0;
        let residual_adaptive_colour_transform_enabled_flag = br.u1()? != 0;
        let mut pps_slice_act_qp_offsets_present_flag = false;
        let mut pps_act_y_qp_offset_plus5 = 0i32;
        let mut pps_act_cb_qp_offset_plus5 = 0i32;
        let mut pps_act_cr_qp_offset_plus3 = 0i32;
        if residual_adaptive_colour_transform_enabled_flag {
            pps_slice_act_qp_offsets_present_flag = br.u1()? != 0;
            pps_act_y_qp_offset_plus5 = br.se()?;
            pps_act_cb_qp_offset_plus5 = br.se()?;
            pps_act_cr_qp_offset_plus3 = br.se()?;
            // §7.4.3.3.3: PpsActQpOffsetY/Cb/Cr (eq. 7-39/40/41 — the
            // raw values minus 5/5/3) must each lie in −12..=12 in a
            // conforming bitstream.
            // Widened arithmetic: se(v) can carry the full i32 range on
            // a malformed stream, so the eq.-7-39..7-41 "minus 5/5/3"
            // must not overflow before the range check rejects it.
            for (field, qp) in [
                (
                    "pps_act_y_qp_offset_plus5",
                    i64::from(pps_act_y_qp_offset_plus5) - 5,
                ),
                (
                    "pps_act_cb_qp_offset_plus5",
                    i64::from(pps_act_cb_qp_offset_plus5) - 5,
                ),
                (
                    "pps_act_cr_qp_offset_plus3",
                    i64::from(pps_act_cr_qp_offset_plus3) - 3,
                ),
            ] {
                if !(-12..=12).contains(&qp) {
                    return Err(PpsError::ValueOutOfRange { field, got: qp });
                }
            }
        }
        let pps_palette_predictor_initializers_present_flag = br.u1()? != 0;
        let mut pps_num_palette_predictor_initializers = 0u32;
        let mut monochrome_palette_flag = false;
        let mut luma_bit_depth_entry_minus8 = 0u32;
        let mut chroma_bit_depth_entry_minus8 = 0u32;
        let mut pps_palette_predictor_initializer = Vec::new();
        if pps_palette_predictor_initializers_present_flag {
            pps_num_palette_predictor_initializers = br.ue()?;
            // §7.4.3.3.3: bounded by PaletteMaxPredictorSize, itself
            // capped by the profile limits (palette_max_size +
            // delta_palette_max_predictor_size); reject anything past
            // the largest representable predictor so a malformed count
            // cannot drive the initializer allocation.
            if pps_num_palette_predictor_initializers > 128 {
                return Err(PpsError::ValueOutOfRange {
                    field: "pps_num_palette_predictor_initializers",
                    got: i64::from(pps_num_palette_predictor_initializers),
                });
            }
            if pps_num_palette_predictor_initializers > 0 {
                monochrome_palette_flag = br.u1()? != 0;
                luma_bit_depth_entry_minus8 = br.ue()?;
                if !monochrome_palette_flag {
                    chroma_bit_depth_entry_minus8 = br.ue()?;
                }
                // §7.4.3.3.3: the entry bit depths are BitDepth values
                // (8..=16), i.e. the minus8 fields lie in 0..=8 — checked
                // before the `+ 8` width arithmetic below.
                for (field, v) in [
                    ("luma_bit_depth_entry_minus8", luma_bit_depth_entry_minus8),
                    (
                        "chroma_bit_depth_entry_minus8",
                        chroma_bit_depth_entry_minus8,
                    ),
                ] {
                    if v > 8 {
                        return Err(PpsError::ValueOutOfRange {
                            field,
                            got: i64::from(v),
                        });
                    }
                }
                let num_comps = if monochrome_palette_flag { 1 } else { 3 };
                let num_entries = pps_num_palette_predictor_initializers as usize;
                pps_palette_predictor_initializer.reserve(num_comps);
                for comp in 0..num_comps {
                    let width = if comp == 0 {
                        (luma_bit_depth_entry_minus8 + 8) as u8
                    } else {
                        (chroma_bit_depth_entry_minus8 + 8) as u8
                    };
                    let mut row = Vec::with_capacity(num_entries);
                    for _ in 0..num_entries {
                        row.push(br.u(width)?);
                    }
                    pps_palette_predictor_initializer.push(row);
                }
            }
        }
        Ok(Self {
            pps_curr_pic_ref_enabled_flag,
            residual_adaptive_colour_transform_enabled_flag,
            pps_slice_act_qp_offsets_present_flag,
            pps_act_y_qp_offset_plus5,
            pps_act_cb_qp_offset_plus5,
            pps_act_cr_qp_offset_plus3,
            pps_palette_predictor_initializers_present_flag,
            pps_num_palette_predictor_initializers,
            monochrome_palette_flag,
            luma_bit_depth_entry_minus8,
            chroma_bit_depth_entry_minus8,
            pps_palette_predictor_initializer,
        })
    }

    /// `PpsActQpOffsetY` (eq. 7-39) — `pps_act_y_qp_offset_plus5 − 5`.
    pub fn pps_act_qp_offset_y(&self) -> i32 {
        self.pps_act_y_qp_offset_plus5 - 5
    }

    /// `PpsActQpOffsetCb` (eq. 7-40) — `pps_act_cb_qp_offset_plus5 − 5`.
    pub fn pps_act_qp_offset_cb(&self) -> i32 {
        self.pps_act_cb_qp_offset_plus5 - 5
    }

    /// `PpsActQpOffsetCr` (eq. 7-41) — `pps_act_cr_qp_offset_plus3 − 3`.
    pub fn pps_act_qp_offset_cr(&self) -> i32 {
        self.pps_act_cr_qp_offset_plus3 - 3
    }
}

/// Parsed Picture Parameter Set per §7.3.2.3.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicParameterSet {
    /// `pps_pic_parameter_set_id` (`ue(v)`, range 0..=63).
    pub pps_id: u8,
    /// `pps_seq_parameter_set_id` (`ue(v)`, range 0..=15) — the SPS
    /// this PPS activates.
    pub sps_id: u8,
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments_enabled_flag: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present_flag: bool,
    /// `num_extra_slice_header_bits` (`u(3)`). The conformance range
    /// is 0..=2 but decoders must accept any value, so this is not
    /// range-checked.
    pub num_extra_slice_header_bits: u8,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_data_hiding_enabled_flag: bool,
    /// `cabac_init_present_flag`.
    pub cabac_init_present_flag: bool,
    /// `num_ref_idx_l0_default_active_minus1` (`ue(v)`, range 0..=14).
    /// `num_ref_idx_l0_default_active = value + 1`.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// `num_ref_idx_l1_default_active_minus1` (`ue(v)`, range 0..=14).
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// `init_qp_minus26` (`se(v)`). `init_qp = value + 26`.
    pub init_qp_minus26: i32,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag`. When set, [`Self::diff_cu_qp_delta_depth`]
    /// is signalled; otherwise it is inferred to 0.
    pub cu_qp_delta_enabled_flag: bool,
    /// `diff_cu_qp_delta_depth` (`ue(v)`). Inferred to 0 when
    /// `cu_qp_delta_enabled_flag` is false (§7.4.3.3.1).
    pub diff_cu_qp_delta_depth: u32,
    /// `pps_cb_qp_offset` (`se(v)`, range −12..=12).
    pub pps_cb_qp_offset: i8,
    /// `pps_cr_qp_offset` (`se(v)`, range −12..=12).
    pub pps_cr_qp_offset: i8,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_flag`.
    pub weighted_bipred_flag: bool,
    /// `transquant_bypass_enabled_flag`.
    pub transquant_bypass_enabled_flag: bool,
    /// `tiles_enabled_flag`. When set, [`Self::tiles`] carries the
    /// signalled values; otherwise it carries the §7.4.3.3.1 inferred
    /// single-tile values.
    pub tiles_enabled_flag: bool,
    /// `entropy_coding_sync_enabled_flag`.
    pub entropy_coding_sync_enabled_flag: bool,
    /// Tiles block. Present (with inferred single-tile values) even
    /// when `tiles_enabled_flag` is false, so callers always have an
    /// effective tile geometry.
    pub tiles: TileInfo,
    /// `loop_filter_across_tiles_enabled_flag`. Inferred to true when
    /// the tiles block is absent (§7.4.3.3.1).
    pub loop_filter_across_tiles_enabled_flag: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// Deblocking-filter-control values, carrying §7.4.3.3.1 inferred
    /// defaults when the control block is absent.
    pub deblocking: DeblockingFilterControl,
    /// `pps_scaling_list_data_present_flag` (§7.3.2.3.1). When set, the
    /// PPS carries an explicit [`Self::scaling_list_data`].
    pub pps_scaling_list_data_present_flag: bool,
    /// The parsed §7.3.4 `scaling_list_data()` structure when
    /// `pps_scaling_list_data_present_flag == 1`; `None` otherwise.
    pub scaling_list_data: Option<ScalingListData>,
    /// `lists_modification_present_flag`.
    pub lists_modification_present_flag: bool,
    /// `log2_parallel_merge_level_minus2` (`ue(v)`).
    /// `Log2ParMrgLevel = value + 2`.
    pub log2_parallel_merge_level_minus2: u32,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_segment_header_extension_present_flag: bool,
    /// `pps_extension_present_flag`. When set, the
    /// [`Self::extension_flags`] block is decoded into typed fields;
    /// any extension body whose flag is set (and the
    /// `pps_extension_data_flag` while-loop gated by
    /// `pps_extension_4bits != 0`) is surfaced inside
    /// [`Self::opaque_tail`].
    pub pps_extension_present_flag: bool,
    /// Typed extension-flag block per §7.3.2.3.1
    /// (`pps_extension_present_flag == 1`). `None` when the gate is
    /// absent; in that case every flag is inferred to 0 per the
    /// §7.4.3.3.1 inference rules.
    pub extension_flags: Option<PpsExtensionFlags>,
    /// Decoded `pps_range_extension()` body (§7.3.2.3.2), present when
    /// `extension_flags.pps_range_extension_flag` is set. `None`
    /// otherwise; per §7.4.3.3.2 every field is then inferred to 0 /
    /// empty.
    pub pps_range_extension: Option<PpsRangeExtension>,
    /// Decoded `pps_scc_extension()` body (§7.3.2.3.3), present when
    /// `extension_flags.pps_scc_extension_flag` is set **and** no opaque
    /// multilayer / 3D body precedes it. `None` otherwise; per
    /// §7.4.3.3.3 every field is then inferred to 0 / empty.
    pub pps_scc_extension: Option<PpsSccExtension>,
    /// Opaque suffix of the PPS RBSP. Populated when
    /// `pps_extension_present_flag == 1` **and**
    /// [`PpsExtensionFlags::has_body`] is true on the decoded flags
    /// — the bit position recorded inside the tail is the start of
    /// the first signalled extension body (`pps_range_extension()`
    /// if `pps_range_extension_flag`, otherwise the next set flag's
    /// body, otherwise the `pps_extension_data_flag` while-loop). The
    /// individual extension-body syntax structures are not yet
    /// decoded. `None` when the PPS ended cleanly after the flag
    /// block (no extension body is signalled and only
    /// `rbsp_trailing_bits()` remains; consumed implicitly) or when
    /// `pps_extension_present_flag == 0`.
    pub opaque_tail: Option<OpaqueTail>,
}

impl PicParameterSet {
    /// Parse `pic_parameter_set_rbsp()` starting from the first bit of
    /// the (already-unescaped) RBSP body — i.e. after the two-byte NAL
    /// header has been removed (see [`crate::hevc::engine::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, PpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br, rbsp)
    }

    fn parse_inner(br: &mut BitReader<'_>, rbsp: &[u8]) -> Result<Self, PpsError> {
        let pps_id_raw = br.ue()?;
        if pps_id_raw > 63 {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_pic_parameter_set_id",
                got: pps_id_raw as i64,
            });
        }
        let sps_id_raw = br.ue()?;
        if sps_id_raw > 15 {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_seq_parameter_set_id",
                got: sps_id_raw as i64,
            });
        }

        let dependent_slice_segments_enabled_flag = br.u1()? != 0;
        let output_flag_present_flag = br.u1()? != 0;
        let num_extra_slice_header_bits = br.u(3)? as u8;
        let sign_data_hiding_enabled_flag = br.u1()? != 0;
        let cabac_init_present_flag = br.u1()? != 0;

        let num_ref_idx_l0_default_active_minus1_raw = br.ue()?;
        if num_ref_idx_l0_default_active_minus1_raw > 14 {
            return Err(PpsError::ValueOutOfRange {
                field: "num_ref_idx_l0_default_active_minus1",
                got: num_ref_idx_l0_default_active_minus1_raw as i64,
            });
        }
        let num_ref_idx_l1_default_active_minus1_raw = br.ue()?;
        if num_ref_idx_l1_default_active_minus1_raw > 14 {
            return Err(PpsError::ValueOutOfRange {
                field: "num_ref_idx_l1_default_active_minus1",
                got: num_ref_idx_l1_default_active_minus1_raw as i64,
            });
        }

        let init_qp_minus26 = br.se()?;
        // §7.4.3.3.1: −( 26 + QpBdOffsetY ) .. +25. The lower bound
        // depends on the active SPS bit depth; QpBdOffsetY is at most
        // 6 * 8 = 48 (the maximum legal bit_depth_luma_minus8), so the
        // loosest legal lower bound is −74. We range-check against the
        // loosest bound here (no SPS context) and provide
        // `init_qp_in_range()` for callers that have resolved the SPS.
        if !(-74..=25).contains(&init_qp_minus26) {
            return Err(PpsError::ValueOutOfRange {
                field: "init_qp_minus26",
                got: init_qp_minus26 as i64,
            });
        }

        let constrained_intra_pred_flag = br.u1()? != 0;
        let transform_skip_enabled_flag = br.u1()? != 0;

        let cu_qp_delta_enabled_flag = br.u1()? != 0;
        let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
            br.ue()?
        } else {
            // §7.4.3.3.1 inference.
            0
        };

        let pps_cb_qp_offset = br.se()?;
        if !(-12..=12).contains(&pps_cb_qp_offset) {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_cb_qp_offset",
                got: pps_cb_qp_offset as i64,
            });
        }
        let pps_cr_qp_offset = br.se()?;
        if !(-12..=12).contains(&pps_cr_qp_offset) {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_cr_qp_offset",
                got: pps_cr_qp_offset as i64,
            });
        }

        let pps_slice_chroma_qp_offsets_present_flag = br.u1()? != 0;
        let weighted_pred_flag = br.u1()? != 0;
        let weighted_bipred_flag = br.u1()? != 0;
        let transquant_bypass_enabled_flag = br.u1()? != 0;
        let tiles_enabled_flag = br.u1()? != 0;
        let entropy_coding_sync_enabled_flag = br.u1()? != 0;

        let (tiles, loop_filter_across_tiles_enabled_flag) = if tiles_enabled_flag {
            let num_tile_columns_minus1 = br.ue()?;
            let num_tile_rows_minus1 = br.ue()?;
            // §A.4.1 item f): "num_tile_columns_minus1 shall be less
            // than MaxTileCols and num_tile_rows_minus1 shall be less
            // than MaxTileRows". The largest Table A.8 entries are
            // MaxTileCols = 40 / MaxTileRows = 44 (levels 7 .. 7.2);
            // the tighter §7.4.3.3.1 PicWidthInCtbsY-relative bound
            // needs the active SPS, which the PPS parse does not see.
            // Rejecting here also bounds the explicit width / height
            // array allocations below.
            if num_tile_columns_minus1 >= 40 {
                return Err(PpsError::ValueOutOfRange {
                    field: "num_tile_columns_minus1",
                    got: num_tile_columns_minus1 as i64,
                });
            }
            if num_tile_rows_minus1 >= 44 {
                return Err(PpsError::ValueOutOfRange {
                    field: "num_tile_rows_minus1",
                    got: num_tile_rows_minus1 as i64,
                });
            }
            let uniform_spacing_flag = br.u1()? != 0;
            let (column_width_minus1, row_height_minus1) = if !uniform_spacing_flag {
                let mut cols = Vec::with_capacity(num_tile_columns_minus1 as usize);
                for _ in 0..num_tile_columns_minus1 {
                    cols.push(br.ue()?);
                }
                let mut rows = Vec::with_capacity(num_tile_rows_minus1 as usize);
                for _ in 0..num_tile_rows_minus1 {
                    rows.push(br.ue()?);
                }
                (cols, rows)
            } else {
                (Vec::new(), Vec::new())
            };
            let loop_filter_across_tiles_enabled_flag = br.u1()? != 0;
            (
                TileInfo {
                    num_tile_columns_minus1,
                    num_tile_rows_minus1,
                    uniform_spacing_flag,
                    column_width_minus1,
                    row_height_minus1,
                },
                loop_filter_across_tiles_enabled_flag,
            )
        } else {
            // §7.4.3.3.1 inference: one column, one row, uniform
            // spacing, loop filtering across tiles enabled.
            (
                TileInfo {
                    num_tile_columns_minus1: 0,
                    num_tile_rows_minus1: 0,
                    uniform_spacing_flag: true,
                    column_width_minus1: Vec::new(),
                    row_height_minus1: Vec::new(),
                },
                true,
            )
        };

        let pps_loop_filter_across_slices_enabled_flag = br.u1()? != 0;

        let deblocking_filter_control_present_flag = br.u1()? != 0;
        let deblocking = if deblocking_filter_control_present_flag {
            let override_enabled_flag = br.u1()? != 0;
            let disabled_flag = br.u1()? != 0;
            let (beta_offset_div2, tc_offset_div2) = if !disabled_flag {
                let beta = br.se()?;
                if !(-6..=6).contains(&beta) {
                    return Err(PpsError::ValueOutOfRange {
                        field: "pps_beta_offset_div2",
                        got: beta as i64,
                    });
                }
                let tc = br.se()?;
                if !(-6..=6).contains(&tc) {
                    return Err(PpsError::ValueOutOfRange {
                        field: "pps_tc_offset_div2",
                        got: tc as i64,
                    });
                }
                (beta as i8, tc as i8)
            } else {
                // §7.4.3.3.1: beta/tc offsets inferred to 0 when the
                // deblocking filter is disabled at PPS level.
                (0, 0)
            };
            DeblockingFilterControl {
                override_enabled_flag,
                disabled_flag,
                beta_offset_div2,
                tc_offset_div2,
            }
        } else {
            DeblockingFilterControl::default()
        };

        let pps_scaling_list_data_present_flag = br.u1()? != 0;
        let scaling_list_data = if pps_scaling_list_data_present_flag {
            // scaling_list_data() (§7.3.4), shared with the SPS path.
            Some(ScalingListData::parse(br)?)
        } else {
            None
        };

        let lists_modification_present_flag = br.u1()? != 0;
        let log2_parallel_merge_level_minus2 = br.ue()?;
        let slice_segment_header_extension_present_flag = br.u1()? != 0;

        let pps_extension_present_flag = br.u1()? != 0;
        let (extension_flags, pps_range_extension, pps_scc_extension, opaque_tail) =
            if pps_extension_present_flag {
                // §7.3.2.3.1: when the gate is open, decode the eight bits
                // of typed extension flags first.
                let pps_range_extension_flag = br.u1()? != 0;
                let pps_multilayer_extension_flag = br.u1()? != 0;
                let pps_3d_extension_flag = br.u1()? != 0;
                let pps_scc_extension_flag = br.u1()? != 0;
                let pps_extension_4bits = br.u(4)? as u8;
                let flags = PpsExtensionFlags {
                    pps_range_extension_flag,
                    pps_multilayer_extension_flag,
                    pps_3d_extension_flag,
                    pps_scc_extension_flag,
                    pps_extension_4bits,
                };
                // §7.3.2.3.1: the range extension (if signalled) is the
                // first body to follow the eight typed flag bits, so decode
                // it in full. Its leading
                // log2_max_transform_skip_block_size_minus2 is present only
                // when transform_skip_enabled_flag was set in the general
                // body.
                let range_ext = if flags.pps_range_extension_flag {
                    Some(PpsRangeExtension::parse(br, transform_skip_enabled_flag)?)
                } else {
                    None
                };
                // §7.3.2.3.1 body order is range, multilayer, 3d, scc. The
                // SCC body can be decoded in place only when no
                // (still-opaque) multilayer / 3D body precedes it;
                // otherwise it stays inside the opaque tail.
                let scc_ext = if flags.scc_decodable_in_place() {
                    Some(PpsSccExtension::parse(br)?)
                } else {
                    None
                };
                // If any still-opaque body (a multilayer / 3D body, an SCC
                // body kept opaque by such a predecessor, or the
                // pps_extension_data_flag while-loop) follows, capture the
                // rest of the RBSP as an opaque tail starting at the first
                // un-decoded body's bit position. Otherwise only
                // rbsp_trailing_bits remains, consumed implicitly.
                let tail = if flags.has_opaque_body_after_decoded() {
                    Some(OpaqueTail::capture_at(br.bit_pos(), rbsp))
                } else {
                    None
                };
                (Some(flags), range_ext, scc_ext, tail)
            } else {
                // §7.4.3.3.1: every extension flag inferred to 0; only
                // rbsp_trailing_bits remains, consumed implicitly.
                (None, None, None, None)
            };

        Ok(Self {
            pps_id: pps_id_raw as u8,
            sps_id: sps_id_raw as u8,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1: num_ref_idx_l0_default_active_minus1_raw as u8,
            num_ref_idx_l1_default_active_minus1: num_ref_idx_l1_default_active_minus1_raw as u8,
            init_qp_minus26,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
            cu_qp_delta_enabled_flag,
            diff_cu_qp_delta_depth,
            pps_cb_qp_offset: pps_cb_qp_offset as i8,
            pps_cr_qp_offset: pps_cr_qp_offset as i8,
            pps_slice_chroma_qp_offsets_present_flag,
            weighted_pred_flag,
            weighted_bipred_flag,
            transquant_bypass_enabled_flag,
            tiles_enabled_flag,
            entropy_coding_sync_enabled_flag,
            tiles,
            loop_filter_across_tiles_enabled_flag,
            pps_loop_filter_across_slices_enabled_flag,
            deblocking_filter_control_present_flag,
            deblocking,
            pps_scaling_list_data_present_flag,
            scaling_list_data,
            lists_modification_present_flag,
            log2_parallel_merge_level_minus2,
            slice_segment_header_extension_present_flag,
            pps_extension_present_flag,
            extension_flags,
            pps_range_extension,
            pps_scc_extension,
            opaque_tail,
        })
    }

    /// `init_qp = init_qp_minus26 + 26` per §7.4.3.3.1 — the
    /// per-picture base QP before any slice-header `slice_qp_delta`.
    pub fn init_qp(&self) -> i32 {
        self.init_qp_minus26 + 26
    }

    /// `num_ref_idx_l0_default_active = num_ref_idx_l0_default_active_minus1 + 1`.
    pub fn num_ref_idx_l0_default_active(&self) -> u8 {
        self.num_ref_idx_l0_default_active_minus1 + 1
    }

    /// `num_ref_idx_l1_default_active = num_ref_idx_l1_default_active_minus1 + 1`.
    pub fn num_ref_idx_l1_default_active(&self) -> u8 {
        self.num_ref_idx_l1_default_active_minus1 + 1
    }

    /// Number of tile columns: `num_tile_columns_minus1 + 1`.
    pub fn num_tile_columns(&self) -> u32 {
        self.tiles.num_tile_columns_minus1 + 1
    }

    /// Number of tile rows: `num_tile_rows_minus1 + 1`.
    pub fn num_tile_rows(&self) -> u32 {
        self.tiles.num_tile_rows_minus1 + 1
    }

    /// `Log2ParMrgLevel = log2_parallel_merge_level_minus2 + 2`
    /// (equation 7-37).
    pub fn log2_par_mrg_level(&self) -> u32 {
        self.log2_parallel_merge_level_minus2 + 2
    }

    /// Re-check `init_qp_minus26` against the exact §7.4.3.3.1 lower
    /// bound `−( 26 + QpBdOffsetY )` once the active SPS bit depth is
    /// known. `bit_depth_luma_minus8` is the value carried on the SPS
    /// this PPS activates (`QpBdOffsetY = 6 * bit_depth_luma_minus8`).
    /// The parser already verified the upper bound (+25) and the
    /// loosest lower bound (−74) during the parse.
    pub fn init_qp_in_range(&self, bit_depth_luma_minus8: u8) -> bool {
        let qp_bd_offset_y = 6 * bit_depth_luma_minus8 as i32;
        let lower = -(26 + qp_bd_offset_y);
        (lower..=25).contains(&self.init_qp_minus26)
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::nal::collect_nal_units;

    /// PPS RBSP body extracted from
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed. The wire PPS (NAL idx 2, type 34) was 6 bytes
    /// including the two-byte header (`44 01`); the body is the
    /// remaining 4 bytes. It carries no emulation-prevention bytes.
    const TINY_PPS_RBSP: &[u8] = &[0xC1, 0x73, 0xC0, 0x89];

    #[test]
    fn parses_tiny_fixture_pps() {
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS parse");
        // Trace cross-check
        // (docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt,
        //  line 3 "PPS ..."):
        //   pps_id=0 sps_id=0 dependent_slice_segments=0
        //   output_flag_present=0 num_extra_slice_header_bits=0
        //   sign_data_hiding=1 cabac_init_present=0
        //   num_ref_idx_l0_default=1 num_ref_idx_l1_default=1 init_qp=26
        //   constrained_intra_pred=0 transform_skip_enabled=0
        //   cu_qp_delta_enabled=1 diff_cu_qp_delta_depth=0
        //   weighted_pred=0 weighted_bipred=0 transquant_bypass=0
        //   tiles_enabled=0 num_tile_columns=1 num_tile_rows=1
        //   entropy_coding_sync_enabled=0 loop_filter_across_tiles=1
        //   deblock_filter_override_enabled=0 lists_modification_present=0
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.dependent_slice_segments_enabled_flag);
        assert!(!pps.output_flag_present_flag);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        assert!(pps.sign_data_hiding_enabled_flag);
        assert!(!pps.cabac_init_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l0_default_active(), 1);
        assert_eq!(pps.num_ref_idx_l1_default_active(), 1);
        assert_eq!(pps.init_qp_minus26, 0);
        assert_eq!(pps.init_qp(), 26);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.transform_skip_enabled_flag);
        assert!(pps.cu_qp_delta_enabled_flag);
        assert_eq!(pps.diff_cu_qp_delta_depth, 0);
        assert_eq!(pps.pps_cb_qp_offset, 0);
        assert_eq!(pps.pps_cr_qp_offset, 0);
        assert!(!pps.pps_slice_chroma_qp_offsets_present_flag);
        assert!(!pps.weighted_pred_flag);
        assert!(!pps.weighted_bipred_flag);
        assert!(!pps.transquant_bypass_enabled_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(!pps.entropy_coding_sync_enabled_flag);
        // tiles_enabled=0 → single tile inferred per §7.4.3.3.1.
        assert_eq!(pps.num_tile_columns(), 1);
        assert_eq!(pps.num_tile_rows(), 1);
        assert!(pps.tiles.uniform_spacing_flag);
        assert!(pps.tiles.column_width_minus1.is_empty());
        assert!(pps.tiles.row_height_minus1.is_empty());
        assert!(pps.loop_filter_across_tiles_enabled_flag); // inferred true
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        // deblocking control absent → defaults inferred.
        assert!(!pps.deblocking_filter_control_present_flag);
        assert!(!pps.deblocking.override_enabled_flag);
        assert!(!pps.deblocking.disabled_flag);
        assert_eq!(pps.deblocking.beta_offset_div2, 0);
        assert_eq!(pps.deblocking.tc_offset_div2, 0);
        assert!(!pps.pps_scaling_list_data_present_flag);
        assert!(!pps.lists_modification_present_flag);
        assert_eq!(pps.log2_parallel_merge_level_minus2, 0);
        assert!(!pps.slice_segment_header_extension_present_flag);
        // No PPS extension in this fixture → no opaque tail.
        assert!(!pps.pps_extension_present_flag);
        assert!(pps.opaque_tail.is_none());
    }

    /// End-to-end: pull the PPS NAL out of the raw Annex B stream
    /// through the walker, then parse its (already-unescaped, header-
    /// stripped) RBSP. The fixture's PPS NAL bytes (start code +
    /// `44 01 C1 73 C0 89`) are captured inline so the test does not
    /// depend on the `docs/` tree at run time.
    #[test]
    fn parses_tiny_fixture_pps_via_nal_walker() {
        // 4-byte start code + PPS NAL (type 34: header 0x44 0x01).
        let stream = [0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xC1, 0x73, 0xC0, 0x89];
        let nals = collect_nal_units(&stream).expect("collect");
        assert_eq!(nals.len(), 1);
        let pps_nal = &nals[0];
        assert_eq!(pps_nal.header.nal_unit_type, 34);
        // `NalUnit::rbsp` already excludes the two header bytes and has
        // emulation-prevention stripped.
        let pps = PicParameterSet::parse(&pps_nal.rbsp).expect("PPS parse via walker");
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert_eq!(pps.init_qp(), 26);
        assert!(pps.cu_qp_delta_enabled_flag);
    }

    /// Hand-assembled PPS exercising the tiles block with explicit
    /// (non-uniform) column/row spacing and the deblocking-control
    /// block with non-zero β / tC offsets.
    #[test]
    fn parses_tiles_and_deblocking_blocks() {
        // Build the RBSP bit-by-bit as a string, then pack MSB-first.
        let mut bits = String::new();
        // pps_pic_parameter_set_id = 0 → ue '1'
        bits += "1";
        // pps_seq_parameter_set_id = 0 → ue '1'
        bits += "1";
        // dependent_slice_segments_enabled_flag = 0
        bits += "0";
        // output_flag_present_flag = 0
        bits += "0";
        // num_extra_slice_header_bits = 0 → u(3)
        bits += "000";
        // sign_data_hiding_enabled_flag = 0
        bits += "0";
        // cabac_init_present_flag = 0
        bits += "0";
        // num_ref_idx_l0_default_active_minus1 = 0 → ue '1'
        bits += "1";
        // num_ref_idx_l1_default_active_minus1 = 0 → ue '1'
        bits += "1";
        // init_qp_minus26 = 0 → se '1'
        bits += "1";
        // constrained_intra_pred_flag = 0
        bits += "0";
        // transform_skip_enabled_flag = 0
        bits += "0";
        // cu_qp_delta_enabled_flag = 0
        bits += "0";
        // pps_cb_qp_offset = 0 → se '1'
        bits += "1";
        // pps_cr_qp_offset = 0 → se '1'
        bits += "1";
        // pps_slice_chroma_qp_offsets_present_flag = 0
        bits += "0";
        // weighted_pred_flag = 0
        bits += "0";
        // weighted_bipred_flag = 0
        bits += "0";
        // transquant_bypass_enabled_flag = 0
        bits += "0";
        // tiles_enabled_flag = 1
        bits += "1";
        // entropy_coding_sync_enabled_flag = 0
        bits += "0";
        //  tiles block:
        //   num_tile_columns_minus1 = 1 → ue codeNum 1 = '010'
        bits += "010";
        //   num_tile_rows_minus1 = 1 → ue '010'
        bits += "010";
        //   uniform_spacing_flag = 0
        bits += "0";
        //   column_width_minus1[0] = 2 → ue codeNum 2 = '011'
        bits += "011";
        //   row_height_minus1[0] = 3 → ue codeNum 3 = '00100'
        bits += "00100";
        //   loop_filter_across_tiles_enabled_flag = 0
        bits += "0";
        // pps_loop_filter_across_slices_enabled_flag = 1
        bits += "1";
        // deblocking_filter_control_present_flag = 1
        bits += "1";
        //   deblocking_filter_override_enabled_flag = 1
        bits += "1";
        //   pps_deblocking_filter_disabled_flag = 0
        bits += "0";
        //   pps_beta_offset_div2 = -2 → se codeNum 4 = '00101'
        bits += "00101";
        //   pps_tc_offset_div2 = 3 → se codeNum 5 = '00110'
        bits += "00110";
        // pps_scaling_list_data_present_flag = 0
        bits += "0";
        // lists_modification_present_flag = 0
        bits += "0";
        // log2_parallel_merge_level_minus2 = 0 → ue '1'
        bits += "1";
        // slice_segment_header_extension_present_flag = 0
        bits += "0";
        // pps_extension_present_flag = 0
        bits += "0";
        // rbsp_trailing_bits(): stop bit '1' then zero-pad to a byte.
        bits += "1";
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);

        let pps = PicParameterSet::parse(&rbsp).expect("tiles/deblock PPS parse");
        assert!(pps.tiles_enabled_flag);
        assert_eq!(pps.tiles.num_tile_columns_minus1, 1);
        assert_eq!(pps.tiles.num_tile_rows_minus1, 1);
        assert_eq!(pps.num_tile_columns(), 2);
        assert_eq!(pps.num_tile_rows(), 2);
        assert!(!pps.tiles.uniform_spacing_flag);
        assert_eq!(pps.tiles.column_width_minus1, vec![2]);
        assert_eq!(pps.tiles.row_height_minus1, vec![3]);
        assert!(!pps.loop_filter_across_tiles_enabled_flag);
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(pps.deblocking.override_enabled_flag);
        assert!(!pps.deblocking.disabled_flag);
        assert_eq!(pps.deblocking.beta_offset_div2, -2);
        assert_eq!(pps.deblocking.tc_offset_div2, 3);
    }

    /// `pps_extension_present_flag == 1` with every typed extension
    /// flag (and `pps_extension_4bits`) equal to 0 decodes the
    /// flag block but consumes only `rbsp_trailing_bits()` afterwards
    /// — no opaque tail is surfaced because no extension body
    /// follows.
    #[test]
    fn decodes_extension_flag_block_without_bodies() {
        let mut bits = minimal_pps_prefix_bits();
        // pps_extension_present_flag = 1
        bits += "1";
        bits += "0000"; // pps_range/multilayer/3d/scc flags = 0
        bits += "0000"; // pps_extension_4bits = 0
        bits += "1"; // rbsp stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("ext PPS parse");
        assert!(pps.pps_extension_present_flag);
        let flags = pps.extension_flags.expect("extension flag block");
        assert!(!flags.pps_range_extension_flag);
        assert!(!flags.pps_multilayer_extension_flag);
        assert!(!flags.pps_3d_extension_flag);
        assert!(!flags.pps_scc_extension_flag);
        assert_eq!(flags.pps_extension_4bits, 0);
        assert!(!flags.has_body());
        // No extension body follows → no opaque tail.
        assert!(pps.opaque_tail.is_none());
    }

    /// `pps_extension_present_flag == 1` with
    /// `pps_range_extension_flag == 1` decodes the flag block then the
    /// `pps_range_extension()` body (§7.3.2.3.2). The prefix has
    /// `transform_skip_enabled_flag == 0`, so the body opens directly
    /// with `cross_component_prediction_enabled_flag`. With no body
    /// after the range extension, no opaque tail is captured.
    #[test]
    fn decodes_pps_range_extension_body() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "1"; // pps_range_extension_flag = 1
        bits += "000"; // pps_multilayer/3d/scc = 0
        bits += "0000"; // pps_extension_4bits = 0
        // pps_range_extension() body (transform_skip off):
        bits += "1"; // cross_component_prediction_enabled_flag = 1
        bits += "0"; // chroma_qp_offset_list_enabled_flag = 0
        bits += "010"; // log2_sao_offset_scale_luma = 1 (ue)
        bits += "1"; // log2_sao_offset_scale_chroma = 0 (ue)
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("range-ext PPS parse");
        assert!(pps.pps_extension_present_flag);
        let flags = pps.extension_flags.expect("extension flag block");
        assert!(flags.pps_range_extension_flag);
        assert!(flags.has_body());
        // No body after the range extension → no opaque tail.
        assert!(pps.opaque_tail.is_none());
        let re = pps.pps_range_extension.expect("range extension body");
        assert_eq!(re.log2_max_transform_skip_block_size_minus2, 0);
        assert!(re.cross_component_prediction_enabled_flag);
        assert!(!re.chroma_qp_offset_list_enabled_flag);
        assert!(re.chroma_qp_offset_list.is_empty());
        assert_eq!(re.log2_sao_offset_scale_luma, 1);
        assert_eq!(re.log2_sao_offset_scale_chroma, 0);
    }

    /// `pps_range_extension()` with `chroma_qp_offset_list_enabled_flag
    /// == 1` decodes the `diff_cu_chroma_qp_offset_depth`,
    /// `chroma_qp_offset_list_len_minus1`, and the `cb/cr_qp_offset_list`
    /// se(v) pairs (§7.3.2.3.2).
    #[test]
    fn decodes_pps_range_extension_chroma_qp_offset_list() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "1"; // pps_range_extension_flag = 1
        bits += "0000000"; // multilayer/3d/scc + 4bits = 0
        // pps_range_extension() body (transform_skip off):
        bits += "0"; // cross_component_prediction_enabled_flag = 0
        bits += "1"; // chroma_qp_offset_list_enabled_flag = 1
        bits += "1"; // diff_cu_chroma_qp_offset_depth = 0 (ue)
        bits += "010"; // chroma_qp_offset_list_len_minus1 = 1 (ue) → 2 entries
        // entry 0: cb=+1 (se '010'), cr=-1 (se '011')
        bits += "010";
        bits += "011";
        // entry 1: cb=+2 (se '00100'), cr=0 (se '1')
        bits += "00100";
        bits += "1";
        bits += "1"; // log2_sao_offset_scale_luma = 0 (ue)
        bits += "1"; // log2_sao_offset_scale_chroma = 0 (ue)
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("range-ext PPS parse");
        let re = pps.pps_range_extension.expect("range extension body");
        assert!(re.chroma_qp_offset_list_enabled_flag);
        assert_eq!(re.diff_cu_chroma_qp_offset_depth, 0);
        assert_eq!(re.chroma_qp_offset_list_len_minus1, 1);
        assert_eq!(re.chroma_qp_offset_list.len(), 2);
        assert_eq!(re.chroma_qp_offset_list[0].cb_qp_offset, 1);
        assert_eq!(re.chroma_qp_offset_list[0].cr_qp_offset, -1);
        assert_eq!(re.chroma_qp_offset_list[1].cb_qp_offset, 2);
        assert_eq!(re.chroma_qp_offset_list[1].cr_qp_offset, 0);
    }

    /// `pps_scc_extension()` (§7.3.2.3.3) with no preceding
    /// range/multilayer/3D body: the body is decoded in place with no
    /// opaque tail. Adaptive colour transform off, palette initializers
    /// absent — only the leading flag and the two trailing flags.
    #[test]
    fn decodes_pps_scc_extension_minimal() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0"; // pps_range_extension_flag = 0
        bits += "0"; // pps_multilayer_extension_flag = 0
        bits += "0"; // pps_3d_extension_flag = 0
        bits += "1"; // pps_scc_extension_flag = 1
        bits += "0000"; // pps_extension_4bits = 0
        // pps_scc_extension():
        bits += "1"; // pps_curr_pic_ref_enabled_flag = 1
        bits += "0"; // residual_adaptive_colour_transform_enabled_flag = 0
        bits += "0"; // pps_palette_predictor_initializers_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("scc PPS parse");
        let flags = pps.extension_flags.expect("extension flag block");
        assert!(flags.pps_scc_extension_flag);
        assert!(pps.opaque_tail.is_none());
        let scc = pps.pps_scc_extension.expect("scc extension body");
        assert!(scc.pps_curr_pic_ref_enabled_flag);
        assert!(!scc.residual_adaptive_colour_transform_enabled_flag);
        assert!(!scc.pps_palette_predictor_initializers_present_flag);
        assert!(scc.pps_palette_predictor_initializer.is_empty());
    }

    /// `pps_scc_extension()` with
    /// `residual_adaptive_colour_transform_enabled_flag == 1` decodes the
    /// `pps_slice_act_qp_offsets_present_flag` and the three
    /// `pps_act_*_qp_offset_*` se(v) fields (§7.3.2.3.3).
    #[test]
    fn decodes_pps_scc_extension_act_offsets() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0001"; // range/multilayer/3d = 0, scc = 1
        bits += "0000"; // pps_extension_4bits = 0
        // pps_scc_extension():
        bits += "0"; // pps_curr_pic_ref_enabled_flag = 0
        bits += "1"; // residual_adaptive_colour_transform_enabled_flag = 1
        bits += "1"; // pps_slice_act_qp_offsets_present_flag = 1
        bits += "010"; // pps_act_y_qp_offset_plus5 = +1 (se '010')
        bits += "011"; // pps_act_cb_qp_offset_plus5 = -1 (se '011')
        bits += "00100"; // pps_act_cr_qp_offset_plus3 = +2 (se '00100')
        bits += "0"; // pps_palette_predictor_initializers_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("scc PPS parse");
        assert!(pps.opaque_tail.is_none());
        let scc = pps.pps_scc_extension.expect("scc extension body");
        assert!(!scc.pps_curr_pic_ref_enabled_flag);
        assert!(scc.residual_adaptive_colour_transform_enabled_flag);
        assert!(scc.pps_slice_act_qp_offsets_present_flag);
        assert_eq!(scc.pps_act_y_qp_offset_plus5, 1);
        assert_eq!(scc.pps_act_cb_qp_offset_plus5, -1);
        assert_eq!(scc.pps_act_cr_qp_offset_plus3, 2);
        // PpsActQpOffset{Y,Cb,Cr} = plus5−5 / plus5−5 / plus3−3.
        assert_eq!(scc.pps_act_qp_offset_y(), -4);
        assert_eq!(scc.pps_act_qp_offset_cb(), -6);
        assert_eq!(scc.pps_act_qp_offset_cr(), -1);
    }

    /// §7.4.3.3.3: a `PpsActQpOffsetY` (here from a large
    /// `pps_act_y_qp_offset_plus5`) outside −12..=12 violates bitstream
    /// conformance and is rejected.
    #[test]
    fn rejects_out_of_range_pps_act_qp_offset() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0001"; // range/multilayer/3d = 0, scc = 1
        bits += "0000"; // pps_extension_4bits = 0
        // pps_scc_extension():
        bits += "0"; // pps_curr_pic_ref_enabled_flag = 0
        bits += "1"; // residual_adaptive_colour_transform_enabled_flag = 1
        bits += "0"; // pps_slice_act_qp_offsets_present_flag = 0
        // pps_act_y_qp_offset_plus5 = +30 (se → codeNum 59,
        // ue '00000111100') → PpsActQpOffsetY = 25 > 12,
        // illegal
        bits += "00000111100";
        bits += "1"; // pps_act_cb_qp_offset_plus5 = 0 (se)
        bits += "1"; // pps_act_cr_qp_offset_plus3 = 0 (se)
        bits += "0"; // pps_palette_predictor_initializers_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let err = PicParameterSet::parse(&rbsp).expect_err("act qp out of range");
        assert!(matches!(
            err,
            PpsError::ValueOutOfRange {
                field: "pps_act_y_qp_offset_plus5",
                got: 25
            }
        ));
    }

    /// `pps_scc_extension()` with palette predictor initializers
    /// present: the per-component `pps_palette_predictor_initializer`
    /// table is sized `u(v)` from this body's own
    /// `luma_bit_depth_entry_minus8` / `chroma_bit_depth_entry_minus8`.
    /// `monochrome_palette_flag == 0` → `numComps == 3`.
    #[test]
    fn decodes_pps_scc_extension_palette_initializers() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0001"; // range/multilayer/3d = 0, scc = 1
        bits += "0000"; // pps_extension_4bits = 0
        // pps_scc_extension():
        bits += "0"; // pps_curr_pic_ref_enabled_flag = 0
        bits += "0"; // residual_adaptive_colour_transform_enabled_flag = 0
        bits += "1"; // pps_palette_predictor_initializers_present_flag = 1
        bits += "011"; // pps_num_palette_predictor_initializers = 2 (ue)
        bits += "0"; // monochrome_palette_flag = 0 → numComps = 3
        bits += "1"; // luma_bit_depth_entry_minus8 = 0 (ue) → 8-bit luma
        bits += "1"; // chroma_bit_depth_entry_minus8 = 0 (ue) → 8-bit chroma
        // comp 0 (8-bit): 0x0A, 0x0B
        bits += "00001010";
        bits += "00001011";
        // comp 1 (8-bit): 0x0C, 0x0D
        bits += "00001100";
        bits += "00001101";
        // comp 2 (8-bit): 0x0E, 0x0F
        bits += "00001110";
        bits += "00001111";
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("scc PPS parse");
        assert!(pps.opaque_tail.is_none());
        let scc = pps.pps_scc_extension.expect("scc extension body");
        assert!(scc.pps_palette_predictor_initializers_present_flag);
        assert_eq!(scc.pps_num_palette_predictor_initializers, 2);
        assert!(!scc.monochrome_palette_flag);
        assert_eq!(scc.luma_bit_depth_entry_minus8, 0);
        assert_eq!(scc.chroma_bit_depth_entry_minus8, 0);
        assert_eq!(
            scc.pps_palette_predictor_initializer,
            vec![vec![0x0A, 0x0B], vec![0x0C, 0x0D], vec![0x0E, 0x0F]],
        );
    }

    /// `pps_range_extension_flag == 1` followed by
    /// `pps_scc_extension_flag == 1` (no multilayer/3D between): both
    /// bodies are decoded in place per the §7.3.2.3.1 body order, with
    /// no opaque tail.
    #[test]
    fn decodes_pps_range_then_scc_body() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "1"; // pps_range_extension_flag = 1
        bits += "00"; // pps_multilayer/3d = 0
        bits += "1"; // pps_scc_extension_flag = 1
        bits += "0000"; // pps_extension_4bits = 0
        // pps_range_extension() body (transform_skip off):
        bits += "0"; // cross_component_prediction_enabled_flag = 0
        bits += "0"; // chroma_qp_offset_list_enabled_flag = 0
        bits += "1"; // log2_sao_offset_scale_luma = 0 (ue)
        bits += "1"; // log2_sao_offset_scale_chroma = 0 (ue)
        // pps_scc_extension():
        bits += "1"; // pps_curr_pic_ref_enabled_flag = 1
        bits += "0"; // residual_adaptive_colour_transform_enabled_flag = 0
        bits += "0"; // pps_palette_predictor_initializers_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("range+scc PPS parse");
        assert!(pps.opaque_tail.is_none());
        let re = pps.pps_range_extension.expect("range extension body");
        assert_eq!(re, PpsRangeExtension::default());
        let scc = pps.pps_scc_extension.expect("scc extension body");
        assert!(scc.pps_curr_pic_ref_enabled_flag);
    }

    /// When a `pps_multilayer_extension()` body precedes the SCC body,
    /// the SCC body cannot be decoded in place; the whole span stays in
    /// the opaque tail.
    #[test]
    fn pps_scc_stays_opaque_behind_multilayer_body() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0"; // pps_range_extension_flag = 0
        bits += "1"; // pps_multilayer_extension_flag = 1
        bits += "0"; // pps_3d_extension_flag = 0
        bits += "1"; // pps_scc_extension_flag = 1
        bits += "0000"; // pps_extension_4bits = 0
        bits += "11001100"; // opaque multilayer + scc span sentinel
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("multilayer PPS parse");
        let flags = pps.extension_flags.expect("extension flag block");
        assert!(flags.pps_multilayer_extension_flag);
        assert!(flags.pps_scc_extension_flag);
        assert!(pps.pps_scc_extension.is_none());
        assert!(pps.opaque_tail.is_some());
    }

    /// `chroma_qp_offset_list_len_minus1` above its §7.4.3.3.2 cap of 5
    /// is rejected.
    #[test]
    fn rejects_oversized_chroma_qp_offset_list_len() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "1"; // pps_range_extension_flag = 1
        bits += "0000000"; // multilayer/3d/scc + 4bits = 0
        bits += "0"; // cross_component_prediction_enabled_flag = 0
        bits += "1"; // chroma_qp_offset_list_enabled_flag = 1
        bits += "1"; // diff_cu_chroma_qp_offset_depth = 0 (ue)
        bits += "00111"; // chroma_qp_offset_list_len_minus1 = 6 (ue) → > 5
        bits += "1"; // padding to keep a well-formed buffer
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let err = PicParameterSet::parse(&rbsp).expect_err("len out of range");
        assert!(matches!(
            err,
            PpsError::ValueOutOfRange {
                field: "chroma_qp_offset_list_len_minus1",
                got: 6
            }
        ));
    }

    /// `pps_extension_present_flag == 1` with the four typed
    /// extension flags all 0 but `pps_extension_4bits != 0` still
    /// surfaces an opaque tail — the §7.3.2.3.1
    /// `while( more_rbsp_data() ) pps_extension_data_flag` block is
    /// gated by `pps_extension_4bits` (the §7.4.3.3.1 decoder rule is
    /// to ignore the data flags but they must be skipped past
    /// rbsp_trailing_bits, so the bytes are surfaced as opaque).
    #[test]
    fn captures_extension_data_flag_tail_when_4bits_nonzero() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "1"; // pps_extension_present_flag = 1
        bits += "0000"; // pps_range/multilayer/3d/scc = 0
        bits += "0001"; // pps_extension_4bits = 1 (reserved, non-zero)
        bits += "0"; // a single pps_extension_data_flag value
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("ext-4bits PPS parse");
        let flags = pps.extension_flags.expect("extension flag block");
        assert_eq!(flags.pps_extension_4bits, 1);
        assert!(flags.has_body());
        assert!(pps.opaque_tail.is_some());
    }

    /// When `pps_extension_present_flag == 0` the typed
    /// extension-flag block is absent and every flag is inferred to
    /// 0 per §7.4.3.3.1.
    #[test]
    fn extension_flags_absent_when_gate_zero() {
        let mut bits = minimal_pps_prefix_bits();
        bits += "0"; // pps_extension_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("no-ext PPS parse");
        assert!(!pps.pps_extension_present_flag);
        assert!(pps.extension_flags.is_none());
        assert!(pps.opaque_tail.is_none());
    }

    /// `pps_scaling_list_data_present_flag == 1` parses an explicit
    /// `scaling_list_data()` (§7.3.4). Here every one of the 24 slots
    /// signals pred_mode=0, delta=0 (use the §7.4.5 default list), so
    /// the parsed lists equal the default tables and the PPS continues
    /// through the rest of the body.
    #[test]
    fn parses_scaling_list_present() {
        let mut bits = minimal_pps_prefix_bits_until_scaling_list();
        bits += "1"; // pps_scaling_list_data_present_flag = 1
        // scaling_list_data(): 24 slots, each '0' (pred_mode
        // = 0) + '1' (scaling_list_pred_matrix_id_delta ue
        // = 0). sizeId 0/1/2 each 6 slots; sizeId 3 only 2.
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                bits += "0";
                bits += "1";
                m += step;
            }
        }
        // Remainder of the PPS body.
        bits += "0"; // lists_modification_present_flag
        bits += "1"; // log2_parallel_merge_level_minus2 ue = 0
        bits += "0"; // slice_segment_header_extension_present_flag
        bits += "0"; // pps_extension_present_flag
        bits += "1"; // rbsp stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("scaling-list PPS parse");
        assert!(pps.pps_scaling_list_data_present_flag);
        let data = pps.scaling_list_data.expect("scaling_list_data present");
        assert_eq!(data.lists[0][0].coef, vec![16u16; 16]);
        assert_eq!(data.lists[1][0].coef[63], 115); // 8x8 intra default tail
    }

    /// `pps_pic_parameter_set_id` out of range (> 63) is rejected.
    #[test]
    fn rejects_pps_id_out_of_range() {
        // ue codeNum 64 = leadingZeroBits 6, suffix = 64+1-(2^6-1)=1 →
        // '0000001000001'. codeNum = 2^6 - 1 + 1 = 64 (> 63).
        let mut bits = String::from("0000001000001");
        bits += "1"; // pad to ensure non-empty buffer for any further read
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        assert_eq!(
            PicParameterSet::parse(&rbsp),
            Err(PpsError::ValueOutOfRange {
                field: "pps_pic_parameter_set_id",
                got: 64,
            })
        );
    }

    /// A truncated PPS RBSP surfaces `Truncated`.
    #[test]
    fn rejects_truncated() {
        // Just the first two ue() ids then end-of-buffer mid-flag-block.
        let rbsp = [0xC0]; // pps_id=0 ('1'), sps_id=0 ('1'), then '000000'
        let err = PicParameterSet::parse(&rbsp).unwrap_err();
        assert_eq!(err, PpsError::Truncated);
    }

    #[test]
    fn init_qp_range_check_uses_sps_bit_depth() {
        // init_qp_minus26 = -70 is within the loosest -74 bound the
        // parser accepts but only legal for high bit depths (where
        // QpBdOffsetY is large enough). Build a complete PPS carrying
        // -70 and exercise the SPS-aware range helper on the result.
        let mut bits = String::new();
        bits += "1"; // pps_id = 0
        bits += "1"; // sps_id = 0
        bits += "0"; // dependent_slice_segments_enabled_flag
        bits += "0"; // output_flag_present_flag
        bits += "000"; // num_extra_slice_header_bits
        bits += "0"; // sign_data_hiding_enabled_flag
        bits += "0"; // cabac_init_present_flag
        bits += "1"; // num_ref_idx_l0_default_active_minus1 = 0
        bits += "1"; // num_ref_idx_l1_default_active_minus1 = 0
        // init_qp_minus26 = -70 → se value -70: codeNum = 2*70 = 140.
        // ue(140): leadingZeroBits = 7, suffix = 140+1-128 = 13.
        bits += "0000000"; // 7 leading zeros
        bits += "1"; // terminator
        bits += "0001101"; // 7-bit suffix = 13 → codeNum = 127 + 13 = 140
        bits += "0"; // constrained_intra_pred_flag
        bits += "0"; // transform_skip_enabled_flag
        bits += "0"; // cu_qp_delta_enabled_flag = 0
        bits += "1"; // pps_cb_qp_offset = 0 (se '1')
        bits += "1"; // pps_cr_qp_offset = 0 (se '1')
        bits += "0"; // pps_slice_chroma_qp_offsets_present_flag
        bits += "0"; // weighted_pred_flag
        bits += "0"; // weighted_bipred_flag
        bits += "0"; // transquant_bypass_enabled_flag
        bits += "0"; // tiles_enabled_flag = 0
        bits += "0"; // entropy_coding_sync_enabled_flag
        bits += "1"; // pps_loop_filter_across_slices_enabled_flag
        bits += "0"; // deblocking_filter_control_present_flag = 0
        bits += "0"; // pps_scaling_list_data_present_flag = 0
        bits += "0"; // lists_modification_present_flag
        bits += "1"; // log2_parallel_merge_level_minus2 = 0 (ue '1')
        bits += "0"; // slice_segment_header_extension_present_flag
        bits += "0"; // pps_extension_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("low-init-qp PPS parse");
        assert_eq!(pps.init_qp_minus26, -70);
        assert_eq!(pps.init_qp(), -44);
        // 8-bit (minus8=0): QpBdOffsetY=0, lower bound -26 → out of range.
        assert!(!pps.init_qp_in_range(0));
        // 12-bit (minus8=4): QpBdOffsetY=24, lower bound -50 → out.
        assert!(!pps.init_qp_in_range(4));
        // 16-bit (minus8=8): QpBdOffsetY=48, lower bound -74 → in range.
        assert!(pps.init_qp_in_range(8));
    }

    /// Fuzz regression (r282 hardening): §A.4.1 item f) bounds the
    /// tile grid (`num_tile_columns_minus1 < MaxTileCols`, at most 40
    /// per Table A.8). An unbounded `ue(v)` count previously drove the
    /// explicit column-width array pre-allocation directly off the
    /// wire value.
    #[test]
    fn rejects_oversized_tile_column_count() {
        let mut bits = minimal_pps_prefix_bits_until_scaling_list();
        // The helper's last four bits are `tiles_enabled_flag = 0`,
        // `entropy_coding_sync = 0`, `loop_filter_across_slices = 1`,
        // `deblocking_filter_control_present = 0` — rewrite the tail
        // with tiles enabled and an out-of-range column count (the
        // parse rejects before reading anything further).
        bits.truncate(bits.len() - 4);
        bits += "1"; // tiles_enabled_flag = 1
        bits += "0"; // entropy_coding_sync_enabled_flag = 0
        bits += "00000101001"; // num_tile_columns_minus1 ue = 40
        bits += "1"; // num_tile_rows_minus1 = 0
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let err = PicParameterSet::parse(&rbsp).unwrap_err();
        assert_eq!(
            err,
            PpsError::ValueOutOfRange {
                field: "num_tile_columns_minus1",
                got: 40
            }
        );
    }

    // --- test helpers --------------------------------------------------

    /// Pack an MSB-first bit string (must be a whole number of bytes)
    /// into a byte vector.
    fn pack_bits(bits: &str) -> Vec<u8> {
        assert_eq!(bits.len() % 8, 0, "bit string must be byte-aligned");
        bits.as_bytes()
            .chunks(8)
            .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | (b - b'0')))
            .collect()
    }

    /// Minimal PPS prefix bits up to (but not including)
    /// `pps_extension_present_flag`, with every field set to the
    /// simplest legal value (no tiles, no deblocking control, no
    /// scaling list). The caller appends the extension flag + tail.
    fn minimal_pps_prefix_bits() -> String {
        let mut bits = minimal_pps_prefix_bits_until_scaling_list();
        // pps_scaling_list_data_present_flag = 0
        bits += "0";
        // lists_modification_present_flag = 0
        bits += "0";
        // log2_parallel_merge_level_minus2 = 0 → ue '1'
        bits += "1";
        // slice_segment_header_extension_present_flag = 0
        bits += "0";
        bits
    }

    /// Minimal PPS prefix bits up to (but not including)
    /// `pps_scaling_list_data_present_flag`.
    fn minimal_pps_prefix_bits_until_scaling_list() -> String {
        let mut bits = String::new();
        bits += "1"; // pps_id = 0
        bits += "1"; // sps_id = 0
        bits += "0"; // dependent_slice_segments_enabled_flag
        bits += "0"; // output_flag_present_flag
        bits += "000"; // num_extra_slice_header_bits
        bits += "0"; // sign_data_hiding_enabled_flag
        bits += "0"; // cabac_init_present_flag
        bits += "1"; // num_ref_idx_l0_default_active_minus1 = 0
        bits += "1"; // num_ref_idx_l1_default_active_minus1 = 0
        bits += "1"; // init_qp_minus26 = 0 (se '1')
        bits += "0"; // constrained_intra_pred_flag
        bits += "0"; // transform_skip_enabled_flag
        bits += "0"; // cu_qp_delta_enabled_flag = 0 (depth not signalled)
        bits += "1"; // pps_cb_qp_offset = 0 (se '1')
        bits += "1"; // pps_cr_qp_offset = 0 (se '1')
        bits += "0"; // pps_slice_chroma_qp_offsets_present_flag
        bits += "0"; // weighted_pred_flag
        bits += "0"; // weighted_bipred_flag
        bits += "0"; // transquant_bypass_enabled_flag
        bits += "0"; // tiles_enabled_flag = 0
        bits += "0"; // entropy_coding_sync_enabled_flag
        bits += "1"; // pps_loop_filter_across_slices_enabled_flag
        bits += "0"; // deblocking_filter_control_present_flag = 0
        bits
    }
}
