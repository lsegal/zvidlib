//! Video Parameter Set (VPS) parser per ITU-T Rec. H.265 §7.3.2.1.
//!
//! Parses the VPS RBSP through the layer-set inclusion matrix and the
//! VPS timing-info block (including `vps_num_units_in_tick` /
//! `vps_time_scale` / `vps_poc_proportional_to_timing_flag` /
//! `vps_num_ticks_poc_diff_one_minus1` and the `vps_num_hrd_parameters`
//! count). The per-HRD `hrd_parameters()` bodies (§E.2.2) are decoded
//! as a vector of [`crate::hevc::engine::hrd::VpsHrdEntry`] values (one per
//! `vps_num_hrd_parameters`), with the §E.2.3 sub-layer HRD payloads
//! folded into each entry's [`crate::hevc::engine::hrd::SubLayerHrd`]. The
//! `vps_extension_flag` follows the HRD loop; when 1, the
//! `vps_extension_data_flag` run + `rbsp_trailing_bits()` are surfaced
//! as an [`crate::hevc::engine::sps::OpaqueTail`] for callers that want the raw
//! bytes.
//!
//! The profile-tier-level subroutine of §7.3.3 is also parsed
//! structurally: the bit positions are walked but only the leading
//! `general_profile_space` / `general_tier_flag` / `general_profile_idc`
//! / `general_level_idc` fields and the per-sub-layer
//! `sub_layer_profile_present_flag` / `sub_layer_level_present_flag`
//! gates are materialised. The remaining (mostly-reserved-zero or
//! constraint-flag) fields are skipped, but the bit-walk advances the
//! reader correctly so subsequent VPS fields land on the right bit
//! boundary.
//!
//! ## Layout summary
//!
//! ```text
//! vps_video_parameter_set_id            u(4)
//! vps_base_layer_internal_flag          u(1)
//! vps_base_layer_available_flag         u(1)
//! vps_max_layers_minus1                 u(6)
//! vps_max_sub_layers_minus1             u(3)
//! vps_temporal_id_nesting_flag          u(1)
//! vps_reserved_0xffff_16bits            u(16)   /* must be 0xFFFF */
//! profile_tier_level( 1, vps_max_sub_layers_minus1 )
//! vps_sub_layer_ordering_info_present_flag  u(1)
//! for( i = (...) ; i <= vps_max_sub_layers_minus1; i++ ) {
//!   vps_max_dec_pic_buffering_minus1[i] ue(v)
//!   vps_max_num_reorder_pics[i]         ue(v)
//!   vps_max_latency_increase_plus1[i]   ue(v)
//! }
//! vps_max_layer_id                       u(6)
//! vps_num_layer_sets_minus1             ue(v)
//! for( i = 1; i <= vps_num_layer_sets_minus1; i++ )
//!   for( j = 0; j <= vps_max_layer_id; j++ )
//!     layer_id_included_flag[i][j]       u(1)
//! vps_timing_info_present_flag           u(1)
//! if( vps_timing_info_present_flag ) {
//!   vps_num_units_in_tick               u(32)
//!   vps_time_scale                      u(32)
//!   vps_poc_proportional_to_timing_flag  u(1)
//!   if( vps_poc_proportional_to_timing_flag )
//!     vps_num_ticks_poc_diff_one_minus1 ue(v)
//!   vps_num_hrd_parameters              ue(v)
//!   for( i = 0; i < vps_num_hrd_parameters; i++ ) {
//!     hrd_layer_set_idx[i]              ue(v)
//!     if( i > 0 ) cprms_present_flag[i] u(1)
//!     hrd_parameters( cprms_present_flag[i], vps_max_sub_layers_minus1 )  /* §E.2.2 */
//!   }
//! }
//! vps_extension_flag                     u(1)
//! /* extension payload + rbsp_trailing_bits() surfaced as opaque */
//! ```

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::hrd::{HrdError, VpsHrdEntry};
use crate::hevc::engine::sps::OpaqueTail;

/// Maximum number of sub-layers an HEVC stream may declare.
/// `vps_max_sub_layers_minus1` is u(3), so the count is bounded at 7
/// in the bitstream; this constant is reused by [`HevcVps`] as the
/// fixed-size capacity for per-sub-layer arrays.
pub const HEVC_MAX_SUB_LAYERS: usize = 7;

/// Maximum number of layer IDs the VPS layer-set inclusion matrix may
/// span. `vps_max_layer_id` is u(6), so the maximum signalled value is
/// 63; the inclusion matrix column count is `vps_max_layer_id + 1`,
/// which is bounded at 64.
pub const HEVC_VPS_MAX_NUM_LAYERS: usize = 64;

/// Upper bound on `vps_num_layer_sets_minus1`. Per §7.4.3.1
/// `vps_num_layer_sets_minus1` shall be in the range 0..=1023, so
/// the layer-set count is bounded at 1024. This crate's parser caps
/// the value here to keep an aberrantly-encoded stream from forcing a
/// 4 MB allocation (the legal max would already be ~64 KB).
pub const HEVC_VPS_MAX_NUM_LAYER_SETS: usize = 1024;

/// Errors that can arise while parsing a VPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VpsError {
    /// The RBSP ran out of bits before the VPS was fully parsed.
    Truncated,
    /// `vps_reserved_0xffff_16bits` was not `0xFFFF`. §7.4.3.1
    /// mandates the literal value.
    ReservedFieldMismatch {
        /// The (incorrect) value that was actually read.
        got: u16,
    },
    /// An Exp-Golomb code's `codeNum` exceeded what the corresponding
    /// syntax element can legally hold.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value.
        got: u32,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
    /// An `hrd_parameters()` body inside the VPS HRD loop was malformed.
    /// Propagated up from [`crate::hevc::engine::hrd::HrdError`] so a caller that
    /// only looks at [`VpsError`] still sees the failure.
    Hrd(HrdError),
}

impl core::fmt::Display for VpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("VPS RBSP truncated"),
            Self::ReservedFieldMismatch { got } => write!(
                f,
                "vps_reserved_0xffff_16bits was 0x{got:04X}, expected 0xFFFF"
            ),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "syntax element {field} out of range: {got}")
            }
            Self::Bitstream(e) => write!(f, "bitstream error during VPS parse: {e}"),
            Self::Hrd(e) => write!(f, "hrd_parameters() error inside VPS: {e}"),
        }
    }
}

impl std::error::Error for VpsError {}

impl From<BitReaderError> for VpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<HrdError> for VpsError {
    fn from(e: HrdError) -> Self {
        // Surface a bitstream-truncation reported through the HRD path
        // as the VPS-level Truncated variant so callers can keep their
        // single-pattern truncation handler. Any other HRD failure mode
        // is opaque to the VPS — preserve it verbatim.
        match e {
            HrdError::Truncated => Self::Truncated,
            other => Self::Hrd(other),
        }
    }
}

/// Parsed profile-tier-level structure (§7.3.3).
///
/// Only the leading "general" fields and the per-sub-layer
/// present-flag gates are materialised at round-2 scope. The
/// constraint flags / reserved-zero blocks are walked over to keep
/// bit alignment but their values are intentionally discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileTierLevel {
    /// `general_profile_space` (`u(2)`).
    pub general_profile_space: u8,
    /// `general_tier_flag` (`u(1)`).
    pub general_tier_flag: bool,
    /// `general_profile_idc` (`u(5)`). Profile mnemonics are listed in
    /// Annex A; for the Main / Main 10 / Main Still / Main 4:2:2 family
    /// values 1..=4 are most common.
    pub general_profile_idc: u8,
    /// `general_level_idc` (`u(8)`). Per A.4 the on-wire value is
    /// `30 × level_number`, e.g. 30 == level 1.0, 90 == level 3.0,
    /// 120 == level 4.0.
    pub general_level_idc: u8,
    /// For each present sub-layer, whether its profile entry was
    /// signalled (`sub_layer_profile_present_flag[i]`).
    pub sub_layer_profile_present: [bool; HEVC_MAX_SUB_LAYERS],
    /// For each present sub-layer, whether its level_idc was signalled
    /// (`sub_layer_level_present_flag[i]`).
    pub sub_layer_level_present: [bool; HEVC_MAX_SUB_LAYERS],
    /// Per-sub-layer `sub_layer_level_idc[i]`; only valid for indices
    /// `i` where `sub_layer_level_present[i]` is true.
    pub sub_layer_level_idc: [u8; HEVC_MAX_SUB_LAYERS],
}

impl ProfileTierLevel {
    /// Parse a `profile_tier_level(profilePresentFlag, maxNumSubLayersMinus1)`
    /// invocation per §7.3.3. `profile_present_flag` is supplied by
    /// the calling context — for the VPS / SPS path it is always 1.
    pub fn parse(
        br: &mut BitReader<'_>,
        profile_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<Self, VpsError> {
        let mut ptl = Self {
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 0,
            general_level_idc: 0,
            sub_layer_profile_present: [false; HEVC_MAX_SUB_LAYERS],
            sub_layer_level_present: [false; HEVC_MAX_SUB_LAYERS],
            sub_layer_level_idc: [0; HEVC_MAX_SUB_LAYERS],
        };

        if profile_present_flag {
            ptl.general_profile_space = br.u(2)? as u8;
            ptl.general_tier_flag = br.u1()? != 0;
            ptl.general_profile_idc = br.u(5)? as u8;
            // 32 compatibility flags — skipped wholesale; the calling
            // application can re-parse them from the bit position if
            // needed later.
            br.skip(32)?;
            // progressive / interlaced / non_packed / frame_only
            br.skip(4)?;
            // The conditional block beneath these flags always consumes
            // exactly 43 bits regardless of profile_idc (per the
            // `/* not affected by this condition */` comment in §7.3.3
            // — the chroma-constraint, range-extension, and reserved
            // alternatives all sum to 43 bits).
            br.skip(43)?;
            // general_inbld_flag OR general_reserved_zero_bit — always 1 bit.
            br.skip(1)?;
        }
        // general_level_idc is always present (no `profilePresentFlag`
        // guard in the §7.3.3 syntax).
        ptl.general_level_idc = br.u(8)? as u8;

        // Per-sub-layer present-flag gates: 2 bits per sublayer up to
        // (but excluding) maxNumSubLayersMinus1.
        let max = max_num_sub_layers_minus1 as usize;
        for i in 0..max {
            let prof = br.u1()? != 0;
            let lvl = br.u1()? != 0;
            ptl.sub_layer_profile_present[i] = prof;
            ptl.sub_layer_level_present[i] = lvl;
        }

        // §7.3.3: if maxNumSubLayersMinus1 > 0, then for i in
        // max..8: reserved_zero_2bits — exactly 2 bits each — to keep
        // the per-sub-layer body byte-aligned regardless of how many
        // sublayers were actually signalled.
        if max_num_sub_layers_minus1 > 0 {
            for _ in max..8 {
                br.skip(2)?;
            }
        }

        // Per-sub-layer profile/level body for each i in 0..max.
        for i in 0..max {
            if ptl.sub_layer_profile_present[i] {
                // 2 + 1 + 5 + 32 + 4 + 43 + 1 = 88 bits, identical
                // layout to the general profile block above.
                br.skip(88)?;
            }
            if ptl.sub_layer_level_present[i] {
                ptl.sub_layer_level_idc[i] = br.u(8)? as u8;
            }
        }

        Ok(ptl)
    }
}

/// One per-sub-layer ordering-info triple from §7.3.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubLayerOrderingInfo {
    /// `vps_max_dec_pic_buffering_minus1[i]` (`ue(v)`).
    /// Implies a DPB size of `value + 1`.
    pub max_dec_pic_buffering_minus1: u32,
    /// `vps_max_num_reorder_pics[i]` (`ue(v)`).
    pub max_num_reorder_pics: u32,
    /// `vps_max_latency_increase_plus1[i]` (`ue(v)`).
    /// 0 disables the constraint.
    pub max_latency_increase_plus1: u32,
}

/// One per-layer-set row of the §7.3.2.1
/// `layer_id_included_flag[i][j]` matrix. `flags[j]` is the
/// `layer_id_included_flag[i][j]` value for layer-set `i` and
/// `nuh_layer_id == j`, with `0 <= j <= vps_max_layer_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerIdInclusionRow {
    /// `layer_id_included_flag[i][0..=vps_max_layer_id]`.
    pub flags: Vec<bool>,
}

/// VPS timing-info block per §7.3.2.1, when
/// `vps_timing_info_present_flag == 1`. The `hrd_parameters()` bodies
/// indexed by `vps_num_hrd_parameters` are decoded as
/// [`HevcVps::hrd_parameters`] entries; consult
/// [`Self::num_hrd_parameters`] for the count and
/// [`HevcVps::hrd_parameters`] for the bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VpsTimingInfo {
    /// `vps_num_units_in_tick` (`u(32)`). Spec constraint: shall be
    /// > 0.
    pub num_units_in_tick: u32,
    /// `vps_time_scale` (`u(32)`). Spec constraint: shall be > 0.
    pub time_scale: u32,
    /// `vps_poc_proportional_to_timing_flag` (`u(1)`).
    pub poc_proportional_to_timing_flag: bool,
    /// `vps_num_ticks_poc_diff_one_minus1` (`ue(v)`), only present
    /// when `poc_proportional_to_timing_flag` is set. Spec range
    /// 0..=2^32 - 2 (the `ue(v)` codec ceiling).
    pub num_ticks_poc_diff_one_minus1: Option<u32>,
    /// `vps_num_hrd_parameters` (`ue(v)`). The corresponding
    /// `hrd_parameters()` bodies are decoded into
    /// [`HevcVps::hrd_parameters`]. Spec constraint:
    /// 0..=`vps_num_layer_sets_minus1 + 1`.
    pub num_hrd_parameters: u32,
}

/// Parsed Video Parameter Set per §7.3.2.1.
///
/// The structural prefix (through the per-sub-layer ordering loop) is
/// fully materialised; the layer-set inclusion matrix and the
/// optional VPS timing-info block follow. When
/// `vps_timing_info_present_flag == 1` and `vps_num_hrd_parameters >
/// 0`, the per-HRD `hrd_parameters()` payloads are decoded into
/// [`Self::hrd_parameters`]. The parser then reads `vps_extension_flag`
/// and, when set, surfaces the `vps_extension_data_flag` payload (plus
/// `rbsp_trailing_bits()`) as [`Self::opaque_tail`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcVps {
    /// `vps_video_parameter_set_id` (4 bits, range 0..=15).
    pub vps_id: u8,
    /// `vps_base_layer_internal_flag`.
    pub base_layer_internal_flag: bool,
    /// `vps_base_layer_available_flag`.
    pub base_layer_available_flag: bool,
    /// `vps_max_layers_minus1` (u(6)). The maximum number of layers is
    /// `value + 1`; for single-layer (non-SHVC) streams this is 0.
    pub max_layers_minus1: u8,
    /// `vps_max_sub_layers_minus1` (u(3), range 0..=6). The number of
    /// temporal sub-layers is `value + 1`.
    pub max_sub_layers_minus1: u8,
    /// `vps_temporal_id_nesting_flag`. When the stream has only one
    /// sub-layer the spec requires this to be 1.
    pub temporal_id_nesting_flag: bool,
    /// Parsed `profile_tier_level()` subroutine.
    pub ptl: ProfileTierLevel,
    /// `vps_sub_layer_ordering_info_present_flag`. When 0, only entry
    /// `[max_sub_layers_minus1]` is signalled and the others inherit
    /// its value.
    pub sub_layer_ordering_info_present_flag: bool,
    /// Per-sub-layer DPB / reorder / latency triples. Indices outside
    /// `0..=max_sub_layers_minus1` are zero-initialised.
    pub sub_layer_ordering_info: [SubLayerOrderingInfo; HEVC_MAX_SUB_LAYERS],
    /// `vps_max_layer_id` (`u(6)`, range 0..=62). The inclusion-matrix
    /// column count is `value + 1`.
    pub max_layer_id: u8,
    /// `vps_num_layer_sets_minus1` (`ue(v)`, range 0..=1023). The
    /// number of layer sets signalled by the inclusion matrix is
    /// `value + 1`; layer set 0 is implicit and not signalled in the
    /// matrix.
    pub num_layer_sets_minus1: u16,
    /// `layer_id_included_flag[i][j]` matrix. The outer index is
    /// `i = 1..=num_layer_sets_minus1`, stored at offset `i - 1`
    /// (layer set 0 is implicit per §7.4.3.1). Each inner row has
    /// length `max_layer_id + 1`.
    pub layer_id_included_flag: Vec<LayerIdInclusionRow>,
    /// `vps_timing_info_present_flag`. Discriminates whether
    /// [`Self::timing_info`] is populated.
    pub timing_info_present_flag: bool,
    /// Parsed [`VpsTimingInfo`] when [`Self::timing_info_present_flag`]
    /// is set; `None` otherwise.
    pub timing_info: Option<VpsTimingInfo>,
    /// Per-`hrd_parameters()` entries from the §7.3.2.1 loop, one per
    /// `vps_num_hrd_parameters`. Length equals
    /// `timing_info.num_hrd_parameters` when timing info is present,
    /// and 0 otherwise.
    pub hrd_parameters: Vec<VpsHrdEntry>,
    /// `vps_extension_flag`. Always populated now that the per-HRD
    /// bodies are decoded inline; the parser unconditionally reads this
    /// `u(1)` after the HRD loop completes.
    pub vps_extension_flag: bool,
    /// Opaque suffix of the RBSP. Populated when `vps_extension_flag ==
    /// 1`: the `vps_extension_data_flag` payload and the
    /// `rbsp_trailing_bits()` are surfaced here for callers that want
    /// the raw bytes. `None` otherwise.
    pub opaque_tail: Option<OpaqueTail>,
}

impl HevcVps {
    /// Parse `video_parameter_set_rbsp()` starting from the first bit
    /// of the (already-unescaped) RBSP body — that is, *after* the
    /// two-byte NAL header has been removed (see
    /// [`crate::hevc::engine::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, VpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br, rbsp)
    }

    fn parse_inner(br: &mut BitReader<'_>, rbsp: &[u8]) -> Result<Self, VpsError> {
        let vps_id = br.u(4)? as u8;
        let base_layer_internal_flag = br.u1()? != 0;
        let base_layer_available_flag = br.u1()? != 0;
        let max_layers_minus1 = br.u(6)? as u8;
        let max_sub_layers_minus1_raw = br.u(3)? as u8;
        if max_sub_layers_minus1_raw > 6 {
            // §7.4.3.1: range is 0..=6 (max 7 sub-layers); a 7 here is
            // illegal even though u(3) can encode it.
            return Err(VpsError::ValueOutOfRange {
                field: "vps_max_sub_layers_minus1",
                got: max_sub_layers_minus1_raw as u32,
            });
        }
        let temporal_id_nesting_flag = br.u1()? != 0;
        let reserved = br.u(16)? as u16;
        if reserved != 0xFFFF {
            return Err(VpsError::ReservedFieldMismatch { got: reserved });
        }

        let ptl = ProfileTierLevel::parse(br, true, max_sub_layers_minus1_raw)?;

        let sub_layer_ordering_info_present_flag = br.u1()? != 0;
        let start = if sub_layer_ordering_info_present_flag {
            0usize
        } else {
            max_sub_layers_minus1_raw as usize
        };
        let mut sub_layer_ordering_info = [SubLayerOrderingInfo::default(); HEVC_MAX_SUB_LAYERS];
        let last = max_sub_layers_minus1_raw as usize;
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
        // When the present flag was 0, propagate the
        // [max_sub_layers_minus1] entry across the lower-indexed
        // sub-layers — §7.4.3.1 dictates that they take the same value.
        if !sub_layer_ordering_info_present_flag {
            let copy = sub_layer_ordering_info[last];
            for entry in sub_layer_ordering_info.iter_mut().take(last) {
                *entry = copy;
            }
        }

        // vps_max_layer_id u(6) — range 0..=62 per §7.4.3.1. u(6) can
        // encode 63, which the spec marks as reserved; we accept the
        // value but the count `max_layer_id + 1` is capped at 64
        // (HEVC_VPS_MAX_NUM_LAYERS).
        let max_layer_id = br.u(6)? as u8;

        // vps_num_layer_sets_minus1 ue(v) — range 0..=1023.
        let num_layer_sets_minus1_raw = br.ue()?;
        if num_layer_sets_minus1_raw as usize >= HEVC_VPS_MAX_NUM_LAYER_SETS {
            return Err(VpsError::ValueOutOfRange {
                field: "vps_num_layer_sets_minus1",
                got: num_layer_sets_minus1_raw,
            });
        }
        let num_layer_sets_minus1 = num_layer_sets_minus1_raw as u16;

        // Layer-set inclusion matrix. The for-loop in the spec starts
        // at i = 1 (layer set 0 is the base set, not signalled), so the
        // signalled-row count is `num_layer_sets_minus1`. Each row has
        // `max_layer_id + 1` u(1) flags.
        let row_width = max_layer_id as usize + 1;
        let signalled_rows = num_layer_sets_minus1 as usize;
        let mut layer_id_included_flag = Vec::with_capacity(signalled_rows);
        for _ in 0..signalled_rows {
            let mut row = Vec::with_capacity(row_width);
            for _ in 0..row_width {
                row.push(br.u1()? != 0);
            }
            layer_id_included_flag.push(LayerIdInclusionRow { flags: row });
        }

        // vps_timing_info_present_flag u(1).
        let timing_info_present_flag = br.u1()? != 0;
        let mut opaque_tail = None;
        let mut hrd_parameters: Vec<VpsHrdEntry> = Vec::new();
        let timing_info = if timing_info_present_flag {
            // §E.2.1 / §7.3.2.1: num_units_in_tick and time_scale are
            // both u(32) and "shall be greater than 0" per the
            // semantics text — enforced here so a zeroed-out stream
            // doesn't silently divide by zero downstream.
            let num_units_in_tick = br.u(32)?;
            if num_units_in_tick == 0 {
                return Err(VpsError::ValueOutOfRange {
                    field: "vps_num_units_in_tick",
                    got: 0,
                });
            }
            let time_scale = br.u(32)?;
            if time_scale == 0 {
                return Err(VpsError::ValueOutOfRange {
                    field: "vps_time_scale",
                    got: 0,
                });
            }
            let poc_proportional_to_timing_flag = br.u1()? != 0;
            let num_ticks_poc_diff_one_minus1 = if poc_proportional_to_timing_flag {
                Some(br.ue()?)
            } else {
                None
            };
            let num_hrd_parameters = br.ue()?;
            // Spec range: 0..=vps_num_layer_sets_minus1 + 1. Bound the
            // value as a sanity check so a malformed stream cannot
            // force an unbounded loop downstream.
            if num_hrd_parameters > num_layer_sets_minus1 as u32 + 1 {
                return Err(VpsError::ValueOutOfRange {
                    field: "vps_num_hrd_parameters",
                    got: num_hrd_parameters,
                });
            }
            let info = VpsTimingInfo {
                num_units_in_tick,
                time_scale,
                poc_proportional_to_timing_flag,
                num_ticks_poc_diff_one_minus1,
                num_hrd_parameters,
            };
            // Decode the per-HRD entries inline per §7.3.2.1:
            //
            //   for( i = 0; i < vps_num_hrd_parameters; i++ ) {
            //     hrd_layer_set_idx[ i ]              ue(v)
            //     if( i > 0 ) cprms_present_flag[ i ]  u(1)
            //     hrd_parameters( cprms_present_flag[ i ], vps_max_sub_layers_minus1 )
            //   }
            //
            // The per-HRD body (§E.2.2) lives in the `crate::hevc::engine::hrd`
            // module — the `cprms_present_flag` inheritance chain is
            // walked here so each entry sees the previous entry's
            // common-info gates when its own flag is 0.
            for i in 0..num_hrd_parameters {
                let prev = hrd_parameters.last();
                let entry = VpsHrdEntry::parse(br, i, max_sub_layers_minus1_raw, prev)?;
                hrd_parameters.push(entry);
            }
            Some(info)
        } else {
            None
        };

        // vps_extension_flag u(1) — always read now that the per-HRD
        // bodies are decoded inline above. When set, the
        // vps_extension_data_flag run plus rbsp_trailing_bits() are
        // surfaced as the opaque tail (this parser does not interpret
        // the extension payload).
        let vps_extension_flag = br.u1()? != 0;
        if vps_extension_flag {
            opaque_tail = Some(OpaqueTail::capture_at(br.bit_pos(), rbsp));
        }

        Ok(Self {
            vps_id,
            base_layer_internal_flag,
            base_layer_available_flag,
            max_layers_minus1,
            max_sub_layers_minus1: max_sub_layers_minus1_raw,
            temporal_id_nesting_flag,
            ptl,
            sub_layer_ordering_info_present_flag,
            sub_layer_ordering_info,
            max_layer_id,
            num_layer_sets_minus1,
            layer_id_included_flag,
            timing_info_present_flag,
            timing_info,
            hrd_parameters,
            vps_extension_flag,
            opaque_tail,
        })
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::nal::{collect_nal_units, strip_emulation_prevention};

    /// VPS RBSP body extracted from the workspace fixture
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed and emulation-prevention bytes stripped. Captured
    /// inline so the test runs without docs/ on the include path.
    ///
    /// Source byte sequence (Annex B): `00 00 00 01 40 01 0C 01 FF FF
    /// 04 08 00 00 03 00 9F A8 00 00 03 00 00 1E BA 02 40` — after
    /// stripping the start code and the `40 01` NAL header, and after
    /// the §7.4.1.1 strip dropping both `03` emulation bytes:
    /// `0C 01 FF FF 04 08 00 00 00 9F A8 00 00 00 00 1E BA 02 40`.
    const TINY_VPS_RBSP: &[u8] = &[
        0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00,
        0x1E, 0xBA, 0x02, 0x40,
    ];

    #[test]
    fn parses_tiny_fixture_vps() {
        let vps = HevcVps::parse(TINY_VPS_RBSP).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert!(vps.base_layer_internal_flag);
        assert!(vps.base_layer_available_flag);
        assert_eq!(vps.max_layers_minus1, 0); // 1 layer
        assert_eq!(vps.max_sub_layers_minus1, 0); // 1 sub-layer
        assert!(vps.temporal_id_nesting_flag);
        assert_eq!(vps.ptl.general_profile_space, 0);
        assert!(!vps.ptl.general_tier_flag);
        assert_eq!(vps.ptl.general_profile_idc, 4);
        // §A.4 maps `30 == level 1.0 × 30` ⇒ level 1.0. (Not "level
        // 3.0" — that would be value 90.) The fixture's `notes.md`
        // groups it as "level 3.0" but the on-wire `level_idc` is the
        // raw encoded value, which is 30 ⇒ level 1.0.
        assert_eq!(vps.ptl.general_level_idc, 30);
        // With max_sub_layers_minus1 == 0 the per-sub-layer ordering
        // loop still runs once for i == 0. The fixture encoder
        // populates DPB size 3 / no reorder / latency 1 for a
        // single-frame stream, so the on-wire `ue(v)` triple decodes
        // to (2, 0, 1).
        assert!(vps.sub_layer_ordering_info_present_flag);
        assert_eq!(
            vps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(vps.sub_layer_ordering_info[0].max_num_reorder_pics, 0);
        assert_eq!(vps.sub_layer_ordering_info[0].max_latency_increase_plus1, 1);
    }

    #[test]
    fn parses_tiny_fixture_vps_via_nal_walker() {
        // End-to-end check: feed the raw Annex B stream through the
        // NAL walker, then parse the first NAL's RBSP as a VPS.
        let raw = &[
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00,
            0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
        ];
        let units = collect_nal_units(raw).expect("walker");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].header.nal_unit_type, 32); // VPS_NUT
        let vps = HevcVps::parse(&units[0].rbsp).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert_eq!(vps.ptl.general_profile_idc, 4);
        assert_eq!(vps.ptl.general_level_idc, 30);
    }

    #[test]
    fn rejects_wrong_reserved_field() {
        // Same prefix as the fixture but flip the last bit of the
        // 16-bit reserved field so it reads 0xFFFE instead of 0xFFFF.
        let mut bad = TINY_VPS_RBSP.to_vec();
        bad[3] = 0xFE; // was 0xFF
        let err = HevcVps::parse(&bad).unwrap_err();
        assert_eq!(err, VpsError::ReservedFieldMismatch { got: 0xFFFE });
    }

    #[test]
    fn rejects_truncated_rbsp() {
        // Truncate to before the reserved field finishes.
        let err = HevcVps::parse(&TINY_VPS_RBSP[..3]).unwrap_err();
        assert_eq!(err, VpsError::Truncated);
    }

    #[test]
    fn strip_emulation_prevention_then_parse_matches_inline_decode() {
        // Build the wire-format RBSP (with 03 bytes), strip them, and
        // confirm the decoded VPS matches the inline-stripped fixture.
        let wire = &[
            0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
        ];
        let unescaped = strip_emulation_prevention(wire);
        assert_eq!(unescaped, TINY_VPS_RBSP);
        let vps = HevcVps::parse(&unescaped).expect("VPS parse");
        let direct = HevcVps::parse(TINY_VPS_RBSP).expect("VPS parse");
        assert_eq!(vps, direct);
    }

    /// Hand-assembled minimal VPS to exercise the loop expansion path
    /// when `sub_layer_ordering_info_present_flag == 1`. This test
    /// stresses the multi-sub-layer code path without any fixture.
    #[test]
    fn parses_two_sub_layer_ordering_info_present() {
        // Build a VPS with max_sub_layers_minus1 = 1 (two sublayers)
        // and sub_layer_ordering_info_present_flag = 1, so two
        // ordering-info triples are read.
        //
        // Field layout bit-by-bit:
        //   vps_id u4                          : 0000
        //   base_layer_internal_flag u1        : 1
        //   base_layer_available_flag u1       : 1
        //   max_layers_minus1 u6               : 000000
        //   max_sub_layers_minus1 u3           : 001
        //   temporal_id_nesting_flag u1        : 1
        //   reserved u16                       : 1111_1111_1111_1111
        //   profile_tier_level:
        //     profile_space u2                 : 00
        //     tier_flag u1                     : 0
        //     profile_idc u5                   : 00001  (Main)
        //     32 compat flags                  : all 0
        //     4 source flags (prog/i/np/fo)    : 1000
        //     43-bit block                     : all 0
        //     1-bit reserved                   : 0
        //     level_idc u8                     : 00011110 (= 30)
        //     2 bits per inner sublayer for i in 0..1:
        //       sub_layer_profile_present[0] u1: 0
        //       sub_layer_level_present[0] u1  : 1
        //     8-N inner reserved 2-bit pads (N=1 → 7 entries × 2 bits = 14 bits): all 0
        //     since sub_layer_profile_present[0] == 0 → no extra body
        //     sub_layer_level_present[0] == 1 → sub_layer_level_idc[0] u8
        //                                       : 00011110 (= 30)
        //   sub_layer_ordering_info_present_flag u1 : 1
        //   loop i = 0..1 (two iterations) each with:
        //     ue(v) = '1' (codeNum 0)
        //     ue(v) = '1' (codeNum 0)
        //     ue(v) = '1' (codeNum 0)
        //
        // We assemble the bit string and then chunk to bytes MSB-first.
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("1"); // base_layer_internal_flag
        push("1"); // base_layer_available_flag
        push("000000"); // max_layers_minus1
        push("001"); // max_sub_layers_minus1 = 1
        push("1"); // temporal_id_nesting_flag
        push("1111111111111111"); // reserved
        push("00"); // profile_space
        push("0"); // tier_flag
        push("00001"); // profile_idc = 1
        push(&"0".repeat(32)); // compat flags
        push("1000"); // prog/interlaced/non_packed/frame_only
        push(&"0".repeat(43)); // 43-bit block
        push("0"); // 1-bit reserved
        push("00011110"); // general_level_idc = 30
        push("01"); // sub_layer_profile_present[0]=0, sub_layer_level_present[0]=1
        push(&"0".repeat(2 * 7)); // 14-bit padding for i=1..8
        push("00011110"); // sub_layer_level_idc[0] = 30
        push("1"); // sub_layer_ordering_info_present_flag
        push("1"); // ue=0 (max_dec_pic_buffering_minus1[0])
        push("1"); // ue=0 (max_num_reorder_pics[0])
        push("1"); // ue=0 (max_latency_increase_plus1[0])
        push("1"); // ue=0 (max_dec_pic_buffering_minus1[1])
        push("1"); // ue=0 (max_num_reorder_pics[1])
        push("1"); // ue=0 (max_latency_increase_plus1[1])
        // VPS tail (round 12 — fully parsed):
        push("000000"); // vps_max_layer_id = 0 (single layer)
        push("1"); // vps_num_layer_sets_minus1 ue=0 (just layer set 0)
        push("0"); // vps_timing_info_present_flag = 0
        push("0"); // vps_extension_flag = 0

        // Pad with zeros to next byte boundary and pack.
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }

        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert_eq!(vps.max_sub_layers_minus1, 1);
        assert!(vps.temporal_id_nesting_flag);
        assert_eq!(vps.ptl.general_profile_idc, 1);
        assert_eq!(vps.ptl.general_level_idc, 30);
        assert!(!vps.ptl.sub_layer_profile_present[0]);
        assert!(vps.ptl.sub_layer_level_present[0]);
        assert_eq!(vps.ptl.sub_layer_level_idc[0], 30);
        assert!(vps.sub_layer_ordering_info_present_flag);
        for i in 0..=1 {
            assert_eq!(
                vps.sub_layer_ordering_info[i].max_dec_pic_buffering_minus1,
                0
            );
            assert_eq!(vps.sub_layer_ordering_info[i].max_num_reorder_pics, 0);
            assert_eq!(vps.sub_layer_ordering_info[i].max_latency_increase_plus1, 0);
        }
        // Tail decoded as expected.
        assert_eq!(vps.max_layer_id, 0);
        assert_eq!(vps.num_layer_sets_minus1, 0);
        assert!(!vps.timing_info_present_flag);
        assert!(vps.timing_info.is_none());
        assert!(!vps.vps_extension_flag);
        assert!(vps.opaque_tail.is_none());
        assert!(vps.hrd_parameters.is_empty());
    }

    #[test]
    fn parses_ordering_info_present_flag_zero_propagates() {
        // max_sub_layers_minus1 = 1, but
        // sub_layer_ordering_info_present_flag = 0 → only the [1]
        // entry is signalled; the [0] entry inherits it per §7.4.3.1.
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("1");
        push("1");
        push("000000");
        push("001"); // max_sub_layers_minus1 = 1
        push("1");
        push("1111111111111111");
        push("00");
        push("0");
        push("00001"); // profile_idc=1
        push(&"0".repeat(32));
        push("1000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level_idc=30
        push("00"); // sub_layer present flags both 0
        push(&"0".repeat(14));
        push("0"); // sub_layer_ordering_info_present_flag = 0
        // Only the i=1 triple is read; we want the [1] DPB = 2 (codeNum 2 = '011').
        push("011"); // max_dec_pic_buffering_minus1[1] = 2
        push("1"); // max_num_reorder_pics[1] = 0
        push("1"); // max_latency_increase_plus1[1] = 0
        // VPS tail (round 12 — fully parsed):
        push("000000"); // vps_max_layer_id = 0
        push("1"); // vps_num_layer_sets_minus1 ue=0
        push("0"); // vps_timing_info_present_flag = 0
        push("0"); // vps_extension_flag = 0

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert!(!vps.sub_layer_ordering_info_present_flag);
        // [1] was signalled; [0] inherits.
        assert_eq!(
            vps.sub_layer_ordering_info[1].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(
            vps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(vps.max_layer_id, 0);
        assert_eq!(vps.num_layer_sets_minus1, 0);
        assert!(!vps.timing_info_present_flag);
    }

    /// Hand-assembled minimal VPS that exercises the new tail: a
    /// single extra layer set + a timing-info block with
    /// `poc_proportional_to_timing_flag == 1` + zero HRDs +
    /// `vps_extension_flag == 0`.
    #[test]
    fn parses_layer_set_matrix_and_timing_info() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        // Minimal prefix: single sub-layer, base Main profile, level 30.
        push("0000"); // vps_id
        push("1"); // base_layer_internal_flag
        push("1"); // base_layer_available_flag
        push("000000"); // max_layers_minus1
        push("000"); // max_sub_layers_minus1 = 0
        push("1"); // temporal_id_nesting_flag
        push("1111111111111111"); // reserved
        push("00"); // profile_space
        push("0"); // tier_flag
        push("00001"); // profile_idc = 1
        push(&"0".repeat(32)); // compat flags
        push("1000"); // prog/interlaced/non_packed/frame_only
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level_idc = 30
        push("1"); // sub_layer_ordering_info_present_flag = 1
        push("1"); // ue=0 (max_dec_pic_buffering_minus1[0])
        push("1"); // ue=0 (max_num_reorder_pics[0])
        push("1"); // ue=0 (max_latency_increase_plus1[0])
        // Tail:
        push("000001"); // vps_max_layer_id = 1 (so row width 2)
        push("010"); // vps_num_layer_sets_minus1 ue=1 (one signalled row)
        // layer_id_included_flag[1][0..=1]: pick 1, 0
        push("10");
        push("1"); // vps_timing_info_present_flag = 1
        // num_units_in_tick = 1001 (NTSC-ish denominator)
        for i in (0..32).rev() {
            let b = (1001u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        // time_scale = 60000
        for i in (0..32).rev() {
            let b = (60000u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        push("1"); // poc_proportional_to_timing_flag = 1
        push("1"); // num_ticks_poc_diff_one_minus1 ue=0
        push("1"); // num_hrd_parameters ue=0 (no HRDs — keeps tail parseable)
        push("0"); // vps_extension_flag = 0

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert_eq!(vps.max_layer_id, 1);
        assert_eq!(vps.num_layer_sets_minus1, 1);
        assert_eq!(vps.layer_id_included_flag.len(), 1);
        assert_eq!(vps.layer_id_included_flag[0].flags, vec![true, false]);
        assert!(vps.timing_info_present_flag);
        let ti = vps.timing_info.as_ref().expect("timing info present");
        assert_eq!(ti.num_units_in_tick, 1001);
        assert_eq!(ti.time_scale, 60000);
        assert!(ti.poc_proportional_to_timing_flag);
        assert_eq!(ti.num_ticks_poc_diff_one_minus1, Some(0));
        assert_eq!(ti.num_hrd_parameters, 0);
        assert!(!vps.vps_extension_flag);
        assert!(vps.opaque_tail.is_none());
        assert!(vps.hrd_parameters.is_empty());
    }

    /// Tail with `vps_num_hrd_parameters == 1` — the parser must
    /// decode the §E.2.2 `hrd_parameters()` body inline rather than
    /// surface it as the opaque tail.
    #[test]
    fn parses_hrd_payload_inline() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000");
        push("1");
        push("1");
        push("000000");
        push("000"); // max_sub_layers_minus1 = 0
        push("1");
        push("1111111111111111");
        push("00");
        push("0");
        push("00001");
        push(&"0".repeat(32));
        push("1000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level_idc
        push("1"); // sub_layer_ordering_info_present_flag = 1
        push("1");
        push("1");
        push("1");
        // Tail:
        push("000000"); // max_layer_id = 0
        push("1"); // num_layer_sets_minus1 ue=0
        push("1"); // timing_info_present_flag = 1
        for i in (0..32).rev() {
            let b = (1u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        for i in (0..32).rev() {
            let b = (30u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        push("0"); // poc_proportional_to_timing_flag = 0
        push("010"); // num_hrd_parameters ue=1
        // HRD entry i=0:
        push("1"); // hrd_layer_set_idx ue=0
        // cprms_present_flag not signalled (i == 0); inferred 1
        // hrd_parameters( 1, 0 ):
        push("0"); // nal_hrd_parameters_present_flag
        push("0"); // vcl_hrd_parameters_present_flag
        // (nal | vcl) = 0 → no common-info inner block
        // sub-layer i=0:
        push("1"); // fixed_pic_rate_general_flag = 1 → within_cvs inferred 1
        push("1"); // elemental_duration_in_tc_minus1 ue=0
        // low_delay not signalled, inferred 0
        push("1"); // cpb_cnt_minus1 ue=0
        // no NAL/VCL HRD bodies (gates = 0)
        push("0"); // vps_extension_flag = 0

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        let ti = vps.timing_info.as_ref().expect("timing info present");
        assert_eq!(ti.num_hrd_parameters, 1);
        assert!(!ti.poc_proportional_to_timing_flag);
        // The HRD body is now decoded inline, not surfaced as the
        // opaque tail. vps_extension_flag is read after the HRD loop.
        assert_eq!(vps.hrd_parameters.len(), 1);
        let entry = &vps.hrd_parameters[0];
        assert_eq!(entry.hrd_layer_set_idx, 0);
        assert!(entry.cprms_present_flag);
        let common = entry.hrd.common.as_ref().expect("common present");
        assert!(!common.nal_hrd_parameters_present_flag);
        assert!(!common.vcl_hrd_parameters_present_flag);
        assert_eq!(entry.hrd.sub_layers.len(), 1);
        let sl = &entry.hrd.sub_layers[0];
        assert!(sl.fixed_pic_rate_general_flag);
        assert!(sl.fixed_pic_rate_within_cvs_flag);
        assert_eq!(sl.elemental_duration_in_tc_minus1, Some(0));
        assert_eq!(sl.cpb_cnt_minus1, 0);
        assert!(sl.nal_hrd.is_none());
        assert!(sl.vcl_hrd.is_none());
        assert!(!vps.vps_extension_flag);
        assert!(vps.opaque_tail.is_none());
    }

    /// `vps_num_units_in_tick == 0` is forbidden by the §E.2.1 /
    /// §7.3.2.1 semantics text.
    #[test]
    fn rejects_zero_num_units_in_tick() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000");
        push("1");
        push("1");
        push("000000");
        push("000");
        push("1");
        push("1111111111111111");
        push("00");
        push("0");
        push("00001");
        push(&"0".repeat(32));
        push("1000");
        push(&"0".repeat(43));
        push("0");
        push("00011110");
        push("1");
        push("1");
        push("1");
        push("1");
        push("000000");
        push("1");
        push("1"); // timing_info_present_flag = 1
        push(&"0".repeat(32)); // num_units_in_tick = 0  → invalid
        push(&"0".repeat(32)); // time_scale (unread)

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let err = HevcVps::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            VpsError::ValueOutOfRange {
                field: "vps_num_units_in_tick",
                got: 0
            }
        );
    }

    /// `vps_num_hrd_parameters == 2` with the second entry's
    /// `cprms_present_flag[1] == 0` — common-info gates inherit from
    /// entry 0 per §7.4.3.1.
    #[test]
    fn parses_two_hrd_entries_with_cprms_inheritance() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        // Same prefix as parses_layer_set_matrix_and_timing_info but
        // with two layer sets (so num_hrd_parameters can be 2).
        push("0000"); // vps_id
        push("1");
        push("1");
        push("000000");
        push("000"); // max_sub_layers_minus1 = 0
        push("1");
        push("1111111111111111");
        push("00");
        push("0");
        push("00001"); // profile_idc = 1
        push(&"0".repeat(32));
        push("1000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level_idc = 30
        push("1"); // sub_layer_ordering_info_present_flag = 1
        push("1");
        push("1");
        push("1");
        push("000000"); // max_layer_id = 0
        push("011"); // vps_num_layer_sets_minus1 ue=2 → 3 layer sets, room for 3 HRDs
        // layer_id_included_flag[1..=2][0..=0]: two rows of one bit each
        push("1");
        push("1");
        push("1"); // timing_info_present_flag = 1
        for i in (0..32).rev() {
            let b = (1u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        for i in (0..32).rev() {
            let b = (30u32 >> i) & 1;
            push(if b == 1 { "1" } else { "0" });
        }
        push("0"); // poc_proportional_to_timing_flag = 0
        push("011"); // num_hrd_parameters ue=2

        // HRD entry i=0: hrd_layer_set_idx=0, cprms inferred 1.
        push("1"); // hrd_layer_set_idx ue=0
        // hrd_parameters( 1, 0 ):
        push("1"); // nal_hrd_parameters_present_flag = 1
        push("0"); // vcl
        push("0"); // sub_pic_hrd_params_present_flag = 0
        // bit_rate_scale / cpb_size_scale u(4) each
        push("0000");
        push("0000");
        // initial / au / dpb_output u(5) each
        push("10111");
        push("10111");
        push("10111");
        // sub-layer i=0:
        push("1"); // fixed_pic_rate_general = 1
        push("1"); // elemental_duration_in_tc_minus1 ue=0
        // cpb_cnt_minus1 ue=0
        push("1");
        // NAL HRD body: CpbCnt = 1
        push("1"); // bit_rate_value_minus1 ue=0
        push("1"); // cpb_size_value_minus1 ue=0
        push("1"); // cbr_flag[0] = 1

        // HRD entry i=1: hrd_layer_set_idx ue=1 ("010"), cprms_present_flag = 0
        push("010"); // hrd_layer_set_idx = 1
        push("0"); // cprms_present_flag[1] = 0 → inherit entry 0's common info
        // hrd_parameters( 0, 0 ): no common info read; gates
        // inherited (nal = true).
        // sub-layer i=0:
        push("1"); // fixed_pic_rate_general = 1
        push("1"); // elemental_duration_in_tc_minus1 ue=0
        // cpb_cnt_minus1 ue=0
        push("1");
        push("1"); // bit_rate_value_minus1 ue=0
        push("1"); // cpb_size_value_minus1 ue=0
        push("0"); // cbr_flag = 0

        push("0"); // vps_extension_flag = 0

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert_eq!(vps.hrd_parameters.len(), 2);
        let e0 = &vps.hrd_parameters[0];
        assert_eq!(e0.hrd_layer_set_idx, 0);
        assert!(e0.cprms_present_flag);
        let common = e0.hrd.common.as_ref().expect("entry 0 common");
        assert!(common.nal_hrd_parameters_present_flag);
        assert_eq!(e0.hrd.sub_layers[0].nal_hrd.as_ref().unwrap().cpb.len(), 1);

        let e1 = &vps.hrd_parameters[1];
        assert_eq!(e1.hrd_layer_set_idx, 1);
        assert!(!e1.cprms_present_flag);
        // common is None (cprms_present_flag = 0), but the inherited
        // gates from entry 0 still drove the per-sub-layer NAL HRD body.
        assert!(e1.hrd.common.is_none());
        let nal = e1.hrd.sub_layers[0]
            .nal_hrd
            .as_ref()
            .expect("nal inherited via cprms");
        assert_eq!(nal.cpb.len(), 1);
        assert!(!nal.cpb[0].cbr_flag);
    }
}
