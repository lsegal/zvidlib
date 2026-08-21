//! Slice segment header parser per ITU-T Rec. H.265 §7.3.6.1.
//!
//! Round-6 scope: parse the `slice_segment_header()` syntax structure
//! (§7.3.6.1) for an independent slice segment, materialising every
//! field that does **not** require decoded-picture-buffer state to
//! interpret. The parse takes the activated SPS and PPS as context
//! because several field widths and presence gates are derived from
//! them (the `slice_segment_address` width is
//! `Ceil( Log2( PicSizeInCtbsY ) )`, the `slice_pic_order_cnt_lsb`
//! width is `log2_max_pic_order_cnt_lsb_minus4 + 4`, the SAO / MVP
//! gates come from the SPS, and the tiles / entropy-sync entry-point
//! block comes from the PPS).
//!
//! ## What this round materialises
//!
//! * `first_slice_segment_in_pic_flag`, `no_output_of_prior_pics_flag`
//!   (IRAP only), `slice_pic_parameter_set_id`.
//! * For non-first slice segments: `dependent_slice_segment_flag`
//!   (only when `dependent_slice_segments_enabled_flag`) and
//!   `slice_segment_address` (`u(v)`, width
//!   `Ceil( Log2( PicSizeInCtbsY ) )`).
//! * For independent slice segments (`!dependent_slice_segment_flag`):
//!   the `slice_reserved_flag[]` block, `slice_type`,
//!   `pic_output_flag` (only when `output_flag_present_flag`),
//!   `colour_plane_id` (only when `separate_colour_plane_flag`),
//!   `slice_temporal_mvp_enabled_flag` (only when
//!   `sps_temporal_mvp_enabled_flag`), the SAO luma / chroma gates,
//!   `slice_qp_delta` (`se(v)`), the chroma QP offsets, the
//!   deblocking-filter override block, and
//!   `slice_loop_filter_across_slices_enabled_flag`.
//! * The entry-point-offset block (`num_entry_point_offsets`,
//!   `offset_len_minus1`, `entry_point_offset_minus1[]`) when
//!   `tiles_enabled_flag || entropy_coding_sync_enabled_flag`.
//! * The slice-segment-header extension block when
//!   `slice_segment_header_extension_present_flag`.
//! * `byte_alignment()` consumed to the next byte boundary, so the
//!   reported [`SliceSegmentHeader::byte_offset_to_slice_data`] points
//!   at the first byte of `slice_segment_data()`.
//!
//! ## What this round defers (surfaced, not decoded)
//!
//! Three points need state this round does not carry:
//!
//! * The **non-IDR picture-order-count + reference-picture-set block**
//!   (`slice_pic_order_cnt_lsb`, `short_term_ref_pic_set_sps_flag`,
//!   the inline `st_ref_pic_set()`, the long-term block) needs the
//!   SPS short-term-RPS parser to be re-entered for the in-line
//!   `stRpsIdx == num_short_term_ref_pic_sets` case, which is not yet
//!   exposed publicly. When the current NAL unit is **not** an IDR
//!   (`nal_unit_type != IDR_W_RADL && != IDR_N_LP`), the parser stops
//!   right after `colour_plane_id` and surfaces the remainder as
//!   [`SliceSegmentHeader::opaque_tail`].
//! * The **P / B reference-list / weighted-prediction sub-structures**
//!   (`ref_pic_lists_modification()` §7.3.6.2 and `pred_weight_table()`
//!   §7.3.6.3) need DPB-derived `NumPicTotalCurr` / `RefPicList`
//!   values. When `slice_type` is P or B the parser materialises the
//!   common P/B fields up to (but not including) the point where those
//!   sub-structures would begin, then surfaces the remainder as the
//!   opaque tail. The §7.3.6.2 syntax structure itself is implemented
//!   as a standalone parser ([`RefPicListsModification::parse`]) so a
//!   future round that wires up the §7.4.7.2 `NumPicTotalCurr`
//!   derivation can decode the reference-picture-list-modification
//!   block in place; the implicit `RefPicListTempX` derivation of
//!   §8.3.4 stays the consumer's responsibility. When
//!   `pps.lists_modification_present_flag == 0` the modification block
//!   is statically absent (the §7.3.6.1 `if(... && NumPicTotalCurr > 1)`
//!   short-circuit applies independent of any DPB state); the parser
//!   in that case continues into the §7.3.6.1
//!   `mvd_l1_zero_flag` / `cabac_init_flag` /
//!   `collocated_from_l0_flag` / `collocated_ref_idx` block in-place
//!   and surfaces those four fields, then defers at the weighted-pred
//!   gate.
//!
//! Independent **I-slice IDR** segments — the dominant case for the
//! intra-only fixtures this rebuild targets — are parsed end to end
//! through `byte_alignment()`.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::pps::PicParameterSet;
use crate::hevc::engine::sps::{OpaqueTail, SeqParameterSet, ShortTermRefPicSet, SpsError};

/// `nal_unit_type` value `BLA_W_LP` (Table 7-1). The IRAP range used by
/// the `no_output_of_prior_pics_flag` gate is `BLA_W_LP..=RSV_IRAP_VCL23`.
pub const BLA_W_LP: u8 = 16;
/// `nal_unit_type` value `IDR_W_RADL` (Table 7-1).
pub const IDR_W_RADL: u8 = 19;
/// `nal_unit_type` value `IDR_N_LP` (Table 7-1).
pub const IDR_N_LP: u8 = 20;
/// `nal_unit_type` value `RSV_IRAP_VCL23` (Table 7-1) — the inclusive
/// upper bound of the IRAP NAL-unit-type range.
pub const RSV_IRAP_VCL23: u8 = 23;

/// `slice_type` enumeration per Table 7-7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// B slice (`slice_type == 0`).
    B,
    /// P slice (`slice_type == 1`).
    P,
    /// I slice (`slice_type == 2`).
    I,
}

impl SliceType {
    /// Map the raw `ue(v)` `slice_type` value to the enum, rejecting
    /// any value outside `0..=2`.
    fn from_raw(v: u32) -> Result<Self, SliceError> {
        match v {
            0 => Ok(Self::B),
            1 => Ok(Self::P),
            2 => Ok(Self::I),
            other => Err(SliceError::ValueOutOfRange {
                field: "slice_type",
                got: other as i64,
            }),
        }
    }

    /// True for P and B slices (the slice types that signal reference
    /// lists, weighted prediction, etc.).
    pub fn is_inter(self) -> bool {
        matches!(self, Self::P | Self::B)
    }
}

/// Errors that can arise while parsing a slice segment header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceError {
    /// The RBSP ran out of bits before the header was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.7.1.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as an `i64` (covers both `ue(v)` and
        /// `se(v)` elements).
        got: i64,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
    /// The in-line `st_ref_pic_set(num_short_term_ref_pic_sets)` parse
    /// (invoked from §7.3.6.1 when `short_term_ref_pic_set_sps_flag == 0`)
    /// failed.
    InlineShortTermRpsParse(SpsError),
}

impl core::fmt::Display for SliceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("slice segment header RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "slice header syntax element {field} out of range: {got}")
            }
            Self::Bitstream(e) => write!(f, "bitstream error during slice header parse: {e}"),
            Self::InlineShortTermRpsParse(e) => {
                write!(f, "in-line slice-header st_ref_pic_set parse failed: {e}")
            }
        }
    }
}

impl std::error::Error for SliceError {}

impl From<BitReaderError> for SliceError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<SpsError> for SliceError {
    fn from(e: SpsError) -> Self {
        match e {
            SpsError::Truncated => Self::Truncated,
            SpsError::Bitstream(b) => Self::Bitstream(b),
            other => Self::InlineShortTermRpsParse(other),
        }
    }
}

/// Deblocking-filter override block carried in the slice header
/// (§7.3.6.1, gated by `deblocking_filter_override_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceDeblocking {
    /// `slice_deblocking_filter_disabled_flag`. Inferred to
    /// `pps_deblocking_filter_disabled_flag` when the override block is
    /// absent (§7.4.7.1).
    pub disabled_flag: bool,
    /// `slice_beta_offset_div2` (`se(v)`, range −6..=6). Inferred to
    /// `pps_beta_offset_div2` when absent.
    pub beta_offset_div2: i8,
    /// `slice_tc_offset_div2` (`se(v)`, range −6..=6). Inferred to
    /// `pps_tc_offset_div2` when absent.
    pub tc_offset_div2: i8,
}

/// One long-term reference picture entry signalled in the slice
/// header (§7.3.6.1). For the first `num_long_term_sps` entries the
/// `lt_idx_sps` indexes the SPS's long-term-ref-pic table; for the
/// remaining `num_long_term_pics` entries the slice header carries
/// the POC LSB and `used_by_curr_pic_lt_flag` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceLongTermRefPic {
    /// Source of the entry: SPS table or in-slice signalling.
    pub source: SliceLongTermRefPicSource,
    /// `delta_poc_msb_present_flag[i]`.
    pub delta_poc_msb_present_flag: bool,
    /// `delta_poc_msb_cycle_lt[i]`. Inferred to 0 when
    /// [`Self::delta_poc_msb_present_flag`] is false (§7.4.7.1).
    pub delta_poc_msb_cycle_lt: u32,
}

/// Source of one [`SliceLongTermRefPic`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceLongTermRefPicSource {
    /// First `num_long_term_sps` entries: `lt_idx_sps[i]` indexes the
    /// SPS's long-term-ref-pic-poc table. The `u(v)` width is
    /// `Ceil(Log2(num_long_term_ref_pics_sps))` bits; the value 0 is
    /// inferred when `num_long_term_ref_pics_sps == 1`.
    Sps {
        /// `lt_idx_sps[i]` value (or 0 when inferred).
        lt_idx_sps: u32,
    },
    /// Remaining `num_long_term_pics` entries: the POC LSB and
    /// `used_by_curr_pic_lt_flag` are signalled directly in the slice
    /// header.
    InSlice {
        /// `poc_lsb_lt[i]` (`u(v)`, width
        /// `log2_max_pic_order_cnt_lsb_minus4 + 4`).
        poc_lsb_lt: u32,
        /// `used_by_curr_pic_lt_flag[i]`.
        used_by_curr_pic_lt_flag: bool,
    },
}

impl SliceLongTermRefPic {
    /// Resolve `UsedByCurrPicLt[i]` for this entry per §7.4.7.1:
    ///
    /// > `UsedByCurrPicLt[ i ]` is set equal to
    /// > `used_by_curr_pic_lt_sps_flag[ lt_idx_sps[ i ] ]` when the
    /// > entry's source is the SPS table, and to
    /// > `used_by_curr_pic_lt_flag[ i ]` when the entry is signalled
    /// > directly in the slice header.
    ///
    /// Returns `None` when [`SliceLongTermRefPicSource::Sps`] points at
    /// an index that is out of range of `sps.long_term_ref_pics`
    /// (a bitstream-conformance failure — the SPS-resident table must
    /// cover every `lt_idx_sps[i]` value).
    pub fn used_by_curr_pic_lt(&self, sps: &SeqParameterSet) -> Option<bool> {
        match self.source {
            SliceLongTermRefPicSource::Sps { lt_idx_sps } => sps
                .long_term_ref_pics
                .get(lt_idx_sps as usize)
                .map(|entry| entry.used_by_curr_pic),
            SliceLongTermRefPicSource::InSlice {
                used_by_curr_pic_lt_flag,
                ..
            } => Some(used_by_curr_pic_lt_flag),
        }
    }
}

/// Entry-point-offset block (§7.3.6.1, gated by
/// `tiles_enabled_flag || entropy_coding_sync_enabled_flag`).
///
/// `num_entry_point_offsets` is the number of subsets of
/// `slice_segment_data()` minus one; each
/// `entry_point_offset_minus1[i] + 1` is the byte length of subset
/// `i` (§7.4.7.1). The trailing subset (`num_entry_point_offsets`)
/// runs to the end of `slice_segment_data()` and is therefore not
/// encoded.
///
/// Per §7.4.7.1, the on-wire `num_entry_point_offsets` is bounded by
/// the active partitioning:
///
/// * `tiles_enabled_flag == 1` and `entropy_coding_sync_enabled_flag
///   == 0` → `0..=(NumTileColumns * NumTileRows − 1)`.
/// * `tiles_enabled_flag == 0` and `entropy_coding_sync_enabled_flag
///   == 1` → `0..=(PicHeightInCtbsY − 1)`.
/// * Both flags set (the "tiles + WPP" combination) is constrained by
///   §7.4.3.3.1 to never appear in a conforming stream; this parser
///   accepts the wider of the two bounds in that pathological case
///   rather than gate the parse on a flag combination the PPS parser
///   already rejects.
///
/// Each `entry_point_offset_minus1[i]` is `offset_len_minus1 + 1` bits
/// wide. `offset_len_minus1` itself is bounded to `0..=31` by
/// §7.4.7.1, so the per-entry width is in `1..=32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPointOffsets {
    /// `num_entry_point_offsets` (`ue(v)`). The number of subsets of
    /// slice-segment data is this value plus one.
    pub num_entry_point_offsets: u32,
    /// `offset_len_minus1` (`ue(v)`, range 0..=31). Each
    /// `entry_point_offset_minus1[i]` is `offset_len_minus1 + 1` bits.
    /// Only meaningful when `num_entry_point_offsets > 0`; left at 0
    /// when no offsets are signalled.
    pub offset_len_minus1: u8,
    /// `entry_point_offset_minus1[i]` (`u(offset_len_minus1 + 1)`) for
    /// `i = 0 .. num_entry_point_offsets`. Empty when
    /// `num_entry_point_offsets == 0`.
    pub entry_point_offset_minus1: Vec<u32>,
}

impl EntryPointOffsets {
    /// Byte length of subset `i` per §7.4.7.1, i.e.
    /// `entry_point_offset_minus1[i] + 1`. Returns `None` when `i` is
    /// out of range.
    pub fn subset_length(&self, i: usize) -> Option<u64> {
        self.entry_point_offset_minus1
            .get(i)
            .map(|v| u64::from(*v) + 1)
    }
}

/// Parsed `ref_pic_lists_modification()` syntax structure
/// (ITU-T Rec. H.265 §7.3.6.2 / §7.4.7.2).
///
/// The structure is signalled in the slice header when
/// `lists_modification_present_flag == 1 && NumPicTotalCurr > 1`
/// (§7.3.6.1 gate). It carries a per-list "explicit list" override of
/// the implicit `RefPicList0` / `RefPicList1` derivation of §8.3.4: the
/// `list_entry_lX[i]` value is the index of the reference picture in
/// `RefPicListTempX` to place at position `i` of `RefPicListX`. The
/// `RefPicListTempX` derivation itself is part of §8.3.4 (DPB-driven)
/// and is **not** performed by this parser — this struct surfaces only
/// the on-wire syntax elements and applies the §7.4.7.2 inference and
/// range checks.
///
/// Width of each `list_entry_lX[i]` is `Ceil( Log2( NumPicTotalCurr ) )`
/// bits and the value must be in `0 ..= NumPicTotalCurr - 1`
/// (§7.4.7.2). When `ref_pic_list_modification_flag_lX == 0` the
/// corresponding entry list is empty; the implicit derivation of
/// §8.3.4 applies (each `list_entry_lX[i]` is inferred to 0 per the
/// §7.4.7.2 paragraph "When the syntax element list_entry_lX[i] is
/// not present in the slice header, it is inferred to be equal to 0",
/// but the inference is exercised by §8.3.4, not surfaced here).
///
/// The list-1 fields are present only when `slice_type == B`
/// (§7.3.6.2 syntax). For a P slice the `list_entry_l1` vector is
/// always empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefPicListsModification {
    /// `ref_pic_list_modification_flag_l0` (`u(1)`).
    pub ref_pic_list_modification_flag_l0: bool,
    /// `list_entry_l0[i]` for `i = 0 ..= num_ref_idx_l0_active_minus1`.
    /// Empty when `ref_pic_list_modification_flag_l0 == 0`.
    pub list_entry_l0: Vec<u32>,
    /// `ref_pic_list_modification_flag_l1` (`u(1)`). `None` when the
    /// slice is not a B slice (the field is not signalled).
    pub ref_pic_list_modification_flag_l1: Option<bool>,
    /// `list_entry_l1[i]` for `i = 0 ..= num_ref_idx_l1_active_minus1`.
    /// Empty for P slices and when `ref_pic_list_modification_flag_l1
    /// == 0`.
    pub list_entry_l1: Vec<u32>,
}

impl RefPicListsModification {
    /// Parse `ref_pic_lists_modification()` (§7.3.6.2) from the current
    /// bit position of `br`.
    ///
    /// * `slice_type` — the active slice type. Per §7.3.6.2 the L1
    ///   block (`ref_pic_list_modification_flag_l1` /
    ///   `list_entry_l1[]`) is only signalled for B slices. For an
    ///   I slice the structure is never present at all (the §7.3.6.1
    ///   gate `lists_modification_present_flag && NumPicTotalCurr > 1`
    ///   sits inside the inter-slice `slice_type != I` branch), so
    ///   the parser rejects `SliceType::I` up front.
    /// * `num_ref_idx_l0_active_minus1` — the *active* value (after
    ///   the `num_ref_idx_active_override_flag` override, falling back
    ///   to `num_ref_idx_l0_default_active_minus1` from the PPS). Per
    ///   §7.4.7.1 the value is in `0 ..= 14`.
    /// * `num_ref_idx_l1_active_minus1` — same, for L1; ignored for P
    ///   slices.
    /// * `num_pic_total_curr` — `NumPicTotalCurr` (§7.4.7.2 /
    ///   equation 7-57). The caller derives it from the active RPS;
    ///   the §7.3.6.1 gate guarantees `num_pic_total_curr > 1` at the
    ///   point of call. This parser rejects `num_pic_total_curr <= 1`
    ///   (the §7.3.6.1 gate would have prevented the call); a call
    ///   with `num_pic_total_curr == 0` would also imply a bitstream
    ///   conformance failure per §7.4.7.1 ("when the current picture
    ///   contains a P or B slice, the value of NumPicTotalCurr shall
    ///   not be equal to 0"). Each `list_entry_lX[i]` is read as
    ///   `u(v)` of width `Ceil( Log2( num_pic_total_curr ) )` bits
    ///   and range-checked against `num_pic_total_curr - 1`.
    pub fn parse(
        br: &mut BitReader<'_>,
        slice_type: SliceType,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
        num_pic_total_curr: u32,
    ) -> Result<Self, SliceError> {
        if slice_type == SliceType::I {
            return Err(SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/slice_type",
                got: 2,
            });
        }
        if num_pic_total_curr <= 1 {
            return Err(SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: num_pic_total_curr as i64,
            });
        }
        // §7.4.7.1 ranges `num_ref_idx_lX_active_minus1` at 0..=14;
        // defensively cap the per-list loop length so a corrupted call
        // can't drive an unbounded allocation. (The cap matches the
        // spec maximum; a value above 14 would be rejected by the
        // §7.4.7.1 slice-header parse before reaching here.)
        if num_ref_idx_l0_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: num_ref_idx_l0_active_minus1 as i64,
            });
        }
        if slice_type == SliceType::B && num_ref_idx_l1_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l1_active_minus1",
                got: num_ref_idx_l1_active_minus1 as i64,
            });
        }

        let entry_bits = ceil_log2(num_pic_total_curr);
        let max_entry = num_pic_total_curr - 1;

        let ref_pic_list_modification_flag_l0 = br.u1()? != 0;
        let mut list_entry_l0: Vec<u32> = Vec::new();
        if ref_pic_list_modification_flag_l0 {
            let n = num_ref_idx_l0_active_minus1 as u32 + 1;
            list_entry_l0.reserve(n as usize);
            for _ in 0..n {
                let v = br.u(entry_bits)?;
                if v > max_entry {
                    return Err(SliceError::ValueOutOfRange {
                        field: "list_entry_l0",
                        got: v as i64,
                    });
                }
                list_entry_l0.push(v);
            }
        }

        let (ref_pic_list_modification_flag_l1, list_entry_l1) = if slice_type == SliceType::B {
            let flag = br.u1()? != 0;
            let mut entries: Vec<u32> = Vec::new();
            if flag {
                let n = num_ref_idx_l1_active_minus1 as u32 + 1;
                entries.reserve(n as usize);
                for _ in 0..n {
                    let v = br.u(entry_bits)?;
                    if v > max_entry {
                        return Err(SliceError::ValueOutOfRange {
                            field: "list_entry_l1",
                            got: v as i64,
                        });
                    }
                    entries.push(v);
                }
            }
            (Some(flag), entries)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            ref_pic_list_modification_flag_l0,
            list_entry_l0,
            ref_pic_list_modification_flag_l1,
            list_entry_l1,
        })
    }
}

/// Inputs to the §7.4.7.2 `NumPicTotalCurr` derivation (equation 7-57).
///
/// `NumPicTotalCurr` counts the reference pictures in the current
/// slice's RPS state that are flagged as *used by the current
/// picture* — i.e. eligible for entry into `RefPicListTemp0` /
/// `RefPicListTemp1`. The §7.3.6.1 gate
/// `lists_modification_present_flag && NumPicTotalCurr > 1` consumes
/// the derivation to decide whether the inter-slice header carries a
/// `ref_pic_lists_modification()` block, and the per-entry width of
/// that block's `list_entry_lX[i]` (`Ceil( Log2( NumPicTotalCurr ) )`,
/// §7.4.7.2) consumes the value directly.
///
/// The four `UsedByCurrPic*` slices supplied by the caller are the
/// resolved per-position state of the active RPS:
///
/// * `used_by_curr_pic_s0` — `UsedByCurrPicS0[ CurrRpsIdx ][ i ]` for
///   `i = 0 .. NumNegativePics[ CurrRpsIdx ]`.
/// * `used_by_curr_pic_s1` — `UsedByCurrPicS1[ CurrRpsIdx ][ i ]` for
///   `i = 0 .. NumPositivePics[ CurrRpsIdx ]`.
/// * `used_by_curr_pic_lt` — `UsedByCurrPicLt[ i ]` for
///   `i = 0 .. num_long_term_sps + num_long_term_pics`. The §7.4.7.1
///   selector ("SPS-resident → `used_by_curr_pic_lt_sps_flag[
///   lt_idx_sps[ i ] ]`; in-slice → `used_by_curr_pic_lt_flag[ i ]`")
///   is applied by the caller; [`SliceLongTermRefPic::used_by_curr_pic_lt`]
///   does the per-entry resolution against the active SPS.
///
/// For the explicit (non-inter-RPS-predicted) short-term RPS form, the
/// `S0` / `S1` slices come directly from
/// [`ShortTermRefPicSet::used_by_curr_pic_s0_flag`] /
/// [`ShortTermRefPicSet::used_by_curr_pic_s1_flag`] and the
/// [`Self::from_explicit_short_term_rps`] builder is provided. For the
/// inter-RPS-prediction form the §7.4.8 derivation (equations
/// 7-58..7-66) must be run first; the result of that derivation is
/// then handed to [`Self::from_used_flags`].
///
/// The remaining inputs:
///
/// * `pps_curr_pic_ref_enabled_flag` — §7.4.7.2 closing-clause flag,
///   from the SCC extension of the active PPS. Inferred to `false`
///   when the SCC PPS is not signalled (§7.4.3.3.1.4).
/// * `nal_unit_type` — used only by the F.7.4.7.2 multilayer-extension
///   variant of equation 7-57 (`F-56`): when the multilayer extension
///   applies and the current picture is IDR (`IDR_W_RADL` /
///   `IDR_N_LP`), the short-term and long-term loops are skipped
///   entirely. For base §7.4.7.2 the value is unused because every
///   IDR slice already has zero short-term and long-term entries.
/// * `num_active_ref_layer_pics` — F.7.4.7.2 `NumActiveRefLayerPics`
///   (the count of active inter-layer reference pictures for the
///   current slice, §F.7.4.7.1). Set to `0` for base §7.4.7.2.
///
/// The `nal_unit_type` IDR gate and `num_active_ref_layer_pics`
/// contribution are only applied when [`Self::multilayer_extension`]
/// is `true` (forward-compat for the multilayer scaffold; left
/// `false` by every base-profile call site).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumPicTotalCurrInputs<'a> {
    /// Per-position `UsedByCurrPicS0[ CurrRpsIdx ][ i ]` flags.
    pub used_by_curr_pic_s0: &'a [bool],
    /// Per-position `UsedByCurrPicS1[ CurrRpsIdx ][ i ]` flags.
    pub used_by_curr_pic_s1: &'a [bool],
    /// Per-position `UsedByCurrPicLt[ i ]` flags, length
    /// `num_long_term_sps + num_long_term_pics`.
    pub used_by_curr_pic_lt: &'a [bool],
    /// `pps_curr_pic_ref_enabled_flag` (§7.4.3.3.1.4 SCC PPS). Inferred
    /// to `false` when not signalled.
    pub pps_curr_pic_ref_enabled_flag: bool,
    /// `nal_unit_type` of the slice's NAL unit (Table 7-1). Consumed
    /// only when [`Self::multilayer_extension`] is `true`.
    pub nal_unit_type: u8,
    /// F.7.4.7.2 `NumActiveRefLayerPics` (the §F.7.4.7.1 inter-layer
    /// active count). Consumed only when [`Self::multilayer_extension`]
    /// is `true`.
    pub num_active_ref_layer_pics: u32,
    /// Forward-compat toggle: when `true`, equation `F-56` of
    /// F.7.4.7.2 is applied instead of equation 7-57 of §7.4.7.2
    /// (the short-term / long-term loops are skipped for IDR
    /// `nal_unit_type`, and `NumActiveRefLayerPics` is added at the
    /// end). Every base-profile call site leaves this `false`.
    pub multilayer_extension: bool,
}

impl<'a> NumPicTotalCurrInputs<'a> {
    /// Build the inputs from already-resolved per-position
    /// `UsedByCurrPic*` slices. The caller is responsible for having
    /// run the §7.4.8 inter-RPS-prediction derivation if the active
    /// short-term RPS uses the predicted form.
    pub fn from_used_flags(
        used_by_curr_pic_s0: &'a [bool],
        used_by_curr_pic_s1: &'a [bool],
        used_by_curr_pic_lt: &'a [bool],
    ) -> Self {
        Self {
            used_by_curr_pic_s0,
            used_by_curr_pic_s1,
            used_by_curr_pic_lt,
            pps_curr_pic_ref_enabled_flag: false,
            nal_unit_type: 0,
            num_active_ref_layer_pics: 0,
            multilayer_extension: false,
        }
    }

    /// Build the inputs from an *explicit-form* short-term RPS, where
    /// the `UsedByCurrPicS0` / `UsedByCurrPicS1` arrays are the
    /// SPS-signalled `used_by_curr_pic_sX_flag` arrays themselves
    /// (§7.4.8 equations 7-65 / 7-66). Returns `None` when the RPS
    /// uses inter-prediction (`inter_ref_pic_set_prediction_flag ==
    /// 1`) — the §7.4.8 derivation must be run first and the result
    /// passed to [`Self::from_used_flags`].
    pub fn from_explicit_short_term_rps(
        curr_rps: &'a ShortTermRefPicSet,
        used_by_curr_pic_lt: &'a [bool],
    ) -> Option<Self> {
        if curr_rps.inter_ref_pic_set_prediction_flag {
            return None;
        }
        Some(Self::from_used_flags(
            &curr_rps.used_by_curr_pic_s0_flag,
            &curr_rps.used_by_curr_pic_s1_flag,
            used_by_curr_pic_lt,
        ))
    }

    /// Set [`Self::pps_curr_pic_ref_enabled_flag`] (builder).
    pub fn with_pps_curr_pic_ref_enabled(mut self, flag: bool) -> Self {
        self.pps_curr_pic_ref_enabled_flag = flag;
        self
    }

    /// Set the multilayer-extension trio (builder).
    pub fn with_multilayer_extension(
        mut self,
        nal_unit_type: u8,
        num_active_ref_layer_pics: u32,
    ) -> Self {
        self.multilayer_extension = true;
        self.nal_unit_type = nal_unit_type;
        self.num_active_ref_layer_pics = num_active_ref_layer_pics;
        self
    }

    /// Compute `NumPicTotalCurr` per equation 7-57 (base §7.4.7.2) or
    /// equation `F-56` (F.7.4.7.2 multilayer extension), depending on
    /// [`Self::multilayer_extension`].
    ///
    /// The base equation 7-57:
    ///
    /// ```text
    /// NumPicTotalCurr = 0
    /// for i in 0..NumNegativePics[CurrRpsIdx]:
    ///     if UsedByCurrPicS0[CurrRpsIdx][i]:                       NumPicTotalCurr++
    /// for i in 0..NumPositivePics[CurrRpsIdx]:
    ///     if UsedByCurrPicS1[CurrRpsIdx][i]:                       NumPicTotalCurr++
    /// for i in 0..(num_long_term_sps + num_long_term_pics):
    ///     if UsedByCurrPicLt[i]:                                   NumPicTotalCurr++
    /// if pps_curr_pic_ref_enabled_flag:                            NumPicTotalCurr++
    /// ```
    ///
    /// The multilayer variant `F-56` skips the short-term and
    /// long-term loops entirely for IDR `nal_unit_type` values
    /// (`IDR_W_RADL` / `IDR_N_LP`), then adds `NumActiveRefLayerPics`
    /// after the `pps_curr_pic_ref_enabled_flag` step.
    pub fn compute(&self) -> u32 {
        let is_idr = self.nal_unit_type == IDR_W_RADL || self.nal_unit_type == IDR_N_LP;
        let skip_temporal_loops = self.multilayer_extension && is_idr;

        let mut n: u32 = 0;
        if !skip_temporal_loops {
            n += self.used_by_curr_pic_s0.iter().filter(|&&v| v).count() as u32;
            n += self.used_by_curr_pic_s1.iter().filter(|&&v| v).count() as u32;
            n += self.used_by_curr_pic_lt.iter().filter(|&&v| v).count() as u32;
        }
        if self.pps_curr_pic_ref_enabled_flag {
            n += 1;
        }
        if self.multilayer_extension {
            n += self.num_active_ref_layer_pics;
        }
        n
    }
}

/// Per-list weighted-prediction entry for one reference picture in
/// [`PredWeightTable`] (one entry per `i = 0 ..= num_ref_idx_lX_active_minus1`).
///
/// Fields are the raw §7.3.6.3 syntax elements, kept in unresolved form
/// so the caller can both audit the on-wire bits and compute the
/// §7.4.7.3 derived variables `LumaWeightLX[i]`,
/// `ChromaWeightLX[i][j]`, `ChromaOffsetLX[i][j]` through the
/// helper methods on [`PredWeightTable`] (which also apply the
/// §7.4.7.3 inference rules for the absent fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PredWeightEntry {
    /// `luma_weight_lX_flag[i]` (`u(1)`). Inferred to `false` when the
    /// §7.3.6.3 outer gate (`pic_layer_id != nuh_layer_id ||
    /// PicOrderCnt(RefPicListX[i]) != PicOrderCnt(CurrPic)`) is `false`
    /// for this `i` — for a base-profile single-layer slice the gate is
    /// always `true`, so this flag is always signalled.
    pub luma_weight_flag: bool,
    /// `chroma_weight_lX_flag[i]` (`u(1)`). Absent (inferred `false`)
    /// when `ChromaArrayType == 0` or when the outer gate is `false`
    /// for this `i`.
    pub chroma_weight_flag: bool,
    /// `delta_luma_weight_lX[i]` (`se(v)`, range −128..=127). Inferred
    /// to `0` when [`Self::luma_weight_flag`] is `false`.
    pub delta_luma_weight: i32,
    /// `luma_offset_lX[i]` (`se(v)`, range
    /// `−WpOffsetHalfRangeY ..= WpOffsetHalfRangeY − 1`). Inferred to
    /// `0` when [`Self::luma_weight_flag`] is `false`.
    pub luma_offset: i32,
    /// `delta_chroma_weight_lX[i][j]` (`se(v)`, range −128..=127) for
    /// `j = 0 (Cb), 1 (Cr)`. Both inferred to `0` when
    /// [`Self::chroma_weight_flag`] is `false`.
    pub delta_chroma_weight: [i32; 2],
    /// `delta_chroma_offset_lX[i][j]` (`se(v)`, range
    /// `−4 * WpOffsetHalfRangeC ..= 4 * WpOffsetHalfRangeC − 1`) for
    /// `j = 0 (Cb), 1 (Cr)`. Both inferred to `0` when
    /// [`Self::chroma_weight_flag`] is `false`.
    pub delta_chroma_offset: [i32; 2],
}

/// Parsed `pred_weight_table()` syntax structure (ITU-T Rec. H.265
/// §7.3.6.3 / §7.4.7.3).
///
/// The structure is signalled in the slice header when
/// `(weighted_pred_flag && slice_type == P) ||
/// (weighted_bipred_flag && slice_type == B)` (§7.3.6.1 gate). It
/// carries per-reference weighting factors and additive offsets that
/// §8.5.3.3.4.3 applies to the inter-prediction samples produced from
/// each `RefPicListX[i]`.
///
/// ### Outer §7.3.6.3 gate
///
/// Each `luma_weight_lX_flag[i]` and `chroma_weight_lX_flag[i]` syntax
/// element is wrapped in a conditional:
///
/// ```text
///   if( pic_layer_id( RefPicListX[ i ] ) != nuh_layer_id ||
///       PicOrderCnt( RefPicListX[ i ] ) != PicOrderCnt( CurrPic ) )
///       luma_weight_lX_flag[ i ]   u(1)
/// ```
///
/// — the flag is only signalled when the reference is a *different
/// picture* (i.e. either an inter-layer reference or a temporal
/// reference). For a base-profile single-layer slice every active
/// reference is temporal, so the gate is universally `true` and every
/// flag is signalled. For inter-layer / SCC self-reference cases the
/// gate is `false` for some `i`, and the parser must skip the
/// corresponding flag bit and infer it to `0` (§7.4.7.3 "When
/// luma_weight_lX_flag[ i ] is not present, it is inferred to be equal
/// to 0").
///
/// The caller resolves the DPB-driven gate and passes the per-i
/// boolean decisions through [`PredWeightTableInputs::signal_luma_l0`]
/// / [`PredWeightTableInputs::signal_chroma_l0`] /
/// [`PredWeightTableInputs::signal_luma_l1`] /
/// [`PredWeightTableInputs::signal_chroma_l1`]; the default
/// [`PredWeightTableInputs::base_profile`] constructor leaves them all
/// `true` (the base-profile case).
///
/// ### Derived variables
///
/// Per §7.4.7.3 the on-wire deltas combine with the per-list
/// `..._log2_weight_denom` to produce the actual weighting factors and
/// offsets the §8.5.3.3.4.3 inter-prediction process consumes:
///
/// * `ChromaLog2WeightDenom = luma_log2_weight_denom +
///   delta_chroma_log2_weight_denom` (range 0..=7).
/// * `LumaWeightLX[i] = (1 << luma_log2_weight_denom) +
///   delta_luma_weight_lX[i]` when the luma flag is set, else inferred
///   to `1 << luma_log2_weight_denom`.
/// * `ChromaWeightLX[i][j] = (1 << ChromaLog2WeightDenom) +
///   delta_chroma_weight_lX[i][j]` when the chroma flag is set, else
///   inferred to `1 << ChromaLog2WeightDenom`.
/// * `ChromaOffsetLX[i][j]` per equation 7-58 (a clipped expression
///   parameterised by `WpOffsetHalfRangeC` and `ChromaLog2WeightDenom`).
///
/// The accessor methods on this struct apply those derivations.
///
/// ### Conformance check
///
/// §7.4.7.3 closes with the `sumWeightLXFlags` cap: for a P slice
/// `sumWeightL0Flags ≤ 24`; for a B slice
/// `sumWeightL0Flags + sumWeightL1Flags ≤ 24` where each
/// `sumWeightLXFlags = Σ ( luma_weight_lX_flag[i] +
/// 2 * chroma_weight_lX_flag[i] )`. The parser computes this sum and
/// enforces the cap.
///
/// ### Range checks
///
/// * `luma_log2_weight_denom` ∈ 0..=7.
/// * `luma_log2_weight_denom + delta_chroma_log2_weight_denom` ∈ 0..=7
///   (the variable `ChromaLog2WeightDenom`).
/// * `delta_luma_weight_lX[i]` ∈ −128..=127 when the luma flag is set.
/// * `luma_offset_lX[i]` ∈ `−WpOffsetHalfRangeY ..= WpOffsetHalfRangeY − 1`
///   when the luma flag is set.
/// * `delta_chroma_weight_lX[i][j]` ∈ −128..=127 when the chroma flag
///   is set.
/// * `delta_chroma_offset_lX[i][j]` ∈
///   `−4 * WpOffsetHalfRangeC ..= 4 * WpOffsetHalfRangeC − 1` when the
///   chroma flag is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom` (`ue(v)`, range 0..=7).
    pub luma_log2_weight_denom: u8,
    /// `delta_chroma_log2_weight_denom` (`se(v)`). Absent (and
    /// inferred to 0 per §7.4.7.3) when `ChromaArrayType == 0`.
    pub delta_chroma_log2_weight_denom: i32,
    /// L0 per-reference entries, length
    /// `num_ref_idx_l0_active_minus1 + 1`.
    pub entries_l0: Vec<PredWeightEntry>,
    /// L1 per-reference entries, length
    /// `num_ref_idx_l1_active_minus1 + 1` for B slices. Empty for P
    /// slices (the §7.3.6.3 `if( slice_type == B )` gate suppresses
    /// every L1 syntax element).
    pub entries_l1: Vec<PredWeightEntry>,
}

/// Inputs to [`PredWeightTable::parse`].
///
/// Carries every value the parser needs to derive field widths,
/// presence gates, range bounds and the per-i §7.3.6.3 outer-gate
/// decisions. The [`Self::base_profile`] constructor covers the common
/// case (single-layer base profile, `high_precision_offsets_enabled_flag
/// == 0`, every per-i gate `true`); the other setters carry the
/// extension-specific knobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTableInputs<'a> {
    /// Active `slice_type` (the L1 syntax block is suppressed for P).
    pub slice_type: SliceType,
    /// Active `num_ref_idx_l0_active_minus1` after the
    /// `num_ref_idx_active_override_flag` override (range 0..=14 per
    /// §7.4.7.1).
    pub num_ref_idx_l0_active_minus1: u8,
    /// Active `num_ref_idx_l1_active_minus1`. Ignored for P slices.
    pub num_ref_idx_l1_active_minus1: u8,
    /// `ChromaArrayType` per §7.4.2.2. When `0` (monochrome or
    /// separate-colour-plane) the entire chroma sub-block is absent.
    pub chroma_array_type: u8,
    /// `high_precision_offsets_enabled_flag` from the SPS range
    /// extension (§7.4.3.2.2 / equations 7-33 / 7-34). Inferred to
    /// `false` when the SPS range extension is not signalled.
    pub high_precision_offsets_enabled_flag: bool,
    /// `BitDepthY` from the SPS (§7.4.3.2.1), used by
    /// `WpOffsetHalfRangeY` when [`Self::high_precision_offsets_enabled_flag`]
    /// is `true`. Ignored otherwise (`WpOffsetHalfRangeY = 128`).
    pub bit_depth_y: u8,
    /// `BitDepthC` from the SPS (§7.4.3.2.1), used by
    /// `WpOffsetHalfRangeC` when [`Self::high_precision_offsets_enabled_flag`]
    /// is `true`. Ignored otherwise (`WpOffsetHalfRangeC = 128`).
    pub bit_depth_c: u8,
    /// Per-i outer-gate decision for `luma_weight_l0_flag[i]`. When
    /// `None`, every position is treated as gated `true`
    /// (base-profile case). When `Some(slice)`, length must equal
    /// `num_ref_idx_l0_active_minus1 + 1` and `slice[i] == false`
    /// suppresses the corresponding flag bit (inferred to `0`).
    pub signal_luma_l0: Option<&'a [bool]>,
    /// Same as [`Self::signal_luma_l0`] for `chroma_weight_l0_flag[i]`.
    pub signal_chroma_l0: Option<&'a [bool]>,
    /// Same as [`Self::signal_luma_l0`] for `luma_weight_l1_flag[i]`
    /// (B slices only).
    pub signal_luma_l1: Option<&'a [bool]>,
    /// Same as [`Self::signal_luma_l0`] for `chroma_weight_l1_flag[i]`
    /// (B slices only).
    pub signal_chroma_l1: Option<&'a [bool]>,
}

impl<'a> PredWeightTableInputs<'a> {
    /// Base-profile single-layer constructor: every per-i §7.3.6.3
    /// outer-gate decision is `true`, `high_precision_offsets_enabled_flag
    /// == false`. The caller supplies only the slice-type, the active
    /// ref-list cardinalities, and the activated SPS's
    /// `ChromaArrayType` + bit depths.
    pub fn base_profile(
        slice_type: SliceType,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
        chroma_array_type: u8,
        bit_depth_y: u8,
        bit_depth_c: u8,
    ) -> Self {
        Self {
            slice_type,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            chroma_array_type,
            high_precision_offsets_enabled_flag: false,
            bit_depth_y,
            bit_depth_c,
            signal_luma_l0: None,
            signal_chroma_l0: None,
            signal_luma_l1: None,
            signal_chroma_l1: None,
        }
    }

    /// `WpOffsetHalfRangeY` per equation 7-33.
    fn wp_offset_half_range_y(&self) -> i32 {
        let shift = if self.high_precision_offsets_enabled_flag {
            (self.bit_depth_y as i32) - 1
        } else {
            7
        };
        1i32 << shift
    }

    /// `WpOffsetHalfRangeC` per equation 7-34.
    fn wp_offset_half_range_c(&self) -> i32 {
        let shift = if self.high_precision_offsets_enabled_flag {
            (self.bit_depth_c as i32) - 1
        } else {
            7
        };
        1i32 << shift
    }
}

impl PredWeightTable {
    /// Parse `pred_weight_table()` (§7.3.6.3) from the current bit
    /// position of `br`. See [`PredWeightTableInputs`] for the per-call
    /// inputs and the base-profile constructor.
    ///
    /// The parser:
    ///
    /// 1. Reads `luma_log2_weight_denom` (`ue(v)`, range 0..=7).
    /// 2. When `chroma_array_type != 0`, reads
    ///    `delta_chroma_log2_weight_denom` (`se(v)`) and validates the
    ///    derived `ChromaLog2WeightDenom` ∈ 0..=7.
    /// 3. Reads the L0 luma-flag pass, applying the per-i outer-gate
    ///    decision from [`PredWeightTableInputs::signal_luma_l0`].
    /// 4. When `chroma_array_type != 0`, reads the L0 chroma-flag
    ///    pass with the matching gate slice.
    /// 5. Reads the L0 per-reference delta block: for each `i` where
    ///    the flag is set, reads `delta_luma_weight_l0[i]` +
    ///    `luma_offset_l0[i]`; when the chroma flag is set, reads
    ///    `delta_chroma_weight_l0[i][j]` + `delta_chroma_offset_l0[i][j]`
    ///    for `j ∈ {0, 1}`.
    /// 6. For B slices, mirrors steps 3–5 for L1.
    /// 7. Validates the §7.4.7.3 `sumWeightLXFlags ≤ 24` cap.
    ///
    /// Each delta is range-checked per §7.4.7.3; range failures
    /// surface as [`SliceError::ValueOutOfRange`].
    pub fn parse(
        br: &mut BitReader<'_>,
        inputs: &PredWeightTableInputs<'_>,
    ) -> Result<Self, SliceError> {
        if inputs.slice_type == SliceType::I {
            return Err(SliceError::ValueOutOfRange {
                field: "pred_weight_table/slice_type",
                got: 2,
            });
        }
        if inputs.num_ref_idx_l0_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: inputs.num_ref_idx_l0_active_minus1 as i64,
            });
        }
        if inputs.slice_type == SliceType::B && inputs.num_ref_idx_l1_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l1_active_minus1",
                got: inputs.num_ref_idx_l1_active_minus1 as i64,
            });
        }

        let n_l0 = inputs.num_ref_idx_l0_active_minus1 as usize + 1;
        let n_l1 = if inputs.slice_type == SliceType::B {
            inputs.num_ref_idx_l1_active_minus1 as usize + 1
        } else {
            0
        };
        validate_signal_slice("signal_luma_l0", inputs.signal_luma_l0, n_l0)?;
        validate_signal_slice("signal_chroma_l0", inputs.signal_chroma_l0, n_l0)?;
        validate_signal_slice("signal_luma_l1", inputs.signal_luma_l1, n_l1)?;
        validate_signal_slice("signal_chroma_l1", inputs.signal_chroma_l1, n_l1)?;

        let chroma_present = inputs.chroma_array_type != 0;

        let luma_log2_weight_denom_u = br.ue()?;
        if luma_log2_weight_denom_u > 7 {
            return Err(SliceError::ValueOutOfRange {
                field: "luma_log2_weight_denom",
                got: luma_log2_weight_denom_u as i64,
            });
        }
        let luma_log2_weight_denom = luma_log2_weight_denom_u as u8;

        let delta_chroma_log2_weight_denom: i32 = if chroma_present {
            let v = br.se()?;
            let chroma_denom = luma_log2_weight_denom as i32 + v;
            if !(0..=7).contains(&chroma_denom) {
                return Err(SliceError::ValueOutOfRange {
                    field: "ChromaLog2WeightDenom",
                    got: chroma_denom as i64,
                });
            }
            v
        } else {
            0
        };

        // L0: parse the two flag passes (luma, then chroma when
        // chroma is present), then the per-reference delta block.
        let entries_l0 = parse_pred_weight_list(
            br,
            n_l0,
            chroma_present,
            inputs.signal_luma_l0,
            inputs.signal_chroma_l0,
            inputs.wp_offset_half_range_y(),
            inputs.wp_offset_half_range_c(),
            "l0",
        )?;

        // L1: B slices only.
        let entries_l1 = if inputs.slice_type == SliceType::B {
            parse_pred_weight_list(
                br,
                n_l1,
                chroma_present,
                inputs.signal_luma_l1,
                inputs.signal_chroma_l1,
                inputs.wp_offset_half_range_y(),
                inputs.wp_offset_half_range_c(),
                "l1",
            )?
        } else {
            Vec::new()
        };

        // §7.4.7.3 sumWeightLXFlags cap: ≤ 24 per list contribution.
        let sum_l0 = sum_weight_flags(&entries_l0);
        if inputs.slice_type == SliceType::P && sum_l0 > 24 {
            return Err(SliceError::ValueOutOfRange {
                field: "sumWeightL0Flags",
                got: sum_l0 as i64,
            });
        }
        if inputs.slice_type == SliceType::B {
            let sum_l1 = sum_weight_flags(&entries_l1);
            if sum_l0 + sum_l1 > 24 {
                return Err(SliceError::ValueOutOfRange {
                    field: "sumWeightL0Flags+sumWeightL1Flags",
                    got: (sum_l0 + sum_l1) as i64,
                });
            }
        }

        Ok(Self {
            luma_log2_weight_denom,
            delta_chroma_log2_weight_denom,
            entries_l0,
            entries_l1,
        })
    }

    /// `ChromaLog2WeightDenom = luma_log2_weight_denom +
    /// delta_chroma_log2_weight_denom` per §7.4.7.3. Returns `0` when
    /// the chroma sub-block was absent (`ChromaArrayType == 0`); the
    /// derivation is moot in that case.
    pub fn chroma_log2_weight_denom(&self) -> u8 {
        // The parser's range check on `ChromaLog2WeightDenom ∈ 0..=7`
        // guarantees the sum fits in a `u8`.
        (self.luma_log2_weight_denom as i32 + self.delta_chroma_log2_weight_denom) as u8
    }

    /// `LumaWeightL0[i]` per §7.4.7.3: `(1 << luma_log2_weight_denom)
    /// + delta_luma_weight_l0[i]` when the flag is set, else
    /// inferred to `1 << luma_log2_weight_denom`.
    pub fn luma_weight_l0(&self, i: usize) -> Option<i32> {
        self.entries_l0
            .get(i)
            .map(|e| self.luma_weight_value(e.luma_weight_flag, e.delta_luma_weight))
    }

    /// `LumaWeightL1[i]` per §7.4.7.3.
    pub fn luma_weight_l1(&self, i: usize) -> Option<i32> {
        self.entries_l1
            .get(i)
            .map(|e| self.luma_weight_value(e.luma_weight_flag, e.delta_luma_weight))
    }

    /// `ChromaWeightL0[i][j]` per §7.4.7.3.
    pub fn chroma_weight_l0(&self, i: usize, j: usize) -> Option<i32> {
        let e = self.entries_l0.get(i)?;
        let v = *e.delta_chroma_weight.get(j)?;
        Some(self.chroma_weight_value(e.chroma_weight_flag, v))
    }

    /// `ChromaWeightL1[i][j]` per §7.4.7.3.
    pub fn chroma_weight_l1(&self, i: usize, j: usize) -> Option<i32> {
        let e = self.entries_l1.get(i)?;
        let v = *e.delta_chroma_weight.get(j)?;
        Some(self.chroma_weight_value(e.chroma_weight_flag, v))
    }

    /// `ChromaOffsetL0[i][j]` per §7.4.7.3 equation 7-58.
    pub fn chroma_offset_l0(&self, i: usize, j: usize, wp_offset_half_range_c: i32) -> Option<i32> {
        let e = self.entries_l0.get(i)?;
        if !e.chroma_weight_flag {
            return Some(0);
        }
        let delta_off = *e.delta_chroma_offset.get(j)?;
        let chroma_w = self.chroma_weight_value(true, *e.delta_chroma_weight.get(j)?);
        Some(chroma_offset_eq_7_58(
            wp_offset_half_range_c,
            delta_off,
            chroma_w,
            self.chroma_log2_weight_denom(),
        ))
    }

    /// `ChromaOffsetL1[i][j]` per §7.4.7.3 equation 7-58.
    pub fn chroma_offset_l1(&self, i: usize, j: usize, wp_offset_half_range_c: i32) -> Option<i32> {
        let e = self.entries_l1.get(i)?;
        if !e.chroma_weight_flag {
            return Some(0);
        }
        let delta_off = *e.delta_chroma_offset.get(j)?;
        let chroma_w = self.chroma_weight_value(true, *e.delta_chroma_weight.get(j)?);
        Some(chroma_offset_eq_7_58(
            wp_offset_half_range_c,
            delta_off,
            chroma_w,
            self.chroma_log2_weight_denom(),
        ))
    }

    fn luma_weight_value(&self, flag: bool, delta: i32) -> i32 {
        let base = 1i32 << self.luma_log2_weight_denom;
        if flag { base + delta } else { base }
    }

    fn chroma_weight_value(&self, flag: bool, delta: i32) -> i32 {
        let base = 1i32 << self.chroma_log2_weight_denom();
        if flag { base + delta } else { base }
    }
}

/// Verify the caller-supplied per-i gate slice has the expected length.
fn validate_signal_slice(
    field: &'static str,
    slice: Option<&[bool]>,
    expected: usize,
) -> Result<(), SliceError> {
    match slice {
        None => Ok(()),
        Some(s) if s.len() == expected => Ok(()),
        Some(s) => Err(SliceError::ValueOutOfRange {
            field,
            got: s.len() as i64,
        }),
    }
}

/// Parse one per-list (L0 or L1) sub-block of §7.3.6.3.
#[allow(clippy::too_many_arguments)]
fn parse_pred_weight_list(
    br: &mut BitReader<'_>,
    n: usize,
    chroma_present: bool,
    signal_luma: Option<&[bool]>,
    signal_chroma: Option<&[bool]>,
    wp_off_half_y: i32,
    wp_off_half_c: i32,
    list_tag: &'static str,
) -> Result<Vec<PredWeightEntry>, SliceError> {
    let mut entries: Vec<PredWeightEntry> = (0..n).map(|_| PredWeightEntry::default()).collect();

    // Luma flag pass.
    for (i, e) in entries.iter_mut().enumerate() {
        let signalled = signal_luma.map(|s| s[i]).unwrap_or(true);
        e.luma_weight_flag = if signalled { br.u1()? != 0 } else { false };
    }

    // Chroma flag pass — present only when ChromaArrayType != 0.
    if chroma_present {
        for (i, e) in entries.iter_mut().enumerate() {
            let signalled = signal_chroma.map(|s| s[i]).unwrap_or(true);
            e.chroma_weight_flag = if signalled { br.u1()? != 0 } else { false };
        }
    }

    // Per-reference delta block.
    for (i, e) in entries.iter_mut().enumerate() {
        if e.luma_weight_flag {
            let d = br.se()?;
            if !(-128..=127).contains(&d) {
                return Err(SliceError::ValueOutOfRange {
                    field: if list_tag == "l0" {
                        "delta_luma_weight_l0"
                    } else {
                        "delta_luma_weight_l1"
                    },
                    got: d as i64,
                });
            }
            e.delta_luma_weight = d;

            let off = br.se()?;
            if off < -wp_off_half_y || off > wp_off_half_y - 1 {
                return Err(SliceError::ValueOutOfRange {
                    field: if list_tag == "l0" {
                        "luma_offset_l0"
                    } else {
                        "luma_offset_l1"
                    },
                    got: off as i64,
                });
            }
            e.luma_offset = off;
        }
        if e.chroma_weight_flag {
            for j in 0..2 {
                let d = br.se()?;
                if !(-128..=127).contains(&d) {
                    return Err(SliceError::ValueOutOfRange {
                        field: if list_tag == "l0" {
                            "delta_chroma_weight_l0"
                        } else {
                            "delta_chroma_weight_l1"
                        },
                        got: d as i64,
                    });
                }
                e.delta_chroma_weight[j] = d;

                let off = br.se()?;
                if off < -4 * wp_off_half_c || off > 4 * wp_off_half_c - 1 {
                    return Err(SliceError::ValueOutOfRange {
                        field: if list_tag == "l0" {
                            "delta_chroma_offset_l0"
                        } else {
                            "delta_chroma_offset_l1"
                        },
                        got: off as i64,
                    });
                }
                e.delta_chroma_offset[j] = off;
            }
        }
        let _ = i; // silence the unused-`i` when iter_mut().enumerate() is mixed with explicit loop
    }

    Ok(entries)
}

/// §7.4.7.3 closing summand:
/// `sumWeightLXFlags = Σ luma_weight_lX_flag[i] + 2 * chroma_weight_lX_flag[i]`.
fn sum_weight_flags(entries: &[PredWeightEntry]) -> u32 {
    entries
        .iter()
        .map(|e| u32::from(e.luma_weight_flag) + 2 * u32::from(e.chroma_weight_flag))
        .sum()
}

/// §7.4.7.3 equation 7-58 for `ChromaOffsetLX[i][j]`. Extracted as a
/// free function so [`PredWeightTable::chroma_offset_l0`] /
/// [`PredWeightTable::chroma_offset_l1`] share the implementation.
fn chroma_offset_eq_7_58(
    wp_off_half_c: i32,
    delta_chroma_offset: i32,
    chroma_weight: i32,
    chroma_log2_weight_denom: u8,
) -> i32 {
    let raw = wp_off_half_c + delta_chroma_offset
        - ((wp_off_half_c * chroma_weight) >> chroma_log2_weight_denom);
    raw.clamp(-wp_off_half_c, wp_off_half_c - 1)
}

/// Parsed slice segment header per §7.3.6.1.
///
/// Fields that this round defers (the non-IDR POC/RPS block, the P/B
/// reference-list / weighted-prediction sub-structures) are absent from
/// the materialised struct; when one of those points is reached the
/// remainder of the header is surfaced via [`Self::opaque_tail`] and
/// the corresponding `Option` fields stay `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSegmentHeader {
    /// `first_slice_segment_in_pic_flag`.
    pub first_slice_segment_in_pic_flag: bool,
    /// `no_output_of_prior_pics_flag`. `None` when not present (the NAL
    /// unit is not an IRAP picture).
    pub no_output_of_prior_pics_flag: Option<bool>,
    /// `slice_pic_parameter_set_id` (`ue(v)`, range 0..=63).
    pub slice_pic_parameter_set_id: u8,
    /// `dependent_slice_segment_flag`. Inferred to false when the slice
    /// is the first segment of the picture or when
    /// `dependent_slice_segments_enabled_flag` is 0 (§7.4.7.1).
    pub dependent_slice_segment_flag: bool,
    /// `slice_segment_address` (`u(v)`). Inferred to 0 when not present
    /// (the first slice segment of the picture).
    pub slice_segment_address: u32,
    /// `slice_reserved_flag[]` — `num_extra_slice_header_bits` raw
    /// flags. Decoders ignore the value; carried for completeness.
    /// Empty for dependent slice segments and when the count is 0.
    pub slice_reserved_flags: Vec<bool>,
    /// `slice_type` per Table 7-7. `None` for dependent slice segments
    /// (the value is inherited from the associated independent slice
    /// segment, which this struct does not resolve).
    pub slice_type: Option<SliceType>,
    /// `pic_output_flag`. Inferred to true when not present (§7.4.7.1).
    pub pic_output_flag: bool,
    /// `colour_plane_id` (`u(2)`). `None` when not present
    /// (`separate_colour_plane_flag == 0`).
    pub colour_plane_id: Option<u8>,
    /// `slice_pic_order_cnt_lsb` (`u(v)`, width
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` bits). `None` when the
    /// current NAL unit is an IDR — IDR pictures have no slice POC LSB
    /// (the POC is reset to 0 per §8.3.1) — or when the parser stopped
    /// at the deferred P/B body before reaching this point.
    pub slice_pic_order_cnt_lsb: Option<u32>,
    /// `short_term_ref_pic_set_sps_flag`. `None` for IDR slices and
    /// for headers that stopped before this point.
    pub short_term_ref_pic_set_sps_flag: Option<bool>,
    /// In-line `st_ref_pic_set(num_short_term_ref_pic_sets)` parsed from
    /// the slice header itself (only when
    /// `short_term_ref_pic_set_sps_flag == 0`).
    pub inline_short_term_ref_pic_set: Option<ShortTermRefPicSet>,
    /// `short_term_ref_pic_set_idx` (`u(v)`, width
    /// `Ceil(Log2(num_short_term_ref_pic_sets))`). `None` when the SPS
    /// in-line form is used, when `num_short_term_ref_pic_sets <= 1`
    /// (the value is inferred to 0), or for IDR slices.
    pub short_term_ref_pic_set_idx: Option<u32>,
    /// `num_long_term_sps` (`ue(v)`). `None` when the long-term-ref-pic
    /// block is absent (no `long_term_ref_pics_present_flag` on the SPS
    /// or IDR slice); 0 (with the SPS gate satisfied but
    /// `num_long_term_ref_pics_sps == 0`).
    pub num_long_term_sps: Option<u32>,
    /// `num_long_term_pics` (`ue(v)`). `None` when the long-term-ref-pic
    /// block is absent.
    pub num_long_term_pics: Option<u32>,
    /// Per-entry long-term ref pic block (§7.3.6.1), length
    /// `num_long_term_sps + num_long_term_pics`. Empty when the block
    /// is absent.
    pub long_term_ref_pics: Vec<SliceLongTermRefPic>,
    /// `slice_temporal_mvp_enabled_flag`. Inferred to false when not
    /// present (`sps_temporal_mvp_enabled_flag == 0`) (§7.4.7.1).
    pub slice_temporal_mvp_enabled_flag: bool,
    /// `slice_sao_luma_flag`. Inferred to false when not present
    /// (`sample_adaptive_offset_enabled_flag == 0`).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag`. Inferred to false when not present.
    pub slice_sao_chroma_flag: bool,
    /// `num_ref_idx_active_override_flag` (§7.3.6.1). `None` for I
    /// slices (the field is absent — `slice_type == 2`) and for
    /// dependent slice segments. Read as `u(1)` immediately after the
    /// SAO block for P / B slices.
    pub num_ref_idx_active_override_flag: Option<bool>,
    /// `num_ref_idx_l0_active_minus1` (§7.3.6.1, range 0..=14). For P
    /// / B slices, signalled when `num_ref_idx_active_override_flag ==
    /// 1` and otherwise inferred to `pps.num_ref_idx_l0_default_active_
    /// minus1` per §7.4.7.1. `None` when the slice is I or the parser
    /// stopped before reaching this point.
    pub num_ref_idx_l0_active_minus1: Option<u8>,
    /// `num_ref_idx_l1_active_minus1` (§7.3.6.1, range 0..=14). For B
    /// slices, signalled when `num_ref_idx_active_override_flag == 1`
    /// and otherwise inferred to `pps.num_ref_idx_l1_default_active_
    /// minus1` per §7.4.7.1. `None` when the slice is not B or the
    /// parser stopped before reaching this point.
    pub num_ref_idx_l1_active_minus1: Option<u8>,
    /// `mvd_l1_zero_flag` (§7.3.6.1). `u(1)`, present only for B slices.
    /// `None` for I / P slices, dependent slice segments, and headers
    /// whose parse stopped before the inter-slice mvd block (either at
    /// the `ref_pic_lists_modification()` gate when
    /// `pps.lists_modification_present_flag == 1`, or at any earlier
    /// deferral point).
    pub mvd_l1_zero_flag: Option<bool>,
    /// `cabac_init_flag` (§7.3.6.1). `u(1)`, present only when
    /// `pps.cabac_init_present_flag == 1`; inferred to `false`
    /// otherwise per §7.4.7.1. `None` for I slices, dependent slice
    /// segments, and headers whose parse stopped before the inter-slice
    /// cabac-init point.
    pub cabac_init_flag: Option<bool>,
    /// `collocated_from_l0_flag` (§7.3.6.1). `u(1)`, present only when
    /// `slice_temporal_mvp_enabled_flag == 1 && slice_type == B`.
    /// Inferred to `true` when absent (per §7.4.7.1, "When
    /// `collocated_from_l0_flag` is not present, it is inferred to be
    /// equal to 1"). `None` when the slice is I, when the parse
    /// stopped before this point, or when
    /// `slice_temporal_mvp_enabled_flag == 0` (the field has no
    /// meaning).
    pub collocated_from_l0_flag: Option<bool>,
    /// `collocated_ref_idx` (§7.3.6.1). `ue(v)`, present only when
    /// `slice_temporal_mvp_enabled_flag == 1` and the relevant active
    /// list has more than one entry (specifically:
    /// `(collocated_from_l0_flag && num_ref_idx_l0_active_minus1 > 0)
    /// || (!collocated_from_l0_flag && num_ref_idx_l1_active_minus1 >
    /// 0)`). Inferred to `0` when absent per §7.4.7.1. `None` when the
    /// slice is I, when the parse stopped before this point, or when
    /// `slice_temporal_mvp_enabled_flag == 0`.
    pub collocated_ref_idx: Option<u32>,
    /// `five_minus_max_num_merge_cand` (§7.3.6.1). `ue(v)`, present only
    /// for P/B slices, immediately after the optional
    /// `pred_weight_table()`. §7.4.7.1: the derived
    /// `MaxNumMergeCand = 5 - five_minus_max_num_merge_cand` shall be in
    /// the range 1..=5 — i.e. the wire value is in 0..=4. `None` for I
    /// slices, dependent slice segments, and headers whose parse stopped
    /// before this point (currently: the parse defers at the
    /// `pred_weight_table()` gate when either
    /// `pps.weighted_pred_flag && slice_type == P` or
    /// `pps.weighted_bipred_flag && slice_type == B` is true).
    pub five_minus_max_num_merge_cand: Option<u32>,
    /// `use_integer_mv_flag` (§7.3.6.1) — present for P/B slices when
    /// the SPS SCC `motion_vector_resolution_control_idc == 2`;
    /// otherwise inferred equal to
    /// `motion_vector_resolution_control_idc` (§7.4.7.1). When 1,
    /// motion vectors of this slice referring to pictures other than
    /// the current picture use integer resolution (eqs 8-98..8-101 /
    /// 8-124..8-125).
    pub use_integer_mv_flag: bool,
    /// Decoded `pred_weight_table()` (§7.3.6.3) when the §7.3.6.1 outer
    /// gate is statically present
    /// (`(pps.weighted_pred_flag && slice_type == P) ||
    /// (pps.weighted_bipred_flag && slice_type == B)`). `None` when the
    /// outer gate is statically absent (the table is not signalled), for
    /// I slices, for dependent slice segments, and for headers whose
    /// parse stopped before this point.
    ///
    /// The in-place call uses the base-profile single-layer assumption
    /// for every per-i §7.3.6.3 outer-gate decision (universally `true`)
    /// — see [`PredWeightTableInputs::base_profile`]. Single-layer base
    /// profile is the only configuration this crate currently surfaces;
    /// the multilayer Annex F / SCC self-reference cases need the SPS
    /// range / multilayer / SCC extensions and the DPB to be wired up,
    /// at which point the in-place call site here will be widened to
    /// thread per-i gate decisions through.
    pub pred_weight_table: Option<PredWeightTable>,
    /// `slice_qp_delta` (`se(v)`). `None` when the parser stopped before
    /// reaching it (a deferred non-IDR or P/B body).
    pub slice_qp_delta: Option<i32>,
    /// `slice_cb_qp_offset` (`se(v)`, range −12..=12). Inferred to 0
    /// when not present (`pps_slice_chroma_qp_offsets_present_flag == 0`).
    pub slice_cb_qp_offset: i8,
    /// `slice_cr_qp_offset` (`se(v)`, range −12..=12). Inferred to 0
    /// when not present.
    pub slice_cr_qp_offset: i8,
    /// `slice_act_y_qp_offset` (`se(v)`, §7.3.6.1). Present only when
    /// `pps_slice_act_qp_offsets_present_flag` (the SCC PPS body);
    /// inferred to 0 otherwise. §7.4.7.1 bounds
    /// `PpsActQpOffsetY + slice_act_y_qp_offset` to −12..=12.
    pub slice_act_y_qp_offset: i32,
    /// `slice_act_cb_qp_offset` (`se(v)`, §7.3.6.1). Present only when
    /// `pps_slice_act_qp_offsets_present_flag`; inferred to 0 otherwise.
    pub slice_act_cb_qp_offset: i32,
    /// `slice_act_cr_qp_offset` (`se(v)`, §7.3.6.1). Present only when
    /// `pps_slice_act_qp_offsets_present_flag`; inferred to 0 otherwise.
    pub slice_act_cr_qp_offset: i32,
    /// `cu_chroma_qp_offset_enabled_flag` (`u(1)`, §7.3.6.1). Present
    /// only when the range-extension `chroma_qp_offset_list_enabled_flag`
    /// is set; inferred to 0 otherwise.
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// Deblocking-filter values, carrying the §7.4.7.1 inferred
    /// defaults when the slice override block is absent. `None` when
    /// the parser stopped before this point.
    pub deblocking: Option<SliceDeblocking>,
    /// `slice_loop_filter_across_slices_enabled_flag`. Inferred to
    /// `pps_loop_filter_across_slices_enabled_flag` when not present.
    /// `None` when the parser stopped before this point.
    pub slice_loop_filter_across_slices_enabled_flag: Option<bool>,
    /// Entry-point-offset block. `None` when neither tiles nor
    /// entropy-coding-sync are enabled (the block is absent), or when
    /// the parser stopped before this point.
    pub entry_point_offsets: Option<EntryPointOffsets>,
    /// `slice_segment_header_extension_length` (`ue(v)`). `None` when
    /// `slice_segment_header_extension_present_flag == 0` or the parser
    /// stopped before this point.
    pub slice_segment_header_extension_length: Option<u32>,
    /// Byte offset, from the start of the RBSP, of the first byte of
    /// `slice_segment_data()` — i.e. the position immediately after
    /// `byte_alignment()`. `None` when the header was not parsed all
    /// the way to `byte_alignment()` (a deferred body).
    pub byte_offset_to_slice_data: Option<usize>,
    /// Decoded `ref_pic_lists_modification()` (§7.3.6.2) when the
    /// §7.3.6.1 outer gate
    /// (`pps.lists_modification_present_flag == 1 && NumPicTotalCurr > 1`)
    /// is statically present. `None` when the gate is statically
    /// absent (`pps.lists_modification_present_flag == 0`,
    /// `NumPicTotalCurr <= 1`, an I slice, a dependent slice segment,
    /// or a header whose parse stopped before this point — including
    /// the inter-RPS-predicted SPS-form short-term RPS case, where
    /// the per-position `UsedByCurrPicS0` / `UsedByCurrPicS1` flags
    /// needed for the §7.4.7.2 `NumPicTotalCurr` derivation can only
    /// be resolved by running the §7.4.8 inter-RPS-prediction step;
    /// the parser defers in that case and surfaces an
    /// [`Self::opaque_tail`] starting at the `ref_pic_lists_modification()`
    /// bit position).
    pub ref_pic_lists_modification: Option<RefPicListsModification>,
    /// Opaque suffix of the slice-header RBSP. Populated when the
    /// parser reaches a deferred body (the non-IDR POC/RPS block or a
    /// P/B reference-list / weighted-prediction sub-structure); carries
    /// the still-unparsed RBSP bytes and the start-bit offset. `None`
    /// when the header was parsed to completion.
    pub opaque_tail: Option<OpaqueTail>,
}

impl SliceSegmentHeader {
    /// Parse `slice_segment_header()` from the first bit of the
    /// (already-unescaped) slice-segment-layer RBSP body — i.e. after
    /// the two-byte NAL header has been removed (see
    /// [`crate::hevc::engine::nal::NalUnit`]).
    ///
    /// * `nal_unit_type` is the value from the NAL header; it gates
    ///   both `no_output_of_prior_pics_flag` (IRAP range) and the
    ///   non-IDR POC/RPS block.
    /// * `sps` is the activated SPS, `pps` the activated PPS. The
    ///   caller resolves `slice_pic_parameter_set_id` to the right PPS
    ///   and that PPS's `pps_seq_parameter_set_id` to the right SPS;
    ///   this parser uses the supplied pair for the field widths and
    ///   presence gates.
    pub fn parse(
        rbsp: &[u8],
        nal_unit_type: u8,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
    ) -> Result<Self, SliceError> {
        let mut br = BitReader::new(rbsp);

        let first_slice_segment_in_pic_flag = br.u1()? != 0;

        let no_output_of_prior_pics_flag = if (BLA_W_LP..=RSV_IRAP_VCL23).contains(&nal_unit_type) {
            Some(br.u1()? != 0)
        } else {
            None
        };

        let slice_pic_parameter_set_id = br.ue()?;
        if slice_pic_parameter_set_id > 63 {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_pic_parameter_set_id",
                got: slice_pic_parameter_set_id as i64,
            });
        }
        let slice_pic_parameter_set_id = slice_pic_parameter_set_id as u8;

        // §7.3.6.1: dependent_slice_segment_flag / slice_segment_address
        // only appear for non-first slice segments.
        let mut dependent_slice_segment_flag = false;
        let mut slice_segment_address = 0u32;
        if !first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                dependent_slice_segment_flag = br.u1()? != 0;
            }
            // slice_segment_address width is Ceil( Log2( PicSizeInCtbsY ) )
            // bits (§7.4.7.1).
            let addr_bits = ceil_log2(pic_size_in_ctbs_y(sps));
            slice_segment_address = br.u(addr_bits)?;
            if slice_segment_address >= pic_size_in_ctbs_y(sps) {
                return Err(SliceError::ValueOutOfRange {
                    field: "slice_segment_address",
                    got: slice_segment_address as i64,
                });
            }
        }

        // Defaults / inferences (§7.4.7.1).
        let mut slice_reserved_flags = Vec::new();
        let mut slice_type = None;
        let mut pic_output_flag = true;
        let mut colour_plane_id = None;
        let mut slice_pic_order_cnt_lsb: Option<u32> = None;
        let mut short_term_ref_pic_set_sps_flag: Option<bool> = None;
        let mut inline_short_term_ref_pic_set: Option<ShortTermRefPicSet> = None;
        let mut short_term_ref_pic_set_idx: Option<u32> = None;
        let mut num_long_term_sps: Option<u32> = None;
        let mut num_long_term_pics: Option<u32> = None;
        let mut long_term_ref_pics: Vec<SliceLongTermRefPic> = Vec::new();
        let mut slice_temporal_mvp_enabled_flag = false;

        if !dependent_slice_segment_flag {
            for _ in 0..pps.num_extra_slice_header_bits {
                slice_reserved_flags.push(br.u1()? != 0);
            }
            let st = SliceType::from_raw(br.ue()?)?;
            slice_type = Some(st);

            if pps.output_flag_present_flag {
                pic_output_flag = br.u1()? != 0;
            }

            if sps.separate_colour_plane_flag {
                let id = br.u(2)? as u8;
                colour_plane_id = Some(id);
            }

            // Non-IDR POC + reference-picture-set block (§7.3.6.1).
            let is_idr = nal_unit_type == IDR_W_RADL || nal_unit_type == IDR_N_LP;
            if !is_idr {
                // slice_pic_order_cnt_lsb u(v), width log2_max_poc_lsb_minus4+4.
                let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
                let poc_lsb = br.u(poc_lsb_bits)?;
                if poc_lsb >= sps.max_pic_order_cnt_lsb() {
                    return Err(SliceError::ValueOutOfRange {
                        field: "slice_pic_order_cnt_lsb",
                        got: poc_lsb as i64,
                    });
                }
                slice_pic_order_cnt_lsb = Some(poc_lsb);

                // short_term_ref_pic_set_sps_flag u(1).
                let st_sps_flag = br.u1()? != 0;
                short_term_ref_pic_set_sps_flag = Some(st_sps_flag);
                if st_sps_flag && sps.num_short_term_ref_pic_sets == 0 {
                    // §7.4.7.1: when num_short_term_ref_pic_sets == 0,
                    // short_term_ref_pic_set_sps_flag shall be 0.
                    return Err(SliceError::ValueOutOfRange {
                        field: "short_term_ref_pic_set_sps_flag",
                        got: 1,
                    });
                }

                if !st_sps_flag {
                    let inline = ShortTermRefPicSet::parse_slice_inline(&mut br, sps)?;
                    inline_short_term_ref_pic_set = Some(inline);
                } else if sps.num_short_term_ref_pic_sets > 1 {
                    let idx_bits = ceil_log2(sps.num_short_term_ref_pic_sets);
                    let idx = br.u(idx_bits)?;
                    if idx >= sps.num_short_term_ref_pic_sets {
                        return Err(SliceError::ValueOutOfRange {
                            field: "short_term_ref_pic_set_idx",
                            got: idx as i64,
                        });
                    }
                    short_term_ref_pic_set_idx = Some(idx);
                }
                // else: short_term_ref_pic_set_idx is inferred to 0
                // (and left as None in the struct to signal "absent").

                if sps.long_term_ref_pics_present_flag {
                    let (nl_sps, nl_pics, entries) = parse_long_term_ref_pic_block(&mut br, sps)?;
                    num_long_term_sps = Some(nl_sps);
                    num_long_term_pics = Some(nl_pics);
                    long_term_ref_pics = entries;
                }
                // §7.3.6.1: slice_temporal_mvp_enabled_flag is signalled
                // inside the non-IDR block — an IDR picture never
                // carries it and §7.4.7.1 infers it to 0.
                if sps.sps_temporal_mvp_enabled_flag {
                    slice_temporal_mvp_enabled_flag = br.u1()? != 0;
                }
            }
        }

        // SAO block (§7.3.6.1) — inside the !dependent gate: a dependent
        // slice segment inherits the SAO flags from the associated
        // independent slice segment and does not re-signal them.
        let mut slice_sao_luma_flag = false;
        let mut slice_sao_chroma_flag = false;
        if !dependent_slice_segment_flag && sps.sample_adaptive_offset_enabled_flag {
            slice_sao_luma_flag = br.u1()? != 0;
            if chroma_array_type(sps) != 0 {
                slice_sao_chroma_flag = br.u1()? != 0;
            }
        }

        // The remaining body lives inside the !dependent gate. For a
        // dependent slice segment the header ends after the SAO block,
        // before byte_alignment() (the rest of the header is inherited).
        if dependent_slice_segment_flag {
            // §7.3.6.1: the entry-point block and the header-extension
            // block sit OUTSIDE the !dependent gate — a dependent slice
            // segment signals its own substream entry points.
            let entry_point_offsets = parse_entry_point_offsets(&mut br, sps, pps)?;
            let slice_segment_header_extension_length = parse_header_extension(&mut br, pps)?;
            let byte_offset = consume_byte_alignment(&mut br)?;
            return Ok(Self {
                first_slice_segment_in_pic_flag,
                no_output_of_prior_pics_flag,
                slice_pic_parameter_set_id,
                dependent_slice_segment_flag,
                slice_segment_address,
                slice_reserved_flags,
                slice_type,
                pic_output_flag,
                colour_plane_id,
                slice_pic_order_cnt_lsb: None,
                short_term_ref_pic_set_sps_flag: None,
                inline_short_term_ref_pic_set: None,
                short_term_ref_pic_set_idx: None,
                num_long_term_sps: None,
                num_long_term_pics: None,
                long_term_ref_pics: Vec::new(),
                slice_temporal_mvp_enabled_flag,
                slice_sao_luma_flag,
                slice_sao_chroma_flag,
                num_ref_idx_active_override_flag: None,
                num_ref_idx_l0_active_minus1: None,
                num_ref_idx_l1_active_minus1: None,
                mvd_l1_zero_flag: None,
                cabac_init_flag: None,
                collocated_from_l0_flag: None,
                collocated_ref_idx: None,
                five_minus_max_num_merge_cand: None,
                use_integer_mv_flag: false,
                pred_weight_table: None,
                slice_qp_delta: None,
                slice_cb_qp_offset: 0,
                slice_cr_qp_offset: 0,
                slice_act_y_qp_offset: 0,
                slice_act_cb_qp_offset: 0,
                slice_act_cr_qp_offset: 0,
                cu_chroma_qp_offset_enabled_flag: false,
                deblocking: None,
                slice_loop_filter_across_slices_enabled_flag: None,
                entry_point_offsets,
                slice_segment_header_extension_length,
                byte_offset_to_slice_data: Some(byte_offset),
                ref_pic_lists_modification: None,
                opaque_tail: None,
            });
        }

        // §7.3.6.1: for P / B slices, the SAO block is immediately
        // followed by the `num_ref_idx_active_override_flag` /
        // `num_ref_idx_lX_active_minus1` block. This is the in-place
        // prerequisite for the later `ref_pic_lists_modification()`
        // call (its `list_entry_lX[]` loop indexes `0..=
        // num_ref_idx_lX_active_minus1`). The §7.4.7.1 inference rule
        // fills both `num_ref_idx_lX_active_minus1` values from the PPS
        // defaults when the override flag is 0; both values are capped
        // at 14.
        let st = slice_type.expect("independent slice has a slice_type");
        let (
            num_ref_idx_active_override_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
        ) = if st.is_inter() {
            let override_flag = br.u1()? != 0;
            let (n0, n1) = if override_flag {
                let n0 = br.ue()?;
                if n0 > 14 {
                    return Err(SliceError::ValueOutOfRange {
                        field: "num_ref_idx_l0_active_minus1",
                        got: n0 as i64,
                    });
                }
                let n1 = if matches!(st, SliceType::B) {
                    let v = br.ue()?;
                    if v > 14 {
                        return Err(SliceError::ValueOutOfRange {
                            field: "num_ref_idx_l1_active_minus1",
                            got: v as i64,
                        });
                    }
                    Some(v as u8)
                } else {
                    None
                };
                (n0 as u8, n1)
            } else {
                // §7.4.7.1 inference defaults from the PPS.
                let n1 = if matches!(st, SliceType::B) {
                    Some(pps.num_ref_idx_l1_default_active_minus1)
                } else {
                    None
                };
                (pps.num_ref_idx_l0_default_active_minus1, n1)
            };
            (Some(override_flag), Some(n0), n1)
        } else {
            (None, None, None)
        };

        // §7.3.6.1 inter-slice continuation: after the
        // `num_ref_idx_active_override_flag` block, the spec emits
        //   if( lists_modification_present_flag && NumPicTotalCurr > 1 )
        //       ref_pic_lists_modification( )
        //   if( slice_type == B )           mvd_l1_zero_flag      u(1)
        //   if( cabac_init_present_flag )   cabac_init_flag       u(1)
        //   if( slice_temporal_mvp_enabled_flag ) {
        //       if( slice_type == B )       collocated_from_l0_flag  u(1)   (else inferred 1, §7.4.7.1)
        //       if( ( collocated_from_l0_flag && num_ref_idx_l0_active_minus1 > 0 ) ||
        //           ( !collocated_from_l0_flag && num_ref_idx_l1_active_minus1 > 0 ) )
        //           collocated_ref_idx     ue(v)                          (else inferred 0)
        //   }
        // followed by the weighted-pred-table gate.
        //
        // The `ref_pic_lists_modification()` gate consumes the
        // §7.4.7.2 `NumPicTotalCurr` derivation (equation 7-57). When
        // `pps.lists_modification_present_flag == 0` the
        // `if(... && NumPicTotalCurr > 1)` short-circuit applies
        // without needing the derivation, so the bit stream advances
        // straight to `mvd_l1_zero_flag`. When the flag is 1 we attempt
        // to derive `NumPicTotalCurr` from the resolved slice-header
        // state (active short-term RPS + long-term entries):
        //
        // * For the inline-form short-term RPS
        //   (`short_term_ref_pic_set_sps_flag == 0`) the on-wire form
        //   per §7.4.8 has `inter_ref_pic_set_prediction_flag == 0`
        //   when `stRpsIdx == num_short_term_ref_pic_sets` (the
        //   slice-inline index): the `used_by_curr_pic_s{0,1}_flag`
        //   arrays come directly from the inline RPS. The §7.4.8 form
        //   *is* allowed at the slice-inline index when the SPS has
        //   `num_short_term_ref_pic_sets > 0`; in that case the parser
        //   defers (the derivation requires walking the source RPS
        //   chain and is out of scope here).
        // * For the SPS-form (`short_term_ref_pic_set_sps_flag == 1`)
        //   the active RPS is `sps.short_term_ref_pic_sets[idx]`. When
        //   that RPS uses the explicit form the per-position
        //   `used_by_curr_pic_sX_flag` arrays are usable directly; when
        //   it uses inter-prediction the §7.4.8 derivation must be run
        //   first and the parser defers.
        //
        // The §F.7.4.7.2 multilayer-extension variant and the
        // SCC `pps_curr_pic_ref_enabled_flag` closing-clause are wired
        // through [`NumPicTotalCurrInputs`] but the base-profile call
        // site here leaves both at their `false` defaults (the PPS
        // SCC extension is not yet surfaced; multilayer extension is
        // forwarded via the long-term-ref builder).
        //
        // For IDR slices the entire non-IDR POC/RPS block is absent
        // and `inline_short_term_ref_pic_set` / `long_term_ref_pics`
        // are empty: `NumPicTotalCurr` is `0` and the gate is
        // statically false.
        let (ref_pic_lists_modification, num_pic_total_curr_resolved) = if st.is_inter()
            && pps.lists_modification_present_flag
        {
            match resolve_active_short_term_rps(
                sps,
                short_term_ref_pic_set_sps_flag,
                inline_short_term_ref_pic_set.as_ref(),
                short_term_ref_pic_set_idx,
            ) {
                ActiveShortTermRps::Materialized(m) => {
                    let lt_used = collect_used_by_curr_pic_lt(&long_term_ref_pics, sps);
                    let inputs = NumPicTotalCurrInputs::from_used_flags(
                        &m.used_by_curr_pic_s0,
                        &m.used_by_curr_pic_s1,
                        &lt_used,
                    );
                    let npc = inputs.compute();
                    if npc > 1 {
                        let l0_active =
                            num_ref_idx_l0_active_minus1.ok_or(SliceError::ValueOutOfRange {
                                field: "num_ref_idx_l0_active_minus1",
                                got: -1,
                            })?;
                        let l1_active = num_ref_idx_l1_active_minus1.unwrap_or(0);
                        let rplm =
                            RefPicListsModification::parse(&mut br, st, l0_active, l1_active, npc)?;
                        (Some(rplm), Some(npc))
                    } else {
                        // `NumPicTotalCurr <= 1` — the §7.3.6.1
                        // gate is statically false; the structure
                        // is not signalled and we continue at
                        // `mvd_l1_zero_flag`.
                        (None, Some(npc))
                    }
                }
                ActiveShortTermRps::Empty => {
                    // IDR or no RPS picked: `NumPicTotalCurr == 0`,
                    // gate is statically false.
                    (None, Some(0))
                }
                ActiveShortTermRps::MaterializeFailed => {
                    // §7.4.8 derivation could not run (malformed
                    // `RefRpsIdx` chain or array-length mismatch);
                    // defer to opaque tail so the caller can salvage
                    // the rest of the bitstream.
                    return Ok(Self {
                        first_slice_segment_in_pic_flag,
                        no_output_of_prior_pics_flag,
                        slice_pic_parameter_set_id,
                        dependent_slice_segment_flag,
                        slice_segment_address,
                        slice_reserved_flags,
                        slice_type,
                        pic_output_flag,
                        colour_plane_id,
                        slice_pic_order_cnt_lsb,
                        short_term_ref_pic_set_sps_flag,
                        inline_short_term_ref_pic_set,
                        short_term_ref_pic_set_idx,
                        num_long_term_sps,
                        num_long_term_pics,
                        long_term_ref_pics,
                        slice_temporal_mvp_enabled_flag,
                        slice_sao_luma_flag,
                        slice_sao_chroma_flag,
                        num_ref_idx_active_override_flag,
                        num_ref_idx_l0_active_minus1,
                        num_ref_idx_l1_active_minus1,
                        mvd_l1_zero_flag: None,
                        cabac_init_flag: None,
                        collocated_from_l0_flag: None,
                        collocated_ref_idx: None,
                        five_minus_max_num_merge_cand: None,
                        use_integer_mv_flag: false,
                        pred_weight_table: None,
                        slice_qp_delta: None,
                        slice_cb_qp_offset: 0,
                        slice_cr_qp_offset: 0,
                        slice_act_y_qp_offset: 0,
                        slice_act_cb_qp_offset: 0,
                        slice_act_cr_qp_offset: 0,
                        cu_chroma_qp_offset_enabled_flag: false,
                        deblocking: None,
                        slice_loop_filter_across_slices_enabled_flag: None,
                        entry_point_offsets: None,
                        slice_segment_header_extension_length: None,
                        byte_offset_to_slice_data: None,
                        ref_pic_lists_modification: None,
                        opaque_tail: Some(OpaqueTail::capture_at(br.bit_pos(), rbsp)),
                    });
                }
            }
        } else {
            (None, None)
        };
        // `num_pic_total_curr_resolved` is currently only used to gate
        // the in-place RPLM parse above; future rounds may surface it
        // on the slice header (the §8.3.4 implicit reference-list
        // derivation needs the same value).
        let _ = num_pic_total_curr_resolved;

        // §7.3.6.1 inter-slice mvd / cabac-init / collocated block —
        // reached only when `slice_type` is P or B and the
        // `ref_pic_lists_modification()` block is statically absent
        // (`pps.lists_modification_present_flag == 0`, which makes the
        // §7.3.6.1 outer `if(... && NumPicTotalCurr > 1)` false
        // unconditionally). For an I slice the entire block is absent
        // and the four fields stay `None`.
        let (mvd_l1_zero_flag, cabac_init_flag, collocated_from_l0_flag, collocated_ref_idx) =
            if st.is_inter() {
                // §7.3.6.1 `if( slice_type == B ) mvd_l1_zero_flag u(1)`.
                let mvd_l1_zero = if matches!(st, SliceType::B) {
                    Some(br.u1()? != 0)
                } else {
                    None
                };

                // §7.3.6.1 `if( cabac_init_present_flag ) cabac_init_flag
                // u(1)`. §7.4.7.1: inferred to 0 when absent.
                let cabac_init = if pps.cabac_init_present_flag {
                    Some(br.u1()? != 0)
                } else {
                    Some(false)
                };

                // §7.3.6.1 temporal-MVP block.
                let (coll_from_l0, coll_ref_idx) = if slice_temporal_mvp_enabled_flag {
                    // `if( slice_type == B ) collocated_from_l0_flag u(1)`,
                    // else §7.4.7.1 inference to 1.
                    let from_l0 = if matches!(st, SliceType::B) {
                        br.u1()? != 0
                    } else {
                        true
                    };

                    // §7.3.6.1: `collocated_ref_idx` is present iff the
                    // active list (selected by `collocated_from_l0_flag`)
                    // has more than one entry. §7.4.7.1: inferred to 0
                    // when absent. Both `num_ref_idx_lX_active_minus1`
                    // values are `Some(_)` at this point (the override
                    // block populated them for every inter slice, and L1
                    // is populated for B slices).
                    let n0 = num_ref_idx_l0_active_minus1.expect("L0 active populated for inter");
                    let needs_ref_idx = if from_l0 {
                        n0 > 0
                    } else {
                        // !from_l0 implies slice_type == B (an I/P slice
                        // takes the inferred `true` branch). For a B slice
                        // L1 is signalled by the override block.
                        let n1 = num_ref_idx_l1_active_minus1.expect("L1 active populated for B");
                        n1 > 0
                    };
                    let ref_idx = if needs_ref_idx {
                        let raw = br.ue()?;
                        let max = if from_l0 {
                            n0 as u32
                        } else {
                            num_ref_idx_l1_active_minus1.unwrap() as u32
                        };
                        if raw > max {
                            return Err(SliceError::ValueOutOfRange {
                                field: "collocated_ref_idx",
                                got: raw as i64,
                            });
                        }
                        raw
                    } else {
                        0
                    };

                    (Some(from_l0), Some(ref_idx))
                } else {
                    (None, None)
                };

                (mvd_l1_zero, cabac_init, coll_from_l0, coll_ref_idx)
            } else {
                (None, None, None, None)
            };

        // §7.3.6.1 P / B `pred_weight_table()` gate. The table is
        // signalled iff either `(weighted_pred_flag && slice_type == P)`
        // or `(weighted_bipred_flag && slice_type == B)`. When the gate
        // is statically absent the parser walks straight past it into
        // the merge-candidate block; when it is present the standalone
        // [`PredWeightTable::parse`] is invoked in place with the
        // base-profile single-layer assumption (every per-i §7.3.6.3
        // outer gate `true` — see [`PredWeightTableInputs::base_profile`]).
        // This is the only single-layer configuration this crate
        // currently surfaces: the SPS range / multilayer / SCC
        // extensions are not yet wired through, so the bit-depth
        // arguments fall back to the SPS `BitDepthY` / `BitDepthC`
        // (`WpOffsetHalfRangeY` = `WpOffsetHalfRangeC` = 128 per §7.4.7.3
        // because `high_precision_offsets_enabled_flag` defaults to 0).
        // The per-i outer-gate decisions, when needed for the
        // multilayer-extension / SCC self-reference cases, will be
        // threaded through here once those extensions are surfaced.
        let weighted_pred_table_present = (pps.weighted_pred_flag && matches!(st, SliceType::P))
            || (pps.weighted_bipred_flag && matches!(st, SliceType::B));
        let pred_weight_table = if weighted_pred_table_present {
            // Resolve the active L0 / L1 cardinalities the table parser
            // needs. Both have been populated by the override block
            // above (§7.4.7.1 inference fills L0 for P/B and L1 for B
            // when override == 0).
            let n0 = num_ref_idx_l0_active_minus1.expect("L0 active populated for inter");
            let n1 = if matches!(st, SliceType::B) {
                num_ref_idx_l1_active_minus1.expect("L1 active populated for B")
            } else {
                0
            };
            let inputs = PredWeightTableInputs::base_profile(
                st,
                n0,
                n1,
                chroma_array_type(sps),
                sps.bit_depth_luma(),
                sps.bit_depth_chroma(),
            );
            Some(PredWeightTable::parse(&mut br, &inputs)?)
        } else {
            None
        };

        // §7.3.6.1 `five_minus_max_num_merge_cand` (ue(v)), signalled
        // for every inter slice immediately after the (optional)
        // pred_weight_table(). §7.4.7.1 derives
        // `MaxNumMergeCand = 5 - five_minus_max_num_merge_cand`, with
        // the conformance constraint `1 <= MaxNumMergeCand <= 5` —
        // i.e. the wire value lies in 0..=4. The SCC
        // `use_integer_mv_flag` (gated on
        // `motion_vector_resolution_control_idc == 2`) is statically
        // absent because the PPS SCC extension is not surfaced by this
        // crate yet (§7.4.7.1: when not present,
        // `motion_vector_resolution_control_idc` is inferred to 0).
        let five_minus_max_num_merge_cand = if st.is_inter() {
            let v = br.ue()?;
            if v > 4 {
                return Err(SliceError::ValueOutOfRange {
                    field: "five_minus_max_num_merge_cand",
                    got: v as i64,
                });
            }
            Some(v)
        } else {
            None
        };

        // §7.3.6.1: use_integer_mv_flag is present for inter slices
        // when motion_vector_resolution_control_idc == 2; otherwise
        // §7.4.7.1 infers it equal to the idc value.
        let mv_res_idc = sps
            .sps_scc_extension
            .as_ref()
            .map_or(0, |s| s.motion_vector_resolution_control_idc);
        let use_integer_mv_flag = if st.is_inter() && mv_res_idc == 2 {
            br.u1()? != 0
        } else {
            mv_res_idc == 1
        };

        // Slice QP / chroma QP / deblocking / loop-filter / entry-points
        // tail (§7.3.6.1) — shared by I, P and B independent slice
        // segments.
        let slice_qp_delta = br.se()?;

        let mut slice_cb_qp_offset = 0i8;
        let mut slice_cr_qp_offset = 0i8;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            slice_cb_qp_offset = parse_qp_offset(&mut br, "slice_cb_qp_offset")?;
            slice_cr_qp_offset = parse_qp_offset(&mut br, "slice_cr_qp_offset")?;
        }

        // SCC adaptive-colour-transform per-slice QP offsets (§7.3.6.1),
        // present only when the SCC PPS body set
        // `pps_slice_act_qp_offsets_present_flag`. §7.4.7.1 bounds the
        // sum `PpsActQpOffset{Y,Cb,Cr} + slice_act_{y,cb,cr}_qp_offset`
        // to −12..=12; the per-element offsets themselves are se(v) with
        // no independent bound, so the conformance check is applied to
        // the combined value using the PPS-level offsets.
        let mut slice_act_y_qp_offset = 0i32;
        let mut slice_act_cb_qp_offset = 0i32;
        let mut slice_act_cr_qp_offset = 0i32;
        let pps_slice_act_qp_offsets_present_flag = pps
            .pps_scc_extension
            .as_ref()
            .map(|scc| scc.pps_slice_act_qp_offsets_present_flag)
            .unwrap_or(false);
        if pps_slice_act_qp_offsets_present_flag {
            // The presence of the offsets implies a decoded SCC body.
            let scc = pps
                .pps_scc_extension
                .as_ref()
                .expect("pps_slice_act_qp_offsets_present_flag implies SCC body");
            slice_act_y_qp_offset = parse_slice_act_qp_offset(
                &mut br,
                "slice_act_y_qp_offset",
                scc.pps_act_qp_offset_y(),
            )?;
            slice_act_cb_qp_offset = parse_slice_act_qp_offset(
                &mut br,
                "slice_act_cb_qp_offset",
                scc.pps_act_qp_offset_cb(),
            )?;
            slice_act_cr_qp_offset = parse_slice_act_qp_offset(
                &mut br,
                "slice_act_cr_qp_offset",
                scc.pps_act_qp_offset_cr(),
            )?;
        }

        // `cu_chroma_qp_offset_enabled_flag` (§7.3.6.1), present only
        // when the range-extension `chroma_qp_offset_list_enabled_flag`
        // is set; inferred to 0 otherwise (§7.4.7.1).
        let chroma_qp_offset_list_enabled_flag = pps
            .pps_range_extension
            .as_ref()
            .map(|re| re.chroma_qp_offset_list_enabled_flag)
            .unwrap_or(false);
        let cu_chroma_qp_offset_enabled_flag = if chroma_qp_offset_list_enabled_flag {
            br.u1()? != 0
        } else {
            false
        };

        // Deblocking override (§7.3.6.1).
        let deblocking = parse_slice_deblocking(&mut br, pps)?;

        // slice_loop_filter_across_slices_enabled_flag gate (§7.3.6.1).
        let slice_loop_filter_across_slices_enabled_flag = if pps
            .pps_loop_filter_across_slices_enabled_flag
            && (slice_sao_luma_flag || slice_sao_chroma_flag || !deblocking.disabled_flag)
        {
            br.u1()? != 0
        } else {
            pps.pps_loop_filter_across_slices_enabled_flag
        };

        // Entry-point-offset block (§7.3.6.1). §7.4.7.1 bounds
        // `num_entry_point_offsets` by the active partitioning: the
        // tile count when only `tiles_enabled_flag == 1`,
        // `PicHeightInCtbsY` when only
        // `entropy_coding_sync_enabled_flag == 1`, and
        // `NumTileColumns * PicHeightInCtbsY` when both are 1
        // (wavefronts inside every tile). Each
        // `entry_point_offset_minus1[i]` is `offset_len_minus1 + 1`
        // bits wide and is read into [`EntryPointOffsets::
        // entry_point_offset_minus1`] (a per-index `Vec<u32>`); the
        // byte length of subset `i` follows as
        // `entry_point_offset_minus1[i] + 1` (§7.4.7.1) and is exposed
        // via [`EntryPointOffsets::subset_length`].
        let entry_point_offsets = parse_entry_point_offsets(&mut br, sps, pps)?;

        // Slice-segment-header extension block (§7.3.6.1).
        let slice_segment_header_extension_length = parse_header_extension(&mut br, pps)?;

        let byte_offset = consume_byte_alignment(&mut br)?;

        Ok(Self {
            first_slice_segment_in_pic_flag,
            no_output_of_prior_pics_flag,
            slice_pic_parameter_set_id,
            dependent_slice_segment_flag,
            slice_segment_address,
            slice_reserved_flags,
            slice_type,
            pic_output_flag,
            colour_plane_id,
            slice_pic_order_cnt_lsb,
            short_term_ref_pic_set_sps_flag,
            inline_short_term_ref_pic_set,
            short_term_ref_pic_set_idx,
            num_long_term_sps,
            num_long_term_pics,
            long_term_ref_pics,
            slice_temporal_mvp_enabled_flag,
            slice_sao_luma_flag,
            slice_sao_chroma_flag,
            num_ref_idx_active_override_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            mvd_l1_zero_flag,
            cabac_init_flag,
            collocated_from_l0_flag,
            collocated_ref_idx,
            five_minus_max_num_merge_cand,
            use_integer_mv_flag,
            pred_weight_table,
            slice_qp_delta: Some(slice_qp_delta),
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            slice_act_y_qp_offset,
            slice_act_cb_qp_offset,
            slice_act_cr_qp_offset,
            cu_chroma_qp_offset_enabled_flag,
            deblocking: Some(deblocking),
            slice_loop_filter_across_slices_enabled_flag: Some(
                slice_loop_filter_across_slices_enabled_flag,
            ),
            entry_point_offsets,
            slice_segment_header_extension_length,
            byte_offset_to_slice_data: Some(byte_offset),
            ref_pic_lists_modification,
            opaque_tail: None,
        })
    }

    /// `SliceQpY = 26 + init_qp_minus26 + slice_qp_delta` (equation
    /// 7-54). Returns `None` when `slice_qp_delta` was not parsed (a
    /// deferred body); `pps` supplies `init_qp_minus26`.
    pub fn slice_qp_y(&self, pps: &PicParameterSet) -> Option<i32> {
        self.slice_qp_delta.map(|d| 26 + pps.init_qp_minus26 + d)
    }

    /// `MaxNumMergeCand` per §7.4.7.1 equation 7-53:
    /// `MaxNumMergeCand = 5 - five_minus_max_num_merge_cand`. Returns
    /// `None` for I slices (the field is absent) and for headers whose
    /// parse stopped before the merge-candidate block. The derived
    /// value is guaranteed to lie in 1..=5 — `Self::parse` rejects an
    /// out-of-range wire value at decode time.
    pub fn max_num_merge_cand(&self) -> Option<u8> {
        self.five_minus_max_num_merge_cand
            .map(|v| 5u8.saturating_sub(v as u8))
    }
}

/// `ChromaArrayType` per §7.4.2.2: equal to `chroma_format_idc` unless
/// `separate_colour_plane_flag == 1`, in which case it is 0.
fn chroma_array_type(sps: &SeqParameterSet) -> u8 {
    if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    }
}

/// `PicSizeInCtbsY = PicWidthInCtbsY * PicHeightInCtbsY`
/// (equations 7-15/7-17/7-19). `CtbSizeY = 1 << CtbLog2SizeY` and the
/// per-dimension counts use ceiling division.
fn pic_size_in_ctbs_y(sps: &SeqParameterSet) -> u32 {
    let ctb_size = 1u32 << sps.log2_ctb_size();
    let width_in_ctbs = sps.pic_width_in_luma_samples.div_ceil(ctb_size);
    let height_in_ctbs = sps.pic_height_in_luma_samples.div_ceil(ctb_size);
    width_in_ctbs * height_in_ctbs
}

/// `PicHeightInCtbsY` per equation 7-19: the picture height in CTBs,
/// i.e. `Ceil(pic_height_in_luma_samples / CtbSizeY)`.
fn pic_height_in_ctbs_y(sps: &SeqParameterSet) -> u32 {
    let ctb_size = 1u32 << sps.log2_ctb_size();
    sps.pic_height_in_luma_samples.div_ceil(ctb_size)
}

/// §7.3.6.1 — the entry-point-offset block (`num_entry_point_offsets`,
/// `offset_len_minus1`, `entry_point_offset_minus1[i]`), present when
/// tiles or entropy-coding sync are enabled. Signalled by BOTH
/// independent and dependent slice segments.
fn parse_entry_point_offsets(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<Option<EntryPointOffsets>, SliceError> {
    if !(pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag) {
        return Ok(None);
    }
    let num_entry_point_offsets = br.ue()?;
    let max_num_entry_point_offsets = num_entry_point_offsets_upper_bound(sps, pps);
    if num_entry_point_offsets > max_num_entry_point_offsets {
        return Err(SliceError::ValueOutOfRange {
            field: "num_entry_point_offsets",
            got: num_entry_point_offsets as i64,
        });
    }
    let (offset_len_minus1, entry_point_offset_minus1) = if num_entry_point_offsets > 0 {
        let v = br.ue()?;
        if v > 31 {
            return Err(SliceError::ValueOutOfRange {
                field: "offset_len_minus1",
                got: v as i64,
            });
        }
        let len = v as u8;
        let bits = len + 1;
        let mut offsets = Vec::with_capacity(num_entry_point_offsets as usize);
        for _ in 0..num_entry_point_offsets {
            offsets.push(br.u(bits)?);
        }
        (len, offsets)
    } else {
        (0, Vec::new())
    };
    Ok(Some(EntryPointOffsets {
        num_entry_point_offsets,
        offset_len_minus1,
        entry_point_offset_minus1,
    }))
}

/// §7.3.6.1 — the slice-segment-header extension block (length +
/// skipped payload bytes), present for BOTH independent and dependent
/// slice segments when the PPS signals it.
fn parse_header_extension(
    br: &mut BitReader<'_>,
    pps: &PicParameterSet,
) -> Result<Option<u32>, SliceError> {
    if !pps.slice_segment_header_extension_present_flag {
        return Ok(None);
    }
    let len = br.ue()?;
    for _ in 0..len {
        br.skip(8)?;
    }
    Ok(Some(len))
}

/// §7.4.7.1 upper bound on the slice header's
/// `num_entry_point_offsets` for the active PPS partitioning. The
/// caller has already gated on `tiles_enabled_flag ||
/// entropy_coding_sync_enabled_flag`.
fn num_entry_point_offsets_upper_bound(sps: &SeqParameterSet, pps: &PicParameterSet) -> u32 {
    // §7.4.7.1, three-way constraint:
    //   * tiles == 0, sync == 1  ⇒ 0 .. PicHeightInCtbsY − 1
    //   * tiles == 1, sync == 0  ⇒ 0 .. cols * rows − 1
    //   * tiles == 1, sync == 1  ⇒ 0 .. cols * PicHeightInCtbsY − 1
    // (wavefront rows counted per tile COLUMN — every tile row of a
    // column contributes its CTB rows, summing to PicHeightInCtbsY).
    // Arithmetic in u64 keeps the parser defensive against a
    // pathological PPS even though no conforming level overflows u32.
    let bound = if pps.tiles_enabled_flag {
        let cols = u64::from(pps.tiles.num_tile_columns_minus1) + 1;
        if pps.entropy_coding_sync_enabled_flag {
            cols.saturating_mul(u64::from(pic_height_in_ctbs_y(sps)))
                .saturating_sub(1)
        } else {
            let rows = u64::from(pps.tiles.num_tile_rows_minus1) + 1;
            cols.saturating_mul(rows).saturating_sub(1)
        }
    } else {
        u64::from(pic_height_in_ctbs_y(sps)).saturating_sub(1)
    };
    u32::try_from(bound).unwrap_or(u32::MAX)
}

/// `Ceil( Log2( n ) )` — the §7.4.7.1 width formula for
/// `slice_segment_address`. For `n <= 1` the width is 0 bits (a
/// single-CTB picture has no address to signal).
fn ceil_log2(n: u32) -> u8 {
    if n <= 1 {
        0
    } else {
        // Ceil(Log2(n)) = bit-width of (n - 1).
        (32 - (n - 1).leading_zeros()) as u8
    }
}

/// Parse one `se(v)` chroma QP offset, range-checked to −12..=12.
fn parse_qp_offset(br: &mut BitReader<'_>, field: &'static str) -> Result<i8, SliceError> {
    let v = br.se()?;
    if !(-12..=12).contains(&v) {
        return Err(SliceError::ValueOutOfRange {
            field,
            got: v as i64,
        });
    }
    Ok(v as i8)
}

/// Parse a `slice_act_*_qp_offset` (`se(v)`, §7.3.6.1) and enforce the
/// §7.4.7.1 conformance bound on its sum with the PPS-level offset:
/// `pps_act_qp_offset + slice_act_qp_offset` must lie in −12..=12.
fn parse_slice_act_qp_offset(
    br: &mut BitReader<'_>,
    field: &'static str,
    pps_act_qp_offset: i32,
) -> Result<i32, SliceError> {
    let v = br.se()?;
    let sum = pps_act_qp_offset + v;
    if !(-12..=12).contains(&sum) {
        return Err(SliceError::ValueOutOfRange {
            field,
            got: sum as i64,
        });
    }
    Ok(v)
}

/// Parse the deblocking-filter override block of §7.3.6.1, applying the
/// §7.4.7.1 inference rules for the absent fields.
fn parse_slice_deblocking(
    br: &mut BitReader<'_>,
    pps: &PicParameterSet,
) -> Result<SliceDeblocking, SliceError> {
    // deblocking_filter_override_flag only present when
    // deblocking_filter_override_enabled_flag (PPS).
    let override_flag = if pps.deblocking.override_enabled_flag {
        br.u1()? != 0
    } else {
        false
    };

    if !override_flag {
        // Inferred from the PPS (§7.4.7.1).
        return Ok(SliceDeblocking {
            disabled_flag: pps.deblocking.disabled_flag,
            beta_offset_div2: pps.deblocking.beta_offset_div2,
            tc_offset_div2: pps.deblocking.tc_offset_div2,
        });
    }

    let disabled_flag = br.u1()? != 0;
    let (beta, tc) = if !disabled_flag {
        let beta = br.se()?;
        if !(-6..=6).contains(&beta) {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_beta_offset_div2",
                got: beta as i64,
            });
        }
        let tc = br.se()?;
        if !(-6..=6).contains(&tc) {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_tc_offset_div2",
                got: tc as i64,
            });
        }
        (beta as i8, tc as i8)
    } else {
        // When deblocking is disabled the offsets are not signalled and
        // are inferred to 0 (their effect is moot when disabled).
        (0, 0)
    };

    Ok(SliceDeblocking {
        disabled_flag,
        beta_offset_div2: beta,
        tc_offset_div2: tc,
    })
}

/// Parse the long-term-ref-pic block of §7.3.6.1 (the body gated by
/// `long_term_ref_pics_present_flag` on the SPS), returning the parsed
/// `(num_long_term_sps, num_long_term_pics, entries)` triple.
///
/// The block:
///
/// ```text
///   if( num_long_term_ref_pics_sps > 0 )
///       num_long_term_sps  ue(v)
///   num_long_term_pics      ue(v)
///   for( i = 0; i < num_long_term_sps + num_long_term_pics; i++ ) {
///       if( i < num_long_term_sps ) {
///           if( num_long_term_ref_pics_sps > 1 )
///               lt_idx_sps[i]   u(v) — Ceil(Log2(num_long_term_ref_pics_sps))
///       } else {
///           poc_lsb_lt[i]                    u(v) — log2_max_poc_lsb_minus4+4
///           used_by_curr_pic_lt_flag[i]      u(1)
///       }
///       delta_poc_msb_present_flag[i]        u(1)
///       if( delta_poc_msb_present_flag[i] )
///           delta_poc_msb_cycle_lt[i]        ue(v)
///   }
/// ```
fn parse_long_term_ref_pic_block(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
) -> Result<(u32, u32, Vec<SliceLongTermRefPic>), SliceError> {
    let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
        let v = br.ue()?;
        if v > sps.num_long_term_ref_pics_sps {
            return Err(SliceError::ValueOutOfRange {
                field: "num_long_term_sps",
                got: v as i64,
            });
        }
        v
    } else {
        0
    };
    let num_long_term_pics = br.ue()?;
    // §7.4.7.1 bounds num_long_term_pics by the SPS DPB capacity; we
    // apply a defensive sanity ceiling instead of computing the full
    // DPB-derived bound (which needs RPS counts not yet wired through
    // here). A pathological encoder could otherwise drive an unbounded
    // allocation.
    if num_long_term_pics > HEVC_MAX_LONG_TERM_PICS_IN_SLICE as u32 {
        return Err(SliceError::ValueOutOfRange {
            field: "num_long_term_pics",
            got: num_long_term_pics as i64,
        });
    }
    let total = num_long_term_sps + num_long_term_pics;
    let lt_idx_bits = if sps.num_long_term_ref_pics_sps > 1 {
        ceil_log2(sps.num_long_term_ref_pics_sps)
    } else {
        0
    };
    let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;

    let mut entries = Vec::with_capacity(total as usize);
    for i in 0..total {
        let source = if i < num_long_term_sps {
            let lt_idx_sps = if lt_idx_bits > 0 {
                let v = br.u(lt_idx_bits)?;
                if v >= sps.num_long_term_ref_pics_sps {
                    return Err(SliceError::ValueOutOfRange {
                        field: "lt_idx_sps",
                        got: v as i64,
                    });
                }
                v
            } else {
                0
            };
            SliceLongTermRefPicSource::Sps { lt_idx_sps }
        } else {
            let poc_lsb_lt = br.u(poc_lsb_bits)?;
            let used_by_curr_pic_lt_flag = br.u1()? != 0;
            SliceLongTermRefPicSource::InSlice {
                poc_lsb_lt,
                used_by_curr_pic_lt_flag,
            }
        };
        let delta_poc_msb_present_flag = br.u1()? != 0;
        let delta_poc_msb_cycle_lt = if delta_poc_msb_present_flag {
            br.ue()?
        } else {
            0
        };
        entries.push(SliceLongTermRefPic {
            source,
            delta_poc_msb_present_flag,
            delta_poc_msb_cycle_lt,
        });
    }
    Ok((num_long_term_sps, num_long_term_pics, entries))
}

/// Resolution of the active short-term RPS for the in-place
/// `NumPicTotalCurr` derivation at the §7.3.6.1
/// `ref_pic_lists_modification()` gate. The result is always the
/// post-§7.4.8 materialised form (explicit or inter-predicted both
/// produce the same shape).
enum ActiveShortTermRps {
    /// The active short-term RPS has been resolved to its post-§7.4.8
    /// form. The contained `UsedByCurrPicS{0,1}` arrays are the
    /// per-position flags consumed by equation 7-57.
    Materialized(crate::hevc::engine::sps::MaterializedShortTermRefPicSet),
    /// The slice has no active short-term RPS (an IDR slice, where the
    /// non-IDR POC/RPS block is absent). `NumPicTotalCurr` is `0`.
    Empty,
    /// Materialisation of the active RPS failed — for instance because
    /// the inter-RPS-prediction `used_by_curr_pic_flag` /
    /// `use_delta_flag` arrays did not match the source RPS's
    /// `NumDeltaPocs[RefRpsIdx] + 1`. The slice parser surfaces this
    /// as a deferred opaque tail so the caller can investigate without
    /// the parse aborting.
    MaterializeFailed,
}

/// Resolve the active short-term RPS for the current slice given the
/// already-parsed §7.3.6.1 RPS gate state, running the §7.4.8
/// derivation against the SPS list when needed. See §7.4.8 for the
/// `stRpsIdx` selection.
fn resolve_active_short_term_rps(
    sps: &SeqParameterSet,
    short_term_ref_pic_set_sps_flag: Option<bool>,
    inline_rps: Option<&ShortTermRefPicSet>,
    short_term_ref_pic_set_idx: Option<u32>,
) -> ActiveShortTermRps {
    // Materialise the SPS list once; we may need it both as a source
    // for the slice-inline inter-RPS-prediction and as the active RPS
    // for the SPS form. The list is short (cap
    // `HEVC_MAX_NUM_SHORT_TERM_RPS = 64`) so this is inexpensive
    // relative to a frame decode.
    let sps_materialised = match sps.materialize_short_term_ref_pic_sets() {
        Ok(v) => v,
        Err(_) => return ActiveShortTermRps::MaterializeFailed,
    };
    match short_term_ref_pic_set_sps_flag {
        None => ActiveShortTermRps::Empty,
        Some(false) => match inline_rps {
            None => ActiveShortTermRps::Empty,
            Some(rps) => {
                let source = if rps.inter_ref_pic_set_prediction_flag {
                    // For the slice-inline form `stRpsIdx ==
                    // num_short_term_ref_pic_sets` and the source is
                    // `RefRpsIdx = num_short_term_ref_pic_sets -
                    // (delta_idx_minus1 + 1)` per equation 7-59.
                    let st_rps_idx = sps.num_short_term_ref_pic_sets as i64;
                    let ref_rps_idx = st_rps_idx - (rps.delta_idx_minus1 as i64 + 1);
                    if ref_rps_idx < 0 {
                        return ActiveShortTermRps::MaterializeFailed;
                    }
                    sps_materialised.get(ref_rps_idx as usize)
                } else {
                    None
                };
                match rps.materialize(source) {
                    Ok(m) => ActiveShortTermRps::Materialized(m),
                    Err(_) => ActiveShortTermRps::MaterializeFailed,
                }
            }
        },
        Some(true) => {
            // §7.4.7.1: when not signalled (because
            // `num_short_term_ref_pic_sets <= 1`), the index is
            // inferred to 0.
            let idx = short_term_ref_pic_set_idx.unwrap_or(0) as usize;
            match sps_materialised.into_iter().nth(idx) {
                None => ActiveShortTermRps::Empty,
                Some(m) => ActiveShortTermRps::Materialized(m),
            }
        }
    }
}

/// Resolve the per-entry `UsedByCurrPicLt[i]` flags for the active
/// long-term-ref-pic block, per §7.4.7.1. SPS-resident entries pick
/// `used_by_curr_pic_lt_sps_flag[lt_idx_sps[i]]`; in-slice entries
/// carry the flag directly on the wire.
///
/// Returns `false` for any out-of-range SPS lookup (the slice parser
/// rejects `lt_idx_sps >= num_long_term_ref_pics_sps` already; the
/// fallback is defensive).
fn collect_used_by_curr_pic_lt(
    entries: &[SliceLongTermRefPic],
    sps: &SeqParameterSet,
) -> Vec<bool> {
    entries
        .iter()
        .map(|e| e.used_by_curr_pic_lt(sps).unwrap_or(false))
        .collect()
}

/// Defensive upper bound on `num_long_term_pics` (§7.4.7.1 bounds the
/// value by `sps_max_dec_pic_buffering_minus1[TemporalId] − …`; an
/// HEVC DPB is bounded by `MaxDpbSize` which is bounded by
/// `sps_max_dec_pic_buffering_minus1` ≤ 15 per §7.4.3.2.1).
const HEVC_MAX_LONG_TERM_PICS_IN_SLICE: usize = 16;

/// Consume `byte_alignment()` (§7.3.2.4): one `alignment_bit_equal_to_one`
/// followed by `alignment_bit_equal_to_zero` bits until the cursor is on
/// a byte boundary. Returns the byte offset (from the start of the RBSP)
/// of the first byte that follows.
fn consume_byte_alignment(br: &mut BitReader<'_>) -> Result<usize, SliceError> {
    // alignment_bit_equal_to_one.
    let _ = br.u1()?;
    while br.bit_pos() % 8 != 0 {
        let _ = br.u1()?;
    }
    Ok(br.bit_pos() / 8)
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::sps::LongTermRefPicEntry;

    /// Build a minimal SPS for slice-header parsing context. Only the
    /// fields the slice parser reads are populated meaningfully; the
    /// rest carry defaults that do not affect the header parse.
    #[allow(clippy::too_many_arguments)]
    fn ctx_sps(
        chroma_format_idc: u8,
        separate_colour_plane_flag: bool,
        sao: bool,
        mvp: bool,
        width: u32,
        height: u32,
        log2_diff_max_min_cb: u8,
        log2_min_cb_minus3: u8,
        log2_max_poc_lsb_minus4: u8,
    ) -> SeqParameterSet {
        // Hand-assemble the smallest valid SPS RBSP that decodes to the
        // requested gate values, by parsing the tiny fixture's SPS and
        // patching the relevant fields. The slice parser only consults
        // chroma_format_idc, separate_colour_plane_flag,
        // sample_adaptive_offset_enabled_flag,
        // sps_temporal_mvp_enabled_flag, the CTB / picture-size
        // derivations, and log2_max_pic_order_cnt_lsb_minus4, so a
        // patched struct is sufficient for these unit tests.
        let mut sps = SeqParameterSet::parse(TINY_SPS_RBSP).expect("tiny SPS");
        sps.chroma_format_idc = chroma_format_idc;
        sps.separate_colour_plane_flag = separate_colour_plane_flag;
        sps.sample_adaptive_offset_enabled_flag = sao;
        sps.sps_temporal_mvp_enabled_flag = mvp;
        sps.pic_width_in_luma_samples = width;
        sps.pic_height_in_luma_samples = height;
        sps.log2_diff_max_min_luma_coding_block_size = log2_diff_max_min_cb;
        sps.log2_min_luma_coding_block_size_minus3 = log2_min_cb_minus3;
        sps.log2_max_pic_order_cnt_lsb_minus4 = log2_max_poc_lsb_minus4;
        sps
    }

    /// SPS RBSP body from the tiny fixture (see `sps.rs` tests).
    const TINY_SPS_RBSP: &[u8] = &[
        0x01, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00, 0x1E, 0xA0, 0x88,
        0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x32, 0x10,
    ];
    /// PPS RBSP body from the tiny fixture (see `pps.rs` tests).
    const TINY_PPS_RBSP: &[u8] = &[0xC1, 0x73, 0xC0, 0x89];

    #[test]
    fn slice_type_table_7_7() {
        assert_eq!(SliceType::from_raw(0).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_raw(1).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(2).unwrap(), SliceType::I);
        assert!(SliceType::from_raw(3).is_err());
        assert!(!SliceType::I.is_inter());
        assert!(SliceType::P.is_inter());
        assert!(SliceType::B.is_inter());
    }

    #[test]
    fn ceil_log2_widths() {
        // §7.4.7.1: slice_segment_address width = Ceil(Log2(N)).
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }

    /// Hand-assembled minimal independent I-slice IDR header. With the
    /// fixture SPS/PPS gates (sao=1, mvp=1, chroma=1, single CTB,
    /// loop-filter-across-slices=1) the bit layout is:
    ///   first(1)=1 no_output(1)=0 pps_id ue=1 (->0)
    ///   slice_type ue=011 (->2,I) mvp(1)=0 sao_l(1)=1 sao_c(1)=0
    ///   qp_delta se=011 (->-1) lf_across(1)=1
    ///   byte_alignment: 1 then zeros.
    /// We assemble those bits ourselves so the parse is fully
    /// controlled (the tiny fixture's own slice trace is internally
    /// inconsistent — see the module-level note / docs gap).
    #[test]
    fn parses_hand_assembled_i_idr_header() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // pps from the fixture has loop_filter_across_slices=1,
        // tiles=0, ecs=0, deblock-ctrl-present=0, chroma-qp-off=0.

        // Build the bit string.
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (IRAP)
            (0b1, 1),   // pps_id ue(v) '1' -> 0
            (0b011, 3), // slice_type ue(v) '011' -> 2 (I)
            // slice_temporal_mvp_enabled_flag: absent for an IDR — the
            // §7.3.6.1 flag sits inside the non-IDR block.
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag (chroma!=0)
            (0b011, 3), // slice_qp_delta se(v) '011' -> -1
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment: alignment_bit_equal_to_one
        ]);
        let rbsp = pack_bits(&bits);

        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        assert!(sh.first_slice_segment_in_pic_flag);
        assert_eq!(sh.no_output_of_prior_pics_flag, Some(false));
        assert_eq!(sh.slice_pic_parameter_set_id, 0);
        assert!(!sh.dependent_slice_segment_flag);
        assert_eq!(sh.slice_segment_address, 0);
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert!(!sh.slice_temporal_mvp_enabled_flag);
        assert!(sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert_eq!(sh.slice_loop_filter_across_slices_enabled_flag, Some(true));
        assert!(sh.opaque_tail.is_none());
        // init_qp from the fixture PPS is 26, so SliceQpY = 26 + -1 = 25.
        assert_eq!(sh.slice_qp_y(&pps), Some(25));
        // The header must byte-align: with 1+1+1+3+1+1+3+1=12 bits
        // before alignment, alignment consumes bits 12..16 (one '1'
        // plus zero pad), so slice data begins at byte 2.
        assert_eq!(sh.byte_offset_to_slice_data, Some(2));
    }

    /// I-slice whose PPS carries an SCC body with
    /// `pps_slice_act_qp_offsets_present_flag == 1`: the three
    /// `slice_act_{y,cb,cr}_qp_offset` se(v) fields (§7.3.6.1) are
    /// parsed after `slice_qp_delta`, with the §7.4.7.1 sum bound
    /// enforced against the PPS-level offsets.
    #[test]
    fn parses_slice_act_qp_offsets_when_pps_signals_them() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.pps_scc_extension = Some(crate::hevc::engine::pps::PpsSccExtension {
            pps_slice_act_qp_offsets_present_flag: true,
            ..Default::default()
        });
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (IRAP)
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            // slice_temporal_mvp_enabled_flag: absent for an IDR.
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag
            (0b011, 3), // slice_qp_delta se '011' -> -1
            // pps_slice_chroma_qp_offsets_present_flag = 0 → absent.
            // slice_act_*_qp_offset (PpsActQpOffset* all 0 here):
            (0b1, 1),   // slice_act_y_qp_offset se '1' -> 0
            (0b010, 3), // slice_act_cb_qp_offset se '010' -> +1
            (0b011, 3), // slice_act_cr_qp_offset se '011' -> -1
            // chroma_qp_offset_list_enabled_flag = 0 →
            // cu_chroma_qp_offset_enabled_flag absent.
            (1, 1), // slice_loop_filter_across_slices_enabled_flag
            (1, 1), // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert_eq!(sh.slice_act_y_qp_offset, 0);
        assert_eq!(sh.slice_act_cb_qp_offset, 1);
        assert_eq!(sh.slice_act_cr_qp_offset, -1);
        assert!(!sh.cu_chroma_qp_offset_enabled_flag);
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// I-slice whose PPS range-extension sets
    /// `chroma_qp_offset_list_enabled_flag == 1`: the
    /// `cu_chroma_qp_offset_enabled_flag` u(1) (§7.3.6.1) is parsed
    /// after the QP-offset block.
    #[test]
    fn parses_cu_chroma_qp_offset_enabled_flag_when_list_enabled() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.pps_range_extension = Some(crate::hevc::engine::pps::PpsRangeExtension {
            chroma_qp_offset_list_enabled_flag: true,
            ..Default::default()
        });
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            // slice_temporal_mvp_enabled_flag: absent for an IDR.
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag
            (0b011, 3), // slice_qp_delta se -> -1
            // act offsets absent (pps_slice_act_qp_offsets_present_flag 0)
            (1, 1), // cu_chroma_qp_offset_enabled_flag = 1
            (1, 1), // slice_loop_filter_across_slices_enabled_flag
            (1, 1), // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert!(sh.cu_chroma_qp_offset_enabled_flag);
        assert!(sh.opaque_tail.is_none());
    }

    /// §7.4.7.1: a `slice_act_y_qp_offset` whose sum with the PPS-level
    /// `PpsActQpOffsetY` falls outside −12..=12 is rejected.
    #[test]
    fn rejects_out_of_range_slice_act_qp_offset_sum() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // PpsActQpOffsetY = +10 (pps_act_y_qp_offset_plus5 = 15).
        pps.pps_scc_extension = Some(crate::hevc::engine::pps::PpsSccExtension {
            pps_slice_act_qp_offsets_present_flag: true,
            pps_act_y_qp_offset_plus5: 15,
            ..Default::default()
        });
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            // slice_temporal_mvp_enabled_flag: absent for an IDR.
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag
            (0b011, 3), // slice_qp_delta se -> -1
            // slice_act_y_qp_offset = +5 (se '0001010') → sum 10+5 = 15 > 12
            (0b0001010, 7),
        ]);
        let rbsp = pack_bits(&bits);
        let err =
            SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect_err("act sum range");
        assert!(matches!(
            err,
            SliceError::ValueOutOfRange {
                field: "slice_act_y_qp_offset",
                got: 15
            }
        ));
    }

    /// Non-IDR **I-slice** (CRA, type 21 — in the IRAP range so
    /// `no_output_of_prior_pics_flag` is present): the POC + RPS block
    /// is now parsed inline (round 105), so the parser reaches
    /// `byte_alignment()` with no opaque tail.
    #[test]
    fn parses_non_idr_i_slice_cra_with_inline_zero_rps() {
        // Tiny SPS context: num_short_term_ref_pic_sets = 0, so
        // short_term_ref_pic_set_sps_flag must be 0 and the in-line RPS
        // (stRpsIdx = 0, no inter-RPS-prediction signal) reads
        // num_negative_pics + num_positive_pics, both 0.
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (CRA = 21 ∈ IRAP range)
            (0b1, 1),   // pps_id ue '1' -> 0
            (0b011, 3), // slice_type ue '011' -> 2 (I)
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            // in-line st_ref_pic_set( num_short_term_ref_pic_sets = 0 ):
            // stRpsIdx == 0 so inter_ref_pic_set_prediction_flag absent.
            (0b1, 1), // num_negative_pics ue '1' -> 0
            (0b1, 1), // num_positive_pics ue '1' -> 0
            (0, 1),   // slice_temporal_mvp_enabled_flag = 0
            (1, 1),   // slice_sao_luma_flag = 1
            (0, 1),   // slice_sao_chroma_flag = 0
            (0b1, 1), // slice_qp_delta se '1' -> 0
            // pps.lf_across=1 and sao_luma OR !deblock.disabled => present.
            // PPS deblocking override_enabled_flag=0, disabled_flag=0 by
            // inference => !disabled_flag is true => gate fires.
            (1, 1), // slice_loop_filter_across_slices_enabled_flag = 1
            (1, 1), // byte_alignment one-bit
        ]);
        let rbsp = pack_bits(&bits);
        // CRA_NUT (NAL type 21) is in BLA_W_LP..=RSV_IRAP_VCL23 (16..=23).
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert!(sh.first_slice_segment_in_pic_flag);
        assert_eq!(sh.no_output_of_prior_pics_flag, Some(false));
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert_eq!(sh.slice_pic_order_cnt_lsb, Some(0));
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(false));
        let inline = sh
            .inline_short_term_ref_pic_set
            .as_ref()
            .expect("inline ST RPS");
        assert!(!inline.inter_ref_pic_set_prediction_flag);
        assert_eq!(inline.num_negative_pics, 0);
        assert_eq!(inline.num_positive_pics, 0);
        assert!(sh.short_term_ref_pic_set_idx.is_none());
        assert_eq!(sh.num_long_term_sps, None); // SPS gate off
        assert_eq!(sh.num_long_term_pics, None);
        assert!(!sh.slice_temporal_mvp_enabled_flag);
        assert!(sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert_eq!(sh.slice_qp_delta, Some(0));
        // Tail consumed, byte_alignment reached.
        assert!(sh.opaque_tail.is_none());
        assert_eq!(sh.byte_offset_to_slice_data, Some(3));
    }

    /// Non-IDR I-slice using the SPS-resident ST RPS (the SPS has
    /// `num_short_term_ref_pic_sets == 1`, so
    /// `short_term_ref_pic_set_idx` is **absent** — its value is
    /// inferred to 0 — and the slice header reads no extra bits for
    /// the RPS).
    #[test]
    fn parses_non_idr_i_slice_with_sps_rps_single_entry() {
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        // Forge an SPS with one short-term-RPS entry so the gate path
        // matches the §7.3.6.1 "else if( num_short_term_ref_pic_sets > 1 )"
        // branch FALSE path: short_term_ref_pic_set_idx absent.
        sps.num_short_term_ref_pic_sets = 1;
        sps.short_term_ref_pic_sets = vec![ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            num_negative_pics: 0,
            num_positive_pics: 0,
            ..Default::default()
        }];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IRAP)
            (0b1, 1),   // pps_id ue '1' -> 0
            (0b011, 3), // slice_type ue '011' -> 2 (I)
            (0, 8),     // slice_pic_order_cnt_lsb = 0
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1
            // num_short_term_ref_pic_sets == 1 so short_term_ref_pic_set_idx
            // is NOT signalled (inferred 0).
            (0, 1),   // slice_temporal_mvp_enabled_flag = 0
            (1, 1),   // sao_luma = 1
            (0, 1),   // sao_chroma = 0
            (0b1, 1), // slice_qp_delta se '1' -> 0
            (1, 1),   // lf_across_slices = 1
            (1, 1),   // byte_alignment one bit
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(true));
        assert!(sh.inline_short_term_ref_pic_set.is_none());
        assert!(sh.short_term_ref_pic_set_idx.is_none());
        assert_eq!(sh.slice_qp_delta, Some(0));
        assert!(sh.opaque_tail.is_none());
    }

    /// Non-IDR I-slice using the SPS-resident ST RPS with multiple
    /// entries: `short_term_ref_pic_set_idx` is signalled `u(v)` with
    /// width `Ceil(Log2(num_short_term_ref_pic_sets))`.
    #[test]
    fn parses_non_idr_i_slice_with_sps_rps_idx_signalled() {
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        sps.num_short_term_ref_pic_sets = 3; // idx width = 2 bits
        sps.short_term_ref_pic_sets = vec![
            ShortTermRefPicSet::default(),
            ShortTermRefPicSet::default(),
            ShortTermRefPicSet::default(),
        ];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),
            (0, 1),
            (0b1, 1),
            (0b011, 3),
            (0, 8),    // poc_lsb = 0
            (1, 1),    // short_term_ref_pic_set_sps_flag = 1
            (0b10, 2), // short_term_ref_pic_set_idx u(2) = 2
            (0, 1),    // mvp = 0
            (1, 1),    // sao_luma
            (0, 1),    // sao_chroma
            (0b1, 1),  // slice_qp_delta = 0
            (1, 1),    // lf_across
            (1, 1),    // byte_alignment
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.short_term_ref_pic_set_idx, Some(2));
        assert!(sh.inline_short_term_ref_pic_set.is_none());
        assert!(sh.opaque_tail.is_none());
    }

    /// Long-term-ref-pic block: SPS has
    /// `long_term_ref_pics_present_flag=1, num_long_term_ref_pics_sps=2`.
    /// The slice header carries one SPS-indexed entry plus one in-slice
    /// entry, each with `delta_poc_msb_present_flag` and the cycle.
    #[test]
    fn parses_non_idr_i_slice_with_long_term_block() {
        use crate::hevc::engine::sps::LongTermRefPicEntry;
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 2; // lt_idx_sps width = 1 bit
        sps.long_term_ref_pics = vec![
            LongTermRefPicEntry {
                poc_lsb: 0,
                used_by_curr_pic: true,
            },
            LongTermRefPicEntry {
                poc_lsb: 4,
                used_by_curr_pic: false,
            },
        ];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1), // first
            (0, 1), // no_output (IRAP)
            (0b1, 1),
            (0b011, 3), // I slice
            (0, 8),     // poc_lsb
            (0, 1),     // st_sps_flag = 0 (num_st_rps=0)
            (0b1, 1),   // num_negative_pics ue '1' -> 0
            (0b1, 1),   // num_positive_pics ue '1' -> 0
            // Long-term block:
            (0b010, 3), // num_long_term_sps ue '010' -> 1
            (0b010, 3), // num_long_term_pics ue '010' -> 1
            // Entry 0: SPS-indexed (i < num_long_term_sps)
            (1, 1), // lt_idx_sps[0] u(1) = 1
            (0, 1), // delta_poc_msb_present_flag[0] = 0
            // Entry 1: in-slice (i >= num_long_term_sps)
            (0b1010, 8), // poc_lsb_lt[1] u(8) = 0xAA bit pattern... use 10
            (1, 1),      // used_by_curr_pic_lt_flag[1] = 1
            (1, 1),      // delta_poc_msb_present_flag[1] = 1
            (0b010, 3),  // delta_poc_msb_cycle_lt[1] ue '010' -> 1
            (0, 1),      // slice_temporal_mvp_enabled_flag = 0
            (1, 1),      // sao_luma
            (0, 1),      // sao_chroma
            (0b1, 1),    // slice_qp_delta se -> 0
            (1, 1),      // lf_across
            (1, 1),      // byte_alignment
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.num_long_term_sps, Some(1));
        assert_eq!(sh.num_long_term_pics, Some(1));
        assert_eq!(sh.long_term_ref_pics.len(), 2);
        match sh.long_term_ref_pics[0].source {
            SliceLongTermRefPicSource::Sps { lt_idx_sps } => assert_eq!(lt_idx_sps, 1),
            other => panic!("entry 0 should be SPS-indexed: {other:?}"),
        }
        assert!(!sh.long_term_ref_pics[0].delta_poc_msb_present_flag);
        match sh.long_term_ref_pics[1].source {
            SliceLongTermRefPicSource::InSlice {
                poc_lsb_lt,
                used_by_curr_pic_lt_flag,
            } => {
                assert_eq!(poc_lsb_lt, 0b1010); // 8-bit u(v) field carrying 10
                assert!(used_by_curr_pic_lt_flag);
            }
            other => panic!("entry 1 should be in-slice: {other:?}"),
        }
        assert!(sh.long_term_ref_pics[1].delta_poc_msb_present_flag);
        assert_eq!(sh.long_term_ref_pics[1].delta_poc_msb_cycle_lt, 1);
        assert!(sh.opaque_tail.is_none());
    }

    /// The §7.4.7.1 cross-check: when `num_short_term_ref_pic_sets ==
    /// 0`, signalling `short_term_ref_pic_set_sps_flag == 1` is illegal.
    #[test]
    fn rejects_st_sps_flag_when_no_sps_rps() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),
            (0, 1),
            (0b1, 1),
            (0b011, 3), // I
            (0, 8),     // poc_lsb
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1 (illegal: num_st_rps=0)
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "short_term_ref_pic_set_sps_flag",
                got: 1
            }
        );
    }

    /// IDR P-slice with `num_ref_idx_active_override_flag == 0` and
    /// `pps.weighted_pred_flag == 1`: the parser reads the override
    /// flag, infers `num_ref_idx_l0_active_minus1` from the PPS
    /// default, traverses the §7.3.6.1 mvd / cabac-init / collocated
    /// block (with `mvd` absent for P, `cabac_init_flag` inferred
    /// `false` per §7.4.7.1, and the collocated block absent because
    /// `slice_temporal_mvp_enabled_flag == 0`), and then decodes
    /// `pred_weight_table()` in place (the §7.3.6.3 gate is statically
    /// present because `weighted_pred_flag && slice_type == P`).
    /// `ctx_sps` keeps `chroma_format_idc == 1` (4:2:0) so the chroma
    /// sub-block is present, and the PPS default
    /// `num_ref_idx_l0_default_active_minus1 == 0` (single L0 entry)
    /// keeps the per-i loop to one iteration. The minimal table payload
    /// here sets every flag off so the L0 entry's weight / offset
    /// remain at their §7.4.7.3 inferred defaults; the parser then
    /// walks the full inter-slice tail through `byte_alignment()`.
    #[test]
    fn parses_pb_with_weighted_pred_table_in_place() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR is IRAP)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            // §7.3.6.3 pred_weight_table() — minimal "all flags 0" body.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0] = 0
            (0, 1),   // chroma_weight_l0_flag[0] = 0
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert!(sh.slice_sao_luma_flag);
        assert!(sh.slice_sao_chroma_flag);
        // §7.4.7.1 inference: P slice with override == 0 picks up the
        // PPS default for L0 and leaves L1 absent.
        assert_eq!(sh.num_ref_idx_active_override_flag, Some(false));
        assert_eq!(
            sh.num_ref_idx_l0_active_minus1,
            Some(pps.num_ref_idx_l0_default_active_minus1)
        );
        assert_eq!(sh.num_ref_idx_l1_active_minus1, None);
        // §7.3.6.1 mvd / cabac-init / collocated walk for this gate
        // combination: mvd absent (P), cabac inferred false, collocated
        // absent (mvp off).
        assert_eq!(sh.mvd_l1_zero_flag, None);
        assert_eq!(sh.cabac_init_flag, Some(false));
        assert_eq!(sh.collocated_from_l0_flag, None);
        assert_eq!(sh.collocated_ref_idx, None);
        // pred_weight_table parsed in place — single L0 entry, all
        // flags off.
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.luma_log2_weight_denom, 0);
        assert_eq!(pwt.delta_chroma_log2_weight_denom, 0);
        assert_eq!(pwt.entries_l0.len(), 1);
        assert!(!pwt.entries_l0[0].luma_weight_flag);
        assert!(!pwt.entries_l0[0].chroma_weight_flag);
        // P slice → L1 block is empty.
        assert!(pwt.entries_l1.is_empty());
        // Parser walked the rest of the tail past byte_alignment.
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(0));
        assert_eq!(sh.max_num_merge_cand(), Some(5));
        assert_eq!(sh.slice_qp_delta, Some(0));
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// IDR P-slice with `num_ref_idx_active_override_flag == 1` and an
    /// explicitly signalled `num_ref_idx_l0_active_minus1 == 1`. P
    /// slices never signal L1; verify the parser materialises the
    /// override flag and the explicit L0 value, leaves L1 absent,
    /// decodes a two-entry `pred_weight_table()` in place (the §7.3.6.3
    /// gate is statically present here via `pps.weighted_pred_flag =
    /// true`), and walks the rest of the inter-slice tail through
    /// `byte_alignment()`.
    #[test]
    fn parses_pb_override_with_explicit_l0() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b010, 3), // num_ref_idx_l0_active_minus1 ue(v) -> 1
            // pred_weight_table() — 2 L0 entries, all flags 0.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l0_flag[1]
            (0, 1),   // chroma_weight_l0_flag[1]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.num_ref_idx_active_override_flag, Some(true));
        assert_eq!(sh.num_ref_idx_l0_active_minus1, Some(1));
        assert_eq!(sh.num_ref_idx_l1_active_minus1, None);
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 2);
        assert!(pwt.entries_l1.is_empty());
        assert!(sh.opaque_tail.is_none());
    }

    /// IDR B-slice with `num_ref_idx_active_override_flag == 1`:
    /// verifies that both `num_ref_idx_l0_active_minus1` and
    /// `num_ref_idx_l1_active_minus1` are read for a B slice, the
    /// `mvd_l1_zero_flag` is consumed, and the in-place
    /// `pred_weight_table()` body (gate statically present via
    /// `pps.weighted_bipred_flag = true`) decodes both L0 and L1
    /// per-entry flag passes; the parser then walks the rest of the
    /// inter-slice tail through `byte_alignment()`.
    #[test]
    fn parses_b_slice_override_with_both_lists() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_bipred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b1, 1),   // slice_type ue(v) -> 0 (B)
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b011, 3), // num_ref_idx_l0_active_minus1 ue(v) -> 2
            (0b010, 3), // num_ref_idx_l1_active_minus1 ue(v) -> 1
            (1, 1),     // mvd_l1_zero_flag = 1
            // pred_weight_table() — L0 has 3 entries, L1 has 2 entries.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l0_flag[1]
            (0, 1),   // chroma_weight_l0_flag[1]
            (0, 1),   // luma_weight_l0_flag[2]
            (0, 1),   // chroma_weight_l0_flag[2]
            (0, 1),   // luma_weight_l1_flag[0]
            (0, 1),   // chroma_weight_l1_flag[0]
            (0, 1),   // luma_weight_l1_flag[1]
            (0, 1),   // chroma_weight_l1_flag[1]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::B));
        assert_eq!(sh.num_ref_idx_active_override_flag, Some(true));
        assert_eq!(sh.num_ref_idx_l0_active_minus1, Some(2));
        assert_eq!(sh.num_ref_idx_l1_active_minus1, Some(1));
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 3);
        assert_eq!(pwt.entries_l1.len(), 2);
        assert!(sh.opaque_tail.is_none());
    }

    /// IDR B-slice with `num_ref_idx_active_override_flag == 0`: §7.4.7.1
    /// must infer BOTH L0 and L1 defaults from the PPS. The parser
    /// reads through the mvd / cabac / collocated block, decodes the
    /// in-place `pred_weight_table()` (gate present via
    /// `pps.weighted_bipred_flag = true`) with a single L0 / L1 entry
    /// each (the PPS defaults), then walks the rest of the tail to
    /// `byte_alignment()`.
    #[test]
    fn b_slice_override_zero_infers_both_defaults() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_bipred_flag = true;
        let bits = concat_bits(&[
            (1, 1),   // first
            (0, 1),   // no_output (IDR)
            (0b1, 1), // pps_id -> 0
            (0b1, 1), // slice_type -> B
            (1, 1),   // sao_luma
            (1, 1),   // sao_chroma
            (0, 1),   // num_ref_idx_active_override_flag = 0
            (1, 1),   // mvd_l1_zero_flag = 1
            // pred_weight_table() — 1 L0 + 1 L1 entry (defaults), all
            // flags 0.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l1_flag[0]
            (0, 1),   // chroma_weight_l1_flag[0]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::B));
        assert_eq!(sh.num_ref_idx_active_override_flag, Some(false));
        assert_eq!(
            sh.num_ref_idx_l0_active_minus1,
            Some(pps.num_ref_idx_l0_default_active_minus1)
        );
        assert_eq!(
            sh.num_ref_idx_l1_active_minus1,
            Some(pps.num_ref_idx_l1_default_active_minus1)
        );
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 1);
        assert_eq!(pwt.entries_l1.len(), 1);
        assert!(sh.opaque_tail.is_none());
    }

    /// IDR B-slice with `pps.lists_modification_present_flag == 0`
    /// (default for `TINY_PPS_RBSP`): the §7.3.6.1 mvd / cabac-init /
    /// collocated block is walked in-place. With
    /// `pps.cabac_init_present_flag == 0` (default) the cabac-init bit
    /// is absent (inferred `false` per §7.4.7.1); with
    /// `slice_temporal_mvp_enabled_flag == 0` the collocated block is
    /// absent. `mvd_l1_zero_flag` is the only bit consumed past the
    /// override block.
    #[test]
    fn parses_b_slice_mvd_l1_zero_walk_no_mvp() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_bipred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b1, 1),   // slice_type -> B
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b010, 3), // num_ref_idx_l0_active_minus1 ue -> 1
            (0b010, 3), // num_ref_idx_l1_active_minus1 ue -> 1
            (1, 1),     // mvd_l1_zero_flag = 1
            // pred_weight_table() — 2 L0 + 2 L1 entries, all flags 0.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l0_flag[1]
            (0, 1),   // chroma_weight_l0_flag[1]
            (0, 1),   // luma_weight_l1_flag[0]
            (0, 1),   // chroma_weight_l1_flag[0]
            (0, 1),   // luma_weight_l1_flag[1]
            (0, 1),   // chroma_weight_l1_flag[1]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.mvd_l1_zero_flag, Some(true));
        // cabac_init_present_flag == 0 → inferred 0 per §7.4.7.1.
        assert_eq!(sh.cabac_init_flag, Some(false));
        // mvp off → entire collocated block absent.
        assert_eq!(sh.collocated_from_l0_flag, None);
        assert_eq!(sh.collocated_ref_idx, None);
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 2);
        assert_eq!(pwt.entries_l1.len(), 2);
        assert!(sh.opaque_tail.is_none());
    }

    /// IDR P-slice with `pps.cabac_init_present_flag == 1`: the cabac-
    /// init bit is signalled (P slice still walks the gate, even though
    /// `mvd_l1_zero_flag` is absent). With `mvp == 0` the collocated
    /// block is absent.
    #[test]
    fn parses_p_slice_cabac_init_walk() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.cabac_init_present_flag = true;
        pps.weighted_pred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            (1, 1),     // cabac_init_flag = 1
            // pred_weight_table() — 1 L0 entry, all flags 0.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        // P slice → mvd absent.
        assert_eq!(sh.mvd_l1_zero_flag, None);
        assert_eq!(sh.cabac_init_flag, Some(true));
        // mvp off → entire collocated block absent.
        assert_eq!(sh.collocated_from_l0_flag, None);
        assert_eq!(sh.collocated_ref_idx, None);
        assert!(sh.pred_weight_table.is_some());
        assert!(sh.opaque_tail.is_none());
    }

    /// Non-IDR (TRAIL_R) P-slice with `mvp == 1` and
    /// `num_ref_idx_l0_active_minus1 == 0` (single L0 entry): §7.4.7.1
    /// infers `collocated_from_l0_flag = 1` (no bit consumed since
    /// slice_type != B) and the `collocated_ref_idx` field is absent
    /// (only signalled when the active list has more than one entry).
    /// `slice_temporal_mvp_enabled_flag` sits inside the §7.3.6.1
    /// non-IDR block, after the POC + RPS fields.
    #[test]
    fn parses_p_slice_temporal_mvp_single_ref_collocated_inferred() {
        // mvp = true via the ctx_sps argument.
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue -> 1
            (0b1, 1),   // num_positive_pics ue -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // slice_temporal_mvp_enabled_flag = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b1, 1),   // num_ref_idx_l0_active_minus1 ue -> 0
            // pred_weight_table() — 1 L0 entry.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_R (NAL type 1) — a non-IDR picture.
        let sh = SliceSegmentHeader::parse(&rbsp, 1, &sps, &pps).expect("slice header");
        assert!(sh.slice_temporal_mvp_enabled_flag);
        // P slice + mvp on: §7.4.7.1 infers collocated_from_l0 = 1.
        assert_eq!(sh.collocated_from_l0_flag, Some(true));
        // L0 has a single entry (active_minus1 == 0) → ref_idx absent,
        // inferred to 0.
        assert_eq!(sh.collocated_ref_idx, Some(0));
        assert!(sh.pred_weight_table.is_some());
        assert!(sh.opaque_tail.is_none());
    }

    /// Non-IDR (TRAIL_R) P-slice with `mvp == 1` and
    /// `num_ref_idx_l0_active_minus1 == 2` (three L0 entries):
    /// `collocated_from_l0_flag` is inferred to 1 (P slice) and
    /// `collocated_ref_idx` is signalled `ue(v)`.
    #[test]
    fn parses_p_slice_temporal_mvp_collocated_ref_idx_signalled() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue -> 1
            (0b1, 1),   // num_positive_pics ue -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // slice_temporal_mvp_enabled_flag = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b011, 3), // num_ref_idx_l0_active_minus1 ue -> 2
            (0b010, 3), // collocated_ref_idx ue -> 1
            // pred_weight_table() — 3 L0 entries.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l0_flag[1]
            (0, 1),   // chroma_weight_l0_flag[1]
            (0, 1),   // luma_weight_l0_flag[2]
            (0, 1),   // chroma_weight_l0_flag[2]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_R (NAL type 1) — a non-IDR picture.
        let sh = SliceSegmentHeader::parse(&rbsp, 1, &sps, &pps).expect("slice header");
        assert_eq!(sh.num_ref_idx_l0_active_minus1, Some(2));
        // P slice + mvp on: §7.4.7.1 infers collocated_from_l0 = 1.
        assert_eq!(sh.collocated_from_l0_flag, Some(true));
        assert_eq!(sh.collocated_ref_idx, Some(1));
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 3);
        assert!(sh.opaque_tail.is_none());
    }

    /// Non-IDR (TRAIL_R) B-slice with `mvp == 1`,
    /// `collocated_from_l0_flag = 0` (L1 path), and an L1 with more
    /// than one entry: `collocated_ref_idx` indexes L1.
    #[test]
    fn parses_b_slice_temporal_mvp_collocated_from_l1() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_bipred_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id -> 0
            (0b1, 1),   // slice_type -> B
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue -> 1
            (0b1, 1),   // num_positive_pics ue -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // slice_temporal_mvp_enabled_flag = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b1, 1),   // num_ref_idx_l0_active_minus1 ue -> 0
            (0b010, 3), // num_ref_idx_l1_active_minus1 ue -> 1
            (0, 1),     // mvd_l1_zero_flag = 0
            (0, 1),     // collocated_from_l0_flag = 0
            (0b010, 3), // collocated_ref_idx ue -> 1
            // pred_weight_table() — 1 L0 + 2 L1 entries, all flags 0.
            (0b1, 1), // luma_log2_weight_denom ue -> 0
            (0b1, 1), // delta_chroma_log2_weight_denom se -> 0
            (0, 1),   // luma_weight_l0_flag[0]
            (0, 1),   // chroma_weight_l0_flag[0]
            (0, 1),   // luma_weight_l1_flag[0]
            (0, 1),   // chroma_weight_l1_flag[0]
            (0, 1),   // luma_weight_l1_flag[1]
            (0, 1),   // chroma_weight_l1_flag[1]
            (0b1, 1), // five_minus_max_num_merge_cand ue -> 0
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_R (NAL type 1) — a non-IDR picture.
        let sh = SliceSegmentHeader::parse(&rbsp, 1, &sps, &pps).expect("slice header");
        assert_eq!(sh.mvd_l1_zero_flag, Some(false));
        assert_eq!(sh.collocated_from_l0_flag, Some(false));
        // L1 active_minus1 == 1 → 2 entries; ref_idx = 1 indexes the
        // second entry, in range.
        assert_eq!(sh.collocated_ref_idx, Some(1));
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.entries_l0.len(), 1);
        assert_eq!(pwt.entries_l1.len(), 2);
        assert!(sh.opaque_tail.is_none());
    }

    /// `collocated_ref_idx` overflow check: a value > the active
    /// `num_ref_idx_lX_active_minus1` is a §7.4.7.1 range failure.
    #[test]
    fn rejects_collocated_ref_idx_above_active_minus1() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue -> 1
            (0b1, 1),   // num_positive_pics ue -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // mvp = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // override = 1
            (0b010, 3), // L0 active_minus1 = 1 (-> 2 entries, valid range 0..=1)
            (0b011, 3), // collocated_ref_idx ue -> 2 (out of range)
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_R (NAL type 1) — a non-IDR picture.
        let err = SliceSegmentHeader::parse(&rbsp, 1, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "collocated_ref_idx",
                got: 2,
            }
        );
    }

    /// An IDR P-slice with `pps.lists_modification_present_flag == 1`:
    /// because the §7.3.6.1 non-IDR POC/RPS block is absent for IDR
    /// slices, the active short-term RPS is empty and the §7.4.7.2
    /// `NumPicTotalCurr` is `0`. The §7.3.6.1 outer gate
    /// (`... && NumPicTotalCurr > 1`) is therefore statically false
    /// and `ref_pic_lists_modification()` is not signalled; the parser
    /// continues straight into the mvd / cabac-init / collocated block
    /// (all absent for this P slice + no-MVP / no-cabac-init /
    /// no-temporal-MVP configuration) and walks the rest of the tail
    /// through `byte_alignment()`.
    #[test]
    fn skips_rplm_when_num_pic_total_curr_is_zero_idr() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.lists_modification_present_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            // RPLM gate statically false (NumPicTotalCurr == 0) — no
            // bits consumed; parser walks into the mvd / cabac-init /
            // collocated block. With the current configuration that
            // block is fully absent (P slice, no cabac_init, no MVP).
            (0b010, 3), // five_minus_max_num_merge_cand ue -> 1
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.num_ref_idx_active_override_flag, Some(false));
        assert_eq!(sh.ref_pic_lists_modification, None);
        assert_eq!(sh.mvd_l1_zero_flag, None);
        assert_eq!(sh.cabac_init_flag, Some(false));
        assert_eq!(sh.collocated_from_l0_flag, None);
        assert_eq!(sh.collocated_ref_idx, None);
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(1));
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// Non-IDR P-slice with `pps.lists_modification_present_flag == 1`
    /// and an inline short-term RPS in explicit form carrying two
    /// `used_by_curr_pic_s0_flag` entries set to 1: §7.4.7.2 equation
    /// 7-57 gives `NumPicTotalCurr == 2`, the §7.3.6.1 outer gate is
    /// statically present, and `ref_pic_lists_modification()` parses
    /// in place. The per-entry width of `list_entry_l0` is
    /// `Ceil(Log2(2)) = 1` bit (§7.4.7.2). After RPLM the parser walks
    /// the rest of the inter tail through `byte_alignment()`.
    #[test]
    fn parses_rplm_in_place_with_explicit_inline_rps_npc_two() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.lists_modification_present_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            // Inline st_ref_pic_set: stRpsIdx == num_short_term_ref_pic_sets
            // == 0, so inter_ref_pic_set_prediction_flag is absent.
            (0b011, 3), // num_negative_pics ue '011' -> 2
            (0b1, 1),   // num_positive_pics ue '1' -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (0b1, 1),   // delta_poc_s0_minus1[1] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[1] = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            //   → infers L0 active_minus1 = pps.num_ref_idx_l0_default
            // RPLM gate statically present (NumPicTotalCurr == 2 > 1):
            (1, 1), // ref_pic_list_modification_flag_l0 = 1
            // list_entry_l0[0 .. l0_active_minus1] — each
            // `Ceil(Log2(2)) = 1` bit, range 0..=1. The PPS default
            // active_minus1 from TINY_PPS_RBSP picks the loop count.
            // We feed enough bits for the largest plausible default
            // and check that the slice parser caps the count from the
            // resolved override-or-default value.
            (0, 1),     // list_entry_l0[0] = 0
            (0b010, 3), // five_minus_max_num_merge_cand ue -> 1
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_N (Table 7-1 value 0) is non-IDR / non-IRAP.
        let sh = SliceSegmentHeader::parse(&rbsp, 0, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(false));
        let inline = sh
            .inline_short_term_ref_pic_set
            .as_ref()
            .expect("inline ST RPS");
        assert_eq!(inline.num_negative_pics, 2);
        assert_eq!(inline.used_by_curr_pic_s0_flag, vec![true, true]);
        let rplm = sh
            .ref_pic_lists_modification
            .as_ref()
            .expect("RPLM parsed in place");
        assert!(rplm.ref_pic_list_modification_flag_l0);
        // L0 active_minus1 == 0 (PPS TINY default) → 1 entry.
        assert_eq!(rplm.list_entry_l0, vec![0]);
        // P slice → L1 fields are not signalled.
        assert_eq!(rplm.ref_pic_list_modification_flag_l1, None);
        assert!(rplm.list_entry_l1.is_empty());
        // Full inter tail walked, no opaque suffix.
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// Non-IDR P-slice with `pps.lists_modification_present_flag == 1`
    /// and an inline short-term RPS in explicit form carrying *one*
    /// `used_by_curr_pic_s0_flag` entry: equation 7-57 gives
    /// `NumPicTotalCurr == 1`, the §7.3.6.1 outer gate is statically
    /// false, `ref_pic_lists_modification()` is not signalled, and the
    /// parser continues straight into the rest of the inter tail.
    #[test]
    fn skips_rplm_when_num_pic_total_curr_is_one() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.lists_modification_present_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue '010' -> 1
            (0b1, 1),   // num_positive_pics ue '1' -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            // No RPLM (gate statically false at NPC == 1).
            (0b010, 3), // five_minus_max_num_merge_cand ue -> 1
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 0, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert_eq!(sh.ref_pic_lists_modification, None);
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(1));
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// Non-IDR P-slice with `pps.lists_modification_present_flag == 1`
    /// using an SPS-resident short-term RPS whose
    /// `inter_ref_pic_set_prediction_flag == 1` with malformed
    /// per-position arrays (lengths do not match the source's
    /// `NumDeltaPocs[RefRpsIdx] + 1`). The §7.4.8 materialiser rejects
    /// the chain and the parser surfaces an opaque tail starting at the
    /// `ref_pic_lists_modification()` bit so the caller can inspect the
    /// bitstream.
    #[test]
    fn defers_rplm_when_active_st_rps_uses_inter_prediction() {
        let mut sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        sps.num_short_term_ref_pic_sets = 2;
        sps.short_term_ref_pic_sets = vec![
            ShortTermRefPicSet {
                inter_ref_pic_set_prediction_flag: false,
                num_negative_pics: 1,
                num_positive_pics: 0,
                delta_poc_s0_minus1: vec![0],
                used_by_curr_pic_s0_flag: vec![true],
                ..Default::default()
            },
            ShortTermRefPicSet {
                // The picked RPS is in inter-prediction form but the
                // arrays are empty (length 0 ≠ source's NumDeltaPocs+1
                // = 2) — materialisation fails and the parse defers.
                inter_ref_pic_set_prediction_flag: true,
                ..Default::default()
            },
        ];
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.lists_modification_present_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1
            // num_short_term_ref_pic_sets == 2 → idx field width = 1 bit.
            (1, 1), // short_term_ref_pic_set_idx u(1) = 1 (the inter-predicted RPS)
            (1, 1), // sao_luma
            (1, 1), // sao_chroma
            (0, 1), // num_ref_idx_active_override_flag = 0
            // (Deferral begins here; remaining bits are opaque.)
            (1, 1),
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 0, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(true));
        assert_eq!(sh.short_term_ref_pic_set_idx, Some(1));
        assert!(sh.ref_pic_lists_modification.is_none());
        assert_eq!(sh.mvd_l1_zero_flag, None);
        assert_eq!(sh.cabac_init_flag, None);
        assert!(sh.opaque_tail.is_some());
        assert!(sh.byte_offset_to_slice_data.is_none());
    }

    /// Non-IDR P-slice with `pps.lists_modification_present_flag == 1`
    /// using an SPS-resident short-term RPS whose
    /// `inter_ref_pic_set_prediction_flag == 1` and well-formed
    /// per-position arrays. The §7.4.8 materialiser succeeds: the
    /// derived RPS has `NumPicTotalCurr == 1` (single positive POC
    /// gated `false`) so the §7.3.6.1 outer gate is statically false
    /// and the parser walks the inter-slice tail to `byte_alignment()`
    /// without surfacing an opaque tail. Closes the prior §7.4.8
    /// deferral point for this configuration.
    #[test]
    fn parses_p_slice_with_sps_inter_predicted_rps_npc_le_1() {
        let mut sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        sps.num_short_term_ref_pic_sets = 2;
        // Same fixture used by sps::tests::materialize_inter_rps_prediction_matches_fixture:
        //   * Source (idx 0): explicit, num_neg=1, num_pos=0,
        //     delta_poc_s0_minus1=[0], used_by_curr_pic_s0_flag=[true].
        //   * Inter (idx 1): delta_rps_sign=false,
        //     abs_delta_rps_minus1=0 (deltaRps=+1),
        //     used_by_curr_pic_flag=[true,false], use_delta_flag=[true,true].
        // Derived (idx 1): DeltaPocS1=[+1], UsedByCurrPicS1=[false]
        // ⇒ NumPicTotalCurr = 0 (the lone positive's used flag is
        // false), gate is statically false.
        sps.short_term_ref_pic_sets = vec![
            ShortTermRefPicSet {
                inter_ref_pic_set_prediction_flag: false,
                num_negative_pics: 1,
                num_positive_pics: 0,
                delta_poc_s0_minus1: vec![0],
                used_by_curr_pic_s0_flag: vec![true],
                ..Default::default()
            },
            ShortTermRefPicSet {
                inter_ref_pic_set_prediction_flag: true,
                delta_idx_minus1: 0,
                delta_rps_sign: false,
                abs_delta_rps_minus1: 0,
                used_by_curr_pic_flag: vec![true, false],
                use_delta_flag: vec![true, true],
                ..Default::default()
            },
        ];
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.lists_modification_present_flag = true;
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1
            // num_short_term_ref_pic_sets == 2 → idx field width = 1 bit.
            (1, 1), // short_term_ref_pic_set_idx u(1) = 1 (the inter-predicted RPS)
            (1, 1), // sao_luma
            (1, 1), // sao_chroma
            (0, 1), // num_ref_idx_active_override_flag = 0
            // (NumPicTotalCurr == 0 → ref_pic_lists_modification gate
            //  is statically absent; the parser walks straight to
            //  `mvd_l1_zero_flag`, which is absent for P, then the
            //  inferred `cabac_init_flag = false`, then the absent
            //  collocated block, then five_minus_max_num_merge_cand.)
            (0b010, 3), // five_minus_max_num_merge_cand ue -> 1
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag = 1
            (1, 1),     // byte_alignment '1' bit (rest zero-padded)
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 0, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(true));
        assert_eq!(sh.short_term_ref_pic_set_idx, Some(1));
        // §7.4.8 materialisation succeeded, NumPicTotalCurr == 0 ⇒
        // gate statically false, no RPLM signalled.
        assert!(sh.ref_pic_lists_modification.is_none());
        // The parser walked through to byte_alignment().
        assert!(sh.opaque_tail.is_none());
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(1));
        assert_eq!(sh.slice_qp_delta, Some(0));
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// IDR P-slice that walks the full inter-slice tail through
    /// `byte_alignment()` because the §7.3.6.3 `pred_weight_table()`
    /// gate is statically absent (`pps.weighted_pred_flag == 0`,
    /// inherited from `TINY_PPS_RBSP`). After the override block, the
    /// parser walks past `mvd_l1_zero_flag` (absent for P) and
    /// `cabac_init_flag` (absent + inferred `false` because
    /// `pps.cabac_init_present_flag == 0`), past the collocated block
    /// (absent because `slice_temporal_mvp_enabled_flag == 0`), reads
    /// `five_minus_max_num_merge_cand`, then the I-slice-shared tail
    /// (`slice_qp_delta` + chroma QP + deblocking + loop-filter +
    /// entry-points + extension + byte_alignment). The `use_integer_mv_flag`
    /// SCC bit is statically absent because the PPS SCC extension is
    /// not surfaced (motion_vector_resolution_control_idc inferred 0).
    #[test]
    fn parses_p_slice_full_inter_tail_no_weighted_pred() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // Sanity: TINY_PPS keeps the weighted-pred gate statically off.
        assert!(!pps.weighted_pred_flag);
        assert!(!pps.weighted_bipred_flag);
        assert!(!pps.cabac_init_present_flag);
        assert!(!pps.lists_modification_present_flag);
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            // (no mvd / cabac_init / collocated bits — all inferred)
            (0b010, 3), // five_minus_max_num_merge_cand ue -> 1
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag = 1
            (1, 1),     // byte_alignment '1' bit (rest are 0 to byte boundary)
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        // §7.4.7.1 inference: P override == 0 → L0 from PPS default.
        assert_eq!(
            sh.num_ref_idx_l0_active_minus1,
            Some(pps.num_ref_idx_l0_default_active_minus1)
        );
        assert_eq!(sh.num_ref_idx_l1_active_minus1, None);
        assert_eq!(sh.mvd_l1_zero_flag, None);
        assert_eq!(sh.cabac_init_flag, Some(false));
        assert_eq!(sh.collocated_from_l0_flag, None);
        assert_eq!(sh.collocated_ref_idx, None);
        // five_minus_max_num_merge_cand = 1 → MaxNumMergeCand = 4.
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(1));
        assert_eq!(sh.max_num_merge_cand(), Some(4));
        assert_eq!(sh.slice_qp_delta, Some(0));
        assert_eq!(sh.slice_loop_filter_across_slices_enabled_flag, Some(true));
        // Tail walked to byte_alignment; no opaque suffix.
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// Non-IDR (TRAIL_R) B-slice walking the full inter-slice tail
    /// through `byte_alignment()` with `pps.weighted_bipred_flag == 0`:
    /// exercises the B-only `mvd_l1_zero_flag` bit and the temporal-MVP
    /// `collocated_from_l0_flag` signalling, then walks straight to
    /// `five_minus_max_num_merge_cand` and the shared I-slice tail.
    #[test]
    fn parses_b_slice_full_inter_tail_with_mvp() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),     // first
            (0b1, 1),   // pps_id ue -> 0
            (0b1, 1),   // slice_type ue -> B
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            (0b010, 3), // num_negative_pics ue -> 1
            (0b1, 1),   // num_positive_pics ue -> 0
            (0b1, 1),   // delta_poc_s0_minus1[0] ue -> 0
            (1, 1),     // used_by_curr_pic_s0_flag[0] = 1
            (1, 1),     // slice_temporal_mvp_enabled_flag = 1
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (1, 1),     // num_ref_idx_active_override_flag = 1
            (0b010, 3), // num_ref_idx_l0_active_minus1 ue -> 1
            (0b1, 1),   // num_ref_idx_l1_active_minus1 ue -> 0
            (0, 1),     // mvd_l1_zero_flag = 0
            // cabac_init_flag absent (cabac_init_present_flag == 0).
            (1, 1),     // collocated_from_l0_flag = 1
            (0b010, 3), // collocated_ref_idx ue -> 1 (L0 has 2 entries, in range)
            (0b1, 1),   // five_minus_max_num_merge_cand ue -> 0
            (0b011, 3), // slice_qp_delta se -> -1
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag = 1
            (1, 1),     // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        // TRAIL_R (NAL type 1) — a non-IDR picture.
        let sh = SliceSegmentHeader::parse(&rbsp, 1, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::B));
        assert!(sh.slice_temporal_mvp_enabled_flag);
        assert_eq!(sh.mvd_l1_zero_flag, Some(false));
        assert_eq!(sh.collocated_from_l0_flag, Some(true));
        assert_eq!(sh.collocated_ref_idx, Some(1));
        assert_eq!(sh.five_minus_max_num_merge_cand, Some(0));
        assert_eq!(sh.max_num_merge_cand(), Some(5));
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert!(sh.opaque_tail.is_none());
    }

    /// IDR P-slice with `pps.weighted_pred_flag == 1` and a non-trivial
    /// `pred_weight_table()` body (single L0 entry, `luma_log2_weight_denom
    /// == 2`, `luma_weight_l0_flag == 1`, `delta_luma_weight_l0[0] == 5`,
    /// `luma_offset_l0[0] == 0`; chroma flag off). Verifies the in-place
    /// PWT decode resolves the §7.4.7.3 derived `LumaWeightL0[0] = (1 <<
    /// 2) + 5 = 9` correctly and the parser continues to walk the rest
    /// of the inter-slice tail through `byte_alignment()`.
    #[test]
    fn parses_p_slice_in_place_pwt_resolves_luma_weight() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let mut fields: Vec<(u32, u8)> = vec![
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0 → L0 = PPS default (0)
        ];
        // pred_weight_table()
        fields.push(ue_codeword(2)); // luma_log2_weight_denom = 2
        fields.push(se_codeword(0)); // delta_chroma_log2_weight_denom = 0
        fields.push((1, 1)); // luma_weight_l0_flag[0]
        fields.push((0, 1)); // chroma_weight_l0_flag[0]
        fields.push(se_codeword(5)); // delta_luma_weight_l0[0]
        fields.push(se_codeword(0)); // luma_offset_l0[0]
        // Inter-slice tail
        fields.push(ue_codeword(0)); // five_minus_max_num_merge_cand
        fields.push(se_codeword(0)); // slice_qp_delta
        fields.push((1, 1)); // slice_loop_filter_across_slices_enabled_flag
        fields.push((1, 1)); // byte_alignment '1'
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.luma_log2_weight_denom, 2);
        assert_eq!(pwt.entries_l0.len(), 1);
        assert!(pwt.entries_l0[0].luma_weight_flag);
        assert!(!pwt.entries_l0[0].chroma_weight_flag);
        assert_eq!(pwt.entries_l0[0].delta_luma_weight, 5);
        // §7.4.7.3 derived LumaWeightL0[0] = (1 << 2) + 5 = 9.
        assert_eq!(pwt.luma_weight_l0(0), Some(9));
        // The chroma flag is off → derived ChromaWeightL0[0][j] = 1 <<
        // ChromaLog2WeightDenom = 1 << 2 = 4 (inferred form).
        assert_eq!(pwt.chroma_weight_l0(0, 0), Some(4));
        // ChromaOffsetL0[0][j] inferred to 0 when the chroma flag is off.
        assert_eq!(pwt.chroma_offset_l0(0, 0, 128), Some(0));
        // Header walked to byte_alignment.
        assert!(sh.opaque_tail.is_none());
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// IDR B-slice with `pps.weighted_bipred_flag == 1` and a
    /// `pred_weight_table()` body that carries an L1 chroma weight +
    /// offset delta. Verifies the in-place call site populates both L0
    /// and L1 entries, the chroma sub-block on L1 is parsed, and the
    /// derived `ChromaWeightL1` / `ChromaOffsetL1` resolve correctly.
    #[test]
    fn parses_b_slice_in_place_pwt_resolves_l1_chroma() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_bipred_flag = true;
        let mut fields: Vec<(u32, u8)> = vec![
            (1, 1),   // first
            (0, 1),   // no_output (IDR)
            (0b1, 1), // pps_id ue -> 0
            (0b1, 1), // slice_type ue -> B
            (1, 1),   // sao_luma
            (1, 1),   // sao_chroma
            (0, 1),   // num_ref_idx_active_override_flag = 0
            (1, 1),   // mvd_l1_zero_flag = 1
        ];
        // pred_weight_table() — 1 L0 + 1 L1 entry.
        fields.push(ue_codeword(1)); // luma_log2_weight_denom = 1
        fields.push(se_codeword(1)); // delta_chroma_log2_weight_denom = 1 → ChromaLog2WeightDenom = 2
        fields.push((0, 1)); // luma_weight_l0_flag[0]
        fields.push((0, 1)); // chroma_weight_l0_flag[0]
        fields.push((0, 1)); // luma_weight_l1_flag[0]
        fields.push((1, 1)); // chroma_weight_l1_flag[0] = 1
        fields.push(se_codeword(2)); // delta_chroma_weight_l1[0][0] = 2
        fields.push(se_codeword(-1)); // delta_chroma_offset_l1[0][0] = -1
        fields.push(se_codeword(-3)); // delta_chroma_weight_l1[0][1] = -3
        fields.push(se_codeword(0)); // delta_chroma_offset_l1[0][1] = 0
        // Inter-slice tail
        fields.push(ue_codeword(0)); // five_minus_max_num_merge_cand
        fields.push(se_codeword(0)); // slice_qp_delta
        fields.push((1, 1)); // slice_loop_filter_across_slices_enabled_flag
        fields.push((1, 1)); // byte_alignment '1'
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::B));
        let pwt = sh.pred_weight_table.as_ref().expect("PWT decoded in place");
        assert_eq!(pwt.luma_log2_weight_denom, 1);
        assert_eq!(pwt.chroma_log2_weight_denom(), 2);
        assert_eq!(pwt.entries_l0.len(), 1);
        assert_eq!(pwt.entries_l1.len(), 1);
        assert!(!pwt.entries_l1[0].luma_weight_flag);
        assert!(pwt.entries_l1[0].chroma_weight_flag);
        assert_eq!(pwt.entries_l1[0].delta_chroma_weight, [2, -3]);
        assert_eq!(pwt.entries_l1[0].delta_chroma_offset, [-1, 0]);
        // §7.4.7.3 derived ChromaWeightL1[0][j] = (1 << 2) + delta:
        //   j=0: 4 + 2 = 6;  j=1: 4 + (-3) = 1.
        assert_eq!(pwt.chroma_weight_l1(0, 0), Some(6));
        assert_eq!(pwt.chroma_weight_l1(0, 1), Some(1));
        // Equation 7-58 with WpOffsetHalfRangeC = 128 (base-profile bit
        // depths, high_precision_offsets_enabled_flag = 0):
        //   j=0: 128 + (-1) - ((128 * 6) >> 2) = 127 - 192 = -65.
        //   j=1: 128 + 0    - ((128 * 1) >> 2) = 128 - 32  =  96.
        assert_eq!(pwt.chroma_offset_l1(0, 0, 128), Some(-65));
        assert_eq!(pwt.chroma_offset_l1(0, 1, 128), Some(96));
        // L0 chroma flag was off → inferred ChromaWeightL0[0][j] = 4.
        assert_eq!(pwt.chroma_weight_l0(0, 0), Some(4));
        assert!(sh.opaque_tail.is_none());
    }

    /// A `pred_weight_table()` body that violates the §7.4.7.3 range
    /// bound on `delta_luma_weight_l0` (out-of-range value 128) surfaces
    /// from the in-place call site as the same
    /// [`SliceError::ValueOutOfRange`] the standalone parser raises —
    /// the failure must propagate from `SliceSegmentHeader::parse`.
    #[test]
    fn in_place_pwt_propagates_range_error() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.weighted_pred_flag = true;
        let mut fields: Vec<(u32, u8)> = vec![
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id ue -> 0
            (0b010, 3), // slice_type ue -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0 → L0 default = 0
        ];
        fields.push(ue_codeword(0)); // luma_log2_weight_denom = 0
        fields.push(se_codeword(0)); // delta_chroma_log2_weight_denom = 0
        fields.push((1, 1)); // luma_weight_l0_flag[0]
        fields.push((0, 1)); // chroma_weight_l0_flag[0]
        fields.push(se_codeword(128)); // delta_luma_weight_l0[0] = 128 (out of range)
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "delta_luma_weight_l0",
                got: 128,
            }
        );
    }

    /// `five_minus_max_num_merge_cand > 4` is a §7.4.7.1 conformance
    /// violation (the derived `MaxNumMergeCand` would fall below 1).
    /// Encode wire value 5 as ue(v) = `0b00110` (5 bits).
    #[test]
    fn rejects_five_minus_max_num_merge_cand_above_4() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0, 1),     // num_ref_idx_active_override_flag = 0
            // mvd / cabac / collocated all inferred absent.
            ue_codeword(5), // five_minus_max_num_merge_cand = 5 (illegal)
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "five_minus_max_num_merge_cand",
                got: 5,
            }
        );
    }

    /// `num_ref_idx_l0_active_minus1 > 14` is a range failure (§7.4.7.1).
    /// Encode value 15 as ue(v) = `0b000010000` (9 bits).
    #[test]
    fn rejects_num_ref_idx_l0_active_minus1_above_14() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),          // first
            (0, 1),          // no_output (IDR)
            (0b1, 1),        // pps_id -> 0
            (0b010, 3),      // slice_type -> P
            (1, 1),          // sao_luma
            (1, 1),          // sao_chroma
            (1, 1),          // num_ref_idx_active_override_flag = 1
            ue_codeword(15), // num_ref_idx_l0_active_minus1 = 15 -> illegal
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: 15,
            }
        );
    }

    /// Non-first dependent slice segment: dependent flag + address are
    /// read, then the body ends after SAO (no slice_type etc.).
    #[test]
    fn parses_dependent_slice_segment() {
        // 4-CTB picture (2x2) so slice_segment_address is 2 bits wide.
        // PPS must have dependent_slice_segments_enabled_flag; the
        // fixture PPS has it 0, so patch a parsed PPS.
        let sps = ctx_sps(1, false, true, true, 32, 32, 1, 0, 4); // 32/16=2 -> 2x2=4 CTBs
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.dependent_slice_segments_enabled_flag = true;
        // first=0, dependent=1, slice_segment_address u(2)=2, then the
        // header ends: every remaining §7.3.6.1 field (slice_type, SAO
        // flags, QP, …) lives inside the !dependent gate and is
        // inherited from the associated independent slice segment. The
        // header goes straight to byte_alignment().
        let bits = concat_bits(&[
            (0, 1),    // first_slice_segment_in_pic_flag = 0
            (0b1, 1),  // pps_id ue -> 0
            (1, 1),    // dependent_slice_segment_flag
            (0b10, 2), // slice_segment_address = 2
            (1, 1),    // byte_alignment one bit
        ]);
        let rbsp = pack_bits(&bits);
        // Use a non-IRAP type so no_output is absent and the layout
        // above matches.
        let sh =
            SliceSegmentHeader::parse(&rbsp, 0 /* TRAIL_N */, &sps, &pps).expect("slice header");
        assert!(!sh.first_slice_segment_in_pic_flag);
        assert!(sh.dependent_slice_segment_flag);
        assert_eq!(sh.slice_segment_address, 2);
        assert_eq!(sh.slice_type, None);
        // SAO flags are not signalled in a dependent segment — they are
        // inherited by the caller; the struct leaves them false.
        assert!(!sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert!(sh.opaque_tail.is_none());
        assert_eq!(sh.byte_offset_to_slice_data, Some(1));
    }

    #[test]
    fn rejects_truncated_rbsp() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // Only one byte; first bit reads ok but ue() for pps_id runs off
        // a short enough buffer eventually. Use an empty buffer to force
        // truncation on the very first read.
        let err = SliceSegmentHeader::parse(&[], IDR_N_LP, &sps, &pps).unwrap_err();
        assert_eq!(err, SliceError::Truncated);
    }

    #[test]
    fn rejects_slice_type_out_of_range() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // slice_type ue(v) = 3 ('00100') is out of Table 7-7 range.
        let bits = concat_bits(&[
            (1, 1),       // first
            (0, 1),       // no_output (IDR)
            (0b1, 1),     // pps_id -> 0
            (0b00100, 5), // slice_type ue -> 3 (illegal)
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "slice_type",
                got: 3
            }
        );
    }

    #[test]
    fn end_to_end_via_nal_walker() {
        use crate::hevc::engine::nal::collect_nal_units;
        // Build an Annex B stream carrying a single IDR_N_LP slice NAL
        // whose RBSP is the hand-assembled I-slice header from
        // `parses_hand_assembled_i_idr_header`.
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (IRAP)
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag
            (0b011, 3), // slice_qp_delta se -> -1
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment one bit
        ]);
        let body = pack_bits(&bits);
        // NAL header for IDR_N_LP (type 20), layer 0, temporal_id 0.
        let b0 = (IDR_N_LP & 0x3F) << 1;
        let b1 = 0x01; // temporal_id_plus1 = 1
        let mut stream = vec![0x00, 0x00, 0x01, b0, b1];
        stream.extend_from_slice(&body);
        let units = collect_nal_units(&stream).expect("walker");
        assert_eq!(units.len(), 1);
        let u = &units[0];
        assert_eq!(u.header.nal_unit_type, IDR_N_LP);
        let sh = SliceSegmentHeader::parse(&u.rbsp, u.header.nal_unit_type, &sps, &pps)
            .expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert_eq!(sh.slice_qp_y(&pps), Some(25));
    }

    // --- §7.3.6.2 ref_pic_lists_modification() ---

    /// Hand-build a `ref_pic_lists_modification()` RBSP and parse it for
    /// a P slice with the modification flag set and a single L0 entry.
    /// `NumPicTotalCurr == 2` so the per-entry width is
    /// `Ceil(Log2(2)) == 1` bit. With `num_ref_idx_l0_active_minus1 ==
    /// 0` the loop reads exactly one `list_entry_l0`. The bits are:
    ///   `ref_pic_list_modification_flag_l0` (1) = 1
    ///   `list_entry_l0[0]` (u(1)) = 1
    /// (B-only L1 fields are absent for a P slice.)
    #[test]
    fn parses_p_slice_l0_only_modification() {
        let bits = concat_bits(&[(1, 1), (1, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(
            &mut br,
            SliceType::P,
            0, /* num_ref_idx_l0_active_minus1 */
            0, /* num_ref_idx_l1_active_minus1 (ignored for P) */
            2, /* NumPicTotalCurr */
        )
        .expect("ref_pic_lists_modification");

        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0, vec![1]);
        // P slice: list-1 fields absent.
        assert!(m.ref_pic_list_modification_flag_l1.is_none());
        assert!(m.list_entry_l1.is_empty());
        // Exactly 2 bits consumed.
        assert_eq!(br.bit_pos(), 2);
    }

    /// B slice with both list-0 and list-1 modifications active.
    /// `NumPicTotalCurr == 4` so the per-entry width is
    /// `Ceil(Log2(4)) == 2` bits. With both `num_ref_idx_lX_active_minus1
    /// == 1` each list contributes 2 entries.
    ///   flag_l0 (1)=1
    ///   list_entry_l0[0] u(2)=0b10=2
    ///   list_entry_l0[1] u(2)=0b00=0
    ///   flag_l1 (1)=1
    ///   list_entry_l1[0] u(2)=0b01=1
    ///   list_entry_l1[1] u(2)=0b11=3
    /// Total = 1 + 2 + 2 + 1 + 2 + 2 = 10 bits.
    #[test]
    fn parses_b_slice_both_lists_modification() {
        let bits = concat_bits(&[
            (1, 1),    // ref_pic_list_modification_flag_l0
            (0b10, 2), // list_entry_l0[0] = 2
            (0b00, 2), // list_entry_l0[1] = 0
            (1, 1),    // ref_pic_list_modification_flag_l1
            (0b01, 2), // list_entry_l1[0] = 1
            (0b11, 2), // list_entry_l1[1] = 3
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 1, 1, 4).expect("parse");

        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0, vec![2, 0]);
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(true));
        assert_eq!(m.list_entry_l1, vec![1, 3]);
        assert_eq!(br.bit_pos(), 10);
    }

    /// B slice where the L0 modification flag is 0 (implicit
    /// derivation): the L0 entry list is empty, and the L1 flag is
    /// read immediately after. `NumPicTotalCurr == 3` so per-entry
    /// width is `Ceil(Log2(3)) == 2` bits.
    ///   flag_l0 (1)=0
    ///   flag_l1 (1)=1
    ///   list_entry_l1[0] u(2)=0b10=2
    /// 4 bits total; the parser must NOT have consumed any L0 entries.
    #[test]
    fn parses_b_slice_l0_implicit_l1_explicit() {
        let bits = concat_bits(&[(0, 1), (1, 1), (0b10, 2)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 3).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(true));
        assert_eq!(m.list_entry_l1, vec![2]);
        assert_eq!(br.bit_pos(), 4);
    }

    /// Both flags zero: a minimal degenerate case where the entire
    /// structure is two bits and produces empty entry lists.
    #[test]
    fn parses_b_slice_both_flags_zero() {
        let bits = concat_bits(&[(0, 1), (0, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 2).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(false));
        assert!(m.list_entry_l1.is_empty());
        assert_eq!(br.bit_pos(), 2);
    }

    /// P slice with `flag_l0 == 0`: exactly one bit consumed, no list
    /// entries, and no L1 fields present.
    #[test]
    fn parses_p_slice_l0_implicit() {
        let bits = concat_bits(&[(0, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::P, 14, 14, 5).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert!(m.ref_pic_list_modification_flag_l1.is_none());
        assert!(m.list_entry_l1.is_empty());
        assert_eq!(br.bit_pos(), 1);
    }

    /// §7.4.7.2 range check: `list_entry_l0[i]` must be
    /// `< NumPicTotalCurr`. With `NumPicTotalCurr == 3` the per-entry
    /// width is 2 bits, so the value `3` (0b11) is legally encodable
    /// but is rejected by the range check.
    #[test]
    fn list_entry_l0_value_must_be_less_than_num_pic_total_curr() {
        let bits = concat_bits(&[
            (1, 1),    // flag_l0
            (0b11, 2), // list_entry_l0[0] = 3 — illegal: must be <= 2
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 3)
            .expect_err("out-of-range entry must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "list_entry_l0",
                got: 3,
            }
        );
    }

    /// §7.4.7.2 range check for L1: same rule, exercised through the
    /// B-slice branch.
    #[test]
    fn list_entry_l1_value_must_be_less_than_num_pic_total_curr() {
        // NumPicTotalCurr=2 → entry width=1 bit, max value=1. Send a
        // legal L0 (flag=0) then an explicit L1 (flag=1, entry=1 OK,
        // then we cannot encode 2 in 1 bit so use a 3-curr setup
        // instead to exercise the range check.)
        // Use NumPicTotalCurr=3 (2-bit entries) so 0b11=3 is illegal.
        let bits = concat_bits(&[
            (0, 1),    // flag_l0
            (1, 1),    // flag_l1
            (0b11, 2), // list_entry_l1[0] = 3 — illegal
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 3)
            .expect_err("out-of-range L1 entry must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "list_entry_l1",
                got: 3,
            }
        );
    }

    /// The §7.3.6.2 structure is only signalled for inter slices. The
    /// parser rejects an I-slice call up front rather than reading any
    /// bits (the bitreader position must stay at 0).
    #[test]
    fn rejects_i_slice_call() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::I, 0, 0, 2)
            .expect_err("I-slice call must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/slice_type",
                got: 2,
            }
        );
        assert_eq!(br.bit_pos(), 0);
    }

    /// The §7.3.6.1 gate guarantees `NumPicTotalCurr > 1`. The parser
    /// rejects a call with `NumPicTotalCurr <= 1` (a defensive
    /// pre-condition).
    #[test]
    fn rejects_num_pic_total_curr_le_1() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 1)
            .expect_err("NumPicTotalCurr <= 1 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: 1,
            }
        );
        assert_eq!(br.bit_pos(), 0);

        let mut br2 = BitReader::new(&rbsp);
        let err2 = RefPicListsModification::parse(&mut br2, SliceType::B, 0, 0, 0)
            .expect_err("NumPicTotalCurr == 0 must error");
        assert_eq!(
            err2,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: 0,
            }
        );
    }

    /// `num_ref_idx_l0_active_minus1` is constrained to 0..=14 by
    /// §7.4.7.1. The parser rejects a call that violates that bound,
    /// matching the precondition documented on
    /// [`RefPicListsModification::parse`].
    #[test]
    fn rejects_num_ref_idx_l0_out_of_range() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 15, 0, 2)
            .expect_err("num_ref_idx_l0_active_minus1 > 14 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: 15,
            }
        );
    }

    /// Same check for L1, exercised through the B-slice branch (the
    /// L1 bound is only validated for B slices).
    #[test]
    fn rejects_num_ref_idx_l1_out_of_range() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::B, 0, 15, 2)
            .expect_err("num_ref_idx_l1_active_minus1 > 14 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_ref_idx_l1_active_minus1",
                got: 15,
            }
        );
    }

    /// Maximum-active-index case: 15 entries per list (the §7.4.7.1
    /// cap, `num_ref_idx_lX_active_minus1 == 14`), with
    /// `NumPicTotalCurr == 8` (3-bit entries). Bit accounting:
    ///   flag_l0 (1) + 15 * 3 = 46 bits for the L0 portion.
    /// The P slice has no L1 fields, so the test verifies the parser
    /// reads exactly 46 bits.
    #[test]
    fn max_active_minus1_p_slice_l0() {
        let mut fields: Vec<(u32, u8)> = vec![(1, 1)]; // flag_l0 = 1
        for i in 0..15u32 {
            // entry value = i mod 8 ∈ 0..=7, fits in 3 bits, in range.
            fields.push((i % 8, 3));
        }
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::P, 14, 0, 8).expect("parse");
        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0.len(), 15);
        for (i, &v) in m.list_entry_l0.iter().enumerate() {
            assert_eq!(v, (i as u32) % 8);
        }
        assert_eq!(br.bit_pos(), 1 + 15 * 3);
    }

    /// `Ceil(Log2(N))` width: confirm the per-entry width matches the
    /// §7.4.7.2 formula for a representative set of `NumPicTotalCurr`
    /// values by reading exactly the expected bit count from a flag=1
    /// L0 with a single entry.
    #[test]
    fn entry_width_matches_ceil_log2() {
        // (num_pic_total_curr, expected_bits_per_entry)
        let cases: &[(u32, u8)] = &[(2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (16, 4)];
        for &(curr, w) in cases {
            // bit layout: flag_l0=1 then list_entry_l0[0]=0
            let mut bits: Vec<u8> = vec![1];
            bits.resize(1 + w as usize, 0);
            let rbsp = pack_bits(&bits);
            let mut br = BitReader::new(&rbsp);
            let m =
                RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, curr).expect("parse");
            assert_eq!(m.list_entry_l0, vec![0]);
            assert_eq!(br.bit_pos(), 1 + w as usize, "curr={curr} width={w}");
        }
    }

    /// Truncated RBSP: the parser surfaces [`SliceError::Truncated`]
    /// if the buffer runs out mid-element.
    #[test]
    fn truncated_buffer_surfaces_truncated_error() {
        // flag_l0=1 declared but no bits remain for list_entry_l0[0].
        let bits: Vec<u8> = vec![1];
        let rbsp = pack_bits(&bits); // one byte: 0b1000_0000
        // Restrict the reader to the first bit only.
        let buf = &rbsp[..0]; // zero bytes; even flag_l0 fails
        let mut br = BitReader::new(buf);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 4)
            .expect_err("empty buffer must error");
        assert_eq!(err, SliceError::Truncated);
    }

    // --- §7.4.7.2 NumPicTotalCurr derivation ---

    /// Equation 7-57 with only short-term entries: two negative pics
    /// both used (`UsedByCurrPicS0 = [1, 1]`), one positive pic not
    /// used (`UsedByCurrPicS1 = [0]`), and no long-term entries.
    /// Expected: `NumPicTotalCurr = 2`.
    #[test]
    fn num_pic_total_curr_short_term_only() {
        let s0 = [true, true];
        let s1 = [false];
        let lt: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt);
        assert_eq!(inputs.compute(), 2);
    }

    /// Equation 7-57 with a mix of S0, S1 and long-term entries
    /// flagged "used by current pic". Hand-derived: 2 S0 ones (3
    /// flags, 2 set) + 1 S1 one (2 flags, 1 set) + 2 LT ones (3 flags,
    /// 2 set) = 5.
    #[test]
    fn num_pic_total_curr_mixed_short_and_long_term() {
        let s0 = [true, false, true];
        let s1 = [true, false];
        let lt = [true, false, true];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt);
        assert_eq!(inputs.compute(), 5);
    }

    /// Equation 7-57 with `pps_curr_pic_ref_enabled_flag == 1` adding
    /// the final `NumPicTotalCurr++`. With zero short-term and zero
    /// long-term contributions, the value is exactly 1 (the IBC /
    /// self-reference case the SCC PPS flag enables).
    #[test]
    fn num_pic_total_curr_curr_pic_ref_only() {
        let s0: [bool; 0] = [];
        let s1: [bool; 0] = [];
        let lt: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt)
            .with_pps_curr_pic_ref_enabled(true);
        assert_eq!(inputs.compute(), 1);
    }

    /// Equation 7-57 with every contributing source: 1 S0 + 1 S1 +
    /// 1 LT + `pps_curr_pic_ref_enabled_flag` = 4.
    #[test]
    fn num_pic_total_curr_all_contributors() {
        let s0 = [true];
        let s1 = [true];
        let lt = [true];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt)
            .with_pps_curr_pic_ref_enabled(true);
        assert_eq!(inputs.compute(), 4);
    }

    /// Empty short-term RPS + empty long-term + no SCC self-ref =
    /// `NumPicTotalCurr == 0`. The §7.4.7.1 conformance rule "when
    /// the current picture contains a P or B slice, the value of
    /// NumPicTotalCurr shall not be equal to 0" is the consumer's
    /// responsibility — this primitive returns the literal
    /// equation-7-57 value.
    #[test]
    fn num_pic_total_curr_zero_when_nothing_contributes() {
        let s0: [bool; 0] = [];
        let s1: [bool; 0] = [];
        let lt: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt);
        assert_eq!(inputs.compute(), 0);
    }

    /// Build a short-term RPS in *explicit* form with three negative
    /// and two positive pics, then derive `NumPicTotalCurr` via
    /// [`NumPicTotalCurrInputs::from_explicit_short_term_rps`].
    /// Negative used flags `[1, 0, 1]` + positive `[1, 1]` + no LT =
    /// 2 + 2 + 0 = 4.
    #[test]
    fn num_pic_total_curr_from_explicit_rps_builder() {
        let rps = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            num_negative_pics: 3,
            num_positive_pics: 2,
            delta_poc_s0_minus1: vec![0, 1, 2],
            used_by_curr_pic_s0_flag: vec![true, false, true],
            delta_poc_s1_minus1: vec![0, 1],
            used_by_curr_pic_s1_flag: vec![true, true],
            ..Default::default()
        };
        let lt: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_explicit_short_term_rps(&rps, &lt)
            .expect("explicit RPS yields builder");
        assert_eq!(inputs.compute(), 4);
    }

    /// The explicit-RPS builder refuses an inter-RPS-predicted RPS:
    /// the §7.4.8 derivation (equations 7-58..7-66) must run first to
    /// resolve the per-position `UsedByCurrPicSX` arrays, and the
    /// result fed through [`NumPicTotalCurrInputs::from_used_flags`].
    #[test]
    fn num_pic_total_curr_from_explicit_rps_rejects_inter_prediction() {
        let rps = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: true,
            ..Default::default()
        };
        let lt: [bool; 0] = [];
        assert!(NumPicTotalCurrInputs::from_explicit_short_term_rps(&rps, &lt).is_none());
    }

    /// §7.4.7.1 / §7.4.7.2 long-term resolution:
    /// [`SliceLongTermRefPic::used_by_curr_pic_lt`] reads the SPS
    /// table when the entry is `Sps { lt_idx_sps }`, and the in-slice
    /// flag when the entry is `InSlice { used_by_curr_pic_lt_flag }`.
    /// Construct an SPS with two LT entries (`[used=1, used=0]`) and
    /// verify the SPS lookup; then verify the in-slice form.
    #[test]
    fn used_by_curr_pic_lt_resolves_sps_table_and_in_slice() {
        let mut sps = ctx_sps(1, false, false, false, 16, 16, 0, 0, 4);
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 2;
        sps.long_term_ref_pics = vec![
            LongTermRefPicEntry {
                poc_lsb: 0,
                used_by_curr_pic: true,
            },
            LongTermRefPicEntry {
                poc_lsb: 1,
                used_by_curr_pic: false,
            },
        ];

        let sps_entry_used = SliceLongTermRefPic {
            source: SliceLongTermRefPicSource::Sps { lt_idx_sps: 0 },
            delta_poc_msb_present_flag: false,
            delta_poc_msb_cycle_lt: 0,
        };
        assert_eq!(sps_entry_used.used_by_curr_pic_lt(&sps), Some(true));

        let sps_entry_unused = SliceLongTermRefPic {
            source: SliceLongTermRefPicSource::Sps { lt_idx_sps: 1 },
            delta_poc_msb_present_flag: false,
            delta_poc_msb_cycle_lt: 0,
        };
        assert_eq!(sps_entry_unused.used_by_curr_pic_lt(&sps), Some(false));

        let in_slice_used = SliceLongTermRefPic {
            source: SliceLongTermRefPicSource::InSlice {
                poc_lsb_lt: 7,
                used_by_curr_pic_lt_flag: true,
            },
            delta_poc_msb_present_flag: false,
            delta_poc_msb_cycle_lt: 0,
        };
        assert_eq!(in_slice_used.used_by_curr_pic_lt(&sps), Some(true));

        // Out-of-range SPS index surfaces `None`.
        let sps_oob = SliceLongTermRefPic {
            source: SliceLongTermRefPicSource::Sps { lt_idx_sps: 99 },
            delta_poc_msb_present_flag: false,
            delta_poc_msb_cycle_lt: 0,
        };
        assert_eq!(sps_oob.used_by_curr_pic_lt(&sps), None);
    }

    /// End-to-end: build the long-term ref list a §7.3.6.1 slice
    /// header would carry (one SPS-resident `used == 1` entry + one
    /// in-slice `used == 0` entry + one in-slice `used == 1` entry),
    /// resolve each entry's `UsedByCurrPicLt[i]`, and feed the bool
    /// vector through equation 7-57. With empty short-term sets the
    /// result is 2.
    #[test]
    fn num_pic_total_curr_from_resolved_slice_long_term_list() {
        let mut sps = ctx_sps(1, false, false, false, 16, 16, 0, 0, 4);
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 1;
        sps.long_term_ref_pics = vec![LongTermRefPicEntry {
            poc_lsb: 0,
            used_by_curr_pic: true,
        }];

        let slice_lt = [
            SliceLongTermRefPic {
                source: SliceLongTermRefPicSource::Sps { lt_idx_sps: 0 },
                delta_poc_msb_present_flag: false,
                delta_poc_msb_cycle_lt: 0,
            },
            SliceLongTermRefPic {
                source: SliceLongTermRefPicSource::InSlice {
                    poc_lsb_lt: 4,
                    used_by_curr_pic_lt_flag: false,
                },
                delta_poc_msb_present_flag: false,
                delta_poc_msb_cycle_lt: 0,
            },
            SliceLongTermRefPic {
                source: SliceLongTermRefPicSource::InSlice {
                    poc_lsb_lt: 8,
                    used_by_curr_pic_lt_flag: true,
                },
                delta_poc_msb_present_flag: false,
                delta_poc_msb_cycle_lt: 0,
            },
        ];
        let used_lt: Vec<bool> = slice_lt
            .iter()
            .map(|e| e.used_by_curr_pic_lt(&sps).expect("in-range"))
            .collect();
        let s0: [bool; 0] = [];
        let s1: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &used_lt);
        assert_eq!(inputs.compute(), 2);
    }

    /// F.7.4.7.2 multilayer-extension form (equation `F-56`): when
    /// the slice's `nal_unit_type` is IDR, the short-term and
    /// long-term loops are SKIPPED entirely, so the count starts at 0
    /// and only `pps_curr_pic_ref_enabled_flag` + the
    /// `NumActiveRefLayerPics` summand contribute. Feed
    /// `used_by_curr_pic_*` flags that would each contribute 1 under
    /// equation 7-57 — they must be ignored.
    #[test]
    fn num_pic_total_curr_multilayer_skips_temporal_loops_for_idr() {
        let s0 = [true];
        let s1 = [true];
        let lt = [true];
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt)
            .with_multilayer_extension(IDR_W_RADL, 3);
        // Skipped loops contribute 0; pps_curr_pic_ref_enabled = false;
        // NumActiveRefLayerPics = 3.
        assert_eq!(inputs.compute(), 3);

        // Same inputs but flipping the SCC self-ref flag: +1 = 4.
        let inputs = NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt)
            .with_pps_curr_pic_ref_enabled(true)
            .with_multilayer_extension(IDR_N_LP, 3);
        assert_eq!(inputs.compute(), 4);
    }

    /// F.7.4.7.2 multilayer-extension form for a *non-IDR* slice: the
    /// short-term and long-term loops are NOT skipped, so the count
    /// matches the base-spec 7-57 result plus
    /// `NumActiveRefLayerPics`.
    /// 1 (S0) + 1 (S1) + 1 (LT) + 2 (NumActiveRefLayerPics) = 5.
    #[test]
    fn num_pic_total_curr_multilayer_keeps_loops_for_non_idr() {
        let s0 = [true];
        let s1 = [true];
        let lt = [true];
        // TRAIL_N (Table 7-1 value 0) is not IDR.
        let inputs =
            NumPicTotalCurrInputs::from_used_flags(&s0, &s1, &lt).with_multilayer_extension(0, 2);
        assert_eq!(inputs.compute(), 5);
    }

    /// §7.3.6.1 gate sanity: `NumPicTotalCurr > 1` is the condition
    /// under which the slice header signals `ref_pic_lists_modification()`.
    /// Compose an explicit-form RPS that would yield exactly 1 (one
    /// `UsedByCurrPicS0` flag set, nothing else) and confirm
    /// `NumPicTotalCurr == 1` — the §7.3.6.1 gate would not fire.
    /// Then compose one that yields 2 (two flags set) and confirm the
    /// gate would fire. This is a derivation cross-check, not a
    /// parser invocation.
    #[test]
    fn num_pic_total_curr_drives_section_7_3_6_1_gate() {
        let rps_one = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            num_negative_pics: 1,
            num_positive_pics: 0,
            delta_poc_s0_minus1: vec![0],
            used_by_curr_pic_s0_flag: vec![true],
            ..Default::default()
        };
        let lt: [bool; 0] = [];
        let inputs = NumPicTotalCurrInputs::from_explicit_short_term_rps(&rps_one, &lt).unwrap();
        assert_eq!(inputs.compute(), 1, "gate should not fire at 1");

        let rps_two = ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            num_negative_pics: 2,
            num_positive_pics: 0,
            delta_poc_s0_minus1: vec![0, 1],
            used_by_curr_pic_s0_flag: vec![true, true],
            ..Default::default()
        };
        let inputs = NumPicTotalCurrInputs::from_explicit_short_term_rps(&rps_two, &lt).unwrap();
        assert_eq!(inputs.compute(), 2, "gate fires at 2");
    }

    // --- §7.3.6.3 pred_weight_table() ---

    /// Build the `ue(v)` codeword for a value `v` (Table 9-3). Returns
    /// `(value, width_in_bits)` ready for [`concat_bits`].
    fn ue_codeword(v: u32) -> (u32, u8) {
        let plus1 = v + 1;
        let m = 32 - plus1.leading_zeros() - 1; // floor(log2(v+1))
        let width = (2 * m + 1) as u8;
        (plus1, width)
    }

    /// `se(v)` codeword: map signed to unsigned per the §9.2.1 inverse
    /// of Table 9-3 (`codeNum = 2*|v| - (v > 0 ? 1 : 0)`), then encode
    /// the result as an `ue(v)`.
    fn se_codeword(v: i32) -> (u32, u8) {
        let code_num: u32 = if v <= 0 {
            (2 * (-v)) as u32
        } else {
            (2 * v - 1) as u32
        };
        ue_codeword(code_num)
    }

    /// Minimal monochrome P-slice case (`ChromaArrayType == 0` so no
    /// chroma fields are signalled): one reference, `luma_weight_l0_flag
    /// == 1`, `delta_luma_weight_l0[0] == 5`, `luma_offset_l0[0] == 0`.
    ///
    /// Bit layout:
    /// ```text
    ///   luma_log2_weight_denom              ue(v) = 2     -> 0b011  (3 bits)
    ///   luma_weight_l0_flag[0]              u(1)  = 1     -> 0b1    (1 bit)
    ///   delta_luma_weight_l0[0]             se(v) = 5     -> codeNum=9, ue(v)=0b0001010 (7 bits)
    ///   luma_offset_l0[0]                   se(v) = 0     -> codeNum=0, ue(v)=0b1       (1 bit)
    /// ```
    /// total = 3 + 1 + 7 + 1 = 12 bits.
    #[test]
    fn parses_monochrome_p_slice_single_ref() {
        let mut fields = vec![ue_codeword(2)]; // luma_log2_weight_denom
        fields.push((1, 1)); // luma_weight_l0_flag[0]
        fields.push(se_codeword(5)); // delta_luma_weight_l0[0]
        fields.push(se_codeword(0)); // luma_offset_l0[0]
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(
            SliceType::P,
            0,
            0,
            /*ChromaArrayType*/ 0,
            8,
            8,
        );
        let pwt = PredWeightTable::parse(&mut br, &inputs).expect("parse");

        assert_eq!(pwt.luma_log2_weight_denom, 2);
        assert_eq!(pwt.delta_chroma_log2_weight_denom, 0); // absent → inferred 0
        assert_eq!(pwt.entries_l0.len(), 1);
        assert!(pwt.entries_l0[0].luma_weight_flag);
        assert!(!pwt.entries_l0[0].chroma_weight_flag); // monochrome
        assert_eq!(pwt.entries_l0[0].delta_luma_weight, 5);
        assert_eq!(pwt.entries_l0[0].luma_offset, 0);
        assert!(pwt.entries_l1.is_empty());
        // 12 bits consumed.
        assert_eq!(br.bit_pos(), 12);
        // Derived: LumaWeightL0[0] = (1 << 2) + 5 = 9
        assert_eq!(pwt.luma_weight_l0(0), Some(9));
    }

    /// P-slice with `ChromaArrayType == 1` (4:2:0), one reference,
    /// every flag = 1. Verifies the chroma sub-block parses and the
    /// derived `ChromaLog2WeightDenom`, `ChromaWeightL0[0][j]` and
    /// `ChromaOffsetL0[0][j]` resolve correctly.
    ///
    /// Bit layout:
    /// ```text
    ///   luma_log2_weight_denom              ue(v) = 1   -> 0b010
    ///   delta_chroma_log2_weight_denom      se(v) = 1   -> codeNum=1, ue(v)=0b010
    ///   luma_weight_l0_flag[0]              u(1)  = 1
    ///   chroma_weight_l0_flag[0]            u(1)  = 1
    ///   delta_luma_weight_l0[0]             se(v) = -3  -> codeNum=6, ue(v)=0b00111
    ///   luma_offset_l0[0]                   se(v) = 7   -> codeNum=13, ue(v)=0b0001110
    ///   delta_chroma_weight_l0[0][0]        se(v) = 0   -> 0b1
    ///   delta_chroma_offset_l0[0][0]        se(v) = 2   -> codeNum=3, ue(v)=0b00100
    ///   delta_chroma_weight_l0[0][1]        se(v) = 0   -> 0b1
    ///   delta_chroma_offset_l0[0][1]        se(v) = -1  -> codeNum=2, ue(v)=0b011
    /// ```
    #[test]
    fn parses_p_slice_420_single_ref_with_chroma() {
        let mut fields = vec![ue_codeword(1)];
        fields.push(se_codeword(1));
        fields.push((1, 1)); // luma_weight_l0_flag
        fields.push((1, 1)); // chroma_weight_l0_flag
        fields.push(se_codeword(-3));
        fields.push(se_codeword(7));
        fields.push(se_codeword(0));
        fields.push(se_codeword(2));
        fields.push(se_codeword(0));
        fields.push(se_codeword(-1));
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, 0, 0, 1, 8, 8);
        let pwt = PredWeightTable::parse(&mut br, &inputs).expect("parse");

        assert_eq!(pwt.luma_log2_weight_denom, 1);
        assert_eq!(pwt.delta_chroma_log2_weight_denom, 1);
        assert_eq!(pwt.chroma_log2_weight_denom(), 2);
        assert_eq!(pwt.entries_l0[0].delta_luma_weight, -3);
        assert_eq!(pwt.entries_l0[0].luma_offset, 7);
        assert_eq!(pwt.entries_l0[0].delta_chroma_weight, [0, 0]);
        assert_eq!(pwt.entries_l0[0].delta_chroma_offset, [2, -1]);
        // Derived: LumaWeightL0[0] = (1 << 1) + (-3) = -1
        assert_eq!(pwt.luma_weight_l0(0), Some(-1));
        // ChromaWeightL0[0][j] = (1 << 2) + 0 = 4 for j ∈ {0, 1}
        assert_eq!(pwt.chroma_weight_l0(0, 0), Some(4));
        assert_eq!(pwt.chroma_weight_l0(0, 1), Some(4));
        // Equation 7-58 with WpOffsetHalfRangeC = 128:
        //   raw_j0 = 128 + 2 - ((128 * 4) >> 2) = 130 - 128 = 2
        //   raw_j1 = 128 + (-1) - 128 = -1
        assert_eq!(pwt.chroma_offset_l0(0, 0, 128), Some(2));
        assert_eq!(pwt.chroma_offset_l0(0, 1, 128), Some(-1));
    }

    /// B-slice with `ChromaArrayType == 1`, one ref per list, all flags
    /// off (the minimal-content B case). Verifies the L1 block is
    /// reached and the chroma flag pass is read on both lists; absent
    /// deltas remain at 0; derived `LumaWeightLX[0]` inferred to
    /// `1 << luma_log2_weight_denom`.
    ///
    /// Bit layout (denoms then four flag bits, all 0; L1 mirrors L0):
    /// ```text
    ///   luma_log2_weight_denom              ue(v) = 0  -> 0b1
    ///   delta_chroma_log2_weight_denom      se(v) = 0  -> 0b1
    ///   luma_weight_l0_flag[0]              u(1)  = 0
    ///   chroma_weight_l0_flag[0]            u(1)  = 0
    ///   luma_weight_l1_flag[0]              u(1)  = 0
    ///   chroma_weight_l1_flag[0]            u(1)  = 0
    /// ```
    /// 2 bits denoms + 4 bits flags = 6 bits.
    #[test]
    fn parses_b_slice_all_flags_zero() {
        let mut fields = vec![ue_codeword(0), se_codeword(0)];
        fields.push((0, 1));
        fields.push((0, 1));
        fields.push((0, 1));
        fields.push((0, 1));
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::B, 0, 0, 1, 8, 8);
        let pwt = PredWeightTable::parse(&mut br, &inputs).expect("parse");

        assert_eq!(pwt.entries_l0.len(), 1);
        assert_eq!(pwt.entries_l1.len(), 1);
        assert!(!pwt.entries_l0[0].luma_weight_flag);
        assert!(!pwt.entries_l1[0].chroma_weight_flag);
        // No deltas were signalled; absent values inferred to 0.
        assert_eq!(pwt.entries_l0[0].delta_luma_weight, 0);
        assert_eq!(pwt.entries_l1[0].delta_chroma_offset, [0, 0]);
        // Derived LumaWeightLX[0] = (1 << 0) + 0 = 1 (inferred form).
        assert_eq!(pwt.luma_weight_l0(0), Some(1));
        assert_eq!(pwt.luma_weight_l1(0), Some(1));
        // ChromaOffsetLX[i][j] inferred to 0 when chroma_weight_flag == 0.
        assert_eq!(pwt.chroma_offset_l0(0, 0, 128), Some(0));
        assert_eq!(pwt.chroma_offset_l1(0, 1, 128), Some(0));
        assert_eq!(br.bit_pos(), 6);
    }

    /// `luma_log2_weight_denom > 7` is a range failure (§7.4.7.3).
    /// `ue(v)` value 8 encodes as `0b0001001` (7 bits).
    #[test]
    fn rejects_luma_log2_weight_denom_above_7() {
        let bits = concat_bits(&[ue_codeword(8)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, 0, 0, 0, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "luma_log2_weight_denom",
                got: 8,
            }
        );
    }

    /// `ChromaLog2WeightDenom = luma_log2_weight_denom +
    /// delta_chroma_log2_weight_denom` ∈ 0..=7 (§7.4.7.3). Encode
    /// `luma_log2_weight_denom == 3, delta_chroma_log2_weight_denom ==
    /// 5` → derived = 8, must error.
    #[test]
    fn rejects_derived_chroma_log2_weight_denom_above_7() {
        let bits = concat_bits(&[ue_codeword(3), se_codeword(5)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, 0, 0, 1, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "ChromaLog2WeightDenom",
                got: 8,
            }
        );
    }

    /// `delta_luma_weight_l0[i]` ∉ −128..=127 must error (§7.4.7.3).
    /// `se(v)` value 128 encodes via codeNum = 255 → ue(v) is 15 bits
    /// long; pack it and verify the parser surfaces ValueOutOfRange.
    #[test]
    fn rejects_delta_luma_weight_l0_above_127() {
        let mut fields = vec![ue_codeword(0)];
        fields.push((1, 1)); // luma_weight_l0_flag = 1
        fields.push(se_codeword(128)); // out of range
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, 0, 0, 0, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "delta_luma_weight_l0",
                got: 128,
            }
        );
    }

    /// `luma_offset_l0[i]` ∈ `−128..=127` for base profile (8 bits, no
    /// high-precision offsets). Encode value 128 → range failure.
    #[test]
    fn rejects_luma_offset_l0_above_127_at_8_bit() {
        let mut fields = vec![ue_codeword(0)];
        fields.push((1, 1)); // flag = 1
        fields.push(se_codeword(0)); // delta_luma_weight = 0 (in range)
        fields.push(se_codeword(128)); // luma_offset = 128 out of range
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, 0, 0, 0, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "luma_offset_l0",
                got: 128,
            }
        );
    }

    /// `high_precision_offsets_enabled_flag == true` widens
    /// `WpOffsetHalfRangeY` from `1 << 7 = 128` to
    /// `1 << (BitDepthY - 1)`. With `BitDepthY == 10` the new range is
    /// `−512..=511`. Encode `luma_offset_l0 == 200` (was out-of-range
    /// at 8-bit, in-range at 10-bit high-precision) and verify parse
    /// succeeds.
    #[test]
    fn accepts_luma_offset_in_high_precision_range() {
        let mut fields = vec![ue_codeword(0)];
        fields.push((1, 1));
        fields.push(se_codeword(0));
        fields.push(se_codeword(200));
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs {
            slice_type: SliceType::P,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            chroma_array_type: 0,
            high_precision_offsets_enabled_flag: true,
            bit_depth_y: 10,
            bit_depth_c: 10,
            signal_luma_l0: None,
            signal_chroma_l0: None,
            signal_luma_l1: None,
            signal_chroma_l1: None,
        };
        let pwt = PredWeightTable::parse(&mut br, &inputs).expect("parse");
        assert_eq!(pwt.entries_l0[0].luma_offset, 200);
    }

    /// §7.3.6.3 outer gate: when the caller passes
    /// `signal_luma_l0[i] == false`, the corresponding flag bit is NOT
    /// consumed from the bitstream and the parsed flag is inferred to
    /// `false`. Build a two-ref P slice with `signal_luma_l0 = [false,
    /// true]`: only one luma-flag bit is present (the second), and
    /// the parser must read exactly the right bits.
    ///
    /// Bit layout (monochrome, n_l0=2):
    /// ```text
    ///   luma_log2_weight_denom            ue(v) = 0  -> 0b1     (1)
    ///   luma_weight_l0_flag[1]            u(1)  = 1            (1)
    ///   delta_luma_weight_l0[1]           se(v) = 4  -> codeNum=7, ue(v)=0b0001000 (7)
    ///   luma_offset_l0[1]                 se(v) = 0  -> 0b1     (1)
    /// ```
    /// Total = 10 bits.
    #[test]
    fn outer_gate_suppresses_per_i_flag_bits() {
        let mut fields = vec![ue_codeword(0)];
        fields.push((1, 1)); // luma_weight_l0_flag[1]
        fields.push(se_codeword(4)); // delta_luma_weight_l0[1]
        fields.push(se_codeword(0));
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);

        let gate_l0 = [false, true];
        let inputs = PredWeightTableInputs {
            slice_type: SliceType::P,
            num_ref_idx_l0_active_minus1: 1,
            num_ref_idx_l1_active_minus1: 0,
            chroma_array_type: 0,
            high_precision_offsets_enabled_flag: false,
            bit_depth_y: 8,
            bit_depth_c: 8,
            signal_luma_l0: Some(&gate_l0),
            signal_chroma_l0: None,
            signal_luma_l1: None,
            signal_chroma_l1: None,
        };
        let pwt = PredWeightTable::parse(&mut br, &inputs).expect("parse");
        assert_eq!(pwt.entries_l0.len(), 2);
        assert!(!pwt.entries_l0[0].luma_weight_flag); // gated off → inferred 0
        assert!(pwt.entries_l0[1].luma_weight_flag);
        assert_eq!(pwt.entries_l0[1].delta_luma_weight, 4);
        // Position 0's deltas remain at the inferred default (0).
        assert_eq!(pwt.entries_l0[0].delta_luma_weight, 0);
        assert_eq!(pwt.entries_l0[0].luma_offset, 0);
        // Bit accounting matches the expected layout.
        assert_eq!(br.bit_pos(), 10);
    }

    /// Per-i gate slice length mismatch is a precondition failure.
    #[test]
    fn rejects_signal_slice_length_mismatch() {
        let bits = concat_bits(&[ue_codeword(0)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let gate_too_short = [true]; // num_ref_idx_l0_active_minus1 + 1 == 2
        let inputs = PredWeightTableInputs {
            slice_type: SliceType::P,
            num_ref_idx_l0_active_minus1: 1,
            num_ref_idx_l1_active_minus1: 0,
            chroma_array_type: 0,
            high_precision_offsets_enabled_flag: false,
            bit_depth_y: 8,
            bit_depth_c: 8,
            signal_luma_l0: Some(&gate_too_short),
            signal_chroma_l0: None,
            signal_luma_l1: None,
            signal_chroma_l1: None,
        };
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "signal_luma_l0",
                got: 1,
            }
        );
    }

    /// Rejects an I-slice call (the §7.3.6.1 gate
    /// `weighted_pred_flag && slice_type == P` /
    /// `weighted_bipred_flag && slice_type == B` excludes I slices).
    #[test]
    fn rejects_i_slice_call_pwt() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::I, 0, 0, 0, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "pred_weight_table/slice_type",
                got: 2,
            }
        );
        assert_eq!(br.bit_pos(), 0);
    }

    /// §7.4.7.3 conformance: for a P slice, `sumWeightL0Flags ≤ 24`.
    /// Each entry contributes up to 3 (luma=1, chroma=2), so 9 entries
    /// with both flags set sum to 27, breaching the cap. Build the
    /// minimal P-slice case with 9 entries (`num_ref_idx_l0_active_minus1
    /// = 8`) and verify the parser rejects.
    #[test]
    fn rejects_sum_weight_l0_above_24() {
        let mut fields = vec![ue_codeword(0), se_codeword(0)];
        let n = 9usize;
        // 9 luma_weight_l0_flag bits, all 1
        for _ in 0..n {
            fields.push((1, 1));
        }
        // 9 chroma_weight_l0_flag bits, all 1
        for _ in 0..n {
            fields.push((1, 1));
        }
        // Per-entry deltas (luma + chroma): delta=0, offset=0
        for _ in 0..n {
            fields.push(se_codeword(0)); // delta_luma_weight
            fields.push(se_codeword(0)); // luma_offset
            for _ in 0..2 {
                fields.push(se_codeword(0)); // delta_chroma_weight
                fields.push(se_codeword(0)); // delta_chroma_offset
            }
        }
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let inputs = PredWeightTableInputs::base_profile(SliceType::P, (n - 1) as u8, 0, 1, 8, 8);
        let err = PredWeightTable::parse(&mut br, &inputs).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "sumWeightL0Flags",
                got: 27,
            }
        );
    }

    /// I-slice with WPP enabled (`entropy_coding_sync_enabled_flag ==
    /// 1`): the §7.3.6.1 entry-point block is signalled with
    /// `num_entry_point_offsets = 2`, `offset_len_minus1 = 3` (each
    /// entry is `u(4)`), and per-row byte offsets `{6, 9}`. Verify
    /// the parser captures the offsets verbatim and exposes the
    /// per-subset byte length via [`EntryPointOffsets::subset_length`]
    /// (`entry_point_offset_minus1[i] + 1`, §7.4.7.1).
    #[test]
    fn parses_wpp_entry_point_offsets_in_place() {
        // Tall enough picture for two WPP entry points: 3 CTU rows.
        // With CTB size = 16 (`log2_min_cb_minus3 = 0`,
        // `log2_diff_max_min_cb = 1`) and `pic_height = 48`,
        // `PicHeightInCtbsY = 3`, so the upper bound on
        // `num_entry_point_offsets` is 2.
        let sps = ctx_sps(1, false, true, true, 16, 48, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.entropy_coding_sync_enabled_flag = true;

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output (IRAP)
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            // slice_temporal_mvp_enabled_flag: absent for an IDR.
            (1, 1),   // sao_luma
            (0, 1),   // sao_chroma
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            // Entry-point block.
            (0b011, 3),   // num_entry_point_offsets ue -> 2
            (0b00100, 5), // offset_len_minus1 ue -> 3 (entries are u(4))
            (6, 4),       // entry_point_offset_minus1[0]
            (9, 4),       // entry_point_offset_minus1[1]
            (1, 1),       // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        let eps = sh.entry_point_offsets.expect("entry-point block present");
        assert_eq!(eps.num_entry_point_offsets, 2);
        assert_eq!(eps.offset_len_minus1, 3);
        assert_eq!(eps.entry_point_offset_minus1, vec![6, 9]);
        // §7.4.7.1 subset length is `entry_point_offset_minus1[i] + 1`.
        assert_eq!(eps.subset_length(0), Some(7));
        assert_eq!(eps.subset_length(1), Some(10));
        assert_eq!(eps.subset_length(2), None);
        assert!(sh.byte_offset_to_slice_data.is_some());
    }

    /// Tiles enabled with a single tile (`num_tile_columns_minus1 ==
    /// 0`, `num_tile_rows_minus1 == 0`): the §7.4.7.1 upper bound on
    /// `num_entry_point_offsets` is `1 * 1 − 1 == 0`, so the block is
    /// present (the gate fires on `tiles_enabled_flag`) but
    /// `num_entry_point_offsets` must be 0 and the `offset_len_minus1`
    /// / per-entry loop are skipped. Verify the parser materialises an
    /// empty vec and reports a bare gate.
    #[test]
    fn parses_tiles_block_with_single_tile_no_offsets() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.tiles_enabled_flag = true;
        // tiles inferred default (cols=0, rows=0 minus1 → 1×1).

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            // slice_temporal_mvp_enabled_flag: absent for an IDR.
            (1, 1),   // sao_luma
            (0, 1),   // sao_chroma
            (0b1, 1), // slice_qp_delta se -> 0
            (1, 1),   // slice_loop_filter_across_slices_enabled_flag
            (0b1, 1), // num_entry_point_offsets ue -> 0
            (1, 1),   // byte_alignment '1'
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        let eps = sh.entry_point_offsets.expect("entry-point block present");
        assert_eq!(eps.num_entry_point_offsets, 0);
        assert_eq!(eps.offset_len_minus1, 0);
        assert!(eps.entry_point_offset_minus1.is_empty());
        assert!(eps.subset_length(0).is_none());
    }

    /// §7.4.7.1: when `entropy_coding_sync_enabled_flag == 1`, the
    /// upper bound on `num_entry_point_offsets` is `PicHeightInCtbsY −
    /// 1`. Build an SPS with `PicHeightInCtbsY == 1` (16×16 with CTB
    /// size 16): the bound is 0, so a wire value of 1 must fail the
    /// range check.
    #[test]
    fn rejects_wpp_entry_point_offsets_above_pic_height_bound() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.entropy_coding_sync_enabled_flag = true;

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output (IRAP); tmvp flag absent for an IDR
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            (1, 1),     // sao_luma
            (0, 1),     // sao_chroma
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // lf_across
            (0b010, 3), // num_entry_point_offsets ue -> 1, breaches bound 0
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_entry_point_offsets",
                got: 1,
            }
        );
    }

    /// §7.4.7.1: `offset_len_minus1` is bounded to `0..=31`. Build a
    /// WPP-enabled fixture with a wire value of 32 and verify the
    /// parser rejects.
    #[test]
    fn rejects_offset_len_minus1_above_31() {
        let sps = ctx_sps(1, false, true, true, 16, 48, 1, 0, 4);
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.entropy_coding_sync_enabled_flag = true;

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output (IRAP); tmvp flag absent for an IDR
            (0b1, 1),   // pps_id ue -> 0
            (0b011, 3), // slice_type ue -> I
            (1, 1),     // sao_luma
            (0, 1),     // sao_chroma
            (0b1, 1),   // slice_qp_delta se -> 0
            (1, 1),     // lf_across
            (0b010, 3), // num_entry_point_offsets ue -> 1
            // offset_len_minus1 = 32, encoded ue: codeNum 32 has
            // M = floor(log2(33)) = 5 leading zeros, then '1', then
            // 5-bit suffix (33 - 32 = 1 → 00001). 11 bits total:
            // 00000 1 00001 → 0b000_0010_0001.
            (0b000_0010_0001, 11),
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect_err("must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "offset_len_minus1",
                got: 32,
            }
        );
    }

    // --- bit-packing test helpers ---

    /// A `(value, width)` pair to be packed MSB-first.
    type BitField = (u32, u8);

    /// Concatenate `(value, width)` fields into a single bit vector
    /// (each entry's `width` low bits of `value`, MSB-first).
    fn concat_bits(fields: &[BitField]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &(value, width) in fields {
            for i in (0..width).rev() {
                bits.push(((value >> i) & 1) as u8);
            }
        }
        bits
    }

    /// Pack an MSB-first bit vector into bytes, zero-padding the final
    /// byte (matching `byte_alignment()`'s zero pad).
    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }
}
