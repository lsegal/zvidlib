//! HRD parameters parser per ITU-T Rec. H.265 §E.2.2 / §E.2.3.
//!
//! Implements the `hrd_parameters( commonInfPresentFlag,
//! maxNumSubLayersMinus1 )` and `sub_layer_hrd_parameters( subLayerId )`
//! syntax structures. The semantics (§E.3.2 / §E.3.3) govern the
//! permitted ranges that are enforced inline as
//! [`HrdError::ValueOutOfRange`].
//!
//! The struct is reused by every HRD callsite: the VPS RBSP
//! (`hrd_parameters( cprms_present_flag[i], vps_max_sub_layers_minus1 )`,
//! §7.3.2.1) and the SPS VUI block (`hrd_parameters( 1,
//! sps_max_sub_layers_minus1 )`, §E.2.1). The VPS wraps a sequence of
//! these (one per `vps_num_hrd_parameters`); §7.3.2.1 prefixes each
//! entry with a `hrd_layer_set_idx[i]` `ue(v)` plus, for `i > 0`,
//! `cprms_present_flag[i]` `u(1)` — see [`VpsHrdEntry`].
//!
//! ## §E.2.2 layout summary
//!
//! ```text
//! hrd_parameters( commonInfPresentFlag, maxNumSubLayersMinus1 ) {
//!   if( commonInfPresentFlag ) {                                       // CommonInf
//!     nal_hrd_parameters_present_flag                u(1)
//!     vcl_hrd_parameters_present_flag                u(1)
//!     if( nal | vcl ) {
//!       sub_pic_hrd_params_present_flag              u(1)
//!       if( sub_pic_hrd_params_present_flag ) {
//!         tick_divisor_minus2                        u(8)
//!         du_cpb_removal_delay_increment_length_minus1 u(5)
//!         sub_pic_cpb_params_in_pic_timing_sei_flag  u(1)
//!         dpb_output_delay_du_length_minus1          u(5)
//!       }
//!       bit_rate_scale                               u(4)
//!       cpb_size_scale                               u(4)
//!       if( sub_pic_hrd_params_present_flag )
//!         cpb_size_du_scale                          u(4)
//!       initial_cpb_removal_delay_length_minus1      u(5)
//!       au_cpb_removal_delay_length_minus1           u(5)
//!       dpb_output_delay_length_minus1               u(5)
//!     }
//!   }
//!   for( i = 0; i <= maxNumSubLayersMinus1; i++ ) {                    // SubLayer
//!     fixed_pic_rate_general_flag[i]                 u(1)
//!     if( !general_flag )
//!       fixed_pic_rate_within_cvs_flag[i]            u(1)
//!     if( fixed_pic_rate_within_cvs_flag[i] )
//!       elemental_duration_in_tc_minus1[i]           ue(v)    // 0..=2047
//!     else
//!       low_delay_hrd_flag[i]                        u(1)
//!     if( !low_delay_hrd_flag[i] )
//!       cpb_cnt_minus1[i]                            ue(v)    // 0..=31
//!     if( nal_hrd_parameters_present_flag )
//!       sub_layer_hrd_parameters( i )                          // §E.2.3
//!     if( vcl_hrd_parameters_present_flag )
//!       sub_layer_hrd_parameters( i )
//!   }
//! }
//! ```
//!
//! ## §E.2.3 layout summary
//!
//! ```text
//! sub_layer_hrd_parameters( subLayerId ) {
//!   for( i = 0; i < CpbCnt; i++ ) {
//!     bit_rate_value_minus1[i]                       ue(v)   // 0..=2^32-2
//!     cpb_size_value_minus1[i]                       ue(v)   // 0..=2^32-2
//!     if( sub_pic_hrd_params_present_flag ) {
//!       cpb_size_du_value_minus1[i]                  ue(v)   // 0..=2^32-2
//!       bit_rate_du_value_minus1[i]                  ue(v)   // 0..=2^32-2
//!     }
//!     cbr_flag[i]                                    u(1)
//!   }
//! }
//! ```

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::vps::HEVC_MAX_SUB_LAYERS;

/// Upper bound on `cpb_cnt_minus1[i]`. §E.3.2 mandates the range
/// 0..=31, so a CPB array has at most 32 entries.
pub const HEVC_MAX_CPB_CNT: usize = 32;

/// Upper bound on `elemental_duration_in_tc_minus1[i]`. §E.3.2 mandates
/// the range 0..=2047.
pub const HEVC_MAX_ELEMENTAL_DURATION_IN_TC_MINUS1: u32 = 2047;

/// Errors that can arise while parsing an `hrd_parameters()` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HrdError {
    /// The RBSP ran out of bits before the structure was fully parsed.
    Truncated,
    /// An Exp-Golomb code's `codeNum` exceeded the legal range of the
    /// containing syntax element.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value read.
        got: u32,
    },
    /// `maxNumSubLayersMinus1` supplied by the caller exceeded the
    /// §A.4 / `u(3)` ceiling of 6. Guarded so a malformed parent does
    /// not push the per-sub-layer array past `HEVC_MAX_SUB_LAYERS`.
    InvalidMaxNumSubLayers {
        /// The (out-of-range) value supplied by the caller.
        got: u8,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for HrdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("hrd_parameters() RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "syntax element {field} out of range: {got}")
            }
            Self::InvalidMaxNumSubLayers { got } => write!(
                f,
                "maxNumSubLayersMinus1 = {got} exceeds the HEVC u(3) ceiling of 6"
            ),
            Self::Bitstream(e) => write!(f, "bitstream error during hrd_parameters() parse: {e}"),
        }
    }
}

impl std::error::Error for HrdError {}

impl From<BitReaderError> for HrdError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

/// "Common information" block from §E.2.2, present when
/// `commonInfPresentFlag` is 1. Encapsulates the conditional sub-pic
/// HRD block and the four common length fields. `None` on a per-HRD
/// entry whose `cprms_present_flag` is 0 — the caller is then
/// responsible for inheriting the value from the previous entry per
/// §7.4.3.1 (the `( i − 1 )-th hrd_parameters( )` inheritance rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HrdCommonInfo {
    /// `nal_hrd_parameters_present_flag` (`u(1)`).
    pub nal_hrd_parameters_present_flag: bool,
    /// `vcl_hrd_parameters_present_flag` (`u(1)`).
    pub vcl_hrd_parameters_present_flag: bool,
    /// `sub_pic_hrd_params_present_flag` (`u(1)`). Only signalled when
    /// at least one of `nal_hrd_parameters_present_flag` or
    /// `vcl_hrd_parameters_present_flag` is 1; inferred to be 0
    /// otherwise per §E.3.2.
    pub sub_pic_hrd_params_present_flag: bool,
    /// `tick_divisor_minus2` (`u(8)`); only present when
    /// `sub_pic_hrd_params_present_flag` is 1.
    pub tick_divisor_minus2: u8,
    /// `du_cpb_removal_delay_increment_length_minus1` (`u(5)`); only
    /// present when `sub_pic_hrd_params_present_flag` is 1.
    pub du_cpb_removal_delay_increment_length_minus1: u8,
    /// `sub_pic_cpb_params_in_pic_timing_sei_flag` (`u(1)`); only
    /// present when `sub_pic_hrd_params_present_flag` is 1. Inferred
    /// to 0 otherwise per §E.3.2.
    pub sub_pic_cpb_params_in_pic_timing_sei_flag: bool,
    /// `dpb_output_delay_du_length_minus1` (`u(5)`); only present when
    /// `sub_pic_hrd_params_present_flag` is 1.
    pub dpb_output_delay_du_length_minus1: u8,
    /// `bit_rate_scale` (`u(4)`).
    pub bit_rate_scale: u8,
    /// `cpb_size_scale` (`u(4)`).
    pub cpb_size_scale: u8,
    /// `cpb_size_du_scale` (`u(4)`); only present when
    /// `sub_pic_hrd_params_present_flag` is 1.
    pub cpb_size_du_scale: u8,
    /// `initial_cpb_removal_delay_length_minus1` (`u(5)`). Inferred
    /// to 23 when absent per §E.3.2 — the parser reports the signalled
    /// value; the §E.3.2 inference is the caller's responsibility.
    pub initial_cpb_removal_delay_length_minus1: u8,
    /// `au_cpb_removal_delay_length_minus1` (`u(5)`). Inferred to 23
    /// when absent per §E.3.2.
    pub au_cpb_removal_delay_length_minus1: u8,
    /// `dpb_output_delay_length_minus1` (`u(5)`). Inferred to 23 when
    /// absent per §E.3.2.
    pub dpb_output_delay_length_minus1: u8,
}

impl HrdCommonInfo {
    /// `nal_hrd_parameters_present_flag` OR `vcl_hrd_parameters_present_flag`
    /// — gates the §E.2.2 inner block and the per-sub-layer
    /// `sub_layer_hrd_parameters()` invocations.
    pub fn has_any_hrd(&self) -> bool {
        self.nal_hrd_parameters_present_flag || self.vcl_hrd_parameters_present_flag
    }

    fn parse(br: &mut BitReader<'_>) -> Result<Self, HrdError> {
        let nal_hrd_parameters_present_flag = br.u1()? != 0;
        let vcl_hrd_parameters_present_flag = br.u1()? != 0;
        let mut info = Self {
            nal_hrd_parameters_present_flag,
            vcl_hrd_parameters_present_flag,
            sub_pic_hrd_params_present_flag: false,
            tick_divisor_minus2: 0,
            du_cpb_removal_delay_increment_length_minus1: 0,
            sub_pic_cpb_params_in_pic_timing_sei_flag: false,
            dpb_output_delay_du_length_minus1: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            cpb_size_du_scale: 0,
            initial_cpb_removal_delay_length_minus1: 0,
            au_cpb_removal_delay_length_minus1: 0,
            dpb_output_delay_length_minus1: 0,
        };
        if nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag {
            info.sub_pic_hrd_params_present_flag = br.u1()? != 0;
            if info.sub_pic_hrd_params_present_flag {
                info.tick_divisor_minus2 = br.u(8)? as u8;
                info.du_cpb_removal_delay_increment_length_minus1 = br.u(5)? as u8;
                info.sub_pic_cpb_params_in_pic_timing_sei_flag = br.u1()? != 0;
                info.dpb_output_delay_du_length_minus1 = br.u(5)? as u8;
            }
            info.bit_rate_scale = br.u(4)? as u8;
            info.cpb_size_scale = br.u(4)? as u8;
            if info.sub_pic_hrd_params_present_flag {
                info.cpb_size_du_scale = br.u(4)? as u8;
            }
            info.initial_cpb_removal_delay_length_minus1 = br.u(5)? as u8;
            info.au_cpb_removal_delay_length_minus1 = br.u(5)? as u8;
            info.dpb_output_delay_length_minus1 = br.u(5)? as u8;
        }
        Ok(info)
    }
}

/// One per-CPB entry from §E.2.3 (`sub_layer_hrd_parameters()`).
///
/// The four `*_minus1` fields are stored as `u32` because §E.3.3
/// mandates the range 0..=2^32 − 2, which would overflow a `u16` and
/// fits exactly in a `u32` `ue(v)` `codeNum`. The legality bounds
/// (`bit_rate_value_minus1[i]` strictly increases, `cpb_size_value_minus1[i]`
/// monotonically non-increases, and so on for the sub-pic variants) are
/// enforced by [`SubLayerHrdParameters::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpbEntry {
    /// `bit_rate_value_minus1[i]` (`ue(v)`, range 0..=2^32 − 2).
    pub bit_rate_value_minus1: u32,
    /// `cpb_size_value_minus1[i]` (`ue(v)`, range 0..=2^32 − 2).
    pub cpb_size_value_minus1: u32,
    /// `cpb_size_du_value_minus1[i]` (`ue(v)`, range 0..=2^32 − 2);
    /// `None` when the parent's `sub_pic_hrd_params_present_flag == 0`.
    pub cpb_size_du_value_minus1: Option<u32>,
    /// `bit_rate_du_value_minus1[i]` (`ue(v)`, range 0..=2^32 − 2);
    /// `None` when the parent's `sub_pic_hrd_params_present_flag == 0`.
    pub bit_rate_du_value_minus1: Option<u32>,
    /// `cbr_flag[i]` (`u(1)`).
    pub cbr_flag: bool,
}

/// Per-sub-layer HRD payload from §E.2.3
/// (`sub_layer_hrd_parameters( subLayerId )`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubLayerHrdParameters {
    /// One entry per `i` in `0..CpbCnt` (i.e. `cpb_cnt_minus1 + 1`
    /// entries). Bounded at [`HEVC_MAX_CPB_CNT`].
    pub cpb: Vec<CpbEntry>,
}

impl SubLayerHrdParameters {
    /// Parse a `sub_layer_hrd_parameters( subLayerId )` body. The
    /// caller passes the active `cpb_cnt = cpb_cnt_minus1 + 1` and the
    /// containing `sub_pic_hrd_params_present_flag` (which gates the
    /// per-CPB `cpb_size_du_value_minus1[i]` / `bit_rate_du_value_minus1[i]`
    /// pair).
    pub fn parse(
        br: &mut BitReader<'_>,
        cpb_cnt: u32,
        sub_pic_hrd_params_present: bool,
    ) -> Result<Self, HrdError> {
        // cpb_cnt = cpb_cnt_minus1 + 1; cpb_cnt_minus1 is in 0..=31, so
        // cpb_cnt is in 1..=32. Bound for safety against a corrupt
        // parent that supplies a larger value.
        if cpb_cnt == 0 || cpb_cnt as usize > HEVC_MAX_CPB_CNT {
            return Err(HrdError::ValueOutOfRange {
                field: "CpbCnt",
                got: cpb_cnt,
            });
        }
        let mut cpb = Vec::with_capacity(cpb_cnt as usize);
        let mut prev_bit_rate: Option<u32> = None;
        let mut prev_cpb_size: Option<u32> = None;
        let mut prev_cpb_size_du: Option<u32> = None;
        let mut prev_bit_rate_du: Option<u32> = None;
        for _ in 0..cpb_cnt {
            let bit_rate_value_minus1 = br.ue()?;
            // §E.3.3: "For any i > 0, bit_rate_value_minus1[ i ] shall
            // be greater than bit_rate_value_minus1[ i − 1 ]."
            if let Some(prev) = prev_bit_rate {
                if bit_rate_value_minus1 <= prev {
                    return Err(HrdError::ValueOutOfRange {
                        field: "bit_rate_value_minus1",
                        got: bit_rate_value_minus1,
                    });
                }
            }
            prev_bit_rate = Some(bit_rate_value_minus1);

            let cpb_size_value_minus1 = br.ue()?;
            // §E.3.3: "For any i greater than 0, cpb_size_value_minus1[ i ]
            // shall be less than or equal to cpb_size_value_minus1[ i − 1 ]."
            if let Some(prev) = prev_cpb_size {
                if cpb_size_value_minus1 > prev {
                    return Err(HrdError::ValueOutOfRange {
                        field: "cpb_size_value_minus1",
                        got: cpb_size_value_minus1,
                    });
                }
            }
            prev_cpb_size = Some(cpb_size_value_minus1);

            let (cpb_size_du_value_minus1, bit_rate_du_value_minus1) = if sub_pic_hrd_params_present
            {
                let du_cpb = br.ue()?;
                // §E.3.3: "For any i greater than 0,
                // cpb_size_du_value_minus1[ i ] shall be less than or
                // equal to cpb_size_du_value_minus1[ i − 1 ]."
                if let Some(prev) = prev_cpb_size_du {
                    if du_cpb > prev {
                        return Err(HrdError::ValueOutOfRange {
                            field: "cpb_size_du_value_minus1",
                            got: du_cpb,
                        });
                    }
                }
                prev_cpb_size_du = Some(du_cpb);
                let du_bit = br.ue()?;
                // §E.3.3: "For any i > 0, bit_rate_du_value_minus1[ i ]
                // shall be greater than bit_rate_du_value_minus1[ i − 1 ]."
                if let Some(prev) = prev_bit_rate_du {
                    if du_bit <= prev {
                        return Err(HrdError::ValueOutOfRange {
                            field: "bit_rate_du_value_minus1",
                            got: du_bit,
                        });
                    }
                }
                prev_bit_rate_du = Some(du_bit);
                (Some(du_cpb), Some(du_bit))
            } else {
                (None, None)
            };

            let cbr_flag = br.u1()? != 0;

            cpb.push(CpbEntry {
                bit_rate_value_minus1,
                cpb_size_value_minus1,
                cpb_size_du_value_minus1,
                bit_rate_du_value_minus1,
                cbr_flag,
            });
        }
        Ok(Self { cpb })
    }
}

/// One per-sub-layer entry of the §E.2.2 outer loop.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubLayerHrd {
    /// `fixed_pic_rate_general_flag[i]` (`u(1)`).
    pub fixed_pic_rate_general_flag: bool,
    /// `fixed_pic_rate_within_cvs_flag[i]` (`u(1)`); only signalled
    /// when `fixed_pic_rate_general_flag[i] == 0`. When
    /// `fixed_pic_rate_general_flag[i] == 1`, §E.3.2 infers this to 1
    /// — the parser pre-fills the inferred value rather than leaving an
    /// `Option`.
    pub fixed_pic_rate_within_cvs_flag: bool,
    /// `elemental_duration_in_tc_minus1[i]` (`ue(v)`, range 0..=2047);
    /// only signalled when `fixed_pic_rate_within_cvs_flag[i] == 1`.
    pub elemental_duration_in_tc_minus1: Option<u32>,
    /// `low_delay_hrd_flag[i]` (`u(1)`); only signalled when
    /// `fixed_pic_rate_within_cvs_flag[i] == 0`. §E.3.2 infers 0
    /// otherwise — pre-filled.
    pub low_delay_hrd_flag: bool,
    /// `cpb_cnt_minus1[i]` (`ue(v)`, range 0..=31); inferred to 0 when
    /// `low_delay_hrd_flag[i] == 1` (the syntax skips the `ue(v)` —
    /// pre-filled to match the spec text's "When not present, the value
    /// of cpb_cnt_minus1[ i ] is inferred to be equal to 0").
    pub cpb_cnt_minus1: u32,
    /// `sub_layer_hrd_parameters( i )` for the NAL HRD path; populated
    /// when the parent's `nal_hrd_parameters_present_flag == 1`.
    pub nal_hrd: Option<SubLayerHrdParameters>,
    /// `sub_layer_hrd_parameters( i )` for the VCL HRD path; populated
    /// when the parent's `vcl_hrd_parameters_present_flag == 1`.
    pub vcl_hrd: Option<SubLayerHrdParameters>,
}

/// Parsed `hrd_parameters( commonInfPresentFlag, maxNumSubLayersMinus1 )`
/// body per §E.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HrdParameters {
    /// `maxNumSubLayersMinus1` echoed back so consumers can iterate
    /// [`Self::sub_layers`] without re-deriving the bound. Capped at 6
    /// (the `u(3)` ceiling).
    pub max_num_sub_layers_minus1: u8,
    /// "Common information" block — present iff `commonInfPresentFlag`
    /// was 1 at this call site. When `None`, the caller must inherit
    /// the previous HRD entry's common info per §7.4.3.1.
    pub common: Option<HrdCommonInfo>,
    /// Per-sub-layer payload — one [`SubLayerHrd`] per `i` in
    /// `0..=maxNumSubLayersMinus1`. Length is therefore
    /// `max_num_sub_layers_minus1 + 1`, bounded at
    /// [`HEVC_MAX_SUB_LAYERS`].
    pub sub_layers: Vec<SubLayerHrd>,
}

impl HrdParameters {
    /// Parse `hrd_parameters( commonInfPresentFlag, maxNumSubLayersMinus1 )`
    /// starting at the current `br` position.
    ///
    /// `prev_common` is consulted when `common_inf_present == false`:
    /// the gates (`nal_hrd_parameters_present_flag`,
    /// `vcl_hrd_parameters_present_flag`,
    /// `sub_pic_hrd_params_present_flag`) drive the per-sub-layer
    /// branches even though no common-info bits are read at this call
    /// site. This matches the §7.4.3.1 contract: when
    /// `cprms_present_flag[i] == 0` the HRD parameters that are common
    /// for all sub-layers "are derived to be the same as the
    /// (i − 1)-th hrd_parameters( ) syntax structure".
    pub fn parse(
        br: &mut BitReader<'_>,
        common_inf_present: bool,
        max_num_sub_layers_minus1: u8,
        prev_common: Option<&HrdCommonInfo>,
    ) -> Result<Self, HrdError> {
        if max_num_sub_layers_minus1 as usize > HEVC_MAX_SUB_LAYERS {
            return Err(HrdError::InvalidMaxNumSubLayers {
                got: max_num_sub_layers_minus1,
            });
        }
        let common = if common_inf_present {
            Some(HrdCommonInfo::parse(br)?)
        } else {
            None
        };
        // Pick the effective common info: the freshly-parsed block if
        // present, else the inherited one from prev_common. When neither
        // is available, sub-layer parsing falls back to "no HRD gates",
        // which matches the spec's silent-default behaviour.
        let effective = common.as_ref().or(prev_common);
        let nal_hrd_present = effective.is_some_and(|c| c.nal_hrd_parameters_present_flag);
        let vcl_hrd_present = effective.is_some_and(|c| c.vcl_hrd_parameters_present_flag);
        let sub_pic_present = effective.is_some_and(|c| c.sub_pic_hrd_params_present_flag);

        let count = max_num_sub_layers_minus1 as usize + 1;
        let mut sub_layers = Vec::with_capacity(count);
        for _ in 0..count {
            let fixed_pic_rate_general_flag = br.u1()? != 0;
            // §E.3.2: when fixed_pic_rate_general_flag[i] == 1, the
            // value of fixed_pic_rate_within_cvs_flag[i] is inferred to
            // be equal to 1. Otherwise we read the explicit bit.
            let fixed_pic_rate_within_cvs_flag = if fixed_pic_rate_general_flag {
                true
            } else {
                br.u1()? != 0
            };
            let mut sl = SubLayerHrd {
                fixed_pic_rate_general_flag,
                fixed_pic_rate_within_cvs_flag,
                ..SubLayerHrd::default()
            };
            if fixed_pic_rate_within_cvs_flag {
                let v = br.ue()?;
                // §E.3.2: "elemental_duration_in_tc_minus1[ i ] shall
                // be in the range of 0 to 2 047, inclusive."
                if v > HEVC_MAX_ELEMENTAL_DURATION_IN_TC_MINUS1 {
                    return Err(HrdError::ValueOutOfRange {
                        field: "elemental_duration_in_tc_minus1",
                        got: v,
                    });
                }
                sl.elemental_duration_in_tc_minus1 = Some(v);
            } else {
                sl.low_delay_hrd_flag = br.u1()? != 0;
            }
            if !sl.low_delay_hrd_flag {
                let v = br.ue()?;
                // §E.3.2: "cpb_cnt_minus1[ i ] shall be in the range
                // of 0 to 31, inclusive."
                if v as usize >= HEVC_MAX_CPB_CNT {
                    return Err(HrdError::ValueOutOfRange {
                        field: "cpb_cnt_minus1",
                        got: v,
                    });
                }
                sl.cpb_cnt_minus1 = v;
            } else {
                // §E.3.2: "When not present, the value of
                // cpb_cnt_minus1[ i ] is inferred to be equal to 0."
                sl.cpb_cnt_minus1 = 0;
            }
            if nal_hrd_present {
                sl.nal_hrd = Some(SubLayerHrdParameters::parse(
                    br,
                    sl.cpb_cnt_minus1 + 1,
                    sub_pic_present,
                )?);
            }
            if vcl_hrd_present {
                sl.vcl_hrd = Some(SubLayerHrdParameters::parse(
                    br,
                    sl.cpb_cnt_minus1 + 1,
                    sub_pic_present,
                )?);
            }
            sub_layers.push(sl);
        }
        Ok(Self {
            max_num_sub_layers_minus1,
            common,
            sub_layers,
        })
    }
}

/// One entry of the VPS `for( i = 0; i < vps_num_hrd_parameters; i++ )`
/// loop in §7.3.2.1: the `hrd_layer_set_idx[i]` reference, the per-entry
/// `cprms_present_flag[i]` (`u(1)`, only present when `i > 0`), and the
/// parsed [`HrdParameters`] body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VpsHrdEntry {
    /// `hrd_layer_set_idx[i]` (`ue(v)`). §7.4.3.1 constrains this to
    /// `0..=vps_num_layer_sets_minus1` for `vps_base_layer_internal_flag == 1`,
    /// and `0..=vps_num_layer_sets_minus1 - 1` otherwise; the caller
    /// validates against the active VPS context.
    pub hrd_layer_set_idx: u32,
    /// `cprms_present_flag[i]` (`u(1)`). Inferred to 1 for `i == 0`
    /// per §7.4.3.1; this field carries the effective value either way.
    pub cprms_present_flag: bool,
    /// Parsed `hrd_parameters( cprms_present_flag[i], vps_max_sub_layers_minus1 )`
    /// body.
    pub hrd: HrdParameters,
}

impl VpsHrdEntry {
    /// Parse one entry of the VPS HRD loop. `index` is `i` (so the
    /// `cprms_present_flag[i]` `u(1)` is only consumed when `i > 0`),
    /// and `vps_max_sub_layers_minus1` plumbs through the
    /// `maxNumSubLayersMinus1` argument to `hrd_parameters()`.
    /// `prev` is the previous [`VpsHrdEntry`] (or `None` for `i == 0`)
    /// — its [`HrdCommonInfo`] is inherited when this entry's
    /// `cprms_present_flag[i] == 0`.
    pub fn parse(
        br: &mut BitReader<'_>,
        index: u32,
        vps_max_sub_layers_minus1: u8,
        prev: Option<&VpsHrdEntry>,
    ) -> Result<Self, HrdError> {
        let hrd_layer_set_idx = br.ue()?;
        // Spec-side allocation safety: hrd_layer_set_idx is bounded by
        // vps_num_layer_sets_minus1 + 1 (= 1024). Anything larger is
        // guaranteed-out-of-range regardless of VPS context; cap here
        // so the cross-check against the active VPS in §7.4.3.1 is
        // free to look at a believable value.
        if hrd_layer_set_idx > crate::hevc::engine::vps::HEVC_VPS_MAX_NUM_LAYER_SETS as u32 {
            return Err(HrdError::ValueOutOfRange {
                field: "hrd_layer_set_idx",
                got: hrd_layer_set_idx,
            });
        }
        let cprms_present_flag = if index > 0 { br.u1()? != 0 } else { true };
        let hrd = HrdParameters::parse(
            br,
            cprms_present_flag,
            vps_max_sub_layers_minus1,
            prev.map(|p| &p.hrd).and_then(|h| h.common.as_ref()),
        )?;
        Ok(Self {
            hrd_layer_set_idx,
            cprms_present_flag,
            hrd,
        })
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    /// Helper that turns a sequence of `'0'`/`'1'` bit strings into a
    /// packed RBSP byte buffer. Bits run MSB-first per H.265 §7.2.
    fn bits_to_bytes(parts: &[&str]) -> Vec<u8> {
        let mut bits = Vec::<u8>::new();
        for s in parts {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        }
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
        bytes
    }

    /// 0-th-order unsigned Exp-Golomb encoder per ITU-T Rec. H.265
    /// §9.2: emit `leading_zero_bits` zeros followed by `binary(code +
    /// 1)` on `leading_zero_bits + 1` bits. Returns the bit string
    /// (each char `'0'` or `'1'`) so [`bits_to_bytes`] can pack it.
    fn ue(code: u32) -> String {
        let v = code as u64 + 1;
        let bits_in_v = 64 - v.leading_zeros();
        let leading_zeros = bits_in_v - 1;
        let mut s = String::new();
        for _ in 0..leading_zeros {
            s.push('0');
        }
        for i in (0..bits_in_v).rev() {
            s.push(if (v >> i) & 1 == 1 { '1' } else { '0' });
        }
        s
    }

    /// Fixed-width binary string (`u(n)` literal) — `width` bits of
    /// `value`, MSB-first.
    fn ub(value: u32, width: u8) -> String {
        let mut s = String::new();
        for i in (0..width).rev() {
            s.push(if (value >> i) & 1 == 1 { '1' } else { '0' });
        }
        s
    }

    /// Minimal payload: commonInfPresentFlag = 1, every common gate 0,
    /// one sub-layer with general fixed-pic-rate flag set and inferred
    /// within-CVS flag, elemental_duration_in_tc_minus1 = 0, no per-CPB
    /// data. Tests the all-flags-off short path through §E.2.2.
    #[test]
    fn parses_minimal_common_info_one_sub_layer() {
        let bytes = bits_to_bytes(&[
            "0", // nal_hrd_parameters_present_flag = 0
            "0", // vcl_hrd_parameters_present_flag = 0
            // (nal | vcl) == 0 → no sub_pic_hrd_params block, no
            // bit_rate_scale / cpb_size_scale / lengths.
            "1", // fixed_pic_rate_general_flag[0] = 1
            // within_cvs_flag inferred = 1, so elemental_duration_in_tc_minus1[0] ue(v)
            "1", // ue(v) = 0
            // low_delay_hrd_flag not signalled (within_cvs == 1); inferred 0.
            // cpb_cnt_minus1 IS read because !low_delay_hrd_flag.
            "1", // cpb_cnt_minus1 ue(v) = 0
                 // no NAL/VCL HRD bodies (gates = 0)
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 0, None).expect("parse");
        let common = hrd.common.expect("common present");
        assert!(!common.nal_hrd_parameters_present_flag);
        assert!(!common.vcl_hrd_parameters_present_flag);
        assert!(!common.has_any_hrd());
        assert_eq!(hrd.sub_layers.len(), 1);
        let sl = &hrd.sub_layers[0];
        assert!(sl.fixed_pic_rate_general_flag);
        assert!(sl.fixed_pic_rate_within_cvs_flag);
        assert_eq!(sl.elemental_duration_in_tc_minus1, Some(0));
        assert_eq!(sl.cpb_cnt_minus1, 0);
        assert!(sl.nal_hrd.is_none());
        assert!(sl.vcl_hrd.is_none());
    }

    /// `nal_hrd_parameters_present_flag = 1` plus
    /// `sub_pic_hrd_params_present_flag = 1`: walks every common-info
    /// branch and one per-sub-layer NAL HRD body with two CPBs.
    #[test]
    fn parses_nal_hrd_with_sub_pic_and_two_cpbs() {
        // Plan:
        // - common: nal = 1, vcl = 0, sub_pic = 1
        //   - tick_divisor_minus2 = 3 (u(8) = 00000011)
        //   - du_cpb_removal_delay_increment_length_minus1 = 4 (u(5) = 00100)
        //   - sub_pic_cpb_params_in_pic_timing_sei_flag = 1
        //   - dpb_output_delay_du_length_minus1 = 5 (u(5) = 00101)
        //   - bit_rate_scale = 2 (u(4) = 0010)
        //   - cpb_size_scale = 3 (u(4) = 0011)
        //   - cpb_size_du_scale = 1 (u(4) = 0001)
        //   - initial_cpb_removal_delay_length_minus1 = 23 (u(5) = 10111)
        //   - au_cpb_removal_delay_length_minus1 = 23 (10111)
        //   - dpb_output_delay_length_minus1 = 23 (10111)
        // - one sub-layer:
        //   - fixed_pic_rate_general_flag = 0
        //   - fixed_pic_rate_within_cvs_flag = 1 → elemental_duration_in_tc_minus1 ue(v) = 0
        //   - cpb_cnt_minus1 = 1 (ue(v) for 1 → "010")
        //   - NAL HRD body with 2 CPBs (CpbCnt = 2):
        //     - cpb[0]: bit_rate_value_minus1 = 0 (ue "1"), cpb_size_value_minus1 = 100 (ue 100), cpb_size_du = 200, bit_rate_du = 50, cbr = 1
        //     - cpb[1]: bit_rate_value_minus1 = 1 (ue 1 → "010"), cpb_size_value_minus1 = 50, cpb_size_du = 100, bit_rate_du = 100, cbr = 0
        let nal_cpb0_brv = ue(0);
        let nal_cpb0_cpb = ue(100);
        let nal_cpb0_cpb_du = ue(200);
        let nal_cpb0_brv_du = ue(50);
        let nal_cpb1_brv = ue(1);
        let nal_cpb1_cpb = ue(50);
        let nal_cpb1_cpb_du = ue(100);
        let nal_cpb1_brv_du = ue(100);
        let bytes = bits_to_bytes(&[
            "1",        // nal_hrd_parameters_present_flag = 1
            "0",        // vcl_hrd_parameters_present_flag = 0
            "1",        // sub_pic_hrd_params_present_flag = 1
            &ub(3, 8),  // tick_divisor_minus2 = 3
            &ub(4, 5),  // du_cpb_removal_delay_increment_length_minus1 = 4
            "1",        // sub_pic_cpb_params_in_pic_timing_sei_flag = 1
            &ub(5, 5),  // dpb_output_delay_du_length_minus1 = 5
            &ub(2, 4),  // bit_rate_scale = 2
            &ub(3, 4),  // cpb_size_scale = 3
            &ub(1, 4),  // cpb_size_du_scale = 1
            &ub(23, 5), // initial_cpb_removal_delay_length_minus1 = 23
            &ub(23, 5), // au_cpb_removal_delay_length_minus1 = 23
            &ub(23, 5), // dpb_output_delay_length_minus1 = 23
            "0",        // fixed_pic_rate_general_flag = 0
            "1",        // fixed_pic_rate_within_cvs_flag = 1
            &ue(0),     // elemental_duration_in_tc_minus1 = 0
            // low_delay_hrd_flag not signalled (within_cvs == 1)
            &ue(1), // cpb_cnt_minus1 = 1
            &nal_cpb0_brv,
            &nal_cpb0_cpb,
            &nal_cpb0_cpb_du,
            &nal_cpb0_brv_du,
            "1", // cbr_flag[0]
            &nal_cpb1_brv,
            &nal_cpb1_cpb,
            &nal_cpb1_cpb_du,
            &nal_cpb1_brv_du,
            "0", // cbr_flag[1]
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 0, None).expect("parse");
        let common = hrd.common.expect("common present");
        assert!(common.nal_hrd_parameters_present_flag);
        assert!(!common.vcl_hrd_parameters_present_flag);
        assert!(common.sub_pic_hrd_params_present_flag);
        assert_eq!(common.tick_divisor_minus2, 3);
        assert_eq!(common.du_cpb_removal_delay_increment_length_minus1, 4);
        assert!(common.sub_pic_cpb_params_in_pic_timing_sei_flag);
        assert_eq!(common.dpb_output_delay_du_length_minus1, 5);
        assert_eq!(common.bit_rate_scale, 2);
        assert_eq!(common.cpb_size_scale, 3);
        assert_eq!(common.cpb_size_du_scale, 1);
        assert_eq!(common.initial_cpb_removal_delay_length_minus1, 23);
        assert_eq!(common.au_cpb_removal_delay_length_minus1, 23);
        assert_eq!(common.dpb_output_delay_length_minus1, 23);
        assert_eq!(hrd.sub_layers.len(), 1);
        let sl = &hrd.sub_layers[0];
        assert!(!sl.fixed_pic_rate_general_flag);
        assert!(sl.fixed_pic_rate_within_cvs_flag);
        assert_eq!(sl.elemental_duration_in_tc_minus1, Some(0));
        assert!(!sl.low_delay_hrd_flag);
        assert_eq!(sl.cpb_cnt_minus1, 1);
        assert!(sl.vcl_hrd.is_none());
        let nal = sl.nal_hrd.as_ref().expect("nal hrd present");
        assert_eq!(nal.cpb.len(), 2);
        assert_eq!(nal.cpb[0].bit_rate_value_minus1, 0);
        assert_eq!(nal.cpb[0].cpb_size_value_minus1, 100);
        assert_eq!(nal.cpb[0].cpb_size_du_value_minus1, Some(200));
        assert_eq!(nal.cpb[0].bit_rate_du_value_minus1, Some(50));
        assert!(nal.cpb[0].cbr_flag);
        assert_eq!(nal.cpb[1].bit_rate_value_minus1, 1);
        assert_eq!(nal.cpb[1].cpb_size_value_minus1, 50);
        assert_eq!(nal.cpb[1].cpb_size_du_value_minus1, Some(100));
        assert_eq!(nal.cpb[1].bit_rate_du_value_minus1, Some(100));
        assert!(!nal.cpb[1].cbr_flag);
    }

    /// Reject bit_rate_value_minus1[1] <= bit_rate_value_minus1[0] per
    /// §E.3.3.
    #[test]
    fn rejects_non_increasing_bit_rate_value() {
        // common: nal=1, vcl=0, sub_pic=0; one sub-layer with cpb_cnt=2.
        // Both bit_rate_value_minus1 = 5 → second entry violates strict-increase.
        let bytes = bits_to_bytes(&[
            "1",        // nal_hrd_parameters_present_flag
            "0",        // vcl_hrd_parameters_present_flag
            "0",        // sub_pic_hrd_params_present_flag
            &ub(0, 4),  // bit_rate_scale
            &ub(0, 4),  // cpb_size_scale
            &ub(23, 5), // initial_cpb_removal_delay_length_minus1
            &ub(23, 5), // au_cpb_removal_delay_length_minus1
            &ub(23, 5), // dpb_output_delay_length_minus1
            "1",        // fixed_pic_rate_general_flag = 1
            // within_cvs inferred = 1
            &ue(0),  // elemental_duration_in_tc_minus1 = 0
            &ue(1),  // cpb_cnt_minus1 = 1 (CpbCnt = 2)
            &ue(5),  // bit_rate_value_minus1[0] = 5
            &ue(10), // cpb_size_value_minus1[0]
            "1",     // cbr_flag[0]
            &ue(5),  // bit_rate_value_minus1[1] = 5 (NOT > previous) → reject
        ]);
        let mut br = BitReader::new(&bytes);
        let err = HrdParameters::parse(&mut br, true, 0, None).unwrap_err();
        assert_eq!(
            err,
            HrdError::ValueOutOfRange {
                field: "bit_rate_value_minus1",
                got: 5
            }
        );
    }

    /// Reject elemental_duration_in_tc_minus1 > 2047 per §E.3.2.
    #[test]
    fn rejects_elemental_duration_above_2047() {
        let bytes = bits_to_bytes(&[
            "0",       // nal=0
            "0",       // vcl=0
            "0",       // fixed_pic_rate_general_flag = 0
            "1",       // fixed_pic_rate_within_cvs_flag = 1
            &ue(2048), // elemental_duration_in_tc_minus1 = 2048 → out of range
        ]);
        let mut br = BitReader::new(&bytes);
        let err = HrdParameters::parse(&mut br, true, 0, None).unwrap_err();
        assert_eq!(
            err,
            HrdError::ValueOutOfRange {
                field: "elemental_duration_in_tc_minus1",
                got: 2048
            }
        );
    }

    /// Reject cpb_cnt_minus1 > 31 per §E.3.2.
    #[test]
    fn rejects_cpb_cnt_above_31() {
        let bytes = bits_to_bytes(&[
            "0", // nal
            "0", // vcl
            "1", // fixed_pic_rate_general_flag = 1
            // within_cvs inferred = 1
            &ue(0),  // elemental_duration_in_tc_minus1 = 0
            &ue(32), // cpb_cnt_minus1 = 32 → out of range
        ]);
        let mut br = BitReader::new(&bytes);
        let err = HrdParameters::parse(&mut br, true, 0, None).unwrap_err();
        assert_eq!(
            err,
            HrdError::ValueOutOfRange {
                field: "cpb_cnt_minus1",
                got: 32
            }
        );
    }

    /// `low_delay_hrd_flag = 1` skips `cpb_cnt_minus1` and infers it to
    /// 0, but still drives the NAL sub-layer HRD body with CpbCnt = 1.
    #[test]
    fn low_delay_infers_cpb_cnt_zero() {
        let bytes = bits_to_bytes(&[
            "1",        // nal_hrd_parameters_present_flag = 1
            "0",        // vcl
            "0",        // sub_pic
            &ub(0, 4),  // bit_rate_scale
            &ub(0, 4),  // cpb_size_scale
            &ub(23, 5), // initial
            &ub(23, 5), // au
            &ub(23, 5), // dpb_output
            "0",        // fixed_pic_rate_general_flag = 0
            "0",        // fixed_pic_rate_within_cvs_flag = 0
            "1",        // low_delay_hrd_flag = 1 → cpb_cnt_minus1 inferred 0
            &ue(7),     // sub_layer NAL: bit_rate_value_minus1[0] = 7
            &ue(15),    // cpb_size_value_minus1[0] = 15
            "0",        // cbr_flag[0]
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 0, None).expect("parse");
        let sl = &hrd.sub_layers[0];
        assert!(!sl.fixed_pic_rate_general_flag);
        assert!(!sl.fixed_pic_rate_within_cvs_flag);
        assert!(sl.low_delay_hrd_flag);
        assert_eq!(sl.cpb_cnt_minus1, 0);
        let nal = sl.nal_hrd.as_ref().expect("nal");
        assert_eq!(nal.cpb.len(), 1);
        assert_eq!(nal.cpb[0].bit_rate_value_minus1, 7);
        assert_eq!(nal.cpb[0].cpb_size_value_minus1, 15);
        assert!(!nal.cpb[0].cbr_flag);
    }

    /// `commonInfPresentFlag = 0` inherits the previous entry's common
    /// info gates — when the previous entry's NAL flag was 1, the
    /// per-sub-layer NAL HRD body is still expected.
    #[test]
    fn cprms_zero_inherits_previous_common_info() {
        let prev = HrdCommonInfo {
            nal_hrd_parameters_present_flag: true,
            vcl_hrd_parameters_present_flag: false,
            sub_pic_hrd_params_present_flag: false,
            tick_divisor_minus2: 0,
            du_cpb_removal_delay_increment_length_minus1: 0,
            sub_pic_cpb_params_in_pic_timing_sei_flag: false,
            dpb_output_delay_du_length_minus1: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            cpb_size_du_scale: 0,
            initial_cpb_removal_delay_length_minus1: 23,
            au_cpb_removal_delay_length_minus1: 23,
            dpb_output_delay_length_minus1: 23,
        };
        // Parse the per-sub-layer block only; commonInfPresentFlag = 0.
        let bytes = bits_to_bytes(&[
            "1",     // fixed_pic_rate_general_flag = 1, within_cvs inferred = 1
            &ue(0),  // elemental_duration_in_tc_minus1 = 0
            &ue(0),  // cpb_cnt_minus1 = 0 (CpbCnt = 1)
            &ue(11), // NAL sub_layer_hrd: bit_rate_value_minus1[0] = 11
            &ue(22), // cpb_size_value_minus1[0] = 22
            "1",     // cbr_flag[0]
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, false, 0, Some(&prev)).expect("parse");
        assert!(hrd.common.is_none());
        let sl = &hrd.sub_layers[0];
        let nal = sl.nal_hrd.as_ref().expect("nal inherited");
        assert_eq!(nal.cpb[0].bit_rate_value_minus1, 11);
        assert_eq!(nal.cpb[0].cpb_size_value_minus1, 22);
        assert!(nal.cpb[0].cbr_flag);
    }

    /// VPS-wrapper entry parses hrd_layer_set_idx + cprms_present_flag
    /// (when i > 0) ahead of the body.
    #[test]
    fn vps_hrd_entry_skips_cprms_for_index_zero() {
        let bytes = bits_to_bytes(&[
            "1", // hrd_layer_set_idx = 0 ue(v)
            // cprms_present_flag not signalled (i == 0); inferred 1
            "0", // nal = 0
            "0", // vcl = 0
            "1", // fixed_pic_rate_general = 1 → within_cvs inferred 1
            "1", // elemental_duration_in_tc_minus1 = 0
            "1", // cpb_cnt_minus1 = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let entry = VpsHrdEntry::parse(&mut br, 0, 0, None).expect("parse");
        assert_eq!(entry.hrd_layer_set_idx, 0);
        assert!(entry.cprms_present_flag);
        assert!(entry.hrd.common.is_some());
    }

    /// For `i > 0`, the `cprms_present_flag[i]` `u(1)` is read; setting
    /// it to 0 inherits the previous entry's common info.
    #[test]
    fn vps_hrd_entry_reads_cprms_for_nonzero_index() {
        let prev = VpsHrdEntry {
            hrd_layer_set_idx: 0,
            cprms_present_flag: true,
            hrd: HrdParameters {
                max_num_sub_layers_minus1: 0,
                common: Some(HrdCommonInfo {
                    nal_hrd_parameters_present_flag: false,
                    vcl_hrd_parameters_present_flag: false,
                    sub_pic_hrd_params_present_flag: false,
                    tick_divisor_minus2: 0,
                    du_cpb_removal_delay_increment_length_minus1: 0,
                    sub_pic_cpb_params_in_pic_timing_sei_flag: false,
                    dpb_output_delay_du_length_minus1: 0,
                    bit_rate_scale: 0,
                    cpb_size_scale: 0,
                    cpb_size_du_scale: 0,
                    initial_cpb_removal_delay_length_minus1: 0,
                    au_cpb_removal_delay_length_minus1: 0,
                    dpb_output_delay_length_minus1: 0,
                }),
                sub_layers: Vec::new(),
            },
        };
        let bytes = bits_to_bytes(&[
            &ue(1), // hrd_layer_set_idx = 1
            "0",    // cprms_present_flag[1] = 0
            "1",    // fixed_pic_rate_general = 1
            &ue(0), // elemental_duration_in_tc_minus1 = 0
            &ue(0), // cpb_cnt_minus1 = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let entry = VpsHrdEntry::parse(&mut br, 1, 0, Some(&prev)).expect("parse");
        assert_eq!(entry.hrd_layer_set_idx, 1);
        assert!(!entry.cprms_present_flag);
        assert!(entry.hrd.common.is_none());
        // Inherited gates were both 0 → no NAL/VCL sub-layer bodies.
        assert!(entry.hrd.sub_layers[0].nal_hrd.is_none());
        assert!(entry.hrd.sub_layers[0].vcl_hrd.is_none());
    }

    /// `commonInfPresentFlag = 0` with no `prev_common` results in no
    /// NAL/VCL bodies being parsed regardless of subsequent state — the
    /// spec's silent-default behaviour when no prior context exists.
    #[test]
    fn cprms_zero_without_previous_yields_no_hrd_bodies() {
        let bytes = bits_to_bytes(&[
            "1", // fixed_pic_rate_general_flag = 1
            "1", // elemental_duration_in_tc_minus1 = 0
            "1", // cpb_cnt_minus1 = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, false, 0, None).expect("parse");
        assert!(hrd.common.is_none());
        let sl = &hrd.sub_layers[0];
        assert!(sl.nal_hrd.is_none());
        assert!(sl.vcl_hrd.is_none());
    }

    /// Multi-sub-layer payload (`maxNumSubLayersMinus1 = 2`) iterates
    /// the per-sub-layer block three times.
    #[test]
    fn parses_three_sub_layers() {
        let bytes = bits_to_bytes(&[
            "0",
            "0", // no HRD bodies
            // sub-layer 0
            "1",    // fixed_pic_rate_general
            &ue(0), // elemental_duration_in_tc_minus1 = 0
            &ue(0), // cpb_cnt_minus1 = 0 (low_delay inferred 0)
            // sub-layer 1
            "0", // general = 0
            "0", // within_cvs = 0
            "1", // low_delay = 1
            // cpb_cnt_minus1 skipped (low_delay = 1)
            // sub-layer 2
            "0",    // general = 0
            "1",    // within_cvs = 1
            &ue(7), // elemental_duration_in_tc_minus1 = 7
            &ue(0), // cpb_cnt_minus1 = 0
        ]);
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 2, None).expect("parse");
        assert_eq!(hrd.sub_layers.len(), 3);
        assert!(hrd.sub_layers[0].fixed_pic_rate_general_flag);
        assert!(hrd.sub_layers[0].fixed_pic_rate_within_cvs_flag);
        assert_eq!(hrd.sub_layers[0].elemental_duration_in_tc_minus1, Some(0));
        assert!(!hrd.sub_layers[1].fixed_pic_rate_general_flag);
        assert!(!hrd.sub_layers[1].fixed_pic_rate_within_cvs_flag);
        assert!(hrd.sub_layers[1].low_delay_hrd_flag);
        assert_eq!(hrd.sub_layers[1].cpb_cnt_minus1, 0);
        assert!(!hrd.sub_layers[2].fixed_pic_rate_general_flag);
        assert!(hrd.sub_layers[2].fixed_pic_rate_within_cvs_flag);
        assert_eq!(hrd.sub_layers[2].elemental_duration_in_tc_minus1, Some(7));
    }
}
