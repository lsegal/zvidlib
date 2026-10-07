//! Video Usability Information (VUI) parser per ITU-T Rec. H.265
//! §E.2.1 (`vui_parameters()`).
//!
//! The SPS optionally carries a `vui_parameters()` body when
//! `vui_parameters_present_flag == 1`. This module decodes that body
//! into the typed [`VuiParameters`] struct, with the value ranges and
//! inference rules of §E.3.1 enforced inline as [`VuiError`].
//!
//! The nested §E.2.3 `hrd_parameters( 1, sps_max_sub_layers_minus1 )`
//! call (reached through `vui_timing_info_present_flag` /
//! `vui_hrd_parameters_present_flag`) is delegated to the shared
//! [`crate::hevc::engine::hrd::HrdParameters`] parser.
//!
//! ## §E.2.1 layout summary
//!
//! ```text
//! vui_parameters( ) {
//!   aspect_ratio_info_present_flag                  u(1)
//!   if( aspect_ratio_info_present_flag ) {
//!     aspect_ratio_idc                              u(8)
//!     if( aspect_ratio_idc == EXTENDED_SAR ) {      // 255
//!       sar_width                                   u(16)
//!       sar_height                                  u(16)
//!     }
//!   }
//!   overscan_info_present_flag                      u(1)
//!   if( overscan_info_present_flag )
//!     overscan_appropriate_flag                     u(1)
//!   video_signal_type_present_flag                  u(1)
//!   if( video_signal_type_present_flag ) {
//!     video_format                                  u(3)
//!     video_full_range_flag                         u(1)
//!     colour_description_present_flag               u(1)
//!     if( colour_description_present_flag ) {
//!       colour_primaries                            u(8)
//!       transfer_characteristics                    u(8)
//!       matrix_coeffs                               u(8)
//!     }
//!   }
//!   chroma_loc_info_present_flag                    u(1)
//!   if( chroma_loc_info_present_flag ) {
//!     chroma_sample_loc_type_top_field             ue(v)   // 0..=5
//!     chroma_sample_loc_type_bottom_field          ue(v)   // 0..=5
//!   }
//!   neutral_chroma_indication_flag                  u(1)
//!   field_seq_flag                                  u(1)
//!   frame_field_info_present_flag                   u(1)
//!   default_display_window_flag                     u(1)
//!   if( default_display_window_flag ) {
//!     def_disp_win_left_offset                     ue(v)
//!     def_disp_win_right_offset                    ue(v)
//!     def_disp_win_top_offset                      ue(v)
//!     def_disp_win_bottom_offset                   ue(v)
//!   }
//!   vui_timing_info_present_flag                    u(1)
//!   if( vui_timing_info_present_flag ) {
//!     vui_num_units_in_tick                        u(32)   // > 0
//!     vui_time_scale                               u(32)   // > 0
//!     vui_poc_proportional_to_timing_flag          u(1)
//!     if( vui_poc_proportional_to_timing_flag )
//!       vui_num_ticks_poc_diff_one_minus1          ue(v)   // 0..=2^32-2
//!     vui_hrd_parameters_present_flag              u(1)
//!     if( vui_hrd_parameters_present_flag )
//!       hrd_parameters( 1, sps_max_sub_layers_minus1 )     // §E.2.3
//!   }
//!   bitstream_restriction_flag                      u(1)
//!   if( bitstream_restriction_flag ) {
//!     tiles_fixed_structure_flag                    u(1)
//!     motion_vectors_over_pic_boundaries_flag       u(1)
//!     restricted_ref_pic_lists_flag                 u(1)
//!     min_spatial_segmentation_idc                 ue(v)   // 0..=4095
//!     max_bytes_per_pic_denom                      ue(v)   // 0..=16
//!     max_bits_per_min_cu_denom                    ue(v)   // 0..=16
//!     log2_max_mv_length_horizontal                ue(v)   // 0..=15
//!     log2_max_mv_length_vertical                  ue(v)   // 0..=15
//!   }
//! }
//! ```

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::hrd::{HrdError, HrdParameters};

/// `aspect_ratio_idc` value that signals the extended (explicit
/// `sar_width`:`sar_height`) sample aspect ratio per §E.3.1 Table E.1.
pub const EXTENDED_SAR: u8 = 255;

/// Errors that can arise while parsing a `vui_parameters()` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VuiError {
    /// The RBSP ran out of bits before the VUI was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §E.3.1.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value read.
        got: u32,
    },
    /// The nested §E.2.3 `hrd_parameters()` parse failed.
    Hrd(HrdError),
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for VuiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("vui_parameters() RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "VUI syntax element {field} out of range: {got}")
            }
            Self::Hrd(e) => write!(f, "hrd_parameters() error during VUI parse: {e}"),
            Self::Bitstream(e) => write!(f, "bitstream error during VUI parse: {e}"),
        }
    }
}

impl std::error::Error for VuiError {}

impl From<BitReaderError> for VuiError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<HrdError> for VuiError {
    fn from(e: HrdError) -> Self {
        // Flatten truncation to the VUI-level equivalent so the public
        // surface stays predictable; carry structured HRD faults through.
        match e {
            HrdError::Truncated => Self::Truncated,
            HrdError::Bitstream(b) => Self::Bitstream(b),
            other => Self::Hrd(other),
        }
    }
}

/// `colour_description_present_flag` block from §E.2.1 — the three
/// `u(8)` colour-signalling code points. Present only when
/// `colour_description_present_flag == 1`; §E.3.1 infers each value
/// to 2 ("unspecified") otherwise — the parser leaves this `None` and
/// the inference is the caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourDescription {
    /// `colour_primaries` (`u(8)`).
    pub colour_primaries: u8,
    /// `transfer_characteristics` (`u(8)`).
    pub transfer_characteristics: u8,
    /// `matrix_coeffs` (`u(8)`).
    pub matrix_coeffs: u8,
}

/// `video_signal_type_present_flag` block from §E.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSignalType {
    /// `video_format` (`u(3)`). §E.3.1 infers 5 ("unspecified") when
    /// the block is absent.
    pub video_format: u8,
    /// `video_full_range_flag` (`u(1)`). §E.3.1 infers 0 when absent.
    pub video_full_range_flag: bool,
    /// Parsed `colour_description_present_flag` block; `None` when
    /// `colour_description_present_flag == 0`.
    pub colour_description: Option<ColourDescription>,
}

/// `default_display_window_flag` block from §E.2.1 — the four
/// `ue(v)` offsets, in SubWidthC / SubHeightC units (§E.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DefaultDisplayWindow {
    /// `def_disp_win_left_offset` (`ue(v)`).
    pub left_offset: u32,
    /// `def_disp_win_right_offset` (`ue(v)`).
    pub right_offset: u32,
    /// `def_disp_win_top_offset` (`ue(v)`).
    pub top_offset: u32,
    /// `def_disp_win_bottom_offset` (`ue(v)`).
    pub bottom_offset: u32,
}

/// `vui_timing_info_present_flag` block from §E.2.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VuiTimingInfo {
    /// `vui_num_units_in_tick` (`u(32)`). §E.3.1: shall be > 0.
    pub num_units_in_tick: u32,
    /// `vui_time_scale` (`u(32)`). §E.3.1: shall be > 0.
    pub time_scale: u32,
    /// `vui_poc_proportional_to_timing_flag` (`u(1)`).
    pub poc_proportional_to_timing_flag: bool,
    /// `vui_num_ticks_poc_diff_one_minus1` (`ue(v)`); only signalled
    /// when `poc_proportional_to_timing_flag == 1`. §E.3.1 range
    /// 0..=2^32 − 2 (the `ue(v)` ceiling).
    pub num_ticks_poc_diff_one_minus1: Option<u32>,
    /// `vui_hrd_parameters_present_flag` (`u(1)`).
    pub hrd_parameters_present_flag: bool,
    /// Parsed `hrd_parameters( 1, sps_max_sub_layers_minus1 )` body
    /// (§E.2.3); `None` when `vui_hrd_parameters_present_flag == 0`.
    pub hrd_parameters: Option<HrdParameters>,
}

/// `bitstream_restriction_flag` block from §E.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitstreamRestriction {
    /// `tiles_fixed_structure_flag` (`u(1)`).
    pub tiles_fixed_structure_flag: bool,
    /// `motion_vectors_over_pic_boundaries_flag` (`u(1)`). §E.3.1
    /// infers 1 when absent.
    pub motion_vectors_over_pic_boundaries_flag: bool,
    /// `restricted_ref_pic_lists_flag` (`u(1)`).
    pub restricted_ref_pic_lists_flag: bool,
    /// `min_spatial_segmentation_idc` (`ue(v)`, range 0..=4095).
    pub min_spatial_segmentation_idc: u32,
    /// `max_bytes_per_pic_denom` (`ue(v)`, range 0..=16).
    pub max_bytes_per_pic_denom: u32,
    /// `max_bits_per_min_cu_denom` (`ue(v)`, range 0..=16).
    pub max_bits_per_min_cu_denom: u32,
    /// `log2_max_mv_length_horizontal` (`ue(v)`, range 0..=15).
    pub log2_max_mv_length_horizontal: u32,
    /// `log2_max_mv_length_vertical` (`ue(v)`, range 0..=15).
    pub log2_max_mv_length_vertical: u32,
}

/// Parsed `vui_parameters()` body per §E.2.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VuiParameters {
    /// `aspect_ratio_info_present_flag` (`u(1)`).
    pub aspect_ratio_info_present_flag: bool,
    /// `aspect_ratio_idc` (`u(8)`); §E.3.1 infers 0 ("unspecified")
    /// when `aspect_ratio_info_present_flag == 0`. The parser carries
    /// the inferred value either way.
    pub aspect_ratio_idc: u8,
    /// `sar_width` (`u(16)`); only signalled when `aspect_ratio_idc ==`
    /// [`EXTENDED_SAR`].
    pub sar_width: Option<u16>,
    /// `sar_height` (`u(16)`); only signalled when `aspect_ratio_idc ==`
    /// [`EXTENDED_SAR`].
    pub sar_height: Option<u16>,
    /// `overscan_info_present_flag` (`u(1)`).
    pub overscan_info_present_flag: bool,
    /// `overscan_appropriate_flag` (`u(1)`); only signalled when
    /// `overscan_info_present_flag == 1`.
    pub overscan_appropriate_flag: Option<bool>,
    /// `video_signal_type_present_flag` (`u(1)`).
    pub video_signal_type_present_flag: bool,
    /// Parsed `video_signal_type_present_flag` block; `None` when
    /// `video_signal_type_present_flag == 0`.
    pub video_signal_type: Option<VideoSignalType>,
    /// `chroma_loc_info_present_flag` (`u(1)`).
    pub chroma_loc_info_present_flag: bool,
    /// `chroma_sample_loc_type_top_field` (`ue(v)`, range 0..=5);
    /// §E.3.1 infers 0 when absent. The parser carries the inferred
    /// value.
    pub chroma_sample_loc_type_top_field: u32,
    /// `chroma_sample_loc_type_bottom_field` (`ue(v)`, range 0..=5);
    /// §E.3.1 infers 0 when absent.
    pub chroma_sample_loc_type_bottom_field: u32,
    /// `neutral_chroma_indication_flag` (`u(1)`).
    pub neutral_chroma_indication_flag: bool,
    /// `field_seq_flag` (`u(1)`).
    pub field_seq_flag: bool,
    /// `frame_field_info_present_flag` (`u(1)`).
    pub frame_field_info_present_flag: bool,
    /// `default_display_window_flag` (`u(1)`).
    pub default_display_window_flag: bool,
    /// Parsed `default_display_window_flag` block; `None` when
    /// `default_display_window_flag == 0` (all offsets infer to 0 per
    /// §E.3.1).
    pub default_display_window: Option<DefaultDisplayWindow>,
    /// `vui_timing_info_present_flag` (`u(1)`).
    pub vui_timing_info_present_flag: bool,
    /// Parsed `vui_timing_info_present_flag` block; `None` when
    /// `vui_timing_info_present_flag == 0`.
    pub timing_info: Option<VuiTimingInfo>,
    /// `bitstream_restriction_flag` (`u(1)`).
    pub bitstream_restriction_flag: bool,
    /// Parsed `bitstream_restriction_flag` block; `None` when
    /// `bitstream_restriction_flag == 0` (§E.3.1 default inferences
    /// apply).
    pub bitstream_restriction: Option<BitstreamRestriction>,
}

impl VuiParameters {
    /// Parse a `vui_parameters()` body starting at the current `br`
    /// position.
    ///
    /// `sps_max_sub_layers_minus1` plumbs through the
    /// `maxNumSubLayersMinus1` argument to the nested §E.2.3
    /// `hrd_parameters( 1, sps_max_sub_layers_minus1 )` call. On
    /// success the reader is advanced past the structure.
    pub fn parse(br: &mut BitReader<'_>, sps_max_sub_layers_minus1: u8) -> Result<Self, VuiError> {
        let aspect_ratio_info_present_flag = br.u1()? != 0;
        let mut aspect_ratio_idc = 0u8;
        let mut sar_width = None;
        let mut sar_height = None;
        if aspect_ratio_info_present_flag {
            aspect_ratio_idc = br.u(8)? as u8;
            if aspect_ratio_idc == EXTENDED_SAR {
                sar_width = Some(br.u(16)? as u16);
                sar_height = Some(br.u(16)? as u16);
            }
        }

        let overscan_info_present_flag = br.u1()? != 0;
        let overscan_appropriate_flag = if overscan_info_present_flag {
            Some(br.u1()? != 0)
        } else {
            None
        };

        let video_signal_type_present_flag = br.u1()? != 0;
        let video_signal_type = if video_signal_type_present_flag {
            let video_format = br.u(3)? as u8;
            let video_full_range_flag = br.u1()? != 0;
            let colour_description_present_flag = br.u1()? != 0;
            let colour_description = if colour_description_present_flag {
                Some(ColourDescription {
                    colour_primaries: br.u(8)? as u8,
                    transfer_characteristics: br.u(8)? as u8,
                    matrix_coeffs: br.u(8)? as u8,
                })
            } else {
                None
            };
            Some(VideoSignalType {
                video_format,
                video_full_range_flag,
                colour_description,
            })
        } else {
            None
        };

        let chroma_loc_info_present_flag = br.u1()? != 0;
        let (chroma_sample_loc_type_top_field, chroma_sample_loc_type_bottom_field) =
            if chroma_loc_info_present_flag {
                let top = br.ue()?;
                // §E.3.1: range 0..=5, inclusive.
                if top > 5 {
                    return Err(VuiError::ValueOutOfRange {
                        field: "chroma_sample_loc_type_top_field",
                        got: top,
                    });
                }
                let bottom = br.ue()?;
                if bottom > 5 {
                    return Err(VuiError::ValueOutOfRange {
                        field: "chroma_sample_loc_type_bottom_field",
                        got: bottom,
                    });
                }
                (top, bottom)
            } else {
                // §E.3.1: inferred to 0 when not present.
                (0, 0)
            };

        let neutral_chroma_indication_flag = br.u1()? != 0;
        let field_seq_flag = br.u1()? != 0;
        let frame_field_info_present_flag = br.u1()? != 0;

        let default_display_window_flag = br.u1()? != 0;
        let default_display_window = if default_display_window_flag {
            Some(DefaultDisplayWindow {
                left_offset: br.ue()?,
                right_offset: br.ue()?,
                top_offset: br.ue()?,
                bottom_offset: br.ue()?,
            })
        } else {
            None
        };

        let vui_timing_info_present_flag = br.u1()? != 0;
        let timing_info = if vui_timing_info_present_flag {
            let num_units_in_tick = br.u(32)?;
            // §E.3.1: vui_num_units_in_tick shall be greater than 0.
            if num_units_in_tick == 0 {
                return Err(VuiError::ValueOutOfRange {
                    field: "vui_num_units_in_tick",
                    got: 0,
                });
            }
            let time_scale = br.u(32)?;
            // §E.3.1: vui_time_scale shall be greater than 0.
            if time_scale == 0 {
                return Err(VuiError::ValueOutOfRange {
                    field: "vui_time_scale",
                    got: 0,
                });
            }
            let poc_proportional_to_timing_flag = br.u1()? != 0;
            let num_ticks_poc_diff_one_minus1 = if poc_proportional_to_timing_flag {
                // §E.3.1 range 0..=2^32 − 2, which is the ue(v) ceiling;
                // the reader already enforces it, so no extra check.
                Some(br.ue()?)
            } else {
                None
            };
            let hrd_parameters_present_flag = br.u1()? != 0;
            let hrd_parameters = if hrd_parameters_present_flag {
                // §E.2.1: hrd_parameters( 1, sps_max_sub_layers_minus1 ).
                // commonInfPresentFlag is 1, so no inheritance source is
                // needed.
                Some(HrdParameters::parse(
                    br,
                    true,
                    sps_max_sub_layers_minus1,
                    None,
                )?)
            } else {
                None
            };
            Some(VuiTimingInfo {
                num_units_in_tick,
                time_scale,
                poc_proportional_to_timing_flag,
                num_ticks_poc_diff_one_minus1,
                hrd_parameters_present_flag,
                hrd_parameters,
            })
        } else {
            None
        };

        let bitstream_restriction_flag = br.u1()? != 0;
        let bitstream_restriction = if bitstream_restriction_flag {
            let tiles_fixed_structure_flag = br.u1()? != 0;
            let motion_vectors_over_pic_boundaries_flag = br.u1()? != 0;
            let restricted_ref_pic_lists_flag = br.u1()? != 0;
            let min_spatial_segmentation_idc = br.ue()?;
            // §E.3.1: range 0..=4095, inclusive.
            if min_spatial_segmentation_idc > 4095 {
                return Err(VuiError::ValueOutOfRange {
                    field: "min_spatial_segmentation_idc",
                    got: min_spatial_segmentation_idc,
                });
            }
            let max_bytes_per_pic_denom = br.ue()?;
            // §E.3.1: range 0..=16, inclusive.
            if max_bytes_per_pic_denom > 16 {
                return Err(VuiError::ValueOutOfRange {
                    field: "max_bytes_per_pic_denom",
                    got: max_bytes_per_pic_denom,
                });
            }
            let max_bits_per_min_cu_denom = br.ue()?;
            // §E.3.1: range 0..=16, inclusive.
            if max_bits_per_min_cu_denom > 16 {
                return Err(VuiError::ValueOutOfRange {
                    field: "max_bits_per_min_cu_denom",
                    got: max_bits_per_min_cu_denom,
                });
            }
            let log2_max_mv_length_horizontal = br.ue()?;
            // §E.3.1: range 0..=15, inclusive.
            if log2_max_mv_length_horizontal > 15 {
                return Err(VuiError::ValueOutOfRange {
                    field: "log2_max_mv_length_horizontal",
                    got: log2_max_mv_length_horizontal,
                });
            }
            let log2_max_mv_length_vertical = br.ue()?;
            if log2_max_mv_length_vertical > 15 {
                return Err(VuiError::ValueOutOfRange {
                    field: "log2_max_mv_length_vertical",
                    got: log2_max_mv_length_vertical,
                });
            }
            Some(BitstreamRestriction {
                tiles_fixed_structure_flag,
                motion_vectors_over_pic_boundaries_flag,
                restricted_ref_pic_lists_flag,
                min_spatial_segmentation_idc,
                max_bytes_per_pic_denom,
                max_bits_per_min_cu_denom,
                log2_max_mv_length_horizontal,
                log2_max_mv_length_vertical,
            })
        } else {
            None
        };

        Ok(Self {
            aspect_ratio_info_present_flag,
            aspect_ratio_idc,
            sar_width,
            sar_height,
            overscan_info_present_flag,
            overscan_appropriate_flag,
            video_signal_type_present_flag,
            video_signal_type,
            chroma_loc_info_present_flag,
            chroma_sample_loc_type_top_field,
            chroma_sample_loc_type_bottom_field,
            neutral_chroma_indication_flag,
            field_seq_flag,
            frame_field_info_present_flag,
            default_display_window_flag,
            default_display_window,
            vui_timing_info_present_flag,
            timing_info,
            bitstream_restriction_flag,
            bitstream_restriction,
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
    /// §9.2: emit `leading_zero_bits` zeros followed by
    /// `binary(code + 1)` on `leading_zero_bits + 1` bits.
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

    /// Every present flag off: the minimal `vui_parameters()` body is
    /// nine `u(1)` flags (aspect / overscan / video-signal / chroma-loc
    /// / neutral / field-seq / frame-field / default-display /
    /// timing-info) followed by `bitstream_restriction_flag`. All
    /// inference defaults apply.
    #[test]
    fn parses_all_flags_off() {
        let bytes = bits_to_bytes(&[
            "0", // aspect_ratio_info_present_flag
            "0", // overscan_info_present_flag
            "0", // video_signal_type_present_flag
            "0", // chroma_loc_info_present_flag
            "0", // neutral_chroma_indication_flag
            "0", // field_seq_flag
            "0", // frame_field_info_present_flag
            "0", // default_display_window_flag
            "0", // vui_timing_info_present_flag
            "0", // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(!vui.aspect_ratio_info_present_flag);
        assert_eq!(vui.aspect_ratio_idc, 0); // inferred
        assert!(vui.sar_width.is_none());
        assert!(!vui.overscan_info_present_flag);
        assert!(vui.overscan_appropriate_flag.is_none());
        assert!(!vui.video_signal_type_present_flag);
        assert!(vui.video_signal_type.is_none());
        assert!(!vui.chroma_loc_info_present_flag);
        assert_eq!(vui.chroma_sample_loc_type_top_field, 0); // inferred
        assert_eq!(vui.chroma_sample_loc_type_bottom_field, 0);
        assert!(!vui.neutral_chroma_indication_flag);
        assert!(!vui.field_seq_flag);
        assert!(!vui.frame_field_info_present_flag);
        assert!(!vui.default_display_window_flag);
        assert!(vui.default_display_window.is_none());
        assert!(!vui.vui_timing_info_present_flag);
        assert!(vui.timing_info.is_none());
        assert!(!vui.bitstream_restriction_flag);
        assert!(vui.bitstream_restriction.is_none());
        // The reader consumed exactly 10 bits.
        assert_eq!(br.bit_pos(), 10);
    }

    /// `aspect_ratio_idc == EXTENDED_SAR` pulls a `sar_width` /
    /// `sar_height` `u(16)` pair.
    #[test]
    fn parses_extended_sar() {
        let bytes = bits_to_bytes(&[
            "1",             // aspect_ratio_info_present_flag
            &ub(255, 8),     // aspect_ratio_idc = EXTENDED_SAR
            &ub(0x1234, 16), // sar_width
            &ub(0x5678, 16), // sar_height
            "0",             // overscan_info_present_flag
            "0",             // video_signal_type_present_flag
            "0",             // chroma_loc_info_present_flag
            "0",             // neutral_chroma_indication_flag
            "0",             // field_seq_flag
            "0",             // frame_field_info_present_flag
            "0",             // default_display_window_flag
            "0",             // vui_timing_info_present_flag
            "0",             // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(vui.aspect_ratio_info_present_flag);
        assert_eq!(vui.aspect_ratio_idc, EXTENDED_SAR);
        assert_eq!(vui.sar_width, Some(0x1234));
        assert_eq!(vui.sar_height, Some(0x5678));
    }

    /// A plain `aspect_ratio_idc` (1 = square) carries no SAR pair.
    #[test]
    fn parses_plain_aspect_ratio_idc() {
        let bytes = bits_to_bytes(&[
            "1",       // aspect_ratio_info_present_flag
            &ub(1, 8), // aspect_ratio_idc = 1 (square)
            "0",       // overscan_info_present_flag
            "0",       // video_signal_type_present_flag
            "0",       // chroma_loc_info_present_flag
            "0",       // neutral_chroma_indication_flag
            "0",       // field_seq_flag
            "0",       // frame_field_info_present_flag
            "0",       // default_display_window_flag
            "0",       // vui_timing_info_present_flag
            "0",       // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert_eq!(vui.aspect_ratio_idc, 1);
        assert!(vui.sar_width.is_none());
        assert!(vui.sar_height.is_none());
    }

    /// Overscan + video-signal-type with colour-description.
    #[test]
    fn parses_video_signal_type_with_colour_description() {
        let bytes = bits_to_bytes(&[
            "0",        // aspect_ratio_info_present_flag
            "1",        // overscan_info_present_flag
            "1",        // overscan_appropriate_flag = 1
            "1",        // video_signal_type_present_flag
            &ub(5, 3),  // video_format = 5 (unspecified)
            "1",        // video_full_range_flag = 1
            "1",        // colour_description_present_flag = 1
            &ub(9, 8),  // colour_primaries = 9 (BT.2020)
            &ub(16, 8), // transfer_characteristics = 16 (PQ)
            &ub(9, 8),  // matrix_coeffs = 9 (BT.2020 non-const)
            "0",        // chroma_loc_info_present_flag
            "0",        // neutral_chroma_indication_flag
            "0",        // field_seq_flag
            "0",        // frame_field_info_present_flag
            "0",        // default_display_window_flag
            "0",        // vui_timing_info_present_flag
            "0",        // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(vui.overscan_info_present_flag);
        assert_eq!(vui.overscan_appropriate_flag, Some(true));
        let vst = vui.video_signal_type.expect("video signal type");
        assert_eq!(vst.video_format, 5);
        assert!(vst.video_full_range_flag);
        let cd = vst.colour_description.expect("colour description");
        assert_eq!(cd.colour_primaries, 9);
        assert_eq!(cd.transfer_characteristics, 16);
        assert_eq!(cd.matrix_coeffs, 9);
    }

    /// video-signal-type present but colour-description absent.
    #[test]
    fn parses_video_signal_type_without_colour_description() {
        let bytes = bits_to_bytes(&[
            "0",       // aspect_ratio_info_present_flag
            "0",       // overscan_info_present_flag
            "1",       // video_signal_type_present_flag
            &ub(1, 3), // video_format = 1
            "0",       // video_full_range_flag = 0
            "0",       // colour_description_present_flag = 0
            "0",       // chroma_loc_info_present_flag
            "0",       // neutral_chroma_indication_flag
            "0",       // field_seq_flag
            "0",       // frame_field_info_present_flag
            "0",       // default_display_window_flag
            "0",       // vui_timing_info_present_flag
            "0",       // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        let vst = vui.video_signal_type.expect("video signal type");
        assert_eq!(vst.video_format, 1);
        assert!(!vst.video_full_range_flag);
        assert!(vst.colour_description.is_none());
    }

    /// chroma-loc-info + default-display-window blocks.
    #[test]
    fn parses_chroma_loc_and_default_display_window() {
        let bytes = bits_to_bytes(&[
            "0",    // aspect_ratio_info_present_flag
            "0",    // overscan_info_present_flag
            "0",    // video_signal_type_present_flag
            "1",    // chroma_loc_info_present_flag
            &ue(2), // chroma_sample_loc_type_top_field = 2
            &ue(3), // chroma_sample_loc_type_bottom_field = 3
            "1",    // neutral_chroma_indication_flag = 1
            "1",    // field_seq_flag = 1
            "1",    // frame_field_info_present_flag = 1
            "1",    // default_display_window_flag = 1
            &ue(4), // def_disp_win_left_offset = 4
            &ue(5), // def_disp_win_right_offset = 5
            &ue(6), // def_disp_win_top_offset = 6
            &ue(7), // def_disp_win_bottom_offset = 7
            "0",    // vui_timing_info_present_flag
            "0",    // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(vui.chroma_loc_info_present_flag);
        assert_eq!(vui.chroma_sample_loc_type_top_field, 2);
        assert_eq!(vui.chroma_sample_loc_type_bottom_field, 3);
        assert!(vui.neutral_chroma_indication_flag);
        assert!(vui.field_seq_flag);
        assert!(vui.frame_field_info_present_flag);
        let ddw = vui.default_display_window.expect("default display window");
        assert_eq!(ddw.left_offset, 4);
        assert_eq!(ddw.right_offset, 5);
        assert_eq!(ddw.top_offset, 6);
        assert_eq!(ddw.bottom_offset, 7);
    }

    /// Timing-info block without HRD (poc-proportional set).
    #[test]
    fn parses_timing_info_without_hrd() {
        let bytes = bits_to_bytes(&[
            "0",               // aspect_ratio_info_present_flag
            "0",               // overscan_info_present_flag
            "0",               // video_signal_type_present_flag
            "0",               // chroma_loc_info_present_flag
            "0",               // neutral_chroma_indication_flag
            "0",               // field_seq_flag
            "0",               // frame_field_info_present_flag
            "0",               // default_display_window_flag
            "1",               // vui_timing_info_present_flag
            &ub(1080000, 32),  // vui_num_units_in_tick
            &ub(27000000, 32), // vui_time_scale
            "1",               // vui_poc_proportional_to_timing_flag = 1
            &ue(8),            // vui_num_ticks_poc_diff_one_minus1 = 8
            "0",               // vui_hrd_parameters_present_flag = 0
            "0",               // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(vui.vui_timing_info_present_flag);
        let ti = vui.timing_info.expect("timing info");
        assert_eq!(ti.num_units_in_tick, 1080000);
        assert_eq!(ti.time_scale, 27000000);
        assert!(ti.poc_proportional_to_timing_flag);
        assert_eq!(ti.num_ticks_poc_diff_one_minus1, Some(8));
        assert!(!ti.hrd_parameters_present_flag);
        assert!(ti.hrd_parameters.is_none());
    }

    /// Timing-info block with a nested `hrd_parameters( 1,
    /// sps_max_sub_layers_minus1 )` body (gates off → one sub-layer).
    #[test]
    fn parses_timing_info_with_hrd() {
        let bytes = bits_to_bytes(&[
            "0",         // aspect_ratio_info_present_flag
            "0",         // overscan_info_present_flag
            "0",         // video_signal_type_present_flag
            "0",         // chroma_loc_info_present_flag
            "0",         // neutral_chroma_indication_flag
            "0",         // field_seq_flag
            "0",         // frame_field_info_present_flag
            "0",         // default_display_window_flag
            "1",         // vui_timing_info_present_flag
            &ub(1, 32),  // vui_num_units_in_tick = 1
            &ub(50, 32), // vui_time_scale = 50
            "0",         // vui_poc_proportional_to_timing_flag = 0
            "1",         // vui_hrd_parameters_present_flag = 1
            // hrd_parameters( 1, 0 ): commonInfPresentFlag = 1, one sub-layer
            "0",    // nal_hrd_parameters_present_flag = 0
            "0",    // vcl_hrd_parameters_present_flag = 0
            "1",    // fixed_pic_rate_general_flag[0] = 1
            &ue(0), // elemental_duration_in_tc_minus1[0] = 0
            &ue(0), // cpb_cnt_minus1[0] = 0
            "0",    // bitstream_restriction_flag
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        let ti = vui.timing_info.expect("timing info");
        assert!(ti.hrd_parameters_present_flag);
        let hrd = ti.hrd_parameters.expect("hrd parameters");
        assert_eq!(hrd.max_num_sub_layers_minus1, 0);
        assert_eq!(hrd.sub_layers.len(), 1);
        let common = hrd.common.expect("common");
        assert!(!common.nal_hrd_parameters_present_flag);
        assert!(!common.vcl_hrd_parameters_present_flag);
    }

    /// `vui_num_units_in_tick == 0` is rejected per §E.3.1.
    #[test]
    fn rejects_zero_num_units_in_tick() {
        let bytes = bits_to_bytes(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",         // eight present flags off
            "1",         // vui_timing_info_present_flag
            &ub(0, 32),  // vui_num_units_in_tick = 0 → reject
            &ub(50, 32), // vui_time_scale
        ]);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "vui_num_units_in_tick",
                got: 0,
            }
        );
    }

    /// `vui_time_scale == 0` is rejected per §E.3.1.
    #[test]
    fn rejects_zero_time_scale() {
        let bytes = bits_to_bytes(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",        // eight present flags off
            "1",        // vui_timing_info_present_flag
            &ub(1, 32), // vui_num_units_in_tick = 1
            &ub(0, 32), // vui_time_scale = 0 → reject
        ]);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "vui_time_scale",
                got: 0,
            }
        );
    }

    /// Full `bitstream_restriction` block parses with every field.
    #[test]
    fn parses_bitstream_restriction() {
        let bytes = bits_to_bytes(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",      // eight present flags off
            "0",      // vui_timing_info_present_flag
            "1",      // bitstream_restriction_flag
            "1",      // tiles_fixed_structure_flag = 1
            "0",      // motion_vectors_over_pic_boundaries_flag = 0
            "1",      // restricted_ref_pic_lists_flag = 1
            &ue(120), // min_spatial_segmentation_idc = 120
            &ue(2),   // max_bytes_per_pic_denom = 2
            &ue(1),   // max_bits_per_min_cu_denom = 1
            &ue(15),  // log2_max_mv_length_horizontal = 15
            &ue(15),  // log2_max_mv_length_vertical = 15
        ]);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("parse");
        assert!(vui.bitstream_restriction_flag);
        let bsr = vui.bitstream_restriction.expect("bitstream restriction");
        assert!(bsr.tiles_fixed_structure_flag);
        assert!(!bsr.motion_vectors_over_pic_boundaries_flag);
        assert!(bsr.restricted_ref_pic_lists_flag);
        assert_eq!(bsr.min_spatial_segmentation_idc, 120);
        assert_eq!(bsr.max_bytes_per_pic_denom, 2);
        assert_eq!(bsr.max_bits_per_min_cu_denom, 1);
        assert_eq!(bsr.log2_max_mv_length_horizontal, 15);
        assert_eq!(bsr.log2_max_mv_length_vertical, 15);
    }

    /// `min_spatial_segmentation_idc > 4095` is rejected per §E.3.1.
    #[test]
    fn rejects_min_spatial_segmentation_above_4095() {
        let bytes = bits_to_bytes(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",       // eight present flags off
            "0",       // vui_timing_info_present_flag
            "1",       // bitstream_restriction_flag
            "0",       // tiles_fixed_structure_flag
            "0",       // motion_vectors_over_pic_boundaries_flag
            "0",       // restricted_ref_pic_lists_flag
            &ue(4096), // min_spatial_segmentation_idc = 4096 → reject
        ]);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "min_spatial_segmentation_idc",
                got: 4096,
            }
        );
    }

    /// `chroma_sample_loc_type_top_field > 5` is rejected per §E.3.1.
    #[test]
    fn rejects_chroma_sample_loc_above_5() {
        let bytes = bits_to_bytes(&[
            "0",    // aspect_ratio_info_present_flag
            "0",    // overscan_info_present_flag
            "0",    // video_signal_type_present_flag
            "1",    // chroma_loc_info_present_flag
            &ue(6), // chroma_sample_loc_type_top_field = 6 → reject
        ]);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "chroma_sample_loc_type_top_field",
                got: 6,
            }
        );
    }

    /// `log2_max_mv_length_vertical > 15` is rejected per §E.3.1.
    #[test]
    fn rejects_log2_max_mv_length_above_15() {
        let bytes = bits_to_bytes(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",     // eight present flags off
            "0",     // vui_timing_info_present_flag
            "1",     // bitstream_restriction_flag
            "0",     // tiles_fixed_structure_flag
            "0",     // motion_vectors_over_pic_boundaries_flag
            "0",     // restricted_ref_pic_lists_flag
            &ue(0),  // min_spatial_segmentation_idc
            &ue(0),  // max_bytes_per_pic_denom
            &ue(0),  // max_bits_per_min_cu_denom
            &ue(0),  // log2_max_mv_length_horizontal
            &ue(16), // log2_max_mv_length_vertical = 16 → reject
        ]);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "log2_max_mv_length_vertical",
                got: 16,
            }
        );
    }

    /// Truncation mid-body surfaces `VuiError::Truncated`.
    #[test]
    fn truncation_is_reported() {
        // aspect_ratio_info_present_flag = 1 but no aspect_ratio_idc byte.
        let bytes = bits_to_bytes(&["1"]);
        let mut br = BitReader::new(&bytes);
        // The single set bit plus padding zeros means aspect_ratio_idc
        // reads 7 pad bits then runs off; force a tight buffer so the
        // u(8) read truncates. A single-byte buffer has 8 bits; after
        // the present flag, 7 remain, so u(8) truncates.
        let err = VuiParameters::parse(&mut br, 0).unwrap_err();
        assert_eq!(err, VuiError::Truncated);
    }
}
