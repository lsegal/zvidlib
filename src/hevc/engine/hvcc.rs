//! `HEVCDecoderConfigurationRecord` (`hvcC`) extradata parse.
//!
//! ISO-BMFF tracks carry the HEVC parameter sets out of band, inside
//! the sample entry's `hvcC` box, and frame the in-band NAL units with
//! big-endian length prefixes instead of Annex B start codes. The
//! record layout implemented here is the `aligned(8)` class of
//! ISO/IEC 14496-15 §8.3.3.1.2 (`HEVCDecoderConfigurationRecord`):
//!
//! ```text
//! unsigned int(8)  configurationVersion = 1;
//! unsigned int(2)  general_profile_space;
//! unsigned int(1)  general_tier_flag;
//! unsigned int(5)  general_profile_idc;
//! unsigned int(32) general_profile_compatibility_flags;
//! unsigned int(48) general_constraint_indicator_flags;
//! unsigned int(8)  general_level_idc;
//! bit(4) reserved; unsigned int(12) min_spatial_segmentation_idc;
//! bit(6) reserved; unsigned int(2)  parallelismType;
//! bit(6) reserved; unsigned int(2)  chroma_format_idc;
//! bit(5) reserved; unsigned int(3)  bit_depth_luma_minus8;
//! bit(5) reserved; unsigned int(3)  bit_depth_chroma_minus8;
//! unsigned int(16) avgFrameRate;
//! unsigned int(2)  constantFrameRate;
//! unsigned int(3)  numTemporalLayers;
//! unsigned int(1)  temporalIdNested;
//! unsigned int(2)  lengthSizeMinusOne;
//! unsigned int(8)  numOfArrays;
//! for (j = 0; j < numOfArrays; j++) {
//!   unsigned int(1)  array_completeness;
//!   bit(1) reserved;
//!   unsigned int(6)  NAL_unit_type;
//!   unsigned int(16) numNalus;
//!   for (i = 0; i < numNalus; i++) {
//!     unsigned int(16)      nalUnitLength;
//!     bit(8*nalUnitLength)  nalUnit;
//!   }
//! }
//! ```
//!
//! The NAL units inside the arrays are complete coded NAL units
//! (two-byte §7.3.1.2 header plus the emulation-escaped payload) —
//! exactly what an Annex B start code would delimit. §8.3.3.1.3
//! restricts the array `NAL_unit_type` values to VPS / SPS / PPS and
//! prefix / suffix SEI; readers ignore arrays with any other type.
//!
//! [`split_length_prefixed`] performs the companion sample-data
//! re-framing: each in-band access unit is a run of
//! `lengthSizeMinusOne + 1`-byte big-endian sizes, each followed by
//! that many coded NAL bytes.

use crate::hevc::engine::nal::{NalError, NalHeader, NalUnit, strip_emulation_prevention};

/// Errors from the `hvcC` record parse or the length-prefixed
/// sample-data split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvccError {
    /// The record is shorter than the fixed 23-byte prefix, or an
    /// array / NAL-unit length points past the end of the record.
    Truncated,
    /// `configurationVersion` was not 1 (the only value ISO/IEC
    /// 14496-15 defines; incompatible records must change it).
    BadVersion(u8),
    /// `lengthSizeMinusOne` was 2 — §8.3.3.1.3 allows only 0, 1
    /// and 3 (1-, 2- and 4-byte length prefixes).
    BadLengthSize,
    /// A carried NAL unit failed the §7.3.1.2 header parse.
    Nal(NalError),
    /// A sample-data buffer ended in the middle of a length prefix
    /// or of the NAL bytes the prefix promised.
    TruncatedSample,
}

impl core::fmt::Display for HvccError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("hvcC record truncated"),
            Self::BadVersion(v) => write!(f, "hvcC configurationVersion {v} (expected 1)"),
            Self::BadLengthSize => f.write_str("hvcC lengthSizeMinusOne 2 is reserved"),
            Self::Nal(e) => write!(f, "hvcC-carried NAL unit: {e}"),
            Self::TruncatedSample => f.write_str("length-prefixed sample data truncated"),
        }
    }
}

impl std::error::Error for HvccError {}

impl From<NalError> for HvccError {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}

/// Parsed `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15
/// §8.3.3.1.2). The profile / level mirror fields carry the §7.3.3
/// `profile_tier_level` values of the stream's parameter sets; the
/// decoder proper re-derives everything from the carried VPS / SPS /
/// PPS, so they are surfaced for inspection only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HvccRecord {
    /// `general_profile_space` (2 bits).
    pub general_profile_space: u8,
    /// `general_tier_flag`.
    pub general_tier_flag: bool,
    /// `general_profile_idc` (5 bits).
    pub general_profile_idc: u8,
    /// `general_profile_compatibility_flags` (32 bits; bit 31 is
    /// `general_profile_compatibility_flag[0]`).
    pub general_profile_compatibility_flags: u32,
    /// `general_constraint_indicator_flags` (48 bits, in the low bits).
    pub general_constraint_indicator_flags: u64,
    /// `general_level_idc`.
    pub general_level_idc: u8,
    /// `min_spatial_segmentation_idc` (12 bits).
    pub min_spatial_segmentation_idc: u16,
    /// `parallelismType` (2 bits): 0 unknown / mixed, 1 slices,
    /// 2 tiles, 3 entropy-coding sync.
    pub parallelism_type: u8,
    /// `chroma_format_idc` mirror (2 bits).
    pub chroma_format_idc: u8,
    /// `bit_depth_luma_minus8` mirror (3 bits).
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8` mirror (3 bits).
    pub bit_depth_chroma_minus8: u8,
    /// `avgFrameRate` in frames per 256 seconds (0 = unspecified).
    pub avg_frame_rate: u16,
    /// `constantFrameRate` (2 bits).
    pub constant_frame_rate: u8,
    /// `numTemporalLayers` (3 bits).
    pub num_temporal_layers: u8,
    /// `temporalIdNested`.
    pub temporal_id_nested: bool,
    /// Byte width of every in-band NAL length prefix
    /// (`lengthSizeMinusOne + 1`): 1, 2 or 4.
    pub length_size: usize,
    /// The carried parameter-set / SEI NAL units, in array order,
    /// parsed exactly as an Annex B walker would surface them.
    /// Arrays with a reserved / unpermitted `NAL_unit_type` are
    /// skipped per §8.3.3.1.3.
    pub nal_units: Vec<NalUnit>,
}

/// The array `NAL_unit_type` values §8.3.3.1.3 permits: VPS (32),
/// SPS (33), PPS (34), prefix SEI (39), suffix SEI (40).
fn array_type_permitted(nal_unit_type: u8) -> bool {
    matches!(nal_unit_type, 32..=34 | 39 | 40)
}

/// Reconstruct a [`NalUnit`] from a complete coded NAL unit (two-byte
/// header + emulation-escaped payload) — the framing-free form both
/// the `hvcC` arrays and length-prefixed sample data carry.
pub fn nal_unit_from_coded(coded: &[u8]) -> Result<NalUnit, NalError> {
    let unescaped = strip_emulation_prevention(coded);
    let header = NalHeader::parse(&unescaped)?;
    Ok(NalUnit {
        header,
        rbsp: unescaped[2..].to_vec(),
        escaped: coded[2..].to_vec(),
    })
}

/// Quick extradata-form probe: an `hvcC` record always starts with
/// `configurationVersion == 1` and is at least 23 bytes, while Annex B
/// extradata starts with a `00 00 01` / `00 00 00 01` start code.
pub fn extradata_is_hvcc(data: &[u8]) -> bool {
    data.len() >= 23 && data[0] == 1
}

/// Parse an `HEVCDecoderConfigurationRecord` per ISO/IEC 14496-15
/// §8.3.3.1.2.
///
/// # Errors
/// [`HvccError`] on truncation, a version other than 1, the reserved
/// length size, or an undecodable carried NAL unit.
pub fn parse_hvcc(data: &[u8]) -> Result<HvccRecord, HvccError> {
    if data.len() < 23 {
        return Err(HvccError::Truncated);
    }
    if data[0] != 1 {
        return Err(HvccError::BadVersion(data[0]));
    }
    let length_size_minus_one = (data[21] & 0x03) as usize;
    if length_size_minus_one == 2 {
        return Err(HvccError::BadLengthSize);
    }

    let mut nal_units = Vec::new();
    let num_of_arrays = data[22] as usize;
    let mut pos = 23;
    for _ in 0..num_of_arrays {
        if pos + 3 > data.len() {
            return Err(HvccError::Truncated);
        }
        let nal_unit_type = data[pos] & 0x3F;
        let num_nalus = u16::from_be_bytes([data[pos + 1], data[pos + 2]]) as usize;
        pos += 3;
        for _ in 0..num_nalus {
            if pos + 2 > data.len() {
                return Err(HvccError::Truncated);
            }
            let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if pos + len > data.len() {
                return Err(HvccError::Truncated);
            }
            if array_type_permitted(nal_unit_type) {
                nal_units.push(nal_unit_from_coded(&data[pos..pos + len])?);
            }
            pos += len;
        }
    }

    Ok(HvccRecord {
        general_profile_space: data[1] >> 6,
        general_tier_flag: data[1] & 0x20 != 0,
        general_profile_idc: data[1] & 0x1F,
        general_profile_compatibility_flags: u32::from_be_bytes([
            data[2], data[3], data[4], data[5],
        ]),
        general_constraint_indicator_flags: u64::from_be_bytes([
            0, 0, data[6], data[7], data[8], data[9], data[10], data[11],
        ]),
        general_level_idc: data[12],
        min_spatial_segmentation_idc: u16::from_be_bytes([data[13] & 0x0F, data[14]]),
        parallelism_type: data[15] & 0x03,
        chroma_format_idc: data[16] & 0x03,
        bit_depth_luma_minus8: data[17] & 0x07,
        bit_depth_chroma_minus8: data[18] & 0x07,
        avg_frame_rate: u16::from_be_bytes([data[19], data[20]]),
        constant_frame_rate: data[21] >> 6,
        num_temporal_layers: (data[21] >> 3) & 0x07,
        temporal_id_nested: data[21] & 0x04 != 0,
        length_size: length_size_minus_one + 1,
        nal_units,
    })
}

/// Split a length-prefixed sample buffer (ISO-BMFF `mdat` payload for
/// one or more access units) into its coded NAL units, `length_size`
/// bytes of big-endian size before each unit.
///
/// # Errors
/// [`HvccError::TruncatedSample`] when the buffer ends mid-prefix or
/// mid-unit; [`HvccError::Nal`] when a unit's header is malformed.
pub fn split_length_prefixed(data: &[u8], length_size: usize) -> Result<Vec<NalUnit>, HvccError> {
    debug_assert!(matches!(length_size, 1 | 2 | 4));
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if pos + length_size > data.len() {
            return Err(HvccError::TruncatedSample);
        }
        let mut len = 0usize;
        for &b in &data[pos..pos + length_size] {
            len = (len << 8) | b as usize;
        }
        pos += length_size;
        if pos + len > data.len() {
            return Err(HvccError::TruncatedSample);
        }
        out.push(nal_unit_from_coded(&data[pos..pos + len])?);
        pos += len;
    }
    Ok(out)
}

#[cfg(any())]
mod tests {
    use super::*;

    /// Build a minimal record: fixed 23-byte prefix + one SPS array
    /// with one 3-byte NAL (header 0x42 0x01 = SPS, + 1 payload byte).
    fn tiny_record(version: u8, length_size_code: u8) -> Vec<u8> {
        let mut r = vec![
            version, // configurationVersion
            0x21,    // space 0, tier 1, profile_idc 1
            0x60,
            0x00,
            0x00,
            0x00, // compatibility flags
            0x90,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00, // constraint flags
            0x5D, // level 93
            0xF0,
            0x07,        // min_spatial_segmentation_idc = 7
            0xFC | 0x01, // parallelismType 1
            0xFC | 0x01, // chroma_format_idc 1
            0xF8 | 0x02, // bit_depth_luma_minus8 = 2
            0xF8 | 0x02, // bit_depth_chroma_minus8 = 2
            0x00,
            0x19,                               // avgFrameRate 25
            (1 << 3) | 0x04 | length_size_code, // 1 layer, nested, length size
            0x01,                               // numOfArrays
        ];
        // Array: SPS (33), 1 NAL of 3 bytes.
        r.extend_from_slice(&[0x21, 0x00, 0x01, 0x00, 0x03, 0x42, 0x01, 0xAA]);
        r
    }

    #[test]
    fn parses_fixed_fields_and_array() {
        let rec = parse_hvcc(&tiny_record(1, 3)).expect("parse");
        assert_eq!(rec.general_profile_space, 0);
        assert!(rec.general_tier_flag);
        assert_eq!(rec.general_profile_idc, 1);
        assert_eq!(rec.general_profile_compatibility_flags, 0x6000_0000);
        assert_eq!(rec.general_constraint_indicator_flags, 0x9000_0000_0000);
        assert_eq!(rec.general_level_idc, 93);
        assert_eq!(rec.min_spatial_segmentation_idc, 7);
        assert_eq!(rec.parallelism_type, 1);
        assert_eq!(rec.chroma_format_idc, 1);
        assert_eq!(rec.bit_depth_luma_minus8, 2);
        assert_eq!(rec.bit_depth_chroma_minus8, 2);
        assert_eq!(rec.avg_frame_rate, 25);
        assert_eq!(rec.num_temporal_layers, 1);
        assert!(rec.temporal_id_nested);
        assert_eq!(rec.length_size, 4);
        assert_eq!(rec.nal_units.len(), 1);
        assert_eq!(rec.nal_units[0].header.nal_unit_type, 33);
        assert_eq!(rec.nal_units[0].rbsp, vec![0xAA]);
    }

    #[test]
    fn rejects_bad_version_and_reserved_length_size() {
        assert_eq!(
            parse_hvcc(&tiny_record(0, 3)).unwrap_err(),
            HvccError::BadVersion(0)
        );
        assert_eq!(
            parse_hvcc(&tiny_record(1, 2)).unwrap_err(),
            HvccError::BadLengthSize
        );
    }

    #[test]
    fn truncation_at_every_seam_is_an_error() {
        let full = tiny_record(1, 3);
        for cut in [10, 22, 24, 26, full.len() - 1] {
            assert_eq!(
                parse_hvcc(&full[..cut]).unwrap_err(),
                HvccError::Truncated,
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn skips_reserved_array_types() {
        let mut r = tiny_record(1, 3);
        r[22] = 2; // numOfArrays = 2
        // Second array with reserved type 41: one 3-byte NAL.
        r.extend_from_slice(&[0x29, 0x00, 0x01, 0x00, 0x03, 0x42, 0x01, 0xBB]);
        let rec = parse_hvcc(&r).expect("parse");
        assert_eq!(rec.nal_units.len(), 1, "reserved-type array ignored");
    }

    #[test]
    fn probe_distinguishes_hvcc_from_annexb() {
        assert!(extradata_is_hvcc(&tiny_record(1, 3)));
        let annexb = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00,
            0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E,
        ];
        assert!(!extradata_is_hvcc(&annexb));
        assert!(!extradata_is_hvcc(&[1, 2, 3]));
    }

    #[test]
    fn splits_length_prefixed_sample_data() {
        // Two NAL units behind 4-byte prefixes: SPS (3 bytes) and an
        // IDR_W_RADL slice (4 bytes) with an emulation escape.
        let sample = [
            0x00, 0x00, 0x00, 0x03, 0x42, 0x01, 0xAA, // SPS
            0x00, 0x00, 0x00, 0x06, 0x26, 0x01, 0x00, 0x00, 0x03, 0x00, // IDR
        ];
        let units = split_length_prefixed(&sample, 4).expect("split");
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].header.nal_unit_type, 33);
        assert_eq!(units[1].header.nal_unit_type, 19);
        // The escape 00 00 03 00 unstripped in `escaped`, stripped in rbsp.
        assert_eq!(units[1].escaped, vec![0x00, 0x00, 0x03, 0x00]);
        assert_eq!(units[1].rbsp, vec![0x00, 0x00, 0x00]);
    }

    #[test]
    fn sample_truncation_is_an_error() {
        let sample = [0x00, 0x00, 0x00, 0x05, 0x42, 0x01];
        assert_eq!(
            split_length_prefixed(&sample, 4).unwrap_err(),
            HvccError::TruncatedSample
        );
        assert_eq!(
            split_length_prefixed(&[0x00, 0x00], 4).unwrap_err(),
            HvccError::TruncatedSample
        );
    }

    #[test]
    fn two_byte_prefix_form() {
        let sample = [0x00, 0x03, 0x42, 0x01, 0xAA];
        let units = split_length_prefixed(&sample, 2).expect("split");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].header.nal_unit_type, 33);
    }
}
