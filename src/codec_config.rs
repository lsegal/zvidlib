//! Portable derivation of WebCodecs codec strings and [`CodecProfile`] values
//! from raw MP4 `hvcC`/`av1C`/`vpcC` codec configuration boxes and for VP8,
//! which has none, and the VP9 codec configuration record
//! ([`Vp9CodecConfig`]).
//!
//! This module contains no browser (`web_sys`) types so it can be unit
//! tested natively. It is consumed by the browser WebCodecs decoder factory
//! to build the `codec` string WebCodecs' `isConfigSupported`/`VideoDecoder`
//! APIs require, and to select a normalized [`CodecProfile`].

use crate::codec::CodecProfile;
use crate::media::Codec;
use crate::{Av1CodecConfigurationRecord, Error, ErrorKind, Limits, Result};

/// A codec string plus the normalized profile it was derived from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedCodecString {
    pub codec_string: String,
    pub profile: CodecProfile,
}

/// Derives a WebCodecs `codec` string and [`CodecProfile`] from a track's
/// complete raw codec configuration box (including its size/fourcc header):
/// `hvcC`, `av1C` or Opus's `dOps`. Vorbis has no MP4 box, so its
/// configuration is the Xiph-laced `CodecPrivate` [`crate::VorbisConfig`]
/// reads.
pub fn derive_codec_string(codec: Codec, decoder_config: &[u8]) -> Result<DerivedCodecString> {
    match codec {
        Codec::Hevc => derive_hevc(decoder_config),
        Codec::Av1 => derive_av1(decoder_config),
        // VP8 has one profile and no configuration record, so its WebCodecs
        // string is the bare codec name.
        Codec::Vp8 => Ok(DerivedCodecString {
            codec_string: "vp8".to_owned(),
            profile: CodecProfile::Vp8,
        }),
        // Opus and Vorbis each have a single codec string; the configuration
        // is still checked so a track the browser is handed is one zvidlib
        // could also decode itself.
        Codec::Opus => {
            crate::OpusHead::from_dops(decoder_config)?;
            Ok(DerivedCodecString {
                codec_string: "opus".into(),
                profile: CodecProfile::Opus,
            })
        }
        Codec::Vorbis => {
            crate::VorbisConfig::from_codec_private(decoder_config)?;
            Ok(DerivedCodecString {
                codec_string: "vorbis".into(),
                profile: CodecProfile::Vorbis,
            })
        }
        Codec::Vp9 => {
            let config = Vp9CodecConfig::parse(decoder_config)?;
            Ok(DerivedCodecString {
                codec_string: config.codec_string(),
                profile: config.codec_profile()?,
            })
        }
        Codec::UncompressedVideo | Codec::H264 | Codec::Aac => Err(Error::new(
            ErrorKind::Unsupported,
            "codec string derivation only supports HEVC, AV1, VP8, VP9, Opus and Vorbis",
        )),
    }
}

/// Returns the payload of a box given its complete bytes (size + fourcc + payload).
pub(crate) fn box_payload<'a>(whole: &'a [u8], expected_fourcc: &[u8; 4]) -> Result<&'a [u8]> {
    if whole.len() < 8 {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "codec configuration box is too short",
        ));
    }
    if &whole[4..8] != expected_fourcc {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "codec configuration box has an unexpected fourcc",
        ));
    }
    Ok(&whole[8..])
}

// --- HEVC (hvcC) -----------------------------------------------------------
//
// ISO/IEC 14496-15 HEVCDecoderConfigurationRecord layout (relevant prefix):
// unsigned int(8) configurationVersion;
// unsigned int(2) general_profile_space;
// unsigned int(1) general_tier_flag;
// unsigned int(5) general_profile_idc;
// unsigned int(32) general_profile_compatibility_flags;
// unsigned int(48) general_constraint_indicator_flags;
// unsigned int(8) general_level_idc;

fn derive_hevc(decoder_config: &[u8]) -> Result<DerivedCodecString> {
    let payload = box_payload(decoder_config, b"hvcC")?;
    if payload.len() < 13 {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "hvcC configuration is too short",
        ));
    }
    let profile_space = (payload[1] >> 6) & 0b11;
    let tier_flag = (payload[1] >> 5) & 0b1;
    let profile_idc = payload[1] & 0b0001_1111;
    let compatibility_flags = u32::from_be_bytes(payload[2..6].try_into().expect("4 bytes"));
    let constraint_flags = &payload[6..12];
    let level_idc = payload[12];

    // General profile/tier/level string per ISO/IEC 14496-15 Annex E and the
    // WebCodecs HEVC registration ("hev1"/"hvc1" fourcc followed by a dot,
    // then the general_profile_space letter (if any) plus profile_idc, tier
    // ('L' or 'H'), level_idc, then '.' plus each constraint byte in hex,
    // trailing zero bytes omitted).
    let profile_space_letter = match profile_space {
        0 => String::new(),
        1 => "A".to_owned(),
        2 => "B".to_owned(),
        3 => "C".to_owned(),
        _ => unreachable!("profile_space is masked to 2 bits"),
    };
    let tier_letter = if tier_flag == 1 { 'H' } else { 'L' };

    let mut constraint_hex = String::new();
    let mut last_nonzero = None;
    for (index, byte) in constraint_flags.iter().enumerate() {
        if *byte != 0 {
            last_nonzero = Some(index);
        }
    }
    if let Some(last) = last_nonzero {
        for byte in &constraint_flags[..=last] {
            constraint_hex.push('.');
            constraint_hex.push_str(&format!("{byte:02X}"));
        }
    }

    let codec_string = format!(
        "hev1.{profile_space_letter}{profile_idc}.{compatibility_flags:X}.{tier_letter}{level_idc}{constraint_hex}"
    );

    let profile = match profile_idc {
        2 => CodecProfile::HevcMain10,
        _ => CodecProfile::HevcMain,
    };

    Ok(DerivedCodecString {
        codec_string,
        profile,
    })
}

// --- AV1 (av1C) --------------------------------------------------------
//
// AV1 Codec ISO Media File Format Binding, AV1CodecConfigurationRecord:
// unsigned int (1) marker = 1;
// unsigned int (7) version = 1;
// unsigned int (3) seq_profile;
// unsigned int (5) seq_level_idx_0;
// unsigned int (1) seq_tier_0;
// unsigned int (1) high_bitdepth;
// unsigned int (1) twelve_bit;
// unsigned int (1) monochrome;
// unsigned int (1) chroma_subsampling_x;
// unsigned int (1) chroma_subsampling_y;
// unsigned int (2) chroma_sample_position;
// ...

fn derive_av1(decoder_config: &[u8]) -> Result<DerivedCodecString> {
    let record = Av1CodecConfigurationRecord::parse(decoder_config, &Limits::default())?;
    let seq_profile = record.seq_profile;
    let seq_level_idx_0 = record.seq_level_idx_0;

    let bit_depth: u32 = if record.high_bitdepth {
        if seq_profile == 2 && record.twelve_bit {
            12
        } else {
            10
        }
    } else {
        8
    };

    let tier_letter = if record.seq_tier_0 { 'H' } else { 'M' };
    let mono = u8::from(record.monochrome);
    let chroma_subsampling_x = u8::from(record.chroma_subsampling_x);
    let chroma_subsampling_y = u8::from(record.chroma_subsampling_y);
    let chroma_sample_position = record.chroma_sample_position;
    let codec_string = format!(
        "av01.{seq_profile}.{seq_level_idx_0:02}{tier_letter}.{bit_depth:02}.{mono}.{chroma_subsampling_x}{chroma_subsampling_y}{chroma_sample_position}"
    );

    Ok(DerivedCodecString {
        codec_string,
        profile: match seq_profile {
            0 => CodecProfile::Av1Main,
            1 => CodecProfile::Av1High,
            2 => CodecProfile::Av1Professional,
            _ => unreachable!("validated AV1 profile"),
        },
    })
}

// --- VP9 (vpcC and WebM CodecPrivate) -------------------------------------

/// A VP9 stream's codec configuration: what an MP4 `vpcC` box or a WebM
/// `CodecPrivate` element says about it.
///
/// The VP9 bitstream describes itself, so this is a summary for capability
/// discovery and codec strings rather than something decoding needs. The
/// `vpcC` box is version 1 of the VP Codec ISO Media File Format Binding
/// (`VPCodecConfigurationRecord`); version 0, an earlier draft some muxers
/// still write, is read as well. WebM's `CodecPrivate` is the optional
/// list of (ID, length, value) features of the WebM VP9 codec mapping, of
/// which profile (1), level (2), bit depth (3) and chroma subsampling (4)
/// are defined; any it leaves out take the profile 0 defaults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Vp9CodecConfig {
    pub profile: u8,
    /// The VP9 level times ten (`31` is level 3.1), or 0 when unspecified.
    pub level: u8,
    pub bit_depth: u8,
    /// 0 and 1 are 4:2:0 (vertical and colocated chroma), 2 is 4:2:2 and
    /// 3 is 4:4:4.
    pub chroma_subsampling: u8,
    pub video_full_range: bool,
    /// ISO/IEC 23091-2 colour description; 2 means unspecified.
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

impl Default for Vp9CodecConfig {
    fn default() -> Self {
        Self {
            profile: 0,
            level: 0,
            bit_depth: 8,
            chroma_subsampling: 0,
            video_full_range: false,
            colour_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
        }
    }
}

const VP9_LEVELS: [u8; 14] = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 52, 60, 61, 62];

impl Vp9CodecConfig {
    /// Reads a complete `vpcC` box (size, type and payload) or, when the
    /// bytes are not one, a WebM `CodecPrivate`. Empty bytes are an empty
    /// `CodecPrivate`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= 8 && &bytes[4..8] == b"vpcC" {
            Self::parse_vpcc(bytes)
        } else {
            Self::parse_webm_codec_private(bytes)
        }
    }

    /// Reads a complete `vpcC` box.
    pub fn parse_vpcc(whole: &[u8]) -> Result<Self> {
        let payload = box_payload(whole, b"vpcC")?;
        let declared = u32::from_be_bytes(whole[..4].try_into().expect("4 bytes"));
        if declared as usize != whole.len() {
            return Err(malformed_config("vpcC box size is inconsistent"));
        }
        if payload.len() < 4 {
            return Err(malformed_config("vpcC box is too short"));
        }
        let version = payload[0];
        let record = &payload[4..];
        let config = match version {
            1 => {
                if record.len() < 8 {
                    return Err(malformed_config("vpcC record is too short"));
                }
                Self {
                    profile: record[0],
                    level: record[1],
                    bit_depth: record[2] >> 4,
                    chroma_subsampling: (record[2] >> 1) & 0b111,
                    video_full_range: record[2] & 1 == 1,
                    colour_primaries: record[3],
                    transfer_characteristics: record[4],
                    matrix_coefficients: record[5],
                }
            }
            0 => {
                // bitDepth(4) colorSpace(4) chromaSubsampling(4)
                // transferFunction(4) videoFullRangeFlag(1) reserved(7)
                if record.len() < 7 {
                    return Err(malformed_config("vpcC record is too short"));
                }
                Self {
                    profile: record[0],
                    level: record[1],
                    bit_depth: record[2] >> 4,
                    chroma_subsampling: record[3] >> 4,
                    video_full_range: record[4] >> 7 == 1,
                    ..Self::default()
                }
            }
            _ => return Err(malformed_config("unsupported vpcC version")),
        };
        config.validate()
    }

    /// Reads a WebM VP9 `CodecPrivate`.
    pub fn parse_webm_codec_private(bytes: &[u8]) -> Result<Self> {
        let mut config = Self::default();
        let mut rest = bytes;
        while !rest.is_empty() {
            if rest.len() < 2 {
                return Err(malformed_config("VP9 CodecPrivate feature is truncated"));
            }
            let (id, length) = (rest[0], usize::from(rest[1]));
            let value = rest
                .get(2..2 + length)
                .ok_or_else(|| malformed_config("VP9 CodecPrivate feature is truncated"))?;
            if (1..=4).contains(&id) {
                if length != 1 {
                    return Err(malformed_config(
                        "VP9 CodecPrivate feature has an invalid length",
                    ));
                }
                match id {
                    1 => config.profile = value[0],
                    2 => config.level = value[0],
                    3 => config.bit_depth = value[0],
                    _ => config.chroma_subsampling = value[0],
                }
            }
            rest = &rest[2 + length..];
        }
        config.validate()
    }

    fn validate(self) -> Result<Self> {
        if self.profile > 3 {
            return Err(malformed_config("VP9 profile must be 0 to 3"));
        }
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(malformed_config("VP9 bit depth must be 8, 10 or 12"));
        }
        if self.chroma_subsampling > 3 {
            return Err(malformed_config("VP9 chroma subsampling is invalid"));
        }
        Ok(self)
    }

    /// Whether the stream is 4:2:0.
    pub fn is_420(&self) -> bool {
        self.chroma_subsampling <= 1
    }

    /// The normalized profile.
    pub fn codec_profile(&self) -> Result<CodecProfile> {
        Ok(match self.profile {
            0 => CodecProfile::Vp9Profile0,
            1 => CodecProfile::Vp9Profile1,
            2 => CodecProfile::Vp9Profile2,
            3 => CodecProfile::Vp9Profile3,
            _ => return Err(malformed_config("VP9 profile must be 0 to 3")),
        })
    }

    /// The complete version 1 `vpcC` box describing this configuration,
    /// with no codec initialization data.
    pub fn to_vpcc(&self) -> Vec<u8> {
        let mut bytes = 20u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"vpcC");
        bytes.extend_from_slice(&[1, 0, 0, 0, self.profile, self.level]);
        bytes.push(
            (self.bit_depth << 4)
                | (self.chroma_subsampling << 1)
                | u8::from(self.video_full_range),
        );
        bytes.extend_from_slice(&[
            self.colour_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
            0,
            0,
        ]);
        bytes
    }

    /// The `vp09.PP.LL.DD` codec string of the VP9 ISO-BMFF binding, which
    /// WebCodecs and `MediaSource` accept. An unspecified or invalid level
    /// is written as level 1.0, since the string requires one.
    pub fn codec_string(&self) -> String {
        let level = if VP9_LEVELS.contains(&self.level) {
            self.level
        } else {
            10
        };
        format!("vp09.{:02}.{level:02}.{:02}", self.profile, self.bit_depth)
    }
}

fn malformed_config(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = (8 + payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn derives_hevc_main_profile_codec_string() {
        // configurationVersion=1, profile_space=0, tier=0, profile_idc=1 (Main)
        let mut payload = vec![1_u8, 0b0000_0001];
        payload.extend_from_slice(&0x6000_0000_u32.to_be_bytes()); // compatibility flags
        payload.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // constraint flags (all zero)
        payload.push(93); // general_level_idc
        payload.extend_from_slice(&[0; 12]); // remaining fixed fields (not parsed)
        let hvcc = boxed(b"hvcC", &payload);

        let derived = derive_codec_string(Codec::Hevc, &hvcc).unwrap();
        assert_eq!(derived.codec_string, "hev1.1.60000000.L93");
        assert_eq!(derived.profile, CodecProfile::HevcMain);
    }

    #[test]
    fn derives_hevc_main10_profile_from_profile_idc_two() {
        let mut payload = vec![1_u8, 0b0000_0010];
        payload.extend_from_slice(&0x6000_0000_u32.to_be_bytes());
        payload.extend_from_slice(&[0xB0, 0, 0, 0, 0, 0]);
        payload.push(120);
        payload.extend_from_slice(&[0; 12]);
        let hvcc = boxed(b"hvcC", &payload);

        let derived = derive_codec_string(Codec::Hevc, &hvcc).unwrap();
        assert_eq!(derived.profile, CodecProfile::HevcMain10);
        assert_eq!(derived.codec_string, "hev1.2.60000000.L120.B0");
    }

    #[test]
    fn derives_av1_main_profile_codec_string() {
        // marker=1, version=1, seq_profile=0, seq_level_idx_0=4 (matching the
        // repo's existing av1C test fixture bytes: [0x81, 0, 0, 0]).
        let av1c = boxed(b"av1C", &[0x81, 0x04, 0, 0]);
        let derived = derive_codec_string(Codec::Av1, &av1c).unwrap();
        assert_eq!(derived.profile, CodecProfile::Av1Main);
        assert_eq!(derived.codec_string, "av01.0.04M.08.0.000");
    }

    #[test]
    fn av1_codec_string_carries_every_chroma_sample_position() {
        // 4:2:0 colour with each `chroma_sample_position`; the value is the last
        // digit of the `av01.` string.
        for pos in 0..4u8 {
            let av1c = boxed(b"av1C", &[0x81, 0x04, 0x0c | pos, 0]);
            let derived = derive_codec_string(Codec::Av1, &av1c).unwrap();
            assert_eq!(derived.profile, CodecProfile::Av1Main);
            assert_eq!(derived.codec_string, format!("av01.0.04M.08.0.11{pos}"));
        }
    }

    #[test]
    fn distinguishes_av1_high_and_professional_profiles() {
        let high = boxed(b"av1C", &[0x81, 0x24, 0, 0]);
        assert_eq!(
            derive_codec_string(Codec::Av1, &high).unwrap().profile,
            CodecProfile::Av1High
        );
        let professional = boxed(b"av1C", &[0x81, 0x44, 0x68, 0]);
        assert_eq!(
            derive_codec_string(Codec::Av1, &professional)
                .unwrap()
                .profile,
            CodecProfile::Av1Professional
        );
    }

    #[test]
    fn rejects_wrong_fourcc() {
        let bogus = boxed(b"esds", &[0, 0, 0, 0]);
        let error = derive_codec_string(Codec::Hevc, &bogus).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    }

    #[test]
    fn rejects_truncated_configuration() {
        let short = boxed(b"hvcC", &[1, 2]);
        let error = derive_codec_string(Codec::Hevc, &short).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);

        let short_av1 = boxed(b"av1C", &[1]);
        let error = derive_codec_string(Codec::Av1, &short_av1).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    }

    #[test]
    fn derives_the_vp8_codec_string_without_a_configuration_record() {
        let derived = derive_codec_string(Codec::Vp8, &[]).unwrap();
        assert_eq!(derived.codec_string, "vp8");
        assert_eq!(derived.profile, CodecProfile::Vp8);
    }

    #[test]
    fn rejects_unsupported_codecs() {
        let error = derive_codec_string(Codec::Aac, &boxed(b"esds", &[0; 4])).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let error =
            derive_codec_string(Codec::UncompressedVideo, &boxed(b"hvcC", &[0; 13])).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    fn vpcc(version: u8, record: &[u8]) -> Vec<u8> {
        let mut payload = vec![version, 0, 0, 0];
        payload.extend_from_slice(record);
        boxed(b"vpcC", &payload)
    }

    #[test]
    fn reads_a_version_1_vpcc_box_and_derives_its_codec_string() {
        // Profile 0, level 3.1, 8-bit, colocated 4:2:0, limited range,
        // BT.709 colour description, no initialization data.
        let bytes = vpcc(1, &[0, 31, 0x82, 1, 1, 1, 0, 0]);
        let config = Vp9CodecConfig::parse(&bytes).unwrap();
        assert_eq!(
            config,
            Vp9CodecConfig {
                profile: 0,
                level: 31,
                bit_depth: 8,
                chroma_subsampling: 1,
                video_full_range: false,
                colour_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
            }
        );
        assert!(config.is_420());
        let derived = derive_codec_string(Codec::Vp9, &bytes).unwrap();
        assert_eq!(derived.codec_string, "vp09.00.31.08");
        assert_eq!(derived.profile, CodecProfile::Vp9Profile0);

        let ten_bit = vpcc(1, &[2, 0, 0xa1, 9, 16, 9, 0, 0]);
        let config = Vp9CodecConfig::parse(&ten_bit).unwrap();
        assert!(config.video_full_range);
        assert_eq!(config.codec_profile().unwrap(), CodecProfile::Vp9Profile2);
        // Level 0 is unspecified, which the codec string cannot say.
        assert_eq!(config.codec_string(), "vp09.02.10.10");
    }

    #[test]
    fn reads_a_version_0_vpcc_box() {
        let bytes = vpcc(0, &[1, 20, 0x80, 0x30, 0x80, 0, 0]);
        let config = Vp9CodecConfig::parse(&bytes).unwrap();
        assert_eq!(config.profile, 1);
        assert_eq!(config.level, 20);
        assert_eq!(config.bit_depth, 8);
        assert_eq!(config.chroma_subsampling, 3);
        assert!(config.video_full_range);
    }

    #[test]
    fn reads_webm_codec_private_features() {
        assert_eq!(
            Vp9CodecConfig::parse(&[]).unwrap(),
            Vp9CodecConfig::default()
        );
        let config = Vp9CodecConfig::parse(&[1, 1, 0, 2, 1, 40, 3, 1, 8, 4, 1, 1]).unwrap();
        assert_eq!((config.profile, config.level, config.bit_depth), (0, 40, 8));
        assert_eq!(config.chroma_subsampling, 1);
        // Unknown features are skipped.
        assert_eq!(
            Vp9CodecConfig::parse(&[9, 2, 7, 7, 1, 1, 2])
                .unwrap()
                .profile,
            2
        );
    }

    #[test]
    fn rejects_malformed_vp9_configurations() {
        for bytes in [
            vec![1, 1],
            vec![1, 2, 0, 0],
            vec![3, 1, 9],
            vpcc(1, &[0, 10]),
            vpcc(2, &[0, 10, 0x80, 1, 1, 1, 0, 0]),
            vpcc(1, &[4, 10, 0x80, 1, 1, 1, 0, 0]),
            vpcc(1, &[0, 10, 0x90, 1, 1, 1, 0, 0]),
        ] {
            let error = Vp9CodecConfig::parse(&bytes).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MalformedMedia, "{bytes:?}");
        }
        let config = Vp9CodecConfig::parse(&[1, 1, 0, 2, 1, 41, 4, 1, 1]).unwrap();
        assert_eq!(Vp9CodecConfig::parse(&config.to_vpcc()).unwrap(), config);
        let mut inconsistent = vpcc(1, &[0, 10, 0x80, 1, 1, 1, 0, 0]);
        inconsistent[3] += 1;
        assert!(Vp9CodecConfig::parse(&inconsistent).is_err());
    }
}
