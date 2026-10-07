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
pub use zvidlib_core::codec_config::Vp9CodecConfig;
use zvidlib_core::codec_config::box_payload;

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
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "codec string derivation only supports HEVC, AV1, VP8, VP9, Opus and Vorbis",
        )),
    }
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
