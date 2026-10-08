//! The codec configuration records more than one zvidlib crate reads: the VP9
//! codec configuration record ([`Vp9CodecConfig`]), and the ISO-BMFF box
//! unwrapping every `*C` record shares.

use crate::{CodecProfile, Error, ErrorKind, Result};

/// Returns the payload of a box given its complete bytes (size + fourcc + payload).
#[doc(hidden)]
pub fn box_payload<'a>(whole: &'a [u8], expected_fourcc: &[u8; 4]) -> Result<&'a [u8]> {
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
    /// ISO/IEC 23091-2 color description; 2 means unspecified.
    pub color_primaries: u8,
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
            color_primaries: 2,
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
                    color_primaries: record[3],
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
            self.color_primaries,
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
