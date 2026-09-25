//! MP4 framing for HEVC produced by an encoder this crate does not write itself.
//!
//! A platform encoder hands back length-prefixed access units and, separately, the VPS, SPS, and
//! PPS it chose. An `hvc1` track declares the parameter sets once, in the sample entry's `hvcC`,
//! and carries only the pictures in its samples. This module is the part of that conversion that
//! does not depend on the platform: building the `hvcC` from the parameter sets, and separating an
//! access unit's out-of-band units from its pictures. It is compiled everywhere so it is tested
//! everywhere, including on hosts whose encoders never call it.

use super::engine::hvcc::nal_unit_from_coded;
use super::engine::sps::SeqParameterSet;
use crate::{Error, ErrorKind, Result};

const NAL_VPS: u8 = 32;
const NAL_SPS: u8 = 33;
const NAL_PPS: u8 = 34;
const NAL_AUD: u8 = 35;

/// The `nal_unit_type` of a coded NAL unit, from its two-byte header.
fn nal_type(unit: &[u8]) -> u8 {
    unit.first().map_or(0, |byte| (byte >> 1) & 0x3f)
}

/// The VPS, SPS, and PPS NAL units of one stream, each coded with its two-byte header and without
/// a start code or length prefix. Exact repeats are kept once.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct ParameterSets {
    pub(super) vps: Vec<Vec<u8>>,
    pub(super) sps: Vec<Vec<u8>>,
    pub(super) pps: Vec<Vec<u8>>,
}

impl ParameterSets {
    /// Records `unit` when it is a parameter set, returning whether it was one.
    pub(super) fn collect(&mut self, unit: &[u8]) -> bool {
        let list = match nal_type(unit) {
            NAL_VPS => &mut self.vps,
            NAL_SPS => &mut self.sps,
            NAL_PPS => &mut self.pps,
            _ => return false,
        };
        if !list.iter().any(|known| known == unit) {
            list.push(unit.to_vec());
        }
        true
    }

    /// Whether every set in `other` is already one of these, so a stream that declared `self` in
    /// its `hvcC` can carry pictures that reference `other`.
    pub(super) fn covers(&self, other: &Self) -> bool {
        [
            (&self.vps, &other.vps),
            (&self.sps, &other.sps),
            (&self.pps, &other.pps),
        ]
        .into_iter()
        .all(|(known, sets)| sets.iter().all(|set| known.contains(set)))
    }

    /// The complete `hvcC` box, header included, declaring these parameter sets with four-byte
    /// NAL unit lengths.
    ///
    /// The profile, tier, level, and constraint fields are copied from the SPS rather than
    /// assumed, because the encoder chose them: `profile_tier_level()` is the first thing in the
    /// SPS after one byte of identifiers, and its general fields are byte-aligned there, so they
    /// are the SPS's own bytes 1 to 12. Only the Main profile at 8-bit 4:2:0 is accepted, which is
    /// the only thing an `HevcMain` caller may be handed.
    pub(super) fn hvcc(&self) -> Result<Vec<u8>> {
        let sps = self
            .sps
            .first()
            .ok_or_else(|| codec("the HEVC encoder produced no sequence parameter set"))?;
        if self.vps.is_empty() || self.pps.is_empty() {
            return Err(codec(
                "the HEVC encoder produced no video or picture parameter set",
            ));
        }
        let unit =
            nal_unit_from_coded(sps).map_err(|e| codec(format!("invalid HEVC SPS: {e}")))?;
        let parsed = SeqParameterSet::parse(&unit.rbsp)
            .map_err(|e| codec(format!("invalid HEVC SPS: {e:?}")))?;
        let rbsp = &unit.rbsp;
        if rbsp.len() < 13 {
            return Err(codec("the HEVC SPS is too short for profile_tier_level()"));
        }
        let profile_idc = rbsp[1] & 0x1f;
        let compatibility = u32::from_be_bytes([rbsp[2], rbsp[3], rbsp[4], rbsp[5]]);
        let main_compatible = profile_idc == 1 || compatibility & (1 << (31 - 1)) != 0;
        if !main_compatible
            || parsed.chroma_format_idc != 1
            || parsed.bit_depth_luma_minus8 != 0
            || parsed.bit_depth_chroma_minus8 != 0
        {
            return Err(codec(
                "the HEVC encoder produced a stream that is not 8-bit 4:2:0 Main profile",
            ));
        }

        let mut body = Vec::with_capacity(
            23 + [&self.vps, &self.sps, &self.pps]
                .iter()
                .flat_map(|sets| sets.iter())
                .map(|set| set.len() + 5)
                .sum::<usize>(),
        );
        body.push(1); // configurationVersion
        body.extend_from_slice(&rbsp[1..13]);
        body.extend_from_slice(&0xf000_u16.to_be_bytes()); // min_spatial_segmentation_idc 0
        body.push(0xfc); // parallelismType 0: unknown
        body.push(0xfc | parsed.chroma_format_idc);
        body.push(0xf8 | parsed.bit_depth_luma_minus8);
        body.push(0xf8 | parsed.bit_depth_chroma_minus8);
        body.extend_from_slice(&0_u16.to_be_bytes()); // avgFrameRate unspecified
        // constantFrameRate 0, numTemporalLayers, temporalIdNested, lengthSizeMinusOne 3.
        body.push(
            ((parsed.max_sub_layers_minus1 + 1) & 7) << 3
                | u8::from(parsed.temporal_id_nesting_flag) << 2
                | 0b11,
        );
        body.push(3); // numOfArrays
        for (kind, sets) in [
            (NAL_VPS, &self.vps),
            (NAL_SPS, &self.sps),
            (NAL_PPS, &self.pps),
        ] {
            body.push(0x80 | kind); // array_completeness 1
            let count = u16::try_from(sets.len())
                .map_err(|_| codec("too many HEVC parameter sets for an hvcC array"))?;
            body.extend_from_slice(&count.to_be_bytes());
            for set in sets {
                let length = u16::try_from(set.len())
                    .map_err(|_| codec("an HEVC parameter set is too long for hvcC"))?;
                body.extend_from_slice(&length.to_be_bytes());
                body.extend_from_slice(set);
            }
        }
        let size = u32::try_from(body.len() + 8).map_err(|_| codec("hvcC box overflows"))?;
        let mut out = Vec::with_capacity(body.len() + 8);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(b"hvcC");
        out.extend(body);
        Ok(out)
    }
}

/// Splits one four-byte length-prefixed access unit into the MP4 sample an `hvc1` track carries
/// and the out-of-band units it does not.
///
/// Parameter sets are collected into `sets`; access unit delimiters are dropped, since an MP4
/// sample is already one access unit; everything else - the slices and any SEI - stays in the
/// sample, in order.
pub(super) fn split_access_unit(data: &[u8], sets: &mut ParameterSets) -> Result<Vec<u8>> {
    let mut sample = Vec::with_capacity(data.len());
    let mut has_picture = false;
    let mut rest = data;
    while !rest.is_empty() {
        let prefix: [u8; 4] = rest
            .get(..4)
            .and_then(|prefix| prefix.try_into().ok())
            .ok_or_else(|| codec("HEVC access unit ends inside a NAL unit length"))?;
        let length = u32::from_be_bytes(prefix) as usize;
        let unit = rest
            .get(4..)
            .and_then(|tail| tail.get(..length))
            .ok_or_else(|| codec("HEVC access unit ends inside a NAL unit"))?;
        rest = &rest[4 + length..];
        if unit.len() < 2 {
            return Err(codec("HEVC NAL unit is shorter than its header"));
        }
        if sets.collect(unit) || nal_type(unit) == NAL_AUD {
            continue;
        }
        has_picture |= nal_type(unit) < 32;
        sample.extend_from_slice(&prefix);
        sample.extend_from_slice(unit);
    }
    if has_picture {
        Ok(sample)
    } else {
        Err(codec("HEVC access unit carries no coded picture"))
    }
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

#[cfg(test)]
mod tests {
    use super::super::encoder::hvcc_box;
    use super::super::engine::encoder::pcm::encode_idr_pcm_au;
    use super::super::engine::hvcc::parse_hvcc;
    use super::super::engine::nal::collect_nal_units;
    use super::*;
    use crate::{
        Codec, CodecProfile, ColorRange, HardwarePreference, Limits, PixelFormat,
        VideoDecoderConfig, VideoDecoderFactory, VideoDimensions,
        native_hevc_video_decoder_factory,
    };

    /// One access unit from this crate's own writer, as coded NAL units with their headers.
    fn pcm_units(width: usize, height: usize) -> Vec<Vec<u8>> {
        let luma = vec![90; width * height];
        let chroma = vec![128; width * height / 4];
        let annexb = encode_idr_pcm_au(&luma, &chroma, &chroma, width, height).unwrap();
        collect_nal_units(&annexb)
            .unwrap()
            .into_iter()
            .map(|unit| {
                let header = unit.header;
                let mut coded = vec![
                    (header.nal_unit_type << 1) | (header.nuh_layer_id >> 5),
                    ((header.nuh_layer_id & 0x1f) << 3) | (header.temporal_id + 1),
                ];
                coded.extend_from_slice(&unit.escaped);
                coded
            })
            .collect()
    }

    fn length_prefixed(units: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for unit in units {
            out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
            out.extend_from_slice(unit);
        }
        out
    }

    #[test]
    fn splits_parameter_sets_and_delimiters_out_of_an_access_unit() {
        let units = pcm_units(32, 32);
        let delimiter: &[u8] = &[NAL_AUD << 1, 1, 0x50];
        let mut all: Vec<&[u8]> = vec![delimiter];
        all.extend(units.iter().map(Vec::as_slice));
        let mut sets = ParameterSets::default();
        let sample = split_access_unit(&length_prefixed(&all), &mut sets).unwrap();

        let pictures: Vec<&[u8]> = units
            .iter()
            .filter(|unit| nal_type(unit) < 32)
            .map(Vec::as_slice)
            .collect();
        assert!(!pictures.is_empty());
        assert_eq!(sample, length_prefixed(&pictures));
        assert_eq!((sets.vps.len(), sets.sps.len(), sets.pps.len()), (1, 1, 1));

        // The same sets again are repeats, not new ones.
        let before = sets.clone();
        split_access_unit(&length_prefixed(&all), &mut sets).unwrap();
        assert_eq!(sets, before);
        assert!(before.covers(&sets));
        assert!(!ParameterSets::default().covers(&sets));
    }

    #[test]
    fn rejects_truncated_or_picture_less_access_units() {
        let units = pcm_units(32, 32);
        let whole = length_prefixed(&units.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let mut sets = ParameterSets::default();
        assert!(split_access_unit(&whole[..whole.len() - 1], &mut sets).is_err());
        assert!(split_access_unit(&whole[..3], &mut sets).is_err());
        let only_sets: Vec<&[u8]> = units
            .iter()
            .filter(|unit| nal_type(unit) >= 32)
            .map(Vec::as_slice)
            .collect();
        assert!(split_access_unit(&length_prefixed(&only_sets), &mut sets).is_err());
    }

    /// The `hvcC` built from a stream's parameter sets declares what the SPS says, agrees with
    /// the one this crate's own encoder writes wherever that one is not a fixed guess, and is
    /// accepted by this crate's decoder.
    #[test]
    fn builds_an_hvcc_the_decoder_accepts_from_parameter_sets() {
        let (width, height) = (64, 48);
        let units = pcm_units(width, height);
        let mut sets = ParameterSets::default();
        for unit in &units {
            sets.collect(unit);
        }
        let built = sets.hvcc().unwrap();
        assert_eq!(&built[4..8], b"hvcC");
        assert_eq!(
            u32::from_be_bytes(built[..4].try_into().unwrap()) as usize,
            built.len()
        );

        let record = parse_hvcc(&built[8..]).unwrap();
        assert_eq!(record.general_profile_idc, 1, "Main profile");
        assert_eq!(record.chroma_format_idc, 1);
        assert_eq!(record.bit_depth_luma_minus8, 0);
        assert_eq!(record.bit_depth_chroma_minus8, 0);
        assert_eq!(record.length_size, 4);
        let kinds: Vec<u8> = record
            .nal_units
            .iter()
            .map(|unit| unit.header.nal_unit_type)
            .collect();
        assert_eq!(kinds, [NAL_VPS, NAL_SPS, NAL_PPS]);

        let annexb = {
            let luma = vec![90; width * height];
            let chroma = vec![128; width * height / 4];
            encode_idr_pcm_au(&luma, &chroma, &chroma, width, height).unwrap()
        };
        let reference = parse_hvcc(&hvcc_box(&annexb).unwrap()[8..]).unwrap();
        assert_eq!(record.general_profile_space, reference.general_profile_space);
        assert_eq!(record.general_tier_flag, reference.general_tier_flag);
        assert_eq!(
            record.general_profile_compatibility_flags,
            reference.general_profile_compatibility_flags
        );
        assert_eq!(record.general_level_idc, reference.general_level_idc);
        assert_eq!(
            record
                .nal_units
                .iter()
                .map(|unit| &unit.escaped)
                .collect::<Vec<_>>(),
            reference
                .nal_units
                .iter()
                .map(|unit| &unit.escaped)
                .collect::<Vec<_>>()
        );

        let limits = Limits::default();
        let configuration = VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(width as u32, height as u32, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: built,
        };
        native_hevc_video_decoder_factory()
            .create(&configuration, &limits)
            .unwrap();
    }

    #[test]
    fn an_hvcc_needs_all_three_parameter_set_kinds() {
        let units = pcm_units(32, 32);
        for missing in [NAL_VPS, NAL_SPS, NAL_PPS] {
            let mut sets = ParameterSets::default();
            for unit in units.iter().filter(|unit| nal_type(unit) != missing) {
                sets.collect(unit);
            }
            assert!(sets.hvcc().is_err(), "an hvcC without type {missing}");
        }
    }
}
