//! Annex B framing for platform HEVC encoders: splitting an encoder's byte
//! stream into NAL units, rewriting an access unit as the four-byte
//! length-prefixed sample an `hvc1` track carries, and building that track's
//! `hvcC` from the VPS/SPS/PPS the encoder emitted.
//!
//! The crate's own encoder never needs this - it writes its parameter sets
//! itself and knows every field of them - but a platform encoder only hands
//! back an Annex B stream, so the `hvcC` has to be recovered from the SPS it
//! actually wrote, profile and constraint flags included. Ported from
//! lsegal/zvid@7fe6a80 `record/bitstream.rs` (issue #487).

pub(super) const NAL_VPS: u8 = 32;
pub(super) const NAL_SPS: u8 = 33;
pub(super) const NAL_PPS: u8 = 34;
pub(super) const NAL_AUD: u8 = 35;

/// The NAL unit type of an HEVC NAL unit (with its two-byte header).
pub(super) fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |byte| (byte >> 1) & 0x3f)
}

/// Whether `nal` is a VCL NAL unit, i.e. carries slice data.
pub(super) fn is_vcl(nal: &[u8]) -> bool {
    nal_type(nal) < 32
}

/// Whether `nal` is a random access point (BLA, IDR or CRA picture).
pub(super) fn is_irap(nal: &[u8]) -> bool {
    (16..=23).contains(&nal_type(nal))
}

/// Whether a NAL unit is a parameter set or delimiter, which `hvc1` tracks
/// carry in the sample entry rather than in samples.
pub(super) fn is_out_of_band(nal: &[u8]) -> bool {
    matches!(nal_type(nal), NAL_VPS | NAL_SPS | NAL_PPS | NAL_AUD)
}

/// Splits an Annex B byte stream into NAL units without start codes.
pub(super) fn split_annex_b(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 <= stream.len() {
        if stream[index] == 0 && stream[index + 1] == 0 && stream[index + 2] == 1 {
            starts.push(index + 3);
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut units = Vec::with_capacity(starts.len());
    for (position, &start) in starts.iter().enumerate() {
        let mut end = starts
            .get(position + 1)
            .map_or(stream.len(), |next| next - 3);
        // Trailing zeros belong to the next four-byte start code.
        while end > start && stream[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            units.push(&stream[start..end]);
        }
    }
    units
}

/// Joins NAL units into one four-byte length-prefixed MP4 sample.
pub(super) fn length_prefixed<'a>(units: impl IntoIterator<Item = &'a [u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
        out.extend_from_slice(unit);
    }
    out
}

/// Removes emulation prevention bytes, giving the RBSP.
fn rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &byte in nal {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

struct Bits<'a> {
    data: &'a [u8],
    position: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.position / 8)?;
        let bit = (byte >> (7 - self.position % 8)) & 1;
        self.position += 1;
        Some(u32::from(bit))
    }

    fn bits(&mut self, count: u32) -> Option<u64> {
        (0..count).try_fold(0u64, |value, _| Some((value << 1) | u64::from(self.bit()?)))
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some(((1u64 << zeros) - 1 + self.bits(zeros)?) as u32)
    }
}

/// Fields of an SPS that `hvcC` repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SpsInfo {
    /// `general_profile_space`, `general_tier_flag`, `general_profile_idc`.
    pub profile_byte: u8,
    pub compatibility_flags: u32,
    /// The 48 constraint indicator bits.
    pub constraint_flags: u64,
    pub level_idc: u8,
    pub max_sub_layers: u8,
    pub temporal_id_nesting: bool,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    /// Luma dimensions after the conformance window crop.
    pub width: u32,
    pub height: u32,
}

/// Parses the fields `hvcC` needs from an SPS NAL unit.
pub(super) fn parse_sps(nal: &[u8]) -> Option<SpsInfo> {
    let data = rbsp(nal);
    let mut bits = Bits {
        data: data.get(2..)?,
        position: 0,
    };
    bits.bits(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = bits.bits(3)? as u8;
    let temporal_id_nesting = bits.bit()? == 1;
    let profile_byte = bits.bits(8)? as u8;
    let compatibility_flags = bits.bits(32)? as u32;
    let constraint_flags = bits.bits(48)?;
    let level_idc = bits.bits(8)? as u8;
    let mut sub_layer_flags = Vec::new();
    for _ in 0..max_sub_layers_minus1 {
        sub_layer_flags.push((bits.bit()?, bits.bit()?));
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            bits.bits(2)?;
        }
    }
    for (profile_present, level_present) in sub_layer_flags {
        if profile_present == 1 {
            bits.bits(88)?;
        }
        if level_present == 1 {
            bits.bits(8)?;
        }
    }
    bits.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = bits.ue()? as u8;
    if chroma_format_idc == 3 {
        bits.bit()?;
    }
    let mut width = bits.ue()?;
    let mut height = bits.ue()?;
    if bits.bit()? == 1 {
        let (sub_width, sub_height) = match chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let (left, right, top, bottom) = (bits.ue()?, bits.ue()?, bits.ue()?, bits.ue()?);
        width = width.checked_sub(sub_width * (left + right))?;
        height = height.checked_sub(sub_height * (top + bottom))?;
    }
    let bit_depth_luma_minus8 = bits.ue()? as u8;
    let bit_depth_chroma_minus8 = bits.ue()? as u8;
    Some(SpsInfo {
        profile_byte,
        compatibility_flags,
        constraint_flags,
        level_idc,
        max_sub_layers: max_sub_layers_minus1 + 1,
        temporal_id_nesting,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        width,
        height,
    })
}

/// Builds an `hvcC` box (with header) from VPS, SPS and PPS NAL units.
pub(super) fn hvcc_box(vps: &[&[u8]], sps: &[&[u8]], pps: &[&[u8]]) -> Option<Vec<u8>> {
    let info = parse_sps(sps.first()?)?;
    if vps.is_empty() || pps.is_empty() {
        return None;
    }
    let mut body = vec![1, info.profile_byte];
    body.extend_from_slice(&info.compatibility_flags.to_be_bytes());
    body.extend_from_slice(&info.constraint_flags.to_be_bytes()[2..]);
    body.push(info.level_idc);
    body.extend_from_slice(&0xf000u16.to_be_bytes()); // min_spatial_segmentation_idc 0
    body.push(0xfc); // parallelismType 0
    body.push(0xfc | info.chroma_format_idc);
    body.push(0xf8 | info.bit_depth_luma_minus8);
    body.push(0xf8 | info.bit_depth_chroma_minus8);
    body.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate unspecified
    // constantFrameRate 0, numTemporalLayers, temporalIdNested,
    // lengthSizeMinusOne 3.
    body.push(((info.max_sub_layers & 7) << 3) | (u8::from(info.temporal_id_nesting) << 2) | 0b11);
    body.push(3);
    for (kind, units) in [(NAL_VPS, vps), (NAL_SPS, sps), (NAL_PPS, pps)] {
        body.push(0x80 | kind); // array_completeness 1
        body.extend_from_slice(&u16::try_from(units.len()).ok()?.to_be_bytes());
        for unit in units {
            body.extend_from_slice(&u16::try_from(unit.len()).ok()?.to_be_bytes());
            body.extend_from_slice(unit);
        }
    }
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(b"hvcC");
    out.extend_from_slice(&body);
    Some(out)
}

/// Collects parameter sets from NAL units, in the order first seen and
/// without exact repeats.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(super) struct ParameterSets {
    pub vps: Vec<Vec<u8>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

impl ParameterSets {
    /// Records `nal` if it is a parameter set, returning whether it was one
    /// this collection had not seen before.
    pub fn collect(&mut self, nal: &[u8]) -> bool {
        let list = match nal_type(nal) {
            NAL_VPS => &mut self.vps,
            NAL_SPS => &mut self.sps,
            NAL_PPS => &mut self.pps,
            _ => return false,
        };
        if list.iter().any(|known| known == nal) {
            return false;
        }
        list.push(nal.to_vec());
        true
    }

    /// Whether every kind of parameter set has been seen.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn is_complete(&self) -> bool {
        !self.vps.is_empty() && !self.sps.is_empty() && !self.pps.is_empty()
    }

    /// Whether every parameter set in `other` is one of these.
    pub fn contains_all(&self, other: &Self) -> bool {
        fn within(known: &[Vec<u8>], units: &[Vec<u8>]) -> bool {
            units.iter().all(|unit| known.contains(unit))
        }
        within(&self.vps, &other.vps)
            && within(&self.sps, &other.sps)
            && within(&self.pps, &other.pps)
    }

    pub fn hvcc(&self) -> Option<Vec<u8>> {
        fn refs(list: &[Vec<u8>]) -> Vec<&[u8]> {
            list.iter().map(Vec::as_slice).collect()
        }
        hvcc_box(&refs(&self.vps), &refs(&self.sps), &refs(&self.pps))
    }
}

/// One access unit from a platform encoder, reframed for an `hvc1` track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AccessUnit {
    /// The VCL and SEI NAL units, four-byte length-prefixed.
    pub data: Vec<u8>,
    /// Whether the picture is a random access point.
    pub is_irap: bool,
}

/// Splits an Annex B access unit into its in-band parameter sets, which are
/// added to `sets`, and the length-prefixed rest. `None` when the unit carries
/// no picture - an encoder that emits its parameter sets on their own, ahead
/// of the first picture, produces one of these.
#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn reframe_access_unit(stream: &[u8], sets: &mut ParameterSets) -> Option<AccessUnit> {
    reframe_units(split_annex_b(stream), sets)
}

/// Splits four-byte length-prefixed NAL units, the framing VideoToolbox
/// already emits. `None` when the data ends inside a length or a unit.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn split_length_prefixed(data: &[u8]) -> Option<Vec<&[u8]>> {
    let mut units = Vec::new();
    let mut rest = data;
    while !rest.is_empty() {
        let length = u32::from_be_bytes(rest.get(..4)?.try_into().ok()?) as usize;
        units.push(rest.get(4..)?.get(..length)?);
        rest = &rest[4 + length..];
    }
    Some(units)
}

/// [`reframe_access_unit`] for an access unit that is already four-byte
/// length-prefixed: the parameter sets and delimiters are taken out and the
/// rest re-emitted. `None` when the data is truncated or carries no picture.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn reframe_length_prefixed(data: &[u8], sets: &mut ParameterSets) -> Option<AccessUnit> {
    reframe_units(split_length_prefixed(data)?, sets)
}

fn reframe_units(units: Vec<&[u8]>, sets: &mut ParameterSets) -> Option<AccessUnit> {
    let mut is_irap = false;
    let mut has_picture = false;
    let mut picture = Vec::new();
    for unit in units {
        if is_out_of_band(unit) {
            sets.collect(unit);
        } else {
            is_irap |= self::is_irap(unit);
            has_picture |= is_vcl(unit);
            picture.push(unit);
        }
    }
    has_picture.then(|| AccessUnit {
        data: length_prefixed(picture),
        is_irap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Codec, CodecProfile, ColorRange, HardwarePreference, Limits, PixelFormat,
        VideoDecoderConfig, VideoDecoderFactory, VideoDimensions, VideoEncoderConfig,
        VideoEncoderFactory,
    };

    /// Parameter sets and the matching `hvcC` from zvidlib's own encoder.
    fn zvidlib_hvcc(width: u32, height: u32) -> Vec<u8> {
        let limits = Limits::default();
        let config = VideoEncoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(width, height, &limits).unwrap(),
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            timescale: 30,
            frame_duration: 1,
            configuration: vec![30],
        };
        crate::native_hevc_video_encoder_factory()
            .create(&config, &limits)
            .unwrap()
            .config()
            .decoder_config
            .clone()
    }

    /// NAL units listed in an `hvcC` box, by type.
    fn hvcc_units(hvcc: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let body = &hvcc[8..];
        let mut rest = &body[23..];
        let mut units = Vec::new();
        for _ in 0..body[22] {
            let kind = rest[0] & 0x3f;
            let count = u16::from_be_bytes([rest[1], rest[2]]);
            rest = &rest[3..];
            for _ in 0..count {
                let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                units.push((kind, rest[2..2 + len].to_vec()));
                rest = &rest[2 + len..];
            }
        }
        units
    }

    #[test]
    fn rebuilds_zvidlibs_hvcc_from_an_annex_b_stream() {
        let expected = zvidlib_hvcc(64, 48);
        let units = hvcc_units(&expected);
        // The parameter sets as an encoder would emit them in-band, with
        // three- and four-byte start codes and an access unit delimiter.
        let mut stream = vec![0, 0, 0, 1, NAL_AUD << 1, 1, 0x50];
        for (index, (_, unit)) in units.iter().enumerate() {
            let start: &[u8] = if index % 2 == 0 {
                &[0, 0, 1]
            } else {
                &[0, 0, 0, 1]
            };
            stream.extend_from_slice(start);
            stream.extend_from_slice(unit);
        }
        let mut sets = ParameterSets::default();
        assert!(
            reframe_access_unit(&stream, &mut sets).is_none(),
            "no picture"
        );
        assert!(sets.is_complete());
        // Repeats are ignored.
        for nal in split_annex_b(&stream) {
            assert!(!sets.collect(nal));
        }
        assert!(sets.contains_all(&sets));
        assert!(!ParameterSets::default().contains_all(&sets));
        let built = sets.hvcc().unwrap();
        assert_eq!(hvcc_units(&built), units);
        // Profile, compatibility flags, level, chroma format and bit depths
        // match. zvidlib leaves the SPS constraint flags and temporal layer
        // fields zero where this copies them from the SPS.
        assert_eq!(built[..14], expected[..14]);
        assert_eq!(built[20..27], expected[20..27]);
        assert_eq!(built[29] & 3, 3, "four-byte NAL lengths");

        let sps = &units.iter().find(|(kind, _)| *kind == NAL_SPS).unwrap().1;
        let info = parse_sps(sps).unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        assert_eq!(info.profile_byte & 0x1f, 1, "Main profile");
        assert_eq!(info.chroma_format_idc, 1);
        assert_eq!(built[14..20], info.constraint_flags.to_be_bytes()[2..]);

        // zvidlib's decoder accepts the rebuilt configuration.
        let limits = Limits::default();
        let config = VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(64, 48, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: built,
        };
        crate::native_hevc_video_decoder_factory()
            .create(&config, &limits)
            .unwrap();
    }

    #[test]
    fn reframes_an_access_unit_without_its_parameter_sets() {
        let stream = [
            0, 0, 0, 1, 0x40, 1, 7, 0, 0, 1, 0x4e, 1, 5, 0, 0, 1, 0x26, 1, 9, 9, 0, 0,
        ];
        let units = split_annex_b(&stream);
        assert_eq!(
            units,
            [&[0x40, 1, 7][..], &[0x4e, 1, 5][..], &[0x26, 1, 9, 9][..]]
        );
        assert_eq!(nal_type(units[0]), NAL_VPS);
        assert!(is_out_of_band(units[0]));
        assert!(is_irap(units[2]), "IDR_W_RADL");

        let mut sets = ParameterSets::default();
        let unit = reframe_access_unit(&stream, &mut sets).unwrap();
        // The VPS moves to the parameter sets; the SEI stays with the picture.
        assert_eq!(sets.vps, [vec![0x40, 1, 7]]);
        assert_eq!(
            unit.data,
            [0, 0, 0, 3, 0x4e, 1, 5, 0, 0, 0, 4, 0x26, 1, 9, 9]
        );
        assert!(unit.is_irap);

        // A TRAIL_R picture is not a random access point.
        let trailing = [0, 0, 1, 0x02, 1, 0xaa];
        let unit = reframe_access_unit(&trailing, &mut sets).unwrap();
        assert!(!unit.is_irap);
        assert_eq!(unit.data, [0, 0, 0, 3, 0x02, 1, 0xaa]);
    }

    #[test]
    fn reframes_a_length_prefixed_access_unit_without_its_parameter_sets() {
        // VPS, AUD, prefix SEI and an IDR slice, as VideoToolbox would frame them.
        let sample = [
            0,
            0,
            0,
            3,
            0x40,
            1,
            7,
            0,
            0,
            0,
            3,
            NAL_AUD << 1,
            1,
            0x50,
            0,
            0,
            0,
            3,
            0x4e,
            1,
            5,
            0,
            0,
            0,
            4,
            0x26,
            1,
            9,
            9,
        ];
        assert_eq!(split_length_prefixed(&sample).unwrap().len(), 4);
        let mut sets = ParameterSets::default();
        let unit = reframe_length_prefixed(&sample, &mut sets).unwrap();
        assert_eq!(sets.vps, [vec![0x40, 1, 7]]);
        assert_eq!(
            unit.data,
            [0, 0, 0, 3, 0x4e, 1, 5, 0, 0, 0, 4, 0x26, 1, 9, 9]
        );
        assert!(unit.is_irap);

        // Truncated in a length, truncated in a unit, and no picture at all.
        assert!(split_length_prefixed(&sample[..2]).is_none());
        assert!(reframe_length_prefixed(&sample[..sample.len() - 1], &mut sets).is_none());
        assert!(reframe_length_prefixed(&sample[..7], &mut sets).is_none());
    }

    #[test]
    fn removes_emulation_prevention() {
        assert_eq!(rbsp(&[1, 0, 0, 3, 1, 0, 0, 3]), [1, 0, 0, 1, 0, 0]);
    }
}
