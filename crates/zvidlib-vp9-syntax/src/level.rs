//! VP9 level selection and the `vpcC` codec configuration record.

use zvidlib_core::VideoDimensions;

/// The lowest VP9 level (as `level_idc`, e.g. 31 for 3.1) whose picture size and
/// luma sample rate limits admit the stream (VP9 Annex A).
pub fn pick_level(dimensions: VideoDimensions, timescale: u32, frame_duration: u32) -> Option<u8> {
    const LEVELS: [(u8, u64, u64); 14] = [
        (10, 36_864, 829_440),
        (11, 73_728, 2_764_800),
        (20, 122_880, 4_608_000),
        (21, 245_760, 9_216_000),
        (30, 552_960, 20_736_000),
        (31, 983_040, 36_864_000),
        (40, 2_228_224, 83_558_400),
        (41, 2_228_224, 160_432_128),
        (50, 8_912_896, 311_951_360),
        (51, 8_912_896, 588_251_136),
        (52, 8_912_896, 1_176_502_272),
        (60, 35_651_584, 1_176_502_272),
        (61, 35_651_584, 2_353_004_544),
        (62, 35_651_584, 4_706_009_088),
    ];
    if timescale == 0 || frame_duration == 0 {
        return None;
    }
    let picture = u64::from(dimensions.width) * u64::from(dimensions.height);
    let sample_rate = (picture * u64::from(timescale)).div_ceil(u64::from(frame_duration));
    LEVELS
        .iter()
        .find(|&&(_, max_picture, max_rate)| picture <= max_picture && sample_rate <= max_rate)
        .map(|&(level, _, _)| level)
}

/// The complete `vpcC` box (`VPCodecConfigurationBox`, version 1) for an
/// 8-bit 4:2:0 profile 0 stream whose bitstream colour space is `color_space`
/// (VP9 section 7.2.2).
pub fn vpcc_box(level: u8, color_space: u8, full_range: bool) -> Vec<u8> {
    // ISO/IEC 23091-2 colour primaries, transfer characteristics and matrix
    // coefficients for each VP9 colour space; unknown and reserved values
    // map to "unspecified".
    let (primaries, transfer, matrix) = match color_space {
        1 | 3 => (6, 6, 6), // BT.601, SMPTE 170M
        2 => (1, 1, 1),     // BT.709
        4 => (7, 7, 7),     // SMPTE 240M
        5 => (9, 14, 9),    // BT.2020 non-constant luminance
        7 => (1, 13, 0),    // sRGB
        _ => (2, 2, 2),
    };
    let mut output = Vec::with_capacity(20);
    output.extend_from_slice(&20_u32.to_be_bytes());
    output.extend_from_slice(b"vpcC");
    output.extend_from_slice(&[1, 0, 0, 0]); // version 1, flags 0
    output.push(0); // profile
    output.push(level);
    // bitDepth 8, chromaSubsampling 1 (4:2:0 co-located with luma), range.
    output.push((8 << 4) | (1 << 1) | u8::from(full_range));
    output.extend_from_slice(&[primaries, transfer, matrix]);
    output.extend_from_slice(&0_u16.to_be_bytes()); // codecInitializationDataSize
    output
}

/// Builds the `vpcC` for a stream from its first key frame, reading the colour
/// space and range the encoder signalled, or `None` when `frame` is not a
/// profile 0 key frame.
///
/// The browser encoder needs this because `WebCodecs` reports no decoder
/// configuration for VP9: like AV1, everything travels in band.
pub fn vpcc_from_key_frame(frame: &[u8], level: u8) -> Option<Vec<u8>> {
    let bit = |index: usize| -> Option<u8> {
        frame
            .get(index / 8)
            .map(|byte| (byte >> (7 - index % 8)) & 1)
    };
    let literal = |start: usize, bits: usize| -> Option<u32> {
        (start..start + bits).try_fold(0_u32, |value, index| {
            Some((value << 1) | u32::from(bit(index)?))
        })
    };
    // frame_marker, profile_low_bit, profile_high_bit, show_existing_frame,
    // frame_type, show_frame, error_resilient_mode, then the sync code.
    if literal(0, 2)? != 2 || literal(2, 2)? != 0 || bit(4)? != 0 || bit(5)? != 0 {
        return None;
    }
    if literal(8, 24)? != 0x49_83_42 {
        return None;
    }
    let color_space = literal(32, 3)? as u8;
    // sRGB is always full range and has no range bit; profile 0 has no
    // subsampling bits.
    let full_range = color_space == 7 || bit(35)? == 1;
    Some(vpcc_box(level, color_space, full_range))
}

/// The `vpcC` a chunk's first frame signals, or `None` when that frame is not
/// a profile 0 key frame. A hardware encoder's sample is a sync sample exactly
/// when this is `Some`, and every key frame has to signal what the track
/// declares.
pub fn key_frame_vpcc(chunk: &[u8], level: u8) -> Option<Vec<u8>> {
    let frames = crate::chunk_frames(chunk).ok()?;
    vpcc_from_key_frame(frames.first()?, level)
}
