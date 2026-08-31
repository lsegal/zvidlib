//! AV1 OBU framing (§5.3, Annex B), the non-reduced sequence header (§5.5), and the lossless
//! intra-keyframe uncompressed frame header (§5.9.2), specialised to the M0 config.
//!
//! The sequence header is non-reduced with `enable_order_hint = 1` and every other optional tool
//! disabled, matching the bounded subset [`crate::av1_inter_decoder::Av1InterDecoder`] accepts
//! (see its `check_sequence_support`), so every native-encoded stream is native-decodable without
//! sacrificing anything this backend actually emits: a still-picture reduced header carries no
//! information this encoder needs, and only the independent, spec-compliant decoder gains from
//! order hints being present.

// Adapted and modified from gamut, Copyright (c) 2026 Justin Chung, MIT licensed.
use super::bitwriter::BitWriter;
use super::leb128::write_leb128;
use crate::{Error, ErrorKind, Result};

/// `OBU_SEQUENCE_HEADER`.
const OBU_SEQUENCE_HEADER: u8 = 1;
/// `OBU_FRAME` (frame header + tile group in one OBU).
pub(crate) const OBU_FRAME: u8 = 6;
/// `order_hint_bits_minus_1 + 1`: enough range for any reasonable seek distance while every
/// frame stays an independent key frame (`refresh_frame_flags` is always implied `0xff`, so the
/// actual order-hint value never affects reference selection).
pub(crate) const ORDER_HINT_BITS: u32 = 3;

/// The sequence-header field values that `gamut-avif` must mirror into `av1C` and `colr`
/// (AV1-ISOBMFF v1.3.0 §2.3.4). Fixed by the M0 config except `seq_level_idx_0`, which depends on
/// the image size.
#[derive(Debug, Clone, Copy)]
pub struct Av1StillConfig {
    /// `seq_profile` = 0 (Main).
    pub seq_profile: u8,
    /// `seq_level_idx[0]`, the smallest level (≤ 6.0) whose limits cover the image.
    pub seq_level_idx_0: u8,
    /// `seq_tier[0]` = 0.
    pub seq_tier_0: u8,
    /// `high_bitdepth` = false (8-bit).
    pub high_bitdepth: bool,
    /// `twelve_bit` = false.
    pub twelve_bit: bool,
    /// `mono_chrome` = true.
    pub monochrome: bool,
    /// `subsampling_x` = 1 (inferred for monochrome).
    pub chroma_subsampling_x: u8,
    /// `subsampling_y` = 1 (inferred for monochrome).
    pub chroma_subsampling_y: u8,
    /// `chroma_sample_position` = 0.
    pub chroma_sample_position: u8,
    /// CICP `color_primaries` (BT.709 = 1).
    pub color_primaries: u16,
    /// CICP `transfer_characteristics` (sRGB = 13).
    pub transfer_characteristics: u16,
    /// CICP `matrix_coefficients` (Identity = 0).
    pub matrix_coefficients: u16,
    /// `color_range` (full = true).
    pub full_range: bool,
}

/// Level limits (`seq_level_idx`, MaxHSize, MaxVSize, MaxPicSize, MaxDisplayRate) through 6.0
/// (AV1 Annex A §10.3). One representative entry per spatial tier — sufficient to choose the
/// smallest covering level.
const LEVELS: [(u8, u32, u32, u64, u64); 7] = [
    (0, 2048, 1152, 147_456, 4_423_680),          // 2.0
    (1, 2816, 1584, 278_784, 8_363_520),          // 2.1
    (4, 4352, 2448, 665_856, 19_975_680),         // 3.0
    (5, 5504, 3096, 1_065_024, 31_950_720),       // 3.1
    (8, 6144, 3456, 2_359_296, 70_778_880),       // 4.0
    (12, 8192, 4352, 8_912_896, 267_386_880),     // 5.0
    (16, 16384, 8704, 35_651_584, 1_069_547_520), // 6.0
];

/// Returns the smallest `seq_level_idx` (≤ 6.0) whose limits cover `width × height`.
///
/// # Errors
///
/// [`Error::Unsupported`] if the image exceeds the level-6.0 limits (out of M0 scope).
pub fn pick_level(width: u32, height: u32, timescale: u32, frame_duration: u32) -> Result<u8> {
    let pixels = u64::from(width) * u64::from(height);
    let display_rate_numerator = u128::from(pixels) * u128::from(timescale);
    for &(idx, max_h, max_v, max_pic, max_display_rate) in &LEVELS {
        if width <= max_h
            && height <= max_v
            && pixels <= max_pic
            && display_rate_numerator <= u128::from(max_display_rate) * u128::from(frame_duration)
        {
            return Ok(idx);
        }
    }
    Err(Error::new(
        ErrorKind::Unsupported,
        "image dimensions exceed AV1 level 6.0 limits",
    ))
}

/// Number of bits needed to hold `value - 1` (the `frame_width/height_bits` field), minimum 1.
fn dimension_bits(value: u32) -> u32 {
    (32 - value.saturating_sub(1).leading_zeros()).max(1)
}

/// `tile_log2(blkSize, target)` (AV1 §5.9.16): smallest `k` with `blkSize << k >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Appends an OBU (header byte, `obu_has_size_field = 1`, LEB128 size, payload) to `out`.
pub(crate) fn write_obu(out: &mut Vec<u8>, obu_type: u8, payload: &[u8]) {
    // obu_forbidden_bit=0, obu_type, obu_extension_flag=0, obu_has_size_field=1, reserved=0.
    out.push((obu_type << 3) | 0b10);
    write_leb128(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Builds the sequence-header OBU payload for a non-reduced, profile 0, 8-bit monochrome
/// stream with `enable_order_hint = 1` and one operating point, terminated with `trailing_bits`
/// (AV1 §5.2, §5.3.4). Non-reduced (rather than a reduced still-picture header) is required so
/// [`crate::av1_inter_decoder::Av1InterDecoder`] can decode this encoder's own output natively —
/// see the module documentation.
pub(crate) fn sequence_header_payload(cfg: &Av1StillConfig, width: u32, height: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(u32::from(cfg.seq_profile), 3); // seq_profile
    w.put_bit(0); // still_picture
    w.put_bit(0); // reduced_still_picture_header
    w.put_bit(0); // timing_info_present_flag
    w.put_bit(0); // initial_display_delay_present_flag
    w.put_bits(0, 5); // operating_points_cnt_minus_1 = 0 (one operating point)
    w.put_bits(0, 12); // operating_point_idc[0]
    w.put_bits(u32::from(cfg.seq_level_idx_0), 5); // seq_level_idx[0]
    if cfg.seq_level_idx_0 > 7 {
        w.put_bits(u32::from(cfg.seq_tier_0), 1); // seq_tier[0]
    }
    // decoder_model_info_present_flag and initial_display_delay_present_flag are both false, so
    // no per-operating-point model/delay bits follow.

    let wbits = dimension_bits(width);
    let hbits = dimension_bits(height);
    w.put_bits(wbits - 1, 4); // frame_width_bits_minus_1
    w.put_bits(hbits - 1, 4); // frame_height_bits_minus_1
    w.put_bits(width - 1, wbits); // max_frame_width_minus_1
    w.put_bits(height - 1, hbits); // max_frame_height_minus_1
    w.put_bit(0); // frame_id_numbers_present_flag = 0
    w.put_bit(0); // use_128x128_superblock = 0
    w.put_bit(0); // enable_filter_intra = 0
    w.put_bit(0); // enable_intra_edge_filter = 0
    w.put_bit(0); // enable_interintra_compound = 0
    w.put_bit(0); // enable_masked_compound = 0
    w.put_bit(0); // enable_warped_motion = 0
    w.put_bit(0); // enable_dual_filter = 0
    w.put_bit(1); // enable_order_hint = 1
    w.put_bit(0); // enable_jnt_comp = 0
    w.put_bit(0); // enable_ref_frame_mvs = 0
    w.put_bit(0); // seq_choose_screen_content_tools = 0
    w.put_bit(0); // seq_force_screen_content_tools = 0
    // seq_force_screen_content_tools == 0 ⇒ seq_force_integer_mv is fixed at
    // SELECT_INTEGER_MV (2) with no bits read.
    w.put_bits(ORDER_HINT_BITS - 1, 3); // order_hint_bits_minus_1
    w.put_bit(0); // enable_superres = 0
    w.put_bit(0); // enable_cdef = 0
    w.put_bit(0); // enable_restoration = 0

    // color_config(): 8-bit Main profile monochrome with an explicit full/video range flag.
    w.put_bit(0); // high_bitdepth
    w.put_bit(u8::from(cfg.monochrome)); // mono_chrome
    w.put_bit(1); // color_description_present_flag
    w.put_bits(u32::from(cfg.color_primaries), 8);
    w.put_bits(u32::from(cfg.transfer_characteristics), 8);
    w.put_bits(u32::from(cfg.matrix_coefficients), 8);
    w.put_bit(u8::from(cfg.full_range)); // color_range

    w.put_bit(0); // film_grain_params_present

    // trailing_bits: a 1 then zero-pad to byte alignment.
    w.put_bit(1);
    w.byte_align();
    w.into_bytes()
}

/// Builds the uncompressed frame header for an intra keyframe (AV1 §5.9.2), byte-aligned
/// (`byte_alignment`, no trailing one — the tile data follows in the same `OBU_FRAME`).
///
/// `order_hint` is the frame's `order_hint` field value (taken mod `2^ORDER_HINT_BITS` by the
/// caller); every frame is an independent shown key frame, so the actual value never affects
/// decoding, but a real, incrementing value keeps the stream honest about frame order.
///
/// `base_q_idx` selects the two quantization profiles the crate implements (§5.9.12): `0` is
/// `CodedLossless`, which suppresses `loop_filter_params` and `read_tx_mode` entirely, and any
/// other value is the non-lossless profile, which signals both. The loop filter is always
/// signalled off — the encoder reconstructs without deblocking, so a nonzero level would make
/// its reconstruction disagree with the decoder's — and `tx_mode_select` is always
/// `TX_MODE_SELECT`, since the encoder chooses a transform size per coding block.
///
/// The returned bytes precede the tile (symbol-coded) data; together they form the frame OBU
/// payload (AV1 §5.10).
pub(crate) fn frame_header_payload(
    width: u32,
    height: u32,
    mi_cols: u32,
    mi_rows: u32,
    order_hint: u32,
    base_q_idx: u8,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bit(0); // show_existing_frame = 0
    w.put_bits(0, 2); // frame_type = KEY_FRAME
    w.put_bit(1); // show_frame = 1
    // error_resilient_mode is implied true for a shown key frame ⇒ no bit.
    w.put_bit(1); // disable_cdf_update = 1
    // seq_force_screen_content_tools = 0 (explicit, non-reduced) ⇒ allow_screen_content_tools is
    // implied 0 with no bit; FrameIsIntra then forces force_integer_mv = 1, also with no bit.
    w.put_bit(0); // frame_size_override_flag = 0
    w.put_bits(order_hint, ORDER_HINT_BITS); // order_hint
    // primary_ref_frame is implied PRIMARY_REF_NONE for an intra frame; refresh_frame_flags is
    // implied 0xff for a shown key frame — neither reads bits.
    // frame_size(): no override ⇒ from seq header; superres disabled. render_size():
    w.put_bit(0); // render_and_frame_size_different = 0
    // disable_frame_end_update_cdf is implied 1 because disable_cdf_update = 1.

    // tile_info(): single tile. Emit increment-stop bits only where the level/size permit > 0 tiles.
    let _ = (width, height);
    let sb_cols = (mi_cols + 15) >> 4;
    let sb_rows = (mi_rows + 15) >> 4;
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    w.put_bit(1); // uniform_tile_spacing_flag
    if max_log2_tile_cols > 0 {
        w.put_bit(0); // increment_tile_cols_log2 = 0 (stay at 1 tile column)
    }
    if max_log2_tile_rows > 0 {
        w.put_bit(0); // increment_tile_rows_log2 = 0 (stay at 1 tile row)
    }
    // TileColsLog2 == TileRowsLog2 == 0 ⇒ no context_update_tile_id / tile_size_bytes.

    // quantization_params(): base_q_idx == 0 ⇒ CodedLossless.
    w.put_bits(u32::from(base_q_idx), 8); // base_q_idx
    w.put_bit(0); // DeltaQYDc: delta_coded = 0
    w.put_bit(0); // using_qmatrix = 0

    w.put_bit(0); // segmentation_enabled = 0
    // delta_q_params / delta_lf_params: never signalled, matching the decoder's parse.
    // cdef_params / lr_params: the sequence header disables both, so neither emits bits.
    if base_q_idx != 0 {
        // loop_filter_params() (§5.9.11): both luma levels 0, which suppresses the chroma
        // levels, then sharpness and the (unused) per-reference delta switch.
        w.put_bits(0, 6); // loop_filter_level[0]
        w.put_bits(0, 6); // loop_filter_level[1]
        w.put_bits(0, 3); // loop_filter_sharpness
        w.put_bit(0); // loop_filter_delta_enabled = 0
        w.put_bit(1); // tx_mode_select = 1 (TX_MODE_SELECT)
    }
    // CodedLossless ⇒ loop_filter / tx_mode emit nothing at all.
    // frame_reference_mode / skip_mode_params (intra) ⇒ no bits. allow_warped_motion=0.
    w.put_bit(1); // reduced_tx_set = 1
    // global_motion_params / film_grain_params (intra, not present) ⇒ no bits.

    w.byte_align();
    w.into_bytes()
}

/// Wraps the sequence-header and frame OBUs into the temporal unit placed in `mdat`.
pub(crate) fn assemble_temporal_unit(seq_payload: &[u8], frame_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    write_obu(&mut out, OBU_SEQUENCE_HEADER, seq_payload);
    write_obu(&mut out, OBU_FRAME, frame_payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_selection() {
        assert_eq!(pick_level(16, 16, 30, 1).unwrap(), 0);
        // 1920x1080 = 2_073_600 px exceeds 3.x MaxPicSize, so level 4.0 (idx 8) is smallest.
        assert_eq!(pick_level(1920, 1080, 30, 1).unwrap(), 8);
        assert_eq!(pick_level(1920, 1080, 60, 1).unwrap(), 12);
        assert!(pick_level(100_000, 100_000, 30, 1).is_err());
    }

    #[test]
    fn dimension_bits_min_one() {
        assert_eq!(dimension_bits(1), 1); // W-1 = 0
        assert_eq!(dimension_bits(16), 4); // 15 -> 4 bits
        assert_eq!(dimension_bits(17), 5); // 16 -> 5 bits
        assert_eq!(dimension_bits(256), 8); // 255 -> 8 bits
        assert_eq!(dimension_bits(257), 9); // 256 -> 9 bits
    }

    #[test]
    fn obu_framing_has_size_field() {
        let mut out = Vec::new();
        write_obu(&mut out, OBU_SEQUENCE_HEADER, &[0xaa, 0xbb]);
        // header byte: type 1 -> (1<<3)|2 = 0x0a; size leb128 = 2; payload.
        assert_eq!(out, vec![0x0a, 0x02, 0xaa, 0xbb]);
    }
}
