//! Sequence header and uncompressed frame header parsing (specification
//! sections 5.5 and 5.9), with the derived frame state of section 6.8.

use super::bits::BitReader;
use super::consts::*;
use super::tables::*;
use super::{Decoder, malformed, unsupported};
use crate::Result;

#[derive(Clone, Debug, Default)]
pub(crate) struct SequenceHeader {
    pub(crate) profile: u8,
    pub(crate) reduced_still_picture_header: bool,
    pub(crate) decoder_model_info_present: bool,
    pub(crate) equal_picture_interval: bool,
    pub(crate) buffer_removal_time_length: usize,
    pub(crate) frame_presentation_time_length: usize,
    pub(crate) operating_point_idc: u32,
    pub(crate) operating_points_idc: Vec<u32>,
    pub(crate) decoder_model_present_for_op: Vec<bool>,
    pub(crate) frame_width_bits: usize,
    pub(crate) frame_height_bits: usize,
    pub(crate) max_frame_width: u32,
    pub(crate) max_frame_height: u32,
    pub(crate) frame_id_numbers_present: bool,
    pub(crate) delta_frame_id_length: usize,
    pub(crate) additional_frame_id_length: usize,
    pub(crate) use_128x128_superblock: bool,
    pub(crate) enable_filter_intra: bool,
    pub(crate) enable_intra_edge_filter: bool,
    pub(crate) enable_interintra_compound: bool,
    pub(crate) enable_masked_compound: bool,
    pub(crate) enable_warped_motion: bool,
    pub(crate) enable_dual_filter: bool,
    pub(crate) enable_order_hint: bool,
    pub(crate) enable_jnt_comp: bool,
    pub(crate) enable_ref_frame_mvs: bool,
    pub(crate) seq_force_screen_content_tools: u32,
    pub(crate) seq_force_integer_mv: u32,
    pub(crate) order_hint_bits: usize,
    pub(crate) enable_superres: bool,
    pub(crate) enable_cdef: bool,
    pub(crate) enable_restoration: bool,
    pub(crate) bit_depth: u8,
    pub(crate) mono_chrome: bool,
    pub(crate) num_planes: usize,
    pub(crate) color_primaries: u8,
    pub(crate) transfer_characteristics: u8,
    pub(crate) matrix_coefficients: u8,
    pub(crate) color_range: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) separate_uv_delta_q: bool,
    pub(crate) film_grain_params_present: bool,
}

impl SequenceHeader {
    /// `sequence_header_obu( )`.
    pub(crate) fn parse(payload: &[u8]) -> Result<Self> {
        let mut b = BitReader::new(payload);
        let mut seq = SequenceHeader {
            profile: b.f(3)? as u8,
            ..Default::default()
        };
        let _still_picture = b.flag()?;
        seq.reduced_still_picture_header = b.flag()?;
        let mut buffer_delay_length = 0;
        if seq.reduced_still_picture_header {
            seq.operating_points_idc = vec![0];
            seq.decoder_model_present_for_op = vec![false];
            let _seq_level_idx = b.f(5)?;
        } else {
            let timing_info_present = b.flag()?;
            if timing_info_present {
                let _num_units_in_display_tick = b.f(32)?;
                let _time_scale = b.f(32)?;
                seq.equal_picture_interval = b.flag()?;
                if seq.equal_picture_interval {
                    let _num_ticks_per_picture_minus_1 = b.uvlc()?;
                }
                seq.decoder_model_info_present = b.flag()?;
                if seq.decoder_model_info_present {
                    buffer_delay_length = b.f(5)? as usize + 1;
                    let _num_units_in_decoding_tick = b.f(32)?;
                    seq.buffer_removal_time_length = b.f(5)? as usize + 1;
                    seq.frame_presentation_time_length = b.f(5)? as usize + 1;
                }
            }
            let initial_display_delay_present = b.flag()?;
            let operating_points = b.f(5)? as usize + 1;
            for _ in 0..operating_points {
                seq.operating_points_idc.push(b.f(12)?);
                let seq_level_idx = b.f(5)?;
                if seq_level_idx > 7 {
                    let _seq_tier = b.f(1)?;
                }
                let mut present = false;
                if seq.decoder_model_info_present {
                    present = b.flag()?;
                    if present {
                        let _decoder_buffer_delay = b.f(buffer_delay_length)?;
                        let _encoder_buffer_delay = b.f(buffer_delay_length)?;
                        let _low_delay_mode_flag = b.f(1)?;
                    }
                }
                seq.decoder_model_present_for_op.push(present);
                if initial_display_delay_present && b.flag()? {
                    let _initial_display_delay_minus_1 = b.f(4)?;
                }
            }
        }
        // choose_operating_point( ) selects operating point 0.
        seq.operating_point_idc = seq.operating_points_idc[0];
        seq.frame_width_bits = b.f(4)? as usize + 1;
        seq.frame_height_bits = b.f(4)? as usize + 1;
        seq.max_frame_width = b.f(seq.frame_width_bits)? + 1;
        seq.max_frame_height = b.f(seq.frame_height_bits)? + 1;
        seq.frame_id_numbers_present = if seq.reduced_still_picture_header {
            false
        } else {
            b.flag()?
        };
        if seq.frame_id_numbers_present {
            seq.delta_frame_id_length = b.f(4)? as usize + 2;
            seq.additional_frame_id_length = b.f(3)? as usize + 1;
        }
        seq.use_128x128_superblock = b.flag()?;
        seq.enable_filter_intra = b.flag()?;
        seq.enable_intra_edge_filter = b.flag()?;
        if seq.reduced_still_picture_header {
            seq.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
            seq.seq_force_integer_mv = SELECT_INTEGER_MV;
        } else {
            seq.enable_interintra_compound = b.flag()?;
            seq.enable_masked_compound = b.flag()?;
            seq.enable_warped_motion = b.flag()?;
            seq.enable_dual_filter = b.flag()?;
            seq.enable_order_hint = b.flag()?;
            if seq.enable_order_hint {
                seq.enable_jnt_comp = b.flag()?;
                seq.enable_ref_frame_mvs = b.flag()?;
            }
            seq.seq_force_screen_content_tools = if b.flag()? {
                SELECT_SCREEN_CONTENT_TOOLS
            } else {
                b.f(1)?
            };
            seq.seq_force_integer_mv = if seq.seq_force_screen_content_tools > 0 {
                if b.flag()? {
                    SELECT_INTEGER_MV
                } else {
                    b.f(1)?
                }
            } else {
                SELECT_INTEGER_MV
            };
            if seq.enable_order_hint {
                seq.order_hint_bits = b.f(3)? as usize + 1;
            }
        }
        seq.enable_superres = b.flag()?;
        seq.enable_cdef = b.flag()?;
        seq.enable_restoration = b.flag()?;
        seq.parse_color_config(&mut b)?;
        seq.film_grain_params_present = b.flag()?;
        Ok(seq)
    }

    /// `color_config( )`.
    fn parse_color_config(&mut self, b: &mut BitReader<'_>) -> Result<()> {
        let high_bitdepth = b.flag()?;
        self.bit_depth = if self.profile == 2 && high_bitdepth {
            if b.flag()? { 12 } else { 10 }
        } else if high_bitdepth {
            10
        } else {
            8
        };
        self.mono_chrome = if self.profile == 1 { false } else { b.flag()? };
        self.num_planes = if self.mono_chrome { 1 } else { 3 };
        let color_description_present = b.flag()?;
        if color_description_present {
            self.color_primaries = b.f(8)? as u8;
            self.transfer_characteristics = b.f(8)? as u8;
            self.matrix_coefficients = b.f(8)? as u8;
        } else {
            self.color_primaries = 2;
            self.transfer_characteristics = 2;
            self.matrix_coefficients = 2;
        }
        if self.mono_chrome {
            self.color_range = b.flag()?;
            self.subsampling_x = true;
            self.subsampling_y = true;
            self.separate_uv_delta_q = false;
            return Ok(());
        } else if self.color_primaries == 1
            && self.transfer_characteristics == 13
            && self.matrix_coefficients == 0
        {
            self.color_range = true;
            self.subsampling_x = false;
            self.subsampling_y = false;
        } else {
            self.color_range = b.flag()?;
            if self.profile == 0 {
                self.subsampling_x = true;
                self.subsampling_y = true;
            } else if self.profile == 1 {
                self.subsampling_x = false;
                self.subsampling_y = false;
            } else if self.bit_depth == 12 {
                self.subsampling_x = b.flag()?;
                self.subsampling_y = if self.subsampling_x { b.flag()? } else { false };
            } else {
                self.subsampling_x = true;
                self.subsampling_y = false;
            }
            if self.subsampling_x && self.subsampling_y {
                let _chroma_sample_position = b.f(2)?;
            }
        }
        self.separate_uv_delta_q = b.flag()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FilmGrainParams {
    pub(crate) apply_grain: bool,
    pub(crate) grain_seed: u16,
    pub(crate) update_grain: bool,
    pub(crate) num_y_points: usize,
    pub(crate) point_y_value: [u8; 16],
    pub(crate) point_y_scaling: [u8; 16],
    pub(crate) chroma_scaling_from_luma: bool,
    pub(crate) num_cb_points: usize,
    pub(crate) point_cb_value: [u8; 16],
    pub(crate) point_cb_scaling: [u8; 16],
    pub(crate) num_cr_points: usize,
    pub(crate) point_cr_value: [u8; 16],
    pub(crate) point_cr_scaling: [u8; 16],
    pub(crate) grain_scaling_minus_8: u8,
    pub(crate) ar_coeff_lag: usize,
    pub(crate) ar_coeffs_y_plus_128: [u8; 24],
    pub(crate) ar_coeffs_cb_plus_128: [u8; 25],
    pub(crate) ar_coeffs_cr_plus_128: [u8; 25],
    pub(crate) ar_coeff_shift_minus_6: u8,
    pub(crate) grain_scale_shift: u8,
    pub(crate) cb_mult: u8,
    pub(crate) cb_luma_mult: u8,
    pub(crate) cb_offset: u16,
    pub(crate) cr_mult: u8,
    pub(crate) cr_luma_mult: u8,
    pub(crate) cr_offset: u16,
    pub(crate) overlap_flag: bool,
    pub(crate) clip_to_restricted_range: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TileInfo {
    pub(crate) tile_cols: usize,
    pub(crate) tile_rows: usize,
    pub(crate) tile_cols_log2: usize,
    pub(crate) tile_rows_log2: usize,
    pub(crate) mi_col_starts: Vec<usize>,
    pub(crate) mi_row_starts: Vec<usize>,
    pub(crate) context_update_tile_id: usize,
    pub(crate) tile_size_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LoopFilterParams {
    pub(crate) level: [u8; 4],
    pub(crate) sharpness: u8,
    pub(crate) delta_enabled: bool,
    pub(crate) ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub(crate) mode_deltas: [i8; 2],
}

impl LoopFilterParams {
    /// The deltas `setup_past_independence` establishes.
    pub(crate) const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SegmentationParams {
    pub(crate) enabled: bool,
    pub(crate) update_map: bool,
    pub(crate) temporal_update: bool,
    pub(crate) feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) seg_id_pre_skip: bool,
    pub(crate) last_active_seg_id: usize,
}

impl SegmentationParams {
    pub(crate) fn feature_active(&self, segment_id: usize, feature: usize) -> bool {
        self.enabled && self.feature_enabled[segment_id][feature]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CdefParams {
    pub(crate) damping: u8,
    pub(crate) bits: u8,
    pub(crate) y_pri_strength: [u8; 8],
    pub(crate) y_sec_strength: [u8; 8],
    pub(crate) uv_pri_strength: [u8; 8],
    pub(crate) uv_sec_strength: [u8; 8],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct FrameHeader {
    pub(crate) show_existing_frame: bool,
    pub(crate) frame_to_show_map_idx: usize,
    pub(crate) frame_type: u8,
    pub(crate) frame_is_intra: bool,
    pub(crate) show_frame: bool,
    pub(crate) showable_frame: bool,
    pub(crate) error_resilient_mode: bool,
    pub(crate) disable_cdf_update: bool,
    pub(crate) allow_screen_content_tools: bool,
    pub(crate) force_integer_mv: bool,
    pub(crate) current_frame_id: u32,
    pub(crate) order_hint: u32,
    pub(crate) primary_ref_frame: usize,
    pub(crate) refresh_frame_flags: u8,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) upscaled_width: usize,
    pub(crate) render_width: usize,
    pub(crate) render_height: usize,
    pub(crate) use_superres: bool,
    pub(crate) superres_denom: usize,
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) allow_intrabc: bool,
    pub(crate) ref_frame_idx: [usize; REFS_PER_FRAME],
    pub(crate) allow_high_precision_mv: bool,
    pub(crate) interpolation_filter: u8,
    pub(crate) is_motion_mode_switchable: bool,
    pub(crate) use_ref_frame_mvs: bool,
    pub(crate) order_hints: [u32; TOTAL_REFS_PER_FRAME],
    pub(crate) ref_frame_sign_bias: [bool; TOTAL_REFS_PER_FRAME],
    pub(crate) disable_frame_end_update_cdf: bool,
    pub(crate) tile_info: TileInfo,
    pub(crate) base_q_idx: u8,
    pub(crate) delta_q_y_dc: i32,
    pub(crate) delta_q_u_dc: i32,
    pub(crate) delta_q_u_ac: i32,
    pub(crate) delta_q_v_dc: i32,
    pub(crate) delta_q_v_ac: i32,
    pub(crate) using_qmatrix: bool,
    pub(crate) qm_y: u8,
    pub(crate) qm_u: u8,
    pub(crate) qm_v: u8,
    pub(crate) segmentation: SegmentationParams,
    pub(crate) delta_q_present: bool,
    pub(crate) delta_q_res: u32,
    pub(crate) delta_lf_present: bool,
    pub(crate) delta_lf_res: u32,
    pub(crate) delta_lf_multi: bool,
    pub(crate) coded_lossless: bool,
    pub(crate) all_lossless: bool,
    pub(crate) lossless_array: [bool; MAX_SEGMENTS],
    pub(crate) seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub(crate) loop_filter: LoopFilterParams,
    pub(crate) cdef: CdefParams,
    pub(crate) frame_restoration_type: [u8; 3],
    pub(crate) loop_restoration_size: [usize; 3],
    pub(crate) uses_lr: bool,
    pub(crate) tx_mode: u8,
    pub(crate) reference_select: bool,
    pub(crate) skip_mode_present: bool,
    pub(crate) skip_mode_frame: [i8; 2],
    pub(crate) allow_warped_motion: bool,
    pub(crate) reduced_tx_set: bool,
    pub(crate) gm_type: [u8; TOTAL_REFS_PER_FRAME],
    pub(crate) gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    pub(crate) film_grain: FilmGrainParams,
    /// The byte length of the header, including its trailing byte alignment.
    pub(crate) header_bytes: usize,
}

pub(crate) const DEFAULT_GM_PARAMS: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];

impl FrameHeader {
    pub(crate) fn get_relative_dist(&self, seq: &SequenceHeader, a: u32, b: u32) -> i32 {
        relative_dist(seq, a, b)
    }

    /// `get_qindex( ignoreDeltaQ, segmentId )` with `CurrentQIndex`.
    pub(crate) fn get_qindex(
        &self,
        ignore_delta_q: bool,
        segment_id: usize,
        current_q_index: i32,
    ) -> i32 {
        if self.segmentation.feature_active(segment_id, SEG_LVL_ALT_Q) {
            let data = i32::from(self.segmentation.feature_data[segment_id][SEG_LVL_ALT_Q]);
            let mut qindex = i32::from(self.base_q_idx) + data;
            if !ignore_delta_q && self.delta_q_present {
                qindex = current_q_index + data;
            }
            return qindex.clamp(0, 255);
        }
        if !ignore_delta_q && self.delta_q_present {
            return current_q_index;
        }
        i32::from(self.base_q_idx)
    }
}

pub(crate) fn relative_dist(seq: &SequenceHeader, a: u32, b: u32) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

fn tile_log2(blk_size: usize, target: usize) -> usize {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

fn decode_subexp(b: &mut BitReader<'_>, num_syms: i32) -> Result<i32> {
    let mut i = 0;
    let mut mk = 0;
    let k = 3;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if num_syms <= mk + 3 * a {
            return Ok(b.ns((num_syms - mk) as u32)? as i32 + mk);
        }
        if b.flag()? {
            i += 1;
            mk += a;
        } else {
            return Ok(b.f(b2 as usize)? as i32 + mk);
        }
    }
}

fn decode_signed_subexp_with_ref(
    b: &mut BitReader<'_>,
    low: i32,
    high: i32,
    r: i32,
) -> Result<i32> {
    let mx = high - low;
    let r = r - low;
    let v = decode_subexp(b, mx)?;
    let x = if (r << 1) <= mx {
        inverse_recenter(r, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - r, v)
    };
    Ok(x + low)
}

impl Decoder {
    /// `uncompressed_header( )` for the current sequence, returning the parsed
    /// header. `temporal_id`/`spatial_id` come from the OBU extension.
    pub(crate) fn parse_frame_header(
        &mut self,
        payload: &[u8],
        temporal_id: u32,
        spatial_id: u32,
    ) -> Result<FrameHeader> {
        let seq = self
            .sequence
            .clone()
            .ok_or_else(|| malformed("AV1 frame header precedes any sequence header"))?;
        let mut b = BitReader::new(payload);
        let mut fh = FrameHeader {
            primary_ref_frame: PRIMARY_REF_NONE,
            ..Default::default()
        };
        let id_len = if seq.frame_id_numbers_present {
            seq.additional_frame_id_length + seq.delta_frame_id_length
        } else {
            0
        };
        let all_frames: u8 = 0xff;
        if seq.reduced_still_picture_header {
            fh.frame_type = KEY_FRAME;
            fh.frame_is_intra = true;
            fh.show_frame = true;
            fh.showable_frame = false;
        } else {
            fh.show_existing_frame = b.flag()?;
            if fh.show_existing_frame {
                fh.frame_to_show_map_idx = b.f(3)? as usize;
                if seq.decoder_model_info_present && !seq.equal_picture_interval {
                    let _frame_presentation_time = b.f(seq.frame_presentation_time_length)?;
                }
                fh.refresh_frame_flags = 0;
                if seq.frame_id_numbers_present {
                    let _display_frame_id = b.f(id_len)?;
                }
                let slot = self.refs[fh.frame_to_show_map_idx]
                    .as_ref()
                    .ok_or_else(|| {
                        malformed("AV1 show_existing_frame names an empty reference slot")
                    })?;
                fh.frame_type = slot.frame_type;
                if fh.frame_type == KEY_FRAME {
                    fh.refresh_frame_flags = all_frames;
                }
                if seq.film_grain_params_present {
                    fh.film_grain = slot.film_grain;
                }
                b.byte_alignment()?;
                fh.header_bytes = b.position() / 8;
                return Ok(fh);
            }
            fh.frame_type = b.f(2)? as u8;
            fh.frame_is_intra = fh.frame_type == INTRA_ONLY_FRAME || fh.frame_type == KEY_FRAME;
            fh.show_frame = b.flag()?;
            if fh.show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
                let _frame_presentation_time = b.f(seq.frame_presentation_time_length)?;
            }
            fh.showable_frame = if fh.show_frame {
                fh.frame_type != KEY_FRAME
            } else {
                b.flag()?
            };
            fh.error_resilient_mode =
                if fh.frame_type == SWITCH_FRAME || (fh.frame_type == KEY_FRAME && fh.show_frame) {
                    true
                } else {
                    b.flag()?
                };
        }
        if fh.frame_type == KEY_FRAME && fh.show_frame {
            for i in 0..NUM_REF_FRAMES {
                self.ref_valid[i] = false;
                self.ref_order_hint[i] = 0;
            }
            for i in 0..REFS_PER_FRAME {
                fh.order_hints[LAST_FRAME as usize + i] = 0;
            }
        }
        fh.disable_cdf_update = b.flag()?;
        fh.allow_screen_content_tools =
            if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
                b.flag()?
            } else {
                seq.seq_force_screen_content_tools != 0
            };
        fh.force_integer_mv = if fh.allow_screen_content_tools {
            if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
                b.flag()?
            } else {
                seq.seq_force_integer_mv != 0
            }
        } else {
            false
        };
        if fh.frame_is_intra {
            fh.force_integer_mv = true;
        }
        if seq.frame_id_numbers_present {
            fh.current_frame_id = b.f(id_len)?;
            // mark_ref_frames( idLen ) only affects conformance checking.
        }
        let frame_size_override_flag = if fh.frame_type == SWITCH_FRAME {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            b.flag()?
        };
        fh.order_hint = b.f(seq.order_hint_bits)?;
        if fh.frame_is_intra || fh.error_resilient_mode {
            fh.primary_ref_frame = PRIMARY_REF_NONE;
        } else {
            fh.primary_ref_frame = b.f(3)? as usize;
        }
        if seq.decoder_model_info_present {
            let buffer_removal_time_present = b.flag()?;
            if buffer_removal_time_present {
                for op in 0..seq.operating_points_idc.len() {
                    if seq.decoder_model_present_for_op[op] {
                        let idc = seq.operating_points_idc[op];
                        let in_temporal = (idc >> temporal_id) & 1 != 0;
                        let in_spatial = (idc >> (spatial_id + 8)) & 1 != 0;
                        if idc == 0 || (in_temporal && in_spatial) {
                            let _buffer_removal_time = b.f(seq.buffer_removal_time_length)?;
                        }
                    }
                }
            }
        }
        fh.allow_high_precision_mv = false;
        fh.use_ref_frame_mvs = false;
        fh.allow_intrabc = false;
        fh.refresh_frame_flags =
            if fh.frame_type == SWITCH_FRAME || (fh.frame_type == KEY_FRAME && fh.show_frame) {
                all_frames
            } else {
                b.f(8)? as u8
            };
        if (!fh.frame_is_intra || fh.refresh_frame_flags != all_frames)
            && fh.error_resilient_mode
            && seq.enable_order_hint
        {
            for i in 0..NUM_REF_FRAMES {
                let ref_order_hint = b.f(seq.order_hint_bits)?;
                if ref_order_hint != self.ref_order_hint[i] {
                    self.ref_valid[i] = false;
                }
            }
        }
        if fh.frame_is_intra {
            self.frame_size(&seq, &mut b, &mut fh, frame_size_override_flag)?;
            self.render_size(&mut b, &mut fh)?;
            if fh.allow_screen_content_tools && fh.upscaled_width == fh.frame_width {
                fh.allow_intrabc = b.flag()?;
            }
        } else {
            let mut frame_refs_short_signaling = false;
            if seq.enable_order_hint {
                frame_refs_short_signaling = b.flag()?;
                if frame_refs_short_signaling {
                    let last_frame_idx = b.f(3)? as usize;
                    let gold_frame_idx = b.f(3)? as usize;
                    self.set_frame_refs(&seq, &mut fh, last_frame_idx, gold_frame_idx);
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !frame_refs_short_signaling {
                    fh.ref_frame_idx[i] = b.f(3)? as usize;
                }
                if seq.frame_id_numbers_present {
                    let _delta_frame_id_minus_1 = b.f(seq.delta_frame_id_length)?;
                }
            }
            for i in 0..REFS_PER_FRAME {
                if self.refs[fh.ref_frame_idx[i]].is_none() {
                    return Err(malformed(
                        "AV1 inter frame references an empty reference slot",
                    ));
                }
            }
            if frame_size_override_flag && !fh.error_resilient_mode {
                self.frame_size_with_refs(&seq, &mut b, &mut fh)?;
            } else {
                self.frame_size(&seq, &mut b, &mut fh, frame_size_override_flag)?;
                self.render_size(&mut b, &mut fh)?;
            }
            fh.allow_high_precision_mv = if fh.force_integer_mv {
                false
            } else {
                b.flag()?
            };
            let is_filter_switchable = b.flag()?;
            fh.interpolation_filter = if is_filter_switchable {
                SWITCHABLE
            } else {
                b.f(2)? as u8
            };
            fh.is_motion_mode_switchable = b.flag()?;
            fh.use_ref_frame_mvs = if fh.error_resilient_mode || !seq.enable_ref_frame_mvs {
                false
            } else {
                b.flag()?
            };
            for i in 0..REFS_PER_FRAME {
                let ref_frame = LAST_FRAME as usize + i;
                let hint = self.ref_order_hint[fh.ref_frame_idx[i]];
                fh.order_hints[ref_frame] = hint;
                fh.ref_frame_sign_bias[ref_frame] =
                    seq.enable_order_hint && relative_dist(&seq, hint, fh.order_hint) > 0;
            }
        }
        fh.disable_frame_end_update_cdf =
            if seq.reduced_still_picture_header || fh.disable_cdf_update {
                true
            } else {
                b.flag()?
            };
        let mut prev_gm_params = [DEFAULT_GM_PARAMS; TOTAL_REFS_PER_FRAME];
        if fh.primary_ref_frame == PRIMARY_REF_NONE {
            // setup_past_independence( )
            fh.loop_filter.delta_enabled = true;
            fh.loop_filter.ref_deltas = LoopFilterParams::DEFAULT_REF_DELTAS;
            fh.loop_filter.mode_deltas = [0, 0];
        } else {
            // load_previous( )
            let prev = self.refs[fh.ref_frame_idx[fh.primary_ref_frame]]
                .as_ref()
                .ok_or_else(|| malformed("AV1 primary reference frame is empty"))?;
            prev_gm_params = prev.gm_params;
            fh.loop_filter.ref_deltas = prev.loop_filter_ref_deltas;
            fh.loop_filter.mode_deltas = prev.loop_filter_mode_deltas;
            fh.segmentation.feature_enabled = prev.feature_enabled;
            fh.segmentation.feature_data = prev.feature_data;
        }
        self.tile_info(&seq, &mut b, &mut fh)?;
        self.quantization_params(&seq, &mut b, &mut fh)?;
        segmentation_params(&mut b, &mut fh)?;
        // delta_q_params( ) and delta_lf_params( )
        if fh.base_q_idx > 0 {
            fh.delta_q_present = b.flag()?;
        }
        if fh.delta_q_present {
            fh.delta_q_res = b.f(2)?;
        }
        if fh.delta_q_present {
            if !fh.allow_intrabc {
                fh.delta_lf_present = b.flag()?;
            }
            if fh.delta_lf_present {
                fh.delta_lf_res = b.f(2)?;
                fh.delta_lf_multi = b.flag()?;
            }
        }
        fh.coded_lossless = true;
        for segment_id in 0..MAX_SEGMENTS {
            let qindex = fh.get_qindex(true, segment_id, 0);
            let lossless = qindex == 0
                && fh.delta_q_y_dc == 0
                && fh.delta_q_u_ac == 0
                && fh.delta_q_u_dc == 0
                && fh.delta_q_v_ac == 0
                && fh.delta_q_v_dc == 0;
            fh.lossless_array[segment_id] = lossless;
            if !lossless {
                fh.coded_lossless = false;
            }
            if fh.using_qmatrix {
                if lossless {
                    fh.seg_qm_level[0][segment_id] = 15;
                    fh.seg_qm_level[1][segment_id] = 15;
                    fh.seg_qm_level[2][segment_id] = 15;
                } else {
                    fh.seg_qm_level[0][segment_id] = fh.qm_y;
                    fh.seg_qm_level[1][segment_id] = fh.qm_u;
                    fh.seg_qm_level[2][segment_id] = fh.qm_v;
                }
            }
        }
        fh.all_lossless = fh.coded_lossless && fh.frame_width == fh.upscaled_width;
        loop_filter_params(&seq, &mut b, &mut fh)?;
        cdef_params(&seq, &mut b, &mut fh)?;
        lr_params(&seq, &mut b, &mut fh)?;
        // read_tx_mode( )
        fh.tx_mode = if fh.coded_lossless {
            ONLY_4X4
        } else if b.flag()? {
            TX_MODE_SELECT
        } else {
            TX_MODE_LARGEST
        };
        // frame_reference_mode( )
        fh.reference_select = if fh.frame_is_intra { false } else { b.flag()? };
        self.skip_mode_params(&seq, &mut b, &mut fh)?;
        fh.allow_warped_motion =
            if fh.frame_is_intra || fh.error_resilient_mode || !seq.enable_warped_motion {
                false
            } else {
                b.flag()?
            };
        fh.reduced_tx_set = b.flag()?;
        global_motion_params(&mut b, &mut fh, &prev_gm_params)?;
        self.film_grain_params(&seq, &mut b, &mut fh)?;
        b.byte_alignment()?;
        fh.header_bytes = b.position() / 8;
        Ok(fh)
    }

    fn frame_size(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
        frame_size_override_flag: bool,
    ) -> Result<()> {
        if frame_size_override_flag {
            fh.frame_width = b.f(seq.frame_width_bits)? as usize + 1;
            fh.frame_height = b.f(seq.frame_height_bits)? as usize + 1;
        } else {
            fh.frame_width = seq.max_frame_width as usize;
            fh.frame_height = seq.max_frame_height as usize;
        }
        superres_params(seq, b, fh)?;
        self.compute_image_size(fh)
    }

    fn render_size(&self, b: &mut BitReader<'_>, fh: &mut FrameHeader) -> Result<()> {
        if b.flag()? {
            fh.render_width = b.f(16)? as usize + 1;
            fh.render_height = b.f(16)? as usize + 1;
        } else {
            fh.render_width = fh.upscaled_width;
            fh.render_height = fh.frame_height;
        }
        Ok(())
    }

    fn frame_size_with_refs(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        for i in 0..REFS_PER_FRAME {
            if b.flag()? {
                let slot = self.refs[fh.ref_frame_idx[i]]
                    .as_ref()
                    .ok_or_else(|| malformed("AV1 frame size refers to an empty slot"))?;
                fh.upscaled_width = slot.upscaled_width;
                fh.frame_width = fh.upscaled_width;
                fh.frame_height = slot.frame_height;
                fh.render_width = slot.render_width;
                fh.render_height = slot.render_height;
                superres_params(seq, b, fh)?;
                return self.compute_image_size(fh);
            }
        }
        self.frame_size(seq, b, fh, true)?;
        self.render_size(b, fh)
    }

    fn compute_image_size(&self, fh: &mut FrameHeader) -> Result<()> {
        if fh.upscaled_width as u64 > u64::from(self.limits.max_width)
            || fh.frame_height as u64 > u64::from(self.limits.max_height)
        {
            return Err(super::limit(
                "AV1 frame dimensions exceed the configured limits",
            ));
        }
        fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
        fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
        Ok(())
    }

    /// The set frame refs process (section 7.8).
    fn set_frame_refs(
        &self,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
        last_frame_idx: usize,
        gold_frame_idx: usize,
    ) {
        let mut ref_frame_idx = [-1i32; REFS_PER_FRAME];
        ref_frame_idx[0] = last_frame_idx as i32;
        ref_frame_idx[(GOLDEN_FRAME - LAST_FRAME) as usize] = gold_frame_idx as i32;
        let mut used_frame = [false; NUM_REF_FRAMES];
        used_frame[last_frame_idx] = true;
        used_frame[gold_frame_idx] = true;
        let cur_frame_hint = 1i32 << (seq.order_hint_bits - 1);
        let mut shifted = [0i32; NUM_REF_FRAMES];
        for (i, hint) in shifted.iter_mut().enumerate() {
            *hint = cur_frame_hint + relative_dist(seq, self.ref_order_hint[i], fh.order_hint);
        }
        // ALTREF_FRAME: latest backward reference.
        {
            let mut r = -1i32;
            let mut latest = 0;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                if !used_frame[i] && hint >= cur_frame_hint && (r < 0 || hint >= latest) {
                    r = i as i32;
                    latest = hint;
                }
            }
            if r >= 0 {
                ref_frame_idx[(ALTREF_FRAME - LAST_FRAME) as usize] = r;
                used_frame[r as usize] = true;
            }
        }
        let find_earliest_backward = |used_frame: &[bool; NUM_REF_FRAMES]| {
            let mut r = -1i32;
            let mut earliest = 0;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                if !used_frame[i] && hint >= cur_frame_hint && (r < 0 || hint < earliest) {
                    r = i as i32;
                    earliest = hint;
                }
            }
            r
        };
        let r = find_earliest_backward(&used_frame);
        if r >= 0 {
            ref_frame_idx[(BWDREF_FRAME - LAST_FRAME) as usize] = r;
            used_frame[r as usize] = true;
        }
        let r = find_earliest_backward(&used_frame);
        if r >= 0 {
            ref_frame_idx[(ALTREF2_FRAME - LAST_FRAME) as usize] = r;
            used_frame[r as usize] = true;
        }
        for &ref_frame in REF_FRAME_LIST.iter() {
            let slot = ref_frame as usize - LAST_FRAME as usize;
            if ref_frame_idx[slot] < 0 {
                let mut r = -1i32;
                let mut latest = 0;
                for i in 0..NUM_REF_FRAMES {
                    let hint = shifted[i];
                    if !used_frame[i] && hint < cur_frame_hint && (r < 0 || hint >= latest) {
                        r = i as i32;
                        latest = hint;
                    }
                }
                if r >= 0 {
                    ref_frame_idx[slot] = r;
                    used_frame[r as usize] = true;
                }
            }
        }
        let mut r = -1i32;
        let mut earliest = 0;
        for (i, &hint) in shifted.iter().enumerate() {
            if r < 0 || hint < earliest {
                r = i as i32;
                earliest = hint;
            }
        }
        for (i, idx) in ref_frame_idx.iter().enumerate() {
            fh.ref_frame_idx[i] = if *idx < 0 { r as usize } else { *idx as usize };
        }
    }

    fn tile_info(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        let sb_cols = if seq.use_128x128_superblock {
            (fh.mi_cols + 31) >> 5
        } else {
            (fh.mi_cols + 15) >> 4
        };
        let sb_rows = if seq.use_128x128_superblock {
            (fh.mi_rows + 31) >> 5
        } else {
            (fh.mi_rows + 15) >> 4
        };
        let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
        let ti = &mut fh.tile_info;
        let uniform = b.flag()?;
        if uniform {
            ti.tile_cols_log2 = min_log2_tile_cols;
            while ti.tile_cols_log2 < max_log2_tile_cols {
                if b.flag()? {
                    ti.tile_cols_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_width_sb = (sb_cols + (1 << ti.tile_cols_log2) - 1) >> ti.tile_cols_log2;
            let mut start_sb = 0;
            while start_sb < sb_cols {
                ti.mi_col_starts.push(start_sb << sb_shift);
                start_sb += tile_width_sb;
            }
            ti.tile_cols = ti.mi_col_starts.len();
            ti.mi_col_starts.push(fh.mi_cols);
            let min_log2_tile_rows = min_log2_tiles.saturating_sub(ti.tile_cols_log2);
            ti.tile_rows_log2 = min_log2_tile_rows;
            while ti.tile_rows_log2 < max_log2_tile_rows {
                if b.flag()? {
                    ti.tile_rows_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_height_sb = (sb_rows + (1 << ti.tile_rows_log2) - 1) >> ti.tile_rows_log2;
            let mut start_sb = 0;
            while start_sb < sb_rows {
                ti.mi_row_starts.push(start_sb << sb_shift);
                start_sb += tile_height_sb;
            }
            ti.tile_rows = ti.mi_row_starts.len();
            ti.mi_row_starts.push(fh.mi_rows);
        } else {
            let mut widest_tile_sb = 0;
            let mut start_sb = 0;
            while start_sb < sb_cols {
                ti.mi_col_starts.push(start_sb << sb_shift);
                let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
                let size_sb = b.ns(max_width as u32)? as usize + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start_sb += size_sb;
            }
            ti.tile_cols = ti.mi_col_starts.len();
            ti.mi_col_starts.push(fh.mi_cols);
            ti.tile_cols_log2 = tile_log2(1, ti.tile_cols);
            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
            let mut start_sb = 0;
            while start_sb < sb_rows {
                ti.mi_row_starts.push(start_sb << sb_shift);
                let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
                let size_sb = b.ns(max_height as u32)? as usize + 1;
                start_sb += size_sb;
            }
            ti.tile_rows = ti.mi_row_starts.len();
            ti.mi_row_starts.push(fh.mi_rows);
            ti.tile_rows_log2 = tile_log2(1, ti.tile_rows);
        }
        if ti.tile_cols > MAX_TILE_COLS || ti.tile_rows > MAX_TILE_ROWS {
            return Err(malformed(
                "AV1 tile layout exceeds the specification's tile limits",
            ));
        }
        if (ti.tile_cols * ti.tile_rows) as u64 > u64::from(self.limits.max_av1_tiles_per_frame) {
            return Err(super::limit(
                "AV1 frame has more tiles than the configured limit",
            ));
        }
        if ti.tile_cols_log2 > 0 || ti.tile_rows_log2 > 0 {
            ti.context_update_tile_id = b.f(ti.tile_rows_log2 + ti.tile_cols_log2)? as usize;
            ti.tile_size_bytes = b.f(2)? as usize + 1;
            if ti.context_update_tile_id >= ti.tile_cols * ti.tile_rows {
                return Err(malformed("AV1 context_update_tile_id names a missing tile"));
            }
        } else {
            ti.context_update_tile_id = 0;
        }
        Ok(())
    }

    fn quantization_params(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        fh.base_q_idx = b.f(8)? as u8;
        fh.delta_q_y_dc = read_delta_q(b)?;
        if seq.num_planes > 1 {
            let diff_uv_delta = if seq.separate_uv_delta_q {
                b.flag()?
            } else {
                false
            };
            fh.delta_q_u_dc = read_delta_q(b)?;
            fh.delta_q_u_ac = read_delta_q(b)?;
            if diff_uv_delta {
                fh.delta_q_v_dc = read_delta_q(b)?;
                fh.delta_q_v_ac = read_delta_q(b)?;
            } else {
                fh.delta_q_v_dc = fh.delta_q_u_dc;
                fh.delta_q_v_ac = fh.delta_q_u_ac;
            }
        }
        fh.using_qmatrix = b.flag()?;
        if fh.using_qmatrix {
            fh.qm_y = b.f(4)? as u8;
            fh.qm_u = b.f(4)? as u8;
            fh.qm_v = if !seq.separate_uv_delta_q {
                fh.qm_u
            } else {
                b.f(4)? as u8
            };
        }
        Ok(())
    }

    fn skip_mode_params(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        let mut skip_mode_allowed = false;
        if !(fh.frame_is_intra || !fh.reference_select || !seq.enable_order_hint) {
            let mut forward_idx = -1i32;
            let mut backward_idx = -1i32;
            let mut forward_hint = 0;
            let mut backward_hint = 0;
            for i in 0..REFS_PER_FRAME {
                let ref_hint = self.ref_order_hint[fh.ref_frame_idx[i]];
                if relative_dist(seq, ref_hint, fh.order_hint) < 0 {
                    if forward_idx < 0 || relative_dist(seq, ref_hint, forward_hint) > 0 {
                        forward_idx = i as i32;
                        forward_hint = ref_hint;
                    }
                } else if relative_dist(seq, ref_hint, fh.order_hint) > 0
                    && (backward_idx < 0 || relative_dist(seq, ref_hint, backward_hint) < 0)
                {
                    backward_idx = i as i32;
                    backward_hint = ref_hint;
                }
            }
            if forward_idx < 0 {
                skip_mode_allowed = false;
            } else if backward_idx >= 0 {
                skip_mode_allowed = true;
                fh.skip_mode_frame[0] = LAST_FRAME + forward_idx.min(backward_idx) as i8;
                fh.skip_mode_frame[1] = LAST_FRAME + forward_idx.max(backward_idx) as i8;
            } else {
                let mut second_forward_idx = -1i32;
                let mut second_forward_hint = 0;
                for i in 0..REFS_PER_FRAME {
                    let ref_hint = self.ref_order_hint[fh.ref_frame_idx[i]];
                    if relative_dist(seq, ref_hint, forward_hint) < 0
                        && (second_forward_idx < 0
                            || relative_dist(seq, ref_hint, second_forward_hint) > 0)
                    {
                        second_forward_idx = i as i32;
                        second_forward_hint = ref_hint;
                    }
                }
                if second_forward_idx >= 0 {
                    skip_mode_allowed = true;
                    fh.skip_mode_frame[0] = LAST_FRAME + forward_idx.min(second_forward_idx) as i8;
                    fh.skip_mode_frame[1] = LAST_FRAME + forward_idx.max(second_forward_idx) as i8;
                }
            }
        }
        fh.skip_mode_present = if skip_mode_allowed { b.flag()? } else { false };
        Ok(())
    }

    fn film_grain_params(
        &self,
        seq: &SequenceHeader,
        b: &mut BitReader<'_>,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        if !seq.film_grain_params_present || (!fh.show_frame && !fh.showable_frame) {
            fh.film_grain = FilmGrainParams::default();
            return Ok(());
        }
        let mut g = FilmGrainParams {
            apply_grain: b.flag()?,
            ..Default::default()
        };
        if !g.apply_grain {
            fh.film_grain = FilmGrainParams::default();
            return Ok(());
        }
        g.grain_seed = b.f(16)? as u16;
        g.update_grain = if fh.frame_type == INTER_FRAME {
            b.flag()?
        } else {
            true
        };
        if !g.update_grain {
            let film_grain_params_ref_idx = b.f(3)? as usize;
            let slot = self.refs[film_grain_params_ref_idx]
                .as_ref()
                .ok_or_else(|| malformed("AV1 film grain references an empty slot"))?;
            let seed = g.grain_seed;
            g = slot.film_grain;
            g.grain_seed = seed;
            fh.film_grain = g;
            return Ok(());
        }
        g.num_y_points = b.f(4)? as usize;
        if g.num_y_points > 14 {
            return Err(malformed("AV1 film grain has too many luma points"));
        }
        for i in 0..g.num_y_points {
            g.point_y_value[i] = b.f(8)? as u8;
            g.point_y_scaling[i] = b.f(8)? as u8;
        }
        g.chroma_scaling_from_luma = if seq.mono_chrome { false } else { b.flag()? };
        if seq.mono_chrome
            || g.chroma_scaling_from_luma
            || (seq.subsampling_x && seq.subsampling_y && g.num_y_points == 0)
        {
            g.num_cb_points = 0;
            g.num_cr_points = 0;
        } else {
            g.num_cb_points = b.f(4)? as usize;
            if g.num_cb_points > 10 {
                return Err(malformed("AV1 film grain has too many cb points"));
            }
            for i in 0..g.num_cb_points {
                g.point_cb_value[i] = b.f(8)? as u8;
                g.point_cb_scaling[i] = b.f(8)? as u8;
            }
            g.num_cr_points = b.f(4)? as usize;
            if g.num_cr_points > 10 {
                return Err(malformed("AV1 film grain has too many cr points"));
            }
            for i in 0..g.num_cr_points {
                g.point_cr_value[i] = b.f(8)? as u8;
                g.point_cr_scaling[i] = b.f(8)? as u8;
            }
        }
        g.grain_scaling_minus_8 = b.f(2)? as u8;
        g.ar_coeff_lag = b.f(2)? as usize;
        let num_pos_luma = 2 * g.ar_coeff_lag * (g.ar_coeff_lag + 1);
        let num_pos_chroma = if g.num_y_points > 0 {
            for i in 0..num_pos_luma {
                g.ar_coeffs_y_plus_128[i] = b.f(8)? as u8;
            }
            num_pos_luma + 1
        } else {
            num_pos_luma
        };
        if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cb_plus_128[i] = b.f(8)? as u8;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
            for i in 0..num_pos_chroma {
                g.ar_coeffs_cr_plus_128[i] = b.f(8)? as u8;
            }
        }
        g.ar_coeff_shift_minus_6 = b.f(2)? as u8;
        g.grain_scale_shift = b.f(2)? as u8;
        if g.num_cb_points > 0 {
            g.cb_mult = b.f(8)? as u8;
            g.cb_luma_mult = b.f(8)? as u8;
            g.cb_offset = b.f(9)? as u16;
        }
        if g.num_cr_points > 0 {
            g.cr_mult = b.f(8)? as u8;
            g.cr_luma_mult = b.f(8)? as u8;
            g.cr_offset = b.f(9)? as u16;
        }
        g.overlap_flag = b.flag()?;
        g.clip_to_restricted_range = b.flag()?;
        fh.film_grain = g;
        Ok(())
    }
}

fn superres_params(
    seq: &SequenceHeader,
    b: &mut BitReader<'_>,
    fh: &mut FrameHeader,
) -> Result<()> {
    fh.use_superres = if seq.enable_superres {
        b.flag()?
    } else {
        false
    };
    fh.superres_denom = if fh.use_superres {
        b.f(SUPERRES_DENOM_BITS)? as usize + SUPERRES_DENOM_MIN
    } else {
        SUPERRES_NUM
    };
    fh.upscaled_width = fh.frame_width;
    fh.frame_width = (fh.upscaled_width * SUPERRES_NUM + fh.superres_denom / 2) / fh.superres_denom;
    Ok(())
}

fn read_delta_q(b: &mut BitReader<'_>) -> Result<i32> {
    if b.flag()? { b.su(7) } else { Ok(0) }
}

fn segmentation_params(b: &mut BitReader<'_>, fh: &mut FrameHeader) -> Result<()> {
    let seg = &mut fh.segmentation;
    seg.enabled = b.flag()?;
    if seg.enabled {
        let update_data = if fh.primary_ref_frame == PRIMARY_REF_NONE {
            seg.update_map = true;
            seg.temporal_update = false;
            true
        } else {
            seg.update_map = b.flag()?;
            seg.temporal_update = if seg.update_map { b.flag()? } else { false };
            b.flag()?
        };
        if update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let feature_enabled = b.flag()?;
                    seg.feature_enabled[i][j] = feature_enabled;
                    let mut clipped = 0i32;
                    if feature_enabled {
                        let bits = usize::from(SEGMENTATION_FEATURE_BITS[j]);
                        let limit = i32::from(SEGMENTATION_FEATURE_MAX[j]);
                        if SEGMENTATION_FEATURE_SIGNED[j] == 1 {
                            clipped = b.su(1 + bits)?.clamp(-limit, limit);
                        } else {
                            clipped = (b.f(bits)? as i32).clamp(0, limit);
                        }
                    }
                    seg.feature_data[i][j] = clipped as i16;
                }
            }
        }
    } else {
        seg.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        seg.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
    }
    seg.seg_id_pre_skip = false;
    seg.last_active_seg_id = 0;
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if seg.feature_enabled[i][j] {
                seg.last_active_seg_id = i;
                if j >= SEG_LVL_REF_FRAME {
                    seg.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(())
}

fn loop_filter_params(
    seq: &SequenceHeader,
    b: &mut BitReader<'_>,
    fh: &mut FrameHeader,
) -> Result<()> {
    let lf = &mut fh.loop_filter;
    if fh.coded_lossless || fh.allow_intrabc {
        lf.level[0] = 0;
        lf.level[1] = 0;
        lf.ref_deltas = LoopFilterParams::DEFAULT_REF_DELTAS;
        lf.mode_deltas = [0, 0];
        return Ok(());
    }
    lf.level[0] = b.f(6)? as u8;
    lf.level[1] = b.f(6)? as u8;
    if seq.num_planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = b.f(6)? as u8;
        lf.level[3] = b.f(6)? as u8;
    }
    lf.sharpness = b.f(3)? as u8;
    lf.delta_enabled = b.flag()?;
    if lf.delta_enabled {
        let delta_update = b.flag()?;
        if delta_update {
            for i in 0..TOTAL_REFS_PER_FRAME {
                if b.flag()? {
                    lf.ref_deltas[i] = b.su(7)? as i8;
                }
            }
            for i in 0..2 {
                if b.flag()? {
                    lf.mode_deltas[i] = b.su(7)? as i8;
                }
            }
        }
    }
    Ok(())
}

fn cdef_params(seq: &SequenceHeader, b: &mut BitReader<'_>, fh: &mut FrameHeader) -> Result<()> {
    let cdef = &mut fh.cdef;
    if fh.coded_lossless || fh.allow_intrabc || !seq.enable_cdef {
        cdef.bits = 0;
        cdef.y_pri_strength[0] = 0;
        cdef.y_sec_strength[0] = 0;
        cdef.uv_pri_strength[0] = 0;
        cdef.uv_sec_strength[0] = 0;
        cdef.damping = 3;
        return Ok(());
    }
    cdef.damping = b.f(2)? as u8 + 3;
    cdef.bits = b.f(2)? as u8;
    for i in 0..(1usize << cdef.bits) {
        cdef.y_pri_strength[i] = b.f(4)? as u8;
        cdef.y_sec_strength[i] = b.f(2)? as u8;
        if cdef.y_sec_strength[i] == 3 {
            cdef.y_sec_strength[i] += 1;
        }
        if seq.num_planes > 1 {
            cdef.uv_pri_strength[i] = b.f(4)? as u8;
            cdef.uv_sec_strength[i] = b.f(2)? as u8;
            if cdef.uv_sec_strength[i] == 3 {
                cdef.uv_sec_strength[i] += 1;
            }
        }
    }
    Ok(())
}

fn lr_params(seq: &SequenceHeader, b: &mut BitReader<'_>, fh: &mut FrameHeader) -> Result<()> {
    if fh.all_lossless || fh.allow_intrabc || !seq.enable_restoration {
        fh.frame_restoration_type = [RESTORE_NONE; 3];
        fh.uses_lr = false;
        return Ok(());
    }
    fh.uses_lr = false;
    let mut uses_chroma_lr = false;
    for i in 0..seq.num_planes {
        let lr_type = b.f(2)? as usize;
        fh.frame_restoration_type[i] = REMAP_LR_TYPE[lr_type];
        if fh.frame_restoration_type[i] != RESTORE_NONE {
            fh.uses_lr = true;
            if i > 0 {
                uses_chroma_lr = true;
            }
        }
    }
    if fh.uses_lr {
        let mut lr_unit_shift = b.f(1)? as usize;
        if seq.use_128x128_superblock {
            lr_unit_shift += 1;
        } else if lr_unit_shift != 0 {
            lr_unit_shift += b.f(1)? as usize;
        }
        fh.loop_restoration_size[0] = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);
        let lr_uv_shift = if seq.subsampling_x && seq.subsampling_y && uses_chroma_lr {
            b.f(1)? as usize
        } else {
            0
        };
        fh.loop_restoration_size[1] = fh.loop_restoration_size[0] >> lr_uv_shift;
        fh.loop_restoration_size[2] = fh.loop_restoration_size[0] >> lr_uv_shift;
    }
    Ok(())
}

fn global_motion_params(
    b: &mut BitReader<'_>,
    fh: &mut FrameHeader,
    prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
) -> Result<()> {
    for r in LAST_FRAME as usize..=ALTREF_FRAME as usize {
        fh.gm_type[r] = IDENTITY;
        fh.gm_params[r] = DEFAULT_GM_PARAMS;
    }
    if fh.frame_is_intra {
        return Ok(());
    }
    for r in LAST_FRAME as usize..=ALTREF_FRAME as usize {
        let typ = if b.flag()? {
            if b.flag()? {
                ROTZOOM
            } else if b.flag()? {
                TRANSLATION
            } else {
                AFFINE
            }
        } else {
            IDENTITY
        };
        fh.gm_type[r] = typ;
        if typ >= ROTZOOM {
            read_global_param(b, fh, prev_gm_params, typ, r, 2)?;
            read_global_param(b, fh, prev_gm_params, typ, r, 3)?;
            if typ == AFFINE {
                read_global_param(b, fh, prev_gm_params, typ, r, 4)?;
                read_global_param(b, fh, prev_gm_params, typ, r, 5)?;
            } else {
                fh.gm_params[r][4] = -fh.gm_params[r][3];
                fh.gm_params[r][5] = fh.gm_params[r][2];
            }
        }
        if typ >= TRANSLATION {
            read_global_param(b, fh, prev_gm_params, typ, r, 0)?;
            read_global_param(b, fh, prev_gm_params, typ, r, 1)?;
        }
    }
    Ok(())
}

fn read_global_param(
    b: &mut BitReader<'_>,
    fh: &mut FrameHeader,
    prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
    typ: u8,
    r: usize,
    idx: usize,
) -> Result<()> {
    let mut abs_bits = GM_ABS_ALPHA_BITS;
    let mut prec_bits = GM_ALPHA_PREC_BITS;
    if idx < 2 {
        if typ == TRANSLATION {
            let hp = i32::from(!fh.allow_high_precision_mv);
            abs_bits = GM_ABS_TRANS_ONLY_BITS - hp;
            prec_bits = GM_TRANS_ONLY_PREC_BITS - hp;
        } else {
            abs_bits = GM_ABS_TRANS_BITS;
            prec_bits = GM_TRANS_PREC_BITS;
        }
    }
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let round = if idx % 3 == 2 {
        1 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    let sub = if idx % 3 == 2 { 1 << prec_bits } else { 0 };
    let mx = 1 << abs_bits;
    let r_value = (prev_gm_params[r][idx] >> prec_diff) - sub;
    fh.gm_params[r][idx] =
        (decode_signed_subexp_with_ref(b, -mx, mx + 1, r_value)? << prec_diff) + round;
    Ok(())
}

/// Rejects bitstreams outside the 8-bit 4:2:0/monochrome Main profile this
/// decoder implements.
pub(crate) fn check_supported(seq: &SequenceHeader) -> Result<()> {
    if seq.profile != 0 {
        return Err(unsupported(
            "native AV1 decoder supports the Main profile only",
        ));
    }
    if seq.bit_depth != 8 {
        return Err(unsupported(
            "native AV1 decoder supports 8-bit streams only",
        ));
    }
    Ok(())
}
