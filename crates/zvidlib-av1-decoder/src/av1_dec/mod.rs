//! A dependency-free AV1 Main-profile decoder following the decoding process
//! of the AV1 Bitstream & Decoding Process Specification (with its errata)
//! for 8-bit 4:2:0 and monochrome streams.
//!
//! The decoder is organised as the specification is: `header` parses the
//! sequence and frame headers (sections 5.5 and 5.9), `tile`, `residual` and
//! `mvpred` implement the tile syntax and motion vector prediction (sections
//! 5.11 and 7.10), `predict` and `recon` the prediction and reconstruction
//! processes (sections 7.11 to 7.13), `postfilter` the loop filter, CDEF,
//! super-resolution and loop restoration (sections 7.14 to 7.17), and `grain`
//! film grain synthesis (section 7.18.3). The lookup tables in `tables` and
//! `qm` are machine-generated from the specification's own tables.
//!
//! Every tool of the Main profile is implemented: all intra modes including
//! palette, intra block copy, filter intra and chroma-from-luma; single,
//! compound, distance-weighted, wedge, difference-weighted and inter-intra
//! prediction; OBMC and local and global warped motion; reference frame
//! motion vectors; variable transform sizes with every transform type;
//! quantizer matrices, segmentation and delta quantizer/loop filter; tiles;
//! CDF adaptation; and every in-loop filter. Malformed input returns
//! [`ErrorKind::MalformedMedia`] rather than panicking.

// The processes below follow the specification's pseudocode, which indexes
// several parallel arrays with one loop variable; keeping that shape makes
// each function checkable line by line against the section it implements.
#![allow(clippy::needless_range_loop)]

mod bits;
mod cdf;
mod consts;
mod coverage;
mod frame;
mod grain;
mod header;
mod mvpred;
mod postfilter;
mod predict;
mod qm;
mod recon;
mod residual;
mod symbol;
mod tables;
mod tile;

use std::sync::Arc;

use bits::BitReader;
use cdf::CdfContext;
use consts::*;
use frame::{FrameBuf, FrameState, LrPlane, ModeInfo};
use header::{FilmGrainParams, FrameHeader, SequenceHeader, relative_dist};
use symbol::SymbolDecoder;
use tile::TileDecoder;

use crate::{Error, ErrorKind, Limits, Result};

pub(crate) fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

pub(crate) fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

pub(crate) fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_TILE_GROUP: u8 = 4;
const OBU_FRAME: u8 = 6;
const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
const OBU_TILE_LIST: u8 = 8;

/// A reference frame slot: the saved state of the reference frame update
/// process (section 7.20).
pub(crate) struct RefSlot {
    pub(crate) upscaled_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) render_width: usize,
    pub(crate) render_height: usize,
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) frame_type: u8,
    pub(crate) order_hint: u32,
    pub(crate) saved_order_hints: [u32; TOTAL_REFS_PER_FRAME],
    pub(crate) frame: Arc<FrameBuf>,
    pub(crate) saved_ref_frames: Arc<Vec<i8>>,
    pub(crate) saved_mvs: Arc<Vec<[i32; 2]>>,
    pub(crate) gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    pub(crate) saved_segment_ids: Arc<Vec<u8>>,
    pub(crate) cdfs: Arc<CdfContext>,
    pub(crate) film_grain: FilmGrainParams,
    pub(crate) loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub(crate) loop_filter_mode_deltas: [i8; 2],
    pub(crate) feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// A decoded picture as the output process produces it: `OutY`, `OutU`
/// and `OutV`, tightly packed.
#[derive(Clone, Debug)]
pub(crate) struct DecodedPicture {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Luma, then (unless monochrome) the two chroma planes.
    pub(crate) planes: Vec<Vec<u8>>,
    pub(crate) monochrome: bool,
    pub(crate) subsampling_x: bool,
    pub(crate) subsampling_y: bool,
    pub(crate) full_range: bool,
    pub(crate) matrix_coefficients: u8,
}

struct FrameInProgress {
    fh: FrameHeader,
    state: FrameState,
    cdf: Box<CdfContext>,
    saved_cdf: Option<Box<CdfContext>>,
    next_tile: usize,
}

/// Records the frame-level tools `fh` uses (see `coverage`).
fn note_frame_tools(seq: &SequenceHeader, fh: &FrameHeader) {
    let flags = [
        (fh.use_superres, coverage::SUPERRES),
        (fh.using_qmatrix, coverage::QUANTIZER_MATRIX),
        (
            fh.tile_info.tile_cols * fh.tile_info.tile_rows > 1,
            coverage::MULTIPLE_TILES,
        ),
        (
            fh.tile_info.context_update_tile_id != 0,
            coverage::CONTEXT_UPDATE_TILE_ID,
        ),
        (seq.use_128x128_superblock, coverage::SUPERBLOCK_128),
        (fh.segmentation.enabled, coverage::SEGMENTATION),
        (fh.delta_q_present, coverage::DELTA_Q),
        (fh.delta_lf_present, coverage::DELTA_LF),
    ];
    for (used, tool) in flags {
        if used {
            coverage::note(tool);
        }
    }
}

/// The stateful decoder.
pub(crate) struct Decoder {
    limits: Limits,
    sequence: Option<SequenceHeader>,
    refs: [Option<Arc<RefSlot>>; NUM_REF_FRAMES],
    ref_valid: [bool; NUM_REF_FRAMES],
    ref_order_hint: [u32; NUM_REF_FRAMES],
    frame: Option<FrameInProgress>,
    output_wanted: bool,
    frames_shown: u64,
}

impl Decoder {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            limits,
            sequence: None,
            refs: Default::default(),
            ref_valid: [false; NUM_REF_FRAMES],
            ref_order_hint: [0; NUM_REF_FRAMES],
            frame: None,
            output_wanted: true,
            frames_shown: 0,
        }
    }

    /// Drops every retained reference and partially decoded frame.
    pub(crate) fn reset(&mut self) {
        self.refs = Default::default();
        self.ref_valid = [false; NUM_REF_FRAMES];
        self.ref_order_hint = [0; NUM_REF_FRAMES];
        self.frame = None;
    }

    /// Whether pictures are converted for output; frames are decoded (and
    /// kept as references) either way.
    pub(crate) fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
    }

    /// How many frames the decoder has shown, whether or not they were
    /// converted for output.
    pub(crate) fn frames_shown(&self) -> u64 {
        self.frames_shown
    }

    /// Decodes one temporal unit (a low-overhead OBU sequence), returning the
    /// pictures it shows.
    pub(crate) fn decode_temporal_unit(&mut self, data: &[u8]) -> Result<Vec<DecodedPicture>> {
        let mut output = Vec::new();
        let mut offset = 0usize;
        let mut obus = 0u32;
        while offset < data.len() {
            obus += 1;
            if obus > self.limits.max_av1_obus_per_unit {
                return Err(limit(
                    "AV1 temporal unit has more OBUs than the configured limit",
                ));
            }
            let header = data[offset];
            if header & 0x80 != 0 {
                return Err(malformed("AV1 OBU forbidden bit is set"));
            }
            let obu_type = (header >> 3) & 0xf;
            let extension_flag = header & 0x04 != 0;
            let has_size_field = header & 0x02 != 0;
            let mut pos = offset + 1;
            let (mut temporal_id, mut spatial_id) = (0u32, 0u32);
            if extension_flag {
                let ext = *data
                    .get(pos)
                    .ok_or_else(|| malformed("AV1 OBU extension is truncated"))?;
                temporal_id = u32::from(ext >> 5);
                spatial_id = u32::from((ext >> 3) & 3);
                pos += 1;
            }
            let obu_size = if has_size_field {
                let mut reader = BitReader::new(&data[pos..]);
                let size = reader.leb128()?;
                pos += reader.position() / 8;
                usize::try_from(size).map_err(|_| malformed("AV1 OBU size overflows"))?
            } else {
                data.len() - pos
            };
            let end = pos
                .checked_add(obu_size)
                .filter(|&end| end <= data.len())
                .ok_or_else(|| malformed("AV1 OBU extends past its temporal unit"))?;
            let payload = &data[pos..end];
            offset = end;
            if obu_type != OBU_SEQUENCE_HEADER
                && obu_type != OBU_TEMPORAL_DELIMITER
                && extension_flag
            {
                if let Some(seq) = &self.sequence {
                    let idc = seq.operating_point_idc;
                    if idc != 0 {
                        let in_temporal = (idc >> temporal_id) & 1 != 0;
                        let in_spatial = (idc >> (spatial_id + 8)) & 1 != 0;
                        if !in_temporal || !in_spatial {
                            continue;
                        }
                    }
                }
            }
            match obu_type {
                OBU_SEQUENCE_HEADER => {
                    let seq = SequenceHeader::parse(payload)?;
                    header::check_supported(&seq)?;
                    self.sequence = Some(seq);
                }
                OBU_TEMPORAL_DELIMITER => {}
                OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER => {
                    if self.frame.is_some() {
                        // frame_header_copy( ): identical to the header in use.
                        continue;
                    }
                    let fh = self.parse_frame_header(payload, temporal_id, spatial_id)?;
                    if fh.show_existing_frame {
                        if let Some(picture) = self.show_existing_frame(&fh)? {
                            output.push(picture);
                        }
                    } else {
                        self.start_frame(fh)?;
                    }
                }
                OBU_FRAME => {
                    if self.frame.is_some() {
                        return Err(malformed(
                            "AV1 frame OBU arrives before the previous frame completed",
                        ));
                    }
                    let fh = self.parse_frame_header(payload, temporal_id, spatial_id)?;
                    if fh.show_existing_frame {
                        return Err(malformed("AV1 frame OBU cannot show an existing frame"));
                    }
                    let header_bytes = fh.header_bytes;
                    self.start_frame(fh)?;
                    if let Some(picture) = self.tile_group(&payload[header_bytes..])? {
                        output.push(picture);
                    }
                }
                OBU_TILE_GROUP => {
                    if self.frame.is_none() {
                        return Err(malformed("AV1 tile group has no frame header"));
                    }
                    if let Some(picture) = self.tile_group(payload)? {
                        output.push(picture);
                    }
                }
                OBU_TILE_LIST => {
                    return Err(unsupported(
                        "AV1 large scale tile decoding is not supported",
                    ));
                }
                _ => {}
            }
        }
        Ok(output)
    }

    fn start_frame(&mut self, fh: FrameHeader) -> Result<()> {
        let seq = self
            .sequence
            .as_ref()
            .expect("frame header parsing requires a sequence header");
        let mi_blocks = (fh.mi_rows as u64) * (fh.mi_cols as u64);
        if mi_blocks > u64::from(self.limits.max_av1_blocks_per_frame) {
            return Err(limit(
                "AV1 frame has more 4x4 blocks than the configured limit",
            ));
        }
        let sb = 128;
        let width = (fh.mi_cols * MI_SIZE).div_ceil(sb) * sb;
        let height = (fh.mi_rows * MI_SIZE).div_ceil(sb) * sb;
        let upscaled = fh.upscaled_width.div_ceil(sb) * sb;
        let curr = FrameBuf::new(
            width.max(upscaled),
            height,
            seq.num_planes,
            seq.subsampling_x,
            seq.subsampling_y,
            &self.limits,
        )?;
        note_frame_tools(seq, &fh);
        let mi = ModeInfo::new(fh.mi_rows, fh.mi_cols);
        let mut prev_segment_ids = vec![0u8; fh.mi_rows * fh.mi_cols];
        let cdf = if fh.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::new(fh.base_q_idx)
        } else {
            let slot = self.refs[fh.ref_frame_idx[fh.primary_ref_frame]]
                .as_ref()
                .ok_or_else(|| malformed("AV1 primary reference frame is empty"))?;
            let mut cdf = Box::new((*slot.cdfs).clone());
            cdf.clear_counts();
            // load_previous_segment_ids( )
            if fh.segmentation.enabled && slot.mi_cols == fh.mi_cols && slot.mi_rows == fh.mi_rows {
                prev_segment_ids.copy_from_slice(&slot.saved_segment_ids);
            }
            cdf
        };
        let cdef_stride = fh.mi_cols.div_ceil(16) + 1;
        let cdef_rows = fh.mi_rows.div_ceil(16) + 1;
        let mut lr: [LrPlane; 3] = Default::default();
        for (plane, lr_plane) in lr.iter_mut().enumerate().take(seq.num_planes) {
            if fh.frame_restoration_type[plane] != RESTORE_NONE {
                let (sx, sy) = if plane > 0 {
                    (
                        usize::from(seq.subsampling_x),
                        usize::from(seq.subsampling_y),
                    )
                } else {
                    (0, 0)
                };
                let (rows, cols) = postfilter::lr_unit_counts(&fh, plane, sx, sy);
                let n = rows * cols;
                *lr_plane = LrPlane {
                    unit_rows: rows,
                    unit_cols: cols,
                    lr_type: vec![RESTORE_NONE; n],
                    wiener: vec![[[0; 3]; 2]; n],
                    sgr_set: vec![0; n],
                    sgr_xqd: vec![[0; 2]; n],
                };
            }
        }
        let chroma_rows = (fh.mi_rows + 1) >> usize::from(seq.subsampling_y);
        let chroma_cols = (fh.mi_cols + 1) >> usize::from(seq.subsampling_x);
        let lf_tx_sizes = [
            vec![0u8; fh.mi_rows * fh.mi_cols],
            vec![0u8; chroma_rows * chroma_cols],
            vec![0u8; chroma_rows * chroma_cols],
        ];
        let lf_tx_stride = [fh.mi_cols, chroma_cols, chroma_cols];
        let motion_field_mvs = if fh.use_ref_frame_mvs {
            self.motion_field_estimation(seq, &fh)
        } else {
            Vec::new()
        };
        let state = FrameState {
            mi,
            curr,
            prev_segment_ids,
            cdef_idx: vec![-1; cdef_stride * cdef_rows],
            cdef_stride,
            lr,
            lf_tx_sizes,
            lf_tx_stride,
            motion_field_mvs,
        };
        self.frame = Some(FrameInProgress {
            fh,
            state,
            cdf,
            saved_cdf: None,
            next_tile: 0,
        });
        Ok(())
    }

    /// `tile_group_obu( sz )`; returns a picture once the frame completes and
    /// is shown.
    fn tile_group(&mut self, data: &[u8]) -> Result<Option<DecodedPicture>> {
        let seq = self
            .sequence
            .clone()
            .expect("a frame in progress has a sequence header");
        let mut frame = self
            .frame
            .take()
            .ok_or_else(|| malformed("AV1 tile group has no frame header"))?;
        let ti = frame.fh.tile_info.clone();
        let num_tiles = ti.tile_cols * ti.tile_rows;
        let mut b = BitReader::new(data);
        let mut tile_start_and_end_present = false;
        if num_tiles > 1 {
            tile_start_and_end_present = b.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present {
            (0, num_tiles - 1)
        } else {
            let tile_bits = ti.tile_cols_log2 + ti.tile_rows_log2;
            (b.f(tile_bits)? as usize, b.f(tile_bits)? as usize)
        };
        b.byte_alignment()?;
        if tg_start != frame.next_tile || tg_end < tg_start || tg_end >= num_tiles {
            return Err(malformed(
                "AV1 tile group does not continue the frame's tiles",
            ));
        }
        let mut pos = b.position() / 8;
        for tile_num in tg_start..=tg_end {
            let tile_row = tile_num / ti.tile_cols;
            let tile_col = tile_num % ti.tile_cols;
            let last_tile = tile_num == tg_end;
            let tile_size = if last_tile {
                data.len()
                    .checked_sub(pos)
                    .ok_or_else(|| malformed("AV1 tile group is truncated"))?
            } else {
                let mut r = BitReader::new(
                    data.get(pos..)
                        .ok_or_else(|| malformed("AV1 tile size is truncated"))?,
                );
                let size = r.le(ti.tile_size_bytes)? as usize + 1;
                pos += ti.tile_size_bytes;
                size
            };
            let tile_data = data
                .get(pos..pos + tile_size)
                .ok_or_else(|| malformed("AV1 tile data is truncated"))?;
            pos += tile_size;
            let mut tile_cdf = frame.cdf.clone();
            {
                let sd = SymbolDecoder::new(tile_data, frame.fh.disable_cdf_update)?;
                let mut td = TileDecoder::new(
                    &seq,
                    &frame.fh,
                    &self.refs,
                    &mut frame.state,
                    &mut tile_cdf,
                    sd,
                    tile_row,
                    tile_col,
                );
                td.decode_tile()?;
            }
            if !frame.fh.disable_frame_end_update_cdf && tile_num == ti.context_update_tile_id {
                frame.saved_cdf = Some(tile_cdf);
            }
        }
        frame.next_tile = tg_end + 1;
        if tg_end != num_tiles - 1 {
            coverage::note(coverage::MULTIPLE_TILE_GROUPS);
            self.frame = Some(frame);
            return Ok(None);
        }
        if !frame.fh.disable_frame_end_update_cdf {
            if let Some(saved) = frame.saved_cdf.take() {
                frame.cdf = saved;
            }
        }
        self.decode_frame_wrapup(&seq, frame)
    }

    /// The decode frame wrapup process (section 7.4) for a decoded frame.
    fn decode_frame_wrapup(
        &mut self,
        seq: &SequenceHeader,
        mut frame: FrameInProgress,
    ) -> Result<Option<DecodedPicture>> {
        let fh = &frame.fh;
        if fh.loop_filter.level[0] != 0 || fh.loop_filter.level[1] != 0 {
            postfilter::loop_filter(seq, fh, &mut frame.state);
        }
        let cdef_enabled = seq.enable_cdef && !fh.coded_lossless && !fh.allow_intrabc;
        let cdef_frame = if cdef_enabled {
            Some(postfilter::cdef(seq, fh, &frame.state))
        } else {
            None
        };
        let alloc = |_plane: usize, w: usize, h: usize| frame::PlaneBuf::new(w, h, 0);
        let lr_frame = if fh.use_superres || fh.uses_lr {
            let upscaled_cdef = postfilter::upscale(
                seq,
                fh,
                cdef_frame.as_ref().unwrap_or(&frame.state.curr),
                alloc,
            );
            let upscaled_curr = if cdef_frame.is_some() {
                postfilter::upscale(seq, fh, &frame.state.curr, alloc)
            } else {
                upscaled_cdef.clone()
            };
            postfilter::loop_restoration(seq, fh, &frame.state, &upscaled_curr, &upscaled_cdef)
        } else {
            match cdef_frame {
                Some(cdef) => cdef,
                None => frame.state.curr.clone(),
            }
        };
        // Motion field motion vector storage (section 7.19).
        let n = fh.mi_rows * fh.mi_cols;
        let mut mf_ref_frames = vec![NONE; n];
        let mut mf_mvs = vec![[0i32; 2]; n];
        for i in 0..n {
            for list in 0..2 {
                let r = frame.state.mi.ref_frames[i][list];
                if r > INTRA_FRAME {
                    let ref_idx = fh.ref_frame_idx[(r - LAST_FRAME) as usize];
                    let dist = relative_dist(seq, self.ref_order_hint[ref_idx], fh.order_hint);
                    if dist < 0 {
                        let mv = frame.state.mi.mvs[i][list];
                        if mv[0].abs() <= REFMVS_LIMIT && mv[1].abs() <= REFMVS_LIMIT {
                            mf_ref_frames[i] = r;
                            mf_mvs[i] = mv;
                        }
                    }
                }
            }
        }
        let segment_ids = if fh.segmentation.enabled && !fh.segmentation.update_map {
            std::mem::take(&mut frame.state.prev_segment_ids)
        } else {
            std::mem::take(&mut frame.state.mi.segment_ids)
        };
        let lr_frame = Arc::new(lr_frame);
        let slot = Arc::new(RefSlot {
            upscaled_width: fh.upscaled_width,
            frame_height: fh.frame_height,
            render_width: fh.render_width,
            render_height: fh.render_height,
            mi_cols: fh.mi_cols,
            mi_rows: fh.mi_rows,
            frame_type: fh.frame_type,
            order_hint: fh.order_hint,
            saved_order_hints: fh.order_hints,
            frame: Arc::clone(&lr_frame),
            saved_ref_frames: Arc::new(mf_ref_frames),
            saved_mvs: Arc::new(mf_mvs),
            gm_params: fh.gm_params,
            saved_segment_ids: Arc::new(segment_ids),
            cdfs: Arc::new(*frame.cdf),
            film_grain: fh.film_grain,
            loop_filter_ref_deltas: fh.loop_filter.ref_deltas,
            loop_filter_mode_deltas: fh.loop_filter.mode_deltas,
            feature_enabled: fh.segmentation.feature_enabled,
            feature_data: fh.segmentation.feature_data,
        });
        // Reference frame update (section 7.20).
        for i in 0..NUM_REF_FRAMES {
            if (fh.refresh_frame_flags >> i) & 1 == 1 {
                self.ref_valid[i] = true;
                self.ref_order_hint[i] = fh.order_hint;
                self.refs[i] = Some(Arc::clone(&slot));
            }
        }
        if fh.show_frame {
            self.frames_shown += 1;
        }
        if fh.show_frame && self.output_wanted {
            Ok(Some(self.output_picture(seq, &slot, &fh.film_grain)))
        } else {
            Ok(None)
        }
    }

    fn show_existing_frame(&mut self, fh: &FrameHeader) -> Result<Option<DecodedPicture>> {
        let seq = self
            .sequence
            .clone()
            .expect("frame header parsing requires a sequence header");
        let slot = self.refs[fh.frame_to_show_map_idx]
            .clone()
            .ok_or_else(|| malformed("AV1 show_existing_frame names an empty reference slot"))?;
        if fh.frame_type == KEY_FRAME {
            // The reference frame loading process followed by the update of
            // every slot (refresh_frame_flags is allFrames).
            for i in 0..NUM_REF_FRAMES {
                self.ref_valid[i] = true;
                self.ref_order_hint[i] = slot.order_hint;
                self.refs[i] = Some(Arc::clone(&slot));
            }
        }
        self.frames_shown += 1;
        if !self.output_wanted {
            return Ok(None);
        }
        Ok(Some(self.output_picture(&seq, &slot, &fh.film_grain)))
    }

    /// The output process (section 7.18) for a stored frame.
    fn output_picture(
        &self,
        seq: &SequenceHeader,
        slot: &RefSlot,
        grain: &FilmGrainParams,
    ) -> DecodedPicture {
        let w = slot.upscaled_width;
        let h = slot.frame_height;
        let sub_x = usize::from(seq.subsampling_x);
        let sub_y = usize::from(seq.subsampling_y);
        let mut planes: Vec<Vec<u16>> = Vec::with_capacity(seq.num_planes);
        for plane in 0..seq.num_planes {
            let (pw, ph) = if plane == 0 {
                (w, h)
            } else {
                ((w + sub_x) >> sub_x, (h + sub_y) >> sub_y)
            };
            let src = &slot.frame.planes[plane];
            let mut out = Vec::with_capacity(pw * ph);
            for y in 0..ph {
                out.extend_from_slice(&src.row(y)[..pw]);
            }
            planes.push(out);
        }
        if seq.film_grain_params_present && grain.apply_grain {
            coverage::note(coverage::FILM_GRAIN);
            let mut out = grain::OutPlanes {
                planes: &mut planes,
                width: w,
                height: h,
                sub_x,
                sub_y,
                mono: seq.mono_chrome,
                bit_depth: u32::from(seq.bit_depth),
                matrix_identity: seq.matrix_coefficients == 0,
            };
            grain::apply_film_grain(grain, &mut out);
        }
        DecodedPicture {
            width: w,
            height: h,
            planes: planes
                .into_iter()
                .map(|p| p.into_iter().map(|v| v.min(255) as u8).collect())
                .collect(),
            monochrome: seq.mono_chrome,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
            full_range: seq.color_range,
            matrix_coefficients: seq.matrix_coefficients,
        }
    }

    /// The motion field estimation process (section 7.9).
    fn motion_field_estimation(
        &self,
        seq: &SequenceHeader,
        fh: &FrameHeader,
    ) -> Vec<Vec<[i32; 2]>> {
        let w8 = fh.mi_cols >> 1;
        let h8 = fh.mi_rows >> 1;
        let invalid = [-1i32 << 15, -1i32 << 15];
        let mut mfmvs = vec![vec![invalid; w8 * h8]; TOTAL_REFS_PER_FRAME];
        let last_idx = fh.ref_frame_idx[0];
        let cur_gold_order_hint = fh.order_hints[GOLDEN_FRAME as usize];
        let last_alt_order_hint = self.refs[last_idx]
            .as_ref()
            .map(|s| s.saved_order_hints[ALTREF_FRAME as usize])
            .unwrap_or(0);
        let use_last = last_alt_order_hint != cur_gold_order_hint;
        if use_last {
            self.projection(seq, fh, &mut mfmvs, LAST_FRAME, -1);
        }
        let mut ref_stamp = MFMV_STACK_SIZE - 2;
        if relative_dist(seq, fh.order_hints[BWDREF_FRAME as usize], fh.order_hint) > 0
            && self.projection(seq, fh, &mut mfmvs, BWDREF_FRAME, 1)
        {
            ref_stamp -= 1;
        }
        if relative_dist(seq, fh.order_hints[ALTREF2_FRAME as usize], fh.order_hint) > 0
            && self.projection(seq, fh, &mut mfmvs, ALTREF2_FRAME, 1)
        {
            ref_stamp -= 1;
        }
        if relative_dist(seq, fh.order_hints[ALTREF_FRAME as usize], fh.order_hint) > 0
            && ref_stamp >= 0
            && self.projection(seq, fh, &mut mfmvs, ALTREF_FRAME, 1)
        {
            ref_stamp -= 1;
        }
        if ref_stamp >= 0 {
            self.projection(seq, fh, &mut mfmvs, LAST2_FRAME, -1);
        }
        mfmvs
    }

    fn projection(
        &self,
        seq: &SequenceHeader,
        fh: &FrameHeader,
        mfmvs: &mut [Vec<[i32; 2]>],
        src: i8,
        dst_sign: i32,
    ) -> bool {
        let src_idx = fh.ref_frame_idx[(src - LAST_FRAME) as usize];
        let w8 = fh.mi_cols >> 1;
        let h8 = fh.mi_rows >> 1;
        let Some(slot) = self.refs[src_idx].as_ref() else {
            return false;
        };
        if slot.mi_rows != fh.mi_rows
            || slot.mi_cols != fh.mi_cols
            || slot.frame_type == INTRA_ONLY_FRAME
            || slot.frame_type == KEY_FRAME
        {
            return false;
        }
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let row = 2 * y8 + 1;
                let col = 2 * x8 + 1;
                let i = row * fh.mi_cols + col;
                let src_ref = slot.saved_ref_frames[i];
                if src_ref > INTRA_FRAME {
                    let ref_to_cur =
                        relative_dist(seq, fh.order_hints[src as usize], fh.order_hint);
                    let ref_offset = relative_dist(
                        seq,
                        fh.order_hints[src as usize],
                        slot.saved_order_hints[src_ref as usize],
                    );
                    let pos_valid = ref_to_cur.abs() <= MAX_FRAME_DISTANCE
                        && ref_offset.abs() <= MAX_FRAME_DISTANCE
                        && ref_offset > 0;
                    if pos_valid {
                        let mv = slot.saved_mvs[i];
                        let proj_mv = get_mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
                        let pos_y8 = project(
                            y8 as i32,
                            proj_mv[0],
                            dst_sign,
                            h8 as i32,
                            MAX_OFFSET_HEIGHT,
                        );
                        let pos_x8 =
                            project(x8 as i32, proj_mv[1], dst_sign, w8 as i32, MAX_OFFSET_WIDTH);
                        if let (Some(pos_y8), Some(pos_x8)) = (pos_y8, pos_x8) {
                            for dst in LAST_FRAME..=ALTREF_FRAME {
                                let ref_to_dst =
                                    relative_dist(seq, fh.order_hint, fh.order_hints[dst as usize]);
                                let proj_mv = get_mv_projection(mv, ref_to_dst, ref_offset);
                                mfmvs[dst as usize][pos_y8 as usize * w8 + pos_x8 as usize] =
                                    proj_mv;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

fn get_mv_projection(mv: [i32; 2], numerator: i32, denominator: i32) -> [i32; 2] {
    let clipped_denominator = denominator.min(MAX_FRAME_DISTANCE);
    let clipped_numerator = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let mut proj = [0i32; 2];
    for i in 0..2 {
        let scaled = i64::from(mv[i])
            * i64::from(clipped_numerator)
            * i64::from(tables::DIV_MULT[clipped_denominator as usize]);
        let rounded = if scaled >= 0 {
            (scaled + (1 << 13)) >> 14
        } else {
            -((-scaled + (1 << 13)) >> 14)
        };
        proj[i] = rounded.clamp(-(1 << 14) + 1, (1 << 14) - 1) as i32;
    }
    proj
}

/// `project( v8, delta, dstSign, max8, maxOff8 )`, returning `None` when the
/// position is invalid.
fn project(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> Option<i32> {
    let base8 = (v8 >> 3) << 3;
    let offset8 = if delta >= 0 {
        delta >> (3 + 1 + MI_SIZE_LOG2)
    } else {
        -((-delta) >> (3 + 1 + MI_SIZE_LOG2))
    };
    let v8 = v8 + dst_sign * offset8;
    if v8 < 0 || v8 >= max8 || v8 < base8 - max_off8 || v8 >= base8 + 8 + max_off8 {
        None
    } else {
        Some(v8)
    }
}

#[cfg(test)]
mod tests;
