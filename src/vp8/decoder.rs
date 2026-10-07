//! VP8 frame decoding (RFC 6386): the frame header, per-macroblock modes and
//! motion vectors, DCT tokens, reconstruction, the loop filter and reference
//! frame management.
//!
//! The decoding process matches libvpx's, which is what the conformance
//! vectors in `tests/fixtures/codec/vp8/` are checked against, including the
//! places where the RFC's reference decoder differs from it: loop-filter
//! deltas that a frame does not update keep their previous values, and the
//! segment of a macroblock is only consulted while segmentation is enabled.

use super::bool_decoder::BoolDecoder;
use super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::predict::{
    Edges, Plane, idct_add, idct_dc_add, idct_dc_add_row, inverse_walsh, macroblock_edges,
    predict_block, predict_inter, predict_subblock,
};
use super::tables::*;
use crate::{Error, ErrorKind, Limits, Result};
use std::sync::Arc;

/// One decoded picture, cropped to the frame's display dimensions. Chroma
/// planes are `width.div_ceil(2)` by `height.div_ceil(2)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Picture {
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u8>; 3],
}

/// A reconstructed frame at macroblock-aligned dimensions.
#[derive(Debug)]
pub(super) struct Frame {
    pub(super) planes: [Plane; 3],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MotionVector {
    pub(super) x: i16,
    pub(super) y: i16,
}

impl MotionVector {
    pub(super) const ZERO: Self = Self { x: 0, y: 0 };

    pub(super) fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

pub(super) const INTRA_FRAME: usize = 0;
pub(super) const LAST_FRAME: usize = 1;
pub(super) const GOLDEN_FRAME: usize = 2;
pub(super) const ALTREF_FRAME: usize = 3;

#[derive(Clone, Copy, Debug)]
pub(super) struct MacroblockInfo {
    pub(super) y_mode: u8,
    pub(super) uv_mode: u8,
    pub(super) reference: usize,
    pub(super) mv: MotionVector,
    /// Subblock intra modes; for a whole-macroblock intra mode, the subblock
    /// mode it implies for its neighbours' contexts.
    pub(super) b_modes: [u8; 16],
    /// Subblock motion vectors; every one equals `mv` unless split.
    pub(super) mvs: [MotionVector; 16],
    pub(super) segment: usize,
    pub(super) skip: bool,
}

impl Default for MacroblockInfo {
    fn default() -> Self {
        Self {
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            reference: INTRA_FRAME,
            mv: MotionVector::ZERO,
            b_modes: [B_DC_PRED; 16],
            mvs: [MotionVector::ZERO; 16],
            segment: 0,
            skip: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Probabilities {
    coefficients: [[[[u8; 11]; 3]; 8]; 4],
    mv: [[u8; 19]; 2],
    y_mode: [u8; 4],
    uv_mode: [u8; 3],
}

impl Default for Probabilities {
    fn default() -> Self {
        Self {
            coefficients: DEFAULT_COEFF_PROBS,
            mv: DEFAULT_MV_PROBS,
            y_mode: DEFAULT_Y_MODE_PROBS,
            uv_mode: DEFAULT_UV_MODE_PROBS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Segmentation {
    enabled: bool,
    update_map: bool,
    absolute: bool,
    quantizer: [i32; 4],
    filter_level: [i32; 4],
    tree_probabilities: [u8; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct FilterDeltas {
    enabled: bool,
    reference: [i32; 4],
    mode: [i32; 4],
}

/// Dequantization factors (DC, AC) for one segment.
#[derive(Clone, Copy, Debug, Default)]
struct Dequantizer {
    y1: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
}

/// Coefficients of one macroblock: 16 Y, 4 U, 4 V and the Y2 block, each in
/// raster order and already dequantized.
pub(super) type Coefficients = [[i16; 16]; 25];

pub(crate) struct Decoder {
    limits: Limits,
    width: usize,
    height: usize,
    mb_cols: usize,
    mb_rows: usize,
    probabilities: Probabilities,
    segmentation: Segmentation,
    filter_deltas: FilterDeltas,
    /// Segment of every macroblock; persists while a frame does not update it.
    segment_map: Vec<u8>,
    /// Last, golden and altref, indexed by `LAST_FRAME..=ALTREF_FRAME`.
    references: [Option<Arc<Frame>>; 4],
    output_wanted: bool,
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

impl Decoder {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            limits,
            width: 0,
            height: 0,
            mb_cols: 0,
            mb_rows: 0,
            probabilities: Probabilities::default(),
            segmentation: Segmentation::default(),
            filter_deltas: FilterDeltas::default(),
            segment_map: Vec::new(),
            references: [None, None, None, None],
            output_wanted: true,
        }
    }

    /// Forgets every reference, so decoding must restart at a key frame.
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.limits);
    }

    pub(crate) fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
    }

    /// Decodes one compressed frame, returning its picture unless the frame
    /// is hidden (typically an alternate reference, which only updates the
    /// references) or output is not wanted.
    pub(crate) fn decode(&mut self, data: &[u8]) -> Result<Option<Picture>> {
        if data.len() < 3 {
            return Err(malformed("VP8 frame is shorter than its frame tag"));
        }
        let tag = u32::from(data[0]) | u32::from(data[1]) << 8 | u32::from(data[2]) << 16;
        let key_frame = tag & 1 == 0;
        let version = (tag >> 1) & 7;
        let shown = (tag >> 4) & 1 == 1;
        let first_partition_size = (tag >> 5) as usize;
        if version > 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!("VP8 bitstream version {version} is not supported"),
            ));
        }
        let mut rest = &data[3..];
        if key_frame {
            if rest.len() < 7 {
                return Err(malformed("VP8 key frame header is truncated"));
            }
            if rest[0..3] != [0x9d, 0x01, 0x2a] {
                return Err(malformed("VP8 key frame has an invalid start code"));
            }
            // The top two bits of each dimension are an upscaling hint for
            // the display, which does not change what is decoded.
            let width = usize::from(u16::from_le_bytes([rest[3], rest[4]]) & 0x3fff);
            let height = usize::from(u16::from_le_bytes([rest[5], rest[6]]) & 0x3fff);
            self.start_key_frame(width, height)?;
            rest = &rest[7..];
        } else if self.references[LAST_FRAME].is_none() {
            return Err(malformed("VP8 inter frame precedes the first key frame"));
        }
        if first_partition_size > rest.len() {
            return Err(malformed("VP8 first partition is truncated"));
        }
        let (first_partition, token_data) = rest.split_at(first_partition_size);
        let mut header = BoolDecoder::new(first_partition);

        if key_frame {
            // Colour space and clamping type: VP8 defines a single colour
            // space, and libvpx clamps reconstructed pixels either way.
            header.read_literal(2);
        }
        self.read_segmentation(&mut header);
        let simple_filter = header.read_flag();
        let filter_level = header.read_literal(6) as i32;
        let sharpness = header.read_literal(3) as u8;
        self.read_filter_deltas(&mut header);
        let partition_count = 1usize << header.read_literal(2);
        let partitions = split_partitions(token_data, partition_count)?;
        let dequantizers = self.read_quantizers(&mut header);

        let (refresh_golden, refresh_altref, copy_to_golden, copy_to_altref, sign_bias) =
            if key_frame {
                (true, true, 0, 0, [false; 4])
            } else {
                let refresh_golden = header.read_flag();
                let refresh_altref = header.read_flag();
                let copy_to_golden = if refresh_golden {
                    0
                } else {
                    header.read_literal(2)
                };
                let copy_to_altref = if refresh_altref {
                    0
                } else {
                    header.read_literal(2)
                };
                let golden_bias = header.read_flag();
                let altref_bias = header.read_flag();
                (
                    refresh_golden,
                    refresh_altref,
                    copy_to_golden,
                    copy_to_altref,
                    [false, false, golden_bias, altref_bias],
                )
            };
        let refresh_probabilities = header.read_flag();
        let refresh_last = key_frame || header.read_flag();

        let saved_probabilities = self.probabilities;
        self.read_coefficient_updates(&mut header);
        let skip_probability = header.read_flag().then(|| header.read_literal(8) as u8);
        let mut inter_probabilities = None;
        if !key_frame {
            let intra = header.read_literal(8) as u8;
            let last = header.read_literal(8) as u8;
            let golden = header.read_literal(8) as u8;
            inter_probabilities = Some((intra, last, golden));
            if header.read_flag() {
                for probability in &mut self.probabilities.y_mode {
                    *probability = header.read_literal(8) as u8;
                }
            }
            if header.read_flag() {
                for probability in &mut self.probabilities.uv_mode {
                    *probability = header.read_literal(8) as u8;
                }
            }
            self.read_mv_updates(&mut header);
        }

        let context = FrameContext {
            key_frame,
            skip_probability,
            inter_probabilities,
            sign_bias,
            filters: if version == 0 {
                &SIXTAP_FILTERS
            } else {
                &BILINEAR_FILTERS
            },
            full_pixel_chroma: version == 3,
        };
        let mut frame = Frame {
            planes: [
                Plane::new(self.mb_cols * 16, self.mb_rows * 16),
                Plane::new(self.mb_cols * 8, self.mb_rows * 8),
                Plane::new(self.mb_cols * 8, self.mb_rows * 8),
            ],
        };
        let mut filters = vec![MacroblockFilter::default(); self.mb_cols * self.mb_rows];
        self.decode_macroblocks(
            &context,
            &mut header,
            partitions,
            &dequantizers,
            filter_level,
            &mut frame,
            &mut filters,
        );
        if filter_level > 0 {
            filter_frame(
                &mut frame.planes,
                &filters,
                self.mb_cols,
                FrameFilter {
                    simple: simple_filter,
                    sharpness,
                    key_frame,
                },
            );
        }

        if !refresh_probabilities {
            self.probabilities = saved_probabilities;
        }

        let frame = Arc::new(frame);
        // Copies happen before refreshes, altref before golden, so a golden
        // "copy from altref" sees an altref this frame has already replaced.
        match copy_to_altref {
            1 => self.references[ALTREF_FRAME] = self.references[LAST_FRAME].clone(),
            2 => self.references[ALTREF_FRAME] = self.references[GOLDEN_FRAME].clone(),
            _ => {}
        }
        match copy_to_golden {
            1 => self.references[GOLDEN_FRAME] = self.references[LAST_FRAME].clone(),
            2 => self.references[GOLDEN_FRAME] = self.references[ALTREF_FRAME].clone(),
            _ => {}
        }
        if refresh_golden {
            self.references[GOLDEN_FRAME] = Some(frame.clone());
        }
        if refresh_altref {
            self.references[ALTREF_FRAME] = Some(frame.clone());
        }
        if refresh_last {
            self.references[LAST_FRAME] = Some(frame.clone());
        }

        Ok((shown && self.output_wanted).then(|| self.crop(&frame)))
    }

    fn start_key_frame(&mut self, width: usize, height: usize) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(malformed("VP8 key frame has zero dimensions"));
        }
        if width as u64 > u64::from(self.limits.max_width)
            || height as u64 > u64::from(self.limits.max_height)
        {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 frame dimensions exceed the configured limits",
            ));
        }
        let mb_cols = width.div_ceil(16);
        let mb_rows = height.div_ceil(16);
        // Four frames of 4:2:0 samples (the one being decoded and three
        // references) is the most this decoder holds at once.
        let frame_bytes = (mb_cols * 16 * mb_rows * 16 * 3 / 2) as u64;
        if frame_bytes.saturating_mul(4) > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 frame exceeds the allocation limit",
            ));
        }
        self.width = width;
        self.height = height;
        self.mb_cols = mb_cols;
        self.mb_rows = mb_rows;
        self.probabilities = Probabilities::default();
        self.segmentation = Segmentation::default();
        self.filter_deltas = FilterDeltas::default();
        self.segment_map = vec![0; mb_cols * mb_rows];
        Ok(())
    }

    fn read_segmentation(&mut self, header: &mut BoolDecoder<'_>) {
        let segmentation = &mut self.segmentation;
        segmentation.enabled = header.read_flag();
        segmentation.update_map = false;
        if !segmentation.enabled {
            return;
        }
        segmentation.update_map = header.read_flag();
        let update_data = header.read_flag();
        if update_data {
            segmentation.absolute = header.read_flag();
            for value in &mut segmentation.quantizer {
                *value = header.read_optional_signed(7);
            }
            for value in &mut segmentation.filter_level {
                *value = header.read_optional_signed(6);
            }
        }
        if segmentation.update_map {
            for probability in &mut segmentation.tree_probabilities {
                *probability = if header.read_flag() {
                    header.read_literal(8) as u8
                } else {
                    255
                };
            }
        }
    }

    fn read_filter_deltas(&mut self, header: &mut BoolDecoder<'_>) {
        let deltas = &mut self.filter_deltas;
        deltas.enabled = header.read_flag();
        if deltas.enabled && header.read_flag() {
            // A delta that is not sent keeps its previous value.
            for value in deltas.reference.iter_mut().chain(deltas.mode.iter_mut()) {
                if header.read_flag() {
                    *value = header.read_signed(6);
                }
            }
        }
    }

    fn read_quantizers(&self, header: &mut BoolDecoder<'_>) -> [Dequantizer; 4] {
        let base = header.read_literal(7) as i32;
        let y1_dc = header.read_optional_signed(4);
        let y2_dc = header.read_optional_signed(4);
        let y2_ac = header.read_optional_signed(4);
        let uv_dc = header.read_optional_signed(4);
        let uv_ac = header.read_optional_signed(4);
        let dc = |q: i32| DC_Q_LOOKUP[q.clamp(0, 127) as usize];
        let ac = |q: i32| AC_Q_LOOKUP[q.clamp(0, 127) as usize];
        let mut dequantizers = [Dequantizer::default(); 4];
        for (segment, dequantizer) in dequantizers.iter_mut().enumerate() {
            let q = if self.segmentation.enabled {
                let value = self.segmentation.quantizer[segment];
                if self.segmentation.absolute {
                    value
                } else {
                    base + value
                }
            } else {
                base
            }
            .clamp(0, 127);
            *dequantizer = Dequantizer {
                y1: [dc(q + y1_dc), ac(q)],
                y2: [dc(q + y2_dc) * 2, (ac(q + y2_ac) * 155 / 100).max(8)],
                uv: [dc(q + uv_dc).min(132), ac(q + uv_ac)],
            };
        }
        dequantizers
    }

    fn read_coefficient_updates(&mut self, header: &mut BoolDecoder<'_>) {
        for (block_type, bands) in self.probabilities.coefficients.iter_mut().enumerate() {
            for (band, contexts) in bands.iter_mut().enumerate() {
                for (context, probabilities) in contexts.iter_mut().enumerate() {
                    for (node, probability) in probabilities.iter_mut().enumerate() {
                        if header.read(COEFF_UPDATE_PROBS[block_type][band][context][node]) {
                            *probability = header.read_literal(8) as u8;
                        }
                    }
                }
            }
        }
    }

    fn read_mv_updates(&mut self, header: &mut BoolDecoder<'_>) {
        for (component, probabilities) in self.probabilities.mv.iter_mut().enumerate() {
            for (index, probability) in probabilities.iter_mut().enumerate() {
                if header.read(MV_UPDATE_PROBS[component][index]) {
                    let value = header.read_literal(7) as u8;
                    *probability = if value == 0 { 1 } else { value << 1 };
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_macroblocks(
        &mut self,
        context: &FrameContext,
        header: &mut BoolDecoder<'_>,
        mut partitions: Vec<BoolDecoder<'_>>,
        dequantizers: &[Dequantizer; 4],
        filter_level: i32,
        frame: &mut Frame,
        filters: &mut [MacroblockFilter],
    ) {
        let mb_cols = self.mb_cols;
        // Mode info with a border row above and a border column to the left,
        // which read as intra DC macroblocks with zero motion.
        let info_stride = mb_cols + 1;
        let mut info = vec![MacroblockInfo::default(); info_stride * (self.mb_rows + 1)];
        let mut above_contexts = vec![[0u8; 9]; mb_cols];
        let partition_count = partitions.len();

        for mb_y in 0..self.mb_rows {
            let partition = &mut partitions[mb_y % partition_count];
            let mut left_context = [0u8; 9];
            for (mb_x, above_context) in above_contexts.iter_mut().enumerate() {
                let index = (mb_y + 1) * info_stride + mb_x + 1;
                let macroblock = self.read_macroblock_header(
                    context,
                    header,
                    &info,
                    index,
                    info_stride,
                    mb_x,
                    mb_y,
                );
                let segment_index = mb_y * mb_cols + mb_x;
                let dequantizer = &dequantizers[if self.segmentation.enabled {
                    macroblock.segment
                } else {
                    0
                }];
                let has_y2 = macroblock.y_mode != B_PRED && macroblock.y_mode != SPLITMV;
                let mut coefficients: Coefficients = [[0; 16]; 25];
                let has_coefficients = if macroblock.skip {
                    let above = above_context;
                    above[..8].fill(0);
                    left_context[..8].fill(0);
                    if has_y2 {
                        above[8] = 0;
                        left_context[8] = 0;
                    }
                    false
                } else {
                    read_tokens(
                        partition,
                        &self.probabilities.coefficients,
                        &mut left_context,
                        above_context,
                        has_y2,
                        dequantizer,
                        &mut coefficients,
                    )
                };
                reconstruct(
                    &self.references,
                    context,
                    &macroblock,
                    &mut coefficients,
                    frame,
                    mb_x,
                    mb_y,
                );

                filters[segment_index] = MacroblockFilter {
                    level: self.filter_level(filter_level, &macroblock),
                    inner_edges: !has_y2 || has_coefficients,
                };
                info[index] = macroblock;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn read_macroblock_header(
        &mut self,
        context: &FrameContext,
        header: &mut BoolDecoder<'_>,
        info: &[MacroblockInfo],
        index: usize,
        info_stride: usize,
        mb_x: usize,
        mb_y: usize,
    ) -> MacroblockInfo {
        let segment_index = mb_y * self.mb_cols + mb_x;
        if self.segmentation.update_map {
            let probabilities = &self.segmentation.tree_probabilities;
            let segment = if header.read(probabilities[0]) {
                2 + usize::from(header.read(probabilities[2]))
            } else {
                usize::from(header.read(probabilities[1]))
            };
            self.segment_map[segment_index] = segment as u8;
        } else if context.key_frame {
            self.segment_map[segment_index] = 0;
        }
        let mut macroblock = MacroblockInfo {
            segment: usize::from(self.segment_map[segment_index]),
            skip: context
                .skip_probability
                .is_some_and(|probability| header.read(probability)),
            ..MacroblockInfo::default()
        };
        let above = &info[index - info_stride];
        let left = &info[index - 1];

        if context.key_frame {
            macroblock.y_mode = header.read_tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS);
            if macroblock.y_mode == B_PRED {
                for block in 0..16 {
                    let above_mode = if block < 4 {
                        above.b_modes[block + 12]
                    } else {
                        macroblock.b_modes[block - 4]
                    };
                    let left_mode = if block & 3 == 0 {
                        left.b_modes[block + 3]
                    } else {
                        macroblock.b_modes[block - 1]
                    };
                    macroblock.b_modes[block] = header.read_tree(
                        &B_MODE_TREE,
                        &KF_B_MODE_PROBS[usize::from(above_mode)][usize::from(left_mode)],
                    );
                }
            } else {
                macroblock.b_modes = [implied_b_mode(macroblock.y_mode); 16];
            }
            macroblock.uv_mode = header.read_tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS);
            return macroblock;
        }

        let (intra_probability, last_probability, golden_probability) = context
            .inter_probabilities
            .expect("inter frames read their reference probabilities");
        if !header.read(intra_probability) {
            macroblock.y_mode = header.read_tree(&Y_MODE_TREE, &self.probabilities.y_mode);
            if macroblock.y_mode == B_PRED {
                for mode in &mut macroblock.b_modes {
                    *mode = header.read_tree(&B_MODE_TREE, &DEFAULT_B_MODE_PROBS);
                }
            } else {
                macroblock.b_modes = [implied_b_mode(macroblock.y_mode); 16];
            }
            macroblock.uv_mode = header.read_tree(&UV_MODE_TREE, &self.probabilities.uv_mode);
            return macroblock;
        }

        macroblock.reference = if header.read(last_probability) {
            if header.read(golden_probability) {
                ALTREF_FRAME
            } else {
                GOLDEN_FRAME
            }
        } else {
            LAST_FRAME
        };
        let above_left = &info[index - info_stride - 1];
        let near = find_near_mvs(
            [above, left, above_left],
            macroblock.reference,
            &context.sign_bias,
        );
        let probabilities = [
            MODE_CONTEXTS[near.counts[0]][0],
            MODE_CONTEXTS[near.counts[1]][1],
            MODE_CONTEXTS[near.counts[2]][2],
            MODE_CONTEXTS[near.counts[3]][3],
        ];
        macroblock.y_mode = header.read_tree(&MV_REF_TREE, &probabilities);
        macroblock.uv_mode = macroblock.y_mode;

        // Motion vectors are bounded to at most one macroblock outside the
        // frame, in eighth samples.
        let bounds = MvBounds::new(mb_x, mb_y, self.mb_cols, self.mb_rows);
        let mv_probabilities = self.probabilities.mv;
        match macroblock.y_mode {
            NEARESTMV => macroblock.mv = bounds.clamp(near.mvs[1]),
            NEARMV => macroblock.mv = bounds.clamp(near.mvs[2]),
            ZEROMV => macroblock.mv = MotionVector::ZERO,
            NEWMV => {
                let best = bounds.clamp(near.mvs[0]);
                macroblock.mv = add_mv(read_mv(header, &mv_probabilities), best);
            }
            SPLITMV => {
                let best = bounds.clamp(near.mvs[0]);
                read_split_mvs(
                    header,
                    &mut macroblock,
                    above,
                    left,
                    best,
                    &mv_probabilities,
                );
                macroblock.mv = macroblock.mvs[15];
                return macroblock;
            }
            _ => unreachable!("the MV reference tree has five leaves"),
        }
        macroblock.mvs = [macroblock.mv; 16];
        macroblock
    }

    fn filter_level(&self, frame_level: i32, macroblock: &MacroblockInfo) -> u8 {
        let mut level = frame_level;
        if self.segmentation.enabled {
            let value = self.segmentation.filter_level[macroblock.segment];
            level = if self.segmentation.absolute {
                value
            } else {
                level + value
            };
            level = level.clamp(0, 63);
        }
        if self.filter_deltas.enabled {
            level += self.filter_deltas.reference[macroblock.reference];
            if macroblock.reference == INTRA_FRAME {
                if macroblock.y_mode == B_PRED {
                    level += self.filter_deltas.mode[0];
                }
            } else if macroblock.y_mode == ZEROMV {
                level += self.filter_deltas.mode[1];
            } else if macroblock.y_mode == SPLITMV {
                level += self.filter_deltas.mode[3];
            } else {
                level += self.filter_deltas.mode[2];
            }
            level = level.clamp(0, 63);
        }
        level as u8
    }

    fn crop(&self, frame: &Frame) -> Picture {
        let chroma_width = self.width.div_ceil(2);
        let chroma_height = self.height.div_ceil(2);
        let crop = |plane: &Plane, width: usize, height: usize| {
            let mut samples = Vec::with_capacity(width * height);
            for row in 0..height {
                let start = row * plane.width;
                samples.extend_from_slice(&plane.data[start..start + width]);
            }
            samples
        };
        Picture {
            width: self.width,
            height: self.height,
            planes: [
                crop(&frame.planes[0], self.width, self.height),
                crop(&frame.planes[1], chroma_width, chroma_height),
                crop(&frame.planes[2], chroma_width, chroma_height),
            ],
        }
    }
}

/// Predicts a macroblock and adds its dequantized residual to `frame`, the
/// frame being reconstructed. The encoder reconstructs its frames through
/// this too, so its references are exactly the decoder's.
pub(super) fn reconstruct(
    references: &[Option<Arc<Frame>>; 4],
    context: &FrameContext,
    macroblock: &MacroblockInfo,
    coefficients: &mut Coefficients,
    frame: &mut Frame,
    mb_x: usize,
    mb_y: usize,
) {
    if macroblock.y_mode != B_PRED && macroblock.y_mode != SPLITMV {
        let dc = inverse_walsh(&coefficients[24]);
        for (block, value) in dc.into_iter().enumerate() {
            coefficients[block][0] = value;
        }
    }
    let [y_plane, u_plane, v_plane] = &mut frame.planes;
    if macroblock.reference == INTRA_FRAME {
        reconstruct_intra_luma(macroblock, coefficients, y_plane, mb_x, mb_y);
        for (plane, first_block) in [(u_plane, 16), (v_plane, 20)] {
            let edges = macroblock_edges::<8, 9>(plane, mb_x, mb_y);
            let stride = plane.width;
            let origin = mb_y * 8 * stride + mb_x * 8;
            predict_block(
                macroblock.uv_mode,
                &edges,
                mb_y > 0,
                mb_x > 0,
                &mut plane.data,
                origin,
                stride,
            );
            add_residual(
                &coefficients[first_block..first_block + 4],
                plane,
                origin,
                2,
            );
        }
        return;
    }

    let reference = references[macroblock.reference]
        .as_ref()
        .expect("every reference is set by the first key frame");
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    if macroblock.y_mode == SPLITMV {
        for block in 0..16 {
            let mv = macroblock.mvs[block];
            predict_inter(
                &reference.planes[0],
                y_plane,
                x0 + (block & 3) * 4,
                y0 + (block >> 2) * 4,
                4,
                4,
                i32::from(mv.x),
                i32::from(mv.y),
                context.filters,
            );
        }
    } else {
        predict_inter(
            &reference.planes[0],
            y_plane,
            x0,
            y0,
            16,
            16,
            i32::from(macroblock.mv.x),
            i32::from(macroblock.mv.y),
            context.filters,
        );
    }
    add_residual(&coefficients[0..16], y_plane, y0 * y_plane.width + x0, 4);

    let chroma_mvs = chroma_mvs(macroblock, context.full_pixel_chroma);
    for (plane_index, plane, first_block) in [(1, u_plane, 16), (2, v_plane, 20)] {
        let reference = &reference.planes[plane_index];
        if macroblock.y_mode == SPLITMV {
            for (block, (mv_x, mv_y)) in chroma_mvs.iter().copied().enumerate() {
                predict_inter(
                    reference,
                    plane,
                    mb_x * 8 + (block & 1) * 4,
                    mb_y * 8 + (block >> 1) * 4,
                    4,
                    4,
                    mv_x,
                    mv_y,
                    context.filters,
                );
            }
        } else {
            let (mv_x, mv_y) = chroma_mvs[0];
            predict_inter(
                reference,
                plane,
                mb_x * 8,
                mb_y * 8,
                8,
                8,
                mv_x,
                mv_y,
                context.filters,
            );
        }
        let origin = mb_y * 8 * plane.width + mb_x * 8;
        add_residual(
            &coefficients[first_block..first_block + 4],
            plane,
            origin,
            2,
        );
    }
}

pub(super) struct FrameContext {
    pub(super) key_frame: bool,
    pub(super) skip_probability: Option<u8>,
    /// Probabilities of intra, last-versus-other and golden-versus-altref.
    pub(super) inter_probabilities: Option<(u8, u8, u8)>,
    pub(super) sign_bias: [bool; 4],
    pub(super) filters: &'static [[i32; 6]; 8],
    pub(super) full_pixel_chroma: bool,
}

/// Splits the token data into its DCT partitions, whose sizes (all but the
/// last) precede them as 24-bit little-endian values.
fn split_partitions(data: &[u8], count: usize) -> Result<Vec<BoolDecoder<'_>>> {
    let sizes_length = 3 * (count - 1);
    if data.len() < sizes_length {
        return Err(malformed("VP8 partition sizes are truncated"));
    }
    let (sizes, mut rest) = data.split_at(sizes_length);
    let mut partitions = Vec::with_capacity(count);
    for index in 0..count {
        let size = if index + 1 < count {
            let bytes = &sizes[index * 3..index * 3 + 3];
            usize::from(bytes[0]) | usize::from(bytes[1]) << 8 | usize::from(bytes[2]) << 16
        } else {
            rest.len()
        };
        if size > rest.len() {
            return Err(malformed("VP8 DCT partition is truncated"));
        }
        let (partition, remaining) = rest.split_at(size);
        partitions.push(BoolDecoder::new(partition));
        rest = remaining;
    }
    Ok(partitions)
}

pub(super) fn implied_b_mode(y_mode: u8) -> u8 {
    match y_mode {
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        _ => B_DC_PRED,
    }
}

pub(super) struct NearMvs {
    /// Best, nearest and near.
    pub(super) mvs: [MotionVector; 3],
    pub(super) counts: [usize; 4],
}

/// Ranks the motion vectors of the above, left and above-left macroblocks
/// (RFC 6386 section 18.3).
pub(super) fn find_near_mvs(
    neighbours: [&MacroblockInfo; 3],
    reference: usize,
    sign_bias: &[bool; 4],
) -> NearMvs {
    let mut mvs = [MotionVector::ZERO; 4];
    let mut counts = [0usize; 4];
    let mut last = 0usize;
    for (neighbour, weight) in neighbours.into_iter().zip([2, 2, 1]) {
        if neighbour.reference == INTRA_FRAME {
            continue;
        }
        if neighbour.mv.is_zero() {
            counts[0] += weight;
            continue;
        }
        let mut mv = neighbour.mv;
        if sign_bias[neighbour.reference] != sign_bias[reference] {
            mv = MotionVector {
                x: mv.x.wrapping_neg(),
                y: mv.y.wrapping_neg(),
            };
        }
        // The above macroblock always starts a new entry; the others merge
        // with the most recent one when they match it.
        if last == 0 || mv != mvs[last] {
            last += 1;
            mvs[last] = mv;
        }
        counts[last] += weight;
    }
    // Three distinct vectors: the third merges with the nearest if equal.
    if counts[3] > 0 && mvs[3] == mvs[1] {
        counts[1] += 1;
    }
    counts[3] = usize::from(neighbours[0].y_mode == SPLITMV) * 2
        + usize::from(neighbours[1].y_mode == SPLITMV) * 2
        + usize::from(neighbours[2].y_mode == SPLITMV);
    if counts[2] > counts[1] {
        counts.swap(1, 2);
        mvs.swap(1, 2);
    }
    if counts[1] >= counts[0] {
        mvs[0] = mvs[1];
    }
    NearMvs {
        mvs: [mvs[0], mvs[1], mvs[2]],
        counts,
    }
}

pub(super) struct MvBounds {
    pub(super) left: i32,
    pub(super) right: i32,
    pub(super) top: i32,
    pub(super) bottom: i32,
}

impl MvBounds {
    /// Motion vectors are bounded to at most one macroblock outside the
    /// frame, in eighth samples.
    pub(super) fn new(mb_x: usize, mb_y: usize, mb_cols: usize, mb_rows: usize) -> Self {
        Self {
            left: -(((mb_x + 1) as i32) << 7),
            right: ((mb_cols - mb_x) as i32) << 7,
            top: -(((mb_y + 1) as i32) << 7),
            bottom: ((mb_rows - mb_y) as i32) << 7,
        }
    }

    pub(super) fn clamp(&self, mv: MotionVector) -> MotionVector {
        MotionVector {
            x: i32::from(mv.x).clamp(self.left, self.right) as i16,
            y: i32::from(mv.y).clamp(self.top, self.bottom) as i16,
        }
    }
}

fn add_mv(a: MotionVector, b: MotionVector) -> MotionVector {
    MotionVector {
        x: a.x.wrapping_add(b.x),
        y: a.y.wrapping_add(b.y),
    }
}

fn read_mv_component(header: &mut BoolDecoder<'_>, probabilities: &[u8; 19]) -> i16 {
    const IS_SHORT: usize = 0;
    const SIGN: usize = 1;
    const SHORT_TREE: usize = 2;
    const LONG_BITS: usize = 9;
    let mut value: i32;
    if header.read(probabilities[IS_SHORT]) {
        value = 0;
        for bit in 0..3 {
            value += i32::from(header.read(probabilities[LONG_BITS + bit])) << bit;
        }
        for bit in (4..10).rev() {
            value += i32::from(header.read(probabilities[LONG_BITS + bit])) << bit;
        }
        // Bit 3 is implicit when no higher bit is set, as the value would
        // otherwise have been coded in the short form.
        if value & 0xfff0 == 0 || header.read(probabilities[LONG_BITS + 3]) {
            value += 8;
        }
    } else {
        value = i32::from(header.read_tree(&SMALL_MV_TREE, &probabilities[SHORT_TREE..]));
    }
    if value != 0 && header.read(probabilities[SIGN]) {
        value = -value;
    }
    (value * 2) as i16
}

fn read_mv(header: &mut BoolDecoder<'_>, probabilities: &[[u8; 19]; 2]) -> MotionVector {
    let y = read_mv_component(header, &probabilities[0]);
    let x = read_mv_component(header, &probabilities[1]);
    MotionVector { x, y }
}

fn read_split_mvs(
    header: &mut BoolDecoder<'_>,
    macroblock: &mut MacroblockInfo,
    above: &MacroblockInfo,
    left: &MacroblockInfo,
    best: MotionVector,
    probabilities: &[[u8; 19]; 2],
) {
    let partitioning = usize::from(header.read_tree(&SPLIT_MV_TREE, &SPLIT_MV_PROBS));
    let layout = &MV_PARTITIONS[partitioning];
    for part in 0..MV_PARTITION_COUNTS[partitioning] {
        let first = layout
            .iter()
            .position(|&value| usize::from(value) == part)
            .expect("every partition has a subblock");
        let left_mv = if first & 3 == 0 {
            left.mvs[first + 3]
        } else {
            macroblock.mvs[first - 1]
        };
        let above_mv = if first < 4 {
            above.mvs[first + 12]
        } else {
            macroblock.mvs[first - 4]
        };
        let context = match (left_mv == above_mv, left_mv.is_zero(), above_mv.is_zero()) {
            (true, true, _) => 4,
            (true, false, _) => 3,
            (false, _, true) => 2,
            (false, true, false) => 1,
            (false, false, false) => 0,
        };
        let mv = match header.read_tree(&SUBMV_REF_TREE, &SUBMV_REF_PROBS[context]) {
            LEFT4X4 => left_mv,
            ABOVE4X4 => above_mv,
            ZERO4X4 => MotionVector::ZERO,
            _ => add_mv(read_mv(header, probabilities), best),
        };
        for (block, &value) in layout.iter().enumerate() {
            if usize::from(value) == part {
                macroblock.mvs[block] = mv;
            }
        }
    }
}

/// Reads one block's tokens and stores the dequantized coefficients,
/// returning the index after the last token (16 when it reaches the end).
fn read_block(
    partition: &mut BoolDecoder<'_>,
    probabilities: &[[[u8; 11]; 3]; 8],
    context: usize,
    first: usize,
    dequantizer: [i32; 2],
    output: &mut [i16; 16],
) -> usize {
    let mut index = first;
    let mut node = &probabilities[COEFF_BANDS[index]][context];
    if !partition.read(node[0]) {
        return index;
    }
    loop {
        if !partition.read(node[1]) {
            // A zero is never followed by an end of block, so the next
            // token skips that branch.
            index += 1;
            if index == 16 {
                return 16;
            }
            node = &probabilities[COEFF_BANDS[index]][0];
            continue;
        }
        let (magnitude, next_context) = if !partition.read(node[2]) {
            (1, 1)
        } else if !partition.read(node[3]) {
            let value = if !partition.read(node[4]) {
                2
            } else {
                3 + i32::from(partition.read(node[5]))
            };
            (value, 2)
        } else if !partition.read(node[6]) {
            let value = if !partition.read(node[7]) {
                5 + i32::from(partition.read(159))
            } else {
                7 + (i32::from(partition.read(165)) << 1) + i32::from(partition.read(145))
            };
            (value, 2)
        } else {
            let (base, extra): (i32, &[u8]) = if !partition.read(node[8]) {
                if !partition.read(node[9]) {
                    (11, &CAT3_PROBS)
                } else {
                    (19, &CAT4_PROBS)
                }
            } else if !partition.read(node[10]) {
                (35, &CAT5_PROBS)
            } else {
                (67, &CAT6_PROBS)
            };
            let mut value = 0;
            for &probability in extra {
                value = (value << 1) | i32::from(partition.read(probability));
            }
            (base + value, 2)
        };
        let value = if partition.read_flag() {
            -magnitude
        } else {
            magnitude
        };
        let factor = dequantizer[usize::from(index > 0)];
        // libvpx stores the dequantized value in a `short`.
        output[ZIGZAG[index]] = (value * factor) as i16;
        index += 1;
        if index == 16 {
            return 16;
        }
        node = &probabilities[COEFF_BANDS[index]][next_context];
        if !partition.read(node[0]) {
            return index;
        }
    }
}

/// Reads a macroblock's tokens, returning whether any block had any.
fn read_tokens(
    partition: &mut BoolDecoder<'_>,
    probabilities: &[[[[u8; 11]; 3]; 8]; 4],
    left: &mut [u8; 9],
    above: &mut [u8; 9],
    has_y2: bool,
    dequantizer: &Dequantizer,
    coefficients: &mut Coefficients,
) -> bool {
    let mut any = false;
    let mut block = |partition: &mut BoolDecoder<'_>,
                     block_type: usize,
                     left_index: usize,
                     above_index: usize,
                     first: usize,
                     factors: [i32; 2],
                     output: &mut [i16; 16]| {
        let context = usize::from(left[left_index] + above[above_index]);
        let end = read_block(
            partition,
            &probabilities[block_type],
            context,
            first,
            factors,
            output,
        );
        let nonzero = end > first;
        left[left_index] = u8::from(nonzero);
        above[above_index] = u8::from(nonzero);
        any |= nonzero;
    };
    let (y_type, y_first) = if has_y2 {
        block(partition, 1, 8, 8, 0, dequantizer.y2, &mut coefficients[24]);
        (0, 1)
    } else {
        (3, 0)
    };
    for (index, output) in coefficients[..16].iter_mut().enumerate() {
        block(
            partition,
            y_type,
            index >> 2,
            index & 3,
            y_first,
            dequantizer.y1,
            output,
        );
    }
    for index in 0..8 {
        let plane = index >> 2;
        let within = index & 3;
        block(
            partition,
            2,
            4 + plane * 2 + (within >> 1),
            4 + plane * 2 + (within & 1),
            0,
            dequantizer.uv,
            &mut coefficients[16 + index],
        );
    }
    any
}

/// Adds the residual of `blocks` (raster order, `per_row` to a row) to the
/// plane at `origin`.
fn add_residual(blocks: &[[i16; 16]], plane: &mut Plane, origin: usize, per_row: usize) {
    let stride = plane.width;
    for (row, blocks) in blocks.chunks_exact(per_row).enumerate() {
        let offset = origin + row * 4 * stride;
        if blocks
            .iter()
            .all(|block| block[1..].iter().all(|&value| value == 0))
        {
            // A row of DC-only blocks, the common case at low rates, is one
            // call rather than `per_row`.
            let dcs: [i16; 4] = std::array::from_fn(|index| blocks.get(index).map_or(0, |b| b[0]));
            if dcs.iter().any(|&dc| dc != 0) {
                idct_dc_add_row(&dcs[..per_row], &mut plane.data, offset, stride);
            }
            continue;
        }
        for (index, block) in blocks.iter().enumerate() {
            add_block_residual(block, &mut plane.data, offset + index * 4, stride);
        }
    }
}

/// Adds one 4x4 block's residual, skipping a block with none and taking the
/// DC-only shortcut for a block with nothing else.
fn add_block_residual(block: &[i16; 16], plane: &mut [u8], offset: usize, stride: usize) {
    if block[1..].iter().all(|&value| value == 0) {
        if block[0] != 0 {
            idct_dc_add(block[0], plane, offset, stride);
        }
    } else {
        idct_add(block, plane, offset, stride);
    }
}

fn reconstruct_intra_luma(
    macroblock: &MacroblockInfo,
    coefficients: &Coefficients,
    plane: &mut Plane,
    mb_x: usize,
    mb_y: usize,
) {
    let stride = plane.width;
    let origin = mb_y * 16 * stride + mb_x * 16;
    let edges = macroblock_edges::<16, 21>(plane, mb_x, mb_y);
    if macroblock.y_mode != B_PRED {
        predict_block(
            macroblock.y_mode,
            &edges,
            mb_y > 0,
            mb_x > 0,
            &mut plane.data,
            origin,
            stride,
        );
        add_residual(&coefficients[0..16], plane, origin, 4);
        return;
    }
    for (block, residual) in coefficients[..16].iter().enumerate() {
        let offset = origin + (block >> 2) * 4 * stride + (block & 3) * 4;
        let (above, left) = subblock_edges(plane, &edges, origin, block);
        predict_subblock(
            macroblock.b_modes[block],
            &above,
            &left,
            &mut plane.data,
            offset,
            stride,
        );
        add_block_residual(residual, &mut plane.data, offset, stride);
    }
}

/// The edges luma subblock `block` of the macroblock at `origin` predicts
/// from: the pixel above-left, the four above and the four above-right, and
/// the four to the left.
pub(super) fn subblock_edges(
    plane: &Plane,
    edges: &Edges<16, 21>,
    origin: usize,
    block: usize,
) -> ([u8; 9], [u8; 4]) {
    let stride = plane.width;
    let row = block >> 2;
    let column = block & 3;
    let offset = origin + row * 4 * stride + column * 4;
    let mut above = [0u8; 9];
    let mut left = [0u8; 4];
    if row == 0 {
        above.copy_from_slice(&edges.above[column * 4..column * 4 + 9]);
    } else {
        above[0] = if column == 0 {
            edges.left[row * 4 - 1]
        } else {
            plane.data[offset - stride - 1]
        };
        above[1..5].copy_from_slice(&plane.data[offset - stride..offset - stride + 4]);
        if column < 3 {
            above[5..9].copy_from_slice(&plane.data[offset - stride + 4..offset - stride + 8]);
        } else {
            // Right-column subblocks use the macroblock's above-right
            // pixels, the ones below them not being decoded yet.
            above[5..9].copy_from_slice(&edges.above[17..21]);
        }
    }
    for (r, sample) in left.iter_mut().enumerate() {
        *sample = if column == 0 {
            edges.left[row * 4 + r]
        } else {
            plane.data[offset + r * stride - 1]
        };
    }
    (above, left)
}

/// The chroma motion vectors, in eighth chroma samples: one for the whole
/// macroblock, or one per 4x4 chroma block of a split macroblock, each the
/// rounded average of the four luma vectors it covers.
pub(super) fn chroma_mvs(macroblock: &MacroblockInfo, full_pixel: bool) -> [(i32, i32); 4] {
    let mask = if full_pixel { !7 } else { !0 };
    if macroblock.y_mode != SPLITMV {
        // Halve, rounding away from zero.
        let halve = |value: i16| {
            let value = i32::from(value);
            ((value + if value < 0 { -1 } else { 1 }) / 2) & mask
        };
        let mv = (halve(macroblock.mv.x), halve(macroblock.mv.y));
        return [mv; 4];
    }
    let mut mvs = [(0, 0); 4];
    for (index, mv) in mvs.iter_mut().enumerate() {
        let first = (index >> 1) * 8 + (index & 1) * 2;
        let blocks = [first, first + 1, first + 4, first + 5];
        let average = |component: fn(&MotionVector) -> i16| {
            let sum: i32 = blocks
                .iter()
                .map(|&block| i32::from(component(&macroblock.mvs[block])))
                .sum();
            ((sum + if sum < 0 { -4 } else { 4 }) / 8) & mask
        };
        *mv = (average(|mv| mv.x), average(|mv| mv.y));
    }
    mvs
}
