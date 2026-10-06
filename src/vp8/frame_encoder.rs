//! VP8 frame encoding: mode decision, motion search, the forward transforms,
//! quantization and the bitstream (RFC 6386).
//!
//! Every macroblock is reconstructed with the decoder's own
//! [`reconstruct`] and the frame is loop-filtered with its
//! [`filter_frame`], so the encoder's reference frames are exactly the
//! pictures a decoder produces from the same bytes.
//!
//! The encoder codes key frames and inter frames predicted from the last
//! frame. Key frames refresh the golden and alternate references, which inter
//! frames never use or update. Macroblocks are coded with any of the four
//! whole-macroblock intra modes or the ten subblock intra modes, or as
//! `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV` with quarter-sample motion;
//! split motion vectors are not used. The token probabilities are adapted to
//! each frame without being kept for the next one, so every frame decodes
//! with the default probabilities as its starting point.

use super::bool_encoder::{BoolEncoder, bit_cost, tree_branches, tree_cost};
use super::decoder::{
    Coefficients, Frame, FrameContext, INTRA_FRAME, LAST_FRAME, MacroblockInfo, MotionVector,
    MvBounds, chroma_mvs, find_near_mvs, implied_b_mode, reconstruct, subblock_edges,
};
use super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::predict::{
    Plane, idct_add, macroblock_edges, predict_block, predict_inter, predict_subblock,
};
use super::tables::*;
use crate::{Error, ErrorKind, Result};
use std::sync::Arc;

/// The planes of one source picture, padded to whole macroblocks by repeating
/// the last column and row.
pub(crate) type Source = [Plane; 3];

/// Encodes a stream of frames of one size, keeping the reference frames the
/// decoder will keep.
pub(crate) struct FrameEncoder {
    width: usize,
    height: usize,
    mb_cols: usize,
    mb_rows: usize,
    references: [Option<Arc<Frame>>; 4],
}

/// Dequantization factors (DC, AC) with no quantizer deltas.
#[derive(Clone, Copy, Debug)]
struct Quantizer {
    y1: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
    /// The rate-distortion multiplier for SATD and SAD, in distortion units
    /// per 1/256 bit.
    lambda: u32,
    /// The multiplier for squared-error distortion. It is well below the
    /// textbook `0.85 * step^2 / 4`: a residual skipped in a frame later
    /// frames predict from is paid for again in each of them, and on test
    /// content this setting gives the best quality for its bit rate.
    sse_lambda: u32,
}

impl Quantizer {
    fn new(q: u8) -> Self {
        let q = usize::from(q.min(127));
        let dc = DC_Q_LOOKUP[q];
        let ac = AC_Q_LOOKUP[q];
        Self {
            y1: [dc, ac],
            y2: [dc * 2, (ac * 155 / 100).max(8)],
            uv: [dc.min(132), ac],
            lambda: (ac as u32 / 2).max(1),
            sse_lambda: (ac * ac / 25).max(1) as u32,
        }
    }

    /// Distortion plus `bits` (in 1/256 bits) weighted by lambda.
    fn cost(&self, distortion: u32, bits: u32) -> u64 {
        (u64::from(distortion) << 8) + u64::from(self.lambda) * u64::from(bits)
    }
}

/// One macroblock's decisions and quantized coefficients.
#[derive(Clone, Copy)]
struct CodedMacroblock {
    info: MacroblockInfo,
    /// Quantized levels in raster order: 16 Y, 4 U, 4 V and the Y2 block.
    levels: Coefficients,
    /// The near-MV counts that select the inter mode probabilities.
    near_counts: [usize; 4],
    /// The vector a `NEWMV` is coded relative to.
    best_mv: MotionVector,
}

impl CodedMacroblock {
    fn has_y2(&self) -> bool {
        self.info.y_mode != B_PRED && self.info.y_mode != SPLITMV
    }
}

/// Where coded tokens go: to the bitstream, or to the counts that choose the
/// frame's coefficient probabilities.
trait TokenSink {
    fn coefficient(&mut self, node: [usize; 4], bit: bool);
    fn fixed(&mut self, probability: u8, bit: bool);
}

struct TokenWriter<'a> {
    encoder: &'a mut BoolEncoder,
    probabilities: &'a [[[[u8; 11]; 3]; 8]; 4],
}

impl TokenSink for TokenWriter<'_> {
    fn coefficient(&mut self, [t, b, c, n]: [usize; 4], bit: bool) {
        self.encoder.write(self.probabilities[t][b][c][n], bit);
    }

    fn fixed(&mut self, probability: u8, bit: bool) {
        self.encoder.write(probability, bit);
    }
}

type BranchCounts = [[[[[u32; 2]; 11]; 3]; 8]; 4];

struct TokenCounter {
    counts: Box<BranchCounts>,
}

impl TokenSink for TokenCounter {
    fn coefficient(&mut self, [t, b, c, n]: [usize; 4], bit: bool) {
        self.counts[t][b][c][n][usize::from(bit)] += 1;
    }

    fn fixed(&mut self, _probability: u8, _bit: bool) {}
}

impl FrameEncoder {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            mb_cols: width.div_ceil(16),
            mb_rows: height.div_ceil(16),
            references: [None, None, None, None],
        }
    }

    /// The macroblock-aligned dimensions source planes must have.
    pub(crate) fn padded_dimensions(&self) -> (usize, usize) {
        (self.mb_cols * 16, self.mb_rows * 16)
    }

    /// Encodes `source` as one shown frame at quantizer index `q` (0 to
    /// 127), as a key frame when asked to or when there is no reference yet.
    pub(crate) fn encode(&mut self, source: &Source, key_frame: bool, q: u8) -> Result<Vec<u8>> {
        let key_frame = key_frame || self.references[LAST_FRAME].is_none();
        let quantizer = Quantizer::new(q);
        let filter_level = loop_filter_level(q);
        let context = FrameContext {
            key_frame,
            skip_probability: None,
            inter_probabilities: None,
            sign_bias: [false; 4],
            filters: &SIXTAP_FILTERS,
            full_pixel_chroma: false,
        };
        let mut frame = Frame {
            planes: [
                Plane::new(self.mb_cols * 16, self.mb_rows * 16),
                Plane::new(self.mb_cols * 8, self.mb_rows * 8),
                Plane::new(self.mb_cols * 8, self.mb_rows * 8),
            ],
        };
        let info_stride = self.mb_cols + 1;
        let mut info = vec![MacroblockInfo::default(); info_stride * (self.mb_rows + 1)];
        let mut coded = Vec::with_capacity(self.mb_cols * self.mb_rows);
        let mut filters = Vec::with_capacity(self.mb_cols * self.mb_rows);
        for mb_y in 0..self.mb_rows {
            for mb_x in 0..self.mb_cols {
                let index = (mb_y + 1) * info_stride + mb_x + 1;
                let neighbours = [
                    &info[index - info_stride],
                    &info[index - 1],
                    &info[index - info_stride - 1],
                ];
                let (macroblock, mut dequantized) = self.encode_macroblock(
                    source, &mut frame, &context, &quantizer, neighbours, mb_x, mb_y,
                );
                reconstruct(
                    &self.references,
                    &context,
                    &macroblock.info,
                    &mut dequantized,
                    &mut frame,
                    mb_x,
                    mb_y,
                );
                filters.push(MacroblockFilter {
                    level: filter_level,
                    inner_edges: !macroblock.has_y2() || !macroblock.info.skip,
                });
                info[index] = macroblock.info;
                coded.push(macroblock);
            }
        }
        if filter_level > 0 {
            filter_frame(
                &mut frame.planes,
                &filters,
                self.mb_cols,
                FrameFilter {
                    simple: false,
                    sharpness: 0,
                    key_frame,
                },
            );
        }

        let data = self.write_frame(&coded, key_frame, q, filter_level)?;

        let frame = Arc::new(frame);
        if key_frame {
            self.references = [None, Some(frame.clone()), Some(frame.clone()), Some(frame)];
        } else {
            self.references[LAST_FRAME] = Some(frame);
        }
        Ok(data)
    }

    /// The last frame coded, cropped to the picture's dimensions, as a
    /// decoder reconstructs it.
    #[cfg(test)]
    pub(crate) fn reconstruction(&self) -> [Vec<u8>; 3] {
        let frame = self.references[LAST_FRAME]
            .as_deref()
            .expect("a frame has been coded");
        let crop = |plane: &Plane, width: usize, height: usize| {
            (0..height)
                .flat_map(|row| &plane.data[row * plane.width..row * plane.width + width])
                .copied()
                .collect()
        };
        let (chroma_width, chroma_height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        [
            crop(&frame.planes[0], self.width, self.height),
            crop(&frame.planes[1], chroma_width, chroma_height),
            crop(&frame.planes[2], chroma_width, chroma_height),
        ]
    }

    /// Chooses a macroblock's modes and quantizes its residual, returning
    /// the coded macroblock and its dequantized coefficients.
    #[allow(clippy::too_many_arguments)]
    fn encode_macroblock(
        &self,
        source: &Source,
        frame: &mut Frame,
        context: &FrameContext,
        quantizer: &Quantizer,
        neighbours: [&MacroblockInfo; 3],
        mb_x: usize,
        mb_y: usize,
    ) -> (CodedMacroblock, Coefficients) {
        let mut macroblock = CodedMacroblock {
            info: MacroblockInfo::default(),
            levels: [[0; 16]; 25],
            near_counts: [0; 4],
            best_mv: MotionVector::ZERO,
        };
        let mut dequantized: Coefficients = [[0; 16]; 25];

        // The best inter candidate, when this is an inter frame.
        let mut inter = None;
        if !context.key_frame {
            let near = find_near_mvs(neighbours, LAST_FRAME, &context.sign_bias);
            let bounds = MvBounds::new(mb_x, mb_y, self.mb_cols, self.mb_rows);
            macroblock.near_counts = near.counts;
            macroblock.best_mv = bounds.clamp(near.mvs[0]);
            inter = Some(self.choose_inter_mode(
                source,
                frame,
                quantizer,
                &near.counts,
                [
                    macroblock.best_mv,
                    bounds.clamp(near.mvs[1]),
                    bounds.clamp(near.mvs[2]),
                ],
                &bounds,
                mb_x,
                mb_y,
            ));
        }

        let intra_probabilities = if context.key_frame {
            None
        } else {
            Some(DEFAULT_Y_MODE_PROBS)
        };
        let (y_mode, intra_cost) =
            choose_luma_mode(source, frame, quantizer, intra_probabilities, mb_x, mb_y);
        let use_inter = inter.is_some_and(|(_, _, cost)| cost <= intra_cost);
        let y_mode = if use_inter {
            let (mode, mv, _) = inter.expect("checked above");
            macroblock.info.reference = LAST_FRAME;
            macroblock.info.y_mode = mode;
            macroblock.info.uv_mode = mode;
            macroblock.info.mv = mv;
            macroblock.info.mvs = [mv; 16];
            mode
        } else {
            y_mode
        };

        let origin = mb_y * 16 * frame.planes[0].width + mb_x * 16;
        if y_mode == B_PRED {
            // Choose and code the subblocks in order, each predicted from
            // the reconstruction of the ones before it.
            let (above_modes, left_modes) = if context.key_frame {
                (Some(neighbours[0].b_modes), Some(neighbours[1].b_modes))
            } else {
                (None, None)
            };
            let modes = code_subblocks(
                source,
                &mut frame.planes[0],
                quantizer,
                above_modes.zip(left_modes),
                mb_x,
                mb_y,
                &mut macroblock.levels,
                &mut dequantized,
            );
            macroblock.info.y_mode = B_PRED;
            macroblock.info.b_modes = modes;
        } else {
            if macroblock.info.reference == INTRA_FRAME {
                macroblock.info.y_mode = y_mode;
                macroblock.info.b_modes = [implied_b_mode(y_mode); 16];
                let edges = macroblock_edges::<16, 21>(&frame.planes[0], mb_x, mb_y);
                let stride = frame.planes[0].width;
                predict_block(
                    y_mode,
                    &edges,
                    mb_y > 0,
                    mb_x > 0,
                    &mut frame.planes[0].data,
                    origin,
                    stride,
                );
            } else {
                predict_inter(
                    &self.reference().planes[0],
                    &mut frame.planes[0],
                    mb_x * 16,
                    mb_y * 16,
                    16,
                    16,
                    i32::from(macroblock.info.mv.x),
                    i32::from(macroblock.info.mv.y),
                    &SIXTAP_FILTERS,
                );
            }
            quantize_luma_with_y2(
                &source[0],
                &frame.planes[0],
                origin,
                quantizer,
                &mut macroblock.levels,
                &mut dequantized,
            );
        }

        // Chroma.
        if macroblock.info.reference == INTRA_FRAME {
            let probabilities = if context.key_frame {
                KF_UV_MODE_PROBS
            } else {
                DEFAULT_UV_MODE_PROBS
            };
            macroblock.info.uv_mode =
                choose_chroma_mode(source, frame, quantizer, &probabilities, mb_x, mb_y);
        } else {
            let (mv_x, mv_y) = chroma_mvs(&macroblock.info, false)[0];
            for plane in 1..3 {
                predict_inter(
                    &self.reference().planes[plane],
                    &mut frame.planes[plane],
                    mb_x * 8,
                    mb_y * 8,
                    8,
                    8,
                    mv_x,
                    mv_y,
                    &SIXTAP_FILTERS,
                );
            }
        }
        for (plane, (source, prediction)) in source.iter().zip(&frame.planes).enumerate().skip(1) {
            let stride = prediction.width;
            let origin = mb_y * 8 * stride + mb_x * 8;
            for block in 0..4 {
                let offset = origin + (block >> 1) * 4 * stride + (block & 1) * 4;
                let index = 16 + (plane - 1) * 4 + block;
                let coefficients = forward_dct(&residual(source, prediction, offset));
                (macroblock.levels[index], dequantized[index]) =
                    quantize(&coefficients, quantizer.uv, 0);
            }
        }

        if macroblock.info.reference != INTRA_FRAME
            && !residual_pays(
                source,
                frame,
                quantizer,
                &macroblock,
                &dequantized,
                mb_x,
                mb_y,
            )
        {
            macroblock.levels = [[0; 16]; 25];
            dequantized = [[0; 16]; 25];
        }

        let first = if macroblock.has_y2() { 1 } else { 0 };
        let has_coefficients = macroblock.levels[..24]
            .iter()
            .enumerate()
            .any(|(index, block)| {
                let start = if index < 16 { first } else { 0 };
                block[start..].iter().any(|&level| level != 0)
            })
            || (macroblock.has_y2() && macroblock.levels[24].iter().any(|&level| level != 0));
        macroblock.info.skip = !has_coefficients;
        (macroblock, dequantized)
    }

    fn reference(&self) -> &Frame {
        self.references[LAST_FRAME]
            .as_deref()
            .expect("inter frames follow a key frame")
    }

    /// Searches the last frame for the macroblock's motion, returning the
    /// cheapest inter mode, its vector and its cost.
    #[allow(clippy::too_many_arguments)]
    fn choose_inter_mode(
        &self,
        source: &Source,
        frame: &mut Frame,
        quantizer: &Quantizer,
        counts: &[usize; 4],
        [best, nearest, near]: [MotionVector; 3],
        bounds: &MvBounds,
        mb_x: usize,
        mb_y: usize,
    ) -> (u8, MotionVector, u64) {
        let reference = &self.reference().planes[0];
        let mode_probabilities = [
            MODE_CONTEXTS[counts[0]][0],
            MODE_CONTEXTS[counts[1]][1],
            MODE_CONTEXTS[counts[2]][2],
            MODE_CONTEXTS[counts[3]][3],
        ];
        let x0 = mb_x * 16;
        let y0 = mb_y * 16;
        let new_mv = motion_search(
            &source[0],
            reference,
            &mut frame.planes[0],
            quantizer,
            [best, nearest, near],
            best,
            bounds,
            x0,
            y0,
        );
        let mut candidates = vec![
            (ZEROMV, MotionVector::ZERO, 0),
            (NEARESTMV, nearest, 0),
            (NEARMV, near, 0),
        ];
        if new_mv != nearest && new_mv != near && new_mv != MotionVector::ZERO {
            candidates.push((NEWMV, new_mv, mv_cost(new_mv, best)));
        }
        let origin = y0 * frame.planes[0].width + x0;
        let mut chosen = (ZEROMV, MotionVector::ZERO, u64::MAX);
        for (mode, mv, mv_bits) in candidates {
            predict_inter(
                reference,
                &mut frame.planes[0],
                x0,
                y0,
                16,
                16,
                i32::from(mv.x),
                i32::from(mv.y),
                &SIXTAP_FILTERS,
            );
            let distortion = satd16(&source[0], &frame.planes[0], origin);
            let bits = tree_cost(&MV_REF_TREE, &mode_probabilities, mode) + mv_bits;
            let cost = quantizer.cost(distortion, bits);
            if cost < chosen.2 {
                chosen = (mode, mv, cost);
            }
        }
        chosen
    }

    /// Writes the frame header, the macroblock modes and the tokens.
    fn write_frame(
        &self,
        coded: &[CodedMacroblock],
        key_frame: bool,
        q: u8,
        filter_level: u8,
    ) -> Result<Vec<u8>> {
        // Adapt the token probabilities to this frame's tokens.
        let mut counter = TokenCounter {
            counts: Box::new([[[[[0; 2]; 11]; 3]; 8]; 4]),
        };
        write_tokens(coded, self.mb_cols, &mut counter);
        let mut probabilities = DEFAULT_COEFF_PROBS;
        let mut updated = [[[[false; 11]; 3]; 8]; 4];
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let [zeros, ones] = counter.counts[t][b][c][n];
                        let total = zeros + ones;
                        if total == 0 {
                            continue;
                        }
                        let old = DEFAULT_COEFF_PROBS[t][b][c][n];
                        let new = ((u64::from(zeros) * 256 + u64::from(total) / 2)
                            / u64::from(total))
                        .clamp(1, 255) as u8;
                        let branch_cost = |probability: u8| {
                            u64::from(zeros) * u64::from(bit_cost(probability, false))
                                + u64::from(ones) * u64::from(bit_cost(probability, true))
                        };
                        let update_probability = COEFF_UPDATE_PROBS[t][b][c][n];
                        let update_cost = u64::from(bit_cost(update_probability, true)) + 8 * 256;
                        let keep_cost = u64::from(bit_cost(update_probability, false));
                        if branch_cost(new) + update_cost < branch_cost(old) + keep_cost {
                            probabilities[t][b][c][n] = new;
                            updated[t][b][c][n] = true;
                        }
                    }
                }
            }
        }

        let mut tokens = BoolEncoder::new();
        write_tokens(
            coded,
            self.mb_cols,
            &mut TokenWriter {
                encoder: &mut tokens,
                probabilities: &probabilities,
            },
        );
        let tokens = tokens.finish();

        let ratio = |count: usize, total: usize| {
            ((count * 256 + total / 2) / total.max(1)).clamp(1, 255) as u8
        };
        let total = coded.len();
        let skip_probability = ratio(coded.iter().filter(|m| !m.info.skip).count(), total);
        let intra_probability = ratio(
            coded
                .iter()
                .filter(|m| m.info.reference == INTRA_FRAME)
                .count(),
            total,
        );

        let mut header = BoolEncoder::new();
        if key_frame {
            // Colour space and clamping type.
            header.write_literal(0, 2);
        }
        header.write_flag(false); // segmentation
        header.write_flag(false); // normal loop filter
        header.write_literal(u32::from(filter_level), 6);
        header.write_literal(0, 3); // sharpness
        header.write_flag(false); // no loop-filter deltas
        header.write_literal(0, 2); // one token partition
        header.write_literal(u32::from(q), 7);
        for _ in 0..5 {
            header.write_flag(false); // no quantizer deltas
        }
        if !key_frame {
            header.write_flag(false); // keep golden
            header.write_flag(false); // keep altref
            header.write_literal(0, 2); // no copy to golden
            header.write_literal(0, 2); // no copy to altref
            header.write_flag(false); // golden sign bias
            header.write_flag(false); // altref sign bias
        }
        // The adapted token probabilities are for this frame only.
        header.write_flag(false);
        if !key_frame {
            header.write_flag(true); // refresh last
        }
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let update = updated[t][b][c][n];
                        header.write(COEFF_UPDATE_PROBS[t][b][c][n], update);
                        if update {
                            header.write_literal(u32::from(probabilities[t][b][c][n]), 8);
                        }
                    }
                }
            }
        }
        header.write_flag(true); // macroblocks may skip their tokens
        header.write_literal(u32::from(skip_probability), 8);
        if !key_frame {
            header.write_literal(u32::from(intra_probability), 8);
            header.write_literal(255, 8); // every inter macroblock uses last
            header.write_literal(128, 8);
            header.write_flag(false); // default luma mode probabilities
            header.write_flag(false); // default chroma mode probabilities
            for component in MV_UPDATE_PROBS {
                for probability in component {
                    header.write(probability, false);
                }
            }
        }

        for (index, macroblock) in coded.iter().enumerate() {
            let info = &macroblock.info;
            header.write(skip_probability, info.skip);
            if key_frame {
                header.write_tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, info.y_mode);
                if info.y_mode == B_PRED {
                    let mb_x = index % self.mb_cols;
                    let mb_y = index / self.mb_cols;
                    let above = (mb_y > 0).then(|| &coded[index - self.mb_cols].info);
                    let left = (mb_x > 0).then(|| &coded[index - 1].info);
                    for block in 0..16 {
                        let above_mode = if block < 4 {
                            above.map_or(B_DC_PRED, |above| above.b_modes[block + 12])
                        } else {
                            info.b_modes[block - 4]
                        };
                        let left_mode = if block & 3 == 0 {
                            left.map_or(B_DC_PRED, |left| left.b_modes[block + 3])
                        } else {
                            info.b_modes[block - 1]
                        };
                        header.write_tree(
                            &B_MODE_TREE,
                            &KF_B_MODE_PROBS[usize::from(above_mode)][usize::from(left_mode)],
                            info.b_modes[block],
                        );
                    }
                }
                header.write_tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, info.uv_mode);
                continue;
            }
            header.write(intra_probability, info.reference != INTRA_FRAME);
            if info.reference == INTRA_FRAME {
                header.write_tree(&Y_MODE_TREE, &DEFAULT_Y_MODE_PROBS, info.y_mode);
                if info.y_mode == B_PRED {
                    for &mode in &info.b_modes {
                        header.write_tree(&B_MODE_TREE, &DEFAULT_B_MODE_PROBS, mode);
                    }
                }
                header.write_tree(&UV_MODE_TREE, &DEFAULT_UV_MODE_PROBS, info.uv_mode);
                continue;
            }
            header.write(255, false); // last frame
            let counts = &macroblock.near_counts;
            let mode_probabilities = [
                MODE_CONTEXTS[counts[0]][0],
                MODE_CONTEXTS[counts[1]][1],
                MODE_CONTEXTS[counts[2]][2],
                MODE_CONTEXTS[counts[3]][3],
            ];
            header.write_tree(&MV_REF_TREE, &mode_probabilities, info.y_mode);
            if info.y_mode == NEWMV {
                write_mv(&mut header, info.mv, macroblock.best_mv);
            }
        }
        let first_partition = header.finish();
        if first_partition.len() > MAX_FIRST_PARTITION {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 frame's mode partition exceeds the 512 KiB the frame tag can describe",
            ));
        }

        let mut data = Vec::with_capacity(10 + first_partition.len() + tokens.len());
        let tag = u32::from(!key_frame) | 1 << 4 | (first_partition.len() as u32) << 5;
        data.extend_from_slice(&tag.to_le_bytes()[..3]);
        if key_frame {
            data.extend_from_slice(&[0x9d, 0x01, 0x2a]);
            data.extend_from_slice(&(self.width as u16).to_le_bytes());
            data.extend_from_slice(&(self.height as u16).to_le_bytes());
        }
        data.extend_from_slice(&first_partition);
        data.extend_from_slice(&tokens);
        Ok(data)
    }
}

/// The largest first partition the frame tag's 19-bit size can describe.
const MAX_FIRST_PARTITION: usize = (1 << 19) - 1;

/// The loop filter level for quantizer index `q`, rising with the
/// quantizer's step as libvpx's levels do.
fn loop_filter_level(q: u8) -> u8 {
    if q < 4 {
        0
    } else {
        (u32::from(q) * 5 / 16 + 2).min(63) as u8
    }
}

/// Sums the cost of tokens with the default probabilities.
struct TokenCost {
    cost: u32,
}

impl TokenSink for TokenCost {
    fn coefficient(&mut self, [t, b, c, n]: [usize; 4], bit: bool) {
        self.cost += bit_cost(DEFAULT_COEFF_PROBS[t][b][c][n], bit);
    }

    fn fixed(&mut self, probability: u8, bit: bool) {
        self.cost += bit_cost(probability, bit);
    }
}

/// Whether coding an inter macroblock's quantized residual improves its
/// picture by more than the residual's bits are worth, rather than skipping
/// it and keeping the prediction now in `frame`. Re-coding the small
/// differences a reference's own coding left behind rarely pays.
fn residual_pays(
    source: &Source,
    frame: &Frame,
    quantizer: &Quantizer,
    macroblock: &CodedMacroblock,
    dequantized: &Coefficients,
    mb_x: usize,
    mb_y: usize,
) -> bool {
    let mut bits = TokenCost { cost: 0 };
    let has_y2 = macroblock.has_y2();
    if has_y2 {
        write_block(&mut bits, 1, 0, 0, &macroblock.levels[24]);
    }
    let (y_type, y_first) = if has_y2 { (0, 1) } else { (3, 0) };
    for block in 0..16 {
        write_block(&mut bits, y_type, 0, y_first, &macroblock.levels[block]);
    }
    for block in 16..24 {
        write_block(&mut bits, 2, 0, 0, &macroblock.levels[block]);
    }

    let mut luma = *dequantized;
    if has_y2 {
        let dc = super::predict::inverse_walsh(&luma[24]);
        for (block, value) in dc.into_iter().enumerate() {
            luma[block][0] = value;
        }
    }
    let mut skip_distortion = 0u64;
    let mut coded_distortion = 0u64;
    for (plane, size, first_block) in [(0, 16, 0), (1, 8, 16), (2, 8, 20)] {
        let stride = frame.planes[plane].width;
        let origin = mb_y * size * stride + mb_x * size;
        let mut reconstruction = [0u8; 256];
        for row in 0..size {
            reconstruction[row * size..(row + 1) * size].copy_from_slice(
                &frame.planes[plane].data[origin + row * stride..origin + row * stride + size],
            );
        }
        let per_row = size / 4;
        for block in 0..per_row * per_row {
            let coefficients = &luma[first_block + block];
            if coefficients.iter().any(|&value| value != 0) {
                let offset = (block / per_row) * 4 * size + (block % per_row) * 4;
                idct_add(coefficients, &mut reconstruction, offset, size);
            }
        }
        for row in 0..size {
            for column in 0..size {
                let at = origin + row * stride + column;
                let original = i64::from(source[plane].data[at]);
                let predicted = i64::from(frame.planes[plane].data[at]);
                let coded = i64::from(reconstruction[row * size + column]);
                skip_distortion += ((original - predicted) * (original - predicted)) as u64;
                coded_distortion += ((original - coded) * (original - coded)) as u64;
            }
        }
    }
    let lambda = u64::from(quantizer.sse_lambda);
    (coded_distortion << 8) + lambda * u64::from(bits.cost) < skip_distortion << 8
}

/// Writes every macroblock's tokens in order, tracking the above and left
/// non-zero contexts as the decoder does.
fn write_tokens(coded: &[CodedMacroblock], mb_cols: usize, sink: &mut impl TokenSink) {
    let mut above_contexts = vec![[0u8; 9]; mb_cols];
    let mut left = [0u8; 9];
    for (index, macroblock) in coded.iter().enumerate() {
        let mb_x = index % mb_cols;
        if mb_x == 0 {
            left = [0; 9];
        }
        let above = &mut above_contexts[mb_x];
        let has_y2 = macroblock.has_y2();
        if macroblock.info.skip {
            above[..8].fill(0);
            left[..8].fill(0);
            if has_y2 {
                above[8] = 0;
                left[8] = 0;
            }
            continue;
        }
        let mut block = |sink: &mut dyn TokenSink,
                         block_type: usize,
                         left_index: usize,
                         above_index: usize,
                         first: usize,
                         levels: &[i16; 16]| {
            let context = usize::from(left[left_index] + above[above_index]);
            let nonzero = write_block(sink, block_type, context, first, levels);
            left[left_index] = u8::from(nonzero);
            above[above_index] = u8::from(nonzero);
        };
        let (y_type, y_first) = if has_y2 {
            block(sink, 1, 8, 8, 0, &macroblock.levels[24]);
            (0, 1)
        } else {
            (3, 0)
        };
        for index in 0..16 {
            block(
                sink,
                y_type,
                index >> 2,
                index & 3,
                y_first,
                &macroblock.levels[index],
            );
        }
        for index in 0..8 {
            let plane = index >> 2;
            let within = index & 3;
            block(
                sink,
                2,
                4 + plane * 2 + (within >> 1),
                4 + plane * 2 + (within & 1),
                0,
                &macroblock.levels[16 + index],
            );
        }
    }
}

/// Writes one block's tokens from position `first`, returning whether it had
/// any; the inverse of the decoder's `read_block`.
fn write_block(
    sink: &mut dyn TokenSink,
    block_type: usize,
    context: usize,
    first: usize,
    levels: &[i16; 16],
) -> bool {
    let Some(last) = (first..16).rev().find(|&index| levels[ZIGZAG[index]] != 0) else {
        sink.coefficient([block_type, COEFF_BANDS[first], context, 0], false);
        return false;
    };
    let mut context = context;
    let mut after_zero = false;
    for index in first..=last {
        let node = |n: usize| [block_type, COEFF_BANDS[index], context, n];
        if !after_zero {
            sink.coefficient(node(0), true);
        }
        let value = i32::from(levels[ZIGZAG[index]]);
        let magnitude = value.unsigned_abs();
        if magnitude == 0 {
            sink.coefficient(node(1), false);
            context = 0;
            after_zero = true;
            continue;
        }
        sink.coefficient(node(1), true);
        if magnitude == 1 {
            sink.coefficient(node(2), false);
        } else {
            sink.coefficient(node(2), true);
            if magnitude <= 4 {
                sink.coefficient(node(3), false);
                if magnitude == 2 {
                    sink.coefficient(node(4), false);
                } else {
                    sink.coefficient(node(4), true);
                    sink.coefficient(node(5), magnitude == 4);
                }
            } else if magnitude <= 10 {
                sink.coefficient(node(3), true);
                sink.coefficient(node(6), false);
                if magnitude <= 6 {
                    sink.coefficient(node(7), false);
                    sink.fixed(159, magnitude == 6);
                } else {
                    sink.coefficient(node(7), true);
                    let extra = magnitude - 7;
                    sink.fixed(165, extra & 2 != 0);
                    sink.fixed(145, extra & 1 != 0);
                }
            } else {
                sink.coefficient(node(3), true);
                sink.coefficient(node(6), true);
                let (base, probabilities): (u32, &[u8]) = if magnitude < 19 {
                    sink.coefficient(node(8), false);
                    sink.coefficient(node(9), false);
                    (11, &CAT3_PROBS)
                } else if magnitude < 35 {
                    sink.coefficient(node(8), false);
                    sink.coefficient(node(9), true);
                    (19, &CAT4_PROBS)
                } else if magnitude < 67 {
                    sink.coefficient(node(8), true);
                    sink.coefficient(node(10), false);
                    (35, &CAT5_PROBS)
                } else {
                    sink.coefficient(node(8), true);
                    sink.coefficient(node(10), true);
                    (67, &CAT6_PROBS)
                };
                let extra = magnitude - base;
                let bits = probabilities.len();
                for (bit, &probability) in probabilities.iter().enumerate() {
                    sink.fixed(probability, (extra >> (bits - 1 - bit)) & 1 != 0);
                }
            }
        }
        sink.fixed(128, value < 0);
        context = if magnitude == 1 { 1 } else { 2 };
        after_zero = false;
    }
    if last + 1 < 16 {
        sink.coefficient([block_type, COEFF_BANDS[last + 1], context, 0], false);
    }
    true
}

/// Writes `mv` relative to `best`, in quarter samples, y first.
fn write_mv(header: &mut BoolEncoder, mv: MotionVector, best: MotionVector) {
    write_mv_component(
        header,
        (i32::from(mv.y) - i32::from(best.y)) / 2,
        &DEFAULT_MV_PROBS[0],
    );
    write_mv_component(
        header,
        (i32::from(mv.x) - i32::from(best.x)) / 2,
        &DEFAULT_MV_PROBS[1],
    );
}

const MV_IS_SHORT: usize = 0;
const MV_SIGN: usize = 1;
const MV_SHORT_TREE: usize = 2;
const MV_LONG_BITS: usize = 9;

/// The largest motion vector difference, in quarter samples.
const MAX_MV_DIFFERENCE: i32 = 1023;

fn write_mv_component(header: &mut BoolEncoder, value: i32, probabilities: &[u8; 19]) {
    mv_component_bits(value, probabilities, &mut |probability, bit| {
        header.write(probability, bit)
    });
}

/// Emits the branches that code one motion vector component, the inverse of
/// the decoder's `read_mv_component`.
fn mv_component_bits(value: i32, probabilities: &[u8; 19], emit: &mut dyn FnMut(u8, bool)) {
    let magnitude = value.unsigned_abs();
    if magnitude < 8 {
        emit(probabilities[MV_IS_SHORT], false);
        let mut path = [(0usize, false); 16];
        let length = tree_branches(&SMALL_MV_TREE, magnitude as u8, &mut path);
        for &(index, bit) in &path[..length] {
            emit(probabilities[MV_SHORT_TREE + (index >> 1)], bit);
        }
    } else {
        emit(probabilities[MV_IS_SHORT], true);
        for bit in 0..3 {
            emit(
                probabilities[MV_LONG_BITS + bit],
                (magnitude >> bit) & 1 != 0,
            );
        }
        for bit in (4..10).rev() {
            emit(
                probabilities[MV_LONG_BITS + bit],
                (magnitude >> bit) & 1 != 0,
            );
        }
        if magnitude & 0xfff0 != 0 {
            emit(probabilities[MV_LONG_BITS + 3], (magnitude >> 3) & 1 != 0);
        }
    }
    if magnitude != 0 {
        emit(probabilities[MV_SIGN], value < 0);
    }
}

/// The cost, in 1/256 bits, of coding `mv` relative to `best`.
fn mv_cost(mv: MotionVector, best: MotionVector) -> u32 {
    let mut cost = 0;
    for (component, probabilities) in [
        (i32::from(mv.y) - i32::from(best.y), &DEFAULT_MV_PROBS[0]),
        (i32::from(mv.x) - i32::from(best.x), &DEFAULT_MV_PROBS[1]),
    ] {
        mv_component_bits(component / 2, probabilities, &mut |probability, bit| {
            cost += bit_cost(probability, bit)
        });
    }
    cost
}

/// Finds the macroblock's motion in `reference`: a whole-sample diamond
/// search from the best of the predicted vectors, refined to half and then
/// quarter samples. Vectors are in eighth samples, as the decoder keeps them,
/// and stay within `bounds` and the range a vector difference can code.
#[allow(clippy::too_many_arguments)]
fn motion_search(
    source: &Plane,
    reference: &Plane,
    scratch: &mut Plane,
    quantizer: &Quantizer,
    starts: [MotionVector; 3],
    best: MotionVector,
    bounds: &MvBounds,
    x0: usize,
    y0: usize,
) -> MotionVector {
    let valid = |mv: MotionVector| {
        let (x, y) = (i32::from(mv.x), i32::from(mv.y));
        x >= bounds.left
            && x <= bounds.right
            && y >= bounds.top
            && y <= bounds.bottom
            && ((x - i32::from(best.x)) / 2).abs() <= MAX_MV_DIFFERENCE
            && ((y - i32::from(best.y)) / 2).abs() <= MAX_MV_DIFFERENCE
    };
    let cost = |sad: u32, mv: MotionVector| quantizer.cost(sad, mv_cost(mv, best));

    // Whole samples.
    let whole = |mv: MotionVector| MotionVector {
        x: (mv.x + 4) & !7,
        y: (mv.y + 4) & !7,
    };
    let full_cost = |mv: MotionVector| {
        cost(
            sad16_full(
                source,
                reference,
                x0,
                y0,
                i32::from(mv.x) >> 3,
                i32::from(mv.y) >> 3,
            ),
            mv,
        )
    };
    let mut centre = MotionVector::ZERO;
    let mut centre_cost = full_cost(centre);
    for start in starts.map(whole) {
        if valid(start) {
            let start_cost = full_cost(start);
            if start_cost < centre_cost {
                (centre, centre_cost) = (start, start_cost);
            }
        }
    }
    for step in [64i16, 32, 16, 8] {
        for _ in 0..16 {
            let mut moved = false;
            for (dx, dy) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let candidate = MotionVector {
                    x: centre.x + dx,
                    y: centre.y + dy,
                };
                if !valid(candidate) {
                    continue;
                }
                let candidate_cost = full_cost(candidate);
                if candidate_cost < centre_cost {
                    (centre, centre_cost) = (candidate, candidate_cost);
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }

    // Half, then quarter samples, measured on the six-tap prediction.
    let origin = y0 * scratch.width + x0;
    let sub_cost = |mv: MotionVector, scratch: &mut Plane| {
        predict_inter(
            reference,
            scratch,
            x0,
            y0,
            16,
            16,
            i32::from(mv.x),
            i32::from(mv.y),
            &SIXTAP_FILTERS,
        );
        cost(sad16(source, scratch, origin), mv)
    };
    let mut centre_cost = sub_cost(centre, scratch);
    for step in [4i16, 2] {
        let start = centre;
        for dy in [-step, 0, step] {
            for dx in [-step, 0, step] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let candidate = MotionVector {
                    x: start.x + dx,
                    y: start.y + dy,
                };
                if !valid(candidate) {
                    continue;
                }
                let candidate_cost = sub_cost(candidate, scratch);
                if candidate_cost < centre_cost {
                    (centre, centre_cost) = (candidate, candidate_cost);
                }
            }
        }
    }
    centre
}

/// The sum of absolute differences between the source macroblock at
/// `(x0, y0)` and the reference displaced by whole samples, with the
/// reference extended beyond its edges.
fn sad16_full(source: &Plane, reference: &Plane, x0: usize, y0: usize, dx: i32, dy: i32) -> u32 {
    let rx = x0 as i32 + dx;
    let ry = y0 as i32 + dy;
    let inside = rx >= 0
        && ry >= 0
        && rx as usize + 16 <= reference.width
        && ry as usize + 16 <= reference.height;
    let mut sum = 0;
    for row in 0..16 {
        let source_row = &source.data[(y0 + row) * source.width + x0..][..16];
        if inside {
            let start = (ry as usize + row) * reference.width + rx as usize;
            let reference_row = &reference.data[start..start + 16];
            for (a, b) in source_row.iter().zip(reference_row) {
                sum += u32::from(a.abs_diff(*b));
            }
        } else {
            let y = (ry + row as i32).clamp(0, reference.height as i32 - 1) as usize;
            for (column, &a) in source_row.iter().enumerate() {
                let x = (rx + column as i32).clamp(0, reference.width as i32 - 1) as usize;
                sum += u32::from(a.abs_diff(reference.data[y * reference.width + x]));
            }
        }
    }
    sum
}

fn sad16(source: &Plane, prediction: &Plane, origin: usize) -> u32 {
    let stride = source.width;
    let mut sum = 0;
    for row in 0..16 {
        let start = origin + row * stride;
        for (a, b) in source.data[start..start + 16]
            .iter()
            .zip(&prediction.data[start..start + 16])
        {
            sum += u32::from(a.abs_diff(*b));
        }
    }
    sum
}

/// The residual of the 4x4 block at `offset` of two planes of one size.
fn residual(source: &Plane, prediction: &Plane, offset: usize) -> [i16; 16] {
    let stride = source.width;
    let mut block = [0i16; 16];
    for row in 0..4 {
        for column in 0..4 {
            let at = offset + row * stride + column;
            block[row * 4 + column] = i16::from(source.data[at]) - i16::from(prediction.data[at]);
        }
    }
    block
}

/// The sum of absolute 4x4 Hadamard-transformed differences, halved: a
/// cheap estimate of a residual's coding cost.
fn satd4(block: &[i16; 16]) -> u32 {
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let r = &block[row * 4..row * 4 + 4];
        let (a, b, c, d) = (
            i32::from(r[0]),
            i32::from(r[1]),
            i32::from(r[2]),
            i32::from(r[3]),
        );
        let (s0, s1, d0, d1) = (a + b, c + d, a - b, c - d);
        temp[row * 4] = s0 + s1;
        temp[row * 4 + 1] = s0 - s1;
        temp[row * 4 + 2] = d0 + d1;
        temp[row * 4 + 3] = d0 - d1;
    }
    let mut sum = 0u32;
    for column in 0..4 {
        let (a, b, c, d) = (
            temp[column],
            temp[4 + column],
            temp[8 + column],
            temp[12 + column],
        );
        let (s0, s1, d0, d1) = (a + b, c + d, a - b, c - d);
        sum += (s0 + s1).unsigned_abs()
            + (s0 - s1).unsigned_abs()
            + (d0 + d1).unsigned_abs()
            + (d0 - d1).unsigned_abs();
    }
    sum / 2
}

/// The SATD of the `N`x`N` block at `origin` of two planes of one size.
fn satd<const N: usize>(source: &Plane, prediction: &Plane, origin: usize) -> u32 {
    let stride = source.width;
    let mut sum = 0;
    for block_y in 0..N / 4 {
        for block_x in 0..N / 4 {
            let offset = origin + block_y * 4 * stride + block_x * 4;
            sum += satd4(&residual(source, prediction, offset));
        }
    }
    sum
}

fn satd16(source: &Plane, prediction: &Plane, origin: usize) -> u32 {
    satd::<16>(source, prediction, origin)
}

/// Chooses the whole-macroblock luma mode, or `B_PRED` when coding the
/// subblocks separately is estimated to be cheaper, returning it and its
/// cost. `probabilities` are the inter-frame luma mode probabilities, or
/// `None` for a key frame. The macroblock's luma is left unspecified.
fn choose_luma_mode(
    source: &Source,
    frame: &mut Frame,
    quantizer: &Quantizer,
    probabilities: Option<[u8; 4]>,
    mb_x: usize,
    mb_y: usize,
) -> (u8, u64) {
    let plane = &mut frame.planes[0];
    let stride = plane.width;
    let origin = mb_y * 16 * stride + mb_x * 16;
    let edges = macroblock_edges::<16, 21>(plane, mb_x, mb_y);
    let mode_bits = |mode: u8| match probabilities {
        Some(probabilities) => tree_cost(&Y_MODE_TREE, &probabilities, mode),
        None => tree_cost(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, mode),
    };
    let mut chosen = (DC_PRED, u64::MAX);
    for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
        predict_block(
            mode,
            &edges,
            mb_y > 0,
            mb_x > 0,
            &mut plane.data,
            origin,
            stride,
        );
        let cost = quantizer.cost(satd16(&source[0], plane, origin), mode_bits(mode));
        if cost < chosen.1 {
            chosen = (mode, cost);
        }
    }

    // Estimate B_PRED by choosing each subblock's mode on the prediction of
    // the source rather than of the reconstruction: close enough to decide,
    // and much cheaper than coding all sixteen.
    let mut subblock_cost = quantizer.cost(0, mode_bits(B_PRED));
    for block in 0..16 {
        let offset = origin + (block >> 2) * 4 * stride + (block & 3) * 4;
        let (above, left) = subblock_edges(&source[0], &edges, origin, block);
        let mut best = u64::MAX;
        for mode in 0..10 {
            let mut prediction = [0u8; 16];
            predict_subblock(mode, &above, &left, &mut prediction, 0, 4);
            let mut difference = [0i16; 16];
            for (index, value) in difference.iter_mut().enumerate() {
                let at = offset + (index >> 2) * stride + (index & 3);
                *value = i16::from(source[0].data[at]) - i16::from(prediction[index]);
            }
            // Roughly the cost of a subblock mode with context.
            let bits = tree_cost(&B_MODE_TREE, &DEFAULT_B_MODE_PROBS, mode);
            best = best.min(quantizer.cost(satd4(&difference), bits));
        }
        subblock_cost += best;
        if subblock_cost >= chosen.1 {
            break;
        }
    }
    if subblock_cost < chosen.1 {
        chosen = (B_PRED, subblock_cost);
    }
    chosen
}

/// Chooses, predicts, quantizes and reconstructs each luma subblock of a
/// `B_PRED` macroblock, returning their modes. `contexts` are the above and
/// left macroblocks' subblock modes on a key frame, whose mode probabilities
/// depend on them.
#[allow(clippy::too_many_arguments)]
fn code_subblocks(
    source: &Source,
    plane: &mut Plane,
    quantizer: &Quantizer,
    contexts: Option<([u8; 16], [u8; 16])>,
    mb_x: usize,
    mb_y: usize,
    levels: &mut Coefficients,
    dequantized: &mut Coefficients,
) -> [u8; 16] {
    let stride = plane.width;
    let origin = mb_y * 16 * stride + mb_x * 16;
    let edges = macroblock_edges::<16, 21>(plane, mb_x, mb_y);
    let mut modes = [B_DC_PRED; 16];
    for block in 0..16 {
        let offset = origin + (block >> 2) * 4 * stride + (block & 3) * 4;
        let (above, left) = subblock_edges(plane, &edges, origin, block);
        let probabilities = match contexts {
            Some((above_modes, left_modes)) => {
                let above_mode = if block < 4 {
                    above_modes[block + 12]
                } else {
                    modes[block - 4]
                };
                let left_mode = if block & 3 == 0 {
                    left_modes[block + 3]
                } else {
                    modes[block - 1]
                };
                &KF_B_MODE_PROBS[usize::from(above_mode)][usize::from(left_mode)]
            }
            None => &DEFAULT_B_MODE_PROBS,
        };
        let mut chosen = (B_DC_PRED, u64::MAX);
        for mode in 0..10 {
            predict_subblock(mode, &above, &left, &mut plane.data, offset, stride);
            let cost = quantizer.cost(
                satd4(&residual(&source[0], plane, offset)),
                tree_cost(&B_MODE_TREE, probabilities, mode),
            );
            if cost < chosen.1 {
                chosen = (mode, cost);
            }
        }
        modes[block] = chosen.0;
        predict_subblock(chosen.0, &above, &left, &mut plane.data, offset, stride);
        let coefficients = forward_dct(&residual(&source[0], plane, offset));
        (levels[block], dequantized[block]) = quantize(&coefficients, quantizer.y1, 0);
        if dequantized[block].iter().any(|&value| value != 0) {
            idct_add(&dequantized[block], &mut plane.data, offset, stride);
        }
    }
    modes
}

/// Chooses the chroma mode of an intra macroblock and leaves its prediction
/// in `frame`.
fn choose_chroma_mode(
    source: &Source,
    frame: &mut Frame,
    quantizer: &Quantizer,
    probabilities: &[u8; 3],
    mb_x: usize,
    mb_y: usize,
) -> u8 {
    let edges = [1, 2].map(|plane| macroblock_edges::<8, 9>(&frame.planes[plane], mb_x, mb_y));
    let predict = |mode: u8, frame: &mut Frame| {
        for (plane, edges) in [1, 2].into_iter().zip(&edges) {
            let stride = frame.planes[plane].width;
            predict_block(
                mode,
                edges,
                mb_y > 0,
                mb_x > 0,
                &mut frame.planes[plane].data,
                mb_y * 8 * stride + mb_x * 8,
                stride,
            );
        }
    };
    let mut chosen = (DC_PRED, u64::MAX);
    for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
        predict(mode, frame);
        let distortion: u32 = [1, 2]
            .into_iter()
            .map(|plane| {
                let stride = frame.planes[plane].width;
                satd::<8>(
                    &source[plane],
                    &frame.planes[plane],
                    mb_y * 8 * stride + mb_x * 8,
                )
            })
            .sum();
        let cost = quantizer.cost(distortion, tree_cost(&UV_MODE_TREE, probabilities, mode));
        if cost < chosen.1 {
            chosen = (mode, cost);
        }
    }
    predict(chosen.0, frame);
    chosen.0
}

/// Transforms and quantizes the luma residual of a macroblock coded with a
/// Y2 block, whose 16 DC coefficients go through the Walsh-Hadamard
/// transform.
fn quantize_luma_with_y2(
    source: &Plane,
    prediction: &Plane,
    origin: usize,
    quantizer: &Quantizer,
    levels: &mut Coefficients,
    dequantized: &mut Coefficients,
) {
    let stride = source.width;
    let mut dc = [0i16; 16];
    for block in 0..16 {
        let offset = origin + (block >> 2) * 4 * stride + (block & 3) * 4;
        let coefficients = forward_dct(&residual(source, prediction, offset));
        dc[block] = coefficients[0];
        (levels[block], dequantized[block]) = quantize(&coefficients, quantizer.y1, 1);
    }
    (levels[24], dequantized[24]) = quantize(&forward_walsh(&dc), quantizer.y2, 0);
}

/// Quantizes a block's coefficients (raster order) from zigzag position
/// `first`, returning the levels and their dequantized values.
fn quantize(coefficients: &[i16; 16], factors: [i32; 2], first: usize) -> ([i16; 16], [i16; 16]) {
    let mut levels = [0i16; 16];
    let mut dequantized = [0i16; 16];
    for (index, &position) in ZIGZAG.iter().enumerate().skip(first) {
        let step = factors[usize::from(index > 0)];
        let value = i32::from(coefficients[position]);
        // A dead zone wider than rounding to nearest: small coefficients
        // cost more to code than they return.
        let bias = if index == 0 { step / 2 } else { step / 4 };
        let level = ((value.abs() + bias) / step).min(2048);
        let level = if value < 0 { -level } else { level };
        levels[position] = level as i16;
        dequantized[position] = (level * step) as i16;
    }
    (levels, dequantized)
}

/// libvpx's forward 4x4 DCT (`vp8_short_fdct4x4_c`), raster order.
fn forward_dct(input: &[i16; 16]) -> [i16; 16] {
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let ip = &input[row * 4..row * 4 + 4];
        let a1 = (i32::from(ip[0]) + i32::from(ip[3])) * 8;
        let b1 = (i32::from(ip[1]) + i32::from(ip[2])) * 8;
        let c1 = (i32::from(ip[1]) - i32::from(ip[2])) * 8;
        let d1 = (i32::from(ip[0]) - i32::from(ip[3])) * 8;
        temp[row * 4] = a1 + b1;
        temp[row * 4 + 2] = a1 - b1;
        temp[row * 4 + 1] = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
        temp[row * 4 + 3] = (d1 * 2217 - c1 * 5352 + 7500) >> 12;
    }
    let mut output = [0i16; 16];
    for column in 0..4 {
        let a1 = temp[column] + temp[12 + column];
        let b1 = temp[4 + column] + temp[8 + column];
        let c1 = temp[4 + column] - temp[8 + column];
        let d1 = temp[column] - temp[12 + column];
        output[column] = ((a1 + b1 + 7) >> 4) as i16;
        output[8 + column] = ((a1 - b1 + 7) >> 4) as i16;
        output[4 + column] = (((c1 * 2217 + d1 * 5352 + 12000) >> 16) + i32::from(d1 != 0)) as i16;
        output[12 + column] = ((d1 * 2217 - c1 * 5352 + 51000) >> 16) as i16;
    }
    output
}

/// libvpx's forward Walsh-Hadamard transform (`vp8_short_walsh4x4_c`) of
/// the 16 luma DC coefficients in raster order.
fn forward_walsh(input: &[i16; 16]) -> [i16; 16] {
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let ip = &input[row * 4..row * 4 + 4];
        let a1 = (i32::from(ip[0]) + i32::from(ip[2])) * 4;
        let d1 = (i32::from(ip[1]) + i32::from(ip[3])) * 4;
        let c1 = (i32::from(ip[1]) - i32::from(ip[3])) * 4;
        let b1 = (i32::from(ip[0]) - i32::from(ip[2])) * 4;
        temp[row * 4] = a1 + d1 + i32::from(a1 != 0);
        temp[row * 4 + 1] = b1 + c1;
        temp[row * 4 + 2] = b1 - c1;
        temp[row * 4 + 3] = a1 - d1;
    }
    let mut output = [0i16; 16];
    for column in 0..4 {
        let a1 = temp[column] + temp[8 + column];
        let d1 = temp[4 + column] + temp[12 + column];
        let c1 = temp[4 + column] - temp[12 + column];
        let b1 = temp[column] - temp[8 + column];
        let round = |value: i32| ((value + i32::from(value < 0) + 3) >> 3) as i16;
        output[column] = round(a1 + d1);
        output[4 + column] = round(b1 + c1);
        output[8 + column] = round(b1 - c1);
        output[12 + column] = round(a1 - d1);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::super::predict::inverse_walsh;
    use super::*;

    #[test]
    fn forward_transforms_invert_the_decoder_transforms() {
        let mut state = 7u32;
        for _ in 0..200 {
            let mut block = [0i16; 16];
            for value in &mut block {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *value = ((state >> 16) % 511) as i16 - 255;
            }
            let mut plane = vec![128u8; 16];
            let mut expected = [0u8; 16];
            for (index, value) in expected.iter_mut().enumerate() {
                *value = (128 + i32::from(block[index])).clamp(0, 255) as u8;
            }
            idct_add(&forward_dct(&block), &mut plane, 0, 4);
            for (got, want) in plane.iter().zip(expected) {
                assert!(got.abs_diff(want) <= 1, "{got} vs {want}");
            }

            let dc: [i16; 16] = block.map(|value| value * 8);
            let round_trip = inverse_walsh(&forward_walsh(&dc));
            for (got, want) in round_trip.iter().zip(dc) {
                assert!((got - want).abs() <= 1, "{got} vs {want}");
            }
        }
    }

    /// A moving pattern with texture, edges and a sliding square.
    fn source(width: usize, height: usize, index: usize) -> Source {
        let (padded_width, padded_height) = (width.div_ceil(16) * 16, height.div_ceil(16) * 16);
        let mut planes = [
            Plane::new(padded_width, padded_height),
            Plane::new(padded_width / 2, padded_height / 2),
            Plane::new(padded_width / 2, padded_height / 2),
        ];
        for (plane_index, plane) in planes.iter_mut().enumerate() {
            let scale = if plane_index == 0 { 1 } else { 2 };
            for y in 0..plane.height {
                for x in 0..plane.width {
                    let (sx, sy) = (
                        (x * scale).min(width - 1) + index * 2,
                        (y * scale).min(height - 1) + index,
                    );
                    let square = (sx / 3 + 7) % 40 < 12 && (sy + 5) % 30 < 10;
                    let value = if square {
                        200 + plane_index * 15
                    } else {
                        40 + (sx * 3 + sy * 2) % 120 + ((sx / 5 + sy / 7) % 2) * 30
                            - plane_index * 10
                    };
                    plane.data[y * plane.width + x] = value as u8;
                }
            }
        }
        planes
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        10.0 * (255.0 * 255.0 / error.max(1e-9)).log10()
    }

    #[test]
    fn the_decoder_reconstructs_exactly_what_the_encoder_predicts_from() {
        let (width, height) = (70, 50);
        for q in [0u8, 10, 24, 60, 127] {
            let mut encoder = FrameEncoder::new(width, height);
            let mut decoder = super::super::decoder::Decoder::new(crate::Limits::default());
            let mut total_psnr = 0.0;
            for index in 0..12 {
                let source = source(width, height, index);
                let data = encoder.encode(&source, index % 6 == 0, q).unwrap();
                assert_eq!(data[0] & 1 == 0, index % 6 == 0, "frame type");
                let picture = decoder.decode(&data).unwrap().expect("a shown frame");
                let expected = encoder.reconstruction();
                assert_eq!(picture.planes, expected, "q {q}, frame {index}");
                let luma: Vec<u8> = (0..height)
                    .flat_map(|row| &source[0].data[row * source[0].width..][..width])
                    .copied()
                    .collect();
                total_psnr += psnr(&luma, &picture.planes[0]);
            }
            let average = total_psnr / 12.0;
            let floor = match q {
                0 => 45.0,
                10 => 40.0,
                24 => 34.0,
                60 => 27.0,
                _ => 18.0,
            };
            assert!(average > floor, "q {q}: {average:.1} dB");
        }
    }

    #[test]
    fn inter_frames_code_motion_instead_of_pictures() {
        let (width, height) = (128, 96);
        let mut encoder = FrameEncoder::new(width, height);
        let key = encoder.encode(&source(width, height, 0), true, 24).unwrap();
        let still = encoder
            .encode(&source(width, height, 0), false, 24)
            .unwrap();
        let moved = encoder
            .encode(&source(width, height, 1), false, 24)
            .unwrap();
        // An unchanged picture skips nearly every macroblock.
        assert!(still.len() * 20 < key.len(), "still {} bytes", still.len());
        assert!(
            moved.len() * 2 < key.len(),
            "key {} bytes, moved {} bytes",
            key.len(),
            moved.len()
        );
    }

    #[test]
    fn motion_vector_costs_grow_with_distance() {
        let zero = MotionVector::ZERO;
        let near = MotionVector { x: 2, y: 0 };
        let far = MotionVector { x: 400, y: -200 };
        assert!(mv_cost(zero, zero) < mv_cost(near, zero));
        assert!(mv_cost(near, zero) < mv_cost(far, zero));
    }
}
