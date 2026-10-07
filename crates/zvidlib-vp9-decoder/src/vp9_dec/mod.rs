//! A dependency-free VP9 profile 0 (8-bit 4:2:0) decoder.
//!
//! The decoding process is the one of the VP9 Bitstream & Decoding Process
//! Specification, implemented to match libvpx, the reference decoder, bit
//! for bit: `bits` holds the two bitstream readers, this module the
//! uncompressed and compressed headers (sections 6.2 and 6.3) and the
//! frame-level state they update, `block` the tile syntax, motion vector
//! prediction and residual decoding (sections 6.4 and 8.4), `recon` the
//! prediction and reconstruction kernels (sections 8.5 to 8.7) with the
//! one-dimensional transforms in `idct1d`, and `loopfilter` the loop filter
//! (section 8.8). `probs` holds the probability contexts and their
//! adaptation, and `tables` the constant tables, generated from libvpx's.
//! The reconstruction kernels and the loop filter run the bit-exact vector
//! kernels of [`crate::vp9_simd`] where the host has SSE4.1, AVX2 or NEON,
//! and their scalar forms here everywhere else.
//!
//! A sample is a VP9 *chunk*: one frame, or a superframe of several frames
//! with an index at its end (Annex B), of which at most one is shown.
//! Hidden frames (an alternate reference typically travels in the same
//! superframe as the frame shown after it), `show_existing_frame`,
//! intra-only frames, reference frames of a different size (scaled motion
//! compensation), tiles, segmentation, lossless coding and every
//! interpolation filter are decoded. Malformed input returns
//! [`ErrorKind::MalformedMedia`] rather than panicking.

// The processes below follow libvpx's C, which indexes several parallel
// arrays with one loop variable; keeping that shape makes each function
// checkable line by line against the code it reproduces.
#![allow(clippy::needless_range_loop)]

mod bits;
mod block;
// What the hardware backends read from a chunk before handing it to the platform decoder.
#[cfg(any(
    test,
    windows,
    all(target_os = "linux", target_pointer_width = "64"),
    target_os = "macos"
))]
mod chunk;
pub(crate) mod idct1d;
pub(crate) mod loopfilter;
mod probs;
pub(crate) mod recon;
pub(crate) mod tables;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use bits::{BitReader, BoolDecoder};
use block::{FrameDecoder, MvRef};
use probs::{FrameContext, FrameCounts};

#[cfg(any(
    test,
    windows,
    all(target_os = "linux", target_pointer_width = "64"),
    target_os = "macos"
))]
pub(crate) use chunk::{ChunkInspector, chunk_frames};
// The NVDEC and Media Foundation backends name the shape they carry to readback.
#[cfg(any(test, windows, all(target_os = "linux", target_pointer_width = "64")))]
pub(crate) use chunk::FrameShape;

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

/// The border, in pixels, kept around every plane so the 8-tap filters can
/// read the taps a whole-pixel position multiplies by zero without bounds
/// checks of their own.
const BORDER: usize = 8;

/// One plane of a decoded frame.
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    pub(crate) data: Vec<u8>,
    pub(crate) stride: usize,
    /// The index of the top-left pixel in `data`.
    pub(crate) origin: usize,
    /// The decoded area, aligned to 8 luma pixels (libvpx's `y_width`).
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// The visible area (libvpx's `y_crop_width`).
    pub(crate) crop_width: usize,
    pub(crate) crop_height: usize,
}

impl Plane {
    fn new(width: usize, height: usize, crop_width: usize, crop_height: usize, sb: usize) -> Self {
        // Blocks may extend past the decoded area to the end of their
        // superblock, so the allocation covers whole superblocks.
        let alloc_width = width.div_ceil(sb) * sb;
        let alloc_height = height.div_ceil(sb) * sb;
        let stride = alloc_width + 2 * BORDER;
        Self {
            data: vec![0; stride * (alloc_height + 2 * BORDER)],
            stride,
            origin: BORDER * stride + BORDER,
            width,
            height,
            crop_width,
            crop_height,
        }
    }

    #[inline]
    pub(crate) fn index(&self, x: usize, y: usize) -> usize {
        self.origin + y * self.stride + x
    }
}

/// A decoded frame: the buffer a reference slot holds.
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) planes: [Plane; 3],
    pub(crate) color_space: u8,
    pub(crate) full_range: bool,
}

impl Frame {
    fn new(width: usize, height: usize, color_space: u8, full_range: bool) -> Self {
        let aligned_width = (width + 7) & !7;
        let aligned_height = (height + 7) & !7;
        let luma = Plane::new(aligned_width, aligned_height, width, height, 64);
        let chroma = || {
            Plane::new(
                aligned_width >> 1,
                aligned_height >> 1,
                (width + 1) >> 1,
                (height + 1) >> 1,
                32,
            )
        };
        Self {
            width,
            height,
            planes: [luma, chroma(), chroma()],
            color_space,
            full_range,
        }
    }

    /// The visible picture, each plane tightly packed.
    pub(crate) fn picture(&self) -> DecodedPicture {
        let planes = std::array::from_fn(|index| {
            let plane = &self.planes[index];
            let mut packed = Vec::with_capacity(plane.crop_width * plane.crop_height);
            for y in 0..plane.crop_height {
                let start = plane.index(0, y);
                packed.extend_from_slice(&plane.data[start..start + plane.crop_width]);
            }
            packed
        });
        DecodedPicture {
            width: self.width,
            height: self.height,
            planes,
            color_space: self.color_space,
            full_range: self.full_range,
        }
    }
}

/// A shown picture, as the output process produces it.
#[derive(Clone, Debug)]
pub(crate) struct DecodedPicture {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Y, U and V, tightly packed; the chroma planes are subsampled 2x2.
    pub(crate) planes: [Vec<u8>; 3],
    /// `color_space` of the uncompressed header (`CS_BT_601` is 1,
    /// `CS_BT_709` 2 and so on).
    pub(crate) color_space: u8,
    pub(crate) full_range: bool,
}

/// The segmentation parameters, which persist from frame to frame
/// (section 7.2.10).
#[derive(Clone, Debug, Default)]
pub(crate) struct Segmentation {
    pub(crate) enabled: bool,
    pub(crate) update_map: bool,
    pub(crate) temporal_update: bool,
    pub(crate) abs_delta: bool,
    pub(crate) tree_probs: [u8; 7],
    pub(crate) pred_probs: [u8; 3],
    pub(crate) feature_enabled: [[bool; 4]; 8],
    pub(crate) feature_data: [[i16; 4]; 8],
}

pub(crate) const SEG_LVL_ALT_Q: usize = 0;
pub(crate) const SEG_LVL_ALT_LF: usize = 1;
pub(crate) const SEG_LVL_REF_FRAME: usize = 2;
pub(crate) const SEG_LVL_SKIP: usize = 3;

impl Segmentation {
    pub(crate) fn feature_active(&self, segment: u8, feature: usize) -> bool {
        self.enabled && self.feature_enabled[usize::from(segment)][feature]
    }

    pub(crate) fn data(&self, segment: u8, feature: usize) -> i32 {
        i32::from(self.feature_data[usize::from(segment)][feature])
    }

    fn clear_features(&mut self) {
        self.feature_enabled = [[false; 4]; 8];
        self.feature_data = [[0; 4]; 8];
    }
}

/// The interpolation filter literal of the uncompressed header that means
/// "chosen per block".
pub(crate) const SWITCHABLE: u8 = 4;

/// Reference-mode values of the compressed header.
pub(crate) const SINGLE_REFERENCE: u8 = 0;
pub(crate) const COMPOUND_REFERENCE: u8 = 1;
pub(crate) const REFERENCE_MODE_SELECT: u8 = 2;

/// The transform mode that selects a size per block.
pub(crate) const TX_MODE_SELECT: u8 = 4;

/// A reference frame's motion compensation scale (`struct scale_factors`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scale {
    pub(crate) x_scale_fp: i32,
    pub(crate) y_scale_fp: i32,
    pub(crate) x_step_q4: i32,
    pub(crate) y_step_q4: i32,
}

const REF_SCALE_SHIFT: u32 = 14;
const REF_NO_SCALE: i32 = 1 << REF_SCALE_SHIFT;
const REF_INVALID_SCALE: i32 = -1;

impl Scale {
    /// `vp9_setup_scale_factors_for_frame`.
    fn new(other_width: usize, other_height: usize, width: usize, height: usize) -> Self {
        if !valid_ref_frame_size(other_width, other_height, width, height) {
            return Self {
                x_scale_fp: REF_INVALID_SCALE,
                y_scale_fp: REF_INVALID_SCALE,
                x_step_q4: 16,
                y_step_q4: 16,
            };
        }
        let x_scale_fp = ((other_width << REF_SCALE_SHIFT) / width) as i32;
        let y_scale_fp = ((other_height << REF_SCALE_SHIFT) / height) as i32;
        let mut scale = Self {
            x_scale_fp,
            y_scale_fp,
            x_step_q4: 16,
            y_step_q4: 16,
        };
        scale.x_step_q4 = scale.scale_x(16);
        scale.y_step_q4 = scale.scale_y(16);
        scale
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.x_scale_fp != REF_INVALID_SCALE && self.y_scale_fp != REF_INVALID_SCALE
    }

    pub(crate) fn is_scaled(&self) -> bool {
        self.is_valid() && (self.x_scale_fp != REF_NO_SCALE || self.y_scale_fp != REF_NO_SCALE)
    }

    pub(crate) fn scale_x(&self, value: i32) -> i32 {
        ((i64::from(value) * i64::from(self.x_scale_fp)) >> REF_SCALE_SHIFT) as i32
    }

    pub(crate) fn scale_y(&self, value: i32) -> i32 {
        ((i64::from(value) * i64::from(self.y_scale_fp)) >> REF_SCALE_SHIFT) as i32
    }
}

fn valid_ref_frame_size(ref_width: usize, ref_height: usize, width: usize, height: usize) -> bool {
    2 * width >= ref_width
        && 2 * height >= ref_height
        && width <= 16 * ref_width
        && height <= 16 * ref_height
}

/// The uncompressed and compressed header of one frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameHeader {
    pub(crate) key_frame: bool,
    pub(crate) show_frame: bool,
    pub(crate) error_resilient: bool,
    pub(crate) intra_only: bool,
    pub(crate) reset_frame_context: u8,
    pub(crate) refresh_frame_flags: u8,
    pub(crate) ref_frame_idx: [usize; 3],
    /// Indexed by reference frame (`LAST_FRAME` is 1).
    pub(crate) ref_frame_sign_bias: [bool; 4],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) allow_high_precision_mv: bool,
    pub(crate) interp_filter: u8,
    pub(crate) refresh_frame_context: bool,
    pub(crate) frame_parallel_decoding_mode: bool,
    pub(crate) frame_context_idx: usize,
    pub(crate) filter_level: u8,
    pub(crate) sharpness: u8,
    pub(crate) base_qindex: u8,
    pub(crate) y_dc_delta_q: i32,
    pub(crate) uv_dc_delta_q: i32,
    pub(crate) uv_ac_delta_q: i32,
    pub(crate) lossless: bool,
    pub(crate) log2_tile_cols: u32,
    pub(crate) log2_tile_rows: u32,
    pub(crate) header_size: usize,
    pub(crate) tx_mode: u8,
    pub(crate) reference_mode: u8,
    pub(crate) comp_fixed_ref: i8,
    pub(crate) comp_var_ref: [i8; 2],
}

impl FrameHeader {
    pub(crate) fn is_intra_only(&self) -> bool {
        self.key_frame || self.intra_only
    }
}

/// The decoder: the reference frames and every piece of state the
/// bitstream carries from one frame to the next.
pub(crate) struct Decoder {
    limits: Limits,
    refs: [Option<Arc<Frame>>; 8],
    frame_contexts: Box<[FrameContext; 4]>,
    segmentation: Segmentation,
    lf_mode_ref_delta_enabled: bool,
    lf_ref_deltas: [i8; 4],
    lf_mode_deltas: [i8; 2],
    last_frame_seg_map: Vec<u8>,
    current_frame_seg_map: Vec<u8>,
    /// `cm->width`/`cm->height`: the size the context buffers have.
    width: usize,
    height: usize,
    last_width: usize,
    last_height: usize,
    last_show_frame: bool,
    /// `cm->frame_type` and `cm->intra_only` of the last frame decoded
    /// (not of `show_existing_frame` headers, which leave them alone).
    frame_was_key: bool,
    intra_only: bool,
    prev_mvs: Option<Arc<Vec<MvRef>>>,
    color_space: u8,
    full_range: bool,
    /// Whether a key frame has been decoded since the last reset.
    have_key_frame: bool,
    output_wanted: bool,
    frames_shown: u64,
}

impl Decoder {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            limits,
            refs: Default::default(),
            frame_contexts: Box::new(std::array::from_fn(|_| FrameContext::default())),
            segmentation: Segmentation::default(),
            lf_mode_ref_delta_enabled: false,
            lf_ref_deltas: [1, 0, -1, -1],
            lf_mode_deltas: [0, 0],
            last_frame_seg_map: Vec::new(),
            current_frame_seg_map: Vec::new(),
            width: 0,
            height: 0,
            last_width: 0,
            last_height: 0,
            last_show_frame: false,
            frame_was_key: true,
            intra_only: false,
            prev_mvs: None,
            color_space: 0,
            full_range: false,
            have_key_frame: false,
            output_wanted: true,
            frames_shown: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        let output_wanted = self.output_wanted;
        *self = Self::new(self.limits);
        self.output_wanted = output_wanted;
    }

    pub(crate) fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
    }

    /// How many frames have been shown since the decoder was created or
    /// reset.
    pub(crate) fn frames_shown(&self) -> u64 {
        self.frames_shown
    }

    /// Decodes one chunk (a frame or a superframe) and returns the picture
    /// it shows, if any and if output is wanted.
    ///
    /// Like libvpx, a chunk shows at most one picture: that of its last
    /// frame, when that frame is shown. A superframe normally carries
    /// hidden frames followed by the one frame it shows; with spatial
    /// layers every layer is marked shown, and the last is the one to
    /// present.
    pub(crate) fn decode_chunk(&mut self, data: &[u8]) -> Result<Option<DecodedPicture>> {
        if data.is_empty() {
            return Err(malformed("VP9 sample is empty"));
        }
        let mut shown = None;
        match superframe_index(data)? {
            Some(sizes) => {
                let mut offset = 0;
                for size in sizes {
                    let end = offset + size;
                    if size == 0 || end > data.len() {
                        return Err(malformed(
                            "VP9 superframe index names an invalid frame size",
                        ));
                    }
                    shown = self.decode_frame(&data[offset..end])?.0;
                    offset = end;
                }
            }
            None => {
                // libvpx decodes frames back to back until the data is used
                // up, skipping zero padding between them.
                let mut offset = 0;
                while offset < data.len() {
                    let (frame, consumed) = self.decode_frame(&data[offset..])?;
                    shown = frame;
                    offset += consumed.max(1);
                    while offset < data.len() && data[offset] == 0 {
                        offset += 1;
                    }
                }
            }
        }
        Ok(shown.and_then(|frame| {
            self.frames_shown += 1;
            self.output_wanted.then(|| frame.picture())
        }))
    }

    /// Decodes one frame, returning it if it is shown and how many bytes
    /// of `data` it used.
    fn decode_frame(&mut self, data: &[u8]) -> Result<(Option<Arc<Frame>>, usize)> {
        let mut reader = BitReader::new(data);
        if reader.literal(2)? != 2 {
            return Err(malformed("VP9 frame marker is invalid"));
        }
        let mut profile = reader.literal(1)? | (reader.literal(1)? << 1);
        if profile > 2 {
            profile += reader.literal(1)?;
        }
        if profile != 0 {
            return Err(unsupported(format!(
                "VP9 profile {profile} is not supported; the decoder implements profile 0 (8-bit 4:2:0)"
            )));
        }
        if reader.bit()? {
            // show_existing_frame
            let index = reader.literal(3)? as usize;
            let frame = self.refs[index].clone().ok_or_else(|| {
                malformed("VP9 show_existing_frame names an empty reference slot")
            })?;
            return Ok((Some(frame), 1));
        }

        let mut header = FrameHeader {
            key_frame: !reader.bit()?,
            show_frame: reader.bit()?,
            error_resilient: reader.bit()?,
            ..FrameHeader::default()
        };
        let last_frame_was_key = self.frame_was_key;
        let last_intra_only = self.intra_only;
        let mut scales = [None; 3];

        if header.key_frame {
            read_sync_code(&mut reader)?;
            self.read_color_config(&mut reader)?;
            header.refresh_frame_flags = 0xff;
            self.read_frame_size(&mut reader, &mut header)?;
            self.have_key_frame = true;
        } else {
            header.intra_only = if header.show_frame {
                false
            } else {
                reader.bit()?
            };
            header.reset_frame_context = if header.error_resilient {
                0
            } else {
                reader.literal(2)? as u8
            };
            if header.intra_only {
                read_sync_code(&mut reader)?;
                // Profile 0 intra-only frames are BT.601 studio-range 4:2:0.
                self.color_space = 1;
                self.full_range = false;
                header.refresh_frame_flags = reader.literal(8)? as u8;
                self.read_frame_size(&mut reader, &mut header)?;
                self.have_key_frame = true;
            } else {
                if !self.have_key_frame {
                    return Err(malformed(
                        "VP9 inter frame received before a key frame or intra-only frame",
                    ));
                }
                header.refresh_frame_flags = reader.literal(8)? as u8;
                for i in 0..3 {
                    let index = reader.literal(3)? as usize;
                    if self.refs[index].is_none() {
                        return Err(malformed("VP9 frame references an empty reference slot"));
                    }
                    header.ref_frame_idx[i] = index;
                    header.ref_frame_sign_bias[1 + i] = reader.bit()?;
                }
                self.read_frame_size_with_refs(&mut reader, &mut header)?;
                header.allow_high_precision_mv = reader.bit()?;
                header.interp_filter = if reader.bit()? {
                    SWITCHABLE
                } else {
                    // literal_to_filter: smooth, regular, sharp, bilinear.
                    [1, 0, 2, 3][reader.literal(2)? as usize]
                };
                for i in 0..3 {
                    let reference = self.refs[header.ref_frame_idx[i]]
                        .as_ref()
                        .expect("checked");
                    scales[i] = Some(Scale::new(
                        reference.width,
                        reference.height,
                        header.width,
                        header.height,
                    ));
                }
            }
        }

        if !header.error_resilient {
            header.refresh_frame_context = reader.bit()?;
            header.frame_parallel_decoding_mode = reader.bit()?;
        } else {
            header.refresh_frame_context = false;
            header.frame_parallel_decoding_mode = true;
        }
        header.frame_context_idx = reader.literal(2)? as usize;

        if header.is_intra_only() || header.error_resilient {
            self.setup_past_independence(&mut header);
        }

        self.read_loop_filter(&mut reader, &mut header)?;
        read_quantization(&mut reader, &mut header)?;
        self.read_segmentation(&mut reader)?;
        read_tile_info(&mut reader, &mut header)?;
        header.header_size = reader.literal(16)? as usize;
        if header.header_size == 0 {
            return Err(malformed("VP9 compressed header size is zero"));
        }
        let compressed_start = reader.bytes_read();
        let compressed_end = compressed_start + header.header_size;
        if compressed_end > data.len() {
            return Err(malformed("VP9 compressed header is truncated"));
        }

        let use_prev_frame_mvs = !header.error_resilient
            && header.width == self.last_width
            && header.height == self.last_height
            && !last_intra_only
            && self.last_show_frame
            && !last_frame_was_key;

        let mut fc = self.frame_contexts[header.frame_context_idx].clone();
        read_compressed_header(
            &data[compressed_start..compressed_end],
            &mut header,
            &mut fc,
        )?;

        let mut frame = Frame::new(
            header.width,
            header.height,
            self.color_space,
            self.full_range,
        );
        let mi_cols = (header.width + 7) >> 3;
        let mi_rows = (header.height + 7) >> 3;
        let references: [Option<(Arc<Frame>, Scale)>; 3] = std::array::from_fn(|i| {
            scales[i].map(|scale| {
                (
                    self.refs[header.ref_frame_idx[i]].clone().expect("checked"),
                    scale,
                )
            })
        });
        let prev_mvs = if use_prev_frame_mvs {
            self.prev_mvs.clone()
        } else {
            None
        };
        let counting = !header.error_resilient && !header.frame_parallel_decoding_mode;
        let mut decoder = FrameDecoder::new(
            &header,
            &fc,
            &self.segmentation,
            self.lf_mode_ref_delta_enabled,
            self.lf_ref_deltas,
            self.lf_mode_deltas,
            &references,
            prev_mvs.as_deref().map(Vec::as_slice),
            &self.last_frame_seg_map,
            &mut self.current_frame_seg_map,
            &mut frame,
            mi_rows,
            mi_cols,
            counting,
        )?;
        let tiles_end = compressed_end + decoder.decode_tiles(&data[compressed_end..])?;
        let counts = decoder.counts.take();
        let cur_mvs = std::mem::take(&mut decoder.cur_mvs);
        drop(decoder);

        if let Some(counts) = counts {
            adapt(
                &mut fc,
                &self.frame_contexts[header.frame_context_idx],
                &counts,
                &header,
                last_frame_was_key,
            );
        }
        if header.refresh_frame_context {
            self.frame_contexts[header.frame_context_idx] = fc;
        }
        // An intra frame records no motion; the next frame never asks for
        // it (`use_prev_frame_mvs` is false after one).
        self.prev_mvs = (!header.is_intra_only()).then(|| Arc::new(cur_mvs));

        let frame = Arc::new(frame);
        for (slot, reference) in self.refs.iter_mut().enumerate() {
            if header.refresh_frame_flags & (1 << slot) != 0 {
                *reference = Some(frame.clone());
            }
        }
        self.last_show_frame = header.show_frame;
        if self.segmentation.enabled {
            std::mem::swap(
                &mut self.last_frame_seg_map,
                &mut self.current_frame_seg_map,
            );
        }
        self.last_width = header.width;
        self.last_height = header.height;
        self.frame_was_key = header.key_frame;
        if !header.key_frame {
            self.intra_only = header.intra_only;
        }
        Ok((header.show_frame.then_some(frame), tiles_end))
    }

    fn read_color_config(&mut self, reader: &mut BitReader) -> Result<()> {
        self.color_space = reader.literal(3)? as u8;
        if self.color_space == 7 {
            return Err(unsupported(
                "VP9 sRGB (4:4:4) color is not supported in profile 0",
            ));
        }
        self.full_range = reader.bit()?;
        Ok(())
    }

    fn read_frame_size(&mut self, reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.width = reader.literal(16)? as usize + 1;
        header.height = reader.literal(16)? as usize + 1;
        self.resize(header.width, header.height)?;
        read_render_size(reader)
    }

    fn read_frame_size_with_refs(
        &mut self,
        reader: &mut BitReader,
        header: &mut FrameHeader,
    ) -> Result<()> {
        let mut found = None;
        for i in 0..3 {
            if reader.bit()? {
                let reference = self.refs[header.ref_frame_idx[i]]
                    .as_ref()
                    .expect("checked");
                found = Some((reference.width, reference.height));
                break;
            }
        }
        let (width, height) = match found {
            Some(size) => size,
            None => (
                reader.literal(16)? as usize + 1,
                reader.literal(16)? as usize + 1,
            ),
        };
        let any_valid = header.ref_frame_idx.iter().any(|&index| {
            let reference = self.refs[index].as_ref().expect("checked");
            valid_ref_frame_size(reference.width, reference.height, width, height)
        });
        if !any_valid {
            return Err(malformed("VP9 frame has no reference of a usable size"));
        }
        header.width = width;
        header.height = height;
        self.resize(width, height)?;
        read_render_size(reader)
    }

    /// `resize_context_buffers`: a new frame size clears the previous
    /// frame's segmentation map.
    fn resize(&mut self, width: usize, height: usize) -> Result<()> {
        if width as u64 > u64::from(self.limits.max_width)
            || height as u64 > u64::from(self.limits.max_height)
        {
            return Err(limit("VP9 frame dimensions exceed configured limits"));
        }
        let luma = (((width + 63) & !63) + 2 * BORDER) as u64
            * (((height + 63) & !63) + 2 * BORDER) as u64;
        if luma.saturating_mul(3) / 2 > self.limits.max_allocation_bytes {
            return Err(limit("VP9 frame exceeds the allocation limit"));
        }
        let mi_count = ((width + 7) >> 3) * ((height + 7) >> 3);
        if width != self.width || height != self.height {
            self.last_frame_seg_map = vec![0; mi_count];
            self.current_frame_seg_map.resize(mi_count, 0);
            self.width = width;
            self.height = height;
        }
        if self.current_frame_seg_map.len() != mi_count {
            self.current_frame_seg_map.resize(mi_count, 0);
        }
        if self.last_frame_seg_map.len() != mi_count {
            self.last_frame_seg_map.resize(mi_count, 0);
        }
        Ok(())
    }

    /// `vp9_setup_past_independence`.
    fn setup_past_independence(&mut self, header: &mut FrameHeader) {
        self.segmentation.clear_features();
        self.segmentation.abs_delta = false;
        self.last_frame_seg_map.fill(0);
        self.current_frame_seg_map.fill(0);
        self.lf_ref_deltas = [1, 0, -1, -1];
        self.lf_mode_deltas = [0, 0];
        self.lf_mode_ref_delta_enabled = true;
        let defaults = FrameContext::default();
        if header.key_frame || header.error_resilient || header.reset_frame_context == 3 {
            for context in self.frame_contexts.iter_mut() {
                *context = defaults.clone();
            }
        } else if header.reset_frame_context == 2 {
            self.frame_contexts[header.frame_context_idx] = defaults;
        }
        header.ref_frame_sign_bias = [false; 4];
        header.frame_context_idx = 0;
    }

    fn read_loop_filter(&mut self, reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.filter_level = reader.literal(6)? as u8;
        header.sharpness = reader.literal(3)? as u8;
        self.lf_mode_ref_delta_enabled = reader.bit()?;
        if self.lf_mode_ref_delta_enabled && reader.bit()? {
            for delta in &mut self.lf_ref_deltas {
                if reader.bit()? {
                    *delta = reader.signed_literal(6)? as i8;
                }
            }
            for delta in &mut self.lf_mode_deltas {
                if reader.bit()? {
                    *delta = reader.signed_literal(6)? as i8;
                }
            }
        }
        Ok(())
    }

    fn read_segmentation(&mut self, reader: &mut BitReader) -> Result<()> {
        let seg = &mut self.segmentation;
        seg.update_map = false;
        seg.enabled = reader.bit()?;
        if !seg.enabled {
            return Ok(());
        }
        seg.update_map = reader.bit()?;
        if seg.update_map {
            for probability in &mut seg.tree_probs {
                *probability = if reader.bit()? {
                    reader.literal(8)? as u8
                } else {
                    255
                };
            }
            seg.temporal_update = reader.bit()?;
            for probability in &mut seg.pred_probs {
                *probability = if seg.temporal_update && reader.bit()? {
                    reader.literal(8)? as u8
                } else {
                    255
                };
            }
        }
        if reader.bit()? {
            seg.abs_delta = reader.bit()?;
            seg.clear_features();
            const MAX: [i32; 4] = [255, 63, 3, 0];
            const BITS: [u32; 4] = [8, 6, 2, 0];
            const SIGNED: [bool; 4] = [true, true, false, false];
            for segment in 0..8 {
                for feature in 0..4 {
                    let mut data = 0;
                    if reader.bit()? {
                        seg.feature_enabled[segment][feature] = true;
                        data = (reader.literal(BITS[feature])? as i32).min(MAX[feature]);
                        if SIGNED[feature] && reader.bit()? {
                            data = -data;
                        }
                    }
                    seg.feature_data[segment][feature] = data as i16;
                }
            }
        }
        Ok(())
    }
}

/// Merges the frame's counts into its probabilities (section 8.4.2).
fn adapt(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &FrameCounts,
    header: &FrameHeader,
    last_frame_was_key: bool,
) {
    probs::adapt_coef_probs(fc, pre, counts, header.is_intra_only(), last_frame_was_key);
    if !header.is_intra_only() {
        probs::adapt_mode_probs(
            fc,
            pre,
            counts,
            header.interp_filter == SWITCHABLE,
            header.tx_mode == TX_MODE_SELECT,
        );
        probs::adapt_mv_probs(fc, pre, counts, header.allow_high_precision_mv);
    }
}

fn read_sync_code(reader: &mut BitReader) -> Result<()> {
    if reader.literal(24)? != 0x49_83_42 {
        return Err(malformed("VP9 frame sync code is invalid"));
    }
    Ok(())
}

fn read_render_size(reader: &mut BitReader) -> Result<()> {
    // The render size is presentation metadata; decoding ignores it.
    if reader.bit()? {
        reader.literal(32)?;
    }
    Ok(())
}

fn read_quantization(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
    header.base_qindex = reader.literal(8)? as u8;
    let mut delta = || -> Result<i32> {
        Ok(if reader.bit()? {
            reader.signed_literal(4)?
        } else {
            0
        })
    };
    header.y_dc_delta_q = delta()?;
    header.uv_dc_delta_q = delta()?;
    header.uv_ac_delta_q = delta()?;
    header.lossless = header.base_qindex == 0
        && header.y_dc_delta_q == 0
        && header.uv_dc_delta_q == 0
        && header.uv_ac_delta_q == 0;
    Ok(())
}

fn read_tile_info(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
    let mi_cols = (header.width + 7) >> 3;
    let sb64_cols = (mi_cols + 7) >> 3;
    let mut min_log2 = 0u32;
    while (64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    header.log2_tile_cols = min_log2;
    let mut max_ones = max_log2.saturating_sub(min_log2);
    while max_ones > 0 && reader.bit()? {
        header.log2_tile_cols += 1;
        max_ones -= 1;
    }
    if header.log2_tile_cols > 6 {
        return Err(malformed("VP9 frame has too many tile columns"));
    }
    header.log2_tile_rows = u32::from(reader.bit()?);
    if header.log2_tile_rows != 0 {
        header.log2_tile_rows += u32::from(reader.bit()?);
    }
    Ok(())
}

/// `vp9_diff_update_prob`.
fn diff_update_prob(bd: &mut BoolDecoder, probability: &mut u8) {
    if bd.read(252) {
        let delta = decode_term_subexp(bd);
        *probability = inv_remap_prob(delta, *probability);
    }
}

fn decode_term_subexp(bd: &mut BoolDecoder) -> u32 {
    if !bd.bit() {
        return bd.literal(4);
    }
    if !bd.bit() {
        return bd.literal(4) + 16;
    }
    if !bd.bit() {
        return bd.literal(5) + 32;
    }
    // decode_uniform
    let m = (1 << 8) - 191;
    let v = bd.literal(7);
    (if v < m {
        v
    } else {
        (v << 1) - m + u32::from(bd.bit())
    }) + 64
}

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

const INV_MAP_TABLE: [u8; 255] = {
    // 7, 20, ..., 254 (every 13th value), then the values in between.
    let mut table = [0u8; 255];
    let mut i = 0;
    while i < 20 {
        table[i] = (7 + 13 * i) as u8;
        i += 1;
    }
    let mut value = 1;
    while i < 254 {
        if (value - 7) % 13 != 0 || value < 7 {
            table[i] = value as u8;
            i += 1;
        }
        value += 1;
    }
    table[254] = 253;
    table
};

fn inv_remap_prob(v: u32, m: u8) -> u8 {
    let v = i32::from(INV_MAP_TABLE[(v as usize).min(254)]);
    let m = i32::from(m) - 1;
    if (m << 1) <= 255 {
        (1 + inv_recenter_nonneg(v, m)) as u8
    } else {
        (255 - inv_recenter_nonneg(v, 255 - 1 - m)) as u8
    }
}

/// `update_mv_probs`.
fn update_mv_prob(bd: &mut BoolDecoder, probability: &mut u8) {
    if bd.read(252) {
        *probability = ((bd.literal(7) << 1) | 1) as u8;
    }
}

/// `read_compressed_header`.
fn read_compressed_header(
    data: &[u8],
    header: &mut FrameHeader,
    fc: &mut FrameContext,
) -> Result<()> {
    let mut bd = BoolDecoder::new(data)?;
    header.tx_mode = if header.lossless {
        0
    } else {
        let mut mode = bd.literal(2) as u8;
        if mode == 3 {
            mode += u8::from(bd.bit());
        }
        mode
    };
    if header.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            diff_update_prob(&mut bd, &mut fc.tx8[i][0]);
        }
        for i in 0..2 {
            for j in 0..2 {
                diff_update_prob(&mut bd, &mut fc.tx16[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                diff_update_prob(&mut bd, &mut fc.tx32[i][j]);
            }
        }
    }
    let max_tx_size = tables::TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(header.tx_mode)];
    for tx_size in 0..=usize::from(max_tx_size) {
        if bd.bit() {
            for plane in 0..2 {
                for reference in 0..2 {
                    for band in 0..6 {
                        let contexts = if band == 0 { 3 } else { 6 };
                        for context in 0..contexts {
                            for node in 0..3 {
                                diff_update_prob(
                                    &mut bd,
                                    &mut fc.coef[tx_size][plane][reference][band][context][node],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    for i in 0..3 {
        diff_update_prob(&mut bd, &mut fc.skip[i]);
    }
    if !header.is_intra_only() {
        for i in 0..7 {
            for j in 0..3 {
                diff_update_prob(&mut bd, &mut fc.inter_mode[i][j]);
            }
        }
        if header.interp_filter == SWITCHABLE {
            for i in 0..4 {
                for j in 0..2 {
                    diff_update_prob(&mut bd, &mut fc.switchable_interp[i][j]);
                }
            }
        }
        for i in 0..4 {
            diff_update_prob(&mut bd, &mut fc.intra_inter[i]);
        }
        // read_frame_reference_mode
        let sign_bias = header.ref_frame_sign_bias;
        let compound_allowed = sign_bias[2] != sign_bias[1] || sign_bias[3] != sign_bias[1];
        header.reference_mode = if compound_allowed && bd.bit() {
            if bd.bit() {
                REFERENCE_MODE_SELECT
            } else {
                COMPOUND_REFERENCE
            }
        } else {
            SINGLE_REFERENCE
        };
        if header.reference_mode != SINGLE_REFERENCE {
            // vp9_setup_compound_reference_mode
            if sign_bias[1] == sign_bias[2] {
                header.comp_fixed_ref = 3;
                header.comp_var_ref = [1, 2];
            } else if sign_bias[1] == sign_bias[3] {
                header.comp_fixed_ref = 2;
                header.comp_var_ref = [1, 3];
            } else {
                header.comp_fixed_ref = 1;
                header.comp_var_ref = [2, 3];
            }
        }
        if header.reference_mode == REFERENCE_MODE_SELECT {
            for i in 0..5 {
                diff_update_prob(&mut bd, &mut fc.comp_inter[i]);
            }
        }
        if header.reference_mode != COMPOUND_REFERENCE {
            for i in 0..5 {
                diff_update_prob(&mut bd, &mut fc.single_ref[i][0]);
                diff_update_prob(&mut bd, &mut fc.single_ref[i][1]);
            }
        }
        if header.reference_mode != SINGLE_REFERENCE {
            for i in 0..5 {
                diff_update_prob(&mut bd, &mut fc.comp_ref[i]);
            }
        }
        for j in 0..4 {
            for i in 0..9 {
                diff_update_prob(&mut bd, &mut fc.y_mode[j][i]);
            }
        }
        for j in 0..16 {
            for i in 0..3 {
                diff_update_prob(&mut bd, &mut fc.partition[j][i]);
            }
        }
        // read_mv_probs
        for probability in &mut fc.mv_joints {
            update_mv_prob(&mut bd, probability);
        }
        for component in &mut fc.mv {
            update_mv_prob(&mut bd, &mut component.sign);
            for probability in &mut component.classes {
                update_mv_prob(&mut bd, probability);
            }
            update_mv_prob(&mut bd, &mut component.class0[0]);
            for probability in &mut component.bits {
                update_mv_prob(&mut bd, probability);
            }
        }
        for component in &mut fc.mv {
            for class0_fp in &mut component.class0_fp {
                for probability in class0_fp {
                    update_mv_prob(&mut bd, probability);
                }
            }
            for probability in &mut component.fp {
                update_mv_prob(&mut bd, probability);
            }
        }
        if header.allow_high_precision_mv {
            for component in &mut fc.mv {
                update_mv_prob(&mut bd, &mut component.class0_hp);
                update_mv_prob(&mut bd, &mut component.hp);
            }
        }
    }
    if bd.has_error() {
        return Err(malformed("VP9 compressed header is corrupt"));
    }
    Ok(())
}

/// The colour range a chunk's first frame signals: `Some(full_range)` when
/// that frame is a profile 0 key frame, or an intra-only frame (which profile
/// 0 makes studio range), and `None` for any other frame or for data that
/// does not parse. A track's first sample is a key frame, so this is the
/// range of its first picture, read without decoding it.
#[cfg_attr(not(all(feature = "web", target_arch = "wasm32")), allow(dead_code))]
pub(crate) fn chunk_full_range(data: &[u8]) -> Option<bool> {
    let first = match superframe_index(data).ok()? {
        Some(sizes) => data.get(..*sizes.first()?)?,
        None => data,
    };
    let mut reader = BitReader::new(first);
    if reader.literal(2).ok()? != 2 || reader.literal(2).ok()? != 0 || reader.bit().ok()? {
        // Not a frame marker, not profile 0, or show_existing_frame.
        return None;
    }
    let key_frame = !reader.bit().ok()?;
    let show_frame = reader.bit().ok()?;
    let error_resilient = reader.bit().ok()?;
    if key_frame {
        if reader.literal(24).ok()? != 0x49_83_42 || reader.literal(3).ok()? == 7 {
            return None;
        }
        return reader.bit().ok();
    }
    let intra_only = !show_frame && reader.bit().ok()?;
    if !error_resilient {
        reader.literal(2).ok()?;
    }
    intra_only.then_some(false)
}

/// Parses a superframe index (Annex B.3), returning the frame sizes it
/// lists, or `None` when the chunk holds a single frame.
fn superframe_index(data: &[u8]) -> Result<Option<Vec<usize>>> {
    if data.is_empty() {
        return Ok(None);
    }
    let marker = data[data.len() - 1];
    if marker & 0xe0 != 0xc0 {
        return Ok(None);
    }
    let frames = usize::from(marker & 0x7) + 1;
    let magnitude = usize::from((marker >> 3) & 0x3) + 1;
    let index_size = 2 + magnitude * frames;
    if data.len() < index_size || data[data.len() - index_size] != marker {
        return Err(malformed("VP9 superframe index is invalid"));
    }
    let mut bytes = &data[data.len() - index_size + 1..];
    let mut sizes = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut size = 0usize;
        for j in 0..magnitude {
            size |= usize::from(bytes[j]) << (j * 8);
        }
        bytes = &bytes[magnitude..];
        sizes.push(size);
    }
    Ok(Some(sizes))
}
