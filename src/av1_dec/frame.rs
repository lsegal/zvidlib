//! Per-frame sample planes and the per-4x4 mode info arrays that the
//! specification keeps as frame-sized globals (`YModes`, `RefFrames`, `Mvs`,
//! and so on).

use super::consts::*;
use super::limit;
use crate::{Limits, Result};

/// One plane of samples. Samples are stored as `u16` so every intermediate
/// the specification writes into `CurrFrame` is representable.
#[derive(Clone, Debug)]
pub(crate) struct PlaneBuf {
    pub(crate) data: Vec<u16>,
    pub(crate) stride: usize,
    pub(crate) height: usize,
}

impl PlaneBuf {
    pub(crate) fn new(width: usize, height: usize, fill: u16) -> Self {
        Self {
            data: vec![fill; width * height],
            stride: width,
            height,
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.stride + x]
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, x: usize, y: usize, value: u16) {
        self.data[y * self.stride + x] = value;
    }

    #[inline(always)]
    pub(crate) fn row(&self, y: usize) -> &[u16] {
        &self.data[y * self.stride..(y + 1) * self.stride]
    }

    #[inline(always)]
    pub(crate) fn row_mut(&mut self, y: usize) -> &mut [u16] {
        &mut self.data[y * self.stride..(y + 1) * self.stride]
    }
}

/// A frame's planes; monochrome frames carry only the luma plane.
#[derive(Clone, Debug)]
pub(crate) struct FrameBuf {
    pub(crate) planes: Vec<PlaneBuf>,
}

impl FrameBuf {
    /// Allocates planes of `width` x `height` luma samples (plus chroma).
    pub(crate) fn new(
        width: usize,
        height: usize,
        num_planes: usize,
        sub_x: bool,
        sub_y: bool,
        limits: &Limits,
    ) -> Result<Self> {
        let chroma_w = (width + usize::from(sub_x)) >> usize::from(sub_x);
        let chroma_h = (height + usize::from(sub_y)) >> usize::from(sub_y);
        let samples = width
            .checked_mul(height)
            .and_then(|luma| luma.checked_add(2 * chroma_w * chroma_h))
            .ok_or_else(|| limit("AV1 frame size overflows"))?;
        if samples as u64 * 2 > limits.max_allocation_bytes {
            return Err(limit("AV1 frame exceeds the allocation limit"));
        }
        let mut planes = vec![PlaneBuf::new(width, height, 0)];
        for _ in 1..num_planes {
            planes.push(PlaneBuf::new(chroma_w, chroma_h, 0));
        }
        Ok(Self { planes })
    }
}

/// The per-4x4 block mode info the decode process stores for each frame.
pub(crate) struct ModeInfo {
    pub(crate) y_modes: Vec<u8>,
    pub(crate) uv_modes: Vec<u8>,
    pub(crate) ref_frames: Vec<[i8; 2]>,
    pub(crate) ref_frames_written: Vec<bool>,
    pub(crate) comp_group_idxs: Vec<u8>,
    pub(crate) compound_idxs: Vec<u8>,
    pub(crate) interp_filters: Vec<[u8; 2]>,
    pub(crate) mvs: Vec<[[i32; 2]; 2]>,
    pub(crate) is_inters: Vec<bool>,
    pub(crate) skip_modes: Vec<bool>,
    pub(crate) skips: Vec<bool>,
    pub(crate) tx_sizes: Vec<u8>,
    pub(crate) inter_tx_sizes: Vec<u8>,
    pub(crate) mi_sizes: Vec<u8>,
    pub(crate) segment_ids: Vec<u8>,
    pub(crate) palette_sizes: [Vec<u8>; 2],
    pub(crate) palette_colors: [Vec<[u16; PALETTE_COLORS]>; 2],
    pub(crate) delta_lfs: Vec<[i8; FRAME_LF_COUNT]>,
    pub(crate) tx_types: Vec<u8>,
}

impl ModeInfo {
    pub(crate) fn new(mi_rows: usize, mi_cols: usize) -> Self {
        let n = mi_rows * mi_cols;
        Self {
            y_modes: vec![0; n],
            uv_modes: vec![0; n],
            ref_frames: vec![[INTRA_FRAME, NONE]; n],
            ref_frames_written: vec![false; n],
            comp_group_idxs: vec![0; n],
            compound_idxs: vec![0; n],
            interp_filters: vec![[0; 2]; n],
            mvs: vec![[[0; 2]; 2]; n],
            is_inters: vec![false; n],
            skip_modes: vec![false; n],
            skips: vec![false; n],
            tx_sizes: vec![0; n],
            inter_tx_sizes: vec![0; n],
            mi_sizes: vec![0; n],
            segment_ids: vec![0; n],
            palette_sizes: [vec![0; n], vec![0; n]],
            palette_colors: [vec![[0; PALETTE_COLORS]; n], vec![[0; PALETTE_COLORS]; n]],
            delta_lfs: vec![[0; FRAME_LF_COUNT]; n],
            tx_types: vec![0; n],
        }
    }
}

/// Loop restoration coefficients signaled for one plane's units.
#[derive(Clone, Debug, Default)]
pub(crate) struct LrPlane {
    pub(crate) unit_rows: usize,
    pub(crate) unit_cols: usize,
    pub(crate) lr_type: Vec<u8>,
    pub(crate) wiener: Vec<[[i32; 3]; 2]>,
    pub(crate) sgr_set: Vec<u8>,
    pub(crate) sgr_xqd: Vec<[i32; 2]>,
}

/// Everything the decoding of one frame accumulates below the frame header.
pub(crate) struct FrameState {
    pub(crate) mi: ModeInfo,
    pub(crate) curr: FrameBuf,
    pub(crate) prev_segment_ids: Vec<u8>,
    /// `cdef_idx`, one entry per 64x64 luma block.
    pub(crate) cdef_idx: Vec<i8>,
    pub(crate) cdef_stride: usize,
    pub(crate) lr: [LrPlane; 3],
    /// `LoopfilterTxSizes[ plane ]`, at 4x4 granularity of each plane.
    pub(crate) lf_tx_sizes: [Vec<u8>; 3],
    pub(crate) lf_tx_stride: [usize; 3],
    /// `MotionFieldMvs[ ref ]` on the 8x8 grid (`-1 << 15` marks invalid).
    pub(crate) motion_field_mvs: Vec<Vec<[i32; 2]>>,
}
