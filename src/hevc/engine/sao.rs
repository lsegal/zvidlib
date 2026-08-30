//! §8.7.3 sample adaptive offset (SAO) apply.
//!
//! SAO is the second in-loop filter (it runs on a CTB basis after the
//! §8.7.2 deblocking filter completes for the whole picture). It takes the
//! reconstructed picture sample arrays and produces the modified
//! `saoPicture` arrays. Two offset types are defined:
//!
//! * **Band offset** (`SaoTypeIdx == 1`): the sample value range is split
//!   into 32 equal bands; four consecutive bands (starting at
//!   `sao_band_position`) each get a signed offset added.
//! * **Edge offset** (`SaoTypeIdx == 2`): each sample is classified by the
//!   sign pattern of its two neighbours along one of four 1-D directions
//!   (Table 8-13 `hPos` / `vPos`), giving an `edgeIdx` in 0..=4 that
//!   selects one of the five offsets.
//!
//! This module implements:
//!
//! * §7.4.9.3 — the `SaoOffsetVal[cIdx][rx][ry][i]` derivation (equation
//!   7-72), including the `sao_offset_sign` inference for edge offset
//!   (categories 0/1 positive, 2/3 negative) and the merge-flag
//!   inheritance of all five SAO arrays from the left / above CTB.
//! * §8.7.3.1 — the picture-level CTB-grid driver
//!   ([`apply_sao_picture`]) that visits every CTB and dispatches the
//!   per-component modification, honouring `slice_sao_luma_flag` /
//!   `slice_sao_chroma_flag`.
//! * §8.7.3.2 — the per-CTB modification process
//!   ([`apply_sao_ctb`]), both the edge (equations 8-409..8-413) and band
//!   (equations 8-414..8-415) paths, with the picture-boundary edge
//!   guard (an out-of-picture neighbour forces `edgeIdx = 0`, i.e. no
//!   offset).
//!
//! The cross-slice / cross-tile edge guards of §8.7.3.2 (the
//! `MinTbAddrZs` slice test and the `loop_filter_across_tiles_enabled_flag`
//! tile test) collapse to the picture-boundary test for a single-slice,
//! single-tile picture; multi-slice / multi-tile boundary masking is a
//! follow-up that threads the per-sample slice / tile id.

use crate::hevc::engine::picture::{Picture, Plane, sub_wh_c};
use crate::hevc::engine::slice_data::{SaoComponent, SaoCtbParams};

/// `Sign( x )` (§5, equation 5-18).
#[inline]
fn sign(x: i32) -> i32 {
    match x.cmp(&0) {
        core::cmp::Ordering::Greater => 1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Less => -1,
    }
}

/// Table 8-13 — `(hPos[0], vPos[0], hPos[1], vPos[1])` for a SAO edge
/// offset class (0 = 0-deg, 1 = 90-deg, 2 = 135-deg, 3 = 45-deg).
/// `pub(crate)` so the encoder's SAO estimation classifies with the
/// same table.
#[inline]
pub(crate) fn eo_pos(eo_class: u8) -> (i32, i32, i32, i32) {
    match eo_class {
        // horizontal: left + right.
        0 => (-1, 0, 1, 0),
        // vertical: above + below.
        1 => (0, -1, 0, 1),
        // 135-degree: above-left + below-right.
        2 => (-1, -1, 1, 1),
        // 45-degree: above-right + below-left.
        _ => (1, -1, -1, 1),
    }
}

/// One CTB's resolved SAO parameters for a single colour component, after
/// merge inheritance and the §7.4.9.3 `SaoOffsetVal` derivation.
///
/// `offset_val[0..5]` is `SaoOffsetVal[cIdx][rx][ry][0..4]` (equation
/// 7-72): `offset_val[0]` is always 0; the four signed offsets follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedSaoComponent {
    /// `SaoTypeIdx[cIdx][rx][ry]` — 0 (off), 1 (band), 2 (edge).
    pub sao_type_idx: u8,
    /// `SaoOffsetVal[cIdx][rx][ry][0..4]` (equation 7-72).
    pub offset_val: [i32; 5],
    /// `sao_band_position[cIdx][rx][ry]` (band offset only).
    pub band_position: u8,
    /// `SaoEoClass[cIdx][rx][ry]` (edge offset only).
    pub eo_class: u8,
}

impl ResolvedSaoComponent {
    /// A "not applied" component (`SaoTypeIdx == 0`).
    #[must_use]
    pub fn off() -> Self {
        Self {
            sao_type_idx: 0,
            offset_val: [0; 5],
            band_position: 0,
            eo_class: 0,
        }
    }

    /// Resolve a decoded [`SaoComponent`] (§7.3.8.3 syntax) into the
    /// applied form, applying the §7.4.9.3 `SaoOffsetVal` derivation
    /// (equation 7-72). `log2_offset_scale` is the §7.4.9.3
    /// `log2OffsetScale` (0 unless the range-extension
    /// `log2_sao_offset_scale_*` PPS fields are set).
    ///
    /// For edge offset the `sao_offset_sign` of the four categories is
    /// inferred per §7.4.9.3 (categories 0/1 positive, 2/3 negative)
    /// rather than read; for band offset the decoded signs are used.
    #[must_use]
    pub fn from_decoded(c: &SaoComponent, log2_offset_scale: u8) -> Self {
        let mut offset_val = [0i32; 5];
        for i in 0..4 {
            // §7.4.9.3: for edge offset the sign is inferred (i<2 ⇒ +,
            // i>=2 ⇒ −); for band offset the decoded sign applies.
            let sign_bit = if c.sao_type_idx == 2 {
                u8::from(i >= 2)
            } else {
                c.offset_sign[i]
            };
            // equation 7-72: (1 − 2*sign) * abs << log2OffsetScale.
            let signed = (1 - 2 * i32::from(sign_bit)) * (c.offset_abs[i] as i32);
            offset_val[i + 1] = signed << log2_offset_scale;
        }
        Self {
            sao_type_idx: c.sao_type_idx,
            offset_val,
            band_position: c.band_position,
            eo_class: c.eo_class,
        }
    }
}

/// One CTB's resolved SAO parameters for all three components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedSao {
    /// Per-component resolved parameters `[Y, Cb, Cr]`.
    pub components: [ResolvedSaoComponent; 3],
}

impl ResolvedSao {
    /// All-off SAO (no CTB is modified).
    #[must_use]
    pub fn off() -> Self {
        Self {
            components: [ResolvedSaoComponent::off(); 3],
        }
    }

    /// Resolve one CTB's decoded [`SaoCtbParams`] into the applied form,
    /// inheriting the left / above neighbour's resolved parameters when
    /// `sao_merge_left_flag` / `sao_merge_up_flag` is set (§7.4.9.3). The
    /// merge sources are the already-resolved CTBs; pass `None` when the
    /// neighbour is unavailable.
    #[must_use]
    pub fn resolve(
        params: &SaoCtbParams,
        left: Option<&ResolvedSao>,
        above: Option<&ResolvedSao>,
        log2_offset_scale_luma: u8,
        log2_offset_scale_chroma: u8,
    ) -> Self {
        if params.merge_left {
            if let Some(l) = left {
                return *l;
            }
        }
        if params.merge_up {
            if let Some(a) = above {
                return *a;
            }
        }
        let mut components = [ResolvedSaoComponent::off(); 3];
        for (cidx, comp) in components.iter_mut().enumerate() {
            let scale = if cidx == 0 {
                log2_offset_scale_luma
            } else {
                log2_offset_scale_chroma
            };
            *comp = ResolvedSaoComponent::from_decoded(&params.components[cidx], scale);
        }
        Self { components }
    }
}

/// §8.7.3.2 — apply the SAO CTB modification process for one colour
/// component of one CTB, reading from `rec` and writing into `sao_out`.
///
/// `(x_ctb, y_ctb)` is the component-plane top-left of the CTB; `n_w` /
/// `n_h` are the CTB width / height in component samples. Out-of-picture
/// edge-offset neighbours force `edgeIdx = 0` (no modification) per the
/// §8.7.3.2 picture-boundary guard.
///
/// `rec` and `sao_out` may be the same picture when SAO is applied in
/// place; the edge classification reads `rec` (the pre-SAO array) so an
/// in-place application is exact only when the neighbour samples have not
/// yet been overwritten. For correctness across the whole CTB grid the
/// driver [`apply_sao_picture`] snapshots the pre-SAO planes.
#[allow(clippy::too_many_arguments)]
pub fn apply_sao_ctb(
    rec: &Picture,
    sao_out: &mut Picture,
    plane: Plane,
    comp: &ResolvedSaoComponent,
    x_ctb: usize,
    y_ctb: usize,
    n_w: usize,
    n_h: usize,
) {
    apply_sao_ctb_with_boundaries(rec, sao_out, plane, comp, x_ctb, y_ctb, n_w, n_h, None);
}

/// §8.7.3.1 in-loop-filter boundary constraints: the per-CTB slice /
/// tile identity grids and the across-boundary enable flags the
/// §8.7.3.2 edge-offset neighbour test consults (a neighbouring sample
/// in a different slice / tile with filtering-across disabled forces
/// `edgeIdx = 0`).
#[derive(Debug, Clone)]
pub struct SaoBoundaries {
    /// Per-CTB `SliceAddrRs` (raster order).
    pub slice_addr_of_ctb: Vec<u32>,
    /// Per-CTB §6.5.1 `TileId` (raster order).
    pub tile_id_of_ctb: Vec<u32>,
    /// `PicWidthInCtbsY`.
    pub pic_w_ctbs: usize,
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `slice_loop_filter_across_slices_enabled_flag` — the uniform
    /// fallback used when [`Self::filter_across_of_ctb`] is `None`.
    pub across_slices: bool,
    /// `loop_filter_across_tiles_enabled_flag`.
    pub across_tiles: bool,
    /// Per-CTB (raster order) `slice_loop_filter_across_slices_enabled_
    /// flag` of the slice owning each CTB. When present, the §8.7.3.2
    /// cross-slice rule is evaluated per slice pair: the neighbour read
    /// is denied when the LATER slice (decode order) of the two has its
    /// flag equal to 0.
    pub filter_across_of_ctb: Option<Vec<bool>>,
    /// Per-CTB (raster order) tile-scan address `CtbAddrRsToTs` — the
    /// decode-order key for the §8.7.3.2 `MinTbAddrZs` comparison.
    /// `None` falls back to raster order (exact for single-tile
    /// pictures).
    pub ctb_ts_of_rs: Option<Vec<u32>>,
}

impl SaoBoundaries {
    /// Whether the §8.7.3.2 edge-offset classification may read the
    /// neighbour at luma position `(x_nb, y_nb)` from the sample at
    /// luma position `(x, y)`.
    fn neighbour_allowed(&self, x: usize, y: usize, x_nb: usize, y_nb: usize) -> bool {
        let idx = |xx: usize, yy: usize| {
            (yy >> self.ctb_log2_size_y) * self.pic_w_ctbs + (xx >> self.ctb_log2_size_y)
        };
        let a = idx(x, y);
        let b = idx(x_nb, y_nb);
        if a == b {
            return true;
        }
        if self.slice_addr_of_ctb.get(a) != self.slice_addr_of_ctb.get(b) {
            match &self.filter_across_of_ctb {
                None => {
                    if !self.across_slices {
                        return false;
                    }
                }
                Some(flags) => {
                    // §8.7.3.2 — the read is denied when the later (in
                    // decode order) of the two slices has
                    // slice_loop_filter_across_slices_enabled_flag == 0.
                    let ts = |i: usize| self.ctb_ts_of_rs.as_ref().map_or(i as u32, |m| m[i]);
                    let later = if ts(b) < ts(a) { a } else { b };
                    if !flags.get(later).copied().unwrap_or(true) {
                        return false;
                    }
                }
            }
        }
        if !self.across_tiles && self.tile_id_of_ctb.get(a) != self.tile_id_of_ctb.get(b) {
            return false;
        }
        true
    }
}

/// [`apply_sao_ctb`] with the optional §8.7.3.2 slice / tile boundary
/// constraints (edge-offset neighbours across a disallowed boundary
/// force `edgeIdx = 0`, i.e. the sample is left unmodified).
#[allow(clippy::too_many_arguments)]
pub fn apply_sao_ctb_with_boundaries(
    rec: &Picture,
    sao_out: &mut Picture,
    plane: Plane,
    comp: &ResolvedSaoComponent,
    x_ctb: usize,
    y_ctb: usize,
    n_w: usize,
    n_h: usize,
    boundaries: Option<&SaoBoundaries>,
) {
    apply_sao_ctb_full(
        rec, sao_out, plane, comp, x_ctb, y_ctb, n_w, n_h, boundaries, None,
    );
}

/// [`apply_sao_ctb_with_boundaries`] with the per-CU loop-filter
/// suppression map: samples of PCM (`pcm_loop_filter_disabled_flag`) /
/// transquant-bypass coding units keep their reconstructed values
/// (§8.7.3.1 treats both `SaoTypeIdx` components as 0 for them).
#[allow(clippy::too_many_arguments)]
pub fn apply_sao_ctb_full(
    rec: &Picture,
    sao_out: &mut Picture,
    plane: Plane,
    comp: &ResolvedSaoComponent,
    x_ctb: usize,
    y_ctb: usize,
    n_w: usize,
    n_h: usize,
    boundaries: Option<&SaoBoundaries>,
    no_filter: Option<&crate::hevc::engine::deblock::NoFilterMap<'_>>,
) {
    if comp.sao_type_idx == 0 {
        return;
    }
    let (nf_sw, nf_sh) = match plane {
        Plane::Luma => (1, 1),
        _ => crate::hevc::engine::picture::sub_wh_c(rec.chroma_array_type()),
    };
    let suppressed =
        |x: usize, y: usize| -> bool { no_filter.is_some_and(|m| m.at_luma(x * nf_sw, y * nf_sh)) };
    let bit_depth = rec.bit_depth(plane);
    let max = (1i32 << bit_depth) - 1;
    let (pw, ph) = rec.plane_dims(plane);
    let w = n_w.min(pw.saturating_sub(x_ctb));
    let h = n_h.min(ph.saturating_sub(y_ctb));

    // Fast path: with no slice / tile boundary mask and no PCM /
    // transquant-bypass suppression, whole runs of a row take the same
    // branch-free treatment, so they go through the vectorized kernels
    // in `super::simd` (which fall back to scalar on targets without a
    // supported vector ISA). The results are bit-exact with the scalar
    // code below, which stays the normative reference.
    let vectorizable = boundaries.is_none() && no_filter.is_none() && w > 0 && h > 0;

    if vectorizable && comp.sao_type_idx == 2 {
        let (h0, v0, h1, v1) = eo_pos(comp.eo_class);
        // A neighbour outside the picture forces edgeIdx = 0, so only the
        // x range where both Table 8-13 neighbours are in-picture is
        // modified at all; the margins keep their reconstructed values,
        // which `sao_out` already holds.
        let left_margin = (-(h0.min(h1))).max(0) as usize;
        let right_margin = h0.max(h1).max(0) as usize;
        let x_lo = x_ctb.max(left_margin);
        let x_hi = (x_ctb + w).min(pw.saturating_sub(right_margin));
        if x_hi > x_lo {
            let n = x_hi - x_lo;
            let src = rec.plane(plane);
            let (dst, dst_stride) = sao_out.plane_mut(plane);
            debug_assert_eq!(dst_stride, pw);
            for j in 0..h {
                let y = (y_ctb + j) as i32;
                let (y0, y1) = (y + v0, y + v1);
                if y0 < 0 || y1 < 0 || y0 as usize >= ph || y1 as usize >= ph {
                    continue; // whole row's neighbours are out of picture
                }
                let cur = y as usize * pw + x_lo;
                let o0 = y0 as usize * pw + (x_lo as i32 + h0) as usize;
                let o1 = y1 as usize * pw + (x_lo as i32 + h1) as usize;
                super::simd::sao_edge_row(
                    &src[cur..cur + n],
                    &src[o0..o0 + n],
                    &src[o1..o1 + n],
                    &mut dst[cur..cur + n],
                    &comp.offset_val,
                    max,
                );
            }
        }
        return;
    }

    if vectorizable && comp.sao_type_idx != 2 {
        // §8.7.3.2 band offset over whole rows (equations 8-414..8-415).
        let band_shift = i32::from(bit_depth) - 5;
        let left = i32::from(comp.band_position);
        let src = rec.plane(plane);
        let (dst, dst_stride) = sao_out.plane_mut(plane);
        debug_assert_eq!(dst_stride, pw);
        for j in 0..h {
            let o = (y_ctb + j) * pw + x_ctb;
            super::simd::sao_band_row(
                &src[o..o + w],
                &mut dst[o..o + w],
                &comp.offset_val,
                left,
                band_shift,
                max,
            );
        }
        return;
    }

    if comp.sao_type_idx == 2 {
        // §8.7.3.2 edge offset (equations 8-409..8-413).
        let (h0, v0, h1, v1) = eo_pos(comp.eo_class);
        for j in 0..h {
            for i in 0..w {
                let xsi = (x_ctb + i) as i32;
                let ysj = (y_ctb + j) as i32;
                let n0x = xsi + h0;
                let n0y = ysj + v0;
                let n1x = xsi + h1;
                let n1y = ysj + v1;
                // §8.7.3.2: a neighbour outside the picture forces
                // edgeIdx = 0 (no offset).
                let in_pic =
                    |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < pw && (y as usize) < ph;
                if !in_pic(n0x, n0y) || !in_pic(n1x, n1y) {
                    continue;
                }
                // §8.7.3.2: a neighbour in a different slice / tile with
                // loop filtering across that boundary disabled also
                // forces edgeIdx = 0. Positions map to luma space for
                // the CTB-grid lookup.
                if let Some(b) = boundaries {
                    let (sw, sh) = match plane {
                        Plane::Luma => (1, 1),
                        _ => crate::hevc::engine::picture::sub_wh_c(rec.chroma_array_type()),
                    };
                    let (lx, ly) = (xsi as usize * sw, ysj as usize * sh);
                    if !b.neighbour_allowed(lx, ly, n0x as usize * sw, n0y as usize * sh)
                        || !b.neighbour_allowed(lx, ly, n1x as usize * sw, n1y as usize * sh)
                    {
                        continue;
                    }
                }
                let cur = rec.sample(plane, xsi as usize, ysj as usize);
                let s0 = rec.sample(plane, n0x as usize, n0y as usize);
                let s1 = rec.sample(plane, n1x as usize, n1y as usize);
                // equation 8-411.
                let mut edge_idx = 2 + sign(cur - s0) + sign(cur - s1);
                // equation 8-412.
                if edge_idx == 0 || edge_idx == 1 || edge_idx == 2 {
                    edge_idx = if edge_idx == 2 { 0 } else { edge_idx + 1 };
                }
                // equation 8-413.
                if suppressed(xsi as usize, ysj as usize) {
                    continue; // §8.7.3.1 PCM / bypass suppression
                }
                let off = comp.offset_val[edge_idx as usize];
                let v = (cur + off).clamp(0, max);
                sao_out.set_sample(plane, xsi as usize, ysj as usize, v);
            }
        }
    } else {
        // §8.7.3.2 band offset (equations 8-414..8-415).
        let band_shift = i32::from(bit_depth) - 5;
        let sao_left_class = i32::from(comp.band_position);
        // equation 8-414: bandTable maps four consecutive bands to 1..=4.
        let mut band_table = [0usize; 32];
        for k in 0..4i32 {
            band_table[((k + sao_left_class) & 31) as usize] = (k + 1) as usize;
        }
        for j in 0..h {
            for i in 0..w {
                let xsi = x_ctb + i;
                let ysj = y_ctb + j;
                if suppressed(xsi, ysj) {
                    continue; // §8.7.3.1 PCM / bypass suppression
                }
                let cur = rec.sample(plane, xsi, ysj);
                let band_idx = band_table[(cur >> band_shift) as usize];
                // equation 8-415.
                let off = comp.offset_val[band_idx];
                let v = (cur + off).clamp(0, max);
                sao_out.set_sample(plane, xsi, ysj, v);
            }
        }
    }
}

/// §8.7.3.1 — apply SAO to a whole picture from the per-CTB resolved
/// parameter grid.
///
/// `ctb_sao` is the row-major `PicHeightInCtbsY * PicWidthInCtbsY` grid of
/// resolved per-CTB SAO parameters (raster order). `ctb_log2_size_y` is
/// the luma CTB log2 side; `chroma_array_type` sizes the chroma CTB step.
/// `slice_sao_luma_flag` / `slice_sao_chroma_flag` gate the luma / chroma
/// passes. SAO is computed against a snapshot of the input planes so the
/// edge classification always reads the pre-SAO samples (§8.7.3.1: the
/// `recPicture` array is the pre-SAO reconstruction).
///
/// Returns the modified picture; the input `pic` is consumed as the
/// `recPicture` source.
#[must_use]
pub fn apply_sao_picture(
    pic: Picture,
    ctb_sao: &[ResolvedSao],
    ctb_log2_size_y: u32,
    chroma_array_type: u8,
    slice_sao_luma_flag: bool,
    slice_sao_chroma_flag: bool,
) -> Picture {
    apply_sao_picture_with_boundaries(
        pic,
        ctb_sao,
        ctb_log2_size_y,
        chroma_array_type,
        slice_sao_luma_flag,
        slice_sao_chroma_flag,
        None,
    )
}

/// [`apply_sao_picture`] with the §8.7.3.2 slice / tile boundary
/// constraints.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn apply_sao_picture_with_boundaries(
    pic: Picture,
    ctb_sao: &[ResolvedSao],
    ctb_log2_size_y: u32,
    chroma_array_type: u8,
    slice_sao_luma_flag: bool,
    slice_sao_chroma_flag: bool,
    boundaries: Option<&SaoBoundaries>,
) -> Picture {
    apply_sao_picture_full(
        pic,
        ctb_sao,
        ctb_log2_size_y,
        chroma_array_type,
        slice_sao_luma_flag,
        slice_sao_chroma_flag,
        boundaries,
        None,
    )
}

/// [`apply_sao_picture_with_boundaries`] with the per-CU loop-filter
/// suppression map (§8.7.3.1 PCM / transquant-bypass sample skip).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn apply_sao_picture_full(
    pic: Picture,
    ctb_sao: &[ResolvedSao],
    ctb_log2_size_y: u32,
    chroma_array_type: u8,
    slice_sao_luma_flag: bool,
    slice_sao_chroma_flag: bool,
    boundaries: Option<&SaoBoundaries>,
    no_filter: Option<&crate::hevc::engine::deblock::NoFilterMap<'_>>,
) -> Picture {
    let ctb_size_y = 1usize << ctb_log2_size_y;
    let pic_width_in_ctbs = pic.width_luma().div_ceil(ctb_size_y);
    let pic_height_in_ctbs = pic.height_luma().div_ceil(ctb_size_y);
    // §8.7.3.1: saoPicture starts equal to recPicture; the edge / band
    // classification reads recPicture (the snapshot below), the write
    // targets saoPicture. `pic` is taken by value so `out` can reuse its
    // buffer directly instead of a second full-picture clone — SAO
    // decodes a 1080p frame's worth of planes (~8MB+ luma/chroma), so an
    // extra clone here is a meaningful per-frame cost.
    let rec = pic.clone();
    let mut out = pic;

    let (sw, sh) = if chroma_array_type == 0 {
        (1, 1)
    } else {
        sub_wh_c(chroma_array_type)
    };
    let n_ctb_chroma_w = ctb_size_y / sw;
    let n_ctb_chroma_h = ctb_size_y / sh;

    for ry in 0..pic_height_in_ctbs {
        for rx in 0..pic_width_in_ctbs {
            let resolved = &ctb_sao[ry * pic_width_in_ctbs + rx];
            if slice_sao_luma_flag {
                apply_sao_ctb_full(
                    &rec,
                    &mut out,
                    Plane::Luma,
                    &resolved.components[0],
                    rx * ctb_size_y,
                    ry * ctb_size_y,
                    ctb_size_y,
                    ctb_size_y,
                    boundaries,
                    no_filter,
                );
            }
            if chroma_array_type != 0 && slice_sao_chroma_flag {
                apply_sao_ctb_full(
                    &rec,
                    &mut out,
                    Plane::Cb,
                    &resolved.components[1],
                    rx * n_ctb_chroma_w,
                    ry * n_ctb_chroma_h,
                    n_ctb_chroma_w,
                    n_ctb_chroma_h,
                    boundaries,
                    no_filter,
                );
                apply_sao_ctb_full(
                    &rec,
                    &mut out,
                    Plane::Cr,
                    &resolved.components[2],
                    rx * n_ctb_chroma_w,
                    ry * n_ctb_chroma_h,
                    n_ctb_chroma_w,
                    n_ctb_chroma_h,
                    boundaries,
                    no_filter,
                );
            }
        }
    }
    out
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::slice_data::{SaoComponent, SaoCtbParams};

    fn band_component(band_position: u8, abs: [u32; 4], sign: [u8; 4]) -> SaoComponent {
        SaoComponent {
            sao_type_idx: 1,
            offset_abs: abs,
            offset_sign: sign,
            band_position,
            eo_class: 0,
        }
    }

    fn edge_component(eo_class: u8, abs: [u32; 4]) -> SaoComponent {
        SaoComponent {
            sao_type_idx: 2,
            offset_abs: abs,
            offset_sign: [0; 4],
            band_position: 0,
            eo_class,
        }
    }

    #[test]
    fn offset_val_band_uses_decoded_signs_eq_7_72() {
        // abs = [1,2,3,4], signs = [0,1,0,1] ⇒ +1,−2,+3,−4.
        let c = band_component(0, [1, 2, 3, 4], [0, 1, 0, 1]);
        let r = ResolvedSaoComponent::from_decoded(&c, 0);
        assert_eq!(r.offset_val, [0, 1, -2, 3, -4]);
    }

    #[test]
    fn offset_val_edge_infers_signs() {
        // edge: i<2 positive, i>=2 negative regardless of decoded sign.
        let c = edge_component(0, [1, 2, 3, 4]);
        let r = ResolvedSaoComponent::from_decoded(&c, 0);
        assert_eq!(r.offset_val, [0, 1, 2, -3, -4]);
    }

    #[test]
    fn offset_val_scales_by_log2_offset_scale() {
        let c = band_component(0, [1, 1, 1, 1], [0, 0, 0, 0]);
        let r = ResolvedSaoComponent::from_decoded(&c, 2);
        // each offset << 2.
        assert_eq!(r.offset_val, [0, 4, 4, 4, 4]);
    }

    #[test]
    fn merge_left_inherits_resolved_params() {
        let left = ResolvedSao {
            components: [
                ResolvedSaoComponent {
                    sao_type_idx: 1,
                    offset_val: [0, 5, 5, 5, 5],
                    band_position: 7,
                    eo_class: 0,
                },
                ResolvedSaoComponent::off(),
                ResolvedSaoComponent::off(),
            ],
        };
        let params = SaoCtbParams {
            merge_left: true,
            merge_up: false,
            components: [SaoComponent::default(); 3],
        };
        let r = ResolvedSao::resolve(&params, Some(&left), None, 0, 0);
        assert_eq!(r, left);
    }

    #[test]
    fn band_offset_adds_to_samples_in_band() {
        // 8-bit, bandShift = 3, sao_band_position = 0 ⇒ bands 0..=3 (sample
        // values 0..31) get offsets +1,+2,+3,+4. A sample of value 10 is in
        // band 1 (10 >> 3 == 1) ⇒ +2.
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 10);
            }
        }
        let comp =
            ResolvedSaoComponent::from_decoded(&band_component(0, [1, 2, 3, 4], [0, 0, 0, 0]), 0);
        let mut out = pic.clone();
        apply_sao_ctb(&pic, &mut out, Plane::Luma, &comp, 0, 0, 16, 16);
        assert_eq!(out.sample(Plane::Luma, 0, 0), 12);
    }

    #[test]
    fn band_offset_leaves_other_bands_untouched() {
        // sample 200 (>> 3 == 25) is outside bands 0..=3 ⇒ unchanged.
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 200);
            }
        }
        let comp =
            ResolvedSaoComponent::from_decoded(&band_component(0, [1, 2, 3, 4], [0, 0, 0, 0]), 0);
        let mut out = pic.clone();
        apply_sao_ctb(&pic, &mut out, Plane::Luma, &comp, 0, 0, 16, 16);
        assert_eq!(out.sample(Plane::Luma, 5, 5), 200);
    }

    #[test]
    fn edge_offset_local_minimum_gets_category_1() {
        // horizontal EO. A sample lower than both horizontal neighbours is a
        // local minimum: edgeIdx = 2 + Sign(cur-left) + Sign(cur-right)
        // = 2 + (−1) + (−1) = 0 → remapped to category 1, offset_val[1].
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 100);
            }
        }
        // Make (5,5) a local minimum along the horizontal axis.
        pic.set_sample(Plane::Luma, 5, 5, 50);
        let comp = ResolvedSaoComponent::from_decoded(&edge_component(0, [3, 0, 0, 0]), 0);
        let mut out = pic.clone();
        apply_sao_ctb(&pic, &mut out, Plane::Luma, &comp, 0, 0, 16, 16);
        // offset_val[1] = +3 (category 1, inferred positive).
        assert_eq!(out.sample(Plane::Luma, 5, 5), 53);
        // a flat-region sample (cur == both neighbours) ⇒ edgeIdx 2 → 0,
        // offset_val[0] = 0 ⇒ unchanged.
        assert_eq!(out.sample(Plane::Luma, 0, 5), 100);
    }

    #[test]
    fn edge_offset_picture_boundary_neighbour_skips() {
        // The left column has no left neighbour for horizontal EO ⇒ the
        // §8.7.3.2 boundary guard leaves it unmodified even if it would
        // otherwise classify.
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 100);
            }
        }
        pic.set_sample(Plane::Luma, 0, 5, 50);
        let comp = ResolvedSaoComponent::from_decoded(&edge_component(0, [3, 0, 0, 0]), 0);
        let mut out = pic.clone();
        apply_sao_ctb(&pic, &mut out, Plane::Luma, &comp, 0, 0, 16, 16);
        // (0,5) has no left neighbour ⇒ unchanged.
        assert_eq!(out.sample(Plane::Luma, 0, 5), 50);
    }

    #[test]
    fn picture_driver_off_grid_is_identity() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 123);
            }
        }
        let grid = vec![ResolvedSao::off(); 1];
        let out = apply_sao_picture(pic.clone(), &grid, 4, 1, true, true);
        assert_eq!(out, pic);
    }

    #[test]
    fn picture_driver_band_offset_one_ctb() {
        // 16×16 picture, one 16×16 CTB, band offset on luma only.
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 10);
                pic.set_sample(Plane::Cb, x / 2, y / 2, 10);
            }
        }
        let resolved = ResolvedSao {
            components: [
                ResolvedSaoComponent::from_decoded(
                    &band_component(0, [1, 2, 3, 4], [0, 0, 0, 0]),
                    0,
                ),
                ResolvedSaoComponent::off(),
                ResolvedSaoComponent::off(),
            ],
        };
        let out = apply_sao_picture(pic, &[resolved], 4, 1, true, true);
        // luma band 1 ⇒ +2; chroma off ⇒ unchanged.
        assert_eq!(out.sample(Plane::Luma, 0, 0), 12);
        assert_eq!(out.sample(Plane::Cb, 0, 0), 10);
    }

    #[test]
    fn edge_offset_classification_reads_presao_snapshot() {
        // A diagonal gradient: SAO must classify each sample against the
        // PRE-SAO neighbours, not the partially-modified output. Build a
        // horizontal ramp so each interior sample is monotonic (edgeIdx 0
        // → category 0 → no offset), proving no double-application.
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, (x as i32) * 4 + 20);
            }
        }
        let comp = ResolvedSaoComponent::from_decoded(&edge_component(0, [9, 9, 9, 9]), 0);
        let resolved = ResolvedSao {
            components: [
                comp,
                ResolvedSaoComponent::off(),
                ResolvedSaoComponent::off(),
            ],
        };
        let out = apply_sao_picture(pic, &[resolved], 4, 1, true, false);
        // Interior column 5: cur=40, left=36, right=44. Sign(40-36)=+1,
        // Sign(40-44)=−1 ⇒ edgeIdx 2 → 0, offset_val[0]=0 ⇒ unchanged.
        assert_eq!(out.sample(Plane::Luma, 5, 5), 40);
    }
}
