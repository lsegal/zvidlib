//! §8.7.2.4 — derivation of the deblocking boundary filtering strength.
//!
//! The deblocking filter (§8.7.2) operates on an 8×8 luma grid: vertical
//! edges first, then horizontal edges. For each candidate edge segment
//! the boundary filtering strength `bS ∈ {0, 1, 2}` is derived per
//! §8.7.2.4 from the intra/inter mode, the transform coded-block flags,
//! and the motion vectors / reference pictures of the two prediction
//! blocks straddling the edge. This module implements that derivation —
//! [`derive_boundary_strength`] — reading the per-4×4-block motion / mode
//! state from a [`crate::hevc::engine::motion::MotionField`].
//!
//! The §8.7.2.2 / §8.7.2.3 edge-flag derivation (which marks the
//! transform-block / prediction-block boundaries an edge falls on, from
//! the coding / transform tree geometry) is the input `edgeFlags` array;
//! producing it from the decoded partition tree is the picture driver's
//! follow-up. The §8.7.2.5 edge filtering process (the actual sample
//! modification using `bS`, `β` and `tC`) is implemented by the
//! per-sample primitives [`luma_beta_tc`] (§8.7.2.5.3 β/tC, Table 8-12
//! via [`beta_prime`] / [`tc_prime`]), [`luma_sample_decision`]
//! (§8.7.2.5.6), [`luma_edge_decision`] (§8.7.2.5.3 `dE`/`dEp`/`dEq`)
//! and [`filter_luma_sample`] (§8.7.2.5.7 strong/weak), plus the chroma
//! path [`chroma_tc`] (§8.7.2.5.5, Table 8-10 via [`chroma_qpc_420`])
//! and [`filter_chroma_sample`] (§8.7.2.5.8). The plane-level block-edge
//! drivers [`filter_luma_block_edge`] (§8.7.2.5.4) and
//! [`filter_chroma_block_edge`] (§8.7.2.5.5) gather/apply those
//! primitives across a 4-row edge segment of a [`SamplePlane`] in place.

use crate::hevc::engine::binarization::PartMode;
use crate::hevc::engine::motion::MotionField;
use crate::hevc::engine::picture::{Picture, Plane, sub_wh_c};

/// §8.7.2.1 — the edge orientation being filtered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// `EDGE_VER` — a vertical edge (filtered left-to-right).
    Vertical,
    /// `EDGE_HOR` — a horizontal edge (filtered top-to-bottom).
    Horizontal,
}

/// The transform-tree split geometry of one luma coding block, used by the
/// §8.7.2.2 edge-flag derivation.
///
/// Each node is either a `Split` (a `split_transform_flag == 1` node whose
/// four quadrants recurse) or a `Leaf` (a `split_transform_flag == 0`
/// transform block). The deblocker only needs the split structure, not the
/// residual coefficients — the leaves mark where transform-block edges
/// fall on the 8×8 grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformSplit {
    /// A `split_transform_flag == 1` node: four quadrants in raster order
    /// (top-left, top-right, bottom-left, bottom-right), each spanning
    /// `1 << (log2TrafoSize − 1)` luma samples.
    Split(Box<[TransformSplit; 4]>),
    /// A `split_transform_flag == 0` transform-block leaf.
    Leaf,
}

impl TransformSplit {
    /// A leaf spanning the whole coding block (no transform split).
    #[inline]
    #[must_use]
    pub fn leaf() -> Self {
        TransformSplit::Leaf
    }

    /// A single-level split into four equal quadrants, each a leaf.
    #[inline]
    #[must_use]
    pub fn split_once() -> Self {
        TransformSplit::Split(Box::new([
            TransformSplit::Leaf,
            TransformSplit::Leaf,
            TransformSplit::Leaf,
            TransformSplit::Leaf,
        ]))
    }

    /// Map a parsed §7.3.8.8 [`crate::hevc::engine::transform_tree::TransformTree`] to the
    /// deblocking transform-split structure the §8.7.2.2 transform-block
    /// boundary recursion consumes (only the split topology matters, not the
    /// per-leaf residual). A CU with no transform tree (skip / `rqt_root_cbf
    /// == 0`) is a single whole-CB leaf.
    #[must_use]
    pub fn from_tree(tree: Option<&crate::hevc::engine::transform_tree::TransformTree>) -> Self {
        use crate::hevc::engine::transform_tree::TransformTree;
        match tree {
            None | Some(TransformTree::Leaf { .. }) => TransformSplit::Leaf,
            Some(TransformTree::Split { children, .. }) => TransformSplit::Split(Box::new([
                Self::from_tree(Some(&children[0])),
                Self::from_tree(Some(&children[1])),
                Self::from_tree(Some(&children[2])),
                Self::from_tree(Some(&children[3])),
            ])),
        }
    }
}

/// The invariant context threaded through the §8.7.2.2 recursion.
struct TbBoundaryCtx<'a> {
    filter_edge_flag: bool,
    edge_type: EdgeType,
    n_cbs: usize,
    edge_flags: &'a mut [bool],
    tb_edge: &'a mut [bool],
}

/// §8.7.2.2 transform-block-boundary recursion: walk one node of the
/// transform tree, marking the `edge_flags` (any block boundary the edge
/// falls on) and `tb_edge` (transform-block boundaries only) grids.
///
/// `(xb0, yb0)` is the node's top-left luma sample relative to the coding
/// block; `log2_trafo_size` is the node side. The `ctx.filter_edge_flag`
/// is the §8.7.2.1 `filterLeftCbEdgeFlag` / `filterTopCbEdgeFlag` that
/// gates the coding-block boundary (xB0/yB0 == 0) edge.
fn transform_block_boundary(
    node: &TransformSplit,
    xb0: usize,
    yb0: usize,
    log2_trafo_size: u32,
    ctx: &mut TbBoundaryCtx<'_>,
) {
    if let TransformSplit::Split(children) = node {
        let half = 1usize << (log2_trafo_size - 1);
        let xb1 = xb0 + half;
        let yb1 = yb0 + half;
        let quad_origins = [(xb0, yb0), (xb1, yb0), (xb0, yb1), (xb1, yb1)];
        for (child, &(cx, cy)) in children.iter().zip(quad_origins.iter()) {
            transform_block_boundary(child, cx, cy, log2_trafo_size - 1, ctx);
        }
        return;
    }
    // Leaf: mark the leading edge of this transform block. EDGE_VER marks
    // the column xB0 for k = 0..size; EDGE_HOR marks the row yB0.
    let size = 1usize << log2_trafo_size;
    let n_cbs = ctx.n_cbs;
    match ctx.edge_type {
        EdgeType::Vertical => {
            // edgeFlags[xB0][yB0+k]: filterEdgeFlag at the CB boundary
            // (xB0 == 0), else 1.
            let val = if xb0 == 0 { ctx.filter_edge_flag } else { true };
            for k in 0..size {
                let idx = (yb0 + k) * n_cbs + xb0;
                ctx.edge_flags[idx] = val;
                ctx.tb_edge[idx] = val;
            }
        }
        EdgeType::Horizontal => {
            let val = if yb0 == 0 { ctx.filter_edge_flag } else { true };
            for k in 0..size {
                let idx = yb0 * n_cbs + (xb0 + k);
                ctx.edge_flags[idx] = val;
                ctx.tb_edge[idx] = val;
            }
        }
    }
}

/// §8.7.2.3 prediction-block-boundary derivation: mark the internal
/// prediction-partition edge of a coding block into `edge_flags` (a
/// prediction edge is *not* a transform-block edge, so `tb_edge` is left
/// untouched).
fn prediction_block_boundary(
    part_mode: PartMode,
    log2_cb_size: u32,
    edge_type: EdgeType,
    n_cbs: usize,
    edge_flags: &mut [bool],
) {
    // The internal partition column (EDGE_VER) / row (EDGE_HOR), if any.
    let half = 1usize << (log2_cb_size - 1);
    let quarter = 1usize << (log2_cb_size.saturating_sub(2));
    let pos = match (edge_type, part_mode) {
        (EdgeType::Vertical, PartMode::PartNx2N | PartMode::PartNxN) => Some(half),
        (EdgeType::Vertical, PartMode::PartNLx2N) => Some(quarter),
        (EdgeType::Vertical, PartMode::PartNRx2N) => Some(3 * quarter),
        (EdgeType::Horizontal, PartMode::Part2NxN | PartMode::PartNxN) => Some(half),
        (EdgeType::Horizontal, PartMode::Part2NxnU) => Some(quarter),
        (EdgeType::Horizontal, PartMode::Part2NxnD) => Some(3 * quarter),
        _ => None,
    };
    if let Some(p) = pos {
        for k in 0..n_cbs {
            let idx = match edge_type {
                EdgeType::Vertical => k * n_cbs + p,
                EdgeType::Horizontal => p * n_cbs + k,
            };
            edge_flags[idx] = true;
        }
    }
}

/// The §8.7.2.2 / §8.7.2.3 edge-flag grids for one coding block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeFlags {
    n_cbs: usize,
    /// Row-major `n_cbs²`, `true` where a transform / prediction block
    /// boundary falls (the §8.7.2.4 bS input).
    edge_flags: Vec<bool>,
    /// Row-major `n_cbs²`, `true` where the edge is *also* a
    /// transform-block boundary (the §8.7.2.4 second-bullet `cbf` test).
    tb_edge: Vec<bool>,
}

impl EdgeFlags {
    /// The `edge_flags` grid (row-major, `[y * nCbS + x]`).
    #[inline]
    #[must_use]
    pub fn edge_flags(&self) -> &[bool] {
        &self.edge_flags
    }

    /// The `tb_edge` grid (row-major, `[y * nCbS + x]`).
    #[inline]
    #[must_use]
    pub fn tb_edge(&self) -> &[bool] {
        &self.tb_edge
    }

    /// `edge_flags[x][y]`.
    #[inline]
    #[must_use]
    pub fn edge_at(&self, x: usize, y: usize) -> bool {
        self.edge_flags[y * self.n_cbs + x]
    }

    /// `tb_edge[x][y]`.
    #[inline]
    #[must_use]
    pub fn tb_at(&self, x: usize, y: usize) -> bool {
        self.tb_edge[y * self.n_cbs + x]
    }
}

/// §8.7.2.2 + §8.7.2.3 — derive the `edge_flags` + `tb_edge` grids for one
/// coding block in one edge direction.
///
/// `transform_split` is the coding block's transform-tree split geometry
/// ([`TransformSplit`]); `part_mode` is the prediction `PartMode`;
/// `log2_cb_size` is the luma coding-block side; `filter_edge_flag` is the
/// §8.7.2.1 `filterLeftCbEdgeFlag` (EDGE_VER) / `filterTopCbEdgeFlag`
/// (EDGE_HOR) gating the coding-block boundary edge.
///
/// The result feeds [`derive_boundary_strength`] (`edge_flags` + `tb_edge`)
/// and, after bS, the §8.7.2.5.1 / .2 CU edge filtering driver.
#[must_use]
pub fn derive_edge_flags(
    transform_split: &TransformSplit,
    part_mode: PartMode,
    log2_cb_size: u32,
    edge_type: EdgeType,
    filter_edge_flag: bool,
) -> EdgeFlags {
    let n_cbs = 1usize << log2_cb_size;
    let mut edge_flags = vec![false; n_cbs * n_cbs];
    let mut tb_edge = vec![false; n_cbs * n_cbs];
    {
        let mut ctx = TbBoundaryCtx {
            filter_edge_flag,
            edge_type,
            n_cbs,
            edge_flags: &mut edge_flags,
            tb_edge: &mut tb_edge,
        };
        transform_block_boundary(transform_split, 0, 0, log2_cb_size, &mut ctx);
    }
    prediction_block_boundary(part_mode, log2_cb_size, edge_type, n_cbs, &mut edge_flags);
    EdgeFlags {
        n_cbs,
        edge_flags,
        tb_edge,
    }
}

/// The §8.7.2.4 boundary filtering strength for one CU's edges, a
/// `(nCbS)×(nCbS)` grid indexed in luma samples (only the §8.7.2.4
/// sampled positions `xDi`/`yDj` carry meaningful values; the rest are 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryStrength {
    n_cbs: usize,
    /// Row-major `n_cbs * n_cbs`, `bs[y * n_cbs + x]`.
    bs: Vec<u8>,
}

impl BoundaryStrength {
    /// `bS[x][y]` at luma offset `(x, y)` within the coding block.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the `(nCbS)×(nCbS)` block.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.bs[y * self.n_cbs + x]
    }

    /// The coding-block side `nCbS` in luma samples.
    #[inline]
    #[must_use]
    pub fn n_cbs(&self) -> usize {
        self.n_cbs
    }
}

/// Whether the §8.7.2.4 "different reference pictures or different number
/// of motion vectors" predicate holds between the p- and q-side blocks.
///
/// The reference-picture comparison uses the picture identities
/// (`ref_poc_lX`), not the `RefIdxLX` list positions, and is order- and
/// list-independent (spec NOTE 1): block p uses reference-picture set
/// `{ p.l0?, p.l1? }`, block q uses `{ q.l0?, q.l1? }`.
fn different_refs_or_count(
    p: &crate::hevc::engine::motion::MotionCell,
    q: &crate::hevc::engine::motion::MotionCell,
) -> bool {
    let p_count = u8::from(p.pred_flag_l0) + u8::from(p.pred_flag_l1);
    let q_count = u8::from(q.pred_flag_l0) + u8::from(q.pred_flag_l1);
    if p_count != q_count {
        return true;
    }
    // Compare the *sets* of reference pictures (POCs), list-independent.
    let mut p_refs = Vec::new();
    if p.pred_flag_l0 {
        p_refs.push(p.ref_poc_l0);
    }
    if p.pred_flag_l1 {
        p_refs.push(p.ref_poc_l1);
    }
    let mut q_refs = Vec::new();
    if q.pred_flag_l0 {
        q_refs.push(q.ref_poc_l0);
    }
    if q.pred_flag_l1 {
        q_refs.push(q.ref_poc_l1);
    }
    p_refs.sort_unstable();
    q_refs.sort_unstable();
    p_refs != q_refs
}

/// `|a − b| >= 4` on either component (the §8.7.2.4 quarter-luma-sample
/// MV-difference test).
#[inline]
fn mv_diff_ge4(a: [i32; 2], b: [i32; 2]) -> bool {
    (a[0] - b[0]).abs() >= 4 || (a[1] - b[1]).abs() >= 4
}

/// §8.7.2.4 motion-only bS test for two inter prediction blocks `p` / `q`
/// (the third bullet onward: same/different references, MV differences).
/// Returns `true` when bS should be 1 on motion grounds.
fn motion_bs_is_one(
    p: &crate::hevc::engine::motion::MotionCell,
    q: &crate::hevc::engine::motion::MotionCell,
) -> bool {
    // Bullet 1: different reference pictures or different number of MVs.
    if different_refs_or_count(p, q) {
        return true;
    }

    let p_count = u8::from(p.pred_flag_l0) + u8::from(p.pred_flag_l1);

    // Bullet 2: one MV each, |Δmv| >= 4.
    if p_count == 1 {
        let pmv = if p.pred_flag_l0 { p.mv_l0 } else { p.mv_l1 };
        let qmv = if q.pred_flag_l0 { q.mv_l0 } else { q.mv_l1 };
        return mv_diff_ge4(pmv, qmv);
    }

    // Two MVs each (p_count == 2). The reference-picture sets are equal
    // (different_refs_or_count returned false).
    if p.ref_poc_l0 != p.ref_poc_l1 {
        // Bullet 3: two different reference pictures. Match each list to
        // the q-side list referencing the same picture.
        // p.l0 ↔ q.(list with same poc as p.l0), p.l1 ↔ the other.
        let (q0, q1) = if q.ref_poc_l0 == p.ref_poc_l0 {
            (q.mv_l0, q.mv_l1)
        } else {
            (q.mv_l1, q.mv_l0)
        };
        mv_diff_ge4(p.mv_l0, q0) || mv_diff_ge4(p.mv_l1, q1)
    } else {
        // Bullet 4: both MVs use the same reference picture. bS = 1 iff
        //   (|Δl0| or |Δl1| >= 4)  AND  (|p.l0−q.l1| or |p.l1−q.l0| >= 4).
        let straight = mv_diff_ge4(p.mv_l0, q.mv_l0) || mv_diff_ge4(p.mv_l1, q.mv_l1);
        let crossed = mv_diff_ge4(p.mv_l0, q.mv_l1) || mv_diff_ge4(p.mv_l1, q.mv_l0);
        straight && crossed
    }
}

/// §8.7.2.4 — derive the `(nCbS)×(nCbS)` boundary filtering strength grid
/// for one luma coding block.
///
/// Inputs:
/// * `field` — the picture's per-4×4-block motion / mode store
///   ([`MotionField`]); the p / q sample look-ups read from it.
/// * `(x_cb, y_cb)` — the coding block's luma top-left position.
/// * `log2_cb_size` — the coding-block side `log2CbSize`.
/// * `edge_type` — `EDGE_VER` or `EDGE_HOR`.
/// * `edge_flags` — the §8.7.2.2/§8.7.2.3 `(nCbS)×(nCbS)` edge-flag grid
///   (row-major, `edge_flags[y * nCbS + x]`), `true` where a transform /
///   prediction block boundary falls.
/// * `tb_edge` — `(nCbS)×(nCbS)` grid, `true` where the edge at that
///   position is *also* a transform-block edge (the §8.7.2.4 second
///   bullet `cbf` test only fires on transform-block edges).
///
/// The bS at every sampled `(xDi, yDj)` is 2 when either neighbouring
/// sample is intra; 1 when the edge is a transform-block edge and either
/// transform block holds a non-zero coefficient; 1 on the §8.7.2.4
/// motion criteria; otherwise 0. Positions not sampled by §8.7.2.4 are 0.
///
/// # Panics
/// Panics if `edge_flags` / `tb_edge` are not `(1 << log2_cb_size)`²
/// long, or if a sampled p / q location falls outside `field`.
pub fn derive_boundary_strength(
    field: &MotionField,
    x_cb: usize,
    y_cb: usize,
    log2_cb_size: u32,
    edge_type: EdgeType,
    edge_flags: &[bool],
    tb_edge: &[bool],
) -> BoundaryStrength {
    let n_cbs = 1usize << log2_cb_size;
    assert_eq!(edge_flags.len(), n_cbs * n_cbs, "edge_flags is nCbS^2");
    assert_eq!(tb_edge.len(), n_cbs * n_cbs, "tb_edge is nCbS^2");
    let mut bs = vec![0u8; n_cbs * n_cbs];

    // §8.7.2.4 sampling grid: EDGE_VER strides 8 in x, 4 in y; EDGE_HOR
    // strides 4 in x, 8 in y.
    let (step_x, step_y) = match edge_type {
        EdgeType::Vertical => (8usize, 4usize),
        EdgeType::Horizontal => (4usize, 8usize),
    };

    let mut y_dj = 0;
    while y_dj < n_cbs {
        let mut x_di = 0;
        while x_di < n_cbs {
            let idx = y_dj * n_cbs + x_di;
            if edge_flags[idx] {
                // p0 / q0 sample positions (§8.7.2.4).
                let (px, py, qx, qy) = match edge_type {
                    EdgeType::Vertical => (x_cb + x_di - 1, y_cb + y_dj, x_cb + x_di, y_cb + y_dj),
                    EdgeType::Horizontal => {
                        (x_cb + x_di, y_cb + y_dj - 1, x_cb + x_di, y_cb + y_dj)
                    }
                };
                let p = field.cell_at(px, py);
                let q = field.cell_at(qx, qy);

                // §8.7.2.4 cascade: intra ⇒ 2; else a coded-coeff
                // transform-block edge OR the motion criteria ⇒ 1; else 0.
                let coeff_edge = tb_edge[idx] && (p.has_nonzero_coeff || q.has_nonzero_coeff);
                let val = if p.is_intra || q.is_intra {
                    2
                } else if coeff_edge || motion_bs_is_one(&p, &q) {
                    1
                } else {
                    0
                };
                bs[idx] = val;
            }
            x_di += step_x;
        }
        y_dj += step_y;
    }

    BoundaryStrength { n_cbs, bs }
}

/// Table 8-12 — `β′` as a function of the input variable `Q` (0..=51).
///
/// The deblocking threshold `β′` is 0 for `Q < 16`, then rises in the
/// pattern documented in Table 8-12. `Q` outside `0..=51` returns the
/// boundary value (the callers clip `Q` to `0..=51` for `β′`, eq. 8-348).
#[must_use]
pub fn beta_prime(q: i32) -> i32 {
    const BETA: [i32; 52] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q 0..15
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, // Q 16..31
        26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, // Q 32..47
        58, 60, 62, 64, // Q 48..51
    ];
    BETA[q.clamp(0, 51) as usize]
}

/// Table 8-12 — `tC′` as a function of the input variable `Q` (0..=53).
///
/// The clipping threshold `tC′` is 0 for `Q <= 17`, `1` at `Q = 18`,
/// then rises per Table 8-12 up to `24` at `Q = 53`. `Q` outside
/// `0..=53` returns the boundary value (callers clip `Q` to `0..=53`,
/// eqs. 8-350 / 8-383).
#[must_use]
pub fn tc_prime(q: i32) -> i32 {
    const TC: [i32; 54] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // Q 0..18
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, // Q 19..37
        5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24, // Q 38..53
    ];
    TC[q.clamp(0, 53) as usize]
}

/// §8.7.2.5.3 — derive `β` and `tC` for a luma block edge.
///
/// `qp_q` / `qp_p` are the `QpY` values of the coding units containing
/// the `q0,0` and `p0,0` samples; `bs` is the §8.7.2.4 boundary strength
/// (must be 1 or 2 for a filtered edge); `beta_offset_div2` /
/// `tc_offset_div2` are the per-slice `slice_beta_offset_div2` /
/// `slice_tc_offset_div2` syntax-element values; `bit_depth` is
/// `BitDepthY`.
///
/// Returns `(β, tC)` per eqs. 8-347..8-351.
#[must_use]
pub fn luma_beta_tc(
    qp_q: i32,
    qp_p: i32,
    bs: u8,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    bit_depth: u8,
) -> (i32, i32) {
    let qp_l = (qp_q + qp_p + 1) >> 1; // eq. 8-347
    let q_beta = (qp_l + (beta_offset_div2 << 1)).clamp(0, 51); // eq. 8-348
    let beta = beta_prime(q_beta) * (1 << (i32::from(bit_depth) - 8)); // eq. 8-349
    let q_tc = (qp_l + 2 * (i32::from(bs) - 1) + (tc_offset_div2 << 1)).clamp(0, 53); // eq. 8-350
    let tc = tc_prime(q_tc) * (1 << (i32::from(bit_depth) - 8)); // eq. 8-351
    (beta, tc)
}

/// §8.7.2.5.6 — decision process for a luma sample.
///
/// Returns `dSam`: `true` (1) when the long (strong) filter conditions
/// across `p0`/`p3`/`q0`/`q3` hold, given `dpq`, `β` and `tC`.
#[inline]
#[must_use]
pub fn luma_sample_decision(
    p0: i32,
    p3: i32,
    q0: i32,
    q3: i32,
    dpq: i32,
    beta: i32,
    tc: i32,
) -> bool {
    dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < (5 * tc + 1) >> 1
}

/// Decisions emitted by the §8.7.2.5.3 luma edge decision process for a
/// 4-row edge segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaEdgeDecision {
    /// `dE`: 0 = no filtering, 1 = weak (normal), 2 = strong (long).
    pub de: u8,
    /// `dEp`: when 1, the `p1` sample is also filtered (weak filter).
    pub dep: u8,
    /// `dEq`: when 1, the `q1` sample is also filtered (weak filter).
    pub deq: u8,
    /// `β` (eq. 8-349).
    pub beta: i32,
    /// `tC` (eq. 8-351).
    pub tc: i32,
}

/// The result of §8.7.2.5.7 luma-sample filtering for one boundary row.
///
/// `p[i]` / `q[i]` are the filtered sample values (`pi'` / `qj'`); only
/// the first `ndp` `p[]` and `ndq` `q[]` entries are valid replacements
/// for the input samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaFilterOut {
    /// `nDp` — number of filtered samples on the p side (0..=3).
    pub ndp: usize,
    /// `nDq` — number of filtered samples on the q side (0..=3).
    pub ndq: usize,
    /// `p0'`, `p1'`, `p2'` (only `0..ndp` valid).
    pub p: [i32; 3],
    /// `q0'`, `q1'`, `q2'` (only `0..ndq` valid).
    pub q: [i32; 3],
}

/// §8.7.2.5.7 — filtering process for a luma sample row.
///
/// `p` / `q` are the four samples each side of the boundary (`p[0]` is
/// `p0`, closest to the edge); `dec` carries `dE` / `dEp` / `dEq` and
/// `tC`; `bit_depth` is `BitDepthY` for the final `Clip1Y`.
///
/// Strong filtering (`dE == 2`) writes `p0'..p2'` / `q0'..q2'`; weak
/// filtering (`dE == 1`) writes `p0'` (+`p1'` when `dEp`) and `q0'`
/// (+`q1'` when `dEq`). The PCM / transquant-bypass / palette
/// suppressions are applied by the caller (they need per-CU state).
#[must_use]
pub fn filter_luma_sample(
    p: [i32; 4],
    q: [i32; 4],
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
) -> LumaFilterOut {
    let [p0, p1, p2, p3] = p;
    let [q0, q1, q2, q3] = q;
    let clip = |v: i32| crate::hevc::engine::picture::clip1(v, bit_depth);
    if de == 2 {
        // Strong filter (eqs. 8-389..8-394).
        let p0p =
            (p0 - 2 * tc).max((p0 + 2 * tc).min((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3));
        let p1p = (p1 - 2 * tc).max((p1 + 2 * tc).min((p2 + p1 + p0 + q0 + 2) >> 2));
        let p2p = (p2 - 2 * tc).max((p2 + 2 * tc).min((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3));
        let q0p =
            (q0 - 2 * tc).max((q0 + 2 * tc).min((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3));
        let q1p = (q1 - 2 * tc).max((q1 + 2 * tc).min((p0 + q0 + q1 + q2 + 2) >> 2));
        let q2p = (q2 - 2 * tc).max((q2 + 2 * tc).min((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3));
        return LumaFilterOut {
            ndp: 3,
            ndq: 3,
            p: [p0p, p1p, p2p],
            q: [q0p, q1p, q2p],
        };
    }
    // Weak filter (eqs. 8-395..8-402).
    let mut out = LumaFilterOut {
        ndp: 0,
        ndq: 0,
        p: [p0, p1, p2],
        q: [q0, q1, q2],
    };
    let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4; // eq. 8-395
    if delta.abs() < tc * 10 {
        let delta = delta.clamp(-tc, tc); // eq. 8-396
        out.p[0] = clip(p0 + delta); // eq. 8-397
        out.q[0] = clip(q0 - delta); // eq. 8-398
        if dep == 1 {
            let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1); // eq. 8-399
            out.p[1] = clip(p1 + dp); // eq. 8-400
        }
        if deq == 1 {
            let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1); // eq. 8-401
            out.q[1] = clip(q1 + dq); // eq. 8-402
        }
        out.ndp = (dep + 1) as usize; // eq. 8-403 (nDp = dEp + 1)
        out.ndq = (deq + 1) as usize; // (nDq = dEq + 1)
    }
    out
}

/// §8.7.2.5.3 — decision process for a luma block edge.
///
/// `p` / `q` are the `(i=0..3) × (k=0,3)` sample grids on each side of
/// the edge, laid out as `p[i][k_idx]` where `k_idx` selects row 0 or
/// row 3 of the edge segment (`p[i][0]` = `pi,0`, `p[i][1]` = `pi,3`).
/// `beta` / `tc` come from [`luma_beta_tc`]. Returns the
/// [`LumaEdgeDecision`] (`dE`, `dEp`, `dEq`) that drives §8.7.2.5.4.
///
/// The result's `beta` / `tc` fields echo the inputs so a caller can
/// pass the whole decision straight into [`filter_luma_sample`].
#[must_use]
pub fn luma_edge_decision(
    p: [[i32; 2]; 4],
    q: [[i32; 2]; 4],
    beta: i32,
    tc: i32,
) -> LumaEdgeDecision {
    // Row 0 (k_idx 0) and row 3 (k_idx 1) second differences (eqs.
    // 8-352..8-360 for EDGE_VER, identical form 8-361..8-369 for HOR).
    let dp0 = (p[2][0] - 2 * p[1][0] + p[0][0]).abs(); // eq. 8-352
    let dp3 = (p[2][1] - 2 * p[1][1] + p[0][1]).abs(); // eq. 8-353
    let dq0 = (q[2][0] - 2 * q[1][0] + q[0][0]).abs(); // eq. 8-354
    let dq3 = (q[2][1] - 2 * q[1][1] + q[0][1]).abs(); // eq. 8-355
    let dpq0 = dp0 + dq0; // eq. 8-356
    let dpq3 = dp3 + dq3; // eq. 8-357
    let dp = dp0 + dp3; // eq. 8-358
    let dq = dq0 + dq3; // eq. 8-359
    let d = dpq0 + dpq3; // eq. 8-360

    let mut dec = LumaEdgeDecision {
        de: 0,
        dep: 0,
        deq: 0,
        beta,
        tc,
    };
    if d < beta {
        // Strong/weak split (eqs. 8-360 step 3).
        let dsam0 = luma_sample_decision(p[0][0], p[3][0], q[0][0], q[3][0], 2 * dpq0, beta, tc);
        let dsam3 = luma_sample_decision(p[0][1], p[3][1], q[0][1], q[3][1], 2 * dpq3, beta, tc);
        dec.de = if dsam0 && dsam3 { 2 } else { 1 };
        let thr = (beta + (beta >> 1)) >> 3;
        if dp < thr {
            dec.dep = 1; // eq. step g
        }
        if dq < thr {
            dec.deq = 1; // eq. step h
        }
    }
    dec
}

/// Table 8-10 — `ChromaArrayType == 1` chroma-QP mapping `QpC = f(qPi)`.
///
/// The piecewise mapping is identity below `qPi == 30`, a flat-ish run
/// `29..37` across `qPi 30..43`, then `qPi − 6` for `qPi >= 44`.
#[must_use]
pub fn chroma_qpc_420(qpi: i32) -> i32 {
    match qpi {
        x if x < 30 => x,
        30 => 29,
        31 => 30,
        32 => 31,
        33 => 32,
        34 => 33,
        35 => 33,
        36 => 34,
        37 => 34,
        38 => 35,
        39 => 35,
        40 => 36,
        41 => 36,
        42 => 37,
        43 => 37,
        x => x - 6,
    }
}

/// §8.7.2.5.5 — derive `tC` for a chroma block edge.
///
/// `qp_q` / `qp_p` are the luma `QpY` values of the CUs containing the
/// `q0,0` / `p0,0` chroma samples; `c_qp_pic_offset` is the picture-level
/// `pps_cb_qp_offset` / `pps_cr_qp_offset` for the filtered component;
/// `chroma_array_type` selects Table 8-10 (`== 1`) vs `Min(qPi, 51)`;
/// `tc_offset_div2` is `slice_tc_offset_div2`; `bit_depth` is
/// `BitDepthC`.
///
/// Returns `tC` per eqs. 8-382..8-384.
#[must_use]
pub fn chroma_tc(
    qp_q: i32,
    qp_p: i32,
    c_qp_pic_offset: i32,
    chroma_array_type: u8,
    tc_offset_div2: i32,
    bit_depth: u8,
) -> i32 {
    let qpi = ((qp_q + qp_p + 1) >> 1) + c_qp_pic_offset; // eq. 8-382
    let qp_c = if chroma_array_type == 1 {
        chroma_qpc_420(qpi)
    } else {
        qpi.min(51)
    };
    let q = (qp_c + 2 + (tc_offset_div2 << 1)).clamp(0, 53); // eq. 8-383
    tc_prime(q) * (1 << (i32::from(bit_depth) - 8)) // eq. 8-384
}

/// §8.7.2.5.8 — filtering process for a chroma sample.
///
/// `p` / `q` are the two samples each side of the boundary (`p[0]` is
/// `p0`, closest to the edge); `tc` is from [`chroma_tc`]; `bit_depth`
/// is `BitDepthC` for the final `Clip1C`. Returns `(p0', q0')`. The
/// PCM / transquant-bypass / palette suppressions (which restore the
/// input sample) are applied by the caller.
#[inline]
#[must_use]
pub fn filter_chroma_sample(p: [i32; 2], q: [i32; 2], tc: i32, bit_depth: u8) -> (i32, i32) {
    let [p0, p1] = p;
    let [q0, q1] = q;
    let delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3; // eq. 8-403
    let delta = delta.clamp(-tc, tc);
    let p0p = crate::hevc::engine::picture::clip1(p0 + delta, bit_depth); // eq. 8-404
    let q0p = crate::hevc::engine::picture::clip1(q0 - delta, bit_depth); // eq. 8-405
    (p0p, q0p)
}

/// A mutable sample plane: row-major `samples[y * stride + x]`.
///
/// The §8.7.2.5.4 / .5 block-edge filtering processes read and write
/// this plane in place. Callers wrap their reconstructed luma / chroma
/// component buffer (as `i32` samples) in a [`SamplePlane`].
pub struct SamplePlane<'a> {
    /// Row-major sample storage, `width * height` long.
    pub samples: &'a mut [i32],
    /// Plane width in samples.
    pub width: usize,
    /// Row stride in samples (usually equal to `width`).
    pub stride: usize,
}

impl core::fmt::Debug for SamplePlane<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SamplePlane")
            .field("len", &self.samples.len())
            .field("width", &self.width)
            .field("stride", &self.stride)
            .finish()
    }
}

impl SamplePlane<'_> {
    #[inline]
    fn get(&self, x: usize, y: usize) -> i32 {
        self.samples[y * self.stride + x]
    }
    #[inline]
    fn set(&mut self, x: usize, y: usize, v: i32) {
        self.samples[y * self.stride + x] = v;
    }
    /// The four samples `[x, x + 4)` of row `y`.
    #[inline]
    fn run4(&self, x: usize, y: usize) -> [i32; 4] {
        let mut out = [0i32; 4];
        out.copy_from_slice(&self.samples[y * self.stride + x..][..4]);
        out
    }
    /// Overwrite the four samples `[x, x + 4)` of row `y`.
    #[inline]
    fn set_run4(&mut self, x: usize, y: usize, v: [i32; 4]) {
        self.samples[y * self.stride + x..][..4].copy_from_slice(&v);
    }
    /// `(x, y)` plane coordinates for the `i`-th sample on the p (`pi`,
    /// negative side) or q (`qi`, positive side) of an edge at `(ex, ey)`
    /// for row offset `k` along the edge.
    #[inline]
    fn edge_xy(
        &self,
        ex: usize,
        ey: usize,
        edge: EdgeType,
        side_p: bool,
        i: usize,
        k: usize,
    ) -> (usize, usize) {
        match (edge, side_p) {
            // EDGE_VER: q at x = ex+i, p at x = ex-i-1; both at y = ey+k.
            (EdgeType::Vertical, false) => (ex + i, ey + k),
            (EdgeType::Vertical, true) => (ex - i - 1, ey + k),
            // EDGE_HOR: q at y = ey+i, p at y = ey-i-1; both at x = ex+k.
            (EdgeType::Horizontal, false) => (ex + k, ey + i),
            (EdgeType::Horizontal, true) => (ex + k, ey - i - 1),
        }
    }
}

/// The position + orientation of one 4-row deblocking edge segment.
///
/// `(ex, ey)` is the plane location of `q0,0`: for `EDGE_VER` the column
/// `ex` is the first q-side column; for `EDGE_HOR` the row `ey` is the
/// first q-side row.
#[derive(Debug, Clone, Copy)]
pub struct EdgePos {
    /// `q0,0` x location in the plane.
    pub ex: usize,
    /// `q0,0` y location in the plane.
    pub ey: usize,
    /// Edge orientation (`EDGE_VER` / `EDGE_HOR`).
    pub edge: EdgeType,
}

/// The §8.7.2.5.3 / .5 QP + offset context for one block edge.
///
/// `qp_q` / `qp_p` are the `QpY` values of the CUs containing `q0,0` /
/// `p0,0`; the `*_div2` fields are the per-slice
/// `slice_beta_offset_div2` / `slice_tc_offset_div2`; `bit_depth` is
/// `BitDepthY` (luma) or `BitDepthC` (chroma).
#[derive(Debug, Clone, Copy)]
pub struct EdgeQp {
    /// `QpY` of the q-side CU.
    pub qp_q: i32,
    /// `QpY` of the p-side CU.
    pub qp_p: i32,
    /// `slice_beta_offset_div2` (luma only; ignored for chroma).
    pub beta_offset_div2: i32,
    /// `slice_tc_offset_div2`.
    pub tc_offset_div2: i32,
    /// `BitDepthY` (luma) or `BitDepthC` (chroma).
    pub bit_depth: u8,
}

/// §8.7.2.5.4 — filtering process for a luma block edge.
///
/// Filters the 4-row edge segment positioned by `pos` (the `q0,0`
/// boundary location). `bs` (1 or 2) and the [`EdgeQp`] context drive
/// [`luma_beta_tc`] / [`luma_edge_decision`] / [`filter_luma_sample`].
///
/// Returns the [`LumaEdgeDecision`] that was applied (for inspection /
/// trace); the plane is modified in place. A `bs == 0` edge is a no-op.
///
/// # Panics
/// Panics if the edge segment reads/writes outside the plane.
pub fn filter_luma_block_edge(
    plane: &mut SamplePlane<'_>,
    pos: EdgePos,
    bs: u8,
    qp: EdgeQp,
) -> LumaEdgeDecision {
    filter_luma_block_edge_gated(plane, pos, bs, qp, None)
}

/// [`filter_luma_block_edge`] with the per-CU loop-filter suppression
/// map: a p- / q-side sample whose cell is flagged is left unmodified
/// (the §8.7.2.5.4 `nDp = 0` / `nDq = 0` rule for PCM /
/// transquant-bypass / palette coding units); the filter decision
/// itself still reads both sides.
pub fn filter_luma_block_edge_gated(
    plane: &mut SamplePlane<'_>,
    pos: EdgePos,
    bs: u8,
    qp: EdgeQp,
    no_filter: Option<&NoFilterMap<'_>>,
) -> LumaEdgeDecision {
    let EdgePos { ex, ey, edge } = pos;
    let zero = LumaEdgeDecision {
        de: 0,
        dep: 0,
        deq: 0,
        beta: 0,
        tc: 0,
    };
    if bs == 0 {
        return zero;
    }
    let (beta, tc) = luma_beta_tc(
        qp.qp_q,
        qp.qp_p,
        bs,
        qp.beta_offset_div2,
        qp.tc_offset_div2,
        qp.bit_depth,
    );
    let bit_depth = qp.bit_depth;
    // Gather the whole 4-row segment once: `seg_p[i][k]` = `pi,k`. The
    // four rows are filtered together (they share one decision), so this
    // is also the layout the vectorized kernel consumes.
    // Both orientations gather four contiguous samples per access, so
    // the whole segment costs eight row reads rather than 32 bounds-
    // checked per-sample reads.
    let mut seg_p: crate::hevc::engine::simd::LumaSeg = [[0i32; 4]; 4];
    let mut seg_q: crate::hevc::engine::simd::LumaSeg = [[0i32; 4]; 4];
    match edge {
        EdgeType::Vertical => {
            for k in 0..4 {
                // p3..p0 run left-to-right at x = ex-4 .. ex-1.
                let run = plane.run4(ex - 4, ey + k);
                for i in 0..4 {
                    seg_p[i][k] = run[3 - i];
                }
                let run = plane.run4(ex, ey + k);
                for i in 0..4 {
                    seg_q[i][k] = run[i];
                }
            }
        }
        EdgeType::Horizontal => {
            for i in 0..4 {
                seg_p[i] = plane.run4(ex, ey - i - 1);
                seg_q[i] = plane.run4(ex, ey + i);
            }
        }
    }
    // The decision grid is rows k=0 and k=3 of the same segment.
    let mut pg = [[0i32; 2]; 4];
    let mut qg = [[0i32; 2]; 4];
    for i in 0..4 {
        pg[i] = [seg_p[i][0], seg_p[i][3]];
        qg[i] = [seg_q[i][0], seg_q[i][3]];
    }
    let dec = luma_edge_decision(pg, qg, beta, tc);
    if dec.de == 0 {
        return dec;
    }
    // Apply the filter to all 4 rows at once (eq. 8-372/8-373 layout).
    // Rows the weak filter leaves alone come back holding their input
    // samples, so writing `0..nDp` / `0..nDq` back is a no-op for them.
    let mut out_p: crate::hevc::engine::simd::LumaSegOut = [[0i32; 4]; 3];
    let mut out_q: crate::hevc::engine::simd::LumaSegOut = [[0i32; 4]; 3];
    crate::hevc::engine::simd::filter_luma_rows(
        &seg_p, &seg_q, dec.de, dec.dep, dec.deq, tc, bit_depth, &mut out_p, &mut out_q,
    );
    let ndp = if dec.de == 2 {
        3
    } else {
        (dec.dep + 1) as usize
    }; // eq. 8-403
    let ndq = if dec.de == 2 {
        3
    } else {
        (dec.deq + 1) as usize
    };
    if no_filter.is_none() && edge == EdgeType::Horizontal {
        // Each filtered `pi'` / `qi'` row is four contiguous samples.
        for (i, row) in out_p.iter().enumerate().take(ndp) {
            plane.set_run4(ex, ey - i - 1, *row);
        }
        for (j, row) in out_q.iter().enumerate().take(ndq) {
            plane.set_run4(ex, ey + j, *row);
        }
        return dec;
    }
    for k in 0..4 {
        for (i, row) in out_p.iter().enumerate().take(ndp) {
            let (px, py) = plane.edge_xy(ex, ey, edge, true, i, k);
            if no_filter.is_some_and(|m| m.at_luma(px, py)) {
                continue; // §8.7.2.5.4 nDp = 0 (PCM / bypass / palette)
            }
            plane.set(px, py, row[k]);
        }
        for (j, row) in out_q.iter().enumerate().take(ndq) {
            let (qx, qy) = plane.edge_xy(ex, ey, edge, false, j, k);
            if no_filter.is_some_and(|m| m.at_luma(qx, qy)) {
                continue; // §8.7.2.5.4 nDq = 0
            }
            plane.set(qx, qy, row[k]);
        }
    }
    dec
}

/// §8.7.2.5.5 — filtering process for a chroma block edge.
///
/// Filters the 4-row chroma edge segment positioned by `pos` (the `q0,0`
/// chroma location). The [`EdgeQp`] context carries the luma `QpY`
/// values + `slice_tc_offset_div2` + `BitDepthC`; `c_qp_pic_offset` is
/// the picture chroma offset; `chroma_array_type` selects Table 8-10
/// (`== 1`) vs `Min(qPi, 51)`. Only `p0`/`q0` are modified (eqs.
/// 8-385..8-388). Returns the `tC` that was applied.
///
/// # Panics
/// Panics if the edge segment reads/writes outside the plane.
pub fn filter_chroma_block_edge(
    plane: &mut SamplePlane<'_>,
    pos: EdgePos,
    qp: EdgeQp,
    c_qp_pic_offset: i32,
    chroma_array_type: u8,
) -> i32 {
    filter_chroma_block_edge_gated(plane, pos, qp, c_qp_pic_offset, chroma_array_type, None)
}

/// [`filter_chroma_block_edge`] with the per-CU loop-filter
/// suppression map (§8.7.2.5.5: the flagged side's `p0'` / `q0'` is
/// not written). Chroma sample positions map onto the luma cell grid
/// through the `SubWidthC` / `SubHeightC` factors of
/// `chroma_array_type`.
pub fn filter_chroma_block_edge_gated(
    plane: &mut SamplePlane<'_>,
    pos: EdgePos,
    qp: EdgeQp,
    c_qp_pic_offset: i32,
    chroma_array_type: u8,
    no_filter: Option<&NoFilterMap<'_>>,
) -> i32 {
    let (sub_w, sub_h) = sub_wh_c(chroma_array_type);
    let EdgePos { ex, ey, edge } = pos;
    let bit_depth = qp.bit_depth;
    let tc = chroma_tc(
        qp.qp_q,
        qp.qp_p,
        c_qp_pic_offset,
        chroma_array_type,
        qp.tc_offset_div2,
        bit_depth,
    );
    // Gather the whole 4-row segment (`seg_p[i][k]` = `pi,k`) and filter
    // its four rows together.
    let mut seg_p: crate::hevc::engine::simd::ChromaSeg = [[0i32; 4]; 2];
    let mut seg_q: crate::hevc::engine::simd::ChromaSeg = [[0i32; 4]; 2];
    match edge {
        EdgeType::Vertical => {
            for k in 0..4 {
                let run = plane.run4(ex - 2, ey + k);
                seg_p[0][k] = run[1];
                seg_p[1][k] = run[0];
                seg_q[0][k] = run[2];
                seg_q[1][k] = run[3];
            }
        }
        EdgeType::Horizontal => {
            for i in 0..2 {
                seg_p[i] = plane.run4(ex, ey - i - 1);
                seg_q[i] = plane.run4(ex, ey + i);
            }
        }
    }
    let mut out_p0 = [0i32; 4];
    let mut out_q0 = [0i32; 4];
    crate::hevc::engine::simd::filter_chroma_rows(
        &seg_p,
        &seg_q,
        tc,
        bit_depth,
        &mut out_p0,
        &mut out_q0,
    );
    if no_filter.is_none() && edge == EdgeType::Horizontal {
        plane.set_run4(ex, ey - 1, out_p0);
        plane.set_run4(ex, ey, out_q0);
        return tc;
    }
    for k in 0..4 {
        let (px, py) = plane.edge_xy(ex, ey, edge, true, 0, k);
        let (qx, qy) = plane.edge_xy(ex, ey, edge, false, 0, k);
        if !no_filter.is_some_and(|m| m.at_luma(px * sub_w, py * sub_h)) {
            plane.set(px, py, out_p0[k]);
        }
        if !no_filter.is_some_and(|m| m.at_luma(qx * sub_w, qy * sub_h)) {
            plane.set(qx, qy, out_q0[k]);
        }
    }
    tc
}

/// The §8.7.2.5.1 / .2 per-CU deblocking context: the QP + offset state a
/// coding unit contributes to its edge filtering.
///
/// `qp_y` is the CU's `QpY`; the `*_offset_div2` are the per-slice
/// `slice_beta_offset_div2` / `slice_tc_offset_div2`; the `pps_c*_qp_offset`
/// are the picture chroma QP offsets; the bit depths are `BitDepthY` /
/// `BitDepthC`. The driver reads neighbour `QpY` from `qp_y_at` so a
/// boundary edge can use the p-side CU's `QpY` for `qP,p` (eq. 8-347).
#[derive(Debug, Clone, Copy)]
pub struct DeblockCuParams {
    /// The CU's `QpY` (the q-side QP at every edge inside the CU).
    pub qp_y: i32,
    /// `slice_beta_offset_div2`.
    pub beta_offset_div2: i32,
    /// `slice_tc_offset_div2`.
    pub tc_offset_div2: i32,
    /// `pps_cb_qp_offset` for the Cb component — the §8.7.2.5.1/.2
    /// `cQpPicOffset` (slice-level and CU-level chroma QP adjustments
    /// are excluded from chroma deblocking by the invocation text).
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset` for the Cr component.
    pub cr_qp_offset: i32,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
    /// `ChromaArrayType` (selects Table 8-10 / `Min(qPi,51)` + subsampling).
    pub chroma_array_type: u8,
}

/// The geometry + QP context of one coding unit for the §8.7.2.5.1 / .2
/// edge-filtering driver.
#[derive(Debug, Clone, Copy)]
pub struct DeblockCu {
    /// The CU's luma top-left x.
    pub x_cb: usize,
    /// The CU's luma top-left y.
    pub y_cb: usize,
    /// `log2CbSize`.
    pub log2_cb_size: u32,
    /// The CU QP / offset / chroma context.
    pub params: DeblockCuParams,
    /// The left-neighbour CU's `QpY` at the vertical coding-block
    /// boundary edge (xDk == 0); interior edges use `params.qp_y` on
    /// both sides.
    pub qp_y_p_left: i32,
    /// The above-neighbour CU's `QpY` at the horizontal coding-block
    /// boundary edge (yDm == 0).
    pub qp_y_p_top: i32,
}

/// §8.7.2.5.1 / .2 — filter all edges of one coding unit in one direction.
///
/// Applies the §8.7.2.5.4 luma block-edge filter at every `bS > 0` sampled
/// position, then (when `ChromaArrayType != 0`) the §8.7.2.5.5 chroma
/// block-edge filter at every `bS == 2`, 8-luma-sample-aligned position.
/// The luma sampling grid is 8-stride along the edge axis and 4-stride
/// across; the chroma grid steps `8 / SubWidthC` (vertical) /
/// `8 / SubHeightC` (horizontal) in the edge axis.
///
/// `cu` carries the CU's geometry + QP context; `bs` is the §8.7.2.4 grid
/// for `edge_type`.
///
/// # Panics
/// Panics if a filtered edge segment reads/writes outside a plane.
pub fn filter_cu_edges(
    pic: &mut Picture,
    cu: &DeblockCu,
    edge_type: EdgeType,
    bs: &BoundaryStrength,
) {
    filter_cu_edges_with_qp_map(pic, cu, edge_type, bs, None);
}

/// [`filter_cu_edges`] with the per-position [`QpMap`] (exact `QpQ` /
/// `QpP` when quantization groups vary inside / around the CU).
pub fn filter_cu_edges_with_qp_map(
    pic: &mut Picture,
    cu: &DeblockCu,
    edge_type: EdgeType,
    bs: &BoundaryStrength,
    qp_map: Option<QpMap<'_>>,
) {
    filter_cu_edges_full(pic, cu, edge_type, bs, qp_map, None);
}

/// [`filter_cu_edges_with_qp_map`] with the per-CU loop-filter
/// suppression map (§8.7.2.5.4 / §8.7.2.5.5 `nDp` / `nDq` zeroing).
pub fn filter_cu_edges_full(
    pic: &mut Picture,
    cu: &DeblockCu,
    edge_type: EdgeType,
    bs: &BoundaryStrength,
    qp_map: Option<QpMap<'_>>,
    no_filter: Option<&NoFilterMap<'_>>,
) {
    let DeblockCu {
        x_cb,
        y_cb,
        log2_cb_size,
        params,
        qp_y_p_left,
        qp_y_p_top,
    } = *cu;
    let n_d = 1usize << (log2_cb_size - 3);
    let (sub_w, sub_h) = sub_wh_c(params.chroma_array_type);

    // ----- luma (§8.7.2.5.1 step 2 / §8.7.2.5.2 step 2) -----
    // EDGE_VER: xDk = k<<3 (k=0..nD-1), yDm = m<<2 (m=0..2nD-1).
    // EDGE_HOR: yDm = m<<3 (m=0..nD-1), xDk = k<<2 (k=0..2nD-1).
    // `edges` indexes the 8-strided axis (nD); `segs` the 4-strided axis.
    let (n_edges, n_segs) = (n_d, n_d * 2);
    for e in 0..n_edges {
        for s_idx in 0..n_segs {
            let (x_dk, y_dm) = match edge_type {
                EdgeType::Vertical => (e << 3, s_idx << 2),
                EdgeType::Horizontal => (s_idx << 2, e << 3),
            };
            let s = bs.at(x_dk, y_dm);
            if s == 0 {
                continue;
            }
            // p-side QpY: the neighbour CU only at the coding-block
            // boundary edge; interior edges share the CU's QpY.
            let (boundary, qp_y_p) = match edge_type {
                EdgeType::Vertical => (x_dk == 0, qp_y_p_left),
                EdgeType::Horizontal => (y_dm == 0, qp_y_p_top),
            };
            // §8.7.2.5.3: QpQ / QpP are the QpY of the coding units
            // containing the q0 / p0 samples — per position from the QP
            // map when supplied, else the per-CU scalars.
            let (qp_q, qp_p) = match qp_map {
                Some(m) => {
                    let (qx, qy) = (x_cb + x_dk, y_cb + y_dm);
                    let (px, py) = match edge_type {
                        EdgeType::Vertical => (qx - 1, qy),
                        EdgeType::Horizontal => (qx, qy - 1),
                    };
                    (m.qp_at(qx, qy), m.qp_at(px, py))
                }
                None => (params.qp_y, if boundary { qp_y_p } else { params.qp_y }),
            };
            let qp = EdgeQp {
                qp_q,
                qp_p,
                beta_offset_div2: params.beta_offset_div2,
                tc_offset_div2: params.tc_offset_div2,
                bit_depth: params.bit_depth_luma,
            };
            let (buf, stride) = pic.plane_mut(Plane::Luma);
            let mut plane = SamplePlane {
                samples: buf,
                width: stride,
                stride,
            };
            let pos = EdgePos {
                ex: x_cb + x_dk,
                ey: y_cb + y_dm,
                edge: edge_type,
            };
            filter_luma_block_edge_gated(&mut plane, pos, s, qp, no_filter);
        }
    }

    // ----- chroma (§8.7.2.5.1 / .2 chroma steps) -----
    if params.chroma_array_type == 0 {
        return;
    }
    let (xc_cb, yc_cb) = (x_cb / sub_w, y_cb / sub_h);
    // edgeSpacing / edgeSections per direction.
    let (spacing, sections, n_edges) = match edge_type {
        EdgeType::Vertical => (8 / sub_w, n_d * (2 / sub_h), n_d),
        EdgeType::Horizontal => (8 / sub_h, n_d * (2 / sub_w), n_d),
    };
    for plane in [Plane::Cb, Plane::Cr] {
        let c_qp_offset = if plane == Plane::Cb {
            params.cb_qp_offset
        } else {
            params.cr_qp_offset
        };
        // EDGE_VER: xDk = k*spacing (k=0..nD-1), yDm = m<<2 (m=0..sections-1).
        // EDGE_HOR: yDm = m*spacing, xDk = k<<2.
        let (k_max, m_max) = match edge_type {
            EdgeType::Vertical => (n_edges, sections),
            EdgeType::Horizontal => (sections, n_edges),
        };
        for kk in 0..k_max {
            for mm in 0..m_max {
                let (x_dk, y_dm) = match edge_type {
                    EdgeType::Vertical => (kk * spacing, mm << 2),
                    EdgeType::Horizontal => (kk << 2, mm * spacing),
                };
                // bS[xDk*SubWidthC][yDm*SubHeightC] == 2.
                let bx = x_dk * sub_w;
                let by = y_dm * sub_h;
                if bs.at(bx, by) != 2 {
                    continue;
                }
                // 8-luma-grid alignment of the chroma edge position.
                let aligned = match edge_type {
                    EdgeType::Vertical => ((xc_cb + x_dk) >> 3) << 3 == xc_cb + x_dk,
                    EdgeType::Horizontal => ((yc_cb + y_dm) >> 3) << 3 == yc_cb + y_dm,
                };
                if !aligned {
                    continue;
                }
                let (boundary, qp_y_p) = match edge_type {
                    EdgeType::Vertical => (bx == 0, qp_y_p_left),
                    EdgeType::Horizontal => (by == 0, qp_y_p_top),
                };
                // §8.7.2.5.5: QpQ / QpP from the coding units containing
                // the chroma edge's q0 / p0 samples (luma positions).
                let (qp_q, qp_p) = match qp_map {
                    Some(m) => {
                        let (qx, qy) = ((xc_cb + x_dk) * sub_w, (yc_cb + y_dm) * sub_h);
                        let (px, py) = match edge_type {
                            EdgeType::Vertical => (qx - 1, qy),
                            EdgeType::Horizontal => (qx, qy - 1),
                        };
                        (m.qp_at(qx, qy), m.qp_at(px, py))
                    }
                    None => (params.qp_y, if boundary { qp_y_p } else { params.qp_y }),
                };
                let qp = EdgeQp {
                    qp_q,
                    qp_p,
                    beta_offset_div2: params.beta_offset_div2,
                    tc_offset_div2: params.tc_offset_div2,
                    bit_depth: params.bit_depth_chroma,
                };
                let (buf, stride) = pic.plane_mut(plane);
                let mut cplane = SamplePlane {
                    samples: buf,
                    width: stride,
                    stride,
                };
                let pos = EdgePos {
                    ex: xc_cb + x_dk,
                    ey: yc_cb + y_dm,
                    edge: edge_type,
                };
                filter_chroma_block_edge_gated(
                    &mut cplane,
                    pos,
                    qp,
                    c_qp_offset,
                    params.chroma_array_type,
                    no_filter,
                );
            }
        }
    }
}

/// One coding unit's full §8.7.2 deblocking description, as the
/// picture-level driver [`deblock_picture`] consumes it.
///
/// Carries everything the §8.7.2.2 / .3 edge-flag derivation, the §8.7.2.4
/// bS derivation and the §8.7.2.5.1 / .2 filtering need for one CU: its
/// geometry + QP context ([`DeblockCu`]), its transform-tree split
/// ([`TransformSplit`]) + prediction [`PartMode`], and the §8.7.2.1
/// `filterLeftCbEdgeFlag` / `filterTopCbEdgeFlag` the caller computes from
/// the picture / tile / slice geometry (`false` at a non-filtered
/// boundary).
#[derive(Debug, Clone)]
pub struct DeblockCuDesc {
    /// Geometry + QP context.
    pub cu: DeblockCu,
    /// The CU's transform-tree split structure.
    pub transform_split: TransformSplit,
    /// The CU's prediction partition mode.
    pub part_mode: PartMode,
    /// §8.7.2.1 `filterLeftCbEdgeFlag` (gates the left CB-boundary
    /// vertical edge).
    pub filter_left: bool,
    /// §8.7.2.1 `filterTopCbEdgeFlag` (gates the top CB-boundary
    /// horizontal edge).
    pub filter_top: bool,
}

/// §8.7.2.1 — the picture-level deblocking filter driver.
///
/// Filters all vertical edges of the picture first, then all horizontal
/// edges (with the vertically-filtered samples as input), per §8.7.2.1.
/// Each pass walks `cus` in order and, for each CU: derives the
/// §8.7.2.2 / .3 edge flags, the §8.7.2.4 boundary strengths from `field`,
/// then applies the §8.7.2.5.1 / .2 luma + chroma filters via
/// [`filter_cu_edges`].
///
/// `cus` must be in coding (z-scan) order; the caller is responsible for
/// the §8.7.2.1 boundary exclusions (picture / tile / slice edges, and the
/// `slice_deblocking_filter_disabled_flag` skip) by setting each
/// descriptor's `filter_left` / `filter_top` (and omitting CUs whose slice
/// disables deblocking).
///
/// # Panics
/// Panics if a CU's edges read/write outside a `pic` plane.
pub fn deblock_picture(pic: &mut Picture, field: &MotionField, cus: &[DeblockCuDesc]) {
    deblock_picture_with_qp_map(pic, field, cus, None);
}

/// Per-4×4-cell `QpY` map for the §8.7.2.5.3 / §8.7.2.5.5 per-position
/// `QpQ` / `QpP` reads (row-major over the picture's 4×4 cells). When a
/// picture carries per-quantization-group `cu_qp_delta` values, the
/// p-side QP can change along a single coding-block edge, so the
/// per-CU scalar fallback in [`DeblockCu`] is not exact.
#[derive(Debug, Clone, Copy)]
pub struct QpMap<'a> {
    /// Per-cell `QpY`.
    pub cells: &'a [i8],
    /// Cell-grid width (`ceil(pic_width / 4)`).
    pub w_cells: usize,
}

impl QpMap<'_> {
    fn qp_at(&self, x_luma: usize, y_luma: usize) -> i32 {
        let idx = (y_luma >> 2) * self.w_cells + (x_luma >> 2);
        self.cells.get(idx).copied().map_or(0, i32::from)
    }
}

/// Per-4×4-luma-cell in-loop-filter suppression map: `true` cells
/// belong to a coding unit whose reconstructed samples the loop
/// filters must not modify — a `pcm_flag == 1` CU when
/// `pcm_loop_filter_disabled_flag == 1` (§7.4.3.2.1), or a
/// `cu_transquant_bypass_flag == 1` CU (§7.4.9.5). The §8.7.2.5.4 /
/// §8.7.2.5.5 filters zero `nDp` / `nDq` on the flagged side (the
/// decision still reads both sides); §8.7.3 SAO skips flagged samples.
#[derive(Debug, Clone, Copy)]
pub struct NoFilterMap<'a> {
    /// Per-cell suppression flags, row-major.
    pub cells: &'a [bool],
    /// Cell-grid width (`ceil(pic_width / 4)`).
    pub w_cells: usize,
}

impl NoFilterMap<'_> {
    /// Suppression flag of the cell covering luma sample
    /// `(x_luma, y_luma)` (out-of-range positions are not suppressed).
    #[must_use]
    pub fn at_luma(&self, x_luma: usize, y_luma: usize) -> bool {
        self.cells
            .get((y_luma >> 2) * self.w_cells + (x_luma >> 2))
            .copied()
            .unwrap_or(false)
    }
}

/// [`deblock_picture`] with the per-position QP map.
pub fn deblock_picture_with_qp_map(
    pic: &mut Picture,
    field: &MotionField,
    cus: &[DeblockCuDesc],
    qp_map: Option<QpMap<'_>>,
) {
    deblock_picture_full(pic, field, cus, qp_map, None);
}

/// [`deblock_picture_with_qp_map`] with the per-CU loop-filter
/// suppression map (§8.7.2.5.4 / §8.7.2.5.5 — PCM with
/// `pcm_loop_filter_disabled_flag` and transquant-bypass coding units
/// keep their reconstructed samples).
pub fn deblock_picture_full(
    pic: &mut Picture,
    field: &MotionField,
    cus: &[DeblockCuDesc],
    qp_map: Option<QpMap<'_>>,
    no_filter: Option<&NoFilterMap<'_>>,
) {
    // Pass 1: all vertical edges.
    for desc in cus {
        apply_cu_direction(pic, field, desc, EdgeType::Vertical, qp_map, no_filter);
    }
    // Pass 2: all horizontal edges (on the vertically-filtered samples).
    for desc in cus {
        apply_cu_direction(pic, field, desc, EdgeType::Horizontal, qp_map, no_filter);
    }
}

/// Derive the edge flags + bS for one CU in one direction and filter it.
fn apply_cu_direction(
    pic: &mut Picture,
    field: &MotionField,
    desc: &DeblockCuDesc,
    edge_type: EdgeType,
    qp_map: Option<QpMap<'_>>,
    no_filter: Option<&NoFilterMap<'_>>,
) {
    let DeblockCu {
        x_cb,
        y_cb,
        log2_cb_size,
        ..
    } = desc.cu;
    let filter_edge_flag = match edge_type {
        EdgeType::Vertical => desc.filter_left,
        EdgeType::Horizontal => desc.filter_top,
    };
    let ef = derive_edge_flags(
        &desc.transform_split,
        desc.part_mode,
        log2_cb_size,
        edge_type,
        filter_edge_flag,
    );
    let bs = derive_boundary_strength(
        field,
        x_cb,
        y_cb,
        log2_cb_size,
        edge_type,
        ef.edge_flags(),
        ef.tb_edge(),
    );
    filter_cu_edges_full(pic, &desc.cu, edge_type, &bs, qp_map, no_filter);
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::motion::MotionCell;

    fn inter_cell(mv0: [i32; 2], poc0: i32) -> MotionCell {
        MotionCell {
            is_intra: false,
            has_nonzero_coeff: false,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_poc_l0: poc0,
            ref_poc_l1: i32::MIN,
            mv_l0: mv0,
            mv_l1: [0, 0],
        }
    }

    /// A 16×16 CU's internal vertical edge at xDi=8, with the left
    /// (p-side) half intra ⇒ bS == 2 at the sampled positions.
    ///
    /// The §8.7.2.4 vertical-edge x stride is 8, so a 16-wide CU samples
    /// xDi ∈ {0, 8}; the internal edge is at xDi=8.
    #[test]
    fn intra_neighbour_gives_bs2() {
        let mut field = MotionField::new(32, 32);
        // CU at (0,0), 16×16. q-side (right half) inter; p-side (left)
        // stays intra (background).
        field.fill_rect(8, 0, 8, 16, inter_cell([0, 0], 0));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        // Internal vertical edge at xDi=8, sampled yDj 0,4,8,12.
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        // p0 = col 7 (intra), q0 = col 8 (inter) ⇒ bS == 2.
        assert_eq!(bs.at(8, 0), 2);
        assert_eq!(bs.at(8, 12), 2);
        // xDi=4 not sampled for EDGE_VER (8-stride).
        assert_eq!(bs.at(4, 0), 0);
    }

    /// Transform-block edge with a non-zero coefficient on an inter block
    /// ⇒ bS == 1.
    #[test]
    fn tb_edge_with_coeff_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        let mut p = inter_cell([0, 0], 0);
        p.has_nonzero_coeff = true;
        let q = inter_cell([0, 0], 0);
        field.fill_rect(0, 0, 16, 16, q);
        field.fill_rect(0, 0, 8, 16, p); // left half carries the coeff
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        // Vertical edge at xDi=8 (a transform-block edge).
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
            tb_edge[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        // p0 = col 7 (coeff block), q0 = col 8 ⇒ bS == 1.
        assert_eq!(bs.at(8, 0), 1);
        assert_eq!(bs.at(8, 4), 1);
    }

    /// Two inter blocks, same reference, MV difference >= 4 ⇒ bS == 1.
    #[test]
    fn mv_diff_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 8, 16, inter_cell([0, 0], 0)); // p side
        field.fill_rect(8, 0, 8, 16, inter_cell([4, 0], 0)); // q side, Δ=4
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
        assert_eq!(bs.at(8, 12), 1);
    }

    /// Two inter blocks, same reference + same MV ⇒ bS == 0.
    #[test]
    fn same_motion_gives_bs0() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 16, 16, inter_cell([3, 1], 0)); // identical motion
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 0);
    }

    /// Different reference pictures ⇒ bS == 1 even with identical MVs.
    #[test]
    fn different_ref_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 8, 16, inter_cell([2, 2], 0)); // poc 0
        field.fill_rect(8, 0, 8, 16, inter_cell([2, 2], 8)); // poc 8
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
    }

    /// Horizontal edges sample 4-stride x, 8-stride y.
    #[test]
    fn horizontal_edge_sampling() {
        let mut field = MotionField::new(32, 32);
        // top half intra (background), bottom half inter; internal
        // horizontal edge at yDj=8.
        field.fill_rect(0, 8, 16, 8, inter_cell([0, 0], 0));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for xi in (0..16).step_by(4) {
            edge_flags[8 * n + xi] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Horizontal, &edge_flags, &tb_edge);
        // yDj=8: p0 = row 7 (intra), q0 = row 8 (inter) ⇒ bS == 2.
        assert_eq!(bs.at(0, 8), 2);
        assert_eq!(bs.at(12, 8), 2);
        // yDj=4 not sampled for EDGE_HOR.
        assert_eq!(bs.at(0, 4), 0);
    }

    /// edgeFlags == 0 ⇒ bS stays 0 regardless of mode.
    #[test]
    fn no_edge_flag_gives_bs0() {
        let field = MotionField::new(16, 16); // intra background
        let n = 8;
        let edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        let bs =
            derive_boundary_strength(&field, 0, 0, 3, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(0, 0), 0);
    }

    /// Bullet-4 path: two MVs each, same single reference picture on both
    /// sides; bS == 1 only when both the straight and crossed MV-pair
    /// differences reach 4.
    #[test]
    fn bipred_same_ref_bullet4() {
        let mut field = MotionField::new(32, 32);
        let bi = |mv0: [i32; 2], mv1: [i32; 2]| MotionCell {
            is_intra: false,
            has_nonzero_coeff: false,
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_poc_l0: 0,
            ref_poc_l1: 0, // same reference picture for both lists
            mv_l0: mv0,
            mv_l1: mv1,
        };
        // p: l0=[0,0] l1=[0,0]; q: l0=[4,0] l1=[4,0].
        // straight: |Δl0|>=4 ⇒ true. crossed: |p.l0−q.l1|=4 ⇒ true ⇒ bS 1.
        field.fill_rect(0, 0, 8, 16, bi([0, 0], [0, 0]));
        field.fill_rect(8, 0, 8, 16, bi([4, 0], [4, 0]));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
    }

    // ----- §8.7.2.2 / §8.7.2.3 edge-flag derivation -----

    /// §8.7.2.2: an un-split 16×16 CB marks only its left CB-boundary
    /// column (EDGE_VER), gated by filterEdgeFlag. No internal TB edges.
    #[test]
    fn edge_flags_single_tb_left_boundary() {
        let n = 16;
        // filterEdgeFlag = true ⇒ the CB-left column (x=0) is an edge.
        let ef = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::Part2Nx2N,
            4,
            EdgeType::Vertical,
            true,
        );
        for y in 0..n {
            assert!(ef.edge_at(0, y), "left CB-boundary column must be set");
            assert!(ef.tb_at(0, y), "left CB-boundary is a TB edge");
        }
        // No internal column is set (no split, PART_2Nx2N).
        for x in 1..n {
            for y in 0..n {
                assert!(!ef.edge_at(x, y), "interior ({x},{y}) must be clear");
            }
        }
    }

    /// filterEdgeFlag = false suppresses the CB-boundary edge entirely.
    #[test]
    fn edge_flags_cb_boundary_gated_off() {
        let ef = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::Part2Nx2N,
            4,
            EdgeType::Vertical,
            false,
        );
        for y in 0..16 {
            assert!(!ef.edge_at(0, y), "gated-off CB boundary stays clear");
        }
    }

    /// §8.7.2.2: a one-level transform split of a 16×16 CB marks the
    /// internal vertical TB edge at column 8 (always 1, not gated), plus
    /// the gated left boundary.
    #[test]
    fn edge_flags_split_marks_internal_tb_edge() {
        let ef = derive_edge_flags(
            &TransformSplit::split_once(),
            PartMode::Part2Nx2N,
            4,
            EdgeType::Vertical,
            true,
        );
        for y in 0..16 {
            assert!(ef.edge_at(0, y), "left boundary set");
            assert!(ef.edge_at(8, y), "internal split column 8 set");
            assert!(ef.tb_at(8, y), "internal split is a TB edge");
        }
    }

    /// §8.7.2.2 EDGE_HOR: the split marks the internal horizontal TB edge
    /// at row 8.
    #[test]
    fn edge_flags_split_horizontal() {
        let ef = derive_edge_flags(
            &TransformSplit::split_once(),
            PartMode::Part2Nx2N,
            4,
            EdgeType::Horizontal,
            true,
        );
        for x in 0..16 {
            assert!(ef.edge_at(x, 0), "top boundary set");
            assert!(ef.edge_at(x, 8), "internal split row 8 set");
            assert!(ef.tb_at(x, 8), "internal split is a TB edge");
        }
    }

    /// §8.7.2.3 PART_Nx2N marks the prediction column at nCbS/2 — and it
    /// is NOT a transform-block edge (tb_edge stays clear there).
    #[test]
    fn edge_flags_part_nx2n_prediction_edge() {
        let ef = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::PartNx2N,
            4,
            EdgeType::Vertical,
            true,
        );
        for y in 0..16 {
            assert!(ef.edge_at(8, y), "PART_Nx2N prediction column 8 set");
            assert!(!ef.tb_at(8, y), "prediction edge is not a TB edge");
        }
    }

    /// §8.7.2.3 AMP PART_nLx2N marks column nCbS/4; PART_nRx2N marks
    /// 3·nCbS/4.
    #[test]
    fn edge_flags_amp_columns() {
        let l = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::PartNLx2N,
            4,
            EdgeType::Vertical,
            true,
        );
        assert!(l.edge_at(4, 0), "PART_nLx2N column 4");
        let r = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::PartNRx2N,
            4,
            EdgeType::Vertical,
            true,
        );
        assert!(r.edge_at(12, 0), "PART_nRx2N column 12");
    }

    /// §8.7.2.3 horizontal AMP: PART_2NxnU row nCbS/4, PART_2NxnD row
    /// 3·nCbS/4.
    #[test]
    fn edge_flags_amp_rows() {
        let u = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::Part2NxnU,
            4,
            EdgeType::Horizontal,
            true,
        );
        assert!(u.edge_at(0, 4), "PART_2NxnU row 4");
        let d = derive_edge_flags(
            &TransformSplit::leaf(),
            PartMode::Part2NxnD,
            4,
            EdgeType::Horizontal,
            true,
        );
        assert!(d.edge_at(0, 12), "PART_2NxnD row 12");
    }

    /// A nested split (one quadrant splits again) marks the deeper TB edge
    /// at the 4-sample sub-block column.
    #[test]
    fn edge_flags_nested_split() {
        // 16×16 CB: TL quadrant (8×8) splits into four 4×4; others leaf.
        let nested = TransformSplit::Split(Box::new([
            TransformSplit::split_once(), // TL 8×8 → 4×4 leaves
            TransformSplit::Leaf,
            TransformSplit::Leaf,
            TransformSplit::Leaf,
        ]));
        let ef = derive_edge_flags(&nested, PartMode::Part2Nx2N, 4, EdgeType::Vertical, true);
        // The TL quadrant's internal vertical edge is at column 4.
        for y in 0..8 {
            assert!(ef.edge_at(4, y), "nested TB edge at column 4, row {y}");
            assert!(ef.tb_at(4, y));
        }
        // The CB-internal split at column 8 is still present.
        for y in 0..16 {
            assert!(ef.edge_at(8, y), "CB-level split column 8");
        }
        // Column 4 below the TL quadrant (y>=8) is not an edge (BL leaf is
        // 8 wide, its only internal edge would be column 8, not 4).
        assert!(!ef.edge_at(4, 8), "column 4 row 8 clear (BL leaf)");
    }

    /// The §8.7.2.4 bS derivation consumes the derived flags: an
    /// intra/inter split CU with a derived internal TB edge yields bS=2
    /// at the internal column.
    #[test]
    fn edge_flags_feed_boundary_strength() {
        // filterEdgeFlag = false: this CB is at the picture's left edge, so
        // only the internal split column (8) is an edge — the bS sampler
        // never reads a p-sample at x = −1.
        let ef = derive_edge_flags(
            &TransformSplit::split_once(),
            PartMode::Part2Nx2N,
            4,
            EdgeType::Vertical,
            false,
        );
        let mut field = MotionField::new(32, 32);
        // Left half intra (background), right half inter.
        field.fill_rect(8, 0, 8, 16, inter_cell([0, 0], 0));
        let bs = derive_boundary_strength(
            &field,
            0,
            0,
            4,
            EdgeType::Vertical,
            ef.edge_flags(),
            ef.tb_edge(),
        );
        // Internal column 8: p (col 7, intra) vs q (col 8, inter) ⇒ bS 2.
        assert_eq!(bs.at(8, 0), 2);
    }

    // ----- §8.7.2.5.3 / .6 / .7 luma filtering primitives -----

    /// Table 8-12 spot checks: β′ is 0 below Q=16 and tracks the listed
    /// breakpoints; tC′ is 0 up to Q=17 and 1 at Q=18.
    #[test]
    fn table_8_12_breakpoints() {
        assert_eq!(beta_prime(15), 0);
        assert_eq!(beta_prime(16), 6);
        assert_eq!(beta_prime(29), 20);
        assert_eq!(beta_prime(51), 64);
        assert_eq!(tc_prime(17), 0);
        assert_eq!(tc_prime(18), 1);
        assert_eq!(tc_prime(53), 24);
        // Clip behaviour at the table edges.
        assert_eq!(beta_prime(-3), 0);
        assert_eq!(tc_prime(99), 24);
    }

    /// §8.7.2.5.3 β/tC at 8-bit: qPL = (QpQ+QpP+1)>>1, no offsets, bS=2.
    /// QpQ=QpP=37 ⇒ qPL=37 ⇒ Q_beta=37 ⇒ β′=36, β=36; Q_tc=37+2=39 ⇒
    /// tC′=5, tC=5.
    #[test]
    fn luma_beta_tc_no_offset_8bit() {
        let (beta, tc) = luma_beta_tc(37, 37, 2, 0, 0, 8);
        assert_eq!(beta, 36);
        assert_eq!(tc, 5);
        // bS=1 lowers the tC index by 2 (Q_tc=37): tC′=4.
        let (_, tc1) = luma_beta_tc(37, 37, 1, 0, 0, 8);
        assert_eq!(tc1, 4);
    }

    /// 10-bit scaling: β and tC are multiplied by 1<<(BitDepthY−8)=4.
    #[test]
    fn luma_beta_tc_10bit_scaled() {
        let (beta, tc) = luma_beta_tc(37, 37, 2, 0, 0, 10);
        assert_eq!(beta, 36 * 4);
        assert_eq!(tc, 5 * 4);
    }

    /// §8.7.2.5.6: a flat region (all-equal samples) passes the decision.
    #[test]
    fn luma_sample_decision_flat_passes() {
        // dpq=0 < β>>2; |p3−p0|+|q0−q3|=0 < β>>3; |p0−q0|=0 < (5tC+1)>>1.
        assert!(luma_sample_decision(128, 128, 128, 128, 0, 32, 4));
        // A large step at the boundary fails the |p0−q0| test.
        assert!(!luma_sample_decision(64, 64, 200, 200, 0, 32, 4));
    }

    /// §8.7.2.5.7 strong filter: a flat block stays flat (idempotent).
    #[test]
    fn strong_filter_flat_is_idempotent() {
        let s = [100; 4];
        let out = filter_luma_sample(s, s, 2, 0, 0, 6, 8);
        assert_eq!(out.ndp, 3);
        assert_eq!(out.ndq, 3);
        assert_eq!(out.p, [100, 100, 100]);
        assert_eq!(out.q, [100, 100, 100]);
    }

    /// §8.7.2.5.7 strong filter clips each output to ±2·tC of the input.
    #[test]
    fn strong_filter_clips_to_two_tc() {
        // A hard step: p side 0, q side 255. tC=2 ⇒ p0' clipped to
        // p0+2·tC = 0+4 = 4; q0' clipped to q0−2·tC = 255−4 = 251.
        let out = filter_luma_sample([0, 0, 0, 0], [255, 255, 255, 255], 2, 0, 0, 2, 8);
        assert_eq!(out.p[0], 4);
        assert_eq!(out.q[0], 251);
    }

    /// §8.7.2.5.7 weak filter: small ramp, dEp=dEq=0 filters only p0/q0.
    /// p0=98,q0=102,p1=q1=100 ⇒ δ=(9·4−3·0+8)>>4=2; |δ|<10·tC; clamp to
    /// [−tC,tC]=2 ⇒ p0'=100, q0'=100; nDp=nDq=1.
    #[test]
    fn weak_filter_centres_small_step() {
        let out = filter_luma_sample([98, 100, 100, 100], [102, 100, 100, 100], 1, 0, 0, 5, 8);
        assert_eq!(out.ndp, 1);
        assert_eq!(out.ndq, 1);
        assert_eq!(out.p[0], 100);
        assert_eq!(out.q[0], 100);
    }

    /// §8.7.2.5.7 weak filter no-op when |δ| ≥ 10·tC.
    #[test]
    fn weak_filter_skips_large_delta() {
        // Big step with tiny tC ⇒ |δ| ≥ 10·tC ⇒ nDp=nDq=0, no change.
        let out = filter_luma_sample([0, 0, 0, 0], [255, 255, 255, 255], 1, 1, 1, 1, 8);
        assert_eq!(out.ndp, 0);
        assert_eq!(out.ndq, 0);
        assert_eq!(out.p[0], 0);
        assert_eq!(out.q[0], 255);
    }

    // ----- §8.7.2.5.3 edge decision + §8.7.2.5.5 / .8 chroma -----

    /// §8.7.2.5.3: a perfectly flat edge (d=0 < β) with no step picks the
    /// strong filter (dE=2) and sets dEp=dEq=1 (dp=dq=0 < threshold).
    #[test]
    fn edge_decision_flat_is_strong() {
        let flat = [[128, 128], [128, 128], [128, 128], [128, 128]];
        let dec = luma_edge_decision(flat, flat, 32, 6);
        assert_eq!(dec.de, 2);
        assert_eq!(dec.dep, 1);
        assert_eq!(dec.deq, 1);
    }

    /// §8.7.2.5.3: a large hard step (d ≥ β) leaves dE=0 (no filtering).
    #[test]
    fn edge_decision_hard_step_no_filter() {
        // p side ramps strongly so the second differences make d ≥ β.
        let p = [[0, 0], [60, 60], [0, 0], [0, 0]];
        let q = [[255, 255], [255, 255], [255, 255], [255, 255]];
        let dec = luma_edge_decision(p, q, 8, 2);
        assert_eq!(dec.de, 0);
        assert_eq!(dec.dep, 0);
        assert_eq!(dec.deq, 0);
    }

    /// §8.7.2.5.3: a smooth region that passes d<β but fails the strong
    /// per-sample decision (a moderate boundary step) selects the weak
    /// filter (dE=1).
    #[test]
    fn edge_decision_moderate_step_is_weak() {
        // Flat halves with a step at the boundary: second differences are
        // 0 (d=0<β) but |p0−q0| is large ⇒ luma_sample_decision fails.
        let p = [[100, 100], [100, 100], [100, 100], [100, 100]];
        let q = [[140, 140], [140, 140], [140, 140], [140, 140]];
        let dec = luma_edge_decision(p, q, 64, 4);
        assert_eq!(dec.de, 1);
    }

    /// Table 8-10 (4:2:0) chroma-QP mapping breakpoints.
    #[test]
    fn table_8_10_chroma_qpc() {
        assert_eq!(chroma_qpc_420(29), 29);
        assert_eq!(chroma_qpc_420(30), 29);
        assert_eq!(chroma_qpc_420(35), 33);
        assert_eq!(chroma_qpc_420(43), 37);
        assert_eq!(chroma_qpc_420(44), 38);
        assert_eq!(chroma_qpc_420(51), 45);
    }

    /// §8.7.2.5.5 chroma tC: 4:2:0, QpY=37 both sides, no offsets.
    /// qPi=37 ⇒ QpC=34 (Table 8-10) ⇒ Q=34+2=36 ⇒ tC′=4, tC=4 at 8-bit.
    #[test]
    fn chroma_tc_420_8bit() {
        assert_eq!(chroma_tc(37, 37, 0, 1, 0, 8), 4);
        // 4:4:4 (chroma_array_type=3): QpC=Min(qPi,51)=37 ⇒ Q=39 ⇒ tC′=5.
        assert_eq!(chroma_tc(37, 37, 0, 3, 0, 8), 5);
        // 10-bit scales by 1<<(BitDepthC−8)=4.
        assert_eq!(chroma_tc(37, 37, 0, 1, 0, 10), 4 * 4);
    }

    /// §8.7.2.5.8 chroma sample filter: a flat edge is unchanged.
    #[test]
    fn chroma_filter_flat_unchanged() {
        let (p0, q0) = filter_chroma_sample([100, 100], [100, 100], 6, 8);
        assert_eq!(p0, 100);
        assert_eq!(q0, 100);
    }

    /// §8.7.2.5.8 chroma sample filter centres a small step and clips δ.
    /// p0=98,q0=102,p1=q1=100 ⇒ δ=((4·4)+0+4)>>3 = 20>>3 = 2 ⇒ clamp 2 ⇒
    /// p0'=100, q0'=100.
    #[test]
    fn chroma_filter_small_step() {
        let (p0, q0) = filter_chroma_sample([98, 100], [102, 100], 6, 8);
        assert_eq!(p0, 100);
        assert_eq!(q0, 100);
        // tC clamp: hard step 0/255, tC=3 ⇒ δ clamped to 3.
        let (p0, q0) = filter_chroma_sample([0, 0], [255, 255], 3, 8);
        assert_eq!(p0, 3);
        assert_eq!(q0, 252);
    }

    // ----- §8.7.2.5.4 / .5 plane-level block-edge filtering -----

    /// A vertical edge at plane `(x, y)`.
    fn vpos(x: usize, y: usize) -> EdgePos {
        EdgePos {
            ex: x,
            ey: y,
            edge: EdgeType::Vertical,
        }
    }

    /// 8-bit QP context with no slice offsets.
    fn qp8(qp_q: i32, qp_p: i32) -> EdgeQp {
        EdgeQp {
            qp_q,
            qp_p,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            bit_depth: 8,
        }
    }

    /// Build a `w×h` plane filled with a constant value plus a single
    /// vertical step at column `step_x` (left = `lo`, right = `hi`).
    fn vstep_plane(w: usize, h: usize, step_x: usize, lo: i32, hi: i32) -> Vec<i32> {
        let mut v = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = if x < step_x { lo } else { hi };
            }
        }
        v
    }

    /// §8.7.2.5.4 EDGE_VER: a small uniform step across a vertical edge
    /// is smoothed (the boundary columns move toward each other), and the
    /// strong filter touches 3 samples each side.
    #[test]
    fn luma_block_edge_ver_smooths_small_step() {
        let (w, h) = (16usize, 8usize);
        // Step of 8 at column 8: left=120, right=128. With low QP the
        // strong/weak decision fires (flat halves ⇒ d=0 < β).
        let mut buf = vstep_plane(w, h, 8, 120, 128);
        let before_p0 = buf[7]; // column 7, row 0 (p0)
        let before_q0 = buf[8]; // column 8, row 0 (q0)
        let mut plane = SamplePlane {
            samples: &mut buf,
            width: w,
            stride: w,
        };
        // QP 37 both sides ⇒ β=36, tC=5 (bS=2) at 8-bit; step 8 < β.
        let dec = filter_luma_block_edge(&mut plane, vpos(8, 0), 2, qp8(37, 37));
        assert_ne!(dec.de, 0, "edge should be filtered");
        // p0 rose toward the boundary, q0 fell toward it.
        assert!(plane.get(7, 0) > before_p0);
        assert!(plane.get(8, 0) < before_q0);
        // Samples far from the edge (col 0, col 15) are untouched.
        assert_eq!(plane.get(0, 0), 120);
        assert_eq!(plane.get(15, 0), 128);
    }

    /// §8.7.2.5.4: a large step at low QP leaves the plane byte-identical.
    /// The flat halves pass `d < β` so `dE` may be 1 (weak filter armed),
    /// but the weak filter's `|δ| < 10·tC` guard fails for the huge step,
    /// so `nDp = nDq = 0` and no sample is modified.
    #[test]
    fn luma_block_edge_ver_skips_large_step() {
        let (w, h) = (16usize, 8usize);
        let mut buf = vstep_plane(w, h, 8, 0, 255);
        let snapshot = buf.clone();
        let mut plane = SamplePlane {
            samples: &mut buf,
            width: w,
            stride: w,
        };
        // Low QP (β=10, tC=1): the weak filter is rejected by the δ guard.
        filter_luma_block_edge(&mut plane, vpos(8, 0), 2, qp8(20, 20));
        assert_eq!(*plane.samples, snapshot[..]);
    }

    /// §8.7.2.5.4 EDGE_HOR mirrors EDGE_VER: a horizontal step at row 4
    /// is smoothed across the boundary rows.
    #[test]
    fn luma_block_edge_hor_smooths_small_step() {
        let (w, h) = (8usize, 16usize);
        let mut buf = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = if y < 8 { 120 } else { 128 };
            }
        }
        let mut plane = SamplePlane {
            samples: &mut buf,
            width: w,
            stride: w,
        };
        // Boundary at row 8 (q0,0 row): filter the EDGE_HOR segment at x=0.
        let pos = EdgePos {
            ex: 0,
            ey: 8,
            edge: EdgeType::Horizontal,
        };
        let dec = filter_luma_block_edge(&mut plane, pos, 2, qp8(37, 37));
        assert_ne!(dec.de, 0);
        assert!(plane.get(0, 7) > 120); // p0 row rose
        assert!(plane.get(0, 8) < 128); // q0 row fell
    }

    /// §8.7.2.5.5 chroma EDGE_VER: only p0/q0 columns are modified; p1/q1
    /// and farther samples stay put.
    #[test]
    fn chroma_block_edge_ver_only_p0_q0() {
        let (w, h) = (16usize, 8usize);
        let mut buf = vstep_plane(w, h, 8, 124, 128);
        let mut plane = SamplePlane {
            samples: &mut buf,
            width: w,
            stride: w,
        };
        // 4:2:0, QpY 37 both sides ⇒ tC=4 (8-bit). Step 4 ⇒ δ clamped.
        filter_chroma_block_edge(&mut plane, vpos(8, 0), qp8(37, 37), 0, 1);
        // p0 (col 7) and q0 (col 8) move; p1 (col 6) and q1 (col 9) don't.
        assert!(plane.get(7, 0) > 124);
        assert!(plane.get(8, 0) < 128);
        assert_eq!(plane.get(6, 0), 124);
        assert_eq!(plane.get(9, 0), 128);
    }

    /// Integration: a 16×16 luma plane with two flat 120/128 halves,
    /// filter the internal vertical edge with strong-filter QP, and
    /// confirm the result is monotonic across the (now smoothed) seam and
    /// stays within the clip bounds of the original samples.
    #[test]
    fn integration_internal_vertical_seam() {
        let (w, h) = (16usize, 16usize);
        let mut buf = vstep_plane(w, h, 8, 120, 128);
        let mut plane = SamplePlane {
            samples: &mut buf,
            width: w,
            stride: w,
        };
        let dec = filter_luma_block_edge(&mut plane, vpos(8, 0), 2, qp8(40, 40));
        assert_ne!(dec.de, 0);
        for y in 0..4 {
            // Across the seam columns 5..11 the row is non-decreasing
            // (the filter cannot create overshoot beyond the 120..128
            // span for this monotone input).
            for x in 5..11 {
                let a = plane.get(x, y);
                let b = plane.get(x + 1, y);
                assert!(a <= b, "row {y}: col {x}={a} > col {}={b}", x + 1);
                assert!(
                    (120..=128).contains(&a),
                    "col {x} row {y} = {a} out of span"
                );
            }
        }
    }

    // ----- §8.7.2.5.1 / .2 CU-level edge filtering driver -----

    fn cu_params_420(qp_y: i32) -> DeblockCuParams {
        DeblockCuParams {
            qp_y,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_array_type: 1,
        }
    }

    /// A `DeblockCu` for a 16×16 CU at `(x_cb, y_cb)`, QpY = 37 both sides.
    fn cu16(x_cb: usize, y_cb: usize) -> DeblockCu {
        DeblockCu {
            x_cb,
            y_cb,
            log2_cb_size: 4,
            params: cu_params_420(37),
            qp_y_p_left: 37,
            qp_y_p_top: 37,
        }
    }

    /// Build a 16×16 4:2:0 picture whose luma has a vertical 120/128 step
    /// at column 8 and flat chroma planes.
    fn stepped_pic() -> Picture {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, if x < 8 { 120 } else { 128 });
                if x < 8 && y < 8 {
                    pic.set_sample(Plane::Cb, x, y, if x < 4 { 124 } else { 128 });
                    pic.set_sample(Plane::Cr, x, y, if x < 4 { 124 } else { 128 });
                }
            }
        }
        pic
    }

    /// §8.7.2.5.1: a 16×16 CU's internal vertical luma edge at column 8
    /// (bS=2) is filtered, smoothing the 120/128 step; the picture-left
    /// edge (bS grid zero there) is left untouched.
    #[test]
    fn cu_driver_vertical_luma_internal_edge() {
        let mut pic = stepped_pic();
        // bS grid: internal vertical edge at column 8, all 4 row-segments.
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
            tb_edge[yj * n + 8] = true;
        }
        // Intra both sides ⇒ bS=2 at the internal edge.
        let field = MotionField::new(16, 16);
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 2);
        let before7 = pic.sample(Plane::Luma, 7, 0);
        let before8 = pic.sample(Plane::Luma, 8, 0);
        filter_cu_edges(&mut pic, &cu16(0, 0), EdgeType::Vertical, &bs);
        // The internal seam smoothed: col 7 rose, col 8 fell.
        assert!(pic.sample(Plane::Luma, 7, 0) > before7);
        assert!(pic.sample(Plane::Luma, 8, 0) < before8);
        // Picture-left column 0 unfiltered (bS=0 there).
        assert_eq!(pic.sample(Plane::Luma, 0, 0), 120);
    }

    /// §8.7.2.5.1 chroma 4:2:0: an *internal* CU edge at luma column 8 maps
    /// to chroma column 4, which is NOT on the 8-chroma-sample grid, so the
    /// chroma filter is correctly skipped (only luma is touched). 4:2:0
    /// chroma deblocking falls only every 16 luma samples.
    #[test]
    fn cu_driver_chroma_skips_non_8_aligned_edge() {
        let mut pic = stepped_pic();
        let before_cb3 = pic.sample(Plane::Cb, 3, 0);
        let before_cb4 = pic.sample(Plane::Cb, 4, 0);
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
            tb_edge[yj * n + 8] = true;
        }
        let field = MotionField::new(16, 16);
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        filter_cu_edges(&mut pic, &cu16(0, 0), EdgeType::Vertical, &bs);
        // Chroma col 4 is not 8-aligned ⇒ untouched; luma seam moved.
        assert_eq!(pic.sample(Plane::Cb, 3, 0), before_cb3);
        assert_eq!(pic.sample(Plane::Cb, 4, 0), before_cb4);
        assert!(pic.sample(Plane::Luma, 7, 0) > 120);
    }

    /// §8.7.2.5.1 chroma 4:2:0: a CU boundary edge that DOES land on the
    /// 8-chroma grid (the right CU's left edge at luma column 16 ⇒ chroma
    /// column 8) fires the chroma filter; p0/q0 (chroma cols 7/8) move.
    #[test]
    fn cu_driver_chroma_filters_8_aligned_edge() {
        // 32×16 picture: left CU 120, right CU 128; chroma 124/128 step at
        // chroma column 8 (= luma column 16).
        let mut pic = Picture::new(32, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..32 {
                pic.set_sample(Plane::Luma, x, y, if x < 16 { 120 } else { 128 });
            }
        }
        // 4:2:0 chroma plane is 16×8.
        for y in 0..8 {
            for x in 0..16 {
                pic.set_sample(Plane::Cb, x, y, if x < 8 { 124 } else { 128 });
                pic.set_sample(Plane::Cr, x, y, if x < 8 { 124 } else { 128 });
            }
        }
        // Right CU at luma (16,0), 16×16. Its left CB-boundary vertical edge
        // (xDk=0) is filtered (filterLeftCbEdgeFlag = 1, intra both sides).
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n] = true; // column 0 = the CU's left boundary
            tb_edge[yj * n] = true;
        }
        // Motion field: left half (cols 0..16) and right CU intra ⇒ bS=2.
        let field = MotionField::new(32, 16);
        let bs =
            derive_boundary_strength(&field, 16, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(0, 0), 2);
        filter_cu_edges(&mut pic, &cu16(16, 0), EdgeType::Vertical, &bs);
        // Chroma edge at chroma column 8 (8-aligned): p0 (col 7) rose,
        // q0 (col 8) fell.
        assert!(pic.sample(Plane::Cb, 7, 0) > 124);
        assert!(pic.sample(Plane::Cb, 8, 0) < 128);
        assert!(pic.sample(Plane::Cr, 7, 0) > 124);
        // p1 (col 6) untouched.
        assert_eq!(pic.sample(Plane::Cb, 6, 0), 124);
    }

    /// §8.7.2.5.2: the horizontal driver mirrors the vertical one — an
    /// internal horizontal luma edge at row 8 is smoothed.
    #[test]
    fn cu_driver_horizontal_luma_internal_edge() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, if y < 8 { 120 } else { 128 });
            }
        }
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        for xi in (0..16).step_by(4) {
            edge_flags[8 * n + xi] = true;
            tb_edge[8 * n + xi] = true;
        }
        let field = MotionField::new(16, 16);
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Horizontal, &edge_flags, &tb_edge);
        assert_eq!(bs.at(0, 8), 2);
        filter_cu_edges(&mut pic, &cu16(0, 0), EdgeType::Horizontal, &bs);
        assert!(pic.sample(Plane::Luma, 0, 7) > 120); // p0 row rose
        assert!(pic.sample(Plane::Luma, 0, 8) < 128); // q0 row fell
    }

    /// A bS=1 edge filters luma but NOT chroma (chroma fires only at
    /// bS==2).
    #[test]
    fn cu_driver_bs1_skips_chroma() {
        let mut pic = stepped_pic();
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        // Two inter blocks, MV diff ⇒ bS=1.
        let mut field = MotionField::new(16, 16);
        field.fill_rect(0, 0, 8, 16, inter_cell([0, 0], 0));
        field.fill_rect(8, 0, 8, 16, inter_cell([4, 0], 0));
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
        filter_cu_edges(&mut pic, &cu16(0, 0), EdgeType::Vertical, &bs);
        // Chroma untouched (bS=1 < 2).
        assert_eq!(pic.sample(Plane::Cb, 3, 0), 124);
        assert_eq!(pic.sample(Plane::Cb, 4, 0), 128);
    }

    /// Monochrome (ChromaArrayType 0): the driver filters luma and never
    /// touches a chroma plane (there is none).
    #[test]
    fn cu_driver_monochrome_luma_only() {
        let mut pic = Picture::new(16, 16, 0, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, if x < 8 { 120 } else { 128 });
            }
        }
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let field = MotionField::new(16, 16);
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        let mut p = cu_params_420(37);
        p.chroma_array_type = 0;
        let cu = DeblockCu {
            x_cb: 0,
            y_cb: 0,
            log2_cb_size: 4,
            params: p,
            qp_y_p_left: 37,
            qp_y_p_top: 37,
        };
        filter_cu_edges(&mut pic, &cu, EdgeType::Vertical, &bs);
        assert!(pic.sample(Plane::Luma, 7, 0) > 120);
    }

    // ----- §8.7.2.1 picture-level deblocking driver -----

    /// §8.7.2.1: a 2×1 grid of 16×16 CUs with a 120/128 vertical step at
    /// the CU boundary (luma column 16). The driver derives the bS=2
    /// boundary edge (intra both sides) and smooths the seam; the picture's
    /// outer-left edge (filter_left=false on CU 0) stays untouched.
    #[test]
    fn picture_driver_two_cu_vertical_seam() {
        let mut pic = Picture::new(32, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..32 {
                pic.set_sample(Plane::Luma, x, y, if x < 16 { 120 } else { 128 });
            }
        }
        for y in 0..8 {
            for x in 0..16 {
                pic.set_sample(Plane::Cb, x, y, if x < 8 { 124 } else { 128 });
                pic.set_sample(Plane::Cr, x, y, if x < 8 { 124 } else { 128 });
            }
        }
        // Whole picture intra ⇒ every block boundary is bS=2.
        let field = MotionField::new(32, 16);
        let cus = vec![
            DeblockCuDesc {
                cu: cu16(0, 0),
                transform_split: TransformSplit::leaf(),
                part_mode: PartMode::Part2Nx2N,
                filter_left: false, // picture-left boundary
                filter_top: false,  // picture-top boundary
            },
            DeblockCuDesc {
                cu: cu16(16, 0),
                transform_split: TransformSplit::leaf(),
                part_mode: PartMode::Part2Nx2N,
                filter_left: true, // internal CU boundary at column 16
                filter_top: false,
            },
        ];
        deblock_picture(&mut pic, &field, &cus);
        // The CU-boundary luma seam at column 16 smoothed.
        assert!(pic.sample(Plane::Luma, 15, 0) > 120);
        assert!(pic.sample(Plane::Luma, 16, 0) < 128);
        // The picture-left column 0 untouched.
        assert_eq!(pic.sample(Plane::Luma, 0, 0), 120);
        // Chroma seam at chroma column 8 (8-aligned, bS=2) smoothed.
        assert!(pic.sample(Plane::Cb, 7, 0) > 124);
        assert!(pic.sample(Plane::Cb, 8, 0) < 128);
    }

    /// §8.7.2.1: with every `filter_left` / `filter_top` false and a single
    /// un-split PART_2Nx2N CU, no internal edge exists, so the picture is
    /// byte-identical after deblocking (the entire CU is one transform /
    /// prediction block at the picture origin).
    #[test]
    fn picture_driver_single_cu_no_internal_edge_is_noop() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, if x < 8 { 0 } else { 255 });
            }
        }
        let snapshot: Vec<i32> = pic.plane(Plane::Luma).to_vec();
        let field = MotionField::new(16, 16);
        let cus = vec![DeblockCuDesc {
            cu: cu16(0, 0),
            transform_split: TransformSplit::leaf(),
            part_mode: PartMode::Part2Nx2N,
            filter_left: false,
            filter_top: false,
        }];
        deblock_picture(&mut pic, &field, &cus);
        assert_eq!(pic.plane(Plane::Luma), snapshot.as_slice());
    }

    /// §8.7.2.1: a single CU with a one-level transform split has an
    /// internal vertical + horizontal TB edge at the 8-sample mid-lines;
    /// the driver smooths both the vertical seam (pass 1) and then the
    /// horizontal seam (pass 2) on the already-modified samples.
    #[test]
    fn picture_driver_internal_split_both_directions() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        // Four quadrants stepping in both x and y for a cross seam.
        for y in 0..16 {
            for x in 0..16 {
                let v = 120 + 4 * i32::from(x >= 8) + 4 * i32::from(y >= 8);
                pic.set_sample(Plane::Luma, x, y, v);
            }
        }
        let field = MotionField::new(16, 16);
        let cus = vec![DeblockCuDesc {
            cu: cu16(0, 0),
            transform_split: TransformSplit::split_once(),
            part_mode: PartMode::Part2Nx2N,
            filter_left: false,
            filter_top: false,
        }];
        deblock_picture(&mut pic, &field, &cus);
        // Vertical seam at column 8 (rows 0..) smoothed: col 7 rose.
        assert!(pic.sample(Plane::Luma, 7, 0) > 120);
        // Horizontal seam at row 8 smoothed: row 7 rose at a left column.
        assert!(pic.sample(Plane::Luma, 0, 7) > 120);
        // A corner far from both seams is untouched.
        assert_eq!(pic.sample(Plane::Luma, 0, 0), 120);
    }
}
