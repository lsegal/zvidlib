//! The VP8 loop filter (RFC 6386 section 15), applied to a whole frame after
//! every macroblock has been reconstructed.

use super::predict::Plane;
use super::simd::{self, EdgeLimits};

/// Per-macroblock filter parameters, already adjusted for segment and mode.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MacroblockFilter {
    pub level: u8,
    /// Whether the edges between the macroblock's own subblocks are filtered.
    pub inner_edges: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameFilter {
    pub simple: bool,
    pub sharpness: u8,
    pub key_frame: bool,
}

#[inline]
fn clamp_i8(value: i32) -> i32 {
    value.clamp(-128, 127)
}

#[inline]
fn u2s(value: u8) -> i32 {
    i32::from(value) - 128
}

#[inline]
fn s2u(value: i32) -> u8 {
    (clamp_i8(value) + 128) as u8
}

/// The samples on either side of an edge: `p[0]` nearest the edge on the
/// near side, `q[0]` nearest it on the far side.
struct Segment<'a> {
    data: &'a mut [u8],
    /// Index of `q0`.
    at: usize,
    /// Distance between successive samples across the edge.
    step: usize,
}

impl Segment<'_> {
    #[inline]
    fn p(&self, i: usize) -> u8 {
        self.data[self.at - (i + 1) * self.step]
    }

    #[inline]
    fn q(&self, i: usize) -> u8 {
        self.data[self.at + i * self.step]
    }

    #[inline]
    fn set_p(&mut self, i: usize, value: u8) {
        self.data[self.at - (i + 1) * self.step] = value;
    }

    #[inline]
    fn set_q(&mut self, i: usize, value: u8) {
        self.data[self.at + i * self.step] = value;
    }

    fn simple_threshold(&self, limit: i32) -> bool {
        (i32::from(self.p(0)) - i32::from(self.q(0))).abs() * 2
            + ((i32::from(self.p(1)) - i32::from(self.q(1))).abs() >> 1)
            <= limit
    }

    fn normal_threshold(&self, edge_limit: i32, interior_limit: i32) -> bool {
        let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).abs();
        self.simple_threshold(edge_limit)
            && d(self.p(3), self.p(2)) <= interior_limit
            && d(self.p(2), self.p(1)) <= interior_limit
            && d(self.p(1), self.p(0)) <= interior_limit
            && d(self.q(3), self.q(2)) <= interior_limit
            && d(self.q(2), self.q(1)) <= interior_limit
            && d(self.q(1), self.q(0)) <= interior_limit
    }

    fn high_edge_variance(&self, threshold: i32) -> bool {
        (i32::from(self.p(1)) - i32::from(self.p(0))).abs() > threshold
            || (i32::from(self.q(1)) - i32::from(self.q(0))).abs() > threshold
    }

    /// The common adjustment of `p0`/`q0`; also adjusts `p1`/`q1` when the
    /// outer taps are not used to compute it.
    fn common(&mut self, use_outer_taps: bool) {
        let p1 = u2s(self.p(1));
        let p0 = u2s(self.p(0));
        let q0 = u2s(self.q(0));
        let q1 = u2s(self.q(1));
        let mut a = 3 * (q0 - p0);
        if use_outer_taps {
            a += clamp_i8(p1 - q1);
        }
        let a = clamp_i8(a);
        let f1 = clamp_i8(a + 4) >> 3;
        let f2 = clamp_i8(a + 3) >> 3;
        self.set_q(0, s2u(q0 - f1));
        self.set_p(0, s2u(p0 + f2));
        if !use_outer_taps {
            let a = (f1 + 1) >> 1;
            self.set_q(1, s2u(q1 - a));
            self.set_p(1, s2u(p1 + a));
        }
    }

    fn macroblock(&mut self) {
        let p2 = u2s(self.p(2));
        let p1 = u2s(self.p(1));
        let p0 = u2s(self.p(0));
        let q0 = u2s(self.q(0));
        let q1 = u2s(self.q(1));
        let q2 = u2s(self.q(2));
        let w = clamp_i8(clamp_i8(p1 - q1) + 3 * (q0 - p0));
        let a = clamp_i8((27 * w + 63) >> 7);
        self.set_q(0, s2u(q0 - a));
        self.set_p(0, s2u(p0 + a));
        let a = clamp_i8((18 * w + 63) >> 7);
        self.set_q(1, s2u(q1 - a));
        self.set_p(1, s2u(p1 + a));
        let a = clamp_i8((9 * w + 63) >> 7);
        self.set_q(2, s2u(q2 - a));
        self.set_p(2, s2u(p2 + a));
    }
}

#[derive(Clone, Copy)]
struct Limits {
    macroblock_edge: i32,
    subblock_edge: i32,
    interior: i32,
    hev_threshold: i32,
}

fn limits(level: u8, frame: FrameFilter) -> Limits {
    let level = i32::from(level);
    let sharpness = i32::from(frame.sharpness);
    let mut interior = level;
    if sharpness > 0 {
        interior >>= if sharpness > 4 { 2 } else { 1 };
        interior = interior.min(9 - sharpness);
    }
    let interior = interior.max(1);
    let mut hev_threshold = 0;
    if level >= 15 {
        hev_threshold += 1;
    }
    if level >= 40 {
        hev_threshold += 1;
    }
    if level >= 20 && !frame.key_frame {
        hev_threshold += 1;
    }
    Limits {
        macroblock_edge: (level + 2) * 2 + interior,
        subblock_edge: level * 2 + interior,
        interior,
        hev_threshold,
    }
}

/// Filters `count` segments across one edge. `at` is the first `q0` index,
/// `step` crosses the edge and `advance` moves along it.
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    data: &mut [u8],
    at: usize,
    step: usize,
    advance: usize,
    count: usize,
    macroblock_edge: bool,
    limits: Limits,
    simple: bool,
) {
    let edge_limit = if macroblock_edge {
        limits.macroblock_edge
    } else {
        limits.subblock_edge
    };
    // The segments along an edge are independent of each other, so the
    // vector kernels filter a vector of them at a time.
    let vector = EdgeLimits {
        edge: edge_limit,
        interior: limits.interior,
        hev_threshold: limits.hev_threshold,
        macroblock_edge,
        simple,
    };
    let vectorized = if step == 1 {
        simd::filter_vertical_edge(data, at, advance, count, vector)
    } else {
        simd::filter_horizontal_edge(data, at, step, count, vector)
    };
    if vectorized {
        return;
    }
    for index in 0..count {
        let mut segment = Segment {
            data: &mut *data,
            at: at + index * advance,
            step,
        };
        if simple {
            if segment.simple_threshold(edge_limit) {
                segment.common(true);
            }
        } else if segment.normal_threshold(edge_limit, limits.interior) {
            let hev = segment.high_edge_variance(limits.hev_threshold);
            if macroblock_edge && !hev {
                segment.macroblock();
            } else {
                segment.common(hev);
            }
        }
    }
}

/// Filters one plane's macroblock (of `size` samples) at `(mb_x, mb_y)`.
fn filter_macroblock(
    plane: &mut Plane,
    mb_x: usize,
    mb_y: usize,
    size: usize,
    limits: Limits,
    inner_edges: bool,
    simple: bool,
) {
    let stride = plane.width;
    let origin = mb_y * size * stride + mb_x * size;
    let data = &mut plane.data;
    if mb_x > 0 {
        filter_edge(data, origin, 1, stride, size, true, limits, simple);
    }
    if inner_edges {
        for x in (4..size).step_by(4) {
            filter_edge(data, origin + x, 1, stride, size, false, limits, simple);
        }
    }
    if mb_y > 0 {
        filter_edge(data, origin, stride, 1, size, true, limits, simple);
    }
    if inner_edges {
        for y in (4..size).step_by(4) {
            filter_edge(
                data,
                origin + y * stride,
                stride,
                1,
                size,
                false,
                limits,
                simple,
            );
        }
    }
}

/// Filters a whole frame in macroblock raster order. The simple filter only
/// touches luma.
pub(crate) fn filter_frame(
    planes: &mut [Plane; 3],
    macroblocks: &[MacroblockFilter],
    mb_cols: usize,
    frame: FrameFilter,
) {
    for (index, macroblock) in macroblocks.iter().enumerate() {
        if macroblock.level == 0 {
            continue;
        }
        let mb_x = index % mb_cols;
        let mb_y = index / mb_cols;
        let limits = limits(macroblock.level, frame);
        let [y, u, v] = planes;
        filter_macroblock(
            y,
            mb_x,
            mb_y,
            16,
            limits,
            macroblock.inner_edges,
            frame.simple,
        );
        if !frame.simple {
            filter_macroblock(u, mb_x, mb_y, 8, limits, macroblock.inner_edges, false);
            filter_macroblock(v, mb_x, mb_y, 8, limits, macroblock.inner_edges, false);
        }
    }
}
