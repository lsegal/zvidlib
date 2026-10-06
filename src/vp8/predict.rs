//! VP8 reconstruction kernels: intra prediction, motion-compensated
//! prediction, the inverse DCT and Walsh-Hadamard transforms (RFC 6386
//! sections 12, 14 and 18).
//!
//! Intermediate values are truncated to 16 bits wherever libvpx stores them
//! in `short`, so even out-of-range coefficients reconstruct exactly as the
//! reference decoder reconstructs them.

/// A plane of 8-bit samples whose dimensions are whole macroblocks.
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Plane {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height],
            width,
            height,
        }
    }

    /// The sample at `(x, y)` with coordinates outside the plane clamped to
    /// its edge, which is how VP8 extends a reference frame.
    #[inline]
    fn clamped(&self, x: isize, y: isize) -> u8 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }
}

#[inline]
fn clamp255(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

// --- Transforms ------------------------------------------------------------

const COS_PI8_SQRT2_MINUS1: i32 = 20091;
const SIN_PI8_SQRT2: i32 = 35468;

/// Inverse 4x4 DCT of dequantized `coefficients` (raster order), added to the
/// 4x4 block at `offset` of `plane`.
pub(crate) fn idct_add(coefficients: &[i16; 16], plane: &mut [u8], offset: usize, stride: usize) {
    if !super::simd::idct_add(coefficients, plane, offset, stride) {
        idct_add_scalar(coefficients, plane, offset, stride);
    }
}

fn idct_add_scalar(
    coefficients: &[i16; 16],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    let mut temp = [0i16; 16];
    for column in 0..4 {
        let i0 = i32::from(coefficients[column]);
        let i4 = i32::from(coefficients[4 + column]);
        let i8 = i32::from(coefficients[8 + column]);
        let i12 = i32::from(coefficients[12 + column]);
        let a1 = i0 + i8;
        let b1 = i0 - i8;
        let c1 = ((i4 * SIN_PI8_SQRT2) >> 16) - (i12 + ((i12 * COS_PI8_SQRT2_MINUS1) >> 16));
        let d1 = (i4 + ((i4 * COS_PI8_SQRT2_MINUS1) >> 16)) + ((i12 * SIN_PI8_SQRT2) >> 16);
        temp[column] = (a1 + d1) as i16;
        temp[12 + column] = (a1 - d1) as i16;
        temp[4 + column] = (b1 + c1) as i16;
        temp[8 + column] = (b1 - c1) as i16;
    }
    for row in 0..4 {
        let i0 = i32::from(temp[row * 4]);
        let i1 = i32::from(temp[row * 4 + 1]);
        let i2 = i32::from(temp[row * 4 + 2]);
        let i3 = i32::from(temp[row * 4 + 3]);
        let a1 = i0 + i2;
        let b1 = i0 - i2;
        let c1 = ((i1 * SIN_PI8_SQRT2) >> 16) - (i3 + ((i3 * COS_PI8_SQRT2_MINUS1) >> 16));
        let d1 = (i1 + ((i1 * COS_PI8_SQRT2_MINUS1) >> 16)) + ((i3 * SIN_PI8_SQRT2) >> 16);
        let residual = [
            ((a1 + d1 + 4) >> 3) as i16,
            ((b1 + c1 + 4) >> 3) as i16,
            ((b1 - c1 + 4) >> 3) as i16,
            ((a1 - d1 + 4) >> 3) as i16,
        ];
        let line = &mut plane[offset + row * stride..offset + row * stride + 4];
        for (sample, residual) in line.iter_mut().zip(residual) {
            *sample = clamp255(i32::from(*sample) + i32::from(residual));
        }
    }
}

/// Inverse Walsh-Hadamard transform of the dequantized Y2 block, returning
/// the DC coefficient of each of the 16 luma blocks in raster order.
pub(crate) fn inverse_walsh(input: &[i16; 16]) -> [i16; 16] {
    super::simd::inverse_walsh(input).unwrap_or_else(|| inverse_walsh_scalar(input))
}

fn inverse_walsh_scalar(input: &[i16; 16]) -> [i16; 16] {
    let mut temp = [0i32; 16];
    for column in 0..4 {
        let i0 = i32::from(input[column]);
        let i4 = i32::from(input[4 + column]);
        let i8 = i32::from(input[8 + column]);
        let i12 = i32::from(input[12 + column]);
        let a1 = i0 + i12;
        let b1 = i4 + i8;
        let c1 = i4 - i8;
        let d1 = i0 - i12;
        // libvpx keeps this pass in `short`.
        temp[column] = i32::from((a1 + b1) as i16);
        temp[4 + column] = i32::from((c1 + d1) as i16);
        temp[8 + column] = i32::from((a1 - b1) as i16);
        temp[12 + column] = i32::from((d1 - c1) as i16);
    }
    let mut output = [0i16; 16];
    for row in 0..4 {
        let i0 = temp[row * 4];
        let i1 = temp[row * 4 + 1];
        let i2 = temp[row * 4 + 2];
        let i3 = temp[row * 4 + 3];
        let a1 = i0 + i3;
        let b1 = i1 + i2;
        let c1 = i1 - i2;
        let d1 = i0 - i3;
        output[row * 4] = ((a1 + b1 + 3) >> 3) as i16;
        output[row * 4 + 1] = ((c1 + d1 + 3) >> 3) as i16;
        output[row * 4 + 2] = ((a1 - b1 + 3) >> 3) as i16;
        output[row * 4 + 3] = ((d1 - c1 + 3) >> 3) as i16;
    }
    output
}

// --- Intra prediction --------------------------------------------------------

/// The reconstructed (unfiltered) pixels around a block: the row above with
/// the pixel above-left at index 0, and the column to the left.
pub(crate) struct Edges<const N: usize, const A: usize> {
    pub above: [u8; A],
    pub left: [u8; N],
}

impl<const N: usize, const A: usize> Edges<N, A> {
    #[inline]
    fn corner(&self) -> i32 {
        i32::from(self.above[0])
    }

    #[inline]
    fn top(&self, index: usize) -> i32 {
        i32::from(self.above[index + 1])
    }
}

/// Gathers a macroblock's prediction edges from the frame being decoded,
/// substituting 127 above the frame and 129 left of it (RFC 6386 section 12).
/// `A` is `N + 1`, or `N + 5` to include the four above-right pixels the
/// luma subblock modes use.
pub(crate) fn macroblock_edges<const N: usize, const A: usize>(
    plane: &Plane,
    mb_x: usize,
    mb_y: usize,
) -> Edges<N, A> {
    let x0 = mb_x * N;
    let y0 = mb_y * N;
    let mut above = [127u8; A];
    let mut left = [129u8; N];
    if mb_y > 0 {
        let row = (y0 - 1) * plane.width;
        above[0] = if mb_x > 0 {
            plane.data[row + x0 - 1]
        } else {
            129
        };
        above[1..=N].copy_from_slice(&plane.data[row + x0..row + x0 + N]);
        if A > N + 1 {
            // The above-right pixels come from the next macroblock of the
            // row above; the last macroblock repeats the row's last pixel.
            for (index, sample) in above.iter_mut().enumerate().skip(N + 1) {
                let x = x0 + index - 1;
                *sample = if x < plane.width {
                    plane.data[row + x]
                } else {
                    plane.data[row + plane.width - 1]
                };
            }
        }
    }
    if mb_x > 0 {
        for (row, sample) in left.iter_mut().enumerate() {
            *sample = plane.data[(y0 + row) * plane.width + x0 - 1];
        }
    }
    Edges { above, left }
}

/// Predicts a whole `N`x`N` luma or chroma block with one of the four
/// macroblock modes. DC prediction averages only the edges that lie inside
/// the frame, and is 128 when neither does.
pub(crate) fn predict_block<const N: usize, const A: usize>(
    mode: u8,
    edges: &Edges<N, A>,
    has_above: bool,
    has_left: bool,
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    use super::tables::{DC_PRED, H_PRED, TM_PRED, V_PRED};
    let shift = N.trailing_zeros();
    match mode {
        DC_PRED => {
            let above: u32 = edges.above[1..=N].iter().map(|&v| u32::from(v)).sum();
            let left: u32 = edges.left.iter().map(|&v| u32::from(v)).sum();
            let value = match (has_above, has_left) {
                (true, true) => (above + left + (1 << shift)) >> (shift + 1),
                (true, false) => (above + (1 << (shift - 1))) >> shift,
                (false, true) => (left + (1 << (shift - 1))) >> shift,
                (false, false) => 128,
            } as u8;
            for row in 0..N {
                plane[offset + row * stride..offset + row * stride + N].fill(value);
            }
        }
        V_PRED => {
            for row in 0..N {
                plane[offset + row * stride..offset + row * stride + N]
                    .copy_from_slice(&edges.above[1..=N]);
            }
        }
        H_PRED => {
            for row in 0..N {
                plane[offset + row * stride..offset + row * stride + N].fill(edges.left[row]);
            }
        }
        TM_PRED => {
            let corner = edges.corner();
            if super::simd::tm_predict(
                &edges.above[1..=N],
                &edges.left,
                corner,
                N,
                plane,
                offset,
                stride,
            ) {
                return;
            }
            for row in 0..N {
                let left = i32::from(edges.left[row]) - corner;
                for column in 0..N {
                    plane[offset + row * stride + column] = clamp255(left + edges.top(column));
                }
            }
        }
        _ => unreachable!("macroblock intra modes are DC, V, H and TM"),
    }
}

/// Predicts one 4x4 luma subblock. `above` holds the pixel above-left, the
/// four above and the four above-right; `left` the four to the left.
pub(crate) fn predict_subblock(
    mode: u8,
    above: &[u8; 9],
    left: &[u8; 4],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    use super::tables::*;
    let p = i32::from(above[0]);
    let a = |i: usize| i32::from(above[i + 1]);
    let l = |i: usize| i32::from(left[i]);
    let avg2 = |x: i32, y: i32| ((x + y + 1) >> 1) as u8;
    let avg3 = |x: i32, y: i32, z: i32| ((x + 2 * y + z + 2) >> 2) as u8;
    let mut block = [[0u8; 4]; 4];
    match mode {
        B_DC_PRED => {
            let sum: i32 = (0..4).map(|i| a(i) + l(i)).sum();
            block = [[((sum + 4) >> 3) as u8; 4]; 4];
        }
        B_TM_PRED => {
            for (r, line) in block.iter_mut().enumerate() {
                for (c, sample) in line.iter_mut().enumerate() {
                    *sample = clamp255(l(r) + a(c) - p);
                }
            }
        }
        B_VE_PRED => {
            let line = [
                avg3(p, a(0), a(1)),
                avg3(a(0), a(1), a(2)),
                avg3(a(1), a(2), a(3)),
                avg3(a(2), a(3), a(4)),
            ];
            block = [line; 4];
        }
        B_HE_PRED => {
            block[0] = [avg3(p, l(0), l(1)); 4];
            block[1] = [avg3(l(0), l(1), l(2)); 4];
            block[2] = [avg3(l(1), l(2), l(3)); 4];
            block[3] = [avg3(l(2), l(3), l(3)); 4];
        }
        B_LD_PRED => {
            for (r, line) in block.iter_mut().enumerate() {
                for (c, sample) in line.iter_mut().enumerate() {
                    let i = r + c;
                    *sample = if i < 6 {
                        avg3(a(i), a(i + 1), a(i + 2))
                    } else {
                        avg3(a(6), a(7), a(7))
                    };
                }
            }
        }
        B_RD_PRED => {
            // Edge pixels from bottom-left, around the corner, to top-right.
            let edge = [l(3), l(2), l(1), l(0), p, a(0), a(1), a(2), a(3)];
            for (r, line) in block.iter_mut().enumerate() {
                for (c, sample) in line.iter_mut().enumerate() {
                    let i = 3 - r + c;
                    *sample = avg3(edge[i], edge[i + 1], edge[i + 2]);
                }
            }
        }
        B_VR_PRED => {
            block[3][0] = avg3(l(2), l(1), l(0));
            block[2][0] = avg3(l(1), l(0), p);
            block[3][1] = avg3(l(0), p, a(0));
            block[1][0] = block[3][1];
            block[2][1] = avg2(p, a(0));
            block[0][0] = block[2][1];
            block[3][2] = avg3(p, a(0), a(1));
            block[1][1] = block[3][2];
            block[2][2] = avg2(a(0), a(1));
            block[0][1] = block[2][2];
            block[3][3] = avg3(a(0), a(1), a(2));
            block[1][2] = block[3][3];
            block[2][3] = avg2(a(1), a(2));
            block[0][2] = block[2][3];
            block[1][3] = avg3(a(1), a(2), a(3));
            block[0][3] = avg2(a(2), a(3));
        }
        B_VL_PRED => {
            block[0][0] = avg2(a(0), a(1));
            block[1][0] = avg3(a(0), a(1), a(2));
            block[2][0] = avg2(a(1), a(2));
            block[0][1] = block[2][0];
            block[1][1] = avg3(a(1), a(2), a(3));
            block[3][0] = block[1][1];
            block[2][1] = avg2(a(2), a(3));
            block[0][2] = block[2][1];
            block[3][1] = avg3(a(2), a(3), a(4));
            block[1][2] = block[3][1];
            block[2][2] = avg2(a(3), a(4));
            block[0][3] = block[2][2];
            block[3][2] = avg3(a(3), a(4), a(5));
            block[1][3] = block[3][2];
            // These two do not follow the pattern.
            block[2][3] = avg3(a(4), a(5), a(6));
            block[3][3] = avg3(a(5), a(6), a(7));
        }
        B_HD_PRED => {
            block[3][0] = avg2(l(3), l(2));
            block[3][1] = avg3(l(3), l(2), l(1));
            block[2][0] = avg2(l(2), l(1));
            block[3][2] = block[2][0];
            block[2][1] = avg3(l(2), l(1), l(0));
            block[3][3] = block[2][1];
            block[2][2] = avg2(l(1), l(0));
            block[1][0] = block[2][2];
            block[2][3] = avg3(l(1), l(0), p);
            block[1][1] = block[2][3];
            block[1][2] = avg2(l(0), p);
            block[0][0] = block[1][2];
            block[1][3] = avg3(l(0), p, a(0));
            block[0][1] = block[1][3];
            block[0][2] = avg3(p, a(0), a(1));
            block[0][3] = avg3(a(0), a(1), a(2));
        }
        B_HU_PRED => {
            block[0][0] = avg2(l(0), l(1));
            block[0][1] = avg3(l(0), l(1), l(2));
            block[0][2] = avg2(l(1), l(2));
            block[1][0] = block[0][2];
            block[0][3] = avg3(l(1), l(2), l(3));
            block[1][1] = block[0][3];
            block[1][2] = avg2(l(2), l(3));
            block[2][0] = block[1][2];
            block[1][3] = avg3(l(2), l(3), l(3));
            block[2][1] = block[1][3];
            let last = left[3];
            block[2][2] = last;
            block[2][3] = last;
            block[3] = [last; 4];
        }
        _ => unreachable!("subblock intra modes are 0..=9"),
    }
    for (r, line) in block.iter().enumerate() {
        plane[offset + r * stride..offset + r * stride + 4].copy_from_slice(line);
    }
}

// --- Inter prediction --------------------------------------------------------

/// Predicts a `width`x`height` block at `(x, y)` of `output` from `reference`
/// displaced by `mv`, in eighth-sample units, with the six-tap or bilinear
/// `filters`. The reference is extended beyond its edges by repeating them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_inter(
    reference: &Plane,
    output: &mut Plane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv_x: i32,
    mv_y: i32,
    filters: &[[i32; 6]; 8],
) {
    let fraction_x = (mv_x & 7) as usize;
    let fraction_y = (mv_y & 7) as usize;
    let source_x = x as isize + (mv_x >> 3) as isize;
    let source_y = y as isize + (mv_y >> 3) as isize;

    // The window the filters read: two samples before the block and three
    // after it in each direction.
    let mut window = [0u8; 21 * 21];
    let window_width = width + 5;
    let window_height = height + 5;
    let left = source_x - 2;
    let top = source_y - 2;
    let inside = left >= 0
        && top >= 0
        && left as usize + window_width <= reference.width
        && top as usize + window_height <= reference.height;
    for row in 0..window_height {
        let line = &mut window[row * window_width..(row + 1) * window_width];
        if inside {
            let start = (top as usize + row) * reference.width + left as usize;
            line.copy_from_slice(&reference.data[start..start + window_width]);
        } else {
            for (column, sample) in line.iter_mut().enumerate() {
                *sample = reference.clamped(left + column as isize, top + row as isize);
            }
        }
    }

    let stride = output.width;
    let destination = y * stride + x;
    if fraction_x == 0 && fraction_y == 0 {
        for row in 0..height {
            let start = (row + 2) * window_width + 2;
            output.data[destination + row * stride..destination + row * stride + width]
                .copy_from_slice(&window[start..start + width]);
        }
        return;
    }

    // Both passes always run, as in libvpx; the zero-fraction filter is the
    // identity, so a whole-sample displacement in one direction is exact.
    let horizontal = &filters[fraction_x];
    let vertical = &filters[fraction_y];
    if super::simd::sixtap(
        &window,
        width,
        height,
        horizontal,
        vertical,
        &mut output.data,
        destination,
        stride,
    ) {
        return;
    }
    filter_window(
        &window,
        width,
        height,
        horizontal,
        vertical,
        &mut output.data,
        destination,
        stride,
    );
}

/// The scalar two-pass filter of `predict_inter`: `horizontal` over the
/// `(width + 5)`-wide `window`, then `vertical` into the block at
/// `destination` of `output`.
#[allow(clippy::too_many_arguments)]
fn filter_window(
    window: &[u8],
    width: usize,
    height: usize,
    horizontal: &[i32; 6],
    vertical: &[i32; 6],
    output: &mut [u8],
    destination: usize,
    stride: usize,
) {
    let window_width = width + 5;
    let window_height = height + 5;
    let mut first = [0u8; 16 * 21];
    for row in 0..window_height {
        let line = &window[row * window_width..];
        for column in 0..width {
            let taps = &line[column..column + 6];
            let sum: i32 = taps
                .iter()
                .zip(horizontal)
                .map(|(&sample, &tap)| i32::from(sample) * tap)
                .sum();
            first[row * width + column] = clamp255((sum + 64) >> 7);
        }
    }
    for row in 0..height {
        for column in 0..width {
            let mut sum = 64;
            for (tap_index, &tap) in vertical.iter().enumerate() {
                sum += i32::from(first[(row + tap_index) * width + column]) * tap;
            }
            output[destination + row * stride + column] = clamp255(sum >> 7);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_inverse_dct_adds_the_rounded_dc_everywhere() {
        let mut coefficients = [0i16; 16];
        coefficients[0] = 100;
        let mut plane = vec![10u8; 16];
        idct_add(&coefficients, &mut plane, 0, 4);
        assert!(plane.iter().all(|&v| v == 10 + ((100 + 4) >> 3)));
    }

    #[test]
    fn dc_only_walsh_spreads_the_rounded_dc() {
        let mut input = [0i16; 16];
        input[0] = 77;
        assert_eq!(inverse_walsh(&input), [((77 + 3) >> 3) as i16; 16]);
    }

    #[test]
    fn inter_prediction_repeats_reference_edges() {
        let mut reference = Plane::new(16, 16);
        for (index, sample) in reference.data.iter_mut().enumerate() {
            *sample = (index % 16) as u8 * 10;
        }
        let mut output = Plane::new(16, 16);
        // Forty samples left of the frame: every sample is the left edge.
        predict_inter(
            &reference,
            &mut output,
            0,
            0,
            16,
            16,
            -40 * 8 + 3,
            0,
            &super::super::tables::SIXTAP_FILTERS,
        );
        assert!(output.data.iter().all(|&v| v == 0));
    }
}
