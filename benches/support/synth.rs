//! Deterministic synthetic YUV420 input.
//!
//! Encoder benchmarks need frames, and decoding something first to get them
//! would fold decode time (and decode-side SIMD dispatch) into an encoder
//! measurement. These generators produce structured, reproducible content
//! instead: smooth gradients so the spatial predictors have something to
//! predict, a per-frame translation so motion estimation has something to
//! find, and a bit of deterministic high-frequency detail so the transforms
//! and in-loop filters are not fed a flat plane.
//!
//! Nothing here uses a random number generator: the same arguments always
//! produce the same bytes, on every host and every run.

/// One 8-bit YUV420 frame.
#[derive(Clone, Debug)]
pub struct Yuv420Frame {
    pub width: usize,
    pub height: usize,
    /// `width * height` luma samples.
    pub y: Vec<u8>,
    /// `chroma_width * chroma_height` Cb samples.
    pub u: Vec<u8>,
    /// `chroma_width * chroma_height` Cr samples.
    pub v: Vec<u8>,
}

impl Yuv420Frame {
    /// Chroma plane width, i.e. luma width halved and rounded up.
    #[must_use]
    pub fn chroma_width(&self) -> usize {
        self.width.div_ceil(2)
    }

    /// Chroma plane height, i.e. luma height halved and rounded up.
    #[must_use]
    pub fn chroma_height(&self) -> usize {
        self.height.div_ceil(2)
    }

    /// Luma samples in this frame.
    #[must_use]
    pub fn pixels(&self) -> u64 {
        (self.width * self.height) as u64
    }
}

/// Generates frame `index` of a synthetic sequence.
///
/// The image is a diagonal gradient translated by `index` samples per frame,
/// overlaid with a coarse checkerboard and a cheap hash-derived dither. The
/// translation is what makes consecutive frames genuinely inter-predictable
/// rather than identical.
#[must_use]
pub fn yuv420_frame(width: usize, height: usize, index: usize) -> Yuv420Frame {
    assert!(width > 0 && height > 0, "frame dimensions must be nonzero");
    let mut y = vec![0u8; width * height];
    for row in 0..height {
        for col in 0..width {
            y[row * width + col] = luma_sample(col + index, row + index / 2);
        }
    }
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = 128u8.wrapping_add(((col + index) % 48) as u8);
            v[row * cw + col] = 128u8.wrapping_add(((row + index) % 32) as u8);
        }
    }
    Yuv420Frame {
        width,
        height,
        y,
        u,
        v,
    }
}

/// Generates `count` consecutive frames of the same synthetic sequence.
#[must_use]
pub fn yuv420_sequence(width: usize, height: usize, count: usize) -> Vec<Yuv420Frame> {
    (0..count)
        .map(|index| yuv420_frame(width, height, index))
        .collect()
}

/// A single synthetic luma plane, for kernels that take one plane rather than
/// a whole frame (the in-loop filters and motion compensation, in particular).
#[must_use]
pub fn luma_plane(width: usize, height: usize) -> Vec<u8> {
    yuv420_frame(width, height, 0).y
}

/// The gradient-plus-detail luma function the generators share.
fn luma_sample(x: usize, y: usize) -> u8 {
    let gradient = ((x * 3 + y * 5) % 224) as u32;
    let checker = if ((x / 16) + (y / 16)) % 2 == 0 {
        16
    } else {
        0
    };
    // A multiplicative hash, not a PRNG: same coordinates, same detail, always.
    let detail = ((x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503)) >> 13) as u32 % 15;
    (gradient + checker + detail).min(255) as u8
}
