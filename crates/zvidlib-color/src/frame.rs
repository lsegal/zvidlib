//! The 8-bit YUV frame the AV1, VP8 and VP9 decoders reconstruct into, and its
//! conversion to packed `Rgba8` output.

use crate::yuv_to_rgba::{self, Conversion};
use crate::{ColorRange, Error, ErrorKind, Limits, Result};

// ---------------------------------------------------------------------
// Plane storage
// ---------------------------------------------------------------------

/// One bounded, owned 8-bit reconstructed plane used through the filter
/// pipeline. Distinct from [`zvidlib_core::Plane`] (which describes packed output
/// buffers) because filters need read/write access with explicit bounds
/// checks at every edge, including malformed/odd-sized inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterPlane {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub data: Vec<u8>,
}

impl FilterPlane {
    /// Creates a zero-filled plane, bounded by `limits`.
    pub fn new(width: usize, height: usize, limits: &Limits) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(invalid("filter plane dimensions must be nonzero"));
        }
        if width > limits.max_width as usize || height > limits.max_height as usize {
            return Err(resource("filter plane dimensions exceed configured limits"));
        }
        let stride = width;
        let bytes = stride
            .checked_mul(height)
            .ok_or_else(|| resource("filter plane size overflow"))?;
        if bytes as u64 > limits.max_allocation_bytes {
            return Err(resource("filter plane exceeds the allocation limit"));
        }
        Ok(Self {
            width,
            height,
            stride,
            data: vec![0u8; bytes],
        })
    }

    /// Builds a plane from existing row-major, tightly packed samples.
    pub fn from_samples(
        width: usize,
        height: usize,
        data: Vec<u8>,
        limits: &Limits,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(invalid("filter plane dimensions must be nonzero"));
        }
        if width > limits.max_width as usize || height > limits.max_height as usize {
            return Err(resource("filter plane dimensions exceed configured limits"));
        }
        if data.len() != width * height {
            return Err(invalid(
                "filter plane data length does not match dimensions",
            ));
        }
        if data.len() as u64 > limits.max_allocation_bytes {
            return Err(resource("filter plane exceeds the allocation limit"));
        }
        Ok(Self {
            width,
            height,
            stride: width,
            data,
        })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }

    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> u8 {
        let cx = x.clamp(0, self.width as isize - 1) as usize;
        let cy = y.clamp(0, self.height as isize - 1) as usize;
        self.get(cx, cy)
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: u8) {
        self.data[y * self.stride + x] = value;
    }
}

/// A reconstructed YUV frame passed through the filter pipeline. Chroma
/// planes are `None` for monochrome sequences (the only chroma layout the
/// AV1 decoders in this crate currently reconstruct is monochrome; the
/// 4:2:0/4:2:2/4:4:4 fields below are accepted so this module's chroma
/// handling, RGBA conversion, and subsampling-edge tests are exercised
/// ahead of chroma reconstruction landing in the inter/intra decoders).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterFrame {
    pub y: FilterPlane,
    pub u: Option<FilterPlane>,
    pub v: Option<FilterPlane>,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
}

impl FilterFrame {
    pub fn new_monochrome(y: FilterPlane) -> Self {
        Self {
            y,
            u: None,
            v: None,
            subsampling_x: true,
            subsampling_y: true,
        }
    }

    pub fn new_yuv(
        y: FilterPlane,
        u: FilterPlane,
        v: FilterPlane,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Result<Self> {
        let expected_w = chroma_dim(y.width, subsampling_x);
        let expected_h = chroma_dim(y.height, subsampling_y);
        if u.width != expected_w
            || u.height != expected_h
            || u.width != v.width
            || u.height != v.height
        {
            return Err(invalid(
                "chroma plane dimensions do not match the signaled subsampling",
            ));
        }
        Ok(Self {
            y,
            u: Some(u),
            v: Some(v),
            subsampling_x,
            subsampling_y,
        })
    }
}

/// Rounds a luma dimension down to its subsampled chroma dimension the way
/// AV1 does: `(dim + subsampling) >> subsampling` (spec §5.9.20 computes
/// `MiCols`/`MiRows` chroma sizes with this same ceil-shift rule so odd
/// luma dimensions never lose a chroma sample column/row).
#[doc(hidden)]
pub fn chroma_dim(luma: usize, subsampled: bool) -> usize {
    if subsampled { luma.div_ceil(2) } else { luma }
}

// ---------------------------------------------------------------------
// Output color conversion to Rgba8 (spec §7.11 range/matrix application
// mirrors ITU-T H.273 matrix coefficients already parsed in `av1::color_config`)
// ---------------------------------------------------------------------

/// Matrix coefficient identifiers this module can convert (ITU-T H.273
/// `MatrixCoefficients`, the same raw values already captured in
/// the AV1 color config's `color_description`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixCoefficients {
    Identity,
    Bt709,
    Bt601,
    Bt2020Ncl,
}

impl MatrixCoefficients {
    pub fn from_raw(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Bt709),
            5 | 6 => Ok(Self::Bt601),
            9 => Ok(Self::Bt2020Ncl),
            _ => Err(unsupported(
                "AV1 output conversion only supports Identity/BT.709/BT.601/BT.2020 NCL matrix coefficients",
            )),
        }
    }

    #[doc(hidden)]
    pub fn kr_kb(self) -> (f64, f64) {
        match self {
            MatrixCoefficients::Identity => (0.0, 0.0),
            MatrixCoefficients::Bt709 => (0.2126, 0.0722),
            MatrixCoefficients::Bt601 => (0.299, 0.114),
            MatrixCoefficients::Bt2020Ncl => (0.2627, 0.0593),
        }
    }
}

/// Converts a reconstructed [`FilterFrame`] to packed `Rgba8` bytes
/// (`width * height * 4`, alpha fixed at 255), applying the signaled
/// `color_range` and `matrix_coefficients`. Chroma is nearest-neighbor
/// upsampled for subsampled formats (4:2:0/4:2:2), which is a documented,
/// bounded-cost choice sufficient for the Main-profile 8-bit scope; it
/// only affects chroma placement smoothness, not determinism or range/matrix
/// correctness.
pub fn convert_to_rgba8(
    frame: &FilterFrame,
    color_range: ColorRange,
    matrix: MatrixCoefficients,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let width = frame.y.width;
    let height = frame.y.height;
    if width > limits.max_width as usize || height > limits.max_height as usize {
        return Err(resource("RGBA output dimensions exceed configured limits"));
    }
    let total = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(4))
        .ok_or_else(|| resource("RGBA output size overflow"))?;
    if total as u64 > limits.max_allocation_bytes {
        return Err(resource("RGBA output exceeds the allocation limit"));
    }

    let mut out = vec![0u8; total];
    let conversion = Conversion::new(color_range, matrix);
    // A monochrome frame converts with neutral chroma, which is one sample
    // reused for the whole row.
    let neutral = [128u8];
    for (py, out_row) in out.chunks_exact_mut(width * 4).enumerate() {
        let luma = &frame.y.data[py * frame.y.stride..py * frame.y.stride + width];
        let (cb, cr, subsampled_x) = match (&frame.u, &frame.v) {
            (Some(u), Some(v)) => {
                let cy = if frame.subsampling_y { py / 2 } else { py }.min(u.height - 1);
                (
                    &u.data[cy * u.stride..cy * u.stride + u.width],
                    &v.data[cy * v.stride..cy * v.stride + u.width.min(v.width)],
                    frame.subsampling_x,
                )
            }
            _ => (&neutral[..], &neutral[..], false),
        };
        let cb = &cb[..cr.len()];
        yuv_to_rgba::convert_row(&conversion, luma, cb, cr, subsampled_x, out_row);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// Error helpers (matching the style used across av1_inter_decoder.rs /
// av1_intra_decoder.rs)
// ---------------------------------------------------------------------

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn resource(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

