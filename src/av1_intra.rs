//! Bounded pixel primitives used by AV1 intra-frame reconstruction.
//!
//! This is deliberately a reconstruction building block rather than a video
//! decoder.  Tile syntax owns mode and coefficient parsing; this module owns
//! checked YUV allocation and applying a decoded intra block to a plane.

use crate::{
    ColorRange, Error, ErrorKind, Limits, PixelFormat, Plane, Result, VideoDimensions, VideoFrame,
};

/// Intra predictors used by the first AV1 reconstruction stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1IntraMode {
    Dc,
    Vertical,
    Horizontal,
    Paeth,
}

/// A bounded 8-bit 4:2:0 reconstruction target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1IntraFrame {
    dimensions: VideoDimensions,
    planes: [Vec<u8>; 3],
    strides: [usize; 3],
}

impl Av1IntraFrame {
    /// Allocates neutral-chroma planes after applying the public frame limits.
    pub fn new(dimensions: VideoDimensions, limits: &Limits) -> Result<Self> {
        let width = usize::try_from(dimensions.width)
            .map_err(|_| limit("AV1 width is not representable"))?;
        let height = usize::try_from(dimensions.height)
            .map_err(|_| limit("AV1 height is not representable"))?;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let y_len = width
            .checked_mul(height)
            .ok_or_else(|| limit("AV1 luma plane size overflows"))?;
        let c_len = chroma_width
            .checked_mul(chroma_height)
            .ok_or_else(|| limit("AV1 chroma plane size overflows"))?;
        let total = y_len
            .checked_add(c_len)
            .and_then(|n| n.checked_add(c_len))
            .ok_or_else(|| limit("AV1 frame size overflows"))?;
        if u64::try_from(total).map_err(|_| limit("AV1 frame size is not representable"))?
            > limits.max_allocation_bytes
        {
            return Err(limit(
                "AV1 reconstructed frame exceeds the allocation limit",
            ));
        }
        Ok(Self {
            dimensions,
            planes: [vec![0; y_len], vec![128; c_len], vec![128; c_len]],
            strides: [width, chroma_width, chroma_width],
        })
    }

    /// Applies one luma or chroma intra block. Residuals are signed spatial
    /// samples after the tile decoder's inverse transform stage.
    pub fn reconstruct_block(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        mode: Av1IntraMode,
        residuals: &[i16],
    ) -> Result<()> {
        if plane > 2 || width == 0 || height == 0 {
            return Err(malformed("AV1 intra block has an invalid plane or size"));
        }
        let count = width
            .checked_mul(height)
            .ok_or_else(|| limit("AV1 intra block size overflows"))?;
        if residuals.len() != count {
            return Err(malformed(
                "AV1 intra residual count does not match block size",
            ));
        }
        let stride = self.strides[plane];
        let rows = self.planes[plane].len() / stride;
        if x.checked_add(width).is_none_or(|end| end > stride)
            || y.checked_add(height).is_none_or(|end| end > rows)
        {
            return Err(malformed(
                "AV1 intra block exceeds its reconstruction plane",
            ));
        }
        let top_left = if x > 0 && y > 0 {
            self.planes[plane][(y - 1) * stride + x - 1]
        } else {
            128
        };
        let mut top = vec![128; width];
        let mut left = vec![128; height];
        if y > 0 {
            for (i, sample) in top.iter_mut().enumerate() {
                *sample = self.planes[plane][(y - 1) * stride + x + i];
            }
        }
        if x > 0 {
            for (i, sample) in left.iter_mut().enumerate() {
                *sample = self.planes[plane][(y + i) * stride + x - 1];
            }
        }
        let dc = (top.iter().map(|&v| u32::from(v)).sum::<u32>()
            + left.iter().map(|&v| u32::from(v)).sum::<u32>())
            / u32::try_from(width + height).expect("nonzero block dimensions");
        for row in 0..height {
            for column in 0..width {
                let prediction = match mode {
                    Av1IntraMode::Dc => dc as u8,
                    Av1IntraMode::Vertical => top[column],
                    Av1IntraMode::Horizontal => left[row],
                    Av1IntraMode::Paeth => paeth(top_left, top[column], left[row]),
                };
                self.planes[plane][(y + row) * stride + x + column] =
                    (i16::from(prediction) + residuals[row * width + column]).clamp(0, 255) as u8;
            }
        }
        Ok(())
    }

    pub fn into_video_frame(self, limits: &Limits) -> Result<VideoFrame> {
        VideoFrame::new(
            self.dimensions,
            PixelFormat::Yuv420p8,
            ColorRange::Limited,
            vec![
                Plane {
                    data: self.planes[0].clone(),
                    stride: self.strides[0],
                },
                Plane {
                    data: self.planes[1].clone(),
                    stride: self.strides[1],
                },
                Plane {
                    data: self.planes[2].clone(),
                    stride: self.strides[2],
                },
            ],
            limits,
        )
    }
}

/// The inverse 4x4 Walsh-Hadamard transform used by lossless AV1 blocks.
pub fn inverse_wht_4x4(coefficients: &[i32; 16]) -> [i16; 16] {
    let mut intermediate = [0i32; 16];
    for row in 0..4 {
        let offset = row * 4;
        let a = coefficients[offset] + coefficients[offset + 2];
        let b = coefficients[offset + 1] + coefficients[offset + 3];
        let c = coefficients[offset + 1] - coefficients[offset + 3];
        let d = coefficients[offset] - coefficients[offset + 2];
        intermediate[offset] = a + b;
        intermediate[offset + 1] = d + c;
        intermediate[offset + 2] = a - b;
        intermediate[offset + 3] = d - c;
    }
    let mut output = [0i16; 16];
    for column in 0..4 {
        let a = intermediate[column] + intermediate[8 + column];
        let b = intermediate[4 + column] + intermediate[12 + column];
        let c = intermediate[4 + column] - intermediate[12 + column];
        let d = intermediate[column] - intermediate[8 + column];
        output[column] = ((a + b) >> 4).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        output[4 + column] = ((d + c) >> 4).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        output[8 + column] = ((a - b) >> 4).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        output[12 + column] = ((d - c) >> 4).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    output
}

fn paeth(top_left: u8, top: u8, left: u8) -> u8 {
    let base = i16::from(top) + i16::from(left) - i16::from(top_left);
    let distance = |candidate: u8| (base - i16::from(candidate)).unsigned_abs();
    let top_distance = distance(top);
    let left_distance = distance(left);
    let corner_distance = distance(top_left);
    if top_distance <= left_distance && top_distance <= corner_distance {
        top
    } else if left_distance <= corner_distance {
        left
    } else {
        top_left
    }
}

fn malformed(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}
fn limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_is_bounded_and_exports_valid_yuv() {
        let limits = Limits {
            max_allocation_bytes: 32,
            ..Limits::default()
        };
        let dimensions = VideoDimensions::new(4, 4, &limits).unwrap();
        let mut frame = Av1IntraFrame::new(dimensions, &limits).unwrap();
        frame
            .reconstruct_block(0, 0, 0, 4, 4, Av1IntraMode::Dc, &[0; 16])
            .unwrap();
        let frame = frame.into_video_frame(&limits).unwrap();
        assert_eq!(frame.planes[0].data, vec![128; 16]);
        assert_eq!(frame.planes[1].data, vec![128; 4]);
    }

    #[test]
    fn blocks_cannot_escape_their_plane() {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(4, 4, &limits).unwrap();
        let mut frame = Av1IntraFrame::new(dimensions, &limits).unwrap();
        assert_eq!(
            frame
                .reconstruct_block(0, 3, 0, 2, 1, Av1IntraMode::Dc, &[0, 0])
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
    }

    #[test]
    fn wht_preserves_dc_energy() {
        let mut coefficients = [0; 16];
        coefficients[0] = 64;
        assert_eq!(inverse_wht_4x4(&coefficients), [4; 16]);
    }
}
