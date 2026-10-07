//! A decoded VP9 picture's conversion to `Rgba8`.

use zvidlib_color::{FilterFrame, FilterPlane, MatrixCoefficients, convert_to_rgba8};
use zvidlib_core::{ColorRange, Limits, PixelFormat, Result, VideoDimensions, VideoFrame};

use crate::{DecodedPicture, limit, malformed};

/// The conversion matrix for a VP9 `color_space`: `CS_BT_709` (2) and
/// `CS_BT_2020` (5) have their own, and everything else, including
/// `CS_UNKNOWN`, `CS_BT_601` and `CS_SMPTE_170`, uses BT.601.
fn matrix_for(color_space: u8) -> MatrixCoefficients {
    match color_space {
        2 => MatrixCoefficients::Bt709,
        5 => MatrixCoefficients::Bt2020Ncl,
        _ => MatrixCoefficients::Bt601,
    }
}

pub fn picture_to_rgba(picture: &DecodedPicture, limits: &Limits) -> Result<VideoFrame> {
    let width = picture.width;
    let height = picture.height;
    let dimensions = VideoDimensions::new(width as u32, height as u32, limits)?;
    let plane = |index: usize, plane_width: usize, plane_height: usize| {
        FilterPlane::from_samples(
            plane_width,
            plane_height,
            picture.planes[index].clone(),
            limits,
        )
        .map_err(|error| malformed(format!("invalid VP9 decoded plane: {error}")))
    };
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let frame = FilterFrame::new_yuv(
        plane(0, width, height)?,
        plane(1, chroma_width, chroma_height)?,
        plane(2, chroma_width, chroma_height)?,
        true,
        true,
    )
    .map_err(|error| malformed(format!("invalid VP9 decoded plane layout: {error}")))?;
    let color_range = if picture.full_range {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };
    let rgba = convert_to_rgba8(&frame, color_range, matrix_for(picture.color_space), limits)?;
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| limit("VP9 RGBA stride overflows"))?;
    VideoFrame::new(
        dimensions,
        PixelFormat::Rgba8,
        color_range,
        vec![zvidlib_core::Plane { data: rgba, stride }],
        limits,
    )
}
