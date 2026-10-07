//! What the hardware VP8 and VP9 backends (NVDEC and Media Foundation) share: recognising a VP8
//! frame's shape from its tag, and turning the NV12 surface a decoder writes into the three 4:2:0
//! planes the software decoder produces, so both go through the same RGBA conversion.

use crate::{
    EncodedVideoSample, Error, ErrorKind, Limits, Result, VideoDecoderConfig, VideoDimensions,
    VideoFrame,
};
use zvidlib_vp9_syntax::{DecodedPicture, FrameShape};

/// Converts a decoded picture, cropped to the configured dimensions and given as three tightly
/// packed 4:2:0 planes, to the frame the decoder returns.
pub type PlanarConverter = fn([Vec<u8>; 3], &VideoDecoderConfig, &Limits) -> Result<VideoFrame>;

/// Checks a VP8 sample before it is handed to a hardware decoder, and returns its frame.
pub fn vp8_sample(sample: &EncodedVideoSample, max_allocation_bytes: u64) -> Result<&[u8]> {
    // A VP8 frame tag is three bytes, and a key frame adds seven more; anything shorter is
    // refused here, as the software decoder refuses it, rather than handed to the driver.
    let minimum = if sample.data.first().is_some_and(|tag| tag & 1 == 0) {
        10
    } else {
        3
    };
    if sample.data.len() < minimum {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "VP8 frame is truncated",
        ));
    }
    if sample.data.len() as u64 > max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "VP8 frame exceeds the allocation limit",
        ));
    }
    Ok(&sample.data)
}

/// The `show_frame` bit of a VP8 frame tag (RFC 6386 section 9.1).
pub fn vp8_frame_is_shown(data: &[u8]) -> bool {
    data.first().is_some_and(|tag| tag & 0x10 != 0)
}

/// Crops an NV12 surface to `dimensions` and splits its interleaved chroma, giving the three
/// 4:2:0 planes a software decoder would have produced.
///
/// `data` starts at the first luma row; rows are `pitch` bytes apart, and the interleaved chroma
/// plane starts `surface_height` rows in.
pub fn nv12_to_planar(
    data: &[u8],
    pitch: usize,
    surface_height: usize,
    dimensions: VideoDimensions,
) -> Result<[Vec<u8>; 3]> {
    let width = dimensions.width as usize;
    let height = dimensions.height as usize;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let chroma_start = pitch.checked_mul(surface_height).ok_or_else(|| {
        Error::new(
            ErrorKind::ResourceLimit,
            "decoded NV12 surface size overflows",
        )
    })?;
    if pitch < chroma_width * 2
        || surface_height < height
        || data.len() < chroma_start + pitch * chroma_height
    {
        return Err(Error::new(
            ErrorKind::Codec,
            "decoded NV12 surface is smaller than the frame",
        ));
    }
    let mut luma = Vec::with_capacity(width * height);
    for row in data.chunks(pitch).take(height) {
        luma.extend_from_slice(&row[..width]);
    }
    let mut u = Vec::with_capacity(chroma_width * chroma_height);
    let mut v = Vec::with_capacity(chroma_width * chroma_height);
    for row in data[chroma_start..].chunks(pitch).take(chroma_height) {
        for pair in row[..chroma_width * 2].chunks_exact(2) {
            u.push(pair[0]);
            v.push(pair[1]);
        }
    }
    Ok([luma, u, v])
}

/// The size a VP9 frame's header gives it, which its picture is cropped to.
pub fn vp9_dimensions(shape: FrameShape, limits: &Limits) -> Result<VideoDimensions> {
    VideoDimensions::new(shape.width as u32, shape.height as u32, limits)
}

/// Converts a VP9 picture, cropped to [`vp9_dimensions`], with the colour its header named, as
/// the software decoder converts its own pictures.
pub fn vp9_frame(planes: [Vec<u8>; 3], shape: FrameShape, limits: &Limits) -> Result<VideoFrame> {
    let picture = DecodedPicture {
        width: shape.width,
        height: shape.height,
        planes,
        color_space: shape.color_space,
        full_range: shape.full_range,
    };
    zvidlib_vp9_syntax::picture_to_rgba(&picture, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_is_cropped_to_odd_dimensions_and_deinterleaved() {
        // A 3x3 picture on a 4x4 surface with a pitch of 6: luma samples are 10 * row + column,
        // and chroma pairs are (100 + n, 200 + n).
        let mut data = vec![0_u8; 6 * 4 + 6 * 2];
        for row in 0..4 {
            for column in 0..4 {
                data[row * 6 + column] = (10 * row + column) as u8;
            }
        }
        for row in 0..2 {
            for pair in 0..2 {
                let n = (row * 2 + pair) as u8;
                data[24 + row * 6 + pair * 2] = 100 + n;
                data[24 + row * 6 + pair * 2 + 1] = 200 + n;
            }
        }
        let dimensions = VideoDimensions::new(3, 3, &Limits::default()).unwrap();

        let [luma, u, v] = nv12_to_planar(&data, 6, 4, dimensions).unwrap();

        assert_eq!(luma, vec![0, 1, 2, 10, 11, 12, 20, 21, 22]);
        assert_eq!(u, vec![100, 101, 102, 103]);
        assert_eq!(v, vec![200, 201, 202, 203]);
    }

    #[test]
    fn a_surface_smaller_than_the_frame_is_an_error() {
        let dimensions = VideoDimensions::new(4, 4, &Limits::default()).unwrap();
        let error = nv12_to_planar(&[0; 4 * 2 + 4], 4, 2, dimensions).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Codec);
    }

    #[test]
    fn a_hidden_vp8_frame_is_recognised_from_its_tag() {
        assert!(vp8_frame_is_shown(&[0x10, 0, 0]));
        assert!(!vp8_frame_is_shown(&[0x00, 0, 0]));
        assert!(!vp8_frame_is_shown(&[]));
    }
}
