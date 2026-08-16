use crate::{Error, ErrorKind, Limits, Result, SampleRange};

/// Container formats understood by capability discovery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Container {
    Mp4,
}

/// Normalized codec identifiers used across containers and backends.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Codec {
    UncompressedVideo,
    Hevc,
    Av1,
    Aac,
}

/// CPU video pixel formats supported by the portable media layer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgb8,
    Gray8,
    Yuv420p8,
}

/// Whether component values use the full numeric range or video range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ColorRange {
    Full,
    Limited,
}

/// Validated coded dimensions for a video frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VideoDimensions {
    pub width: u32,
    pub height: u32,
}

impl VideoDimensions {
    pub fn new(width: u32, height: u32, limits: &Limits) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(invalid_media("video dimensions must be nonzero"));
        }
        if width > limits.max_width || height > limits.max_height {
            return Err(resource_limit("video dimensions exceed configured limits"));
        }
        Ok(Self { width, height })
    }
}

/// One owned CPU image plane with an explicit row stride.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
}

/// A validated, owned CPU video frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoFrame {
    pub dimensions: VideoDimensions,
    pub pixel_format: PixelFormat,
    pub color_range: ColorRange,
    pub planes: Vec<Plane>,
}

impl VideoFrame {
    pub fn new(
        dimensions: VideoDimensions,
        pixel_format: PixelFormat,
        color_range: ColorRange,
        planes: Vec<Plane>,
        limits: &Limits,
    ) -> Result<Self> {
        let layouts = required_plane_layouts(dimensions, pixel_format)?;
        if planes.len() != layouts.len() {
            return Err(invalid_media("plane count does not match the pixel format"));
        }

        let mut allocation = 0_u64;
        for (plane, (row_bytes, rows)) in planes.iter().zip(layouts) {
            if plane.stride < row_bytes {
                return Err(invalid_media("a video plane stride is too small"));
            }
            let required = plane
                .stride
                .checked_mul(rows)
                .ok_or_else(|| resource_limit("video plane size overflow"))?;
            if plane.data.len() < required {
                return Err(invalid_media(
                    "a video plane is shorter than its stride and height",
                ));
            }
            allocation =
                allocation
                    .checked_add(u64::try_from(plane.data.len()).map_err(|_| {
                        resource_limit("video frame allocation cannot be represented")
                    })?)
                    .ok_or_else(|| resource_limit("video frame allocation overflow"))?;
        }
        if allocation > limits.max_allocation_bytes {
            return Err(resource_limit("video frame exceeds the allocation limit"));
        }

        Ok(Self {
            dimensions,
            pixel_format,
            color_range,
            planes,
        })
    }
}

/// Interleaved, normalized `f32` audio samples for an exact sample interval.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub range: SampleRange,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn new(
        range: SampleRange,
        sample_rate: u32,
        channels: u16,
        samples: Vec<f32>,
        limits: &Limits,
    ) -> Result<Self> {
        if sample_rate == 0 || sample_rate > limits.max_sample_rate {
            return Err(resource_limit(
                "audio sample rate is outside configured limits",
            ));
        }
        if channels == 0 || channels > limits.max_audio_channels {
            return Err(resource_limit(
                "audio channel count is outside configured limits",
            ));
        }
        let expected = range
            .len()
            .checked_mul(u64::from(channels))
            .ok_or_else(|| resource_limit("audio sample count overflow"))?;
        if u64::try_from(samples.len()).ok() != Some(expected) {
            return Err(invalid_media(
                "audio data length does not match its range and channels",
            ));
        }
        let bytes = expected
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| resource_limit("audio allocation size overflow"))?;
        if bytes > limits.max_allocation_bytes {
            return Err(resource_limit("audio buffer exceeds the allocation limit"));
        }
        Ok(Self {
            range,
            sample_rate,
            channels,
            samples,
        })
    }
}

fn required_plane_layouts(
    dimensions: VideoDimensions,
    pixel_format: PixelFormat,
) -> Result<Vec<(usize, usize)>> {
    let width = usize::try_from(dimensions.width)
        .map_err(|_| resource_limit("video width cannot be represented"))?;
    let height = usize::try_from(dimensions.height)
        .map_err(|_| resource_limit("video height cannot be represented"))?;
    let packed = |bytes_per_pixel: usize| -> Result<Vec<(usize, usize)>> {
        Ok(vec![(
            width
                .checked_mul(bytes_per_pixel)
                .ok_or_else(|| resource_limit("video row size overflow"))?,
            height,
        )])
    };
    match pixel_format {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => packed(4),
        PixelFormat::Rgb8 => packed(3),
        PixelFormat::Gray8 => packed(1),
        PixelFormat::Yuv420p8 => Ok(vec![
            (width, height),
            (width.div_ceil(2), height.div_ceil(2)),
            (width.div_ceil(2), height.div_ceil(2)),
        ]),
    }
}

fn invalid_media(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn resource_limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_apply_resource_limits() {
        let limits = Limits {
            max_width: 1_920,
            max_height: 1_080,
            ..Limits::default()
        };
        assert!(VideoDimensions::new(1_920, 1_080, &limits).is_ok());
        assert_eq!(
            VideoDimensions::new(3_840, 2_160, &limits)
                .unwrap_err()
                .kind(),
            ErrorKind::ResourceLimit
        );
    }

    #[test]
    fn packed_frame_validates_stride_and_length() {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(2, 2, &limits).unwrap();
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Full,
            vec![Plane {
                data: vec![0; 16],
                stride: 8,
            }],
            &limits,
        );
        assert!(frame.is_ok());

        let error = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Full,
            vec![Plane {
                data: vec![0; 14],
                stride: 8,
            }],
            &limits,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn planar_frame_supports_odd_chroma_dimensions() {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(3, 3, &limits).unwrap();
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Yuv420p8,
            ColorRange::Limited,
            vec![
                Plane {
                    data: vec![0; 9],
                    stride: 3,
                },
                Plane {
                    data: vec![0; 4],
                    stride: 2,
                },
                Plane {
                    data: vec![0; 4],
                    stride: 2,
                },
            ],
            &limits,
        );
        assert!(frame.is_ok());
    }

    #[test]
    fn audio_length_must_match_range_and_channels() {
        let limits = Limits::default();
        let range = SampleRange::new(100, 103).unwrap();
        assert!(AudioBuffer::new(range, 48_000, 2, vec![0.0; 6], &limits).is_ok());
        let error = AudioBuffer::new(range, 48_000, 2, vec![0.0; 5], &limits).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
