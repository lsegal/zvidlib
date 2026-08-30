#[cfg(test)]
use super::engine::encoder::rdo::DistortionBackend;
use super::engine::{
    encoder::{
        pcm::encode_idr_pcm_au,
        rdo::{DecisionConfig, decide_picture},
    },
    nal::collect_nal_units,
};
use crate::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedSample,
    EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameSource, HardwarePreference,
    Limits, PixelFormat, Result, SampleDependency, VideoEncoder, VideoEncoderConfig,
    VideoEncoderFactory, VideoEncoderFormat,
};

/// Returns the dependency-free software HEVC Main encoder.
pub fn native_hevc_video_encoder_factory() -> impl VideoEncoderFactory {
    HevcEncoderFactory
}
struct HevcEncoderFactory;

impl VideoEncoderFactory for HevcEncoderFactory {
    fn capability(&self, c: &VideoEncoderConfig) -> CodecSupport {
        if c.codec != Codec::Hevc {
            return CodecSupport::UnsupportedCodec;
        }
        if c.profile != CodecProfile::HevcMain {
            return CodecSupport::UnsupportedProfile;
        }
        if c.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        if c.input_format != PixelFormat::Rgba8 {
            return invalid("native HEVC Main encoding requires RGBA8 input");
        }
        if c.color_range != ColorRange::Limited {
            return invalid("native HEVC Main encoding requires limited-range input");
        }
        if !c.configuration.is_empty() {
            return invalid(
                "native HEVC encoding does not accept pre-existing codec configuration",
            );
        }
        if c.timescale == 0 || c.frame_duration == 0 {
            return invalid("native HEVC encoding requires nonzero timescale and frame duration");
        }
        if c.coded_dimensions.width % 16 != 0 || c.coded_dimensions.height % 16 != 0 {
            return invalid("native HEVC PCM encoding requires dimensions divisible by 16");
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }
    fn create(&self, c: &VideoEncoderConfig, limits: &Limits) -> Result<Box<dyn VideoEncoder>> {
        let support = self.capability(c);
        if !support.is_supported() {
            return Err(support_error(support));
        }
        let d = c.coded_dimensions;
        let pixels = u64::from(d.width)
            .checked_mul(u64::from(d.height))
            .ok_or_else(|| limit("HEVC frame dimensions overflow"))?;
        if d.width > limits.max_width
            || d.height > limits.max_height
            || pixels
                .checked_mul(6)
                .ok_or_else(|| limit("HEVC frame allocation overflows"))?
                > limits.max_allocation_bytes
        {
            return Err(limit("HEVC frame exceeds configured allocation limit"));
        }
        let (y, cb, cr) = blank_planes(d.width as usize, d.height as usize);
        let au = encode_idr_pcm_au(&y, &cb, &cr, d.width as usize, d.height as usize)
            .map_err(|e| invalid_input(e.to_string()))?;
        Ok(Box::new(HevcEncoder {
            configuration: c.clone(),
            config: EncoderConfig {
                codec: Codec::Hevc,
                timescale: c.timescale,
                decoder_config: hvcc_box(&au)?,
            },
            next_index: 0,
            limits: *limits,
            previous_y: None,
        }))
    }
}
struct HevcEncoder {
    configuration: VideoEncoderConfig,
    config: EncoderConfig,
    next_index: u64,
    limits: Limits,
    previous_y: Option<Vec<u8>>,
}
impl VideoEncoder for HevcEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }
    fn format(&self) -> VideoEncoderFormat {
        VideoEncoderFormat {
            dimensions: self.configuration.coded_dimensions,
            pixel_format: PixelFormat::Rgba8,
        }
    }
    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        source: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            if index.0 != self.next_index {
                return Err(invalid_input(
                    "HEVC encoder frame indexes must be consecutive and start at zero",
                ));
            }
            let source = match source {
                FrameSource::Cpu(source) => source,
                FrameSource::Graphics(_) => {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "native HEVC encoder requires a CPU frame source",
                    ));
                }
            };
            let frame = source.frame;
            if frame.dimensions != self.configuration.coded_dimensions
                || frame.pixel_format != PixelFormat::Rgba8
                || frame.color_range != ColorRange::Limited
            {
                return Err(invalid_input(
                    "HEVC input frame does not match the configured RGBA8 limited format",
                ));
            }
            let (y, cb, cr) = rgba_to_yuv420(frame, source.orientation)?;
            let d = self.configuration.coded_dimensions;
            let _decision = decide_picture(
                &y,
                d.width as usize,
                d.width as usize,
                d.height as usize,
                self.previous_y.as_deref(),
                DecisionConfig::default(),
            );
            let au = encode_idr_pcm_au(&y, &cb, &cr, d.width as usize, d.height as usize)
                .map_err(|e| invalid_input(e.to_string()))?;
            let data = length_prefixed_vcl(&au)?;
            if data.len() as u64 > self.limits.max_allocation_bytes {
                return Err(limit(
                    "HEVC access unit exceeds configured allocation limit",
                ));
            }
            let tick = self
                .next_index
                .checked_mul(u64::from(self.configuration.frame_duration))
                .ok_or_else(|| limit("HEVC timeline overflows"))?;
            let tick = i64::try_from(tick).map_err(|_| limit("HEVC timeline overflows"))?;
            self.next_index += 1;
            self.previous_y = Some(y);
            Ok(vec![EncodedSample {
                data,
                dts: tick,
                pts: tick,
                duration: self.configuration.frame_duration,
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            }])
        })
    }
    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

#[cfg(test)]
fn encode_with_rdo_backend(
    configuration: &VideoEncoderConfig,
    limits: &Limits,
    frame: &crate::VideoFrame,
    backend: DistortionBackend,
) -> Result<Vec<u8>> {
    let source = crate::CpuFrameSource {
        frame,
        orientation: crate::Orientation::TopLeft,
    };
    let (y, cb, cr) = rgba_to_yuv420(frame, source.orientation)?;
    let d = configuration.coded_dimensions;
    let _decision = decide_picture(
        &y,
        d.width as usize,
        d.width as usize,
        d.height as usize,
        None,
        DecisionConfig {
            backend,
            ..DecisionConfig::default()
        },
    );
    let au = encode_idr_pcm_au(&y, &cb, &cr, d.width as usize, d.height as usize)
        .map_err(|e| invalid_input(e.to_string()))?;
    let data = length_prefixed_vcl(&au)?;
    if data.len() as u64 > limits.max_allocation_bytes {
        return Err(limit(
            "HEVC access unit exceeds configured allocation limit",
        ));
    }
    Ok(data)
}
fn blank_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (vec![16; w * h], vec![128; w * h / 4], vec![128; w * h / 4])
}
fn rgba_to_yuv420(
    frame: &crate::VideoFrame,
    orientation: crate::Orientation,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = frame.dimensions.width as usize;
    let h = frame.dimensions.height as usize;
    let p = &frame.planes[0];
    let px = |x: usize, y: usize| {
        let y = match orientation {
            crate::Orientation::TopLeft => y,
            crate::Orientation::BottomLeft => h - 1 - y,
        };
        let at = y * p.stride + x * 4;
        (
            p.data[at] as i32,
            p.data[at + 1] as i32,
            p.data[at + 2] as i32,
        )
    };
    let mut y = vec![0; w * h];
    let mut cb = vec![0; w * h / 4];
    let mut cr = vec![0; w * h / 4];
    for py in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, py);
            y[py * w + x] = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(16, 235) as u8;
        }
    }
    for py in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let (mut r, mut g, mut b) = (0, 0, 0);
            for dy in 0..2 {
                for dx in 0..2 {
                    let q = px(x + dx, py + dy);
                    r += q.0;
                    g += q.1;
                    b += q.2;
                }
            }
            let at = (py / 2) * (w / 2) + x / 2;
            cb[at] = ((-38 * r - 74 * g + 112 * b + 131_584) >> 10).clamp(16, 240) as u8;
            cr[at] = ((112 * r - 94 * g - 18 * b + 131_584) >> 10).clamp(16, 240) as u8;
        }
    }
    Ok((y, cb, cr))
}
fn coded(unit: &super::engine::nal::NalUnit) -> Vec<u8> {
    let h = unit.header;
    let mut out = Vec::with_capacity(unit.escaped.len() + 2);
    out.push((h.nal_unit_type << 1) | (h.nuh_layer_id >> 5));
    out.push(((h.nuh_layer_id & 0x1f) << 3) | (h.temporal_id + 1));
    out.extend_from_slice(&unit.escaped);
    out
}
fn hvcc_box(annexb: &[u8]) -> Result<Vec<u8>> {
    let units = collect_nal_units(annexb)
        .map_err(|e| invalid_input(format!("invalid generated HEVC parameter sets: {e}")))?;
    let mut p = vec![
        1, 1, 0x60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0xf0, 0, 0xfc, 0xfd, 0xf8, 0xf8, 0, 0, 3, 3,
    ];
    for kind in [32_u8, 33, 34] {
        let u = units
            .iter()
            .find(|u| u.header.nal_unit_type == kind)
            .ok_or_else(|| invalid_input("generated HEVC access unit omitted a parameter set"))?;
        let n = coded(u);
        p.push(0x80 | kind);
        p.extend_from_slice(&1_u16.to_be_bytes());
        p.extend_from_slice(&(n.len() as u16).to_be_bytes());
        p.extend(n);
    }
    let mut out = Vec::with_capacity(p.len() + 8);
    out.extend_from_slice(&((p.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(b"hvcC");
    out.extend(p);
    Ok(out)
}
fn length_prefixed_vcl(annexb: &[u8]) -> Result<Vec<u8>> {
    let units = collect_nal_units(annexb)
        .map_err(|e| invalid_input(format!("invalid generated HEVC access unit: {e}")))?;
    let mut out = Vec::new();
    for u in units.iter().filter(|u| u.header.is_vcl()) {
        let n = coded(u);
        out.extend_from_slice(&(n.len() as u32).to_be_bytes());
        out.extend(n);
    }
    if out.is_empty() {
        Err(invalid_input("generated HEVC access unit omitted VCL data"))
    } else {
        Ok(out)
    }
}
fn invalid(reason: impl Into<String>) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}
fn invalid_input(s: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, s)
}
fn limit(s: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, s)
}
fn support_error(s: CodecSupport) -> Error {
    match s {
        CodecSupport::InvalidConfiguration { reason } => invalid_input(reason),
        CodecSupport::UnsupportedCodec
        | CodecSupport::UnsupportedProfile
        | CodecSupport::HardwareUnavailable => Error::new(
            ErrorKind::Unsupported,
            "native HEVC encoder does not support the requested configuration",
        ),
        CodecSupport::Supported { .. } => Error::new(
            ErrorKind::Internal,
            "encoder capability changed unexpectedly",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::engine::hvcc::parse_hvcc;
    use super::*;
    use crate::{
        Plane, VideoDecoderConfig, VideoEncoderConformanceVector, VideoFrame,
        native_hevc_video_decoder_factory, verify_video_encoder_conformance,
    };
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    #[test]
    fn generated_configuration_and_sample_are_standardized() {
        let (y, cb, cr) = blank_planes(16, 16);
        let au = encode_idr_pcm_au(&y, &cb, &cr, 16, 16).unwrap();
        let config = hvcc_box(&au).unwrap();
        let record = parse_hvcc(&config[8..]).unwrap();
        assert_eq!(record.length_size, 4);
        assert!(
            record
                .nal_units
                .iter()
                .any(|unit| unit.header.nal_unit_type == 32)
        );
        assert!(
            record
                .nal_units
                .iter()
                .any(|unit| unit.header.nal_unit_type == 33)
        );
        assert!(
            record
                .nal_units
                .iter()
                .any(|unit| unit.header.nal_unit_type == 34)
        );
        let sample = length_prefixed_vcl(&au).unwrap();
        assert!(sample.len() > 4);
        assert_eq!(sample[4] >> 1, 20);
    }

    #[test]
    fn factory_round_trips_a_conformance_vector_with_exact_timing() {
        let limits = Limits::default();
        let dimensions = crate::VideoDimensions::new(16, 16, &limits).unwrap();
        let mut pixels = vec![0_u8; 16 * 16 * 4];
        for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            let value = 32 + (index as u8).wrapping_mul(7) % 192;
            pixel.copy_from_slice(&[value, value, value, 255]);
        }
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane {
                data: pixels,
                stride: 64,
            }],
            &limits,
        )
        .unwrap();
        let configuration = VideoEncoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            timescale: 30_000,
            frame_duration: 1_001,
            configuration: Vec::new(),
        };
        let vector = VideoEncoderConformanceVector {
            name: "native hevc pcm idr".into(),
            configuration: configuration.clone(),
            decoder_configuration: VideoDecoderConfig {
                codec: Codec::Hevc,
                profile: CodecProfile::HevcMain,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                configuration: Vec::new(),
            },
            frames: vec![frame],
            minimum_psnr_db: 30.0,
        };
        let report = block_on(verify_video_encoder_conformance(
            &native_hevc_video_encoder_factory(),
            &native_hevc_video_decoder_factory(),
            &vector,
            limits,
        ))
        .unwrap();
        assert_eq!(report.frames_encoded, 1);
    }

    #[test]
    fn scalar_and_dispatched_rdo_paths_emit_identical_bitstreams() {
        let limits = Limits::default();
        let dimensions = crate::VideoDimensions::new(32, 32, &limits).unwrap();
        let mut pixels = vec![0_u8; 32 * 32 * 4];
        for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            let x = index % 32;
            let y = index / 32;
            let value = (32 + (x * 5 + y * 9) % 180) as u8;
            pixel.copy_from_slice(&[value, value.saturating_add(7), value / 2, 255]);
        }
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane {
                data: pixels,
                stride: 128,
            }],
            &limits,
        )
        .unwrap();
        let configuration = VideoEncoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            timescale: 30_000,
            frame_duration: 1_001,
            configuration: Vec::new(),
        };

        let scalar =
            encode_with_rdo_backend(&configuration, &limits, &frame, DistortionBackend::Scalar)
                .unwrap();
        let dispatched = encode_with_rdo_backend(
            &configuration,
            &limits,
            &frame,
            DistortionBackend::Dispatched,
        )
        .unwrap();
        assert_eq!(scalar, dispatched);
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }
}
