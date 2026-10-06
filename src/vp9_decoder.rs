//! Native VP9 decoding, registered as a [`VideoDecoderFactory`]: a
//! dependency-free software decoder, and the platform's hardware decoder
//! where the host has one.
//!
//! This wraps the crate's VP9 profile 0 decoder (`crate::vp9_dec`), which
//! decodes 8-bit 4:2:0 streams bit for bit as libvpx does: hidden frames,
//! superframes, `show_existing_frame`, intra-only frames, reference frames
//! of another size, tiles, segmentation, lossless coding and every
//! interpolation filter.
//!
//! A sample is one VP9 chunk as the VP9 ISO-BMFF binding defines it: a frame,
//! or a superframe whose hidden frames precede the one it shows. Each sample
//! must show a frame. Decoded pictures are converted to `Rgba8` with
//! [`convert_to_rgba8`], using the matrix the frame's `color_space` names
//! (BT.601 when it is unknown or one the conversion does not implement), at
//! the size the frame was coded at: a stream that changes resolution emits
//! frames of each size.
//!
//! The hardware backends decode the same chunks. A chunk's headers are read
//! before it reaches the platform decoder, so each sample still shows exactly
//! one frame, and the decoded picture is cropped to the size its header names
//! and converted by [`picture_to_rgba`] with the colour space and range the
//! header names, exactly as a software picture is.

use crate::vp9_dec::{DecodedPicture, Decoder};
use crate::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FilterFrame, FilterPlane,
    HardwarePreference, Limits, MatrixCoefficients, PixelFormat, Result, VideoDecoder,
    VideoDecoderConfig, VideoDecoderFactory, VideoDimensions, VideoFrame, Vp9CodecConfig,
    convert_to_rgba8,
};

/// Returns the native VP9 profile 0 (8-bit 4:2:0) decoder backend.
///
/// `Prefer` and `Require` select a hardware decoder where the host has one
/// that decodes VP9 profile 0 at the configured size: NVIDIA NVDEC on 64-bit
/// Windows and Linux, then Media Foundation on Windows adapters that expose the
/// D3D11 VP9 profile 0 decoder, and VideoToolbox on Macs whose media engine
/// decodes VP9.
/// `Prefer` falls back to the dependency-free software
/// decoder when none is available, and `Require` reports
/// [`CodecSupport::HardwareUnavailable`]. `Avoid` always selects software.
pub fn native_vp9_video_decoder_factory() -> impl VideoDecoderFactory {
    Vp9DecoderFactory
}

#[derive(Clone, Copy, Debug)]
struct Vp9DecoderFactory;

impl VideoDecoderFactory for Vp9DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        match self.capability_without_parsing(configuration) {
            CodecSupport::Supported { .. } => {}
            other => return other,
        }
        if let Err(error) = parse_configuration(configuration, &Limits::default()) {
            return invalid_configuration(error.message());
        }
        if configuration.hardware != HardwarePreference::Avoid && hardware_available(configuration)
        {
            return CodecSupport::Supported {
                implementation: CodecImplementation::Hardware,
            };
        }
        if configuration.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }

    fn create(
        &self,
        configuration: &VideoDecoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoDecoder>> {
        match self.capability_without_parsing(configuration) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::UnsupportedCodec => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP9 decoder requires VP9",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP9 decoder supports profile 0",
                ));
            }
            CodecSupport::HardwareUnavailable => unreachable!("hardware is not checked here"),
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        parse_configuration(configuration, limits)?;
        if configuration.hardware != HardwarePreference::Avoid {
            let hardware_errors = match create_hardware(configuration, limits) {
                Ok(decoder) => return Ok(decoder),
                Err(errors) => errors,
            };
            if configuration.hardware == HardwarePreference::Require {
                let detail = if hardware_errors.is_empty() {
                    "no accelerated backend exists for this target".to_owned()
                } else {
                    hardware_errors.join("; ")
                };
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    format!("hardware VP9 decoding is unavailable ({detail})"),
                ));
            }
        }
        Ok(Box::new(Vp9Decoder {
            limits: *limits,
            inner: Decoder::new(*limits),
        }))
    }
}

impl Vp9DecoderFactory {
    fn capability_without_parsing(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Vp9 {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Vp9Profile0 {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.output_format != PixelFormat::Rgba8 {
            return invalid_configuration("native VP9 decoding currently outputs RGBA8");
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }
}

fn hardware_available(_configuration: &VideoDecoderConfig) -> bool {
    #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
    if crate::hevc::nvdec::is_vp9_available(_configuration.coded_dimensions) {
        return true;
    }
    #[cfg(windows)]
    if crate::hevc::windows_mf::is_vp9_available(_configuration.coded_dimensions) {
        return true;
    }
    #[cfg(target_os = "macos")]
    if crate::hevc::videotoolbox_vp9::is_vp9_available(_configuration.coded_dimensions) {
        return true;
    }
    false
}

/// Creates the first hardware decoder that accepts the configuration, in the
/// order [`hardware_available`] checks them, or says why each refused it.
fn create_hardware(
    _configuration: &VideoDecoderConfig,
    _limits: &Limits,
) -> std::result::Result<Box<dyn VideoDecoder>, Vec<String>> {
    #[cfg_attr(
        not(any(
            windows,
            all(target_os = "linux", target_pointer_width = "64"),
            target_os = "macos"
        )),
        allow(unused_mut)
    )]
    let mut errors = Vec::new();
    #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
    match crate::hevc::nvdec::create_vp9(_configuration, _limits) {
        Ok(decoder) => return Ok(decoder),
        Err(error) => errors.push(format!("NVDEC: {}", error.message())),
    }
    #[cfg(windows)]
    match crate::hevc::windows_mf::create_vp9(_configuration, _limits) {
        Ok(decoder) => return Ok(decoder),
        Err(error) => errors.push(format!("Media Foundation: {}", error.message())),
    }
    #[cfg(target_os = "macos")]
    match crate::hevc::videotoolbox_vp9::create_vp9(_configuration, _limits) {
        Ok(decoder) => return Ok(decoder),
        Err(error) => errors.push(format!("VideoToolbox: {}", error.message())),
    }
    Err(errors)
}

fn invalid_configuration(reason: impl Into<String>) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

/// Validates the track's `vpcC` box (or WebM `CodecPrivate`). Either may be
/// empty or absent from a stream, since the VP9 bitstream describes itself;
/// when present it must describe a profile 0 8-bit 4:2:0 stream.
fn parse_configuration(configuration: &VideoDecoderConfig, limits: &Limits) -> Result<()> {
    if configuration.configuration.len() as u64 > limits.max_allocation_bytes {
        return Err(limit("VP9 configuration exceeds the allocation limit"));
    }
    let record = Vp9CodecConfig::parse(&configuration.configuration)
        .map_err(|error| malformed(format!("invalid VP9 configuration: {error}")))?;
    if record.profile != 0 || record.bit_depth != 8 || !record.is_420() {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "native VP9 decoder requires a profile 0, 8-bit 4:2:0 configuration",
        ));
    }
    Ok(())
}

struct Vp9Decoder {
    limits: Limits,
    inner: Decoder,
}

impl VideoDecoder for Vp9Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(limit("VP9 sample exceeds the allocation limit"));
        }
        let shown_before = self.inner.frames_shown();
        let picture = self.inner.decode_chunk(&sample.data)?;
        if self.inner.frames_shown() == shown_before {
            return Err(malformed("VP9 sample does not show a frame"));
        }
        let Some(picture) = picture else {
            // The caller asked not to see this frame.
            return Ok(Vec::new());
        };
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame: picture_to_rgba(&picture, &self.limits)?,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        // Every sample shows its frame as it is decoded; nothing is held
        // back for reordering.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset();
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        // The frame is still decoded and kept as a reference; only the
        // copy out of the reference buffer and the YUV-to-RGBA pass are
        // skipped.
        self.inner.set_output_wanted(wanted);
    }
}

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

pub(crate) fn picture_to_rgba(picture: &DecodedPicture, limits: &Limits) -> Result<VideoFrame> {
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
        vec![crate::Plane { data: rgba, stride }],
        limits,
    )
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vpcc(record: &[u8]) -> Vec<u8> {
        let mut bytes = (12u32 + record.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(b"vpcC");
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(record);
        bytes
    }

    fn config() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: Codec::Vp9,
            profile: CodecProfile::Vp9Profile0,
            coded_dimensions: VideoDimensions::new(16, 16, &Limits::default()).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: vpcc(&[0, 10, 0x80, 2, 2, 2, 0, 0]),
        }
    }

    #[test]
    fn capability_distinguishes_codec_profile_configuration_and_hardware() {
        let factory = native_vp9_video_decoder_factory();
        assert!(factory.capability(&config()).is_supported());

        let mut candidate = config();
        candidate.configuration = Vec::new();
        assert!(factory.capability(&candidate).is_supported());

        candidate = config();
        candidate.codec = Codec::Av1;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedCodec
        );

        candidate = config();
        candidate.profile = CodecProfile::Vp9Profile2;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedProfile
        );

        // Require and Prefer agree on whether this host has hardware.
        candidate = config();
        candidate.hardware = HardwarePreference::Require;
        let required = factory.capability(&candidate);
        candidate.hardware = HardwarePreference::Prefer;
        let preferred = factory.capability(&candidate);
        if hardware_available(&candidate) {
            let hardware = CodecSupport::Supported {
                implementation: CodecImplementation::Hardware,
            };
            assert_eq!((required, preferred), (hardware.clone(), hardware));
        } else {
            assert_eq!(required, CodecSupport::HardwareUnavailable);
            assert_eq!(
                preferred,
                CodecSupport::Supported {
                    implementation: CodecImplementation::Software
                }
            );
        }

        candidate = config();
        candidate.output_format = PixelFormat::Yuv420p8;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        // A 10-bit, a 4:4:4 and a truncated configuration.
        for configuration in [
            vpcc(&[0, 10, 0xa0, 2, 2, 2, 0, 0]),
            vpcc(&[1, 10, 0x86, 2, 2, 2, 0, 0]),
            vec![1, 2, 3],
        ] {
            candidate = config();
            candidate.configuration = configuration;
            assert!(matches!(
                factory.capability(&candidate),
                CodecSupport::InvalidConfiguration { .. }
            ));
        }
    }

    #[test]
    fn create_rejects_malformed_configuration_and_samples() {
        let mut candidate = config();
        candidate.configuration = vec![1, 2, 3];
        let error = native_vp9_video_decoder_factory()
            .create(&candidate, &Limits::default())
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);

        let mut decoder = native_vp9_video_decoder_factory()
            .create(&config(), &Limits::default())
            .unwrap();
        for data in [vec![], vec![0xff, 0xff], vec![0x82, 0x49, 0x83]] {
            let sample = EncodedVideoSample {
                presentation_index: crate::FrameIndex(0),
                random_access: true,
                data,
            };
            let error = decoder
                .submit(&sample, &CancellationToken::new())
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MalformedMedia);
        }
    }

    #[test]
    fn create_enforces_allocation_limit() {
        let restrictive = Limits {
            max_allocation_bytes: 0,
            ..Limits::default()
        };
        let error = native_vp9_video_decoder_factory()
            .create(&config(), &restrictive)
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    }

    #[test]
    fn color_spaces_pick_their_matrix() {
        assert_eq!(matrix_for(0), MatrixCoefficients::Bt601);
        assert_eq!(matrix_for(1), MatrixCoefficients::Bt601);
        assert_eq!(matrix_for(2), MatrixCoefficients::Bt709);
        assert_eq!(matrix_for(3), MatrixCoefficients::Bt601);
        assert_eq!(matrix_for(5), MatrixCoefficients::Bt2020Ncl);
    }

    /// Each hardware backend on its own, held to the software decoder.
    ///
    /// `tests/vp9_hardware.rs` checks whichever backend the factory selects,
    /// which is NVDEC on a host that has both it and Media Foundation; this
    /// reaches every backend the host has. For each fixture, every frame of one
    /// uninterrupted decode and of `ExactFrameReader`'s sequential, reverse and
    /// alternating seeks must match the software decoder's, and so must every
    /// reference slot shown again by `show_existing_frame`. A backend the host
    /// lacks is skipped with the reason.
    #[cfg(any(
        windows,
        all(target_os = "linux", target_pointer_width = "64"),
        target_os = "macos"
    ))]
    #[test]
    fn each_hardware_backend_matches_the_software_decoder_and_seeks_exactly() {
        use crate::{
            ExpectedVideoFrame, FrameDigest, FrameIndex, VideoDecoderConformanceVector,
            verify_video_decoder_conformance,
        };

        type Create = fn(&VideoDecoderConfig, &Limits) -> Result<Box<dyn VideoDecoder>>;

        struct Backend(Create);

        impl VideoDecoderFactory for Backend {
            fn capability(&self, _configuration: &VideoDecoderConfig) -> CodecSupport {
                CodecSupport::Supported {
                    implementation: CodecImplementation::Hardware,
                }
            }

            fn create(
                &self,
                configuration: &VideoDecoderConfig,
                limits: &Limits,
            ) -> Result<Box<dyn VideoDecoder>> {
                (self.0)(configuration, limits)
            }
        }

        fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
            use std::task::{Context, Poll, Waker};
            let mut context = Context::from_waker(Waker::noop());
            let mut future = Box::pin(future);
            loop {
                if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                    return value;
                }
            }
        }

        let mut backends: Vec<(&str, Create)> = Vec::new();
        #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
        backends.push(("NVDEC", crate::hevc::nvdec::create_vp9));
        #[cfg(windows)]
        backends.push(("Media Foundation", crate::hevc::windows_mf::create_vp9));
        #[cfg(target_os = "macos")]
        backends.push(("VideoToolbox", crate::hevc::videotoolbox_vp9::create_vp9));

        let limits = Limits::default();
        let mut streams = Vec::new();
        for (name, file) in [
            (
                "VP9 256x144 with hidden frames",
                &include_bytes!("../tests/fixtures/codec/vp9_bbb_256x144.mp4")[..],
            ),
            (
                "VP9 250x142",
                &include_bytes!("../tests/fixtures/codec/vp9_bbb_250x142.mp4")[..],
            ),
        ] {
            let source = crate::io::MemorySource::new(file.to_vec());
            let movie = block_on(crate::Mp4Demuxer::open(&source, Default::default())).unwrap();
            let track = movie.track(1).unwrap();
            let configuration = VideoDecoderConfig {
                codec: Codec::Vp9,
                profile: CodecProfile::Vp9Profile0,
                coded_dimensions: track.dimensions.unwrap(),
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                configuration: track.decoder_config.clone(),
            };
            let samples = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
            streams.push((name, configuration, samples));
        }
        // The first frames of the first fixture, then each reference slot shown
        // again by a one-byte `show_existing_frame` chunk: frame marker 2,
        // profile 0, show_existing_frame 1, then the slot.
        let (_, configuration, samples) = &streams[0];
        let mut existing = samples[..4].to_vec();
        for slot in 0..8u8 {
            existing.push(EncodedVideoSample {
                presentation_index: FrameIndex(existing.len() as u64),
                random_access: false,
                data: vec![0x88 | slot],
            });
        }
        streams.push(("VP9 show_existing_frame", configuration.clone(), existing));

        let cancellation = CancellationToken::new();
        let software = native_vp9_video_decoder_factory();
        for (backend, create) in backends {
            let factory = Backend(create);
            if let Err(error) = factory.create(&streams[0].1, &limits) {
                eprintln!("skipping {backend}: hardware VP9 decoding unavailable: {error}");
                continue;
            }
            for (name, configuration, samples) in &streams {
                let decode = |factory: &dyn VideoDecoderFactory| {
                    let mut decoder = factory.create(configuration, &limits).unwrap();
                    let mut digests = Vec::new();
                    for sample in samples {
                        let outputs = decoder
                            .submit(sample, &cancellation)
                            .unwrap_or_else(|error| panic!("{backend} {name}: {error}"));
                        assert_eq!(outputs.len(), 1, "{backend} {name}");
                        assert_eq!(outputs[0].presentation_index, sample.presentation_index);
                        digests.push(FrameDigest::from_frame(&outputs[0].frame).unwrap());
                    }
                    assert!(decoder.drain(&cancellation).unwrap().is_empty());
                    digests
                };
                let expected = decode(&software);
                assert_eq!(decode(&factory), expected, "{backend} {name}");
                let vector = VideoDecoderConformanceVector {
                    name: (*name).into(),
                    configuration: configuration.clone(),
                    expected_frames: expected
                        .iter()
                        .enumerate()
                        .map(|(index, &digest)| ExpectedVideoFrame {
                            presentation_index: FrameIndex(index as u64),
                            digest,
                        })
                        .collect(),
                    samples: samples.clone(),
                };
                verify_video_decoder_conformance(&factory, &vector, limits)
                    .unwrap_or_else(|error| panic!("{backend} {name}: {error}"));
            }
            eprintln!("{backend}: every VP9 frame matches the software decoder");
        }
    }
}
