//! Backend-independent video codec conformance helpers.
//!
//! Decoder vectors carry encoded samples plus fingerprints of the expected
//! presentation frames. Encoder vectors are decoded by a separately supplied
//! decoder backend and checked against their source frames with a PSNR floor.

use crate::io::ByteSource;
use crate::{
    CancellationToken, CodecSupport, CpuFrameSource, DecodeStatistics, EncodedSample,
    EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, FrameSource, Limits,
    Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Result, VideoDecoderConfig,
    VideoDecoderFactory, VideoEncoderConfig, VideoEncoderFactory, VideoFrame,
};
use std::collections::{HashMap, HashSet};

/// A SHA-256 fingerprint of a frame's normalized metadata and active pixels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameDigest(pub [u8; 32]);

impl FrameDigest {
    /// Fingerprints active rows only, so backend-specific stride padding does
    /// not affect conformance results.
    pub fn from_frame(frame: &VideoFrame) -> Result<Self> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&frame.dimensions.width.to_be_bytes());
        bytes.extend_from_slice(&frame.dimensions.height.to_be_bytes());
        bytes.push(pixel_format_tag(frame.pixel_format));
        bytes.push(match frame.color_range {
            crate::ColorRange::Full => 0,
            crate::ColorRange::Limited => 1,
        });
        bytes.push(u8::try_from(frame.planes.len()).map_err(|_| {
            Error::new(ErrorKind::ResourceLimit, "video frame has too many planes")
        })?);
        append_active_pixels(frame, &mut bytes)?;
        Ok(Self(sha256(&bytes)))
    }

    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }

    pub fn from_hex(value: &str) -> Result<Self> {
        if value.len() != 64 || !value.is_ascii() {
            return Err(invalid(
                "frame fingerprints must contain 64 hexadecimal digits",
            ));
        }
        let mut digest = [0_u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            let start = index * 2;
            *byte = u8::from_str_radix(&value[start..start + 2], 16)
                .map_err(|_| invalid("frame fingerprints must contain hexadecimal digits"))?;
        }
        Ok(Self(digest))
    }
}

/// One expected presentation frame in a decoder conformance vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpectedVideoFrame {
    pub presentation_index: FrameIndex,
    pub digest: FrameDigest,
}

/// Encoded input and expected output for a decoder backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoDecoderConformanceVector {
    pub name: String,
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub expected_frames: Vec<ExpectedVideoFrame>,
}

impl VideoDecoderConformanceVector {
    /// Builds a complete vector from one checked-in MP4 track and its expected
    /// presentation fingerprints. Container configuration and decode-order
    /// samples come from the same bounded demux path used by applications.
    pub async fn from_mp4<S: ByteSource + ?Sized>(
        name: impl Into<String>,
        source: &S,
        options: Mp4DemuxerOptions,
        track_id: u32,
        mut configuration: VideoDecoderConfig,
        expected_digests: &[FrameDigest],
    ) -> Result<Self> {
        let limits = options.limits;
        let movie = Mp4Demuxer::open(source, options).await?;
        let track = movie.track(track_id).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "codec vector MP4 track does not exist",
            )
        })?;
        if track.codec != configuration.codec
            || track.dimensions != Some(configuration.coded_dimensions)
        {
            return Err(invalid(
                "codec vector MP4 track does not match its decoder configuration",
            ));
        }
        let samples = track.to_encoded_video_samples(source, &limits).await?;
        if samples.len() != expected_digests.len() {
            return Err(invalid(
                "codec vector MP4 requires one fingerprint per presentation frame",
            ));
        }
        configuration.configuration = track.decoder_config.clone();
        let expected_frames = expected_digests
            .iter()
            .copied()
            .enumerate()
            .map(|(index, digest)| ExpectedVideoFrame {
                presentation_index: FrameIndex(index as u64),
                digest,
            })
            .collect();
        Ok(Self {
            name: name.into(),
            configuration,
            samples,
            expected_frames,
        })
    }
}

/// Work completed while verifying a decoder vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VideoDecoderConformanceReport {
    pub frames_verified: u64,
    pub access_patterns_verified: u8,
    pub statistics: DecodeStatistics,
}

/// Verifies a decoder against sequential, reverse, and seek-heavy access.
pub fn verify_video_decoder_conformance(
    factory: &dyn VideoDecoderFactory,
    vector: &VideoDecoderConformanceVector,
    limits: Limits,
) -> Result<VideoDecoderConformanceReport> {
    validate_decoder_vector(vector)?;
    let mut sequential: Vec<_> = vector
        .expected_frames
        .iter()
        .map(|frame| frame.presentation_index)
        .collect();
    sequential.sort_by_key(|index| index.0);
    let mut reverse = sequential.clone();
    reverse.reverse();
    let mut seek_heavy = Vec::with_capacity(sequential.len());
    let (mut low, mut high) = (0_usize, sequential.len());
    while low < high {
        high -= 1;
        seek_heavy.push(sequential[high]);
        if low < high {
            seek_heavy.push(sequential[low]);
            low += 1;
        }
    }

    let expected: HashMap<_, _> = vector
        .expected_frames
        .iter()
        .map(|frame| (frame.presentation_index, frame.digest))
        .collect();
    let mut report = VideoDecoderConformanceReport::default();
    for pattern in [&sequential, &reverse, &seek_heavy] {
        let mut reader = ExactFrameReader::new(
            factory,
            vector.configuration.clone(),
            vector.samples.clone(),
            limits,
        )?;
        let cancellation = CancellationToken::new();
        for &index in pattern {
            let frame = reader.get(index, &cancellation)?;
            let actual = FrameDigest::from_frame(&frame)?;
            let wanted = expected[&index];
            if actual != wanted {
                return Err(Error::new(
                    ErrorKind::Codec,
                    format!(
                        "codec vector '{}' frame {} fingerprint mismatch: expected {}, got {}",
                        vector.name,
                        index.0,
                        wanted.to_hex(),
                        actual.to_hex()
                    ),
                ));
            }
            report.frames_verified = report.frames_verified.saturating_add(1);
        }
        report.statistics = add_statistics(report.statistics, reader.statistics());
        report.access_patterns_verified = report.access_patterns_verified.saturating_add(1);
    }
    Ok(report)
}

/// Source frames and quality requirements for an encoder conformance run.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoEncoderConformanceVector {
    pub name: String,
    pub configuration: VideoEncoderConfig,
    /// The decoder configuration is completed with the encoder's emitted
    /// standardized configuration bytes before round-trip decoding.
    pub decoder_configuration: VideoDecoderConfig,
    pub frames: Vec<VideoFrame>,
    pub minimum_psnr_db: f64,
}

/// Work completed while verifying an encoder vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VideoEncoderConformanceReport {
    pub frames_encoded: u64,
    pub packets_emitted: u64,
    pub minimum_observed_psnr_db: f64,
}

/// Encodes a vector, validates packet timing/configuration, then decodes it
/// with an independently supplied conforming decoder and checks image quality.
pub async fn verify_video_encoder_conformance(
    encoder_factory: &dyn VideoEncoderFactory,
    decoder_factory: &dyn VideoDecoderFactory,
    vector: &VideoEncoderConformanceVector,
    limits: Limits,
) -> Result<VideoEncoderConformanceReport> {
    validate_encoder_vector(vector)?;
    require_supported(
        encoder_factory.capability(&vector.configuration),
        "encoder does not support the conformance vector",
    )?;
    let mut encoder = encoder_factory.create(&vector.configuration, &limits)?;
    let format = encoder.format();
    if format.dimensions != vector.configuration.coded_dimensions
        || format.pixel_format != vector.configuration.input_format
    {
        return Err(invalid(
            "encoder format does not match its conformance configuration",
        ));
    }

    let mut packets = Vec::new();
    for (index, frame) in vector.frames.iter().enumerate() {
        let index =
            FrameIndex(u64::try_from(index).map_err(|_| {
                Error::new(ErrorKind::ResourceLimit, "encoder frame index overflow")
            })?);
        packets.extend(
            encoder
                .encode(
                    index,
                    FrameSource::Cpu(CpuFrameSource {
                        frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await?,
        );
    }
    packets.extend(encoder.finish().await?);
    let encoder_config = encoder.config().clone();
    validate_encoded_packets(&packets, vector.frames.len(), &encoder_config, vector)?;
    let packet_count = packets.len();

    let mut decoder_configuration = vector.decoder_configuration.clone();
    decoder_configuration.configuration = encoder_config.decoder_config;
    let samples = packets_to_video_samples(packets)?;
    let mut reader =
        ExactFrameReader::new(decoder_factory, decoder_configuration, samples, limits)?;
    let cancellation = CancellationToken::new();
    let mut minimum_observed = f64::INFINITY;
    for (index, source) in vector.frames.iter().enumerate() {
        let index = FrameIndex(index as u64);
        let decoded = reader.get(index, &cancellation)?;
        let psnr = frame_psnr(source, &decoded)?;
        minimum_observed = minimum_observed.min(psnr);
        if psnr < vector.minimum_psnr_db {
            return Err(Error::new(
                ErrorKind::Codec,
                format!(
                    "codec vector '{}' frame {} PSNR {psnr:.3} dB is below {:.3} dB",
                    vector.name, index.0, vector.minimum_psnr_db
                ),
            ));
        }
    }
    Ok(VideoEncoderConformanceReport {
        frames_encoded: vector.frames.len() as u64,
        packets_emitted: packet_count as u64,
        minimum_observed_psnr_db: minimum_observed,
    })
}

fn validate_decoder_vector(vector: &VideoDecoderConformanceVector) -> Result<()> {
    if vector.name.trim().is_empty() || vector.samples.is_empty() {
        return Err(invalid(
            "decoder conformance vectors require a name and samples",
        ));
    }
    if vector.expected_frames.len() != vector.samples.len() {
        return Err(invalid(
            "decoder conformance vectors require one expected frame per sample",
        ));
    }
    let sample_indexes: HashSet<_> = vector
        .samples
        .iter()
        .map(|sample| sample.presentation_index)
        .collect();
    let expected_indexes: HashSet<_> = vector
        .expected_frames
        .iter()
        .map(|frame| frame.presentation_index)
        .collect();
    if sample_indexes.len() != vector.samples.len()
        || expected_indexes.len() != vector.expected_frames.len()
        || sample_indexes != expected_indexes
    {
        return Err(invalid(
            "decoder conformance frame identities must be unique and complete",
        ));
    }
    Ok(())
}

fn validate_encoder_vector(vector: &VideoEncoderConformanceVector) -> Result<()> {
    if vector.name.trim().is_empty() || vector.frames.is_empty() {
        return Err(invalid(
            "encoder conformance vectors require a name and frames",
        ));
    }
    if !vector.minimum_psnr_db.is_finite() || vector.minimum_psnr_db < 0.0 {
        return Err(invalid(
            "encoder conformance PSNR must be finite and nonnegative",
        ));
    }
    if vector.configuration.timescale == 0 || vector.configuration.frame_duration == 0 {
        return Err(invalid(
            "encoder conformance requires a nonzero timescale and frame duration",
        ));
    }
    if vector.decoder_configuration.codec != vector.configuration.codec
        || vector.decoder_configuration.coded_dimensions != vector.configuration.coded_dimensions
        || vector.decoder_configuration.output_format != vector.configuration.input_format
        || vector.decoder_configuration.color_range != vector.configuration.color_range
    {
        return Err(invalid(
            "encoder and decoder conformance configurations must describe the same codec and frame format",
        ));
    }
    for frame in &vector.frames {
        if frame.dimensions != vector.configuration.coded_dimensions
            || frame.pixel_format != vector.configuration.input_format
            || frame.color_range != vector.configuration.color_range
        {
            return Err(invalid("encoder conformance source frame format changed"));
        }
    }
    Ok(())
}

fn require_supported(support: CodecSupport, message: &str) -> Result<()> {
    if support.is_supported() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::Unsupported,
            format!("{message}: {support:?}"),
        ))
    }
}

fn validate_encoded_packets(
    packets: &[EncodedSample],
    frame_count: usize,
    configuration: &crate::EncoderConfig,
    vector: &VideoEncoderConformanceVector,
) -> Result<()> {
    if configuration.codec != vector.configuration.codec
        || configuration.timescale != vector.configuration.timescale
        || configuration.timescale == 0
    {
        return Err(Error::new(
            ErrorKind::Codec,
            "encoder emitted invalid codec configuration",
        ));
    }
    if packets.len() != frame_count {
        return Err(Error::new(
            ErrorKind::Codec,
            "encoder must emit exactly one video access unit per conformance frame",
        ));
    }
    if packets.first().is_none_or(|packet| !packet.is_sync) {
        return Err(Error::new(
            ErrorKind::Codec,
            "encoder conformance output must begin with a random-access sample",
        ));
    }
    let mut pts = HashSet::new();
    for (decode_index, packet) in packets.iter().enumerate() {
        let expected_dts = i64::try_from(decode_index)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(vector.configuration.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "encoder DTS overflow"))?;
        if packet.data.is_empty()
            || packet.duration != vector.configuration.frame_duration
            || packet.dts != expected_dts
            || !pts.insert(packet.pts)
        {
            return Err(Error::new(
                ErrorKind::Codec,
                "encoder emitted invalid data, duration, DTS, or duplicate PTS",
            ));
        }
        let duration = i64::from(vector.configuration.frame_duration);
        if packet.pts < 0 || packet.pts % duration != 0 {
            return Err(Error::new(
                ErrorKind::Codec,
                "encoder sample PTS is outside the configured frame clock",
            ));
        }
    }
    for presentation_index in 0..frame_count {
        let expected_pts = i64::try_from(presentation_index)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(vector.configuration.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "encoder PTS overflow"))?;
        if !pts.contains(&expected_pts) {
            return Err(Error::new(
                ErrorKind::Codec,
                "encoder sample PTS values do not cover the input timeline",
            ));
        }
    }
    Ok(())
}

fn packets_to_video_samples(packets: Vec<EncodedSample>) -> Result<Vec<EncodedVideoSample>> {
    let mut presentation = (0..packets.len()).collect::<Vec<_>>();
    presentation.sort_by_key(|&index| (packets[index].pts, packets[index].dts, index));
    let mut indexes = vec![FrameIndex(0); packets.len()];
    for (presentation_index, decode_index) in presentation.into_iter().enumerate() {
        indexes[decode_index] =
            FrameIndex(u64::try_from(presentation_index).map_err(|_| {
                Error::new(ErrorKind::ResourceLimit, "presentation index overflow")
            })?);
    }
    Ok(packets
        .into_iter()
        .enumerate()
        .map(|(decode_index, packet)| EncodedVideoSample {
            presentation_index: indexes[decode_index],
            random_access: packet.is_sync,
            data: packet.data,
        })
        .collect())
}

fn frame_psnr(reference: &VideoFrame, candidate: &VideoFrame) -> Result<f64> {
    if reference.dimensions != candidate.dimensions
        || reference.pixel_format != candidate.pixel_format
        || reference.color_range != candidate.color_range
    {
        return Err(Error::new(
            ErrorKind::Codec,
            "round-trip decoder changed frame metadata",
        ));
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    append_active_pixels(reference, &mut a)?;
    append_active_pixels(candidate, &mut b)?;
    if a.len() != b.len() || a.is_empty() {
        return Err(Error::new(
            ErrorKind::Codec,
            "round-trip frame layout changed",
        ));
    }
    let squared_error: f64 = a
        .iter()
        .zip(&b)
        .map(|(&left, &right)| {
            let difference = f64::from(left) - f64::from(right);
            difference * difference
        })
        .sum();
    if squared_error == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = squared_error / a.len() as f64;
    Ok(10.0 * ((255.0 * 255.0) / mse).log10())
}

fn append_active_pixels(frame: &VideoFrame, output: &mut Vec<u8>) -> Result<()> {
    let layouts = plane_layouts(frame)?;
    for (plane, (row_bytes, rows)) in frame.planes.iter().zip(layouts) {
        for row in 0..rows {
            let start = row.checked_mul(plane.stride).ok_or_else(|| {
                Error::new(ErrorKind::ResourceLimit, "video plane offset overflow")
            })?;
            let end = start
                .checked_add(row_bytes)
                .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video plane row overflow"))?;
            let bytes = plane.data.get(start..end).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "video plane is shorter than its active rows",
                )
            })?;
            output.extend_from_slice(bytes);
        }
    }
    Ok(())
}

fn plane_layouts(frame: &VideoFrame) -> Result<Vec<(usize, usize)>> {
    let width = usize::try_from(frame.dimensions.width)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "video width is too large"))?;
    let height = usize::try_from(frame.dimensions.height)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "video height is too large"))?;
    let packed = |components: usize| {
        width
            .checked_mul(components)
            .map(|row_bytes| vec![(row_bytes, height)])
    };
    let layouts = match frame.pixel_format {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => packed(4),
        PixelFormat::Rgb8 => packed(3),
        PixelFormat::Gray8 => packed(1),
        PixelFormat::Yuv420p8 => {
            let chroma_width = width.div_ceil(2);
            let chroma_height = height.div_ceil(2);
            Some(vec![
                (width, height),
                (chroma_width, chroma_height),
                (chroma_width, chroma_height),
            ])
        }
    }
    .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video row size overflow"))?;
    if layouts.len() != frame.planes.len() {
        return Err(invalid("video plane count does not match its pixel format"));
    }
    Ok(layouts)
}

fn pixel_format_tag(format: PixelFormat) -> u8 {
    match format {
        PixelFormat::Rgba8 => 0,
        PixelFormat::Bgra8 => 1,
        PixelFormat::Rgb8 => 2,
        PixelFormat::Gray8 => 3,
        PixelFormat::Yuv420p8 => 4,
    }
}

fn add_statistics(left: DecodeStatistics, right: DecodeStatistics) -> DecodeStatistics {
    DecodeStatistics {
        samples_submitted: left
            .samples_submitted
            .saturating_add(right.samples_submitted),
        cache_hits: left.cache_hits.saturating_add(right.cache_hits),
        resets: left.resets.saturating_add(right.resets),
        drains: left.drains.saturating_add(right.drains),
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_length = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());
    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let upper = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let first = h
                .wrapping_add(upper)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let lower = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = lower.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        for (target, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *target = target.wrapping_add(value);
        }
    }
    let mut output = [0_u8; 32];
    for (chunk, value) in output.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Codec, CodecImplementation, CodecProfile, ColorRange, HardwarePreference, Plane,
        SampleDependency, VideoDimensions, VideoEncoder, VideoEncoderFormat,
        uncompressed_video_decoder_factory,
    };
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = Box::pin(future);
        loop {
            match Pin::new(&mut future).poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn config() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: Codec::UncompressedVideo,
            profile: CodecProfile::UncompressedGray8,
            coded_dimensions: VideoDimensions::new(2, 1, &Limits::default()).unwrap(),
            output_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        }
    }

    fn frame(value: u8) -> VideoFrame {
        VideoFrame::new(
            config().coded_dimensions,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: vec![value, value + 1],
                stride: 2,
            }],
            &Limits::default(),
        )
        .unwrap()
    }

    fn decoder_vector() -> VideoDecoderConformanceVector {
        let values = [10_u8, 30, 20];
        VideoDecoderConformanceVector {
            name: "uncompressed reorder".into(),
            configuration: config(),
            samples: values
                .iter()
                .enumerate()
                .map(|(decode_index, &value)| EncodedVideoSample {
                    presentation_index: FrameIndex([0, 2, 1][decode_index]),
                    random_access: decode_index == 0,
                    data: vec![value, value + 1],
                })
                .collect(),
            expected_frames: [10_u8, 20, 30]
                .iter()
                .enumerate()
                .map(|(index, &value)| ExpectedVideoFrame {
                    presentation_index: FrameIndex(index as u64),
                    digest: FrameDigest::from_frame(&frame(value)).unwrap(),
                })
                .collect(),
        }
    }

    #[test]
    fn sha256_matches_published_empty_string_vector() {
        assert_eq!(
            FrameDigest(sha256(b"")).to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn frame_digest_hex_round_trips_and_rejects_bad_input() {
        let digest = FrameDigest::from_frame(&frame(11)).unwrap();
        assert_eq!(FrameDigest::from_hex(&digest.to_hex()).unwrap(), digest);
        assert_eq!(
            FrameDigest::from_hex("xyz").unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn frame_digest_ignores_stride_padding() {
        let mut padded = frame(7);
        padded.planes[0] = Plane {
            data: vec![7, 8, 99, 99],
            stride: 4,
        };
        assert_eq!(
            FrameDigest::from_frame(&frame(7)).unwrap(),
            FrameDigest::from_frame(&padded).unwrap()
        );
    }

    #[test]
    fn decoder_runner_checks_three_access_patterns() {
        let report = verify_video_decoder_conformance(
            &uncompressed_video_decoder_factory(),
            &decoder_vector(),
            Limits::default(),
        )
        .unwrap();
        assert_eq!(report.frames_verified, 9);
        assert_eq!(report.access_patterns_verified, 3);
        assert!(report.statistics.resets > 0);
    }

    struct PassThroughEncoder {
        config: crate::EncoderConfig,
        format: VideoEncoderFormat,
        frame_duration: u32,
    }

    impl VideoEncoder for PassThroughEncoder {
        fn config(&self) -> &crate::EncoderConfig {
            &self.config
        }

        fn format(&self) -> VideoEncoderFormat {
            self.format
        }

        fn encode<'a>(
            &'a mut self,
            index: FrameIndex,
            source: FrameSource<'a>,
        ) -> crate::EncoderFuture<'a, Vec<EncodedSample>> {
            Box::pin(async move {
                let FrameSource::Cpu(source) = source else {
                    return Err(invalid("fixture requires CPU frames"));
                };
                Ok(vec![EncodedSample {
                    data: source.frame.planes[0].data.clone(),
                    dts: i64::try_from(index.0)
                        .map_err(|_| invalid("fixture timestamp overflow"))?
                        * i64::from(self.frame_duration),
                    pts: i64::try_from(index.0)
                        .map_err(|_| invalid("fixture timestamp overflow"))?
                        * i64::from(self.frame_duration),
                    duration: self.frame_duration,
                    is_sync: true,
                    dependency: SampleDependency::INDEPENDENT,
                }])
            })
        }

        fn finish<'a>(&'a mut self) -> crate::EncoderFuture<'a, Vec<EncodedSample>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    struct PassThroughEncoderFactory;

    impl VideoEncoderFactory for PassThroughEncoderFactory {
        fn capability(&self, configuration: &VideoEncoderConfig) -> CodecSupport {
            if configuration.codec == Codec::UncompressedVideo {
                CodecSupport::Supported {
                    implementation: CodecImplementation::Software,
                }
            } else {
                CodecSupport::UnsupportedCodec
            }
        }

        fn create(
            &self,
            configuration: &VideoEncoderConfig,
            _limits: &Limits,
        ) -> Result<Box<dyn VideoEncoder>> {
            Ok(Box::new(PassThroughEncoder {
                config: crate::EncoderConfig {
                    codec: Codec::UncompressedVideo,
                    timescale: configuration.timescale,
                    decoder_config: Vec::new(),
                },
                format: VideoEncoderFormat {
                    dimensions: configuration.coded_dimensions,
                    pixel_format: configuration.input_format,
                },
                frame_duration: configuration.frame_duration,
            }))
        }
    }

    #[test]
    fn encoder_runner_validates_independent_round_trip() {
        let decoder = config();
        let vector = VideoEncoderConformanceVector {
            name: "pass through".into(),
            configuration: VideoEncoderConfig {
                codec: Codec::UncompressedVideo,
                profile: CodecProfile::UncompressedGray8,
                coded_dimensions: decoder.coded_dimensions,
                input_format: PixelFormat::Gray8,
                color_range: ColorRange::Full,
                hardware: HardwarePreference::Avoid,
                timescale: 1,
                frame_duration: 1,
                configuration: Vec::new(),
            },
            decoder_configuration: decoder,
            frames: vec![frame(1), frame(3)],
            minimum_psnr_db: 60.0,
        };
        let report = block_on(verify_video_encoder_conformance(
            &PassThroughEncoderFactory,
            &uncompressed_video_decoder_factory(),
            &vector,
            Limits::default(),
        ))
        .unwrap();
        assert_eq!(report.frames_encoded, 2);
        assert!(report.minimum_observed_psnr_db.is_infinite());
    }
}
