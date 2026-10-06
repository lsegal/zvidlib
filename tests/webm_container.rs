//! WebM round trips through `WebmMuxer` and `WebmDemuxer` (issue #526).
//!
//! The bundled AV1 sample is remuxed from MP4 into WebM, and the WebM index
//! must hand a decoder exactly the samples the MP4 index does, so any frame
//! decodes to the same picture from either container. The ffprobe and ffmpeg
//! checks skip themselves where those tools are not installed.

use std::future::Future;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, Container, EncodedSample, EncoderConfig,
    ErrorKind, ExactFrameReader, FrameIndex, HardwarePreference, Limits, Mp4Demuxer,
    Mp4DemuxerOptions, Mp4Track, PixelFormat, SampleDependency, TrackKind, VideoDecoderConfig,
    VideoDimensions, WebmDemuxer, WebmDemuxerOptions, WebmMuxer, container_capabilities,
    native_av1_video_decoder_factory, probe_container,
};

const AV1_MP4: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = Pin::new(&mut future).poll(&mut context) {
            return value;
        }
    }
}

fn mp4_video_track() -> (MemorySource, Mp4Track) {
    let source = MemorySource::new(AV1_MP4.to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = movie
        .tracks
        .into_iter()
        .find(|track| track.kind == TrackKind::Video)
        .unwrap();
    (source, track)
}

/// The bundled sample's video track, remuxed sample for sample into WebM.
fn remux_to_webm() -> Vec<u8> {
    let (source, track) = mp4_video_track();
    let config = Mp4TrackConfig {
        encoder: EncoderConfig {
            codec: track.codec,
            timescale: track.timescale,
            decoder_config: track.decoder_config.clone(),
        },
        format: Mp4TrackFormat::Video(track.dimensions.unwrap()),
    };
    block_on(async {
        let mut muxer = WebmMuxer::new(MemorySink::new(), vec![config], 1_000_000)
            .await
            .unwrap();
        for (index, sample) in track.samples.iter().enumerate() {
            let mut data = vec![0; sample.size as usize];
            track
                .read_sample_into(&source, index, &mut data)
                .await
                .unwrap();
            muxer
                .write_sample(
                    0,
                    EncodedSample {
                        data,
                        dts: i64::try_from(sample.dts).unwrap(),
                        pts: sample.pts,
                        duration: sample.duration,
                        is_sync: sample.is_sync,
                        dependency: sample.dependency,
                    },
                )
                .await
                .unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    })
}

fn demux(bytes: Vec<u8>) -> (MemorySource, WebmDemuxer) {
    let source = MemorySource::new(bytes);
    let demuxer = block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
    (source, demuxer)
}

fn decoder_config(track: &Mp4Track) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Av1,
        profile: CodecProfile::Av1Main,
        coded_dimensions: track.dimensions.unwrap(),
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: track.decoder_config.clone(),
    }
}

#[test]
fn a_remuxed_webm_indexes_exactly_the_mp4_samples() {
    let (mp4_source, mp4_track) = mp4_video_track();
    let (webm_source, demuxer) = demux(remux_to_webm());
    assert_eq!(demuxer.doc_type, "webm");
    assert_eq!(demuxer.tracks.len(), 1);
    let track = &demuxer.tracks[0];
    assert_eq!(track.id, 1);
    assert_eq!(track.kind, TrackKind::Video);
    assert_eq!(track.codec, Codec::Av1);
    assert_eq!(track.dimensions, mp4_track.dimensions);
    assert_eq!(track.decoder_config, mp4_track.decoder_config);
    assert_eq!(track.timescale, 1_000);

    let limits = Limits::default();
    let expected = block_on(mp4_track.to_encoded_video_samples(&mp4_source, &limits)).unwrap();
    let actual = block_on(track.to_encoded_video_samples(&webm_source, &limits)).unwrap();
    assert_eq!(actual, expected);

    // Presentation times survive at the millisecond WebM stores them in.
    for (webm, mp4) in track.samples.iter().zip(&mp4_track.samples) {
        let expected_ms = (mp4.pts as f64 * 1_000.0 / f64::from(mp4_track.timescale)).round();
        assert_eq!(webm.pts as f64, expected_ms);
    }
    let duration = demuxer.duration_seconds.unwrap();
    let expected_duration = mp4_track.duration as f64 / f64::from(mp4_track.timescale);
    assert!((duration - expected_duration).abs() < 1e-3);
}

#[test]
fn any_frame_of_a_webm_decodes_to_the_mp4_picture() {
    let (mp4_source, mp4_track) = mp4_video_track();
    let (webm_source, demuxer) = demux(remux_to_webm());
    let track = &demuxer.tracks[0];
    let limits = Limits::default();
    let factory = native_av1_video_decoder_factory();
    let mut from_mp4 = ExactFrameReader::new(
        &factory,
        decoder_config(&mp4_track),
        block_on(mp4_track.to_encoded_video_samples(&mp4_source, &limits)).unwrap(),
        limits,
    )
    .unwrap();
    let mut from_webm = ExactFrameReader::new(
        &factory,
        decoder_config(track),
        block_on(track.to_encoded_video_samples(&webm_source, &limits)).unwrap(),
        limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    // Out of order on purpose: a backward request restarts the decode. Frames
    // near the start keep the software decode short; the sample equality
    // above is what covers the rest of the track.
    for index in [3, 0, 5] {
        let index = FrameIndex(index);
        assert_eq!(
            from_webm.get(index, &cancellation).unwrap(),
            from_mp4.get(index, &cancellation).unwrap(),
            "frame {} differs between containers",
            index.0
        );
    }
}

#[test]
fn the_written_cues_are_seekable_cluster_starts() {
    let bytes = remux_to_webm();
    let (_, demuxer) = demux(bytes.clone());
    let track = &demuxer.tracks[0];
    let sync: Vec<_> = track
        .samples
        .iter()
        .filter(|sample| sample.is_sync)
        .collect();
    assert!(!sync.is_empty());
    assert_eq!(demuxer.cues.len(), sync.len());
    for (cue, sample) in demuxer.cues.iter().zip(&sync) {
        assert_eq!(cue.track, track.id);
        assert_eq!(cue.time, sample.pts);
        let offset = cue.cluster_offset as usize;
        assert_eq!(&bytes[offset..offset + 4], &[0x1F, 0x43, 0xB6, 0x75]);
    }
    let last = track.samples.last().unwrap();
    let seek = demuxer.seek_point(track.id, last.pts).unwrap();
    assert!(seek.from_cues);
    assert_eq!(seek.time, sync.last().unwrap().pts);
    assert_eq!(seek.offset, demuxer.cues.last().unwrap().cluster_offset);
}

#[test]
fn containers_are_detected_by_signature_and_reported() {
    let webm = MemorySource::new(remux_to_webm());
    let mp4 = MemorySource::new(AV1_MP4.to_vec());
    let neither = MemorySource::new(b"not a media file".to_vec());
    assert_eq!(
        block_on(probe_container(&webm)).unwrap(),
        Some(Container::WebM)
    );
    assert_eq!(
        block_on(probe_container(&mp4)).unwrap(),
        Some(Container::Mp4)
    );
    assert_eq!(block_on(probe_container(&neither)).unwrap(), None);

    let names: Vec<_> = container_capabilities()
        .into_iter()
        .map(|capability| {
            assert!(capability.support.is_available());
            capability.name
        })
        .collect();
    assert_eq!(names, ["mp4", "webm"]);
    assert_eq!(Container::WebM.mime_type(), "video/webm");
    assert_eq!(Container::from_name("WebM"), Some(Container::WebM));
}

#[test]
fn webm_output_refuses_aac_audio_and_non_webm_video() {
    let limits = Limits::default();
    let aac = Mp4TrackConfig {
        encoder: EncoderConfig {
            codec: Codec::Aac,
            timescale: 48_000,
            decoder_config: vec![0, 0, 0, 8, b'e', b's', b'd', b's'],
        },
        format: Mp4TrackFormat::Audio { channels: 2 },
    };
    let error = block_on(WebmMuxer::new(MemorySink::new(), vec![aac], 16))
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("AAC"), "{}", error.message());

    let hevc = Mp4TrackConfig {
        encoder: EncoderConfig {
            codec: Codec::Hevc,
            timescale: 30,
            decoder_config: vec![0, 0, 0, 8, b'h', b'v', b'c', b'C'],
        },
        format: Mp4TrackFormat::Video(VideoDimensions::new(16, 16, &limits).unwrap()),
    };
    let error = block_on(WebmMuxer::new(MemorySink::new(), vec![hevc], 16))
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[test]
fn samples_must_arrive_in_presentation_order_across_tracks() {
    let limits = Limits::default();
    let track = Mp4TrackConfig {
        encoder: EncoderConfig {
            codec: Codec::Av1,
            timescale: 1_000,
            decoder_config: vec![0, 0, 0, 12, b'a', b'v', b'1', b'C', 0x81, 0, 0, 0],
        },
        format: Mp4TrackFormat::Video(VideoDimensions::new(16, 16, &limits).unwrap()),
    };
    let sample = |pts: i64| EncodedSample {
        data: vec![0x12, 0],
        dts: pts,
        pts,
        duration: 10,
        is_sync: true,
        dependency: SampleDependency::INDEPENDENT,
    };
    block_on(async {
        let mut muxer = WebmMuxer::new(MemorySink::new(), vec![track.clone(), track], 16)
            .await
            .unwrap();
        muxer.write_sample(0, sample(0)).await.unwrap();
        muxer.write_sample(1, sample(0)).await.unwrap();
        muxer.write_sample(0, sample(10)).await.unwrap();
        let error = muxer.write_sample(1, sample(5)).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        let bytes = muxer.finish().await.unwrap().into_inner();
        let (_, demuxer) = demux(bytes);
        assert_eq!(demuxer.tracks.len(), 2);
        assert_eq!(demuxer.tracks[0].samples.len(), 2);
        assert_eq!(demuxer.tracks[1].samples.len(), 1);
    });
}

fn ffmpeg_available() -> bool {
    ["ffmpeg", "ffprobe"].into_iter().all(|tool| {
        Command::new(tool)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

fn framemd5(path: &std::path::Path) -> Vec<String> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-f", "framemd5", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.rsplit(',').next().unwrap().trim().to_owned())
        .collect()
}

/// An independent demuxer reads the WebM as seekable AV1 with Cues and
/// decodes every frame to the same picture it decodes from the MP4.
#[test]
fn ffmpeg_reads_the_webm_as_the_same_seekable_av1() {
    if !ffmpeg_available() {
        eprintln!("skipping the independent check because ffmpeg is unavailable");
        return;
    }
    let directory = std::env::temp_dir();
    let webm = directory.join(format!("zvidlib-webm-{}.webm", std::process::id()));
    let mp4 = directory.join(format!("zvidlib-webm-{}.mp4", std::process::id()));
    std::fs::write(&webm, remux_to_webm()).unwrap();
    std::fs::write(&mp4, AV1_MP4).unwrap();

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name:stream=codec_name,width,height",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&webm)
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&probe.stdout).into_owned();
    // `-ss` before `-i` seeks through the container's index, which for a
    // Matroska file is its Cues.
    let seek = Command::new("ffmpeg")
        .args(["-v", "error", "-ss", "5", "-i"])
        .arg(&webm)
        .args(["-frames:v", "1", "-f", "null", "-"])
        .output()
        .unwrap();
    let from_webm = framemd5(&webm);
    let from_mp4 = framemd5(&mp4);
    let _ = std::fs::remove_file(&webm);
    let _ = std::fs::remove_file(&mp4);

    assert!(probe.status.success(), "ffprobe failed: {report}");
    assert!(report.contains("format_name=matroska,webm"), "{report}");
    assert!(report.contains("codec_name=av1"), "{report}");
    assert!(
        seek.status.success(),
        "ffmpeg could not seek: {}",
        String::from_utf8_lossy(&seek.stderr)
    );
    assert!(!from_webm.is_empty());
    assert_eq!(from_webm, from_mp4);
}
