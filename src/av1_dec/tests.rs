use super::*;
use crate::io::MemorySource;
use crate::{
    ColorRange, EncodedVideoSample, FrameDigest, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, Plane,
    VideoDimensions, VideoFrame,
};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

const COLOR_AV1: &[u8] = include_bytes!("../../examples/media/BigBuckBunny.av1.mp4");

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = Pin::new(&mut future).poll(&mut context) {
            return value;
        }
    }
}

fn samples() -> Vec<EncodedVideoSample> {
    samples_of(COLOR_AV1)
}

fn samples_of(mp4: &[u8]) -> Vec<EncodedVideoSample> {
    let source = MemorySource::new(mp4.to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = movie.track(1).unwrap();
    block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap()
}

fn yuv_digest(picture: &DecodedPicture) -> FrameDigest {
    let limits = Limits::default();
    let chroma_width = picture.width.div_ceil(2);
    let frame = VideoFrame::new(
        VideoDimensions::new(picture.width as u32, picture.height as u32, &limits).unwrap(),
        PixelFormat::Yuv420p8,
        if picture.full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
        vec![
            Plane {
                data: picture.planes[0].clone(),
                stride: picture.width,
            },
            Plane {
                data: picture.planes[1].clone(),
                stride: chroma_width,
            },
            Plane {
                data: picture.planes[2].clone(),
                stride: chroma_width,
            },
        ],
        &limits,
    )
    .unwrap();
    FrameDigest::from_frame(&frame).unwrap()
}

/// Every frame of the bundled SVT-AV1 colour sample (8-bit 4:2:0 Main, with
/// CDF adaptation, CDEF, loop restoration, warped motion, inter-intra and
/// reference frame motion vectors) decodes to exactly the samples an
/// independent decoder produces. The digests are of FFmpeg/libdav1d's decode;
/// see `tests/fixtures/codec/README.md`.
#[test]
fn colour_sample_decodes_bit_exactly_against_an_independent_decoder() {
    let expected: Vec<FrameDigest> =
        include_str!("../../tests/fixtures/codec/big_buck_bunny_av1_yuv420.sha256")
            .lines()
            .map(|line| FrameDigest::from_hex(line.split_once(' ').unwrap().1).unwrap())
            .collect();
    assert_eq!(expected.len(), 768);
    let mut decoder = Decoder::new(Limits::default());
    let mut shown = 0;
    for sample in samples() {
        for picture in decoder.decode_temporal_unit(&sample.data).unwrap() {
            assert_eq!((picture.width, picture.height), (960, 540));
            assert!(!picture.monochrome && !picture.full_range);
            assert_eq!(yuv_digest(&picture), expected[shown], "frame {shown}");
            shown += 1;
        }
    }
    assert_eq!(shown, 768);
}

#[test]
fn an_inter_frame_after_reset_is_refused_without_panicking() {
    let samples = samples();
    let mut decoder = Decoder::new(Limits::default());
    assert_eq!(
        decoder
            .decode_temporal_unit(&samples[0].data)
            .unwrap()
            .len(),
        1
    );
    decoder.reset();
    let error = decoder.decode_temporal_unit(&samples[1].data).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    // The key frame restarts decoding cleanly.
    assert_eq!(
        decoder
            .decode_temporal_unit(&samples[0].data)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn damaged_temporal_units_are_rejected_or_decoded_without_panicking() {
    let samples = samples();
    let key = &samples[0].data;
    for cut in [1, 4, 16, key.len() / 3, key.len() / 2, key.len() - 1] {
        let mut decoder = Decoder::new(Limits::default());
        let _ = decoder.decode_temporal_unit(&key[..cut]);
    }
    // Corrupt single bytes throughout the key frame and the frame after it;
    // each must decode to something or fail cleanly.
    for (index, sample) in samples.iter().take(2).enumerate() {
        let step = (sample.data.len() / 97).max(1);
        for position in (0..sample.data.len()).step_by(step) {
            let mut decoder = Decoder::new(Limits::default());
            if index > 0 {
                decoder.decode_temporal_unit(&samples[0].data).unwrap();
            }
            let mut damaged = sample.data.clone();
            damaged[position] ^= 0x5a;
            let _ = decoder.decode_temporal_unit(&damaged);
        }
    }
}

#[test]
fn limits_bound_frame_size_and_obu_count() {
    let samples = samples();
    let small = Limits {
        max_width: 320,
        ..Limits::default()
    };
    let error = Decoder::new(small)
        .decode_temporal_unit(&samples[0].data)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    let few_blocks = Limits {
        max_av1_blocks_per_frame: 1000,
        ..Limits::default()
    };
    let error = Decoder::new(few_blocks)
        .decode_temporal_unit(&samples[0].data)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    let one_obu = Limits {
        max_av1_obus_per_unit: 1,
        ..Limits::default()
    };
    let error = Decoder::new(one_obu)
        .decode_temporal_unit(&samples[0].data)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
}

#[test]
#[ignore]
fn probe() {
    let data = std::fs::read(std::env::var("AV1_PROBE").unwrap()).unwrap();
    let mut decoder = Decoder::new(Limits::default());
    let _ = coverage::take();
    let mut shown = 0;
    let mut out = String::new();
    for sample in samples_of(&data) {
        match decoder.decode_temporal_unit(&sample.data) {
            Ok(pictures) => for picture in pictures {
                out += &format!("{shown} {}
", yuv_digest(&picture).to_hex());
                shown += 1;
            },
            Err(e) => { out += &format!("ERR {e}
"); break; }
        }
    }
    out += &format!("TOOLS {:#x}
", coverage::take());
    std::fs::write(std::env::var("AV1_PROBE_OUT").unwrap(), out).unwrap();
}
