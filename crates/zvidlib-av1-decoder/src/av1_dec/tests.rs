use super::*;
use crate::io::MemorySource;
use crate::{
    ColorRange, EncodedVideoSample, FrameDigest, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, Plane,
    VideoDimensions, VideoFrame,
};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

const COLOR_AV1: &[u8] = include_bytes!("../../../../examples/media/BigBuckBunny.av1.mp4");

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

fn digests(text: &str) -> Vec<FrameDigest> {
    text.lines()
        .map(|line| FrameDigest::from_hex(line.split_once(' ').unwrap().1).unwrap())
        .collect()
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
    let expected = digests(include_str!(
        "../../tests/fixtures/big_buck_bunny_av1_yuv420.sha256"
    ));
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

/// A small libaom or SVT-AV1 fixture made to reach tools the colour sample
/// above never does, with the per-frame digests of FFmpeg/libdav1d's decode
/// of it (film grain applied) and the tools it must reach. See
/// `tests/fixtures/codec/README.md` for how each was made.
struct ToolFixture {
    name: &'static str,
    mp4: &'static [u8],
    digests: &'static str,
    size: (usize, usize),
    tools: u32,
}

macro_rules! tool_fixture {
    ($name:literal, $size:expr, $tools:expr) => {
        ToolFixture {
            name: $name,
            mp4: include_bytes!(concat!("../../tests/fixtures/", $name, ".mp4")),
            digests: include_str!(concat!(
                "../../tests/fixtures/",
                $name,
                "_yuv420.sha256"
            )),
            size: $size,
            tools: $tools,
        }
    };
}

const PALETTE_INTRABC: ToolFixture = tool_fixture!(
    "av1_palette_intrabc",
    (256, 144),
    coverage::PALETTE | coverage::INTRA_BLOCK_COPY
);
const FILM_GRAIN: ToolFixture = tool_fixture!("av1_film_grain", (256, 144), coverage::FILM_GRAIN);
const SUPERRES: ToolFixture = tool_fixture!("av1_superres", (256, 144), coverage::SUPERRES);
const QMATRIX: ToolFixture = tool_fixture!("av1_qmatrix", (256, 144), coverage::QUANTIZER_MATRIX);
const TILES: ToolFixture = tool_fixture!(
    "av1_tiles",
    (256, 144),
    coverage::MULTIPLE_TILES | coverage::MULTIPLE_TILE_GROUPS | coverage::CONTEXT_UPDATE_TILE_ID
);
const SB128: ToolFixture = tool_fixture!("av1_sb128", (250, 142), coverage::SUPERBLOCK_128);
const SEGMENTATION_DELTAS: ToolFixture = tool_fixture!(
    "av1_segmentation_deltas",
    (256, 144),
    coverage::SEGMENTATION | coverage::DELTA_Q | coverage::DELTA_LF
);
const COMPOUND_FILTERS: ToolFixture = tool_fixture!(
    "av1_compound_filters",
    (256, 144),
    coverage::FILTER_INTRA
        | coverage::WEDGE_COMPOUND
        | coverage::DIFF_WEIGHTED_COMPOUND
        | coverage::DISTANCE_WEIGHTED_COMPOUND
        | coverage::DUAL_FILTER
);

const TOOL_FIXTURES: [&ToolFixture; 8] = [
    &PALETTE_INTRABC,
    &FILM_GRAIN,
    &SUPERRES,
    &QMATRIX,
    &TILES,
    &SB128,
    &SEGMENTATION_DELTAS,
    &COMPOUND_FILTERS,
];

/// Decodes every frame of `fixture`, compares each with the independent
/// decoder's, and checks the decode reached the tools the fixture is for.
fn check_tool_fixture(fixture: &ToolFixture) {
    let name = fixture.name;
    let expected = digests(fixture.digests);
    let mut decoder = Decoder::new(Limits::default());
    coverage::take();
    let mut shown = 0;
    for sample in samples_of(fixture.mp4) {
        for picture in decoder.decode_temporal_unit(&sample.data).unwrap() {
            assert_eq!((picture.width, picture.height), fixture.size, "{name}");
            assert!(!picture.monochrome && !picture.full_range, "{name}");
            assert!(shown < expected.len(), "{name} shows too many frames");
            assert_eq!(
                yuv_digest(&picture),
                expected[shown],
                "{name} frame {shown}"
            );
            shown += 1;
        }
    }
    assert_eq!(shown, expected.len(), "{name}");
    let reached = coverage::take();
    assert_eq!(
        reached & fixture.tools,
        fixture.tools,
        "{name} reached tools {reached:#x}, not all of {:#x}",
        fixture.tools
    );
}

#[test]
fn palette_and_intra_block_copy_decode_bit_exactly() {
    check_tool_fixture(&PALETTE_INTRABC);
}

#[test]
fn film_grain_decodes_bit_exactly() {
    check_tool_fixture(&FILM_GRAIN);
}

#[test]
fn super_resolution_decodes_bit_exactly() {
    check_tool_fixture(&SUPERRES);
}

#[test]
fn quantizer_matrices_decode_bit_exactly() {
    check_tool_fixture(&QMATRIX);
}

#[test]
fn tiles_and_tile_groups_decode_bit_exactly() {
    check_tool_fixture(&TILES);
}

#[test]
fn superblocks_128x128_decode_bit_exactly() {
    check_tool_fixture(&SB128);
}

#[test]
fn segmentation_and_delta_q_and_loop_filter_decode_bit_exactly() {
    check_tool_fixture(&SEGMENTATION_DELTAS);
}

#[test]
fn filter_intra_masked_and_distance_compound_and_dual_filter_decode_bit_exactly() {
    check_tool_fixture(&COMPOUND_FILTERS);
}

/// Between them the fixtures are required to reach every tool `coverage`
/// tracks, so none is left without a reference check.
#[test]
fn tool_fixtures_cover_every_tracked_tool() {
    let required = TOOL_FIXTURES
        .iter()
        .fold(0, |tools, fixture| tools | fixture.tools);
    assert_eq!(required, coverage::ALL);
}
