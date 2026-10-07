//! Tests of the VP9 decoder against libvpx, the reference decoder.

use super::*;
use crate::ErrorKind;
use zvidlib_vp9_syntax::{ChunkInspector, FrameShape, chunk_full_range};

/// A minimal MD5 (RFC 1321), for comparing decoded frames with the
/// per-frame digests libvpx's test vectors ship with.
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
        .collect();
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for chunk in message.chunks(64) {
        let m: Vec<u32> = chunk
            .chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(k[i])
                .wrapping_add(m[g])
                .rotate_left(S[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }
    let mut digest = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    digest
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The MD5 libvpx's test harness computes for a picture: the visible
/// rows of Y, then U, then V.
pub fn picture_md5(picture: &DecodedPicture) -> String {
    let mut bytes = Vec::new();
    for plane in &picture.planes {
        bytes.extend_from_slice(plane);
    }
    hex(&md5(&bytes))
}

/// The frames of an IVF file.
pub fn ivf_frames(data: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&data[..4], b"DKIF", "not an IVF file");
    let header_size = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut offset = header_size;
    let mut frames = Vec::new();
    while offset + 12 <= data.len() {
        let size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 12;
        frames.push(&data[offset..offset + size]);
        offset += size;
    }
    frames
}

#[test]
fn md5_matches_known_digests() {
    assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(
        hex(&md5(b"The quick brown fox jumps over the lazy dog")),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
}

#[test]
fn superframe_index_lists_its_frames() {
    // Two frames of 3 and 2 bytes, then a one-byte-magnitude index.
    let marker = 0b1100_0001;
    let data = [1, 2, 3, 4, 5, marker, 3, 2, marker];
    assert_eq!(superframe_index(&data).unwrap(), Some(vec![3, 2]));
    assert_eq!(superframe_index(&[1, 2, 3]).unwrap(), None);
    // A marker without its matching first byte is not an index.
    assert!(superframe_index(&[1, 2, 3, 4, marker]).is_err());
}

#[test]
fn inverse_probability_remapping_table_matches_libvpx() {
    assert_eq!(
        &INV_MAP_TABLE[..21],
        &[
            7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241,
            254, 1
        ]
    );
    assert_eq!(&INV_MAP_TABLE[248..], &[248, 249, 250, 251, 252, 253, 253]);
}

/// The frames of a WebM file's first video track, in file order: the
/// `SimpleBlock`s and `BlockGroup` `Block`s of every `Cluster`. Only what
/// the libvpx test vectors use is read; lacing is refused.
pub fn webm_frames(data: &[u8]) -> Vec<&[u8]> {
    fn vint(data: &[u8], offset: usize, keep_marker: bool) -> (u64, usize) {
        let first = data[offset];
        let length = first.leading_zeros() as usize + 1;
        assert!(length <= 8, "invalid EBML variable-length integer");
        let mut value = if keep_marker {
            u64::from(first)
        } else {
            u64::from(first) & ((1u64 << (8 - length)) - 1)
        };
        for &byte in &data[offset + 1..offset + length] {
            value = (value << 8) | u64::from(byte);
        }
        (value, length)
    }
    fn walk<'a>(data: &'a [u8], mut offset: usize, end: usize, frames: &mut Vec<&'a [u8]>) {
        while offset < end {
            let (id, id_length) = vint(data, offset, true);
            let (size, size_length) = vint(data, offset + id_length, false);
            let body = offset + id_length + size_length;
            let unknown = size == (1u64 << (7 * size_length)) - 1;
            let body_end = if unknown {
                end
            } else {
                (body + size as usize).min(end)
            };
            match id {
                // Segment, Cluster and BlockGroup hold the blocks.
                0x1853_8067 | 0x1F43_B675 | 0xA0 => walk(data, body, body_end, frames),
                // SimpleBlock and Block.
                0xA3 | 0xA1 => {
                    let (track, track_length) = vint(data, body, false);
                    let flags = data[body + track_length + 2];
                    assert_eq!(flags & 0x06, 0, "laced WebM blocks are not supported");
                    if track == 1 {
                        frames.push(&data[body + track_length + 3..body_end]);
                    }
                }
                _ => {}
            }
            offset = body_end;
        }
    }
    let mut frames = Vec::new();
    walk(data, 0, data.len(), &mut frames);
    frames
}

/// The libvpx VP9 profile 0 test vectors, listed in
/// `crates/zvidlib-vp9-decoder/tests/fixtures/libvpx_vp9_test_vectors.txt` (the profile 0 part of
/// `test/test_vectors.cc`).
const LIBVPX_VECTORS: &str = include_str!("../../tests/fixtures/libvpx_vp9_test_vectors.txt");

/// Decodes every libvpx VP9 profile 0 test vector in the directory named by
/// `ZVIDLIB_VP9_VECTORS` and compares each shown frame with the per-frame
/// MD5 libvpx's own test harness checks (`<vector>.md5` beside the vector).
/// CI downloads the vectors from the WebM project and runs this with
/// `--ignored`. `ZVIDLIB_VP9_VECTOR_FILTER` restricts a run to the vectors
/// whose names contain it.
#[test]
#[ignore = "needs the libvpx test vectors; set ZVIDLIB_VP9_VECTORS"]
fn libvpx_test_vectors() {
    let directory = std::path::PathBuf::from(
        std::env::var("ZVIDLIB_VP9_VECTORS")
            .expect("ZVIDLIB_VP9_VECTORS names the directory holding the libvpx test vectors"),
    );
    let filter = std::env::var("ZVIDLIB_VP9_VECTOR_FILTER").unwrap_or_default();
    let names: Vec<&str> = LIBVPX_VECTORS
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty() && name.contains(&filter))
        .collect();
    assert!(!names.is_empty(), "no test vectors match the filter");
    let mut failures = Vec::new();
    for name in &names {
        let path = directory.join(name);
        let data = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let expected: Vec<String> = std::fs::read_to_string(directory.join(format!("{name}.md5")))
            .unwrap_or_else(|error| panic!("cannot read the digests of {name}: {error}"))
            .lines()
            .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
            .collect();
        let chunks = if name.ends_with(".ivf") {
            ivf_frames(&data)
        } else {
            webm_frames(&data)
        };
        let mut decoder = Decoder::new(Limits {
            max_width: 65536,
            max_height: 65536,
            max_allocation_bytes: 1 << 32,
            ..Limits::default()
        });
        let mut inspector = ChunkInspector::default();
        let mut shown = 0usize;
        let mut failure = None;
        for (index, chunk) in chunks.into_iter().enumerate() {
            let shape = inspector.inspect(chunk);
            match decoder.decode_chunk(chunk) {
                Ok(Some(picture)) => {
                    if expected.get(shown) != Some(&picture_md5(&picture)) {
                        failure = Some(format!("frame {shown} (chunk {index}) differs"));
                    } else if shape.as_ref().ok().copied().flatten() != Some(shape_of(&picture)) {
                        failure = Some(format!("chunk {index} is inspected as {shape:?}"));
                    }
                    shown += 1;
                }
                Ok(None) => {
                    if !matches!(shape, Ok(None)) {
                        failure = Some(format!("chunk {index} is inspected as {shape:?}"));
                    }
                }
                Err(error) => failure = Some(format!("chunk {index}: {error}")),
            }
            if failure.is_some() {
                break;
            }
        }
        if failure.is_none() && shown != expected.len() {
            failure = Some(format!("{shown} frames shown, {} expected", expected.len()));
        }
        match failure {
            Some(reason) => {
                eprintln!("FAIL {name}: {reason}");
                failures.push(*name);
            }
            None => eprintln!("ok   {name} ({shown} frames)"),
        }
    }
    eprintln!(
        "{} of {} vectors match libvpx",
        names.len() - failures.len(),
        names.len()
    );
    assert!(failures.is_empty(), "mismatching vectors: {failures:?}");
}

const BBB_256X144: &[u8] = include_bytes!("../../tests/fixtures/vp9_bbb_256x144.mp4");
const BBB_250X142: &[u8] = include_bytes!("../../tests/fixtures/vp9_bbb_250x142.mp4");

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

fn mp4_samples(mp4: &[u8]) -> Vec<crate::EncodedVideoSample> {
    let source = crate::io::MemorySource::new(mp4.to_vec());
    let movie = block_on(crate::Mp4Demuxer::open(
        &source,
        crate::Mp4DemuxerOptions::default(),
    ))
    .unwrap();
    let track = movie.track(1).unwrap();
    assert_eq!(track.codec, crate::Codec::Vp9);
    block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap()
}

fn digests(text: &str) -> Vec<crate::FrameDigest> {
    text.lines()
        .map(|line| crate::FrameDigest::from_hex(line.split_once(' ').unwrap().1).unwrap())
        .collect()
}

fn yuv_digest(picture: &DecodedPicture) -> crate::FrameDigest {
    let limits = Limits::default();
    let chroma_width = picture.width.div_ceil(2);
    let frame = crate::VideoFrame::new(
        crate::VideoDimensions::new(picture.width as u32, picture.height as u32, &limits).unwrap(),
        crate::PixelFormat::Yuv420p8,
        if picture.full_range {
            crate::ColorRange::Full
        } else {
            crate::ColorRange::Limited
        },
        picture
            .planes
            .iter()
            .enumerate()
            .map(|(index, plane)| crate::Plane {
                data: plane.clone(),
                stride: if index == 0 {
                    picture.width
                } else {
                    chroma_width
                },
            })
            .collect(),
        &limits,
    )
    .unwrap();
    crate::FrameDigest::from_frame(&frame).unwrap()
}

fn shape_of(picture: &DecodedPicture) -> FrameShape {
    FrameShape {
        width: picture.width,
        height: picture.height,
        color_space: picture.color_space,
        full_range: picture.full_range,
    }
}

/// Both MP4 fixtures, encoded by libvpx from the bundled sample, decode to
/// exactly the frames libvpx decodes them to. The 256x144 one is a two-pass
/// encode whose superframes carry hidden alternate reference frames, which
/// is checked too so a regenerated fixture cannot quietly lose them; the
/// 250x142 one exercises frame edges that are not a multiple of 8. See
/// `tests/fixtures/codec/README.md`.
#[test]
fn mp4_fixtures_decode_bit_exactly_against_libvpx() {
    for (mp4, expected, size, frames) in [
        (
            BBB_256X144,
            include_str!("../../tests/fixtures/vp9_bbb_256x144_yuv420.sha256"),
            (256, 144),
            48,
        ),
        (
            BBB_250X142,
            include_str!("../../tests/fixtures/vp9_bbb_250x142_yuv420.sha256"),
            (250, 142),
            12,
        ),
    ] {
        let expected = digests(expected);
        let samples = mp4_samples(mp4);
        assert_eq!((samples.len(), expected.len()), (frames, frames));
        let mut decoder = Decoder::new(Limits::default());
        let mut inspector = ChunkInspector::default();
        let mut superframes_with_hidden_frames = 0;
        for (index, sample) in samples.iter().enumerate() {
            if superframe_index(&sample.data)
                .unwrap()
                .is_some_and(|sizes| sizes.len() > 1)
            {
                superframes_with_hidden_frames += 1;
            }
            let shape = inspector.inspect(&sample.data).unwrap();
            let picture = decoder.decode_chunk(&sample.data).unwrap().unwrap();
            assert_eq!(shape, Some(shape_of(&picture)), "frame {index}");
            assert_eq!((picture.width, picture.height), size);
            assert!(!picture.full_range);
            assert_eq!(yuv_digest(&picture), expected[index], "frame {index}");
        }
        if size == (256, 144) {
            assert_eq!(superframes_with_hidden_frames, 4);
        }
        assert_eq!(decoder.frames_shown(), frames as u64);
    }
}

/// The same fixtures decode to libvpx's frames under every instruction set
/// the host supports, scalar included, through `crate::simd`'s override: the
/// test above only exercises the widest one. A vector kernel that diverged
/// from the scalar reference anywhere in a real decode fails here (#570).
#[test]
fn mp4_fixtures_decode_bit_exactly_under_every_instruction_set() {
    let _guard = crate::simd::test_lock();
    for (mp4, expected) in [
        (
            BBB_256X144,
            include_str!("../../tests/fixtures/vp9_bbb_256x144_yuv420.sha256"),
        ),
        (
            BBB_250X142,
            include_str!("../../tests/fixtures/vp9_bbb_250x142_yuv420.sha256"),
        ),
    ] {
        let expected = digests(expected);
        let samples = mp4_samples(mp4);
        for isa in crate::simd::available() {
            crate::simd::set_override(Some(isa));
            assert_eq!(crate::vp9_simd::active_isa(), isa);
            let mut decoder = Decoder::new(Limits::default());
            for (index, sample) in samples.iter().enumerate() {
                let picture = decoder.decode_chunk(&sample.data).unwrap().unwrap();
                assert_eq!(
                    yuv_digest(&picture),
                    expected[index],
                    "{} frame {index}",
                    isa.name()
                );
            }
        }
    }
    crate::simd::set_override(None);
}

#[test]
fn an_inter_frame_after_reset_is_refused_without_panicking() {
    let samples = mp4_samples(BBB_256X144);
    let mut decoder = Decoder::new(Limits::default());
    assert!(decoder.decode_chunk(&samples[0].data).unwrap().is_some());
    decoder.reset();
    let error = decoder.decode_chunk(&samples[2].data).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    // The key frame restarts decoding cleanly.
    assert!(decoder.decode_chunk(&samples[0].data).unwrap().is_some());
}

#[test]
fn show_existing_frame_shows_a_reference_again() {
    let samples = mp4_samples(BBB_256X144);
    let mut decoder = Decoder::new(Limits::default());
    let key = decoder.decode_chunk(&samples[0].data).unwrap().unwrap();
    let next = decoder.decode_chunk(&samples[1].data).unwrap().unwrap();
    assert_ne!(key.planes, next.planes);
    // A key frame refreshes every slot; slot 7 is still the key frame here
    // unless the next frame refreshed it, so pick a slot by looking.
    let shown_before = decoder.frames_shown();
    let mut found = false;
    for slot in 0..8u8 {
        // frame_marker 2, profile 0, show_existing_frame 1, then the slot.
        let picture = decoder.decode_chunk(&[0x88 | slot]).unwrap().unwrap();
        found |= picture.planes == key.planes;
    }
    assert!(found, "some reference slot still holds the key frame");
    assert_eq!(decoder.frames_shown(), shown_before + 8);
    // Output that is not wanted is still counted.
    decoder.set_output_wanted(false);
    assert!(decoder.decode_chunk(&[0x88]).unwrap().is_none());
    assert_eq!(decoder.frames_shown(), shown_before + 9);
}

#[test]
fn the_inspector_follows_show_existing_frame_and_refuses_what_the_decoder_refuses() {
    let samples = mp4_samples(BBB_256X144);
    let mut inspector = ChunkInspector::default();
    // An inter frame before any key frame.
    let error = inspector.inspect(&samples[2].data).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    let key = inspector.inspect(&samples[0].data).unwrap().unwrap();
    assert_eq!((key.width, key.height), (256, 144));
    // A key frame refreshes every slot, so every slot shows its shape.
    for slot in 0..8u8 {
        assert_eq!(inspector.inspect(&[0x88 | slot]).unwrap(), Some(key));
    }
    for data in [&[][..], &[0x00, 0x00], &[0x90, 0x00]] {
        assert!(inspector.inspect(data).is_err(), "{data:?}");
    }
    inspector.reset();
    let error = inspector.inspect(&[0x88]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);
}

#[test]
fn damaged_samples_are_rejected_or_decoded_without_panicking() {
    let samples = mp4_samples(BBB_256X144);
    for index in [0usize, 1, 2] {
        let data = &samples[index].data;
        for cut in [
            1,
            2,
            4,
            9,
            16,
            data.len() / 3,
            data.len() / 2,
            data.len() - 1,
        ] {
            let mut decoder = Decoder::new(Limits::default());
            for previous in &samples[..index] {
                decoder.decode_chunk(&previous.data).unwrap();
            }
            let _ = decoder.decode_chunk(&data[..cut]);
        }
        let step = (data.len() / 211).max(1);
        for position in (0..data.len()).step_by(step) {
            for pattern in [0x5a, 0xff, 0x01] {
                let mut decoder = Decoder::new(Limits::default());
                for previous in &samples[..index] {
                    decoder.decode_chunk(&previous.data).unwrap();
                }
                let mut damaged = data.clone();
                damaged[position] ^= pattern;
                let _ = decoder.decode_chunk(&damaged);
                // Whatever the damaged frame did, the decoder keeps working.
                let _ = decoder.decode_chunk(&samples[index + 1].data);
            }
        }
    }
}

#[test]
fn oversized_frames_are_refused_before_allocation() {
    let samples = mp4_samples(BBB_256X144);
    let mut decoder = Decoder::new(Limits {
        max_width: 128,
        ..Limits::default()
    });
    let error = decoder.decode_chunk(&samples[0].data).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
}

#[test]
fn the_colour_range_of_a_key_frame_is_read_without_decoding_it() {
    let samples = mp4_samples(BBB_256X144);
    assert_eq!(chunk_full_range(&samples[0].data), Some(false));
    // An inter frame, a superframe that starts with one, and junk say nothing.
    assert_eq!(chunk_full_range(&samples[1].data), None);
    assert_eq!(chunk_full_range(&samples[2].data), None);
    assert_eq!(chunk_full_range(&[]), None);
    assert_eq!(chunk_full_range(&[0xff]), None);
    // The same key frame with its color_range bit set. The bit follows the
    // frame marker byte, the 24-bit sync code and the 3-bit color_space.
    let mut full = samples[0].data.clone();
    full[4] |= 0x10;
    assert_eq!(chunk_full_range(&full), Some(true));
}

/// The 256x144 fixture remuxed to WebM (`V_VP9`, no `CodecPrivate`): the
/// WebM demuxer indexes the same 48 chunks, describes the track with the
/// `vpcC` box an MP4 one would carry, and they decode to libvpx's frames.
#[test]
fn webm_vp9_track_decodes_like_its_mp4_original() {
    let source = crate::io::MemorySource::new(
        include_bytes!("../../tests/fixtures/vp9_bbb_256x144.webm").to_vec(),
    );
    let tracks = block_on(crate::container::open_tracks(&source, &Limits::default())).unwrap();
    let track = &tracks[0];
    assert_eq!(track.codec, crate::Codec::Vp9);
    assert_eq!(&track.decoder_config[4..8], b"vpcC");
    let config = crate::Vp9CodecConfig::parse(&track.decoder_config).unwrap();
    assert_eq!((config.profile, config.bit_depth), (0, 8));
    let samples = block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap();
    let mp4 = mp4_samples(BBB_256X144);
    assert_eq!(samples.len(), mp4.len());
    let expected = digests(include_str!(
        "../../tests/fixtures/vp9_bbb_256x144_yuv420.sha256"
    ));
    let mut decoder = Decoder::new(Limits::default());
    for (index, (sample, original)) in samples.iter().zip(&mp4).enumerate() {
        assert_eq!(sample.data, original.data, "chunk {index}");
        assert_eq!(
            sample.random_access, original.random_access,
            "chunk {index}"
        );
        let picture = decoder.decode_chunk(&sample.data).unwrap().unwrap();
        assert_eq!(yuv_digest(&picture), expected[index], "frame {index}");
    }
}
