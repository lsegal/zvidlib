//! Bit-exactness against libvpx on the VP8 test vectors.
//!
//! `tests/fixtures/codec/vp8/` holds libvpx's `vp80-00-comprehensive` vectors
//! together with the `.md5` files libvpx publishes beside them: one MD5 of
//! each shown frame's I420 output, cropped to the display size. See
//! `tests/fixtures/codec/README.md`.

use super::decoder::{Decoder, Picture};
use crate::Limits;

/// Splits an IVF file into its frames.
pub(crate) fn ivf_frames(file: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&file[0..4], b"DKIF", "not an IVF file");
    assert_eq!(&file[8..12], b"VP80", "not a VP8 IVF file");
    let header_length = usize::from(u16::from_le_bytes([file[6], file[7]]));
    let mut frames = Vec::new();
    let mut position = header_length;
    while position + 12 <= file.len() {
        let size = u32::from_le_bytes(file[position..position + 4].try_into().unwrap()) as usize;
        let start = position + 12;
        frames.push(&file[start..start + size]);
        position = start + size;
    }
    frames
}

/// RFC 1321 MD5, enough to compare against libvpx's published digests.
fn md5(input: &[u8]) -> [u8; 16] {
    const SHIFTS: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let constants: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
        .collect();
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((input.len() as u64).wrapping_mul(8)).to_le_bytes());
    let mut state = [0x6745_2301_u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for chunk in message.chunks(64) {
        let words: Vec<u32> = chunk
            .chunks(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
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
                .wrapping_add(constants[i])
                .wrapping_add(words[g])
                .rotate_left(SHIFTS[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        for (value, add) in state.iter_mut().zip([a, b, c, d]) {
            *value = value.wrapping_add(add);
        }
    }
    let mut digest = [0u8; 16];
    for (bytes, value) in digest.chunks_mut(4).zip(state) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    digest
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn i420_md5(picture: &Picture) -> String {
    let mut bytes = Vec::new();
    for plane in &picture.planes {
        bytes.extend_from_slice(plane);
    }
    hex(&md5(&bytes))
}

#[test]
fn md5_matches_rfc_1321() {
    assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(
        hex(&md5(b"The quick brown fox jumps over the lazy dog")),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
}

macro_rules! vectors {
    ($($number:literal),* $(,)?) => {
        [$((
            concat!("vp80-00-comprehensive-", $number),
            &include_bytes!(concat!(
                "../../tests/fixtures/codec/vp8/vp80-00-comprehensive-",
                $number,
                ".ivf"
            ))[..],
            include_str!(concat!(
                "../../tests/fixtures/codec/vp8/vp80-00-comprehensive-",
                $number,
                ".ivf.md5"
            )),
        )),*]
    };
}

#[test]
fn decodes_every_test_vector_exactly_as_libvpx_does() {
    let vectors = vectors!(
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013",
        "014", "015", "016", "017", "018",
    );
    for (name, ivf, expected) in vectors {
        let expected: Vec<&str> = expected
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        let mut decoder = Decoder::new(Limits::default());
        let mut actual = Vec::new();
        for (index, frame) in ivf_frames(ivf).into_iter().enumerate() {
            let picture = decoder
                .decode(frame)
                .unwrap_or_else(|error| panic!("{name} frame {index}: {error}"));
            if let Some(picture) = picture {
                actual.push(i420_md5(&picture));
            }
        }
        assert_eq!(actual.len(), expected.len(), "{name}: shown frame count");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual, expected, "{name}: shown frame {index}");
        }
    }
}

#[test]
fn a_hidden_frame_updates_references_without_a_picture() {
    // Vector 018 opens with a key frame that is not shown.
    let ivf = include_bytes!("../../tests/fixtures/codec/vp8/vp80-00-comprehensive-018.ivf");
    let frames = ivf_frames(ivf);
    let mut decoder = Decoder::new(Limits::default());
    assert!(decoder.decode(frames[0]).unwrap().is_none());
    // The next frame predicts from it.
    assert!(decoder.decode(frames[1]).unwrap().is_some());
}

#[test]
fn an_inter_frame_before_any_key_frame_is_refused() {
    let ivf = include_bytes!("../../tests/fixtures/codec/vp8/vp80-00-comprehensive-001.ivf");
    let frames = ivf_frames(ivf);
    let mut decoder = Decoder::new(Limits::default());
    let error = decoder.decode(frames[1]).unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::MalformedMedia);
}

#[test]
fn truncated_and_oversized_frames_are_errors_not_panics() {
    let ivf = include_bytes!("../../tests/fixtures/codec/vp8/vp80-00-comprehensive-001.ivf");
    let key_frame = ivf_frames(ivf)[0];
    for length in [0, 2, 3, 9, 10, 20] {
        let mut decoder = Decoder::new(Limits::default());
        assert!(decoder.decode(&key_frame[..length]).is_err(), "{length}");
    }
    let limits = Limits {
        max_width: 64,
        ..Limits::default()
    };
    let error = Decoder::new(limits).decode(key_frame).unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::ResourceLimit);
}
