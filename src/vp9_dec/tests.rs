//! Tests of the VP9 decoder against libvpx, the reference decoder.

use super::*;

/// A minimal MD5 (RFC 1321), for comparing decoded frames with the
/// per-frame digests libvpx's test vectors ship with.
pub(crate) fn md5(data: &[u8]) -> [u8; 16] {
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

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The MD5 libvpx's test harness computes for a picture: the visible
/// rows of Y, then U, then V.
pub(crate) fn picture_md5(picture: &DecodedPicture) -> String {
    let mut bytes = Vec::new();
    for plane in &picture.planes {
        bytes.extend_from_slice(plane);
    }
    hex(&md5(&bytes))
}

/// The frames of an IVF file.
pub(crate) fn ivf_frames(data: &[u8]) -> Vec<&[u8]> {
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
pub(crate) fn webm_frames(data: &[u8]) -> Vec<&[u8]> {
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
/// `tests/fixtures/codec/libvpx_vp9_test_vectors.txt` (the profile 0 part of
/// `test/test_vectors.cc`).
const LIBVPX_VECTORS: &str = include_str!("../../tests/fixtures/codec/libvpx_vp9_test_vectors.txt");

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
        let mut shown = 0usize;
        let mut failure = None;
        for (index, chunk) in chunks.into_iter().enumerate() {
            match decoder.decode_chunk(chunk) {
                Ok(Some(picture)) => {
                    if expected.get(shown) != Some(&picture_md5(&picture)) {
                        failure = Some(format!("frame {shown} (chunk {index}) differs"));
                    }
                    shown += 1;
                }
                Ok(None) => {}
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
