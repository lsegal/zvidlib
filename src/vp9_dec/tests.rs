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

/// Decodes every libvpx VP9 profile 0 test vector in the directory named
/// by `ZVIDLIB_VP9_VECTORS` (IVF files with their `.md5` files beside
/// them) and compares each shown frame with libvpx's digest.
/// `ZVIDLIB_VP9_VECTOR_FILTER` restricts the run to names containing it.
#[test]
#[ignore = "needs the libvpx test vectors; set ZVIDLIB_VP9_VECTORS"]
fn libvpx_test_vectors() {
    let Ok(directory) = std::env::var("ZVIDLIB_VP9_VECTORS") else {
        return;
    };
    let filter = std::env::var("ZVIDLIB_VP9_VECTOR_FILTER").unwrap_or_default();
    let mut names: Vec<_> = std::fs::read_dir(&directory)
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            (path.extension()? == "ivf").then_some(path)
        })
        .filter(|path| path.to_string_lossy().contains(&filter))
        .collect();
    names.sort();
    let mut failures = Vec::new();
    for path in &names {
        let data = std::fs::read(path).unwrap();
        let expected: Vec<String> = std::fs::read_to_string(path.with_extension("md5"))
            .unwrap()
            .lines()
            .map(|line| line.split_whitespace().next().unwrap().to_owned())
            .collect();
        let mut decoder = Decoder::new(Limits {
            max_width: 65536,
            max_height: 65536,
            max_allocation_bytes: 1 << 32,
            ..Limits::default()
        });
        let mut shown = 0usize;
        let mut failure = None;
        for (index, frame) in ivf_frames(&data).into_iter().enumerate() {
            match decoder.decode_chunk(frame) {
                Ok(picture) => {
                    if let Some(picture) = picture {
                        let digest = picture_md5(&picture);
                        if expected.get(shown) != Some(&digest) {
                            failure = Some(format!("frame {shown} (chunk {index}) differs"));
                        }
                        shown += 1;
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
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        match failure {
            Some(reason) => {
                eprintln!("FAIL {name}: {reason}");
                failures.push(name);
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
