//! The `codec` target's support: the checked-in AV1 conformance vectors,
//! decoded once per process, on top of `zvidlib-bench-support`'s shared
//! helpers.
// Each bench target of this package compiles this module and uses only what
// its own groups need, so unused-here is not dead.
#![allow(dead_code)]

use std::sync::OnceLock;

use zvidlib_av1::decode_av1_lossless_intra;
use zvidlib_core::{Limits, VideoFrame};

zvidlib_bench_support::bench_support!(zvidlib_av1::simd_sites, zvidlib_color::simd_sites);

/// The standardized AV1 Main lossless monochrome intra vector (17x9).
pub fn av1_lossless_intra_stream() -> &'static [u8] {
    static STREAM: OnceLock<Vec<u8>> = OnceLock::new();
    STREAM.get_or_init(|| from_hex(include_str!("../../tests/fixtures/av1_lossless_17x9.hex")))
}

/// The standardized AV1 inter + `show_existing_frame` vector (16x16).
pub fn av1_inter_stream() -> &'static [u8] {
    static STREAM: OnceLock<Vec<u8>> = OnceLock::new();
    STREAM.get_or_init(|| {
        from_hex(include_str!(
            "../../tests/fixtures/av1_inter_show_existing_16x16.hex"
        ))
    })
}

/// Byte ranges of the temporal units in [`av1_inter_stream`], split at each
/// temporal-delimiter OBU (`obu_type == 2`).
pub fn av1_inter_temporal_units() -> &'static [std::ops::Range<usize>] {
    static UNITS: OnceLock<Vec<std::ops::Range<usize>>> = OnceLock::new();
    UNITS.get_or_init(|| {
        let stream = av1_inter_stream();
        let mut starts = Vec::new();
        let mut cursor = 0usize;
        while cursor < stream.len() {
            let start = cursor;
            let header = stream[cursor];
            cursor += 1;
            let obu_type = (header >> 3) & 0x0f;
            assert_ne!(header & 0x02, 0, "fixture OBU must carry a size field");
            let mut payload_len = 0usize;
            let mut shift = 0usize;
            loop {
                let byte = stream[cursor];
                cursor += 1;
                payload_len |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            cursor += payload_len;
            assert!(cursor <= stream.len(), "fixture OBU length is in bounds");
            if obu_type == 2 {
                starts.push(start);
            }
        }
        starts
            .iter()
            .enumerate()
            .map(|(index, &start)| {
                let end = starts.get(index + 1).copied().unwrap_or(stream.len());
                start..end
            })
            .collect()
    })
}

/// The AV1 intra vector decoded once, for benchmarks that consume decoded planes
/// rather than measuring the decode itself.
pub fn av1_lossless_intra_frame() -> &'static VideoFrame {
    static FRAME: OnceLock<VideoFrame> = OnceLock::new();
    FRAME.get_or_init(|| {
        decode_av1_lossless_intra(av1_lossless_intra_stream(), &Limits::default())
            .expect("the checked-in AV1 intra vector decodes")
    })
}
