// `zvidlib::hevc_encoder_bench` is itself gated to non-wasm targets, so this
// guard keeps `wasm-pack test`'s `cargo build --tests` from failing to resolve
// it, the way the other native-only integration tests do.
#![cfg(not(target_arch = "wasm32"))]

//! The widened bit writer changes throughput, never bits.
//!
//! Issue #233 widened `BitWriter::put_bits` and gave §7.3.8.7 PCM sample data a
//! byte-aligned bulk path that bypasses the bit accumulator. Both are pure
//! rewrites of how bits reach the buffer, so the access units they produce must
//! stay byte-identical to the ones the `put_bit`-at-a-time writer produced. The
//! golden digests below were captured from the pre-rewrite implementation; a
//! change here is a bitstream regression, not a benchmark result.

use zvidlib::hevc_encoder_bench as encoder_bench;

/// FNV-1a over the access unit. A digest rather than a stored bitstream keeps
/// the guard readable while still being byte-exact — any single flipped bit
/// changes it.
fn digest(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x1000_0193);
    }
    hash
}

/// A deterministic picture with structure in every plane, so the digest depends
/// on the sample data and not just on the headers.
fn planes(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = (0..width * height)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let chroma_len = (width / 2) * (height / 2);
    let cb = (0..chroma_len)
        .map(|i| (i.wrapping_mul(97) ^ 0x5A) as u8)
        .collect();
    let cr = (0..chroma_len)
        .map(|i| (i.wrapping_mul(131) ^ 0xA5) as u8)
        .collect();
    (y, cb, cr)
}

#[test]
fn pcm_access_units_are_byte_identical_to_the_bit_at_a_time_writer() {
    for &(width, height, golden) in &[
        (64usize, 32usize, 0x4614_e05b_fff0_d487u64),
        (128, 64, 0x96f5_25ab_2efe_bab9),
        (48, 48, 0x1262_5b7f_c893_da03),
    ] {
        let (y, cb, cr) = planes(width, height);
        let au = encoder_bench::write_idr_pcm_access_unit(&y, &cb, &cr, width, height);
        assert_eq!(
            digest(&au),
            golden,
            "PCM access unit changed at {width}x{height} ({} bytes)",
            au.len()
        );
    }
}

#[test]
fn syntax_writes_are_byte_identical_to_the_bit_at_a_time_writer() {
    let values: Vec<u32> = (0..4096u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 11)
        .collect();
    let bytes = encoder_bench::bitwriter_write_syntax(&values);
    assert_eq!(
        digest(&bytes),
        0x5464_09b1_d575_7999u64,
        "syntax bitstream changed"
    );
}
