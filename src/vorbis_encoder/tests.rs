//! Tests against data captured from the C reference (libvorbis 1.3.7, see
//! tools/vorbis_encoder/make_fixtures.py) plus API behaviour tests. No external tools needed.

use super::test_fixtures::{BITRATES, ENCODES, HEADERS};
use super::*;

fn fnv64(data: &[u8], mut h: u64) -> u64 {
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Deterministic test signal; must match `synth` in tools/vorbis_encoder/make_fixtures.py.
/// Integer LCG noise + per-channel triangle + periodic 8x bursts (transients),
/// every sample an exact f32.
fn synth(frames: usize, channels: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    let mut state = seed;
    for i in 0..frames {
        for c in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 16) & 0x7ff) as i32 - 1024;
            let p = 97 + 13 * c as i32;
            let ph = i as i32 % p;
            let tri = if ph < p / 2 { ph } else { p - ph };
            let mut x = noise * 3 + tri * 300 - (p / 4) * 300;
            if (i / 64) % 150 == 0 {
                x *= 8;
            }
            out.push(x as f32 / 32768.0);
        }
    }
    out
}

fn encode_all(enc: &mut VorbisEncoder, pcm: &[f32], chunk_frames: usize) -> Vec<(Vec<u8>, u64)> {
    let ch = enc.channels;
    let mut packets = Vec::new();
    for c in pcm.chunks(chunk_frames * ch) {
        packets.extend(enc.encode(c).unwrap());
    }
    packets.extend(enc.finish().unwrap());
    packets
}

#[test]
fn headers_match_libvorbis() {
    for &(rate, ch, q, id, comment, setup_len, setup_hash) in HEADERS {
        let enc = VorbisEncoder::new(rate, ch, q).unwrap();
        let [h0, h1, h2] = enc.headers(LIBVORBIS_VENDOR);
        assert_eq!(h0, id, "identification header {rate} {ch} {q}");
        assert_eq!(h1, comment, "comment header {rate} {ch} {q}");
        assert_eq!(h2.len(), setup_len, "setup header length {rate} {ch} {q}");
        assert_eq!(
            fnv64(&h2, FNV_OFFSET),
            setup_hash,
            "setup header {rate} {ch} {q}"
        );
    }
}

#[test]
fn identification_header_fields() {
    let enc = VorbisEncoder::new(44100, 2, 0.5).unwrap();
    let id = &enc.headers("x")[0];
    assert_eq!(id.len(), 30);
    assert_eq!(&id[..7], b"\x01vorbis");
    assert_eq!(id[11], 2);
    assert_eq!(u32::from_le_bytes(id[12..16].try_into().unwrap()), 44100);
    let nominal = i32::from_le_bytes(id[20..24].try_into().unwrap());
    assert_eq!(nominal, enc.nominal_bitrate());
    assert_eq!(enc.nominal_bitrate(), 160000);
    assert_eq!(enc.blocksizes(), (256, 2048));
    assert_eq!(id[28], 0xb8); // log2(256)=8 | log2(2048)=11 << 4
    assert_eq!(id[29], 1);
}

#[test]
fn comment_header_carries_vendor() {
    let enc = VorbisEncoder::new(48000, 1, 0.2).unwrap();
    let c = &enc.headers("zvidlib")[1];
    assert_eq!(&c[..7], b"\x03vorbis");
    assert_eq!(u32::from_le_bytes(c[7..11].try_into().unwrap()), 7);
    assert_eq!(&c[11..18], b"zvidlib");
    assert_eq!(&c[18..22], &[0, 0, 0, 0]);
    assert_eq!(c[22], 1);
    assert_eq!(c.len(), 23);
}

#[test]
fn encodes_match_libvorbis() {
    for &(rate, ch, q, frames, seed, chunk, npk, nbytes, last_granule, hash) in ENCODES {
        let pcm = synth(frames, usize::from(ch), seed);
        let mut enc = VorbisEncoder::new(rate, ch, q).unwrap();
        let packets = encode_all(&mut enc, &pcm, chunk);
        let mut h = FNV_OFFSET;
        let mut bytes = 0;
        for (p, g) in &packets {
            h = fnv64(&(p.len() as u32).to_le_bytes(), h);
            h = fnv64(p, h);
            h = fnv64(&g.to_le_bytes(), h);
            bytes += p.len();
        }
        let ctx = format!("{rate} Hz {ch} ch q={q} frames={frames}");
        assert_eq!(packets.len(), npk, "packet count, {ctx}");
        assert_eq!(bytes, nbytes, "total bytes, {ctx}");
        assert_eq!(
            packets.last().unwrap().1,
            last_granule,
            "last granule, {ctx}"
        );
        assert_eq!(last_granule, frames as u64);
        assert_eq!(h, hash, "packet hash, {ctx}");
    }
}

#[test]
fn bitrate_mapping_matches_libvorbis() {
    for &(rate, ch, br, want) in BITRATES {
        assert_eq!(
            quality_for_nominal_bitrate(rate, ch, br),
            want,
            "{rate} Hz {ch} ch {br} bit/s"
        );
    }
    assert_eq!(quality_for_nominal_bitrate(44100, 2, 0), None);
    assert_eq!(quality_for_nominal_bitrate(44100, 3, 128_000), None);
    // rates above 50 kHz use setup_X, which has no bitrate table
    assert_eq!(quality_for_nominal_bitrate(96000, 2, 128_000), None);
}

#[test]
fn granules_are_monotonic_and_end_at_input_length() {
    let frames = 25_000;
    let pcm = synth(frames, 2, 9);
    let mut enc = VorbisEncoder::new(48000, 2, 0.4).unwrap();
    let packets = encode_all(&mut enc, &pcm, 777);
    assert_eq!(packets[0].1, 0);
    for w in packets.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
    assert_eq!(packets.last().unwrap().1, frames as u64);
    // every audio packet starts with the audio packet type bit cleared
    assert!(packets.iter().all(|(p, _)| p[0] & 1 == 0));
}

#[test]
fn chunking_after_stream_start_does_not_change_output() {
    let frames = 20_000;
    let pcm = synth(frames, 1, 11);
    let run = |tail_chunk: usize| {
        let mut enc = VorbisEncoder::new(44100, 1, 0.6).unwrap();
        let mut out = Vec::new();
        // identical first writes (the start-of-stream extrapolation happens
        // once more than one long block is buffered)
        for c in pcm[..4096].chunks(1024) {
            out.extend(enc.encode(c).unwrap());
        }
        for c in pcm[4096..].chunks(tail_chunk) {
            out.extend(enc.encode(c).unwrap());
        }
        out.extend(enc.finish().unwrap());
        out
    };
    let a = run(1);
    assert_eq!(a, run(7));
    assert_eq!(a, run(5000));
    assert_eq!(a, run(MAX_WRITE_FRAMES * 3 + 1));
}

#[test]
fn empty_and_tiny_streams() {
    let mut enc = VorbisEncoder::new(44100, 2, 0.5).unwrap();
    let p = enc.finish().unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].1, 0);

    let mut enc = VorbisEncoder::new(44100, 1, 0.5).unwrap();
    assert!(enc.encode(&[]).unwrap().is_empty());
    assert!(enc.encode(&[0.25]).unwrap().is_empty());
    let p = enc.finish().unwrap();
    assert_eq!(p.last().unwrap().1, 1);
}

#[test]
fn api_errors() {
    assert!(VorbisEncoder::new(44100, 0, 0.5).is_err());
    assert!(VorbisEncoder::new(44100, 3, 0.5).is_err());
    assert!(VorbisEncoder::new(44100, 2, 1.01).is_err());
    assert!(VorbisEncoder::new(44100, 2, -0.2).is_err());
    assert!(VorbisEncoder::new(44100, 2, f32::NAN).is_err());
    assert!(VorbisEncoder::new(0, 2, 0.5).is_err());
    assert!(VorbisEncoder::new(250_000, 2, 0.5).is_err());

    let mut enc = VorbisEncoder::new(44100, 2, 0.5).unwrap();
    assert!(enc.encode(&[0.0, 0.0, 0.0]).is_err());
    enc.finish().unwrap();
    assert!(enc.encode(&[0.0, 0.0]).is_err());
    assert!(enc.finish().is_err());
    let e = VorbisEncoder::new(44100, 7, 0.5).unwrap_err();
    assert!(e.to_string().contains("7 channels"));
}

#[test]
fn every_supported_rate_and_quality_initializes() {
    for rate in [
        4000, 8000, 8500, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 96000,
        192_000,
    ] {
        for ch in 1..=2 {
            for qi in -1..=10 {
                let q = qi as f32 / 10.0;
                let enc = VorbisEncoder::new(rate, ch, q).unwrap();
                let (s, l) = enc.blocksizes();
                assert!(s.is_power_of_two() && l.is_power_of_two() && s <= l);
            }
        }
    }
}

/// Absurd amplitudes, infinities and NaNs must not panic (C has undefined
/// behaviour there; the port uses wrapping integer arithmetic and saturating
/// float->int conversions instead). Run in debug builds too (overflow checks).
#[test]
fn pathological_input_does_not_panic() {
    type Gen = fn(usize) -> f32;
    let cases: [(&str, Gen); 6] = [
        ("1e4", |i| if i % 2 == 0 { 1e4 } else { -1e4 }),
        ("1e10", |i| ((i * 7919) % 1000) as f32 * 1e7),
        ("f32max", |i| if i % 3 == 0 { f32::MAX } else { -f32::MAX }),
        ("inf", |i| if i % 100 == 0 { f32::INFINITY } else { 0.1 }),
        ("nan", |i| if i % 50 == 0 { f32::NAN } else { 0.2 }),
        ("denormal", |i| if i % 2 == 0 { 1e-40 } else { -1e-42 }),
    ];
    for (name, gen_fn) in cases {
        for (rate, ch, q) in [(44100u32, 2u16, -0.1f32), (48000, 1, 0.5), (8000, 1, 1.0)] {
            let mut enc = VorbisEncoder::new(rate, ch, q).unwrap();
            let pcm: Vec<f32> = (0..6000 * usize::from(ch)).map(gen_fn).collect();
            let packets = encode_all(&mut enc, &pcm, 1024);
            assert_eq!(packets.last().unwrap().1, 6000, "{name}");
        }
    }
}

#[test]
fn encoder_is_send_sync_and_debug_is_short() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VorbisEncoder>();
    let enc = VorbisEncoder::new(44100, 2, 0.5).unwrap();
    let s = format!("{enc:?}");
    assert!(s.len() < 200, "{s}");
    assert!(s.contains("44100"));
}
