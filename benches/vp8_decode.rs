//! Scalar-versus-SIMD benchmarks for zvidlib's pure-Rust VP8 software decoder
//! (issue #568).
//!
//! The decode-side counterpart to `benches/vp8_encode.rs`, which times the
//! reconstruction kernels the encoder and decoder share (`vp8_recon`). This
//! target times what only the decoder runs, and the decoder as a whole. Every
//! group runs once per instruction set `zvidlib::simd::available()` reports,
//! through the crate-wide override in [`zvidlib::simd`], and
//! `benches/support/isa.rs` asserts both that each arm is bit-exact with scalar
//! before timing it and that the override really landed in every dispatch
//! family.
//!
//! # Groups
//!
//! | Group | Stage | Site |
//! | --- | --- | --- |
//! | `vp8_decode_720p` | whole-frame decode of a synthetic 1280x720 stream, to the decoded planes | every VP8 site |
//! | `vp8_decode_conformance` | whole-frame decode of all 18 `vp80-00-comprehensive` vectors | every VP8 site |
//! | `vp8_decode_stage_subblock_pred` | the ten 4x4 subblock intra predictors over a whole 1280x720 luma plane | `vp8_decode` |
//! | `vp8_decode_stage_dc_idct` | the DC-only inverse DCT over every 4x4 block of a frame | `vp8_decode` |
//! | `vp8_decode_stage_bilinear` | bilinear sub-pixel prediction (bitstream versions 1-3) of 16x16, 8x8 and 4x4 blocks | `vp8_recon` |
//! | `vp8_decode_stage_loop_filter_simple` | the simple loop filter over a whole frame's luma | `vp8_recon` |
//!
//! The whole-frame groups include the boolean entropy decoder, which is serial
//! and has no vector path, and neither converts to RGBA (that is
//! `vpx_decode`'s `vp8_decode_frame`), so their ratio sits below the per-stage
//! ones.
//!
//! The per-stage inputs come from `zvidlib::vp8_decoder_bench`, a narrow public
//! surface over the otherwise crate-private decoder.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::vp8_decoder_bench::{Vp8StageInputs, decode_pictures, synthetic_stream};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};

/// Dimensions of the per-stage groups and the synthetic whole-frame stream.
const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

/// Frames of the synthetic stream: one key frame and then inter frames.
const STREAM_FRAMES: usize = 8;

fn kernel_workload<'a>(codec: &'a str) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, FrameWork::new(1, WIDTH as u64, HEIGHT as u64))
    }
}

fn stage_inputs() -> &'static Vp8StageInputs {
    static INPUTS: OnceLock<Vp8StageInputs> = OnceLock::new();
    INPUTS.get_or_init(|| Vp8StageInputs::new(WIDTH, HEIGHT))
}

// ---------------------------------------------------------------------------
// Whole-frame decode
// ---------------------------------------------------------------------------

/// The synthetic 720p stream, encoded once per process by the native VP8
/// encoder, decoded end to end once per instruction set.
fn vp8_decode_720p(criterion: &mut Criterion) {
    let stream =
        synthetic_stream(WIDTH, HEIGHT, STREAM_FRAMES).expect("the synthetic frames encode");
    let frames: Vec<&[u8]> = stream.iter().map(Vec::as_slice).collect();
    let workload = IsaWorkload {
        measurement_time: Duration::from_secs(5),
        ..IsaWorkload::new(
            "vp8_decode_720p",
            FrameWork::new(STREAM_FRAMES as u64, WIDTH as u64, HEIGHT as u64),
        )
    };
    bench_across_isas(criterion, &workload, || {
        decode_pictures(&frames).expect("the synthetic stream decodes")
    });
}

/// The VP8 test vectors' frames, split out of their IVF files.
fn conformance_vectors() -> Vec<Vec<&'static [u8]>> {
    macro_rules! vectors {
        ($($number:literal),* $(,)?) => {
            [$(&include_bytes!(concat!(
                "../tests/fixtures/codec/vp8/vp80-00-comprehensive-",
                $number,
                ".ivf"
            ))[..]),*]
        };
    }
    let files = vectors!(
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013",
        "014", "015", "016", "017", "018",
    );
    files
        .into_iter()
        .map(|file| {
            let mut frames = Vec::new();
            let mut position = usize::from(u16::from_le_bytes([file[6], file[7]]));
            while position + 12 <= file.len() {
                let size = u32::from_le_bytes(file[position..position + 4].try_into().unwrap());
                let start = position + 12;
                frames.push(&file[start..start + size as usize]);
                position = start + size as usize;
            }
            frames
        })
        .collect()
}

/// All 18 `vp80-00-comprehensive` vectors, decoded end to end. They are small
/// (176x144) but were coded by libvpx and between them reach every feature
/// the decoder has, so this is the real-content mix the synthetic stream is
/// not: split motion vectors, the bilinear filters, the simple loop filter,
/// every intra mode.
fn vp8_decode_conformance(criterion: &mut Criterion) {
    let vectors = conformance_vectors();
    let frames: u64 = vectors.iter().map(|frames| frames.len() as u64).sum();
    let workload = IsaWorkload {
        // One pass is over half a second, so criterion's ten samples need
        // more than its default window.
        measurement_time: Duration::from_secs(15),
        ..IsaWorkload::new("vp8_decode_conformance", FrameWork::new(frames, 176, 144))
    };
    bench_across_isas(criterion, &workload, || {
        let mut out = Vec::new();
        for frames in &vectors {
            out.extend(decode_pictures(frames).expect("the test vectors decode"));
        }
        out
    });
}

// ---------------------------------------------------------------------------
// Per-stage groups
// ---------------------------------------------------------------------------

fn vp8_decode_stages(criterion: &mut Criterion) {
    let inputs = stage_inputs();
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_decode_stage_subblock_pred"),
        || inputs.subblock_prediction(),
    );
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_decode_stage_dc_idct"),
        || inputs.dc_inverse_dct(),
    );
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_decode_stage_bilinear"),
        || inputs.bilinear_prediction(),
    );
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_decode_stage_loop_filter_simple"),
        || inputs.simple_loop_filter(),
    );
}

criterion_group!(
    benches,
    log_host_isas,
    vp8_decode_720p,
    vp8_decode_conformance,
    vp8_decode_stages
);
criterion_main!(benches);
