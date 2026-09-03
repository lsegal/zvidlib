//! Where HEVC decode time actually goes, on the bundled 1080p sample.
//!
//! The per-stage criterion groups in `benches/hevc_decode.rs` time each kernel
//! on a synthetic workload of its own: they say how fast §8.7.3 SAO is, not how
//! much SAO a 1080p frame runs. Issue #189 is the gap that leaves — the
//! whole-frame `hevc_decode/<isa>` arms move only ~1.06x while the individual
//! kernels measure 1.3x-2.4x, and neither number explains the other without a
//! share-of-total breakdown.
//!
//! Issue #280 then split §8.5.3.3 inter prediction into `inter_pred_filter`,
//! `inter_pred_write` and `inter_pred_setup`, because the single row it used to
//! print was a third §8.4.4.1 write-back and reference-plane setup that no
//! vector kernel reaches — see `benches/README.md` for what that answered.
//!
//! This example produces that breakdown. It decodes frames of
//! `examples/media/BigBuckBunny.mp4` through the ordinary public decoder with
//! `zvidlib::hevc_decode_profile` running, then prints each stage's exclusive
//! share, the fraction of the decode the crate's vector kernels cover, and the
//! Amdahl ceiling that fraction implies.
//!
//! It is an example rather than a benchmark on purpose: criterion reports the
//! distribution of one workload's wall time, and the question here is how a
//! single decode divides across stages, which is a composition rather than a
//! measurement to regress against.
//!
//! ```sh
//! cargo run --release --features native --example hevc_decode_profile
//! cargo run --release --features native --example hevc_decode_profile -- 120
//! cargo run --release --features native --example hevc_decode_profile -- 60 scalar
//! ```
//!
//! The `native` feature gates no code here; it is what keeps the wasm build,
//! which has no HEVC decoder at all, from trying to compile this target.
//!
//! The second argument pins an instruction set through `zvidlib::simd`, so the
//! same breakdown can be read with and without the vector kernels. Run under
//! `--release`: the decoder is a pure-Rust software codec and a debug build's
//! stage mix is not the shipped build's.
//!
//! # `--pair`: what one decision is worth to a whole decode
//!
//! `--pair` answers a different question from the breakdown: not how a decode
//! divides, but how much a single kernel decision moves it. It A/Bs the
//! §8.5.3.3 16-bit interpolation accumulation #404 landed, running both arms
//! **in one process, interleaved**, and — the part that makes it usable — a
//! third *null control* arm that runs the same code as the first, so the
//! instrument reports its own noise floor next to the effect.
//!
//! ```sh
//! cargo run --release --features native --example hevc_decode_profile -- --pair
//! cargo run --release --features native --example hevc_decode_profile -- 48 avx2 --pair
//! cargo run --release --features native --example hevc_decode_profile -- --pair --rounds 20
//! ```
//!
//! Issue #426 is why the control is there. #404 paired two *builds* in two
//! *processes* and its control arm — `scalar`, where both binaries provably
//! execute identical code — read 1.06x and 1.07x against a true answer of
//! 1.00x, which is a wider miss than the effect being asked about. Reading the
//! two arms out of one process is the same arrangement
//! `measure_narrow_vs_wide_block` and `measure_2d_ring_vs_flat` already use,
//! and for the reason both of them state: separate benchmark processes
//! disagree with each other by more than the effect being measured.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use zvidlib::hevc_decode_profile as profile;
use zvidlib::hevc_narrow_interp as narrow_interp;
use zvidlib::io::MemorySource;
use zvidlib::simd::{self, SimdIsa};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, DecodedVideoFrame, EncodedVideoSample,
    HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig,
    VideoDecoderFactory, native_hevc_video_decoder_factory,
};

/// Frames to decode when the caller names no count.
///
/// Enough to cross several inter-coded pictures — an I-frame-only profile would
/// read as an intra decoder — and few enough to finish in seconds.
const DEFAULT_FRAMES: usize = 48;

/// Interleaved rounds `--pair` takes the elementwise minimum over.
///
/// Twelve because that is what issue #426 records the unusable cross-process
/// pairing as having used, so the in-process instrument is asked for its
/// answer at the same round count rather than at a more generous one.
const PAIR_ROUNDS: usize = 12;

fn main() {
    let mut positional = Vec::new();
    let mut pair = false;
    let mut rounds = PAIR_ROUNDS;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pair" => pair = true,
            "--rounds" => {
                rounds = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .expect("--rounds takes a positive round count");
                assert!(rounds > 0, "--rounds takes a positive round count");
            }
            _ => positional.push(arg),
        }
    }
    let frames: usize = positional
        .first()
        .map_or(DEFAULT_FRAMES, |arg| arg.parse().expect("frame count"));
    let isa = positional.get(1).map(|name| parse_isa(name));

    if let Some(isa) = isa {
        simd::set_override(Some(isa));
    }
    let active = simd::active();

    let (configuration, samples) = bundled_sample();

    if pair {
        pair_narrow_accumulation(&configuration, &samples, frames, rounds, active);
        if isa.is_some() {
            simd::set_override(None);
        }
        return;
    }

    let (decoded, report) = decode_profiled(&configuration, &samples, frames, |_| {});

    if isa.is_some() {
        simd::set_override(None);
    }

    println!("# HEVC decode stage attribution");
    println!();
    println!(
        "{decoded} frames of examples/media/BigBuckBunny.mp4, instruction set `{active:?}`, \
         {:.1} ms total ({:.2} ms/frame)",
        report.total().as_secs_f64() * 1e3,
        report.total().as_secs_f64() * 1e3 / decoded.max(1) as f64,
    );
    println!();
    print!("{}", report.markdown_table(decoded));
    println!();

    println!(
        "Vectorized stages cover {:.1}% of the measured total, and {:.1}% of decode \
         proper (total minus `color_convert`, which is output conversion, not decoding).",
        report.vectorized_share() * 100.0,
        report.vectorized_decode_share() * 100.0,
    );
    println!(
        "Amdahl ceiling on the measured total: infinitely fast vector kernels would run \
         {:.2}x faster; a uniform 2x on those stages gives {:.2}x, a uniform 4x gives {:.2}x.",
        report.max_whole_frame_speedup(),
        report.speedup_at(2.0),
        report.speedup_at(4.0),
    );
    println!(
        "`color_convert` alone is {:.1}% of the measured total; it is output conversion \
         rather than decoding, and since issue #219 it has a vector kernel of its own.",
        report.share(profile::Stage::ColorConvert) * 100.0,
    );
    println!(
        "Serial entropy decoding (`slice_data_cabac` + `residual_cabac`) is {:.1}% of the total.",
        (report.share(profile::Stage::SliceData) + report.share(profile::Stage::Residual)) * 100.0,
    );
    println!(
        "Profiler overhead is at most {:.1} ms of that ({:.1}%, {} scopes).",
        report.overhead().as_secs_f64() * 1e3,
        report.overhead().as_secs_f64() / report.total().as_secs_f64() * 100.0,
        report.scopes(),
    );
}

/// Decodes `frames` frames of the bundled sample under whatever `simd` and
/// narrow-accumulation overrides are currently in force, handing each decoded
/// frame to `each`, and returns the frame count and the stage profile.
///
/// A few access units are decoded before profiling starts, so the report covers
/// steady state rather than the first picture's parameter-set activation and
/// allocation. The decoder is built here rather than passed in because each arm
/// of a pairing has to start from the same decoder state to be comparable.
fn decode_profiled(
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    frames: usize,
    mut each: impl FnMut(&DecodedVideoFrame),
) -> (usize, profile::Report) {
    let factory = native_hevc_video_decoder_factory();
    let mut decoder = factory
        .create(configuration, &Limits::default())
        .expect("the software HEVC decoder is constructible");
    let cancellation = CancellationToken::new();

    let mut warmup = 0usize;
    let mut iter = samples.iter();
    for encoded in iter.by_ref() {
        warmup += decoder
            .submit(encoded, &cancellation)
            .expect("the bundled sample decodes")
            .len();
        if warmup >= 2 {
            break;
        }
    }

    assert!(
        profile::start(),
        "stage profiling needs a monotonic clock; this target has none"
    );
    let mut decoded = 0usize;
    for encoded in iter {
        let batch = decoder
            .submit(encoded, &cancellation)
            .expect("the bundled sample decodes");
        decoded += batch.len();
        for frame in &batch {
            each(frame);
        }
        if decoded >= frames {
            break;
        }
    }
    let report = profile::finish().expect("profiling was started");
    (decoded, report)
}

/// A/Bs the §8.5.3.3 16-bit interpolation accumulation over a whole decode,
/// with a null control that reports the instrument's own noise floor.
///
/// Three arms run per round, interleaved in the same order every round:
///
/// * **wide** — the accumulation forced off, which is the `i32` path
///   everywhere and so exactly what `main` executed before #404 landed.
/// * **shipped** — `inter_pred::narrows` deciding, which is what this crate
///   executes today.
/// * **control** — forced off again. It runs the *same code* as the wide arm,
///   so its ratio against that arm has the known answer 1.00x, and whatever it
///   actually reads is this instrument's noise floor on this host. No reading
///   of the shipped arm means anything the control cannot underwrite.
///
/// This is the arrangement `measure_narrow_vs_wide_block` and
/// `measure_2d_ring_vs_flat` already use, lifted from a block to a decode, and
/// for the reason both of them give: separate benchmark processes disagree with
/// each other by more than the effect being measured. Issue #426 is that
/// failure in its acute form — a cross-process, cross-build pairing whose
/// `scalar` control, where both binaries provably execute identical code, still
/// read 1.06x and 1.07x against a true answer of 1.00x.
///
/// The three arms are asserted to decode bit-identically before anything is
/// timed, which is the whole-decode form of the guard the block-level A/B
/// already applies.
fn pair_narrow_accumulation(
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    frames: usize,
    rounds: usize,
    active: SimdIsa,
) {
    /// The arms, in the order every round runs them.
    const ARMS: [(&str, Option<bool>); 3] = [
        ("wide (`i32`, pre-#404)", Some(false)),
        ("shipped (`narrows` decides)", None),
        ("control (wide again)", Some(false)),
    ];

    // Bit-exactness first, and untimed: a ratio between two arms that decode
    // different pictures would not be a measurement of anything.
    let mut digests = Vec::new();
    let mut counts = Vec::new();
    for (_, narrow) in ARMS {
        narrow_interp::set_override(narrow);
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        let (decoded, _) = decode_profiled(configuration, samples, frames, |frame| {
            digest = fnv1a(digest, &frame.presentation_index.0.to_le_bytes());
            for plane in &frame.frame.planes {
                digest = fnv1a(digest, &plane.data);
            }
        });
        digests.push(digest);
        counts.push(decoded);
    }
    narrow_interp::set_override(None);
    assert_eq!(
        digests[0], digests[1],
        "the narrow accumulation changed a decoded sample"
    );
    assert_eq!(
        digests[0], digests[2],
        "the decode is not reproducible between two identical arms"
    );
    assert!(
        counts.iter().all(|&count| count == counts[0]),
        "the arms decoded different frame counts: {counts:?}"
    );
    let decoded = counts[0];

    // `INFINITY` so the first round's reading always wins the minimum.
    let mut best_total = [f64::INFINITY; ARMS.len()];
    let mut best_filter = [f64::INFINITY; ARMS.len()];
    for _ in 0..rounds {
        for (slot, (_, narrow)) in ARMS.iter().enumerate() {
            narrow_interp::set_override(*narrow);
            let (count, report) = decode_profiled(configuration, samples, frames, |_| {});
            let per_frame = count.max(1) as f64;
            best_total[slot] = best_total[slot].min(report.total().as_secs_f64() * 1e3 / per_frame);
            best_filter[slot] = best_filter[slot]
                .min(report.stage(profile::Stage::InterPredFilter).as_secs_f64() * 1e3 / per_frame);
        }
    }
    narrow_interp::set_override(None);

    println!("# HEVC narrow interpolation accumulation, whole-decode A/B");
    println!();
    println!(
        "{decoded} frames of examples/media/BigBuckBunny.mp4, instruction set `{active:?}`, \
         {rounds} interleaved rounds in one process, elementwise minimum."
    );
    println!();
    println!("| Arm | ms/frame | `inter_pred_filter` ms/frame |");
    println!("| --- | ---: | ---: |");
    for (slot, (name, _)) in ARMS.iter().enumerate() {
        println!(
            "| {name} | {:.3} | {:.3} |",
            best_total[slot], best_filter[slot]
        );
    }
    println!();

    // Every ratio is taken against the wide arm, including the control's, so
    // the control answers the same question the effect does.
    let effect_total = best_total[0] / best_total[1];
    let effect_filter = best_filter[0] / best_filter[1];
    let control_total = best_total[0] / best_total[2];
    let control_filter = best_filter[0] / best_filter[2];
    println!(
        "Shipped against wide: **{effect_total:.4}x** whole decode, \
         **{effect_filter:.4}x** `inter_pred_filter`."
    );
    println!(
        "Control against wide (identical code; the true answer is 1.0000x): \
         **{control_total:.4}x** whole decode, **{control_filter:.4}x** `inter_pred_filter`."
    );
    println!();

    // The control's own miss is the smallest difference this instrument can
    // tell from nothing, so an effect inside it is not resolved by this run.
    let floor_total = (control_total - 1.0).abs();
    let floor_filter = (control_filter - 1.0).abs();
    println!(
        "Noise floor (the control's miss of its known answer): {:.2}% whole decode, \
         {:.2}% `inter_pred_filter`.",
        floor_total * 100.0,
        floor_filter * 100.0,
    );
    for (label, effect, floor) in [
        ("whole decode", effect_total, floor_total),
        ("`inter_pred_filter`", effect_filter, floor_filter),
    ] {
        let effect_size = (effect - 1.0).abs();
        if effect_size > floor {
            println!(
                "  {label}: resolved — the {:.2}% effect is larger than the {:.2}% floor.",
                effect_size * 100.0,
                floor * 100.0,
            );
        } else {
            println!(
                "  {label}: not resolved — the {:.2}% effect is inside the {:.2}% floor, \
                 which bounds it.",
                effect_size * 100.0,
                floor * 100.0,
            );
        }
    }
}

/// FNV-1a over `bytes`, continuing from `hash`.
///
/// A decode digest only has to detect a changed sample, and this is a few
/// lines against a dependency. It runs in the untimed verification pass.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Maps a command-line instruction-set name onto [`SimdIsa`].
fn parse_isa(name: &str) -> SimdIsa {
    simd::available()
        .iter()
        .copied()
        .find(|isa| format!("{isa:?}").eq_ignore_ascii_case(name))
        .unwrap_or_else(|| {
            panic!(
                "unknown instruction set {name:?}; this host offers {:?}",
                simd::available()
            )
        })
}

/// Demuxes the bundled sample into a decoder configuration and its access units.
fn bundled_sample() -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("media/BigBuckBunny.mp4").to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .expect("the bundled sample is a readable MP4");
    let track = movie.track(1).expect("the bundled sample has track 1");
    let dimensions = track
        .dimensions
        .expect("the bundled sample's video track is dimensioned");
    let samples = block_on(track.to_encoded_video_samples(&source, &limits))
        .expect("the bundled sample's video samples are readable");
    (
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            // The question is where *this crate's* decode time goes, not the
            // host's fixed-function block's.
            hardware: HardwarePreference::Avoid,
            configuration: track.decoder_config.clone(),
        },
        samples,
    )
}

/// Minimal executor for the crate's `async` I/O entry points.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
