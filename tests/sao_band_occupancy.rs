//! How many of the 32 §8.7.3.2 bands a single CTB actually occupies.
//!
//! Issue #406 names a transposed formulation of the band search — for each of
//! the 32 bands, a masked horizontal sum over the CTB, trading 32 passes of
//! pure SIMD for one pass of serial scatter — and says up front that "whether
//! that pays depends on how sparse a CTB's band occupancy is, which is itself
//! worth measuring first". This is that measurement, taken before any kernel
//! was written, so the decision to write or not write that kernel is a measured
//! one rather than an assumed one.
//!
//! A transposed pass costs work proportional to the number of bands it visits.
//! It can only visit fewer than 32 if the occupied bands can be *bounded*
//! cheaply, which for a value-range classification means the band range
//! `max − min + 1` rather than the count of distinct bands: a CTB occupying
//! bands 2, 3 and 29 is sparse by count and dense by range, and only the range
//! is derivable from the vector min/max such a pass could afford. Both are
//! reported.
//!
//! Ignored by default because it measures rather than asserts; run it with
//! `cargo test --features native --release --test sao_band_occupancy --
//! --ignored --nocapture`.

#![cfg(all(feature = "native", not(target_arch = "wasm32")))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, FrameDigest, HardwarePreference, Limits,
    Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig, VideoDecoderConformanceVector,
    VideoDecoderFactory, VideoDimensions, native_hevc_video_decoder_factory,
};

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

/// §8.7.3.2 `bandShift` at this crate's 8-bit geometry.
const BAND_SHIFT: u32 = 3;

/// One plane's samples with its row pitch.
struct Plane8 {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

/// The occupancy of every `size`x`size` block of `plane`, as a histogram of
/// distinct occupied bands and one of the band range `max − min + 1`.
fn occupancy(plane: &Plane8, size: usize) -> (Vec<u64>, Vec<u64>) {
    let mut distinct = vec![0u64; 33];
    let mut range = vec![0u64; 33];
    let mut y0 = 0;
    while y0 + size <= plane.height {
        let mut x0 = 0;
        while x0 + size <= plane.width {
            let mut mask = 0u32;
            for y in y0..y0 + size {
                let row = &plane.data[y * plane.width..];
                for &s in &row[x0..x0 + size] {
                    mask |= 1 << (u32::from(s) >> BAND_SHIFT);
                }
            }
            let lo = mask.trailing_zeros() as usize;
            let hi = 31 - mask.leading_zeros() as usize;
            distinct[mask.count_ones() as usize] += 1;
            range[hi - lo + 1] += 1;
            x0 += size;
        }
        y0 += size;
    }
    (distinct, range)
}

/// Prints a histogram as a cumulative distribution, which is the form the
/// decision reads off: what share of CTBs a transposed pass visiting at most
/// `k` bands would cover.
fn report(label: &str, histogram: &[u64]) {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        println!("{label}: no blocks");
        return;
    }
    let mean: f64 = histogram
        .iter()
        .enumerate()
        .map(|(k, &n)| k as f64 * n as f64)
        .sum::<f64>()
        / total as f64;
    print!("{label:<34} mean {mean:>5.1}  ");
    for k in [4usize, 8, 12, 16, 24] {
        let cumulative: u64 = histogram[..=k].iter().sum();
        print!(
            "<={k:<2}: {:>5.1}%  ",
            100.0 * cumulative as f64 / total as f64
        );
    }
    println!("n={total}");
}

/// Rebuilds the luma plane `benches/support::synthetic_yuv420_sequence` feeds
/// the encoder groups, which is the content `hevc_encode_640x352_reconstruct` —
/// the group the issue's acceptance criterion names — actually searches.
///
/// Kept identical to that generator on purpose: measuring occupancy on some
/// other content would not answer whether a transposed kernel could separate in
/// the group that decides.
fn synthetic_luma(width: usize, height: usize, frame: usize) -> Plane8 {
    let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ frame as u64;
    let mut next_noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 58) as i32
    };
    let shift = (frame * 3) as i32;
    let data = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let gradient = (x as i32 + y as i32 + shift) / 2;
            ((gradient + next_noise()) & 0xff) as u8
        })
        .collect::<Vec<_>>();
    Plane8 {
        data,
        width,
        height,
    }
}

/// The synthetic chroma plane, likewise.
fn synthetic_chroma(width: usize, height: usize, frame: usize, offset: i32) -> Plane8 {
    let shift = (frame * 3) as i32;
    let data = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| (128 + ((x as i32 - y as i32 + shift + offset) % 24) - 12) as u8)
        .collect::<Vec<_>>();
    Plane8 {
        data,
        width,
        height,
    }
}

/// Real decoded luma, as the counterpart to the synthetic content: the encoder
/// group is what the acceptance criterion names, but a decomposition chosen
/// against synthetic content alone would be chosen against a wrapped gradient
/// rather than against video.
fn decoded_luma(frames: usize) -> Vec<Plane8> {
    let expected = include_str!("fixtures/codec/big_buck_bunny_hevc_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("../examples/media/BigBuckBunny.mp4").to_vec());
    let vector = block_on(VideoDecoderConformanceVector::from_mp4(
        "bundled HEVC Main sample",
        &source,
        Mp4DemuxerOptions::default(),
        1,
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(1920, 1080, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        &expected,
    ))
    .unwrap();
    let factory = native_hevc_video_decoder_factory();
    let mut decoder = factory.create(&vector.configuration, &limits).unwrap();
    let cancellation = CancellationToken::new();
    let mut out = Vec::new();
    for sample in vector.samples.iter().take(frames) {
        for frame in decoder.submit(sample, &cancellation).unwrap() {
            let (y, _, _) = zvidlib::hevc_encoder_bench::rgba_to_yuv420_planes(&frame.frame);
            out.push(Plane8 {
                data: y,
                width: frame.frame.dimensions.width as usize,
                height: frame.frame.dimensions.height as usize,
            });
        }
    }
    out
}

/// Sums per-plane histograms into one.
fn accumulate(into: &mut (Vec<u64>, Vec<u64>), from: (Vec<u64>, Vec<u64>)) {
    for (a, b) in into.0.iter_mut().zip(from.0) {
        *a += b;
    }
    for (a, b) in into.1.iter_mut().zip(from.1) {
        *a += b;
    }
}

#[test]
#[ignore = "measurement; run explicitly with --ignored --nocapture"]
fn how_many_bands_a_ctb_occupies() {
    println!("# §8.7.3.2 band occupancy per CTB, luma 16x16 and chroma 8x8");
    println!("# `distinct` is how many of the 32 bands hold a sample;");
    println!("# `range` is max_band - min_band + 1, the bound a transposed pass");
    println!("# could actually derive from a vector min/max.");
    println!();

    let mut luma = (vec![0u64; 33], vec![0u64; 33]);
    let mut chroma = (vec![0u64; 33], vec![0u64; 33]);
    for frame in 0..2 {
        accumulate(&mut luma, occupancy(&synthetic_luma(640, 352, frame), 16));
        accumulate(
            &mut chroma,
            occupancy(&synthetic_chroma(320, 176, frame, 0), 8),
        );
        accumulate(
            &mut chroma,
            occupancy(&synthetic_chroma(320, 176, frame, 7), 8),
        );
    }
    println!("## synthetic 640x352, the content hevc_encode_640x352_reconstruct searches");
    report("luma 16x16 distinct bands", &luma.0);
    report("luma 16x16 band range", &luma.1);
    report("chroma 8x8 distinct bands", &chroma.0);
    report("chroma 8x8 band range", &chroma.1);
    println!();

    let mut real = (vec![0u64; 33], vec![0u64; 33]);
    for plane in decoded_luma(24) {
        accumulate(&mut real, occupancy(&plane, 16));
    }
    println!("## bundled 1920x1080 sample, decoded luma");
    report("luma 16x16 distinct bands", &real.0);
    report("luma 16x16 band range", &real.1);
}
