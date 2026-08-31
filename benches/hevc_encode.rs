//! Criterion benchmarks for zvidlib's pure-Rust HEVC encoder.
//!
//! Two axes, both required to read an encoder number correctly:
//!
//! * **Whole-frame versus per-stage.** The whole-frame groups encode a
//!   synthetic sequence through the public
//!   [`zvidlib::native_hevc_video_encoder_factory`] and report frames/sec and
//!   megapixels/sec. The per-stage groups time the pipeline's individual stages
//!   through [`zvidlib::hevc_encoder_bench`], so the mode-search cost — which
//!   dominates — is not mistaken for bitstream-writing cost.
//! * **Instruction set.** Every group runs once per entry in
//!   `zvidlib::simd::available()` through `support::isa::bench_across_isas`,
//!   which pins the crate-wide override, asserts it reached every dispatch
//!   family, and checks each arm is bit-exact with scalar before timing it.
//!
//! Inputs are the synthetic sequences from `benches/support`, never a decoded
//! file: decoding first would fold decoder cost into the encoder numbers.
//!
//! This is a second bench target rather than more groups in `benches/codec.rs`.
//! That target is one process so its decoded-fixture cache is paid once; these
//! groups touch none of those fixtures, and the encoder's mode search is slow
//! enough that keeping it out of the default `--bench codec` run is worth more
//! than sharing a process with it.
//!
//! ## The SIMD axis, and where it is expected to read flat
//!
//! The encoder has three SIMD dispatch families of its own: `hevc_rdcost`,
//! the SAD and SATD distortion metrics the mode search calls;
//! `hevc_fwd_transform_quant`, the forward transform's butterfly and the
//! quantization loop; and `hevc_recon`, the §8.6.6 reconstruction loop and the
//! encode-side §8.7.3 SAO parameter search. Bitstream writing, CABAC, and the
//! RGBA-to-YUV420 conversion have no vector path, so their arms are expected to
//! read the same under every instruction set. That is the measured result the
//! issue asks for, not a broken benchmark: it says the next encoder-side
//! vectorization target is entropy coding or color conversion, and it is why
//! every group asserts through `simd::active_by_site()` that the override
//! landed rather than inferring it from the clock.
//!
//! The exception is the reconstruction group. It runs `hevc_recon` over every
//! partition and every CTB, and then the decoder's own vectorized §8.7.2
//! deblocking and §8.7.3 SAO filter kernels over the result, so it is the one
//! encoder-side group outside mode search that is expected to move with the
//! instruction set.
//!
//! ## Stages this encoder does not have yet
//!
//! Every stage the tracking issue names is now benchmarked, but the encoder's
//! access-unit writer is still a lossless PCM bootstrap writer: it does not
//! call the forward transform + quantization stage or the encode-side
//! reconstruction path, both of which are measured here directly rather than
//! through it. [`report_absent_stages`] prints that on every run so a group is
//! never read as covering more of the encoder than it does.

mod support;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::hevc_encoder_bench as encoder_bench;
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, VideoDimensions, VideoEncoderConfig, VideoEncoderFactory,
    VideoFrame, native_hevc_video_encoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas};

/// Environment variable that opts into the 1080p-scale groups.
///
/// Mode search over a 1920x1088 picture is seconds of work per iteration, so a
/// default `cargo bench` would take minutes. Keeping it opt-in leaves the
/// default run usable in an ordinary edit loop.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The small resolution both the whole-frame and per-stage groups always run.
///
/// The encoder's PCM writer requires dimensions divisible by 16, so the issue's
/// nominal 640x360 is not encodable; 640x352 is the nearest valid size and the
/// same 640-wide scale.
const SMALL: (u32, u32) = (640, 352);

/// The large resolution, behind [`LARGE_GROUP_ENV`].
///
/// 1920x1080 is likewise not divisible by 16 vertically, so this is the
/// 1080p-class size the encoder actually accepts (1088 = 68 CTB rows of 16).
const LARGE: (u32, u32) = (1920, 1088);

/// Frames per whole-frame iteration. Small enough that one criterion sample of
/// the 1080p-class group stays in the seconds range.
const WHOLE_FRAME_FRAMES: usize = 2;

/// Prints the pipeline stages the tracking issue names that this encoder does
/// not implement yet.
///
/// A benchmark suite reports what it measured; the stages it *could not* measure
/// have to be reported too, or their absence reads as zero cost.
fn report_absent_stages(_: &mut Criterion) {
    println!(
        "# hevc_encode: every stage the tracking issue names is benchmarked below: mode\n\
         # search/RDO, forward transform + quantization, encode-side reconstruction + in-loop\n\
         # filtering, CABAC + bitwriting, whole-picture PCM access-unit writing, and the\n\
         # RGBA8->YUV420 input conversion. The access-unit writer is still a lossless PCM\n\
         # writer, so it reaches neither the transform/quantization nor the reconstruction\n\
         # stage; both are measured directly instead.\n\
         # hevc_encode: hevc_rdcost, hevc_fwd_transform_quant and hevc_recon are the encoder's\n\
         # three SIMD dispatch families, so apart from the mode-search, forward-transform and\n\
         # reconstruction groups (the last also running the decoder's vectorized in-loop filter\n\
         # kernels) the arms are expected to read flat across instruction sets, which is the\n\
         # measured result, not a broken bench."
    );
}

/// A YUV420 picture, its planes already separated the way the encoder's later
/// stages take them.
struct Planes {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: usize,
    height: usize,
}

/// Two consecutive synthetic pictures at `(width, height)`, converted once.
///
/// The second picture's luma is the reference the inter mode search predicts
/// from, so both come from the same moving-gradient sequence rather than being
/// unrelated frames.
fn planes_pair(width: u32, height: u32) -> (Planes, Vec<u8>) {
    let frames = support::synthetic_yuv420_sequence(width, height, 2);
    let extract = |frame: &VideoFrame| Planes {
        y: frame.planes[0].data.clone(),
        cb: frame.planes[1].data.clone(),
        cr: frame.planes[2].data.clone(),
        width: width as usize,
        height: height as usize,
    };
    (extract(&frames[1]), frames[0].planes[0].data.clone())
}

/// Both pictures of the synthetic pair, all three planes each.
///
/// [`planes_pair`] keeps only the reference's luma, which is all the mode
/// search needs; reconstruction predicts chroma too, so it needs the whole
/// reference picture.
fn planes_sequence(width: u32, height: u32) -> (Planes, Planes) {
    let frames = support::synthetic_yuv420_sequence(width, height, 2);
    let extract = |frame: &VideoFrame| Planes {
        y: frame.planes[0].data.clone(),
        cb: frame.planes[1].data.clone(),
        cr: frame.planes[2].data.clone(),
        width: width as usize,
        height: height as usize,
    };
    (extract(&frames[0]), extract(&frames[1]))
}

/// Whether the 1080p-class groups were opted into.
fn large_enabled(group: &str) -> bool {
    if std::env::var_os(LARGE_GROUP_ENV).is_some() {
        return true;
    }
    println!("# skipping {group}; set {LARGE_GROUP_ENV}=1 to run it");
    false
}

/// The encoder configuration the whole-frame groups measure.
fn encoder_config(width: u32, height: u32) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions: VideoDimensions::new(width, height, &Limits::default())
            .expect("benchmark dimensions are valid"),
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        // Measure the crate's own encode work, not whichever fixed-function
        // block the host happens to ship.
        hardware: HardwarePreference::Avoid,
        timescale: 90_000,
        frame_duration: 3_000,
        configuration: Vec::new(),
    }
}

/// Encodes a fixed-length synthetic sequence through the public encoder, once
/// per instruction set.
fn whole_frame(criterion: &mut Criterion, (width, height): (u32, u32), group: &str) {
    let frames = support::synthetic_rgba8_sequence(width, height, WHOLE_FRAME_FRAMES);
    let configuration = encoder_config(width, height);
    let workload = IsaWorkload::new(
        group,
        FrameWork::new(
            WHOLE_FRAME_FRAMES as u64,
            u64::from(width),
            u64::from(height),
        ),
    );
    bench_across_isas(criterion, &workload, || {
        let mut encoder = native_hevc_video_encoder_factory()
            .create(&configuration, &Limits::default())
            .expect("the software HEVC encoder is constructible");
        let mut bitstream = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let source = FrameSource::Cpu(CpuFrameSource {
                frame,
                orientation: Orientation::TopLeft,
            });
            for sample in support::block_on(encoder.encode(FrameIndex(index as u64), source))
                .expect("the synthetic sequence encodes")
            {
                bitstream.extend_from_slice(&sample.data);
            }
        }
        for sample in support::block_on(encoder.finish()).expect("the encoder flushes") {
            bitstream.extend_from_slice(&sample.data);
        }
        bitstream
    });
}

/// Mode search / RDO, the stage the encoder spends most of its time in and the
/// only one the SIMD override can reach.
///
/// Run twice: with no reference picture (intra decisions only) and against the
/// previous picture (the coarse whole-pel inter search, which is where the SAD
/// and SATD kernels do the bulk of their work).
fn mode_search(criterion: &mut Criterion, size: (u32, u32), group_prefix: &str) {
    let (current, reference) = planes_pair(size.0, size.1);
    let work = FrameWork::new(1, u64::from(size.0), u64::from(size.1));
    let config = zvidlib_rdo_defaults();

    let intra_name = format!("{group_prefix}_rdo_intra");
    let intra = IsaWorkload::new(&intra_name, work);
    bench_across_isas(criterion, &intra, || {
        encoder_bench::rdo_decide_picture(
            &current.y,
            current.width,
            current.width,
            current.height,
            None,
            config.0,
            config.1,
        )
    });

    let inter_name = format!("{group_prefix}_rdo_inter");
    let inter = IsaWorkload::new(&inter_name, work);
    bench_across_isas(criterion, &inter, || {
        encoder_bench::rdo_decide_picture(
            &current.y,
            current.width,
            current.width,
            current.height,
            Some(&reference),
            config.0,
            config.1,
        )
    });
}

/// The QP and whole-pel search radius the encoder itself uses, so the benchmark
/// measures the shipped search rather than a tuned one.
fn zvidlib_rdo_defaults() -> (i32, i32) {
    (26, 4)
}

/// Encode-side reconstruction and in-loop filtering.
///
/// The stage that makes the encoder predict from what a decoder will hold
/// rather than from the source it was handed: every coded block is
/// reconstructed (predict + add residual), then the §8.7.3 SAO parameters are
/// searched and the §8.7.2 deblocking filter and §8.7.3 SAO run over the whole
/// picture through the decoder's own kernels. The reconstruction loop and the
/// SAO search dispatch through `hevc_recon` and the filters through
/// `hevc_prediction_filters`, which is why this is the one non-mode-search
/// encoder group that can show an instruction-set delta.
///
/// The mode-search plan the reconstruction consumes is built in setup, not in
/// the timed loop — mode search costs an order of magnitude more than
/// everything else and would swamp the measurement.
///
/// Both loop filters are enabled here. The shipped writer emits an access unit
/// that neutralizes them (`pcm_loop_filter_disabled_flag == 1`), which is what
/// keeps its PCM encode exactly lossless; the filters are what this group
/// exists to measure, so it models the access unit that leaves them on.
fn reconstruct(criterion: &mut Criterion, size: (u32, u32), group_prefix: &str) {
    let (reference, current) = planes_sequence(size.0, size.1);
    let config = zvidlib_rdo_defaults();
    let workload = encoder_bench::plan_reconstruct(
        &current.y,
        &current.cb,
        &current.cr,
        current.width,
        current.height,
        Some((&reference.y, &reference.cb, &reference.cr)),
        config.0,
        config.1,
    );

    let name = format!("{group_prefix}_reconstruct");
    let isa_workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(size.0), u64::from(size.1)),
    );
    bench_across_isas(criterion, &isa_workload, || {
        encoder_bench::reconstruct_encoded_picture(&workload, true, true)
    });
}

/// Whole-picture bitstream writing: parameter sets, slice header, and the
/// CABAC-coded CU syntax carrying the PCM samples.
fn pcm_write(criterion: &mut Criterion, size: (u32, u32), group_prefix: &str) {
    let (current, _) = planes_pair(size.0, size.1);
    let name = format!("{group_prefix}_pcm_write");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(size.0), u64::from(size.1)),
    );
    bench_across_isas(criterion, &workload, || {
        encoder_bench::write_idr_pcm_access_unit(
            &current.y,
            &current.cb,
            &current.cr,
            current.width,
            current.height,
        )
    });
}

/// The forward transform and quantization stage, over every transform size.
///
/// One iteration transforms and quantizes the whole picture four times, once
/// per 4x4 / 8x8 / 16x16 / 32x32 block size, so all four §8.6.4.2 matrices and
/// the 4x4 DST-VII the intra path selects are covered. This is the encoder's
/// second SIMD dispatch family, so its arms are expected to move with the
/// instruction set the way the mode-search groups do.
fn fwd_transform_quant(criterion: &mut Criterion, size: (u32, u32), group_prefix: &str) {
    let (current, _) = planes_pair(size.0, size.1);
    let name = format!("{group_prefix}_fwd_transform_quant");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(size.0), u64::from(size.1)),
    );
    let qp = zvidlib_rdo_defaults().0;
    bench_across_isas(criterion, &workload, || {
        encoder_bench::fwd_transform_quant_picture(
            &current.y,
            current.width,
            current.width,
            current.height,
            qp,
        )
    });
}

/// Bins per CABAC iteration. Roughly the CU-syntax bin count of a small
/// picture, large enough that per-call setup is negligible.
const CABAC_BINS: usize = 1 << 18;

/// Distinct context models the bin sequence cycles through.
const CABAC_CONTEXTS: usize = 64;

/// Syntax elements per bitwriter iteration.
const BITWRITER_VALUES: usize = 1 << 16;

/// The arithmetic coder and the raw bitwriter, isolated from picture data.
///
/// `pcm_write` above measures them as a picture-shaped whole; these two groups
/// separate the CABAC engine's per-bin cost from the fixed-length / `ue(v)` /
/// `se(v)` writers the parameter sets and slice headers go through.
fn entropy_coding(criterion: &mut Criterion) {
    let bins: Vec<u8> = (0..CABAC_BINS)
        .map(|i| ((i * 2_654_435_761) >> 13) as u8)
        .collect();
    let values: Vec<u32> = (0..BITWRITER_VALUES)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) >> 11)
        .collect();

    // These two workloads are bin and syntax-element streams, not pictures.
    // Counting bins and values as the elements makes criterion's `elem/s` line
    // read directly as bins/sec and syntax elements/sec, and makes the harness's
    // "Mpx/s" line read as millions of bins or values per second — the unit that
    // is actually meaningful here.
    let cabac = IsaWorkload {
        sample_size: 20,
        ..IsaWorkload::new("hevc_encode_cabac", FrameWork::new(CABAC_BINS as u64, 1, 1))
    };
    bench_across_isas(criterion, &cabac, || {
        encoder_bench::cabac_encode_bins(&bins, CABAC_CONTEXTS)
    });

    let bitwriter = IsaWorkload {
        sample_size: 20,
        ..IsaWorkload::new(
            "hevc_encode_bitwriter",
            FrameWork::new(BITWRITER_VALUES as u64, 1, 1),
        )
    };
    bench_across_isas(criterion, &bitwriter, || {
        encoder_bench::bitwriter_write_syntax(&values)
    });
}

/// The RGBA8 to YUV420 conversion every encoded frame pays before mode search.
///
/// Not one of the stages the tracking issue lists, and not a SIMD dispatch site,
/// but it is real per-frame encoder cost: without it the per-stage groups do not
/// add up to the whole-frame number.
fn color_conversion(criterion: &mut Criterion, size: (u32, u32), group_prefix: &str) {
    let frame = support::synthetic_rgba8_sequence(size.0, size.1, 1).remove(0);
    let name = format!("{group_prefix}_rgba_to_yuv420");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(size.0), u64::from(size.1)),
    );
    bench_across_isas(criterion, &workload, || {
        let (y, cb, cr) = encoder_bench::rgba_to_yuv420_planes(&frame);
        let mut out = y;
        out.extend_from_slice(&cb);
        out.extend_from_slice(&cr);
        out
    });
}

fn hevc_encode_small(criterion: &mut Criterion) {
    whole_frame(criterion, SMALL, "hevc_encode_640x352");
}

fn hevc_encode_large(criterion: &mut Criterion) {
    if !large_enabled("the 1920x1088 whole-frame encode group") {
        return;
    }
    whole_frame(criterion, LARGE, "hevc_encode_1920x1088");
}

fn hevc_encode_stages_small(criterion: &mut Criterion) {
    mode_search(criterion, SMALL, "hevc_encode_640x352");
    fwd_transform_quant(criterion, SMALL, "hevc_encode_640x352");
    reconstruct(criterion, SMALL, "hevc_encode_640x352");
    pcm_write(criterion, SMALL, "hevc_encode_640x352");
    color_conversion(criterion, SMALL, "hevc_encode_640x352");
}

fn hevc_encode_stages_large(criterion: &mut Criterion) {
    if !large_enabled("the 1920x1088 per-stage encoder groups") {
        return;
    }
    mode_search(criterion, LARGE, "hevc_encode_1920x1088");
    fwd_transform_quant(criterion, LARGE, "hevc_encode_1920x1088");
    reconstruct(criterion, LARGE, "hevc_encode_1920x1088");
    pcm_write(criterion, LARGE, "hevc_encode_1920x1088");
    color_conversion(criterion, LARGE, "hevc_encode_1920x1088");
}

criterion_group!(
    benches,
    report_absent_stages,
    hevc_encode_small,
    hevc_encode_stages_small,
    entropy_coding,
    hevc_encode_large,
    hevc_encode_stages_large
);
criterion_main!(benches);
