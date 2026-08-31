//! Benchmark-only access to the HEVC encoder's individual pipeline stages.
//!
//! `crate::hevc` is a private module, and criterion benchmarks are a separate
//! crate, so the per-stage encoder groups in `benches/hevc_encode.rs` cannot
//! reach [`rdo::decide_picture`], the CABAC encoding engine, or the PCM
//! bitstream writer through the public API. The public
//! [`crate::native_hevc_video_encoder_factory`] runs all of them at once, which
//! is exactly what a per-stage breakdown must avoid.
//!
//! This module is that access, and nothing more: thin wrappers that own their
//! inputs, return plain bytes, and add no logic the benchmark could accidentally
//! measure instead of the encoder. It is `#[doc(hidden)]` and explicitly not
//! part of the stable API, matching the `crate::hevc::engine` convention of
//! exposing internals for tests and fuzzing without promising them.
//!
//! Each wrapper returns the bytes that identify its result, because
//! `benches/support/isa.rs` compares those bytes across instruction sets before
//! timing anything: a stage whose return value did not depend on the kernels
//! under test would silently disarm that guard.

use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::pcm::encode_idr_pcm_au;
use crate::hevc::engine::encoder::rdo::{DecisionConfig, decide_picture};
use crate::hevc::engine::cabac::ContextModel;

/// Runs the encoder's mode-search / RDO stage over one luma picture.
///
/// This is the stage that reaches `hevc_rdcost`, the crate's only
/// encoder-side SIMD dispatch family, through its SAD and SATD distortion
/// metrics.
///
/// `reference_y`, when present, enables the coarse whole-pel inter search;
/// passing `None` measures the intra-only decision cost. Returns the search's
/// decisions serialized in traversal order, so the bit-exactness guard covers
/// every partition's motion vector and cost rather than only the picture total.
#[must_use]
pub fn rdo_decide_picture(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&[u8]>,
    qp: i32,
    search_radius: i32,
) -> Vec<u8> {
    let decision = decide_picture(
        y,
        stride,
        width,
        height,
        reference_y,
        DecisionConfig {
            qp,
            search_radius,
            ..DecisionConfig::default()
        },
    );
    let mut out = Vec::with_capacity(decision.blocks.len() * 32 + 16);
    out.extend_from_slice(&decision.rd_cost.to_le_bytes());
    out.extend_from_slice(&(decision.pcm_blocks as u64).to_le_bytes());
    for block in &decision.blocks {
        out.extend_from_slice(&(block.x as u32).to_le_bytes());
        out.extend_from_slice(&(block.y as u32).to_le_bytes());
        out.extend_from_slice(&(block.size as u32).to_le_bytes());
        out.extend_from_slice(&block.rd_cost.to_le_bytes());
        out.extend_from_slice(&block.pcm_cost.to_le_bytes());
        for partition in &block.partitions {
            out.extend_from_slice(&(partition.x as u32).to_le_bytes());
            out.extend_from_slice(&(partition.y as u32).to_le_bytes());
            out.extend_from_slice(&(partition.w as u32).to_le_bytes());
            out.extend_from_slice(&(partition.h as u32).to_le_bytes());
            out.extend_from_slice(&partition.mv_x.to_le_bytes());
            out.extend_from_slice(&partition.mv_y.to_le_bytes());
            out.extend_from_slice(&partition.sad.to_le_bytes());
            out.extend_from_slice(&partition.satd.to_le_bytes());
            out.extend_from_slice(&partition.bit_cost.to_le_bytes());
            out.extend_from_slice(&partition.rd_cost.to_le_bytes());
        }
    }
    out
}

/// Writes one IDR access unit for a YUV420 picture: parameter sets, slice
/// header, and the CABAC-coded CU syntax carrying the PCM samples.
///
/// This is the encoder's entropy-coding and bitwriting stage as it actually
/// runs today. The returned Annex B access unit is the stage's own output, so
/// timing it measures the writer and the bit-exactness guard covers the
/// bitstream itself.
///
/// # Panics
///
/// Panics if the picture does not satisfy the PCM writer's requirements
/// (dimensions divisible by 16 and correctly sized planes), which a benchmark
/// input always does.
#[must_use]
pub fn write_idr_pcm_access_unit(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    encode_idr_pcm_au(y, cb, cr, width, height).expect("benchmark pictures are writable as PCM")
}

/// Drives the §9.3.5 CABAC arithmetic encoder over a deterministic bin
/// sequence, isolated from any picture data.
///
/// `bins` alternates context-coded and bypass bins so both `encode_decision`
/// (with its context-model state transition) and `encode_bypass` are exercised;
/// `contexts` sets how many distinct context models the sequence cycles
/// through, which is what makes the state transitions non-degenerate. The
/// returned bytes are the arithmetic codeword, so this measures the CABAC
/// engine plus the [`BitWriter`] it writes through.
#[must_use]
pub fn cabac_encode_bins(bins: &[u8], contexts: usize) -> Vec<u8> {
    assert!(contexts > 0, "the bin sequence needs at least one context");
    // §9.3.2.2 initialization at a representative slice QP. Spreading the
    // `initValue` entries keeps the models from all starting in the same state,
    // which would make the state transitions degenerate.
    let mut models: Vec<ContextModel> = (0..contexts)
        .map(|index| ContextModel::init(154 - (index % 32) as u8 * 2, 26))
        .collect();
    let mut writer = BitWriter::new();
    let mut cabac = CabacEncoder::new();
    for (index, &bin) in bins.iter().enumerate() {
        if index % 2 == 0 {
            cabac.encode_decision(&mut writer, &mut models[index % contexts], bin & 1);
        } else {
            cabac.encode_bypass(&mut writer, bin & 1);
        }
    }
    cabac.encode_terminate(&mut writer, 1);
    writer.finish()
}

/// Writes a deterministic syntax-element sequence through the raw
/// [`BitWriter`], with no arithmetic coding on top.
///
/// The parameter sets and slice headers of every access unit go through these
/// fixed-length, `ue(v)` and `se(v)` writers, so measuring them separately
/// separates raw bitwriting cost from the CABAC engine's.
#[must_use]
pub fn bitwriter_write_syntax(values: &[u32]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    for (index, &value) in values.iter().enumerate() {
        match index % 3 {
            0 => writer.put_bits(value & 0xffff, 16),
            1 => writer.ue(value),
            _ => writer.se(value as i32 - (u32::MAX / 2) as i32),
        }
    }
    writer.rbsp_trailing_bits();
    writer.finish()
}

/// Converts one RGBA8 frame to the YUV420 planes the encoder's later stages
/// consume.
///
/// The public encoder takes RGBA8 input, so this conversion runs once per frame
/// ahead of mode search and bitstream writing. It is not one of the stages the
/// SIMD override reaches, and measuring it separately is what makes the
/// whole-frame number decomposable into stages that add up.
///
/// # Panics
///
/// Panics unless `frame` is an RGBA8 frame, which a benchmark input always is.
#[must_use]
pub fn rgba_to_yuv420_planes(frame: &crate::VideoFrame) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    super::encoder::rgba_to_yuv420(frame, crate::Orientation::TopLeft)
        .expect("benchmark frames are RGBA8")
}
