//! Runtime-dispatched SIMD kernels for the §8.5.3.3 inter-prediction hot
//! loops (`crate::hevc::engine::inter_pred`).
//!
//! Two primitives cover every vectorizable loop in the fractional-sample
//! interpolation and weighted-sample-prediction processes:
//!
//! * [`filter_taps`] — the separable filter accumulation
//!   `out[i] = ( Σ coeff[t] · tap[t][i] ) >> shift`. The §8.5.3.3.3.2
//!   8-tap luma and §8.5.3.3.3.3 4-tap chroma filters both reduce to it,
//!   for the horizontal pass (the taps are overlapping windows of one
//!   source row) and for the vertical pass (the taps are consecutive rows
//!   of the intermediate buffer) alike.
//! * [`combine_weighted`] — the sample combine
//!   `out[i] = Clip3( 0, max, ( ( Σ w[t] · tap[t][i] ) + round ) >> shift
//!   + post )`. The §8.5.3.3.4.2 default uni-/bi-predictive average and
//!   the §8.5.3.3.4.3 explicit weighted combine are both instances of it.
//!
//! Each primitive has an SSE4.1 and an AVX2 implementation on `x86_64`, a
//! NEON implementation on `aarch64`, and a scalar implementation used as
//! the fallback everywhere else (including `wasm32`). The backend is
//! selected once per process by [`detected_isa`] from the runtime CPU
//! feature flags; passing an [`Isa`] the running CPU does not support to
//! any kernel silently falls back to the scalar path, so the `_with`
//! entry points are safe to call with any value.
//!
//! Every backend is **bit-exact** with the scalar one: the accumulation
//! is plain wrapping-free `i32` arithmetic in the same order, and the
//! shifts are arithmetic right shifts. The interpolation intermediates
//! stay far inside `i32` for all supported bit depths (the widest case,
//! 16-bit two-dimensional luma, peaks near 2^20), and the explicit
//! weighted combine is only vectorized when a bound check proves the
//! `i32` product cannot overflow — otherwise it stays on the `i64`
//! scalar path (see `inter_pred`).
//!
//! # What each backend is actually worth
//!
//! The scalar implementations here are not a slow reference kept only for
//! portability: under the crate's `lto = "fat"` / `codegen-units = 1`
//! release profile LLVM auto-vectorizes them, so a hand-written kernel
//! has to beat *vectorized* scalar code, not a sample-at-a-time loop.
//! Where it cannot, the dispatch prefers scalar rather than pretending
//! otherwise. Measured with the `simd_inter_pred_benchmark` and
//! `in_loop::tests::bench_in_loop_filters` benchmarks, each reporting the
//! best of five interleaved rounds, on Apple silicon (NEON) and on three
//! AVX2-capable x86_64 hosts: an AMD EPYC-class `ubuntu-latest` runner
//! (Zen 3/4), an Intel Coffee Lake `macos-15-intel` runner (i7-8700B), and
//! an Intel Emerald Rapids Xeon Platinum 8573C (family 6 model 207, drawn
//! from the same `ubuntu-latest` pool; `sse4_1` + `avx2` + `avx512f`). The
//! x86_64 columns give **AMD / Coffee Lake / Emerald Rapids**, because the
//! two Intel generations do not agree with each other — see below. The two
//! [`filter_taps`] buffer rows were re-taken for issue #321 on a dozen further
//! `ubuntu-latest` draws (EPYC 7763, EPYC 9V74, Xeon Platinum 8573C, Xeon
//! 6973P-C) and a fresh `macos-15-intel` one; the AMD column there spans Zen 3
//! and Zen 4, and the Intel one the 8573C. The **whole Coffee Lake column** was
//! then re-taken for issue #326 on five independent `macos-15-intel` draws of
//! the i7-8700B, three runs each, on 2026-09-01 under rustc 1.98.0; thirteen of
//! its sixteen figures reproduced and the three [`combine_weighted`] ones did
//! not. The column below is that re-take — see "The Coffee Lake column, re-taken".
//!
//! | kernel | SSE4.1 (AMD / CFL / EMR) | AVX2 (AMD / CFL / EMR) | NEON |
//! | --- | --- | --- | --- |
//! | §8.5.3.3.3.2 8-tap luma [`filter_taps`] (block path) | 2.1-2.3x / 1.6-1.8x / 1.9-2.0x | 2.3-2.7x / 1.8-1.9x / 2.3-2.5x | 1.6-1.9x |
//! | §8.5.3.3.3.3 4-tap chroma [`filter_taps`] (block path) | 1.4-1.6x / 1.5x / 1.45-1.50x | 1.5-1.6x / 1.4-1.6x / 1.5-1.6x | 1.5-1.7x |
//! | [`filter_taps`] (one long L1-resident buffer) | 1.1-1.4x / 0.88-1.01x / 0.85-0.87x | 2.5-2.8x / 1.7-1.9x / 1.5-1.7x | ~1.0x |
//! | [`filter_taps`] (same buffer, coefficients opaque) | 3.3x / 0.90-0.98x / 2.3x | 6.0-7.1x / 1.7-1.8x / 4.4-4.7x | ~1.0x |
//! | §8.5.3.3.4 [`combine_weighted`] (block path) | 0.95x / ~1.0x / ~1.0x — dispatched to scalar | 1.4x / ~1.0x / 1.13-1.25x | 0.91x — dispatched to scalar |
//! | §8.5.3.3.4 [`combine_weighted`] (L1-resident buffer) | ~1.0x / ~1.0x / ~1.0x | 2.0-2.2x / 2.0-2.2x / 1.5-1.6x | 0.91x |
//! | §8.7.2 `in_loop::filter_luma_rows` / `filter_chroma_rows` | 1.2-1.3x / 1.26-1.36x / 1.24-1.29x | 1.2-1.3x / 1.29-1.45x / 1.26-1.29x | ~1.3x |
//! | §8.7.3 `in_loop::sao_band_row` / `sao_edge_row` | 4.6-5.4x / 2.6-3.1x / 4.2-4.5x | 6.3-7.4x / 3.0-3.4x / 4.2-4.5x | ~2.3x |
//!
//! # What the microarchitecture split does and does not change
//!
//! Every kernel is on the same side of 1.00x on all three x86_64 hosts, so
//! no dispatch decision moves: what differs is how much each win is worth.
//! The spread is a *microarchitecture* spread, not a vendor one. Coffee Lake
//! reads lower than AMD Zen on the *filter* kernels, which are bounded by
//! vector width — §8.7.3 SAO is 2.6-3.4x there against 4.6-7.4x on Zen, and
//! AVX2 [`filter_taps`] over a buffer is 1.7-1.9x against 2.5-2.8x — while
//! §8.7.2 deblocking, which is bounded by shape rather than by width, is the
//! one row where it reads slightly *higher*. It is **not** lower on
//! everything bounded by vector width: AVX2 [`combine_weighted`] over a
//! buffer reads 2.0-2.2x there, level with Zen, which is the correction
//! issue #326 made to this paragraph. Emerald Rapids sits between the two on
//! most rows and alongside Zen on several (SAO 4.2-4.5x, two-dimensional
//! 8-tap luma 2.3-2.5x on AVX2), which is what makes "Intel" the wrong axis
//! to read the Coffee Lake figures on.
//!
//! The kernel that made this worth measuring is the AVX2
//! [`combine_weighted`] one, and its story is now shorter than it was.
//! #224 read it at 1.4x in the block path and 2.0-2.2x on a bare buffer on
//! Zen but at 0.93-0.95x and 0.92-0.93x on Coffee Lake, and #301 explained
//! that gap as a Skylake-family cost: 256-bit `vpmulld` is two uops on
//! Skylake-derived cores against the four-lane `pmulld` LLVM auto-vectorizes
//! the scalar loop into, and a single uop from Ice Lake onward. **The Coffee
//! Lake half of that gap does not reproduce.** Re-measured for #326 over
//! five `macos-15-intel` draws with the kernel unchanged since #127, the
//! same arms read ~1.00x in the block path and **2.0-2.2x** on a bare
//! buffer — level with Zen, and the eight-lane kernel simply halving a
//! four-lane scalar loop (0.28-0.41 ms against 0.57-0.82 ms). There is no
//! Skylake-family regression left to explain; the `vpmulld` argument is
//! kept only as the record of a figure that no longer stands.
//!
//! So the kernel is dispatched to unconditionally on every AVX2 host
//! because all three measured microarchitectures profit — Zen and Coffee
//! Lake at 2.0-2.2x on a buffer, Emerald Rapids at 1.5-1.6x — rather than
//! in spite of one that did not. The SSE4.1 combine, by contrast, is
//! confirmed at or below scalar on all three (0.95x on Zen, ~1.00x on both
//! arms on Coffee Lake and on Emerald Rapids), so its dispatch to scalar
//! holds unconditionally and there is no SSE4.1 kernel to keep.
//!
//! # The SSE4.1 buffer row measures the optimizer, not the kernel
//!
//! The row that looked as though it changed side is SSE4.1 [`filter_taps`]
//! over the long L1-resident buffer: 1.1-1.4x on Zen against 0.85-0.87x on
//! an Intel server core carrying `avx512f`. Issue #321 ran it down, and the
//! cause is in the *baseline*, not in the kernel — the arm was never
//! comparing two implementations of the same computation.
//!
//! The crate sets no `-C target-cpu` anywhere, so every host executes the
//! same baseline-`x86-64` machine code and no scalar loop in this module is
//! ever compiled to AVX-512 or to 256-bit form. That disposes of the
//! leading hypothesis by inspection; what differs between the arms is
//! *which* scalar loop is being timed. The benchmark passes the
//! compile-time literal `LUMA_FILTER[2]`, and [`filter_taps`] is `#[inline]`,
//! so its scalar arm inlines into the timing loop with all eight
//! coefficients known. LLVM then compiles the reference to something
//! §8.5.3.3.3 can never call: the ±1 taps become `paddd` / `psubd`, the 4
//! taps `pslld $2`, and — because `[-1, 4, -11, 40, 40, -11, 4, -1]` is
//! symmetric — the two `-11` taps and the two `40` taps are summed *before*
//! multiplying, leaving two vector multiplies per four output samples
//! (4 `pmuludq` plus 8 shuffle-class instructions, in a 31-instruction
//! loop). The SSE4.1 arm, by contrast, is a call to the shared
//! `filter_taps_sse41::<8>` instantiation, which cannot be specialized
//! because the block path passes a run-time `LUMA_FILTER[x_frac]`: it loads
//! the coefficients into eight splat registers and issues eight `pmulld`
//! and eight `paddd` per four samples. The row was timing a constant-folded,
//! symmetry-halved SSE2 loop against a general eight-multiply SSE4.1 kernel.
//!
//! The benchmark now times both. The second arm hides the coefficients
//! behind `black_box`, so the scalar baseline compiles to the same generic
//! form the block path calls, and it is the kernel-against-kernel
//! comparison. Across a dozen `ubuntu-latest` draws and a `macos-15-intel`
//! one, three runs of five interleaved rounds each, the *vector* kernels'
//! absolute times are identical between the two arms to the hundredth of a
//! millisecond; only the scalar baseline moves, and how far it moves is
//! exactly the host split:
//!
//! | host | scalar folded | scalar opaque | SSE4.1 folded / opaque |
//! | --- | ---: | ---: | --- |
//! | AMD EPYC 7763 (Zen 3) | 1.33 ms | 3.37 ms | 1.32x / 3.35x |
//! | AMD EPYC 9V74 (Zen 4) | 1.38 ms | 3.38 ms | 1.37-1.39x / 3.26-3.27x |
//! | Intel Xeon Platinum 8573C (Emerald Rapids) | 1.15-1.18 ms | 3.21-3.26 ms | 0.85x / 2.33-2.36x |
//! | Intel Xeon 6973P-C (`avx512f`) | 1.01 ms | 2.75 ms | 0.87x / 2.36x |
//! | Intel i7-8700B (Coffee Lake) | 1.22-1.76 ms | 1.22-1.75 ms | 0.88-1.01x / 0.90-0.98x |
//!
//! Folding the coefficients is worth 2.4-2.8x to the scalar loop on Zen and
//! on the newer Intel server cores, and nothing at all on Coffee Lake, whose
//! single shuffle port has to retire the `pshufd` / `punpckldq` traffic that
//! the folded loop's `pmuludq` pairs generate either way. That is the whole
//! of the apparent sign flip: the SSE4.1 kernel is not slower on the newer
//! Intel cores than on the older one — 1.35-1.40 ms on the 8573C against
//! 1.51 ms on the i7-8700B — the baseline it is divided by is 2.8x faster
//! there, in a way no §8.5.3.3.3 caller can reproduce. The Coffee Lake row
//! spans five draws rather than one (#326), which is why its range is the
//! widest in the table; what matters in it is that its two columns stay equal
//! draw by draw — across fifteen runs the folded and opaque scalar arms never
//! differ by more than 0.09 ms, against the 2.0 ms that separates them on
//! Zen. The 8573C itself (family 6 model 207) was drawn again for this and
//! reproduces #301's number exactly: 0.85x on the folded arm across three
//! runs, and 2.33-2.36x on the opaque one. A Xeon 6973P-C drawn from the same pool reads 0.87x
//! and 2.36x, so this is a property of that class of Intel core rather than
//! of one part number.
//!
//! None of this reaches a call shape §8.5.3.3.3 issues, so no dispatch
//! decision moves. `interp_luma_block_with` and `interp_chroma_block_with`
//! index the filter tables by a fractional position known only at run time,
//! so every production call lands on the same generic instantiation the
//! opaque arm times — where SSE4.1 reads 2.33-2.36x on the 8573C, against
//! 1.96-2.07x for the same kernel in that host's block path. The SSE4.1
//! kernel stays dispatched to unconditionally, and this row is not grounds
//! to revisit it a third time: it measured what the optimizer was allowed to
//! do to the reference, not what either kernel is worth.
//!
//! One recorded number moved under re-measurement and is not smoothed over:
//! #301 gave the Coffee Lake SSE4.1 buffer figure as 1.3x, and a fresh
//! `macos-15-intel` draw of the same i7-8700B reads 0.92-0.94x across three
//! runs, with 0.93-0.95x on the opaque arm — consistent with folding buying
//! that core nothing. The table carries the re-measured pair, widened by the
//! five-draw re-take below. Whether the rest of that column had drifted the
//! same way was issue #326, and the next section is the answer.
//!
//! # The Coffee Lake column, re-taken
//!
//! #301 recorded the Coffee Lake column from one `macos-15-intel` draw, and
//! #321 found two of the four figures it happened to re-touch disagreeing with
//! it in both directions. Issue #326 re-took the **whole** column: five
//! independent `macos-15-intel` draws of the i7-8700B (family 6, `Intel(R)
//! Core(TM) i7-8700B CPU @ 3.20GHz`), three runs of `simd_inter_pred_benchmark`
//! and `in_loop::tests::bench_in_loop_filters` each, best of five interleaved
//! rounds per run, on 2026-09-01 under rustc 1.98.0 (88d9e12ae). Fifteen runs
//! per figure.
//!
//! | row | as recorded | re-taken (15 runs, 5 draws) | |
//! | --- | --- | --- | --- |
//! | §8.5.3.3.3.2 8-tap luma, SSE4.1 / AVX2 | 1.6-1.7x / 1.8-1.9x | 1.63-1.77x / 1.79-1.94x | confirmed |
//! | §8.5.3.3.3.3 4-tap chroma, SSE4.1 / AVX2 | 1.5x / 1.4-1.5x | 1.47-1.54x / 1.37-1.56x | confirmed |
//! | [`filter_taps`] buffer, SSE4.1 / AVX2 | 0.92-0.94x / 1.7-1.8x | 0.88-1.01x / 1.66-1.85x | confirmed |
//! | [`filter_taps`] buffer opaque, SSE4.1 / AVX2 | 0.93-0.95x / 1.8x | 0.90-0.98x / 1.70-1.84x | confirmed |
//! | §8.7.2 deblocking, SSE4.1 / AVX2 | 1.3-1.4x / 1.3-1.4x | 1.26-1.36x / 1.29-1.45x | confirmed |
//! | §8.7.3 SAO, SSE4.1 / AVX2 | 2.7-2.8x / 3.0x | 2.62-3.08x / 2.98-3.36x | confirmed, wider |
//! | §8.5.3.3.4 [`combine_weighted`] block, SSE4.1 / AVX2 | ~1.0x / 0.93-0.95x | 0.99-1.10x / 0.98-1.05x | **AVX2 moved** |
//! | §8.5.3.3.4 [`combine_weighted`] buffer, SSE4.1 / AVX2 | 0.75x / 0.92-0.93x | 0.96-1.03x / 1.99-2.18x | **both moved** |
//!
//! Thirteen of the sixteen figures reproduced. Five draws see more spread
//! than one did, so several ranges widen at both ends — most of all SSE4.1
//! SAO, whose 2.7-2.8x becomes 2.62-3.08x — but none of them changes which
//! side of 1.00x the kernel is on or what the row says. Every figure that
//! moved in the sense that matters is a [`combine_weighted`] one, and the
//! AVX2 buffer arm moved by more than a factor of two, from a loss to a win.
//!
//! **It is the recorded figures that moved, not the harness or the toolchain.**
//! The same workflow, on the same day and the same rustc, drew an EPYC 9V74 and
//! a Xeon Platinum 8573C from the `ubuntu-latest` pool and reproduced their
//! recorded columns throughout: on the 9V74, SSE4.1 / AVX2 [`filter_taps`] over
//! a buffer at 1.34-1.35x / 2.49-2.52x and opaque at 3.19-3.23x / 6.05-6.10x,
//! AVX2 [`combine_weighted`] at 1.37-1.38x block and 2.02x buffer, SAO at
//! 5.28-5.30x / 7.22-7.28x; on the 8573C, 0.85x / 1.61-1.64x and 2.35-2.36x /
//! 4.52-4.59x, with AVX2 [`combine_weighted`] at 1.25-1.27x and 1.59-1.60x.
//! Those are the AMD and Emerald Rapids columns above. A harness or codegen
//! change would have moved them too.
//!
//! What the moved figures say is that Coffee Lake never had an AVX2
//! [`combine_weighted`] regression to explain. The kernel is untouched since
//! #127, and over a bare buffer it takes 0.28-0.41 ms against the scalar
//! loop's 0.57-0.82 ms — eight lanes halving four, exactly as on Zen. The
//! #224 reading of 0.92-0.93x is not reproducible on that host and is
//! superseded. In the block path both arms sit at ~1.00x, where the
//! per-block allocation dominates a kernel this small, which is what that
//! arm reads on every host.
//!
//! **No dispatch decision moves.** AVX2 [`combine_weighted`] was already
//! dispatched to unconditionally and now is so on a unanimous measurement
//! instead of a two-of-three one; the SSE4.1 combine is confirmed at ~1.00x
//! rather than 0.75x, which is still not a reason to write an SSE4.1 kernel,
//! so its dispatch to the scalar reference stands. Every other kernel keeps
//! the side of 1.00x it was recorded on. The one conclusion that had to be
//! corrected is prose, not dispatch: "Coffee Lake reads lower than AMD Zen on
//! everything bounded by vector width" was true of the figures as recorded and
//! is not true of the re-taken ones, since AVX2 [`combine_weighted`] over a
//! buffer now reads level with Zen.
//!
//! # The block-path rows are a *small*-block figure
//!
//! The two [`filter_taps`] block-path rows are measured over the small blocks
//! the §8.5.3.3 benchmark used to issue, and issue #280 measured what happens
//! as the block grows. On aarch64, all bi-predicted and luma only, the 8-tap
//! luma kernel reads **1.43x at 8x8, 1.25x at 16x16, 1.10x at 32x32 and 1.04x
//! at 64x64** — a monotonic walk from the block-path figure towards the buffer
//! one. That matters because it is the *large* end real content spends its
//! time at: 48 frames of the bundled 1080p sample put 62% of predicted luma
//! samples in 64x64 units and 31% in 32x32, so the mix reads 1.09x rather than
//! the 1.6-1.9x above. Read the block-path row as what the kernel is worth on a
//! small block, not as what it is worth to a decode; `benches/README.md` has
//! the sweep, the measured prediction-unit mix and the whole-frame accounting.
//!
//! #280 attributed that walk to the `w x ( h + 7 )` intermediate the
//! two-dimensional path materializes between its two passes, and issue #309
//! measured the two passes apart to check it. **It is not the intermediate.**
//! The horizontal pass alone decays 2.12x → 1.03x over the same sweep, and so
//! do the one-dimensional `x_frac == 0` / `y_frac == 0` phases, which build no
//! intermediate at all. `measure_filter_taps_by_row_length` below strips the
//! block walk, the allocation and the intermediate out entirely — one
//! L1-resident tap buffer, the same total sample count at every row length —
//! and still reproduces the whole decay: **3.84x at 4 samples per call, 3.21x
//! at 8, 2.06x at 16, 1.50x at 32, 1.33x at 64, 1.19x at 128 and 1.10x at
//! 256.** The variable is the per-call row length, which the block walk fixes
//! at the block width; at 64x64 the intermediate is 18 KiB and sits inside a
//! 128 KiB L1D, so it was never spilling. That is the mechanism the paragraph
//! below describes, now measured directly rather than inferred.
//!
//! The two [`filter_taps`] block-path rows and the buffer row are the same
//! kernel at different call sizes, and on NEON the difference is the point:
//! the win comes from the short 4..16-sample rows the block walk actually
//! issues, where the kernel's tight inner loop beats the scalar call's
//! per-invocation setup, and over one long buffer that setup amortizes away
//! and the two are level. On x86_64 the ordering is the other way around for
//! AVX2, whose eight lanes have more to gain the longer the run is — on AMD
//! clearly so (2.4x in the block path against 2.5-2.7x over a buffer, from
//! a much lower SSE4.1 buffer figure), and on both Intel hosts only
//! marginally or not at all, where the block and buffer arms read within a
//! tenth of each other on Coffee Lake and the block path reads *higher* on
//! Emerald Rapids.
//!
//! [`combine_weighted`] is the kernel that splits by vector *width* rather
//! than by architecture. At four lanes — SSE4.1 and NEON alike — it only
//! does what LLVM already does to the scalar loop, and measured at or below
//! it on every host: below on Zen in the block path, level on Zen on a bare
//! buffer and on both arms on Coffee Lake and Emerald Rapids — so both
//! dispatch to the scalar reference. At AVX2's eight lanes it is a real win
//! on all three x86_64 hosts and is kept. §8.7.2 deblocking is bounded by
//! shape rather than by codegen instead: a four-row edge segment is exactly
//! one 4-lane vector, so there is no width left for AVX2 to exploit, which
//! is why it deliberately runs the same 128-bit kernel as SSE4.1 and the two
//! measure identically on all three hosts; its §8.7.2.5.3 decisions stay
//! scalar.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::OnceLock;

/// The instruction-set backend a kernel runs on.
///
/// Only the variants that can exist on the target architecture are
/// compiled in; [`Isa::Scalar`] is always available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable scalar fallback — the bit-exactness reference.
    Scalar,
    /// x86_64 SSE4.1 (`pmulld` / `pminsd` / `pmaxsd`), 4 lanes.
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86_64 AVX2, 8 lanes.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AArch64 NEON, 4 lanes.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Detects the widest backend the running CPU supports, once per process.
///
/// NEON is architecturally mandatory on AArch64, so `aarch64` reports
/// [`Isa::Neon`] unconditionally; `x86_64` probes AVX2 then SSE4.1 at
/// runtime; every other target (including `wasm32`) reports
/// [`Isa::Scalar`].
///
/// A [`crate::simd::set_override`] override is consulted ahead of the cache on
/// every call, so pinning an instruction set still reaches these kernels after
/// detection has resolved.
#[must_use]
pub fn detected_isa() -> Isa {
    if let Some(isa) = overridden_isa() {
        return isa;
    }
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(detect)
}

/// Maps the crate-wide SIMD override, if any, onto this module's [`Isa`].
///
/// Variants the target architecture does not compile in collapse to
/// [`Isa::Scalar`]; [`crate::simd::set_override`] already refuses to pin an
/// instruction set the host cannot execute, so that arm is unreachable in
/// practice and only exists to keep the mapping total.
#[inline]
fn overridden_isa() -> Option<Isa> {
    use crate::simd::SimdIsa;
    Some(match crate::simd::override_isa()? {
        SimdIsa::Scalar => Isa::Scalar,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 => Isa::Sse41,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Avx2 => Isa::Avx2,
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => Isa::Neon,
        #[allow(unreachable_patterns)]
        _ => Isa::Scalar,
    })
}

fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            Isa::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            Isa::Sse41
        } else {
            Isa::Scalar
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Isa::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Isa::Scalar
    }
}

/// Every backend usable on this CPU, narrowest first, always starting
/// with [`Isa::Scalar`].
///
/// Intended for tests and benchmarks that want to exercise or time each
/// available backend against the scalar reference.
#[must_use]
pub fn available_isas() -> Vec<Isa> {
    let mut isas = vec![Isa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            isas.push(Isa::Sse41);
        }
        if is_x86_feature_detected!("avx2") {
            isas.push(Isa::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        isas.push(Isa::Neon);
    }
    isas
}

/// Fill `out` with `value`, on the requested backend.
#[inline]
pub fn fill_i32(isa: Isa, value: i32, out: &mut [i32]) {
    if !supported(isa) {
        return out.fill(value);
    }
    match isa {
        Isa::Scalar => out.fill(value),
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe { fill_i32_sse41(value, out) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { fill_i32_avx2(value, out) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { fill_i32_neon(value, out) },
    }
}

/// Generate `out[i] = (base + i * step + round) >> shift`.
#[inline]
pub fn affine_i32(isa: Isa, base: i32, step: i32, round: i32, shift: i32, out: &mut [i32]) {
    debug_assert!((0..32).contains(&shift));
    if !supported(isa) {
        return affine_i32_scalar(base, step, round, shift, out);
    }
    match isa {
        Isa::Scalar => affine_i32_scalar(base, step, round, shift, out),
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe { affine_i32_sse41(base, step, round, shift, out) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { affine_i32_avx2(base, step, round, shift, out) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { affine_i32_neon(base, step, round, shift, out) },
    }
}

/// Generate `out[i] = (base + i * step + source[i] * source_scale + round) >> shift`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn affine_source_i32(
    isa: Isa,
    base: i32,
    step: i32,
    source: &[i32],
    source_scale: i32,
    round: i32,
    shift: i32,
    out: &mut [i32],
) {
    debug_assert!((0..32).contains(&shift));
    debug_assert!(source.len() >= out.len());
    if !supported(isa) {
        return affine_source_i32_scalar(base, step, source, source_scale, round, shift, out);
    }
    match isa {
        Isa::Scalar => {
            affine_source_i32_scalar(base, step, source, source_scale, round, shift, out)
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => unsafe {
            affine_source_i32_sse41(base, step, source, source_scale, round, shift, out)
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe {
            affine_source_i32_avx2(base, step, source, source_scale, round, shift, out)
        },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe {
            affine_source_i32_neon(base, step, source, source_scale, round, shift, out)
        },
    }
}

fn affine_i32_scalar(base: i32, step: i32, round: i32, shift: i32, out: &mut [i32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = (base + i as i32 * step + round) >> shift;
    }
}

fn affine_source_i32_scalar(
    base: i32,
    step: i32,
    source: &[i32],
    source_scale: i32,
    round: i32,
    shift: i32,
    out: &mut [i32],
) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = (base + i as i32 * step + source[i] * source_scale + round) >> shift;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn fill_i32_sse41(value: i32, out: &mut [i32]) {
    unsafe {
        let v = _mm_set1_epi32(value);
        let mut i = 0usize;
        while i + 4 <= out.len() {
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), v);
            i += 4;
        }
        out[i..].fill(value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_i32_avx2(value: i32, out: &mut [i32]) {
    unsafe {
        let v = _mm256_set1_epi32(value);
        let mut i = 0usize;
        while i + 8 <= out.len() {
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), v);
            i += 8;
        }
        if i + 4 <= out.len() {
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm256_castsi256_si128(v));
            i += 4;
        }
        out[i..].fill(value);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_i32_neon(value: i32, out: &mut [i32]) {
    unsafe {
        let v = vdupq_n_s32(value);
        let mut i = 0usize;
        while i + 4 <= out.len() {
            vst1q_s32(out.as_mut_ptr().add(i), v);
            i += 4;
        }
        out[i..].fill(value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn affine_i32_sse41(base: i32, step: i32, round: i32, shift: i32, out: &mut [i32]) {
    unsafe {
        let idx0 = _mm_setr_epi32(0, 1, 2, 3);
        let four_steps = _mm_set1_epi32(4 * step);
        let step_v = _mm_set1_epi32(step);
        let round_v = _mm_set1_epi32(round);
        let sh = _mm_cvtsi32_si128(shift);
        let mut b = _mm_add_epi32(_mm_set1_epi32(base), _mm_mullo_epi32(idx0, step_v));
        let mut i = 0usize;
        while i + 4 <= out.len() {
            _mm_storeu_si128(
                out.as_mut_ptr().add(i).cast(),
                _mm_sra_epi32(_mm_add_epi32(b, round_v), sh),
            );
            b = _mm_add_epi32(b, four_steps);
            i += 4;
        }
        affine_i32_scalar(base + i as i32 * step, step, round, shift, &mut out[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn affine_source_i32_sse41(
    base: i32,
    step: i32,
    source: &[i32],
    source_scale: i32,
    round: i32,
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let idx0 = _mm_setr_epi32(0, 1, 2, 3);
        let four_steps = _mm_set1_epi32(4 * step);
        let step_v = _mm_set1_epi32(step);
        let scale_v = _mm_set1_epi32(source_scale);
        let round_v = _mm_set1_epi32(round);
        let sh = _mm_cvtsi32_si128(shift);
        let mut b = _mm_add_epi32(_mm_set1_epi32(base), _mm_mullo_epi32(idx0, step_v));
        let mut i = 0usize;
        while i + 4 <= out.len() {
            let src = _mm_loadu_si128(source.as_ptr().add(i).cast());
            let acc = _mm_add_epi32(_mm_add_epi32(b, _mm_mullo_epi32(src, scale_v)), round_v);
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sra_epi32(acc, sh));
            b = _mm_add_epi32(b, four_steps);
            i += 4;
        }
        affine_source_i32_scalar(
            base + i as i32 * step,
            step,
            &source[i..],
            source_scale,
            round,
            shift,
            &mut out[i..],
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn affine_i32_avx2(base: i32, step: i32, round: i32, shift: i32, out: &mut [i32]) {
    unsafe {
        let idx0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let eight_steps = _mm256_set1_epi32(8 * step);
        let step_v = _mm256_set1_epi32(step);
        let round_v = _mm256_set1_epi32(round);
        let sh = _mm_cvtsi32_si128(shift);
        let mut b = _mm256_add_epi32(_mm256_set1_epi32(base), _mm256_mullo_epi32(idx0, step_v));
        let mut i = 0usize;
        while i + 8 <= out.len() {
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i).cast(),
                _mm256_sra_epi32(_mm256_add_epi32(b, round_v), sh),
            );
            b = _mm256_add_epi32(b, eight_steps);
            i += 8;
        }
        affine_i32_scalar(base + i as i32 * step, step, round, shift, &mut out[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn affine_source_i32_avx2(
    base: i32,
    step: i32,
    source: &[i32],
    source_scale: i32,
    round: i32,
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let idx0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let eight_steps = _mm256_set1_epi32(8 * step);
        let step_v = _mm256_set1_epi32(step);
        let scale_v = _mm256_set1_epi32(source_scale);
        let round_v = _mm256_set1_epi32(round);
        let sh = _mm_cvtsi32_si128(shift);
        let mut b = _mm256_add_epi32(_mm256_set1_epi32(base), _mm256_mullo_epi32(idx0, step_v));
        let mut i = 0usize;
        while i + 8 <= out.len() {
            let src = _mm256_loadu_si256(source.as_ptr().add(i).cast());
            let acc = _mm256_add_epi32(
                _mm256_add_epi32(b, _mm256_mullo_epi32(src, scale_v)),
                round_v,
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_sra_epi32(acc, sh));
            b = _mm256_add_epi32(b, eight_steps);
            i += 8;
        }
        affine_source_i32_scalar(
            base + i as i32 * step,
            step,
            &source[i..],
            source_scale,
            round,
            shift,
            &mut out[i..],
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_i32_neon(base: i32, step: i32, round: i32, shift: i32, out: &mut [i32]) {
    unsafe {
        let idx0 = [0, 1, 2, 3];
        let idx0 = vld1q_s32(idx0.as_ptr());
        let four_steps = vdupq_n_s32(4 * step);
        let step_v = vdupq_n_s32(step);
        let round_v = vdupq_n_s32(round);
        let sh = vdupq_n_s32(-shift);
        let mut b = vmlaq_s32(vdupq_n_s32(base), idx0, step_v);
        let mut i = 0usize;
        while i + 4 <= out.len() {
            vst1q_s32(
                out.as_mut_ptr().add(i),
                vshlq_s32(vaddq_s32(b, round_v), sh),
            );
            b = vaddq_s32(b, four_steps);
            i += 4;
        }
        affine_i32_scalar(base + i as i32 * step, step, round, shift, &mut out[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_source_i32_neon(
    base: i32,
    step: i32,
    source: &[i32],
    source_scale: i32,
    round: i32,
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let idx0 = [0, 1, 2, 3];
        let idx0 = vld1q_s32(idx0.as_ptr());
        let four_steps = vdupq_n_s32(4 * step);
        let step_v = vdupq_n_s32(step);
        let scale_v = vdupq_n_s32(source_scale);
        let round_v = vdupq_n_s32(round);
        let sh = vdupq_n_s32(-shift);
        let mut b = vmlaq_s32(vdupq_n_s32(base), idx0, step_v);
        let mut i = 0usize;
        while i + 4 <= out.len() {
            let src = vld1q_s32(source.as_ptr().add(i));
            let acc = vaddq_s32(vmlaq_s32(b, src, scale_v), round_v);
            vst1q_s32(out.as_mut_ptr().add(i), vshlq_s32(acc, sh));
            b = vaddq_s32(b, four_steps);
            i += 4;
        }
        affine_source_i32_scalar(
            base + i as i32 * step,
            step,
            &source[i..],
            source_scale,
            round,
            shift,
            &mut out[i..],
        );
    }
}

/// Whether the running CPU can execute `isa`'s kernels.
#[inline]
fn supported(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => matches!(detected_isa(), Isa::Sse41 | Isa::Avx2),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => detected_isa() == Isa::Avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => true,
    }
}

// ---------------------------------------------------------------------------
// Separable filter accumulation
// ---------------------------------------------------------------------------

/// `out[i] = ( Σ coeffs[t] · taps[t][i] ) >> shift` for every `i` in
/// `out`, on the requested backend.
///
/// `N` is the tap count (8 for the §8.5.3.3.3.2 luma filter, 4 for the
/// §8.5.3.3.3.3 chroma filter). Every `taps[t]` must be at least
/// `out.len()` long. `shift` must be in `0..32`. An `isa` the running
/// CPU does not support falls back to [`Isa::Scalar`].
#[inline]
pub fn filter_taps<const N: usize>(
    isa: Isa,
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    debug_assert!((0..32).contains(&shift));
    debug_assert!(taps.iter().all(|t| t.len() >= out.len()));
    if !supported(isa) {
        return filter_taps_scalar(taps, coeffs, shift, out);
    }
    match isa {
        Isa::Scalar => filter_taps_scalar(taps, coeffs, shift, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed SSE4.1 above, and the tap slices
        // are at least `out.len()` long so every load stays in bounds.
        Isa::Sse41 => unsafe { filter_taps_sse41(taps, coeffs, shift, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed AVX2 above; same bounds argument.
        Isa::Avx2 => unsafe { filter_taps_avx2(taps, coeffs, shift, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; same bounds argument.
        Isa::Neon => unsafe { filter_taps_neon(taps, coeffs, shift, out) },
    }
}

fn filter_taps_scalar<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0i32;
        for (&c, tap) in coeffs.iter().zip(taps.iter()) {
            acc += c * tap[i];
        }
        *o = acc >> shift;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn filter_taps_sse41<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(shift);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = _mm_setzero_si128();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm_loadu_si128(tap.as_ptr().add(i).cast());
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(c)));
            }
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sra_epi32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_taps_avx2<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(shift);
        let mut i = 0usize;
        while i + 8 <= count {
            let mut acc = _mm256_setzero_si256();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm256_loadu_si256(tap.as_ptr().add(i).cast());
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(c)));
            }
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_sra_epi32(acc, sh));
            i += 8;
        }
        while i + 4 <= count {
            let mut acc = _mm_setzero_si128();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm_loadu_si128(tap.as_ptr().add(i).cast());
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(c)));
            }
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sra_epi32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_taps_neon<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        // A negative `vshlq_s32` amount is an arithmetic right shift.
        let sh = vdupq_n_s32(-shift);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = vdupq_n_s32(0);
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                acc = vmlaq_n_s32(acc, vld1q_s32(tap.as_ptr().add(i)), c);
            }
            vst1q_s32(out.as_mut_ptr().add(i), vshlq_s32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

/// The scalar remainder of a vector kernel, for the `out.len() % lanes`
/// samples the vector loops could not cover.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn filter_taps_tail<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
    from: usize,
) {
    for i in from..out.len() {
        let mut acc = 0i32;
        for (&c, tap) in coeffs.iter().zip(taps.iter()) {
            acc += c * tap[i];
        }
        out[i] = acc >> shift;
    }
}

// ---------------------------------------------------------------------------
// Weighted sample combine
// ---------------------------------------------------------------------------

/// `out[i] = Clip3( 0, max_val, ( ( ( Σ weights[t] · taps[t][i] ) + round )
/// >> shift ) + post )` for every `i` in `out`, on the requested backend.
///
/// `N` is 1 for the uni-predictive combines (§8.5.3.3.4.2 equations
/// 8-262 / 8-263, §8.5.3.3.4.3 equations 8-275 / 8-276) and 2 for the
/// bi-predictive ones (equations 8-264 / 8-277). Every `taps[t]` must be
/// at least `out.len()` long and `shift` must be in `0..32`. The caller
/// is responsible for having established that the accumulation cannot
/// overflow `i32`. An `isa` the running CPU does not support falls back
/// to [`Isa::Scalar`].
// The round / shift / post / max quartet are the four distinct spec
// quantities of equations 8-262..8-277; grouping them into the private
// `CombineParams` would force that type into the public signature.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn combine_weighted<const N: usize>(
    isa: Isa,
    taps: &[&[i32]; N],
    weights: &[i32; N],
    round: i32,
    shift: i32,
    post: i32,
    max_val: i32,
    out: &mut [i32],
) {
    debug_assert!((0..32).contains(&shift));
    debug_assert!(taps.iter().all(|t| t.len() >= out.len()));
    let p = CombineParams {
        round,
        shift,
        post,
        max_val,
    };
    if !supported(isa) {
        return combine_weighted_scalar(taps, weights, p, out);
    }
    match isa {
        Isa::Scalar => combine_weighted_scalar(taps, weights, p, out),
        // On SSE4.1 the scalar reference *is* the faster backend, for the same
        // reason it is on NEON below: the combine is one or two
        // multiply-accumulates, a shift and a clamp per sample, so a
        // hand-written four-lane kernel is doing exactly what LLVM already
        // does to `combine_weighted_scalar` under this crate's `lto = "fat"` /
        // `codegen-units = 1` profile, only with worse scheduling. Measured on
        // both x86_64 vendors (best of five interleaved rounds, three runs
        // each) with the kernel temporarily dispatched to so it could be
        // timed: on AMD EPYC it ran 0.93-0.95x of scalar in the §8.5.3.3.4
        // block path and level with it on an L1-resident buffer with the
        // allocator kept out of the timing, and on an Intel Coffee Lake
        // i7-8700B it runs ~1.00x on both arms (0.99-1.10x block,
        // 0.96-1.03x buffer over five `macos-15-intel` draws for #326,
        // superseding the 1.00x / 0.75x #224 recorded from one draw).
        // Vendor was the open question (#218) and it does not move this
        // decision: the four-lane kernel is at or below scalar on both, and
        // reads ~1.00x on both arms on an Emerald Rapids Xeon Platinum
        // 8573C too (#285), so the SSE4.1 dispatch to scalar is confirmed
        // on all three x86_64 hosts measured.
        //
        // AVX2 is dispatched to below, unconditionally, and #285 is why
        // that is now a positive result rather than a tolerated cost.
        // Eight lanes is width the four-lane auto-vectorization does not
        // reach: the kernel measures 1.38x in the block path and 2.0-2.2x
        // on a bare buffer on EPYC, and 1.13-1.25x / 1.5-1.6x on an
        // Emerald Rapids Xeon Platinum 8573C (#285). #224 read 0.93-0.95x
        // and 0.92-0.93x on an Intel Coffee Lake i7-8700B, and #301 built
        // a Skylake-family `vpmulld` argument to explain it — 256-bit
        // `vpmulld` costs two uops there against one from Ice Lake onward.
        // That reading does not reproduce. Re-measured for #326 over five
        // `macos-15-intel` draws of the same part, with this kernel
        // unchanged, it reads ~1.00x in the block path and 2.0-2.2x on a
        // bare buffer — level with Zen, and the eight-lane kernel simply
        // halving the four-lane scalar loop. So the kernel stays dispatched
        // to on every AVX2 host, now because all three measured
        // microarchitectures profit rather than in spite of one that did
        // not. See the module docs for the full per-host table.
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => combine_weighted_scalar(taps, weights, p, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed AVX2 above, and the tap slices are at
        // least `out.len()` long so every load stays in bounds.
        Isa::Avx2 => unsafe { combine_weighted_avx2(taps, weights, p, out) },
        // On AArch64 the scalar reference *is* the faster backend, so the
        // dispatch prefers it and there is no NEON kernel to call.
        //
        // The combine is one or two multiply-accumulates, a shift and a
        // clamp per sample — no reduction, no shuffle, nothing a hand
        // kernel can express that the auto-vectorizer cannot. Under this
        // crate's `lto = "fat"` / `codegen-units = 1` profile LLVM already
        // vectorizes `combine_weighted_scalar` to four-lane NEON, and its
        // version schedules better than the intrinsics one did: on Apple
        // silicon the hand-written kernel measured 0.91x of scalar both in
        // the §8.5.3.3.4 block path and on an L1-resident buffer with the
        // allocator kept out of the timing (best of five interleaved
        // rounds — see `inter_pred`'s `simd_inter_pred_benchmark`).
        // Widening it to eight and sixteen samples per iteration, and
        // dropping its redundant `#[target_feature(enable = "neon")]`,
        // were both tried and both made it slower still.
        //
        // `filter_taps` above is a different case and keeps its NEON
        // kernel: at the 4..16-sample row lengths the decoder actually
        // asks for, it runs 1.6-1.9x the scalar path.
        //
        // The same measurement has since been made on x86_64 hardware and
        // splits the same way by vector width rather than by architecture:
        // the four-lane SSE4.1 combine lost to scalar too and is dispatched
        // to it above, while the eight-lane AVX2 one won and is kept.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => combine_weighted_scalar(taps, weights, p, out),
    }
}

/// The scalar parameters of [`combine_weighted`], grouped so the backend
/// helpers stay under the argument-count lint.
#[derive(Debug, Clone, Copy)]
struct CombineParams {
    round: i32,
    shift: i32,
    post: i32,
    max_val: i32,
}

fn combine_weighted_scalar<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = p.round;
        for (&w, tap) in weights.iter().zip(taps.iter()) {
            acc += w * tap[i];
        }
        *o = ((acc >> p.shift) + p.post).clamp(0, p.max_val);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn combine_weighted_avx2<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(p.shift);
        let round = _mm256_set1_epi32(p.round);
        let post = _mm256_set1_epi32(p.post);
        let lo = _mm256_setzero_si256();
        let hi = _mm256_set1_epi32(p.max_val);
        let mut i = 0usize;
        while i + 8 <= count {
            let mut acc = round;
            for (&w, tap) in weights.iter().zip(taps.iter()) {
                let v = _mm256_loadu_si256(tap.as_ptr().add(i).cast());
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(w)));
            }
            let v = _mm256_add_epi32(_mm256_sra_epi32(acc, sh), post);
            let v = _mm256_min_epi32(_mm256_max_epi32(v, lo), hi);
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), v);
            i += 8;
        }
        combine_weighted_tail(taps, weights, p, out, i);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn combine_weighted_tail<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
    from: usize,
) {
    for i in from..out.len() {
        let mut acc = p.round;
        for (&w, tap) in weights.iter().zip(taps.iter()) {
            acc += w * tap[i];
        }
        out[i] = ((acc >> p.shift) + p.post).clamp(0, p.max_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random `i32` samples in `[-limit, limit]`.
    fn samples(seed: u64, len: usize, limit: i32) -> Vec<i32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let v = (state >> 33) as i64 % (2 * i64::from(limit) + 1);
                (v - i64::from(limit)) as i32
            })
            .collect()
    }

    /// Times [`filter_taps`] against the auto-vectorized scalar
    /// reference as a function of the per-call row length alone.
    ///
    /// Issue #280 inferred, and issue #309 set out to confirm, that the
    /// 8-tap luma kernel's advantage decays with prediction-unit size
    /// because the two-dimensional path materializes a `w x ( h + 7 )`
    /// intermediate between its two passes. This sweep removes the
    /// intermediate, the block walk and the allocation from the picture
    /// entirely: every row length reads the same L1-resident tap buffer,
    /// writes the same output buffer and covers the same total sample
    /// count, so the only variable left is how many samples one
    /// `filter_taps` call is asked for.
    ///
    /// Ignored by default because it is a timing measurement, not an
    /// assertion. Run it with
    /// `cargo test --release --features native --lib
    /// measure_filter_taps_by_row_length -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn measure_filter_taps_by_row_length() {
        use std::time::Instant;

        // 4 KiB of taps: the whole working set stays in L1 at every row
        // length, so a cache effect cannot masquerade as a length effect.
        const BUF: usize = 1024;
        const TOTAL: usize = 1 << 22;
        let src: Vec<Vec<i32>> = (0..8).map(|t| samples(t + 11, BUF, 255)).collect();
        let coeffs = [-1, 4, -11, 40, 40, -11, 4, -1];
        let isas = available_isas();
        let rounds = 9;

        println!("\n8-tap filter_taps by row length, best of {rounds} interleaved rounds");
        println!("  (same total sample count and same L1-resident buffer at every length)");
        println!("   row   isa          ms   ratio");
        for &len in &[4usize, 8, 16, 32, 64, 128, 256] {
            let taps: [&[i32]; 8] = std::array::from_fn(|t| &src[t][..len]);
            let calls = TOTAL / len;
            let mut out = vec![0i32; len];
            let mut best = vec![f64::INFINITY; isas.len()];
            for _ in 0..rounds {
                for (i, &isa) in isas.iter().enumerate() {
                    let start = Instant::now();
                    for _ in 0..calls {
                        filter_taps(isa, &taps, &coeffs, 6, std::hint::black_box(&mut out));
                    }
                    best[i] = best[i].min(start.elapsed().as_secs_f64());
                }
            }
            for (isa, t) in isas.iter().zip(best.iter().copied()) {
                println!(
                    "  {len:>4}  {:>7}  {:8.2}  {:5.2}x",
                    format!("{isa:?}"),
                    t * 1e3,
                    best[0] / t
                );
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_filter_taps() {
        let src: Vec<Vec<i32>> = (0..8).map(|t| samples(t + 7, 200, 40_000)).collect();
        let taps8: [&[i32]; 8] = std::array::from_fn(|t| src[t].as_slice());
        let taps4: [&[i32]; 4] = std::array::from_fn(|t| src[t].as_slice());
        let c8 = [-1, 4, -11, 40, 40, -11, 4, -1];
        let c4 = [-2, 58, 10, -2];
        // Widths that exercise the 8-, 4- and 1-wide code paths, plus the
        // lengths that straddle each vector step and leave a partial tail.
        for len in [
            1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 20, 24, 31, 33, 48, 64, 200,
        ] {
            for shift in [0i32, 2, 4, 6] {
                let mut reference = vec![0i32; len];
                filter_taps(Isa::Scalar, &taps8, &c8, shift, &mut reference);
                let mut reference4 = vec![0i32; len];
                filter_taps(Isa::Scalar, &taps4, &c4, shift, &mut reference4);
                for isa in available_isas() {
                    let mut got = vec![0i32; len];
                    filter_taps(isa, &taps8, &c8, shift, &mut got);
                    assert_eq!(got, reference, "{isa:?} 8-tap len={len} shift={shift}");
                    let mut got4 = vec![0i32; len];
                    filter_taps(isa, &taps4, &c4, shift, &mut got4);
                    assert_eq!(got4, reference4, "{isa:?} 4-tap len={len} shift={shift}");
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_combine_weighted() {
        let a = samples(3, 200, 30_000);
        let b = samples(11, 200, 30_000);
        for len in [1usize, 3, 4, 5, 8, 13, 16, 17, 20, 24, 33, 48, 64, 200] {
            for (weights, round, shift, post) in [
                ([1, 1], 32, 6, 0),
                ([1, 1], 64, 7, 0),
                ([64, -12], 1 << 12, 13, 0),
                ([-5, 3], 0, 5, 17),
            ] {
                let taps2: [&[i32]; 2] = [&a, &b];
                let taps1: [&[i32]; 1] = [&a];
                let w1 = [weights[0]];
                let mut reference = vec![0i32; len];
                combine_weighted(
                    Isa::Scalar,
                    &taps2,
                    &weights,
                    round,
                    shift,
                    post,
                    255,
                    &mut reference,
                );
                let mut reference1 = vec![0i32; len];
                combine_weighted(
                    Isa::Scalar,
                    &taps1,
                    &w1,
                    round,
                    shift,
                    post,
                    1023,
                    &mut reference1,
                );
                for isa in available_isas() {
                    let mut got = vec![0i32; len];
                    combine_weighted(isa, &taps2, &weights, round, shift, post, 255, &mut got);
                    assert_eq!(got, reference, "{isa:?} bi len={len}");
                    let mut got1 = vec![0i32; len];
                    combine_weighted(isa, &taps1, &w1, round, shift, post, 1023, &mut got1);
                    assert_eq!(got1, reference1, "{isa:?} uni len={len}");
                }
            }
        }
    }

    /// An unsupported backend must not be executed; it degrades to the
    /// scalar path rather than issuing an illegal instruction.
    #[test]
    fn unsupported_backend_falls_back_to_scalar() {
        let a = vec![7i32; 16];
        let taps: [&[i32]; 1] = [&a];
        let mut out = vec![0i32; 16];
        for isa in [
            Isa::Scalar,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2,
            #[cfg(target_arch = "x86_64")]
            Isa::Sse41,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon,
        ] {
            filter_taps(isa, &taps, &[64], 6, &mut out);
            assert!(out.iter().all(|&v| v == 7));
        }
    }
}

// ---------------------------------------------------------------------------
// HEVC in-loop filters
// ---------------------------------------------------------------------------

pub(crate) mod in_loop {
    //! Vectorized kernels for the §8.7.2 deblocking and §8.7.3 SAO in-loop
    //! filters, with runtime CPU feature detection and a scalar fallback.
    //!
    //! The in-loop filters are the two stages of the HEVC decode pipeline that
    //! touch *every* reconstructed sample of *every* picture, so they dominate
    //! the software decoder's per-pixel cost once prediction and the inverse
    //! transform have been optimized. This module keeps the spec-shaped scalar
    //! implementations in [`deblock`] / [`sao`] as the normative reference and
    //! adds bit-exact SIMD kernels underneath them:
    //!
    //! * **SAO band offset** (§8.7.3.2 equations 8-414..8-415) and **SAO edge
    //!   offset** (equations 8-409..8-413) run over long runs of contiguous
    //!   samples, so they use the widest available vector: AVX2 (8 x `i32`)
    //!   when the CPU has it, otherwise SSE4.1 (4 x `i32`) on `x86_64` or NEON
    //!   (4 x `i32`) on `aarch64`.
    //! * **Deblocking luma strong / weak filtering** (§8.7.2.5.7 equations
    //!   8-389..8-402) and **chroma filtering** (equations 8-403..8-405) are
    //!   defined on a four-row edge *segment* with one shared decision, so the
    //!   natural vectorization maps the segment's four rows onto four `i32`
    //!   lanes. That is exactly one SSE4.1 / NEON register; AVX2's extra width
    //!   has nothing to fill it with here, so the AVX2-capable path
    //!   deliberately uses the same 128-bit kernels for deblocking and spends
    //!   its width on SAO instead.
    //!
    //! Every kernel is written once as a generic function over the [`Ops`]
    //! vector trait and instantiated per instruction set, so the SSE4.1, AVX2
    //! and NEON paths cannot drift apart. Targets without a supported vector
    //! ISA (notably `wasm32`) compile to the scalar path only; the dispatcher
    //! is a cached runtime feature probe, so a binary built for a baseline
    //! `x86_64` still uses AVX2 on a machine that has it.
    //!
    //! Bit-exactness against the scalar reference is asserted by the module
    //! tests over exhaustive boundary-strength / QP / bit-depth / SAO
    //! type / edge-class sweeps, and `bench_in_loop_filters` (an `#[ignore]`d
    //! timing test) reports the measured speedup on a representative
    //! reconstructed frame.

    use core::sync::atomic::{AtomicU8, Ordering};

    /// The selected kernel family. Cached in [`ISA`] after the first probe.
    const ISA_UNKNOWN: u8 = 0;
    const ISA_SCALAR: u8 = 1;
    #[cfg(target_arch = "x86_64")]
    const ISA_SSE41: u8 = 2;
    #[cfg(target_arch = "x86_64")]
    const ISA_AVX2: u8 = 3;
    #[cfg(target_arch = "aarch64")]
    const ISA_NEON: u8 = 4;

    static ISA: AtomicU8 = AtomicU8::new(ISA_UNKNOWN);

    /// Switch that forces the scalar path, so a benchmark can time both
    /// families in one process and the bit-exactness tests can pin the
    /// reference side.
    ///
    /// Kept as an in-crate shorthand for the public
    /// [`crate::simd::set_override`]`(Some(SimdIsa::Scalar))`, which the
    /// dispatcher honours as well; it is no longer `#[cfg(test)]`-gated
    /// because the in-loop filter kernels have to be switchable from an
    /// external `benches/` target too.
    pub(crate) static FORCE_SCALAR: core::sync::atomic::AtomicBool =
        core::sync::atomic::AtomicBool::new(false);

    /// Probe (once) and return the kernel family to use.
    ///
    /// The crate-wide override and [`FORCE_SCALAR`] are both checked ahead of
    /// the cached probe, so either one takes effect immediately.
    #[inline]
    fn isa() -> u8 {
        if FORCE_SCALAR.load(Ordering::Relaxed) {
            return ISA_SCALAR;
        }
        if let Some(isa) = overridden_isa_code() {
            return isa;
        }
        let cached = ISA.load(Ordering::Relaxed);
        if cached != ISA_UNKNOWN {
            return cached;
        }
        let detected = detect();
        ISA.store(detected, Ordering::Relaxed);
        detected
    }

    #[cfg(target_arch = "x86_64")]
    fn detect() -> u8 {
        if std::is_x86_feature_detected!("avx2") {
            ISA_AVX2
        } else if std::is_x86_feature_detected!("sse4.1") {
            ISA_SSE41
        } else {
            ISA_SCALAR
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect() -> u8 {
        // NEON (`asimd`) is architecturally guaranteed on aarch64.
        ISA_NEON
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect() -> u8 {
        ISA_SCALAR
    }

    /// Maps the crate-wide SIMD override, if any, onto this module's kernel
    /// family codes.
    #[inline]
    fn overridden_isa_code() -> Option<u8> {
        use crate::simd::SimdIsa;
        Some(match crate::simd::override_isa()? {
            SimdIsa::Scalar => ISA_SCALAR,
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => ISA_SSE41,
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => ISA_AVX2,
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => ISA_NEON,
            #[allow(unreachable_patterns)]
            _ => ISA_SCALAR,
        })
    }

    // ---------------------------------------------------------------------------
    // Vector abstraction
    // ---------------------------------------------------------------------------

    /// The lane operations every kernel in this module needs, over a vector of
    /// signed 32-bit lanes (HEVC sample arrays are stored as `i32`).
    ///
    /// Implementations are thin wrappers over one instruction set's
    /// intrinsics; each carries the `#[target_feature]` that makes calling it
    /// sound once the corresponding [`detect`] probe has succeeded.
    trait Ops: Copy {
        /// Lanes per vector.
        const LANES: usize;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn splat(v: i32) -> Self;
        /// # Safety
        /// `src` must be readable for `LANES` `i32`s, and the feature present.
        unsafe fn load(src: *const i32) -> Self;
        /// # Safety
        /// `dst` must be writable for `LANES` `i32`s, and the feature present.
        unsafe fn store(self, dst: *mut i32);
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn add(self, o: Self) -> Self;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn sub(self, o: Self) -> Self;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn min(self, o: Self) -> Self;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn max(self, o: Self) -> Self;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn and(self, o: Self) -> Self;
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn or(self, o: Self) -> Self;
        /// `(!self) & o`.
        ///
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn andnot(self, o: Self) -> Self;
        /// Lanewise `self > o` as an all-ones / all-zeros mask.
        ///
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn cmpgt(self, o: Self) -> Self;
        /// Lanewise `self == o` as an all-ones / all-zeros mask.
        ///
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn cmpeq(self, o: Self) -> Self;
        /// Arithmetic right shift by a runtime count in `0..32`.
        ///
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn sra(self, n: i32) -> Self;
        /// Left shift by a runtime count in `0..32`.
        ///
        /// # Safety
        /// The caller must have verified the implementation's CPU feature.
        unsafe fn sll(self, n: i32) -> Self;
    }

    /// `mask ? a : b` lanewise, for an all-ones / all-zeros `mask`.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature.
    #[inline(always)]
    unsafe fn blend<V: Ops>(mask: V, a: V, b: V) -> V {
        unsafe { mask.and(a).or(mask.andnot(b)) }
    }

    /// Lanewise `|x|`.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature.
    #[inline(always)]
    unsafe fn vabs<V: Ops>(x: V) -> V {
        unsafe { x.max(V::splat(0).sub(x)) }
    }

    // ---------------------------------------------------------------------------
    // SSE4.1
    // ---------------------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use super::Ops;
        use core::arch::x86_64::*;

        /// 4 x `i32` over SSE4.1.
        #[derive(Clone, Copy)]
        pub(super) struct V4(pub __m128i);

        impl Ops for V4 {
            const LANES: usize = 4;
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn splat(v: i32) -> Self {
                V4(_mm_set1_epi32(v))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn load(src: *const i32) -> Self {
                unsafe { V4(_mm_loadu_si128(src.cast())) }
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn store(self, dst: *mut i32) {
                unsafe { _mm_storeu_si128(dst.cast(), self.0) }
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn add(self, o: Self) -> Self {
                V4(_mm_add_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn sub(self, o: Self) -> Self {
                V4(_mm_sub_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn min(self, o: Self) -> Self {
                V4(_mm_min_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn max(self, o: Self) -> Self {
                V4(_mm_max_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn and(self, o: Self) -> Self {
                V4(_mm_and_si128(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn or(self, o: Self) -> Self {
                V4(_mm_or_si128(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn andnot(self, o: Self) -> Self {
                V4(_mm_andnot_si128(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn cmpgt(self, o: Self) -> Self {
                V4(_mm_cmpgt_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn cmpeq(self, o: Self) -> Self {
                V4(_mm_cmpeq_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn sra(self, n: i32) -> Self {
                V4(_mm_sra_epi32(self.0, _mm_cvtsi32_si128(n)))
            }
            #[inline]
            #[target_feature(enable = "sse4.1")]
            unsafe fn sll(self, n: i32) -> Self {
                V4(_mm_sll_epi32(self.0, _mm_cvtsi32_si128(n)))
            }
        }

        /// 8 x `i32` over AVX2.
        #[derive(Clone, Copy)]
        pub(super) struct V8(pub __m256i);

        impl Ops for V8 {
            const LANES: usize = 8;
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn splat(v: i32) -> Self {
                V8(_mm256_set1_epi32(v))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn load(src: *const i32) -> Self {
                unsafe { V8(_mm256_loadu_si256(src.cast())) }
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn store(self, dst: *mut i32) {
                unsafe { _mm256_storeu_si256(dst.cast(), self.0) }
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn add(self, o: Self) -> Self {
                V8(_mm256_add_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn sub(self, o: Self) -> Self {
                V8(_mm256_sub_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn min(self, o: Self) -> Self {
                V8(_mm256_min_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn max(self, o: Self) -> Self {
                V8(_mm256_max_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn and(self, o: Self) -> Self {
                V8(_mm256_and_si256(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn or(self, o: Self) -> Self {
                V8(_mm256_or_si256(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn andnot(self, o: Self) -> Self {
                V8(_mm256_andnot_si256(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn cmpgt(self, o: Self) -> Self {
                V8(_mm256_cmpgt_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn cmpeq(self, o: Self) -> Self {
                V8(_mm256_cmpeq_epi32(self.0, o.0))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn sra(self, n: i32) -> Self {
                V8(_mm256_sra_epi32(self.0, _mm_cvtsi32_si128(n)))
            }
            #[inline]
            #[target_feature(enable = "avx2")]
            unsafe fn sll(self, n: i32) -> Self {
                V8(_mm256_sll_epi32(self.0, _mm_cvtsi32_si128(n)))
            }
        }
    }

    // ---------------------------------------------------------------------------
    // NEON
    // ---------------------------------------------------------------------------

    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::Ops;
        use core::arch::aarch64::*;

        /// 4 x `i32` over NEON (architecturally guaranteed on `aarch64`).
        #[derive(Clone, Copy)]
        pub(super) struct V4(pub int32x4_t);

        impl Ops for V4 {
            const LANES: usize = 4;
            #[inline]
            unsafe fn splat(v: i32) -> Self {
                unsafe { V4(vdupq_n_s32(v)) }
            }
            #[inline]
            unsafe fn load(src: *const i32) -> Self {
                unsafe { V4(vld1q_s32(src)) }
            }
            #[inline]
            unsafe fn store(self, dst: *mut i32) {
                unsafe { vst1q_s32(dst, self.0) }
            }
            #[inline]
            unsafe fn add(self, o: Self) -> Self {
                unsafe { V4(vaddq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn sub(self, o: Self) -> Self {
                unsafe { V4(vsubq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn min(self, o: Self) -> Self {
                unsafe { V4(vminq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn max(self, o: Self) -> Self {
                unsafe { V4(vmaxq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn and(self, o: Self) -> Self {
                unsafe { V4(vandq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn or(self, o: Self) -> Self {
                unsafe { V4(vorrq_s32(self.0, o.0)) }
            }
            #[inline]
            unsafe fn andnot(self, o: Self) -> Self {
                // NEON's `vbicq` is `a & !b`, so the operands swap.
                unsafe { V4(vbicq_s32(o.0, self.0)) }
            }
            #[inline]
            unsafe fn cmpgt(self, o: Self) -> Self {
                unsafe { V4(vreinterpretq_s32_u32(vcgtq_s32(self.0, o.0))) }
            }
            #[inline]
            unsafe fn cmpeq(self, o: Self) -> Self {
                unsafe { V4(vreinterpretq_s32_u32(vceqq_s32(self.0, o.0))) }
            }
            #[inline]
            unsafe fn sra(self, n: i32) -> Self {
                unsafe { V4(vshlq_s32(self.0, vdupq_n_s32(-n))) }
            }
            #[inline]
            unsafe fn sll(self, n: i32) -> Self {
                unsafe { V4(vshlq_s32(self.0, vdupq_n_s32(n))) }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // SAO band offset (§8.7.3.2 equations 8-414..8-415)
    // ---------------------------------------------------------------------------

    /// Scalar reference for one row run of SAO band offset.
    ///
    /// `left` is `sao_band_position`, `band_shift` is `BitDepth - 5`, and
    /// `off` is `SaoOffsetVal[..]` (`off[0]` is always 0). Equation 8-414's
    /// `bandTable` is inverted here: band `b` is one of the four selected
    /// bands exactly when `(b - left) & 31 < 4`, and then takes
    /// `off[((b - left) & 31) + 1]`.
    #[inline]
    fn sao_band_row_scalar(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        for (d, &cur) in dst.iter_mut().zip(src.iter()) {
            let k = ((cur >> band_shift) - left) & 31;
            let o = if k < 4 { off[(k + 1) as usize] } else { 0 };
            *d = (cur + o).clamp(0, max);
        }
    }

    /// Vector body of [`sao_band_row_scalar`], with a scalar tail.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature.
    #[inline(always)]
    unsafe fn sao_band_row_simd<V: Ops>(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        unsafe {
            let n = src.len().min(dst.len());
            let zero = V::splat(0);
            let vmax = V::splat(max);
            let vleft = V::splat(left);
            let m31 = V::splat(31);
            let ks = [V::splat(0), V::splat(1), V::splat(2), V::splat(3)];
            let os = [
                V::splat(off[1]),
                V::splat(off[2]),
                V::splat(off[3]),
                V::splat(off[4]),
            ];
            let mut i = 0usize;
            while i + V::LANES <= n {
                let cur = V::load(src.as_ptr().add(i));
                let k = cur.sra(band_shift).sub(vleft).and(m31);
                let mut o = zero;
                o = blend(k.cmpeq(ks[0]), os[0], o);
                o = blend(k.cmpeq(ks[1]), os[1], o);
                o = blend(k.cmpeq(ks[2]), os[2], o);
                o = blend(k.cmpeq(ks[3]), os[3], o);
                cur.add(o)
                    .max(zero)
                    .min(vmax)
                    .store(dst.as_mut_ptr().add(i));
                i += V::LANES;
            }
            sao_band_row_scalar(&src[i..n], &mut dst[i..n], off, left, band_shift, max);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sao_band_row_sse41(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        unsafe { sao_band_row_simd::<x86::V4>(src, dst, off, left, band_shift, max) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sao_band_row_avx2(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        unsafe { sao_band_row_simd::<x86::V8>(src, dst, off, left, band_shift, max) }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sao_band_row_neon(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        unsafe { sao_band_row_simd::<arm::V4>(src, dst, off, left, band_shift, max) }
    }

    /// Apply SAO band offset to one contiguous run of a plane row.
    ///
    /// `src` and `dst` are the pre-SAO and output runs (equal length);
    /// see [`sao_band_row_scalar`] for the parameter meanings.
    pub(crate) fn sao_band_row(
        src: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        left: i32,
        band_shift: i32,
        max: i32,
    ) {
        match isa() {
            #[cfg(target_arch = "x86_64")]
            ISA_AVX2 => unsafe { sao_band_row_avx2(src, dst, off, left, band_shift, max) },
            #[cfg(target_arch = "x86_64")]
            ISA_SSE41 => unsafe { sao_band_row_sse41(src, dst, off, left, band_shift, max) },
            #[cfg(target_arch = "aarch64")]
            ISA_NEON => unsafe { sao_band_row_neon(src, dst, off, left, band_shift, max) },
            _ => sao_band_row_scalar(src, dst, off, left, band_shift, max),
        }
    }

    // ---------------------------------------------------------------------------
    // SAO edge offset (§8.7.3.2 equations 8-409..8-413)
    // ---------------------------------------------------------------------------

    /// Scalar reference for one row run of SAO edge offset.
    ///
    /// `cur` is the pre-SAO run; `n0` / `n1` are the matching runs of the two
    /// Table 8-13 neighbours (already offset by `hPos` / `vPos`), which the
    /// caller has guaranteed to lie inside the picture.
    #[inline]
    fn sao_edge_row_scalar(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        for i in 0..dst.len() {
            let c = cur[i];
            // equation 8-411.
            let mut idx = 2 + (c - n0[i]).signum() + (c - n1[i]).signum();
            // equation 8-412.
            if idx <= 2 {
                idx = if idx == 2 { 0 } else { idx + 1 };
            }
            // equation 8-413.
            dst[i] = (c + off[idx as usize]).clamp(0, max);
        }
    }

    /// Vector body of [`sao_edge_row_scalar`], with a scalar tail.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature.
    #[inline(always)]
    unsafe fn sao_edge_row_simd<V: Ops>(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        unsafe {
            let n = dst.len();
            let zero = V::splat(0);
            let one = V::splat(1);
            let two = V::splat(2);
            let vmax = V::splat(max);
            let idxs = [
                V::splat(0),
                V::splat(1),
                V::splat(2),
                V::splat(3),
                V::splat(4),
            ];
            let os = [
                V::splat(off[1]),
                V::splat(off[2]),
                V::splat(off[3]),
                V::splat(off[4]),
            ];
            let mut i = 0usize;
            while i + V::LANES <= n {
                let c = V::load(cur.as_ptr().add(i));
                let a = V::load(n0.as_ptr().add(i));
                let b = V::load(n1.as_ptr().add(i));
                // Sign( x ) as `(x < 0) - (x > 0)` over all-ones masks.
                let d0 = c.sub(a);
                let d1 = c.sub(b);
                let s0 = zero.cmpgt(d0).sub(d0.cmpgt(zero));
                let s1 = zero.cmpgt(d1).sub(d1.cmpgt(zero));
                // equation 8-411.
                let raw = two.add(s0).add(s1);
                // equation 8-412: 0 -> 1, 1 -> 2, 2 -> 0, 3 and 4 unchanged.
                let idx = blend(
                    raw.cmpeq(idxs[2]),
                    zero,
                    blend(two.cmpgt(raw), raw.add(one), raw),
                );
                // equation 8-413 (`off[0]` is 0, so index 0 needs no blend).
                let mut o = zero;
                o = blend(idx.cmpeq(idxs[1]), os[0], o);
                o = blend(idx.cmpeq(idxs[2]), os[1], o);
                o = blend(idx.cmpeq(idxs[3]), os[2], o);
                o = blend(idx.cmpeq(idxs[4]), os[3], o);
                c.add(o).max(zero).min(vmax).store(dst.as_mut_ptr().add(i));
                i += V::LANES;
            }
            sao_edge_row_scalar(&cur[i..n], &n0[i..n], &n1[i..n], &mut dst[i..n], off, max);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sao_edge_row_sse41(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        unsafe { sao_edge_row_simd::<x86::V4>(cur, n0, n1, dst, off, max) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sao_edge_row_avx2(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        unsafe { sao_edge_row_simd::<x86::V8>(cur, n0, n1, dst, off, max) }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sao_edge_row_neon(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        unsafe { sao_edge_row_simd::<arm::V4>(cur, n0, n1, dst, off, max) }
    }

    /// Apply SAO edge offset to one contiguous run of a plane row whose two
    /// Table 8-13 neighbour runs (`n0` / `n1`) are entirely inside the
    /// picture and unmasked by slice / tile / PCM guards.
    pub(crate) fn sao_edge_row(
        cur: &[i32],
        n0: &[i32],
        n1: &[i32],
        dst: &mut [i32],
        off: &[i32; 5],
        max: i32,
    ) {
        debug_assert_eq!(cur.len(), dst.len());
        debug_assert_eq!(n0.len(), dst.len());
        debug_assert_eq!(n1.len(), dst.len());
        match isa() {
            #[cfg(target_arch = "x86_64")]
            ISA_AVX2 => unsafe { sao_edge_row_avx2(cur, n0, n1, dst, off, max) },
            #[cfg(target_arch = "x86_64")]
            ISA_SSE41 => unsafe { sao_edge_row_sse41(cur, n0, n1, dst, off, max) },
            #[cfg(target_arch = "aarch64")]
            ISA_NEON => unsafe { sao_edge_row_neon(cur, n0, n1, dst, off, max) },
            _ => sao_edge_row_scalar(cur, n0, n1, dst, off, max),
        }
    }

    // ---------------------------------------------------------------------------
    // Deblocking luma filtering (§8.7.2.5.7 equations 8-389..8-402)
    // ---------------------------------------------------------------------------

    /// The four rows of one luma edge segment, as `p[i][k]` = `pi,k`.
    pub(crate) type LumaSeg = [[i32; 4]; 4];
    /// The filtered `p0'..p2'` / `q0'..q2'` of one luma edge segment.
    pub(crate) type LumaSegOut = [[i32; 4]; 3];

    /// Vectorized §8.7.2.5.7 luma filtering of a whole four-row edge segment.
    ///
    /// The segment's four rows share one `dE` / `dEp` / `dEq` / `tC`
    /// decision, so they map onto the four lanes of a 128-bit vector.
    ///
    /// Rows for which the weak filter's equation 8-395 `|delta| < tC * 10`
    /// test fails keep their input values in `out_p` / `out_q`, which makes
    /// writing them back a no-op — the scalar path expresses the same thing
    /// as `nDp = nDq = 0`.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature; `V` must be 4-lane.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn filter_luma_rows_simd<V: Ops>(
        p: &LumaSeg,
        q: &LumaSeg,
        de: u8,
        dep: u8,
        deq: u8,
        tc: i32,
        bit_depth: u8,
        out_p: &mut LumaSegOut,
        out_q: &mut LumaSegOut,
    ) {
        unsafe {
            let p0 = V::load(p[0].as_ptr());
            let p1 = V::load(p[1].as_ptr());
            let p2 = V::load(p[2].as_ptr());
            let p3 = V::load(p[3].as_ptr());
            let q0 = V::load(q[0].as_ptr());
            let q1 = V::load(q[1].as_ptr());
            let q2 = V::load(q[2].as_ptr());
            let q3 = V::load(q[3].as_ptr());
            let two_tc = V::splat(2 * tc);

            if de == 2 {
                // Strong filter (equations 8-389..8-394); the ±2*tC clip is
                // the whole clipping, there is no Clip1 here.
                let four = V::splat(4);
                let two = V::splat(2);
                let clip3 = |x: V, v: V| x.sub(two_tc).max(x.add(two_tc).min(v));
                // p0' = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3
                let v = p2
                    .add(p1.sll(1))
                    .add(p0.sll(1))
                    .add(q0.sll(1))
                    .add(q1)
                    .add(four)
                    .sra(3);
                clip3(p0, v).store(out_p[0].as_mut_ptr());
                // p1' = ( p2 + p1 + p0 + q0 + 2 ) >> 2
                let v = p2.add(p1).add(p0).add(q0).add(two).sra(2);
                clip3(p1, v).store(out_p[1].as_mut_ptr());
                // p2' = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3
                let v = p3
                    .sll(1)
                    .add(p2.sll(1))
                    .add(p2)
                    .add(p1)
                    .add(p0)
                    .add(q0)
                    .add(four)
                    .sra(3);
                clip3(p2, v).store(out_p[2].as_mut_ptr());
                // q0' = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3
                let v = p1
                    .add(p0.sll(1))
                    .add(q0.sll(1))
                    .add(q1.sll(1))
                    .add(q2)
                    .add(four)
                    .sra(3);
                clip3(q0, v).store(out_q[0].as_mut_ptr());
                // q1' = ( p0 + q0 + q1 + q2 + 2 ) >> 2
                let v = p0.add(q0).add(q1).add(q2).add(two).sra(2);
                clip3(q1, v).store(out_q[1].as_mut_ptr());
                // q2' = ( p0 + q0 + q1 + 3*q2 + 2*q3 + 4 ) >> 3
                let v = p0
                    .add(q0)
                    .add(q1)
                    .add(q2.sll(1))
                    .add(q2)
                    .add(q3.sll(1))
                    .add(four)
                    .sra(3);
                clip3(q2, v).store(out_q[2].as_mut_ptr());
                return;
            }

            // Weak filter (equations 8-395..8-402).
            let zero = V::splat(0);
            let one = V::splat(1);
            let vhigh = V::splat((1i32 << bit_depth) - 1);
            let clip1 = |x: V| x.max(zero).min(vhigh);
            let a = q0.sub(p0);
            let b = q1.sub(p1);
            // delta = ( 9*(q0 - p0) - 3*(q1 - p1) + 8 ) >> 4
            let delta = a.sll(3).add(a).sub(b.sll(1).add(b)).add(V::splat(8)).sra(4);
            // The rows that pass |delta| < tC * 10 are the filtered ones.
            let keep = V::splat(tc * 10).cmpgt(vabs(delta));
            let d = delta.max(V::splat(-tc)).min(V::splat(tc)); // equation 8-396
            blend(keep, clip1(p0.add(d)), p0).store(out_p[0].as_mut_ptr()); // eq. 8-397
            blend(keep, clip1(q0.sub(d)), q0).store(out_q[0].as_mut_ptr()); // eq. 8-398
            let half_lo = V::splat(-(tc >> 1));
            let half_hi = V::splat(tc >> 1);
            if dep == 1 {
                // equations 8-399 / 8-400.
                let dp = p2
                    .add(p0)
                    .add(one)
                    .sra(1)
                    .sub(p1)
                    .add(d)
                    .sra(1)
                    .max(half_lo)
                    .min(half_hi);
                blend(keep, clip1(p1.add(dp)), p1).store(out_p[1].as_mut_ptr());
            } else {
                p1.store(out_p[1].as_mut_ptr());
            }
            if deq == 1 {
                // equations 8-401 / 8-402.
                let dq = q2
                    .add(q0)
                    .add(one)
                    .sra(1)
                    .sub(q1)
                    .sub(d)
                    .sra(1)
                    .max(half_lo)
                    .min(half_hi);
                blend(keep, clip1(q1.add(dq)), q1).store(out_q[1].as_mut_ptr());
            } else {
                q1.store(out_q[1].as_mut_ptr());
            }
            p2.store(out_p[2].as_mut_ptr());
            q2.store(out_q[2].as_mut_ptr());
        }
    }

    /// Scalar reference: [`super::super::deblock::filter_luma_sample`] per row.
    #[allow(clippy::too_many_arguments)]
    fn filter_luma_rows_scalar(
        p: &LumaSeg,
        q: &LumaSeg,
        de: u8,
        dep: u8,
        deq: u8,
        tc: i32,
        bit_depth: u8,
        out_p: &mut LumaSegOut,
        out_q: &mut LumaSegOut,
    ) {
        for k in 0..4 {
            let row_p = [p[0][k], p[1][k], p[2][k], p[3][k]];
            let row_q = [q[0][k], q[1][k], q[2][k], q[3][k]];
            let out = super::super::deblock::filter_luma_sample(
                row_p, row_q, de, dep, deq, tc, bit_depth,
            );
            // `out.p` / `out.q` hold the input samples wherever the filter
            // did not apply, so copying all three is a no-op there.
            for i in 0..3 {
                out_p[i][k] = out.p[i];
                out_q[i][k] = out.q[i];
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn filter_luma_rows_sse41(
        p: &LumaSeg,
        q: &LumaSeg,
        de: u8,
        dep: u8,
        deq: u8,
        tc: i32,
        bit_depth: u8,
        out_p: &mut LumaSegOut,
        out_q: &mut LumaSegOut,
    ) {
        unsafe {
            filter_luma_rows_simd::<x86::V4>(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn filter_luma_rows_neon(
        p: &LumaSeg,
        q: &LumaSeg,
        de: u8,
        dep: u8,
        deq: u8,
        tc: i32,
        bit_depth: u8,
        out_p: &mut LumaSegOut,
        out_q: &mut LumaSegOut,
    ) {
        unsafe {
            filter_luma_rows_simd::<arm::V4>(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
        }
    }

    /// §8.7.2.5.7 luma filtering of one four-row edge segment.
    ///
    /// `p[i][k]` / `q[i][k]` are the segment's samples (`i` = distance from
    /// the edge, `k` = row along it); `out_p` / `out_q` receive `p0'..p2'` /
    /// `q0'..q2'`. Only `0..nDp` / `0..nDq` of them are replacements, exactly
    /// as for [`super::super::deblock::filter_luma_sample`]; the remaining entries
    /// hold the unmodified input samples.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn filter_luma_rows(
        p: &LumaSeg,
        q: &LumaSeg,
        de: u8,
        dep: u8,
        deq: u8,
        tc: i32,
        bit_depth: u8,
        out_p: &mut LumaSegOut,
        out_q: &mut LumaSegOut,
    ) {
        match isa() {
            // AVX2 has no extra width to spend on a four-row segment.
            #[cfg(target_arch = "x86_64")]
            ISA_AVX2 | ISA_SSE41 => unsafe {
                filter_luma_rows_sse41(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
            },
            #[cfg(target_arch = "aarch64")]
            ISA_NEON => unsafe {
                filter_luma_rows_neon(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
            },
            _ => filter_luma_rows_scalar(p, q, de, dep, deq, tc, bit_depth, out_p, out_q),
        }
    }

    // ---------------------------------------------------------------------------
    // Deblocking chroma filtering (§8.7.2.5.7 equations 8-403..8-405)
    // ---------------------------------------------------------------------------

    /// The four rows of one chroma edge segment, as `p[i][k]` = `pi,k`.
    pub(crate) type ChromaSeg = [[i32; 4]; 2];

    /// Vectorized chroma filtering of a whole four-row edge segment.
    ///
    /// # Safety
    /// The caller must have verified `V`'s CPU feature; `V` must be 4-lane.
    #[inline(always)]
    unsafe fn filter_chroma_rows_simd<V: Ops>(
        p: &ChromaSeg,
        q: &ChromaSeg,
        tc: i32,
        bit_depth: u8,
        out_p0: &mut [i32; 4],
        out_q0: &mut [i32; 4],
    ) {
        unsafe {
            let p0 = V::load(p[0].as_ptr());
            let p1 = V::load(p[1].as_ptr());
            let q0 = V::load(q[0].as_ptr());
            let q1 = V::load(q[1].as_ptr());
            let zero = V::splat(0);
            let vhigh = V::splat((1i32 << bit_depth) - 1);
            // delta = Clip3( -tC, tC, ( ( ( q0 - p0 ) << 2 ) + p1 - q1 + 4 ) >> 3 )
            let d = q0
                .sub(p0)
                .sll(2)
                .add(p1)
                .sub(q1)
                .add(V::splat(4))
                .sra(3)
                .max(V::splat(-tc))
                .min(V::splat(tc));
            p0.add(d).max(zero).min(vhigh).store(out_p0.as_mut_ptr()); // eq. 8-404
            q0.sub(d).max(zero).min(vhigh).store(out_q0.as_mut_ptr()); // eq. 8-405
        }
    }

    /// Scalar reference: [`super::super::deblock::filter_chroma_sample`] per row.
    fn filter_chroma_rows_scalar(
        p: &ChromaSeg,
        q: &ChromaSeg,
        tc: i32,
        bit_depth: u8,
        out_p0: &mut [i32; 4],
        out_q0: &mut [i32; 4],
    ) {
        for k in 0..4 {
            let (a, b) = super::super::deblock::filter_chroma_sample(
                [p[0][k], p[1][k]],
                [q[0][k], q[1][k]],
                tc,
                bit_depth,
            );
            out_p0[k] = a;
            out_q0[k] = b;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn filter_chroma_rows_sse41(
        p: &ChromaSeg,
        q: &ChromaSeg,
        tc: i32,
        bit_depth: u8,
        out_p0: &mut [i32; 4],
        out_q0: &mut [i32; 4],
    ) {
        unsafe { filter_chroma_rows_simd::<x86::V4>(p, q, tc, bit_depth, out_p0, out_q0) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn filter_chroma_rows_neon(
        p: &ChromaSeg,
        q: &ChromaSeg,
        tc: i32,
        bit_depth: u8,
        out_p0: &mut [i32; 4],
        out_q0: &mut [i32; 4],
    ) {
        unsafe { filter_chroma_rows_simd::<arm::V4>(p, q, tc, bit_depth, out_p0, out_q0) }
    }

    /// §8.7.2.5.7 chroma filtering of one four-row edge segment, producing
    /// `p0'` / `q0'` for each row.
    #[inline]
    pub(crate) fn filter_chroma_rows(
        p: &ChromaSeg,
        q: &ChromaSeg,
        tc: i32,
        bit_depth: u8,
        out_p0: &mut [i32; 4],
        out_q0: &mut [i32; 4],
    ) {
        match isa() {
            #[cfg(target_arch = "x86_64")]
            ISA_AVX2 | ISA_SSE41 => unsafe {
                filter_chroma_rows_sse41(p, q, tc, bit_depth, out_p0, out_q0);
            },
            #[cfg(target_arch = "aarch64")]
            ISA_NEON => unsafe {
                filter_chroma_rows_neon(p, q, tc, bit_depth, out_p0, out_q0);
            },
            _ => filter_chroma_rows_scalar(p, q, tc, bit_depth, out_p0, out_q0),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::hevc::engine::deblock::{
            EdgePos, EdgeQp, EdgeType, SamplePlane, filter_chroma_block_edge,
            filter_luma_block_edge,
        };
        use crate::hevc::engine::picture::{Picture, Plane};
        use crate::hevc::engine::sao::{ResolvedSaoComponent, SaoBoundaries, apply_sao_ctb_full};
        use std::sync::MutexGuard;

        /// A pinned reference run: everything inside `f` uses the scalar
        /// kernels.
        ///
        /// Serialization goes through the crate-wide override lock rather than
        /// a local mutex: [`FORCE_SCALAR`] and `crate::simd`'s override now
        /// both steer this dispatcher, so these tests have to exclude the AV1
        /// ones that pin an instruction set too.
        fn with_scalar<T>(f: impl FnOnce() -> T) -> (T, MutexGuard<'static, ()>) {
            let guard = crate::simd::test_lock();
            FORCE_SCALAR.store(true, Ordering::SeqCst);
            let out = f();
            FORCE_SCALAR.store(false, Ordering::SeqCst);
            (out, guard)
        }

        /// Deterministic xorshift so failures reproduce exactly.
        struct Rng(u64);
        impl Rng {
            fn new(seed: u64) -> Self {
                Rng(seed | 1)
            }
            fn next(&mut self) -> u32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                (x >> 32) as u32
            }
            fn sample(&mut self, bit_depth: u8) -> i32 {
                (self.next() % (1u32 << bit_depth)) as i32
            }
        }

        fn offsets(rng: &mut Rng, scale: i32) -> [i32; 5] {
            let mut o = [0i32; 5];
            for v in o.iter_mut().skip(1) {
                *v = (rng.next() % (2 * scale as u32 + 1)) as i32 - scale;
            }
            o
        }

        #[test]
        fn sao_band_row_kernel_is_bit_exact_across_bands_and_bit_depths() {
            // Hold PIN so no concurrently running test can pin the
            // scalar path underneath us; this must exercise the vector one.
            let _pin = crate::simd::test_lock();
            let mut rng = Rng::new(0x5A0B);
            for &bit_depth in &[8u8, 10, 12] {
                let max = (1i32 << bit_depth) - 1;
                let band_shift = i32::from(bit_depth) - 5;
                for band_position in 0..32i32 {
                    let off = offsets(&mut rng, 7 << (bit_depth - 8));
                    // Lengths straddle every vector width so the scalar tail
                    // of each kernel is exercised too.
                    for len in 0..40usize {
                        let src: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                        let mut want = vec![0i32; len];
                        let mut got = vec![0i32; len];
                        sao_band_row_scalar(&src, &mut want, &off, band_position, band_shift, max);
                        sao_band_row(&src, &mut got, &off, band_position, band_shift, max);
                        assert_eq!(got, want, "bd={bit_depth} band={band_position} len={len}");
                    }
                }
            }
        }

        #[test]
        fn sao_edge_row_kernel_is_bit_exact_for_every_sign_pattern() {
            // Hold PIN so no concurrently running test can pin the
            // scalar path underneath us; this must exercise the vector one.
            let _pin = crate::simd::test_lock();
            let mut rng = Rng::new(0xED9E);
            for &bit_depth in &[8u8, 10, 12] {
                let max = (1i32 << bit_depth) - 1;
                let off = offsets(&mut rng, 7 << (bit_depth - 8));
                for len in 0..40usize {
                    // Neighbours drawn from a tiny alphabet so all nine
                    // (sign, sign) combinations of equation 8-411 appear.
                    let src: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                    let n0: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                    let n1: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                    let mut want = vec![0i32; len];
                    let mut got = vec![0i32; len];
                    sao_edge_row_scalar(&src, &n0, &n1, &mut want, &off, max);
                    sao_edge_row(&src, &n0, &n1, &mut got, &off, max);
                    assert_eq!(got, want, "sign sweep bd={bit_depth} len={len}");
                    // ... and again over the full sample range, which also
                    // exercises the equation 8-413 clip at both ends.
                    let src: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                    let n0: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                    let n1: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                    let mut want = vec![0i32; len];
                    let mut got = vec![0i32; len];
                    sao_edge_row_scalar(&src, &n0, &n1, &mut want, &off, max);
                    sao_edge_row(&src, &n0, &n1, &mut got, &off, max);
                    assert_eq!(got, want, "range sweep bd={bit_depth} len={len}");
                }
            }
        }

        #[test]
        fn deblock_luma_rows_are_bit_exact_across_decisions_and_tc() {
            // Hold PIN so no concurrently running test can pin the
            // scalar path underneath us; this must exercise the vector one.
            let _pin = crate::simd::test_lock();
            let mut rng = Rng::new(0xDEB1);
            for &bit_depth in &[8u8, 10] {
                // Every tC the §8.7.2.5.3 table can produce at this depth.
                for q_tc in 0..=53i32 {
                    let tc = super::super::super::deblock::tc_prime(q_tc) * (1 << (bit_depth - 8));
                    for de in 1..=2u8 {
                        for dep in 0..=1u8 {
                            for deq in 0..=1u8 {
                                for _ in 0..8 {
                                    let mut p: LumaSeg = [[0; 4]; 4];
                                    let mut q: LumaSeg = [[0; 4]; 4];
                                    for i in 0..4 {
                                        for k in 0..4 {
                                            p[i][k] = rng.sample(bit_depth);
                                            q[i][k] = rng.sample(bit_depth);
                                        }
                                    }
                                    let mut want_p: LumaSegOut = [[0; 4]; 3];
                                    let mut want_q: LumaSegOut = [[0; 4]; 3];
                                    let mut got_p: LumaSegOut = [[0; 4]; 3];
                                    let mut got_q: LumaSegOut = [[0; 4]; 3];
                                    filter_luma_rows_scalar(
                                        &p,
                                        &q,
                                        de,
                                        dep,
                                        deq,
                                        tc,
                                        bit_depth,
                                        &mut want_p,
                                        &mut want_q,
                                    );
                                    filter_luma_rows(
                                        &p, &q, de, dep, deq, tc, bit_depth, &mut got_p, &mut got_q,
                                    );
                                    let ndp = if de == 2 { 3 } else { (dep + 1) as usize };
                                    let ndq = if de == 2 { 3 } else { (deq + 1) as usize };
                                    assert_eq!(
                                        got_p[..ndp],
                                        want_p[..ndp],
                                        "p side bd={bit_depth} tc={tc} dE={de} dEp={dep}"
                                    );
                                    assert_eq!(
                                        got_q[..ndq],
                                        want_q[..ndq],
                                        "q side bd={bit_depth} tc={tc} dE={de} dEq={deq}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn deblock_chroma_rows_are_bit_exact_across_tc() {
            // Hold PIN so no concurrently running test can pin the
            // scalar path underneath us; this must exercise the vector one.
            let _pin = crate::simd::test_lock();
            let mut rng = Rng::new(0xC780);
            for &bit_depth in &[8u8, 10] {
                for q_tc in 0..=53i32 {
                    let tc = super::super::super::deblock::tc_prime(q_tc) * (1 << (bit_depth - 8));
                    for _ in 0..16 {
                        let mut p: ChromaSeg = [[0; 4]; 2];
                        let mut q: ChromaSeg = [[0; 4]; 2];
                        for i in 0..2 {
                            for k in 0..4 {
                                p[i][k] = rng.sample(bit_depth);
                                q[i][k] = rng.sample(bit_depth);
                            }
                        }
                        let (mut want_p, mut want_q) = ([0i32; 4], [0i32; 4]);
                        let (mut got_p, mut got_q) = ([0i32; 4], [0i32; 4]);
                        filter_chroma_rows_scalar(&p, &q, tc, bit_depth, &mut want_p, &mut want_q);
                        filter_chroma_rows(&p, &q, tc, bit_depth, &mut got_p, &mut got_q);
                        assert_eq!((got_p, got_q), (want_p, want_q), "bd={bit_depth} tc={tc}");
                    }
                }
            }
        }

        /// A `SaoBoundaries` whose CTBs all share one slice and one tile, so
        /// `neighbour_allowed` is always true. Passing it keeps
        /// `apply_sao_ctb_full` on its normative scalar loop with exactly the
        /// semantics of the `None` (vectorized) path.
        fn permissive_boundaries(pic: &Picture, ctb_log2: u32) -> SaoBoundaries {
            let w_ctbs = pic.width_luma().div_ceil(1 << ctb_log2);
            let h_ctbs = pic.height_luma().div_ceil(1 << ctb_log2);
            SaoBoundaries {
                slice_addr_of_ctb: vec![0; w_ctbs * h_ctbs],
                tile_id_of_ctb: vec![0; w_ctbs * h_ctbs],
                pic_w_ctbs: w_ctbs,
                ctb_log2_size_y: ctb_log2,
                across_slices: true,
                across_tiles: true,
                filter_across_of_ctb: None,
                ctb_ts_of_rs: None,
            }
        }

        fn filled_picture(w: usize, h: usize, bit_depth: u8, seed: u64) -> Picture {
            let mut pic = Picture::new(w, h, 1, bit_depth, bit_depth);
            let mut rng = Rng::new(seed);
            for y in 0..h {
                for x in 0..w {
                    pic.set_sample(Plane::Luma, x, y, rng.sample(bit_depth));
                }
            }
            let (cw, ch) = pic.plane_dims(Plane::Cb);
            for y in 0..ch {
                for x in 0..cw {
                    let v = rng.sample(bit_depth);
                    pic.set_sample(Plane::Cb, x, y, v);
                    let v = rng.sample(bit_depth);
                    pic.set_sample(Plane::Cr, x, y, v);
                }
            }
            pic
        }

        #[test]
        fn sao_ctb_vector_path_matches_the_normative_scalar_loop() {
            // Hold PIN so no concurrently running test can pin the
            // scalar path underneath us; this must exercise the vector one.
            let _pin = crate::simd::test_lock();
            // 43 x 37 is deliberately not a multiple of any vector width or
            // of the CTB size, so partial CTBs and scalar tails are covered.
            let mut rng = Rng::new(0x5A0C7B);
            for &bit_depth in &[8u8, 10] {
                let rec = filled_picture(48, 40, bit_depth, 0x1234 + u64::from(bit_depth));
                let bounds = permissive_boundaries(&rec, 4);
                for sao_type_idx in 1..=2u8 {
                    // All four Table 8-13 classes / all 32 band positions.
                    for param in 0..32u8 {
                        let comp = ResolvedSaoComponent {
                            sao_type_idx,
                            offset_val: offsets(&mut rng, 7 << (bit_depth - 8)),
                            band_position: param,
                            eo_class: param & 3,
                        };
                        for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
                            let (pw, ph) = rec.plane_dims(plane);
                            for y_ctb in (0..ph).step_by(16) {
                                for x_ctb in (0..pw).step_by(16) {
                                    let mut want = rec.clone();
                                    let mut got = rec.clone();
                                    apply_sao_ctb_full(
                                        &rec,
                                        &mut want,
                                        plane,
                                        &comp,
                                        x_ctb,
                                        y_ctb,
                                        16,
                                        16,
                                        Some(&bounds),
                                        None,
                                    );
                                    apply_sao_ctb_full(
                                        &rec, &mut got, plane, &comp, x_ctb, y_ctb, 16, 16, None,
                                        None,
                                    );
                                    assert_eq!(
                                        got.plane(plane),
                                        want.plane(plane),
                                        "type={sao_type_idx} param={param} ctb=({x_ctb},{y_ctb}) bd={bit_depth}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn deblock_block_edges_are_bit_exact_across_bs_qp_and_orientation() {
            let (w, h) = (32usize, 32usize);
            for &bit_depth in &[8u8, 10] {
                for edge in [EdgeType::Vertical, EdgeType::Horizontal] {
                    for bs in 1..=2u8 {
                        for qp in (0..=51i32).step_by(3) {
                            for &off in &[-6i32, 0, 6] {
                                let base = filled_picture(w, h, bit_depth, 0xABCD + qp as u64);
                                let qpx = EdgeQp {
                                    qp_q: qp,
                                    qp_p: (qp + 4).min(51),
                                    beta_offset_div2: off,
                                    tc_offset_div2: off,
                                    bit_depth,
                                };
                                let pos = EdgePos { ex: 8, ey: 8, edge };
                                let run = |pic: &mut Picture| {
                                    let (buf, stride) = pic.plane_mut(Plane::Luma);
                                    let mut sp = SamplePlane {
                                        samples: buf,
                                        width: w,
                                        stride,
                                    };
                                    let dec = filter_luma_block_edge(&mut sp, pos, bs, qpx);
                                    let tc = filter_chroma_block_edge(&mut sp, pos, qpx, 0, 1);
                                    (dec, tc)
                                };
                                let mut want = base.clone();
                                let (want_dec, guard) = with_scalar(|| run(&mut want));
                                let mut got = base.clone();
                                let got_dec = run(&mut got);
                                drop(guard);
                                assert_eq!(got_dec, want_dec);
                                assert_eq!(
                                    got.plane(Plane::Luma),
                                    want.plane(Plane::Luma),
                                    "bd={bit_depth} bs={bs} qp={qp} off={off} {edge:?}"
                                );
                            }
                        }
                    }
                }
            }
        }

        /// Reports the measured in-loop-filter speedup on a representative
        /// reconstructed frame. Ignored by default (it is a timing
        /// measurement, not an assertion):
        ///
        /// ```text
        /// cargo test --release --features native --lib \
        ///     hevc::engine::simd::in_loop::tests::bench_in_loop_filters -- --ignored --nocapture
        /// ```
        #[test]
        #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
        fn bench_in_loop_filters() {
            use std::time::{Duration, Instant};
            const W: usize = 1920;
            const H: usize = 1080;
            const CTB: usize = 64;
            // A representative *reconstructed* frame: smooth gradients with a
            // per-16x16-block DC step and light residual noise. Uniform random
            // noise would leave `d >= beta` at every edge, so the §8.7.2.5.3
            // decision would reject every edge and the filter taps would never
            // run; real reconstructions engage both the weak and strong filter.
            let mut rec = Picture::new(W, H, 1, 8, 8);
            let mut noise = Rng::new(0xF00D);
            for y in 0..H {
                for x in 0..W {
                    let smooth = (x * 200 / W + y * 55 / H) as i32;
                    let dc = (((x / 16) * 7 + (y / 16) * 11) % 23) as i32;
                    let v = (smooth + dc + (noise.next() % 5) as i32 - 2).clamp(0, 255);
                    rec.set_sample(Plane::Luma, x, y, v);
                }
            }
            let mut rng = Rng::new(0xBEE5);
            // One representative SAO parameter set per CTB, cycling the
            // band / edge types the way a real slice does.
            let comps: Vec<ResolvedSaoComponent> = (0..(W / CTB + 1) * (H / CTB + 1))
                .map(|i| ResolvedSaoComponent {
                    sao_type_idx: [1u8, 2, 2, 1][i % 4],
                    offset_val: offsets(&mut rng, 7),
                    band_position: (i % 32) as u8,
                    eo_class: (i % 4) as u8,
                })
                .collect();

            let sao_pass = || {
                let mut out = rec.clone();
                let mut c = 0usize;
                for y in (0..H).step_by(CTB) {
                    for x in (0..W).step_by(CTB) {
                        apply_sao_ctb_full(
                            &rec,
                            &mut out,
                            Plane::Luma,
                            &comps[c % comps.len()],
                            x,
                            y,
                            CTB,
                            CTB,
                            None,
                            None,
                        );
                        c += 1;
                    }
                }
                out
            };

            let deblock_pass = || {
                let mut pic = rec.clone();
                let (buf, stride) = pic.plane_mut(Plane::Luma);
                let mut sp = SamplePlane {
                    samples: buf,
                    width: W,
                    stride,
                };
                let qp = EdgeQp {
                    qp_q: 32,
                    qp_p: 30,
                    beta_offset_div2: 0,
                    tc_offset_div2: 0,
                    bit_depth: 8,
                };
                // The §8.7.2.5.1 sampling grid: every 8 samples across the
                // edge, every 4 along it, both orientations.
                for y in (4..H - 8).step_by(4) {
                    for x in (8..W - 8).step_by(8) {
                        let pos = EdgePos {
                            ex: x,
                            ey: y,
                            edge: EdgeType::Vertical,
                        };
                        filter_luma_block_edge(&mut sp, pos, 2, qp);
                    }
                }
                for y in (8..H - 8).step_by(8) {
                    for x in (4..W - 8).step_by(4) {
                        let pos = EdgePos {
                            ex: x,
                            ey: y,
                            edge: EdgeType::Horizontal,
                        };
                        filter_luma_block_edge(&mut sp, pos, 2, qp);
                    }
                }
            };

            // Each round times one pass of each filter on every backend the
            // host offers, and every figure reported below is the *minimum*
            // over the rounds. A single timed pass on a machine that is
            // doing anything else swings by more than 2x — enough to
            // report a real speedup as a regression — so one pass is not
            // a measurement of the kernel at all. The minimum is the round
            // that suffered least interference, which is as close to the
            // kernel's own cost as wall-clock timing gets. The backends
            // alternate inside the round so a burst of interference cannot
            // land on only one of them.
            //
            // Every available instruction set gets its own arm rather than
            // only the detected one: on an AVX2 host that is the only way to
            // read the SSE4.1 kernels, and §8.7.2 deblocking deliberately
            // runs the same 128-bit kernel on both, so the two arms
            // agreeing is itself the expected result there.
            let rounds = 5;
            let reps = 10;
            let isas = crate::simd::available();
            let mut sao = vec![Duration::MAX; isas.len()];
            let mut deblock = vec![Duration::MAX; isas.len()];
            let guard = crate::simd::test_lock();
            for _ in 0..rounds {
                for (slot, &pinned) in isas.iter().enumerate() {
                    crate::simd::set_override(Some(pinned));
                    // A timing arm is only meaningful if the pin reached these
                    // kernels: on a host whose scalar code auto-vectorizes,
                    // "the override did not land" and "the vector path is not
                    // faster here" produce the same numbers.
                    assert_eq!(
                        crate::simd::active_by_site()
                            .into_iter()
                            .find(|(site, _)| *site == "hevc_prediction_filters")
                            .map(|(_, isa)| isa),
                        Some(pinned),
                        "override did not reach the in-loop filters"
                    );
                    let t = Instant::now();
                    for _ in 0..reps {
                        std::hint::black_box(sao_pass());
                    }
                    sao[slot] = sao[slot].min(t.elapsed());
                    let t = Instant::now();
                    for _ in 0..reps {
                        deblock_pass();
                    }
                    deblock[slot] = deblock[slot].min(t.elapsed());
                }
            }
            crate::simd::set_override(None);
            drop(guard);

            let ratio =
                |a: Duration, b: Duration| a.as_secs_f64() / b.as_secs_f64().max(f64::EPSILON);
            println!(
                "in-loop filter benchmark, {W}x{H} luma, best of {rounds} rounds x {reps} frames, \
                 detected isa={}",
                isa()
            );
            let scalar_slot = isas
                .iter()
                .position(|i| *i == crate::simd::SimdIsa::Scalar)
                .unwrap_or(0);
            for (name, times) in [("SAO     ", &sao), ("deblock ", &deblock)] {
                let baseline = times[scalar_slot];
                for (isa, &t) in isas.iter().zip(times.iter()) {
                    println!(
                        "  {name} {:>7} {:>9.3?}  => {:.2}x",
                        isa.name(),
                        t / reps,
                        ratio(baseline, t)
                    );
                }
            }
        }
    }
}
