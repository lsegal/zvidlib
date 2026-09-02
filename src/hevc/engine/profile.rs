//! Exclusive-time stage attribution for a whole-frame HEVC decode.
//!
//! The per-stage criterion groups in `benches/hevc_decode.rs` measure each
//! kernel in isolation: they say how fast §8.7.3 SAO is, not how much SAO a
//! 1080p frame actually runs. That bounds what vectorizing a stage *could*
//! buy and says nothing about what it *does* buy, which is the gap issue #189
//! asks to close — the whole-frame `hevc_decode/<isa>` arms move only ~1.06x
//! while the individual kernels measure 1.3x-2.4x, and only a share-of-total
//! breakdown explains why.
//!
//! This module is that breakdown. [`scope`] opens a stage; the returned guard
//! closes it on drop. Scopes nest, and time is attributed *exclusively*: the
//! §7.3.8.11 residual parse that runs inside the slice-data walk is charged to
//! [`Stage::Residual`] and subtracted from [`Stage::SliceData`], so the stage
//! shares sum to the profiled total rather than double-counting every level of
//! the call tree.
//!
//! # Cost when off
//!
//! Profiling is off unless [`start`] was called on this thread, and a scope
//! opened while it is off reads one thread-local `Cell<bool>` and returns an
//! inert guard. That is cheap enough to leave the instrumentation on the
//! ordinary decode path rather than behind a cargo feature, which matters:
//! a feature-gated profiler measures a build nobody ships. The scopes are also
//! placed per block, per prediction unit and per picture — never per sample or
//! per coefficient — so even with profiling *on* the two `Instant::now` calls
//! per scope stay small against the work they bracket. [`Report::overhead`]
//! quantifies what is left.
//!
//! # Sub-stages
//!
//! Three variants — [`Stage::InterPred`], [`Stage::InterPredFilter`] and
//! [`Stage::InterPredWrite`] — decompose one stage rather than naming three,
//! because issue #280 found that the single `inter_pred` row this used to
//! report was a third work no vector kernel touches. Exclusive attribution
//! makes that decomposition free to read: the two inner scopes take their own
//! time and [`Stage::InterPred`] keeps the remainder, so the three sum to what
//! the one row used to say and each is separately actionable.
//!
//! # Threads and wasm
//!
//! State is thread-local, so a profile covers the decode work that happens on
//! the calling thread. The software decoder is single-threaded, so that is the
//! whole decode. On `wasm32` there is no usable [`std::time::Instant`], so
//! every entry point here compiles to a no-op and [`start`] reports failure
//! rather than returning zeroed timings that read as a measurement.

use std::cell::{Cell, RefCell};
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

/// A stage of the decode pipeline, in the order issue #189 names them.
///
/// The set is deliberately the one an optimization decision turns on: each
/// variant is either a place a vector kernel already exists, a place one could
/// exist, or a place one provably cannot ([`Stage::SliceData`] and
/// [`Stage::Residual`] are both serial arithmetic decoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Stage {
    /// NAL unescaping, parameter-set parsing and §7.3.6 slice segment headers.
    HeaderParse = 0,
    /// §7.3.8 slice-segment-data CABAC decode *other than* residual coding:
    /// split flags, prediction modes, motion data and SAO parameters.
    SliceData = 1,
    /// §7.3.8.11 residual coding — the coefficient-level CABAC decode.
    Residual = 2,
    /// §8.6 dequantization and the inverse DCT/DST.
    InverseTransform = 3,
    /// §8.4.4.2 intra prediction, plus the §8.6.7 residual add and store.
    IntraPred = 4,
    /// §8.5.3.3 inter prediction, *excluding* the two sub-stages below: the
    /// §8.5.3.2 reference-plane setup and the per-prediction-unit allocation
    /// that the interpolation and the write-back happen between.
    InterPred = 5,
    /// §8.5.3.3.3 fractional-sample interpolation and the §8.5.3.3.4 weighted
    /// combine — the part of inter prediction the `engine::simd` kernels are.
    InterPredFilter = 11,
    /// §8.4.4.1 `recSamples = Clip1( predSamples + resSamples )`: writing each
    /// prediction unit's luma and chroma prediction, plus its residual, back
    /// into the picture. Inside inter prediction, reached by no vector kernel.
    InterPredWrite = 12,
    /// §8.5.3.2 motion-vector derivation: merge and AMVP candidate lists.
    MotionDerive = 10,
    /// §8.7.2 in-loop deblocking.
    Deblock = 6,
    /// §8.7.3 sample adaptive offset, *excluding* the two sub-stages below:
    /// the §7.4.9.3 `SaoOffsetVal` resolution over the CTB grid and the
    /// §8.7.3.1 boundary-constraint grids built for it. Reached by no vector
    /// kernel.
    Sao = 7,
    /// The §8.7.3.1 `saoPicture = recPicture` snapshot the CTB loop classifies
    /// against — a whole-picture copy, per picture. Inside SAO, reached by no
    /// vector kernel.
    SaoSnapshot = 13,
    /// §8.7.3.2 the per-CTB modification process itself — the band and edge
    /// classifiers the `engine::simd::in_loop` kernels are.
    SaoFilter = 14,
    /// §8.3 reference marking, DPB insertion and reorder/output handling.
    DpbOutput = 8,
    /// Decoded-picture to RGBA conversion on the way out of the decoder.
    ColorConvert = 9,
}

/// Number of [`Stage`] variants — the width of every accumulator array here.
pub const STAGE_COUNT: usize = 15;

/// Every [`Stage`], in declaration order, for reporting.
pub const STAGES: [Stage; STAGE_COUNT] = [
    Stage::HeaderParse,
    Stage::SliceData,
    Stage::Residual,
    Stage::InverseTransform,
    Stage::IntraPred,
    Stage::InterPred,
    Stage::InterPredFilter,
    Stage::InterPredWrite,
    Stage::MotionDerive,
    Stage::Deblock,
    Stage::Sao,
    Stage::SaoSnapshot,
    Stage::SaoFilter,
    Stage::DpbOutput,
    Stage::ColorConvert,
];

impl Stage {
    /// A short, stable name for tables and reports.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Stage::HeaderParse => "header_parse",
            Stage::SliceData => "slice_data_cabac",
            Stage::Residual => "residual_cabac",
            Stage::InverseTransform => "inverse_transform",
            Stage::IntraPred => "intra_pred",
            Stage::InterPred => "inter_pred_setup",
            Stage::InterPredFilter => "inter_pred_filter",
            Stage::InterPredWrite => "inter_pred_write",
            Stage::MotionDerive => "motion_derive",
            Stage::Deblock => "deblock",
            Stage::Sao => "sao_setup",
            Stage::SaoSnapshot => "sao_snapshot",
            Stage::SaoFilter => "sao_filter",
            Stage::DpbOutput => "dpb_output",
            Stage::ColorConvert => "color_convert",
        }
    }

    /// Whether this stage reaches one of the crate's HEVC vector kernels.
    ///
    /// This is the classification the Amdahl ceiling in [`Report`] is computed
    /// from: the sum of the vectorized stages' shares is the only part of a
    /// decode any amount of SIMD can move.
    ///
    /// [`Stage::ColorConvert`] counts here as of issue #219, which gave the
    /// output conversion a kernel of its own — it is not decoding, but it is
    /// on the path of every whole-frame measurement, so it is part of what
    /// SIMD moves in the number those groups report. It still contributes
    /// nothing to [`Report::vectorized_decode_share`], whose denominator
    /// excludes it.
    #[must_use]
    pub fn is_vectorized(self) -> bool {
        matches!(
            self,
            Stage::InverseTransform
                | Stage::IntraPred
                | Stage::InterPredFilter
                | Stage::Deblock
                | Stage::SaoFilter
                | Stage::ColorConvert
        )
    }
}

/// One thread's in-flight profile.
struct State {
    /// Exclusive nanoseconds charged to each stage.
    nanos: [u64; STAGE_COUNT],
    /// How many scopes each stage opened, so a share can be read next to a
    /// call count when a stage looks surprisingly cheap or dear.
    entries: [u64; STAGE_COUNT],
    /// Open scopes, innermost last, each with the instant its *current*
    /// uninterrupted run began.
    #[cfg(not(target_arch = "wasm32"))]
    stack: Vec<(Stage, Instant)>,
    /// When the profile started, for the wall-clock denominator.
    #[cfg(not(target_arch = "wasm32"))]
    started: Instant,
    /// Scopes opened and closed, for the overhead estimate.
    scopes: u64,
}

thread_local! {
    /// Fast path: a scope opened while this is `false` does nothing else.
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Begins profiling on the calling thread, discarding any previous profile.
///
/// Returns `false` — and profiles nothing — on `wasm32`, where there is no
/// monotonic clock to attribute against.
pub fn start() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        STATE.with(|state| {
            *state.borrow_mut() = Some(State {
                nanos: [0; STAGE_COUNT],
                entries: [0; STAGE_COUNT],
                stack: Vec::with_capacity(8),
                started: Instant::now(),
                scopes: 0,
            });
        });
        ENABLED.with(|enabled| enabled.set(true));
        true
    }
}

/// Ends profiling on the calling thread and returns what it attributed.
///
/// Returns `None` when [`start`] was never called (or returned `false`). Any
/// scope still open is closed first, so a profile finished mid-decode is
/// self-consistent rather than short by the open frames.
pub fn finish() -> Option<Report> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !ENABLED.with(std::cell::Cell::get) {
            return None;
        }
        ENABLED.with(|enabled| enabled.set(false));
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let mut state = state.take()?;
            let now = Instant::now();
            // Charge whatever is still open, innermost first, so an early
            // finish loses no time rather than silently dropping it.
            while let Some((stage, since)) = state.stack.pop() {
                state.nanos[stage as usize] += now.duration_since(since).as_nanos() as u64;
            }
            let total = now.duration_since(state.started);
            Some(Report {
                nanos: state.nanos,
                entries: state.entries,
                total,
                scopes: state.scopes,
            })
        })
    }
}

/// Opens `stage`, returning a guard that closes it when dropped.
///
/// While the guard is alive, time is charged to `stage`; time spent inside a
/// *nested* scope is charged to that inner stage instead and resumes here when
/// the inner guard drops.
#[must_use]
pub fn scope(stage: Stage) -> StageGuard {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = stage;
        StageGuard { active: false }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !ENABLED.with(std::cell::Cell::get) {
            return StageGuard { active: false };
        }
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let Some(state) = state.as_mut() else {
                return StageGuard { active: false };
            };
            let now = Instant::now();
            if let Some((parent, since)) = state.stack.last_mut() {
                let parent = *parent;
                let elapsed = now.duration_since(*since).as_nanos() as u64;
                *since = now;
                state.nanos[parent as usize] += elapsed;
            }
            state.entries[stage as usize] += 1;
            state.scopes += 1;
            state.stack.push((stage, now));
            StageGuard { active: true }
        })
    }
}

/// The guard returned by [`scope`]; closing it charges the elapsed time.
#[derive(Debug)]
pub struct StageGuard {
    active: bool,
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.active {
            STATE.with(|state| {
                let mut state = state.borrow_mut();
                let Some(state) = state.as_mut() else {
                    return;
                };
                let now = Instant::now();
                if let Some((stage, since)) = state.stack.pop() {
                    state.nanos[stage as usize] += now.duration_since(since).as_nanos() as u64;
                }
                // The parent resumes from now, so the time this scope owned is
                // not charged to it a second time.
                if let Some((_, since)) = state.stack.last_mut() {
                    *since = now;
                }
            });
        }
    }
}

/// One thread's finished stage attribution.
#[derive(Debug, Clone)]
pub struct Report {
    nanos: [u64; STAGE_COUNT],
    entries: [u64; STAGE_COUNT],
    total: Duration,
    scopes: u64,
}

impl Report {
    /// Exclusive time charged to `stage`.
    #[must_use]
    pub fn stage(&self, stage: Stage) -> Duration {
        Duration::from_nanos(self.nanos[stage as usize])
    }

    /// How many scopes `stage` opened.
    #[must_use]
    pub fn entries(&self, stage: Stage) -> u64 {
        self.entries[stage as usize]
    }

    /// Wall-clock time between [`start`] and [`finish`].
    #[must_use]
    pub fn total(&self) -> Duration {
        self.total
    }

    /// Total time charged to any stage.
    #[must_use]
    pub fn attributed(&self) -> Duration {
        Duration::from_nanos(self.nanos.iter().sum())
    }

    /// Profiled time no stage claimed.
    ///
    /// This is real decode work outside every instrumented scope — bitstream
    /// bookkeeping, allocation, the per-CTU walks between stages — not an
    /// error term, and it is reported as its own row rather than spread across
    /// the stages so a share is never inflated by work it does not do.
    #[must_use]
    pub fn unattributed(&self) -> Duration {
        self.total.saturating_sub(self.attributed())
    }

    /// `stage`'s share of [`Report::total`], in `0.0..=1.0`.
    #[must_use]
    pub fn share(&self, stage: Stage) -> f64 {
        let total = self.total.as_nanos();
        if total == 0 {
            return 0.0;
        }
        self.nanos[stage as usize] as f64 / total as f64
    }

    /// Profiled time other than [`Stage::ColorConvert`].
    ///
    /// Colour conversion is on the path every whole-frame measurement takes
    /// but is not decoding, and it is large enough on this crate's sample to
    /// move every other stage's share materially. Reporting both denominators
    /// keeps "share of what the benchmark measures" and "share of the decoder"
    /// from being confused for each other.
    #[must_use]
    pub fn decode_total(&self) -> Duration {
        self.total.saturating_sub(self.stage(Stage::ColorConvert))
    }

    /// `stage`'s share of [`Report::decode_total`], in `0.0..=1.0`.
    #[must_use]
    pub fn decode_share(&self, stage: Stage) -> f64 {
        let total = self.decode_total().as_nanos();
        if total == 0 || stage == Stage::ColorConvert {
            return 0.0;
        }
        self.nanos[stage as usize] as f64 / total as f64
    }

    /// The combined share of every stage with a vector kernel.
    ///
    /// This is the fraction of a decode SIMD can touch at all: by Amdahl, a
    /// kernel speedup of `s` over this fraction `p` bounds whole-frame speedup
    /// at `1 / (1 - p + p / s)`, and at `s → ∞` at `1 / (1 - p)`.
    #[must_use]
    pub fn vectorized_share(&self) -> f64 {
        STAGES
            .iter()
            .filter(|stage| stage.is_vectorized())
            .map(|stage| self.share(*stage))
            .sum()
    }

    /// [`Report::vectorized_share`] against [`Report::decode_total`].
    #[must_use]
    pub fn vectorized_decode_share(&self) -> f64 {
        STAGES
            .iter()
            .filter(|stage| stage.is_vectorized())
            .map(|stage| self.decode_share(*stage))
            .sum()
    }

    /// The largest whole-frame speedup vectorized stages could ever produce,
    /// even if every one of them became infinitely fast.
    #[must_use]
    pub fn max_whole_frame_speedup(&self) -> f64 {
        // Division already yields infinity when no serial time is left, so
        // this needs no branch — only a clamp against a negative denominator.
        let serial = (1.0 - self.vectorized_share()).max(0.0);
        1.0 / serial
    }

    /// Whole-frame speedup implied by making every vectorized stage `factor`
    /// times faster, holding everything else fixed (Amdahl's law).
    #[must_use]
    pub fn speedup_at(&self, factor: f64) -> f64 {
        if factor <= 0.0 {
            return 1.0;
        }
        let p = self.vectorized_share();
        1.0 / (1.0 - p + p / factor)
    }

    /// A rough upper bound on the profiler's own cost, at 50 ns per scope for
    /// its two clock reads and thread-local hops.
    ///
    /// It is an upper bound rather than a measurement, and it is reported so a
    /// breakdown can be discarded when instrumentation is a material share of
    /// what it measured.
    #[must_use]
    pub fn overhead(&self) -> Duration {
        Duration::from_nanos(self.scopes.saturating_mul(50))
    }

    /// How many scopes the profile opened in total.
    #[must_use]
    pub fn scopes(&self) -> u64 {
        self.scopes
    }

    /// A Markdown table of the breakdown, heaviest stage first.
    ///
    /// `frames` scales the per-stage column to a per-frame cost; pass the
    /// number of frames the profile covered, or `0` to omit that column's
    /// meaning (it is then reported as the whole-profile time).
    #[must_use]
    pub fn markdown_table(&self, frames: usize) -> String {
        let mut rows: Vec<(Stage, u64)> = STAGES
            .iter()
            .map(|stage| (*stage, self.nanos[*stage as usize]))
            .collect();
        rows.sort_by_key(|(_, nanos)| std::cmp::Reverse(*nanos));
        let divisor = frames.max(1) as f64;
        let mut out = String::from(
            "| Stage | Share of total | Share of decode | ms/frame | Vectorized |\n\
             | --- | ---: | ---: | ---: | --- |\n",
        );
        for (stage, nanos) in rows {
            let decode_share = match stage {
                Stage::ColorConvert => "n/a".to_string(),
                stage => format!("{:.1}%", self.decode_share(stage) * 100.0),
            };
            let vectorized = match stage.is_vectorized() {
                true => "yes",
                false => "no",
            };
            out.push_str(&format!(
                "| `{}` | {:.1}% | {decode_share} | {:.2} | {vectorized} |\n",
                stage.name(),
                self.share(stage) * 100.0,
                nanos as f64 / 1e6 / divisor,
            ));
        }
        let unattributed = self.unattributed().as_nanos() as f64;
        let percent_of = |denominator: u128| match denominator {
            0 => 0.0,
            denominator => unattributed / denominator as f64 * 100.0,
        };
        out.push_str(&format!(
            "| _unattributed_ | {:.1}% | {:.1}% | {:.2} | n/a |\n",
            percent_of(self.total.as_nanos()),
            percent_of(self.decode_total().as_nanos()),
            unattributed / 1e6 / divisor,
        ));
        out
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Busy-wait rather than sleep: the profiler charges elapsed wall time, and
    /// a spin keeps the test's timing independent of scheduler granularity.
    ///
    /// A spin burns *at least* what it was asked for and, on a contended
    /// runner, arbitrarily more — so a test may assert that a stage recorded
    /// no less than the time it spun, and never that two spins kept the ratio
    /// they were asked for. Anything that needs an exact split builds a
    /// [`Report`] with [`report_of`] instead.
    fn spin(duration: Duration) {
        let until = Instant::now() + duration;
        while Instant::now() < until {
            std::hint::spin_loop();
        }
    }

    /// A [`Report`] with the attribution stated outright.
    ///
    /// The arithmetic over a finished report — the shares, the Amdahl
    /// ceilings, the table — is a pure function of the nanos charged to each
    /// stage, so it is tested against durations that are given rather than
    /// measured. `total` is the profiled wall clock; anything it exceeds the
    /// stage times by is unattributed, exactly as a real profile's is.
    fn report_of(stages: &[(Stage, Duration)], total: Duration) -> Report {
        let mut nanos = [0; STAGE_COUNT];
        let mut entries = [0; STAGE_COUNT];
        for (stage, duration) in stages {
            nanos[*stage as usize] += duration.as_nanos() as u64;
            entries[*stage as usize] += 1;
        }
        Report {
            nanos,
            entries,
            total,
            scopes: stages.len() as u64,
        }
    }

    #[test]
    fn scopes_are_inert_until_started() {
        {
            let _guard = scope(Stage::Sao);
            spin(Duration::from_millis(1));
        }
        assert!(finish().is_none());
    }

    #[test]
    fn nested_scopes_charge_time_exclusively() {
        assert!(start());
        {
            let _outer = scope(Stage::SliceData);
            spin(Duration::from_millis(5));
            {
                let _inner = scope(Stage::Residual);
                spin(Duration::from_millis(20));
            }
            spin(Duration::from_millis(5));
        }
        let report = finish().expect("profile was started");
        let outer = report.stage(Stage::SliceData);
        let inner = report.stage(Stage::Residual);
        // The inner stage's 20 ms belongs to it alone, and the outer keeps
        // only the two 5 ms spins it ran outside the nested scope — so each
        // is charged no less than what it spun, and the inner scope's time is
        // charged to it *instead of* the outer rather than to both. Which of
        // the two ends up larger is the runner's business, not the
        // attribution's: a stall inside either spin lands on that stage.
        assert!(inner >= Duration::from_millis(20), "inner was {inner:?}");
        assert!(outer >= Duration::from_millis(10), "outer was {outer:?}");
        assert_eq!(report.entries(Stage::Residual), 1);
        assert_eq!(report.entries(Stage::SliceData), 1);
        assert_eq!(report.attributed(), outer + inner);
        assert!(report.attributed() <= report.total());
    }

    #[test]
    fn finish_closes_stages_left_open() {
        assert!(start());
        let open = scope(Stage::Deblock);
        spin(Duration::from_millis(5));
        let report = finish().expect("profile was started");
        drop(open);
        // The spin burns at least its 5 ms before `finish` reads the clock,
        // so the only way this comes up short is the open scope being dropped
        // rather than closed.
        assert!(report.stage(Stage::Deblock) >= Duration::from_millis(5));
        assert_eq!(report.attributed(), report.stage(Stage::Deblock));
    }

    #[test]
    fn shares_and_ceilings_follow_the_attribution() {
        // 30 ms of serial residual decode against 10 ms of vectorizable SAO filtering:
        // a quarter of the profile is in a vectorized stage, so the ceiling on
        // any whole-frame speedup from SIMD is 1/(1 - 0.25). The durations are
        // stated rather than spun, because what is under test is the
        // arithmetic over an attribution, not how much wall clock two spins
        // happen to take relative to each other on a contended runner.
        let report = report_of(
            &[
                (Stage::Residual, Duration::from_millis(30)),
                (Stage::SaoFilter, Duration::from_millis(10)),
            ],
            Duration::from_millis(40),
        );
        let vector = report.vectorized_share();
        assert!((vector - 0.25).abs() < 1e-9, "vectorized share {vector}");
        // No colour conversion was profiled, so both denominators agree.
        assert!((report.vectorized_decode_share() - vector).abs() < 1e-9);
        assert_eq!(report.decode_total(), report.total());
        assert!(report.markdown_table(1).contains("`residual_cabac`"));
        assert!((report.max_whole_frame_speedup() - 1.0 / 0.75).abs() < 1e-9);
        assert!(report.max_whole_frame_speedup() > 1.0);
        assert!(report.speedup_at(2.0) < report.max_whole_frame_speedup());
        assert!((report.speedup_at(1.0) - 1.0).abs() < 1e-9);
        assert!(!Stage::Residual.is_vectorized());
        assert!(Stage::SaoFilter.is_vectorized());
        assert!(!Stage::Sao.is_vectorized());
        assert!(Stage::ColorConvert.is_vectorized());
    }

    #[test]
    fn a_share_is_the_time_the_profile_charged_the_stage() {
        // The shares a real profile reports are the ones its own recorded
        // attribution implies — the link `shares_and_ceilings_follow_the_attribution`
        // used to cover by spinning, asserted against what the profile
        // charged rather than against what the spins were asked for.
        assert!(start());
        {
            let _serial = scope(Stage::Residual);
            spin(Duration::from_millis(3));
        }
        {
            let _vector = scope(Stage::SaoFilter);
            spin(Duration::from_millis(1));
        }
        let report = finish().expect("profile was started");
        let total = report.total().as_secs_f64();
        let expected = report.stage(Stage::SaoFilter).as_secs_f64() / total;
        let vector = report.vectorized_share();
        assert!(
            (vector - expected).abs() < 1e-9,
            "vectorized share {vector}"
        );
        assert!(
            (report.share(Stage::Residual) - report.stage(Stage::Residual).as_secs_f64() / total)
                .abs()
                < 1e-9
        );
        assert!(vector > 0.0 && vector < 1.0, "vectorized share {vector}");
        assert!((report.vectorized_decode_share() - vector).abs() < 1e-9);
        assert_eq!(report.decode_total(), report.total());
    }

    #[test]
    fn colour_conversion_is_out_of_the_decode_denominator() {
        // The one case where the two denominators are meant to disagree, which
        // an injected report can state exactly: 20 ms of colour conversion in
        // a 100 ms profile is 20% of the total and none of the decode, and
        // every other stage's decode share is against the remaining 80 ms.
        let report = report_of(
            &[
                (Stage::ColorConvert, Duration::from_millis(20)),
                (Stage::SaoFilter, Duration::from_millis(40)),
                (Stage::Residual, Duration::from_millis(40)),
            ],
            Duration::from_millis(100),
        );
        assert_eq!(report.decode_total(), Duration::from_millis(80));
        assert_eq!(report.unattributed(), Duration::ZERO);
        assert!((report.share(Stage::ColorConvert) - 0.2).abs() < 1e-9);
        assert!((report.decode_share(Stage::ColorConvert) - 0.0).abs() < 1e-9);
        assert!((report.share(Stage::SaoFilter) - 0.4).abs() < 1e-9);
        assert!((report.decode_share(Stage::SaoFilter) - 0.5).abs() < 1e-9);
        // `ColorConvert` is vectorized, so it counts towards the share of the
        // total and drops out of the share of the decode.
        assert!((report.vectorized_share() - 0.6).abs() < 1e-9);
        assert!((report.vectorized_decode_share() - 0.5).abs() < 1e-9);
        assert!(report.markdown_table(1).contains("| n/a |"));
    }
}
