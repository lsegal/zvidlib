//! Per-picture rate control for the lossy HEVC writer.
//!
//! [`crate::hevc::engine::encoder::lossy`] codes a picture at whatever
//! `SliceQpY` it is handed; deciding that QP against a bitrate is this
//! module's job. The writer is intra-only and every access unit is an IDR, so
//! there is no frame-type hierarchy to distribute a budget across and no
//! propagation to model: each picture gets the same share of the target and is
//! judged on its own.
//!
//! The model is the standard one — a quantizer step doubles the rate every six
//! QP, so `bits(qp) ~= K * 2^(-qp/6)` — used in the only way an encoder with
//! no lookahead can use it: `K` is whatever the *previous* picture's actual
//! size implies, and the next picture's QP is the one that model puts on the
//! budget. That is a one-step feedback loop, not a lookahead rate allocator;
//! it converges within a picture or two of a scene change and tracks a
//! stationary source closely, which is what a target-bitrate operating point
//! on an intra-only writer needs.
//!
//! Everything here is integer arithmetic, including the base-2 logarithm the
//! model needs, because a QP that lands on a different value on a different
//! host would make the bitstream unreproducible — the failure
//! `benches`-adjacent reproducibility tests exist to catch.

/// The `SliceQpY` range the writer accepts at 8-bit depth
/// (`QpBdOffsetY == 0`).
const QP_RANGE: core::ops::RangeInclusive<i32> = 0..=51;

/// How far one picture's QP may move from the last, in QP steps.
///
/// The model's own answer is trusted, but not to jump the whole range on one
/// observation: a single picture that codes very differently from the model's
/// prediction — a cut, a flash — would otherwise slam the quantizer to a limit
/// and spend the next pictures walking back. Six steps is one full doubling of
/// the rate per picture, which is as fast as the loop can move while still
/// being a loop.
const MAX_QP_STEP: i32 = 6;

/// QP the initial-guess model puts at one bit per pixel.
///
/// Measured against the writer this controls rather than assumed: over the
/// QP 26..=51 span where the doubling-per-six-QP slope actually holds for it,
/// an access unit of one bit per luma sample lands between QP 35 and 37.
const QP_AT_UNIT_BPP: i32 = 36;

/// A one-step per-picture QP feedback loop against a target bitrate.
#[derive(Clone, Copy, Debug)]
pub struct RateController {
    /// Bits one picture is allowed, from the target bitrate and the frame
    /// duration. At least 1, so the model never takes the log of zero.
    picture_bits: u64,
    /// The QP the next picture is coded at.
    qp: i32,
}

impl RateController {
    /// A controller for `bits_per_second` over pictures of `frame_duration`
    /// ticks in a `timescale`-tick second, covering `pixels` luma samples.
    ///
    /// The first picture has no observation to work from, so its QP comes from
    /// the budget's bits per pixel alone; from the second on it is the
    /// feedback loop's.
    pub fn new(bits_per_second: u32, timescale: u32, frame_duration: u32, pixels: u64) -> Self {
        let picture_bits = (u64::from(bits_per_second) * u64::from(frame_duration)
            / u64::from(timescale.max(1)))
        .max(1);
        Self {
            picture_bits,
            qp: initial_qp(picture_bits, pixels.max(1)),
        }
    }

    /// The `SliceQpY` the next picture is to be coded at.
    pub fn qp(&self) -> i32 {
        self.qp
    }

    /// The bits one picture is allowed at the configured target.
    pub fn picture_bits(&self) -> u64 {
        self.picture_bits
    }

    /// Fold in what a picture coded at [`Self::qp`] actually cost, moving the
    /// QP the next picture will use.
    pub fn observe(&mut self, coded_bits: u64) {
        let coded = coded_bits.max(1);
        // bits ~= K * 2^(-qp/6), so the QP that would have hit the budget is
        // `qp + 6 * log2(coded / budget)`.
        let error_q8 = log2_q8(coded) - log2_q8(self.picture_bits);
        let step = round_shift(i64::from(error_q8) * 6, 8) as i32;
        self.qp = (self.qp + step.clamp(-MAX_QP_STEP, MAX_QP_STEP))
            .clamp(*QP_RANGE.start(), *QP_RANGE.end());
    }
}

/// The QP to open at, from the budget's bits per pixel.
///
/// The same doubling-per-six-QP model, anchored at the QP this writer spends
/// one bit per luma sample at. A budget the writer cannot reach at any codable
/// QP opens at the closest end of the range and stays there, which is the
/// honest answer for an intra-only writer asked for a rate only inter coding
/// could hit. It only has to be close enough that the feedback loop is
/// correcting rather than searching.
fn initial_qp(picture_bits: u64, pixels: u64) -> i32 {
    // log2(bits per pixel) in Q8, without leaving the integers.
    let log2_bpp_q8 = log2_q8(picture_bits) - log2_q8(pixels);
    let qp = i64::from(QP_AT_UNIT_BPP) - round_shift(i64::from(log2_bpp_q8) * 6, 8);
    (qp as i32).clamp(*QP_RANGE.start(), *QP_RANGE.end())
}

/// `round(x / 2^shift)` for a signed numerator, rounding halves away from
/// zero, since `/` alone truncates towards zero and would bias every negative
/// QP correction towards doing nothing.
fn round_shift(x: i64, shift: u32) -> i64 {
    let half = 1i64 << (shift - 1);
    if x >= 0 {
        (x + half) >> shift
    } else {
        -((-x + half) >> shift)
    }
}

/// Base-2 logarithm of a positive integer in Q8 fixed point.
///
/// The integer part is the position of the leading bit; the fraction comes
/// from one linear interpolation across the mantissa, which is within about
/// 0.09 of the true logarithm — well under the quarter-QP the caller's
/// rounding discards anyway.
fn log2_q8(x: u64) -> i32 {
    let x = x.max(1);
    let integer = x.ilog2();
    // The mantissa in [1, 2) scaled to Q8: 256..=511.
    let mantissa_q8 = if integer >= 8 {
        x >> (integer - 8)
    } else {
        x << (8 - integer)
    };
    (integer as i32) * 256 + (mantissa_q8 as i32 - 256)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 30000/1001 pictures a second, the rate the encoder's own tests use.
    const TIMESCALE: u32 = 30_000;
    const FRAME_DURATION: u32 = 1_001;

    #[test]
    fn log2_q8_tracks_the_real_logarithm() {
        for x in [1u64, 2, 3, 7, 100, 1_000, 65_535, 1 << 20, u64::from(u32::MAX)] {
            let approximated = f64::from(log2_q8(x)) / 256.0;
            let exact = (x as f64).log2();
            assert!(
                (approximated - exact).abs() < 0.09,
                "log2({x}) approximated as {approximated} against {exact}"
            );
        }
        // Exact on the powers of two the interpolation is anchored at.
        for shift in 0..40u32 {
            assert_eq!(log2_q8(1 << shift), (shift as i32) * 256);
        }
    }

    #[test]
    fn the_budget_is_the_bitrate_over_the_frame_rate() {
        let c = RateController::new(1_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        // 1 Mbit/s at 29.97 fps is ~33_366 bits a picture.
        assert_eq!(c.picture_bits(), 1_000_000 * 1_001 / 30_000);
        // Never zero, however small the target, so the model stays defined.
        let starved = RateController::new(1, TIMESCALE, FRAME_DURATION, 64 * 64);
        assert_eq!(starved.picture_bits(), 1);
    }

    #[test]
    fn the_opening_qp_is_coarser_the_tighter_the_budget() {
        // 64x64 at 29.97 pictures a second, so these five targets are 8, 4, 2,
        // 1 and 0.5 bits a luma sample — the span this writer codes across.
        let pixels = 64 * 64;
        let qps: Vec<i32> = [1_000_000u32, 500_000, 250_000, 125_000, 62_000]
            .iter()
            .map(|&bitrate| RateController::new(bitrate, TIMESCALE, FRAME_DURATION, pixels).qp())
            .collect();
        for pair in qps.windows(2) {
            assert!(
                pair[1] > pair[0],
                "a tighter budget must open coarser: {qps:?}"
            );
        }
        assert!(
            qps.iter().all(|qp| QP_RANGE.contains(qp)),
            "the opening QP must be codable: {qps:?}"
        );
        // The anchor itself: one bit per luma sample a picture.
        let one_bpp = RateController::new(125_000, TIMESCALE, FRAME_DURATION, pixels);
        assert!(
            (one_bpp.qp() - QP_AT_UNIT_BPP).abs() <= 1,
            "one bit per pixel should open near {QP_AT_UNIT_BPP}, got {}",
            one_bpp.qp()
        );
        // A rate no codable QP reaches saturates rather than wrapping.
        assert_eq!(
            RateController::new(100, TIMESCALE, FRAME_DURATION, pixels).qp(),
            *QP_RANGE.end()
        );
    }

    #[test]
    fn overshooting_the_budget_coarsens_and_undershooting_refines() {
        let mut c = RateController::new(1_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        let opened = c.qp();
        c.observe(c.picture_bits() * 2);
        assert!(
            c.qp() > opened,
            "a picture over budget must coarsen: {opened} -> {}",
            c.qp()
        );
        let mut c = RateController::new(1_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        c.observe(c.picture_bits() / 4);
        assert!(
            c.qp() < opened,
            "a picture under budget must refine: {opened} -> {}",
            c.qp()
        );
        // Exactly on budget is exactly no movement.
        let mut c = RateController::new(1_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        c.observe(c.picture_bits());
        assert_eq!(c.qp(), opened, "hitting the budget must not move the QP");
    }

    #[test]
    fn one_observation_moves_at_most_a_full_doubling_and_stays_in_range() {
        let mut c = RateController::new(1_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        let opened = c.qp();
        c.observe(u64::from(u32::MAX));
        assert_eq!(
            c.qp(),
            (opened + MAX_QP_STEP).min(*QP_RANGE.end()),
            "a wildly oversized picture must still move one step"
        );
        // Repeated extremes clamp at the ends rather than running away.
        for _ in 0..64 {
            c.observe(u64::from(u32::MAX));
        }
        assert_eq!(c.qp(), *QP_RANGE.end());
        for _ in 0..64 {
            c.observe(1);
        }
        assert_eq!(c.qp(), *QP_RANGE.start());
    }

    /// The loop's whole claim: pointed at a source whose rate follows the
    /// model, it settles on the QP that spends the budget.
    #[test]
    fn the_loop_converges_on_the_qp_that_spends_the_budget() {
        // A synthetic picture costing `K * 2^(-qp/6)` bits.
        let k = 1_000_000.0f64;
        let cost = |qp: i32| (k * 2f64.powf(-f64::from(qp) / 6.0)) as u64;
        let mut c = RateController::new(2_000_000, TIMESCALE, FRAME_DURATION, 64 * 64);
        for _ in 0..12 {
            let bits = cost(c.qp());
            c.observe(bits);
        }
        let settled = cost(c.qp()) as f64 / c.picture_bits() as f64;
        assert!(
            (0.75..1.35).contains(&settled),
            "the loop settled at QP {} spending {settled:.3} of its budget",
            c.qp()
        );
    }
}

