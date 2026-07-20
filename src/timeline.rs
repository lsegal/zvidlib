use crate::{Error, ErrorKind, Result};
use std::cmp::Ordering;

/// A normalized signed rational number with a positive denominator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    pub fn new(numerator: i64, denominator: i64) -> Result<Self> {
        Self::from_wide(i128::from(numerator), i128::from(denominator))
    }

    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    pub const fn denominator(self) -> i64 {
        self.denominator
    }

    pub const fn is_positive(self) -> bool {
        self.numerator > 0
    }

    pub fn checked_add(self, other: Self) -> Result<Self> {
        let numerator = i128::from(self.numerator) * i128::from(other.denominator)
            + i128::from(other.numerator) * i128::from(self.denominator);
        let denominator = i128::from(self.denominator) * i128::from(other.denominator);
        Self::from_wide(numerator, denominator)
    }

    pub fn checked_sub(self, other: Self) -> Result<Self> {
        let numerator = i128::from(self.numerator) * i128::from(other.denominator)
            - i128::from(other.numerator) * i128::from(self.denominator);
        let denominator = i128::from(self.denominator) * i128::from(other.denominator);
        Self::from_wide(numerator, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self> {
        Self::from_wide(
            i128::from(self.numerator) * i128::from(other.numerator),
            i128::from(self.denominator) * i128::from(other.denominator),
        )
    }

    pub fn checked_div(self, other: Self) -> Result<Self> {
        if other.numerator == 0 {
            return Err(invalid_timeline("cannot divide by zero"));
        }
        Self::from_wide(
            i128::from(self.numerator) * i128::from(other.denominator),
            i128::from(self.denominator) * i128::from(other.numerator),
        )
    }

    fn from_wide(mut numerator: i128, mut denominator: i128) -> Result<Self> {
        if denominator == 0 {
            return Err(invalid_timeline("a rational denominator cannot be zero"));
        }
        if numerator == 0 {
            return Ok(Self {
                numerator: 0,
                denominator: 1,
            });
        }
        if denominator < 0 {
            numerator = numerator
                .checked_neg()
                .ok_or_else(|| timeline_overflow("normalizing a rational"))?;
            denominator = denominator
                .checked_neg()
                .ok_or_else(|| timeline_overflow("normalizing a rational"))?;
        }

        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
        let numerator = numerator / divisor as i128;
        let denominator = denominator / divisor as i128;
        Ok(Self {
            numerator: i64::try_from(numerator)
                .map_err(|_| timeline_overflow("storing a rational numerator"))?,
            denominator: i64::try_from(denominator)
                .map_err(|_| timeline_overflow("storing a rational denominator"))?,
        })
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        (i128::from(self.numerator) * i128::from(other.denominator))
            .cmp(&(i128::from(other.numerator) * i128::from(self.denominator)))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A zero-based presentation-order video frame index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FrameIndex(pub u64);

/// A positive number of frames per second.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameRate(Rational);

impl FrameRate {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        let value = Rational::new(i64::from(numerator), i64::from(denominator))?;
        if !value.is_positive() {
            return Err(invalid_timeline("a frame rate must be positive"));
        }
        Ok(Self(value))
    }

    pub const fn as_rational(self) -> Rational {
        self.0
    }
}

/// A half-open range in an audio track's sample clock.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampleRange {
    pub start: u64,
    pub end: u64,
}

impl SampleRange {
    pub fn new(start: u64, end: u64) -> Result<Self> {
        if end < start {
            return Err(invalid_timeline(
                "a sample range cannot end before it starts",
            ));
        }
        Ok(Self { start, end })
    }

    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// A constant-frame-rate video timeline aligned to an audio sample clock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Timeline {
    frame_rate: FrameRate,
    audio_sample_rate: u32,
}

impl Timeline {
    pub fn new(frame_rate: FrameRate, audio_sample_rate: u32) -> Result<Self> {
        if audio_sample_rate == 0 {
            return Err(invalid_timeline("an audio sample rate must be positive"));
        }
        Ok(Self {
            frame_rate,
            audio_sample_rate,
        })
    }

    pub const fn frame_rate(self) -> FrameRate {
        self.frame_rate
    }

    pub const fn audio_sample_rate(self) -> u32 {
        self.audio_sample_rate
    }

    /// Returns the exact half-open audio interval covered by a video frame.
    ///
    /// Boundaries use floor rounding. Applying the same rule to both sides
    /// guarantees that adjacent frame requests neither duplicate nor lose a
    /// sample, including fractional rates such as 30000/1001.
    pub fn audio_interval_for_frame(self, frame: FrameIndex) -> Result<SampleRange> {
        let next_frame = frame
            .0
            .checked_add(1)
            .ok_or_else(|| timeline_overflow("advancing a frame index"))?;
        let rate = self.frame_rate.as_rational();
        let numerator = u128::try_from(rate.numerator())
            .map_err(|_| invalid_timeline("a frame rate must be positive"))?;
        let denominator = u128::try_from(rate.denominator())
            .map_err(|_| invalid_timeline("a frame-rate denominator must be positive"))?;
        let samples = u128::from(self.audio_sample_rate);

        let boundary = |index: u64| -> Result<u64> {
            let value = u128::from(index)
                .checked_mul(samples)
                .and_then(|value| value.checked_mul(denominator))
                .ok_or_else(|| timeline_overflow("mapping a frame to audio samples"))?
                / numerator;
            u64::try_from(value).map_err(|_| timeline_overflow("storing an audio sample boundary"))
        };

        SampleRange::new(boundary(frame.0)?, boundary(next_frame)?)
    }
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn invalid_timeline(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn timeline_overflow(operation: &str) -> Error {
    Error::new(
        ErrorKind::ResourceLimit,
        format!("timeline overflow while {operation}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_normalizes_sign_and_common_factors() {
        assert_eq!(Rational::new(6, -8), Rational::new(-3, 4));
        assert_eq!(Rational::new(0, -9), Rational::new(0, 1));
    }

    #[test]
    fn rational_rejects_zero_denominator() {
        let error = Rational::new(1, 0).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn rational_arithmetic_is_checked_and_exact() {
        let half = Rational::new(1, 2).unwrap();
        let third = Rational::new(1, 3).unwrap();
        assert_eq!(
            half.checked_add(third).unwrap(),
            Rational::new(5, 6).unwrap()
        );
        assert_eq!(
            half.checked_sub(third).unwrap(),
            Rational::new(1, 6).unwrap()
        );
        assert_eq!(
            half.checked_mul(third).unwrap(),
            Rational::new(1, 6).unwrap()
        );
        assert_eq!(
            half.checked_div(third).unwrap(),
            Rational::new(3, 2).unwrap()
        );
    }

    #[test]
    fn integral_frame_rate_maps_to_integral_sample_ranges() {
        let timeline = Timeline::new(FrameRate::new(30, 1).unwrap(), 44_100).unwrap();
        assert_eq!(
            timeline.audio_interval_for_frame(FrameIndex(5)).unwrap(),
            SampleRange {
                start: 7_350,
                end: 8_820
            }
        );
    }

    #[test]
    fn fractional_frame_rate_has_contiguous_sample_ranges() {
        let timeline = Timeline::new(FrameRate::new(30_000, 1_001).unwrap(), 48_000).unwrap();
        let first = timeline.audio_interval_for_frame(FrameIndex(0)).unwrap();
        let second = timeline.audio_interval_for_frame(FrameIndex(1)).unwrap();
        let third = timeline.audio_interval_for_frame(FrameIndex(2)).unwrap();
        assert_eq!(
            first,
            SampleRange {
                start: 0,
                end: 1_601
            }
        );
        assert_eq!(first.end, second.start);
        assert_eq!(second.end, third.start);
        assert_eq!(third.end, 4_804);
    }

    #[test]
    fn maximum_frame_index_reports_overflow() {
        let timeline = Timeline::new(FrameRate::new(24, 1).unwrap(), 48_000).unwrap();
        let error = timeline
            .audio_interval_for_frame(FrameIndex(u64::MAX))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    }
}
