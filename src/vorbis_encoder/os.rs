//! Numeric helpers from libvorbis' os.h and scales.h.
//!
//! The scales.h macros are type-generic in C; the callers in this port spell
//! out the C promotions explicitly (f32 vs f64) instead of using a generic
//! helper, so only the bit-level helpers live here.

/// `rint()` as used by libvorbis: round to nearest, ties to even (C99 `rint`
/// in the default rounding mode).
///
/// Note: when libvorbis is compiled for Windows, os.h replaces `rint(x)` by
/// `floor((x)+0.5f)` (round half up). The port
/// follows the C99 semantics that every other platform (Linux, macOS, BSD,
/// i.e. distribution and ffmpeg builds) uses. The two only differ for exact
/// `.5` ties, which do occur in floor1's line fit.
#[inline]
pub(crate) fn rint(x: f64) -> f64 {
    x.round_ties_even()
}

/// Port of scales.h `unitnorm`: +-1.0 with the sign of `x`.
#[inline]
pub(crate) fn unitnorm(x: f32) -> f32 {
    f32::from_bits((x.to_bits() & 0x8000_0000) | 0x3f80_0000)
}

/// Port of scales.h `todB` (IEEE-754 bit-pattern approximation of 20log10|x|).
#[inline]
pub(crate) fn todb(x: f32) -> f32 {
    let i = x.to_bits() & 0x7fff_ffff;
    // C: (float)(ix.i * 7.17711438e-7f - 764.6161886f)
    (i as f32) * 7.177_114_4e-7_f32 - 764.616_2_f32
}

/// `ov_ilog` (sharedbook.c): number of significant bits of `v`.
#[inline]
pub(crate) fn ilog(v: u32) -> i32 {
    32 - v.leading_zeros() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todb_matches_c_constants() {
        // the literals must equal C's float literals exactly
        assert_eq!(7.177_114_4e-7_f32, "7.17711438e-7".parse::<f32>().unwrap());
        assert_eq!(764.616_2_f32, "764.6161886".parse::<f32>().unwrap());
        // 1.0 -> ~0 dB
        assert!(todb(1.0).abs() < 0.5);
        assert!((todb(-10.0) - 20.0).abs() < 0.5);
    }

    #[test]
    fn rint_ties_even() {
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(rint(3.5), 4.0);
        assert_eq!(rint(-1.5), -2.0);
        assert_eq!(rint(1.49), 1.0);
    }

    #[test]
    fn unitnorm_sign() {
        assert_eq!(unitnorm(-3.0), -1.0);
        assert_eq!(unitnorm(0.25), 1.0);
        assert_eq!(unitnorm(0.0), 1.0);
    }

    #[test]
    fn ilog_values() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
    }
}
