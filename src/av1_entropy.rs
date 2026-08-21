//! Bounded arithmetic-symbol primitives used by the native AV1 decoder.
//!
//! AV1 entropy coding uses 15-bit cumulative distribution functions (CDFs).
//! This module owns no frame state: callers provide a CDF for every symbol and
//! are responsible for the context-update rules specified by the bitstream.

use crate::{Error, ErrorKind, Result};

/// The terminal value required by an AV1 CDF.
pub const AV1_CDF_MAX: u16 = 32_768;

/// Validates an AV1 cumulative distribution function and returns its symbol
/// count.  Each entry is a symbol's upper bound, so the final entry is both
/// the final symbol's bound and the terminal value.
pub fn validate_cdf(cdf: &[u16]) -> Result<usize> {
    if cdf.is_empty() {
        return malformed("AV1 CDF must describe at least one symbol");
    }
    if *cdf.last().expect("length was checked") != AV1_CDF_MAX {
        return malformed("AV1 CDF terminal value must be 32768");
    }
    let mut previous = 0;
    for &value in cdf {
        if value <= previous {
            return malformed("AV1 CDF values must be strictly increasing");
        }
        previous = value;
    }
    Ok(cdf.len())
}

/// A bounded, MSB-first arithmetic decoder over AV1's 15-bit CDF domain.
///
/// It rejects truncated streams rather than synthesizing padding bits.  The
/// decoder deliberately has no allocation path and consumes at most the bytes
/// supplied by its caller.
#[derive(Clone, Debug)]
pub struct Av1SymbolDecoder<'a> {
    input: &'a [u8],
    bit_offset: usize,
    low: u64,
    high: u64,
    code: u64,
}

impl<'a> Av1SymbolDecoder<'a> {
    const PRECISION: u32 = 32;
    const TOP: u64 = (1u64 << Self::PRECISION) - 1;
    const HALF: u64 = 1u64 << (Self::PRECISION - 1);
    const QUARTER: u64 = Self::HALF >> 1;
    const THREE_QUARTERS: u64 = Self::QUARTER * 3;

    pub fn new(input: &'a [u8]) -> Result<Self> {
        let mut decoder = Self {
            input,
            bit_offset: 0,
            low: 0,
            high: Self::TOP,
            code: 0,
        };
        for _ in 0..Self::PRECISION {
            decoder.code = (decoder.code << 1) | decoder.read_bit()?;
        }
        Ok(decoder)
    }

    /// Reads one symbol using `cdf`, whose values are upper cumulative bounds
    /// in the inclusive interval `1..=32768`.
    pub fn symbol(&mut self, cdf: &[u16]) -> Result<usize> {
        let symbols = validate_cdf(cdf)?;
        let range = self
            .high
            .checked_sub(self.low)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| malformed_error("AV1 arithmetic interval overflow"))?;
        let scaled = (((self.code - self.low + 1) * u64::from(AV1_CDF_MAX) - 1) / range) as u16;
        let symbol = cdf
            .iter()
            .position(|&upper| scaled < upper)
            .ok_or_else(|| malformed_error("AV1 arithmetic code is outside its CDF"))?;
        debug_assert!(symbol < symbols);
        let lower = if symbol == 0 { 0 } else { cdf[symbol - 1] };
        let upper = cdf[symbol];
        self.high = self.low + (range * u64::from(upper) / u64::from(AV1_CDF_MAX)) - 1;
        self.low += range * u64::from(lower) / u64::from(AV1_CDF_MAX);
        self.renormalize()?;
        Ok(symbol)
    }

    pub fn bits_consumed(&self) -> usize {
        self.bit_offset
    }

    fn renormalize(&mut self) -> Result<()> {
        loop {
            if self.high < Self::HALF {
                // The leading bit is zero.
            } else if self.low >= Self::HALF {
                self.low -= Self::HALF;
                self.high -= Self::HALF;
                self.code -= Self::HALF;
            } else if self.low >= Self::QUARTER && self.high < Self::THREE_QUARTERS {
                self.low -= Self::QUARTER;
                self.high -= Self::QUARTER;
                self.code -= Self::QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | self.read_bit()?;
        }
        Ok(())
    }

    fn read_bit(&mut self) -> Result<u64> {
        let byte = *self.input.get(self.bit_offset / 8).ok_or_else(|| {
            Error::new(
                ErrorKind::MalformedMedia,
                "AV1 arithmetic stream is truncated",
            )
        })?;
        let bit = u64::from((byte >> (7 - (self.bit_offset & 7))) & 1);
        self.bit_offset += 1;
        Ok(bit)
    }
}

fn malformed<T>(message: &str) -> Result<T> {
    Err(malformed_error(message))
}

fn malformed_error(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_validation_rejects_invalid_tables() {
        assert_eq!(
            validate_cdf(&[]).unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );
        assert_eq!(
            validate_cdf(&[16_384, 32_767]).unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );
        assert_eq!(
            validate_cdf(&[20_000, 20_000, AV1_CDF_MAX])
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
        assert_eq!(validate_cdf(&[16_384, AV1_CDF_MAX]).unwrap(), 2);
    }

    #[test]
    fn truncated_arithmetic_stream_is_rejected() {
        assert_eq!(
            Av1SymbolDecoder::new(&[]).unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );
    }

    #[test]
    fn a_single_symbol_cdf_decodes_without_context_specific_state() {
        let mut decoder = Av1SymbolDecoder::new(&[0; 8]).unwrap();
        assert_eq!(decoder.symbol(&[AV1_CDF_MAX]).unwrap(), 0);
    }
}
