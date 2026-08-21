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

/// A bounded AV1 range-symbol decoder over the 15-bit CDF domain.
///
/// This is the `init_symbol`/`read_symbol` process from AV1 section 8.2, not a
/// generic arithmetic coder.  In particular, AV1 uses an inverted code value,
/// a 15-bit normalized range, and its specified end padding.  The decoder
/// deliberately has no allocation path and never reads more source bits than
/// its caller supplied.
#[derive(Clone, Debug)]
pub struct Av1SymbolDecoder<'a> {
    input: &'a [u8],
    bit_offset: usize,
    value: u32,
    range: u32,
    remaining_bits: i64,
}

impl<'a> Av1SymbolDecoder<'a> {
    const PROB_SHIFT: u32 = 6;
    const MIN_PROB: u32 = 4;

    pub fn new(input: &'a [u8]) -> Result<Self> {
        if input.is_empty() {
            return malformed("AV1 arithmetic stream is truncated");
        }
        let mut decoder = Self {
            input,
            bit_offset: 0,
            value: 0,
            range: AV1_CDF_MAX.into(),
            remaining_bits: i64::try_from(input.len())
                .ok()
                .and_then(|bytes| bytes.checked_mul(8))
                .and_then(|bits| bits.checked_sub(15))
                .ok_or_else(|| malformed_error("AV1 arithmetic stream is too large"))?,
        };
        let initial_bits = input.len().saturating_mul(8).min(15);
        let initial = decoder.read_bits(initial_bits)?;
        decoder.value = (u32::from(AV1_CDF_MAX) - 1) ^ (initial << (15 - initial_bits));
        Ok(decoder)
    }

    /// Reads one symbol using `cdf`, whose values are upper cumulative bounds
    /// in the inclusive interval `1..=32768`.
    pub fn symbol(&mut self, cdf: &[u16]) -> Result<usize> {
        validate_cdf(cdf)?;
        let symbols = u32::try_from(cdf.len())
            .map_err(|_| malformed_error("AV1 CDF has too many symbols"))?;
        let mut symbol = 0usize;
        let mut lower = self.range;
        let mut upper;
        loop {
            upper = lower;
            let inverse_cdf = u32::from(AV1_CDF_MAX) - u32::from(cdf[symbol]);
            lower =
                ((self.range >> 8) * (inverse_cdf >> Self::PROB_SHIFT)) >> (7 - Self::PROB_SHIFT);
            let index = u32::try_from(symbol)
                .map_err(|_| malformed_error("AV1 CDF symbol index overflows"))?;
            lower = lower
                .checked_add(Self::MIN_PROB * (symbols - index - 1))
                .ok_or_else(|| malformed_error("AV1 arithmetic interval overflow"))?;
            if self.value >= lower {
                break;
            }
            symbol += 1;
            if symbol >= cdf.len() {
                return malformed("AV1 arithmetic code is outside its CDF");
            }
        }
        self.range = upper
            .checked_sub(lower)
            .ok_or_else(|| malformed_error("AV1 arithmetic interval underflow"))?;
        self.value = self
            .value
            .checked_sub(lower)
            .ok_or_else(|| malformed_error("AV1 arithmetic value underflow"))?;
        let shift = 15 - self.range.ilog2();
        self.range <<= shift;
        let available = self.remaining_bits.max(0).min(i64::from(shift)) as usize;
        let bits = self.read_bits(available)?;
        self.value = (bits << (shift - u32::try_from(available).unwrap()))
            ^ (((self.value + 1) << shift) - 1);
        self.remaining_bits -= i64::from(shift);
        Ok(symbol)
    }

    pub fn bits_consumed(&self) -> usize {
        self.bit_offset
    }

    fn read_bits(&mut self, count: usize) -> Result<u32> {
        let end = self
            .bit_offset
            .checked_add(count)
            .ok_or_else(|| malformed_error("AV1 arithmetic bit offset overflow"))?;
        if end > self.input.len().saturating_mul(8) {
            return malformed("AV1 arithmetic stream is truncated");
        }
        let mut value = 0;
        while self.bit_offset < end {
            let byte = self.input[self.bit_offset / 8];
            value = (value << 1) | u32::from((byte >> (7 - (self.bit_offset & 7))) & 1);
            self.bit_offset += 1;
        }
        Ok(value)
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
