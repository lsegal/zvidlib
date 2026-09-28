//! Header bit reading: the `f(n)`, `su(n)`, `ns(n)`, `le(n)`, `leb128()` and
//! `uvlc()` descriptors of AV1 specification section 4.10.

use super::malformed;
use crate::Result;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// The bit position, as `get_position()` returns it.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn bit(&mut self) -> Result<u32> {
        let byte = self
            .data
            .get(self.position >> 3)
            .ok_or_else(|| malformed("AV1 header is truncated"))?;
        let bit = (byte >> (7 - (self.position & 7))) & 1;
        self.position += 1;
        Ok(u32::from(bit))
    }

    /// `f(n)` for n up to 32.
    pub(crate) fn f(&mut self, n: usize) -> Result<u32> {
        let mut x = 0u64;
        for _ in 0..n {
            x = (x << 1) | u64::from(self.bit()?);
        }
        Ok(x as u32)
    }

    pub(crate) fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? == 1)
    }

    /// `su(n)`: an n-bit two's complement signed value.
    pub(crate) fn su(&mut self, n: usize) -> Result<i32> {
        let value = self.f(n)? as i64;
        let sign_mask = 1i64 << (n - 1);
        Ok(if value & sign_mask != 0 {
            (value - 2 * sign_mask) as i32
        } else {
            value as i32
        })
    }

    /// `ns(n)`: a non-symmetric unsigned value in `0..n`.
    pub(crate) fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let v = self.f(w as usize - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// `le(n)`: an n-byte little-endian value.
    pub(crate) fn le(&mut self, n: usize) -> Result<u64> {
        let mut t = 0u64;
        for i in 0..n {
            t += u64::from(self.f(8)?) << (i * 8);
        }
        Ok(t)
    }

    pub(crate) fn leb128(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.f(8)?;
            value |= u64::from(byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(malformed("AV1 leb128 value is longer than eight bytes"))
    }

    pub(crate) fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros = 0;
        while self.bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        Ok(self.f(leading_zeros)? + ((1u64 << leading_zeros) - 1) as u32)
    }

    pub(crate) fn byte_alignment(&mut self) -> Result<()> {
        while self.position & 7 != 0 {
            self.bit()?;
        }
        Ok(())
    }
}
