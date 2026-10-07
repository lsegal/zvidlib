//! The AV1 symbol decoder (specification section 8.2), including the
//! adaptive CDF update.

use super::malformed;
use crate::Result;

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub(crate) struct SymbolDecoder<'a> {
    data: &'a [u8],
    bit_position: usize,
    value: u32,
    range: u32,
    max_bits: i64,
    disable_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol( sz )` over the whole of `data`.
    pub(crate) fn new(data: &'a [u8], disable_update: bool) -> Result<Self> {
        if data.is_empty() {
            return Err(malformed("AV1 tile data is empty"));
        }
        let mut decoder = Self {
            data,
            bit_position: 0,
            value: 0,
            range: 1 << 15,
            max_bits: 8 * data.len() as i64 - 15,
            disable_update,
        };
        let num_bits = (data.len() * 8).min(15);
        let buf = decoder.read_bits(num_bits);
        let padded = buf << (15 - num_bits);
        decoder.value = ((1 << 15) - 1) ^ padded;
        Ok(decoder)
    }

    fn read_bits(&mut self, count: usize) -> u32 {
        let mut x = 0u32;
        for _ in 0..count {
            let byte = self.data.get(self.bit_position >> 3).copied().unwrap_or(0);
            let bit = (byte >> (7 - (self.bit_position & 7))) & 1;
            self.bit_position += 1;
            x = (x << 1) | u32::from(bit);
        }
        x
    }

    /// `read_symbol( cdf )`, where `cdf` holds the N cumulative values
    /// followed by the adaptation counter.
    pub(crate) fn symbol(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let mut cur = self.range;
        let mut symbol: usize = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol]);
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if self.value >= cur {
                break;
            }
            symbol += 1;
            if symbol >= n {
                // Unreachable for a valid CDF (its last value is 32768, which
                // makes `cur` zero); keep a malformed table from indexing out.
                symbol = n - 1;
                break;
            }
        }
        self.range = prev - cur;
        self.value -= cur;
        self.renormalize();
        if !self.disable_update {
            let count = cdf[n];
            let rate =
                3 + u32::from(count > 15) + u32::from(count > 31) + (n as u32).ilog2().min(2);
            let mut tmp = 0u32;
            for (i, value) in cdf.iter_mut().enumerate().take(n - 1) {
                if i == symbol {
                    tmp = 1 << 15;
                }
                let current = u32::from(*value);
                if tmp < current {
                    *value = (current - ((current - tmp) >> rate)) as u16;
                } else {
                    *value = (current + ((tmp - current) >> rate)) as u16;
                }
            }
            if count < 32 {
                cdf[n] = count + 1;
            }
        }
        symbol
    }

    fn renormalize(&mut self) {
        let bits = 15 - self.range.ilog2();
        if bits == 0 {
            return;
        }
        self.range <<= bits;
        let num_bits = (bits as i64).min(self.max_bits.max(0)) as u32;
        let new_data = self.read_bits(num_bits as usize);
        let padded = new_data << (bits - num_bits);
        self.value = padded ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= i64::from(bits);
    }

    /// `read_bool( )`: an equiprobable bit whose CDF is never adapted.
    pub(crate) fn bool(&mut self) -> u32 {
        let mut cdf = [1u16 << 14, 1 << 15, 0];
        let saved = self.disable_update;
        self.disable_update = true;
        let bit = self.symbol(&mut cdf) as u32;
        self.disable_update = saved;
        bit
    }

    /// `L(n)`: `read_literal( n )`.
    pub(crate) fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = 2 * x + self.bool();
        }
        x
    }

    /// `NS(n)`: a non-symmetric value in `0..n` read with `read_literal`.
    pub(crate) fn ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let v = self.literal(w - 1);
        if v < m {
            return v;
        }
        let extra_bit = self.literal(1);
        (v << 1) - m + extra_bit
    }

    /// Whether the decoder has consumed more padding than the specification's
    /// exit process allows, which only a malformed tile can cause.
    pub(crate) fn overran(&self) -> bool {
        self.max_bits < -14
    }
}
