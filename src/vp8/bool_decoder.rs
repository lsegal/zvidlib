//! The VP8 boolean entropy decoder (RFC 6386 section 7).
//!
//! The arithmetic is the reference decoder's; only the bit buffering differs,
//! holding up to eight bytes at once instead of refilling one byte at a time.
//! Reading past the end of a partition supplies zero bits, as libvpx does, so
//! a truncated partition decodes deterministically instead of failing.

const VALUE_BITS: i32 = 64;

pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    position: usize,
    /// The not-yet-consumed bits, most significant first.
    value: u64,
    /// How many valid bits `value` holds beyond its top byte.
    count: i32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            position: 0,
            value: 0,
            count: -8,
            range: 255,
        };
        decoder.fill();
        decoder
    }

    fn fill(&mut self) {
        let mut shift = VALUE_BITS - 8 - (self.count + 8);
        while shift >= 0 {
            let Some(&byte) = self.data.get(self.position) else {
                // Past the end: the remaining bits are zeros, and there is
                // never anything more to fetch.
                self.count = i32::MAX / 2;
                return;
            };
            self.position += 1;
            self.value |= u64::from(byte) << shift;
            self.count += 8;
            shift -= 8;
        }
    }

    #[inline]
    pub(crate) fn read(&mut self, probability: u8) -> bool {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        if self.count < 0 {
            self.fill();
        }
        let big_split = u64::from(split) << (VALUE_BITS - 8);
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        // `range` is in 1..=255 here; shift it back into 128..=255.
        let shift = (self.range as u8).leading_zeros();
        self.range <<= shift;
        self.value <<= shift;
        self.count -= shift as i32;
        bit
    }

    #[inline]
    pub(crate) fn read_flag(&mut self) -> bool {
        self.read(128)
    }

    /// An unsigned `bits`-wide literal, most significant bit first.
    pub(crate) fn read_literal(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_flag());
        }
        value
    }

    /// A magnitude followed by a sign bit.
    pub(crate) fn read_signed(&mut self, bits: u32) -> i32 {
        let magnitude = self.read_literal(bits) as i32;
        if self.read_flag() {
            -magnitude
        } else {
            magnitude
        }
    }

    /// A flag, then a signed value when the flag is set; zero otherwise.
    pub(crate) fn read_optional_signed(&mut self, bits: u32) -> i32 {
        if self.read_flag() {
            self.read_signed(bits)
        } else {
            0
        }
    }

    pub(crate) fn read_tree(&mut self, tree: &[i8], probabilities: &[u8]) -> u8 {
        let mut index = 0usize;
        loop {
            let next = tree[index + usize::from(self.read(probabilities[index >> 1]))];
            if next <= 0 {
                return (-next) as u8;
            }
            index = next as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A boolean encoder (RFC 6386 section 7.3) to round-trip the decoder.
    struct BoolEncoder {
        output: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl BoolEncoder {
        fn new() -> Self {
            Self {
                output: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn add_one_to_output(&mut self) {
            for byte in self.output.iter_mut().rev() {
                if *byte == 255 {
                    *byte = 0;
                } else {
                    *byte += 1;
                    break;
                }
            }
        }

        fn write(&mut self, probability: u8, bit: bool) {
            let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
            if bit {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                if self.bottom & (1 << 31) != 0 {
                    self.add_one_to_output();
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    self.output.push((self.bottom >> 24) as u8);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            for _ in 0..32 {
                self.write(128, false);
            }
            self.output
        }
    }

    #[test]
    fn decodes_what_the_reference_encoder_writes() {
        let mut encoder = BoolEncoder::new();
        let mut expected = Vec::new();
        let mut state = 0x1234_5678_u32;
        for _ in 0..10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let probability = ((state >> 8) & 0xff) as u8;
            let probability = probability.max(1);
            let bit = (state >> 20) % 256 >= u32::from(probability);
            encoder.write(probability, bit);
            expected.push((probability, bit));
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for (index, (probability, bit)) in expected.into_iter().enumerate() {
            assert_eq!(decoder.read(probability), bit, "bit {index}");
        }
    }

    #[test]
    fn reads_zeros_past_the_end() {
        let mut decoder = BoolDecoder::new(&[]);
        for _ in 0..100 {
            assert!(!decoder.read(128));
        }
    }
}
