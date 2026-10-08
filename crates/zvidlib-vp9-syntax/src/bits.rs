//! The two readers of the VP9 bitstream: the plain bit reader for the
//! uncompressed header (section 9.1 of the VP9 specification) and the
//! boolean decoder for the compressed header and tile data (section 9.2).

use super::malformed;
use crate::Result;

/// A most-significant-bit-first reader over the uncompressed header.
pub struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    pub fn bit(&mut self) -> Result<bool> {
        let byte = self
            .data
            .get(self.position >> 3)
            .ok_or_else(|| malformed("VP9 uncompressed header is truncated"))?;
        let bit = (byte >> (7 - (self.position & 7))) & 1;
        self.position += 1;
        Ok(bit == 1)
    }

    pub fn literal(&mut self, bits: u32) -> Result<u32> {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.bit()?);
        }
        Ok(value)
    }

    /// `su(n)`: a magnitude followed by a sign bit.
    pub fn signed_literal(&mut self, bits: u32) -> Result<i32> {
        let value = self.literal(bits)? as i32;
        Ok(if self.bit()? { -value } else { value })
    }

    /// The header's length in whole bytes (`trailing_bits` pad it to a byte).
    pub fn bytes_read(&self) -> usize {
        self.position.div_ceil(8)
    }
}

/// The boolean (arithmetic) decoder, a direct port of libvpx's
/// `vpx_reader`, including its 64-bit window and its end-of-buffer
/// handling: reading past the end yields zero bits, which is what lets
/// [`BoolDecoder::find_end`] report where a frame's data stopped.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    position: usize,
    value: u64,
    count: i32,
    range: u32,
}

const BD_VALUE_SIZE: i32 = 64;
const LOTS_OF_BITS: i32 = 0x4000_0000;

impl<'a> BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(malformed("VP9 bool-coded partition is empty"));
        }
        let mut decoder = Self {
            data,
            position: 0,
            value: 0,
            count: -8,
            range: 255,
        };
        decoder.fill();
        // The marker bit (section 9.2.1) must be zero.
        if decoder.read(128) {
            return Err(malformed("VP9 bool decoder marker bit is set"));
        }
        Ok(decoder)
    }

    fn fill(&mut self) {
        let mut shift = BD_VALUE_SIZE - 8 - (self.count + 8);
        let bytes_left = self.data.len() - self.position;
        let bits_left = (bytes_left * 8) as i64;
        if bits_left > i64::from(BD_VALUE_SIZE) {
            let bits = (shift & !7) + 8;
            let mut window = [0u8; 8];
            window.copy_from_slice(&self.data[self.position..self.position + 8]);
            let big_endian = u64::from_be_bytes(window);
            let next = big_endian >> (BD_VALUE_SIZE - bits);
            self.count += bits;
            self.position += (bits >> 3) as usize;
            self.value |= next << (shift & 7);
        } else {
            let bits_over = shift + 8 - bits_left as i32;
            let mut loop_end = 0;
            if bits_over >= 0 {
                self.count += LOTS_OF_BITS;
                loop_end = bits_over;
            }
            if bits_over < 0 || bits_left != 0 {
                while shift >= loop_end {
                    self.count += 8;
                    self.value |= u64::from(self.data[self.position]) << shift;
                    self.position += 1;
                    shift -= 8;
                }
            }
        }
    }

    #[inline]
    pub fn read(&mut self, probability: u8) -> bool {
        let split = (self.range * u32::from(probability) + (256 - u32::from(probability))) >> 8;
        if self.count < 0 {
            self.fill();
        }
        let big_split = u64::from(split) << (BD_VALUE_SIZE - 8);
        let (range, bit) = if self.value >= big_split {
            self.value -= big_split;
            (self.range - split, true)
        } else {
            (split, false)
        };
        let shift = NORM[range as usize];
        self.range = range << shift;
        self.value <<= shift;
        self.count -= i32::from(shift);
        bit
    }

    #[inline]
    pub fn bit(&mut self) -> bool {
        self.read(128)
    }

    pub fn literal(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.bit());
        }
        value
    }

    /// Reads a symbol from a libvpx-style tree: each node is a pair of
    /// entries, where a non-positive entry is a leaf holding the negated
    /// symbol and a positive one indexes the next node.
    #[inline]
    pub fn tree(&mut self, tree: &[i8], probabilities: &[u8]) -> u8 {
        let mut index = 0i8;
        loop {
            let node = index as usize;
            index = tree[node + usize::from(self.read(probabilities[node >> 1]))];
            if index <= 0 {
                return (-index) as u8;
            }
        }
    }

    /// Whether more bits were read than the partition holds (libvpx's
    /// `vpx_reader_has_error`).
    pub fn has_error(&self) -> bool {
        self.count > BD_VALUE_SIZE && self.count < LOTS_OF_BITS
    }

    /// The offset just past the last byte the decoder consumed (libvpx's
    /// `vpx_reader_find_end`).
    pub fn find_end(&self) -> usize {
        let mut count = self.count;
        let mut position = self.position;
        while count > 8 && count < BD_VALUE_SIZE {
            count -= 8;
            position -= 1;
        }
        position
    }
}

/// `vpx_norm`: the left shift that renormalizes a range back to 128..255.
static NORM: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut index = 1;
    while index < 256 {
        let mut shift = 0;
        while (index << shift) < 128 {
            shift += 1;
        }
        table[index] = shift as u8;
        index += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_matches_libvpx() {
        assert_eq!(NORM[0], 0);
        assert_eq!(NORM[1], 7);
        assert_eq!(NORM[2], 6);
        assert_eq!(NORM[127], 1);
        assert_eq!(NORM[128], 0);
        assert_eq!(NORM[255], 0);
    }

    #[test]
    fn bit_reader_reads_literals_and_signed_values() {
        let mut reader = BitReader::new(&[0b1010_0111, 0b1000_0000]);
        assert!(reader.bit().unwrap());
        assert_eq!(reader.literal(3).unwrap(), 0b010);
        assert_eq!(reader.signed_literal(3).unwrap(), -0b011);
        assert_eq!(reader.bytes_read(), 1);
        assert_eq!(reader.literal(4).unwrap(), 0b1000);
        assert_eq!(reader.bytes_read(), 2);
        assert!(reader.literal(5).is_err());
    }

    #[test]
    fn bool_decoder_rejects_empty_and_marked_partitions() {
        assert!(BoolDecoder::new(&[]).is_err());
        assert!(BoolDecoder::new(&[0xff, 0xff]).is_err());
        let mut decoder = BoolDecoder::new(&[0x00, 0x00]).unwrap();
        assert!(!decoder.bit());
        assert!(!decoder.has_error());
    }
}
