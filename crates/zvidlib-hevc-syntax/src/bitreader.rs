//! MSB-first bit reader for H.265 syntax-element parsing.
//!
//! H.265 syntax elements are read most-significant-bit-first from the
//! RBSP byte stream (see ITU-T Rec. H.265 §7.2 — the `read_bits(n)`
//! function described there returns the next `n` bits as an unsigned
//! integer with the MSB written first). This module provides the
//! minimum surface needed for §7.3.2.1 (VPS) and §7.3.3
//! (profile_tier_level):
//!
//! * [`BitReader::u`] — fixed-width unsigned integer (the `u(n)`
//!   descriptor of §7.2).
//! * [`BitReader::ue`] — 0-th-order unsigned Exp-Golomb (the `ue(v)`
//!   descriptor, parsed per §9.2).
//!
//! The reader takes an already-unescaped RBSP buffer (see
//! [`crate::hevc::engine::nal::strip_emulation_prevention`]) and does not attempt to
//! locate the RBSP stop bit — that is left to the calling parser.

/// Errors that can arise while consuming bits from an RBSP buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitReaderError {
    /// A read requested more bits than remained in the buffer.
    EndOfBuffer,
    /// More than 32 bits requested in a single `u()` read.
    TooManyBits,
    /// A 0-th-order Exp-Golomb code's leading-zero run exceeded 32
    /// (the §9.2 contract caps `codeNum` at `2^32 - 2`). A run of 33
    /// zeros would overflow the 32-bit return type.
    ExpGolombOverflow,
}

impl core::fmt::Display for BitReaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EndOfBuffer => f.write_str("read past end of RBSP buffer"),
            Self::TooManyBits => f.write_str("u(n) called with n > 32"),
            Self::ExpGolombOverflow => f.write_str("ue(v) leading-zero run exceeded 32"),
        }
    }
}

impl std::error::Error for BitReaderError {}

/// MSB-first bit reader.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Bit cursor measured from the start of `buf` (0 == MSB of
    /// `buf[0]`).
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Construct a bit reader over `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Number of bits still available.
    pub fn bits_left(&self) -> usize {
        self.buf
            .len()
            .saturating_mul(8)
            .saturating_sub(self.bit_pos)
    }

    /// Read a single bit (the `u(1)` descriptor).
    #[inline]
    pub fn u1(&mut self) -> Result<u8, BitReaderError> {
        if self.bit_pos >= self.buf.len() * 8 {
            return Err(BitReaderError::EndOfBuffer);
        }
        let byte = self.buf[self.bit_pos / 8];
        let bit = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        Ok((byte >> bit) & 0x01)
    }

    /// Read `n` bits MSB-first as an unsigned integer (the `u(n)`
    /// descriptor, with `n` in `0..=32`). `n == 0` returns 0 and does
    /// not advance the cursor.
    #[inline]
    pub fn u(&mut self, n: u8) -> Result<u32, BitReaderError> {
        if n > 32 {
            return Err(BitReaderError::TooManyBits);
        }
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(BitReaderError::EndOfBuffer);
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | (self.u1()? as u32);
        }
        Ok(value)
    }

    /// Skip `n` bits without interpreting them. Useful when a syntax
    /// element appears in the bitstream but is not needed by the
    /// caller (e.g. the 32 profile_compatibility flags after the
    /// caller has already recorded the few of interest).
    pub fn skip(&mut self, n: usize) -> Result<(), BitReaderError> {
        if self.bits_left() < n {
            return Err(BitReaderError::EndOfBuffer);
        }
        self.bit_pos += n;
        Ok(())
    }

    /// Read a 0-th-order unsigned Exp-Golomb code (the `ue(v)`
    /// descriptor). Per ITU-T Rec. H.265 §9.2: count `leadingZeroBits`
    /// 0 bits until the first 1 bit is seen; then read another
    /// `leadingZeroBits` bits and compute
    /// `codeNum = 2^leadingZeroBits + read_bits(leadingZeroBits) - 1`.
    pub fn ue(&mut self) -> Result<u32, BitReaderError> {
        let mut leading_zero_bits: u32 = 0;
        loop {
            if leading_zero_bits > 32 {
                return Err(BitReaderError::ExpGolombOverflow);
            }
            let b = self.u1()?;
            if b == 1 {
                break;
            }
            leading_zero_bits += 1;
        }
        if leading_zero_bits == 0 {
            return Ok(0);
        }
        if leading_zero_bits == 32 {
            // 2^32 would overflow u32; the suffix must therefore be 0
            // to keep `codeNum` representable. Treat any other input
            // as an overflow.
            let suffix = self.u(32)?;
            return suffix
                .checked_sub(1)
                .and_then(|v| (1u64 << 32).checked_add(v as u64))
                .and_then(|big| u32::try_from(big).ok())
                .ok_or(BitReaderError::ExpGolombOverflow);
        }
        let suffix = self.u(leading_zero_bits as u8)?;
        Ok((1u32 << leading_zero_bits) - 1 + suffix)
    }

    /// Read a 0-th-order signed Exp-Golomb code (the `se(v)`
    /// descriptor). Per ITU-T Rec. H.265 §9.2.2: read a `ue(v)`
    /// `codeNum`, then map it to a signed value per Table 9-3 as
    /// `(-1)^(codeNum + 1) * Ceil(codeNum / 2)` — `codeNum` 0 → 0,
    /// 1 → +1, 2 → -1, 3 → +2, 4 → -2, …
    pub fn se(&mut self) -> Result<i32, BitReaderError> {
        let code_num = self.ue()?;
        // Ceil(codeNum / 2) without overflow: (codeNum + 1) / 2.
        let magnitude = ((code_num as i64) + 1) / 2;
        let value = if code_num % 2 == 0 {
            // Even codeNum (0, 2, 4, …) maps to a non-positive value
            // (0, -1, -2, …).
            -magnitude
        } else {
            // Odd codeNum (1, 3, 5, …) maps to a positive value.
            magnitude
        };
        // `value` is bounded by ±2^31 because `ue()` caps `codeNum`
        // at `2^32 - 2`, whose mapped magnitude is `2^31 - 1`.
        i32::try_from(value).map_err(|_| BitReaderError::ExpGolombOverflow)
    }

    /// Current bit position from the start of the buffer.
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn reads_individual_bits_msb_first() {
        // 0b1010_0110, 0b1100_0011 → bits: 1 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1
        let buf = [0xA6, 0xC3];
        let mut br = BitReader::new(&buf);
        let want = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1];
        for &w in &want {
            assert_eq!(br.u1().unwrap(), w);
        }
        assert!(br.u1().is_err());
    }

    #[test]
    fn reads_u_n_across_byte_boundary() {
        // Two bytes: 0xAB, 0xCD → 0b1010_1011_1100_1101
        let buf = [0xAB, 0xCD];
        let mut br = BitReader::new(&buf);
        // Take 12 bits: 0b1010_1011_1100 = 0xABC
        assert_eq!(br.u(12).unwrap(), 0xABC);
        // Remaining 4 bits: 0xD
        assert_eq!(br.u(4).unwrap(), 0xD);
    }

    #[test]
    fn u_zero_returns_zero_and_does_not_consume() {
        let buf = [0xFF];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.u(0).unwrap(), 0);
        assert_eq!(br.bit_pos(), 0);
    }

    #[test]
    fn u_thirty_two_full_word() {
        let buf = [0x12, 0x34, 0x56, 0x78];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.u(32).unwrap(), 0x1234_5678);
    }

    #[test]
    fn ue_table_b9_2() {
        // Per H.265 §9.2 Table 9-2 first few codewords:
        //   codeNum=0 → '1'         (1 bit)
        //   codeNum=1 → '010'       (3 bits)
        //   codeNum=2 → '011'       (3 bits)
        //   codeNum=3 → '00100'     (5 bits)
        //   codeNum=4 → '00101'     (5 bits)
        //   codeNum=5 → '00110'     (5 bits)
        //   codeNum=6 → '00111'     (5 bits)
        // Pack codewords 0..=6 into a byte stream MSB-first:
        // Bits: 1 010 011 00100 00101 00110 00111 = 27 bits, padded
        // with 5 zero bits to land on a 32-bit byte boundary.
        // Group as bytes (MSB first):
        //   bit 0..7  : 1 0 1 0 0 1 1 0  = 0xA6
        //   bit 8..15 : 0 1 0 0 0 0 1 0  = 0x42
        //   bit 16..23: 1 0 0 1 1 0 0 0  = 0x98
        //   bit 24..31: 1 1 1 0 0 0 0 0  = 0xE0  (last 5 bits are pad)
        let buf = [0xA6, 0x42, 0x98, 0xE0];
        let mut br = BitReader::new(&buf);
        let got: Vec<u32> = (0..7).map(|_| br.ue().unwrap()).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ue_zero_is_single_one_bit() {
        let buf = [0b1000_0000];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.ue().unwrap(), 0);
        assert_eq!(br.bit_pos(), 1);
    }

    #[test]
    fn end_of_buffer_errors() {
        let buf = [0xFF];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.u(8).unwrap(), 0xFF);
        assert_eq!(br.u(1), Err(BitReaderError::EndOfBuffer));
    }

    #[test]
    fn too_many_bits_errors() {
        let buf = [0xFF; 8];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.u(33), Err(BitReaderError::TooManyBits));
    }

    #[test]
    fn skip_advances_cursor() {
        let buf = [0xFF, 0xAA];
        let mut br = BitReader::new(&buf);
        br.skip(8).unwrap();
        assert_eq!(br.u(8).unwrap(), 0xAA);
    }

    #[test]
    fn se_table_9_3() {
        // Per H.265 §9.2.2 Table 9-3, codeNum → se(v) value:
        //   0 → 0, 1 → +1, 2 → -1, 3 → +2, 4 → -2, 5 → +3, 6 → -3.
        // codeNum is itself ue(v); reuse the codewords 0..=6 packing
        // from `ue_table_b9_2` (bits 1 010 011 00100 00101 00110 00111).
        let buf = [0xA6, 0x42, 0x98, 0xE0];
        let mut br = BitReader::new(&buf);
        let got: Vec<i32> = (0..7).map(|_| br.se().unwrap()).collect();
        assert_eq!(got, vec![0, 1, -1, 2, -2, 3, -3]);
    }

    #[test]
    fn se_zero_is_single_one_bit() {
        // codeNum 0 ('1') maps to value 0 and consumes one bit.
        let buf = [0b1000_0000];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.se().unwrap(), 0);
        assert_eq!(br.bit_pos(), 1);
    }

    #[test]
    fn ue_overflow_run_too_long() {
        // 33 leading zero bits with no terminating 1 in the buffer.
        // The reader should surface ExpGolombOverflow before running
        // off the end (we never reach the suffix read).
        let buf = [0u8; 8]; // 64 zero bits
        let mut br = BitReader::new(&buf);
        assert_eq!(br.ue(), Err(BitReaderError::ExpGolombOverflow));
    }
}
