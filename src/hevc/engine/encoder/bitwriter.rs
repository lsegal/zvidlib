//! MSB-first RBSP bit writer — the write-side dual of
//! [`crate::bitreader::BitReader`].
//!
//! Produces raw RBSP bytes: fixed-width `u(n)` fields, the §9.2
//! Exp-Golomb `ue(v)` / `se(v)` encodings, the §7.3.2 `rbsp_trailing_
//! bits()` terminator, and byte alignment. Emulation prevention
//! (§7.4.1.1) is NOT applied here — it belongs to the NAL encapsulation
//! ([`super::nal::escape_rbsp`]), which turns an RBSP into the on-wire
//! escaped payload.

/// MSB-first bit accumulator over a growable byte buffer.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bits already used in the trailing partial byte (0..=7). When 0,
    /// the writer is byte-aligned and `buf` holds only complete bytes.
    partial_bits: u8,
}

impl BitWriter {
    /// A fresh, empty, byte-aligned writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        if self.partial_bits == 0 {
            self.buf.len() * 8
        } else {
            (self.buf.len() - 1) * 8 + self.partial_bits as usize
        }
    }

    /// `byte_aligned()` per §7.2.
    #[must_use]
    pub fn byte_aligned(&self) -> bool {
        self.partial_bits == 0
    }

    /// Append a single bit (the low bit of `bit`).
    pub fn put_bit(&mut self, bit: u8) {
        if self.partial_bits == 0 {
            self.buf.push(0);
        }
        let last = self.buf.last_mut().expect("partial byte exists");
        *last |= (bit & 1) << (7 - self.partial_bits);
        self.partial_bits = (self.partial_bits + 1) & 7;
    }

    /// Append the `n` low bits of `value`, most significant first
    /// (`u(n)` / `f(n)`). `n == 0` writes nothing; `n <= 32`.
    pub fn put_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.put_bit(((value >> i) & 1) as u8);
        }
    }

    /// §9.2 — 0-th order Exp-Golomb, unsigned (`ue(v)`).
    pub fn ue(&mut self, value: u32) {
        // codeNum = value; the codeword is `leadingZeroBits` zeros, a
        // one, then `leadingZeroBits` LSBs of (codeNum + 1).
        let code_num = value as u64 + 1;
        let bits = 64 - code_num.leading_zeros() as u8; // >= 1
        self.put_bits(0, bits - 1);
        for i in (0..bits).rev() {
            self.put_bit(((code_num >> i) & 1) as u8);
        }
    }

    /// §9.2.2 — 0-th order Exp-Golomb, signed (`se(v)`):
    /// `codeNum = value > 0 ? 2*value − 1 : −2*value`.
    pub fn se(&mut self, value: i32) {
        let code_num = if value > 0 {
            2 * value as u32 - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(code_num);
    }

    /// §7.3.2 `rbsp_trailing_bits()`: `rbsp_stop_one_bit` then zero
    /// bits to the byte boundary.
    pub fn rbsp_trailing_bits(&mut self) {
        self.put_bit(1);
        self.align_zero();
    }

    /// Append zero bits until byte-aligned (`pcm_alignment_zero_bit` /
    /// `rbsp_alignment_zero_bit` runs).
    pub fn align_zero(&mut self) {
        while self.partial_bits != 0 {
            self.put_bit(0);
        }
    }

    /// Consume the writer and return the accumulated bytes. Any
    /// trailing partial byte is zero-padded (callers producing RBSPs
    /// terminate with [`Self::rbsp_trailing_bits`] first, so a
    /// conforming RBSP is already aligned).
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the bytes written so far (trailing partial byte included,
    /// zero-padded).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;

    #[test]
    fn bits_pack_msb_first() {
        let mut w = BitWriter::new();
        w.put_bit(1);
        w.put_bits(0b0110, 4);
        w.put_bits(0b101, 3);
        assert_eq!(w.finish(), vec![0b1011_0101]);
    }

    #[test]
    fn ue_matches_reader_for_first_values() {
        for v in 0..200u32 {
            let mut w = BitWriter::new();
            w.ue(v);
            w.align_zero();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.ue().unwrap(), v, "ue({v})");
        }
    }

    #[test]
    fn se_matches_reader() {
        for v in -100..=100i32 {
            let mut w = BitWriter::new();
            w.se(v);
            w.align_zero();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.se().unwrap(), v, "se({v})");
        }
    }

    #[test]
    fn u_fields_roundtrip() {
        let mut w = BitWriter::new();
        w.put_bits(0x2A, 6);
        w.put_bits(0x1FFFF, 17);
        w.put_bits(1, 1);
        w.align_zero();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u(6).unwrap(), 0x2A);
        assert_eq!(r.u(17).unwrap(), 0x1FFFF);
        assert_eq!(r.u1().unwrap(), 1);
    }

    #[test]
    fn trailing_bits_terminate_and_align() {
        let mut w = BitWriter::new();
        w.put_bits(0b101, 3);
        w.rbsp_trailing_bits();
        assert!(w.byte_aligned());
        assert_eq!(w.finish(), vec![0b1011_0000]);
    }

    #[test]
    fn bit_len_tracks_partial_bytes() {
        let mut w = BitWriter::new();
        assert_eq!(w.bit_len(), 0);
        w.put_bit(0);
        assert_eq!(w.bit_len(), 1);
        w.put_bits(0, 7);
        assert_eq!(w.bit_len(), 8);
        assert!(w.byte_aligned());
        w.put_bits(0, 3);
        assert_eq!(w.bit_len(), 11);
    }
}
