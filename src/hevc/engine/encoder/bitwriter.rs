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
    ///
    /// The field is written a chunk at a time rather than a bit at a
    /// time: first whatever fills the trailing partial byte, then whole
    /// bytes, then the remainder as a new partial byte. Widening the
    /// path this way is what makes a 32-bit field one masked OR plus
    /// four pushes instead of 32 read-modify-writes of the same byte.
    pub fn put_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        // Widen to u64 so `1 << n` is representable for n == 32 and the
        // masks below never overflow.
        let value = u64::from(value) & (u64::MAX >> (64 - u32::from(n)));
        let mut left = u32::from(n);

        if self.partial_bits != 0 {
            let free = 8 - u32::from(self.partial_bits);
            let take = free.min(left);
            left -= take;
            let chunk = ((value >> left) & (u64::MAX >> (64 - take))) as u8;
            let last = self.buf.last_mut().expect("partial byte exists");
            *last |= chunk << (free - take);
            self.partial_bits = (self.partial_bits + take as u8) & 7;
            if left == 0 {
                return;
            }
        }

        // Byte-aligned from here, so whole bytes append directly.
        while left >= 8 {
            left -= 8;
            self.buf.push(((value >> left) & 0xFF) as u8);
        }
        if left != 0 {
            let chunk = (value & (u64::MAX >> (64 - left))) as u8;
            self.buf.push(chunk << (8 - left));
            self.partial_bits = left as u8;
        }
    }

    /// Append whole bytes, most significant bit of each byte first.
    ///
    /// When the writer is already byte-aligned — which is how §7.3.8.7
    /// `pcm_sample[]` data always arrives, since `pcm_alignment_zero_
    /// bit` precedes it — this bypasses the bit accumulator entirely and
    /// becomes a bulk copy into the output buffer. Unaligned callers
    /// still get the correct MSB-first packing, one byte at a time.
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        if self.partial_bits == 0 {
            self.buf.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.put_bits(u32::from(b), 8);
            }
        }
    }

    /// Largest `codeNum` a `ue(v)` codeword may carry: `2^32 - 2`, the
    /// cap §9.2 states and [`BitReader::ue`] enforces
    /// ([`BitReaderError::ExpGolombOverflow`] above it). `2^32 - 1`
    /// would need a 33-bit codeword with a 32-zero prefix, which is
    /// outside the range the spec allows and which no conforming
    /// decoder — this crate's reader included — accepts.
    ///
    /// [`BitReader::ue`]: crate::hevc::engine::bitreader::BitReader::ue
    /// [`BitReaderError::ExpGolombOverflow`]:
    ///     crate::hevc::engine::bitreader::BitReaderError::ExpGolombOverflow
    pub const MAX_UE: u32 = u32::MAX - 1;

    /// Most negative value `se(v)` can carry. Table 9-3 maps it to
    /// `codeNum == 2 * 2^31 - 2 ==` [`Self::MAX_UE`]; `i32::MIN` itself
    /// would map to `2^32`, one past the cap.
    pub const MIN_SE: i32 = i32::MIN + 1;

    /// Most positive value `se(v)` can carry, mapping to
    /// `codeNum == 2^32 - 3`, comfortably inside [`Self::MAX_UE`].
    pub const MAX_SE: i32 = i32::MAX;

    /// §9.2 — 0-th order Exp-Golomb, unsigned (`ue(v)`).
    ///
    /// # Panics
    ///
    /// If `value` exceeds [`Self::MAX_UE`]. That is a caller contract
    /// violation rather than bad input data: every `ue(v)` field the
    /// HEVC syntax defines is bounded far below the cap, so reaching it
    /// means the caller computed a field value wrongly. Emitting the
    /// out-of-contract codeword instead would put a codeword in the
    /// bitstream that no conforming decoder can read back.
    pub fn ue(&mut self, value: u32) {
        assert!(
            value <= Self::MAX_UE,
            "ue(v) codeNum {value} exceeds the §9.2 cap of {}",
            Self::MAX_UE
        );
        // codeNum = value; the codeword is `leadingZeroBits` zeros, a
        // one, then `leadingZeroBits` LSBs of (codeNum + 1). The cap
        // keeps `codeNum + 1` inside `u32` and the codeword at 63 bits
        // or fewer, so the payload always fits `put_bits`' 32-bit field.
        let code_num = value + 1;
        let bits = 32 - code_num.leading_zeros() as u8; // 1..=32
        self.put_bits(0, bits - 1);
        self.put_bits(code_num, bits);
    }

    /// §9.2.2 — 0-th order Exp-Golomb, signed (`se(v)`):
    /// `codeNum = value > 0 ? 2*value − 1 : −2*value`.
    ///
    /// # Panics
    ///
    /// If `value` is below [`Self::MIN_SE`] — that is, `i32::MIN`, the
    /// one `i32` whose Table 9-3 `codeNum` is past [`Self::MAX_UE`].
    pub fn se(&mut self, value: i32) {
        assert!(
            value >= Self::MIN_SE,
            "se(v) value {value} maps to a codeNum past the §9.2 cap"
        );
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
        // Partial bytes are pushed zeroed and only ever ORed into, so the
        // padding bits are already zero; only the counter has to move.
        self.partial_bits = 0;
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
    use crate::hevc::engine::bitreader::{BitReader, BitReaderError};

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

    /// The widened `put_bits` must agree bit for bit with the
    /// bit-at-a-time reference it replaced, at every field width and
    /// every starting bit offset (the partial-byte fill, whole-byte, and
    /// tail chunks all move relative to each other as the offset walks).
    #[test]
    fn put_bits_matches_bit_at_a_time_at_every_width_and_offset() {
        fn reference(w: &mut BitWriter, value: u32, n: u8) {
            for i in (0..n).rev() {
                w.put_bit(((value >> i) & 1) as u8);
            }
        }

        for offset in 0..8u8 {
            for n in 0..=32u8 {
                for &value in &[0u32, 1, 0xA5A5_5A5A, 0xFFFF_FFFF, 0x8000_0001] {
                    let mut widened = BitWriter::new();
                    let mut reference_writer = BitWriter::new();
                    widened.put_bits(0b0101_0101, offset);
                    reference(&mut reference_writer, 0b0101_0101, offset);
                    widened.put_bits(value, n);
                    reference(&mut reference_writer, value, n);
                    assert_eq!(
                        widened.bit_len(),
                        reference_writer.bit_len(),
                        "bit_len for value {value:#x} n {n} offset {offset}"
                    );
                    assert_eq!(
                        widened.byte_aligned(),
                        reference_writer.byte_aligned(),
                        "alignment for value {value:#x} n {n} offset {offset}"
                    );
                    assert_eq!(
                        widened.finish(),
                        reference_writer.finish(),
                        "bytes for value {value:#x} n {n} offset {offset}"
                    );
                }
            }
        }
    }

    /// The widest codewords the §9.2 cap permits must still be the
    /// bit-at-a-time reference codeword, and must read back through
    /// [`BitReader::ue`] — the writer's bound and the reader's are the
    /// same bound.
    #[test]
    fn ue_emits_the_reference_codeword_up_to_the_code_num_cap() {
        for v in [
            BitWriter::MAX_UE,
            BitWriter::MAX_UE - 1,
            1 << 31,
            (1 << 31) - 1,
            (1 << 16) - 2,
        ] {
            let code_num = u64::from(v) + 1;
            let bits = 64 - code_num.leading_zeros() as u8;
            let mut expected = BitWriter::new();
            for _ in 0..bits - 1 {
                expected.put_bit(0);
            }
            for i in (0..bits).rev() {
                expected.put_bit(((code_num >> i) & 1) as u8);
            }
            expected.align_zero();

            let mut w = BitWriter::new();
            w.ue(v);
            let bit_len = w.bit_len();
            w.align_zero();
            assert_eq!(bit_len, usize::from(2 * bits - 1), "ue({v}) length");
            let bytes = w.finish();
            assert_eq!(bytes, expected.finish(), "ue({v}) bits");

            let mut r = BitReader::new(&bytes);
            assert_eq!(r.ue().unwrap(), v, "ue({v}) roundtrip");
        }
    }

    /// The boundary in both directions: `2^32 - 2` is the largest
    /// `codeNum` the writer emits and the reader accepts; `2^32 - 1` is
    /// the first one neither may produce.
    #[test]
    fn ue_cap_matches_the_reader() {
        let mut w = BitWriter::new();
        w.ue(BitWriter::MAX_UE);
        w.align_zero();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue().unwrap(), BitWriter::MAX_UE);

        // The codeword one past the cap — 32 zeros, a one, then 32 more
        // bits — is what the writer used to emit for `u32::MAX`. The
        // reader rejects it, which is why the writer must not write it.
        let mut hand_rolled = BitWriter::new();
        for _ in 0..32 {
            hand_rolled.put_bit(0);
        }
        hand_rolled.put_bit(1);
        hand_rolled.put_bits(0, 32);
        hand_rolled.align_zero();
        let over_cap = hand_rolled.finish();
        let mut r = BitReader::new(&over_cap);
        assert_eq!(r.ue(), Err(BitReaderError::ExpGolombOverflow));
    }

    #[test]
    #[should_panic(expected = "exceeds the §9.2 cap")]
    fn ue_rejects_the_first_code_num_past_the_cap() {
        BitWriter::new().ue(u32::MAX);
    }

    /// `se`'s own extremes map onto the same `ue` cap: `i32::MIN + 1`
    /// lands exactly on it and must round-trip, `i32::MIN` is one past.
    #[test]
    fn se_roundtrips_its_extreme_values() {
        for v in [BitWriter::MIN_SE, BitWriter::MAX_SE, -1, 1] {
            let mut w = BitWriter::new();
            w.se(v);
            w.align_zero();
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.se().unwrap(), v, "se({v})");
        }
    }

    #[test]
    #[should_panic(expected = "past the §9.2 cap")]
    fn se_rejects_i32_min() {
        BitWriter::new().se(i32::MIN);
    }

    /// `put_bytes` is a bulk copy when aligned and MSB-first packing when
    /// not; both must equal the per-byte `put_bits` path it replaces.
    #[test]
    fn put_bytes_matches_per_byte_writes_aligned_and_unaligned() {
        let data: Vec<u8> = (0..64u32)
            .map(|i| (i.wrapping_mul(37) ^ 0x5A) as u8)
            .collect();
        for offset in 0..8u8 {
            let mut bulk = BitWriter::new();
            let mut per_byte = BitWriter::new();
            bulk.put_bits(0b1101_1011, offset);
            per_byte.put_bits(0b1101_1011, offset);
            bulk.put_bytes(&data);
            for &b in &data {
                per_byte.put_bits(u32::from(b), 8);
            }
            assert_eq!(
                bulk.bit_len(),
                per_byte.bit_len(),
                "bit_len at offset {offset}"
            );
            assert_eq!(bulk.finish(), per_byte.finish(), "bytes at offset {offset}");
        }
    }

    #[test]
    fn put_bytes_reads_back_as_the_original_bytes() {
        let data: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();
        let mut w = BitWriter::new();
        w.put_bits(0b101, 3);
        w.align_zero();
        w.put_bytes(&data);
        assert!(w.byte_aligned());
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u(8).unwrap(), 0b1010_0000);
        for &b in &data {
            assert_eq!(r.u(8).unwrap(), u32::from(b));
        }
    }

    /// `align_zero` stopped writing padding bits and now only resets the
    /// counter; the padding must still read back as zero.
    #[test]
    fn align_zero_pads_with_zero_bits() {
        for used in 1..8u8 {
            let mut w = BitWriter::new();
            w.put_bits(u32::MAX, used);
            w.align_zero();
            assert!(w.byte_aligned());
            assert_eq!(w.bit_len(), 8);
            let expected = (0xFFu32 << (8 - used)) as u8;
            assert_eq!(w.finish(), vec![expected], "padding after {used} bits");
        }
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
