//! LSb-first bit packer: the write side of libogg's `oggpack_buffer`
//! (bitwise.c `oggpack_write`, `oggpack_bytes`, `oggpack_reset`).

/// Port of the writing half of libogg's `oggpack_buffer`.
#[derive(Debug, Default, Clone)]
pub(crate) struct OggPackBuffer {
    buf: Vec<u8>,
    /// Bit accumulator; `acc_bits` (< 8 after every write) bits are pending.
    acc: u64,
    acc_bits: u32,
}

impl OggPackBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// `oggpack_reset`
    pub(crate) fn reset(&mut self) {
        self.buf.clear();
        self.acc = 0;
        self.acc_bits = 0;
    }

    /// `oggpack_write`: append the low `bits` (0..=32) bits of `value`.
    #[inline]
    pub(crate) fn write(&mut self, value: u32, bits: u32) {
        debug_assert!(bits <= 32);
        if bits == 0 {
            return;
        }
        let v = if bits == 32 {
            u64::from(value)
        } else {
            u64::from(value & ((1u32 << bits) - 1))
        };
        self.acc |= v << self.acc_bits;
        self.acc_bits += bits;
        while self.acc_bits >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.acc_bits -= 8;
        }
    }

    /// `oggpack_bytes`: bytes used, counting a partial last byte.
    pub(crate) fn bytes(&self) -> usize {
        self.buf.len() + usize::from(self.acc_bits > 0)
    }

    /// `oggpack_get_buffer` + `oggpack_bytes`: the packet contents.
    pub(crate) fn to_vec(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.bytes());
        v.extend_from_slice(&self.buf);
        if self.acc_bits > 0 {
            v.push(self.acc as u8);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_first_packing() {
        let mut b = OggPackBuffer::new();
        b.write(1, 1);
        b.write(0, 2);
        b.write(5, 3);
        b.write(0xabcd, 16);
        b.write(0xffff_ffff, 32);
        // bits: 1 | 00 | 101 -> 0b0010_1001 = 0x29 (low 6 bits), then 0xabcd ...
        let v = b.to_vec();
        assert_eq!(v.len(), 7);
        assert_eq!(v[0], 0x29 | ((0xcd & 0x3) << 6) as u8);
        assert_eq!(b.bytes(), 7);
        b.reset();
        assert_eq!(b.bytes(), 0);
    }
}
