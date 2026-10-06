//! The two VP9 bit writers: the plain MSB-first writer for the uncompressed
//! frame header, and the boolean (arithmetic) encoder for the compressed header
//! and tile data (VP9 specification sections 9.1 and 9.2).

use std::sync::OnceLock;

/// Writes the uncompressed header's fixed-width fields, most significant bit
/// first.
#[derive(Default)]
pub(super) struct BitWriter {
    bytes: Vec<u8>,
    bit: u32,
}

impl BitWriter {
    pub(super) fn bit(&mut self, value: bool) {
        if self.bit % 8 == 0 {
            self.bytes.push(0);
        }
        if value {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 0x80 >> (self.bit % 8);
        }
        self.bit += 1;
    }

    pub(super) fn literal(&mut self, value: u32, bits: u32) {
        for shift in (0..bits).rev() {
            self.bit((value >> shift) & 1 != 0);
        }
    }

    /// Returns the header, zero-padded to a whole byte.
    pub(super) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// The VP9 boolean encoder, bit-exact with libvpx's `vpx_writer`.
pub(super) struct BoolEncoder {
    bytes: Vec<u8>,
    low: u32,
    range: u32,
    count: i32,
}

impl BoolEncoder {
    /// Starts a boolean-coded partition, including the zero marker bit every
    /// VP9 boolean decoder reads first.
    pub(super) fn new() -> Self {
        let mut encoder = Self {
            bytes: Vec::new(),
            low: 0,
            range: 255,
            count: -24,
        };
        encoder.write(false, 128);
        encoder
    }

    pub(super) fn bit(&mut self, bit: bool) {
        self.write(bit, 128);
    }

    pub(super) fn literal(&mut self, value: u32, bits: u32) {
        for shift in (0..bits).rev() {
            self.bit((value >> shift) & 1 != 0);
        }
    }

    /// Flushes the partition the way libvpx's `vpx_stop_encode` does.
    pub(super) fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.bit(false);
        }
        // Keep the final byte from looking like a superframe index marker.
        if self.bytes.last().is_some_and(|byte| byte & 0xe0 == 0xc0) {
            self.bytes.push(0);
        }
        self.bytes
    }
}

impl BoolSink for BoolEncoder {
    fn write(&mut self, bit: bool, probability: u8) {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        let mut range = split;
        let mut low = self.low;
        if bit {
            low += split;
            range = self.range - split;
        }
        let mut shift = range.leading_zeros() as i32 - 24;
        range <<= shift;
        let mut count = self.count + shift;
        if count >= 0 {
            let offset = shift - count;
            if (low << (offset - 1)) & 0x8000_0000 != 0 {
                // Propagate the carry into the bytes already written.
                let mut index = self.bytes.len();
                while index > 0 && self.bytes[index - 1] == 0xff {
                    self.bytes[index - 1] = 0;
                    index -= 1;
                }
                if index > 0 {
                    self.bytes[index - 1] += 1;
                }
            }
            self.bytes.push((low >> (24 - offset)) as u8);
            low <<= offset;
            shift = count;
            low &= 0x00ff_ffff;
            count -= 8;
        }
        low <<= shift;
        self.count = count;
        self.low = low;
        self.range = range;
    }
}

/// Where boolean-coded symbols go: the [`BoolEncoder`] itself, or a
/// [`BitCost`] that totals what they would cost to write.
pub(super) trait BoolSink {
    fn write(&mut self, bit: bool, probability: u8);

    /// Writes `value` with a libvpx tree: positive entries index the next
    /// node pair, and zero or negative entries are leaves holding `-value`.
    fn tree(&mut self, tree: &[i8], probabilities: &[u8], value: u8) {
        let mut path = [false; 16];
        let mut length = 0;
        let found = tree_path(tree, 0, i16::from(value), &mut path, &mut length);
        debug_assert!(found, "value {value} is not a leaf of the tree");
        let mut node = 0_usize;
        for &bit in &path[..length] {
            self.write(bit, probabilities[node >> 1]);
            let next = tree[node + usize::from(bit)];
            if next <= 0 {
                break;
            }
            node = next as usize;
        }
    }
}

/// Totals the ideal cost, in bits, of the symbols written to it.
#[derive(Default)]
pub(super) struct BitCost(pub(super) f64);

impl BoolSink for BitCost {
    fn write(&mut self, bit: bool, probability: u8) {
        self.0 += bit_cost(bit, probability);
    }
}

/// The cost in bits of coding `bit` when a zero has `probability` / 256.
pub(super) fn bit_cost(bit: bool, probability: u8) -> f64 {
    static COSTS: OnceLock<[f64; 256]> = OnceLock::new();
    let costs = COSTS.get_or_init(|| {
        core::array::from_fn(|p| -(p.max(1) as f64 / 256.0).log2())
    });
    let p = usize::from(probability);
    costs[if bit { 256 - p } else { p }]
}

fn tree_path(
    tree: &[i8],
    node: usize,
    value: i16,
    path: &mut [bool; 16],
    length: &mut usize,
) -> bool {
    for bit in [false, true] {
        let entry = tree[node + usize::from(bit)];
        path[*length] = bit;
        *length += 1;
        if entry <= 0 {
            if -i16::from(entry) == value {
                return true;
            }
        } else if tree_path(tree, entry as usize, value, path, length) {
            return true;
        }
        *length -= 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straightforward VP9 boolean decoder (specification section 9.2),
    /// independent of the encoder's carry handling.
    struct BoolDecoder<'a> {
        data: &'a [u8],
        position: usize,
        value: u64,
        range: u32,
        bits: i32,
    }

    impl<'a> BoolDecoder<'a> {
        fn new(data: &'a [u8]) -> Self {
            let mut decoder = Self {
                data,
                position: 0,
                value: 0,
                range: 255,
                bits: -8,
            };
            decoder.fill();
            assert!(!decoder.read(128), "marker bit must be zero");
            decoder
        }

        fn fill(&mut self) {
            while self.bits < 0 {
                let byte = self.data.get(self.position).copied().unwrap_or(0);
                self.position += 1;
                self.value = (self.value << 8) | u64::from(byte);
                self.bits += 8;
            }
        }

        fn read(&mut self, probability: u8) -> bool {
            let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
            let big_split = u64::from(split) << self.bits;
            let bit = if self.value >= big_split {
                self.range -= split;
                self.value -= big_split;
                true
            } else {
                self.range = split;
                false
            };
            while self.range < 128 {
                self.range <<= 1;
                self.bits -= 1;
                self.fill();
            }
            bit
        }
    }

    #[test]
    fn boolean_coder_round_trips_skewed_and_even_probabilities() {
        let mut state = 0x1234_5678_u32;
        let mut symbols = Vec::new();
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let probability = ((state >> 8) % 255 + 1) as u8;
            let bit = (state >> 20) % 256 >= u32::from(probability);
            symbols.push((bit, probability));
        }
        let mut encoder = BoolEncoder::new();
        for &(bit, probability) in &symbols {
            encoder.write(bit, probability);
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for &(bit, probability) in &symbols {
            assert_eq!(decoder.read(probability), bit);
        }
    }

    #[test]
    fn tree_writes_each_leaf_along_its_path() {
        // The VP9 inter mode tree: ZEROMV(2), NEARESTMV(0), NEARMV(1), NEWMV(3).
        let tree = [-2, 2, 0, 4, -1, -3];
        let probabilities = [40, 90, 200];
        let mut encoder = BoolEncoder::new();
        for value in [2, 0, 1, 3, 3, 0] {
            encoder.tree(&tree, &probabilities, value);
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for expected in [2, 0, 1, 3, 3, 0] {
            let mut node = 0_usize;
            let value = loop {
                let entry = tree[node + usize::from(decoder.read(probabilities[node >> 1]))];
                if entry <= 0 {
                    break -entry;
                }
                node = entry as usize;
            };
            assert_eq!(value, expected);
        }
    }

    #[test]
    fn bit_writer_packs_msb_first() {
        let mut writer = BitWriter::default();
        writer.literal(2, 2);
        writer.bit(true);
        writer.literal(0x1ff, 9);
        assert_eq!(writer.finish(), vec![0b1011_1111, 0b1111_0000]);
    }
}
