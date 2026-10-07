//! The VP8 boolean entropy encoder (RFC 6386 section 7.3), the inverse of
//! [`super::bool_decoder::BoolDecoder`].

pub(crate) struct BoolEncoder {
    output: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    pub(crate) fn new() -> Self {
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

    /// Writes `bit`, which is false with probability `probability / 256`.
    pub(crate) fn write(&mut self, probability: u8, bit: bool) {
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

    pub(crate) fn write_flag(&mut self, bit: bool) {
        self.write(128, bit);
    }

    /// An unsigned `bits`-wide literal, most significant bit first.
    pub(crate) fn write_literal(&mut self, value: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write_flag((value >> bit) & 1 == 1);
        }
    }

    /// Writes `value`'s path through `tree`, the inverse of
    /// [`super::bool_decoder::BoolDecoder::read_tree`].
    pub(crate) fn write_tree(&mut self, tree: &[i8], probabilities: &[u8], value: u8) {
        let mut path = [(0usize, false); 16];
        let length = tree_branches(tree, value, &mut path);
        for &(index, bit) in &path[..length] {
            self.write(probabilities[index >> 1], bit);
        }
    }

    /// Pads the output so the decoder can read every bit written.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_flag(false);
        }
        self.output
    }
}

/// Stores the branches, as tree index and bit, from the root of `tree` to
/// leaf `value` in `path`, returning their number.
pub(crate) fn tree_branches(tree: &[i8], value: u8, path: &mut [(usize, bool); 16]) -> usize {
    tree_path(tree, 0, value, path, 0).expect("the value is a leaf of the tree")
}

/// The cost, in 1/256 bits, of writing `value` with `tree`.
pub(crate) fn tree_cost(tree: &[i8], probabilities: &[u8], value: u8) -> u32 {
    let mut path = [(0usize, false); 16];
    let length = tree_branches(tree, value, &mut path);
    path[..length]
        .iter()
        .map(|&(index, bit)| bit_cost(probabilities[index >> 1], bit))
        .sum()
}

/// Finds the branches from node `index` to leaf `value`, returning the path
/// length.
fn tree_path(
    tree: &[i8],
    index: usize,
    value: u8,
    path: &mut [(usize, bool); 16],
    depth: usize,
) -> Option<usize> {
    for bit in [false, true] {
        path[depth] = (index, bit);
        let next = tree[index + usize::from(bit)];
        if next <= 0 {
            if (-next) as u8 == value {
                return Some(depth + 1);
            }
        } else if let Some(length) = tree_path(tree, next as usize, value, path, depth + 1) {
            return Some(length);
        }
    }
    None
}

/// The approximate cost, in 1/256 bits, of coding `bit` with `probability`.
pub(crate) fn bit_cost(probability: u8, bit: bool) -> u32 {
    let probability = if bit {
        256 - u32::from(probability)
    } else {
        u32::from(probability)
    };
    COST_TABLE[probability as usize]
}

/// `-log2(p / 256) * 256` for `p` in `0..=256`, with `p = 0` saturated.
static COST_TABLE: [u32; 257] = {
    let mut table = [0u32; 257];
    // log2 by integer bisection, to 1/256 bit, without floating point in a
    // const context: cost(p) = 256 * (8 - log2(p)).
    let mut p = 1;
    while p <= 256 {
        // log2(p) * 256 by repeated squaring of p normalized to [1, 2).
        let mut integer = 0u32;
        let mut value = p as u64;
        while value >= 2 {
            value >>= 1;
            integer += 1;
        }
        // Fraction: x = p / 2^integer in [1, 2), as 16.16 fixed point.
        let mut x = ((p as u64) << 16) >> integer;
        let mut fraction = 0u32;
        let mut bit = 128u32;
        while bit > 0 {
            x = (x * x) >> 16;
            if x >= 2 << 16 {
                x >>= 1;
                fraction |= bit;
            }
            bit >>= 1;
        }
        table[p] = 8 * 256 - (integer * 256 + fraction);
        p += 1;
    }
    table[0] = 8 * 256 * 2;
    table
};

#[cfg(test)]
mod tests {
    use super::super::bool_decoder::BoolDecoder;
    use super::super::tables::{B_MODE_TREE, DEFAULT_B_MODE_PROBS};
    use super::*;

    #[test]
    fn trees_and_literals_round_trip() {
        let mut encoder = BoolEncoder::new();
        for value in 0..10 {
            encoder.write_tree(&B_MODE_TREE, &DEFAULT_B_MODE_PROBS, value);
            encoder.write_literal(u32::from(value) * 13, 7);
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for value in 0..10 {
            assert_eq!(
                decoder.read_tree(&B_MODE_TREE, &DEFAULT_B_MODE_PROBS),
                value
            );
            assert_eq!(decoder.read_literal(7), u32::from(value) * 13);
        }
    }

    #[test]
    fn costs_are_negative_log_probabilities() {
        assert_eq!(bit_cost(128, false), 256);
        assert_eq!(bit_cost(128, true), 256);
        assert_eq!(bit_cost(64, false), 512);
        assert!(bit_cost(255, false) < 4);
        assert!(bit_cost(255, true) >= 8 * 256 - 1);
    }
}
