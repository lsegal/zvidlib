//! Encode-side codebook handling: sharedbook.c (`_make_words`,
//! `_book_maptype1_quantvals`, `_float32_unpack`, `vorbis_book_init_encode`)
//! and codebook.c (`vorbis_staticbook_pack`, `vorbis_book_encode`).

use super::bitpack::OggPackBuffer;
use super::os::{ilog, rint};
use super::tables::types::StaticCodebook;

const VQ_FMAN: i32 = 21;
const VQ_FEXP_BIAS: i32 = 768;

/// Port of sharedbook.c `_float32_unpack`.
pub(crate) fn float32_unpack(val: i32) -> f32 {
    let val = val as u32;
    let mut mant = f64::from(val & 0x1f_ffff);
    let sign = val & 0x8000_0000;
    let mut exp = ((val & 0x7fe0_0000) >> VQ_FMAN) as i32;
    if sign != 0 {
        mant = -mant;
    }
    exp = exp - (VQ_FMAN - 1) - VQ_FEXP_BIAS;
    exp = exp.clamp(-63, 63);
    // ldexp(mant, exp): exact scaling by a power of two
    (mant * 2f64.powi(exp)) as f32
}

/// Port of sharedbook.c `_make_words` (with `sparsecount == 0`): canonical
/// Huffman codewords for the length list, bit-reversed for the LSb-first
/// packer. Returns `None` for an over- or under-populated tree.
pub(crate) fn make_words(l: &[u8]) -> Option<Vec<u32>> {
    let n = l.len();
    let mut marker = [0u32; 33];
    let mut r = vec![0u32; n];
    let mut count = 0usize;
    for &len in l {
        let length = usize::from(len);
        if length > 0 {
            let mut entry = marker[length];
            if length < 32 && (entry >> length) != 0 {
                return None;
            }
            r[count] = entry;
            count += 1;
            let mut j = length;
            while j > 0 {
                if marker[j] & 1 != 0 {
                    if j == 1 {
                        marker[1] += 1;
                    } else {
                        marker[j] = marker[j - 1] << 1;
                    }
                    break;
                }
                marker[j] += 1;
                j -= 1;
            }
            for j in length + 1..33 {
                if (marker[j] >> 1) == entry {
                    entry = marker[j];
                    marker[j] = marker[j - 1] << 1;
                } else {
                    break;
                }
            }
        } else {
            count += 1;
        }
    }
    if !(count == 1 && marker[2] == 2) {
        for (i, &m) in marker.iter().enumerate().skip(1) {
            if m & (0xffff_ffffu32 >> (32 - i)) != 0 {
                return None;
            }
        }
    }
    for (i, &len) in l.iter().enumerate() {
        let mut temp = 0u32;
        for j in 0..u32::from(len) {
            temp <<= 1;
            temp |= (r[i] >> j) & 1;
        }
        r[i] = temp;
    }
    Some(r)
}

/// Port of sharedbook.c `_book_maptype1_quantvals`: the largest `vals` with
/// `vals^dim <= entries`.
pub(crate) fn book_maptype1_quantvals(b: &StaticCodebook) -> i32 {
    if b.entries < 1 {
        return 0;
    }
    let entries = i64::from(b.entries);
    let dim = b.dim;
    // initial guess as in C; the integer loop below makes the result exact
    let mut vals = (f64::from(b.entries as f32))
        .powf(f64::from(1.0f32 / dim as f32))
        .floor() as i64;
    if vals < 1 {
        vals = 1;
    }
    loop {
        let mut acc: i64 = 1;
        let mut acc1: i64 = 1;
        let mut i = 0;
        while i < dim {
            if entries / vals < acc {
                break;
            }
            acc *= vals;
            if i64::MAX / (vals + 1) < acc1 {
                acc1 = i64::MAX;
            } else {
                acc1 *= vals + 1;
            }
            i += 1;
        }
        if i >= dim && acc <= entries && acc1 > entries {
            return vals as i32;
        } else if i < dim || acc > entries {
            vals -= 1;
        } else {
            vals += 1;
        }
    }
}

/// Encoder-side `codebook` (codebook.h) as set up by `vorbis_book_init_encode`.
#[derive(Debug)]
pub(crate) struct Codebook {
    pub(crate) dim: i32,
    pub(crate) entries: i32,
    pub(crate) c: &'static StaticCodebook,
    pub(crate) codelist: Vec<u32>,
    pub(crate) quantvals: i32,
    pub(crate) minval: i32,
    pub(crate) delta: i32,
}

impl Codebook {
    /// Port of sharedbook.c `vorbis_book_init_encode`.
    pub(crate) fn init_encode(s: &'static StaticCodebook) -> Option<Self> {
        let lengths = &s.lengthlist[..s.entries as usize];
        Some(Codebook {
            dim: s.dim,
            entries: s.entries,
            c: s,
            codelist: make_words(lengths)?,
            quantvals: book_maptype1_quantvals(s),
            minval: rint(f64::from(float32_unpack(s.q_min))) as i32,
            delta: rint(f64::from(float32_unpack(s.q_delta))) as i32,
        })
    }

    /// Port of codebook.c `vorbis_book_encode`; returns the bits written.
    #[inline]
    pub(crate) fn encode(&self, a: i32, opb: &mut OggPackBuffer) -> i32 {
        if a < 0 || a >= self.c.entries {
            return 0;
        }
        let len = self.c.lengthlist[a as usize];
        opb.write(self.codelist[a as usize], u32::from(len));
        i32::from(len)
    }
}

/// Port of codebook.c `vorbis_staticbook_pack`. Returns `false` on the error
/// paths of the C code (impossible with the built-in books).
pub(crate) fn staticbook_pack(c: &StaticCodebook, opb: &mut OggPackBuffer) -> bool {
    let entries = c.entries as usize;
    let ll = c.lengthlist;
    opb.write(0x564342, 24);
    opb.write(c.dim as u32, 16);
    opb.write(c.entries as u32, 24);

    let mut i = 1;
    while i < entries {
        if ll[i - 1] == 0 || ll[i] < ll[i - 1] {
            break;
        }
        i += 1;
    }
    let ordered = i == entries;

    if ordered {
        let mut count: i64 = 0;
        opb.write(1, 1);
        opb.write(u32::from(ll[0]).wrapping_sub(1), 5);
        let mut i = 1;
        while i < entries {
            let this = ll[i];
            let last = ll[i - 1];
            if this > last {
                for _ in last..this {
                    opb.write(
                        (i as i64 - count) as u32,
                        ilog((c.entries as i64 - count) as u32) as u32,
                    );
                    count = i as i64;
                }
            }
            i += 1;
        }
        opb.write(
            (i as i64 - count) as u32,
            ilog((c.entries as i64 - count) as u32) as u32,
        );
    } else {
        opb.write(0, 1);
        if ll[..entries].iter().all(|&l| l != 0) {
            opb.write(0, 1);
            for &l in &ll[..entries] {
                opb.write(u32::from(l).wrapping_sub(1), 5);
            }
        } else {
            opb.write(1, 1);
            for &l in &ll[..entries] {
                if l == 0 {
                    opb.write(0, 1);
                } else {
                    opb.write(1, 1);
                    opb.write(u32::from(l) - 1, 5);
                }
            }
        }
    }

    opb.write(c.maptype as u32, 4);
    match c.maptype {
        0 => {}
        1 | 2 => {
            let Some(quantlist) = c.quantlist else {
                return false;
            };
            opb.write(c.q_min as u32, 32);
            opb.write(c.q_delta as u32, 32);
            opb.write((c.q_quant - 1) as u32, 4);
            opb.write(c.q_sequencep as u32, 1);
            let quantvals = if c.maptype == 1 {
                book_maptype1_quantvals(c)
            } else {
                c.entries * c.dim
            };
            for &q in &quantlist[..quantvals as usize] {
                opb.write(q.unsigned_abs(), c.q_quant as u32);
            }
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_words_canonical() {
        // lengths 2,2,2,3,3 -> 00,01,10,110,111 (MSb-first), stored bit-reversed
        let w = make_words(&[2, 2, 2, 3, 3]).unwrap();
        assert_eq!(w, vec![0b00, 0b10, 0b01, 0b011, 0b111]);
        // overpopulated
        assert!(make_words(&[1, 1, 1]).is_none());
        // underpopulated
        assert!(make_words(&[2, 2, 2]).is_none());
        // single-entry book
        assert_eq!(make_words(&[1]).unwrap(), vec![0]);
    }

    #[test]
    fn float32_unpack_values() {
        // q_min/q_delta of a typical residue book (_44c0_s_p1_0): -1.0 and 1.0
        assert_eq!(float32_unpack(-535822336), -1.0);
        assert_eq!(float32_unpack(1611661312), 1.0);
    }
}
