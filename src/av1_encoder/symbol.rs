//! AV1 multi-symbol arithmetic (range) encoder (AV1 §8.2, encoder side).
//!
//! The AV1 spec only defines the *decoder* (§8.2 "Parsing process for symbol decoder"). This is
//! the matching encoder: it produces a byte stream that the §8.2 decoder maps back to the symbols
//! that were encoded. The arithmetic mirrors the well-known `od_ec` range coder (the same one in
//! libaom / rav1e), which is purpose-built for this decoder.
//!
//! CDF convention (matches §8.2.6): a CDF for `N` symbols is a slice of `N` cumulative values in
//! `[0, 32768]`, strictly non-decreasing, with `cdf[N - 1] == 32768`. `cdf[i]` is the cumulative
//! probability (× 32768) of symbols `0..=i`. The adaptation counter the spec stores as a trailing
//! `cdf[N]` element is irrelevant here: this MVP runs with `disable_cdf_update = 1`, so CDFs are
//! static and never adapted. Adaptation is deferred to M1 (see `gamut-avif/STATUS.md`).
//!
//! The hermetic `SymbolDecoder` in this module's tests is a direct transcription of §8.2 and is
//! the oracle that proves the encoder correct without any external decoder.

/// Number of bits to reduce CDF precision during arithmetic coding (AV1 `EC_PROB_SHIFT`, §3).
// Adapted and modified from gamut, Copyright (c) 2026 Justin Chung, MIT licensed.
const EC_PROB_SHIFT: u32 = 6;
/// Minimum probability assigned to each symbol during arithmetic coding (AV1 `EC_MIN_PROB`, §3).
const EC_MIN_PROB: u32 = 4;
/// CDFs are expressed on a 1 << 15 scale (AV1 §8.2.6: `cdf[N - 1] == 1 << 15`).
const CDF_PROB_TOP: u32 = 1 << 15;

/// Encoder for the AV1 symbol (range) coder.
///
/// Feed symbols with [`SymbolEncoder::encode_symbol`] (CDF-coded) and equiprobable bits with
/// [`SymbolEncoder::encode_literal`], then call [`SymbolEncoder::finish`] to flush and obtain the
/// coded bytes. Those bytes are exactly what a decoder consumes via `init_symbol(sz)` (AV1 §8.2.2)
/// where `sz` is the returned length.
#[derive(Debug, Clone)]
pub struct SymbolEncoder {
    /// Low end of the coding interval, kept wider than 16 bits so carries accumulate losslessly
    /// (resolved in [`SymbolEncoder::finish`]).
    low: u64,
    /// Current range, renormalised into `[1 << 15, 1 << 16)`.
    rng: u32,
    /// Bit counter; starts at `-9` so the first carry/byte crosses zero at the right moment.
    cnt: i32,
    /// Coded bytes, already carry-resolved. `od_ec` buffers every byte as a `u16` and resolves the
    /// pending carries in a second reverse pass over the whole stream at `finish`; this keeps the
    /// sink half as wide and drops that pass by normalising each byte as it arrives (see
    /// [`SymbolEncoder::push_byte`]).
    out: Vec<u8>,
    /// Length of the run of `0xFF` bytes at the end of `out`, which is exactly the span an
    /// incoming carry has to sweep before it lands.
    ff_run: usize,
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolEncoder {
    /// Creates an encoder with the initial range state of AV1's symbol coder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            low: 0,
            rng: CDF_PROB_TOP,
            cnt: -9,
            out: Vec::new(),
            ff_run: 0,
        }
    }

    /// Encodes `symbol` against a static cumulative `cdf` (`cdf.len()` symbols, `cdf[last] == 32768`).
    ///
    /// # Panics
    ///
    /// Debug builds assert `symbol < cdf.len()` and the CDF normalisation invariants.
    pub fn encode_symbol(&mut self, symbol: usize, cdf: &[u16]) {
        let nsyms = cdf.len();
        debug_assert!(symbol < nsyms);
        debug_assert_eq!(u32::from(cdf[nsyms - 1]), CDF_PROB_TOP);
        // `f(j) = (1 << 15) - cdf[j]` is the inverse-CDF term used by the §8.2.6 decoder; `fl`/`fh`
        // bracket the chosen symbol's sub-interval. For symbol 0, the upper bracket is the full top.
        let fl = if symbol > 0 {
            CDF_PROB_TOP - u32::from(cdf[symbol - 1])
        } else {
            CDF_PROB_TOP
        };
        let fh = CDF_PROB_TOP - u32::from(cdf[symbol]);
        self.encode_q15(fl, fh, symbol as u32, nsyms as u32);
    }

    /// Encodes the low `n` bits of `value` as equiprobable bits, most-significant bit first.
    ///
    /// This is the inverse of the decoder's `read_literal(n)` (AV1 §8.2.5), which itself calls
    /// `read_bool()` (§8.2.3) with the fixed CDF `{1 << 14, 1 << 15}`.
    pub fn encode_literal(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        // Specialisation of `encode_q15` for the fixed CDF `{1 << 14, 1 << 15}`, which is the only
        // one `read_bool` ever uses. With `nsyms = 2` both branches collapse onto the same
        // split point
        //
        //     w = ((rng >> 8) << 7) + EC_MIN_PROB
        //
        // because `fh >> EC_PROB_SHIFT` is `256` for a zero bit and the `EC_MIN_PROB` term is `4`
        // either way: a zero bit keeps `low` and takes `rng - w`, a one bit adds `rng - w` to `low`
        // and takes `w`. That is bit-for-bit what the CDF path computes, so the stream is
        // unchanged; what the run buys is keeping `low`, `rng` and `cnt` in registers across the
        // whole run instead of reloading them, and dropping the CDF slice indexing and the
        // per-symbol branch on `fl < CDF_PROB_TOP`.
        let mut low = self.low;
        let mut rng = self.rng;
        let mut cnt = self.cnt;
        for i in (0..n).rev() {
            debug_assert!(rng >= CDF_PROB_TOP);
            let w = ((rng >> 8) << 7) + EC_MIN_PROB;
            if (value >> i) & 1 == 0 {
                rng -= w;
            } else {
                low += u64::from(rng - w);
                rng = w;
            }
            // `normalize`, inlined so the state stays in locals for the next bit.
            let d = rng.leading_zeros() - 16;
            let mut s = cnt + d as i32;
            if s >= 0 {
                cnt += 16;
                let mut m = (1u64 << cnt) - 1;
                if s >= 8 {
                    self.push_byte((low >> cnt) as u16);
                    low &= m;
                    cnt -= 8;
                    m = (1u64 << cnt) - 1;
                }
                self.push_byte((low >> cnt) as u16);
                s = cnt + d as i32 - 24;
                low &= m;
            }
            low <<= d;
            rng <<= d;
            cnt = s;
        }
        self.low = low;
        self.rng = rng;
        self.cnt = cnt;
    }

    /// Core interval update for one symbol; `fl`/`fh` are the inverse-CDF brackets, `s` the symbol,
    /// `nsyms` the alphabet size. Mirrors `od_ec_encode_q15`, which inverts the §8.2.6 boundaries.
    fn encode_q15(&mut self, fl: u32, fh: u32, s: u32, nsyms: u32) {
        let mut low = self.low;
        let mut r = self.rng;
        debug_assert!(r >= CDF_PROB_TOP);
        let n = nsyms - 1;
        if fl < CDF_PROB_TOP {
            let u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - (s - 1));
            let v =
                (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s);
            debug_assert!(u <= r && v < u);
            low += u64::from(r - u);
            r = u - v;
        } else {
            // Symbol 0: the interval reaches the top, so `low` is unchanged.
            let v =
                (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s);
            debug_assert!(v < r);
            r -= v;
        }
        self.normalize(low, r);
    }

    /// Renormalises `(low, rng)` back into `[1 << 15, 1 << 16)`, emitting completed bytes into
    /// `precarry`. Mirrors `od_ec_enc_normalize`.
    fn normalize(&mut self, mut low: u64, rng: u32) {
        // `d` = number of left shifts to bring `rng` to 16 bits. `rng` is in `[1, 0xFFFF]` here.
        let d = rng.leading_zeros() - 16;
        let mut c = self.cnt;
        let mut s = c + d as i32;
        if s >= 0 {
            c += 16;
            let mut m = (1u64 << c) - 1;
            if s >= 8 {
                self.push_byte((low >> c) as u16);
                low &= m;
                c -= 8;
                m = (1u64 << c) - 1;
            }
            self.push_byte((low >> c) as u16);
            s = c + d as i32 - 24;
            low &= m;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// Appends one nine-bit `od_ec` output digit, resolving its carry immediately.
    ///
    /// The buffered stream is a base-256 numeral whose digits arrive most-significant first, so a
    /// digit above `0xFF` carries one into the byte already written. `out` is kept normalised, so
    /// that carry can only sweep the trailing run of `0xFF` bytes — which is what `ff_run` tracks —
    /// before landing on a byte that can absorb it. Each `0xFF` is pushed once and swept at most
    /// once, so the sweep is amortised constant time. A carry off the front of the stream is
    /// discarded, exactly as `od_ec_enc_done`'s reverse pass discards it.
    fn push_byte(&mut self, value: u16) {
        debug_assert!(value <= 0x1ff, "od_ec output digits are nine bits wide");
        if value > 0xff {
            let end = self.out.len();
            let start = end - self.ff_run;
            self.out[start..end].fill(0);
            if start > 0 {
                self.out[start - 1] += 1;
            }
            self.ff_run = 0;
        }
        let byte = (value & 0xff) as u8;
        self.out.push(byte);
        if byte == 0xff {
            self.ff_run += 1;
        } else {
            self.ff_run = 0;
        }
    }

    /// Flushes the coder and returns the coded bytes. Mirrors `od_ec_enc_done`: it emits the
    /// minimum number of bits that decode correctly regardless of trailing padding. The carries
    /// are already resolved by [`SymbolEncoder::push_byte`], so no second pass is needed.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10 + c;
        let m: u64 = 0x3FFF;
        let mut e = ((l + m) & !m) | (m + 1);
        if s > 0 {
            let mut n = (1u64 << (c + 16)) - 1;
            loop {
                self.push_byte((e >> (c + 16)) as u16);
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct transcription of the AV1 §8.2 symbol decoder — the hermetic oracle for the encoder.
    struct SymbolDecoder<'a> {
        data: &'a [u8],
        bit_pos: usize,
        value: u32,
        range: u32,
        max_bits: i64,
    }

    impl<'a> SymbolDecoder<'a> {
        /// `f(n)` parsing process (AV1 §8.1): MSB-first, zero-padded past the end of `data`.
        fn read_f(&mut self, n: u32) -> u32 {
            let mut x = 0u32;
            for _ in 0..n {
                let idx = self.bit_pos >> 3;
                let bit = if idx < self.data.len() {
                    (self.data[idx] >> (7 - (self.bit_pos & 7))) & 1
                } else {
                    0
                };
                x = (x << 1) | u32::from(bit);
                self.bit_pos += 1;
            }
            x
        }

        /// `init_symbol(sz)` (AV1 §8.2.2).
        fn new(data: &'a [u8]) -> Self {
            let sz = data.len();
            let mut d = Self {
                data,
                bit_pos: 0,
                value: 0,
                range: 1 << 15,
                max_bits: 8 * sz as i64 - 15,
            };
            let num_bits = core::cmp::min(sz * 8, 15) as u32;
            let buf = d.read_f(num_bits);
            let padded = buf << (15 - num_bits);
            d.value = ((1 << 15) - 1) ^ padded;
            d
        }

        /// `read_symbol(cdf)` (AV1 §8.2.6); `cdf` is the cumulative form (no trailing count needed
        /// because adaptation is disabled).
        fn read_symbol(&mut self, cdf: &[u16]) -> usize {
            let n = cdf.len() as u32;
            let mut cur = self.range;
            let mut symbol: i64 = -1;
            let mut prev;
            loop {
                symbol += 1;
                prev = cur;
                let f = (1u32 << 15) - u32::from(cdf[symbol as usize]);
                cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
                cur += EC_MIN_PROB * (n - symbol as u32 - 1);
                if self.value >= cur {
                    break;
                }
            }
            self.range = prev - cur;
            self.value -= cur;
            // Renormalisation (AV1 §8.2.6 ordered steps).
            let bits = 15 - (31 - self.range.leading_zeros());
            self.range <<= bits;
            let num_bits = core::cmp::min(i64::from(bits), self.max_bits.max(0)) as u32;
            let new_data = self.read_f(num_bits);
            let padded = new_data << (bits - num_bits);
            self.value = padded ^ (((self.value + 1) << bits) - 1);
            self.max_bits -= i64::from(bits);
            symbol as usize
        }

        fn read_literal(&mut self, n: u32) -> u32 {
            const BOOL_CDF: [u16; 2] = [1 << 14, 1 << 15];
            let mut x = 0;
            for _ in 0..n {
                x = (x << 1) | self.read_symbol(&BOOL_CDF) as u32;
            }
            x
        }
    }

    /// Small deterministic LCG so tests are reproducible without `rand`.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, bound: u32) -> u32 {
            self.next_u32() % bound
        }
    }

    /// Builds a random strictly-increasing cumulative CDF for `nsyms` symbols, `cdf[last] = 32768`.
    fn random_cdf(rng: &mut Lcg, nsyms: usize) -> Vec<u16> {
        // Pick `nsyms - 1` distinct breakpoints in 1..32768, sorted, then append 32768.
        let mut points = Vec::new();
        while points.len() < nsyms - 1 {
            let p = 1 + rng.below(32767) as u16;
            if !points.contains(&p) {
                points.push(p);
            }
        }
        points.sort_unstable();
        points.push(32768);
        points
    }

    #[test]
    fn empty_stream_roundtrips() {
        let enc = SymbolEncoder::new();
        let bytes = enc.finish();
        // Nothing to decode; just ensure init does not panic.
        let _ = SymbolDecoder::new(&bytes);
    }

    #[test]
    fn single_symbol_streams_roundtrip() {
        // Exhaustively exercise small alphabets with a skewed CDF and every symbol value.
        for nsyms in 2..=12usize {
            let mut cdf: Vec<u16> = (1..nsyms).map(|i| (i * 32768 / nsyms) as u16).collect();
            cdf.push(32768);
            for s in 0..nsyms {
                let mut enc = SymbolEncoder::new();
                enc.encode_symbol(s, &cdf);
                let bytes = enc.finish();
                let mut dec = SymbolDecoder::new(&bytes);
                assert_eq!(dec.read_symbol(&cdf), s, "nsyms={nsyms} s={s}");
            }
        }
    }

    #[test]
    fn long_random_symbol_stream_roundtrips() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        // Pre-generate a mix of CDFs of varying sizes.
        let cdfs: Vec<Vec<u16>> = (2..=14).map(|n| random_cdf(&mut rng, n)).collect();
        let mut events = Vec::new();
        let mut enc = SymbolEncoder::new();
        for _ in 0..20_000 {
            let cdf = &cdfs[rng.below(cdfs.len() as u32) as usize];
            let s = rng.below(cdf.len() as u32) as usize;
            enc.encode_symbol(s, cdf);
            events.push((s, cdf.clone()));
        }
        let bytes = enc.finish();
        let mut dec = SymbolDecoder::new(&bytes);
        for (i, (s, cdf)) in events.iter().enumerate() {
            assert_eq!(dec.read_symbol(cdf), *s, "event {i}");
        }
    }

    #[test]
    fn literals_roundtrip() {
        let mut rng = Lcg(0xdead_beef_0bad_f00d);
        let mut enc = SymbolEncoder::new();
        let mut events = Vec::new();
        for _ in 0..5000 {
            let n = 1 + rng.below(16);
            let v = rng.next_u32() & ((1u32 << n) - 1);
            enc.encode_literal(v, n);
            events.push((v, n));
        }
        let bytes = enc.finish();
        let mut dec = SymbolDecoder::new(&bytes);
        for (v, n) in events {
            assert_eq!(dec.read_literal(n), v);
        }
    }

    /// Bit-at-a-time reference for [`SymbolEncoder::encode_literal`]: the fixed-CDF `read_bool`
    /// path the AV1 spec defines `read_literal(n)` in terms of (§8.2.5, §8.2.3).
    fn encode_literal_bit_at_a_time(enc: &mut SymbolEncoder, value: u32, n: u32) {
        const BOOL_CDF: [u16; 2] = [1 << 14, 1 << 15];
        for i in (0..n).rev() {
            enc.encode_symbol(((value >> i) & 1) as usize, &BOOL_CDF);
        }
    }

    #[test]
    fn unrolled_literal_run_matches_bit_at_a_time_at_every_run_length() {
        // Every run length the coefficient path can produce (signs are 1, golomb tails run up to
        // the 2 * len - 1 field width), each checked byte-identical against the reference engine.
        let mut rng = Lcg(0xa5a5_1234_dead_0001);
        for n in 1..=24u32 {
            let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            // Exhaustive for the narrow widths, sampled plus the extremes for the wide ones.
            let values: Vec<u32> = if n <= 10 {
                (0..=mask).collect()
            } else {
                let mut v = vec![0, 1, mask, mask >> 1, mask ^ 1];
                v.extend((0..256).map(|_| rng.next_u32() & mask));
                v
            };
            for &value in &values {
                let mut fast = SymbolEncoder::new();
                fast.encode_literal(value, n);
                let mut reference = SymbolEncoder::new();
                encode_literal_bit_at_a_time(&mut reference, value, n);
                assert_eq!(
                    fast.finish(),
                    reference.finish(),
                    "run length {n}, value {value:#x}"
                );
            }
        }
    }

    #[test]
    fn unrolled_literal_runs_match_bit_at_a_time_when_interleaved_with_symbols() {
        // A single run in isolation starts from the initial state; the coefficient path reaches
        // `encode_literal` at arbitrary `(low, rng, cnt)`, including with carries pending, so drive
        // both engines through the same long mixed stream and compare the whole bitstream.
        let mut rng = Lcg(0x1357_9bdf_2468_ace0);
        let cdfs: Vec<Vec<u16>> = (2..=14).map(|n| random_cdf(&mut rng, n)).collect();
        let mut fast = SymbolEncoder::new();
        let mut reference = SymbolEncoder::new();
        for _ in 0..50_000 {
            if rng.next_u32() & 1 == 0 {
                let cdf = &cdfs[rng.below(cdfs.len() as u32) as usize];
                let s = rng.below(cdf.len() as u32) as usize;
                fast.encode_symbol(s, cdf);
                reference.encode_symbol(s, cdf);
            } else {
                let n = 1 + rng.below(24);
                let value = rng.next_u32() & ((1u32 << n) - 1);
                fast.encode_literal(value, n);
                encode_literal_bit_at_a_time(&mut reference, value, n);
            }
        }
        assert_eq!(fast.finish(), reference.finish());
    }

    /// `od_ec_enc_done`'s original carry resolution: buffer every nine-bit digit, then sweep the
    /// whole stream from the end. This is the oracle the eager sink has to match exactly.
    fn resolve_carries_reference(digits: &[u16]) -> Vec<u8> {
        let mut out = vec![0u8; digits.len()];
        let mut carry: u32 = 0;
        for i in (0..digits.len()).rev() {
            let val = u32::from(digits[i]) + carry;
            out[i] = (val & 0xff) as u8;
            carry = val >> 8;
        }
        out
    }

    #[test]
    fn eager_carry_sink_matches_the_reverse_pass() {
        let mut rng = Lcg(0xffff_0000_ffff_0001);
        let mut cases: Vec<Vec<u16>> = vec![
            vec![],
            vec![0x000],
            // A carry off the front of the stream, which both engines discard.
            vec![0x1ff],
            vec![0x0ff, 0x1ff],
            // A carry sweeping a long run of 0xFF bytes back onto a byte that can absorb it.
            {
                let mut v = vec![0x012];
                v.extend(std::iter::repeat_n(0x0ff, 64));
                v.push(0x1ab);
                v
            },
            // The same run, but with nothing in front of it to absorb the carry.
            {
                let mut v = vec![0x0ff; 32];
                v.push(0x100);
                v
            },
            // Back-to-back carries, so a digit is swept more than once across the stream.
            {
                let mut v = vec![0x001];
                for _ in 0..16 {
                    v.extend([0x0ff, 0x0ff, 0x1fe]);
                }
                v
            },
        ];
        // Plus random digit streams biased toward the values that make carries interesting.
        for _ in 0..200 {
            let len = 1 + rng.below(128) as usize;
            cases.push(
                (0..len)
                    .map(|_| match rng.below(4) {
                        0 => 0x0ff,
                        1 => 0x100 + rng.below(0x100) as u16,
                        _ => rng.below(0x100) as u16,
                    })
                    .collect(),
            );
        }
        for digits in &cases {
            let mut enc = SymbolEncoder::new();
            for &d in digits {
                enc.push_byte(d);
            }
            assert_eq!(
                enc.out,
                resolve_carries_reference(digits),
                "digits {digits:x?}"
            );
        }
    }

    #[test]
    fn carry_heavy_streams_roundtrip() {
        // A CDF skewed hard toward the symbol that grows `low` is what drives carries through the
        // sink in real coding, so check the decoder still reads such a stream back exactly.
        let mut rng = Lcg(0x2468_ace0_1357_9bdf);
        for &(lo, bound) in &[(4u16, 64u32), (32764, 3), (16384, 2)] {
            let skewed: Vec<u16> = vec![lo, 32768];
            let mut enc = SymbolEncoder::new();
            let mut events = Vec::new();
            for _ in 0..20_000 {
                let s = usize::from(rng.below(bound) != 0);
                enc.encode_symbol(s, &skewed);
                events.push(s);
            }
            let bytes = enc.finish();
            let mut dec = SymbolDecoder::new(&bytes);
            for (i, s) in events.iter().enumerate() {
                assert_eq!(dec.read_symbol(&skewed), *s, "lo={lo} event {i}");
            }
        }
    }

    #[test]
    fn mixed_symbols_and_literals_roundtrip() {
        let mut rng = Lcg(0x0f0f_0f0f_1234_9999);
        let cdf = random_cdf(&mut rng, 8);
        let mut enc = SymbolEncoder::new();
        let mut events: Vec<(bool, u32)> = Vec::new(); // (is_literal, payload)
        for _ in 0..8000 {
            if rng.next_u32() & 1 == 0 {
                let s = rng.below(cdf.len() as u32);
                enc.encode_symbol(s as usize, &cdf);
                events.push((false, s));
            } else {
                let v = rng.next_u32() & 0xff;
                enc.encode_literal(v, 8);
                events.push((true, v));
            }
        }
        let bytes = enc.finish();
        let mut dec = SymbolDecoder::new(&bytes);
        for (is_lit, payload) in events {
            if is_lit {
                assert_eq!(dec.read_literal(8), payload);
            } else {
                assert_eq!(dec.read_symbol(&cdf) as u32, payload);
            }
        }
    }
}
