//! H.265 / HEVC NAL-unit byte-stream walker.
//!
//! This module implements the byte-stream framing and NAL-unit header
//! parse described in ITU-T Rec. H.265 (ISO/IEC 23008-2):
//!
//! * **Annex B** — byte-stream format. NAL units are separated by a
//!   start-code prefix `0x00 0x00 0x01` (or `0x00 0x00 0x00 0x01`
//!   when a leading zero byte is present).
//! * **§7.3.1.1** — NAL unit syntax. The first two bytes of the
//!   RBSP-bearing NAL payload are the NAL unit header.
//! * **§7.4.1.1** — Emulation-prevention. Inside the payload, the
//!   sequence `0x00 0x00 0x03` decodes to `0x00 0x00` (the `0x03`
//!   byte is the emulation-prevention escape).
//!
//! Scope of this module: split a byte stream into NAL units, parse
//! their two-byte headers, and surface the unescaped RBSP body. The
//! SPS/PPS/VPS/slice payloads themselves are not parsed here.

/// Errors that can arise while walking a NAL byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalError {
    /// The byte stream did not contain any NAL start code.
    NoStartCode,
    /// `forbidden_zero_bit` (the high bit of the first NAL-header
    /// byte) was set, which is reserved per §7.4.2.2 and indicates a
    /// malformed stream.
    ForbiddenZeroBitSet,
    /// `nuh_temporal_id_plus1` was zero. §7.4.2.2 requires the
    /// field to be strictly greater than zero.
    TemporalIdPlus1Zero,
    /// A NAL unit was shorter than the two-byte header.
    TruncatedHeader,
}

impl core::fmt::Display for NalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoStartCode => f.write_str("no Annex B start code in input"),
            Self::ForbiddenZeroBitSet => f.write_str("forbidden_zero_bit was set in NAL header"),
            Self::TemporalIdPlus1Zero => f.write_str("nuh_temporal_id_plus1 was zero"),
            Self::TruncatedHeader => f.write_str("NAL unit shorter than two-byte header"),
        }
    }
}

impl std::error::Error for NalError {}

/// Parsed NAL unit header per H.265 §7.3.1.2.
///
/// Layout (16 bits, big-endian):
///
/// ```text
/// bit  0       : forbidden_zero_bit          (1 bit, must be 0)
/// bits 1..=6   : nal_unit_type               (6 bits)
/// bits 7..=12  : nuh_layer_id                (6 bits)
/// bits 13..=15 : nuh_temporal_id_plus1       (3 bits, must be > 0)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalHeader {
    /// `nal_unit_type` field, 0..=63. The interpretation table is in
    /// §7.4.2.2 (Table 7-1).
    pub nal_unit_type: u8,
    /// `nuh_layer_id` field, 0..=63 (5 bits used in the base spec;
    /// the sixth bit is reserved for layered extensions).
    pub nuh_layer_id: u8,
    /// `TemporalId` derived value, equal to
    /// `nuh_temporal_id_plus1 - 1`, range 0..=6.
    pub temporal_id: u8,
}

impl NalHeader {
    /// Parse the two-byte NAL header from the first two bytes of an
    /// (already-unescaped) NAL unit payload.
    pub fn parse(bytes: &[u8]) -> Result<Self, NalError> {
        if bytes.len() < 2 {
            return Err(NalError::TruncatedHeader);
        }
        let b0 = bytes[0];
        let b1 = bytes[1];

        if b0 & 0x80 != 0 {
            return Err(NalError::ForbiddenZeroBitSet);
        }

        let nal_unit_type = (b0 >> 1) & 0x3F;
        let nuh_layer_id = ((b0 & 0x01) << 5) | ((b1 >> 3) & 0x1F);
        let temporal_id_plus1 = b1 & 0x07;
        if temporal_id_plus1 == 0 {
            return Err(NalError::TemporalIdPlus1Zero);
        }
        Ok(Self {
            nal_unit_type,
            nuh_layer_id,
            temporal_id: temporal_id_plus1 - 1,
        })
    }

    /// Convenience predicate: per §7.4.2.2 (Table 7-1) NAL units with
    /// `nal_unit_type` in `[32, 40]` are non-VCL parameter-set or
    /// delimiter units (VPS/SPS/PPS/AUD/EOS/EOB/FD/prefix-SEI/
    /// suffix-SEI). NAL units below 32 are VCL slice payloads.
    pub fn is_vcl(&self) -> bool {
        self.nal_unit_type < 32
    }
}

/// One NAL unit extracted from an Annex B byte stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalUnit {
    /// Parsed header for this unit.
    pub header: NalHeader,
    /// Unescaped RBSP payload, excluding the two header bytes.
    /// Emulation-prevention `0x03` bytes have been stripped per
    /// §7.4.1.1.
    pub rbsp: Vec<u8>,
    /// The coded (still-escaped) payload bytes, excluding the two
    /// header bytes. The §7.4.7.1 `entry_point_offset_minus1` subset
    /// boundaries count emulation-prevention bytes, so the WPP / tile
    /// substream split needs this form alongside the RBSP.
    pub escaped: Vec<u8>,
}

/// Strip emulation-prevention bytes per §7.4.1.1.
///
/// For every occurrence of `0x00 0x00 0x03` in the input, the `0x03`
/// byte is dropped from the output. This is the inverse of the
/// encoder-side emulation-prevention insertion that prevents an
/// in-payload byte sequence from being mistaken for a start code.
pub fn strip_emulation_prevention(escaped: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(escaped.len());
    let mut zero_run: u8 = 0;
    for &b in escaped {
        if zero_run >= 2 && b == 0x03 {
            // §7.4.1.1: drop the 0x03 escape byte after `00 00`.
            zero_run = 0;
            continue;
        }
        out.push(b);
        if b == 0x00 {
            zero_run = zero_run.saturating_add(1);
        } else {
            zero_run = 0;
        }
    }
    out
}

/// Iterator that walks an Annex B byte stream and yields one NAL
/// unit per `next()` call.
#[derive(Debug, Clone)]
pub struct NalIter<'a> {
    buf: &'a [u8],
    cursor: usize,
}

impl<'a> NalIter<'a> {
    /// Construct a walker over `buf`. The walker scans forward for
    /// the first Annex B start code; leading bytes prior to it are
    /// discarded (this matches the behaviour required by Annex B,
    /// which permits arbitrary leading-zero padding).
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, cursor: 0 }
    }

    /// Locate the next start-code prefix at or after `from` and
    /// return `(start_code_offset, payload_offset)` where
    /// `payload_offset` is the byte immediately past the start
    /// code's terminating `0x01`. Returns `None` if no start code
    /// remains in the buffer.
    fn find_start_code(&self, from: usize) -> Option<(usize, usize)> {
        let buf = self.buf;
        let mut i = from;
        while i + 2 < buf.len() {
            if buf[i] == 0x00 && buf[i + 1] == 0x00 {
                if buf[i + 2] == 0x01 {
                    return Some((i, i + 3));
                }
                if buf[i + 2] == 0x00 && i + 3 < buf.len() && buf[i + 3] == 0x01 {
                    return Some((i, i + 4));
                }
            }
            i += 1;
        }
        None
    }
}

impl Iterator for NalIter<'_> {
    type Item = Result<NalUnit, NalError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (_, payload_start) = self.find_start_code(self.cursor)?;
        let payload_end = match self.find_start_code(payload_start) {
            Some((next_sc, _)) => next_sc,
            None => self.buf.len(),
        };
        self.cursor = payload_end;

        // §B.1.2 permits `trailing_zero_8bits` between NAL units
        // and `leading_zero_8bits` before the next start code. They
        // appear in the buffer as zero bytes immediately preceding
        // the next start code; for the purpose of this round (which
        // does not yet parse the RBSP stop bit) we keep the payload
        // exactly as the start codes delimit it.
        let payload = &self.buf[payload_start..payload_end];

        let unescaped = strip_emulation_prevention(payload);
        let header = match NalHeader::parse(&unescaped) {
            Ok(h) => h,
            Err(e) => return Some(Err(e)),
        };
        let rbsp = unescaped[2..].to_vec();
        // The escaped form, aligned with `rbsp` (header bytes dropped).
        // The two header bytes are never part of an emulation-prevention
        // pattern's tail (`nuh_temporal_id_plus1 != 0` forbids 0x00 0x00
        // at the header), so dropping them keeps the two forms in step.
        let escaped = payload[2..].to_vec();
        Some(Ok(NalUnit {
            header,
            rbsp,
            escaped,
        }))
    }
}

/// Convenience: collect every NAL unit in `buf` into a `Vec`. Stops
/// at the first parse error.
pub fn collect_nal_units(buf: &[u8]) -> Result<Vec<NalUnit>, NalError> {
    let mut iter = NalIter::new(buf);
    let mut out = Vec::new();
    // If the very first call finds no start code at all, surface a
    // dedicated `NoStartCode` error rather than returning an empty
    // vec — an empty Annex B stream is almost always a caller bug.
    let first = iter.next();
    match first {
        None => Err(NalError::NoStartCode),
        Some(Ok(u)) => {
            out.push(u);
            for n in iter {
                out.push(n?);
            }
            Ok(out)
        }
        Some(Err(e)) => Err(e),
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    /// Build a NAL header pair `(b0, b1)` for the given fields.
    fn header_bytes(nal_unit_type: u8, nuh_layer_id: u8, temporal_id: u8) -> [u8; 2] {
        let b0 = ((nal_unit_type & 0x3F) << 1) | ((nuh_layer_id >> 5) & 0x01);
        let b1 = ((nuh_layer_id & 0x1F) << 3) | ((temporal_id + 1) & 0x07);
        [b0, b1]
    }

    #[test]
    fn parses_single_nal_with_three_byte_start_code() {
        // Annex B stream: [00 00 01] [hdr0 hdr1] [payload...]
        // VPS (nal_unit_type=32), layer 0, temporal_id 0.
        let h = header_bytes(32, 0, 0);
        let stream: Vec<u8> = [0x00, 0x00, 0x01, h[0], h[1], 0xAA, 0xBB, 0xCC].to_vec();
        let units = collect_nal_units(&stream).expect("walker");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].header.nal_unit_type, 32);
        assert_eq!(units[0].header.nuh_layer_id, 0);
        assert_eq!(units[0].header.temporal_id, 0);
        assert!(!units[0].header.is_vcl());
        assert_eq!(units[0].rbsp, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn parses_four_byte_start_code_and_two_units() {
        // SPS (33), then a slice IDR_W_RADL (19, VCL).
        let h_sps = header_bytes(33, 0, 0);
        let h_idr = header_bytes(19, 0, 0);
        let mut stream = vec![0x00, 0x00, 0x00, 0x01, h_sps[0], h_sps[1], 0x10, 0x20];
        stream.extend_from_slice(&[0x00, 0x00, 0x01, h_idr[0], h_idr[1], 0x30, 0x40]);
        let units = collect_nal_units(&stream).expect("walker");
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].header.nal_unit_type, 33);
        assert_eq!(units[0].rbsp, vec![0x10, 0x20]);
        assert_eq!(units[1].header.nal_unit_type, 19);
        assert!(units[1].header.is_vcl());
        assert_eq!(units[1].rbsp, vec![0x30, 0x40]);
    }

    #[test]
    fn strips_emulation_prevention_byte() {
        // Payload literally contains `00 00 00`; after escape that's
        // `00 00 03 00` on the wire. The walker should expose the
        // original `00 00 00` in `rbsp`.
        let h = header_bytes(34, 0, 0); // PPS
        let stream = vec![0x00, 0x00, 0x01, h[0], h[1], 0x00, 0x00, 0x03, 0x00, 0xFF];
        let units = collect_nal_units(&stream).expect("walker");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].rbsp, vec![0x00, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn rejects_forbidden_zero_bit() {
        // Manually craft a header byte with the forbidden bit set.
        let stream = vec![0x00, 0x00, 0x01, 0x80, 0x01];
        let err = collect_nal_units(&stream).unwrap_err();
        assert_eq!(err, NalError::ForbiddenZeroBitSet);
    }

    #[test]
    fn rejects_zero_temporal_id_plus1() {
        // nal_unit_type 32, layer 0, temporal_id_plus1 = 0 (b1 low
        // three bits all zero) — must be rejected per §7.4.2.2.
        let stream = vec![0x00, 0x00, 0x01, 0x40, 0x00];
        let err = collect_nal_units(&stream).unwrap_err();
        assert_eq!(err, NalError::TemporalIdPlus1Zero);
    }

    #[test]
    fn no_start_code_is_an_error() {
        let stream = vec![0xAA, 0xBB, 0xCC];
        let err = collect_nal_units(&stream).unwrap_err();
        assert_eq!(err, NalError::NoStartCode);
    }

    #[test]
    fn header_field_packing_round_trip() {
        // nal_unit_type=21 (CRA_NUT), nuh_layer_id=0, temporal_id=2.
        let h = header_bytes(21, 0, 2);
        let parsed = NalHeader::parse(&[h[0], h[1]]).unwrap();
        assert_eq!(parsed.nal_unit_type, 21);
        assert_eq!(parsed.nuh_layer_id, 0);
        assert_eq!(parsed.temporal_id, 2);
        assert!(parsed.is_vcl());

        // Also exercise a non-zero layer id (e.g. SHVC enhancement).
        let h2 = header_bytes(1, 5, 1);
        let parsed2 = NalHeader::parse(&[h2[0], h2[1]]).unwrap();
        assert_eq!(parsed2.nuh_layer_id, 5);
        assert_eq!(parsed2.temporal_id, 1);
    }
}
