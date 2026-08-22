//! NAL-unit encapsulation — the write-side dual of [`crate::nal`].
//!
//! Wraps an RBSP into a complete NAL unit (the two-byte §7.3.1.2
//! header + the §7.4.1.1 emulation-prevention escaped payload) and
//! frames NAL units into an Annex B byte stream.

/// §7.4.1.1 — insert `emulation_prevention_three_byte`s: within the
/// payload, any `00 00` followed by a byte `<= 0x03` becomes
/// `00 00 03 xx` so no start-code prefix can appear inside a NAL unit.
#[must_use]
pub fn escape_rbsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 4);
    let mut zero_run = 0usize;
    for &b in rbsp {
        if zero_run >= 2 && b <= 0x03 {
            out.push(0x03);
            zero_run = 0;
        }
        out.push(b);
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
    out
}

/// Build a complete coded NAL unit: the §7.3.1.2 two-byte header
/// (`forbidden_zero_bit == 0`, `nal_unit_type`, `nuh_layer_id`,
/// `nuh_temporal_id_plus1 = temporal_id + 1`) followed by the escaped
/// RBSP.
#[must_use]
pub fn nal_unit(nal_unit_type: u8, nuh_layer_id: u8, temporal_id: u8, rbsp: &[u8]) -> Vec<u8> {
    debug_assert!(nal_unit_type < 64 && nuh_layer_id < 64 && temporal_id < 7);
    let b0 = (nal_unit_type << 1) | (nuh_layer_id >> 5);
    let b1 = ((nuh_layer_id & 0x1F) << 3) | (temporal_id + 1);
    let mut out = vec![b0, b1];
    out.extend(escape_rbsp(rbsp));
    out
}

/// Frame complete coded NAL units into an Annex B byte stream: a
/// four-byte start code (`00 00 00 01`) before parameter-set /
/// first-of-AU units keeps players happy; this writer simply uses the
/// four-byte form everywhere (§B.1.2 permits leading zeros).
#[must_use]
pub fn annexb(units: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for u in units {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(u);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::nal::{collect_nal_units, strip_emulation_prevention};

    #[test]
    fn escape_inserts_before_low_bytes() {
        // 00 00 00 -> 00 00 03 00; 00 00 01 -> 00 00 03 01; 00 00 04 stays.
        assert_eq!(
            escape_rbsp(&[0, 0, 0, 0, 1]),
            vec![0, 0, 3, 0, 0, 3, 1],
            "double escape over a zero run"
        );
        assert_eq!(escape_rbsp(&[0, 0, 4]), vec![0, 0, 4]);
        assert_eq!(escape_rbsp(&[0, 0, 2]), vec![0, 0, 3, 2]);
    }

    #[test]
    fn escape_roundtrips_through_reader_strip() {
        let payloads: [&[u8]; 5] = [
            &[],
            &[0, 0, 0, 0, 0, 0],
            &[0, 0, 1, 0, 0, 2, 0, 0, 3],
            &[0xFF, 0, 0, 0, 0xFF],
            &[0, 0],
        ];
        for p in payloads {
            let escaped = escape_rbsp(p);
            assert_eq!(strip_emulation_prevention(&escaped), p, "{p:02x?}");
            // No start code may survive inside the escaped payload.
            assert!(!escaped.windows(3).any(|w| w == [0, 0, 1] || w == [0, 0, 0]));
        }
    }

    #[test]
    fn nal_unit_header_roundtrips_through_walker() {
        let rbsp = vec![0xAA, 0x00, 0x00, 0x00, 0xBB];
        let unit = nal_unit(19, 0, 0, &rbsp); // IDR_W_RADL
        let stream = annexb(&[unit]);
        let units = collect_nal_units(&stream).expect("walk");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].header.nal_unit_type, 19);
        assert_eq!(units[0].header.nuh_layer_id, 0);
        assert_eq!(units[0].header.temporal_id, 0);
        assert_eq!(units[0].rbsp, rbsp);
    }

    #[test]
    fn annexb_frames_multiple_units() {
        let a = nal_unit(32, 0, 0, &[0x01]);
        let b = nal_unit(33, 0, 0, &[0x02]);
        let stream = annexb(&[a, b]);
        let units = collect_nal_units(&stream).expect("walk");
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].header.nal_unit_type, 32);
        assert_eq!(units[1].header.nal_unit_type, 33);
    }
}
