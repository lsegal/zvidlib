//! EBML element IDs and variable-size integer coding shared by the WebM
//! demuxer ([`crate::webm_demux`]) and muxer ([`crate::webm`]).
//!
//! Only the Matroska elements the WebM subset uses, and that zvidlib reads or
//! writes, are named here. Element IDs keep their length marker, as the
//! Matroska specification writes them.

use crate::{Error, ErrorKind, Result};

pub(crate) const EBML: u32 = 0x1A45_DFA3;
pub(crate) const EBML_VERSION: u32 = 0x4286;
pub(crate) const EBML_READ_VERSION: u32 = 0x42F7;
pub(crate) const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
pub(crate) const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
pub(crate) const DOC_TYPE: u32 = 0x4282;
pub(crate) const DOC_TYPE_VERSION: u32 = 0x4287;
pub(crate) const DOC_TYPE_READ_VERSION: u32 = 0x4285;
pub(crate) const VOID: u32 = 0xEC;
pub(crate) const CRC32: u32 = 0xBF;

pub(crate) const SEGMENT: u32 = 0x1853_8067;
pub(crate) const SEEK_HEAD: u32 = 0x114D_9B74;
pub(crate) const SEEK: u32 = 0x4DBB;
pub(crate) const SEEK_ID: u32 = 0x53AB;
pub(crate) const SEEK_POSITION: u32 = 0x53AC;
pub(crate) const INFO: u32 = 0x1549_A966;
pub(crate) const TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
pub(crate) const DURATION: u32 = 0x4489;
pub(crate) const MUXING_APP: u32 = 0x4D80;
pub(crate) const WRITING_APP: u32 = 0x5741;
pub(crate) const TRACKS: u32 = 0x1654_AE6B;
pub(crate) const TRACK_ENTRY: u32 = 0xAE;
pub(crate) const TRACK_NUMBER: u32 = 0xD7;
pub(crate) const TRACK_UID: u32 = 0x73C5;
pub(crate) const TRACK_TYPE: u32 = 0x83;
pub(crate) const FLAG_LACING: u32 = 0x9C;
pub(crate) const CODEC_ID: u32 = 0x86;
pub(crate) const CODEC_PRIVATE: u32 = 0x63A2;
pub(crate) const DEFAULT_DURATION: u32 = 0x23_E383;
pub(crate) const CONTENT_ENCODINGS: u32 = 0x6D80;
pub(crate) const VIDEO: u32 = 0xE0;
pub(crate) const PIXEL_WIDTH: u32 = 0xB0;
pub(crate) const PIXEL_HEIGHT: u32 = 0xBA;
pub(crate) const CLUSTER: u32 = 0x1F43_B675;
pub(crate) const TIMESTAMP: u32 = 0xE7;
pub(crate) const SIMPLE_BLOCK: u32 = 0xA3;
pub(crate) const BLOCK_GROUP: u32 = 0xA0;
pub(crate) const BLOCK: u32 = 0xA1;
pub(crate) const BLOCK_DURATION: u32 = 0x9B;
pub(crate) const REFERENCE_BLOCK: u32 = 0xFB;
pub(crate) const CUES: u32 = 0x1C53_BB6B;
pub(crate) const CUE_POINT: u32 = 0xBB;
pub(crate) const CUE_TIME: u32 = 0xB3;
pub(crate) const CUE_TRACK_POSITIONS: u32 = 0xB7;
pub(crate) const CUE_TRACK: u32 = 0xF7;
pub(crate) const CUE_CLUSTER_POSITION: u32 = 0xF1;
pub(crate) const CUE_RELATIVE_POSITION: u32 = 0xF0;
pub(crate) const TAGS: u32 = 0x1254_C367;
pub(crate) const CHAPTERS: u32 = 0x1043_A770;
pub(crate) const ATTACHMENTS: u32 = 0x1941_A469;

/// Matroska `TrackType` of a video track.
pub(crate) const TRACK_TYPE_VIDEO: u64 = 1;

/// Whether `id` is a child of a Segment. Meeting one inside an unknown-size
/// Cluster ends that Cluster, which is how a live recording such as a
/// `MediaRecorder` capture marks where each Cluster stops.
pub(crate) fn is_segment_child(id: u32) -> bool {
    matches!(
        id,
        SEEK_HEAD | INFO | TRACKS | CLUSTER | CUES | TAGS | CHAPTERS | ATTACHMENTS
    )
}

/// Decodes an element ID at the start of `bytes`, marker included, and
/// returns it with its length in bytes.
pub(crate) fn read_id(bytes: &[u8]) -> Result<(u32, usize)> {
    let first = *bytes
        .first()
        .ok_or_else(|| malformed("EBML element ID is truncated"))?;
    let length = first.leading_zeros() as usize + 1;
    if length > 4 {
        return Err(malformed("EBML element ID is longer than four bytes"));
    }
    let encoded = bytes
        .get(..length)
        .ok_or_else(|| malformed("EBML element ID is truncated"))?;
    let id = encoded
        .iter()
        .fold(0_u32, |value, &byte| (value << 8) | u32::from(byte));
    Ok((id, length))
}

/// Decodes a variable-size integer at the start of `bytes` with its length
/// marker removed. An all-ones value is `None`: the reserved "unknown size".
pub(crate) fn read_vint(bytes: &[u8]) -> Result<(Option<u64>, usize)> {
    let first = *bytes
        .first()
        .ok_or_else(|| malformed("EBML variable-size integer is truncated"))?;
    if first == 0 {
        return Err(malformed(
            "EBML variable-size integer is longer than eight bytes",
        ));
    }
    let length = first.leading_zeros() as usize + 1;
    let encoded = bytes
        .get(..length)
        .ok_or_else(|| malformed("EBML variable-size integer is truncated"))?;
    let marker_mask = if length == 8 { 0 } else { 0xFF_u8 >> length };
    let value = encoded[1..]
        .iter()
        .fold(u64::from(first & marker_mask), |value, &byte| {
            (value << 8) | u64::from(byte)
        });
    let all_ones = (1_u64 << (7 * length)) - 1;
    Ok(((value != all_ones).then_some(value), length))
}

/// Decodes a known (not reserved) variable-size integer, as a block's track
/// number or a lace size is.
pub(crate) fn read_known_vint(bytes: &[u8]) -> Result<(u64, usize)> {
    match read_vint(bytes)? {
        (Some(value), length) => Ok((value, length)),
        (None, _) => Err(malformed(
            "EBML variable-size integer uses the reserved unknown value",
        )),
    }
}

/// Decodes the signed variable-size integer EBML lacing stores size
/// differences in: the unsigned value minus half its range.
pub(crate) fn read_signed_vint(bytes: &[u8]) -> Result<(i64, usize)> {
    let (value, length) = read_vint(bytes)?;
    let value = value.ok_or_else(|| malformed("EBML lace size difference is reserved"))?;
    let bias = (1_i64 << (7 * length - 1)) - 1;
    Ok((value as i64 - bias, length))
}

/// Reads an EBML unsigned integer payload of zero to eight bytes.
pub(crate) fn read_uint(payload: &[u8]) -> Result<u64> {
    if payload.len() > 8 {
        return Err(malformed(
            "EBML unsigned integer is longer than eight bytes",
        ));
    }
    Ok(payload
        .iter()
        .fold(0_u64, |value, &byte| (value << 8) | u64::from(byte)))
}

/// Reads an EBML float payload of zero, four or eight bytes.
pub(crate) fn read_float(payload: &[u8]) -> Result<f64> {
    match payload.len() {
        0 => Ok(0.0),
        4 => Ok(f64::from(f32::from_be_bytes(
            payload.try_into().expect("four bytes"),
        ))),
        8 => Ok(f64::from_be_bytes(payload.try_into().expect("eight bytes"))),
        _ => Err(malformed("EBML float is not four or eight bytes")),
    }
}

/// Reads an EBML string payload, dropping the zero padding EBML permits.
pub(crate) fn read_string(payload: &[u8]) -> Result<String> {
    let end = payload
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(payload.len());
    String::from_utf8(payload[..end].to_vec())
        .map_err(|_| malformed("EBML string is not valid UTF-8"))
}

/// Iterates the children of an in-memory master element payload as
/// `(id, payload)` pairs. Unknown sizes are only meaningful while streaming a
/// Segment or Cluster, so they are rejected here.
pub(crate) fn children(payload: &[u8]) -> Children<'_> {
    Children { payload, cursor: 0 }
}

pub(crate) struct Children<'a> {
    payload: &'a [u8],
    cursor: usize,
}

impl<'a> Iterator for Children<'a> {
    type Item = Result<(u32, &'a [u8])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.payload.len() {
            return None;
        }
        let item = (|| {
            let rest = &self.payload[self.cursor..];
            let (id, id_length) = read_id(rest)?;
            let (size, size_length) = read_vint(&rest[id_length..])?;
            let size = size.ok_or_else(|| {
                malformed("only a Segment or Cluster may have an unknown EBML size")
            })?;
            let start = id_length + size_length;
            let end = usize::try_from(size)
                .ok()
                .and_then(|size| start.checked_add(size))
                .filter(|&end| end <= rest.len())
                .ok_or_else(|| malformed("EBML element exceeds its parent"))?;
            self.cursor += end;
            Ok((id, &rest[start..end]))
        })();
        if item.is_err() {
            self.cursor = self.payload.len();
        }
        Some(item)
    }
}

/// Appends an element ID in its own minimal length.
pub(crate) fn write_id(output: &mut Vec<u8>, id: u32) {
    let length = match id {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFF_FFFF => 3,
        _ => 4,
    };
    output.extend_from_slice(&id.to_be_bytes()[4 - length..]);
}

/// Appends `value` as the shortest variable-size integer that does not collide
/// with the reserved all-ones value of its length.
pub(crate) fn write_vint(output: &mut Vec<u8>, value: u64) {
    let length = (1..=8)
        .find(|&length| value < (1_u64 << (7 * length)) - 1)
        .unwrap_or(8);
    write_vint_fixed(output, value, length);
}

/// Appends `value` as a variable-size integer of exactly `length` bytes, so a
/// size can be rewritten in place once it is known.
pub(crate) fn write_vint_fixed(output: &mut Vec<u8>, value: u64, length: usize) {
    debug_assert!((1..=8).contains(&length));
    debug_assert!(value < (1_u64 << (7 * length)) - 1);
    let marked = value | (1_u64 << (7 * length));
    output.extend_from_slice(&marked.to_be_bytes()[8 - length..]);
}

/// Appends a complete element with a minimal size.
pub(crate) fn write_element(output: &mut Vec<u8>, id: u32, payload: &[u8]) {
    write_id(output, id);
    write_vint(output, payload.len() as u64);
    output.extend_from_slice(payload);
}

/// Appends an unsigned integer element in its minimal byte length.
pub(crate) fn write_uint(output: &mut Vec<u8>, id: u32, value: u64) {
    let length = (8 - value.leading_zeros() as usize / 8).max(1);
    write_element(output, id, &value.to_be_bytes()[8 - length..]);
}

/// Appends an unsigned integer element that always occupies eight bytes, so
/// it can be rewritten in place.
pub(crate) fn write_uint_fixed(output: &mut Vec<u8>, id: u32, value: u64) {
    write_element(output, id, &value.to_be_bytes());
}

/// Appends an eight-byte float element.
pub(crate) fn write_float(output: &mut Vec<u8>, id: u32, value: f64) {
    write_element(output, id, &value.to_be_bytes());
}

/// Appends a string element.
pub(crate) fn write_string(output: &mut Vec<u8>, id: u32, value: &str) {
    write_element(output, id, value.as_bytes());
}

fn malformed(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_keep_their_marker_and_length() {
        assert_eq!(read_id(&[0xA3, 0]).unwrap(), (SIMPLE_BLOCK, 1));
        assert_eq!(read_id(&[0x1A, 0x45, 0xDF, 0xA3]).unwrap(), (EBML, 4));
        assert!(read_id(&[0x08, 0, 0, 0, 0]).is_err());
        assert!(read_id(&[0x1A, 0x45]).is_err());
        let mut written = Vec::new();
        write_id(&mut written, TIMESTAMP_SCALE);
        assert_eq!(written, [0x2A, 0xD7, 0xB1]);
    }

    #[test]
    fn vints_round_trip_and_reserve_all_ones_for_unknown() {
        for value in [0_u64, 1, 126, 127, 128, 16_382, 16_383, 1 << 40] {
            let mut written = Vec::new();
            write_vint(&mut written, value);
            assert_eq!(read_vint(&written).unwrap(), (Some(value), written.len()));
        }
        // 127 is all ones in one byte, so it needs two.
        let mut written = Vec::new();
        write_vint(&mut written, 127);
        assert_eq!(written, [0x40, 0x7F]);
        assert_eq!(read_vint(&[0xFF]).unwrap(), (None, 1));
        assert_eq!(
            read_vint(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]).unwrap(),
            (None, 8)
        );
        assert!(read_vint(&[0x00]).is_err());
        assert!(read_known_vint(&[0xFF]).is_err());
        let mut fixed = Vec::new();
        write_vint_fixed(&mut fixed, 5, 8);
        assert_eq!(fixed, [0x01, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(read_vint(&fixed).unwrap(), (Some(5), 8));
    }

    #[test]
    fn signed_vints_are_biased_by_half_their_range() {
        assert_eq!(read_signed_vint(&[0xBF]).unwrap(), (0, 1));
        assert_eq!(read_signed_vint(&[0x80]).unwrap(), (-63, 1));
        assert_eq!(read_signed_vint(&[0x5F, 0xFF]).unwrap(), (0, 2));
    }

    #[test]
    fn scalars_decode_and_reject_bad_lengths() {
        assert_eq!(read_uint(&[]).unwrap(), 0);
        assert_eq!(read_uint(&[0x0F, 0x42, 0x40]).unwrap(), 1_000_000);
        assert!(read_uint(&[0; 9]).is_err());
        assert_eq!(read_float(&1.5_f32.to_be_bytes()).unwrap(), 1.5);
        assert_eq!(read_float(&2.25_f64.to_be_bytes()).unwrap(), 2.25);
        assert!(read_float(&[0; 3]).is_err());
        assert_eq!(read_string(b"webm\0\0").unwrap(), "webm");
        let mut written = Vec::new();
        write_uint(&mut written, TRACK_NUMBER, 0);
        assert_eq!(written, [0xD7, 0x81, 0x00]);
    }

    #[test]
    fn children_walk_a_payload_and_reject_overruns() {
        let mut payload = Vec::new();
        write_uint(&mut payload, TRACK_NUMBER, 1);
        write_string(&mut payload, CODEC_ID, "V_AV1");
        let parsed: Vec<_> = children(&payload).collect::<Result<_>>().unwrap();
        assert_eq!(
            parsed,
            vec![(TRACK_NUMBER, &[1_u8][..]), (CODEC_ID, &b"V_AV1"[..])]
        );
        assert!(children(&[0xD7, 0x85, 1]).next().unwrap().is_err());
        assert!(children(&[0xD7, 0xFF]).next().unwrap().is_err());
    }
}
