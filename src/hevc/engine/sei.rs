//! H.265 / HEVC Supplemental Enhancement Information (SEI) parse.
//!
//! This module implements the SEI framing of ITU-T Rec. H.265
//! (ISO/IEC 23008-2):
//!
//! * **§7.3.2.4** — `sei_rbsp()`: a `do … while( more_rbsp_data() )`
//!   loop of `sei_message()` structures followed by
//!   `rbsp_trailing_bits()`.
//! * **§7.3.5** — `sei_message()`: the extensible `payloadType` /
//!   `payloadSize` framing where each `0xFF` byte adds 255 and the
//!   first non-`0xFF` byte is the final additive term.
//! * **§D.2** — `sei_payload( payloadType, payloadSize )`: the
//!   payload-type dispatch. The `nal_unit_type` (PREFIX_SEI_NUT vs
//!   SUFFIX_SEI_NUT) selects which payload types are legal at a given
//!   position; each message body is delimited to exactly `payloadSize`
//!   bytes.
//!
//! Because §7.3.5 makes every message length-prefixed in whole bytes,
//! the framing layer is robust against a payload body it does not
//! decode: an unknown or not-yet-implemented `payloadType` is surfaced
//! as [`SeiPayload::Reserved`] carrying the verbatim `payloadSize`
//! bytes. The framing therefore always advances by `payloadSize` bytes
//! per message regardless of which payload bodies are typed.
//!
//! The byte-aligned payload trailer (§D.3.1 `more_data_in_payload()`:
//! the `payload_bit_equal_to_one` + zero-padding to a byte boundary
//! that may follow a sub-byte payload body) is the responsibility of
//! the individual payload parser when it reads a non-byte-aligned body;
//! the framing layer only needs the byte count.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};

/// The SEI NAL-unit type, which selects the §D.2 payload dispatch
/// branch. Per §7.4.2.2 (Table 7-1), `nal_unit_type == 39` is
/// `PREFIX_SEI_NUT` and `nal_unit_type == 40` is `SUFFIX_SEI_NUT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeiNalType {
    /// `PREFIX_SEI_NUT` (`nal_unit_type == 39`). Prefix SEI precede the
    /// associated coded picture in decoding order.
    Prefix,
    /// `SUFFIX_SEI_NUT` (`nal_unit_type == 40`). Suffix SEI follow the
    /// associated coded picture in decoding order; the §D.2 dispatch
    /// admits a smaller set of payload types here.
    Suffix,
}

/// `nal_unit_type` value for `PREFIX_SEI_NUT` (§7.4.2.2, Table 7-1).
pub const PREFIX_SEI_NUT: u8 = 39;
/// `nal_unit_type` value for `SUFFIX_SEI_NUT` (§7.4.2.2, Table 7-1).
pub const SUFFIX_SEI_NUT: u8 = 40;

impl SeiNalType {
    /// Map a raw `nal_unit_type` to its SEI category, or `None` if the
    /// value is neither of the two SEI NAL-unit types.
    pub fn from_nal_unit_type(nal_unit_type: u8) -> Option<Self> {
        match nal_unit_type {
            PREFIX_SEI_NUT => Some(Self::Prefix),
            SUFFIX_SEI_NUT => Some(Self::Suffix),
            _ => None,
        }
    }
}

/// Errors that can arise while parsing an SEI RBSP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeiError {
    /// The declared `payloadSize` extended past the end of the RBSP
    /// (the message claimed more bytes than remain in the buffer).
    PayloadSizeOverrun {
        /// The decoded §7.3.5 `payloadSize`.
        payload_size: usize,
        /// The number of bytes remaining in the RBSP at the start of
        /// the payload.
        available: usize,
    },
    /// The RBSP ended in the middle of the §7.3.5 `payloadType` /
    /// `payloadSize` extensible byte run (no terminating non-`0xFF`
    /// byte before end-of-buffer).
    TruncatedHeader,
    /// A typed payload body was shorter than its fixed §D.2 layout
    /// requires (the `payloadSize` was smaller than the syntax demands).
    TruncatedPayload {
        /// The §D.2 `payloadType` whose body was short.
        payload_type: u32,
    },
    /// A bit-level read inside a typed payload body failed.
    Bit(BitReaderError),
}

impl From<BitReaderError> for SeiError {
    fn from(e: BitReaderError) -> Self {
        Self::Bit(e)
    }
}

impl core::fmt::Display for SeiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PayloadSizeOverrun {
                payload_size,
                available,
            } => write!(
                f,
                "SEI payloadSize {payload_size} exceeds {available} remaining RBSP bytes"
            ),
            Self::TruncatedHeader => f.write_str("SEI message header truncated"),
            Self::TruncatedPayload { payload_type } => {
                write!(f, "SEI payload type {payload_type} body truncated")
            }
            Self::Bit(e) => write!(f, "SEI bit read error: {e}"),
        }
    }
}

impl std::error::Error for SeiError {}

/// The §D.2 `recovery_point` SEI message (`payloadType == 6`): a
/// recovery-point indication for random access (§D.3.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryPoint {
    /// `recovery_poc_cnt` (`se(v)`): the picture-order-count offset of
    /// the recovery point relative to the current picture.
    pub recovery_poc_cnt: i32,
    /// `exact_match_flag` (`u(1)`).
    pub exact_match_flag: bool,
    /// `broken_link_flag` (`u(1)`).
    pub broken_link_flag: bool,
}

/// The §D.2 `active_parameter_sets` SEI message (`payloadType == 129`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveParameterSets {
    /// `active_video_parameter_set_id` (`u(4)`).
    pub active_video_parameter_set_id: u8,
    /// `self_contained_cvs_flag` (`u(1)`).
    pub self_contained_cvs_flag: bool,
    /// `no_parameter_set_update_flag` (`u(1)`).
    pub no_parameter_set_update_flag: bool,
    /// `active_seq_parameter_set_id[ i ]` for
    /// `i = 0 .. num_sps_ids_minus1` (`ue(v)` each).
    pub active_seq_parameter_set_ids: Vec<u32>,
}

/// The §D.2 `content_light_level_info` SEI message
/// (`payloadType == 144`): HDR content light-level metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLightLevelInfo {
    /// `max_content_light_level` (`u(16)`), in candelas per square
    /// metre (0 ⇒ unknown).
    pub max_content_light_level: u16,
    /// `max_pic_average_light_level` (`u(16)`), in candelas per square
    /// metre (0 ⇒ unknown).
    pub max_pic_average_light_level: u16,
}

/// The §D.2 `mastering_display_colour_volume` SEI message
/// (`payloadType == 137`): HDR mastering-display metadata (SMPTE
/// ST 2086 carriage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplayColourVolume {
    /// `display_primaries_x[ c ]` for the three colour primaries in the
    /// G, B, R order in which they appear on the wire (`u(16)` each).
    pub display_primaries_x: [u16; 3],
    /// `display_primaries_y[ c ]` (`u(16)` each).
    pub display_primaries_y: [u16; 3],
    /// `white_point_x` (`u(16)`).
    pub white_point_x: u16,
    /// `white_point_y` (`u(16)`).
    pub white_point_y: u16,
    /// `max_display_mastering_luminance` (`u(32)`).
    pub max_display_mastering_luminance: u32,
    /// `min_display_mastering_luminance` (`u(32)`).
    pub min_display_mastering_luminance: u32,
}

/// The §D.2 `alternative_transfer_characteristics` SEI message
/// (`payloadType == 147`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlternativeTransferCharacteristics {
    /// `preferred_transfer_characteristics` (`u(8)`): a code point in
    /// the same value space as the VUI `transfer_characteristics`.
    pub preferred_transfer_characteristics: u8,
}

/// The §D.2 `decoded_picture_hash` SEI message (`payloadType == 132`,
/// suffix-only): a per-component hash of the decoded picture used for
/// decoder conformance checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPictureHash {
    /// `hash_type` (`u(8)`): 0 ⇒ MD5, 1 ⇒ CRC, 2 ⇒ checksum.
    pub hash_type: u8,
    /// The per-component hash payload, one entry per colour component
    /// in the order they appear on the wire. The interpretation
    /// depends on `hash_type`.
    pub component_hashes: Vec<PictureHash>,
}

/// One component's decoded-picture hash value (§D.2 `decoded_picture_hash`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PictureHash {
    /// `hash_type == 0`: the 16-byte `picture_md5[ cIdx ][ i ]` digest.
    Md5([u8; 16]),
    /// `hash_type == 1`: the `picture_crc[ cIdx ]` (`u(16)`).
    Crc(u16),
    /// `hash_type == 2`: the `picture_checksum[ cIdx ]` (`u(32)`).
    Checksum(u32),
}

/// The §D.2 `user_data_unregistered` SEI message (`payloadType == 5`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDataUnregistered {
    /// `uuid_iso_iec_11578` (`u(128)`): the 16-byte UUID, stored in
    /// wire order (most-significant byte first).
    pub uuid_iso_iec_11578: [u8; 16],
    /// The remaining `user_data_payload_byte` bytes
    /// (`payloadSize - 16`).
    pub user_data_payload: Vec<u8>,
}

/// The §D.2 `user_data_registered_itu_t_t35` SEI message
/// (`payloadType == 4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDataRegisteredItuTT35 {
    /// `itu_t_t35_country_code` (`b(8)`).
    pub country_code: u8,
    /// `itu_t_t35_country_code_extension_byte` (`b(8)`), present only
    /// when `country_code == 0xFF` per §D.2.6.
    pub country_code_extension: Option<u8>,
    /// The remaining `itu_t_t35_payload_byte` bytes.
    pub payload: Vec<u8>,
}

/// A decoded SEI payload body, or the verbatim bytes of a payload type
/// this module does not yet type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeiPayload {
    /// `payloadType == 4`.
    UserDataRegisteredItuTT35(UserDataRegisteredItuTT35),
    /// `payloadType == 5`.
    UserDataUnregistered(UserDataUnregistered),
    /// `payloadType == 6`.
    RecoveryPoint(RecoveryPoint),
    /// `payloadType == 129`.
    ActiveParameterSets(ActiveParameterSets),
    /// `payloadType == 132` (suffix SEI).
    DecodedPictureHash(DecodedPictureHash),
    /// `payloadType == 137`.
    MasteringDisplayColourVolume(MasteringDisplayColourVolume),
    /// `payloadType == 144`.
    ContentLightLevelInfo(ContentLightLevelInfo),
    /// `payloadType == 147`.
    AlternativeTransferCharacteristics(AlternativeTransferCharacteristics),
    /// Any payload type this module does not decode (or a §D.2
    /// `reserved_sei_message`): the verbatim `payloadSize` bytes.
    Reserved {
        /// The §D.2 `payloadType`.
        payload_type: u32,
        /// The verbatim payload bytes (`payloadSize` of them).
        data: Vec<u8>,
    },
}

/// One parsed `sei_message()` (§7.3.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiMessage {
    /// The §7.3.5 `payloadType`.
    pub payload_type: u32,
    /// The §7.3.5 `payloadSize`, in bytes.
    pub payload_size: usize,
    /// The decoded (or verbatim) payload body.
    pub payload: SeiPayload,
}

/// Decode the extensible §7.3.5 `payloadType` / `payloadSize` byte run:
/// every `0xFF` byte contributes 255 and the first non-`0xFF` byte is
/// the final additive term. `cursor` is advanced past the consumed
/// bytes. Returns the accumulated value, or [`SeiError::TruncatedHeader`]
/// when the run is not terminated before end-of-buffer.
fn read_extensible_byte_run(rbsp: &[u8], cursor: &mut usize) -> Result<u32, SeiError> {
    let mut value: u32 = 0;
    loop {
        let byte = *rbsp.get(*cursor).ok_or(SeiError::TruncatedHeader)?;
        *cursor += 1;
        if byte == 0xFF {
            value = value.saturating_add(255);
        } else {
            value = value.saturating_add(byte as u32);
            return Ok(value);
        }
    }
}

impl SeiMessage {
    /// Parse one `sei_message()` (§7.3.5) starting at `cursor` within
    /// `rbsp`, dispatching the payload per §D.2 against `nal_type`.
    /// `cursor` is advanced to the byte immediately after this
    /// message's payload (i.e. past `payloadSize` bytes).
    pub fn parse(rbsp: &[u8], cursor: &mut usize, nal_type: SeiNalType) -> Result<Self, SeiError> {
        let payload_type = read_extensible_byte_run(rbsp, cursor)?;
        let payload_size = read_extensible_byte_run(rbsp, cursor)? as usize;

        let available = rbsp.len().saturating_sub(*cursor);
        if payload_size > available {
            return Err(SeiError::PayloadSizeOverrun {
                payload_size,
                available,
            });
        }
        let body = &rbsp[*cursor..*cursor + payload_size];
        *cursor += payload_size;

        let payload = decode_payload(payload_type, body, nal_type)?;
        Ok(Self {
            payload_type,
            payload_size,
            payload,
        })
    }
}

/// Dispatch a single SEI payload body (already delimited to
/// `payloadSize` bytes) per §D.2. Payload types this module does not
/// type are surfaced as [`SeiPayload::Reserved`].
fn decode_payload(
    payload_type: u32,
    body: &[u8],
    nal_type: SeiNalType,
) -> Result<SeiPayload, SeiError> {
    // The §D.2 dispatch is split by nal_unit_type. The suffix branch
    // admits only a subset of payload types; payload 132
    // (decoded_picture_hash) is suffix-only and 129 / 137 / 144 / 147 /
    // 6 are prefix-only. Types legal in both branches (4 / 5) decode
    // identically. A payload type that is not legal in the active
    // branch is treated as reserved (carried verbatim).
    match (nal_type, payload_type) {
        (_, 4) => Ok(SeiPayload::UserDataRegisteredItuTT35(
            decode_user_data_registered(body),
        )),
        (_, 5) => Ok(SeiPayload::UserDataUnregistered(
            decode_user_data_unregistered(body)?,
        )),
        (SeiNalType::Prefix, 6) => Ok(SeiPayload::RecoveryPoint(decode_recovery_point(body)?)),
        (SeiNalType::Prefix, 129) => Ok(SeiPayload::ActiveParameterSets(
            decode_active_parameter_sets(body)?,
        )),
        (SeiNalType::Suffix, 132) => Ok(SeiPayload::DecodedPictureHash(
            decode_decoded_picture_hash(body)?,
        )),
        (SeiNalType::Prefix, 137) => Ok(SeiPayload::MasteringDisplayColourVolume(
            decode_mastering_display_colour_volume(body)?,
        )),
        (SeiNalType::Prefix, 144) => Ok(SeiPayload::ContentLightLevelInfo(
            decode_content_light_level_info(body)?,
        )),
        (SeiNalType::Prefix, 147) => Ok(SeiPayload::AlternativeTransferCharacteristics(
            decode_alternative_transfer_characteristics(body)?,
        )),
        _ => Ok(SeiPayload::Reserved {
            payload_type,
            data: body.to_vec(),
        }),
    }
}

fn decode_recovery_point(body: &[u8]) -> Result<RecoveryPoint, SeiError> {
    let mut r = BitReader::new(body);
    let recovery_poc_cnt = r
        .se()
        .map_err(|_| SeiError::TruncatedPayload { payload_type: 6 })?;
    let exact_match_flag = r
        .u1()
        .map_err(|_| SeiError::TruncatedPayload { payload_type: 6 })?
        != 0;
    let broken_link_flag = r
        .u1()
        .map_err(|_| SeiError::TruncatedPayload { payload_type: 6 })?
        != 0;
    Ok(RecoveryPoint {
        recovery_poc_cnt,
        exact_match_flag,
        broken_link_flag,
    })
}

fn decode_active_parameter_sets(body: &[u8]) -> Result<ActiveParameterSets, SeiError> {
    let mut r = BitReader::new(body);
    let err = || SeiError::TruncatedPayload { payload_type: 129 };
    let active_video_parameter_set_id = r.u(4).map_err(|_| err())? as u8;
    let self_contained_cvs_flag = r.u1().map_err(|_| err())? != 0;
    let no_parameter_set_update_flag = r.u1().map_err(|_| err())? != 0;
    let num_sps_ids_minus1 = r.ue().map_err(|_| err())?;
    let count = (num_sps_ids_minus1 as usize)
        .checked_add(1)
        .ok_or_else(err)?;
    let mut active_seq_parameter_set_ids = Vec::with_capacity(count);
    for _ in 0..count {
        active_seq_parameter_set_ids.push(r.ue().map_err(|_| err())?);
    }
    // The trailing `layer_sps_idx[ i ]` loop runs only for the
    // multilayer (Annex F) extension (`MaxLayersMinus1 > 0`); for the
    // single-layer base stream it is empty. Those entries, when
    // present, fall into the byte-aligned payload trailer and are not
    // surfaced here.
    Ok(ActiveParameterSets {
        active_video_parameter_set_id,
        self_contained_cvs_flag,
        no_parameter_set_update_flag,
        active_seq_parameter_set_ids,
    })
}

fn decode_content_light_level_info(body: &[u8]) -> Result<ContentLightLevelInfo, SeiError> {
    if body.len() < 4 {
        return Err(SeiError::TruncatedPayload { payload_type: 144 });
    }
    Ok(ContentLightLevelInfo {
        max_content_light_level: u16::from_be_bytes([body[0], body[1]]),
        max_pic_average_light_level: u16::from_be_bytes([body[2], body[3]]),
    })
}

fn decode_mastering_display_colour_volume(
    body: &[u8],
) -> Result<MasteringDisplayColourVolume, SeiError> {
    // 3 * (u16 + u16) + u16 + u16 + u32 + u32 = 24 bytes.
    if body.len() < 24 {
        return Err(SeiError::TruncatedPayload { payload_type: 137 });
    }
    let mut display_primaries_x = [0u16; 3];
    let mut display_primaries_y = [0u16; 3];
    let mut off = 0usize;
    for c in 0..3 {
        display_primaries_x[c] = u16::from_be_bytes([body[off], body[off + 1]]);
        display_primaries_y[c] = u16::from_be_bytes([body[off + 2], body[off + 3]]);
        off += 4;
    }
    let white_point_x = u16::from_be_bytes([body[off], body[off + 1]]);
    let white_point_y = u16::from_be_bytes([body[off + 2], body[off + 3]]);
    off += 4;
    let max_display_mastering_luminance =
        u32::from_be_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
    off += 4;
    let min_display_mastering_luminance =
        u32::from_be_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
    Ok(MasteringDisplayColourVolume {
        display_primaries_x,
        display_primaries_y,
        white_point_x,
        white_point_y,
        max_display_mastering_luminance,
        min_display_mastering_luminance,
    })
}

fn decode_alternative_transfer_characteristics(
    body: &[u8],
) -> Result<AlternativeTransferCharacteristics, SeiError> {
    let preferred_transfer_characteristics = *body
        .first()
        .ok_or(SeiError::TruncatedPayload { payload_type: 147 })?;
    Ok(AlternativeTransferCharacteristics {
        preferred_transfer_characteristics,
    })
}

fn decode_decoded_picture_hash(body: &[u8]) -> Result<DecodedPictureHash, SeiError> {
    let err = || SeiError::TruncatedPayload { payload_type: 132 };
    let hash_type = *body.first().ok_or_else(err)?;
    let mut off = 1usize;
    let mut component_hashes = Vec::new();
    // The §D.2 loop bound is `chroma_format_idc == 0 ? 1 : 3`. Without
    // the activated SPS in hand, the body length disambiguates: each
    // component contributes a fixed number of bytes for the given
    // hash_type, so the count is `remaining / per_component`.
    let per_component = match hash_type {
        0 => 16,
        1 => 2,
        2 => 4,
        // Unknown hash_type: surface the raw tail as a single
        // checksum-sized read is not defined; treat the whole body as
        // reserved by returning the parsed hash_type with no
        // components (the framing already retained payload_size).
        _ => {
            return Ok(DecodedPictureHash {
                hash_type,
                component_hashes,
            });
        }
    };
    while off + per_component <= body.len() {
        let chunk = &body[off..off + per_component];
        off += per_component;
        let entry = match hash_type {
            0 => {
                let mut md5 = [0u8; 16];
                md5.copy_from_slice(chunk);
                PictureHash::Md5(md5)
            }
            1 => PictureHash::Crc(u16::from_be_bytes([chunk[0], chunk[1]])),
            2 => {
                PictureHash::Checksum(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
            _ => unreachable!(),
        };
        component_hashes.push(entry);
    }
    Ok(DecodedPictureHash {
        hash_type,
        component_hashes,
    })
}

fn decode_user_data_unregistered(body: &[u8]) -> Result<UserDataUnregistered, SeiError> {
    if body.len() < 16 {
        return Err(SeiError::TruncatedPayload { payload_type: 5 });
    }
    let mut uuid_iso_iec_11578 = [0u8; 16];
    uuid_iso_iec_11578.copy_from_slice(&body[..16]);
    Ok(UserDataUnregistered {
        uuid_iso_iec_11578,
        user_data_payload: body[16..].to_vec(),
    })
}

fn decode_user_data_registered(body: &[u8]) -> UserDataRegisteredItuTT35 {
    // §D.2.6: country_code is b(8); when 0xFF a second extension byte
    // follows. The remainder is itu_t_t35_payload_byte. An empty body
    // degenerates to a zero country code with no payload.
    let Some(&country_code) = body.first() else {
        return UserDataRegisteredItuTT35 {
            country_code: 0,
            country_code_extension: None,
            payload: Vec::new(),
        };
    };
    if country_code != 0xFF {
        UserDataRegisteredItuTT35 {
            country_code,
            country_code_extension: None,
            payload: body[1..].to_vec(),
        }
    } else {
        let country_code_extension = body.get(1).copied();
        let payload_start = if country_code_extension.is_some() {
            2
        } else {
            1
        };
        UserDataRegisteredItuTT35 {
            country_code,
            country_code_extension,
            payload: body.get(payload_start..).unwrap_or(&[]).to_vec(),
        }
    }
}

/// Parse a full `sei_rbsp()` (§7.3.2.4): a sequence of `sei_message()`
/// structures up to the `rbsp_trailing_bits()`. The `rbsp` is the
/// (already emulation-prevention-stripped) NAL payload **excluding the
/// two-byte NAL header**, and `nal_type` is the SEI category derived
/// from the NAL header's `nal_unit_type`.
///
/// Per §7.3.5 every message is byte-length-prefixed, so the trailer is
/// recognised positionally: parsing stops once the only bytes that
/// remain are the `rbsp_trailing_bits()` (a single `0x80` byte, or a
/// `0x80` followed by zero padding). A buffer of all-zero / empty tail
/// is also treated as the end of messages.
pub fn parse_sei_rbsp(rbsp: &[u8], nal_type: SeiNalType) -> Result<Vec<SeiMessage>, SeiError> {
    let mut messages = Vec::new();
    let mut cursor = 0usize;
    loop {
        if !more_sei_messages(rbsp, cursor) {
            break;
        }
        let msg = SeiMessage::parse(rbsp, &mut cursor, nal_type)?;
        messages.push(msg);
    }
    Ok(messages)
}

/// `more_rbsp_data()` for the SEI message loop (§7.3.2.4): another
/// `sei_message()` follows iff at least one bit set to 1 remains
/// before the `rbsp_trailing_bits()` `0x80` stop bit. Operating on
/// whole bytes (every message is byte-aligned), this reduces to "the
/// remaining bytes are not exactly the trailing-bits structure."
fn more_sei_messages(rbsp: &[u8], cursor: usize) -> bool {
    let rest = &rbsp[cursor.min(rbsp.len())..];
    if rest.is_empty() {
        return false;
    }
    // rbsp_trailing_bits() is `rbsp_stop_one_bit` (0x80 when
    // byte-aligned) followed by zero `rbsp_alignment_zero_bit`s. If the
    // first remaining byte is 0x80 and every following byte is 0x00,
    // the rest is purely the trailing-bits structure: no more messages.
    if rest[0] == 0x80 && rest[1..].iter().all(|&b| b == 0) {
        return false;
    }
    // An all-zero tail (no stop bit reached, e.g. a degenerate buffer)
    // carries no further sei_message() and is treated as the end.
    if rest.iter().all(|&b| b == 0) {
        return false;
    }
    true
}
