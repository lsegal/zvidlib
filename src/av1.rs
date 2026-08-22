//! Bounded AV1 low-overhead bitstream and configuration parsing.
//!
//! This module intentionally provides syntax/state only. It does not implement
//! pixel reconstruction and is not registered as a [`VideoDecoderFactory`](crate::VideoDecoderFactory).

use crate::{Error, ErrorKind, Limits, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Av1SyntaxSupport {
    MainProfile,
    UnsupportedProfile(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Av1ObuType {
    Reserved = 0,
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
    RedundantFrameHeader = 7,
    TileList = 8,
    Padding = 15,
}

impl Av1ObuType {
    fn from_bits(value: u8) -> Self {
        match value {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            _ => Self::Reserved,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Av1ObuHeader {
    pub obu_type: Av1ObuType,
    pub extension_flag: bool,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub header_len: u8,
    pub size_field_len: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1OperatingPoint {
    pub idc: u16,
    pub level: u8,
    pub tier: bool,
    pub decoder_model_present: bool,
    pub initial_display_delay: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1ColorConfig {
    pub bit_depth: u8,
    pub monochrome: bool,
    pub color_description: Option<(u8, u8, u8)>,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1SequenceHeader {
    pub seq_profile: u8,
    pub support: Av1SyntaxSupport,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub operating_points: Vec<Av1OperatingPoint>,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length: u8,
    pub additional_frame_id_length: u8,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub order_hint_bits: u8,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_config: Av1ColorConfig,
    pub film_grain_params_present: bool,
    pub decoder_model_info_present: bool,
    pub equal_picture_interval: bool,
    pub buffer_delay_length: u8,
    pub buffer_removal_time_length: u8,
    pub frame_presentation_time_length: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Av1FrameType {
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: Option<u8>,
    pub frame_type: Av1FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: Option<u32>,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: Option<u8>,
    pub refresh_frame_flags: u8,
    /// Number of header bits consumed before syntax owned by reconstruction stages.
    pub parsed_bits: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1TileGroup {
    pub tile_start: u32,
    pub tile_end: u32,
    pub data: Vec<u8>,
    /// Byte ranges for each tile, relative to `data`.
    pub tile_ranges: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1Metadata {
    pub metadata_type: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Av1Obu {
    SequenceHeader {
        header: Av1ObuHeader,
        sequence: Av1SequenceHeader,
    },
    TemporalDelimiter {
        header: Av1ObuHeader,
    },
    FrameHeader {
        header: Av1ObuHeader,
        frame: Av1FrameHeader,
        payload: Vec<u8>,
    },
    TileGroup {
        header: Av1ObuHeader,
        group: Av1TileGroup,
    },
    Metadata {
        header: Av1ObuHeader,
        metadata: Av1Metadata,
    },
    Frame {
        header: Av1ObuHeader,
        frame: Av1FrameHeader,
        payload: Vec<u8>,
    },
    Skipped {
        header: Av1ObuHeader,
        payload_len: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1CodecConfigurationRecord {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: bool,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub initial_presentation_delay: Option<u8>,
    pub config_obus: Vec<Av1Obu>,
    pub support: Av1SyntaxSupport,
}

impl Av1CodecConfigurationRecord {
    /// Parses a complete `av1C` box, including its 32-bit size and fourcc.
    pub fn parse(bytes: &[u8], limits: &Limits) -> Result<Self> {
        check_allocation(bytes.len(), limits, "av1C box")?;
        if bytes.len() < 12 {
            return malformed("av1C box is truncated");
        }
        let size32 = u32::from_be_bytes(bytes[0..4].try_into().expect("four bytes"));
        let (declared, payload_offset) = if size32 == 1 {
            let large = bytes.get(8..16).ok_or_else(|| {
                Error::new(ErrorKind::MalformedMedia, "extended av1C box is truncated")
            })?;
            let declared = u64::from_be_bytes(large.try_into().expect("eight bytes"));
            let declared = usize::try_from(declared)
                .map_err(|_| resource("av1C box size does not fit this platform"))?;
            (declared, 16)
        } else {
            let declared = usize::try_from(size32)
                .map_err(|_| resource("av1C box size does not fit this platform"))?;
            (declared, 8)
        };
        if declared != bytes.len() || &bytes[4..8] != b"av1C" {
            return malformed("av1C box size or fourcc is invalid");
        }
        let p = bytes
            .get(payload_offset..)
            .ok_or_else(|| Error::new(ErrorKind::MalformedMedia, "av1C record is truncated"))?;
        if p.len() < 4 {
            return malformed("av1C record is truncated");
        }
        if p[0] != 0x81 {
            return malformed("av1C marker or version is invalid");
        }
        if p[2] & 0b11 != 0 || p[3] & 0b1110_0000 != 0 {
            return malformed("av1C reserved bits are nonzero");
        }
        let seq_profile = p[1] >> 5;
        if seq_profile > 2 {
            return malformed("av1C sequence profile is reserved");
        }
        let delay = if p[3] & 0x10 != 0 {
            Some((p[3] & 0x0f) + 1)
        } else {
            None
        };
        let mut parser = Av1Parser::new(*limits)?;
        let config_obus = parser.parse_low_overhead(&p[4..])?;
        for obu in &config_obus {
            if !matches!(obu, Av1Obu::SequenceHeader { .. } | Av1Obu::Metadata { .. }) {
                return malformed("av1C configOBUs contain a disallowed OBU type");
            }
        }
        if let Some(Av1Obu::SequenceHeader { sequence, .. }) = config_obus
            .iter()
            .find(|obu| matches!(obu, Av1Obu::SequenceHeader { .. }))
        {
            let c = &sequence.color_config;
            let expected_high = c.bit_depth > 8;
            let expected_twelve = c.bit_depth == 12;
            if sequence.seq_profile != seq_profile
                || expected_high != (p[2] & 0x40 != 0)
                || expected_twelve != (p[2] & 0x20 != 0)
                || c.monochrome != (p[2] & 0x10 != 0)
                || c.subsampling_x != (p[2] & 0x08 != 0)
                || c.subsampling_y != (p[2] & 0x04 != 0)
                || c.chroma_sample_position != p[2] & 0x03
            {
                return malformed("av1C fields disagree with its sequence header OBU");
            }
        }
        Ok(Self {
            seq_profile,
            seq_level_idx_0: p[1] & 0x1f,
            seq_tier_0: p[2] & 0x80 != 0,
            high_bitdepth: p[2] & 0x40 != 0,
            twelve_bit: p[2] & 0x20 != 0,
            monochrome: p[2] & 0x10 != 0,
            chroma_subsampling_x: p[2] & 0x08 != 0,
            chroma_subsampling_y: p[2] & 0x04 != 0,
            chroma_sample_position: p[2] & 0x03,
            initial_presentation_delay: delay,
            config_obus,
            support: support_for_profile(seq_profile),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Av1Parser {
    limits: Limits,
    pub sequence: Option<Av1SequenceHeader>,
    pub frame: Option<Av1FrameHeader>,
    seen_frame_header: bool,
}

impl Av1Parser {
    pub fn new(limits: Limits) -> Result<Self> {
        if limits.max_av1_obus_per_unit == 0
            || limits.max_av1_operating_points == 0
            || limits.max_av1_operating_points > 32
            || limits.max_av1_tiles_per_frame == 0
        {
            return Err(resource("AV1 parser limits must be nonzero"));
        }
        Ok(Self {
            limits,
            sequence: None,
            frame: None,
            seen_frame_header: false,
        })
    }

    /// Parses one low-overhead AV1 byte stream. Every OBU must carry a leb128 size.
    pub fn parse_low_overhead(&mut self, bytes: &[u8]) -> Result<Vec<Av1Obu>> {
        check_allocation(bytes.len(), &self.limits, "AV1 input")?;
        let capacity = bytes.len().min(self.limits.max_av1_obus_per_unit as usize);
        let mut obus = Vec::with_capacity(capacity);
        let mut offset = 0usize;
        while offset < bytes.len() {
            if obus.len() >= self.limits.max_av1_obus_per_unit as usize {
                return Err(resource("AV1 OBU count exceeds configured limit"));
            }
            let (header, header_bytes) = parse_obu_header(&bytes[offset..])?;
            offset = offset
                .checked_add(header_bytes)
                .ok_or_else(|| resource("AV1 offset overflow"))?;
            let (payload_size, size_len) = parse_leb128(&bytes[offset..])?;
            offset = offset
                .checked_add(size_len)
                .ok_or_else(|| resource("AV1 offset overflow"))?;
            let payload_len = usize::try_from(payload_size)
                .map_err(|_| resource("AV1 OBU payload does not fit this platform"))?;
            check_allocation(payload_len, &self.limits, "AV1 OBU payload")?;
            let end = offset
                .checked_add(payload_len)
                .ok_or_else(|| resource("AV1 OBU end overflow"))?;
            let payload = bytes.get(offset..end).ok_or_else(|| {
                Error::new(ErrorKind::MalformedMedia, "AV1 OBU payload is truncated")
            })?;
            offset = end;
            let mut header = header;
            header.size_field_len = u8::try_from(size_len).expect("leb128 is at most eight bytes");
            obus.push(self.parse_obu(header, payload)?);
        }
        Ok(obus)
    }

    /// Parses a tile-group payload once the frame header's tile layout is known.
    pub fn parse_tile_group(
        payload: &[u8],
        tile_cols_log2: u8,
        tile_rows_log2: u8,
        tile_size_bytes: u8,
        limits: &Limits,
    ) -> Result<Av1TileGroup> {
        check_allocation(payload.len(), limits, "AV1 tile group")?;
        let tile_bits = tile_cols_log2
            .checked_add(tile_rows_log2)
            .ok_or_else(|| resource("AV1 tile bit count overflow"))?;
        if tile_bits >= 32 {
            return Err(resource("AV1 tile count is too large"));
        }
        let tile_count = 1u32
            .checked_shl(u32::from(tile_bits))
            .ok_or_else(|| resource("AV1 tile count overflow"))?;
        if tile_count == 0 || tile_count > limits.max_av1_tiles_per_frame {
            return Err(resource("AV1 tile count exceeds configured limit"));
        }
        if !(1..=4).contains(&tile_size_bytes) {
            return malformed("AV1 tile size byte count is invalid");
        }
        let mut bits = Bits::new(payload);
        let (tile_start, tile_end) = if tile_count == 1 {
            (0, 0)
        } else if bits.flag("tile_start_and_end_present_flag")? {
            (
                bits.u32(usize::from(tile_bits), "tg_start")?,
                bits.u32(usize::from(tile_bits), "tg_end")?,
            )
        } else {
            (0, tile_count - 1)
        };
        if tile_start > tile_end || tile_end >= tile_count {
            return malformed("AV1 tile group range is invalid");
        }
        while bits.position() & 7 != 0 {
            if bits.flag("tile group alignment bit")? {
                return malformed("AV1 tile group alignment bit is nonzero");
            }
        }
        let data_offset = bits.position() / 8;
        let data = payload
            .get(data_offset..)
            .ok_or_else(|| Error::new(ErrorKind::MalformedMedia, "AV1 tile group is truncated"))?
            .to_vec();
        let group_count = tile_end - tile_start + 1;
        let range_capacity = usize::try_from(group_count)
            .map_err(|_| resource("AV1 tile group count does not fit this platform"))?;
        let mut tile_ranges = Vec::new();
        tile_ranges
            .try_reserve(range_capacity)
            .map_err(|_| resource("AV1 tile range allocation failed"))?;
        let mut cursor = 0usize;
        for index in 0..group_count {
            let tile_len = if index + 1 == group_count {
                data.len().checked_sub(cursor).ok_or_else(|| {
                    Error::new(ErrorKind::MalformedMedia, "AV1 tile data is truncated")
                })?
            } else {
                let size_len = usize::from(tile_size_bytes);
                let size_end = cursor
                    .checked_add(size_len)
                    .ok_or_else(|| resource("AV1 tile size offset overflow"))?;
                let size_bytes = data.get(cursor..size_end).ok_or_else(|| {
                    Error::new(ErrorKind::MalformedMedia, "AV1 tile size is truncated")
                })?;
                let mut minus_one = 0u32;
                for (shift, byte) in size_bytes.iter().enumerate() {
                    minus_one |= u32::from(*byte) << (shift * 8);
                }
                cursor = size_end;
                usize::try_from(minus_one)
                    .map_err(|_| resource("AV1 tile size does not fit this platform"))?
                    .checked_add(1)
                    .ok_or_else(|| resource("AV1 tile size overflow"))?
            };
            let end = cursor
                .checked_add(tile_len)
                .ok_or_else(|| resource("AV1 tile end overflow"))?;
            if end > data.len() {
                return malformed("AV1 tile data is truncated");
            }
            tile_ranges.push((cursor, end));
            cursor = end;
        }
        if cursor != data.len() {
            return malformed("AV1 tile group has unconsumed data");
        }
        Ok(Av1TileGroup {
            tile_start,
            tile_end,
            data,
            tile_ranges,
        })
    }

    fn parse_obu(&mut self, header: Av1ObuHeader, payload: &[u8]) -> Result<Av1Obu> {
        match header.obu_type {
            Av1ObuType::SequenceHeader => {
                let sequence = parse_sequence_header(payload, &self.limits)?;
                self.sequence = Some(sequence.clone());
                Ok(Av1Obu::SequenceHeader { header, sequence })
            }
            Av1ObuType::TemporalDelimiter => {
                if !payload.is_empty() {
                    return malformed("temporal delimiter OBU must be empty");
                }
                self.seen_frame_header = false;
                Ok(Av1Obu::TemporalDelimiter { header })
            }
            Av1ObuType::FrameHeader | Av1ObuType::RedundantFrameHeader => {
                let sequence = self.sequence.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorKind::MalformedMedia,
                        "frame header precedes sequence header",
                    )
                })?;
                let frame = if header.obu_type == Av1ObuType::RedundantFrameHeader {
                    self.frame.clone().ok_or_else(|| {
                        Error::new(
                            ErrorKind::MalformedMedia,
                            "redundant frame header has no frame to copy",
                        )
                    })?
                } else {
                    parse_frame_header(payload, sequence)?
                };
                self.seen_frame_header = !frame.show_existing_frame;
                self.frame = Some(frame.clone());
                Ok(Av1Obu::FrameHeader {
                    header,
                    frame,
                    payload: payload.to_vec(),
                })
            }
            Av1ObuType::Frame => {
                let sequence = self.sequence.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorKind::MalformedMedia,
                        "frame OBU precedes sequence header",
                    )
                })?;
                let frame = parse_frame_header(payload, sequence)?;
                self.seen_frame_header = false;
                self.frame = Some(frame.clone());
                Ok(Av1Obu::Frame {
                    header,
                    frame,
                    payload: payload.to_vec(),
                })
            }
            Av1ObuType::TileGroup => {
                if !self.seen_frame_header {
                    return malformed("tile group has no active frame header");
                }
                // Tile layout parsing belongs to the reconstruction milestone. Until
                // it supplies a count, one bounded tile covers the group payload.
                let tile_count = 1u32;
                if tile_count > self.limits.max_av1_tiles_per_frame {
                    return Err(resource("AV1 tile count exceeds configured limit"));
                }
                Ok(Av1Obu::TileGroup {
                    header,
                    group: Self::parse_tile_group(payload, 0, 0, 1, &self.limits)?,
                })
            }
            Av1ObuType::Metadata => {
                let (metadata_type, used) = parse_leb128(payload)?;
                let data = payload
                    .get(used..)
                    .ok_or_else(|| {
                        Error::new(ErrorKind::MalformedMedia, "AV1 metadata is truncated")
                    })?
                    .to_vec();
                Ok(Av1Obu::Metadata {
                    header,
                    metadata: Av1Metadata {
                        metadata_type,
                        data,
                    },
                })
            }
            Av1ObuType::Padding | Av1ObuType::Reserved | Av1ObuType::TileList => {
                Ok(Av1Obu::Skipped {
                    header,
                    payload_len: payload.len() as u64,
                })
            }
        }
    }
}

fn parse_obu_header(bytes: &[u8]) -> Result<(Av1ObuHeader, usize)> {
    let first = *bytes
        .first()
        .ok_or_else(|| Error::new(ErrorKind::MalformedMedia, "AV1 OBU header is truncated"))?;
    if first & 0x80 != 0 || first & 0x01 != 0 {
        return malformed("AV1 OBU header reserved bits are nonzero");
    }
    if first & 0x02 == 0 {
        return malformed("low-overhead AV1 OBU has no size field");
    }
    let extension_flag = first & 0x04 != 0;
    let (temporal_id, spatial_id, len) = if extension_flag {
        let ext = *bytes.get(1).ok_or_else(|| {
            Error::new(
                ErrorKind::MalformedMedia,
                "AV1 OBU extension header is truncated",
            )
        })?;
        if ext & 0x07 != 0 {
            return malformed("AV1 OBU extension reserved bits are nonzero");
        }
        (ext >> 5, (ext >> 3) & 0x03, 2)
    } else {
        (0, 0, 1)
    };
    Ok((
        Av1ObuHeader {
            obu_type: Av1ObuType::from_bits((first >> 3) & 0x0f),
            extension_flag,
            temporal_id,
            spatial_id,
            header_len: len as u8,
            size_field_len: 0,
        },
        len,
    ))
}

fn parse_sequence_header(payload: &[u8], limits: &Limits) -> Result<Av1SequenceHeader> {
    let mut b = Bits::new(payload);
    let seq_profile = b.u8(3, "seq_profile")?;
    if seq_profile > 2 {
        return malformed("AV1 sequence profile is reserved");
    }
    let still_picture = b.flag("still_picture")?;
    let reduced = b.flag("reduced_still_picture_header")?;
    if reduced && !still_picture {
        return malformed("reduced AV1 header is not a still picture");
    }
    let mut decoder_model_info_present = false;
    let mut equal_picture_interval = false;
    let mut buffer_delay_length = 0;
    let mut buffer_removal_time_length = 0;
    let mut frame_presentation_time_length = 0;
    let mut operating_points = Vec::new();
    if reduced {
        operating_points.push(Av1OperatingPoint {
            idc: 0,
            level: b.u8(5, "seq_level_idx")?,
            tier: false,
            decoder_model_present: false,
            initial_display_delay: None,
        });
    } else {
        let timing_present = b.flag("timing_info_present_flag")?;
        if timing_present {
            b.skip(64, "timing_info")?;
            equal_picture_interval = b.flag("equal_picture_interval")?;
            if equal_picture_interval {
                let _ = b.uvlc("num_ticks_per_picture_minus_1")?;
            }
            decoder_model_info_present = b.flag("decoder_model_info_present_flag")?;
            if decoder_model_info_present {
                buffer_delay_length = b
                    .u8(5, "buffer_delay_length_minus_1")?
                    .checked_add(1)
                    .ok_or_else(|| resource("buffer delay length overflow"))?;
                b.skip(32, "num_units_in_decoding_tick")?;
                buffer_removal_time_length = b
                    .u8(5, "buffer_removal_time_length_minus_1")?
                    .checked_add(1)
                    .ok_or_else(|| resource("buffer removal time length overflow"))?;
                frame_presentation_time_length = b
                    .u8(5, "frame_presentation_time_length_minus_1")?
                    .checked_add(1)
                    .ok_or_else(|| resource("frame presentation time length overflow"))?;
            }
        }
        let initial_display_delay_present = b.flag("initial_display_delay_present_flag")?;
        let count = usize::from(b.u8(5, "operating_points_cnt_minus_1")?) + 1;
        if count > usize::from(limits.max_av1_operating_points) {
            return Err(resource(
                "AV1 operating point count exceeds configured limit",
            ));
        }
        operating_points
            .try_reserve(count)
            .map_err(|_| resource("AV1 operating point allocation failed"))?;
        for _ in 0..count {
            let idc = b.u16(12, "operating_point_idc")?;
            let level = b.u8(5, "seq_level_idx")?;
            let tier = level > 7 && b.flag("seq_tier")?;
            let model =
                decoder_model_info_present && b.flag("decoder_model_present_for_this_op")?;
            if model {
                b.skip(
                    usize::from(buffer_delay_length) * 2 + 1,
                    "operating parameters",
                )?;
            }
            let delay = if initial_display_delay_present
                && b.flag("initial_display_delay_present_for_this_op")?
            {
                Some(b.u8(4, "initial_display_delay_minus_1")? + 1)
            } else {
                None
            };
            operating_points.push(Av1OperatingPoint {
                idc,
                level,
                tier,
                decoder_model_present: model,
                initial_display_delay: delay,
            });
        }
    }
    let frame_width_bits = b.u8(4, "frame_width_bits_minus_1")? + 1;
    let frame_height_bits = b.u8(4, "frame_height_bits_minus_1")? + 1;
    let max_frame_width = b
        .u32(usize::from(frame_width_bits), "max_frame_width_minus_1")?
        .checked_add(1)
        .ok_or_else(|| resource("AV1 width overflow"))?;
    let max_frame_height = b
        .u32(usize::from(frame_height_bits), "max_frame_height_minus_1")?
        .checked_add(1)
        .ok_or_else(|| resource("AV1 height overflow"))?;
    if max_frame_width > limits.max_width || max_frame_height > limits.max_height {
        return Err(resource("AV1 dimensions exceed configured limits"));
    }
    let frame_id_numbers_present = !reduced && b.flag("frame_id_numbers_present_flag")?;
    let (delta_frame_id_length, additional_frame_id_length) = if frame_id_numbers_present {
        (
            b.u8(4, "delta_frame_id_length_minus_2")? + 2,
            b.u8(3, "additional_frame_id_length_minus_1")? + 1,
        )
    } else {
        (0, 0)
    };
    let _use_128x128_superblock = b.flag("use_128x128_superblock")?;
    let _enable_filter_intra = b.flag("enable_filter_intra")?;
    let _enable_intra_edge_filter = b.flag("enable_intra_edge_filter")?;
    let (
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
    ) = if reduced {
        (false, false, false, false, false, false, false, 2, 2, 0)
    } else {
        let enable_interintra_compound = b.flag("enable_interintra_compound")?;
        let enable_masked_compound = b.flag("enable_masked_compound")?;
        let enable_warped_motion = b.flag("enable_warped_motion")?;
        let enable_dual_filter = b.flag("enable_dual_filter")?;
        let enable_order_hint = b.flag("enable_order_hint")?;
        let (enable_jnt_comp, enable_ref_frame_mvs) = if enable_order_hint {
            (b.flag("enable_jnt_comp")?, b.flag("enable_ref_frame_mvs")?)
        } else {
            (false, false)
        };
        let screen = if b.flag("seq_choose_screen_content_tools")? {
            2
        } else {
            u8::from(b.flag("seq_force_screen_content_tools")?)
        };
        let integer = if screen > 0 {
            if b.flag("seq_choose_integer_mv")? {
                2
            } else {
                u8::from(b.flag("seq_force_integer_mv")?)
            }
        } else {
            2
        };
        let bits = if enable_order_hint {
            b.u8(3, "order_hint_bits_minus_1")? + 1
        } else {
            0
        };
        (
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            screen,
            integer,
            bits,
        )
    };
    let enable_superres = b.flag("enable_superres")?;
    let enable_cdef = b.flag("enable_cdef")?;
    let enable_restoration = b.flag("enable_restoration")?;
    let color_config = parse_color_config(&mut b, seq_profile)?;
    let film_grain_params_present = b.flag("film_grain_params_present")?;
    b.trailing_bits()?;
    Ok(Av1SequenceHeader {
        seq_profile,
        support: support_for_profile(seq_profile),
        still_picture,
        reduced_still_picture_header: reduced,
        operating_points,
        max_frame_width,
        max_frame_height,
        frame_width_bits,
        frame_height_bits,
        frame_id_numbers_present,
        delta_frame_id_length,
        additional_frame_id_length,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        order_hint_bits,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color_config,
        film_grain_params_present,
        decoder_model_info_present,
        equal_picture_interval,
        buffer_delay_length,
        buffer_removal_time_length,
        frame_presentation_time_length,
    })
}

fn parse_color_config(b: &mut Bits<'_>, profile: u8) -> Result<Av1ColorConfig> {
    let high = b.flag("high_bitdepth")?;
    let twelve = profile == 2 && high && b.flag("twelve_bit")?;
    let bit_depth = if twelve {
        12
    } else if high {
        10
    } else {
        8
    };
    let monochrome = profile != 1 && b.flag("mono_chrome")?;
    let color_description = if b.flag("color_description_present_flag")? {
        Some((
            b.u8(8, "color_primaries")?,
            b.u8(8, "transfer_characteristics")?,
            b.u8(8, "matrix_coefficients")?,
        ))
    } else {
        None
    };
    let identity_rgb = color_description == Some((1, 13, 0));
    let (color_range, sx, sy, pos, separate) = if monochrome {
        (b.flag("color_range")?, true, true, 0, false)
    } else if identity_rgb {
        if profile != 1 {
            return malformed("identity RGB requires AV1 profile 1");
        }
        (true, false, false, 0, false)
    } else {
        let range = b.flag("color_range")?;
        let (sx, sy) = match profile {
            0 => (true, true),
            1 => (false, false),
            2 if bit_depth == 12 => (b.flag("subsampling_x")?, b.flag("subsampling_y")?),
            2 => (true, false),
            _ => unreachable!(),
        };
        let pos = if sx && sy {
            b.u8(2, "chroma_sample_position")?
        } else {
            0
        };
        (range, sx, sy, pos, b.flag("separate_uv_delta_q")?)
    };
    Ok(Av1ColorConfig {
        bit_depth,
        monochrome,
        color_description,
        color_range,
        subsampling_x: sx,
        subsampling_y: sy,
        chroma_sample_position: pos,
        separate_uv_delta_q: separate,
    })
}

fn parse_frame_header(payload: &[u8], seq: &Av1SequenceHeader) -> Result<Av1FrameHeader> {
    let mut b = Bits::new(payload);
    let mut show_existing = false;
    let mut map_idx = None;
    let (frame_type, show_frame, showable_frame, error_resilient) = if seq
        .reduced_still_picture_header
    {
        (Av1FrameType::Key, true, false, true)
    } else {
        show_existing = b.flag("show_existing_frame")?;
        if show_existing {
            map_idx = Some(b.u8(3, "frame_to_show_map_idx")?);
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                b.skip(
                    usize::from(seq.frame_presentation_time_length),
                    "frame_presentation_time",
                )?;
            }
            return Ok(Av1FrameHeader {
                show_existing_frame: true,
                frame_to_show_map_idx: map_idx,
                frame_type: Av1FrameType::Inter,
                show_frame: true,
                showable_frame: false,
                error_resilient_mode: false,
                disable_cdf_update: false,
                allow_screen_content_tools: false,
                force_integer_mv: false,
                current_frame_id: None,
                frame_size_override_flag: false,
                order_hint: 0,
                primary_ref_frame: None,
                refresh_frame_flags: 0,
                parsed_bits: b.position(),
            });
        }
        let frame_type = match b.u8(2, "frame_type")? {
            0 => Av1FrameType::Key,
            1 => Av1FrameType::Inter,
            2 => Av1FrameType::IntraOnly,
            _ => Av1FrameType::Switch,
        };
        let show = b.flag("show_frame")?;
        if show && seq.decoder_model_info_present && !seq.equal_picture_interval {
            b.skip(
                usize::from(seq.frame_presentation_time_length),
                "frame_presentation_time",
            )?;
        }
        let showable = if show {
            frame_type != Av1FrameType::Key
        } else {
            b.flag("showable_frame")?
        };
        let error =
            if frame_type == Av1FrameType::Switch || (frame_type == Av1FrameType::Key && show) {
                true
            } else {
                b.flag("error_resilient_mode")?
            };
        (frame_type, show, showable, error)
    };
    let disable_cdf_update = b.flag("disable_cdf_update")?;
    let allow_screen_content_tools = match seq.seq_force_screen_content_tools {
        2 => b.flag("allow_screen_content_tools")?,
        v => v != 0,
    };
    let force_integer_mv = if allow_screen_content_tools {
        match seq.seq_force_integer_mv {
            2 => b.flag("force_integer_mv")?,
            v => v != 0,
        }
    } else {
        false
    };
    let current_frame_id = if seq.frame_id_numbers_present {
        Some(b.u32(
            usize::from(seq.delta_frame_id_length + seq.additional_frame_id_length),
            "current_frame_id",
        )?)
    } else {
        None
    };
    let frame_size_override_flag = if frame_type == Av1FrameType::Switch {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        b.flag("frame_size_override_flag")?
    };
    let order_hint = if seq.enable_order_hint {
        b.u32(usize::from(seq.order_hint_bits), "order_hint")?
    } else {
        0
    };
    let intra = matches!(frame_type, Av1FrameType::Key | Av1FrameType::IntraOnly);
    let primary_ref_frame = if intra || error_resilient {
        None
    } else {
        Some(b.u8(3, "primary_ref_frame")?)
    };
    let refresh_frame_flags =
        if frame_type == Av1FrameType::Switch || (frame_type == Av1FrameType::Key && show_frame) {
            0xff
        } else {
            b.u8(8, "refresh_frame_flags")?
        };
    Ok(Av1FrameHeader {
        show_existing_frame: show_existing,
        frame_to_show_map_idx: map_idx,
        frame_type,
        show_frame,
        showable_frame,
        error_resilient_mode: error_resilient,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        parsed_bits: b.position(),
    })
}

fn parse_leb128(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8usize {
        let byte = *bytes.get(i).ok_or_else(|| {
            Error::new(ErrorKind::MalformedMedia, "AV1 leb128 value is truncated")
        })?;
        let part = u64::from(byte & 0x7f)
            .checked_shl((i * 7) as u32)
            .ok_or_else(|| resource("AV1 leb128 shift overflow"))?;
        value = value
            .checked_add(part)
            .ok_or_else(|| resource("AV1 leb128 value overflow"))?;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    malformed("AV1 leb128 value exceeds eight bytes")
}

struct Bits<'a> {
    bytes: &'a [u8],
    bit: usize,
}
impl<'a> Bits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }
    fn position(&self) -> usize {
        self.bit
    }
    fn u32(&mut self, n: usize, name: &str) -> Result<u32> {
        if n > 32 {
            return Err(resource(format!("{name} is wider than 32 bits")));
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.flag(name)?);
        }
        Ok(value)
    }
    fn u16(&mut self, n: usize, name: &str) -> Result<u16> {
        Ok(self.u32(n, name)? as u16)
    }
    fn u8(&mut self, n: usize, name: &str) -> Result<u8> {
        Ok(self.u32(n, name)? as u8)
    }
    fn flag(&mut self, name: &str) -> Result<bool> {
        let byte = self.bytes.get(self.bit / 8).ok_or_else(|| {
            Error::new(
                ErrorKind::MalformedMedia,
                format!("AV1 {name} is truncated"),
            )
        })?;
        let value = byte & (0x80 >> (self.bit & 7)) != 0;
        self.bit = self
            .bit
            .checked_add(1)
            .ok_or_else(|| resource("AV1 bit offset overflow"))?;
        Ok(value)
    }
    fn skip(&mut self, n: usize, name: &str) -> Result<()> {
        for _ in 0..n {
            self.flag(name)?;
        }
        Ok(())
    }
    fn uvlc(&mut self, name: &str) -> Result<u32> {
        let mut leading = 0usize;
        while !self.flag(name)? {
            leading += 1;
            if leading >= 32 {
                return Err(resource(format!("AV1 {name} exceeds 32 bits")));
            }
        }
        if leading == 0 {
            return Ok(0);
        }
        let suffix = self.u32(leading, name)?;
        (1u32 << leading)
            .checked_sub(1)
            .and_then(|v| v.checked_add(suffix))
            .ok_or_else(|| resource(format!("AV1 {name} overflow")))
    }
    fn trailing_bits(&mut self) -> Result<()> {
        if !self.flag("trailing_one_bit")? {
            return malformed("AV1 trailing one bit is zero");
        }
        while self.bit & 7 != 0 {
            if self.flag("trailing_zero_bit")? {
                return malformed("AV1 trailing zero bit is nonzero");
            }
        }
        if self.bit != self.bytes.len() * 8 {
            return malformed("AV1 sequence header has trailing bytes");
        }
        Ok(())
    }
}

fn support_for_profile(profile: u8) -> Av1SyntaxSupport {
    if profile == 0 {
        Av1SyntaxSupport::MainProfile
    } else {
        Av1SyntaxSupport::UnsupportedProfile(profile)
    }
}
fn check_allocation(bytes: usize, limits: &Limits, what: &str) -> Result<()> {
    if bytes as u128 > u128::from(limits.max_allocation_bytes) {
        Err(resource(format!(
            "{what} exceeds configured allocation limit"
        )))
    } else {
        Ok(())
    }
}
fn malformed<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::new(ErrorKind::MalformedMedia, message))
}
fn resource(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl BitWriter {
        fn bits(&mut self, value: u32, width: usize) {
            for shift in (0..width).rev() {
                if self.bit & 7 == 0 {
                    self.bytes.push(0);
                }
                if value & (1 << shift) != 0 {
                    let last = self.bytes.len() - 1;
                    self.bytes[last] |= 0x80 >> (self.bit & 7);
                }
                self.bit += 1;
            }
        }

        fn trailing_bits(&mut self) {
            self.bits(1, 1);
            while self.bit & 7 != 0 {
                self.bits(0, 1);
            }
        }
    }

    fn reduced_sequence_payload() -> Vec<u8> {
        let mut b = BitWriter::default();
        b.bits(0, 3); // Main profile
        b.bits(1, 1); // still_picture
        b.bits(1, 1); // reduced_still_picture_header
        b.bits(4, 5); // level 2.0
        b.bits(0, 4); // one width bit
        b.bits(0, 4); // one height bit
        b.bits(0, 1); // max width minus one = 0
        b.bits(0, 1); // max height minus one = 0
        b.bits(0, 1); // use_128x128_superblock
        b.bits(0, 1); // enable_filter_intra
        b.bits(0, 1); // enable_intra_edge_filter
        b.bits(0, 1); // enable_superres
        b.bits(0, 1); // enable_cdef
        b.bits(0, 1); // enable_restoration
        b.bits(0, 1); // high_bitdepth
        b.bits(1, 1); // monochrome
        b.bits(0, 1); // color description absent
        b.bits(0, 1); // studio range
        b.bits(0, 1); // film grain absent
        b.trailing_bits();
        b.bytes
    }

    fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 128);
        let mut bytes = vec![(obu_type << 3) | 0x02, payload.len() as u8];
        bytes.extend_from_slice(payload);
        bytes
    }

    fn av1c(payload: &[u8]) -> Vec<u8> {
        let mut bytes = (8u32 + payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(b"av1C");
        bytes.extend_from_slice(payload);
        bytes
    }

    fn extended_av1c(payload: &[u8]) -> Vec<u8> {
        let mut bytes = 1u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"av1C");
        bytes.extend_from_slice(&(16u64 + payload.len() as u64).to_be_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn parses_sequence_frame_tile_metadata_and_padding_obus() {
        let mut stream = obu(1, &reduced_sequence_payload());
        stream.extend_from_slice(&obu(2, &[]));
        stream.extend_from_slice(&obu(3, &[0]));
        stream.extend_from_slice(&obu(4, &[0xaa, 0xbb]));
        stream.extend_from_slice(&obu(5, &[0xac, 0x02, 7, 8]));
        stream.extend_from_slice(&obu(15, &[0, 0]));

        let mut parser = Av1Parser::new(Limits::default()).unwrap();
        let parsed = parser.parse_low_overhead(&stream).unwrap();
        assert_eq!(parsed.len(), 6);
        let sequence = parser.sequence.as_ref().unwrap();
        assert_eq!(sequence.max_frame_width, 1);
        assert_eq!(sequence.max_frame_height, 1);
        assert_eq!(sequence.support, Av1SyntaxSupport::MainProfile);
        assert_eq!(sequence.color_config.bit_depth, 8);
        assert!(sequence.color_config.monochrome);
        assert!(matches!(parsed[2], Av1Obu::FrameHeader { .. }));
        assert!(matches!(parsed[3], Av1Obu::TileGroup { .. }));
        match &parsed[4] {
            Av1Obu::Metadata { metadata, .. } => {
                assert_eq!(metadata.metadata_type, 300);
                assert_eq!(metadata.data, [7, 8]);
            }
            other => panic!("unexpected OBU: {other:?}"),
        }
        assert!(matches!(parsed[5], Av1Obu::Skipped { payload_len: 2, .. }));
    }

    #[test]
    fn parses_and_cross_checks_complete_av1c_record() {
        let sequence_obu = obu(1, &reduced_sequence_payload());
        let mut record = vec![0x81, 0x04, 0x1c, 0];
        record.extend_from_slice(&sequence_obu);
        let parsed =
            Av1CodecConfigurationRecord::parse(&av1c(&record), &Limits::default()).unwrap();
        assert_eq!(parsed.seq_profile, 0);
        assert_eq!(parsed.seq_level_idx_0, 4);
        assert_eq!(parsed.config_obus.len(), 1);
        assert_eq!(parsed.support, Av1SyntaxSupport::MainProfile);

        let extended =
            Av1CodecConfigurationRecord::parse(&extended_av1c(&record), &Limits::default())
                .unwrap();
        assert_eq!(extended, parsed);
    }

    #[test]
    fn av1c_rejects_disallowed_configuration_obus() {
        let mut record = vec![0x81, 0x04, 0, 0];
        record.extend_from_slice(&obu(2, &[]));
        let error =
            Av1CodecConfigurationRecord::parse(&av1c(&record), &Limits::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    }

    #[test]
    fn unsupported_profile_is_parsed_but_never_marked_main() {
        let parsed =
            Av1CodecConfigurationRecord::parse(&av1c(&[0x81, 0x24, 0, 0]), &Limits::default())
                .unwrap();
        assert_eq!(parsed.support, Av1SyntaxSupport::UnsupportedProfile(1));
    }

    #[test]
    fn malformed_corpus_returns_stable_errors_without_panics() {
        let corpus: &[&[u8]] = &[
            &[0x0a],
            &[0x0a, 0x80],
            &[0x0a, 4, 0],
            &[0x80, 0],
            &[0x0e, 0],
            &[0x06, 0],
            &[0x16, 0],
            &[0x2a, 1, 0x80],
        ];
        for bytes in corpus {
            let result = std::panic::catch_unwind(|| {
                Av1Parser::new(Limits::default())
                    .unwrap()
                    .parse_low_overhead(bytes)
            });
            assert!(result.is_ok(), "parser panicked for {bytes:02x?}");
            let error = result.unwrap().unwrap_err();
            assert!(matches!(
                error.kind(),
                ErrorKind::MalformedMedia | ErrorKind::ResourceLimit
            ));
        }
    }

    #[test]
    fn gates_dimensions_obu_counts_and_allocations() {
        let sequence = obu(1, &reduced_sequence_payload());
        let dimensions = Limits {
            max_width: 0,
            ..Limits::default()
        };
        let error = Av1Parser::new(dimensions)
            .unwrap()
            .parse_low_overhead(&sequence)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);

        let counts = Limits {
            max_av1_obus_per_unit: 1,
            ..Limits::default()
        };
        let mut two = obu(2, &[]);
        two.extend_from_slice(&obu(2, &[]));
        let error = Av1Parser::new(counts)
            .unwrap()
            .parse_low_overhead(&two)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);

        let allocations = Limits {
            max_allocation_bytes: 1,
            ..Limits::default()
        };
        let error = Av1Parser::new(allocations)
            .unwrap()
            .parse_low_overhead(&[0x12, 0])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    }

    #[test]
    fn parses_bounded_multi_tile_group_sizes() {
        // Four tiles, explicit range 1..2, then a one-byte size-minus-one for
        // the first tile followed by both tile payloads.
        let payload = [0b1011_0000, 1, 0xaa, 0xbb, 0xcc];
        let group = Av1Parser::parse_tile_group(&payload, 2, 0, 1, &Limits::default()).unwrap();
        assert_eq!((group.tile_start, group.tile_end), (1, 2));
        assert_eq!(group.tile_ranges, [(1, 3), (3, 4)]);
        assert_eq!(group.data, [1, 0xaa, 0xbb, 0xcc]);

        let limits = Limits {
            max_av1_tiles_per_frame: 2,
            ..Limits::default()
        };
        assert_eq!(
            Av1Parser::parse_tile_group(&payload, 2, 0, 1, &limits)
                .unwrap_err()
                .kind(),
            ErrorKind::ResourceLimit
        );
    }
}
