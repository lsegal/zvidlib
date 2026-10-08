//! Cover art generated from a recorded video frame (issue #510).
//!
//! [`CoverArt`] from issue #502 is a picture the caller encodes. Most callers
//! only want a file-browser thumbnail of the recording, and zvidlib already
//! sees every frame, so `zvidlib::MediaOutput` captures one by
//! default and writes it as a PNG when the output finishes.
//!
//! The capture happens as frames are written rather than by decoding at
//! `finish`: the writer has the raw pixels in hand, and decoding would need a
//! decoder for every codec an encoder can emit. Frames up to and including the
//! chosen index are shrunk to a thumbnail as they arrive, each replacing the
//! last, so a stream shorter than the index still ends up with its last frame.
//! Only one thumbnail is held at a time: at most
//! [`COVER_THUMBNAIL_MAX_EDGE`]² x 3 bytes (768 KiB), and 480x270 x 3 bytes
//! (380 KiB) for 1080p. Frames after the chosen index cost nothing.

use crate::media::{ColorRange, PixelFormat, Plane, VideoFrame};
use crate::mp4::{CoverArt, CoverArtFormat};
use crate::transfer::{FrameSource, Orientation};
use crate::{Error, ErrorKind, Result};

/// The zero-based presentation index [`CoverSource::default`] picks. Frame 0
/// is often black or mid fade-in, so the default skips a few frames.
pub const DEFAULT_COVER_FRAME: u64 = 4;

/// The longest edge, in pixels, of a cover generated from a video frame.
/// Larger frames are shrunk by a whole-number factor until they fit.
pub const COVER_THUMBNAIL_MAX_EDGE: u32 = 512;

/// Where the cover art of a `zvidlib::MediaOutput` comes from when the caller
/// has not set one with `MediaOutput::set_cover_art`, which always takes
/// precedence.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CoverSource {
    /// Write no generated cover.
    None,
    /// Use the video frame with this zero-based presentation index, shrunk to
    /// a thumbnail and encoded as PNG. A stream with fewer frames uses its
    /// last frame, and a stream with no video frames gets no cover. A frame
    /// supplied as a GPU resource cannot be read back, so it gives no cover.
    Frame(u64),
}

impl Default for CoverSource {
    fn default() -> Self {
        Self::Frame(DEFAULT_COVER_FRAME)
    }
}

impl CoverArt {
    /// Encodes `frame` as a PNG thumbnail no larger than
    /// [`COVER_THUMBNAIL_MAX_EDGE`] on either edge. Limited-range frames are
    /// expanded to full range, `Yuv420p8` is converted with the BT.601 matrix
    /// zvidlib's decoders use, and alpha is dropped.
    pub fn from_video_frame(frame: &VideoFrame) -> Result<Self> {
        Ok(Thumbnail::new(frame, Orientation::TopLeft)?.to_cover_art())
    }
}

/// Tracks the frame a [`CoverSource`] selects while a stream is written.
#[derive(Debug)]
#[doc(hidden)]
pub struct CoverCapture {
    source: CoverSource,
    thumbnail: Option<Thumbnail>,
}

impl CoverCapture {
    #[doc(hidden)]
    pub fn new(source: CoverSource) -> Self {
        Self {
            source,
            thumbnail: None,
        }
    }

    /// Offers the frame at presentation `index`. A frame at or before the
    /// chosen index replaces the held thumbnail; later frames are ignored.
    #[doc(hidden)]
    pub fn offer(&mut self, index: u64, frame: FrameSource<'_>) {
        let CoverSource::Frame(target) = self.source else {
            return;
        };
        if index > target {
            return;
        }
        // A thumbnail is never worth failing a recording over, and one that
        // cannot be made must not leave an earlier frame standing in for it.
        self.thumbnail = match frame {
            FrameSource::Cpu(source) => Thumbnail::new(source.frame, source.orientation).ok(),
            FrameSource::Graphics(_) => None,
        };
    }

    /// Encodes the captured frame, if there is one.
    #[doc(hidden)]
    pub fn cover_art(&self) -> Option<CoverArt> {
        self.thumbnail.as_ref().map(Thumbnail::to_cover_art)
    }
}

/// A shrunk, full-range, top-down RGB copy of one frame.
#[derive(Debug)]
struct Thumbnail {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

impl Thumbnail {
    fn new(frame: &VideoFrame, orientation: Orientation) -> Result<Self> {
        let source_width = frame.dimensions.width as usize;
        let source_height = frame.dimensions.height as usize;
        let longest = frame.dimensions.width.max(frame.dimensions.height);
        let scale = longest.div_ceil(COVER_THUMBNAIL_MAX_EDGE).max(1) as usize;
        let width = source_width.div_ceil(scale);
        let height = source_height.div_ceil(scale);
        let reader = PixelReader::new(frame)?;
        let mut rgb = vec![0_u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let mut totals = [0_u32; 3];
                let mut count = 0_u32;
                for source_y in y * scale..((y + 1) * scale).min(source_height) {
                    let row = match orientation {
                        Orientation::TopLeft => source_y,
                        Orientation::BottomLeft => source_height - 1 - source_y,
                    };
                    for source_x in x * scale..((x + 1) * scale).min(source_width) {
                        let pixel = reader.rgb(source_x, row);
                        for (total, channel) in totals.iter_mut().zip(pixel) {
                            *total += u32::from(channel);
                        }
                        count += 1;
                    }
                }
                let offset = (y * width + x) * 3;
                for (channel, total) in rgb[offset..offset + 3].iter_mut().zip(totals) {
                    *channel = ((total + count / 2) / count) as u8;
                }
            }
        }
        Ok(Self { width, height, rgb })
    }

    fn to_cover_art(&self) -> CoverArt {
        CoverArt {
            format: CoverArtFormat::Png,
            data: encode_png_rgb(self.width, self.height, &self.rgb),
        }
    }
}

/// Reads one full-range RGB pixel at a time from a validated frame.
struct PixelReader<'a> {
    format: PixelFormat,
    range: ColorRange,
    planes: &'a [Plane],
}

impl<'a> PixelReader<'a> {
    fn new(frame: &'a VideoFrame) -> Result<Self> {
        let planes_needed = match frame.pixel_format {
            PixelFormat::Yuv420p8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Rgb8 | PixelFormat::Gray8 => 1,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "cover art cannot be made from this pixel format",
                ));
            }
        };
        if frame.planes.len() < planes_needed {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "video frame is missing a plane",
            ));
        }
        Ok(Self {
            format: frame.pixel_format,
            range: frame.color_range,
            planes: &frame.planes,
        })
    }

    fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let plane = &self.planes[0];
        let row = &plane.data[y * plane.stride..];
        let rgb = match self.format {
            PixelFormat::Rgba8 => [row[x * 4], row[x * 4 + 1], row[x * 4 + 2]],
            PixelFormat::Bgra8 => [row[x * 4 + 2], row[x * 4 + 1], row[x * 4]],
            PixelFormat::Rgb8 => [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]],
            PixelFormat::Gray8 => [row[x]; 3],
            PixelFormat::Yuv420p8 => {
                let chroma = |plane: &Plane| plane.data[(y / 2) * plane.stride + x / 2];
                return yuv_to_rgb(
                    row[x],
                    chroma(&self.planes[1]),
                    chroma(&self.planes[2]),
                    self.range,
                );
            }
            _ => unreachable!("PixelReader::new rejects every other pixel format"),
        };
        match self.range {
            ColorRange::Limited => rgb.map(|component| {
                ((u32::from(component.saturating_sub(16)) * 255 + 109) / 219).min(255) as u8
            }),
            _ => rgb,
        }
    }
}

/// BT.601 in Q16 fixed point: the limited-range matrix zvidlib's decoders
/// convert with, or its JPEG full-range form.
fn yuv_to_rgb(y: u8, u: u8, v: u8, range: ColorRange) -> [u8; 3] {
    let (y, u, v) = (i32::from(y), i32::from(u) - 128, i32::from(v) - 128);
    let (luma, r, g_u, g_v, b) = match range {
        ColorRange::Limited => ((y - 16) * 76_309, 104_597, -25_675, -53_279, 132_201),
        _ => (y << 16, 91_881, -22_554, -46_802, 116_130),
    };
    let clip = |value: i32| ((value + (1 << 15)) >> 16).clamp(0, 255) as u8;
    [
        clip(luma + r * v),
        clip(luma + g_u * u + g_v * v),
        clip(luma + b * u),
    ]
}

/// Encodes 8-bit RGB as a PNG. Each row takes whichever filter minimizes its
/// sum of absolute differences, and the zlib stream is fixed-Huffman deflate
/// with greedy LZ77 matching, or stored blocks if that would be smaller.
pub(crate) fn encode_png_rgb(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    let row_bytes = width * 3;
    let mut filtered = Vec::with_capacity((row_bytes + 1) * height);
    let mut candidate = vec![0_u8; row_bytes];
    let mut best = vec![0_u8; row_bytes];
    for y in 0..height {
        let row = &rgb[y * row_bytes..(y + 1) * row_bytes];
        let previous = (y > 0).then(|| &rgb[(y - 1) * row_bytes..y * row_bytes]);
        let mut best_filter = 0_u8;
        let mut best_cost = u64::MAX;
        for filter in 0..5_u8 {
            for x in 0..row_bytes {
                let left = if x >= 3 { row[x - 3] } else { 0 };
                let up = previous.map_or(0, |previous| previous[x]);
                let up_left = if x >= 3 {
                    previous.map_or(0, |previous| previous[x - 3])
                } else {
                    0
                };
                let prediction = match filter {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                    _ => paeth(left, up, up_left),
                };
                candidate[x] = row[x].wrapping_sub(prediction);
            }
            let cost = candidate
                .iter()
                .map(|&value| u64::from((value as i8).unsigned_abs()))
                .sum();
            if cost < best_cost {
                best_cost = cost;
                best_filter = filter;
                best.copy_from_slice(&candidate);
            }
        }
        filtered.push(best_filter);
        filtered.extend_from_slice(&best);
    }

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    // Bit depth 8, color type 2 (RGB), deflate, adaptive filtering, no interlace.
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &zlib(&filtered));
    png_chunk(&mut png, b"IEND", &[]);
    png
}

fn paeth(left: u8, up: u8, up_left: u8) -> u8 {
    let estimate = i16::from(left) + i16::from(up) - i16::from(up_left);
    let distance_left = (estimate - i16::from(left)).abs();
    let distance_up = (estimate - i16::from(up)).abs();
    let distance_up_left = (estimate - i16::from(up_left)).abs();
    if distance_left <= distance_up && distance_left <= distance_up_left {
        left
    } else if distance_up <= distance_up_left {
        up
    } else {
        up_left
    }
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for chunk in bytes.chunks(5_552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= 65_521;
        b %= 65_521;
    }
    (b << 16) | a
}

fn zlib(data: &[u8]) -> Vec<u8> {
    // 32 KiB window, no dictionary; 0x7801 is a multiple of 31 as FCHECK needs.
    let mut output = vec![0x78, 0x01];
    let compressed = deflate_fixed(data);
    if compressed.len() < stored_len(data.len()) {
        output.extend_from_slice(&compressed);
    } else {
        deflate_stored(data, &mut output);
    }
    output.extend_from_slice(&adler32(data).to_be_bytes());
    output
}

fn stored_len(length: usize) -> usize {
    length + 5 * length.div_ceil(65_535).max(1)
}

fn deflate_stored(data: &[u8], output: &mut Vec<u8>) {
    let mut blocks = data.chunks(65_535).peekable();
    if blocks.peek().is_none() {
        output.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        output.push(u8::from(blocks.peek().is_none()));
        let length = block.len() as u16;
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&(!length).to_le_bytes());
        output.extend_from_slice(block);
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const WINDOW: usize = 32_768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const HASH_BITS: u32 = 15;
const MAX_CHAIN: usize = 64;

/// LSB-first bit packer for a deflate stream.
struct BitWriter {
    bytes: Vec<u8>,
    buffer: u64,
    bits: u32,
}

impl BitWriter {
    fn bits(&mut self, value: u32, count: u32) {
        self.buffer |= u64::from(value) << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.bytes.push(self.buffer as u8);
            self.buffer >>= 8;
            self.bits -= 8;
        }
    }

    /// Writes a Huffman code, which deflate packs most significant bit first.
    fn code(&mut self, code: u32, length: u32) {
        self.bits(code.reverse_bits() >> (32 - length), length);
    }

    fn literal_length(&mut self, symbol: u16) {
        let symbol = u32::from(symbol);
        match symbol {
            0..=143 => self.code(0x30 + symbol, 8),
            144..=255 => self.code(0x190 + symbol - 144, 9),
            256..=279 => self.code(symbol - 256, 7),
            _ => self.code(0xc0 + symbol - 280, 8),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.bytes.push(self.buffer as u8);
        }
        self.bytes
    }
}

fn deflate_fixed(data: &[u8]) -> Vec<u8> {
    let mut writer = BitWriter {
        bytes: Vec::with_capacity(data.len() / 2),
        buffer: 0,
        bits: 0,
    };
    // BFINAL = 1, BTYPE = 01 (fixed Huffman codes).
    writer.bits(1, 1);
    writer.bits(1, 2);

    let hash = |position: usize| {
        let value = u32::from(data[position])
            | u32::from(data[position + 1]) << 8
            | u32::from(data[position + 2]) << 16;
        (value.wrapping_mul(0x9e37_79b1) >> (32 - HASH_BITS)) as usize
    };
    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut previous = vec![usize::MAX; data.len()];
    let insert = |position: usize, head: &mut [usize], previous: &mut [usize]| {
        if position + MIN_MATCH <= data.len() {
            let key = hash(position);
            previous[position] = head[key];
            head[key] = position;
        }
    };

    let mut position = 0;
    while position < data.len() {
        let mut best_length = 0;
        let mut best_distance = 0;
        if position + MIN_MATCH <= data.len() {
            let limit = (data.len() - position).min(MAX_MATCH);
            let mut candidate = head[hash(position)];
            let mut chain = 0;
            while candidate != usize::MAX && position - candidate <= WINDOW && chain < MAX_CHAIN {
                let length = data[candidate..]
                    .iter()
                    .zip(&data[position..position + limit])
                    .take_while(|(a, b)| a == b)
                    .count();
                if length > best_length {
                    best_length = length;
                    best_distance = position - candidate;
                    if length == limit {
                        break;
                    }
                }
                candidate = previous[candidate];
                chain += 1;
            }
        }
        if best_length >= MIN_MATCH {
            let length_code = LENGTH_BASE
                .iter()
                .rposition(|&base| usize::from(base) <= best_length)
                .unwrap_or(0);
            writer.literal_length(257 + length_code as u16);
            writer.bits(
                (best_length - usize::from(LENGTH_BASE[length_code])) as u32,
                u32::from(LENGTH_EXTRA[length_code]),
            );
            let distance_code = DISTANCE_BASE
                .iter()
                .rposition(|&base| usize::from(base) <= best_distance)
                .unwrap_or(0);
            writer.code(distance_code as u32, 5);
            writer.bits(
                (best_distance - usize::from(DISTANCE_BASE[distance_code])) as u32,
                u32::from(DISTANCE_EXTRA[distance_code]),
            );
            for offset in 0..best_length {
                insert(position + offset, &mut head, &mut previous);
            }
            position += best_length;
        } else {
            writer.literal_length(u16::from(data[position]));
            insert(position, &mut head, &mut previous);
            position += 1;
        }
    }
    writer.literal_length(256);
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Limits;
    use crate::media::VideoDimensions;
    use crate::transfer::CpuFrameSource;

    /// A minimal inflater for the two block types [`zlib`] emits, so the
    /// encoder is checked against a decoder rather than against itself.
    fn inflate(zlib: &[u8]) -> Vec<u8> {
        assert_eq!(&zlib[..2], &[0x78, 0x01]);
        let stream = &zlib[2..zlib.len() - 4];
        let mut output = Vec::new();
        let mut bit = 0_usize;
        let read = |bit: &mut usize, count: usize| {
            let mut value = 0_u32;
            for index in 0..count {
                let byte = stream[(*bit + index) / 8];
                value |= u32::from((byte >> ((*bit + index) % 8)) & 1) << index;
            }
            *bit += count;
            value
        };
        let read_code = |bit: &mut usize, count: usize| {
            let mut value = 0_u32;
            for _ in 0..count {
                value = (value << 1) | read(bit, 1);
            }
            value
        };
        loop {
            let last = read(&mut bit, 1);
            match read(&mut bit, 2) {
                0 => {
                    bit = bit.div_ceil(8) * 8;
                    let at = bit / 8;
                    let length = usize::from(u16::from_le_bytes([stream[at], stream[at + 1]]));
                    output.extend_from_slice(&stream[at + 4..at + 4 + length]);
                    bit = (at + 4 + length) * 8;
                }
                1 => loop {
                    let mut code = read_code(&mut bit, 7);
                    let symbol = if code <= 0x17 {
                        code + 256
                    } else {
                        code = (code << 1) | read(&mut bit, 1);
                        if (0x30..=0xbf).contains(&code) {
                            code - 0x30
                        } else if (0xc0..=0xc7).contains(&code) {
                            code - 0xc0 + 280
                        } else {
                            code = (code << 1) | read(&mut bit, 1);
                            code - 0x190 + 144
                        }
                    } as usize;
                    if symbol < 256 {
                        output.push(symbol as u8);
                        continue;
                    }
                    if symbol == 256 {
                        break;
                    }
                    let index = symbol - 257;
                    let length = usize::from(LENGTH_BASE[index])
                        + read(&mut bit, usize::from(LENGTH_EXTRA[index])) as usize;
                    let distance_code = read_code(&mut bit, 5) as usize;
                    let distance = usize::from(DISTANCE_BASE[distance_code])
                        + read(&mut bit, usize::from(DISTANCE_EXTRA[distance_code])) as usize;
                    for _ in 0..length {
                        output.push(output[output.len() - distance]);
                    }
                },
                other => panic!("unexpected block type {other}"),
            }
            if last == 1 {
                break;
            }
        }
        assert_eq!(
            u32::from_be_bytes(zlib[zlib.len() - 4..].try_into().unwrap()),
            adler32(&output)
        );
        output
    }

    /// Decodes a PNG [`encode_png_rgb`] wrote into its dimensions and RGB rows.
    fn decode_png(png: &[u8]) -> (usize, usize, Vec<u8>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut at = 8;
        let (mut width, mut height, mut idat) = (0, 0, Vec::new());
        while at < png.len() {
            let length = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let body = &png[at + 4..at + 8 + length];
            let crc =
                u32::from_be_bytes(png[at + 8 + length..at + 12 + length].try_into().unwrap());
            assert_eq!(crc, crc32(body));
            let data = &body[4..];
            match &body[..4] {
                b"IHDR" => {
                    width = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
                    height = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
                    assert_eq!(&data[8..], &[8, 2, 0, 0, 0]);
                }
                b"IDAT" => idat.extend_from_slice(data),
                b"IEND" => break,
                _ => {}
            }
            at += 12 + length;
        }
        let filtered = inflate(&idat);
        let row_bytes = width * 3;
        let mut rgb = vec![0_u8; row_bytes * height];
        for y in 0..height {
            let line = &filtered[y * (row_bytes + 1)..(y + 1) * (row_bytes + 1)];
            for x in 0..row_bytes {
                let left = if x >= 3 {
                    rgb[y * row_bytes + x - 3]
                } else {
                    0
                };
                let up = if y > 0 {
                    rgb[(y - 1) * row_bytes + x]
                } else {
                    0
                };
                let up_left = if x >= 3 && y > 0 {
                    rgb[(y - 1) * row_bytes + x - 3]
                } else {
                    0
                };
                let prediction = match line[0] {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                    4 => paeth(left, up, up_left),
                    other => panic!("unexpected filter {other}"),
                };
                rgb[y * row_bytes + x] = line[x + 1].wrapping_add(prediction);
            }
        }
        (width, height, rgb)
    }

    fn frame(width: u32, height: u32, format: PixelFormat, planes: Vec<Plane>) -> VideoFrame {
        let limits = Limits::default();
        VideoFrame::new(
            VideoDimensions::new(width, height, &limits).unwrap(),
            format,
            ColorRange::Full,
            planes,
            &limits,
        )
        .unwrap()
    }

    fn pattern(width: usize, height: usize, seed: u8) -> Vec<u8> {
        (0..width * height * 3)
            .map(|index| {
                let pixel = index / 3;
                let (x, y) = (pixel % width, pixel / width);
                ((x * 7 + y * 13 + (index % 3) * 50) as u8).wrapping_add(seed)
            })
            .collect()
    }

    #[test]
    fn png_round_trips_patterned_flat_and_noisy_images() {
        let mut noise = 0x1234_5678_u32;
        let noisy: Vec<u8> = (0..97 * 31 * 3)
            .map(|_| {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                noise as u8
            })
            .collect();
        for (width, height, rgb) in [
            (1, 1, vec![1, 2, 3]),
            (64, 48, pattern(64, 48, 9)),
            (300, 200, vec![77; 300 * 200 * 3]),
            (97, 31, noisy),
        ] {
            let png = encode_png_rgb(width, height, &rgb);
            assert_eq!(decode_png(&png), (width, height, rgb));
        }
    }

    #[test]
    fn stored_blocks_are_used_when_huffman_coding_would_grow_the_data() {
        let mut noise = 0x9e37_79b9_u32;
        let data: Vec<u8> = (0..70_000)
            .map(|_| {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                noise as u8
            })
            .collect();
        let stream = zlib(&data);
        assert_eq!(stream.len(), 2 + stored_len(data.len()) + 4);
        assert_eq!(inflate(&stream), data);
        assert_eq!(inflate(&zlib(&[])), Vec::<u8>::new());
    }

    #[test]
    fn a_flat_frame_compresses_far_below_its_raw_size() {
        let png = encode_png_rgb(512, 288, &vec![40; 512 * 288 * 3]);
        assert!(png.len() < 4_096, "{} bytes", png.len());
    }

    #[test]
    fn large_frames_shrink_to_the_thumbnail_edge_by_averaging() {
        // 1030 wide needs a factor of 3 to fit 512: 344x2 from 1030x6.
        let (width, height) = (1030_usize, 6_usize);
        let mut rgba = vec![0_u8; width * height * 4];
        for (pixel, value) in rgba.chunks_mut(4).enumerate() {
            let (x, y) = (pixel % width, pixel / width);
            // Each 3x3 block holds 0 and 30 in its first column, 60 elsewhere.
            let level = if x % 3 == 0 { (y % 2 * 30) as u8 } else { 60 };
            value.copy_from_slice(&[level, level / 2, 255 - level, 0]);
        }
        let source = frame(
            width as u32,
            height as u32,
            PixelFormat::Rgba8,
            vec![Plane {
                data: rgba,
                stride: width * 4,
            }],
        );
        let thumbnail = Thumbnail::new(&source, Orientation::TopLeft).unwrap();
        assert_eq!((thumbnail.width, thumbnail.height), (344, 2));
        // Rows 0..3 average (0 + 30 + 0 + 6 * 60) / 9 = 43.3; rows 3..6
        // average (30 + 0 + 30 + 360) / 9 = 46.7.
        assert_eq!(&thumbnail.rgb[..3], &[43, 22, 212]);
        assert_eq!(&thumbnail.rgb[344 * 3..344 * 3 + 3], &[47, 23, 208]);
        // The last column covers only source column 1029, a block-first column.
        assert_eq!(&thumbnail.rgb[343 * 3..344 * 3], &[10, 5, 245]);
    }

    #[test]
    fn every_pixel_format_and_orientation_reads_as_top_down_rgb() {
        let rgb = [[200_u8, 100, 50], [10, 20, 30]];
        let packed = |bytes: &dyn Fn([u8; 3]) -> Vec<u8>| -> Vec<u8> {
            rgb.iter().flat_map(|&pixel| bytes(pixel)).collect()
        };
        let one_by_two = |format, data: Vec<u8>| {
            let stride = data.len() / 2;
            frame(1, 2, format, vec![Plane { data, stride }])
        };
        for source in [
            one_by_two(PixelFormat::Rgba8, packed(&|[r, g, b]| vec![r, g, b, 255])),
            one_by_two(PixelFormat::Bgra8, packed(&|[r, g, b]| vec![b, g, r, 255])),
            one_by_two(PixelFormat::Rgb8, packed(&|pixel| pixel.to_vec())),
        ] {
            let thumbnail = Thumbnail::new(&source, Orientation::TopLeft).unwrap();
            assert_eq!(thumbnail.rgb, rgb.concat());
            let flipped = Thumbnail::new(&source, Orientation::BottomLeft).unwrap();
            assert_eq!(flipped.rgb, [rgb[1], rgb[0]].concat());
        }

        let gray = one_by_two(PixelFormat::Gray8, vec![9, 250]);
        let thumbnail = Thumbnail::new(&gray, Orientation::TopLeft).unwrap();
        assert_eq!(thumbnail.rgb, [9, 9, 9, 250, 250, 250]);

        let mut limited = one_by_two(PixelFormat::Gray8, vec![16, 235]);
        limited.color_range = ColorRange::Limited;
        let thumbnail = Thumbnail::new(&limited, Orientation::TopLeft).unwrap();
        assert_eq!(thumbnail.rgb, [0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn yuv420_converts_with_bt601_in_either_range() {
        let yuv = |y: u8, u: u8, v: u8, range| {
            let mut source = frame(
                2,
                2,
                PixelFormat::Yuv420p8,
                vec![
                    Plane {
                        data: vec![y; 4],
                        stride: 2,
                    },
                    Plane {
                        data: vec![u],
                        stride: 1,
                    },
                    Plane {
                        data: vec![v],
                        stride: 1,
                    },
                ],
            );
            source.color_range = range;
            Thumbnail::new(&source, Orientation::TopLeft).unwrap().rgb[..3].to_vec()
        };
        assert_eq!(yuv(16, 128, 128, ColorRange::Limited), [0, 0, 0]);
        assert_eq!(yuv(235, 128, 128, ColorRange::Limited), [255, 255, 255]);
        // BT.601 limited-range red, green and blue, as rounded 8-bit YCbCr.
        assert_eq!(yuv(81, 90, 240, ColorRange::Limited), [254, 0, 0]);
        assert_eq!(yuv(145, 54, 34, ColorRange::Limited), [0, 255, 1]);
        assert_eq!(yuv(41, 240, 110, ColorRange::Limited), [0, 0, 255]);
        assert_eq!(yuv(128, 128, 128, ColorRange::Full), [128, 128, 128]);
        assert_eq!(yuv(76, 85, 255, ColorRange::Full), [254, 0, 0]);
    }

    #[test]
    fn capture_keeps_the_chosen_frame_or_the_last_one_before_it() {
        let solid = |value: u8| {
            frame(
                2,
                2,
                PixelFormat::Gray8,
                vec![Plane {
                    data: vec![value; 4],
                    stride: 2,
                }],
            )
        };
        let frames: Vec<VideoFrame> = (0..8).map(|index| solid(index * 10)).collect();
        let offer = |capture: &mut CoverCapture, count: usize| {
            for (index, frame) in frames.iter().take(count).enumerate() {
                capture.offer(
                    index as u64,
                    FrameSource::Cpu(CpuFrameSource {
                        frame,
                        orientation: Orientation::TopLeft,
                    }),
                );
            }
        };
        let first_pixel = |capture: &CoverCapture| {
            capture
                .cover_art()
                .map(|cover| decode_png(&cover.data).2[0])
        };

        let mut default = CoverCapture::new(CoverSource::default());
        offer(&mut default, 8);
        assert_eq!(first_pixel(&default), Some(40));

        let mut short = CoverCapture::new(CoverSource::Frame(6));
        offer(&mut short, 3);
        assert_eq!(first_pixel(&short), Some(20));

        let mut none = CoverCapture::new(CoverSource::None);
        offer(&mut none, 8);
        assert_eq!(first_pixel(&none), None);

        let empty = CoverCapture::new(CoverSource::default());
        assert_eq!(first_pixel(&empty), None);
    }
}
