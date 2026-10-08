//! VP9 bitstream inspection shared by zvidlib's VP9 decoder and encoder and its
//! platform hardware backends: the uncompressed-header reader, superframe and
//! frame inspection, the decoded picture and its `Rgba8` conversion, and VP9
//! level and `vpcC` selection.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib).
//! Depend on `zvidlib` rather than on this crate directly.

// The processes below follow libvpx's C, which indexes several parallel
// arrays with one loop variable; keeping that shape makes each function
// checkable line by line against the code it reproduces.
#![allow(clippy::needless_range_loop)]

#[doc(hidden)]
pub mod bits;
#[doc(hidden)]
pub mod chunk;
mod level;
mod picture;

pub use chunk::{ChunkInspector, FrameShape, chunk_frames};
pub use level::{key_frame_vpcc, pick_level, vpcc_box, vpcc_from_key_frame};
pub use picture::{matrix_for, picture_to_rgba};

use bits::BitReader;
use zvidlib_core::{Error, ErrorKind, Result};

#[doc(hidden)]
pub fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

#[doc(hidden)]
pub fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

#[doc(hidden)]
pub fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

/// A shown picture, as the output process produces it.
#[derive(Clone, Debug)]
pub struct DecodedPicture {
    pub width: usize,
    pub height: usize,
    /// Y, U and V, tightly packed; the chroma planes are subsampled 2x2.
    pub planes: [Vec<u8>; 3],
    /// `color_space` of the uncompressed header (`CS_BT_601` is 1,
    /// `CS_BT_709` 2 and so on).
    pub color_space: u8,
    pub full_range: bool,
}

#[doc(hidden)]
pub fn read_sync_code(reader: &mut BitReader) -> Result<()> {
    if reader.literal(24)? != 0x49_83_42 {
        return Err(malformed("VP9 frame sync code is invalid"));
    }
    Ok(())
}

#[doc(hidden)]
pub fn read_render_size(reader: &mut BitReader) -> Result<()> {
    // The render size is presentation metadata; decoding ignores it.
    if reader.bit()? {
        reader.literal(32)?;
    }
    Ok(())
}

/// The color range a chunk's first frame signals: `Some(full_range)` when
/// that frame is a profile 0 key frame, or an intra-only frame (which profile
/// 0 makes studio range), and `None` for any other frame or for data that
/// does not parse. A track's first sample is a key frame, so this is the
/// range of its first picture, read without decoding it.
#[doc(hidden)]
pub fn chunk_full_range(data: &[u8]) -> Option<bool> {
    let first = match superframe_index(data).ok()? {
        Some(sizes) => data.get(..*sizes.first()?)?,
        None => data,
    };
    let mut reader = BitReader::new(first);
    if reader.literal(2).ok()? != 2 || reader.literal(2).ok()? != 0 || reader.bit().ok()? {
        // Not a frame marker, not profile 0, or show_existing_frame.
        return None;
    }
    let key_frame = !reader.bit().ok()?;
    let show_frame = reader.bit().ok()?;
    let error_resilient = reader.bit().ok()?;
    if key_frame {
        if reader.literal(24).ok()? != 0x49_83_42 || reader.literal(3).ok()? == 7 {
            return None;
        }
        return reader.bit().ok();
    }
    let intra_only = !show_frame && reader.bit().ok()?;
    if !error_resilient {
        reader.literal(2).ok()?;
    }
    intra_only.then_some(false)
}

/// Parses a superframe index (Annex B.3), returning the frame sizes it
/// lists, or `None` when the chunk holds a single frame.
#[doc(hidden)]
pub fn superframe_index(data: &[u8]) -> Result<Option<Vec<usize>>> {
    if data.is_empty() {
        return Ok(None);
    }
    let marker = data[data.len() - 1];
    if marker & 0xe0 != 0xc0 {
        return Ok(None);
    }
    let frames = usize::from(marker & 0x7) + 1;
    let magnitude = usize::from((marker >> 3) & 0x3) + 1;
    let index_size = 2 + magnitude * frames;
    if data.len() < index_size || data[data.len() - index_size] != marker {
        return Err(malformed("VP9 superframe index is invalid"));
    }
    let mut bytes = &data[data.len() - index_size + 1..];
    let mut sizes = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut size = 0usize;
        for j in 0..magnitude {
            size |= usize::from(bytes[j]) << (j * 8);
        }
        bytes = &bytes[magnitude..];
        sizes.push(size);
    }
    Ok(Some(sizes))
}
