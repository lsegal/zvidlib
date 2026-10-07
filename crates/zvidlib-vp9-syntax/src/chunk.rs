//! What a hardware decoder needs to know about a VP9 chunk before handing it to the platform:
//! whether it shows a frame, and that frame's size and colour.
//!
//! The platform decoders decode the chunk; this reads only the start of each frame's
//! uncompressed header (section 6.2), as far as the frame size, and follows the reference slots
//! so that `show_existing_frame` and a size taken from a reference resolve as the software
//! decoder resolves them. It accepts and refuses exactly the profile 0 streams [`Decoder`]
//! does at that depth.
//!
//! [`Decoder`]: super::Decoder

use super::bits::BitReader;
use super::{malformed, read_render_size, read_sync_code, superframe_index, unsupported};
use crate::Result;

/// The size and colour of a decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameShape {
    pub width: usize,
    pub height: usize,
    /// `color_space` of the uncompressed header, as [`DecodedPicture`](super::DecodedPicture)
    /// reports it.
    pub color_space: u8,
    pub full_range: bool,
}

/// What the start of one frame's header says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameInfo {
    /// The frame's size and colour, or for `show_existing_frame` those of the frame it shows.
    pub shape: FrameShape,
    pub shown: bool,
    /// Whether the frame is a key frame, which refreshes every reference slot.
    pub key_frame: bool,
    /// The reference slot a `show_existing_frame` header shows again. Such a header decodes
    /// nothing and refreshes no slot.
    pub existing: Option<usize>,
    /// The reference slots the decoded frame replaces.
    pub refresh_frame_flags: u8,
}

/// Follows a VP9 stream chunk by chunk, as a [`Decoder`](super::Decoder) would, without
/// decoding it.
#[derive(Debug, Default)]
pub struct ChunkInspector {
    refs: [Option<FrameShape>; 8],
    color_space: u8,
    full_range: bool,
    have_key_frame: bool,
}

impl ChunkInspector {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Reads one chunk (a frame or a superframe) and returns the shape of the frame it shows,
    /// which is that of its last frame when that frame is shown.
    #[doc(hidden)]
    pub fn inspect(&mut self, data: &[u8]) -> Result<Option<FrameShape>> {
        if data.is_empty() {
            return Err(malformed("VP9 sample is empty"));
        }
        let mut shown = None;
        for frame in chunk_frames(data)? {
            let info = self.inspect_frame(frame)?;
            shown = info.shown.then_some(info.shape);
        }
        Ok(shown)
    }

    /// Reads one frame of a chunk and records the reference slots it refreshes.
    pub fn inspect_frame(&mut self, data: &[u8]) -> Result<FrameInfo> {
        let mut reader = BitReader::new(data);
        if reader.literal(2)? != 2 {
            return Err(malformed("VP9 frame marker is invalid"));
        }
        let mut profile = reader.literal(1)? | (reader.literal(1)? << 1);
        if profile > 2 {
            profile += reader.literal(1)?;
        }
        if profile != 0 {
            return Err(unsupported(format!(
                "VP9 profile {profile} is not supported; the decoder implements profile 0 (8-bit 4:2:0)"
            )));
        }
        if reader.bit()? {
            // show_existing_frame
            let index = reader.literal(3)? as usize;
            let shape = self.refs[index].ok_or_else(|| {
                malformed("VP9 show_existing_frame names an empty reference slot")
            })?;
            return Ok(FrameInfo {
                shape,
                shown: true,
                key_frame: false,
                existing: Some(index),
                refresh_frame_flags: 0,
            });
        }
        let key_frame = !reader.bit()?;
        let show_frame = reader.bit()?;
        let error_resilient = reader.bit()?;
        let (refresh_frame_flags, width, height) = if key_frame {
            read_sync_code(&mut reader)?;
            self.color_space = reader.literal(3)? as u8;
            if self.color_space == 7 {
                return Err(unsupported(
                    "VP9 sRGB (4:4:4) color is not supported in profile 0",
                ));
            }
            self.full_range = reader.bit()?;
            let (width, height) = read_frame_size(&mut reader)?;
            self.have_key_frame = true;
            (0xff, width, height)
        } else {
            let intra_only = !show_frame && reader.bit()?;
            if !error_resilient {
                reader.literal(2)?; // reset_frame_context
            }
            if intra_only {
                read_sync_code(&mut reader)?;
                // Profile 0 intra-only frames are BT.601 studio-range 4:2:0.
                self.color_space = 1;
                self.full_range = false;
                let refresh_frame_flags = reader.literal(8)? as u8;
                let (width, height) = read_frame_size(&mut reader)?;
                self.have_key_frame = true;
                (refresh_frame_flags, width, height)
            } else {
                if !self.have_key_frame {
                    return Err(malformed(
                        "VP9 inter frame received before a key frame or intra-only frame",
                    ));
                }
                let refresh_frame_flags = reader.literal(8)? as u8;
                let mut references = [FrameShape {
                    width: 0,
                    height: 0,
                    color_space: 0,
                    full_range: false,
                }; 3];
                for reference in &mut references {
                    let index = reader.literal(3)? as usize;
                    *reference = self.refs[index]
                        .ok_or_else(|| malformed("VP9 frame references an empty reference slot"))?;
                    reader.bit()?; // ref_frame_sign_bias
                }
                let mut size = None;
                for reference in &references {
                    if reader.bit()? {
                        size = Some((reference.width, reference.height));
                        break;
                    }
                }
                let (width, height) = match size {
                    Some(size) => {
                        read_render_size(&mut reader)?;
                        size
                    }
                    None => read_frame_size(&mut reader)?,
                };
                (refresh_frame_flags, width, height)
            }
        };
        let shape = FrameShape {
            width,
            height,
            color_space: self.color_space,
            full_range: self.full_range,
        };
        for (slot, reference) in self.refs.iter_mut().enumerate() {
            if refresh_frame_flags & (1 << slot) != 0 {
                *reference = Some(shape);
            }
        }
        Ok(FrameInfo {
            shape,
            shown: show_frame,
            key_frame,
            existing: None,
            refresh_frame_flags,
        })
    }
}

/// The frames of a chunk: those its superframe index lists, or the whole chunk when it has none.
pub fn chunk_frames(data: &[u8]) -> Result<Vec<&[u8]>> {
    let Some(sizes) = superframe_index(data)? else {
        return Ok(vec![data]);
    };
    let mut frames = Vec::with_capacity(sizes.len());
    let mut offset = 0;
    for size in sizes {
        let end = offset + size;
        if size == 0 || end > data.len() {
            return Err(malformed(
                "VP9 superframe index names an invalid frame size",
            ));
        }
        frames.push(&data[offset..end]);
        offset = end;
    }
    Ok(frames)
}

fn read_frame_size(reader: &mut BitReader) -> Result<(usize, usize)> {
    let width = reader.literal(16)? as usize + 1;
    let height = reader.literal(16)? as usize + 1;
    read_render_size(reader)?;
    Ok((width, height))
}
