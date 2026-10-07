//! Vorbis I encoder: a faithful, dependency-free port of the libvorbis 1.3.7
//! VBR (quality mode) encoder.
//!
//! The port follows libvorbis structurally (one module per C file, functions
//! named after the C ones) and numerically (C `float` is `f32`, C `double` is
//! `f64`, the operation order and the C promotions are preserved), so that
//! its output is bit-identical to libvorbis for the same input, quality and
//! write sizes.
//!
//! Reference semantics: libvorbis as built on LP64 POSIX systems (Linux,
//! macOS, BSD; e.g. distribution and ffmpeg builds). Windows builds of
//! libvorbis differ in two places, both of which the port does *not* copy:
//! os.h turns `rint()` into `floor(x+.5f)` there (differs on exact `.5` ties
//! in floor1's line fit, so occasional packets differ), and the 32-bit Windows
//! `long` makes scales.h `toBARK` overflow for sample rates above ~92.7 kHz.
//! See `os::rint` and `psy::to_bark_long`.
//!
//! Transcendental functions (`exp`, `ln`, `sin`, `cos`, `atan`) are only used
//! to build lookup tables, and their results are rounded to `f32` or
//! truncated to integers before they influence the bitstream, so last-ulp
//! differences between platform math libraries do not change the output in
//! practice (verified against the C build on this host: every tested
//! configuration is bit-identical).
//!
//! Supported: 1 or 2 channels at every sample rate libvorbis accepts for
//! them (all `setup_list` templates of vorbisenc.c except 5.1 surround), and
//! the full VBR quality range `-0.1..=1.0`.
//! Not supported: bitrate-managed (ABR/CBR) encoding and more than 2
//! channels.
//!
//! ```text
//! C libvorbis                               this module
//! vorbis_encode_init_vbr                    VorbisEncoder::new
//! vorbis_analysis_headerout                 VorbisEncoder::headers
//! vorbis_analysis_buffer/_wrote(n)+drain    VorbisEncoder::encode
//! vorbis_analysis_wrote(0)+drain            VorbisEncoder::finish
//! ```
//!
//! libvorbis is Copyright (c) 2002-2020 Xiph.org Foundation and is
//! distributed under the BSD-3-Clause license; this port and the generated
//! tables are derived from it under the same terms.

// Only the `simd` kernels may use `unsafe`; everything else stays safe code.
#![deny(unsafe_code)]

mod bitpack;
mod block;
mod codebook;
mod envelope;
mod floor1;
mod info;
mod lpc;
mod mapping0;
mod mdct;
mod os;
mod psy;
mod res0;
mod setup;
#[allow(unsafe_code)]
#[doc(hidden)]
pub mod simd;
mod smallft;
mod tables;
#[cfg(test)]
#[rustfmt::skip]
mod test_fixtures;
#[cfg(test)]
mod tests;
mod window;

use std::fmt;

/// Encoder error (invalid configuration or API misuse); the payload is a
/// human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VorbisEncoderError(pub String);

impl fmt::Display for VorbisEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vorbis encoder: {}", self.0)
    }
}

impl std::error::Error for VorbisEncoderError {}

/// Shorthand for an `Err(VorbisEncoderError)`.
fn err<T>(msg: impl Into<String>) -> Result<T, VorbisEncoderError> {
    Err(VorbisEncoderError(msg.into()))
}

/// Largest number of frames handed to the analysis stage in one
/// `vorbis_analysis_wrote` call once the stream start has been extrapolated;
/// longer `encode` inputs are split, which keeps the internal PCM buffer
/// bounded.
///
/// libvorbis' output depends on the write chunking in exactly one place: the
/// backwards LPC extrapolation of the stream start (block.c
/// `_preextrapolate_helper`) uses every sample present when the buffered
/// amount first exceeds one long block. Until that has happened each `encode`
/// call is therefore passed on whole (exactly like one
/// `vorbis_analysis_wrote`), and splitting afterwards is invisible in the
/// output: the packets are bit-identical to libvorbis fed with the same
/// sequence of write sizes, whatever those sizes are.
pub const MAX_WRITE_FRAMES: usize = 4096;

/// libvorbis 1.3.7's `ENCODE_VENDOR_STRING`, for callers that want
/// byte-identical comment headers.
#[cfg(test)]
pub const LIBVORBIS_VENDOR: &str = "Xiph.Org libVorbis I 20200704 (Reducing Environment)";

/// Streaming Vorbis encoder (libvorbis 1.3.7 VBR port).
///
/// Memory use is bounded: besides the fixed lookup tables (a few hundred KB),
/// the encoder buffers about three long blocks of PCM plus at most one
/// `encode` call's worth of input before the stream start has been
/// extrapolated (see [`MAX_WRITE_FRAMES`]).
pub struct VorbisEncoder {
    analysis: block::Analysis,
    channels: usize,
    finished: bool,
}

impl fmt::Debug for VorbisEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ci = &self.analysis.backend.ci;
        f.debug_struct("VorbisEncoder")
            .field("sample_rate", &ci.rate)
            .field("channels", &self.channels)
            .field("blocksizes", &ci.blocksizes)
            .field("nominal_bitrate", &ci.bitrate_nominal)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl VorbisEncoder {
    /// Port of `vorbis_encode_init_vbr` + `vorbis_analysis_init`.
    ///
    /// `quality` is libvorbis' `base_quality` in `-0.1..=1.0` (oggenc's
    /// `-q` divided by 10).
    pub fn new(sample_rate: u32, channels: u16, quality: f32) -> Result<Self, VorbisEncoderError> {
        if !(1..=2).contains(&channels) {
            return err(format!("{channels} channels not supported (1 or 2 only)"));
        }
        if !quality.is_finite() || !(-0.1..=1.0).contains(&quality) {
            return err(format!("quality {quality} outside -0.1..=1.0"));
        }
        let Ok(rate) = i32::try_from(sample_rate) else {
            return err(format!("sample rate {sample_rate} not supported"));
        };
        let Some(ci) = setup::encode_init_vbr(i32::from(channels), rate, quality) else {
            return err(format!(
                "no libvorbis setup template for {sample_rate} Hz, {channels} channel(s), quality {quality}"
            ));
        };
        let Some(analysis) = block::Analysis::new(ci) else {
            return err("internal codebook setup failure");
        };
        Ok(VorbisEncoder {
            analysis,
            channels: usize::from(channels),
            finished: false,
        })
    }

    /// The three header packets (identification, comment, setup), packed as
    /// libvorbis' `vorbis_analysis_headerout` does with an empty comment list.
    /// Pass [`LIBVORBIS_VENDOR`] to get libvorbis' exact comment header.
    pub fn headers(&self, vendor: &str) -> [Vec<u8>; 3] {
        let ci = &self.analysis.backend.ci;
        [
            info::pack_info(ci),
            info::pack_comment(vendor),
            info::pack_books(ci),
        ]
    }

    /// Feeds interleaved PCM (`frames * channels` samples, nominally in
    /// -1.0..=1.0) and returns every audio packet completed so far with its
    /// granule position. An empty slice is a no-op (unlike
    /// `vorbis_analysis_wrote(0)`, end of stream is signalled by `finish`).
    pub fn encode(
        &mut self,
        interleaved: &[f32],
    ) -> Result<Vec<(Vec<u8>, u64)>, VorbisEncoderError> {
        if self.finished {
            return err("encode() called after finish()");
        }
        if interleaved.len() % self.channels != 0 {
            return err(format!(
                "interleaved buffer length {} is not a multiple of {} channels",
                interleaved.len(),
                self.channels
            ));
        }
        let mut out = Vec::new();
        let mut rest = interleaved;
        while !rest.is_empty() {
            let take = if self.analysis.preextrapolated() {
                rest.len().min(MAX_WRITE_FRAMES * self.channels)
            } else {
                rest.len()
            };
            let (chunk, tail) = rest.split_at(take);
            self.analysis.write_interleaved(chunk);
            self.analysis.drain(&mut out);
            rest = tail;
        }
        Ok(out)
    }

    /// Ends the stream (`vorbis_analysis_wrote(0)`: LPC extrapolation past the
    /// end, last packet's granule = total input frames) and returns the
    /// remaining packets. The last packet is the end-of-stream packet.
    pub fn finish(&mut self) -> Result<Vec<(Vec<u8>, u64)>, VorbisEncoderError> {
        if self.finished {
            return err("finish() called twice");
        }
        self.finished = true;
        self.analysis.wrote_eof();
        let mut out = Vec::new();
        self.analysis.drain(&mut out);
        Ok(out)
    }

    /// (short, long) block sizes in samples.
    #[cfg(test)]
    pub fn blocksizes(&self) -> (usize, usize) {
        let bs = self.analysis.backend.ci.blocksizes;
        (bs[0] as usize, bs[1] as usize)
    }

    /// The nominal bitrate libvorbis writes into the identification header.
    #[cfg(test)]
    pub fn nominal_bitrate(&self) -> i32 {
        self.analysis.backend.ci.bitrate_nominal
    }
}

/// The VBR quality that libvorbis' bitrate-driven setup
/// (`vorbis_encode_init`/`vorbis_encode_setup_managed` with only a nominal
/// bitrate) maps `bits_per_second` to: the template's `rate_mapping` table is
/// interpolated exactly as `get_setup_template` does, and the resulting
/// setting is converted back through the template's `quality_mapping`.
/// Returns `None` where libvorbis fails (unsupported rate/channels, bitrate
/// outside the template's range, or rates such as > 50 kHz whose templates
/// have no bitrate table).
pub fn quality_for_nominal_bitrate(
    sample_rate: u32,
    channels: u16,
    bits_per_second: u32,
) -> Option<f32> {
    if !(1..=2).contains(&channels) || bits_per_second == 0 || sample_rate == 0 {
        return None;
    }
    setup::quality_for_bitrate(
        i64::from(sample_rate),
        i64::from(channels),
        f64::from(bits_per_second),
    )
}
