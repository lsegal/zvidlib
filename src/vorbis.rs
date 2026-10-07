//! Vorbis audio: decoder configuration, packet timing and native decoding.
//!
//! A Vorbis stream is described by three header packets - identification,
//! comment and setup - which Matroska and WebM carry Xiph-laced as
//! `CodecPrivate` and a `WebCodecs` `AudioDecoderConfig.description` takes in
//! the same form. [`VorbisConfig`] parses and writes them, and reads from the
//! setup header the one thing a container does not record exactly: how many
//! samples each audio packet decodes to.
//!
//! Decoding runs on Symphonia's pure-Rust Vorbis decoder, and encoding on
//! zvidlib's own port of the libvorbis encoder, so both work on every target
//! zvidlib builds for, `wasm32` included.

#[doc(inline)]
pub use zvidlib_vorbis_decoder::{NativeVorbisDecoder, VORBIS_PREROLL_PACKETS, VorbisConfig};
#[doc(inline)]
pub use zvidlib_vorbis_encoder::native_vorbis_audio_encoder_factory;
