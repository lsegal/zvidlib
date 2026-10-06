//! Static encoder data converted from libvorbis 1.3.7 by `tools/gen_tables.py`.
//!
//! The `gen_*` modules are generated; do not edit them by hand. Only objects
//! reachable from the supported setup templates (every `setup_list` entry of
//! vorbisenc.c except the 5.1 surround one) are converted, and the
//! bitrate-managed-only books are left out.

pub mod types;

#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]
pub mod gen_books_floor;
#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]
pub mod gen_books_stereo;
#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]
pub mod gen_books_uncoupled;
#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]
pub mod gen_misc;
#[rustfmt::skip]
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]
pub mod gen_modes;
