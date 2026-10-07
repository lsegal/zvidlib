//! The HEVC engine zvidlib's HEVC decoder and encoder share: parameter sets,
//! CABAC, prediction, transforms, the in-loop filters and the sequence driver,
//! with the runtime-dispatched SIMD kernels that vectorize them. The encoder
//! reconstructs its pictures with the same tools, so it builds on this crate
//! rather than on the decoder (#635).
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports its public items under the paths documented there. Depend
//! on `zvidlib` rather than on this crate directly.

// internal — the HEVC engine; not part of the stable API
#[doc(hidden)]
pub mod engine;

#[allow(unused_imports)]
use zvidlib_core::*;
