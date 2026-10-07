//! The AV1 coding tools zvidlib's AV1 encoder and its bounded reference
//! decoders share: entropy coding and CDFs, transforms, intra prediction,
//! motion compensation and the in-loop filters, with the runtime-dispatched
//! SIMD kernels that vectorize them.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports its public items under the paths documented there. Depend
//! on `zvidlib` rather than on this crate directly.

#[doc(hidden)]
pub mod av1_cdf;
mod av1_cdf_tables;
/// The encoder-side transform and coefficient-context pieces the SIMD kernels
/// in [`av1_simd`] vectorize; the rest of the encoder is `zvidlib-av1-encoder`.
#[doc(hidden)]
pub mod av1_encoder {
    pub mod cdf;
    pub mod transform;
    pub mod wht;
}
pub mod av1_entropy;
pub mod av1_filters;
pub mod av1_inter_decoder;
pub mod av1_intra;
pub mod av1_intra_decoder;
pub mod av1_intra_pred;
pub mod av1_mc;
pub mod av1_simd;

#[allow(unused_imports)]
use zvidlib_av1_syntax as av1;
#[allow(unused_imports)]
use zvidlib_av1_syntax::*;
#[allow(unused_imports)]
use zvidlib_color::yuv_to_rgba;
#[allow(unused_imports)]
use zvidlib_core::*;
// The containers and conformance harness the tests read their fixtures with.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::*;

pub use av1_encoder::transform::forward_transform;
pub use av1_entropy::{AV1_CDF_MAX, Av1SymbolDecoder, validate_cdf};
pub use av1_filters::{
    CdefStrength, FilmGrainParams, FilterFrame, FilterPlane, LoopFilterParams, MatrixCoefficients,
    RestorationUnit, TxSizeGrid, apply_film_grain, apply_restoration_unit, cdef_frame,
    convert_to_rgba8, deblock_frame, super_resolution_upscale,
};
pub use av1_inter_decoder::Av1InterDecoder;
pub use av1_intra::{
    Av1IntraBlock, Av1IntraFrame, Av1IntraMode, Av1TxType, Tx1d, get_ac_quant, get_dc_quant,
    inverse_transform, inverse_wht_4x4,
};
pub use av1_intra_decoder::{decode_av1_lossless_intra, decode_av1_lossless_intra_with_tx_sizes};
pub use av1_intra_pred::{
    Av1IntraSimd, SmoothMode, add_residual_row, av1_intra_simd, directional_row, paeth_row,
    smooth_row, sum_samples,
};
pub use av1_simd::SimdIsa;
