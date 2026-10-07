//! The `vorbis_encode` bench support: `zvidlib-bench-support`'s shared helpers, bound
//! to the dispatch sites this package's benchmarks exercise.
// Each bench target of this package compiles this module and uses only what
// its own groups need, so unused-here is not dead.
#![allow(dead_code)]

zvidlib_bench_support::bench_support!(
    zvidlib_vorbis_encoder::simd_sites,
    zvidlib_vorbis_decoder::simd_sites
);
