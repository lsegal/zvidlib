//! The `av1_encode` bench support: `zvidlib-bench-support`'s shared helpers, bound
//! to the dispatch sites this package's benchmarks exercise.
// Each bench target of this package compiles this module and uses only what
// its own groups need, so unused-here is not dead.
#![allow(dead_code)]

zvidlib_bench_support::bench_support!(zvidlib_av1::simd_sites, zvidlib_color::simd_sites);
