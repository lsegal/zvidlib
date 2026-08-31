//! Shared scaffolding for zvidlib's criterion benchmarks.
//!
//! Everything a bench target needs that is not the measurement itself lives
//! here, so individual benches stay a list of workloads rather than a pile of
//! fixture plumbing:
//!
//! * [`fixtures`] loads the bundled MP4 and the checked-in elementary streams
//!   **once per process** and hands out borrowed, already-demuxed samples, so
//!   the per-iteration cost criterion measures is codec work and nothing else.
//! * [`synth`] generates deterministic YUV420 frame sequences, so encoder-side
//!   benchmarks have input without decoding something first.
//! * [`harness`] runs a bench group once per instruction set
//!   [`zvidlib::simd::available`] reports, names the groups `<codec>/<isa>`,
//!   reports throughput in frames and megapixels per second, and guards every
//!   comparison with a scalar-vs-SIMD equality check.
//!
//! The guard in [`harness::assert_bit_exact_across_isas`] is not optional
//! ceremony. Every vector backend in the crate is documented as bit-exact with
//! its scalar reference, and a "SIMD speedup" produced by a kernel that
//! silently diverged would be worse than having no benchmark at all.

// Each bench target uses the parts of this module it needs; the rest would
// otherwise warn under `-D warnings` in `cargo clippy --all-targets`.
#![allow(dead_code)]

pub mod fixtures;
pub mod harness;
pub mod synth;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

/// Drives a future to completion on the calling thread.
///
/// zvidlib's demuxing entry points are `async` but never actually suspend on
/// an in-memory source, so a no-op waker and a spin loop are enough and keep
/// the benches free of an async runtime dependency.
pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// FNV-1a over a byte buffer.
///
/// Used only to compare one backend's output against another's, so it needs to
/// be cheap and order-sensitive, not cryptographic.
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
