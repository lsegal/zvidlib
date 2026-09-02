//! Scratch A/B harness: interleaved scalar-vs-vector forward DCT timing.
use std::hint::black_box;
use std::time::Instant;
use zvidlib::av1_intra::Av1TxType;
use zvidlib::simd::{self, SimdIsa};

fn slab(residual: &[i32], size: usize, blocks: usize, tx: Av1TxType) -> u64 {
    let mut digest = 0u64;
    for _ in 0..blocks {
        let c = zvidlib::forward_transform(black_box(residual), size, tx);
        digest ^= c[0] as u64;
    }
    digest
}

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(15);
    let sizes: Vec<usize> = std::env::args().nth(2)
        .map(|a| a.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16, 32]);
    let isas = simd::available();
    println!("available: {isas:?}  rounds={rounds}");
    for size in sizes {
        let residual: Vec<i32> = (0..size * size).map(|i| (i as i32 * 53) % 511 - 255).collect();
        let blocks = (1920 / size) * (1080 / size);
        let mut best = vec![f64::MAX; isas.len()];
        // Warm up.
        for &isa in &isas { simd::set_override(Some(isa)); black_box(slab(&residual, size, blocks, Av1TxType::DctDct)); }
        for _ in 0..rounds {
            for (i, &isa) in isas.iter().enumerate() {
                simd::set_override(Some(isa));
                let start = Instant::now();
                black_box(slab(&residual, size, blocks, Av1TxType::DctDct));
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if elapsed < best[i] { best[i] = elapsed; }
            }
        }
        simd::set_override(None);
        let scalar = best[0];
        let cells: Vec<String> = isas.iter().zip(&best)
            .map(|(isa, ms)| format!("{isa:?} {ms:8.3} ms ({:.2}x)", scalar / ms))
            .collect();
        println!("{size:>2}x{size:<2} | {}", cells.join(" | "));
    }
}
