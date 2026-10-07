//! Port of window.c `_vorbis_apply_window` (the `vwin*` tables themselves are
//! generated into `tables::gen_misc`).

use super::tables::gen_misc::{
    VWIN64, VWIN128, VWIN256, VWIN512, VWIN1024, VWIN2048, VWIN4096, VWIN8192,
};

/// `vwin[n]`: the rising half of the Vorbis power-sine window for block size
/// `64 << n`.
pub(crate) fn window_table(n: usize) -> &'static [f32] {
    match n {
        0 => &VWIN64,
        1 => &VWIN128,
        2 => &VWIN256,
        3 => &VWIN512,
        4 => &VWIN1024,
        5 => &VWIN2048,
        6 => &VWIN4096,
        _ => &VWIN8192,
    }
}

/// Port of window.c `_vorbis_apply_window`.
pub(crate) fn apply_window(
    d: &mut [f32],
    winno: &[usize; 2],
    blocksizes: &[i32; 2],
    lw: usize,
    w: usize,
    nw: usize,
) {
    let lw = if w != 0 { lw } else { 0 };
    let nw = if w != 0 { nw } else { 0 };

    let window_lw = window_table(winno[lw]);
    let window_nw = window_table(winno[nw]);

    let n = blocksizes[w] as usize;
    let ln = blocksizes[lw] as usize;
    let rn = blocksizes[nw] as usize;

    let leftbegin = n / 4 - ln / 4;
    let leftend = leftbegin + ln / 2;

    let rightbegin = n / 2 + n / 4 - rn / 4;
    let rightend = rightbegin + rn / 2;

    for v in &mut d[..leftbegin] {
        *v = 0.;
    }
    for (v, &wv) in d[leftbegin..leftend].iter_mut().zip(window_lw) {
        *v *= wv;
    }
    for (i, v) in d[rightbegin..rightend].iter_mut().enumerate() {
        *v *= window_nw[rn / 2 - 1 - i];
    }
    for v in &mut d[rightend..n] {
        *v = 0.;
    }
}
