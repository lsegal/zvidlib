//! Which coding tools a decode reached, so the conformance tests can show
//! that their fixtures exercise every path they are meant to check. Outside
//! tests `note` compiles to nothing.

pub(crate) const PALETTE: u32 = 1 << 0;
pub(crate) const INTRA_BLOCK_COPY: u32 = 1 << 1;
pub(crate) const FILTER_INTRA: u32 = 1 << 2;
pub(crate) const WEDGE_COMPOUND: u32 = 1 << 3;
pub(crate) const DIFF_WEIGHTED_COMPOUND: u32 = 1 << 4;
pub(crate) const DISTANCE_WEIGHTED_COMPOUND: u32 = 1 << 5;
pub(crate) const DUAL_FILTER: u32 = 1 << 6;
pub(crate) const FILM_GRAIN: u32 = 1 << 7;
pub(crate) const SUPERRES: u32 = 1 << 8;
pub(crate) const QUANTIZER_MATRIX: u32 = 1 << 9;
pub(crate) const MULTIPLE_TILES: u32 = 1 << 10;
pub(crate) const MULTIPLE_TILE_GROUPS: u32 = 1 << 11;
pub(crate) const CONTEXT_UPDATE_TILE_ID: u32 = 1 << 12;
pub(crate) const SUPERBLOCK_128: u32 = 1 << 13;
pub(crate) const SEGMENTATION: u32 = 1 << 14;
pub(crate) const DELTA_Q: u32 = 1 << 15;
pub(crate) const DELTA_LF: u32 = 1 << 16;
#[cfg(test)]
pub(crate) const ALL: u32 = (1 << 17) - 1;

#[cfg(test)]
thread_local! {
    static REACHED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Records that the current decode reached `tools`.
#[inline(always)]
pub(crate) fn note(tools: u32) {
    #[cfg(test)]
    REACHED.with(|reached| reached.set(reached.get() | tools));
    #[cfg(not(test))]
    let _ = tools;
}

/// The tools reached on this thread since the last call.
#[cfg(test)]
pub(crate) fn take() -> u32 {
    REACHED.with(|reached| reached.replace(0))
}
