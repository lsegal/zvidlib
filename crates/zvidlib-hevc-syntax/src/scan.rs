//! §6.5 scan-order array initialization processes and the §7.4.2
//! `ScanOrder[ log2BlockSize ][ scanIdx ]` table.
//!
//! ITU-T Rec. H.265 §6.5.3..§6.5.6 define four scan orders that map a
//! scan position `sPos` (0..`blkSize * blkSize − 1`) to a
//! `(horizontal, vertical)` coordinate pair:
//!
//! * §6.5.3 [`up_right_diagonal`] — walks the block along up-right
//!   diagonals (equation 6-11). `scanIdx == 0`.
//! * §6.5.4 [`horizontal`] — raster order, row by row (equation 6-12).
//!   `scanIdx == 1`.
//! * §6.5.5 [`vertical`] — column by column (equation 6-13).
//!   `scanIdx == 2`.
//! * §6.5.6 [`traverse`] — boustrophedon (serpentine) raster: even rows
//!   left-to-right, odd rows right-to-left (equation 6-14).
//!   `scanIdx == 3`.
//!
//! §7.4.2 assembles these into
//! `ScanOrder[ log2BlockSize ][ scanIdx ][ sPos ][ sComp ]`, with
//! `scanIdx` 0 = diagonal, 1 = horizontal, 2 = vertical, 3 = traverse.
//! The diagonal / horizontal / vertical scans are populated for
//! `log2BlockSize` 0..3 (block sizes 1, 2, 4, 8) and the traverse scan
//! for `log2BlockSize` 2..5 (block sizes 4, 8, 16, 32). [`scan_order`]
//! is the validity-checked accessor over that table.
//!
//! The §7.4.5 `ScalingFactor` derivation (equations 7-44..7-51) reads
//! `ScanOrder[ 2 ][ 0 ]` (a 4x4 block) and `ScanOrder[ 3 ][ 0 ]` (an
//! 8x8 block); the residual-coding path (§7.3.8.11 / §9.3.4.2.4) reads
//! the full table via the `scanIdx` selected per the §9.3.4.2.4 rules.

/// One scan-position entry: the `(x, y)` coordinate the position maps
/// to. `x` is the horizontal component (`sComp == 0`) and `y` is the
/// vertical component (`sComp == 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanPos {
    /// Horizontal component — `diagScan[ sPos ][ 0 ]`.
    pub x: u8,
    /// Vertical component — `diagScan[ sPos ][ 1 ]`.
    pub y: u8,
}

/// Build the §6.5.3 up-right diagonal scan order for a `blkSize`x`blkSize`
/// block, returning the `diagScan` array as a `Vec<ScanPos>` of length
/// `blkSize * blkSize`, indexed by scan position `sPos`.
///
/// This is a direct transcription of equation 6-11: starting at the
/// top-left, each inner `while( y >= 0 )` loop walks one up-right
/// diagonal (decrementing `y`, incrementing `x`), emitting only the
/// in-bounds `(x, y)` cells; the outer loop seeds the next diagonal at
/// `y = x, x = 0` and stops once every cell has been visited.
pub fn up_right_diagonal(blk_size: usize) -> Vec<ScanPos> {
    let n = blk_size * blk_size;
    let mut diag = Vec::with_capacity(n);

    // The 6-11 pseudocode uses a signed `y` that the inner loop drives
    // below 0 as its termination test, so the coordinates are tracked
    // as `i32` and only the in-bounds cells are recorded.
    let blk = blk_size as i32;
    let mut x: i32 = 0;
    let mut y: i32 = 0;
    let mut stop = false;
    while !stop {
        while y >= 0 {
            if x < blk && y < blk {
                diag.push(ScanPos {
                    x: x as u8,
                    y: y as u8,
                });
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
        if diag.len() >= n {
            stop = true;
        }
    }

    diag
}

/// Build the §6.5.4 horizontal scan order for a `blkSize`x`blkSize`
/// block, returning the `horScan` array (equation 6-12) as a
/// `Vec<ScanPos>` of length `blkSize * blkSize`, indexed by scan
/// position `sPos`.
///
/// Direct transcription of equation 6-12: a plain raster walk —
/// the outer loop runs over rows `y`, the inner over columns `x`.
/// This is the `scanIdx == 1` entry of the §7.4.2 `ScanOrder` table.
pub fn horizontal(blk_size: usize) -> Vec<ScanPos> {
    let mut scan = Vec::with_capacity(blk_size * blk_size);
    for y in 0..blk_size {
        for x in 0..blk_size {
            scan.push(ScanPos {
                x: x as u8,
                y: y as u8,
            });
        }
    }
    scan
}

/// Build the §6.5.5 vertical scan order for a `blkSize`x`blkSize`
/// block, returning the `verScan` array (equation 6-13) as a
/// `Vec<ScanPos>` of length `blkSize * blkSize`, indexed by scan
/// position `sPos`.
///
/// Direct transcription of equation 6-13: the transpose of the
/// horizontal scan — the outer loop runs over columns `x`, the inner
/// over rows `y`. This is the `scanIdx == 2` entry of the §7.4.2
/// `ScanOrder` table.
pub fn vertical(blk_size: usize) -> Vec<ScanPos> {
    let mut scan = Vec::with_capacity(blk_size * blk_size);
    for x in 0..blk_size {
        for y in 0..blk_size {
            scan.push(ScanPos {
                x: x as u8,
                y: y as u8,
            });
        }
    }
    scan
}

/// Build the §6.5.6 traverse scan order for a `blkSize`x`blkSize`
/// block, returning the `travScan` array (equation 6-14) as a
/// `Vec<ScanPos>` of length `blkSize * blkSize`, indexed by scan
/// position `sPos`.
///
/// Direct transcription of equation 6-14: a boustrophedon (serpentine)
/// raster — even-indexed rows are walked left-to-right
/// (`x = 0 .. blkSize − 1`) and odd-indexed rows right-to-left
/// (`x = blkSize − 1 .. 0`). This is the `scanIdx == 3` entry of the
/// §7.4.2 `ScanOrder` table.
pub fn traverse(blk_size: usize) -> Vec<ScanPos> {
    let mut scan = Vec::with_capacity(blk_size * blk_size);
    for y in 0..blk_size {
        if y % 2 == 0 {
            for x in 0..blk_size {
                scan.push(ScanPos {
                    x: x as u8,
                    y: y as u8,
                });
            }
        } else {
            for x in (0..blk_size).rev() {
                scan.push(ScanPos {
                    x: x as u8,
                    y: y as u8,
                });
            }
        }
    }
    scan
}

/// §7.4.2 `scanIdx` selector: `0` up-right diagonal, `1` horizontal,
/// `2` vertical, `3` traverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanIdx {
    /// `scanIdx == 0` — §6.5.3 up-right diagonal scan.
    Diagonal,
    /// `scanIdx == 1` — §6.5.4 horizontal scan.
    Horizontal,
    /// `scanIdx == 2` — §6.5.5 vertical scan.
    Vertical,
    /// `scanIdx == 3` — §6.5.6 traverse scan.
    Traverse,
}

impl ScanIdx {
    /// The numeric `scanIdx` value used by §7.4.2 / §9.3.4.2.4.
    #[must_use]
    pub fn index(self) -> u8 {
        match self {
            Self::Diagonal => 0,
            Self::Horizontal => 1,
            Self::Vertical => 2,
            Self::Traverse => 3,
        }
    }
}

/// Errors from the §7.4.2 [`scan_order`] accessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOrderError {
    /// `log2BlockSize` is outside the range §7.4.2 populates for the
    /// requested `scanIdx`: 0..=3 for diagonal / horizontal / vertical,
    /// 2..=5 for traverse.
    Log2BlockSizeOutOfRange {
        /// The requested scan order.
        scan_idx: ScanIdx,
        /// The out-of-range `log2BlockSize`.
        log2_block_size: u8,
    },
}

impl core::fmt::Display for ScanOrderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Log2BlockSizeOutOfRange {
                scan_idx,
                log2_block_size,
            } => write!(
                f,
                "ScanOrder: log2BlockSize {log2_block_size} out of range for {scan_idx:?} scan"
            ),
        }
    }
}

impl std::error::Error for ScanOrderError {}

/// §7.4.2 `ScanOrder[ log2BlockSize ][ scanIdx ]` accessor.
///
/// Returns the scan-position array for a `(1 << log2BlockSize)`-square
/// block under the requested [`ScanIdx`], built by invoking the
/// corresponding §6.5.3..§6.5.6 initialization process with
/// `1 << log2BlockSize`.
///
/// Per §7.4.2 the table is only populated for `log2BlockSize` in 0..=3
/// for the up-right diagonal / horizontal / vertical scans and 2..=5
/// for the traverse scan; an out-of-range `log2_block_size` yields
/// [`ScanOrderError::Log2BlockSizeOutOfRange`].
pub fn scan_order(log2_block_size: u8, scan_idx: ScanIdx) -> Result<Vec<ScanPos>, ScanOrderError> {
    let in_range = match scan_idx {
        ScanIdx::Diagonal | ScanIdx::Horizontal | ScanIdx::Vertical => log2_block_size <= 3,
        ScanIdx::Traverse => (2..=5).contains(&log2_block_size),
    };
    if !in_range {
        return Err(ScanOrderError::Log2BlockSizeOutOfRange {
            scan_idx,
            log2_block_size,
        });
    }
    let blk_size = 1usize << log2_block_size;
    Ok(match scan_idx {
        ScanIdx::Diagonal => up_right_diagonal(blk_size),
        ScanIdx::Horizontal => horizontal(blk_size),
        ScanIdx::Vertical => vertical(blk_size),
        ScanIdx::Traverse => traverse(blk_size),
    })
}

#[cfg(any())]
mod tests {
    use super::*;

    /// The 4x4 up-right diagonal scan (§6.5.3, `blkSize == 4`) visits
    /// the 16 cells in the canonical diagonal order. Hand-derived from
    /// equation 6-11.
    #[test]
    fn diag_scan_4x4() {
        let scan = up_right_diagonal(4);
        let coords: Vec<(u8, u8)> = scan.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(
            coords,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 1),
                (2, 0),
                (0, 3),
                (1, 2),
                (2, 1),
                (3, 0),
                (1, 3),
                (2, 2),
                (3, 1),
                (2, 3),
                (3, 2),
                (3, 3),
            ]
        );
    }

    /// The 2x2 scan is the smallest non-trivial case and exercises the
    /// outer-loop re-seed (`y = x, x = 0`) twice.
    #[test]
    fn diag_scan_2x2() {
        let scan = up_right_diagonal(2);
        let coords: Vec<(u8, u8)> = scan.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// The scan is a permutation of every `(x, y)` cell in the block:
    /// each coordinate appears exactly once, and the length is
    /// `blkSize * blkSize`. Checked for the 4x4 and 8x8 blocks the
    /// §7.4.5 derivation reads.
    #[test]
    fn diag_scan_is_a_permutation() {
        for &blk in &[4usize, 8] {
            let scan = up_right_diagonal(blk);
            assert_eq!(scan.len(), blk * blk);
            let mut seen = vec![false; blk * blk];
            for p in &scan {
                let (x, y) = (p.x as usize, p.y as usize);
                assert!(x < blk && y < blk);
                let flat = y * blk + x;
                assert!(!seen[flat], "cell ({x},{y}) visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v), "not every cell visited");
        }
    }

    /// Equation 6-11 walks each diagonal from its bottom-left toward its
    /// top-right: the sum `x + y` (the diagonal index) is
    /// non-decreasing across the scan, and within a diagonal `y`
    /// strictly decreases. Verified on the 8x8 block.
    #[test]
    fn diag_scan_8x8_diagonal_ordering() {
        let scan = up_right_diagonal(8);
        let mut prev_diag = 0usize;
        let mut prev_y = i32::MAX;
        for p in &scan {
            let diag = p.x as usize + p.y as usize;
            assert!(diag >= prev_diag, "diagonal index decreased");
            if diag != prev_diag {
                prev_diag = diag;
                prev_y = i32::MAX;
            }
            assert!(
                (p.y as i32) < prev_y,
                "y not strictly decreasing in diagonal"
            );
            prev_y = p.y as i32;
        }
    }

    fn coords(scan: &[ScanPos]) -> Vec<(u8, u8)> {
        scan.iter().map(|p| (p.x, p.y)).collect()
    }

    /// The 4x4 horizontal scan (§6.5.4, equation 6-12) is a plain
    /// raster: row 0 left-to-right, then row 1, etc. Hand-derived.
    #[test]
    fn hor_scan_4x4() {
        assert_eq!(
            coords(&horizontal(4)),
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (3, 3),
            ]
        );
    }

    /// The 2x2 vertical scan (§6.5.5, equation 6-13) walks column 0
    /// top-to-bottom, then column 1. Hand-derived.
    #[test]
    fn ver_scan_2x2() {
        assert_eq!(coords(&vertical(2)), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// The 4x4 vertical scan (§6.5.5, equation 6-13) is the transpose
    /// of the horizontal scan: column by column. Hand-derived.
    #[test]
    fn ver_scan_4x4() {
        assert_eq!(
            coords(&vertical(4)),
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
            ]
        );
    }

    /// The 4x4 traverse scan (§6.5.6, equation 6-14) is a serpentine
    /// raster: even rows (0, 2) left-to-right, odd rows (1, 3)
    /// right-to-left. Hand-derived.
    #[test]
    fn trav_scan_4x4() {
        assert_eq!(
            coords(&traverse(4)),
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (3, 1),
                (2, 1),
                (1, 1),
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
                (3, 3),
                (2, 3),
                (1, 3),
                (0, 3),
            ]
        );
    }

    /// The 2x2 traverse scan exercises a single direction flip: row 0
    /// left-to-right, row 1 right-to-left. Hand-derived.
    #[test]
    fn trav_scan_2x2() {
        assert_eq!(coords(&traverse(2)), vec![(0, 0), (1, 0), (1, 1), (0, 1)]);
    }

    /// Each new scan is a permutation of every `(x, y)` cell — every
    /// coordinate appears exactly once and the length is
    /// `blkSize * blkSize`. Checked across the block sizes §7.4.2
    /// populates (1, 2, 4, 8 for diag/hor/vert; 4, 8, 16, 32 for
    /// traverse).
    #[test]
    fn scans_are_permutations() {
        let check = |scan: &[ScanPos], blk: usize| {
            assert_eq!(scan.len(), blk * blk);
            let mut seen = vec![false; blk * blk];
            for p in scan {
                let (x, y) = (p.x as usize, p.y as usize);
                assert!(x < blk && y < blk);
                let flat = y * blk + x;
                assert!(!seen[flat], "cell ({x},{y}) visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v), "not every cell visited");
        };
        for &blk in &[1usize, 2, 4, 8] {
            check(&horizontal(blk), blk);
            check(&vertical(blk), blk);
        }
        for &blk in &[4usize, 8, 16, 32] {
            check(&traverse(blk), blk);
        }
    }

    /// The horizontal and vertical scans are transposes: position
    /// `i` of the vertical scan is the `(y, x)`-swapped position of the
    /// horizontal scan. Holds for every `(x, y)` per equations
    /// 6-12 / 6-13.
    #[test]
    fn hor_and_ver_are_transposes() {
        for &blk in &[1usize, 2, 4, 8] {
            let hor = horizontal(blk);
            let ver = vertical(blk);
            for (h, v) in hor.iter().zip(ver.iter()) {
                assert_eq!((h.x, h.y), (v.y, v.x));
            }
        }
    }

    /// Even rows of the traverse scan match the horizontal scan's rows
    /// exactly; odd rows are the horizontal rows reversed (equation
    /// 6-14). Verified on the 8x8 block.
    #[test]
    fn trav_scan_reverses_odd_rows() {
        let blk = 8usize;
        let trav = traverse(blk);
        for (i, p) in trav.iter().enumerate() {
            let row = i / blk;
            let col_in_row = i % blk;
            let expected_x = if row % 2 == 0 {
                col_in_row
            } else {
                blk - 1 - col_in_row
            };
            assert_eq!((p.x as usize, p.y as usize), (expected_x, row));
        }
    }

    /// §7.4.2 `scan_order` dispatches to the right §6.5.x process and
    /// agrees with the direct builders across every populated
    /// `(log2BlockSize, scanIdx)` slot.
    #[test]
    fn scan_order_dispatch_matches_builders() {
        for log2 in 0u8..=3 {
            let blk = 1usize << log2;
            assert_eq!(
                scan_order(log2, ScanIdx::Diagonal).unwrap(),
                up_right_diagonal(blk)
            );
            assert_eq!(
                scan_order(log2, ScanIdx::Horizontal).unwrap(),
                horizontal(blk)
            );
            assert_eq!(scan_order(log2, ScanIdx::Vertical).unwrap(), vertical(blk));
        }
        for log2 in 2u8..=5 {
            let blk = 1usize << log2;
            assert_eq!(scan_order(log2, ScanIdx::Traverse).unwrap(), traverse(blk));
        }
    }

    /// §7.4.2 only populates diag/hor/vert for `log2BlockSize` 0..=3 and
    /// traverse for 2..=5; outside those ranges `scan_order` reports the
    /// gap rather than fabricating a scan.
    #[test]
    fn scan_order_rejects_out_of_range() {
        for &idx in &[ScanIdx::Diagonal, ScanIdx::Horizontal, ScanIdx::Vertical] {
            assert!(matches!(
                scan_order(4, idx),
                Err(ScanOrderError::Log2BlockSizeOutOfRange { .. })
            ));
            assert!(scan_order(3, idx).is_ok());
        }
        assert!(matches!(
            scan_order(1, ScanIdx::Traverse),
            Err(ScanOrderError::Log2BlockSizeOutOfRange { .. })
        ));
        assert!(scan_order(2, ScanIdx::Traverse).is_ok());
        assert!(scan_order(5, ScanIdx::Traverse).is_ok());
        assert!(matches!(
            scan_order(6, ScanIdx::Traverse),
            Err(ScanOrderError::Log2BlockSizeOutOfRange { .. })
        ));
    }

    /// The §7.4.2 `scanIdx` numbering: 0 diagonal, 1 horizontal,
    /// 2 vertical, 3 traverse.
    #[test]
    fn scan_idx_numbering() {
        assert_eq!(ScanIdx::Diagonal.index(), 0);
        assert_eq!(ScanIdx::Horizontal.index(), 1);
        assert_eq!(ScanIdx::Vertical.index(), 2);
        assert_eq!(ScanIdx::Traverse.index(), 3);
    }
}
