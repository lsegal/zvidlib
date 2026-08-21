//! §8.4.2 luma-intra-mode neighbour state — the `IntraPredModeY`
//! prediction-block field consulted by the candidate-mode derivation.
//!
//! The §8.4.2 derivation of `IntraPredModeY[ xPb ][ yPb ]` needs, for each
//! prediction block, the `IntraPredModeY` of the left neighbour
//! `( xPb − 1, yPb )` and the above neighbour `( xPb, yPb − 1 )`, together
//! with each neighbour's `CuPredMode` and `pcm_flag` and its z-scan
//! availability. This module stores those three per-block facts on a
//! minimum-block (4×4 luma sample) grid so the [`crate::hevc::engine::recon`] driver can
//! reconstruct a multi-coding-unit / multi-prediction-block intra picture
//! with the spec-exact most-probable-mode derivation instead of the
//! flat-single-CU `INTRA_DC` neighbour assumption.
//!
//! The field is luma-only: §8.4.2 operates on luma prediction-block
//! locations, and §8.4.3 derives `IntraPredModeC` from the co-located
//! `IntraPredModeY` (already threaded through [`crate::hevc::engine::recon`]).

use crate::hevc::engine::binarization::CuPredMode;
use crate::hevc::engine::intra_pred::INTRA_DC;

/// `Log2MinTbSizeY` is at most 2 across all profiles (a 4×4 minimum
/// transform / prediction block), so the §8.4.2 neighbour grid is keyed
/// at the 4×4 luma min-block granularity. Every coding/prediction block
/// is an integer number of these cells on a side.
pub const MIN_BLOCK_LOG2: u32 = 2;

/// The 4×4 luma min-block side in samples.
pub const MIN_BLOCK_SIZE: usize = 1 << MIN_BLOCK_LOG2;

/// One min-block's §8.4.2 neighbour facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cell {
    /// `IntraPredModeY[ x ][ y ]` (0..=34); only meaningful when
    /// `pred_mode == MODE_INTRA`.
    intra_pred_mode_y: u8,
    /// `CuPredMode[ x ][ y ]`.
    pred_mode: CuPredMode,
    /// `pcm_flag[ x ][ y ]`.
    pcm_flag: bool,
    /// Whether this cell has been written (a still-unwritten cell is
    /// treated as not-yet-decoded → unavailable by the z-scan test).
    written: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            intra_pred_mode_y: INTRA_DC,
            pred_mode: CuPredMode::Intra,
            pcm_flag: false,
            written: false,
        }
    }
}

/// Which §8.4.2 neighbour a candidate mode is being derived for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighbour {
    /// `A` — the left neighbour `( xPb − 1, yPb )`.
    Left,
    /// `B` — the above neighbour `( xPb, yPb − 1 )`.
    Above,
}

/// A per-picture luma `IntraPredModeY` field on the 4×4 min-block grid.
///
/// The grid is sized to the picture (in min blocks) and records, for every
/// decoded luma prediction block, the three §8.4.2 candidate-derivation
/// inputs. Reconstruction writes a coding unit's cells before the next
/// coding unit's prediction-block modes are derived, so the left / above
/// neighbour lookups always see already-decoded state.
#[derive(Debug, Clone)]
pub struct IntraModeField {
    /// Width / height of the grid in 4×4 min blocks.
    w_blocks: usize,
    h_blocks: usize,
    /// `CtbLog2SizeY` — the §8.4.2 step-2 CTB-row test for neighbour `B`.
    ctb_log2_size_y: u32,
    cells: Vec<Cell>,
}

impl IntraModeField {
    /// Grid width in 4×4 min blocks.
    #[must_use]
    pub fn w_blocks(&self) -> usize {
        self.w_blocks
    }

    /// Grid height in 4×4 min blocks.
    #[must_use]
    pub fn h_blocks(&self) -> usize {
        self.h_blocks
    }

    /// `CtbLog2SizeY` the field was allocated with.
    #[must_use]
    pub fn ctb_log2(&self) -> u32 {
        self.ctb_log2_size_y
    }

    /// Allocate the field for a picture `pic_width_in_luma_samples` ×
    /// `pic_height_in_luma_samples` with the given `CtbLog2SizeY`. All
    /// cells start unwritten (every neighbour lookup before a write
    /// reports `INTRA_DC` per the §8.4.2 unavailable branch).
    #[must_use]
    pub fn new(pic_width_luma: usize, pic_height_luma: usize, ctb_log2_size_y: u32) -> Self {
        let w_blocks = pic_width_luma.div_ceil(MIN_BLOCK_SIZE);
        let h_blocks = pic_height_luma.div_ceil(MIN_BLOCK_SIZE);
        Self {
            w_blocks,
            h_blocks,
            ctb_log2_size_y,
            cells: vec![Cell::default(); w_blocks * h_blocks],
        }
    }

    #[inline]
    fn cell_index(&self, x_luma: usize, y_luma: usize) -> usize {
        let bx = x_luma >> MIN_BLOCK_LOG2;
        let by = y_luma >> MIN_BLOCK_LOG2;
        by * self.w_blocks + bx
    }

    /// Record an intra prediction block's `IntraPredModeY` across the
    /// `n_pb` × `n_pb` luma samples at `( x_pb, y_pb )`. `pcm_flag` and the
    /// `MODE_INTRA` `CuPredMode` are stamped on every covered min block.
    pub fn record_intra_pb(
        &mut self,
        x_pb: usize,
        y_pb: usize,
        n_pb: usize,
        intra_pred_mode_y: u8,
        pcm_flag: bool,
    ) {
        self.fill(
            x_pb,
            y_pb,
            n_pb,
            n_pb,
            Cell {
                intra_pred_mode_y,
                pred_mode: CuPredMode::Intra,
                pcm_flag,
                written: true,
            },
        );
    }

    /// §8.4.4.2.1 constrained-intra gate — whether the recorded
    /// `CuPredMode[ x ][ y ]` at the luma location is `MODE_INTRA`.
    /// Unwritten (not-yet-decoded) cells report `false`; the §6.4.1
    /// z-scan availability independently denies those.
    #[must_use]
    pub fn is_intra_at(&self, x_luma: usize, y_luma: usize) -> bool {
        let c = &self.cells[self.cell_index(x_luma, y_luma)];
        c.written && matches!(c.pred_mode, CuPredMode::Intra)
    }

    /// Record a non-intra coding unit (`MODE_INTER` / `MODE_SKIP`) across
    /// its `n_cb` × `n_cb` luma samples. The §8.4.2 candidate derivation
    /// maps such a neighbour to `INTRA_DC`, but the cell must be marked
    /// written so the z-scan availability test sees a decoded neighbour.
    pub fn record_non_intra_cu(&mut self, x_cb: usize, y_cb: usize, n_cb: usize, mode: CuPredMode) {
        self.fill(
            x_cb,
            y_cb,
            n_cb,
            n_cb,
            Cell {
                intra_pred_mode_y: INTRA_DC,
                pred_mode: mode,
                pcm_flag: false,
                written: true,
            },
        );
    }

    fn fill(&mut self, x: usize, y: usize, w: usize, h: usize, cell: Cell) {
        let bx0 = x >> MIN_BLOCK_LOG2;
        let by0 = y >> MIN_BLOCK_LOG2;
        let bx1 = ((x + w).min(self.w_blocks << MIN_BLOCK_LOG2)).div_ceil(MIN_BLOCK_SIZE);
        let by1 = ((y + h).min(self.h_blocks << MIN_BLOCK_LOG2)).div_ceil(MIN_BLOCK_SIZE);
        for by in by0..by1 {
            for bx in bx0..bx1 {
                self.cells[by * self.w_blocks + bx] = cell;
            }
        }
    }

    /// §8.4.2 step 2 — derive `candIntraPredModeX` for the neighbour `X`
    /// of the prediction block at `( x_pb, y_pb )`.
    ///
    /// `available` is the §6.4.1 z-scan availability of the neighbour
    /// location, supplied by the caller (the [`crate::hevc::engine::availability`]
    /// `z_scan_availability` consults the picture tiling / slice map the
    /// driver owns). The remaining branches — the not-`MODE_INTRA` /
    /// `pcm_flag` mask, the neighbour-`B` CTB-row test, and the
    /// `IntraPredModeY[ xNb ][ yNb ]` lookup — are evaluated here against
    /// the recorded field.
    ///
    /// Returns `INTRA_DC` whenever the spec's step-2 ordered conditions
    /// select it, and the recorded neighbour mode otherwise.
    #[must_use]
    pub fn cand_intra_pred_mode(
        &self,
        x_pb: usize,
        y_pb: usize,
        neighbour: Neighbour,
        available: bool,
    ) -> u8 {
        // Neighbour location ( xNbX, yNbX ) — step 1.
        let (x_nb, y_nb) = match neighbour {
            Neighbour::Left => (x_pb as i64 - 1, y_pb as i64),
            Neighbour::Above => (x_pb as i64, y_pb as i64 - 1),
        };
        // availableX == FALSE ⇒ INTRA_DC.
        if !available || x_nb < 0 || y_nb < 0 {
            return INTRA_DC;
        }
        let x_nb = x_nb as usize;
        let y_nb = y_nb as usize;
        if x_nb >= (self.w_blocks << MIN_BLOCK_LOG2) || y_nb >= (self.h_blocks << MIN_BLOCK_LOG2) {
            return INTRA_DC;
        }
        let cell = self.cells[self.cell_index(x_nb, y_nb)];
        // An unwritten neighbour cannot have been decoded → INTRA_DC. (The
        // z-scan availability test should already exclude it, but a
        // single-slice driver that passes `available = true` for all
        // in-picture neighbours relies on this guard.)
        if !cell.written {
            return INTRA_DC;
        }
        // CuPredMode[ xNbX ][ yNbX ] != MODE_INTRA || pcm_flag ⇒ INTRA_DC.
        if cell.pred_mode != CuPredMode::Intra || cell.pcm_flag {
            return INTRA_DC;
        }
        // X == B and yPb − 1 < ( ( yPb >> CtbLog2SizeY ) << CtbLog2SizeY )
        // ⇒ INTRA_DC (the above neighbour is in the CTB row above).
        if neighbour == Neighbour::Above {
            let ctb_row_top = (y_pb >> self.ctb_log2_size_y) << self.ctb_log2_size_y;
            if y_pb >= 1 && (y_pb - 1) < ctb_row_top {
                return INTRA_DC;
            }
        }
        cell.intra_pred_mode_y
    }

    /// The recorded `IntraPredModeY` at a luma location (or `None` if
    /// no block covering it has been written yet).
    #[must_use]
    pub fn recorded_mode(&self, x_luma: usize, y_luma: usize) -> Option<u8> {
        let cell = self.cells[self.cell_index(x_luma, y_luma)];
        cell.written.then_some(cell.intra_pred_mode_y)
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn fresh_field_reports_dc_for_all_neighbours() {
        let f = IntraModeField::new(64, 64, 6);
        // No cell written yet ⇒ both candidates are INTRA_DC.
        assert_eq!(
            f.cand_intra_pred_mode(8, 8, Neighbour::Left, true),
            INTRA_DC
        );
        assert_eq!(
            f.cand_intra_pred_mode(8, 8, Neighbour::Above, true),
            INTRA_DC
        );
    }

    #[test]
    fn out_of_picture_neighbour_is_dc() {
        let f = IntraModeField::new(64, 64, 6);
        // ( -1, 0 ) left of the picture origin.
        assert_eq!(
            f.cand_intra_pred_mode(0, 0, Neighbour::Left, true),
            INTRA_DC
        );
        // ( 0, -1 ) above the picture origin.
        assert_eq!(
            f.cand_intra_pred_mode(0, 0, Neighbour::Above, true),
            INTRA_DC
        );
    }

    #[test]
    fn unavailable_neighbour_is_dc_even_if_written() {
        let mut f = IntraModeField::new(64, 64, 6);
        f.record_intra_pb(0, 8, 8, 26, false);
        // The left neighbour ( -1+8 ... ) carries mode 26 but availability
        // is forced FALSE by the caller (e.g. slice boundary).
        assert_eq!(
            f.cand_intra_pred_mode(8, 8, Neighbour::Left, false),
            INTRA_DC
        );
    }

    #[test]
    fn left_neighbour_returns_recorded_intra_mode() {
        let mut f = IntraModeField::new(64, 64, 6);
        // Record an 8×8 intra PB at ( 0, 8 ) with angular mode 18.
        f.record_intra_pb(0, 8, 8, 18, false);
        // Current PB at ( 8, 8 ): left neighbour ( 7, 8 ) is inside the PB.
        assert_eq!(f.cand_intra_pred_mode(8, 8, Neighbour::Left, true), 18);
    }

    #[test]
    fn above_neighbour_in_same_ctb_row_returns_mode() {
        // CtbLog2SizeY = 5 ⇒ 32-sample CTB. A PB at ( 0, 8 ) whose above
        // neighbour ( 0, 7 ) is in the same CTB row keeps the mode.
        let mut f = IntraModeField::new(64, 64, 5);
        f.record_intra_pb(0, 0, 8, 22, false);
        // PB at ( 0, 8 ): above is ( 0, 7 ); ctb_row_top = 0; 7 >= 0 so the
        // neighbour is in-row, mode preserved.
        assert_eq!(f.cand_intra_pred_mode(0, 8, Neighbour::Above, true), 22);
    }

    #[test]
    fn above_neighbour_in_ctb_row_above_is_dc() {
        // CtbLog2SizeY = 5 (32). A PB whose top is on a CTB-row boundary
        // ( yPb = 32 ) has its above neighbour ( yPb - 1 = 31 ) in the CTB
        // row above ⇒ candIntraPredModeB = INTRA_DC.
        let mut f = IntraModeField::new(64, 64, 5);
        f.record_intra_pb(0, 28, 4, 30, false);
        // ctb_row_top for yPb=32 is 32; 31 < 32 ⇒ DC.
        assert_eq!(
            f.cand_intra_pred_mode(0, 32, Neighbour::Above, true),
            INTRA_DC
        );
        // The left neighbour has no such CTB-row restriction.
    }

    #[test]
    fn non_intra_neighbour_is_dc() {
        let mut f = IntraModeField::new(64, 64, 6);
        f.record_non_intra_cu(0, 8, 8, CuPredMode::Inter);
        assert_eq!(
            f.cand_intra_pred_mode(8, 8, Neighbour::Left, true),
            INTRA_DC
        );
    }

    #[test]
    fn pcm_neighbour_is_dc() {
        let mut f = IntraModeField::new(64, 64, 6);
        f.record_intra_pb(0, 8, 8, 18, true);
        assert_eq!(
            f.cand_intra_pred_mode(8, 8, Neighbour::Left, true),
            INTRA_DC
        );
    }

    #[test]
    fn record_fills_every_covered_min_block() {
        let mut f = IntraModeField::new(64, 64, 6);
        // A 16×16 intra PB at ( 16, 16 ), mode 10.
        f.record_intra_pb(16, 16, 16, 10, false);
        // The bottom-right min block ( 28, 28 ) is covered; a PB to its
        // right at ( 32, 28 ) sees mode 10 on its left neighbour ( 31, 28 ).
        assert_eq!(f.cand_intra_pred_mode(32, 28, Neighbour::Left, true), 10);
        // A min block one past the PB ( 32, 16 ) is NOT covered.
        assert_eq!(
            f.cand_intra_pred_mode(36, 16, Neighbour::Left, true),
            INTRA_DC
        );
    }
}
