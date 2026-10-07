//! §6.4 availability processes + the §6.5.1 / §6.5.2 picture-level
//! scanning conversions they depend on.
//!
//! Intra (and, later, inter) sample prediction asks, for each candidate
//! neighbour location `( xNbY, yNbY )`, whether that neighbour is
//! *available for prediction* — i.e. it lies inside the picture, has
//! already been decoded in the current decoding order, and sits in the
//! same slice segment and tile as the current block. ITU-T Rec. H.265
//! §6.4.1 answers that question, and it does so in terms of a
//! *minimum-transform-block address in z-scan order*, `MinTbAddrZs`.
//!
//! `MinTbAddrZs` (§6.5.2) in turn is built from `CtbAddrRsToTs`
//! (§6.5.1) — the CTB raster-scan ↔ tile-scan address conversion that
//! folds the picture's tile layout into the decoding order. So this
//! module implements three layers, bottom-up:
//!
//! * [`PictureTiling`] — §6.5.1. From the picture geometry in CTBs and
//!   the §7.3.2.3.1 tile-partitioning parameters it derives `colWidth`
//!   / `rowHeight` (eqs. 6-3 / 6-4), `colBd` / `rowBd` (eqs. 6-5 /
//!   6-6), `CtbAddrRsToTs` / `CtbAddrTsToRs` (eqs. 6-7 / 6-8), and
//!   `TileId` (eq. 6-9).
//! * [`PictureTiling::min_tb_addr_zs`] — §6.5.2. The
//!   `MinTbAddrZs[ x ][ y ]` array (eq. 6-10) mapping a minimum-block
//!   `( x, y )` to its z-scan address, interleaving the within-CTB
//!   Morton (z) order with the tile-scan CTB order.
//! * [`PictureTiling::z_scan_availability`] — §6.4.1, and
//!   [`PictureTiling::prediction_block_availability`] — §6.4.2.
//!
//! ## Scope
//!
//! The numerics are self-contained. The caller supplies the picture
//! geometry (in CTBs and in luma samples), the `Log2SizeY` shifts, the
//! tile parameters from the active PPS, and a `slice_addr_rs` lookup
//! that maps a CTB raster address to the `SliceAddrRs` of the slice
//! segment that owns it (§6.4.1's slice-boundary test). For §6.4.2 the
//! caller additionally supplies a `CuPredMode` lookup (the final
//! `MODE_INTRA` masking step). Building those lookups is the slice-data
//! walk's responsibility; this module starts at the geometry and stops
//! at `availableN`.

/// `CuPredMode` value `MODE_INTRA` — used by the §6.4.2 final masking
/// step. Mirrors the Table 7-10 enumeration without depending on the
/// slice module's richer type.
pub const MODE_INTRA: u8 = 1;

/// Errors from constructing a [`PictureTiling`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailabilityError {
    /// A geometry input was zero where the spec requires a positive
    /// value (`PicWidthInCtbsY` / `PicHeightInCtbsY` must be ≥ 1).
    ZeroGeometry {
        /// Name of the offending field.
        field: &'static str,
    },
    /// `CtbLog2SizeY` is smaller than `MinTbLog2SizeY`, which would make
    /// the §6.5.2 `( CtbLog2SizeY − MinTbLog2SizeY )` shift negative.
    InvalidLog2Sizes {
        /// `CtbLog2SizeY`.
        ctb_log2: u32,
        /// `MinTbLog2SizeY`.
        min_tb_log2: u32,
    },
    /// The explicit (`uniform_spacing_flag == 0`) tile-size array had a
    /// length other than the required `num_tile_*_minus1`.
    TileArrayLength {
        /// Name of the offending array.
        field: &'static str,
        /// Length the spec requires.
        expected: usize,
        /// Length that was supplied.
        got: usize,
    },
    /// An explicit tile column/row was wider/taller than the picture, so
    /// the implied last tile would have a non-positive size.
    TileOverflow {
        /// `"column"` or `"row"`.
        dimension: &'static str,
    },
}

impl core::fmt::Display for AvailabilityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroGeometry { field } => {
                write!(f, "picture geometry field {field} must be positive")
            }
            Self::InvalidLog2Sizes {
                ctb_log2,
                min_tb_log2,
            } => write!(
                f,
                "CtbLog2SizeY ({ctb_log2}) must be >= MinTbLog2SizeY ({min_tb_log2})"
            ),
            Self::TileArrayLength {
                field,
                expected,
                got,
            } => write!(f, "tile array {field} length {got} != expected {expected}"),
            Self::TileOverflow { dimension } => {
                write!(f, "explicit tile {dimension} sizes overflow the picture")
            }
        }
    }
}

impl std::error::Error for AvailabilityError {}

/// The §7.3.2.3.1 tile-partitioning parameters needed to build the
/// CTB-scan conversion, as carried by the active PPS.
///
/// When `tiles_enabled_flag == 0` the picture is a single tile; pass
/// [`TilingParams::single_tile`] (one column, one row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilingParams {
    /// `num_tile_columns_minus1` — the number of tile columns is this
    /// plus one.
    pub num_tile_columns_minus1: u32,
    /// `num_tile_rows_minus1` — the number of tile rows is this plus
    /// one.
    pub num_tile_rows_minus1: u32,
    /// `uniform_spacing_flag`. When `true` the columns/rows are spread
    /// as evenly as possible (eqs. 6-3 / 6-4 uniform branch) and the
    /// explicit arrays are ignored.
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[ i ]` for `i` in
    /// `0..num_tile_columns_minus1` (only when not uniform).
    pub column_width_minus1: Vec<u32>,
    /// `row_height_minus1[ j ]` for `j` in `0..num_tile_rows_minus1`
    /// (only when not uniform).
    pub row_height_minus1: Vec<u32>,
}

impl TilingParams {
    /// The single-tile layout used when `tiles_enabled_flag == 0`.
    pub fn single_tile() -> Self {
        Self {
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
        }
    }
}

/// Picture-level geometry plus the derived §6.5.1 / §6.5.2 conversion
/// tables. Construct with [`PictureTiling::new`], then query availability
/// with [`PictureTiling::z_scan_availability`] /
/// [`PictureTiling::prediction_block_availability`].
#[derive(Debug, Clone)]
pub struct PictureTiling {
    pic_width_in_ctbs_y: u32,
    pic_height_in_ctbs_y: u32,
    pic_width_in_luma_samples: u32,
    pic_height_in_luma_samples: u32,
    ctb_log2_size_y: u32,
    min_tb_log2_size_y: u32,
    /// `colBd[ 0..=num_tile_columns_minus1 + 1 ]` (eq. 6-5).
    col_bd: Vec<u32>,
    /// `rowBd[ 0..=num_tile_rows_minus1 + 1 ]` (eq. 6-6).
    row_bd: Vec<u32>,
    /// `CtbAddrRsToTs[ ctbAddrRs ]` (eq. 6-7).
    ctb_addr_rs_to_ts: Vec<u32>,
    /// `CtbAddrTsToRs[ ctbAddrTs ]` (eq. 6-8).
    ctb_addr_ts_to_rs: Vec<u32>,
    /// `TileId[ ctbAddrTs ]` (eq. 6-9).
    tile_id: Vec<u32>,
}

impl PictureTiling {
    /// Build the §6.5.1 conversion tables from the picture geometry and
    /// tile parameters.
    ///
    /// * `pic_width_in_ctbs_y` / `pic_height_in_ctbs_y` —
    ///   `PicWidthInCtbsY` / `PicHeightInCtbsY` (eqs. 7-20 / 7-21).
    /// * `pic_width_in_luma_samples` / `pic_height_in_luma_samples` —
    ///   the SPS conformance-relevant `pic_width_in_luma_samples` /
    ///   `pic_height_in_luma_samples` used by §6.4.1's boundary test.
    /// * `ctb_log2_size_y` — `CtbLog2SizeY` (eq. 7-11).
    /// * `min_tb_log2_size_y` — `MinTbLog2SizeY`
    ///   (= `log2_min_luma_transform_block_size_minus2 + 2`).
    /// * `tiles` — the active PPS tile-partitioning parameters.
    pub fn new(
        pic_width_in_ctbs_y: u32,
        pic_height_in_ctbs_y: u32,
        pic_width_in_luma_samples: u32,
        pic_height_in_luma_samples: u32,
        ctb_log2_size_y: u32,
        min_tb_log2_size_y: u32,
        tiles: &TilingParams,
    ) -> Result<Self, AvailabilityError> {
        if pic_width_in_ctbs_y == 0 {
            return Err(AvailabilityError::ZeroGeometry {
                field: "PicWidthInCtbsY",
            });
        }
        if pic_height_in_ctbs_y == 0 {
            return Err(AvailabilityError::ZeroGeometry {
                field: "PicHeightInCtbsY",
            });
        }
        if ctb_log2_size_y < min_tb_log2_size_y {
            return Err(AvailabilityError::InvalidLog2Sizes {
                ctb_log2: ctb_log2_size_y,
                min_tb_log2: min_tb_log2_size_y,
            });
        }

        let num_cols = (tiles.num_tile_columns_minus1 + 1) as usize;
        let num_rows = (tiles.num_tile_rows_minus1 + 1) as usize;

        // §6.5.1 eqs. 6-3 / 6-4 — colWidth / rowHeight (in CTBs).
        let col_width = derive_tile_sizes(
            pic_width_in_ctbs_y,
            tiles.num_tile_columns_minus1,
            tiles.uniform_spacing_flag,
            &tiles.column_width_minus1,
            "column_width_minus1",
            "column",
        )?;
        let row_height = derive_tile_sizes(
            pic_height_in_ctbs_y,
            tiles.num_tile_rows_minus1,
            tiles.uniform_spacing_flag,
            &tiles.row_height_minus1,
            "row_height_minus1",
            "row",
        )?;

        // §6.5.1 eqs. 6-5 / 6-6 — colBd / rowBd (cumulative boundaries).
        let mut col_bd = vec![0u32; num_cols + 1];
        for i in 0..num_cols {
            col_bd[i + 1] = col_bd[i] + col_width[i];
        }
        let mut row_bd = vec![0u32; num_rows + 1];
        for j in 0..num_rows {
            row_bd[j + 1] = row_bd[j] + row_height[j];
        }

        let pic_size_in_ctbs_y = (pic_width_in_ctbs_y * pic_height_in_ctbs_y) as usize;

        // §6.5.1 eq. 6-7 — CtbAddrRsToTs.
        let mut ctb_addr_rs_to_ts = vec![0u32; pic_size_in_ctbs_y];
        for ctb_addr_rs in 0..pic_size_in_ctbs_y as u32 {
            let tb_x = ctb_addr_rs % pic_width_in_ctbs_y;
            let tb_y = ctb_addr_rs / pic_width_in_ctbs_y;
            let mut tile_x = 0usize;
            for (i, &bd) in col_bd.iter().take(num_cols).enumerate() {
                if tb_x >= bd {
                    tile_x = i;
                }
            }
            let mut tile_y = 0usize;
            for (j, &bd) in row_bd.iter().take(num_rows).enumerate() {
                if tb_y >= bd {
                    tile_y = j;
                }
            }
            let mut val = 0u32;
            for &cw in col_width.iter().take(tile_x) {
                val += row_height[tile_y] * cw;
            }
            for &rh in row_height.iter().take(tile_y) {
                val += pic_width_in_ctbs_y * rh;
            }
            val += (tb_y - row_bd[tile_y]) * col_width[tile_x] + tb_x - col_bd[tile_x];
            ctb_addr_rs_to_ts[ctb_addr_rs as usize] = val;
        }

        // §6.5.1 eq. 6-8 — CtbAddrTsToRs (inverse permutation).
        let mut ctb_addr_ts_to_rs = vec![0u32; pic_size_in_ctbs_y];
        for ctb_addr_rs in 0..pic_size_in_ctbs_y as u32 {
            ctb_addr_ts_to_rs[ctb_addr_rs_to_ts[ctb_addr_rs as usize] as usize] = ctb_addr_rs;
        }

        // §6.5.1 eq. 6-9 — TileId.
        let mut tile_id = vec![0u32; pic_size_in_ctbs_y];
        let mut tile_idx = 0u32;
        for j in 0..num_rows {
            for i in 0..num_cols {
                for y in row_bd[j]..row_bd[j + 1] {
                    for x in col_bd[i]..col_bd[i + 1] {
                        let rs = (y * pic_width_in_ctbs_y + x) as usize;
                        tile_id[ctb_addr_rs_to_ts[rs] as usize] = tile_idx;
                    }
                }
                tile_idx += 1;
            }
        }

        Ok(Self {
            pic_width_in_ctbs_y,
            pic_height_in_ctbs_y,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            ctb_log2_size_y,
            min_tb_log2_size_y,
            col_bd,
            row_bd,
            ctb_addr_rs_to_ts,
            ctb_addr_ts_to_rs,
            tile_id,
        })
    }

    /// `CtbAddrRsToTs[ ctbAddrRs ]` (eq. 6-7). Panics on out-of-range
    /// `ctb_addr_rs`; valid indices are `0..PicSizeInCtbsY`.
    pub fn ctb_addr_rs_to_ts(&self, ctb_addr_rs: u32) -> u32 {
        self.ctb_addr_rs_to_ts[ctb_addr_rs as usize]
    }

    /// `CtbAddrTsToRs[ ctbAddrTs ]` (eq. 6-8).
    pub fn ctb_addr_ts_to_rs(&self, ctb_addr_ts: u32) -> u32 {
        self.ctb_addr_ts_to_rs[ctb_addr_ts as usize]
    }

    /// `TileId[ ctbAddrTs ]` (eq. 6-9) — the tile a tile-scan CTB
    /// address belongs to.
    pub fn tile_id(&self, ctb_addr_ts: u32) -> u32 {
        self.tile_id[ctb_addr_ts as usize]
    }

    /// `colBd[ i ]` (eq. 6-5) — the i-th tile-column boundary in units
    /// of CTBs, for `i` in `0..=num_tile_columns_minus1 + 1`.
    pub fn col_bd(&self) -> &[u32] {
        &self.col_bd
    }

    /// `rowBd[ j ]` (eq. 6-6) — the j-th tile-row boundary in units of
    /// CTBs, for `j` in `0..=num_tile_rows_minus1 + 1`.
    pub fn row_bd(&self) -> &[u32] {
        &self.row_bd
    }

    /// `PicWidthInCtbsY`.
    pub fn pic_width_in_ctbs_y(&self) -> u32 {
        self.pic_width_in_ctbs_y
    }

    /// `PicHeightInCtbsY`.
    pub fn pic_height_in_ctbs_y(&self) -> u32 {
        self.pic_height_in_ctbs_y
    }

    /// The TileId of the tile covering luma sample `( xY, yY )`.
    /// Convenience over [`Self::tile_id`] that converts the luma
    /// location through the CTB raster→tile-scan conversion. The caller
    /// is responsible for keeping `( xY, yY )` inside the picture.
    fn tile_id_at_luma(&self, x_y: u32, y_y: u32) -> u32 {
        let ctb_x = x_y >> self.ctb_log2_size_y;
        let ctb_y = y_y >> self.ctb_log2_size_y;
        let ctb_addr_rs = ctb_y * self.pic_width_in_ctbs_y + ctb_x;
        self.tile_id[self.ctb_addr_rs_to_ts[ctb_addr_rs as usize] as usize]
    }

    /// `MinTbAddrZs[ x ][ y ]` (§6.5.2 eq. 6-10) — the minimum-block
    /// location `( x, y )` (in units of `MinTbLog2SizeY`-sized blocks)
    /// converted to a minimum-block address in z-scan order.
    ///
    /// Combines the tile-scan CTB order (high bits) with the within-CTB
    /// Morton / z interleave of the block's `( x, y )` low bits. The
    /// caller is responsible for keeping `( x, y )` within the picture's
    /// minimum-block grid.
    pub fn min_tb_addr_zs(&self, x: u32, y: u32) -> u32 {
        let shift = self.ctb_log2_size_y - self.min_tb_log2_size_y;
        let tb_x = (x << self.min_tb_log2_size_y) >> self.ctb_log2_size_y;
        let tb_y = (y << self.min_tb_log2_size_y) >> self.ctb_log2_size_y;
        let ctb_addr_rs = self.pic_width_in_ctbs_y * tb_y + tb_x;
        let mut addr = self.ctb_addr_rs_to_ts[ctb_addr_rs as usize] << (shift * 2);
        let mut p = 0u32;
        for i in 0..shift {
            let m = 1u32 << i;
            p += if m & x != 0 { m * m } else { 0 } + if m & y != 0 { 2 * m * m } else { 0 };
        }
        addr += p;
        addr
    }

    /// §6.4.1 — derivation process for z-scan order block availability.
    ///
    /// Returns whether the neighbouring block covering luma location
    /// `( x_nb_y, y_nb_y )` is available for the current block whose
    /// top-left luma sample is `( x_curr, y_curr )`.
    ///
    /// `slice_addr_rs` maps a CTB raster-scan address to the
    /// `SliceAddrRs` of the slice segment that owns that CTB — the
    /// slice-boundary test of §6.4.1. The caller's slice-data walk
    /// builds it; for a single-slice picture it is the constant `0`.
    pub fn z_scan_availability<F>(
        &self,
        x_curr: u32,
        y_curr: u32,
        x_nb_y: i32,
        y_nb_y: i32,
        slice_addr_rs: F,
    ) -> bool
    where
        F: Fn(u32) -> u32,
    {
        // eq. 6-1.
        let min_block_addr_curr = self.min_tb_addr_zs(
            x_curr >> self.min_tb_log2_size_y,
            y_curr >> self.min_tb_log2_size_y,
        );

        // eq. 6-2 — out-of-picture neighbours get minBlockAddrN = −1.
        if x_nb_y < 0
            || y_nb_y < 0
            || x_nb_y as u32 >= self.pic_width_in_luma_samples
            || y_nb_y as u32 >= self.pic_height_in_luma_samples
        {
            return false;
        }
        let x_nb_y = x_nb_y as u32;
        let y_nb_y = y_nb_y as u32;
        let min_block_addr_n = self.min_tb_addr_zs(
            x_nb_y >> self.min_tb_log2_size_y,
            y_nb_y >> self.min_tb_log2_size_y,
        );

        // The neighbour must already be decoded (z-scan order test).
        if min_block_addr_n > min_block_addr_curr {
            return false;
        }

        // Slice-segment boundary: compare the SliceAddrRs of the CTBs
        // owning each block. The block addresses are in z-scan order;
        // map them back to the owning CTB raster address.
        let ctb_rs_curr = self.ctb_rs_of_z(min_block_addr_curr);
        let ctb_rs_n = self.ctb_rs_of_z(min_block_addr_n);
        if slice_addr_rs(ctb_rs_n) != slice_addr_rs(ctb_rs_curr) {
            return false;
        }

        // Tile boundary: the neighbour must be in the same tile.
        if self.tile_id_at_luma(x_nb_y, y_nb_y) != self.tile_id_at_luma(x_curr, y_curr) {
            return false;
        }

        true
    }

    /// §6.4.2 — derivation process for prediction block availability.
    ///
    /// Wraps [`Self::z_scan_availability`] with the `sameCb` short-cut
    /// (a neighbour in the *same* coding block is unavailable when it
    /// would not yet have been decoded under the §7.4.9 partitioning
    /// order) and the final `MODE_INTRA` masking.
    ///
    /// * `( x_cb, y_cb )` / `n_cb_s` — the current coding block.
    /// * `( x_pb, y_pb )` / `n_pb_w` / `n_pb_h` / `part_idx` — the
    ///   current prediction block and its partition index.
    /// * `( x_nb_y, y_nb_y )` — the neighbour location.
    /// * `slice_addr_rs` — as in [`Self::z_scan_availability`].
    /// * `cu_pred_mode` — maps a luma location to its `CuPredMode`
    ///   (`MODE_INTRA` / `MODE_INTER` / `MODE_SKIP`). Only consulted for
    ///   in-picture neighbours that are otherwise available.
    #[allow(clippy::too_many_arguments)]
    pub fn prediction_block_availability<S, M>(
        &self,
        x_cb: u32,
        y_cb: u32,
        n_cb_s: u32,
        x_pb: u32,
        y_pb: u32,
        n_pb_w: u32,
        n_pb_h: u32,
        part_idx: u32,
        x_nb_y: i32,
        y_nb_y: i32,
        slice_addr_rs: S,
        cu_pred_mode: M,
    ) -> bool
    where
        S: Fn(u32) -> u32,
        M: Fn(u32, u32) -> u8,
    {
        // sameCb: does the neighbour cover the current coding block?
        let same_cb = x_nb_y >= 0
            && y_nb_y >= 0
            && x_cb <= x_nb_y as u32
            && y_cb <= y_nb_y as u32
            && (x_cb + n_cb_s) > x_nb_y as u32
            && (y_cb + n_cb_s) > y_nb_y as u32;

        let mut available_n = if !same_cb {
            self.z_scan_availability(x_pb, y_pb, x_nb_y, y_nb_y, slice_addr_rs)
        } else if (n_pb_w << 1) == n_cb_s
            && (n_pb_h << 1) == n_cb_s
            && part_idx == 1
            && (y_cb + n_pb_h) <= y_nb_y as u32
            && (x_cb + n_pb_w) > x_nb_y as u32
        {
            // The Nx2N / 2NxN second partition cannot reference the not-
            // yet-decoded partition-0 region inside the same CU.
            false
        } else {
            true
        };

        // Final §6.4.2 step: intra neighbours are unavailable for the
        // (inter) prediction-block availability query.
        if available_n {
            let x = x_nb_y as u32;
            let y = y_nb_y as u32;
            if cu_pred_mode(x, y) == MODE_INTRA {
                available_n = false;
            }
        }

        available_n
    }

    /// Map a minimum-block z-scan address back to the CTB raster-scan
    /// address that contains it. The high
    /// `( CtbLog2SizeY − MinTbLog2SizeY ) * 2` bits of a z-scan block
    /// address are the tile-scan CTB address (see eq. 6-10).
    fn ctb_rs_of_z(&self, min_block_addr_z: u32) -> u32 {
        let shift = (self.ctb_log2_size_y - self.min_tb_log2_size_y) * 2;
        let ctb_addr_ts = min_block_addr_z >> shift;
        self.ctb_addr_ts_to_rs[ctb_addr_ts as usize]
    }
}

/// §6.5.1 eqs. 6-3 / 6-4 — derive the per-tile sizes (`colWidth` /
/// `rowHeight`) in CTBs, for one dimension.
fn derive_tile_sizes(
    pic_size_in_ctbs: u32,
    num_tiles_minus1: u32,
    uniform_spacing_flag: bool,
    explicit_minus1: &[u32],
    array_field: &'static str,
    dimension: &'static str,
) -> Result<Vec<u32>, AvailabilityError> {
    let num_tiles = (num_tiles_minus1 + 1) as usize;
    let mut sizes = vec![0u32; num_tiles];
    if uniform_spacing_flag {
        for (i, slot) in sizes.iter_mut().enumerate() {
            let i = i as u32;
            *slot = ((i + 1) * pic_size_in_ctbs) / (num_tiles_minus1 + 1)
                - (i * pic_size_in_ctbs) / (num_tiles_minus1 + 1);
        }
    } else {
        if explicit_minus1.len() != num_tiles_minus1 as usize {
            return Err(AvailabilityError::TileArrayLength {
                field: array_field,
                expected: num_tiles_minus1 as usize,
                got: explicit_minus1.len(),
            });
        }
        // The last tile takes the remainder; the first num_tiles_minus1
        // are explicit.
        sizes[num_tiles - 1] = pic_size_in_ctbs;
        for i in 0..num_tiles_minus1 as usize {
            sizes[i] = explicit_minus1[i] + 1;
            if sizes[i] > sizes[num_tiles - 1] {
                return Err(AvailabilityError::TileOverflow { dimension });
            }
            sizes[num_tiles - 1] -= sizes[i];
        }
        if sizes[num_tiles - 1] == 0 {
            return Err(AvailabilityError::TileOverflow { dimension });
        }
    }
    Ok(sizes)
}
