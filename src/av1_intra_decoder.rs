//! Dependency-free reconstruction of bounded AV1 Main-profile intra frames.
//! The implemented syntax is the standards-compliant reduced-still, 8-bit
//! monochrome, single-tile subset used by zvidlib's native AV1 encoder.
//! Every other syntax branch is rejected explicitly.
//!
//! Two quantization profiles are supported, selected by `base_q_idx` (spec
//! §5.9.12 `quantization_params`):
//!
//! - `base_q_idx == 0` (`CodedLossless`): every 4x4 block is reconstructed
//!   with the normative inverse Walsh-Hadamard transform
//!   ([`crate::inverse_wht_4x4`]). Lossless streams never signal
//!   `loop_filter_params` (spec §5.9.11: `CodedLossless` forces every loop
//!   filter level to 0), so deblocking is unreachable on this path, matching
//!   the spec.
//! - `base_q_idx != 0` (non-lossless): each coding block signals its own
//!   `tx_size` through spec §5.11.16 `read_tx_size` over the square
//!   transforms this crate's kernels implement (`TX_4X4` through
//!   `TX_64X64`), under either `TX_MODE_LARGEST` or `TX_MODE_SELECT` as
//!   the frame header's `tx_mode_select` bit selects, and its `tx_type`
//!   (`DCT_DCT` or `IDTX`; `ADST_ADST` is rejected as unsupported, and
//!   `TX_32X32` and above are `TX_SET_DCTONLY` so they signal no
//!   `tx_type` at all), coefficients are
//!   dequantized per spec §7.12 (`get_dc_quant`/`get_ac_quant`) and inverse
//!   transformed ([`crate::av1_intra::inverse_transform`]). `loop_filter_params`
//!   is parsed and the chosen per-block transform sizes are recorded into a
//!   [`crate::av1_filters::TxSizeGrid`] so [`crate::av1_filters::deblock_frame`]
//!   can select the correct filter length at each edge (spec §7.14.5).
//!   [`decode_av1_lossless_intra_with_tx_sizes`] exposes both the
//!   deblocked frame and the grid that produced it.

use crate::av1_cdf as cdf;
use crate::av1_filters::{FilterFrame, FilterPlane, LoopFilterParams, TxSizeGrid, deblock_frame};
use crate::av1_intra::{Av1IntraMode, Av1TxType, get_ac_quant, get_dc_quant, inverse_transform};
use crate::av1_intra_pred::{
    SmoothMode, add_residual_row, directional_row, paeth_row, smooth_row, sum_samples,
};
use crate::{
    Av1FrameType, Av1IntraFrame, Av1Obu, Av1ObuType, Av1Parser, Av1SymbolDecoder, Av1SyntaxSupport,
    ColorRange, Error, ErrorKind, Limits, Result, VideoDimensions, VideoFrame, inverse_wht_4x4,
};

const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_PLUS_RANGE: i32 = 14;
/// The widest square transform this crate's kernels implement
/// (`TX_64X64`), which is also AV1's largest transform.
const MAX_TX_WIDTH: usize = 64;
/// AV1 codes coefficients only inside a transform block's upper-left
/// 32x32 quadrant (spec §5.11.39 clamps the coded extent to 32 samples);
/// every position outside it reconstructs as zero.
const MAX_CODED_TX_WIDTH: usize = 32;

/// Decodes one low-overhead AV1 temporal unit into validated YUV planes.
/// Equivalent to [`decode_av1_lossless_intra_with_tx_sizes`] for callers
/// that do not need the per-block transform-size metadata.
pub fn decode_av1_lossless_intra(bytes: &[u8], limits: &Limits) -> Result<VideoFrame> {
    decode_av1_lossless_intra_with_tx_sizes(bytes, limits).map(|(frame, _)| frame)
}

/// Decodes one low-overhead AV1 temporal unit into validated YUV planes,
/// also returning the [`TxSizeGrid`] recording the transform size chosen
/// for every 4x4 luma unit. For `base_q_idx == 0` (lossless) streams every
/// unit is `TX_4X4` (lossless AV1 streams never signal `loop_filter_params`,
/// per spec §5.9.11, so the frame is returned unfiltered). For
/// `base_q_idx != 0` streams, the grid reflects the real per-block
/// `read_tx_size` choice and the frame has already been passed through
/// [`deblock_frame`] using the parsed `loop_filter_params`.
pub fn decode_av1_lossless_intra_with_tx_sizes(
    bytes: &[u8],
    limits: &Limits,
) -> Result<(VideoFrame, TxSizeGrid)> {
    if limits.max_av1_blocks_per_frame == 0 {
        return Err(resource("AV1 reconstruction block limit must be nonzero"));
    }
    let mut parser = Av1Parser::new(*limits)?;
    let obus = parser.parse_low_overhead(bytes)?;
    let sequence_count = obus
        .iter()
        .filter(|obu| matches!(obu, Av1Obu::SequenceHeader { .. }))
        .count();
    let frame_count = obus
        .iter()
        .filter(|obu| matches!(obu, Av1Obu::Frame { .. }))
        .count();
    if sequence_count != 1 || frame_count != 1 {
        return Err(malformed_error(
            "AV1 intra unit must contain exactly one sequence header and one frame OBU",
        ));
    }
    for obu in &obus {
        match obu {
            Av1Obu::SequenceHeader { .. }
            | Av1Obu::Frame { .. }
            | Av1Obu::TemporalDelimiter { .. }
            | Av1Obu::Metadata { .. } => {}
            Av1Obu::Skipped { header, .. } if header.obu_type == Av1ObuType::Padding => {}
            _ => {
                return Err(unsupported(
                    "AV1 intra unit contains an unsupported OBU type",
                ));
            }
        }
    }
    let sequence = obus
        .iter()
        .find_map(|obu| match obu {
            Av1Obu::SequenceHeader { sequence, .. } => Some(sequence),
            _ => None,
        })
        .ok_or_else(|| malformed_error("AV1 intra unit has no sequence header"))?;
    if sequence.support != Av1SyntaxSupport::MainProfile
        || sequence.color_config.bit_depth != 8
        || !sequence.color_config.monochrome
        || !sequence.reduced_still_picture_header
    {
        return Err(unsupported(
            "AV1 intra decoder supports reduced-still Main-profile 8-bit monochrome streams",
        ));
    }
    if sequence.enable_superres
        || sequence.enable_cdef
        || sequence.enable_restoration
        || sequence.film_grain_params_present
    {
        return Err(unsupported(
            "AV1 intra decoder does not support super-resolution, filters, restoration, or film grain",
        ));
    }
    let (frame, payload) = obus
        .iter()
        .find_map(|obu| match obu {
            Av1Obu::Frame { frame, payload, .. } => Some((frame, payload.as_slice())),
            _ => None,
        })
        .ok_or_else(|| malformed_error("AV1 intra unit has no frame OBU"))?;
    if frame.frame_type != Av1FrameType::Key || !frame.show_frame || !frame.disable_cdf_update {
        return Err(unsupported(
            "AV1 intra decoder requires a shown key frame with static CDFs",
        ));
    }
    let dimensions =
        VideoDimensions::new(sequence.max_frame_width, sequence.max_frame_height, limits)?;
    let width = usize::try_from(dimensions.width)
        .map_err(|_| resource("AV1 frame width is not representable"))?;
    let height = usize::try_from(dimensions.height)
        .map_err(|_| resource("AV1 frame height is not representable"))?;
    let mi_cols = 2usize
        .checked_mul(width.saturating_add(7) >> 3)
        .ok_or_else(|| resource("AV1 MI column count overflows"))?;
    let mi_rows = 2usize
        .checked_mul(height.saturating_add(7) >> 3)
        .ok_or_else(|| resource("AV1 MI row count overflows"))?;
    let header = parse_supported_frame_header(payload, mi_cols, mi_rows)?;
    let tile = payload
        .get(header.tile_offset..)
        .ok_or_else(|| malformed_error("AV1 tile data is truncated"))?;
    let mut decoder = LosslessTileDecoder::new(
        tile,
        width,
        height,
        mi_cols,
        mi_rows,
        header.base_q_idx,
        header.tx_mode_select,
        limits,
    )?;
    let luma = decoder.decode()?;
    let tx_sizes = decoder.tx_sizes;
    let range = if sequence.color_config.color_range {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };
    let mut video_frame =
        Av1IntraFrame::from_luma(dimensions, luma, range, limits)?.into_video_frame(limits)?;
    if header.base_q_idx != 0 {
        let mut filter_frame = FilterFrame::new_monochrome(FilterPlane::from_samples(
            width,
            height,
            video_frame.planes[0].data.clone(),
            limits,
        )?);
        deblock_frame(&mut filter_frame, &header.loop_filter, Some(&tx_sizes))?;
        video_frame.planes[0].data = filter_frame.y.data;
    }
    Ok((video_frame, tx_sizes))
}

/// The subset of `uncompressed_header()` this decoder parses, plus the tile
/// data's byte offset.
struct SupportedFrameHeader {
    base_q_idx: u8,
    loop_filter: LoopFilterParams,
    /// `tx_mode == TX_MODE_SELECT` (spec §5.9.20 `read_tx_mode`); `false`
    /// is `TX_MODE_LARGEST`, and lossless frames force `TX_MODE_ONLY_4X4`
    /// with no bit read.
    tx_mode_select: bool,
    tile_offset: usize,
}

fn parse_supported_frame_header(
    payload: &[u8],
    mi_cols: usize,
    mi_rows: usize,
) -> Result<SupportedFrameHeader> {
    let mut bits = HeaderBits::new(payload);
    require_bit(&mut bits, true, "disable_cdf_update")?;
    require_bit(&mut bits, false, "allow_screen_content_tools")?;
    require_bit(&mut bits, false, "render_and_frame_size_different")?;
    require_bit(&mut bits, true, "uniform_tile_spacing_flag")?;
    let sb_cols = mi_cols.saturating_add(15) >> 4;
    let sb_rows = mi_rows.saturating_add(15) >> 4;
    if tile_log2(1, sb_cols.min(64)) > 0 {
        require_bit(&mut bits, false, "increment_tile_cols_log2")?;
    }
    if tile_log2(1, sb_rows.min(64)) > 0 {
        require_bit(&mut bits, false, "increment_tile_rows_log2")?;
    }
    // quantization_params() (spec §5.9.12): base_q_idx selects between the
    // lossless (WHT) and non-lossless (dequantized DCT/IDTX) reconstruction
    // paths below. delta_q_y_dc/using_qmatrix are only ever signaled by this
    // decoder's encoder as 0 (no per-block delta-Q or quantizer-matrix
    // support), regardless of losslessness, so they stay required-false.
    let base_q_idx = u8::try_from(bits.read(8, "base_q_idx")?).expect("8 bits fit u8");
    let lossless = base_q_idx == 0;
    require_bit(&mut bits, false, "delta_q_y_dc")?;
    require_bit(&mut bits, false, "using_qmatrix")?;
    // segmentation_params()
    require_bit(&mut bits, false, "segmentation_enabled")?;
    // delta_q_params()/delta_lf_params(): both are only present when
    // base_q_idx > 0, and even then only when segmentation/delta-Q are
    // enabled; this decoder never signals either, so no bits are read.
    // loop_filter_params() (spec §5.9.11): `CodedLossless` (base_q_idx == 0
    // with every delta-Q field 0, which is exactly this decoder's lossless
    // path) forces every filter level to 0 with no bits read at all. The
    // non-lossless path parses a minimal but real subset: two luma levels
    // (vertical/horizontal), and when they're not both zero, the two chroma
    // levels, a 3-bit sharpness, and (deliberately out of scope for this
    // bounded decoder, see the module docs) no per-reference delta syntax —
    // `loop_filter_delta_enabled` is required to be 0.
    let loop_filter = if lossless {
        LoopFilterParams::DISABLED
    } else {
        let y_vertical_level =
            u8::try_from(bits.read(6, "loop_filter_level[0]")?).expect("6 bits fit u8");
        let y_horizontal_level =
            u8::try_from(bits.read(6, "loop_filter_level[1]")?).expect("6 bits fit u8");
        let (u_level, v_level) = if y_vertical_level != 0 || y_horizontal_level != 0 {
            let u_level =
                u8::try_from(bits.read(6, "loop_filter_level[2]")?).expect("6 bits fit u8");
            let v_level =
                u8::try_from(bits.read(6, "loop_filter_level[3]")?).expect("6 bits fit u8");
            (u_level, v_level)
        } else {
            (0, 0)
        };
        let sharpness =
            u8::try_from(bits.read(3, "loop_filter_sharpness")?).expect("3 bits fit u8");
        require_bit(&mut bits, false, "loop_filter_delta_enabled")?;
        LoopFilterParams {
            y_vertical_level,
            y_horizontal_level,
            u_level,
            v_level,
            sharpness,
        }
    };
    // cdef_params(), lr_params(): skipped because the sequence header is
    // required to have enable_cdef == 0 and enable_restoration == 0.
    // read_tx_mode() (spec §5.9.20): CodedLossless forces TX_MODE_ONLY_4X4
    // with no bit read at all; otherwise a single `tx_mode_select` bit
    // chooses between TX_MODE_LARGEST (one transform per coding block,
    // sized Max_Tx_Size_Rect[MiSize]) and TX_MODE_SELECT (a per-block
    // `tx_depth` symbol, read by `LosslessTileDecoder::read_tx_size`).
    let tx_mode_select = if lossless {
        false
    } else {
        bits.read(1, "tx_mode_select")? != 0
    };
    require_bit(&mut bits, true, "reduced_tx_set")?;
    while bits.position() & 7 != 0 {
        require_bit(&mut bits, false, "frame header byte alignment")?;
    }
    Ok(SupportedFrameHeader {
        base_q_idx,
        loop_filter,
        tx_mode_select,
        tile_offset: bits.position() / 8,
    })
}

fn require_bit(bits: &mut HeaderBits<'_>, expected: bool, name: &str) -> Result<()> {
    let value = bits.read(1, name)? != 0;
    if value != expected {
        Err(unsupported(format!("unsupported AV1 {name} value")))
    } else {
        Ok(())
    }
}

fn tile_log2(blocks: usize, target: usize) -> usize {
    let mut value = blocks;
    let mut log2 = 0;
    while value < target {
        value <<= 1;
        log2 += 1;
    }
    log2
}

struct HeaderBits<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> HeaderBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn position(&self) -> usize {
        self.bit
    }

    fn read(&mut self, count: usize, name: &str) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self
                .data
                .get(self.bit >> 3)
                .ok_or_else(|| malformed_error(format!("AV1 {name} is truncated")))?;
            value = (value << 1) | u32::from((byte >> (7 - (self.bit & 7))) & 1);
            self.bit = self
                .bit
                .checked_add(1)
                .ok_or_else(|| resource("AV1 frame header offset overflows"))?;
        }
        Ok(value)
    }
}

struct LosslessTileDecoder<'a> {
    symbols: Av1SymbolDecoder<'a>,
    width: usize,
    height: usize,
    mi_cols: usize,
    mi_rows: usize,
    coded_width: usize,
    coded_height: usize,
    base_q_idx: u8,
    tx_mode_select: bool,
    pixels: Vec<u8>,
    above_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_level: Vec<u8>,
    left_dc: Vec<u8>,
    mi_bsl: Vec<u8>,
    tx_sizes: TxSizeGrid,
    decoded_blocks: u32,
    max_blocks: u32,
}

impl<'a> LosslessTileDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tile: &'a [u8],
        width: usize,
        height: usize,
        mi_cols: usize,
        mi_rows: usize,
        base_q_idx: u8,
        tx_mode_select: bool,
        limits: &Limits,
    ) -> Result<Self> {
        let coded_width = mi_cols
            .checked_mul(4)
            .ok_or_else(|| resource("AV1 coded width overflows"))?;
        let coded_height = mi_rows
            .checked_mul(4)
            .ok_or_else(|| resource("AV1 coded height overflows"))?;
        let pixels = coded_width
            .checked_mul(coded_height)
            .ok_or_else(|| resource("AV1 coded plane size overflows"))?;
        let contexts = mi_cols
            .checked_mul(mi_rows)
            .ok_or_else(|| resource("AV1 partition context size overflows"))?;
        let allocation = pixels
            .checked_add(contexts)
            .and_then(|value| value.checked_add(mi_cols.saturating_mul(2)))
            .and_then(|value| value.checked_add(mi_rows.saturating_mul(2)))
            .ok_or_else(|| resource("AV1 reconstruction allocation overflows"))?;
        if u64::try_from(allocation).map_err(|_| resource("AV1 allocation is not representable"))?
            > limits.max_allocation_bytes
        {
            return Err(resource("AV1 reconstruction exceeds the allocation limit"));
        }
        Ok(Self {
            symbols: Av1SymbolDecoder::new(tile)?,
            width,
            height,
            mi_cols,
            mi_rows,
            coded_width,
            coded_height,
            base_q_idx,
            tx_mode_select,
            pixels: vec![0; pixels],
            above_level: vec![0; mi_cols],
            above_dc: vec![0; mi_cols],
            left_level: vec![0; mi_rows],
            left_dc: vec![0; mi_rows],
            mi_bsl: vec![0; contexts],
            tx_sizes: TxSizeGrid::new(width, height),
            decoded_blocks: 0,
            max_blocks: limits.max_av1_blocks_per_frame,
        })
    }

    fn decode(&mut self) -> Result<Vec<u8>> {
        const SUPERBLOCK_MI: usize = 16;
        let mut row = 0;
        while row < self.mi_rows {
            self.left_level.fill(0);
            self.left_dc.fill(0);
            let mut column = 0;
            while column < self.mi_cols {
                self.decode_partition(row, column, 64)?;
                column += SUPERBLOCK_MI;
            }
            row += SUPERBLOCK_MI;
        }
        let mut cropped = Vec::with_capacity(
            self.width
                .checked_mul(self.height)
                .ok_or_else(|| resource("AV1 cropped plane size overflows"))?,
        );
        for row in 0..self.height {
            let start = row * self.coded_width;
            cropped.extend_from_slice(&self.pixels[start..start + self.width]);
        }
        Ok(cropped)
    }

    fn decode_partition(&mut self, row: usize, column: usize, block_width: usize) -> Result<()> {
        if row >= self.mi_rows || column >= self.mi_cols {
            return Ok(());
        }
        let units = block_width / 4;
        let half = units >> 1;
        let has_rows = row + half < self.mi_rows;
        let has_columns = column + half < self.mi_cols;
        let bsl = units.trailing_zeros() as usize;
        let split = if block_width < 8 {
            false
        } else if has_rows && has_columns {
            let context = self.partition_context(row, column, bsl);
            match self.symbols.symbol(partition_cdf(bsl, context))? {
                0 => false,
                3 => true,
                _ => {
                    return Err(unsupported(
                        "AV1 intra decoder supports NONE and SPLIT partitions",
                    ));
                }
            }
        } else if has_columns {
            let context = self.partition_context(row, column, bsl);
            if self
                .symbols
                .symbol(&split_or_horz_cdf(partition_cdf(bsl, context)))?
                != 1
            {
                return Err(unsupported(
                    "AV1 horizontal edge partitions are not supported",
                ));
            }
            true
        } else if has_rows {
            let context = self.partition_context(row, column, bsl);
            if self
                .symbols
                .symbol(&split_or_vert_cdf(partition_cdf(bsl, context)))?
                != 1
            {
                return Err(unsupported(
                    "AV1 vertical edge partitions are not supported",
                ));
            }
            true
        } else {
            true
        };
        if split {
            let next = block_width / 2;
            self.decode_partition(row, column, next)?;
            self.decode_partition(row, column + half, next)?;
            self.decode_partition(row + half, column, next)?;
            self.decode_partition(row + half, column + half, next)
        } else {
            self.decode_block(row, column, block_width)
        }
    }

    fn partition_context(&self, row: usize, column: usize, bsl: usize) -> usize {
        let above = row > 0 && usize::from(self.mi_bsl[(row - 1) * self.mi_cols + column]) < bsl;
        let left = column > 0 && usize::from(self.mi_bsl[row * self.mi_cols + column - 1]) < bsl;
        usize::from(left) * 2 + usize::from(above)
    }

    fn decode_block(&mut self, row: usize, column: usize, block_width: usize) -> Result<()> {
        if self.symbols.symbol(&cdf::SKIP[0])? != 0 {
            return Err(unsupported("AV1 skipped intra blocks are not supported"));
        }
        let mode = read_intra_y_mode(self.symbols.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC)?)?;
        let units = block_width / 4;
        let bsl = units.trailing_zeros() as u8;
        for y in 0..units {
            for x in 0..units {
                let (mi_row, mi_column) = (row + y, column + x);
                if mi_row < self.mi_rows && mi_column < self.mi_cols {
                    self.mi_bsl[mi_row * self.mi_cols + mi_column] = bsl;
                }
            }
        }
        let tx_width = self.read_tx_size(block_width)?;
        let step = tx_width / 4;
        let mut transform_y = 0;
        while transform_y < units {
            let mut transform_x = 0;
            while transform_x < units {
                let x = column * 4 + transform_x * 4;
                let y = row * 4 + transform_y * 4;
                if x < self.coded_width && y < self.coded_height {
                    // Reconstruction writes the whole transform block, and
                    // this decoder's frame buffer is only MI-aligned rather
                    // than superblock-aligned, so a transform that would
                    // hang off the coded frame is rejected instead of
                    // clipped. Forced edge splits (§5.11.4) keep every
                    // conformant stream this decoder accepts inside the
                    // buffer for transforms up to the coded dimensions.
                    if x + tx_width > self.coded_width || y + tx_width > self.coded_height {
                        return Err(unsupported(
                            "AV1 transform block extends past the coded frame",
                        ));
                    }
                    self.decode_transform_block(x, y, tx_width, mode)?;
                }
                transform_x += step;
            }
            transform_y += step;
        }
        Ok(())
    }

    /// `read_tx_size` (spec §5.11.16) for a square `block_width` coding
    /// block, over the square transform sizes this crate's inverse
    /// transform kernels implement (`TX_4X4` through `TX_64X64`).
    ///
    /// Lossless frames use `TX_MODE_ONLY_4X4` (§5.9.20) and read nothing.
    /// Otherwise the block's largest transform is `Max_Tx_Size_Rect[MiSize]`,
    /// which for the square block sizes this decoder codes is the block
    /// width capped at `TX_64X64`. Under `TX_MODE_LARGEST` that is the
    /// answer with no symbol read; under `TX_MODE_SELECT` a `tx_depth`
    /// symbol halves it once (8x8 blocks, whose `Max_Tx_Depth` is 1) or up
    /// to twice (larger blocks, where the spec caps the coded depth at 2).
    fn read_tx_size(&mut self, block_width: usize) -> Result<usize> {
        if self.base_q_idx == 0 {
            return Ok(4);
        }
        let largest = block_width.min(MAX_TX_WIDTH);
        if largest <= 4 || !self.tx_mode_select {
            return Ok(largest);
        }
        let (depth_cdf, max_depth) = cdf::tx_depth_cdf(block_width);
        let depth = self.symbols.symbol(depth_cdf)?;
        if depth > max_depth {
            return Err(malformed_error("AV1 tx_depth exceeds the block maximum"));
        }
        Ok((largest >> depth).max(4))
    }

    fn decode_transform_block(
        &mut self,
        x: usize,
        y: usize,
        tx_width: usize,
        mode: Av1IntraMode,
    ) -> Result<()> {
        self.decoded_blocks = self
            .decoded_blocks
            .checked_add(1)
            .ok_or_else(|| resource("AV1 reconstruction work counter overflows"))?;
        if self.decoded_blocks > self.max_blocks {
            return Err(resource(
                "AV1 reconstruction exceeds the configured block limit",
            ));
        }
        if self.base_q_idx == 0 {
            let coefficients = self.decode_coefficients_4x4(x >> 2, y >> 2)?;
            let residuals = inverse_wht_4x4(&coefficients);
            self.apply_intra_prediction(x, y, 4, mode, &residuals);
            self.tx_sizes.set_block(x, y, 4, 4);
            return Ok(());
        }
        let (coefficients, tx_type) =
            self.decode_coefficients_nonlossless(x >> 2, y >> 2, tx_width)?;
        let dc_quant = get_dc_quant(self.base_q_idx);
        let ac_quant = get_ac_quant(self.base_q_idx);
        let residuals = inverse_transform(&coefficients, tx_width, tx_type, dc_quant, ac_quant);
        self.apply_intra_prediction(x, y, tx_width, mode, &residuals);
        self.tx_sizes.set_block(x, y, tx_width, tx_width);
        Ok(())
    }

    /// Writes a uniform intra `prediction` over a `size x size` block and
    /// adds the inverse-transformed residual through the SIMD kernels.
    fn apply_prediction(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        prediction: u8,
        residuals: &[i16],
    ) {
        for row in 0..size {
            let start = (y + row) * self.coded_width + x;
            let destination = &mut self.pixels[start..start + size];
            destination.fill(prediction);
            add_residual_row(&residuals[row * size..row * size + size], destination);
        }
    }

    fn apply_intra_prediction(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        mode: Av1IntraMode,
        residuals: &[i16],
    ) {
        if mode == Av1IntraMode::Dc {
            self.apply_prediction(x, y, size, self.dc_prediction_sized(x, y, size), residuals);
            return;
        }
        let top_left = if x > 0 && y > 0 {
            self.pixels[(y - 1) * self.coded_width + x - 1]
        } else {
            128
        };
        let mut top = vec![128; size];
        let mut left = vec![128; size];
        if y > 0 {
            top.copy_from_slice(&self.pixels[(y - 1) * self.coded_width + x..][..size]);
        }
        if x > 0 {
            for (row, sample) in left.iter_mut().enumerate() {
                *sample = self.pixels[(y + row) * self.coded_width + x - 1];
            }
        }
        let mut prediction = vec![0u8; size];
        for row in 0..size {
            match mode {
                Av1IntraMode::Dc => unreachable!(),
                Av1IntraMode::Vertical => prediction.copy_from_slice(&top),
                Av1IntraMode::Horizontal => prediction.fill(left[row]),
                Av1IntraMode::Paeth => paeth_row(top_left, &top, left[row], &mut prediction),
                Av1IntraMode::Smooth => {
                    smooth_row(SmoothMode::Smooth, &top, &left, row, &mut prediction)
                }
                Av1IntraMode::SmoothVertical => smooth_row(
                    SmoothMode::SmoothVertical,
                    &top,
                    &left,
                    row,
                    &mut prediction,
                ),
                Av1IntraMode::SmoothHorizontal => smooth_row(
                    SmoothMode::SmoothHorizontal,
                    &top,
                    &left,
                    row,
                    &mut prediction,
                ),
                Av1IntraMode::D45 => directional_row(45, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D63 => directional_row(63, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D67 => directional_row(67, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D113 => directional_row(113, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D135 => directional_row(135, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D157 => directional_row(157, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D203 => directional_row(203, &top, &left, row, true, &mut prediction),
                Av1IntraMode::Directional {
                    angle,
                    filter_edges,
                } => directional_row(angle, &top, &left, row, filter_edges, &mut prediction),
            }
            let start = (y + row) * self.coded_width + x;
            let destination = &mut self.pixels[start..start + size];
            destination.copy_from_slice(&prediction);
            add_residual_row(&residuals[row * size..row * size + size], destination);
        }
    }

    /// Spec §7.11.2 DC intra prediction, generalized from the original
    /// fixed 4x4 window to any `size x size` (4 or 8) transform block: the
    /// prediction is the rounded average of the `size` samples immediately
    /// above and/or to the left, or 128 when neither neighbor is available.
    fn dc_prediction_sized(&self, x: usize, y: usize, size: usize) -> u8 {
        match (y > 0, x > 0) {
            (true, true) => {
                let above = (y - 1) * self.coded_width + x;
                let mut sum = sum_samples(&self.pixels[above..above + size]);
                for offset in 0..size {
                    sum += u32::from(self.pixels[(y + offset) * self.coded_width + x - 1]);
                }
                let count = 2 * size as u32;
                ((sum + count / 2) / count) as u8
            }
            (false, true) => {
                let sum = (0..size)
                    .map(|offset| u32::from(self.pixels[(y + offset) * self.coded_width + x - 1]))
                    .sum::<u32>();
                let count = size as u32;
                ((sum + count / 2) / count) as u8
            }
            (true, false) => {
                let above = (y - 1) * self.coded_width + x;
                let sum = sum_samples(&self.pixels[above..above + size]);
                let count = size as u32;
                ((sum + count / 2) / count) as u8
            }
            (false, false) => 128,
        }
    }

    fn decode_coefficients_4x4(&mut self, x4: usize, y4: usize) -> Result<[i32; 16]> {
        let (coefficients, _skipped) =
            self.decode_coefficient_levels(x4, y4, 4, &cdf::DEFAULT_SCAN_4X4)?;
        let mut levels = [0i32; 16];
        levels.copy_from_slice(&coefficients);
        Ok(levels)
    }

    /// Decodes one non-lossless transform block's `tx_type` (spec §5.11.47
    /// `read_tx_type`, restricted to this decoder's reduced intra set,
    /// `{IDTX, DCT_DCT}`; `ADST_ADST` is rejected as unsupported) and
    /// dequantized-domain coefficient levels, returning coefficients in
    /// row-major order ready for [`inverse_transform`].
    fn decode_coefficients_nonlossless(
        &mut self,
        x4: usize,
        y4: usize,
        tx_width: usize,
    ) -> Result<(Vec<i32>, Av1TxType)> {
        let scan = cdf::up_right_diagonal_scan(tx_width.min(MAX_CODED_TX_WIDTH));
        let (coefficients, skipped) = self.decode_coefficient_levels(x4, y4, tx_width, &scan)?;
        // read_tx_type() (spec §5.11.47): tx_type is only signaled for a
        // transform block that actually has nonzero coefficients (a fully
        // skipped block is implicitly DCT_DCT, though its value is
        // irrelevant since inverse_transform of an all-zero input is zero
        // regardless of tx_type), and only when `get_tx_set` is not
        // TX_SET_DCTONLY — which for intra blocks excludes every transform
        // of 32x32 or larger.
        let tx_type = if skipped {
            Av1TxType::DctDct
        } else if let Some(ext_tx) = cdf::ext_tx_cdf(tx_width) {
            match self.symbols.symbol(ext_tx)? {
                0 => Av1TxType::Idtx,
                1 => Av1TxType::DctDct,
                _ => {
                    return Err(unsupported(
                        "AV1 intra decoder does not support the ADST_ADST transform type",
                    ));
                }
            }
        } else {
            Av1TxType::DctDct
        };
        Ok((coefficients, tx_type))
    }

    #[allow(clippy::too_many_lines)]
    fn decode_coefficient_levels(
        &mut self,
        x4: usize,
        y4: usize,
        size: usize,
        scan: &[usize],
    ) -> Result<(Vec<i32>, bool)> {
        let plane_type = 0;
        let count = size * size;
        let units = (size / 4).max(1);
        // Coefficients are coded only in the transform block's upper-left
        // 32x32 quadrant, so a 64x64 transform reads a 32x32 coefficient
        // block (with 32x32 scan order and contexts) and zeroes the rest.
        let coded = size.min(MAX_CODED_TX_WIDTH);
        let coded_count = coded * coded;
        debug_assert_eq!(scan.len(), coded_count);
        let skip_context = self.txb_skip_context(x4, y4, units);
        if self.symbols.symbol(&cdf::TXB_SKIP[skip_context])? == 1 {
            self.set_coefficient_context(x4, y4, units, 0, 0);
            return Ok((vec![0; count], true));
        }
        let eob_point = self.symbols.symbol(cdf::eob_pt_cdf(coded, plane_type))? + 1;
        let eob = if eob_point < 2 {
            eob_point
        } else {
            let bit_count = eob_point - 2;
            let mut extra = 0usize;
            if eob_point >= 3 {
                extra = self
                    .symbols
                    .symbol(&cdf::EOB_EXTRA[plane_type][eob_point - 3])?
                    << (bit_count - 1);
                if bit_count > 1 {
                    extra |= usize::try_from(self.symbols.literal((bit_count - 1) as u8)?)
                        .expect("literal is representable as usize");
                }
            }
            (1usize << (eob_point - 2)) + 1 + extra
        };
        if eob == 0 || eob > coded_count {
            return Err(malformed_error(
                "AV1 coefficient EOB is outside the transform block",
            ));
        }
        let mut levels = vec![0i32; coded_count];
        for coefficient in (0..eob).rev() {
            let position = scan[coefficient];
            let mut level = if coefficient == eob - 1 {
                i32::try_from(
                    self.symbols.symbol(
                        &cdf::COEFF_BASE_EOB[plane_type]
                            [coeff_base_eob_context(coefficient, coded_count)],
                    )? + 1,
                )
                .expect("coefficient base level fits i32")
            } else {
                i32::try_from(self.symbols.symbol(
                    &cdf::COEFF_BASE[plane_type][coeff_base_context(position, &levels, coded)],
                )?)
                .expect("coefficient base level fits i32")
            };
            if level > NUM_BASE_LEVELS {
                let context = coeff_br_context(position, &levels, coded);
                for _ in 0..4 {
                    let value =
                        i32::try_from(self.symbols.symbol(&cdf::COEFF_BR[plane_type][context])?)
                            .expect("coefficient range value fits i32");
                    level += value;
                    if value < 3 {
                        break;
                    }
                }
            }
            levels[position] = level;
        }
        for (coefficient, &position) in scan.iter().enumerate().take(eob) {
            if levels[position] == 0 {
                continue;
            }
            let negative = if coefficient == 0 {
                self.symbols
                    .symbol(&cdf::DC_SIGN[plane_type][self.dc_sign_context(x4, y4, units)])?
                    != 0
            } else {
                self.symbols.literal(1)? != 0
            };
            if levels[position] > COEFF_BASE_PLUS_RANGE {
                levels[position] = COEFF_BASE_PLUS_RANGE
                    + i32::try_from(self.decode_golomb()?)
                        .map_err(|_| resource("AV1 coefficient magnitude overflows"))?;
            }
            if negative {
                levels[position] = -levels[position];
            }
        }
        let cumulative = levels.iter().map(|value| value.abs()).sum::<i32>().min(63) as u8;
        let dc = if levels[0] == 0 {
            0
        } else if levels[0] < 0 {
            1
        } else {
            2
        };
        self.set_coefficient_context(x4, y4, units, cumulative, dc);
        if coded == size {
            return Ok((levels, false));
        }
        // Scatter the coded quadrant into the full transform block.
        let mut coefficients = vec![0i32; count];
        for row in 0..coded {
            coefficients[row * size..row * size + coded]
                .copy_from_slice(&levels[row * coded..(row + 1) * coded]);
        }
        Ok((coefficients, false))
    }

    fn decode_golomb(&mut self) -> Result<u32> {
        let mut leading = 0u8;
        while self.symbols.literal(1)? == 0 {
            leading = leading
                .checked_add(1)
                .ok_or_else(|| resource("AV1 Golomb prefix overflows"))?;
            if leading >= 31 {
                return Err(resource("AV1 Golomb coefficient exceeds 31 bits"));
            }
        }
        let suffix = if leading == 0 {
            0
        } else {
            self.symbols.literal(leading)?
        };
        Ok((1u32 << leading) | suffix)
    }

    /// Spec §5.11.39's trailing context update: a transform block leaves
    /// its cumulative level and DC sign category behind on *every* 4x4
    /// column and row it covers, not just its first.
    fn set_coefficient_context(&mut self, x4: usize, y4: usize, units: usize, level: u8, dc: u8) {
        for column in x4..(x4 + units).min(self.mi_cols) {
            self.above_level[column] = level;
            self.above_dc[column] = dc;
        }
        for row in y4..(y4 + units).min(self.mi_rows) {
            self.left_level[row] = level;
            self.left_dc[row] = dc;
        }
    }

    /// `getTXBSkipCtx` (spec §8.3.2), whose `top`/`left` are the maxima of
    /// the neighbouring level contexts across the transform block's own
    /// width and height in 4x4 units.
    fn txb_skip_context(&self, x4: usize, y4: usize, units: usize) -> usize {
        let top = self.above_level[x4..(x4 + units).min(self.mi_cols)]
            .iter()
            .copied()
            .max()
            .map_or(0, i32::from);
        let left = self.left_level[y4..(y4 + units).min(self.mi_rows)]
            .iter()
            .copied()
            .max()
            .map_or(0, i32::from);
        if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + usize::from(top.max(left) > 3)
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        }
    }

    /// `getDcSignCtx` (spec §8.3.2), summed over every 4x4 column and row
    /// the transform block covers.
    fn dc_sign_context(&self, x4: usize, y4: usize, units: usize) -> usize {
        let mut sum = 0i32;
        let above = &self.above_dc[x4..(x4 + units).min(self.mi_cols)];
        let left = &self.left_dc[y4..(y4 + units).min(self.mi_rows)];
        for &category in above.iter().chain(left.iter()) {
            if category == 1 {
                sum -= 1;
            } else if category == 2 {
                sum += 1;
            }
        }
        if sum < 0 {
            1
        } else if sum > 0 {
            2
        } else {
            0
        }
    }
}

fn partition_cdf(bsl: usize, context: usize) -> &'static [u16] {
    match bsl {
        1 => &cdf::PARTITION_W8[context],
        2 => &cdf::PARTITION_W16[context],
        3 => &cdf::PARTITION_W32[context],
        _ => &cdf::PARTITION_W64[context],
    }
}

fn split_or_horz_cdf(cdf: &[u16]) -> [u16; 2] {
    let probability = (cdf[2] - cdf[1])
        + (cdf[3] - cdf[2])
        + (cdf[4] - cdf[3])
        + (cdf[6] - cdf[5])
        + (cdf[7] - cdf[6])
        + (cdf[9] - cdf[8]);
    [32_768 - probability, 32_768]
}

fn split_or_vert_cdf(cdf: &[u16]) -> [u16; 2] {
    let probability = (cdf[1] - cdf[0])
        + (cdf[3] - cdf[2])
        + (cdf[4] - cdf[3])
        + (cdf[5] - cdf[4])
        + (cdf[6] - cdf[5])
        + (cdf[8] - cdf[7]);
    [32_768 - probability, 32_768]
}

/// `get_lower_levels_ctx_eob` (spec §8.3.2): the last coded coefficient's
/// context is chosen by where it falls in the block's scan, so the
/// thresholds scale with the block's coefficient `count` rather than being
/// fixed at TX_4X4's 2 and 4.
fn coeff_base_eob_context(coefficient: usize, count: usize) -> usize {
    if coefficient == 0 {
        0
    } else if coefficient <= count / 8 {
        1
    } else if coefficient <= count / 4 {
        2
    } else {
        3
    }
}

fn coeff_base_context(position: usize, levels: &[i32], size: usize) -> usize {
    let (row, column) = (position / size, position % size);
    let mut magnitude = 0i32;
    for &(delta_row, delta_column) in &cdf::SIG_REF_DIFF_OFFSET_2D {
        let (neighbor_row, neighbor_column) = (row + delta_row, column + delta_column);
        if neighbor_row < size && neighbor_column < size {
            magnitude += levels[neighbor_row * size + neighbor_column].abs().min(3);
        }
    }
    let context = (((magnitude + 1) >> 1).min(4)) as usize;
    if row == 0 && column == 0 {
        0
    } else {
        context + cdf::coeff_base_ctx_offset(row, column)
    }
}

fn coeff_br_context(position: usize, levels: &[i32], size: usize) -> usize {
    let (row, column) = (position / size, position % size);
    let mut magnitude = 0i32;
    for &(delta_row, delta_column) in &cdf::MAG_REF_OFFSET_2D {
        let (neighbor_row, neighbor_column) = (row + delta_row, column + delta_column);
        if neighbor_row < size && neighbor_column < size {
            magnitude += levels[neighbor_row * size + neighbor_column].abs().min(15);
        }
    }
    let magnitude = (((magnitude + 1) >> 1).min(6)) as usize;
    if position == 0 {
        magnitude
    } else if row < 2 && column < 2 {
        magnitude + 7
    } else {
        magnitude + 14
    }
}

fn malformed_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn resource(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn read_intra_y_mode(symbol: usize) -> Result<Av1IntraMode> {
    match symbol {
        0 => Ok(Av1IntraMode::Dc),
        1 => Ok(Av1IntraMode::Vertical),
        2 => Ok(Av1IntraMode::Horizontal),
        3 => Ok(Av1IntraMode::D45),
        4 => Ok(Av1IntraMode::D135),
        5 => Ok(Av1IntraMode::D113),
        6 => Ok(Av1IntraMode::D157),
        7 => Ok(Av1IntraMode::D203),
        8 => Ok(Av1IntraMode::D67),
        9 => Ok(Av1IntraMode::Smooth),
        10 => Ok(Av1IntraMode::SmoothVertical),
        11 => Ok(Av1IntraMode::SmoothHorizontal),
        12 => Ok(Av1IntraMode::Paeth),
        _ => Err(unsupported("AV1 intra decoder read an invalid y mode")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1_inter_decoder::test_encoder::{BitWriter, SymbolEncoder};

    #[test]
    fn rejects_missing_sequence_and_zero_work_limit() {
        assert_eq!(
            decode_av1_lossless_intra(&[0x32, 0], &Limits::default())
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
        let limits = Limits {
            max_av1_blocks_per_frame: 0,
            ..Limits::default()
        };
        assert_eq!(
            decode_av1_lossless_intra(&[0], &limits).unwrap_err().kind(),
            ErrorKind::ResourceLimit
        );
    }

    const FRAME_DIM: u32 = 16;

    /// Appends one OBU (`obu_header` with no extension and always a size
    /// field, followed by a leb128 payload length), mirroring the helper of
    /// the same name in [`crate::av1_inter_decoder`]'s own tests.
    fn push_obu(stream: &mut Vec<u8>, obu_type: u8, payload: &[u8]) {
        stream.push((obu_type << 3) | 0x02);
        let mut len = payload.len() as u64;
        loop {
            let mut byte = (len & 0x7f) as u8;
            len >>= 7;
            if len != 0 {
                byte |= 0x80;
            }
            stream.push(byte);
            if len == 0 {
                break;
            }
        }
        stream.extend_from_slice(payload);
    }

    /// A reduced-still, 8-bit monochrome sequence header for a
    /// `width x height` frame, matching every bound
    /// `decode_av1_lossless_intra_with_tx_sizes` requires.
    fn sequence_header_payload(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(0, 3); // seq_profile = Main
        w.bits(1, 1); // still_picture
        w.bits(1, 1); // reduced_still_picture_header
        w.bits(0, 5); // seq_level_idx (<= 7, so no seq_tier bit)
        let width_bits = (u32::BITS - (width - 1).leading_zeros()).max(1) as usize;
        let height_bits = (u32::BITS - (height - 1).leading_zeros()).max(1) as usize;
        w.bits(width_bits as u32 - 1, 4); // frame_width_bits_minus_1
        w.bits(height_bits as u32 - 1, 4); // frame_height_bits_minus_1
        w.bits(width - 1, width_bits); // max_frame_width_minus_1
        w.bits(height - 1, height_bits); // max_frame_height_minus_1
        w.bits(0, 1); // use_128x128_superblock
        w.bits(0, 1); // enable_filter_intra
        w.bits(0, 1); // enable_intra_edge_filter
        w.bits(0, 1); // enable_superres
        w.bits(0, 1); // enable_cdef
        w.bits(0, 1); // enable_restoration
        // color_config()
        w.bits(0, 1); // high_bitdepth -> 8-bit
        w.bits(1, 1); // mono_chrome
        w.bits(0, 1); // color_description_present_flag
        w.bits(0, 1); // color_range (monochrome path only reads this flag)
        w.bits(0, 1); // film_grain_params_present
        w.bits(1, 1); // trailing_one_bit
        w.into_bytes()
    }

    /// Frame-header bits up to and including `loop_filter_params()`,
    /// matching `parse_supported_frame_header` exactly for a reduced-still
    /// key frame.
    fn frame_header_payload(
        base_q_idx: u8,
        loop_filter: Option<(u8, u8, u8, u8, u8)>,
        tx_mode_select: bool,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(1, 1); // disable_cdf_update = 1
        w.bits(0, 1); // allow_screen_content_tools
        w.bits(0, 1); // render_and_frame_size_different
        w.bits(1, 1); // uniform_tile_spacing_flag
        // mi_cols = mi_rows = 4 for a 16x16 frame -> sb_cols = sb_rows = 1,
        // so tile_log2(1, 1) == 0 and no increment bits are read.
        w.bits(base_q_idx.into(), 8); // base_q_idx
        w.bits(0, 1); // delta_q_y_dc
        w.bits(0, 1); // using_qmatrix
        w.bits(0, 1); // segmentation_enabled
        if let Some((y_v, y_h, u, v, sharpness)) = loop_filter {
            w.bits(y_v.into(), 6);
            w.bits(y_h.into(), 6);
            if y_v != 0 || y_h != 0 {
                w.bits(u.into(), 6);
                w.bits(v.into(), 6);
            }
            w.bits(sharpness.into(), 3);
            w.bits(0, 1); // loop_filter_delta_enabled
        }
        if base_q_idx != 0 {
            // read_tx_mode(): only present when the frame is not
            // CodedLossless.
            w.bits(u32::from(tx_mode_select), 1);
        }
        w.bits(1, 1); // reduced_tx_set
        w.byte_align();
        w.into_bytes()
    }

    /// A non-lossless 16x16 tile: the top-level 64/32-unit partitions are
    /// forced splits with no symbol read (mi_cols = mi_rows = 4 is smaller
    /// than every half-block threshold above bsl=2), landing on a real
    /// `PARTITION_W16` read that selects SPLIT into four 8x8 coding blocks.
    /// Each 8x8 block reads `PARTITION_W8` = NONE, `skip` = 0, `DC_PRED`,
    /// then (because `base_q_idx != 0`) a `tx_type` symbol and one 8x8
    /// transform's coefficients. Blocks (0,0) and (2,2) (mi coordinates)
    /// carry a large DC coefficient of opposite sign (`DCT_DCT`); blocks
    /// (0,2) and (2,0) are `TXB_SKIP`. Because DC_PRED inherits from
    /// already-reconstructed neighbor samples, a single dark block would
    /// otherwise cascade through every later DC-predicted block and leave
    /// the whole plane flat; alternating a strong positive DC onto block
    /// (2,2) (diagonal from block (0,0), so neither of its own two DC_PRED
    /// neighbors is block (0,0) directly) keeps a real, non-cascading step
    /// edge in the final reconstruction for [`deblock_frame`] to smooth.
    ///
    /// `txb_skip_context` at 8x8 depends on `above_level`/`left_level` left
    /// behind by earlier blocks in the same row/column, precomputed here by
    /// walking the same recurrence `LosslessTileDecoder::txb_skip_context`/
    /// `set_coefficient_context` use, in decode order (blocks visited in mi
    /// order (0,0), (0,2), (2,0), (2,2), each an 8x8 block spanning a 2x2 mi
    /// footprint, `x4`/`y4` at mi-column/row 0 or 2): block (0,0) sees no
    /// neighbors set yet (`top == 0 && left == 0` -> context 1), leaving
    /// `above_level[0]` and `left_level[0]` both at the clamped cumulative
    /// level (`14`, from the large DC coefficient below). Block (0,2) then
    /// sees `above_level[2] == 0`, `left_level[0] == 14` (one zero
    /// neighbor, `max > 3`) -> context 3; block (2,0) sees the symmetric
    /// `above_level[0] == 14`, `left_level[2] == 0` -> context 3. Both are
    /// skipped, so they reset `above_level[2]`/`left_level[2]` back to 0,
    /// leaving block (2,2) with both neighbors 0 -> context 1 again.
    fn non_lossless_key_frame_tile() -> Vec<u8> {
        const CONTEXTS: [usize; 4] = [1, 3, 3, 1];
        let mut e = SymbolEncoder::new();
        e.symbol(&cdf::PARTITION_W16[0], 3); // SPLIT into four 8x8 blocks
        for (block, &context) in CONTEXTS.iter().enumerate() {
            e.symbol(&cdf::PARTITION_W8[0], 0); // NONE -> decode_block(_, _, 8)
            e.symbol(&cdf::SKIP[0], 0); // skip = 0
            e.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC, 0); // DC_PRED
            if block == 0 || block == 3 {
                e.symbol(&cdf::TXB_SKIP[context], 0); // not skipped
                e.symbol(&cdf::EOB_PT_64[0][0], 0); // eob_point = 1 -> eob = 1
                e.symbol(&cdf::COEFF_BASE_EOB[0][0], 2); // level = 3 (max base)
                e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
                e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
                e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
                e.symbol(&cdf::COEFF_BR[0][0], 2); // +2, stop (level = 14)
                e.symbol(&cdf::DC_SIGN[0][0], usize::from(block == 0)); // block 0 negative, block 3 positive
                e.symbol(&cdf::EXT_TX_INTRA_REDUCED[1], 1); // DCT_DCT (8x8)
            } else {
                e.symbol(&cdf::TXB_SKIP[context], 1); // skipped -> all zero
            }
        }
        e.finish()
    }

    /// Wraps `sequence_header_payload()` and `non_lossless_key_frame_tile()`
    /// into a complete low-overhead temporal unit.
    fn non_lossless_key_frame_temporal_unit(
        base_q_idx: u8,
        loop_filter: Option<(u8, u8, u8, u8, u8)>,
    ) -> Vec<u8> {
        let mut payload = frame_header_payload(base_q_idx, loop_filter, false);
        payload.extend_from_slice(&non_lossless_key_frame_tile());

        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]); // temporal delimiter
        push_obu(
            &mut stream,
            1,
            &sequence_header_payload(FRAME_DIM, FRAME_DIM),
        );
        push_obu(&mut stream, 6, &payload); // Frame OBU
        stream
    }

    #[test]
    fn non_lossless_stream_decodes_with_an_8x8_transform_and_nonzero_dc() {
        let limits = Limits::default();
        let stream = non_lossless_key_frame_temporal_unit(40, Some((30, 30, 0, 0, 0)));
        let (frame, tx_sizes) = decode_av1_lossless_intra_with_tx_sizes(&stream, &limits).unwrap();
        assert_eq!(
            (frame.dimensions.width, frame.dimensions.height),
            (FRAME_DIM, FRAME_DIM)
        );
        // Every coding block in this fixture is 8x8 (TX_MODE_LARGEST picks
        // an 8x8 transform for any coding block that size or larger), so
        // the whole grid should record non-default (non-4x4) transforms.
        let mut expected = TxSizeGrid::new(FRAME_DIM as usize, FRAME_DIM as usize);
        expected.set_block(0, 0, 8, 8);
        expected.set_block(8, 0, 8, 8);
        expected.set_block(0, 8, 8, 8);
        expected.set_block(8, 8, 8, 8);
        assert_eq!(tx_sizes, expected);
    }

    #[test]
    fn non_lossless_tx_size_grid_reaches_deblock_frame_and_changes_the_result() {
        let limits = Limits::default();
        let stream = non_lossless_key_frame_temporal_unit(40, Some((30, 30, 0, 0, 0)));
        let (frame, tx_sizes) = decode_av1_lossless_intra_with_tx_sizes(&stream, &limits).unwrap();

        // Re-run the reconstruction without deblocking (base_q_idx == 0
        // reuses the exact same tile decoder path) by decoding the
        // unfiltered plane directly, to compare against the filtered
        // output above and confirm deblocking actually changed samples.
        let mut parser = Av1Parser::new(limits).unwrap();
        let obus = parser.parse_low_overhead(&stream).unwrap();
        let sequence = obus
            .iter()
            .find_map(|obu| match obu {
                Av1Obu::SequenceHeader { sequence, .. } => Some(sequence),
                _ => None,
            })
            .unwrap();
        let (_, payload) = obus
            .iter()
            .find_map(|obu| match obu {
                Av1Obu::Frame { frame, payload, .. } => Some((frame, payload.as_slice())),
                _ => None,
            })
            .unwrap();
        let dimensions =
            VideoDimensions::new(sequence.max_frame_width, sequence.max_frame_height, &limits)
                .unwrap();
        let width = dimensions.width as usize;
        let height = dimensions.height as usize;
        let mi_cols = 2 * ((width + 7) >> 3);
        let mi_rows = 2 * ((height + 7) >> 3);
        let header = parse_supported_frame_header(payload, mi_cols, mi_rows).unwrap();
        let tile = &payload[header.tile_offset..];
        let mut decoder = LosslessTileDecoder::new(
            tile,
            width,
            height,
            mi_cols,
            mi_rows,
            header.base_q_idx,
            header.tx_mode_select,
            &limits,
        )
        .unwrap();
        let unfiltered_luma = decoder.decode().unwrap();

        assert_ne!(unfiltered_luma, frame.planes[0].data);

        // The grid genuinely reaches deblock_frame end to end: build a
        // fresh FilterFrame from the unfiltered reconstruction and confirm
        // running deblock_frame with the decoded (non-default) TxSizeGrid
        // reproduces the same filtered output the decoder already returned.
        let mut filter_frame = FilterFrame::new_monochrome(
            FilterPlane::from_samples(width, height, unfiltered_luma.clone(), &limits).unwrap(),
        );
        deblock_frame(&mut filter_frame, &header.loop_filter, Some(&tx_sizes)).unwrap();
        assert_eq!(filter_frame.y.data, frame.planes[0].data);
    }

    /// A frame exactly one 64x64 superblock across, so `PARTITION_NONE`
    /// yields a single 64x64 coding block and `read_tx_size` can reach
    /// every transform size this crate's kernels implement.
    const SUPERBLOCK_DIM: u32 = 64;
    /// The quantizer index the large-transform fixtures below code at.
    const SUPERBLOCK_Q: u8 = 40;

    /// Wraps a hand-authored single-superblock tile into a complete
    /// low-overhead temporal unit at [`SUPERBLOCK_DIM`], with every loop
    /// filter level 0 so `deblock_frame` is a no-op and the assertions
    /// below see the raw reconstruction.
    fn superblock_temporal_unit(tile: &[u8], tx_mode_select: bool) -> Vec<u8> {
        let mut payload = frame_header_payload(SUPERBLOCK_Q, Some((0, 0, 0, 0, 0)), tx_mode_select);
        payload.extend_from_slice(tile);
        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]); // temporal delimiter
        push_obu(
            &mut stream,
            1,
            &sequence_header_payload(SUPERBLOCK_DIM, SUPERBLOCK_DIM),
        );
        push_obu(&mut stream, 6, &payload); // Frame OBU
        stream
    }

    /// The symbols shared by every fixture below: one `PARTITION_NONE`
    /// 64x64 coding block, not skipped, `DC_PRED`.
    fn superblock_prefix(e: &mut SymbolEncoder) {
        e.symbol(&cdf::PARTITION_W64[0], 0); // PARTITION_NONE
        e.symbol(&cdf::SKIP[0], 0); // skip = 0
        e.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC, 0); // DC_PRED
    }

    /// Codes one transform block holding a single positive DC coefficient
    /// of level 4 (`COEFF_BASE_EOB` = 3, then one `COEFF_BR` increment of
    /// 1, which is below 3 and so terminates the range extension).
    fn encode_dc_only_transform_block(e: &mut SymbolEncoder, skip_context: usize, coded: usize) {
        e.symbol(&cdf::TXB_SKIP[skip_context], 0); // not skipped
        e.symbol(cdf::eob_pt_cdf(coded, 0), 0); // eob_point = 1 -> eob = 1
        e.symbol(&cdf::COEFF_BASE_EOB[0][0], 2); // level = 3 (max base)
        e.symbol(&cdf::COEFF_BR[0][0], 1); // +1, stop (level = 4)
        e.symbol(&cdf::DC_SIGN[0][0], 0); // positive
    }

    /// The reconstruction a `size x size` transform carrying DC level 4
    /// produces on top of the 128 DC prediction an unbordered block gets.
    fn dc_only_reconstruction(size: usize) -> Vec<u8> {
        let mut coefficients = vec![0i32; size * size];
        coefficients[0] = 4;
        let residuals = inverse_transform(
            &coefficients,
            size,
            Av1TxType::DctDct,
            get_dc_quant(SUPERBLOCK_Q),
            get_ac_quant(SUPERBLOCK_Q),
        );
        residuals
            .iter()
            .map(|&residual| (128 + i32::from(residual)).clamp(0, 255) as u8)
            .collect()
    }

    #[test]
    fn tx_mode_largest_reaches_the_64_point_kernel_from_a_bitstream() {
        let mut e = SymbolEncoder::new();
        superblock_prefix(&mut e);
        // TX_MODE_LARGEST: no tx_depth symbol, so Max_Tx_Size_Rect for a
        // 64x64 block selects TX_64X64. Its coefficients are coded in the
        // upper-left 32x32 quadrant only, hence the 32-wide eob_pt class,
        // and TX_64X64's transform set is TX_SET_DCTONLY, so no tx_type
        // symbol follows the coefficients either.
        encode_dc_only_transform_block(&mut e, 1, 32);
        let stream = superblock_temporal_unit(&e.finish(), false);

        let limits = Limits::default();
        let (frame, tx_sizes) = decode_av1_lossless_intra_with_tx_sizes(&stream, &limits).unwrap();
        assert_eq!(
            (frame.dimensions.width, frame.dimensions.height),
            (SUPERBLOCK_DIM, SUPERBLOCK_DIM)
        );
        let mut expected = TxSizeGrid::new(SUPERBLOCK_DIM as usize, SUPERBLOCK_DIM as usize);
        expected.set_block(0, 0, 64, 64);
        assert_eq!(tx_sizes, expected);
        // The plane really came through the 64-point inverse DCT: a lone
        // DC coefficient spreads a constant offset over all 64x64 samples.
        assert_eq!(frame.planes[0].data, dc_only_reconstruction(64));
        assert_ne!(frame.planes[0].data[0], 128);
    }

    #[test]
    fn tx_mode_select_signals_a_32_point_transform_and_reconstructs_through_it() {
        let mut e = SymbolEncoder::new();
        superblock_prefix(&mut e);
        e.symbol(cdf::tx_depth_cdf(64).0, 1); // tx_depth = 1 -> TX_32X32
        // The first of the block's four 32x32 transforms carries a DC
        // coefficient; the other three are TXB_SKIP. Their skip contexts
        // follow the same `set_coefficient_context` recurrence the decoder
        // walks: the coded transform leaves its clamped cumulative level
        // (4) on 4x4 columns/rows 0..8, so the transforms to its right and
        // below each see exactly one nonzero, greater-than-3 neighbour
        // (context 3), and the last one sees neither (context 1).
        encode_dc_only_transform_block(&mut e, 1, 32);
        for context in [3, 3, 1] {
            e.symbol(&cdf::TXB_SKIP[context], 1); // skipped -> all zero
        }
        let stream = superblock_temporal_unit(&e.finish(), true);

        let limits = Limits::default();
        let (frame, tx_sizes) = decode_av1_lossless_intra_with_tx_sizes(&stream, &limits).unwrap();
        let mut expected = TxSizeGrid::new(SUPERBLOCK_DIM as usize, SUPERBLOCK_DIM as usize);
        for (x, y) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
            expected.set_block(x, y, 32, 32);
        }
        assert_eq!(tx_sizes, expected);

        let reconstruction = dc_only_reconstruction(32);
        let width = SUPERBLOCK_DIM as usize;
        for row in 0..32 {
            assert_eq!(
                &frame.planes[0].data[row * width..row * width + 32],
                &reconstruction[row * 32..row * 32 + 32],
                "row {row}"
            );
        }
    }

    #[test]
    fn tx_mode_select_depth_two_signals_16x16_transforms() {
        let mut e = SymbolEncoder::new();
        superblock_prefix(&mut e);
        e.symbol(cdf::tx_depth_cdf(64).0, 2); // tx_depth = 2 -> TX_16X16
        // All sixteen 16x16 transforms are skipped, so every one of them
        // sees zeroed level contexts and codes at skip context 1.
        for _ in 0..16 {
            e.symbol(&cdf::TXB_SKIP[1], 1);
        }
        let stream = superblock_temporal_unit(&e.finish(), true);

        let limits = Limits::default();
        let (frame, tx_sizes) = decode_av1_lossless_intra_with_tx_sizes(&stream, &limits).unwrap();
        let mut expected = TxSizeGrid::new(SUPERBLOCK_DIM as usize, SUPERBLOCK_DIM as usize);
        for y in (0..64).step_by(16) {
            for x in (0..64).step_by(16) {
                expected.set_block(x, y, 16, 16);
            }
        }
        assert_eq!(tx_sizes, expected);
        // Every transform block is skipped, so the frame is the flat 128
        // DC prediction an unbordered first block starts from.
        assert!(frame.planes[0].data.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn a_transform_block_hanging_off_the_coded_frame_is_rejected() {
        // A 40x40 frame is 10 MI units across, so the forced-split rules
        // still let a 64x64 PARTITION_NONE block through (8 < 10) even
        // though the coded plane is only 40 samples wide. The resulting
        // TX_64X64 would write outside the reconstruction buffer.
        let mut e = SymbolEncoder::new();
        superblock_prefix(&mut e);
        encode_dc_only_transform_block(&mut e, 1, 32);
        let tile = e.finish();
        let mut payload = frame_header_payload(SUPERBLOCK_Q, Some((0, 0, 0, 0, 0)), false);
        payload.extend_from_slice(&tile);
        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]);
        push_obu(&mut stream, 1, &sequence_header_payload(40, 40));
        push_obu(&mut stream, 6, &payload);

        assert_eq!(
            decode_av1_lossless_intra(&stream, &Limits::default())
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }
}

/// Focused, lower-level tests of [`LosslessTileDecoder::decode_coefficient_levels`]
/// in isolation from the full partition/prediction pipeline exercised by
/// `mod tests` above.
#[cfg(test)]
mod coefficient_level_tests {
    use super::LosslessTileDecoder;
    use crate::Limits;
    use crate::av1_cdf as cdf;
    use crate::av1_inter_decoder::test_encoder::SymbolEncoder;

    #[test]
    fn decodes_a_golomb_extended_negative_dc_coefficient() {
        let mut e = SymbolEncoder::new();
        e.symbol(&cdf::TXB_SKIP[1], 0); // not skipped
        e.symbol(&cdf::EOB_PT_64[0][0], 0); // eob_point = 1 -> eob = 1
        e.symbol(&cdf::COEFF_BASE_EOB[0][0], 2); // level = 3 (max base)
        e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
        e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
        e.symbol(&cdf::COEFF_BR[0][0], 3); // +3, keep extending
        e.symbol(&cdf::COEFF_BR[0][0], 2); // +2, stop (level = 14)
        e.symbol(&cdf::DC_SIGN[0][0], 1); // negative
        let bytes = e.finish();
        let limits = Limits::default();
        let mut decoder =
            LosslessTileDecoder::new(&bytes, 16, 16, 4, 4, 40, false, &limits).unwrap();
        let scan = cdf::up_right_diagonal_scan(8);
        let (levels, skipped) = decoder.decode_coefficient_levels(0, 0, 8, &scan).unwrap();
        assert!(!skipped);
        assert_eq!(levels[0], -14);
    }
}
