//! Dependency-free reconstruction of bounded AV1 Main-profile lossless intra
//! frames. The implemented syntax is the standards-compliant reduced-still,
//! 8-bit monochrome, single-tile subset used by zvidlib's native AV1 encoder.
//! Every other syntax branch is rejected explicitly.

use crate::av1_cdf as cdf;
use crate::{
    Av1FrameType, Av1IntraFrame, Av1Obu, Av1ObuType, Av1Parser, Av1SymbolDecoder, Av1SyntaxSupport,
    ColorRange, Error, ErrorKind, Limits, Result, VideoDimensions, VideoFrame, inverse_wht_4x4,
};

const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_PLUS_RANGE: i32 = 14;

/// Decodes one low-overhead AV1 temporal unit into validated YUV planes.
pub fn decode_av1_lossless_intra(bytes: &[u8], limits: &Limits) -> Result<VideoFrame> {
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
    let tile_offset = parse_supported_frame_header(payload, mi_cols, mi_rows)?;
    let tile = payload
        .get(tile_offset..)
        .ok_or_else(|| malformed_error("AV1 tile data is truncated"))?;
    let mut decoder = LosslessTileDecoder::new(tile, width, height, mi_cols, mi_rows, limits)?;
    let luma = decoder.decode()?;
    let range = if sequence.color_config.color_range {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };
    Av1IntraFrame::from_luma(dimensions, luma, range, limits)?.into_video_frame(limits)
}

fn parse_supported_frame_header(payload: &[u8], mi_cols: usize, mi_rows: usize) -> Result<usize> {
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
    if bits.read(8, "base_q_idx")? != 0 {
        return Err(unsupported(
            "AV1 intra decoder currently requires lossless quantization",
        ));
    }
    require_bit(&mut bits, false, "delta_q_y_dc")?;
    require_bit(&mut bits, false, "using_qmatrix")?;
    require_bit(&mut bits, false, "segmentation_enabled")?;
    require_bit(&mut bits, true, "reduced_tx_set")?;
    while bits.position() & 7 != 0 {
        require_bit(&mut bits, false, "frame header byte alignment")?;
    }
    Ok(bits.position() / 8)
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
    pixels: Vec<u8>,
    above_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_level: Vec<u8>,
    left_dc: Vec<u8>,
    mi_bsl: Vec<u8>,
    decoded_blocks: u32,
    max_blocks: u32,
}

impl<'a> LosslessTileDecoder<'a> {
    fn new(
        tile: &'a [u8],
        width: usize,
        height: usize,
        mi_cols: usize,
        mi_rows: usize,
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
            pixels: vec![0; pixels],
            above_level: vec![0; mi_cols],
            above_dc: vec![0; mi_cols],
            left_level: vec![0; mi_rows],
            left_dc: vec![0; mi_rows],
            mi_bsl: vec![0; contexts],
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
        if self.symbols.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC)? != 0 {
            return Err(unsupported(
                "AV1 intra decoder currently supports DC_PRED blocks",
            ));
        }
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
        for transform_y in 0..units {
            for transform_x in 0..units {
                let x = column * 4 + transform_x * 4;
                let y = row * 4 + transform_y * 4;
                if x < self.coded_width && y < self.coded_height {
                    self.decode_transform_block(x, y, block_width)?;
                }
            }
        }
        Ok(())
    }

    fn decode_transform_block(&mut self, x: usize, y: usize, block_width: usize) -> Result<()> {
        self.decoded_blocks = self
            .decoded_blocks
            .checked_add(1)
            .ok_or_else(|| resource("AV1 reconstruction work counter overflows"))?;
        if self.decoded_blocks > self.max_blocks {
            return Err(resource(
                "AV1 reconstruction exceeds the configured block limit",
            ));
        }
        let coefficients = self.decode_coefficients(x >> 2, y >> 2, block_width)?;
        let residuals = inverse_wht_4x4(&coefficients);
        let prediction = self.dc_prediction(x, y);
        for row in 0..4 {
            for column in 0..4 {
                self.pixels[(y + row) * self.coded_width + x + column] =
                    (i16::from(prediction) + residuals[row * 4 + column]).clamp(0, 255) as u8;
            }
        }
        Ok(())
    }

    fn dc_prediction(&self, x: usize, y: usize) -> u8 {
        match (y > 0, x > 0) {
            (true, true) => {
                let mut sum = 0u32;
                for offset in 0..4 {
                    sum += u32::from(self.pixels[(y - 1) * self.coded_width + x + offset]);
                    sum += u32::from(self.pixels[(y + offset) * self.coded_width + x - 1]);
                }
                ((sum + 4) >> 3) as u8
            }
            (false, true) => {
                let sum = (0..4)
                    .map(|offset| u32::from(self.pixels[(y + offset) * self.coded_width + x - 1]))
                    .sum::<u32>();
                ((sum + 2) >> 2) as u8
            }
            (true, false) => {
                let sum = (0..4)
                    .map(|offset| u32::from(self.pixels[(y - 1) * self.coded_width + x + offset]))
                    .sum::<u32>();
                ((sum + 2) >> 2) as u8
            }
            (false, false) => 128,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn decode_coefficients(
        &mut self,
        x4: usize,
        y4: usize,
        block_width: usize,
    ) -> Result<[i32; 16]> {
        let plane_type = 0;
        let scan = &cdf::DEFAULT_SCAN_4X4;
        let skip_context = self.txb_skip_context(x4, y4, block_width);
        if self.symbols.symbol(&cdf::TXB_SKIP[skip_context])? == 1 {
            self.set_coefficient_context(x4, y4, 0, 0);
            return Ok([0; 16]);
        }
        let eob_point = self.symbols.symbol(&cdf::EOB_PT_16[plane_type][0])? + 1;
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
        if eob == 0 || eob > 16 {
            return Err(malformed_error(
                "AV1 coefficient EOB is outside a 4x4 transform",
            ));
        }
        let mut levels = [0i32; 16];
        for coefficient in (0..eob).rev() {
            let position = scan[coefficient];
            let mut level =
                if coefficient == eob - 1 {
                    i32::try_from(
                        self.symbols.symbol(
                            &cdf::COEFF_BASE_EOB[plane_type][coeff_base_eob_context(coefficient)],
                        )? + 1,
                    )
                    .expect("coefficient base level fits i32")
                } else {
                    i32::try_from(self.symbols.symbol(
                        &cdf::COEFF_BASE[plane_type][coeff_base_context(position, &levels)],
                    )?)
                    .expect("coefficient base level fits i32")
                };
            if level > NUM_BASE_LEVELS {
                let context = coeff_br_context(position, &levels);
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
                    .symbol(&cdf::DC_SIGN[plane_type][self.dc_sign_context(x4, y4)])?
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
        self.set_coefficient_context(x4, y4, cumulative, dc);
        Ok(levels)
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

    fn set_coefficient_context(&mut self, x4: usize, y4: usize, level: u8, dc: u8) {
        self.above_level[x4] = level;
        self.above_dc[x4] = dc;
        self.left_level[y4] = level;
        self.left_dc[y4] = dc;
    }

    fn txb_skip_context(&self, x4: usize, y4: usize, block_width: usize) -> usize {
        if block_width == 4 {
            return 0;
        }
        let top = i32::from(self.above_level[x4]);
        let left = i32::from(self.left_level[y4]);
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

    fn dc_sign_context(&self, x4: usize, y4: usize) -> usize {
        let mut sum = 0i32;
        for &category in &[self.above_dc[x4], self.left_dc[y4]] {
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

fn coeff_base_eob_context(coefficient: usize) -> usize {
    if coefficient == 0 {
        0
    } else if coefficient <= 2 {
        1
    } else if coefficient <= 4 {
        2
    } else {
        3
    }
}

fn coeff_base_context(position: usize, levels: &[i32; 16]) -> usize {
    let (row, column) = (position >> 2, position & 3);
    let mut magnitude = 0i32;
    for &(delta_row, delta_column) in &cdf::SIG_REF_DIFF_OFFSET_2D {
        let (neighbor_row, neighbor_column) = (row + delta_row, column + delta_column);
        if neighbor_row < 4 && neighbor_column < 4 {
            magnitude += levels[(neighbor_row << 2) + neighbor_column].abs().min(3);
        }
    }
    let context = (((magnitude + 1) >> 1).min(4)) as usize;
    if row == 0 && column == 0 {
        0
    } else {
        context + usize::from(cdf::COEFF_BASE_CTX_OFFSET_4X4[row.min(4)][column.min(4)])
    }
}

fn coeff_br_context(position: usize, levels: &[i32; 16]) -> usize {
    let (row, column) = (position >> 2, position & 3);
    let mut magnitude = 0i32;
    for &(delta_row, delta_column) in &cdf::MAG_REF_OFFSET_2D {
        let (neighbor_row, neighbor_column) = (row + delta_row, column + delta_column);
        if neighbor_row < 4 && neighbor_column < 4 {
            magnitude += levels[(neighbor_row << 2) + neighbor_column].abs().min(15);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
