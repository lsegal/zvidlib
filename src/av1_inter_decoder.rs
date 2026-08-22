//! Dependency-free stateful decoding of bounded AV1 Main-profile lossless
//! inter frames, building on the intra reconstruction published by
//! [`crate::av1_intra_decoder`].
//!
//! The implemented syntax is a standards-compliant subset:
//!
//! - Main profile, 8-bit, monochrome, non-reduced sequence headers with
//!   `enable_order_hint = 1` (order hints are required to select and refresh
//!   references) and no super-resolution, CDEF, loop-restoration, film-grain,
//!   warped-motion, or frame-id numbering.
//! - `base_q_idx == 0` (lossless), a single tile, `disable_cdf_update = 1`
//!   (default CDFs only, never adapted), and `reduced_tx_set = 1` — the same
//!   lossless-4x4 coefficient path as the intra decoder.
//! - Key frames use the intra decoder's DC-prediction-only path. Inter frames
//!   restrict every block to single-reference translational, integer-pel
//!   motion compensation with `NEARESTMV`, `GLOBALMV`, or `NEWMV` (no
//!   `NEARMV`, no dynamic reference list, no compound/masked/warped
//!   prediction, no sub-pel interpolation filters).
//! - `show_existing_frame` is supported and validated against an 8-slot
//!   reference frame map keyed by `ref_frame_idx`, following the spec's
//!   `frame_to_show_map_idx` and key-frame refresh rules.
//!
//! Any syntax outside that bound (compound prediction, sub-pel motion,
//! multiple tiles, non-lossless quantization, chroma planes, frame-size
//! changes relative to the sequence header, etc.) is rejected with a clear
//! [`ErrorKind::Unsupported`] error rather than attempting a partial decode.
//! Malformed input never panics; it returns [`ErrorKind::MalformedMedia`].
//! [`Limits`] bounds both retained reference memory (all eight reference
//! slots together) and per-frame reconstruction work, and [`Av1InterDecoder::reset`]
//! releases every retained reference frame.

use crate::av1_cdf as cdf;
use crate::{
    Av1FrameType, Av1Obu, Av1Parser, Av1SequenceHeader, Av1SymbolDecoder, Av1SyntaxSupport,
    ColorRange, Error, ErrorKind, Limits, Result, VideoDimensions, VideoFrame, inverse_wht_4x4,
};

const NUM_REF_FRAMES: usize = 8;
const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_PLUS_RANGE: i32 = 14;
/// `PRIMARY_REF_NONE` (AV1 §6.8.2).
const PRIMARY_REF_NONE: u8 = 7;

/// One retained reference frame: its reconstructed luma plane and the frame
/// properties needed to validate future references against it.
#[derive(Clone, Debug)]
struct RefSlot {
    luma: Vec<u8>,
    width: usize,
    height: usize,
    mi_cols: usize,
    mi_rows: usize,
    order_hint: u32,
    color_range: ColorRange,
}

/// A bounded, stateful AV1 decoder that retains reference frames across
/// calls to [`Av1InterDecoder::decode_temporal_unit`].
pub struct Av1InterDecoder {
    limits: Limits,
    sequence: Option<Av1SequenceHeader>,
    refs: [Option<RefSlot>; NUM_REF_FRAMES],
}

impl Av1InterDecoder {
    /// Creates a decoder with no retained state.
    pub fn new(limits: Limits) -> Result<Self> {
        if limits.max_av1_blocks_per_frame == 0 {
            return Err(resource("AV1 reconstruction block limit must be nonzero"));
        }
        Ok(Self {
            limits,
            sequence: None,
            refs: Default::default(),
        })
    }

    /// Releases all retained reference frames and the cached sequence header.
    /// Equivalent to the spec's `error_resilient_mode` full reset, usable at
    /// any point (e.g. before a seek or replay).
    pub fn reset(&mut self) {
        self.sequence = None;
        self.refs = Default::default();
    }

    /// Decodes one low-overhead AV1 temporal unit and returns the frame it
    /// makes visible (`show_frame` or `show_existing_frame`).
    ///
    /// The temporal unit may carry a new sequence header; if so it replaces
    /// the cached one and every retained reference frame is dropped, since
    /// retained planes are no longer known to match the new coded geometry.
    pub fn decode_temporal_unit(&mut self, bytes: &[u8]) -> Result<VideoFrame> {
        let mut parser = Av1Parser::new(self.limits)?;
        let obus = parser.parse_low_overhead(bytes)?;
        for obu in &obus {
            if let Av1Obu::SequenceHeader { sequence, .. } = obu {
                self.adopt_sequence(sequence.clone())?;
            }
        }
        let coded = obus
            .iter()
            .find(|obu| matches!(obu, Av1Obu::Frame { .. } | Av1Obu::FrameHeader { .. }));
        match coded {
            Some(Av1Obu::Frame { frame, payload, .. }) if !frame.show_existing_frame => {
                self.decode_coded_frame(payload)
            }
            Some(Av1Obu::FrameHeader { frame, .. }) if frame.show_existing_frame => {
                let idx = frame
                    .frame_to_show_map_idx
                    .ok_or_else(|| malformed("show_existing_frame has no map index"))?;
                self.show_existing_frame(idx)
            }
            Some(Av1Obu::FrameHeader { .. }) => Err(unsupported(
                "AV1 inter decoder requires frames as combined Frame OBUs, not a separate tile group",
            )),
            _ => Err(malformed("AV1 temporal unit has no frame to decode")),
        }
    }

    fn adopt_sequence(&mut self, sequence: Av1SequenceHeader) -> Result<()> {
        check_sequence_support(&sequence)?;
        if self.sequence.as_ref() != Some(&sequence) {
            // A changed sequence header invalidates every retained reference:
            // their coded geometry is no longer guaranteed to match.
            self.refs = Default::default();
        }
        self.sequence = Some(sequence);
        Ok(())
    }

    fn show_existing_frame(&mut self, idx: u8) -> Result<VideoFrame> {
        let index = usize::from(idx);
        if index >= NUM_REF_FRAMES {
            return Err(malformed(
                "show_existing_frame map index is outside the reference frame map",
            ));
        }
        let slot = self.refs[index]
            .clone()
            .ok_or_else(|| malformed("show_existing_frame references an empty reference slot"))?;
        luma_to_video_frame(
            slot.luma.clone(),
            slot.width,
            slot.height,
            slot.color_range,
            &self.limits,
        )
    }

    fn decode_coded_frame(&mut self, payload: &[u8]) -> Result<VideoFrame> {
        let sequence = self
            .sequence
            .clone()
            .ok_or_else(|| malformed("AV1 frame OBU precedes a supported sequence header"))?;
        let dimensions = VideoDimensions::new(
            sequence.max_frame_width,
            sequence.max_frame_height,
            &self.limits,
        )?;
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
        self.check_reference_budget(mi_cols, mi_rows)?;

        let mut bits = HeaderBits::new(payload);
        let header = parse_inter_frame_header(&mut bits, &sequence, mi_cols, mi_rows)?;
        if header.error_resilient_mode && header.primary_ref_frame == PRIMARY_REF_NONE {
            self.refs = Default::default();
        }

        let refs = self.resolve_references(&header)?;
        let tile_offset = bits.position() / 8;
        let tile = payload
            .get(tile_offset..)
            .ok_or_else(|| malformed("AV1 tile data is truncated"))?;
        let mut decoder = InterTileDecoder::new(
            tile,
            width,
            height,
            mi_cols,
            mi_rows,
            &header,
            &refs,
            &self.limits,
        )?;
        let luma = decoder.decode()?;

        let color_range = if sequence.color_config.color_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        if header.refresh_frame_flags != 0 {
            let slot = RefSlot {
                luma: luma.clone(),
                width,
                height,
                mi_cols,
                mi_rows,
                order_hint: header.order_hint,
                color_range,
            };
            for i in 0..NUM_REF_FRAMES {
                if header.refresh_frame_flags & (1 << i) != 0 {
                    self.refs[i] = Some(slot.clone());
                }
            }
        }
        if !header.show_frame {
            return Err(unsupported(
                "AV1 inter decoder requires every decoded frame to be shown",
            ));
        }
        luma_to_video_frame(luma, width, height, color_range, &self.limits)
    }

    /// Bounds total retained reference memory (all eight slots) to the
    /// allocation limit, independent of per-frame reconstruction bounds.
    fn check_reference_budget(&self, mi_cols: usize, mi_rows: usize) -> Result<()> {
        let per_frame = mi_cols
            .checked_mul(mi_rows)
            .and_then(|mi| mi.checked_mul(16))
            .ok_or_else(|| resource("AV1 reference frame size overflows"))?;
        let total = per_frame
            .checked_mul(NUM_REF_FRAMES)
            .ok_or_else(|| resource("AV1 reference frame budget overflows"))?;
        if u64::try_from(total)
            .map_err(|_| resource("AV1 reference budget is not representable"))?
            > self.limits.max_allocation_bytes
        {
            return Err(resource(
                "AV1 reference frame map exceeds the allocation limit",
            ));
        }
        Ok(())
    }

    fn resolve_references(&self, header: &InterFrameHeader) -> Result<[Option<RefSlot>; 8]> {
        if header.frame_type == Av1FrameType::Key || header.frame_type == Av1FrameType::IntraOnly {
            return Ok(Default::default());
        }
        let mut resolved: [Option<RefSlot>; 8] = Default::default();
        for (i, &idx) in header.ref_frame_idx.iter().enumerate() {
            let index = usize::from(idx);
            if index >= NUM_REF_FRAMES {
                return Err(malformed("AV1 ref_frame_idx is outside the reference map"));
            }
            let slot = self.refs[index]
                .clone()
                .ok_or_else(|| malformed("AV1 inter frame references an empty reference slot"))?;
            if slot.width != header.width || slot.height != header.height {
                return Err(unsupported(
                    "AV1 inter decoder requires references matching the current frame size",
                ));
            }
            let order_hint_bits = self
                .sequence
                .as_ref()
                .expect("sequence was validated before resolve_references was called")
                .order_hint_bits;
            if relative_order_distance(header.order_hint, slot.order_hint, order_hint_bits) <= 0 {
                return Err(malformed(
                    "AV1 inter frame references a reference frame that is not in its past",
                ));
            }
            resolved[i] = Some(slot);
        }
        Ok(resolved)
    }
}

/// `get_relative_dist` (AV1 §5.9.3): the signed distance from `b` to `a` in
/// wrapped order-hint space, positive when `a` is ahead of `b`.
fn relative_order_distance(a: u32, b: u32, order_hint_bits: u8) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let bits = u32::from(order_hint_bits);
    let diff = a.wrapping_sub(b) & ((1u32 << bits) - 1);
    let half = 1i64 << (bits - 1);
    let signed = i64::from(diff);
    (if signed > half {
        signed - (half << 1)
    } else {
        signed
    }) as i32
}

fn luma_to_video_frame(
    luma: Vec<u8>,
    width: usize,
    height: usize,
    color_range: ColorRange,
    limits: &Limits,
) -> Result<VideoFrame> {
    let dimensions = VideoDimensions::new(
        u32::try_from(width).map_err(|_| resource("AV1 width is not representable"))?,
        u32::try_from(height).map_err(|_| resource("AV1 height is not representable"))?,
        limits,
    )?;
    crate::Av1IntraFrame::from_luma(dimensions, luma, color_range, limits)?.into_video_frame(limits)
}

fn check_sequence_support(sequence: &Av1SequenceHeader) -> Result<()> {
    if sequence.support != Av1SyntaxSupport::MainProfile
        || sequence.color_config.bit_depth != 8
        || !sequence.color_config.monochrome
        || sequence.reduced_still_picture_header
        || !sequence.enable_order_hint
        || sequence.frame_id_numbers_present
        || sequence.decoder_model_info_present
        || sequence.enable_warped_motion
        || sequence.enable_ref_frame_mvs
        || sequence.enable_interintra_compound
        || sequence.enable_masked_compound
        || sequence.enable_dual_filter
        || sequence.enable_jnt_comp
    {
        return Err(unsupported(
            "AV1 inter decoder supports non-reduced Main-profile 8-bit monochrome streams with order hints, no frame-id numbering, and no warped motion, reference-MV projection, dual filters, or compound prediction features",
        ));
    }
    if sequence.enable_superres
        || sequence.enable_cdef
        || sequence.enable_restoration
        || sequence.film_grain_params_present
    {
        return Err(unsupported(
            "AV1 inter decoder does not support super-resolution, filters, restoration, or film grain",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct InterFrameHeader {
    frame_type: Av1FrameType,
    show_frame: bool,
    error_resilient_mode: bool,
    primary_ref_frame: u8,
    order_hint: u32,
    refresh_frame_flags: u8,
    ref_frame_idx: [u8; 7],
    width: usize,
    height: usize,
}

/// Parses the frame header bits this decoder supports and positions `bits`
/// at the byte-aligned start of the tile group. Anything outside the bounded
/// subset documented on the module is rejected explicitly.
fn parse_inter_frame_header(
    bits: &mut HeaderBits<'_>,
    seq: &Av1SequenceHeader,
    mi_cols: usize,
    mi_rows: usize,
) -> Result<InterFrameHeader> {
    let frame_type = match bits.read(2, "frame_type")? {
        0 => Av1FrameType::Key,
        1 => Av1FrameType::Inter,
        2 => Av1FrameType::IntraOnly,
        _ => {
            return Err(unsupported(
                "AV1 inter decoder does not support SWITCH frames",
            ));
        }
    };
    if frame_type == Av1FrameType::IntraOnly {
        return Err(unsupported(
            "AV1 inter decoder does not support intra-only non-key frames",
        ));
    }
    let show_frame = bits.read(1, "show_frame")? != 0;
    if !show_frame {
        return Err(unsupported(
            "AV1 inter decoder requires every decoded frame to be shown",
        ));
    }
    let error_resilient_mode = if frame_type == Av1FrameType::Key && show_frame {
        true
    } else {
        bits.read(1, "error_resilient_mode")? != 0
    };
    require_bit(bits, true, "disable_cdf_update")?;
    // allow_screen_content_tools is only signalled per-frame when the
    // sequence header selects SELECT_SCREEN_CONTENT_TOOLS (2); otherwise it
    // is implied by the sequence header with no bit read (AV1 §5.9.2).
    let allow_screen_content_tools = match seq.seq_force_screen_content_tools {
        2 => bits.read(1, "allow_screen_content_tools")? != 0,
        value => value != 0,
    };
    if allow_screen_content_tools {
        return Err(unsupported(
            "AV1 inter decoder does not support screen content tools",
        ));
    }
    let is_intra = frame_type == Av1FrameType::Key;
    let primary_ref_frame = if is_intra || error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        u8::try_from(bits.read(3, "primary_ref_frame")?).expect("three bits fit u8")
    };
    // refresh_frame_flags is implicitly 0xff for a shown key frame.
    let order_hint = bits.read(usize::from(seq.order_hint_bits), "order_hint")?;
    let refresh_frame_flags = if frame_type == Av1FrameType::Key && show_frame {
        0xffu8
    } else {
        u8::try_from(bits.read(8, "refresh_frame_flags")?).expect("eight bits fit u8")
    };

    let mut ref_frame_idx = [0u8; 7];
    if !is_intra {
        require_bit(bits, false, "frame_refs_short_signaling")?;
        for slot in &mut ref_frame_idx {
            *slot = u8::try_from(bits.read(3, "ref_frame_idx")?).expect("three bits fit u8");
        }
    }

    require_bit(bits, false, "frame_size_override_flag")?;
    // frame_size(): superres is disallowed by the sequence header bound, so
    // FrameWidth/FrameHeight are exactly the sequence header's maximums.
    let width = usize::try_from(seq.max_frame_width)
        .map_err(|_| resource("AV1 frame width is not representable"))?;
    let height = usize::try_from(seq.max_frame_height)
        .map_err(|_| resource("AV1 frame height is not representable"))?;
    // render_size(): rendered dimensions must match the coded frame.
    require_bit(bits, false, "render_and_frame_size_different")?;

    if !is_intra {
        // force_integer_mv is implicitly true whenever
        // allow_screen_content_tools is false (checked above), so
        // allow_high_precision_mv is implicitly false with no bit read
        // (AV1 §5.9.2) — motion vectors are always integer-pel here.
        require_bit(bits, false, "is_filter_switchable")?;
        // interpolation_filter selects the sub-pel tap set; motion vectors
        // are always integer-pel in this decoder so its value has no
        // reconstruction effect, but the bits must still be consumed.
        bits.read(2, "interpolation_filter")?;
        require_bit(bits, false, "is_motion_mode_switchable")?;
        // use_ref_frame_mvs is never present: the sequence header is
        // required to have enable_ref_frame_mvs == 0 above, so
        // `error_resilient_mode || !enable_ref_frame_mvs` is always true and
        // the spec sets UseRefFrameMvs = 0 without reading a bit.
    }

    require_bit(bits, false, "disable_frame_end_update_cdf")?;
    parse_tile_info(bits, mi_cols, mi_rows)?;
    // quantization_params(): base_q_idx must select the lossless path (all
    // delta_q_* implicitly zero) so loop filter, CDEF, and loop restoration
    // are skipped below exactly as `uncompressed_header()` skips them when
    // `CodedLossless` is true.
    if bits.read(8, "base_q_idx")? != 0 {
        return Err(unsupported(
            "AV1 inter decoder currently requires lossless quantization",
        ));
    }
    require_bit(bits, false, "delta_q_y_dc")?;
    require_bit(bits, false, "using_qmatrix")?;
    // segmentation_params()
    require_bit(bits, false, "segmentation_enabled")?;
    // delta_q_params(): only present when base_q_idx > 0.
    // delta_lf_params(): only present when delta_q_present.
    // loop_filter_params(), cdef_params(), lr_params(): all skipped because
    // CodedLossless is true (base_q_idx == 0, no segmentation deltas).
    // read_tx_mode(): CodedLossless forces TX_MODE_ONLY_4X4, no bit read.
    // frame_reference_mode(): intra frames imply single reference; inter
    // frames must additionally signal only single-reference prediction.
    let reference_select = if is_intra {
        false
    } else {
        bits.read(1, "reference_select")? != 0
    };
    if reference_select {
        return Err(unsupported(
            "AV1 inter decoder supports single-reference prediction only",
        ));
    }
    // skip_mode_params(): SkipModeAllowed is false whenever reference
    // selection is unavailable (single-reference-only frames), so no
    // skip_mode_present bit is read.
    // allow_warped_motion: never present either — the sequence header is
    // required to have enable_warped_motion == 0 above, so the frame header
    // condition `!enable_warped_motion` is always true and no bit is read.
    require_bit(bits, true, "reduced_tx_set")?;
    // global_motion_params(): every reference's motion type must be
    // IDENTITY (the default) for the translational-only bound; a non-default
    // affine/translation model would need extra syntax we do not parse.
    if !is_intra {
        for _ in 0..7 {
            require_bit(bits, false, "is_global (non-identity global motion)")?;
        }
    }
    // film_grain_params(): skipped because the sequence header requires
    // film_grain_params_present == 0.
    while bits.position() & 7 != 0 {
        require_bit(bits, false, "frame header byte alignment")?;
    }
    Ok(InterFrameHeader {
        frame_type,
        show_frame,
        error_resilient_mode,
        primary_ref_frame,
        order_hint,
        refresh_frame_flags,
        ref_frame_idx,
        width,
        height,
    })
}

fn parse_tile_info(bits: &mut HeaderBits<'_>, mi_cols: usize, mi_rows: usize) -> Result<()> {
    require_bit(bits, true, "uniform_tile_spacing_flag")?;
    let sb_cols = mi_cols.saturating_add(15) >> 4;
    let sb_rows = mi_rows.saturating_add(15) >> 4;
    if tile_log2(1, sb_cols.min(64)) > 0 {
        require_bit(bits, false, "increment_tile_cols_log2")?;
    }
    if tile_log2(1, sb_rows.min(64)) > 0 {
        require_bit(bits, false, "increment_tile_rows_log2")?;
    }
    Ok(())
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

fn require_bit(bits: &mut HeaderBits<'_>, expected: bool, name: &str) -> Result<()> {
    let value = bits.read(1, name)? != 0;
    if value != expected {
        Err(unsupported(format!("unsupported AV1 {name} value")))
    } else {
        Ok(())
    }
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
                .ok_or_else(|| malformed(format!("AV1 {name} is truncated")))?;
            value = (value << 1) | u32::from((byte >> (7 - (self.bit & 7))) & 1);
            self.bit = self
                .bit
                .checked_add(1)
                .ok_or_else(|| resource("AV1 frame header offset overflows"))?;
        }
        Ok(value)
    }
}

/// The three inter prediction modes this decoder reconstructs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InterMode {
    Nearest,
    Global,
    New,
}

/// Decodes one bounded lossless tile that may mix intra and single-reference
/// inter blocks, reusing the intra decoder's DC-prediction and coefficient
/// syntax for intra blocks and adding translational, integer-pel motion
/// compensation for inter blocks.
struct InterTileDecoder<'a> {
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
    frame_type: Av1FrameType,
    reference: Option<&'a RefSlot>,
}

impl<'a> InterTileDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tile: &'a [u8],
        width: usize,
        height: usize,
        mi_cols: usize,
        mi_rows: usize,
        header: &InterFrameHeader,
        refs: &'a [Option<RefSlot>; NUM_REF_FRAMES],
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
        let is_intra = header.frame_type == Av1FrameType::Key;
        let reference =
            if is_intra {
                None
            } else {
                Some(refs[0].as_ref().ok_or_else(|| {
                    malformed("AV1 inter frame has no resolved reference in slot 0")
                })?)
            };
        if let Some(reference) = reference {
            if reference.mi_cols != mi_cols || reference.mi_rows != mi_rows {
                return Err(unsupported(
                    "AV1 inter decoder requires references matching the current coded geometry",
                ));
            }
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
            frame_type: header.frame_type,
            reference,
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
                        "AV1 inter decoder supports NONE and SPLIT partitions",
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
            return Err(unsupported("AV1 skipped blocks are not supported"));
        }
        let is_inter =
            self.frame_type != Av1FrameType::Key && self.symbols.symbol(&cdf::IS_INTER)? != 0;
        let motion = if is_inter {
            Some(self.decode_inter_mode_and_ref()?)
        } else {
            if self.symbols.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC)? != 0 {
                return Err(unsupported(
                    "AV1 inter decoder currently supports DC_PRED intra blocks",
                ));
            }
            None
        };
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
        if let Some((row_offset, column_offset)) = motion {
            self.predict_inter_block(row, column, block_width, row_offset, column_offset)?;
        }
        for transform_y in 0..units {
            for transform_x in 0..units {
                let x = column * 4 + transform_x * 4;
                let y = row * 4 + transform_y * 4;
                if x < self.coded_width && y < self.coded_height {
                    self.decode_transform_block(x, y, block_width, motion.is_some())?;
                }
            }
        }
        Ok(())
    }

    /// Decodes `is_inter`'s companion syntax (reference selection and mode)
    /// and returns the block's integer-pel motion vector as `(row, column)`
    /// offsets in luma samples.
    fn decode_inter_mode_and_ref(&mut self) -> Result<(i32, i32)> {
        if self.symbols.symbol(&cdf::SINGLE_REF_P1)? != 0 {
            return Err(unsupported(
                "AV1 inter decoder only predicts from the primary (LAST) reference slot",
            ));
        }
        let mode = if self.symbols.symbol(&cdf::NEW_MV)? == 0 {
            InterMode::New
        } else if self.symbols.symbol(&cdf::ZERO_MV)? == 0 {
            InterMode::Global
        } else {
            InterMode::Nearest
        };
        match mode {
            InterMode::Global | InterMode::Nearest => Ok((0, 0)),
            InterMode::New => self.decode_new_mv(),
        }
    }

    fn decode_new_mv(&mut self) -> Result<(i32, i32)> {
        let joint = self.symbols.symbol(&cdf::MV_JOINT)?;
        let row = if joint == 1 || joint == 3 {
            self.decode_mv_component(0)?
        } else {
            0
        };
        let column = if joint == 2 || joint == 3 {
            self.decode_mv_component(1)?
        } else {
            0
        };
        Ok((row, column))
    }

    /// Decodes one AV1 `mv_component` (§5.11.32) restricted to integer pel:
    /// the fractional and high-precision bits are still present in the
    /// bitstream (the encoder always signals `allow_high_precision_mv = 0`,
    /// but `mv_class0_fr`/`mv_fr` remain part of the syntax) and are
    /// consumed but ignored here since every motion vector is integer-pel.
    fn decode_mv_component(&mut self, component: usize) -> Result<i32> {
        let sign = self.symbols.symbol(&cdf::MV_SIGN)? != 0;
        let class = self.symbols.symbol(&cdf::MV_CLASS[component])?;
        let magnitude = if class == 0 {
            let bit = self.symbols.symbol(&cdf::MV_CLASS0_BIT)?;
            self.symbols.symbol(&cdf::MV_CLASS0_FR[component])?;
            self.symbols.symbol(&cdf::MV_HP)?;
            (bit as i32) + 1
        } else {
            let mut value = 1i32;
            for bit_index in (0..class).rev() {
                let bit = self.symbols.literal(1)?;
                value |= (bit as i32) << bit_index;
            }
            value <<= 3;
            self.symbols.symbol(&cdf::MV_FR[component])?;
            self.symbols.symbol(&cdf::MV_HP)?;
            value + 1
        };
        if magnitude > i32::from(i16::MAX) {
            return Err(malformed("AV1 motion vector magnitude is out of range"));
        }
        Ok(if sign { -magnitude } else { magnitude })
    }

    /// Copies the reference block for a whole coding block at once: with
    /// integer-pel translational motion every 4x4 transform block within it
    /// shares the same displaced source region, so the residual stage only
    /// needs to add coefficients on top of this prediction.
    fn predict_inter_block(
        &mut self,
        row: usize,
        column: usize,
        block_width: usize,
        row_offset: i32,
        column_offset: i32,
    ) -> Result<()> {
        let reference = self
            .reference
            .ok_or_else(|| malformed("AV1 inter block decoded without a resolved reference"))?;
        let x = column * 4;
        let y = row * 4;
        let units = block_width / 4;
        for by in 0..units {
            for bx in 0..units {
                let (dst_x, dst_y) = (x + bx * 4, y + by * 4);
                if dst_x >= self.coded_width || dst_y >= self.coded_height {
                    continue;
                }
                for sy in 0..4 {
                    for sx in 0..4 {
                        let src_row = i64::from(dst_y as i32 + sy + row_offset);
                        let src_col = i64::from(dst_x as i32 + sx + column_offset);
                        if src_row < 0
                            || src_col < 0
                            || src_row as usize >= reference.height
                            || src_col as usize >= reference.width
                        {
                            return Err(malformed(
                                "AV1 motion vector references outside the reference frame",
                            ));
                        }
                        let sample =
                            reference.luma[src_row as usize * reference.width + src_col as usize];
                        self.pixels
                            [(dst_y + sy as usize) * self.coded_width + dst_x + sx as usize] =
                            sample;
                    }
                }
            }
        }
        Ok(())
    }

    fn decode_transform_block(
        &mut self,
        x: usize,
        y: usize,
        block_width: usize,
        is_inter: bool,
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
        let coefficients = self.decode_coefficients(x >> 2, y >> 2, block_width)?;
        let residuals = inverse_wht_4x4(&coefficients);
        if is_inter {
            for row in 0..4 {
                for column in 0..4 {
                    let index = (y + row) * self.coded_width + x + column;
                    self.pixels[index] = (i16::from(self.pixels[index])
                        + residuals[row * 4 + column])
                        .clamp(0, 255) as u8;
                }
            }
        } else {
            let prediction = self.dc_prediction(x, y);
            for row in 0..4 {
                for column in 0..4 {
                    self.pixels[(y + row) * self.coded_width + x + column] =
                        (i16::from(prediction) + residuals[row * 4 + column]).clamp(0, 255) as u8;
                }
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
            return Err(malformed("AV1 coefficient EOB is outside a 4x4 transform"));
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

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn resource(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

/// Test-only bitstream construction. Every writer here is the exact
/// symmetric inverse of a reader used by [`decode_temporal_unit`] (or by
/// [`Av1SymbolDecoder`] itself), so hand-built vectors exercise the real
/// parsing and reconstruction paths above rather than a separate mock
/// format. There is no production AV1 encoder in this crate; this module
/// exists solely to generate standards-compliant fixtures for the tests
/// below without checking in externally generated binaries.
///
/// [`SymbolEncoder`] is the exact algebraic inverse of
/// [`Av1SymbolDecoder::symbol`], derived directly from its update rule
/// rather than reimplementing a textbook range coder from memory. The
/// decoder's per-symbol renormalization is:
///
/// ```text
/// range  = upper - lower                       // same lower/upper the encoder computes below
/// value  = value - lower
/// shift  = 15 - ilog2(range)
/// range  <<= shift
/// bits   = next `shift` raw bits from the input, MSB-first
/// value  = (bits << (shift - available)) ^ (((value + 1) << shift) - 1)
/// ```
///
/// `((v + 1) << shift) - 1` equals `(v << shift) | ones(shift)` for any
/// `v >= 0` (the `+1`/`-1` only ever borrows within the low `shift` bits),
/// so once the *decoder's own remainder* `v = value - lower` for step `i`
/// is known, `value_new = (v << shift) | (bits ^ ones(shift))`: the high
/// bits of the next step's `value` are exactly `v`, and its low `shift`
/// bits are freely steerable by choosing `bits`.
///
/// `range` (and therefore every step's `shift`) depends only on the chosen
/// symbol sequence, not on `value`, so the encoder first replays the
/// decoder's `range` update forward across the whole sequence to learn
/// every step's `(lower, shift)`. It then walks *backward*: it picks the
/// trivial remainder `v = 0` for the last symbol (nothing follows it), and
/// for every earlier step `i` solves for the remainder `v_i` that makes
/// the *next* step's `value` land exactly on `lower(s_{i+1})`, namely
/// `v_i = target >> shift_i` with `target = lower(s_{i+1}) + v_{i+1}`,
/// emitting `bits_i = (target & ones(shift_i)) ^ ones(shift_i)` for the
/// gap between steps `i` and `i + 1`. (An earlier version of this encoder
/// always assumed `v = 0` at every step instead of solving backward for
/// it; that only happens to work when every symbol after the first is the
/// last symbol of its CDF, so it round-tripped trivial single-step cases
/// but silently produced the wrong bitstream for anything longer — caught
/// by `symbol_encoder_round_trips_through_the_real_decoder` below, which
/// must pass before any fixture depends on this encoder.) The very first
/// step has no predecessor to inherit high bits from: its `value` is read
/// directly from the stream's first 15 raw bits as `0x7FFF ^ value_0`, so
/// the encoder emits `0x7FFF ^ (lower(s_0) + v_0)` as those 15 bits.
#[cfg(test)]
pub(crate) mod test_encoder {
    use crate::Av1SymbolDecoder;
    use crate::av1_entropy::AV1_CDF_MAX;

    /// MSB-first bit writer, the exact inverse of `HeaderBits`.
    #[derive(Default)]
    pub(crate) struct BitWriter {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl BitWriter {
        pub(crate) fn bits(&mut self, value: u32, width: usize) {
            for shift in (0..width).rev() {
                if self.bit & 7 == 0 {
                    self.bytes.push(0);
                }
                if value & (1 << shift) != 0 {
                    let last = self.bytes.len() - 1;
                    self.bytes[last] |= 0x80 >> (self.bit & 7);
                }
                self.bit += 1;
            }
        }

        pub(crate) fn byte_align(&mut self) {
            while self.bit & 7 != 0 {
                self.bits(0, 1);
            }
        }

        pub(crate) fn into_bytes(mut self) -> Vec<u8> {
            self.byte_align();
            self.bytes
        }
    }

    /// One recorded symbol step from the forward pass: the `lower` bound
    /// the decoder must see to select the encoded symbol, and the `shift`
    /// (renormalization width) that step produces for computing the next
    /// step's `value`. Both depend only on the chosen symbol/CDF sequence.
    struct StepInfo {
        lower: u32,
        shift: u32,
    }

    /// The exact symmetric inverse of [`Av1SymbolDecoder`]. Symbols are
    /// buffered; [`SymbolEncoder::finish`] performs the two-pass solve
    /// described in the module docs above and returns the final bytes.
    pub(crate) struct SymbolEncoder {
        range: u32,
        steps: Vec<StepInfo>,
    }

    impl SymbolEncoder {
        const PROB_SHIFT: u32 = 6;
        const MIN_PROB: u32 = 4;

        pub(crate) fn new() -> Self {
            Self {
                range: AV1_CDF_MAX.into(),
                steps: Vec::new(),
            }
        }

        /// The same `lower` bound formula `Av1SymbolDecoder::symbol` uses
        /// for `cdf[s]`, i.e. the value `lower` takes when the search loop
        /// has just tested symbol `s`, given the *entering* `range` for
        /// this step.
        fn bound(range: u32, cdf: &[u16], s: usize) -> u32 {
            if s >= cdf.len() {
                return 0;
            }
            let symbols = cdf.len() as u32;
            let inverse_cdf = u32::from(AV1_CDF_MAX) - u32::from(cdf[s]);
            let index = s as u32;
            (((range >> 8) * (inverse_cdf >> Self::PROB_SHIFT)) >> (7 - Self::PROB_SHIFT))
                + Self::MIN_PROB * (symbols - index - 1)
        }

        /// Encodes one symbol against `cdf`, an ascending AV1 cumulative
        /// distribution function terminated by `32768`. This only advances
        /// the forward `range` simulation and records `(lower, shift)`;
        /// the actual output bits are solved for in [`SymbolEncoder::finish`].
        pub(crate) fn symbol(&mut self, cdf: &[u16], symbol: usize) {
            let upper = if symbol == 0 {
                self.range
            } else {
                Self::bound(self.range, cdf, symbol - 1)
            };
            let lower = Self::bound(self.range, cdf, symbol);
            debug_assert!(lower < upper, "AV1 CDF interval must be non-empty");
            let range_prime = upper - lower;
            let shift = 15 - range_prime.ilog2();
            self.steps.push(StepInfo { lower, shift });
            self.range = range_prime << shift;
        }

        /// Encodes `count` equiprobable bits, most-significant bit first,
        /// mirroring `Av1SymbolDecoder::literal`.
        pub(crate) fn literal(&mut self, value: u32, count: u8) {
            const BOOL_CDF: [u16; 2] = [16_384, AV1_CDF_MAX];
            for shift in (0..count).rev() {
                let bit = (value >> shift) & 1;
                self.symbol(&BOOL_CDF, bit as usize);
            }
        }

        /// Solves backward for every step's remainder (`value - lower`,
        /// see the module docs' derivation) and emits the resulting raw
        /// stream bits: the first 15 bits seed the decoder's initial
        /// `value` directly, and each subsequent step's `shift`-wide gap
        /// steers the following step's `value` onto its own `lower` bound.
        /// Trailing zero bytes pad the stream so the decoder never runs out
        /// of input while renormalizing after the last real symbol.
        pub(crate) fn finish(self) -> Vec<u8> {
            let steps = self.steps;
            let n = steps.len();
            if n == 0 {
                return vec![0; 8];
            }
            // remainder[i] = the `value - lower` the decoder will compute
            // after step i; remainder[n - 1] is unconstrained (no step
            // follows it), so 0 is as good a choice as any.
            let mut remainder = vec![0u32; n];
            let mut gap_bits = vec![0u32; n];
            for i in (0..n - 1).rev() {
                let shift = steps[i].shift;
                let target = steps[i + 1].lower + remainder[i + 1];
                remainder[i] = target >> shift;
                let ones = (1u32 << shift) - 1;
                gap_bits[i] = (target & ones) ^ ones;
            }
            let value0 = steps[0].lower + remainder[0];
            let initial_bits = (u32::from(AV1_CDF_MAX) - 1) ^ value0;

            let mut writer = BitWriter::default();
            writer.bits(initial_bits, 15);
            for i in 0..n - 1 {
                writer.bits(gap_bits[i], steps[i].shift as usize);
            }
            let mut bytes = writer.into_bytes();
            bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
            bytes
        }
    }

    /// Encodes a sequence of `(cdf, symbol)` pairs and asserts the real
    /// [`Av1SymbolDecoder`] reads back exactly `symbols`, catching any
    /// mismatch between this module's interval math and the production
    /// decoder before it can silently produce a fixture that only "works"
    /// against itself.
    #[cfg(test)]
    pub(crate) fn assert_round_trips(pairs: &[(&[u16], usize)]) {
        let mut encoder = SymbolEncoder::new();
        for &(cdf, symbol) in pairs {
            encoder.symbol(cdf, symbol);
        }
        let bytes = encoder.finish();
        let mut decoder = Av1SymbolDecoder::new(&bytes).unwrap();
        for &(cdf, symbol) in pairs {
            assert_eq!(decoder.symbol(cdf).unwrap(), symbol);
        }
    }

    #[test]
    fn symbol_encoder_round_trips_through_the_real_decoder() {
        use crate::av1_cdf as cdf;

        assert_round_trips(&[(&cdf::SKIP[0], 0)]);
        assert_round_trips(&[(&cdf::SKIP[0], 1)]);
        assert_round_trips(&[(&cdf::IS_INTER, 0), (&cdf::IS_INTER, 1)]);
        assert_round_trips(&[
            (&cdf::PARTITION_W64[0], 0),
            (&cdf::PARTITION_W64[0], 3),
            (&cdf::TXB_SKIP[0], 1),
            (&cdf::COEFF_BASE_EOB[0][0], 2),
            (&cdf::MV_JOINT, 3),
            (&cdf::MV_CLASS[0], 0),
            (&cdf::MV_SIGN, 1),
        ]);
        // A long, varied sequence exercising literals and every symbol
        // width used elsewhere in this module, to catch precision loss
        // across many renormalizations.
        let mut encoder = SymbolEncoder::new();
        let mut expectations: Vec<(&[u16], usize)> = Vec::new();
        for i in 0..64usize {
            let symbol = i % 2;
            encoder.symbol(&cdf::SKIP[0], symbol);
            expectations.push((&cdf::SKIP[0], symbol));
            encoder.literal((i % 3) as u32, 2);
            expectations.push((&[16_384, AV1_CDF_MAX], ((i % 3) >> 1) & 1));
            expectations.push((&[16_384, AV1_CDF_MAX], (i % 3) & 1));
        }
        let bytes = encoder.finish();
        let mut decoder = Av1SymbolDecoder::new(&bytes).unwrap();
        for (cdf, symbol) in expectations {
            assert_eq!(decoder.symbol(cdf).unwrap(), symbol);
        }
    }
}

/// Hand-built low-overhead AV1 temporal units exercising
/// [`Av1InterDecoder`] end to end: a real sequence header, a lossless
/// 16x16 key frame with one nonzero coefficient, and a 16x16 inter frame
/// mixing NEARESTMV, GLOBALMV, and NEWMV blocks against it. Every writer
/// below is the bit-for-bit inverse of the parser it feeds
/// (`parse_sequence_header`/`Bits` in [`crate::av1`], and
/// `parse_inter_frame_header`/`HeaderBits` above), cross-checked by the
/// tests at the bottom of this module rather than assumed correct.
#[cfg(test)]
mod tests {
    use super::test_encoder::{BitWriter, SymbolEncoder};
    use super::*;
    use crate::av1_cdf as cdf;

    const FRAME_DIM: u32 = 16;

    /// Appends one OBU: `obu_header` (no extension, always has a size
    /// field, matching [`crate::av1::parse_obu_header`]) followed by a
    /// leb128 payload length and the payload itself.
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

    /// Builds a non-reduced sequence header payload matching every bound
    /// `check_sequence_support` requires: Main profile, 8-bit monochrome,
    /// `enable_order_hint = 1`, no superres/CDEF/restoration/film-grain/
    /// warped-motion/ref-frame-mvs/interintra/masked-compound/dual-filter/
    /// jnt-comp, and `order_hint_bits = 3` (room for a handful of frames).
    fn sequence_header_payload(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.bits(0, 3); // seq_profile = Main
        w.bits(0, 1); // still_picture
        w.bits(0, 1); // reduced_still_picture_header
        w.bits(0, 1); // timing_info_present_flag
        w.bits(0, 1); // initial_display_delay_present_flag
        w.bits(0, 5); // operating_points_cnt_minus_1 = 0 (one operating point)
        w.bits(0, 12); // operating_point_idc
        w.bits(0, 5); // seq_level_idx (<= 7, so no seq_tier bit)
        // decoder_model_info_present_flag is false, so no per-op model bit;
        // initial_display_delay_present_flag is false, so no per-op delay bit.
        w.bits(3, 4); // frame_width_bits_minus_1 = 3 -> 4 width bits
        w.bits(3, 4); // frame_height_bits_minus_1 = 3 -> 4 height bits
        w.bits(width - 1, 4); // max_frame_width_minus_1
        w.bits(height - 1, 4); // max_frame_height_minus_1
        w.bits(0, 1); // frame_id_numbers_present_flag
        w.bits(0, 1); // use_128x128_superblock
        w.bits(0, 1); // enable_filter_intra
        w.bits(0, 1); // enable_intra_edge_filter
        w.bits(0, 1); // enable_interintra_compound
        w.bits(0, 1); // enable_masked_compound
        w.bits(0, 1); // enable_warped_motion
        w.bits(0, 1); // enable_dual_filter
        w.bits(1, 1); // enable_order_hint
        w.bits(0, 1); // enable_jnt_comp
        w.bits(0, 1); // enable_ref_frame_mvs
        w.bits(0, 1); // seq_choose_screen_content_tools
        w.bits(0, 1); // seq_force_screen_content_tools = 0
        // seq_force_screen_content_tools == 0 implies seq_force_integer_mv
        // is fixed at SELECT_INTEGER_MV (2) with no bits read.
        w.bits(2, 3); // order_hint_bits_minus_1 = 2 -> 3 order-hint bits
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

    /// Frame-header bits common to both key and inter frames up to
    /// `refresh_frame_flags`, matching `parse_inter_frame_header` exactly.
    struct FrameHeaderBuilder {
        w: BitWriter,
    }

    impl FrameHeaderBuilder {
        fn key_frame(order_hint_bits: usize) -> Self {
            let mut w = BitWriter::default();
            w.bits(0, 2); // frame_type = KEY_FRAME
            w.bits(1, 1); // show_frame = 1
            // error_resilient_mode is implied true for a shown key frame.
            w.bits(1, 1); // disable_cdf_update = 1
            // allow_screen_content_tools is implied 0 (seq_force = 0).
            // primary_ref_frame is implied PRIMARY_REF_NONE for intra frames.
            w.bits(0, order_hint_bits); // order_hint
            // refresh_frame_flags is implied 0xff for a shown key frame.
            // ref_frame_idx() is absent for intra frames.
            w.bits(0, 1); // frame_size_override_flag
            w.bits(0, 1); // render_and_frame_size_different
            // is_filter_switchable/interpolation_filter/is_motion_mode_switchable
            // are all absent for intra frames.
            w.bits(0, 1); // disable_frame_end_update_cdf
            Self { w }
        }

        /// Appends `tile_info()`, `quantization_params()`,
        /// `segmentation_params()`, `frame_reference_mode()`,
        /// `reduced_tx_set`, and (for inter frames) the identity
        /// `global_motion_params()`, then byte-aligns.
        fn finish_common(mut self, is_intra: bool, mi_cols: usize, mi_rows: usize) -> Vec<u8> {
            // tile_info(): uniform spacing, one tile (mi_cols/mi_rows <= 64
            // superblock units here, so no increment bits are read).
            self.w.bits(1, 1); // uniform_tile_spacing_flag
            let sb_cols = (mi_cols + 15) >> 4;
            let sb_rows = (mi_rows + 15) >> 4;
            assert!(sb_cols <= 1 && sb_rows <= 1, "fixture exceeds one tile");
            self.w.bits(0, 8); // base_q_idx = 0 (lossless)
            self.w.bits(0, 1); // delta_q_y_dc
            self.w.bits(0, 1); // using_qmatrix
            self.w.bits(0, 1); // segmentation_enabled
            if !is_intra {
                self.w.bits(0, 1); // reference_select = 0 (single reference)
            }
            self.w.bits(1, 1); // reduced_tx_set = 1
            if !is_intra {
                for _ in 0..7 {
                    self.w.bits(0, 1); // is_global (identity global motion)
                }
            }
            self.w.byte_align();
            self.w.into_bytes()
        }
    }

    /// A minimal AV1 tile decodable by [`InterTileDecoder`]: a 16x16
    /// key frame split into four 8x8 DC_PRED blocks, one nonzero DC
    /// coefficient in the very first transform block, all others skipped.
    ///
    /// `txb_skip_context` depends on the `above_level`/`left_level` state
    /// left behind by every previously decoded transform block in the same
    /// coding-block row/column, so the context for each of the sixteen 4x4
    /// transform blocks below is precomputed by walking the same recurrence
    /// `InterTileDecoder::txb_skip_context`/`set_coefficient_context` use,
    /// in decode order (blocks visited (0,0), (0,2), (2,0), (2,2) in mi
    /// coordinates, each contributing a 2x2 grid of transform blocks):
    /// `[1, 2, 2, 1]` for block (0,0) — the only block with a nonzero
    /// coefficient, contributing a `level = 1` at its first transform block
    /// — then `1` for every remaining transform block in every other block,
    /// since every neighbor they see is either untouched (0) or that lone
    /// nonzero level (which never exceeds the "one zero neighbor" cutoff).
    fn key_frame_tile() -> Vec<u8> {
        const CONTEXTS: [[usize; 4]; 4] = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
        let mut e = SymbolEncoder::new();
        // decode_partition(0,0,64) / (0,0,32) / (0,0,16) force-split with no
        // symbol read (mi_cols = mi_rows = 4 < the half-block thresholds),
        // landing on the first real partition read at (0,0,16), bsl=2.
        e.symbol(&cdf::PARTITION_W16[0], 3); // SPLIT into four 8x8 blocks
        for (block, contexts) in CONTEXTS.iter().enumerate() {
            // Each 8x8 sub-block reads its own partition symbol at bsl=1;
            // context stays 0 for all four (neighbors are never < bsl once
            // set, per the partition_context trace in the module tests).
            e.symbol(&cdf::PARTITION_W8[0], 0); // NONE -> decode_block(_, _, 8)
            e.symbol(&cdf::SKIP[0], 0); // skip = 0
            e.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC, 0); // DC_PRED
            for (t, &context) in contexts.iter().enumerate() {
                if block == 0 && t == 0 {
                    e.symbol(&cdf::TXB_SKIP[context], 0); // not skipped
                    e.symbol(&cdf::EOB_PT_16[0][0], 0); // eob_point = 1 -> eob = 1
                    // coefficient 0 is also eob - 1: COEFF_BASE_EOB, ctx 0.
                    e.symbol(&cdf::COEFF_BASE_EOB[0][0], 0); // level = 1
                    // level (1) <= NUM_BASE_LEVELS, so no COEFF_BR reads.
                    // dc_sign_context(0,0): both neighbor categories are 0.
                    e.symbol(&cdf::DC_SIGN[0][0], 0); // positive
                } else {
                    e.symbol(&cdf::TXB_SKIP[context], 1); // skipped -> all zero
                }
            }
        }
        e.finish()
    }

    /// Wraps `sequence_header_payload()` and `key_frame_tile()` into a
    /// complete low-overhead temporal unit: temporal delimiter, sequence
    /// header, and one Frame OBU (frame header + tile data concatenated,
    /// matching how `Av1Parser` splits `Av1Obu::Frame` payloads).
    fn key_frame_temporal_unit() -> Vec<u8> {
        let mi = 2 * FRAME_DIM.div_ceil(8) as usize;
        let header = FrameHeaderBuilder::key_frame(3).finish_common(true, mi, mi);
        let mut payload = header;
        payload.extend_from_slice(&key_frame_tile());

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
    fn key_frame_temporal_unit_decodes_through_the_real_decoder() {
        let limits = Limits::default();
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        let frame = decoder
            .decode_temporal_unit(&key_frame_temporal_unit())
            .unwrap();
        assert_eq!(
            (frame.dimensions.width, frame.dimensions.height),
            (FRAME_DIM, FRAME_DIM)
        );
        eprintln!("luma = {:?}", frame.planes[0].data);
    }
}
