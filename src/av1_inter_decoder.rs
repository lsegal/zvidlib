//! Dependency-free stateful decoding of bounded AV1 Main-profile inter
//! frames, building on the intra reconstruction published by
//! [`crate::av1_intra_decoder`].
//!
//! The implemented syntax is a standards-compliant subset:
//!
//! - Main profile, 8-bit, monochrome, non-reduced sequence headers with
//!   `enable_order_hint = 1` (order hints are required to select and refresh
//!   references) and no super-resolution, CDEF, loop-restoration, film-grain,
//!   warped-motion, or frame-id numbering.
//! - A single tile, `disable_cdf_update = 1` (default CDFs only, never
//!   adapted), and `reduced_tx_set = 1`. `base_q_idx` selects between two
//!   quantization profiles (spec §5.9.12), mirroring the intra decoder:
//!   `base_q_idx == 0` (`CodedLossless`) reconstructs every 4x4 block with
//!   the normative inverse Walsh-Hadamard transform and never signals
//!   `loop_filter_params` (spec §5.9.11 forces every level to 0); otherwise
//!   each coding block signals its own `tx_size` through spec §5.11.16
//!   `read_tx_size` over the square transforms this crate's kernels
//!   implement (`TX_4X4` through `TX_64X64`), under either
//!   `TX_MODE_LARGEST` or `TX_MODE_SELECT` as the frame header's
//!   `tx_mode_select` bit selects, and its `tx_type` (`DCT_DCT` or `IDTX`,
//!   and `DCT_DCT` only from `TX_32X32` up, where the transform set is
//!   `TX_SET_DCTONLY`), coefficients are dequantized
//!   ([`crate::av1_intra::get_dc_quant`]/[`crate::av1_intra::get_ac_quant`])
//!   and inverse transformed ([`crate::av1_intra::inverse_transform`]),
//!   `loop_filter_params` is parsed, and the chosen per-block transform
//!   sizes are recorded into a [`crate::av1_filters::TxSizeGrid`] so
//!   [`crate::av1_filters::deblock_frame`] can select the correct filter
//!   length at each edge (spec §7.14.5) before the frame is shown or stored
//!   as a reference.
//! - Key frames use the intra decoder's DC-prediction-only path. Inter frames
//!   restrict blocks to translational, whole-pel motion compensation with
//!   `NEARESTMV`, `GLOBALMV`, or bounded `NEWMV`, plus average LAST/LAST2
//!   compound prediction with `NEAREST_NEARESTMV` or `GLOBAL_GLOBALMV` (no
//!   `NEARMV`, masked/warped prediction, or sub-pel interpolation).
//! - `show_existing_frame` is supported and validated against an 8-slot
//!   reference frame map keyed by `ref_frame_idx`, following the spec's
//!   `frame_to_show_map_idx` and key-frame refresh rules.
//!
//! Any syntax outside that bound (masked compound, sub-pel motion, multiple
//! tiles, chroma planes, frame-size changes relative to the sequence
//! header, etc.) is rejected with a clear [`ErrorKind::Unsupported`] error
//! rather than attempting a partial decode.
//! Malformed input never panics; it returns [`ErrorKind::MalformedMedia`].
//! [`Limits`] bounds both retained reference memory (all eight reference
//! slots together) and per-frame reconstruction work, and [`Av1InterDecoder::reset`]
//! releases every retained reference frame.

use crate::av1_cdf as cdf;
use crate::av1_filters::{FilterFrame, FilterPlane, LoopFilterParams, TxSizeGrid, deblock_frame};
use crate::av1_intra::{Av1IntraMode, Av1TxType, get_ac_quant, get_dc_quant, inverse_transform};
use crate::av1_intra_pred::{
    SmoothMode, add_residual_row, directional_row, paeth_row, smooth_row, sum_samples,
};
use crate::av1_mc::{InterpFilter, McContext, RefPlane, blend_average};
use crate::{
    Av1FrameType, Av1Obu, Av1Parser, Av1SequenceHeader, Av1SymbolDecoder, Av1SyntaxSupport,
    ColorRange, Error, ErrorKind, Limits, Result, VideoDimensions, VideoFrame, inverse_wht_4x4,
};

const NUM_REF_FRAMES: usize = 8;
const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_PLUS_RANGE: i32 = 14;
/// The widest square transform this crate's kernels implement
/// (`TX_64X64`), which is also AV1's largest transform.
const MAX_TX_WIDTH: usize = 64;
/// AV1 codes coefficients only inside a transform block's upper-left
/// 32x32 quadrant (spec §5.11.39 clamps the coded extent to 32 samples);
/// every position outside it reconstructs as zero.
const MAX_CODED_TX_WIDTH: usize = 32;
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
    frame_type: Av1FrameType,
    showable_frame: bool,
    generation: u64,
    key_shown_existing: bool,
}

/// A bounded, stateful AV1 decoder that retains reference frames across
/// calls to [`Av1InterDecoder::decode_temporal_unit`].
pub struct Av1InterDecoder {
    limits: Limits,
    sequence: Option<Av1SequenceHeader>,
    refs: [Option<RefSlot>; NUM_REF_FRAMES],
    next_generation: u64,
    last_tx_sizes: Option<TxSizeGrid>,
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
            next_generation: 0,
            last_tx_sizes: None,
        })
    }

    /// Releases all retained reference frames and the cached sequence header.
    /// Equivalent to the spec's `error_resilient_mode` full reset, usable at
    /// any point (e.g. before a seek or replay).
    pub fn reset(&mut self) {
        self.sequence = None;
        self.refs = Default::default();
        self.next_generation = 0;
        self.last_tx_sizes = None;
    }

    /// Returns the per-4x4-luma-unit transform-size grid recorded while
    /// reconstructing the most recently *coded* frame (the
    /// filter-length-selection metadata [`crate::deblock_frame`] takes as
    /// `luma_tx_sizes`), or `None` if no coded frame has been reconstructed
    /// yet or the most recent call instead showed a retained
    /// `show_existing_frame` (which reconstructs nothing new, so there is
    /// no metadata for the frame it shows).
    ///
    /// Every block in a `CodedLossless` frame's grid is the AV1 spec's
    /// mandatory `TX_4X4` transform size, because the spec forces
    /// `TX_MODE_ONLY_4X4` whenever `CodedLossless` is true (see
    /// `parse_inter_frame_header`'s `read_tx_mode` comment). Non-lossless
    /// frames record the real per-block `read_tx_size` choice instead.
    pub fn last_frame_tx_sizes(&self) -> Option<&TxSizeGrid> {
        self.last_tx_sizes.as_ref()
    }

    /// Decodes one low-overhead AV1 temporal unit and returns the frame it
    /// makes visible (`show_frame` or `show_existing_frame`).
    ///
    /// The temporal unit may carry a new sequence header; if so it replaces
    /// the cached one and every retained reference frame is dropped, since
    /// retained planes are no longer known to match the new coded geometry.
    pub fn decode_temporal_unit(&mut self, bytes: &[u8]) -> Result<VideoFrame> {
        let mut parser = Av1Parser::new(self.limits)?;
        if let Some(sequence) = self.sequence.clone() {
            parser.set_sequence(sequence);
        }
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
            self.next_generation = 0;
        }
        self.sequence = Some(sequence);
        Ok(())
    }

    fn show_existing_frame(&mut self, idx: u8) -> Result<VideoFrame> {
        // Showing a retained frame reconstructs nothing new, and retained
        // reference slots don't keep their own tx-size grid, so there is no
        // metadata for the frame this call shows.
        self.last_tx_sizes = None;
        let index = usize::from(idx);
        if index >= NUM_REF_FRAMES {
            return Err(malformed(
                "show_existing_frame map index is outside the reference frame map",
            ));
        }
        let slot = self.refs[index]
            .clone()
            .ok_or_else(|| malformed("show_existing_frame references an empty reference slot"))?;
        if !slot.showable_frame {
            return Err(malformed(
                "show_existing_frame references a frame that is not showable",
            ));
        }
        if slot.frame_type == Av1FrameType::Key {
            if slot.key_shown_existing {
                return Err(malformed(
                    "show_existing_frame may show a retained key frame only once",
                ));
            }
            for retained in self.refs.iter_mut().flatten() {
                if retained.generation == slot.generation {
                    retained.key_shown_existing = true;
                }
            }
        }
        luma_to_video_frame(
            slot.luma,
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
        if let Some(ref_order_hints) = header.ref_order_hints {
            for (index, order_hint) in ref_order_hints.into_iter().enumerate() {
                if self.refs[index]
                    .as_ref()
                    .is_some_and(|slot| slot.order_hint != order_hint)
                {
                    self.refs[index] = None;
                }
            }
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
        let mut luma = decoder.decode()?;
        let tx_sizes = decoder.tx_sizes;
        if header.base_q_idx != 0 {
            let mut filter_frame = FilterFrame::new_monochrome(FilterPlane::from_samples(
                width,
                height,
                luma.clone(),
                &self.limits,
            )?);
            deblock_frame(&mut filter_frame, &header.loop_filter, Some(&tx_sizes))?;
            luma = filter_frame.y.data;
        }
        self.last_tx_sizes = Some(tx_sizes);

        let color_range = if sequence.color_config.color_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        if header.refresh_frame_flags != 0 {
            let generation = self.next_generation;
            self.next_generation = self
                .next_generation
                .checked_add(1)
                .ok_or_else(|| resource("AV1 reference generation counter overflows"))?;
            let slot = RefSlot {
                luma: luma.clone(),
                width,
                height,
                mi_cols,
                mi_rows,
                order_hint: header.order_hint,
                color_range,
                frame_type: header.frame_type,
                showable_frame: header.showable_frame,
                generation,
                key_shown_existing: false,
            };
            for i in 0..NUM_REF_FRAMES {
                if header.refresh_frame_flags & (1 << i) != 0 {
                    self.refs[i] = Some(slot.clone());
                }
            }
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
            resolved[i] = Some(slot);
        }
        Ok(resolved)
    }
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
            "AV1 inter decoder supports non-reduced Main-profile 8-bit monochrome streams with order hints, no frame-id numbering, and no warped motion, reference-MV projection, dual filters, interintra, or masked compound prediction",
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
    showable_frame: bool,
    order_hint: u32,
    refresh_frame_flags: u8,
    ref_order_hints: Option<[u32; NUM_REF_FRAMES]>,
    ref_frame_idx: [u8; 7],
    reference_select: bool,
    width: usize,
    height: usize,
    base_q_idx: u8,
    loop_filter: LoopFilterParams,
    /// `tx_mode == TX_MODE_SELECT` (spec §5.9.20 `read_tx_mode`); `false`
    /// is `TX_MODE_LARGEST`, and lossless frames force `TX_MODE_ONLY_4X4`
    /// with no bit read.
    tx_mode_select: bool,
    interpolation_filter: InterpFilter,
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
    require_bit(bits, false, "show_existing_frame in a coded Frame OBU")?;
    // Intra-only frames never code an interpolation filter; the default is
    // never consulted because they never motion-compensate.
    let mut interpolation_filter = InterpFilter::Regular;
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
    let show_frame = bits.read(1, "show_frame")? != 0;
    let showable_frame = if show_frame {
        frame_type != Av1FrameType::Key
    } else {
        bits.read(1, "showable_frame")? != 0
    };
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
    let is_intra = matches!(frame_type, Av1FrameType::Key | Av1FrameType::IntraOnly);
    require_bit(bits, false, "frame_size_override_flag")?;
    let order_hint = bits.read(usize::from(seq.order_hint_bits), "order_hint")?;
    let _primary_ref_frame = if is_intra || error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        u8::try_from(bits.read(3, "primary_ref_frame")?).expect("three bits fit u8")
    };
    // refresh_frame_flags is implicitly 0xff for a shown key frame.
    let refresh_frame_flags = if frame_type == Av1FrameType::Key && show_frame {
        0xffu8
    } else {
        u8::try_from(bits.read(8, "refresh_frame_flags")?).expect("eight bits fit u8")
    };

    let ref_order_hints = if error_resilient_mode && !is_intra {
        let mut hints = [0u32; NUM_REF_FRAMES];
        for hint in &mut hints {
            *hint = bits.read(usize::from(seq.order_hint_bits), "ref_order_hint")?;
        }
        Some(hints)
    } else {
        None
    };

    let mut ref_frame_idx = [0u8; 7];
    if !is_intra {
        require_bit(bits, false, "frame_refs_short_signaling")?;
        for slot in &mut ref_frame_idx {
            *slot = u8::try_from(bits.read(3, "ref_frame_idx")?).expect("three bits fit u8");
        }
    }

    // frame_size(): superres is disallowed by the sequence header bound, so
    // FrameWidth/FrameHeight are exactly the sequence header's maximums.
    let width = usize::try_from(seq.max_frame_width)
        .map_err(|_| resource("AV1 frame width is not representable"))?;
    let height = usize::try_from(seq.max_frame_height)
        .map_err(|_| resource("AV1 frame height is not representable"))?;
    // render_size(): rendered dimensions must match the coded frame.
    require_bit(bits, false, "render_and_frame_size_different")?;

    if !is_intra {
        require_bit(bits, false, "allow_high_precision_mv")?;
        // force_integer_mv is implicitly true whenever
        // allow_screen_content_tools is false (checked above), so
        // allow_high_precision_mv is implicitly false with no bit read
        // (AV1 §5.9.2) — motion vectors are always integer-pel here.
        require_bit(bits, false, "is_filter_switchable")?;
        // interpolation_filter selects the sub-pel tap set. Motion vectors
        // are always integer-pel in this decoder, so every tap set reduces
        // to the same whole-sample copy, but the choice is still carried
        // through to the motion-compensation primitives rather than being
        // discarded.
        let code = bits.read(2, "interpolation_filter")?;
        interpolation_filter =
            InterpFilter::from_code(code).ok_or_else(|| malformed("AV1 interpolation_filter"))?;
        require_bit(bits, false, "is_motion_mode_switchable")?;
        // use_ref_frame_mvs is never present: the sequence header is
        // required to have enable_ref_frame_mvs == 0 above, so
        // `error_resilient_mode || !enable_ref_frame_mvs` is always true and
        // the spec sets UseRefFrameMvs = 0 without reading a bit.
    }

    // disable_frame_end_update_cdf is inferred to 1 because
    // disable_cdf_update is required to be 1; no bit is present here.
    parse_tile_info(bits, mi_cols, mi_rows)?;
    // quantization_params(): base_q_idx selects between the lossless (WHT)
    // and non-lossless (dequantized DCT/IDTX) reconstruction paths, mirroring
    // the intra decoder's `parse_supported_frame_header`.
    let base_q_idx = u8::try_from(bits.read(8, "base_q_idx")?).expect("8 bits fit u8");
    let lossless = base_q_idx == 0;
    require_bit(bits, false, "delta_q_y_dc")?;
    require_bit(bits, false, "using_qmatrix")?;
    // segmentation_params()
    require_bit(bits, false, "segmentation_enabled")?;
    // delta_q_params()/delta_lf_params(): only present when base_q_idx > 0,
    // and even then only when segmentation/delta-Q are enabled; this
    // decoder never signals either, so no bits are read.
    // loop_filter_params() (spec §5.9.11): `CodedLossless` forces every
    // filter level to 0 with no bits read. The non-lossless path parses a
    // minimal but real subset, mirroring the intra decoder: two luma
    // levels, and when they're not both zero, the two chroma levels, a
    // 3-bit sharpness, and (out of scope for this bounded decoder) no
    // per-reference delta syntax — `loop_filter_delta_enabled` is required
    // to be 0.
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
        require_bit(bits, false, "loop_filter_delta_enabled")?;
        LoopFilterParams {
            y_vertical_level,
            y_horizontal_level,
            u_level,
            v_level,
            sharpness,
        }
    };
    // cdef_params(), lr_params(): skipped because the sequence header
    // requires enable_cdef == 0 and enable_restoration == 0.
    // read_tx_mode() (spec §5.9.20): CodedLossless forces TX_MODE_ONLY_4X4
    // with no bit read at all; otherwise a single `tx_mode_select` bit
    // chooses between TX_MODE_LARGEST and TX_MODE_SELECT (see
    // `InterTileDecoder::read_tx_size`).
    let tx_mode_select = if base_q_idx == 0 {
        false
    } else {
        bits.read(1, "tx_mode_select")? != 0
    };
    // frame_reference_mode(): intra frames imply single reference; inter
    // frames may signal per-block single or compound prediction.
    let reference_select = if is_intra {
        false
    } else {
        bits.read(1, "reference_select")? != 0
    };
    if reference_select {
        require_bit(bits, false, "skip_mode_present")?;
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
        showable_frame,
        order_hint,
        refresh_frame_flags,
        ref_order_hints,
        ref_frame_idx,
        reference_select,
        width,
        height,
        base_q_idx,
        loop_filter,
        tx_mode_select,
        interpolation_filter,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MotionVector {
    row: i32,
    column: i32,
}

#[derive(Clone, Copy, Debug)]
struct InterPrediction {
    reference_indices: [usize; 2],
    motion_vectors: [MotionVector; 2],
    compound: bool,
}

#[derive(Clone, Copy, Debug)]
struct BlockMotion {
    reference_indices: [usize; 2],
    motion_vectors: [MotionVector; 2],
    compound: bool,
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
    references: [Option<&'a RefSlot>; 7],
    reference_select: bool,
    interpolation_filter: InterpFilter,
    /// Reusable motion-compensation state, bound to the best SIMD backend
    /// this host supports.
    mc: McContext,
    /// Per-reference compound prediction scratch, at the 16x compound scale
    /// the blend kernels consume.
    compound: [Vec<i16>; 2],
    motion_grid: Vec<Option<BlockMotion>>,
    base_q_idx: u8,
    tx_mode_select: bool,
    tx_sizes: TxSizeGrid,
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
        let references = std::array::from_fn(|index| refs[index].as_ref());
        for reference in references.iter().flatten() {
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
            references,
            reference_select: header.reference_select,
            interpolation_filter: header.interpolation_filter,
            mc: McContext::new(),
            compound: [Vec::new(), Vec::new()],
            motion_grid: vec![None; contexts],
            base_q_idx: header.base_q_idx,
            tx_mode_select: header.tx_mode_select,
            tx_sizes: TxSizeGrid::new(width, height),
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
        let prediction = if is_inter {
            Some(self.decode_inter_mode_and_ref(row, column, block_width)?)
        } else {
            None
        };
        let intra_mode = if prediction.is_none() {
            read_intra_y_mode(self.symbols.symbol(&cdf::INTRA_FRAME_Y_MODE_DC_DC)?)?
        } else {
            Av1IntraMode::Dc
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
        if let Some(prediction) = prediction {
            self.predict_inter_block(row, column, block_width, prediction)?;
            let block_motion = BlockMotion {
                reference_indices: prediction.reference_indices,
                motion_vectors: prediction.motion_vectors,
                compound: prediction.compound,
            };
            for y in 0..units {
                for x in 0..units {
                    let (mi_row, mi_column) = (row + y, column + x);
                    if mi_row < self.mi_rows && mi_column < self.mi_cols {
                        self.motion_grid[mi_row * self.mi_cols + mi_column] = Some(block_motion);
                    }
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
                    // clipped, mirroring the intra decoder.
                    if x + tx_width > self.coded_width || y + tx_width > self.coded_height {
                        return Err(unsupported(
                            "AV1 transform block extends past the coded frame",
                        ));
                    }
                    self.decode_transform_block(x, y, tx_width, prediction.is_some(), intra_mode)?;
                }
                transform_x += step;
            }
            transform_y += step;
        }
        Ok(())
    }

    /// Decodes `is_inter`'s companion syntax (reference selection and mode)
    /// and returns the block's integer-pel motion vector as `(row, column)`
    /// offsets in luma samples.
    fn decode_inter_mode_and_ref(
        &mut self,
        row: usize,
        column: usize,
        block_width: usize,
    ) -> Result<InterPrediction> {
        let compound =
            self.reference_select && block_width >= 8 && self.symbols.symbol(&cdf::COMP_MODE)? != 0;
        if compound {
            if self.symbols.symbol(&cdf::COMP_REF_TYPE)? != 0
                || self.symbols.symbol(&cdf::UNI_COMP_REF)? != 0
                || self.symbols.symbol(&cdf::UNI_COMP_REF_P1)? != 0
            {
                return Err(unsupported(
                    "AV1 inter decoder supports LAST/LAST2 average compound prediction",
                ));
            }
            let compound_mode = self.symbols.symbol(&cdf::COMPOUND_MODE)?;
            let motion_vectors = match compound_mode {
                0 => [
                    self.nearest_motion_vector(row, column, 0),
                    self.nearest_motion_vector(row, column, 1),
                ],
                6 => [MotionVector::default(); 2],
                _ => {
                    return Err(unsupported(
                        "AV1 compound prediction supports NEAREST_NEARESTMV and GLOBAL_GLOBALMV",
                    ));
                }
            };
            return Ok(InterPrediction {
                reference_indices: [0, 1],
                motion_vectors,
                compound: true,
            });
        }

        if self.symbols.symbol(&cdf::SINGLE_REF_P1)? != 0
            || self.symbols.symbol(&cdf::SINGLE_REF_P3)? != 0
            || self.symbols.symbol(&cdf::SINGLE_REF_P4)? != 0
        {
            return Err(unsupported(
                "AV1 single prediction currently supports the LAST reference",
            ));
        }
        let predicted = self.nearest_motion_vector(row, column, 0);
        let mode = if self.symbols.symbol(&cdf::NEW_MV)? == 0 {
            InterMode::New
        } else if self.symbols.symbol(&cdf::ZERO_MV)? == 0 {
            InterMode::Global
        } else {
            if self.symbols.symbol(&cdf::REF_MV)? != 0 {
                return Err(unsupported("AV1 NEARMV prediction is not supported"));
            }
            InterMode::Nearest
        };
        let motion =
            match mode {
                InterMode::Global => MotionVector::default(),
                InterMode::Nearest => predicted,
                InterMode::New => {
                    let delta = self.decode_new_mv()?;
                    MotionVector {
                        row: predicted.row.checked_add(delta.row).ok_or_else(|| {
                            malformed("AV1 predicted motion-vector row overflows")
                        })?,
                        column: predicted.column.checked_add(delta.column).ok_or_else(|| {
                            malformed("AV1 predicted motion-vector column overflows")
                        })?,
                    }
                }
            };
        Ok(InterPrediction {
            reference_indices: [0, 0],
            motion_vectors: [motion, MotionVector::default()],
            compound: false,
        })
    }

    fn decode_new_mv(&mut self) -> Result<MotionVector> {
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
        Ok(MotionVector { row, column })
    }

    /// Decodes one AV1 `mv_component` (§5.11.32) restricted to integer pel:
    /// the fractional and high-precision bits are still present in the
    /// bitstream (the encoder always signals `allow_high_precision_mv = 0`,
    /// but `mv_class0_fr`/`mv_fr` remain part of the syntax) and are
    /// consumed but ignored here since every motion vector is integer-pel.
    fn decode_mv_component(&mut self, component: usize) -> Result<i32> {
        let sign = self.symbols.symbol(&cdf::MV_SIGN)? != 0;
        let class = self.symbols.symbol(&cdf::MV_CLASS[component])?;
        if class != 0 {
            return Err(unsupported(
                "AV1 motion-vector differences larger than two pixels are not supported",
            ));
        }
        let class0_bit = self.symbols.symbol(&cdf::MV_CLASS0_BIT)?;
        let fraction = self.symbols.symbol(&cdf::MV_CLASS0_FR[component])?;
        if fraction != 3 {
            return Err(unsupported(
                "AV1 inter decoder requires whole-pel motion vectors",
            ));
        }
        let magnitude_eighths = ((class0_bit as i32) << 3) | ((fraction as i32) << 1) | 1;
        let magnitude_eighths = magnitude_eighths + 1;
        let magnitude = magnitude_eighths / 8;
        if magnitude == 0 || magnitude > 2 {
            return Err(malformed("AV1 motion vector magnitude is out of range"));
        }
        Ok(if sign { -magnitude } else { magnitude })
    }

    fn nearest_motion_vector(
        &self,
        row: usize,
        column: usize,
        reference_index: usize,
    ) -> MotionVector {
        let candidates = [
            row.checked_sub(1)
                .map(|above| above * self.mi_cols + column),
            column.checked_sub(1).map(|left| row * self.mi_cols + left),
        ];
        for index in candidates.into_iter().flatten() {
            let Some(block) = self.motion_grid[index] else {
                continue;
            };
            let count = if block.compound { 2 } else { 1 };
            for slot in 0..count {
                if block.reference_indices[slot] == reference_index {
                    return block.motion_vectors[slot];
                }
            }
        }
        MotionVector::default()
    }

    /// Motion-compensates a whole coding block at once through the shared
    /// SIMD primitives in [`crate::av1_mc`]: with integer-pel translational
    /// motion every 4x4 transform block within it shares the same displaced
    /// source region, so the residual stage only needs to add coefficients
    /// on top of this prediction.
    ///
    /// Compound blocks go through the 16x-scale compound prediction path and
    /// the vectorized average blend, which is bit-identical to the
    /// `(a + b + 1) >> 1` this used to compute one sample at a time.
    fn predict_inter_block(
        &mut self,
        row: usize,
        column: usize,
        block_width: usize,
        prediction: InterPrediction,
    ) -> Result<()> {
        let x = column * 4;
        let y = row * 4;
        if x >= self.coded_width || y >= self.coded_height {
            return Ok(());
        }
        // Coding blocks may hang off the right or bottom edge of the coded
        // plane; only the visible part is reconstructed.
        let width = block_width.min(self.coded_width - x);
        let height = block_width.min(self.coded_height - y);
        let reference_count = if prediction.compound { 2 } else { 1 };
        let stride = self.coded_width;
        let filter = self.interpolation_filter;
        for slot in 0..reference_count {
            let reference = self.references[prediction.reference_indices[slot]]
                .ok_or_else(|| malformed("AV1 inter block references an unavailable frame"))?;
            let motion = prediction.motion_vectors[slot];
            let source_x = x as i64 + i64::from(motion.column);
            let source_y = y as i64 + i64::from(motion.row);
            if source_x < 0
                || source_y < 0
                || source_x + width as i64 > reference.width as i64
                || source_y + height as i64 > reference.height as i64
            {
                return Err(malformed(
                    "AV1 motion vector references outside the reference frame",
                ));
            }
            let plane = RefPlane::new(&reference.luma, reference.width, reference.height);
            let (source_x, source_y) = (source_x as i32, source_y as i32);
            if prediction.compound {
                self.compound[slot].clear();
                self.compound[slot].resize(width * height, 0);
                self.mc.predict_compound(
                    plane,
                    source_x,
                    source_y,
                    width,
                    height,
                    0,
                    0,
                    filter,
                    &mut self.compound[slot],
                    width,
                );
            } else {
                self.mc.predict_single(
                    plane,
                    source_x,
                    source_y,
                    width,
                    height,
                    0,
                    0,
                    filter,
                    &mut self.pixels[y * stride + x..],
                    stride,
                );
            }
        }
        if prediction.compound {
            let [first, second] = &self.compound;
            blend_average(
                self.mc.level(),
                first,
                width,
                second,
                width,
                width,
                height,
                &mut self.pixels[y * stride + x..],
                stride,
            );
        }
        Ok(())
    }

    /// `read_tx_size` (spec §5.11.16) for a square `block_width` coding
    /// block, over the square transform sizes this crate's inverse
    /// transform kernels implement (`TX_4X4` through `TX_64X64`); see
    /// [`crate::av1_intra_decoder`]'s identically shaped implementation.
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
            return Err(malformed("AV1 tx_depth exceeds the block maximum"));
        }
        Ok((largest >> depth).max(4))
    }

    fn decode_transform_block(
        &mut self,
        x: usize,
        y: usize,
        tx_width: usize,
        is_inter: bool,
        intra_mode: Av1IntraMode,
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
            if is_inter {
                self.add_residuals(x, y, 4, &residuals);
            } else {
                self.apply_intra_prediction(x, y, 4, intra_mode, &residuals);
            }
            self.tx_sizes.set_block(x, y, 4, 4);
            return Ok(());
        }
        let (coefficients, tx_type) =
            self.decode_coefficients_nonlossless(x >> 2, y >> 2, tx_width)?;
        let dc_quant = get_dc_quant(self.base_q_idx);
        let ac_quant = get_ac_quant(self.base_q_idx);
        let residuals = inverse_transform(&coefficients, tx_width, tx_type, dc_quant, ac_quant);
        if is_inter {
            self.add_residuals(x, y, tx_width, &residuals);
        } else {
            self.apply_intra_prediction(x, y, tx_width, intra_mode, &residuals);
        }
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

    /// Adds the inverse-transformed residual of a `size x size` block on top
    /// of the samples already reconstructed into the plane.
    fn add_residuals(&mut self, x: usize, y: usize, size: usize, residuals: &[i16]) {
        for row in 0..size {
            let start = (y + row) * self.coded_width + x;
            add_residual_row(
                &residuals[row * size..row * size + size],
                &mut self.pixels[start..start + size],
            );
        }
    }

    /// Spec §7.11.2 DC intra prediction, generalized from the original
    /// fixed 4x4 window to any `size x size` (4 or 8) transform block.
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
    /// `read_tx_type`, restricted to this decoder's reduced intra/inter set,
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
        // regardless of tx_type).
        let tx_type = if skipped {
            Av1TxType::DctDct
        } else if let Some(ext_tx) = cdf::ext_tx_cdf(tx_width) {
            match self.symbols.symbol(ext_tx)? {
                0 => Av1TxType::Idtx,
                1 => Av1TxType::DctDct,
                _ => {
                    return Err(unsupported(
                        "AV1 inter decoder does not support the ADST_ADST transform type",
                    ));
                }
            }
        } else {
            // TX_SET_DCTONLY (spec §5.11.47 `get_tx_set`): no symbol.
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
            return Err(malformed(
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

fn malformed(message: impl Into<String>) -> Error {
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
        _ => Err(unsupported("AV1 inter decoder read an invalid y mode")),
    }
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
    use crate::{
        ErrorKind, FilterFrame, FilterPlane, FrameDigest, LoopFilterParams, deblock_frame,
    };

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
            w.bits(0, 1); // show_existing_frame = 0
            w.bits(0, 2); // frame_type = KEY_FRAME
            w.bits(1, 1); // show_frame = 1
            // error_resilient_mode is implied true for a shown key frame.
            w.bits(1, 1); // disable_cdf_update = 1
            // allow_screen_content_tools is implied 0 (seq_force = 0).
            // primary_ref_frame is implied PRIMARY_REF_NONE for intra frames.
            w.bits(0, 1); // frame_size_override_flag
            w.bits(0, order_hint_bits); // order_hint
            // refresh_frame_flags is implied 0xff for a shown key frame.
            // ref_frame_idx() is absent for intra frames.
            w.bits(0, 1); // render_and_frame_size_different
            Self { w }
        }

        fn inter_frame(
            order_hint_bits: usize,
            order_hint: u32,
            refresh_frame_flags: u8,
            ref_frame_idx: [u8; 7],
            ref_order_hints: Option<[u32; NUM_REF_FRAMES]>,
        ) -> Self {
            let mut w = BitWriter::default();
            w.bits(0, 1); // show_existing_frame = 0
            w.bits(1, 2); // frame_type = INTER_FRAME
            w.bits(1, 1); // show_frame = 1 (showable_frame is inferred true)
            w.bits(ref_order_hints.is_some().into(), 1); // error_resilient_mode
            w.bits(1, 1); // disable_cdf_update = 1
            // allow_screen_content_tools is implied 0.
            w.bits(0, 1); // frame_size_override_flag
            w.bits(order_hint, order_hint_bits);
            if ref_order_hints.is_none() {
                w.bits(PRIMARY_REF_NONE.into(), 3);
            }
            w.bits(refresh_frame_flags.into(), 8);
            if let Some(hints) = ref_order_hints {
                for hint in hints {
                    w.bits(hint, order_hint_bits);
                }
            }
            w.bits(0, 1); // frame_refs_short_signaling = 0
            for index in ref_frame_idx {
                w.bits(index.into(), 3);
            }
            w.bits(0, 1); // render_and_frame_size_different
            w.bits(0, 1); // allow_high_precision_mv
            w.bits(0, 1); // is_filter_switchable
            w.bits(0, 2); // interpolation_filter = EIGHTTAP
            w.bits(0, 1); // is_motion_mode_switchable
            Self { w }
        }

        /// Appends `tile_info()`, `quantization_params()`,
        /// `segmentation_params()`, `frame_reference_mode()`,
        /// `reduced_tx_set`, and (for inter frames) the identity
        /// `global_motion_params()`, then byte-aligns. `base_q_idx == 0`
        /// selects the lossless path (no `loop_filter_params()` bits);
        /// otherwise `loop_filter` supplies the six loop filter levels and
        /// sharpness written between `segmentation_params()` and
        /// `reduced_tx_set`, matching `parse_inter_frame_header`.
        fn finish_common(
            self,
            is_intra: bool,
            reference_select: bool,
            mi_cols: usize,
            mi_rows: usize,
        ) -> Vec<u8> {
            self.finish_common_with_quantizer(
                is_intra,
                reference_select,
                mi_cols,
                mi_rows,
                0,
                None,
                false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn finish_common_with_quantizer(
            mut self,
            is_intra: bool,
            reference_select: bool,
            mi_cols: usize,
            mi_rows: usize,
            base_q_idx: u8,
            loop_filter: Option<(u8, u8, u8, u8, u8)>,
            tx_mode_select: bool,
        ) -> Vec<u8> {
            // tile_info(): uniform spacing, one tile (mi_cols/mi_rows <= 64
            // superblock units here, so no increment bits are read).
            self.w.bits(1, 1); // uniform_tile_spacing_flag
            let sb_cols = (mi_cols + 15) >> 4;
            let sb_rows = (mi_rows + 15) >> 4;
            assert!(sb_cols <= 1 && sb_rows <= 1, "fixture exceeds one tile");
            self.w.bits(base_q_idx.into(), 8);
            self.w.bits(0, 1); // delta_q_y_dc
            self.w.bits(0, 1); // using_qmatrix
            self.w.bits(0, 1); // segmentation_enabled
            if base_q_idx != 0 {
                let (y_v, y_h, u, v, sharpness) = loop_filter.unwrap_or((0, 0, 0, 0, 0));
                self.w.bits(y_v.into(), 6);
                self.w.bits(y_h.into(), 6);
                if y_v != 0 || y_h != 0 {
                    self.w.bits(u.into(), 6);
                    self.w.bits(v.into(), 6);
                }
                self.w.bits(sharpness.into(), 3);
                self.w.bits(0, 1); // loop_filter_delta_enabled
                // read_tx_mode(): only present when the frame is not
                // CodedLossless.
                self.w.bits(tx_mode_select.into(), 1);
            }
            if !is_intra {
                self.w.bits(reference_select.into(), 1);
                if reference_select {
                    self.w.bits(0, 1); // skip_mode_present = 0
                }
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

    fn inter_frame_tile(compound: bool, add_dc_residual: bool) -> Vec<u8> {
        let mut e = SymbolEncoder::new();
        e.symbol(&cdf::PARTITION_W16[0], 0); // one 16x16 block
        e.symbol(&cdf::SKIP[0], 0);
        e.symbol(&cdf::IS_INTER, 1);
        if compound {
            e.symbol(&cdf::COMP_MODE, 1);
            e.symbol(&cdf::COMP_REF_TYPE, 0);
            e.symbol(&cdf::UNI_COMP_REF, 0);
            e.symbol(&cdf::UNI_COMP_REF_P1, 0);
            e.symbol(&cdf::COMPOUND_MODE, 6); // GLOBAL_GLOBALMV
        } else {
            e.symbol(&cdf::SINGLE_REF_P1, 0);
            e.symbol(&cdf::SINGLE_REF_P3, 0);
            e.symbol(&cdf::SINGLE_REF_P4, 0);
            e.symbol(&cdf::NEW_MV, 1);
            e.symbol(&cdf::ZERO_MV, 0); // GLOBALMV
        }
        for transform in 0..16 {
            let context = match transform {
                0 => 1,
                1 | 4 if add_dc_residual => 2,
                _ => 1,
            };
            if add_dc_residual && transform == 0 {
                e.symbol(&cdf::TXB_SKIP[context], 0);
                e.symbol(&cdf::EOB_PT_16[0][0], 0);
                e.symbol(&cdf::COEFF_BASE_EOB[0][0], 0);
                e.symbol(&cdf::DC_SIGN[0][0], 0);
            } else {
                e.symbol(&cdf::TXB_SKIP[context], 1);
            }
        }
        e.finish()
    }

    fn inter_frame_temporal_unit(
        order_hint: u32,
        refresh_frame_flags: u8,
        ref_frame_idx: [u8; 7],
        compound: bool,
        add_dc_residual: bool,
    ) -> Vec<u8> {
        let mi = 2 * FRAME_DIM.div_ceil(8) as usize;
        let header = FrameHeaderBuilder::inter_frame(
            3,
            order_hint,
            refresh_frame_flags,
            ref_frame_idx,
            None,
        )
        .finish_common(false, compound, mi, mi);
        let mut payload = header;
        payload.extend_from_slice(&inter_frame_tile(compound, add_dc_residual));
        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]);
        push_obu(&mut stream, 6, &payload);
        stream
    }

    fn error_resilient_inter_temporal_unit(
        order_hint: u32,
        refresh_frame_flags: u8,
        ref_frame_idx: [u8; 7],
        ref_order_hints: [u32; NUM_REF_FRAMES],
    ) -> Vec<u8> {
        let mi = 2 * FRAME_DIM.div_ceil(8) as usize;
        let header = FrameHeaderBuilder::inter_frame(
            3,
            order_hint,
            refresh_frame_flags,
            ref_frame_idx,
            Some(ref_order_hints),
        )
        .finish_common(false, false, mi, mi);
        let mut payload = header;
        payload.extend_from_slice(&inter_frame_tile(false, false));
        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]);
        push_obu(&mut stream, 6, &payload);
        stream
    }

    fn show_existing_temporal_unit(map_index: u8) -> Vec<u8> {
        let mut bits = BitWriter::default();
        bits.bits(1, 1); // show_existing_frame
        bits.bits(map_index.into(), 3);
        bits.bits(1, 1); // trailing_one_bit
        bits.byte_align();
        let mut stream = Vec::new();
        push_obu(&mut stream, 2, &[]);
        push_obu(&mut stream, 3, &bits.into_bytes());
        stream
    }

    /// Wraps `sequence_header_payload()` and `key_frame_tile()` into a
    /// complete low-overhead temporal unit: temporal delimiter, sequence
    /// header, and one Frame OBU (frame header + tile data concatenated,
    /// matching how `Av1Parser` splits `Av1Obu::Frame` payloads).
    fn key_frame_temporal_unit() -> Vec<u8> {
        let mi = 2 * FRAME_DIM.div_ceil(8) as usize;
        let header = FrameHeaderBuilder::key_frame(3).finish_common(true, false, mi, mi);
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
        assert_eq!(frame.planes[0].data.len(), 256);
    }

    /// End-to-end: decoding a real (`CodedLossless`) frame through
    /// [`Av1InterDecoder`] records a [`TxSizeGrid`] reachable via
    /// [`Av1InterDecoder::last_frame_tx_sizes`], and that grid produces the
    /// exact same `deblock_frame` output as `None` would. This is the
    /// spec-correct behavior for every stream this decoder accepts: AV1
    /// forces `TX_MODE_ONLY_4X4` whenever `CodedLossless` is true (see the
    /// `read_tx_mode` comment in `parse_inter_frame_header`), so a real
    /// decode from this decoder can never produce a non-4x4 transform size
    /// to thread into the wide 8/14-tap filters — `filter_length_selection_follows_transform_size`
    /// and `deblock_with_wide_tx_sizes_smooths_further_from_the_edge_than_narrow`
    /// in `av1_filters` already cover that selection logic directly against
    /// a synthetic [`TxSizeGrid`], independent of decoder support for
    /// non-lossless streams.
    #[test]
    fn decoded_frame_tx_size_grid_matches_lossless_narrow_only_filtering() {
        let limits = Limits::default();
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        assert!(decoder.last_frame_tx_sizes().is_none());

        decoder
            .decode_temporal_unit(&key_frame_temporal_unit())
            .unwrap();
        // A shown key frame's own `showable_frame` is spec-false (it was
        // already shown), so reconstruct a showable inter frame to exercise
        // `show_existing_frame` below.
        let frame = decoder
            .decode_temporal_unit(&inter_frame_temporal_unit(1, 0x01, [0; 7], false, false))
            .unwrap();
        let grid = decoder
            .last_frame_tx_sizes()
            .expect("a coded frame was just reconstructed")
            .clone();
        assert_eq!(
            grid,
            TxSizeGrid::new(FRAME_DIM as usize, FRAME_DIM as usize)
        );

        let luma = frame.planes[0].data.clone();
        let mut with_grid = FilterFrame::new_monochrome(
            FilterPlane::from_samples(
                FRAME_DIM as usize,
                FRAME_DIM as usize,
                luma.clone(),
                &limits,
            )
            .unwrap(),
        );
        let mut without_grid = FilterFrame::new_monochrome(
            FilterPlane::from_samples(FRAME_DIM as usize, FRAME_DIM as usize, luma, &limits)
                .unwrap(),
        );
        let params = LoopFilterParams {
            y_vertical_level: 30,
            y_horizontal_level: 30,
            u_level: 0,
            v_level: 0,
            sharpness: 0,
        };
        deblock_frame(&mut with_grid, &params, Some(&grid)).unwrap();
        deblock_frame(&mut without_grid, &params, None).unwrap();
        assert_eq!(with_grid, without_grid);

        // Showing a retained frame reconstructs nothing new, so it clears
        // the grid rather than reporting stale metadata for a different
        // frame.
        decoder
            .decode_temporal_unit(&show_existing_temporal_unit(0))
            .unwrap();
        assert!(decoder.last_frame_tx_sizes().is_none());
    }

    #[test]
    fn sequential_compound_show_existing_and_reset_replay_are_deterministic() {
        let limits = Limits::default();
        let key = key_frame_temporal_unit();
        let first = inter_frame_temporal_unit(1, 0x01, [1; 7], false, true);
        let second = inter_frame_temporal_unit(3, 0x02, [0; 7], false, true);
        let compound = inter_frame_temporal_unit(2, 0x04, [0, 1, 0, 0, 0, 0, 0], true, false);
        let shown = show_existing_temporal_unit(2);

        let decode = |decoder: &mut Av1InterDecoder| -> Result<FrameDigest> {
            decoder.decode_temporal_unit(&key)?;
            decoder.decode_temporal_unit(&first)?;
            decoder.decode_temporal_unit(&second)?;
            let compound_frame = decoder.decode_temporal_unit(&compound)?;
            let digest = FrameDigest::from_frame(&compound_frame)?;
            let shown_frame = decoder.decode_temporal_unit(&shown)?;
            assert_eq!(FrameDigest::from_frame(&shown_frame)?, digest);
            Ok(digest)
        };

        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        decoder.decode_temporal_unit(&key).unwrap();
        assert_eq!(
            decoder
                .decode_temporal_unit(&show_existing_temporal_unit(0))
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
        decoder.reset();
        let first_digest = decode(&mut decoder).unwrap();
        assert_eq!(
            first_digest,
            FrameDigest::from_hex(
                "7ea14a38407023bc5ab57e0e4e49deb4c7aad077b662e5aca2378e033049b829"
            )
            .unwrap()
        );
        decoder.reset();
        assert_eq!(
            decoder.decode_temporal_unit(&shown).unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );
        let replay_digest = decode(&mut decoder).unwrap();
        assert_eq!(replay_digest, first_digest);
    }

    #[test]
    fn invalid_reference_and_motion_data_fail_without_panicking() {
        let limits = Limits::default();
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        decoder
            .decode_temporal_unit(&key_frame_temporal_unit())
            .unwrap();
        decoder.refs[7] = None;

        let invalid_reference = inter_frame_temporal_unit(1, 0, [0, 7, 0, 0, 0, 0, 0], true, false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decoder.decode_temporal_unit(&invalid_reference)
        }));
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );

        let reference = decoder.refs[0].as_ref().unwrap().clone();
        let header = InterFrameHeader {
            frame_type: Av1FrameType::Inter,
            showable_frame: true,
            order_hint: 1,
            refresh_frame_flags: 0,
            ref_order_hints: None,
            ref_frame_idx: [0; 7],
            reference_select: false,
            width: 16,
            height: 16,
            base_q_idx: 0,
            loop_filter: LoopFilterParams::DISABLED,
            tx_mode_select: false,
            interpolation_filter: InterpFilter::Regular,
        };
        let refs: [Option<RefSlot>; 8] = std::array::from_fn(|index| {
            if index == 0 {
                Some(reference.clone())
            } else {
                None
            }
        });
        let mut tile =
            InterTileDecoder::new(&[0xff; 8], 16, 16, 4, 4, &header, &refs, &limits).unwrap();
        let prediction = InterPrediction {
            reference_indices: [0, 0],
            motion_vectors: [MotionVector { row: -1, column: 0 }, MotionVector::default()],
            compound: false,
        };
        assert_eq!(
            tile.predict_inter_block(0, 0, 16, prediction)
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
    }

    #[test]
    fn error_resilient_order_hints_invalidate_stale_reference_slots() {
        let limits = Limits::default();
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        decoder
            .decode_temporal_unit(&key_frame_temporal_unit())
            .unwrap();
        decoder
            .decode_temporal_unit(&inter_frame_temporal_unit(1, 0x01, [1; 7], false, true))
            .unwrap();
        decoder
            .decode_temporal_unit(&inter_frame_temporal_unit(3, 0x02, [0; 7], false, true))
            .unwrap();
        assert_eq!(decoder.refs[1].as_ref().unwrap().order_hint, 3);

        decoder
            .decode_temporal_unit(&error_resilient_inter_temporal_unit(
                4,
                0,
                [0; 7],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ))
            .unwrap();
        assert!(decoder.refs[0].is_some());
        assert!(decoder.refs[1].is_none());
    }

    #[test]
    fn motion_vector_deltas_feed_nearest_prediction() {
        let limits = Limits::default();
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        decoder
            .decode_temporal_unit(&key_frame_temporal_unit())
            .unwrap();
        let reference = decoder.refs[0].as_ref().unwrap().clone();
        let refs: [Option<RefSlot>; 8] = std::array::from_fn(|_| Some(reference.clone()));
        let header = InterFrameHeader {
            frame_type: Av1FrameType::Inter,
            showable_frame: true,
            order_hint: 1,
            refresh_frame_flags: 0,
            ref_order_hints: None,
            ref_frame_idx: [0; 7],
            reference_select: false,
            width: 16,
            height: 16,
            base_q_idx: 0,
            loop_filter: LoopFilterParams::DISABLED,
            tx_mode_select: false,
            interpolation_filter: InterpFilter::Regular,
        };
        let mut symbols = SymbolEncoder::new();
        symbols.symbol(&cdf::MV_JOINT, 1); // nonzero row, zero column
        symbols.symbol(&cdf::MV_SIGN, 0);
        symbols.symbol(&cdf::MV_CLASS[0], 0);
        symbols.symbol(&cdf::MV_CLASS0_BIT, 0);
        symbols.symbol(&cdf::MV_CLASS0_FR[0], 3); // whole-pel +1
        let bytes = symbols.finish();
        let mut tile =
            InterTileDecoder::new(&bytes, 16, 16, 4, 4, &header, &refs, &limits).unwrap();
        let delta = tile.decode_new_mv().unwrap();
        assert_eq!(delta, MotionVector { row: 1, column: 0 });

        tile.motion_grid[0] = Some(BlockMotion {
            reference_indices: [0, 0],
            motion_vectors: [delta, MotionVector::default()],
            compound: false,
        });
        assert_eq!(tile.nearest_motion_vector(0, 1, 0), delta);
        let prediction = InterPrediction {
            reference_indices: [0, 0],
            motion_vectors: [delta, MotionVector::default()],
            compound: false,
        };
        tile.predict_inter_block(0, 0, 8, prediction).unwrap();
        assert_eq!(tile.pixels[0], reference.luma[reference.width]);
    }

    #[test]
    #[ignore = "local standards cross-check"]
    fn generated_sequence_decodes_with_ffmpeg() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut stream = key_frame_temporal_unit();
        stream.extend(inter_frame_temporal_unit(1, 0x01, [1; 7], false, true));
        stream.extend(inter_frame_temporal_unit(3, 0x02, [0; 7], false, true));
        stream.extend(inter_frame_temporal_unit(
            2,
            0x04,
            [0, 1, 0, 0, 0, 0, 0],
            true,
            false,
        ));
        stream.extend(show_existing_temporal_unit(2));
        let mut child = Command::new("ffmpeg")
            .args([
                "-v", "error", "-f", "obu", "-i", "pipe:0", "-pix_fmt", "yuv420p", "-f",
                "rawvideo", "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&stream).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        let frame_bytes = 16 * 16 + 2 * 8 * 8;
        assert_eq!(output.stdout.len(), 5 * frame_bytes);

        let units = temporal_units_from_generated_stream(&stream);
        let mut decoder = Av1InterDecoder::new(Limits::default()).unwrap();
        let mut decoded = None;
        for unit in units.iter().take(4) {
            decoded = Some(decoder.decode_temporal_unit(unit).unwrap());
        }
        let decoded = decoded.unwrap();
        let ffmpeg = &output.stdout[3 * frame_bytes..4 * frame_bytes];
        assert_eq!(&decoded.planes[0].data, &ffmpeg[..256]);
        assert_eq!(&decoded.planes[1].data, &ffmpeg[256..320]);
        assert_eq!(&decoded.planes[2].data, &ffmpeg[320..384]);
    }

    fn temporal_units_from_generated_stream(stream: &[u8]) -> Vec<&[u8]> {
        let starts = stream
            .windows(2)
            .enumerate()
            .filter_map(|(index, bytes)| (bytes == [0x12, 0x00]).then_some(index))
            .collect::<Vec<_>>();
        starts
            .iter()
            .enumerate()
            .map(|(index, &start)| {
                let end = starts.get(index + 1).copied().unwrap_or(stream.len());
                &stream[start..end]
            })
            .collect()
    }

    /// A non-lossless 16x16 key frame tile mirroring
    /// `av1_intra_decoder`'s `non_lossless_key_frame_tile`: the top-level
    /// 64/32-unit partitions are forced splits with no symbol read
    /// (mi_cols = mi_rows = 4 is smaller than every half-block threshold
    /// above bsl=2), landing on a real `PARTITION_W16` read that selects
    /// SPLIT into four 8x8 coding blocks. Each 8x8 block reads
    /// `PARTITION_W8` = NONE, `skip` = 0, and (for a key frame, `is_inter`
    /// is never read) `DC_PRED`, then a `tx_type` symbol and one 8x8
    /// transform's coefficients. Blocks (0,0) and (2,2) (mi coordinates)
    /// carry a large DC coefficient of opposite sign (`DCT_DCT`); blocks
    /// (0,2) and (2,0) are `TXB_SKIP`, producing a real step edge for
    /// [`deblock_frame`] to smooth without cascading through every later
    /// DC-predicted block.
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
        let mi = 2 * FRAME_DIM.div_ceil(8) as usize;
        let header = FrameHeaderBuilder::key_frame(3).finish_common_with_quantizer(
            true,
            false,
            mi,
            mi,
            base_q_idx,
            loop_filter,
            false,
        );
        let mut payload = header;
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
    fn non_lossless_stream_reaches_deblock_frame_with_non_default_transform_sizes() {
        let limits = Limits::default();
        let stream = non_lossless_key_frame_temporal_unit(40, Some((30, 30, 0, 0, 0)));
        let mut decoder = Av1InterDecoder::new(limits).unwrap();
        let frame = decoder.decode_temporal_unit(&stream).unwrap();
        assert_eq!(
            (frame.dimensions.width, frame.dimensions.height),
            (FRAME_DIM, FRAME_DIM)
        );

        // Re-run the reconstruction without deblocking by decoding the
        // tile directly, to confirm deblocking actually changed samples
        // and that the production path genuinely reaches `deblock_frame`
        // with a non-default (non-4x4) `TxSizeGrid`.
        let mut parser = Av1Parser::new(limits).unwrap();
        let obus = parser.parse_low_overhead(&stream).unwrap();
        let sequence = obus
            .iter()
            .find_map(|obu| match obu {
                Av1Obu::SequenceHeader { sequence, .. } => Some(sequence.clone()),
                _ => None,
            })
            .unwrap();
        let payload = obus
            .iter()
            .find_map(|obu| match obu {
                Av1Obu::Frame { payload, .. } => Some(payload.clone()),
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
        let mut bits = HeaderBits::new(&payload);
        let header = parse_inter_frame_header(&mut bits, &sequence, mi_cols, mi_rows).unwrap();
        let tile_offset = bits.position() / 8;
        let tile = &payload[tile_offset..];
        let refs: [Option<RefSlot>; NUM_REF_FRAMES] = Default::default();
        let mut tile_decoder = InterTileDecoder::new(
            tile, width, height, mi_cols, mi_rows, &header, &refs, &limits,
        )
        .unwrap();
        let unfiltered_luma = tile_decoder.decode().unwrap();
        let tx_sizes = tile_decoder.tx_sizes;

        assert_ne!(unfiltered_luma, frame.planes[0].data);

        let mut expected = TxSizeGrid::new(FRAME_DIM as usize, FRAME_DIM as usize);
        expected.set_block(0, 0, 8, 8);
        expected.set_block(8, 0, 8, 8);
        expected.set_block(0, 8, 8, 8);
        expected.set_block(8, 8, 8, 8);
        assert_eq!(tx_sizes, expected);

        let mut filter_frame = FilterFrame::new_monochrome(
            FilterPlane::from_samples(width, height, unfiltered_luma, &limits).unwrap(),
        );
        deblock_frame(&mut filter_frame, &header.loop_filter, Some(&tx_sizes)).unwrap();
        assert_eq!(filter_frame.y.data, frame.planes[0].data);
    }
}
