//! Whole-bitstream decode driver — Annex B byte stream to output
//! pictures.
//!
//! This is the §8.1 general decoding process expressed over the crate's
//! subsystems: the [`crate::hevc::engine::nal`] Annex B demux feeds parameter-set
//! activation ([`crate::hevc::engine::sps`] / [`crate::hevc::engine::pps`]), each coded picture's
//! slice segments are parsed ([`crate::hevc::engine::slice`]) and their
//! `slice_segment_data()` CABAC-decoded through the §7.3.8 syntax walk
//! ([`crate::hevc::engine::slice_data`]), and the decoded coding tree units are handed
//! to the picture-level reconstruction + in-loop-filter driver
//! ([`crate::hevc::engine::inter_recon::reconstruct_inter_picture`]) with the §8.3
//! POC / RPS / reference-list cycle threaded by
//! [`crate::hevc::engine::decode::PictureSequenceState`]. Decoded pictures are
//! returned in output order (§8.3.1 `PicOrderCntVal` order within each
//! coded video sequence).

use std::collections::BTreeMap;

use crate::hevc::engine::availability::{PictureTiling, TilingParams};
use crate::hevc::engine::bitreader::BitReader;
use crate::hevc::engine::cabac::{CabacEngine, init_type};
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::decode::{PictureHeaderInfo, PictureSequenceState, SliceRefParams};
use crate::hevc::engine::dpb::{LongTermEntry, RefPicLists};
use crate::hevc::engine::inter_pred::WpListWeights;
use crate::hevc::engine::inter_recon::{
    InterSliceContext, PlacedInterCtu, RefListAccess, SliceWpTables, reconstruct_inter_picture,
};
use crate::hevc::engine::nal::{NalError, NalIter, NalUnit};
use crate::hevc::engine::picture::Picture;
use crate::hevc::engine::poc::NalKind;
use crate::hevc::engine::pps::{PicParameterSet, PpsError};
use crate::hevc::engine::recon::{ReconError, ReconParams};
use crate::hevc::engine::residual::ResidualCodingError;
use crate::hevc::engine::slice::{
    SliceError, SliceLongTermRefPicSource, SliceSegmentHeader, SliceType,
};
use crate::hevc::engine::slice_data::{
    CodingTreeUnit, PictureParseState, SliceDataParams, decode_coding_tree_unit_in_picture,
    end_of_slice_segment_flag,
};
use crate::hevc::engine::sps::{
    MaterializedShortTermRefPicSet, SeqParameterSet, ShortTermRefPicSetMaterializeError, SpsError,
};

/// NAL unit type: video parameter set (Table 7-1).
const NAL_VPS: u8 = 32;
/// NAL unit type: sequence parameter set.
const NAL_SPS: u8 = 33;
/// NAL unit type: picture parameter set.
const NAL_PPS: u8 = 34;

/// Errors from the whole-bitstream decode driver.
#[derive(Debug)]
pub enum SequenceError {
    /// Annex B demux / NAL header error.
    Nal(NalError),
    /// SPS parse error.
    Sps(SpsError),
    /// PPS parse error.
    Pps(PpsError),
    /// Slice-segment-header parse error.
    Slice(SliceError),
    /// §7.3.8 slice-data CABAC walk error.
    SliceData(ResidualCodingError),
    /// Picture reconstruction error.
    Recon(ReconError),
    /// §7.4.8 short-term-RPS materialization error.
    Rps(ShortTermRefPicSetMaterializeError),
    /// A referenced parameter set was never activated.
    MissingParameterSet {
        /// `"sps"` or `"pps"`.
        kind: &'static str,
        /// The referenced parameter-set id.
        id: u8,
    },
    /// A structural bitstream-conformance failure.
    Malformed(&'static str),
    /// A conformant configuration this driver does not decode yet.
    Unsupported(&'static str),
}

impl core::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Nal(e) => write!(f, "NAL demux error: {e}"),
            Self::Sps(e) => write!(f, "SPS parse error: {e}"),
            Self::Pps(e) => write!(f, "PPS parse error: {e}"),
            Self::Slice(e) => write!(f, "slice header parse error: {e}"),
            Self::SliceData(e) => write!(f, "slice data decode error: {e}"),
            Self::Recon(e) => write!(f, "picture reconstruction error: {e}"),
            Self::Rps(e) => write!(f, "short-term RPS materialization error: {e:?}"),
            Self::MissingParameterSet { kind, id } => {
                write!(f, "referenced {kind} id {id} was never activated")
            }
            Self::Malformed(what) => write!(f, "malformed bitstream: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported configuration: {what}"),
        }
    }
}

impl std::error::Error for SequenceError {}

impl From<NalError> for SequenceError {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}
impl From<SpsError> for SequenceError {
    fn from(e: SpsError) -> Self {
        Self::Sps(e)
    }
}
impl From<PpsError> for SequenceError {
    fn from(e: PpsError) -> Self {
        Self::Pps(e)
    }
}
impl From<SliceError> for SequenceError {
    fn from(e: SliceError) -> Self {
        Self::Slice(e)
    }
}
impl From<ResidualCodingError> for SequenceError {
    fn from(e: ResidualCodingError) -> Self {
        Self::SliceData(e)
    }
}
impl From<ReconError> for SequenceError {
    fn from(e: ReconError) -> Self {
        Self::Recon(e)
    }
}
impl From<ShortTermRefPicSetMaterializeError> for SequenceError {
    fn from(e: ShortTermRefPicSetMaterializeError) -> Self {
        Self::Rps(e)
    }
}

/// One decoded picture with its output-ordering keys.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Index of the coded video sequence this picture belongs to
    /// (incremented at each IRAP with `NoRaslOutputFlag == 1`).
    pub cvs_index: u32,
    /// `PicOrderCntVal` (§8.3.1).
    pub poc: i32,
    /// `pic_output_flag` — `false` pictures are decoded (they may be
    /// referenced) but not output.
    pub output: bool,
    /// The reconstructed, in-loop-filtered picture.
    pub picture: Picture,
}

/// One slice segment of the picture being assembled.
#[derive(Debug)]
struct SegmentData {
    nal_type: u8,
    temporal_id: u8,
    layer_id: u8,
    rbsp: Vec<u8>,
    /// Coded (escaped) payload — the §7.4.7.1 entry-point offsets are
    /// expressed in this byte space.
    escaped: Vec<u8>,
    header: SliceSegmentHeader,
}

/// The whole-bitstream decoder: parameter-set activation + per-picture
/// slice-data decode + the §8.3 reference cycle.
#[derive(Debug, Default)]
pub struct SequenceDecoder {
    sps: BTreeMap<u8, SeqParameterSet>,
    pps: BTreeMap<u8, PicParameterSet>,
    state: PictureSequenceState,
    frames: Vec<DecodedFrame>,
    pending: Vec<SegmentData>,
    cvs_index: u32,
    seen_picture: bool,
    /// Debug: tolerate an end_of_slice_segment_flag mismatch (decode
    /// as much as possible instead of erroring).
    tolerant: bool,
}

impl SequenceDecoder {
    /// A fresh decoder with no activated parameter sets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Debug: keep decoding past an `end_of_slice_segment_flag`
    /// mismatch. Not part of the stable API.
    #[doc(hidden)]
    pub fn set_tolerant(&mut self, tolerant: bool) {
        self.tolerant = tolerant;
    }

    /// Feed a whole Annex B byte stream, decoding every access unit.
    ///
    /// # Errors
    /// Any demux / parse / decode error; the decoder state is
    /// unspecified after an error.
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<(), SequenceError> {
        for unit in NalIter::new(data) {
            let unit = {
                // Issue #189 stage attribution: start-code scan and §7.3.1.1
                // emulation-prevention unescaping.
                let _profile = crate::hevc::engine::profile::scope(
                    crate::hevc::engine::profile::Stage::HeaderParse,
                );
                unit?
            };
            self.push_nal_unit(unit)?;
        }
        Ok(())
    }

    /// Feed one demuxed NAL unit.
    ///
    /// # Errors
    /// Any parse / decode error.
    pub fn push_nal_unit(&mut self, unit: NalUnit) -> Result<(), SequenceError> {
        let NalUnit {
            header,
            rbsp,
            escaped,
        } = unit;
        if header.is_vcl() {
            // §7.4.2.4.4: a VCL NAL with first_slice_segment_in_pic_flag
            // set starts a new access unit.
            let first_in_pic = rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first_in_pic {
                self.finish_picture()?;
            }
            // Issue #189 stage attribution: §7.3.6 slice segment header.
            let _profile = crate::hevc::engine::profile::scope(
                crate::hevc::engine::profile::Stage::HeaderParse,
            );
            let pps_id = peek_slice_pps_id(&rbsp, header.nal_unit_type)?;
            let pps = self
                .pps
                .get(&pps_id)
                .ok_or(SequenceError::MissingParameterSet {
                    kind: "pps",
                    id: pps_id,
                })?;
            let sps = self
                .sps
                .get(&pps.sps_id)
                .ok_or(SequenceError::MissingParameterSet {
                    kind: "sps",
                    id: pps.sps_id,
                })?;
            let parsed = SliceSegmentHeader::parse(&rbsp, header.nal_unit_type, sps, pps)?;
            self.pending.push(SegmentData {
                nal_type: header.nal_unit_type,
                temporal_id: header.temporal_id,
                layer_id: header.nuh_layer_id,
                rbsp,
                escaped,
                header: parsed,
            });
            return Ok(());
        }
        match header.nal_unit_type {
            NAL_VPS => {
                // §7.4.2.4.4: a VPS / SPS / PPS NAL unit (nuh_layer_id
                // 0) succeeding a VCL NAL unit starts a NEW access
                // unit — the pending picture is complete. Decode it
                // BEFORE the arriving parameter set can overwrite the
                // sets it was coded against (streams legally re-send
                // a parameter set with the same id and new content for
                // the next CVS / picture).
                if header.nuh_layer_id == 0 {
                    self.finish_picture()?;
                }
                // The VPS carries no fields the single-layer decode
                // needs; activation is otherwise a no-op.
            }
            NAL_SPS => {
                if header.nuh_layer_id == 0 {
                    self.finish_picture()?;
                }
                let _profile = crate::hevc::engine::profile::scope(
                    crate::hevc::engine::profile::Stage::HeaderParse,
                );
                let sps = SeqParameterSet::parse(&rbsp)?;
                self.sps.insert(sps.sps_id, sps);
            }
            NAL_PPS => {
                if header.nuh_layer_id == 0 {
                    self.finish_picture()?;
                }
                let _profile = crate::hevc::engine::profile::scope(
                    crate::hevc::engine::profile::Stage::HeaderParse,
                );
                let pps = PicParameterSet::parse(&rbsp)?;
                self.pps.insert(pps.pps_id, pps);
            }
            // AUD / EOS / EOB / FD / SEI: nothing to activate.
            _ => {}
        }
        Ok(())
    }

    /// Decode any picture still being assembled (a flush point — call
    /// when the input stream ends but the decoder object lives on).
    ///
    /// # Errors
    /// Any decode error from the pending picture.
    pub fn flush(&mut self) -> Result<(), SequenceError> {
        self.finish_picture()
    }

    /// Drain the pictures decoded so far, in decode order. The caller
    /// owns output reordering (the streaming [`crate::hevc::engine::decoder`] holds a
    /// `sps_max_num_reorder_pics`-deep queue; [`Self::finish`] sorts a
    /// whole sequence at once).
    pub fn take_decoded(&mut self) -> Vec<DecodedFrame> {
        std::mem::take(&mut self.frames)
    }

    /// `sps_max_num_reorder_pics` of the highest sub-layer of the most
    /// recently activated SPS (`None` before any SPS).
    #[must_use]
    pub fn max_num_reorder_pics(&self) -> Option<u32> {
        self.sps.values().next_back().map(|sps| {
            let idx =
                usize::from(sps.max_sub_layers_minus1).min(sps.sub_layer_ordering_info.len() - 1);
            sps.sub_layer_ordering_info[idx].max_num_reorder_pics
        })
    }

    /// Decode any picture still being assembled and return every
    /// decoded frame in output order.
    ///
    /// # Errors
    /// Any decode error from the final pending picture.
    pub fn finish(mut self) -> Result<Vec<DecodedFrame>, SequenceError> {
        self.finish_picture()?;
        // §C.5.2.2 output order: `PicOrderCntVal` order within each
        // coded video sequence, sequences in decode order.
        let mut frames = self.frames;
        frames.sort_by_key(|f| (f.cvs_index, f.poc));
        Ok(frames)
    }

    /// Decode the pending picture's slice segments into a picture.
    fn finish_picture(&mut self) -> Result<(), SequenceError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let segs = std::mem::take(&mut self.pending);
        self.decode_picture(&segs)
    }

    fn decode_picture(&mut self, segs: &[SegmentData]) -> Result<(), SequenceError> {
        let indep = &segs[0];
        if indep.header.dependent_slice_segment_flag {
            return Err(SequenceError::Malformed(
                "first slice segment of a picture is dependent",
            ));
        }
        let pps = self
            .pps
            .get(&indep.header.slice_pic_parameter_set_id)
            .ok_or(SequenceError::MissingParameterSet {
                kind: "pps",
                id: indep.header.slice_pic_parameter_set_id,
            })?;
        let sps = self
            .sps
            .get(&pps.sps_id)
            .ok_or(SequenceError::MissingParameterSet {
                kind: "sps",
                id: pps.sps_id,
            })?;

        let geom = Geometry::derive(sps, pps)?;

        // §7.4.2.4.4 CVS bookkeeping: an IRAP with NoRaslOutputFlag
        // starts a new coded video sequence (for output ordering).
        let nal_kind = NalKind::new(indep.nal_type);
        let no_rasl_output =
            nal_kind.is_idr() || nal_kind.is_bla() || (nal_kind.is_irap() && !self.seen_picture);
        if nal_kind.is_irap() && no_rasl_output && self.seen_picture {
            self.cvs_index += 1;
        }
        self.seen_picture = true;

        // ---- §7.3.8 slice-data CABAC decode of every slice segment ----
        let pic_size_in_ctbs = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
        let mut decoded: Vec<(u32, u32, CodingTreeUnit)> = Vec::new();
        let mut slice_addr_of: Vec<Option<u32>> = vec![None; pic_size_in_ctbs];
        let first_slice_type = segs[0]
            .header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let mut parse_state = PictureParseState::new(&build_slice_data_params(
            &segs[0].header,
            sps,
            pps,
            &geom,
            first_slice_type,
        ));

        // §7.4.7.1 — a dependent slice segment inherits the slice-level
        // header values (and SliceAddrRs) from the preceding independent
        // slice segment; §9.3.2.2 restores its CABAC context variables
        // from the state stored at the end of the previous segment
        // (TableStateIdxDs, §9.3.2.4).
        let mut cur_indep: &SegmentData = indep;
        let mut ds_stored: Option<SliceContexts> = None;
        // §9.3.2.4 WPP snapshot — ONE picture-wide storage: a CTU row
        // started by a later slice segment of the same slice
        // synchronizes from the state stored while an earlier segment
        // decoded the row above (§9.3.2.5, T-availability gated).
        let mut wpp_stored: Option<SliceContexts> = None;
        for seg in segs {
            if seg.header.dependent_slice_segment_flag {
                if ds_stored.is_none() {
                    return Err(SequenceError::Malformed(
                        "dependent slice segment without a preceding segment's context state",
                    ));
                }
            } else {
                cur_indep = seg;
            }
            decode_slice_segment_data(
                seg,
                &cur_indep.header,
                sps,
                pps,
                &geom,
                &mut parse_state,
                &mut decoded,
                &mut slice_addr_of,
                &mut ds_stored,
                &mut wpp_stored,
                self.tolerant,
            )?;
        }

        // ---- §8.3 reference cycle + §8.4/§8.5/§8.7 reconstruction ----
        let slice_type = indep
            .header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let header_info = self.build_header_info(indep, sps, nal_kind, no_rasl_output)?;
        let slice_ref = build_slice_ref_params(&indep.header, pps, slice_type, &header_info);

        let ref_state = {
            // Issue #189 stage attribution: the §8.3.1-§8.3.5 per-picture
            // reference derivation, which is DPB bookkeeping rather than
            // sample work.
            let _profile =
                crate::hevc::engine::profile::scope(crate::hevc::engine::profile::Stage::DpbOutput);
            self.state.begin_picture(&header_info, &slice_ref)
        };
        let lists = ref_state.ref_pic_lists.clone().unwrap_or(RefPicLists {
            list0: Vec::new(),
            list1: None,
        });
        let entries = self.state.dpb().entries();
        let col_field = ref_state.col_pic.map(|idx| &entries[idx].motion);
        let col_poc = ref_state
            .col_pic
            .map(|idx| entries[idx].poc)
            .unwrap_or_default();
        let refs = RefListAccess {
            lists: &lists,
            entries,
        };

        let recon_params = build_recon_params(&indep.header, sps, pps, &geom)?;
        let slice_ctx = build_inter_slice_context(
            &indep.header,
            sps,
            pps,
            &geom,
            &recon_params,
            ref_state.poc.val,
            col_poc,
            ref_state.no_backward_pred,
            slice_type,
        );

        // Per-slice slice_loop_filter_across_slices_enabled_flag
        // (§7.4.7.1: inferred from the PPS flag when absent).
        let mut across_of_slice: BTreeMap<u32, bool> = BTreeMap::new();
        for seg in segs {
            if !seg.header.dependent_slice_segment_flag {
                across_of_slice.insert(
                    seg.header.slice_segment_address,
                    seg.header
                        .slice_loop_filter_across_slices_enabled_flag
                        .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag),
                );
            }
        }
        let placed: Vec<PlacedInterCtu<'_>> = decoded
            .iter()
            .map(|(x, y, ctu)| {
                let rs = (y >> geom.ctb_log2) * geom.pic_w_ctbs + (x >> geom.ctb_log2);
                let slice_addr_rs = slice_addr_of[rs as usize].unwrap_or(0);
                PlacedInterCtu {
                    x_ctb: *x,
                    y_ctb: *y,
                    slice_addr_rs,
                    filter_across_slices: across_of_slice
                        .get(&slice_addr_rs)
                        .copied()
                        .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag),
                    ctu,
                }
            })
            .collect();

        let (picture, motion) = reconstruct_inter_picture(
            geom.width as usize,
            geom.height as usize,
            &recon_params,
            &slice_ctx,
            &geom.tiles,
            &placed,
            &refs,
            col_field,
        )?;

        let output = indep.header.pic_output_flag;
        let poc = ref_state.poc;
        // The picture clone into the output queue and the §8.3.2 DPB insertion
        // are both output-side handling, charged together.
        let _profile =
            crate::hevc::engine::profile::scope(crate::hevc::engine::profile::Stage::DpbOutput);
        self.frames.push(DecodedFrame {
            cvs_index: self.cvs_index,
            poc: poc.val,
            output,
            picture: picture.clone(),
        });
        self.state
            .store_picture(poc, indep.layer_id, picture, motion);
        Ok(())
    }

    /// Assemble the §8.3 [`PictureHeaderInfo`] from the independent
    /// slice segment header.
    fn build_header_info(
        &self,
        seg: &SegmentData,
        sps: &SeqParameterSet,
        nal_kind: NalKind,
        no_rasl_output: bool,
    ) -> Result<PictureHeaderInfo, SequenceError> {
        let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let short_term_rps = materialize_slice_rps(&seg.header, sps)?;
        let mut long_term = Vec::new();
        for lt in &seg.header.long_term_ref_pics {
            let poc_lsb_lt = match lt.source {
                SliceLongTermRefPicSource::Sps { lt_idx_sps } => {
                    sps.long_term_ref_pics
                        .get(lt_idx_sps as usize)
                        .ok_or(SequenceError::Malformed(
                            "lt_idx_sps out of range of the SPS long-term table",
                        ))?
                        .poc_lsb
                }
                SliceLongTermRefPicSource::InSlice { poc_lsb_lt, .. } => poc_lsb_lt,
            };
            let used = lt.used_by_curr_pic_lt(sps).ok_or(SequenceError::Malformed(
                "lt_idx_sps out of range of the SPS long-term table",
            ))?;
            long_term.push(LongTermEntry {
                poc_lsb_lt,
                used_by_curr_pic_lt: used,
                delta_poc_msb_present: lt.delta_poc_msb_present_flag,
                delta_poc_msb_cycle_lt: lt.delta_poc_msb_cycle_lt,
            });
        }
        Ok(PictureHeaderInfo {
            nal_kind,
            temporal_id: seg.temporal_id,
            layer_id: seg.layer_id,
            no_rasl_output,
            poc_lsb: seg.header.slice_pic_order_cnt_lsb.unwrap_or(0),
            max_poc_lsb,
            short_term_rps,
            long_term,
        })
    }
}

/// Decode a whole Annex B byte stream to its output-order frames.
///
/// # Errors
/// Any demux / parse / decode error.
pub fn decode_annexb_sequence(data: &[u8]) -> Result<Vec<DecodedFrame>, SequenceError> {
    let mut dec = SequenceDecoder::new();
    dec.push_annexb(data)?;
    dec.finish()
}

/// Debug helper: CABAC-decode the FIRST picture's slice-segment data and
/// return whatever CTUs were decoded, even when the walk diverges (the
/// `end_of_slice_segment_flag` never fires). Not part of the stable API.
#[doc(hidden)]
pub fn decode_annexb_sequence_debug(
    data: &[u8],
) -> Result<Vec<(u32, u32, CodingTreeUnit)>, SequenceError> {
    // Which picture (1-based) to CABAC-decode; earlier pictures are
    // fully decoded through the normal driver so the DPB and parse
    // state are real.
    let target: usize = std::env::var("H265_DEBUG_PIC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut dec = SequenceDecoder::new();
    let mut segs: Vec<SegmentData> = Vec::new();
    let mut pic_no = 0usize;
    for unit in NalIter::new(data) {
        let unit = unit?;
        if unit.header.is_vcl() {
            let first = unit.rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first {
                pic_no += 1;
                if pic_no > target && !segs.is_empty() {
                    break;
                }
            }
            if pic_no < target {
                dec.push_nal_unit(unit)?;
                continue;
            }
            let pps_id = peek_slice_pps_id(&unit.rbsp, unit.header.nal_unit_type)?;
            let pps = dec.pps.get(&pps_id).unwrap();
            let sps = dec.sps.get(&pps.sps_id).unwrap();
            let parsed =
                SliceSegmentHeader::parse(&unit.rbsp, unit.header.nal_unit_type, sps, pps)?;
            segs.push(SegmentData {
                nal_type: unit.header.nal_unit_type,
                temporal_id: unit.header.temporal_id,
                layer_id: unit.header.nuh_layer_id,
                rbsp: unit.rbsp,
                escaped: unit.escaped,
                header: parsed,
            });
        } else {
            dec.push_nal_unit(unit)?;
        }
    }
    let seg = &segs[0];
    let pps = dec.pps.get(&seg.header.slice_pic_parameter_set_id).unwrap();
    let sps = dec.sps.get(&pps.sps_id).unwrap();
    let geom = Geometry::derive(sps, pps)?;
    let pic_size = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
    let mut decoded = Vec::new();
    let mut slice_addr_of = vec![None; pic_size];
    let st = seg.header.slice_type.unwrap();
    let mut parse_state =
        PictureParseState::new(&build_slice_data_params(&seg.header, sps, pps, &geom, st));
    let mut ds_stored = None;
    let mut wpp_stored = None;
    let res = decode_slice_segment_data(
        seg,
        &seg.header,
        sps,
        pps,
        &geom,
        &mut parse_state,
        &mut decoded,
        &mut slice_addr_of,
        &mut ds_stored,
        &mut wpp_stored,
        false,
    );
    if let Err(e) = res {
        eprintln!("(walk error: {e})");
    }
    Ok(decoded)
}

/// Debug helper: reconstruct the FIRST picture even when the CABAC walk
/// diverges, returning the in-loop-filtered picture. Not stable API.
#[doc(hidden)]
pub fn decode_annexb_first_picture_tolerant(data: &[u8]) -> Result<Picture, SequenceError> {
    let mut dec = SequenceDecoder::new();
    let mut segs: Vec<SegmentData> = Vec::new();
    for unit in NalIter::new(data) {
        let unit = unit?;
        if unit.header.is_vcl() {
            let first = unit.rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first && !segs.is_empty() {
                break;
            }
            let pps_id = peek_slice_pps_id(&unit.rbsp, unit.header.nal_unit_type)?;
            let pps = dec.pps.get(&pps_id).unwrap();
            let sps = dec.sps.get(&pps.sps_id).unwrap();
            let parsed =
                SliceSegmentHeader::parse(&unit.rbsp, unit.header.nal_unit_type, sps, pps)?;
            segs.push(SegmentData {
                nal_type: unit.header.nal_unit_type,
                temporal_id: unit.header.temporal_id,
                layer_id: unit.header.nuh_layer_id,
                rbsp: unit.rbsp,
                escaped: unit.escaped,
                header: parsed,
            });
        } else {
            dec.push_nal_unit(unit)?;
        }
    }
    let seg = &segs[0];
    let pps = dec
        .pps
        .get(&seg.header.slice_pic_parameter_set_id)
        .unwrap()
        .clone();
    let sps = dec.sps.get(&pps.sps_id).unwrap().clone();
    let geom = Geometry::derive(&sps, &pps)?;
    let pic_size = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
    let mut decoded = Vec::new();
    let mut slice_addr_of = vec![None; pic_size];
    let st0 = seg.header.slice_type.unwrap();
    let mut parse_state = PictureParseState::new(&build_slice_data_params(
        &seg.header,
        &sps,
        &pps,
        &geom,
        st0,
    ));
    let mut ds_stored = None;
    let mut wpp_stored = None;
    if let Err(e) = decode_slice_segment_data(
        seg,
        &seg.header,
        &sps,
        &pps,
        &geom,
        &mut parse_state,
        &mut decoded,
        &mut slice_addr_of,
        &mut ds_stored,
        &mut wpp_stored,
        true,
    ) {
        eprintln!("(walk error: {e})");
    }
    let slice_type = seg.header.slice_type.unwrap();
    let recon_params = build_recon_params(&seg.header, &sps, &pps, &geom)?;
    let slice_ctx = build_inter_slice_context(
        &seg.header,
        &sps,
        &pps,
        &geom,
        &recon_params,
        0,
        0,
        true,
        slice_type,
    );
    let placed: Vec<PlacedInterCtu<'_>> = decoded
        .iter()
        .map(|(x, y, ctu)| PlacedInterCtu {
            x_ctb: *x,
            y_ctb: *y,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu,
        })
        .collect();
    let lists = RefPicLists {
        list0: Vec::new(),
        list1: None,
    };
    let refs = RefListAccess {
        lists: &lists,
        entries: &[],
    };
    let (picture, _) = reconstruct_inter_picture(
        geom.width as usize,
        geom.height as usize,
        &recon_params,
        &slice_ctx,
        &geom.tiles,
        &placed,
        &refs,
        None,
    )?;
    Ok(picture)
}

/// The per-SPS geometry constants (§7.4.3.2.1 derived variables).
struct Geometry {
    width: u32,
    height: u32,
    ctb_log2: u32,
    min_cb_log2: u32,
    min_tb_log2: u32,
    max_tb_log2: u32,
    chroma_array_type: u8,
    pic_w_ctbs: u32,
    pic_h_ctbs: u32,
    tiles: TilingParams,
}

impl Geometry {
    fn derive(sps: &SeqParameterSet, pps: &PicParameterSet) -> Result<Self, SequenceError> {
        let min_cb_log2 = u32::from(sps.log2_min_luma_coding_block_size_minus3) + 3;
        let ctb_log2 = min_cb_log2 + u32::from(sps.log2_diff_max_min_luma_coding_block_size);
        let min_tb_log2 = u32::from(sps.log2_min_luma_transform_block_size_minus2) + 2;
        let max_tb_log2 = min_tb_log2 + u32::from(sps.log2_diff_max_min_luma_transform_block_size);
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        if width == 0 || height == 0 {
            return Err(SequenceError::Malformed("zero picture dimensions"));
        }
        let ctb = 1u32 << ctb_log2;
        let chroma_array_type = if sps.separate_colour_plane_flag {
            0
        } else {
            sps.chroma_format_idc
        };
        let tiles = if pps.tiles_enabled_flag {
            TilingParams {
                num_tile_columns_minus1: pps.tiles.num_tile_columns_minus1,
                num_tile_rows_minus1: pps.tiles.num_tile_rows_minus1,
                uniform_spacing_flag: pps.tiles.uniform_spacing_flag,
                column_width_minus1: pps.tiles.column_width_minus1.clone(),
                row_height_minus1: pps.tiles.row_height_minus1.clone(),
            }
        } else {
            TilingParams::single_tile()
        };
        Ok(Self {
            width,
            height,
            ctb_log2,
            min_cb_log2,
            min_tb_log2,
            max_tb_log2,
            chroma_array_type,
            pic_w_ctbs: width.div_ceil(ctb),
            pic_h_ctbs: height.div_ceil(ctb),
            tiles,
        })
    }

    fn tiling(&self) -> Result<PictureTiling, SequenceError> {
        PictureTiling::new(
            self.pic_w_ctbs,
            self.pic_h_ctbs,
            self.width,
            self.height,
            self.ctb_log2,
            self.min_tb_log2,
            &self.tiles,
        )
        .map_err(|_| SequenceError::Malformed("invalid tile geometry"))
    }
}

/// Pre-read `slice_pic_parameter_set_id` from a slice-segment RBSP (the
/// two leading fields before it are fixed-width).
fn peek_slice_pps_id(rbsp: &[u8], nal_unit_type: u8) -> Result<u8, SequenceError> {
    let mut br = BitReader::new(rbsp);
    let _first = br.u1().map_err(|_| {
        SequenceError::Malformed("slice header truncated before first_slice_segment_in_pic_flag")
    })?;
    if (NalKind::BLA_W_LP..=NalKind::RSV_IRAP_VCL23).contains(&nal_unit_type) {
        let _ = br.u1().map_err(|_| {
            SequenceError::Malformed("slice header truncated at no_output_of_prior_pics_flag")
        })?;
    }
    let pps_id = br.ue().map_err(|_| {
        SequenceError::Malformed("slice header truncated at slice_pic_parameter_set_id")
    })?;
    if pps_id > 63 {
        return Err(SequenceError::Malformed(
            "slice_pic_parameter_set_id out of range",
        ));
    }
    Ok(pps_id as u8)
}

/// §7.4.8 — materialize the slice's short-term RPS (SPS-indexed or
/// slice-inline, explicit or inter-RPS-predicted).
fn materialize_slice_rps(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
) -> Result<MaterializedShortTermRefPicSet, SequenceError> {
    // IDR: no RPS block — empty set.
    let Some(sps_flag) = header.short_term_ref_pic_set_sps_flag else {
        return Ok(MaterializedShortTermRefPicSet {
            delta_poc_s0: Vec::new(),
            used_by_curr_pic_s0: Vec::new(),
            delta_poc_s1: Vec::new(),
            used_by_curr_pic_s1: Vec::new(),
        });
    };
    // Materialize the SPS chain (set i may inter-predict from an
    // earlier set).
    let mut chain: Vec<MaterializedShortTermRefPicSet> =
        Vec::with_capacity(sps.short_term_ref_pic_sets.len());
    for (idx, set) in sps.short_term_ref_pic_sets.iter().enumerate() {
        let source = if set.inter_ref_pic_set_prediction_flag {
            let ref_idx = idx
                .checked_sub(set.delta_idx_minus1 as usize + 1)
                .ok_or(SequenceError::Malformed("RefRpsIdx underflow"))?;
            Some(&chain[ref_idx])
        } else {
            None
        };
        chain.push(set.materialize(source)?);
    }
    if sps_flag {
        let idx = header.short_term_ref_pic_set_idx.unwrap_or(0) as usize;
        chain
            .into_iter()
            .nth(idx)
            .ok_or(SequenceError::Malformed("short_term_ref_pic_set_idx OOR"))
    } else {
        let set = header
            .inline_short_term_ref_pic_set
            .as_ref()
            .ok_or(SequenceError::Malformed("missing inline st_ref_pic_set"))?;
        let source = if set.inter_ref_pic_set_prediction_flag {
            // stRpsIdx == num_short_term_ref_pic_sets for the inline set.
            let ref_idx = chain
                .len()
                .checked_sub(set.delta_idx_minus1 as usize + 1)
                .ok_or(SequenceError::Malformed("RefRpsIdx underflow"))?;
            Some(&chain[ref_idx])
        } else {
            None
        };
        Ok(set.materialize(source)?)
    }
}

/// `NumPicTotalCurr` (§7.4.7.2) from the already-resolved picture
/// header info (single-layer; `pps_curr_pic_ref_enabled_flag`
/// contributes the closing `NumPicTotalCurr++`).
fn num_pic_total_curr(info: &PictureHeaderInfo, curr_pic_ref_enabled: bool) -> u32 {
    let st = info
        .short_term_rps
        .used_by_curr_pic_s0
        .iter()
        .chain(info.short_term_rps.used_by_curr_pic_s1.iter())
        .filter(|&&u| u)
        .count();
    let lt = info
        .long_term
        .iter()
        .filter(|e| e.used_by_curr_pic_lt)
        .count();
    (st + lt) as u32 + u32::from(curr_pic_ref_enabled)
}

fn build_slice_ref_params(
    header: &SliceSegmentHeader,
    pps: &PicParameterSet,
    slice_type: SliceType,
    info: &PictureHeaderInfo,
) -> SliceRefParams {
    let is_b = slice_type == SliceType::B;
    let is_inter = slice_type != SliceType::I;
    let curr_pic_ref_enabled = pps
        .pps_scc_extension
        .as_ref()
        .is_some_and(|s| s.pps_curr_pic_ref_enabled_flag);
    SliceRefParams {
        is_inter,
        is_b,
        num_ref_idx_l0_active_minus1: u32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ),
        num_ref_idx_l1_active_minus1: u32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ),
        num_pic_total_curr: num_pic_total_curr(info, curr_pic_ref_enabled),
        temporal_mvp_enabled: header.slice_temporal_mvp_enabled_flag,
        collocated_from_l0_flag: header.collocated_from_l0_flag.unwrap_or(true),
        collocated_ref_idx: header.collocated_ref_idx.unwrap_or(0),
        curr_pic_ref_enabled,
    }
}

fn build_recon_params(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
) -> Result<ReconParams, SequenceError> {
    let slice_qp_y = header
        .slice_qp_y(pps)
        .ok_or(SequenceError::Malformed("slice header without slice_qp"))?;
    let range = sps.sps_range_extension.as_ref();
    // §7.4.5: when scaling_list_enabled_flag == 1 the active
    // scaling-list data is the PPS body if present, else the SPS body
    // if present, else the default lists.
    let scaling = if sps.scaling_list_enabled_flag {
        let factors = match (&pps.scaling_list_data, &sps.scaling_list_data) {
            (Some(d), _) => d.scaling_factors(geom.chroma_array_type),
            (None, Some(d)) => d.scaling_factors(geom.chroma_array_type),
            (None, None) => crate::hevc::engine::scaling_list::ScalingListData::all_default()
                .scaling_factors(geom.chroma_array_type),
        };
        Some(factors)
    } else {
        None
    };
    Ok(ReconParams {
        chroma_array_type: geom.chroma_array_type,
        bit_depth_luma: sps.bit_depth_luma_minus8 + 8,
        bit_depth_chroma: sps.bit_depth_chroma_minus8 + 8,
        intra_smoothing_disabled: range.is_some_and(|r| r.intra_smoothing_disabled_flag),
        strong_intra_smoothing_enabled: sps.strong_intra_smoothing_enabled_flag,
        slice_qp_y,
        cb_qp_offset: i32::from(pps.pps_cb_qp_offset) + i32::from(header.slice_cb_qp_offset),
        cr_qp_offset: i32::from(pps.pps_cr_qp_offset) + i32::from(header.slice_cr_qp_offset),
        // §7.4.3.3.3: PpsActQpOffset{Y,Cb,Cr} = pps_act_{y,cb}_qp_offset_plus5 − 5
        // / pps_act_cr_qp_offset_plus3 − 3; the slice offsets add on top
        // (§7.4.7.1), each 0 when absent.
        act_y_qp_offset: pps
            .pps_scc_extension
            .as_ref()
            .map_or(-5, |s| s.pps_act_y_qp_offset_plus5 - 5)
            + header.slice_act_y_qp_offset,
        act_cb_qp_offset: pps
            .pps_scc_extension
            .as_ref()
            .map_or(-5, |s| s.pps_act_cb_qp_offset_plus5 - 5)
            + header.slice_act_cb_qp_offset,
        act_cr_qp_offset: pps
            .pps_scc_extension
            .as_ref()
            .map_or(-3, |s| s.pps_act_cr_qp_offset_plus3 - 3)
            + header.slice_act_cr_qp_offset,
        transform_skip_rotation_enabled: range
            .is_some_and(|r| r.transform_skip_rotation_enabled_flag),
        implicit_rdpcm_enabled: range.is_some_and(|r| r.implicit_rdpcm_enabled_flag),
        intra_boundary_filtering_disabled: sps
            .sps_scc_extension
            .as_ref()
            .is_some_and(|s| s.intra_boundary_filtering_disabled_flag),
        extended_precision: range.is_some_and(|r| r.extended_precision_processing_flag),
        scaling,
        chroma_qp_offset_list: pps
            .pps_range_extension
            .as_ref()
            .map(|r| {
                r.chroma_qp_offset_list
                    .iter()
                    .map(|e| (i32::from(e.cb_qp_offset), i32::from(e.cr_qp_offset)))
                    .collect()
            })
            .unwrap_or_default(),
        cu_qp_offset_c: core::cell::Cell::new((0, 0)),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_inter_slice_context(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    recon: &ReconParams,
    curr_poc: i32,
    col_poc: i32,
    no_backward_pred: bool,
    slice_type: SliceType,
) -> InterSliceContext {
    let pps_range = pps.pps_range_extension.as_ref();
    let deblock = header.deblocking.as_ref();
    // §8.5.3.3.4.1 — weightedPredFlag: weighted_pred_flag for P slices,
    // weighted_bipred_flag for B slices.
    let weighted_pred_flag = match slice_type {
        SliceType::P => pps.weighted_pred_flag,
        SliceType::B => pps.weighted_bipred_flag,
        SliceType::I => false,
    };
    let wp = if weighted_pred_flag {
        header
            .pred_weight_table
            .as_ref()
            .map(|pwt| build_slice_wp_tables(pwt, sps))
    } else {
        None
    };
    InterSliceContext {
        curr_poc,
        constrained_intra_pred: pps.constrained_intra_pred_flag,
        slice_is_b: slice_type == SliceType::B,
        ctb_log2_size_y: geom.ctb_log2,
        pic_width_luma: geom.width,
        pic_height_luma: geom.height,
        max_num_merge_cand: usize::from(header.max_num_merge_cand().unwrap_or(5)),
        num_ref_idx_l0_active: i32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ) + 1,
        num_ref_idx_l1_active: i32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ) + 1,
        log2_par_mrg_level: pps.log2_parallel_merge_level_minus2 + 2,
        temporal_mvp_enabled: header.slice_temporal_mvp_enabled_flag,
        collocated_from_l0_flag: header.collocated_from_l0_flag.unwrap_or(true),
        col_poc,
        no_backward_pred,
        min_tb_log2_size_y: geom.min_tb_log2,
        log2_min_cu_qp_delta_size: geom.ctb_log2 - pps.diff_cu_qp_delta_depth,
        wpp_qp_row_reset: pps.entropy_coding_sync_enabled_flag,
        filter_across_slices: header
            .slice_loop_filter_across_slices_enabled_flag
            .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag),
        filter_across_tiles: pps.loop_filter_across_tiles_enabled_flag,
        deblock_enabled: deblock.is_none_or(|d| !d.disabled_flag),
        beta_offset_div2: deblock.map_or(0, |d| i32::from(d.beta_offset_div2)),
        tc_offset_div2: deblock.map_or(0, |d| i32::from(d.tc_offset_div2)),
        slice_qp_y: recon.slice_qp_y,
        cb_qp_offset: recon.cb_qp_offset,
        cr_qp_offset: recon.cr_qp_offset,
        pps_cb_qp_offset: i32::from(pps.pps_cb_qp_offset),
        pps_cr_qp_offset: i32::from(pps.pps_cr_qp_offset),
        slice_sao_luma_flag: header.slice_sao_luma_flag,
        slice_sao_chroma_flag: header.slice_sao_chroma_flag,
        log2_sao_offset_scale_luma: pps_range.map_or(0, |r| r.log2_sao_offset_scale_luma as u8),
        log2_sao_offset_scale_chroma: pps_range.map_or(0, |r| r.log2_sao_offset_scale_chroma as u8),
        wp,
        pcm_loop_filter_disabled: sps
            .pcm
            .as_ref()
            .is_some_and(|p| p.loop_filter_disabled_flag),
        use_integer_mv: header.use_integer_mv_flag,
        // §7.4.3.3.3 eq. 7-40: TwoVersionsOfCurrDecPicFlag =
        // pps_curr_pic_ref_enabled_flag && ( sao enabled ||
        // !pps_deblocking_filter_disabled_flag ||
        // deblocking_filter_override_enabled_flag ).
        two_versions_curr_pic: pps
            .pps_scc_extension
            .as_ref()
            .is_some_and(|s| s.pps_curr_pic_ref_enabled_flag)
            && (sps.sample_adaptive_offset_enabled_flag
                || !pps.deblocking.disabled_flag
                || pps.deblocking.override_enabled_flag),
    }
}

/// §7.4.7.3 — resolve a parsed `pred_weight_table()` into the
/// per-reference values the §8.5.3.3.4.3 combine reads: `LumaWeightLX[i]`
/// / `ChromaWeightLX[i][j]` (weight-flag inference included), the
/// `WpOffsetBdShiftY`- / `WpOffsetBdShiftC`-scaled offsets (equations
/// 7-31 / 7-32 + 8-268 / 8-269 / 8-273 / 8-274), and the equation-7-58
/// `ChromaOffsetLX` derivation.
fn build_slice_wp_tables(
    pwt: &crate::hevc::engine::slice::PredWeightTable,
    sps: &SeqParameterSet,
) -> SliceWpTables {
    let hp = sps
        .sps_range_extension
        .as_ref()
        .is_some_and(|r| r.high_precision_offsets_enabled_flag);
    let bd_y = i32::from(sps.bit_depth_luma_minus8) + 8;
    let bd_c = i32::from(sps.bit_depth_chroma_minus8) + 8;
    // Equations 7-31 / 7-32 / 7-34.
    let bd_shift_y = if hp { 0 } else { bd_y - 8 };
    let bd_shift_c = if hp { 0 } else { bd_c - 8 };
    let half_range_c = 1i32 << (if hp { bd_c - 1 } else { 7 });
    let chroma_denom = pwt.chroma_log2_weight_denom();

    let resolve = |l0: bool, n: usize| -> Vec<WpListWeights> {
        (0..n)
            .map(|i| {
                let (lw, lo, cw0, cw1, co0, co1) = if l0 {
                    (
                        pwt.luma_weight_l0(i),
                        pwt.entries_l0.get(i).map(|e| e.luma_offset),
                        pwt.chroma_weight_l0(i, 0),
                        pwt.chroma_weight_l0(i, 1),
                        pwt.chroma_offset_l0(i, 0, half_range_c),
                        pwt.chroma_offset_l0(i, 1, half_range_c),
                    )
                } else {
                    (
                        pwt.luma_weight_l1(i),
                        pwt.entries_l1.get(i).map(|e| e.luma_offset),
                        pwt.chroma_weight_l1(i, 0),
                        pwt.chroma_weight_l1(i, 1),
                        pwt.chroma_offset_l1(i, 0, half_range_c),
                        pwt.chroma_offset_l1(i, 1, half_range_c),
                    )
                };
                WpListWeights {
                    w_luma: lw.unwrap_or(1 << pwt.luma_log2_weight_denom),
                    o_luma: lo.unwrap_or(0) << bd_shift_y,
                    w_cb: cw0.unwrap_or(1 << chroma_denom),
                    o_cb: co0.unwrap_or(0) << bd_shift_c,
                    w_cr: cw1.unwrap_or(1 << chroma_denom),
                    o_cr: co1.unwrap_or(0) << bd_shift_c,
                }
            })
            .collect()
    };

    SliceWpTables {
        luma_log2_weight_denom: pwt.luma_log2_weight_denom,
        chroma_log2_weight_denom: chroma_denom,
        l0: resolve(true, pwt.entries_l0.len()),
        l1: resolve(false, pwt.entries_l1.len()),
    }
}

/// §7.4.3 — derive the [`SliceDataParams`] for one slice segment.
fn build_slice_data_params(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    slice_type: SliceType,
) -> SliceDataParams {
    let pps_range = pps.pps_range_extension.as_ref();
    let (log2_min_ipcm, log2_max_ipcm) = sps.pcm.as_ref().map_or((3, 5), |p| {
        let min = u32::from(p.log2_min_pcm_luma_coding_block_size_minus3) + 3;
        (
            min,
            min + u32::from(p.log2_diff_max_min_pcm_luma_coding_block_size),
        )
    });
    let cu_chroma_qp_offset_enabled = header.cu_chroma_qp_offset_enabled_flag;
    let log2_min_cu_chroma_qp_offset_size =
        geom.ctb_log2 - pps_range.map_or(0, |r| r.diff_cu_chroma_qp_offset_depth);
    let scc = sps.sps_scc_extension.as_ref();
    let palette_max_size = scc.map_or(0, |e| e.palette_max_size);
    SliceDataParams {
        ctb_log2_size_y: geom.ctb_log2,
        min_cb_log2_size_y: geom.min_cb_log2,
        max_tb_log2_size_y: geom.max_tb_log2,
        min_tb_log2_size_y: geom.min_tb_log2,
        pic_width_in_luma_samples: geom.width,
        pic_height_in_luma_samples: geom.height,
        chroma_array_type: geom.chroma_array_type,
        bit_depth_luma: u32::from(sps.bit_depth_luma_minus8) + 8,
        bit_depth_chroma: u32::from(sps.bit_depth_chroma_minus8) + 8,
        slice_type_is_i: slice_type == SliceType::I,
        slice_type_is_b: slice_type == SliceType::B,
        slice_sao_luma_flag: header.slice_sao_luma_flag,
        slice_sao_chroma_flag: header.slice_sao_chroma_flag,
        transquant_bypass_enabled_flag: pps.transquant_bypass_enabled_flag,
        cu_qp_delta_enabled_flag: pps.cu_qp_delta_enabled_flag,
        log2_min_cu_qp_delta_size: geom.ctb_log2 - pps.diff_cu_qp_delta_depth,
        cu_chroma_qp_offset_enabled_flag: cu_chroma_qp_offset_enabled,
        log2_min_cu_chroma_qp_offset_size,
        chroma_qp_offset_list_len_minus1: pps_range
            .map_or(0, |r| r.chroma_qp_offset_list_len_minus1),
        amp_enabled_flag: sps.amp_enabled_flag,
        pcm_enabled_flag: sps.pcm_enabled_flag,
        log2_min_ipcm_cb_size_y: log2_min_ipcm,
        log2_max_ipcm_cb_size_y: log2_max_ipcm,
        pcm_bit_depth_luma: sps
            .pcm
            .as_ref()
            .map_or(8, |p| u32::from(p.bit_depth_luma_minus1) + 1),
        pcm_bit_depth_chroma: sps
            .pcm
            .as_ref()
            .map_or(8, |p| u32::from(p.bit_depth_chroma_minus1) + 1),
        max_transform_hierarchy_depth_intra: u32::from(sps.max_transform_hierarchy_depth_intra),
        max_transform_hierarchy_depth_inter: u32::from(sps.max_transform_hierarchy_depth_inter),
        max_num_merge_cand: u32::from(header.max_num_merge_cand().unwrap_or(5)),
        num_ref_idx_l0_active_minus1: u32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ),
        num_ref_idx_l1_active_minus1: u32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ),
        mvd_l1_zero_flag: header.mvd_l1_zero_flag.unwrap_or(false),
        sign_data_hiding_enabled_flag: pps.sign_data_hiding_enabled_flag,
        cross_component_prediction_enabled_flag: pps_range
            .is_some_and(|r| r.cross_component_prediction_enabled_flag),
        residual_adaptive_colour_transform_enabled_flag: pps
            .pps_scc_extension
            .as_ref()
            .is_some_and(|s| s.residual_adaptive_colour_transform_enabled_flag),
        transform_skip_enabled_flag: pps.transform_skip_enabled_flag,
        log2_max_transform_skip_size: pps_range
            .map_or(2, |r| r.log2_max_transform_skip_block_size_minus2 + 2),
        implicit_rdpcm_enabled_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.implicit_rdpcm_enabled_flag),
        explicit_rdpcm_enabled_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.explicit_rdpcm_enabled_flag),
        transform_skip_context_enabled_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.transform_skip_context_enabled_flag),
        persistent_rice_adaptation_enabled_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.persistent_rice_adaptation_enabled_flag),
        cabac_bypass_alignment_enabled_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.cabac_bypass_alignment_enabled_flag),
        extended_precision_processing_flag: sps
            .sps_range_extension
            .as_ref()
            .is_some_and(|r| r.extended_precision_processing_flag),
        palette_mode_enabled_flag: scc.is_some_and(|e| e.palette_mode_enabled_flag),
        palette_max_size,
        palette_max_predictor_size: palette_max_size
            + scc.map_or(0, |e| e.delta_palette_max_predictor_size),
    }
}

/// §7.3.8.1 — CABAC-decode one slice segment's `slice_segment_data()`,
/// appending its CTUs (in tile-scan order) to `decoded` and recording
/// each CTB's `SliceAddrRs` in `slice_addr_of`.
///
/// `effective_header` supplies the slice-level values — for an
/// independent segment it is `seg.header` itself; for a dependent
/// segment it is the preceding independent segment's header (§7.4.7.1
/// inheritance). `ds_stored` is the picture's §9.3.2.4
/// `TableStateIdxDs` context store: read (synchronized, §9.3.2.5 /
/// §9.3.2.2) at a dependent segment's start, written at every
/// segment's `end_of_slice_segment_flag == 1` while
/// `dependent_slice_segments_enabled_flag` is set.
#[allow(clippy::too_many_arguments)]
fn decode_slice_segment_data(
    seg: &SegmentData,
    effective_header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    state: &mut PictureParseState,
    decoded: &mut Vec<(u32, u32, CodingTreeUnit)>,
    slice_addr_of: &mut [Option<u32>],
    ds_stored: &mut Option<SliceContexts>,
    wpp_stored: &mut Option<SliceContexts>,
    tolerant: bool,
) -> Result<(), SequenceError> {
    let header = effective_header;
    let slice_type = header
        .slice_type
        .ok_or(SequenceError::Malformed("independent slice without type"))?;
    let params = build_slice_data_params(header, sps, pps, geom, slice_type);
    let slice_qp_y = header
        .slice_qp_y(pps)
        .ok_or(SequenceError::Malformed("slice header without slice_qp"))?;

    let data_offset = seg
        .header
        .byte_offset_to_slice_data
        .ok_or(SequenceError::Malformed("slice header without data offset"))?;
    if data_offset >= seg.rbsp.len() {
        return Err(SequenceError::Malformed("slice data offset out of range"));
    }

    // §7.4.7.1 — split `slice_segment_data( )` into its subsets. The
    // entry-point offsets count CODED bytes (emulation-prevention bytes
    // included), so map each escaped boundary onto the stripped RBSP.
    // (Entry points are per-segment syntax: read them from the
    // segment's own header even when it is dependent.)
    let substreams = split_substreams(
        &seg.escaped,
        seg.rbsp.len(),
        data_offset,
        seg.header.entry_point_offsets.as_ref(),
    )?;

    // Table 9-4 initType: I => 0; P => cabac_init ? 2 : 1;
    // B => cabac_init ? 1 : 2 (crate::hevc::engine::cabac::init_type on the raw
    // slice_type value).
    let raw_slice_type = match slice_type {
        SliceType::B => 0,
        SliceType::P => 1,
        SliceType::I => 2,
    };
    let it = init_type(raw_slice_type, header.cabac_init_flag.unwrap_or(false));

    let tiling = geom.tiling()?;
    let tiles_on = pps.tiles_enabled_flag;
    let wpp = pps.entropy_coding_sync_enabled_flag;
    // §7.4.7.1: SliceAddrRs is the INDEPENDENT segment's address; a
    // dependent segment starts decoding at its own segment address but
    // its CTBs belong to the inherited slice.
    let slice_addr_rs = header.slice_segment_address;
    let mut ctb_addr_ts = tiling.ctb_addr_rs_to_ts(seg.header.slice_segment_address);
    let pic_size_in_ctbs = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;

    let sub_range = |idx: usize| -> Result<&[u8], SequenceError> {
        let &(a, b) = substreams
            .get(idx)
            .ok_or(SequenceError::Malformed("more CTB rows than substreams"))?;
        seg.rbsp
            .get(a..b)
            .ok_or(SequenceError::Malformed("substream range out of RBSP"))
    };
    let mut sub_idx = 0usize;
    let mut engine = CabacEngine::new(BitReader::new(sub_range(0)?))
        .map_err(|_| SequenceError::Malformed("slice data too short for CABAC init"))?;
    // §9.3.2.2 — a dependent slice segment synchronizes its context
    // variables from TableStateIdxDs (§9.3.2.5) instead of
    // re-initializing.
    // §9.3.2.3 — the palette predictor re-initialization value (the
    // PPS initializers if present, else the SPS initializers, else
    // empty), applied wherever §9.3.2.1 re-initializes the context
    // variables. A dependent segment SYNCHRONIZES the predictor from
    // the stored state instead (it travels inside SliceContexts).
    let num_comps = if geom.chroma_array_type == 0 { 1 } else { 3 };
    let base_palette_predictor = pps
        .pps_scc_extension
        .as_ref()
        .filter(|e| e.pps_palette_predictor_initializers_present_flag)
        .map(|e| {
            crate::hevc::engine::palette::PalettePredictor::from_initializers(
                &e.pps_palette_predictor_initializer,
                num_comps,
            )
        })
        .or_else(|| {
            sps.sps_scc_extension
                .as_ref()
                .filter(|e| e.sps_palette_predictor_initializers_present_flag)
                .map(|e| {
                    crate::hevc::engine::palette::PalettePredictor::from_initializers(
                        &e.sps_palette_predictor_initializer,
                        num_comps,
                    )
                })
        })
        .unwrap_or_default();
    let fresh_contexts = || {
        let mut c = SliceContexts::init(it, slice_qp_y);
        c.palette_predictor = base_palette_predictor.clone();
        c
    };
    // §6.4.1-gated availability of the spatial neighbour T (eq. 9-3,
    // the above-right CTB) for the §9.3.2.5 WPP synchronization: T
    // must exist, lie in the SAME slice (the stored snapshot may come
    // from an earlier slice segment of that slice) and the same tile.
    let t_available = |ctb_addr_ts: u32, slice_addr_of: &[Option<u32>]| {
        let rs = tiling.ctb_addr_ts_to_rs(ctb_addr_ts);
        let (rx, ry) = (rs % geom.pic_w_ctbs, rs / geom.pic_w_ctbs);
        ry > 0 && rx + 1 < geom.pic_w_ctbs && {
            let t_rs = (ry - 1) * geom.pic_w_ctbs + rx + 1;
            slice_addr_of[t_rs as usize] == Some(slice_addr_rs)
                && tiling.tile_id(tiling.ctb_addr_rs_to_ts(t_rs)) == tiling.tile_id(ctb_addr_ts)
        }
    };
    // §9.3.2.1 — the initial context state of this slice segment. For
    // a DEPENDENT segment the branch order matters: a segment whose
    // first CTU is the first CTU of a tile RE-INITIALIZES (§9.3.2.2 /
    // §9.3.2.3), one whose first CTU starts a CTU row of a tile under
    // entropy_coding_sync SYNCHRONIZES from the WPP snapshot
    // (§9.3.2.5, T-availability gated), and only otherwise does the
    // §9.3.2.5 dependent-segment synchronization from TableStateIdxDs
    // apply. An independent segment re-initializes.
    let mut ctx = if seg.header.dependent_slice_segment_flag {
        let first_rs = seg.header.slice_segment_address;
        let (rx, _ry) = (first_rs % geom.pic_w_ctbs, first_rs / geom.pic_w_ctbs);
        let tile_start = tiles_on
            && ctb_addr_ts > 0
            && tiling.tile_id(ctb_addr_ts) != tiling.tile_id(ctb_addr_ts - 1);
        let wpp_row_start = wpp
            && !tile_start
            && (rx == 0
                || tiling.tile_id(tiling.ctb_addr_rs_to_ts(first_rs - 1))
                    != tiling.tile_id(ctb_addr_ts));
        if tile_start {
            fresh_contexts()
        } else if wpp_row_start {
            match (&*wpp_stored, t_available(ctb_addr_ts, slice_addr_of)) {
                (Some(stored), true) => stored.clone(),
                _ => fresh_contexts(),
            }
        } else {
            ds_stored.clone().ok_or(SequenceError::Malformed(
                "dependent segment without Ds state",
            ))?
        }
    } else {
        fresh_contexts()
    };
    let mut first_ctu = true;
    // Set after the row-final CTU's end_of_subset_one_bit: the next CTU
    // starts a new substream.
    let mut advance_substream = false;

    loop {
        if (ctb_addr_ts as usize) >= pic_size_in_ctbs {
            return Err(SequenceError::Malformed(
                "slice segment runs past the last CTB of the picture",
            ));
        }
        let ctb_addr_rs = tiling.ctb_addr_ts_to_rs(ctb_addr_ts);
        let rx = ctb_addr_rs % geom.pic_w_ctbs;
        let ry = ctb_addr_rs / geom.pic_w_ctbs;
        let x_ctb = rx << geom.ctb_log2;
        let y_ctb = ry << geom.ctb_log2;
        slice_addr_of[ctb_addr_rs as usize] = Some(slice_addr_rs);

        // §9.3.1 / §9.3.2.1 — subset-boundary context handling inside
        // one slice segment. Item 2: the first CTU of a tile
        // re-initializes the context variables (§9.3.2.2). Item 3
        // (WPP): the first luma CTB of a CTU row of a tile either
        // synchronizes from the stored above-right state (§9.3.2.5) or
        // re-initializes (§9.3.2.2). Either way the next entry-point
        // substream starts here. The first CTU of the segment keeps
        // the §9.3.1-item-1 slice initialization done above.
        if !first_ctu {
            let tile_start =
                tiles_on && tiling.tile_id(ctb_addr_ts) != tiling.tile_id(ctb_addr_ts - 1);
            // §9.3.2.1: CtbAddrInRs % PicWidthInCtbsY == 0, or the
            // raster-left neighbour lies in a different tile.
            let wpp_row_start = wpp
                && !tile_start
                && (rx == 0
                    || tiling.tile_id(tiling.ctb_addr_rs_to_ts(ctb_addr_rs - 1))
                        != tiling.tile_id(ctb_addr_ts));
            if tile_start || wpp_row_start {
                if !advance_substream {
                    return Err(SequenceError::Malformed(
                        "subset start without end_of_subset_one_bit",
                    ));
                }
                sub_idx += 1;
                engine = CabacEngine::new(BitReader::new(sub_range(sub_idx)?))
                    .map_err(|_| SequenceError::Malformed("substream too short for CABAC init"))?;
                if tile_start {
                    // §9.3.2.2 / §9.3.2.3 — fresh contexts (and
                    // re-initialized palette predictor) at the tile
                    // start.
                    ctx = fresh_contexts();
                } else {
                    // Spatial neighbour T = the CTB at ( x0 + CtbSizeY,
                    // y0 − CtbSizeY ) (eq. 9-3), §6.4.1-gated.
                    ctx = match (&*wpp_stored, t_available(ctb_addr_ts, slice_addr_of)) {
                        (Some(stored), true) => stored.clone(),
                        _ => fresh_contexts(),
                    };
                }
            }
        }
        advance_substream = false;
        first_ctu = false;

        // §7.3.8.3 SAO merge-candidate availability: the left / above
        // CTB must exist, lie in the same slice segment sequence
        // (same SliceAddrRs) and the same tile.
        let tile_here = tiling.tile_id(ctb_addr_ts);
        let merge_left = rx > 0 && {
            let left_rs = ctb_addr_rs - 1;
            slice_addr_of[left_rs as usize] == Some(slice_addr_rs)
                && tiling.tile_id(tiling.ctb_addr_rs_to_ts(left_rs)) == tile_here
        };
        let merge_up = ry > 0 && {
            let up_rs = ctb_addr_rs - geom.pic_w_ctbs;
            slice_addr_of[up_rs as usize] == Some(slice_addr_rs)
                && tiling.tile_id(tiling.ctb_addr_rs_to_ts(up_rs)) == tile_here
        };

        let ctu = decode_coding_tree_unit_in_picture(
            &mut engine,
            &mut ctx,
            &params,
            state,
            x_ctb,
            y_ctb,
            slice_addr_rs,
            tile_here,
            merge_left,
            merge_up,
        )?;
        decoded.push((x_ctb, y_ctb, ctu));

        // §9.3.1 / §9.3.2.4 — store the context state after the SECOND
        // CTB of a CTU row of a tile: CtbAddrInRs % PicWidthInCtbsY
        // == 1, or CtbAddrInRs > 1 and the CTB two to the raster-left
        // lies in a different tile.
        if wpp
            && (rx == 1
                || (ctb_addr_rs > 1
                    && tiles_on
                    && tiling.tile_id(ctb_addr_ts)
                        != tiling.tile_id(tiling.ctb_addr_rs_to_ts(ctb_addr_rs - 2))))
        {
            *wpp_stored = Some(ctx.clone());
        }

        let eos = end_of_slice_segment_flag(&mut engine)
            .map_err(|_| SequenceError::Malformed("CABAC underrun at end_of_slice_segment"))?;
        ctb_addr_ts += 1;
        if eos {
            // §9.3.1 / §9.3.2.4 — store the context variables into
            // TableStateIdxDs for a following dependent slice segment.
            if pps.dependent_slice_segments_enabled_flag {
                *ds_stored = Some(ctx.clone());
            }
            break;
        }
        if (ctb_addr_ts as usize) >= pic_size_in_ctbs {
            if tolerant {
                eprintln!("(tolerant: end_of_slice_segment_flag not set on the last CTB)");
                break;
            }
            return Err(SequenceError::Malformed(
                "end_of_slice_segment_flag not set on the last CTB",
            ));
        }
        // §7.3.8.1 — end_of_subset_one_bit + byte_alignment( ) when
        // the NEXT CTB (CtbAddrInTs already incremented) starts a new
        // tile, or (WPP) a new CTU row of a tile; the next CTU reads
        // from the following substream.
        let next_rs = tiling.ctb_addr_ts_to_rs(ctb_addr_ts);
        let tile_boundary =
            tiles_on && tiling.tile_id(ctb_addr_ts) != tiling.tile_id(ctb_addr_ts - 1);
        let wpp_boundary = wpp
            && (next_rs % geom.pic_w_ctbs == 0
                || tiling.tile_id(ctb_addr_ts)
                    != tiling.tile_id(tiling.ctb_addr_rs_to_ts(next_rs - 1)));
        if tile_boundary || wpp_boundary {
            let one = end_of_slice_segment_flag(&mut engine)
                .map_err(|_| SequenceError::Malformed("CABAC underrun at end_of_subset_one_bit"))?;
            if !one && !tolerant {
                return Err(SequenceError::Malformed("end_of_subset_one_bit not set"));
            }
            advance_substream = true;
        }
    }
    Ok(())
}

/// §7.4.7.1 — the stripped-RBSP byte ranges of the
/// `num_entry_point_offsets + 1` subsets of `slice_segment_data( )`.
///
/// The wire offsets count coded (escaped) bytes from the first byte of
/// the slice segment data, so walk the escaped payload with the
/// §7.4.1.1 emulation state machine and translate each boundary into
/// the stripped-RBSP index space.
fn split_substreams(
    escaped: &[u8],
    rbsp_len: usize,
    stripped_data_offset: usize,
    entry_points: Option<&crate::hevc::engine::slice::EntryPointOffsets>,
) -> Result<Vec<(usize, usize)>, SequenceError> {
    let n_offsets = entry_points.map_or(0, |e| e.entry_point_offset_minus1.len());
    if n_offsets == 0 {
        return Ok(vec![(stripped_data_offset, rbsp_len)]);
    }
    // stripped index -> escaped index of the slice-data start.
    let mut stripped_of_escaped = vec![0usize; escaped.len() + 1];
    let mut zeros = 0u32;
    let mut stripped = 0usize;
    for (i, &b) in escaped.iter().enumerate() {
        stripped_of_escaped[i] = stripped;
        if zeros >= 2 && b == 0x03 {
            // Emulation-prevention byte: consumed, not emitted.
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        stripped += 1;
    }
    stripped_of_escaped[escaped.len()] = stripped;
    // Escaped index of the slice-data start.
    let escaped_start = stripped_of_escaped
        .iter()
        .position(|&sidx| sidx == stripped_data_offset)
        .ok_or(SequenceError::Malformed("slice data offset unmappable"))?;

    let entry_points = entry_points.expect("checked above");
    let mut ranges = Vec::with_capacity(n_offsets + 1);
    let mut first_escaped = escaped_start;
    let mut first_stripped = stripped_data_offset;
    for &off_m1 in &entry_points.entry_point_offset_minus1 {
        let len = off_m1 as usize + 1;
        let last_escaped = first_escaped
            .checked_add(len)
            .filter(|&e| e <= escaped.len())
            .ok_or(SequenceError::Malformed("entry point past slice data"))?;
        let last_stripped = stripped_of_escaped[last_escaped];
        ranges.push((first_stripped, last_stripped));
        first_escaped = last_escaped;
        first_stripped = last_stripped;
    }
    ranges.push((first_stripped, rbsp_len));
    Ok(ranges)
}
