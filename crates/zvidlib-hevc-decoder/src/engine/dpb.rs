//! §8.3.2 reference-picture-set marking, §8.3.4 reference-picture-list
//! construction, §8.3.5 collocated-picture selection, and the
//! decoded-picture-buffer (DPB) storage that ties them together.
//!
//! A [`Dpb`] holds the previously-decoded pictures available for
//! inter prediction of the current and future pictures. Each
//! [`DpbEntry`] carries:
//!
//! * the reconstructed [`crate::hevc::engine::picture::Picture`] samples
//!   (`refPicLXL` / `refPicLXCb` / `refPicLXCr` of §8.5.3.3.2),
//! * its `PicOrderCntVal` (the §8.3.1 identity used to address it),
//! * its short/long-term reference [`Marking`],
//! * its per-PU [`crate::hevc::engine::motion::MotionField`] (the `MvLXCol` /
//!   `PredFlagLXCol` / `RefIdxLXCol`-equivalent arrays the §8.5.3.2.9
//!   collocated-MV derivation reads).
//!
//! The picture-decode driver, after deriving the current picture's POC
//! (§8.3.1), calls [`Dpb::apply_rps`] to build the five §8.3.2 RPS lists
//! (`RefPicSetStCurrBefore` … `RefPicSetLtFoll`) and mark every DPB
//! picture accordingly. For a P / B slice it then calls
//! [`Dpb::build_ref_pic_lists`] (§8.3.4) to construct `RefPicList0` /
//! `RefPicList1`, and [`select_col_pic`] (§8.3.5) to pick the collocated
//! picture for temporal MV prediction.

use crate::hevc::engine::motion::MotionField;
use crate::hevc::engine::picture::Picture;
use crate::hevc::engine::poc::diff_pic_order_cnt;
use crate::hevc::engine::sps::MaterializedShortTermRefPicSet;

/// A decoded picture's reference marking (§8.3.2). Exactly one of these
/// applies at any moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marking {
    /// "unused for reference".
    Unused,
    /// "used for short-term reference".
    ShortTerm,
    /// "used for long-term reference".
    LongTerm,
}

/// One decoded picture stored in the [`Dpb`].
#[derive(Debug)]
pub struct DpbEntry {
    /// `PicOrderCntVal` (§8.3.1).
    pub poc: i32,
    /// `nuh_layer_id` of the picture.
    pub layer_id: u8,
    /// The current reference [`Marking`].
    pub marking: Marking,
    /// The reconstructed sample planes (`SL` / `SCb` / `SCr`).
    pub picture: Picture,
    /// The per-PU motion field — the §8.5.3.2.9 collocated arrays
    /// (`PredFlagLXCol`, `MvLXCol`, the per-cell reference-picture POC
    /// identity standing in for `RefIdxLXCol` → `refPicListCol`).
    pub motion: MotionField,
}

impl DpbEntry {
    /// `( PicOrderCntVal & ( MaxPicOrderCntLsb − 1 ) )` — the LSB-only
    /// identity used for the §8.3.2 short-term-vs-long-term lookup of a
    /// long-term entry whose `delta_poc_msb_present_flag` is 0.
    #[inline]
    #[must_use]
    fn poc_lsb(&self, max_poc_lsb: u32) -> u32 {
        (self.poc as u32) & (max_poc_lsb - 1)
    }
}

/// The five §8.3.2 RPS POC lists (equation 8-5).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RpsPocLists {
    /// `PocStCurrBefore`.
    pub st_curr_before: Vec<i32>,
    /// `PocStCurrAfter`.
    pub st_curr_after: Vec<i32>,
    /// `PocStFoll`.
    pub st_foll: Vec<i32>,
    /// `PocLtCurr`.
    pub lt_curr: Vec<i32>,
    /// `PocLtFoll`.
    pub lt_foll: Vec<i32>,
    /// `CurrDeltaPocMsbPresentFlag[i]` (paired with `lt_curr`).
    pub curr_delta_poc_msb_present: Vec<bool>,
    /// `FollDeltaPocMsbPresentFlag[i]` (paired with `lt_foll`).
    pub foll_delta_poc_msb_present: Vec<bool>,
}

/// One long-term RPS input entry — the §7.4.7.1 / §8.3.2 long-term
/// reference, already resolved to `PocLsbLt[i]` plus the MSB-cycle block.
#[derive(Debug, Clone, Copy)]
pub struct LongTermEntry {
    /// `PocLsbLt[i]`.
    pub poc_lsb_lt: u32,
    /// `UsedByCurrPicLt[i]`.
    pub used_by_curr_pic_lt: bool,
    /// `delta_poc_msb_present_flag[i]`.
    pub delta_poc_msb_present: bool,
    /// `DeltaPocMsbCycleLt[i]` (the §7.4.7.1 accumulated cycle).
    pub delta_poc_msb_cycle_lt: u32,
}

/// Build the five §8.3.2 POC lists (equation 8-5) for the current
/// picture from its current short-term RPS + long-term entries.
///
/// `is_idr` short-circuits all five lists to empty (the §8.3.2 IDR rule).
/// `poc` is the current `PicOrderCntVal`; `max_poc_lsb` is
/// `MaxPicOrderCntLsb`.
#[must_use]
pub fn build_rps_poc_lists(
    is_idr: bool,
    poc: i32,
    max_poc_lsb: u32,
    st_rps: &MaterializedShortTermRefPicSet,
    lt_entries: &[LongTermEntry],
) -> RpsPocLists {
    let mut out = RpsPocLists::default();
    if is_idr {
        return out;
    }
    // Negative (S0) pics — equation 8-5 first loop.
    for (i, &delta) in st_rps.delta_poc_s0.iter().enumerate() {
        let p = poc + delta;
        if st_rps.used_by_curr_pic_s0[i] {
            out.st_curr_before.push(p);
        } else {
            out.st_foll.push(p);
        }
    }
    // Positive (S1) pics — equation 8-5 second loop.
    for (i, &delta) in st_rps.delta_poc_s1.iter().enumerate() {
        let p = poc + delta;
        if st_rps.used_by_curr_pic_s1[i] {
            out.st_curr_after.push(p);
        } else {
            out.st_foll.push(p);
        }
    }
    // Long-term — equation 8-5 third loop.
    for e in lt_entries {
        let mut poc_lt = e.poc_lsb_lt as i32;
        if e.delta_poc_msb_present {
            poc_lt += poc
                - (e.delta_poc_msb_cycle_lt as i32) * (max_poc_lsb as i32)
                - (poc & (max_poc_lsb as i32 - 1));
        }
        if e.used_by_curr_pic_lt {
            out.lt_curr.push(poc_lt);
            out.curr_delta_poc_msb_present.push(e.delta_poc_msb_present);
        } else {
            out.lt_foll.push(poc_lt);
            out.foll_delta_poc_msb_present.push(e.delta_poc_msb_present);
        }
    }
    out
}

/// The decoded-picture buffer.
#[derive(Debug, Default)]
pub struct Dpb {
    entries: Vec<DpbEntry>,
}

impl Dpb {
    /// An empty DPB.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pictures currently stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the DPB holds no pictures.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the stored entries (output / bumping inspection).
    #[must_use]
    pub fn entries(&self) -> &[DpbEntry] {
        &self.entries
    }

    /// Insert a freshly-decoded picture, marked "used for short-term
    /// reference" per the §8.3.2 default. The picture-decode driver calls
    /// this after reconstruction + in-loop filtering, having already run
    /// [`Dpb::apply_rps`] for the picture's own RPS.
    pub fn insert(&mut self, entry: DpbEntry) {
        self.entries.push(entry);
    }

    /// Find a stored picture by exact `PicOrderCntVal` and `nuh_layer_id`.
    #[must_use]
    pub fn find_by_poc(&self, poc: i32, layer_id: u8) -> Option<&DpbEntry> {
        self.entries
            .iter()
            .find(|e| e.poc == poc && e.layer_id == layer_id)
    }

    /// Find a short-term reference picture by exact POC (§8.3.2 step 3).
    fn find_short_term(&self, poc: i32, layer_id: u8) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.poc == poc && e.layer_id == layer_id && e.marking == Marking::ShortTerm)
    }

    /// Find a long-term candidate by exact POC or by LSB-only POC
    /// (§8.3.2 step 1).
    fn find_long_term(
        &self,
        target: i32,
        msb_present: bool,
        max_poc_lsb: u32,
        layer_id: u8,
    ) -> Option<usize> {
        self.entries.iter().position(|e| {
            e.layer_id == layer_id
                && if msb_present {
                    e.poc == target
                } else {
                    e.poc_lsb(max_poc_lsb) == (target as u32) & (max_poc_lsb - 1)
                }
        })
    }

    /// §8.3.2 — apply the reference-picture-set marking for the current
    /// picture.
    ///
    /// `is_irap_no_rasl` is `true` for an IRAP picture with
    /// `NoRaslOutputFlag == 1` (all current references become "unused");
    /// `layer_id` is the current picture's `nuh_layer_id`; `lists` is the
    /// output of [`build_rps_poc_lists`]; `max_poc_lsb` is
    /// `MaxPicOrderCntLsb`.
    ///
    /// Returns the five resolved RPS reference-picture lists as index
    /// vectors into [`Dpb::entries`]; a `None` entry is the §8.3.2 "no
    /// reference picture". After this call every DPB picture carries its
    /// updated [`Marking`].
    pub fn apply_rps(
        &mut self,
        is_irap_no_rasl: bool,
        layer_id: u8,
        lists: &RpsPocLists,
        max_poc_lsb: u32,
    ) -> ResolvedRps {
        // §8.3.2 — an IRAP with NoRaslOutputFlag == 1 marks every current
        // reference of this layer "unused for reference" up front.
        if is_irap_no_rasl {
            for e in &mut self.entries {
                if e.layer_id == layer_id {
                    e.marking = Marking::Unused;
                }
            }
        }

        let mut resolved = ResolvedRps::default();

        // Step 1 — resolve the long-term RPS sets (LSB or full POC).
        for (i, &poc_lt) in lists.lt_curr.iter().enumerate() {
            resolved.lt_curr.push(self.find_long_term(
                poc_lt,
                lists.curr_delta_poc_msb_present[i],
                max_poc_lsb,
                layer_id,
            ));
        }
        for (i, &poc_lt) in lists.lt_foll.iter().enumerate() {
            resolved.lt_foll.push(self.find_long_term(
                poc_lt,
                lists.foll_delta_poc_msb_present[i],
                max_poc_lsb,
                layer_id,
            ));
        }

        // Step 2 — mark the long-term picks "used for long-term reference".
        for idx in resolved
            .lt_curr
            .iter()
            .chain(resolved.lt_foll.iter())
            .flatten()
        {
            self.entries[*idx].marking = Marking::LongTerm;
        }

        // Step 3 — resolve the short-term RPS sets (exact POC).
        for &poc in &lists.st_curr_before {
            resolved
                .st_curr_before
                .push(self.find_short_term(poc, layer_id));
        }
        for &poc in &lists.st_curr_after {
            resolved
                .st_curr_after
                .push(self.find_short_term(poc, layer_id));
        }
        for &poc in &lists.st_foll {
            resolved.st_foll.push(self.find_short_term(poc, layer_id));
        }

        // Step 4 — any picture of this layer not in any RPS set is
        // "unused for reference".
        let mut keep = vec![false; self.entries.len()];
        for idx in resolved
            .lt_curr
            .iter()
            .chain(resolved.lt_foll.iter())
            .chain(resolved.st_curr_before.iter())
            .chain(resolved.st_curr_after.iter())
            .chain(resolved.st_foll.iter())
            .flatten()
        {
            keep[*idx] = true;
        }
        for (i, e) in self.entries.iter_mut().enumerate() {
            if e.layer_id == layer_id && !keep[i] {
                e.marking = Marking::Unused;
            }
        }

        resolved
    }

    /// §8.3.4 — construct `RefPicList0` (and, for a B slice,
    /// `RefPicList1`) as index vectors into [`Dpb::entries`].
    ///
    /// `rps` is the [`apply_rps`](Dpb::apply_rps) output; `params` carries
    /// the §8.3.4 sizing + modification inputs. With
    /// `curr_pic_ref_enabled` (`pps_curr_pic_ref_enabled_flag`) the
    /// equation 8-8 / 8-10 wrap appends the current picture — the
    /// [`CURR_PIC`] sentinel — after `RefPicSetLtCurr` on every pass,
    /// and the equation 8-9 closing clause overrides the last active
    /// `RefPicList0` slot when the temp list was longer than the
    /// active count and no explicit modification was signalled.
    #[must_use]
    pub fn build_ref_pic_lists(
        &self,
        rps: &ResolvedRps,
        params: &RefPicListParams<'_>,
    ) -> RefPicLists {
        // RefPicListTemp0 (equation 8-8): StCurrBefore, StCurrAfter, LtCurr,
        // (+ currPic with pps_curr_pic_ref_enabled_flag), repeated until
        // NumRpsCurrTempList0 entries are produced.
        let active0 = (params.num_ref_idx_l0_active_minus1 + 1) as usize;
        let num_temp0 = active0.max(params.num_pic_total_curr as usize);
        let temp0 = build_temp_list(
            num_temp0,
            &[&rps.st_curr_before, &rps.st_curr_after, &rps.lt_curr],
            params.curr_pic_ref_enabled,
        );
        let mut list0 = apply_list_modification(&temp0, active0, params.list_entry_l0);
        // Equation 8-9 closing clause.
        if params.curr_pic_ref_enabled && params.list_entry_l0.is_none() && num_temp0 > active0 {
            if let Some(last) = list0.last_mut() {
                *last = Some(CURR_PIC);
            }
        }

        let list1 = if params.is_b {
            // RefPicListTemp1 (equation 8-10): StCurrAfter, StCurrBefore,
            // LtCurr (note the swapped first two sets).
            let active1 = (params.num_ref_idx_l1_active_minus1 + 1) as usize;
            let num_temp1 = active1.max(params.num_pic_total_curr as usize);
            let temp1 = build_temp_list(
                num_temp1,
                &[&rps.st_curr_after, &rps.st_curr_before, &rps.lt_curr],
                params.curr_pic_ref_enabled,
            );
            Some(apply_list_modification(
                &temp1,
                active1,
                params.list_entry_l1,
            ))
        } else {
            None
        };

        RefPicLists { list0, list1 }
    }
}

/// The [`RefPicLists`] sentinel index standing for the CURRENT decoded
/// picture (before in-loop filtering) — the §8.3.4 `currPic` appended
/// when `pps_curr_pic_ref_enabled_flag` is 1. It is deliberately out of
/// range for every real DPB entry vector; consumers must test for it
/// before indexing.
pub const CURR_PIC: usize = usize::MAX;

/// §8.3.4 sizing + modification inputs for
/// [`Dpb::build_ref_pic_lists`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RefPicListParams<'a> {
    /// `num_ref_idx_l0_active_minus1`.
    pub num_ref_idx_l0_active_minus1: u32,
    /// `num_ref_idx_l1_active_minus1`.
    pub num_ref_idx_l1_active_minus1: u32,
    /// `NumPicTotalCurr` (§7.4.7.2).
    pub num_pic_total_curr: u32,
    /// `true` for a B slice (builds `RefPicList1`).
    pub is_b: bool,
    /// `list_entry_l0[]` (§7.3.6.2) — `Some` reorders `RefPicListTemp0`.
    pub list_entry_l0: Option<&'a [u32]>,
    /// `list_entry_l1[]` (§7.3.6.2) — `Some` reorders `RefPicListTemp1`.
    pub list_entry_l1: Option<&'a [u32]>,
    /// `pps_curr_pic_ref_enabled_flag` — appends the [`CURR_PIC`]
    /// sentinel per equations 8-8 / 8-9 / 8-10.
    pub curr_pic_ref_enabled: bool,
}

/// The §8.3.2 RPS resolved to DPB entry indices. A `None` is the spec's
/// "no reference picture".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedRps {
    /// `RefPicSetStCurrBefore`.
    pub st_curr_before: Vec<Option<usize>>,
    /// `RefPicSetStCurrAfter`.
    pub st_curr_after: Vec<Option<usize>>,
    /// `RefPicSetStFoll`.
    pub st_foll: Vec<Option<usize>>,
    /// `RefPicSetLtCurr`.
    pub lt_curr: Vec<Option<usize>>,
    /// `RefPicSetLtFoll`.
    pub lt_foll: Vec<Option<usize>>,
}

/// The §8.3.4 reference picture lists, as DPB entry indices. A `None`
/// entry is a "no reference picture" carried through from the RPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefPicLists {
    /// `RefPicList0`.
    pub list0: Vec<Option<usize>>,
    /// `RefPicList1` (only for B slices).
    pub list1: Option<Vec<Option<usize>>>,
}

/// §8.3.5 — select the collocated picture `ColPic` for temporal MV
/// prediction, returning its DPB entry index.
///
/// `slice_type_is_b` + `collocated_from_l0_flag` choose the list per
/// §8.3.5; `collocated_ref_idx` indexes it. Returns `None` when the
/// chosen entry is "no reference picture" or the index is out of range.
#[must_use]
pub fn select_col_pic(
    lists: &RefPicLists,
    slice_type_is_b: bool,
    collocated_from_l0_flag: bool,
    collocated_ref_idx: u32,
) -> Option<usize> {
    let idx = collocated_ref_idx as usize;
    let picked = if slice_type_is_b && !collocated_from_l0_flag {
        lists.list1.as_ref()?.get(idx).copied().flatten()
    } else {
        lists.list0.get(idx).copied().flatten()
    };
    // §7.4.7.1: collocated_ref_idx shall not refer to the current
    // picture — treat a malformed selection as "no collocated picture".
    picked.filter(|&i| i != CURR_PIC)
}

/// §8.3.5 — `NoBackwardPredFlag`: true iff every reference picture in
/// `RefPicList0` / `RefPicList1` has `DiffPicOrderCnt( aPic, CurrPic )
/// <= 0`. `curr_poc` is the current picture's `PicOrderCntVal`; the
/// closure resolves a DPB index to its POC.
#[must_use]
pub fn no_backward_pred_flag(dpb: &Dpb, lists: &RefPicLists, curr_poc: i32) -> bool {
    let mut all_le = true;
    let check = |idx: &Option<usize>, all_le: &mut bool| {
        if let Some(i) = idx {
            // The current picture (CURR_PIC) has DiffPicOrderCnt == 0.
            if *i != CURR_PIC && diff_pic_order_cnt(dpb.entries[*i].poc, curr_poc) > 0 {
                *all_le = false;
            }
        }
    };
    for idx in &lists.list0 {
        check(idx, &mut all_le);
    }
    if let Some(l1) = &lists.list1 {
        for idx in l1 {
            check(idx, &mut all_le);
        }
    }
    all_le
}

/// Build a §8.3.4 `RefPicListTempX` by concatenating the given RPS sets
/// in order — appending the [`CURR_PIC`] sentinel after each pass when
/// `curr_pic_ref_enabled` (`pps_curr_pic_ref_enabled_flag`) — and
/// wrapping around until `len` entries are produced (equations
/// 8-8 / 8-10).
fn build_temp_list(
    len: usize,
    sets: &[&[Option<usize>]],
    curr_pic_ref_enabled: bool,
) -> Vec<Option<usize>> {
    let mut temp = Vec::with_capacity(len);
    if sets.iter().all(|s| s.is_empty()) && !curr_pic_ref_enabled {
        return temp;
    }
    while temp.len() < len {
        for set in sets {
            for &e in *set {
                if temp.len() >= len {
                    break;
                }
                temp.push(e);
            }
        }
        if curr_pic_ref_enabled {
            // The equation 8-8 / 8-10 currPic append carries no
            // `rIdx < NumRpsCurrTempListX` guard — the temp list may
            // end one entry longer than `len` (reachable through
            // `list_entry_lX`).
            temp.push(Some(CURR_PIC));
        }
    }
    temp
}

/// Apply the §8.3.4 `RefPicListX` selection (equation 8-9 / 8-11): take
/// the first `active` entries of `temp`, reordered through `list_entry`
/// when present.
fn apply_list_modification(
    temp: &[Option<usize>],
    active: usize,
    list_entry: Option<&[u32]>,
) -> Vec<Option<usize>> {
    (0..active)
        .map(|r_idx| match list_entry {
            Some(le) => temp.get(le[r_idx] as usize).copied().flatten(),
            None => temp.get(r_idx).copied().flatten(),
        })
        .collect()
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::picture::Picture;

    fn entry(poc: i32, marking: Marking) -> DpbEntry {
        DpbEntry {
            poc,
            layer_id: 0,
            marking,
            picture: Picture::new(16, 16, 1, 8, 8),
            motion: MotionField::new(16, 16),
        }
    }

    fn empty_st_rps() -> MaterializedShortTermRefPicSet {
        MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![],
            used_by_curr_pic_s0: vec![],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        }
    }

    #[test]
    fn idr_rps_lists_are_empty() {
        let st = empty_st_rps();
        let lists = build_rps_poc_lists(true, 0, 256, &st, &[]);
        assert!(lists.st_curr_before.is_empty());
        assert!(lists.st_curr_after.is_empty());
        assert!(lists.lt_curr.is_empty());
    }

    #[test]
    fn p_frame_short_term_before_resolves_idr() {
        // One negative pic at delta −1, used by curr ⇒ PocStCurrBefore
        // = [poc 1 + (−1)] = [0], which resolves to the IDR in the DPB.
        let st = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        let lists = build_rps_poc_lists(false, 1, 256, &st, &[]);
        assert_eq!(lists.st_curr_before, vec![0]);

        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        assert_eq!(rps.st_curr_before, vec![Some(0)]);
        // The IDR stays short-term (it is in the RPS).
        assert_eq!(dpb.entries[0].marking, Marking::ShortTerm);
    }

    #[test]
    fn picture_not_in_rps_becomes_unused() {
        let st = empty_st_rps();
        let lists = build_rps_poc_lists(false, 5, 256, &st, &[]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        dpb.apply_rps(false, 0, &lists, 256);
        // poc 0 is in no RPS list ⇒ unused for reference.
        assert_eq!(dpb.entries[0].marking, Marking::Unused);
    }

    #[test]
    fn irap_no_rasl_unmarks_all_references() {
        let st = empty_st_rps();
        let lists = build_rps_poc_lists(true, 0, 256, &st, &[]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        dpb.insert(entry(2, Marking::LongTerm));
        dpb.apply_rps(true, 0, &lists, 256);
        assert!(dpb.entries.iter().all(|e| e.marking == Marking::Unused));
    }

    #[test]
    fn p_ref_pic_list0_takes_curr_before() {
        // DPB holds POC 0 (IDR). Current P-frame POC 1 with one
        // short-term-before pic at POC 0.
        let st = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        let lists = build_rps_poc_lists(false, 1, 256, &st, &[]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        let rpl = dpb.build_ref_pic_lists(
            &rps,
            &RefPicListParams {
                num_pic_total_curr: 1,
                ..Default::default()
            },
        );
        assert_eq!(rpl.list0, vec![Some(0)]);
        assert!(rpl.list1.is_none());
    }

    /// §8.3.4 with `pps_curr_pic_ref_enabled_flag`: the current picture
    /// (CURR_PIC) is appended after the RPS sets (eq. 8-8) — a P slice
    /// with one short-term reference and two active entries lists
    /// [ref, currPic].
    #[test]
    fn curr_pic_ref_appends_current_picture() {
        let st = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        let lists = build_rps_poc_lists(false, 1, 256, &st, &[]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        let rpl = dpb.build_ref_pic_lists(
            &rps,
            &RefPicListParams {
                num_ref_idx_l0_active_minus1: 1,
                num_pic_total_curr: 2,
                curr_pic_ref_enabled: true,
                ..Default::default()
            },
        );
        assert_eq!(rpl.list0, vec![Some(0), Some(CURR_PIC)]);
    }

    /// An IRAP-only stream (empty RPS) with current-picture referencing
    /// still builds a list: [currPic] (the intra-block-copy-only case).
    #[test]
    fn curr_pic_ref_fills_empty_rps_list() {
        let dpb = Dpb::new();
        let rps = ResolvedRps::default();
        let rpl = dpb.build_ref_pic_lists(
            &rps,
            &RefPicListParams {
                num_ref_idx_l0_active_minus1: 0,
                num_pic_total_curr: 1,
                curr_pic_ref_enabled: true,
                ..Default::default()
            },
        );
        assert_eq!(rpl.list0, vec![Some(CURR_PIC)]);
    }

    /// The eq. 8-9 closing clause: with the temp list longer than the
    /// active count and no explicit modification, the LAST active slot
    /// is overridden to currPic — one short-term ref, one active entry,
    /// NumPicTotalCurr 2 ⇒ list0 = [currPic].
    #[test]
    fn curr_pic_ref_overrides_last_active_slot() {
        let st = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        };
        let lists = build_rps_poc_lists(false, 1, 256, &st, &[]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm));
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        let rpl = dpb.build_ref_pic_lists(
            &rps,
            &RefPicListParams {
                num_ref_idx_l0_active_minus1: 0,
                num_pic_total_curr: 2,
                curr_pic_ref_enabled: true,
                ..Default::default()
            },
        );
        assert_eq!(rpl.list0, vec![Some(CURR_PIC)]);
    }

    /// select_col_pic never yields the CURR_PIC sentinel (§7.4.7.1
    /// forbids collocated_ref_idx referring to the current picture).
    #[test]
    fn col_pic_selection_rejects_curr_pic() {
        let lists = RefPicLists {
            list0: vec![Some(CURR_PIC)],
            list1: None,
        };
        assert_eq!(select_col_pic(&lists, false, true, 0), None);
    }

    /// no_backward_pred_flag treats currPic as DiffPicOrderCnt == 0
    /// (not a future reference).
    #[test]
    fn no_backward_pred_ignores_curr_pic() {
        let dpb = Dpb::new();
        let lists = RefPicLists {
            list0: vec![Some(CURR_PIC)],
            list1: None,
        };
        assert!(no_backward_pred_flag(&dpb, &lists, 5));
    }

    #[test]
    fn b_ref_pic_lists_swap_before_after() {
        // Current B POC 1, with a before-pic at POC 0 and an after-pic at
        // POC 2. RefPicList0 = [before, after]; RefPicList1 = [after,
        // before].
        let st = MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![1],
            used_by_curr_pic_s1: vec![true],
        };
        let lists = build_rps_poc_lists(false, 1, 256, &st, &[]);
        assert_eq!(lists.st_curr_before, vec![0]);
        assert_eq!(lists.st_curr_after, vec![2]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm)); // index 0 = POC 0
        dpb.insert(entry(2, Marking::ShortTerm)); // index 1 = POC 2
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        let rpl = dpb.build_ref_pic_lists(
            &rps,
            &RefPicListParams {
                num_ref_idx_l0_active_minus1: 1,
                num_ref_idx_l1_active_minus1: 1,
                num_pic_total_curr: 2,
                is_b: true,
                ..Default::default()
            },
        );
        assert_eq!(rpl.list0, vec![Some(0), Some(1)]);
        assert_eq!(rpl.list1, Some(vec![Some(1), Some(0)]));
    }

    #[test]
    fn col_pic_selection_picks_list0_for_p() {
        let lists = RefPicLists {
            list0: vec![Some(3)],
            list1: None,
        };
        // P slice ⇒ list0[collocated_ref_idx 0].
        assert_eq!(select_col_pic(&lists, false, true, 0), Some(3));
    }

    #[test]
    fn col_pic_selection_picks_list1_when_b_and_not_from_l0() {
        let lists = RefPicLists {
            list0: vec![Some(3)],
            list1: Some(vec![Some(7)]),
        };
        assert_eq!(select_col_pic(&lists, true, false, 0), Some(7));
        // collocated_from_l0_flag true falls back to list0.
        assert_eq!(select_col_pic(&lists, true, true, 0), Some(3));
    }

    #[test]
    fn no_backward_pred_flag_detects_future_ref() {
        let mut dpb = Dpb::new();
        dpb.insert(entry(0, Marking::ShortTerm)); // index 0 (past)
        dpb.insert(entry(4, Marking::ShortTerm)); // index 1 (future)
        // Current POC 2; list0 references the past pic only ⇒ no backward.
        let past_only = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        assert!(no_backward_pred_flag(&dpb, &past_only, 2));
        // list1 references the future pic ⇒ backward prediction present.
        let with_future = RefPicLists {
            list0: vec![Some(0)],
            list1: Some(vec![Some(1)]),
        };
        assert!(!no_backward_pred_flag(&dpb, &with_future, 2));
    }

    #[test]
    fn long_term_lsb_lookup_marks_long_term() {
        // A long-term entry referencing POC 100 by LSB only (msb not
        // present). MaxPicOrderCntLsb 256 ⇒ lsb 100. The DPB picture at
        // POC 100 is found by its LSB and marked long-term.
        let lt = LongTermEntry {
            poc_lsb_lt: 100,
            used_by_curr_pic_lt: true,
            delta_poc_msb_present: false,
            delta_poc_msb_cycle_lt: 0,
        };
        let st = empty_st_rps();
        let lists = build_rps_poc_lists(false, 300, 256, &st, &[lt]);
        assert_eq!(lists.lt_curr, vec![100]);
        let mut dpb = Dpb::new();
        dpb.insert(entry(100, Marking::ShortTerm));
        let rps = dpb.apply_rps(false, 0, &lists, 256);
        assert_eq!(rps.lt_curr, vec![Some(0)]);
        assert_eq!(dpb.entries[0].marking, Marking::LongTerm);
    }
}
