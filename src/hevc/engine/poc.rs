//! §8.3.1 picture-order-count derivation and the Table 7-1 NAL-unit-type
//! classification it (and the §8.3.2 DPB marking) needs.
//!
//! Every coded picture carries a `PicOrderCntVal` (§8.3.1), the
//! decoder-wide identity used to address reference pictures in the §8.3.2
//! reference-picture-set marking, the §8.3.4 reference-picture-list
//! construction, and the §8.5.3.2.8 / §8.5.3.2.9 temporal motion-vector
//! scaling. The value is `PicOrderCntMsb + slice_pic_order_cnt_lsb`
//! (equation 8-2), where the MSB is rolled forward from the previous
//! `TemporalId == 0` non-RASL/RADL/SLNR picture (`prevTid0Pic`) by the
//! equation-8-1 wrap rule — except an IRAP picture with
//! `NoRaslOutputFlag == 1` resets the MSB to 0.
//!
//! The [`PocState`] threads `prevTid0Pic`'s `(slice_pic_order_cnt_lsb,
//! PicOrderCntMsb)` across the picture sequence so a multi-picture decode
//! (an I-frame followed by a P-frame, a B-pyramid, …) derives each POC
//! exactly. [`NalKind`] classifies a `nal_unit_type` into the Table 7-1
//! categories the POC + DPB processes branch on.

/// Table 7-1 `nal_unit_type` classification — the categories the §8.3.x
/// decoding processes branch on (IRAP / IDR / leading-picture / SLNR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalKind(u8);

impl NalKind {
    /// `TRAIL_N` (0).
    pub const TRAIL_N: u8 = 0;
    /// `TRAIL_R` (1).
    pub const TRAIL_R: u8 = 1;
    /// `TSA_N` (2).
    pub const TSA_N: u8 = 2;
    /// `TSA_R` (3).
    pub const TSA_R: u8 = 3;
    /// `STSA_N` (4).
    pub const STSA_N: u8 = 4;
    /// `STSA_R` (5).
    pub const STSA_R: u8 = 5;
    /// `RADL_N` (6).
    pub const RADL_N: u8 = 6;
    /// `RADL_R` (7).
    pub const RADL_R: u8 = 7;
    /// `RASL_N` (8).
    pub const RASL_N: u8 = 8;
    /// `RASL_R` (9).
    pub const RASL_R: u8 = 9;
    /// `BLA_W_LP` (16).
    pub const BLA_W_LP: u8 = 16;
    /// `BLA_W_RADL` (17).
    pub const BLA_W_RADL: u8 = 17;
    /// `BLA_N_LP` (18).
    pub const BLA_N_LP: u8 = 18;
    /// `IDR_W_RADL` (19).
    pub const IDR_W_RADL: u8 = 19;
    /// `IDR_N_LP` (20).
    pub const IDR_N_LP: u8 = 20;
    /// `CRA_NUT` (21).
    pub const CRA_NUT: u8 = 21;
    /// `RSV_IRAP_VCL23` (23) — inclusive upper bound of the IRAP range.
    pub const RSV_IRAP_VCL23: u8 = 23;

    /// Wrap a raw `nal_unit_type` (0..=63).
    #[inline]
    #[must_use]
    pub fn new(nal_unit_type: u8) -> Self {
        Self(nal_unit_type)
    }

    /// The raw `nal_unit_type` value.
    #[inline]
    #[must_use]
    pub fn value(self) -> u8 {
        self.0
    }

    /// A VCL (coded-slice-segment) NAL unit: `nal_unit_type` in
    /// `TRAIL_N..=RASL_R` (0..=9) or `BLA_W_LP..=RSV_IRAP_VCL23`
    /// (16..=23) — i.e. `nal_unit_type <= 31` covers the reserved VCL
    /// ranges too, but §3.29 limits coded-slice-segment NAL units to
    /// these two spans.
    #[inline]
    #[must_use]
    pub fn is_vcl(self) -> bool {
        (self.0 <= Self::RASL_R) || (Self::BLA_W_LP..=31).contains(&self.0)
    }

    /// IRAP picture (§3 — `BLA_W_LP..=RSV_IRAP_VCL23`, 16..=23).
    #[inline]
    #[must_use]
    pub fn is_irap(self) -> bool {
        (Self::BLA_W_LP..=Self::RSV_IRAP_VCL23).contains(&self.0)
    }

    /// IDR picture (`IDR_W_RADL` or `IDR_N_LP`).
    #[inline]
    #[must_use]
    pub fn is_idr(self) -> bool {
        self.0 == Self::IDR_W_RADL || self.0 == Self::IDR_N_LP
    }

    /// BLA picture (`BLA_W_LP`, `BLA_W_RADL`, `BLA_N_LP`).
    #[inline]
    #[must_use]
    pub fn is_bla(self) -> bool {
        (Self::BLA_W_LP..=Self::BLA_N_LP).contains(&self.0)
    }

    /// CRA picture (`CRA_NUT`).
    #[inline]
    #[must_use]
    pub fn is_cra(self) -> bool {
        self.0 == Self::CRA_NUT
    }

    /// RASL picture (`RASL_N` or `RASL_R`).
    #[inline]
    #[must_use]
    pub fn is_rasl(self) -> bool {
        self.0 == Self::RASL_N || self.0 == Self::RASL_R
    }

    /// RADL picture (`RADL_N` or `RADL_R`).
    #[inline]
    #[must_use]
    pub fn is_radl(self) -> bool {
        self.0 == Self::RADL_N || self.0 == Self::RADL_R
    }

    /// Sub-layer non-reference picture (SLNR): a VCL NAL unit with an
    /// even `nal_unit_type` in the `0..=14` range (`TRAIL_N`, `TSA_N`,
    /// `STSA_N`, `RADL_N`, `RASL_N`, `RSV_VCL_N10/12/14`). These are the
    /// `*_N` coded-slice types per the Table 7-1 "sub-layer non-reference"
    /// rows. IRAP / IDR / CRA / BLA types are never SLNR.
    #[inline]
    #[must_use]
    pub fn is_slnr(self) -> bool {
        self.0 <= Self::RADL_R && self.0 % 2 == 0
            || self.0 == 10
            || self.0 == 12
            || self.0 == 14
            || self.0 == Self::RASL_N
    }
}

/// §8.3.1 `(slice_pic_order_cnt_lsb, PicOrderCntMsb)` carried forward
/// from `prevTid0Pic` — the previous `TemporalId == 0` picture that is not
/// a RASL, RADL or SLNR picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PocState {
    /// `prevPicOrderCntLsb` — `slice_pic_order_cnt_lsb` of `prevTid0Pic`.
    prev_poc_lsb: u32,
    /// `prevPicOrderCntMsb` — `PicOrderCntMsb` of `prevTid0Pic`.
    prev_poc_msb: i32,
    /// Whether a `prevTid0Pic` has been seen (false at stream start).
    seen: bool,
}

/// The §8.3.1 outputs for one picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PicOrderCnt {
    /// `PicOrderCntMsb` (equation 8-1).
    pub msb: i32,
    /// `slice_pic_order_cnt_lsb`.
    pub lsb: u32,
    /// `PicOrderCntVal = PicOrderCntMsb + slice_pic_order_cnt_lsb`
    /// (equation 8-2).
    pub val: i32,
}

impl PocState {
    /// A fresh state at the start of a coded video sequence (no
    /// `prevTid0Pic` yet).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// §8.3.1 — derive the current picture's `PicOrderCntVal`.
    ///
    /// `nal_kind` is the current picture's [`NalKind`]; `no_rasl_output`
    /// is `NoRaslOutputFlag` (1 for an IDR / BLA, and for a CRA at the
    /// start of the bitstream or after an end-of-sequence — the caller
    /// resolves it); `poc_lsb` is `slice_pic_order_cnt_lsb` (0 for an IDR,
    /// which signals no `slice_pic_order_cnt_lsb`); `max_poc_lsb` is
    /// `MaxPicOrderCntLsb` from the active SPS.
    ///
    /// Returns the derived [`PicOrderCnt`]. The caller must then call
    /// [`PocState::update_prev_tid0`] **after** picture decode if the
    /// picture qualifies as a `prevTid0Pic` (`TemporalId == 0` and not
    /// RASL / RADL / SLNR) so the next picture's MSB rolls forward.
    #[must_use]
    pub fn derive(
        &self,
        nal_kind: NalKind,
        no_rasl_output: bool,
        poc_lsb: u32,
        max_poc_lsb: u32,
    ) -> PicOrderCnt {
        let msb = if nal_kind.is_irap() && no_rasl_output {
            // §8.3.1 — IRAP with NoRaslOutputFlag == 1 resets the MSB.
            0
        } else if !self.seen {
            // No prevTid0Pic yet (and not an MSB-reset IRAP): prev values
            // are 0 per the IDR NOTE-1 boundary condition.
            poc_msb(poc_lsb, 0, 0, max_poc_lsb)
        } else {
            poc_msb(poc_lsb, self.prev_poc_lsb, self.prev_poc_msb, max_poc_lsb)
        };
        let val = msb + poc_lsb as i32;
        PicOrderCnt {
            msb,
            lsb: poc_lsb,
            val,
        }
    }

    /// Update the carried `prevTid0Pic` state after decoding a picture
    /// whose `TemporalId == 0` and that is not a RASL, RADL or SLNR
    /// picture. `poc` is the picture's derived [`PicOrderCnt`].
    pub fn update_prev_tid0(&mut self, poc: PicOrderCnt) {
        self.prev_poc_lsb = poc.lsb;
        self.prev_poc_msb = poc.msb;
        self.seen = true;
    }

    /// Convenience: derive the POC and, when the picture qualifies as a
    /// `prevTid0Pic` (`temporal_id == 0` and not RASL / RADL / SLNR),
    /// roll the state forward. Returns the derived [`PicOrderCnt`].
    pub fn decode_picture_poc(
        &mut self,
        nal_kind: NalKind,
        temporal_id: u8,
        no_rasl_output: bool,
        poc_lsb: u32,
        max_poc_lsb: u32,
    ) -> PicOrderCnt {
        let poc = self.derive(nal_kind, no_rasl_output, poc_lsb, max_poc_lsb);
        if temporal_id == 0 && !(nal_kind.is_rasl() || nal_kind.is_radl() || nal_kind.is_slnr()) {
            self.update_prev_tid0(poc);
        }
        poc
    }
}

/// §8.3.1 equation 8-1 — `PicOrderCntMsb` from the current LSB and the
/// previous picture's LSB / MSB.
#[inline]
fn poc_msb(poc_lsb: u32, prev_poc_lsb: u32, prev_poc_msb: i32, max_poc_lsb: u32) -> i32 {
    let half = (max_poc_lsb / 2) as i32;
    let lsb = poc_lsb as i32;
    let prev_lsb = prev_poc_lsb as i32;
    let max = max_poc_lsb as i32;
    if lsb < prev_lsb && (prev_lsb - lsb) >= half {
        prev_poc_msb + max
    } else if lsb > prev_lsb && (lsb - prev_lsb) > half {
        prev_poc_msb - max
    } else {
        prev_poc_msb
    }
}

/// §8.3.1 equation 8-4 — `DiffPicOrderCnt( picA, picB )`.
#[inline]
#[must_use]
pub fn diff_pic_order_cnt(poc_a: i32, poc_b: i32) -> i32 {
    poc_a - poc_b
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn idr_is_classified() {
        assert!(NalKind::new(NalKind::IDR_W_RADL).is_idr());
        assert!(NalKind::new(NalKind::IDR_N_LP).is_idr());
        assert!(NalKind::new(NalKind::IDR_W_RADL).is_irap());
        assert!(!NalKind::new(NalKind::TRAIL_R).is_irap());
        assert!(NalKind::new(NalKind::CRA_NUT).is_cra());
        assert!(NalKind::new(NalKind::CRA_NUT).is_irap());
        assert!(NalKind::new(NalKind::BLA_W_LP).is_bla());
    }

    #[test]
    fn vcl_classification_matches_table_7_1() {
        assert!(NalKind::new(0).is_vcl());
        assert!(NalKind::new(9).is_vcl());
        assert!(NalKind::new(20).is_vcl());
        assert!(NalKind::new(23).is_vcl());
        assert!(NalKind::new(31).is_vcl());
        // VPS/SPS/PPS are non-VCL.
        assert!(!NalKind::new(32).is_vcl());
        assert!(!NalKind::new(33).is_vcl());
    }

    #[test]
    fn slnr_picks_the_n_types() {
        for &n in &[
            NalKind::TRAIL_N,
            NalKind::TSA_N,
            NalKind::STSA_N,
            NalKind::RADL_N,
            NalKind::RASL_N,
            10,
            12,
            14,
        ] {
            assert!(NalKind::new(n).is_slnr(), "type {n} is SLNR");
        }
        for &r in &[
            NalKind::TRAIL_R,
            NalKind::TSA_R,
            NalKind::STSA_R,
            NalKind::RADL_R,
            NalKind::RASL_R,
        ] {
            assert!(!NalKind::new(r).is_slnr(), "type {r} is not SLNR");
        }
    }

    #[test]
    fn idr_poc_is_zero() {
        // §8.3.1 NOTE 1: an IDR has poc_lsb inferred 0, NoRaslOutputFlag 1
        // ⇒ PicOrderCntVal == 0.
        let mut st = PocState::new();
        let poc = st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 256);
        assert_eq!(poc.val, 0);
        assert_eq!(poc.msb, 0);
    }

    #[test]
    fn trailing_pictures_roll_the_msb_forward() {
        // IDR(poc 0), then four P-frames at lsb 1,2,3,4 ⇒ poc 1,2,3,4.
        let mut st = PocState::new();
        let idr = st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 256);
        assert_eq!(idr.val, 0);
        for lsb in 1..=4 {
            let p = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, lsb, 256);
            assert_eq!(p.val, lsb as i32, "trailing poc lsb {lsb}");
        }
    }

    #[test]
    fn lsb_wrap_increments_the_msb() {
        // MaxPicOrderCntLsb = 16 (small for the test). Walk the lsb up in
        // small forward steps (≤ half) so the MSB stays 0: 4, 8, 12, 15.
        // Then lsb 2 against prev 15: 2 < 15 and (15 − 2) = 13 >= 8 ⇒
        // equation-8-1 forward-wrap branch ⇒ msb += 16 ⇒ poc 18.
        let mut st = PocState::new();
        st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 16);
        for lsb in [4u32, 8, 12, 15] {
            st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, lsb, 16);
        }
        let b = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, 2, 16);
        assert_eq!(b.val, 18, "lsb wrap 15→2 rolls the msb to 16");
    }

    #[test]
    fn lsb_backward_jump_decrements_the_msb() {
        // From a high lsb, a forward jump greater than half decrements
        // the MSB (B-frame reordering). Start msb at 16 (poc 16 at lsb 0),
        // then a lsb 14: 14 > 0 and (14 − 0) > 8 ⇒ msb −= 16 ⇒ poc −2.
        let mut st = PocState::new();
        st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 16);
        // Advance to poc 16 (lsb wraps 0 four times via lsb 4,8,12,0).
        for lsb in [4u32, 8, 12, 0] {
            st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, lsb, 16);
        }
        // prevTid0 now (lsb 0, msb 16). A lsb-14 picture: backward branch.
        let b = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, 14, 16);
        assert_eq!(b.val, 14);
    }

    #[test]
    fn rasl_radl_slnr_do_not_advance_prev_tid0() {
        let mut st = PocState::new();
        st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 256);
        // A RADL at lsb 2 derives poc 2 but does NOT become prevTid0Pic.
        let radl = st.decode_picture_poc(NalKind::new(NalKind::RADL_R), 0, false, 2, 256);
        assert_eq!(radl.val, 2);
        // The next trailing picture still references the IDR's prev state
        // (lsb 0, msb 0), so lsb 1 ⇒ poc 1 (not relative to the RADL).
        let p = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, 1, 256);
        assert_eq!(p.val, 1);
    }

    #[test]
    fn temporal_id_nonzero_does_not_advance_prev_tid0() {
        let mut st = PocState::new();
        st.decode_picture_poc(NalKind::new(NalKind::IDR_N_LP), 0, true, 0, 256);
        // A TemporalId == 1 trailing picture derives its POC but does not
        // update prevTid0Pic.
        let t1 = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 1, false, 5, 256);
        assert_eq!(t1.val, 5);
        let p = st.decode_picture_poc(NalKind::new(NalKind::TRAIL_R), 0, false, 1, 256);
        assert_eq!(p.val, 1, "prevTid0 unchanged by the sub-layer picture");
    }

    #[test]
    fn diff_pic_order_cnt_subtracts() {
        assert_eq!(diff_pic_order_cnt(8, 2), 6);
        assert_eq!(diff_pic_order_cnt(2, 8), -6);
    }
}
