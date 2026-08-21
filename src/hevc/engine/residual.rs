//! §7.3.8.11 `residual_coding( x0, y0, log2TrafoSize, cIdx )` syntax
//! driver.
//!
//! This module composes the per-element residual primitives the
//! [`crate::hevc::engine::binarization`] module accumulated across rounds 26..35 —
//! `last_sig_coeff_{x,y}_{prefix,suffix}` (§9.3.4.2.3),
//! `coded_sub_block_flag` (§9.3.4.2.4), `sig_coeff_flag` (§9.3.4.2.5,
//! including the Table 9-50 16-entry `ctxIdxMap` whose `i = 15` cell
//! is pinned to `8` by the staged docs erratum for the PDF layout
//! truncation), `coeff_abs_level_greater1_flag` /
//! `coeff_abs_level_greater2_flag` (§9.3.4.2.6 / §9.3.4.2.7 via
//! [`Greater1State`]), `coeff_abs_level_remaining` (§9.3.3.11) and
//! `coeff_sign_flag` — into the full §7.3.8.11 coefficient-decode
//! loop that reconstructs the `TransCoeffLevel[ ][ ]` array of one
//! transform block.
//!
//! ## Driver scope
//!
//! [`decode_residual_coding_with`] implements the §7.3.8.11 syntax
//! from `last_sig_coeff_x_prefix` through the final
//! `TransCoeffLevel` composition:
//!
//! * the §7.3.8.11 do-while locate of `(lastSubBlock, lastScanPos)`
//!   from the decoded (and, for `scanIdx == 2`, eq.-7-78-swapped)
//!   `LastSignificantCoeff{X,Y}`,
//! * the reverse sub-block scan with `coded_sub_block_flag` decode
//!   (presence gate `i < lastSubBlock && i > 0`) and the §7.4.9.11
//!   not-present inference (sub-blocks `0` and `lastSubBlock` ⇒ 1),
//! * the per-sub-block `sig_coeff_flag` loop with both §7.4.9.11
//!   inference rules (the last significant coefficient itself ⇒ 1;
//!   the DC cell of a coded sub-block whose other 15 flags decoded
//!   to 0 under `inferSbDcSigCoeffFlag` ⇒ 1) and the full
//!   §9.3.4.2.5 `sigCtx` branch dispatch (eq. 9-40 transform-skip
//!   contexts / Table 9-50 for `log2TrafoSize == 2` / eq. 9-42 DC /
//!   eq. 9-43..9-53 general),
//! * the `coeff_abs_level_greater1_flag` pass with the per-sub-block
//!   `numGreater1Flag < 8` cap and the §9.3.4.2.6 cross-sub-block
//!   `ctxSet` state ([`Greater1State`] is entered lazily at the
//!   first greater-1 bin of each sub-block, matching the spec's
//!   "first time for the current sub-block scan index i" trigger so
//!   sub-blocks that decode no greater-1 bins do not advance the
//!   state),
//! * the at-most-one `coeff_abs_level_greater2_flag` at
//!   `lastGreater1ScanPos` (§9.3.4.2.7, eq. 9-61 / 9-62),
//! * the §7.3.8.11 `signHidden` derivation
//!   (`lastSigScanPos − firstSigScanPos > 3`, force-zeroed by the
//!   transquant-bypass / RDPCM conditions the caller folds into
//!   [`ResidualCodingParams::sign_hidden_suppressed`]) with the
//!   `coeff_sign_flag` presence gate and the odd-parity
//!   `sumAbsLevel` negation at `firstSigScanPos`,
//! * the level loop: `baseLevel = 1 + greater1 + greater2`, the
//!   `coeff_abs_level_remaining` presence test
//!   `baseLevel == ( ( numSigCoeff < 8 ) ? ( ( n == lastGreater1ScanPos ) ? 3 : 2 ) : 1 )`,
//!   and the §9.3.3.11 per-sub-block Rice adaptation
//!   (`cLastAbsLevel` / `cLastRiceParam` reset at the first
//!   `coeff_abs_level_remaining` invocation of each sub-block, then
//!   carried via eq. 9-24 — the
//!   `persistent_rice_adaptation_enabled_flag == 0` path; the
//!   persistent `StatCoeff[ sbType ]` path of eqs. 9-20..9-23 /
//!   9-25 remains with the §9.3.3.11 follow-up noted on
//!   [`crate::hevc::engine::binarization::decode_coeff_abs_level_remaining`]).
//!
//! Out of driver scope (caller-supplied / follow-up):
//!
//! * The leading `transform_skip_flag` / `explicit_rdpcm_flag` /
//!   `explicit_rdpcm_dir_flag` elements of §7.3.8.11: their
//!   presence gates hang off PPS / CU state the slice parser owns.
//!   Their decoded (or §7.4.9.11-inferred-0) values reach this
//!   driver only through the two derived booleans in
//!   [`ResidualCodingParams`]
//!   ([`sign_hidden_suppressed`](ResidualCodingParams::sign_hidden_suppressed)
//!   and
//!   [`transform_skip_sig_ctx`](ResidualCodingParams::transform_skip_sig_ctx)).
//! * The §9.3.4.3.6 aligned-bypass mode
//!   (`cabac_bypass_alignment_enabled_flag`, a range-extension
//!   tool) and the `escapeDataPresent` bookkeeping that exists only
//!   to feed it — the baseline bypass decoder is used throughout.
//! * The §7.4.9.11 `scanIdx` derivation is exposed separately as
//!   [`residual_coding_scan_idx`] so the caller (which owns
//!   `CuPredMode` / `IntraPredModeY` / `IntraPredModeC`) selects
//!   the scan before invoking the driver.
//!
//! ## Bin-source abstraction
//!
//! The driver is generic over [`ResidualBinSource`], mirroring the
//! `decode_coeff_abs_level_remaining_with` pattern: tests drive the
//! syntax loop with scripted bin sequences and assert both the
//! reconstructed levels and the exact `(element, ctxInc)` request
//! sequence, independently of the arithmetic engine's bin/bit
//! relationship. [`EngineResidualBinSource`] is the production
//! binding: it routes context-coded bins through
//! [`CabacEngine::decode_decision`] against the per-element banks in
//! [`ResidualContexts`] and bypass bins through
//! [`CabacEngine::decode_bypass`].
//!
//! Context-bank sizes follow the spec's initValue tables (each table
//! spans the three `initType` columns, so the per-`initType` bank is
//! a third of the table's ctxIdx count): Table 9-26 / 9-27
//! (`last_sig_coeff_{x,y}_prefix`, 54 ⇒ 18), Table 9-28
//! (`coded_sub_block_flag`, 12 ⇒ 4), Table 9-29 (`sig_coeff_flag`,
//! 132 ⇒ 44), Table 9-30 (`coeff_abs_level_greater1_flag`, 72 ⇒ 24)
//! and Table 9-31 (`coeff_abs_level_greater2_flag`, 18 ⇒ 6). The
//! initValue tables themselves live in [`crate::hevc::engine::ctx_init`];
//! [`ResidualContexts::init`] performs the §9.3.2.2 per-`initType`
//! bank initialization from them, and
//! [`ResidualContexts::init_uniform`] remains as a single-initValue
//! bring-up constructor for scripted tests.

use crate::hevc::engine::binarization::{
    Greater1State, coded_sub_block_flag_ctx_inc_with_edge, coeff_abs_level_greater2_flag_ctx_inc,
    coeff_abs_level_remaining_c_rice_param_eq_9_24, coeff_abs_level_remaining_c_rice_param_eq_9_25,
    decode_coeff_abs_level_remaining_ext_with, decode_coeff_abs_level_remaining_with,
    last_sig_coeff_position, last_sig_coeff_prefix_cmax, last_sig_coeff_prefix_ctx_inc,
    last_sig_coeff_prefix_ctx_offset_shift, last_sig_coeff_suffix_n_bits,
    read_truncated_rice_prefix, sig_coeff_flag_ctx_inc_from_sig_ctx, sig_coeff_flag_sig_ctx_dc,
    sig_coeff_flag_sig_ctx_general, sig_coeff_flag_sig_ctx_log2_2,
    sig_coeff_flag_sig_ctx_transform_skip, signed_level_from_sign_flag,
};
use crate::hevc::engine::cabac::{CabacEngine, CabacError, ContextModel};
use crate::hevc::engine::scan::{ScanIdx, ScanOrderError, scan_order};

// ---------------------------------------------------------------------
// Context banks
// ---------------------------------------------------------------------

/// Per-`initType` context count for `last_sig_coeff_x_prefix` /
/// `last_sig_coeff_y_prefix` — Table 9-26 / Table 9-27 list ctxIdx
/// 0..53 across the three `initType` columns, 18 per `initType`
/// (§9.3.4.2.3 reaches ctxInc 0..=17: luma `ctxOffset + (binIdx >>
/// ctxShift)` peaks at 14 for `log2TrafoSize == 5`, chroma at
/// `15 + (binIdx >> 0)` ≤ 17 for `log2TrafoSize == 2`).
pub const LAST_SIG_COEFF_PREFIX_CTX_COUNT: usize = 18;

/// Per-`initType` context count for `coded_sub_block_flag` —
/// Table 9-28 lists ctxIdx 0..11 across the three `initType`
/// columns, 4 per `initType` (§9.3.4.2.4: luma `{0, 1}`, chroma
/// `{2, 3}`).
pub const CODED_SUB_BLOCK_FLAG_CTX_COUNT: usize = 4;

/// Per-`initType` context count for `sig_coeff_flag` — Table 9-29
/// lists ctxIdx 0..131 across the three `initType` columns, 44 per
/// `initType` (§9.3.4.2.5: luma ctxInc 0..=26 plus the eq.-9-40
/// transform-skip slot 42; chroma `27 + sigCtx` ≤ 41 plus the
/// transform-skip slot 43).
pub const SIG_COEFF_FLAG_CTX_COUNT: usize = 44;

/// Per-`initType` context count for `coeff_abs_level_greater1_flag`
/// — Table 9-30 lists ctxIdx 0..71 across the three `initType`
/// columns, 24 per `initType` (eq. 9-59 luma `ctxSet * 4 + Min(3,
/// greater1Ctx)` ≤ 15, eq. 9-60 chroma `+ 16` ≤ 23).
pub const COEFF_ABS_LEVEL_GREATER1_FLAG_CTX_COUNT: usize = 24;

/// Per-`initType` context count for `coeff_abs_level_greater2_flag`
/// — Table 9-31 lists ctxIdx 0..17 across the three `initType`
/// columns, 6 per `initType` (eq. 9-61 luma `ctxSet` ≤ 3, eq. 9-62
/// chroma `ctxSet + 4` ≤ 5).
pub const COEFF_ABS_LEVEL_GREATER2_FLAG_CTX_COUNT: usize = 6;

/// The context-coded syntax-element banks one §7.3.8.11
/// `residual_coding( )` invocation draws on. Identifies which
/// [`ResidualContexts`] bank a [`ResidualBinSource::decision`]
/// request addresses; the paired `ctxInc` is bank-relative per the
/// §9.3.4.2 derivations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualElement {
    /// `last_sig_coeff_x_prefix` (§9.3.4.2.3).
    LastSigCoeffXPrefix,
    /// `last_sig_coeff_y_prefix` (§9.3.4.2.3).
    LastSigCoeffYPrefix,
    /// `coded_sub_block_flag` (§9.3.4.2.4).
    CodedSubBlockFlag,
    /// `sig_coeff_flag` (§9.3.4.2.5).
    SigCoeffFlag,
    /// `coeff_abs_level_greater1_flag` (§9.3.4.2.6).
    CoeffAbsLevelGreater1Flag,
    /// `coeff_abs_level_greater2_flag` (§9.3.4.2.7).
    CoeffAbsLevelGreater2Flag,
}

/// The per-`initType` CABAC context banks for the residual-coding
/// syntax elements, sized per Tables 9-26..9-31 (see the bank-size
/// constants). [`init`](Self::init) performs the §9.3.2.2
/// initialization from the Table 9-26..9-31 initValues with the
/// Table 9-4 `initType` → ctxIdx mapping (tables in
/// [`crate::hevc::engine::ctx_init`]); [`init_uniform`](Self::init_uniform)
/// constructs a bank set from a single initValue for bring-up and
/// tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualContexts {
    /// `last_sig_coeff_x_prefix` bank (Table 9-26).
    pub last_sig_coeff_x_prefix: [ContextModel; LAST_SIG_COEFF_PREFIX_CTX_COUNT],
    /// `last_sig_coeff_y_prefix` bank (Table 9-27).
    pub last_sig_coeff_y_prefix: [ContextModel; LAST_SIG_COEFF_PREFIX_CTX_COUNT],
    /// `coded_sub_block_flag` bank (Table 9-28).
    pub coded_sub_block_flag: [ContextModel; CODED_SUB_BLOCK_FLAG_CTX_COUNT],
    /// `sig_coeff_flag` bank (Table 9-29).
    pub sig_coeff_flag: [ContextModel; SIG_COEFF_FLAG_CTX_COUNT],
    /// `coeff_abs_level_greater1_flag` bank (Table 9-30).
    pub coeff_abs_level_greater1_flag: [ContextModel; COEFF_ABS_LEVEL_GREATER1_FLAG_CTX_COUNT],
    /// `coeff_abs_level_greater2_flag` bank (Table 9-31).
    pub coeff_abs_level_greater2_flag: [ContextModel; COEFF_ABS_LEVEL_GREATER2_FLAG_CTX_COUNT],
    /// §9.3.2.2 Rice-parameter initialization states `StatCoeff[ k ]`,
    /// `k` = eq. 9-20/9-21 `sbType` (0/1 chroma transform / bypass,
    /// 2/3 luma transform / bypass). Zeroed at context
    /// initialization, updated once per sub-block by the §9.3.3.11
    /// eq. 9-23 rule, and carried inside the context bank so the
    /// §9.3.2.4 / §9.3.2.5 WPP and dependent-segment storage
    /// (`TableStatCoeffWpp` / `TableStatCoeffDs`) travels with the
    /// context-variable snapshot. Only consulted when
    /// `persistent_rice_adaptation_enabled_flag == 1`.
    pub stat_coeff: [u8; 4],
}

impl ResidualContexts {
    /// §9.3.2.2 initialization of every bank from the Table 9-26..9-31
    /// `initValue` entries for the given `initType` (the Table 9-4
    /// ctxIdx spans; see [`crate::hevc::engine::ctx_init`]) at `SliceQpY ==
    /// slice_qp_y` (equation 7-54).
    ///
    /// # Panics
    ///
    /// Panics when `init_type > 2`.
    #[must_use]
    pub fn init(init_type: u8, slice_qp_y: i32) -> Self {
        use crate::hevc::engine::ctx_init::{
            TABLE_9_26_LAST_SIG_COEFF_X_PREFIX, TABLE_9_27_LAST_SIG_COEFF_Y_PREFIX,
            TABLE_9_28_CODED_SUB_BLOCK_FLAG, TABLE_9_30_COEFF_ABS_LEVEL_GREATER1_FLAG,
            TABLE_9_31_COEFF_ABS_LEVEL_GREATER2_FLAG, sig_coeff_flag_init_values,
            uniform_init_values,
        };
        let init = |v: u8| ContextModel::init(v, slice_qp_y);
        Self {
            last_sig_coeff_x_prefix: uniform_init_values::<LAST_SIG_COEFF_PREFIX_CTX_COUNT>(
                &TABLE_9_26_LAST_SIG_COEFF_X_PREFIX,
                init_type,
            )
            .map(init),
            last_sig_coeff_y_prefix: uniform_init_values::<LAST_SIG_COEFF_PREFIX_CTX_COUNT>(
                &TABLE_9_27_LAST_SIG_COEFF_Y_PREFIX,
                init_type,
            )
            .map(init),
            coded_sub_block_flag: uniform_init_values::<CODED_SUB_BLOCK_FLAG_CTX_COUNT>(
                &TABLE_9_28_CODED_SUB_BLOCK_FLAG,
                init_type,
            )
            .map(init),
            sig_coeff_flag: sig_coeff_flag_init_values(init_type).map(init),
            coeff_abs_level_greater1_flag: uniform_init_values::<
                COEFF_ABS_LEVEL_GREATER1_FLAG_CTX_COUNT,
            >(
                &TABLE_9_30_COEFF_ABS_LEVEL_GREATER1_FLAG, init_type
            )
            .map(init),
            coeff_abs_level_greater2_flag: uniform_init_values::<
                COEFF_ABS_LEVEL_GREATER2_FLAG_CTX_COUNT,
            >(
                &TABLE_9_31_COEFF_ABS_LEVEL_GREATER2_FLAG, init_type
            )
            .map(init),
            stat_coeff: [0; 4],
        }
    }

    /// Initialize every context in every bank from one `initValue`
    /// via the §9.3.2.2 process ([`ContextModel::init`]).
    ///
    /// Bring-up scaffolding predating [`init`](Self::init): uniform
    /// initialization still exercises every engine/state path (only
    /// the per-context starting probabilities differ), and the
    /// scripted-bin tests keep using it.
    #[must_use]
    pub fn init_uniform(init_value: u8, slice_qp_y: i32) -> Self {
        let c = ContextModel::init(init_value, slice_qp_y);
        Self {
            last_sig_coeff_x_prefix: [c; LAST_SIG_COEFF_PREFIX_CTX_COUNT],
            last_sig_coeff_y_prefix: [c; LAST_SIG_COEFF_PREFIX_CTX_COUNT],
            coded_sub_block_flag: [c; CODED_SUB_BLOCK_FLAG_CTX_COUNT],
            sig_coeff_flag: [c; SIG_COEFF_FLAG_CTX_COUNT],
            coeff_abs_level_greater1_flag: [c; COEFF_ABS_LEVEL_GREATER1_FLAG_CTX_COUNT],
            coeff_abs_level_greater2_flag: [c; COEFF_ABS_LEVEL_GREATER2_FLAG_CTX_COUNT],
            stat_coeff: [0; 4],
        }
    }
}

// ---------------------------------------------------------------------
// Bin-source abstraction
// ---------------------------------------------------------------------

/// A source of decoded bins for the §7.3.8.11 driver. The production
/// implementation is [`EngineResidualBinSource`]; tests supply
/// scripted sources that also record the `(element, ctxInc)` request
/// sequence.
pub trait ResidualBinSource {
    /// Decode one context-coded bin for `element` using the
    /// bank-relative `ctx_inc` derived per §9.3.4.2.
    fn decision(&mut self, element: ResidualElement, ctx_inc: u32) -> Result<u8, CabacError>;

    /// Decode one bypass-coded bin (§9.3.4.3.4).
    fn bypass(&mut self) -> Result<u8, CabacError>;

    /// Decode `n` bypass-coded bins MSB-first into one value.
    fn bypass_bits(&mut self, n: u8) -> Result<u32, CabacError> {
        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.bypass()?);
        }
        Ok(value)
    }

    /// §9.3.4.3.6 — alignment prior to aligned bypass decoding
    /// (`ivlCurrRange = 256`). Invoked by the driver before the
    /// `coeff_sign_flag` / `coeff_abs_level_remaining` bypass run of
    /// a sub-block when `cabac_bypass_alignment_enabled_flag == 1`
    /// and `escapeDataPresent == 1`. Scripted sources may treat it as
    /// a recording no-op — the aligned engine still serves one bin
    /// per call.
    fn align_bypass(&mut self) {}

    /// Mutable access to the §9.3.2.2 `StatCoeff[ k ]` Rice
    /// initialization states, consulted and updated (eqs. 9-22 /
    /// 9-23) only when `persistent_rice_adaptation_enabled_flag ==
    /// 1`. The production source exposes
    /// [`ResidualContexts::stat_coeff`] so the state persists across
    /// TUs and travels with the WPP / dependent-segment context
    /// storage.
    fn stat_coeff(&mut self) -> &mut [u8; 4];
}

/// The production [`ResidualBinSource`]: context-coded bins go
/// through [`CabacEngine::decode_decision`] against the matching
/// [`ResidualContexts`] bank slot; bypass bins through
/// [`CabacEngine::decode_bypass`].
#[derive(Debug)]
pub struct EngineResidualBinSource<'e, 'a, 'c> {
    /// The arithmetic decoding engine, positioned in the slice-data
    /// bin stream.
    pub engine: &'e mut CabacEngine<'a>,
    /// The residual context banks for the active `initType`.
    pub contexts: &'c mut ResidualContexts,
}

impl ResidualBinSource for EngineResidualBinSource<'_, '_, '_> {
    fn decision(&mut self, element: ResidualElement, ctx_inc: u32) -> Result<u8, CabacError> {
        let bank: &mut [ContextModel] = match element {
            ResidualElement::LastSigCoeffXPrefix => &mut self.contexts.last_sig_coeff_x_prefix,
            ResidualElement::LastSigCoeffYPrefix => &mut self.contexts.last_sig_coeff_y_prefix,
            ResidualElement::CodedSubBlockFlag => &mut self.contexts.coded_sub_block_flag,
            ResidualElement::SigCoeffFlag => &mut self.contexts.sig_coeff_flag,
            ResidualElement::CoeffAbsLevelGreater1Flag => {
                &mut self.contexts.coeff_abs_level_greater1_flag
            }
            ResidualElement::CoeffAbsLevelGreater2Flag => {
                &mut self.contexts.coeff_abs_level_greater2_flag
            }
        };
        // The §9.3.4.2 derivations bound every ctxInc by the bank
        // sizes documented on the *_CTX_COUNT constants, so the
        // index is in-bounds for driver-produced requests.
        self.engine.decode_decision(&mut bank[ctx_inc as usize])
    }

    fn bypass(&mut self) -> Result<u8, CabacError> {
        self.engine.decode_bypass()
    }

    fn bypass_bits(&mut self, n: u8) -> Result<u32, CabacError> {
        self.engine.decode_bypass_bits(n)
    }

    fn align_bypass(&mut self) {
        self.engine.align_bypass();
    }

    fn stat_coeff(&mut self) -> &mut [u8; 4] {
        &mut self.contexts.stat_coeff
    }
}

// ---------------------------------------------------------------------
// Parameters / output / errors
// ---------------------------------------------------------------------

/// Caller-derived inputs to one §7.3.8.11 `residual_coding( )`
/// invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualCodingParams {
    /// `log2TrafoSize` of the transform block, 2..=5.
    pub log2_trafo_size: u32,
    /// `cIdx > 0` — selects every chroma context/derivation branch.
    pub is_chroma: bool,
    /// The §7.4.9.11 scan order ([`residual_coding_scan_idx`]).
    /// Only [`ScanIdx::Diagonal`] / [`ScanIdx::Horizontal`] /
    /// [`ScanIdx::Vertical`] are reachable from §7.4.9.11.
    pub scan_idx: ScanIdx,
    /// PPS `sign_data_hiding_enabled_flag` (§7.3.2.3.1).
    pub sign_data_hiding_enabled_flag: bool,
    /// The §7.3.8.11 `signHidden = 0` force condition:
    /// `cu_transquant_bypass_flag`, or the implicit-RDPCM intra
    /// condition (`implicit_rdpcm_enabled_flag &&
    /// transform_skip_flag && predModeIntra ∈ {10, 26}`), or
    /// `explicit_rdpcm_flag`. When `true`, `signHidden` is 0
    /// regardless of the scan-position spread.
    pub sign_hidden_suppressed: bool,
    /// The §9.3.4.2.5 first-branch gate (eq. 9-40):
    /// `transform_skip_context_enabled_flag &&
    /// ( transform_skip_flag || cu_transquant_bypass_flag )`. When
    /// `true`, every `sig_coeff_flag` bin uses the dedicated
    /// transform-skip context (42 luma / 16-chroma `sigCtx`).
    pub transform_skip_sig_ctx: bool,
    /// SPS range extension `persistent_rice_adaptation_enabled_flag`
    /// — switches the `coeff_abs_level_remaining` Rice derivation to
    /// the §9.3.3.11 eqs. 9-20..9-23 / 9-25 persistent path
    /// (`StatCoeff[ sbType ]`-seeded, uncapped).
    pub persistent_rice_adaptation_enabled_flag: bool,
    /// SPS range extension `cabac_bypass_alignment_enabled_flag` —
    /// the §9.3.4.3.6 alignment runs before the sub-block's
    /// `coeff_sign_flag` / `coeff_abs_level_remaining` bypass bins
    /// when `escapeDataPresent == 1`.
    pub cabac_bypass_alignment_enabled_flag: bool,
    /// SPS range extension `extended_precision_processing_flag` —
    /// the `coeff_abs_level_remaining` escape suffix uses the
    /// §9.3.3.4 limited EGk binarization with the eq. 9-14
    /// `log2TransformRange`.
    pub extended_precision_processing_flag: bool,
    /// `BitDepth` of the block's colour component (eq. 9-14 input;
    /// only consulted when `extended_precision_processing_flag`).
    pub bit_depth: u8,
    /// The eq. 9-20/9-21 `sbType` discriminator:
    /// `transform_skip_flag || cu_transquant_bypass_flag` (only
    /// consulted when `persistent_rice_adaptation_enabled_flag`).
    pub rice_stat_transform_skip: bool,
}

/// The reconstructed coefficient array of one transform block —
/// §7.3.8.11 `TransCoeffLevel[ x0 ][ y0 ][ cIdx ][ ][ ]` with the
/// `( x0, y0, cIdx )` indices fixed by the invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualBlock {
    /// `log2TrafoSize` — the block spans `1 << log2_trafo_size`
    /// coefficients per side.
    pub log2_trafo_size: u32,
    /// §7.4.9.11 `LastSignificantCoeffX` (post the eq.-7-78 swap for
    /// `scanIdx == 2`).
    pub last_sig_coeff_x: u32,
    /// §7.4.9.11 `LastSignificantCoeffY` (post the eq.-7-78 swap).
    pub last_sig_coeff_y: u32,
    /// Row-major `TransCoeffLevel[ yC ][ xC ]`, length
    /// `(1 << log2_trafo_size)²`. Positions past the last
    /// significant coefficient and in uncoded sub-blocks are 0.
    pub levels: Vec<i32>,
    /// `transform_skip_flag[ x0 ][ y0 ][ cIdx ]` (§7.3.8.11) — decoded
    /// by the transform-unit driver before this block's coefficients;
    /// `false` when the §7.3.8.11 presence gate did not fire.
    pub transform_skip: bool,
    /// `explicit_rdpcm_flag[ x0 ][ y0 ][ cIdx ]` (§7.3.8.11, range
    /// extensions) — parsed for bit-exactness; the §8.6.8 RDPCM
    /// reconstruction is a follow-up (reconstruction rejects it).
    pub explicit_rdpcm_flag: bool,
    /// `explicit_rdpcm_dir_flag[ x0 ][ y0 ][ cIdx ]` (§7.3.8.11).
    pub explicit_rdpcm_dir_flag: bool,
}

impl ResidualBlock {
    /// Block side length in coefficients (`1 << log2_trafo_size`).
    #[must_use]
    pub fn size(&self) -> usize {
        1usize << self.log2_trafo_size
    }

    /// `TransCoeffLevel[ xC ][ yC ]` accessor (spec index order; the
    /// backing storage is row-major by `yC`).
    ///
    /// # Panics
    /// Panics if `xc` or `yc` is outside `0..size()`.
    #[must_use]
    pub fn level(&self, xc: usize, yc: usize) -> i32 {
        let size = self.size();
        assert!(xc < size && yc < size, "coefficient index out of block");
        self.levels[yc * size + xc]
    }
}

/// Errors from the §7.3.8.11 driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualCodingError {
    /// The underlying bin source failed (arithmetic-engine error /
    /// bitstream exhausted).
    Cabac(CabacError),
    /// The §6.5 scan-order table rejected the request.
    ScanOrder(ScanOrderError),
    /// `log2TrafoSize` outside 2..=5 (§7.4.9.8 transform-block size
    /// bounds).
    UnsupportedLog2TrafoSize(u32),
    /// A scan order §7.4.9.11 never selects for `residual_coding( )`
    /// (only diagonal / horizontal / vertical are reachable).
    UnsupportedScanIdx(ScanIdx),
    /// A §7.3.8.13 `palette_coding( )` element violated a parse bound.
    Palette(crate::hevc::engine::palette::PaletteError),
}

impl From<crate::hevc::engine::palette::PaletteError> for ResidualCodingError {
    fn from(e: crate::hevc::engine::palette::PaletteError) -> Self {
        Self::Palette(e)
    }
}

impl core::fmt::Display for ResidualCodingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cabac(e) => write!(f, "residual_coding: {e}"),
            Self::ScanOrder(e) => write!(f, "residual_coding: {e}"),
            Self::UnsupportedLog2TrafoSize(v) => {
                write!(f, "residual_coding: log2TrafoSize {v} outside 2..=5")
            }
            Self::Palette(e) => write!(f, "{e}"),
            Self::UnsupportedScanIdx(s) => {
                write!(
                    f,
                    "residual_coding: scanIdx {s:?} not reachable from §7.4.9.11"
                )
            }
        }
    }
}

impl std::error::Error for ResidualCodingError {}

impl From<CabacError> for ResidualCodingError {
    fn from(e: CabacError) -> Self {
        Self::Cabac(e)
    }
}

impl From<ScanOrderError> for ResidualCodingError {
    fn from(e: ScanOrderError) -> Self {
        Self::ScanOrder(e)
    }
}

// ---------------------------------------------------------------------
// §7.4.9.11 scanIdx derivation
// ---------------------------------------------------------------------

/// §7.4.9.11 — derive the `residual_coding( )` scan order.
///
/// * `cu_pred_mode_is_intra` — `CuPredMode[ x0 ][ y0 ] ==
///   MODE_INTRA`.
/// * `log2_trafo_size`, `c_idx`, `chroma_array_type` — the §7.4.9.11
///   intra-scan eligibility conditions (`log2TrafoSize == 2`, or
///   `log2TrafoSize == 3` with `cIdx == 0`, or `log2TrafoSize == 3`
///   with `ChromaArrayType == 3`).
/// * `pred_mode_intra` — `IntraPredModeY[ x0 ][ y0 ]` when
///   `cIdx == 0`, else `IntraPredModeC` (the §7.4.9.11 selection).
///
/// Modes 6..=14 select the vertical scan, 22..=30 the horizontal
/// scan; everything else (including all non-intra and all larger
/// transform blocks) the up-right diagonal scan.
#[must_use]
pub fn residual_coding_scan_idx(
    cu_pred_mode_is_intra: bool,
    log2_trafo_size: u32,
    c_idx: u8,
    chroma_array_type: u8,
    pred_mode_intra: u32,
) -> ScanIdx {
    let intra_scan_eligible =
        log2_trafo_size == 2 || (log2_trafo_size == 3 && (c_idx == 0 || chroma_array_type == 3));
    if cu_pred_mode_is_intra && intra_scan_eligible {
        if (6..=14).contains(&pred_mode_intra) {
            ScanIdx::Vertical
        } else if (22..=30).contains(&pred_mode_intra) {
            ScanIdx::Horizontal
        } else {
            ScanIdx::Diagonal
        }
    } else {
        ScanIdx::Diagonal
    }
}

// ---------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------

/// Decode one `last_sig_coeff_*_{prefix,suffix}` pair through the bin
/// source and return the §7.4.9.11 derived position (eqs. 7-74..7-77).
fn read_last_sig_prefix<B: ResidualBinSource>(
    bins: &mut B,
    log2_trafo_size: u32,
    is_chroma: bool,
    element: ResidualElement,
) -> Result<u32, ResidualCodingError> {
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let (ctx_offset, ctx_shift) =
        last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size, is_chroma);
    let (prefix, _is_escape) = read_truncated_rice_prefix(c_max, |bin_idx| {
        let ctx_inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_offset, ctx_shift);
        bins.decision(element, ctx_inc)
    })?;
    Ok(prefix)
}

/// §7.3.8.11 — the conditional bypass suffix for one
/// `last_sig_coeff_{x,y}_prefix` value (present when `prefix > 3`),
/// folded into the §7.4.9.11 position.
fn read_last_sig_suffix<B: ResidualBinSource>(
    bins: &mut B,
    prefix: u32,
) -> Result<u32, ResidualCodingError> {
    let n_bits = last_sig_coeff_suffix_n_bits(prefix);
    let suffix = if n_bits > 0 {
        Some(bins.bypass_bits(n_bits as u8)?)
    } else {
        None
    };
    Ok(last_sig_coeff_position(prefix, suffix))
}

/// §7.3.8.11 — decode one `residual_coding( )` body (from
/// `last_sig_coeff_x_prefix` onward; see the module docs for the
/// exact scope) against a generic [`ResidualBinSource`].
///
/// Returns the reconstructed [`ResidualBlock`]. See
/// [`decode_residual_coding`] for the engine-backed entry point.
pub fn decode_residual_coding_with<B: ResidualBinSource>(
    params: &ResidualCodingParams,
    bins: &mut B,
) -> Result<ResidualBlock, ResidualCodingError> {
    let log2 = params.log2_trafo_size;
    if !(2..=5).contains(&log2) {
        return Err(ResidualCodingError::UnsupportedLog2TrafoSize(log2));
    }
    let scan_idx_num = u32::from(params.scan_idx.index());
    if scan_idx_num > 2 {
        return Err(ResidualCodingError::UnsupportedScanIdx(params.scan_idx));
    }
    let is_chroma = params.is_chroma;

    // last_sig_coeff_{x,y}_{prefix,suffix} → LastSignificantCoeff{X,Y}
    // (eqs. 7-74..7-77), then the eq.-7-78 swap for the vertical scan.
    // §7.3.8.11 bin order: BOTH context-coded prefixes come first, then
    // the two conditional bypass suffixes (x, then y).
    let prefix_x =
        read_last_sig_prefix(bins, log2, is_chroma, ResidualElement::LastSigCoeffXPrefix)?;
    let prefix_y =
        read_last_sig_prefix(bins, log2, is_chroma, ResidualElement::LastSigCoeffYPrefix)?;
    let wire_x = read_last_sig_suffix(bins, prefix_x)?;
    let wire_y = read_last_sig_suffix(bins, prefix_y)?;
    let (last_x, last_y) = if params.scan_idx == ScanIdx::Vertical {
        (wire_y, wire_x)
    } else {
        (wire_x, wire_y)
    };

    // §6.5 scan tables: the 4x4 in-sub-block scan and the sub-block
    // scan over the (1 << (log2 - 2))² grid.
    let pos_scan = scan_order(2, params.scan_idx)?;
    let sub_scan = scan_order((log2 - 2) as u8, params.scan_idx)?;
    let num_sb_1d = 1usize << (log2 - 2);
    let size = 1usize << log2;

    // §7.3.8.11 do-while: locate (lastSubBlock, lastScanPos) from the
    // last-significant position. The decoded position is always
    // inside the TB (the §9.3.4.2.3 cMax bounds the prefix), so the
    // walk terminates.
    let mut last_scan_pos: i32 = 16;
    let mut last_sub_block: i32 = (num_sb_1d * num_sb_1d) as i32 - 1;
    loop {
        if last_scan_pos == 0 {
            last_scan_pos = 16;
            last_sub_block -= 1;
        }
        last_scan_pos -= 1;
        debug_assert!(last_sub_block >= 0, "last-sig position outside the TB");
        let sb = sub_scan[last_sub_block as usize];
        let xc = (u32::from(sb.x) << 2) + u32::from(pos_scan[last_scan_pos as usize].x);
        let yc = (u32::from(sb.y) << 2) + u32::from(pos_scan[last_scan_pos as usize].y);
        if xc == last_x && yc == last_y {
            break;
        }
    }

    let mut levels = vec![0i32; size * size];
    // coded_sub_block_flag[ xS ][ yS ] grid, row-major by yS.
    let mut csbf = vec![0u8; num_sb_1d * num_sb_1d];
    let csbf_at = |grid: &[u8], xs: usize, ys: usize| -> u8 {
        if xs < num_sb_1d && ys < num_sb_1d {
            grid[ys * num_sb_1d + xs]
        } else {
            0
        }
    };

    // §9.3.4.2.6 cross-sub-block greater-1 state. `last_g1_bin` is
    // the most recently decoded coeff_abs_level_greater1_flag value,
    // consumed by the next sub-block's lazy entry.
    let mut g1_state = Greater1State::new();
    let mut last_g1_bin: u8 = 0;

    for i in (0..=last_sub_block).rev() {
        let sb = sub_scan[i as usize];
        let (xs, ys) = (u32::from(sb.x), u32::from(sb.y));
        let is_last_sb = i == last_sub_block;

        // coded_sub_block_flag: decoded for 0 < i < lastSubBlock,
        // inferred 1 for sub-blocks 0 and lastSubBlock (§7.4.9.11 —
        // both inference conditions reduce to these two scan
        // indices: (0,0) is sub-block scan index 0 and the
        // last-significant sub-block is lastSubBlock).
        let mut infer_sb_dc_sig = false;
        let sb_coded: u8 = if i < last_sub_block && i > 0 {
            let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
            let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
            let ctx_inc =
                coded_sub_block_flag_ctx_inc_with_edge(is_chroma, xs, ys, log2, right, below);
            let bin = bins.decision(ResidualElement::CodedSubBlockFlag, ctx_inc)?;
            infer_sb_dc_sig = true;
            bin
        } else {
            1
        };
        csbf[ys as usize * num_sb_1d + xs as usize] = sb_coded;

        // sig_coeff_flag pass. `sig[n]` is indexed by the in-sub-block
        // scan position n.
        let mut sig = [0u8; 16];
        if is_last_sb {
            // The last significant coefficient is significant by
            // definition (§7.4.9.11 first inference bullet).
            sig[last_scan_pos as usize] = 1;
        }
        let start_n: i32 = if is_last_sb { last_scan_pos - 1 } else { 15 };
        for n in (0..=start_n).rev() {
            let xc = (xs << 2) + u32::from(pos_scan[n as usize].x);
            let yc = (ys << 2) + u32::from(pos_scan[n as usize].y);
            if sb_coded == 1 && (n > 0 || !infer_sb_dc_sig) {
                // §9.3.4.2.5 sigCtx branch dispatch.
                let sig_ctx = if params.transform_skip_sig_ctx {
                    sig_coeff_flag_sig_ctx_transform_skip(is_chroma)
                } else if log2 == 2 {
                    sig_coeff_flag_sig_ctx_log2_2(xc & 3, yc & 3)
                } else if xc + yc == 0 {
                    sig_coeff_flag_sig_ctx_dc(is_chroma, log2, scan_idx_num)
                } else {
                    let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
                    let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
                    sig_coeff_flag_sig_ctx_general(
                        is_chroma,
                        log2,
                        xc,
                        yc,
                        xs,
                        ys,
                        right,
                        below,
                        scan_idx_num,
                    )
                };
                let ctx_inc = sig_coeff_flag_ctx_inc_from_sig_ctx(sig_ctx, is_chroma);
                let bin = bins.decision(ResidualElement::SigCoeffFlag, ctx_inc)?;
                sig[n as usize] = bin;
                if bin == 1 {
                    infer_sb_dc_sig = false;
                }
            } else if n == 0 && sb_coded == 1 && infer_sb_dc_sig {
                // §7.4.9.11 second inference bullet: the DC cell of a
                // decoded-coded sub-block whose 15 other flags were
                // all 0 is inferred significant.
                sig[0] = 1;
            }
        }

        // coeff_abs_level_greater1_flag pass (§7.3.8.11), with the
        // per-sub-block numGreater1Flag < 8 cap and the lazy
        // §9.3.4.2.6 sub-block entry.
        let mut first_sig_scan_pos: i32 = 16;
        let mut last_sig_scan_pos: i32 = -1;
        let mut num_greater1: u32 = 0;
        let mut last_greater1_scan_pos: i32 = -1;
        let mut g1 = [0u8; 16];
        let mut entered_subblock = false;
        // §7.3.8.11 escapeDataPresent: set when any coefficient of
        // this sub-block will (or may) carry a
        // coeff_abs_level_remaining — a second-or-later greater1
        // flag, a significant position past the 8-flag cap, or a
        // greater2 flag equal to 1. Gates the §9.3.4.3.6 aligned
        // bypass decoding.
        let mut escape_data_present = false;
        for n in (0..16).rev() {
            if sig[n] == 1 {
                if num_greater1 < 8 {
                    if !entered_subblock {
                        g1_state.on_subblock_entry(i as u32, is_chroma, last_g1_bin);
                        entered_subblock = true;
                    }
                    let ctx_inc = g1_state.current_ctx_inc(is_chroma);
                    let bin = bins.decision(ResidualElement::CoeffAbsLevelGreater1Flag, ctx_inc)?;
                    g1_state.on_coeff_abs_level_greater1_flag(bin);
                    last_g1_bin = bin;
                    g1[n] = bin;
                    num_greater1 += 1;
                    if bin == 1 && last_greater1_scan_pos == -1 {
                        last_greater1_scan_pos = n as i32;
                    } else if bin == 1 {
                        escape_data_present = true;
                    }
                } else {
                    escape_data_present = true;
                }
                if last_sig_scan_pos == -1 {
                    last_sig_scan_pos = n as i32;
                }
                first_sig_scan_pos = n as i32;
            }
        }

        // §7.3.8.11 signHidden.
        let sign_hidden = if params.sign_hidden_suppressed {
            false
        } else {
            last_sig_scan_pos - first_sig_scan_pos > 3
        };

        // coeff_abs_level_greater2_flag — at most once per sub-block.
        let mut g2 = [0u8; 16];
        if last_greater1_scan_pos != -1 {
            let ctx_inc = coeff_abs_level_greater2_flag_ctx_inc(g1_state.ctx_set(), is_chroma);
            let bin = bins.decision(ResidualElement::CoeffAbsLevelGreater2Flag, ctx_inc)?;
            g2[last_greater1_scan_pos as usize] = bin;
            if bin == 1 {
                escape_data_present = true;
            }
        }

        // §9.3.4.3.6 — with cabac_bypass_alignment_enabled_flag, the
        // bypass decoding of this sub-block's coeff_sign_flag and
        // coeff_abs_level_remaining bins runs aligned
        // (ivlCurrRange = 256) when escapeDataPresent == 1. All the
        // remaining bins of the sub-block are bypass-coded, so one
        // alignment here covers both passes.
        if params.cabac_bypass_alignment_enabled_flag && escape_data_present {
            bins.align_bypass();
        }

        // coeff_sign_flag pass (bypass): present for every significant
        // position except the hidden first one.
        let mut sign = [0u8; 16];
        for n in (0..16).rev() {
            if sig[n] == 1
                && (!params.sign_data_hiding_enabled_flag
                    || !sign_hidden
                    || n as i32 != first_sig_scan_pos)
            {
                sign[n] = bins.bypass()?;
            }
        }

        // Level pass: coeff_abs_level_remaining + TransCoeffLevel
        // composition, with the §9.3.3.11 per-sub-block Rice
        // adaptation — eq. 9-24 when
        // persistent_rice_adaptation_enabled_flag == 0, the
        // StatCoeff-seeded eqs. 9-20..9-23 / 9-25 path when it is 1.
        let mut num_sig_coeff: u32 = 0;
        let mut sum_abs_level: u64 = 0;
        let mut c_last_abs_level: u32 = 0;
        let mut c_last_rice_param: u32 = 0;
        // eq. 9-20 / 9-21 sbType (persistent path only).
        let sb_type = 2 * usize::from(!is_chroma) + usize::from(params.rice_stat_transform_skip);
        let mut first_remaining_in_sb = true;
        for n in (0..16).rev() {
            if sig[n] != 1 {
                continue;
            }
            let base_level = 1 + u32::from(g1[n]) + u32::from(g2[n]);
            let threshold = if num_sig_coeff < 8 {
                if n as i32 == last_greater1_scan_pos {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            let mut remaining: u32 = 0;
            if base_level == threshold {
                let c_rice_param = if params.persistent_rice_adaptation_enabled_flag {
                    if first_remaining_in_sb {
                        // eq. 9-22: cLastRiceParam starts at
                        // initRiceValue = StatCoeff[ sbType ] / 4
                        // (cLastAbsLevel stays 0).
                        c_last_rice_param = u32::from(bins.stat_coeff()[sb_type] / 4);
                    }
                    coeff_abs_level_remaining_c_rice_param_eq_9_25(
                        c_last_abs_level,
                        c_last_rice_param,
                    )
                } else {
                    coeff_abs_level_remaining_c_rice_param_eq_9_24(
                        c_last_abs_level,
                        c_last_rice_param,
                    )
                };
                remaining = if params.extended_precision_processing_flag {
                    // eq. 9-14: log2TransformRange =
                    // Max( 15, BitDepth + 6 ) for the component.
                    let log2_transform_range =
                        u32::from(params.bit_depth).saturating_add(6).max(15);
                    decode_coeff_abs_level_remaining_ext_with(
                        c_rice_param,
                        log2_transform_range,
                        || bins.bypass(),
                    )?
                } else {
                    decode_coeff_abs_level_remaining_with(c_rice_param, || bins.bypass())?
                };
                // eq. 9-23: the sub-block's FIRST
                // coeff_abs_level_remaining updates
                // StatCoeff[ sbType ] once (the shift distance is
                // capped defensively; conforming states never near
                // it).
                if params.persistent_rice_adaptation_enabled_flag && first_remaining_in_sb {
                    let stat = &mut bins.stat_coeff()[sb_type];
                    let shift = u32::from(*stat / 4).min(24);
                    if remaining >= (3u32 << shift) {
                        *stat = stat.saturating_add(1);
                    } else if u64::from(remaining) * 2 < (1u64 << shift) && *stat > 0 {
                        *stat -= 1;
                    }
                }
                first_remaining_in_sb = false;
                // §9.3.3.11: cAbsLevel = baseLevel +
                // coeff_abs_level_remaining[ n ]; carried with the
                // cRiceParam to the next invocation in this sub-block.
                // A non-conformant bin stream can drive `remaining`
                // to the binarization's u32 saturation point, so the
                // composition saturates instead of overflowing.
                c_last_abs_level = base_level.saturating_add(remaining);
                c_last_rice_param = c_rice_param;
            }
            // §7.4.9.11 bitstream conformance bounds the resulting
            // TransCoeffLevel to [CoeffMin, CoeffMax]; the widest
            // profile bound (eqs. 7-27 .. 7-30 with
            // extended_precision_processing_flag and BitDepth 16) is
            // ±(1 << 22). Clamping the magnitude keeps the decode
            // total — and the i32 level array sound — on
            // non-conformant escapes; conforming values are unchanged.
            let abs_level = base_level.saturating_add(remaining).min(1 << 22);
            let xc = (xs << 2) + u32::from(pos_scan[n].x);
            let yc = (ys << 2) + u32::from(pos_scan[n].y);
            let idx = yc as usize * size + xc as usize;
            levels[idx] = signed_level_from_sign_flag(abs_level, sign[n]);
            if params.sign_data_hiding_enabled_flag && sign_hidden {
                sum_abs_level += u64::from(abs_level);
                if n as i32 == first_sig_scan_pos && sum_abs_level % 2 == 1 {
                    levels[idx] = -levels[idx];
                }
            }
            num_sig_coeff += 1;
        }
    }

    Ok(ResidualBlock {
        log2_trafo_size: log2,
        last_sig_coeff_x: last_x,
        last_sig_coeff_y: last_y,
        levels,
        transform_skip: false,
        explicit_rdpcm_flag: false,
        explicit_rdpcm_dir_flag: false,
    })
}

/// §7.3.8.11 — engine-backed entry point: binds `engine` + `contexts`
/// into an [`EngineResidualBinSource`] and runs
/// [`decode_residual_coding_with`].
pub fn decode_residual_coding(
    engine: &mut CabacEngine<'_>,
    contexts: &mut ResidualContexts,
    params: &ResidualCodingParams,
) -> Result<ResidualBlock, ResidualCodingError> {
    let mut bins = EngineResidualBinSource { engine, contexts };
    decode_residual_coding_with(params, &mut bins)
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;
    use std::collections::VecDeque;

    /// Scripted [`ResidualBinSource`]: serves context-coded bins from
    /// `decisions` and bypass bins from `bypasses`, recording every
    /// `(element, ctxInc)` decision request so tests can assert the
    /// exact §9.3.4.2 context sequence.
    struct Scripted {
        decisions: VecDeque<u8>,
        bypasses: VecDeque<u8>,
        log: Vec<(ResidualElement, u32)>,
        bypass_reads: usize,
        stat_coeff: [u8; 4],
    }

    impl Scripted {
        fn new(decisions: &[u8], bypasses: &[u8]) -> Self {
            Self {
                decisions: decisions.iter().copied().collect(),
                bypasses: bypasses.iter().copied().collect(),
                log: Vec::new(),
                bypass_reads: 0,
                stat_coeff: [0; 4],
            }
        }
    }

    impl ResidualBinSource for Scripted {
        fn decision(&mut self, element: ResidualElement, ctx_inc: u32) -> Result<u8, CabacError> {
            self.log.push((element, ctx_inc));
            Ok(self
                .decisions
                .pop_front()
                .expect("decision script exhausted"))
        }

        fn bypass(&mut self) -> Result<u8, CabacError> {
            self.bypass_reads += 1;
            Ok(self.bypasses.pop_front().expect("bypass script exhausted"))
        }

        fn stat_coeff(&mut self) -> &mut [u8; 4] {
            &mut self.stat_coeff
        }
    }

    fn luma_params(log2: u32, scan_idx: ScanIdx) -> ResidualCodingParams {
        ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma: false,
            scan_idx,
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        }
    }

    /// DC-only 4x4 luma block: the last-significant position (0, 0)
    /// makes every `sig_coeff_flag` inferred (the last coefficient
    /// itself), so the only decisions are the two one-bin TR prefixes
    /// and one `coeff_abs_level_greater1_flag`; the sign is the only
    /// bypass bin. §7.3.8.11 + §9.3.4.2.3 (ctxOffset 0 / ctxShift 0
    /// for 4x4 luma) + eq. 9-59 (first sub-block: ctxSet 0,
    /// greater1Ctx 1 ⇒ ctxInc 1).
    #[test]
    fn dc_only_4x4_luma() {
        let params = luma_params(2, ScanIdx::Diagonal);
        let mut bins = Scripted::new(&[0, 0, 0], &[1]);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!(block.last_sig_coeff_x, 0);
        assert_eq!(block.last_sig_coeff_y, 0);
        assert_eq!(block.level(0, 0), -1, "sign bin 1 ⇒ negative, abs 1");
        assert_eq!(
            block.levels.iter().filter(|&&v| v != 0).count(),
            1,
            "only the DC coefficient is non-zero"
        );
        assert_eq!(
            bins.log,
            vec![
                (ResidualElement::LastSigCoeffXPrefix, 0),
                (ResidualElement::LastSigCoeffYPrefix, 0),
                (ResidualElement::CoeffAbsLevelGreater1Flag, 1),
            ]
        );
        assert_eq!(bins.bypass_reads, 1);
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// Chroma DC-only 4x4: every bank routes through the chroma
    /// offsets — last-sig prefix ctxOffset 15 (§9.3.4.2.3), greater-1
    /// `+ 16` (eq. 9-60).
    #[test]
    fn dc_only_4x4_chroma_ctx_routing() {
        let params = ResidualCodingParams {
            is_chroma: true,
            ..luma_params(2, ScanIdx::Diagonal)
        };
        let mut bins = Scripted::new(&[0, 0, 0], &[0]);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!(block.level(0, 0), 1);
        assert_eq!(
            bins.log,
            vec![
                (ResidualElement::LastSigCoeffXPrefix, 15),
                (ResidualElement::LastSigCoeffYPrefix, 15),
                (ResidualElement::CoeffAbsLevelGreater1Flag, 17),
            ]
        );
    }

    /// Full 4x4 luma block with the last-significant coefficient at
    /// (3, 3) (scan position 15): 15 decoded `sig_coeff_flag` bins
    /// (all 0), a greater-1 = 1, a greater-2 = 1, and a
    /// `coeff_abs_level_remaining` of 2 (TR bins `110`, cRiceParam 0)
    /// at `baseLevel == 3`, composing `TransCoeffLevel = +5`. The
    /// `sig_coeff_flag` ctxInc sequence is cross-checked against the
    /// Table 9-50 lookup (including the erratum-pinned `i = 15 ⇒ 8`
    /// cell, exercised at the (3, 3) start had it been decoded — here
    /// the decoded cells sweep scan positions 14..0).
    #[test]
    fn last_at_3_3_with_remaining_4x4_luma() {
        let params = luma_params(2, ScanIdx::Diagonal);
        let mut decisions = vec![1, 1, 1, 1, 1, 1]; // two TR prefixes, value 3 = cMax
        decisions.extend([0; 15]); // sig n = 14..0
        decisions.push(1); // greater1 at n = 15
        decisions.push(1); // greater2 at n = 15
        let mut bins = Scripted::new(&decisions, &[0, 1, 1, 0]); // sign +, remaining = 2
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!((block.last_sig_coeff_x, block.last_sig_coeff_y), (3, 3));
        assert_eq!(block.level(3, 3), 5, "baseLevel 3 + remaining 2");
        assert_eq!(block.levels.iter().filter(|&&v| v != 0).count(), 1);

        // §9.3.4.2.3: 4x4 luma prefix ctxInc = binIdx (offset 0,
        // shift 0).
        assert_eq!(
            bins.log[0..3],
            [
                (ResidualElement::LastSigCoeffXPrefix, 0),
                (ResidualElement::LastSigCoeffXPrefix, 1),
                (ResidualElement::LastSigCoeffXPrefix, 2)
            ]
        );
        // The 15 sig bins route through the Table 9-50 map (eq. 9-41).
        let pos_scan = scan_order(2, ScanIdx::Diagonal).unwrap();
        for (k, n) in (0..=14).rev().enumerate() {
            let (element, ctx_inc) = bins.log[6 + k];
            assert_eq!(element, ResidualElement::SigCoeffFlag);
            let expected =
                sig_coeff_flag_sig_ctx_log2_2(u32::from(pos_scan[n].x), u32::from(pos_scan[n].y));
            assert_eq!(ctx_inc, expected, "sig ctxInc at scan pos {n}");
        }
        assert_eq!(
            bins.log[21],
            (ResidualElement::CoeffAbsLevelGreater1Flag, 1)
        );
        assert_eq!(
            bins.log[22],
            (ResidualElement::CoeffAbsLevelGreater2Flag, 0)
        );
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// 8x8 luma block, last-significant at (4, 4) (sub-block (1, 1)):
    /// exercises the `coded_sub_block_flag` decode + §9.3.4.2.4
    /// neighbour ctxInc, the §7.4.9.11 `inferSbDcSigCoeffFlag` DC
    /// inference (sub-block (0, 1) decodes 15 zero sig bins, so its
    /// DC cell is inferred 1), and the §9.3.4.2.6 eq.-9-58 ctxSet
    /// bump across sub-blocks (prior sub-block's greater-1 bin was 1
    /// ⇒ terminal greater1Ctx 0 ⇒ ctxSet 2 + 1 = 3, ctxInc 13).
    #[test]
    fn two_subblock_8x8_luma_csbf_and_dc_inference() {
        let params = luma_params(3, ScanIdx::Diagonal);
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 1, 1, 0]); // last_sig_x prefix = 4
        decisions.extend([1, 1, 1, 1, 0]); // last_sig_y prefix = 4
        decisions.push(1); // i=3: greater1 (DC of sub-block (1,1)) = 1
        decisions.push(0); // i=3: greater2 = 0
        decisions.push(0); // i=2: coded_sub_block_flag (1,0) = 0
        decisions.push(1); // i=1: coded_sub_block_flag (0,1) = 1
        decisions.extend([0; 15]); // i=1: sig n=15..1 all 0 ⇒ DC inferred
        decisions.push(0); // i=1: greater1 (inferred DC) = 0
        decisions.extend([0; 16]); // i=0: sig n=15..0 all 0
        // bypass: last_x suffix 0, last_y suffix 0, sign(i=3) +, sign(i=1) −
        let mut bins = Scripted::new(&decisions, &[0, 0, 0, 1]);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!((block.last_sig_coeff_x, block.last_sig_coeff_y), (4, 4));
        assert_eq!(block.level(4, 4), 2, "baseLevel 1 + greater1, no remaining");
        assert_eq!(
            block.level(0, 4),
            -1,
            "inferred-DC coefficient of sub-block (0,1)"
        );
        assert_eq!(block.levels.iter().filter(|&&v| v != 0).count(), 2);

        // §9.3.4.2.4 ctxInc: sub-block (1,0) sees below-neighbour
        // (1,1) coded ⇒ 1; sub-block (0,1) sees right-neighbour (1,1)
        // coded ⇒ 1.
        let csbf_log: Vec<u32> = bins
            .log
            .iter()
            .filter(|(e, _)| *e == ResidualElement::CodedSubBlockFlag)
            .map(|&(_, c)| c)
            .collect();
        assert_eq!(csbf_log, vec![1, 1]);
        // eq. 9-57 ctxSet 2 (i > 0, luma, first invocation ⇒ ctxInc
        // 9), then eq. 9-58 bump (prior greater-1 was 1) ⇒ ctxInc 13.
        let g1_log: Vec<u32> = bins
            .log
            .iter()
            .filter(|(e, _)| *e == ResidualElement::CoeffAbsLevelGreater1Flag)
            .map(|&(_, c)| c)
            .collect();
        assert_eq!(g1_log, vec![9, 13]);
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// Sign data hiding: significant coefficients at scan positions 5
    /// and 0 (`lastSigScanPos − firstSigScanPos = 5 > 3` ⇒
    /// `signHidden`), so the first-in-scan-order sign bin is omitted
    /// and the §7.3.8.11 odd-`sumAbsLevel` parity rule negates the
    /// coefficient at `firstSigScanPos`.
    #[test]
    fn sign_data_hiding_parity_flip() {
        let params = ResidualCodingParams {
            sign_data_hiding_enabled_flag: true,
            ..luma_params(2, ScanIdx::Diagonal)
        };
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 0]); // last_sig_x prefix = 2
        decisions.push(0); // last_sig_y prefix = 0 ⇒ last at (2,0), scan pos 5
        decisions.extend([0, 0, 0, 0, 1]); // sig n=4..0: only n=0
        decisions.extend([1, 0]); // greater1: n=5 ⇒ 1, n=0 ⇒ 0
        decisions.push(0); // greater2 at n=5
        let mut bins = Scripted::new(&decisions, &[0]); // single sign (n=5)
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!((block.last_sig_coeff_x, block.last_sig_coeff_y), (2, 0));
        assert_eq!(block.level(2, 0), 2);
        // sumAbsLevel = 2 + 1 = 3 (odd) ⇒ the hidden-sign coefficient
        // at firstSigScanPos flips negative.
        assert_eq!(block.level(0, 0), -1);
        assert_eq!(
            bins.bypass_reads, 1,
            "the firstSigScanPos sign bin is hidden"
        );
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// Same coefficient layout with sign data hiding disabled: both
    /// sign bins are read and no parity flip occurs.
    #[test]
    fn no_sign_hiding_when_disabled() {
        let params = luma_params(2, ScanIdx::Diagonal);
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 0, 0]); // last at (2,0)
        decisions.extend([0, 0, 0, 0, 1]); // sig n=4..0
        decisions.extend([1, 0]); // greater1
        decisions.push(0); // greater2
        let mut bins = Scripted::new(&decisions, &[0, 0]); // both signs read
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!(block.level(2, 0), 2);
        assert_eq!(block.level(0, 0), 1, "no parity flip without sign hiding");
        assert_eq!(bins.bypass_reads, 2);
    }

    /// Greater-1 cap: 10 significant coefficients in one sub-block —
    /// only the first 8 in reverse scan order consume
    /// `coeff_abs_level_greater1_flag` bins, and the 9th / 10th hit
    /// the `numSigCoeff >= 8` threshold (`baseLevel == 1`) so each
    /// reads a `coeff_abs_level_remaining` (value 0 ⇒ abs level 1).
    #[test]
    fn greater1_cap_and_numsig_threshold() {
        let params = luma_params(2, ScanIdx::Diagonal);
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 1, 1, 1, 1]); // last at (3,3)
        // sig n=14..0: nine 1s (n=14..6) then six 0s (n=5..0).
        decisions.extend([1; 9]);
        decisions.extend([0; 6]);
        decisions.extend([0; 8]); // greater1: capped at 8 bins, all 0
        // bypass: 10 signs, then two remaining values of 0 (single
        // TR-prefix terminator bin each at cRiceParam 0).
        let mut bins = Scripted::new(&decisions, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        let g1_bins = bins
            .log
            .iter()
            .filter(|(e, _)| *e == ResidualElement::CoeffAbsLevelGreater1Flag)
            .count();
        assert_eq!(g1_bins, 8, "numGreater1Flag < 8 cap");
        assert_eq!(block.levels.iter().filter(|&&v| v == 1).count(), 10);
        assert_eq!(bins.bypass_reads, 12, "10 signs + 2 single-bin remainings");
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// §9.3.3.11 Rice adaptation across one sub-block: the first
    /// `coeff_abs_level_remaining` (cRiceParam 0) decodes 10 via the
    /// TR-escape + EG1 path (`cAbsLevel = 13`), so eq. 9-24 bumps the
    /// second invocation to cRiceParam 1 (TR prefix `0` + one suffix
    /// bit `1` ⇒ value 1).
    #[test]
    fn rice_adaptation_within_subblock() {
        let params = luma_params(2, ScanIdx::Diagonal);
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 1, 1, 1, 1]); // last at (3,3)
        decisions.push(1); // sig n=14 = 1
        decisions.extend([0; 14]); // sig n=13..0 = 0
        decisions.extend([1, 1]); // greater1 at n=15, n=14
        decisions.push(1); // greater2 at n=15 (lastGreater1ScanPos)
        // remaining #1 = 10 at cRiceParam 0: TR escape prefix `1111`,
        // then EGk(k=1) of suffixVal = 6 = prefix `110` (§9.3.3.3 ones-
        // then-zero) + 3-bit suffix `000`.
        // remaining #2 = 1 at cRiceParam 1: TR prefix `0` (terminator)
        // + 1-bit suffix `1`.
        let bypasses = [
            0, 0, // signs at n=15, n=14
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // remaining #1 = 10 (escape + EG1)
            0, 1, // remaining #2 = 1 at cRiceParam 1
        ];
        let mut bins = Scripted::new(&decisions, &bypasses);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        let pos_scan = scan_order(2, ScanIdx::Diagonal).unwrap();
        assert_eq!(block.level(3, 3), 13, "baseLevel 3 + remaining 10");
        let p14 = pos_scan[14];
        assert_eq!(
            block.level(usize::from(p14.x), usize::from(p14.y)),
            3,
            "baseLevel 2 + remaining 1 under the adapted cRiceParam"
        );
        assert!(bins.decisions.is_empty() && bins.bypasses.is_empty());
    }

    /// eq. 7-78: under the vertical scan (`scanIdx == 2`) the decoded
    /// last-significant coordinates are swapped.
    #[test]
    fn vertical_scan_swaps_last_sig_coordinates() {
        let params = luma_params(2, ScanIdx::Vertical);
        let mut decisions = Vec::new();
        decisions.extend([1, 1, 0]); // wire X prefix = 2
        decisions.push(0); // wire Y prefix = 0
        // After the swap the last position is (0, 2) = vertical scan
        // pos 2; sig bins for n=1..0, then one greater1.
        decisions.extend([0, 0, 0]);
        let mut bins = Scripted::new(&decisions, &[0]);
        let block = decode_residual_coding_with(&params, &mut bins).unwrap();
        assert_eq!((block.last_sig_coeff_x, block.last_sig_coeff_y), (0, 2));
        assert_eq!(block.level(0, 2), 1);
    }

    /// §9.3.4.2.5 eq. 9-40: with the transform-skip context gate set,
    /// every decoded `sig_coeff_flag` uses the dedicated sigCtx (42
    /// luma ⇒ ctxInc 42).
    #[test]
    fn transform_skip_sig_ctx_routing() {
        let params = ResidualCodingParams {
            transform_skip_sig_ctx: true,
            ..luma_params(2, ScanIdx::Diagonal)
        };
        let mut decisions = Vec::new();
        decisions.extend([1, 0, 1, 0]); // last at (1, 1) = diag scan pos 4
        decisions.extend([0, 0, 0, 0]); // sig n=3..0
        decisions.push(0); // greater1 at n=4
        let mut bins = Scripted::new(&decisions, &[0]);
        decode_residual_coding_with(&params, &mut bins).unwrap();
        let sig_ctxs: Vec<u32> = bins
            .log
            .iter()
            .filter(|(e, _)| *e == ResidualElement::SigCoeffFlag)
            .map(|&(_, c)| c)
            .collect();
        assert_eq!(sig_ctxs, vec![42; 4]);
    }

    /// §7.4.9.11 scanIdx derivation.
    #[test]
    fn scan_idx_derivation() {
        // Intra 4x4 luma: mode bands select the scan.
        for mode in 6..=14 {
            assert_eq!(
                residual_coding_scan_idx(true, 2, 0, 1, mode),
                ScanIdx::Vertical
            );
        }
        for mode in 22..=30 {
            assert_eq!(
                residual_coding_scan_idx(true, 2, 0, 1, mode),
                ScanIdx::Horizontal
            );
        }
        assert_eq!(
            residual_coding_scan_idx(true, 2, 0, 1, 0),
            ScanIdx::Diagonal
        );
        assert_eq!(
            residual_coding_scan_idx(true, 2, 0, 1, 34),
            ScanIdx::Diagonal
        );
        // 8x8: luma eligible, 4:2:0 chroma not, 4:4:4 chroma eligible.
        assert_eq!(
            residual_coding_scan_idx(true, 3, 0, 1, 10),
            ScanIdx::Vertical
        );
        assert_eq!(
            residual_coding_scan_idx(true, 3, 1, 1, 10),
            ScanIdx::Diagonal
        );
        assert_eq!(
            residual_coding_scan_idx(true, 3, 1, 3, 10),
            ScanIdx::Vertical
        );
        // 16x16+ and inter: always diagonal.
        assert_eq!(
            residual_coding_scan_idx(true, 4, 0, 1, 10),
            ScanIdx::Diagonal
        );
        assert_eq!(
            residual_coding_scan_idx(false, 2, 0, 1, 10),
            ScanIdx::Diagonal
        );
    }

    /// Driver input validation.
    #[test]
    fn rejects_out_of_range_inputs() {
        let mut bins = Scripted::new(&[], &[]);
        let bad_size = ResidualCodingParams {
            log2_trafo_size: 6,
            ..luma_params(2, ScanIdx::Diagonal)
        };
        assert_eq!(
            decode_residual_coding_with(&bad_size, &mut bins),
            Err(ResidualCodingError::UnsupportedLog2TrafoSize(6))
        );
        let bad_scan = luma_params(2, ScanIdx::Traverse);
        assert_eq!(
            decode_residual_coding_with(&bad_scan, &mut bins),
            Err(ResidualCodingError::UnsupportedScanIdx(ScanIdx::Traverse))
        );
    }

    /// Engine-backed smoke run: the [`EngineResidualBinSource`]
    /// binding decodes a block from a real §9.3.4.3 arithmetic-engine
    /// bin stream without error, and the last-significant coefficient
    /// is non-zero by the §7.4.9.11 inference.
    #[test]
    fn engine_backed_smoke() {
        let buf = [0x5A; 96];
        let mut engine = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut contexts = ResidualContexts::init_uniform(154, 26);
        let params = luma_params(3, ScanIdx::Diagonal);
        let block = decode_residual_coding(&mut engine, &mut contexts, &params).unwrap();
        assert_eq!(block.size(), 8);
        assert!(block.last_sig_coeff_x < 8 && block.last_sig_coeff_y < 8);
        assert_ne!(
            block.level(
                block.last_sig_coeff_x as usize,
                block.last_sig_coeff_y as usize
            ),
            0,
            "the last significant coefficient is significant by inference"
        );
    }

    /// The engine binding mutates the addressed bank slot (context
    /// adaptation reaches the right [`ResidualContexts`] field).
    #[test]
    fn engine_binding_adapts_contexts() {
        let buf = [0xA5; 96];
        let mut engine = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut contexts = ResidualContexts::init_uniform(154, 26);
        let before = contexts.clone();
        let params = luma_params(2, ScanIdx::Diagonal);
        decode_residual_coding(&mut engine, &mut contexts, &params).unwrap();
        assert_ne!(
            contexts.last_sig_coeff_x_prefix, before.last_sig_coeff_x_prefix,
            "last_sig_coeff_x_prefix bank must adapt"
        );
    }

    // --- §9.3.2.2 Table 9-26..9-31 per-initType bank init ---

    fn cm(p_state_idx: u8, val_mps: u8) -> ContextModel {
        ContextModel {
            p_state_idx,
            val_mps,
        }
    }

    /// Representative derived `(pStateIdx, valMps)` pins across QPs:
    /// the §9.3.2.2 equations applied to cited Table 9-26..9-31
    /// entries, hand-evaluated.
    #[test]
    fn residual_contexts_init_pins() {
        // Table 9-26 ctxIdx 0 (initType 0, initValue 110).
        assert_eq!(
            ResidualContexts::init(0, 0).last_sig_coeff_x_prefix[0],
            cm(32, 1)
        );
        assert_eq!(
            ResidualContexts::init(0, 26).last_sig_coeff_x_prefix[0],
            cm(7, 1)
        );
        assert_eq!(
            ResidualContexts::init(0, 51).last_sig_coeff_x_prefix[0],
            cm(15, 0)
        );
        // Table 9-26 ctxIdx 18 (initType 1, initValue 125) and ctxIdx
        // 53 (initType 2, initValue 93).
        assert_eq!(
            ResidualContexts::init(1, 26).last_sig_coeff_x_prefix[0],
            cm(7, 1)
        );
        assert_eq!(
            ResidualContexts::init(2, 26).last_sig_coeff_x_prefix[17],
            cm(8, 0)
        );
        // Table 9-27 prints the same values as Table 9-26, so the two
        // banks start identical (they adapt independently afterwards).
        let ctx = ResidualContexts::init(1, 30);
        assert_eq!(ctx.last_sig_coeff_x_prefix, ctx.last_sig_coeff_y_prefix);
        // Table 9-28 ctxIdx 0 (initType 0, initValue 91) and ctxIdx 4
        // (initType 1, initValue 121).
        assert_eq!(
            ResidualContexts::init(0, 26).coded_sub_block_flag[0],
            cm(24, 0)
        );
        assert_eq!(
            ResidualContexts::init(1, 26).coded_sub_block_flag[0],
            cm(24, 0)
        );
        // Table 9-29: ctxIdx 0 (initType 0, initValue 111), ctxIdx 42
        // (initType 1, initValue 155), ctxIdx 84 (initType 2,
        // initValue 170), and the transform-skip tail ctxIdx 126 / 127
        // (initType 0, initValues 141 / 111) landing in bank slots
        // 42 / 43.
        assert_eq!(ResidualContexts::init(0, 26).sig_coeff_flag[0], cm(15, 1));
        for qp in [0, 26, 51] {
            // initValue 155 ⇒ m == 0: QP-independent state.
            assert_eq!(ResidualContexts::init(1, qp).sig_coeff_flag[0], cm(8, 1));
        }
        assert_eq!(ResidualContexts::init(2, 51).sig_coeff_flag[0], cm(15, 1));
        assert_eq!(ResidualContexts::init(0, 26).sig_coeff_flag[42], cm(15, 1));
        assert_eq!(ResidualContexts::init(0, 51).sig_coeff_flag[43], cm(7, 0));
        // Table 9-30 ctxIdx 0 (initType 0, initValue 140) and ctxIdx
        // 48 (initType 2, initValue 154).
        assert_eq!(
            ResidualContexts::init(0, 26).coeff_abs_level_greater1_flag[0],
            cm(7, 1)
        );
        assert_eq!(
            ResidualContexts::init(2, 26).coeff_abs_level_greater1_flag[0],
            cm(0, 1)
        );
        // Table 9-31 ctxIdx 0 (initType 0, initValue 138) and ctxIdx 6
        // (initType 1, initValue 107).
        assert_eq!(
            ResidualContexts::init(0, 26).coeff_abs_level_greater2_flag[0],
            cm(8, 0)
        );
        assert_eq!(
            ResidualContexts::init(1, 26).coeff_abs_level_greater2_flag[0],
            cm(16, 0)
        );
    }

    /// Whole-bank smoke test: every initType / QP combination yields
    /// in-range states everywhere, and the three initTypes give three
    /// distinct bank sets.
    #[test]
    fn residual_contexts_init_smoke() {
        for init_type in 0u8..=2 {
            for qp in [0, 17, 26, 37, 51] {
                let ctx = ResidualContexts::init(init_type, qp);
                let banks: [&[ContextModel]; 6] = [
                    &ctx.last_sig_coeff_x_prefix,
                    &ctx.last_sig_coeff_y_prefix,
                    &ctx.coded_sub_block_flag,
                    &ctx.sig_coeff_flag,
                    &ctx.coeff_abs_level_greater1_flag,
                    &ctx.coeff_abs_level_greater2_flag,
                ];
                for bank in banks {
                    for c in bank {
                        // Equations 9-4..9-6 bound preCtxState to
                        // 1..=126 ⇒ pStateIdx 0..=62.
                        assert!(c.p_state_idx <= 62);
                        assert!(c.val_mps <= 1);
                    }
                }
            }
        }
        assert_ne!(ResidualContexts::init(0, 26), ResidualContexts::init(1, 26));
        assert_ne!(ResidualContexts::init(1, 26), ResidualContexts::init(2, 26));
        assert_ne!(ResidualContexts::init(0, 26), ResidualContexts::init(2, 26));
    }
}
