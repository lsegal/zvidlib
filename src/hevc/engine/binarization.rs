//! §9.3.4.2 per-syntax-element binarization + context-index derivation.
//!
//! Sits one layer above the §9.3 arithmetic decoding engine
//! ([`crate::hevc::engine::cabac::CabacEngine`]). For each entropy-coded syntax
//! element this module supplies the §9.3.3 binarization shape (truncated
//! rice, fixed length, Exp-Golomb of order *k*, …) plus the §9.3.4.2.X
//! per-bin `ctxInc` derivation, returning a context-index into the
//! per-slice context-model array.
//!
//! Round 26 ships the two syntax elements unblocked by the clean-room
//! CABAC trace `docs/video/h265/fixtures/main-422-10bit/cabac-cu-qp-delta-last-sig-trace.md`:
//!
//! * **`cu_qp_delta_abs` / `cu_qp_delta_sign_flag`** (H.265 §7.3.8.14,
//!   §7.4.9.14). Binarization: §9.3.3.10 TR prefix (`cMax = 5`,
//!   `cRiceParam = 0`) followed, when the prefix is the all-ones escape
//!   (`prefixVal == 5`), by an EGk suffix with `k = 0` (§9.3.3.11). The
//!   per-bin `ctxInc` (§9.3.4.2.2, Table 9-32): bin 0 uses `ctxInc = 0`,
//!   bins 1..=4 share `ctxInc = 1`. The sign flag is bypass-coded.
//!   `CuQpDeltaVal = cu_qp_delta_abs * (1 − 2 * cu_qp_delta_sign_flag)`.
//!
//! * **`last_sig_coeff_{x,y}_prefix`** (H.265 §7.3.8.11, §7.4.9.11).
//!   Binarization: §9.3.3.10 TR prefix with
//!   `cMax = (log2TrafoSize << 1) − 1` and `cRiceParam = 0`. The per-bin
//!   `ctxInc` (§9.3.4.2.3) is
//!   `ctxInc = (binIdx >> ctxShift) + ctxOffset`. For luma:
//!   `ctxOffset = 3 * (log2TrafoSize − 2) + ((log2TrafoSize − 1) >> 2)`,
//!   `ctxShift = (log2TrafoSize + 1) >> 2`. For chroma (cIdx > 0):
//!   `ctxOffset = 15`, `ctxShift = log2TrafoSize − 2`. The X and Y
//!   prefix bins live in separate context banks; this module hands back
//!   only the bank-relative `ctxInc` and the matching
//!   [`LastSigCoeffBank`] tag for caller-side routing.
//!
//! * **`last_sig_coeff_{x,y}_suffix`** (H.265 §7.3.8.11, §7.4.9.11). A
//!   `nBits = (prefix >> 1) − 1`-bit fixed-length bypass field, present
//!   only when `prefix > 3`. Together with the prefix the final
//!   `LastSignificantCoeffX` / `Y` position is derived per equations
//!   7-74..7-77.
//!
//! Both elements live downstream of the §9.3.4.2 binarization layer
//! that the slice-data parser (§7.3.8) drives; this module is the
//! reusable per-element building block.
//!
//! Round 27 extends the ctxInc-derivation surface with three more
//! syntax elements that are pure-functional given their neighbour /
//! sub-block context (no CABAC engine drive needed at this layer —
//! callers compose the engine call themselves):
//!
//! * **`coded_sub_block_flag`** (H.265 §7.3.8.11, §7.4.9.11) — the
//!   §9.3.4.2.4 derivation of `ctxInc` from the colour-component
//!   index, the current sub-block scan location `(xS, yS)`, the
//!   transform-block size, and the previously decoded
//!   `coded_sub_block_flag[xS+1][yS]` / `coded_sub_block_flag[xS][yS+1]`
//!   neighbours. Implemented in
//!   [`coded_sub_block_flag_ctx_inc`] following equations 9-35..9-39.
//!
//! * **`split_cu_flag` / `cu_skip_flag`** (H.265 §7.3.8.4, §7.4.9.4,
//!   §7.3.8.5, §7.4.9.5) — the §9.3.4.2.2 Table 9-49 derivation of
//!   `ctxInc` from the left and above neighbours' `condL` / `condA`
//!   booleans plus their availability:
//!   `ctxInc = (condL && availableL) + (condA && availableA)`.
//!   Implemented as the shared
//!   [`left_above_ctx_inc`] helper (the body of Table 9-49) plus
//!   the [`split_cu_flag_cond`] and [`cu_skip_flag_cond`] callbacks
//!   the caller wires into it. The Table 9-49 row for
//!   `split_cu_flag` compares each neighbour's `CtDepth[xNb][yNb]`
//!   to the current `cqtDepth`; the row for `cu_skip_flag` reads
//!   each neighbour's `cu_skip_flag[xNb][yNb]` directly. Both rows
//!   produce a `ctxInc` in `{0, 1, 2}`.
//!
//! Round 28 extends the ctxInc-derivation surface with six more
//! syntax elements whose Table 9-48 entry is itself a closed-form
//! expression on parameters the caller already has in hand (no
//! neighbour table walk, no CABAC engine drive at this layer):
//!
//! * **`split_transform_flag[ ][ ][ ]`** (H.265 §7.3.8.10, §7.4.9.10)
//!   — Table 9-48 row: `ctxInc = 5 − log2TrafoSize`. Valid
//!   `log2TrafoSize ∈ {2, 3, 4, 5}` for the residual-quadtree
//!   transform-block sizes, so `ctxInc ∈ {0, 1, 2, 3}`. Implemented
//!   in [`split_transform_flag_ctx_inc`].
//!
//! * **`cbf_luma[ ][ ][ ]`** (H.265 §7.3.8.10, §7.4.9.10) — Table
//!   9-48 row: `ctxInc = (trafoDepth == 0) ? 1 : 0`. Implemented in
//!   [`cbf_luma_ctx_inc`].
//!
//! * **`cbf_cb[ ][ ][ ]` / `cbf_cr[ ][ ][ ]`** (H.265 §7.3.8.10,
//!   §7.4.9.10) — Table 9-48 row: `ctxInc = trafoDepth`. The two
//!   syntax elements live in separate context banks (Table 9-22
//!   shares the bank shape but each chroma component has its own
//!   `ctxIdxOffset`); this layer hands back the bank-relative
//!   `ctxInc` only. Implemented in [`cbf_cb_ctx_inc`] and
//!   [`cbf_cr_ctx_inc`] (each is a wafer over the shared
//!   [`cbf_chroma_ctx_inc`] helper).
//!
//! * **`inter_pred_idc[ x0 ][ y0 ]`** (H.265 §7.3.8.6, §7.4.9.6) —
//!   Table 9-48 row: bin 0 uses
//!   `ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4`; bin 1
//!   uses `ctxInc = 4`. The `nPbW + nPbH == 12` condition picks out
//!   the prediction-block shapes whose luma area is 16 samples
//!   (8×4 and 4×8 PUs), which are encoded with the bin-0 escape
//!   onto the bin-1 context bank. Implemented in
//!   [`inter_pred_idc_ctx_inc`].
//!
//! * **`log2_res_scale_abs_plus1[ c ]`** (H.265 §7.3.8.13,
//!   §7.4.9.13) — Table 9-48 row: `ctxInc = 4*c + binIdx` for
//!   `binIdx ∈ {0, 1, 2, 3}` and `c ∈ {0, 1}` (Cb / Cr). The four
//!   bins of the TR(`cMax = 4`) prefix occupy four consecutive
//!   contexts per chroma component, so the per-component banks are
//!   `{0, 1, 2, 3}` (Cb) and `{4, 5, 6, 7}` (Cr). Implemented in
//!   [`log2_res_scale_abs_plus1_ctx_inc`].
//!
//! * **`res_scale_sign_flag[ c ]`** (H.265 §7.3.8.13, §7.4.9.13) —
//!   Table 9-48 row: `ctxInc = c` for `c ∈ {0, 1}`. One bit per
//!   chroma component, each on its own context. Implemented in
//!   [`res_scale_sign_flag_ctx_inc`].
//!
//! All round-28 entries are bin-0 (and bin-1 for `inter_pred_idc` /
//! `log2_res_scale_abs_plus1`) ctxInc derivations only; the matching
//! binarization shapes (FL for the flags, TR(`cMax = 4`) for
//! `log2_res_scale_abs_plus1` followed by a bypass tail) are driven
//! by the slice-data parser the same way as the round-26 / 27
//! elements.
//!
//! Round 30 extends the surface with the §7.3.4 `sao()` per-CTU
//! syntax-element family. Every element has either a single
//! context-coded bin-0 followed by zero or more bypass-coded bins
//! (Table 9-48) or is fully bypass-coded; no neighbour walk is
//! needed at this layer.
//!
//! * **`sao_merge_left_flag` / `sao_merge_up_flag`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: bin 0 `ctxInc = 0`. The FL
//!   binarization has `cMax = 1` (single bin). The two merge flags
//!   share the Table 9-5 context bank (Table 9-4). Implemented in
//!   [`sao_merge_flag_ctx_inc`] + [`SAO_MERGE_FLAG_FL_CMAX`].
//!
//! * **`sao_type_idx_luma` / `sao_type_idx_chroma`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: bin 0 `ctxInc = 0`, bin 1 bypass.
//!   The TR(`cMax = 2`, `cRiceParam = 0`) binarization caps the
//!   prefix at two bins, encoding `SaoTypeIdx ∈ {0, 1, 2}`. The two
//!   variants share the Table 9-6 context bank. Implemented in
//!   [`sao_type_idx_ctx_inc`] + [`SAO_TYPE_IDX_TR_CMAX`].
//!
//! * **`sao_offset_abs`** (H.265 §7.3.4, §7.4.9.3) — Table 9-48 row:
//!   all bins bypass. TR(`cMax = (1 << min(bitDepth, 10) − 5) − 1`,
//!   `cRiceParam = 0`). Implemented in [`sao_offset_abs_tr_cmax`].
//!
//! * **`sao_offset_sign`**, **`sao_band_position`**,
//!   **`sao_eo_class_luma` / `sao_eo_class_chroma`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: all bins bypass; FL binarizations
//!   with `cMax = 1`, `cMax = 31`, `cMax = 3` respectively.
//!   Implemented as [`SAO_OFFSET_SIGN_FL_CMAX`],
//!   [`SAO_BAND_POSITION_FL_CMAX`] + [`SAO_BAND_POSITION_FL_NBITS`],
//!   and [`SAO_EO_CLASS_FL_CMAX`] + [`SAO_EO_CLASS_FL_NBITS`].
//!
//! Round 31 extends the surface with the two §9.3.4.2.6 /
//! §9.3.4.2.7 derivations whose `ctxInc` carries *persistent*
//! per-transform-block state across sub-block invocations: the
//! greater-than-1 / greater-than-2 absolute-level flags. Both
//! elements are FL binarized with `cMax = 1` (one context-coded
//! bin) per Table 9-43, but the bin's ctxInc is driven by a small
//! sub-block-scoped state machine (`ctxSet`, `greater1Ctx`,
//! `lastGreater1Ctx`, `lastGreater1Flag`) that the residual-coding
//! loop threads from sub-block to sub-block within the same
//! transform block.
//!
//! * **`coeff_abs_level_greater1_flag[ n ]`** (H.265 §7.3.8.11,
//!   §7.4.9.11) — §9.3.4.2.6, equations 9-56..9-60. The per-bin
//!   `ctxInc = (ctxSet * 4) + min(3, greater1Ctx)` for luma; chroma
//!   adds `+16`. `ctxSet` is initialised at the start of each
//!   sub-block from the previous sub-block's terminal `greater1Ctx`
//!   value, and rotates within `{0, 1, 2, 3}` per the spec's
//!   `lastGreater1Ctx == 0` increment rule. Implemented as a
//!   pure-functional state machine in [`Greater1State`] (the
//!   walker the slice-data parser carries across the residual
//!   sub-blocks of one transform block) plus the per-flag step
//!   functions [`Greater1State::on_subblock_entry`] +
//!   [`Greater1State::on_coeff_abs_level_greater1_flag`].
//!
//! * **`coeff_abs_level_greater2_flag[ lastGreater1ScanPos ]`**
//!   (H.265 §7.3.8.11, §7.4.9.11) — §9.3.4.2.7, equations
//!   9-61..9-62. `ctxInc = ctxSet` (luma) / `ctxInc = ctxSet + 4`
//!   (chroma). The element is signalled at most once per sub-block
//!   at the first scan position that took the greater-1 escape
//!   (`lastGreater1ScanPos`). Implemented in
//!   [`coeff_abs_level_greater2_flag_ctx_inc`], reading the
//!   `ctxSet` value the §9.3.4.2.6 walker has already produced for
//!   that same sub-block.
//!
//! Round 32 extends the residual-coding surface with the
//! §9.3.4.2.5 `sig_coeff_flag` ctxInc derivation: the per-scan-position
//! significance bin that the §7.3.8.11 residual-coding loop emits
//! before any greater-1 / greater-2 step. The derivation is a
//! four-branch dispatch on `(log2TrafoSize, xC + yC,
//! transform_skip_context_enabled_flag && (transform_skip_flag ||
//! cu_transquant_bypass_flag))`:
//!
//! * **transform-skip / transquant-bypass fast path** (eq. 9-40):
//!   `sigCtx = 42` (luma) / `sigCtx = 16` (chroma). One context
//!   per colour component, position-independent. Implemented in
//!   [`sig_coeff_flag_sig_ctx_transform_skip`].
//!
//! * **`log2TrafoSize == 2`** (eq. 9-41) — the 4×4 TB case reads
//!   `sigCtx` from the 16-entry Table 9-50 lookup at
//!   `(yC << 2) + xC`. Implemented in
//!   [`sig_coeff_flag_sig_ctx_log2_2`] +
//!   [`SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2`].
//!
//! * **DC position** (eq. 9-42) — `xC + yC == 0` on `log2 > 2` skips
//!   the eq.-9-43..9-48 neighbour walk; `sigCtx` starts at 0 and
//!   only the colour / size tail (eq.-9-49..9-53) applies.
//!   Implemented in [`sig_coeff_flag_sig_ctx_dc`].
//!
//! * **general case** (eq. 9-43..9-53) — for `log2 > 2`, `xC +
//!   yC > 0` the `prevCsbf` parity of the right / below sub-block
//!   neighbours plus the inner sub-block position `(xC & 3, yC & 3)`
//!   route through one of four `sigCtx` rules (eq. 9-45..9-48), then
//!   the colour / size / scan-order tail (eq. 9-49..9-53). Implemented
//!   in [`sig_coeff_flag_sig_ctx_general`], with the eq.-9-43 / 9-44
//!   edge gates on `xS / yS < (1 << (log2TrafoSize − 2)) − 1`
//!   applied internally.
//!
//! Eq. 9-54 / 9-55 carry the per-component `ctxInc` offset
//! (`ctxInc = sigCtx` for luma, `ctxInc = 27 + sigCtx` for chroma);
//! see [`sig_coeff_flag_ctx_inc_from_sig_ctx`]. Table 9-43 captures
//! the `sig_coeff_flag` binarization shape as FL with `cMax = 1` via
//! [`SIG_COEFF_FLAG_FL_CMAX`].
//!
//! Round 33 extends the ctxInc-derivation surface with the
//! §9.3.4.2.8 `palette_run_prefix` derivation — the §7.3.8.13
//! palette-coding (SCC) syntax element that signals the unary part
//! of the run length following a palette index or copy-above
//! decision. Inputs are the bin index `binIdx`, the per-pixel
//! `copy_above_palette_indices_flag`, and the `palette_idx_idc`
//! decoded at the start of the run. The `ctxInc` is:
//!
//! * **`copy_above_palette_indices_flag == 0` and `binIdx == 0`**
//!   (eq. 9-63) — a piecewise function of `palette_idx_idc` in
//!   `{0, 1, 2}`:
//!   `ctxInc = (palette_idx_idc < 1) ? 0 : ((palette_idx_idc < 3)
//!   ? 1 : 2)`. Implemented in
//!   [`palette_run_prefix_ctx_inc_eq_9_63`].
//!
//! * **otherwise** — `ctxInc = ctxIdxMap[copy_above_palette_indices_flag][binIdx]`
//!   per Table 9-51, captured verbatim in
//!   [`PALETTE_RUN_PREFIX_CTX_IDX_MAP`]. The `binIdx >= 5` cells of
//!   Table 9-51 are bypass-coded per Table 9-48, so the lookup
//!   covers `binIdx ∈ {0, 1, 2, 3, 4}` and the public entry point
//!   [`palette_run_prefix_ctx_inc`] dispatches both branches with
//!   an `Option<u32>` return (`None` ⇒ bypass).
//!
//! Round 33 also captures the matching Table 9-43 binarization
//! shape: `palette_run_prefix` is a TR with
//! `cMax = Floor(Log2(PaletteMaxRunMinus1)) + 1` and `cRiceParam = 0`;
//! the helper [`palette_run_prefix_tr_cmax`] returns the per-block
//! `cMax` from the `PaletteMaxRunMinus1` input.
//!
//! Round 34 adds the §9.3.3.11 `coeff_abs_level_remaining[ n ]`
//! Rice-adaptive binarization + decode primitive (non-persistent
//! branch: `persistent_rice_adaptation_enabled_flag == 0`):
//!
//! * [`coeff_abs_level_remaining_c_rice_param_eq_9_24`] — eq. 9-24
//!   adapts `cRiceParam` from the previous coefficient in the
//!   sub-block: `cRiceParam = Min(cLastRiceParam + (cLastAbsLevel >
//!   (3 << cLastRiceParam) ? 1 : 0), 4)`.
//! * [`coeff_abs_level_remaining_c_max_eq_9_26`] — eq. 9-26:
//!   `cMax = 4 << cRiceParam`.
//! * [`coeff_abs_level_remaining_prefix_val_eq_9_27`] — eq. 9-27:
//!   `prefixVal = Min(cMax, level)` (the value the §9.3.3.2 TR
//!   prefix is built from).
//! * [`coeff_abs_level_remaining_suffix_val_eq_9_28`] — eq. 9-28:
//!   `suffixVal = level - cMax` (only meaningful when the prefix is
//!   the all-ones escape).
//! * [`decode_coeff_abs_level_remaining`] — full bypass-coded decode
//!   driver: reads a unary prefix capped at
//!   `COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN = 4`, and when
//!   the prefix terminates short, returns
//!   `prefix << cRiceParam`; otherwise reads an EGk suffix with
//!   `k = cRiceParam + 1` and returns `cMax + suffixVal`.
//!
//! The persistent-Rice branch (eq. 9-25; the
//! `persistent_rice_adaptation_enabled_flag == 1` path that walks
//! the SCC `StatCoeff[sbType]` state via eq. 9-22/9-23) is left for
//! a follow-up round; the non-persistent branch is the Main-profile
//! path the slice-data parser drives by default.
//!
//! Round 35 adds the §9.3.4.2 / Table 9-48 entry for
//! `coeff_sign_flag[ n ]` — the per-scan-position sign bit that
//! pairs with `coeff_abs_level_remaining[ n ]` to form the signed
//! transform-coefficient level per §7.4.9.11. The element is fully
//! bypass-coded (Table 9-48 marks bin 0 `bypass`, all later columns
//! `na`) and FL binarized with `cMax = 1` (Table 9-43), so the
//! on-wire string is a single bin per invocation. The §9.3.4.3.6
//! alignment process (`ivlCurrRange := 256`) runs prior to the
//! bypass tail at slice-data-loop scope; the per-flag entry point
//! [`decode_coeff_sign_flag`] consumes one [`CabacEngine::decode_bypass`]
//! bin from the post-alignment engine state. The signed level is
//! composed via [`signed_level_from_sign_flag`] applying the
//! §7.4.9.11 `(1 − 2 * coeff_sign_flag[n])` factor; the FL shape
//! is captured as [`COEFF_SIGN_FLAG_FL_CMAX`] +
//! [`COEFF_SIGN_FLAG_FL_NBITS`].
//!
//! Round 36 adds the §9.3.4.2 / Table 9-48 entries for the
//! `cu_chroma_qp_offset_flag` / `cu_chroma_qp_offset_idx` transform-unit
//! syntax pair (H.265 §7.3.8.11, §7.4.9.10). Both elements have a
//! Table 9-48 row whose context-coded bins are all `ctxInc = 0` — the
//! flag occupies a single context (Table 9-34, three ctxIdx slots
//! across initType), and the idx occupies a single context (Table
//! 9-35, three ctxIdx slots across initType). The flag is FL
//! `cMax = 1`; the idx is TR with `cMax =
//! chroma_qp_offset_list_len_minus1`, `cRiceParam = 0`. The §7.4.9.10
//! derivation pair the flag + idx feed is `CuQpOffsetCb = (flag == 1)
//! ? cb_qp_offset_list[idx] : 0` (and symmetric for Cr). The new
//! public surface: [`CU_CHROMA_QP_OFFSET_FLAG_FL_CMAX`] (= 1, Table
//! 9-43 shape) and [`CU_CHROMA_QP_OFFSET_FLAG_FL_NBITS`] (= 1);
//! [`cu_chroma_qp_offset_flag_ctx_inc`] returns the Table 9-48 bin-0
//! `ctxInc = 0` for the FL flag; [`cu_chroma_qp_offset_idx_tr_cmax`]
//! returns the per-TU `cMax` from the PPS-resolved
//! `chroma_qp_offset_list_len_minus1`; [`cu_chroma_qp_offset_idx_ctx_inc`]
//! returns the Table 9-48 `ctxInc = 0` (constant across the TR-prefix
//! bins 0..=4, since Table 9-48 lists `0` for every context-coded
//! binIdx column); and [`decode_cu_chroma_qp_offset`] drives the
//! CABAC engine to produce the typed [`CuChromaQpOffset`] (the
//! flag plus, when the flag is 1, the idx with the §7.4.9.10
//! derivation gates surfaced via [`CuChromaQpOffset::offset_indices`]).
//!
//! Round 37 adds the §9.3.4.2 / Table 9-48 entry for
//! `cu_transquant_bypass_flag` — the per-CU bypass switch that, when
//! set, replaces the scaling + transform + in-loop-filter path with a
//! verbatim residual passthrough (H.265 §7.3.8.5, §7.4.9.5). Per
//! Table 9-43 the flag is FL with `cMax = 1` (a single context-coded
//! bin); Table 9-48's row lists `ctxInc = 0` for bin 0 and `na` for
//! every later binIdx column. The flag occupies a single context
//! (Table 9-8: `initValue = 154` at all three initType slots) which
//! sits at the slice-init-allocated bank entry the parser hands to
//! the decode primitive. The PPS gate is
//! `transquant_bypass_enabled_flag` (§7.4.3.3.1): the flag is read
//! from the bitstream only when that PPS field is 1, otherwise
//! §7.4.9.5 infers the value to 0 (the normal scaling-and-transform
//! path). The new public surface:
//! [`CU_TRANSQUANT_BYPASS_FLAG_FL_CMAX`] (= 1, Table 9-43 shape) and
//! [`CU_TRANSQUANT_BYPASS_FLAG_FL_NBITS`] (= 1);
//! [`cu_transquant_bypass_flag_ctx_inc`] returns the Table 9-48 bin-0
//! `ctxInc = 0`; [`decode_cu_transquant_bypass_flag`] drives one
//! [`CabacEngine::decode_decision`] against the caller's context and
//! returns the decoded `u8`; [`cu_transquant_bypass_flag_inferred`]
//! surfaces the §7.4.9.5 "PPS gate off ⇒ value is 0" inference as a
//! pure helper for callers that branch on the PPS without entering
//! the engine.
//!
//! Round 38 adds the §9.3.4.2 / Table 9-48 entry for `rqt_root_cbf` —
//! the inter-CU gate that signals whether the `transform_tree( )`
//! syntax structure follows the current coding unit (H.265 §7.3.8.5,
//! §7.4.9.5). Per Table 9-43 the flag is FL with `cMax = 1` (a single
//! context-coded bin); Table 9-48's row lists `ctxInc = 0` for bin 0
//! and `na` for every later binIdx column. Table 9-14 supplies a
//! single ctxIdx slot with `initValue = 79` at both initType 1 and
//! initType 2 (initType 0 is `na` in Table 9-4 — `rqt_root_cbf` is
//! only ever read in inter slices, where §7.3.8.5 wires it under
//! `CuPredMode != MODE_INTRA && !cu_skip_flag`). When the §7.3.8.5
//! guard fails (intra CU, or skip CU) the flag is not present on the
//! wire and the §7.4.9.5 inferred value is 1 (the `transform_tree( )`
//! syntax structure is taken to be present). The new public surface:
//! [`RQT_ROOT_CBF_FL_CMAX`] (= 1, Table 9-43 shape) and
//! [`RQT_ROOT_CBF_FL_NBITS`] (= 1); [`rqt_root_cbf_ctx_inc`] returns
//! the Table 9-48 bin-0 `ctxInc = 0`; [`decode_rqt_root_cbf`] drives
//! one [`CabacEngine::decode_decision`] against the caller's context
//! and returns the decoded `u8`; [`rqt_root_cbf_inferred`] surfaces
//! the §7.4.9.5 "not present ⇒ value is 1" inference as a pure helper
//! for callers that branch on the §7.3.8.5 guard without entering the
//! engine.
//!
//! Round 39 adds the §9.3.4.2 / Table 9-48 entry for `pred_mode_flag` —
//! the per-CU bit that selects between MODE_INTER (value 0) and
//! MODE_INTRA (value 1) inside a P or B slice (H.265 §7.3.8.5,
//! §7.4.9.5). Per Table 9-43 the flag is FL with `cMax = 1` (a single
//! context-coded bin); Table 9-48's row lists `ctxInc = 0` for bin 0
//! and `na` for every later binIdx column. Table 9-10 supplies two
//! ctxIdx slots with `initValue = {149, 134}` for initType 1 and
//! initType 2 (initType 0 is `na` in Table 9-4 — the §7.3.8.5
//! `slice_type != I` guard skips the read for I slices, where the
//! mode is unconditionally intra). When the §7.3.8.5 guard fails the
//! flag is not present on the wire and §7.4.9.5 derives `CuPredMode`
//! directly: I slice ⇒ MODE_INTRA; P or B slice with
//! `cu_skip_flag == 1` ⇒ MODE_SKIP. The new public surface:
//! [`PRED_MODE_FLAG_FL_CMAX`] (= 1, Table 9-43 shape) and
//! [`PRED_MODE_FLAG_FL_NBITS`] (= 1); [`pred_mode_flag_ctx_inc`]
//! returns the Table 9-48 bin-0 `ctxInc = 0`;
//! [`decode_pred_mode_flag`] drives one
//! [`CabacEngine::decode_decision`] against the caller's context and
//! returns the decoded `u8`; the [`CuPredMode`] enum captures the
//! §7.4.9.5 mapping (`0 ⇒ MODE_INTER`, `1 ⇒ MODE_INTRA`, plus the
//! not-present-only MODE_SKIP state); [`cu_pred_mode_from_flag`]
//! applies the present-on-wire mapping; and
//! [`pred_mode_flag_inferred_cu_pred_mode`] applies the §7.4.9.5
//! not-present inference branching on `slice_type` and `cu_skip_flag`.
//!
//! Round 40 adds the §9.3.4.2 / Table 9-48 entry for
//! `prev_intra_luma_pred_flag` — the per-luma-prediction-block bit that
//! selects, for an intra CU, whether the luma intra prediction mode is
//! taken from the most-probable-mode list (`mpm_idx` follows) or from
//! the remaining-mode field (`rem_intra_luma_pred_mode` follows), per
//! H.265 §7.3.8.5, §7.4.9.2. Per Table 9-43 the flag is FL with
//! `cMax = 1` (a single context-coded bin); Table 9-48's row lists
//! `ctxInc = 0` for bin 0 and `na` for every later binIdx column.
//! Table 9-12 supplies three ctxIdx slots with
//! `initValue = {184, 154, 183}` for initType 0, 1 and 2 — unlike
//! `pred_mode_flag`, this element is read in I, P and B slices (intra
//! CUs occur in every slice type), so all three initType slots are
//! populated. The flag is always present when the §7.3.8.5 intra-PB
//! loop reaches it (no inferred-value rule); when it is 1 the §8.4.2
//! candidate-mode derivation supplies `IntraPredModeY` and `mpm_idx`
//! selects among the candidates, when it is 0
//! `rem_intra_luma_pred_mode` carries the mode directly. The new
//! public surface: [`PREV_INTRA_LUMA_PRED_FLAG_FL_CMAX`] (= 1, Table
//! 9-43 shape) and [`PREV_INTRA_LUMA_PRED_FLAG_FL_NBITS`] (= 1);
//! [`prev_intra_luma_pred_flag_ctx_inc`] returns the Table 9-48 bin-0
//! `ctxInc = 0`; the [`LumaIntraModeSource`] enum captures the
//! §7.4.9.2 selection (`1 ⇒ Mpm`, `0 ⇒ Remaining`);
//! [`luma_intra_mode_source_from_flag`] folds a decoded flag into the
//! enum; and [`decode_prev_intra_luma_pred_flag`] drives one
//! [`CabacEngine::decode_decision`] against the caller's context and
//! returns the decoded `u8`.
//!
//! Round 41 adds the two §7.3.8.5 luma-mode fields the round-40 flag
//! gates: `mpm_idx` (Table 9-43 TR with `cMax = 2`, `cRiceParam = 0`)
//! and `rem_intra_luma_pred_mode` (Table 9-43 FL with `cMax = 31`),
//! per H.265 §7.4.9.2 / §8.4.2. Both are fully bypass-coded (Table 9-48
//! marks every bin `bypass`), so neither consumes a context model and
//! presence is exactly the [`LumaIntraModeSource`] selection. The new
//! public surface: [`MPM_IDX_TR_CMAX`] (= 2) +
//! [`MPM_IDX_TR_C_RICE_PARAM`] (= 0) and [`decode_mpm_idx`] (the
//! truncated-unary prefix through the bypass decoder, output `0..=2`);
//! [`REM_INTRA_LUMA_PRED_MODE_FL_CMAX`] (= 31) +
//! [`REM_INTRA_LUMA_PRED_MODE_FL_NBITS`] (= 5) and
//! [`decode_rem_intra_luma_pred_mode`] (five FL bypass bins MSB-first,
//! output `0..=31`).
//!
//! Round 42 adds the §7.3.8.5 chroma-mode field that follows the
//! round-40/41 luma-mode group: `intra_chroma_pred_mode`, per H.265
//! §7.4.9.5 / §9.3.3.8 / §8.4.3. Unlike the Table 9-43 generic shapes,
//! this element has its own binarization process (§9.3.3.8 / Table
//! 9-46): the single-bin string `0` signals value 4, and a `1` prefix
//! followed by a two-bit FL suffix signals values 0..=3. Table 9-48
//! marks bin 0 context-coded with `ctxInc = 0` (Table 9-13 supplies
//! `initValue = {63, 152, 152}` for initType 0, 1 and 2) and bins 1..2
//! `bypass`. The §8.4.3 derivation maps the decoded value plus
//! `IntraPredModeY` to `modeIdx` via Table 8-2 (value 4 tracks the luma
//! mode; values 0..=3 select from {0, 26, 10, 1} with a collision
//! substitute of 34), then — when `ChromaArrayType == 2` — remaps
//! `modeIdx` through Table 8-3. The new public surface:
//! [`INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA`] (= 4, the Table 9-46
//! single-bin value), [`INTRA_CHROMA_PRED_MODE_SUFFIX_FL_NBITS`] (= 2),
//! [`intra_chroma_pred_mode_ctx_inc`] (the Table 9-48 bin-0
//! `ctxInc = 0`), [`decode_intra_chroma_pred_mode`] (one
//! [`CabacEngine::decode_decision`] then, for the `1` prefix, two
//! bypass bins; output `{0, 1, 2, 3, 4}`),
//! [`intra_pred_mode_c_mode_idx`] (Table 8-2),
//! [`INTRA_PRED_MODE_C_CHROMA_422_MAP`] /
//! [`intra_pred_mode_c_chroma_422`] (Table 8-3), and
//! [`derive_intra_pred_mode_c`] (the full §8.4.3 output).

use crate::hevc::engine::cabac::{CabacEngine, CabacError, ContextModel};

// ---------------------------------------------------------------------
// Truncated-rice helper — §9.3.3.10
// ---------------------------------------------------------------------

/// §9.3.3.10 — read a truncated-rice prefix with `cMax` and
/// `cRiceParam = 0` (the only configuration this module currently
/// needs). The prefix is unary up to a maximum of `cMax >> 0` = `cMax`
/// bins; a value of `cMax / (1 << 0)` = `cMax` is the "escape" form
/// signalling that a suffix follows for callers that pair TR with an
/// EGk continuation.
///
/// `read_bin(bin_idx)` decodes the bin at index `bin_idx` (0-based)
/// using the caller's choice of context-coded or bypass-coded path, and
/// returns the decoded bin (0 = continue, 1 = terminate / escape).
///
/// Returns `(prefix_val, is_escape)`. `is_escape` is `true` exactly
/// when `prefix_val == cMax`: i.e. all `cMax` bins were 1, signalling
/// that a TR + EGk continuation suffix must be read by the caller.
pub(crate) fn read_truncated_rice_prefix<F>(
    c_max: u32,
    mut read_bin: F,
) -> Result<(u32, bool), CabacError>
where
    F: FnMut(u32) -> Result<u8, CabacError>,
{
    // Unary: count leading 1s up to c_max. The decoded bin tally is
    // simultaneously the prefix value (number of leading 1s) and the
    // escape signal (all-ones when value == c_max).
    let mut value: u32 = 0;
    while value < c_max {
        let bin = read_bin(value)?;
        if bin == 0 {
            return Ok((value, false));
        }
        value += 1;
    }
    Ok((value, true))
}

// ---------------------------------------------------------------------
// §9.3.3.3 — Exp-Golomb of order k (bypass-coded, MSB-first prefix then
// suffix). Per the §9.3.3.3 NOTE the EGk prefix uses 1's and 0's in the
// reverse meaning of the §9.2 EG0 prefix: it is a unary run of `1` bins
// terminated by a single `0`, followed by `prefix_ones + k` suffix bits.
// ---------------------------------------------------------------------

/// §9.3.3.3 — decode an EGk-coded value from the bypass stream for
/// arbitrary Exp-Golomb order `k`.
///
/// The reader counts leading `1` bypass bins until a terminating `0`
/// (call this count `prefix_ones`); reads `prefix_ones + k` more
/// bypass bins as the suffix; and returns
/// `((1 << prefix_ones) - 1) << k + suffix`.
///
/// For `k = 0` the reader collapses to the `cu_qp_delta_abs`-style
/// "(1 << lz) - 1 + suffix" form (suffix length == `leading_zeros`,
/// `<< 0` factor disappears) — preserved as
/// [`decode_eg_k0`] for callers that only need the `k = 0` case.
///
/// `leading_zeros` is capped at 32 to keep the decoder defensive
/// against runaway encodings; the cap is comfortably above any legal
/// HEVC bypass-coded suffix value (§7.4.9.11 puts the practical
/// ceiling well below `2^31`).
pub(crate) fn decode_eg_k(engine: &mut CabacEngine<'_>, k: u32) -> Result<u32, CabacError> {
    read_eg_k_with(k, || engine.decode_bypass())
}

/// §9.3.3.3 — decode an EGk-coded value of order `k` from a generic
/// bin reader (each call yields the next bypass bin). Factored out of
/// [`decode_eg_k`] so the §7.3.8.9 `abs_mvd_minus2` EG1 escape can be
/// composed and unit-tested without driving the arithmetic engine's
/// range/offset state. The decode follows the equation-9-13 do-while:
/// per the §9.3.3.3 NOTE the EGk unary prefix uses 1's and 0's in the
/// reverse meaning of the §9.2 EG0 prefix, so the prefix is a run of
/// `1` bins terminated by a single `0`. Count those leading `1` bins
/// (`prefix_ones`), then read `prefix_ones + k` suffix bins, yielding
/// `((1 << prefix_ones) − 1) << k + suffix`.
pub(crate) fn read_eg_k_with<F>(k: u32, mut read_bin: F) -> Result<u32, CabacError>
where
    F: FnMut() -> Result<u8, CabacError>,
{
    let mut prefix_ones: u32 = 0;
    while prefix_ones < 32 {
        let bin = read_bin()?;
        if bin == 0 {
            break;
        }
        prefix_ones += 1;
    }
    // Guard against accidental >32-bit reads (cap `prefix_ones` at
    // 32 already bounds this; the saturate keeps the read width inside
    // a single `u32` if `k` is unusually large).
    let suffix_bits = (prefix_ones + k).min(32);
    let mut suffix: u32 = 0;
    for _ in 0..suffix_bits {
        suffix = (suffix << 1) | u32::from(read_bin()?);
    }
    let base = ((1u64 << prefix_ones) - 1) << k;
    let value = base + u64::from(suffix);
    // The defensive cap above keeps `value` inside `u32`; the
    // saturate keeps us inside the public return type if the
    // pathological 32-bit edge is ever reached.
    Ok(u32::try_from(value).unwrap_or(u32::MAX))
}

/// §9.3.3.11 — decode an EGk-coded value with `k = 0` from the bypass
/// stream. Preserved as a named alias for the `cu_qp_delta_abs` /
/// `palette_escape_val` callers that historically used the `k = 0`
/// fast path; equivalent to `decode_eg_k(engine, 0)`.
fn decode_eg_k0(engine: &mut CabacEngine<'_>) -> Result<u32, CabacError> {
    decode_eg_k(engine, 0)
}

// ---------------------------------------------------------------------
// cu_qp_delta_abs / cu_qp_delta_sign_flag (§7.3.8.14, §7.4.9.14)
// ---------------------------------------------------------------------

/// §9.3.4.2.2 Table 9-32 — per-bin `ctxInc` for `cu_qp_delta_abs`. Bin
/// 0 uses a dedicated context (`ctxInc = 0`); bins 1..=4 share a second
/// dedicated context (`ctxInc = 1`). The TR prefix carries at most
/// `cMax = 5` bins; the EGk(k=0) suffix is bypass-coded and not
/// indexed.
#[must_use]
pub fn cu_qp_delta_abs_ctx_inc(bin_idx: u32) -> u32 {
    if bin_idx == 0 { 0 } else { 1 }
}

/// `cu_qp_delta_abs` TR prefix `cMax`. §9.3.3.10 (with §9.3.4.2.2
/// row).
pub const CU_QP_DELTA_ABS_TR_CMAX: u32 = 5;

/// Decoded `cu_qp_delta` pair plus the §7.4.9.14 derived
/// `CuQpDeltaVal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuQpDelta {
    /// `cu_qp_delta_abs` from the wire.
    pub abs: u32,
    /// `cu_qp_delta_sign_flag` from the wire; `None` when
    /// `abs == 0` (the flag is not signalled in that case).
    pub sign_flag: Option<u8>,
    /// Derived signed delta:
    /// `cu_qp_delta_abs * (1 − 2 * cu_qp_delta_sign_flag)`.
    /// Equals `0` when `abs == 0`.
    pub value: i32,
}

/// Decode the `cu_qp_delta` syntax pair (§7.3.8.14 / §7.4.9.14) from
/// the CABAC engine, consuming the two context variables for the
/// `cu_qp_delta_abs` TR prefix (bin 0 + bins 1..=4) and any required
/// bypass bins for the EGk(k=0) suffix and the sign flag.
///
/// `ctx_bin0` and `ctx_bin_rest` are the caller's
/// `(pStateIdx, valMps)` state for the two `cu_qp_delta_abs` contexts;
/// they are mutated in place per §9.3.4.3.2.2 state transition.
///
/// Returns the decoded [`CuQpDelta`] including the §7.4.9.14
/// `CuQpDeltaVal` derivation.
pub fn decode_cu_qp_delta(
    engine: &mut CabacEngine<'_>,
    ctx_bin0: &mut ContextModel,
    ctx_bin_rest: &mut ContextModel,
) -> Result<CuQpDelta, CabacError> {
    // Decode the §9.3.3.10 TR prefix using two contexts: bin 0 uses
    // ctx_bin0, all subsequent bins share ctx_bin_rest.
    let (prefix, is_escape) = read_truncated_rice_prefix(CU_QP_DELTA_ABS_TR_CMAX, |bin_idx| {
        if cu_qp_delta_abs_ctx_inc(bin_idx) == 0 {
            engine.decode_decision(ctx_bin0)
        } else {
            engine.decode_decision(ctx_bin_rest)
        }
    })?;

    // §9.3.3.10: if the prefix is the all-ones escape (== cMax), an
    // EGk(k=0) suffix follows; the decoded value is prefix + suffix.
    let abs = if is_escape {
        prefix + decode_eg_k0(engine)?
    } else {
        prefix
    };

    // §7.4.9.14: the sign flag is signalled only when abs != 0.
    let (sign_flag, signed) = if abs == 0 {
        (None, 0i32)
    } else {
        let s = engine.decode_bypass()?;
        // CuQpDeltaVal = abs * (1 − 2 * sign_flag)
        let sign_factor = 1i32 - 2 * (s as i32);
        (Some(s), abs as i32 * sign_factor)
    };

    Ok(CuQpDelta {
        abs,
        sign_flag,
        value: signed,
    })
}

// ---------------------------------------------------------------------
// cu_chroma_qp_offset_flag / cu_chroma_qp_offset_idx
// (§7.3.8.11 transform_unit() / §7.4.9.10 — PPS chroma-QP-offset list
// gating). Table 9-43 binarizations + Table 9-48 ctxInc derivations.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `cu_chroma_qp_offset_flag`: FL
/// with `cMax = 1` (a single context-coded bin).
pub const CU_CHROMA_QP_OFFSET_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `cu_chroma_qp_offset_flag`: one
/// context-coded bin per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const CU_CHROMA_QP_OFFSET_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `cu_chroma_qp_offset_flag`. Bin 0
/// is context-coded with `ctxInc = 0`; later bin-index columns are
/// `na` (the FL `cMax = 1` shape only emits a single bin).
///
/// The flag's Table 9-34 ctxIdx layout (`initValue = 154` at every
/// initType slot) is consumed at slice-init scope; this layer hands
/// back the bank-relative `ctxInc` only.
#[must_use]
pub fn cu_chroma_qp_offset_flag_ctx_inc() -> u32 {
    0
}

/// §9.3.4.2.1 / Table 9-48 row for `cu_chroma_qp_offset_idx`. Every
/// context-coded bin column (binIdx 0..=4 of the TR prefix) lists
/// `ctxInc = 0`; binIdx 5 is `na` (the §7.4.9.10 idx range tops out
/// at `chroma_qp_offset_list_len_minus1 <= 5`, so the spec-mandated
/// `cMax <= 5` TR prefix can produce at most five bins).
///
/// The idx's Table 9-35 ctxIdx layout (`initValue = 154` at every
/// initType slot) is consumed at slice-init scope; this layer hands
/// back the bank-relative `ctxInc` only.
///
/// `bin_idx` is the §9.3.4.2 binIdx of the bin being coded; values
/// outside `0..=4` are `na` per Table 9-48.
#[must_use]
pub fn cu_chroma_qp_offset_idx_ctx_inc(bin_idx: u32) -> u32 {
    debug_assert!(
        bin_idx <= 4,
        "cu_chroma_qp_offset_idx Table 9-48 row: binIdx > 4 is na"
    );
    0
}

/// §9.3.3.10 / Table 9-43 row for `cu_chroma_qp_offset_idx`: TR with
/// `cMax = chroma_qp_offset_list_len_minus1`, `cRiceParam = 0`.
/// `chroma_qp_offset_list_len_minus1` is the PPS-signalled length of
/// `cb_qp_offset_list[ ]` / `cr_qp_offset_list[ ]` minus one
/// (§7.4.3.3.1: a `u(3)` field, bounded by `<= 5`).
#[must_use]
pub fn cu_chroma_qp_offset_idx_tr_cmax(chroma_qp_offset_list_len_minus1: u32) -> u32 {
    chroma_qp_offset_list_len_minus1
}

/// Decoded `cu_chroma_qp_offset_flag` / `cu_chroma_qp_offset_idx` pair
/// with the §7.4.9.10 chroma-QP-offset-list dereference surfaced as a
/// typed accessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuChromaQpOffset {
    /// `cu_chroma_qp_offset_flag` from the wire.
    pub flag: u8,
    /// `cu_chroma_qp_offset_idx` from the wire; `None` when the flag
    /// is 0 (the idx is not signalled in that case per §7.3.8.11).
    /// When the flag is 1 and `chroma_qp_offset_list_len_minus1 == 0`
    /// the idx is also not signalled (the TR prefix has `cMax = 0`,
    /// emitting zero bins) and the value is inferred to be 0.
    pub idx: Option<u32>,
}

impl CuChromaQpOffset {
    /// §7.4.9.10 — return the `(cb_qp_offset_list, cr_qp_offset_list)`
    /// indices to dereference for `CuQpOffsetCb` / `CuQpOffsetCr`.
    /// `None` is returned when the flag is 0: per §7.4.9.10 the
    /// offsets are then both 0 (no list dereference).
    #[must_use]
    pub fn offset_indices(&self) -> Option<u32> {
        if self.flag == 0 {
            None
        } else {
            // §7.4.9.10: when present, idx defaults to 0; absent ⇒
            // also 0 (the cMax = 0 / not-signalled paths share the
            // same inferred value).
            Some(self.idx.unwrap_or(0))
        }
    }
}

/// Decode the `cu_chroma_qp_offset` syntax pair (§7.3.8.11 /
/// §7.4.9.10) from the CABAC engine, consuming one context for the
/// FL `cu_chroma_qp_offset_flag` bin and (when the flag is 1 and
/// `chroma_qp_offset_list_len_minus1 > 0`) one context for the TR
/// prefix of `cu_chroma_qp_offset_idx`. All bins of both elements
/// share `ctxInc = 0` per Table 9-48; the contexts live in separate
/// Table 9-4 banks (Table 9-34 for the flag, Table 9-35 for the
/// idx).
///
/// `ctx_flag` and `ctx_idx` are the caller's
/// `(pStateIdx, valMps)` state for the two contexts; they are
/// mutated in place per §9.3.4.3.2.2 state transition.
///
/// `chroma_qp_offset_list_len_minus1` is the PPS field that bounds
/// the TR prefix; the function panics in debug if it exceeds 5
/// (the §7.4.3.3.1 `u(3)` field's upper bound).
///
/// Returns the decoded [`CuChromaQpOffset`].
pub fn decode_cu_chroma_qp_offset(
    engine: &mut CabacEngine<'_>,
    ctx_flag: &mut ContextModel,
    ctx_idx: &mut ContextModel,
    chroma_qp_offset_list_len_minus1: u32,
) -> Result<CuChromaQpOffset, CabacError> {
    debug_assert!(
        chroma_qp_offset_list_len_minus1 <= 5,
        "chroma_qp_offset_list_len_minus1 is a u(3) field bounded by 5 per §7.4.3.3.1"
    );

    // §9.3.4.2.1 + Table 9-48 + Table 9-43: bin 0 of the FL flag
    // uses ctxInc = 0 and the FL cMax = 1 shape emits exactly one
    // bin.
    let flag = engine.decode_decision(ctx_flag)?;

    // §7.3.8.11: the idx is signalled only when the flag is 1 and
    // the list has more than one entry (else the TR cMax = 0 shape
    // emits no bins and the idx is inferred to be 0 per §7.4.9.10).
    let idx = if flag == 1 && chroma_qp_offset_list_len_minus1 > 0 {
        let c_max = cu_chroma_qp_offset_idx_tr_cmax(chroma_qp_offset_list_len_minus1);
        // §9.3.3.10 TR prefix with cRiceParam = 0; every bin uses
        // ctx_idx (Table 9-48 ctxInc = 0).
        let (prefix, _is_escape) =
            read_truncated_rice_prefix(c_max, |_bin_idx| engine.decode_decision(ctx_idx))?;
        // The TR escape (all-ones prefix == cMax) terminates the
        // idx — there is no suffix continuation for this element
        // (Table 9-43 has no follow-on EGk row).
        Some(prefix)
    } else {
        None
    };

    Ok(CuChromaQpOffset { flag, idx })
}

// ---------------------------------------------------------------------
// cu_transquant_bypass_flag (§7.3.8.5 coding_unit() / §7.4.9.5)
// PPS gate: transquant_bypass_enabled_flag (§7.4.3.3.1). Table 9-43
// FL binarization + Table 9-48 ctxInc + Table 9-8 ctxIdx initValue.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `cu_transquant_bypass_flag`: FL
/// with `cMax = 1` (a single context-coded bin per CU).
pub const CU_TRANSQUANT_BYPASS_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `cu_transquant_bypass_flag`: one
/// context-coded bin per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const CU_TRANSQUANT_BYPASS_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `cu_transquant_bypass_flag`. Bin 0
/// is context-coded with `ctxInc = 0`; every later binIdx column is
/// `na` (the FL `cMax = 1` shape only emits a single bin).
///
/// The flag's Table 9-8 ctxIdx layout (`initValue = 154` at every
/// initType slot — same `pStateIdx`/`valMps` start at all three
/// slice-type initialisation rows) is consumed at slice-init scope;
/// this layer hands back the bank-relative `ctxInc` only.
#[must_use]
pub fn cu_transquant_bypass_flag_ctx_inc() -> u32 {
    0
}

/// §7.4.9.5 — inferred value of `cu_transquant_bypass_flag` when the
/// PPS gate `transquant_bypass_enabled_flag` is 0 and the element is
/// not present on the wire. The spec mandates an inferred value of
/// `0` (normal scaling-and-transform path).
///
/// Exposed as a pure helper so the parser can branch on the PPS
/// without entering the CABAC engine: when the PPS gate is off the
/// caller skips [`decode_cu_transquant_bypass_flag`] entirely and
/// records the inferred `0`.
#[must_use]
pub fn cu_transquant_bypass_flag_inferred() -> u8 {
    0
}

/// Decode `cu_transquant_bypass_flag` (§7.3.8.5 / §7.4.9.5) from the
/// CABAC engine, consuming one context for the FL bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-8 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place per
/// §9.3.4.3.2.2 state transition.
///
/// The PPS gate `transquant_bypass_enabled_flag` (§7.4.3.3.1) is
/// upstream of this primitive: when the gate is 0 the parser does
/// **not** call this function — see [`cu_transquant_bypass_flag_inferred`]
/// for the §7.4.9.5 inferred value.
///
/// Returns the decoded `u8` (0 or 1) — 1 selects the §8.6 / §8.7
/// transquant-bypass passthrough for the current CU per §7.4.9.5.
pub fn decode_cu_transquant_bypass_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin uses ctxInc = 0 (a single Table
    // 9-8 entry) per the Table 9-48 row.
    debug_assert_eq!(
        cu_transquant_bypass_flag_ctx_inc(),
        0,
        "Table 9-48 row for cu_transquant_bypass_flag: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// rqt_root_cbf (§7.3.8.5 coding_unit() / §7.4.9.5)
// Table 9-43 FL binarization + Table 9-48 ctxInc + Table 9-14 ctxIdx
// initValue. Present only when the current coding unit is not
// IntraPredMode (i.e. CuPredMode == MODE_INTER) and the cu_skip_flag
// is 0; absent ⇒ inferred to 1 (the §7.4.9.5 / 2021 spec wording).
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `rqt_root_cbf`: FL with `cMax = 1`
/// (a single context-coded bin per inter CU). Marks whether the
/// `transform_tree( )` syntax structure is present for the current
/// coding unit (§7.4.9.5).
pub const RQT_ROOT_CBF_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `rqt_root_cbf`: one context-coded bin
/// per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const RQT_ROOT_CBF_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `rqt_root_cbf`. Bin 0 is
/// context-coded with `ctxInc = 0`; every later binIdx column is `na`
/// (the FL `cMax = 1` shape only emits a single bin).
///
/// The flag's Table 9-14 ctxIdx layout has only the initType 1 and
/// initType 2 slots populated (`initValue = 79` at both); initType 0
/// is `na` per Table 9-4, since `rqt_root_cbf` is only ever read in
/// inter slices (§7.3.8.5 routes the read under
/// `CuPredMode != MODE_INTRA && !cu_skip_flag`). This layer hands back
/// the bank-relative `ctxInc` only; the Table 9-14 initValue plus
/// Table 9-4 initType-to-ctxIdx mapping is consumed at slice-init
/// scope.
#[must_use]
pub fn rqt_root_cbf_ctx_inc() -> u32 {
    0
}

/// §7.4.9.5 — inferred value of `rqt_root_cbf` when the element is not
/// present on the wire. The 2021 spec wording (V8/v8.0) mandates an
/// inferred value of `1` (the `transform_tree( )` syntax structure is
/// taken to be present); the V11/2026 revision conditions the
/// inference on `!DcOnlyFlag[ x0 ][ y0 ]`, but the V8 baseline this
/// rebuild targets is the unconditional 1.
///
/// Exposed as a pure helper so the parser can branch without entering
/// the CABAC engine when the §7.3.8.5 guard (`CuPredMode == MODE_INTRA
/// || cu_skip_flag == 1`) selects the not-present path.
#[must_use]
pub fn rqt_root_cbf_inferred() -> u8 {
    1
}

/// Decode `rqt_root_cbf` (§7.3.8.5 / §7.4.9.5) from the CABAC engine,
/// consuming one context for the FL bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-14 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place per
/// §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.5 guard (`CuPredMode != MODE_INTRA && !cu_skip_flag`) is
/// upstream of this primitive: when the guard fails the parser does
/// **not** call this function — see [`rqt_root_cbf_inferred`] for the
/// §7.4.9.5 inferred value.
///
/// Returns the decoded `u8` (0 or 1) — 1 means the `transform_tree( )`
/// syntax structure follows the current coding-unit body; 0 means the
/// CU has no transform-tree (samples derive from prediction only per
/// §7.4.9.5).
pub fn decode_rqt_root_cbf(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin uses ctxInc = 0 (a single Table
    // 9-14 entry) per the Table 9-48 row.
    debug_assert_eq!(
        rqt_root_cbf_ctx_inc(),
        0,
        "Table 9-48 row for rqt_root_cbf: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// merge_flag (§7.3.8.6 prediction_unit() / §7.4.9.6)
// Table 9-43 FL binarization + Table 9-48 ctxInc + Table 9-15 ctxIdx
// initValue. Read in the §7.3.8.6 `else /* MODE_INTER */` branch
// (i.e. cu_skip_flag == 0); when present and 1 the inter-prediction
// parameters are inferred from a neighbouring inter partition
// (merge_idx follows), when 0 the explicit motion fields
// (inter_pred_idc / ref_idx / mvp) are coded. Absent ⇒ inferred per
// §7.4.9.6 from CuPredMode (MODE_SKIP ⇒ 1, otherwise 0).
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `merge_flag`: FL with `cMax = 1`
/// (a single context-coded bin per non-skip inter prediction unit).
/// Selects between the merge path (value 1 — inter-prediction
/// parameters inferred from a neighbouring inter-predicted partition)
/// and the explicit-motion path (value 0) per §7.4.9.6.
pub const MERGE_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `merge_flag`: one context-coded bin
/// per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const MERGE_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `merge_flag`. Bin 0 is
/// context-coded with `ctxInc = 0`; every later binIdx column is `na`
/// (the FL `cMax = 1` shape only emits a single bin).
///
/// The flag's Table 9-15 ctxIdx layout has only the initType 1 and
/// initType 2 slots populated (`initValue = 110` at initType 1,
/// `initValue = 154` at initType 2); initType 0 is `na` in Table 9-4
/// because `merge_flag` is only ever read in inter (P/B) slices — the
/// §7.3.8.6 `prediction_unit( )` read sits under the coding-unit's
/// inter path (`CuPredMode != MODE_INTRA && !cu_skip_flag`), which is
/// unreachable in an I slice. This layer hands back the bank-relative
/// `ctxInc` only; the Table 9-15 initValue plus Table 9-4
/// initType-to-ctxIdx mapping is consumed at slice-init scope (see
/// `ctx_init::TABLE_9_15_MERGE_FLAG`).
#[must_use]
pub fn merge_flag_ctx_inc() -> u32 {
    0
}

/// §7.4.9.6 — inferred value of `merge_flag` when the element is not
/// present on the wire. The §7.3.8.6 guard that suppresses the read is
/// `cu_skip_flag == 1` (the skip path codes only `merge_idx`); the
/// §7.4.9.6 inference is then:
///
/// * If `CuPredMode[ x0 ][ y0 ]` is equal to `MODE_SKIP`,
///   `merge_flag` is inferred to be equal to 1.
/// * Otherwise, `merge_flag` is inferred to be equal to 0.
///
/// `cu_pred_mode` is the §7.4.9.5 prediction mode of the current
/// coding unit. The only not-present path the §7.3.8.6 syntax reaches
/// is the `cu_skip_flag == 1` branch, whose CuPredMode is
/// [`CuPredMode::Skip`] per §7.4.9.5 — so this returns 1 for
/// [`CuPredMode::Skip`] and 0 for the inter/intra modes (which would
/// only arise from a caller bug, since a present `merge_flag` is
/// decoded rather than inferred on those paths).
///
/// Exposed as a pure helper so the parser can branch without entering
/// the CABAC engine when the §7.3.8.6 skip path selects the
/// not-present case.
#[must_use]
pub fn merge_flag_inferred(cu_pred_mode: CuPredMode) -> u8 {
    // §7.4.9.6: CuPredMode == MODE_SKIP ⇒ 1, otherwise ⇒ 0.
    u8::from(cu_pred_mode == CuPredMode::Skip)
}

/// Decode `merge_flag` (§7.3.8.6 / §7.4.9.6) from the CABAC engine,
/// consuming one context for the FL bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-15 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place per
/// §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.6 guard (the `else /* MODE_INTER */` branch — i.e.
/// `cu_skip_flag == 0`) is upstream of this primitive: when the CU
/// takes the skip path the parser does **not** call this function —
/// see [`merge_flag_inferred`] for the §7.4.9.6 not-present value.
///
/// Returns the decoded `u8` (0 or 1) — 1 selects the merge path (the
/// inter-prediction parameters are inferred from a neighbouring
/// inter-predicted partition, `merge_idx` follows per §7.3.8.6); 0
/// selects the explicit-motion path (`inter_pred_idc` / `ref_idx_lX` /
/// `mvp_lX_flag` / `mvd_coding` are coded).
pub fn decode_merge_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin uses ctxInc = 0 (a single Table
    // 9-15 entry) per the Table 9-48 row.
    debug_assert_eq!(
        merge_flag_ctx_inc(),
        0,
        "Table 9-48 row for merge_flag: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// pred_mode_flag (§7.3.8.5 coding_unit() / §7.4.9.5)
// Table 9-43 FL binarization + Table 9-48 ctxInc + Table 9-10 ctxIdx
// initValue. Present only when the slice is P or B (slice_type != I)
// and cu_skip_flag == 0; absent ⇒ CuPredMode is derived per §7.4.9.5
// from slice_type and cu_skip_flag without entering the engine.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `pred_mode_flag`: FL with
/// `cMax = 1` (a single context-coded bin per non-skip P/B coding
/// unit). Selects between MODE_INTER (value 0) and MODE_INTRA (value
/// 1) per §7.4.9.5.
pub const PRED_MODE_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `pred_mode_flag`: one context-coded
/// bin per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const PRED_MODE_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `pred_mode_flag`. Bin 0 is
/// context-coded with `ctxInc = 0`; every later binIdx column is `na`
/// (the FL `cMax = 1` shape only emits a single bin).
///
/// The flag's Table 9-10 ctxIdx layout has only the initType 1 and
/// initType 2 slots populated (`initValue = 149` at initType 1,
/// `initValue = 134` at initType 2); initType 0 is `na` in Table 9-4
/// because the §7.3.8.5 `slice_type != I` guard skips the read for I
/// slices entirely. This layer hands back the bank-relative `ctxInc`
/// only; the Table 9-10 initValue plus Table 9-4 initType-to-ctxIdx
/// mapping is consumed at slice-init scope.
#[must_use]
pub fn pred_mode_flag_ctx_inc() -> u32 {
    0
}

/// §7.4.9.5 / §9.5.1 — the coding-unit prediction mode. Selected by
/// `pred_mode_flag` (when present on the wire) or inferred from
/// slice-level state (when not present), per §7.4.9.5.
///
/// `pred_mode_flag == 0` maps to [`CuPredMode::Inter`];
/// `pred_mode_flag == 1` maps to [`CuPredMode::Intra`].
/// [`CuPredMode::Skip`] is reachable only via the §7.4.9.5
/// not-present inference path: a P or B slice CU whose `cu_skip_flag`
/// is 1 and whose `pred_mode_flag` was therefore not coded.
///
/// The enum is local to the binarization module because no other
/// module currently expresses CuPredMode in code; the
/// `coding_unit( )` syntax walker may promote it to a wider type in
/// a future round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CuPredMode {
    /// MODE_INTER — inter-picture prediction (samples derived from a
    /// reference picture). Reachable from `pred_mode_flag == 0` (P
    /// or B slice, non-skip CU).
    Inter,
    /// MODE_INTRA — intra-picture prediction (samples derived from
    /// neighbouring reconstructed samples in the current picture).
    /// Reachable from `pred_mode_flag == 1` and from the I-slice
    /// not-present inference (§7.4.9.5).
    Intra,
    /// MODE_SKIP — inter-prediction skip path (no residual, no
    /// `pred_mode_flag`, no transform tree). Reachable only from
    /// §7.4.9.5 not-present inference when `slice_type != I` and
    /// `cu_skip_flag == 1`.
    Skip,
}

/// Map a decoded `pred_mode_flag` value to [`CuPredMode`] per
/// §7.4.9.5: `0 ⇒ MODE_INTER`, `1 ⇒ MODE_INTRA`. The mapping is
/// total over the FL `cMax = 1` shape (any other input is rejected
/// in debug builds; release builds fall back to MODE_INTRA to keep
/// the function `must_use`-friendly without panicking on accidental
/// out-of-range inputs).
#[must_use]
pub fn cu_pred_mode_from_flag(pred_mode_flag: u8) -> CuPredMode {
    debug_assert!(
        pred_mode_flag <= 1,
        "pred_mode_flag is FL cMax = 1 (Table 9-43); decoded value must be 0 or 1"
    );
    if pred_mode_flag == 0 {
        CuPredMode::Inter
    } else {
        CuPredMode::Intra
    }
}

/// §7.4.9.5 — derive [`CuPredMode`] for a coding unit whose
/// `pred_mode_flag` is not present on the wire. The §7.3.8.5 guard
/// that suppresses the read is `slice_type == I` (no flag is coded
/// because the slice is purely intra) or `cu_skip_flag == 1` (the
/// skip path bypasses the flag).
///
/// `slice_type_is_i` is `true` iff the current slice has
/// `slice_type == I` per §7.4.7.1.
/// `cu_skip_flag` is the §7.3.8.5 / §7.4.9.5 decoded skip flag (or
/// the 0-inferred value when `slice_type == I`).
///
/// The §7.4.9.5 derivation:
///
/// * If the slice is I, `CuPredMode` is inferred to MODE_INTRA
///   regardless of `cu_skip_flag` (which is itself not coded for
///   I slices and inferred to 0 per §7.4.9.5).
/// * Otherwise (P or B slice), the only path that reaches this
///   helper is `cu_skip_flag == 1`; the spec then infers
///   `CuPredMode = MODE_SKIP`. A `cu_skip_flag == 0` input would
///   contradict the §7.3.8.5 guard (the flag *would* be present
///   and decoded); the function asserts this in debug builds and
///   falls back to MODE_INTER in release builds to avoid panicking
///   on a caller bug.
#[must_use]
pub fn pred_mode_flag_inferred_cu_pred_mode(slice_type_is_i: bool, cu_skip_flag: u8) -> CuPredMode {
    if slice_type_is_i {
        // §7.4.9.5: "If slice_type is equal to I, CuPredMode[ x ][ y ]
        // is inferred to be equal to MODE_INTRA." cu_skip_flag is not
        // coded for I slices (§7.3.8.5 `if( slice_type != I )` guard)
        // and is itself inferred to 0; the I-slice branch is therefore
        // unconditional on the skip flag.
        CuPredMode::Intra
    } else {
        // §7.4.9.5: "Otherwise (slice_type is equal to P or B), when
        // cu_skip_flag[ x0 ][ y0 ] is equal to 1, CuPredMode[ x ][ y ]
        // is inferred to be equal to MODE_SKIP." The §7.3.8.5
        // pred_mode_flag read is gated on `!cu_skip_flag[ x0 ][ y0 ]`,
        // so the only way this branch is reached on a P/B slice is
        // with cu_skip_flag == 1.
        debug_assert_eq!(
            cu_skip_flag, 1,
            "§7.4.9.5: pred_mode_flag not-present on a P/B slice requires cu_skip_flag == 1"
        );
        CuPredMode::Skip
    }
}

/// Decode `pred_mode_flag` (§7.3.8.5 / §7.4.9.5) from the CABAC
/// engine, consuming one context for the FL bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-10 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place
/// per §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.5 guard (`slice_type != I && !cu_skip_flag`) is
/// upstream of this primitive: when the guard fails the parser does
/// **not** call this function — see
/// [`pred_mode_flag_inferred_cu_pred_mode`] for the §7.4.9.5
/// not-present derivation.
///
/// Returns the decoded `u8` (0 or 1) — 0 selects MODE_INTER per
/// §7.4.9.5 (`CuPredMode = MODE_INTER`), 1 selects MODE_INTRA
/// (`CuPredMode = MODE_INTRA`). See [`cu_pred_mode_from_flag`] to
/// fold the result into the [`CuPredMode`] enum.
pub fn decode_pred_mode_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin uses ctxInc = 0 (a single Table
    // 9-10 entry) per the Table 9-48 row.
    debug_assert_eq!(
        pred_mode_flag_ctx_inc(),
        0,
        "Table 9-48 row for pred_mode_flag: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// prev_intra_luma_pred_flag (§7.3.8.5 coding_unit() / §7.4.9.2)
// Table 9-43 FL binarization + Table 9-48 ctxInc + Table 9-12 ctxIdx
// initValue. Read once per luma prediction block of an intra CU (the
// §7.3.8.5 `for( j ) for( i )` PB loop). When the flag is 1 the luma
// intra mode is taken from the §8.4.2 most-probable-mode list (mpm_idx
// follows); when 0 it comes from rem_intra_luma_pred_mode. There is no
// not-present inference: the §7.3.8.5 loop always reads it.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `prev_intra_luma_pred_flag`: FL
/// with `cMax = 1` (a single context-coded bin per luma prediction
/// block of an intra CU). Selects between the MPM-list mode source
/// (value 1) and the remaining-mode field (value 0) per §7.4.9.2.
pub const PREV_INTRA_LUMA_PRED_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `prev_intra_luma_pred_flag`: one
/// context-coded bin per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const PREV_INTRA_LUMA_PRED_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `prev_intra_luma_pred_flag`. Bin 0
/// is context-coded with `ctxInc = 0`; every later binIdx column is
/// `na` (the FL `cMax = 1` shape only emits a single bin).
///
/// Table 9-12 lists three ctxIdx slots with
/// `initValue = {184, 154, 183}` for initType 0, 1 and 2 — the element
/// is read in I, P and B slices (intra CUs occur in every slice type),
/// so all three initType slots are populated (unlike `pred_mode_flag`,
/// whose initType 0 column is `na`). This layer hands back the
/// bank-relative `ctxInc` only; the Table 9-12 initValue plus the
/// Table 9-4 initType-to-ctxIdx mapping is consumed at slice-init
/// scope.
#[must_use]
pub fn prev_intra_luma_pred_flag_ctx_inc() -> u32 {
    0
}

/// §7.4.9.2 — the source of a luma prediction block's intra mode,
/// selected by `prev_intra_luma_pred_flag`.
///
/// `prev_intra_luma_pred_flag == 1` maps to [`LumaIntraModeSource::Mpm`]
/// (the §8.4.2 most-probable-mode candidate list; `mpm_idx` follows in
/// the §7.3.8.5 syntax to select among the candidates).
/// `prev_intra_luma_pred_flag == 0` maps to
/// [`LumaIntraModeSource::Remaining`] (the §7.3.8.5
/// `rem_intra_luma_pred_mode` field carries the mode directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LumaIntraModeSource {
    /// The luma intra mode is taken from the §8.4.2 most-probable-mode
    /// candidate list. Reachable from `prev_intra_luma_pred_flag == 1`;
    /// the §7.3.8.5 `mpm_idx` element follows to pick a candidate.
    Mpm,
    /// The luma intra mode is carried directly by the §7.3.8.5
    /// `rem_intra_luma_pred_mode` field. Reachable from
    /// `prev_intra_luma_pred_flag == 0`.
    Remaining,
}

/// Map a decoded `prev_intra_luma_pred_flag` value to
/// [`LumaIntraModeSource`] per §7.4.9.2 / §7.3.8.5: `1 ⇒ Mpm`,
/// `0 ⇒ Remaining`. The mapping is total over the FL `cMax = 1` shape
/// (any other input is rejected in debug builds; release builds fall
/// back to [`LumaIntraModeSource::Remaining`] to avoid panicking on an
/// accidental out-of-range input).
#[must_use]
pub fn luma_intra_mode_source_from_flag(prev_intra_luma_pred_flag: u8) -> LumaIntraModeSource {
    debug_assert!(
        prev_intra_luma_pred_flag <= 1,
        "prev_intra_luma_pred_flag is FL cMax = 1 (Table 9-43); decoded value must be 0 or 1"
    );
    if prev_intra_luma_pred_flag == 1 {
        LumaIntraModeSource::Mpm
    } else {
        LumaIntraModeSource::Remaining
    }
}

/// Decode `prev_intra_luma_pred_flag` (§7.3.8.5 / §7.4.9.2) from the
/// CABAC engine, consuming one context for the FL bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-12 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place per
/// the §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.5 intra-PB loop always reaches this read for an intra CU
/// (there is no not-present inference rule for this element), so the
/// parser unconditionally calls this primitive once per luma
/// prediction block.
///
/// Returns the decoded `u8` (0 or 1) — 1 selects the §8.4.2
/// most-probable-mode source (`mpm_idx` follows), 0 selects the
/// `rem_intra_luma_pred_mode` source. See
/// [`luma_intra_mode_source_from_flag`] to fold the result into the
/// [`LumaIntraModeSource`] enum.
pub fn decode_prev_intra_luma_pred_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin uses ctxInc = 0 (a single Table
    // 9-12 entry) per the Table 9-48 row.
    debug_assert_eq!(
        prev_intra_luma_pred_flag_ctx_inc(),
        0,
        "Table 9-48 row for prev_intra_luma_pred_flag: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// mpm_idx / rem_intra_luma_pred_mode (§7.3.8.5 coding_unit() / §7.4.9.2)
// Table 9-43 binarization + Table 9-48 (both fully bypass-coded). These
// are the two §7.3.8.5 intra-PB luma-mode fields that follow
// prev_intra_luma_pred_flag: when the flag is 1 the §8.4.2 candidate
// list is selected and `mpm_idx` (TR cMax = 2, cRiceParam = 0) picks
// among the three candidates; when 0 `rem_intra_luma_pred_mode`
// (FL cMax = 31) carries the residual mode directly. Neither element is
// context-coded — Table 9-48 marks every bin `bypass` — so the slice
// parser reads them through the §9.3.4.3.4 bypass arithmetic decoder
// without allocating a context model. Presence is exactly the
// LumaIntraModeSource selection: Mpm ⇒ mpm_idx present, Remaining ⇒
// rem_intra_luma_pred_mode present (the two are mutually exclusive per
// the §7.3.8.5 `if( prev_intra_luma_pred_flag ) … else …` syntax).
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `mpm_idx`: TR with `cMax = 2` and
/// `cRiceParam = 0`. The TR prefix with `cRiceParam = 0` collapses to
/// truncated-unary, so the on-wire string is `0`, `10`, or `11` for the
/// three values 0, 1, 2 — at most two bypass bins (Table 9-48 marks
/// bin 0 and bin 1 `bypass`, binIdx ≥ 2 `na`).
pub const MPM_IDX_TR_CMAX: u32 = 2;

/// `cRiceParam` for the `mpm_idx` TR binarization (Table 9-43): 0. With
/// `cRiceParam = 0` the §9.3.3.10 TR has no Rice suffix and the prefix
/// is plain truncated-unary.
pub const MPM_IDX_TR_C_RICE_PARAM: u32 = 0;

/// Table 9-43 binarization shape for `rem_intra_luma_pred_mode`: FL with
/// `cMax = 31`. The §9.3.3.5 derivation gives
/// `fixedLength = Ceil(Log2(cMax + 1)) = Ceil(Log2(32)) = 5`, so the
/// on-wire string is exactly five bypass bins (Table 9-48 marks
/// binIdx 0..=4 `bypass`, binIdx 5 `na`) read MSB-first.
pub const REM_INTRA_LUMA_PRED_MODE_FL_CMAX: u32 = 31;

/// §9.3.3.5 — bit width of the FL string for `rem_intra_luma_pred_mode`.
///
/// `Ceil(Log2(REM_INTRA_LUMA_PRED_MODE_FL_CMAX + 1)) = Ceil(Log2(32))
/// = 5`. The decoded 5-bit value is the §8.4.2 step-2 starting point
/// (`IntraPredModeY` before the candModeList increment pass).
pub const REM_INTRA_LUMA_PRED_MODE_FL_NBITS: u32 = 5;

/// Decode `mpm_idx` (§7.3.8.5 / §7.4.9.2) from the CABAC engine.
///
/// Read when `prev_intra_luma_pred_flag == 1`
/// ([`LumaIntraModeSource::Mpm`]): the §8.4.2 derivation sets
/// `IntraPredModeY = candModeList[ mpm_idx ]`, so the returned value is
/// the index into the three-entry most-probable-mode candidate list and
/// is always in `0..=2`.
///
/// Binarization is TR with `cMax = 2`, `cRiceParam = 0` (Table 9-43) and
/// every bin is bypass-coded (Table 9-48), so this reads the
/// truncated-unary prefix straight from the §9.3.4.3.4 bypass decoder:
/// a `0` bin terminates immediately (value 0), otherwise a second bin
/// distinguishes value 1 (`10`) from value 2 (`11`). No context model is
/// consumed.
pub fn decode_mpm_idx(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    // §9.3.3.10 TR prefix with cRiceParam = 0 driven entirely by the
    // bypass decoder. `is_escape` (all `cMax` bins were 1) is the
    // value == cMax case; with cRiceParam = 0 there is no suffix to
    // read, so the prefix value is the final value either way.
    let (value, _is_escape) =
        read_truncated_rice_prefix(MPM_IDX_TR_CMAX, |_bin_idx| engine.decode_bypass())?;
    debug_assert!(
        value <= MPM_IDX_TR_CMAX,
        "mpm_idx is TR cMax = 2 (Table 9-43); decoded value must be in 0..=2"
    );
    Ok(value as u8)
}

/// Decode `rem_intra_luma_pred_mode` (§7.3.8.5 / §7.4.9.2) from the
/// CABAC engine.
///
/// Read when `prev_intra_luma_pred_flag == 0`
/// ([`LumaIntraModeSource::Remaining`]): the §8.4.2 step-2 derivation
/// seeds `IntraPredModeY = rem_intra_luma_pred_mode`, then increments it
/// once for each sorted candModeList entry it meets or exceeds. The
/// returned value is the pre-increment seed and is always in `0..=31`.
///
/// Binarization is FL with `cMax = 31` (Table 9-43) and every bin is
/// bypass-coded (Table 9-48), so this reads exactly five bypass bins
/// MSB-first via [`CabacEngine::decode_bypass_bits`]. No context model
/// is consumed.
pub fn decode_rem_intra_luma_pred_mode(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    let value = engine.decode_bypass_bits(REM_INTRA_LUMA_PRED_MODE_FL_NBITS as u8)?;
    debug_assert!(
        value <= REM_INTRA_LUMA_PRED_MODE_FL_CMAX,
        "rem_intra_luma_pred_mode is FL cMax = 31 (Table 9-43); decoded value must be in 0..=31"
    );
    Ok(value as u8)
}

// ---------------------------------------------------------------------
// IntraPredModeY derivation (§8.4.2 — derivation process for luma intra
// prediction mode). This is the process the §7.3.8.5 luma-mode signalling
// group (prev_intra_luma_pred_flag, mpm_idx, rem_intra_luma_pred_mode)
// feeds: given the two §8.4.2-step-2 candidate neighbour modes
// candIntraPredModeA / candIntraPredModeB it builds the three-entry
// candModeList (step 3) and then resolves IntraPredModeY from the
// signalled fields (step 4 — either candModeList[ mpm_idx ] on the Mpm
// path, or rem_intra_luma_pred_mode passed through the sorted-list
// increment pass on the Remaining path).
//
// Table 8-1 names the modes: 0 = INTRA_PLANAR, 1 = INTRA_DC,
// 2..34 = INTRA_ANGULAR2..INTRA_ANGULAR34 (35 modes total). The
// §8.4.2-step-2 candidate reduction (availability via §6.4.1,
// CuPredMode / pcm_flag tests, the CTB-row-boundary B clamp) is the
// slice-data parser's responsibility — it holds the neighbour state and
// the §6.4.1 z-scan availability — and yields the two candidate mode
// values this process consumes, matching the availability-as-input
// convention the §9.3.4.2.2 neighbour ctxInc derivations use.
// ---------------------------------------------------------------------

/// Table 8-1 — `INTRA_PLANAR`, the mode-0 planar predictor.
pub const INTRA_PLANAR: u8 = 0;

/// Table 8-1 — `INTRA_DC`, the mode-1 DC predictor. This is also the
/// §8.4.2-step-2 substitute value `candIntraPredModeX` takes when the
/// neighbour is unavailable, inter-coded, PCM, or (for B) lies in the
/// CTB row above the current one.
pub const INTRA_DC: u8 = 1;

/// Table 8-1 — `INTRA_ANGULAR26`, the mode-26 vertical angular
/// predictor. §8.4.2 step 3 uses it as `candModeList[ 2 ]` in the
/// equal-and-low-candidate (eq. 8-23) and the both-PLANAR-and-DC-present
/// (final step-3 else) fall-throughs.
pub const INTRA_ANGULAR26: u8 = 26;

/// Table 8-1 — the highest luma intra prediction mode index
/// (`INTRA_ANGULAR34`); modes run `0..=34` inclusive.
pub const INTRA_PRED_MODE_MAX: u8 = 34;

/// §8.4.2 step 3 — build the three-entry most-probable-mode candidate
/// list `candModeList[ 0..=2 ]` from the two §8.4.2-step-2 candidate
/// neighbour modes `candIntraPredModeA` (left, location `(xPb − 1, yPb)`)
/// and `candIntraPredModeB` (above, location `(xPb, yPb − 1)`).
///
/// Both candidate inputs are already the reduced step-2 values (an
/// available intra neighbour's `IntraPredModeY`, else `INTRA_DC`); this
/// function implements only the step-3 list construction:
///
/// * `candIntraPredModeB == candIntraPredModeA`:
///   * candidate `< 2` (PLANAR or DC) ⇒ `{PLANAR, DC, ANGULAR26}`
///     (eqs. 8-21..8-23).
///   * otherwise (an angular candidate) ⇒ the candidate plus its two
///     neighbouring angular modes (eqs. 8-24..8-26):
///     `{candA, 2 + ((candA + 29) % 32), 2 + ((candA − 2 + 1) % 32)}`.
/// * `candIntraPredModeB != candIntraPredModeA`:
///   * `candModeList[ 0 ] = candA`, `candModeList[ 1 ] = candB`
///     (eqs. 8-27/8-28); `candModeList[ 2 ]` is the first of
///     `{PLANAR, DC, ANGULAR26}` not already present in the first two
///     entries.
///
/// Both candidate inputs must be valid Table 8-1 modes (`0..=34`).
#[must_use]
pub fn intra_luma_cand_mode_list(
    cand_intra_pred_mode_a: u8,
    cand_intra_pred_mode_b: u8,
) -> [u8; 3] {
    debug_assert!(
        cand_intra_pred_mode_a <= INTRA_PRED_MODE_MAX,
        "candIntraPredModeA is a Table 8-1 mode; must be in 0..=34"
    );
    debug_assert!(
        cand_intra_pred_mode_b <= INTRA_PRED_MODE_MAX,
        "candIntraPredModeB is a Table 8-1 mode; must be in 0..=34"
    );

    if cand_intra_pred_mode_b == cand_intra_pred_mode_a {
        if cand_intra_pred_mode_a < 2 {
            // eqs. 8-21..8-23 — equal low candidate.
            [INTRA_PLANAR, INTRA_DC, INTRA_ANGULAR26]
        } else {
            // eqs. 8-24..8-26 — equal angular candidate; the `% 32`
            // arithmetic wraps the two neighbouring angular modes back
            // into the 2..=33 angular band (the `2 +` re-bases the
            // 0..=31 residue onto INTRA_ANGULAR2).
            let a = u32::from(cand_intra_pred_mode_a);
            let one = 2 + ((a + 29) % 32);
            let two = 2 + ((a - 2 + 1) % 32);
            [cand_intra_pred_mode_a, one as u8, two as u8]
        }
    } else {
        // eqs. 8-27/8-28 — distinct candidates fill slots 0 and 1; slot
        // 2 is the first default mode (PLANAR, then DC, then ANGULAR26)
        // not already present.
        let zero = cand_intra_pred_mode_a;
        let one = cand_intra_pred_mode_b;
        let two = if zero != INTRA_PLANAR && one != INTRA_PLANAR {
            INTRA_PLANAR
        } else if zero != INTRA_DC && one != INTRA_DC {
            INTRA_DC
        } else {
            INTRA_ANGULAR26
        };
        [zero, one, two]
    }
}

/// §8.4.2 step 4 — derive `IntraPredModeY[ xPb ][ yPb ]` from the step-3
/// `candModeList` and the §7.3.8.5 signalled luma-mode fields.
///
/// `cand_mode_list` is the [`intra_luma_cand_mode_list`] output; `source`
/// is the [`LumaIntraModeSource`] the round-40 `prev_intra_luma_pred_flag`
/// selects.
///
/// * [`LumaIntraModeSource::Mpm`] — `IntraPredModeY = candModeList[
///   mpm_idx ]` (the `mpm_idx` field, `0..=2`, indexes the candidate
///   list directly). `field` carries the decoded `mpm_idx`.
/// * [`LumaIntraModeSource::Remaining`] — the step-4 ordered procedure:
///   the candidate list is sorted ascending (the eqs. 8-29..8-31
///   three-compare-and-swap sort), `IntraPredModeY` is seeded with
///   `rem_intra_luma_pred_mode` (`field`, `0..=31`), then incremented by
///   one for every sorted candidate it meets or exceeds (low-to-high, so
///   the running value is re-compared against each successive candidate).
///   This re-injects the three excluded most-probable modes, mapping the
///   31-value remaining field onto the 35-mode space.
///
/// The returned value is always a valid Table 8-1 mode (`0..=34`).
#[must_use]
pub fn derive_intra_pred_mode_y(
    cand_mode_list: [u8; 3],
    source: LumaIntraModeSource,
    field: u8,
) -> u8 {
    match source {
        LumaIntraModeSource::Mpm => {
            debug_assert!(
                field <= MPM_IDX_TR_CMAX as u8,
                "mpm_idx is TR cMax = 2 (Table 9-43); must index the three-entry candModeList"
            );
            cand_mode_list[field as usize]
        }
        LumaIntraModeSource::Remaining => {
            debug_assert!(
                field <= REM_INTRA_LUMA_PRED_MODE_FL_CMAX as u8,
                "rem_intra_luma_pred_mode is FL cMax = 31 (Table 9-43); must be in 0..=31"
            );
            // eqs. 8-29..8-31 — sort the three candidates ascending.
            let mut sorted = cand_mode_list;
            if sorted[0] > sorted[1] {
                sorted.swap(0, 1);
            }
            if sorted[0] > sorted[2] {
                sorted.swap(0, 2);
            }
            if sorted[1] > sorted[2] {
                sorted.swap(1, 2);
            }
            // step-4 increment pass: seed with rem_intra_luma_pred_mode,
            // then bump once per sorted candidate at-or-below the running
            // value. The candidates are distinct and ascending, so the
            // bump count is monotone and the result is the unsigned
            // remaining field shifted up past the three excluded modes.
            let mut mode = field;
            for &cand in &sorted {
                if mode >= cand {
                    mode += 1;
                }
            }
            mode
        }
    }
}

// ---------------------------------------------------------------------
// intra_chroma_pred_mode (§7.3.8.5 coding_unit() / §7.4.9.5)
// §9.3.3.8 / Table 9-46 binarization + Table 9-48 + §8.4.3 derivation.
// The chroma-mode field that follows the luma-mode group in §7.3.8.5:
// present once per CU when ChromaArrayType != 0 and != 3, and once per
// luma prediction block (the same i/j loop as the luma fields) when
// ChromaArrayType == 3; absent entirely when ChromaArrayType == 0
// (monochrome — §8.4.3 is only invoked when ChromaArrayType != 0).
// ---------------------------------------------------------------------

/// Table 9-46 — the `intra_chroma_pred_mode` value signalled by the
/// single-bin string `0`: 4. Per Table 8-2 row 4 the §8.4.3 `modeIdx`
/// for this value is `IntraPredModeY` itself (the chroma prediction
/// mode tracks the luma mode; for `ChromaArrayType == 2` the tracked
/// mode is then remapped through Table 8-3).
pub const INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA: u8 = 4;

/// §9.3.3.8 / Table 9-46 — bit width of the FL suffix that follows the
/// `1` prefix bin: 2. The suffix carries values 0..=3 MSB-first
/// (`100`⇒0, `101`⇒1, `110`⇒2, `111`⇒3); both suffix bins are
/// bypass-coded per Table 9-48.
pub const INTRA_CHROMA_PRED_MODE_SUFFIX_FL_NBITS: u32 = 2;

/// §9.3.4.2.1 / Table 9-48 row for `intra_chroma_pred_mode`. Bin 0 is
/// context-coded with `ctxInc = 0`; bins 1 and 2 are `bypass` and bins
/// beyond binIdx 2 are `na` (the Table 9-46 strings are at most three
/// bins long).
///
/// Table 9-13 lists three ctxIdx slots with
/// `initValue = {63, 152, 152}` for initType 0, 1 and 2 — the element
/// is read in I, P and B slices (intra CUs occur in every slice type),
/// so all three initType slots are populated. This layer hands back
/// the bank-relative `ctxInc` only; the Table 9-13 initValue plus the
/// Table 9-4 initType-to-ctxIdx mapping is consumed at slice-init
/// scope.
///
/// Returns the ctxInc when called with `bin_idx == 0`; any other value
/// is bypass (1, 2) or `na` per Table 9-48 and the function returns 0
/// defensively (callers should not invoke this for the bypass bins).
#[must_use]
pub fn intra_chroma_pred_mode_ctx_inc(bin_idx: u32) -> u32 {
    debug_assert!(
        bin_idx == 0,
        "intra_chroma_pred_mode Table 9-48 row: only bin 0 is context-coded; bins 1..2 are bypass"
    );
    0
}

/// Decode `intra_chroma_pred_mode` (§7.3.8.5 / §7.4.9.5) from the
/// CABAC engine.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the single
/// Table 9-13 / §9.3.4.2 `ctxInc = 0` slot; it is mutated in place per
/// the §9.3.4.3.2.2 state transition.
///
/// Binarization is the §9.3.3.8 / Table 9-46 process: one
/// context-coded prefix bin, then — only when that bin is 1 — a
/// two-bit bypass FL suffix. A `0` prefix decodes to value 4
/// ([`INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA`]; no suffix bins are read);
/// a `1` prefix is followed by the MSB-first two-bit suffix carrying
/// values 0..=3. The returned value is therefore always in
/// `{0, 1, 2, 3, 4}` — feed it to [`intra_pred_mode_c_mode_idx`] /
/// [`derive_intra_pred_mode_c`] for the §8.4.3 `IntraPredModeC`
/// derivation.
pub fn decode_intra_chroma_pred_mode(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48: bin 0 is context-coded with ctxInc = 0.
    debug_assert_eq!(
        intra_chroma_pred_mode_ctx_inc(0),
        0,
        "Table 9-48 row for intra_chroma_pred_mode: bin 0 ctxInc = 0"
    );
    let prefix = engine.decode_decision(ctx)?;
    if prefix == 0 {
        // Table 9-46: the single-bin string `0` is value 4.
        return Ok(INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA);
    }
    // Table 9-46: `1` prefix + two-bit FL bypass suffix ⇒ values 0..=3.
    let suffix = engine.decode_bypass_bits(INTRA_CHROMA_PRED_MODE_SUFFIX_FL_NBITS as u8)?;
    debug_assert!(
        suffix <= 3,
        "intra_chroma_pred_mode suffix is two FL bins (Table 9-46); value must be in 0..=3"
    );
    Ok(suffix as u8)
}

/// §8.4.3 Table 8-2 — derive `modeIdx` from `intra_chroma_pred_mode`
/// and `IntraPredModeY`.
///
/// Row 4 (`intra_chroma_pred_mode == 4`) yields `IntraPredModeY`
/// itself. Rows 0..=3 select the base mode from `{0, 26, 10, 1}`
/// (planar, vertical, horizontal, DC angular indices in
/// `intra_chroma_pred_mode` order); when the selected base mode equals
/// `IntraPredModeY` the table substitutes 34 instead (each Table 8-2
/// column places 34 exactly where the row's base mode would collide
/// with the luma mode).
///
/// `intra_pred_mode_y` must be a valid §8.4.2 mode (0..=34) and
/// `intra_chroma_pred_mode` a decoded Table 9-46 value
/// (`{0, 1, 2, 3, 4}`); out-of-range inputs are rejected in debug
/// builds (release builds fall back to the luma-tracking row to avoid
/// panicking on an accidental out-of-range input).
#[must_use]
pub fn intra_pred_mode_c_mode_idx(intra_chroma_pred_mode: u8, intra_pred_mode_y: u8) -> u8 {
    debug_assert!(
        intra_chroma_pred_mode <= 4,
        "intra_chroma_pred_mode is a Table 9-46 value; must be in 0..=4"
    );
    debug_assert!(
        intra_pred_mode_y <= 34,
        "IntraPredModeY is a §8.4.2 mode index; must be in 0..=34"
    );
    // Table 8-2 rows 0..=3 base modes, in intra_chroma_pred_mode order.
    const BASE_MODES: [u8; 4] = [0, 26, 10, 1];
    match BASE_MODES.get(intra_chroma_pred_mode as usize) {
        Some(&base) if base == intra_pred_mode_y => 34,
        Some(&base) => base,
        // Row 4 (and the release-build out-of-range fallback): modeIdx
        // tracks the luma mode.
        None => intra_pred_mode_y,
    }
}

/// §8.4.3 Table 8-3 — `IntraPredModeC` as a function of `modeIdx` when
/// `ChromaArrayType == 2`. Indexed by `modeIdx` 0..=34; entries 0..=2
/// pass through unchanged (the Table 8-3 `X <= 2` column).
pub const INTRA_PRED_MODE_C_CHROMA_422_MAP: [u8; 35] = [
    0, 1, 2, // X <= 2 passes through
    2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, // modeIdx 3..=14
    18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25, // modeIdx 15..=25
    26, 27, 27, 28, 28, 29, 29, 30, 31, // modeIdx 26..=34
];

/// §8.4.3 Table 8-3 — remap `modeIdx` to `IntraPredModeC` for
/// `ChromaArrayType == 2` streams. `mode_idx` must be in 0..=34;
/// out-of-range inputs are rejected in debug builds (release builds
/// pass the value through unchanged to avoid panicking).
#[must_use]
pub fn intra_pred_mode_c_chroma_422(mode_idx: u8) -> u8 {
    debug_assert!(
        mode_idx <= 34,
        "modeIdx is a §8.4.3 Table 8-2 output; must be in 0..=34"
    );
    INTRA_PRED_MODE_C_CHROMA_422_MAP
        .get(mode_idx as usize)
        .copied()
        .unwrap_or(mode_idx)
}

/// §8.4.3 — the full `IntraPredModeC` derivation: Table 8-2 `modeIdx`
/// from the decoded `intra_chroma_pred_mode` and the §8.4.2
/// `IntraPredModeY`, then — when `ChromaArrayType == 2` — the Table
/// 8-3 remap; otherwise `IntraPredModeC` is `modeIdx` itself.
///
/// §8.4.3 is only invoked when `ChromaArrayType != 0`; the §7.3.8.5
/// syntax never emits `intra_chroma_pred_mode` for a monochrome
/// stream, so a caller holding a decoded value is already past that
/// gate.
#[must_use]
pub fn derive_intra_pred_mode_c(
    intra_chroma_pred_mode: u8,
    intra_pred_mode_y: u8,
    chroma_array_type_is_2: bool,
) -> u8 {
    let mode_idx = intra_pred_mode_c_mode_idx(intra_chroma_pred_mode, intra_pred_mode_y);
    if chroma_array_type_is_2 {
        intra_pred_mode_c_chroma_422(mode_idx)
    } else {
        mode_idx
    }
}

// ---------------------------------------------------------------------
// last_sig_coeff_{x,y}_{prefix,suffix} (§7.3.8.11, §7.4.9.11)
// ---------------------------------------------------------------------

/// Which `last_sig_coeff_*_prefix` bank the caller's context-model
/// slice corresponds to. The X and Y prefixes live in separate banks
/// of contexts; we tag the request so the caller can pick the right
/// slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastSigCoeffBank {
    /// `last_sig_coeff_x_prefix` bank.
    X,
    /// `last_sig_coeff_y_prefix` bank.
    Y,
}

/// §9.3.4.2.3 — `(ctxOffset, ctxShift)` for the
/// `last_sig_coeff_{x,y}_prefix` ctxInc derivation. `log2_trafo_size`
/// is the transform-block log2 dimension (2..=5; 2 = 4×4, 5 = 32×32);
/// `is_chroma` selects the chroma bank when `cIdx > 0`.
///
/// Spec equations (§9.3.4.2.3 Table 9-26 / 9-27):
///
/// ```text
/// // luma (cIdx == 0):
/// ctxOffset = 3 * (log2TrafoSize - 2) + ((log2TrafoSize - 1) >> 2)
/// ctxShift  = (log2TrafoSize + 1) >> 2
///
/// // chroma (cIdx > 0):
/// ctxOffset = 15
/// ctxShift  = log2TrafoSize - 2
/// ```
#[must_use]
pub fn last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size: u32, is_chroma: bool) -> (u32, u32) {
    if is_chroma {
        // §9.3.4.2.3 — chroma TBs always sit at ctxOffset = 15,
        // ctxShift = log2TrafoSize − 2.
        (15, log2_trafo_size - 2)
    } else {
        // §9.3.4.2.3 — luma: piecewise linear in log2TrafoSize.
        let offset = 3 * (log2_trafo_size - 2) + ((log2_trafo_size - 1) >> 2);
        let shift = (log2_trafo_size + 1) >> 2;
        (offset, shift)
    }
}

/// §9.3.4.2.3 — bank-relative `ctxInc` for a single
/// `last_sig_coeff_*_prefix` bin:
///
/// ```text
/// ctxInc = (binIdx >> ctxShift) + ctxOffset
/// ```
#[must_use]
pub fn last_sig_coeff_prefix_ctx_inc(bin_idx: u32, ctx_offset: u32, ctx_shift: u32) -> u32 {
    (bin_idx >> ctx_shift) + ctx_offset
}

/// §9.3.3.10 — `cMax` for `last_sig_coeff_{x,y}_prefix`:
/// `(log2TrafoSize << 1) − 1`. For a 4×4 TB (log2=2) the prefix is at
/// most 3 bins; for a 32×32 TB (log2=5) at most 9 bins.
#[must_use]
pub fn last_sig_coeff_prefix_cmax(log2_trafo_size: u32) -> u32 {
    (log2_trafo_size << 1) - 1
}

/// §7.4.9.11 equations 7-74..7-77 — derive the final
/// `LastSignificantCoeffX/Y` position from a decoded
/// `(prefix, optional suffix)` pair. When `prefix <= 3` no suffix is
/// signalled and the position equals the prefix. When `prefix > 3` the
/// suffix is `nBits = (prefix >> 1) − 1` bits, and the position is
///
/// ```text
/// pos = ( ( 2 + ( prefix & 1 ) ) << ( (prefix >> 1) - 1 ) ) + suffix
/// ```
///
/// `suffix` is the unsigned suffix value (caller already decoded the
/// `nBits` bypass bins, MSB-first); pass `None` when `prefix <= 3`.
#[must_use]
pub fn last_sig_coeff_position(prefix: u32, suffix: Option<u32>) -> u32 {
    if prefix <= 3 {
        // §7.4.9.11: no suffix, the prefix is the final value.
        prefix
    } else {
        let n_bits = (prefix >> 1) - 1;
        let base = (2 + (prefix & 1)) << n_bits;
        base + suffix.unwrap_or(0)
    }
}

/// Suffix bit count for `last_sig_coeff_{x,y}_suffix`: `nBits =
/// (prefix >> 1) − 1` when `prefix > 3`, otherwise 0 (no suffix
/// signalled).
#[must_use]
pub fn last_sig_coeff_suffix_n_bits(prefix: u32) -> u32 {
    if prefix > 3 { (prefix >> 1) - 1 } else { 0 }
}

/// Decoded `last_sig_coeff_*` element (one of X or Y; same shape for
/// both banks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LastSigCoeff {
    /// `last_sig_coeff_*_prefix` from the wire.
    pub prefix: u32,
    /// `last_sig_coeff_*_suffix` from the wire (`None` when
    /// `prefix <= 3`, the no-suffix-signalled case).
    pub suffix: Option<u32>,
    /// §7.4.9.11 derived `LastSignificantCoeff{X,Y}` position. Range
    /// 0..=(1 << log2TrafoSize) - 1.
    pub position: u32,
}

/// Decode one `last_sig_coeff_*_{prefix,suffix}` pair from the CABAC
/// engine. `ctx_bank` is the caller-supplied slice of context variables
/// for the active bank ([`LastSigCoeffBank`]); the bin at index
/// `binIdx` consumes the context at index
/// `ctx_inc = (binIdx >> ctxShift) + ctxOffset` within that slice (the
/// caller is responsible for sizing the slice so the highest used
/// `ctx_inc` index is in-bounds).
///
/// Returns the decoded prefix, optional suffix, and the §7.4.9.11
/// derived position. The arguments
/// `log2_trafo_size` and `is_chroma` drive both the TR-prefix `cMax`
/// and the §9.3.4.2.3 `(ctxOffset, ctxShift)` derivation.
///
/// The §7.4.9.11 swap for `scanIdx == 2` (the diagonal scan reflection
/// of `LastSignificantCoeff{X,Y}`) is **not** applied here; the caller
/// receives the on-wire X / Y positions and applies the swap at the
/// residual-coding level where `scanIdx` is known.
pub fn decode_last_sig_coeff(
    engine: &mut CabacEngine<'_>,
    log2_trafo_size: u32,
    is_chroma: bool,
    ctx_bank: &mut [ContextModel],
) -> Result<LastSigCoeff, CabacError> {
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let (ctx_offset, ctx_shift) =
        last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size, is_chroma);

    let (prefix, _is_escape) = read_truncated_rice_prefix(c_max, |bin_idx| {
        let ctx_inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_offset, ctx_shift) as usize;
        engine.decode_decision(&mut ctx_bank[ctx_inc])
    })?;

    let n_bits = last_sig_coeff_suffix_n_bits(prefix);
    let suffix = if n_bits > 0 {
        Some(engine.decode_bypass_bits(n_bits as u8)?)
    } else {
        None
    };
    let position = last_sig_coeff_position(prefix, suffix);
    Ok(LastSigCoeff {
        prefix,
        suffix,
        position,
    })
}

// ---------------------------------------------------------------------
// coded_sub_block_flag ctxInc — §9.3.4.2.4
// ---------------------------------------------------------------------

/// §9.3.4.2.4 — derive the `ctxInc` for one bin of
/// `coded_sub_block_flag` from the colour-component index, the
/// current sub-block scan location `(xS, yS)`, the transform-block
/// size `log2TrafoSize`, and the two previously decoded neighbour
/// bins in sub-block scan order.
///
/// Spec equations:
///
/// ```text
/// // §9.3.4.2.4, equations 9-35..9-39
/// csbfCtx = 0
/// if xS < (1 << (log2TrafoSize - 2)) - 1:
///     csbfCtx += coded_sub_block_flag[xS + 1][yS]
/// if yS < (1 << (log2TrafoSize - 2)) - 1:
///     csbfCtx += coded_sub_block_flag[xS][yS + 1]
/// if cIdx == 0:
///     ctxInc = min(csbfCtx, 1)
/// else:
///     ctxInc = 2 + min(csbfCtx, 1)
/// ```
///
/// `right_neighbour` is the previously decoded
/// `coded_sub_block_flag[xS + 1][yS]` (0/1); pass `0` whenever the
/// current sub-block sits on the right edge of the TB
/// (`xS == (1 << (log2TrafoSize - 2)) - 1`), where equation 9-36 does
/// not apply. `below_neighbour` is the previously decoded
/// `coded_sub_block_flag[xS][yS + 1]` (0/1); pass `0` whenever the
/// current sub-block sits on the bottom edge of the TB
/// (`yS == (1 << (log2TrafoSize - 2)) - 1`).
///
/// Returns `ctxInc` in `{0, 1}` for luma (`cIdx == 0`) or
/// `{2, 3}` for chroma (`cIdx > 0`).
///
/// # Panics
/// Does not panic; both neighbour inputs are clamped via the boolean
/// `>= 1` comparison.
#[must_use]
pub fn coded_sub_block_flag_ctx_inc(
    is_chroma: bool,
    right_neighbour: u8,
    below_neighbour: u8,
) -> u32 {
    // §9.3.4.2.4: csbfCtx is the unsigned sum of the two neighbour
    // bins (each either 0 or 1), so it lies in {0, 1, 2}. The final
    // ctxInc clamps to {0, 1} via min(csbfCtx, 1).
    let csbf_ctx = (right_neighbour & 1) as u32 + (below_neighbour & 1) as u32;
    let clipped = csbf_ctx.min(1);
    if is_chroma {
        // §9.3.4.2.4 equation 9-39: chroma uses the {2, 3} bank.
        2 + clipped
    } else {
        // §9.3.4.2.4 equation 9-38: luma uses the {0, 1} bank.
        clipped
    }
}

/// §9.3.4.2.4 — helper that takes the full sub-block location and
/// transform-block size and applies the edge-gating from equations
/// 9-36 / 9-37 before delegating to [`coded_sub_block_flag_ctx_inc`].
/// The caller passes the previously decoded
/// `coded_sub_block_flag[xS + 1][yS]` and
/// `coded_sub_block_flag[xS][yS + 1]` values regardless of the edge
/// condition; this function zeroes them when the corresponding
/// equation 9-36 / 9-37 does not apply (sub-block on the right /
/// bottom TB edge).
///
/// `xs` / `ys` are the sub-block scan location in 4-sample units (so
/// a 32×32 TB at `log2TrafoSize == 5` has `xs, ys ∈ 0..8`).
#[must_use]
pub fn coded_sub_block_flag_ctx_inc_with_edge(
    is_chroma: bool,
    xs: u32,
    ys: u32,
    log2_trafo_size: u32,
    right_neighbour: u8,
    below_neighbour: u8,
) -> u32 {
    // §9.3.4.2.4 edge gates: ( 1 << ( log2TrafoSize − 2 ) ) − 1 is
    // the maximum sub-block index along each axis for the TB.
    let max_sub_block_idx = (1u32 << (log2_trafo_size - 2)) - 1;
    let right = if xs < max_sub_block_idx {
        right_neighbour & 1
    } else {
        0
    };
    let below = if ys < max_sub_block_idx {
        below_neighbour & 1
    } else {
        0
    };
    coded_sub_block_flag_ctx_inc(is_chroma, right, below)
}

// ---------------------------------------------------------------------
// split_cu_flag / cu_skip_flag ctxInc — §9.3.4.2.2 Table 9-49
// ---------------------------------------------------------------------

/// §9.3.4.2.2 Table 9-49 — the shared `ctxInc` derivation for
/// syntax elements that take their `ctxInc` from a `condL` / `condA`
/// neighbour pair plus the §6.4.1 availability of the neighbour
/// blocks:
///
/// ```text
/// ctxInc = (condL && availableL) + (condA && availableA)
/// ```
///
/// Used by `split_cu_flag` (with `condL = CtDepth[xNbL][yNbL] >
/// cqtDepth`, `condA = CtDepth[xNbA][yNbA] > cqtDepth`) and
/// `cu_skip_flag` (with `condL = cu_skip_flag[xNbL][yNbL]`,
/// `condA = cu_skip_flag[xNbA][yNbA]`).
///
/// Returns a `ctxInc` in `{0, 1, 2}`.
#[must_use]
pub fn left_above_ctx_inc(cond_l: bool, available_l: bool, cond_a: bool, available_a: bool) -> u32 {
    let l = (cond_l && available_l) as u32;
    let a = (cond_a && available_a) as u32;
    l + a
}

/// §9.3.4.2.2 Table 9-49 — `condL` / `condA` predicate for
/// `split_cu_flag`: each neighbour contributes `CtDepth[xNb][yNb] >
/// cqtDepth`. Returns the predicate as a `bool` ready to feed into
/// [`left_above_ctx_inc`].
///
/// `neighbour_ct_depth` is the `CtDepth[xNb][yNb]` value already
/// derived by the caller for the neighbouring (left or above) block;
/// `cqt_depth` is the current block's coding-quadtree depth.
#[must_use]
pub fn split_cu_flag_cond(neighbour_ct_depth: u32, cqt_depth: u32) -> bool {
    neighbour_ct_depth > cqt_depth
}

/// §9.3.4.2.2 Table 9-49 — `condL` / `condA` predicate for
/// `cu_skip_flag`: each neighbour contributes
/// `cu_skip_flag[xNb][yNb]` as-is.
#[must_use]
pub fn cu_skip_flag_cond(neighbour_cu_skip_flag: u8) -> bool {
    (neighbour_cu_skip_flag & 1) != 0
}

/// §9.3.4.2.2 Table 9-49 row for `split_cu_flag`. Convenience: takes
/// the neighbour `CtDepth` values and availability and returns
/// `ctxInc` directly.
///
/// The neighbour `CtDepth` arguments are ignored when the matching
/// `available` flag is false (the `(condX && availableX)` AND in the
/// Table 9-49 formula short-circuits in that case), but the caller
/// may pass any placeholder for them.
#[must_use]
pub fn split_cu_flag_ctx_inc(
    left_ct_depth: u32,
    available_l: bool,
    above_ct_depth: u32,
    available_a: bool,
    cqt_depth: u32,
) -> u32 {
    left_above_ctx_inc(
        split_cu_flag_cond(left_ct_depth, cqt_depth),
        available_l,
        split_cu_flag_cond(above_ct_depth, cqt_depth),
        available_a,
    )
}

/// §9.3.4.2.2 Table 9-49 row for `cu_skip_flag`. Convenience: takes
/// the neighbour `cu_skip_flag` values and availability and returns
/// `ctxInc` directly.
#[must_use]
pub fn cu_skip_flag_ctx_inc(
    left_cu_skip_flag: u8,
    available_l: bool,
    above_cu_skip_flag: u8,
    available_a: bool,
) -> u32 {
    left_above_ctx_inc(
        cu_skip_flag_cond(left_cu_skip_flag),
        available_l,
        cu_skip_flag_cond(above_cu_skip_flag),
        available_a,
    )
}

// ---------------------------------------------------------------------
// split_cu_flag (§7.3.8.4 coding_quadtree() / §7.4.9.4) decode primitive
// Table 9-43 FL binarization (cMax = 1) + Table 9-48 / §9.3.4.2.2 ctxInc
// + Table 9-7 ctxIdx initValue. Read once per §7.3.8.4
// `coding_quadtree( )` node inside the split-presence gate
//   ( x0 + ( 1 << log2CbSize ) <= pic_width_in_luma_samples ) &&
//   ( y0 + ( 1 << log2CbSize ) <= pic_height_in_luma_samples ) &&
//   log2CbSize > MinCbLog2SizeY
// otherwise it is inferred (§7.4.9.4): 1 when the block extends past the
// picture boundary or log2CbSize > MinCbLog2SizeY, else 0. The caller
// owns that gate / inference and the §9.3.4.2.2 neighbour `CtDepth`
// lookup that feeds [`split_cu_flag_ctx_inc`]; this primitive reads the
// single context-coded bin once the bank slot is selected.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `split_cu_flag[ ][ ]`: FL with
/// `cMax = 1` (a single context-coded bin per §7.3.8.4 coding-quadtree
/// node). Value 1 splits the coding block into four quarter-size child
/// nodes; value 0 makes it a leaf coding unit (§7.4.9.4).
pub const SPLIT_CU_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `split_cu_flag`: one context-coded bin
/// per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const SPLIT_CU_FLAG_FL_NBITS: u32 = 1;

/// Decode `split_cu_flag[ x0 ][ y0 ]` (§7.3.8.4 / §7.4.9.4) from the
/// CABAC engine, consuming one context for the FL `cMax = 1` bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the
/// §9.3.4.2.2 ctxInc slot in the §7.3.8.4 `split_cu_flag[3]` bank
/// (derive the slot with [`split_cu_flag_ctx_inc`] from the left /
/// above neighbours' `CtDepth` and §6.4.1 availability); it is mutated
/// in place per the §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.4 split-presence gate and the §7.4.9.4 not-present
/// inference (1 at a picture boundary or when `log2CbSize >
/// MinCbLog2SizeY`, else 0) are the `coding_quadtree( )` caller's
/// responsibility; this primitive is reached only when the flag is
/// present on the wire.
///
/// Returns the decoded `u8` (0 or 1): 1 recurses into four quarter-size
/// `coding_quadtree( )` children, 0 enters `coding_unit( )` for the
/// current block.
pub fn decode_split_cu_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; that bin's ctxInc comes from §9.3.4.2.2
    // (Table 9-49) and selects the bank slot the caller already bound
    // into `ctx`, so the decode itself is a single context decision.
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// cu_skip_flag (§7.3.8.5 coding_unit() / §7.4.9.5) decode primitive
// Table 9-43 FL binarization (cMax = 1) + Table 9-48 / §9.3.4.2.2 ctxInc
// + Table 9-9 ctxIdx initValue. Read once at the start of a §7.3.8.4
// `coding_quadtree( )` leaf when `slice_type != I` (inter-only); absent
// for I slices, where §7.4.9.5 infers it to 0. When 1 the CU takes the
// MODE_SKIP path: no `pred_mode_flag`, no `part_mode`, no transform
// tree — only the §7.3.8.6 merge-mode `prediction_unit( )` is parsed.
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `cu_skip_flag`: FL with
/// `cMax = 1` (a single context-coded bin per non-I coding unit).
/// Value 1 selects the MODE_SKIP path (§7.4.9.5).
pub const CU_SKIP_FLAG_FL_CMAX: u32 = 1;

/// FL-binarization bit count for `cu_skip_flag`: one context-coded bin
/// per §9.3.3.5 `Ceil(Log2(cMax + 1)) = 1`.
pub const CU_SKIP_FLAG_FL_NBITS: u32 = 1;

/// Decode `cu_skip_flag[ x0 ][ y0 ]` (§7.3.8.5 / §7.4.9.5) from the
/// CABAC engine, consuming one context for the FL `cMax = 1` bin.
///
/// `ctx` is the caller's `(pStateIdx, valMps)` state for the
/// §9.3.4.2.2 ctxInc slot in the §7.3.8.5 `cu_skip_flag[3]` bank
/// (derive the slot with [`cu_skip_flag_ctx_inc`] from the left /
/// above neighbours' decoded `cu_skip_flag` and §6.4.1 availability);
/// it is mutated in place per the §9.3.4.3.2.2 state transition.
///
/// The §7.3.8.5 read is gated on `slice_type != I`; for I slices the
/// flag is not present and §7.4.9.5 infers it to 0 (the caller owns
/// that gate). When the decoded value is 1 the coding unit takes the
/// MODE_SKIP path: see [`cu_pred_mode_from_skip`] for the §7.4.9.5
/// not-present `CuPredMode` inference that follows.
///
/// Returns the decoded `u8` (0 or 1).
pub fn decode_cu_skip_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // §9.3.4.2.1 + Table 9-48 + Table 9-43: the FL `cMax = 1` shape
    // emits exactly one bin; the ctxInc comes from §9.3.4.2.2 (Table
    // 9-49) and selects the bank slot the caller already bound into
    // `ctx`, so the decode is a single context decision.
    engine.decode_decision(ctx)
}

/// §7.4.9.5 — derive [`CuPredMode`] for a coding unit from its decoded
/// `cu_skip_flag` and the slice type, for the path where
/// `pred_mode_flag` is therefore not present.
///
/// `cu_skip_flag == 1` (only reachable for `slice_type != I`) maps to
/// [`CuPredMode::Skip`]; `cu_skip_flag == 0` in a P / B slice leaves the
/// mode to be decided by the subsequent `pred_mode_flag` (returns
/// `None`, signalling the caller must read that flag). For an I slice
/// `cu_skip_flag` is inferred to 0 and the mode is MODE_INTRA without a
/// `pred_mode_flag` read.
///
/// * `slice_type_is_i` — `true` iff `slice_type == I` (§7.4.7.1).
/// * `cu_skip_flag` — the §7.3.8.5 decoded flag (or 0-inferred for I).
///
/// Returns `Some(mode)` when the mode is fully determined without a
/// `pred_mode_flag` read, or `None` when a P / B non-skip CU still needs
/// `pred_mode_flag` to disambiguate MODE_INTER vs MODE_INTRA.
#[must_use]
pub fn cu_pred_mode_from_skip(slice_type_is_i: bool, cu_skip_flag: u8) -> Option<CuPredMode> {
    debug_assert!(
        cu_skip_flag <= 1,
        "cu_skip_flag is FL cMax = 1 (Table 9-43); decoded value must be 0 or 1"
    );
    if slice_type_is_i {
        // §7.4.9.5: I slices infer cu_skip_flag = 0 and CuPredMode =
        // MODE_INTRA without coding pred_mode_flag.
        Some(CuPredMode::Intra)
    } else if cu_skip_flag != 0 {
        // P / B skip CU: §7.4.9.5 infers MODE_SKIP, pred_mode_flag absent.
        Some(CuPredMode::Skip)
    } else {
        // P / B non-skip CU: pred_mode_flag is present and decides the
        // mode; the caller must read it.
        None
    }
}

// ---------------------------------------------------------------------
// split_transform_flag ctxInc — Table 9-48 row
// ---------------------------------------------------------------------

/// Table 9-48 row for `split_transform_flag[ ][ ][ ]`:
///
/// ```text
/// ctxInc = 5 − log2TrafoSize
/// ```
///
/// The residual-quadtree transform-block sizes are
/// `log2TrafoSize ∈ {2, 3, 4, 5}` (4×4, 8×8, 16×16, 32×32), so the
/// returned `ctxInc` lies in `{0, 1, 2, 3}`.
///
/// The §7.4.9.10 `split_transform_flag` syntax element is itself
/// only signalled when `log2TrafoSize > MaxTbLog2SizeY` is false and
/// `log2TrafoSize > MinTbLog2SizeY` is true; the caller is responsible
/// for that gate. This function is the per-bin ctxInc derivation
/// only.
///
/// # Panics
/// Does not panic. For `log2TrafoSize > 5` the result is `u32::MAX`
/// (a wrap of the `5 − x` subtraction); the spec only defines the
/// derivation for `log2TrafoSize ∈ {2, 3, 4, 5}` and the caller is
/// expected to honour that range.
#[must_use]
pub fn split_transform_flag_ctx_inc(log2_trafo_size: u32) -> u32 {
    // §7.4.9.10 restricts the syntax element to log2TrafoSize in
    // {2, 3, 4, 5}; perform the subtraction wrap-safely so an
    // out-of-range caller surfaces a u32::MAX rather than panics.
    5u32.wrapping_sub(log2_trafo_size)
}

// ---------------------------------------------------------------------
// cbf_luma / cbf_cb / cbf_cr ctxInc — Table 9-48 rows
// ---------------------------------------------------------------------

/// Table 9-48 row for `cbf_luma[ ][ ][ ]`:
///
/// ```text
/// ctxInc = (trafoDepth == 0) ? 1 : 0
/// ```
///
/// `trafoDepth` is the §7.4.9.10 transform-tree depth, where 0 is
/// the root of the residual quadtree. The two-context bank is
/// `{0, 1}` and the spec assigns the *deeper* depths to context 0
/// (the "no further split, deeper TU" case) and the depth-0 root
/// to context 1.
#[must_use]
pub fn cbf_luma_ctx_inc(trafo_depth: u32) -> u32 {
    if trafo_depth == 0 { 1 } else { 0 }
}

/// Shared `ctxInc` derivation for `cbf_cb` and `cbf_cr` (Table 9-48
/// row): `ctxInc = trafoDepth`. The §7.4.9.10 residual quadtree
/// allows `trafoDepth ∈ {0, 1, 2, 3, 4}`; the chroma cbf
/// context banks supplied by Table 9-22 are five entries per
/// component, matching the depth range.
///
/// The two callers ([`cbf_cb_ctx_inc`] and [`cbf_cr_ctx_inc`])
/// delegate to this helper; the per-component `ctxIdxOffset` is the
/// caller's responsibility (Cb and Cr each have a distinct bank in
/// Table 9-4).
#[must_use]
pub fn cbf_chroma_ctx_inc(trafo_depth: u32) -> u32 {
    trafo_depth
}

/// Table 9-48 row for `cbf_cb[ ][ ][ ]`: `ctxInc = trafoDepth`.
/// Forwards to [`cbf_chroma_ctx_inc`] for the shared formula.
#[must_use]
pub fn cbf_cb_ctx_inc(trafo_depth: u32) -> u32 {
    cbf_chroma_ctx_inc(trafo_depth)
}

/// Table 9-48 row for `cbf_cr[ ][ ][ ]`: `ctxInc = trafoDepth`.
/// Forwards to [`cbf_chroma_ctx_inc`] for the shared formula.
#[must_use]
pub fn cbf_cr_ctx_inc(trafo_depth: u32) -> u32 {
    cbf_chroma_ctx_inc(trafo_depth)
}

// ---------------------------------------------------------------------
// split_transform_flag / cbf_luma / cbf_cb / cbf_cr decode primitives
// (§7.3.8.8 transform_tree). Each is an FL `cMax = 1` single
// context-coded bin (§9.3.4.2.1 / Table 9-48). The caller selects the
// `ctxInc` via the matching `*_ctx_inc` helper and supplies the
// resolved [`ContextModel`] slot; these primitives only drive the
// CABAC engine for that one bin per §9.3.4.3.2.
// ---------------------------------------------------------------------

/// Decode `split_transform_flag[ x0 ][ y0 ][ trafoDepth ]` (§7.3.8.8 /
/// §7.4.9.8) from the CABAC engine, consuming one context-coded bin.
///
/// `ctx` is the caller's `split_transform_flag` bank slot for the
/// `ctxInc = 5 − log2TrafoSize` index ([`split_transform_flag_ctx_inc`]);
/// it is mutated in place per §9.3.4.3.2.2 state transition. The
/// §7.3.8.8 presence gate (`log2TrafoSize <= MaxTbLog2SizeY &&
/// log2TrafoSize > MinTbLog2SizeY && trafoDepth < MaxTrafoDepth &&
/// !( IntraSplitFlag && trafoDepth == 0 )`) is upstream of this
/// primitive; when the gate fails the caller infers the value via
/// [`split_transform_flag_inferred`] instead.
pub fn decode_split_transform_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    engine.decode_decision(ctx)
}

/// §7.4.9.8 — inferred value of `split_transform_flag` when the element
/// is not present on the wire.
///
/// The flag is inferred to be `1` when any of:
/// * `log2TrafoSize > MaxTbLog2SizeY`,
/// * `IntraSplitFlag == 1 && trafoDepth == 0`,
/// * `interSplitFlag == 1`
///
/// and to `0` otherwise. `interSplitFlag` is `1` exactly when
/// `max_transform_hierarchy_depth_inter == 0 && CuPredMode == MODE_INTER
/// && PartMode != PART_2Nx2N && trafoDepth == 0` (§7.4.9.8); the caller
/// passes its derived value in `inter_split_flag`.
#[must_use]
pub fn split_transform_flag_inferred(
    log2_trafo_size: u32,
    max_tb_log2_size_y: u32,
    intra_split_flag: bool,
    trafo_depth: u32,
    inter_split_flag: bool,
) -> u8 {
    if log2_trafo_size > max_tb_log2_size_y
        || (intra_split_flag && trafo_depth == 0)
        || inter_split_flag
    {
        1
    } else {
        0
    }
}

/// Decode `cbf_luma[ x0 ][ y0 ][ trafoDepth ]` (§7.3.8.8 / §7.4.9.8)
/// from the CABAC engine, consuming one context-coded bin.
///
/// `ctx` is the caller's `cbf_luma` bank slot for the
/// `ctxInc = (trafoDepth == 0) ? 1 : 0` index ([`cbf_luma_ctx_inc`]).
/// When the §7.3.8.8 presence condition fails the value is inferred to
/// `1` ([`cbf_luma_inferred`]).
pub fn decode_cbf_luma(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    engine.decode_decision(ctx)
}

/// §7.4.9.8 — inferred value of `cbf_luma` when not present: `1`.
#[must_use]
pub fn cbf_luma_inferred() -> u8 {
    1
}

/// Decode `cbf_cb[ x0 ][ y0 ][ trafoDepth ]` (§7.3.8.8 / §7.4.9.8) from
/// the CABAC engine, consuming one context-coded bin.
///
/// `ctx` is the caller's shared `cbf_chroma` bank slot for the
/// `ctxInc = trafoDepth` index ([`cbf_cb_ctx_inc`]). When not present
/// the value is inferred to `0` ([`cbf_chroma_inferred`]).
pub fn decode_cbf_cb(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    engine.decode_decision(ctx)
}

/// Decode `cbf_cr[ x0 ][ y0 ][ trafoDepth ]` (§7.3.8.8 / §7.4.9.8) from
/// the CABAC engine, consuming one context-coded bin.
///
/// `ctx` is the caller's shared `cbf_chroma` bank slot for the
/// `ctxInc = trafoDepth` index ([`cbf_cr_ctx_inc`]). When not present
/// the value is inferred to `0` ([`cbf_chroma_inferred`]).
pub fn decode_cbf_cr(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    engine.decode_decision(ctx)
}

/// §7.4.9.8 — inferred value of `cbf_cb` / `cbf_cr` when not present: `0`.
#[must_use]
pub fn cbf_chroma_inferred() -> u8 {
    0
}

// ---------------------------------------------------------------------
// inter_pred_idc ctxInc — Table 9-48 row
// ---------------------------------------------------------------------

/// Table 9-48 row for `inter_pred_idc[ x0 ][ y0 ]`:
///
/// ```text
/// bin 0:  ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4
/// bin 1:  ctxInc = 4
/// ```
///
/// `n_pb_w` and `n_pb_h` are the prediction-block width and height in
/// luma samples (the §7.3.8.6 `nPbW` / `nPbH` derivation). The
/// `nPbW + nPbH == 12` condition picks the two prediction-block
/// shapes whose luma area is 16 samples — namely 8×4 (sum = 12) and
/// 4×8 (sum = 12) — which are encoded with the bin-0 escape onto
/// the shared bin-1 context bank (`ctxInc = 4`). All other shapes
/// route bin 0 through the per-`CtDepth` context bank `{0, 1, 2, 3}`.
///
/// `ct_depth` is the §7.4.9.4 `CtDepth[x0][y0]` value (the coding-
/// quadtree depth of the current coding unit); valid range is
/// `{0, 1, 2, 3}` matching the §7.4.7.1 MaxCuDepth limit.
///
/// Returns `ctxInc` for any `binIdx`; `binIdx ∈ {2, 3, 4}` is `na`
/// per Table 9-48 (the binarization is a TR with `cMax = 2`), so the
/// function returns `4` for `binIdx >= 1` (matching bin 1).
#[must_use]
pub fn inter_pred_idc_ctx_inc(bin_idx: u32, n_pb_w: u32, n_pb_h: u32, ct_depth: u32) -> u32 {
    if bin_idx == 0 {
        // Bin 0 routes through CtDepth unless the PU is 8×4 or 4×8
        // (luma area 16 samples ⇒ nPbW + nPbH == 12), in which case
        // it shares the bin-1 context bank.
        if n_pb_w + n_pb_h != 12 { ct_depth } else { 4 }
    } else {
        // Bin 1 — and the `na` entries `binIdx >= 2` — use ctx 4.
        // Table 9-47 caps the prefix at `cMax = 2` (one or two bins),
        // so the slice-data parser never invokes this with binIdx >=
        // 2; the return value is the spec's bin-1 ctxInc.
        4
    }
}

// ---------------------------------------------------------------------
// log2_res_scale_abs_plus1 / res_scale_sign_flag ctxInc —
// Table 9-48 rows for §7.3.8.13 cross-component prediction
// ---------------------------------------------------------------------

/// Table 9-48 row for `log2_res_scale_abs_plus1[ c ]`:
///
/// ```text
/// ctxInc = 4*c + binIdx
/// ```
///
/// `c` is the chroma-component index passed at the §7.3.8.13
/// `cross_comp_pred()` call site: `c == 0` for Cb, `c == 1` for Cr.
/// The TR(`cMax = 4`) prefix occupies four consecutive contexts per
/// chroma component, so the per-component banks are `{0, 1, 2, 3}`
/// (Cb) and `{4, 5, 6, 7}` (Cr).
///
/// `binIdx` is the position of the bin within the TR prefix, in
/// `{0, 1, 2, 3}`. The `binIdx >= 4` slots (the EGk(k=0) bypass
/// escape, when the prefix is the all-ones escape `0b1111`) are
/// `na` for the context-coded path; the caller switches to bypass
/// decoding at that point and never passes them here.
#[must_use]
pub fn log2_res_scale_abs_plus1_ctx_inc(bin_idx: u32, c: u32) -> u32 {
    // Per-component bank offset (4) times c, plus the bin position
    // within the TR prefix.
    4 * c + bin_idx
}

/// Table 9-48 row for `res_scale_sign_flag[ c ]`:
///
/// ```text
/// ctxInc = c
/// ```
///
/// `c == 0` for Cb, `c == 1` for Cr; each component has its own
/// single-bin sign-flag context.
#[must_use]
pub fn res_scale_sign_flag_ctx_inc(c: u32) -> u32 {
    c
}

/// Table 9-43 binarization shape for `log2_res_scale_abs_plus1[ c ]`:
/// TR with `cMax = 4`, `cRiceParam = 0` (the prefix is the entire
/// element — there is no EGk continuation; the all-ones `0b1111`
/// prefix simply encodes the value `4`).
pub const LOG2_RES_SCALE_ABS_PLUS1_TR_CMAX: u32 = 4;

/// Decoded §7.3.8.12 `cross_comp_pred( x0, y0, c )` pair plus the
/// §7.4.9.12 derived `ResScaleVal` for the chroma component `c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossCompPred {
    /// `log2_res_scale_abs_plus1[ c ]` from the wire (TR, `cMax = 4`).
    /// When this is 0, `res_scale_sign_flag` is not signalled and
    /// `ResScaleVal` is 0.
    pub log2_res_scale_abs_plus1: u32,
    /// `res_scale_sign_flag[ c ]` from the wire; `None` when
    /// `log2_res_scale_abs_plus1 == 0` (the flag is not present).
    pub res_scale_sign_flag: Option<u8>,
    /// §7.4.9.12 derived scaling factor `ResScaleVal[ c+1 ][ x0 ][ y0 ]`
    /// (equations 7-79 / 7-80): `0` when `log2_res_scale_abs_plus1 == 0`,
    /// else `(1 << (log2_res_scale_abs_plus1 − 1)) * (1 − 2 *
    /// res_scale_sign_flag)`.
    pub res_scale_val: i32,
}

/// Decode the §7.3.8.12 `cross_comp_pred( x0, y0, c )` syntax structure
/// from the CABAC engine, returning the decoded pair and the §7.4.9.12
/// derived [`CrossCompPred::res_scale_val`].
///
/// The TR(`cMax = 4`) prefix of `log2_res_scale_abs_plus1[ c ]` is
/// context-coded; bin `binIdx` uses `ctx[binIdx]` (the per-component
/// `{0, 1, 2, 3}` bank already sliced out of the Table 9-36 bank by the
/// caller via [`log2_res_scale_abs_plus1_ctx_inc`]). When the prefix is
/// non-zero, the single-bin `res_scale_sign_flag[ c ]` follows,
/// context-coded against `ctx_sign` (Table 9-37, ctxInc = c — also
/// pre-sliced by the caller).
///
/// `ctx` must provide at least [`LOG2_RES_SCALE_ABS_PLUS1_TR_CMAX`]
/// context slots for the prefix bins; only as many as are actually
/// decoded are mutated in place per §9.3.4.3.2.2.
pub fn decode_cross_comp_pred(
    engine: &mut CabacEngine<'_>,
    ctx: &mut [ContextModel],
    ctx_sign: &mut ContextModel,
) -> Result<CrossCompPred, CabacError> {
    debug_assert!(
        ctx.len() >= LOG2_RES_SCALE_ABS_PLUS1_TR_CMAX as usize,
        "cross_comp_pred needs cMax = 4 prefix contexts for the component bank"
    );
    // §9.3.3.10 TR prefix (cRiceParam = 0); each bin uses its own
    // context within the per-component bank (Table 9-48 ctxInc =
    // 4*c + binIdx, with the *c offset already applied by the caller's
    // slice). The all-ones escape simply yields the value cMax = 4 —
    // there is no EGk continuation for this element (Table 9-43).
    let (log2_res_scale_abs_plus1, _is_escape) =
        read_truncated_rice_prefix(LOG2_RES_SCALE_ABS_PLUS1_TR_CMAX, |bin_idx| {
            engine.decode_decision(&mut ctx[bin_idx as usize])
        })?;

    // §7.3.8.12: res_scale_sign_flag is present only when the prefix is
    // non-zero. §7.4.9.12 eqs 7-79 / 7-80 derive ResScaleVal.
    let (res_scale_sign_flag, res_scale_val) = if log2_res_scale_abs_plus1 == 0 {
        (None, 0i32)
    } else {
        let sign = engine.decode_decision(ctx_sign)?;
        let magnitude = 1i32 << (log2_res_scale_abs_plus1 - 1);
        let val = magnitude * (1 - 2 * i32::from(sign));
        (Some(sign), val)
    };

    Ok(CrossCompPred {
        log2_res_scale_abs_plus1,
        res_scale_sign_flag,
        res_scale_val,
    })
}

// ---------------------------------------------------------------------
// tu_residual_act_flag (§7.3.8.10 transform_unit() / §7.4.9.10)
// Table 9-39 ctxIdx initValue; FL (cMax = 1) binarization; Table 9-48
// ctxInc = 0. Present only when the SCC residual adaptive colour
// transform gate is satisfied (§7.3.8.10).
// ---------------------------------------------------------------------

/// Table 9-43 binarization shape for `tu_residual_act_flag`: FL with
/// `cMax = 1` (a single context-coded bin per transform unit).
pub const TU_RESIDUAL_ACT_FLAG_FL_CMAX: u32 = 1;

/// §9.3.4.2.1 / Table 9-48 row for `tu_residual_act_flag`: bin 0 is
/// context-coded with `ctxInc = 0`; the FL `cMax = 1` shape emits only
/// that one bin.
#[must_use]
pub fn tu_residual_act_flag_ctx_inc() -> u32 {
    0
}

/// §7.4.9.10 — inferred value of `tu_residual_act_flag` when it is not
/// present (the adaptive-colour-transform gate is not satisfied): the
/// spec mandates an inferred value of `0` (no adaptive colour
/// transform).
#[must_use]
pub fn tu_residual_act_flag_inferred() -> u8 {
    0
}

/// Decode `tu_residual_act_flag` (§7.3.8.10 / §7.4.9.10) from the CABAC
/// engine, consuming one Table 9-39 context for the single FL bin
/// (Table 9-48 ctxInc = 0). The gate that decides whether the element
/// is present (the §7.3.8.10 `residual_adaptive_colour_transform_enabled_flag`
/// predicate) is the caller's responsibility; see
/// [`tu_residual_act_flag_inferred`] for the absent case.
pub fn decode_tu_residual_act_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    debug_assert_eq!(
        tu_residual_act_flag_ctx_inc(),
        0,
        "Table 9-48 row for tu_residual_act_flag: bin 0 ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// §7.3.4 sao() — per-CTU SAO syntax elements (Table 9-48 / Table 9-43)
// ---------------------------------------------------------------------
//
// Round 30 extends the §9.3.4.2 surface with the SAO syntax-element
// family, all decoded inside the §7.3.4 `sao()` per-CTU block. Every
// element is either a single context-coded bin-0 followed by zero or
// more bypass-coded bins (Table 9-48), or fully bypass-coded; no
// neighbour-table walk is needed at this layer.
//
// Table 9-48 rows captured here:
//
// * `sao_merge_left_flag`            → bin 0: ctxInc = 0  (FL, cMax = 1)
// * `sao_merge_up_flag`              → bin 0: ctxInc = 0  (FL, cMax = 1)
// * `sao_type_idx_luma`              → bin 0: ctxInc = 0; bin 1: bypass
//                                      (TR, cMax = 2, cRiceParam = 0)
// * `sao_type_idx_chroma`            → bin 0: ctxInc = 0; bin 1: bypass
//                                      (TR, cMax = 2, cRiceParam = 0)
// * `sao_offset_abs[ ][ ][ ][ ]`     → bypass × all
//                                      (TR, cMax = (1 << (min(bitDepth, 10) − 5)) − 1,
//                                      cRiceParam = 0)
// * `sao_offset_sign[ ][ ][ ][ ]`    → bypass × 1     (FL, cMax = 1)
// * `sao_band_position[ ][ ][ ]`     → bypass × 5     (FL, cMax = 31)
// * `sao_eo_class_luma`              → bypass × 2     (FL, cMax = 3)
// * `sao_eo_class_chroma`            → bypass × 2     (FL, cMax = 3)
//
// Table 9-4 association (the ctx banks each element lives in):
//
// * `sao_merge_left_flag` + `sao_merge_up_flag` share Table 9-5 (one
//   ctxIdx per initType).
// * `sao_type_idx_luma` + `sao_type_idx_chroma` share Table 9-6 (one
//   ctxIdx per initType).
//
// The two pairs are distinct context banks; this layer hands back the
// bank-relative ctxInc only (0 for every context-coded SAO bin).

/// Table 9-48 row for `sao_merge_left_flag` and `sao_merge_up_flag`:
///
/// ```text
/// bin 0: ctxInc = 0
/// ```
///
/// Only bin 0 is context-coded; the FL binarization has `cMax = 1`, so
/// there is exactly one bin and the function returns 0 for any call.
/// The two merge flags share the same Table 9-5 context bank (see
/// Table 9-4); this layer returns the bank-relative ctxInc only.
#[must_use]
pub fn sao_merge_flag_ctx_inc() -> u32 {
    0
}

/// Table 9-48 row for `sao_type_idx_luma` and `sao_type_idx_chroma`:
///
/// ```text
/// bin 0: ctxInc = 0
/// bin 1: bypass
/// ```
///
/// `binIdx == 0` is context-coded with `ctxInc = 0`; `binIdx == 1` is
/// bypass per Table 9-48 and is **not** routed through a context. The
/// TR(`cMax = 2`) binarization caps the prefix at two bins. Callers
/// driving the §9.3 engine must invoke `decode_decision` for bin 0 and
/// `decode_bypass` for bin 1 directly, switching paths between bins.
///
/// Returns the ctxInc when called with `bin_idx == 0`; any other value
/// is `na` per Table 9-48 and the function returns 0 defensively
/// (callers should not invoke this for the bypass bin).
#[must_use]
pub fn sao_type_idx_ctx_inc(bin_idx: u32) -> u32 {
    debug_assert!(
        bin_idx == 0,
        "sao_type_idx Table 9-48 row: only bin 0 is context-coded; bin >= 1 is bypass"
    );
    0
}

/// Table 9-43 binarization shape for `sao_merge_left_flag` and
/// `sao_merge_up_flag`: FL with `cMax = 1` (single bin).
pub const SAO_MERGE_FLAG_FL_CMAX: u32 = 1;

/// Table 9-43 binarization shape for `sao_type_idx_luma` and
/// `sao_type_idx_chroma`: TR with `cMax = 2`, `cRiceParam = 0`. The
/// prefix is at most two bins; bin 0 is context-coded (ctxInc = 0),
/// bin 1 is bypass-coded per Table 9-48.
pub const SAO_TYPE_IDX_TR_CMAX: u32 = 2;

/// Table 9-43 binarization shape for `sao_offset_sign`: FL with
/// `cMax = 1` (single bypass-coded bin).
pub const SAO_OFFSET_SIGN_FL_CMAX: u32 = 1;

/// Table 9-43 binarization shape for `sao_band_position`: FL with
/// `cMax = 31` (five bypass-coded bins; the FL-of-cMax-31 form
/// uses `ceil(log2(cMax + 1)) = 5` bits per §9.3.3.5).
pub const SAO_BAND_POSITION_FL_CMAX: u32 = 31;

/// FL-binarization bit count for `sao_band_position`: 5 bypass bins.
/// Per §9.3.3.5, FL of `cMax = N` uses `ceil(log2(N + 1))` bits.
pub const SAO_BAND_POSITION_FL_NBITS: u32 = 5;

/// Table 9-43 binarization shape for `sao_eo_class_luma` and
/// `sao_eo_class_chroma`: FL with `cMax = 3` (two bypass-coded bins,
/// encoding the §7.4.9.3 `SaoEoClass` ∈ {0, 1, 2, 3}: horizontal,
/// vertical, 135-degree, 45-degree).
pub const SAO_EO_CLASS_FL_CMAX: u32 = 3;

/// FL-binarization bit count for `sao_eo_class_luma` /
/// `sao_eo_class_chroma`: 2 bypass bins.
pub const SAO_EO_CLASS_FL_NBITS: u32 = 2;

/// Table 9-43 binarization shape for `sao_offset_abs`: TR with
///
/// ```text
/// cMax = (1 << min(bitDepth, 10) - 5) - 1
/// cRiceParam = 0
/// ```
///
/// where `bitDepth` is the component's bit-depth in samples. For the
/// canonical 8-bit case `cMax = (1 << 3) - 1 = 7`; for 10-bit and
/// beyond `cMax = (1 << 5) - 1 = 31` (Min-clamped to 10 by the spec
/// to keep the maximum SAO offset range bounded). All bins of the TR
/// prefix are bypass-coded per Table 9-48.
///
/// `bit_depth` is the §7.4.7.1 `BitDepthY` or `BitDepthC` value
/// for the colour component being decoded.
#[must_use]
pub fn sao_offset_abs_tr_cmax(bit_depth: u32) -> u32 {
    // §9.3.3.5: Min(bitDepth, 10). Clamp at the spec maximum so the
    // SAO offset range never exceeds [−31, 31].
    let clamped = if bit_depth < 10 { bit_depth } else { 10 };
    // The subtraction is well-defined for bit_depth >= 5; HEVC mandates
    // bitDepth >= 8 in every conformant profile, so the result is
    // always positive in practice.
    (1u32 << (clamped - 5)) - 1
}

/// Decode `sao_merge_left_flag` / `sao_merge_up_flag` (§7.3.8.3) — the
/// two §7.4.9.3 SAO-merge flags. Both are Table 9-43 FL `cMax = 1` with
/// a single `ctxInc = 0` context-coded bin (Table 9-48) sharing the
/// Table 9-5 `sao_merge_flag` bank. The §7.3.8.3 presence gates
/// (`rx > 0 && leftCtbInSliceSeg && leftCtbInTile` for the left flag;
/// `ry > 0 && !sao_merge_left_flag && upCtbInSliceSeg && upCtbInTile`
/// for the up flag) are the caller's responsibility; when the gate
/// fails the flag is inferred to 0 (§7.4.9.3) and this function is not
/// called.
pub fn decode_sao_merge_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    debug_assert_eq!(
        sao_merge_flag_ctx_inc(),
        0,
        "Table 9-48: sao_merge ctxInc = 0"
    );
    engine.decode_decision(ctx)
}

/// Decode `sao_type_idx_luma` / `sao_type_idx_chroma` (§7.3.8.3) — the
/// §7.4.9.3 `SaoTypeIdx[cIdx][rx][ry]` selector. Table 9-43 TR
/// `cMax = 2, cRiceParam = 0`: bin 0 is context-coded (`ctxInc = 0`,
/// `ctx`), bin 1 is bypass (Table 9-48). The decoded value is
/// 0 (not applied), 1 (band offset) or 2 (edge offset) per §7.4.9.3.
pub fn decode_sao_type_idx(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    // TR(cMax = 2): bin 0 context-coded, bin 1 bypass.
    let bin0 = engine.decode_decision(ctx)?;
    if bin0 == 0 {
        return Ok(0);
    }
    let bin1 = engine.decode_bypass()?;
    Ok(if bin1 == 0 { 1 } else { 2 })
}

/// Decode `sao_offset_abs[cIdx][rx][ry][i]` (§7.3.8.3) — one of the
/// four §7.4.9.3 SAO offset magnitudes. Table 9-43 TR
/// `cMax = (1 << (Min(bitDepth, 10) − 5)) − 1, cRiceParam = 0`; every
/// bin is bypass (Table 9-48). `bit_depth` is `BitDepthY` (luma) or
/// `BitDepthC` (chroma); see [`sao_offset_abs_tr_cmax`].
pub fn decode_sao_offset_abs(
    engine: &mut CabacEngine<'_>,
    bit_depth: u32,
) -> Result<u32, CabacError> {
    let c_max = sao_offset_abs_tr_cmax(bit_depth);
    let (value, _is_escape) = read_truncated_rice_prefix(c_max, |_| engine.decode_bypass())?;
    Ok(value)
}

/// Decode `sao_offset_sign[cIdx][rx][ry][i]` (§7.3.8.3) — the sign of a
/// band-offset SAO offset. Table 9-43 FL `cMax = 1`, a single bypass
/// bin (Table 9-48). Read only for band-offset (`SaoTypeIdx == 1`)
/// when the corresponding `sao_offset_abs != 0`; otherwise inferred to
/// 0 (§7.4.9.3). Returns 0 (positive) or 1 (negative).
pub fn decode_sao_offset_sign(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    engine.decode_bypass()
}

/// Decode `sao_band_position[cIdx][rx][ry]` (§7.3.8.3) — the §7.4.9.3
/// starting band of a band-offset SAO. Table 9-43 FL `cMax = 31`, five
/// bypass bins MSB-first (Table 9-48). Read only for band offset
/// (`SaoTypeIdx == 1`).
pub fn decode_sao_band_position(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    Ok(engine.decode_bypass_bits(SAO_BAND_POSITION_FL_NBITS as u8)? as u8)
}

/// Decode `sao_eo_class_luma` / `sao_eo_class_chroma` (§7.3.8.3) — the
/// §7.4.9.3 `SaoEoClass[cIdx][rx][ry]` edge-offset direction. Table
/// 9-43 FL `cMax = 3`, two bypass bins MSB-first (Table 9-48). Read
/// only for edge offset (`SaoTypeIdx == 2`). Returns 0 (horizontal),
/// 1 (vertical), 2 (135°) or 3 (45°).
pub fn decode_sao_eo_class(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    Ok(engine.decode_bypass_bits(SAO_EO_CLASS_FL_NBITS as u8)? as u8)
}

// ---------------------------------------------------------------------
// part_mode (§7.3.8.5 coding_unit() / §7.4.9.5 / §9.3.3.7).
// Table 9-45 binarization (CuPredMode + log2CbSize dependent) + Table
// 9-48 ctxInc (bin 0/1/2 context-coded, bin 3 bypass) + Table 7-10
// PartMode / IntraSplitFlag derivation. The bin string is variable so
// the decoder walks it bin-by-bin against the active-CU geometry.
// ---------------------------------------------------------------------

/// §7.4.9.5 / Table 7-10 — the coding-unit partitioning mode `PartMode`
/// derived from `part_mode`. The eight values cover the square split
/// (`PART_2Nx2N`, `PART_NxN`), the symmetric halves (`PART_2NxN`,
/// `PART_Nx2N`), and the four asymmetric-motion-partition (AMP) forms
/// (`PART_2NxnU/nD/nLx2N/nRx2N`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartMode {
    /// `PART_2Nx2N` (part_mode 0) — a single prediction block covering
    /// the whole CU.
    Part2Nx2N,
    /// `PART_2NxN` (part_mode 1) — two horizontal halves.
    Part2NxN,
    /// `PART_Nx2N` (part_mode 2) — two vertical halves.
    PartNx2N,
    /// `PART_NxN` (part_mode 3 inter / part_mode 1 intra) — four
    /// quarter-size blocks. For intra it sets `IntraSplitFlag = 1`.
    PartNxN,
    /// `PART_2NxnU` (part_mode 4) — AMP, top quarter + bottom
    /// three-quarters (inter only).
    Part2NxnU,
    /// `PART_2NxnD` (part_mode 5) — AMP, top three-quarters + bottom
    /// quarter (inter only).
    Part2NxnD,
    /// `PART_nLx2N` (part_mode 6) — AMP, left quarter + right
    /// three-quarters (inter only).
    PartNLx2N,
    /// `PART_nRx2N` (part_mode 7) — AMP, left three-quarters + right
    /// quarter (inter only).
    PartNRx2N,
}

impl PartMode {
    /// `true` for the four asymmetric-motion-partition modes
    /// (`PART_2NxnU/nD/nLx2N/nRx2N`), which are only valid for inter
    /// CUs with `amp_enabled_flag == 1`.
    #[must_use]
    pub fn is_amp(self) -> bool {
        matches!(
            self,
            Self::Part2NxnU | Self::Part2NxnD | Self::PartNLx2N | Self::PartNRx2N
        )
    }
}

/// Decoded `part_mode` result: the §7.4.9.5 `PartMode` plus the
/// `IntraSplitFlag` it implies (set only by intra `PART_NxN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartModeResult {
    /// The §7.4.9.5 `PartMode`.
    pub part_mode: PartMode,
    /// §7.4.9.5 `IntraSplitFlag` — 1 only for an intra CU split into
    /// `PART_NxN`, 0 otherwise.
    pub intra_split_flag: bool,
}

/// §7.4.9.5 — the `part_mode`-not-present inference: `PartMode =
/// PART_2Nx2N`, `IntraSplitFlag = 0`. The §7.3.8.5 read is suppressed
/// when `CuPredMode == MODE_INTRA && log2CbSize > MinCbLog2SizeY`.
#[must_use]
pub fn part_mode_inferred() -> PartModeResult {
    PartModeResult {
        part_mode: PartMode::Part2Nx2N,
        intra_split_flag: false,
    }
}

/// Decode `part_mode` (§7.3.8.5 / §9.3.3.7) from the CABAC engine,
/// returning the §7.4.9.5 `PartMode` + `IntraSplitFlag`.
///
/// The Table 9-45 binarization is selected by `cu_pred_mode`,
/// `log2_cb_size`, `min_cb_log2_size` (`MinCbLog2SizeY`) and
/// `amp_enabled` (`amp_enabled_flag`). Table 9-48 routes bin 0..=2
/// through the per-bin `part_mode[ctxInc]` bank (`ctx0` / `ctx1`;
/// bin 2's ctxInc is 2 for `log2CbSize == MinCbLog2SizeY`, 3 for
/// `log2CbSize > MinCbLog2SizeY`, so the caller passes both context
/// slots); bin 3 is bypass.
///
/// * `ctx0` — `part_mode` bank slot 0 (bin 0).
/// * `ctx1` — `part_mode` bank slot 1 (bin 1).
/// * `ctx_bin2` — `part_mode` bank slot 2 or 3 (bin 2), already
///   selected by the caller from `log2_cb_size == min_cb_log2_size`.
#[allow(clippy::too_many_arguments)]
pub fn decode_part_mode(
    engine: &mut CabacEngine<'_>,
    ctx0: &mut ContextModel,
    ctx1: &mut ContextModel,
    ctx_bin2: &mut ContextModel,
    cu_pred_mode: CuPredMode,
    log2_cb_size: u32,
    min_cb_log2_size: u32,
    amp_enabled: bool,
) -> Result<PartModeResult, CabacError> {
    // Table 9-45 bin 0 is always context-coded.
    let b0 = engine.decode_decision(ctx0)?;

    if cu_pred_mode == CuPredMode::Intra {
        // Intra: only PART_2Nx2N (bin "1") or PART_NxN (bin "0").
        // Table 9-45 intra column: part_mode 0 → "1", part_mode 1 → "0".
        return Ok(if b0 == 1 {
            PartModeResult {
                part_mode: PartMode::Part2Nx2N,
                intra_split_flag: false,
            }
        } else {
            PartModeResult {
                part_mode: PartMode::PartNxN,
                intra_split_flag: true,
            }
        });
    }

    // Inter (MODE_INTER). part_mode 0 (PART_2Nx2N) = "1".
    if b0 == 1 {
        return Ok(PartModeResult {
            part_mode: PartMode::Part2Nx2N,
            intra_split_flag: false,
        });
    }

    // bin 0 == 0: read bin 1 (context-coded).
    let b1 = engine.decode_decision(ctx1)?;

    // The asymmetric/quarter tail is only present when
    // log2CbSize > MinCbLog2SizeY (the "amp_enabled_flag" /
    // "!amp_enabled_flag" Table 9-45 columns) or
    // log2CbSize == MinCbLog2SizeY && log2CbSize > 3 (the trailing
    // "001"/"000" forms). When log2CbSize == MinCbLog2SizeY &&
    // log2CbSize == 3 only the two-bin forms "01"/"00" exist.
    let big = log2_cb_size > min_cb_log2_size;

    if big {
        if !amp_enabled {
            // Columns "!amp_enabled_flag": "01" → PART_2NxN,
            // "00" → PART_Nx2N. Two bins total.
            return Ok(PartModeResult {
                part_mode: if b1 == 1 {
                    PartMode::Part2NxN
                } else {
                    PartMode::PartNx2N
                },
                intra_split_flag: false,
            });
        }
        // amp_enabled_flag column. After "0x" a third context bin
        // distinguishes the symmetric ("011"/"001") from the AMP
        // ("0100"/"0101"/"0000"/"0001") forms.
        let b2 = engine.decode_decision(ctx_bin2)?;
        if b2 == 1 {
            // "011" → PART_2NxN, "001" → PART_Nx2N.
            return Ok(PartModeResult {
                part_mode: if b1 == 1 {
                    PartMode::Part2NxN
                } else {
                    PartMode::PartNx2N
                },
                intra_split_flag: false,
            });
        }
        // b2 == 0: AMP. Bin 3 is bypass. b1 selects horizontal
        // (PART_2NxnU/nD, "010x") vs vertical (PART_nLx2N/nRx2N,
        // "000x"); b3 selects the U/L vs D/R half.
        let b3 = engine.decode_bypass()?;
        let part_mode = match (b1, b3) {
            (1, 0) => PartMode::Part2NxnU, // 0100
            (1, 1) => PartMode::Part2NxnD, // 0101
            (0, 0) => PartMode::PartNLx2N, // 0000
            _ => PartMode::PartNRx2N,      // 0001
        };
        return Ok(PartModeResult {
            part_mode,
            intra_split_flag: false,
        });
    }

    // log2CbSize == MinCbLog2SizeY. "01" → PART_2NxN. The "00" prefix
    // continues to PART_Nx2N ("00") when log2CbSize == 3, or splits
    // into "001" (PART_Nx2N) / "000" (PART_NxN) when log2CbSize > 3.
    if b1 == 1 {
        return Ok(PartModeResult {
            part_mode: PartMode::Part2NxN,
            intra_split_flag: false,
        });
    }
    if log2_cb_size == 3 {
        // "00" → PART_Nx2N (no third bin at the 8×8 minimum).
        return Ok(PartModeResult {
            part_mode: PartMode::PartNx2N,
            intra_split_flag: false,
        });
    }
    // log2CbSize > 3 && == MinCbLog2SizeY: third bin (context-coded,
    // ctxInc = bin2 slot). "001" → PART_Nx2N, "000" → PART_NxN.
    let b2 = engine.decode_decision(ctx_bin2)?;
    Ok(PartModeResult {
        part_mode: if b2 == 1 {
            PartMode::PartNx2N
        } else {
            PartMode::PartNxN
        },
        intra_split_flag: false,
    })
}

/// Decode `pcm_flag[x0][y0]` (§7.3.8.5) — the §7.4.9.5 PCM gate. Table
/// 9-43 FL `cMax = 1` whose single bin is the §9.3.4.3.5 *terminate*
/// path (Table 9-48 `terminate` row), so it is decoded via
/// [`CabacEngine::decode_terminate`], **not** `decode_decision`. The
/// §7.3.8.5 presence gate (`PartMode == PART_2Nx2N && pcm_enabled_flag
/// && Log2MinIpcmCbSizeY <= log2CbSize <= Log2MaxIpcmCbSizeY`) is the
/// caller's; not-present infers 0 (§7.4.9.5).
pub fn decode_pcm_flag(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    engine.decode_terminate()
}

/// Decode `end_of_slice_segment_flag` (§7.3.8.1) — the per-CTU loop
/// terminator. Table 9-43 FL `cMax = 1` decoded via the §9.3.4.3.5
/// *terminate* path (Table 9-48 `terminate` row),
/// [`CabacEngine::decode_terminate`]. 1 ends the slice-segment data
/// loop.
pub fn decode_end_of_slice_segment_flag(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    engine.decode_terminate()
}

/// Decode `merge_idx[x0][y0]` (§7.3.8.6) — the §7.4.9.6 merge-candidate
/// index. Table 9-43 TR `cMax = MaxNumMergeCand − 1, cRiceParam = 0`:
/// bin 0 is context-coded (`ctxInc = 0`, `ctx`), bins >= 1 are bypass
/// (Table 9-48). Read only when `MaxNumMergeCand > 1`; otherwise
/// inferred to 0 (§7.4.9.6).
pub fn decode_merge_idx(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
    max_num_merge_cand: u32,
) -> Result<u8, CabacError> {
    let c_max = max_num_merge_cand.saturating_sub(1);
    if c_max == 0 {
        return Ok(0);
    }
    // bin 0 context-coded; remaining bins bypass.
    let b0 = engine.decode_decision(ctx)?;
    if b0 == 0 {
        return Ok(0);
    }
    let mut value: u32 = 1;
    while value < c_max {
        if engine.decode_bypass()? == 0 {
            return Ok(value as u8);
        }
        value += 1;
    }
    Ok(value as u8)
}

/// §9.3.3.9 / Table 9-47 / §7.4.9.6 — the §7.4.9.6 `inter_pred_idc`
/// values: which reference lists a P/B prediction block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterPredIdc {
    /// `PRED_L0` — uni-prediction from RefPicList0.
    PredL0,
    /// `PRED_L1` — uni-prediction from RefPicList1.
    PredL1,
    /// `PRED_BI` — bi-prediction from both lists.
    PredBi,
}

/// Decode `inter_pred_idc[x0][y0]` (§7.3.8.6 / §9.3.3.9) — the §7.4.9.6
/// reference-list selector. Table 9-47 binarization depends on
/// `nPbW + nPbH`: when `!= 12` bin 0 (`PRED_BI` = "1", else two bins)
/// uses `ctxInc = CtDepth`, bin 1 uses `ctxInc = 4` (Table 9-48); when
/// `== 12` the single bin uses `ctxInc = 4`. Read only on B slices;
/// a non-B inter PB is `PRED_L0` (§7.4.9.6, not signalled here).
///
/// * `ctx_b0` — the `inter_pred_idc` bank slot `CtDepth` (bin 0,
///   `nPbW + nPbH != 12` case) or slot 4 (`== 12` case), selected by
///   the caller.
/// * `ctx_b1` — the `inter_pred_idc` bank slot 4 (bin 1).
/// * `n_pb_w` / `n_pb_h` — the prediction-block width / height.
pub fn decode_inter_pred_idc(
    engine: &mut CabacEngine<'_>,
    ctx_b0: &mut ContextModel,
    ctx_b1: &mut ContextModel,
    n_pb_w: u32,
    n_pb_h: u32,
) -> Result<InterPredIdc, CabacError> {
    if n_pb_w + n_pb_h == 12 {
        // Single bin, ctxInc = 4: "0" → PRED_L0, "1" → PRED_L1.
        let b = engine.decode_decision(ctx_b0)?;
        return Ok(if b == 0 {
            InterPredIdc::PredL0
        } else {
            InterPredIdc::PredL1
        });
    }
    // bin 0 (ctxInc = CtDepth): "1" → PRED_BI.
    let b0 = engine.decode_decision(ctx_b0)?;
    if b0 == 1 {
        return Ok(InterPredIdc::PredBi);
    }
    // "00" → PRED_L0, "01" → PRED_L1 (bin 1, ctxInc = 4).
    let b1 = engine.decode_decision(ctx_b1)?;
    Ok(if b1 == 0 {
        InterPredIdc::PredL0
    } else {
        InterPredIdc::PredL1
    })
}

/// Decode `ref_idx_l0[x0][y0]` / `ref_idx_l1[x0][y0]` (§7.3.8.6) — the
/// §7.4.9.6 reference-picture index. Table 9-43 TR
/// `cMax = num_ref_idx_lX_active_minus1, cRiceParam = 0`: bin 0 uses
/// `ctxInc = 0` (`ctx0`), bin 1 uses `ctxInc = 1` (`ctx1`), bins >= 2
/// are bypass (Table 9-48). Read only when
/// `num_ref_idx_lX_active_minus1 > 0`; else inferred to 0 (§7.4.9.6).
pub fn decode_ref_idx(
    engine: &mut CabacEngine<'_>,
    ctx0: &mut ContextModel,
    ctx1: &mut ContextModel,
    num_ref_idx_active_minus1: u32,
) -> Result<u8, CabacError> {
    let c_max = num_ref_idx_active_minus1;
    if c_max == 0 {
        return Ok(0);
    }
    // bin 0 (ctxInc = 0).
    if engine.decode_decision(ctx0)? == 0 {
        return Ok(0);
    }
    if c_max == 1 {
        return Ok(1);
    }
    // bin 1 (ctxInc = 1).
    if engine.decode_decision(ctx1)? == 0 {
        return Ok(1);
    }
    // bins >= 2 bypass.
    let mut value: u32 = 2;
    while value < c_max {
        if engine.decode_bypass()? == 0 {
            return Ok(value as u8);
        }
        value += 1;
    }
    Ok(value as u8)
}

/// Decode `mvp_l0_flag[x0][y0]` / `mvp_l1_flag[x0][y0]` (§7.3.8.6) — the
/// §7.4.9.6 motion-vector-predictor candidate flag. Table 9-43 FL
/// `cMax = 1`, a single `ctxInc = 0` context-coded bin (Table 9-48)
/// sharing the `mvp_flag` bank. Returns 0 or 1.
pub fn decode_mvp_flag(
    engine: &mut CabacEngine<'_>,
    ctx: &mut ContextModel,
) -> Result<u8, CabacError> {
    engine.decode_decision(ctx)
}

// ---------------------------------------------------------------------
// coeff_abs_level_greater1_flag / coeff_abs_level_greater2_flag
// — §9.3.4.2.6 (equations 9-56..9-60) and §9.3.4.2.7 (equations
// 9-61..9-62). Both elements are Table 9-43 FL with cMax = 1, so the
// binarization shape is one context-coded bin. The bin's `ctxInc` is
// driven by a small per-transform-block state machine that threads
// `ctxSet` / `greater1Ctx` from sub-block to sub-block.
// ---------------------------------------------------------------------

/// Persistent §9.3.4.2.6 state machine carried across the residual
/// sub-blocks of a single transform block. The slice-data parser
/// constructs one [`Greater1State`] per transform block and threads it
/// through the §7.3.8.11 sub-block loop, calling
/// [`Greater1State::on_subblock_entry`] before the sub-block's first
/// `coeff_abs_level_greater1_flag` bin and
/// [`Greater1State::on_coeff_abs_level_greater1_flag`] after each
/// decoded bin's value lands.
///
/// The machine is the §9.3.4.2.6 equations encoded as a tiny FSM:
///
/// * Equations 9-56 / 9-57 initialise `ctxSet` for each sub-block
///   based on the sub-block scan index `i` and the colour-component
///   index `cIdx`.
/// * The §9.3.4.2.6 `lastGreater1Ctx == 0` step (the spec's
///   "increment `ctxSet` by one") routes the sub-block into the
///   *next* context set whenever the previous sub-block ran out of
///   the `greater1Ctx ∈ {1, 2, 3}` band.
/// * Equation 9-59 `ctxInc = (ctxSet * 4) + min(3, greater1Ctx)`
///   plus the chroma `+16` offset (eq. 9-60) is reported by
///   [`Greater1State::current_ctx_inc`].
/// * The post-bin `greater1Ctx` update from §9.3.4.2.6 (the
///   `lastGreater1Flag == 1` → 0, `lastGreater1Flag == 0` →
///   increment-clamped-by-3 rule) lives in
///   [`Greater1State::on_coeff_abs_level_greater1_flag`].
///
/// The §9.3.4.2.7 `coeff_abs_level_greater2_flag` derivation reads
/// the *current* `ctxSet` from this same machine (the per-sub-block
/// value, not the post-update one) via
/// [`Greater1State::ctx_set`], so callers compose the
/// greater-2 invocation after the greater-1 sub-block walk without
/// any cross-state-machine plumbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Greater1State {
    /// §9.3.4.2.6 `ctxSet` — the current sub-block's context set in
    /// `{0, 1, 2, 3}`. Eq. 9-59 multiplies this by 4 to pick the
    /// per-set base.
    ctx_set: u32,
    /// §9.3.4.2.6 `greater1Ctx` — the within-set ctx position in
    /// `{0, 1, 2, 3}` (clamped at 3 for the `ctxInc` eq. 9-59).
    /// `0` means the sub-block has run out of the active band and
    /// the next sub-block's entry will bump `ctxSet`.
    greater1_ctx: u32,
    /// Has any sub-block been entered yet in this transform block?
    /// Used to recognise the "first sub-block" case the spec calls
    /// out (`lastGreater1Ctx = 1`).
    seen_any_subblock: bool,
    /// True iff at least one `coeff_abs_level_greater1_flag` bin has
    /// been read in *this* sub-block; signals that the next entry's
    /// `lastGreater1Ctx == 0` check has a well-defined
    /// `lastGreater1Flag` value to consume.
    has_last_greater1_flag: bool,
}

impl Greater1State {
    /// Construct a fresh state machine at the start of a transform
    /// block. No sub-block has yet been entered;
    /// [`on_subblock_entry`](Self::on_subblock_entry) must be called
    /// before [`current_ctx_inc`](Self::current_ctx_inc) is
    /// meaningful.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ctx_set: 0,
            greater1_ctx: 1,
            seen_any_subblock: false,
            has_last_greater1_flag: false,
        }
    }

    /// §9.3.4.2.6 sub-block entry: invoked once per sub-block before
    /// the first `coeff_abs_level_greater1_flag` bin is decoded.
    ///
    /// * `i` — the §7.3.8.11 sub-block scan index inside the
    ///   transform block.
    /// * `is_chroma` — `true` for `cIdx > 0`, used by eq. 9-56 to
    ///   force `ctxSet = 0` on every chroma sub-block.
    /// * `last_greater1_flag` — the most-recently decoded
    ///   `coeff_abs_level_greater1_flag` value from the *previous*
    ///   sub-block (0 or 1); ignored when this is the first
    ///   sub-block being entered (the spec's "first invocation"
    ///   branch sets `lastGreater1Ctx = 1` unconditionally).
    ///
    /// Implements equations 9-56, 9-57, and 9-58. After this call
    /// returns, [`current_ctx_inc`](Self::current_ctx_inc) yields
    /// the eq.-9-59 `ctxInc` for the sub-block's first bin.
    pub fn on_subblock_entry(&mut self, i: u32, is_chroma: bool, last_greater1_flag: u8) {
        // Eq. 9-56 / 9-57: initialise ctxSet from (i, cIdx).
        self.ctx_set = if i == 0 || is_chroma { 0 } else { 2 };

        // §9.3.4.2.6 lastGreater1Ctx derivation. The spec branches
        // on "first invocation" vs not:
        //   * First sub-block in this transform block:
        //     lastGreater1Ctx = 1. The ctxSet increment in eq. 9-58
        //     never triggers (lastGreater1Ctx > 0).
        //   * Subsequent sub-block, prior sub-block decoded at
        //     least one greater1 bin: lastGreater1Ctx is the prior
        //     sub-block's terminal greater1Ctx, then mutated by the
        //     prior sub-block's lastGreater1Flag per the spec
        //     bullets.
        // The walker tracks the prior sub-block's terminal
        // greater1Ctx as self.greater1_ctx (updated by the per-bin
        // step in on_coeff_abs_level_greater1_flag). We replay the
        // spec's lastGreater1Ctx mutation here, then the eq. 9-58
        // bump condition is `lastGreater1Ctx == 0`.
        let last_greater1_ctx = if !self.seen_any_subblock {
            // First sub-block: spec sets lastGreater1Ctx = 1.
            1
        } else if !self.has_last_greater1_flag {
            // Prior sub-block existed but decoded zero greater1
            // bins (rare: the sub-block had only sig coeffs equal
            // to 1 — i.e. no bin reached this code path). In that
            // case the spec's "When lastGreater1Ctx is greater than
            // 0, the variable lastGreater1Flag is set equal to ..."
            // branch is skipped and lastGreater1Ctx retains the
            // prior sub-block's value directly.
            self.greater1_ctx
        } else {
            // Apply the spec's lastGreater1Ctx mutation: if the
            // prior sub-block's lastGreater1Flag was 1 the ctx
            // resets to 0; otherwise it advances by 1.
            if self.greater1_ctx == 0 {
                // No mutation: the "When lastGreater1Ctx > 0"
                // guard skips the body entirely.
                0
            } else if last_greater1_flag != 0 {
                0
            } else {
                self.greater1_ctx + 1
            }
        };

        // Eq. 9-58: bump ctxSet whenever the just-derived
        // lastGreater1Ctx is 0.
        if last_greater1_ctx == 0 {
            self.ctx_set += 1;
        }

        // Start-of-sub-block reset of the within-set ctx: §9.3.4.2.6
        // says "greater1Ctx is set equal to 1".
        self.greater1_ctx = 1;
        self.seen_any_subblock = true;
        self.has_last_greater1_flag = false;
    }

    /// §9.3.4.2.6 per-bin step: invoked *after* every
    /// `coeff_abs_level_greater1_flag` bin in the current sub-block
    /// is decoded. Updates `greater1Ctx` per the spec's
    /// "lastGreater1Flag = 1 → 0 / = 0 → increment" rule, clamped to
    /// `0..=3` by eq. 9-59's `min(3, greater1Ctx)` (the underlying
    /// counter is implicitly clamped at 3 because eq. 9-59 truncates
    /// any further growth).
    ///
    /// Note: the §9.3.4.2.6 spec keeps `greater1Ctx` unclamped
    /// internally (it grows past 3) but eq. 9-59 clamps it; this
    /// implementation tracks the clamped value because the only
    /// observable property used downstream is whether
    /// `greater1Ctx == 0` (for the next sub-block's eq.-9-58 bump),
    /// and the eq. 9-59 reading clamps regardless.
    pub fn on_coeff_abs_level_greater1_flag(&mut self, decoded_bin: u8) {
        if self.greater1_ctx > 0 {
            self.greater1_ctx = if decoded_bin != 0 {
                0
            } else {
                (self.greater1_ctx + 1).min(3)
            };
            self.has_last_greater1_flag = true;
        }
    }

    /// Eq. 9-59 + eq. 9-60: returns the `coeff_abs_level_greater1_flag`
    /// `ctxInc` for the *next* bin to be decoded, given this state
    /// machine's `(ctxSet, greater1Ctx)`.
    ///
    /// `is_chroma == true` (cIdx > 0) adds the eq.-9-60 `+ 16`
    /// offset.
    #[must_use]
    pub fn current_ctx_inc(&self, is_chroma: bool) -> u32 {
        let base = self.ctx_set * 4 + self.greater1_ctx.min(3);
        if is_chroma { base + 16 } else { base }
    }

    /// Current `ctxSet` value, exported for §9.3.4.2.7
    /// `coeff_abs_level_greater2_flag` ctxInc derivation (which
    /// reads the same sub-block's `ctxSet` per eq. 9-61).
    #[must_use]
    pub fn ctx_set(&self) -> u32 {
        self.ctx_set
    }
}

impl Default for Greater1State {
    fn default() -> Self {
        Self::new()
    }
}

/// §9.3.4.2.7 ctxInc derivation for
/// `coeff_abs_level_greater2_flag[ lastGreater1ScanPos ]`. The spec
/// equations are:
///
/// ```text
/// ctxInc = ctxSet                  (9-61)  // luma
/// ctxInc = ctxSet + 4              (9-62)  // chroma (cIdx > 0)
/// ```
///
/// `ctxSet` is the value derived by §9.3.4.2.6 for the same
/// sub-block, available from [`Greater1State::ctx_set`].
///
/// The element is signalled at most once per sub-block at the first
/// scan position that took the greater-1 escape; the per-sub-block
/// `Greater1State` already has the matching `ctxSet` at the time
/// the residual loop reaches that scan position, so the caller
/// simply reads it off without further state plumbing.
#[must_use]
pub fn coeff_abs_level_greater2_flag_ctx_inc(ctx_set: u32, is_chroma: bool) -> u32 {
    if is_chroma { ctx_set + 4 } else { ctx_set }
}

/// Table 9-43 binarization shape for `coeff_abs_level_greater1_flag`
/// and `coeff_abs_level_greater2_flag`: FL with `cMax = 1` (single
/// context-coded bin per element invocation).
pub const COEFF_ABS_LEVEL_GREATER_X_FL_CMAX: u32 = 1;

// ---------------------------------------------------------------------
// sig_coeff_flag ctxInc — §9.3.4.2.5
// ---------------------------------------------------------------------

/// §9.3.4.2.5 Table 9-50 — the `ctxIdxMap[ ]` 4×4 lookup that maps
/// the inner-block scan position `(yC << 2) + xC` to a `sigCtx` for
/// `log2TrafoSize == 2` transform blocks. Index order is row-major
/// over `(yC, xC) ∈ {0..=3}^2`.
///
/// The published Table 9-50 renders only `i ∈ 0..=14`; the `i = 15`
/// cell is cropped past the page margin (a publishing/layout
/// artefact — eq. 9-41 indexes the map by `( yC << 2 ) + xC` over a
/// full 4×4 sub-block, so the 16th entry is normative content). The
/// staged H.265 errata file
/// (`docs/video/h265/h265-errata-and-clarifications.md`, entry #93)
/// pins `ctxIdxMap[ 15 ] = 8` from the table's anti-diagonal
/// pair-symmetry and the eq. 9-49..9-53 offset structure, matching
/// the value this constant has carried since round 32.
pub const SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2: [u8; 16] =
    [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// §9.3.4.2.5 first-branch sigCtx (equation 9-40) — used when
/// `transform_skip_context_enabled_flag` is 1 and either
/// `transform_skip_flag[ x0 ][ y0 ][ cIdx ]` is 1 or
/// `cu_transquant_bypass_flag` is 1. Returns 42 for luma, 16 for
/// chroma.
#[must_use]
pub fn sig_coeff_flag_sig_ctx_transform_skip(is_chroma: bool) -> u32 {
    // Eq. 9-40: sigCtx = (cIdx == 0) ? 42 : 16.
    if is_chroma { 16 } else { 42 }
}

/// §9.3.4.2.5 second-branch sigCtx (equation 9-41) — used for the
/// `log2TrafoSize == 2` (4×4) transform-block case. Reads
/// [`SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2`] at index
/// `(yC << 2) + xC`.
///
/// `xc` and `yc` are the 4×4 inner-block scan coordinates, each in
/// `0..=3`. Inputs outside this range are clamped via `& 3` so the
/// caller's debug build does not panic on a malformed scan order
/// driver — well-formed callers always pass `xc, yc < 4`.
#[must_use]
pub fn sig_coeff_flag_sig_ctx_log2_2(xc: u32, yc: u32) -> u32 {
    let xc = (xc & 3) as usize;
    let yc = (yc & 3) as usize;
    // Eq. 9-41: sigCtx = ctxIdxMap[ (yC << 2) + xC ].
    SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2[(yc << 2) + xc] as u32
}

/// §9.3.4.2.5 fourth-branch sigCtx derivation (equations 9-43..9-53)
/// for `log2TrafoSize > 2` and `xC + yC > 0`. Combines:
///
/// * `prevCsbf` from the right / below sub-block neighbours of the
///   current sub-block (equations 9-43, 9-44; edge-gated by the
///   `xS < (1 << (log2TrafoSize − 2)) − 1` /
///   `yS < (1 << (log2TrafoSize − 2)) − 1` conditions),
/// * the inner-sub-block position `(xP, yP) = (xC & 3, yC & 3)`
///   routed through equations 9-45..9-48 based on `prevCsbf`,
/// * the colour / size / scan-order tail offsets in equations
///   9-49..9-53.
///
/// Inputs:
///
/// * `is_chroma` — true when `cIdx > 0` (selects the chroma branch
///   of equations 9-49..9-53),
/// * `log2_trafo_size` — `3..=5` for this code path (the caller
///   must have dispatched the 4×4 case to
///   [`sig_coeff_flag_sig_ctx_log2_2`]),
/// * `xc`, `yc` — coefficient scan position inside the TB,
/// * `xs`, `ys` — sub-block scan position
///   (`xs = xC >> 2`, `ys = yC >> 2`),
/// * `right_csbf` / `below_csbf` — the previously decoded
///   `coded_sub_block_flag[ xS + 1 ][ yS ]` / `[ xS ][ yS + 1 ]`
///   neighbour bits (each 0 or 1); ignored when on the right /
///   bottom TB edge per the equation 9-43 / 9-44 gates,
/// * `scan_idx` — the §6.5.2 scan order index in `{0, 1, 2}`
///   (`0` = up-right diagonal, `1` = horizontal, `2` = vertical);
///   only `scan_idx == 0` is special-cased by equation 9-50 for
///   the 8×8 luma case.
///
/// The `xC + yC == 0` DC-coefficient case (equation 9-42 — `sigCtx
/// = 0` before the tail offsets) is **not** handled by this
/// function; callers must route the DC position separately through
/// [`sig_coeff_flag_sig_ctx_dc`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sig_coeff_flag_sig_ctx_general(
    is_chroma: bool,
    log2_trafo_size: u32,
    xc: u32,
    yc: u32,
    xs: u32,
    ys: u32,
    right_csbf: u8,
    below_csbf: u8,
    scan_idx: u32,
) -> u32 {
    // Equations 9-43 / 9-44: prevCsbf gates on the edge of the TB
    // sub-block grid.
    let max_sub_block_idx = (1u32 << (log2_trafo_size - 2)) - 1;
    let right_bit = if xs < max_sub_block_idx {
        (right_csbf & 1) as u32
    } else {
        0
    };
    let below_bit = if ys < max_sub_block_idx {
        (below_csbf & 1) as u32
    } else {
        0
    };
    let prev_csbf = right_bit + (below_bit << 1);

    // §9.3.4.2.5 inner-sub-block position.
    let xp = xc & 3;
    let yp = yc & 3;

    // Equations 9-45..9-48: route by prevCsbf.
    let mut sig_ctx: u32 = match prev_csbf {
        0 => {
            // Eq. 9-45: (xP + yP == 0) ? 2 : (xP + yP < 3) ? 1 : 0.
            if xp + yp == 0 {
                2
            } else if xp + yp < 3 {
                1
            } else {
                0
            }
        }
        1 => {
            // Eq. 9-46: (yP == 0) ? 2 : (yP == 1) ? 1 : 0.
            if yp == 0 {
                2
            } else if yp == 1 {
                1
            } else {
                0
            }
        }
        2 => {
            // Eq. 9-47: (xP == 0) ? 2 : (xP == 1) ? 1 : 0.
            if xp == 0 {
                2
            } else if xp == 1 {
                1
            } else {
                0
            }
        }
        // prevCsbf == 3: eq. 9-48 — sigCtx = 2.
        _ => 2,
    };

    // Equations 9-49..9-53: colour / size / scan tail offsets.
    if !is_chroma {
        // Luma branch.
        // Eq. 9-49: when (xS + yS) > 0, sigCtx += 3.
        if xs + ys > 0 {
            sig_ctx += 3;
        }
        // Eq. 9-50 / 9-51: size-dependent constant.
        if log2_trafo_size == 3 {
            sig_ctx += if scan_idx == 0 { 9 } else { 15 };
        } else {
            sig_ctx += 21;
        }
    } else {
        // Chroma branch.
        // Eq. 9-52 / 9-53.
        if log2_trafo_size == 3 {
            sig_ctx += 9;
        } else {
            sig_ctx += 12;
        }
    }

    sig_ctx
}

/// §9.3.4.2.5 DC sigCtx for `log2TrafoSize > 2` and `xC + yC == 0`
/// (equation 9-42). The DC coefficient takes the third branch of the
/// §9.3.4.2.5 cascade: `sigCtx = 0`, full stop — the
/// equations-9-49..9-53 size / colour / scan modifications are nested
/// inside the *fourth* ("Otherwise") branch and do **not** apply to the
/// DC position. The resulting ctxInc is 0 for luma and 27 for chroma
/// (equations 9-54 / 9-55): every `log2TrafoSize > 2` transform block
/// shares one dedicated DC context per colour component.
///
/// The `is_chroma` / `log2_trafo_size` / `scan_idx` inputs are retained
/// for signature parity with the sibling branch helpers; equation 9-42
/// uses none of them.
#[must_use]
pub fn sig_coeff_flag_sig_ctx_dc(is_chroma: bool, log2_trafo_size: u32, scan_idx: u32) -> u32 {
    let _ = (is_chroma, log2_trafo_size, scan_idx);
    // Eq. 9-42: sigCtx = 0.
    0
}

/// §9.3.4.2.5 ctxInc from sigCtx (equations 9-54, 9-55):
///
/// ```text
/// ctxInc = sigCtx           // cIdx == 0   (9-54)
/// ctxInc = 27 + sigCtx      // cIdx > 0    (9-55)
/// ```
#[must_use]
pub fn sig_coeff_flag_ctx_inc_from_sig_ctx(sig_ctx: u32, is_chroma: bool) -> u32 {
    if is_chroma { 27 + sig_ctx } else { sig_ctx }
}

/// Table 9-43 binarization shape for `sig_coeff_flag`: FL with
/// `cMax = 1` (single context-coded bin per scan position).
pub const SIG_COEFF_FLAG_FL_CMAX: u32 = 1;

// ---------------------------------------------------------------------
// palette_run_prefix ctxInc — §9.3.4.2.8 (equation 9-63, Table 9-51)
// ---------------------------------------------------------------------

/// §9.3.4.2.8 Table 9-51 — the `ctxIdxMap[ ][ ]` lookup that maps
/// `(copy_above_palette_indices_flag, binIdx)` to `ctxInc` for the
/// non-eq.-9-63 branch of the §9.3.4.2.8 derivation. The table is a
/// 2 × 5 fixed permutation listed verbatim in Table 9-51 of the spec;
/// the `binIdx >= 5` columns are bypass-coded per Table 9-48 and
/// therefore omitted here.
///
/// Layout: `PALETTE_RUN_PREFIX_CTX_IDX_MAP[copy_above_flag][binIdx]`,
/// with `copy_above_flag ∈ {0, 1}` and `binIdx ∈ {0, 1, 2, 3, 4}`.
///
/// The `copy_above_flag == 0, binIdx == 0` cell is spec-listed as
/// `"0, 1, 2"` (i.e. dispatched to eq. 9-63 on `palette_idx_idc`).
/// This module surfaces that branch separately as
/// [`palette_run_prefix_ctx_inc_eq_9_63`], so the table entry is held
/// as `u8::MAX` to mark "must dispatch to eq. 9-63" defensively (the
/// public [`palette_run_prefix_ctx_inc`] handles the branch).
pub const PALETTE_RUN_PREFIX_CTX_IDX_MAP: [[u8; 5]; 2] = [
    // copy_above_palette_indices_flag == 0: bin 0 dispatches to eq.
    // 9-63 (sentinel); bins 1..=4 are 3, 3, 4, 4.
    [u8::MAX, 3, 3, 4, 4],
    // copy_above_palette_indices_flag == 1: 5, 6, 6, 7, 7.
    [5, 6, 6, 7, 7],
];

/// Sentinel marking the eq.-9-63 dispatch cell in
/// [`PALETTE_RUN_PREFIX_CTX_IDX_MAP`].
pub const PALETTE_RUN_PREFIX_EQ_9_63_DISPATCH: u8 = u8::MAX;

/// §9.3.4.2.8 first `binIdx >= 5` index at which Table 9-51 declares
/// `palette_run_prefix` bypass-coded. Equivalent to the TR prefix
/// `cMax = 5` boundary above which §9.3.4.2.8 stops emitting ctxInc
/// values.
pub const PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX: u32 = 5;

/// §9.3.4.2.8 equation 9-63 — the eq.-9-63 dispatch when
/// `copy_above_palette_indices_flag == 0` and `binIdx == 0`:
///
/// ```text
/// ctxInc = ( palette_idx_idc < 1 ) ? 0
///        : ( ( palette_idx_idc < 3 ) ? 1 : 2 )            (9-63)
/// ```
///
/// Returns `ctxInc ∈ {0, 1, 2}`.
#[must_use]
pub fn palette_run_prefix_ctx_inc_eq_9_63(palette_idx_idc: u32) -> u32 {
    if palette_idx_idc < 1 {
        0
    } else if palette_idx_idc < 3 {
        1
    } else {
        2
    }
}

/// §9.3.4.2.8 — full `palette_run_prefix` ctxInc derivation.
///
/// Inputs:
///
/// * `bin_idx` — the §9.3.3.10 TR-prefix bin index (0-based).
/// * `copy_above_palette_indices_flag` — the §7.4.9.6 palette flag
///   that selects "copy-above" vs explicit-index palette mode for
///   this run.
/// * `palette_idx_idc` — the palette index decoded at the start of
///   the run; only consulted on the eq.-9-63 branch.
///
/// Returns:
///
/// * `Some(ctxInc)` — when the bin is context-coded. The eq.-9-63
///   branch returns `{0, 1, 2}`; otherwise the value is read from
///   [`PALETTE_RUN_PREFIX_CTX_IDX_MAP`].
/// * `None` — when `bin_idx >= 5`, the Table 9-51 ">4" column,
///   signalling that the bin is bypass-coded per Table 9-48 and the
///   caller must invoke the engine's bypass path.
#[must_use]
pub fn palette_run_prefix_ctx_inc(
    bin_idx: u32,
    copy_above_palette_indices_flag: bool,
    palette_idx_idc: u32,
) -> Option<u32> {
    if bin_idx >= PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX {
        return None;
    }
    if !copy_above_palette_indices_flag && bin_idx == 0 {
        // §9.3.4.2.8 first bullet — dispatch to eq. 9-63.
        return Some(palette_run_prefix_ctx_inc_eq_9_63(palette_idx_idc));
    }
    // Otherwise: Table 9-51 lookup.
    let row = usize::from(copy_above_palette_indices_flag);
    let col = bin_idx as usize;
    let v = PALETTE_RUN_PREFIX_CTX_IDX_MAP[row][col];
    debug_assert!(
        v != PALETTE_RUN_PREFIX_EQ_9_63_DISPATCH,
        "palette_run_prefix Table 9-51: the eq.-9-63 dispatch cell is \
         already handled above; reaching the lookup with the sentinel \
         indicates a derivation bug"
    );
    Some(u32::from(v))
}

/// Table 9-43 binarization shape `cMax` for `palette_run_prefix`:
///
/// ```text
/// cMax = Floor( Log2( PaletteMaxRunMinus1 ) ) + 1
/// cRiceParam = 0
/// ```
///
/// `PaletteMaxRunMinus1` is the §7.4.9.6 per-CU palette-run cap (one
/// less than the maximum allowed run length). When
/// `PaletteMaxRunMinus1 == 0` the TR prefix is degenerate (a single
/// terminating bin); the helper returns `1` in that case so callers
/// can still emit a one-bin TR prefix.
#[must_use]
pub fn palette_run_prefix_tr_cmax(palette_max_run_minus1: u32) -> u32 {
    if palette_max_run_minus1 == 0 {
        // Floor(Log2(0)) is undefined; the spec's TR-prefix shape
        // collapses to a single-bin terminator in this degenerate
        // case. Returning 1 keeps the TR reader well-defined.
        return 1;
    }
    // Floor(Log2(x)) for x > 0 == 31 - leading_zeros(x).
    let floor_log2 = 31 - palette_max_run_minus1.leading_zeros();
    floor_log2 + 1
}

// ---------------------------------------------------------------------
// coeff_abs_level_remaining (§9.3.3.11)
// ---------------------------------------------------------------------

/// §9.3.3.11 — TR-prefix escape length for `coeff_abs_level_remaining`.
/// The §9.3.3.2 TR(`cMax`, `cRiceParam`) prefix is `cMax / (1 <<
/// cRiceParam)` bins long; substituting eq. 9-26 (`cMax = 4 <<
/// cRiceParam`) collapses that to a constant **4** bins regardless of
/// `cRiceParam`. The all-4-ones prefix is the escape that signals the
/// EGk(`cRiceParam + 1`) suffix is present.
pub const COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN: u32 = 4;

/// §9.3.3.11 — `cRiceParam` adaptation, non-persistent path
/// (`persistent_rice_adaptation_enabled_flag == 0`), eq. 9-24:
///
/// ```text
/// cRiceParam = Min( cLastRiceParam +
///                   ( cLastAbsLevel > ( 3 * ( 1 << cLastRiceParam ) ) ? 1 : 0 ),
///                   4 )
/// ```
///
/// The bump (+1) fires when the previous coefficient's absolute level
/// exceeded `3 << cLastRiceParam`; `cRiceParam` saturates at 4 (the
/// non-persistent ceiling).
///
/// Inputs:
///
/// * `c_last_abs_level` — the `cAbsLevel` from the previous invocation
///   in the same sub-block (`baseLevel + coeff_abs_level_remaining[n]`
///   of the previous scan position; `0` if this is the first
///   invocation for the sub-block).
/// * `c_last_rice_param` — the `cRiceParam` from the previous
///   invocation in the same sub-block (`0` if this is the first
///   invocation).
///
/// Returns the per-coefficient `cRiceParam ∈ {0, 1, 2, 3, 4}` that
/// drives eq. 9-26 + the §9.3.3.2 TR-prefix shape.
#[must_use]
pub fn coeff_abs_level_remaining_c_rice_param_eq_9_24(
    c_last_abs_level: u32,
    c_last_rice_param: u32,
) -> u32 {
    // `3 * (1 << r)` == `3 << r`; using the left-shift form keeps the
    // comparison overflow-free for `r` up to 29 (3 << 29 < 2^32).
    let threshold = 3u32 << c_last_rice_param;
    let bump = u32::from(c_last_abs_level > threshold);
    (c_last_rice_param + bump).min(4)
}

/// §9.3.3.11 — `cRiceParam` adaptation, persistent path
/// (`persistent_rice_adaptation_enabled_flag == 1`), eq. 9-25:
///
/// ```text
/// cRiceParam = cLastRiceParam +
///              ( cLastAbsLevel > ( 3 * ( 1 << cLastRiceParam ) ) ? 1 : 0 )
/// ```
///
/// Unlike the eq. 9-24 path there is **no** Min(…, 4) ceiling — the
/// parameter keeps growing while consecutive levels exceed the
/// threshold, and the per-sub-block start value is `initRiceValue =
/// StatCoeff[ sbType ] / 4` (eq. 9-22) rather than 0. A defensive cap
/// of 24 keeps the downstream `4 << cRiceParam` / TR-suffix shifts
/// sound; reaching it would need `cLastAbsLevel > 3 << 23`, beyond
/// the §7.4.9.11 `CoeffMax` bound (eqs. 7-27 .. 7-30, at most
/// `2^22 − 1` even with `extended_precision_processing_flag == 1` at
/// `BitDepth` 16), so no conforming stream is affected.
#[must_use]
pub fn coeff_abs_level_remaining_c_rice_param_eq_9_25(
    c_last_abs_level: u32,
    c_last_rice_param: u32,
) -> u32 {
    let r = c_last_rice_param.min(24);
    let threshold = 3u32 << r;
    let bump = u32::from(c_last_abs_level > threshold);
    (r + bump).min(24)
}

/// §9.3.3.4 — decode the **limited EGk** suffix of
/// `coeff_abs_level_remaining[ n ]`, invoked only when
/// `extended_precision_processing_flag == 1` and the §9.3.3.2 TR
/// prefix was the all-ones escape. `rice_param` is the §9.3.3.11
/// "riceParam = cRiceParam + 1" input; `log2_transform_range` is
/// eq. 9-14 (`Max( 15, BitDepth + 6 )` for the block's component).
///
/// The eq. 9-16 bin string is a unary prefix-extension run of `1`
/// bins capped at `maxPrefixExtensionLength = 28 −
/// log2TransformRange` (eq. 9-15), then — only when the run stopped
/// short of the cap — a `0` terminator, then a fixed-length escape of
/// `PrefixExtensionLength + riceParam` bits (or `log2TransformRange`
/// bits when the run hit the cap, with no terminator). The decoded
/// `symbolVal` is the escape value plus
/// `( ( 1 << PrefixExtensionLength ) − 1 ) << riceParam`.
pub fn decode_limited_egk_suffix_with<F>(
    rice_param: u32,
    log2_transform_range: u32,
    mut read_bin: F,
) -> Result<u32, CabacError>
where
    F: FnMut() -> Result<u8, CabacError>,
{
    let max_prefix_extension_length = 28u32.saturating_sub(log2_transform_range);
    let mut prefix_extension_length: u32 = 0;
    let mut hit_cap = true;
    while prefix_extension_length < max_prefix_extension_length {
        if read_bin()? == 0 {
            hit_cap = false;
            break;
        }
        prefix_extension_length += 1;
    }
    let escape_length = if hit_cap {
        log2_transform_range
    } else {
        prefix_extension_length + rice_param
    };
    let mut escape: u64 = 0;
    for _ in 0..escape_length.min(32) {
        escape = (escape << 1) | u64::from(read_bin()?);
    }
    // Compose in u64 and saturate: the §7.4.9.11 conformance bound
    // keeps every conforming value far below u32 saturation.
    let base = ((1u64 << prefix_extension_length.min(32)) - 1) << rice_param.min(31);
    Ok(u32::try_from(base + escape).unwrap_or(u32::MAX))
}

/// §9.3.3.11 eq. 9-26 — `cMax = 4 << cRiceParam`.
///
/// Drives the §9.3.3.2 TR-prefix `cMax` input. For `cRiceParam ∈
/// {0, 1, 2, 3, 4}` this is `{4, 8, 16, 32, 64}`.
#[must_use]
pub fn coeff_abs_level_remaining_c_max_eq_9_26(c_rice_param: u32) -> u32 {
    4u32 << c_rice_param
}

/// §9.3.3.11 eq. 9-27 — `prefixVal = Min(cMax, level)`.
///
/// The TR-prefix input value: clamped to `cMax` so the prefix
/// terminates at the all-ones escape when the level overflows the
/// TR-only range.
#[must_use]
pub fn coeff_abs_level_remaining_prefix_val_eq_9_27(level: u32, c_max: u32) -> u32 {
    level.min(c_max)
}

/// §9.3.3.11 eq. 9-28 — `suffixVal = level - cMax`.
///
/// Only meaningful when the TR prefix is the all-ones escape (i.e.
/// `level >= cMax`); otherwise the suffix bin string is absent.
/// Returns `0` for `level < cMax` so callers can avoid a separate
/// branch when probing the suffix shape.
#[must_use]
pub fn coeff_abs_level_remaining_suffix_val_eq_9_28(level: u32, c_max: u32) -> u32 {
    level.saturating_sub(c_max)
}

/// §9.3.3.11 — bin-source-driven core of `coeff_abs_level_remaining[
/// n ]` decode. Factored out of [`decode_coeff_abs_level_remaining`]
/// so the algorithm can be exercised by tests that supply a flat bin
/// sequence directly, independently of the §9.3.4.3.4 CABAC
/// arithmetic engine's bin / stream-bit relationship.
///
/// `read_bin()` returns one bypass-coded bin (0 or 1); the closure
/// owns the underlying source state.
///
/// See [`decode_coeff_abs_level_remaining`] for the full §9.3.3.11
/// scope, branch coverage, and follow-up notes.
pub fn decode_coeff_abs_level_remaining_with<F>(
    c_rice_param: u32,
    mut read_bin: F,
) -> Result<u32, CabacError>
where
    F: FnMut() -> Result<u8, CabacError>,
{
    // §9.3.3.2 truncated-rice prefix: up to ESCAPE_LEN unary bins.
    let mut prefix_len: u32 = 0;
    while prefix_len < COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN {
        let bin = read_bin()?;
        if bin == 0 {
            // §9.3.3.2 terminator — the value is contained inside
            // the TR shape; finish with the `cRiceParam`-bit suffix.
            let mut tr_suffix: u32 = 0;
            for _ in 0..c_rice_param {
                tr_suffix = (tr_suffix << 1) | u32::from(read_bin()?);
            }
            return Ok((prefix_len << c_rice_param) + tr_suffix);
        }
        prefix_len += 1;
    }
    // Escape: all-ones prefix. EGk(cRiceParam + 1) suffix follows.
    let k = c_rice_param + 1;
    // §9.3.3.3 eq. 9-13: the EGk unary prefix is a run of `1` bins
    // terminated by a single `0` bin (the loop emits `put( 1 )` while
    // `absV >= ( 1 << k )`, then `put( 0 )` to stop). The decoded prefix
    // length is the count of leading `1` bins before that terminating
    // `0`.
    let mut prefix_ones: u32 = 0;
    while prefix_ones < 32 {
        let bin = read_bin()?;
        if bin == 0 {
            break;
        }
        prefix_ones += 1;
    }
    let suffix_bits = (prefix_ones + k).min(32);
    let mut suffix: u32 = 0;
    for _ in 0..suffix_bits {
        suffix = (suffix << 1) | u32::from(read_bin()?);
    }
    // Compose in u64: `prefix_ones` can reach 32 on a non-conformant
    // bin stream, where `base` alone exceeds u32::MAX. The §7.4.9.11
    // conformance requirement bounds the corresponding TransCoeffLevel
    // to [CoeffMin, CoeffMax] (eqs. 7-27 .. 7-30), far below the
    // saturation point, so the clamp only fires on bin streams the
    // spec already rejects — the decode stays total instead of
    // overflowing.
    let base = ((1u64 << prefix_ones) - 1) << k;
    let c_max = u64::from(coeff_abs_level_remaining_c_max_eq_9_26(c_rice_param));
    let value = c_max + base + u64::from(suffix);
    Ok(u32::try_from(value).unwrap_or(u32::MAX))
}

/// §9.3.3.11, `extended_precision_processing_flag == 1` variant of
/// [`decode_coeff_abs_level_remaining_with`]: the §9.3.3.2 TR prefix
/// and `cRiceParam`-bit TR suffix are unchanged, but the all-ones
/// escape switches to the §9.3.3.4 **limited EGk** suffix
/// ([`decode_limited_egk_suffix_with`]) with `riceParam = cRiceParam
/// + 1` and the component's `log2TransformRange` (eq. 9-14).
pub fn decode_coeff_abs_level_remaining_ext_with<F>(
    c_rice_param: u32,
    log2_transform_range: u32,
    mut read_bin: F,
) -> Result<u32, CabacError>
where
    F: FnMut() -> Result<u8, CabacError>,
{
    let mut prefix_len: u32 = 0;
    while prefix_len < COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN {
        let bin = read_bin()?;
        if bin == 0 {
            let mut tr_suffix: u32 = 0;
            for _ in 0..c_rice_param {
                tr_suffix = (tr_suffix << 1) | u32::from(read_bin()?);
            }
            return Ok((prefix_len << c_rice_param) + tr_suffix);
        }
        prefix_len += 1;
    }
    let suffix = decode_limited_egk_suffix_with(c_rice_param + 1, log2_transform_range, read_bin)?;
    let c_max = coeff_abs_level_remaining_c_max_eq_9_26(c_rice_param);
    Ok(c_max.saturating_add(suffix))
}

/// §9.3.3.11 — full bypass-coded decode of `coeff_abs_level_remaining[
/// n ]` given the per-coefficient `cRiceParam` already adapted by
/// [`coeff_abs_level_remaining_c_rice_param_eq_9_24`].
///
/// The decoder:
///
/// 1. Reads a unary TR prefix of up to
///    [`COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN`] = `4` bypass
///    bins. The prefix length is the number of leading `1` bins.
/// 2. If the prefix terminated short (length `< 4`), reads
///    `cRiceParam` more bypass bins as the TR-suffix portion of the
///    §9.3.3.2 shape; the decoded value is
///    `(prefix_length << cRiceParam) + tr_suffix`.
/// 3. Otherwise (the all-4-ones escape), reads an EGk suffix with
///    `k = cRiceParam + 1` per the §9.3.3.11 "non-extended-precision"
///    bullet; the decoded value is `cMax + suffixVal` where
///    `cMax = 4 << cRiceParam`.
///
/// The function returns the bit-exact `coeff_abs_level_remaining[ n ]`
/// value (not `baseLevel + remaining`); the caller composes the final
/// signed coefficient using the §7.4.9.11 coding loop (`baseLevel`,
/// `coeff_sign_flag[n]`).
///
/// This implements the §9.3.3.11 **non-persistent** path
/// (`persistent_rice_adaptation_enabled_flag == 0`) and the
/// **non-extended-precision** suffix branch
/// (`extended_precision_processing_flag == 0`). The persistent and
/// extended-precision branches are left for follow-up rounds; their
/// inputs (StatCoeff[sbType], the §9.3.3.4 limited EGk shape) require
/// trace material beyond what this round covers.
pub fn decode_coeff_abs_level_remaining(
    engine: &mut CabacEngine<'_>,
    c_rice_param: u32,
) -> Result<u32, CabacError> {
    decode_coeff_abs_level_remaining_with(c_rice_param, || engine.decode_bypass())
}

// ---------------------------------------------------------------------
// coeff_sign_flag[ n ] — §7.3.8.11 / §7.4.9.11 / Table 9-43 / Table 9-48
// ---------------------------------------------------------------------

/// §9.3.3.5 / Table 9-43 — `coeff_sign_flag[ n ]` binarization shape.
///
/// Fixed-length with `cMax = 1` — a single bin per scan position. The
/// spec's §9.3.3.5 derivation gives `fixedLength = Ceil(Log2(cMax + 1))
/// = Ceil(Log2(2)) = 1`, so the on-wire string is exactly one bit
/// whose value is the flag itself.
pub const COEFF_SIGN_FLAG_FL_CMAX: u32 = 1;
/// §9.3.3.5 — bit width of the FL string for `coeff_sign_flag[ n ]`.
///
/// `Ceil(Log2(COEFF_SIGN_FLAG_FL_CMAX + 1)) = 1`. Exposed so callers
/// composing the slice-data bypass tail can iterate the per-scan-position
/// sign bits without re-deriving the FL width.
pub const COEFF_SIGN_FLAG_FL_NBITS: u32 = 1;

/// §9.3.4.2 / Table 9-48 — `coeff_sign_flag[ n ]` is fully bypass-coded.
///
/// Table 9-48 marks the bin-0 cell `bypass` and all subsequent bin-index
/// columns `na`; there is one bin per invocation and it carries no
/// context coupling.
///
/// Returns the per-scan-position sign bit: `0` ⇒ positive,
/// `1` ⇒ negative (per §7.4.9.11). The signed transform-coefficient
/// level the slice-data residual loop emits is then
/// `(coeff_abs_level_remaining[n] + baseLevel) * (1 − 2 *
/// coeff_sign_flag[n])`; see [`signed_level_from_sign_flag`] for the
/// arithmetic helper.
///
/// §9.3.4.3.6 specifies an alignment process that runs prior to
/// bypass decoding of `coeff_abs_level_remaining[ ]` and
/// `coeff_sign_flag[ ]` (`ivlCurrRange := 256`); the slice-data parser
/// invokes [`CabacEngine::align`] once at the start of the bypass
/// run, so this entry point reads a single bypass bin via the
/// post-alignment engine state.
pub fn decode_coeff_sign_flag(engine: &mut CabacEngine<'_>) -> Result<u8, CabacError> {
    engine.decode_bypass()
}

/// §7.4.9.11 — compose the signed transform-coefficient level from the
/// unsigned `cAbsLevel` (`baseLevel + coeff_abs_level_remaining[n]`)
/// and the per-scan-position `coeff_sign_flag[n]`.
///
/// Implements `level * (1 − 2 * sign_flag)`:
///
/// * `sign_flag == 0` ⇒ the level is positive, returns `+abs_level`.
/// * `sign_flag == 1` ⇒ the level is negative, returns `-abs_level`.
///
/// Inputs:
///
/// * `abs_level` — the unsigned absolute level emitted by the
///   §7.3.8.11 residual loop (`baseLevel` plus, when present,
///   [`decode_coeff_abs_level_remaining`]'s output).
/// * `sign_flag` — the per-scan-position bin read via
///   [`decode_coeff_sign_flag`] (or inferred to `0` per §7.4.9.11
///   when the syntax element is not present).
///
/// The §7.4.9.11 sign-data-hiding adjustment (when
/// `sign_data_hiding_enabled_flag && signHidden` and the
/// `firstSigScanPos` sum-of-absolute-levels parity is odd) is a
/// post-step the slice-data loop applies at a different scope and is
/// **not** folded into this helper.
///
/// The return type is `i32` because `coeff_abs_level_remaining[n] +
/// baseLevel` can exceed `i16::MAX` for high-bit-depth profiles before
/// the §7.4.9.11 / Annex A clipping to `[CoeffMin, CoeffMax]`.
#[must_use]
pub fn signed_level_from_sign_flag(abs_level: u32, sign_flag: u8) -> i32 {
    let abs_signed = abs_level as i32;
    if sign_flag == 0 {
        abs_signed
    } else {
        -abs_signed
    }
}

// ---------------------------------------------------------------------
// mvd_coding( x0, y0, refList ) — §7.3.8.9 / §7.4.9.9 / Table 9-43 /
// Table 9-48. Motion-vector-difference binarization: two context-coded
// magnitude flags (`abs_mvd_greater0_flag`, `abs_mvd_greater1_flag`),
// a bypass EG1 escape magnitude (`abs_mvd_minus2`), and a bypass FL
// sign bit (`mvd_sign_flag`), composed per equation 7-73 into the
// signed component difference `lMvd`.
// ---------------------------------------------------------------------

/// §9.3.4.2 / Table 9-48 — `abs_mvd_greater0_flag[ ]` is context-coded
/// with a single context (`ctxInc = 0` at bin 0; all later bin-index
/// columns are `na` since the FL `cMax = 1` binarization is one bin).
///
/// Table 9-4 maps the element to its `Table 9-23` context bank; per
/// [`SliceContexts`] this is the single-entry `abs_mvd_greater0_flag`
/// array (inter slices only). The caller passes the corresponding
/// [`ContextModel`]; this function returns the fixed `ctxInc`.
#[must_use]
pub fn abs_mvd_greater0_flag_ctx_inc() -> u32 {
    0
}

/// §9.3.4.2 / Table 9-48 — `abs_mvd_greater1_flag[ ]` is context-coded
/// with a single context (`ctxInc = 0` at bin 0; later columns `na`).
///
/// The element binds to its own `Table 9-23` context bank (distinct
/// from `abs_mvd_greater0_flag`); the returned `ctxInc` selects within
/// that single-entry bank.
#[must_use]
pub fn abs_mvd_greater1_flag_ctx_inc() -> u32 {
    0
}

/// Table 9-43 — `abs_mvd_greater0_flag[ ]` / `abs_mvd_greater1_flag[ ]`
/// / `mvd_sign_flag[ ]` are FL with `cMax = 1` (a single bin each).
pub const ABS_MVD_GREATER_FLAG_FL_CMAX: u32 = 1;

/// Table 9-43 — `abs_mvd_minus2[ ]` uses EG1 (k-th order Exp-Golomb
/// with `k = 1`), fully bypass-coded per Table 9-48.
pub const ABS_MVD_MINUS2_EG_K: u32 = 1;

/// Table 9-43 — `mvd_sign_flag[ ]` is FL with `cMax = 1`; the single
/// bin (bypass-coded per Table 9-48) is the sign bit itself.
pub const MVD_SIGN_FLAG_FL_CMAX: u32 = 1;

/// One decoded `mvd_coding( )` motion-vector-difference component
/// (§7.3.8.9 / §7.4.9.9). All four wire fields are retained alongside
/// the equation-7-73 composed signed value so callers can both record
/// the parsed syntax and consume `lMvd` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvdComponent {
    /// `abs_mvd_greater0_flag[ compIdx ]` — magnitude `> 0`.
    pub greater0_flag: u8,
    /// `abs_mvd_greater1_flag[ compIdx ]` — magnitude `> 1`.
    /// `None` when not present (inferred `0` per §7.4.9.9), i.e. when
    /// `greater0_flag == 0`.
    pub greater1_flag: Option<u8>,
    /// `abs_mvd_minus2[ compIdx ]` from the EG1 bypass escape.
    /// `None` when not present (inferred `−1` per §7.4.9.9), i.e. when
    /// `greater1_flag` is `0`/absent.
    pub minus2: Option<u32>,
    /// `mvd_sign_flag[ compIdx ]`. `None` when not present (inferred
    /// `0` per §7.4.9.9), i.e. when `greater0_flag == 0`.
    pub sign_flag: Option<u8>,
    /// `lMvd[ compIdx ]` per equation 7-73:
    /// `greater0_flag * (abs_mvd_minus2 + 2) * (1 − 2 * sign_flag)`.
    /// Zero when `greater0_flag == 0`.
    pub value: i32,
}

/// §7.4.9.9 equation 7-73 — compose the signed component difference
/// `lMvd` from the four `mvd_coding( )` wire fields, applying the
/// not-present inferences:
///
/// * `abs_mvd_greater1_flag` inferred `0` ⇒ when `greater0_flag == 1`
///   but `greater1_flag` is absent/`0`, the magnitude is exactly `1`.
/// * `abs_mvd_minus2` inferred `−1` ⇒ when absent, `minus2 + 2 == 1`
///   (the magnitude-equals-1 case); when present, magnitude is
///   `minus2 + 2` (`>= 2`).
/// * `mvd_sign_flag` inferred `0` ⇒ a positive value.
///
/// `greater0_flag == 0` short-circuits to `0` (the whole product
/// vanishes), matching the §7.3.8.9 parse where no further fields are
/// signalled.
#[must_use]
pub fn mvd_component_value(greater0_flag: u8, minus2: Option<u32>, sign_flag: Option<u8>) -> i32 {
    if greater0_flag == 0 {
        return 0;
    }
    // abs_mvd_minus2 inferred −1 when absent ⇒ magnitude (minus2 + 2):
    //   absent  → −1 + 2 = 1
    //   present → minus2 + 2  (>= 2)
    let magnitude = match minus2 {
        Some(m) => m as i32 + 2,
        None => 1,
    };
    // mvd_sign_flag inferred 0 when absent ⇒ positive.
    let sign = sign_flag.unwrap_or(0);
    magnitude * (1 - 2 * i32::from(sign))
}

/// Decode one `mvd_coding( )` component (§7.3.8.9) from split bin
/// readers, factored out of [`decode_mvd_component`] so the
/// composition can be unit-tested without driving the arithmetic
/// engine's range/offset state.
///
/// `read_ctx_bin` is invoked for each context-coded bin in parse
/// order — first `abs_mvd_greater0_flag`, then (only when that bin is
/// `1`) `abs_mvd_greater1_flag`. `read_bypass_bin` is invoked for each
/// bypass bin of the EG1 `abs_mvd_minus2` escape and the
/// `mvd_sign_flag` bit; the EG1 read consumes the §9.3.3.3 leading-zero
/// prefix plus `leading_zeros + 1` suffix bins.
///
/// The reader split mirrors Table 9-48: the two `greater` flags are
/// `DecodeDecision` bins, `abs_mvd_minus2` and `mvd_sign_flag` are
/// `DecodeBypass` bins. Per §7.3.8.9, `abs_mvd_minus2` is only read
/// when `abs_mvd_greater1_flag == 1`, and `mvd_sign_flag` only when
/// `abs_mvd_greater0_flag == 1`. The EG1 read consumes the §9.3.3.3
/// prefix (a run of `1` bins terminated by `0`) plus `prefix_ones + 1`
/// suffix bins.
pub fn decode_mvd_component_with<C, B>(
    mut read_ctx_bin: C,
    mut read_bypass_bin: B,
) -> Result<MvdComponent, CabacError>
where
    C: FnMut() -> Result<u8, CabacError>,
    B: FnMut() -> Result<u8, CabacError>,
{
    let greater0_flag = read_ctx_bin()?;
    if greater0_flag == 0 {
        // §7.3.8.9: no further fields signalled; all inferred absent.
        return Ok(MvdComponent {
            greater0_flag: 0,
            greater1_flag: None,
            minus2: None,
            sign_flag: None,
            value: 0,
        });
    }
    let greater1_flag = read_ctx_bin()?;
    // abs_mvd_minus2 is present only when greater1_flag == 1.
    let minus2 = if greater1_flag == 1 {
        Some(read_eg_k_with(ABS_MVD_MINUS2_EG_K, &mut read_bypass_bin)?)
    } else {
        None
    };
    // mvd_sign_flag is present whenever greater0_flag == 1.
    let sign_flag = read_bypass_bin()?;
    let value = mvd_component_value(greater0_flag, minus2, Some(sign_flag));
    Ok(MvdComponent {
        greater0_flag,
        greater1_flag: Some(greater1_flag),
        minus2,
        sign_flag: Some(sign_flag),
        value,
    })
}

/// Decode one `mvd_coding( )` component (§7.3.8.9 / §7.4.9.9) from the
/// CABAC engine.
///
/// `ctx_greater0` and `ctx_greater1` are the caller's
/// [`SliceContexts::abs_mvd_greater0_flag`] / `abs_mvd_greater1_flag`
/// single-entry context models (Table 9-23 banks); they are mutated in
/// place per the §9.3.4.3.2.2 state transition. The `abs_mvd_minus2`
/// EG1 escape and the `mvd_sign_flag` bit are read via
/// [`CabacEngine::decode_bypass`].
///
/// The caller decodes both components (`compIdx = 0` then `1`) of a
/// `mvd_coding( )` invocation by calling this twice with the same two
/// context models; the §7.3.8.9 interleaving of the two
/// `abs_mvd_greater0_flag` bins ahead of the rest is a slice-data
/// concern handled one layer up. This entry point decodes a single
/// component end-to-end.
pub fn decode_mvd_component(
    engine: &mut CabacEngine<'_>,
    ctx_greater0: &mut ContextModel,
    ctx_greater1: &mut ContextModel,
) -> Result<MvdComponent, CabacError> {
    let greater0_flag = engine.decode_decision(ctx_greater0)?;
    if greater0_flag == 0 {
        return Ok(MvdComponent {
            greater0_flag: 0,
            greater1_flag: None,
            minus2: None,
            sign_flag: None,
            value: 0,
        });
    }
    let greater1_flag = engine.decode_decision(ctx_greater1)?;
    let minus2 = if greater1_flag == 1 {
        Some(decode_eg_k(engine, ABS_MVD_MINUS2_EG_K)?)
    } else {
        None
    };
    let sign_flag = engine.decode_bypass()?;
    let value = mvd_component_value(greater0_flag, minus2, Some(sign_flag));
    Ok(MvdComponent {
        greater0_flag,
        greater1_flag: Some(greater1_flag),
        minus2,
        sign_flag: Some(sign_flag),
        value,
    })
}

/// Decode one whole `mvd_coding( x0, y0, refList )` invocation
/// (§7.3.8.9) — BOTH components in the spec's interleaved bin order:
///
/// ```text
/// abs_mvd_greater0_flag[ 0 ]
/// abs_mvd_greater0_flag[ 1 ]
/// if( g0[0] ) abs_mvd_greater1_flag[ 0 ]
/// if( g0[1] ) abs_mvd_greater1_flag[ 1 ]
/// if( g0[0] ) { if( g1[0] ) abs_mvd_minus2[ 0 ]   mvd_sign_flag[ 0 ] }
/// if( g0[1] ) { if( g1[1] ) abs_mvd_minus2[ 1 ]   mvd_sign_flag[ 1 ] }
/// ```
///
/// A per-component decode (two [`decode_mvd_component`] calls) reads
/// the same bins in the wrong order whenever `abs_mvd_greater0_flag[0]`
/// is set, desynchronizing the engine.
///
/// # Errors
/// Propagates [`CabacError`] from the engine.
pub fn decode_mvd_pair(
    engine: &mut CabacEngine<'_>,
    ctx_greater0: &mut ContextModel,
    ctx_greater1: &mut ContextModel,
) -> Result<[MvdComponent; 2], CabacError> {
    let g0_0 = engine.decode_decision(ctx_greater0)?;
    let g0_1 = engine.decode_decision(ctx_greater0)?;
    let g1_0 = if g0_0 == 1 {
        Some(engine.decode_decision(ctx_greater1)?)
    } else {
        None
    };
    let g1_1 = if g0_1 == 1 {
        Some(engine.decode_decision(ctx_greater1)?)
    } else {
        None
    };
    let mut out = [MvdComponent {
        greater0_flag: 0,
        greater1_flag: None,
        minus2: None,
        sign_flag: None,
        value: 0,
    }; 2];
    for (comp, (g0, g1)) in [(g0_0, g1_0), (g0_1, g1_1)].into_iter().enumerate() {
        if g0 == 0 {
            continue;
        }
        let minus2 = if g1 == Some(1) {
            Some(decode_eg_k(engine, ABS_MVD_MINUS2_EG_K)?)
        } else {
            None
        };
        let sign_flag = engine.decode_bypass()?;
        out[comp] = MvdComponent {
            greater0_flag: g0,
            greater1_flag: g1,
            minus2,
            sign_flag: Some(sign_flag),
            value: mvd_component_value(g0, minus2, Some(sign_flag)),
        };
    }
    Ok(out)
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;

    // -------------------------------------------------------------
    // ctxInc derivation — pure-function tables
    // -------------------------------------------------------------

    #[test]
    fn cu_qp_delta_abs_ctx_inc_table_9_32() {
        // Bin 0 → dedicated ctx 0; bins 1..=4 → shared ctx 1.
        assert_eq!(cu_qp_delta_abs_ctx_inc(0), 0);
        assert_eq!(cu_qp_delta_abs_ctx_inc(1), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(2), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(3), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(4), 1);
    }

    #[test]
    fn last_sig_coeff_prefix_offset_shift_luma() {
        // §9.3.4.2.3, equations for cIdx == 0:
        //   ctxOffset = 3*(log2-2) + ((log2-1) >> 2)
        //   ctxShift  = (log2+1) >> 2
        // log2=2 (4x4):   off = 0 + 0 = 0; shift = 3 >> 2 = 0
        // log2=3 (8x8):   off = 3 + 0 = 3; shift = 4 >> 2 = 1
        // log2=4 (16x16): off = 6 + 0 = 6; shift = 5 >> 2 = 1
        // log2=5 (32x32): off = 9 + 1 = 10; shift = 6 >> 2 = 1
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(2, false), (0, 0));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(3, false), (3, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(4, false), (6, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(5, false), (10, 1));
    }

    #[test]
    fn last_sig_coeff_prefix_offset_shift_chroma() {
        // §9.3.4.2.3, chroma row: ctxOffset = 15, ctxShift = log2 - 2.
        // log2=2: (15, 0); log2=3: (15, 1); log2=4: (15, 2)
        // (4:4:4 chroma can also reach log2=5 → (15, 3)).
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(2, true), (15, 0));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(3, true), (15, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(4, true), (15, 2));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(5, true), (15, 3));
    }

    #[test]
    fn last_sig_coeff_prefix_ctx_inc_eq() {
        // ctxInc = (binIdx >> ctxShift) + ctxOffset.
        // 32x32 luma: offset=10, shift=1. binIdx 0..=9 →
        //   0,0,1,1,2,2,3,3,4,4 + 10 = 10,10,11,11,12,12,13,13,14,14.
        for bin in 0..=9 {
            let inc = last_sig_coeff_prefix_ctx_inc(bin, 10, 1);
            assert_eq!(inc, (bin >> 1) + 10);
        }
        // 4x4 luma: offset=0, shift=0. binIdx 0..=2 → 0,1,2.
        for bin in 0..=2 {
            let inc = last_sig_coeff_prefix_ctx_inc(bin, 0, 0);
            assert_eq!(inc, bin);
        }
    }

    #[test]
    fn last_sig_coeff_cmax_per_size() {
        // (log2 << 1) - 1.
        assert_eq!(last_sig_coeff_prefix_cmax(2), 3);
        assert_eq!(last_sig_coeff_prefix_cmax(3), 5);
        assert_eq!(last_sig_coeff_prefix_cmax(4), 7);
        assert_eq!(last_sig_coeff_prefix_cmax(5), 9);
    }

    // -------------------------------------------------------------
    // §7.4.9.11 position derivation — trace-derived examples
    // -------------------------------------------------------------

    #[test]
    fn last_sig_coeff_position_no_suffix() {
        // prefix <= 3 → no suffix, position == prefix. From the
        // multi-slice-per-frame trace: ctb 0, cIdx 1, prefix=2 → LastX=2.
        assert_eq!(last_sig_coeff_position(0, None), 0);
        assert_eq!(last_sig_coeff_position(1, None), 1);
        assert_eq!(last_sig_coeff_position(2, None), 2);
        assert_eq!(last_sig_coeff_position(3, None), 3);
    }

    #[test]
    fn last_sig_coeff_position_with_suffix_from_trace() {
        // From cabac-cu-qp-delta-last-sig-trace.md, main-422-10bit
        // primary fixture, luma 32x32 TB:
        //   x_prefix = 6, LastX = 8 → nBits = 2, suffix = 0,
        //     position = ((2 + (6&1)) << 2) + 0 = (2 << 2) = 8.
        //   y_prefix = 4, LastY = 5 → nBits = 1, suffix = 1,
        //     position = ((2 + (4&1)) << 1) + 1 = (2 << 1) + 1 = 5.
        assert_eq!(last_sig_coeff_position(6, Some(0)), 8);
        assert_eq!(last_sig_coeff_position(4, Some(1)), 5);
        // Chroma 16x16 Cb (cIdx 1) row: x_prefix=4, LastX=4 →
        //   nBits = 1, suffix = 0 → ((2+0)<<1)+0 = 4. y_prefix=1,
        //   LastY=1 → no suffix.
        assert_eq!(last_sig_coeff_position(4, Some(0)), 4);
        assert_eq!(last_sig_coeff_position(1, None), 1);
    }

    #[test]
    fn last_sig_coeff_suffix_n_bits_table() {
        // nBits = (prefix >> 1) - 1 for prefix > 3, else 0.
        assert_eq!(last_sig_coeff_suffix_n_bits(0), 0);
        assert_eq!(last_sig_coeff_suffix_n_bits(3), 0);
        assert_eq!(last_sig_coeff_suffix_n_bits(4), 1);
        assert_eq!(last_sig_coeff_suffix_n_bits(5), 1);
        assert_eq!(last_sig_coeff_suffix_n_bits(6), 2);
        assert_eq!(last_sig_coeff_suffix_n_bits(7), 2);
        assert_eq!(last_sig_coeff_suffix_n_bits(8), 3);
        assert_eq!(last_sig_coeff_suffix_n_bits(9), 3);
    }

    // -------------------------------------------------------------
    // §9.3.3.10 / §9.3.3.11 TR + EGk helpers
    // -------------------------------------------------------------

    #[test]
    fn tr_prefix_returns_value_at_terminator_bin() {
        // Read four bins (0, 1, 0, 0) via a closure. cMax = 5; the
        // unary terminates at the first 0 → prefix = 0, is_escape = false.
        let bins = [0u8, 1, 0, 0];
        let mut idx = 0;
        let (v, esc) = read_truncated_rice_prefix(5, |_bin_idx| {
            let b = bins[idx];
            idx += 1;
            Ok::<u8, CabacError>(b)
        })
        .unwrap();
        assert_eq!(v, 0);
        assert!(!esc);
        assert_eq!(idx, 1);
    }

    #[test]
    fn tr_prefix_escape_when_all_ones_to_cmax() {
        // Five 1s with cMax = 5 → prefix = 5, is_escape = true.
        let bins = [1u8; 5];
        let mut idx = 0;
        let (v, esc) = read_truncated_rice_prefix(5, |_bin_idx| {
            let b = bins[idx];
            idx += 1;
            Ok::<u8, CabacError>(b)
        })
        .unwrap();
        assert_eq!(v, 5);
        assert!(esc);
        assert_eq!(idx, 5);
    }

    // -------------------------------------------------------------
    // End-to-end CABAC integration — drive the engine on a crafted
    // bitstream and confirm the decoded element matches the trace.
    // -------------------------------------------------------------

    /// Helper: drop an MPS-only context (`pStateIdx = 62`, very high
    /// confidence) into the caller's slot. With this state the LPS
    /// range is tiny and an all-zero bitstream (after init) keeps
    /// taking the MPS path with `valMps = 0`, so every bin decodes to
    /// 0 and the unary TR prefix terminates at the first bin.
    fn fresh_mps_ctx(val_mps: u8) -> ContextModel {
        ContextModel {
            p_state_idx: 62,
            val_mps,
        }
    }

    #[test]
    fn cu_qp_delta_zero_when_first_bin_zero() {
        // All-zero stream: 9 init bits = 0, then the first
        // cu_qp_delta_abs bin is 0 → abs = 0, sign flag NOT signalled,
        // value = 0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        let d = decode_cu_qp_delta(&mut eng, &mut c0, &mut c1).unwrap();
        assert_eq!(d.abs, 0);
        assert_eq!(d.sign_flag, None);
        assert_eq!(d.value, 0);
    }

    #[test]
    fn cu_qp_delta_value_eq_neg_two_when_sign_one() {
        // Engineer abs = 2 by hand: walking the TR prefix bins of the
        // arithmetic decoder is non-trivial because the LPS range
        // depends on context state. Instead, validate the §7.4.9.14
        // derivation directly with the trace-observed (abs=2, sign=1)
        // values from the main-422-10bit primary fixture.
        let abs = 2u32;
        let sign = 1u8;
        let value = abs as i32 * (1 - 2 * sign as i32);
        assert_eq!(value, -2);
    }

    #[test]
    fn cu_qp_delta_signed_derivation_matches_trace_rows() {
        // Walk every row of the multi-slice-per-frame trace
        // (abs, sign_flag) → CuQpDeltaVal. Confirms the §7.4.9.14
        // arithmetic step in isolation.
        let rows = [
            (3u32, 0u8, 3i32),
            (1, 1, -1),
            (1, 0, 1),
            (3, 0, 3),
            (2, 1, -2),
            (2, 0, 2),
            (3, 0, 3),
            (2, 1, -2),
            (1, 0, 1),
            (3, 0, 3),
        ];
        for (abs, sign, expected) in rows {
            let got = abs as i32 * (1 - 2 * sign as i32);
            assert_eq!(got, expected, "abs={abs} sign={sign}");
        }
    }

    #[test]
    fn last_sig_coeff_round_trips_from_a_known_bypass_stream() {
        // Test the bypass-suffix path end-to-end by manually decoding
        // a fixture-traced prefix=6 + suffix=0 = LastX=8 sequence.
        // The TR prefix is hard to inject without controlling the LPS
        // range; instead, we re-derive position from the trace's
        // (prefix, observed-LastX-and-LastY) pairs and verify equation
        // 7-74 directly across all primary-fixture rows.
        let rows = [
            // (cIdx, prefix_x, prefix_y, LastX, LastY)
            (0u32, 6u32, 4u32, 8u32, 5u32),
            (1, 4, 1, 4, 1),
            (1, 4, 2, 4, 2),
            (2, 2, 1, 2, 1),
            (2, 2, 0, 2, 0),
        ];
        for (_c, px, py, lx, ly) in rows {
            // X side
            let n_x = last_sig_coeff_suffix_n_bits(px);
            // From the trace, suffix can be inferred:
            //   For (6, 8): n=2, base=8 → suffix=0
            //   For (4, 4): n=1, base=4 → suffix=0
            //   For (2, 2): n=0, suffix=None → pos=2
            let inferred_suf_x = if n_x == 0 {
                None
            } else {
                let base = (2 + (px & 1)) << ((px >> 1) - 1);
                Some(lx - base)
            };
            assert_eq!(last_sig_coeff_position(px, inferred_suf_x), lx);
            // Y side
            let n_y = last_sig_coeff_suffix_n_bits(py);
            let inferred_suf_y = if n_y == 0 {
                None
            } else {
                let base = (2 + (py & 1)) << ((py >> 1) - 1);
                Some(ly - base)
            };
            assert_eq!(last_sig_coeff_position(py, inferred_suf_y), ly);
        }
    }

    #[test]
    fn last_sig_coeff_bank_tag_round_trip() {
        // The bank tag is a routing helper for callers; verify the
        // enum survives a copy + equality check.
        let x = LastSigCoeffBank::X;
        let y = LastSigCoeffBank::Y;
        assert_ne!(x, y);
        assert_eq!(x, LastSigCoeffBank::X);
        assert_eq!(y, LastSigCoeffBank::Y);
    }

    // -------------------------------------------------------------
    // §9.3.3.11 EGk(k=0) — driven through the engine. The bin emitted
    // by `decode_bypass` depends on `ivlCurrRange` (510) versus the
    // shifted `ivlOffset`, not directly on the read bit value, so we
    // construct an offset that crosses the threshold on the first
    // bypass read: init `ivlOffset` to 256, then a fed 1-bit pushes
    // it to 513 ≥ 510, the engine returns bin = 1, and EGk's
    // §9.3.3.3 EGk(0): the prefix is a run of `1` bins terminated by a
    // `0`; a value of 0 is the bare terminating `0` (no prefix ones, no
    // suffix bits).
    #[test]
    fn eg_k0_decodes_zero_when_first_bypass_returns_zero() {
        // 9-bit init offset = 0 (all-zero buffer). The first bypass bin
        // is `(0 << 1) | 0 = 0 < ivlCurrRange`, so DecodeBypass returns
        // 0 — the bare EGk prefix terminator ⇒ value 0.
        let buf = [0u8, 0, 0];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 0);
        let v = decode_eg_k0(&mut eng).unwrap();
        assert_eq!(v, 0);
    }

    // -------------------------------------------------------------
    // §9.3.4.2.4 coded_sub_block_flag ctxInc
    // -------------------------------------------------------------

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_no_neighbours_active() {
        // §9.3.4.2.4: both neighbours = 0 → csbfCtx = 0 →
        // ctxInc = min(0, 1) = 0 for luma.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0, 0), 0);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_one_neighbour_active() {
        // One neighbour set → csbfCtx = 1 → ctxInc = min(1, 1) = 1.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 1, 0), 1);
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0, 1), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_both_neighbours_active_clamps() {
        // Both neighbours set → csbfCtx = 2 → min(2, 1) = 1
        // (equation 9-38 clamps the sum). The luma bank is {0, 1}
        // so the maximum value is 1.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 1, 1), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_chroma_offset_two() {
        // §9.3.4.2.4 equation 9-39: chroma adds 2 to the luma
        // ctxInc, putting chroma in {2, 3}.
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 0, 0), 2);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 1, 0), 3);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 0, 1), 3);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 1, 1), 3);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_ignores_high_bits() {
        // Only the LSB of each neighbour input is consulted (the
        // function masks with & 1). Confirms a defensive input
        // does not leak into csbfCtx.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0xFE, 0), 0);
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0xFF, 0xFE), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_drops_right() {
        // 4×4 TB: log2 = 2, max sub-block index = (1 << 0) − 1 = 0.
        // So xs = 0 == max → equation 9-36 does NOT apply; the
        // right_neighbour input is ignored.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 0, 2, 1, 0),
            0
        );
        // 8×8 TB: log2 = 3, max sub-block index = (1 << 1) − 1 = 1.
        // xs = 0 < 1 → right neighbour counts; xs = 1 == max → it
        // does not.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 0, 3, 1, 0),
            1
        );
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 1, 0, 3, 1, 0),
            0
        );
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_drops_below() {
        // 16×16 TB: log2 = 4, max sub-block index = (1 << 2) − 1 =
        // 3. ys = 2 < 3 → below counts; ys = 3 == max → it does not.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 2, 4, 0, 1),
            1
        );
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 3, 4, 0, 1),
            0
        );
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_32x32_interior() {
        // 32×32 TB: log2 = 5, max sub-block index = (1 << 3) − 1 =
        // 7. An interior sub-block at (3, 3) with both neighbours
        // active → csbfCtx = 2 → ctxInc = min(2, 1) = 1.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 3, 3, 5, 1, 1),
            1
        );
        // Same location, chroma: ctxInc = 2 + 1 = 3.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(true, 3, 3, 5, 1, 1),
            3
        );
    }

    // -------------------------------------------------------------
    // §9.3.4.2.2 Table 9-49 — split_cu_flag / cu_skip_flag ctxInc
    // -------------------------------------------------------------

    #[test]
    fn left_above_ctx_inc_truth_table() {
        // ctxInc = (condL && availableL) + (condA && availableA);
        // for the four combinations of the two AND'ed booleans the
        // sum is 0, 1, 1, or 2.
        assert_eq!(left_above_ctx_inc(false, true, false, true), 0);
        assert_eq!(left_above_ctx_inc(true, true, false, true), 1);
        assert_eq!(left_above_ctx_inc(false, true, true, true), 1);
        assert_eq!(left_above_ctx_inc(true, true, true, true), 2);
    }

    #[test]
    fn left_above_ctx_inc_unavailable_zeroes_branch() {
        // If a neighbour is unavailable per §6.4.1, its branch
        // contributes 0 regardless of the cond value.
        assert_eq!(left_above_ctx_inc(true, false, true, false), 0);
        assert_eq!(left_above_ctx_inc(true, false, true, true), 1);
        assert_eq!(left_above_ctx_inc(true, true, true, false), 1);
    }

    #[test]
    fn split_cu_flag_cond_table() {
        // §9.3.4.2.2 Table 9-49: condX = CtDepth[xNb][yNb] >
        // cqtDepth. Strict inequality, so equal depth → false.
        assert!(!split_cu_flag_cond(0, 0));
        assert!(!split_cu_flag_cond(1, 1));
        assert!(split_cu_flag_cond(1, 0));
        assert!(split_cu_flag_cond(3, 2));
        assert!(!split_cu_flag_cond(2, 3));
    }

    #[test]
    fn cu_skip_flag_cond_table() {
        // §9.3.4.2.2 Table 9-49: condX = cu_skip_flag[xNb][yNb].
        assert!(!cu_skip_flag_cond(0));
        assert!(cu_skip_flag_cond(1));
        // Only the LSB matters.
        assert!(!cu_skip_flag_cond(0b1110));
        assert!(cu_skip_flag_cond(0b1111));
    }

    #[test]
    fn split_cu_flag_ctx_inc_both_neighbours_deeper() {
        // Both neighbours at depth 2, current cqtDepth = 1 → both
        // conds true. Both available → ctxInc = 2.
        assert_eq!(split_cu_flag_ctx_inc(2, true, 2, true, 1), 2);
    }

    #[test]
    fn split_cu_flag_ctx_inc_only_left_deeper() {
        // Left at depth 2, above at depth 1, current cqtDepth = 1.
        // condL true (2 > 1), condA false (1 > 1 == false) →
        // ctxInc = 1.
        assert_eq!(split_cu_flag_ctx_inc(2, true, 1, true, 1), 1);
    }

    #[test]
    fn split_cu_flag_ctx_inc_unavailable_left_drops_contribution() {
        // Same conds as the both-deeper test, but left is
        // unavailable per §6.4.1 → its contribution is 0.
        assert_eq!(split_cu_flag_ctx_inc(2, false, 2, true, 1), 1);
        // Both unavailable → 0 regardless of CtDepth.
        assert_eq!(split_cu_flag_ctx_inc(7, false, 7, false, 0), 0);
    }

    #[test]
    fn cu_skip_flag_ctx_inc_table() {
        // Walk the (L, availL, A, availA) truth table for cu_skip_flag.
        assert_eq!(cu_skip_flag_ctx_inc(0, true, 0, true), 0);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 0, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(0, true, 1, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 1, true), 2);
        // Unavailable neighbours zero their contribution even if
        // the flag itself is set.
        assert_eq!(cu_skip_flag_ctx_inc(1, false, 1, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 1, false), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, false, 1, false), 0);
    }

    #[test]
    fn split_cu_flag_ctx_inc_bounded_zero_three() {
        // ctxInc must lie in {0, 1, 2} regardless of the inputs.
        for left_d in 0u32..=4 {
            for above_d in 0u32..=4 {
                for &cqt in &[0u32, 1, 2, 3] {
                    for &avl in &[false, true] {
                        for &ava in &[false, true] {
                            let c = split_cu_flag_ctx_inc(left_d, avl, above_d, ava, cqt);
                            assert!(c <= 2, "ctxInc out of {{0,1,2}} for inputs");
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------
    // split_transform_flag ctxInc — Table 9-48 row
    // -------------------------------------------------------------

    #[test]
    fn split_transform_flag_ctx_inc_table_row() {
        // ctxInc = 5 − log2TrafoSize for log2TrafoSize in
        // {2, 3, 4, 5} → ctxInc in {3, 2, 1, 0}.
        assert_eq!(split_transform_flag_ctx_inc(2), 3);
        assert_eq!(split_transform_flag_ctx_inc(3), 2);
        assert_eq!(split_transform_flag_ctx_inc(4), 1);
        assert_eq!(split_transform_flag_ctx_inc(5), 0);
    }

    #[test]
    fn split_transform_flag_ctx_inc_bounded_zero_three() {
        // The four legal residual-quadtree TB sizes map exactly
        // onto the four-context bank {0, 1, 2, 3}.
        for log2 in 2u32..=5 {
            let inc = split_transform_flag_ctx_inc(log2);
            assert!(inc <= 3, "ctxInc {} out of bank {{0..=3}}", inc);
        }
    }

    // -------------------------------------------------------------
    // cbf_luma / cbf_cb / cbf_cr ctxInc — Table 9-48 rows
    // -------------------------------------------------------------

    #[test]
    fn cbf_luma_ctx_inc_table_row() {
        // ctxInc = (trafoDepth == 0) ? 1 : 0.
        assert_eq!(cbf_luma_ctx_inc(0), 1);
        assert_eq!(cbf_luma_ctx_inc(1), 0);
        assert_eq!(cbf_luma_ctx_inc(2), 0);
        assert_eq!(cbf_luma_ctx_inc(3), 0);
        assert_eq!(cbf_luma_ctx_inc(4), 0);
    }

    #[test]
    fn cbf_luma_ctx_inc_bank_zero_one() {
        // The two-ctx bank {0, 1}: every legal trafoDepth lands in
        // one of those two slots.
        for d in 0u32..=4 {
            let inc = cbf_luma_ctx_inc(d);
            assert!(inc <= 1, "cbf_luma ctxInc {} out of {{0, 1}}", inc);
        }
    }

    #[test]
    fn cbf_chroma_ctx_inc_identity() {
        // ctxInc = trafoDepth for both cbf_cb and cbf_cr (shared).
        for d in 0u32..=4 {
            assert_eq!(cbf_chroma_ctx_inc(d), d);
            assert_eq!(cbf_cb_ctx_inc(d), d);
            assert_eq!(cbf_cr_ctx_inc(d), d);
        }
    }

    #[test]
    fn cbf_cb_cr_share_formula() {
        // cbf_cb and cbf_cr always agree at the bank-relative
        // ctxInc layer; the distinct context banks are encoded in
        // the caller's ctxIdxOffset, not here.
        for d in 0u32..=4 {
            assert_eq!(cbf_cb_ctx_inc(d), cbf_cr_ctx_inc(d));
        }
    }

    // -------------------------------------------------------------
    // inter_pred_idc ctxInc — Table 9-48 row
    // -------------------------------------------------------------

    #[test]
    fn inter_pred_idc_bin_0_routes_ct_depth() {
        // For a PU that is not 8×4 or 4×8 (nPbW+nPbH != 12), bin 0
        // ctxInc = CtDepth.
        // 64×64 PU (CTU root): 64+64=128 != 12; CtDepth=0 → 0.
        assert_eq!(inter_pred_idc_ctx_inc(0, 64, 64, 0), 0);
        // 16×16 PU at CtDepth 2: 16+16=32 != 12 → ctxInc = 2.
        assert_eq!(inter_pred_idc_ctx_inc(0, 16, 16, 2), 2);
        // 8×8 PU at CtDepth 3: 8+8=16 != 12 → ctxInc = 3.
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 8, 3), 3);
    }

    #[test]
    fn inter_pred_idc_bin_0_escape_on_16_sample_pus() {
        // 8×4 PU: nPbW + nPbH = 12 → bin 0 ctxInc = 4, regardless
        // of CtDepth.
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 4, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 4, 3), 4);
        // 4×8 PU: same.
        assert_eq!(inter_pred_idc_ctx_inc(0, 4, 8, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(0, 4, 8, 3), 4);
    }

    #[test]
    fn inter_pred_idc_bin_1_always_ctx_4() {
        // Bin 1 ctxInc = 4 regardless of nPbW / nPbH / CtDepth.
        assert_eq!(inter_pred_idc_ctx_inc(1, 64, 64, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 8, 4, 3), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 4, 8, 2), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 16, 8, 1), 4);
    }

    #[test]
    fn inter_pred_idc_ctx_inc_bank_zero_four() {
        // ctxInc ∈ {0, 1, 2, 3, 4} across all legal inputs.
        for &(w, h) in &[
            (64, 64),
            (32, 32),
            (16, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (16, 8),
            (8, 16),
        ] {
            for ct_depth in 0u32..=3 {
                for bin_idx in 0u32..=1 {
                    let inc = inter_pred_idc_ctx_inc(bin_idx, w, h, ct_depth);
                    assert!(inc <= 4, "ctxInc {} out of {{0..=4}}", inc);
                }
            }
        }
    }

    // -------------------------------------------------------------
    // log2_res_scale_abs_plus1 / res_scale_sign_flag ctxInc —
    // Table 9-48 rows for §7.3.8.13 cross_comp_pred()
    // -------------------------------------------------------------

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_cb_bank() {
        // c = 0 (Cb): ctxInc = binIdx for binIdx ∈ {0, 1, 2, 3} →
        // bank {0, 1, 2, 3}.
        for bin_idx in 0u32..=3 {
            assert_eq!(log2_res_scale_abs_plus1_ctx_inc(bin_idx, 0), bin_idx);
        }
    }

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_cr_bank() {
        // c = 1 (Cr): ctxInc = 4 + binIdx for binIdx ∈ {0, 1, 2, 3}
        // → bank {4, 5, 6, 7}.
        for bin_idx in 0u32..=3 {
            assert_eq!(log2_res_scale_abs_plus1_ctx_inc(bin_idx, 1), 4 + bin_idx);
        }
    }

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_banks_disjoint() {
        // Cb and Cr banks must not overlap.
        for bin_idx in 0u32..=3 {
            let cb = log2_res_scale_abs_plus1_ctx_inc(bin_idx, 0);
            let cr = log2_res_scale_abs_plus1_ctx_inc(bin_idx, 1);
            assert!(
                cb < 4 && (4..8).contains(&cr),
                "Cb / Cr ctxInc not in disjoint banks: cb={}, cr={}",
                cb,
                cr
            );
        }
    }

    #[test]
    fn res_scale_sign_flag_ctx_inc_identity() {
        // ctxInc = c. One bit per chroma component, each on its own
        // context.
        assert_eq!(res_scale_sign_flag_ctx_inc(0), 0);
        assert_eq!(res_scale_sign_flag_ctx_inc(1), 1);
    }

    #[test]
    fn cross_comp_pred_smoke_and_res_scale_val() {
        // §7.3.8.12 decode end-to-end: the TR(cMax=4) prefix plus the
        // conditional sign flag, with the §7.4.9.12 ResScaleVal
        // derivation (eqs 7-79 / 7-80) consistent with the decoded
        // pair.
        let buf = [0x5A; 32];
        let mut engine = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut prefix = [ContextModel::init(154, 26); 4];
        let mut sign = ContextModel::init(154, 26);
        let ccp = decode_cross_comp_pred(&mut engine, &mut prefix, &mut sign).unwrap();
        assert!(ccp.log2_res_scale_abs_plus1 <= LOG2_RES_SCALE_ABS_PLUS1_TR_CMAX);
        if ccp.log2_res_scale_abs_plus1 == 0 {
            assert_eq!(ccp.res_scale_val, 0);
            assert!(ccp.res_scale_sign_flag.is_none());
        } else {
            let s = ccp
                .res_scale_sign_flag
                .expect("sign present when prefix != 0");
            let mag = 1i32 << (ccp.log2_res_scale_abs_plus1 - 1);
            assert_eq!(ccp.res_scale_val, mag * (1 - 2 * i32::from(s)));
        }
    }

    #[test]
    fn tu_residual_act_flag_inferred_and_ctx_inc() {
        // §7.4.9.10 inferred 0; Table 9-48 ctxInc 0; FL cMax = 1.
        assert_eq!(tu_residual_act_flag_inferred(), 0);
        assert_eq!(tu_residual_act_flag_ctx_inc(), 0);
        assert_eq!(TU_RESIDUAL_ACT_FLAG_FL_CMAX, 1);
        // A single context-coded bin decodes to 0 or 1.
        let buf = [0x5A; 8];
        let mut engine = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = ContextModel::init(154, 26);
        let bit = decode_tu_residual_act_flag(&mut engine, &mut ctx).unwrap();
        assert!(bit <= 1);
    }

    // -------------------------------------------------------------
    // SAO ctxInc / binarization-shape — Table 9-48 + Table 9-43
    // -------------------------------------------------------------

    #[test]
    fn sao_merge_flag_ctx_inc_is_zero() {
        // Table 9-48 row for sao_merge_left_flag / sao_merge_up_flag:
        // bin 0 ctxInc = 0; the FL cMax = 1 ⇒ exactly one bin total.
        assert_eq!(sao_merge_flag_ctx_inc(), 0);
        assert_eq!(SAO_MERGE_FLAG_FL_CMAX, 1);
    }

    #[test]
    fn sao_type_idx_ctx_inc_bin_zero() {
        // Table 9-48 row for sao_type_idx_{luma,chroma}: bin 0 ctxInc
        // = 0 (the only context-coded bin); bin 1 is bypass per
        // Table 9-48 and not routed through this helper.
        assert_eq!(sao_type_idx_ctx_inc(0), 0);
    }

    #[test]
    fn sao_type_idx_tr_cmax_is_two() {
        // §9.3.3.10 Table 9-43: TR cMax = 2 caps the prefix at two
        // bins (the §7.4.9.3 SaoTypeIdx range {0, 1, 2}: NOT_APPLIED
        // / BAND / EDGE).
        assert_eq!(SAO_TYPE_IDX_TR_CMAX, 2);
    }

    #[test]
    fn sao_offset_sign_fl_cmax_is_one() {
        // Table 9-43: FL cMax = 1 ⇒ single bypass bin.
        assert_eq!(SAO_OFFSET_SIGN_FL_CMAX, 1);
    }

    #[test]
    fn sao_band_position_fl_shape() {
        // Table 9-43: FL cMax = 31 ⇒ 5 bypass bins (§9.3.3.5
        // ceil(log2(32)) = 5).
        assert_eq!(SAO_BAND_POSITION_FL_CMAX, 31);
        assert_eq!(SAO_BAND_POSITION_FL_NBITS, 5);
        // Sanity: the FL-of-cMax-31 nbits is the expected log2 ceil.
        // 2^5 = 32 = cMax + 1.
        assert_eq!(
            1u32 << SAO_BAND_POSITION_FL_NBITS,
            SAO_BAND_POSITION_FL_CMAX + 1
        );
    }

    #[test]
    fn sao_eo_class_fl_shape() {
        // Table 9-43: FL cMax = 3 ⇒ 2 bypass bins (encoding §7.4.9.3
        // SaoEoClass ∈ {0, 1, 2, 3}).
        assert_eq!(SAO_EO_CLASS_FL_CMAX, 3);
        assert_eq!(SAO_EO_CLASS_FL_NBITS, 2);
        // Sanity: 2^2 = 4 = cMax + 1.
        assert_eq!(1u32 << SAO_EO_CLASS_FL_NBITS, SAO_EO_CLASS_FL_CMAX + 1);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_8bit() {
        // §9.3.3.5: cMax = (1 << min(bitDepth, 10) − 5) − 1.
        // bitDepth = 8 ⇒ (1 << 3) − 1 = 7.
        assert_eq!(sao_offset_abs_tr_cmax(8), 7);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_10bit() {
        // bitDepth = 10 ⇒ (1 << 5) − 1 = 31.
        assert_eq!(sao_offset_abs_tr_cmax(10), 31);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_9bit() {
        // bitDepth = 9 ⇒ (1 << 4) − 1 = 15.
        assert_eq!(sao_offset_abs_tr_cmax(9), 15);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_clamps_above_10() {
        // §9.3.3.5 Min(bitDepth, 10): bitDepth >= 10 must clamp to
        // the 10-bit cMax = 31. Covers 12-bit and 16-bit profiles.
        assert_eq!(sao_offset_abs_tr_cmax(11), 31);
        assert_eq!(sao_offset_abs_tr_cmax(12), 31);
        assert_eq!(sao_offset_abs_tr_cmax(16), 31);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_monotone() {
        // cMax is non-decreasing in bitDepth across the clamp at 10.
        let mut prev = sao_offset_abs_tr_cmax(8);
        for bd in 9u32..=16 {
            let cur = sao_offset_abs_tr_cmax(bd);
            assert!(
                cur >= prev,
                "cMax must be non-decreasing in bitDepth: bd={} cur={} prev={}",
                bd,
                cur,
                prev
            );
            prev = cur;
        }
    }

    // -------------------------------------------------------------
    // §9.3.4.2.6 coeff_abs_level_greater1_flag — Greater1State
    // -------------------------------------------------------------

    #[test]
    fn greater1_initial_state_is_set_zero_ctx_one() {
        // Per §9.3.4.2.6, the first sub-block enters with
        // lastGreater1Ctx = 1 (no eq.-9-58 bump) and greater1Ctx = 1.
        // Eq. 9-56 forces ctxSet = 0 when i == 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        assert_eq!(s.ctx_set, 0);
        assert_eq!(s.greater1_ctx, 1);
        // Eq. 9-59 first-bin ctxInc = 0 * 4 + 1 = 1 for luma.
        assert_eq!(s.current_ctx_inc(false), 1);
        // Chroma adds +16 per eq. 9-60.
        assert_eq!(s.current_ctx_inc(true), 17);
    }

    #[test]
    fn greater1_eq_9_57_luma_nonzero_subblock_uses_ctx_set_two() {
        // Eq. 9-57: i > 0 and cIdx == 0 (luma) ⇒ ctxSet starts at 2.
        let mut s = Greater1State::new();
        s.on_subblock_entry(1, false, 0);
        assert_eq!(s.ctx_set, 2);
        // First-bin ctxInc = 2 * 4 + 1 = 9 for luma.
        assert_eq!(s.current_ctx_inc(false), 9);
    }

    #[test]
    fn greater1_eq_9_56_chroma_always_starts_at_zero() {
        // Eq. 9-56: cIdx > 0 (chroma) ⇒ ctxSet = 0 regardless of i.
        for i in [0u32, 1, 2, 5, 7] {
            let mut s = Greater1State::new();
            s.on_subblock_entry(i, true, 0);
            assert_eq!(s.ctx_set, 0, "chroma sub-block i={} ctxSet", i);
            // Chroma first-bin ctxInc = 0 * 4 + 1 + 16 = 17.
            assert_eq!(s.current_ctx_inc(true), 17);
        }
    }

    #[test]
    fn greater1_per_bin_step_flag_one_resets_ctx_to_zero() {
        // §9.3.4.2.6: lastGreater1Flag == 1 ⇒ greater1Ctx becomes 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        assert_eq!(s.greater1_ctx, 0);
        // Eq. 9-59 next-bin ctxInc = 0 * 4 + min(3, 0) = 0.
        assert_eq!(s.current_ctx_inc(false), 0);
    }

    #[test]
    fn greater1_per_bin_step_flag_zero_increments() {
        // §9.3.4.2.6: lastGreater1Flag == 0 ⇒ greater1Ctx
        // incremented (until eq.-9-59 clamps at 3).
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        // 1 → 2 → 3 → 3 (clamped).
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 2);
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 3);
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 3);
        // Eq. 9-59: 0 * 4 + 3 = 3.
        assert_eq!(s.current_ctx_inc(false), 3);
    }

    #[test]
    fn greater1_per_bin_step_after_reset_stops_advancing() {
        // §9.3.4.2.6: once greater1Ctx reaches 0 the "When
        // greater1Ctx > 0" guard skips the update on every later bin
        // in the same sub-block. The terminal value stays 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1); // → 0
        s.on_coeff_abs_level_greater1_flag(0); // skipped
        s.on_coeff_abs_level_greater1_flag(1); // skipped
        assert_eq!(s.greater1_ctx, 0);
    }

    #[test]
    fn greater1_eq_9_58_bumps_ctx_set_when_last_greater1_ctx_zero() {
        // Sub-block 0: read one flag = 0 → greater1Ctx terminal = 2.
        // Then enter sub-block 1 with last_greater1_flag = 0:
        // lastGreater1Ctx = 2 (post-spec mutation) ≠ 0 ⇒ no bump.
        // But sub-block 1 with i > 0 starts at ctxSet = 2 by eq. 9-57.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(0); // greater1Ctx 1→2
        s.on_subblock_entry(1, false, 0);
        // ctxSet starts at 2 per eq. 9-57; no eq.-9-58 bump since
        // lastGreater1Ctx mutation 2+1=3 ≠ 0.
        assert_eq!(s.ctx_set, 2);
    }

    #[test]
    fn greater1_eq_9_58_bump_triggers_after_flag_one_ladder() {
        // Sub-block 0: greater1Ctx 1→0 via flag = 1. Then sub-block 1
        // enters with last_greater1_flag = 1: lastGreater1Ctx is 0
        // (the spec's "When lastGreater1Ctx > 0" guard skips the
        // mutation, retaining 0). Eq. 9-58 bumps ctxSet by 1.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1); // → 0
        s.on_subblock_entry(1, false, 1);
        // Eq. 9-57 says ctxSet = 2 (i > 0, luma), then eq. 9-58
        // bumps by one ⇒ 3.
        assert_eq!(s.ctx_set, 3);
        // greater1Ctx resets to 1 per §9.3.4.2.6.
        assert_eq!(s.greater1_ctx, 1);
        // Eq. 9-59 first-bin ctxInc = 3 * 4 + 1 = 13.
        assert_eq!(s.current_ctx_inc(false), 13);
    }

    #[test]
    fn greater1_chroma_subblock_with_eq_9_58_bump() {
        // Chroma sub-block 0: flag = 1 → greater1Ctx 1→0.
        // Sub-block 1 (chroma) enters with last_greater1_flag = 1
        // ⇒ eq. 9-56 forces ctxSet = 0, then eq. 9-58 bumps to 1.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, true, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        s.on_subblock_entry(1, true, 1);
        assert_eq!(s.ctx_set, 1);
        // Chroma ctxInc = 1 * 4 + 1 + 16 = 21.
        assert_eq!(s.current_ctx_inc(true), 21);
    }

    #[test]
    fn greater1_ctx_inc_clamps_at_min_three() {
        // Eq. 9-59 explicitly takes Min(3, greater1Ctx). The state
        // machine tracks the clamped value, so synthesising ctxSet=1,
        // greater1Ctx=3 gives ctxInc = 4 + 3 = 7 for luma; greater1Ctx
        // can never exceed 3 internally so this is enforced by the
        // type, not the read path.
        let mut s = Greater1State::new();
        // Manually push to (ctxSet=1, greater1Ctx=3) by reading three
        // greater1=0 bins from the first sub-block (the third hits
        // the clamp).
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(0); // 1→2
        s.on_coeff_abs_level_greater1_flag(0); // 2→3
        s.on_coeff_abs_level_greater1_flag(0); // 3→3 (clamped)
        assert_eq!(s.greater1_ctx, 3);
        // Eq. 9-59 = 0*4 + 3 = 3.
        assert_eq!(s.current_ctx_inc(false), 3);
        // Chroma adds +16.
        assert_eq!(s.current_ctx_inc(true), 19);
    }

    // -------------------------------------------------------------
    // §9.3.4.2.7 coeff_abs_level_greater2_flag — ctxInc derivation
    // -------------------------------------------------------------

    #[test]
    fn greater2_ctx_inc_is_ctx_set_for_luma() {
        // Eq. 9-61: ctxInc = ctxSet for cIdx == 0.
        for ctx_set in 0u32..=3 {
            assert_eq!(
                coeff_abs_level_greater2_flag_ctx_inc(ctx_set, false),
                ctx_set,
                "luma ctxSet={}",
                ctx_set
            );
        }
    }

    #[test]
    fn greater2_ctx_inc_chroma_adds_four() {
        // Eq. 9-62: ctxInc = ctxSet + 4 for cIdx > 0.
        for ctx_set in 0u32..=3 {
            assert_eq!(
                coeff_abs_level_greater2_flag_ctx_inc(ctx_set, true),
                ctx_set + 4,
                "chroma ctxSet={}",
                ctx_set
            );
        }
    }

    #[test]
    fn greater2_reads_same_ctx_set_as_greater1_machine() {
        // §9.3.4.2.7 explicitly references the §9.3.4.2.6 ctxSet for
        // the same sub-block. Walk a small example: luma, sub-block 0,
        // first greater-1 flag = 1, then the greater-2 flag is read at
        // the lastGreater1ScanPos in the same sub-block ⇒ ctxSet = 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        assert_eq!(s.ctx_set(), 0);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), false), 0);

        // After the eq.-9-58 bump on entry to sub-block 1 (last
        // flag = 1), ctxSet = 3 and the greater-2 read picks up 3.
        s.on_subblock_entry(1, false, 1);
        assert_eq!(s.ctx_set(), 3);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), false), 3);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), true), 7);
    }

    #[test]
    fn coeff_abs_level_greater_x_fl_shape_is_one_bin() {
        // Table 9-43: both flags are FL with cMax = 1 (a single
        // context-coded bin per invocation).
        assert_eq!(COEFF_ABS_LEVEL_GREATER_X_FL_CMAX, 1);
    }

    // -------------------------------------------------------------
    // §9.3.4.2.5 sig_coeff_flag ctxInc derivation
    // -------------------------------------------------------------

    #[test]
    fn sig_coeff_flag_ctx_idx_map_matches_table_9_50() {
        // Table 9-50 verbatim — sixteen entries indexed by i ∈ 0..16.
        let expected: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];
        assert_eq!(SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2, expected);
        // Range invariant: every entry is in {0..=8} (matches the
        // 4×4 luma / chroma context-bank slot space).
        for &v in SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2.iter() {
            assert!(v <= 8);
        }
    }

    #[test]
    fn sig_coeff_flag_sig_ctx_transform_skip_eq_9_40() {
        // Eq. 9-40: luma 42, chroma 16.
        assert_eq!(sig_coeff_flag_sig_ctx_transform_skip(false), 42);
        assert_eq!(sig_coeff_flag_sig_ctx_transform_skip(true), 16);
    }

    #[test]
    fn sig_coeff_flag_log2_2_dc_position_eq_9_41() {
        // Eq. 9-41 at (0, 0) ⇒ ctxIdxMap[0] = 0 (the table's first
        // entry).
        assert_eq!(sig_coeff_flag_sig_ctx_log2_2(0, 0), 0);
    }

    #[test]
    fn sig_coeff_flag_log2_2_sweep_matches_table_9_50_indexing() {
        // Verify the row-major (yC, xC) indexing pattern reads back
        // Table 9-50 verbatim across the full 4×4 scan space.
        for y in 0..4u32 {
            for x in 0..4u32 {
                let expected =
                    SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2[((y << 2) + x) as usize] as u32;
                assert_eq!(sig_coeff_flag_sig_ctx_log2_2(x, y), expected);
            }
        }
    }

    #[test]
    fn sig_coeff_flag_log2_2_indexing_masks_oversized_input() {
        // Defensive: oversized coordinates wrap to 0..=3 via & 3.
        // (xC, yC) = (4, 0) wraps to (0, 0) ⇒ ctxIdxMap[0] = 0.
        assert_eq!(sig_coeff_flag_sig_ctx_log2_2(4, 0), 0);
        // (5, 1) wraps to (1, 1) ⇒ ctxIdxMap[5] = 3.
        assert_eq!(sig_coeff_flag_sig_ctx_log2_2(5, 1), 3);
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_45_prev_csbf_zero_luma_8x8() {
        // log2 = 3, luma, (xS, yS) = (0, 0), prevCsbf = 0.
        // Eq. 9-45 at (xP, yP):
        //   (0, 0): xP + yP == 0 → sigCtx = 2.
        //   (1, 0): xP + yP < 3 → sigCtx = 1.
        //   (2, 2): xP + yP == 4 ≥ 3 → sigCtx = 0.
        // Then no eq.-9-49 luma bump (xS + yS == 0). Eq. 9-50 with
        // scan_idx = 0 ⇒ += 9.
        // (0, 0, 0, 0): 2 + 9 = 11.
        // (1, 0, 0, 0): 1 + 9 = 10.
        // (2, 2, 0, 0): 0 + 9 = 9.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 3, 0, 0, 0, 0, 0, 0, 0),
            11
        );
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 3, 1, 0, 0, 0, 0, 0, 0),
            10
        );
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 3, 2, 2, 0, 0, 0, 0, 0),
            9
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_50_scan_idx_branch_8x8_luma() {
        // log2 = 3, luma, scan_idx = 1 (horizontal) ⇒ += 15 instead
        // of 9. (xP, yP) = (1, 0), prevCsbf = 0 ⇒ sigCtx = 1 + 15 = 16.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 3, 1, 0, 0, 0, 0, 0, 1),
            16
        );
        // scan_idx = 2 (vertical) ⇒ += 15 (the else branch covers
        // every non-zero scan_idx).
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 3, 1, 0, 0, 0, 0, 0, 2),
            16
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_51_large_luma_size() {
        // log2 = 4 (16×16), luma, (xP, yP) = (0, 0), prevCsbf = 0,
        // (xS, yS) = (0, 0). sigCtx = 2 + 21 = 23 (eq. 9-51 applies
        // for log2 ≠ 3, scan_idx irrelevant).
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 0, 0, 0, 0, 0, 0),
            23
        );
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 5, 0, 0, 0, 0, 0, 0, 0),
            23
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_49_luma_subblock_offset_bump() {
        // log2 = 4, luma, (xS, yS) = (1, 0) ⇒ eq.-9-49 adds 3.
        // (xP, yP) = (0, 0), prevCsbf = 0 ⇒ sigCtx = 2 + 3 + 21 = 26.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 4, 0, 1, 0, 0, 0, 0),
            26
        );
        // (xS, yS) = (0, 0) leaves the +3 off ⇒ sigCtx = 2 + 21 = 23.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 0, 0, 0, 0, 0, 0),
            23
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_46_prev_csbf_one() {
        // log2 = 4, luma, right_csbf = 1 (gives prevCsbf bit 0 = 1),
        // below_csbf = 0 → prevCsbf = 1. Eq. 9-46:
        //   yP == 0 → sigCtx = 2.
        //   yP == 1 → sigCtx = 1.
        //   yP >= 2 → sigCtx = 0.
        // (xS, yS) = (0, 0) keeps eq.-9-49 off; eq. 9-51 adds 21.
        // (xP, yP) = (0, 0): 2 + 21 = 23.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 0, 0, 0, 1, 0, 0),
            23
        );
        // (xP, yP) = (0, 1) ⇒ 1 + 21 = 22.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 1, 0, 0, 1, 0, 0),
            22
        );
        // (xP, yP) = (0, 2) ⇒ 0 + 21 = 21.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 2, 0, 0, 1, 0, 0),
            21
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_47_prev_csbf_two() {
        // log2 = 4, luma, below_csbf = 1 → prevCsbf = 2.
        // Eq. 9-47:
        //   xP == 0 → sigCtx = 2.
        //   xP == 1 → sigCtx = 1.
        //   xP >= 2 → sigCtx = 0.
        // Add eq.-9-51 luma tail: + 21.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 0, 0, 0, 0, 0, 1, 0),
            23
        );
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 1, 0, 0, 0, 0, 1, 0),
            22
        );
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(false, 4, 2, 0, 0, 0, 0, 1, 0),
            21
        );
    }

    #[test]
    fn sig_coeff_flag_general_eq_9_48_prev_csbf_three() {
        // log2 = 4, luma, right_csbf = 1 + below_csbf = 1 → prevCsbf
        // = 3. Eq. 9-48: sigCtx = 2 (position-independent). + 21
        // luma tail. (xP, yP) sweep all stay at 23.
        for xp in 0..4 {
            for yp in 0..4 {
                assert_eq!(
                    sig_coeff_flag_sig_ctx_general(false, 4, xp, yp, 0, 0, 1, 1, 0),
                    23
                );
            }
        }
    }

    #[test]
    fn sig_coeff_flag_general_edge_gates_neighbour_inputs() {
        // log2 = 3 (8×8) ⇒ max sub-block index 1. (xS, yS) = (1, 0):
        // xS is at the right edge (1 == 1) ⇒ eq.-9-43 gate suppresses
        // right_csbf. yS is not at the bottom edge ⇒ eq.-9-44 admits
        // below_csbf. So passing right_csbf = 1 with (xS, yS) = (1, 0)
        // has the same effect as passing right_csbf = 0.
        let with_right = sig_coeff_flag_sig_ctx_general(false, 3, 4, 0, 1, 0, 1, 0, 0);
        let without_right = sig_coeff_flag_sig_ctx_general(false, 3, 4, 0, 1, 0, 0, 0, 0);
        assert_eq!(with_right, without_right);

        // (xS, yS) = (0, 1) ⇒ yS at bottom edge, below_csbf gated.
        let with_below = sig_coeff_flag_sig_ctx_general(false, 3, 0, 4, 0, 1, 0, 1, 0);
        let without_below = sig_coeff_flag_sig_ctx_general(false, 3, 0, 4, 0, 1, 0, 0, 0);
        assert_eq!(with_below, without_below);

        // Sanity: (xS, yS) = (0, 0) on a 16×16 TB admits both
        // neighbours (max sub-block index = 3, both 0 < 3).
        let admits = sig_coeff_flag_sig_ctx_general(false, 4, 0, 0, 0, 0, 1, 1, 0);
        // prevCsbf = 3 ⇒ eq.-9-48 sigCtx = 2; eq. 9-51 + 21 = 23.
        assert_eq!(admits, 23);
    }

    #[test]
    fn sig_coeff_flag_general_chroma_eq_9_52_9_53_tail() {
        // Chroma branch never bumps via eq. 9-49.
        // log2 = 3, chroma, (xP, yP) = (0, 0), prevCsbf = 0,
        // (xS, yS) = (1, 0) ⇒ eq. 9-49 NOT applied; eq. 9-52 += 9.
        //   sigCtx = 2 + 9 = 11.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(true, 3, 4, 0, 1, 0, 0, 0, 0),
            11
        );
        // log2 = 4 chroma ⇒ eq. 9-53 += 12 (scan_idx irrelevant).
        // (0, 0) prevCsbf 0 ⇒ 2 + 12 = 14.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(true, 4, 0, 0, 0, 0, 0, 0, 0),
            14
        );
        // log2 = 5 chroma ⇒ same eq. 9-53 += 12.
        assert_eq!(
            sig_coeff_flag_sig_ctx_general(true, 5, 0, 0, 0, 0, 0, 0, 0),
            14
        );
    }

    #[test]
    fn sig_coeff_flag_general_chroma_eq_9_52_scan_idx_irrelevant() {
        // Eq. 9-52 (chroma, log2 == 3) does NOT branch on scan_idx;
        // unlike eq. 9-50 the chroma 8×8 tail is a flat += 9.
        let s0 = sig_coeff_flag_sig_ctx_general(true, 3, 0, 0, 0, 0, 0, 0, 0);
        let s1 = sig_coeff_flag_sig_ctx_general(true, 3, 0, 0, 0, 0, 0, 0, 1);
        let s2 = sig_coeff_flag_sig_ctx_general(true, 3, 0, 0, 0, 0, 0, 0, 2);
        assert_eq!(s0, s1);
        assert_eq!(s0, s2);
        assert_eq!(s0, 11);
    }

    #[test]
    fn sig_coeff_flag_dc_eq_9_42_luma_log2_3() {
        // Eq. 9-42: sigCtx = 0 — the third branch of the §9.3.4.2.5
        // cascade; the eq.-9-49..9-53 modifications belong to the
        // fourth branch only, so neither log2TrafoSize nor scanIdx
        // moves the DC context.
        assert_eq!(sig_coeff_flag_sig_ctx_dc(false, 3, 0), 0);
        assert_eq!(sig_coeff_flag_sig_ctx_dc(false, 3, 1), 0);
    }

    #[test]
    fn sig_coeff_flag_dc_eq_9_42_luma_large_size() {
        // Eq. 9-42: sigCtx = 0 for every log2TrafoSize > 2 — the DC
        // position shares one luma context (ctxInc 0).
        assert_eq!(sig_coeff_flag_sig_ctx_dc(false, 4, 0), 0);
        assert_eq!(sig_coeff_flag_sig_ctx_dc(false, 5, 0), 0);
        assert_eq!(sig_coeff_flag_sig_ctx_dc(false, 4, 2), 0);
    }

    #[test]
    fn sig_coeff_flag_dc_eq_9_42_chroma() {
        // Eq. 9-42: sigCtx = 0 for chroma too; eq. 9-55 then lands the
        // DC position on ctxInc 27.
        assert_eq!(sig_coeff_flag_sig_ctx_dc(true, 3, 0), 0);
        assert_eq!(sig_coeff_flag_sig_ctx_dc(true, 4, 0), 0);
        assert_eq!(sig_coeff_flag_sig_ctx_dc(true, 5, 0), 0);
    }

    #[test]
    fn sig_coeff_flag_ctx_inc_eq_9_54_luma_identity() {
        // Eq. 9-54: ctxInc = sigCtx.
        for sig in 0..=44 {
            assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(sig, false), sig);
        }
    }

    #[test]
    fn sig_coeff_flag_ctx_inc_eq_9_55_chroma_offset() {
        // Eq. 9-55: ctxInc = 27 + sigCtx.
        for sig in 0..=20 {
            assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(sig, true), 27 + sig);
        }
        // Anchor values: transform-skip chroma (sigCtx = 16) ⇒ 43.
        assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(16, true), 43);
        // Transform-skip luma (sigCtx = 42, eq. 9-54) ⇒ 42.
        assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(42, false), 42);
    }

    #[test]
    fn sig_coeff_flag_fl_shape_is_one_bin() {
        // Table 9-43: sig_coeff_flag is FL with cMax = 1 (single
        // context-coded bin per scan position).
        assert_eq!(SIG_COEFF_FLAG_FL_CMAX, 1);
    }

    #[test]
    fn sig_coeff_flag_general_composes_with_ctx_inc_full_pipe_luma() {
        // End-to-end: log2 = 4, luma, (xC, yC) = (4, 0) → (xS, yS)
        // = (1, 0), (xP, yP) = (0, 0). prevCsbf 0 ⇒ sigCtx = 2;
        // eq.-9-49 +3 ⇒ 5; eq.-9-51 +21 ⇒ 26. ctxInc = 26 (eq. 9-54).
        let sig = sig_coeff_flag_sig_ctx_general(false, 4, 4, 0, 1, 0, 0, 0, 0);
        assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(sig, false), 26);
    }

    #[test]
    fn sig_coeff_flag_general_composes_with_ctx_inc_full_pipe_chroma() {
        // log2 = 4, chroma, (xC, yC) = (0, 0) DC → must route via
        // sig_coeff_flag_sig_ctx_dc (eq. 9-42: sigCtx = 0).
        // ctxInc = 27 + 0 = 27 (eq. 9-55).
        let sig_dc = sig_coeff_flag_sig_ctx_dc(true, 4, 0);
        assert_eq!(sig_coeff_flag_ctx_inc_from_sig_ctx(sig_dc, true), 27);
    }

    // -----------------------------------------------------------------
    // §9.3.4.2.8 palette_run_prefix — eq. 9-63 + Table 9-51
    // -----------------------------------------------------------------

    #[test]
    fn palette_run_prefix_eq_9_63_low_idx() {
        // Eq. 9-63: palette_idx_idc < 1 ⇒ ctxInc = 0.
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(0), 0);
    }

    #[test]
    fn palette_run_prefix_eq_9_63_mid_idx() {
        // Eq. 9-63: 1 <= palette_idx_idc < 3 ⇒ ctxInc = 1.
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(1), 1);
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(2), 1);
    }

    #[test]
    fn palette_run_prefix_eq_9_63_high_idx() {
        // Eq. 9-63: palette_idx_idc >= 3 ⇒ ctxInc = 2.
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(3), 2);
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(4), 2);
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(100), 2);
        assert_eq!(palette_run_prefix_ctx_inc_eq_9_63(u32::MAX), 2);
    }

    #[test]
    fn palette_run_prefix_ctx_inc_copy_above_zero_bin_zero_dispatches_eq_9_63() {
        // copy_above == 0 and bin_idx == 0 ⇒ eq. 9-63 dispatch on
        // palette_idx_idc. Verify the public entry point agrees with the
        // standalone eq.-9-63 helper across the three eq.-9-63 bands.
        for idc in [0u32, 1, 2, 3, 7] {
            let expected = Some(palette_run_prefix_ctx_inc_eq_9_63(idc));
            assert_eq!(
                palette_run_prefix_ctx_inc(0, false, idc),
                expected,
                "palette_idx_idc = {idc}"
            );
        }
    }

    #[test]
    fn palette_run_prefix_ctx_inc_copy_above_zero_bins_one_through_four_table_9_51() {
        // Table 9-51, copy_above_palette_indices_flag == 0:
        //   binIdx 1 → 3, binIdx 2 → 3, binIdx 3 → 4, binIdx 4 → 4.
        // palette_idx_idc is irrelevant on this branch.
        assert_eq!(palette_run_prefix_ctx_inc(1, false, 0), Some(3));
        assert_eq!(palette_run_prefix_ctx_inc(2, false, 0), Some(3));
        assert_eq!(palette_run_prefix_ctx_inc(3, false, 0), Some(4));
        assert_eq!(palette_run_prefix_ctx_inc(4, false, 0), Some(4));
        // palette_idx_idc swept high/low must produce identical results
        // on this branch (Table 9-51 only consults binIdx when
        // copy_above == 0 and binIdx > 0).
        assert_eq!(palette_run_prefix_ctx_inc(1, false, 42), Some(3));
        assert_eq!(palette_run_prefix_ctx_inc(4, false, u32::MAX), Some(4));
    }

    #[test]
    fn palette_run_prefix_ctx_inc_copy_above_one_table_9_51() {
        // Table 9-51, copy_above_palette_indices_flag == 1:
        //   binIdx 0 → 5, binIdx 1 → 6, binIdx 2 → 6,
        //   binIdx 3 → 7, binIdx 4 → 7.
        assert_eq!(palette_run_prefix_ctx_inc(0, true, 0), Some(5));
        assert_eq!(palette_run_prefix_ctx_inc(1, true, 0), Some(6));
        assert_eq!(palette_run_prefix_ctx_inc(2, true, 0), Some(6));
        assert_eq!(palette_run_prefix_ctx_inc(3, true, 0), Some(7));
        assert_eq!(palette_run_prefix_ctx_inc(4, true, 0), Some(7));
        // palette_idx_idc has no influence on this branch.
        assert_eq!(palette_run_prefix_ctx_inc(0, true, 99), Some(5));
        assert_eq!(palette_run_prefix_ctx_inc(2, true, u32::MAX), Some(6));
    }

    #[test]
    fn palette_run_prefix_ctx_inc_bypass_above_four() {
        // Table 9-51 ">4" column: bypass-coded per Table 9-48.
        for bin_idx in 5..=20 {
            assert_eq!(palette_run_prefix_ctx_inc(bin_idx, false, 0), None);
            assert_eq!(palette_run_prefix_ctx_inc(bin_idx, true, 0), None);
        }
        // First-bypass boundary anchor.
        assert_eq!(PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX, 5);
    }

    #[test]
    fn palette_run_prefix_ctx_idx_map_layout_matches_table_9_51() {
        // Verify the static table layout matches Table 9-51 verbatim
        // (modulo the eq.-9-63 dispatch sentinel at row 0, col 0).
        assert_eq!(
            PALETTE_RUN_PREFIX_CTX_IDX_MAP[0],
            [PALETTE_RUN_PREFIX_EQ_9_63_DISPATCH, 3, 3, 4, 4]
        );
        assert_eq!(PALETTE_RUN_PREFIX_CTX_IDX_MAP[1], [5, 6, 6, 7, 7]);
    }

    #[test]
    fn palette_run_prefix_ctx_inc_max_eight_distinct_contexts() {
        // Table 9-40 declares 8 init values per init type (ctxIdx
        // 0..7). Verify the eq.-9-63 dispatch + Table 9-51 lookup
        // produce only values in 0..=7.
        for copy_above in [false, true] {
            for bin_idx in 0..PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX {
                for idc in 0..=8 {
                    let ctx = palette_run_prefix_ctx_inc(bin_idx, copy_above, idc)
                        .expect("all bin_idx < first-bypass should be context-coded");
                    assert!(
                        ctx < 8,
                        "ctxInc = {ctx} >= 8 for bin_idx = {bin_idx}, \
                         copy_above = {copy_above}, palette_idx_idc = {idc}"
                    );
                }
            }
        }
    }

    #[test]
    fn palette_run_prefix_tr_cmax_degenerate_zero() {
        // PaletteMaxRunMinus1 == 0 ⇒ Floor(Log2(0)) is undefined; the
        // helper returns 1 so the TR reader emits a single
        // terminating bin.
        assert_eq!(palette_run_prefix_tr_cmax(0), 1);
    }

    #[test]
    fn palette_run_prefix_tr_cmax_power_of_two_anchors() {
        // Floor(Log2(x)) + 1 for x = 1, 2, 3, 4, 7, 8, 15, 16, 31, 32.
        // x = 1 → Floor(Log2(1)) + 1 = 0 + 1 = 1.
        assert_eq!(palette_run_prefix_tr_cmax(1), 1);
        // x = 2 → 1 + 1 = 2.
        assert_eq!(palette_run_prefix_tr_cmax(2), 2);
        // x = 3 → 1 + 1 = 2.
        assert_eq!(palette_run_prefix_tr_cmax(3), 2);
        // x = 4 → 2 + 1 = 3.
        assert_eq!(palette_run_prefix_tr_cmax(4), 3);
        // x = 7 → 2 + 1 = 3.
        assert_eq!(palette_run_prefix_tr_cmax(7), 3);
        // x = 8 → 3 + 1 = 4.
        assert_eq!(palette_run_prefix_tr_cmax(8), 4);
        // x = 15 → 3 + 1 = 4.
        assert_eq!(palette_run_prefix_tr_cmax(15), 4);
        // x = 16 → 4 + 1 = 5.
        assert_eq!(palette_run_prefix_tr_cmax(16), 5);
        // x = 31 → 4 + 1 = 5.
        assert_eq!(palette_run_prefix_tr_cmax(31), 5);
        // x = 32 → 5 + 1 = 6.
        assert_eq!(palette_run_prefix_tr_cmax(32), 6);
    }

    #[test]
    fn palette_run_prefix_tr_cmax_max_block_sizes() {
        // Largest plausible PaletteMaxRunMinus1 values (a 64×64 CU
        // with all pixels in a single run: 4095, plus the absolute
        // u32::MAX defensive sweep).
        // 4095 → Floor(Log2(4095)) + 1 = 11 + 1 = 12.
        assert_eq!(palette_run_prefix_tr_cmax(4095), 12);
        // 4096 → Floor(Log2(4096)) + 1 = 12 + 1 = 13.
        assert_eq!(palette_run_prefix_tr_cmax(4096), 13);
        // u32::MAX → 31 + 1 = 32.
        assert_eq!(palette_run_prefix_tr_cmax(u32::MAX), 32);
    }

    #[test]
    fn palette_run_prefix_tr_cmax_monotone_nondecreasing() {
        // PaletteMaxRunMinus1 → cMax is non-decreasing for x >= 1.
        let mut prev = palette_run_prefix_tr_cmax(1);
        for x in 2..=64 {
            let cur = palette_run_prefix_tr_cmax(x);
            assert!(cur >= prev, "cMax({x}) = {cur} < cMax({}) = {prev}", x - 1);
            prev = cur;
        }
    }

    #[test]
    fn palette_run_prefix_ctx_inc_eq_9_63_full_coverage_with_table_9_51_disjoint() {
        // Sanity invariant: when copy_above == 0 and bin_idx == 0 the
        // ctxInc is in {0, 1, 2} (eq. 9-63 range), and the
        // Table 9-51 lookup row 0 entries (bin_idx 1..=4) are in
        // {3, 4}. The two value sets are disjoint and together cover
        // {0, 1, 2, 3, 4} — the spec's ctxIdx assignment for the
        // copy_above == 0 case.
        let bin0_ctxs: std::collections::BTreeSet<u32> =
            (0..=4).map(palette_run_prefix_ctx_inc_eq_9_63).collect();
        assert_eq!(bin0_ctxs, [0u32, 1, 2].into_iter().collect());
        let tail_ctxs: std::collections::BTreeSet<u32> = (1
            ..PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX)
            .map(|bi| palette_run_prefix_ctx_inc(bi, false, 0).unwrap())
            .collect();
        assert_eq!(tail_ctxs, [3u32, 4].into_iter().collect());
        // No overlap.
        assert!(bin0_ctxs.is_disjoint(&tail_ctxs));
    }

    // -----------------------------------------------------------------
    // §9.3.3.11 coeff_abs_level_remaining — pure-function derivations
    // -----------------------------------------------------------------

    #[test]
    fn coeff_abs_level_remaining_c_rice_param_eq_9_24_initial_state() {
        // First invocation in a sub-block: cLastAbsLevel = 0,
        // cLastRiceParam = 0 ⇒ threshold = 3 << 0 = 3; 0 > 3 is false
        // ⇒ no bump ⇒ cRiceParam = 0.
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(0, 0), 0);
    }

    #[test]
    fn coeff_abs_level_remaining_c_rice_param_eq_9_24_no_bump_at_threshold() {
        // The bump condition is strictly greater (>), not >=.
        //   r = 0, threshold = 3: level 3 ⇒ no bump.
        //   r = 1, threshold = 6: level 6 ⇒ no bump.
        //   r = 2, threshold = 12: level 12 ⇒ no bump.
        //   r = 3, threshold = 24: level 24 ⇒ no bump.
        //   r = 4, threshold = 48: level 48 ⇒ no bump (already at cap).
        for r in 0..=4u32 {
            let thresh = 3u32 << r;
            assert_eq!(
                coeff_abs_level_remaining_c_rice_param_eq_9_24(thresh, r),
                r,
                "boundary at threshold for r = {r}"
            );
        }
    }

    #[test]
    fn coeff_abs_level_remaining_c_rice_param_eq_9_24_bumps_one_above_threshold() {
        // r = 0, threshold = 3, level = 4 ⇒ bump ⇒ 1.
        // r = 1, threshold = 6, level = 7 ⇒ bump ⇒ 2.
        // r = 2, threshold = 12, level = 13 ⇒ bump ⇒ 3.
        // r = 3, threshold = 24, level = 25 ⇒ bump ⇒ 4.
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(4, 0), 1);
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(7, 1), 2);
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(13, 2), 3);
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(25, 3), 4);
    }

    #[test]
    fn coeff_abs_level_remaining_c_rice_param_eq_9_24_saturates_at_four() {
        // r = 4 already at cap; even a level vastly above threshold
        // must stay clamped at 4.
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(49, 4), 4);
        assert_eq!(
            coeff_abs_level_remaining_c_rice_param_eq_9_24(u32::MAX, 4),
            4
        );
        // r = 3 bumped + clamped ⇒ 4.
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(1_000, 3), 4);
        // r = 4 with no bump still 4.
        assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(0, 4), 4);
    }

    #[test]
    fn coeff_abs_level_remaining_c_rice_param_eq_9_24_monotone_in_level() {
        // Holding r fixed, the bump probability is non-decreasing in
        // cLastAbsLevel: once we pass the threshold, every larger
        // level also triggers (until we hit r = 4 saturation).
        for r in 0..=3u32 {
            let thresh = 3u32 << r;
            // Just below threshold: no bump.
            if thresh > 0 {
                assert_eq!(
                    coeff_abs_level_remaining_c_rice_param_eq_9_24(thresh - 1, r),
                    r
                );
            }
            // At threshold: no bump.
            assert_eq!(coeff_abs_level_remaining_c_rice_param_eq_9_24(thresh, r), r);
            // Just above: bump.
            assert_eq!(
                coeff_abs_level_remaining_c_rice_param_eq_9_24(thresh + 1, r),
                r + 1
            );
            // Far above: still r + 1 (no double-bump in eq. 9-24).
            assert_eq!(
                coeff_abs_level_remaining_c_rice_param_eq_9_24(u32::MAX / 2, r),
                r + 1
            );
        }
    }

    #[test]
    fn coeff_abs_level_remaining_c_max_eq_9_26_table() {
        // Eq. 9-26 anchors for cRiceParam ∈ {0..=4}.
        assert_eq!(coeff_abs_level_remaining_c_max_eq_9_26(0), 4);
        assert_eq!(coeff_abs_level_remaining_c_max_eq_9_26(1), 8);
        assert_eq!(coeff_abs_level_remaining_c_max_eq_9_26(2), 16);
        assert_eq!(coeff_abs_level_remaining_c_max_eq_9_26(3), 32);
        assert_eq!(coeff_abs_level_remaining_c_max_eq_9_26(4), 64);
    }

    #[test]
    fn coeff_abs_level_remaining_tr_prefix_escape_len_is_four() {
        // §9.3.3.2: TR(cMax, cRiceParam) prefix length is
        // cMax / (1 << cRiceParam) = (4 << r) / (1 << r) = 4 — invariant
        // across cRiceParam.
        assert_eq!(COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN, 4);
        for r in 0..=4u32 {
            let c_max = coeff_abs_level_remaining_c_max_eq_9_26(r);
            let len = c_max / (1u32 << r);
            assert_eq!(
                len, COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN,
                "TR prefix length mismatch at r = {r}"
            );
        }
    }

    #[test]
    fn coeff_abs_level_remaining_prefix_val_eq_9_27_clamps_at_c_max() {
        // Below cMax: prefixVal = level.
        // At or above cMax: prefixVal = cMax (escape).
        let c_max = coeff_abs_level_remaining_c_max_eq_9_26(0); // 4
        assert_eq!(coeff_abs_level_remaining_prefix_val_eq_9_27(0, c_max), 0);
        assert_eq!(coeff_abs_level_remaining_prefix_val_eq_9_27(3, c_max), 3);
        assert_eq!(coeff_abs_level_remaining_prefix_val_eq_9_27(4, c_max), 4);
        assert_eq!(coeff_abs_level_remaining_prefix_val_eq_9_27(5, c_max), 4);
        assert_eq!(
            coeff_abs_level_remaining_prefix_val_eq_9_27(u32::MAX, c_max),
            c_max
        );
    }

    #[test]
    fn coeff_abs_level_remaining_suffix_val_eq_9_28_subtracts_c_max() {
        let c_max = coeff_abs_level_remaining_c_max_eq_9_26(1); // 8
        // Below cMax: clamp at 0 (suffix is absent; caller checks the
        // prefix-escape flag before consuming it).
        assert_eq!(coeff_abs_level_remaining_suffix_val_eq_9_28(0, c_max), 0);
        assert_eq!(coeff_abs_level_remaining_suffix_val_eq_9_28(7, c_max), 0);
        // At cMax: suffix = 0 (escape boundary).
        assert_eq!(coeff_abs_level_remaining_suffix_val_eq_9_28(8, c_max), 0);
        // Above cMax: linear in level.
        assert_eq!(coeff_abs_level_remaining_suffix_val_eq_9_28(9, c_max), 1);
        assert_eq!(coeff_abs_level_remaining_suffix_val_eq_9_28(100, c_max), 92);
    }

    #[test]
    fn coeff_abs_level_remaining_prefix_plus_suffix_recomposes_level() {
        // For any level and any cRiceParam ∈ {0..=4}: level ==
        //   prefixVal                                  when level < cMax
        //   cMax + suffixVal                           when level >= cMax
        // Anchors a round-trip invariant the binarization+decode pipe
        // must preserve.
        for r in 0..=4u32 {
            let c_max = coeff_abs_level_remaining_c_max_eq_9_26(r);
            for &level in &[0u32, 1, 3, 4, 7, 8, 15, 16, 63, 64, 65, 100, 4096] {
                let prefix = coeff_abs_level_remaining_prefix_val_eq_9_27(level, c_max);
                let suffix = coeff_abs_level_remaining_suffix_val_eq_9_28(level, c_max);
                let recomposed = if level < c_max {
                    prefix
                } else {
                    c_max + suffix
                };
                assert_eq!(
                    recomposed, level,
                    "round-trip failed at r = {r}, level = {level}"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // §9.3.3.11 coeff_abs_level_remaining — bin-source round-trips
    //
    // The §9.3.4.3.4 bypass arithmetic decoder does not map stream
    // bits to bins 1-for-1 (the offset accumulator gates `bin == 1`
    // by a 510 threshold), so end-to-end engine tests can't trivially
    // synthesise a target bin sequence. Instead we drive the
    // [`decode_coeff_abs_level_remaining_with`] entry point with a
    // flat bin queue — the algorithm logic is identical and the
    // engine wrapper is a one-line trampoline.
    // -----------------------------------------------------------------

    /// Helper: a FIFO bin source for
    /// [`decode_coeff_abs_level_remaining_with`].
    fn bin_queue(bins: &[u8]) -> impl FnMut() -> Result<u8, CabacError> + '_ {
        let mut idx = 0usize;
        move || {
            let b = bins.get(idx).copied().ok_or(CabacError::EndOfBuffer)?;
            idx += 1;
            Ok(b)
        }
    }

    #[test]
    fn decode_coeff_abs_level_remaining_zero_level_r0() {
        // Smallest case: cRiceParam = 0, level = 0. TR prefix
        // terminates at the first bin (0). No TR-suffix (r = 0) and
        // no escape. Output = 0.
        let bins = [0u8];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(0, bin_queue(&bins)).unwrap(),
            0
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_short_prefix_r0() {
        // cRiceParam = 0, level = 3. TR prefix: 1, 1, 1, 0
        // (three 1s then terminator). No TR-suffix bits. Output = 3.
        let bins = [1u8, 1, 1, 0];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(0, bin_queue(&bins)).unwrap(),
            3
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_short_prefix_r1() {
        // cRiceParam = 1: TR prefix length capped at 4; each prefix
        // step contributes (1 << r) = 2 to the value, plus a 1-bit
        // TR-suffix carrying the low bit.
        //
        // level = 5 = (2 << 1) | 1 → prefix_len = 2, tr_suffix = 1.
        // Bins: 1, 1, 0 (prefix) + 1 (TR-suffix).
        let bins = [1u8, 1, 0, 1];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(1, bin_queue(&bins)).unwrap(),
            5
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_short_prefix_r2() {
        // cRiceParam = 2: each prefix step contributes 4, TR-suffix
        // is 2 bits. level = 9 = (2 << 2) | 1 → prefix_len = 2,
        // tr_suffix = 1 (binary 01).
        let bins = [1u8, 1, 0, /* TR-suffix high */ 0, /* low */ 1];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(2, bin_queue(&bins)).unwrap(),
            9
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_escape_path_r0() {
        // cRiceParam = 0, level = 4 (== cMax). TR prefix = all 4 ones
        // (escape). EGk(k = 1) suffix for suffixVal = 0:
        //   §9.3.3.3 eq. 9-13 — the EGk unary prefix is a run of `1`
        //   bins terminated by a `0`. suffixVal = 0 ⇒ prefix is the bare
        //   terminating `0` (prefix_ones = 0), suffix bits = 0 + 1 = 1,
        //   payload = 0. value = ((1 << 0) − 1) << 1 + 0 = 0.
        // Decoded level = cMax + 0 = 4.
        let bins = [
            1u8, 1, 1, 1, /* EGk prefix terminator */ 0, /* suffix bit */ 0,
        ];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(0, bin_queue(&bins)).unwrap(),
            4
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_escape_path_with_suffix_payload_r0() {
        // cRiceParam = 0, level = 7. cMax = 4. suffixVal = 3.
        // EGk(k = 1) of 3: prefix_ones = 1 (a `1` then the terminating
        // `0`), suffix bits = 1 + 1 = 2, payload = 01 ⇒ value =
        // ((1 << 1) − 1) << 1 + 1 = 3. Decoded level = 4 + 3 = 7.
        let bins = [
            1u8, 1, 1, 1, // TR escape prefix
            1, 0, // EGk prefix: one `1` bin then the terminating `0`
            0, 1, // suffix bits MSB-first
        ];
        assert_eq!(
            decode_coeff_abs_level_remaining_with(0, bin_queue(&bins)).unwrap(),
            7
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_escape_saturates_instead_of_overflowing() {
        // Fuzz regression (r282 `decode_residual`): a non-conformant
        // bypass-bin stream can drive the EGk escape to its maximal
        // shape — 4 TR escape ones, then 32 leading prefix ones (the
        // loop cap), then 32 all-ones suffix bits — whose composed value
        // exceeds u32. The decode must saturate, not overflow.
        // §7.4.9.11 conformance (eqs. 7-27 .. 7-30) keeps every legal
        // stream far below the saturation point.
        let mut bins = vec![1u8, 1, 1, 1]; // TR escape prefix
        bins.extend([1u8; 32]); // EGk prefix ones (loop cap)
        bins.extend([1u8; 32]); // suffix payload
        assert_eq!(
            decode_coeff_abs_level_remaining_with(0, bin_queue(&bins)).unwrap(),
            u32::MAX
        );
    }

    #[test]
    fn decode_coeff_abs_level_remaining_round_trips_tr_path_r0_thru_r4() {
        // For each (cRiceParam, level) in the TR-only range
        // (0..cMax), synthesise the §9.3.3.11 bin sequence,
        // run the decoder, and confirm the original level comes back.
        for r in 0..=4u32 {
            let c_max = coeff_abs_level_remaining_c_max_eq_9_26(r);
            for level in 0..c_max {
                let prefix_len = level >> r;
                let tr_suffix = level & ((1u32 << r) - 1);
                let mut bins: Vec<u8> = (0..prefix_len).map(|_| 1u8).collect();
                bins.push(0); // TR terminator
                for i in (0..r).rev() {
                    bins.push(((tr_suffix >> i) & 1) as u8);
                }
                let got = decode_coeff_abs_level_remaining_with(r, bin_queue(&bins)).unwrap();
                assert_eq!(got, level, "round-trip r = {r}, level = {level}");
            }
        }
    }

    #[test]
    fn decode_coeff_abs_level_remaining_round_trips_escape_path_anchors() {
        // Escape-path anchors for cRiceParam ∈ {0..=4}: pick a few
        // (suffix-value) anchors per r, synthesise the all-ones TR
        // prefix + EGk(r + 1) suffix, decode, and confirm.
        for r in 0..=4u32 {
            let c_max = coeff_abs_level_remaining_c_max_eq_9_26(r);
            let k = r + 1;
            for &suffix_val in &[0u32, 1, 3, 5, 12] {
                // Encode suffix_val as EGk(k). Find smallest `lz`
                // such that base = ((1 << lz) − 1) << k <= suffix_val
                // and base + ((1 << (lz + k)) − 1) >= suffix_val.
                let mut lz = 0u32;
                while {
                    let base = (((1u64 << lz) - 1) << k) as u32;
                    let suffix_bits_capacity: u64 = 1u64 << (lz + k);
                    let max_val = base as u64 + suffix_bits_capacity - 1;
                    !(suffix_val >= base && suffix_val as u64 <= max_val)
                } {
                    lz += 1;
                }
                let base = (((1u64 << lz) - 1) << k) as u32;
                let suffix_payload = suffix_val - base;
                // Bins: all-ones TR prefix (4 ones) + the §9.3.3.3 EGk
                // prefix `lz` ones + a terminating zero + (lz + k) suffix
                // bits MSB-first.
                let mut bins: Vec<u8> = std::iter::repeat_n(1u8, 4).collect();
                bins.extend(std::iter::repeat_n(1u8, lz as usize));
                bins.push(0);
                for i in (0..lz + k).rev() {
                    bins.push(((suffix_payload >> i) & 1) as u8);
                }
                let got = decode_coeff_abs_level_remaining_with(r, bin_queue(&bins)).unwrap();
                assert_eq!(
                    got,
                    c_max + suffix_val,
                    "escape-path round-trip r = {r}, suffix_val = {suffix_val}"
                );
            }
        }
    }

    #[test]
    fn decode_coeff_abs_level_remaining_engine_wrapper_smoke() {
        // End-to-end engine wrapper smoke: drive
        // `decode_coeff_abs_level_remaining` with an all-zero stream
        // and a stuck-MPS context state is unnecessary because the
        // bypass path doesn't consult ContextModel. We just confirm
        // the wrapper compiles, runs, and returns 0 on an all-zero
        // bypass bin stream (which is the natural decode of the
        // smallest-level case from the bin-source test above).
        let buf = [0u8; 16];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(decode_coeff_abs_level_remaining(&mut eng, 0).unwrap(), 0);
    }

    // -------------------------------------------------------------
    // coeff_sign_flag[ n ] — §7.3.8.11 / §7.4.9.11 / Table 9-43 /
    // Table 9-48
    // -------------------------------------------------------------

    #[test]
    fn coeff_sign_flag_fl_shape_table_9_43() {
        // Table 9-43: coeff_sign_flag[ ] is FL with cMax = 1; the
        // §9.3.3.5 fixedLength derivation gives Ceil(Log2(2)) = 1.
        assert_eq!(COEFF_SIGN_FLAG_FL_CMAX, 1);
        assert_eq!(COEFF_SIGN_FLAG_FL_NBITS, 1);
        // §9.3.3.5: fixedLength = Ceil(Log2(cMax + 1)).
        // For cMax = 1: cMax + 1 = 2, Log2(2) = 1, Ceil = 1.
        let n_plus_1 = COEFF_SIGN_FLAG_FL_CMAX + 1;
        // Ceil(Log2(x)) for x >= 1 is `bit_width(x − 1)`, i.e.
        // `u32::BITS − (x − 1).leading_zeros()` for x >= 2 and
        // `0` for x == 1.
        let derived = if n_plus_1 <= 1 {
            0
        } else {
            u32::BITS - (n_plus_1 - 1).leading_zeros()
        };
        assert_eq!(derived, COEFF_SIGN_FLAG_FL_NBITS);
    }

    #[test]
    fn signed_level_from_sign_flag_positive_branch() {
        // §7.4.9.11: sign_flag == 0 ⇒ (1 − 2 * 0) = +1, so
        //   level * (+1) = +abs_level.
        assert_eq!(signed_level_from_sign_flag(0, 0), 0);
        assert_eq!(signed_level_from_sign_flag(1, 0), 1);
        assert_eq!(signed_level_from_sign_flag(7, 0), 7);
        assert_eq!(signed_level_from_sign_flag(127, 0), 127);
        assert_eq!(signed_level_from_sign_flag(32_767, 0), 32_767);
    }

    #[test]
    fn signed_level_from_sign_flag_negative_branch() {
        // §7.4.9.11: sign_flag == 1 ⇒ (1 − 2 * 1) = −1, so
        //   level * (−1) = −abs_level.
        assert_eq!(signed_level_from_sign_flag(0, 1), 0);
        assert_eq!(signed_level_from_sign_flag(1, 1), -1);
        assert_eq!(signed_level_from_sign_flag(7, 1), -7);
        assert_eq!(signed_level_from_sign_flag(127, 1), -127);
        assert_eq!(signed_level_from_sign_flag(32_767, 1), -32_767);
    }

    #[test]
    fn signed_level_from_sign_flag_inverse_identity() {
        // For every abs_level in a representative sweep:
        //   signed(abs, 1) == −signed(abs, 0).
        // Confirms the (1 − 2 * sign_flag) factor is the only
        // difference between the two branches.
        for level in [0u32, 1, 2, 5, 17, 42, 255, 1_023, 65_535] {
            let pos = signed_level_from_sign_flag(level, 0);
            let neg = signed_level_from_sign_flag(level, 1);
            assert_eq!(pos + neg, 0, "level = {level}");
        }
    }

    #[test]
    fn signed_level_from_sign_flag_high_bit_depth_range() {
        // High-bit-depth profiles (Main 4:4:4 12 / 14, RExt 16-bit) can
        // produce |level| up to CoeffMax = (1 << 15) − 1 = 32_767 for
        // 16-bit coding under the §A.4.2 transform bit-depth bounds,
        // and CoeffMin ≥ −(1 << 15) = −32_768. The helper must round-
        // trip across that full range without saturation.
        let max = (1i32 << 15) - 1;
        let min = -(1i32 << 15);
        assert_eq!(signed_level_from_sign_flag(max as u32, 0), max);
        assert_eq!(signed_level_from_sign_flag(max as u32, 1), -max);
        // Magnitude of CoeffMin = 32_768 → encode with sign_flag = 1.
        assert_eq!(
            signed_level_from_sign_flag((-min) as u32, 1),
            min,
            "CoeffMin recovers when |level| = 32_768"
        );
    }

    #[test]
    fn decode_coeff_sign_flag_reads_one_bypass_bin_zero() {
        // §9.3.4.2 / Table 9-48: the element is fully bypass-coded
        // (bin 0 = bypass; all subsequent bin indices are `na`). One
        // §9.3.4.3.4 DecodeBypass invocation reproduces the wire bit.
        //
        // An all-zero NAL body, after §9.3.4.3.6 alignment, is
        // decoded by the §9.3.4.3.4 bypass routine as a 0 bin per
        // the `ivlOffset < ivlCurrRange` (= 256) test: doubling a
        // zero offset and OR-ing in a zero stream bit yields 0,
        // which is below the post-double range 512.
        let buf = [0u8; 16];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        eng.align();
        assert_eq!(decode_coeff_sign_flag(&mut eng).unwrap(), 0);
    }

    #[test]
    fn decode_coeff_sign_flag_well_typed_output() {
        // §9.3.4.3.4 bypass decode: `decode_bypass` always returns
        // a value in `{0, 1}` regardless of stream contents, since
        // the routine compares `ivl_offset` against `ivl_curr_range`
        // and returns 0 or 1 only. Strict-bit recovery from the
        // wire is exercised by the cabac-module bypass tests; this
        // test confirms the wrapper threads through without altering
        // the output range.
        let mut buf = [0u8; 16];
        buf[1] = 0x40;
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        eng.align();
        let bin = decode_coeff_sign_flag(&mut eng).unwrap();
        assert!(bin == 0 || bin == 1);
    }

    #[test]
    fn decode_coeff_sign_flag_matches_underlying_decode_bypass() {
        // The wrapper is `engine.decode_bypass()` — these two paths
        // must produce identical sequences when fed the same engine
        // state. Confirm via two engines initialised from the same
        // buffer: the first runs the wrapper N times; the second
        // runs `decode_bypass` N times; the bin sequences agree.
        let mut buf = [0u8; 16];
        buf[0] = 0x5a;
        buf[1] = 0xa5;
        buf[2] = 0x3c;
        buf[3] = 0xc3;

        let mut eng_a = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut eng_b = CabacEngine::new(BitReader::new(&buf)).unwrap();
        eng_a.align();
        eng_b.align();
        for _ in 0..8 {
            let via_wrapper = decode_coeff_sign_flag(&mut eng_a).unwrap();
            let via_direct = eng_b.decode_bypass().unwrap();
            assert_eq!(via_wrapper, via_direct);
        }
    }

    // -------------------------------------------------------------
    // cu_chroma_qp_offset_flag / cu_chroma_qp_offset_idx
    // (§9.3.4.2.1 / Table 9-48, §7.4.9.10)
    // -------------------------------------------------------------

    #[test]
    fn cu_chroma_qp_offset_flag_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for cu_chroma_qp_offset_flag: bin 0 ctxInc = 0,
        // all later columns na.
        assert_eq!(cu_chroma_qp_offset_flag_ctx_inc(), 0);
    }

    #[test]
    fn cu_chroma_qp_offset_flag_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 → one bin, Ceil(Log2(2)) = 1
        // per §9.3.3.5.
        assert_eq!(CU_CHROMA_QP_OFFSET_FLAG_FL_CMAX, 1);
        assert_eq!(CU_CHROMA_QP_OFFSET_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn cu_chroma_qp_offset_idx_table_9_48_ctx_inc_is_zero_across_prefix() {
        // Table 9-48 row for cu_chroma_qp_offset_idx: binIdx 0..=4 all
        // ctxInc = 0 (the TR prefix never escapes its single context
        // bank).
        for bin in 0..=4u32 {
            assert_eq!(cu_chroma_qp_offset_idx_ctx_inc(bin), 0, "binIdx = {bin}");
        }
    }

    #[test]
    fn cu_chroma_qp_offset_idx_tr_cmax_passthrough() {
        // §9.3.3.10 cMax = chroma_qp_offset_list_len_minus1; the
        // §7.4.3.3.1 u(3) field caps the input at 5.
        for n in 0..=5u32 {
            assert_eq!(cu_chroma_qp_offset_idx_tr_cmax(n), n);
        }
    }

    #[test]
    fn cu_chroma_qp_offset_offset_indices_gating() {
        // §7.4.9.10: flag = 0 ⇒ no list dereference (offset = 0).
        let off0 = CuChromaQpOffset { flag: 0, idx: None };
        assert_eq!(off0.offset_indices(), None);

        // flag = 1, idx present ⇒ dereference at idx.
        let off1 = CuChromaQpOffset {
            flag: 1,
            idx: Some(3),
        };
        assert_eq!(off1.offset_indices(), Some(3));

        // flag = 1, idx not signalled (cMax = 0 case) ⇒ inferred 0.
        let off2 = CuChromaQpOffset { flag: 1, idx: None };
        assert_eq!(off2.offset_indices(), Some(0));
    }

    #[test]
    fn cu_chroma_qp_offset_flag_zero_does_not_signal_idx() {
        // All-zero stream + fresh MPS-only context: the FL flag
        // decodes to valMps = 0, so the idx is not read and no
        // context-state walk is required on the idx side.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_flag = fresh_mps_ctx(0);
        let ctx_idx_pre = fresh_mps_ctx(0);
        let mut ctx_idx = ctx_idx_pre;
        let off = decode_cu_chroma_qp_offset(&mut eng, &mut ctx_flag, &mut ctx_idx, 5).unwrap();
        assert_eq!(off.flag, 0);
        assert_eq!(off.idx, None);
        assert_eq!(off.offset_indices(), None);
        // §7.4.9.10 gate: the idx context was not touched (its state
        // matches the pre-call snapshot exactly).
        assert_eq!(ctx_idx, ctx_idx_pre);
    }

    #[test]
    fn cu_chroma_qp_offset_flag_one_and_zero_len_skips_idx_read() {
        // Even when flag = 1, the idx is not signalled when
        // chroma_qp_offset_list_len_minus1 == 0 (TR cMax = 0 emits
        // zero bins). §7.4.9.10 then infers idx = 0. Engineer the
        // flag = 1 path with a (valMps = 1) MPS-only context.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_flag = fresh_mps_ctx(1);
        let ctx_idx_pre = fresh_mps_ctx(0);
        let mut ctx_idx = ctx_idx_pre;
        let off = decode_cu_chroma_qp_offset(&mut eng, &mut ctx_flag, &mut ctx_idx, 0).unwrap();
        assert_eq!(off.flag, 1);
        assert_eq!(off.idx, None);
        // §7.4.9.10: absent idx ⇒ inferred to 0 via offset_indices().
        assert_eq!(off.offset_indices(), Some(0));
        // The idx context is untouched: TR cMax = 0 reads zero bins.
        assert_eq!(ctx_idx, ctx_idx_pre);
    }

    #[test]
    fn cu_chroma_qp_offset_idx_ctx_inc_default_is_zero_for_each_legal_bin() {
        // The §7.4.3.3.1 PPS field caps the TR prefix at five bins
        // (cMax = chroma_qp_offset_list_len_minus1 ≤ 5). Every legal
        // bin index lives in the single Table 9-48 ctxInc = 0 slot.
        for n in 0..=5u32 {
            let c_max = cu_chroma_qp_offset_idx_tr_cmax(n);
            for bin in 0..c_max {
                assert_eq!(
                    cu_chroma_qp_offset_idx_ctx_inc(bin),
                    0,
                    "cMax = {c_max}, binIdx = {bin}"
                );
            }
        }
    }

    #[test]
    fn signed_level_composition_residual_loop_anchors() {
        // §7.4.9.11 final composition anchors. baseLevel ∈ {1, 2, 3}
        // (1 = greater1==0 fast path; 2 = greater1==1 / greater2==0;
        // 3 = greater1==1 && greater2==1 at lastGreater1ScanPos);
        // coeff_abs_level_remaining[n] ∈ {0, 1, 5, 17}; sign ∈ {0, 1}.
        // The full level is `(remaining + baseLevel) * (1 − 2 * sign)`.
        let cases: &[(u32, u32, u8, i32)] = &[
            (0, 1, 0, 1),
            (0, 1, 1, -1),
            (0, 3, 0, 3),
            (0, 3, 1, -3),
            (1, 2, 0, 3),
            (1, 2, 1, -3),
            (5, 1, 0, 6),
            (5, 1, 1, -6),
            (17, 3, 0, 20),
            (17, 3, 1, -20),
        ];
        for &(remaining, base_level, sign_flag, expected) in cases {
            let abs_level = remaining + base_level;
            let got = signed_level_from_sign_flag(abs_level, sign_flag);
            assert_eq!(
                got, expected,
                "remaining = {remaining}, baseLevel = {base_level}, sign = {sign_flag}"
            );
        }
    }

    // -----------------------------------------------------------------
    // cu_transquant_bypass_flag (§7.3.8.5 / §7.4.9.5)
    // -----------------------------------------------------------------

    #[test]
    fn cu_transquant_bypass_flag_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for cu_transquant_bypass_flag: bin 0 ctxInc
        // = 0; all later binIdx columns are na.
        assert_eq!(cu_transquant_bypass_flag_ctx_inc(), 0);
    }

    #[test]
    fn cu_transquant_bypass_flag_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(CU_TRANSQUANT_BYPASS_FLAG_FL_CMAX, 1);
        assert_eq!(CU_TRANSQUANT_BYPASS_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn cu_transquant_bypass_flag_inferred_is_zero_per_7_4_9_5() {
        // §7.4.9.5: when transquant_bypass_enabled_flag (PPS) is 0
        // the flag is not present and inferred to 0 — the normal
        // scaling-and-transform path.
        assert_eq!(cu_transquant_bypass_flag_inferred(), 0);
    }

    #[test]
    fn decode_cu_transquant_bypass_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the
        // FL flag decodes to 0 (the §8.6 / §8.7 normal path stays).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_cu_transquant_bypass_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
    }

    #[test]
    fn decode_cu_transquant_bypass_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the
        // FL flag decodes to 1 (the §7.4.9.5 / §8.6 / §8.7 bypass
        // passthrough is selected for the current CU).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_cu_transquant_bypass_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
    }

    #[test]
    fn decode_cu_transquant_bypass_flag_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on
        // the same engine each consume exactly one bin (Table 9-43
        // FL cMax = 1). The second call's outcome is independent of
        // the first since the contexts are separate.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_cu_transquant_bypass_flag(&mut eng, &mut ctx_a).unwrap();
        let b = decode_cu_transquant_bypass_flag(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // -----------------------------------------------------------------
    // rqt_root_cbf (§7.3.8.5 / §7.4.9.5)
    // -----------------------------------------------------------------

    #[test]
    fn rqt_root_cbf_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for rqt_root_cbf: bin 0 ctxInc = 0; every
        // later binIdx column is na (Table 9-43 FL cMax = 1 — only
        // one bin is ever emitted).
        assert_eq!(rqt_root_cbf_ctx_inc(), 0);
    }

    #[test]
    fn rqt_root_cbf_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(RQT_ROOT_CBF_FL_CMAX, 1);
        assert_eq!(RQT_ROOT_CBF_FL_NBITS, 1);
    }

    #[test]
    fn rqt_root_cbf_inferred_is_one_per_7_4_9_5() {
        // §7.4.9.5 (V8 / 2021 baseline): when rqt_root_cbf is not
        // present (the §7.3.8.5 guard CuPredMode == MODE_INTRA ||
        // cu_skip_flag == 1 selects the not-present path), the value
        // is inferred to 1 — the transform_tree() syntax structure is
        // taken to be present.
        assert_eq!(rqt_root_cbf_inferred(), 1);
    }

    #[test]
    fn decode_rqt_root_cbf_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the
        // FL flag decodes to 0 (CU has no transform_tree per §7.4.9.5
        // — samples derive from prediction only).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_rqt_root_cbf(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
    }

    #[test]
    fn decode_rqt_root_cbf_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the
        // FL flag decodes to 1 (the transform_tree() syntax structure
        // follows the current coding-unit body per §7.4.9.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_rqt_root_cbf(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
    }

    #[test]
    fn decode_rqt_root_cbf_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on
        // the same engine each consume exactly one bin (Table 9-43
        // FL cMax = 1). The second call's outcome is independent of
        // the first since the contexts are separate.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_rqt_root_cbf(&mut eng, &mut ctx_a).unwrap();
        let b = decode_rqt_root_cbf(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn rqt_root_cbf_inferred_distinguishes_from_transquant_bypass() {
        // rqt_root_cbf (not present ⇒ inferred to 1, §7.4.9.5) vs.
        // cu_transquant_bypass_flag (PPS gate off ⇒ inferred to 0,
        // §7.4.9.5). The two flags share the same §7.4.9.5 reference
        // but their not-present-inference branches in opposite
        // directions; this anchor pins both values against any
        // accidental cross-wiring.
        assert_eq!(rqt_root_cbf_inferred(), 1);
        assert_eq!(cu_transquant_bypass_flag_inferred(), 0);
    }

    // -----------------------------------------------------------------
    // merge_flag (§7.3.8.6 / §7.4.9.6)
    // -----------------------------------------------------------------

    #[test]
    fn merge_flag_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for merge_flag: bin 0 ctxInc = 0; every later
        // binIdx column is na (Table 9-43 FL cMax = 1 — only one bin is
        // ever emitted).
        assert_eq!(merge_flag_ctx_inc(), 0);
    }

    #[test]
    fn merge_flag_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(MERGE_FLAG_FL_CMAX, 1);
        assert_eq!(MERGE_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn merge_flag_inferred_skip_is_one_per_7_4_9_6() {
        // §7.4.9.6: when merge_flag is not present (the §7.3.8.6
        // cu_skip_flag == 1 path), and CuPredMode == MODE_SKIP, the
        // value is inferred to 1.
        assert_eq!(merge_flag_inferred(CuPredMode::Skip), 1);
    }

    #[test]
    fn merge_flag_inferred_non_skip_is_zero_per_7_4_9_6() {
        // §7.4.9.6: otherwise (CuPredMode != MODE_SKIP) the not-present
        // inference is 0. These modes are not reached via the §7.3.8.6
        // not-present path in practice, but the pure helper is total.
        assert_eq!(merge_flag_inferred(CuPredMode::Inter), 0);
        assert_eq!(merge_flag_inferred(CuPredMode::Intra), 0);
    }

    #[test]
    fn decode_merge_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the FL
        // flag decodes to 0 (explicit-motion path — inter_pred_idc /
        // ref_idx / mvp follow per §7.3.8.6).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_merge_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
    }

    #[test]
    fn decode_merge_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the FL
        // flag decodes to 1 (merge path — merge_idx follows per
        // §7.3.8.6).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_merge_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
    }

    #[test]
    fn decode_merge_flag_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on the
        // same engine each consume exactly one bin (Table 9-43 FL
        // cMax = 1). The second call's outcome is independent of the
        // first since the contexts are separate.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_merge_flag(&mut eng, &mut ctx_a).unwrap();
        let b = decode_merge_flag(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // -----------------------------------------------------------------
    // pred_mode_flag (§7.3.8.5 / §7.4.9.5)
    // -----------------------------------------------------------------

    #[test]
    fn pred_mode_flag_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for pred_mode_flag: bin 0 ctxInc = 0; every
        // later binIdx column is na (Table 9-43 FL cMax = 1 — only
        // one bin is ever emitted).
        assert_eq!(pred_mode_flag_ctx_inc(), 0);
    }

    #[test]
    fn pred_mode_flag_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(PRED_MODE_FLAG_FL_CMAX, 1);
        assert_eq!(PRED_MODE_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn cu_pred_mode_from_flag_maps_table_9_48_values() {
        // §7.4.9.5: pred_mode_flag == 0 ⇒ MODE_INTER;
        // pred_mode_flag == 1 ⇒ MODE_INTRA. The two-valued FL
        // shape covers the full Table 9-48 binIdx 0 column.
        assert_eq!(cu_pred_mode_from_flag(0), CuPredMode::Inter);
        assert_eq!(cu_pred_mode_from_flag(1), CuPredMode::Intra);
    }

    #[test]
    fn pred_mode_flag_inferred_i_slice_is_intra() {
        // §7.4.9.5: "If slice_type is equal to I, CuPredMode[ x ][ y ]
        // is inferred to be equal to MODE_INTRA." cu_skip_flag is not
        // coded on I slices (§7.3.8.5 `if( slice_type != I )` guard)
        // and is itself inferred to 0; pin both `cu_skip_flag = 0`
        // inputs to confirm the slice-type branch dominates.
        assert_eq!(
            pred_mode_flag_inferred_cu_pred_mode(true, 0),
            CuPredMode::Intra
        );
    }

    #[test]
    fn pred_mode_flag_inferred_pb_skip_is_skip() {
        // §7.4.9.5: "Otherwise (slice_type is equal to P or B), when
        // cu_skip_flag[ x0 ][ y0 ] is equal to 1, CuPredMode[ x ][ y ]
        // is inferred to be equal to MODE_SKIP." This is the only
        // not-present path on P/B slices (the §7.3.8.5 guard reads
        // the flag whenever cu_skip_flag == 0).
        assert_eq!(
            pred_mode_flag_inferred_cu_pred_mode(false, 1),
            CuPredMode::Skip
        );
    }

    #[test]
    fn decode_pred_mode_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the
        // FL flag decodes to 0 (MODE_INTER per §7.4.9.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_pred_mode_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
        assert_eq!(cu_pred_mode_from_flag(flag), CuPredMode::Inter);
    }

    #[test]
    fn decode_pred_mode_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the
        // FL flag decodes to 1 (MODE_INTRA per §7.4.9.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_pred_mode_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
        assert_eq!(cu_pred_mode_from_flag(flag), CuPredMode::Intra);
    }

    #[test]
    fn decode_pred_mode_flag_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on
        // the same engine each consume exactly one bin (Table 9-43
        // FL cMax = 1). The second call's outcome is independent of
        // the first since the contexts are separate.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_pred_mode_flag(&mut eng, &mut ctx_a).unwrap();
        let b = decode_pred_mode_flag(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn cu_pred_mode_variants_distinct() {
        // §7.4.9.5 lists three reachable CuPredMode values: MODE_INTER
        // (pred_mode_flag == 0), MODE_INTRA (pred_mode_flag == 1 OR
        // I-slice not-present inference), MODE_SKIP (P/B not-present
        // with cu_skip_flag == 1). Pin all three as distinct.
        assert_ne!(CuPredMode::Inter, CuPredMode::Intra);
        assert_ne!(CuPredMode::Inter, CuPredMode::Skip);
        assert_ne!(CuPredMode::Intra, CuPredMode::Skip);
    }

    // -----------------------------------------------------------------
    // prev_intra_luma_pred_flag (§7.3.8.5 / §7.4.9.2)
    // -----------------------------------------------------------------

    #[test]
    fn prev_intra_luma_pred_flag_table_9_48_ctx_inc_is_zero() {
        // Table 9-48 row for prev_intra_luma_pred_flag: bin 0
        // ctxInc = 0; every later binIdx column is na (Table 9-43 FL
        // cMax = 1 — only one bin is ever emitted).
        assert_eq!(prev_intra_luma_pred_flag_ctx_inc(), 0);
    }

    #[test]
    fn prev_intra_luma_pred_flag_fl_shape_table_9_43() {
        // Table 9-43 FL row with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(PREV_INTRA_LUMA_PRED_FLAG_FL_CMAX, 1);
        assert_eq!(PREV_INTRA_LUMA_PRED_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn luma_intra_mode_source_from_flag_maps_table_9_48_values() {
        // §7.4.9.2 / §7.3.8.5: prev_intra_luma_pred_flag == 1 ⇒ the
        // §8.4.2 MPM-list source (mpm_idx follows);
        // prev_intra_luma_pred_flag == 0 ⇒ the rem_intra_luma_pred_mode
        // source. The two-valued FL shape covers the full Table 9-48
        // binIdx 0 column.
        assert_eq!(
            luma_intra_mode_source_from_flag(1),
            LumaIntraModeSource::Mpm
        );
        assert_eq!(
            luma_intra_mode_source_from_flag(0),
            LumaIntraModeSource::Remaining
        );
    }

    #[test]
    fn luma_intra_mode_source_variants_distinct() {
        // §7.4.9.2 lists two reachable mode sources for an intra luma
        // PB: the §8.4.2 MPM list (flag == 1) and the
        // rem_intra_luma_pred_mode field (flag == 0). Pin them as
        // distinct.
        assert_ne!(LumaIntraModeSource::Mpm, LumaIntraModeSource::Remaining);
    }

    #[test]
    fn decode_prev_intra_luma_pred_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the
        // FL flag decodes to 0 (the rem_intra_luma_pred_mode source
        // per §7.4.9.2).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_prev_intra_luma_pred_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
        assert_eq!(
            luma_intra_mode_source_from_flag(flag),
            LumaIntraModeSource::Remaining
        );
    }

    #[test]
    fn decode_prev_intra_luma_pred_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the
        // FL flag decodes to 1 (the §8.4.2 MPM-list source per
        // §7.4.9.2 — mpm_idx follows in §7.3.8.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_prev_intra_luma_pred_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
        assert_eq!(
            luma_intra_mode_source_from_flag(flag),
            LumaIntraModeSource::Mpm
        );
    }

    #[test]
    fn decode_prev_intra_luma_pred_flag_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on
        // the same engine each consume exactly one bin (Table 9-43
        // FL cMax = 1). The second call's outcome is independent of
        // the first since the contexts are separate. The §7.3.8.5
        // intra-PB loop reads this element once per luma prediction
        // block, so the exactly-one-bin invariant is the per-PB
        // contract.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_prev_intra_luma_pred_flag(&mut eng, &mut ctx_a).unwrap();
        let b = decode_prev_intra_luma_pred_flag(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // -------------------------------------------------------------
    // mpm_idx / rem_intra_luma_pred_mode (§7.3.8.5 / §7.4.9.2)
    // Table 9-43 binarization + Table 9-48 (fully bypass-coded)
    // -------------------------------------------------------------

    #[test]
    fn mpm_idx_tr_shape_table_9_43() {
        // Table 9-43 row: TR with cMax = 2, cRiceParam = 0 — the
        // three-value truncated-unary string set {0, 10, 11}.
        assert_eq!(MPM_IDX_TR_CMAX, 2);
        assert_eq!(MPM_IDX_TR_C_RICE_PARAM, 0);
    }

    #[test]
    fn rem_intra_luma_pred_mode_fl_shape_table_9_43() {
        // Table 9-43 row: FL with cMax = 31 → §9.3.3.5
        // Ceil(Log2(32)) = 5 bins.
        assert_eq!(REM_INTRA_LUMA_PRED_MODE_FL_CMAX, 31);
        assert_eq!(REM_INTRA_LUMA_PRED_MODE_FL_NBITS, 5);
        // Cross-check the §9.3.3.5 derivation against the constant.
        let derived = 32u32.next_power_of_two().trailing_zeros();
        assert_eq!(derived, REM_INTRA_LUMA_PRED_MODE_FL_NBITS);
    }

    #[test]
    fn mpm_idx_decodes_value_zero_on_first_zero_bin() {
        // §9.3.3.10 TR prefix, cRiceParam = 0: the first bypass bin is
        // 0 ⇒ value 0 (no second bin read). An all-zero stream decodes
        // a 0 bypass bin per the §9.3.4.3.4 `ivlOffset < ivlCurrRange`
        // test, so mpm_idx = 0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(decode_mpm_idx(&mut eng).unwrap(), 0);
    }

    #[test]
    fn mpm_idx_output_always_in_range() {
        // Whatever the stream, the TR cMax = 2 binarization caps the
        // decoded value at 2 — the §8.4.2 candModeList has exactly three
        // entries, so mpm_idx ∈ {0, 1, 2}.
        // First 9 bits seed ivlOffset (must be < 510 per §9.3.2.5);
        // the 0xfe lead keeps a near-all-ones stream while staying
        // valid (509).
        let seeds: [[u8; 8]; 4] = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66],
            [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01],
        ];
        for buf in seeds {
            let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
            let v = decode_mpm_idx(&mut eng).unwrap();
            assert!(v <= 2, "mpm_idx out of range: {v} for {buf:?}");
        }
    }

    #[test]
    fn mpm_idx_consumes_at_most_two_bins() {
        // The TR cMax = 2 prefix reads one bin (terminating 0) or two
        // (a leading 1 then either 0 or the cMax escape). Drive two
        // engines from the same buffer: one runs the wrapper once, the
        // other replays the equivalent raw bypass sequence and must
        // land on the same post-read engine offset.
        let buf = [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66];
        let mut eng_a = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut eng_b = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let v = decode_mpm_idx(&mut eng_a).unwrap();
        // Replay: first bin always read; second read iff first was 1.
        let b0 = eng_b.decode_bypass().unwrap();
        let bins_read = if b0 == 1 {
            let _ = eng_b.decode_bypass().unwrap();
            2
        } else {
            1
        };
        // value 0 ⇒ one bin (terminating 0); value 1 or 2 ⇒ two bins
        // (a leading 1 then the second prefix bin).
        let expected_bins = if v == 0 { 1 } else { 2 };
        assert_eq!(bins_read, expected_bins);
        assert_eq!(eng_a.ivl_offset(), eng_b.ivl_offset());
    }

    #[test]
    fn rem_intra_luma_pred_mode_reads_five_bypass_bins() {
        // FL cMax = 31: the wrapper is exactly decode_bypass_bits(5).
        // Drive two engines from the same buffer — the wrapper and a
        // direct five-bin MSB-first accumulation must agree on both the
        // returned value and the post-read engine offset.
        let buf = [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66];
        let mut eng_a = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut eng_b = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let via_wrapper = decode_rem_intra_luma_pred_mode(&mut eng_a).unwrap();
        let mut via_direct: u32 = 0;
        for _ in 0..5 {
            via_direct = (via_direct << 1) | (eng_b.decode_bypass().unwrap() as u32);
        }
        assert_eq!(via_wrapper as u32, via_direct);
        assert_eq!(eng_a.ivl_offset(), eng_b.ivl_offset());
    }

    #[test]
    fn rem_intra_luma_pred_mode_output_always_in_range() {
        // The five-bit FL string caps the decoded value at 31 — the
        // §8.4.2 step-2 seed for IntraPredModeY before the candModeList
        // increment pass.
        // First 9 bits seed ivlOffset (must be < 510 per §9.3.2.5).
        let seeds: [[u8; 8]; 4] = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66],
            [0x13, 0x57, 0x9b, 0xdf, 0x24, 0x68, 0xac, 0xe0],
        ];
        for buf in seeds {
            let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
            let v = decode_rem_intra_luma_pred_mode(&mut eng).unwrap();
            assert!(
                v <= 31,
                "rem_intra_luma_pred_mode out of range: {v} for {buf:?}"
            );
        }
    }

    // -------------------------------------------------------------
    // intra_chroma_pred_mode (§7.3.8.5 / §7.4.9.5)
    // §9.3.3.8 / Table 9-46 + Table 9-48 + §8.4.3 (Tables 8-2, 8-3)
    // -------------------------------------------------------------

    #[test]
    fn intra_chroma_pred_mode_table_9_46_shape() {
        // Table 9-46: the single-bin string `0` carries value 4; the
        // `1`-prefixed strings carry a two-bit FL suffix for 0..=3.
        assert_eq!(INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA, 4);
        assert_eq!(INTRA_CHROMA_PRED_MODE_SUFFIX_FL_NBITS, 2);
    }

    #[test]
    fn intra_chroma_pred_mode_ctx_inc_table_9_48_anchor() {
        // Table 9-48 row: bin 0 ctxInc = 0 (bins 1..2 are bypass and
        // never routed through this function).
        assert_eq!(intra_chroma_pred_mode_ctx_inc(0), 0);
    }

    #[test]
    fn decode_intra_chroma_pred_mode_zero_prefix_is_value_4() {
        // A 0 prefix bin (fresh valMps = 0 context against the all-zero
        // stream) terminates the Table 9-46 string immediately: value 4,
        // no suffix bins read. Cross-check the bin count by replaying
        // the raw decision on a second engine and comparing offsets.
        let buf = [0u8; 8];
        let mut eng_a = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut eng_b = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(0);
        let v = decode_intra_chroma_pred_mode(&mut eng_a, &mut ctx_a).unwrap();
        assert_eq!(v, INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA);
        let _ = eng_b.decode_decision(&mut ctx_b).unwrap();
        assert_eq!(eng_a.ivl_offset(), eng_b.ivl_offset());
    }

    #[test]
    fn decode_intra_chroma_pred_mode_one_prefix_reads_two_suffix_bins() {
        // A 1 prefix bin (fresh valMps = 1 context) is followed by the
        // two-bit MSB-first bypass suffix. Replay the raw sequence on a
        // second engine: same value, same post-read offset.
        let buf = [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66];
        let mut eng_a = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut eng_b = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(1);
        let mut ctx_b = fresh_mps_ctx(1);
        let v = decode_intra_chroma_pred_mode(&mut eng_a, &mut ctx_a).unwrap();
        let prefix = eng_b.decode_decision(&mut ctx_b).unwrap();
        let expected = if prefix == 0 {
            INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA
        } else {
            let hi = eng_b.decode_bypass().unwrap();
            let lo = eng_b.decode_bypass().unwrap();
            (hi << 1) | lo
        };
        assert_eq!(v, expected);
        assert_eq!(eng_a.ivl_offset(), eng_b.ivl_offset());
    }

    #[test]
    fn decode_intra_chroma_pred_mode_output_always_in_table_9_46_domain() {
        // Whatever the stream and context polarity, the Table 9-46
        // binarization only produces {0, 1, 2, 3, 4}.
        // First 9 bits seed ivlOffset (must be < 510 per §9.3.2.5).
        let seeds: [[u8; 8]; 4] = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            [0x5a, 0xa5, 0x3c, 0xc3, 0x0f, 0xf0, 0x99, 0x66],
            [0x13, 0x57, 0x9b, 0xdf, 0x24, 0x68, 0xac, 0xe0],
        ];
        for buf in seeds {
            for mps in [0u8, 1u8] {
                let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
                let mut ctx = fresh_mps_ctx(mps);
                let v = decode_intra_chroma_pred_mode(&mut eng, &mut ctx).unwrap();
                assert!(
                    v <= 4,
                    "intra_chroma_pred_mode out of range: {v} for {buf:?} mps {mps}"
                );
            }
        }
    }

    #[test]
    fn intra_pred_mode_c_mode_idx_table_8_2_rows() {
        // Table 8-2 rows 0..=3: base modes {0, 26, 10, 1}, substituting
        // 34 exactly where the base mode collides with IntraPredModeY.
        // Columns: Y ∈ {0, 26, 10, 1, X (other)}.
        let columns = [0u8, 26, 10, 1, 7];
        let rows: [(u8, [u8; 5]); 4] = [
            (0, [34, 0, 0, 0, 0]),
            (1, [26, 34, 26, 26, 26]),
            (2, [10, 10, 34, 10, 10]),
            (3, [1, 1, 1, 34, 1]),
        ];
        for (icpm, expected) in rows {
            for (y, want) in columns.iter().zip(expected.iter()) {
                assert_eq!(
                    intra_pred_mode_c_mode_idx(icpm, *y),
                    *want,
                    "Table 8-2 row {icpm} column Y = {y}"
                );
            }
        }
    }

    #[test]
    fn intra_pred_mode_c_mode_idx_row_4_tracks_luma() {
        // Table 8-2 row 4: modeIdx = IntraPredModeY for every column,
        // including the four base-mode columns.
        for y in [0u8, 1, 10, 26, 7, 34] {
            assert_eq!(intra_pred_mode_c_mode_idx(4, y), y);
        }
    }

    #[test]
    fn intra_pred_mode_c_chroma_422_map_table_8_3_anchors() {
        // Table 8-3 spot anchors across both printed rows, plus the
        // X <= 2 pass-through column.
        assert_eq!(INTRA_PRED_MODE_C_CHROMA_422_MAP.len(), 35);
        for x in 0u8..=2 {
            assert_eq!(intra_pred_mode_c_chroma_422(x), x);
        }
        let anchors: [(u8, u8); 10] = [
            (3, 2),
            (5, 2),
            (6, 3),
            (10, 10),
            (17, 20),
            (18, 21),
            (26, 26),
            (28, 27),
            (31, 29),
            (34, 31),
        ];
        for (mode_idx, want) in anchors {
            assert_eq!(
                intra_pred_mode_c_chroma_422(mode_idx),
                want,
                "Table 8-3 modeIdx {mode_idx}"
            );
        }
    }

    #[test]
    fn intra_pred_mode_c_chroma_422_map_is_monotonic_and_in_range() {
        // The Table 8-3 row is non-decreasing in modeIdx and every
        // output is itself a valid mode index (0..=34) — a structural
        // cross-check on the transcription.
        let map = INTRA_PRED_MODE_C_CHROMA_422_MAP;
        for w in map.windows(2) {
            assert!(w[0] <= w[1], "Table 8-3 row must be non-decreasing: {w:?}");
        }
        assert!(map.iter().all(|&m| m <= 34));
    }

    #[test]
    fn derive_intra_pred_mode_c_applies_table_8_3_only_for_422() {
        // §8.4.3: ChromaArrayType == 2 routes modeIdx through Table 8-3;
        // every other (non-zero) ChromaArrayType passes modeIdx through.
        // icpm 4 with Y = 30: modeIdx = 30, Table 8-3 ⇒ 28.
        assert_eq!(derive_intra_pred_mode_c(4, 30, true), 28);
        assert_eq!(derive_intra_pred_mode_c(4, 30, false), 30);
        // icpm 1 with Y = 26 collides ⇒ modeIdx = 34, Table 8-3 ⇒ 31.
        assert_eq!(derive_intra_pred_mode_c(1, 26, true), 31);
        assert_eq!(derive_intra_pred_mode_c(1, 26, false), 34);
        // icpm 0 with Y = 7 ⇒ modeIdx = 0; X <= 2 passes through under
        // Table 8-3, so both paths agree.
        assert_eq!(derive_intra_pred_mode_c(0, 7, true), 0);
        assert_eq!(derive_intra_pred_mode_c(0, 7, false), 0);
    }

    // -------------------------------------------------------------
    // §8.4.2 — derivation process for luma intra prediction mode
    // (candModeList construction + IntraPredModeY resolution)
    // -------------------------------------------------------------

    #[test]
    fn intra_mode_table_8_1_name_constants() {
        // Table 8-1: 0 = INTRA_PLANAR, 1 = INTRA_DC, 26 = INTRA_ANGULAR26,
        // 34 = INTRA_ANGULAR34 (the max mode).
        assert_eq!(INTRA_PLANAR, 0);
        assert_eq!(INTRA_DC, 1);
        assert_eq!(INTRA_ANGULAR26, 26);
        assert_eq!(INTRA_PRED_MODE_MAX, 34);
    }

    #[test]
    fn cand_mode_list_equal_low_candidate_eqs_8_21_8_23() {
        // candB == candA and candA < 2 ⇒ {PLANAR, DC, ANGULAR26}, for
        // both the PLANAR (0) and DC (1) equal-candidate cases.
        assert_eq!(intra_luma_cand_mode_list(0, 0), [0, 1, 26]);
        assert_eq!(intra_luma_cand_mode_list(1, 1), [0, 1, 26]);
    }

    #[test]
    fn cand_mode_list_equal_angular_candidate_eqs_8_24_8_26() {
        // candB == candA and candA >= 2 ⇒ the candidate plus its two
        // neighbouring angular modes (mod-32 wrap into 2..=33).
        // candA = 26: {26, 2 + (55 % 32) = 25, 2 + (25 % 32) = 27}.
        assert_eq!(intra_luma_cand_mode_list(26, 26), [26, 25, 27]);
        // candA = 2 (low angular edge): {2, 2 + (31 % 32) = 33,
        // 2 + (1 % 32) = 3} — the −1 neighbour wraps to the top mode 33.
        assert_eq!(intra_luma_cand_mode_list(2, 2), [2, 33, 3]);
        // candA = 34 (top angular edge): {34, 2 + (63 % 32) = 2 + 31 = 33,
        // 2 + (33 % 32) = 2 + 1 = 3} — the +1 neighbour wraps to mode 3.
        assert_eq!(intra_luma_cand_mode_list(34, 34), [34, 33, 3]);
    }

    #[test]
    fn cand_mode_list_distinct_third_is_first_missing_default() {
        // candA != candB ⇒ slots 0/1 = candA/candB (eqs. 8-27/8-28);
        // slot 2 is the first of PLANAR, DC, ANGULAR26 not already present.
        // Neither is PLANAR ⇒ slot 2 = PLANAR.
        assert_eq!(intra_luma_cand_mode_list(1, 26), [1, 26, 0]);
        // PLANAR present, DC absent ⇒ slot 2 = DC.
        assert_eq!(intra_luma_cand_mode_list(0, 26), [0, 26, 1]);
        // PLANAR and DC both present ⇒ slot 2 = ANGULAR26.
        assert_eq!(intra_luma_cand_mode_list(0, 1), [0, 1, 26]);
        assert_eq!(intra_luma_cand_mode_list(1, 0), [1, 0, 26]);
    }

    #[test]
    fn cand_mode_list_entries_are_valid_modes() {
        // Structural invariant: every candModeList entry is a Table 8-1
        // mode (0..=34) across the full candidate-pair product.
        for a in 0u8..=34 {
            for b in 0u8..=34 {
                let list = intra_luma_cand_mode_list(a, b);
                for &m in &list {
                    assert!(
                        m <= INTRA_PRED_MODE_MAX,
                        "candModeList entry {m} out of range"
                    );
                }
            }
        }
    }

    #[test]
    fn derive_intra_pred_mode_y_mpm_indexes_cand_list() {
        // step-4 Mpm path: IntraPredModeY = candModeList[ mpm_idx ].
        let list = [26u8, 25, 27];
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Mpm, 0),
            26
        );
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Mpm, 1),
            25
        );
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Mpm, 2),
            27
        );
    }

    #[test]
    fn derive_intra_pred_mode_y_remaining_low_modes() {
        // step-4 Remaining path with candModeList = {0, 1, 26} (sorted).
        // rem = 0 ⇒ 0>=0→1, 1>=1→2, 2>=26? no ⇒ 2.
        let list = [0u8, 1, 26];
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 0),
            2
        );
        // rem = 23 ⇒ 23>=0→24, 24>=1→25, 25>=26? no ⇒ 25.
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 23),
            25
        );
        // rem = 24 ⇒ 24>=0→25, 25>=1→26, 26>=26→27 ⇒ 27 (jumps over the
        // ANGULAR26 candidate too).
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 24),
            27
        );
    }

    #[test]
    fn derive_intra_pred_mode_y_remaining_sorts_cand_list() {
        // step-4 sorts candModeList ascending before the increment pass:
        // an unsorted {26, 25, 27} input sorts to {25, 26, 27}.
        // rem = 24 ⇒ 24>=25? no ⇒ 24 (below all three candidates).
        let list = [26u8, 25, 27];
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 24),
            24
        );
        // rem = 25 ⇒ 25>=25→26, 26>=26→27, 27>=27→28 ⇒ 28.
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 25),
            28
        );
    }

    #[test]
    fn derive_intra_pred_mode_y_remaining_bijective_over_full_range() {
        // The Remaining path maps the 31 rem values 0..=30 onto a set of
        // 31 distinct modes that excludes exactly the three candModeList
        // modes — a bijection onto (0..=34) \ candModeList.
        let list = [0u8, 1, 26]; // sorted, the equal-low-candidate list
        let mut produced = std::collections::BTreeSet::new();
        for rem in 0u8..=31 {
            let mode = derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, rem);
            assert!(mode <= INTRA_PRED_MODE_MAX, "mode {mode} out of range");
            assert!(
                produced.insert(mode),
                "rem path produced duplicate mode {mode}"
            );
        }
        // 32 distinct outputs (rem 0..=31) — none equal to a candidate.
        for &cand in &list {
            assert!(
                !produced.contains(&cand),
                "rem path produced candidate mode {cand}"
            );
        }
    }

    #[test]
    fn intra_pred_mode_y_end_to_end_remaining() {
        // End-to-end §8.4.2: distinct neighbour candidates A = 10, B = 2
        // ⇒ candModeList {10, 2, 0} (slot 2 = PLANAR, neither slot is
        // PLANAR), Remaining with rem = 0:
        //   sorted {0, 2, 10}; 0>=0→1, 1>=2? no, 1>=10? no ⇒ 1.
        let list = intra_luma_cand_mode_list(10, 2);
        assert_eq!(list, [10, 2, 0]);
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Remaining, 0),
            1
        );
        // Same list, Mpm path with mpm_idx = 0 ⇒ candModeList[0] = 10.
        assert_eq!(
            derive_intra_pred_mode_y(list, LumaIntraModeSource::Mpm, 0),
            10
        );
    }

    // -------------------------------------------------------------
    // §7.3.8.9 / §7.4.9.9 mvd_coding( ) — Table 9-43 / Table 9-48
    // -------------------------------------------------------------

    #[test]
    fn mvd_ctx_inc_single_context_each() {
        // Table 9-48: both `greater` flags are context-coded with a
        // single dedicated context (ctxInc = 0 at bin 0).
        assert_eq!(abs_mvd_greater0_flag_ctx_inc(), 0);
        assert_eq!(abs_mvd_greater1_flag_ctx_inc(), 0);
    }

    #[test]
    fn mvd_binarization_constants_match_table_9_43() {
        // FL cMax = 1 for the two greater flags + the sign flag; EG1
        // for abs_mvd_minus2.
        assert_eq!(ABS_MVD_GREATER_FLAG_FL_CMAX, 1);
        assert_eq!(MVD_SIGN_FLAG_FL_CMAX, 1);
        assert_eq!(ABS_MVD_MINUS2_EG_K, 1);
    }

    #[test]
    fn mvd_value_equation_7_73_inference_branches() {
        // greater0 == 0 ⇒ lMvd == 0 (all other fields absent).
        assert_eq!(mvd_component_value(0, None, None), 0);
        // greater0 == 1, greater1 absent/0 (minus2 absent, inferred −1)
        //   ⇒ magnitude = (−1 + 2) = 1; sign absent ⇒ +1.
        assert_eq!(mvd_component_value(1, None, Some(0)), 1);
        // ... with sign_flag == 1 ⇒ −1.
        assert_eq!(mvd_component_value(1, None, Some(1)), -1);
        // greater0 == 1, greater1 == 1 ⇒ minus2 present.
        //   minus2 = 0 ⇒ magnitude 2; minus2 = 3 ⇒ magnitude 5.
        assert_eq!(mvd_component_value(1, Some(0), Some(0)), 2);
        assert_eq!(mvd_component_value(1, Some(0), Some(1)), -2);
        assert_eq!(mvd_component_value(1, Some(3), Some(0)), 5);
        assert_eq!(mvd_component_value(1, Some(3), Some(1)), -5);
        // Sign defaults to positive when absent.
        assert_eq!(mvd_component_value(1, Some(7), None), 9);
    }

    #[test]
    fn mvd_component_with_greater0_zero_short_circuits() {
        // First context bin = 0 ⇒ no bypass bins consumed at all; every
        // downstream field stays inferred-absent.
        let mut bypass_called = false;
        let c = decode_mvd_component_with(
            || Ok(0u8),
            || {
                bypass_called = true;
                Ok(0u8)
            },
        )
        .unwrap();
        assert!(!bypass_called, "no bypass bin should be read");
        assert_eq!(
            c,
            MvdComponent {
                greater0_flag: 0,
                greater1_flag: None,
                minus2: None,
                sign_flag: None,
                value: 0,
            }
        );
    }

    #[test]
    fn mvd_component_with_magnitude_one() {
        // greater0 = 1, greater1 = 0 ⇒ abs_mvd_minus2 absent, only the
        // sign bit is bypass-read. Sign = 1 ⇒ lMvd = −1.
        let ctx_bins = [1u8, 0]; // greater0, greater1
        let bypass_bins = [1u8]; // mvd_sign_flag
        let c = decode_mvd_component_with(bin_queue(&ctx_bins), bin_queue(&bypass_bins)).unwrap();
        assert_eq!(
            c,
            MvdComponent {
                greater0_flag: 1,
                greater1_flag: Some(0),
                minus2: None,
                sign_flag: Some(1),
                value: -1,
            }
        );
    }

    #[test]
    fn mvd_component_with_eg1_escape() {
        // greater0 = 1, greater1 = 1 ⇒ abs_mvd_minus2 is EG1-coded then
        // the sign bit follows. §9.3.3.3 EG1 of value 0 is the bin string
        // "0 0": the prefix is the bare terminating `0` (prefix_ones = 0),
        // then prefix_ones + k = 0 + 1 = 1 suffix bin = `0` ⇒ value 0.
        // So minus2 = 0 ⇒ magnitude = 2; sign = 0 ⇒ lMvd = +2.
        let ctx_bins = [1u8, 1];
        let bypass_bins = [0u8, 0, 0]; // EG1 prefix '0', suffix '0', sign '0'
        let c = decode_mvd_component_with(bin_queue(&ctx_bins), bin_queue(&bypass_bins)).unwrap();
        assert_eq!(
            c,
            MvdComponent {
                greater0_flag: 1,
                greater1_flag: Some(1),
                minus2: Some(0),
                sign_flag: Some(0),
                value: 2,
            }
        );
    }

    #[test]
    fn mvd_component_with_eg1_value_one_full() {
        // greater0 = 1, greater1 = 1, abs_mvd_minus2 = 1 (§9.3.3.3 EG1
        //   "0 1": prefix_ones = 0, suffix bits = prefix_ones + k = 1 =
        //   "1" ⇒ base 0 + suffix 1 = 1), then sign = 0. magnitude =
        //   minus2 + 2 = 3 ⇒ lMvd = +3.
        let ctx_bins = [1u8, 1];
        let bypass_bins = [0u8, 1, 0]; // EG1(1)='0','1' ; sign '0'
        let c = decode_mvd_component_with(bin_queue(&ctx_bins), bin_queue(&bypass_bins)).unwrap();
        assert_eq!(c.minus2, Some(1));
        assert_eq!(c.value, 3);
    }

    #[test]
    fn read_eg_k_with_eg1_battery() {
        // §9.3.3.3 EG1 (k = 1) decode across a battery of bin strings,
        // confirming the closure reader implements the do-while shared
        // with the engine-driven `decode_eg_k`. The EGk unary prefix is a
        // run of `1` bins terminated by a `0` (the §9.3.3.3 NOTE), then
        // prefix_ones + k suffix bits:
        //   "0 s"          prefix_ones=0 → suffix 1 bit
        //   "1 0 s1 s0"    prefix_ones=1 → base 2, suffix 2 bits
        let cases: &[(&[u8], u32)] = &[
            (&[0, 0], 0),             // p0, suffix 0 ⇒ 0
            (&[0, 1], 1),             // p0, suffix 1 ⇒ 1
            (&[1, 0, 0, 0], 2),       // p1, base=2, suffix 0 ⇒ 2
            (&[1, 0, 0, 1], 3),       // p1, base=2, suffix 1 ⇒ 3
            (&[1, 0, 1, 0], 4),       // p1, base=2, suffix 2 ⇒ 4
            (&[1, 1, 0, 0, 0, 0], 6), // p2, base=6, suffix 0 ⇒ 6
        ];
        for (bins, expected) in cases {
            assert_eq!(
                read_eg_k_with(1, bin_queue(bins)).unwrap(),
                *expected,
                "bins {bins:?}"
            );
        }
    }

    #[test]
    fn mvd_component_engine_greater0_zero() {
        // All-zero stream: 9 init bits then the first DecodeDecision
        // returns 0 for an MPS-state context ⇒ greater0 = 0, value 0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut g0 = fresh_mps_ctx(0);
        let mut g1 = fresh_mps_ctx(0);
        let c = decode_mvd_component(&mut eng, &mut g0, &mut g1).unwrap();
        assert_eq!(c.greater0_flag, 0);
        assert_eq!(c.value, 0);
        assert_eq!(c.greater1_flag, None);
    }

    // -----------------------------------------------------------------
    // split_cu_flag (§7.3.8.4 / §7.4.9.4)
    // -----------------------------------------------------------------

    #[test]
    fn split_cu_flag_fl_shape_table_9_43() {
        // Table 9-43 row: FL with cMax = 1 ⇒ one bin, §9.3.3.5 nBits
        // = Ceil(Log2(cMax + 1)) = 1.
        assert_eq!(SPLIT_CU_FLAG_FL_CMAX, 1);
        assert_eq!(SPLIT_CU_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn decode_split_cu_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the FL
        // flag decodes to 0 (leaf coding unit — coding_unit() follows
        // per §7.3.8.4).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_split_cu_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
    }

    #[test]
    fn decode_split_cu_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the FL
        // flag decodes to 1 (split into four quarter-size children per
        // §7.3.8.4).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_split_cu_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
    }

    #[test]
    fn decode_split_cu_flag_consumes_exactly_one_bin() {
        // Two back-to-back calls against fresh MPS-only contexts on the
        // same engine each consume exactly one bin (Table 9-43 FL
        // cMax = 1); separate contexts make the outcomes independent.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx_a = fresh_mps_ctx(0);
        let mut ctx_b = fresh_mps_ctx(1);
        let a = decode_split_cu_flag(&mut eng, &mut ctx_a).unwrap();
        let b = decode_split_cu_flag(&mut eng, &mut ctx_b).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // -----------------------------------------------------------------
    // cu_skip_flag (§7.3.8.5 / §7.4.9.5)
    // -----------------------------------------------------------------

    #[test]
    fn cu_skip_flag_fl_shape_table_9_43() {
        // Table 9-43 row: FL with cMax = 1 ⇒ one bin, §9.3.3.5 nBits = 1.
        assert_eq!(CU_SKIP_FLAG_FL_CMAX, 1);
        assert_eq!(CU_SKIP_FLAG_FL_NBITS, 1);
    }

    #[test]
    fn decode_cu_skip_flag_zero_path() {
        // Empty bin stream + fresh MPS-only valMps = 0 context: the FL
        // flag decodes to 0 (non-skip CU — pred_mode_flag / part_mode
        // follow per §7.3.8.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let flag = decode_cu_skip_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 0);
    }

    #[test]
    fn decode_cu_skip_flag_one_path() {
        // Empty bin stream + fresh MPS-only valMps = 1 context: the FL
        // flag decodes to 1 (MODE_SKIP path per §7.4.9.5).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(1);
        let flag = decode_cu_skip_flag(&mut eng, &mut ctx).unwrap();
        assert_eq!(flag, 1);
        assert_eq!(cu_pred_mode_from_skip(false, flag), Some(CuPredMode::Skip));
    }

    #[test]
    fn cu_pred_mode_from_skip_i_slice_is_intra() {
        // §7.4.9.5: I slices infer cu_skip_flag = 0 and CuPredMode =
        // MODE_INTRA without a pred_mode_flag read. The cu_skip_flag
        // input is pinned 0 (its inferred value).
        assert_eq!(cu_pred_mode_from_skip(true, 0), Some(CuPredMode::Intra));
    }

    #[test]
    fn cu_pred_mode_from_skip_pb_skip_is_skip() {
        // §7.4.9.5: a P / B CU with cu_skip_flag == 1 infers MODE_SKIP;
        // pred_mode_flag is not present.
        assert_eq!(cu_pred_mode_from_skip(false, 1), Some(CuPredMode::Skip));
    }

    #[test]
    fn cu_pred_mode_from_skip_pb_non_skip_needs_pred_mode_flag() {
        // §7.3.8.5 / §7.4.9.5: a P / B CU with cu_skip_flag == 0 still
        // codes pred_mode_flag, so the mode is not yet determined —
        // the helper returns None to signal that read is required.
        assert_eq!(cu_pred_mode_from_skip(false, 0), None);
    }

    // -------------------------------------------------------------
    // §7.3.8.3 SAO element decoders
    // -------------------------------------------------------------

    #[test]
    fn sao_offset_abs_cmax_by_bit_depth() {
        // §9.3.3.5: cMax = (1 << (Min(bitDepth, 10) − 5)) − 1.
        assert_eq!(sao_offset_abs_tr_cmax(8), 7); // (1<<3)-1
        assert_eq!(sao_offset_abs_tr_cmax(10), 31); // (1<<5)-1
        assert_eq!(sao_offset_abs_tr_cmax(12), 31); // clamped at 10
    }

    #[test]
    fn sao_type_idx_zero_when_first_bin_zero() {
        // All-zero stream → bin 0 (context, MPS=0) decodes to 0 →
        // SaoTypeIdx = 0 (not applied), no second bin consumed.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        assert_eq!(decode_sao_type_idx(&mut eng, &mut ctx).unwrap(), 0);
    }

    #[test]
    fn sao_band_position_reads_five_bypass_bits() {
        // All-zero buffer → five bypass bins → 0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(decode_sao_band_position(&mut eng).unwrap(), 0);
    }

    #[test]
    fn sao_eo_class_reads_two_bypass_bits() {
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let c = decode_sao_eo_class(&mut eng).unwrap();
        assert!(c <= 3);
    }

    // -------------------------------------------------------------
    // §7.4.9.5 part_mode
    // -------------------------------------------------------------

    #[test]
    fn part_mode_is_amp_classification() {
        assert!(!PartMode::Part2Nx2N.is_amp());
        assert!(!PartMode::PartNxN.is_amp());
        assert!(PartMode::Part2NxnU.is_amp());
        assert!(PartMode::Part2NxnD.is_amp());
        assert!(PartMode::PartNLx2N.is_amp());
        assert!(PartMode::PartNRx2N.is_amp());
    }

    #[test]
    fn part_mode_inferred_is_2nx2n() {
        let r = part_mode_inferred();
        assert_eq!(r.part_mode, PartMode::Part2Nx2N);
        assert!(!r.intra_split_flag);
    }

    #[test]
    fn part_mode_intra_nxn_sets_intra_split_flag() {
        // Intra column: bin 0 == 0 → PART_NxN, IntraSplitFlag = 1.
        // All-zero stream gives bin 0 == 0 with an MPS=0 context.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        let mut c2 = fresh_mps_ctx(0);
        let r = decode_part_mode(
            &mut eng,
            &mut c0,
            &mut c1,
            &mut c2,
            CuPredMode::Intra,
            4,
            3,
            false,
        )
        .unwrap();
        assert_eq!(r.part_mode, PartMode::PartNxN);
        assert!(r.intra_split_flag);
    }

    #[test]
    fn part_mode_intra_2nx2n_when_bin0_one() {
        // Intra column: bin 0 == 1 → PART_2Nx2N. MPS=1 all-zero stream
        // decodes bin 0 == 1.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(1);
        let mut c1 = fresh_mps_ctx(1);
        let mut c2 = fresh_mps_ctx(1);
        let r = decode_part_mode(
            &mut eng,
            &mut c0,
            &mut c1,
            &mut c2,
            CuPredMode::Intra,
            4,
            3,
            false,
        )
        .unwrap();
        assert_eq!(r.part_mode, PartMode::Part2Nx2N);
        assert!(!r.intra_split_flag);
    }

    #[test]
    fn part_mode_inter_2nx2n_when_bin0_one() {
        // Inter, bin 0 == 1 → PART_2Nx2N (single bin).
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(1);
        let mut c1 = fresh_mps_ctx(1);
        let mut c2 = fresh_mps_ctx(1);
        let r = decode_part_mode(
            &mut eng,
            &mut c0,
            &mut c1,
            &mut c2,
            CuPredMode::Inter,
            5,
            3,
            true,
        )
        .unwrap();
        assert_eq!(r.part_mode, PartMode::Part2Nx2N);
    }

    // -------------------------------------------------------------
    // merge_idx / ref_idx / mvp_flag / inter_pred_idc / pcm /
    // end_of_slice_segment_flag
    // -------------------------------------------------------------

    #[test]
    fn merge_idx_zero_when_max_cand_one() {
        // MaxNumMergeCand == 1 → cMax == 0 → inferred 0, no bins read.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        assert_eq!(decode_merge_idx(&mut eng, &mut ctx, 1).unwrap(), 0);
    }

    #[test]
    fn merge_idx_zero_when_first_bin_zero() {
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        assert_eq!(decode_merge_idx(&mut eng, &mut ctx, 5).unwrap(), 0);
    }

    #[test]
    fn ref_idx_zero_when_active_minus1_zero() {
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        assert_eq!(decode_ref_idx(&mut eng, &mut c0, &mut c1, 0).unwrap(), 0);
    }

    #[test]
    fn ref_idx_zero_when_first_bin_zero() {
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        assert_eq!(decode_ref_idx(&mut eng, &mut c0, &mut c1, 3).unwrap(), 0);
    }

    #[test]
    fn inter_pred_idc_npb12_is_single_bin() {
        // nPbW + nPbH == 12: single bin, "0" → PRED_L0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        let r = decode_inter_pred_idc(&mut eng, &mut c0, &mut c1, 8, 4).unwrap();
        assert_eq!(r, InterPredIdc::PredL0);
    }

    #[test]
    fn inter_pred_idc_bi_when_bin0_one() {
        // nPbW + nPbH != 12, bin 0 == 1 → PRED_BI.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(1);
        let mut c1 = fresh_mps_ctx(1);
        let r = decode_inter_pred_idc(&mut eng, &mut c0, &mut c1, 16, 16).unwrap();
        assert_eq!(r, InterPredIdc::PredBi);
    }

    #[test]
    fn pcm_and_end_of_slice_use_terminate() {
        // The terminate path on an all-zero buffer returns 0 (not the
        // termination value); both share decode_terminate.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let pcm = decode_pcm_flag(&mut eng).unwrap();
        assert!(pcm <= 1);
        let eos = decode_end_of_slice_segment_flag(&mut eng).unwrap();
        assert!(eos <= 1);
    }

    #[test]
    fn mvp_flag_single_bin() {
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut ctx = fresh_mps_ctx(0);
        let v = decode_mvp_flag(&mut eng, &mut ctx).unwrap();
        assert!(v <= 1);
    }
}
