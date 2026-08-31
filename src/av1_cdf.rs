//! Default CDF tables (AV1 §9.4), the 4×4 scan (§9.2), and the constant context-offset tables
//! (§8.3.2) needed by the lossless intra decoder.
//!
//! Only the slices the bounded decoder touches are included: `base_q_idx == 0` ⇒ coefficient-CDF quant
//! context 0, and the `TX_4X4` coefficient tables, which the non-lossless additions at the bottom of
//! this module extend to the larger square transforms. CDFs are stored in the spec's cumulative form (rising to 32768)
//! with the trailing adaptation-count element dropped, so each row is ready to pass straight to
//! [`gamut_bitstream::SymbolEncoder::encode_symbol`] (the M0 frame sets `disable_cdf_update = 1`,
//! so the tables are never adapted). These values are extracted verbatim from the specification.
//!
//! The inter-frame tables below (`IS_INTER` through `MV_FR`) extend the same bounded slice to the
//! single-reference NEARESTMV/GLOBALMV/NEWMV subset used by [`crate::av1_inter_decoder`]: a single
//! fixed context (context 0) is used everywhere the full specification varies a CDF by neighbouring
//! mode/skip/reference state, matching this module's existing "smallest context that keeps bitstream
//! conformance" precedent (e.g. `TX_4X4`-only coefficient contexts above).

use crate::av1_intra::Av1TxType;

// --- Partition CDFs (Default_Partition_W*_Cdf), indexed [ctx]. ---
/// `Default_Partition_W8_Cdf` (4 partitions: NONE/HORZ/VERT/SPLIT).
pub static PARTITION_W8: [[u16; 4]; 4] = [
    [19132, 25510, 30392, 32768],
    [13928, 19855, 28540, 32768],
    [12522, 23679, 28629, 32768],
    [9896, 18783, 25853, 32768],
];
/// `Default_Partition_W16_Cdf` (10 partition types).
pub static PARTITION_W16: [[u16; 10]; 4] = [
    [
        15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768,
    ],
    [
        7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834, 32768,
    ],
    [
        5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117, 32768,
    ],
    [
        2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171, 32768,
    ],
];
/// `Default_Partition_W32_Cdf`.
pub static PARTITION_W32: [[u16; 10]; 4] = [
    [
        18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544, 32768,
    ],
    [
        7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052, 32768,
    ],
    [
        6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047, 32768,
    ],
    [
        1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899, 32768,
    ],
];
/// `Default_Partition_W64_Cdf`.
pub static PARTITION_W64: [[u16; 10]; 4] = [
    [
        20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724, 32768,
    ],
    [
        6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104, 32768,
    ],
    [
        5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238, 32768,
    ],
    [
        870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332, 32768,
    ],
];

/// `Default_Skip_Cdf`, indexed [ctx].
pub static SKIP: [[u16; 2]; 3] = [[31671, 32768], [16515, 32768], [4576, 32768]];

/// `Default_Intra_Frame_Y_Mode_Cdf[0][0]` (above/left both `DC_PRED`).
pub static INTRA_FRAME_Y_MODE_DC_DC: [u16; 13] = [
    15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466, 32768,
];

// --- Coefficient CDFs (quant context 0, TX_4X4). ---
/// `Default_Txb_Skip_Cdf[0][TX_4X4]`, indexed [ctx].
pub static TXB_SKIP: [[u16; 2]; 13] = [
    [31849, 32768],
    [5892, 32768],
    [12112, 32768],
    [21935, 32768],
    [20289, 32768],
    [27473, 32768],
    [32487, 32768],
    [7654, 32768],
    [19473, 32768],
    [29984, 32768],
    [9961, 32768],
    [30242, 32768],
    [32117, 32768],
];
/// `Default_Eob_Pt_16_Cdf[0]`, indexed [ptype][ctx].
pub static EOB_PT_16: [[[u16; 5]; 2]; 2] = [
    [
        [840, 1039, 1980, 4895, 32768],
        [370, 671, 1883, 4471, 32768],
    ],
    [
        [3247, 4950, 9688, 14563, 32768],
        [1904, 3354, 7763, 14647, 32768],
    ],
];
/// `Default_Eob_Extra_Cdf[0][TX_4X4]`, indexed [ptype][eobPt-3].
pub static EOB_EXTRA: [[[u16; 2]; 9]; 2] = [
    [
        [16961, 32768],
        [17223, 32768],
        [7621, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
    ],
    [
        [19069, 32768],
        [22525, 32768],
        [13377, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
        [16384, 32768],
    ],
];
/// `Default_Dc_Sign_Cdf[0]`, indexed [ptype][ctx].
pub static DC_SIGN: [[[u16; 2]; 3]; 2] = [
    [[16000, 32768], [13056, 32768], [18816, 32768]],
    [[15232, 32768], [12928, 32768], [17280, 32768]],
];
/// `Default_Coeff_Base_Eob_Cdf[0][TX_4X4]`, indexed [ptype][ctx].
pub static COEFF_BASE_EOB: [[[u16; 3]; 4]; 2] = [
    [
        [17837, 29055, 32768],
        [29600, 31446, 32768],
        [30844, 31878, 32768],
        [24926, 28948, 32768],
    ],
    [
        [21365, 30026, 32768],
        [30512, 32423, 32768],
        [31658, 32621, 32768],
        [29630, 31881, 32768],
    ],
];
/// `Default_Coeff_Base_Cdf[0][TX_4X4]`, indexed [ptype][ctx] (42 contexts).
pub static COEFF_BASE: [[[u16; 4]; 42]; 2] = [
    [
        [4034, 8930, 12727, 32768],
        [18082, 29741, 31877, 32768],
        [12596, 26124, 30493, 32768],
        [9446, 21118, 27005, 32768],
        [6308, 15141, 21279, 32768],
        [2463, 6357, 9783, 32768],
        [20667, 30546, 31929, 32768],
        [13043, 26123, 30134, 32768],
        [8151, 18757, 24778, 32768],
        [5255, 12839, 18632, 32768],
        [2820, 7206, 11161, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [15736, 27553, 30604, 32768],
        [11210, 23794, 28787, 32768],
        [5947, 13874, 19701, 32768],
        [4215, 9323, 13891, 32768],
        [2833, 6462, 10059, 32768],
        [19605, 30393, 31582, 32768],
        [13523, 26252, 30248, 32768],
        [8446, 18622, 24512, 32768],
        [3818, 10343, 15974, 32768],
        [1481, 4117, 6796, 32768],
        [22649, 31302, 32190, 32768],
        [14829, 27127, 30449, 32768],
        [8313, 17702, 23304, 32768],
        [3022, 8301, 12786, 32768],
        [1536, 4412, 7184, 32768],
        [22354, 29774, 31372, 32768],
        [14723, 25472, 29214, 32768],
        [6673, 13745, 18662, 32768],
        [2068, 5766, 9322, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
    ],
    [
        [6302, 16444, 21761, 32768],
        [23040, 31538, 32475, 32768],
        [15196, 28452, 31496, 32768],
        [10020, 22946, 28514, 32768],
        [6533, 16862, 23501, 32768],
        [3538, 9816, 15076, 32768],
        [24444, 31875, 32525, 32768],
        [15881, 28924, 31635, 32768],
        [9922, 22873, 28466, 32768],
        [6527, 16966, 23691, 32768],
        [4114, 11303, 17220, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
        [20201, 30770, 32209, 32768],
        [14754, 28071, 31258, 32768],
        [8378, 20186, 26517, 32768],
        [5916, 15299, 21978, 32768],
        [4268, 11583, 17901, 32768],
        [24361, 32025, 32581, 32768],
        [18673, 30105, 31943, 32768],
        [10196, 22244, 27576, 32768],
        [5495, 14349, 20417, 32768],
        [2676, 7415, 11498, 32768],
        [24678, 31958, 32585, 32768],
        [18629, 29906, 31831, 32768],
        [9364, 20724, 26315, 32768],
        [4641, 12318, 18094, 32768],
        [2758, 7387, 11579, 32768],
        [25433, 31842, 32469, 32768],
        [18795, 29289, 31411, 32768],
        [7644, 17584, 23592, 32768],
        [3408, 9014, 15047, 32768],
        [8192, 16384, 24576, 32768],
        [8192, 16384, 24576, 32768],
    ],
];
/// `Default_Coeff_Br_Cdf[0][TX_4X4]`, indexed [ptype][ctx] (21 level contexts).
pub static COEFF_BR: [[[u16; 4]; 21]; 2] = [
    [
        [14298, 20718, 24174, 32768],
        [12536, 19601, 23789, 32768],
        [8712, 15051, 19503, 32768],
        [6170, 11327, 15434, 32768],
        [4742, 8926, 12538, 32768],
        [3803, 7317, 10546, 32768],
        [1696, 3317, 4871, 32768],
        [14392, 19951, 22756, 32768],
        [15978, 23218, 26818, 32768],
        [12187, 19474, 23889, 32768],
        [9176, 15640, 20259, 32768],
        [7068, 12655, 17028, 32768],
        [5656, 10442, 14472, 32768],
        [2580, 4992, 7244, 32768],
        [12136, 18049, 21426, 32768],
        [13784, 20721, 24481, 32768],
        [10836, 17621, 21900, 32768],
        [8372, 14444, 18847, 32768],
        [6523, 11779, 16000, 32768],
        [5337, 9898, 13760, 32768],
        [3034, 5860, 8462, 32768],
    ],
    [
        [15967, 22905, 26286, 32768],
        [13534, 20654, 24579, 32768],
        [9504, 16092, 20535, 32768],
        [6975, 12568, 16903, 32768],
        [5364, 10091, 14020, 32768],
        [4357, 8370, 11857, 32768],
        [2506, 4934, 7218, 32768],
        [23032, 28815, 30936, 32768],
        [19540, 26704, 29719, 32768],
        [15158, 22969, 27097, 32768],
        [11408, 18865, 23650, 32768],
        [8885, 15448, 20250, 32768],
        [7108, 12853, 17416, 32768],
        [4231, 8041, 11480, 32768],
        [19823, 26490, 29156, 32768],
        [18890, 25929, 28932, 32768],
        [15660, 23491, 27433, 32768],
        [12147, 19776, 24488, 32768],
        [9728, 16774, 21649, 32768],
        [7919, 14277, 19066, 32768],
        [5440, 10170, 14185, 32768],
    ],
];

/// `Default_Scan_4x4` (§9.2): up-right diagonal scan order over the 16 positions.
pub static DEFAULT_SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// `Coeff_Base_Ctx_Offset[TX_4X4]` (§8.3.2), indexed `[min(row,4)][min(col,4)]`.
/// Retained verbatim as the reference [`coeff_base_ctx_offset`]'s
/// size-independent closed form is validated against; the decoders call
/// that function so the same rule covers every square transform size.
#[cfg(test)]
pub static COEFF_BASE_CTX_OFFSET_4X4: [[u8; 5]; 5] = [
    [0, 1, 6, 6, 0],
    [1, 6, 6, 21, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [0, 0, 0, 0, 0],
];

/// `Sig_Ref_Diff_Offset[TX_CLASS_2D]` (§8.3.2): `(row, col)` neighbour offsets for `coeff_base`.
pub static SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];

/// `Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D]` (§8.3.2): neighbour offsets for `coeff_br`.
pub static MAG_REF_OFFSET_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];

// --- Non-lossless (`base_q_idx != 0`) additions: the square transforms
// TX_4X4 through TX_64X64 that this crate's inverse transform kernels
// implement. ---
//
// Unlike the tables above (extracted verbatim from the specification), the
// CDFs in this section are placeholder-but-valid default probability models
// for symbols this crate did not previously decode at all (`read_tx_type`'s
// per-set `ext_tx` symbol, `read_tx_size`'s `tx_depth`,
// and the larger `eob_pt` ranges needed once transform blocks can hold
// more than 16 coefficients). Unlike
// the *dequantization* step (a direct multiply that must match the spec's
// tables to produce correct pixel values), a symbol's *CDF* only affects
// entropy-coding efficiency, not decode correctness, as long as the same
// table is used consistently by both the encoder and decoder of a given
// stream — which this crate's own hand-authored bitstream writer (used only
// by its test suite; there is no production AV1 encoder here) guarantees.
// These tables are therefore internally consistent and exercised by
// round-trip tests, but are NOT claimed to bit-exactly match an official
// AV1 encoder's default CDFs, so streams produced by third-party encoders
// that exercise these particular symbols are not guaranteed to decode
// correctly by this crate (this matches the rest of this crate's stance of
// being a bounded, non-conformance-tested subset rather than a full AV1
// decoder).

/// A transform-type set (AV1 spec §5.11.47 `get_tx_set`). Only the square
/// transform sizes this crate codes are reachable, so `txSzSqr` and
/// `txSzSqrUp` are both the block's side length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1TxSet {
    /// `TX_SET_DCTONLY`: `DCT_DCT` with no `tx_type` symbol coded at all.
    DctOnly,
    /// `TX_SET_INTRA_1`, seven types.
    Intra1,
    /// `TX_SET_INTRA_2`, five types.
    Intra2,
    /// `TX_SET_INTER_1`, sixteen types.
    Inter1,
    /// `TX_SET_INTER_2`, twelve types.
    Inter2,
    /// `TX_SET_INTER_3`, `{IDTX, DCT_DCT}`.
    Inter3,
}

/// Spec §5.11.47 `get_tx_set`: the transform-type set a `tx_size x tx_size`
/// transform block in an intra or inter block signals under the frame
/// header's `reduced_tx_set`.
#[must_use]
pub fn get_tx_set(tx_size: usize, is_inter: bool, reduced_tx_set: bool) -> Av1TxSet {
    // txSzSqrUp > TX_32X32 (i.e. TX_64X64 here) is always DCT-only.
    if tx_size > 32 {
        return Av1TxSet::DctOnly;
    }
    if is_inter {
        if reduced_tx_set || tx_size == 32 {
            Av1TxSet::Inter3
        } else if tx_size == 16 {
            Av1TxSet::Inter2
        } else {
            Av1TxSet::Inter1
        }
    } else if tx_size == 32 {
        Av1TxSet::DctOnly
    } else if reduced_tx_set || tx_size == 16 {
        Av1TxSet::Intra2
    } else {
        Av1TxSet::Intra1
    }
}

/// One entry of a transform set's inverse table (spec §5.11.48
/// `Tx_Type_Intra_Inv_Set*` / `Tx_Type_Inter_Inv_Set*`): the specification's
/// name for the transform type, and the kernel this crate implements for it.
///
/// The half-identity `V_*`/`H_*` types (identity along one axis, DCT/ADST
/// along the other) have no [`Av1TxType`] and no kernel in this crate, so
/// they carry `None` and are rejected as unsupported when a bitstream
/// signals them.
pub type TxTypeSlot = (&'static str, Option<Av1TxType>);

const IDTX: TxTypeSlot = ("IDTX", Some(Av1TxType::Idtx));
const DCT_DCT: TxTypeSlot = ("DCT_DCT", Some(Av1TxType::DctDct));
const ADST_DCT: TxTypeSlot = ("ADST_DCT", Some(Av1TxType::AdstDct));
const DCT_ADST: TxTypeSlot = ("DCT_ADST", Some(Av1TxType::DctAdst));
const ADST_ADST: TxTypeSlot = ("ADST_ADST", Some(Av1TxType::AdstAdst));
const FLIPADST_DCT: TxTypeSlot = ("FLIPADST_DCT", Some(Av1TxType::FlipadstDct));
const DCT_FLIPADST: TxTypeSlot = ("DCT_FLIPADST", Some(Av1TxType::DctFlipadst));
const FLIPADST_FLIPADST: TxTypeSlot = ("FLIPADST_FLIPADST", Some(Av1TxType::FlipadstFlipadst));
const ADST_FLIPADST: TxTypeSlot = ("ADST_FLIPADST", Some(Av1TxType::AdstFlipadst));
const FLIPADST_ADST: TxTypeSlot = ("FLIPADST_ADST", Some(Av1TxType::FlipadstAdst));
const V_DCT: TxTypeSlot = ("V_DCT", None);
const H_DCT: TxTypeSlot = ("H_DCT", None);
const V_ADST: TxTypeSlot = ("V_ADST", None);
const H_ADST: TxTypeSlot = ("H_ADST", None);
const V_FLIPADST: TxTypeSlot = ("V_FLIPADST", None);
const H_FLIPADST: TxTypeSlot = ("H_FLIPADST", None);

static TX_TYPE_DCT_ONLY: [TxTypeSlot; 1] = [DCT_DCT];

/// `Tx_Type_Intra_Inv_Set1` (spec §5.11.48).
static TX_TYPE_INTRA_INV_SET1: [TxTypeSlot; 7] =
    [IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

/// `Tx_Type_Intra_Inv_Set2` (spec §5.11.48).
static TX_TYPE_INTRA_INV_SET2: [TxTypeSlot; 5] = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

/// `Tx_Type_Inter_Inv_Set1` (spec §5.11.48).
static TX_TYPE_INTER_INV_SET1: [TxTypeSlot; 16] = [
    IDTX,
    V_DCT,
    H_DCT,
    V_ADST,
    H_ADST,
    V_FLIPADST,
    H_FLIPADST,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set2` (spec §5.11.48).
static TX_TYPE_INTER_INV_SET2: [TxTypeSlot; 12] = [
    IDTX,
    V_DCT,
    H_DCT,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];

/// `Tx_Type_Inter_Inv_Set3` (spec §5.11.48).
static TX_TYPE_INTER_INV_SET3: [TxTypeSlot; 2] = [IDTX, DCT_DCT];

/// The inverse table that maps a decoded `tx_type` symbol index to its
/// transform type, for `set`. Its length always matches the length of the
/// CDF [`tx_type_cdf`] returns for the same set.
#[must_use]
pub fn tx_type_inverse_set(set: Av1TxSet) -> &'static [TxTypeSlot] {
    match set {
        Av1TxSet::DctOnly => &TX_TYPE_DCT_ONLY,
        Av1TxSet::Intra1 => &TX_TYPE_INTRA_INV_SET1,
        Av1TxSet::Intra2 => &TX_TYPE_INTRA_INV_SET2,
        Av1TxSet::Inter1 => &TX_TYPE_INTER_INV_SET1,
        Av1TxSet::Inter2 => &TX_TYPE_INTER_INV_SET2,
        Av1TxSet::Inter3 => &TX_TYPE_INTER_INV_SET3,
    }
}

/// The number of `YMode` values `Default_Intra_Ext_Tx_Cdf` is indexed by
/// (spec `INTRA_MODES`).
pub const INTRA_MODES: usize = 13;

/// A valid, strictly increasing `N`-symbol CDF that varies with `seed`.
///
/// The `tx_type` CDFs below are placeholders in the sense the section
/// header describes, but the specification selects a *different* row per
/// transform size and (for intra blocks) per intra direction, so the rows
/// here must genuinely differ or that indexing would be unobservable and
/// untestable. Each row is a uniform split perturbed by less than half a
/// step, which keeps it strictly increasing and ending at 32768.
const fn placeholder_cdf<const N: usize>(seed: usize) -> [u16; N] {
    let mut cdf = [0u16; N];
    let step = 32768 / N;
    let mut index = 0;
    while index + 1 < N {
        cdf[index] = ((index + 1) * step + ((seed + index) % 7) * (step / 16)) as u16;
        index += 1;
    }
    cdf[N - 1] = 32768;
    cdf
}

/// Builds a `[rows][INTRA_MODES][N]` intra `tx_type` CDF table.
const fn intra_tx_type_table<const ROWS: usize, const N: usize>() -> [[[u16; N]; INTRA_MODES]; ROWS]
{
    let mut table = [[[0u16; N]; INTRA_MODES]; ROWS];
    let mut size = 0;
    while size < ROWS {
        let mut direction = 0;
        while direction < INTRA_MODES {
            table[size][direction] = placeholder_cdf::<N>(size * INTRA_MODES + direction);
            direction += 1;
        }
        size += 1;
    }
    table
}

/// Builds a `[rows][N]` inter `tx_type` CDF table.
const fn inter_tx_type_table<const ROWS: usize, const N: usize>() -> [[u16; N]; ROWS] {
    let mut table = [[0u16; N]; ROWS];
    let mut size = 0;
    while size < ROWS {
        table[size] = placeholder_cdf::<N>(size);
        size += 1;
    }
    table
}

/// `Default_Intra_Ext_Tx_Cdf[TX_SET_INTRA_1]`, indexed by `txSzSqr`
/// (`0` = `TX_4X4`, `1` = `TX_8X8`) then by the block's intra direction.
/// `TX_16X16` selects `TX_SET_INTRA_2` and `TX_32X32` selects
/// `TX_SET_DCTONLY`, so neither has a row here.
pub static INTRA_TX_TYPE_SET1: [[[u16; 7]; INTRA_MODES]; 2] = intra_tx_type_table();

/// `Default_Intra_Ext_Tx_Cdf[TX_SET_INTRA_2]`, indexed by `txSzSqr`
/// (`0` = `TX_4X4` .. `2` = `TX_16X16`) then by intra direction.
pub static INTRA_TX_TYPE_SET2: [[[u16; 5]; INTRA_MODES]; 3] = intra_tx_type_table();

/// `Default_Inter_Ext_Tx_Cdf[TX_SET_INTER_1]`, indexed by `txSzSqr`
/// (`0` = `TX_4X4`, `1` = `TX_8X8`).
pub static INTER_TX_TYPE_SET1: [[u16; 16]; 2] = inter_tx_type_table();

/// `Default_Inter_Ext_Tx_Cdf[TX_SET_INTER_2]`; only `TX_16X16` selects this
/// set, so it has the single row.
pub static INTER_TX_TYPE_SET2: [[u16; 12]; 1] = inter_tx_type_table();

/// `Default_Inter_Ext_Tx_Cdf[TX_SET_INTER_3]`, indexed by `txSzSqr`
/// (`0` = `TX_4X4` .. `3` = `TX_32X32`).
pub static INTER_TX_TYPE_SET3: [[u16; 2]; 4] = inter_tx_type_table();

/// `txSzSqr` as a table row index: `TX_4X4` is 0 through `TX_32X32` at 3.
fn tx_size_sqr_index(tx_size: usize) -> usize {
    match tx_size {
        4 => 0,
        8 => 1,
        16 => 2,
        _ => 3,
    }
}

/// The `tx_type` CDF for a `tx_size x tx_size` transform block in `set`,
/// or `None` when the set is `TX_SET_DCTONLY` and `read_tx_type` codes no
/// symbol at all (spec §5.11.48).
///
/// `intra_dir` is the block's `YMode` (spec `Default_Intra_Ext_Tx_Cdf` is
/// indexed by intra direction as well as by transform size) and is ignored
/// for the inter sets.
#[must_use]
pub fn tx_type_cdf(set: Av1TxSet, tx_size: usize, intra_dir: usize) -> Option<&'static [u16]> {
    let size = tx_size_sqr_index(tx_size);
    let direction = intra_dir.min(INTRA_MODES - 1);
    match set {
        Av1TxSet::DctOnly => None,
        Av1TxSet::Intra1 => Some(&INTRA_TX_TYPE_SET1[size][direction]),
        Av1TxSet::Intra2 => Some(&INTRA_TX_TYPE_SET2[size][direction]),
        Av1TxSet::Inter1 => Some(&INTER_TX_TYPE_SET1[size]),
        Av1TxSet::Inter2 => Some(&INTER_TX_TYPE_SET2[0]),
        Av1TxSet::Inter3 => Some(&INTER_TX_TYPE_SET3[size]),
    }
}

/// `tx_depth` symbol CDF for an 8x8 coding block (`Max_Tx_Depth[BLOCK_8X8]`
/// is 1, so the symbol is binary), at the fixed context 0 this module uses
/// wherever the specification varies a CDF by neighbouring block state.
pub static TX_SIZE_DEPTH_8X8: [u16; 2] = [19968, 32768];

/// `tx_depth` symbol CDF for a coding block of 16x16 or larger, where the
/// spec caps the coded depth at 2 and the symbol is ternary.
pub static TX_SIZE_DEPTH_LARGE: [u16; 3] = [16384, 27000, 32768];

/// The `tx_depth` CDF for a coding block `block_width` samples wide (spec
/// §5.11.16 `read_tx_size`), and the number of splits it can signal.
#[must_use]
pub fn tx_depth_cdf(block_width: usize) -> (&'static [u16], usize) {
    if block_width <= 8 {
        (&TX_SIZE_DEPTH_8X8, 1)
    } else {
        (&TX_SIZE_DEPTH_LARGE, 2)
    }
}

/// `eob_pt` symbol CDF for transform blocks up to 64 coefficients (TX_8X8),
/// indexed `[plane_type][ctx]`, extending [`EOB_PT_16`]'s shape to the
/// larger symbol count an 8x8 block's end-of-block position needs.
pub static EOB_PT_64: [[[u16; 9]; 2]; 2] = [
    [
        [2500, 8000, 14000, 19000, 23500, 27000, 29800, 31600, 32768],
        [4000, 10500, 16500, 21500, 25500, 28500, 30800, 32100, 32768],
    ],
    [
        [2200, 7300, 13200, 18200, 22800, 26500, 29500, 31500, 32768],
        [3600, 9800, 15600, 20700, 25000, 28200, 30600, 32000, 32768],
    ],
];

/// `eob_pt` symbol CDF for transform blocks of up to 256 coefficients
/// (TX_16X16), indexed `[plane_type][ctx]`.
pub static EOB_PT_256: [[[u16; 9]; 2]; 2] = [
    [
        [1800, 6200, 11500, 16200, 20500, 24200, 27600, 30400, 32768],
        [3000, 8400, 14000, 19000, 23200, 26800, 29600, 31500, 32768],
    ],
    [
        [1600, 5600, 10600, 15300, 19600, 23500, 27200, 30200, 32768],
        [2700, 7700, 13100, 18000, 22400, 26200, 29300, 31300, 32768],
    ],
];

/// `eob_pt` symbol CDF for transform blocks of up to 1024 coded
/// coefficients (TX_32X32, and TX_64X64 whose coefficients are confined to
/// the upper-left 32x32 quadrant), indexed `[plane_type][ctx]`.
pub static EOB_PT_1024: [[[u16; 11]; 2]; 2] = [
    [
        [
            1400, 4900, 9200, 13200, 16900, 20300, 23400, 26200, 28700, 30900, 32768,
        ],
        [
            2400, 6800, 11600, 16000, 19900, 23300, 26200, 28600, 30500, 31900, 32768,
        ],
    ],
    [
        [
            1300, 4500, 8600, 12500, 16200, 19600, 22800, 25700, 28300, 30600, 32768,
        ],
        [
            2200, 6300, 10900, 15200, 19100, 22600, 25700, 28300, 30300, 31800, 32768,
        ],
    ],
];

/// Selects the `eob_pt` CDF for a `size x size` block of *coded*
/// coefficients (spec §5.11.39's `eobMultisize` classes, restricted to the
/// square transforms this crate decodes). `size` is the coded coefficient
/// extent, so a 64x64 transform passes 32: AV1 codes coefficients only in
/// a 64x64 transform's upper-left 32x32 quadrant.
#[must_use]
pub fn eob_pt_cdf(size: usize, plane_type: usize) -> &'static [u16] {
    match size {
        4 => &EOB_PT_16[plane_type][0],
        8 => &EOB_PT_64[plane_type][0],
        16 => &EOB_PT_256[plane_type][0],
        _ => &EOB_PT_1024[plane_type][0],
    }
}

/// `Coeff_Base_Ctx_Offset` (§8.3.2) for `TX_CLASS_2D`, as the closed form
/// the specification's per-size tables all expand to: the offset depends
/// only on the anti-diagonal `row + col` the position sits on, so one rule
/// covers every square transform size. [`COEFF_BASE_CTX_OFFSET_4X4`] is the
/// TX_4X4 slice of exactly this rule (asserted by this module's tests), and
/// the position `(0, 0)` never reaches here - `coeff_base`'s DC context is
/// 0 unconditionally.
#[must_use]
pub fn coeff_base_ctx_offset(row: usize, column: usize) -> usize {
    match row + column {
        0 | 1 => 1,
        2 | 3 => 6,
        _ => 21,
    }
}

/// Generates the "up-right diagonal" scan order (spec §9.2's `Default_Scan`
/// construction) for a `size x size` (4 or 8) transform block: positions
/// are grouped by anti-diagonal `row + col`, each diagonal visited in row-
/// ascending order when `row + col` is odd and row-descending order when it
/// is even. Reproduces [`DEFAULT_SCAN_4X4`] exactly at `size == 4` (checked
/// by this module's tests).
pub fn up_right_diagonal_scan(size: usize) -> Vec<usize> {
    let mut scan = Vec::with_capacity(size * size);
    for diagonal in 0..(2 * size - 1) {
        let row_start = diagonal.saturating_sub(size - 1);
        let row_end = diagonal.min(size - 1);
        let rows: Box<dyn Iterator<Item = usize>> = if diagonal % 2 == 1 {
            Box::new(row_start..=row_end)
        } else {
            Box::new((row_start..=row_end).rev())
        };
        for row in rows {
            let col = diagonal - row;
            scan.push(row * size + col);
        }
    }
    scan
}

// --- Inter-frame mode/reference/motion-vector CDFs, fixed context 0 only. ---

/// `Default_Is_Inter_Cdf[0]`: is the block inter- or intra-predicted.
pub static IS_INTER: [u16; 2] = [806, 32768];

/// `Default_Comp_Mode_Cdf[1]`: no-neighbour single versus compound prediction.
pub static COMP_MODE: [u16; 2] = [24035, 32768];

/// `Default_Comp_Ref_Type_Cdf[2]`: no-neighbour compound direction type.
pub static COMP_REF_TYPE: [u16; 2] = [9166, 32768];

/// `Default_Uni_Comp_Ref_Cdf[1][0]`: no-neighbour forward versus backward pair.
pub static UNI_COMP_REF: [u16; 2] = [23152, 32768];

/// `Default_Uni_Comp_Ref_Cdf[1][1]`: no-neighbour LAST/LAST2 selection.
pub static UNI_COMP_REF_P1: [u16; 2] = [14173, 32768];

/// `Default_Compound_Mode_Cdf[0]`: compound motion-vector mode.
pub static COMPOUND_MODE: [u16; 8] = [7760, 13823, 15808, 17641, 19156, 20666, 26891, 32768];

/// `Default_Single_Ref_Cdf[1][0]`: no-neighbour `single_ref_p1`.
pub static SINGLE_REF_P1: [u16; 2] = [16973, 32768];

/// `Default_Single_Ref_Cdf[0][2]`: LAST/LAST2 versus LAST3/GOLDEN.
pub static SINGLE_REF_P3: [u16; 2] = [19647, 32768];

/// `Default_Single_Ref_Cdf[0][3]`: LAST versus LAST2.
pub static SINGLE_REF_P4: [u16; 2] = [24773, 32768];

/// `Default_New_Mv_Cdf[0]`: `new_mv` flag (0 selects `NEWMV`).
pub static NEW_MV: [u16; 2] = [24035, 32768];

/// `Default_Zero_Mv_Cdf[0]`: `zero_mv` flag (0 selects `GLOBALMV`) once `new_mv` is 1.
pub static ZERO_MV: [u16; 2] = [2175, 32768];

/// `Default_Ref_Mv_Cdf[0]`: nearest versus near motion-vector prediction.
pub static REF_MV: [u16; 2] = [23974, 32768];

/// `Default_Mv_Joint_Cdf`: which of the two motion-vector components change.
pub static MV_JOINT: [u16; 4] = [4096, 11264, 19328, 32768];

/// `Default_Mv_Class_Cdf[comp]`: motion-vector class per component (row 0, col 1).
pub static MV_CLASS: [[u16; 11]; 2] = [
    [
        28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768,
    ],
    [
        28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768,
    ],
];

/// `Default_Mv_Class0_Bit_Cdf[comp]`: MV_CLASS_0 integer bit.
pub static MV_CLASS0_BIT: [u16; 2] = [27648, 32768];

/// `Default_Mv_Class0_Fr_Cdf[comp]`: MV_CLASS_0 fractional part.
pub static MV_CLASS0_FR: [[u16; 4]; 2] =
    [[16384, 24576, 26624, 32768], [16384, 24576, 26624, 32768]];

/// `Default_Mv_Sign_Cdf[comp]`: motion-vector component sign.
pub static MV_SIGN: [u16; 2] = [16384, 32768];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn up_right_diagonal_scan_matches_the_known_4x4_table() {
        assert_eq!(up_right_diagonal_scan(4), DEFAULT_SCAN_4X4.to_vec());
    }

    #[test]
    fn up_right_diagonal_scan_is_a_permutation_of_all_positions_at_every_size() {
        for size in [8usize, 16, 32] {
            let scan = up_right_diagonal_scan(size);
            assert_eq!(scan.len(), size * size);
            let mut sorted = scan.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..size * size).collect::<Vec<_>>());
        }
    }

    #[test]
    fn coeff_base_ctx_offset_reproduces_the_tx_4x4_table() {
        for (row, offsets) in COEFF_BASE_CTX_OFFSET_4X4.iter().enumerate().take(4) {
            for (column, &offset) in offsets.iter().enumerate().take(4) {
                if row == 0 && column == 0 {
                    continue; // the DC position never consults the offset
                }
                assert_eq!(
                    coeff_base_ctx_offset(row, column),
                    usize::from(offset),
                    "row {row}, column {column}"
                );
            }
        }
    }

    #[test]
    fn eob_pt_cdfs_span_their_transform_blocks_coefficient_count() {
        // eobPt is coded with log2(count) + 1 symbols, and the largest
        // eobPt must be able to address the block's final coefficient.
        for (size, cdf) in [
            (4usize, eob_pt_cdf(4, 0)),
            (8, eob_pt_cdf(8, 0)),
            (16, eob_pt_cdf(16, 0)),
            (32, eob_pt_cdf(32, 0)),
        ] {
            let count = size * size;
            let max_eob_pt = cdf.len();
            let max_eob = (1usize << (max_eob_pt - 2)) + 1 + ((1 << (max_eob_pt - 2)) - 1);
            assert!(
                max_eob >= count,
                "size {size}: eob_pt reaches {max_eob}, needs {count}"
            );
            assert_eq!(*cdf.last().unwrap(), 32768);
        }
    }

    #[test]
    fn tx_depth_cdf_symbol_counts_match_the_spec_depth_cap() {
        assert_eq!(tx_depth_cdf(8).0.len(), 2);
        assert_eq!(tx_depth_cdf(8).1, 1);
        for width in [16usize, 32, 64] {
            assert_eq!(tx_depth_cdf(width).0.len(), 3);
            assert_eq!(tx_depth_cdf(width).1, 2);
        }
    }

    #[test]
    fn get_tx_set_follows_the_spec_derivation() {
        use Av1TxSet::{DctOnly, Inter1, Inter2, Inter3, Intra1, Intra2};
        // Intra, reduced_tx_set = 0.
        assert_eq!(get_tx_set(4, false, false), Intra1);
        assert_eq!(get_tx_set(8, false, false), Intra1);
        assert_eq!(get_tx_set(16, false, false), Intra2);
        assert_eq!(get_tx_set(32, false, false), DctOnly);
        assert_eq!(get_tx_set(64, false, false), DctOnly);
        // Inter, reduced_tx_set = 0.
        assert_eq!(get_tx_set(4, true, false), Inter1);
        assert_eq!(get_tx_set(8, true, false), Inter1);
        assert_eq!(get_tx_set(16, true, false), Inter2);
        assert_eq!(get_tx_set(32, true, false), Inter3);
        assert_eq!(get_tx_set(64, true, false), DctOnly);
        // reduced_tx_set = 1 collapses to the two reduced sets.
        for size in [4usize, 8, 16] {
            assert_eq!(get_tx_set(size, false, true), Intra2);
            assert_eq!(get_tx_set(size, true, true), Inter3);
        }
        assert_eq!(get_tx_set(32, false, true), DctOnly);
        assert_eq!(get_tx_set(32, true, true), Inter3);
    }

    #[test]
    fn tx_type_cdf_is_absent_only_for_the_dct_only_set() {
        assert!(tx_type_cdf(Av1TxSet::DctOnly, 32, 0).is_none());
        assert!(tx_type_cdf(Av1TxSet::DctOnly, 64, 0).is_none());
        for (set, size) in [
            (Av1TxSet::Intra1, 4usize),
            (Av1TxSet::Intra1, 8),
            (Av1TxSet::Intra2, 16),
            (Av1TxSet::Inter1, 4),
            (Av1TxSet::Inter2, 16),
            (Av1TxSet::Inter3, 32),
        ] {
            assert!(tx_type_cdf(set, size, 0).is_some());
        }
    }

    /// Every `tx_type` CDF must be a valid probability model (strictly
    /// increasing, ending at 32768) and must have exactly as many symbols
    /// as its set's inverse table has entries, since `read_tx_type` indexes
    /// one by the symbol decoded from the other.
    #[test]
    fn tx_type_cdfs_are_valid_and_match_their_inverse_sets() {
        for (set, sizes) in [
            (Av1TxSet::Intra1, &[4usize, 8][..]),
            (Av1TxSet::Intra2, &[4, 8, 16][..]),
            (Av1TxSet::Inter1, &[4, 8][..]),
            (Av1TxSet::Inter2, &[16][..]),
            (Av1TxSet::Inter3, &[4, 8, 16, 32][..]),
        ] {
            let inverse = tx_type_inverse_set(set);
            for &size in sizes {
                for direction in 0..INTRA_MODES {
                    let cdf = tx_type_cdf(set, size, direction).unwrap();
                    assert_eq!(cdf.len(), inverse.len(), "{set:?} {size} {direction}");
                    assert_eq!(*cdf.last().unwrap(), 32768);
                    assert!(cdf[0] > 0);
                    assert!(
                        cdf.windows(2).all(|pair| pair[0] < pair[1]),
                        "{set:?} {size} {direction} is not strictly increasing"
                    );
                }
            }
        }
    }

    /// `Default_Intra_Ext_Tx_Cdf` is indexed by intra direction as well as
    /// by transform size, so the rows must actually differ or that indexing
    /// would be unobservable.
    #[test]
    fn intra_tx_type_cdfs_vary_by_size_and_intra_direction() {
        let dc = tx_type_cdf(Av1TxSet::Intra2, 8, 0).unwrap();
        let paeth = tx_type_cdf(Av1TxSet::Intra2, 8, INTRA_MODES - 1).unwrap();
        assert_ne!(dc, paeth);
        assert_ne!(dc, tx_type_cdf(Av1TxSet::Intra2, 16, 0).unwrap());
    }

    /// The `V_*`/`H_*` half-identity types have no kernel in this crate and
    /// must be reported as absent rather than silently aliased onto another
    /// transform type.
    #[test]
    fn inverse_sets_mark_the_unimplemented_half_identity_types() {
        assert_eq!(
            tx_type_inverse_set(Av1TxSet::Intra1)
                .iter()
                .filter(|(_, kernel)| kernel.is_none())
                .map(|(name, _)| *name)
                .collect::<Vec<_>>(),
            ["V_DCT", "H_DCT"]
        );
        assert!(
            tx_type_inverse_set(Av1TxSet::Intra2)
                .iter()
                .all(|(_, kernel)| kernel.is_some())
        );
        assert!(
            tx_type_inverse_set(Av1TxSet::Inter3)
                .iter()
                .all(|(_, kernel)| kernel.is_some())
        );
        assert_eq!(
            tx_type_inverse_set(Av1TxSet::Inter1)
                .iter()
                .filter(|(_, kernel)| kernel.is_none())
                .count(),
            6
        );
    }
}
