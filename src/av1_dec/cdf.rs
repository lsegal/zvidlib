//! The complete set of adaptive CDF arrays named in the semantics of
//! `init_non_coeff_cdfs` and `init_coeff_cdfs` (specification section 6.8.2),
//! which frames save to and load from their reference slots.

use super::tables::*;

/// A (possibly nested) array of CDFs, each ending with its adaptation counter.
pub(crate) trait CdfArray {
    fn clear_counts(&mut self);
}

impl<const N: usize> CdfArray for [u16; N] {
    fn clear_counts(&mut self) {
        self[N - 1] = 0;
    }
}

impl<T: CdfArray, const M: usize> CdfArray for [T; M] {
    fn clear_counts(&mut self) {
        for inner in self.iter_mut() {
            inner.clear_counts();
        }
    }
}

macro_rules! cdf_context {
    ($( $field:ident : $ty:ty ),* $(,)?) => {
        #[derive(Clone)]
        pub(crate) struct CdfContext {
            $( pub(crate) $field: $ty, )*
        }

        impl CdfContext {
            /// `load_cdfs`: once loaded, every symbol counter restarts at zero.
            pub(crate) fn clear_counts(&mut self) {
                $( self.$field.clear_counts(); )*
            }
        }
    };
}

cdf_context! {
    y_mode: [[u16; 14]; 4],
    uv_mode_cfl_not_allowed: [[u16; 14]; 13],
    uv_mode_cfl_allowed: [[u16; 15]; 13],
    angle_delta: [[u16; 8]; 8],
    intrabc: [u16; 3],
    partition_w8: [[u16; 5]; 4],
    partition_w16: [[u16; 11]; 4],
    partition_w32: [[u16; 11]; 4],
    partition_w64: [[u16; 11]; 4],
    partition_w128: [[u16; 9]; 4],
    segment_id: [[u16; 9]; 3],
    segment_id_predicted: [[u16; 3]; 3],
    tx_8x8: [[u16; 3]; 3],
    tx_16x16: [[u16; 4]; 3],
    tx_32x32: [[u16; 4]; 3],
    tx_64x64: [[u16; 4]; 3],
    txfm_split: [[u16; 3]; 21],
    filter_intra_mode: [u16; 6],
    filter_intra: [[u16; 3]; 22],
    interp_filter: [[u16; 4]; 16],
    motion_mode: [[u16; 4]; 22],
    new_mv: [[u16; 3]; 6],
    zero_mv: [[u16; 3]; 2],
    ref_mv: [[u16; 3]; 6],
    compound_mode: [[u16; 9]; 8],
    drl_mode: [[u16; 3]; 3],
    is_inter: [[u16; 3]; 4],
    comp_mode: [[u16; 3]; 5],
    skip_mode: [[u16; 3]; 3],
    skip: [[u16; 3]; 3],
    comp_ref: [[[u16; 3]; 3]; 3],
    comp_bwd_ref: [[[u16; 3]; 2]; 3],
    single_ref: [[[u16; 3]; 6]; 3],
    mv_joint: [[u16; 5]; 2],
    mv_class: [[[u16; 12]; 2]; 2],
    mv_class0_bit: [[[u16; 3]; 2]; 2],
    mv_fr: [[[u16; 5]; 2]; 2],
    mv_class0_fr: [[[[u16; 5]; 2]; 2]; 2],
    mv_class0_hp: [[[u16; 3]; 2]; 2],
    mv_sign: [[[u16; 3]; 2]; 2],
    mv_bit: [[[[u16; 3]; 10]; 2]; 2],
    mv_hp: [[[u16; 3]; 2]; 2],
    palette_y_mode: [[[u16; 3]; 3]; 7],
    palette_uv_mode: [[u16; 3]; 2],
    palette_y_size: [[u16; 8]; 7],
    palette_uv_size: [[u16; 8]; 7],
    palette_size_2_y_color: [[u16; 3]; 5],
    palette_size_3_y_color: [[u16; 4]; 5],
    palette_size_4_y_color: [[u16; 5]; 5],
    palette_size_5_y_color: [[u16; 6]; 5],
    palette_size_6_y_color: [[u16; 7]; 5],
    palette_size_7_y_color: [[u16; 8]; 5],
    palette_size_8_y_color: [[u16; 9]; 5],
    palette_size_2_uv_color: [[u16; 3]; 5],
    palette_size_3_uv_color: [[u16; 4]; 5],
    palette_size_4_uv_color: [[u16; 5]; 5],
    palette_size_5_uv_color: [[u16; 6]; 5],
    palette_size_6_uv_color: [[u16; 7]; 5],
    palette_size_7_uv_color: [[u16; 8]; 5],
    palette_size_8_uv_color: [[u16; 9]; 5],
    delta_q: [u16; 5],
    delta_lf: [u16; 5],
    delta_lf_multi: [[u16; 5]; 4],
    intra_tx_type_set1: [[[u16; 8]; 13]; 2],
    intra_tx_type_set2: [[[u16; 6]; 13]; 3],
    inter_tx_type_set1: [[u16; 17]; 2],
    inter_tx_type_set2: [u16; 13],
    inter_tx_type_set3: [[u16; 3]; 4],
    use_obmc: [[u16; 3]; 22],
    inter_intra: [[u16; 3]; 3],
    comp_ref_type: [[u16; 3]; 5],
    cfl_sign: [u16; 9],
    uni_comp_ref: [[[u16; 3]; 3]; 3],
    wedge_inter_intra: [[u16; 3]; 22],
    comp_group_idx: [[u16; 3]; 6],
    compound_idx: [[u16; 3]; 6],
    compound_type: [[u16; 3]; 22],
    inter_intra_mode: [[u16; 5]; 3],
    wedge_index: [[u16; 17]; 22],
    cfl_alpha: [[u16; 17]; 6],
    use_wiener: [u16; 3],
    use_sgrproj: [u16; 3],
    restoration_type: [u16; 4],
    txb_skip: [[[u16; 3]; 13]; 5],
    eob_pt_16: [[[u16; 6]; 2]; 2],
    eob_pt_32: [[[u16; 7]; 2]; 2],
    eob_pt_64: [[[u16; 8]; 2]; 2],
    eob_pt_128: [[[u16; 9]; 2]; 2],
    eob_pt_256: [[[u16; 10]; 2]; 2],
    eob_pt_512: [[u16; 11]; 2],
    eob_pt_1024: [[u16; 12]; 2],
    eob_extra: [[[[u16; 3]; 9]; 2]; 5],
    dc_sign: [[[u16; 3]; 3]; 2],
    coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5],
    coeff_base: [[[[u16; 5]; 42]; 2]; 5],
    coeff_br: [[[[u16; 5]; 21]; 2]; 5],
}

impl CdfContext {
    /// `init_non_coeff_cdfs( )` followed by `init_coeff_cdfs( )` for
    /// `base_q_idx`.
    pub(crate) fn new(base_q_idx: u8) -> Box<Self> {
        let mut context = Box::new(Self {
            y_mode: DEFAULT_Y_MODE_CDF,
            uv_mode_cfl_not_allowed: DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
            uv_mode_cfl_allowed: DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
            angle_delta: DEFAULT_ANGLE_DELTA_CDF,
            intrabc: DEFAULT_INTRABC_CDF,
            partition_w8: DEFAULT_PARTITION_W8_CDF,
            partition_w16: DEFAULT_PARTITION_W16_CDF,
            partition_w32: DEFAULT_PARTITION_W32_CDF,
            partition_w64: DEFAULT_PARTITION_W64_CDF,
            partition_w128: DEFAULT_PARTITION_W128_CDF,
            segment_id: DEFAULT_SEGMENT_ID_CDF,
            segment_id_predicted: DEFAULT_SEGMENT_ID_PREDICTED_CDF,
            tx_8x8: DEFAULT_TX_8X8_CDF,
            tx_16x16: DEFAULT_TX_16X16_CDF,
            tx_32x32: DEFAULT_TX_32X32_CDF,
            tx_64x64: DEFAULT_TX_64X64_CDF,
            txfm_split: DEFAULT_TXFM_SPLIT_CDF,
            filter_intra_mode: DEFAULT_FILTER_INTRA_MODE_CDF,
            filter_intra: DEFAULT_FILTER_INTRA_CDF,
            interp_filter: DEFAULT_INTERP_FILTER_CDF,
            motion_mode: DEFAULT_MOTION_MODE_CDF,
            new_mv: DEFAULT_NEW_MV_CDF,
            zero_mv: DEFAULT_ZERO_MV_CDF,
            ref_mv: DEFAULT_REF_MV_CDF,
            compound_mode: DEFAULT_COMPOUND_MODE_CDF,
            drl_mode: DEFAULT_DRL_MODE_CDF,
            is_inter: DEFAULT_IS_INTER_CDF,
            comp_mode: DEFAULT_COMP_MODE_CDF,
            skip_mode: DEFAULT_SKIP_MODE_CDF,
            skip: DEFAULT_SKIP_CDF,
            comp_ref: DEFAULT_COMP_REF_CDF,
            comp_bwd_ref: DEFAULT_COMP_BWD_REF_CDF,
            single_ref: DEFAULT_SINGLE_REF_CDF,
            mv_joint: [DEFAULT_MV_JOINT_CDF; 2],
            mv_class: [DEFAULT_MV_CLASS_CDF; 2],
            mv_class0_bit: [[DEFAULT_MV_CLASS0_BIT_CDF; 2]; 2],
            mv_fr: [DEFAULT_MV_FR_CDF; 2],
            mv_class0_fr: [DEFAULT_MV_CLASS0_FR_CDF; 2],
            mv_class0_hp: [[DEFAULT_MV_CLASS0_HP_CDF; 2]; 2],
            mv_sign: [[DEFAULT_MV_SIGN_CDF; 2]; 2],
            mv_bit: [[DEFAULT_MV_BIT_CDF; 2]; 2],
            mv_hp: [[DEFAULT_MV_HP_CDF; 2]; 2],
            palette_y_mode: DEFAULT_PALETTE_Y_MODE_CDF,
            palette_uv_mode: DEFAULT_PALETTE_UV_MODE_CDF,
            palette_y_size: DEFAULT_PALETTE_Y_SIZE_CDF,
            palette_uv_size: DEFAULT_PALETTE_UV_SIZE_CDF,
            palette_size_2_y_color: DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF,
            palette_size_3_y_color: DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF,
            palette_size_4_y_color: DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF,
            palette_size_5_y_color: DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF,
            palette_size_6_y_color: DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF,
            palette_size_7_y_color: DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF,
            palette_size_8_y_color: DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF,
            palette_size_2_uv_color: DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF,
            palette_size_3_uv_color: DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF,
            palette_size_4_uv_color: DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF,
            palette_size_5_uv_color: DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF,
            palette_size_6_uv_color: DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF,
            palette_size_7_uv_color: DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF,
            palette_size_8_uv_color: DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF,
            delta_q: DEFAULT_DELTA_Q_CDF,
            delta_lf: DEFAULT_DELTA_LF_CDF,
            delta_lf_multi: [DEFAULT_DELTA_LF_CDF; 4],
            intra_tx_type_set1: DEFAULT_INTRA_TX_TYPE_SET1_CDF,
            intra_tx_type_set2: DEFAULT_INTRA_TX_TYPE_SET2_CDF,
            inter_tx_type_set1: DEFAULT_INTER_TX_TYPE_SET1_CDF,
            inter_tx_type_set2: DEFAULT_INTER_TX_TYPE_SET2_CDF,
            inter_tx_type_set3: DEFAULT_INTER_TX_TYPE_SET3_CDF,
            use_obmc: DEFAULT_USE_OBMC_CDF,
            inter_intra: DEFAULT_INTER_INTRA_CDF,
            comp_ref_type: DEFAULT_COMP_REF_TYPE_CDF,
            cfl_sign: DEFAULT_CFL_SIGN_CDF,
            uni_comp_ref: DEFAULT_UNI_COMP_REF_CDF,
            wedge_inter_intra: DEFAULT_WEDGE_INTER_INTRA_CDF,
            comp_group_idx: DEFAULT_COMP_GROUP_IDX_CDF,
            compound_idx: DEFAULT_COMPOUND_IDX_CDF,
            compound_type: DEFAULT_COMPOUND_TYPE_CDF,
            inter_intra_mode: DEFAULT_INTER_INTRA_MODE_CDF,
            wedge_index: DEFAULT_WEDGE_INDEX_CDF,
            cfl_alpha: DEFAULT_CFL_ALPHA_CDF,
            use_wiener: DEFAULT_USE_WIENER_CDF,
            use_sgrproj: DEFAULT_USE_SGRPROJ_CDF,
            restoration_type: DEFAULT_RESTORATION_TYPE_CDF,
            txb_skip: DEFAULT_TXB_SKIP_CDF[0],
            eob_pt_16: DEFAULT_EOB_PT_16_CDF[0],
            eob_pt_32: DEFAULT_EOB_PT_32_CDF[0],
            eob_pt_64: DEFAULT_EOB_PT_64_CDF[0],
            eob_pt_128: DEFAULT_EOB_PT_128_CDF[0],
            eob_pt_256: DEFAULT_EOB_PT_256_CDF[0],
            eob_pt_512: DEFAULT_EOB_PT_512_CDF[0],
            eob_pt_1024: DEFAULT_EOB_PT_1024_CDF[0],
            eob_extra: DEFAULT_EOB_EXTRA_CDF[0],
            dc_sign: DEFAULT_DC_SIGN_CDF[0],
            coeff_base_eob: DEFAULT_COEFF_BASE_EOB_CDF[0],
            coeff_base: DEFAULT_COEFF_BASE_CDF[0],
            coeff_br: DEFAULT_COEFF_BR_CDF[0],
        });
        context.init_coeff_cdfs(base_q_idx);
        context
    }

    /// `init_coeff_cdfs( )`.
    pub(crate) fn init_coeff_cdfs(&mut self, base_q_idx: u8) {
        let idx = match base_q_idx {
            0..=20 => 0,
            21..=60 => 1,
            61..=120 => 2,
            _ => 3,
        };
        self.txb_skip = DEFAULT_TXB_SKIP_CDF[idx];
        self.eob_pt_16 = DEFAULT_EOB_PT_16_CDF[idx];
        self.eob_pt_32 = DEFAULT_EOB_PT_32_CDF[idx];
        self.eob_pt_64 = DEFAULT_EOB_PT_64_CDF[idx];
        self.eob_pt_128 = DEFAULT_EOB_PT_128_CDF[idx];
        self.eob_pt_256 = DEFAULT_EOB_PT_256_CDF[idx];
        self.eob_pt_512 = DEFAULT_EOB_PT_512_CDF[idx];
        self.eob_pt_1024 = DEFAULT_EOB_PT_1024_CDF[idx];
        self.eob_extra = DEFAULT_EOB_EXTRA_CDF[idx];
        self.dc_sign = DEFAULT_DC_SIGN_CDF[idx];
        self.coeff_base_eob = DEFAULT_COEFF_BASE_EOB_CDF[idx];
        self.coeff_base = DEFAULT_COEFF_BASE_CDF[idx];
        self.coeff_br = DEFAULT_COEFF_BR_CDF[idx];
    }

    /// Replaces the coefficient CDFs with another context's, as
    /// `load_cdfs` would for them alone.
    pub(crate) fn copy_coeff_cdfs_from(&mut self, other: &Self) {
        self.txb_skip = other.txb_skip;
        self.eob_pt_16 = other.eob_pt_16;
        self.eob_pt_32 = other.eob_pt_32;
        self.eob_pt_64 = other.eob_pt_64;
        self.eob_pt_128 = other.eob_pt_128;
        self.eob_pt_256 = other.eob_pt_256;
        self.eob_pt_512 = other.eob_pt_512;
        self.eob_pt_1024 = other.eob_pt_1024;
        self.eob_extra = other.eob_extra;
        self.dc_sign = other.dc_sign;
        self.coeff_base_eob = other.coeff_base_eob;
        self.coeff_base = other.coeff_base;
        self.coeff_br = other.coeff_br;
    }
}
