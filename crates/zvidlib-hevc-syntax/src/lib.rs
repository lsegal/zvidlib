//! HEVC bitstream syntax shared by zvidlib's HEVC decoder, encoder and
//! hardware backends: bit reading and writing, CABAC and the binarization of
//! its syntax elements, NAL units, `hvcC` records, and the VPS, SPS, PPS, SEI,
//! VUI, HRD and slice header syntax.
//!
//! It is derived from `oxideav-h265`; see `NOTICE.md` and `LICENSE`. This is
//! an internal crate of [zvidlib](https://crates.io/crates/zvidlib). Depend on
//! `zvidlib` rather than on this crate directly.

#![warn(missing_debug_implementations)]
#![allow(dead_code, unused_imports)]

#[doc(hidden)]
pub mod binarization;
#[doc(hidden)]
pub mod bitreader;
#[doc(hidden)]
pub mod cabac;
#[doc(hidden)]
pub mod hrd;
#[doc(hidden)]
pub mod hvcc;
#[doc(hidden)]
pub mod nal;
#[doc(hidden)]
pub mod pps;
#[doc(hidden)]
pub mod scaling_list;
#[doc(hidden)]
pub mod scan;
#[doc(hidden)]
pub mod sei;
#[doc(hidden)]
pub mod slice;
#[doc(hidden)]
pub mod sps;
#[doc(hidden)]
pub mod vps;
#[doc(hidden)]
pub mod vui;
#[doc(hidden)]
pub mod encoder {
    pub mod bitwriter;
    pub mod cabac;
    pub mod nal;
}

#[doc(hidden)]
pub use bitreader::{BitReader, BitReaderError};
#[doc(hidden)]
pub use cabac::{CabacEngine, CabacError, ContextModel, init_type};
pub use hrd::HrdError;
#[doc(hidden)]
pub use hrd::{
    CpbEntry, HEVC_MAX_CPB_CNT, HEVC_MAX_ELEMENTAL_DURATION_IN_TC_MINUS1, HrdCommonInfo,
    HrdParameters, SubLayerHrd, SubLayerHrdParameters, VpsHrdEntry,
};
pub use hvcc::{
    HvccError, HvccRecord, extradata_is_hvcc, nal_unit_from_coded, parse_hvcc,
    split_length_prefixed,
};
pub use nal::{NalError, NalHeader, NalIter, NalUnit, collect_nal_units};
pub use pps::PpsError;
#[doc(hidden)]
pub use pps::{
    ChromaQpOffsetListEntry, DeblockingFilterControl, PicParameterSet, PpsRangeExtension, TileInfo,
};
#[doc(hidden)]
pub use scaling_list::{
    MAX_COEF_NUM, NUM_MATRIX_IDS, NUM_SIZE_IDS, ScalingFactorMatrix, ScalingFactors,
    ScalingListData, ScalingListError, ScalingListMatrix,
};
#[doc(hidden)]
pub use scan::{
    ScanIdx, ScanOrderError, ScanPos, horizontal, scan_order, traverse, up_right_diagonal, vertical,
};
pub use slice::SliceError;
#[doc(hidden)]
pub use slice::{
    BLA_W_LP, EntryPointOffsets, IDR_N_LP, IDR_W_RADL, NumPicTotalCurrInputs, PredWeightEntry,
    PredWeightTable, PredWeightTableInputs, RSV_IRAP_VCL23, RefPicListsModification,
    SliceDeblocking, SliceLongTermRefPic, SliceLongTermRefPicSource, SliceSegmentHeader, SliceType,
};
#[doc(hidden)]
pub use sps::{
    ConformanceWindow, HEVC_MAX_NUM_LONG_TERM_RPS, HEVC_MAX_NUM_SHORT_TERM_RPS, HEVC_MAX_RPS_PICS,
    LongTermRefPicEntry, MaterializedShortTermRefPicSet, OpaqueTail, PcmInfo, SeqParameterSet,
    ShortTermRefPicSet, SpsExtensionFlags, SpsRangeExtension,
};
pub use sps::{ShortTermRefPicSetMaterializeError, SpsError};
pub use vps::VpsError;
#[doc(hidden)]
pub use vps::{
    HEVC_MAX_SUB_LAYERS, HEVC_VPS_MAX_NUM_LAYER_SETS, HEVC_VPS_MAX_NUM_LAYERS, HevcVps,
    LayerIdInclusionRow, ProfileTierLevel, SubLayerOrderingInfo, VpsTimingInfo,
};
pub use vui::VuiError;
#[doc(hidden)]
pub use vui::{
    BitstreamRestriction, ColourDescription, DefaultDisplayWindow, EXTENDED_SAR, VideoSignalType,
    VuiParameters, VuiTimingInfo,
};
