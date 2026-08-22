use std::error;
use std::fmt;

/// A specialized result returned by zvidlib operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable, backend-independent error categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    Unsupported,
    MalformedMedia,
    ResourceLimit,
    Io,
    Codec,
    Graphics,
    Cancelled,
    InvalidState,
    Internal,
}

/// An error with a stable category and human-readable context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl error::Error for Error {}

/// Whether a capability can be used in the current configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Support {
    Available,
    Unavailable { reason: String },
}

impl Support {
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

/// The cost class for moving a video image to or from a graphics resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TransferMode {
    Shared,
    GpuCopy,
    CpuCopy,
    Unsupported,
}

/// A named feature and its runtime support state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capability {
    pub name: String,
    pub support: Support,
    pub transfer_mode: Option<TransferMode>,
}

/// Resource limits applied before media-controlled allocations occur.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_audio_channels: u16,
    pub max_sample_rate: u32,
    pub max_tracks: u16,
    pub max_allocation_bytes: u64,
    pub max_cached_frames: u32,
    pub max_decode_samples_per_seek: u32,
    /// Maximum number of OBUs accepted in one AV1 low-overhead unit.
    pub max_av1_obus_per_unit: u32,
    /// Maximum number of operating points accepted in an AV1 sequence header.
    pub max_av1_operating_points: u8,
    /// Maximum number of AV1 tiles accepted in one coded frame.
    pub max_av1_tiles_per_frame: u32,
    /// Maximum number of 4x4 AV1 reconstruction blocks decoded per frame.
    pub max_av1_blocks_per_frame: u32,
    /// Maximum number of CDEF strength units accepted in an AV1 frame header.
    pub max_av1_cdef_units: u8,
    /// Maximum number of loop-restoration units accepted per plane per frame.
    pub max_av1_restoration_units: u32,
    /// Maximum number of film-grain scaling points accepted per channel.
    pub max_av1_film_grain_points: u8,
    /// Maximum number of autoregressive film-grain coefficients accepted per channel.
    pub max_av1_film_grain_ar_coeffs: u16,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_width: 8_192,
            max_height: 8_192,
            max_audio_channels: 32,
            max_sample_rate: 384_000,
            max_tracks: 64,
            max_allocation_bytes: 512 * 1024 * 1024,
            max_cached_frames: 32,
            max_decode_samples_per_seek: 4_096,
            max_av1_obus_per_unit: 4_096,
            max_av1_operating_points: 32,
            max_av1_tiles_per_frame: 4_096,
            max_av1_blocks_per_frame: 4_194_304,
            max_av1_cdef_units: 8,
            max_av1_restoration_units: 4_096,
            max_av1_film_grain_points: 14,
            max_av1_film_grain_ar_coeffs: 24,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_keep_their_stable_category() {
        let error = Error::new(ErrorKind::MalformedMedia, "box exceeds its parent");
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
        assert_eq!(error.to_string(), "box exceeds its parent");
    }

    #[test]
    fn default_limits_are_conservative_and_nonzero() {
        let limits = Limits::default();
        assert!(limits.max_width > 0);
        assert!(limits.max_allocation_bytes > 0);
        assert!(limits.max_cached_frames > 0);
        assert!(limits.max_decode_samples_per_seek > 0);
        assert!(limits.max_av1_obus_per_unit > 0);
        assert!(limits.max_av1_operating_points > 0);
        assert!(limits.max_av1_tiles_per_frame > 0);
        assert!(limits.max_av1_blocks_per_frame > 0);
        assert!(limits.max_av1_cdef_units > 0);
        assert!(limits.max_av1_restoration_units > 0);
        assert!(limits.max_av1_film_grain_points > 0);
        assert!(limits.max_av1_film_grain_ar_coeffs > 0);
    }
}
