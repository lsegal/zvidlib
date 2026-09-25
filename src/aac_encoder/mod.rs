//! Native AAC-LC encoding, delegating to a platform codec.
//!
//! zvidlib deliberately ships no AAC bitstream implementation of its own (see
//! the reasoning on [`crate::AudioEncoder`]); this module fills the seam with
//! platform adapters instead: macOS AudioToolbox and, on Windows, Microsoft's
//! Media Foundation AAC encoder MFT. Other platforms report
//! [`CodecSupport::HardwareUnavailable`] rather than falling back to a
//! software encoder this crate does not carry.
//!
//! Both backends report their own encoder delay as
//! [`crate::AudioGapless::priming`] and measure
//! [`crate::AudioGapless::padding`] from the access units they actually
//! emitted, so a muxed track trims to exactly the samples the caller encoded.

#[cfg(target_os = "macos")]
mod audiotoolbox;
#[cfg(windows)]
mod windows_mf;

use crate::{
    AudioEncoder, AudioEncoderConfig, AudioEncoderFactory, Codec, CodecImplementation,
    CodecProfile, CodecSupport, Error, ErrorKind, Limits, Result,
};

/// AAC-LC always encodes 1024 samples per frame.
#[cfg(any(target_os = "macos", windows, test))]
const FRAME_LENGTH: u32 = 1024;

/// Returns the native AAC-LC encoder backend.
///
/// macOS encodes through AudioToolbox and Windows through Media Foundation's
/// AAC encoder MFT; every other target reports
/// [`CodecSupport::HardwareUnavailable`] from `capability`, consistent with
/// the crate carrying no software AAC encoder. The Media Foundation encoder
/// accepts only 44.1 and 48 kHz input and reports other rates the same way.
///
/// [`AudioEncoderConfig::configuration`] is either empty, leaving the bit rate
/// to the backend, or four big-endian bytes naming a nonzero target bit rate in
/// bits a second, the same form the native HEVC encoder takes. Each backend
/// rounds the request to the nearest rate it offers for the configured sample
/// rate and channel count; Media Foundation's are 96, 128, 160 and 192 kb/s.
pub fn native_aac_audio_encoder_factory() -> impl AudioEncoderFactory {
    AacEncoderFactory
}

#[derive(Clone, Copy, Debug)]
struct AacEncoderFactory;

impl AudioEncoderFactory for AacEncoderFactory {
    fn capability(&self, configuration: &AudioEncoderConfig) -> CodecSupport {
        if let unsupported @ (CodecSupport::UnsupportedCodec
        | CodecSupport::UnsupportedProfile
        | CodecSupport::InvalidConfiguration { .. }) = capability_without_backend(configuration)
        {
            return unsupported;
        }
        let bit_rate = parse_bit_rate(&configuration.configuration).flatten();
        #[cfg(target_os = "macos")]
        {
            audiotoolbox::capability(configuration, bit_rate)
        }
        #[cfg(windows)]
        {
            windows_mf::capability(configuration, bit_rate)
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            let _ = bit_rate;
            CodecSupport::HardwareUnavailable
        }
    }

    fn create(
        &self,
        configuration: &AudioEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn AudioEncoder>> {
        match capability_without_backend(configuration) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::UnsupportedCodec => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native AAC encoder requires the AAC codec",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native AAC encoder supports the Low Complexity profile",
                ));
            }
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
            CodecSupport::HardwareUnavailable => unreachable!("hardware is not checked here"),
        }
        let bit_rate = parse_bit_rate(&configuration.configuration).flatten();
        #[cfg(target_os = "macos")]
        {
            audiotoolbox::create(configuration, bit_rate, limits)
        }
        #[cfg(windows)]
        {
            windows_mf::create(configuration, bit_rate, limits)
        }
        #[cfg(not(any(target_os = "macos", windows)))]
        {
            let _ = (bit_rate, limits);
            Err(Error::new(
                ErrorKind::Unsupported,
                "native AAC encoding is only available on macOS and Windows",
            ))
        }
    }
}

fn capability_without_backend(configuration: &AudioEncoderConfig) -> CodecSupport {
    if configuration.codec != Codec::Aac {
        return CodecSupport::UnsupportedCodec;
    }
    if configuration.profile != CodecProfile::AacLowComplexity {
        return CodecSupport::UnsupportedProfile;
    }
    if parse_bit_rate(&configuration.configuration).is_none() {
        return CodecSupport::InvalidConfiguration {
            reason: "native AAC encoder configuration is either empty or four big-endian bytes \
                     giving a nonzero target bit rate in bits a second"
                .into(),
        };
    }
    if sampling_frequency_index(configuration.sample_rate).is_none() {
        return CodecSupport::InvalidConfiguration {
            reason: format!(
                "{} Hz is not one of the MPEG-4 audio sampling rates",
                configuration.sample_rate
            ),
        };
    }
    if configuration.channels != 1 && configuration.channels != 2 {
        return CodecSupport::InvalidConfiguration {
            reason: "native AAC encoder supports mono or stereo input".into(),
        };
    }
    if configuration.timescale != configuration.sample_rate {
        return CodecSupport::InvalidConfiguration {
            reason: "native AAC encoder timescale must equal the sample rate".into(),
        };
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Hardware,
    }
}

/// Reads the backend-private configuration: `Some(None)` for an empty one
/// (the backend's default bit rate), `Some(Some(bits_per_second))` for four
/// big-endian bytes naming a nonzero rate, and `None` for anything else. Zero
/// is rejected rather than read as "no target", as the HEVC encoder does.
fn parse_bit_rate(configuration: &[u8]) -> Option<Option<u32>> {
    match configuration {
        [] => Some(None),
        [a, b, c, d] => match u32::from_be_bytes([*a, *b, *c, *d]) {
            0 => None,
            bits_per_second => Some(Some(bits_per_second)),
        },
        _ => None,
    }
}

/// The end padding of a stream whose emitted access units cover
/// `encoded_frames` PCM frames, `priming` of them encoder delay ahead of the
/// `input_frames` the caller supplied. An encoder that emitted too few frames
/// to hold every input frame has lost audio, which is an error rather than a
/// padding a caller could correct.
#[cfg(any(target_os = "macos", windows, test))]
fn gapless_padding(encoded_frames: u64, priming: u32, input_frames: u64) -> Result<u32> {
    let presented = input_frames + u64::from(priming);
    let padding = encoded_frames.checked_sub(presented).ok_or_else(|| {
        Error::new(
            ErrorKind::Codec,
            format!(
                "the AAC encoder emitted {encoded_frames} frames, fewer than the {presented} its \
                 priming and input need"
            ),
        )
    })?;
    u32::try_from(padding)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AAC padding exceeds u32"))
}

/// The MPEG-4 Audio `samplingFrequencyIndex` for one of the fixed rates the
/// `AudioSpecificConfig` short form can name; `None` for anything else,
/// including the "escape" rates that need the long form this crate does not
/// build.
fn sampling_frequency_index(sample_rate: u32) -> Option<u8> {
    match sample_rate {
        96_000 => Some(0),
        88_200 => Some(1),
        64_000 => Some(2),
        48_000 => Some(3),
        44_100 => Some(4),
        32_000 => Some(5),
        24_000 => Some(6),
        22_050 => Some(7),
        16_000 => Some(8),
        12_000 => Some(9),
        11_025 => Some(10),
        8_000 => Some(11),
        7_350 => Some(12),
        _ => None,
    }
}

/// Builds the 2-byte `AudioSpecificConfig` (ISO/IEC 14496-3) for AAC-LC at
/// `sample_rate`/`channels`: `audioObjectType(5)=2`, `samplingFrequencyIndex(4)`,
/// `channelConfiguration(4)`, then `frameLengthFlag`, `dependsOnCoreCoder`, and
/// `extensionFlag` all zero (1024-sample frames, no dependency, no extension).
#[cfg(any(target_os = "macos", windows, test))]
fn audio_specific_config(sample_rate: u32, channels: u16) -> [u8; 2] {
    const AAC_LC: u8 = 2;
    let frequency_index =
        sampling_frequency_index(sample_rate).expect("caller already validated the sample rate");
    let channel_configuration: u8 = match channels {
        1 => 1,
        2 => 2,
        other => unreachable!("caller already validated the channel count: {other}"),
    };
    let byte0 = (AAC_LC << 3) | (frequency_index >> 1);
    let byte1 = ((frequency_index & 1) << 7) | (channel_configuration << 3);
    [byte0, byte1]
}

/// Writes an MPEG-4 descriptor length, big-endian base-128 with the
/// continuation bit set on every byte but the last, as `esds` and its nested
/// descriptors require.
#[cfg(any(target_os = "macos", windows, test))]
fn write_descriptor_length(out: &mut Vec<u8>, length: u32) {
    let mut chunks = [0_u8; 4];
    let mut remaining = length;
    let mut count = 0;
    loop {
        chunks[count] = (remaining & 0x7f) as u8;
        remaining >>= 7;
        count += 1;
        if remaining == 0 || count == chunks.len() {
            break;
        }
    }
    for (position, index) in (0..count).rev().enumerate() {
        let mut byte = chunks[index];
        if position != count - 1 {
            byte |= 0x80;
        }
        out.push(byte);
    }
}

#[cfg(any(target_os = "macos", windows, test))]
fn write_descriptor(out: &mut Vec<u8>, tag: u8, content: &[u8]) {
    out.push(tag);
    write_descriptor_length(out, u32::try_from(content.len()).unwrap_or(u32::MAX));
    out.extend_from_slice(content);
}

/// Builds the complete `esds` MP4 box declaring an AAC-LC track: an
/// `ES_Descriptor` wrapping a `DecoderConfigDescriptor` (object type `0x40`,
/// MPEG-4 audio stream type) whose `DecoderSpecificInfo` is the
/// `AudioSpecificConfig`, followed by the file-format `SLConfigDescriptor`.
#[cfg(any(target_os = "macos", windows, test))]
fn esds_box(sample_rate: u32, channels: u16) -> Vec<u8> {
    let audio_specific_config = audio_specific_config(sample_rate, channels);

    let mut decoder_specific_info = Vec::new();
    write_descriptor(&mut decoder_specific_info, 0x05, &audio_specific_config);

    let mut decoder_config_descriptor = vec![
        0x40, // objectTypeIndication: MPEG-4 Audio
        0x15, // streamType (5, audio) << 2 | upStream (0) << 1 | reserved (1)
        0, 0, 0, // bufferSizeDB
        0, 0, 0, 0, // maxBitrate
        0, 0, 0, 0, // avgBitrate
    ];
    decoder_config_descriptor.extend_from_slice(&decoder_specific_info);

    let mut es_descriptor = vec![0, 0]; // ES_ID
    es_descriptor.push(0); // no stream dependence, no URL, no OCR stream
    write_descriptor(&mut es_descriptor, 0x04, &decoder_config_descriptor);
    write_descriptor(&mut es_descriptor, 0x06, &[0x02]); // SLConfigDescriptor, MP4 predefined

    let mut payload = vec![0, 0, 0, 0]; // FullBox version/flags
    write_descriptor(&mut payload, 0x03, &es_descriptor);

    let mut output = u32::try_from(payload.len() + 8)
        .unwrap_or(u32::MAX)
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(b"esds");
    output.extend_from_slice(&payload);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_specific_config_encodes_aac_lc_stereo_48k() {
        // audioObjectType=2, samplingFrequencyIndex=3 (48000), channelConfiguration=2:
        // 00010 0011 0010 000 -> 0x11, 0x90.
        assert_eq!(audio_specific_config(48_000, 2), [0x11, 0x90]);
    }

    #[test]
    fn audio_specific_config_encodes_aac_lc_mono_44100() {
        // audioObjectType=2, samplingFrequencyIndex=4 (44100), channelConfiguration=1:
        // 00010 0100 0001 000 -> 0x12, 0x08.
        assert_eq!(audio_specific_config(44_100, 1), [0x12, 0x08]);
    }

    #[test]
    fn esds_box_round_trips_descriptor_lengths() {
        let esds = esds_box(48_000, 2);
        assert_eq!(&esds[4..8], b"esds");
        let declared_len = u32::from_be_bytes(esds[0..4].try_into().unwrap()) as usize;
        assert_eq!(declared_len, esds.len());
        // ES_Descriptor tag directly follows the 4-byte FullBox version/flags.
        assert_eq!(esds[12], 0x03);
    }

    #[test]
    fn write_descriptor_length_uses_continuation_bits_for_large_lengths() {
        let mut out = Vec::new();
        write_descriptor_length(&mut out, 0x1234);
        // 0x1234 = 4660 = 36 * 128 + 52 -> base-128 groups [0x24, 0x34], with
        // the continuation bit set on every byte but the last.
        assert_eq!(out, vec![0xa4, 0x34]);
    }

    #[test]
    fn write_descriptor_length_handles_a_single_byte_length() {
        let mut out = Vec::new();
        write_descriptor_length(&mut out, 0x2a);
        assert_eq!(out, vec![0x2a]);
    }

    #[test]
    fn configuration_is_empty_or_a_nonzero_big_endian_bit_rate() {
        assert_eq!(parse_bit_rate(&[]), Some(None));
        assert_eq!(
            parse_bit_rate(&128_000_u32.to_be_bytes()),
            Some(Some(128_000))
        );
        assert_eq!(parse_bit_rate(&0_u32.to_be_bytes()), None);
        assert_eq!(parse_bit_rate(&[1, 2]), None);
    }

    #[test]
    fn gapless_padding_is_what_the_emitted_frames_hold_past_priming_and_input() {
        // 1024 priming + 48_000 input = 49_024 frames; 48 access units hold 49_152.
        assert_eq!(gapless_padding(48 * 1024, 1024, 48_000).unwrap(), 128);
        assert_eq!(gapless_padding(2 * 1024, 1024, 1024).unwrap(), 0);
        assert!(gapless_padding(47 * 1024, 1024, 48_000).is_err());
    }

    #[test]
    fn capability_rejects_wrong_codec_profile_and_configuration() {
        let factory = native_aac_audio_encoder_factory();
        let base = AudioEncoderConfig {
            codec: Codec::Aac,
            profile: CodecProfile::AacLowComplexity,
            sample_rate: 48_000,
            channels: 2,
            timescale: 48_000,
            configuration: Vec::new(),
        };
        let mut candidate = base.clone();
        candidate.codec = Codec::Av1;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedCodec
        );

        candidate = base.clone();
        candidate.profile = CodecProfile::Av1Main;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedProfile
        );

        candidate = base.clone();
        candidate.configuration = vec![0];
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = base.clone();
        candidate.configuration = 0_u32.to_be_bytes().to_vec();
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = base.clone();
        candidate.sample_rate = 44_101;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = base.clone();
        candidate.channels = 6;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = base;
        candidate.timescale = 44_100;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));
    }
}
