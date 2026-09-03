//! Guards `examples/web_canvas/samples.js` against the files it describes.
//!
//! The page asks the browser whether it can decode a sample *before* it fetches one, which is
//! what keeps a browser with no HEVC decoder from downloading the HEVC copy of the clip only to
//! find out (issue #441). Asking without the file means the codec string and the coded size have
//! to be written down in JavaScript, and a written-down codec string is a claim about a file that
//! nothing in the browser ever checks: a re-encoded sample would keep being probed under its old
//! string, and the answer - yes or no - would be about a track that is no longer there.
//!
//! So the claim is checked here instead, against the same derivation
//! [`zvidlib::derive_codec_string`] performs on the track's own `hvcC`/`av1C` box when the page
//! actually decodes it.
//!
//! Deliberately a hand-rolled scan rather than a JavaScript parse, for the reason
//! `ci_workflows_cache_cargo` gives for reading YAML line by line: the alternative is a
//! dependency for a hygiene check, and the file is one object literal per sample with one field
//! per line.

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::path::{Path, PathBuf};
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySource;
use zvidlib::{Codec, Mp4Demuxer, Mp4DemuxerOptions, Mp4Track, TrackKind, derive_codec_string};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut boxed = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match boxed.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("unexpected pending future"),
    }
}

fn example_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/web_canvas")
}

/// One entry of `SAMPLES`, as the page reads it.
#[derive(Debug)]
struct Declared {
    file: String,
    codec: String,
    width: u32,
    height: u32,
    description: String,
}

fn string_field(entry: &str, key: &str) -> String {
    let needle = format!("{key}: \"");
    let at = entry
        .find(&needle)
        .unwrap_or_else(|| panic!("a SAMPLES entry without a `{key}` string:\n{entry}"));
    let rest = &entry[at + needle.len()..];
    let end = rest
        .find('"')
        .unwrap_or_else(|| panic!("an unterminated `{key}` string:\n{entry}"));
    rest[..end].to_owned()
}

fn number_field(entry: &str, key: &str) -> u32 {
    let needle = format!("{key}: ");
    let at = entry
        .find(&needle)
        .unwrap_or_else(|| panic!("a SAMPLES entry without a `{key}` number:\n{entry}"));
    let rest = &entry[at + needle.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits
        .parse()
        .unwrap_or_else(|error| panic!("`{key}` is not a number ({error}):\n{entry}"))
}

fn declared_samples(source: &str) -> Vec<Declared> {
    let marker = "export const SAMPLES = [";
    let start = source
        .find(marker)
        .expect("samples.js exports a SAMPLES array");
    let body = &source[start + marker.len()..];
    let end = body
        .find("\n];")
        .expect("the SAMPLES array is closed by `];` in the first column");
    body[..end]
        .split('{')
        .skip(1)
        .map(|entry| {
            let entry = &entry[..entry.find('}').expect("a closed SAMPLES entry")];
            Declared {
                file: string_field(entry, "file"),
                codec: string_field(entry, "codec"),
                width: number_field(entry, "width"),
                height: number_field(entry, "height"),
                description: string_field(entry, "description"),
            }
        })
        .collect()
}

/// Reads a declared sample. The bundled files are symlinks into `examples/media/`, and a checkout
/// made without symlink support leaves the link target's path as the file's contents instead, so
/// a short unrecognizable "file" is followed by hand rather than read as an MP4.
fn read_sample(file: &str) -> Vec<u8> {
    let path = example_dir().join(file.trim_start_matches("./"));
    let bytes =
        std::fs::read(&path).unwrap_or_else(|error| panic!("reading {} ({error})", path.display()));
    if bytes.len() > 4096 || bytes.get(4..8) == Some(b"ftyp".as_slice()) {
        return bytes;
    }
    let target = String::from_utf8(bytes)
        .unwrap_or_else(|_| panic!("{} is neither an MP4 nor a symlink", path.display()));
    let linked = path
        .parent()
        .expect("a sample has a parent directory")
        .join(target.trim());
    std::fs::read(&linked).unwrap_or_else(|error| panic!("reading {} ({error})", linked.display()))
}

fn video_track(demuxer: &Mp4Demuxer) -> &Mp4Track {
    demuxer
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .expect("a bundled sample has a video track")
}

#[test]
fn every_declared_sample_carries_the_track_it_claims() {
    let source = std::fs::read_to_string(example_dir().join("samples.js"))
        .expect("reading examples/web_canvas/samples.js");
    let declared = declared_samples(&source);
    assert!(
        declared.len() >= 2,
        "the page needs more than one sample to have a choice to make; found {}",
        declared.len()
    );

    let mut frame_counts = Vec::new();
    for sample in &declared {
        let bytes = read_sample(&sample.file);
        let demuxer = block_on(Mp4Demuxer::open(
            &MemorySource::new(bytes),
            Mp4DemuxerOptions::default(),
        ))
        .unwrap_or_else(|error| panic!("demuxing {} ({error})", sample.file));
        let video = video_track(&demuxer);

        let derived = derive_codec_string(video.codec, &video.decoder_config)
            .unwrap_or_else(|error| panic!("deriving {}'s codec string ({error})", sample.file));
        assert_eq!(
            derived.codec_string, sample.codec,
            "{} ({}) declares a codec string its track does not derive",
            sample.file, sample.description
        );

        let dimensions = video
            .dimensions
            .unwrap_or_else(|| panic!("{} reports no coded dimensions", sample.file));
        assert_eq!(
            (dimensions.width, dimensions.height),
            (sample.width, sample.height),
            "{} declares a coded size its track does not have",
            sample.file
        );

        // The page decodes AAC out of whichever sample it chose, so a sample without an audio
        // track would silently make the audio half of the example depend on which video codec
        // the browser happened to support.
        assert!(
            demuxer
                .tracks
                .iter()
                .any(|track| track.kind == TrackKind::Audio && track.codec == Codec::Aac),
            "{} has no AAC audio track",
            sample.file
        );
        frame_counts.push((sample.file.clone(), video.presentation_order.len()));
    }

    // Whichever sample the browser picks, the timeline has to mean the same thing: the page's
    // range input is indexed in frames, and the preview stride, the scrub walk and the recorded
    // seek costs are all quoted against that index.
    let (first_file, first_count) = &frame_counts[0];
    for (file, count) in &frame_counts[1..] {
        assert_eq!(
            count, first_count,
            "{file} has {count} frames but {first_file} has {first_count}: the samples are \
             interchangeable only if the timeline is the same length in both"
        );
    }
}
