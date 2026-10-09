//! Plays an MP4 or WebM on demand natively with `OnDemandPlayer` (issue #689):
//! only the container's header and index are read up front, and then only the
//! compressed samples playback reaches, within a 4 MiB budget. Audio plays on
//! the default output device; the decoded frames are counted rather than
//! drawn.
//!
//! ```console
//! cargo run --release --example on_demand_player --features native -- [movie.mp4] [language]
//! ```
//!
//! With no path it plays the bundled AV1 sample. With a language, such as
//! `fra`, it switches to that audio track halfway through.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use zvidlib::{OnDemandOptions, OnDemandPlayer, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/media/BigBuckBunny.av1.mp4")
    });
    let language = args.next();

    let mut player = OnDemandPlayer::open(
        &path,
        OnDemandOptions {
            video_budget_bytes: 3_500_000,
            audio_budget_bytes: 512 * 1024,
            audio_track: 0,
            // Decoded pictures are not covered by the budgets above.
            max_cached_frames: 8,
        },
    )?;
    let dimensions = player.dimensions();
    // A WebM with Cues is indexed as playback reaches it, so its frame count
    // is unknown and its duration estimated until playback nears its end.
    let frames = player
        .frame_count()
        .map_or_else(|| "indexed as it plays".to_owned(), |count| format!("{count} frames"));
    println!(
        "Opened {}: {}x{}, {:.1} s, {frames}",
        path.display(),
        dimensions.width,
        dimensions.height,
        player.duration().as_secs_f64(),
    );
    for (index, track) in player.audio_tracks().iter().enumerate() {
        println!(
            "  audio track {index}: {:?}, language {}",
            track.codec,
            track.language.as_deref().unwrap_or("unknown")
        );
    }

    player.play()?;
    let started = Instant::now();
    let mut presented = 0;
    let mut switched = language.is_none();
    let mut sought = false;
    loop {
        let presentation = player.present()?;
        if presentation.finished || started.elapsed() > Duration::from_secs(10) {
            break;
        }
        if presentation.frame.is_some() {
            presented += 1;
        }
        if !switched && started.elapsed() > Duration::from_secs(5) {
            let language = language.as_deref().unwrap_or_default();
            player.select_audio_language(language)?;
            println!(
                "Switched to the {language} audio track at {:.2} s",
                presentation.time.as_secs_f64()
            );
            switched = true;
        }
        if !sought && started.elapsed() > Duration::from_secs(3) {
            // Jump back, as a seek bar would. A seek decodes from the key frame
            // before its target, and the bundled sample has only one, so this
            // stays near the start.
            player.seek(Duration::from_millis(500))?;
            println!("Sought back to 0.5 s");
            sought = true;
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    player.pause()?;
    println!(
        "Presented {presented} frames holding at most {} compressed bytes: {} video, {} audio",
        player.options().video_budget_bytes + player.options().audio_budget_bytes,
        player.video_resident_bytes(),
        player.audio_resident_bytes()
    );
    Ok(())
}
