//! The preview tier over a VP8 track with hidden alternate references
//! (#543). The rest of the VP8 exact-frame checks are `zvidlib-vp8`'s own, in
//! `crates/zvidlib-vp8/tests/vp8_conformance.rs`; this one stays with the root
//! package because [`zvidlib::PreviewIndex`] is the root package's (#612).
#![cfg(not(target_arch = "wasm32"))]

use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, HardwarePreference, Limits, PixelFormat, PreviewIndex,
    PreviewOptions, VideoDecoderConfig, VideoDimensions, WebmDemuxer,
    native_vp8_video_decoder_factory,
};

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn configuration(dimensions: VideoDimensions) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Vp8,
        profile: CodecProfile::Vp8,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: Vec::new(),
    }
}

#[test]
fn a_preview_index_covers_only_the_shown_frames() {
    // Issue #543: the three hidden alternate references are decode-only
    // samples, so the index plans its slots over the 40 shown frames, and
    // every slot it plans is one a decode fills.
    let source = MemorySource::new(
        include_bytes!("../crates/zvidlib-vp8/tests/fixtures/vp8/vp8_altref_98x66.webm").to_vec(),
    );
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    let samples = block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap();
    assert_eq!(samples.len(), 43);
    // One preview per frame, so a slot planned for a hidden sample's identity
    // would show up in the total rather than vanish into a wider stride.
    let options = PreviewOptions {
        previews_per_second: 30,
        ..PreviewOptions::for_frame_rate(30)
    };
    let index = PreviewIndex::with_frame_count(
        &native_vp8_video_decoder_factory(),
        configuration(track.dimensions.unwrap()),
        samples,
        track.presentation_order.len() as u64,
        Limits::default(),
        options,
    )
    .unwrap();
    assert_eq!(index.store().stride(), 1);
    index.wait_for_coverage();
    assert_eq!(index.coverage(), (40, 40));
}
