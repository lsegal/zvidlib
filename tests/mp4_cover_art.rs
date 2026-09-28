use std::future::Future;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    Codec, CoverArt, CoverArtFormat, EncodedSample, EncoderConfig, ErrorKind, Mp4Demuxer,
    Mp4DemuxerOptions, SampleDependency, VideoDimensions,
};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn codec_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn video_track() -> Mp4TrackConfig {
    Mp4TrackConfig {
        encoder: EncoderConfig {
            codec: Codec::Av1,
            timescale: 30_000,
            decoder_config: codec_box(b"av1C", &[0x81, 0, 0, 0]),
        },
        format: Mp4TrackFormat::Video(VideoDimensions {
            width: 64,
            height: 48,
        }),
    }
}

async fn write_samples(muxer: &mut Mp4Muxer<MemorySink>) {
    for index in 0..3_u8 {
        muxer
            .write_sample(
                0,
                EncodedSample {
                    data: vec![index + 1; 32 + usize::from(index)],
                    dts: i64::from(index) * 1_001,
                    pts: i64::from(index) * 1_001,
                    duration: 1_001,
                    is_sync: index == 0,
                    dependency: if index == 0 {
                        SampleDependency::INDEPENDENT
                    } else {
                        SampleDependency::DEPENDENT
                    },
                },
            )
            .await
            .unwrap();
    }
}

/// Muxes the fixture track, setting `cover_art` after every sample is written
/// the way a recorder picks a frame once capture ends.
fn mux(cover_art: Option<Option<CoverArt>>) -> Vec<u8> {
    block_on(async {
        let mut muxer = Mp4Muxer::new(MemorySink::new(), vec![video_track()], 16)
            .await
            .unwrap();
        write_samples(&mut muxer).await;
        if let Some(cover_art) = cover_art {
            muxer.set_cover_art(cover_art).unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    })
}

fn demux(bytes: Vec<u8>) -> Mp4Demuxer {
    block_on(Mp4Demuxer::open(
        &MemorySource::new(bytes),
        Mp4DemuxerOptions::default(),
    ))
    .unwrap()
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/cover_art/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn jpeg() -> CoverArt {
    CoverArt {
        format: CoverArtFormat::Jpeg,
        data: fixture("cover.jpg"),
    }
}

fn png() -> CoverArt {
    CoverArt {
        format: CoverArtFormat::Png,
        data: fixture("cover.png"),
    }
}

fn make_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    codec_box(kind, payload)
}

/// Returns the offset of the top-level `moov` box.
fn moov_offset(bytes: &[u8]) -> usize {
    let mut at = 0;
    while at + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let size = if size == 1 {
            u64::from_be_bytes(bytes[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if &bytes[at + 4..at + 8] == b"moov" {
            return at;
        }
        at += size;
    }
    panic!("no moov box");
}

/// Appends `child` to the end of the (final) `moov` box and fixes its size.
fn append_to_moov(bytes: &[u8], child: &[u8]) -> Vec<u8> {
    let moov = moov_offset(bytes);
    assert_eq!(
        moov + u32::from_be_bytes(bytes[moov..moov + 4].try_into().unwrap()) as usize,
        bytes.len(),
        "the muxer writes moov last"
    );
    let mut output = bytes.to_vec();
    output.extend_from_slice(child);
    let size = u32::try_from(output.len() - moov).unwrap();
    output[moov..moov + 4].copy_from_slice(&size.to_be_bytes());
    output
}

/// FNV-1a, enough to pin the exact bytes of a small fixture file.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn output_without_cover_art_is_byte_identical_to_earlier_releases() {
    // Pinned from the muxer before cover art existed.
    for bytes in [mux(None), mux(Some(None)), {
        // Setting and then clearing a cover leaves no trace.
        block_on(async {
            let mut muxer = Mp4Muxer::new(MemorySink::new(), vec![video_track()], 16)
                .await
                .unwrap();
            muxer.set_cover_art(Some(jpeg())).unwrap();
            write_samples(&mut muxer).await;
            muxer.set_cover_art(None).unwrap();
            muxer.finish().await.unwrap().into_inner()
        })
    }] {
        assert_eq!(bytes.len(), 790);
        assert_eq!(fnv1a(&bytes), 0xebdf_4e99_6bfe_4d33);
        assert_eq!(demux(bytes).cover_art, None);
    }
}

#[test]
fn jpeg_and_png_cover_art_round_trip_through_mux_and_demux() {
    let plain = mux(None);
    for cover_art in [jpeg(), png()] {
        let bytes = mux(Some(Some(cover_art.clone())));

        // Everything but the appended udta box is unchanged.
        let moov = moov_offset(&plain);
        assert_eq!(bytes[..moov], plain[..moov]);
        assert_eq!(bytes[moov + 4..plain.len()], plain[moov + 4..]);

        let data_type: u32 = match cover_art.format {
            CoverArtFormat::Jpeg => 13,
            CoverArtFormat::Png => 14,
        };
        let mut data = data_type.to_be_bytes().to_vec();
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&cover_art.data);
        let mut hdlr = vec![0; 8];
        hdlr.extend_from_slice(b"mdirappl");
        hdlr.extend_from_slice(&[0; 9]);
        let mut meta = vec![0; 4];
        meta.extend_from_slice(&make_box(b"hdlr", &hdlr));
        meta.extend_from_slice(&make_box(
            b"ilst",
            &make_box(b"covr", &make_box(b"data", &data)),
        ));
        let udta = make_box(b"udta", &make_box(b"meta", &meta));
        assert_eq!(bytes, append_to_moov(&plain, &udta));

        let demuxer = demux(bytes);
        assert_eq!(demuxer.cover_art, Some(cover_art));
        assert_eq!(demuxer.tracks.len(), 1);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
    }
}

#[test]
fn cover_art_can_be_set_before_any_sample_is_written() {
    let bytes = block_on(async {
        let mut muxer = Mp4Muxer::new(MemorySink::new(), vec![video_track()], 16)
            .await
            .unwrap();
        muxer.set_cover_art(Some(png())).unwrap();
        write_samples(&mut muxer).await;
        muxer.finish().await.unwrap().into_inner()
    });
    assert_eq!(bytes, mux(Some(Some(png()))));
}

#[test]
fn empty_cover_art_is_rejected() {
    block_on(async {
        let mut muxer = Mp4Muxer::new(MemorySink::new(), vec![video_track()], 16)
            .await
            .unwrap();
        let error = muxer
            .set_cover_art(Some(CoverArt {
                format: CoverArtFormat::Jpeg,
                data: Vec::new(),
            }))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    });
}

#[test]
fn demuxer_reads_quicktime_style_meta_and_skips_other_items() {
    let plain = mux(None);
    let mut other = 1_u32.to_be_bytes().to_vec();
    other.extend_from_slice(&[0; 4]);
    other.extend_from_slice(b"zvid");
    let mut bmp = 27_u32.to_be_bytes().to_vec();
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(b"BM");
    let mut png_data = 14_u32.to_be_bytes().to_vec();
    png_data.extend_from_slice(&[0; 4]);
    png_data.extend_from_slice(&png().data);
    let mut covr = make_box(b"data", &bmp);
    covr.extend_from_slice(&make_box(b"data", &png_data));
    let mut ilst = make_box(b"\xa9nam", &make_box(b"data", &other));
    ilst.extend_from_slice(&make_box(b"covr", &covr));
    // QuickTime writes meta as a plain container with no version and flags.
    let mut hdlr = vec![0; 8];
    hdlr.extend_from_slice(b"mdirappl");
    hdlr.extend_from_slice(&[0; 9]);
    let mut meta = make_box(b"hdlr", &hdlr);
    meta.extend_from_slice(&make_box(b"ilst", &ilst));
    let mut udta = make_box(b"\xa9too", b"zvid");
    udta.extend_from_slice(&make_box(b"meta", &meta));
    let bytes = append_to_moov(&plain, &make_box(b"udta", &udta));
    assert_eq!(demux(bytes).cover_art, Some(png()));
}

#[test]
fn malformed_user_data_does_not_prevent_opening() {
    let plain = mux(None);
    for udta in [
        // A child that claims to run past the end of udta.
        vec![0, 0, 0, 0x40, b'm', b'e', b't', b'a', 0, 0, 0, 0],
        // A data box too short to hold its type and locale.
        make_box(
            b"meta",
            &[
                vec![0; 4],
                make_box(b"ilst", &make_box(b"covr", &make_box(b"data", &[0, 0, 0, 13]))),
            ]
            .concat(),
        ),
    ] {
        let bytes = append_to_moov(&plain, &make_box(b"udta", &udta));
        let demuxer = demux(bytes);
        assert_eq!(demuxer.cover_art, None);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
    }
}

#[test]
fn ffprobe_reports_the_cover_as_an_attached_picture() {
    if Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        eprintln!("skipping the independent probe because ffprobe is unavailable");
        return;
    }
    for (cover_art, codec) in [(jpeg(), "mjpeg"), (png(), "png")] {
        let path = std::env::temp_dir().join(format!(
            "zvidlib-cover-art-{}-{codec}.mp4",
            std::process::id()
        ));
        std::fs::write(&path, mux(Some(Some(cover_art)))).unwrap();
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height:stream_disposition=attached_pic",
                "-of",
                "csv=p=0",
            ])
            .arg(&path)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&path);
        let report = String::from_utf8_lossy(&probe.stdout);
        assert!(probe.status.success(), "ffprobe failed: {report}");
        assert!(
            report
                .lines()
                .any(|line| line == format!("{codec},32,24,1")),
            "no {codec} attached picture in:\n{report}"
        );
    }
}
