use zvidlib::{ErrorKind, FrameDigest, Limits, PixelFormat, decode_av1_lossless_intra};

fn vector() -> Vec<u8> {
    let hex = include_str!("fixtures/codec/av1_lossless_17x9.hex").trim();
    assert_eq!(hex.len() & 1, 0);
    (0..hex.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
        .collect()
}

#[test]
fn standardized_lossless_intra_vector_reconstructs_canonical_yuv() {
    let frame = decode_av1_lossless_intra(&vector(), &Limits::default()).unwrap();
    assert_eq!((frame.dimensions.width, frame.dimensions.height), (17, 9));
    assert_eq!(frame.pixel_format, PixelFormat::Yuv420p8);
    assert_eq!(frame.planes[0].stride, 17);
    assert_eq!(frame.planes[1].stride, 9);
    assert_eq!(frame.planes[2].stride, 9);
    let expected_luma = (0..9u32)
        .flat_map(|y| (0..17u32).map(move |x| ((x * 11 + y * 17 + 29) & 0xff) as u8))
        .collect::<Vec<_>>();
    assert_eq!(frame.planes[0].data, expected_luma);
    assert!(frame.planes[1].data.iter().all(|&sample| sample == 128));
    assert!(frame.planes[2].data.iter().all(|&sample| sample == 128));
    let expected = FrameDigest::from_hex(
        include_str!("fixtures/codec/av1_lossless_17x9_yuv420.sha256").trim(),
    )
    .unwrap();
    assert_eq!(FrameDigest::from_frame(&frame).unwrap(), expected);
}

#[test]
fn malformed_and_over_budget_intra_units_fail_explicitly() {
    let bytes = vector();
    assert_eq!(
        decode_av1_lossless_intra(&bytes[..10], &Limits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::MalformedMedia
    );
    let limits = Limits {
        max_av1_blocks_per_frame: 1,
        ..Limits::default()
    };
    assert_eq!(
        decode_av1_lossless_intra(&bytes, &limits)
            .unwrap_err()
            .kind(),
        ErrorKind::ResourceLimit
    );

    let mut malformed_tile = bytes;
    malformed_tile[15..].fill(0);
    let result =
        std::panic::catch_unwind(|| decode_av1_lossless_intra(&malformed_tile, &Limits::default()));
    assert!(result.is_ok(), "malformed AV1 tile data must not panic");
    assert!(matches!(
        result.unwrap().unwrap_err().kind(),
        ErrorKind::MalformedMedia | ErrorKind::Unsupported | ErrorKind::ResourceLimit
    ));
}
