use zvidlib::{
    ErrorKind, FilterFrame, FilterPlane, FrameDigest, Limits, LoopFilterParams, PixelFormat,
    TxSizeGrid, deblock_frame, decode_av1_lossless_intra, decode_av1_lossless_intra_with_tx_sizes,
};

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

/// End-to-end: decoding a real (`CodedLossless`) standardized vector
/// through [`decode_av1_lossless_intra_with_tx_sizes`] records a
/// [`TxSizeGrid`] that produces the exact same `deblock_frame` output as
/// `None` would. This is the spec-correct behavior for every stream this
/// decoder accepts: AV1 forces `TX_MODE_ONLY_4X4` whenever `CodedLossless`
/// is true, so a real decode from this decoder can never produce a
/// non-4x4 transform size to thread into the wide 8/14-tap filters — the
/// `av1_filters` unit tests already cover that selection logic directly
/// against a synthetic `TxSizeGrid`, independent of decoder support for
/// non-lossless streams.
#[test]
fn standardized_lossless_intra_vector_tx_size_grid_matches_narrow_only_filtering() {
    let limits = Limits::default();
    let (frame, grid) = decode_av1_lossless_intra_with_tx_sizes(&vector(), &limits).unwrap();
    assert_eq!(grid, TxSizeGrid::new(17, 9));

    let luma = frame.planes[0].data.clone();
    let mut with_grid = FilterFrame::new_monochrome(
        FilterPlane::from_samples(17, 9, luma.clone(), &limits).unwrap(),
    );
    let mut without_grid =
        FilterFrame::new_monochrome(FilterPlane::from_samples(17, 9, luma, &limits).unwrap());
    let params = LoopFilterParams {
        y_vertical_level: 30,
        y_horizontal_level: 30,
        u_level: 0,
        v_level: 0,
        sharpness: 0,
    };
    deblock_frame(&mut with_grid, &params, Some(&grid)).unwrap();
    deblock_frame(&mut without_grid, &params, None).unwrap();
    assert_eq!(with_grid, without_grid);

    // The plain (no-grid) entry point still returns just the frame,
    // matching every existing caller's signature.
    let plain = decode_av1_lossless_intra(&vector(), &limits).unwrap();
    assert_eq!(plain.planes[0].data, frame.planes[0].data);
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
