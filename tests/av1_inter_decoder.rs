use zvidlib::{Av1InterDecoder, ErrorKind, FrameDigest, Limits};

fn vector() -> Vec<u8> {
    let hex = include_str!("fixtures/codec/av1_inter_show_existing_16x16.hex").trim();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn temporal_units(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut cursor = 0usize;
    while cursor < stream.len() {
        let start = cursor;
        let header = stream[cursor];
        cursor += 1;
        let obu_type = (header >> 3) & 0x0f;
        assert_ne!(header & 0x02, 0, "fixture OBU must carry a size field");
        let mut payload_len = 0usize;
        let mut shift = 0usize;
        loop {
            let byte = stream[cursor];
            cursor += 1;
            payload_len |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        cursor += payload_len;
        assert!(cursor <= stream.len(), "fixture OBU length is in bounds");
        if obu_type == 2 {
            starts.push(start);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = starts.get(index + 1).copied().unwrap_or(stream.len());
            &stream[start..end]
        })
        .collect()
}

#[test]
fn standardized_inter_and_show_existing_vector_matches_canonical_digest() {
    // This low-overhead OBU sequence is generated from the normative AV1
    // syntax tables and independently decoded by FFmpeg/libdav1d. It contains
    // a key frame, two refreshed inter references, LAST/LAST2 average compound
    // prediction, and a show-existing header for the retained compound frame.
    let stream = vector();
    let units = temporal_units(&stream);
    assert_eq!(units.len(), 5);
    let expected = FrameDigest::from_hex(
        include_str!("fixtures/codec/av1_inter_show_existing_16x16.sha256").trim(),
    )
    .unwrap();

    let mut decoder = Av1InterDecoder::new(Limits::default()).unwrap();
    let mut outputs = Vec::new();
    for unit in &units {
        outputs.push(decoder.decode_temporal_unit(unit).unwrap());
    }
    assert_eq!(FrameDigest::from_frame(&outputs[3]).unwrap(), expected);
    assert_eq!(FrameDigest::from_frame(&outputs[4]).unwrap(), expected);

    decoder.reset();
    for unit in &units[..4] {
        decoder.decode_temporal_unit(unit).unwrap();
    }
    let replay = decoder.decode_temporal_unit(units[4]).unwrap();
    assert_eq!(FrameDigest::from_frame(&replay).unwrap(), expected);
}

#[test]
fn retained_reference_memory_is_bounded_by_limits() {
    let stream = vector();
    let units = temporal_units(&stream);
    let limits = Limits {
        max_allocation_bytes: 1_024,
        ..Limits::default()
    };
    let mut decoder = Av1InterDecoder::new(limits).unwrap();
    assert_eq!(
        decoder.decode_temporal_unit(units[0]).unwrap_err().kind(),
        ErrorKind::ResourceLimit
    );
}
