//! Header packet generation: info.c (`_vorbis_pack_info`,
//! `_vorbis_pack_comment`, `_vorbis_pack_books`) together with the backend
//! pack functions it dispatches to (floor1.c `floor1_pack`, res0.c
//! `res0_pack`, mapping0.c `mapping0_pack`).

use super::bitpack::OggPackBuffer;
use super::codebook::staticbook_pack;
use super::os::ilog;
use super::setup::CodecSetup;
use super::tables::types::{InfoFloor1, InfoMapping0, InfoResidue0};

/// info.c `_v_writestring`
fn write_string(opb: &mut OggPackBuffer, s: &[u8]) {
    for &b in s {
        opb.write(u32::from(b), 8);
    }
}

/// Port of info.c `_vorbis_pack_info` (identification header).
pub(crate) fn pack_info(ci: &CodecSetup) -> Vec<u8> {
    let mut opb = OggPackBuffer::new();
    opb.write(0x01, 8);
    write_string(&mut opb, b"vorbis");
    opb.write(0x00, 32);
    opb.write(ci.channels as u32, 8);
    opb.write(ci.rate as u32, 32);
    opb.write(ci.bitrate_upper as u32, 32);
    opb.write(ci.bitrate_nominal as u32, 32);
    opb.write(ci.bitrate_lower as u32, 32);
    opb.write(ilog((ci.blocksizes[0] - 1) as u32) as u32, 4);
    opb.write(ilog((ci.blocksizes[1] - 1) as u32) as u32, 4);
    opb.write(1, 1);
    opb.to_vec()
}

/// Port of info.c `_vorbis_pack_comment` with an empty comment list and the
/// given vendor string (libvorbis always writes its `ENCODE_VENDOR_STRING`).
pub(crate) fn pack_comment(vendor: &str) -> Vec<u8> {
    let mut opb = OggPackBuffer::new();
    opb.write(0x03, 8);
    write_string(&mut opb, b"vorbis");
    let v = vendor.as_bytes();
    opb.write(v.len() as u32, 32);
    write_string(&mut opb, v);
    opb.write(0, 32); // no user comments
    opb.write(1, 1);
    opb.to_vec()
}

/// Port of floor1.c `floor1_pack`.
fn floor1_pack(info: &InfoFloor1, opb: &mut OggPackBuffer) {
    let maxposit = info.postlist[1];
    let mut maxclass = -1;
    opb.write(info.partitions as u32, 5);
    for j in 0..info.partitions as usize {
        opb.write(info.partitionclass[j] as u32, 4);
        maxclass = maxclass.max(info.partitionclass[j]);
    }
    for j in 0..(maxclass + 1) as usize {
        opb.write((info.class_dim[j] - 1) as u32, 3);
        opb.write(info.class_subs[j] as u32, 2);
        if info.class_subs[j] != 0 {
            opb.write(info.class_book[j] as u32, 8);
        }
        for k in 0..(1usize << info.class_subs[j]) {
            opb.write((info.class_subbook[j][k] + 1) as u32, 8);
        }
    }
    opb.write((info.mult - 1) as u32, 2);
    let rangebits = ilog((maxposit - 1) as u32) as u32;
    opb.write(rangebits, 4);
    let mut count = 0;
    let mut k = 0;
    for j in 0..info.partitions as usize {
        count += info.class_dim[info.partitionclass[j] as usize];
        while k < count {
            opb.write(info.postlist[(k + 2) as usize] as u32, rangebits);
            k += 1;
        }
    }
}

/// Port of res0.c `res0_pack`.
fn res0_pack(info: &InfoResidue0, opb: &mut OggPackBuffer) {
    let mut acc = 0;
    opb.write(info.begin as u32, 24);
    opb.write(info.end as u32, 24);
    opb.write((info.grouping - 1) as u32, 24);
    opb.write((info.partitions - 1) as u32, 6);
    opb.write(info.groupbook as u32, 8);
    for j in 0..info.partitions as usize {
        let ss = info.secondstages[j] as u32;
        if ilog(ss) > 3 {
            opb.write(ss, 3);
            opb.write(1, 1);
            opb.write(ss >> 3, 5);
        } else {
            opb.write(ss, 4);
        }
        acc += ss.count_ones() as usize;
    }
    for j in 0..acc {
        opb.write(info.booklist[j] as u32, 8);
    }
}

/// Port of mapping0.c `mapping0_pack`.
fn mapping0_pack(channels: i32, info: &InfoMapping0, opb: &mut OggPackBuffer) {
    if info.submaps > 1 {
        opb.write(1, 1);
        opb.write((info.submaps - 1) as u32, 4);
    } else {
        opb.write(0, 1);
    }
    if info.coupling_steps > 0 {
        opb.write(1, 1);
        opb.write((info.coupling_steps - 1) as u32, 8);
        let bits = ilog((channels - 1) as u32) as u32;
        for i in 0..info.coupling_steps as usize {
            opb.write(info.coupling_mag[i] as u32, bits);
            opb.write(info.coupling_ang[i] as u32, bits);
        }
    } else {
        opb.write(0, 1);
    }
    opb.write(0, 2);
    if info.submaps > 1 {
        for i in 0..channels as usize {
            opb.write(info.chmuxlist[i] as u32, 4);
        }
    }
    for i in 0..info.submaps as usize {
        opb.write(0, 8);
        opb.write(info.floorsubmap[i] as u32, 8);
        opb.write(info.residuesubmap[i] as u32, 8);
    }
}

/// Port of info.c `_vorbis_pack_books` (setup header).
pub(crate) fn pack_books(ci: &CodecSetup) -> Vec<u8> {
    let mut opb = OggPackBuffer::new();
    opb.write(0x05, 8);
    write_string(&mut opb, b"vorbis");
    opb.write((ci.books.len() - 1) as u32, 8);
    for b in &ci.books {
        let ok = staticbook_pack(b, &mut opb);
        debug_assert!(ok);
    }
    // times; hook placeholders
    opb.write(0, 6);
    opb.write(0, 16);
    // floors (all type 1)
    opb.write((ci.floors.len() - 1) as u32, 6);
    for f in &ci.floors {
        opb.write(1, 16);
        floor1_pack(f, &mut opb);
    }
    // residues
    opb.write((ci.residues.len() - 1) as u32, 6);
    for (r, &t) in ci.residues.iter().zip(&ci.residue_types) {
        opb.write(t as u32, 16);
        res0_pack(r, &mut opb);
    }
    // maps (all type 0)
    opb.write((ci.maps.len() - 1) as u32, 6);
    for m in &ci.maps {
        opb.write(0, 16);
        mapping0_pack(ci.channels, m, &mut opb);
    }
    // modes
    opb.write((ci.modes.len() - 1) as u32, 6);
    for m in &ci.modes {
        opb.write(m.blockflag as u32, 1);
        opb.write(m.windowtype as u32, 16);
        opb.write(m.transformtype as u32, 16);
        opb.write(m.mapping as u32, 8);
    }
    opb.write(1, 1);
    opb.to_vec()
}
