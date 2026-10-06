//! Port of the encode side of res0.c: `res0_look`, the classifiers
//! (`_01class` for residue type 1, `_2class` for type 2) and the multi-stage
//! cascade VQ encoder (`_01forward`, `_encodepart`, `local_book_besterror`),
//! plus the `res1_*`/`res2_*` wrappers.

use super::bitpack::OggPackBuffer;
use super::codebook::Codebook;
use super::os::ilog;
use super::tables::types::InfoResidue0;

/// Port of `vorbis_look_residue0` (res0.c), encode members only.
#[derive(Debug, Clone)]
pub(crate) struct LookResidue0 {
    info: InfoResidue0,
    stages: usize,
    phrasebook: usize,
    /// [part][stage] -> codebook index
    partbooks: Vec<Vec<Option<usize>>>,
}

impl LookResidue0 {
    /// Port of res0.c `res0_look`.
    pub(crate) fn new(info: &InfoResidue0) -> Self {
        let parts = info.partitions as usize;
        let mut acc = 0usize;
        let mut maxstage = 0usize;
        let mut partbooks = vec![Vec::new(); parts];
        for (j, pb) in partbooks.iter_mut().enumerate() {
            let stages = ilog(info.secondstages[j] as u32) as usize;
            if stages != 0 {
                maxstage = maxstage.max(stages);
                *pb = vec![None; stages];
                for (k, slot) in pb.iter_mut().enumerate() {
                    if info.secondstages[j] & (1 << k) != 0 {
                        *slot = Some(info.booklist[acc] as usize);
                        acc += 1;
                    }
                }
            }
        }
        LookResidue0 {
            info: *info,
            stages: maxstage,
            phrasebook: info.groupbook as usize,
            partbooks,
        }
    }

    /// Port of res0.c `_01class` (residue 1 classification, per channel).
    fn class01(&self, input: &[&[i32]]) -> Vec<Vec<i32>> {
        let info = &self.info;
        let samples_per_partition = info.grouping as usize;
        let possible_partitions = info.partitions as usize;
        let n = (info.end - info.begin) as usize;
        let partvals = n / samples_per_partition;
        // C: float scale=100./samples_per_partition;
        let scale = (100. / samples_per_partition as f64) as f32;

        let mut partword = vec![vec![0i32; partvals]; input.len()];
        for i in 0..partvals {
            let offset = i * samples_per_partition + info.begin as usize;
            for (pw, inj) in partword.iter_mut().zip(input) {
                let mut max = 0;
                let mut ent = 0i32;
                for k in 0..samples_per_partition {
                    let a = inj[offset + k].wrapping_abs();
                    if a > max {
                        max = a;
                    }
                    ent = ent.wrapping_add(a);
                }
                // C: ent*=scale;  (int = int * float)
                ent = (ent as f32 * scale) as i32;

                let mut k = 0;
                while k < possible_partitions - 1 {
                    if max <= info.classmetric1[k]
                        && (info.classmetric2[k] < 0 || ent < info.classmetric2[k])
                    {
                        break;
                    }
                    k += 1;
                }
                pw[i] = k as i32;
            }
        }
        partword
    }

    /// Port of res0.c `_2class` (residue 2 classification over interleaved
    /// channels).
    fn class2(&self, input: &[&[i32]]) -> Vec<Vec<i32>> {
        let info = &self.info;
        let ch = input.len();
        let samples_per_partition = info.grouping as usize;
        let possible_partitions = info.partitions as usize;
        let n = (info.end - info.begin) as usize;
        let partvals = n / samples_per_partition;

        let mut partword = vec![0i32; partvals];
        let mut l = info.begin as usize / ch;
        for pw in partword.iter_mut() {
            let mut magmax = 0;
            let mut angmax = 0;
            let mut j = 0;
            while j < samples_per_partition {
                let a = input[0][l].wrapping_abs();
                if a > magmax {
                    magmax = a;
                }
                for inp in input.iter().skip(1) {
                    let a = inp[l].wrapping_abs();
                    if a > angmax {
                        angmax = a;
                    }
                }
                l += 1;
                j += ch;
            }

            let mut j = 0;
            while j < possible_partitions - 1 {
                if magmax <= info.classmetric1[j] && angmax <= info.classmetric2[j] {
                    break;
                }
                j += 1;
            }
            *pw = j as i32;
        }
        vec![partword]
    }

    /// Port of res0.c `_01forward`.
    fn forward01(
        &self,
        opb: &mut OggPackBuffer,
        books: &[Codebook],
        input: &mut [&mut [i32]],
        partword: &[Vec<i32>],
    ) {
        let info = &self.info;
        let samples_per_partition = info.grouping as usize;
        let possible_partitions = info.partitions;
        let phrasebook = &books[self.phrasebook];
        let partitions_per_word = phrasebook.dim as usize;
        let n = (info.end - info.begin) as usize;
        let partvals = n / samples_per_partition;

        // we code the partition words for each channel, then the residual
        // words for a partition per channel until we've written all the
        // residual words for that partition word. Then write the next
        // partition channel words...
        for s in 0..self.stages {
            let mut i = 0;
            while i < partvals {
                // first we encode a partition codeword for each channel
                if s == 0 {
                    for pw in partword {
                        let mut val = pw[i];
                        for k in 1..partitions_per_word {
                            val *= possible_partitions;
                            if i + k < partvals {
                                val += pw[i + k];
                            }
                        }
                        // training hack
                        if val < phrasebook.entries {
                            phrasebook.encode(val, opb);
                        }
                    }
                }

                // now we encode interleaved residual values for the partitions
                let mut k = 0;
                while k < partitions_per_word && i < partvals {
                    let offset = i * samples_per_partition + info.begin as usize;
                    for (j, pw) in partword.iter().enumerate() {
                        let part = pw[i] as usize;
                        if info.secondstages[part] & (1 << s) != 0
                            && let Some(Some(book)) = self.partbooks[part].get(s)
                        {
                            encodepart(
                                opb,
                                &mut input[j][offset..offset + samples_per_partition],
                                &books[*book],
                            );
                        }
                    }
                    k += 1;
                    i += 1;
                }
            }
        }
    }

    /// Port of res0.c `res1_class` + `res2_class`: classification for
    /// residue type `rtype` (1 or 2). Returns `None` where C returns NULL
    /// (no nonzero channel).
    pub(crate) fn class(
        &self,
        rtype: i32,
        input: &[&[i32]],
        nonzero: &[bool],
    ) -> Option<Vec<Vec<i32>>> {
        if rtype == 2 {
            if nonzero.iter().any(|&z| z) {
                Some(self.class2(input))
            } else {
                None
            }
        } else {
            let used: Vec<&[i32]> = input
                .iter()
                .zip(nonzero)
                .filter(|(_, z)| **z)
                .map(|(v, _)| *v)
                .collect();
            if used.is_empty() {
                None
            } else {
                Some(self.class01(&used))
            }
        }
    }

    /// Port of res0.c `res1_forward` / `res2_forward`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        rtype: i32,
        opb: &mut OggPackBuffer,
        books: &[Codebook],
        input: &mut [&mut [i32]],
        nonzero: &[bool],
        partword: Option<&[Vec<i32>]>,
        half_n: usize,
        work: &mut Vec<i32>,
    ) {
        let Some(partword) = partword else {
            return;
        };
        if rtype == 2 {
            // all the channels are interleaved into a single vector and
            // encoded as a single channel res1
            let ch = input.len();
            work.clear();
            work.resize(ch * half_n, 0);
            let mut used = 0;
            for (i, pcm) in input.iter().enumerate() {
                if nonzero[i] {
                    used += 1;
                }
                let mut k = i;
                for &v in pcm[..half_n].iter() {
                    work[k] = v;
                    k += ch;
                }
            }
            if used != 0 {
                let mut one: [&mut [i32]; 1] = [&mut work[..]];
                self.forward01(opb, books, &mut one, partword);
            }
        } else {
            let mut used: Vec<&mut [i32]> = input
                .iter_mut()
                .zip(nonzero)
                .filter(|(_, z)| **z)
                .map(|(v, _)| &mut **v)
                .collect();
            if !used.is_empty() {
                self.forward01(opb, books, &mut used, partword);
            }
        }
    }
}

/// Port of res0.c `local_book_besterror`: quantize `a` (book.dim values) to
/// the nearest lattice entry, subtract it from `a` and return the entry.
fn local_book_besterror(book: &Codebook, a: &mut [i32]) -> i32 {
    let dim = book.dim as usize;
    let minval = book.minval;
    let del = book.delta;
    let qv = book.quantvals;
    let ze = qv >> 1;
    let mut index = 0i32;
    // assumes integer/centered encoder codebook maptype 1 no more than dim 8
    let mut p = [0i32; 8];

    if del != 1 {
        for o in (0..dim).rev() {
            let v = a[o].wrapping_sub(minval).wrapping_add(del >> 1) / del;
            let m = if v < ze {
                (ze.wrapping_sub(v) << 1).wrapping_sub(1)
            } else {
                v.wrapping_sub(ze) << 1
            };
            index = index * qv + m.clamp(0, qv - 1);
            p[o] = v.wrapping_mul(del).wrapping_add(minval);
        }
    } else {
        for o in (0..dim).rev() {
            let v = a[o].wrapping_sub(minval);
            let m = if v < ze {
                (ze.wrapping_sub(v) << 1).wrapping_sub(1)
            } else {
                v.wrapping_sub(ze) << 1
            };
            index = index * qv + m.clamp(0, qv - 1);
            p[o] = v.wrapping_mul(del).wrapping_add(minval);
        }
    }

    if book.c.lengthlist[index as usize] == 0 {
        let c = book.c;
        let mut best = -1i32;
        // assumes integer/centered encoder codebook maptype 1 no more than dim 8
        let mut e = [0i32; 8];
        let maxval = book.minval + book.delta * (book.quantvals - 1);
        for i in 0..book.entries {
            if c.lengthlist[i as usize] > 0 {
                let mut this = 0i32;
                for j in 0..dim {
                    let val = e[j].wrapping_sub(a[j]);
                    this = this.wrapping_add(val.wrapping_mul(val));
                }
                if best == -1 || this < best {
                    p = e;
                    best = this;
                    index = i;
                }
            }
            // assumes the value patterning created by the tools in vq/
            let mut j = 0;
            while j < 8 && e[j] >= maxval {
                e[j] = 0;
                j += 1;
            }
            if j == 8 {
                // only reachable after the last entry of a dim-8 book (C reads
                // past the array here; the value is never used)
                break;
            }
            if e[j] >= 0 {
                e[j] += book.delta;
            }
            e[j] = -e[j];
        }
    }

    if index > -1 {
        for i in 0..dim {
            a[i] = a[i].wrapping_sub(p[i]);
        }
    }
    index
}

/// Port of res0.c `_encodepart`.
fn encodepart(opb: &mut OggPackBuffer, vec: &mut [i32], book: &Codebook) -> i32 {
    let dim = book.dim as usize;
    let step = vec.len() / dim;
    let mut bits = 0;
    for i in 0..step {
        let entry = local_book_besterror(book, &mut vec[i * dim..(i + 1) * dim]);
        bits += book.encode(entry, opb);
    }
    bits
}
