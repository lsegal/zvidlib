//! Port of the encode side of floor1.c: `floor1_look`, `floor1_fit` (with
//! `accumulate_fit`, `fit_line`, `inspect_error`), `floor1_encode` and
//! `render_line0`.

use super::bitpack::OggPackBuffer;
use super::codebook::Codebook;
use super::os::{ilog, rint};
use super::tables::types::InfoFloor1;

const VIF_POSIT: usize = 63;

/// Port of `vorbis_look_floor1` (codec_internal.h).
#[derive(Debug, Clone)]
pub(crate) struct LookFloor1 {
    sorted_index: [i32; VIF_POSIT + 2],
    forward_index: [i32; VIF_POSIT + 2],
    reverse_index: [i32; VIF_POSIT + 2],
    hineighbor: [i32; VIF_POSIT],
    loneighbor: [i32; VIF_POSIT],
    posts: usize,
    n: i32,
    quant_q: i32,
    pub(crate) vi: InfoFloor1,
}

/// Port of `lsfit_acc` (floor1.c).
#[derive(Debug, Clone, Copy, Default)]
struct LsfitAcc {
    x0: i32,
    x1: i32,
    xa: i32,
    ya: i32,
    x2a: i32,
    y2a: i32,
    xya: i32,
    an: i32,
    xb: i32,
    yb: i32,
    x2b: i32,
    y2b: i32,
    xyb: i32,
    bn: i32,
}

impl LookFloor1 {
    /// Port of floor1.c `floor1_look`.
    pub(crate) fn new(info: &InfoFloor1) -> Self {
        let mut look = LookFloor1 {
            sorted_index: [0; VIF_POSIT + 2],
            forward_index: [0; VIF_POSIT + 2],
            reverse_index: [0; VIF_POSIT + 2],
            hineighbor: [0; VIF_POSIT],
            loneighbor: [0; VIF_POSIT],
            posts: 0,
            n: info.postlist[1],
            quant_q: 0,
            vi: *info,
        };
        let mut n = 0usize;
        for i in 0..info.partitions as usize {
            n += info.class_dim[info.partitionclass[i] as usize] as usize;
        }
        n += 2;
        look.posts = n;

        // also store a sorted position index (post values are distinct)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| info.postlist[i]);
        for (i, &o) in order.iter().enumerate() {
            look.forward_index[i] = o as i32;
        }
        for i in 0..n {
            look.reverse_index[look.forward_index[i] as usize] = i as i32;
        }
        for i in 0..n {
            look.sorted_index[i] = info.postlist[look.forward_index[i] as usize];
        }

        look.quant_q = match info.mult {
            1 => 256,
            2 => 128,
            3 => 86,
            4 => 64,
            _ => 0,
        };

        // discover our neighbors for decode where we don't use fit flags
        for i in 0..n.saturating_sub(2) {
            let mut lo = 0;
            let mut hi = 1;
            let mut lx = 0;
            let mut hx = look.n;
            let currentx = info.postlist[i + 2];
            for j in 0..i + 2 {
                let x = info.postlist[j];
                if x > lx && x < currentx {
                    lo = j;
                    lx = x;
                }
                if x < hx && x > currentx {
                    hi = j;
                    hx = x;
                }
            }
            look.loneighbor[i] = lo as i32;
            look.hineighbor[i] = hi as i32;
        }
        look
    }

    /// Port of floor1.c `floor1_fit`. Returns the fitted posts (with 0x8000
    /// marking posts that interpolation predicts), or `None` for an all-zero
    /// (unused) floor.
    pub(crate) fn fit(&self, logmdct: &[f32], logmask: &[f32]) -> Option<Vec<i32>> {
        let info = &self.vi;
        let n = self.n;
        let posts = self.posts;
        let mut nonzero = 0;
        let mut fits = [LsfitAcc::default(); VIF_POSIT + 1];
        let mut fit_value_a = [-200i32; VIF_POSIT + 2];
        let mut fit_value_b = [-200i32; VIF_POSIT + 2];
        let mut loneighbor = [0i32; VIF_POSIT + 2];
        let mut hineighbor = [1i32; VIF_POSIT + 2];
        let mut memo = [-1i32; VIF_POSIT + 2];

        // quantize the relevant floor points and collect them into line fit
        // structures (one per minimal division) at the same time
        if posts == 0 {
            nonzero += accumulate_fit(logmask, logmdct, 0, n, &mut fits[0], n, info);
        } else {
            for (i, fit) in fits.iter_mut().enumerate().take(posts - 1) {
                nonzero += accumulate_fit(
                    logmask,
                    logmdct,
                    self.sorted_index[i],
                    self.sorted_index[i + 1],
                    fit,
                    n,
                    info,
                );
            }
        }

        if nonzero == 0 {
            return None;
        }

        // start by fitting the implicit base case....
        let mut y0 = -200;
        let mut y1 = -200;
        fit_line(&fits[..posts - 1], &mut y0, &mut y1, info);

        fit_value_a[0] = y0;
        fit_value_b[0] = y0;
        fit_value_b[1] = y1;
        fit_value_a[1] = y1;

        // Non degenerate case: start progressive splitting. This is a greedy,
        // non-optimal algorithm, but simple and close enough to the best
        // answer.
        for i in 2..posts {
            let sortpos = self.reverse_index[i] as usize;
            let ln = loneighbor[sortpos] as usize;
            let hn = hineighbor[sortpos] as usize;

            // eliminate repeat searches of a particular range with a memo
            if memo[ln] != hn as i32 {
                // haven't performed this error search yet
                let lsortpos = self.reverse_index[ln] as usize;
                let hsortpos = self.reverse_index[hn] as usize;
                memo[ln] = hn as i32;

                // A note: we want to bound/minimize *local*, not global, error
                let lx = info.postlist[ln];
                let hx = info.postlist[hn];
                let ly = post_y(&fit_value_a, &fit_value_b, ln);
                let hy = post_y(&fit_value_a, &fit_value_b, hn);

                // (C calls exit(1) for ly==-1 || hy==-1, which cannot happen)
                if inspect_error(lx, hx, ly, hy, logmask, logmdct, info) {
                    // outside error bounds/begin search area. Split it.
                    let mut ly0 = -200;
                    let mut ly1 = -200;
                    let mut hy0 = -200;
                    let mut hy1 = -200;
                    let ret0 = fit_line(&fits[lsortpos..sortpos], &mut ly0, &mut ly1, info);
                    let ret1 = fit_line(&fits[sortpos..hsortpos], &mut hy0, &mut hy1, info);

                    if ret0 {
                        ly0 = ly;
                        ly1 = hy0;
                    }
                    if ret1 {
                        hy0 = ly1;
                        hy1 = hy;
                    }

                    if ret0 && ret1 {
                        fit_value_a[i] = -200;
                        fit_value_b[i] = -200;
                    } else {
                        // store new edge values
                        fit_value_b[ln] = ly0;
                        if ln == 0 {
                            fit_value_a[ln] = ly0;
                        }
                        fit_value_a[i] = ly1;
                        fit_value_b[i] = hy0;
                        fit_value_a[hn] = hy1;
                        if hn == 1 {
                            fit_value_b[hn] = hy1;
                        }

                        if ly1 >= 0 || hy0 >= 0 {
                            // store new neighbor values
                            for j in (0..sortpos).rev() {
                                if hineighbor[j] == hn as i32 {
                                    hineighbor[j] = i as i32;
                                } else {
                                    break;
                                }
                            }
                            for lnb in loneighbor.iter_mut().take(posts).skip(sortpos + 1) {
                                if *lnb == ln as i32 {
                                    *lnb = i as i32;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    fit_value_a[i] = -200;
                    fit_value_b[i] = -200;
                }
            }
        }

        let mut output = vec![0i32; posts];
        output[0] = post_y(&fit_value_a, &fit_value_b, 0);
        output[1] = post_y(&fit_value_a, &fit_value_b, 1);

        // fill in posts marked as not using a fit; we will zero back out to
        // 'unused' when encoding them so long as curve interpolation doesn't
        // force them into use
        for i in 2..posts {
            let ln = self.loneighbor[i - 2] as usize;
            let hn = self.hineighbor[i - 2] as usize;
            let x0 = info.postlist[ln];
            let x1 = info.postlist[hn];
            let y0 = output[ln];
            let y1 = output[hn];

            let predicted = render_point(x0, x1, y0, y1, info.postlist[i]);
            let vx = post_y(&fit_value_a, &fit_value_b, i);

            if vx >= 0 && predicted != vx {
                output[i] = vx;
            } else {
                output[i] = predicted | 0x8000;
            }
        }
        Some(output)
    }

    /// Port of floor1.c `floor1_encode`. Writes the floor to `opb`, fills
    /// `ilogmask[..pcmend/2]` with the integer floor curve and returns whether
    /// the floor is nonzero.
    pub(crate) fn encode(
        &self,
        opb: &mut OggPackBuffer,
        books: &[Codebook],
        post: Option<&mut [i32]>,
        ilogmask: &mut [i32],
        half_n: usize,
    ) -> bool {
        let info = &self.vi;
        let posts = self.posts;
        let mut out = [0i32; VIF_POSIT + 2];

        let Some(post) = post else {
            opb.write(0, 1);
            for v in &mut ilogmask[..half_n] {
                *v = 0;
            }
            return false;
        };

        // quantize values to multiplier spec
        for p in post.iter_mut().take(posts) {
            let mut val = *p & 0x7fff;
            match info.mult {
                1 => val >>= 2, // 1024 -> 256
                2 => val >>= 3, // 1024 -> 128
                3 => val /= 12, // 1024 -> 86
                4 => val >>= 4, // 1024 -> 64
                _ => {}
            }
            *p = val | (*p & 0x8000);
        }

        out[0] = post[0];
        out[1] = post[1];

        // find prediction values for each post and subtract them
        for i in 2..posts {
            let ln = self.loneighbor[i - 2] as usize;
            let hn = self.hineighbor[i - 2] as usize;
            let x0 = info.postlist[ln];
            let x1 = info.postlist[hn];
            let y0 = post[ln];
            let y1 = post[hn];

            let predicted = render_point(x0, x1, y0, y1, info.postlist[i]);

            if (post[i] & 0x8000) != 0 || predicted == post[i] {
                post[i] = predicted | 0x8000; // in case there was roundoff jitter in interpolation
                out[i] = 0;
            } else {
                let headroom = if self.quant_q - predicted < predicted {
                    self.quant_q - predicted
                } else {
                    predicted
                };

                let mut val = post[i] - predicted;

                // at this point the 'deviation' value is in the range +/- max
                // range, but the real, unique range can always be mapped to
                // only [0-maxrange). So we want to wrap the deviation into
                // this limited range, but do it in the way that least screws
                // an essentially gaussian probability distribution.
                if val < 0 {
                    if val < -headroom {
                        val = headroom - val - 1;
                    } else {
                        val = -1 - (val << 1);
                    }
                } else if val >= headroom {
                    val += headroom;
                } else {
                    val <<= 1;
                }

                out[i] = val;
                post[ln] &= 0x7fff;
                post[hn] &= 0x7fff;
            }
        }

        // we have everything we need. pack it out
        // mark nontrivial floor
        opb.write(1, 1);

        // beginning/end post
        let qbits = ilog((self.quant_q - 1) as u32) as u32;
        opb.write(out[0] as u32, qbits);
        opb.write(out[1] as u32, qbits);

        // partition by partition
        let mut j = 2usize;
        for i in 0..info.partitions as usize {
            let class = info.partitionclass[i] as usize;
            let cdim = info.class_dim[class] as usize;
            let csubbits = info.class_subs[class];
            let csub = 1usize << csubbits;
            let mut bookas = [0usize; 8];
            let mut cval = 0i32;
            let mut cshift = 0;

            // generate the partition's first stage cascade value
            if csubbits != 0 {
                let mut maxval = [0i32; 8];
                for (k, mv) in maxval.iter_mut().enumerate().take(csub) {
                    let booknum = info.class_subbook[class][k];
                    *mv = if booknum < 0 {
                        1
                    } else {
                        books[booknum as usize].c.entries
                    };
                }
                for k in 0..cdim {
                    let val = out[j + k];
                    if let Some(l) = maxval[..csub].iter().position(|&mv| val < mv) {
                        bookas[k] = l;
                    }
                    cval |= (bookas[k] as i32) << cshift;
                    cshift += csubbits;
                }
                // write it
                books[info.class_book[class] as usize].encode(cval, opb);
            }

            // write post values
            for k in 0..cdim {
                let book = info.class_subbook[class][bookas[k]];
                if book >= 0 {
                    // hack to allow training with 'bad' books
                    let b = &books[book as usize];
                    if out[j + k] < b.entries {
                        b.encode(out[j + k], opb);
                    }
                }
            }
            j += cdim;
        }

        // generate quantized floor equivalent to what we'd unpack in decode
        // render the lines
        let mut hx = 0;
        let mut lx = 0;
        let mut ly = post[0] * info.mult;
        let n = half_n as i32;

        for j in 1..posts {
            let current = self.forward_index[j] as usize;
            let mut hy = post[current] & 0x7fff;
            if hy == post[current] {
                hy *= info.mult;
                hx = info.postlist[current];
                render_line0(n, lx, hx, ly, hy, ilogmask);
                lx = hx;
                ly = hy;
            }
        }
        for v in &mut ilogmask[hx as usize..half_n] {
            *v = ly; // be certain
        }
        true
    }
}

/// floor1.c `render_point`
fn render_point(x0: i32, x1: i32, y0: i32, y1: i32, x: i32) -> i32 {
    let y0 = y0 & 0x7fff; // mask off flag
    let y1 = y1 & 0x7fff;
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx;
    if dy < 0 { y0 - off } else { y0 + off }
}

/// floor1.c `vorbis_dBquant`
#[inline]
fn db_quant(x: f32) -> i32 {
    let i = (x * 7.314_285_8_f32 + 1023.5_f32) as i32;
    i.clamp(0, 1023)
}

/// floor1.c `render_line0`
fn render_line0(n: i32, x0: i32, x1: i32, y0: i32, y1: i32, d: &mut [i32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let mut ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let mut x = x0;
    let mut y = y0;
    let mut err = 0;

    ady -= (base * adx).abs();

    let n = n.min(x1);

    if x < n {
        d[x as usize] = y;
    }

    loop {
        x += 1;
        if x >= n {
            break;
        }
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        d[x as usize] = y;
    }
}

/// Port of floor1.c `accumulate_fit`. Returns the number of "a" points.
fn accumulate_fit(
    flr: &[f32],
    mdct: &[f32],
    x0: i32,
    x1: i32,
    a: &mut LsfitAcc,
    n: i32,
    info: &InfoFloor1,
) -> i32 {
    let (mut xa, mut ya, mut x2a, mut y2a, mut xya, mut na) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut xb, mut yb, mut x2b, mut y2b, mut xyb, mut nb) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);

    *a = LsfitAcc::default();
    a.x0 = x0;
    a.x1 = x1;
    let x1 = if x1 >= n { n - 1 } else { x1 };

    for i in x0..=x1 {
        let iu = i as usize;
        let quantized = db_quant(flr[iu]);
        if quantized != 0 {
            if mdct[iu] + info.twofitatten >= flr[iu] {
                xa += i;
                ya += quantized;
                x2a += i * i;
                y2a += quantized * quantized;
                xya += i * quantized;
                na += 1;
            } else {
                xb += i;
                yb += quantized;
                x2b += i * i;
                y2b += quantized * quantized;
                xyb += i * quantized;
                nb += 1;
            }
        }
    }

    a.xa = xa;
    a.ya = ya;
    a.x2a = x2a;
    a.y2a = y2a;
    a.xya = xya;
    a.an = na;

    a.xb = xb;
    a.yb = yb;
    a.x2b = x2b;
    a.y2b = y2b;
    a.xyb = xyb;
    a.bn = nb;

    na
}

/// Port of floor1.c `fit_line`. Returns `true` when the fit is degenerate
/// (C returns 1). (The C code also accumulates a `y2b` sum it never uses;
/// that dead accumulation is omitted.)
fn fit_line(a: &[LsfitAcc], y0: &mut i32, y1: &mut i32, info: &InfoFloor1) -> bool {
    let mut xb = 0f64;
    let mut yb = 0f64;
    let mut x2b = 0f64;
    let mut xyb = 0f64;
    let mut bn = 0f64;
    let x0 = a[0].x0;
    let x1 = a[a.len() - 1].x1;

    for ai in a {
        // C: (a[i].bn+a[i].an)*info->twofitweight/(a[i].an+1)+1.
        let weight = ((ai.bn + ai.an) as f32 * info.twofitweight / (ai.an + 1) as f32) as f64 + 1.;

        xb += f64::from(ai.xb) + f64::from(ai.xa) * weight;
        yb += f64::from(ai.yb) + f64::from(ai.ya) * weight;
        x2b += f64::from(ai.x2b) + f64::from(ai.x2a) * weight;
        xyb += f64::from(ai.xyb) + f64::from(ai.xya) * weight;
        bn += f64::from(ai.bn) + f64::from(ai.an) * weight;
    }

    if *y0 >= 0 {
        xb += f64::from(x0);
        yb += f64::from(*y0);
        x2b += f64::from(x0 * x0);
        xyb += f64::from(*y0 * x0);
        bn += 1.;
    }

    if *y1 >= 0 {
        xb += f64::from(x1);
        yb += f64::from(*y1);
        x2b += f64::from(x1 * x1);
        xyb += f64::from(*y1 * x1);
        bn += 1.;
    }

    let denom = bn * x2b - xb * xb;

    if denom > 0. {
        let a = (yb * x2b - xyb * xb) / denom;
        let b = (bn * xyb - xb * yb) / denom;
        *y0 = rint(a + b * f64::from(x0)) as i32;
        *y1 = rint(a + b * f64::from(x1)) as i32;

        // limit to our range!
        *y0 = (*y0).clamp(0, 1023);
        *y1 = (*y1).clamp(0, 1023);
        false
    } else {
        *y0 = 0;
        *y1 = 0;
        true
    }
}

/// Port of floor1.c `inspect_error`. Returns `true` when the segment needs
/// to be split.
fn inspect_error(
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    mask: &[f32],
    mdct: &[f32],
    info: &InfoFloor1,
) -> bool {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let mut ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let mut x = x0;
    let mut y = y0;
    let mut err = 0;
    let mut val = db_quant(mask[x as usize]);
    let mut n = 0;

    ady -= (base * adx).abs();

    let mut mse = y - val;
    mse *= mse;
    n += 1;
    if mdct[x as usize] + info.twofitatten >= mask[x as usize] {
        if y as f32 + info.maxover < val as f32 {
            return true;
        }
        if y as f32 - info.maxunder > val as f32 {
            return true;
        }
    }

    loop {
        x += 1;
        if x >= x1 {
            break;
        }
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }

        val = db_quant(mask[x as usize]);
        mse += (y - val) * (y - val);
        n += 1;
        if mdct[x as usize] + info.twofitatten >= mask[x as usize] && val != 0 {
            if y as f32 + info.maxover < val as f32 {
                return true;
            }
            if y as f32 - info.maxunder > val as f32 {
                return true;
            }
        }
    }

    if info.maxover * info.maxover / n as f32 > info.maxerr {
        return false;
    }
    if info.maxunder * info.maxunder / n as f32 > info.maxerr {
        return false;
    }
    if (mse / n) as f32 > info.maxerr {
        return true;
    }
    false
}

/// floor1.c `post_Y`
#[inline]
fn post_y(a: &[i32], b: &[i32], pos: usize) -> i32 {
    if a[pos] < 0 {
        return b[pos];
    }
    if b[pos] < 0 {
        return a[pos];
    }
    (a[pos] + b[pos]) >> 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_quant_constant_matches_c_literal() {
        assert_eq!(7.314_285_8_f32, "7.3142857".parse::<f32>().unwrap());
        assert_eq!(db_quant(0.0), 1023);
        assert_eq!(db_quant(-140.0), 0);
        assert_eq!(db_quant(-70.0), 511);
    }

    #[test]
    fn render_point_and_line() {
        assert_eq!(render_point(0, 10, 0, 100, 5), 50);
        assert_eq!(render_point(0, 10, 100, 0, 5), 50);
        let mut d = [0i32; 8];
        render_line0(8, 0, 8, 0, 7, &mut d);
        assert_eq!(d, [0, 0, 1, 2, 3, 4, 5, 6]);
    }
}
