//! Decoded-picture sample storage — the §8 reconstruction target.
//!
//! A [`Picture`] holds the three reconstructed sample planes (luma `SL`
//! and the two chroma planes `SCb` / `SCr`) sized from the active SPS
//! geometry and `ChromaArrayType`. Reconstruction (§8.4 intra / §8.5
//! inter sample prediction, §8.6 residual add, §8.7 in-loop filters)
//! writes into these planes; the DPB and any output cropping read out of
//! them.
//!
//! Samples are stored one `i32` per pixel so the prediction + residual
//! arithmetic of §8.4.4 / §8.6.2 (which works in the full `i32` range
//! before the final `Clip1Y` / `Clip1C` clip) can write the clipped
//! result without an intermediate type change. The stored values are
//! always already clipped to `[0, (1 << bitDepth) − 1]`.

/// One reconstructed picture: the luma plane and (unless monochrome) the
/// two chroma planes, each row-major.
#[derive(Debug, Clone, Eq)]
pub struct Picture {
    /// `pic_width_in_luma_samples`.
    width_luma: usize,
    /// `pic_height_in_luma_samples`.
    height_luma: usize,
    /// Chroma plane width (`width_luma >> SubWidthC`), 0 if monochrome.
    width_chroma: usize,
    /// Chroma plane height (`height_luma >> SubHeightC`), 0 if monochrome.
    height_chroma: usize,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    chroma_array_type: u8,
    /// `BitDepthY`.
    bit_depth_luma: u8,
    /// `BitDepthC`.
    bit_depth_chroma: u8,
    /// `SL[ x ][ y ]`, row-major (`luma[y * width_luma + x]`).
    luma: Vec<i32>,
    /// `SCb[ x ][ y ]`, row-major; empty when monochrome.
    cb: Vec<i32>,
    /// `SCr[ x ][ y ]`, row-major; empty when monochrome.
    cr: Vec<i32>,
    /// The eight-bit `i16` mirror of the three planes, in
    /// [`Plane`] order, built by [`Picture::build_narrow_mirror`] once
    /// the picture is finished and about to be used as a reference.
    ///
    /// This is a *representation* of the planes above, not a second set
    /// of samples: element `i` of each entry always equals element `i`
    /// of the plane it mirrors. It exists so the §8.5.3.3.3 16-bit
    /// interpolation path can borrow narrow tap windows out of a
    /// reference plane instead of materializing them per block (issue
    /// #427); every mutator drops it, so a picture that is still being
    /// reconstructed simply has none and the interpolation falls back to
    /// what it did before.
    narrow: Option<[Vec<i16>; 3]>,
}

/// Two pictures are equal when their geometry and their samples are,
/// which is what every caller means by it. The `i16` mirror is a
/// representation of the same samples — built or dropped by when a
/// picture last became a reference, not by what it contains — so it is
/// deliberately not part of the comparison.
impl PartialEq for Picture {
    fn eq(&self, other: &Self) -> bool {
        self.width_luma == other.width_luma
            && self.height_luma == other.height_luma
            && self.width_chroma == other.width_chroma
            && self.height_chroma == other.height_chroma
            && self.chroma_array_type == other.chroma_array_type
            && self.bit_depth_luma == other.bit_depth_luma
            && self.bit_depth_chroma == other.bit_depth_chroma
            && self.luma == other.luma
            && self.cb == other.cb
            && self.cr == other.cr
    }
}

/// The three colour components addressable in a [`Picture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// The luma plane `SL`.
    Luma,
    /// The first chroma plane `SCb`.
    Cb,
    /// The second chroma plane `SCr`.
    Cr,
}

/// `(SubWidthC, SubHeightC)` from Table 6-1 for a `ChromaArrayType`.
/// `ChromaArrayType == 0` (monochrome) has no chroma planes; this
/// returns `(2, 2)` for it purely so callers that never touch chroma do
/// not special-case it (the chroma planes are zero-sized in that case).
#[must_use]
pub fn sub_wh_c(chroma_array_type: u8) -> (usize, usize) {
    match chroma_array_type {
        // 4:2:0.
        1 => (2, 2),
        // 4:2:2.
        2 => (2, 1),
        // 4:4:4.
        3 => (1, 1),
        // monochrome (no chroma) — never indexed.
        _ => (2, 2),
    }
}

impl Picture {
    /// Allocate a black (all-zero) picture of the given geometry. The
    /// chroma planes are sized by `chroma_array_type` per Table 6-1; a
    /// monochrome picture (`chroma_array_type == 0`) carries no chroma
    /// samples.
    #[must_use]
    pub fn new(
        width_luma: usize,
        height_luma: usize,
        chroma_array_type: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
    ) -> Self {
        let (width_chroma, height_chroma) = if chroma_array_type == 0 {
            (0, 0)
        } else {
            let (sw, sh) = sub_wh_c(chroma_array_type);
            (width_luma / sw, height_luma / sh)
        };
        let luma = vec![0i32; width_luma * height_luma];
        let cb = vec![0i32; width_chroma * height_chroma];
        let cr = vec![0i32; width_chroma * height_chroma];
        Self {
            width_luma,
            height_luma,
            width_chroma,
            height_chroma,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            luma,
            cb,
            cr,
            narrow: None,
        }
    }

    /// `pic_width_in_luma_samples`.
    #[inline]
    #[must_use]
    pub fn width_luma(&self) -> usize {
        self.width_luma
    }

    /// `pic_height_in_luma_samples`.
    #[inline]
    #[must_use]
    pub fn height_luma(&self) -> usize {
        self.height_luma
    }

    /// `ChromaArrayType`.
    #[inline]
    #[must_use]
    pub fn chroma_array_type(&self) -> u8 {
        self.chroma_array_type
    }

    /// `BitDepthY`.
    #[inline]
    #[must_use]
    pub fn bit_depth_luma(&self) -> u8 {
        self.bit_depth_luma
    }

    /// `BitDepthC`.
    #[inline]
    #[must_use]
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma
    }

    /// Plane dimensions `(width, height)` in samples for `plane`.
    #[must_use]
    pub fn plane_dims(&self, plane: Plane) -> (usize, usize) {
        match plane {
            Plane::Luma => (self.width_luma, self.height_luma),
            Plane::Cb | Plane::Cr => (self.width_chroma, self.height_chroma),
        }
    }

    /// `BitDepth` of `plane`.
    #[must_use]
    pub fn bit_depth(&self, plane: Plane) -> u8 {
        match plane {
            Plane::Luma => self.bit_depth_luma,
            Plane::Cb | Plane::Cr => self.bit_depth_chroma,
        }
    }

    #[inline]
    fn plane_slice(&self, plane: Plane) -> (&[i32], usize) {
        match plane {
            Plane::Luma => (&self.luma, self.width_luma),
            Plane::Cb => (&self.cb, self.width_chroma),
            Plane::Cr => (&self.cr, self.width_chroma),
        }
    }

    #[inline]
    fn plane_slice_mut(&mut self, plane: Plane) -> (&mut [i32], usize) {
        match plane {
            Plane::Luma => (&mut self.luma, self.width_luma),
            Plane::Cb => (&mut self.cb, self.width_chroma),
            Plane::Cr => (&mut self.cr, self.width_chroma),
        }
    }

    /// Read one sample at `(x, y)` of `plane`.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the plane.
    #[must_use]
    pub fn sample(&self, plane: Plane, x: usize, y: usize) -> i32 {
        let (buf, stride) = self.plane_slice(plane);
        buf[y * stride + x]
    }

    /// Write one sample at `(x, y)` of `plane`.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the plane.
    pub fn set_sample(&mut self, plane: Plane, x: usize, y: usize, v: i32) {
        self.narrow = None;
        let (buf, stride) = self.plane_slice_mut(plane);
        buf[y * stride + x] = v;
    }

    /// Build the eight-bit `i16` mirror of the three planes, so a
    /// §8.5.3.3.3 reference plane can hand the 16-bit interpolation path
    /// its tap windows without materializing them
    /// ([`crate::hevc::engine::inter_pred::RefPlane::with_narrow`]).
    ///
    /// Call it once, on a finished picture, at the point it becomes a
    /// reference — the decoder does so on the way into the DPB. Every
    /// mutator drops the mirror again, so it can never describe samples
    /// the picture no longer holds; rebuilding it costs one pass over
    /// the planes.
    ///
    /// A no-op unless every plane the picture carries is eight-bit,
    /// which is the only range the `i16` tap accumulator of
    /// `simd::filter_taps_narrow` stays inside: at nine bits `88 · 511`
    /// leaves `i16` and the higher-bit-depth paths keep the `i32`
    /// kernel.
    pub fn build_narrow_mirror(&mut self) {
        if self.bit_depth_luma != 8 || (self.chroma_array_type != 0 && self.bit_depth_chroma != 8) {
            self.narrow = None;
            return;
        }
        let narrow_of = |src: &[i32]| {
            debug_assert!(
                src.iter().all(|&s| (0..=255).contains(&s)),
                "an eight-bit picture holds a sample outside [0, 255]"
            );
            src.iter().map(|&s| s as i16).collect::<Vec<i16>>()
        };
        self.narrow = Some([
            narrow_of(&self.luma),
            narrow_of(&self.cb),
            narrow_of(&self.cr),
        ]);
    }

    /// The [`Picture::build_narrow_mirror`] mirror of `plane`, or `None`
    /// when this picture carries none.
    #[inline]
    #[must_use]
    pub fn narrow_plane(&self, plane: Plane) -> Option<&[i16]> {
        let narrow = self.narrow.as_ref()?;
        Some(match plane {
            Plane::Luma => &narrow[0],
            Plane::Cb => &narrow[1],
            Plane::Cr => &narrow[2],
        })
    }

    /// Borrow the raw row-major plane buffer (read-only).
    #[must_use]
    pub fn plane(&self, plane: Plane) -> &[i32] {
        self.plane_slice(plane).0
    }

    /// Borrow the raw row-major plane buffer + its row stride (mutable).
    ///
    /// Used by the §8.7.2 deblocking driver to wrap a component plane in a
    /// [`crate::hevc::engine::deblock::SamplePlane`] for in-place edge filtering.
    pub fn plane_mut(&mut self, plane: Plane) -> (&mut [i32], usize) {
        self.narrow = None;
        self.plane_slice_mut(plane)
    }

    /// Pack the three planes into a single planar 8-bit buffer in
    /// `Y` then `Cb` then `Cr` order, each plane row-major. Only valid
    /// for `BitDepth == 8` planes (the common `yuv420p` / `yuv444p`
    /// fixture layout); samples are already clipped to `[0, 255]` by the
    /// reconstruction step so the cast is exact.
    ///
    /// Returns `None` if any plane has a bit depth other than 8.
    #[must_use]
    pub fn to_planar_u8(&self) -> Option<Vec<u8>> {
        if self.bit_depth_luma != 8 {
            return None;
        }
        if self.chroma_array_type != 0 && self.bit_depth_chroma != 8 {
            return None;
        }
        let mut out = Vec::with_capacity(self.luma.len() + self.cb.len() + self.cr.len());
        out.extend(self.luma.iter().map(|&v| v as u8));
        out.extend(self.cb.iter().map(|&v| v as u8));
        out.extend(self.cr.iter().map(|&v| v as u8));
        Some(out)
    }

    /// Pack the three planes into a single planar little-endian 16-bit
    /// buffer in `Y` then `Cb` then `Cr` order, each plane row-major —
    /// the `yuv420p10le` / `yuv422p10le` / `yuv444p10le` fixture layout
    /// for bit depths above 8. Samples are already clipped to
    /// `[0, (1 << BitDepth) − 1]` by the reconstruction step so the
    /// `u16` cast is exact.
    #[must_use]
    pub fn to_planar_le16(&self) -> Vec<u8> {
        let n = self.luma.len() + self.cb.len() + self.cr.len();
        let mut out = Vec::with_capacity(n * 2);
        for plane in [&self.luma, &self.cb, &self.cr] {
            for &v in plane.iter() {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        out
    }
}

/// `Clip3( 0, (1 << bitDepth) − 1, x )` — the §8 `Clip1Y` / `Clip1C`
/// sample clip.
#[inline]
#[must_use]
pub fn clip1(x: i32, bit_depth: u8) -> i32 {
    let max = (1i32 << bit_depth) - 1;
    x.clamp(0, max)
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn allocates_420_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 1, 8, 8);
        assert_eq!(p.plane_dims(Plane::Luma), (16, 16));
        assert_eq!(p.plane_dims(Plane::Cb), (8, 8));
        assert_eq!(p.plane_dims(Plane::Cr), (8, 8));
    }

    #[test]
    fn allocates_422_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 2, 8, 8);
        // SubWidthC = 2, SubHeightC = 1.
        assert_eq!(p.plane_dims(Plane::Cb), (8, 16));
    }

    #[test]
    fn allocates_444_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 3, 8, 8);
        assert_eq!(p.plane_dims(Plane::Cb), (16, 16));
    }

    #[test]
    fn monochrome_has_no_chroma() {
        let p = Picture::new(16, 16, 0, 8, 8);
        assert_eq!(p.plane_dims(Plane::Cb), (0, 0));
        assert!(p.plane(Plane::Cb).is_empty());
    }

    #[test]
    fn sample_roundtrips() {
        let mut p = Picture::new(8, 8, 1, 8, 8);
        p.set_sample(Plane::Luma, 3, 4, 200);
        assert_eq!(p.sample(Plane::Luma, 3, 4), 200);
        assert_eq!(p.sample(Plane::Luma, 0, 0), 0);
    }

    #[test]
    fn clip1_clamps_to_bit_depth_range() {
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(300, 8), 255);
        assert_eq!(clip1(81, 8), 81);
        assert_eq!(clip1(2000, 10), 1023);
    }

    #[test]
    fn planar_u8_packs_y_cb_cr() {
        let mut p = Picture::new(2, 2, 1, 8, 8);
        for y in 0..2 {
            for x in 0..2 {
                p.set_sample(Plane::Luma, x, y, 0x51);
            }
        }
        p.set_sample(Plane::Cb, 0, 0, 0x5a);
        p.set_sample(Plane::Cr, 0, 0, 0xf0);
        let packed = p.to_planar_u8().unwrap();
        // 4 luma + 1 cb + 1 cr.
        assert_eq!(packed.len(), 6);
        assert_eq!(&packed[0..4], &[0x51, 0x51, 0x51, 0x51]);
        assert_eq!(packed[4], 0x5a);
        assert_eq!(packed[5], 0xf0);
    }

    #[test]
    fn the_narrow_mirror_reproduces_every_plane() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        for y in 0..4 {
            for x in 0..4 {
                p.set_sample(Plane::Luma, x, y, (y * 4 + x) as i32 * 17);
            }
        }
        p.set_sample(Plane::Cb, 1, 1, 250);
        p.set_sample(Plane::Cr, 0, 1, 3);
        assert_eq!(p.narrow_plane(Plane::Luma), None);
        p.build_narrow_mirror();
        for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
            let wide = p.plane(plane);
            let narrow = p.narrow_plane(plane).expect("an eight-bit picture mirrors");
            assert_eq!(narrow.len(), wide.len());
            assert!(narrow.iter().zip(wide).all(|(&n, &w)| i32::from(n) == w));
        }
    }

    /// The mirror describes samples the picture holds, so anything that
    /// writes one drops it. A stale mirror would be read as reference
    /// samples by the §8.5.3.3.3 interpolation path.
    #[test]
    fn writing_a_sample_drops_the_narrow_mirror() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        p.build_narrow_mirror();
        assert!(p.narrow_plane(Plane::Luma).is_some());
        p.set_sample(Plane::Luma, 0, 0, 7);
        assert!(p.narrow_plane(Plane::Luma).is_none());

        p.build_narrow_mirror();
        assert!(p.narrow_plane(Plane::Luma).is_some());
        let _ = p.plane_mut(Plane::Cb);
        assert!(p.narrow_plane(Plane::Luma).is_none());
    }

    /// Above eight bits the `i16` tap accumulator the mirror exists for
    /// is out of range, so there is no mirror to build.
    #[test]
    fn a_higher_bit_depth_picture_has_no_narrow_mirror() {
        let mut p = Picture::new(4, 4, 1, 10, 10);
        p.build_narrow_mirror();
        assert!(p.narrow_plane(Plane::Luma).is_none());
    }

    /// Equality is about the samples, not about whether one of the two
    /// pictures happens to have been made a reference.
    #[test]
    fn the_narrow_mirror_does_not_affect_equality() {
        let mut a = Picture::new(4, 4, 1, 8, 8);
        a.set_sample(Plane::Luma, 2, 2, 99);
        let b = a.clone();
        a.build_narrow_mirror();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }
}
