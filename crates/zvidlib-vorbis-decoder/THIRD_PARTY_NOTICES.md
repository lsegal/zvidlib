# Third-party notices

The parts of `zvidlib-vorbis-decoder` derived from third-party code, as zvidlib's own `THIRD_PARTY_NOTICES.md` lists them for the whole library.

`crates/zvidlib-vorbis-decoder/src/vorbis_decoder/` is the Vorbis decoder of
[Symphonia](https://github.com/pdeljanov/Symphonia), `symphonia-codec-vorbis`
0.5.5, modified by zvidlib to decode streams of more than two channels
correctly and in the Vorbis channel order; the changes are listed at the top of
`crates/zvidlib-vorbis-decoder/src/vorbis_decoder/mod.rs`. `crates/zvidlib-vorbis-decoder/src/vorbis_simd/kernels.rs` and
`crates/zvidlib-vorbis-decoder/src/vorbis_simd/imdct.rs` restructure its synthesis loops and
`symphonia-core` 0.5.5's inverse MDCT for runtime-dispatched SIMD. Those files
are covered by the Mozilla Public License, version 2.0, and keep its notice. A copy of the license is at
<https://mozilla.org/MPL/2.0/>, and their source, as modified, is the files
themselves.

Copyright (c) 2019-2022 The Project Symphonia Developers.
