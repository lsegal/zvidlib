# Third-party notices

The parts of `zvidlib-vp9-syntax` derived from third-party code, as zvidlib's own `THIRD_PARTY_NOTICES.md` lists them for the whole library.

The VP9 decoder's constant tables (`crates/zvidlib-vp9-decoder/src/vp9_dec/tables.rs`) and its
one-dimensional inverse transforms (`crates/zvidlib-vp9-decoder/src/vp9_dec/idct1d.rs`) are generated
from libvpx's `vp9/common` and `vpx_dsp` sources, and the rest of
`crates/zvidlib-vp9-decoder/src/vp9_dec/` follows the structure of libvpx's VP9 decoder, so that it
decodes bit for bit as libvpx does.

The VP9 default probability, quantizer and interpolation filter tables in
`crates/zvidlib-vp9-encoder/src/tables.rs` are generated from [libvpx](https://chromium.googlesource.com/webm/libvpx)
(`vp9/common/vp9_entropy.c`, `vp9_entropymode.c`, `vp9_quant_common.c` and
`vp9_filter.c`), and the 4x4 forward transform in `crates/zvidlib-vp9-encoder/src/dsp.rs`
follows its `vp9/encoder/vp9_dct.c`.

Copyright (c) 2010, The WebM Project authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google, nor the WebM Project, nor the names
    of its contributors may be used to endorse or promote products
    derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
