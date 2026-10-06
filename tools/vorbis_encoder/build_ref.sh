#!/bin/sh
# Builds the C reference encoders the Vorbis encoder port is checked against,
# into target/vorbis-ref/.
#
#   VORBIS=path/to/libvorbis-1.3.7 OGG=path/to/libogg-1.3.5 \
#     sh tools/vorbis_encoder/build_ref.sh
#
#   ref_enc.exe        unmodified libvorbis 1.3.7 + libogg 1.3.5 as gcc builds it
#                          on this host. On Windows (_WIN32) libvorbis' os.h replaces
#                          rint(x) by floor((x)+0.5f) (round-half-up).
#   ref_enc_posix.exe  the same sources with two minimal patches that make the
#                          Windows/mingw build behave like libvorbis on LP64 POSIX
#                          systems (Linux, macOS, BSD), which is what the Rust port
#                          reproduces (see src/vorbis_encoder/os.rs):
#                          1. os.h: the Windows-only `#define rint(x) (floor((x)+0.5f))`
#                             is removed, so rint() is C99 rint (round-half-even);
#                          2. scales.h: toBARK's `(n)*(n)` is evaluated in 64 bits
#                             (`(long long)(n)*(n)`), as with LP64 `long`; with the
#                             32-bit Windows `long` it overflows for sample rates above
#                             ~92.7 kHz.
#
# Flags: -O2, no -ffast-math, no -march (x86-64 SSE2 baseline, no FMA contraction).
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
HERE=$(cd "$TOOLS/../.." && pwd)
: "${VORBIS:?set VORBIS to an unpacked libvorbis-1.3.7 source release}"
: "${OGG:?set OGG to an unpacked libogg-1.3.5 source release}"
OUT=$HERE/target/vorbis-ref
mkdir -p "$OUT/posix_lib" "$OUT/include/ogg"
# libogg's configure normally writes this; the build needs nothing but the
# fixed-width typedefs.
cat > "$OUT/include/ogg/config_types.h" <<'EOF'
#ifndef __CONFIG_TYPES_H__
#define __CONFIG_TYPES_H__
#include <stdint.h>
typedef int16_t ogg_int16_t;
typedef uint16_t ogg_uint16_t;
typedef int32_t ogg_int32_t;
typedef uint32_t ogg_uint32_t;
typedef int64_t ogg_int64_t;
typedef uint64_t ogg_uint64_t;
#endif
EOF
SRCS=""
for f in "$VORBIS"/lib/*.c; do
  case "$(basename "$f")" in
    psytune.c|barkmel.c|tone.c|vorbisfile.c) ;;
    *) SRCS="$SRCS $f" ;;
  esac
done
CFLAGS="-O2 -std=gnu99 -w"
INC="-I$OUT/include -I$OGG/include -I$VORBIS/include -I$VORBIS/lib"
gcc $CFLAGS $INC -o "$OUT/ref_enc.exe" "$TOOLS/ref_enc.c" $SRCS \
    "$OGG/src/bitwise.c" "$OGG/src/framing.c" -lm
echo "built $OUT/ref_enc.exe"

# POSIX-rint variant: copy lib/ (including modes/ and books/) and patch os.h.
rm -rf "$OUT/posix_lib"
cp -r "$VORBIS/lib" "$OUT/posix_lib"
sed -i 's|^#  define rint(x)   (floor((x)+0.5f))$|/* rint define removed by build_ref.sh */|' "$OUT/posix_lib/os.h"
grep -q "rint define removed" "$OUT/posix_lib/os.h" || { echo "os.h patch failed"; exit 1; }
sed -i 's|atan((n)\*(n)\*1.85e-8f)|atan((long long)(n)*(n)*1.85e-8f)|' "$OUT/posix_lib/scales.h"
grep -q "atan((long long)(n)\*(n)\*1.85e-8f)" "$OUT/posix_lib/scales.h" || { echo "scales.h patch failed"; exit 1; }
PSRCS=""
for f in $SRCS; do PSRCS="$PSRCS $OUT/posix_lib/$(basename "$f")"; done
INC2="-I$OUT/include -I$OGG/include -I$VORBIS/include -I$OUT/posix_lib"
gcc $CFLAGS $INC2 -o "$OUT/ref_enc_posix.exe" "$TOOLS/ref_enc.c" $PSRCS \
    "$OGG/src/bitwise.c" "$OGG/src/framing.c" -lm
echo "built $OUT/ref_enc_posix.exe"

# bitrate -> quality reference table (used to build the unit-test fixture in
# src/vorbis_encoder/test_fixtures.rs)
BSRCS=""
for f in $PSRCS; do case "$(basename "$f")" in vorbisenc.c) ;; *) BSRCS="$BSRCS $f";; esac; done
gcc $CFLAGS $INC2 -o "$OUT/ref_bitrate.exe" "$TOOLS/ref_bitrate.c" $BSRCS \
    "$OGG/src/bitwise.c" "$OGG/src/framing.c" -lm
echo "built $OUT/ref_bitrate.exe"
