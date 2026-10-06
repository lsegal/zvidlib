/* ref_bitrate: prints libvorbis' bitrate -> setting mapping for a grid of
 * (rate, channels, nominal bitrate): runs vorbis_encode_setup_managed() with
 * only a nominal bitrate and reports the resulting base_setting and the VBR
 * quality at that setting (template quality_mapping interpolated at
 * base_setting). vorbisenc.c is #included to reach its static template types.
 * Built by tools/build_ref.sh against the POSIX-semantics copy of libvorbis. */
#include "vorbisenc.c"
#include <stdio.h>

int main(void) {
  static const long rates[] = {8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000};
  static const long brs[] = {8000, 16000, 24000, 32000, 45000, 48000, 64000, 80000,
                             96000, 112000, 128000, 160000, 192000, 256000, 320000, 500000};
  for (int ri = 0; ri < 8; ri++)
    for (int ch = 1; ch <= 2; ch++)
      for (int bi = 0; bi < 16; bi++) {
        vorbis_info vi;
        vorbis_info_init(&vi);
        int ret = vorbis_encode_setup_managed(&vi, ch, rates[ri], -1, brs[bi], -1);
        if (ret) {
          printf("%ld %d %ld none\n", rates[ri], ch, brs[bi]);
        } else {
          codec_setup_info *ci = vi.codec_setup;
          highlevel_encode_setup *hi = &ci->hi;
          const ve_setup_data_template *t = hi->setup;
          int is = hi->base_setting;
          double ds = hi->base_setting - is;
          double q = t->quality_mapping[is] * (1. - ds) + t->quality_mapping[is + 1] * ds;
          printf("%ld %d %ld %.17g %.9g\n", rates[ri], ch, brs[bi], hi->base_setting, (float)q);
        }
        vorbis_info_clear(&vi);
      }
  return 0;
}
