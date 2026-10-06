/* ref_enc: C reference encoder built from unmodified libvorbis 1.3.7 + libogg 1.3.5.
 *
 * usage: ref_enc RATE CHANNELS QUALITY IN.f32 OUT.pkt [CHUNK_FRAMES]
 *
 * Reads interleaved little-endian f32 PCM, encodes with
 * vorbis_encode_init_vbr(rate, channels, quality) and an empty comment
 * block, feeding CHUNK_FRAMES (default 1024) frames per
 * vorbis_analysis_buffer/vorbis_analysis_wrote call, then writes every
 * packet to OUT.pkt as:
 *   file header: "VPKT" + u32 packet count placeholder (0)
 *   per packet : u32 length, i64 granulepos, u8 flags (1=bos,2=eos), bytes
 * (all little endian). Headers come first (granulepos 0).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vorbis/codec.h>
#include <vorbis/vorbisenc.h>

static void put_u32(FILE *f, uint32_t v) {
  unsigned char b[4] = {v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255};
  fwrite(b, 1, 4, f);
}
static void put_i64(FILE *f, int64_t v) {
  uint64_t u = (uint64_t)v;
  put_u32(f, (uint32_t)(u & 0xffffffffu));
  put_u32(f, (uint32_t)(u >> 32));
}
static void put_packet(FILE *f, ogg_packet *op) {
  unsigned char flags = (op->b_o_s ? 1 : 0) | (op->e_o_s ? 2 : 0);
  put_u32(f, (uint32_t)op->bytes);
  put_i64(f, op->granulepos);
  fwrite(&flags, 1, 1, f);
  fwrite(op->packet, 1, op->bytes, f);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s RATE CHANNELS QUALITY IN.f32 OUT.pkt [CHUNK_FRAMES]\n", argv[0]);
    return 2;
  }
  long rate = atol(argv[1]);
  int channels = atoi(argv[2]);
  float quality = (float)atof(argv[3]);
  int chunk = argc > 6 ? atoi(argv[6]) : 1024;
  FILE *in = fopen(argv[4], "rb");
  FILE *out = fopen(argv[5], "wb");
  if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 1; }

  vorbis_info vi;
  vorbis_comment vc;
  vorbis_dsp_state vd;
  vorbis_block vb;
  ogg_packet op, opc, opcode;

  vorbis_info_init(&vi);
  int ret = vorbis_encode_init_vbr(&vi, channels, rate, quality);
  if (ret) { fprintf(stderr, "vorbis_encode_init_vbr failed: %d\n", ret); return 1; }
  vorbis_comment_init(&vc);
  vorbis_analysis_init(&vd, &vi);
  vorbis_block_init(&vd, &vb);

  fwrite("VPKT", 1, 4, out);
  put_u32(out, 0);

  vorbis_analysis_headerout(&vd, &vc, &op, &opc, &opcode);
  put_packet(out, &op);
  put_packet(out, &opc);
  put_packet(out, &opcode);

  float *buf = malloc(sizeof(float) * chunk * channels);
  int eos = 0;
  clock_t t0 = clock();
  while (!eos) {
    size_t got = fread(buf, sizeof(float) * channels, chunk, in);
    if (got == 0) {
      vorbis_analysis_wrote(&vd, 0);
    } else {
      float **b = vorbis_analysis_buffer(&vd, (int)got);
      for (size_t i = 0; i < got; i++)
        for (int c = 0; c < channels; c++) b[c][i] = buf[i * channels + c];
      vorbis_analysis_wrote(&vd, (int)got);
    }
    while (vorbis_analysis_blockout(&vd, &vb) == 1) {
      vorbis_analysis(&vb, NULL);
      vorbis_bitrate_addblock(&vb);
      while (vorbis_bitrate_flushpacket(&vd, &op)) {
        put_packet(out, &op);
        if (op.e_o_s) eos = 1;
      }
    }
  }
  fprintf(stderr, "encode_seconds=%.4f\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
  fprintf(stderr, "nominal=%ld blocksizes=%d/%d\n", vi.bitrate_nominal,
          vorbis_info_blocksize(&vi, 0), vorbis_info_blocksize(&vi, 1));
  vorbis_block_clear(&vb);
  vorbis_dsp_clear(&vd);
  vorbis_comment_clear(&vc);
  vorbis_info_clear(&vi);
  fclose(out);
  fclose(in);
  free(buf);
  return 0;
}
