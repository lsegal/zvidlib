# `zvidlib-vorbis-decoder` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`vorbis_stereo_44k.ogg` and `vorbis_transient_mono.ogg` are half-second Vorbis
streams encoded by libvorbis: music from the bundled sample at 44.1 kHz, and a
tone with sharp clicks at 48 kHz, which makes the encoder switch to short
blocks so packet durations vary. `vorbis_6ch_16k.ogg` is a quarter second of
5.1, one tone per channel, whose setup header has several coupling steps.
`vorbis_6ch_48k.ogg` and `vorbis_4ch_44k.ogg` are a quarter second of the same
music mixed differently into each channel, in two of the configurations
Symphonia 0.5.5 decoded wrongly (issue #547): at 48 kHz and `-q:a 4`
libvorbis's 5.1 mapping couples the left channel in three of its four steps,
and four channels share one uncoupled residue whose partitions do not fill its
last classword.

```sh
ffmpeg -ss 20 -t 0.5 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ac 2 \
  -ar 44100 -c:a libvorbis -q:a 4 vorbis_stereo_44k.ogg
ffmpeg -f lavfi \
  -i "sine=f=440:d=4,aeval=val(0)*0.3+0.6*(lt(mod(t\,0.5)\,0.003))*(random(0)-0.5)|val(0)*0.3:c=stereo" \
  -ar 48000 -f f32le transient.f32
ffmpeg -f f32le -ar 48000 -ac 2 -i transient.f32 -t 0.5 -ac 1 -c:a libvorbis -q:a 2 \
  vorbis_transient_mono.ogg
ffmpeg -f lavfi -i "aevalsrc=0.4*sin(2*PI*300*t)|0.4*sin(2*PI*500*t)|0.4*sin(2*PI*700*t)|0.4*sin(2*PI*900*t)|0.4*sin(2*PI*1100*t)|0.4*sin(2*PI*1300*t):s=16000:d=0.25:c=5.1" \
  -c:a libvorbis -q:a 0 vorbis_6ch_16k.ogg
ffmpeg -ss 20 -t 0.25 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ar 48000 \
  -af "aformat=channel_layouts=stereo,pan=5.1|c0=c0|c1=c1|c2=0.5*c0+0.5*c1|c3=0.3*c0+0.3*c1|c4=0.7*c0-0.3*c1|c5=0.3*c0-0.7*c1" \
  -c:a libvorbis -q:a 4 vorbis_6ch_48k.ogg
ffmpeg -ss 20 -t 0.25 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ar 44100 \
  -af "aformat=channel_layouts=stereo,pan=quad|c0=c0|c1=c1|c2=0.7*c0-0.3*c1|c3=0.3*c0-0.7*c1" \
  -c:a libvorbis -q:a 3 vorbis_4ch_44k.ogg
```

The references below interleave each frame's channels in the order the stream
codes them, which is the Vorbis channel order (Vorbis I section 4.3.9).

Each `vorbis_*_libvorbis.s16` is libvorbis's own decode through `vorbisfile`,
which trims the stream to its granule positions, written as interleaved 16-bit
little-endian PCM by a few lines of C built against libvorbis 1.3.7 and libogg
1.3.5:

```c
OggVorbis_File vf;
ov_fopen(argv[1], &vf);
int channels = ov_info(&vf, -1)->channels;
float **pcm;
int section;
long n;
while ((n = ov_read_float(&vf, &pcm, 4096, &section)) > 0)
  for (long i = 0; i < n; i++)
    for (int c = 0; c < channels; c++) {
      long s = lrintf(pcm[c][i] * 32768.f);
      short o = s > 32767 ? 32767 : s < -32768 ? -32768 : s;
      fwrite(&o, 2, 1, out);
    }
```
