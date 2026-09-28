# Cover-art fixtures

`cover.jpg` and `cover.png` are 32x24 frames of FFmpeg's `testsrc` pattern,
used as MP4 cover art by `tests/mp4_cover_art.rs`:

```sh
ffmpeg -f lavfi -i testsrc=s=32x24 -frames:v 1 -q:v 5 cover.jpg
ffmpeg -f lavfi -i testsrc=s=32x24 -frames:v 1 cover.png
```
