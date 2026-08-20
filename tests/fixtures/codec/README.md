# Codec conformance fixtures

`big_buck_bunny_hevc_rgba.sha256` contains canonical `FrameDigest` values for
all 120 presentation frames in the repository's existing
`examples/media/BigBuckBunny.mp4` HEVC Main sample. The reference RGBA frames
were decoded with FFmpeg 6.0, then fingerprinted with the dimensions, RGBA8
format, limited color range, plane count, and active pixels defined by
`FrameDigest::from_frame`.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.
