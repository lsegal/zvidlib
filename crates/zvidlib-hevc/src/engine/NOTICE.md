# HEVC engine provenance

The decoder engine in this directory is derived from `oxideav-h265` commit
`0cd76e5a425829b8045d9057b0b2b787a87e54ed`, a clean-room pure-Rust HEVC
implementation by Karpelès Lab Inc. It is incorporated as source so zvidlib's
native decoder has no Cargo, native-library, operating-system codec, or
subprocess dependency.

The upstream project is <https://github.com/OxideAV/oxideav-h265>. Its MIT
license is reproduced in [`LICENSE`](LICENSE). Adapter code, resource-limit
enforcement, output reordering, and RGBA conversion live one directory above.
