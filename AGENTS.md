# Agent instructions

## Testing

Do not write tests that check the existence or the correctness of support
files: READMEs, LICENSE and NOTICE files, Markdown files and other docs,
changelogs, and configuration files (Cargo manifests, CI workflows,
`rust-toolchain.toml`, `rustfmt.toml`, and similar). Tests exercise the
library, its examples and its scripts' behavior, not the repository's
documentation or configuration.
