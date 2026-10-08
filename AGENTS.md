# Agent instructions

## Testing

Do not write tests that check the existence or the correctness of support
files: READMEs, LICENSE and NOTICE files, Markdown files and other docs,
changelogs, and configuration files (Cargo manifests, CI workflows,
`rust-toolchain.toml`, `rustfmt.toml`, and similar). Tests exercise the
library, its examples and its scripts' behavior, not the repository's
documentation or configuration.

## Spelling

Use American English everywhere: prose, code comments and doc comments,
documentation, identifiers (types, functions, fields, variables, modules,
constants, test names), error and log messages, JavaScript, file names, and
commit, pull-request, and changelog text. Write `color`, `neighbor`,
`behavior`, `canceled`, `signaled`, `labeled`, `modeled`, `center`, `gray`,
`favor`, `honor`, `artifact`, and `-ize`/`-ization` (`initialize`,
`normalize`, `serialize`, `optimize`, `materialize`), not the British forms.
`cancellation` is already the American spelling.

This applies to names taken from the codec specifications as well: the
ITU-T and AOM documents spell syntax elements such as `color_primaries` and
`separate_color_plane_flag` the British way, and the code still names them
`color_primaries` and `separate_color_plane_flag`. Keep a British spelling only
where it is not ours to change: verbatim third-party license and notice text,
and the names of external APIs and web standards (such as the HTML
`aria-labelledby` attribute).
