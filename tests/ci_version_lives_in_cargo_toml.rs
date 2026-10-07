//! Guards against a copy of the release version outside `Cargo.toml`.
//!
//! The last release bump (#633) had to edit `README.md` in five places on
//! top of `Cargo.toml`, and any copy a bump misses goes stale without
//! anything noticing (#641). So the version is written once, as `workspace.package`'s
//! `version` in the root `Cargo.toml`: Rust code reads it with
//! `env!("CARGO_PKG_VERSION")`, the release workflow from `cargo metadata` and
//! the pushed tag, and documentation names it with a placeholder such as
//! `X.Y.Z`. Only `Cargo.lock`, which Cargo rewrites, and `CHANGELOG.md`, whose
//! headings record each release, may name it as well.
//!
//! Every tracked text file is searched, as `git ls-files` lists them, so a new
//! file is covered without being named here.

mod workspace;

use std::process::Command;

/// The tracked files that may name the release version.
const ALLOWED: [&str; 3] = ["Cargo.toml", "Cargo.lock", "CHANGELOG.md"];

/// `workspace.package.version` from the root `Cargo.toml`.
fn workspace_version() -> String {
    let path = workspace::root().join("Cargo.toml");
    let manifest = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let mut in_section = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_section = line == "[workspace.package]";
        } else if in_section {
            let value = line
                .strip_prefix("version")
                .and_then(|rest| rest.trim_start().strip_prefix('='));
            if let Some(value) = value {
                return value.trim().trim_matches('"').to_string();
            }
        }
    }
    panic!("{} has no [workspace.package] version", path.display());
}

/// The repository's tracked files, relative to its root.
fn tracked_files() -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(workspace::root())
        .output()
        .unwrap_or_else(|error| panic!("running git ls-files: {error}"));
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git ls-files lists UTF-8 paths")
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether `line` names `version` as a whole version, so `1.2.3` is found in
/// `v1.2.3` and `zvidlib-1.2.3.crate` but not in `11.2.3` or `1.2.34`.
fn names_version(line: &str, version: &str) -> bool {
    line.match_indices(version).any(|(start, _)| {
        let before = line[..start].chars().next_back();
        let mut after = line[start + version.len()..].chars();
        let continues_before = before.is_some_and(|c| c.is_ascii_digit() || c == '.');
        let continues_after = match after.next() {
            Some(c) if c.is_ascii_digit() => true,
            Some('.') => after.next().is_some_and(|c| c.is_ascii_digit()),
            _ => false,
        };
        !continues_before && !continues_after
    })
}

#[test]
fn release_version_is_written_only_in_cargo_toml() {
    let version = workspace_version();
    let root = workspace::root();
    let mut hits = Vec::new();
    for path in tracked_files() {
        if ALLOWED.contains(&path.as_str()) {
            continue;
        }
        let Ok(bytes) = std::fs::read(root.join(&path)) else {
            // Listed but deleted in the working tree.
            continue;
        };
        // Binary media and fixtures are not where a version is written by hand.
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            if names_version(line, &version) {
                hits.push(format!("{path}:{}: {}", index + 1, line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "the release version {version} is written outside {ALLOWED:?}; read it from \
         Cargo.toml (env!(\"CARGO_PKG_VERSION\"), cargo metadata or the release tag) or \
         use a placeholder such as X.Y.Z:\n{}",
        hits.join("\n")
    );
}

#[test]
fn names_version_matches_whole_versions_only() {
    for line in [
        "tag = \"v1.2.3\"",
        "zvidlib-1.2.3.crate",
        "version 1.2.3.",
        "1.2.3",
    ] {
        assert!(names_version(line, "1.2.3"), "{line}");
    }
    for line in [
        "11.2.3",
        "1.2.34",
        "1.2.3.4",
        "0.1.2.3",
        "1.2.30 and 21.2.3",
    ] {
        assert!(!names_version(line, "1.2.3"), "{line}");
    }
}
