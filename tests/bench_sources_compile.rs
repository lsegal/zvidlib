//! Guards against orphaned sources under `benches/`.
//!
//! Every bench target is its own crate root, so a file under `benches/` is only
//! compiled if some target reaches it through a `mod` declaration. A file no
//! declaration names is invisible to `cargo check --all-targets`,
//! `cargo clippy --all-targets -- -D warnings` and `cargo bench` alike: it can
//! rot, or stop compiling outright, without any of them saying so. That is what
//! happened to `benches/support/harness.rs`, which sat unreferenced long enough
//! to accumulate a `use super::checksum;` that no longer resolved.
//!
//! This walks the module graph from the `[[bench]]` targets every workspace
//! package declares and asserts it covers every `.rs` file in that package's
//! `benches/` directory.

mod workspace;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Names declared by non-inline `mod` items in one source file.
///
/// Deliberately line-based rather than a real parse: bench sources declare
/// their submodules one per line at the top level, and the alternative is a
/// syn dependency for a hygiene check.
fn declared_submodules(source: &str) -> Vec<String> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//"))
        .filter_map(|line| {
            line.strip_prefix("pub ")
                .unwrap_or(line)
                .strip_prefix("mod ")
        })
        .filter_map(|rest| rest.strip_suffix(';'))
        .map(|name| name.trim().to_string())
        .collect()
}

/// The directory a file's `mod name;` declarations resolve against.
///
/// A crate root and a `mod.rs` own the directory they sit in; any other module
/// owns the subdirectory named after it.
fn module_directory(path: &Path, is_crate_root: bool) -> PathBuf {
    if is_crate_root || path.file_name().is_some_and(|name| name == "mod.rs") {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.with_extension("")
    }
}

/// Resolves `mod name;` to `name.rs` or `name/mod.rs` under `directory`.
fn resolve_submodule(directory: &Path, name: &str) -> Option<PathBuf> {
    [
        directory.join(format!("{name}.rs")),
        directory.join(name).join("mod.rs"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

/// Every `.rs` file under `directory`, recursively.
fn rust_sources(directory: &Path, found: &mut BTreeSet<PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("benches/ is readable") {
        let path = entry.expect("benches/ entries are readable").path();
        if path.is_dir() {
            rust_sources(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.insert(path);
        }
    }
}

#[test]
fn every_bench_source_is_reachable_from_a_bench_target() {
    let root = workspace::root();
    let packages = workspace::packages();
    let mut pending: Vec<(PathBuf, bool)> = packages
        .iter()
        .flat_map(|package| {
            package
                .bench_targets()
                .into_iter()
                .map(|(_, path)| (package.path().join(path), true))
        })
        .collect();
    assert!(
        !pending.is_empty(),
        "the workspace declares no [[bench]] targets, so this guard would pass vacuously"
    );

    let mut reachable = BTreeSet::new();
    while let Some((path, is_crate_root)) = pending.pop() {
        assert!(
            path.is_file(),
            "bench source {} does not exist",
            path.display()
        );
        if !reachable.insert(path.clone()) {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} is readable: {error}", path.display()));
        let directory = module_directory(&path, is_crate_root);
        for name in declared_submodules(&source) {
            let child = resolve_submodule(&directory, &name).unwrap_or_else(|| {
                panic!(
                    "{} declares `mod {name};` with no matching file",
                    path.display()
                )
            });
            pending.push((child, false));
        }
    }

    let mut present = BTreeSet::new();
    for package in &packages {
        let benches = package.path().join("benches");
        if benches.is_dir() {
            rust_sources(&benches, &mut present);
        }
    }
    let orphaned = present
        .difference(&reachable)
        .map(|path| {
            path.strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert!(
        orphaned.is_empty(),
        "these files under a package's benches/ are compiled by no bench target, so \
         no lint or format check covers them; declare them with `mod` or delete \
         them: {}",
        orphaned.join(", ")
    );
}
