//! Guards against a CI guard test that no CI job runs.
//!
//! The `Rust checks` job names its integration tests one
//! `cargo test --test <name>` step at a time - there is no blanket
//! `cargo test --tests` - so a test file that is not given a step is compiled
//! by `cargo check --all-targets`, passes review as a test that exists, and is
//! never executed. `ci_concurrency_spares_main` and
//! `ci_staleness_report_sees_markdown_changes` sat that way from the day they
//! landed (#464): both guard arrangements whose own failure mode is silent, and
//! neither would have failed CI if the arrangement it pins were undone.
//!
//! The per-test steps are kept rather than collapsed into one glob step,
//! because each step's comment is where the reason that guard exists is
//! written, and `cargo test --test 'ci_*'` would leave a list of names with
//! nothing saying what any of them protects. What made the omission possible
//! was that nothing compared the two lists, so that is what this does: the
//! `tests/ci_*.rs` files and the `--test` names in `.github/workflows/` have to
//! be the same set, and a guard added without a step fails here.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and a `run:` line is read here as the shell text it is.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Every `.yml` file under `.github/workflows/`.
fn workflow_files() -> Vec<PathBuf> {
    let dir = manifest_dir().join(".github/workflows");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .map(|entry| entry.expect("workflow directory entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "yml" || ext == "yaml")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no workflows found in {}", dir.display());
    files
}

/// The `tests/ci_*.rs` files, by target name.
///
/// The `ci_` prefix is the convention these guards are named by, and it is what
/// distinguishes them from the integration tests that exercise the library:
/// those cover code a unit test could reach, while these read the repository's
/// own configuration and are worth nothing unless something runs them.
fn ci_guard_targets() -> BTreeSet<String> {
    let dir = manifest_dir().join("tests");
    let targets: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .map(|entry| entry.expect("tests directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .filter_map(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .filter(|stem| stem.starts_with("ci_"))
        .collect();
    assert!(
        targets.len() >= 3,
        "expected the repository's ci_* guard tests, found {targets:?}"
    );
    targets
}

/// Every target named by a `--test <name>` argument anywhere in the workflows.
///
/// A name is taken from the argument that follows `--test` on a line invoking
/// cargo, so a step is counted however its command is otherwise spelled. The
/// prose in the comments beside those steps is not - a comment mentioning a
/// test by name is what made these two look covered while they were not - and
/// neither is `node --test`, which takes a path rather than a cargo target.
fn targets_run_by_workflows() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for path in workflow_files() {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        for line in source.lines() {
            let code = line.trim_start();
            if code.starts_with('#') || !code.contains("cargo ") {
                continue;
            }
            let mut words = code.split_whitespace();
            while let Some(word) = words.next() {
                if word == "--test" {
                    if let Some(name) = words.next() {
                        names.insert(name.trim_matches(['"', '\'']).to_string());
                    }
                }
            }
        }
    }
    names
}

#[test]
fn every_ci_guard_test_is_named_by_a_workflow_step() {
    let run = targets_run_by_workflows();
    let unrun: Vec<String> = ci_guard_targets()
        .into_iter()
        .filter(|target| !run.contains(target))
        .collect();

    assert!(
        unrun.is_empty(),
        "these tests/ci_*.rs guards have no `cargo test --test <name>` step in \
         .github/workflows/, so they are compiled and never executed: {unrun:?}"
    );
}

#[test]
fn every_test_a_workflow_runs_exists() {
    let dir = manifest_dir().join("tests");
    let missing: Vec<String> = targets_run_by_workflows()
        .into_iter()
        .filter(|name| !dir.join(format!("{name}.rs")).exists())
        .collect();

    assert!(
        missing.is_empty(),
        "these `cargo test --test <name>` steps in .github/workflows/ name a \
         test that does not exist under tests/: {missing:?}"
    );
}
