//! Guards against a CI guard test that no CI job runs.
//!
//! CI names the integration tests it runs one `--test <name>` at a time -
//! there is no blanket `--tests` - so a test file that is not named is compiled
//! by `cargo clippy --all-targets`, passes review as a test that exists, and is
//! never executed. `ci_concurrency_spares_main` and
//! `ci_staleness_report_sees_markdown_changes` sat that way from the day they
//! landed (#464): both guard arrangements whose own failure mode is silent, and
//! neither would have failed CI if the arrangement it pins were undone.
//!
//! The names are kept explicit rather than collapsed into a glob, because the
//! comment beside the list is where the reason each guard exists is written,
//! and `--test 'ci_*'` would leave nothing saying what any of them protects.
//! The list used to be one `cargo test --test <name>` step per test; since
//! #595 it is the `--test` arguments of the `cargo nextest archive` that the
//! test shards run in full. What made the omission possible was that nothing
//! compared the two lists, so that is what this does: the `tests/ci_*.rs`
//! files and the `--test` names of the cargo invocations that *run* tests in
//! `.github/workflows/` have to be the same set, and a guard added without a
//! name there fails here.
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

/// The shell commands of a workflow, one string per command.
///
/// A command continued onto the next line with a trailing `\` is joined back
/// into one, so a `--test` on a continuation line is read as part of the
/// command it belongs to. Comment lines are skipped: the prose beside a step
/// mentioning a test by name is what made two guards look covered while they
/// were not (#464).
fn commands(source: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut pending = String::new();
    for line in source.lines() {
        let code = line.trim();
        if code.starts_with('#') {
            continue;
        }
        let (code, continued) = match code.strip_suffix('\') {
            Some(head) => (head.trim_end(), true),
            None => (code, false),
        };
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(code);
        if !continued {
            commands.push(std::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        commands.push(pending);
    }
    commands
}

/// Whether a command runs the test targets it names, rather than only
/// building them.
///
/// `cargo test` and `cargo nextest run` run what they build.
/// `cargo nextest archive` does not by itself, but the shards in `ci.yml` run
/// every binary in the archive, so naming a target there is what runs it.
/// `cargo build --test`, `cargo check` and `cargo clippy` only compile it,
/// which is exactly the state these guards sat in unnoticed, and `node --test`
/// takes a path rather than a cargo target.
fn runs_tests(command: &str) -> bool {
    ["cargo test ", "cargo nextest run ", "cargo nextest archive "]
        .iter()
        .any(|invocation| command.contains(invocation))
}

/// Every target named by a `--test <name>` argument of a command in `source`
/// that runs it.
fn targets_run_by(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for command in commands(source) {
        if !runs_tests(&command) {
            continue;
        }
        let mut words = command.split_whitespace();
        while let Some(word) = words.next() {
            if word == "--test" {
                if let Some(name) = words.next() {
                    names.insert(name.trim_matches(['"', '\'']).to_string());
                }
            }
        }
    }
    names
}

/// Every target run by a `--test <name>` argument anywhere in the workflows.
fn targets_run_by_workflows() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for path in workflow_files() {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        names.extend(targets_run_by(&source));
    }
    names
}

#[test]
fn every_ci_guard_test_is_run_by_a_workflow() {
    let run = targets_run_by_workflows();
    let unrun: Vec<String> = ci_guard_targets()
        .into_iter()
        .filter(|target| !run.contains(target))
        .collect();

    assert!(
        unrun.is_empty(),
        "these tests/ci_*.rs guards are not named by a `--test <name>` of any cargo \
         invocation in .github/workflows/ that runs tests - the `cargo nextest archive` \
         the test shards run, or a `cargo test` - so they are compiled and never \
         executed: {unrun:?}"
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
        "these `--test <name>` arguments in .github/workflows/ name a \
         test that does not exist under tests/: {missing:?}"
    );
}

/// The reader itself, on workflow text written for it: a guard that is only
/// compiled, or only mentioned in a comment, must not count as run, and one
/// named on a continuation line of a command that runs it must.
#[test]
fn only_a_command_that_runs_a_test_counts_as_running_it() {
    let workflow = r#"
      # `--test ci_in_a_comment` is prose, not a command.
      - run: cargo build --test ci_only_built
      - run: cargo clippy --all-targets --test ci_only_linted
      - run: node --test 'examples/*.test.js'
      - run: cargo test --features native --test ci_run_by_cargo_test
      - run: |
          cargo nextest archive --features native --lib \
            --test ci_on_a_continuation_line \
            --archive-file nextest-archive.tar.zst
"#;
    let run: Vec<String> = targets_run_by(workflow).into_iter().collect();
    assert_eq!(run, ["ci_on_a_continuation_line", "ci_run_by_cargo_test"]);
}
