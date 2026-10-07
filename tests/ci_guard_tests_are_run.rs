//! Guards against a CI guard test, or any other integration test, that no CI
//! job runs.
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
//! #595 it is run by cargo-nextest, and since the workspace split (#604) it is
//! written as `targets <package> --test <name> ...` lines, each adding a
//! package's targets to the `cargo nextest run` of the test job when the pull
//! request can affect the package - on Linux, the job's leg for that package
//! (#612). What made the omission possible was that nothing compared the two
//! lists, so that is what this does: the `tests/ci_*.rs` files and the
//! `--test` names of the cargo invocations that *run* tests in
//! `.github/workflows/` have to be the same set, and a guard added without a
//! name there fails here.
//!
//! The integration tests that exercise the library had the same gap: nine of
//! them were compiled and never run (#599). So every other `tests/*.rs` file
//! has to be run by a workflow too, unless [`NOT_RUN_BY_CI`] names it with the
//! reason it is left out.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and a `run:` line is read here as the shell text it is.

mod workspace;

use std::collections::BTreeSet;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    workspace::root()
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

/// Integration tests that no workflow runs, each with the reason it is left out.
///
/// Every test in these files is `#[ignore]`d, so naming one in a test job would
/// only add a binary whose every test is skipped. A file that gains a test that
/// asserts something belongs in a workflow instead.
/// [`every_test_left_out_of_ci_exists_and_is_not_run`] fails if a workflow
/// starts running one of these files while it is still listed here.
const NOT_RUN_BY_CI: &[(&str, &str)] = &[
    (
        "preview_index",
        "its one test times the preview tier on the host that runs it (#395); a \
         shared runner's numbers say nothing, so it is run by hand with --ignored",
    ),
    (
        "sao_band_occupancy",
        "it measures how many SAO bands a CTB occupies, to decide whether a kernel \
         was worth writing (#406); it reports rather than asserts, and is run by hand \
         with --ignored --nocapture",
    ),
];

/// Every `tests/*.rs` file of every workspace package, by target name. Since
/// the workspace split (#604) a codec's integration tests live in its crate.
fn test_targets() -> BTreeSet<String> {
    let mut targets = BTreeSet::new();
    for package in workspace::packages() {
        let dir = package.path().join("tests");
        if !dir.is_dir() {
            continue;
        }
        targets.extend(
            std::fs::read_dir(&dir)
                .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
                .map(|entry| entry.expect("tests directory entry").path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
                .filter_map(|path| {
                    path.file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                }),
        );
    }
    targets
}

/// The `tests/ci_*.rs` files, by target name. They are all the root package's.
///
/// The `ci_` prefix is the convention these guards are named by, and it is what
/// distinguishes them from the integration tests that exercise the library:
/// those cover code a unit test could reach, while these read the repository's
/// own configuration and are worth nothing unless something runs them. That is
/// why no guard may be on [`NOT_RUN_BY_CI`].
fn ci_guard_targets() -> BTreeSet<String> {
    let targets: BTreeSet<String> = test_targets()
        .into_iter()
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
        let (code, continued) = match code.strip_suffix('\\') {
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
/// `cargo test` and `cargo nextest run` run what they build, and a
/// `targets <package> ...` line is how the test jobs in `ci.yml` name a target
/// for their `cargo nextest run`. `cargo nextest archive` only packs the
/// binaries; nothing in CI runs an archive since the per-package test jobs
/// replaced the shards that did (#612). `cargo build --test`, `cargo check`
/// and `cargo clippy` only compile it,
/// which is exactly the state these guards sat in unnoticed, and `node --test`
/// takes a path rather than a cargo target.
fn runs_tests(command: &str) -> bool {
    command.starts_with("targets ")
        || ["cargo test ", "cargo nextest run "]
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
         invocation in .github/workflows/ that runs tests - a `targets` line of a test \
         job, a `cargo nextest run` or a `cargo test` - so they are compiled and never \
         executed: {unrun:?}"
    );
}

#[test]
fn every_integration_test_is_run_by_a_workflow_or_left_out_with_a_reason() {
    let run = targets_run_by_workflows();
    let left_out: BTreeSet<&str> = NOT_RUN_BY_CI.iter().map(|(name, _)| *name).collect();
    let unrun: Vec<String> = test_targets()
        .into_iter()
        .filter(|target| !run.contains(target) && !left_out.contains(target.as_str()))
        .collect();

    assert!(
        unrun.is_empty(),
        "these packages' tests/*.rs files are not named by a `--test <name>` of any cargo \
         invocation in .github/workflows/ that runs tests, so they are compiled and \
         never executed (#599). Name each one on the `targets` line of the job that \
         should run it, or add it to NOT_RUN_BY_CI with the reason it is left out: {unrun:?}"
    );
}

#[test]
fn every_test_left_out_of_ci_exists_and_is_not_run() {
    let run = targets_run_by_workflows();
    let targets = test_targets();
    for (name, reason) in NOT_RUN_BY_CI {
        assert!(
            !reason.trim().is_empty(),
            "NOT_RUN_BY_CI names {name} without a reason"
        );
        assert!(
            !name.starts_with("ci_"),
            "NOT_RUN_BY_CI names the CI guard {name}, and a guard nothing runs protects nothing"
        );
        assert!(
            targets.contains(*name),
            "NOT_RUN_BY_CI names {name}, which is not a file under any package's tests/"
        );
        assert!(
            !run.contains(*name),
            "NOT_RUN_BY_CI names {name}, but a workflow runs it; take it off the list"
        );
    }
}

#[test]
fn every_test_a_workflow_runs_exists() {
    let packages = workspace::packages();
    let missing: Vec<String> = targets_run_by_workflows()
        .into_iter()
        .filter(|name| {
            !packages
                .iter()
                .any(|package| package.path().join(format!("tests/{name}.rs")).exists())
        })
        .collect();

    assert!(
        missing.is_empty(),
        "these `--test <name>` arguments in .github/workflows/ name a \
         test that does not exist under any package's tests/: {missing:?}"
    );
}

/// Each `targets <package> --test <name>` line in `source`, as
/// `(package, name)`.
fn package_targets(source: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for command in commands(source) {
        let Some(rest) = command.strip_prefix("targets ") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(package) = words.next() else {
            continue;
        };
        while let Some(word) = words.next() {
            if word == "--test" {
                if let Some(name) = words.next() {
                    pairs.push((package.to_string(), name.to_string()));
                }
            }
        }
    }
    pairs
}

/// A test listed under the wrong package is never built: `-p` selects the
/// package and `--test` then names a target cargo looks for among the selected
/// packages, so one under a package the pull request did not select fails the
/// run, and on Linux, where each package runs in a leg of its own, it fails
/// every leg of the package it is listed under (#612).
#[test]
fn every_test_a_workflow_runs_belongs_to_the_package_it_is_listed_under() {
    let packages = workspace::packages();
    let mut misplaced = Vec::new();
    for path in workflow_files() {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        for (package, name) in package_targets(&source) {
            let owner = workspace::package(&packages, &package);
            if !owner.path().join(format!("tests/{name}.rs")).exists() {
                misplaced.push(format!("{package} --test {name}"));
            }
        }
    }
    assert!(
        misplaced.is_empty(),
        "these `targets` lines in .github/workflows/ list a test under a package \
         whose tests/ does not have it: {misplaced:?}"
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
      - run: cargo nextest archive --lib --test ci_only_archived
      - run: |
          cargo nextest run --profile ci --lib \
            --test ci_on_a_continuation_line
      - run: |
          targets zvidlib \
            --test ci_listed_for_a_package
"#;
    let run: Vec<String> = targets_run_by(workflow).into_iter().collect();
    assert_eq!(
        run,
        [
            "ci_listed_for_a_package",
            "ci_on_a_continuation_line",
            "ci_run_by_cargo_test"
        ]
    );
}
