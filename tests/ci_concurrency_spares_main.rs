//! Guards against a concurrency block that cancels the runs it documents as
//! never cancelled.
//!
//! `ci.yml` cancels superseded pull-request runs on purpose and exempts `main`
//! pushes and `workflow_dispatch` runs on purpose: the benchmark job's stored
//! baseline is a chain, and a cancelled link is a missing comparison for the
//! run after it. The exemption was written as
//! `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` and did
//! the opposite (#451). Actions substitutes that expression before reading the
//! key, so a `push` rendered the string `false`, every non-empty string is
//! truthy, and the flag meant to protect `main` cancelled it - but only when
//! two `main` pushes overlapped, so the file's comment and its behaviour
//! disagreed for as long as nobody pushed twice in a few minutes.
//!
//! The exemption therefore lives in the group key instead, where it is
//! structural rather than conditional: only a `pull_request` shares a group,
//! and a `push` or `workflow_dispatch` keys on `github.run_id` and is alone in
//! its own, so `cancel-in-progress` has nothing to act on. This asserts both
//! halves, plus the repository-wide rule that made the original wrong -
//! `cancel-in-progress` is a literal in every workflow, never an expression.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and these files are written with one key per line at a
//! fixed indent.

use std::path::{Path, PathBuf};

/// The workflows whose group key must spare every non-`pull_request` event.
/// Both run on a `push` to `main`, and both have been cancelled there.
const EXEMPTING_WORKFLOWS: [&str; 2] = ["ci.yml", "baseline-staleness.yml"];

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows")
}

fn workflow_files() -> Vec<PathBuf> {
    let dir = workflows_dir();
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

/// The value of one key of a workflow's top-level `concurrency:` block, or
/// `None` when the workflow declares no such block or key.
///
/// The block ends at the next line with no leading whitespace, and its keys
/// are the only ones at two-space indent under it.
fn concurrency_value(workflow: &str, key: &str) -> Option<String> {
    let prefix = format!("  {key}:");
    let mut inside = false;
    for line in workflow.lines() {
        if !line.starts_with(char::is_whitespace) && !line.trim().is_empty() {
            if inside {
                break;
            }
            inside = line.trim_end() == "concurrency:";
            continue;
        }
        if inside && line.starts_with(&prefix) {
            return Some(line[prefix.len()..].trim().to_string());
        }
    }
    None
}

fn read(name: &str) -> String {
    let path = workflows_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

/// `cancel-in-progress` is read as a string, so an expression that renders
/// `false` still cancels. Nothing in this repository may write one.
#[test]
fn no_workflow_makes_cancel_in_progress_an_expression() {
    for path in workflow_files() {
        let workflow = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let Some(value) = concurrency_value(&workflow, "cancel-in-progress") else {
            continue;
        };
        assert!(
            value == "true" || value == "false",
            "{}: `cancel-in-progress: {value}` is not a literal - Actions \
             substitutes the expression and reads the result as a string, so a \
             rendered `false` is truthy and cancels anyway (#451). Put the \
             condition in the group key instead.",
            path.display()
        );
    }
}

/// A `push` or a `workflow_dispatch` must key on something unique to the run,
/// so its group has one member and cancellation cannot reach it.
#[test]
fn only_pull_requests_share_a_concurrency_group() {
    for name in EXEMPTING_WORKFLOWS {
        let workflow = read(name);
        let group = concurrency_value(&workflow, "group")
            .unwrap_or_else(|| panic!("{name}: no `concurrency.group` declared"));
        assert!(
            group.contains("github.event_name == 'pull_request'"),
            "{name}: group `{group}` does not distinguish a pull request from \
             the events that must never be cancelled (#451)"
        );
        assert!(
            group.contains("github.run_id"),
            "{name}: group `{group}` gives a `push` or `workflow_dispatch` no \
             per-run key, so two `main` pushes still share a group and the \
             first is cancelled (#451)"
        );
        let (pull_request_arm, other_arm) = group
            .split_once("&&")
            .and_then(|(_, rest)| rest.split_once("||"))
            .unwrap_or_else(|| panic!("{name}: group `{group}` is not a conditional key"));
        assert!(
            pull_request_arm.contains("github.ref"),
            "{name}: a pull request must key on `github.ref` so its own \
             superseded runs are still cancelled, but its arm is \
             `{pull_request_arm}`"
        );
        assert!(
            other_arm.contains("github.run_id"),
            "{name}: every other event must key on `github.run_id`, but its \
             arm is `{other_arm}`"
        );
    }
}

/// The exemption is worth nothing if the flag is off, since then no pull
/// request is superseded either.
#[test]
fn superseded_pull_request_runs_are_still_cancelled() {
    for name in EXEMPTING_WORKFLOWS {
        let workflow = read(name);
        assert_eq!(
            concurrency_value(&workflow, "cancel-in-progress").as_deref(),
            Some("true"),
            "{name}: cancellation is off, so a superseded pull-request run \
             holds a runner the run somebody is waiting on needs"
        );
    }
}
