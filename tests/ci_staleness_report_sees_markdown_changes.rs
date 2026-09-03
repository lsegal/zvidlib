//! Guards against the baseline staleness report being filtered out of the
//! changes it exists to check.
//!
//! `ci.yml` is `paths-ignore: '**/*.md'`, which is right for a Rust matrix and
//! wrong for one step in it: the staleness report's entire input is
//! `benches/README.md`, so while it lived there the changes that edited a
//! committed baseline table were exactly the changes that skipped checking it,
//! on the branch and after the merge (#449). It now lives in a workflow of its
//! own with a `paths:` filter naming its inputs, and this asserts that
//! arrangement holds: the report is somewhere that a Markdown-only change
//! reaches, the Rust matrix is still somewhere it does not, and the report
//! still only reports.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and these files are written with one key per line at a
//! fixed indent.

use std::path::{Path, PathBuf};

/// The invocation that identifies the report wherever it has been moved to.
const STALENESS_COMMAND: &str = "criterion_baseline.py staleness";

/// The file whose changes must reach the report. It is the report's subject:
/// every stamp and every row it reads comes out of this one path.
const REPORT_SUBJECT: &str = "benches/README.md";

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

/// The lines of a workflow's top-level `on:` block, comments and blanks
/// dropped.
///
/// The block ends at the next line with no leading whitespace, which is how
/// every workflow here is written.
fn trigger_block(workflow: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut inside = false;
    for line in workflow.lines() {
        if !line.starts_with(char::is_whitespace) && !line.trim().is_empty() {
            if inside {
                break;
            }
            inside = line.trim_end() == "on:";
            continue;
        }
        if inside && !line.trim().is_empty() && !line.trim_start().starts_with('#') {
            lines.push(line.to_string());
        }
    }
    assert!(!lines.is_empty(), "no `on:` block found in the workflow");
    lines
}

/// The `paths:` list of one trigger of the `on:` block, or `None` when that
/// trigger is absent or carries no filter.
///
/// A trigger is the only key at two-space indent under `on:`, and its `paths:`
/// list is at four, with one quoted entry per line at six.
fn trigger_paths(block: &[String], trigger: &str) -> Option<Vec<String>> {
    let mut in_trigger = false;
    let mut in_paths = false;
    let mut paths: Option<Vec<String>> = None;
    for line in block {
        let indent = line.len() - line.trim_start().len();
        let key = line.trim();
        if indent == 2 {
            in_trigger = key.trim_end_matches(':') == trigger;
            in_paths = false;
            continue;
        }
        if !in_trigger {
            continue;
        }
        if indent == 4 {
            in_paths = key == "paths:";
            if in_paths {
                paths = Some(Vec::new());
            }
            continue;
        }
        if in_paths && indent >= 6 {
            if let Some(entry) = key.strip_prefix("- ") {
                paths
                    .as_mut()
                    .expect("a `paths:` list is open")
                    .push(entry.trim_matches(['\'', '"']).to_string());
            }
        }
    }
    paths
}

/// The workflow that runs the report, as `(file name, source)`.
fn report_workflow() -> (String, String) {
    let mut found: Vec<(String, String)> = Vec::new();
    for path in workflow_files() {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        // The `run:` line, not a comment mentioning where the report moved to.
        let runs_it = source.lines().any(|line| {
            let line = line.trim();
            !line.starts_with('#') && line.contains(STALENESS_COMMAND)
        });
        if runs_it {
            let name = path
                .file_name()
                .expect("workflow file name")
                .to_string_lossy()
                .into_owned();
            found.push((name, source));
        }
    }
    assert_eq!(
        found.len(),
        1,
        "expected exactly one workflow to run `{STALENESS_COMMAND}`, found {:?}",
        found.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    found.into_iter().next().expect("one workflow")
}

/// The report has to run on a change to the file it reads, from either
/// direction: a pull request that edits a committed table, and the push to
/// `main` that merges it.
#[test]
fn a_markdown_only_change_to_the_committed_tables_runs_the_staleness_report() {
    let (name, source) = report_workflow();
    let block = trigger_block(&source);

    assert!(
        !block.iter().any(|line| line.contains("paths-ignore")),
        "{name} runs the staleness report behind a `paths-ignore`, which is the \
         filter that hid it from Markdown-only changes in the first place (#449)"
    );

    for trigger in ["push", "pull_request"] {
        let paths = trigger_paths(&block, trigger).unwrap_or_else(|| {
            panic!(
                "{name} has no `paths:` filter on its `{trigger}` trigger, so either the \
                 trigger is missing or it runs on changes it has nothing to say about"
            )
        });
        assert!(
            paths.iter().any(|path| path == REPORT_SUBJECT),
            "{name}'s `{trigger}` filter does not name `{REPORT_SUBJECT}`, so a change to \
             the tables the report reads would not run it: {paths:?}"
        );
    }

    // Actions resolves no YAML alias, so the two lists are written out twice
    // and can drift; a report that runs on a pull request and not on the merge
    // is the #449 gap arriving through the other door.
    assert_eq!(
        trigger_paths(&block, "push"),
        trigger_paths(&block, "pull_request"),
        "{name}'s `push` and `pull_request` path filters disagree, so the report runs on \
         only one side of a merge"
    );

    // The registry the report diffs each stamp against. Without it a site
    // landing in `src/simd.rs` - the event that makes a clean table stale -
    // never runs the check that would say so.
    let paths = trigger_paths(&block, "pull_request").expect("a pull_request filter");
    assert!(
        paths.iter().any(|path| path == "src/simd.rs"),
        "{name} does not run on `src/simd.rs`, whose dispatch-site registry is the other \
         half of what the report reads: {paths:?}"
    );
}

/// A stale row is a table to redraw, not a broken build, and moving the report
/// into a job of its own is what would make it a gate by accident: on its own
/// the job's conclusion is the step's.
#[test]
fn the_staleness_report_reports_rather_than_gates() {
    let (name, source) = report_workflow();
    let step = source
        .split("- name:")
        .find(|step| step.contains(STALENESS_COMMAND))
        .unwrap_or_else(|| panic!("{name} has no step running `{STALENESS_COMMAND}`"));
    assert!(
        step.contains("continue-on-error: true"),
        "the staleness report in {name} lost `continue-on-error: true`, so a stale row - or \
         a rate-limited stamp lookup - now fails the build instead of reporting"
    );
}

/// The other half of #449: the report reaching Markdown must not drag the Rust
/// matrix along with it. `ci.yml` keeps its `paths-ignore`, so a prose change
/// still pays for a checkout and a Python script rather than for the matrix,
/// the benchmark suite and the delta report.
#[test]
fn a_markdown_only_change_still_does_not_run_the_rust_matrix() {
    let path = workflows_dir().join("ci.yml");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let block = trigger_block(&source);
    for trigger in ["push", "pull_request"] {
        let ignored: Vec<&String> = block
            .iter()
            .skip_while(|line| line.trim().trim_end_matches(':') != trigger)
            .take_while(|line| {
                let indent = line.len() - line.trim_start().len();
                indent >= 4 || line.trim().trim_end_matches(':') == trigger
            })
            .collect();
        assert!(
            ignored.iter().any(|line| line.contains("paths-ignore")),
            "ci.yml's `{trigger}` trigger lost its `paths-ignore`, so a Markdown-only change \
             now pays for the whole Rust matrix (#449 asked for the opposite)"
        );
        assert!(
            ignored.iter().any(|line| line.contains("'**/*.md'")),
            "ci.yml's `{trigger}` trigger no longer ignores Markdown: {ignored:?}"
        );
    }
}
