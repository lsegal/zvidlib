//! Guards against a test that CI compiles and never runs.
//!
//! Two guards (#464) and nine library tests (#599) once sat that way, because
//! CI named the integration tests it ran one `--test <name>` at a time and a
//! file nobody named was built by `cargo clippy --all-targets` and executed by
//! nothing. Keeping that list in step by hand was the fix for a while, and it
//! made every new test an edit to `ci.yml` (#616).
//!
//! So CI names no test at all: each test archive is built with every test
//! target of every package it selects - `--lib`, `--tests` and the examples
//! that set `test = true` - and the shards run the whole archive. That every
//! package is selected by each archive is `ci_path_filters_are_conservative`'s
//! to check; what is checked here is that no archive goes back to naming its
//! targets, and that no shard filters what it runs, since either would let a
//! test be compiled and left out again. A test that must not run on some host
//! says so itself, with `#[cfg]` or `#[ignore]`, or in `.config/nextest.toml`.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and a `run:` line is read here as the shell text it is.

mod workspace;

/// `.github/workflows/ci.yml`.
fn ci_workflow() -> String {
    let path = workspace::root().join(".github/workflows/ci.yml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

/// The shell commands of a workflow, one string per command.
///
/// A command continued onto the next line with a trailing `\` is joined back
/// into one, so an argument on a continuation line is read as part of the
/// command it belongs to. Comment lines are skipped.
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

/// The `cargo nextest archive` commands in `source`.
fn archives(source: &str) -> Vec<String> {
    commands(source)
        .into_iter()
        .filter(|command| command.contains("cargo nextest archive "))
        .collect()
}

/// Arguments that pick out individual targets, which an archive must not
/// take: every target it leaves out is one the shards never run.
const TARGET_SELECTORS: &[&str] = &["--test", "--example", "--bin", "--bench"];

/// Why `archive` would leave a test target of a package it selects unbuilt,
/// if it would.
fn archive_problem(archive: &str) -> Option<String> {
    let words: Vec<&str> = archive.split_whitespace().collect();
    for required in ["--lib", "--tests", "--examples"] {
        if !words.contains(&required) {
            return Some(format!("it does not take `{required}`"));
        }
    }
    words
        .iter()
        .find(|word| {
            TARGET_SELECTORS
                .iter()
                .any(|selector| *word == selector || word.starts_with(&format!("{selector}=")))
        })
        .map(|word| format!("it names individual targets with `{word}`"))
}

#[test]
fn every_test_archive_builds_every_test_target_of_its_packages() {
    let workflow = ci_workflow();
    let archives = archives(&workflow);
    assert!(
        archives.len() >= 2,
        "expected the Linux and native test archives in ci.yml, found {archives:?}"
    );
    let problems: Vec<String> = archives
        .iter()
        .filter_map(|archive| archive_problem(archive).map(|why| format!("{why}: {archive}")))
        .collect();
    assert!(
        problems.is_empty(),
        "a test archive has to build every test target of each package it selects, so a \
         new test runs without a change to the workflow (#616): {problems:?}"
    );
}

/// The shards run the whole archive. A filter on one would leave out the tests
/// it does not match, which is the per-test list again by another name; a
/// host-specific exclusion belongs in the test or in `.config/nextest.toml`.
/// The one job allowed a filter runs an `#[ignore]`d test on its own
/// (`--run-ignored only`), so it can only add to what the shards run.
#[test]
fn every_shard_runs_its_whole_archive() {
    let workflow = ci_workflow();
    let runs: Vec<String> = commands(&workflow)
        .into_iter()
        .filter(|command| command.contains("nextest run ") && command.contains("--archive-file"))
        .collect();
    assert!(
        runs.iter().any(|run| run.contains("--partition")),
        "expected the sharded `nextest run`s in ci.yml, found {runs:?}"
    );
    let filtered: Vec<&String> = runs
        .iter()
        .filter(|run| !run.contains("--run-ignored only"))
        .filter(|run| {
            run.split_whitespace().any(|word| {
                word == "-E"
                    || word.starts_with("--filterset")
                    || word.starts_with("--filter-expr")
                    || word == "--"
            })
        })
        .collect();
    assert!(
        filtered.is_empty(),
        "these test shards filter what they run, so a test the filter does not match is \
         compiled and never executed: {filtered:?}"
    );
}

/// The reader itself, on workflow text written for it.
#[test]
fn an_archive_that_names_a_target_or_skips_a_kind_is_caught() {
    let workflow = r#"
      # `cargo nextest archive --test in_a_comment` is prose, not a command.
      - run: |
          cargo nextest archive --features native --lib --tests --examples -p a \
            --archive-file whole.tar.zst
      - run: |
          cargo nextest archive --features native --lib --tests --examples -p a \
            --test named_on_a_continuation_line \
            --archive-file named.tar.zst
      - run: cargo nextest archive --lib -p a --archive-file no-tests.tar.zst
"#;
    let problems: Vec<Option<String>> = archives(workflow)
        .iter()
        .map(|archive| archive_problem(archive))
        .collect();
    assert_eq!(
        problems,
        [
            None,
            Some("it names individual targets with `--test`".to_string()),
            Some("it does not take `--tests`".to_string()),
        ]
    );
}
