//! Guards against a test that CI compiles and never runs.
//!
//! Two guards (#464) and nine library tests (#599) once sat that way, because
//! CI named the integration tests it ran one `--test <name>` at a time and a
//! file nobody named was built by `cargo clippy --all-targets` and executed by
//! nothing. Keeping that list in step by hand was the fix for a while, and it
//! made every new test an edit to `ci.yml` (#616).
//!
//! So CI names no test at all: each test job runs every test target of every
//! package it selects - `--lib`, `--tests` and the examples that set
//! `test = true` - on Linux in a leg per package and on Windows and macOS in
//! one run per host (#612). That every package is selected by each job is
//! `ci_path_filters_are_conservative`'s to check; what is checked here is that
//! no test run goes back to naming its targets, and that none filters or
//! shards what it runs, since either would let a test be compiled and left out
//! again. A test that must not run on some host
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

/// The `nextest run` commands in `source` that run a package's tests, as
/// opposed to the one that runs an `#[ignore]`d test on its own
/// (`--run-ignored only`), which can only add to what the others run.
fn test_runs(source: &str) -> Vec<String> {
    commands(source)
        .into_iter()
        .filter(|command| {
            command.contains("nextest run ") && !command.contains("--run-ignored only")
        })
        .collect()
}

/// Arguments that pick out individual targets, which a test run must not
/// take: every target it leaves out is one CI never runs.
const TARGET_SELECTORS: &[&str] = &["--test", "--example", "--bin", "--bench"];

/// Why `run` would leave a test target of a package it selects unrun, if it
/// would.
fn run_problem(run: &str) -> Option<String> {
    let words: Vec<&str> = run.split_whitespace().collect();
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
fn every_test_run_runs_every_test_target_of_its_packages() {
    let workflow = ci_workflow();
    let runs = test_runs(&workflow);
    assert!(
        runs.len() >= 2,
        "expected the Linux and native test runs in ci.yml, found {runs:?}"
    );
    let problems: Vec<String> = runs
        .iter()
        .filter_map(|run| run_problem(run).map(|why| format!("{why}: {run}")))
        .collect();
    assert!(
        problems.is_empty(),
        "a test run has to run every test target of each package it selects, so a new \
         test runs without a change to the workflow (#616): {problems:?}"
    );
}

/// A test run runs everything it builds. A filter on one would leave out the
/// tests it does not match, which is the per-test list again by another name,
/// and a partition would deal them out over shards again, each running tests
/// of packages the change cannot reach (#612). A host-specific exclusion
/// belongs in the test or in `.config/nextest.toml`.
#[test]
fn every_test_run_runs_everything_it_builds() {
    let workflow = ci_workflow();
    let filtered: Vec<String> = test_runs(&workflow)
        .into_iter()
        .filter(|run| {
            run.split_whitespace().any(|word| {
                word == "-E"
                    || word.starts_with("--filterset")
                    || word.starts_with("--filter-expr")
                    || word.starts_with("--partition")
                    || word == "--"
            })
        })
        .collect();
    assert!(
        filtered.is_empty(),
        "these test runs filter or shard what they run, so a test is compiled and never \
         executed, or run beside packages the change cannot reach: {filtered:?}"
    );
}

/// The reader itself, on workflow text written for it.
#[test]
fn a_run_that_names_a_target_or_skips_a_kind_is_caught() {
    let workflow = r#"
      # `cargo nextest run --test in_a_comment` is prose, not a command.
      - run: |
          cargo nextest run --profile ci --features native -p a \
            --lib --tests --examples
      - run: |
          cargo nextest run --profile ci --lib --tests --examples -p a \
            --test named_on_a_continuation_line
      - run: cargo nextest run --lib -p a
      - run: cargo nextest run -p a --lib --run-ignored only -E 'test(=one)'
"#;
    let problems: Vec<Option<String>> = test_runs(workflow)
        .iter()
        .map(|run| run_problem(run))
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
