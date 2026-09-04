//! Guards the two things the per-target benchmark matrix can silently lose.
//!
//! Fanning the timed benchmark run out over its `[[bench]]` targets (#459) put
//! the list of targets in two places: `Cargo.toml`, where they are declared,
//! and `.github/workflows/ci.yml`, where the matrix names the ones that
//! actually get measured. A target added to the first and not the second is
//! not broken and not slow - it is simply never run on `main` again, and the
//! only symptom is a baseline that quietly stops carrying its groups, which
//! reads as benchmarks that were deleted rather than a matrix that was not
//! updated. The compile job still builds it, so nothing else notices.
//!
//! The second guard is the `rm -rf target/criterion` that the timed step opens
//! with. `Swatinem/rust-cache` walks every directory under `target/` that is
//! not a profile directory and deletes the *files* it finds, leaving the
//! directory skeleton behind, so a restored cache hands criterion an empty
//! `<id>/base/`. Criterion checks that the directory exists, tries to load
//! `<id>/base/sample.json` out of it for the previous-run comparison, and logs
//! `Criterion.rs ERROR: ... No such file or directory` once per benchmark -
//! 267 of them in run 33834381334. Nothing fails, so the line that prevents it
//! can be dropped without any check objecting.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo.rs` is: the alternative is a `serde_yaml`
//! dependency for a hygiene check, and these files are written with one entry
//! per line at a fixed indent.

use std::collections::BTreeSet;
use std::path::Path;

/// Names of the `[[bench]]` targets `Cargo.toml` declares.
fn declared_bench_targets(manifest: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_bench = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_bench = line == "[[bench]]";
            continue;
        }
        if !in_bench {
            continue;
        }
        if let Some(value) = line.strip_prefix("name") {
            let value = value.trim_start().trim_start_matches('=').trim();
            names.insert(value.trim_matches('"').to_string());
        }
    }
    names
}

/// The targets the `benchmarks` job's `bench:` matrix axis lists.
///
/// The axis is a block sequence of bare scalars, so its entries are the run of
/// `- ` lines that follows `bench:` and stops at the first line that is not
/// one.
fn matrix_bench_targets(workflow: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_axis = false;
    for line in workflow.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "bench:" {
            in_axis = true;
            continue;
        }
        if !in_axis {
            continue;
        }
        match trimmed.strip_prefix("- ") {
            Some(name) => {
                names.insert(name.trim().to_string());
            }
            None => break,
        }
    }
    names
}

#[test]
fn the_matrix_measures_every_declared_bench_target() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("reading Cargo.toml");
    let workflow = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("reading .github/workflows/ci.yml");

    let declared = declared_bench_targets(&manifest);
    let measured = matrix_bench_targets(&workflow);
    assert!(
        !declared.is_empty(),
        "no [[bench]] targets found in Cargo.toml; the parser is looking at the wrong thing"
    );

    let unmeasured: Vec<&String> = declared.difference(&measured).collect();
    assert!(
        unmeasured.is_empty(),
        "these [[bench]] targets are declared in Cargo.toml but absent from the \
         `benchmarks` matrix in ci.yml, so `main` never measures them: {unmeasured:?}"
    );

    let undeclared: Vec<&String> = measured.difference(&declared).collect();
    assert!(
        undeclared.is_empty(),
        "the `benchmarks` matrix in ci.yml names targets Cargo.toml does not \
         declare, so `cargo bench --bench` will fail on them: {undeclared:?}"
    );
}

#[test]
fn the_timed_step_clears_the_criterion_directory_first() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflow = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("reading .github/workflows/ci.yml");
    let lines: Vec<&str> = workflow.lines().map(str::trim).collect();

    // The timed invocation, not the compile check next to it: the compile step
    // runs the same subcommand with `--no-run` and writes nothing criterion
    // reads.
    let measured = lines
        .iter()
        .position(|line| {
            line.starts_with("cargo bench --features native --bench") && !line.contains("--no-run")
        })
        .expect(
            "ci.yml no longer runs a single bench target with \
             `cargo bench --features native --bench <target>`",
        );
    let cleared = lines[..measured]
        .iter()
        .rposition(|line| *line == "rm -rf target/criterion");

    assert!(
        cleared.is_some(),
        "the timed `cargo bench` in ci.yml is no longer preceded by \
         `rm -rf target/criterion`; a restored rust-cache leaves an empty `base/` \
         directory behind and criterion logs an error per benchmark against it"
    );
}
