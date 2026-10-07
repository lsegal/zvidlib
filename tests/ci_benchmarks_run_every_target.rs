//! Guards the two things the per-target benchmark matrix can silently lose.
//!
//! Fanning the timed benchmark run out over its `[[bench]]` targets (#459) put
//! the list of targets in two places: the manifests, where they are declared,
//! and `.github/workflows/ci.yml`, where the matrix names the ones that
//! actually get measured. A target added to the first and not the second is
//! not broken and not slow - it is simply never run on `main` again, and the
//! only symptom is a baseline that quietly stops carrying its groups, which
//! reads as benchmarks that were deleted rather than a matrix that was not
//! updated. The compile job still builds it, so nothing else notices. Since
//! the workspace split (#604) each target is declared by the package it
//! measures, so the matrix names the package too, and the compile job lists
//! the packages that have targets at all; both are checked here.
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

mod workspace;

use std::collections::BTreeSet;

fn ci_workflow() -> String {
    std::fs::read_to_string(workspace::root().join(".github/workflows/ci.yml"))
        .expect("reading .github/workflows/ci.yml")
}

/// Every `[[bench]]` target the workspace declares, with its package.
fn declared_bench_targets() -> BTreeSet<(String, String)> {
    workspace::packages()
        .iter()
        .flat_map(|package| {
            package
                .bench_targets()
                .into_iter()
                .map(|(name, _)| (package.name.clone(), name))
        })
        .collect()
}

/// The targets the `benchmarks` job's matrix lists, with their packages.
///
/// The matrix is an `include:` list of `{ bench: <target>, package: <name> }`
/// flow mappings, one per line; the run of those lines that follows the
/// `include:` they belong to is the list.
fn matrix_bench_targets(workflow: &str) -> BTreeSet<(String, String)> {
    let mut entries = BTreeSet::new();
    let mut in_list = false;
    for line in workflow.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "include:" {
            in_list = true;
            continue;
        }
        if !in_list {
            continue;
        }
        let Some(entry) = trimmed.strip_prefix("- {") else {
            in_list = false;
            continue;
        };
        let field = |key: &str| {
            entry.trim_end_matches('}').split(',').find_map(|pair| {
                let (k, v) = pair.split_once(':')?;
                (k.trim() == key).then(|| v.trim().to_string())
            })
        };
        if let (Some(bench), Some(package)) = (field("bench"), field("package")) {
            entries.insert((package, bench));
        }
    }
    entries
}

/// The packages the compile-only job's `benches <package>` lines list.
fn compiled_bench_packages(workflow: &str) -> BTreeSet<String> {
    workflow
        .lines()
        .filter_map(|line| line.trim().strip_prefix("benches "))
        .map(|package| package.trim().to_string())
        .collect()
}

#[test]
fn the_matrix_measures_every_declared_bench_target() {
    let declared = declared_bench_targets();
    let measured = matrix_bench_targets(&ci_workflow());

    assert!(
        !declared.is_empty(),
        "no [[bench]] targets found in the workspace; the parser is looking at the wrong thing"
    );
    let unmeasured: Vec<&(String, String)> = declared.difference(&measured).collect();
    assert!(
        unmeasured.is_empty(),
        "these (package, [[bench]] target) pairs are declared in the manifests but \
         absent from the `benchmarks` matrix in ci.yml, so `main` never measures \
         them: {unmeasured:?}"
    );
    let undeclared: Vec<&(String, String)> = measured.difference(&declared).collect();
    assert!(
        undeclared.is_empty(),
        "the `benchmarks` matrix in ci.yml names (package, target) pairs no manifest \
         declares, so `cargo bench -p <package> --bench <target>` will fail on \
         them: {undeclared:?}"
    );
}

#[test]
fn the_compile_check_covers_every_package_with_bench_targets() {
    let with_targets: BTreeSet<String> = declared_bench_targets()
        .into_iter()
        .map(|(package, _)| package)
        .collect();
    let compiled = compiled_bench_packages(&ci_workflow());
    let uncompiled: Vec<&String> = with_targets.difference(&compiled).collect();
    assert!(
        uncompiled.is_empty(),
        "these packages declare [[bench]] targets the compile-only job never \
         builds, so their benchmarks can rot unnoticed on a pull request: {uncompiled:?}"
    );
    let without: Vec<&String> = compiled.difference(&with_targets).collect();
    assert!(
        without.is_empty(),
        "the compile-only job lists packages that declare no [[bench]] target: {without:?}"
    );
}

#[test]
fn the_timed_step_clears_the_criterion_directory_first() {
    let workflow = ci_workflow();
    let lines: Vec<&str> = workflow.lines().map(str::trim).collect();

    // The timed invocation, not the compile check next to it: the compile step
    // runs the same subcommand with `--no-run` and writes nothing criterion
    // reads.
    let measured = lines
        .iter()
        .position(|line| {
            line.starts_with("cargo bench -p ${{ matrix.package }} --bench")
                && !line.contains("--no-run")
        })
        .expect(
            "ci.yml no longer runs a single bench target with \
             `cargo bench -p <package> --bench <target>`",
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
