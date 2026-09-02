//! Guards against a CI job that builds Rust without a build cache.
//!
//! Caching in GitHub Actions is opt-in per job, and a job that omits it is not
//! broken - it is only slow, which is the one failure mode CI itself never
//! reports (issue #352). Every job in `.github/workflows/` that invokes cargo
//! therefore has to carry a `Swatinem/rust-cache` step, and this asserts it,
//! so a job added later cannot silently go back to rebuilding every dependency
//! from a cold `target/` on every run.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `bench_sources_compile` walks `mod` declarations by hand: the alternative is
//! a `serde_yaml` dependency for a hygiene check, and these files are written
//! with one step per `- name:` at a fixed indent.

use std::path::{Path, PathBuf};

/// The action that provides the cache, matched on its repository rather than
/// its version so a version bump does not have to be mirrored here.
const CACHE_ACTION: &str = "Swatinem/rust-cache@";

/// Every `.yml` file under `.github/workflows/`.
fn workflow_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
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

/// One job of a workflow: its id, and the lines of its body.
struct Job {
    id: String,
    body: String,
}

/// Splits a workflow into its jobs.
///
/// A job id is the only key at four-space indent under a top-level `jobs:`,
/// which is what every workflow here is written as.
fn jobs(workflow: &str) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut in_jobs = false;
    for line in workflow.lines() {
        if !line.starts_with(char::is_whitespace) && !line.trim().is_empty() {
            in_jobs = line.trim_end() == "jobs:";
            continue;
        }
        if !in_jobs {
            continue;
        }
        let is_job_key = line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_start().ends_with(':')
            && !line.trim_start().starts_with('#');
        if is_job_key {
            let id = line.trim().trim_end_matches(':').to_string();
            jobs.push(Job {
                id,
                body: String::new(),
            });
        } else if let Some(job) = jobs.last_mut() {
            job.body.push_str(line);
            job.body.push('\n');
        }
    }
    jobs
}

/// Whether a job body runs cargo at all.
///
/// `rustup show` alone does not need a cache: the deploy job installs nothing
/// and builds nothing, and requiring a cache of it would be noise.
fn builds_with_cargo(body: &str) -> bool {
    body.lines()
        .map(str::trim)
        .any(|line| line.starts_with("cargo ") || line.contains("run: cargo "))
}

#[test]
fn every_cargo_job_caches_its_build() {
    let workflows = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut checked = 0usize;
    let mut uncached: Vec<String> = Vec::new();

    for path in workflow_files(&workflows) {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let name = path
            .file_name()
            .expect("workflow file name")
            .to_string_lossy()
            .into_owned();
        for job in jobs(&source) {
            if !builds_with_cargo(&job.body) {
                continue;
            }
            checked += 1;
            if !job.body.contains(CACHE_ACTION) {
                uncached.push(format!("{name}: {}", job.id));
            }
        }
    }

    assert!(
        checked >= 4,
        "expected to find the cargo jobs of ci.yml and docs.yml, found {checked}"
    );
    assert!(
        uncached.is_empty(),
        "these CI jobs run cargo without a `{CACHE_ACTION}...` step, so they rebuild \
         every dependency from scratch on every run: {uncached:?}"
    );
}
