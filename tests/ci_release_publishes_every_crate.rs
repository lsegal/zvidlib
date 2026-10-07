//! Guards the release workflow's crates.io publish against leaving a crate out.
//!
//! `zvidlib` depends on every `zvidlib-*` crate at exactly its own version, so
//! a release whose publish skips one of them cannot be installed from
//! crates.io at all (#610). `release.yml` names what it leaves out of
//! `cargo package` and `cargo publish` with `--exclude`, a second copy of the
//! `publish = false` the manifests declare; this asserts the copies agree, so
//! a crate added to the workspace is published, and one marked development
//! only is not.
//!
//! It also pins the two properties of the publish step a release depends on
//! and nothing else would report: a tag pushed before the token secret exists
//! still gets its GitHub release, and the publish runs before that release is
//! created, so re-running a job that failed partway repeats both.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and a `run:` line is read here as the shell text it is.

mod workspace;

use std::collections::BTreeSet;

fn release_workflow() -> String {
    let path = workspace::root().join(".github/workflows/release.yml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

/// The packages whose manifest declares `publish = false`.
fn unpublished_packages() -> BTreeSet<String> {
    workspace::packages()
        .into_iter()
        .filter(|package| {
            package
                .manifest
                .lines()
                .any(|line| line.replace(' ', "") == "publish=false")
        })
        .map(|package| package.name)
        .collect()
}

/// The `--exclude` names of the one line of `release.yml` that runs `command`.
fn excluded_by(workflow: &str, command: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = workflow
        .lines()
        .filter(|line| line.trim_start().starts_with(command))
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "release.yml should run `{command}` exactly once, but runs it on {lines:?}"
    );
    let words: Vec<&str> = lines[0].split_whitespace().collect();
    words
        .windows(2)
        .filter(|pair| pair[0] == "--exclude")
        .map(|pair| pair[1].to_string())
        .collect()
}

#[test]
fn every_published_crate_is_packaged_and_published() {
    let workflow = release_workflow();
    let unpublished = unpublished_packages();
    assert!(
        !unpublished.is_empty(),
        "no package declares `publish = false`, so this guard checks nothing"
    );
    for command in ["cargo package --workspace", "cargo publish --workspace"] {
        assert_eq!(
            excluded_by(&workflow, command),
            unpublished,
            "release.yml's `{command}` must exclude exactly the packages that \
             declare `publish = false`: any other crate left out is missing \
             from crates.io, and `zvidlib` cannot be installed without it"
        );
    }
}

#[test]
fn publishing_is_skipped_without_the_token() {
    let workflow = release_workflow();
    assert!(
        workflow.contains("CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}"),
        "the publish step must take its token from the CARGO_REGISTRY_TOKEN secret"
    );
    let check = workflow
        .lines()
        .position(|line| line.trim() == r#"if [ -z "$CARGO_REGISTRY_TOKEN" ]; then"#)
        .expect("the publish step must check for a missing token");
    let skip: Vec<&str> = workflow.lines().skip(check + 1).take(2).collect();
    assert!(
        skip.iter().any(|line| line.contains("::notice::"))
            && skip.iter().any(|line| line.trim() == "exit 0"),
        "without the token the publish step must say why it skipped and \
         succeed, so the GitHub release is still created, but it runs {skip:?}"
    );
}

#[test]
fn crates_are_published_before_the_github_release() {
    let workflow = release_workflow();
    let position = |needle: &str| {
        workflow
            .find(needle)
            .unwrap_or_else(|| panic!("release.yml has no `{needle}`"))
    };
    assert!(
        position("cargo publish --workspace") < position("gh release create"),
        "the crates must be published before the GitHub release is created: \
         `gh release create` fails once the release exists, so a publish after \
         it could not be retried by re-running the job"
    );
}
