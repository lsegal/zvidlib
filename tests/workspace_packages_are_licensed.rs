//! Guards against a workspace package that is published without its license.
//!
//! zvidlib is MIT licensed (#613). crates.io refuses a crate that declares
//! neither `license` nor `license-file`, and a packaged `.crate` carries only
//! the files under its own package directory, so each crate under `crates/`
//! needs its own copy of `LICENSE` as well as the field. A crate added later
//! without either would publish without its license text, or not at all, and
//! nothing short of a release would say so. This asserts every package
//! inherits `license` from `[workspace.package]`, which declares MIT, and
//! keeps a `LICENSE` identical to the root one that its `exclude` list does
//! not leave out.
//!
//! Deliberately line-based rather than a TOML parse, for the same reason the
//! other guards are: the manifests are written one key per line.

mod workspace;

#[test]
fn workspace_declares_mit() {
    let root = workspace::package(&workspace::packages(), "zvidlib")
        .manifest
        .clone();
    let section = root
        .split("[workspace.package]")
        .nth(1)
        .and_then(|rest| rest.split("\n[").next())
        .expect("the root Cargo.toml has a [workspace.package] table");
    assert!(
        section
            .lines()
            .any(|line| line.trim() == "license = \"MIT\""),
        "[workspace.package] must declare license = \"MIT\""
    );
}

#[test]
fn every_package_inherits_the_license() {
    let missing: Vec<String> = workspace::packages()
        .into_iter()
        .filter(|package| {
            !package
                .manifest
                .lines()
                .any(|line| line.trim() == "license.workspace = true")
        })
        .map(|package| package.name)
        .collect();
    assert!(
        missing.is_empty(),
        "these packages do not set `license.workspace = true`: {missing:?}"
    );
}

#[test]
fn every_package_ships_the_license_text() {
    let text = std::fs::read_to_string(workspace::root().join("LICENSE"))
        .expect("the repository root has a LICENSE");
    assert!(
        text.starts_with("MIT License\n"),
        "LICENSE is not the MIT license"
    );
    assert!(
        text.contains("Copyright (c) 2026 Loren Segal\n"),
        "LICENSE does not name its copyright holder"
    );
    for package in workspace::packages() {
        let path = package.path().join("LICENSE");
        let copy = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} has no LICENSE: {error}", package.name));
        assert_eq!(
            copy,
            text,
            "{} differs from the root LICENSE",
            package.repo_path("LICENSE")
        );
        let excluded = package.manifest.lines().any(|line| {
            let line = line.trim();
            line.starts_with("exclude") && line.contains("LICENSE")
        });
        assert!(
            !excluded,
            "{} excludes its LICENSE from the package",
            package.name
        );
    }
}
