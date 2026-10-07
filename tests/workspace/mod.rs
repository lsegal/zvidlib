//! The workspace's packages, read from their manifests, for the guard tests
//! that check the repository's own configuration against them.
//!
//! zvidlib is a workspace of crates under `crates/` around the root package
//! (#604), so a guard that used to read the root `Cargo.toml` and `benches/`
//! has to read every package's. Deliberately line-based rather than a TOML
//! parse, for the same reason the workflow guards are: the alternative is a
//! `toml` dependency for a hygiene check, and the manifests are written one
//! key per line.

// Each guard compiles this module and uses only what it needs.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The repository root, which is the root package's directory.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// One workspace package.
#[derive(Debug)]
pub struct Package {
    pub name: String,
    /// The package directory, relative to the repository root: `.` for the
    /// root package, `crates/<name>` for the others.
    pub dir: String,
    pub manifest: String,
}

impl Package {
    /// The package directory as an absolute path.
    pub fn path(&self) -> PathBuf {
        root().join(&self.dir)
    }

    /// `relative`, a path inside the package, relative to the repository root.
    pub fn repo_path(&self, relative: &str) -> String {
        if self.dir == "." {
            relative.to_string()
        } else {
            format!("{}/{relative}", self.dir)
        }
    }

    /// The `[[bench]]` targets the manifest declares, by name, each with its
    /// source path relative to the package directory.
    pub fn bench_targets(&self) -> Vec<(String, String)> {
        let mut targets = Vec::new();
        let mut current: Option<(Option<String>, Option<String>)> = None;
        let finish = |current: Option<(Option<String>, Option<String>)>,
                      targets: &mut Vec<(String, String)>| {
            if let Some((Some(name), path)) = current {
                let path = path.unwrap_or_else(|| format!("benches/{name}.rs"));
                targets.push((name, path));
            }
        };
        for line in self.manifest.lines() {
            let line = line.trim();
            if line.starts_with('[') {
                finish(current.take(), &mut targets);
                if line == "[[bench]]" {
                    current = Some((None, None));
                }
                continue;
            }
            let Some((name, path)) = current.as_mut() else {
                continue;
            };
            if let Some(value) = string_value(line, "name") {
                *name = Some(value);
            } else if let Some(value) = string_value(line, "path") {
                *path = Some(value);
            }
        }
        finish(current, &mut targets);
        targets
    }

    /// The workspace crates this package depends on through its
    /// `[dependencies]` and `[build-dependencies]`, platform-specific ones
    /// included.
    pub fn dependencies(&self) -> BTreeSet<String> {
        workspace_dependencies(&self.manifest, false)
    }

    /// The workspace crates this package's tests, benches and examples depend
    /// on through its `[dev-dependencies]`.
    pub fn dev_dependencies(&self) -> BTreeSet<String> {
        workspace_dependencies(&self.manifest, true)
    }
}

/// `key = "value"` on one line, as `value`.
fn string_value(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?;
    Some(rest.trim().trim_matches('"').to_string())
}

/// The `zvidlib-*` keys of the dependency tables of one kind.
fn workspace_dependencies(manifest: &str, dev: bool) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_table = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            let table = line.trim_matches(['[', ']']);
            let kind = table.rsplit('.').next().unwrap_or(table);
            in_table = if dev {
                kind == "dev-dependencies"
            } else {
                kind == "dependencies" || kind == "build-dependencies"
            } && !table.starts_with("workspace");
            continue;
        }
        if !in_table {
            continue;
        }
        if let Some((key, _)) = line.split_once('=') {
            let key = key.trim();
            if key.starts_with("zvidlib-") {
                names.insert(key.to_string());
            }
        }
    }
    names
}

/// Every workspace package: the root package and each `crates/*` member.
pub fn packages() -> Vec<Package> {
    let root = root();
    let read = |path: &Path| {
        std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
    };
    let mut packages = vec![Package {
        name: "zvidlib".to_string(),
        dir: ".".to_string(),
        manifest: read(&root.join("Cargo.toml")),
    }];
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("crates/ is readable")
        .map(|entry| entry.expect("crates/ entry").path())
        .filter(|path| path.join("Cargo.toml").is_file())
        .collect();
    dirs.sort();
    for dir in dirs {
        let manifest = read(&dir.join("Cargo.toml"));
        let name = manifest
            .lines()
            .find_map(|line| string_value(line.trim(), "name"))
            .unwrap_or_else(|| panic!("{} has no package name", dir.display()));
        let relative = dir
            .strip_prefix(&root)
            .expect("crates/ is under the root")
            .to_string_lossy()
            .replace('\\', "/");
        packages.push(Package {
            name,
            dir: relative,
            manifest,
        });
    }
    assert!(
        packages.len() > 10,
        "expected the workspace's crates, found {:?}",
        packages.iter().map(|p| &p.name).collect::<Vec<_>>()
    );
    packages
}

/// The package named `name`.
pub fn package<'a>(packages: &'a [Package], name: &str) -> &'a Package {
    packages
        .iter()
        .find(|package| package.name == name)
        .unwrap_or_else(|| panic!("no workspace package is named {name}"))
}

/// The workspace crates a package's own tests, benches and examples can be
/// affected by: its dependencies, build-dependencies and dev-dependencies,
/// and every crate those depend on in turn.
pub fn affected_by(packages: &[Package], name: &str) -> BTreeSet<String> {
    let mut closure = BTreeSet::new();
    let mut pending: Vec<String> = {
        let package = package(packages, name);
        package
            .dependencies()
            .into_iter()
            .chain(package.dev_dependencies())
            .collect()
    };
    while let Some(next) = pending.pop() {
        if closure.insert(next.clone()) {
            pending.extend(package(packages, &next).dependencies());
        }
    }
    closure.insert(name.to_string());
    closure
}
