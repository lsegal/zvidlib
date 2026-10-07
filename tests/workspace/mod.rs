//! The workspace's packages, read from their manifests, for the tests that
//! check every package's bench sources.
//!
//! zvidlib is a workspace of crates under `crates/` around the root package
//! (#604), so a guard that used to read the root `Cargo.toml` and `benches/`
//! has to read every package's. Deliberately line-based rather than a TOML
//! parse: the alternative is a `toml` dependency for a hygiene check, and the
//! manifests are written one key per line.

// Each test compiles this module and uses only what it needs.
#![allow(dead_code)]

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

}

/// `key = "value"` on one line, as `value`.
fn string_value(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?;
    Some(rest.trim().trim_matches('"').to_string())
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

