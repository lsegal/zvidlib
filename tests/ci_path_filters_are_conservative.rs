//! Guards the path filters that let a pull request skip CI jobs it does not
//! touch.
//!
//! Neither cargo nor nextest caches a test result, so `ci.yml`'s `changes` job
//! decides from a pull request's changed files which jobs have to run, and the
//! others are skipped (#600). A filter that is too wide only costs a runner; a
//! filter that is too narrow skips a job on exactly the change that would have
//! failed it, and a skipped job reports itself as nothing at all - the failure
//! mode none of CI's own reporting can see. So this pins the arrangement from
//! the side that matters:
//!
//! - `main` pushes and `workflow_dispatch` run every gated job, so the
//!   benchmark baseline chain and `main`'s history stay complete;
//! - a change to the build definition runs everything;
//! - each filter matches the files its job is known to read, including every
//!   file a source pulls in through `include_str!` or `include_bytes!`;
//! - each exclusion names a module whose `cfg` keeps it out of the excluding
//!   job's target, so a module that starts compiling for that target again
//!   fails here instead of going unchecked;
//! - each workspace package has a filter that matches its own files, the files
//!   of every crate it depends on and every file it includes, and nothing of
//!   the crates it does not depend on (#604), so a change to one crate tests
//!   and benchmarks that crate and its dependents and skips the rest.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and the filter block is written one pattern per line.
//! The globs are matched by a reader that knows only `*`, `**` and `{a,b}`, and
//! `every_pattern_uses_only_the_glob_syntax_this_test_reads` keeps the filters
//! to that subset of what `dorny/paths-filter`'s picomatch accepts.

mod workspace;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The filter whose match runs every gated job.
const EVERYTHING: &str = "everything";

/// The files a change to which has to run every gated job (#600): the
/// manifest, the lockfile, the build script, nextest's configuration and the
/// workflow itself, plus the toolchain and cargo configuration, which change
/// every build just as much.
const BUILD_DEFINITION: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    ".config/nextest.toml",
    ".github/workflows/ci.yml",
    "rust-toolchain.toml",
    ".cargo/config.toml",
];

/// The gated jobs, by job id, and the `changes` output each is gated on.
const GATED_JOBS: &[(&str, &str)] = &[
    ("rust", "rust"),
    ("wasm", "wasm"),
    ("rust-tests", "rust_tests"),
    ("native-tests", "native_tests"),
    ("macos-swift-rpath", "macos_swift_rpath"),
];

/// The `changes` output that lists the packages a pull request can affect.
const PACKAGES: &str = "packages";

/// The `changes` output the per-package test matrix expands (#612): the same
/// selection as [`PACKAGES`], as package names only.
const CRATES: &str = "crates";

/// The job that runs each selected package's tests as a leg of its own.
const PER_PACKAGE_TESTS: &str = "rust-tests";

/// The filters whose jobs build the native target, and so compile every
/// `include_str!` and `include_bytes!` that is not browser-only.
const NATIVE_FILTERS: &[&str] = &["rust_tests", "native_tests"];

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn ci_workflow() -> String {
    let path = manifest_dir().join(".github/workflows/ci.yml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The filters of the `filters: |` block, by name, each as its patterns in
/// order.
fn filters(workflow: &str) -> BTreeMap<String, Vec<String>> {
    let mut lines = workflow.lines();
    let header = lines
        .by_ref()
        .find(|line| line.trim() == "filters: |")
        .expect("ci.yml has a `filters: |` block");
    let block_indent = indent(header);
    let mut filters: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in lines {
        let code = line.trim();
        if code.is_empty() || code.starts_with('#') {
            continue;
        }
        if indent(line) <= block_indent {
            break;
        }
        if let Some(pattern) = code.strip_prefix("- ") {
            let name = current.as_ref().expect("a pattern belongs to a filter");
            filters
                .get_mut(name)
                .expect("the filter is open")
                .push(pattern.trim_matches(['\'', '"']).to_string());
        } else {
            let name = code
                .strip_suffix(':')
                .unwrap_or_else(|| panic!("unexpected line in the filter block: {line:?}"));
            assert!(
                filters.insert(name.to_string(), Vec::new()).is_none(),
                "filter `{name}` is declared twice"
            );
            current = Some(name.to_string());
        }
    }
    assert!(!filters.is_empty(), "the `filters: |` block is empty");
    filters
}

/// One job of a workflow: its id, and the lines of its body.
struct Job {
    id: String,
    body: String,
}

/// Splits a workflow into its jobs, as `ci_workflows_cache_cargo` does: a job
/// id is the only key at two-space indent under a top-level `jobs:`.
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
            jobs.push(Job {
                id: line.trim().trim_end_matches(':').to_string(),
                body: String::new(),
            });
        } else if let Some(job) = jobs.last_mut() {
            job.body.push_str(line);
            job.body.push('\n');
        }
    }
    jobs
}

fn job<'a>(jobs: &'a [Job], id: &str) -> &'a Job {
    jobs.iter()
        .find(|job| job.id == id)
        .unwrap_or_else(|| panic!("ci.yml has no `{id}` job"))
}

/// The value of a job-level key (four-space indent), comments dropped.
fn job_key(job: &Job, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    job.body
        .lines()
        .find(|line| indent(line) == 4 && line.trim().starts_with(&prefix))
        .map(|line| line.trim()[prefix.len()..].trim().to_string())
}

/// The `changes` job's outputs, by name.
fn outputs(changes: &Job) -> BTreeMap<String, String> {
    let mut outputs = BTreeMap::new();
    let mut inside = false;
    for line in changes.body.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if indent(line) == 4 {
            inside = line.trim() == "outputs:";
            continue;
        }
        if inside && indent(line) == 6 {
            let (name, value) = line
                .trim()
                .split_once(':')
                .expect("an output is `name: value`");
            outputs.insert(name.to_string(), value.trim().to_string());
        }
    }
    outputs
}

/// Brace expansion, innermost-first: `a/{b,c}.rs` is `a/b.rs` and `a/c.rs`.
fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(close) = pattern.find('}') else {
        return vec![pattern.to_string()];
    };
    let open = pattern[..close]
        .rfind('{')
        .unwrap_or_else(|| panic!("unbalanced `}}` in {pattern:?}"));
    let (head, tail) = (&pattern[..open], &pattern[close + 1..]);
    pattern[open + 1..close]
        .split(',')
        .flat_map(|choice| expand_braces(&format!("{head}{choice}{tail}")))
        .collect()
}

/// Whether one path segment matches one pattern segment, where `*` is any run
/// of characters.
fn segment_matches(pattern: &str, segment: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == segment,
        Some((head, rest)) => {
            let Some(remaining) = segment.strip_prefix(head) else {
                return false;
            };
            (0..=remaining.len())
                .filter(|&at| remaining.is_char_boundary(at))
                .any(|at| segment_matches(rest, &remaining[at..]))
        }
    }
}

fn segments_match(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|skip| segments_match(rest, &path[skip..])),
        Some((first, rest)) => path.split_first().is_some_and(|(segment, tail)| {
            segment_matches(first, segment) && segments_match(rest, tail)
        }),
    }
}

/// Whether a repository-relative path matches a glob without a leading `!`.
fn glob_matches(pattern: &str, path: &str) -> bool {
    let path: Vec<&str> = path.split('/').collect();
    expand_braces(pattern).iter().any(|expanded| {
        let pattern: Vec<&str> = expanded.split('/').collect();
        segments_match(&pattern, &path)
    })
}

/// `predicate-quantifier: some-with-excludes`: matched by any pattern of the
/// filter and by none of its `!` patterns.
fn filter_matches(patterns: &[String], path: &str) -> bool {
    let excluded = patterns
        .iter()
        .filter_map(|pattern| pattern.strip_prefix('!'))
        .any(|pattern| glob_matches(pattern, path));
    let included = patterns
        .iter()
        .filter(|pattern| !pattern.starts_with('!'))
        .any(|pattern| glob_matches(pattern, path));
    included && !excluded
}

/// Whether a change to `path` runs the job gated on `output`.
fn runs(filters: &BTreeMap<String, Vec<String>>, output: &str, path: &str) -> bool {
    filter_matches(&filters[EVERYTHING], path) || filter_matches(&filters[output], path)
}

/// Every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

fn relative(path: &Path) -> String {
    path.strip_prefix(manifest_dir())
        .expect("a path inside the repository")
        .to_string_lossy()
        .replace('\\', "/")
}

/// The `cfg` attributes written directly above a `mod <name>;` declaration
/// in `parent`, joined, or `None` when the module is not declared there.
fn module_cfg(parent: &Path, name: &str) -> Option<String> {
    let source = std::fs::read_to_string(parent)
        .unwrap_or_else(|error| panic!("reading {}: {error}", parent.display()));
    let lines: Vec<&str> = source.lines().collect();
    let declaration = lines.iter().position(|line| {
        let code = line.trim();
        let code = code
            .strip_prefix("pub(crate) ")
            .or_else(|| code.strip_prefix("pub "))
            .unwrap_or(code);
        code == format!("mod {name};")
    })?;
    let attributes: Vec<&str> = lines[..declaration]
        .iter()
        .rev()
        .map(|line| line.trim())
        .take_while(|line| line.starts_with("#[") || line.starts_with("//"))
        .filter(|line| line.starts_with("#[cfg("))
        .collect();
    Some(attributes.join(" "))
}

/// The source directory an excluded path belongs to: `src` for the root
/// package, `crates/<name>/src` for a workspace crate.
fn source_root(path: &str) -> &str {
    match path.strip_prefix("crates/") {
        Some(rest) => {
            let name = rest.split('/').next().expect("a crate directory");
            &path[.."crates/".len() + name.len() + "/src".len()]
        }
        None => "src",
    }
}

/// The crate-level `cfg` of a crate root: its `#![cfg(...)]` attributes,
/// written as the outer `#[cfg(...)]` they act as.
fn crate_cfg(lib: &Path) -> String {
    let source = std::fs::read_to_string(lib)
        .unwrap_or_else(|error| panic!("reading {}: {error}", lib.display()));
    source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("#![cfg("))
        .map(|line| line.replacen("#![", "#[", 1))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The `cfg` an excluded path's module is declared under: `src/a/**` is module
/// `a` of `src/lib.rs`, `src/a.rs` likewise, and `src/a/b.rs` is module `b` of
/// `src/a/mod.rs` or `src/a.rs`; the same holds under `crates/<name>/src`, and
/// `crates/<name>/src/**` is the whole crate, under its crate-level `cfg`.
fn excluded_module_cfg(excluded: &str) -> String {
    let root = manifest_dir();
    let path = excluded.strip_suffix("/**").unwrap_or(excluded);
    assert!(
        root.join(path).exists(),
        "`!{excluded}` names nothing in the repository; a renamed module is now matched \
         again, so this exclusion is dead and should follow it or go"
    );
    let source_root = source_root(path);
    if path == source_root {
        return crate_cfg(&root.join(source_root).join("lib.rs"));
    }
    let file = Path::new(path);
    let name = file
        .file_stem()
        .expect("an excluded module has a name")
        .to_string_lossy()
        .into_owned();
    let dir = file.parent().expect("an excluded module has a directory");
    let parents = if dir == Path::new(source_root) {
        vec![root.join(source_root).join("lib.rs")]
    } else {
        vec![
            root.join(dir).join("mod.rs"),
            root.join(dir).with_extension("rs"),
        ]
    };
    parents
        .iter()
        .filter(|parent| parent.exists())
        .find_map(|parent| module_cfg(parent, &name))
        .unwrap_or_else(|| panic!("no `mod {name};` declaration found for `!{excluded}`"))
}

#[test]
fn main_pushes_and_dispatches_run_every_gated_job() {
    let workflow = ci_workflow();
    let jobs = jobs(&workflow);
    let changes = job(&jobs, "changes");
    let outputs = outputs(changes);
    let filters = filters(&workflow);

    // The filter step runs on a pull request only, so on a push its outputs
    // are empty and could only ever read as `false`.
    assert!(
        changes
            .body
            .contains("if: github.event_name == 'pull_request'"),
        "the `changes` job's filter step is no longer limited to pull requests"
    );

    for (id, output) in GATED_JOBS {
        let expression = outputs
            .get(*output)
            .unwrap_or_else(|| panic!("the `changes` job has no `{output}` output for `{id}`"));
        let expected = format!(
            "${{{{ github.event_name != 'pull_request' || steps.filter.outputs.{EVERYTHING} == 'true' \
             || steps.filter.outputs.{output} == 'true' }}}}"
        );
        assert_eq!(
            expression, &expected,
            "the `{output}` output must be `true` on anything but a pull request, and on a \
             pull request whenever the build definition or `{id}`'s own inputs changed"
        );
        assert!(
            filters.contains_key(*output),
            "the `{output}` output reads a filter that does not exist"
        );
    }
    let packages_expression = format!(
        "${{{{ (github.event_name != 'pull_request' || steps.filter.outputs.{EVERYTHING} == 'true') \
         && 'all' || steps.filter.outputs.changes }}}}"
    );
    assert_eq!(
        outputs.get(PACKAGES),
        Some(&packages_expression),
        "the `{PACKAGES}` output must be `all` on anything but a pull request and when the \
         build definition changed, and otherwise the filters a pull request matched"
    );
    assert_eq!(
        outputs.get(CRATES).map(String::as_str),
        Some("${{ steps.crates.outputs.crates }}"),
        "the `{CRATES}` output must be what the step that expands `{PACKAGES}` lists"
    );
    // The step that lists them reads the same selection `packages` is, so on a
    // push it is `all` and every package's tests run.
    assert!(
        changes
            .body
            .lines()
            .any(|line| line.trim() == format!("PACKAGES: {packages_expression}")),
        "the step behind the `{CRATES}` output no longer reads the `{PACKAGES}` selection"
    );
    assert_eq!(
        outputs.len(),
        GATED_JOBS.len() + 2,
        "the `changes` job has outputs this test does not know: {:?}",
        outputs.keys().collect::<Vec<_>>()
    );
}

#[test]
fn each_gated_job_is_skipped_only_by_its_own_filter() {
    let workflow = ci_workflow();
    let jobs = jobs(&workflow);
    for (id, output) in GATED_JOBS {
        let job = job(&jobs, id);
        assert_eq!(
            job_key(job, "needs").as_deref(),
            Some("changes"),
            "`{id}` must need the `changes` job to read its output"
        );
        assert_eq!(
            job_key(job, "if"),
            Some(format!("needs.changes.outputs.{output} == 'true'")),
            "`{id}` is not gated on the `{output}` output"
        );
    }
    // A job gated on an output nobody sets would never run at all.
    for job in &jobs {
        if let Some(condition) = job_key(job, "if") {
            if let Some(output) = condition.strip_prefix("needs.changes.outputs.") {
                let output = output.split_whitespace().next().unwrap_or_default();
                assert!(
                    GATED_JOBS.iter().any(|(_, known)| *known == output),
                    "`{}` is gated on `{output}`, which is not one of the known outputs",
                    job.id
                );
            }
        }
    }
}

#[test]
fn a_change_to_the_build_definition_runs_everything() {
    let filters = filters(&ci_workflow());
    let everything = &filters[EVERYTHING];
    for path in BUILD_DEFINITION {
        assert!(
            filter_matches(everything, path),
            "a change to `{path}` no longer runs every job: {everything:?}"
        );
    }
    assert!(
        !everything.iter().any(|pattern| pattern.starts_with('!')),
        "the `{EVERYTHING}` filter excludes something, which only narrows it"
    );
}

/// What each job is known to read, as `(output, path, runs)`. These are the
/// files a regression in a filter would most plausibly stop covering.
#[test]
fn each_job_runs_on_the_files_it_reads() {
    let filters = filters(&ci_workflow());
    let cases: &[(&str, &str, bool)] = &[
        // Formatting and lints cover every target, browser modules included.
        ("rust", "src/lib.rs", true),
        ("rust", "src/wasm_api.rs", true),
        (
            "rust",
            "crates/zvidlib-hevc-decoder/src/engine/mod.rs",
            true,
        ),
        ("rust", "crates/zvidlib-opus/tests/opus_codec.rs", true),
        ("rust", "crates/zvidlib-av1/benches/codec.rs", true),
        ("rust", "benches/audio_mux.rs", true),
        ("rust", "examples/native_gl/main.rs", true),
        ("rust", "examples/web_canvas/scrub.test.js", true),
        ("rust", ".github/scripts/criterion_baseline.py", true),
        ("rust", "rustfmt.toml", true),
        // #600's own example: a change to native backends only skips these.
        ("wasm", "crates/zvidlib-hardware/src/nvdec.rs", false),
        ("wasm", "crates/zvidlib-hardware/src/windows_mf.rs", false),
        ("wasm", "src/native_audio/symphonia.rs", false),
        ("wasm", "crates/zvidlib-aac-encoder/src/lib.rs", false),
        ("wasm", "crates/zvidlib-hevc-encoder/src/encoder.rs", false),
        // A native-only crate's manifest still decides what wasm32 resolves.
        ("wasm", "crates/zvidlib-aac-encoder/Cargo.toml", true),
        ("wasm", "crates/zvidlib-hardware/src/lib.rs", true),
        ("wasm", "src/lib.rs", true),
        ("wasm", "src/wasm_api.rs", true),
        ("wasm", "crates/zvidlib-hevc-decoder/src/lib.rs", true),
        (
            "wasm",
            "crates/zvidlib-hevc-decoder/src/engine/mod.rs",
            true,
        ),
        ("wasm", "crates/zvidlib-color/src/color_convert.rs", true),
        ("wasm", "js/browser.js", true),
        (
            "wasm",
            "crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4",
            true,
        ),
        ("wasm", "examples/hevc_decode_profile.rs", true),
        ("wasm", ".github/scripts/criterion_baseline.py", false),
        // The Linux tests run the guards, which read every workflow.
        ("rust_tests", "crates/zvidlib-hardware/src/nvdec.rs", true),
        ("rust_tests", "src/wasm_api.rs", false),
        ("rust_tests", "js/browser.js", false),
        (
            "rust_tests",
            "crates/zvidlib-vp9-decoder/tests/fixtures/libvpx_vp9_test_vectors.txt",
            true,
        ),
        ("rust_tests", "examples/media/BigBuckBunny.mp4", true),
        ("rust_tests", "examples/web_canvas/samples.js", true),
        ("rust_tests", "benches/README.md", true),
        (
            "rust_tests",
            ".github/workflows/baseline-staleness.yml",
            true,
        ),
        ("rust_tests", ".github/workflows/docs.yml", true),
        (
            "native_tests",
            "crates/zvidlib-hardware/src/videotoolbox.rs",
            true,
        ),
        (
            "native_tests",
            "crates/zvidlib-aac-encoder/src/windows_mf.rs",
            true,
        ),
        ("native_tests", "src/web_decoder.rs", false),
        (
            "native_tests",
            "crates/zvidlib-hevc-encoder/tests/support/avfoundation_decode.swift",
            true,
        ),
        (
            "native_tests",
            "crates/zvidlib-hevc-decoder/benches/hevc_hardware.rs",
            true,
        ),
        (
            "macos_swift_rpath",
            "crates/zvidlib-hardware/src/videotoolbox.rs",
            true,
        ),
        (
            "macos_swift_rpath",
            "tests/macos_swift_runtime_rpath.rs",
            true,
        ),
        (
            "macos_swift_rpath",
            "tests/media_output_cover_art.rs",
            false,
        ),
        ("macos_swift_rpath", "src/web_previews.rs", false),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter(|(output, path, expected)| runs(&filters, output, path) != *expected)
        .map(|(output, path, expected)| {
            let verb = if *expected { "skips" } else { "runs" };
            format!("`{output}` {verb} on `{path}`")
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "path filters disagree with their jobs' inputs: {wrong:?}"
    );

    // Nothing but the build definition and documentation goes unread: a file
    // outside every filter is a file no change to which runs any check.
    for path in ["README.md", "tools/vorbis_encoder/ref_enc.c"] {
        assert!(
            !GATED_JOBS
                .iter()
                .any(|(_, output)| runs(&filters, output, path)),
            "`{path}` is expected to run no gated job"
        );
    }
}

/// A file a source pulls in at compile time is an input of every job that
/// compiles that source, wherever in the repository it lives -
/// `src/simd.rs`'s units read `benches/README.md` and a script under
/// `.github/scripts/`, which a filter written from the directory layout alone
/// would miss.
#[test]
fn every_file_a_source_includes_runs_the_native_test_jobs() {
    let filters = filters(&ci_workflow());
    let root = manifest_dir();
    let mut sources = Vec::new();
    for dir in ["src", "tests", "benches", "examples"] {
        rust_files(&root.join(dir), &mut sources);
    }
    rust_files(&root.join("crates"), &mut sources);
    let mut checked = 0;
    let mut missed = Vec::new();
    for source in sources {
        let text = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("reading {}: {error}", source.display()));
        for macro_name in ["include_str!(\"", "include_bytes!(\""] {
            for (at, _) in text.match_indices(macro_name) {
                let rest = &text[at + macro_name.len()..];
                let literal = &rest[..rest.find('"').expect("a closed string literal")];
                let target = source.parent().expect("a source directory").join(literal);
                let Ok(target) = target.canonicalize() else {
                    continue;
                };
                let Ok(path) = target.strip_prefix(root.canonicalize().expect("the root")) else {
                    continue;
                };
                let path = path.to_string_lossy().replace('\\', "/");
                checked += 1;
                for output in NATIVE_FILTERS {
                    if !runs(&filters, output, &path) {
                        missed.push(format!("{} -> {path} ({output})", relative(&source)));
                    }
                }
            }
        }
    }
    assert!(
        checked > 0,
        "found no `include_str!` or `include_bytes!` to check"
    );
    assert!(
        missed.is_empty(),
        "these files are compiled into a native target, so a change to them has to run \
         the job that builds it: {missed:?}"
    );
}

/// The exclusions are the one place a filter is narrower than a directory, so
/// each one has to name a module the job's target genuinely does not compile.
/// The WebAssembly checks skip modules compiled only off wasm32, and the
/// native jobs skip modules compiled only on it.
#[test]
fn an_exclusion_names_a_module_the_target_does_not_compile() {
    const NOT_WASM: &str = "not(target_arch = \"wasm32\")";
    let filters = filters(&ci_workflow());
    for (name, patterns) in &filters {
        for excluded in patterns
            .iter()
            .filter_map(|pattern| pattern.strip_prefix('!'))
        {
            let in_a_crate = excluded
                .strip_prefix("crates/")
                .is_some_and(|rest| rest.split('/').nth(1) == Some("src"));
            assert!(
                (excluded.starts_with("src/") || in_a_crate)
                    && !excluded.contains('{')
                    && (excluded.ends_with(".rs")
                        || excluded.ends_with("/**")
                            && !excluded[..excluded.len() - 3].contains('*')),
                "`{name}` excludes `{excluded}`; exclusions name one module - a source file \
                 or module directory under `src/` or a crate's `src/` - or one whole crate's \
                 `src/`, so each can be held to its `cfg`"
            );
            let cfg = excluded_module_cfg(excluded);
            let positive_wasm = cfg
                .replace(NOT_WASM, "")
                .contains("target_arch = \"wasm32\"");
            if name == "wasm" {
                let off_wasm = cfg.contains(NOT_WASM)
                    || ["windows", "target_os = \"macos\"", "target_os = \"linux\""]
                        .iter()
                        .any(|platform| cfg.contains(platform));
                assert!(
                    off_wasm && !positive_wasm,
                    "`{excluded}` is skipped by the WebAssembly checks, but its module is \
                     declared under {cfg:?}, which does not keep it out of wasm32"
                );
            } else {
                assert!(
                    (positive_wasm && cfg.starts_with("#[cfg(all("))
                        || cfg == "#[cfg(target_arch = \"wasm32\")]",
                    "`{excluded}` is skipped by `{name}`, but its module is declared under \
                     {cfg:?}, which does not limit it to wasm32"
                );
            }
        }
    }
}

/// The filters that gate jobs; every other filter is a package's.
const JOB_FILTERS: &[&str] = &[
    EVERYTHING,
    "rust",
    "wasm",
    "rust_tests",
    "native_tests",
    "macos_swift_rpath",
];

/// Files of a package a change to it would touch: its manifest and library
/// root, and for the root package a file of each of its target directories.
fn representative_files(package: &workspace::Package) -> Vec<String> {
    if package.dir == "." {
        return [
            "src/lib.rs",
            "tests/media_output_cover_art.rs",
            "benches/audio_mux.rs",
            "examples/native_encode.rs",
        ]
        .iter()
        .map(|path| path.to_string())
        .collect();
    }
    ["Cargo.toml", "src/lib.rs"]
        .iter()
        .map(|path| package.repo_path(path))
        .collect()
}

#[test]
fn every_package_has_a_filter_named_after_it() {
    let filters = filters(&ci_workflow());
    let packages: Vec<String> = workspace::packages()
        .into_iter()
        .map(|package| package.name)
        .collect();
    let missing: Vec<&String> = packages
        .iter()
        .filter(|name| !filters.contains_key(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "these workspace packages have no filter, so `packages` never selects them and \
         a pull request never tests them: {missing:?}"
    );
    let unknown: Vec<&String> = filters
        .keys()
        .filter(|name| !JOB_FILTERS.contains(&name.as_str()) && !packages.contains(name))
        .collect();
    assert!(
        unknown.is_empty(),
        "these filters name neither a job nor a workspace package: {unknown:?}"
    );
}

/// A package's tests and benchmarks have to run when anything they compile
/// against changes: the package itself and every crate it depends on, by any
/// kind of dependency, transitively.
#[test]
fn a_package_filter_runs_on_its_own_and_its_dependencies_files() {
    let filters = filters(&ci_workflow());
    let packages = workspace::packages();
    let mut missed = Vec::new();
    for package in &packages {
        let patterns = &filters[&package.name];
        for dependency in workspace::affected_by(&packages, &package.name) {
            for path in representative_files(workspace::package(&packages, &dependency)) {
                if !filter_matches(patterns, &path) {
                    missed.push(format!("{} skips {path} ({dependency})", package.name));
                }
            }
        }
    }
    assert!(
        missed.is_empty(),
        "these package filters skip a change their package compiles against: {missed:?}"
    );
}

/// The other half of #604: a change to one crate does not select the crates
/// that do not depend on it, which is the whole point of the split.
#[test]
fn a_package_filter_skips_the_crates_it_does_not_depend_on() {
    let filters = filters(&ci_workflow());
    let packages = workspace::packages();
    let mut extra = Vec::new();
    for package in &packages {
        let patterns = &filters[&package.name];
        let affected = workspace::affected_by(&packages, &package.name);
        for other in packages
            .iter()
            .filter(|other| !affected.contains(&other.name))
        {
            for path in representative_files(other) {
                if filter_matches(patterns, &path) {
                    extra.push(format!("{} runs on {path} ({})", package.name, other.name));
                }
            }
        }
    }
    assert!(
        extra.is_empty(),
        "these package filters select a package on a change to a crate it does not \
         depend on: {extra:?}"
    );
    // The leaf a pull request most often touches, checked by name.
    let vp8 = &filters["zvidlib-vp8"];
    assert!(filter_matches(vp8, "crates/zvidlib-vp8/src/decoder.rs"));
    assert!(!filter_matches(
        vp8,
        "crates/zvidlib-hevc-decoder/src/lib.rs"
    ));
    assert!(!filter_matches(
        vp8,
        "crates/zvidlib-av1-encoder/src/lib.rs"
    ));
    assert!(filter_matches(
        &filters["zvidlib"],
        "crates/zvidlib-vp8/src/decoder.rs"
    ));
}

/// As `every_file_a_source_includes_runs_the_native_test_jobs`, per package: a
/// fixture a package's tests read from another crate's directory, or from the
/// bundled examples, is an input of that package.
#[test]
fn every_file_a_package_includes_runs_its_filter() {
    let filters = filters(&ci_workflow());
    let root = manifest_dir().canonicalize().expect("the root");
    let mut checked = 0;
    let mut missed = Vec::new();
    for package in workspace::packages() {
        let patterns = &filters[&package.name];
        let mut sources = Vec::new();
        let dirs: &[&str] = if package.dir == "." {
            &["src", "tests", "benches", "examples"]
        } else {
            &["src", "tests", "benches"]
        };
        for dir in dirs {
            rust_files(&package.path().join(dir), &mut sources);
        }
        for source in sources {
            let text = std::fs::read_to_string(&source)
                .unwrap_or_else(|error| panic!("reading {}: {error}", source.display()));
            for macro_name in ["include_str!(\"", "include_bytes!(\""] {
                for (at, _) in text.match_indices(macro_name) {
                    let rest = &text[at + macro_name.len()..];
                    let literal = &rest[..rest.find('"').expect("a closed string literal")];
                    let target = source.parent().expect("a source directory").join(literal);
                    let Ok(target) = target.canonicalize() else {
                        continue;
                    };
                    let Ok(path) = target.strip_prefix(&root) else {
                        continue;
                    };
                    let path = path.to_string_lossy().replace('\\', "/");
                    checked += 1;
                    if !filter_matches(patterns, &path)
                        && !filter_matches(&filters[EVERYTHING], &path)
                    {
                        missed.push(format!(
                            "{} -> {path} ({})",
                            relative(&source),
                            package.name
                        ));
                    }
                }
            }
        }
    }
    assert!(
        checked > 0,
        "found no `include_str!` or `include_bytes!` to check"
    );
    assert!(
        missed.is_empty(),
        "these files are compiled into a package's targets, so a change to them has to \
         select the package: {missed:?}"
    );
}

/// The packages the native test job lists with `targets`: every package has
/// to be listed, or selecting it would build and run none of its tests. The
/// Linux job runs a leg per package instead, which
/// `the_per_package_test_matrix_covers_every_package` checks.
#[test]
fn every_package_is_listed_in_the_native_test_job() {
    let workflow = ci_workflow();
    let jobs = jobs(&workflow);
    let packages = workspace::packages();
    let listed: Vec<&str> = job(&jobs, "native-tests")
        .body
        .lines()
        .filter_map(|line| line.trim().strip_prefix("targets "))
        .filter_map(|rest| rest.split_whitespace().next())
        .collect();
    let missing: Vec<&str> = packages
        .iter()
        .map(|package| package.name.as_str())
        .filter(|name| !listed.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "`native-tests` lists no `targets` line for these packages, so their library units \
         never run: {missing:?}"
    );
}

/// The package names the `changes` job expands `all` to, from its
/// `WORKSPACE: |` block, one per line.
fn listed_workspace(changes: &Job) -> Vec<String> {
    let mut lines = changes.body.lines();
    let header = lines
        .by_ref()
        .find(|line| line.trim() == "WORKSPACE: |")
        .expect("the `changes` job lists the workspace's packages under `WORKSPACE: |`");
    let block_indent = indent(header);
    lines
        .take_while(|line| line.trim().is_empty() || indent(line) > block_indent)
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// The per-package test matrix runs one leg per name in `crates`, and on a
/// `main` push that is every name the `changes` job lists. A package missing
/// from that list would never be tested on `main`, and one a pull request
/// selected would be dropped from `crates` and go untested there too (#612).
#[test]
fn the_per_package_test_matrix_covers_every_package() {
    let workflow = ci_workflow();
    let jobs = jobs(&workflow);
    let listed = listed_workspace(job(&jobs, "changes"));
    let mut packages: Vec<String> = workspace::packages()
        .into_iter()
        .map(|package| package.name)
        .collect();
    packages.sort();
    let mut sorted = listed.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        listed.len(),
        "`WORKSPACE` lists a package twice: {listed:?}"
    );
    assert_eq!(
        sorted, packages,
        "`WORKSPACE` in the `changes` job must list every workspace package and nothing \
         else, or the per-package test matrix skips the ones it leaves out"
    );
    let tests = job(&jobs, PER_PACKAGE_TESTS);
    assert!(
        tests.body.lines().any(|line| line.trim()
            == format!("package: ${{{{ fromJSON(needs.changes.outputs.{CRATES}) }}}}")),
        "`{PER_PACKAGE_TESTS}` no longer runs one leg per package in the `{CRATES}` output"
    );
    assert!(
        !workflow.contains("--partition"),
        "a job deals its tests out over shards again; each package runs its own (#612)"
    );
}

/// A step limited to one leg of the matrix, such as the libvpx VP9 vectors,
/// runs only while that leg exists: a condition naming a package the workspace
/// no longer has would skip the step on every run without saying so.
#[test]
fn a_step_for_one_package_names_a_workspace_package() {
    let workflow = ci_workflow();
    let packages = workspace::packages();
    let mut named = 0;
    for (at, _) in workflow.match_indices("matrix.package == '") {
        let rest = &workflow[at + "matrix.package == '".len()..];
        let name = &rest[..rest.find('\'').expect("a closed package name")];
        workspace::package(&packages, name);
        named += 1;
    }
    assert!(
        named >= 2,
        "expected the Opus and VP9 vector steps to be limited to their packages' legs"
    );
}

/// The packages a pull request's change selects, as the `changes` job would:
/// every package whose filter matches one of the paths.
fn selected(filters: &BTreeMap<String, Vec<String>>, paths: &[&str]) -> Vec<String> {
    workspace::packages()
        .into_iter()
        .map(|package| package.name)
        .filter(|name| {
            paths
                .iter()
                .any(|path| filter_matches(&filters[name], path))
        })
        .collect()
}

/// #612's acceptance cases, by name: a leaf crate's change tests that crate and
/// the crates depending on it and nothing unrelated, and a shared crate's
/// change tests every crate that depends on it.
#[test]
fn a_change_selects_its_crate_and_the_crates_that_depend_on_it() {
    let filters = filters(&ci_workflow());
    let packages = workspace::packages();
    let dependents = |name: &str| -> Vec<String> {
        packages
            .iter()
            .filter(|package| workspace::affected_by(&packages, &package.name).contains(name))
            .map(|package| package.name.clone())
            .collect()
    };

    let vp8 = selected(&filters, &["crates/zvidlib-vp8/src/lib.rs"]);
    assert_eq!(vp8, dependents("zvidlib-vp8"));
    assert_eq!(vp8, ["zvidlib", "zvidlib-vp8"]);

    let core = selected(&filters, &["crates/zvidlib-core/src/lib.rs"]);
    assert_eq!(core, dependents("zvidlib-core"));
    for name in ["zvidlib-aac-encoder", "zvidlib-hevc-decoder", "zvidlib-vp8"] {
        assert!(
            core.contains(&name.to_string()),
            "{name} depends on zvidlib-core"
        );
    }
}

#[test]
fn every_pattern_uses_only_the_glob_syntax_this_test_reads() {
    let filters = filters(&ci_workflow());
    for (name, patterns) in &filters {
        assert!(!patterns.is_empty(), "filter `{name}` has no patterns");
        for pattern in patterns {
            let body = pattern.strip_prefix('!').unwrap_or(pattern);
            assert!(
                !body.contains(['?', '[', ']', '(', ')', '+', '@', '!', '\\']),
                "`{name}` has the pattern `{pattern}`, which uses glob syntax this test does \
                 not read; keep to `*`, `**` and `{{a,b}}`, or teach the reader here first"
            );
            assert!(
                !body.starts_with('/') && !body.starts_with("./"),
                "`{pattern}` is not repository-relative the way paths-filter matches"
            );
        }
    }
}

/// The reader itself, on cases checked against picomatch with `dot: true`,
/// which is how `dorny/paths-filter` matches.
#[test]
fn the_glob_reader_agrees_with_picomatch() {
    let cases: &[(&str, &str, bool)] = &[
        ("src/**", "src/lib.rs", true),
        ("src/**", "src/hevc/engine/mod.rs", true),
        ("src/**", "srcx/lib.rs", false),
        ("src/**", "tests/src/lib.rs", false),
        ("examples/**/*.rs", "examples/native_encode.rs", true),
        ("examples/**/*.rs", "examples/native_gl/main.rs", true),
        ("examples/**/*.rs", "examples/web_canvas/main.js", false),
        (".github/**", ".github/workflows/ci.yml", true),
        ("Cargo.toml", "Cargo.toml", true),
        ("Cargo.toml", "tools/Cargo.toml", false),
        ("src/web_*.rs", "src/web_decoder.rs", true),
        ("src/web_*.rs", "src/hevc/web_decoder.rs", false),
        ("src/hevc/{nvdec,planar}.rs", "src/hevc/planar.rs", true),
        ("src/hevc/{nvdec,planar}.rs", "src/hevc/encoder.rs", false),
    ];
    for (pattern, path, expected) in cases {
        assert_eq!(
            glob_matches(pattern, path),
            *expected,
            "`{pattern}` against `{path}`"
        );
    }
    let patterns: Vec<String> = ["src/**", "!src/a/**", "!src/b.rs"]
        .iter()
        .map(|pattern| pattern.to_string())
        .collect();
    assert!(filter_matches(&patterns, "src/c.rs"));
    assert!(!filter_matches(&patterns, "src/a/mod.rs"));
    assert!(!filter_matches(&patterns, "src/b.rs"));
    assert!(!filter_matches(&patterns, "tests/b.rs"));
}
