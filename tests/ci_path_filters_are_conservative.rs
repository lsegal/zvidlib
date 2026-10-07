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
//!   fails here instead of going unchecked.
//!
//! Deliberately line-based rather than a YAML parse, for the same reason
//! `ci_workflows_cache_cargo` is: the alternative is a `serde_yaml` dependency
//! for a hygiene check, and the filter block is written one pattern per line.
//! The globs are matched by a reader that knows only `*`, `**` and `{a,b}`, and
//! `every_pattern_uses_only_the_glob_syntax_this_test_reads` keeps the filters
//! to that subset of what `dorny/paths-filter`'s picomatch accepts.

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
    ("rust-tests-build", "rust_tests"),
    ("native-tests-build", "native_tests"),
    ("macos-swift-rpath", "macos_swift_rpath"),
];

/// Jobs that are skipped with the build job they need rather than by a filter
/// of their own: a shard has nothing to run without the archive.
const SKIPPED_WITH_THEIR_BUILD: &[(&str, &str)] = &[
    ("rust-tests", "rust-tests-build"),
    ("rust-tests-vp9-vectors", "rust-tests-build"),
    ("native-tests", "native-tests-build"),
];

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

/// The `cfg` an excluded path's module is declared under: `src/a/**` is module
/// `a` of `src/lib.rs`, `src/a.rs` likewise, and `src/a/b.rs` is module `b` of
/// `src/a/mod.rs` or `src/a.rs`.
fn excluded_module_cfg(excluded: &str) -> String {
    let root = manifest_dir();
    let path = excluded.strip_suffix("/**").unwrap_or(excluded);
    let file = Path::new(path);
    let name = file
        .file_stem()
        .expect("an excluded module has a name")
        .to_string_lossy()
        .into_owned();
    let dir = file.parent().expect("an excluded module has a directory");
    assert!(
        root.join(path).exists(),
        "`!{excluded}` names nothing in the repository; a renamed module is now matched \
         again, so this exclusion is dead and should follow it or go"
    );
    let parents = if dir == Path::new("src") {
        vec![root.join("src/lib.rs")]
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
    assert_eq!(
        outputs.len(),
        GATED_JOBS.len(),
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
    for (id, build) in SKIPPED_WITH_THEIR_BUILD {
        assert_eq!(
            job_key(job(&jobs, id), "needs").as_deref(),
            Some(*build),
            "`{id}` no longer needs `{build}`, so it would run without the archive it tests"
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
        ("rust", "src/hevc/engine/mod.rs", true),
        ("rust", "tests/opus_codec.rs", true),
        ("rust", "benches/codec.rs", true),
        ("rust", "examples/native_gl/main.rs", true),
        ("rust", "examples/web_canvas/scrub.test.js", true),
        ("rust", ".github/scripts/criterion_baseline.py", true),
        ("rust", "rustfmt.toml", true),
        // #600's own example: a change to native backends only skips these.
        ("wasm", "src/hevc/nvdec.rs", false),
        ("wasm", "src/hevc/windows_mf.rs", false),
        ("wasm", "src/native_audio/symphonia.rs", false),
        ("wasm", "src/aac_encoder/mod.rs", false),
        ("wasm", "src/lib.rs", true),
        ("wasm", "src/wasm_api.rs", true),
        ("wasm", "src/hevc/mod.rs", true),
        ("wasm", "src/hevc/engine/mod.rs", true),
        ("wasm", "src/hevc/color_convert.rs", true),
        ("wasm", "js/browser.js", true),
        ("wasm", "tests/fixtures/codec/vp9_bbb_256x144.mp4", true),
        ("wasm", "examples/hevc_decode_profile.rs", true),
        ("wasm", ".github/scripts/criterion_baseline.py", false),
        // The Linux shards run the guards, which read every workflow.
        ("rust_tests", "src/hevc/nvdec.rs", true),
        ("rust_tests", "src/wasm_api.rs", false),
        ("rust_tests", "js/browser.js", false),
        (
            "rust_tests",
            "tests/fixtures/codec/libvpx_vp9_test_vectors.txt",
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
        ("native_tests", "src/hevc/videotoolbox.rs", true),
        ("native_tests", "src/aac_encoder/windows_mf.rs", true),
        ("native_tests", "src/web_decoder.rs", false),
        (
            "native_tests",
            "tests/support/avfoundation_decode.swift",
            true,
        ),
        ("native_tests", "benches/hevc_hardware.rs", true),
        ("macos_swift_rpath", "src/hevc/videotoolbox.rs", true),
        (
            "macos_swift_rpath",
            "tests/macos_swift_runtime_rpath.rs",
            true,
        ),
        ("macos_swift_rpath", "tests/opus_codec.rs", false),
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
            assert!(
                excluded.starts_with("src/")
                    && !excluded.contains('{')
                    && (excluded.ends_with(".rs")
                        || excluded.ends_with("/**")
                            && !excluded[..excluded.len() - 3].contains('*')),
                "`{name}` excludes `{excluded}`; exclusions name one module, a `src/` file or a \
                 `src/` module directory, so each can be held to its `cfg`"
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
