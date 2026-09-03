//! Guards against two bench targets claiming one criterion group name.
//!
//! Criterion keys a group by its name and nothing else. The name is not
//! namespaced by the target that registered it, and every target in this crate
//! writes to the same `target/criterion/` tree, so two targets that both
//! register `av1_deblock` write the same `target/criterion/av1_deblock/<isa>`
//! directory: the target that runs second overwrites the first, and a baseline
//! collected afterwards can only ever see one of them.
//!
//! Nothing about that is visible while it is happening. Both groups run, both
//! print their timings, both pass their bit-exactness guard, and the collision
//! is silent unless somebody reads the log and notices the same group name
//! twice. It only surfaces later, as a committed baseline row that moved for no
//! reason any commit explains — issue #414, where `av1_deblock`'s `scalar` arm
//! read 25% slower between two x86_64 table draws while both of its vector arms
//! stayed within 1%. The two draws had run the bench targets in opposite orders
//! (`cargo bench` walks them alphabetically, so `codec` follows `av1_decode`;
//! the recipe in `benches/README.md` names `codec` first), so each collected the
//! opposite side of the collision. The scalar arms of the two groups are 27%
//! apart and their vector arms agree to 0.1%, which is exactly the shape the
//! issue reports.
//!
//! Deliberately line-based rather than a real parse, for the same reason
//! `bench_sources_compile.rs` is: a group name is written as a string literal
//! argument to one of four constructors, and the alternative is a `syn`
//! dependency for a hygiene check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The constructors a bench target names a criterion group through.
///
/// `group_name` and `benchmark_group` are the plain criterion path;
/// `IsaWorkload::new` and `kernel_workload` are `support::isa`'s, which is what
/// every `bench_across_isas` group — the only groups the committed baseline
/// tables have rows for — is built with.
const GROUP_CONSTRUCTORS: [&str; 4] = [
    "IsaWorkload::new(",
    "kernel_workload(",
    "group_name(",
    "benchmark_group(",
];

/// Paths of the `[[bench]]` targets `Cargo.toml` declares.
fn declared_bench_targets(manifest: &str) -> Vec<String> {
    let mut paths = Vec::new();
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
        if let Some(value) = line.strip_prefix("path") {
            let value = value.trim_start().trim_start_matches('=').trim();
            paths.push(value.trim_matches('"').to_string());
        }
    }
    paths
}

/// The string literal that opens `rest`, if `rest` starts with one.
///
/// A group name is sometimes written on the line after the constructor's open
/// parenthesis (`codec.rs` wraps its `IsaWorkload::new` calls that way), so the
/// caller passes everything that follows the parenthesis and this skips over
/// whitespace to reach the literal.
fn leading_string_literal(rest: &str) -> Option<&str> {
    let rest = rest.trim_start();
    let body = rest.strip_prefix('"')?;
    let end = body.find('"')?;
    Some(&body[..end])
}

/// Group names one bench target's crate root registers.
///
/// `av1_encode.rs` builds several of its names with `format!`, so a format
/// string's literal prefix counts too: `format!("av1_encode_stage_wht{suffix}")`
/// registers `av1_encode_stage_wht` and `av1_encode_stage_wht_1080p`, and the
/// prefix is the part two targets could collide on. A format string that opens
/// with its first placeholder contributes nothing, which is right — the name is
/// not knowable from the source.
fn group_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (index, _) in source.match_indices("format!(") {
        let rest = &source[index + "format!(".len()..];
        if let Some(literal) = leading_string_literal(rest) {
            let prefix = literal.split('{').next().unwrap_or_default();
            if !prefix.is_empty() {
                names.insert(prefix.to_string());
            }
        }
    }
    for constructor in GROUP_CONSTRUCTORS {
        for (index, _) in source.match_indices(constructor) {
            let rest = &source[index + constructor.len()..];
            if let Some(literal) = leading_string_literal(rest) {
                names.insert(literal.to_string());
            }
        }
    }
    names
}

#[test]
fn no_two_bench_targets_register_the_same_group_name() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest =
        std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml is readable");
    let targets = declared_bench_targets(&manifest);
    assert!(
        !targets.is_empty(),
        "Cargo.toml declares no [[bench]] targets, so this guard would pass vacuously"
    );

    // name -> the targets that register it, so the failure names both sides of
    // a collision rather than only the one that happened to be read second.
    let mut owners: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for target in &targets {
        let path = root.join(target);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} is readable: {error}", path.display()));
        for name in group_names(&source) {
            owners.entry(name).or_default().push(target.clone());
        }
    }

    assert!(
        owners.values().any(|targets| targets.len() == 1),
        "no group name was found in any bench target, so this guard would pass vacuously"
    );

    let collisions: Vec<String> = owners
        .iter()
        .filter(|(_, targets)| targets.len() > 1)
        .map(|(name, targets)| format!("`{name}` in {}", targets.join(" and ")))
        .collect();
    assert!(
        collisions.is_empty(),
        "these criterion group names are registered by more than one bench target, so \
         whichever target runs second silently overwrites the other's results under \
         target/criterion/ and a collected baseline records only one of them: {}",
        collisions.join("; ")
    );
}

#[test]
fn a_group_name_is_read_from_every_shape_a_bench_target_writes_one_in() {
    let source = r#"
        let workload = IsaWorkload::new(
            "wrapped_across_lines",
            FrameWork::new(1, 2, 3),
        );
        let workload = kernel_workload("same_line", work);
        let name = group_name("plain_criterion");
        let mut group = criterion.benchmark_group("group_taken_directly");
        let name = format!("formatted_prefix{suffix}");
        let name = format!("{leading_placeholder}_tail");
    "#;

    assert_eq!(
        group_names(source),
        [
            "formatted_prefix",
            "group_taken_directly",
            "plain_criterion",
            "same_line",
            "wrapped_across_lines",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>(),
        "a name written in one of these shapes would be invisible to the collision check"
    );
}

#[test]
fn the_collision_this_guard_exists_for_is_one_it_would_have_caught() {
    // `benches/codec.rs` as it read before #414 renamed its group, against
    // `benches/av1_decode.rs`'s unchanged one.
    let codec_before = r#"IsaWorkload::new(
            "av1_deblock",
            FrameWork::new(1, ISA_WIDTH as u64, ISA_HEIGHT as u64),
        )"#;
    let av1_decode = r#"kernel_workload("av1_deblock", frame_work())"#;

    let shared: Vec<String> = group_names(codec_before)
        .intersection(&group_names(av1_decode))
        .cloned()
        .collect();
    assert_eq!(shared, vec!["av1_deblock".to_string()]);
}
