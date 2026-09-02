#!/usr/bin/env python3
"""Collect criterion estimates into a baseline file and diff two baselines.

Criterion already tracks a previous run, but only in `target/criterion/`, which
does not survive a CI runner. This script turns a completed run into one small
JSON file that a workflow can store as an artifact, and compares two such files
into a Markdown report.

Both halves deliberately do nothing clever with statistics. Criterion's own
change detection needs its raw sample data; this compares point estimates across
two different physical machines drawn from a shared runner pool, where the noise
floor is far larger than anything a t-test on the samples would resolve. A loose
percentage threshold on the median is the honest instrument for that.

    collect --criterion-dir target/criterion --out baseline.json
    compare --previous old.json --current new.json --out report.md
    table --baseline run1.json run2.json --host 'Apple M1' --out table.md
    variance --baseline run1.json run2.json run3.json --out variance.md
    staleness --readme benches/README.md --out staleness.md

`table` renders the committed scalar-vs-ISA table in `benches/README.md`. It
takes the elementwise *minimum* across however many baselines it is given,
because a contended host can only ever make a measurement slower: the fastest
observed time for an arm is the closest any of the runs got to the uncontended
one. That is also why it wants several runs rather than one.

`variance` answers the question `compare`'s threshold was guessed at: given the
baselines from a run of consecutive `main` pushes, how far does a benchmark move
between two runs when nothing about it changed? It reduces those baselines to
the per-group spread of the same delta `compare` reports, so a threshold can be
read off measured noise instead of picked. It reports per group rather than one
number for the suite because a whole-frame 1080p group and a microbenchmark do
not share a noise floor.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import re
import subprocess
import sys

SCHEMA = 1

# Median rather than mean: a single descheduled iteration on a shared runner
# drags the mean and leaves the median alone.
METRIC = "median"

# The dispatch-site registry `staleness` reads, at HEAD and at a table's stamp.
SIMD_SOURCE = "src/simd.rs"


def _point_estimate(estimates: dict, metric: str) -> float | None:
    entry = estimates.get(metric)
    if not isinstance(entry, dict):
        return None
    value = entry.get("point_estimate")
    return float(value) if isinstance(value, (int, float)) else None


def collect(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.criterion_dir)
    if not root.is_dir():
        print(f"error: {root} does not exist; did the benchmark run?", file=sys.stderr)
        return 1

    benchmarks: dict[str, dict[str, float]] = {}
    for estimates_path in sorted(root.glob("**/new/estimates.json")):
        try:
            estimates = json.loads(estimates_path.read_text())
        except (OSError, ValueError) as error:
            print(f"warning: skipping {estimates_path}: {error}", file=sys.stderr)
            continue
        directory = estimates_path.parent
        # `target/criterion/report/` holds rendered HTML, not measurements.
        if "report" in directory.parts:
            continue
        median = _point_estimate(estimates, METRIC)
        if median is None:
            continue
        identifier = _benchmark_id_relative(directory, root)
        benchmarks[identifier] = {
            "median_ns": median,
            "mean_ns": _point_estimate(estimates, "mean") or median,
            "std_dev_ns": _point_estimate(estimates, "std_dev") or 0.0,
        }

    if not benchmarks:
        print(f"error: no criterion estimates under {root}", file=sys.stderr)
        return 1

    baseline = {
        "schema": SCHEMA,
        "metric": METRIC,
        "commit": args.commit,
        "run_url": args.run_url,
        "host": args.host,
        "benchmarks": benchmarks,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(benchmarks)} benchmark(s) to {out}")
    return 0


def _benchmark_id_relative(directory: pathlib.Path, root: pathlib.Path) -> str:
    """The criterion id for a `.../new/` directory.

    `benchmark.json` carries the unsanitised `full_id` (`av1_deblock/scalar`),
    which is what a reader recognises; criterion sanitises the directory names
    themselves. Falling back to the path keeps a benchmark in the report if that
    file is ever missing rather than dropping it silently.
    """
    metadata = directory / "benchmark.json"
    if metadata.is_file():
        try:
            full_id = json.loads(metadata.read_text()).get("full_id")
        except (OSError, ValueError):
            full_id = None
        if isinstance(full_id, str) and full_id:
            return full_id
    return "/".join(directory.parent.relative_to(root).parts)


def _group_of(identifier: str) -> str:
    """The criterion group an id belongs to.

    Ids are `<group>/<benchmark>`, and this suite's group names themselves
    contain a slash (`smoke/simd=off`, and `av1_deblock/scalar` where the arm is
    the instruction set), so the split is on the last separator.
    """
    head, _, tail = identifier.rpartition("/")
    return head or tail


def compare(args: argparse.Namespace) -> int:
    current = json.loads(pathlib.Path(args.current).read_text())
    lines: list[str] = ["## Benchmark deltas", ""]

    previous_path = pathlib.Path(args.previous) if args.previous else None
    if previous_path is None or not previous_path.is_file():
        lines += [
            "No previous baseline was available, so this run establishes one.",
            "",
            f"Recorded {len(current.get('benchmarks', {}))} benchmark(s) "
            f"at `{current.get('commit') or 'unknown'}`.",
            "",
        ]
        _write(args.out, lines)
        return 0

    previous = json.loads(previous_path.read_text())
    old = previous.get("benchmarks", {})
    new = current.get("benchmarks", {})

    regressions: list[str] = []
    rows: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for identifier in sorted(set(old) | set(new)):
        before = old.get(identifier, {}).get("median_ns")
        after = new.get(identifier, {}).get("median_ns")
        group = _group_of(identifier)
        arm = identifier.rpartition("/")[2]
        if before is None:
            rows.setdefault(group, []).append(
                (arm, "—", _duration(after), "new", "🆕")
            )
            continue
        if after is None:
            rows.setdefault(group, []).append(
                (arm, _duration(before), "—", "gone", "⚠️")
            )
            regressions.append(f"`{identifier}` disappeared from the suite")
            continue
        delta = (after - before) / before * 100.0 if before else 0.0
        if delta > args.threshold:
            marker = "🔴"
            regressions.append(f"`{identifier}` is {delta:+.1f}% slower")
        elif delta < -args.threshold:
            marker = "🟢"
        else:
            marker = ""
        rows.setdefault(group, []).append(
            (arm, _duration(before), _duration(after), f"{delta:+.1f}%", marker)
        )

    lines += [
        f"Comparing `{current.get('commit') or 'this run'}` against "
        f"`{previous.get('commit') or 'the previous main run'}`"
        + (f" ([previous run]({previous['run_url']}))" if previous.get("run_url") else "")
        + ".",
        "",
        f"Metric: criterion's **{current.get('metric', METRIC)}** point estimate. "
        f"Threshold: **{args.threshold:g}%**.",
        "",
    ]
    for group in sorted(rows):
        lines += [
            f"### `{group}`",
            "",
            "| Arm | Previous | Current | Delta | |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for arm, before, after, delta, marker in rows[group]:
            lines.append(f"| `{arm}` | {before} | {after} | {delta} | {marker} |")
        lines.append("")

    if regressions:
        lines += [
            f"### ⚠️ {len(regressions)} regression(s) past {args.threshold:g}%",
            "",
        ]
        lines += [f"- {item}" for item in regressions]
        lines += [
            "",
            "This is a report, not a gate. Shared CI runners have a noise floor "
            "well above a few percent, so a single crossing here is a prompt to "
            "look, not a verdict. Reproduce on a quiet host before treating it "
            "as real.",
            "",
        ]
    else:
        lines += [f"No benchmark moved by more than {args.threshold:g}%.", ""]

    _write(args.out, lines)
    if regressions and args.fail_on_regression:
        return 1
    return 0


def table(args: argparse.Namespace) -> int:
    """Render the committed scalar-vs-ISA table from one or more baselines.

    Only the groups whose arms are instruction sets are in scope: those are the
    ones `bench_across_isas` builds, and they are the only ones for which
    "scalar vs each ISA" is a question. Everything else in the suite is a single
    arm with nothing to compare it against, and listing it here would suggest a
    comparison that does not exist.
    """
    merged: dict[str, float] = {}
    hosts: list[str] = []
    commits: list[str] = []
    for path in args.baseline:
        baseline = json.loads(pathlib.Path(path).read_text())
        recorded = baseline.get("host")
        if recorded and recorded not in hosts:
            hosts.append(recorded)
        commit = baseline.get("commit")
        if commit and commit not in commits:
            commits.append(commit)
        for identifier, entry in baseline.get("benchmarks", {}).items():
            median = entry.get("median_ns")
            if not isinstance(median, (int, float)):
                continue
            # Minimum, not mean: contention only ever adds time, so the fastest
            # observation is the least contaminated one.
            if identifier not in merged or median < merged[identifier]:
                merged[identifier] = float(median)

    # Which arms are instruction sets is read off the data rather than kept in a
    # list here. The arm names come from `SimdIsa`'s `Display` on the Rust side
    # ("sse4.1", not "sse41"), and a hardcoded allowlist that drifts from it
    # drops the column silently: the group still has the arm, the table just
    # stops having somewhere to put it, which reads as a host that could not run
    # it. `bench_across_isas` always emits a `scalar` arm, so a group having one
    # is what identifies it as a per-ISA group, and every other arm of such a
    # group is an instruction set by construction - including one added after
    # this script was last touched.
    groups: dict[str, dict[str, float]] = {}
    for identifier, median in merged.items():
        group, _, arm = identifier.rpartition("/")
        if group:
            groups.setdefault(group, {})[arm] = median
    groups = {group: arms for group, arms in groups.items() if "scalar" in arms}

    if not groups:
        print("error: no per-instruction-set groups in the given baselines", file=sys.stderr)
        return 1

    # Scalar is the reference every other column is a ratio against, so it leads.
    # The rest are ordered narrowest-first where the width is known, so the
    # columns read as a progression, and alphabetically after that so an
    # unrecognised instruction set still lands somewhere stable instead of
    # moving between runs.
    width_order = ["sse4.1", "avx2", "avx512", "neon"]

    def isa_sort_key(isa: str) -> tuple[int, str]:
        return (width_order.index(isa), "") if isa in width_order else (len(width_order), isa)

    found = {isa for arms in groups.values() for isa in arms if isa != "scalar"}
    present = ["scalar"] + sorted(found, key=isa_sort_key)
    host = args.host or (hosts[0] if hosts else "unknown host")

    # The commit is part of the measurement, not decoration: kernels land often
    # enough that a table without one cannot be told from a stale table.
    if len(commits) == 1:
        provenance = f"Measured on **{host}**, at `{commits[0][:12]}`."
    elif commits:
        provenance = (
            f"Measured on **{host}**, across {len(commits)} commits "
            f"(newest `{commits[-1][:12]}`)."
        )
    else:
        provenance = f"Measured on **{host}**."

    lines = [
        provenance,
        "",
        "| Group | " + " | ".join(f"`{isa}`" for isa in present) + " | Best |",
        "| --- | " + " | ".join("---:" for _ in present) + " | ---: |",
    ]
    for group in sorted(groups):
        arms = groups[group]
        scalar = arms.get("scalar")
        cells = []
        for isa in present:
            median = arms.get(isa)
            if median is None:
                cells.append("—")
            elif isa == "scalar" or scalar is None:
                cells.append(_duration(median))
            else:
                cells.append(f"{_duration(median)} ({scalar / median:.2f}x)")
        vector = {isa: arms[isa] for isa in present if isa != "scalar" and isa in arms}
        if scalar and vector:
            fastest = min(vector, key=lambda isa: vector[isa])
            best = f"{scalar / vector[fastest]:.2f}x `{fastest}`"
        else:
            best = "—"
        lines.append(f"| `{group}` | " + " | ".join(cells) + f" | {best} |")
    lines.append("")

    _write(args.out, lines)
    return 0


# Below this many consecutive-run pairs the tail of the distribution is not
# measured, it is guessed: a p95 read off a handful of samples is the largest
# thing that happened to be seen, which is not the same quantity. The report is
# still rendered, and says so.
MIN_PAIRS_FOR_A_TAIL = 10


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than an interpolating definition because these are
    observed deltas from a small sample; interpolating between two measurements
    invents a number that was never measured.
    """
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percentile / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _suggested_threshold(worst: float) -> int:
    """The smallest whole 5% step strictly above the worst observed move.

    A threshold sitting exactly on the worst observation flags it, so the step
    is strict. Rounding to 5% rather than to the observation itself keeps the
    number from reading as more precise than the sample behind it.
    """
    return int(math.floor(worst / 5.0) + 1) * 5


def variance(args: argparse.Namespace) -> int:
    """Report the per-group run-to-run spread across successive baselines.

    Every consecutive pair of baselines contributes one delta per benchmark
    present in both, which is exactly the quantity `compare` measures a
    threshold against. Taking the absolute value folds the two directions
    together on purpose: the threshold is symmetric, and a group that swings
    -20% between two runs of the same code will swing +20% just as readily.
    """
    if len(args.baseline) < 2:
        print("error: variance needs at least two baselines to form a pair", file=sys.stderr)
        return 1

    loaded: list[dict] = []
    for path in args.baseline:
        try:
            loaded.append(json.loads(pathlib.Path(path).read_text()))
        except (OSError, ValueError) as error:
            print(f"error: could not read {path}: {error}", file=sys.stderr)
            return 1

    # Arms that come and go are not deltas and must not be averaged into one.
    # They are reported separately because an arm that vanished when the runner
    # pool changed is the one thing a percentage cannot express, and it settles
    # whether a group can be gated on at all.
    deltas: dict[str, list[float]] = {}
    unstable: dict[str, set[str]] = {}
    pairs = 0
    for previous, current in zip(loaded, loaded[1:]):
        pairs += 1
        old = previous.get("benchmarks", {})
        new = current.get("benchmarks", {})
        for identifier in sorted(set(old) | set(new)):
            group = _group_of(identifier)
            before = old.get(identifier, {}).get("median_ns")
            after = new.get(identifier, {}).get("median_ns")
            if not before or after is None:
                unstable.setdefault(group, set()).add(identifier)
                continue
            deltas.setdefault(group, []).append(abs((after - before) / before * 100.0))

    if not deltas:
        print("error: no benchmark appears in two consecutive baselines", file=sys.stderr)
        return 1

    everything = sorted(value for values in deltas.values() for value in values)
    lines: list[str] = [
        "## Benchmark run-to-run variance",
        "",
        f"{len(loaded)} baseline(s), {pairs} consecutive pair(s), "
        f"{len(everything)} observed delta(s) across {len(deltas)} group(s).",
        "",
        "Each delta is one benchmark's `|median|` change between two successive "
        "runs, the same quantity the delta report thresholds. Absolute value: "
        "the threshold is symmetric.",
        "",
    ]
    if pairs < args.min_pairs:
        lines += [
            f"> ⚠️ **Provisional.** {pairs} pair(s) is below the {args.min_pairs} "
            "this report wants before its p95 column means anything. With this "
            "few samples the p95 is just the worst thing seen so far, and a "
            "threshold read off it will be too tight. Collect more `main` runs.",
            "",
        ]

    lines += [
        "| Group | n | Median | p95 | Max | Suggested threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in sorted(deltas):
        values = sorted(deltas[group])
        worst = values[-1]
        lines.append(
            f"| `{group}` | {len(values)} | {_percentile(values, 50):.1f}% | "
            f"{_percentile(values, 95):.1f}% | {worst:.1f}% | "
            f"{_suggested_threshold(worst)}% |"
        )
    lines += [
        f"| **whole suite** | {len(everything)} | {_percentile(everything, 50):.1f}% | "
        f"{_percentile(everything, 95):.1f}% | {everything[-1]:.1f}% | "
        f"{_suggested_threshold(everything[-1])}% |",
        "",
        "The suggested threshold is the smallest whole 5% step above the worst "
        "delta observed for that group, so nothing in this sample would have "
        "been flagged. It is a floor on a defensible threshold, not a "
        "recommendation on its own: a sample that never saw a bad run suggests "
        "a number a bad run will cross.",
        "",
    ]

    # A group can only be gated on if its own noise fits under the gate, so this
    # is the whole answer to "is anything stable enough for --fail-on-regression".
    gateable = [
        group
        for group in sorted(deltas)
        if group not in unstable and max(deltas[group]) < args.gate_at
    ]
    lines += [f"### Groups whose spread fits under {args.gate_at:g}%", ""]
    if gateable:
        lines += [f"- `{group}`" for group in gateable]
        lines += [
            "",
            "Fitting under the gate in this sample is a precondition for "
            "`--fail-on-regression`, not a reason to enable it.",
            "",
        ]
    else:
        lines += [
            f"None. No group stayed inside {args.gate_at:g}% across every pair "
            "in this sample, so nothing here is a candidate for gating yet.",
            "",
        ]

    if unstable:
        lines += [
            "### Arms that appeared or disappeared",
            "",
            "These contribute no delta. An arm present in one run and absent "
            "from the next usually means the runner pool changed instruction "
            "sets, not that anything regressed — but a group that does this "
            "cannot be gated on.",
            "",
        ]
        for group in sorted(unstable):
            for identifier in sorted(unstable[group]):
                lines.append(f"- `{identifier}`")
        lines.append("")

    _write(args.out, lines)
    return 0


# The dispatch sites `zvidlib::simd::active_by_site` registers, mapped to the
# benchmark groups whose row in a committed table *is* that site's own number.
#
# Whole-frame groups are deliberately absent. `av1_decode_frame` or
# `hevc_encode_640x352` cross every site at once, so a landed kernel moves them
# by an amount no single row can be attributed to, and naming them would bury
# the rows that can be. So are the groups with no vector kernel at all
# (`hevc_cabac`, `av1_encode_stage_symbol`): they are the same code on both
# arms whatever lands.
#
# Prefixes, not exact names, because the transform families are one site each
# spread over a row per size. `test_criterion_baseline.py` asserts every site
# documented in `src/simd.rs` appears here, so a new dispatch site cannot be
# added without deciding which rows it invalidates - the same guard
# `the_documented_site_table_lists_every_dispatch_site` puts on the site table
# itself.
SITE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "av1_simd": (
        "av1_cdef",
        "av1_deblock",
        "av1_wiener",
        "av1_self_guided",
        "av1_inverse_",
        "av1_forward_",
        "av1_encode_stage_wht",
        "av1_encode_stage_iwht",
    ),
    "av1_mc": ("av1_mc_", "av1_motion_compensation"),
    "av1_intra_pred": ("av1_intra_",),
    "av1_coeff_ctx": ("av1_encode_stage_coeff_ctx", "av1_encode_stage_tile"),
    "hevc_prediction_filters": (
        "hevc_inter_pred",
        "hevc_intra_pred",
        "hevc_deblock",
        "hevc_sao",
    ),
    "hevc_transforms": ("hevc_inverse_transform",),
    "hevc_rdcost": ("hevc_encode_640x352_rdo_",),
    "hevc_recon": ("hevc_encode_640x352_reconstruct",),
    "hevc_fwd_transform_quant": ("hevc_encode_640x352_fwd_transform_quant",),
    "hevc_colorconv": ("hevc_encode_640x352_rgba_to_yuv420",),
    "hevc_color_convert": ("hevc_color_convert",),
}

# `Measured on **<host>**, at `<sha>`.` - the line `table` renders above every
# committed baseline. The host is bolded there and nowhere else in the file,
# which is what keeps this from matching the prose that discusses *superseded*
# draws by commit ("the one #261 recorded at `e115506f8bf6`").
_STAMP = re.compile(r"Measured on \*\*(?P<host>[^*]+)\*\*,\s+at\s+`(?P<commit>[0-9a-f]{7,40})`")

# The `| Site | Kernels |` table in `active_by_site`'s rustdoc. A dispatch site
# is a Rust value, and reading it out of a build would mean building every
# commit a table is stamped with; the doc table is the same set by test, and it
# can be read out of a blob.
_SITE_ROW = re.compile(r"^///\s*\|\s*`(?P<site>[a-z0-9_]+)`\s*\|")


def _documented_dispatch_sites(source: str) -> list[str]:
    """The site names `active_by_site`'s rustdoc table lists, in order."""
    sites: list[str] = []
    for line in source.splitlines():
        match = _SITE_ROW.match(line)
        if match:
            site = match.group("site")
            if site not in sites:
                sites.append(site)
    return sites


def _table_rows(readme: str, start: int) -> list[str]:
    """The group names of the first `| Group |` table at or after `start`."""
    rows: list[str] = []
    seen_header = False
    for line in readme[start:].splitlines():
        stripped = line.strip()
        if not seen_header:
            if stripped.startswith("| Group |"):
                seen_header = True
            continue
        if not stripped.startswith("|"):
            break
        cell = stripped.split("|")[1].strip()
        if cell.startswith("`") and cell.endswith("`"):
            rows.append(cell.strip("`"))
    return rows


def _committed_tables(readme: str) -> list[dict]:
    return [
        {
            "host": match.group("host"),
            "commit": match.group("commit"),
            "groups": _table_rows(readme, match.end()),
        }
        for match in _STAMP.finditer(readme)
    ]


def _sites_at_commit(commit: str, repo_root: pathlib.Path, slug: str | None) -> list[str] | None:
    """`src/simd.rs`'s documented sites as of `commit`, or None if unreadable.

    Local git first, then the GitHub contents API. The fallback is not a
    convenience: a table is stamped with the commit that measured it, and that
    is routinely a checkpoint commit on a branch whose ref is deleted at merge,
    so `git show` fails on exactly the commits this check exists to read. Both
    tables committed today are such a commit.
    """
    source = _run(["git", "-C", str(repo_root), "show", f"{commit}:{SIMD_SOURCE}"])
    if source is not None:
        return _documented_dispatch_sites(source)
    if not slug:
        return None
    encoded = _run(
        ["gh", "api", f"repos/{slug}/contents/{SIMD_SOURCE}?ref={commit}", "--jq", ".content"]
    )
    if encoded is None:
        return None
    try:
        return _documented_dispatch_sites(base64.b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _repository_slug(repo_root: pathlib.Path) -> str | None:
    url = _run(["git", "-C", str(repo_root), "remote", "get-url", "origin"])
    if not url:
        return None
    match = re.search(r"github\.com[:/](?P<slug>[^/]+/[^/\s]+?)(?:\.git)?\s*$", url)
    return match.group("slug") if match else None


def staleness(args: argparse.Namespace) -> int:
    """Report committed baseline rows whose dispatch site postdates the table.

    Each table in `benches/README.md` is stamped with the commit it was measured
    at, and nothing until now compared that commit's dispatch-site set against
    the one the crate has today. A row silently stops describing the crate the
    moment a kernel family lands under it, and the only signal was somebody
    noticing a ratio that moved for no attributable reason - which is the whole
    of the work issue #361 turned out to be.

    This reports and never fails, in the same spirit as the delta report's 15%
    threshold: a stale table is a measurement to redraw, not a broken build.
    """
    repo_root = pathlib.Path(args.repo_root)
    readme = pathlib.Path(args.readme).read_text()
    head_sites = _documented_dispatch_sites((repo_root / SIMD_SOURCE).read_text())
    tables = _committed_tables(readme)

    lines = ["## Committed baseline staleness", ""]
    if not head_sites:
        print(f"error: no dispatch sites documented in {SIMD_SOURCE}", file=sys.stderr)
        return 1
    if not tables:
        print(f"error: no stamped baseline tables in {args.readme}", file=sys.stderr)
        return 1

    slug = _repository_slug(repo_root)
    flagged = 0
    for entry in tables:
        commit, host = entry["commit"], entry["host"]
        lines.append(f"### {host} — `{commit}`")
        lines.append("")
        then = _sites_at_commit(commit, repo_root, slug)
        if then is None:
            lines.append(
                f"Could not read `{SIMD_SOURCE}` at `{commit}`: neither this "
                "clone nor the GitHub contents API has that commit. The table "
                "is unverified rather than clean."
            )
            lines.append("")
            continue

        landed = [site for site in head_sites if site not in then]
        retired = [site for site in then if site not in head_sites]
        if not landed and not retired:
            lines.append(
                f"Clean: the same {len(then)} dispatch sites are registered now "
                "as when this table was measured."
            )
            lines.append("")
            continue

        if retired:
            lines.append(
                "Dispatch sites present at the stamp and gone now: "
                + ", ".join(f"`{site}`" for site in retired)
                + "."
            )
            lines.append("")
        if landed:
            lines.append(
                f"{len(landed)} dispatch site(s) landed after this table was "
                "measured: " + ", ".join(f"`{site}`" for site in landed) + "."
            )
            lines.append("")
            affected = [
                (group, site)
                for group in entry["groups"]
                for site in landed
                if _group_matches_site(group, site)
            ]
            if affected:
                flagged += len(affected)
                lines.append("| Row | Dispatch site that landed after the stamp |")
                lines.append("| --- | --- |")
                lines.extend(f"| `{group}` | `{site}` |" for group, site in affected)
            else:
                lines.append(
                    "No row of this table is attributed to any of them, so the "
                    "rows it does have still describe the crate."
                )
            lines.append("")

    lines.append(
        f"{flagged} row(s) flagged. A flagged row was measured before the "
        "dispatch site it names existed, so its ratio describes code that is "
        "no longer there; redrawing the table is what clears it. This is a "
        "report, not a gate."
    )
    _write(args.out, lines)
    return 0


def _group_matches_site(group: str, site: str) -> bool:
    """Does `group` measure `site`?

    The `_1080p` variants are the same group over a larger frame, so they are
    stripped rather than listed twice.
    """
    name = group[: -len("_1080p")] if group.endswith("_1080p") else group
    return any(name.startswith(prefix) for prefix in SITE_GROUP_PREFIXES.get(site, ()))


def _duration(nanoseconds: float | None) -> str:
    if nanoseconds is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if nanoseconds >= scale:
            return f"{nanoseconds / scale:.3f} {unit}"
    return f"{nanoseconds:.1f} ns"


def _write(destination: str | None, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    if destination:
        path = pathlib.Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sys.stdout.write(text)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    collector = subcommands.add_parser("collect", help="summarise a criterion run")
    collector.add_argument("--criterion-dir", default="target/criterion")
    collector.add_argument("--out", required=True)
    collector.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    collector.add_argument("--run-url", default="")
    collector.add_argument("--host", default="")
    collector.set_defaults(handler=collect)

    comparer = subcommands.add_parser("compare", help="diff two baseline files")
    comparer.add_argument("--previous")
    comparer.add_argument("--current", required=True)
    comparer.add_argument("--out")
    comparer.add_argument("--threshold", type=float, default=15.0)
    comparer.add_argument("--fail-on-regression", action="store_true")
    comparer.set_defaults(handler=compare)

    tabler = subcommands.add_parser("table", help="render the committed baseline table")
    tabler.add_argument("--baseline", nargs="+", required=True)
    tabler.add_argument("--host", default="")
    tabler.add_argument("--out")
    tabler.set_defaults(handler=table)

    variancer = subcommands.add_parser(
        "variance", help="per-group run-to-run spread across successive baselines"
    )
    variancer.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="baseline files in chronological order, oldest first",
    )
    variancer.add_argument("--out")
    variancer.add_argument("--min-pairs", type=int, default=MIN_PAIRS_FOR_A_TAIL)
    variancer.add_argument(
        "--gate-at",
        type=float,
        default=15.0,
        help="candidate gate, in percent, to test each group's spread against",
    )
    variancer.set_defaults(handler=variance)

    staler = subcommands.add_parser(
        "staleness", help="flag committed table rows that predate a dispatch site"
    )
    staler.add_argument("--readme", default="benches/README.md")
    staler.add_argument(
        "--repo-root",
        default=".",
        help="the checkout to read src/simd.rs and the stamped commits from",
    )
    staler.add_argument("--out")
    staler.set_defaults(handler=staleness)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
