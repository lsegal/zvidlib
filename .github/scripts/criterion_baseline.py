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
import json
import math
import os
import pathlib
import sys

SCHEMA = 1

# Median rather than mean: a single descheduled iteration on a shared runner
# drags the mean and leaves the median alone.
METRIC = "median"


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

    isa_names = ["scalar", "sse41", "avx2", "neon"]
    groups: dict[str, dict[str, float]] = {}
    for identifier, median in merged.items():
        group, _, arm = identifier.rpartition("/")
        if arm in isa_names and group:
            groups.setdefault(group, {})[arm] = median

    if not groups:
        print("error: no per-instruction-set groups in the given baselines", file=sys.stderr)
        return 1

    present = [isa for isa in isa_names if any(isa in arms for arms in groups.values())]
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

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
