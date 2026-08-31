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
"""

from __future__ import annotations

import argparse
import json
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

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
