#!/usr/bin/env python3
"""Collect criterion estimates and diff two runs of them.

Criterion already prints a change-vs-previous line, but only against whatever
happens to be in the same `target/criterion` tree, which a fresh CI runner never
has. This script instead reduces a criterion output tree to one small JSON file
that CI stores as a workflow artifact, and compares the current run against the
artifact the previous successful `main` run left behind.

    bench_delta.py collect --criterion-dir target/criterion --output cur.json
    bench_delta.py compare --previous prev.json --current cur.json --threshold 15

`compare` writes Markdown to stdout and always exits 0: shared GitHub runners
have a noise floor well above any threshold worth enforcing, so a regression is
reported in the job summary rather than failing the build. See `benches/README.md`.
"""

import argparse
import json
import os
import sys

# Criterion writes a `report/` subtree of rendered HTML next to the benchmark
# directories; it holds no estimates and must not be walked as one.
REPORT_DIR = "report"


def collect(criterion_dir):
    """Map every benchmark's full id to its mean and median estimate, in ns."""
    benchmarks = {}
    for root, dirs, files in os.walk(criterion_dir):
        if REPORT_DIR in dirs:
            dirs.remove(REPORT_DIR)
        if os.path.basename(root) != "new" or "estimates.json" not in files:
            continue
        with open(os.path.join(root, "estimates.json")) as handle:
            estimates = json.load(handle)
        # `benchmark.json` carries the criterion id (group name included);
        # deriving it from the directory path would lose the original
        # separators, since group names themselves contain `/`.
        meta_path = os.path.join(root, "benchmark.json")
        if os.path.exists(meta_path):
            with open(meta_path) as handle:
                full_id = json.load(handle).get("full_id")
        else:
            full_id = os.path.relpath(os.path.dirname(root), criterion_dir)
        benchmarks[full_id] = {
            "mean_ns": estimates["mean"]["point_estimate"],
            "median_ns": estimates["median"]["point_estimate"],
        }
    return benchmarks


def format_duration(nanos):
    for unit, scale in (("ns", 1.0), ("us", 1e3), ("ms", 1e6), ("s", 1e9)):
        if nanos < scale * 1000.0:
            return f"{nanos / scale:.3g} {unit}"
    return f"{nanos / 1e9:.3g} s"


def compare(previous, current, threshold):
    lines = ["## Benchmark deltas", ""]
    if not current:
        lines.append("No criterion estimates were produced by this run.")
        return "\n".join(lines) + "\n"

    if previous is None:
        lines.append(
            "No baseline from a previous successful `main` run was available; "
            "this run's numbers are stored as the new baseline."
        )
        lines += ["", "| Benchmark | Mean | Median |", "| --- | ---: | ---: |"]
        for name in sorted(current):
            entry = current[name]
            lines.append(
                f"| `{name}` | {format_duration(entry['mean_ns'])} "
                f"| {format_duration(entry['median_ns'])} |"
            )
        return "\n".join(lines) + "\n"

    lines += [
        f"Compared against the previous successful `main` run. "
        f"A mean above +{threshold:g}% is flagged; shared runners are too noisy "
        f"for this to be a build failure.",
        "",
        "| Benchmark | Previous mean | Current mean | Delta | |",
        "| --- | ---: | ---: | ---: | :-: |",
    ]
    regressions = 0
    for name in sorted(set(previous) | set(current)):
        before = previous.get(name)
        after = current.get(name)
        if before is None:
            lines.append(
                f"| `{name}` | — | {format_duration(after['mean_ns'])} | new | 🆕 |"
            )
            continue
        if after is None:
            lines.append(
                f"| `{name}` | {format_duration(before['mean_ns'])} | — | gone | ⚠️ |"
            )
            continue
        delta = (after["mean_ns"] - before["mean_ns"]) / before["mean_ns"] * 100.0
        flag = ""
        if delta > threshold:
            flag = "🔴"
            regressions += 1
        elif delta < -threshold:
            flag = "🟢"
        lines.append(
            f"| `{name}` | {format_duration(before['mean_ns'])} "
            f"| {format_duration(after['mean_ns'])} | {delta:+.1f}% | {flag} |"
        )

    lines.append("")
    if regressions:
        lines.append(
            f"**{regressions} benchmark(s) slowed by more than {threshold:g}%.** "
            "Re-run the job to rule out runner noise before treating it as a real "
            "regression."
        )
    else:
        lines.append(f"No benchmark slowed by more than {threshold:g}%.")
    return "\n".join(lines) + "\n"


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collector = commands.add_parser("collect")
    collector.add_argument("--criterion-dir", default="target/criterion")
    collector.add_argument("--output", required=True)

    comparer = commands.add_parser("compare")
    comparer.add_argument("--previous")
    comparer.add_argument("--current", required=True)
    comparer.add_argument("--threshold", type=float, default=15.0)

    args = parser.parse_args(argv)
    if args.command == "collect":
        benchmarks = collect(args.criterion_dir)
        with open(args.output, "w") as handle:
            json.dump(benchmarks, handle, indent=2, sort_keys=True)
        print(f"collected {len(benchmarks)} benchmark estimate(s) into {args.output}")
        return 0

    sys.stdout.write(
        compare(load(args.previous), load(args.current) or {}, args.threshold)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
