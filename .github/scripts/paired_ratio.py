#!/usr/bin/env python3
"""Reads a paired branch-against-base ratio for #382's reconstruction groups.

Temporary, and paired with `.github/workflows/measure-382.yml`: both are the
apparatus for #382 and are removed once `benches/README.md` records the answer.

The workflow builds both trees on one host and runs them alternately within
each round, so the two arms of every ratio share a host and a round. This
takes the per-benchmark minimum across rounds and prints base/branch, which
reads above 1.00x when the branch is faster. The group's own `scalar` arm is
the control: it resolves to the same scalar reference in both trees, so it has
to read 1.00x, and a ratio is only worth reading as far as that arm is.

    python3 paired_ratio.py <rounds-dir> [host-label]
"""

from __future__ import annotations

import json
import pathlib
import sys


def best(rounds_dir: pathlib.Path, tree: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(rounds_dir.glob(f"{tree}-round-*.json")):
        for identifier, entry in json.loads(path.read_text()).get("benchmarks", {}).items():
            ns = entry["median_ns"]
            if identifier not in out or ns < out[identifier]:
                out[identifier] = ns
    return out


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    rounds_dir = pathlib.Path(argv[1])
    host = argv[2] if len(argv) == 3 else "unknown host"

    base, branch = best(rounds_dir, "base"), best(rounds_dir, "branch")
    if not base or not branch:
        print(f"error: missing base or branch rounds under {rounds_dir}", file=sys.stderr)
        return 1

    print(f"## Paired branch-against-base -- {host}")
    print()
    print("base / branch; above 1.00x means the branch is faster.")
    print()
    print("| benchmark | base | branch | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for identifier in sorted(set(base) & set(branch)):
        b, n = base[identifier], branch[identifier]
        note = "  <- control" if identifier.endswith("/scalar") else ""
        print(
            f"| `{identifier}`{note} | {b / 1e6:.3f} ms | {n / 1e6:.3f} ms "
            f"| {b / n:.3f}x |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
