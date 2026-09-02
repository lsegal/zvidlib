#!/usr/bin/env python3
"""Reads the band-offset search's share of the HEVC reconstruction group.

Temporary, and paired with `.github/workflows/measure-382.yml`: both are the
apparatus for #382 and are removed once `benches/README.md` records the answer.

`measure-382.yml` runs the four `hevc_encode_640x352_reconstruct*` groups in
one criterion process per round, so the searched arm and its
`_no_band_search` counterpart are always timed within the same round on the
same host. This takes the per-benchmark minimum across those rounds -- the
statistic `benches/README.md` documents for a host whose round-to-round spread
is larger than the effect being read -- and prints the share as a markdown
table.

    python3 band_search_share.py <rounds-dir> [host-label]
"""

from __future__ import annotations

import json
import pathlib
import sys

SUFFIX = "_no_band_search"


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    rounds_dir = pathlib.Path(argv[1])
    host = argv[2] if len(argv) == 3 else "unknown host"

    rounds: list[dict[str, float]] = []
    for path in sorted(rounds_dir.glob("round-*.json")):
        benchmarks = json.loads(path.read_text()).get("benchmarks", {})
        rounds.append({k: v["median_ns"] for k, v in benchmarks.items()})
    if not rounds:
        print(f"error: no round-*.json under {rounds_dir}", file=sys.stderr)
        return 1

    best: dict[str, float] = {}
    for round_times in rounds:
        for identifier, ns in round_times.items():
            if identifier not in best or ns < best[identifier]:
                best[identifier] = ns

    print(f"## Band-search share -- {host}")
    print()
    print(f"{len(rounds)} interleaved rounds, per-benchmark minimum.")
    print()
    print("| group | searched | edge only | band search | share |")
    print("| --- | ---: | ---: | ---: | ---: |")

    rows = 0
    for identifier in sorted(best):
        if SUFFIX in identifier:
            continue
        # `<group>/<isa>` -- the stub arm carries the suffix on the group half.
        group, _, isa = identifier.rpartition("/")
        stub = f"{group}{SUFFIX}/{isa}" if isa else f"{identifier}{SUFFIX}"
        if stub not in best:
            continue
        searched, edge_only = best[identifier], best[stub]
        delta = searched - edge_only
        share = delta / searched if searched else 0.0
        print(
            f"| `{identifier}` | {searched / 1e6:.3f} ms | {edge_only / 1e6:.3f} ms "
            f"| {delta / 1e6:.3f} ms | {share * 100:.2f}% |"
        )
        rows += 1
    if rows == 0:
        print("| *no paired arms found* | | | | |")

    # The spread is what says whether a share this size is readable at all on
    # this host: a 2% share under a 5% round-to-round spread is not a number.
    print()
    print("| benchmark | min | max | spread |")
    print("| --- | ---: | ---: | ---: |")
    for identifier in sorted(best):
        times = [r[identifier] for r in rounds if identifier in r]
        lo, hi = min(times), max(times)
        print(
            f"| `{identifier}` | {lo / 1e6:.3f} ms | {hi / 1e6:.3f} ms "
            f"| {(hi / lo - 1) * 100:.2f}% |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
