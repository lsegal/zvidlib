#!/usr/bin/env python3
"""Tests for `criterion_baseline.py`.

Standard library only, and run by the `Rust checks` job with
`python3 -m unittest discover`. The script has no third-party dependencies
precisely so a CI runner can execute it without an install step, and its tests
inherit that constraint.

`variance` is the part worth pinning: it produces the number a regression
threshold gets set from, and getting it quietly wrong would justify a threshold
nobody could tell was unjustified.
"""

from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

import criterion_baseline as cb


def _baseline(commit: str, medians: dict[str, float]) -> dict:
    return {
        "schema": cb.SCHEMA,
        "metric": cb.METRIC,
        "commit": commit,
        "benchmarks": {
            identifier: {"median_ns": value, "mean_ns": value, "std_dev_ns": 0.0}
            for identifier, value in medians.items()
        },
    }


class VarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = pathlib.Path(self._directory.name)

    def _write(self, name: str, medians: dict[str, float]) -> str:
        path = self.root / name
        path.write_text(json.dumps(_baseline(name, medians)))
        return str(path)

    def _run(self, paths: list[str], **overrides) -> tuple[int, str]:
        out = self.root / "variance.md"
        argv = ["variance", "--baseline", *paths, "--out", str(out)]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cb.main(argv)
        return code, (out.read_text() if out.is_file() else "")

    def test_delta_is_the_absolute_move_between_consecutive_runs(self) -> None:
        # 100 -> 110 -> 99: +10% then -10%, so both directions land on 10%.
        paths = [
            self._write("a.json", {"grp/scalar": 100.0}),
            self._write("b.json", {"grp/scalar": 110.0}),
            self._write("c.json", {"grp/scalar": 99.0}),
        ]
        code, report = self._run(paths)
        self.assertEqual(code, 0)
        self.assertIn("| `grp` | 2 | 10.0% | 10.0% | 10.0% | 15% |", report)
        self.assertIn("2 consecutive pair(s)", report)

    def test_groups_are_reported_separately(self) -> None:
        paths = [
            self._write("a.json", {"micro/scalar": 100.0, "frame/scalar": 100.0}),
            self._write("b.json", {"micro/scalar": 102.0, "frame/scalar": 140.0}),
        ]
        code, report = self._run(paths)
        self.assertEqual(code, 0)
        self.assertIn("| `micro` | 1 | 2.0% | 2.0% | 2.0% | 5% |", report)
        self.assertIn("| `frame` | 1 | 40.0% | 40.0% | 40.0% | 45% |", report)

    def test_suggested_threshold_clears_the_worst_observation(self) -> None:
        # Exactly 15% must not suggest 15%: a threshold on the observation
        # flags it. It has to step past.
        self.assertEqual(cb._suggested_threshold(15.0), 20)
        self.assertEqual(cb._suggested_threshold(14.9), 15)
        self.assertEqual(cb._suggested_threshold(0.0), 5)

    def test_percentile_is_nearest_rank(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(cb._percentile(values, 50), 2.0)
        self.assertEqual(cb._percentile(values, 95), 4.0)
        self.assertEqual(cb._percentile([], 95), 0.0)

    def test_small_samples_are_marked_provisional(self) -> None:
        paths = [
            self._write("a.json", {"grp/scalar": 100.0}),
            self._write("b.json", {"grp/scalar": 101.0}),
        ]
        _, report = self._run(paths)
        self.assertIn("Provisional", report)
        _, permissive = self._run(paths, min_pairs=1)
        self.assertNotIn("Provisional", permissive)

    def test_appearing_and_disappearing_arms_are_not_deltas(self) -> None:
        paths = [
            self._write("a.json", {"grp/scalar": 100.0, "grp/avx2": 50.0}),
            self._write("b.json", {"grp/scalar": 101.0}),
        ]
        code, report = self._run(paths)
        self.assertEqual(code, 0)
        self.assertIn("Arms that appeared or disappeared", report)
        self.assertIn("- `grp/avx2`", report)
        # One delta, not two: the vanished arm contributes nothing.
        self.assertIn("| `grp` | 1 |", report)
        # ...and its group cannot be gated on.
        self.assertNotIn("- `grp`\n", report.split("### Arms")[0])

    def test_gate_candidates_are_groups_that_stayed_under_the_gate(self) -> None:
        paths = [
            self._write("a.json", {"quiet/scalar": 100.0, "loud/scalar": 100.0}),
            self._write("b.json", {"quiet/scalar": 103.0, "loud/scalar": 130.0}),
        ]
        _, report = self._run(paths, gate_at=15)
        section = report.split("### Groups whose spread fits under")[1]
        self.assertIn("- `quiet`", section)
        self.assertNotIn("- `loud`", section)

    def test_no_group_fits_under_the_gate(self) -> None:
        paths = [
            self._write("a.json", {"loud/scalar": 100.0}),
            self._write("b.json", {"loud/scalar": 130.0}),
        ]
        _, report = self._run(paths, gate_at=15)
        self.assertIn("None. No group stayed inside 15%", report)

    def test_one_baseline_forms_no_pair(self) -> None:
        path = self._write("a.json", {"grp/scalar": 100.0})
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cb.main(["variance", "--baseline", path])
        self.assertEqual(code, 1)

    def test_disjoint_baselines_share_no_benchmark(self) -> None:
        paths = [
            self._write("a.json", {"one/scalar": 100.0}),
            self._write("b.json", {"two/scalar": 100.0}),
        ]
        code, _ = self._run(paths)
        self.assertEqual(code, 1)

    def test_a_zero_previous_median_is_not_divided_by(self) -> None:
        paths = [
            self._write("a.json", {"grp/scalar": 0.0}),
            self._write("b.json", {"grp/scalar": 100.0}),
        ]
        code, report = self._run(paths)
        self.assertEqual(code, 1)
        self.assertEqual(report, "")


class GroupOfTest(unittest.TestCase):
    def test_the_group_is_everything_before_the_last_separator(self) -> None:
        self.assertEqual(cb._group_of("av1_deblock/scalar"), "av1_deblock")
        self.assertEqual(cb._group_of("smoke/simd=off/decode"), "smoke/simd=off")
        self.assertEqual(cb._group_of("bare"), "bare")


if __name__ == "__main__":
    unittest.main()
