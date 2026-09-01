#!/usr/bin/env python3
"""Unit tests for `criterion_baseline.py`.

The `table` subcommand renders numbers that get committed to
`benches/README.md` and then read as a reference point for months, so its
failure mode is not a crash but a plausible-looking table that is quietly
missing a column. That is exactly what happened: the renderer matched the arm
name `sse41` while the suite emits `sse4.1`, so every x86_64 SSE4.1
measurement was dropped and the result was indistinguishable from a host that
cannot execute SSE4.1 at all.

Run with `python3 -m unittest discover -s .github/scripts -p 'test_*.py'`.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "criterion_baseline", pathlib.Path(__file__).with_name("criterion_baseline.py")
)
criterion_baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(criterion_baseline)


def _baseline(path, benchmarks, *, host="Test Host", commit="0" * 40):
    payload = {
        "schema": 1,
        "host": host,
        "commit": commit,
        "metric": "median",
        "benchmarks": {
            identifier: {"median_ns": median} for identifier, median in benchmarks.items()
        },
    }
    pathlib.Path(path).write_text(json.dumps(payload))
    return str(path)


class RenderTable(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = pathlib.Path(self._dir.name)

    def render(self, *baselines, host=None):
        out = self.root / "table.md"
        args = criterion_baseline.argparse.Namespace(
            baseline=list(baselines), host=host, out=str(out)
        )
        self.assertEqual(criterion_baseline.table(args), 0)
        return out.read_text()

    def test_x86_arms_each_get_their_own_column(self):
        """The regression this file exists for: `sse4.1`, not `sse41`."""
        path = _baseline(
            self.root / "x86.json",
            {
                "av1_deblock/scalar": 100.0,
                "av1_deblock/sse4.1": 50.0,
                "av1_deblock/avx2": 25.0,
            },
        )
        rendered = self.render(path)
        self.assertIn("| Group | `scalar` | `sse4.1` | `avx2` | Best |", rendered)
        self.assertIn("2.00x", rendered)
        self.assertIn("4.00x `avx2`", rendered)

    def test_aarch64_arms_still_render(self):
        """The committed Apple M1 table must survive the same code path."""
        path = _baseline(
            self.root / "arm.json",
            {"av1_deblock/scalar": 100.0, "av1_deblock/neon": 20.0},
        )
        rendered = self.render(path)
        self.assertIn("| Group | `scalar` | `neon` | Best |", rendered)
        self.assertNotIn("sse4.1", rendered)
        self.assertIn("5.00x `neon`", rendered)

    def test_unrecognised_instruction_set_is_not_dropped(self):
        """A new ISA gets a column without this script being edited first."""
        path = _baseline(
            self.root / "future.json",
            {
                "av1_deblock/scalar": 100.0,
                "av1_deblock/avx2": 50.0,
                "av1_deblock/avx512": 25.0,
            },
        )
        rendered = self.render(path)
        self.assertIn("| Group | `scalar` | `avx2` | `avx512` | Best |", rendered)

    def test_single_arm_groups_are_excluded(self):
        """Only `bench_across_isas` groups belong in the table."""
        path = _baseline(
            self.root / "mixed.json",
            {
                "av1_deblock/scalar": 100.0,
                "av1_deblock/avx2": 50.0,
                "aac_decode/access_units_mono_48k": 10.0,
                "mp4_mux/media_output_1s_30fps": 10.0,
            },
        )
        rendered = self.render(path)
        self.assertIn("`av1_deblock`", rendered)
        self.assertNotIn("aac_decode", rendered)
        self.assertNotIn("mp4_mux", rendered)

    def test_rounds_are_merged_by_elementwise_minimum(self):
        """Contention only ever adds time, so the fastest round wins per arm."""
        slow = _baseline(
            self.root / "round1.json",
            {"av1_deblock/scalar": 100.0, "av1_deblock/avx2": 40.0},
        )
        fast = _baseline(
            self.root / "round2.json",
            {"av1_deblock/scalar": 90.0, "av1_deblock/avx2": 50.0},
        )
        rendered = self.render(slow, fast)
        # min(100, 90) = 90 against min(40, 50) = 40, so 2.25x - not the 2.50x or
        # 1.80x either round would have reported on its own.
        self.assertIn("2.25x `avx2`", rendered)

    def test_absent_arm_reads_as_absent_rather_than_as_scalar(self):
        """A host that cannot run an arm gets an em dash, not a filled-in number."""
        path = _baseline(
            self.root / "partial.json",
            {
                "av1_deblock/scalar": 100.0,
                "av1_deblock/avx2": 50.0,
                "av1_cdef/scalar": 100.0,
            },
        )
        rendered = self.render(path)
        cdef = next(line for line in rendered.splitlines() if line.startswith("| `av1_cdef`"))
        self.assertIn("—", cdef)

    def test_no_per_isa_groups_is_an_error(self):
        path = _baseline(self.root / "none.json", {"aac_decode/access_units_mono_48k": 10.0})
        args = criterion_baseline.argparse.Namespace(
            baseline=[path], host=None, out=str(self.root / "table.md")
        )
        self.assertEqual(criterion_baseline.table(args), 1)

    def test_host_and_commit_are_stamped(self):
        path = _baseline(self.root / "stamp.json", {"g/scalar": 10.0, "g/avx2": 5.0}, commit="abcdef0123456789" + "0" * 24)
        rendered = self.render(path, host="Some CPU (Linux, x86_64)")
        self.assertIn("Measured on **Some CPU (Linux, x86_64)**, at `abcdef012345`.", rendered)


if __name__ == "__main__":
    unittest.main()
