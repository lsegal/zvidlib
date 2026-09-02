#!/usr/bin/env python3
"""Tests for `criterion_baseline.py`.

Standard library only, and run by the `Rust checks` job with
`python3 -m unittest discover`. The script has no third-party dependencies
precisely so a CI runner can execute it without an install step, and its tests
inherit that constraint.

`variance` is the part worth pinning: it produces the number a regression
threshold gets set from, and getting it quietly wrong would justify a threshold
nobody could tell was unjustified.

`table` is the other: it renders numbers that get committed to
`benches/README.md` and then read as a reference point for months, so its
failure mode is not a crash but a plausible-looking table that is quietly
missing a column. That is exactly what happened - the renderer matched the arm
name `sse41` while the suite emits `sse4.1`, so every x86_64 SSE4.1
measurement was dropped and the result was indistinguishable from a host that
cannot execute SSE4.1 at all.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

import criterion_baseline as cb

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def _isa_baseline(path, benchmarks, *, host="Test Host", commit="0" * 40):
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
        args = argparse.Namespace(
            baseline=list(baselines), host=host, out=str(out)
        )
        self.assertEqual(cb.table(args), 0)
        return out.read_text()

    def test_x86_arms_each_get_their_own_column(self):
        """The regression this file exists for: `sse4.1`, not `sse41`."""
        path = _isa_baseline(
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
        path = _isa_baseline(
            self.root / "arm.json",
            {"av1_deblock/scalar": 100.0, "av1_deblock/neon": 20.0},
        )
        rendered = self.render(path)
        self.assertIn("| Group | `scalar` | `neon` | Best |", rendered)
        self.assertNotIn("sse4.1", rendered)
        self.assertIn("5.00x `neon`", rendered)

    def test_unrecognised_instruction_set_is_not_dropped(self):
        """A new ISA gets a column without this script being edited first."""
        path = _isa_baseline(
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
        path = _isa_baseline(
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
        slow = _isa_baseline(
            self.root / "round1.json",
            {"av1_deblock/scalar": 100.0, "av1_deblock/avx2": 40.0},
        )
        fast = _isa_baseline(
            self.root / "round2.json",
            {"av1_deblock/scalar": 90.0, "av1_deblock/avx2": 50.0},
        )
        rendered = self.render(slow, fast)
        # min(100, 90) = 90 against min(40, 50) = 40, so 2.25x - not the 2.50x or
        # 1.80x either round would have reported on its own.
        self.assertIn("2.25x `avx2`", rendered)

    def test_absent_arm_reads_as_absent_rather_than_as_scalar(self):
        """A host that cannot run an arm gets an em dash, not a filled-in number."""
        path = _isa_baseline(
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
        path = _isa_baseline(self.root / "none.json", {"aac_decode/access_units_mono_48k": 10.0})
        args = argparse.Namespace(
            baseline=[path], host=None, out=str(self.root / "table.md")
        )
        self.assertEqual(cb.table(args), 1)

    def test_host_and_commit_are_stamped(self):
        path = _isa_baseline(self.root / "stamp.json", {"g/scalar": 10.0, "g/avx2": 5.0}, commit="abcdef0123456789" + "0" * 24)
        rendered = self.render(path, host="Some CPU (Linux, x86_64)")
        self.assertIn("Measured on **Some CPU (Linux, x86_64)**, at `abcdef012345`.", rendered)


# A stand-in for `active_by_site`'s rustdoc table, with the surrounding prose
# the parser has to walk past.
def _simd_source(sites: list[str]) -> str:
    rows = "\n".join(f"/// | `{site}` | some kernels |" for site in sites)
    return (
        "//! module docs\n"
        "/// The instruction set every dispatch site resolves to.\n"
        "///\n"
        "/// | Site | Kernels |\n"
        "/// | --- | --- |\n"
        f"{rows}\n"
        "///\n"
        "/// The `hevc_*` sites are absent on `wasm32`.\n"
        "pub fn active_by_site() -> Vec<(&'static str, SimdIsa)> {\n"
        "    vec![(\"av1_simd\", SimdIsa::Scalar)]\n"
        "}\n"
    )


def _readme(host: str, commit: str, groups: list[str]) -> str:
    rows = "\n".join(f"| `{group}` | 1.000 ms | 1.000 ms (1.00x) | 1.00x `neon` |" for group in groups)
    return (
        "## Committed baselines\n\n"
        "This table replaces the one #261 recorded at `e115506f8bf6`.\n\n"
        f"Measured on **{host}**, at `{commit}`.\n\n"
        "| Group | `scalar` | `neon` | Best |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"{rows}\n\n"
        "#### Reading the sub-parity rows\n\nProse that is not a table.\n"
    )


class DispatchSiteParsing(unittest.TestCase):
    """The site table is read out of a blob, so its parser is the whole check."""

    def test_sites_are_read_off_the_rustdoc_table(self) -> None:
        sites = cb._documented_dispatch_sites(_simd_source(["av1_simd", "hevc_recon"]))
        self.assertEqual(sites, ["av1_simd", "hevc_recon"])

    def test_prose_outside_the_table_is_not_a_site(self) -> None:
        # `hevc_*` appears in the sentence below the table and must not be read
        # as a twelfth site.
        sites = cb._documented_dispatch_sites(_simd_source(["av1_simd"]))
        self.assertEqual(sites, ["av1_simd"])

    def test_the_real_site_table_parses(self) -> None:
        source = (_REPO_ROOT / cb.SIMD_SOURCE).read_text()
        sites = cb._documented_dispatch_sites(source)
        self.assertIn("av1_simd", sites)
        self.assertIn("hevc_color_convert", sites)

    def test_every_documented_site_names_the_rows_it_invalidates(self) -> None:
        # The guard that keeps `SITE_GROUP_PREFIXES` from being the next thing
        # that goes quiet: a dispatch site added to `active_by_site` without
        # deciding which committed rows it makes stale fails here.
        source = (_REPO_ROOT / cb.SIMD_SOURCE).read_text()
        for site in cb._documented_dispatch_sites(source):
            self.assertIn(site, cb.SITE_GROUP_PREFIXES, f"{site} has no mapped groups")


class ReadmeStampParsing(unittest.TestCase):
    def test_the_stamp_is_the_bolded_host_line(self) -> None:
        # Not `e115506f8bf6`, which the prose mentions as a *superseded* draw.
        tables = cb._committed_tables(_readme("Apple M1", "b6655bad215f", ["hevc_sao"]))
        self.assertEqual([t["commit"] for t in tables], ["b6655bad215f"])

    def test_the_rows_under_a_stamp_are_its_groups(self) -> None:
        tables = cb._committed_tables(
            _readme("Apple M1", "b6655bad215f", ["hevc_sao", "hevc_color_convert"])
        )
        self.assertEqual(tables[0]["groups"], ["hevc_sao", "hevc_color_convert"])

    def test_prose_after_the_table_ends_it(self) -> None:
        readme = _readme("Apple M1", "b6655bad215f", ["hevc_sao"])
        self.assertEqual(cb._committed_tables(readme)[0]["groups"], ["hevc_sao"])

    def test_the_committed_readme_has_both_stamped_tables(self) -> None:
        tables = cb._committed_tables((_REPO_ROOT / "benches/README.md").read_text())
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(table["groups"] for table in tables))


class GroupToSiteAttribution(unittest.TestCase):
    def test_a_size_family_is_matched_by_prefix(self) -> None:
        self.assertTrue(cb._group_matches_site("av1_forward_dct_16x16", "av1_simd"))

    def test_the_1080p_variant_is_the_same_group(self) -> None:
        self.assertTrue(cb._group_matches_site("av1_encode_stage_tile_1080p", "av1_coeff_ctx"))

    def test_a_whole_frame_group_is_attributed_to_no_site(self) -> None:
        # `hevc_encode_640x352` crosses every HEVC site, so no row of it can be
        # blamed on one; the `_rdo_` and `_reconstruct` rows under it can.
        self.assertFalse(
            any(cb._group_matches_site("hevc_encode_640x352", site) for site in cb.SITE_GROUP_PREFIXES)
        )
        self.assertTrue(cb._group_matches_site("hevc_encode_640x352_reconstruct", "hevc_recon"))

    def test_a_serial_group_is_attributed_to_no_site(self) -> None:
        for group in ("hevc_cabac", "av1_encode_stage_symbol", "av1_entropy_symbol"):
            self.assertFalse(
                any(cb._group_matches_site(group, site) for site in cb.SITE_GROUP_PREFIXES),
                group,
            )


class StalenessReport(unittest.TestCase):
    """`staleness` end to end, with the stamp resolution stubbed.

    Reading `src/simd.rs` at an arbitrary commit needs either that commit in the
    clone or the network, and neither belongs in a unit test - so the resolver
    is replaced and everything around it is exercised for real.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "benches").mkdir()
        self.real_resolver = cb._sites_at_commit
        self.addCleanup(setattr, cb, "_sites_at_commit", self.real_resolver)

    def run_report(self, head: list[str], then: list[str] | None, groups: list[str]) -> str:
        (self.root / cb.SIMD_SOURCE).write_text(_simd_source(head))
        readme = self.root / "benches/README.md"
        readme.write_text(_readme("Apple M1", "b6655bad215f", groups))
        cb._sites_at_commit = lambda commit, repo_root, slug: then
        args = argparse.Namespace(readme=str(readme), repo_root=str(self.root), out=None)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(cb.staleness(args), 0)
        return buffer.getvalue()

    def test_a_site_that_landed_after_the_stamp_names_its_row(self) -> None:
        report = self.run_report(
            head=["av1_simd", "hevc_color_convert"],
            then=["av1_simd"],
            groups=["hevc_sao", "hevc_color_convert"],
        )
        self.assertIn("| `hevc_color_convert` | `hevc_color_convert` |", report)
        self.assertNotIn("| `hevc_sao` |", report)
        self.assertIn("1 row(s) flagged", report)

    def test_an_unchanged_site_set_is_clean(self) -> None:
        report = self.run_report(
            head=["av1_simd", "hevc_recon"],
            then=["av1_simd", "hevc_recon"],
            groups=["hevc_sao"],
        )
        self.assertIn("Clean", report)
        self.assertIn("0 row(s) flagged", report)

    def test_a_landed_site_no_row_measures_is_said_so(self) -> None:
        report = self.run_report(
            head=["av1_simd", "hevc_recon"], then=["av1_simd"], groups=["hevc_sao"]
        )
        self.assertIn("`hevc_recon`", report)
        self.assertIn("No row of this table is attributed", report)
        self.assertIn("0 row(s) flagged", report)

    def test_an_unreadable_stamp_is_unverified_rather_than_clean(self) -> None:
        report = self.run_report(head=["av1_simd"], then=None, groups=["hevc_sao"])
        self.assertIn("unverified rather than clean", report)
        self.assertNotIn("Clean", report)

    def test_a_retired_site_is_reported_too(self) -> None:
        report = self.run_report(
            head=["av1_simd"], then=["av1_simd", "hevc_gone"], groups=["hevc_sao"]
        )
        self.assertIn("present at the stamp and gone now", report)

    def test_a_readme_with_no_stamped_table_is_an_error(self) -> None:
        (self.root / cb.SIMD_SOURCE).write_text(_simd_source(["av1_simd"]))
        readme = self.root / "benches/README.md"
        readme.write_text("## Committed baselines\n\nNothing measured yet.\n")
        args = argparse.Namespace(readme=str(readme), repo_root=str(self.root), out=None)
        self.assertEqual(cb.staleness(args), 1)


if __name__ == "__main__":
    unittest.main()
