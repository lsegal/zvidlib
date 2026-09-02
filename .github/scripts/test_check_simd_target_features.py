#!/usr/bin/env python3
"""Unit tests for `check_simd_target_features.py`.

The check itself needs an x86_64 host and a release build to say anything, so
these cover the half that can be tested anywhere: the assembly parser, and the
decision it makes about a symbol. The fixtures are real symbols, lifted from an
`--emit=asm` build of this crate with and without `#[inline(always)]` on
`av1_simd::filters::deblock_edge_vertical` - the demonstration issue #341 asks
for, in a form that runs on every CI host rather than only the x86_64 one.
"""

from __future__ import annotations

import io
import unittest

import check_simd_target_features as checker

# `filters::deblock_edge_vertical::<vector::x86::Avx2>`, the kernel #336
# measured at 0.13-0.24x of scalar, as emitted when the wrapper did not inline
# it.
OUTLINED_KERNEL = (
    "__RINvNtNtCs7lEMBtiCmc_7zvidlib8av1_simd7filters21deblock_edge_vertical"
    "NtNtNtB4_6vector3x864Avx2EB6_"
)

# `core::core_arch::x86::avx2::_mm256_and_si256`, an intrinsic emitted as a
# function because the caller was not compiled with AVX2 enabled.
INTRINSIC = "__RNvNtNtNtCsl7QZrza34zr_4core9core_arch3x864avx216__mm256_and_si256"

# The `#[target_feature]` wrapper the dispatch macro generates. Not generic, so
# never a violation on its own, and what tells the check it is reading the right
# assembly.
WRAPPER = "__RNvNtCs7lEMBtiCmc_7zvidlib8av1_simd14deblock_v_avx2"


def asm(*lines: str) -> io.StringIO:
    return io.StringIO("\n".join(lines) + "\n")


class ReadableTest(unittest.TestCase):
    def test_renders_a_generic_kernel_with_its_vector_argument(self):
        # The vector type is the whole point of the rendering: without it a
        # reader cannot tell which instruction set's arm regressed.
        self.assertEqual(
            checker.readable(OUTLINED_KERNEL),
            "zvidlib::av1_simd::filters::deblock_edge_vertical::vector::x86::Avx2",
        )

    def test_renders_an_intrinsic(self):
        self.assertEqual(
            checker.readable(INTRINSIC),
            "core::core_arch::x86::avx2::_mm256_and_si256",
        )

    def test_leaves_an_unparsable_symbol_alone(self):
        self.assertEqual(checker.readable("_main"), "_main")


class ClassificationTest(unittest.TestCase):
    def test_a_wrapper_is_not_a_violation(self):
        self.assertFalse(checker.is_outlined_kernel(WRAPPER))
        self.assertFalse(checker.is_core_arch(WRAPPER))

    def test_a_generic_kernel_instantiation_is_outlined(self):
        self.assertTrue(checker.is_outlined_kernel(OUTLINED_KERNEL))

    def test_an_intrinsic_is_recognized(self):
        self.assertTrue(checker.is_core_arch(INTRINSIC))

    def test_the_word_alone_is_not_a_symbol(self):
        # Without the length prefix, a mention of `core_arch` in a string
        # literal or a `.ascii` directive would read as an intrinsic call.
        self.assertFalse(checker.is_core_arch("core_arch"))


class AnalyzeTest(unittest.TestCase):
    def test_a_clean_module_reports_nothing(self):
        violations, seen = checker.analyze(
            asm(
                f"{WRAPPER}:",
                "\tvpaddd\t%ymm0, %ymm1, %ymm0",
                "LBB0_1:",
                "\tjmp\tLBB0_1",
                "\tretq",
            )
        )
        self.assertEqual(violations, [])
        self.assertEqual(seen, 1)

    def test_an_outlined_kernel_is_a_violation(self):
        violations, _ = checker.analyze(asm(f"{OUTLINED_KERNEL}:", "\tretq"))
        self.assertEqual(len(violations), 1)
        self.assertIn("outlined generic kernel", violations[0])
        self.assertIn("deblock_edge_vertical", violations[0])

    def test_an_emitted_intrinsic_is_a_violation(self):
        violations, _ = checker.analyze(asm(f"{INTRINSIC}:", "\tretq"))
        self.assertEqual(len(violations), 1)
        self.assertIn("out-of-line intrinsic", violations[0])

    def test_calls_are_grouped_and_attributed_to_their_caller(self):
        violations, _ = checker.analyze(
            asm(
                f"{OUTLINED_KERNEL}:",
                f"\tcallq\t{INTRINSIC}",
                "LBB1_4:",
                f"\tcallq\t{INTRINSIC}",
                f"\tcallq\t{INTRINSIC}",
                "\tretq",
            )
        )
        # One for the definition, one for the three calls - the local label
        # between them must not be mistaken for a new enclosing function.
        self.assertEqual(len(violations), 2)
        grouped = violations[-1]
        self.assertIn("deblock_edge_vertical", grouped.splitlines()[0])
        self.assertIn("3 times", grouped)

    def test_a_tail_call_into_a_kernel_is_a_violation(self):
        violations, _ = checker.analyze(
            asm(f"{WRAPPER}:", f"\tjmp\t{OUTLINED_KERNEL}")
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("branches to", violations[0])

    def test_a_pic_call_is_still_recognized(self):
        violations, _ = checker.analyze(
            asm(f"{WRAPPER}:", f"\tcallq\t*{INTRINSIC}@GOTPCREL(%rip)")
        )
        self.assertEqual(len(violations), 1)

    def test_assembly_without_the_module_is_not_evidence(self):
        _, seen = checker.analyze(asm("_main:", "\tretq"))
        self.assertEqual(seen, 0)


class CheckTest(unittest.TestCase):
    """`check`'s exit statuses, which are what CI acts on."""

    def _run(self, body: str) -> int:
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "crate.s"
            path.write_text(body, encoding="utf-8")
            return checker.main(["check", "--asm", str(path)])

    def test_clean_assembly_passes(self):
        self.assertEqual(self._run(f"{WRAPPER}:\n\tvpaddd\t%ymm0, %ymm1, %ymm0\n"), 0)

    def test_a_violation_fails(self):
        self.assertEqual(self._run(f"{OUTLINED_KERNEL}:\n\tretq\n"), 1)

    def test_assembly_without_the_module_is_an_error_not_a_pass(self):
        self.assertEqual(self._run("_main:\n\tretq\n"), 2)

    def test_a_missing_file_is_an_error(self):
        self.assertEqual(checker.main(["check", "--asm", "/nonexistent/crate.s"]), 2)


if __name__ == "__main__":
    unittest.main()
