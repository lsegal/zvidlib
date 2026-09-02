#!/usr/bin/env python3
"""Fail when a SIMD kernel is compiled without the instruction set its
`#[target_feature]` wrapper declares.

`#[target_feature]` applies to the function it is written on, never to what
that function calls. The `av1_simd` dispatch macro therefore only widens the
generic kernel it wraps for as long as LLVM keeps inlining that kernel into the
wrapper; the moment it declines, the kernel body is codegen'd at the target's
*baseline* instruction set and every intrinsic in it becomes an out-of-line
call through `core::core_arch`. Issue #336 measured what that costs -
`deblock_edge_vertical::<Avx2>` compiled to 961 calls and not one AVX2
instruction, and those arms ran at 0.13-0.24x of the scalar reference they
replace. Nothing about the output changes, so every bit-exactness test still
passes and only a benchmark on an x86_64 host notices.

The defect has a crisp signature in the emitted object code, which is what this
checks (issue #341):

  * a branch into `core::core_arch`, or a `core::core_arch` symbol defined at
    all - an intrinsic that was not inlined, anywhere in the crate; and
  * a symbol for an `av1_simd` generic kernel monomorphized over one of the
    `av1_simd::vector` types - the kernel the wrapper was supposed to absorb,
    left standing on its own.

The first rule is crate-wide and so covers every `#[target_feature]` site the
crate has, `hevc::engine::simd`, `hevc::engine::transform_simd`,
`hevc::color_convert` and `av1_mc` included: an out-of-line intrinsic call is
the same defect wherever it appears. The second is `av1_simd`-specific because
only that module dispatches through generic kernels; the other sites write
their intrinsics directly inside the `#[target_feature]` function, where there
is no separate body for the inliner to leave behind.

    build --target-dir target/simd-feature-check
    check --asm path/to/crate.s

`build` emits the assembly with `--emit=asm -C codegen-units=1` and then runs
`check` over it. It forces `-C symbol-mangling-version=v0` so the generic
argument is *in* the symbol - the legacy mangling erases it to a hash, and
`inverse_transform4::<Sse4>` and `inverse_transform4::<Avx2>` become
indistinguishable.

This is meaningful on x86_64 only. On aarch64 NEON is in the baseline, so a
kernel that lost its `#[target_feature]` still compiles to NEON and the defect
is invisible; `build` says so and skips rather than passing quietly.
"""

from __future__ import annotations

import argparse
import pathlib
import platform
import re
import subprocess
import sys

# How many findings to print. One un-inlined kernel instantiates every intrinsic
# it uses, so the untruncated list runs to hundreds of lines for a single
# mistake and the first few already name the kernel.
REPORT_LIMIT = 20

# A global symbol definition. Local labels (`LBB0_3:`, `Ltmp4:`) are skipped:
# the enclosing *function* is what a report has to name, and on Mach-O every
# local label starts with `L`.
SYMBOL_DEF = re.compile(r"^([A-Za-z_$][A-Za-z0-9_$.]*):\s*$")

# A branch with a symbolic target. `bl`/`b` are the aarch64 spellings; they cost
# nothing to accept and keep the parser honest if this is ever pointed at an
# aarch64 build by hand.
BRANCH = re.compile(r"^\s+(call|callq|jmp|jmpq|bl|b)\s+([^\s;#]+)")

# v0 mangling encodes a path as length-prefixed identifiers, so `core_arch`
# appears as `9core_arch` and `vector::x86` as `6vector3x86`. Matching the
# length prefix as well is what keeps `core_arch` in a string literal or a
# comment from reading as a symbol.
CORE_ARCH = re.compile(r"9core_arch")
AV1_SIMD = re.compile(r"8av1_simd")
VECTOR_TYPE = re.compile(r"6vector3(x86|arm)")

# A crate disambiguator (`Cs7lEMBtiCmc_`) and a legacy mangling hash
# (`17h1f0e2d3c4b5a6978`) are length-prefixed identifier-shaped runs that mean
# nothing to a reader, so they are dropped rather than rendered.
DISAMBIGUATOR = re.compile(r"Cs[0-9A-Za-z]+_")
LEGACY_HASH = re.compile(r"^h[0-9a-f]{16}$")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def readable(symbol: str) -> str:
    """Render `symbol`'s path components as `a::b::c`, or return it unchanged.

    A real demangler would resolve v0's backreferences, its generic-argument
    brackets and its lifetimes; a failure only needs to say which kernel it is
    talking about, which the length-prefixed identifiers alone do. Both
    manglings encode those the same way, so this reads either.
    """
    trimmed = DISAMBIGUATOR.sub("", symbol)
    parts = []
    index = 0
    while index < len(trimmed):
        if not trimmed[index].isdigit() or trimmed[index] == "0":
            index += 1
            continue
        # `B4_` is a backreference to a path already spelled out earlier in the
        # symbol, not a length-prefixed identifier; reading its base-62 index as
        # a length produces a fragment of the following component.
        if index and trimmed[index - 1] == "B":
            index += 1
            continue
        end = index
        while end < len(trimmed) and trimmed[end].isdigit():
            end += 1
        length = int(trimmed[index:end])
        # v0 separates the length from an identifier that would otherwise start
        # with a digit or run into it with an underscore, and that underscore is
        # part of neither. `17__mm256_slli_epi32` is 17 bytes of
        # `_mm256_slli_epi32`.
        if trimmed[end : end + 1] == "_":
            end += 1
        name = trimmed[end : end + length]
        if len(name) != length or not IDENTIFIER.match(name):
            index += 1
            continue
        if not LEGACY_HASH.match(name):
            parts.append(name)
        index = end + length
    if not parts:
        return symbol
    return "::".join(parts)


def _clean(target: str) -> str:
    """Strip a branch operand down to the symbol it names."""
    target = target.lstrip("*")
    for suffix in ("@PLT", "@plt", "@GOTPCREL", "(%rip)"):
        target = target.replace(suffix, "")
    return target


def is_core_arch(symbol: str) -> bool:
    return bool(CORE_ARCH.search(symbol))


def is_outlined_kernel(symbol: str) -> bool:
    """True for an `av1_simd` item monomorphized over an `av1_simd::vector` type.

    That combination only occurs for a generic kernel instantiation: the
    `#[target_feature]` wrappers are not generic, and the vector types' own
    inherent items would not mention a second `av1_simd` path component.
    """
    return bool(AV1_SIMD.search(symbol) and VECTOR_TYPE.search(symbol))


def analyze(lines) -> tuple[list[str], int]:
    """Return (violations, number of `av1_simd` symbols seen).

    The second value is a sanity check on the input rather than a result: an
    assembly file with no `av1_simd` symbols in it is not evidence that the
    kernels are clean, it is evidence that the wrong file was read.
    """
    violations = []
    enclosing = "<file scope>"
    av1_symbols = 0
    # Counted per (enclosing function, callee) rather than reported per line:
    # one un-inlined kernel produces hundreds of identical intrinsic calls.
    intrinsic_calls: dict[tuple[str, str], int] = {}

    for line in lines:
        definition = SYMBOL_DEF.match(line)
        if definition:
            symbol = definition.group(1)
            if symbol.startswith("L"):
                continue
            enclosing = symbol
            if AV1_SIMD.search(symbol):
                av1_symbols += 1
            if is_outlined_kernel(symbol):
                violations.append(
                    f"outlined generic kernel: {readable(symbol)}\n"
                    f"    a kernel the `#[target_feature]` wrapper did not absorb, so its "
                    f"body was compiled at the baseline instruction set"
                )
            elif is_core_arch(symbol):
                violations.append(
                    f"out-of-line intrinsic: {readable(symbol)}\n"
                    f"    an intrinsic emitted as a real function instead of an instruction"
                )
            continue

        branch = BRANCH.match(line)
        if branch:
            target = _clean(branch.group(2))
            if is_core_arch(target) or is_outlined_kernel(target):
                key = (enclosing, target)
                intrinsic_calls[key] = intrinsic_calls.get(key, 0) + 1

    for (caller, callee), count in sorted(intrinsic_calls.items()):
        plural = "" if count == 1 else "s"
        violations.append(
            f"{readable(caller)}\n"
            f"    branches to {readable(callee)} {count} time{plural}"
        )

    return violations, av1_symbols


def check(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(p) for p in args.asm]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(f"no such assembly file: {', '.join(missing)}", file=sys.stderr)
        return 2

    violations: list[str] = []
    av1_symbols = 0
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            found, seen = analyze(handle)
        violations.extend(f"{path}: {v}" for v in found)
        av1_symbols += seen

    if av1_symbols == 0:
        print(
            "found no `av1_simd` symbols in "
            f"{', '.join(str(p) for p in paths)}: this is not the crate's assembly, "
            "and a clean result from it would mean nothing",
            file=sys.stderr,
        )
        return 2

    if violations:
        print(
            f"{len(violations)} sign(s) of a SIMD kernel compiled without the "
            "instruction set its `#[target_feature]` wrapper declares:\n",
            file=sys.stderr,
        )
        # One un-inlined kernel instantiates every intrinsic it uses, so the
        # list is hundreds of lines long for a single mistake. The first
        # `REPORT_LIMIT` name the kernel; the rest only repeat it.
        for violation in violations[:REPORT_LIMIT]:
            print(f"  {violation}", file=sys.stderr)
        if len(violations) > REPORT_LIMIT:
            print(
                f"  ... and {len(violations) - REPORT_LIMIT} more",
                file=sys.stderr,
            )
        print(
            "\nEvery kernel reached from `simd_entry_points!` must be "
            "`#[inline(always)]` so it is absorbed into the wrapper that enables "
            "its instruction set; see issue #336.",
            file=sys.stderr,
        )
        return 1

    print(
        f"no out-of-line intrinsics and no outlined generic kernels "
        f"({av1_symbols} `av1_simd` symbols inspected)"
    )
    return 0


def build(args: argparse.Namespace) -> int:
    # A `--target` is what makes this runnable from an aarch64 workstation:
    # cross-compiling the assembly is enough, nothing has to execute. Without
    # one the host decides, and an aarch64 host has nothing to say.
    architecture = (args.target or platform.machine()).lower()
    if not architecture.startswith(("x86_64", "amd64")):
        print(
            f"host is {platform.machine()}; skipping. The check is x86_64-only: "
            "NEON is in the aarch64 baseline, so a kernel that lost its "
            "`#[target_feature]` is still compiled to NEON there and the defect "
            "leaves no trace. Pass `--target x86_64-...` to cross-compile it here."
        )
        return 0

    target_dir = pathlib.Path(args.target_dir)
    command = [
        "cargo",
        "rustc",
        "--lib",
        "--release",
        "--target-dir",
        str(target_dir),
    ]
    if args.target:
        command += ["--target", args.target]
    for feature in args.features:
        command += ["--features", feature]
    command += [
        "--",
        "--emit=asm,link",
        "-C",
        "codegen-units=1",
        "-C",
        "symbol-mangling-version=v0",
    ]
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print("the assembly build failed", file=sys.stderr)
        return completed.returncode

    # Cargo nests the artifacts under the triple as soon as `--target` is given.
    prefix = f"{args.target}/" if args.target else ""
    produced = sorted(target_dir.glob(f"{prefix}release/deps/zvidlib*.s"))
    if not produced:
        print(
            f"the build produced no assembly under "
            f"{target_dir}/{prefix}release/deps",
            file=sys.stderr,
        )
        return 2

    args.asm = [str(p) for p in produced]
    return check(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    checker = sub.add_parser("check", help="inspect assembly files already emitted")
    checker.add_argument("--asm", nargs="+", required=True)
    checker.set_defaults(func=check)

    builder = sub.add_parser("build", help="emit the crate's assembly, then check it")
    builder.add_argument("--target-dir", default="target/simd-feature-check")
    builder.add_argument(
        "--target",
        help="cross-compile for this triple instead of building for the host",
    )
    builder.add_argument("--features", nargs="*", default=["native"])
    builder.set_defaults(func=build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
