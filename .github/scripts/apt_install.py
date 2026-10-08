#!/usr/bin/env python3
"""Install apt packages on a CI runner, giving up on a stalled attempt and retrying.

The workflows installed their native dependencies with a bare `sudo apt-get
update && sudo apt-get install -y ...`. Now and then a mirror stalls and
`apt-get` waits on it forever: on 2026-10-07 several `Rust tests (<package>)`
jobs sat in that step for over an hour, and a job that is not canceled by hand
holds its run until the 6-hour job timeout (#647).

This runs `apt-get update` and `apt-get install` under `timeout`, so a stalled
command is killed, along with the download and dpkg processes it started, after
a few minutes. A failed or killed attempt is retried a bounded number of times
after a short pause, finishing any install it interrupted with `dpkg
--configure -a` first. The first retry also drops the first mirror from the
runner's mirror list, which is where the stalls have come from, so apt fetches
from the next one instead. apt itself also retries each failed download and gives
up on a connection that stops sending, so a single stalled request usually
recovers within the attempt. Packages are installed without their
recommendations, which for ffmpeg are mostly video drivers the tests never load
and only add to the download. Every attempt and its outcome is logged.

    apt_install.py libasound2-dev ffmpeg
    apt_install.py --attempts 3 --attempt-timeout 240 libasound2-dev

Standard library only, like the other scripts here, so a runner can execute it
without an install step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import Callable, Sequence

# apt's own per-request resilience: retry a failed download, abandon a
# connection that sends nothing for this many seconds, and wait for, rather
# than fail on, a dpkg lock another process (such as unattended-upgrades) holds.
APT_OPTIONS = (
    "-o", "Acquire::Retries=3",
    "-o", "Acquire::http::Timeout=30",
    "-o", "Acquire::https::Timeout=30",
    "-o", "DPkg::Lock::Timeout=120",
)

# The mirror list GitHub's Ubuntu runners point apt at, one mirror per line in
# order of preference, starting with Azure's own mirror.
MIRROR_LIST = "/etc/apt/apt-mirrors.txt"

# `timeout` exits with this status when it had to stop the command.
TIMED_OUT = 124

Runner = Callable[[Sequence[str]], int]


def run(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def limited(command: Sequence[str], seconds: int) -> list[str]:
    """`command` as root under `timeout`, which signals the command's whole
    process group, and kills it if it ignores the signal for 30 seconds."""
    return ["sudo", "timeout", "--kill-after=30", str(seconds), *command]


def drop_first_mirror(path: str, runner: Runner) -> None:
    """Stop apt from trying the first mirror in the runner's mirror list,
    when there is another to fall back to.

    apt goes back to the first mirror for every file, so an unresponsive one
    costs a connection timeout per package: a stalled Azure mirror held each
    attempt to install ffmpeg's hundred packages past its time limit (#647)."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    mirrors = [line for line in lines if line.strip()]
    if len(mirrors) < 2:
        return
    print(f"apt_install: no longer using mirror {mirrors[0].split()[0]}", flush=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.writelines(mirrors[1:])
    try:
        runner(["sudo", "cp", f.name, path])
    finally:
        os.unlink(f.name)


def install(
    packages: Sequence[str],
    attempts: int,
    attempt_timeout: int,
    runner: Runner | None = None,
    sleep: Callable[[float], None] | None = None,
    mirror_list: str | None = None,
) -> bool:
    """Install `packages`, trying up to `attempts` times. Returns whether an
    attempt succeeded."""
    runner = runner or run
    sleep = sleep or time.sleep
    mirror_list = mirror_list or MIRROR_LIST
    commands = {
        "update": ["apt-get", *APT_OPTIONS, "update"],
        "install": ["apt-get", *APT_OPTIONS, "install", "-y", "--no-install-recommends", *packages],
    }
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            pause = 10 * (attempt - 1)
            print(f"apt_install: retrying in {pause} s", flush=True)
            sleep(pause)
            if attempt == 2:
                drop_first_mirror(mirror_list, runner)
            # An install killed partway leaves packages unpacked but not
            # configured, which the next `apt-get install` refuses to run past.
            runner(limited(["dpkg", "--configure", "-a"], attempt_timeout))
        print(f"apt_install: attempt {attempt} of {attempts}", flush=True)
        for name, command in commands.items():
            status = runner(limited(command, attempt_timeout))
            if status != 0:
                reason = (
                    f"timed out after {attempt_timeout} s"
                    if status == TIMED_OUT
                    else f"exited with status {status}"
                )
                print(f"apt_install: `apt-get {name}` {reason}", flush=True)
                break
        else:
            print(f"apt_install: installed {' '.join(packages)}", flush=True)
            return True
    print(f"apt_install: giving up after {attempts} attempts", flush=True)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--attempt-timeout",
        type=int,
        default=240,
        help="seconds each apt-get command may run before it is killed",
    )
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args(argv)
    return 0 if install(args.packages, args.attempts, args.attempt_timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
