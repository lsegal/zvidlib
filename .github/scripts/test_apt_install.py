#!/usr/bin/env python3
"""Tests for `apt_install.py`.

Standard library only, and run by the `Rust checks` job with
`python3 -m unittest discover`, like the other script tests here.

What is pinned is what a CI run only shows once a mirror stalls: that every
apt-get command runs under a time limit, that a timed-out or failed attempt is
retried after recovering an interrupted install, and that the retries are
bounded so a broken install still fails the job.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import apt_install
from apt_install import TIMED_OUT, install


class FakeApt:
    """Records each command and answers it from a scripted list of statuses,
    succeeding once the list runs out."""

    def __init__(self, statuses: list[int] | None = None, mirror_list: str = "/nonexistent/apt-mirrors.txt"):
        self.mirror_list = mirror_list
        self.statuses = list(statuses or [])
        self.commands: list[list[str]] = []
        self.slept: list[float] = []

    def run(self, command):
        self.commands.append(list(command))
        if command[:2] == ["sudo", "cp"]:
            # Stand in for root writing the mirror list.
            with open(command[2]) as source, open(command[3], "w") as target:
                target.write(source.read())
            return 0
        return self.statuses.pop(0) if self.statuses else 0

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def install(self, attempts: int = 3) -> tuple[bool, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            ok = install(
                ["libasound2-dev", "ffmpeg"], attempts, 240, self.run, self.sleep, self.mirror_list
            )
        return ok, out.getvalue()

    def steps(self) -> list[str]:
        """Each command reduced to its apt-get or dpkg action."""
        steps = {"update": "update", "-a": "-a"}
        return [
            "cp" if c[1] == "cp" else steps.get(c[-1], "install") for c in self.commands
        ]


class InstallTest(unittest.TestCase):
    def test_success_updates_then_installs_once(self):
        apt = FakeApt()
        ok, _ = apt.install()
        self.assertTrue(ok)
        self.assertEqual(apt.steps(), ["update", "install"])
        self.assertEqual(apt.slept, [])
        self.assertEqual(
            apt.commands[1][-4:], ["-y", "--no-install-recommends", "libasound2-dev", "ffmpeg"]
        )

    def test_every_command_runs_as_root_under_a_time_limit(self):
        apt = FakeApt([TIMED_OUT])
        apt.install()
        for command in apt.commands:
            self.assertEqual(command[:4], ["sudo", "timeout", "--kill-after=30", "240"])

    def test_apt_retries_and_gives_up_on_silent_connections(self):
        apt = FakeApt()
        apt.install()
        for command in apt.commands:
            self.assertIn("Acquire::Retries=3", command)
            self.assertIn("Acquire::http::Timeout=30", command)
            self.assertIn("DPkg::Lock::Timeout=120", command)

    def test_stalled_install_is_recovered_and_retried(self):
        apt = FakeApt([0, TIMED_OUT])
        ok, log = apt.install()
        self.assertTrue(ok)
        self.assertEqual(
            apt.steps(), ["update", "install", "-a", "update", "install"]
        )
        self.assertEqual(apt.commands[2][4:], ["dpkg", "--configure", "-a"])
        self.assertEqual(apt.slept, [10])
        self.assertIn("`apt-get install` timed out after 240 s", log)

    def test_failed_update_skips_the_install_and_retries(self):
        apt = FakeApt([100])
        ok, log = apt.install()
        self.assertTrue(ok)
        self.assertEqual(apt.steps(), ["update", "-a", "update", "install"])
        self.assertIn("`apt-get update` exited with status 100", log)

    def test_gives_up_after_the_last_attempt(self):
        apt = FakeApt([TIMED_OUT, 0, TIMED_OUT, 0, TIMED_OUT])
        ok, log = apt.install(attempts=3)
        self.assertFalse(ok)
        self.assertEqual(
            apt.steps(), ["update", "-a", "update", "-a", "update"]
        )
        self.assertEqual(apt.slept, [10, 20])
        self.assertIn("giving up after 3 attempts", log)

    def test_first_retry_drops_the_first_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "apt-mirrors.txt")
            with open(path, "w") as f:
                f.write(
                    "http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n"
                    "https://archive.ubuntu.com/ubuntu/\tpriority:2\n"
                    "https://security.ubuntu.com/ubuntu/\tpriority:3\n"
                )
            apt = FakeApt([0, TIMED_OUT, 0, 0, TIMED_OUT], mirror_list=path)
            ok, log = apt.install()
            with open(path) as f:
                mirrors = f.read()
        self.assertTrue(ok)
        self.assertEqual(
            apt.steps(),
            ["update", "install", "cp", "-a", "update", "install", "-a", "update", "install"],
        )
        self.assertEqual(
            mirrors,
            "https://archive.ubuntu.com/ubuntu/\tpriority:2\n"
            "https://security.ubuntu.com/ubuntu/\tpriority:3\n",
        )
        self.assertIn("no longer using mirror http://azure.archive.ubuntu.com/ubuntu/", log)

    def test_a_single_mirror_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "apt-mirrors.txt")
            with open(path, "w") as f:
                f.write("https://archive.ubuntu.com/ubuntu/\tpriority:1\n")
            apt = FakeApt([TIMED_OUT], mirror_list=path)
            ok, _ = apt.install()
        self.assertTrue(ok)
        self.assertNotIn("cp", apt.steps())

    def test_main_exit_status_reflects_the_install(self):
        apt = FakeApt([1, 0, 1])
        original = apt_install.run, apt_install.time.sleep, apt_install.MIRROR_LIST
        apt_install.run, apt_install.time.sleep = apt.run, apt.sleep
        # Keep the runner's own mirror list out of reach, as on CI it exists.
        apt_install.MIRROR_LIST = apt.mirror_list
        try:
            with redirect_stdout(io.StringIO()):
                failed = apt_install.main(["--attempts", "2", "ffmpeg"])
                passed = apt_install.main(["ffmpeg"])
        finally:
            apt_install.run, apt_install.time.sleep, apt_install.MIRROR_LIST = original
        self.assertEqual((failed, passed), (1, 0))


if __name__ == "__main__":
    unittest.main()
