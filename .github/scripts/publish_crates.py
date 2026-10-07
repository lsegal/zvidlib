#!/usr/bin/env python3
"""Publish the workspace crates to crates.io one at a time, within its rate limits.

The release workflow used to upload every crate in one `cargo publish
--workspace` call. crates.io limits how quickly an account may publish crates,
and a first release of a workspace whose crates are mostly new names runs
straight into the limit on new crates: the upload answers `429 Too Many
Requests` partway through and the release job fails (#637).

This publishes each crate with its own `cargo publish --package`, in dependency
order, and paces the uploads with a local copy of the token bucket crates.io
keeps for the account (https://crates.io/docs/rate-limits). A first release
therefore waits out the limit instead of hitting it. The local bucket cannot
know what an earlier run already spent, so a 429 is still expected after a
failed attempt; it is answered by waiting until the time crates.io names and
retrying the same crate, and the bucket is emptied so later uploads are paced
from there.

It runs in the Publish crates workflow, which Finalize release starts without
waiting for it, so a slow publish never holds back the GitHub release (#649).

A crate already on crates.io at the release version is skipped, so a re-run
continues where the last one stopped. Every upload, skip, and wait is logged,
with the time a wait ends, so a slow run cannot be mistaken for a hung one.

    publish --version X.Y.Z
    publish --version X.Y.Z --dry-run

Standard library only, like the other scripts here, so a runner can execute it
without an install step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

# crates.io's default limits, from `LimitedAction` in its `rate_limiter.rs`.
NEW_CRATE_RATE = dt.timedelta(minutes=10)
NEW_CRATE_BURST = 5
NEW_VERSION_RATE = dt.timedelta(minutes=1)
NEW_VERSION_BURST = 30

# Added to every wait crates.io names, so the retry lands after its clock
# rather than on it.
RETRY_MARGIN = dt.timedelta(seconds=5)
# A 429 for one crate this many times in a row is not a rate limit being waited
# out, and is reported instead of retried forever.
MAX_RATE_LIMITED_ATTEMPTS = 6

RETRY_AFTER = re.compile(r"try again after (.+? GMT)")


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def clock(moment: dt.datetime) -> str:
    return moment.strftime("%H:%M:%S UTC")


@dataclass
class Bucket:
    """A local copy of one of crates.io's per-account token buckets.

    crates.io starts a bucket with `burst` tokens and adds one every `rate`;
    each upload takes one. Waiting for a token here is what keeps an upload
    from being refused.
    """

    rate: dt.timedelta
    burst: int
    tokens: int
    refilled: dt.datetime

    @classmethod
    def full(cls, rate: dt.timedelta, burst: int, at: dt.datetime) -> Bucket:
        return cls(rate, burst, burst, at)

    def _refill(self, at: dt.datetime) -> None:
        if self.tokens >= self.burst:
            self.refilled = max(self.refilled, at)
            return
        earned = int((at - self.refilled) / self.rate) if at > self.refilled else 0
        if earned:
            self.tokens = min(self.burst, self.tokens + earned)
            self.refilled += earned * self.rate

    def ready_at(self, at: dt.datetime) -> dt.datetime:
        """When the next upload may be made, which is `at` if a token is left."""
        self._refill(at)
        return at if self.tokens > 0 else self.refilled + self.rate

    def take(self, at: dt.datetime) -> None:
        self._refill(at)
        self.tokens = max(0, self.tokens - 1)

    def empty_until(self, until: dt.datetime) -> None:
        """Record that crates.io has no token left before `until`."""
        self.tokens = 0
        self.refilled = until - self.rate


def publish_order(packages: list[dict]) -> list[str]:
    """The names of `packages`, each after the workspace crates it depends on.

    Development dependencies are left out: the workspace's point only at a
    path, so `cargo publish` drops them, and they run in cycles. Ties keep
    `cargo metadata` order, so the log reads the same from run to run.
    """
    names = [package["name"] for package in packages]
    needs = {
        package["name"]: {
            dependency["name"]
            for dependency in package["dependencies"]
            if dependency.get("kind") != "dev" and dependency["name"] in names
        }
        for package in packages
    }
    order: list[str] = []
    while len(order) < len(names):
        ready = [
            name for name in names if name not in order and needs[name] <= set(order)
        ]
        if not ready:
            stuck = sorted(set(names) - set(order))
            raise SystemExit(f"these crates depend on each other in a cycle: {stuck}")
        order.append(ready[0])
    return order


def retry_after(output: str) -> dt.datetime | None:
    """The time a crates.io 429 says to try again after, if it names one."""
    match = RETRY_AFTER.search(output)
    if not match:
        return None
    try:
        return email.utils.parsedate_to_datetime(match.group(1))
    except (TypeError, ValueError):
        return None


def rate_limited(output: str) -> bool:
    return "429" in output and "Too Many Requests" in output


def already_uploaded(output: str) -> bool:
    """Whether `cargo publish` failed only because this version already exists.

    The sparse index lags an upload by a little, so a re-run can find a crate
    missing from it that the last run did publish.
    """
    return "already exists" in output or "is already uploaded" in output


def index_path(name: str) -> str:
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def published_versions(name: str) -> set[str] | None:
    """The versions of `name` on crates.io, or `None` if it is a new crate."""
    url = f"https://index.crates.io/{index_path(name)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            lines = response.read().decode().splitlines()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise SystemExit(f"::error::the crates.io index answered {error.code} for {name}")
    return {json.loads(line)["vers"] for line in lines if line.strip()}


def run_streaming(command: list[str]) -> tuple[int, str]:
    """Run `command`, echoing its output as it arrives, and return both."""
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return process.wait(), "".join(lines)


@dataclass
class Publisher:
    version: str
    lookup: Callable[[str], set[str] | None] = published_versions
    run: Callable[[list[str]], tuple[int, str]] = run_streaming
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], dt.datetime] = now
    dry_run: bool = False

    def __post_init__(self) -> None:
        started = self.clock()
        self.new_crates = Bucket.full(NEW_CRATE_RATE, NEW_CRATE_BURST, started)
        self.new_versions = Bucket.full(NEW_VERSION_RATE, NEW_VERSION_BURST, started)

    def wait_until(self, until: dt.datetime, name: str, why: str) -> None:
        seconds = (until - self.clock()).total_seconds()
        if seconds <= 0:
            return
        minutes, rest = divmod(round(seconds), 60)
        print(
            f"waiting {minutes}m{rest:02d}s, until {clock(until)}, "
            f"before publishing {name}: {why}",
            flush=True,
        )
        self.sleep(seconds)

    def publish(self, name: str) -> str:
        """Publish one crate, returning what was done with it."""
        versions = self.lookup(name)
        if versions is not None and self.version in versions:
            print(f"{name} {self.version} is already on crates.io; not publishing it again")
            return "skipped"
        new = versions is None
        bucket = self.new_crates if new else self.new_versions
        kind = "a new crate" if new else "a new version"
        if self.dry_run:
            print(f"would publish {name} {self.version} as {kind}")
            return "planned"
        limit = (
            f"crates.io allows {NEW_CRATE_BURST} new crates at once, then one every "
            f"{NEW_CRATE_RATE.seconds // 60} minutes"
            if new
            else f"crates.io allows {NEW_VERSION_BURST} new versions at once, then one "
            f"every {NEW_VERSION_RATE.seconds // 60} minute"
        )
        for _ in range(MAX_RATE_LIMITED_ATTEMPTS):
            self.wait_until(bucket.ready_at(self.clock()), name, limit)
            print(f"publishing {name} {self.version} as {kind}", flush=True)
            status, output = self.run(
                ["cargo", "publish", "--package", name, "--locked", "--no-verify"]
            )
            if status == 0:
                bucket.take(self.clock())
                return "published"
            if already_uploaded(output):
                print(f"{name} {self.version} was already uploaded; continuing")
                return "skipped"
            if not rate_limited(output):
                raise SystemExit(f"::error::publishing {name} {self.version} failed")
            until = retry_after(output)
            if until is None:
                until = self.clock() + bucket.rate
                print(f"crates.io refused {name} with 429 and no retry time; backing off")
            bucket.empty_until(until + RETRY_MARGIN)
        raise SystemExit(
            f"::error::crates.io refused {name} {self.version} with 429 "
            f"{MAX_RATE_LIMITED_ATTEMPTS} times in a row"
        )


def publishable_packages() -> list[dict]:
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    # `publish = false` reads back as an empty registry list.
    return [package for package in metadata["packages"] if package["publish"] != []]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="the version being released")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be published"
    )
    args = parser.parse_args(argv)

    packages = publishable_packages()
    # crates.io refuses a package without a license, which would fail the
    # release partway through the workspace.
    unlicensed = [
        package["name"]
        for package in packages
        if package["license"] is None and package["license_file"] is None
    ]
    if unlicensed:
        print("::error::these packages declare no license or license-file, which crates.io requires:")
        print("\n".join(unlicensed))
        return 1

    publisher = Publisher(args.version, dry_run=args.dry_run)
    results = {name: publisher.publish(name) for name in publish_order(packages)}
    published = [name for name, result in results.items() if result == "published"]
    print(
        f"{len(published)} published, "
        f"{sum(result == 'skipped' for result in results.values())} already on crates.io"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
