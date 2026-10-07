#!/usr/bin/env python3
"""Tests for `publish_crates.py`.

Standard library only, and run by the `Rust checks` job with
`python3 -m unittest discover`, like the other script tests here.

What is pinned is what a release cannot observe until it is already failing:
that a 429 is waited out until the time crates.io names and the same crate is
retried, that new crates are paced to crates.io's limit before it is reached,
that new versions of existing crates are not slowed down, and that crates
already published are skipped so a re-run resumes.
"""

from __future__ import annotations

import datetime as dt
import io
import unittest
from contextlib import redirect_stdout

import publish_crates
from publish_crates import Bucket, Publisher, publish_order, retry_after

START = dt.datetime(2026, 10, 7, 17, 0, tzinfo=dt.timezone.utc)

TOO_MANY_NEW = (
    "error: failed to publish zvidlib-opus v0.4.0 to registry at https://crates.io\n"
    "Caused by:\n"
    "  the remote server responded with an error (status 429 Too Many Requests): "
    "You have published too many new crates in a short period of time. Please try "
    "again after Wed, 07 Oct 2026 17:17:34 GMT and see "
    "https://crates.io/docs/rate-limits for more details.\n"
)


def package(name: str, *dependencies: str, dev: tuple[str, ...] = ()) -> dict:
    return {
        "name": name,
        "dependencies": [{"name": d, "kind": None} for d in dependencies]
        + [{"name": d, "kind": "dev"} for d in dev],
    }


class FakeRegistry:
    """A clock, a sleep that advances it, and a `cargo publish` that answers
    from a scripted list of results."""

    def __init__(self, published: dict[str, set[str]], answers: list[tuple[int, str]] | None = None):
        self.now = START
        self.published = published
        self.answers = list(answers or [])
        self.uploads: list[tuple[str, dt.datetime]] = []
        self.slept: list[float] = []

    def clock(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += dt.timedelta(seconds=seconds)

    def lookup(self, name: str) -> set[str] | None:
        return self.published.get(name)

    def run(self, command: list[str]) -> tuple[int, str]:
        name = command[command.index("--package") + 1]
        self.uploads.append((name, self.now))
        status, output = self.answers.pop(0) if self.answers else (0, "")
        if status == 0:
            self.published.setdefault(name, set()).add("0.4.0")
        return status, output

    def publisher(self) -> Publisher:
        return Publisher(
            "0.4.0", lookup=self.lookup, run=self.run, sleep=self.sleep, clock=self.clock
        )


def quietly(action):
    with redirect_stdout(io.StringIO()) as output:
        result = action()
    return result, output.getvalue()


class PublishOrderTest(unittest.TestCase):
    def test_each_crate_follows_its_dependencies(self):
        packages = [
            package("zvidlib", "zvidlib-container", "zvidlib-core"),
            package("zvidlib-container", "zvidlib-opus", "zvidlib-core"),
            package("zvidlib-opus", "zvidlib-core", dev=("zvidlib-container",)),
            package("zvidlib-core"),
        ]
        self.assertEqual(
            publish_order(packages),
            ["zvidlib-core", "zvidlib-opus", "zvidlib-container", "zvidlib"],
        )

    def test_dev_dependency_cycles_do_not_block_the_order(self):
        packages = [package("a", dev=("b",)), package("b", "a")]
        self.assertEqual(publish_order(packages), ["a", "b"])

    def test_a_real_cycle_is_reported(self):
        with self.assertRaises(SystemExit):
            publish_order([package("a", "b"), package("b", "a")])


class RetryAfterTest(unittest.TestCase):
    def test_reads_the_time_from_a_crates_io_429(self):
        self.assertEqual(
            retry_after(TOO_MANY_NEW),
            dt.datetime(2026, 10, 7, 17, 17, 34, tzinfo=dt.timezone.utc),
        )

    def test_none_without_a_time(self):
        self.assertIsNone(retry_after("status 429 Too Many Requests"))


class BucketTest(unittest.TestCase):
    def test_burst_then_one_per_rate(self):
        bucket = Bucket.full(dt.timedelta(minutes=10), 2, START)
        for _ in range(2):
            self.assertEqual(bucket.ready_at(START), START)
            bucket.take(START)
        self.assertEqual(bucket.ready_at(START), START + dt.timedelta(minutes=10))

    def test_an_idle_full_bucket_does_not_bank_extra_tokens(self):
        bucket = Bucket.full(dt.timedelta(minutes=10), 1, START)
        later = START + dt.timedelta(hours=1)
        bucket.take(later)
        self.assertEqual(bucket.ready_at(later), later + dt.timedelta(minutes=10))


class PublisherTest(unittest.TestCase):
    def test_a_429_waits_until_the_named_time_and_retries_the_same_crate(self):
        registry = FakeRegistry({}, answers=[(101, TOO_MANY_NEW), (0, "")])
        result, log = quietly(lambda: registry.publisher().publish("zvidlib-opus"))
        self.assertEqual(result, "published")
        retried_at = dt.datetime(2026, 10, 7, 17, 17, 39, tzinfo=dt.timezone.utc)
        self.assertEqual(
            registry.uploads, [("zvidlib-opus", START), ("zvidlib-opus", retried_at)]
        )
        self.assertIn("until 17:17:39 UTC, before publishing zvidlib-opus", log)

    def test_a_429_without_a_time_backs_off_by_the_rate(self):
        registry = FakeRegistry({}, answers=[(101, "status 429 Too Many Requests"), (0, "")])
        quietly(lambda: registry.publisher().publish("zvidlib-opus"))
        self.assertEqual(registry.uploads[1][1], START + dt.timedelta(minutes=10, seconds=5))

    def test_later_new_crates_are_paced_after_a_429(self):
        registry = FakeRegistry({}, answers=[(101, TOO_MANY_NEW)])
        publisher = registry.publisher()
        quietly(lambda: [publisher.publish(name) for name in ["a-crate", "b-crate"]])
        a_at, b_at = registry.uploads[1][1], registry.uploads[2][1]
        self.assertEqual(b_at - a_at, publish_crates.NEW_CRATE_RATE)

    def test_new_crates_are_paced_before_crates_io_refuses_them(self):
        registry = FakeRegistry({})
        publisher = registry.publisher()
        names = [f"crate-{i}" for i in range(publish_crates.NEW_CRATE_BURST + 2)]
        quietly(lambda: [publisher.publish(name) for name in names])
        times = [at for _, at in registry.uploads]
        burst = publish_crates.NEW_CRATE_BURST
        self.assertEqual(times[:burst], [START] * burst)
        self.assertEqual(times[burst] - START, publish_crates.NEW_CRATE_RATE)
        self.assertEqual(times[burst + 1] - times[burst], publish_crates.NEW_CRATE_RATE)

    def test_new_versions_of_existing_crates_are_not_slowed_down(self):
        names = [f"crate-{i}" for i in range(21)]
        registry = FakeRegistry({name: {"0.3.0"} for name in names})
        publisher = registry.publisher()
        quietly(lambda: [publisher.publish(name) for name in names])
        self.assertEqual(registry.slept, [])
        self.assertEqual(len(registry.uploads), 21)

    def test_a_crate_already_at_the_version_is_skipped(self):
        registry = FakeRegistry({"zvidlib-core": {"0.4.0"}})
        result, _ = quietly(lambda: registry.publisher().publish("zvidlib-core"))
        self.assertEqual(result, "skipped")
        self.assertEqual(registry.uploads, [])

    def test_an_upload_the_index_has_not_shown_yet_is_skipped(self):
        registry = FakeRegistry(
            {}, answers=[(101, "error: crate zvidlib-opus@0.4.0 already exists on crates.io index")]
        )
        result, _ = quietly(lambda: registry.publisher().publish("zvidlib-opus"))
        self.assertEqual(result, "skipped")

    def test_any_other_failure_stops_the_publish(self):
        registry = FakeRegistry({}, answers=[(101, "error: failed to verify manifest")])
        with self.assertRaises(SystemExit):
            quietly(lambda: registry.publisher().publish("zvidlib-opus"))

    def test_a_429_that_never_clears_is_reported(self):
        answers = [(101, TOO_MANY_NEW)] * publish_crates.MAX_RATE_LIMITED_ATTEMPTS
        registry = FakeRegistry({}, answers=answers)
        with self.assertRaises(SystemExit):
            quietly(lambda: registry.publisher().publish("zvidlib-opus"))


class IndexPathTest(unittest.TestCase):
    def test_sparse_index_paths(self):
        self.assertEqual(publish_crates.index_path("a"), "1/a")
        self.assertEqual(publish_crates.index_path("ab"), "2/ab")
        self.assertEqual(publish_crates.index_path("abc"), "3/a/abc")
        self.assertEqual(publish_crates.index_path("Zvidlib-Core"), "zv/id/zvidlib-core")


if __name__ == "__main__":
    unittest.main()
