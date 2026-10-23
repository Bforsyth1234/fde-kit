"""`AgentActivities._run_with_heartbeat`: the side-thread wait loop (PR #12).

`fut.result(timeout=...)` raises TimeoutError in two different situations: the wait timed out
while `fn` is still running (heartbeat and keep waiting), and `fn` itself raised TimeoutError
(propagate it). A future can also finish between the wait timing out and the `done()` check;
its own outcome must win. No Temporal server is needed: these drive the loop directly.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("temporalio")

from fde_kit.durable import activities  # noqa: E402


@pytest.fixture
def beats(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(activities.activity, "heartbeat", lambda detail: calls.append(detail))
    return calls


class Runner:
    heartbeat_every = 0.02
    _run_with_heartbeat = activities.AgentActivities._run_with_heartbeat

    def __init__(self, pool):
        self._pool = pool


def _run(runner, fn, *, limit: float = 3.0):
    """Run in a thread so a regression to the infinite loop fails the test instead of hanging."""
    box: dict = {}

    def go():
        try:
            box["value"] = runner._run_with_heartbeat(fn, "detail")
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(limit)
    assert not t.is_alive(), "_run_with_heartbeat did not return: it is looping"
    return box


def test_slow_function_heartbeats_until_it_returns(beats):
    def slow():
        time.sleep(0.15)
        return 42

    with ThreadPoolExecutor(1) as pool:
        box = _run(Runner(pool), slow)
    assert box == {"value": 42}
    assert beats and set(beats) == {"detail"}


def test_timeout_error_raised_by_fn_propagates_instead_of_looping(beats):
    def times_out():
        raise TimeoutError("socket timed out")

    with ThreadPoolExecutor(1) as pool:
        box = _run(Runner(pool), times_out)
    assert isinstance(box["error"], TimeoutError) and "socket timed out" in str(box["error"])


def test_other_exceptions_from_fn_propagate(beats):
    def boom():
        raise RuntimeError("tool broke")

    with ThreadPoolExecutor(1) as pool:
        box = _run(Runner(pool), boom)
    assert isinstance(box["error"], RuntimeError)


def test_future_that_finishes_between_timeout_and_check_returns_its_value(beats):
    """The wait times out, then fn completes successfully before `done()` is checked. The
    result must be returned; re-raising the wait's TimeoutError would fail an activity whose
    tool succeeded, and Temporal's retry would run the tool a second time."""

    class FinishesDuringTimeout:
        calls = 0

        def result(self, timeout=None):
            FinishesDuringTimeout.calls += 1
            if FinishesDuringTimeout.calls == 1:
                raise TimeoutError()  # the wait itself timed out
            return "tool result"

        def done(self):
            return True  # ...and fn finished in the meantime

    class Pool:
        def submit(self, *a, **k):
            return FinishesDuringTimeout()

    box = _run(Runner(Pool()), lambda: None)
    assert box == {"value": "tool result"}
    assert beats == []
