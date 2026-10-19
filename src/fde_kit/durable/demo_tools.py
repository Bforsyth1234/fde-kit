"""Tools used by the crash demo: the built-ins plus one that takes a while.

A slow tool gives the demo a window in which to kill the worker mid-activity.
"""

from __future__ import annotations

import time

from ..builtin_tools import registry as _builtin
from ..tools import ToolRegistry

registry = ToolRegistry()
for _name in _builtin.names():
    registry.tool(_builtin.get(_name).fn)


@registry.tool
def wait_seconds(seconds: float) -> str:
    """Block for the given number of seconds, then return. Simulates a slow external call."""
    time.sleep(seconds)
    return f"waited {seconds} s"


@registry.tool(requires_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email on the user's behalf. This is an irreversible external action and
    requires human approval before it runs."""
    # Stub: a real deployment would call the mail provider here.
    return f"email queued to {to} with subject {subject!r} ({len(body)} chars)"


@registry.tool
def fetch_report(report_id: str) -> str:
    """Fetch a long internal report by id. Returns several thousand characters."""
    line = f"Report {report_id}: line {{i}} — metric value {{v}} units, status nominal.\n"
    return "".join(line.format(i=i, v=100 + i * 3) for i in range(60))
