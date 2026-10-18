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
