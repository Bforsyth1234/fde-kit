"""A tool with an observable side effect, for the journal crash-window tests.

`charge_card` appends a line to the file named by FDE_COUNTER_FILE (the side effect) and
returns. The activity journals the result, then (because the test sets
FDE_TEST_HANG_AFTER_JOURNAL) holds the activity open so the test can kill the worker after the
journal write but before Temporal records the activity as complete. That is the window the
journal exists for, widened from microseconds to seconds so a test can hit it reliably.
"""

from __future__ import annotations

import os
import time

from fde_kit.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool
def charge_card(amount: int) -> str:
    """Charge the customer's card. Irreversible."""
    path = os.environ["FDE_COUNTER_FILE"]
    with open(path, "a") as f:
        f.write(f"CHARGED {amount} at {time.time()}\n")
    return f"charged ${amount}"
