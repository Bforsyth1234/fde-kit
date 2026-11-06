"""Break one thing on purpose, then run the evals. A good eval set fails.

    python evals/sabotage.py freight-rate --provider anthropic --ids tool-01,tool-02
    python evals/sabotage.py no-tools --provider anthropic

Extra flags go to `fde eval`. Run from the project folder.
"""

import os
import sys

from fde_kit.evals.runner import main

which, rest = sys.argv[1], sys.argv[2:]
sys.path.insert(0, os.getcwd())

if which == "freight-rate":
    # A typo in the tool: zone B at $0.29/lb instead of $0.27.
    import acme_tools

    acme_tools.ZONE_RATES["B"] = 0.29
elif which == "no-tools":
    # A prompt edit that tells the model to answer from general knowledge.
    os.environ["FDE_SYSTEM_PROMPT"] = (
        "You are a helpful HVAC supply assistant. Answer from general industry knowledge. "
        "Be brief. Do not call tools."
    )
else:
    sys.exit(f"unknown sabotage {which!r}")

sys.exit(main([*rest, "--golden", "evals/golden_acme.jsonl", "--tools", "acme_tools:tools"]))
