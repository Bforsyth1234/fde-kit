"""Moved to `fde_kit.evals.scorers`; this shim keeps `import evals.scorers` working."""

import sys

from fde_kit.evals import scorers as _scorers

sys.modules[__name__] = _scorers
