"""Moved to `fde_kit.evals.runner` so an installed fde-kit can run evals. This shim keeps
`python -m evals.run_evals` and `import evals.run_evals` working from a checkout; the import
form returns the real module, so monkeypatching it patches the harness."""

import sys

from fde_kit.evals import runner as _runner

if __name__ == "__main__":
    sys.exit(_runner.main())
sys.modules[__name__] = _runner
