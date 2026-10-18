import os
import sys
from pathlib import Path

# Make `evals` importable as a package when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Never try to export spans or hit a real API from the test suite.
os.environ.setdefault("FDE_OTEL_ENABLED", "false")
os.environ.setdefault("FDE_PROVIDER", "mock")
