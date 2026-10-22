import os
import sys
from pathlib import Path

# Make `evals` importable as a package when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Never try to export spans or hit a real API from the test suite.
os.environ.setdefault("FDE_OTEL_ENABLED", "false")
os.environ.setdefault("FDE_PROVIDER", "mock")


# Temporal test environments hand out a client with the default converter. When a payload key
# is in the environment (CI sets one), subprocess workers encrypt, so the test client must
# decrypt: patch start_local to connect its client with the kit's codec.
try:
    from temporalio.testing import WorkflowEnvironment as _WE

    from fde_kit.durable.crypto import data_converter as _dc
    from fde_kit.durable.worker import payload_cipher as _pc

    _orig_start_local = _WE.start_local.__func__  # type: ignore[attr-defined]

    async def _start_local_with_codec(cls, *a, **kw):
        kw.setdefault("data_converter", _dc(_pc()))
        return await _orig_start_local(cls, *a, **kw)

    _WE.start_local = classmethod(_start_local_with_codec)  # type: ignore[method-assign]
except ModuleNotFoundError:
    pass
