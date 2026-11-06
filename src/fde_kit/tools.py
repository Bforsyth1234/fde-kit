"""Tool registry.

A tool is a plain Python function decorated with `@tool`. The JSON schema the model sees is
derived from the function's type hints and docstring via Pydantic, so there is one source of
truth. Tools are grouped in a `ToolRegistry`, which is what you hand to the agent.

    registry = ToolRegistry()

    @registry.tool
    def add(a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        return a + b

Execution errors are caught and returned as `ToolResult(is_error=True)` so the model can recover
instead of the run crashing.
"""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, create_model


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., Any]
    requires_approval: bool = False
    """If True, the durable runtime pauses and waits for a human signal before executing."""

    def to_api_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


@dataclass(frozen=True)
class Source:
    """One record a tool drew on. The unit of citation.

    A tool that returns `ToolResult(content, sources=(...))` is saying: this content came from
    these records. The agent loop numbers every Source it sees across a run (`[1]`, `[2]`, ...),
    the model cites by that number, and `AgentResult.cited` resolves the numbers back to Sources.
    The citation is therefore produced by the tool layer and checked by the loop, never asserted
    by the model. See citations.py.
    """

    source_id: str
    """Stable id of the record: a chunk id, a fact id, a row key."""
    uri: str
    """Where a person can look it up: a file path, a URL, a document reference."""
    title: str = ""
    doc_id: str | None = None
    content_hash: str | None = None
    """sha256 of the record body at ingest, so a citation can prove what text it pointed at."""
    span: str | None = None
    """Location inside the document, e.g. "chars:1200-1850" or "rows:14-19"."""
    tenant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    sources: tuple[Source, ...] = ()
    """Records the content came from. Content may refer to them with local markers `[#1]`,
    `[#2]`, ... (1-based into `sources`); the loop rewrites those to run-wide `[n]` numbers."""


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    tenant_id: str | None = None
    """Set when every tool in this registry serves one tenant. An Agent using the registry then
    runs inside `tracing.tenant_scope(tenant_id)`: spans tagged, content redacted."""

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        requires_approval: bool = False,
    ):
        """Decorator. Works bare (`@registry.tool`) or with args (`@registry.tool(name=...)`).

        `requires_approval=True` marks a tool as gated: the in-process Agent executes it
        normally, but the durable runtime pauses the run and waits for an approve/reject signal.
        """

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            spec = _spec_from_function(f, name=name)
            spec.requires_approval = requires_approval
            if spec.name in self._tools:
                raise ValueError(f"tool already registered: {spec.name}")
            self._tools[spec.name] = spec
            return f

        return register(fn) if fn is not None else register

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def api_schemas(self) -> list[dict[str, Any]]:
        return [t.to_api_schema() for t in self._tools.values()]

    def requires_approval(self, name: str) -> bool:
        spec = self._tools.get(name)
        return bool(spec and spec.requires_approval)

    def execute(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}",
                is_error=True,
            )
        try:
            args = spec.input_model(**raw_input)
        except ValidationError as exc:
            return ToolResult(f"Invalid input for '{name}': {exc}", is_error=True)
        try:
            out = spec.fn(**args.model_dump())
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            return ToolResult(f"{type(exc).__name__}: {exc}\n{tb}", is_error=True)
        if isinstance(out, ToolResult):
            return out
        return ToolResult(_stringify(out))


def _spec_from_function(fn: Callable[..., Any], *, name: str | None) -> ToolSpec:
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        annotation = param.annotation if param.annotation is not inspect._empty else Any
        default = ... if param.default is inspect._empty else param.default
        fields[pname] = (annotation, default)
    model = create_model(f"{fn.__name__}_input", **fields)  # type: ignore[call-overload]
    doc = inspect.getdoc(fn) or fn.__name__
    return ToolSpec(name=name or fn.__name__, description=doc, input_model=model, fn=fn)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        import json

        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def load_registry(spec: str) -> ToolRegistry:
    """Load a customer's tools from `module:attr` or `path/to/file.py:attr`.

    `attr` is a `ToolRegistry`, or a callable taking no arguments that returns one. A module
    name is imported with the current directory on `sys.path`, so `fde ... --tools
    acme_agent:tools` works from the project folder without installing it. This is how the CLI
    (`fde run`, `fde rag ask`, `fde eval`) takes tools it doesn't ship with; for a tenant run the
    retrieval tools are added on top (`tenant_search_registry` copies, never mutates, the base).
    """
    import importlib
    import importlib.util
    import os
    import sys
    from pathlib import Path

    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(f"--tools {spec!r}: expected module:attr or path/to/file.py:attr")
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.is_file():
            raise ValueError(f"--tools {spec!r}: no such file {path}")
        mod_name = f"_fde_tools_{path.stem}"
        loader_spec = importlib.util.spec_from_file_location(mod_name, path)
        if loader_spec is None or loader_spec.loader is None:  # pragma: no cover - .py always loads
            raise ValueError(f"--tools {spec!r}: cannot load {path}")
        module = importlib.util.module_from_spec(loader_spec)
        # The file's own folder first, so it can import its siblings.
        sys.path.insert(0, str(path.parent))
        sys.modules[mod_name] = module
        loader_spec.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(target)
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise ValueError(f"--tools {spec!r}: {target} has no attribute {attr!r}") from None
    if callable(obj) and not isinstance(obj, ToolRegistry):
        obj = obj()
    if not isinstance(obj, ToolRegistry):
        raise ValueError(
            f"--tools {spec!r}: expected a ToolRegistry (or a function returning one), "
            f"got {type(obj).__name__}"
        )
    return obj
