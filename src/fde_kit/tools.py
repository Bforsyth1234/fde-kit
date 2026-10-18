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
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, create_model


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., Any]

    def to_api_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None):
        """Decorator. Works bare (`@registry.tool`) or with args (`@registry.tool(name=...)`)."""

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            spec = _spec_from_function(f, name=name)
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
