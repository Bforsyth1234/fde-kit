"""A small set of deterministic tools used by the examples, tests and the golden eval set.

They exist so the kit is runnable end-to-end on day one. Replace them with real integrations
per deployment; the eval set should be rewritten alongside.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import operator

from .tools import Source, ToolRegistry, ToolResult

registry = ToolRegistry()

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@registry.tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Supports + - * / ** % and parentheses.
    Use this for any calculation instead of doing math in your head."""
    tree = ast.parse(expression, mode="eval")
    value = _eval(tree.body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@registry.tool
def current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return dt.date.today().isoformat()


_KB = {
    "warranty": "Standard warranty is 24 months from the invoice date.",
    "support hours": "Support is staffed 08:00-18:00 US Eastern, Monday through Friday.",
    "refund policy": "Refunds are issued within 30 days of purchase for unused licences.",
    "data retention": "Customer data is retained for 90 days after contract termination.",
    "sla": "Uptime SLA is 99.9% monthly, measured on the API gateway.",
}


def _policy_result(key: str) -> ToolResult:
    """The policy text as a numbered passage with its Source, so answers can cite it."""
    text = _KB[key]
    slug = key.replace(" ", "-")
    src = Source(
        source_id=f"policy:{slug}",
        uri=f"policies/{slug}.md",
        title=key.title(),
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    return ToolResult(f"[#1] {src.title} ({src.uri})\n{text}", sources=(src,))


@registry.tool
def lookup_policy(topic: str) -> ToolResult:
    """Look up an internal policy by topic. Known topics: warranty, support hours,
    refund policy, data retention, sla. Returns the policy as a numbered passage, e.g. [1],
    or a not-found message. Cite the passage number when you use it."""
    key = topic.strip().lower()
    if key in _KB:
        return _policy_result(key)
    matches = [k for k in _KB if key in k or k in key]
    if len(matches) == 1:
        return _policy_result(matches[0])
    return ToolResult(f"No policy found for '{topic}'. Known topics: {', '.join(_KB)}")
