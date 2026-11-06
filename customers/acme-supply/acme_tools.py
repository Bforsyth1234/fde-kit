"""Acme Supply's own tools. The kit adds the retrieval tools for tenant `acme` on top:

fde rag ask --tenant acme --tools acme_tools:tools "What does a 200 lb order to zone B cost?"
fde eval --golden evals/golden_acme.jsonl --tools acme_tools:tools
"""

from __future__ import annotations

from fde_kit import ToolRegistry

ZONE_RATES = {"A": 0.18, "B": 0.27, "C": 0.41}
HANDLING = 35.0


def tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def freight_quote(weight_lb: float, zone: str) -> str:
        """Freight cost in USD for an Acme Supply order: $35 handling plus a per-pound zone rate.
        zone is A (up to 40 miles), B (41-120 miles) or C (over 120 miles). Use this for any
        freight or shipping cost question instead of doing the arithmetic yourself."""
        z = zone.strip().upper()
        if z not in ZONE_RATES:
            raise ValueError(f"unknown zone {zone!r}; use A, B or C")
        if weight_lb <= 0:
            raise ValueError("weight_lb must be positive")
        return f"${HANDLING + weight_lb * ZONE_RATES[z]:.2f} (zone {z}, {weight_lb:g} lb)"

    return reg
