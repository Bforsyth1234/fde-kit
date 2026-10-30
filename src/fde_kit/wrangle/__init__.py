"""Wrangle: messy CSV/XLSX tables into cited facts, with no LLM in the loop.

A person (later, an agent) writes a mapping spec; the code reads the file cell by cell, runs a
closed set of parse ops, and loads facts whose `source_ref` names the exact cell they came from.
See docs/pr5-wrangle-core.md.
"""

from .apply import ApplyResult, SpecMismatch, gate, parse_source_ref, run, write_outputs
from .labels import header_fingerprint, header_labels
from .parse import ParseError, run_ops
from .profile import Profile, profile, profile_file
from .read import Cell, Grid, read_grid
from .resolve import Resolver, check_crosswalk, norm_key
from .spec import LoadedSpec, MappingSpec, SpecError, approve, is_approved, load_project_spec

__all__ = [
    "ApplyResult",
    "Cell",
    "Grid",
    "LoadedSpec",
    "MappingSpec",
    "ParseError",
    "Profile",
    "Resolver",
    "SpecError",
    "SpecMismatch",
    "approve",
    "check_crosswalk",
    "gate",
    "header_fingerprint",
    "header_labels",
    "is_approved",
    "load_project_spec",
    "norm_key",
    "parse_source_ref",
    "profile",
    "profile_file",
    "read_grid",
    "run",
    "run_ops",
    "write_outputs",
]
