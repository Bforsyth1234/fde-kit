"""Wrangle: messy CSV/XLSX tables into cited facts, with no LLM in the loop.

A person (later, an agent) writes a mapping spec; the code reads the file cell by cell, runs a
closed set of parse ops, and loads facts whose `source_ref` names the exact cell they came from.
See docs/pr5-wrangle-core.md.
"""
