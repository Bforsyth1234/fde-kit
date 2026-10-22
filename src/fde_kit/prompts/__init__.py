"""Versioned prompts.

A prompt is a Markdown file, Git-tracked, named by its stem. `load_prompt("default")` returns a
`PromptRef` carrying the text and its sha256, and every run records the name and hash: on
`AgentResult`, on the `agent.run` and `llm.call` spans, in the eval JSON, and in a durable run's
workflow history. Two eval runs with different hashes are two experiments.

Resolution order for `load_prompt(name)`:

1. `name` is a path to an existing file -> that file.
2. `<prompts_dir>/<name>.md` when `prompts_dir` (FDE_PROMPTS_DIR, default `./prompts`) exists.
3. The prompt packaged with fde-kit, `fde_kit/prompts/<name>.md`.

Git is the version store. The hash is what lets a run point at an exact revision of the prompt
without needing Git at runtime, and what makes "which prompt produced this?" answerable from a
trace, an eval file or a workflow history alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PACKAGED = Path(__file__).parent


@dataclass(frozen=True)
class PromptRef:
    name: str
    sha256: str
    text: str

    @classmethod
    def inline(cls, text: str, name: str = "inline") -> PromptRef:
        """A prompt passed as a raw string. Hashed like any other so it is still traceable."""
        return cls(name=name, sha256=prompt_hash(text), text=text)

    @property
    def short(self) -> str:
        return f"{self.name}@{self.sha256[:12]}"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt(name: str = "default", prompts_dir: str | Path | None = None) -> PromptRef:
    path = _resolve(name, prompts_dir)
    text = path.read_text(encoding="utf-8").strip()
    ref_name = path.stem if Path(name).suffix else name
    return PromptRef(name=ref_name, sha256=prompt_hash(text), text=text)


def _resolve(name: str, prompts_dir: str | Path | None) -> Path:
    direct = Path(name)
    if direct.suffix and direct.is_file():
        return direct
    for base in (Path(prompts_dir) if prompts_dir else Path("prompts"), PACKAGED):
        candidate = base / f"{name}.md"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"prompt {name!r} not found: no file at {name!r}, no "
        f"{(Path(prompts_dir) if prompts_dir else Path('prompts')) / (name + '.md')}, "
        f"and nothing packaged at {PACKAGED / (name + '.md')}"
    )
