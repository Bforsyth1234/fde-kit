"""Prompt versioning: a prompt is a named, hashed file, and every run records which one it used."""

import hashlib

from fde_kit import Agent, Settings
from fde_kit.builtin_tools import registry
from fde_kit.llm import MockProvider
from fde_kit.prompts import PACKAGED, PromptRef, load_prompt


def test_packaged_default_prompt_loads_with_hash():
    ref = load_prompt("default")
    text = (PACKAGED / "default.md").read_text().strip()
    assert ref.name == "default" and ref.text == text
    assert ref.sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert "[1]" in ref.text  # the default asks for numbered citations


def test_project_prompts_dir_shadows_the_packaged_one(tmp_path):
    (tmp_path / "default.md").write_text("Project prompt.\n")
    ref = load_prompt("default", prompts_dir=tmp_path)
    assert ref.text == "Project prompt." and ref.name == "default"
    assert ref.sha256 != load_prompt("default").sha256


def test_prompt_by_path(tmp_path):
    f = tmp_path / "terse.md"
    f.write_text("Answer in one word.")
    ref = load_prompt(str(f))
    assert ref.name == "terse" and ref.text == "Answer in one word."


def test_run_records_prompt_name_and_hash():
    settings = Settings(provider="mock", otel_enabled=False)
    result = Agent(registry, provider=MockProvider(["ok"]), settings=settings).run("hi")
    assert result.prompt_name == "default"
    assert result.prompt_sha256 == load_prompt("default").sha256
    assert result.model == "mock"


def test_inline_prompt_is_still_hashed():
    settings = Settings(provider="mock", otel_enabled=False)
    provider = MockProvider(["ok"])
    agent = Agent(registry, provider=provider, settings=settings, system_prompt="Be brief.")
    result = agent.run("hi")
    assert result.prompt_name == "inline"
    assert result.prompt_sha256 == PromptRef.inline("Be brief.").sha256
    assert provider.calls[0]["system"] == "Be brief."


def test_env_system_prompt_overrides_named_prompt():
    settings = Settings(provider="mock", otel_enabled=False, system_prompt="From env.")
    provider = MockProvider(["ok"])
    result = Agent(registry, provider=provider, settings=settings).run("hi")
    assert provider.calls[0]["system"] == "From env." and result.prompt_name == "inline"
