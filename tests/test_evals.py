import json
from pathlib import Path

import pytest

from evals.run_evals import GOLDEN, load_cases, main, run_case, summarise


def test_golden_set_has_twenty_two_valid_cases():
    cases = load_cases(GOLDEN)
    assert len(cases) == 22
    assert len({c["id"] for c in cases}) == 22, "duplicate ids"
    for c in cases:
        assert "mock" in c, f"{c['id']} has no mock script; offline eval would be vacuous"


def test_every_case_passes_in_mock_mode(monkeypatch):
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    results = [run_case(c, "mock", judge=None) for c in load_cases(GOLDEN)]
    failing = [(r.id, r.details) for r in results if not r.passed]
    assert not failing, failing
    summary = summarise(results)
    assert summary["pass_rate"] == 1.0
    assert summary["by_category"]["calculator"]["passed"] == 8


def test_gate_exit_codes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    out = tmp_path / "r.json"
    assert main(["--provider", "mock", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["summary"]["total"] == 22

    # Flip one case's mock to a wrong answer -> below a 100% threshold -> exit 1.
    cases = load_cases(GOLDEN)
    cases[0]["mock"] = ["wrong"]
    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join(json.dumps(c) for c in cases))
    assert main(["--provider", "mock", "--golden", str(broken), "--threshold", "1.0"]) == 1
    assert main(["--provider", "mock", "--golden", str(broken), "--threshold", "0.9"]) == 0


def test_unknown_check_type_rejected(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "prompt": "p", "checks": [{"type": "nope"}]}))
    with pytest.raises(ValueError, match="unknown check type"):
        load_cases(bad)
