"""The trace store as a cross-tenant leak, and the fix.

Every span the agent creates is captured with an in-memory exporter. For a run bound to a tenant,
no attribute on any span may contain the tenant's data (the question, the retrieved passages,
the tool arguments, the answer), and every span must carry `tenant.id`.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fde_kit import Agent, MockProvider, ToolRegistry
from fde_kit.tracing import TENANT_ATTRIBUTE, TenantSpanProcessor, tenant_scope, trace_content

QUESTION = "What is the late fee on my lease?"
PASSAGE = "LATE FEES: Rent received more than 3 days late incurs a late fee of $50."
ANSWER = "Your late fee is $50, after a 3-day grace period."
SECRETS = (QUESTION, PASSAGE, ANSWER, "$50", "late fee", "my lease")


def _registry(tenant_id: str | None) -> ToolRegistry:
    reg = ToolRegistry(tenant_id=tenant_id)

    @reg.tool
    def search_documents(query: str, k: int = 5) -> str:
        """Search."""
        return PASSAGE

    return reg


def _run(registry: ToolRegistry) -> list:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(TenantSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    mock = MockProvider(
        [
            {"tool_calls": [{"name": "search_documents", "input": {"query": "late fee"}}]},
            ANSWER,
        ]
    )
    agent = Agent(registry, provider=mock)
    agent._tracer = provider.get_tracer("test")
    result = agent.run(QUESTION)
    assert result.output == ANSWER  # redaction is for traces only; the caller gets the answer
    return list(exporter.get_finished_spans())


def _attribute_text(spans: list) -> str:
    parts = [f"{s.name} {k}={v}" for s in spans for k, v in (s.attributes or {}).items()]
    for s in spans:
        if s.status and s.status.description:
            parts.append(f"{s.name} status.description={s.status.description}")
        for ev in getattr(s, "events", ()) or ():
            for k, v in (ev.attributes or {}).items():
                parts.append(f"{s.name} event.{ev.name}.{k}={v}")
    return "\n".join(parts)


def test_tenant_run_records_no_tenant_content():
    spans = _run(_registry("summit"))
    text = _attribute_text(spans)
    for secret in SECRETS:
        assert secret not in text, f"{secret!r} reached a span"
    assert "[redacted]" in text


def test_redacted_marker_does_not_leak_content_length():
    """On a shared template, '$50' vs '5% of the monthly rent' changes the passage length by
    19 chars. The redaction marker must be constant so length cannot fingerprint the answer."""
    with tenant_scope("summit"):
        short = trace_content("late fee is $50")
    with tenant_scope("oakline"):
        long = trace_content("late fee is 5% of the monthly rent, after a 7-day grace period")
    assert short == long == "[redacted]"


def test_exception_in_tenant_run_scrubs_event_message_stacktrace_and_status():
    """OpenTelemetry's start_as_current_span records an `exception` event and sets the error
    status description to `str(exc)` by default. If a provider error echoes the prompt or
    passage, TenantSpanProcessor must scrub both before export while keeping exception.type."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(TenantSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    class FailingProvider:
        model = "failing"

        def complete(self, **kwargs):
            raise RuntimeError(f"provider rejected prompt: {QUESTION} / {PASSAGE}")

    agent = Agent(_registry("summit"), provider=FailingProvider())
    agent._tracer = provider.get_tracer("test")
    with pytest.raises(RuntimeError):
        agent.run(QUESTION)

    spans = list(exporter.get_finished_spans())
    text = _attribute_text(spans)
    for secret in SECRETS:
        assert secret not in text, f"{secret!r} leaked via span exception event or status"
    llm_span = next(s for s in spans if s.name == "llm.call")
    assert llm_span.events and llm_span.events[0].attributes["exception.type"] == "RuntimeError"
    assert llm_span.events[0].attributes["exception.message"] == "[redacted]"
    assert llm_span.events[0].attributes["exception.stacktrace"] == "[redacted]"
    assert llm_span.status.description == "[redacted]"


def test_tenant_run_tags_every_span():
    spans = _run(_registry("summit"))
    assert {s.name for s in spans} == {"agent.run", "llm.call", "tool.search_documents"}
    assert all((s.attributes or {}).get(TENANT_ATTRIBUTE) == "summit" for s in spans)


def test_tenant_run_keeps_the_structure_operators_need():
    spans = {s.name: s.attributes or {} for s in _run(_registry("summit"))}
    assert spans["tool.search_documents"]["tool.name"] == "search_documents"
    assert spans["tool.search_documents"]["tool.is_error"] is False
    assert spans["agent.run"]["agent.turns"] == 2
    assert spans["agent.run"]["llm.token_count.prompt"] > 0
    assert spans["llm.call"]["llm.stop_reason"] in ("tool_use", "end_turn")


def test_untenanted_run_still_records_content():
    """Single-tenant use of the kit keeps full traces; redaction is tied to a tenant scope."""
    spans = _run(_registry(None))
    text = _attribute_text(spans)
    assert QUESTION in text and PASSAGE in text and ANSWER in text
    assert not any(TENANT_ATTRIBUTE in (s.attributes or {}) for s in spans)


def test_content_can_be_turned_back_on_explicitly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FDE_OTEL_TENANT_CONTENT", "true")
    spans = _run(_registry("summit"))
    assert PASSAGE in _attribute_text(spans)
    assert all((s.attributes or {}).get(TENANT_ATTRIBUTE) == "summit" for s in spans)


def test_scope_ends_with_the_run():
    _run(_registry("summit"))
    assert trace_content("after the run") == "after the run"


def test_nested_scopes_restore_the_outer_tenant():
    with tenant_scope("harbor"):
        with tenant_scope("summit"):
            assert trace_content("x") == "[redacted]"
        assert trace_content("x") == "[redacted]"
    assert trace_content("x") == "x"
