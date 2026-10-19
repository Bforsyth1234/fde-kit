"""OpenTelemetry setup.

One call to `init_tracing()` at startup. Spans are exported over OTLP/HTTP to whatever is at
`FDE_OTEL_ENDPOINT` (Phoenix by default). If tracing is disabled, or the exporter can't be
imported, the kit falls back to a no-op tracer so nothing else has to care.

Span attribute names follow the OpenInference conventions where Phoenix has a view for them
(llm.*, tool.*), so traces render with token counts and tool I/O without extra configuration.

Tenant-scoped runs. A trace store is usually one shared system: every run's prompt, retrieved
passages and answer, side by side, readable by anyone with access to the UI. For a multi-tenant
deployment that is a cross-tenant leak outside the database. So inside `tenant_scope(tenant)`:

- every span gets a `tenant.id` attribute (TenantSpanProcessor), so traces can be filtered and
  access-scoped by tenant;
- content (prompts, messages, tool arguments and results, answers) is replaced at the source by
  `[redacted: N chars]` before it is set on a span, so it never reaches an exporter, a batch
  queue or a collector. Structure stays: span names, tool names, turns, token counts, latency,
  error flags.

`FDE_OTEL_TENANT_CONTENT=true` turns content back on for tenant runs. Only do that if the trace
store itself is partitioned or access-controlled per tenant. An Agent whose registry is bound to
a tenant (see `fde_kit.rag.tenant_search_registry`) enters the scope by itself.
"""

from __future__ import annotations

import json
import logging
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Tracer

from .config import Settings, get_settings

log = logging.getLogger(__name__)

_initialised = False
_tenant: ContextVar[str | None] = ContextVar("fde_tenant", default=None)
TENANT_ATTRIBUTE = "tenant.id"


@contextmanager
def tenant_scope(tenant_id: str | None) -> Iterator[None]:
    """Mark everything traced inside as belonging to `tenant_id`. None is a no-op."""
    if tenant_id is None:
        yield
        return
    token = _tenant.set(tenant_id)
    try:
        yield
    finally:
        _tenant.reset(token)


def current_tenant() -> str | None:
    return _tenant.get()


def trace_content(value: Any, limit: int = 4000) -> str:
    """What to put in a content attribute (input.value, output.value). Inside a tenant scope,
    a redaction marker unless FDE_OTEL_TENANT_CONTENT is on; otherwise `safe_json`."""
    text = value if isinstance(value, str) else safe_json(value, limit=10**9)
    if _tenant.get() is not None and not get_settings().otel_tenant_content:
        return f"[redacted: {len(text)} chars]"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class TenantSpanProcessor(SpanProcessor):
    """Stamps `tenant.id` on every span started inside a tenant scope."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        tid = _tenant.get()
        if tid is not None:
            span.set_attribute(TENANT_ATTRIBUTE, tid)


def init_tracing(settings: Settings | None = None) -> Tracer:
    """Configure a global tracer provider once and return the kit's tracer."""
    global _initialised
    settings = settings or get_settings()

    if _initialised or not settings.otel_enabled:
        return trace.get_tracer(settings.otel_service_name)

    if not _collector_reachable(settings.otel_endpoint):
        _initialised = True
        log.warning(
            "no trace collector at %s; tracing disabled for this process "
            "(start Phoenix with `docker compose up -d`, or set FDE_OTEL_ENABLED=false)",
            settings.otel_endpoint,
        )
        return trace.get_tracer(settings.otel_service_name)

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.otel_service_name})
        )
        provider.add_span_processor(TenantSpanProcessor())
        # Short timeouts: if no collector is running, fail fast instead of stalling exit.
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_endpoint, timeout=2),
                export_timeout_millis=2000,
            )
        )
        logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.ERROR)
        trace.set_tracer_provider(provider)
        _initialised = True
        log.info("tracing enabled -> %s", settings.otel_endpoint)
    except Exception as exc:  # pragma: no cover - depends on environment
        log.warning("tracing disabled: %s", exc)

    return trace.get_tracer(settings.otel_service_name)


def _collector_reachable(endpoint: str, timeout: float = 0.5) -> bool:
    """TCP connect to the collector's host:port. Cheap, and avoids a retry storm on exit."""
    try:
        from urllib.parse import urlparse

        u = urlparse(endpoint)
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((u.hostname or "localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def get_tracer() -> Tracer:
    return trace.get_tracer(get_settings().otel_service_name)


def safe_json(value: Any, limit: int = 4000) -> str:
    """Serialise anything for a span attribute, truncating so traces stay small."""
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
