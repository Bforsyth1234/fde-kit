"""OpenTelemetry setup.

One call to `init_tracing()` at startup. Spans are exported over OTLP/HTTP to whatever is at
`FDE_OTEL_ENDPOINT` (Phoenix by default). If tracing is disabled, or the exporter can't be
imported, the kit falls back to a no-op tracer so nothing else has to care.

Span attribute names follow the OpenInference conventions where Phoenix has a view for them
(llm.*, tool.*), so traces render with token counts and tool I/O without extra configuration.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

from .config import Settings, get_settings

log = logging.getLogger(__name__)

_initialised = False


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
