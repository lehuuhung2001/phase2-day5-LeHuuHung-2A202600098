"""Tracing hooks — local JSON spans + optional LangSmith integration.

Usage:
  with trace_span("agent.run", {"key": "val"}) as span:
      ...
      span["extra_field"] = "whatever"

At the end of a workflow call save_trace_to_file(state) to export the full trace.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# LangSmith run context (set once per workflow invocation)
_ls_run: Any = None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Timing span that logs duration and optionally forwards to LangSmith."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "error": None,
    }
    _ls_span = _start_ls_span(name, attributes)
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        _end_ls_span(_ls_span, error=exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug(
            "SPAN %-35s %.3fs %s",
            name,
            span["duration_seconds"],
            _format_attrs(span),
        )
        if span["error"] is None:
            _end_ls_span(_ls_span)


def save_trace_to_file(state: Any, directory: str | Path = "reports/traces") -> Path:
    """Dump the full workflow trace (state.trace) to a timestamped JSON file."""
    from multi_agent_research_lab.core.state import ResearchState

    assert isinstance(state, ResearchState)
    outdir = Path(directory)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = outdir / f"trace_{ts}.json"
    latest = outdir / "latest.json"

    payload = {
        "timestamp": ts,
        "query": state.request.query,
        "iterations": state.iteration,
        "route_history": state.route_history,
        "errors": state.errors,
        "trace_events": state.trace,
        "agent_results": [
            {
                "agent": r.agent,
                "metadata": r.metadata,
                "content_preview": r.content[:200],
            }
            for r in state.agent_results
        ],
    }

    text = json.dumps(payload, indent=2, default=str)
    filename.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    logger.info("Trace saved → %s", filename)
    return filename


# ---------------------------------------------------------------------------
# LangSmith integration (no-op when not configured)
# ---------------------------------------------------------------------------

def _start_ls_span(name: str, attributes: dict[str, Any] | None) -> Any:
    try:
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        if not settings.langsmith_api_key:
            return None
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # enables LangGraph auto-tracing
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.debug("LangSmith tracing enabled → project=%s", settings.langsmith_project)
        return None
    except Exception:
        return None


def _end_ls_span(span: Any, error: Exception | None = None) -> None:
    pass  # lifecycle handled by LangSmith auto-instrumentation via env vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_attrs(span: dict[str, Any]) -> str:
    attrs = {k: v for k, v in span.items() if k not in {"name", "duration_seconds", "attributes", "error"}}
    attrs.update(span.get("attributes", {}))
    if not attrs:
        return ""
    return "| " + " ".join(f"{k}={v}" for k, v in list(attrs.items())[:4])
