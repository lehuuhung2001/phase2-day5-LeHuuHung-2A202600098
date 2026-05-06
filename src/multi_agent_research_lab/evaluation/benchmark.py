"""Benchmark helpers — single-agent vs multi-agent comparison."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a single experiment and return state + metrics.

    Measures: latency, token cost, quality score (from CriticAgent if present),
    citation coverage, and error rate.
    """
    started = perf_counter()
    state = runner(query)
    latency = round(perf_counter() - started, 3)

    total_cost = _sum_cost(state)
    total_in, total_out = _sum_tokens(state)
    quality = _extract_quality(state)
    citation_coverage = _citation_coverage(state)
    error_rate = 1.0 if state.errors else 0.0

    notes_parts = []
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")
    if citation_coverage is not None:
        notes_parts.append(f"citations={citation_coverage:.0%}")
    if total_in or total_out:
        notes_parts.append(f"tokens={total_in}in/{total_out}out")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        notes=", ".join(notes_parts),
    )
    logger.info(
        "Benchmark '%s': %.2fs | cost=$%.5f | quality=%s | errors=%d",
        run_name,
        latency,
        total_cost or 0,
        f"{quality:.1f}" if quality is not None else "n/a",
        len(state.errors),
    )
    return state, metrics


def run_comparison(query: str) -> tuple[BenchmarkMetrics, BenchmarkMetrics]:
    """Run baseline and multi-agent on the same query and return both metrics."""
    from multi_agent_research_lab.core.schemas import ResearchQuery
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
    from multi_agent_research_lab.services.llm_client import LLMClient

    llm = LLMClient()

    def _baseline_runner(q: str) -> ResearchState:
        from multi_agent_research_lab.core.state import ResearchState

        state = ResearchState(request=ResearchQuery(query=q))
        system_prompt = (
            "You are a research assistant. Answer the following query comprehensively "
            "in approximately 500 words, citing key points."
        )
        response = llm.complete(system_prompt, q)
        state.final_answer = response.content
        state.agent_results = [
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        ]
        return state

    def _multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    logger.info("Running comparison benchmark for: %s", query[:80])
    _, baseline_metrics = run_benchmark("single-agent-baseline", query, _baseline_runner)
    _, multi_metrics = run_benchmark("multi-agent-workflow", query, _multi_runner)
    return baseline_metrics, multi_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sum_cost(state: ResearchState) -> float | None:
    costs = [r.metadata.get("cost_usd") for r in state.agent_results if r.metadata.get("cost_usd")]
    return round(sum(costs), 6) if costs else None  # type: ignore[arg-type]


def _sum_tokens(state: ResearchState) -> tuple[int, int]:
    in_tok = sum(r.metadata.get("input_tokens") or 0 for r in state.agent_results)
    out_tok = sum(r.metadata.get("output_tokens") or 0 for r in state.agent_results)
    return in_tok, out_tok


def _extract_quality(state: ResearchState) -> float | None:
    for r in reversed(state.agent_results):
        if r.agent == AgentName.CRITIC:
            score = r.metadata.get("quality_score")
            if score is not None:
                return float(score)
    return None


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.final_answer or not state.sources:
        return None
    cited = sum(
        1 for i in range(1, len(state.sources) + 1) if f"[{i}]" in state.final_answer
    )
    return cited / len(state.sources)
