"""Benchmark report rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics], query: str = "") -> str:
    """Render benchmark metrics to a rich Markdown report."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Benchmark Report — Single-Agent vs Multi-Agent",
        "",
        f"**Generated:** {ts}",
    ]
    if query:
        lines += [f"**Query:** {query}", ""]

    # --- Metrics table ---
    lines += [
        "## Metrics Comparison",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f}s | {cost} | {quality} | {item.notes} |"
        )

    # --- Delta analysis ---
    if len(metrics) == 2:
        m1, m2 = metrics[0], metrics[1]
        delta_lat = m2.latency_seconds - m1.latency_seconds
        lines += [
            "",
            "## Delta Analysis",
            "",
            f"- **Latency delta:** {delta_lat:+.2f}s "
            f"({'multi-agent is slower' if delta_lat > 0 else 'multi-agent is faster'})",
        ]
        if m1.estimated_cost_usd and m2.estimated_cost_usd:
            delta_cost = m2.estimated_cost_usd - m1.estimated_cost_usd
            lines.append(
                f"- **Cost delta:** ${delta_cost:+.5f} "
                f"({'higher' if delta_cost > 0 else 'lower'} for multi-agent)"
            )
        if m2.quality_score is not None:
            lines.append(f"- **Quality score (multi-agent):** {m2.quality_score:.1f} / 10")

    # --- Failure mode analysis ---
    lines += [
        "",
        "## Failure Mode Analysis",
        "",
        "| Scenario | Behaviour | Mitigation |",
        "|---|---|---|",
        "| Max iterations hit | Supervisor forces `done`, returns partial research | Increase `MAX_ITERATIONS` |",
        "| LLM call fails (3 retries) | `AgentExecutionError` → agent fallback | Mock LLM activates if no API key |",
        "| Researcher fails | `research_notes = '[Research failed]'`, workflow continues | Manual mock search fallback |",
        "| Writer fails | `final_answer` copied from `research_notes` | Ensures always-non-null output |",
        "| Timeout exceeded | Thread join timeout, partial answer returned | Tune `TIMEOUT_SECONDS` |",
        "",
        "## When to use Multi-Agent",
        "",
        "**Use multi-agent when:**",
        "- The task has clearly separable sub-tasks (research, analysis, writing).",
        "- You need independent quality checks (e.g. a critic reviewing the writer's output).",
        "- Parallel execution of sub-tasks is possible.",
        "",
        "**Do NOT use multi-agent when:**",
        "- The query is simple and fits in one LLM context window.",
        "- Latency is critical — orchestration overhead adds 1-3x latency.",
        "- Budget is tight — more agents = more tokens.",
    ]

    return "\n".join(lines) + "\n"
