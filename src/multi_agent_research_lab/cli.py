"""Command-line entrypoint for the lab."""

from __future__ import annotations

from time import perf_counter
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import save_trace_to_file
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ---------------------------------------------------------------------------
# baseline — single-agent
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save: Annotated[bool, typer.Option("--save", help="Save result to reports/")] = False,
) -> None:
    """Run a real single-agent baseline: one LLM call, measured latency + cost."""
    _init()
    from multi_agent_research_lab.services.llm_client import LLMClient

    llm = LLMClient()
    system_prompt = (
        "You are a research assistant. Answer the following query comprehensively "
        "in approximately 500 words, citing key facts."
    )

    started = perf_counter()
    response = llm.complete(system_prompt, query)
    latency = perf_counter() - started

    state = ResearchState(request=ResearchQuery(query=query))
    state.final_answer = response.content

    # Display
    console.print(Panel.fit(response.content, title="[bold green]Single-Agent Baseline"))

    table = Table(title="Metrics", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Latency", f"{latency:.2f}s")
    table.add_row("Input tokens", str(response.input_tokens or "n/a"))
    table.add_row("Output tokens", str(response.output_tokens or "n/a"))
    table.add_row("Cost (USD)", f"${response.cost_usd:.5f}" if response.cost_usd else "n/a")
    console.print(table)

    if save:
        store = LocalArtifactStore()
        store.write_text("baseline_result.md", f"# Baseline\n\n{response.content}")
        console.print("[dim]Saved to reports/baseline_result.md")


# ---------------------------------------------------------------------------
# multi-agent
# ---------------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", help="Max search results")] = 5,
    save_trace: Annotated[bool, typer.Option("--trace", help="Save JSON trace")] = True,
) -> None:
    """Run the full multi-agent workflow: Supervisor → Researcher → Analyst → Writer."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query, max_sources=max_sources))
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    result = workflow.run(state)
    latency = perf_counter() - started

    # Display final answer
    console.print(Panel.fit(result.final_answer or "(no answer)", title="[bold blue]Multi-Agent Result"))

    # Trace table
    table = Table(title="Agent Trace", show_header=True)
    table.add_column("Agent")
    table.add_column("Tokens out", justify="right")
    table.add_column("Cost (USD)", justify="right")
    for r in result.agent_results:
        table.add_row(
            r.agent,
            str(r.metadata.get("output_tokens") or "n/a"),
            f"${r.metadata['cost_usd']:.5f}" if r.metadata.get("cost_usd") else "n/a",
        )
    table.add_row("[bold]TOTAL", "", "")
    console.print(table)

    console.print(f"[dim]Iterations: {result.iteration} | Latency: {latency:.2f}s | Errors: {len(result.errors)}")

    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="[red]Errors", style="red"))

    if save_trace:
        path = save_trace_to_file(result)
        console.print(f"[dim]Trace → {path}")


# ---------------------------------------------------------------------------
# benchmark — comparison report
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output file")] = "reports/benchmark_report.md",
) -> None:
    """Run baseline + multi-agent on the same query and produce a comparison report."""
    _init()
    from multi_agent_research_lab.evaluation.benchmark import run_comparison
    from multi_agent_research_lab.evaluation.report import render_markdown_report

    console.print(f"[bold]Running benchmark...[/bold] query={query[:60]!r}")

    baseline_m, multi_m = run_comparison(query)

    # Print summary table
    table = Table(title="Benchmark Results", show_header=True)
    for col in ("Run", "Latency (s)", "Cost (USD)", "Quality", "Notes"):
        table.add_column(col, justify="right" if col != "Run" and col != "Notes" else "left")

    for m in (baseline_m, multi_m):
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "n/a",
            f"{m.quality_score:.1f}" if m.quality_score is not None else "n/a",
            m.notes,
        )
    console.print(table)

    # Write report
    report_md = render_markdown_report([baseline_m, multi_m], query=query)
    store = LocalArtifactStore()
    import os
    rel_path = os.path.relpath(output)
    saved_path = store.write_text(os.path.basename(output), report_md)
    console.print(f"[green]Report saved → {saved_path}")


if __name__ == "__main__":
    app()
