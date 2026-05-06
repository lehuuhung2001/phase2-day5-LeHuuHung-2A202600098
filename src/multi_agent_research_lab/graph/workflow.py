"""Multi-agent workflow — LangGraph StateGraph with simple-loop fallback.

Engine selection:
  - Primary: LangGraph StateGraph (if `langgraph` is installed)
  - Fallback: simple while-loop with identical routing logic

Agent-level fallback: each worker is wrapped in try/except so the workflow
never crashes — errors are captured in state.errors and a graceful partial
answer is always produced.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_DONE,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-level fallback helpers
# ---------------------------------------------------------------------------

def _run_with_fallback(
    agent_name: str,
    run_fn: Any,
    state: ResearchState,
    fallback_fn: Any,
) -> ResearchState:
    """Run an agent; on failure apply fallback and continue."""
    try:
        return run_fn(state)
    except AgentExecutionError as exc:
        logger.error("Agent '%s' failed: %s — applying fallback", agent_name, exc)
        state.errors.append(f"{agent_name}: {exc}")
        return fallback_fn(state)


def _researcher_fallback(state: ResearchState) -> ResearchState:
    state.research_notes = "[Research failed — proceeding with empty notes]"
    return state


def _analyst_fallback(state: ResearchState) -> ResearchState:
    state.analysis_notes = "[Analysis skipped due to error]"
    return state


def _writer_fallback(state: ResearchState) -> ResearchState:
    state.final_answer = (
        state.research_notes or "No information could be gathered."
    ) + "\n\n[Note: Final synthesis failed; this is the raw research output.]"
    return state


# ---------------------------------------------------------------------------
# LangGraph engine
# ---------------------------------------------------------------------------

def _build_langgraph(
    supervisor: SupervisorAgent,
    researcher: ResearcherAgent,
    analyst: AnalystAgent,
    writer: WriterAgent,
) -> Any:
    from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]
    from typing import TypedDict

    class WorkflowState(TypedDict, total=False):
        request: dict[str, Any]
        iteration: int
        route_history: list[str]
        sources: list[dict[str, Any]]
        research_notes: str | None
        analysis_notes: str | None
        final_answer: str | None
        agent_results: list[dict[str, Any]]
        trace: list[dict[str, Any]]
        errors: list[str]

    def _to_rs(s: dict[str, Any]) -> ResearchState:
        return ResearchState.model_validate(s)

    def _sup_node(s: WorkflowState) -> WorkflowState:
        rs = supervisor.run(_to_rs(s))
        return rs.model_dump()  # type: ignore[return-value]

    def _res_node(s: WorkflowState) -> WorkflowState:
        rs = _run_with_fallback("researcher", researcher.run, _to_rs(s), _researcher_fallback)
        return rs.model_dump()  # type: ignore[return-value]

    def _ana_node(s: WorkflowState) -> WorkflowState:
        rs = _run_with_fallback("analyst", analyst.run, _to_rs(s), _analyst_fallback)
        return rs.model_dump()  # type: ignore[return-value]

    def _wri_node(s: WorkflowState) -> WorkflowState:
        rs = _run_with_fallback("writer", writer.run, _to_rs(s), _writer_fallback)
        return rs.model_dump()  # type: ignore[return-value]

    def _route(s: dict[str, Any]) -> str:  # plain dict annotation avoids LangGraph ForwardRef issues
        history: list[str] = s.get("route_history", [])  # type: ignore[assignment]
        last = history[-1] if history else ROUTE_DONE
        return END if last == ROUTE_DONE else last

    graph: StateGraph = StateGraph(WorkflowState)
    graph.add_node("supervisor", _sup_node)
    graph.add_node("researcher", _res_node)
    graph.add_node("analyst", _ana_node)
    graph.add_node("writer", _wri_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Simple-loop engine (fallback when langgraph not installed)
# ---------------------------------------------------------------------------

def _run_simple_loop(
    state: ResearchState,
    supervisor: SupervisorAgent,
    researcher: ResearcherAgent,
    analyst: AnalystAgent,
    writer: WriterAgent,
) -> ResearchState:
    settings = get_settings()
    workers = {
        "researcher": lambda s: _run_with_fallback("researcher", researcher.run, s, _researcher_fallback),
        "analyst": lambda s: _run_with_fallback("analyst", analyst.run, s, _analyst_fallback),
        "writer": lambda s: _run_with_fallback("writer", writer.run, s, _writer_fallback),
    }

    while state.iteration < settings.max_iterations:
        state = supervisor.run(state)
        last_route = state.route_history[-1] if state.route_history else ROUTE_DONE
        if last_route == ROUTE_DONE:
            break
        worker = workers.get(last_route)
        if worker:
            state = worker(state)

    return state


# ---------------------------------------------------------------------------
# Public workflow class
# ---------------------------------------------------------------------------

class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Return a compiled LangGraph graph or the string 'simple_loop'."""
        try:
            import langgraph  # noqa: F401  # type: ignore[import-untyped]

            return _build_langgraph(
                SupervisorAgent(), ResearcherAgent(), AnalystAgent(), WriterAgent()
            )
        except ImportError:
            logger.info("langgraph not installed — will use simple-loop engine")
            return "simple_loop"

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow with timeout guard and return final state."""
        settings = get_settings()
        result_holder: list[ResearchState] = []
        exc_holder: list[BaseException] = []

        def _target() -> None:
            try:
                result_holder.append(self._run_inner(state))
            except Exception as exc:
                exc_holder.append(exc)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=settings.timeout_seconds)

        if thread.is_alive():
            logger.error("Workflow timed out after %ds", settings.timeout_seconds)
            state.errors.append(f"Workflow timed out after {settings.timeout_seconds}s")
            if state.final_answer is None:
                state.final_answer = (
                    "Research timed out. "
                    f"Partial findings: {state.research_notes or 'none collected'}."
                )
            return state

        if exc_holder:
            raise exc_holder[0]

        return result_holder[0]

    def _run_inner(self, state: ResearchState) -> ResearchState:
        with trace_span("workflow.run", {"query": state.request.query[:80]}) as span:
            supervisor = SupervisorAgent()
            researcher = ResearcherAgent()
            analyst = AnalystAgent()
            writer = WriterAgent()

            try:
                import langgraph  # noqa: F401  # type: ignore[import-untyped]

                logger.info("Running LangGraph workflow")
                graph = _build_langgraph(supervisor, researcher, analyst, writer)
                result_dict: dict[str, Any] = graph.invoke(state.model_dump())
                result = ResearchState.model_validate(result_dict)
                span["engine"] = "langgraph"
            except ImportError:
                logger.info("Using simple-loop workflow engine")
                result = _run_simple_loop(state, supervisor, researcher, analyst, writer)
                span["engine"] = "simple_loop"

            # Run CriticAgent after main workflow to produce quality_score
            result = _run_with_fallback("critic", CriticAgent().run, result, lambda s: s)

            result.add_trace_event(
                "workflow.complete",
                {
                    "iterations": result.iteration,
                    "route_history": result.route_history,
                    "errors": result.errors,
                },
            )
            span["iterations"] = result.iteration
        return result
