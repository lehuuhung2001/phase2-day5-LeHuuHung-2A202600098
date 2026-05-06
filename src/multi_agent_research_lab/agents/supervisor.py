"""Supervisor / router agent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (deterministic — no LLM call needed):
      1. No research_notes → researcher
      2. No analysis_notes → analyst
      3. No final_answer  → writer
      4. All fields filled → done
    Guardrail: max_iterations exceeded → force done with fallback answer.
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("supervisor.run", {"iteration": state.iteration}) as span:
            route = self._decide_route(state)
            state.record_route(route)
            state.add_trace_event(
                "supervisor.route",
                {"route": route, "iteration": state.iteration},
            )
            span["route"] = route
            logger.info(
                "Supervisor [iter=%d] → %s | research=%s, analysis=%s, answer=%s",
                state.iteration,
                route,
                "✓" if state.research_notes else "✗",
                "✓" if state.analysis_notes else "✗",
                "✓" if state.final_answer else "✗",
            )
        return state

    def _decide_route(self, state: ResearchState) -> str:
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            logger.warning(
                "Max iterations (%d) reached — forcing done with fallback answer",
                settings.max_iterations,
            )
            if state.final_answer is None:
                state.final_answer = (
                    "Research incomplete: max iteration limit reached. "
                    f"Partial findings: {state.research_notes or 'none collected'}."
                )
            return ROUTE_DONE

        if state.research_notes is None:
            return ROUTE_RESEARCHER
        if state.analysis_notes is None:
            return ROUTE_ANALYST
        if state.final_answer is None:
            return ROUTE_WRITER
        return ROUTE_DONE
