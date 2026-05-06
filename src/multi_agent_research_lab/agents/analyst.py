"""Analyst agent — extracts structured insights from research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights with evidence grading."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("analyst.run") as span:
            system_prompt = (
                "You are an analyst agent. Given research notes, produce a structured analysis:\n"
                "1. Key claims — label each as STRONG / MODERATE / WEAK based on evidence.\n"
                "2. Conflicting viewpoints or knowledge gaps.\n"
                "3. Practical implications for the target audience.\n"
                "Be critical and concise. Flag unsupported assertions explicitly."
            )
            user_prompt = (
                f"Research query: {state.request.query}\n"
                f"Target audience: {state.request.audience}\n\n"
                f"Research Notes:\n{state.research_notes}\n\n"
                "Provide structured analysis."
            )

            response = self._llm.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst.complete",
                {"tokens_out": response.output_tokens},
            )
            span["tokens_out"] = response.output_tokens
            logger.info("Analyst: analysis complete (%d output tokens)", response.output_tokens or 0)
        return state
