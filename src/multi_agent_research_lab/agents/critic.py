"""Critic agent — optional fact-check and quality review."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and quality-review agent.

    Checks final_answer for unsupported claims, citation coverage, and
    produces a quality_score (0-10) stored in the last AgentResult metadata.
    """

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic.run") as span:
            if not state.final_answer:
                logger.warning("Critic skipped: no final_answer to review")
                return state

            sources_list = "\n".join(
                f"[{i + 1}] {s.title}" for i, s in enumerate(state.sources)
            ) or "(none)"

            system_prompt = (
                "You are a critic agent performing fact-checking and quality review.\n"
                "For the given final answer:\n"
                "1. List any UNSUPPORTED claims (prefix each with 'UNSUPPORTED:').\n"
                "2. State citation coverage: X out of Y key claims have a source citation.\n"
                "3. Give a quality_score between 0 and 10 on the last line as: "
                "'quality_score: <number>'\n"
                "Be concise and specific."
            )
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Final Answer:\n{state.final_answer}\n\n"
                f"Sources available:\n{sources_list}\n\n"
                "Provide your critique."
            )

            response = self._llm.complete(system_prompt, user_prompt)

            # Parse quality score from LLM output
            quality_score = _parse_quality_score(response.content)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "quality_score": quality_score,
                    },
                )
            )
            state.add_trace_event(
                "critic.complete",
                {"quality_score": quality_score, "tokens_out": response.output_tokens},
            )
            span["quality_score"] = quality_score
            logger.info("Critic: review complete, quality_score=%.1f", quality_score or 0)
        return state


def _parse_quality_score(text: str) -> float | None:
    match = re.search(r"quality_score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        except ValueError:
            pass
    return None
