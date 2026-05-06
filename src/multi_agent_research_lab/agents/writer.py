"""Writer agent — synthesises final answer with citations."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with citations."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("writer.run") as span:
            # Build source reference block
            sources_block = ""
            if state.sources:
                refs = "\n".join(
                    f"[{i + 1}] {s.title} — {s.url or 'N/A'}"
                    for i, s in enumerate(state.sources)
                )
                sources_block = f"\n\nAvailable sources for citation:\n{refs}"

            system_prompt = (
                f"You are a writer agent creating a clear, well-structured response for "
                f"{state.request.audience}. "
                "Synthesise the research notes and analysis into a compelling ~500-word final answer. "
                "Include inline citations [1], [2] where appropriate. "
                "Structure: brief intro, key points, conclusion."
            )
            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Research Notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes}"
                f"{sources_block}\n\n"
                "Write a comprehensive, well-cited final answer."
            )

            response = self._llm.complete(system_prompt, user_prompt)
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "writer.complete",
                {"tokens_out": response.output_tokens},
            )
            span["tokens_out"] = response.output_tokens
            logger.info("Writer: final answer complete (%d output tokens)", response.output_tokens or 0)
        return state
