"""Researcher agent — gathers sources and synthesises research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes with citations."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher.run", {"query": state.request.query[:80]}) as span:
            # 1. Search for sources
            sources = self._search.search(state.request.query, state.request.max_sources)
            state.sources = sources
            span["num_sources"] = len(sources)
            logger.info("Researcher: found %d sources", len(sources))

            # 2. Build prompt with source snippets
            snippets = "\n\n".join(
                f"[{i + 1}] {s.title}\nURL: {s.url or 'N/A'}\n{s.snippet}"
                for i, s in enumerate(sources)
            )
            system_prompt = (
                "You are a researcher agent. Your job is to synthesise the provided sources into "
                "clear, concise research notes with inline citations like [1], [2]. "
                "Focus on verified facts, key claims, and data points. "
                "Do not editorialize — stay close to what the sources say."
            )
            user_prompt = (
                f"Research query: {state.request.query}\n"
                f"Target audience: {state.request.audience}\n\n"
                f"Sources:\n{snippets}\n\n"
                "Write comprehensive research notes with citations."
            )

            # 3. Call LLM
            response = self._llm.complete(system_prompt, user_prompt)
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "num_sources": len(sources),
                    },
                )
            )
            state.add_trace_event(
                "researcher.complete",
                {"num_sources": len(sources), "tokens_out": response.output_tokens},
            )
            span["tokens_out"] = response.output_tokens
        return state
