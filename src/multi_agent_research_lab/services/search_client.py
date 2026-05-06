"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client — Tavily with mock fallback."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        settings = get_settings()
        if settings.tavily_api_key:
            try:
                return self._tavily_search(query, max_results, settings.tavily_api_key)
            except Exception as exc:
                logger.warning("Tavily search failed (%s) — falling back to mock", exc)
        else:
            logger.info("TAVILY_API_KEY not set — using mock search")
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=api_key)
        result = client.search(query, max_results=max_results)
        docs = []
        for r in result.get("results", []):
            docs.append(
                SourceDocument(
                    title=r.get("title", "Untitled"),
                    url=r.get("url"),
                    snippet=(r.get("content") or "")[:500],
                    metadata={"score": r.get("score")},
                )
            )
        logger.info("Tavily returned %d results for query '%s'", len(docs), query[:60])
        return docs[:max_results]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        topic = query[:60]
        docs = [
            SourceDocument(
                title=f"Survey: State-of-the-Art in {topic}",
                url="https://arxiv.org/abs/mock-2024-001",
                snippet=(
                    f"This comprehensive survey reviews recent advances in {topic}. "
                    "We identify key contributions across 150+ papers and highlight open challenges "
                    "including scalability, evaluation methodology, and reproducibility."
                ),
                metadata={"source": "mock", "year": 2024},
            ),
            SourceDocument(
                title=f"Practical Guide to {topic}",
                url="https://example.com/practical-guide",
                snippet=(
                    f"A practitioner's guide to deploying {topic} in production systems. "
                    "Covers architecture decisions, performance trade-offs, and monitoring strategies. "
                    "Includes case studies from three large-scale deployments."
                ),
                metadata={"source": "mock", "year": 2024},
            ),
            SourceDocument(
                title=f"Benchmarking {topic}: A Comparative Analysis",
                url="https://proceedings.example.com/mock-2024-bench",
                snippet=(
                    f"We benchmark {topic} approaches across 10 standard datasets. "
                    "Results show 30-40% performance improvement over baseline methods. "
                    "Ablation studies isolate the contribution of each component."
                ),
                metadata={"source": "mock", "year": 2024},
            ),
            SourceDocument(
                title=f"Failure Modes in {topic}: Lessons Learned",
                url="https://blog.example.com/failure-modes",
                snippet=(
                    f"Analysis of common failure modes when deploying {topic} at scale. "
                    "Edge cases, data distribution shifts, and latency spikes are the top causes. "
                    "Mitigation strategies and monitoring dashboards are discussed."
                ),
                metadata={"source": "mock", "year": 2023},
            ),
            SourceDocument(
                title=f"Future Directions for {topic}",
                url="https://arxiv.org/abs/mock-2024-future",
                snippet=(
                    f"Research roadmap for {topic} over the next 3-5 years. "
                    "Key open problems: multimodal integration, real-time inference, "
                    "and robust evaluation benchmarks."
                ),
                metadata={"source": "mock", "year": 2024},
            ),
        ]
        return docs[:max_results]
