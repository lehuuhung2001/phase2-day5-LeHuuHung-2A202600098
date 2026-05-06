"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Cost per 1K tokens (USD) — update as pricing changes
_COST_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class LLMClient:
    """Provider-agnostic LLM client — OpenAI with mock fallback."""

    _client: object = field(default=None, init=False, repr=False)

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        settings = get_settings()
        if settings.openai_api_key:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                self._client = OpenAI(api_key=settings.openai_api_key)
                logger.info("LLMClient: using OpenAI (%s)", settings.openai_model)
            except ImportError:
                logger.warning("openai package not installed — falling back to mock")
                self._client = "mock"
        else:
            logger.warning("OPENAI_API_KEY not set — using mock LLM")
            self._client = "mock"
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, token tracking, and cost estimate."""
        client = self._get_client()
        if client == "mock":
            return self._mock_complete(system_prompt, user_prompt)
        return self._openai_complete(client, system_prompt, user_prompt)

    def _openai_complete(self, client: object, system_prompt: str, user_prompt: str) -> LLMResponse:
        settings = get_settings()
        try:
            from tenacity import (  # type: ignore[import-untyped]
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                wait_exponential,
            )
            from openai import APIError, RateLimitError  # type: ignore[import-untyped]

            @retry(
                retry=retry_if_exception_type((APIError, RateLimitError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            )
            def _call() -> LLMResponse:
                response = client.chat.completions.create(  # type: ignore[union-attr]
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    timeout=settings.timeout_seconds,
                )
                text = response.choices[0].message.content or ""
                in_tok = response.usage.prompt_tokens if response.usage else None
                out_tok = response.usage.completion_tokens if response.usage else None
                cost = _calc_cost(settings.openai_model, in_tok, out_tok)
                logger.debug(
                    "LLM complete: %d in / %d out tokens, $%.5f",
                    in_tok or 0,
                    out_tok or 0,
                    cost or 0,
                )
                return LLMResponse(
                    content=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                )

            return _call()
        except Exception as exc:
            raise AgentExecutionError(f"LLM call failed after retries: {exc}") from exc

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return plausible mock response for offline testing."""
        sp_lower = system_prompt.lower()
        query_snippet = user_prompt[:120].replace("\n", " ")

        if "researcher" in sp_lower:
            content = (
                "[MOCK Research Notes]\n\n"
                f"Query: {query_snippet}\n\n"
                "Key findings:\n"
                "1. [1] The topic has seen significant advances in recent years, with multiple"
                " production deployments reported.\n"
                "2. [2] Performance benchmarks show 30-40% improvement over naive baselines.\n"
                "3. [3] Open challenges remain in scalability and evaluation methodology.\n\n"
                "Sources:\n"
                "[1] Survey Paper — https://arxiv.org/abs/mock-001\n"
                "[2] Benchmark Study — https://proceedings.example.com/mock-002\n"
                "[3] Practical Guide — https://example.com/guide"
            )
        elif "analyst" in sp_lower:
            content = (
                "[MOCK Analysis]\n\n"
                "Key claims:\n"
                "- STRONG: Performance gains validated by [1] and [2].\n"
                "- MODERATE: Scalability claims lack large-scale empirical data.\n"
                "- WEAK: Cost estimates vary widely across implementations.\n\n"
                "Gaps: Limited comparison across diverse domains.\n"
                "Conclusion: The evidence supports adoption with careful benchmarking."
            )
        elif "writer" in sp_lower:
            content = (
                "[MOCK Final Answer]\n\n"
                f"Based on research into '{query_snippet[:60]}...', here is a comprehensive summary:\n\n"
                "**Overview**\nThe technology represents a meaningful advance with documented "
                "production use cases [1]. Benchmarks confirm 30-40% gains on standard metrics [2].\n\n"
                "**Key Points**\n"
                "1. Architecture improvements enable better performance at scale.\n"
                "2. Implementation requires careful configuration but practical guides exist [3].\n"
                "3. Ongoing research addresses remaining scalability challenges.\n\n"
                "**Conclusion**\nAdoption is recommended with proper evaluation of trade-offs.\n\n"
                "References:\n[1] Survey Paper  [2] Benchmark Study  [3] Practical Guide"
            )
        elif "critic" in sp_lower or "fact" in sp_lower:
            content = (
                "[MOCK Critique]\n\n"
                "Citation coverage: 3/3 key claims have sources (100%).\n"
                "Unsupported: None detected.\n"
                "Quality score: 7/10 — solid structure, minor depth gaps.\n"
                "Suggestion: Add quantitative comparisons."
            )
        else:
            content = f"[MOCK Response]\n\n{query_snippet}..."

        return LLMResponse(content=content, input_tokens=200, output_tokens=250, cost_usd=0.0002)


def _calc_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    rates = _COST_TABLE.get(model, _COST_TABLE["gpt-4o-mini"])
    return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
