# Lab 20: Multi-Agent Research System

Hệ thống nghiên cứu multi-agent gồm **Supervisor + Researcher + Analyst + Writer + Critic**, benchmark với single-agent baseline.

## Architecture

```text
User Query
   |
   v
Supervisor (deterministic router)
   |--iter 1--> Researcher  -> sources + research_notes
   |--iter 2--> Analyst     -> analysis_notes
   |--iter 3--> Writer      -> final_answer
   |--iter 4--> done
   |
   v
Critic (quality score 0-10)
   |
   v
Trace (LangSmith) + Benchmark Report
```

## Kết quả benchmark

| Run | Latency (s) | Cost (USD) | Quality (0-10) | Notes |
|---|---:|---:|---:|---|
| single-agent-baseline | 11.34 | $0.00044 | n/a | 1 LLM call |
| multi-agent-workflow | 28.03 | $0.00151 | 6.0 | citations=100%, 4 iterations |

**Nhận xét:** Multi-agent chậm hơn ~2.5x và tốn ~3.4x cost, nhưng output có citations đầy đủ và được Critic đánh giá 6/10.

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Supervisor, Researcher, Analyst, Writer, Critic
│   ├── core/                # Config, ResearchState, schemas, errors
│   ├── graph/               # LangGraph workflow + simple-loop fallback
│   ├── services/            # LLMClient (OpenAI), SearchClient (Tavily/mock)
│   ├── evaluation/          # Benchmark comparison + markdown report
│   ├── observability/       # Tracing (local JSON + LangSmith)
│   └── cli.py               # CLI: baseline / multi-agent / benchmark
├── docs/                    # Design doc, rubric
├── tests/                   # 12 unit tests (all pass)
├── notebooks/demo.ipynb     # Demo notebook
├── reports/
│   ├── benchmark_report.md  # Live benchmark results
│   └── traces/              # JSON trace files
└── screenshots/             # LangSmith trace screenshots
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[llm,dev]" -i https://mirrors.aliyun.com/pypi/simple/
cp .env.example .env
```

### 2. Cấu hình API keys

```bash
OPENAI_API_KEY=...       # bắt buộc
LANGSMITH_API_KEY=...    # optional — bật LangSmith dashboard
TAVILY_API_KEY=...       # optional — bật real web search
```

Nếu không có key, hệ thống tự dùng **mock mode** — workflow vẫn chạy end-to-end.

### 3. Chạy tests

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -v
# 12 passed
```

### 4. Chạy single-agent baseline

```bash
PYTHONIOENCODING=utf-8 python -m multi_agent_research_lab.cli baseline \
  --query "Explain GraphRAG"
```

### 5. Chạy multi-agent workflow

```bash
PYTHONIOENCODING=utf-8 python -m multi_agent_research_lab.cli multi-agent \
  --query "Explain GraphRAG"
```

### 6. Chạy benchmark (so sánh 2 cách)

```bash
PYTHONIOENCODING=utf-8 python -m multi_agent_research_lab.cli benchmark \
  --query "Explain GraphRAG"
# sinh ra reports/benchmark_report.md
```

## Guardrails

| Guardrail | Cơ chế |
|---|---|
| Max iterations | Supervisor dừng khi `iteration >= MAX_ITERATIONS` (default 6) |
| Timeout | Workflow chạy trong daemon thread, `join(timeout=60s)` |
| Retry LLM | tenacity 3 lần, exponential backoff |
| Agent fallback | Researcher lỗi → `research_notes="[failed]"`, Analyst lỗi → skip, Writer lỗi → dùng research_notes |
| Mock fallback | Không có API key → mock tự động bật |

## Failure modes

| Scenario | Behaviour | Mitigation |
|---|---|---|
| Max iterations hit | Supervisor force `done`, trả partial answer | Tăng `MAX_ITERATIONS` |
| LLM API lỗi (3 retries) | `AgentExecutionError` → agent fallback | Mock LLM nếu không có key |
| Researcher fails | `research_notes = "[Research failed]"`, tiếp tục | Mock search fallback |
| Writer fails | `final_answer = research_notes` | Luôn có output |
| Timeout | Thread join timeout, partial state trả về | Tăng `TIMEOUT_SECONDS` |

## Trace

LangSmith trace screenshots: [`screenshots/`](screenshots/)

JSON trace local: [`reports/traces/latest.json`](reports/traces/latest.json)

## References

- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- OpenAI Agents SDK — https://developers.openai.com/api/docs/guides/agents/orchestration
- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
