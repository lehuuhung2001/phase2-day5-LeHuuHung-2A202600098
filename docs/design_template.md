# Design — Multi-Agent Research System

## Problem

Người dùng đặt câu hỏi nghiên cứu dài (ví dụ: "GraphRAG state-of-the-art là gì?").
Hệ thống cần: (1) tìm nguồn tài liệu liên quan, (2) phân tích và trích xuất insights,
(3) tổng hợp câu trả lời ~500 từ có citations.
Một single-agent call không đủ chất lượng vì cần phân tách rõ "tìm kiếm", "phân tích" và "viết".

## Why multi-agent?

Single-agent bị giới hạn vì:
- Một LLM call không thể vừa search web, vừa phân tích độ tin cậy nguồn, vừa viết có citations.
- Khó debug khi fail — không biết bước nào sai.
- Không thể tái sử dụng từng bước (ví dụ chỉ chạy lại Analyst nếu research đã xong).

Multi-agent giải quyết bằng cách tách trách nhiệm rõ ràng, mỗi agent có input/output cụ thể,
Supervisor điều phối và có thể retry từng bước.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing + guardrails | `ResearchState` toàn bộ | `route_history` updated | Max iterations → force done |
| Researcher | Search sources + summarise | `request.query`, `request.max_sources` | `state.sources`, `state.research_notes` | Search fails → mock fallback; LLM fails → AgentExecutionError |
| Analyst | Extract claims + grade evidence | `state.research_notes` | `state.analysis_notes` | LLM fail → skip, mark "[Analysis skipped]" |
| Writer | Synthesise final answer | `research_notes`, `analysis_notes`, `sources` | `state.final_answer` | LLM fail → copy research_notes as fallback |
| Critic (bonus) | Fact-check + quality score | `state.final_answer`, `state.sources` | `quality_score` in metadata | Skip silently if no final_answer |

## Shared state

`ResearchState` (Pydantic model) — single source of truth:

| Field | Type | Lý do cần |
|---|---|---|
| `request` | `ResearchQuery` | Query + params không thay đổi |
| `iteration` | `int` | Guardrail: đếm số vòng lặp |
| `route_history` | `list[str]` | Audit trail; routing dùng `[-1]` |
| `sources` | `list[SourceDocument]` | Researcher → Analyst → Writer đều cần |
| `research_notes` | `str\|None` | Output Researcher, input Analyst + Writer |
| `analysis_notes` | `str\|None` | Output Analyst, input Writer |
| `final_answer` | `str\|None` | Output Writer; `None` = workflow chưa xong |
| `agent_results` | `list[AgentResult]` | Token/cost tracking + debug |
| `trace` | `list[dict]` | Event log cho observability |
| `errors` | `list[str]` | Lưu lỗi không crash workflow |

## Routing policy

```
START
  │
  ▼
Supervisor
  ├─ research_notes is None  ──► Researcher ──► (back to Supervisor)
  ├─ analysis_notes is None  ──► Analyst    ──► (back to Supervisor)
  ├─ final_answer is None    ──► Writer     ──► (back to Supervisor)
  └─ all filled / max_iter   ──► END
```

Deterministic policy (no LLM needed) — tiết kiệm cost, dễ test.

## Guardrails

- **Max iterations:** `MAX_ITERATIONS=6` (default). Supervisor check mỗi vòng.
  Nếu hit: force `route="done"`, ghi partial answer.
- **Timeout:** `TIMEOUT_SECONDS=60`. Workflow chạy trong thread riêng.
  Nếu timeout: thread bị bỏ, partial state trả về với error message.
- **Retry:** LLM calls được retry 3 lần với exponential backoff (`tenacity`).
- **Agent fallback:**
  - Researcher fail → `research_notes = "[Research failed]"`, tiếp tục
  - Analyst fail → `analysis_notes = "[Analysis skipped]"`, tiếp tục
  - Writer fail → `final_answer = research_notes`, tiếp tục
- **Mock fallback:** Khi không có API key, LLMClient và SearchClient tự động dùng mock.
  Workflow chạy end-to-end không crash.

## Benchmark plan

**Queries thử nghiệm:**
1. "Research GraphRAG state-of-the-art and write a 500-word summary"
2. "Explain the trade-offs of multi-agent vs single-agent AI systems"
3. "What are the best practices for LLM observability in production?"

**Metrics:**
| Metric | Cách đo | Expected (multi > single?) |
|---|---|---|
| Latency | wall-clock seconds | Multi slower (+1-3x) |
| Cost | token count × price/1K | Multi higher (~3x agents) |
| Quality | CriticAgent score 0-10 | Multi higher (+2-3 pts) |
| Citation coverage | citations / sources | Multi higher |
| Error rate | errors / total runs | Both ~0 với mock |

**Expected outcome:** Multi-agent có quality cao hơn (~7-8 vs ~5-6) với chi phí
và latency cao hơn 2-4x. Trade-off phù hợp với tasks nghiên cứu phức tạp.
