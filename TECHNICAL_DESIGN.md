# Technical Design: Faithfulness Metric + Generator Hardening

> **Sprint goal**: Close the "generation-quality measurement gap" identified from DoTA-RAG paper review,
> and harden prompts against answer-truncation faithfulness collapse.
>
> Author: Lead Engineer (AI-assisted SDLC)  
> Date: 2026-04

---

## 1. Problem Statement

### 1.1 Gap: Retrieval metrics ≠ Generation quality

Current `/api/evaluate` only measures **retrieval** quality (Recall@K, NDCG@K).  
These answer "did we find the right document?" — not "is the generated answer grounded in what we found?"

The DoTA-RAG paper demonstrates this gap empirically: their system scored 0.702 Faithfulness internally
but dropped to 0.336 after a 300-word output truncation — an issue invisible to Recall@K metrics.

### 1.2 Gap: Prompt truncation risk

`GEN_PROMPT` and `GEN_ANALYSIS_PROMPT` currently instruct the LLM to synthesize multiple passages
but do not require the key answer to appear early in the output. If a client or proxy truncates long
responses, the direct answer may be cut, lowering perceived faithfulness.

---

## 2. Solution Design

### 2.1 Feature A — Faithfulness LLM-as-a-Judge (`src/core/faithfulness.py`)

**Approach**: LLM judge with structured output (3-level verdict), normalized to [0.0, 1.0].

| Verdict    | Score | Meaning |
|------------|-------|---------|
| `"full"`   | 1.0   | Every material claim is directly supported by at least one passage |
| `"partial"`| 0.5   | Most claims are supported; minor extrapolations present |
| `"none"`   | 0.0   | Answer contains significant unsupported claims / hallucinations |

**Why 3-level vs continuous?**  
A finer scale (e.g. 0–10) requires the LLM to reason about magnitude precisely — calibration is poor
in practice without fine-tuning. A 3-level verdict matches judge LLM capabilities and is reproducible.
DoTA-RAG used a continuous -1 to 1 scale and found high variance on borderline cases.

**Dependency injection**: `judge_faithfulness(answer, passages, llm)` — caller provides the LLM.
This keeps `src/core/faithfulness.py` free from singleton LLM initialization and makes it testable
with a simple mock.

**Layer compliance**: `src/core/` is permitted to use third-party libraries (langchain, pydantic)
directly; it must not import `src.rag_core`, `src.hybrid_rag`, or `src.api` (enforced by lint-imports).
The `llm` parameter is typed as `Any` to avoid importing a concrete VertexAI type into core.

### 2.2 Feature B — Faithfulness integration into `run_evaluation`

`run_evaluation(query, expected_substring, use_hybrid, compute_faithfulness)`:
- When `compute_faithfulness=True`: run the generation step (reuse existing pipeline), then call
  `judge_faithfulness(answer, ranked_texts, llm)`.
- Result dict gains: `faithfulness_score` (float), `faithfulness_verdict` (str), `faithfulness_reasoning` (str).

Schema change: `EvalRequest.compute_faithfulness: bool = False` (opt-in, default off to preserve latency).
`EvalBatchRequest` propagates this flag to all cases; `aggregate_mean` includes the new key.

### 2.3 Feature C — Generator prompt: conclusion-first instruction

Add a single sentence to both `GEN_PROMPT` (EN) and `GEN_ANALYSIS_PROMPT` (ZH):
- EN: `"Lead with the direct answer in 1–2 sentences before elaborating."`
- ZH: `"请先用1–2句话直接回答问题，再展开引用与分析。"`

**Risk**: None — additive instruction, no schema change, no new external call.  
**Benefit**: Prevents faithfulness collapse under response truncation (DoTA-RAG lesson).

---

## 3. Architecture Compliance

| Rule | Check |
|------|-------|
| `src/core/` must not import `src.api`, `src.rag_core` | `faithfulness.py` uses only pydantic + `Any` typed llm param |
| `src/infrastructure/` must not import application layers | No infrastructure changes |
| New schema fields are backward-compatible | `compute_faithfulness` defaults to `False` |
| Tests must not import `rag_core` / Vertex (CI rule) | `test_faithfulness.py` mocks the llm entirely |

---

## 4. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Judge LLM adds 2–5s latency to eval | High | Opt-in flag; batch eval notes latency impact |
| Structured output schema mismatch (Vertex) | Low | Fallback to `{"score": 0.0, "verdict": "none", "reasoning": "judge_error"}` |
| Prompt change degrades answer quality | Very low | Purely additive instruction; covered by regression tests |
