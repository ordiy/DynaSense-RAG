# Implementation Plan

> Derived from `TECHNICAL_DESIGN.md` — Sprint 2026-04

## Task List

### P0 — Core: Faithfulness LLM-as-a-Judge
- [ ] **T1** Create `src/core/faithfulness.py`
  - `FaithfulnessVerdict` Pydantic model (`verdict: Literal["full","partial","none"]`, `reasoning: str`)
  - `judge_faithfulness(answer, passages, llm) -> dict` with try/except fallback
  - `verdict_to_score(verdict) -> float`  (full=1.0, partial=0.5, none=0.0)

### P0 — Eval integration
- [ ] **T2** Update `src/api/schemas.py`
  - `EvalRequest`: add `compute_faithfulness: bool = False`
  - `EvalBatchRequest`: propagate `compute_faithfulness`

- [ ] **T3** Update `src/recall_metrics.py`
  - `aggregate_mean()`: include `faithfulness_score` key if present

- [ ] **T4** Update `src/rag_core.py` — `run_evaluation()`
  - Accept `compute_faithfulness=False` param
  - When True: invoke generation, call `judge_faithfulness`, add to return dict

- [ ] **T5** Update `src/api/routers/eval.py`
  - Pass `compute_faithfulness` flag from request to `run_evaluation`

### P1 — Prompt hardening
- [ ] **T6** Update `GEN_PROMPT` in `src/rag_core.py`
  - Prepend: "Lead with the direct answer in 1–2 sentences before elaborating."

- [ ] **T7** Update `GEN_ANALYSIS_PROMPT` in `src/rag_core.py`
  - Prepend: "请先用1–2句话直接回答问题，再展开引用与分析。"

### P2 — Tests
- [ ] **T8** Create `tests/test_faithfulness.py`
  - `test_verdict_to_score`: full/partial/none mapping
  - `test_judge_faithfulness_mocked_full`: LLM returns full → score=1.0
  - `test_judge_faithfulness_mocked_none`: LLM returns none → score=0.0
  - `test_judge_faithfulness_error_fallback`: LLM raises → returns safe fallback
  - `test_eval_request_faithfulness_flag`: schema accepts new field

- [ ] **T9** Run regression: `make test` — all existing tests must pass

### P3 — Documentation
- [ ] **T10** Update `docs/bitter_lesson_roadmap.md` — mark Faithfulness metric as implemented
- [ ] **T11** Update `docs/testing.md` — document new `compute_faithfulness` flag

## Dependency Graph

```
T1 → T4 → T9
T2 → T4, T5
T3 → T9
T6, T7 → T9
T8 → T1
T10, T11 → T9 (after passing)
```

## Success Criteria

1. `make test` passes with zero new failures
2. `POST /api/evaluate` with `compute_faithfulness=true` returns `faithfulness_score` (0.0–1.0)
3. `POST /api/evaluate/batch` aggregates mean faithfulness across cases
4. Both generator prompts contain conclusion-first instruction
5. `test_faithfulness.py` — all 5 tests pass without live LLM
