# M2 — Orchestrator + Gradio (base / RAG modes) — Design Spec

- **Date:** 2026-05-03
- **Author:** continuation of `2026-05-02-legal-privatellm-mvp-design.md` (frozen) — this doc focuses just the Serving slice that Plan 2 ships.
- **Status:** Ready for plan
- **Scope:** Implements the M2 milestone from the MVP design doc (§4.4 Serving + §10 Milestones — M2). Does NOT change retrieval, ingestion, or evaluation surfaces.

## 0. Why this doc exists

The MVP design (2026-05-02) covers the whole 5-milestone PoC at architectural granularity. Plan 1 has shipped M0+M1 (RAG retrieval is live; see `m2-readiness.md`). This document is the **plan-input refinement** for M2 only — file-by-file responsibilities, public APIs, what is intentionally deferred to M3, and the test contract. Anything not stated here defers to the MVP design.

## 1. Goal

Wire the existing retriever (M1) to the Qwen2.5-3B 4-bit base model (M0) behind a single `Orchestrator.generate(query, mode)` entry point, and expose it via a minimal Gradio UI that lets a user compare `base` and `rag` answers for the same query side-by-side. This is the slice that lets a human start eyeballing RAG quality on real civil-law questions before any QLoRA work begins in M3.

**Out of M2 (deferred to M3 / M4):**
- `qlora` and `rag_qlora` modes — no adapter exists yet.
- Evaluation runner / judge / score aggregation — `eval_set.jsonl` is also empty. Citation extraction belongs to M2 because the UI uses it; the evaluation pipeline that consumes it does not.
- Adapter attach/detach machinery in `model_loader.py` — designed-for but not implemented in M2.

## 2. Components

### 2.1 `src/serving/model_loader.py`

**Responsibility.** Single point of truth for "load the base 4-bit Qwen2.5-3B-Instruct on this host." Process-singleton — once loaded, subsequent calls return the same `(model, tokenizer)` pair.

**Public API.**

```python
def get_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]: ...
```

**Implementation note.** Reuse the same Unsloth-first / transformers+bitsandbytes-fallback dance already validated in `scripts/verify_unsloth.py` and `scripts/verify_combined_vram.py`. Factor that fallback out so all three call sites share one path. The loader reads `cfg["model"]["base_id"]` and `cfg["model"]["max_seq_len"]` from `default.yaml`.

**Adapter design hook (M3-ready, not implemented now).** The function should be structured so an `attach_adapter(name)` / `detach_adapter()` extension lands later without rewriting the singleton. Keep the singleton's state explicit (a dict / module-level dataclass), not implicit (lambda closures).

### 2.2 `src/serving/prompt_builder.py`

**Responsibility.** Pure logic — no model, no GPU. Turn `(query, mode, retrieved_chunks)` into a chat-template-ready message list. TDD-friendly.

**Public API.**

```python
def build_messages(
    query: str,
    mode: Literal["base", "rag"],
    chunks: list[Chunk] | None = None,
    *,
    max_context_tokens: int = 1500,
) -> list[dict]: ...
```

**Behavior.**
- `base` mode: ignore `chunks`; emit a single user turn with the system instruction prepended (or as a `system` role, per Qwen's chat template).
- `rag` mode: build the `[참고자료]` block from `chunks` in the order given; format each chunk as `[조문번호 또는 판례번호] {text}` (statute_no preferred, fall back to case_no, fall back to chunk.id).
- **Token cap enforcement.** Tokenize the assembled context (using the loaded tokenizer — passed in or shared via `get_base_model()`); if the context exceeds `max_context_tokens`, drop chunks from the end (the lowest-ranked) until it fits. Log a warning when truncation happens; never silently drop the user's query.
- The exact system text comes from `config/prompts/rag.txt` and `config/prompts/no_rag.txt`. Plan 2 creates these two files (the `prompts/` directory is currently empty).

**System prompt (Korean, civil-law).** Follow §5 "RAG Prompt Template" of the MVP design. Restated here for clarity:

```
당신은 한국 민법 전문가입니다. 아래 참고자료를 근거로 답변하고,
근거가 부족하면 "참고자료에 명시되지 않음"이라고 답하세요.
인용은 [조문번호] 또는 [판례번호] 형식으로 본문에 표시하세요.
본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

The non-RAG variant replaces `참고자료를 근거로` with `귀하의 일반 지식으로` and removes the `[참고자료]` block.

### 2.3 `src/serving/orchestrator.py`

**Responsibility.** Glue. Wraps `Retriever` + `model_loader` + `prompt_builder` behind the single MVP-spec'd entry point, plus citation extraction on the answer.

**Public API.**

```python
@dataclass
class Response:
    answer: str
    mode: Literal["base", "rag"]
    citations: list[str]              # whatever citation_checker.extract returns
    retrieved: list[Chunk]            # empty in base mode
    latency_ms: int

class Orchestrator:
    @classmethod
    def open(cls) -> "Orchestrator": ...
    def generate(self, query: str, mode: Literal["base", "rag"]) -> Response: ...
```

**Behavior.**
- `base`: skip retrieval; build messages; generate; extract citations from the answer (purely informational — base mode rarely emits valid citations); return.
- `rag`: retrieve (top-K from `cfg["retrieval"]["top_k"]`, default 5); build messages; generate; extract citations; return.
- Generation hyperparameters: `max_new_tokens=512`, `do_sample=False`, `temperature` / `top_p` / `top_k` cleared (avoid the noisy warnings from `verify_combined_vram`).
- `Orchestrator.open()` is the singleton-aware constructor. Reuses `Retriever.open()` and `get_base_model()`.

**Singleton & threading.** Gradio runs handlers on a worker pool. The model + tokenizer + retriever are read-only post-init, but `model.generate(...)` is not safe to call concurrently on the same model instance. Plan 2 serializes generation behind a `threading.Lock` inside the orchestrator. Document the contention; it's acceptable for a 1-user PoC.

### 2.4 `src/eval/citation_checker.py`

**Responsibility.** Regex-extract citations from a model answer + (optionally) verify them against a corpus index. Plan 2 only needs extraction; corpus verification can be a stub that returns `True` for any well-formed citation, and the M4 plan will fill in the real lookup.

**Public API.**

```python
@dataclass
class Citation:
    raw: str               # the matched text, e.g. "제618조"
    kind: Literal["statute", "case"]
    normalized: str        # e.g. "618" for statute, "2019다12345" for case
    found_in_corpus: bool  # M2: stub-true; M4: real lookup

def extract_citations(text: str) -> list[Citation]: ...
def verify_citations(citations: list[Citation], corpus: CorpusIndex | None = None) -> list[Citation]: ...
```

**Regex patterns.** Two from the MVP design §4.5:
- Statute: `\[?민법\s*제\s*(\d+)\s*조(?:의\s*\d+)?\]?` and the bare `제\s*\d+\s*조` form (without the `민법` prefix) — both should be captured. The latter is common in answers that already established the law.
- Case: `대법원\s*\d{4}[가-힣]+\d+` (the `2019다12345` style) and bracketed variants.

**Why this lives in `src/eval/` and not `src/serving/`:** the MVP design module graph (§4.7) puts the citation checker under eval. The Gradio UI imports it through `from src.eval.citation_checker import extract_citations` — this is a cross-cutting concern but matches the spec's dependency direction (`serving → eval` is an inverted edge; M4's runner will own the corpus index, M2 just consumes the extractor).

### 2.5 `src/serving/app_gradio.py`

**Responsibility.** Single-file Gradio app. Two-column layout: same query rendered against `base` and `rag` modes. Each column shows: answer, extracted citations (with green/red badges for `found_in_corpus`), and (RAG column only) the retrieved chunks with their statute/case ids.

**Inputs.** A textarea (query), a "Run" button, an optional "k" slider (default 5).

**Wiring.** On click → `Orchestrator.generate(q, "base")` and `Orchestrator.generate(q, "rag")` (sequentially under the lock; Gradio's queue handles concurrent users by serializing). Display latency_ms and the chunk previews.

**Make target.** `make serve` already maps to `python -m src.serving.app_gradio` — wire it to spin the app on `http://127.0.0.1:7860` with `share=False`.

## 3. Configuration

`config/default.yaml` already has `model:` and `retrieval:`; M2 adds:

```yaml
serving:
  max_new_tokens: 512
  max_context_tokens: 1500
  do_sample: false
```

Plan 2 also creates the empty `config/prompts/` directory anchor with `rag.txt` and `no_rag.txt` containing the system prompts above.

## 4. Test contract (TDD-applicable surfaces in bold)

| File | Test | Type |
|---|---|---|
| **`prompt_builder.py`** | base vs rag template diverge; chunk-id formatting; token cap drops lowest-ranked | unit (TDD) |
| **`citation_checker.py`** | statute regex (with/without 민법 prefix); case regex; multi-citation extraction; no false positive on prose | unit (TDD) |
| `model_loader.py` | singleton: two calls return the same id; subprocess marker test for fallback path | integration (slow, gated) |
| `orchestrator.py` (mocked) | mode dispatch: base → no retriever call; rag → retriever called with the query; citation list flows through | integration (mocks `Retriever` + `model.generate`) |
| `app_gradio.py` | not unit-tested; manual smoke per the M2 acceptance criteria below |

Slow integration: one test that loads the real model + retriever via `Orchestrator.open()`, runs both modes for the lease query, asserts the rag answer's `retrieved` list is non-empty and `latency_ms` is recorded. Gated behind `-m slow` like the existing E2E.

## 5. Acceptance criteria (= Plan 2 "done")

1. `make serve` opens the Gradio UI; entering "임대차의 정의를 설명해 주세요" returns answers in both columns within ~10 s on the target host.
2. The RAG column shows at least one retrieved chunk whose `statute_no == "618"`.
3. The citation-checker view shows the model's `[제618조]`-shaped citations as extracted (corpus verification stub-true is fine for M2).
4. Both columns display non-empty Korean text.
5. `pytest -m "not slow"` stays green; the new prompt-builder + citation-checker unit tests are part of it. Slow suite gains one orchestrator integration test.
6. `scripts/verify_combined_vram.py` continues to PASS (peak < 5.5 GB) — running the orchestrator for a full RAG turn must not regress the headroom.

## 6. Risk & mitigation

| Risk | Mitigation |
|---|---|
| Gradio worker thread re-enters `model.generate` concurrently | `threading.Lock` in `Orchestrator.generate`; documented contention. |
| `prompt_builder` token cap dropping the user's only relevant chunk | Token cap drops from the END (lowest-ranked) only; warn at INFO. The 1500-token cap with k=5 is empirically safe for civil-code chunks (avg ~400 tokens/chunk per `chunk.py` ceilings). |
| Citation regex over-matches body prose like "제3장" | Anchor patterns to the `\d+조` core; explicit unit tests for false-positive prose. |
| `make serve` blocks the test runner if someone runs it during CI | Document; CI doesn't run `make serve`. |
| Latency target wrong on a cold model load | Orchestrator's first call eagerly warms via `Orchestrator.open()`; the UI greets the user only after the singleton is initialized. |

## 7. Out of M2 (re-stated, intentional)

- QLoRA adapter attach/detach (M3).
- Real `corpus.find()` for citation verification (M4).
- `eval/runner.py`, `eval/judge.py`, `eval/aggregate.py` (M4).
- Multi-turn dialog / session memory (out of MVP entirely).
- Token-by-token streaming in the UI (defer; full-completion is fine for a PoC).

## 8. Suggested first three tasks (Plan 2)

These were already drafted in `m2-readiness.md` §"Suggested Plan 2 first-three tasks" but the bootstrap-polish item is now done. Updated:

1. **`src/serving/__init__.py` + `model_loader.py`** — extract the Unsloth/transformers fallback into a shared helper; expose `get_base_model()`. One slow integration test that asserts the singleton.
2. **`src/serving/prompt_builder.py` + tests (TDD)** — the smallest pure-logic piece. Drives the prompt files in `config/prompts/`.
3. **`src/eval/__init__.py` + `citation_checker.py` + tests (TDD)** — pure regex; needed by orchestrator and the UI.

After those three, the orchestrator and Gradio UI are mechanical wiring.
