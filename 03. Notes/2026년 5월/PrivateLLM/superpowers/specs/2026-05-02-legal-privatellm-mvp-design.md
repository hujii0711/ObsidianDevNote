# Legal PrivateLLM MVP — Design Spec

- **Date:** 2026-05-02
- **Author:** brainstormed with Claude (superpowers:brainstorming)
- **Status:** Draft for review
- **Type:** Technical validation PoC (RAG + QLoRA in Korean civil law)

---

## 1. Goal

Build a Proof-of-Concept that quantitatively validates whether **RAG and QLoRA fine-tuning improve a small open-weight LLM's performance on Korean civil law (민법) Q&A**. The deliverable is a 4-way ablation report (base / RAG / QLoRA / RAG+QLoRA) on a hand-curated evaluation set, plus a minimal Gradio UI for interactive comparison.

This is a **technical PoC**, not a product. UI is minimal; the primary output is the evaluation report.

## 2. Decisions Summary

| Area | Decision |
|---|---|
| PoC type | Technical validation — measure ablation effects |
| Legal domain | Korean civil law (민법); evaluation focused on contract law + lease (계약법 + 임대차) |
| Hardware | Local: RTX 3060 Laptop **6GB VRAM**, 32GB RAM, ~437GB free disk, Windows 11 |
| Strategy | Unsloth + 3B-class QLoRA, full local training |
| Base model | Qwen2.5-3B-Instruct (Apache 2.0) |
| Data sourcing | Public Korean legal data + synthetic Q&A via Claude API |
| RAG embedding | bge-m3 (BAAI, MIT) |
| Vector store | ChromaDB (embedded) |
| Retrieval | Hybrid: dense + BM25, fused via RRF, top-K=5 |
| Evaluation metrics | LLM-as-Judge (Claude) + Citation Accuracy |
| Eval set size | 100~150 hand-reviewed items (start at 80, expand) |
| UI | Gradio with mode toggles (RAG on/off, QLoRA on/off) |

## 3. Architecture Overview

```
[1] Ingestion              [2] Training                [3] Serving
  ─────────────              ────────────                ───────────
  법령 API ──┐                                              ┌── Gradio UI
  판례 크롤 ─┤                                              │   (ablation
  공개 QA  ──┤                                              │    toggle)
             ▼                                              │
   raw/                                                     ▼
   ├ statutes/                                       ┌─────────────┐
   ├ cases/         ┌──────────────┐                 │   Inference │
   └ qa/            │ QLoRA Train  │  adapter        │  Orchestrat │
             │      │ (Unsloth +   │ ──────────────► │  - base     │
             ▼      │  Qwen2.5-3B) │                 │  - +adapter │
   processed/       └──────────────┘                 │  - ±RAG     │
   ├ chunks.jsonl         ▲                          └──────┬──────┘
   └ qa_train.jsonl       │                                 │
             │            │ uses                            │ retrieve
             ▼            │                                 ▼
   ┌──────────────┐       │                        ┌──────────────┐
   │ Embed (bge-  │ ──────┴────────────────────────│  ChromaDB +  │
   │ m3) → Index  │                                │  BM25 hybrid │
   └──────────────┘                                └──────────────┘
                                                          ▲
                                                          │
  [4] Evaluation                                          │
  ──────────────                                          │
  eval_set.jsonl ──► run 4-way ablation ──► report.md ◄───┘
                     (no/RAG × no/QLoRA)    + scores.csv
                          │
                          └─► LLM-as-Judge (Claude/GPT-4 API)
                              + Citation checker (정규식 + 코퍼스 매칭)
```

Four core subsystems communicate only through files (JSONL/Parquet) and ChromaDB. Each subsystem is independently runnable and testable.

## 4. Components

### 4.1 Ingestion (`src/ingestion/`)

- **Responsibility:** External sources → normalized text → chunks + metadata.
- **Inputs:** Government statute OpenAPI, Supreme Court site, public Korean LegalQA datasets.
- **Outputs:** `data/processed/chunks.jsonl`, `data/processed/qa_seed.jsonl`.
- **Files:** `fetch_statutes.py`, `fetch_cases.py`, `chunk.py`, `synth_qa.py`.
- **Chunk schema:** `{id, source, doc_type[조문|판례], statute_no, case_no, title, text, char_range, hash}`.
- **Reproducibility:** All transforms are deterministic; raw API responses cached under `data/cache/`. Pipeline version recorded in JSONL header metadata.

### 4.2 Embedding & Index (`src/rag/`)

- **Responsibility:** Chunks → dense vectors + BM25 index → ChromaDB.
- **Inputs:** `data/processed/chunks.jsonl`.
- **Outputs:** `data/chroma/`, `data/bm25.pkl`.
- **Files:** `embed.py` (bge-m3 via FlagEmbedding), `index.py`, `bm25.py`, `retriever.py`.
- **Public API:** `Retriever.search(query: str, k: int = 5) -> list[Chunk]`.
- **Hybrid retrieval:** dense top-K and BM25 top-K, fused via Reciprocal Rank Fusion (RRF), final K=5.
- **VRAM note:** bge-m3 indexing is one-shot GPU work; query-time embedding is lightweight. The base LLM and bge-m3 are NOT loaded simultaneously to avoid OOM on 6GB VRAM.

### 4.3 Training (`src/training/`)

- **Responsibility:** QLoRA adapter training.
- **Inputs:** `data/processed/qa_train.jsonl` (instruction-completion pairs).
- **Outputs:** `models/adapters/qwen2.5-3b-civil-v{N}/`.
- **Files:** `prepare_dataset.py`, `train_qlora.py`.
- **Default hyperparameters:** Unsloth `FastLanguageModel`, 4-bit, LoRA r=16/α=32, gradient checkpointing, max_seq=2048 (auto-shrunk to 1024 if OOM), batch=1 + grad_accum=8, lr=2e-4, epochs=2~3, warmup=10%.
- **Logging:** TensorBoard under `runs/`, plus JSONL training log.
- **Resume:** Checkpoints every 200 steps.

### 4.4 Serving / Orchestrator (`src/serving/`)

- **Responsibility:** 4-mode inference (base / +adapter × ±RAG).
- **Inputs:** `query: str`, `mode ∈ {base, qlora, rag, rag_qlora}`.
- **Outputs:** `{answer: str, citations: list, retrieved: list[Chunk], latency_ms}`.
- **Files:** `model_loader.py`, `prompt_builder.py`, `orchestrator.py`, `app_gradio.py`.
- **Public API:** `Orchestrator.generate(query, mode) -> Response`.
- **Model loading:** Base model loaded once (Qwen2.5-3B 4-bit, ~2.5GB VRAM). Adapter is attached/detached at runtime to switch between QLoRA on/off without reloading base.
- **Inference caps:** `max_new_tokens=512`; RAG context capped at ~1500 tokens (drop lowest-ranked chunks if over).

### 4.5 Evaluation (`src/eval/`)

- **Responsibility:** 4-way ablation runner + scorer.
- **Inputs:** `data/eval/eval_set.jsonl` (`{question, reference_answer, expected_citations, reviewed_by, reviewed_at}`).
- **Outputs:** `reports/{run_id}/scores.csv`, `reports/{run_id}/report.md`.
- **Files:** `runner.py`, `judge.py`, `citation_checker.py`, `aggregate.py`.
- **Judge rubric:** 1–5 on accuracy (legal correctness), citation appropriateness, hallucination, readability. Temperature=0, explicit rubric prompt.
- **Citation checker:** Regex-extract `[민법 제○○조]`, `[대법원 ○○○○다○○○○○]` patterns; lookup in normalized corpus index. Reports three metrics: `citation_accuracy = found / total_cited` (precision of model's citations), `hallucination_rate = 1 - citation_accuracy`, and `citation_recall = expected_citations_cited / expected_citations_total` (coverage vs the eval item's `expected_citations` field).
- **Run id:** `{timestamp}_{git_sha[:8]}`.

### 4.6 Common (`src/common/`)

Config loader (YAML), structured logging, path helpers, Pydantic schemas (`Chunk`, `QAPair`, `EvalItem`, `Response`). No business logic.

### 4.7 Module Dependency Graph

```
common ◄── all modules
ingestion ──► common
rag ──► common, ingestion (chunks)
training ──► common, ingestion (qa)
serving ──► common, rag, training (adapter)
eval ──► common, serving
```

No cycles. Strictly top-down.

## 5. Data Flow

```
[Stage 1: Ingest]
  Statute API / Case crawl / Public QA → raw/*.json (cached)

[Stage 2: Process]
  raw/* → chunk.py → chunks.jsonl
        → qa_seed.jsonl

[Stage 3: Synthesize]
  qa_seed + chunks → Claude API (synth_qa) → qa_train.jsonl, qa_val.jsonl
                                            (+ hand-reviewed eval_set.jsonl)

[Stage 4: Index]
  chunks.jsonl → bge-m3 embed → chroma/
              → BM25 tokenize → bm25.pkl

[Stage 5: Train]
  qa_train.jsonl → Unsloth QLoRA → adapters/qwen2.5-3b-civil-v1/

[Stage 6: Serve]
  user_query → retriever → prompt_builder → Qwen2.5-3B [+adapter] → answer + citations

[Stage 7: Evaluate]
  eval_set + 4 modes → run all → judge + citation_check → scores.csv + report.md
```

Every stage is idempotent; rerunning a stage with unchanged inputs produces identical outputs (modulo non-determinism from external APIs, which are cached). Exposed through a `Makefile`: `make ingest / embed / train / eval / serve`.

### RAG Prompt Template

```
[시스템]
당신은 한국 민법 전문가입니다. 아래 참고자료를 근거로 답변하고,
근거가 부족하면 "참고자료에 명시되지 않음"이라고 답하세요.
인용은 [조문번호] 또는 [판례번호] 형식으로 본문에 표시하세요.
본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.

[참고자료]
{retrieved_chunks_with_ids}

[질문]
{user_query}

[답변]
```

In non-RAG modes, the `[참고자료]` section is removed and replaced with an instruction that no reference material is provided.

## 6. Repository Layout

```
PrivateLLM/
├── README.md
├── pyproject.toml
├── Makefile                    # make ingest/embed/train/eval/serve
├── .env.example
├── config/
│   ├── default.yaml
│   └── prompts/
│       ├── rag.txt
│       ├── no_rag.txt
│       └── synth_qa.txt
├── src/
│   ├── common/         (schemas.py, config.py, paths.py, logging.py)
│   ├── ingestion/      (fetch_statutes.py, fetch_cases.py, chunk.py, synth_qa.py)
│   ├── rag/            (embed.py, index.py, retriever.py, bm25.py)
│   ├── training/       (prepare_dataset.py, train_qlora.py)
│   ├── serving/        (model_loader.py, prompt_builder.py, orchestrator.py, app_gradio.py)
│   └── eval/           (runner.py, judge.py, citation_checker.py, aggregate.py)
├── tests/
│   ├── test_chunk.py
│   ├── test_retriever.py
│   ├── test_orchestrator.py
│   ├── test_citation_checker.py
│   ├── test_smoke.py
│   └── fixtures/
│       └── sample_chunks.jsonl
├── data/                       # gitignored
│   ├── raw/, cache/, processed/, chroma/, bm25.pkl
├── models/                     # gitignored
│   └── adapters/qwen2.5-3b-civil-v1/
├── reports/                    # gitignored
│   └── {run_id}/scores.csv, report.md
├── runs/                       # gitignored, TensorBoard
└── docs/superpowers/specs/
    └── 2026-05-02-legal-privatellm-mvp-design.md
```

`.gitignore`: `data/`, `models/`, `runs/`, `reports/`, `.env`, `__pycache__/`, `*.pyc`.

## 7. Dependencies

```
python = ">=3.10,<3.13"
torch = "==2.4.*"               # Unsloth-compatible pin
unsloth, transformers, peft, bitsandbytes, trl
chromadb, FlagEmbedding, rank-bm25
gradio, pydantic, anthropic
pytest, ruff
```

Windows note: `bitsandbytes` may require a prebuilt wheel or the `bitsandbytes-windows-webui` fork. Validated as part of M0. WSL2 fallback if Windows-native install fails.

## 8. Error Handling & Safeguards

### 8.1 VRAM OOM (primary risk)

| Site | Mitigation |
|---|---|
| Training start | Fail-fast on first step; log `max_memory_allocated()` and explicit recovery hints (reduce `max_seq` to 1024, `lora_r` to 8, `batch=1`). |
| Mid-training (long sequences) | `prepare_dataset.py` reports token-length distribution; `max_seq` auto-suggested at 95th percentile; over-length samples truncated. |
| Embedding indexing | Batch size from `EMBED_BATCH` env var (default 8); auto-halve on OOM. |
| Inference | `max_new_tokens=512`; RAG context capped at 1500 tokens; lowest-ranked chunks dropped first. |

### 8.2 External API failures

| API | Mitigation |
|---|---|
| Statute OpenAPI | On-disk response cache; 3 retries with exponential backoff. |
| Supreme Court crawl | Respect robots.txt; ≥1 s/request; failures logged to `data/cache/failed.jsonl`; pipeline continues. |
| Claude API (synth + judge) | SDK retries; cost cap via `MAX_API_USD` env var; judge results streamed to JSONL so reruns skip already-scored items. |

### 8.3 Data quality gates

- Chunking: drop empty / <200 char chunks; deduplicate by hash.
- Synthetic Q&A: drop responses without `[조문번호]`-style citation, <50 char answers, exact duplicates. Log seed→synthesized ratio.
- Eval set: 100% hand-reviewed; `reviewed_by` and `reviewed_at` mandatory in JSONL.

### 8.4 Citation hallucination guard

Citation checker is the primary objective signal independent of LLM-as-Judge. Any citation produced by the model that does not exist in the corpus index counts as hallucination, contributing to `hallucination_rate`.

### 8.5 Reproducibility

- Fixed seeds (`torch`, `transformers.set_seed`).
- Adapter directories include `training_config.yaml`.
- All processed JSONLs include `pipeline_version` and `source_hash`.
- Run id ties evaluation reports to git SHA.

### 8.6 Out of scope (intentional)

User auth, distributed/multi-GPU training, custom quantization, advanced monitoring/alerting, complex retry libraries, DB migrations.

## 9. Testing Strategy

### 9.1 Pyramid (PoC)

```
              ┌──────────┐
              │  E2E (1) │   smoke: end-to-end one-shot
              ├──────────┤
              │ Integ (5)│   module boundaries
              ├──────────┤
              │  Unit    │   deterministic logic
              │  (15+)   │
              └──────────┘
```

### 9.2 Unit tests (TDD applied)

| Module | Cases |
|---|---|
| `chunk.py` | Article-unit splitting, dedup, drop <200 char, metadata preservation |
| `citation_checker.py` | Regex extraction (`[민법 제618조]`, `[대법원 2019다12345]`), corpus lookup, hallucination separation |
| `bm25.py` | Tokenization, doc/query scoring |
| `prompt_builder.py` | RAG vs no-RAG templates, chunk-id formatting, token cap |
| `schemas.py` | Pydantic round-trip, required-field validation |
| `synth_qa.py` | Post-processing (citation filter, length filter, dedup) |
| `aggregate.py` | 4-way score table, mean/std, mode comparison |

Fixtures: `tests/fixtures/sample_chunks.jsonl` (10 items), `sample_eval.jsonl` (5 items).

### 9.3 Integration tests

| Test | Coverage |
|---|---|
| `test_retriever_e2e` | Build Chroma index from fixtures; assert known query retrieves expected chunks. |
| `test_orchestrator_modes` | All 4 modes dispatch and produce distinct prompts (model call mocked). |
| `test_eval_runner` | Mock orchestrator + mock judge; assert `scores.csv` shape and content. |
| `test_citation_checker_with_corpus` | Real fixture corpus + sample answers; verify accuracy math. |
| `test_dataset_prep` | Seed → train/val split, chat template, max_seq truncation. |

### 9.4 E2E smoke (one)

`test_smoke.py`: 5-chunk + 3-QA fixtures → load Qwen2.5-3B 4-bit (no fine-tuning) → 1 query in `base` mode → citation checker passes → eval runner over a 1-item set succeeds. Training is NOT in smoke.

### 9.5 Not tested (intentional)

Real LLM output quality, external APIs (mocked), Gradio UI (manual), training convergence (visual via TensorBoard), bge-m3 internal accuracy.

### 9.6 TDD scope

- **TDD applied (test first):** citation_checker, chunk, bm25, prompt_builder, aggregate, schemas, synth_qa post-processing.
- **Test-after acceptable:** train_qlora, embed.py (library wrappers), Gradio app, fetch_*.py, synth_qa API call layer.

### 9.7 Evaluation pipeline self-check

- Run the same eval twice → measure judge variance.
- Sanity subset (10 items): 5 designed RAG-only, 5 designed QLoRA-only. After 4-way run, check expected pattern (RAG-only > base on RAG items; QLoRA-only > base on QLoRA items). Pattern violation indicates an orchestrator/eval bug. Sanity subset is always run as part of the eval set.

## 10. Milestones

### M0 — Project bootstrap (~½ day)
- `git init`, `pyproject.toml`, `Makefile`, `README.md`, `.env.example`.
- `src/common/` skeleton.
- Trivial pytest passes.
- Validate Unsloth + bitsandbytes on Windows: load Qwen2.5-3B 4-bit + 1-sentence generation; record VRAM in README.

**Done:** Qwen2.5-3B 4-bit loads and generates locally; VRAM measurement recorded.

### M1 — Ingestion + RAG baseline (1~2 days)
- Fetch all 民法 statutes (~1,118 articles).
- `chunk.py`, `embed.py`, `index.py`, `bm25.py`, `retriever.py` + tests.

**Done:** `python -m src.rag.retriever "임대차계약 갱신요구권"` returns top-5 chunks containing relevant articles (e.g., 제643조, 제654조).

### M2 — Orchestrator + Gradio + base/RAG modes (1 day)
- `model_loader.py`, `prompt_builder.py`, `orchestrator.py` (base + rag modes only).
- `app_gradio.py` with RAG on/off toggle.
- `citation_checker.py` + tests.

**Done:** Gradio UI shows a question answered in both base and RAG modes side-by-side; citation_checker displays extracted citations and corpus-match status.

### M3 — Synthesis + QLoRA training (2~3 days)
- `synth_qa.py` produces 1,000~3,000 synthetic Q&A pairs.
- `prepare_dataset.py` + `train_qlora.py`; one full training run.
- Adapter integration into orchestrator → all 4 modes work.
- Gradio adds QLoRA on/off toggle.

**Done:** Adapter directory created; UI compares all 4 modes; loss curve does not diverge.

### M4 — Eval set + 4-way ablation (2~3 days)
- Hand-review 100~150 items in `eval_set.jsonl` (contract law + lease focus).
- `runner.py`, `judge.py`, `aggregate.py` → first `report.md`.

**Done:** 4-way score table including LLM-as-Judge mean, Citation Accuracy, hallucination rate; statistical differences across modes documented.

### M5 (optional) — Refinement cycle (~1 week)
Identify weak areas from M4 → augment synthetic Q&A → retrain (v2 adapter) → re-evaluate; expand eval set as needed.

### Critical Path

```
M0 ─► M1 ─┬─► M2 ─┬─► M3 ─► M4 ─► M5
          │       │
          └───────┘
```

The eval-set hand review in M4 is the bottleneck; start it in parallel during M0~M3.

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Windows + bitsandbytes incompatibility | Medium | Critical | Validate in M0; fall back to WSL2. |
| 6GB VRAM training OOM | Medium | High | Step down `max_seq`, `lora_r`; final fallback to Qwen2.5-1.5B. |
| Synthetic Q&A quality too low | Medium | High | Hand-review 50 items at end of M3; if <70% pass, refine prompt and regenerate. |
| Supreme Court crawl blocked / structure changed | Medium | Medium | Start M1 with statutes only; defer cases to M5. |
| LLM judge variance | Low | Medium | Two-run variance check; rubric explicit; expand human-reviewed sample if variance is high. |
| Hand-review time shortfall | Medium | High | Treat 80 items as the M4 minimum; expand toward 100~150 in M5. |

## 12. Definition of Done (PoC)

The PoC is complete when:

1. All four modes (base / rag / qlora / rag_qlora) are operational on the same infrastructure.
2. ≥80-item evaluation set runs through 4-way ablation at least once successfully.
3. Citation Accuracy + LLM-as-Judge scores + hallucination rate are tabulated per mode.
4. Sanity-check eval subset shows expected pattern (eval pipeline trustworthy).
5. Gradio UI compares all 4 modes for arbitrary user input.
6. `report.md` states explicit conclusions: *"In this domain, RAG contributes ___, QLoRA contributes ___, combined they ___"*.

## 13. Out of Scope (this PoC)

- Case-law crawling beyond M5 exploration
- Multi-turn dialog / session management
- User auth, audit logging
- Production-grade serving (vLLM, TGI, etc.)
- Korean morphological analyzer tuning for BM25
- v2/v3 adapter comparison automation (M5 manual is acceptable)
