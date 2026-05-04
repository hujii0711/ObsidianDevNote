# M2 Readiness — Pre-Plan Checklist

This document captures the state at the end of Plan 1 (M0 + M1) and lists everything that must be true before starting Plan 2 (M2 — Orchestrator + Gradio UI with base / RAG modes). It is a working document: tick items off as they are addressed; new items can be appended.

## Plan 1 Summary

**Tags:** `m0-bootstrap`, `m1-rag-baseline`
**Tests:** 29 passed (27 fast + 2 slow opt-in via `-m slow`)
**Acceptance signals captured:**
- Qwen2.5-3B 4-bit loads on RTX 3060 Laptop 6 GB → VRAM 2.07 GB; generates Korean text correctly (`scripts/verify_unsloth.py`).
- E2E retriever: chunk(fixture) → bge-m3 embed → ChromaDB+BM25 hybrid → 民法 第618조 retrieved for query "임대차의 정의를 설명해 주세요" (`tests/test_e2e_rag.py`).

### Modules delivered

| Module | Files | Tested via |
|---|---|---|
| `src/common/` | `schemas.py`, `paths.py`, `config.py`, `logging.py` | `test_schemas.py` (TDD, 11 tests) |
| `src/ingestion/` | `fetch_statutes.py`, `chunk.py` | `test_ingestion_smoke.py` (mocked, 2), `test_chunk.py` (TDD, 6) |
| `src/rag/` | `bm25.py`, `embed.py`, `index.py`, `retriever.py` | `test_bm25.py` (TDD, 4), `test_retriever.py` (TDD, 4 unit + 1 slow integration), `test_e2e_rag.py` (1 slow E2E) |
| `scripts/verify_unsloth.py` | M0 acceptance script with transformers fallback | manual (recorded VRAM + generation in README) |

### Empirical adaptations from the original plan

These were forced by the current Windows + Python 3.12 ecosystem:

- `requires-python = ">=3.10,<3.13"` (kept) — README mandates 3.12 specifically
- `transformers>=4.44,<4.50` (was `<5`) — peft / sentence_transformers / FlagEmbedding break on 4.50+
- `bitsandbytes>=0.43,<0.49` (was unconstrained upper) — 0.49.x skips C++ kernels
- `pyarrow>=21,<24` (added) — pyarrow 24 caused Windows ACCESS_VIOLATION with torch+sklearn+pandas
- `unsloth_zoo` added to `[unsloth]` extra
- `tool.ruff.target-version = "py310"` (was `py311`) — match the actual lower bound
- `scripts/verify_unsloth.py` falls back to vanilla `transformers` because Unsloth requires `torch>=2.5`

## Open issues from final review (Plan 1)

These were flagged "Important" by the final code reviewer. None block Plan 1 closure; all should be resolved during M2 prep or the first M2 task.

### Important (resolve before adding M2 code)

#### A. FlagEmbedding venv patch is undocumented in source control

`.venv/Lib/site-packages/FlagEmbedding/finetune/embedder/encoder_only/m3/runner.py` was patched in place (`dtype=` → `torch_dtype=`) so that `BGEM3FlagModel` works with transformers 4.49. The patch is gitignored — anyone who recreates the venv must reapply it.

**Options:**
1. Pin `FlagEmbedding` to a version that does not have the bug (research required: 1.3.x predates the issue but may not support bge-m3-fp16 the same way).
2. Add `scripts/postinstall.py` that applies the sed-style patch and document it in README quickstart.
3. Vendor a minimal embedder wrapper that doesn't import `FlagEmbedding.finetune.*`.

**Decision needed:** which option, by whom, when.

#### B. Combined VRAM budget unverified

bge-m3 (≈ 1.3–1.5 GB fp16) + Qwen2.5-3B 4-bit (≈ 2.1 GB) + KV cache + RAG context (≈ 0.5–1 GB) ≈ 4 GB on a 6 GB card. Tight but likely OK.

**Action:** add `scripts/verify_combined_vram.py` as **the first M2 task** — load both models, run one RAG query, assert `torch.cuda.max_memory_allocated() < 5.5 GB`. If it fails, mitigate (CPU embedder at query time, lower fp16 setting on bge-m3, etc.) BEFORE building the orchestrator.

#### C. ChromaDB `Settings` coupling

Both `src/rag/index.py:51` and `src/rag/retriever.py:88` pass `Settings(allow_reset=True)` to `chromadb.PersistentClient(...)`. The orchestrator (M2) is the third caller. If it forgets, the cached settings mismatch raises.

**Action:** factor a `_make_chroma_client(path)` helper into `src/rag/index.py` (or a new `src/rag/_chroma.py`) and use it from all three sites.

#### D. Retriever's dense `(id, score)` tuple is misleading

`dense_ranked = [(doc_id, 1.0 - dist) for ...]` in `retriever.py`. With cosine distance, `1 - dist ∈ [-1, 1]` — negative scores are not a sane filtering threshold. RRF only uses ranks, so the score is unused, but a future caller may misuse it.

**Action:** change to `dense_ranked: list[str]` (ranks-only) and update `rrf_fuse` signature accordingly. Add one assertion of exact RRF score (`1/(k+rank+1)`) so a future regression to a broken formula gets caught.

### Minor (queue or defer)

- `config/default.yaml` has dead keys: top-level `paths:` block is never read, `chunk.overlap` is never read.
- `src/ingestion/chunk.py:43` defines `_ARTICLE_NO_RE` but never uses it.
- `src/rag/retriever.py:30` defines `RetrievalHit` dataclass but `Retriever.search` returns `list[Chunk]`. Either drop or expose for serving.
- `src/common/logging.py` `_initialized` flag is not thread-safe; if M2's Gradio uses worker threads, two threads could double-init the root handler.
- `Makefile` `clean` target uses POSIX `rm -rf` (Windows needs git-bash).
- `.gitignore` whitelists `data/eval/.gitkeep` but the file does not exist; the directory has no anchor.

## Required before opening Plan 2

- [x] **A** — FlagEmbedding patch automated as `scripts/postinstall.py` (option 2). Idempotent rewrite of `from_pretrained(... dtype=)` -> `torch_dtype=`. Pinned by `tests/test_postinstall.py`. `make postinstall`. — 2026-05-03
- [x] **B** — `scripts/verify_combined_vram.py` runs an end-to-end RAG turn with both models loaded; **measured peak 3.17 GB / 5.5 GB budget on the target RTX 3060 Laptop**, well under headroom. — 2026-05-03
- [x] **C** — `make_chroma_client(path)` factory in `src/rag/index.py`; both `index.py` and `retriever.py` call it. — 2026-05-03
- [x] **D** — `rrf_fuse(dense_ids, sparse_ids, ...)` now takes bare `list[str]`; canonical `1/(k+rank+1)` formula pinned by a regression test. — 2026-05-03

All four resolved on 2026-05-03. Next: write `docs/superpowers/specs/2026-05-03-m2-orchestrator-design.md` (or extend the existing spec's Serving subsystem section) and run `/superpowers:writing-plans` for Plan 2.

## What Plan 2 will deliver (preview)

From the design spec §4.4 (Serving / Orchestrator):

- `src/serving/model_loader.py` — base 4-bit + (later) adapter attach/detach
- `src/serving/prompt_builder.py` — RAG vs no-RAG prompt templates
- `src/serving/orchestrator.py` — `Orchestrator.generate(query, mode)` with modes `{base, rag}` (M2) and later `{qlora, rag_qlora}` (M3)
- `src/serving/app_gradio.py` — single-file Gradio UI with mode toggles, retrieved-context display, citation extraction (depends on `src/eval/citation_checker.py` being available)

Plan 2 should also include `src/eval/citation_checker.py` (regex extraction + corpus lookup), as the Gradio UI uses it for visual citation status. The full evaluation runner (`runner.py`, `judge.py`, `aggregate.py`) belongs to Plan 4 (M4).

## Suggested Plan 2 first-three tasks

1. **Bootstrap polish** — implement A, B, C, D above; commit each fix separately.
2. **`src/serving/__init__.py` + `model_loader.py`** — singleton 4-bit base loader, sharing the existing transformers fallback logic from `verify_unsloth.py`.
3. **`src/serving/prompt_builder.py` + tests (TDD)** — RAG vs no-RAG templates, chunk-id formatting, max-context-token cap. This is the smallest piece that's pure logic and TDD-friendly.

After these three, the orchestrator and Gradio UI are mechanical.

---

## Plan 2 — closed (2026-05-03)

**Tag:** `m2-orchestrator`
**Tests:** 52 fast + 4 slow. Slow runtime ~80 s on the target host.

### Modules delivered

| Module | Files | Tested via |
|---|---|---|
| `src/serving/` | `model_loader.py`, `prompt_builder.py`, `orchestrator.py`, `app_gradio.py` | `test_prompt_builder.py` (7 TDD), `test_orchestrator.py` (4 unit incl. thread-lock), `test_serving_integration.py` (2 slow — singleton + real-model round-trip) |
| `src/eval/` | `citation_checker.py` | `test_citation_checker.py` (9 TDD; regex variants + dedup + corpus stub) |
| `config/prompts/` | `rag.txt`, `no_rag.txt` | exercised by every prompt_builder test |
| `scripts/` | `verify_combined_vram.py` (refactored to use shared loader) | rerun gate per `make` |
| `pyproject.toml` | `gradio>=4,<6` declared | UI smoke + HTTP probe |

### Deviations from the original Plan 2 doc

- The plan's `tests/test_serving_integration.py::test_orchestrator_real_models_round_trip` literal asserted `c.statute_no == "618"`; corrected to `"제618조" in c.statute_no` to match the existing `chunk_civil_code` schema (mirrors the `test_e2e_rag.py` convention). Plan doc was updated in commit `cef8262`.
- The plan assumed `gradio` was already in `[dev]` extras — it was not. Added to main `dependencies` (commit `9e85cd1`) before retrying Task 6.
- Task 2 added a truncation-warning log + `caplog` regression test (Important-rated review feedback on the spec promise that "truncation never happens silently"). Implementation is now strictly in line with spec §2.2.
- Task 1 fixed a DCL publish-order race (Important-rated review feedback): the singleton's `_state.model` is now assigned LAST so the fast-path read at the top of `get_base_model()` cannot observe a half-initialized state.

### Acceptance signals captured (per plan §5)

- ✅ `make serve` opens `http://127.0.0.1:7860`; HTTP `/` and `/config` respond 200 on the target host.
- ✅ Slow E2E: orchestrator returns base + rag for "임대차의 정의를 설명해 주세요." in <15 s after warmup; rag retrieves 民法 第618條 reliably.
- ✅ Citation extractor lifts `[민법 제618조]`-shaped citations from generated answers (`extract_citations`), and the UI badges them via `verify_citations(corpus=None)` stub.
- ✅ `pytest -m "not slow"` is green at 52 tests.
- ✅ `scripts/verify_combined_vram.py` continues to PASS at peak 3.17 GB / 5.5 GB budget — Task 5's full RAG turn through the orchestrator did not regress headroom.

### Punchlist for Plan 3 (M3 — QLoRA training)

These are NOT M2 blockers but should be addressed before / during M3:

- Resolve the torch 2.4 → 2.5+ pin to unblock Unsloth (the env-pins memory documents the cascade). The transformers fallback works on 2.4, but Unsloth's optimizations are why we're here.
- Adapter attach/detach in `model_loader.py` (the `_State` dataclass is M3-ready).
- `Response.mode` literal already includes `qlora` and `rag_qlora`; runtime dispatch in `Orchestrator.generate` will need to widen the `Mode` alias and branch the prompt template.
- `Orchestrator.open()` is not thread-safe — the lazy module-level `_ORCH` in `app_gradio.py` plus Gradio's request queue make this fine for a 1-user PoC; revisit if a multi-user entry point appears.

### Minor cleanup that landed during Plan 2

- `config/default.yaml` lost the dead top-level `paths:` block and `chunk.overlap` key; the per-call `serving:` section was added.
- README's status table, file map, and test counts were synced to the new tree.

The minor items (`_ARTICLE_NO_RE` unused, `RetrievalHit` unused, logging thread-safety, `make clean` POSIX-only, `data/eval/.gitkeep` missing) from the §"Minor (queue or defer)" list above are still open and should be picked up in M3 / M4 prep — none of them blocked Plan 2.

---

## M3a — code complete, smoke gate pending (2026-05-03)

**Status:** code path delivered (Tasks 1–7 of `docs/superpowers/plans/2026-05-03-m3a-qlora-smoke.md`); the real `make smoke-train` was NOT executed because `ANTHROPIC_API_KEY` and `LAW_OPEN_API_KEY` are not in `.env`. **No `m3a-qlora-smoke` tag yet** — it lands once the smoke gate passes.

**HEAD:** `eb79a84` (Task 7 commit).
**Tests:** 71 fast + 5 slow all green (the slow set includes the new `test_smoke_train.py` E2E with mocked Claude + 2 training steps).

### Modules delivered

- `src/training/__init__.py`, `synth_qa.py`, `prepare_dataset.py`, `train_qlora.py`
- `src/serving/model_loader.py` extended with `attach_adapter` / `detach_adapter` / `current_adapter`
- `data/processed/qa_seed.jsonl` (12 hand-curated pairs covering 임대차/매매/채권/손해배상/시효/동시이행/해제/변제/보증/위임/사용대차)
- `config/prompts/synth_qa.txt` few-shot template
- `config/default.yaml` extended with `synth:` and `training:` blocks
- `Makefile`: `synth`, `prepare`, `smoke-train` targets

### Deviations from the M3a plan as written

- Task 1 follow-up: `[unsloth]` extras now pin `xformers / torchao / torchvision / diffusers / triton-windows` to the working cu126 set (commit `2369ccc`). Without this, a future clean-venv `make install-unsloth` re-breaks.
- Task 5 lint: ruff E741 forced `l → lab` rename in one test (cosmetic).
- Task 7 patches:
  - `_JsonlLossCallback` inherits `transformers.TrainerCallback` (must — Trainer dispatches `on_init_end` etc. on every callback).
  - `bf16=torch.cuda.is_bf16_supported()` instead of fp16=True. RTX 3060 Laptop is Ampere → bf16 path; Unsloth's LoRA kernel raised `RuntimeError: self and mat2 must have the same dtype, but got Half and Float` on fp16. bf16 is also generally better for LoRA stability.
  - Slow integration test bumps mocked synth from 5 → 20 pairs and overrides `smoke_max_steps=2` + `grad_accum=1` so that 2 optimizer steps actually log.

### To finish M3a (when API keys are available)

1. Place `ANTHROPIC_API_KEY` + `LAW_OPEN_API_KEY` in `C:\claudeProject\PrivateLLM\.env`.
2. Run, in order:
   ```powershell
   make ingest                             # ~1 min, free
   python -m src.ingestion.chunk           # writes data/processed/chunks.jsonl
   make synth                              # ~5–10 min, ~$1–3 Anthropic API
   make prepare                            # ~10 s
   make smoke-train                        # ~10–20 min GPU
   ```
3. If the smoke gate prints PASS (`final_loss < first_loss × 0.95`):
   ```powershell
   git tag m3a-qlora-smoke -m "M3a: QLoRA smoke pipeline gate passed"
   ```
4. If FAIL: capture `runs/smoke-*/train_log.jsonl`, the adapter dir state, and the verdict line; report back to decide between (a) prompt-engineering iteration on `config/prompts/synth_qa.txt`, (b) hyperparameter tweak (lr / r), (c) larger smoke (target_pairs=400).

### M3b open questions (deferred until M3a tag)

- Full synth target_pairs: design says 1000–1500. Decide based on M3a synth quality.
- ~~Whether `Mode` widening to `qlora` and `rag_qlora` should also unlock per-mode prompt template differences.~~ — **Decided 2026-05-03 in M3b-prep T1: NO. qlora reuses no_rag.txt and rag_qlora reuses rag.txt; the adapter encodes the domain knowledge, the prompt encodes the role. Per-mode template tweaks are revisitable if M4 eval shows they'd help.**
- Adapter directory naming convention as more versions are produced (`v1` / `v2` / ...).

---

## M3b prep — landed (2026-05-03)

While waiting on M3a smoke API keys, the M3b code that doesn't need a real adapter went in. HEAD `c870250`. **No tag yet** — M3b proper closes after the 4-mode slow integration test runs against a real adapter (post-M3a).

**Tests:** 82 fast (75 → +7 orchestrator dispatch) + 5 slow (unchanged). 4-mode slow integration test deferred — wiring exists but needs a real adapter to be meaningful.

### Modules delivered

- `src/serving/prompt_builder.py` — `Mode` widened to 4-way; `_PROMPT_FILES`, `_RAG_MODES`, `_VALID_MODES` constants; `_RAG_MODES`-driven dispatch.
- `src/serving/orchestrator.py` — 4-mode dispatch, `adapter_path` field on the dataclass (read from `cfg.serving.adapter_path` at `.open()`), per-call attach/detach policy under the existing `_lock`. qlora/rag_qlora without an adapter raise `RuntimeError` before any other work.
- `src/serving/app_gradio.py` — 2x2 grid (base / rag / qlora / rag_qlora). `_generate_safely` wraps each call so per-mode failures (e.g. `RuntimeError("no adapter configured")`) render in the cell instead of crashing the UI.
- `config/prompts/qlora.txt`, `config/prompts/rag_qlora.txt` — content-identical to no_rag/rag respectively.
- `config/default.yaml` — `serving.adapter_path: null` (set this to a `models/adapters/<name>/` path after M3a smoke training completes).

### Deviations from a strict "M3b-as-written" plan

- Live-model rebinding in `Orchestrator.generate` is gated on `swapped=True` (only after attach or detach actually ran), not the spec's always-on pattern. Reason: alphabetical test ordering puts `tests/test_adapter_hooks.py` (which writes a MagicMock to `_ml._state.model`) before `tests/test_orchestrator.py`; an always-on rebind would pick up the leftover mock instead of each test's tracked `_FakeModel`. The gate preserves real-path semantics while keeping unit tests isolated.
- The `test_invalid_mode_raises` orchestrator test was updated to use `mode="bogus"` (qlora is now valid) — same commit since it's part of the Mode widening.

### What's still open for M3b proper

1. **Real-adapter slow integration test** — exercises all 4 modes end-to-end with a trained adapter. Code shape is ready (mirrors `test_orchestrator_real_models_round_trip`); blocked on M3a producing `models/adapters/qwen2.5-3b-civil-smoke-v0/`.
2. **Set `serving.adapter_path`** in `config/default.yaml` to the smoke adapter's path once it exists.
3. **Tag** `m3b-orchestrator-4-mode` after the slow E2E test passes.
4. **Full training run (target_pairs ≥ 1000)** — separate plan; produces `qwen2.5-3b-civil-v1/` for M4 evaluation.

---

## M4 prep — landed (2026-05-03)

Same-session continuation while M3 awaits API keys. HEAD `32a85b0`. M4 proper (runner + judge + first ablation report) is a separate plan that opens after M3a tags.

**Tests:** 95 fast (88 → +7 aggregate; the +6 corpus tests landed in T2). Slow suite unchanged at 5.

### Modules delivered

- `data/eval/.gitkeep` + `data/eval/eval_set.jsonl` — 10 hand-curated `EvalItem`s covering qa_seed-overlapping topics (임대차/매매/손해배상/시효), adjacent territory (청약·승낙/부당이득/불법행위), and compositional items (갱신요구권/하자담보책임/채권자대위권). `expected_citations` are bare normalized statute numbers so M4's `citation_recall` can be computed by string equality.
- `src/eval/corpus.py` — `load_corpus(chunks_jsonl) -> set[str]` reads M1 chunks output, extracts normalized statute numbers (regex shared with `citation_checker`) + bare case numbers. Used by the Gradio UI's `_format_citations` (graceful fallback to None=stub-true when `chunks.jsonl` missing).
- `src/common/schemas.EvalRow` — wire-format Pydantic model for one judged-and-citation-checked answer (eval_id, mode, answer, judge_score, citation_accuracy, hallucination_rate, citation_recall, latency_ms).
- `src/eval/aggregate.py` — `aggregate(rows)` (per-mode mean/std using statistics.fmean/stdev; single-sample modes get std=0), `write_csv(table, path)`, `write_markdown(table, path, run_id=...)`. Canonical mode order is `base / rag / qlora / rag_qlora`; unknown modes append.

### Gradio integration

`app_gradio._get_corpus()` lazy-loads from `data/processed/chunks.jsonl` once, falls back to None when chunks.jsonl is absent. So a fresh clone with no ingest run still gets a working UI; once `make ingest` lands chunks, the corpus auto-attaches on next launch.

### What's still open for M4 proper

1. **`src/eval/runner.py`** — for each (eval_item, mode) pair: `Orchestrator.generate(...)` → `extract_citations` on the answer → `verify_citations(corpus=load_corpus(...))` → compute citation_accuracy / hallucination_rate / citation_recall → write `EvalRow` to a JSONL. Streams to disk so reruns can skip already-scored items.
2. **`src/eval/judge.py`** — Claude LLM-as-Judge with the MVP design's 1–5 rubric on accuracy / citation appropriateness / hallucination / readability. Anthropic SDK; needs `ANTHROPIC_API_KEY`.
3. **`run_id` convention** — `{timestamp}_{git_sha[:8]}` per the MVP design.
4. **First ablation run** over the 10 starter items (or expanded set) once a trained adapter exists; produces `reports/{run_id}/scores.csv` + `report.md`. The "Definition of Done" criteria 1–6 in the MVP design come from this report.
5. **Eval set expansion** — the 10 items here are starter; user / domain expert is expected to review and expand toward the 80–150 items the MVP design calls for. The `reviewed_at: 2026-05-03` placeholder bumps when items are re-reviewed.
