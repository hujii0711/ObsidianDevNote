# M3a — QLoRA smoke pipeline (synth → prepare → train → smoke gate) — Design Spec

- **Date:** 2026-05-03
- **Author:** continuation of `2026-05-02-legal-privatellm-mvp-design.md` (frozen) — refines §4.3 Training for the smoke slice.
- **Status:** Ready for plan
- **Scope:** First half of milestone M3. Lands the full QLoRA training pipeline + adapter attach/detach hook + a 200-pair / 1-epoch smoke run that gates the full training run (M3b).

## 0. Why split M3 into M3a + M3b

The MVP design treated M3 as one milestone. In practice the work has two regimes with different blast radii:

- **M3a (this doc):** torch ecosystem upgrade + new training subsystem + adapter integration + smoke. Several environment risks; first time any of this code runs end-to-end.
- **M3b (next plan, opened after M3a closes):** 1k+ pair synth, multi-epoch training, Orchestrator widening to 4 modes (`base`, `rag`, `qlora`, `rag_qlora`), Gradio mode toggle. Real time + API spend; pipeline already proven.

Splitting means an unexpected smoke failure does not waste hours on a doomed full training run, and the API spend is staged ($3 cap for smoke vs. $10–20 for full).

## 1. Goal

Wire the existing M0 base-model loader and M1 RAG retriever to a brand-new QLoRA training pipeline, deliver an adapter attach/detach hook on the model_loader singleton, and validate the whole chain by training one smoke adapter on a small synthetic dataset and confirming it loads + generates non-empty Korean.

**Out of M3a (deferred to M3b):**
- 1,000–3,000 pair full synth
- Multi-epoch training, full hyperparameter sweep
- `Orchestrator.generate(qlora|rag_qlora, ...)` runtime dispatch
- Gradio 4-mode toggle UI

**Definitely deferred (out of M3 entirely):**
- Adapter quality evaluation (M4 / `eval/runner.py`)
- Adapter v2/v3 comparison automation (M5 manual is fine)

## 2. Architecture

### 2.1 New / changed surfaces

```
src/training/                                  ← new subpackage
├── __init__.py
├── synth_qa.py         # Claude API few-shot expansion of qa_seed.jsonl + chunks.jsonl
├── prepare_dataset.py  # tokenize + chat-template + 90/10 split
└── train_qlora.py      # Unsloth FastLanguageModel + LoRA → models/adapters/<name>/

src/serving/model_loader.py  ← extended (additive)
  +attach_adapter(path: str | Path) -> None
  +detach_adapter() -> None
  +current_adapter() -> str | None

config/prompts/synth_qa.txt  ← new (system + few-shot framing)
data/processed/qa_seed.jsonl ← new, 12 hand-curated pairs
data/processed/qa_train.jsonl ← new, synth output (smoke: ~200 pairs)
models/adapters/qwen2.5-3b-civil-smoke-v0/  ← new, smoke adapter
runs/smoke-{timestamp}/  ← new, TensorBoard + train_log.jsonl

pyproject.toml  ← torch pin bump (Plan task 0)
Makefile        ← install-cuda-torch URL update; new make synth/prepare/smoke-train targets
```

### 2.2 Adapter integration design

`src/serving/model_loader._State` already exposes the additive shape (see Plan 2 review feedback that locked the publish-order). M3a extends without breaking the existing API:

```python
@dataclass
class _State:
    model: Any = None
    base_model: Any = None        # NEW — preserved reference for detach
    tokenizer: Any = None
    loader: str = ""
    adapter_path: str | None = None  # NEW — diagnostic + idempotency check
```

`attach_adapter(path)`: assert no adapter currently attached (or detach first); `_state.base_model = _state.base_model or _state.model`; `_state.model = PeftModel.from_pretrained(_state.base_model, str(path))`; `_state.adapter_path = str(path)`.

`detach_adapter()`: noop if no adapter; otherwise `_state.model = _state.base_model`; `_state.adapter_path = None`. (`PeftModel.unload()` could also work but the swap pattern is more transparent.)

`get_base_model()` semantics unchanged. The singleton invariant — one model in VRAM at a time — is preserved.

### 2.3 Why M3a does NOT widen Orchestrator

The smoke gate validates `attach_adapter` + `model.generate(...)` directly (in a slow integration test). Adding `qlora` and `rag_qlora` to `Orchestrator.generate`'s `Mode` literal AND extending the prompt builder AND updating Gradio in the same plan would couple unrelated risks. M3b will do that as one focused plan once the smoke adapter exists.

## 3. Pre-flight: torch 2.5+ ecosystem upgrade (Plan task 0)

The current pin `torch==2.4.*` blocks Unsloth (`AttributeError: torch._inductor has no attribute 'config'` because torchao≥0.13 wants torch≥2.5). Plan 1 worked around this with the transformers fallback. Plan 3 cannot — Unsloth's 2x speedup + memory savings is what makes 3B-class QLoRA fit on a 6 GB card.

### 3.1 Pin updates

- `pyproject.toml`: `torch>=2.5,<2.7` (researched at task time; the upper bound is the next-known-incompatible).
- `pyproject.toml`: revisit `transformers>=4.44,<4.50` — Unsloth's current pin range is `>=4.51.3,<=5.5.0`. Resolution: bump to `transformers>=4.51,<5.0`. The FlagEmbedding `dtype=` patch must still apply; `tests/test_postinstall.py` is the regression gate.
- `pyproject.toml`: `bitsandbytes>=0.43,<0.49` — confirm current 0.45.5 still works on torch 2.5+; if not, bump to `<0.50` and re-test.
- `Makefile`: `install-cuda-torch` URL: `cu124` → `cu126` (or whatever matches torch 2.5+'s latest CUDA wheel).

### 3.2 Re-validation gate (must pass before any other M3a code lands)

Run in order; each must succeed:

1. `make install-dev && make install-cuda-torch && make postinstall`
2. `.venv/Scripts/python -c "from unsloth import FastLanguageModel; print('ok')"` — Unsloth import works (was the broken signal).
3. `.venv/Scripts/python scripts/verify_unsloth.py` — loads via the Unsloth path now, generates Korean.
4. `.venv/Scripts/python scripts/verify_combined_vram.py` — peak < 5.5 GB.
5. `.venv/Scripts/python -m pytest -m "not slow" -q` — all 52 fast tests pass.
6. `.venv/Scripts/python -m pytest -m slow -v` — all 4 slow tests pass.

If step 2 fails (Unsloth still broken on torch 2.5+), abort the plan, escalate to the user with options:
- (b1) try a specific Unsloth commit known-good with current torch
- (b2) fall back to vanilla transformers + peft Trainer for the rest of M3a (slower but verified path)

### 3.3 Commit policy

The pre-flight is a single commit (`build(deps): torch 2.5+ ecosystem upgrade for Unsloth-enabled QLoRA training`). If smoke or any later task needs to revert, this commit is one `git revert` away.

## 4. Synthesis pipeline (`src/training/synth_qa.py`)

### 4.1 Inputs

- `data/processed/qa_seed.jsonl` — 12 hand-curated `QAPair`s (`source="seed"`). Drafted by the implementer from `chunks.jsonl`, reviewed by the user before synth runs. Topic coverage: 임대차 (lease), 매매 (sale), 채권 (claims), 손해배상 (damages), 시효 (prescription) — civil-law fundamentals.
- `data/processed/chunks.jsonl` — M1 output.
- `config/prompts/synth_qa.txt` — system + few-shot framing. Ships with the plan.

### 4.2 Public API

```python
def run_synth(
    *,
    target_pairs: int,
    model: Literal["sonnet", "opus"] = "sonnet",
    max_usd: float = 3.0,
    cache_dir: Path | None = None,
) -> Path:
    """Run synthesis until target_pairs reached or cost cap hit. Returns the
    written qa_train.jsonl path. Resumable via cache_dir."""
```

### 4.3 Per-batch flow

1. Sample 3–5 chunks clustered by `statute_no` proximity (so the few-shot answer can plausibly cite multiple related articles).
2. Format prompt: system prompt + 3 random seed examples + the sampled chunks, asking for 2 grounded Q&A pairs with `[민법 제○○조]` citations.
3. Call Anthropic SDK (`claude-sonnet-4-6` default; `claude-opus-4-7` via `model="opus"`).
4. Parse response (expects fenced JSON; on parse failure, log + skip the batch).
5. Cache the raw response under `cache_dir / f"{batch_hash}.json"` so reruns skip the API.

### 4.4 Post-processing filters

Reuses `src.eval.citation_checker._STATUTE_RE` to enforce citation presence. Drop a synthesized row if:
- No statute regex match in `output`.
- `len(output) < 50` characters.
- `(instruction, output)` hash matches an already-emitted row.

Each drop is logged at INFO with the reason; final summary prints `kept N / synthesized M (drop reasons: …)`.

### 4.5 Cost gate

- Anthropic SDK responses include `usage.input_tokens` + `usage.output_tokens`. The script tracks running cost using static price tables for the chosen model.
- Before each API call, if `running_cost + estimated_call_cost > max_usd`, abort and write whatever was produced so far. Exit code 0 (the partial output is usable).
- The smoke run uses `max_usd=3.0` and `target_pairs=200`.

## 5. Dataset prep (`src/training/prepare_dataset.py`)

### 5.1 Public API

```python
def prepare(
    *,
    qa_path: Path,
    out_dir: Path,
    tokenizer,
    max_seq: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Returns (train_path, val_path). 90/10 split. Writes JSONL of
    {input_ids, attention_mask, labels} per row."""
```

### 5.2 Behavior

- Reads `qa_path` as a stream of `QAPair` rows.
- Applies the Qwen2.5 chat template using `tokenizer.apply_chat_template([{role:'user', content:instruction}, {role:'assistant', content:output}], tokenize=True, return_tensors=None)` — gets the full sequence ids.
- Computes the assistant-only mask: tokens before the assistant's response are labeled `-100` (ignored by cross-entropy); assistant tokens keep their id. The boundary is detected by re-tokenizing with `add_generation_prompt=True` to find the prompt prefix length.
- Reports `len(input_ids)` p50 / p95 / max; if `max_seq is None`, suggests `ceil_to_64(p95)`. The actual training cap is set by the caller.
- Truncates rows over `max_seq` (assistant-side truncation; instruction is preserved).
- Deterministic 90/10 split via `random.Random(seed).shuffle`.

## 6. Training (`src/training/train_qlora.py`)

### 6.1 Public API

```python
def train(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    smoke: bool = False,
    epochs: int = 2,
    max_seq: int = 2048,
) -> Path:
    """Returns adapter directory. JSONL log + TensorBoard land under runs/."""
```

### 6.2 Hyperparameters

Per MVP design §4.3 with smoke overrides:

| Param | Full (M3b default) | Smoke (M3a) |
|---|---|---|
| LoRA r | 16 | 16 |
| LoRA α | 32 | 32 |
| LoRA dropout | 0 | 0 |
| target_modules | q/k/v/o/gate/up/down_proj | same |
| gradient_checkpointing | True | True |
| max_seq | 2048 (auto-shrink to 1024 if OOM) | 1024 |
| batch_size | 1 | 1 |
| grad_accum | 8 | 8 |
| lr | 2e-4 | 2e-4 |
| warmup_ratio | 0.1 | 0.1 |
| weight_decay | 0.01 | 0.01 |
| optim | adamw_8bit | adamw_8bit |
| epochs | 2–3 | 1 |
| max_steps | (epoch-driven) | ≈25 |

### 6.3 Logging

- TensorBoard: `report_to=["tensorboard"]`; `logging_dir=runs/{name}/tb/`.
- Custom JSONL log via `TrainerCallback`: one line per logging step at `runs/{name}/train_log.jsonl` containing `{step, loss, lr, epoch, time_ms}`.
- Final adapter: `output_dir / "adapter_config.json"` + `adapter_model.safetensors`.

### 6.4 Resume

Checkpoints every 100 steps (`save_steps=100`, `save_total_limit=2`). The smoke run lands at most one checkpoint; the full M3b run uses these for resume.

## 7. Smoke gate (`make smoke-train` end)

After `train(smoke=True)` returns:

1. Adapter dir exists and contains both `adapter_config.json` and `adapter_model.safetensors`.
2. `train_log.jsonl` contains ≥2 step rows.
3. **Loss sanity:** `final_loss < first_loss * 0.95` (≥5% reduction). Hard fail on divergence; soft warn (PASS with concern) if reduction is between 0% and 5%.
4. **Adapter integration smoke:** in a fresh subprocess, `from src.serving.model_loader import get_base_model, attach_adapter; m, t = get_base_model(); attach_adapter(adapter_dir); out = m.generate(t.encode("임대차의 의의는?", return_tensors="pt").to(m.device), max_new_tokens=32); print(t.decode(out[0]))` — assert non-empty, contains Korean characters.

Pass → exit 0, print one-line summary, recommend M3b. Fail → exit 1, summary specifies which gate failed, M3b plan not opened.

## 8. Test contract (TDD-applicable surfaces in **bold**)

| File | Tests | Type |
|---|---|---|
| **`synth_qa.py` post-processing** | citation filter, length filter, dedupe, cost-cap abort | unit (mock SDK) |
| **`prepare_dataset.py`** | chat-template tokenization, label mask of instruction tokens, 90/10 split determinism, max_seq truncation, p95 token-length report | unit (uses real tokenizer; fast — no model load) |
| **`model_loader.attach_adapter / detach_adapter`** | argument coercion (str vs Path), idempotency error on double-attach, current_adapter() roundtrip | unit (no GPU; use a stub `_state.model` mock) |
| `train_qlora.py` | not unit-tested (library wrapper); covered by `test_smoke_train.py` | — |
| `test_smoke_train.py` | full pipeline: mocked Claude → real prepare_dataset → real Unsloth train (`max_steps=2, target_pairs=5`) → real attach_adapter → one model.generate. CUDA-gated. | slow (real GPU) |

Counts goal at end of M3a: ~62 fast (52 prior + ~10 new unit) + 5 slow (4 prior + 1 smoke). Fast suite stays under ~5 s.

## 9. Acceptance criteria (= M3a "done")

1. Pre-flight commit (§3) lands; all 6 re-validation steps green.
2. `make synth` produces ≥150 pairs in `data/processed/qa_train.jsonl` within `MAX_API_USD=3` cap.
3. `make prepare` writes `qa_train_tokenized.jsonl` and `qa_val_tokenized.jsonl` with 90/10 split.
4. `make smoke-train` produces a smoke adapter and prints PASS, with `final_loss < first_loss * 0.95`.
5. `model_loader.attach_adapter(smoke_dir)` followed by `model.generate(...)` returns non-empty Korean text.
6. All fast tests green; slow suite green.
7. `verify_combined_vram.py` peak still < 5.5 GB after the torch upgrade.

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unsloth still broken on torch 2.5+ | Low | High | §3.2 step 2 catches it; fall back to peft + transformers Trainer (option (b) from brainstorm) |
| 6 GB OOM on smoke despite Unsloth | Medium | High | max_seq=1024, batch=1 + grad_accum=8, gradient_checkpointing. If OOM persists: r=8 → r=4 → final fallback Qwen2.5-1.5B |
| FlagEmbedding patch breaks after upgrade | Medium | Medium | `tests/test_postinstall.py` regex regression test; reapply via `make postinstall` |
| Synth API cost overrun | Low | Low | $3 cap; per-batch cost log; cached responses skip on resume |
| Synth quality too low for smoke | Medium | Low | Smoke checks plumbing not quality; loss sanity ≥5% covers gross divergence |
| Smoke loss decreases <5% (genuine model issue) | Medium | Medium | Soft warn → PASS with concern; let user decide whether to proceed to M3b or iterate prompts |
| `attach_adapter` leaks the prior base model | Low | Medium | `_state.base_model` reference + explicit detach swap; no `del` in hot path |

## 11. M3b preview (NOT this plan)

When M3a closes, write `docs/superpowers/specs/<date>-m3b-full-training-and-orchestrator-design.md`:

- Full synth: 1,000–1,500 pairs (re-uses `synth_qa.run_synth(target_pairs=1500, max_usd=15)`)
- Full train: 2 epochs, max_seq=2048, save adapter `qwen2.5-3b-civil-v1/`
- `Orchestrator.generate` widening: `Mode = Literal["base","rag","qlora","rag_qlora"]`; mode dispatch attaches/detaches adapter as needed
- Gradio 4-mode toggle: radio buttons + 4-column comparison or 2x2 grid
- New slow integration test exercising all 4 modes
