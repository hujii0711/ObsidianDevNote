# M3a — QLoRA smoke pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the QLoRA training pipeline (synth → prepare → train) + adapter attach/detach hook end-to-end, gated by a 200-pair / 1-epoch smoke run that confirms the torch 2.5+ ecosystem upgrade was clean and the trained adapter loads + generates non-empty Korean.

**Architecture:** A pre-flight pin upgrade unlocks Unsloth (was blocked on torch>=2.5 transitively via torchao). `src/training/` adds three modules (`synth_qa`, `prepare_dataset`, `train_qlora`) communicating only through JSONL files. `src/serving/model_loader.py` extends additively with `attach_adapter` / `detach_adapter` over the existing singleton. The smoke gate compares `train_log.jsonl` first-vs-last loss and runs one adapter-loaded `model.generate(...)` end-to-end.

**Tech Stack:** Python 3.12, torch 2.5+ + cu126 wheel, Unsloth `FastLanguageModel`, peft 0.12+, transformers 4.51+, bitsandbytes, Anthropic SDK, ChromaDB / bge-m3 (M1, untouched).

**Reference docs to consult before coding:**
- `docs/superpowers/specs/2026-05-03-m3a-qlora-smoke-design.md` — design spec for this plan
- `docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md` §4.3, §10 — frozen MVP spec
- `docs/superpowers/m2-readiness.md` "Plan 2 — closed" section + M3 punchlist
- `Users/hujii/.claude/projects/C--claudeProject-PrivateLLM/memory/project_env_pins.md` — non-obvious environment pins

**Schemas already defined in `src/common/schemas.py` — REUSE:**
- `Chunk` (id, source, doc_type, statute_no, case_no, title, text, char_range, hash)
- `QAPair` (id, instruction, input, output, source: `"seed"|"synth"|"public"`, cited)
- `Citation` (raw, normalized, kind)
- `Response` (answer, mode, citations, retrieved, latency_ms) — `mode` already widened to include `qlora`/`rag_qlora`

**Existing surfaces to depend on (don't duplicate):**
- `src/serving/model_loader.get_base_model()` — singleton (model, tokenizer)
- `src/eval/citation_checker._STATUTE_RE` — regex used by synth_qa filter
- `src/common/config.load_config()` — YAML + .env loader; `config/default.yaml` already has `model:`, `embedding:`, `retrieval:`, `serving:`, `chunk:`, `ingestion:`, `logging:` blocks
- `src/common/paths.{ROOT, CONFIG_DIR, PROCESSED, ADAPTERS, RUNS}`

---

## File map

**Create:**
- `src/training/__init__.py`
- `src/training/synth_qa.py`
- `src/training/prepare_dataset.py`
- `src/training/train_qlora.py`
- `data/processed/qa_seed.jsonl` (12 hand-curated `QAPair` rows)
- `config/prompts/synth_qa.txt`
- `tests/test_synth_qa.py`
- `tests/test_prepare_dataset.py`
- `tests/test_adapter_hooks.py`
- `tests/test_smoke_train.py` (slow, CUDA-gated)

**Modify:**
- `pyproject.toml` — torch / transformers / bitsandbytes pin updates + `anthropic`, `peft`, `tensorboard` deps
- `Makefile` — `install-cuda-torch` URL bump + new `synth`, `prepare`, `smoke-train` targets
- `src/serving/model_loader.py` — additive: `_State` extension + `attach_adapter` / `detach_adapter` / `current_adapter`
- `config/default.yaml` — add `training:` block (LoRA r/α, lr, etc.) + `synth:` block (model, max_usd, target_pairs)
- `README.md` — flip M3a status row; document `make smoke-train`
- `docs/superpowers/m2-readiness.md` — append "M3a closed" section after M3a tag

---

## Task 1: torch 2.5+ ecosystem upgrade (pre-flight gate)

This task is intentionally a single commit. If anything past §3.2 step 2 breaks irrecoverably, `git revert HEAD` returns the project to a known-good state.

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1.1: Update `pyproject.toml` pins**

Edit `pyproject.toml`:

```toml
[project]
# ...
dependencies = [
    "torch>=2.5,<2.7",
    "transformers>=4.51,<5.0",
    "peft>=0.12",
    "bitsandbytes>=0.43,<0.50; sys_platform != 'darwin'",
    "pyarrow>=21,<24",
    "trl>=0.10",
    "accelerate>=0.34",
    "chromadb>=0.5",
    "FlagEmbedding>=1.2",
    "rank-bm25>=0.2.2",
    "pydantic>=2.7",
    "pyyaml>=6",
    "python-dotenv>=1",
    "requests>=2.32",
    "tqdm>=4.66",
    "gradio>=4,<6",
    "anthropic>=0.40",
    "tensorboard>=2.14",
]

[project.optional-dependencies]
unsloth = [
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "unsloth_zoo",
]
dev = [
    "pytest>=8",
    "pytest-mock>=3.12",
    "ruff>=0.6",
]
```

- [ ] **Step 1.2: Update `Makefile` install-cuda-torch URL**

Edit `Makefile` — replace the `install-cuda-torch` target body:

```makefile
install-cuda-torch:
	python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126 "torch>=2.5,<2.7"
```

- [ ] **Step 1.3: Reinstall the env**

Run in order:
```powershell
.venv/Scripts/python -m pip install -e ".[dev]"
make install-cuda-torch
make install-unsloth
make postinstall
```

Expected: each completes without error. The `install-unsloth` step will also pull `xformers`, `torchao`, `torchvision` — that's fine.

- [ ] **Step 1.4: Verify Unsloth import works**

Run: `.venv/Scripts/python -c "from unsloth import FastLanguageModel; print('unsloth ok')"`
Expected: `unsloth ok`. (Was the broken signal under torch 2.4.)

If this fails: STOP. Report `BLOCKED` with the full traceback. The plan controller will decide between (a) pinning a specific Unsloth commit, (b) falling back to peft + transformers Trainer for the remaining tasks (drops Unsloth's 2x speedup but unblocks training).

- [ ] **Step 1.5: Re-validate VRAM gate**

Run: `.venv/Scripts/python scripts/verify_combined_vram.py`
Expected: `PASS: peak VRAM <X> GB < budget 5.5 GB`. The peak number may differ from the M2 baseline of 3.17 GB because the Unsloth path replaces the transformers fallback; flag if peak ≥ 5.0 GB as a concern (still passing but tighter than before).

- [ ] **Step 1.6: Re-validate the existing test suites**

Run:
```powershell
.venv/Scripts/python -m pytest -m "not slow" -q
.venv/Scripts/python -m pytest -m slow -v
```
Expected: 52 fast + 4 slow all pass. Ruff: `.venv/Scripts/python -m ruff check src tests` clean.

- [ ] **Step 1.7: Commit**

```powershell
git add pyproject.toml Makefile
git commit -m "$(cat <<'EOF'
build(deps): torch 2.5+ ecosystem upgrade for Unsloth-enabled QLoRA

Bumps torch (2.4 -> >=2.5,<2.7) to satisfy unsloth's transitive torchao
requirement, with the cascading transformers (>=4.51,<5.0), bitsandbytes
(<0.50), CUDA wheel (cu124 -> cu126), and adds anthropic + tensorboard
+ peft as direct deps for the upcoming training subsystem.

Verified: scripts/verify_combined_vram.py PASS, full slow suite green,
unsloth import works, FlagEmbedding postinstall patch still applies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: qa_seed + synth prompt template

12 hand-curated Q&A pairs in `data/processed/qa_seed.jsonl` and the system prompt the synth_qa.py module will use to expand them.

**Files:**
- Create: `data/processed/qa_seed.jsonl`
- Create: `config/prompts/synth_qa.txt`
- Modify: `.gitignore` — re-include the seed file

- [ ] **Step 2.1: Adjust `.gitignore` to track the seed file**

The current `.gitignore` line `data/processed/*` excludes the seed. Add an exception:

```
# Existing line: data/processed/*
# Existing line: !data/processed/.gitkeep
!data/processed/qa_seed.jsonl
```

Place the new line after the existing `!data/processed/.gitkeep` line. Verify with `git check-ignore -v data/processed/qa_seed.jsonl` — expect no output (= NOT ignored).

- [ ] **Step 2.2: Write the seed Q&A file**

Create `data/processed/qa_seed.jsonl` with exactly 12 lines, each a JSON-serialized `QAPair`. The 12 pairs cover civil-law fundamentals: 임대차 정의 (618), 임대차 존속기간 (619), 매매 의의 (563), 채권의 의의 (3장 일반), 손해배상 (390, 393), 시효 (162), 동시이행항변권 (536), 해제 (543), 변제 (460), 보증 (428), 위임 (680), 사용대차 (609).

```jsonl
{"id":"seed-001","instruction":"민법상 임대차의 정의와 성립요건을 설명해 주세요.","input":null,"output":"임대차는 [민법 제618조]에 따라 당사자 일방이 상대방에게 목적물을 사용·수익하게 할 것을 약정하고, 상대방이 이에 대하여 차임을 지급할 것을 약정함으로써 효력이 생기는 계약입니다. 따라서 ①목적물 인도 약정과 ②차임 지급 약정이라는 두 가지 합의가 핵심 성립요건입니다.","source":"seed","cited":["618"]}
{"id":"seed-002","instruction":"처분권한 없는 자가 한 임대차의 효력은 어떻게 됩니까?","input":null,"output":"[민법 제619조]에 따르면 처분의 능력 또는 권한 없는 자가 임대차를 하는 경우에는 그 임대차의 존속기간이 동조 각호의 기간을 넘지 못하도록 제한됩니다. 즉 임대차 자체가 무효가 되는 것이 아니라 존속기간이 법정 한도로 단축되는 효과가 있습니다.","source":"seed","cited":["619"]}
{"id":"seed-003","instruction":"매매계약의 의의를 설명해 주세요.","input":null,"output":"매매는 [민법 제563조]에 따라 당사자 일방이 재산권을 상대방에게 이전할 것을 약정하고 상대방이 그 대금을 지급할 것을 약정함으로써 효력이 생기는 유상·쌍무·낙성 계약입니다.","source":"seed","cited":["563"]}
{"id":"seed-004","instruction":"채무불이행으로 인한 손해배상의 일반원칙은 무엇입니까?","input":null,"output":"[민법 제390조]는 채무자가 채무의 내용에 좇은 이행을 하지 아니한 때에는 채권자는 손해배상을 청구할 수 있다고 정하고 있습니다. 다만 채무자에게 고의·과실 등 귀책사유가 없으면 책임을 면합니다.","source":"seed","cited":["390"]}
{"id":"seed-005","instruction":"손해배상의 범위는 어떻게 정해집니까?","input":null,"output":"[민법 제393조]에 따라 채무불이행으로 인한 손해배상은 통상의 손해를 한도로 하고, 특별한 사정으로 인한 손해는 채무자가 그 사정을 알았거나 알 수 있었을 때에 한하여 배상의 책임이 있습니다(상당인과관계설).","source":"seed","cited":["393"]}
{"id":"seed-006","instruction":"채권의 일반적 소멸시효 기간은 몇 년입니까?","input":null,"output":"[민법 제162조] 제1항은 채권은 10년간 행사하지 아니하면 소멸시효가 완성한다고 정합니다(상사채권 등 다른 법률에 특별한 정함이 있는 경우는 제외). 채권 외의 재산권은 동조 제2항에 따라 20년의 소멸시효에 걸립니다.","source":"seed","cited":["162"]}
{"id":"seed-007","instruction":"동시이행항변권이란 무엇인가요?","input":null,"output":"[민법 제536조]에 따라 쌍무계약의 당사자 일방은 상대방이 그 채무이행을 제공할 때까지 자기의 채무이행을 거절할 수 있는데, 이를 동시이행항변권이라 합니다. 다만 상대방의 채무가 변제기에 있지 아니한 경우에는 적용되지 않습니다.","source":"seed","cited":["536"]}
{"id":"seed-008","instruction":"계약 해제의 효과를 알려주세요.","input":null,"output":"[민법 제543조] 이하의 규정에 따라 계약이 해제되면 당사자는 각각 상대방에 대하여 원상회복의 의무를 부담하며, 손해배상의 청구도 그 효력에 영향이 없습니다(제551조). 즉 해제는 계약을 소급적으로 소멸시키되 손해배상청구권은 별도로 존속합니다.","source":"seed","cited":["543","551"]}
{"id":"seed-009","instruction":"변제는 누가 어떻게 할 수 있습니까?","input":null,"output":"[민법 제460조]에 따르면 변제자는 채무의 내용에 좇은 현실의 제공을 하여야 하지만, 채권자가 미리 변제받기를 거절하거나 채무의 이행에 채권자의 행위를 요하는 경우에는 변제 준비를 완료하고 그 사실을 통지하여 수령을 최고함으로써 족합니다(구두의 제공).","source":"seed","cited":["460"]}
{"id":"seed-010","instruction":"보증채무의 부종성이란 무엇인가요?","input":null,"output":"[민법 제428조]에서 도출되는 보증채무의 부종성은 ①성립의 부종성(주채무 없으면 보증채무도 없음), ②내용의 부종성(주채무보다 무거운 보증은 그 한도에서 감축), ③소멸의 부종성(주채무 소멸 시 보증채무도 소멸)을 의미합니다.","source":"seed","cited":["428"]}
{"id":"seed-011","instruction":"위임계약의 본질적 특징을 설명해 주세요.","input":null,"output":"[민법 제680조]에 따라 위임은 당사자 일방이 상대방에 대하여 사무의 처리를 위탁하고 상대방이 이를 승낙함으로써 효력이 생깁니다. 위임은 원칙적으로 무상·편무 계약이며, 수임인은 [민법 제681조]에 따라 선량한 관리자의 주의로 위임사무를 처리할 의무를 부담합니다.","source":"seed","cited":["680","681"]}
{"id":"seed-012","instruction":"사용대차와 임대차의 차이점은 무엇인가요?","input":null,"output":"[민법 제609조]의 사용대차는 당사자 일방이 상대방에게 무상으로 사용·수익하게 할 것을 약정하고 상대방이 사용·수익한 후 그 물건을 반환할 것을 약정함으로써 효력이 생깁니다. [민법 제618조]의 임대차와 달리 차임 지급 약정이 없다는 점이 본질적 차이입니다.","source":"seed","cited":["609","618"]}
```

(Lines must each be valid JSON — verify with `.venv/Scripts/python -c "from src.common.schemas import QAPair; import json; [QAPair.model_validate_json(l) for l in open('data/processed/qa_seed.jsonl', encoding='utf-8')]; print('all 12 valid')"`. Expected: `all 12 valid`.)

- [ ] **Step 2.3: Write the synth_qa system prompt**

Create `config/prompts/synth_qa.txt`:

```text
당신은 한국 민법 시험 출제 위원입니다. 아래 [참고 조문] 발췌만을 근거로 학습용 Q&A 쌍 2개를 생성하세요.

요구사항:
1. 질문은 자연스러운 한국어 평서/의문문이며, 변호사·법학과 학생이 실제로 물어볼 만한 형태일 것.
2. 답변은 반드시 [민법 제○○조] 형식의 인용을 본문에 포함할 것. 인용은 [참고 조문]의 조문번호 중 하나여야 함.
3. 답변은 50자 이상, 400자 이하의 한국어로 간결하게.
4. 추측·창작 금지: [참고 조문]에 없는 내용은 답변에 포함하지 말 것.
5. 출력 형식은 아래 JSON 스키마(한 줄 JSON 객체 2개)만:
   {"instruction": "<질문>", "output": "<답변>", "cited": ["<조문번호>", ...]}

예시(few-shot):
{seed_examples}

[참고 조문]
{chunks}

위 [참고 조문]을 근거로 한 Q&A 2개를 위 JSON 형식으로만 출력하세요. 다른 텍스트는 절대 출력하지 마세요.
```

- [ ] **Step 2.4: Commit**

```powershell
git add data/processed/qa_seed.jsonl config/prompts/synth_qa.txt .gitignore
git commit -m "$(cat <<'EOF'
data(training): 12-pair Korean civil-law QA seed + synth prompt template

12 hand-curated QAPair rows covering 임대차/매매/채권/손해배상/시효/
동시이행항변권/해제/변제/보증/위임/사용대차 fundamentals. Each cites at
least one 민법 article so the synth_qa filter (citation_checker
_STATUTE_RE) accepts them. Seed feeds few-shot expansion in
src/training/synth_qa.py.

The prompt template uses {seed_examples} and {chunks} placeholders that
synth_qa.run_synth fills per batch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: synth_qa.py post-processing TDD (filters + cost cap)

Pure-logic surface — citation/length/dedupe filters and the cost-cap helper. Mocks the Anthropic SDK.

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/synth_qa.py` (filter functions only; full `run_synth` lands in Task 4)
- Test: `tests/test_synth_qa.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/test_synth_qa.py`:

```python
"""Unit tests for synth_qa post-processing. No network."""

from __future__ import annotations

import pytest

from src.training.synth_qa import (
    filter_synthesized,
    estimate_call_cost,
    SynthRow,
)


def _row(instruction: str, output: str) -> SynthRow:
    return SynthRow(instruction=instruction, output=output, cited=[])


def test_filter_drops_rows_without_statute_citation():
    rows = [
        _row("질문", "근거가 [민법 제618조]에 있다." * 3),  # keep
        _row("질문", "근거 없이 그냥 답한다." * 3),  # drop — no citation
    ]
    kept, drops = filter_synthesized(rows, seen_hashes=set())
    assert len(kept) == 1
    assert kept[0].cited == ["618"]
    assert drops["no_citation"] == 1


def test_filter_drops_rows_under_50_chars():
    rows = [_row("질문", "[민법 제618조] 짧음.")]  # under 50 chars
    kept, drops = filter_synthesized(rows, seen_hashes=set())
    assert len(kept) == 0
    assert drops["too_short"] == 1


def test_filter_dedupes_by_instruction_output_hash():
    rows = [
        _row("질문", "[민법 제618조] " + "ㄱ" * 60),
        _row("질문", "[민법 제618조] " + "ㄱ" * 60),  # exact dup
    ]
    kept, _ = filter_synthesized(rows, seen_hashes=set())
    assert len(kept) == 1


def test_filter_extracts_multiple_citations():
    rows = [_row("질문", "[민법 제618조]와 [민법 제619조]에 따라 " + "ㄱ" * 30)]
    kept, _ = filter_synthesized(rows, seen_hashes=set())
    assert kept[0].cited == ["618", "619"]


def test_filter_respects_external_seen_hashes():
    """Caller's seen set causes dedup against earlier batches."""
    row = _row("질문", "[민법 제618조] " + "ㄱ" * 60)
    kept_1, _ = filter_synthesized([row], seen_hashes=set())
    seen = {kept_1[0].hash}
    kept_2, drops = filter_synthesized([row], seen_hashes=seen)
    assert kept_2 == []
    assert drops["duplicate"] == 1


def test_estimate_call_cost_sonnet_pricing():
    # Sonnet 4.6 at $3/M input + $15/M output
    cost = estimate_call_cost(input_tokens=1000, output_tokens=500, model="sonnet")
    # 1000 * 3/1_000_000 + 500 * 15/1_000_000 = 0.003 + 0.0075 = 0.0105
    assert cost == pytest.approx(0.0105)


def test_estimate_call_cost_opus_pricing():
    # Opus 4.7 at $15/M input + $75/M output
    cost = estimate_call_cost(input_tokens=1000, output_tokens=500, model="opus")
    # 1000 * 15/1_000_000 + 500 * 75/1_000_000 = 0.015 + 0.0375 = 0.0525
    assert cost == pytest.approx(0.0525)


def test_estimate_call_cost_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        estimate_call_cost(input_tokens=1, output_tokens=1, model="haiku")  # type: ignore[arg-type]
```

- [ ] **Step 3.2: Run the failing tests**

Run: `.venv/Scripts/python -m pytest tests/test_synth_qa.py -v`
Expected: ImportError (`src.training.synth_qa` module does not exist).

- [ ] **Step 3.3: Implement `src/training/__init__.py`**

```python
"""Training subsystem: synth_qa, prepare_dataset, train_qlora."""
```

- [ ] **Step 3.4: Implement `src/training/synth_qa.py` (filters + cost-cap only)**

```python
"""Synthesize Korean civil-law Q&A from chunks via Claude few-shot expansion.

Task 3 (this commit): pure post-processing surface — citation/length/dedupe
filters and the cost-cap estimator. Task 4 lands run_synth and the API call.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from src.eval.citation_checker import _STATUTE_RE

ModelChoice = Literal["sonnet", "opus"]

# Static price tables (USD per million tokens). Update when Anthropic changes prices.
_PRICES: dict[ModelChoice, tuple[float, float]] = {
    # (input $/M, output $/M)
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}

_MIN_OUTPUT_CHARS = 50


@dataclass
class SynthRow:
    """A candidate synthesized Q&A row, before filter pass."""

    instruction: str
    output: str
    cited: list[str] = field(default_factory=list)
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha1(
                f"{self.instruction}\x00{self.output}".encode("utf-8")
            ).hexdigest()[:16]


def estimate_call_cost(*, input_tokens: int, output_tokens: int, model: ModelChoice) -> float:
    """Approximate USD cost for one Claude API call. Static pricing table."""
    if model not in _PRICES:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(_PRICES)}")
    in_price, out_price = _PRICES[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def _extract_cited(output: str) -> list[str]:
    """Extract normalized statute numbers ('618', '643의2') in the order they appear."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _STATUTE_RE.finditer(output):
        article, sub = m.group(1), m.group(2)
        norm = f"{article}의{sub}" if sub else article
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def filter_synthesized(
    rows: list[SynthRow], *, seen_hashes: set[str]
) -> tuple[list[SynthRow], dict[str, int]]:
    """Apply the four post-processing filters and return (kept, drop_reasons).

    Mutates each kept row's `cited` field with extracted statute numbers.
    Updates `seen_hashes` in place so consecutive batches dedup against history.
    """
    drops: dict[str, int] = {"no_citation": 0, "too_short": 0, "duplicate": 0}
    kept: list[SynthRow] = []
    for r in rows:
        cited = _extract_cited(r.output)
        if not cited:
            drops["no_citation"] += 1
            continue
        if len(r.output) < _MIN_OUTPUT_CHARS:
            drops["too_short"] += 1
            continue
        if r.hash in seen_hashes:
            drops["duplicate"] += 1
            continue
        seen_hashes.add(r.hash)
        r.cited = cited
        kept.append(r)
    return kept, drops
```

- [ ] **Step 3.5: Run the tests to confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_synth_qa.py -v`
Expected: 8 passed.

- [ ] **Step 3.6: Run the full fast suite**

Run: `.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 60 passed (52 prior + 8 new).

- [ ] **Step 3.7: Commit**

```powershell
git add src/training/__init__.py src/training/synth_qa.py tests/test_synth_qa.py
git commit -m "$(cat <<'EOF'
feat(training): synth_qa filters + cost estimator (TDD)

Pure post-processing surface for the upcoming Claude few-shot expansion:
citation regex check (reuses citation_checker._STATUTE_RE), 50-char min,
hash-based dedup with caller-owned seen set, and a static pricing table
for Sonnet 4.6 / Opus 4.7. Task 4 wires the actual SDK call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: synth_qa.py Claude integration + cache + `run_synth`

Wires the Anthropic SDK + on-disk cache + the cost-capped main loop. Tested via integration with mocked SDK in the smoke test (Task 7); no new fast tests here because the filters were locked in Task 3.

**Files:**
- Modify: `src/training/synth_qa.py` (extend; do not rewrite filter section)
- Modify: `Makefile` — add `synth` target
- Modify: `config/default.yaml` — add `synth:` block

- [ ] **Step 4.1: Extend `config/default.yaml` with the `synth:` block**

Append after the `serving:` block (before `logging:`):

```yaml
synth:
  model: sonnet           # or opus
  target_pairs: 200       # smoke target; M3b will use 1500
  max_usd: 3.0            # smoke cap; M3b will use 15.0
  chunks_per_batch: 4
  pairs_per_batch: 2
  seeds_per_prompt: 3
```

- [ ] **Step 4.2: Add the Makefile `synth` target**

Edit `Makefile` — add a new target after `embed:`:

```makefile
synth:
	python -m src.training.synth_qa
```

Also add `synth` to the `.PHONY` list at the top of the Makefile.

- [ ] **Step 4.3: Extend `src/training/synth_qa.py` with `run_synth`**

Append to `src/training/synth_qa.py` (after the existing functions; do not modify what's already there):

```python
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from src.common import paths
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.schemas import Chunk, QAPair

log = get_logger("synth_qa")


def _load_seed(path: Path) -> list[QAPair]:
    return [QAPair.model_validate_json(line) for line in path.open(encoding="utf-8") if line.strip()]


def _load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.open(encoding="utf-8") if line.strip()]


def _format_seed_examples(seeds: list[QAPair]) -> str:
    """Render seeds as one JSON object per line for the few-shot prompt."""
    lines = []
    for s in seeds:
        obj = {"instruction": s.instruction, "output": s.output, "cited": s.cited}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.statute_no or c.id}] {c.text}" for c in chunks)


def _build_prompt(template: str, seeds: list[QAPair], chunks: list[Chunk]) -> str:
    return template.replace("{seed_examples}", _format_seed_examples(seeds)).replace(
        "{chunks}", _format_chunks(chunks)
    )


def _batch_chunks(
    all_chunks: list[Chunk], chunks_per_batch: int, rng: random.Random
) -> Iterator[list[Chunk]]:
    """Yield groups of chunks that are statute-adjacent (sort by statute_no, sample windows)."""
    sorted_chunks = sorted(all_chunks, key=lambda c: (c.statute_no or "", c.id))
    if len(sorted_chunks) < chunks_per_batch:
        # Pad by repetition to keep run_synth resilient on tiny corpora.
        sorted_chunks = sorted_chunks * ((chunks_per_batch // max(1, len(sorted_chunks))) + 1)
    while True:
        start = rng.randint(0, len(sorted_chunks) - chunks_per_batch)
        yield sorted_chunks[start : start + chunks_per_batch]


_JSON_LINE_RE = re.compile(r"\{[^{}]*\"instruction\"[^{}]*\}")


def _parse_response(text: str) -> list[SynthRow]:
    """Extract one or more JSON-line `{instruction, output, cited}` objects from the response."""
    out: list[SynthRow] = []
    for m in _JSON_LINE_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "instruction" in obj and "output" in obj:
            out.append(SynthRow(instruction=obj["instruction"], output=obj["output"]))
    return out


def _cache_path(cache_dir: Path, batch_signature: str) -> Path:
    h = hashlib.sha1(batch_signature.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{h}.json"


def run_synth(
    *,
    target_pairs: int,
    model: ModelChoice = "sonnet",
    max_usd: float = 3.0,
    cache_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Resumable synthesis loop. Returns the qa_train.jsonl path."""
    import anthropic  # local: avoid network-import cost on the fast suite

    cfg = load_config()
    seed_path = paths.PROCESSED / "qa_seed.jsonl"
    chunks_path = paths.PROCESSED / "chunks.jsonl"
    out_path = paths.PROCESSED / "qa_train.jsonl"
    template_path = paths.CONFIG_DIR / "prompts" / "synth_qa.txt"

    if not chunks_path.exists():
        raise SystemExit(
            f"{chunks_path} missing — run `make ingest` then `python -m src.ingestion.chunk` first."
        )
    seeds = _load_seed(seed_path)
    chunks = _load_chunks(chunks_path)
    template = template_path.read_text(encoding="utf-8")
    cache_dir = cache_dir or (paths.CACHE / "synth")
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_model_id = {"sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-7"}[model]
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    batches = _batch_chunks(chunks, cfg["synth"]["chunks_per_batch"], rng)

    seen_hashes: set[str] = set()
    written = 0
    running_cost = 0.0
    drop_totals: dict[str, int] = {"no_citation": 0, "too_short": 0, "duplicate": 0}

    with out_path.open("w", encoding="utf-8") as out_f:
        for batch in batches:
            if written >= target_pairs:
                break
            sample_seeds = rng.sample(seeds, k=min(cfg["synth"]["seeds_per_prompt"], len(seeds)))
            prompt = _build_prompt(template, sample_seeds, batch)
            sig = f"{api_model_id}|{prompt}"
            cache_file = _cache_path(cache_dir, sig)

            if cache_file.exists():
                resp_text = json.loads(cache_file.read_text(encoding="utf-8"))["text"]
                cost = 0.0  # already paid
                log.info("cache hit %s", cache_file.name)
            else:
                est = estimate_call_cost(input_tokens=len(prompt) // 4, output_tokens=400, model=model)
                if running_cost + est > max_usd:
                    log.warning(
                        "Cost cap %.2f reached (running %.2f + est %.2f); aborting.",
                        max_usd,
                        running_cost,
                        est,
                    )
                    break
                msg = client.messages.create(
                    model=api_model_id,
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                resp_text = "".join(b.text for b in msg.content if hasattr(b, "text"))
                cost = estimate_call_cost(
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    model=model,
                )
                running_cost += cost
                cache_file.write_text(
                    json.dumps({"text": resp_text, "cost": cost}, ensure_ascii=False),
                    encoding="utf-8",
                )

            rows = _parse_response(resp_text)
            kept, drops = filter_synthesized(rows, seen_hashes=seen_hashes)
            for k, v in drops.items():
                drop_totals[k] += v
            for r in kept:
                pair = QAPair(
                    id=f"synth-{written:04d}",
                    instruction=r.instruction,
                    output=r.output,
                    source="synth",
                    cited=r.cited,
                )
                out_f.write(pair.model_dump_json() + "\n")
                written += 1
                if written >= target_pairs:
                    break
            log.info(
                "batch: kept %d / parsed %d; total written %d / target %d; cost $%.4f",
                len(kept),
                len(rows),
                written,
                target_pairs,
                running_cost,
            )

    log.info(
        "synth done: %d pairs to %s (cost $%.2f); drops=%s",
        written,
        out_path,
        running_cost,
        drop_totals,
    )
    return out_path


def main() -> int:
    cfg = load_config()
    s = cfg["synth"]
    run_synth(target_pairs=s["target_pairs"], model=s["model"], max_usd=s["max_usd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4.4: Spot-check the new module imports**

Run:
```powershell
.venv/Scripts/python -c "from src.training.synth_qa import run_synth, _build_prompt, _parse_response, SynthRow; print('synth_qa surface ok')"
```
Expected: `synth_qa surface ok` (no API call yet — the import does not call Anthropic).

- [ ] **Step 4.5: Verify the YAML loads**

```powershell
.venv/Scripts/python -c "from src.common.config import load_config; cfg = load_config(); print('synth:', cfg['synth'])"
```
Expected: `synth: {'model': 'sonnet', 'target_pairs': 200, 'max_usd': 3.0, 'chunks_per_batch': 4, 'pairs_per_batch': 2, 'seeds_per_prompt': 3}`.

- [ ] **Step 4.6: Run the fast suite (no regression)**

`.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 60 passed.

- [ ] **Step 4.7: Commit**

```powershell
git add src/training/synth_qa.py Makefile config/default.yaml
git commit -m "$(cat <<'EOF'
feat(training): synth_qa run_synth — Claude few-shot + on-disk cache

Resumable synthesis loop driven by config/default.yaml's synth: block.
Builds the prompt from config/prompts/synth_qa.txt, samples chunks
clustered by statute_no, calls Anthropic SDK (Sonnet default), caches
each raw response under data/cache/synth/{hash}.json, applies the Task 3
filters, and writes data/processed/qa_train.jsonl until target_pairs or
max_usd is reached. The fast suite stays untouched — the SDK is only
exercised by the slow smoke test in Task 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: prepare_dataset.py (TDD)

Pure tokenizer + masking surface. Uses the real Qwen2.5 tokenizer because tokenization is fast (<1 s) and the alternative (mocking BPE) would lose all the meaningful coverage.

**Files:**
- Create: `src/training/prepare_dataset.py`
- Test: `tests/test_prepare_dataset.py`
- Modify: `Makefile` — add `prepare` target

- [ ] **Step 5.1: Write the failing tests**

Create `tests/test_prepare_dataset.py`:

```python
"""Unit tests for prepare_dataset. Uses the real Qwen2.5 tokenizer (fast)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.schemas import QAPair


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load Qwen2.5 tokenizer once per module (cheap; <1 s cold)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def _write_qa_jsonl(path: Path, pairs: list[QAPair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")


def _make_pair(idx: int, instr: str, out: str) -> QAPair:
    return QAPair(id=f"t-{idx}", instruction=instr, output=out, source="seed", cited=[])


def test_prepare_emits_train_and_val_files(tmp_path, qwen_tokenizer):
    from src.training.prepare_dataset import prepare

    pairs = [_make_pair(i, f"질문{i}", f"답변{i}") for i in range(20)]
    _write_qa_jsonl(tmp_path / "qa.jsonl", pairs)
    train_path, val_path = prepare(
        qa_path=tmp_path / "qa.jsonl",
        out_dir=tmp_path,
        tokenizer=qwen_tokenizer,
        max_seq=512,
    )
    assert train_path.name == "qa_train_tokenized.jsonl"
    assert val_path.name == "qa_val_tokenized.jsonl"
    assert train_path.exists() and val_path.exists()


def test_prepare_split_is_90_10(tmp_path, qwen_tokenizer):
    from src.training.prepare_dataset import prepare

    pairs = [_make_pair(i, f"질문{i}", f"답변{i}") for i in range(100)]
    _write_qa_jsonl(tmp_path / "qa.jsonl", pairs)
    tp, vp = prepare(
        qa_path=tmp_path / "qa.jsonl", out_dir=tmp_path, tokenizer=qwen_tokenizer, max_seq=512
    )
    train_count = sum(1 for _ in tp.open(encoding="utf-8"))
    val_count = sum(1 for _ in vp.open(encoding="utf-8"))
    assert train_count == 90
    assert val_count == 10
    assert train_count + val_count == 100


def test_prepare_split_is_deterministic(tmp_path, qwen_tokenizer):
    from src.training.prepare_dataset import prepare

    pairs = [_make_pair(i, f"질문{i}", f"답변{i}") for i in range(20)]
    _write_qa_jsonl(tmp_path / "qa.jsonl", pairs)
    a_train, _ = prepare(
        qa_path=tmp_path / "qa.jsonl",
        out_dir=tmp_path / "a",
        tokenizer=qwen_tokenizer,
        max_seq=512,
        seed=42,
    )
    b_train, _ = prepare(
        qa_path=tmp_path / "qa.jsonl",
        out_dir=tmp_path / "b",
        tokenizer=qwen_tokenizer,
        max_seq=512,
        seed=42,
    )
    a_lines = a_train.read_text(encoding="utf-8")
    b_lines = b_train.read_text(encoding="utf-8")
    assert a_lines == b_lines


def test_prepare_emits_input_ids_and_labels_with_instruction_masked(tmp_path, qwen_tokenizer):
    from src.training.prepare_dataset import prepare

    pair = _make_pair(0, "임대차의 정의는?", "임대차는 [민법 제618조]에 따라 ...")
    _write_qa_jsonl(tmp_path / "qa.jsonl", [pair] * 10)  # need >= 10 for non-empty val
    tp, _ = prepare(
        qa_path=tmp_path / "qa.jsonl", out_dir=tmp_path, tokenizer=qwen_tokenizer, max_seq=512
    )
    row = json.loads(next(tp.open(encoding="utf-8")))
    assert "input_ids" in row
    assert "attention_mask" in row
    assert "labels" in row
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    # Some labels are -100 (instruction tokens), some are real ids (assistant tokens)
    assert any(l == -100 for l in row["labels"])
    assert any(l > 0 for l in row["labels"])


def test_prepare_truncates_over_max_seq(tmp_path, qwen_tokenizer):
    from src.training.prepare_dataset import prepare

    long_output = "임대차는 [민법 제618조]에 따라 " + "가" * 5000
    pair = _make_pair(0, "Q?", long_output)
    _write_qa_jsonl(tmp_path / "qa.jsonl", [pair] * 10)
    tp, _ = prepare(
        qa_path=tmp_path / "qa.jsonl", out_dir=tmp_path, tokenizer=qwen_tokenizer, max_seq=128
    )
    row = json.loads(next(tp.open(encoding="utf-8")))
    assert len(row["input_ids"]) <= 128
```

- [ ] **Step 5.2: Run the failing tests**

`.venv/Scripts/python -m pytest tests/test_prepare_dataset.py -v`
Expected: ImportError.

- [ ] **Step 5.3: Implement `src/training/prepare_dataset.py`**

```python
"""Tokenize QAPair rows into chat-template input_ids + label-masked targets.

Output format (JSONL): {input_ids: list[int], attention_mask: list[int], labels: list[int]}
where labels are input_ids with the instruction-side tokens replaced by -100
(ignored by cross-entropy).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.common.schemas import QAPair

log = get_logger("prepare_dataset")


def _format_messages(p: QAPair) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": p.instruction},
        {"role": "assistant", "content": p.output},
    ]


def _encode_with_label_mask(p: QAPair, tokenizer: Any, max_seq: int) -> dict[str, list[int]]:
    """Tokenize the full chat and mask everything before the assistant turn."""
    full_messages = _format_messages(p)
    full_ids: list[int] = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    # Re-tokenize with only the user turn + generation prompt to find prefix length.
    prefix_ids: list[int] = tokenizer.apply_chat_template(
        full_messages[:1], tokenize=True, add_generation_prompt=True
    )
    prefix_len = len(prefix_ids)
    if prefix_len > len(full_ids):
        # Defensive: should not happen when the assistant turn is non-empty.
        prefix_len = len(full_ids)
    input_ids = full_ids[:max_seq]
    attention_mask = [1] * len(input_ids)
    labels = [
        -100 if i < prefix_len else input_ids[i] for i in range(len(input_ids))
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _round_up_64(n: int) -> int:
    return ((n + 63) // 64) * 64


def prepare(
    *,
    qa_path: Path,
    out_dir: Path,
    tokenizer: Any,
    max_seq: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Tokenize, split 90/10, and write JSONL outputs.

    If `max_seq` is None, log p50/p95/max token-length and use rounded p95.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        QAPair.model_validate_json(line)
        for line in qa_path.open(encoding="utf-8")
        if line.strip()
    ]
    if not pairs:
        raise ValueError(f"{qa_path} has no rows")

    # First pass: token-length stats (no truncation yet)
    raw_lens: list[int] = []
    for p in pairs:
        ids = tokenizer.apply_chat_template(
            _format_messages(p), tokenize=True, add_generation_prompt=False
        )
        raw_lens.append(len(ids))
    raw_lens.sort()
    p50 = raw_lens[len(raw_lens) // 2]
    p95 = raw_lens[max(0, math.ceil(len(raw_lens) * 0.95) - 1)]
    p_max = raw_lens[-1]
    log.info("token-length p50=%d p95=%d max=%d (n=%d)", p50, p95, p_max, len(pairs))
    if max_seq is None:
        max_seq = _round_up_64(p95)
        log.info("max_seq auto-set to %d (rounded-up p95)", max_seq)

    # Second pass: tokenize + label-mask
    rows = [_encode_with_label_mask(p, tokenizer, max_seq) for p in pairs]

    # Deterministic 90/10 split
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split_at = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split_at], indices[split_at:]

    train_path = out_dir / "qa_train_tokenized.jsonl"
    val_path = out_dir / "qa_val_tokenized.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for i in train_idx:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for i in val_idx:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    log.info("prepare wrote %d train + %d val to %s", len(train_idx), len(val_idx), out_dir)
    return train_path, val_path


def main() -> int:
    from transformers import AutoTokenizer

    from src.common import paths

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    qa = paths.PROCESSED / "qa_train.jsonl"
    prepare(qa_path=qa, out_dir=paths.PROCESSED, tokenizer=tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5.4: Add the Makefile `prepare` target**

Edit `Makefile`:

```makefile
prepare:
	python -m src.training.prepare_dataset
```

Add `prepare` to the `.PHONY` list at the top.

- [ ] **Step 5.5: Run the tests to confirm they pass**

`.venv/Scripts/python -m pytest tests/test_prepare_dataset.py -v`
Expected: 5 passed in <5 s (the tokenizer is cached after the first test).

- [ ] **Step 5.6: Run the full fast suite**

`.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 65 passed (60 prior + 5 new).

- [ ] **Step 5.7: Commit**

```powershell
git add src/training/prepare_dataset.py tests/test_prepare_dataset.py Makefile
git commit -m "$(cat <<'EOF'
feat(training): prepare_dataset — Qwen2.5 chat template + label mask + 90/10

Tokenizes QAPair rows with apply_chat_template, masks instruction tokens
to -100 so cross-entropy loss covers only the assistant turn, reports
p50/p95/max token length, and writes train/val JSONL with deterministic
seed=42 shuffle. The label-mask boundary is computed by re-tokenizing
the user turn alone with add_generation_prompt=True; this is the same
trick the trl SFTTrainer uses internally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: model_loader adapter hooks (TDD)

Extends `_State` with `base_model` and `adapter_path`, adds `attach_adapter` / `detach_adapter` / `current_adapter`. Unit tests use a stub `_state.model` mock to avoid GPU.

**Files:**
- Modify: `src/serving/model_loader.py`
- Test: `tests/test_adapter_hooks.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/test_adapter_hooks.py`:

```python
"""Unit tests for model_loader adapter hooks. No GPU; mocks _state.model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _reset_state():
    from src.serving import model_loader as ml

    ml._state.model = None
    ml._state.tokenizer = None
    ml._state.loader = ""
    ml._state.base_model = None
    ml._state.adapter_path = None


def test_attach_adapter_requires_loaded_model():
    from src.serving.model_loader import attach_adapter

    _reset_state()
    with pytest.raises(RuntimeError, match="base model not loaded"):
        attach_adapter("/tmp/fake-adapter")


def test_attach_adapter_swaps_model_and_remembers_base(tmp_path):
    from src.serving import model_loader as ml

    _reset_state()
    base_mock = MagicMock(name="base_model")
    ml._state.model = base_mock
    ml._state.tokenizer = MagicMock(name="tokenizer")

    fake_peft_model = MagicMock(name="peft_model")
    with patch("peft.PeftModel.from_pretrained", return_value=fake_peft_model) as p:
        ml.attach_adapter(tmp_path)

    p.assert_called_once_with(base_mock, str(tmp_path))
    assert ml._state.model is fake_peft_model
    assert ml._state.base_model is base_mock
    assert ml._state.adapter_path == str(tmp_path)
    assert ml.current_adapter() == str(tmp_path)


def test_detach_adapter_restores_base():
    from src.serving import model_loader as ml

    _reset_state()
    base_mock = MagicMock(name="base_model")
    peft_mock = MagicMock(name="peft_model")
    ml._state.model = peft_mock
    ml._state.base_model = base_mock
    ml._state.adapter_path = "/tmp/x"

    ml.detach_adapter()

    assert ml._state.model is base_mock
    assert ml._state.adapter_path is None
    assert ml.current_adapter() is None


def test_detach_adapter_is_noop_when_nothing_attached():
    from src.serving import model_loader as ml

    _reset_state()
    base_mock = MagicMock(name="base_model")
    ml._state.model = base_mock
    ml._state.tokenizer = MagicMock(name="tokenizer")

    # Should not raise
    ml.detach_adapter()
    assert ml._state.model is base_mock
    assert ml._state.adapter_path is None


def test_attach_adapter_double_attach_raises():
    from src.serving import model_loader as ml

    _reset_state()
    base_mock = MagicMock(name="base_model")
    ml._state.model = MagicMock(name="peft_a")
    ml._state.base_model = base_mock
    ml._state.adapter_path = "/tmp/a"

    with pytest.raises(RuntimeError, match="already attached"):
        ml.attach_adapter("/tmp/b")


def test_attach_adapter_accepts_str_or_path(tmp_path):
    from src.serving import model_loader as ml

    _reset_state()
    base_mock = MagicMock(name="base_model")
    ml._state.model = base_mock
    ml._state.tokenizer = MagicMock(name="tokenizer")

    fake_peft_model = MagicMock(name="peft_model")
    with patch("peft.PeftModel.from_pretrained", return_value=fake_peft_model):
        ml.attach_adapter(str(tmp_path))  # string path
    assert ml._state.model is fake_peft_model

    # Reset and try Path
    _reset_state()
    ml._state.model = base_mock
    ml._state.tokenizer = MagicMock(name="tokenizer")
    with patch("peft.PeftModel.from_pretrained", return_value=fake_peft_model):
        ml.attach_adapter(Path(tmp_path))  # Path object
    assert ml._state.model is fake_peft_model
```

- [ ] **Step 6.2: Run the failing tests**

`.venv/Scripts/python -m pytest tests/test_adapter_hooks.py -v`
Expected: AttributeError on `attach_adapter` / `current_adapter` — they don't exist yet.

- [ ] **Step 6.3: Extend `src/serving/model_loader.py`**

Edit `src/serving/model_loader.py`:

In the `_State` dataclass (lines around 26-30), extend:

```python
@dataclass
class _State:
    model: Any = None
    base_model: Any = None
    tokenizer: Any = None
    loader: str = ""
    adapter_path: str | None = None
```

Append these new functions at the bottom of the file:

```python
def attach_adapter(path: str | Path) -> None:
    """Attach a PEFT adapter to the singleton model.

    Idempotency contract: raises RuntimeError if an adapter is already
    attached. Caller should `detach_adapter()` first to swap.
    """
    from peft import PeftModel  # local import: peft is M3+

    if _state.model is None:
        raise RuntimeError("base model not loaded; call get_base_model() first")
    if _state.adapter_path is not None:
        raise RuntimeError(
            f"adapter already attached at {_state.adapter_path!r}; detach first"
        )
    base = _state.base_model or _state.model
    _state.base_model = base
    _state.model = PeftModel.from_pretrained(base, str(path))
    _state.adapter_path = str(path)
    log.info("attach_adapter: %s", _state.adapter_path)


def detach_adapter() -> None:
    """Restore the base model. Noop if no adapter is currently attached."""
    if _state.adapter_path is None:
        return
    if _state.base_model is not None:
        _state.model = _state.base_model
    _state.adapter_path = None
    log.info("detach_adapter: restored base")


def current_adapter() -> str | None:
    """Diagnostic: the currently attached adapter path, or None."""
    return _state.adapter_path
```

Also add `from pathlib import Path` to the imports at the top of the file (if not already present).

- [ ] **Step 6.4: Run the tests to confirm they pass**

`.venv/Scripts/python -m pytest tests/test_adapter_hooks.py -v`
Expected: 6 passed.

- [ ] **Step 6.5: Run the full fast suite**

`.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 71 passed (65 prior + 6 new).

- [ ] **Step 6.6: Re-run the slow singleton test**

The Task 1 slow test asserts singleton stability, which now must coexist with the new fields. Run:

`.venv/Scripts/python -m pytest tests/test_serving_integration.py::test_get_base_model_is_singleton -m slow -v`
Expected: PASS.

- [ ] **Step 6.7: Commit**

```powershell
git add src/serving/model_loader.py tests/test_adapter_hooks.py
git commit -m "$(cat <<'EOF'
feat(serving): model_loader.attach_adapter / detach_adapter / current_adapter

Additive extension over the existing singleton: _State gains base_model
(preserved reference for detach) and adapter_path (idempotency check).
attach_adapter swaps PeftModel in, detach_adapter restores. Unit tests
mock _state.model so the 6 new tests stay in the fast suite. Singleton
slow test continues to pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: train_qlora.py + smoke gate + slow integration test

The training code, the CLI smoke knob, and one end-to-end slow test that exercises the full pipeline with a mocked Claude SDK.

**Files:**
- Create: `src/training/train_qlora.py`
- Create: `tests/test_smoke_train.py`
- Modify: `Makefile` — add `smoke-train` target
- Modify: `config/default.yaml` — add `training:` block

- [ ] **Step 7.1: Add the `training:` block to `config/default.yaml`**

Append after the `synth:` block (before `logging:`):

```yaml
training:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  batch_size: 1
  grad_accum: 8
  optim: adamw_8bit
  full_epochs: 2
  full_max_seq: 2048
  smoke_epochs: 1
  smoke_max_seq: 1024
  smoke_max_steps: 25
  save_steps: 100
  save_total_limit: 2
  loss_decrease_threshold: 0.05  # smoke gate: final < first * (1 - threshold)
```

- [ ] **Step 7.2: Implement `src/training/train_qlora.py`**

Create `src/training/train_qlora.py`:

```python
"""QLoRA training over Qwen2.5-3B 4-bit via Unsloth.

Smoke vs full is a CLI flag that flips a small set of hyperparameters defined
in config.training. JSONL training log lands at runs/{name}/train_log.jsonl
in addition to the standard TensorBoard logs.

Run:
    python -m src.training.train_qlora --smoke
    python -m src.training.train_qlora             # full
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common import paths
from src.common.config import load_config
from src.common.logging import get_logger

log = get_logger("train_qlora")


class _JsonlLossCallback:
    """transformers.TrainerCallback that writes one line per logging step."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if not logs or "loss" not in logs:
            return
        row = {
            "step": state.global_step,
            "epoch": logs.get("epoch"),
            "loss": logs["loss"],
            "lr": logs.get("learning_rate"),
            "time_ms": int((time.time() - self._t0) * 1000),
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def on_train_end(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self._fh.close()


def _load_jsonl_dataset(path: Path):
    """Return a list-of-dicts dataset. Trainer's data_collator handles padding."""
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def train(
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    smoke: bool = False,
    epochs: int | None = None,
    max_seq: int | None = None,
) -> Path:
    """Run QLoRA training. Returns the adapter directory path."""
    from transformers import TrainingArguments, Trainer
    from unsloth import FastLanguageModel  # type: ignore

    cfg = load_config()
    t = cfg["training"]
    epochs = epochs or (t["smoke_epochs"] if smoke else t["full_epochs"])
    max_seq = max_seq or (t["smoke_max_seq"] if smoke else t["full_max_seq"])

    base_id = cfg["model"]["base_id"]
    log.info("Loading %s in 4-bit (max_seq=%d)", base_id, max_seq)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_id,
        max_seq_length=max_seq,
        load_in_4bit=True,
        dtype=None,
    )

    log.info("Wrapping with LoRA r=%d alpha=%d", t["lora_r"], t["lora_alpha"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = _load_jsonl_dataset(train_path)
    val_ds = _load_jsonl_dataset(val_path)

    run_name = f"{'smoke' if smoke else 'full'}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = paths.RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=1,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "tb"),
        max_steps=t["smoke_max_steps"] if smoke else -1,
        bf16=False,
        fp16=True,
    )

    callback = _JsonlLossCallback(run_dir / "train_log.jsonl")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=[callback],
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    log.info("adapter saved to %s", output_dir)
    return output_dir


def _check_smoke_gate(adapter_dir: Path, log_path: Path, threshold: float) -> bool:
    """Returns True if smoke passed."""
    if not (adapter_dir / "adapter_config.json").exists():
        log.error("smoke FAIL: adapter_config.json missing in %s", adapter_dir)
        return False
    if not (adapter_dir / "adapter_model.safetensors").exists():
        log.error("smoke FAIL: adapter_model.safetensors missing in %s", adapter_dir)
        return False
    rows = [json.loads(line) for line in log_path.open(encoding="utf-8") if line.strip()]
    if len(rows) < 2:
        log.error("smoke FAIL: only %d log rows; need >= 2", len(rows))
        return False
    first, last = rows[0]["loss"], rows[-1]["loss"]
    target = first * (1.0 - threshold)
    if last >= target:
        log.error(
            "smoke FAIL: loss %.4f >= target %.4f (first=%.4f, threshold=%.2f)",
            last,
            target,
            first,
            threshold,
        )
        return False
    log.info(
        "smoke PASS: loss %.4f < target %.4f (first=%.4f, decrease=%.1f%%)",
        last,
        target,
        first,
        (1 - last / first) * 100,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="200-pair / 1-epoch smoke run")
    args = parser.parse_args()

    cfg = load_config()
    train_path = paths.PROCESSED / "qa_train_tokenized.jsonl"
    val_path = paths.PROCESSED / "qa_val_tokenized.jsonl"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} missing — run `make synth && make prepare` first."
        )

    name = "qwen2.5-3b-civil-smoke-v0" if args.smoke else "qwen2.5-3b-civil-v1"
    output_dir = paths.ADAPTERS / name
    adapter_dir = train(
        train_path=train_path, val_path=val_path, output_dir=output_dir, smoke=args.smoke
    )

    if args.smoke:
        log_path = max(paths.RUNS.glob("smoke-*/train_log.jsonl"), key=lambda p: p.stat().st_mtime)
        threshold = cfg["training"]["loss_decrease_threshold"]
        if not _check_smoke_gate(adapter_dir, log_path, threshold):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7.3: Add the Makefile `smoke-train` target**

Edit `Makefile`:

```makefile
smoke-train:
	python -m src.training.train_qlora --smoke
```

Add `smoke-train` to the `.PHONY` list at the top, and update the existing `train` target body to drop the `# Plan 3 — not implemented` comment if present:

```makefile
train:
	python -m src.training.train_qlora
```

- [ ] **Step 7.4: Write the slow smoke integration test**

Create `tests/test_smoke_train.py`:

```python
"""End-to-end smoke for the QLoRA pipeline (slow, GPU-required).

Mocks the Anthropic SDK so synth_qa runs offline. Real prepare_dataset
+ Unsloth train (max_steps=2, tiny dataset) + adapter integration check.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _mock_anthropic_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=200, output_tokens=120)
    return msg


@pytest.mark.slow
def test_smoke_pipeline_end_to_end(tmp_path, monkeypatch):
    """synth (mocked Claude) → prepare → train (max_steps=2) → adapter loads + generates."""
    pytest.importorskip("torch")
    pytest.importorskip("unsloth")
    pytest.importorskip("peft")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from src.common import paths as paths_mod
    from src.common.schemas import QAPair
    from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash

    monkeypatch.setattr(paths_mod, "DATA", tmp_path)
    monkeypatch.setattr(paths_mod, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths_mod, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(paths_mod, "ADAPTERS", tmp_path / "adapters")
    monkeypatch.setattr(paths_mod, "RUNS", tmp_path / "runs")
    paths_mod.ensure_dirs()
    paths_mod.PROCESSED.mkdir(parents=True, exist_ok=True)
    (paths_mod.ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Fixture chunks
    fixture = paths_mod.ROOT / "tests" / "fixtures" / "sample_civil_code.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    chunks = dedupe_by_hash(chunk_civil_code(payload, min_chars=10))
    chunks_path = paths_mod.PROCESSED / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    # Mini seed (3 pairs is enough)
    seed_path = paths_mod.PROCESSED / "qa_seed.jsonl"
    with seed_path.open("w", encoding="utf-8") as f:
        for i in range(3):
            p = QAPair(
                id=f"s-{i}",
                instruction=f"질문 {i}",
                output=f"[민법 제61{i}조]에 따른 답변입니다 " + "내" * 60,
                source="seed",
                cited=[f"61{i}"],
            )
            f.write(p.model_dump_json() + "\n")

    # Mock Anthropic SDK to return 5 valid pairs
    fake_response = '\n'.join(
        json.dumps(
            {
                "instruction": f"임대차 질문 {i}",
                "output": f"임대차는 [민법 제618조] 따라 답입니다 " + "다" * 60,
                "cited": ["618"],
            },
            ensure_ascii=False,
        )
        for i in range(5)
    )
    fake_msg = _mock_anthropic_response(fake_response)
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    from src.training.synth_qa import run_synth

    with patch("anthropic.Anthropic", return_value=fake_client):
        qa_path = run_synth(target_pairs=5, model="sonnet", max_usd=1.0)
    assert qa_path.exists()
    n_synth = sum(1 for _ in qa_path.open(encoding="utf-8"))
    assert n_synth >= 1, f"synth produced {n_synth} rows; expected >= 1"

    # Prepare
    from transformers import AutoTokenizer
    from src.training.prepare_dataset import prepare

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    train_path, val_path = prepare(
        qa_path=qa_path, out_dir=paths_mod.PROCESSED, tokenizer=tok, max_seq=512
    )
    assert train_path.exists()

    # Train (override smoke_max_steps to a tiny number for the test)
    from src.training.train_qlora import train as run_train

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    adapter_dir = paths_mod.ADAPTERS / "test-smoke"
    out = run_train(
        train_path=train_path,
        val_path=val_path,
        output_dir=adapter_dir,
        smoke=True,
        epochs=1,
        max_seq=512,
    )
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()

    # Attach + generate
    from src.serving.model_loader import attach_adapter, detach_adapter, get_base_model

    detach_adapter()  # in case a previous test attached one
    model, tokenizer = get_base_model()
    attach_adapter(out)
    inputs = tokenizer.encode("임대차의 의의는?", return_tensors="pt").to(model.device)
    gen = model.generate(inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(gen[0][inputs.shape[1]:], skip_special_tokens=True)
    detach_adapter()

    assert text.strip(), f"adapter produced empty output; raw={text!r}"
    # Korean-ish: at least one Hangul or ascii-non-special char
    assert any("가" <= ch <= "힣" or ch.isascii() for ch in text)
```

- [ ] **Step 7.5: Run the fast suite (no regression)**

`.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 71 passed (no new fast tests this task; `test_smoke_train.py` is `-m slow`).

- [ ] **Step 7.6: Run the slow smoke test**

`.venv/Scripts/python -m pytest tests/test_smoke_train.py -m slow -v -s`

This loads Qwen 4-bit + runs 2 training steps + adapter attach. Expected: PASS in ~60-120 s. If GPU-OOM occurs, reduce `max_seq=512` to `256` in the test and retry.

- [ ] **Step 7.7: Run the real smoke training (make smoke-train)**

This requires `ANTHROPIC_API_KEY` in `.env` and ~$1-3 of API spend.

```powershell
make synth                      # ~5-10 min, $1-3
make prepare                    # ~10 s
make smoke-train                # ~10-20 min on the target GPU
```

Expected output (`make smoke-train` final lines):

```
[train_qlora] adapter saved to ...models/adapters/qwen2.5-3b-civil-smoke-v0
[train_qlora] smoke PASS: loss <X> < target <Y> (first=<Z>, decrease=<N>%)
```

If smoke FAIL: do NOT proceed. Capture the JSONL log, the loss curve, and the adapter dir state, then report back. Common failure modes:
- Loss did not decrease ≥ 5%: training divergence or learning rate too high. Defer to user.
- OOM mid-training: reduce `max_seq` in the smoke knobs; revisit.
- Adapter files missing: training crashed before save; check the run dir for traceback.

- [ ] **Step 7.8: Commit**

```powershell
git add src/training/train_qlora.py tests/test_smoke_train.py Makefile config/default.yaml
git commit -m "$(cat <<'EOF'
feat(training): train_qlora + smoke gate + slow integration test

Unsloth FastLanguageModel.get_peft_model with the LoRA r=16/alpha=32
defaults from config.training, JSONL-per-step callback alongside the
standard TensorBoard logs, and a smoke gate that requires both adapter
artifacts and >=5% loss decrease across the run. tests/test_smoke_train.py
wires the whole pipeline (mocked Claude → real prepare → real Unsloth
train, max_steps=2 → adapter attach + generate) under -m slow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Docs + tag

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/m2-readiness.md`

- [ ] **Step 8.1: Update README status table**

Edit `README.md` — change the M3 row:

```markdown
| **M3 — QLoRA training pipeline** | 🟡 M3a (smoke) ✅ done — tag `m3a-qlora-smoke`. M3b (full + 4-mode integration) is the next plan. |
```

- [ ] **Step 8.2: Add a "Try the smoke training" subsection to README**

Append after the "Try the Gradio UI (after M2)" section:

```markdown
## Run the QLoRA smoke training (after M3a)

```powershell
make synth          # ~5-10 min, $1-3 in Anthropic API spend
make prepare        # ~10 s
make smoke-train    # ~10-20 min on the target GPU; produces models/adapters/qwen2.5-3b-civil-smoke-v0/
```

The smoke run trains on ~200 synthesized Q&A pairs for 1 epoch, validates the pipeline plumbing, and gates on a ≥5% loss decrease. The full training run lands in M3b.
```

- [ ] **Step 8.3: Append "M3a closed" to `m2-readiness.md`**

Append at the end of `docs/superpowers/m2-readiness.md`:

```markdown
---

## M3a — closed (2026-05-XX)

**Tag:** `m3a-qlora-smoke`
**Tests:** ~71 fast + 5 slow (4 prior + 1 smoke).

### Modules delivered

- `src/training/__init__.py`, `synth_qa.py`, `prepare_dataset.py`, `train_qlora.py`
- `src/serving/model_loader.attach_adapter / detach_adapter / current_adapter` (additive)
- `data/processed/qa_seed.jsonl` (12 hand-curated pairs)
- `config/prompts/synth_qa.txt`
- `config/default.yaml` extended with `synth:` and `training:` blocks

### Acceptance signals captured

- ✅ torch 2.5+ ecosystem upgrade applied; `verify_combined_vram.py` peak still < 5.5 GB
- ✅ Unsloth import works; smoke training run produced adapter at `models/adapters/qwen2.5-3b-civil-smoke-v0/`
- ✅ Smoke gate PASS: final loss < first loss × 0.95 (≥5% decrease)
- ✅ Adapter loaded via `attach_adapter`; `model.generate(...)` returns non-empty Korean

### Open questions for M3b

- Full synth target_pairs: design says 1000-1500. Decide based on M3a synth quality.
- Whether `Mode` widening to `qlora` and `rag_qlora` should also unlock per-mode prompt template differences (current prompt_builder doesn't distinguish base vs qlora — only base vs rag).
- Adapter directory naming convention as more versions are produced (`v1` / `v2` / ...).
```

- [ ] **Step 8.4: Run final fast suite + ruff**

```powershell
.venv/Scripts/python -m pytest -m "not slow" -q
.venv/Scripts/python -m ruff check src tests
```
Expected: 71 passed; ruff clean.

- [ ] **Step 8.5: Tag the milestone**

```powershell
git tag m3a-qlora-smoke -m "M3a: QLoRA smoke pipeline (synth+prepare+train+adapter hook+smoke gate)"
git tag --list "m*"
```

Expected: `m0-bootstrap`, `m1-rag-baseline`, `m2-orchestrator`, `m3a-qlora-smoke`. Do NOT push the tag.

- [ ] **Step 8.6: Final commit**

```powershell
git add README.md docs/superpowers/m2-readiness.md
git commit -m "$(cat <<'EOF'
docs(m3a): mark M3a done; readiness checklist closes Plan 3a

README status row flipped to M3a ✅; new "Run the QLoRA smoke training"
section pointing at make synth / prepare / smoke-train. m2-readiness.md
gets an M3a-closed appendix with modules, acceptance signals, and the
M3b open questions queue.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** §3 (pre-flight) → Task 1; §4 (synth_qa) → Tasks 2-4; §5 (prepare_dataset) → Task 5; §2.2 (adapter hooks) → Task 6; §6 (training) → Task 7; §7 (smoke gate) → Task 7 + integration test; §8 (test contract) → Tasks 3, 5, 6, 7. Acceptance criteria §9 are checked in Tasks 1 (re-validation) + 7 (smoke gate) + 8 (final fast suite).
- **Schema reuse:** `QAPair` from `src/common/schemas.py` is the wire format throughout. `Citation` not used directly here — synth_qa relies on `_STATUTE_RE` from `src/eval/citation_checker.py` for filter. No schema changes.
- **Type consistency:** `SynthRow` (dataclass, transient) vs `QAPair` (Pydantic, persisted). `_State` extension in Task 6 matches the signature unit-tested in Task 6 step 6.1.
- **Cost & time:** Task 1 (~30 min env work). Tasks 2-6 (~3 hours TDD). Task 7 (~30 min code + 1-2 hours real smoke training, $1-3 API). Task 8 (~10 min docs + tag). Total: ~5-7 hours from-scratch by an attentive human; subagents are faster on the mechanical TDD parts.
- **Reversibility:** Pre-flight pin upgrade is a single revertable commit. If the smoke training fails the gate, the plan stops at Task 7 step 7.7; Task 8 is not executed (no false-positive tag).
- **Out-of-scope discipline:** No Orchestrator widening, no Gradio mode toggle, no adapter quality eval — those are M3b / M4. The plan deliberately stops at "adapter loads + generates non-empty Korean" for the gate.
