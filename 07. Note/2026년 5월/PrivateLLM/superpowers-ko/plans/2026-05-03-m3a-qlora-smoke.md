# M3a — QLoRA 스모크 파이프라인 구현 계획

> **에이전틱 워커용:** REQUIRED SUB-SKILL: 이 계획을 task-by-task로 구현하려면 superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans를 사용하세요. 단계는 추적을 위해 체크박스(`- [ ]`) 구문을 사용합니다.

**목표:** QLoRA 학습 파이프라인(synth → prepare → train) + adapter attach/detach 훅을 end-to-end로 안착시키되, 200-pair / 1-epoch smoke run으로 게이팅하여 torch 2.5+ ecosystem upgrade가 깔끔했고 학습된 adapter가 로드되고 비어있지 않은 한국어를 생성하는지 확인합니다.

**아키텍처:** pre-flight pin upgrade로 Unsloth가 잠금 해제됩니다(이전에는 torchao를 통해 transitively하게 torch>=2.5에 막혀 있었음). `src/training/`은 세 개의 모듈(`synth_qa`, `prepare_dataset`, `train_qlora`)을 추가하며 오직 JSONL 파일을 통해서만 통신합니다. `src/serving/model_loader.py`는 기존 singleton 위에 `attach_adapter` / `detach_adapter`로 additive하게 확장됩니다. smoke gate는 `train_log.jsonl`의 first-vs-last loss를 비교하고, adapter가 로드된 상태에서 `model.generate(...)`를 한 번 end-to-end로 실행합니다.

**기술 스택:** Python 3.12, torch 2.5+ + cu126 wheel, Unsloth `FastLanguageModel`, peft 0.12+, transformers 4.51+, bitsandbytes, Anthropic SDK, ChromaDB / bge-m3 (M1, 변경 없음).

**코딩 전에 참고할 문서:**
- `docs/superpowers/specs/2026-05-03-m3a-qlora-smoke-design.md` — 본 계획의 design spec
- `docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md` §4.3, §10 — 동결된 MVP spec
- `docs/superpowers/m2-readiness.md`의 "Plan 2 — closed" 섹션 + M3 punchlist
- `Users/hujii/.claude/projects/C--claudeProject-PrivateLLM/memory/project_env_pins.md` — 비자명한 환경 pins

**`src/common/schemas.py`에 이미 정의된 스키마 — 재사용:**
- `Chunk` (id, source, doc_type, statute_no, case_no, title, text, char_range, hash)
- `QAPair` (id, instruction, input, output, source: `"seed"|"synth"|"public"`, cited)
- `Citation` (raw, normalized, kind)
- `Response` (answer, mode, citations, retrieved, latency_ms) — `mode`는 `qlora`/`rag_qlora`를 포함하도록 이미 확장됨

**의존할 기존 surface(중복 금지):**
- `src/serving/model_loader.get_base_model()` — singleton (model, tokenizer)
- `src/eval/citation_checker._STATUTE_RE` — synth_qa filter가 사용하는 regex
- `src/common/config.load_config()` — YAML + .env 로더; `config/default.yaml`에는 이미 `model:`, `embedding:`, `retrieval:`, `serving:`, `chunk:`, `ingestion:`, `logging:` 블록이 있음
- `src/common/paths.{ROOT, CONFIG_DIR, PROCESSED, ADAPTERS, RUNS}`

---

## 파일 맵

**생성:**
- `src/training/__init__.py`
- `src/training/synth_qa.py`
- `src/training/prepare_dataset.py`
- `src/training/train_qlora.py`
- `data/processed/qa_seed.jsonl` (12개 수작업 큐레이션 `QAPair` row)
- `config/prompts/synth_qa.txt`
- `tests/test_synth_qa.py`
- `tests/test_prepare_dataset.py`
- `tests/test_adapter_hooks.py`
- `tests/test_smoke_train.py` (slow, CUDA-gated)

**수정:**
- `pyproject.toml` — torch / transformers / bitsandbytes pin 업데이트 + `anthropic`, `peft`, `tensorboard` 의존성 추가
- `Makefile` — `install-cuda-torch` URL 변경 + 새로운 `synth`, `prepare`, `smoke-train` 타겟
- `src/serving/model_loader.py` — additive: `_State` 확장 + `attach_adapter` / `detach_adapter` / `current_adapter`
- `config/default.yaml` — `training:` 블록(LoRA r/α, lr 등) + `synth:` 블록(model, max_usd, target_pairs) 추가
- `README.md` — M3a 상태 row 변경; `make smoke-train` 문서화
- `docs/superpowers/m2-readiness.md` — M3a 태그 이후 "M3a closed" 섹션 추가

---

## Task 1: torch 2.5+ ecosystem upgrade (pre-flight gate)

이 task는 의도적으로 단일 커밋입니다. §3.2 step 2 이후로 어떤 부분이 회복 불가능하게 깨지면 `git revert HEAD`로 프로젝트를 known-good 상태로 되돌릴 수 있습니다.

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1.1: `pyproject.toml` pin 업데이트**

`pyproject.toml`을 편집:

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

- [ ] **Step 1.2: `Makefile`의 install-cuda-torch URL 업데이트**

`Makefile` 편집 — `install-cuda-torch` 타겟 본문 교체:

```makefile
install-cuda-torch:
	python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126 "torch>=2.5,<2.7"
```

- [ ] **Step 1.3: 환경 재설치**

순서대로 실행:
```powershell
.venv/Scripts/python -m pip install -e ".[dev]"
make install-cuda-torch
make install-unsloth
make postinstall
```

기대 결과: 각 단계가 오류 없이 완료. `install-unsloth` 단계는 `xformers`, `torchao`, `torchvision`도 함께 가져오는데 — 그것은 정상입니다.

- [ ] **Step 1.4: Unsloth import 작동 확인**

실행: `.venv/Scripts/python -c "from unsloth import FastLanguageModel; print('unsloth ok')"`
기대 결과: `unsloth ok`. (이것이 torch 2.4에서의 broken signal이었습니다.)

이것이 실패하면: 중단. 전체 traceback과 함께 `BLOCKED`로 보고합니다. plan controller가 (a) 특정 Unsloth commit pinning, (b) 남은 task에 대해 peft + transformers Trainer로 fallback (Unsloth의 2x 속도 향상은 포기하지만 학습은 unblock됨) 중에서 결정할 것입니다.

- [ ] **Step 1.5: VRAM gate 재검증**

실행: `.venv/Scripts/python scripts/verify_combined_vram.py`
기대 결과: `PASS: peak VRAM <X> GB < budget 5.5 GB`. peak 수치는 M2 baseline인 3.17 GB와 다를 수 있는데, Unsloth 경로가 transformers fallback을 대체하기 때문입니다; peak가 5.0 GB 이상이면 우려사항으로 플래그하세요(여전히 통과지만 이전보다 빡빡함).

- [ ] **Step 1.6: 기존 테스트 suite 재검증**

실행:
```powershell
.venv/Scripts/python -m pytest -m "not slow" -q
.venv/Scripts/python -m pytest -m slow -v
```
기대 결과: 52 fast + 4 slow 모두 통과. Ruff: `.venv/Scripts/python -m ruff check src tests` 깨끗.

- [ ] **Step 1.7: 커밋**

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

`data/processed/qa_seed.jsonl`의 12개 수작업 큐레이션 Q&A pair와 synth_qa.py 모듈이 이를 확장하는 데 사용할 system prompt.

**Files:**
- Create: `data/processed/qa_seed.jsonl`
- Create: `config/prompts/synth_qa.txt`
- Modify: `.gitignore` — seed 파일 재포함

- [ ] **Step 2.1: seed 파일을 추적하도록 `.gitignore` 조정**

현재 `.gitignore`의 `data/processed/*` 라인이 seed를 제외시킵니다. 예외 추가:

```
# Existing line: data/processed/*
# Existing line: !data/processed/.gitkeep
!data/processed/qa_seed.jsonl
```

새 라인을 기존 `!data/processed/.gitkeep` 라인 뒤에 배치합니다. `git check-ignore -v data/processed/qa_seed.jsonl`로 확인 — 출력 없음을 기대(= 무시되지 않음).

- [ ] **Step 2.2: seed Q&A 파일 작성**

`data/processed/qa_seed.jsonl`을 정확히 12 라인으로 생성하며, 각각은 JSON-직렬화된 `QAPair`. 12쌍은 민법 기본을 다룹니다: 임대차 정의 (618), 임대차 존속기간 (619), 매매 의의 (563), 채권의 의의 (3장 일반), 손해배상 (390, 393), 시효 (162), 동시이행항변권 (536), 해제 (543), 변제 (460), 보증 (428), 위임 (680), 사용대차 (609).

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

(각 라인은 유효한 JSON이어야 함 — `.venv/Scripts/python -c "from src.common.schemas import QAPair; import json; [QAPair.model_validate_json(l) for l in open('data/processed/qa_seed.jsonl', encoding='utf-8')]; print('all 12 valid')"`로 검증. 기대 결과: `all 12 valid`.)

- [ ] **Step 2.3: synth_qa system prompt 작성**

`config/prompts/synth_qa.txt` 생성:

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

- [ ] **Step 2.4: 커밋**

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

## Task 3: synth_qa.py 후처리 TDD (filter + cost cap)

순수 로직 surface — citation/length/dedupe filter와 cost-cap helper. Anthropic SDK는 mock합니다.

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/synth_qa.py` (filter 함수만; 완전한 `run_synth`는 Task 4에서 안착)
- Test: `tests/test_synth_qa.py`

- [ ] **Step 3.1: 실패하는 테스트 작성**

`tests/test_synth_qa.py` 생성:

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

- [ ] **Step 3.2: 실패하는 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_synth_qa.py -v`
기대 결과: ImportError (`src.training.synth_qa` 모듈이 존재하지 않음).

- [ ] **Step 3.3: `src/training/__init__.py` 구현**

```python
"""Training subsystem: synth_qa, prepare_dataset, train_qlora."""
```

- [ ] **Step 3.4: `src/training/synth_qa.py` 구현 (filter + cost-cap만)**

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

- [ ] **Step 3.5: 통과 확인을 위한 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_synth_qa.py -v`
기대 결과: 8 passed.

- [ ] **Step 3.6: 전체 fast suite 실행**

실행: `.venv/Scripts/python -m pytest -m "not slow" -q`
기대 결과: 60 passed (이전 52개 + 신규 8개).

- [ ] **Step 3.7: 커밋**

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

## Task 4: synth_qa.py Claude 통합 + cache + `run_synth`

Anthropic SDK + 디스크 cache + cost-capped 메인 루프를 연결합니다. mock된 SDK를 사용한 smoke 테스트(Task 7)를 통한 통합으로 테스트되며; filter는 Task 3에서 잠겨 있으므로 여기서 새로운 fast 테스트는 없습니다.

**Files:**
- Modify: `src/training/synth_qa.py` (확장; filter 섹션은 다시 작성하지 말 것)
- Modify: `Makefile` — `synth` 타겟 추가
- Modify: `config/default.yaml` — `synth:` 블록 추가

- [ ] **Step 4.1: `config/default.yaml`을 `synth:` 블록으로 확장**

`serving:` 블록 뒤(`logging:` 앞)에 추가:

```yaml
synth:
  model: sonnet           # or opus
  target_pairs: 200       # smoke target; M3b will use 1500
  max_usd: 3.0            # smoke cap; M3b will use 15.0
  chunks_per_batch: 4
  pairs_per_batch: 2
  seeds_per_prompt: 3
```

- [ ] **Step 4.2: Makefile `synth` 타겟 추가**

`Makefile` 편집 — `embed:` 뒤에 새 타겟 추가:

```makefile
synth:
	python -m src.training.synth_qa
```

또한 Makefile 상단의 `.PHONY` 목록에 `synth`를 추가합니다.

- [ ] **Step 4.3: `src/training/synth_qa.py`를 `run_synth`로 확장**

`src/training/synth_qa.py`에 추가(기존 함수 뒤; 이미 있는 것은 수정 금지):

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

- [ ] **Step 4.4: 신규 모듈 import 스팟 체크**

실행:
```powershell
.venv/Scripts/python -c "from src.training.synth_qa import run_synth, _build_prompt, _parse_response, SynthRow; print('synth_qa surface ok')"
```
기대 결과: `synth_qa surface ok` (아직 API 호출 없음 — import는 Anthropic을 호출하지 않음).

- [ ] **Step 4.5: YAML 로드 검증**

```powershell
.venv/Scripts/python -c "from src.common.config import load_config; cfg = load_config(); print('synth:', cfg['synth'])"
```
기대 결과: `synth: {'model': 'sonnet', 'target_pairs': 200, 'max_usd': 3.0, 'chunks_per_batch': 4, 'pairs_per_batch': 2, 'seeds_per_prompt': 3}`.

- [ ] **Step 4.6: fast suite 실행 (회귀 없음)**

`.venv/Scripts/python -m pytest -m "not slow" -q`
기대 결과: 60 passed.

- [ ] **Step 4.7: 커밋**

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

순수 tokenizer + masking surface. tokenization이 빠르기 때문에(<1 s) 실제 Qwen2.5 tokenizer를 사용하며, 대안(BPE를 mock하는 것)은 의미 있는 coverage를 모두 잃게 됩니다.

**Files:**
- Create: `src/training/prepare_dataset.py`
- Test: `tests/test_prepare_dataset.py`
- Modify: `Makefile` — `prepare` 타겟 추가

- [ ] **Step 5.1: 실패하는 테스트 작성**

`tests/test_prepare_dataset.py` 생성:

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

- [ ] **Step 5.2: 실패하는 테스트 실행**

`.venv/Scripts/python -m pytest tests/test_prepare_dataset.py -v`
기대 결과: ImportError.

- [ ] **Step 5.3: `src/training/prepare_dataset.py` 구현**

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

- [ ] **Step 5.4: Makefile `prepare` 타겟 추가**

`Makefile` 편집:

```makefile
prepare:
	python -m src.training.prepare_dataset
```

상단의 `.PHONY` 목록에 `prepare`를 추가합니다.

- [ ] **Step 5.5: 통과 확인을 위한 테스트 실행**

`.venv/Scripts/python -m pytest tests/test_prepare_dataset.py -v`
기대 결과: 5 passed in <5 s (tokenizer는 첫 테스트 후 캐시됨).

- [ ] **Step 5.6: 전체 fast suite 실행**

`.venv/Scripts/python -m pytest -m "not slow" -q`
기대 결과: 65 passed (이전 60개 + 신규 5개).

- [ ] **Step 5.7: 커밋**

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

## Task 6: model_loader adapter 훅 (TDD)

`_State`를 `base_model`과 `adapter_path`로 확장하고 `attach_adapter` / `detach_adapter` / `current_adapter`를 추가합니다. 단위 테스트는 GPU 회피를 위해 stub `_state.model` mock을 사용합니다.

**Files:**
- Modify: `src/serving/model_loader.py`
- Test: `tests/test_adapter_hooks.py`

- [ ] **Step 6.1: 실패하는 테스트 작성**

`tests/test_adapter_hooks.py` 생성:

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

- [ ] **Step 6.2: 실패하는 테스트 실행**

`.venv/Scripts/python -m pytest tests/test_adapter_hooks.py -v`
기대 결과: `attach_adapter` / `current_adapter`에 대한 AttributeError — 아직 존재하지 않음.

- [ ] **Step 6.3: `src/serving/model_loader.py` 확장**

`src/serving/model_loader.py` 편집:

`_State` dataclass(라인 26-30 근처)에서 확장:

```python
@dataclass
class _State:
    model: Any = None
    base_model: Any = None
    tokenizer: Any = None
    loader: str = ""
    adapter_path: str | None = None
```

파일 하단에 다음 새 함수들을 추가:

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

또한 파일 상단의 import에 `from pathlib import Path`를 추가(아직 없으면).

- [ ] **Step 6.4: 통과 확인을 위한 테스트 실행**

`.venv/Scripts/python -m pytest tests/test_adapter_hooks.py -v`
기대 결과: 6 passed.

- [ ] **Step 6.5: 전체 fast suite 실행**

`.venv/Scripts/python -m pytest -m "not slow" -q`
기대 결과: 71 passed (이전 65개 + 신규 6개).

- [ ] **Step 6.6: slow singleton 테스트 재실행**

Task 1의 slow 테스트는 singleton 안정성을 확인하는데, 이제 새 필드와 공존해야 합니다. 실행:

`.venv/Scripts/python -m pytest tests/test_serving_integration.py::test_get_base_model_is_singleton -m slow -v`
기대 결과: PASS.

- [ ] **Step 6.7: 커밋**

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

## Task 7: train_qlora.py + smoke gate + slow 통합 테스트

학습 코드, CLI smoke 노브, 그리고 mock된 Claude SDK로 전체 파이프라인을 작동시키는 하나의 end-to-end slow 테스트.

**Files:**
- Create: `src/training/train_qlora.py`
- Create: `tests/test_smoke_train.py`
- Modify: `Makefile` — `smoke-train` 타겟 추가
- Modify: `config/default.yaml` — `training:` 블록 추가

- [ ] **Step 7.1: `config/default.yaml`에 `training:` 블록 추가**

`synth:` 블록 뒤(`logging:` 앞)에 추가:

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

- [ ] **Step 7.2: `src/training/train_qlora.py` 구현**

`src/training/train_qlora.py` 생성:

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

- [ ] **Step 7.3: Makefile `smoke-train` 타겟 추가**

`Makefile` 편집:

```makefile
smoke-train:
	python -m src.training.train_qlora --smoke
```

상단의 `.PHONY` 목록에 `smoke-train`를 추가하고, 기존 `train` 타겟 본문이 있다면 `# Plan 3 — not implemented` 주석을 삭제하도록 업데이트:

```makefile
train:
	python -m src.training.train_qlora
```

- [ ] **Step 7.4: slow smoke 통합 테스트 작성**

`tests/test_smoke_train.py` 생성:

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

- [ ] **Step 7.5: fast suite 실행 (회귀 없음)**

`.venv/Scripts/python -m pytest -m "not slow" -q`
기대 결과: 71 passed (이번 task에는 신규 fast 테스트 없음; `test_smoke_train.py`는 `-m slow`).

- [ ] **Step 7.6: slow smoke 테스트 실행**

`.venv/Scripts/python -m pytest tests/test_smoke_train.py -m slow -v -s`

이는 Qwen 4-bit를 로드하고 + 2 학습 step을 실행하고 + adapter attach합니다. 기대 결과: ~60-120 s 안에 PASS. GPU-OOM이 발생하면 테스트의 `max_seq=512`를 `256`으로 줄이고 재시도합니다.

- [ ] **Step 7.7: 실제 smoke 학습 실행 (make smoke-train)**

이는 `.env`에 `ANTHROPIC_API_KEY`와 ~$1-3의 API 지출이 필요합니다.

```powershell
make synth                      # ~5-10 min, $1-3
make prepare                    # ~10 s
make smoke-train                # ~10-20 min on the target GPU
```

기대 출력 (`make smoke-train` 마지막 라인들):

```
[train_qlora] adapter saved to ...models/adapters/qwen2.5-3b-civil-smoke-v0
[train_qlora] smoke PASS: loss <X> < target <Y> (first=<Z>, decrease=<N>%)
```

smoke FAIL이면: 진행하지 마세요. JSONL 로그, loss curve, adapter dir 상태를 캡처한 후 다시 보고하세요. 일반적인 실패 모드:
- Loss가 5% 이상 감소하지 않음: 학습 발산 또는 learning rate 과다. 사용자에게 위임.
- 학습 중 OOM: smoke 노브의 `max_seq`를 줄이고 재검토.
- Adapter 파일 누락: 저장 전에 학습이 충돌; run dir에서 traceback 확인.

- [ ] **Step 7.8: 커밋**

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

## Task 8: 문서 + 태그

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/m2-readiness.md`

- [ ] **Step 8.1: README 상태 테이블 업데이트**

`README.md` 편집 — M3 row 변경:

```markdown
| **M3 — QLoRA training pipeline** | 🟡 M3a (smoke) ✅ done — tag `m3a-qlora-smoke`. M3b (full + 4-mode integration) is the next plan. |
```

- [ ] **Step 8.2: README에 "Try the smoke training" 하위 섹션 추가**

"Try the Gradio UI (after M2)" 섹션 뒤에 추가:

```markdown
## Run the QLoRA smoke training (after M3a)

```powershell
make synth          # ~5-10 min, $1-3 in Anthropic API spend
make prepare        # ~10 s
make smoke-train    # ~10-20 min on the target GPU; produces models/adapters/qwen2.5-3b-civil-smoke-v0/
```

The smoke run trains on ~200 synthesized Q&A pairs for 1 epoch, validates the pipeline plumbing, and gates on a ≥5% loss decrease. The full training run lands in M3b.
```

- [ ] **Step 8.3: `m2-readiness.md`에 "M3a closed" 추가**

`docs/superpowers/m2-readiness.md`의 끝에 추가:

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

- [ ] **Step 8.4: 최종 fast suite + ruff 실행**

```powershell
.venv/Scripts/python -m pytest -m "not slow" -q
.venv/Scripts/python -m ruff check src tests
```
기대 결과: 71 passed; ruff clean.

- [ ] **Step 8.5: 마일스톤 태그**

```powershell
git tag m3a-qlora-smoke -m "M3a: QLoRA smoke pipeline (synth+prepare+train+adapter hook+smoke gate)"
git tag --list "m*"
```

기대 결과: `m0-bootstrap`, `m1-rag-baseline`, `m2-orchestrator`, `m3a-qlora-smoke`. 태그를 push하지 마세요.

- [ ] **Step 8.6: 최종 커밋**

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

## 셀프 리뷰 노트

- **Spec coverage:** §3 (pre-flight) → Task 1; §4 (synth_qa) → Tasks 2-4; §5 (prepare_dataset) → Task 5; §2.2 (adapter hooks) → Task 6; §6 (training) → Task 7; §7 (smoke gate) → Task 7 + 통합 테스트; §8 (test contract) → Tasks 3, 5, 6, 7. Acceptance criteria §9는 Tasks 1 (재검증) + 7 (smoke gate) + 8 (최종 fast suite)에서 확인됩니다.
- **Schema reuse:** `src/common/schemas.py`의 `QAPair`가 전 구간의 wire format입니다. `Citation`은 여기서 직접 사용되지 않음 — synth_qa는 filter를 위해 `src/eval/citation_checker.py`의 `_STATUTE_RE`에 의존. 스키마 변경 없음.
- **Type consistency:** `SynthRow` (dataclass, transient) vs `QAPair` (Pydantic, persisted). Task 6의 `_State` 확장은 Task 6 step 6.1에서 unit-test된 시그니처와 일치.
- **비용 & 시간:** Task 1 (~30분 환경 작업). Tasks 2-6 (~3시간 TDD). Task 7 (~30분 코드 + 1-2시간 실제 smoke 학습, $1-3 API). Task 8 (~10분 문서 + 태그). 총: 주의 깊은 인간이 처음부터 ~5-7시간; subagent는 기계적인 TDD 부분에서 더 빠름.
- **Reversibility:** Pre-flight pin upgrade는 단일 revertable commit. smoke 학습이 게이트에 실패하면 계획은 Task 7 step 7.7에서 멈춤; Task 8은 실행되지 않음 (잘못된 양성 태그 없음).
