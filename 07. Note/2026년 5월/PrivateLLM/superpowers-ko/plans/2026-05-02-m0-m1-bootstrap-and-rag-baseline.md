# Legal PrivateLLM — Plan 1: M0 Bootstrap + M1 RAG Baseline

> **에이전트형 작업자를 위한 안내:** 필수 서브 스킬: superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 를 사용하여 본 plan 을 task 단위로 구현하세요. 단계는 추적을 위해 체크박스(`- [ ]`) 문법을 사용합니다.

**목표:** 프로젝트 골격을 마련하고, Windows + RTX 3060 6GB 호스트에서 로컬 Unsloth + Qwen2.5-3B 4-bit 환경을 검증한 뒤, 한국 민법(民法)에 대한 동작하는 hybrid(dense + BM25 + RRF) RAG retriever 를 제공합니다.

**아키텍처:** `pyproject.toml` 을 사용하는 Python 3.11 프로젝트(pip 설치 가능, uv 친화적). 코드는 `src/common/`(schemas, config, paths, logging), `src/ingestion/`(법령 fetcher + chunker + synth_qa stub), `src/rag/`(embedder, BM25, ChromaDB index, hybrid retriever)로 분리됩니다. JSONL 경계에서는 Pydantic 모델을 사용하며, ChromaDB embedded mode 외에 별도 DB 는 없습니다. 결정론적 로직(chunker, BM25, RRF retriever)에는 TDD 를 적용하고, 얇은 라이브러리 wrapper(Unsloth model loader, bge-m3 embedder, ChromaDB client)에는 TDD 대신 smoke test 를 적용합니다.

**기술 스택:** Python 3.11, PyTorch 2.4.x, Unsloth, transformers, peft, bitsandbytes, FlagEmbedding (bge-m3), chromadb, rank-bm25, pydantic, pyyaml, pytest, ruff, requests, python-dotenv.

**Spec 참조:** `docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md`

---

## Phase 개요

| Phase | Tasks | 결과 |
|---|---|---|
| **M0 — Bootstrap** | Tasks 1–6 | 프로젝트가 컴파일되고, 테스트가 실행되며, Qwen2.5-3B 4-bit 가 로드되고, 한 문장을 생성 |
| **M1 — Ingestion + RAG** | Tasks 7–14 | `python -m src.rag.retriever "..."` 가 top-5 民法 chunk 를 반환 |

이 plan 이 성공적으로 완료된 후에는 M2(orchestrator + Gradio + base/RAG 모드)를 다루는 Plan 2 를 작성합니다.

---

## 파일 구조 (이 plan 에서 생성)

```
PrivateLLM/
├── pyproject.toml                  # Task 1
├── Makefile                        # Task 1
├── .gitignore                      # Task 1
├── .env.example                    # Task 1
├── README.md                       # Task 1
├── config/
│   └── default.yaml                # Task 1
├── src/
│   ├── __init__.py                 # Task 1
│   ├── common/
│   │   ├── __init__.py             # Task 2
│   │   ├── schemas.py              # Task 2
│   │   ├── config.py               # Task 3
│   │   ├── paths.py                # Task 3
│   │   └── logging.py              # Task 3
│   ├── ingestion/
│   │   ├── __init__.py             # Task 7
│   │   ├── fetch_statutes.py       # Task 7
│   │   └── chunk.py                # Task 8
│   └── rag/
│       ├── __init__.py             # Task 9
│       ├── bm25.py                 # Task 9
│       ├── embed.py                # Task 10
│       ├── index.py                # Task 11
│       └── retriever.py            # Task 12
├── tests/
│   ├── __init__.py                 # Task 1
│   ├── test_schemas.py             # Task 2
│   ├── test_chunk.py               # Task 8
│   ├── test_bm25.py                # Task 9
│   ├── test_retriever.py           # Task 12
│   ├── test_ingestion_smoke.py     # Task 7 (smoke)
│   ├── test_e2e_rag.py             # Task 14
│   └── fixtures/
│       └── sample_chunks.jsonl     # Task 8
├── scripts/
│   └── verify_unsloth.py           # Task 5
└── data/                           # gitignored
    ├── raw/, cache/, processed/, chroma/
```

**모든 파일은 하나의 명확한 책임을 가집니다.** 파일은 ~250 줄을 초과하지 않습니다. 구현 중 그 이상 늘어나면 분리합니다(예: `retriever.py` → `retriever.py` + `rrf.py`).

---

# Phase M0 — Bootstrap

## Task 1: 프로젝트 골격 (pyproject.toml, Makefile, gitignore, env, README, config, src/tests dirs)

**파일:**
- 생성: `pyproject.toml`
- 생성: `Makefile`
- 생성: `.gitignore`
- 생성: `.env.example`
- 생성: `README.md`
- 생성: `config/default.yaml`
- 생성: `src/__init__.py` (빈 파일)
- 생성: `tests/__init__.py` (빈 파일)

- [ ] **Step 1: `pyproject.toml` 생성**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "legal-privatellm"
version = "0.1.0"
description = "Legal PrivateLLM PoC — RAG + QLoRA on Korean civil law"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch==2.4.*",
    "transformers>=4.44,<5",
    "peft>=0.12",
    "bitsandbytes>=0.43; sys_platform == 'linux'",
    "bitsandbytes>=0.43; sys_platform == 'win32'",
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
]

[project.optional-dependencies]
unsloth = [
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
]
dev = [
    "pytest>=8",
    "pytest-mock>=3.12",
    "ruff>=0.6",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: `Makefile` 생성**

```makefile
.PHONY: install install-dev install-unsloth lint test ingest embed train eval serve clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

install-unsloth:
	python -m pip install -e ".[unsloth]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q

ingest:
	python -m src.ingestion.fetch_statutes

embed:
	python -m src.rag.index

train:
	python -m src.training.train_qlora

eval:
	python -m src.eval.runner

serve:
	python -m src.serving.app_gradio

clean:
	rm -rf data/chroma data/processed/*.jsonl data/cache reports/* runs/* models/adapters/*
```

- [ ] **Step 3: `.gitignore` 생성**

```
# Data + models (large, generated)
data/raw/
data/cache/
data/processed/
data/chroma/
data/bm25.pkl
models/
runs/
reports/

# Env
.env
.env.local

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# But keep
!data/processed/.gitkeep
!data/eval/.gitkeep
```

- [ ] **Step 4: `.env.example` 생성**

```
# Korean statute OpenAPI key (https://open.law.go.kr)
LAW_OPEN_API_KEY=your_key_here

# Anthropic API for synthetic Q&A and LLM-as-Judge
ANTHROPIC_API_KEY=sk-ant-...

# Cost guard for synthesis + judge runs (USD)
MAX_API_USD=10

# HuggingFace cache (point at large disk if needed)
HF_HOME=./data/cache/hf
TRANSFORMERS_CACHE=./data/cache/hf

# Embedding batch size (auto-halved on OOM by embed.py)
EMBED_BATCH=8
```

- [ ] **Step 5: `README.md` 생성**

```markdown
# Legal PrivateLLM (PoC)

Korean civil-law (民法) Q&A PoC validating RAG + QLoRA on a 6GB local GPU.

See [docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md](docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md) for the full design.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
make install-dev
make install-unsloth   # may need WSL2 fallback on Windows
copy .env.example .env  # then edit keys

make test
python scripts/verify_unsloth.py   # confirms 4-bit base model loads
```

## Layout

- `src/common/` — schemas, config, paths, logging
- `src/ingestion/` — fetch + chunk corpus
- `src/rag/` — embed, index, retrieve
- `src/training/` — QLoRA (later plan)
- `src/serving/` — orchestrator + Gradio (later plan)
- `src/eval/` — 4-way ablation (later plan)
- `tests/` — pytest

## Status

Phase M0 + M1 — see `docs/superpowers/plans/`.
```

- [ ] **Step 6: `config/default.yaml` 생성**

```yaml
model:
  base_id: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
  max_seq_len: 2048
  load_in_4bit: true

embedding:
  model_id: BAAI/bge-m3
  batch_size: 8
  device: cuda

retrieval:
  top_k: 5
  dense_top_n: 20
  bm25_top_n: 20
  rrf_k: 60

chunk:
  max_chars: 1200
  min_chars: 200
  overlap: 0   # 조문 단위라 보통 0

paths:
  raw_dir: data/raw
  cache_dir: data/cache
  processed_dir: data/processed
  chroma_dir: data/chroma
  bm25_path: data/bm25.pkl

ingestion:
  source: nlic   # 국가법령정보 OpenAPI
  law_id: 001706 # 민법 법령 ID (검증 대상; fallback은 검색)

logging:
  level: INFO
```

- [ ] **Step 7: 빈 패키지 파일들 생성**

```bash
mkdir -p src/common src/ingestion src/rag src/training src/serving src/eval tests/fixtures scripts data/raw data/cache data/processed data/chroma
echo "" > src/__init__.py
echo "" > tests/__init__.py
echo "" > data/processed/.gitkeep
```

- [ ] **Step 8: 첫 commit**

```bash
git add pyproject.toml Makefile .gitignore .env.example README.md config/default.yaml src/__init__.py tests/__init__.py data/processed/.gitkeep
git commit -m "feat(m0): project skeleton — pyproject, Makefile, config, package dirs"
```

---

## Task 2: Pydantic schemas (TDD)

**파일:**
- 생성: `src/common/__init__.py`
- 생성: `src/common/schemas.py`
- 생성: `tests/test_schemas.py`

- [ ] **Step 1: `tests/test_schemas.py` 에 실패하는 테스트 작성**

```python
"""Tests for src/common/schemas.py — Pydantic data models on the JSONL boundary."""
import json
import pytest
from pydantic import ValidationError

from src.common.schemas import Chunk, QAPair, EvalItem, Response, Citation


def test_chunk_round_trip():
    payload = {
        "id": "stat-618-1",
        "source": "nlic",
        "doc_type": "조문",
        "statute_no": "민법 제618조",
        "case_no": None,
        "title": "임대차의 의의",
        "text": "임대차는 당사자 일방이 ...",
        "char_range": [0, 35],
        "hash": "abc123",
    }
    chunk = Chunk.model_validate(payload)
    assert chunk.statute_no == "민법 제618조"
    assert chunk.doc_type == "조문"
    # JSONL round-trip
    line = chunk.model_dump_json()
    chunk2 = Chunk.model_validate_json(line)
    assert chunk2 == chunk


def test_chunk_doc_type_must_be_known():
    with pytest.raises(ValidationError):
        Chunk(
            id="x",
            source="nlic",
            doc_type="해설",   # not allowed
            text="...",
            char_range=[0, 3],
            hash="h",
        )


def test_qapair_minimum_fields():
    qa = QAPair(
        id="qa-1",
        instruction="임대차 갱신요구권은 무엇인가?",
        output="민법 제643조에 따라 ...",
        source="seed",
    )
    assert qa.input is None  # optional


def test_eval_item_requires_review_metadata():
    with pytest.raises(ValidationError):
        EvalItem(
            id="e1",
            question="Q?",
            reference_answer="A.",
            expected_citations=["민법 제618조"],
        )  # missing reviewed_by / reviewed_at


def test_eval_item_ok_when_reviewed():
    item = EvalItem(
        id="e1",
        question="Q?",
        reference_answer="A.",
        expected_citations=["민법 제618조"],
        reviewed_by="hujii",
        reviewed_at="2026-05-02",
    )
    assert item.expected_citations == ["민법 제618조"]


def test_response_default_empty_citations():
    r = Response(answer="hi", retrieved=[], latency_ms=12.3)
    assert r.citations == []


def test_citation_parsed_form():
    c = Citation(raw="[민법 제618조]", normalized="민법 제618조", kind="statute")
    assert c.kind == "statute"
```

- [ ] **Step 2: 테스트가 실패함을 확인하기 위해 실행**

실행: `pytest tests/test_schemas.py -v`
예상: collection error / `ModuleNotFoundError: No module named 'src.common.schemas'`

- [ ] **Step 3: `src/common/__init__.py` 구현**

```python
"""Common building blocks: schemas, config, paths, logging."""
```

- [ ] **Step 4: `src/common/schemas.py` 구현**

```python
"""Pydantic data models used at every JSONL / function boundary.

Keep these small and stable — they define the wire format between modules.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["조문", "판례"]
CitationKind = Literal["statute", "case"]


class Chunk(BaseModel):
    id: str
    source: str
    doc_type: DocType
    statute_no: Optional[str] = None
    case_no: Optional[str] = None
    title: Optional[str] = None
    text: str
    char_range: list[int] = Field(min_length=2, max_length=2)
    hash: str


class QAPair(BaseModel):
    id: str
    instruction: str
    input: Optional[str] = None
    output: str
    source: Literal["seed", "synth", "public"]
    cited: list[str] = Field(default_factory=list)


class EvalItem(BaseModel):
    id: str
    question: str
    reference_answer: str
    expected_citations: list[str] = Field(default_factory=list)
    reviewed_by: str
    reviewed_at: str  # ISO date


class Citation(BaseModel):
    raw: str
    normalized: str
    kind: CitationKind


class Response(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[Chunk] = Field(default_factory=list)
    latency_ms: float
```

- [ ] **Step 5: 테스트가 통과함을 확인하기 위해 실행**

실행: `pytest tests/test_schemas.py -v`
예상: 6 개의 테스트 모두 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/common/__init__.py src/common/schemas.py tests/test_schemas.py
git commit -m "feat(common): Pydantic schemas for Chunk/QAPair/EvalItem/Response/Citation"
```

---

## Task 3: Config 로더, paths, logging

**파일:**
- 생성: `src/common/paths.py`
- 생성: `src/common/config.py`
- 생성: `src/common/logging.py`

이들은 인프라 헬퍼입니다 — 충분히 얇아서 별도 단위 테스트 대신 이후 task 들의 사용을 통해 테스트합니다. 작게 유지하세요.

- [ ] **Step 1: `src/common/paths.py` 구현**

```python
"""Filesystem paths derived from the project root.

ROOT is resolved as the parent of the `src/` directory. All paths are absolute.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CONFIG_DIR = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"
CHROMA = DATA / "chroma"
BM25_PATH = DATA / "bm25.pkl"
MODELS = ROOT / "models"
ADAPTERS = MODELS / "adapters"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    """Create all data directories if missing. Safe to call repeatedly."""
    for p in (RAW, CACHE, PROCESSED, CHROMA.parent, MODELS, ADAPTERS, RUNS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: `src/common/config.py` 구현**

```python
"""YAML config loader. Supports overrides via env vars and explicit paths.

Usage:
    cfg = load_config()
    cfg["model"]["base_id"]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.common.paths import CONFIG_DIR, ROOT

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config. Loads `.env` from project root as a side effect."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Allow EMBED_BATCH env override (used by embed.py)
    if (env_batch := os.getenv("EMBED_BATCH")):
        cfg.setdefault("embedding", {})["batch_size"] = int(env_batch)
    return cfg
```

- [ ] **Step 3: `src/common/logging.py` 구현**

```python
"""Structured logging setup. One `get_logger(name)` call per module."""
from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_initialized = False


def _init() -> None:
    global _initialized
    if _initialized:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    _init()
    return logging.getLogger(name)
```

- [ ] **Step 4: Python REPL 에서 smoke check**

실행:

```powershell
python -c "from src.common.config import load_config; print(load_config()['model']['base_id'])"
```

예상: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`

- [ ] **Step 5: Commit**

```bash
git add src/common/paths.py src/common/config.py src/common/logging.py
git commit -m "feat(common): paths, YAML config loader, logging helper"
```

---

## Task 4: 개발 환경 설치

이 task 는 환경 작업입니다 — 명령을 실행하고 버전을 확인하세요. 파일 변경은 없습니다.

- [ ] **Step 1: venv 생성 + 개발 의존성 설치**

실행 (PowerShell, 프로젝트 루트):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install -e ".[dev]"
```

예상: 오류 없이 완료. `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` 가 `True` 와 CUDA 버전(예: `12.4`)을 출력.

- [ ] **Step 2: `pytest` 실행 확인**

실행: `pytest -q`
예상: 6 개 테스트 통과 (Task 2 의 schema 테스트).

- [ ] **Step 3: Unsloth 설치 (best-effort 네이티브 Windows)**

네이티브 Windows 에서 Unsloth 공식 wheel 설치는 깨지기 쉽습니다. 먼저 프로젝트의 자체 extra 를 시도하고, 실패하면 Task 5 Step 3 의 WSL2 fallback 을 따르세요.

```powershell
pip install -e ".[unsloth]"
```

실패하더라도 막히지 마세요 — 명시적 fallback 경로가 있는 Task 5 로 진행하세요.

- [ ] **Step 4: lockfile 이 있다면 commit**

pip 가 lockfile 을 작성하지 않았다면 건너뜁니다. 그렇지 않으면 commit 하세요. (`pip freeze > requirements.lock` 은 선택 사항입니다 — spec 은 `pyproject.toml` 만 사용합니다.)

---

## Task 5: Qwen2.5-3B 4-bit 로드 + 생성 검증

**파일:**
- 생성: `scripts/verify_unsloth.py`

이것은 **M0 acceptance gate** 입니다. 이 스크립트가 실행되지 않으면 M1 은 진행할 수 없습니다. 왜냐하면 아직은 모델이 필요한 다른 곳이 없지만, 이후 모든 phase 는 모델을 필요로 하기 때문입니다.

- [ ] **Step 1: `scripts/verify_unsloth.py` 작성**

```python
"""Verify the local environment can load Qwen2.5-3B 4-bit and generate text.

Run: python scripts/verify_unsloth.py
"""
from __future__ import annotations

import time

import torch

from src.common.config import load_config
from src.common.logging import get_logger

log = get_logger("verify")


def main() -> int:
    cfg = load_config()
    base_id = cfg["model"]["base_id"]

    log.info("CUDA available: %s; device: %s", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    log.info("Loading %s in 4-bit ...", base_id)
    t0 = time.time()
    try:
        from unsloth import FastLanguageModel  # type: ignore
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_id,
            max_seq_length=cfg["model"]["max_seq_len"],
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        loader = "unsloth"
    except Exception as e:
        log.warning("Unsloth load failed (%s); falling back to vanilla transformers + bitsandbytes.", e)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=bnb,
            device_map="auto",
        )
        loader = "transformers"

    log.info("Loaded via %s in %.1fs", loader, time.time() - t0)
    log.info("VRAM allocated: %.2f GB", torch.cuda.max_memory_allocated() / 1024 ** 3)

    prompt = "민법 제618조의 임대차 정의를 한 문장으로 요약하세요."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    t0 = time.time()
    out = model.generate(inputs, max_new_tokens=128, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    log.info("Generated in %.1fs (loader=%s):\n%s", time.time() - t0, loader, text)

    if not text.strip():
        log.error("Empty generation — environment is unhealthy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 스크립트 실행**

실행: `python scripts/verify_unsloth.py`

예상:
- CUDA available: True
- `unsloth` 또는 `transformers` 중 하나로 로드됨
- VRAM 할당량 < 5.0 GB
- 민법 제618조에 대한 비어 있지 않은 한국어 응답

VRAM 이 5.0 GB 를 초과하거나 생성이 hang 되면 M0 실패로 간주하고 — 계속하기 전에 조사하세요.

- [ ] **Step 3: WSL2 fallback (Step 2 가 네이티브 Windows 에서 실패한 경우에만)**

PowerShell 에서 실행:

```powershell
wsl --install -d Ubuntu-22.04
# After WSL is ready:
wsl
# Inside WSL:
sudo apt update && sudo apt install -y python3.11 python3.11-venv build-essential
cd /mnt/c/ClaudeProject/PrivateLLM
python3.11 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install -e ".[dev,unsloth]"
python scripts/verify_unsloth.py
```

예상: Step 2 와 동일한 성공 출력, 단 WSL2 내부.

WSL2 도 실패하면, plan-spec §11 row 2 의 fallback 으로 이동합니다: `config/default.yaml` 의 `model.base_id` 를 `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` 로 편집하여 `Qwen2.5-1.5B-Instruct`(더 작고 낮은 VRAM)로 떨어뜨리고 Step 2 를 재실행합니다. 변경 사항을 `README.md` "Status" 에 문서화합니다.

- [ ] **Step 4: README 에 VRAM 기록**

`README.md` 에 짧은 섹션을 추가:

```markdown
## Environment-Verified VRAM

- GPU: <fill in `nvidia-smi` GPU name>
- Loader: <unsloth | transformers>
- Qwen2.5-3B 4-bit max VRAM allocated: <X.XX> GB
- Verified on: <YYYY-MM-DD>
```

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_unsloth.py README.md
git commit -m "feat(m0): verify Qwen2.5-3B 4-bit loads + generates locally"
```

---

## Task 6: M0 acceptance gate

이것은 코드 task 가 아니라 **review gate** 입니다. M1 을 시작하기 전에 모든 M0 checklist 항목을 확인하세요.

- [ ] **Step 1: 최종 M0 checklist 실행**

```powershell
.venv\Scripts\Activate.ps1
make test               # all tests pass
ruff check src tests    # no lint errors (or all fixable with --fix)
python scripts/verify_unsloth.py   # exits 0
```

예상: 세 명령 모두 성공.

- [ ] **Step 2: commit 에 태그**

```bash
git tag m0-bootstrap
```

실패한 항목이 있다면 태그하기 전에 수정하세요. 건너뛰지 마세요 — 이후 plan 의 나머지는 모든 M0 가 작동한다고 가정합니다.

---

# Phase M1 — Ingestion + RAG Baseline

## Task 7: 법령 fetcher (ingestion/fetch_statutes.py) + smoke test

**파일:**
- 생성: `src/ingestion/__init__.py`
- 생성: `src/ingestion/fetch_statutes.py`
- 생성: `tests/test_ingestion_smoke.py`

한국 정부의 "국가법령정보 OpenAPI"(`open.law.go.kr`)는 법령 데이터를 JSON 또는 XML 로 반환합니다. 단순한 캐시 fetch — 민법 전체 한 번 다운로드 — 를 `data/raw/statutes/civil_code.json` 로 지원합니다.

- [ ] **Step 1: `src/ingestion/__init__.py` 구현**

```python
"""Data ingestion: fetchers and chunkers."""
```

- [ ] **Step 2: `src/ingestion/fetch_statutes.py` 구현**

```python
"""Fetch the Korean Civil Code (民法) from the National Law Information Center OpenAPI.

Cached on-disk under data/raw/statutes/. Re-running with the cache present is a no-op.

Run: python -m src.ingestion.fetch_statutes
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.paths import RAW, ensure_dirs

log = get_logger("fetch_statutes")

API_BASE = "https://www.law.go.kr/DRF/lawService.do"
CIVIL_CODE_QUERY = "민법"


def _api_key() -> str:
    key = os.getenv("LAW_OPEN_API_KEY")
    if not key:
        raise RuntimeError(
            "LAW_OPEN_API_KEY is not set. Register at https://open.law.go.kr "
            "and put the key in .env."
        )
    return key


def _fetch_civil_code(api_key: str, retries: int = 3) -> dict:
    """Fetch civil code JSON. Retries with exponential backoff on transient errors."""
    params = {
        "OC": api_key,
        "target": "law",
        "type": "JSON",
        "MST": "001706",   # 민법 마스터 ID; if invalid, fall back to query.
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            # Some responses come wrapped in a single-key envelope.
            return payload
        except (requests.RequestException, ValueError) as e:
            last_err = e
            wait = 2 ** attempt
            log.warning("fetch attempt %d failed: %s (retry in %ds)", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"All {retries} fetch attempts failed: {last_err}")


def fetch_to_disk(out_path: Path | None = None, force: bool = False) -> Path:
    """Fetch civil code into data/raw/statutes/civil_code.json. Idempotent."""
    ensure_dirs()
    statutes_dir = RAW / "statutes"
    statutes_dir.mkdir(parents=True, exist_ok=True)
    out = out_path or statutes_dir / "civil_code.json"
    if out.exists() and not force:
        log.info("Cached: %s (use force=True to refetch)", out)
        return out

    payload = _fetch_civil_code(_api_key())
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %s (%.1f KB)", out, out.stat().st_size / 1024)
    return out


def main() -> int:
    cfg = load_config()
    log.info("Fetching civil code (source=%s)", cfg["ingestion"]["source"])
    fetch_to_disk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: `tests/test_ingestion_smoke.py` 에 mock 된 smoke test 작성**

```python
"""Smoke test for fetch_statutes — network is mocked."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion import fetch_statutes


def test_fetch_to_disk_writes_cache_and_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAW_OPEN_API_KEY", "dummy")
    fake_payload = {"law": {"name": "민법", "articles": [{"no": "618", "text": "임대차는 ..."}]}}

    calls = {"n": 0}

    def fake_fetch(api_key: str, retries: int = 3):
        calls["n"] += 1
        return fake_payload

    monkeypatch.setattr(fetch_statutes, "_fetch_civil_code", fake_fetch)

    out = tmp_path / "civil_code.json"
    fetch_statutes.fetch_to_disk(out_path=out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == fake_payload
    assert calls["n"] == 1

    # Idempotence: second call without force should not refetch.
    fetch_statutes.fetch_to_disk(out_path=out)
    assert calls["n"] == 1

    # force=True triggers refetch.
    fetch_statutes.fetch_to_disk(out_path=out, force=True)
    assert calls["n"] == 2


def test_fetch_to_disk_requires_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LAW_OPEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LAW_OPEN_API_KEY"):
        fetch_statutes._fetch_civil_code(fetch_statutes._api_key())
```

- [ ] **Step 4: smoke test 실행**

실행: `pytest tests/test_ingestion_smoke.py -v`
예상: 2 개 테스트 통과.

- [ ] **Step 5: 실제 fetch (수동, API key 필요)**

실행 (`.env` 에 `LAW_OPEN_API_KEY` 가 있을 때만):

```powershell
python -m src.ingestion.fetch_statutes
```

예상: `data/raw/statutes/civil_code.json` 이 존재하고 JSON 을 포함. 아직 키가 없다면 **이 단계를 건너뛰세요** — chunker(Task 8)는 fixture 기반 경로를 지원하며 Task 14 가 실제 데이터를 처리합니다.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/__init__.py src/ingestion/fetch_statutes.py tests/test_ingestion_smoke.py
git commit -m "feat(ingestion): civil-code fetcher with on-disk cache + retries"
```

---

## Task 8: Chunker (TDD)

**파일:**
- 생성: `src/ingestion/chunk.py`
- 생성: `tests/test_chunk.py`
- 생성: `tests/fixtures/sample_civil_code.json`
- 생성: `tests/fixtures/sample_chunks.jsonl` (테스트로 재생성됨)

chunker 는 raw 법령 payload 를 조문 단위 `Chunk` 레코드로 변환합니다. 결정론적입니다: 같은 입력 → 같은 출력, 안정적인 `id` 와 콘텐츠 `hash` 포함.

- [ ] **Step 1: 테스트 fixture `tests/fixtures/sample_civil_code.json` 생성**

```json
{
  "law": {
    "name": "민법",
    "articles": [
      {"no": "618", "title": "임대차의 의의", "text": "임대차는 당사자 일방이 상대방에게 목적물을 사용, 수익하게 할 것을 약정하고 상대방이 이에 대하여 차임을 지급할 것을 약정함으로써 그 효력이 생긴다."},
      {"no": "619", "title": "처분능력 없는 자의 임대차", "text": "처분의 능력 또는 권한 없는 자가 임대차를 하는 경우에는 그 임대차는 다음 각호의 기간을 넘지 못한다."},
      {"no": "stub", "title": "stub", "text": "짧음"}
    ]
  }
}
```

- [ ] **Step 2: `tests/test_chunk.py` 에 실패하는 테스트 작성**

```python
"""Tests for src/ingestion/chunk.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.schemas import Chunk
from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash, make_chunk_id

FIXTURES = Path(__file__).parent / "fixtures"


def _load_sample():
    return json.loads((FIXTURES / "sample_civil_code.json").read_text(encoding="utf-8"))


def test_chunk_civil_code_yields_chunks_per_article():
    chunks = chunk_civil_code(_load_sample(), min_chars=10)
    assert all(isinstance(c, Chunk) for c in chunks)
    nos = [c.statute_no for c in chunks]
    assert "민법 제618조" in nos
    assert "민법 제619조" in nos


def test_chunk_civil_code_drops_short_chunks():
    # min_chars=50 → the 'stub' article (text "짧음", 2 chars) is dropped.
    chunks = chunk_civil_code(_load_sample(), min_chars=50)
    assert all(len(c.text) >= 50 for c in chunks)
    assert all(c.statute_no != "민법 제stub조" for c in chunks)


def test_chunk_id_stable():
    a = make_chunk_id("nlic", "민법", "618", 0)
    b = make_chunk_id("nlic", "민법", "618", 0)
    assert a == b
    c = make_chunk_id("nlic", "민법", "618", 1)
    assert a != c


def test_dedupe_by_hash():
    chunks = chunk_civil_code(_load_sample(), min_chars=10)
    duplicated = chunks + chunks
    deduped = dedupe_by_hash(duplicated)
    assert len(deduped) == len(chunks)


def test_chunk_metadata_preserved():
    chunks = chunk_civil_code(_load_sample(), min_chars=10)
    art_618 = next(c for c in chunks if c.statute_no == "민법 제618조")
    assert art_618.doc_type == "조문"
    assert art_618.title == "임대차의 의의"
    assert art_618.source == "nlic"
    assert art_618.char_range[0] == 0
    assert art_618.char_range[1] == len(art_618.text)
    assert len(art_618.hash) == 16  # short SHA-1


def test_long_text_split_with_overlap_zero():
    # An article longer than max_chars should split into multiple sequential chunks.
    long_text = "가" * 3000
    payload = {"law": {"name": "민법", "articles": [{"no": "999", "title": "장문", "text": long_text}]}}
    chunks = chunk_civil_code(payload, max_chars=1000, min_chars=200)
    assert len(chunks) >= 3
    # Their concatenated text equals the original (no overlap).
    assert "".join(c.text for c in chunks) == long_text
```

- [ ] **Step 3: 테스트가 실패함을 확인하기 위해 실행**

실행: `pytest tests/test_chunk.py -v`
예상: collection error / `ModuleNotFoundError`.

- [ ] **Step 4: `src/ingestion/chunk.py` 구현**

```python
"""Chunk a Civil Code payload into per-article Chunk records.

Public API:
    chunk_civil_code(payload, *, max_chars=1200, min_chars=200) -> list[Chunk]
    dedupe_by_hash(chunks) -> list[Chunk]
    make_chunk_id(source, law_name, article_no, idx) -> str
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from src.common.logging import get_logger
from src.common.schemas import Chunk

log = get_logger("chunk")


def make_chunk_id(source: str, law_name: str, article_no: str, idx: int) -> str:
    raw = f"{source}|{law_name}|{article_no}|{idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _split_long(text: str, max_chars: int) -> list[tuple[int, int, str]]:
    """Split text into <=max_chars windows. Returns list of (start, end, sub)."""
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        out.append((pos, end, text[pos:end]))
        pos = end
    return out


_ARTICLE_NO_RE = re.compile(r"^[0-9]+(-[0-9]+)?$")


def chunk_civil_code(
    payload: dict[str, Any],
    *,
    max_chars: int = 1200,
    min_chars: int = 200,
) -> list[Chunk]:
    """Convert a raw Civil Code payload into Chunk records."""
    law = payload.get("law", {})
    law_name = law.get("name", "민법")
    articles: Iterable[dict[str, Any]] = law.get("articles", [])

    out: list[Chunk] = []
    for art in articles:
        no = str(art.get("no", "")).strip()
        title = art.get("title")
        text = (art.get("text") or "").strip()
        if not no or not text:
            continue
        statute_no = f"{law_name} 제{no}조"
        # Split if longer than max_chars; otherwise keep as one chunk.
        for idx, (start, end, sub) in enumerate(_split_long(text, max_chars)):
            if len(sub) < min_chars:
                continue
            cid = make_chunk_id("nlic", law_name, no, idx)
            out.append(
                Chunk(
                    id=cid,
                    source="nlic",
                    doc_type="조문",
                    statute_no=statute_no,
                    case_no=None,
                    title=title,
                    text=sub,
                    char_range=[start, end],
                    hash=_short_hash(sub),
                )
            )
    log.info("chunk_civil_code: %d chunks from %d articles", len(out), len(list(articles)) if isinstance(articles, list) else -1)
    return out


def dedupe_by_hash(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.hash in seen:
            continue
        seen.add(c.hash)
        out.append(c)
    return out
```

- [ ] **Step 5: 테스트가 통과함을 확인하기 위해 재실행**

실행: `pytest tests/test_chunk.py -v`
예상: 6 개 테스트 PASS.

- [ ] **Step 6: 실제 데이터 chunking 을 위한 CLI driver 추가**

`src/ingestion/chunk.py` 의 맨 아래에 추가:

```python
def main() -> int:
    """Read data/raw/statutes/civil_code.json → write data/processed/chunks.jsonl."""
    import json
    from src.common.config import load_config
    from src.common.paths import PROCESSED, RAW, ensure_dirs

    ensure_dirs()
    cfg = load_config()
    src = RAW / "statutes" / "civil_code.json"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run `make ingest` first.")

    payload = json.loads(src.read_text(encoding="utf-8"))
    chunks = dedupe_by_hash(
        chunk_civil_code(
            payload,
            max_chars=cfg["chunk"]["max_chars"],
            min_chars=cfg["chunk"]["min_chars"],
        )
    )
    out = PROCESSED / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    log.info("Wrote %d chunks to %s", len(chunks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/chunk.py tests/test_chunk.py tests/fixtures/sample_civil_code.json
git commit -m "feat(ingestion): chunker over Civil Code articles with dedup + length filter"
```

---

## Task 9: BM25 모듈 (TDD)

**파일:**
- 생성: `src/rag/__init__.py`
- 생성: `src/rag/bm25.py`
- 생성: `tests/test_bm25.py`

`rank_bm25.BM25Okapi` 위에 얇은 wrapper 입니다. 흥미로운 로직은 **tokenization** 입니다: 한국어 텍스트는 법률 용어(조문 번호, 법령명)의 BM25 에 유용하려면 최소한 character n-gram + 공백 fallback 이 필요합니다.

- [ ] **Step 1: `src/rag/__init__.py` 구현**

```python
"""RAG components: tokenization, embedding, indexing, retrieval."""
```

- [ ] **Step 2: `tests/test_bm25.py` 에 실패하는 테스트 작성**

```python
"""Tests for src/rag/bm25.py."""
from __future__ import annotations

import pytest

from src.common.schemas import Chunk
from src.rag.bm25 import BM25Index, tokenize


def _make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        source="nlic",
        doc_type="조문",
        statute_no=cid,
        text=text,
        char_range=[0, len(text)],
        hash=cid,
    )


def test_tokenize_keeps_korean_words_and_numbers():
    tokens = tokenize("민법 제618조 임대차의 의의")
    assert "민법" in tokens
    assert "제618조" in tokens or "618" in tokens
    assert "임대차의" in tokens or "임대차" in tokens
    # Lowercased ASCII.
    assert all(t == t.lower() for t in tokens if t.isascii())


def test_tokenize_drops_punctuation():
    tokens = tokenize("임대차, 보증금. 갱신!")
    assert "," not in tokens
    assert "." not in tokens
    assert "!" not in tokens


def test_bm25_returns_relevant_top_k():
    chunks = [
        _make_chunk("c1", "임대차는 당사자 일방이 상대방에게 목적물을 사용, 수익하게 할 것을 약정한다."),
        _make_chunk("c2", "매매는 당사자 일방이 재산권을 상대방에게 이전한다."),
        _make_chunk("c3", "임대차 계약의 갱신요구권은 임차인이 행사한다."),
    ]
    idx = BM25Index.build(chunks)
    hits = idx.search("임대차 갱신요구권", top_n=2)
    ids = [h.chunk.id for h in hits]
    assert "c3" in ids


def test_bm25_save_and_load(tmp_path):
    chunks = [
        _make_chunk("c1", "임대차의 의의"),
        _make_chunk("c2", "매매의 의의"),
    ]
    idx = BM25Index.build(chunks)
    p = tmp_path / "bm25.pkl"
    idx.save(p)
    loaded = BM25Index.load(p)
    a = idx.search("임대차", top_n=1)
    b = loaded.search("임대차", top_n=1)
    assert [h.chunk.id for h in a] == [h.chunk.id for h in b]
    assert pytest.approx(a[0].score) == b[0].score
```

- [ ] **Step 3: 테스트가 실패함을 확인하기 위해 실행**

실행: `pytest tests/test_bm25.py -v`
예상: `ModuleNotFoundError`.

- [ ] **Step 4: `src/rag/bm25.py` 구현**

```python
"""BM25 sparse index over Chunks.

Tokenization rules:
- Lowercase ASCII.
- Keep Korean syllables, ASCII letters, digits, and the characters '제', '호', '조' joined.
- Split on whitespace AND on standard punctuation.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from src.common.logging import get_logger
from src.common.schemas import Chunk

log = get_logger("bm25")

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation split, lowercased ASCII, Korean kept whole."""
    if not text:
        return []
    return [m.group(0).lower() if m.group(0).isascii() else m.group(0) for m in _TOKEN_RE.finditer(text)]


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi):
        self.chunks = chunks
        self._bm25 = bm25

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> "BM25Index":
        chunks = list(chunks)
        if not chunks:
            raise ValueError("BM25Index.build requires at least one chunk")
        tokenized = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        log.info("BM25Index: built over %d chunks", len(chunks))
        return cls(chunks, bm25)

    def search(self, query: str, top_n: int = 20) -> list[BM25Hit]:
        toks = tokenize(query)
        scores = self._bm25.get_scores(toks)
        ranked = sorted(zip(self.chunks, scores), key=lambda t: t[1], reverse=True)
        return [BM25Hit(chunk=c, score=float(s)) for c, s in ranked[:top_n]]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunks": [c.model_dump() for c in self.chunks], "bm25": self._bm25}, f)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        chunks = [Chunk.model_validate(d) for d in payload["chunks"]]
        return cls(chunks, payload["bm25"])
```

- [ ] **Step 5: 테스트 통과 확인을 위해 실행**

실행: `pytest tests/test_bm25.py -v`
예상: 4 개 테스트 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rag/__init__.py src/rag/bm25.py tests/test_bm25.py
git commit -m "feat(rag): BM25 sparse index with Korean-aware tokenizer"
```

---

## Task 10: Embedder (bge-m3 wrapper) — smoke 만

**파일:**
- 생성: `src/rag/embed.py`

bge-m3(`FlagEmbedding.BGEM3FlagModel`)는 dense, sparse, ColBERT 의 세 가지 신호를 반환합니다. 우리는 vector index 에 **dense 만** 사용합니다. Task 11 의 index 통합으로 테스트(모델을 로드하고, 5 개 문서를 embedding 하고, shape 를 assert)되며 — 별도 단위 테스트는 없습니다(FlagEmbedding 내부를 그저 실행할 뿐이기 때문).

- [ ] **Step 1: `src/rag/embed.py` 구현**

```python
"""Encode texts with bge-m3, returning dense vectors only.

Singleton model load: instantiating Embedder more than once is fine (caches the model).
On OOM, halve batch_size and retry once.
"""
from __future__ import annotations

import gc
from typing import Iterable

import numpy as np
import torch

from src.common.logging import get_logger

log = get_logger("embed")

_MODEL_CACHE: dict[str, "Embedder"] = {}


class Embedder:
    def __init__(self, model_id: str = "BAAI/bge-m3", batch_size: int = 8, device: str | None = None):
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading embedder %s on %s", model_id, self.device)
        self._model = BGEM3FlagModel(model_id, use_fp16=self.device == "cuda")
        self.dim = 1024  # bge-m3 dense dim

    @classmethod
    def get(cls, model_id: str = "BAAI/bge-m3", batch_size: int = 8) -> "Embedder":
        if model_id not in _MODEL_CACHE:
            _MODEL_CACHE[model_id] = cls(model_id, batch_size=batch_size)
        return _MODEL_CACHE[model_id]

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Returns float32 array of shape (len(texts), dim)."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = self.batch_size
        while True:
            try:
                out = self._model.encode(
                    texts,
                    batch_size=batch,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
                return vecs
            except torch.cuda.OutOfMemoryError:
                if batch <= 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                log.warning("OOM during embed; halving batch to %d", batch)
```

- [ ] **Step 2: Python REPL 에서 smoke check**

실행:

```powershell
python -c "from src.rag.embed import Embedder; e = Embedder(); v = e.encode(['임대차 갱신요구권', '매매 계약']); print(v.shape, v.dtype)"
```

예상: `(2, 1024) float32` (첫 실행 시 bge-m3 가중치 다운로드, ~1.5–2.0 GB).

- [ ] **Step 3: Commit**

```bash
git add src/rag/embed.py
git commit -m "feat(rag): bge-m3 dense embedder with OOM-aware batching"
```

---

## Task 11: ChromaDB index builder

**파일:**
- 생성: `src/rag/index.py`

`data/processed/chunks.jsonl` 로부터 `data/chroma/` collection 을 빌드하고, 같은 source 로부터 BM25 pickle 도 빌드합니다. 둘 다 함께 실행됩니다 — 같은 입력을 사용하며 동기화되어 있어야 하기 때문입니다.

- [ ] **Step 1: `src/rag/index.py` 구현**

참고: 경로는 `from paths import X` 가 아니라 `paths.X`(모듈 속성 접근)로 조회되므로, 테스트가 `monkeypatch.setattr(paths, "CHROMA", tmp)` 를 할 수 있습니다.

```python
"""Build the dense (ChromaDB) and sparse (BM25 pickle) indexes from chunks.jsonl.

Run: python -m src.rag.index
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.common import paths
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.schemas import Chunk
from src.rag.bm25 import BM25Index
from src.rag.embed import Embedder

log = get_logger("index")

COLLECTION = "civil_code"


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def build_indexes(chunks_path: Path | None = None, *, recreate: bool = True) -> tuple[Path, Path]:
    """Build both indexes. Returns (chroma_dir, bm25_path).

    Reads paths via `paths.X` so tests can monkeypatch `paths` attributes.
    """
    paths.ensure_dirs()
    cfg = load_config()
    chunks_path = chunks_path or paths.PROCESSED / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} missing — run `make ingest` then `python -m src.ingestion.chunk`.")

    chunks = load_chunks(chunks_path)
    log.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # --- Dense / Chroma ---
    client = chromadb.PersistentClient(path=str(paths.CHROMA), settings=Settings(allow_reset=True))
    if recreate:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    coll = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    embedder = Embedder.get(model_id=cfg["embedding"]["model_id"], batch_size=cfg["embedding"]["batch_size"])
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vecs = embedder.encode([c.text for c in batch])
        coll.add(
            ids=[c.id for c in batch],
            embeddings=vecs.tolist(),
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "statute_no": c.statute_no or "",
                    "case_no": c.case_no or "",
                    "doc_type": c.doc_type,
                    "title": c.title or "",
                }
                for c in batch
            ],
        )
        log.info("Indexed %d / %d", min(i + BATCH, len(chunks)), len(chunks))

    # --- Sparse / BM25 ---
    bm25 = BM25Index.build(chunks)
    bm25.save(paths.BM25_PATH)
    log.info("BM25 saved to %s", paths.BM25_PATH)

    return paths.CHROMA, paths.BM25_PATH


def main() -> int:
    build_indexes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 테스트 fixture 로 수동 smoke 실행**

Python REPL 에서 실행:

```python
from pathlib import Path
import json
from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash
from src.rag.index import build_indexes
from src.common.paths import PROCESSED, ensure_dirs

ensure_dirs()
payload = json.loads(Path("tests/fixtures/sample_civil_code.json").read_text(encoding="utf-8"))
chunks = dedupe_by_hash(chunk_civil_code(payload, min_chars=10))

out = PROCESSED / "chunks.jsonl"
with out.open("w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c.model_dump_json() + "\n")

build_indexes(out)
```

예상:
- `data/chroma/` 디렉터리에 영속화 파일이 존재.
- `data/bm25.pkl` 이 존재.
- 로그에 N == fixture chunk 수에 해당하는 "Indexed N / N" 표시.

- [ ] **Step 3: Commit**

```bash
git add src/rag/index.py
git commit -m "feat(rag): ChromaDB + BM25 index builder over chunks.jsonl"
```

---

## Task 12: RRF 를 사용한 Hybrid Retriever (TDD)

**파일:**
- 생성: `src/rag/retriever.py`
- 생성: `tests/test_retriever.py`

retriever 는 dense(ChromaDB)와 sparse(BM25)를 Reciprocal Rank Fusion 으로 융합합니다: `score(d) = Σ 1 / (k + rank_i(d))`.

- [ ] **Step 1: `tests/test_retriever.py` 에 실패하는 테스트 작성**

```python
"""Tests for src/rag/retriever.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.retriever import rrf_fuse

FIXTURES = Path(__file__).parent / "fixtures"


def test_rrf_fuse_orders_by_combined_score():
    # dense ranking: a, b, c
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    # sparse ranking: c, a, b — c wins on sparse
    sparse = [("c", 5.0), ("a", 3.0), ("b", 1.0)]
    fused = rrf_fuse(dense, sparse, k=60, top_n=3)
    ids = [d for d, _ in fused]
    # 'a' is in both top-1/top-2 → highest combined; 'c' top-1 sparse + top-3 dense; 'b' bottom in both.
    assert ids[0] == "a"
    assert ids[-1] == "b"


def test_rrf_fuse_handles_disjoint_lists():
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("c", 2.0)]
    fused = rrf_fuse(dense, sparse, k=60, top_n=5)
    ids = [d for d, _ in fused]
    assert set(ids) == {"a", "b", "c"}


def test_rrf_fuse_dedupes():
    dense = [("a", 0.9), ("a", 0.5)]   # pathological duplicate; pick best rank
    sparse = [("a", 1.0)]
    fused = rrf_fuse(dense, sparse, k=60, top_n=3)
    assert [d for d, _ in fused] == ["a"]


def test_rrf_fuse_top_n_limit():
    dense = [(str(i), 1.0 - i * 0.01) for i in range(20)]
    sparse = [(str(i), 1.0) for i in range(20)]
    fused = rrf_fuse(dense, sparse, k=60, top_n=5)
    assert len(fused) == 5


@pytest.mark.slow
def test_retriever_integration_with_fixture(tmp_path, monkeypatch):
    """Build small index from fixture, retrieve a lease-related query → expect 民法 第618條 chunk."""
    pytest.importorskip("FlagEmbedding")
    pytest.importorskip("chromadb")

    from src.common import paths as paths_mod

    monkeypatch.setattr(paths_mod, "DATA", tmp_path)
    monkeypatch.setattr(paths_mod, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths_mod, "CHROMA", tmp_path / "chroma")
    monkeypatch.setattr(paths_mod, "BM25_PATH", tmp_path / "bm25.pkl")

    from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash
    from src.rag.index import build_indexes
    from src.rag.retriever import Retriever

    paths_mod.ensure_dirs()
    payload = json.loads((FIXTURES / "sample_civil_code.json").read_text(encoding="utf-8"))
    chunks = dedupe_by_hash(chunk_civil_code(payload, min_chars=10))
    p = paths_mod.PROCESSED / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    build_indexes(p)
    r = Retriever.open()
    hits = r.search("임대차의 의의가 무엇인가?", k=2)
    assert any("618" in (h.statute_no or "") for h in hits)
```

integration test 는 실제 bge-m3 를 사용하므로 느립니다(첫 실행 시 ~30 초). 빠른 반복 동안에는 `pytest -k "not integration"` 으로 skip 가능하게 표시해도 괜찮습니다; CI 가 있다면 CI 에는 포함시키고자 합니다.

- [ ] **Step 2: 테스트가 실패함을 확인하기 위해 실행**

실행: `pytest tests/test_retriever.py -v -k "not integration"`
예상: `ModuleNotFoundError: src.rag.retriever`.

- [ ] **Step 3: `src/rag/retriever.py` 구현**

```python
"""Hybrid retriever: dense (ChromaDB) + sparse (BM25), fused via RRF.

Public API:
    Retriever.open() -> Retriever                 (loads persisted indexes)
    Retriever.search(query, k=5) -> list[Chunk]
    rrf_fuse(dense_ranked, sparse_ranked, k=60, top_n=5) -> list[(id, score)]

Run: python -m src.rag.retriever "<query>"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import chromadb

from src.common import paths
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.schemas import Chunk
from src.rag.bm25 import BM25Index
from src.rag.embed import Embedder
from src.rag.index import COLLECTION

log = get_logger("retriever")


@dataclass
class RetrievalHit:
    id: str
    text: str
    statute_no: str | None
    case_no: str | None
    doc_type: str
    score: float


def rrf_fuse(
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    *,
    k: int = 60,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Higher combined RRF score = better.

    dense_ranked / sparse_ranked are already sorted best-first.
    Duplicates within a single list keep the BEST rank (lowest index).
    """
    def best_ranks(ranked: list[tuple[str, float]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, (doc_id, _) in enumerate(ranked):
            if doc_id not in out:
                out[doc_id] = i
        return out

    dr = best_ranks(dense_ranked)
    sr = best_ranks(sparse_ranked)
    all_ids = set(dr) | set(sr)
    scored: list[tuple[str, float]] = []
    for doc_id in all_ids:
        s = 0.0
        if doc_id in dr:
            s += 1.0 / (k + dr[doc_id] + 1)
        if doc_id in sr:
            s += 1.0 / (k + sr[doc_id] + 1)
        scored.append((doc_id, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_n]


class Retriever:
    def __init__(self, chroma_collection, bm25: BM25Index, embedder: Embedder, cfg: dict):
        self._coll = chroma_collection
        self._bm25 = bm25
        self._embedder = embedder
        self._cfg = cfg
        self._chunk_by_id = {c.id: c for c in bm25.chunks}

    @classmethod
    def open(cls) -> "Retriever":
        """Open persisted indexes. Looks up paths via `paths.X` so tests can monkeypatch."""
        cfg = load_config()
        client = chromadb.PersistentClient(path=str(paths.CHROMA))
        coll = client.get_collection(name=COLLECTION)
        bm25 = BM25Index.load(paths.BM25_PATH)
        embedder = Embedder.get(
            model_id=cfg["embedding"]["model_id"],
            batch_size=cfg["embedding"]["batch_size"],
        )
        return cls(coll, bm25, embedder, cfg)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        cfg = self._cfg["retrieval"]
        k = k or cfg["top_k"]
        # Dense top-N
        qvec = self._embedder.encode([query])[0].tolist()
        dn = cfg["dense_top_n"]
        dense_res = self._coll.query(query_embeddings=[qvec], n_results=dn, include=["distances"])
        dense_ids = dense_res["ids"][0]
        # Convert distance → score (lower distance = better; we just need rank order).
        dense_ranked = [(doc_id, 1.0 - dist) for doc_id, dist in zip(dense_ids, dense_res["distances"][0])]
        # Sparse top-N
        sn = cfg["bm25_top_n"]
        sparse_hits = self._bm25.search(query, top_n=sn)
        sparse_ranked = [(h.chunk.id, h.score) for h in sparse_hits]
        # RRF
        fused = rrf_fuse(dense_ranked, sparse_ranked, k=cfg["rrf_k"], top_n=k)
        out: list[Chunk] = []
        for doc_id, _score in fused:
            chunk = self._chunk_by_id.get(doc_id)
            if chunk is not None:
                out.append(chunk)
        log.info("retriever: query=%r → %d hits", query[:60], len(out))
        return out


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag.retriever "<query>"', file=sys.stderr)
        return 2
    q = sys.argv[1]
    r = Retriever.open()
    hits = r.search(q)
    for i, h in enumerate(hits, 1):
        print(f"--- #{i} [{h.statute_no or h.case_no}] ---")
        print(h.text[:400])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트 실행**

실행: `pytest tests/test_retriever.py -v -k "not integration"`
예상: 4 개 단위 테스트 PASS (RRF 로직).

- [ ] **Step 5: integration test 실행 (느림, 첫 실행 시 bge-m3 가중치 다운로드)**

실행: `pytest tests/test_retriever.py::test_retriever_integration_with_fixture -v`
예상: PASS. 시간: 첫 실행 시 ~30–90 초 (모델 다운로드), 이후 ~10 초.

- [ ] **Step 6: Commit**

```bash
git add src/rag/retriever.py tests/test_retriever.py
git commit -m "feat(rag): hybrid retriever (dense + BM25) with RRF fusion"
```

---

## Task 13: 실제 민법으로 `make ingest`/`make embed` end-to-end 연결

**파일:** 신규 없음 — 시퀀싱 task.

이것은 다음 phase 들이 필요로 할 실제 산출물을 만드는 **수동 실행** 입니다. 아직 `LAW_OPEN_API_KEY` 가 없다면 건너뛰세요. M1 acceptance(Task 14)는 fixture 기반 또는 실제 데이터 검증 중 하나를 받아들입니다.

- [ ] **Step 1: `.env` 에 `LAW_OPEN_API_KEY` 가 설정되어 있는지 확인**

실행:

```powershell
.venv\Scripts\Activate.ps1
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('LAW_OPEN_API_KEY')))"
```

예상: `True`.

- [ ] **Step 2: 민법 fetch**

실행: `make ingest`
예상: `data/raw/statutes/civil_code.json` 이 존재 (크기 > 100 KB).

- [ ] **Step 3: Chunk**

실행: `python -m src.ingestion.chunk`
예상: `data/processed/chunks.jsonl` 이 존재, 라인 수 ≥ 1,000 (民法은 약 1,118 조).

- [ ] **Step 4: index 빌드**

실행: `make embed`
예상:
- `data/chroma/` 가 채워짐.
- `data/bm25.pkl` 이 존재.
- 로그에 Step 3 의 라인 수와 일치하는 N 으로 "Indexed N / N" 이 표시됨.
- embedding 동안 VRAM 이 4 GB 미만 유지(다른 셸에서 `nvidia-smi` 로 확인).

- [ ] **Step 5: 실제 데이터에서 retrieval spot-check**

실행:

```powershell
python -m src.rag.retriever "임대차계약 갱신요구권"
```

예상: top-5 가 民法 第618조, 第623조, 第643조, 第654조 또는 관련 임대차 조항을 포함. 출력은 사람이 읽을 수 있어야 함.

- [ ] **Step 6: Commit 없음**

(`data/` 는 gitignored 이므로 commit 할 것이 없습니다. 필요한 경우 이상 사항을 `README.md` "Status" 에 기록하세요.)

---

## Task 14: M1 acceptance gate (fixture 를 사용한 E2E 테스트)

**파일:**
- 생성: `tests/test_e2e_rag.py`

실제 네트워크에 의존하지 않는 end-to-end smoke — fixture `sample_civil_code.json` 사용. 이는 retriever 또는 chunker 가 나중에 변경될 때 회귀를 방어합니다.

- [ ] **Step 1: `tests/test_e2e_rag.py` 에 E2E 테스트 작성**

```python
"""End-to-end RAG smoke: fixture → chunks → indexes → retrieve → expect 民法 第618条."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.slow


def test_e2e_rag_fixture(tmp_path, monkeypatch):
    pytest.importorskip("FlagEmbedding")
    pytest.importorskip("chromadb")

    from src.common import paths as paths_mod

    monkeypatch.setattr(paths_mod, "DATA", tmp_path)
    monkeypatch.setattr(paths_mod, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths_mod, "CHROMA", tmp_path / "chroma")
    monkeypatch.setattr(paths_mod, "BM25_PATH", tmp_path / "bm25.pkl")

    from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash
    from src.rag.index import build_indexes
    from src.rag.retriever import Retriever

    paths_mod.ensure_dirs()
    payload = json.loads((FIXTURES / "sample_civil_code.json").read_text(encoding="utf-8"))
    chunks = dedupe_by_hash(chunk_civil_code(payload, min_chars=10))
    p = paths_mod.PROCESSED / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    build_indexes(p)
    r = Retriever.open()

    hits = r.search("임대차의 정의를 설명해 주세요.", k=3)
    assert hits, "retriever returned no chunks"
    assert any("제618조" in (h.statute_no or "") for h in hits), f"618 missing from {[h.statute_no for h in hits]}"
```

- [ ] **Step 2: `pyproject.toml` 에 slow 테스트 마킹**

`pyproject.toml` 의 `[tool.pytest.ini_options]` 아래에 추가:

```toml
markers = [
    "slow: tests that load real models (bge-m3, Qwen) — opt in",
]
```

- [ ] **Step 3: E2E 테스트 실행**

실행: `pytest tests/test_e2e_rag.py -v`
예상: PASS. 시간: 첫 실행 시 ~30–90 초 (모델 다운로드), 이후 ~10 초.

- [ ] **Step 4: 빠른 전체 테스트 스위트 실행**

실행: `pytest -m "not slow" -v`
예상: 모든 unit + integration 테스트가 빠르게 통과.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_rag.py pyproject.toml
git commit -m "test(rag): end-to-end fixture smoke for chunk → index → retrieve"
git tag m1-rag-baseline
```

- [ ] **Step 6: README "Status" 업데이트**

추가:

```markdown
- M0 — Bootstrap: ✅ done (commit 82651e0…)
- M1 — RAG Baseline: ✅ done (tag m1-rag-baseline)
- Next: M2 — Orchestrator + Gradio + base/RAG modes
```

```bash
git add README.md
git commit -m "docs: mark M0 + M1 done in README status"
```

---

# Plan 의 Spec 대비 커버리지 (self-review)

| Spec 섹션 | 커버하는 곳 |
|---|---|
| §2 Decisions Summary — base model | Task 5 (loader), `config/default.yaml` (Task 1) |
| §2 Decisions Summary — embedding bge-m3 | Task 10 |
| §2 Decisions Summary — ChromaDB | Task 11 |
| §2 Decisions Summary — Hybrid retrieval | Task 12 (RRF) |
| §3 Architecture — Ingestion subsystem | Tasks 7, 8 |
| §3 Architecture — RAG subsystem | Tasks 9–12 |
| §3 Architecture — Serving / Eval / Training | **범위 밖 (이후 plan)** |
| §4.1 Ingestion module | Tasks 7, 8 |
| §4.2 RAG module | Tasks 9–12 |
| §4.6 Common module | Tasks 2, 3 |
| §6 Repo layout | Task 1 (skeleton), Tasks 2–12 (파일들) |
| §7 Dependencies | Task 1 (`pyproject.toml`) |
| §8.1 VRAM OOM (embedding) | Task 10 (auto-halve), Task 13 Step 4 (검증) |
| §8.2 External API failures (statute) | Task 7 (retries + cache + idempotent fetch) |
| §8.3 Data quality gates (chunking) | Task 8 (dedupe, min_chars) |
| §8.5 Reproducibility | Task 8 (deterministic id + hash) |
| §9 Testing — TDD modules | Tasks 2 (schemas), 8 (chunk), 9 (BM25), 12 (retriever) |
| §9 Testing — smoke wrappers | Tasks 7 (mocked), 10 (manual REPL) |
| §9 Testing — E2E smoke | Task 14 |
| §10 M0 milestone | Tasks 1–6 |
| §10 M1 milestone | Tasks 7–14 |
| §11 Risk — Windows bnb compatibility | Task 5 Step 3 (WSL2 fallback) |
| §11 Risk — VRAM OOM | Task 10 (auto-halve), Task 5 (record VRAM) |

**이 plan 에 포함되지 않은 항목 (의도적, 이후 plan 으로 연기):**
- Eval set 수동 검토 (M4)
- QLoRA 학습 (M3)
- 합성 Q&A 생성 (M3)
- Gradio UI / orchestrator / prompt builder (M2)
- Citation checker (M2 — Gradio UI 표시에 필요)
- 판례 크롤러 (M5 만)

---

# 실행 노트

- **TDD 모듈:** Tasks 2, 8, 9, 12 — 테스트를 먼저 작성하고, 실패를 보고, 구현하고, 통과를 확인합니다.
- **Smoke 모듈:** Tasks 7 (mocked), 10 (manual), 11 (manual + Task 14 E2E). 독립 단위 테스트는 없습니다.
- **task 당 한 번의 commit** 이 최소 기준입니다 — 부분 상태가 일관성이 있다면 task 내 step 사이에 자유롭게 commit 하세요.
- **`make embed` 가 실제 corpus 에서 OOM 이면**: `.env` 에 `EMBED_BATCH=4` 를 두고 재시도하세요. Task 10 이 자동으로 절반으로 줄이지만, env 변수로 처음부터 더 낮게 시작할 수 있습니다.
- **Unsloth 가 Windows 에서 깨졌다면**: `verify_unsloth.py` (Task 5) 가 자동으로 vanilla transformers + bitsandbytes 로 fallback 합니다. M1 은 Unsloth 에 전혀 의존하지 않으므로(embedder + chroma + BM25 만 사용) Unsloth 가 깨져도 M3 (QLoRA 학습)만 막힐 뿐 M1 은 막히지 않습니다.

---

## 이 plan 이 성공적으로 완료될 때

다음을 갖게 됩니다:
- 민법 전체에 대한 동작하는 `python -m src.rag.retriever "<query>"`.
- 로컬 환경이 Qwen2.5-3B 4-bit 를 로드할 수 있다는 검증 (Unsloth 또는 transformers fallback).
- 빠른 경로에서 <5 초로 실행되며 chunking, BM25, RRF, schemas 를 완전히 운동시키는 pytest 스위트.
- 작은 fixture 에 묶인 재현 가능한 E2E smoke 테스트, regression guard 로 적합.

그런 다음 **Plan 2: M2 — Serving Orchestrator + Gradio (base / RAG modes)** 을 작성합니다.
