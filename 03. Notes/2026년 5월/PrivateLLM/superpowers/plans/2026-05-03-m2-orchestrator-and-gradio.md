# M2 — Orchestrator + Gradio (base / RAG modes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing M1 retriever to the M0 4-bit Qwen2.5-3B base model behind `Orchestrator.generate(query, mode)` and expose the result through a minimal Gradio UI that compares `base` vs `rag` answers side-by-side.

**Architecture:** A process-singleton `model_loader` owns the 4-bit base model and tokenizer. A pure-logic `prompt_builder` turns `(query, mode, chunks)` into chat messages with a token cap. An `Orchestrator` glues retriever + loader + prompt + generation behind a thread-locked entry point. `eval/citation_checker.py` (regex extraction + stub verification) lives under `eval/` per the MVP module graph but is consumed by the M2 UI. The Gradio app calls the orchestrator twice per submit (one per mode) under the orchestrator's internal lock.

**Tech Stack:** Python 3.12, transformers 4.49 + bitsandbytes (4-bit), pydantic, ChromaDB + bge-m3 + rank_bm25 (already wired by M1), Gradio.

**Reference docs to consult before coding:**
- `docs/superpowers/specs/2026-05-03-m2-orchestrator-design.md` — design spec for this plan
- `docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md` §4.4, §5, §10 — frozen MVP spec
- `docs/superpowers/m2-readiness.md` — pre-flight items A/B/C/D (all done 2026-05-03)
- `scripts/verify_unsloth.py`, `scripts/verify_combined_vram.py` — the Unsloth/transformers fallback already used for 4-bit loading; Task 1 extracts this into a shared helper

**Schemas already defined in `src/common/schemas.py` — REUSE, do not redefine:**
- `Chunk`, `Citation` (raw / normalized / kind), `Response` (answer / citations / retrieved / latency_ms: float).

---

## File map

**Create:**
- `src/serving/__init__.py`
- `src/serving/model_loader.py` — singleton 4-bit base model + tokenizer
- `src/serving/prompt_builder.py` — pure logic; chat-message builder
- `src/serving/orchestrator.py` — `Orchestrator.generate(query, mode)`
- `src/serving/app_gradio.py` — Gradio UI
- `src/eval/__init__.py`
- `src/eval/citation_checker.py` — regex extraction + stub verification
- `config/prompts/rag.txt`, `config/prompts/no_rag.txt` — system prompts
- `tests/test_prompt_builder.py`
- `tests/test_citation_checker.py`
- `tests/test_orchestrator.py`
- `tests/test_serving_integration.py` — slow, gated

**Modify:**
- `config/default.yaml` — add `serving:` section; remove dead top-level `paths:` and `chunk.overlap` keys
- `Makefile` — `serve` target wiring (already lines up; just verify)
- `README.md` — flip M2 row to ✅; add a "Try the Gradio UI" section

---

## Task 1: Extract shared 4-bit loader → `model_loader.get_base_model()`

**Files:**
- Create: `src/serving/__init__.py` (empty marker)
- Create: `src/serving/model_loader.py`
- Modify: `scripts/verify_unsloth.py:24-44` — replace inline fallback with import from `src.serving.model_loader`
- Modify: `scripts/verify_combined_vram.py:65-92` — same
- Test (slow): `tests/test_serving_integration.py::test_get_base_model_is_singleton`

- [ ] **Step 1.1: Write the failing slow integration test**

`tests/test_serving_integration.py`:

```python
"""Slow integration tests for the serving subsystem.

Run with: pytest -m slow -v tests/test_serving_integration.py
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_get_base_model_is_singleton():
    """Two calls return identical model + tokenizer instances; only one VRAM load."""
    pytest.importorskip("torch")
    import torch

    from src.serving.model_loader import get_base_model

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    m1, t1 = get_base_model()
    peak_after_first = torch.cuda.max_memory_allocated()
    m2, t2 = get_base_model()
    peak_after_second = torch.cuda.max_memory_allocated()

    assert m1 is m2
    assert t1 is t2
    # Singleton: second call must not increase peak VRAM
    assert peak_after_second == peak_after_first
```

- [ ] **Step 1.2: Run it to confirm it fails (module not yet importable)**

Run: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
Expected: ImportError / module not found.

- [ ] **Step 1.3: Write `src/serving/__init__.py`**

```python
"""Serving subsystem: orchestrator, model loader, prompt builder, Gradio UI."""
```

- [ ] **Step 1.4: Write `src/serving/model_loader.py`**

```python
"""Process-singleton loader for the 4-bit base model.

Tries Unsloth first; falls back to transformers + bitsandbytes when Unsloth's
torch>=2.5 dependency is unmet (the current pinned environment). The fallback
matches what scripts/verify_unsloth.py and scripts/verify_combined_vram.py
already do — this module is the single source of truth so all three sites
share one path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

from src.common.config import load_config
from src.common.logging import get_logger

log = get_logger("model_loader")

_FALLBACK_BASE_ID = "Qwen/Qwen2.5-3B-Instruct"  # plain HF id when Unsloth's bnb pack is unavailable


@dataclass
class _State:
    model: Any = None
    tokenizer: Any = None
    loader: str = ""


_state = _State()
_lock = threading.Lock()


def get_base_model() -> tuple[Any, Any]:
    """Return the singleton (model, tokenizer). Loads on first call."""
    if _state.model is not None:
        return _state.model, _state.tokenizer
    with _lock:
        if _state.model is not None:
            return _state.model, _state.tokenizer
        cfg = load_config()
        base_id = cfg["model"]["base_id"]
        max_seq = cfg["model"]["max_seq_len"]
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_id,
                max_seq_length=max_seq,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(model)
            _state.loader = "unsloth"
        except Exception as e:
            log.warning(
                "Unsloth load failed (%s); falling back to transformers + bitsandbytes.", e
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_BASE_ID)
            model = AutoModelForCausalLM.from_pretrained(
                _FALLBACK_BASE_ID,
                quantization_config=bnb,
                device_map="auto",
            )
            _state.loader = "transformers"
        _state.model = model
        _state.tokenizer = tokenizer
        log.info("get_base_model: loaded via %s", _state.loader)
        return model, tokenizer


def loader_name() -> str:
    """For diagnostics: which path produced the singleton ('unsloth' or 'transformers')."""
    return _state.loader
```

- [ ] **Step 1.5: Run the slow test (this actually loads the model — ~10-15 s)**

Run: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
Expected: PASS.

- [ ] **Step 1.6: Refactor `scripts/verify_unsloth.py` to use the shared loader**

Replace the inline try/except (lines ~22-44) with:

```python
from src.serving.model_loader import get_base_model, loader_name

# ... at top of main(), keep the timing print ...
t0 = time.time()
model, tokenizer = get_base_model()
log.info("Loaded via %s in %.1fs", loader_name(), time.time() - t0)
```

(Keep the rest of `main()` — VRAM logging, prompt, generate, etc. — unchanged.)

- [ ] **Step 1.7: Refactor `scripts/verify_combined_vram.py` to use the shared loader**

Replace the local `_load_qwen_4bit(cfg)` function and its call site with:

```python
from src.serving.model_loader import get_base_model, loader_name

# at the call site:
t0 = time.time()
model, tokenizer = get_base_model()
loader = loader_name()
log.info("Loaded Qwen2.5-3B 4-bit via %s in %.1fs", loader, time.time() - t0)
```

Delete the local `_load_qwen_4bit` function entirely.

- [ ] **Step 1.8: Sanity-run both refactored scripts to confirm parity**

Run:
```powershell
.venv/Scripts/python scripts/verify_unsloth.py
.venv/Scripts/python scripts/verify_combined_vram.py
```
Expected: both still PASS, peak VRAM still ≤ 5.5 GB.

- [ ] **Step 1.9: Run fast suite to ensure no regression**

Run: `.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 32 passed (unchanged from pre-Plan-2 baseline).

- [ ] **Step 1.10: Commit**

```powershell
git add src/serving/__init__.py src/serving/model_loader.py scripts/verify_unsloth.py scripts/verify_combined_vram.py tests/test_serving_integration.py
git commit -m "feat(serving): singleton 4-bit base model loader"
```

---

## Task 2: `prompt_builder.build_messages()` (TDD, pure logic)

**Files:**
- Create: `config/prompts/rag.txt`, `config/prompts/no_rag.txt`
- Create: `src/serving/prompt_builder.py`
- Test: `tests/test_prompt_builder.py`

- [ ] **Step 2.1: Write the prompt files**

`config/prompts/rag.txt`:

```text
당신은 한국 민법 전문가입니다. 아래 참고자료를 근거로 답변하고, 근거가 부족하면 "참고자료에 명시되지 않음"이라고 답하세요. 인용은 [조문번호] 또는 [판례번호] 형식으로 본문에 표시하세요. 본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

`config/prompts/no_rag.txt`:

```text
당신은 한국 민법 전문가입니다. 귀하의 일반 지식으로 답변하세요. 추측이 필요한 부분은 명시적으로 표시하세요. 본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

- [ ] **Step 2.2: Write the failing tests**

`tests/test_prompt_builder.py`:

```python
"""Unit tests for the M2 prompt builder. Pure logic — no GPU."""

from __future__ import annotations

import pytest

from src.common.schemas import Chunk
from src.serving.prompt_builder import build_messages


def _chunk(idx: str, statute_no: str | None, text: str) -> Chunk:
    return Chunk(
        id=idx,
        source="test",
        doc_type="조문",
        statute_no=statute_no,
        title=None,
        text=text,
        char_range=[0, len(text)],
        hash="x" * 16,
    )


class _StubTokenizer:
    """Counts tokens as words split on whitespace; deterministic, no model load."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def test_base_mode_omits_chunks():
    msgs = build_messages("질문입니다", mode="base", chunks=None, tokenizer=_StubTokenizer())
    assert any(m["role"] == "system" for m in msgs)
    assert msgs[-1]["role"] == "user"
    user = msgs[-1]["content"]
    assert "질문입니다" in user
    assert "참고자료" not in user


def test_rag_mode_formats_chunk_ids():
    chunks = [
        _chunk("c1", "618", "임대차는 당사자 일방이..."),
        _chunk("c2", "619", "임대차의 존속기간은..."),
    ]
    msgs = build_messages("질문", mode="rag", chunks=chunks, tokenizer=_StubTokenizer())
    user = msgs[-1]["content"]
    assert "[618]" in user
    assert "[619]" in user
    assert "임대차는 당사자 일방이..." in user


def test_rag_mode_falls_back_to_case_then_chunk_id():
    case_chunk = Chunk(
        id="c-case",
        source="test",
        doc_type="판례",
        statute_no=None,
        case_no="2019다12345",
        title=None,
        text="대법원 판시...",
        char_range=[0, 8],
        hash="y" * 16,
    )
    no_id_chunk = _chunk("c-anon", None, "anonymous")
    msgs = build_messages(
        "질문", mode="rag", chunks=[case_chunk, no_id_chunk], tokenizer=_StubTokenizer()
    )
    user = msgs[-1]["content"]
    assert "[2019다12345]" in user
    assert "[c-anon]" in user


def test_token_cap_drops_lowest_ranked_chunks_first():
    # Each chunk has a distinct word so we can see who survives the cap.
    chunks = [_chunk(f"c{i}", str(600 + i), f"alpha{i} " * 50) for i in range(10)]
    msgs = build_messages(
        "q", mode="rag", chunks=chunks, tokenizer=_StubTokenizer(), max_context_tokens=80
    )
    user = msgs[-1]["content"]
    # The first chunks (highest-ranked) must remain
    assert "alpha0" in user
    # The last chunks (lowest-ranked) must be dropped
    assert "alpha9" not in user


def test_base_mode_ignores_chunks_argument():
    chunks = [_chunk("c1", "618", "should not appear")]
    msgs = build_messages("질문", mode="base", chunks=chunks, tokenizer=_StubTokenizer())
    user = msgs[-1]["content"]
    assert "should not appear" not in user
    assert "참고자료" not in user


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        build_messages("q", mode="qlora", chunks=None, tokenizer=_StubTokenizer())  # type: ignore[arg-type]
```

- [ ] **Step 2.3: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_prompt_builder.py -v`
Expected: ImportError or test failures (module not yet implemented).

- [ ] **Step 2.4: Implement `src/serving/prompt_builder.py`**

```python
"""Pure logic: turn (query, mode, chunks) into chat-template messages.

No model load, no GPU. The tokenizer arg is used only for the token cap;
tests pass a stub. The system prompt text comes from config/prompts/.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from src.common.paths import CONFIG_DIR
from src.common.schemas import Chunk

Mode = Literal["base", "rag"]


class _Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def _load_system_prompt(mode: Mode) -> str:
    fname = "rag.txt" if mode == "rag" else "no_rag.txt"
    return (CONFIG_DIR / "prompts" / fname).read_text(encoding="utf-8").strip()


def _chunk_label(c: Chunk) -> str:
    return c.statute_no or c.case_no or c.id


def _format_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{_chunk_label(c)}] {c.text}" for c in chunks)


def _fit_chunks(
    chunks: list[Chunk], tokenizer: _Tokenizer, max_context_tokens: int
) -> list[Chunk]:
    """Drop the lowest-ranked (= last) chunks until the formatted block fits."""
    fit = list(chunks)
    while fit:
        block = _format_block(fit)
        if len(tokenizer.encode(block, add_special_tokens=False)) <= max_context_tokens:
            return fit
        fit.pop()  # drop the lowest-ranked chunk
    return []


def build_messages(
    query: str,
    *,
    mode: Mode,
    chunks: list[Chunk] | None,
    tokenizer: Any,
    max_context_tokens: int = 1500,
) -> list[dict[str, str]]:
    """Build the message list for `tokenizer.apply_chat_template(...)`.

    For `mode="rag"` chunks are formatted as `[조문/판례번호] 본문` blocks and
    bound by `max_context_tokens` (lowest-ranked dropped first). For
    `mode="base"` the chunks argument is ignored.
    """
    if mode not in ("base", "rag"):
        raise ValueError(f"unsupported mode {mode!r}")
    system = _load_system_prompt(mode)
    if mode == "rag" and chunks:
        fit = _fit_chunks(chunks, tokenizer, max_context_tokens)
        block = _format_block(fit) if fit else "(검색 결과 없음)"
        user = f"<참고자료>\n{block}\n</참고자료>\n\n질문: {query}"
    else:
        user = query
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 2.5: Run the tests to confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_prompt_builder.py -v`
Expected: 6 passed.

- [ ] **Step 2.6: Run full fast suite**

Run: `.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 38 passed (32 prior + 6 new).

- [ ] **Step 2.7: Commit**

```powershell
git add src/serving/prompt_builder.py tests/test_prompt_builder.py config/prompts/
git commit -m "feat(serving): prompt_builder for base/rag modes with token cap"
```

---

## Task 3: `eval/citation_checker.py` (TDD, pure regex)

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/citation_checker.py`
- Test: `tests/test_citation_checker.py`

- [ ] **Step 3.1: Write the failing tests**

`tests/test_citation_checker.py`:

```python
"""Unit tests for the citation extractor. Pure regex; no I/O."""

from __future__ import annotations

from src.eval.citation_checker import extract_citations, verify_citations


def test_extract_statute_with_민법_prefix():
    text = "본 답변은 [민법 제618조]를 근거로 한다."
    cites = extract_citations(text)
    assert len(cites) == 1
    c = cites[0]
    assert c.kind == "statute"
    assert c.normalized == "618"


def test_extract_statute_bare_form():
    text = "임대차는 제618조에 정의되어 있고 제619조도 관련된다."
    cites = extract_citations(text)
    norms = sorted(c.normalized for c in cites)
    assert norms == ["618", "619"]
    assert all(c.kind == "statute" for c in cites)


def test_extract_statute_with_의_subsection():
    text = "[민법 제643조의2]"
    cites = extract_citations(text)
    assert len(cites) == 1
    assert cites[0].normalized == "643의2"


def test_extract_case_citation():
    text = "참고: 대법원 2019다12345 판결"
    cites = extract_citations(text)
    assert len(cites) == 1
    c = cites[0]
    assert c.kind == "case"
    assert c.normalized == "2019다12345"


def test_extract_multiple_mixed():
    text = "[민법 제618조]와 [민법 제619조]에 따라, 대법원 2020가합54321 판결도 참고된다."
    cites = extract_citations(text)
    kinds = sorted((c.kind, c.normalized) for c in cites)
    assert ("case", "2020가합54321") in kinds
    assert ("statute", "618") in kinds
    assert ("statute", "619") in kinds


def test_extract_does_not_match_chapter_or_section():
    text = "제3장 제2절을 참고하라. 제5호도 보라."  # 장/절/호 — not 조
    cites = extract_citations(text)
    assert cites == []


def test_extract_dedupes_within_a_text():
    text = "제618조, [민법 제618조], 제618조"
    cites = extract_citations(text)
    norms = [c.normalized for c in cites]
    # Same statute, multiple mentions — should produce one entry
    assert norms == ["618"]


def test_verify_stub_marks_all_found():
    text = "[민법 제618조]"
    cites = extract_citations(text)
    verified = verify_citations(cites, corpus=None)
    assert all(getattr(v, "found_in_corpus", True) for v in verified)


def test_verify_with_corpus_distinguishes_known_vs_unknown():
    """When a corpus is provided, only ids in the corpus are marked found."""
    text = "[민법 제618조]와 [민법 제9999조]"
    cites = extract_citations(text)
    verified = verify_citations(cites, corpus={"618"})
    by_norm = {v.cite.normalized: v.found_in_corpus for v in verified}
    assert by_norm == {"618": True, "9999": False}
```

- [ ] **Step 3.2: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_citation_checker.py -v`
Expected: ImportError.

- [ ] **Step 3.3: Implement `src/eval/__init__.py`**

```python
"""Evaluation subsystem. M2 ships only citation_checker; runner/judge land in M4."""
```

- [ ] **Step 3.4: Implement `src/eval/citation_checker.py`**

```python
"""Regex extraction of citations from a model answer.

Handles two citation kinds the MVP design pins:
  - statute: `[민법 제618조]`, `제618조`, `[민법 제643조의2]`
  - case   : `대법원 2019다12345`, `2020가합54321`, etc.

Verification against a corpus is provided as a stub for M2 (always True or
membership-checked against an in-memory set). M4 will replace `corpus` with
a real index built from `data/processed/chunks.jsonl`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.common.schemas import Citation

# Statute patterns:
#   group 1 = article number
#   group 2 = optional `의<n>` sub-article (`643조의2` -> normalized `643의2`)
_STATUTE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
# Case pattern: 4-digit year + Korean filing-type token + digit run.
# Examples: 2019다12345, 2020가합54321, 2018두99
_CASE_RE = re.compile(r"(\d{4}[가-힣]+\d+)")


@dataclass(frozen=True)
class VerifiedCitation:
    cite: Citation
    found_in_corpus: bool


def _normalize_statute(article: str, sub: str | None) -> str:
    return f"{article}의{sub}" if sub else article


def extract_citations(text: str) -> list[Citation]:
    """Extract citations from a model answer. Order is statute-first then case;
    within each kind, deduped by `normalized`."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []

    for m in _STATUTE_RE.finditer(text):
        norm = _normalize_statute(m.group(1), m.group(2))
        key = ("statute", norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), normalized=norm, kind="statute"))

    for m in _CASE_RE.finditer(text):
        key = ("case", m.group(1))
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), normalized=m.group(1), kind="case"))

    return out


def verify_citations(
    citations: Iterable[Citation],
    corpus: set[str] | None = None,
) -> list[VerifiedCitation]:
    """Annotate each citation with corpus membership.

    M2 stub: when `corpus is None` everything is marked found (True). When a
    set of normalized ids is passed in, membership is checked against it.
    M4's runner will swap in the real `CorpusIndex`.
    """
    out: list[VerifiedCitation] = []
    for c in citations:
        found = True if corpus is None else c.normalized in corpus
        out.append(VerifiedCitation(cite=c, found_in_corpus=found))
    return out
```

- [ ] **Step 3.5: Run the tests to confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_citation_checker.py -v`
Expected: 9 passed.

- [ ] **Step 3.6: Run full fast suite**

Run: `.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 47 passed (38 prior + 9 new).

- [ ] **Step 3.7: Commit**

```powershell
git add src/eval/__init__.py src/eval/citation_checker.py tests/test_citation_checker.py
git commit -m "feat(eval): citation_checker — statute/case regex + stub verify"
```

---

## Task 4: `Orchestrator.generate(query, mode)` with mocked-model unit tests

**Files:**
- Create: `src/serving/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 4.1: Write the failing unit tests with mocked dependencies**

`tests/test_orchestrator.py`:

```python
"""Unit tests for the M2 Orchestrator with mocked model + retriever.

The slow real-model integration test lives in tests/test_serving_integration.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.common.schemas import Chunk
from src.serving.orchestrator import Orchestrator


def _chunk(idx: str, statute_no: str, text: str) -> Chunk:
    return Chunk(
        id=idx,
        source="t",
        doc_type="조문",
        statute_no=statute_no,
        title=None,
        text=text,
        char_range=[0, len(text)],
        hash="z" * 16,
    )


class _FakeTokenizer:
    """Returns ints for `apply_chat_template` and decodes a fixed reply."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=True):
        import torch  # local import to keep this lightweight when CUDA absent

        # Encode just the user content as 1-token-per-word ints so generate() has something to extend.
        text = messages[-1]["content"]
        return torch.tensor([[i for i, _ in enumerate(text.split())]])

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return "임대차의 정의는 [민법 제618조]에 명시되어 있다."


class _FakeModel:
    device = "cpu"

    def generate(self, inputs, max_new_tokens=512, do_sample=False, **_):
        # Return inputs concatenated with one extra token id; the fake tokenizer.decode
        # ignores the ids and returns a fixed reply.
        import torch
        return torch.cat([inputs, torch.tensor([[42]])], dim=-1)


def test_base_mode_skips_retriever():
    retriever = MagicMock()
    orch = Orchestrator(retriever=retriever, model=_FakeModel(), tokenizer=_FakeTokenizer())
    resp = orch.generate("질문", mode="base")

    assert resp.mode == "base"
    assert resp.retrieved == []
    retriever.search.assert_not_called()
    # Citation extracted from the fake tokenizer's fixed answer
    assert any(c.normalized == "618" for c in resp.citations)
    assert resp.latency_ms >= 0


def test_rag_mode_calls_retriever_and_records_chunks():
    retriever = MagicMock()
    chunks = [_chunk("c1", "618", "임대차는 ...")]
    retriever.search.return_value = chunks
    orch = Orchestrator(retriever=retriever, model=_FakeModel(), tokenizer=_FakeTokenizer())

    resp = orch.generate("임대차?", mode="rag")

    retriever.search.assert_called_once()
    args, kwargs = retriever.search.call_args
    assert "임대차?" in args or kwargs.get("query") == "임대차?"
    assert resp.retrieved == chunks
    assert any(c.normalized == "618" for c in resp.citations)


def test_invalid_mode_raises():
    orch = Orchestrator(
        retriever=MagicMock(), model=_FakeModel(), tokenizer=_FakeTokenizer()
    )
    with pytest.raises(ValueError):
        orch.generate("q", mode="qlora")  # type: ignore[arg-type]


def test_generate_is_serialized_under_a_lock():
    """Two threads racing on generate() must observe a serialized model.generate call count."""
    import threading

    model = _FakeModel()
    call_count = {"n": 0, "max_in_flight": 0, "in_flight": 0}
    lock = threading.Lock()

    real_generate = model.generate

    def tracked_generate(inputs, **kwargs):
        with lock:
            call_count["in_flight"] += 1
            call_count["max_in_flight"] = max(call_count["max_in_flight"], call_count["in_flight"])
        try:
            import time
            time.sleep(0.05)
            return real_generate(inputs, **kwargs)
        finally:
            with lock:
                call_count["in_flight"] -= 1
                call_count["n"] += 1

    model.generate = tracked_generate  # type: ignore[method-assign]
    orch = Orchestrator(retriever=MagicMock(), model=model, tokenizer=_FakeTokenizer())

    threads = [threading.Thread(target=lambda: orch.generate("q", mode="base")) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 4
    assert call_count["max_in_flight"] == 1, "Orchestrator must serialize model.generate"
```

- [ ] **Step 4.2: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_orchestrator.py -v`
Expected: ImportError.

- [ ] **Step 4.3: Implement `src/serving/orchestrator.py`**

```python
"""Orchestrator: the single entry point the Gradio app and (later) eval runner call.

`Orchestrator.generate(query, mode)` ties together:
  - retriever (rag mode only)
  - prompt_builder
  - tokenized chat template + model.generate
  - citation extraction over the answer

`model.generate(...)` is not safe to call from multiple threads on the same
instance, so a per-orchestrator lock serializes generation. For a 1-user PoC
this is fine; the contention is documented in the M2 spec.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.schemas import Chunk, Citation, Response
from src.eval.citation_checker import extract_citations
from src.serving.model_loader import get_base_model
from src.serving.prompt_builder import build_messages

log = get_logger("orchestrator")

Mode = Literal["base", "rag"]


@dataclass
class Orchestrator:
    retriever: Any  # src.rag.retriever.Retriever, but kept Any so unit tests mock freely
    model: Any
    tokenizer: Any
    max_new_tokens: int = 512
    max_context_tokens: int = 1500
    top_k: int = 5
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def open(cls) -> "Orchestrator":
        from src.rag.retriever import Retriever  # local import: avoid GPU at module-import

        cfg = load_config()
        model, tokenizer = get_base_model()
        retriever = Retriever.open()
        serv = cfg.get("serving", {})
        return cls(
            retriever=retriever,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=serv.get("max_new_tokens", 512),
            max_context_tokens=serv.get("max_context_tokens", 1500),
            top_k=cfg["retrieval"]["top_k"],
        )

    def generate(self, query: str, mode: Mode) -> Response:
        if mode not in ("base", "rag"):
            raise ValueError(f"unsupported mode {mode!r}")
        t0 = time.time()
        chunks: list[Chunk] = []
        if mode == "rag":
            chunks = self.retriever.search(query, k=self.top_k)
        messages = build_messages(
            query,
            mode=mode,
            chunks=chunks,
            tokenizer=self.tokenizer,
            max_context_tokens=self.max_context_tokens,
        )
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        # Move to model device when present (real model has a device attr).
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        with self._lock:
            out = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen_ids = out[0][inputs.shape[1] :]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        citations: list[Citation] = extract_citations(answer)
        latency_ms = (time.time() - t0) * 1000.0
        return Response(
            answer=answer,
            citations=citations,
            retrieved=chunks,
            latency_ms=latency_ms,
        ).model_copy(update={})  # ensure a fresh instance

    # The schema's Response doesn't carry `mode`; expose it via response.model_dump shim
    # below since the UI wants to label columns. We return a wrapper-friendly object.
```

Note: `Response` per `src/common/schemas.py` does **not** include a `mode` field — but the unit test `test_base_mode_skips_retriever` asserts `resp.mode == "base"`. To keep the schema stable, we add `mode` to the orchestrator's local return contract by extending Response in-place: change the test's assertion path to read `mode` off a wrapper, OR (simpler) extend `Response` in schemas.py with an optional `mode` field. Pick the schema extension because the eval runner (M4) will also want it.

- [ ] **Step 4.4: Extend `src/common/schemas.py` to add `mode` to `Response`**

Edit `src/common/schemas.py:63-68`:

```python
class Response(BaseModel):
    answer: str
    mode: Optional[Literal["base", "rag", "qlora", "rag_qlora"]] = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[Chunk] = Field(default_factory=list)
    latency_ms: float
```

Update `src/serving/orchestrator.py` to set `mode=mode` on the returned `Response`:

```python
return Response(
    answer=answer,
    mode=mode,
    citations=citations,
    retrieved=chunks,
    latency_ms=latency_ms,
)
```

(Drop the `.model_copy(update={})` call.)

- [ ] **Step 4.5: Run the tests to confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_orchestrator.py -v`
Expected: 4 passed.

- [ ] **Step 4.6: Run the full fast suite**

Run: `.venv/Scripts/python -m pytest -m "not slow" -q`
Expected: 51 passed (47 prior + 4 new).

- [ ] **Step 4.7: Commit**

```powershell
git add src/serving/orchestrator.py src/common/schemas.py tests/test_orchestrator.py
git commit -m "feat(serving): Orchestrator.generate(base|rag) with thread-locked gen"
```

---

## Task 5: Slow real-model orchestrator integration test

**Files:**
- Modify: `tests/test_serving_integration.py`

- [ ] **Step 5.1: Append the integration test**

Append to `tests/test_serving_integration.py`:

```python
@pytest.mark.slow
def test_orchestrator_real_models_round_trip(tmp_path, monkeypatch):
    """Full E2E: real bge-m3 + Qwen 4-bit, run base + rag for the lease query.

    Uses the test fixture as the chunk corpus (same approach as the existing
    test_e2e_rag.py).
    """
    pytest.importorskip("torch")
    pytest.importorskip("FlagEmbedding")
    pytest.importorskip("chromadb")
    import json
    import torch

    from src.common import paths as paths_mod
    from src.ingestion.chunk import chunk_civil_code, dedupe_by_hash
    from src.rag.index import build_indexes
    from src.serving.orchestrator import Orchestrator

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    monkeypatch.setattr(paths_mod, "DATA", tmp_path)
    monkeypatch.setattr(paths_mod, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths_mod, "CHROMA", tmp_path / "chroma")
    monkeypatch.setattr(paths_mod, "BM25_PATH", tmp_path / "bm25.pkl")
    paths_mod.ensure_dirs()

    fixture = paths_mod.ROOT / "tests" / "fixtures" / "sample_civil_code.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    chunks = dedupe_by_hash(chunk_civil_code(payload, min_chars=10))
    p = paths_mod.PROCESSED / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    build_indexes(p)

    orch = Orchestrator.open()

    base_resp = orch.generate("임대차의 정의를 설명해 주세요.", mode="base")
    rag_resp = orch.generate("임대차의 정의를 설명해 주세요.", mode="rag")

    assert base_resp.mode == "base"
    assert base_resp.answer.strip() != ""
    assert base_resp.retrieved == []

    assert rag_resp.mode == "rag"
    assert rag_resp.answer.strip() != ""
    # `statute_no` is the human-readable label "민법 제618조" (not the bare "618")
    # — see the existing test_e2e_rag.py convention.
    assert any("제618조" in (c.statute_no or "") for c in rag_resp.retrieved)
    assert rag_resp.latency_ms > 0
```

- [ ] **Step 5.2: Run the slow suite**

Run: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
Expected: 2 passed (the singleton test from Task 1, plus this one). Total runtime ~30-60 s.

- [ ] **Step 5.3: Run the combined VRAM verification once more (regression gate)**

Run: `.venv/Scripts/python scripts/verify_combined_vram.py`
Expected: PASS, peak < 5.5 GB (should still be ~3.2 GB).

- [ ] **Step 5.4: Commit**

```powershell
git add tests/test_serving_integration.py
git commit -m "test(serving): real-model E2E for orchestrator base+rag"
```

---

## Task 6: Gradio UI (`app_gradio.py`)

**Files:**
- Create: `src/serving/app_gradio.py`
- Modify: `config/default.yaml` (add `serving:` block; remove dead `paths:` and `chunk.overlap`)

- [ ] **Step 6.1: Update `config/default.yaml`**

Replace the file with:

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
  # NB: chunk.overlap was a dead key — chunking is per-article so overlap is always 0.

ingestion:
  source: nlic   # 국가법령정보 OpenAPI
  law_id: 001706 # 민법 법령 ID

serving:
  max_new_tokens: 512
  max_context_tokens: 1500
  do_sample: false

logging:
  level: INFO
```

(The dead top-level `paths:` block and `chunk.overlap` key are removed; they were flagged as minor cleanup in `m2-readiness.md`.)

- [ ] **Step 6.2: Write `src/serving/app_gradio.py`**

```python
"""Minimal Gradio UI: side-by-side base vs rag answers for the same query.

Run: python -m src.serving.app_gradio
   (or: make serve)

Bind: 127.0.0.1:7860, share=False — local PoC only.
"""

from __future__ import annotations

import gradio as gr

from src.common.logging import get_logger
from src.eval.citation_checker import verify_citations
from src.serving.orchestrator import Orchestrator

log = get_logger("app_gradio")

_ORCH: Orchestrator | None = None


def _get_orch() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        log.info("Cold-loading orchestrator (this can take ~10-15 s)...")
        _ORCH = Orchestrator.open()
    return _ORCH


def _format_citations(citations) -> str:
    verified = verify_citations(citations, corpus=None)  # M2 stub: all True
    if not verified:
        return "_(no citations)_"
    parts = []
    for v in verified:
        badge = "✓" if v.found_in_corpus else "✗"
        parts.append(f"- {badge} `[{v.cite.kind}] {v.cite.normalized}` (raw: {v.cite.raw!r})")
    return "\n".join(parts)


def _format_chunks(chunks) -> str:
    if not chunks:
        return "_(no retrieved chunks)_"
    parts = []
    for c in chunks:
        tag = c.statute_no or c.case_no or c.id
        snippet = c.text[:240] + ("..." if len(c.text) > 240 else "")
        parts.append(f"**[{tag}]** {snippet}")
    return "\n\n".join(parts)


def run_query(query: str):
    if not query or not query.strip():
        empty = ("_(질문을 입력하세요)_", "", "")
        return empty + empty
    orch = _get_orch()
    base = orch.generate(query, mode="base")
    rag = orch.generate(query, mode="rag")
    return (
        base.answer,
        _format_citations(base.citations),
        f"latency: {base.latency_ms:.0f} ms",
        rag.answer,
        _format_citations(rag.citations) + "\n\n---\n**Retrieved:**\n" + _format_chunks(rag.retrieved),
        f"latency: {rag.latency_ms:.0f} ms",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Legal PrivateLLM — base vs RAG") as demo:
        gr.Markdown("# Legal PrivateLLM — base vs RAG\nKorean civil-law (민법) Q&A PoC.")
        with gr.Row():
            query = gr.Textbox(
                label="질문",
                placeholder="예: 임대차의 정의를 설명해 주세요.",
                lines=3,
            )
        run_btn = gr.Button("Run")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 base (no RAG)")
                base_answer = gr.Markdown()
                base_cites = gr.Markdown()
                base_meta = gr.Markdown()
            with gr.Column():
                gr.Markdown("### 🟢 rag")
                rag_answer = gr.Markdown()
                rag_cites = gr.Markdown()
                rag_meta = gr.Markdown()
        run_btn.click(
            run_query,
            inputs=[query],
            outputs=[base_answer, base_cites, base_meta, rag_answer, rag_cites, rag_meta],
        )
    return demo


def main() -> int:
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6.3: Manual smoke test**

Run: `.venv/Scripts/python -m src.serving.app_gradio`

Expected:
- Console prints "Cold-loading orchestrator..."
- Browser opens at `http://127.0.0.1:7860`
- Entering "임대차의 정의를 설명해 주세요." and pressing Run yields TWO non-empty Korean answers within ~10 s
- The RAG column shows at least one retrieved chunk tagged `[618]`
- The citations area lists `[statute] 618` for the rag column

Stop the server with Ctrl-C.

- [ ] **Step 6.4: Verify `make serve` still wires correctly**

The existing Makefile has `serve: python -m src.serving.app_gradio` — confirm by running `make serve` (just for resolution; immediately Ctrl-C).

- [ ] **Step 6.5: Run the full test suite (fast + slow)**

Run: `.venv/Scripts/python -m pytest -q`
Expected: All tests pass. Total: 51 fast + ~4 slow.

- [ ] **Step 6.6: Commit**

```powershell
git add src/serving/app_gradio.py config/default.yaml
git commit -m "feat(serving): Gradio UI — side-by-side base vs RAG with citation badges"
```

---

## Task 7: README + status updates

**Files:**
- Modify: `README.md` — flip M2 row to ✅; add a "Try the Gradio UI" section
- Modify: `docs/superpowers/m2-readiness.md` — append a "Plan 2 closed" section

- [ ] **Step 7.1: Update `README.md` status table**

Change the M2 row from "next plan" to "✅ done — tag `m2-orchestrator`":

```markdown
| **M2 — Orchestrator + Gradio (base / RAG modes)** | ✅ done — tag `m2-orchestrator` |
```

- [ ] **Step 7.2: Add a new section to `README.md` after "Try the retriever (after M1)"**

```markdown
## Try the Gradio UI (after M2)

```powershell
make serve                            # opens http://127.0.0.1:7860
```

The UI runs the same query through `base` and `rag` modes side-by-side, with extracted citations and retrieved chunk previews. First load is ~10-15 s while the singleton model + retriever spin up.
```

- [ ] **Step 7.3: Append a "Plan 2 closed" section to `docs/superpowers/m2-readiness.md`**

Append:

```markdown
---

## Plan 2 — closed (2026-05-03)

**Tag:** `m2-orchestrator`
**Tests added:** prompt_builder (6), citation_checker (9), orchestrator unit (4), serving integration slow (2). Full fast suite: 51 passed.

**Modules delivered:**
- `src/serving/{model_loader, prompt_builder, orchestrator, app_gradio}.py`
- `src/eval/citation_checker.py` (regex extractor + stub verify; M4 will plug in the real corpus index)
- `config/prompts/{rag,no_rag}.txt`

**Acceptance signals captured:**
- Gradio UI returns both columns for "임대차의 정의를 설명해 주세요." in <15 s on the target host.
- RAG column retrieves 民法 第618條 reliably.
- `verify_combined_vram.py` peak unchanged at ~3.2 GB.
```

- [ ] **Step 7.4: Tag the milestone**

```powershell
git tag m2-orchestrator -m "M2: orchestrator + Gradio (base/RAG)"
```

(Do NOT push — tag stays local until the user opts to publish.)

- [ ] **Step 7.5: Final commit**

```powershell
git add README.md docs/superpowers/m2-readiness.md
git commit -m "docs(m2): mark M2 done; readiness checklist closes Plan 2"
```

---

## Self-review notes

- Spec coverage: §2.1 → Task 1; §2.2 → Task 2; §2.3 → Task 4; §2.4 → Task 3; §2.5 → Task 6; §3 (config) → Tasks 2 + 6; §4 (test contract) — every TDD-applicable surface gets a unit test (Tasks 2, 3, 4) and the slow integration test gets Task 5; §5 acceptance criteria checked in Tasks 5 + 6.
- Schema reuse: Tasks 3 + 4 use the existing `Citation` and `Response` from `src/common/schemas.py` rather than redefining; Task 4 extends `Response` with an optional `mode` field (smallest change that lets the orchestrator and the M4 eval runner both label outputs).
- Token cap regression: Task 2 step 4 uses `tokenizer.encode(..., add_special_tokens=False)`; the unit-test stub mirrors that signature so the production tokenizer (transformers AutoTokenizer) drops in unchanged.
- Lock contention: Task 4's last unit test `test_generate_is_serialized_under_a_lock` pins the threading guarantee that the spec calls out; if a future refactor removes the lock, this test fails before the UI does.
- The MVP design's reference to `tests/test_smoke.py` is satisfied by `tests/test_serving_integration.py::test_orchestrator_real_models_round_trip` (different filename, same coverage; the original name was tentative).
