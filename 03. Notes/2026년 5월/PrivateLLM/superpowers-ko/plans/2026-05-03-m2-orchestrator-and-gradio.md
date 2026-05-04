# M2 — Orchestrator + Gradio (base / RAG 모드) 구현 계획

> **에이전트 워커용:** 필수 서브 스킬: superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans를 사용하여 본 계획을 task 단위로 구현하라. 각 단계는 추적을 위해 체크박스(`- [ ]`) 구문을 사용한다.

**목표:** 기존 M1 retriever를 M0의 4-bit Qwen2.5-3B base 모델과 `Orchestrator.generate(query, mode)` 뒤편에서 연결하고, `base` 모드와 `rag` 모드 답변을 좌우로 비교하는 최소한의 Gradio UI로 결과를 노출한다.

**아키텍처:** 프로세스 싱글톤 `model_loader`가 4-bit base 모델과 tokenizer를 소유한다. 순수 로직 `prompt_builder`가 `(query, mode, chunks)`를 토큰 상한이 적용된 chat 메시지로 변환한다. `Orchestrator`가 retriever + loader + prompt + generation을 thread-locked 진입점 뒤편에서 묶는다. `eval/citation_checker.py`(정규식 추출 + 스텁 검증)는 MVP 모듈 그래프에 따라 `eval/` 아래에 위치하지만 M2 UI에서 소비된다. Gradio 앱은 submit마다 orchestrator의 내부 lock 아래에서 모드별로 한 번씩 총 두 번 orchestrator를 호출한다.

**기술 스택:** Python 3.12, transformers 4.49 + bitsandbytes (4-bit), pydantic, ChromaDB + bge-m3 + rank_bm25 (M1에서 이미 연결됨), Gradio.

**코딩 전 참고할 문서:**
- `docs/superpowers/specs/2026-05-03-m2-orchestrator-design.md` — 본 계획의 design spec
- `docs/superpowers/specs/2026-05-02-legal-privatellm-mvp-design.md` §4.4, §5, §10 — 동결된 MVP spec
- `docs/superpowers/m2-readiness.md` — 사전 점검 항목 A/B/C/D (모두 2026-05-03에 완료)
- `scripts/verify_unsloth.py`, `scripts/verify_combined_vram.py` — 4-bit 로딩에 이미 사용 중인 Unsloth/transformers 폴백; Task 1에서 이를 공유 헬퍼로 추출한다

**`src/common/schemas.py`에 이미 정의된 스키마 — 재사용하라, 재정의 금지:**
- `Chunk`, `Citation` (raw / normalized / kind), `Response` (answer / citations / retrieved / latency_ms: float).

---

## 파일 맵

**생성:**
- `src/serving/__init__.py`
- `src/serving/model_loader.py` — 싱글톤 4-bit base 모델 + tokenizer
- `src/serving/prompt_builder.py` — 순수 로직; chat 메시지 빌더
- `src/serving/orchestrator.py` — `Orchestrator.generate(query, mode)`
- `src/serving/app_gradio.py` — Gradio UI
- `src/eval/__init__.py`
- `src/eval/citation_checker.py` — 정규식 추출 + 스텁 검증
- `config/prompts/rag.txt`, `config/prompts/no_rag.txt` — 시스템 프롬프트
- `tests/test_prompt_builder.py`
- `tests/test_citation_checker.py`
- `tests/test_orchestrator.py`
- `tests/test_serving_integration.py` — slow, gated

**수정:**
- `config/default.yaml` — `serving:` 섹션 추가; 죽은 최상위 `paths:` 와 `chunk.overlap` 키 제거
- `Makefile` — `serve` 타겟 연결 (이미 일치함; 확인만 수행)
- `README.md` — M2 행을 ✅로 전환; "Try the Gradio UI" 섹션 추가

---

## Task 1: 공유 4-bit loader 추출 → `model_loader.get_base_model()`

**파일:**
- 생성: `src/serving/__init__.py` (빈 마커)
- 생성: `src/serving/model_loader.py`
- 수정: `scripts/verify_unsloth.py:24-44` — 인라인 폴백을 `src.serving.model_loader`에서의 import로 교체
- 수정: `scripts/verify_combined_vram.py:65-92` — 동일 처리
- 테스트(slow): `tests/test_serving_integration.py::test_get_base_model_is_singleton`

- [ ] **Step 1.1: 실패하는 slow integration 테스트 작성**

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

- [ ] **Step 1.2: 실패를 확인하기 위해 실행 (모듈이 아직 import 불가)**

실행: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
기대값: ImportError / module not found.

- [ ] **Step 1.3: `src/serving/__init__.py` 작성**

```python
"""Serving subsystem: orchestrator, model loader, prompt builder, Gradio UI."""
```

- [ ] **Step 1.4: `src/serving/model_loader.py` 작성**

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

- [ ] **Step 1.5: slow 테스트 실행 (실제로 모델을 로드함 — 약 10-15초)**

실행: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
기대값: PASS.

- [ ] **Step 1.6: `scripts/verify_unsloth.py`를 공유 loader 사용으로 리팩터링**

인라인 try/except (대략 22-44행)를 다음으로 교체:

```python
from src.serving.model_loader import get_base_model, loader_name

# ... at top of main(), keep the timing print ...
t0 = time.time()
model, tokenizer = get_base_model()
log.info("Loaded via %s in %.1fs", loader_name(), time.time() - t0)
```

(`main()`의 나머지 부분 — VRAM 로깅, prompt, generate 등 — 은 변경하지 않는다.)

- [ ] **Step 1.7: `scripts/verify_combined_vram.py`를 공유 loader 사용으로 리팩터링**

로컬 `_load_qwen_4bit(cfg)` 함수와 그 호출 지점을 다음으로 교체:

```python
from src.serving.model_loader import get_base_model, loader_name

# at the call site:
t0 = time.time()
model, tokenizer = get_base_model()
loader = loader_name()
log.info("Loaded Qwen2.5-3B 4-bit via %s in %.1fs", loader, time.time() - t0)
```

로컬 `_load_qwen_4bit` 함수는 통째로 삭제한다.

- [ ] **Step 1.8: 두 리팩터링된 스크립트를 sanity-run하여 동등성을 확인**

실행:
```powershell
.venv/Scripts/python scripts/verify_unsloth.py
.venv/Scripts/python scripts/verify_combined_vram.py
```
기대값: 모두 여전히 PASS, 피크 VRAM 여전히 ≤ 5.5 GB.

- [ ] **Step 1.9: 회귀가 없음을 보장하기 위해 fast suite 실행**

실행: `.venv/Scripts/python -m pytest -m "not slow" -q`
기대값: 32 passed (Plan-2 이전 베이스라인과 동일).

- [ ] **Step 1.10: 커밋**

```powershell
git add src/serving/__init__.py src/serving/model_loader.py scripts/verify_unsloth.py scripts/verify_combined_vram.py tests/test_serving_integration.py
git commit -m "feat(serving): singleton 4-bit base model loader"
```

---

## Task 2: `prompt_builder.build_messages()` (TDD, 순수 로직)

**파일:**
- 생성: `config/prompts/rag.txt`, `config/prompts/no_rag.txt`
- 생성: `src/serving/prompt_builder.py`
- 테스트: `tests/test_prompt_builder.py`

- [ ] **Step 2.1: 프롬프트 파일 작성**

`config/prompts/rag.txt`:

```text
당신은 한국 민법 전문가입니다. 아래 참고자료를 근거로 답변하고, 근거가 부족하면 "참고자료에 명시되지 않음"이라고 답하세요. 인용은 [조문번호] 또는 [판례번호] 형식으로 본문에 표시하세요. 본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

`config/prompts/no_rag.txt`:

```text
당신은 한국 민법 전문가입니다. 귀하의 일반 지식으로 답변하세요. 추측이 필요한 부분은 명시적으로 표시하세요. 본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

- [ ] **Step 2.2: 실패하는 테스트 작성**

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

- [ ] **Step 2.3: 실패를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_prompt_builder.py -v`
기대값: ImportError 또는 테스트 실패 (모듈 미구현).

- [ ] **Step 2.4: `src/serving/prompt_builder.py` 구현**

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

- [ ] **Step 2.5: 통과를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_prompt_builder.py -v`
기대값: 6 passed.

- [ ] **Step 2.6: 전체 fast suite 실행**

실행: `.venv/Scripts/python -m pytest -m "not slow" -q`
기대값: 38 passed (이전 32 + 신규 6).

- [ ] **Step 2.7: 커밋**

```powershell
git add src/serving/prompt_builder.py tests/test_prompt_builder.py config/prompts/
git commit -m "feat(serving): prompt_builder for base/rag modes with token cap"
```

---

## Task 3: `eval/citation_checker.py` (TDD, 순수 정규식)

**파일:**
- 생성: `src/eval/__init__.py`
- 생성: `src/eval/citation_checker.py`
- 테스트: `tests/test_citation_checker.py`

- [ ] **Step 3.1: 실패하는 테스트 작성**

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

- [ ] **Step 3.2: 실패를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_citation_checker.py -v`
기대값: ImportError.

- [ ] **Step 3.3: `src/eval/__init__.py` 구현**

```python
"""Evaluation subsystem. M2 ships only citation_checker; runner/judge land in M4."""
```

- [ ] **Step 3.4: `src/eval/citation_checker.py` 구현**

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

- [ ] **Step 3.5: 통과를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_citation_checker.py -v`
기대값: 9 passed.

- [ ] **Step 3.6: 전체 fast suite 실행**

실행: `.venv/Scripts/python -m pytest -m "not slow" -q`
기대값: 47 passed (이전 38 + 신규 9).

- [ ] **Step 3.7: 커밋**

```powershell
git add src/eval/__init__.py src/eval/citation_checker.py tests/test_citation_checker.py
git commit -m "feat(eval): citation_checker — statute/case regex + stub verify"
```

---

## Task 4: 모델을 mock한 단위 테스트와 함께 `Orchestrator.generate(query, mode)` 구현

**파일:**
- 생성: `src/serving/orchestrator.py`
- 테스트: `tests/test_orchestrator.py`

- [ ] **Step 4.1: 의존성을 mock한 실패하는 단위 테스트 작성**

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

- [ ] **Step 4.2: 실패를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_orchestrator.py -v`
기대값: ImportError.

- [ ] **Step 4.3: `src/serving/orchestrator.py` 구현**

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

참고: `src/common/schemas.py`상의 `Response`는 `mode` 필드를 **포함하지 않는다** — 그러나 단위 테스트 `test_base_mode_skips_retriever`는 `resp.mode == "base"`를 단언한다. 스키마를 안정적으로 유지하려면, orchestrator의 로컬 반환 계약에 `mode`를 추가하는 방법으로 Response를 그 자리에서 확장한다: 테스트의 단언 경로를 wrapper에서 `mode`를 읽도록 변경하거나, (더 단순하게) schemas.py의 `Response`를 optional `mode` 필드로 확장한다. (M4) eval runner도 이를 원할 것이므로 스키마 확장을 택한다.

- [ ] **Step 4.4: `src/common/schemas.py`를 확장하여 `Response`에 `mode` 추가**

`src/common/schemas.py:63-68`을 편집:

```python
class Response(BaseModel):
    answer: str
    mode: Optional[Literal["base", "rag", "qlora", "rag_qlora"]] = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[Chunk] = Field(default_factory=list)
    latency_ms: float
```

`src/serving/orchestrator.py`를 업데이트하여 반환되는 `Response`에 `mode=mode`를 설정:

```python
return Response(
    answer=answer,
    mode=mode,
    citations=citations,
    retrieved=chunks,
    latency_ms=latency_ms,
)
```

(`.model_copy(update={})` 호출은 제거한다.)

- [ ] **Step 4.5: 통과를 확인하기 위해 테스트 실행**

실행: `.venv/Scripts/python -m pytest tests/test_orchestrator.py -v`
기대값: 4 passed.

- [ ] **Step 4.6: 전체 fast suite 실행**

실행: `.venv/Scripts/python -m pytest -m "not slow" -q`
기대값: 51 passed (이전 47 + 신규 4).

- [ ] **Step 4.7: 커밋**

```powershell
git add src/serving/orchestrator.py src/common/schemas.py tests/test_orchestrator.py
git commit -m "feat(serving): Orchestrator.generate(base|rag) with thread-locked gen"
```

---

## Task 5: 실제 모델 기반 slow orchestrator integration 테스트

**파일:**
- 수정: `tests/test_serving_integration.py`

- [ ] **Step 5.1: integration 테스트 추가**

`tests/test_serving_integration.py`에 다음을 추가:

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

- [ ] **Step 5.2: slow suite 실행**

실행: `.venv/Scripts/python -m pytest tests/test_serving_integration.py -m slow -v`
기대값: 2 passed (Task 1의 싱글톤 테스트 + 본 테스트). 총 실행 시간 약 30-60초.

- [ ] **Step 5.3: 회귀 게이트로서 결합 VRAM 검증을 한 번 더 수행**

실행: `.venv/Scripts/python scripts/verify_combined_vram.py`
기대값: PASS, 피크 < 5.5 GB (여전히 약 3.2 GB 이어야 함).

- [ ] **Step 5.4: 커밋**

```powershell
git add tests/test_serving_integration.py
git commit -m "test(serving): real-model E2E for orchestrator base+rag"
```

---

## Task 6: Gradio UI (`app_gradio.py`)

**파일:**
- 생성: `src/serving/app_gradio.py`
- 수정: `config/default.yaml` (`serving:` 블록 추가; 죽은 `paths:` 와 `chunk.overlap` 제거)

- [ ] **Step 6.1: `config/default.yaml` 업데이트**

파일을 다음으로 교체:

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

(죽은 최상위 `paths:` 블록과 `chunk.overlap` 키는 제거되었다; `m2-readiness.md`에서 마이너 정리 항목으로 표시되어 있었다.)

- [ ] **Step 6.2: `src/serving/app_gradio.py` 작성**

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

- [ ] **Step 6.3: 수동 smoke 테스트**

실행: `.venv/Scripts/python -m src.serving.app_gradio`

기대값:
- 콘솔에 "Cold-loading orchestrator..."가 출력됨
- 브라우저가 `http://127.0.0.1:7860`에서 열림
- "임대차의 정의를 설명해 주세요."를 입력하고 Run을 누르면 약 10초 이내에 비어있지 않은 한국어 답변 두 개가 산출됨
- RAG 컬럼에 `[618]`로 태깅된 retrieved chunk가 최소 하나 이상 표시됨
- 인용 영역에 rag 컬럼용 `[statute] 618`이 나열됨

Ctrl-C로 서버를 중지한다.

- [ ] **Step 6.4: `make serve`가 여전히 올바르게 연결되는지 확인**

기존 Makefile에는 `serve: python -m src.serving.app_gradio`가 있다 — `make serve`를 실행하여 (해석만 확인하고 즉시 Ctrl-C) 검증한다.

- [ ] **Step 6.5: 전체 테스트 스위트(fast + slow) 실행**

실행: `.venv/Scripts/python -m pytest -q`
기대값: 모든 테스트 통과. 합계: fast 51개 + slow 약 4개.

- [ ] **Step 6.6: 커밋**

```powershell
git add src/serving/app_gradio.py config/default.yaml
git commit -m "feat(serving): Gradio UI — side-by-side base vs RAG with citation badges"
```

---

## Task 7: README + 상태 업데이트

**파일:**
- 수정: `README.md` — M2 행을 ✅로 전환; "Try the Gradio UI" 섹션 추가
- 수정: `docs/superpowers/m2-readiness.md` — "Plan 2 closed" 섹션 추가

- [ ] **Step 7.1: `README.md` 상태 표 업데이트**

M2 행을 "next plan"에서 "✅ done — tag `m2-orchestrator`"로 변경:

```markdown
| **M2 — Orchestrator + Gradio (base / RAG modes)** | ✅ done — tag `m2-orchestrator` |
```

- [ ] **Step 7.2: "Try the retriever (after M1)" 다음에 새 섹션을 `README.md`에 추가**

```markdown
## Try the Gradio UI (after M2)

```powershell
make serve                            # opens http://127.0.0.1:7860
```

The UI runs the same query through `base` and `rag` modes side-by-side, with extracted citations and retrieved chunk previews. First load is ~10-15 s while the singleton model + retriever spin up.
```

- [ ] **Step 7.3: `docs/superpowers/m2-readiness.md`에 "Plan 2 closed" 섹션 추가**

추가:

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

- [ ] **Step 7.4: 마일스톤에 태그 부여**

```powershell
git tag m2-orchestrator -m "M2: orchestrator + Gradio (base/RAG)"
```

(푸시하지 말 것 — 사용자가 공개를 선택할 때까지 태그는 로컬에 유지된다.)

- [ ] **Step 7.5: 최종 커밋**

```powershell
git add README.md docs/superpowers/m2-readiness.md
git commit -m "docs(m2): mark M2 done; readiness checklist closes Plan 2"
```

---

## 자체 리뷰 메모

- Spec 커버리지: §2.1 → Task 1; §2.2 → Task 2; §2.3 → Task 4; §2.4 → Task 3; §2.5 → Task 6; §3 (config) → Task 2 + 6; §4 (test contract) — TDD가 적용 가능한 모든 표면이 단위 테스트(Task 2, 3, 4)를 가지며 slow integration 테스트는 Task 5에서 다룬다; §5 acceptance criteria는 Task 5 + 6에서 점검된다.
- 스키마 재사용: Task 3 + 4는 기존 `Citation`과 `Response`를 `src/common/schemas.py`에서 재사용하며 재정의하지 않는다; Task 4는 `Response`를 optional `mode` 필드로 확장한다 (orchestrator와 M4 eval runner가 모두 출력에 라벨을 붙이게 하는 가장 작은 변경).
- 토큰 상한 회귀: Task 2 step 4는 `tokenizer.encode(..., add_special_tokens=False)`를 사용한다; 단위 테스트 스텁이 같은 시그니처를 따르므로 production tokenizer (transformers AutoTokenizer)가 변경 없이 그대로 들어맞는다.
- Lock 경합: Task 4의 마지막 단위 테스트 `test_generate_is_serialized_under_a_lock`는 spec이 명시한 threading 보장을 고정한다; 향후 리팩터링이 lock을 제거하면 UI보다 먼저 본 테스트가 실패한다.
- MVP 디자인의 `tests/test_smoke.py` 언급은 `tests/test_serving_integration.py::test_orchestrator_real_models_round_trip`가 충족한다 (파일명만 다르고 커버리지는 동일; 원래 이름은 잠정적이었음).
