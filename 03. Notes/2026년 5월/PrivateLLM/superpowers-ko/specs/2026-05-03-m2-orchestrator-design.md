# M2 — Orchestrator + Gradio (base / RAG 모드) — 설계 명세

- **Date:** 2026-05-03
- **Author:** `2026-05-02-legal-privatellm-mvp-design.md` (동결됨)의 연속 — 본 문서는 Plan 2가 출하하는 Serving 슬라이스에만 집중한다.
- **Status:** Plan 작성 준비 완료
- **Scope:** MVP 설계 문서의 M2 마일스톤(§4.4 Serving + §10 Milestones — M2)을 구현한다. 검색(retrieval), 인제스션(ingestion), 평가(evaluation) 영역은 변경하지 않는다.

## 0. 본 문서가 존재하는 이유

MVP 설계 문서(2026-05-02)는 5개 마일스톤 PoC 전체를 아키텍처 단위로 다룬다. Plan 1은 M0+M1을 출하했다(RAG 검색이 동작 중이며, `m2-readiness.md` 참고). 본 문서는 M2에 한정된 **plan-input 정련안**이다 — 파일별 책임, 공개 API, M3로 의도적으로 미룬 항목, 그리고 테스트 계약이다. 여기서 명시되지 않은 사항은 모두 MVP 설계를 따른다.

## 1. 목표

기존 retriever(M1)와 Qwen2.5-3B 4-bit 베이스 모델(M0)을 단일한 `Orchestrator.generate(query, mode)` 진입점 뒤에 연결하고, 사용자가 동일한 질의에 대해 `base`와 `rag` 응답을 나란히 비교할 수 있는 최소 Gradio UI를 통해 노출한다. 이는 M3에서 QLoRA 작업이 시작되기 전에 사람이 실제 민법 질문에 대한 RAG 품질을 직접 눈으로 확인하기 시작할 수 있는 슬라이스다.

**M2에서 제외(M3 / M4로 이연):**
- `qlora` 및 `rag_qlora` 모드 — 아직 어댑터가 존재하지 않음.
- 평가 러너 / 저지(judge) / 점수 집계 — `eval_set.jsonl`도 비어 있음. 인용(citation) 추출은 UI가 사용하므로 M2에 속하나, 이를 소비하는 평가 파이프라인은 그렇지 않다.
- `model_loader.py`의 어댑터 attach/detach 메커니즘 — 설계는 반영하되 M2에서는 구현하지 않음.

## 2. 컴포넌트

### 2.1 `src/serving/model_loader.py`

**책임.** "이 호스트에서 base 4-bit Qwen2.5-3B-Instruct를 로드한다"의 단일 진실 공급원. 프로세스 싱글톤 — 한 번 로드된 후의 후속 호출은 동일한 `(model, tokenizer)` 쌍을 반환한다.

**Public API.**

```python
def get_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]: ...
```

**구현 노트.** `scripts/verify_unsloth.py` 및 `scripts/verify_combined_vram.py`에서 이미 검증된 Unsloth-우선 / transformers+bitsandbytes-폴백 절차를 재사용하라. 그 폴백 로직을 추출해 세 호출 지점이 하나의 경로를 공유하도록 한다. 로더는 `default.yaml`에서 `cfg["model"]["base_id"]`와 `cfg["model"]["max_seq_len"]`을 읽는다.

**어댑터 설계 훅 (M3-ready, 지금은 미구현).** 함수는 추후 `attach_adapter(name)` / `detach_adapter()` 확장이 싱글톤 재작성 없이 안착할 수 있도록 구조화되어야 한다. 싱글톤 상태는 묵시적(람다 클로저)이 아니라 명시적(dict / 모듈 수준 dataclass)으로 유지하라.

### 2.2 `src/serving/prompt_builder.py`

**책임.** 순수 로직 — 모델도 GPU도 필요 없음. `(query, mode, retrieved_chunks)`를 chat-template-ready 메시지 리스트로 변환. TDD 친화적.

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

**동작.**
- `base` 모드: `chunks`를 무시하고, system instruction을 앞에 붙인 단일 user 턴을 생성한다(또는 Qwen의 chat template에 따라 `system` 역할로).
- `rag` 모드: 주어진 순서대로 `chunks`로부터 `[참고자료]` 블록을 구성하고, 각 chunk를 `[조문번호 또는 판례번호] {text}`로 포맷한다(statute_no 우선, 없으면 case_no, 그것도 없으면 chunk.id).
- **토큰 캡 강제.** 조립된 컨텍스트를 토큰화하고(로드된 tokenizer 사용 — 인자로 전달하거나 `get_base_model()`을 통해 공유), 컨텍스트가 `max_context_tokens`를 초과하면 끝부분(가장 낮은 순위)부터 chunk를 제거하여 맞춘다. 절단이 발생하면 경고를 로깅하라. 사용자의 query는 절대 조용히 버리지 말라.
- 정확한 system 텍스트는 `config/prompts/rag.txt`와 `config/prompts/no_rag.txt`에서 가져온다. Plan 2가 이 두 파일을 생성한다(`prompts/` 디렉터리는 현재 비어 있다).

**시스템 프롬프트 (한국어, 민법).** MVP 설계의 §5 "RAG Prompt Template"을 따른다. 명확성을 위해 다시 적는다:

```
당신은 한국 민법 전문가입니다. 아래 참고자료를 근거로 답변하고,
근거가 부족하면 "참고자료에 명시되지 않음"이라고 답하세요.
인용은 [조문번호] 또는 [판례번호] 형식으로 본문에 표시하세요.
본 답변은 정보 제공 목적이며 법률 자문이 아닙니다.
```

비-RAG 변형은 `참고자료를 근거로`를 `귀하의 일반 지식으로`로 치환하고 `[참고자료]` 블록을 제거한다.

### 2.3 `src/serving/orchestrator.py`

**책임.** 접착제. `Retriever` + `model_loader` + `prompt_builder`를 MVP 명세의 단일 진입점 뒤에 감싸고, 응답에 대한 인용 추출까지 담당한다.

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

**동작.**
- `base`: 검색 생략, 메시지 빌드, generate, 응답에서 인용 추출(순수히 정보용 — base 모드는 유효한 인용을 거의 내지 않음), 반환.
- `rag`: 검색 수행(`cfg["retrieval"]["top_k"]`의 top-K, 기본 5), 메시지 빌드, generate, 인용 추출, 반환.
- 생성 하이퍼파라미터: `max_new_tokens=512`, `do_sample=False`, `temperature` / `top_p` / `top_k`는 비움(`verify_combined_vram`에서 발생하는 시끄러운 경고 회피).
- `Orchestrator.open()`이 싱글톤 인지 생성자다. `Retriever.open()`과 `get_base_model()`을 재사용한다.

**싱글톤 & 스레딩.** Gradio는 워커 풀에서 핸들러를 실행한다. model + tokenizer + retriever는 init 후 read-only이지만, `model.generate(...)`는 동일 모델 인스턴스에서 동시 호출이 안전하지 않다. Plan 2는 orchestrator 내부의 `threading.Lock` 뒤에서 생성을 직렬화한다. 경합(contention)을 문서화하라; 1-사용자 PoC에서는 허용 가능하다.

### 2.4 `src/eval/citation_checker.py`

**책임.** 모델 응답에서 인용을 정규식으로 추출하고, (선택적으로) corpus 인덱스에 대해 검증한다. Plan 2는 추출만 필요하다. corpus 검증은 형식이 올바른 인용에 대해 `True`를 반환하는 stub로 두고, 실제 조회는 M4 plan에서 채워 넣는다.

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

**정규식 패턴.** MVP 설계 §4.5의 두 가지:
- 조문: `\[?민법\s*제\s*(\d+)\s*조(?:의\s*\d+)?\]?` 와 `민법` 접두어가 없는 단순한 `제\s*\d+\s*조` 형태 — 둘 다 캡처되어야 한다. 후자는 이미 법률이 확정된 응답에서 흔하다.
- 판례: `대법원\s*\d{4}[가-힣]+\d+` (`2019다12345` 스타일)와 대괄호 변형들.

**왜 `src/serving/`이 아닌 `src/eval/`에 두는가:** MVP 설계 모듈 그래프(§4.7)는 citation checker를 eval 아래에 둔다. Gradio UI는 `from src.eval.citation_checker import extract_citations`를 통해 임포트한다 — 이는 횡단(cross-cutting) 관심사지만 명세의 의존성 방향과 일치한다(`serving → eval`은 역방향 엣지다; M4의 러너가 corpus index를 소유하고, M2는 추출기만 소비한다).

### 2.5 `src/serving/app_gradio.py`

**책임.** 단일 파일 Gradio 앱. 2-컬럼 레이아웃: 동일한 query를 `base`와 `rag` 모드에 대해 렌더링. 각 컬럼은 다음을 표시한다: 응답, 추출된 인용(각 `found_in_corpus`에 대한 녹색/빨간색 배지), 그리고 (RAG 컬럼 한정) statute/case id와 함께 표시된 검색된 chunk들.

**입력.** textarea (query), "Run" 버튼, 선택적 "k" 슬라이더(기본 5).

**Wiring.** 클릭 시 → `Orchestrator.generate(q, "base")`와 `Orchestrator.generate(q, "rag")`(lock 아래 순차적; Gradio의 큐가 동시 사용자를 직렬화하여 처리). latency_ms와 chunk 미리보기를 표시한다.

**Make 타깃.** `make serve`는 이미 `python -m src.serving.app_gradio`에 매핑되어 있다 — `share=False`로 `http://127.0.0.1:7860`에서 앱을 띄우도록 연결하라.

## 3. 설정

`config/default.yaml`은 이미 `model:`과 `retrieval:`을 갖고 있다; M2가 추가하는 것:

```yaml
serving:
  max_new_tokens: 512
  max_context_tokens: 1500
  do_sample: false
```

Plan 2는 또한 위의 system 프롬프트들을 담은 `rag.txt`와 `no_rag.txt`를 포함하는 빈 `config/prompts/` 디렉터리 앵커를 생성한다.

## 4. 테스트 계약 (TDD-적용 가능 영역은 굵게)

| File | Test | Type |
|---|---|---|
| **`prompt_builder.py`** | base 대 rag 템플릿 분기; chunk-id 포맷팅; 토큰 캡이 가장 낮은 순위를 드롭 | unit (TDD) |
| **`citation_checker.py`** | 조문 정규식(민법 접두어 유/무); 판례 정규식; 다중 인용 추출; 일반 산문에 대한 false positive 없음 | unit (TDD) |
| `model_loader.py` | 싱글톤: 두 호출이 같은 id를 반환; 폴백 경로용 subprocess marker 테스트 | integration (slow, gated) |
| `orchestrator.py` (mocked) | 모드 디스패치: base → retriever 호출 없음; rag → 쿼리로 retriever 호출됨; 인용 리스트가 흘러감 | integration (`Retriever` + `model.generate` 모킹) |
| `app_gradio.py` | unit-test 하지 않음; 아래 M2 수용 기준에 따른 수동 스모크 |

Slow integration: `Orchestrator.open()`을 통해 실제 모델 + retriever를 로드하고, 임대차 query에 대해 두 모드를 모두 실행하며, rag 응답의 `retrieved` 리스트가 비어 있지 않고 `latency_ms`가 기록되었음을 검증하는 테스트 하나. 기존 E2E와 마찬가지로 `-m slow` 뒤에 게이팅된다.

## 5. 수용 기준 (= Plan 2 "완료")

1. `make serve`가 Gradio UI를 열고; "임대차의 정의를 설명해 주세요"를 입력하면 대상 호스트에서 약 10초 이내에 양 컬럼 모두에서 응답이 반환된다.
2. RAG 컬럼은 `statute_no == "618"`인 검색 chunk를 적어도 하나 표시한다.
3. citation-checker 뷰는 모델의 `[제618조]`-형태 인용이 추출된 것으로 표시한다(M2에서는 corpus 검증 stub-true로도 충분).
4. 양 컬럼 모두 비어 있지 않은 한국어 텍스트를 표시한다.
5. `pytest -m "not slow"`가 계속 green을 유지한다; 새로운 prompt-builder + citation-checker 단위 테스트가 그 일부다. Slow suite는 orchestrator 통합 테스트 하나를 추가로 얻는다.
6. `scripts/verify_combined_vram.py`가 계속 PASS(피크 < 5.5 GB)한다 — 전체 RAG 턴에 대해 orchestrator를 실행해도 헤드룸이 회귀해서는 안 된다.

## 6. 리스크 & 완화

| Risk | Mitigation |
|---|---|
| Gradio 워커 스레드가 `model.generate`에 동시에 재진입 | `Orchestrator.generate` 내 `threading.Lock`; 경합 문서화. |
| `prompt_builder` 토큰 캡이 사용자의 유일하게 관련 있는 chunk를 떨어뜨림 | 토큰 캡은 끝부분(가장 낮은 순위)에서만 드롭; INFO 수준에서 경고. k=5에서의 1500-토큰 캡은 민법 chunk에 대해 경험적으로 안전하다(`chunk.py` 상한 기준 평균 ~400 tokens/chunk). |
| 인용 정규식이 "제3장"과 같은 본문 산문에 과매칭 | 패턴을 `\d+조` 코어에 앵커링; false-positive 산문에 대한 명시적 단위 테스트. |
| 누가 CI 중에 `make serve`를 돌리면 테스트 러너가 블로킹됨 | 문서화; CI는 `make serve`를 실행하지 않는다. |
| 콜드 모델 로드 시 지연 목표가 어긋남 | Orchestrator의 첫 호출이 `Orchestrator.open()`을 통해 사전에 워밍업; UI는 싱글톤 초기화가 끝난 후에만 사용자에게 인사한다. |

## 7. M2 외 (재진술, 의도적)

- QLoRA 어댑터 attach/detach (M3).
- 인용 검증을 위한 실제 `corpus.find()` (M4).
- `eval/runner.py`, `eval/judge.py`, `eval/aggregate.py` (M4).
- 멀티턴 대화 / 세션 메모리 (MVP 전체에서 제외).
- UI에서의 토큰 단위 스트리밍 (이연; PoC에서는 full-completion으로 충분).

## 8. 권장 첫 세 가지 작업 (Plan 2)

이는 이미 `m2-readiness.md` §"Suggested Plan 2 first-three tasks"에 초안이 있었으나 bootstrap-polish 항목은 이제 완료되었다. 갱신:

1. **`src/serving/__init__.py` + `model_loader.py`** — Unsloth/transformers 폴백을 공유 헬퍼로 추출; `get_base_model()` 노출. 싱글톤을 검증하는 slow integration 테스트 하나.
2. **`src/serving/prompt_builder.py` + 테스트 (TDD)** — 가장 작은 순수 로직 조각. `config/prompts/`의 prompt 파일들을 견인한다.
3. **`src/eval/__init__.py` + `citation_checker.py` + 테스트 (TDD)** — 순수 정규식; orchestrator와 UI에서 필요.

이 세 가지 이후에는, orchestrator와 Gradio UI는 기계적인 wiring이다.
