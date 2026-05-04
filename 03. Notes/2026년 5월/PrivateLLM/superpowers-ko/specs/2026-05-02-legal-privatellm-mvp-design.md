# Legal PrivateLLM MVP — 설계 명세

- **일자:** 2026-05-02
- **작성자:** Claude와 함께 브레인스토밍 (superpowers:brainstorming)
- **상태:** 검토용 초안
- **유형:** 기술 검증 PoC (한국 민법 도메인의 RAG + QLoRA)

---

## 1. 목표

**RAG와 QLoRA 파인튜닝이 한국 민법 Q&A에서 소형 오픈웨이트 LLM의 성능을 향상시키는지** 정량적으로 검증하는 Proof-of-Concept 구축. 산출물은 직접 큐레이션한 평가셋에 대한 4-way ablation 리포트(base / RAG / QLoRA / RAG+QLoRA)와 인터랙티브 비교를 위한 최소한의 Gradio UI이다.

본 작업은 **기술 PoC**이며 제품이 아니다. UI는 최소화하며, 핵심 산출물은 평가 리포트이다.

## 2. 결정 사항 요약

| 영역 | 결정 |
|---|---|
| PoC 유형 | 기술 검증 — ablation 효과 측정 |
| 법률 도메인 | 한국 민법; 평가는 계약법 + 임대차 중심 |
| 하드웨어 | 로컬: RTX 3060 Laptop **6GB VRAM**, 32GB RAM, 약 437GB 여유 디스크, Windows 11 |
| 전략 | Unsloth + 3B급 QLoRA, 전 과정 로컬 학습 |
| Base 모델 | Qwen2.5-3B-Instruct (Apache 2.0) |
| 데이터 소싱 | 공개 한국 법률 데이터 + Claude API를 통한 합성 Q&A |
| RAG embedding | bge-m3 (BAAI, MIT) |
| 벡터 스토어 | ChromaDB (embedded) |
| Retrieval | Hybrid: dense + BM25, RRF로 융합, top-K=5 |
| 평가 지표 | LLM-as-Judge (Claude) + Citation Accuracy |
| 평가셋 규모 | 직접 검수한 100~150개 (80개에서 시작하여 확장) |
| UI | Gradio, 모드 토글 제공 (RAG on/off, QLoRA on/off) |

## 3. 아키텍처 개요

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

네 개의 핵심 서브시스템은 오직 파일(JSONL/Parquet)과 ChromaDB를 통해서만 통신한다. 각 서브시스템은 독립적으로 실행 및 테스트 가능하다.

## 4. 컴포넌트

### 4.1 Ingestion (`src/ingestion/`)

- **책임:** 외부 소스 → 정규화된 텍스트 → chunks + 메타데이터.
- **입력:** 정부 법령 OpenAPI, 대법원 사이트, 공개 한국 LegalQA 데이터셋.
- **출력:** `data/processed/chunks.jsonl`, `data/processed/qa_seed.jsonl`.
- **파일:** `fetch_statutes.py`, `fetch_cases.py`, `chunk.py`, `synth_qa.py`.
- **Chunk 스키마:** `{id, source, doc_type[조문|판례], statute_no, case_no, title, text, char_range, hash}`.
- **재현성:** 모든 변환은 결정론적이며, 원본 API 응답은 `data/cache/`에 캐싱된다. 파이프라인 버전은 JSONL 헤더 메타데이터에 기록된다.

### 4.2 Embedding & Index (`src/rag/`)

- **책임:** Chunks → dense vectors + BM25 인덱스 → ChromaDB.
- **입력:** `data/processed/chunks.jsonl`.
- **출력:** `data/chroma/`, `data/bm25.pkl`.
- **파일:** `embed.py` (FlagEmbedding을 통한 bge-m3), `index.py`, `bm25.py`, `retriever.py`.
- **공개 API:** `Retriever.search(query: str, k: int = 5) -> list[Chunk]`.
- **Hybrid retrieval:** dense top-K와 BM25 top-K를 Reciprocal Rank Fusion (RRF)으로 융합, 최종 K=5.
- **VRAM 노트:** bge-m3 인덱싱은 일회성 GPU 작업이며, 쿼리 시점의 embedding은 가볍다. 6GB VRAM에서의 OOM을 피하기 위해 base LLM과 bge-m3는 동시에 로드하지 않는다.

### 4.3 Training (`src/training/`)

- **책임:** QLoRA adapter 학습.
- **입력:** `data/processed/qa_train.jsonl` (instruction-completion 쌍).
- **출력:** `models/adapters/qwen2.5-3b-civil-v{N}/`.
- **파일:** `prepare_dataset.py`, `train_qlora.py`.
- **기본 하이퍼파라미터:** Unsloth `FastLanguageModel`, 4-bit, LoRA r=16/α=32, gradient checkpointing, max_seq=2048 (OOM 시 1024로 자동 축소), batch=1 + grad_accum=8, lr=2e-4, epochs=2~3, warmup=10%.
- **로깅:** `runs/` 하위 TensorBoard, 추가로 JSONL 학습 로그.
- **Resume:** 200 step마다 체크포인트.

### 4.4 Serving / Orchestrator (`src/serving/`)

- **책임:** 4-mode 추론 (base / +adapter × ±RAG).
- **입력:** `query: str`, `mode ∈ {base, qlora, rag, rag_qlora}`.
- **출력:** `{answer: str, citations: list, retrieved: list[Chunk], latency_ms}`.
- **파일:** `model_loader.py`, `prompt_builder.py`, `orchestrator.py`, `app_gradio.py`.
- **공개 API:** `Orchestrator.generate(query, mode) -> Response`.
- **모델 로딩:** Base 모델은 한 번만 로드 (Qwen2.5-3B 4-bit, 약 2.5GB VRAM). Adapter는 런타임에 attach/detach되어 base 재로드 없이 QLoRA on/off를 전환한다.
- **추론 상한:** `max_new_tokens=512`; RAG 컨텍스트는 약 1500 토큰으로 제한 (초과 시 가장 낮은 순위 chunk부터 제거).

### 4.5 Evaluation (`src/eval/`)

- **책임:** 4-way ablation runner + scorer.
- **입력:** `data/eval/eval_set.jsonl` (`{question, reference_answer, expected_citations, reviewed_by, reviewed_at}`).
- **출력:** `reports/{run_id}/scores.csv`, `reports/{run_id}/report.md`.
- **파일:** `runner.py`, `judge.py`, `citation_checker.py`, `aggregate.py`.
- **Judge 루브릭:** 정확성(법적 정합성), 인용 적절성, 환각, 가독성에 대해 1~5점. Temperature=0, 명시적 루브릭 프롬프트.
- **Citation checker:** 정규식으로 `[민법 제○○조]`, `[대법원 ○○○○다○○○○○]` 패턴을 추출; 정규화된 코퍼스 인덱스에서 조회. 세 가지 지표를 보고한다: `citation_accuracy = found / total_cited` (모델 인용의 precision), `hallucination_rate = 1 - citation_accuracy`, `citation_recall = expected_citations_cited / expected_citations_total` (평가 항목의 `expected_citations` 필드 대비 coverage).
- **Run id:** `{timestamp}_{git_sha[:8]}`.

### 4.6 Common (`src/common/`)

Config 로더 (YAML), 구조화 로깅, path 헬퍼, Pydantic 스키마 (`Chunk`, `QAPair`, `EvalItem`, `Response`). 비즈니스 로직 없음.

### 4.7 모듈 의존성 그래프

```
common ◄── all modules
ingestion ──► common
rag ──► common, ingestion (chunks)
training ──► common, ingestion (qa)
serving ──► common, rag, training (adapter)
eval ──► common, serving
```

순환 없음. 엄격한 top-down 구조.

## 5. 데이터 흐름

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

모든 단계는 멱등적이다; 입력이 동일하면 동일 출력이 재현된다 (캐싱된 외부 API의 비결정성은 제외). `Makefile`을 통해 노출된다: `make ingest / embed / train / eval / serve`.

### RAG 프롬프트 템플릿

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

non-RAG 모드에서는 `[참고자료]` 섹션이 제거되고, 참고자료가 제공되지 않는다는 지시문으로 대체된다.

## 6. 저장소 레이아웃

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

## 7. 의존성

```
python = ">=3.10,<3.13"
torch = "==2.4.*"               # Unsloth-compatible pin
unsloth, transformers, peft, bitsandbytes, trl
chromadb, FlagEmbedding, rank-bm25
gradio, pydantic, anthropic
pytest, ruff
```

Windows 노트: `bitsandbytes`는 prebuilt wheel 또는 `bitsandbytes-windows-webui` fork가 필요할 수 있다. M0의 일부로 검증한다. Windows-native 설치 실패 시 WSL2로 fallback.

## 8. 에러 처리 및 안전장치

### 8.1 VRAM OOM (주요 리스크)

| 지점 | 완화책 |
|---|---|
| 학습 시작 | 첫 step에서 fail-fast; `max_memory_allocated()` 로깅 및 명시적 복구 힌트 (`max_seq` 1024로, `lora_r` 8로, `batch=1`로 감소). |
| 학습 중간 (긴 시퀀스) | `prepare_dataset.py`가 토큰 길이 분포를 보고; `max_seq`는 95 percentile에서 자동 권고; 초과 샘플은 truncation. |
| Embedding 인덱싱 | `EMBED_BATCH` 환경 변수에서 배치 크기 (기본 8); OOM 시 자동 절반 축소. |
| 추론 | `max_new_tokens=512`; RAG 컨텍스트 1500 토큰 제한; 가장 낮은 순위 chunk부터 제거. |

### 8.2 외부 API 장애

| API | 완화책 |
|---|---|
| 법령 OpenAPI | 디스크 응답 캐시; exponential backoff와 함께 3회 재시도. |
| 대법원 크롤 | robots.txt 준수; 요청당 최소 1초; 실패는 `data/cache/failed.jsonl`에 로깅; 파이프라인 계속 진행. |
| Claude API (synth + judge) | SDK 재시도; `MAX_API_USD` 환경 변수로 비용 상한; judge 결과를 JSONL로 스트리밍하여 재실행 시 이미 채점된 항목은 skip. |

### 8.3 데이터 품질 게이트

- Chunking: 빈/<200자 chunk 제거; hash로 중복 제거.
- 합성 Q&A: `[조문번호]` 형식 인용이 없는 응답, <50자 답변, 완전 중복 제거. seed→synthesized 비율 로깅.
- 평가셋: 100% 직접 검수; JSONL에서 `reviewed_by`, `reviewed_at` 필수.

### 8.4 인용 환각 방지

Citation checker는 LLM-as-Judge와 독립적인 핵심 객관 지표이다. 모델이 생성한 인용 중 코퍼스 인덱스에 존재하지 않는 것은 환각으로 집계되어 `hallucination_rate`에 반영된다.

### 8.5 재현성

- 고정 seed (`torch`, `transformers.set_seed`).
- Adapter 디렉터리에 `training_config.yaml` 포함.
- 모든 처리된 JSONL에 `pipeline_version` 및 `source_hash` 포함.
- Run id가 평가 리포트를 git SHA에 연결.

### 8.6 Out of scope (의도적 제외)

사용자 인증, 분산/멀티 GPU 학습, 커스텀 양자화, 고급 모니터링/알림, 복잡한 재시도 라이브러리, DB 마이그레이션.

## 9. 테스트 전략

### 9.1 피라미드 (PoC)

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

### 9.2 Unit 테스트 (TDD 적용)

| 모듈 | 케이스 |
|---|---|
| `chunk.py` | 조문 단위 분할, dedup, <200자 제거, 메타데이터 보존 |
| `citation_checker.py` | 정규식 추출 (`[민법 제618조]`, `[대법원 2019다12345]`), 코퍼스 조회, 환각 분리 |
| `bm25.py` | 토크나이즈, doc/query 스코어링 |
| `prompt_builder.py` | RAG vs no-RAG 템플릿, chunk-id 포맷팅, 토큰 상한 |
| `schemas.py` | Pydantic round-trip, 필수 필드 검증 |
| `synth_qa.py` | 후처리 (citation 필터, 길이 필터, dedup) |
| `aggregate.py` | 4-way 점수 테이블, mean/std, 모드 비교 |

Fixture: `tests/fixtures/sample_chunks.jsonl` (10개), `sample_eval.jsonl` (5개).

### 9.3 Integration 테스트

| 테스트 | 커버리지 |
|---|---|
| `test_retriever_e2e` | Fixture로 Chroma 인덱스 빌드; 알려진 쿼리가 기대한 chunk를 retrieve하는지 검증. |
| `test_orchestrator_modes` | 4개 모드 모두가 dispatch되며 서로 다른 prompt를 생성하는지 확인 (모델 호출은 mocked). |
| `test_eval_runner` | Mock orchestrator + mock judge; `scores.csv`의 형태와 내용 검증. |
| `test_citation_checker_with_corpus` | 실제 fixture 코퍼스 + 샘플 답변; accuracy 산식 검증. |
| `test_dataset_prep` | Seed → train/val split, chat template, max_seq truncation. |

### 9.4 E2E smoke (1개)

`test_smoke.py`: 5-chunk + 3-QA fixture → Qwen2.5-3B 4-bit 로드 (파인튜닝 없음) → `base` 모드에서 1개 쿼리 → citation checker 통과 → 1개 항목 평가셋에 대해 eval runner 성공. 학습은 smoke에 포함하지 않는다.

### 9.5 테스트 제외 (의도적)

실제 LLM 출력 품질, 외부 API (mocked), Gradio UI (수동), 학습 수렴 (TensorBoard로 시각 확인), bge-m3 내부 정확도.

### 9.6 TDD 범위

- **TDD 적용 (테스트 우선):** citation_checker, chunk, bm25, prompt_builder, aggregate, schemas, synth_qa 후처리.
- **Test-after 허용:** train_qlora, embed.py (라이브러리 wrapper), Gradio app, fetch_*.py, synth_qa API 호출 레이어.

### 9.7 평가 파이프라인 자가 점검

- 동일한 평가를 두 번 실행 → judge variance 측정.
- Sanity 서브셋 (10개 항목): RAG-only 5개, QLoRA-only 5개로 설계. 4-way 실행 후 기대 패턴 확인 (RAG 항목에서 RAG-only > base, QLoRA 항목에서 QLoRA-only > base). 패턴 위반은 orchestrator/eval 버그를 시사한다. Sanity 서브셋은 항상 평가셋의 일부로 실행된다.

## 10. 마일스톤

### M0 — 프로젝트 부트스트랩 (~½일)
- `git init`, `pyproject.toml`, `Makefile`, `README.md`, `.env.example`.
- `src/common/` 스켈레톤.
- 단순 pytest 통과.
- Windows에서 Unsloth + bitsandbytes 검증: Qwen2.5-3B 4-bit 로드 + 1문장 생성; VRAM을 README에 기록.

**Done:** Qwen2.5-3B 4-bit가 로컬에서 로드 및 생성됨; VRAM 측정치 기록 완료.

### M1 — Ingestion + RAG baseline (1~2일)
- 民法 전체 조문 수집 (약 1,118개 조).
- `chunk.py`, `embed.py`, `index.py`, `bm25.py`, `retriever.py` + 테스트.

**Done:** `python -m src.rag.retriever "임대차계약 갱신요구권"`이 관련 조문(예: 제643조, 제654조)을 포함한 top-5 chunk를 반환.

### M2 — Orchestrator + Gradio + base/RAG 모드 (1일)
- `model_loader.py`, `prompt_builder.py`, `orchestrator.py` (base + rag 모드만).
- RAG on/off 토글이 있는 `app_gradio.py`.
- `citation_checker.py` + 테스트.

**Done:** Gradio UI가 base와 RAG 모드의 답변을 나란히 표시; citation_checker가 추출된 인용과 코퍼스 매칭 상태를 표시.

### M3 — Synthesis + QLoRA 학습 (2~3일)
- `synth_qa.py`로 1,000~3,000개의 합성 Q&A 쌍 생성.
- `prepare_dataset.py` + `train_qlora.py`; 한 차례의 풀 학습.
- Adapter를 orchestrator에 통합 → 4개 모드 모두 동작.
- Gradio에 QLoRA on/off 토글 추가.

**Done:** Adapter 디렉터리 생성; UI에서 4개 모드 비교; loss curve가 발산하지 않음.

### M4 — 평가셋 + 4-way ablation (2~3일)
- `eval_set.jsonl`의 100~150 항목 직접 검수 (계약법 + 임대차 중심).
- `runner.py`, `judge.py`, `aggregate.py` → 첫 `report.md`.

**Done:** LLM-as-Judge 평균, Citation Accuracy, 환각률을 포함한 4-way 점수 테이블; 모드 간 통계적 차이 문서화.

### M5 (선택) — 개선 사이클 (~1주)
M4에서 약점 영역 식별 → 합성 Q&A 보강 → 재학습 (v2 adapter) → 재평가; 필요 시 평가셋 확장.

### Critical Path

```
M0 ─► M1 ─┬─► M2 ─┬─► M3 ─► M4 ─► M5
          │       │
          └───────┘
```

M4의 평가셋 직접 검수가 병목이며, M0~M3 동안 병렬로 시작한다.

## 11. 리스크

| 리스크 | 가능성 | 영향 | 완화책 |
|---|---|---|---|
| Windows + bitsandbytes 비호환 | Medium | Critical | M0에서 검증; WSL2로 fallback. |
| 6GB VRAM 학습 OOM | Medium | High | `max_seq`, `lora_r` 단계적 축소; 최종 fallback은 Qwen2.5-1.5B. |
| 합성 Q&A 품질 부족 | Medium | High | M3 종료 시점에 50 항목 직접 검수; <70% 통과 시 prompt 개선 후 재생성. |
| 대법원 크롤 차단 / 구조 변경 | Medium | Medium | M1은 조문만으로 시작; 판례는 M5로 연기. |
| LLM judge variance | Low | Medium | 2회 실행 variance 점검; 루브릭 명시; variance 높으면 휴먼 검수 샘플 확장. |
| 직접 검수 시간 부족 | Medium | High | 80개를 M4 최소 기준으로 처리; M5에서 100~150개로 확장. |

## 12. Definition of Done (PoC)

PoC는 다음 조건이 모두 충족되면 완료된다:

1. All four modes (base / rag / qlora / rag_qlora) are operational on the same infrastructure.
2. ≥80-item evaluation set runs through 4-way ablation at least once successfully.
3. Citation Accuracy + LLM-as-Judge scores + hallucination rate are tabulated per mode.
4. Sanity-check eval subset shows expected pattern (eval pipeline trustworthy).
5. Gradio UI compares all 4 modes for arbitrary user input.
6. `report.md` states explicit conclusions: *"In this domain, RAG contributes ___, QLoRA contributes ___, combined they ___"*.

## 13. Out of Scope (본 PoC)

- M5 탐색 범위를 넘어서는 판례 크롤링
- 멀티턴 대화 / 세션 관리
- 사용자 인증, 감사 로깅
- 프로덕션급 서빙 (vLLM, TGI 등)
- BM25를 위한 한국어 형태소 분석기 튜닝
- v2/v3 adapter 비교 자동화 (M5 수동 작업으로 허용)
