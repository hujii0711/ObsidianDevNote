# 프로젝트 디렉터리 구조 및 모듈 역할

이 문서는 Legal PrivateLLM PoC의 코드 구성 요소(.py 파일)와 각 모듈의 역할을 정리한다. 마지막 업데이트: 2026-05-03.

## 1. 디렉터리 트리 (소스 영역만)

```
PrivateLLM/
├── pyproject.toml         # 프로젝트 메타데이터 + 의존성
├── Makefile               # 워크플로 단축 명령
├── README.md
├── config/                # YAML 설정 + 프롬프트 템플릿
├── scripts/               # 검증 / postinstall 헬퍼
├── src/                   # 핵심 소스
│   ├── common/            # 스키마, config, paths, logging
│   ├── ingestion/         # 법령 fetch + chunking
│   ├── rag/               # 임베딩 + BM25 + ChromaDB + hybrid retriever
│   ├── serving/           # 모델 로더 + prompt builder + orchestrator + Gradio UI
│   ├── training/          # synth Q&A + tokenize + QLoRA 학습
│   └── eval/              # citation 추출 (M4 evaluation runner는 미구현)
├── tests/                 # pytest (fast + slow marker로 분리)
└── docs/                  # 디자인 spec, plan, 본 문서
```

## 2. `src/common/` — 공통 기반

| 파일 | 역할 |
|---|---|
| `__init__.py` | "스키마, config, paths, logging" 묶음의 sub-package marker (docstring 한 줄). |
| `schemas.py` | 모듈 경계의 wire format을 정의하는 Pydantic 모델(`Chunk`, `QAPair`, `EvalItem`, `Citation`, `Response`). |
| `config.py` | `config/default.yaml`을 읽고 `.env`를 부수효과로 로드하는 `load_config()`; `EMBED_BATCH` env override 지원. |
| `paths.py` | `src/`의 부모를 `ROOT`로 잡아 `RAW`/`PROCESSED`/`CHROMA`/`ADAPTERS` 등 절대 경로 상수와 `ensure_dirs()` 제공. |
| `logging.py` | 모듈별 `get_logger(name)` 진입점; `LOG_LEVEL` env로 stderr 핸들러를 1회 초기화. |

## 3. `src/ingestion/` — 데이터 인제스션

| 파일                  | 역할                                                                                                             |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| `__init__.py`       | "Data ingestion: fetchers and chunkers" sub-package marker.                                                    |
| `fetch_statutes.py` | 국가법령정보 OpenAPI에서 민법 JSON을 받아 `data/raw/statutes/civil_code.json`에 멱등 캐시(`fetch_to_disk`, 지수 백오프 retry).        |
| `chunk.py`          | 민법 payload를 조문 단위 `Chunk`로 변환(`chunk_civil_code`, `dedupe_by_hash`, `make_chunk_id`)하고 결과를 `chunks.jsonl`로 저장. |

## 4. `src/rag/` — 검색

| 파일 | 역할 |
|---|---|
| `__init__.py` | "RAG components" sub-package marker. |
| `bm25.py` | 한글/영문/숫자 토큰화 + `rank_bm25` 기반 `BM25Index`(build/search/save/load) 및 `BM25Hit` 정의. |
| `embed.py` | `BAAI/bge-m3` dense 임베더 Singleton(`Embedder.get`); CUDA OOM 발생 시 batch_size를 절반으로 재시도. |
| `index.py` | `chunks.jsonl`로부터 ChromaDB(`PersistentClient`)와 BM25 pickle을 함께 빌드하는 `build_indexes`; 공유 `make_chroma_client` 팩토리 제공. |
| `retriever.py` | dense+sparse 결과를 RRF로 융합하는 hybrid `Retriever.open/search` 및 순수함수 `rrf_fuse`. |

## 5. `src/serving/` — 추론 서빙

| 파일                  | 역할                                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`       | "Serving subsystem" sub-package marker.                                                                                                                                      |
| `model_loader.py`   | Qwen2.5-3B 4-bit 로딩 process Singleton; Unsloth 시도 후 실패 시 transformers+bitsandbytes로 fallback, `attach_adapter`/`detach_adapter`/`current_adapter` PEFT 훅 포함.                 |
| `prompt_builder.py` | (query, mode, chunks)를 chat-template 메시지로 변환하는 순수 로직 `build_messages`; `config/prompts/`의 system prompt를 mode별 로드, RAG context는 `max_context_tokens` 캡으로 lowest-rank부터 drop. |
| `orchestrator.py`   | base/rag/qlora/rag_qlora 4모드의 단일 진입점 `Orchestrator.generate`; retriever 호출, 모드별 adapter 정합 관리, `model.generate`까지 lock으로 직렬화 후 citation 추출하여 `Response` 반환.                  |
| `app_gradio.py`     | 동일 질의에 대해 4모드를 2x2 그리드로 비교하는 Gradio Blocks UI(`build_ui`, `run_query`); 127.0.0.1:7860 로컬 바인드.                                                                               |

## 6. `src/training/` — QLoRA 학습 파이프라인

| 파일 | 역할 |
|---|---|
| `__init__.py` | "Training subsystem" sub-package marker. |
| `synth_qa.py` | Claude few-shot으로 한국어 민법 Q&A를 합성하는 resumable `run_synth`; citation/length/dedupe 필터(`filter_synthesized`), USD 비용 추정(`estimate_call_cost`), 디스크 캐시 포함. |
| `prepare_dataset.py` | `QAPair`를 chat-template로 토크나이즈하고 instruction 측을 `-100`으로 마스킹한 JSONL을 90/10 split으로 출력하는 `prepare`; `max_seq` 미지정 시 p95 기반 자동 산정. |
| `train_qlora.py` | Unsloth `FastLanguageModel` 위에 LoRA를 wrap하고 `Trainer`로 smoke/full 모드 학습; JSONL loss 콜백(`_JsonlLossCallback`)과 smoke gate(`_check_smoke_gate`)로 첫/마지막 loss 감소율 검증. |

## 7. `src/eval/` — 평가

| 파일 | 역할 |
|---|---|
| `__init__.py` | M2에서는 citation_checker만 ship; runner/judge는 M4로 미룬다는 docstring marker. |
| `citation_checker.py` | 답변 텍스트에서 조문(`제618조`, `제643조의2`)과 판례(`2019다12345`) 인용을 regex로 추출(`extract_citations`)하고 corpus 멤버십을 stub으로 검증(`verify_citations`). |

## 8. `scripts/` — 환경 검증 / postinstall

| 파일 | 역할 |
|---|---|
| `postinstall.py` | FlagEmbedding 1.4.0의 `dtype=` kwarg를 transformers≥4.46과 호환되는 `torch_dtype=`으로 idempotent하게 in-place 패치. |
| `verify_unsloth.py` | Qwen2.5-3B 4-bit 로딩 + 한 번의 generation으로 로컬 환경 헬스체크; `model_loader`의 fallback 경로도 함께 검증. |
| `verify_combined_vram.py` | M2 readiness gate B — embedder + 4-bit LLM을 한 프로세스에 동시 로드해 end-to-end RAG 1턴을 돌리고 peak VRAM이 5.5 GB 미만인지 확인(초과 시 비0 종료). |

## 9. `tests/` — 테스트

테스트 파일은 대응되는 `src/` 모듈명을 따른다. fast(default)와 slow(`-m slow`)로 분리.

| 파일 | 대상 모듈 | 종류 |
|---|---|---|
| `__init__.py` | (빈 marker) | — |
| `test_schemas.py` | `src/common/schemas.py` | fast |
| `test_chunk.py` | `src/ingestion/chunk.py` | fast |
| `test_ingestion_smoke.py` | `src/ingestion/fetch_statutes.py` (network mocked) | fast |
| `test_bm25.py` | `src/rag/bm25.py` | fast |
| `test_retriever.py` | `src/rag/retriever.py` (`rrf_fuse` 단위) | fast |
| `test_e2e_rag.py` | `src/ingestion` + `src/rag` 전체 (fixture → index → retrieve) | slow |
| `test_citation_checker.py` | `src/eval/citation_checker.py` | fast |
| `test_prompt_builder.py` | `src/serving/prompt_builder.py` | fast |
| `test_orchestrator.py` | `src/serving/orchestrator.py` (mocked model + retriever) | fast |
| `test_adapter_hooks.py` | `src/serving/model_loader.py` (`attach_adapter`/`detach_adapter`) | fast |
| `test_serving_integration.py` | `src/serving/model_loader.py` 실제 GPU Singleton | slow |
| `test_synth_qa.py` | `src/training/synth_qa.py` post-processing | fast |
| `test_prepare_dataset.py` | `src/training/prepare_dataset.py` (실 Qwen2.5 tokenizer) | fast |
| `test_smoke_train.py` | `src/training/train_qlora.py` 전체 파이프라인(GPU+Unsloth) | slow |
| `test_postinstall.py` | `scripts/postinstall.py` regex | fast |
