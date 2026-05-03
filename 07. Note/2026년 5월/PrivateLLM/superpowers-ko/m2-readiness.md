# M2 준비도 — 사전 계획 체크리스트

이 문서는 Plan 1(M0 + M1) 종료 시점의 상태를 기록하고, Plan 2(M2 — Orchestrator + Gradio UI, base / RAG 모드) 착수 전에 충족되어야 할 모든 항목을 나열한다. 이는 작업 문서이다: 항목이 처리될 때마다 체크하고, 새로운 항목은 추가할 수 있다.

## Plan 1 요약

**Tags:** `m0-bootstrap`, `m1-rag-baseline`
**Tests:** 29 passed (27 fast + 2 slow opt-in via `-m slow`)
**확보된 수용 신호:**
- Qwen2.5-3B 4-bit가 RTX 3060 Laptop 6 GB에서 로드됨 → VRAM 2.07 GB; 한국어 텍스트를 정확히 생성함 (`scripts/verify_unsloth.py`).
- E2E retriever: chunk(fixture) → bge-m3 embed → ChromaDB+BM25 hybrid → "임대차의 정의를 설명해 주세요" 쿼리에 대해 民法 第618조가 검색됨 (`tests/test_e2e_rag.py`).

### 전달된 모듈

| 모듈                          | 파일                                                  | 테스트 방법                                                                                                          |
| --------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/common/`               | `schemas.py`, `paths.py`, `config.py`, `logging.py` | `test_schemas.py` (TDD, 11 tests)                                                                               |
| `src/ingestion/`            | `fetch_statutes.py`, `chunk.py`                     | `test_ingestion_smoke.py` (mocked, 2), `test_chunk.py` (TDD, 6)                                                 |
| `src/rag/`                  | `bm25.py`, `embed.py`, `index.py`, `retriever.py`   | `test_bm25.py` (TDD, 4), `test_retriever.py` (TDD, 4 unit + 1 slow integration), `test_e2e_rag.py` (1 slow E2E) |
| `scripts/verify_unsloth.py` | transformers fallback이 적용된 M0 수용 스크립트               | 수동 (README에 VRAM + 생성 결과 기록)                                                                                    |

### 원래 계획에서의 경험적 조정 사항

다음은 현재 Windows + Python 3.12 생태계에 의해 강제된 사항이다:

- `requires-python = ">=3.10,<3.13"` (유지) — README가 3.12를 명시적으로 요구함
- `transformers>=4.44,<4.50` (이전: `<5`) — 4.50+ 에서 peft / sentence_transformers / FlagEmbedding이 깨짐
- `bitsandbytes>=0.43,<0.49` (이전: 상한 미설정) — 0.49.x는 C++ 커널을 건너뜀
- `pyarrow>=21,<24` (추가됨) — pyarrow 24가 torch+sklearn+pandas 조합에서 Windows ACCESS_VIOLATION 유발
- `unsloth_zoo`를 `[unsloth]` extra에 추가
- `tool.ruff.target-version = "py310"` (이전: `py311`) — 실제 하한선과 일치시킴
- `scripts/verify_unsloth.py`는 Unsloth가 `torch>=2.5`를 요구하므로 vanilla `transformers`로 fallback

## 최종 리뷰의 미해결 이슈 (Plan 1)

다음 항목들은 최종 코드 리뷰어가 "Important"로 표시한 것이다. Plan 1 종결을 막는 항목은 없으며, 모두 M2 준비 단계 또는 첫 M2 작업 중에 해결되어야 한다.

### Important (M2 코드 추가 전 해결 필요)

#### A. FlagEmbedding venv 패치가 소스 컨트롤에 문서화되지 않음

`.venv/Lib/site-packages/FlagEmbedding/finetune/embedder/encoder_only/m3/runner.py`가 in-place로 패치되어 (`dtype=` → `torch_dtype=`) `BGEM3FlagModel`이 transformers 4.49와 호환되도록 했다. 이 패치는 gitignore되어 있으므로 — venv를 다시 생성하는 사람은 누구나 이를 다시 적용해야 한다.

**Options:**
1. `FlagEmbedding`을 해당 버그가 없는 버전으로 핀(pin) 고정 (조사 필요: 1.3.x는 이 이슈 이전이지만 bge-m3-fp16을 동일한 방식으로 지원하지 않을 수 있음).
2. sed 스타일의 패치를 적용하는 `scripts/postinstall.py`를 추가하고 README quickstart에 문서화.
3. `FlagEmbedding.finetune.*`를 import하지 않는 최소한의 embedder wrapper를 vendor화.

**필요한 결정:** 어떤 옵션을, 누가, 언제.

#### B. 결합 VRAM 예산이 검증되지 않음

bge-m3 (≈ 1.3–1.5 GB fp16) + Qwen2.5-3B 4-bit (≈ 2.1 GB) + KV cache + RAG context (≈ 0.5–1 GB) ≈ 6 GB 카드에서 4 GB. 빡빡하지만 가능성 있음.

**Action:** **첫 M2 작업으로** `scripts/verify_combined_vram.py`를 추가 — 두 모델을 모두 로드하고, 하나의 RAG 쿼리를 실행하고, `torch.cuda.max_memory_allocated() < 5.5 GB`를 assert. 실패하면 orchestrator를 만들기 전에 완화 조치 (쿼리 시 CPU embedder, bge-m3의 fp16 설정 낮추기 등) 적용.

#### C. ChromaDB `Settings` 결합

`src/rag/index.py:51`과 `src/rag/retriever.py:88` 모두 `chromadb.PersistentClient(...)`에 `Settings(allow_reset=True)`를 전달한다. orchestrator(M2)는 세 번째 호출자이다. 이를 잊으면 캐시된 settings 불일치로 예외가 발생한다.

**Action:** `_make_chroma_client(path)` helper를 `src/rag/index.py` (또는 새로운 `src/rag/_chroma.py`)로 분리하고 세 곳 모두에서 이를 사용한다.

#### D. Retriever의 dense `(id, score)` 튜플이 오해를 부름

`retriever.py`의 `dense_ranked = [(doc_id, 1.0 - dist) for ...]`. cosine distance에서 `1 - dist ∈ [-1, 1]` — 음수 score는 합리적인 필터링 임계값이 아니다. RRF는 rank만 사용하므로 score는 미사용이지만, 향후 호출자가 오용할 수 있다.

**Action:** `dense_ranked: list[str]` (rank만)로 변경하고 `rrf_fuse` signature도 그에 맞게 갱신한다. 향후 잘못된 공식으로 회귀되는 것을 잡기 위해 정확한 RRF score (`1/(k+rank+1)`)에 대한 assertion을 하나 추가한다.

### Minor (대기열 추가 또는 보류)

- `config/default.yaml`에 dead key가 있음: 최상위 `paths:` 블록은 한 번도 읽히지 않으며, `chunk.overlap`도 한 번도 읽히지 않음.
- `src/ingestion/chunk.py:43`은 `_ARTICLE_NO_RE`를 정의하지만 사용하지 않음.
- `src/rag/retriever.py:30`은 `RetrievalHit` dataclass를 정의하지만 `Retriever.search`는 `list[Chunk]`를 반환함. 제거하거나 serving 용도로 노출.
- `src/common/logging.py`의 `_initialized` 플래그는 thread-safe하지 않음; M2의 Gradio가 worker thread를 사용한다면 두 thread가 root handler를 중복 초기화할 수 있음.
- `Makefile` `clean` target은 POSIX `rm -rf`를 사용함 (Windows는 git-bash가 필요함).
- `.gitignore`는 `data/eval/.gitkeep`을 whitelist하지만 해당 파일이 존재하지 않음; 디렉터리에 anchor가 없음.

## Plan 2 시작 전 필수 사항

- [x] **A** — FlagEmbedding 패치를 `scripts/postinstall.py` (option 2)로 자동화. `from_pretrained(... dtype=)` -> `torch_dtype=`로의 멱등(idempotent) 재작성. `tests/test_postinstall.py`로 고정. `make postinstall`. — 2026-05-03
- [x] **B** — `scripts/verify_combined_vram.py`가 두 모델이 모두 로드된 상태에서 end-to-end RAG turn을 실행; **타겟 RTX 3060 Laptop에서 측정된 peak 3.17 GB / 5.5 GB 예산**, 충분한 헤드룸 확보. — 2026-05-03
- [x] **C** — `src/rag/index.py`에 `make_chroma_client(path)` factory; `index.py`와 `retriever.py` 모두 이를 호출. — 2026-05-03
- [x] **D** — `rrf_fuse(dense_ids, sparse_ids, ...)`가 이제 bare `list[str]`을 받음; canonical `1/(k+rank+1)` 공식을 회귀 테스트로 고정. — 2026-05-03

네 항목 모두 2026-05-03에 해결됨. 다음: `docs/superpowers/specs/2026-05-03-m2-orchestrator-design.md`를 작성 (또는 기존 spec의 Serving subsystem 섹션을 확장)하고 Plan 2를 위해 `/superpowers:writing-plans`를 실행.

## Plan 2가 전달할 것 (preview)

design spec §4.4 (Serving / Orchestrator)에서:

- `src/serving/model_loader.py` — base 4-bit + (이후) adapter attach/detach
- `src/serving/prompt_builder.py` — RAG vs no-RAG 프롬프트 템플릿
- `src/serving/orchestrator.py` — `Orchestrator.generate(query, mode)`, 모드는 `{base, rag}` (M2)와 이후 `{qlora, rag_qlora}` (M3)
- `src/serving/app_gradio.py` — 모드 토글, 검색된 컨텍스트 표시, citation 추출 기능을 갖춘 single-file Gradio UI (`src/eval/citation_checker.py`가 가용해야 함)

Plan 2는 `src/eval/citation_checker.py` (regex 추출 + corpus lookup)도 포함해야 하는데, Gradio UI가 이를 시각적인 citation 상태 표시에 사용하기 때문이다. 전체 evaluation runner (`runner.py`, `judge.py`, `aggregate.py`)는 Plan 4 (M4)에 속한다.

## Plan 2의 권장 첫 세 작업

1. **Bootstrap polish** — 위의 A, B, C, D를 구현; 각 수정을 별도로 커밋.
2. **`src/serving/__init__.py` + `model_loader.py`** — singleton 4-bit base loader, `verify_unsloth.py`의 기존 transformers fallback 로직을 공유.
3. **`src/serving/prompt_builder.py` + tests (TDD)** — RAG vs no-RAG 템플릿, chunk-id 포맷팅, max-context-token 한도. 순수 로직이며 TDD 친화적인 가장 작은 조각.

이 셋이 끝나면, orchestrator와 Gradio UI는 기계적으로 따라온다.

---

## Plan 2 — closed (2026-05-03)

**Tag:** `m2-orchestrator`
**Tests:** 52 fast + 4 slow. 타겟 호스트에서 slow 실행 시간 ~80 s.

### 전달된 모듈

| 모듈 | 파일 | 테스트 방법 |
|---|---|---|
| `src/serving/` | `model_loader.py`, `prompt_builder.py`, `orchestrator.py`, `app_gradio.py` | `test_prompt_builder.py` (7 TDD), `test_orchestrator.py` (4 unit, thread-lock 포함), `test_serving_integration.py` (2 slow — singleton + 실제 모델 round-trip) |
| `src/eval/` | `citation_checker.py` | `test_citation_checker.py` (9 TDD; regex variants + dedup + corpus stub) |
| `config/prompts/` | `rag.txt`, `no_rag.txt` | 모든 prompt_builder 테스트에서 사용됨 |
| `scripts/` | `verify_combined_vram.py` (공유 loader를 사용하도록 리팩터링) | `make`마다 재실행 게이트 |
| `pyproject.toml` | `gradio>=4,<6` 선언 | UI smoke + HTTP probe |

### 원래 Plan 2 문서로부터의 편차

- 계획의 `tests/test_serving_integration.py::test_orchestrator_real_models_round_trip`은 리터럴로 `c.statute_no == "618"`을 assert했음; 기존 `chunk_civil_code` schema와 일치시키기 위해 `"제618조" in c.statute_no`로 수정 (기존 `test_e2e_rag.py` 컨벤션을 따름). Plan 문서는 commit `cef8262`에서 갱신됨.
- 계획은 `gradio`가 이미 `[dev]` extras에 있다고 가정했으나 — 그렇지 않았음. Task 6 재시도 전에 main `dependencies`에 추가 (commit `9e85cd1`).
- Task 2는 truncation-warning 로그 + `caplog` 회귀 테스트를 추가 (spec의 "truncation never happens silently" 약속에 대한 Important-rated 리뷰 피드백). 구현은 이제 spec §2.2와 엄격히 일치함.
- Task 1은 DCL publish-order race를 수정 (Important-rated 리뷰 피드백): singleton의 `_state.model`이 이제 마지막에 할당되므로, `get_base_model()` 상단의 fast-path 읽기가 절반만 초기화된 상태를 관찰할 수 없음.

### 확보된 수용 신호 (plan §5 기준)

- ✅ `make serve`가 `http://127.0.0.1:7860`을 엶; 타겟 호스트에서 HTTP `/`와 `/config`가 200을 응답.
- ✅ Slow E2E: orchestrator가 "임대차의 정의를 설명해 주세요."에 대해 warmup 후 <15 s 안에 base + rag을 반환; rag은 民法 第618條을 안정적으로 검색.
- ✅ Citation extractor가 생성된 답변에서 `[민법 제618조]` 형태의 citation을 추출 (`extract_citations`)하고, UI는 `verify_citations(corpus=None)` stub을 통해 이를 badge로 표시.
- ✅ `pytest -m "not slow"`가 52개 테스트에서 green.
- ✅ `scripts/verify_combined_vram.py`가 peak 3.17 GB / 5.5 GB 예산에서 계속 PASS — Task 5의 orchestrator를 통한 full RAG turn이 헤드룸을 회귀시키지 않았음.

### Plan 3 (M3 — QLoRA training)을 위한 Punchlist

다음 항목들은 M2 블로커는 아니지만 M3 전/중에 다뤄져야 한다:

- Unsloth를 unblock하기 위해 torch 2.4 → 2.5+ pin을 해결 (env-pins 메모리에 cascade가 문서화됨). transformers fallback은 2.4에서 작동하지만, 우리가 여기 있는 이유는 Unsloth의 최적화 때문이다.
- `model_loader.py`의 adapter attach/detach (`_State` dataclass는 M3-ready 상태).
- `Response.mode` 리터럴은 이미 `qlora`와 `rag_qlora`를 포함; `Orchestrator.generate`의 런타임 dispatch는 `Mode` alias를 넓히고 prompt 템플릿을 분기시켜야 함.
- `Orchestrator.open()`은 thread-safe하지 않음 — `app_gradio.py`의 lazy module-level `_ORCH`와 Gradio의 request queue 덕분에 1-user PoC에서는 문제가 없으나, multi-user 진입점이 생기면 재검토.

### Plan 2 동안 처리된 minor cleanup

- `config/default.yaml`에서 dead 최상위 `paths:` 블록과 `chunk.overlap` key가 제거됨; per-call `serving:` 섹션이 추가됨.
- README의 status table, file map, test count가 새 트리에 맞춰 동기화됨.

위의 §"Minor (queue or defer)" 목록의 minor 항목들 (`_ARTICLE_NO_RE` 미사용, `RetrievalHit` 미사용, logging thread-safety, `make clean` POSIX 전용, `data/eval/.gitkeep` 누락)은 여전히 미해결 상태이며 M3 / M4 준비 중에 처리되어야 한다 — 이들 중 어느 것도 Plan 2를 막지 않았다.

---

## M3a — 코드 완료, smoke gate 대기 중 (2026-05-03)

**Status:** 코드 경로 전달 완료 (`docs/superpowers/plans/2026-05-03-m3a-qlora-smoke.md`의 Tasks 1–7); 실제 `make smoke-train`은 `ANTHROPIC_API_KEY`와 `LAW_OPEN_API_KEY`가 `.env`에 없어 아직 실행되지 않음. **`m3a-qlora-smoke` 태그는 아직 없음** — smoke gate 통과 시점에 부여 예정.

**HEAD:** `eb79a84` (Task 7 commit).
**Tests:** 71 fast + 5 slow 전부 green (slow set에는 mocked Claude + 2 학습 step의 신규 `test_smoke_train.py` E2E가 포함됨).

### 전달된 모듈

- `src/training/__init__.py`, `synth_qa.py`, `prepare_dataset.py`, `train_qlora.py`
- `src/serving/model_loader.py`가 `attach_adapter` / `detach_adapter` / `current_adapter`로 확장됨
- `data/processed/qa_seed.jsonl` (임대차/매매/채권/손해배상/시효/동시이행/해제/변제/보증/위임/사용대차를 다루는 12개의 손수 큐레이션 pair)
- `config/prompts/synth_qa.txt` few-shot 템플릿
- `config/default.yaml`이 `synth:` 및 `training:` 블록으로 확장됨
- `Makefile`: `synth`, `prepare`, `smoke-train` 타겟

### 작성된 M3a 계획으로부터의 편차

- Task 1 follow-up: `[unsloth]` extras가 이제 동작하는 cu126 세트로 `xformers / torchao / torchvision / diffusers / triton-windows`를 핀(commit `2369ccc`). 이게 없으면 향후 clean-venv `make install-unsloth`가 다시 깨짐.
- Task 5 lint: ruff E741 때문에 한 테스트에서 `l → lab` 리네임 강제됨 (코스메틱).
- Task 7 patches:
  - `_JsonlLossCallback`이 `transformers.TrainerCallback`을 상속 (필수 — Trainer가 모든 callback에 `on_init_end` 등을 디스패치함).
  - `fp16=True` 대신 `bf16=torch.cuda.is_bf16_supported()`. RTX 3060 Laptop은 Ampere → bf16 경로; Unsloth의 LoRA 커널이 fp16에서 `RuntimeError: self and mat2 must have the same dtype, but got Half and Float`을 발생시켰음. bf16은 LoRA 안정성 면에서도 일반적으로 더 나음.
  - Slow integration 테스트가 mocked synth를 5 → 20쌍으로 늘리고 `smoke_max_steps=2` + `grad_accum=1`로 오버라이드하여 실제로 2번의 optimizer step이 로깅되도록 함.

### M3a를 마치기 위해 필요한 것 (API 키가 가용해질 때)

1. `C:\claudeProject\PrivateLLM\.env`에 `ANTHROPIC_API_KEY` + `LAW_OPEN_API_KEY` 배치.
2. 순서대로 실행:
   ```powershell
   make ingest                             # ~1 min, free
   python -m src.ingestion.chunk           # writes data/processed/chunks.jsonl
   make synth                              # ~5–10 min, ~$1–3 Anthropic API
   make prepare                            # ~10 s
   make smoke-train                        # ~10–20 min GPU
   ```
3. smoke gate가 PASS를 출력하면 (`final_loss < first_loss × 0.95`):
   ```powershell
   git tag m3a-qlora-smoke -m "M3a: QLoRA smoke pipeline gate passed"
   ```
4. FAIL이면: `runs/smoke-*/train_log.jsonl`, adapter 디렉터리 상태, verdict 라인을 캡처하여 (a) `config/prompts/synth_qa.txt`의 prompt-engineering iteration, (b) 하이퍼파라미터 조정(lr / r), (c) 더 큰 smoke (target_pairs=400) 중 결정하기 위해 보고.

### M3b open questions (M3a 태그까지 보류)

- Full synth target_pairs: design은 1000–1500을 명시. M3a synth 품질에 따라 결정.
- ~~`Mode`를 `qlora`와 `rag_qlora`로 확장하는 것이 모드별 prompt 템플릿 차이도 함께 풀어야 하는지 여부.~~ — **2026-05-03 M3b-prep T1에서 결정: NO. qlora는 no_rag.txt를, rag_qlora는 rag.txt를 재사용; adapter가 도메인 지식을 인코딩하고, prompt는 역할을 인코딩한다. 모드별 템플릿 조정은 M4 eval에서 도움이 되겠다는 신호가 나오면 재검토 가능.**
- 더 많은 버전이 생성됨에 따른 adapter 디렉터리 명명 규칙 (`v1` / `v2` / ...).

---

## M3b prep — landed (2026-05-03)

M3a smoke API key를 기다리는 동안, 실제 adapter가 필요하지 않은 M3b 코드가 들어갔다. HEAD `c870250`. **태그는 아직 없음** — M3b proper는 4-mode slow integration 테스트가 실제 adapter를 대상으로 실행된 후에 종결됨 (M3a 이후).

**Tests:** 82 fast (75 → +7 orchestrator dispatch) + 5 slow (변동 없음). 4-mode slow integration 테스트 보류 — wiring은 존재하지만 의미를 가지려면 실제 adapter가 필요.

### 전달된 모듈

- `src/serving/prompt_builder.py` — `Mode`가 4-way로 확장; `_PROMPT_FILES`, `_RAG_MODES`, `_VALID_MODES` 상수; `_RAG_MODES` 기반 디스패치.
- `src/serving/orchestrator.py` — 4-mode 디스패치, dataclass에 `adapter_path` 필드 (`.open()`에서 `cfg.serving.adapter_path` 읽음), 기존 `_lock` 아래에서의 per-call attach/detach 정책. adapter 없이 qlora/rag_qlora를 호출하면 다른 작업을 하기 전에 `RuntimeError`를 발생.
- `src/serving/app_gradio.py` — 2x2 grid (base / rag / qlora / rag_qlora). `_generate_safely`가 각 호출을 wrap하여 mode별 실패(예: `RuntimeError("no adapter configured")`)가 UI를 죽이는 대신 셀에 렌더링되도록 함.
- `config/prompts/qlora.txt`, `config/prompts/rag_qlora.txt` — 각각 no_rag/rag와 내용이 동일.
- `config/default.yaml` — `serving.adapter_path: null` (M3a smoke 학습이 완료된 후 `models/adapters/<name>/` 경로로 설정).

### 엄격한 "M3b-as-written" 계획으로부터의 편차

- `Orchestrator.generate`에서 live-model rebinding은 `swapped=True` (attach 또는 detach가 실제로 실행된 후에만)에 게이팅됨, spec의 always-on 패턴이 아님. 이유: 알파벳 순 테스트 정렬이 `tests/test_adapter_hooks.py`(MagicMock을 `_ml._state.model`에 쓰는)를 `tests/test_orchestrator.py`보다 앞에 놓는데, always-on rebind는 각 테스트의 추적 중인 `_FakeModel` 대신 남아 있는 mock을 집어들게 됨. 게이트는 unit 테스트를 격리하면서 real-path 시맨틱을 보존.
- `test_invalid_mode_raises` orchestrator 테스트는 `mode="bogus"`를 사용하도록 갱신됨 (qlora가 이제 valid이므로) — Mode 확장의 일부이므로 같은 커밋.

### M3b proper에 여전히 열려 있는 것

1. **Real-adapter slow integration 테스트** — 학습된 adapter로 4개 모드 모두를 end-to-end 실행. 코드 형태는 준비됨 (`test_orchestrator_real_models_round_trip` 미러); M3a가 `models/adapters/qwen2.5-3b-civil-smoke-v0/`를 생산하는 데 막혀 있음.
2. smoke adapter 경로가 존재하면 `config/default.yaml`의 **`serving.adapter_path`를 설정**.
3. slow E2E 테스트가 통과한 후 **`m3b-orchestrator-4-mode` 태그**.
4. **Full 학습 실행 (target_pairs ≥ 1000)** — 별도의 plan; M4 평가용으로 `qwen2.5-3b-civil-v1/`을 생산.
