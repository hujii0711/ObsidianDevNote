# M3a — QLoRA smoke 파이프라인 (synth → prepare → train → smoke gate) — 설계 명세

- **Date:** 2026-05-03
- **Author:** `2026-05-02-legal-privatellm-mvp-design.md` (frozen)의 연속 — smoke slice를 위해 §4.3 Training을 정제.
- **Status:** Ready for plan
- **Scope:** 마일스톤 M3의 전반부. 전체 QLoRA 학습 파이프라인 + adapter attach/detach hook + 200쌍 / 1-epoch smoke 실행을 도입하여 전체 학습 실행(M3b)을 게이팅한다.

## 0. 왜 M3를 M3a + M3b로 분리하는가

MVP 설계는 M3를 하나의 마일스톤으로 다뤘다. 실제로는 작업이 폭발 반경이 다른 두 영역으로 나뉜다:

- **M3a (이 문서):** torch 생태계 업그레이드 + 새로운 학습 서브시스템 + adapter 통합 + smoke. 여러 환경 리스크가 있고, 이 코드가 처음으로 end-to-end 동작하는 시점.
- **M3b (다음 plan, M3a 종료 후 개시):** 1k+ 쌍 synth, 다중 epoch 학습, Orchestrator를 4개 모드(`base`, `rag`, `qlora`, `rag_qlora`)로 확장, Gradio 모드 토글. 실제 시간 + API 비용 소모; 파이프라인은 이미 검증된 상태.

분리하면 예상치 못한 smoke 실패가 절망적인 전체 학습 실행에 시간을 낭비시키지 않으며, API 비용은 단계적으로 집행된다(smoke는 $3 cap, 전체는 $10–20).

## 1. 목표

기존 M0 base-model 로더와 M1 RAG retriever를 새로운 QLoRA 학습 파이프라인에 연결하고, model_loader 싱글톤에 adapter attach/detach hook을 제공하며, 작은 합성 데이터셋으로 smoke adapter 하나를 학습한 뒤 그것이 로드되어 비어 있지 않은 한국어를 생성하는지 확인하여 전체 체인을 검증한다.

**M3a 범위 외 (M3b로 연기):**
- 1,000–3,000쌍 전체 synth
- 다중 epoch 학습, 전체 하이퍼파라미터 sweep
- `Orchestrator.generate(qlora|rag_qlora, ...)` 런타임 디스패치
- Gradio 4-모드 토글 UI

**확정적으로 연기 (M3 전체 범위 외):**
- Adapter 품질 평가 (M4 / `eval/runner.py`)
- Adapter v2/v3 비교 자동화 (M5 수동 처리로 충분)

## 2. 아키텍처

### 2.1 신규 / 변경되는 표면(surface)

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

### 2.2 Adapter 통합 설계

`src/serving/model_loader._State`는 이미 additive 형태를 노출한다(publish 순서를 잠근 Plan 2 리뷰 피드백 참조). M3a는 기존 API를 깨지 않고 확장한다:

```python
@dataclass
class _State:
    model: Any = None
    base_model: Any = None        # NEW — preserved reference for detach
    tokenizer: Any = None
    loader: str = ""
    adapter_path: str | None = None  # NEW — diagnostic + idempotency check
```

`attach_adapter(path)`: 현재 부착된 adapter가 없는지 assert(또는 먼저 detach); `_state.base_model = _state.base_model or _state.model`; `_state.model = PeftModel.from_pretrained(_state.base_model, str(path))`; `_state.adapter_path = str(path)`.

`detach_adapter()`: adapter가 없으면 noop; 그렇지 않으면 `_state.model = _state.base_model`; `_state.adapter_path = None`. (`PeftModel.unload()`도 동작할 수 있지만 swap 패턴이 더 투명하다.)

`get_base_model()` 시맨틱은 변하지 않는다. 싱글톤 불변식 — VRAM에 동시에 하나의 모델만 — 은 보존된다.

### 2.3 왜 M3a는 Orchestrator를 확장하지 않는가

smoke gate는 `attach_adapter` + `model.generate(...)`를 직접 검증한다(slow integration test에서). 같은 plan에서 `Orchestrator.generate`의 `Mode` literal에 `qlora`와 `rag_qlora`를 추가하고 prompt builder를 확장하며 Gradio까지 갱신하면 무관한 리스크들이 결합된다. M3b가 smoke adapter가 존재하면 하나의 집중된 plan으로 그것을 처리할 것이다.

## 3. Pre-flight: torch 2.5+ 생태계 업그레이드 (Plan task 0)

현재 핀인 `torch==2.4.*`는 Unsloth를 막는다(`AttributeError: torch._inductor has no attribute 'config'` — torchao≥0.13이 torch≥2.5를 요구하기 때문). Plan 1은 transformers 폴백으로 우회했다. Plan 3은 그렇게 할 수 없다 — Unsloth의 2배 속도 향상 + 메모리 절약이 3B-class QLoRA를 6 GB 카드에 맞추는 핵심이다.

### 3.1 Pin 업데이트

- `pyproject.toml`: `torch>=2.5,<2.7` (작업 시점에 조사; 상한은 다음으로 알려진 비호환 버전).
- `pyproject.toml`: `transformers>=4.44,<4.50` 재검토 — Unsloth의 현재 핀 범위는 `>=4.51.3,<=5.5.0`. 결론: `transformers>=4.51,<5.0`로 bump. FlagEmbedding `dtype=` patch는 여전히 적용되어야 하며; `tests/test_postinstall.py`가 회귀 게이트.
- `pyproject.toml`: `bitsandbytes>=0.43,<0.49` — 현재 0.45.5가 torch 2.5+에서 여전히 동작하는지 확인; 아니면 `<0.50`로 bump 후 재테스트.
- `Makefile`: `install-cuda-torch` URL: `cu124` → `cu126` (또는 torch 2.5+의 최신 CUDA wheel과 일치하는 것).

### 3.2 재검증 게이트 (다른 M3a 코드가 들어오기 전에 반드시 통과)

순서대로 실행하며 각각이 성공해야 한다:

1. `make install-dev && make install-cuda-torch && make postinstall`
2. `.venv/Scripts/python -c "from unsloth import FastLanguageModel; print('ok')"` — Unsloth import가 동작 (이전에 깨진 신호였음).
3. `.venv/Scripts/python scripts/verify_unsloth.py` — 이제 Unsloth 경로로 로드, 한국어 생성.
4. `.venv/Scripts/python scripts/verify_combined_vram.py` — 피크 < 5.5 GB.
5. `.venv/Scripts/python -m pytest -m "not slow" -q` — 모든 52개 fast 테스트 통과.
6. `.venv/Scripts/python -m pytest -m slow -v` — 모든 4개 slow 테스트 통과.

step 2가 실패하면(torch 2.5+에서 Unsloth가 여전히 깨져 있음) plan을 중단하고 사용자에게 옵션과 함께 에스컬레이션:
- (b1) 현재 torch와 known-good인 특정 Unsloth commit 시도
- (b2) M3a 나머지를 vanilla transformers + peft Trainer로 폴백 (느리지만 검증된 경로)

### 3.3 커밋 정책

Pre-flight는 단일 커밋(`build(deps): torch 2.5+ ecosystem upgrade for Unsloth-enabled QLoRA training`). smoke나 이후 어떤 작업이라도 되돌려야 한다면 이 커밋은 한 번의 `git revert`로 충분하다.

## 4. 합성 파이프라인 (`src/training/synth_qa.py`)

### 4.1 입력

- `data/processed/qa_seed.jsonl` — 12개의 손수 큐레이션된 `QAPair`(`source="seed"`). 구현자가 `chunks.jsonl`로부터 초안을 작성하고, synth 실행 전에 사용자가 검토. 토픽 커버리지: 임대차 (lease), 매매 (sale), 채권 (claims), 손해배상 (damages), 시효 (prescription) — 민법 기초.
- `data/processed/chunks.jsonl` — M1 출력.
- `config/prompts/synth_qa.txt` — system + few-shot framing. plan과 함께 제공.

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

### 4.3 배치별 흐름

1. `statute_no` 근접성으로 클러스터된 3–5개의 chunk 샘플링 (few-shot 답변이 여러 관련 조문을 그럴듯하게 인용할 수 있도록).
2. 프롬프트 포맷: system prompt + 3개 랜덤 seed 예시 + 샘플된 chunks, `[민법 제○○조]` 인용을 포함한 2개의 grounded Q&A pair 요청.
3. Anthropic SDK 호출 (`claude-sonnet-4-6` 기본; `claude-opus-4-7`은 `model="opus"`로).
4. 응답 파싱(fenced JSON 기대; 파싱 실패 시 로그 + 배치 skip).
5. `cache_dir / f"{batch_hash}.json"`에 raw 응답 캐시하여 rerun 시 API skip.

### 4.4 후처리 필터

`src.eval.citation_checker._STATUTE_RE`를 재사용하여 인용 존재를 강제한다. 다음의 경우 합성된 row를 drop한다:
- `output`에 statute regex 매치가 없음.
- `len(output) < 50` 글자.
- `(instruction, output)` 해시가 이미 emit된 row와 일치.

각 drop은 사유와 함께 INFO 레벨로 로깅; 최종 요약은 `kept N / synthesized M (drop reasons: …)`을 출력.

### 4.5 비용 게이트

- Anthropic SDK 응답은 `usage.input_tokens` + `usage.output_tokens`를 포함. 스크립트는 선택된 모델의 정적 가격표를 사용하여 누적 비용을 추적.
- 각 API 호출 전, `running_cost + estimated_call_cost > max_usd`이면 중단하고 지금까지 생성된 것을 기록. Exit code 0(부분 출력은 사용 가능).
- smoke 실행은 `max_usd=3.0`과 `target_pairs=200` 사용.

## 5. 데이터셋 준비 (`src/training/prepare_dataset.py`)

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

### 5.2 동작

- `qa_path`를 `QAPair` row 스트림으로 읽음.
- `tokenizer.apply_chat_template([{role:'user', content:instruction}, {role:'assistant', content:output}], tokenize=True, return_tensors=None)`을 사용하여 Qwen2.5 chat template 적용 — 전체 시퀀스 id를 얻음.
- Assistant-only 마스크 계산: assistant 응답 이전 토큰들은 `-100`으로 라벨링(cross-entropy에서 무시); assistant 토큰들은 자신의 id를 유지. 경계는 `add_generation_prompt=True`로 재토큰화하여 prompt prefix 길이를 찾는 방식으로 감지.
- `len(input_ids)`의 p50 / p95 / max 보고; `max_seq is None`이면 `ceil_to_64(p95)` 제안. 실제 학습 cap은 호출자가 설정.
- `max_seq` 초과 row는 truncate (assistant 측 truncation; instruction은 보존).
- `random.Random(seed).shuffle`을 통한 결정적 90/10 split.

## 6. 학습 (`src/training/train_qlora.py`)

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

### 6.2 하이퍼파라미터

MVP 설계 §4.3에 따르되 smoke 오버라이드:

| Param | Full (M3b default) | Smoke (M3a) |
|---|---|---|
| LoRA r | 16 | 16 |
| LoRA α | 32 | 32 |
| LoRA dropout | 0 | 0 |
| target_modules | q/k/v/o/gate/up/down_proj | same |
| gradient_checkpointing | True | True |
| max_seq | 2048 (OOM 시 자동 1024로 축소) | 1024 |
| batch_size | 1 | 1 |
| grad_accum | 8 | 8 |
| lr | 2e-4 | 2e-4 |
| warmup_ratio | 0.1 | 0.1 |
| weight_decay | 0.01 | 0.01 |
| optim | adamw_8bit | adamw_8bit |
| epochs | 2–3 | 1 |
| max_steps | (epoch-driven) | ≈25 |

### 6.3 로깅

- TensorBoard: `report_to=["tensorboard"]`; `logging_dir=runs/{name}/tb/`.
- `TrainerCallback`을 통한 커스텀 JSONL 로그: `runs/{name}/train_log.jsonl`에 logging step당 한 줄, `{step, loss, lr, epoch, time_ms}` 포함.
- 최종 adapter: `output_dir / "adapter_config.json"` + `adapter_model.safetensors`.

### 6.4 Resume

100 step마다 checkpoint(`save_steps=100`, `save_total_limit=2`). smoke 실행은 최대 한 개의 checkpoint를 남기며; 전체 M3b 실행은 이를 resume에 사용.

## 7. Smoke gate (`make smoke-train` 종료 시)

`train(smoke=True)` 반환 후:

1. Adapter 디렉터리가 존재하며 `adapter_config.json`과 `adapter_model.safetensors` 둘 다 포함.
2. `train_log.jsonl`에 ≥2 step row 포함.
3. **Loss 정상성:** `final_loss < first_loss * 0.95` (≥5% 감소). 발산 시 hard fail; 감소가 0%~5% 사이면 soft warn (PASS with concern).
4. **Adapter 통합 smoke:** 새 subprocess에서 `from src.serving.model_loader import get_base_model, attach_adapter; m, t = get_base_model(); attach_adapter(adapter_dir); out = m.generate(t.encode("임대차의 의의는?", return_tensors="pt").to(m.device), max_new_tokens=32); print(t.decode(out[0]))` — 비어 있지 않고 한국어 문자를 포함함을 assert.

Pass → exit 0, 한 줄 요약 출력, M3b 권고. Fail → exit 1, 요약은 어떤 게이트가 실패했는지 명시, M3b plan 미개시.

## 8. 테스트 계약 (TDD 적용 가능 표면을 **굵게**)

| File | Tests | Type |
|---|---|---|
| **`synth_qa.py` post-processing** | citation filter, length filter, dedupe, cost-cap abort | unit (mock SDK) |
| **`prepare_dataset.py`** | chat-template tokenization, label mask of instruction tokens, 90/10 split determinism, max_seq truncation, p95 token-length report | unit (uses real tokenizer; fast — no model load) |
| **`model_loader.attach_adapter / detach_adapter`** | argument coercion (str vs Path), idempotency error on double-attach, current_adapter() roundtrip | unit (no GPU; use a stub `_state.model` mock) |
| `train_qlora.py` | not unit-tested (library wrapper); covered by `test_smoke_train.py` | — |
| `test_smoke_train.py` | full pipeline: mocked Claude → real prepare_dataset → real Unsloth train (`max_steps=2, target_pairs=5`) → real attach_adapter → one model.generate. CUDA-gated. | slow (real GPU) |

M3a 종료 시점 카운트 목표: ~62 fast (이전 52 + 신규 ~10 unit) + 5 slow (이전 4 + 1 smoke). Fast suite는 ~5초 미만 유지.

## 9. 수락 기준 (= M3a "완료")

1. Pre-flight 커밋(§3) 랜딩; 6개 재검증 단계 모두 green.
2. `make synth`가 `MAX_API_USD=3` cap 내에서 `data/processed/qa_train.jsonl`에 ≥150쌍 생성.
3. `make prepare`가 90/10 split으로 `qa_train_tokenized.jsonl`과 `qa_val_tokenized.jsonl`을 기록.
4. `make smoke-train`이 smoke adapter를 생성하고, `final_loss < first_loss * 0.95`로 PASS 출력.
5. `model_loader.attach_adapter(smoke_dir)` 후 `model.generate(...)`이 비어 있지 않은 한국어 텍스트 반환.
6. 모든 fast 테스트 green; slow suite green.
7. `verify_combined_vram.py` 피크가 torch 업그레이드 후에도 여전히 < 5.5 GB.

## 10. 리스크 & 완화

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| torch 2.5+에서 Unsloth가 여전히 깨짐 | Low | High | §3.2 step 2가 포착; peft + transformers Trainer로 폴백 (브레인스토밍의 옵션 (b)) |
| Unsloth에도 불구하고 smoke에서 6 GB OOM | Medium | High | max_seq=1024, batch=1 + grad_accum=8, gradient_checkpointing. OOM 지속 시: r=8 → r=4 → 최후의 폴백 Qwen2.5-1.5B |
| 업그레이드 후 FlagEmbedding patch가 깨짐 | Medium | Medium | `tests/test_postinstall.py` regex 회귀 테스트; `make postinstall`로 재적용 |
| Synth API 비용 초과 | Low | Low | $3 cap; 배치별 비용 로그; 캐시된 응답은 resume 시 skip |
| Smoke를 위한 synth 품질이 너무 낮음 | Medium | Low | smoke는 품질이 아니라 plumbing을 점검; loss 정상성 ≥5%가 큰 발산을 커버 |
| Smoke loss 감소 <5% (genuine model issue) | Medium | Medium | Soft warn → PASS with concern; 사용자가 M3b로 진행할지 prompt를 iterate할지 결정 |
| `attach_adapter`가 이전 base model을 누수 | Low | Medium | `_state.base_model` 참조 + 명시적 detach swap; hot path에 `del` 없음 |

## 11. M3b 미리보기 (NOT this plan)

M3a 종료 시 `docs/superpowers/specs/<date>-m3b-full-training-and-orchestrator-design.md` 작성:

- 전체 synth: 1,000–1,500쌍 (`synth_qa.run_synth(target_pairs=1500, max_usd=15)` 재사용)
- 전체 학습: 2 epoch, max_seq=2048, adapter `qwen2.5-3b-civil-v1/` 저장
- `Orchestrator.generate` 확장: `Mode = Literal["base","rag","qlora","rag_qlora"]`; 모드 디스패치는 필요에 따라 adapter를 attach/detach
- Gradio 4-모드 토글: 라디오 버튼 + 4-컬럼 비교 또는 2x2 그리드
- 4개 모드 모두를 행사하는 새로운 slow integration test
