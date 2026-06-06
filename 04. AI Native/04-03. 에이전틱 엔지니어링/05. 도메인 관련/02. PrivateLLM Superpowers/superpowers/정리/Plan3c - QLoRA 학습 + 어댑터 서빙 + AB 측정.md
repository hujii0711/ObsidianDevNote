
**Goal:** Plan 3B의 학습 데이터(`data/ft/`)로 QLoRA 어댑터를 MLX로 학습하고, 어댑터를 서빙에 적용해 동일 평가셋으로 재측정한 뒤 베이스라인과 **A/B 비교**하여 파인튜닝의 정량적 기여도를 산출한다. **전체 연구의 결론.**

**Architecture:** uv 워크스페이스에 `packages/finetune` 추가(`api`·`eval` 재사용). 학습은 `mlx_lm.lora`(MLX 네이티브 QLoRA) CLI. `api.llm.MlxLLM`에 `adapter_path` 인자 추가(`mlx_lm.load(model, adapter_path=)`). `eval.cli`에 `--adapter` 추가(생성은 어댑터, **judge는 base 유지** → arm 간 judge 일치). 동일 평가셋(16문항, 학습 질문과 disjoint)으로 baseline·qlora를 같은 세션에서 측정 → `finetune.compare`로 델타 산출.

**Tech Stack:** Python 3.11, uv(workspace), pytest. 라이브: mlx_lm.lora(학습) + MlxLLM(어댑터 추론) + Chroma.

```
packages/finetune/
├── pyproject.toml                 # deps: api, eval
├── src/finetune/
│   ├── __init__.py
│   ├── train.py                   # build_lora_command (순수, TDD)
│   ├── compare.py                 # load_summary / compare_runs / to_markdown (순수, TDD)
│   └── compare_cli.py             # baseline.json vs qlora.json → 결론 문서
└── tests/
    ├── conftest.py
    ├── test_train.py
    └── test_compare.py
```

Task 1: MlxLLM 어댑터 지원
**Files:**
- Modify: `apps/api/src/api/llm.py`
- Test: `apps/api/tests/test_llm_adapter.py`
`MlxLLM(__init__)`에 `adapter_path: str | None = None`를 추가하고 `mlx_lm.load(model_name, adapter_path=adapter_path)`로 로드한다. 기존 동작(어댑터 없음)은 그대로.

Task 2: eval.cli `--adapter` (생성=어댑터, judge=base)
**Files:**
- Modify: `packages/eval/src/eval/cli.py`
A/B의 qlora arm을 위해 생성 LLM에 어댑터를 적용하되, **judge는 base 유지**(arm 간 judge 일치 → 공정 비교). 결정적 지표는 judge와 무관하므로 주 신호는 그대로 비교 가능.

Task 3: 학습 커맨드 빌더
**Files:**
- Create: `packages/finetune/src/finetune/train.py`
- Test: `packages/finetune/tests/test_train.py`
`mlx_lm.lora` 학습 명령을 순수 함수로 구성(라이브 실행은 Task 5에서 subprocess). 플래그는 mlx-lm 0.31.3 기준(`--num-layers`/`--learning-rate`/`--adapter-path`).

 Task 4: A/B 비교 모듈
**Files:**
- Create: `packages/finetune/src/finetune/compare.py`
- Test: `packages/finetune/tests/test_compare.py`
두 평가 리포트(baseline.json / qlora.json)의 요약 지표를 읽어 지표별 델타를 내고 마크다운 표로 만든다.

Task 5: 라이브 — 학습 + A/B 측정 + 결론
**Files:** 없음(라이브 실행). 산출물 `data/adapters/`·`data/eval_runs/`(gitignore). 결론은 `docs/superpowers/notes/`에 커밋.