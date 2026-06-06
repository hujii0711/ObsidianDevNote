**Goal:** 주택임대차 보증금 반환 RAG 챗봇의 품질을 정량 측정하는 평가 하니스를 만들고, 현재 RAG 베이스라인(파인튜닝 없음)의 점수를 산출한다. 이 점수가 Plan 3C에서 QLoRA 어댑터와 A/B 비교할 기준선이 된다.

**Architecture:** uv 워크스페이스에 `packages/eval` 추가. `apps/api`의 `run_chat`(서빙과 동일 코드 → 평가 일관성)과 `packages/rag`의 retriever를 재사용한다. 지표는 ① **검색 recall@k**(기대 법조항이 top-k에 검색됐는가, 결정적) ② **답변 형식**(인용 `[n]`·상담형 구조·면책·출처 유무, 결정적) ③ **키워드 충실도**(기대 키워드 포함, 결정적) ④ **LLM-as-judge 충실도/말투**(주입 가능, 선택). retriever·llm·judge는 주입 가능하게 해 모델 없이 단위 테스트한다.

**Tech Stack:** Python 3.11, uv(workspace), pytest. 라이브 실행 시 `apps/api`의 MlxLLM(Qwen2.5-7B) + `data/chroma`(Plan 1 인덱스) + bge-m3.

```
packages/eval/
├── pyproject.toml                 # eval 패키지 (rag + api workspace 의존)
├── src/eval/
│   ├── __init__.py
│   ├── dataset.py                 # EvalItem 스키마 + jsonl 로더
│   ├── retrieval_metrics.py       # recall@k / hit@k (결정적)
│   ├── answer_metrics.py          # 형식·키워드 지표 (결정적)
│   ├── judge.py                   # LLM-as-judge (주입 가능)
│   ├── runner.py                  # 평가셋 → run_chat → per-item 결과
│   ├── report.py                  # 집계 → 요약 지표
│   └── cli.py                     # 라이브 실행 엔트리
├── eval_set.jsonl                 # 큐레이션된 평가셋(그라운드 트루스, 커밋됨)
└── tests/
    ├── conftest.py
    ├── test_dataset.py
    ├── test_retrieval_metrics.py
    ├── test_answer_metrics.py
    ├── test_judge.py
    ├── test_runner.py
    └── test_report.py
```

Task 1: EvalItem 스키마 + 로더
**Files:**
- Create: `packages/eval/src/eval/dataset.py`
- Test: `packages/eval/tests/test_dataset.py`

Task 2: 큐레이션된 평가셋 (그라운드 트루스)
**Files:**
- Create: `packages/eval/eval_set.jsonl`
- Test: `packages/eval/tests/test_eval_set_valid.py`
주택임대차보호법 보증금 반환 관련 대표 질문 16문항. `expected_refs`는 Plan 1이 색인한 실제 조문 ref(주임법 조문번호)와 일치해야 한다.

Task 3: 검색 지표 (recall@k / hit@k)
**Files:**
- Create: `packages/eval/src/eval/retrieval_metrics.py`
- Test: `packages/eval/tests/test_retrieval_metrics.py`

Task 4: 답변 형식·키워드 지표
**Files:**
- Create: `packages/eval/src/eval/answer_metrics.py`
- Test: `packages/eval/tests/test_answer_metrics.py`
베이스라인 연구 가설(인용·구조·면책·키워드 준수율)을 결정적으로 측정한다.

Task 5: LLM-as-judge 충실도 (주입 가능)
**Files:**
- Create: `packages/eval/src/eval/judge.py`
- Test: `packages/eval/tests/test_judge.py`

답변이 검색 근거에 기반했는지(groundedness)를 LLM judge로 0~1 점수화. judge 호출은 주입 가능(테스트는 가짜 judge).

Task 6: 평가 러너
**Files:**
- Create: `packages/eval/src/eval/runner.py`
- Test: `packages/eval/tests/test_runner.py`
각 평가 항목을 검색 + `run_chat`(서빙과 동일 코드)으로 돌려 항목별 결과를 만든다. retriever·llm·judge_fn 주입.

Task 7: 리포트 집계 + CLI
**Files:**
- Create: `packages/eval/src/eval/report.py`
- Create: `packages/eval/src/eval/cli.py`
- Test: `packages/eval/tests/test_report.py`

Task 8: 라이브 베이스라인 측정
**Files:** 없음(측정 실행). 결과는 `data/eval_runs/baseline.json`(gitignore).
실제 Qwen2.5-7B + 실제 Chroma 인덱스로 평가셋 16문항을 돌려 베이스라인 점수를 산출한다.