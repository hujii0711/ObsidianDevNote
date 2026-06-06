
**Goal:** QLoRA 학습용 데이터셋을 합성한다. 평가셋과 겹치지 않는 보증금 반환 질문 풀에 대해, 실제 RAG 파이프라인으로 후보 답변을 다수 생성하고, **평가의 형식 지표를 필터로 재사용**해 상담형 구조·`[n]` 인용·면책을 모두 갖춘 답변만 채택해 MLX chat JSONL(train/valid)로 만든다.

**Architecture:** uv 워크스페이스에 `packages/ftdata` 추가(`rag`·`api`·`eval` 재사용). 질문 풀(평가셋과 disjoint, 커밋) → `api.pipeline.run_chat`로 K개 후보 생성(temp 다양성, 서빙=학습데이터 동일 입력) → `eval.answer_metrics` 형식 필터 + `eval.judge` 근거 점수로 best 채택 → `rag.prompt.build_messages`로 학습 입력 재구성(system+user(근거)) + assistant(답변) → JSONL. 순수 모듈(필터·빌더·분할)은 모델 없이 단위 테스트, 생성·CLI는 라이브.

**Tech Stack:** Python 3.11, uv(workspace), pytest. 라이브: MlxLLM(Qwen2.5-7B) + `data/chroma`.

```
packages/ftdata/
├── pyproject.toml                 # deps: rag, api, eval
├── src/ftdata/
│   ├── __init__.py
│   ├── questions.py               # 질문 풀 로더 (eval set과 disjoint)
│   ├── filter.py                  # format_ok (eval.answer_metrics 재사용)
│   ├── builder.py                 # to_chat_example(build_messages 재사용) + train/valid 분할 + write
│   ├── generate.py                # Candidate + generate_candidates (run_chat 재사용)
│   └── cli.py                     # 라이브 빌드 엔트리
├── question_pool.jsonl            # 시드 질문 풀 30 (커밋, eval set과 disjoint)
└── tests/
    ├── conftest.py
    ├── test_questions.py
    ├── test_filter.py
    ├── test_builder.py
    └── test_generate.py
```


Task 1: 질문 풀 (평가셋과 disjoint)
**Files:**
- Create: `packages/ftdata/question_pool.jsonl`
- Create: `packages/ftdata/src/ftdata/questions.py`
- Test: `packages/ftdata/tests/test_questions.py`
보증금 반환 관련 질문 30개. **평가셋 16문항과 질문 문자열이 하나도 겹치지 않아야** 한다(오염 방지).

Task 2: 형식 필터 (eval 지표 재사용)
**Files:**
- Create: `packages/ftdata/src/ftdata/filter.py`
- Test: `packages/ftdata/tests/test_filter.py`
학습 데이터로 채택할 후보는 **상담형 구조·`[n]` 인용·면책·출처를 모두** 갖춰야 한다. 평가의 `answer_metrics`를 그대로 필터로 재사용(평가=학습 데이터 품질 기준 일치).

Task 3: chat-example 빌더 + train/valid 분할
**Files:**
- Create: `packages/ftdata/src/ftdata/builder.py`
- Test: `packages/ftdata/tests/test_builder.py`
채택된 (질문, 근거, 답변)을 **서빙과 동일한 입력 형식**(`build_messages`: system+user(근거))에 assistant(답변)를 붙여 MLX chat 예제로 만들고, 결정적으로 train/valid를 분할해 jsonl로 쓴다.

Task 4: 후보 생성기
**Files:**
- Create: `packages/ftdata/src/ftdata/generate.py`
- Test: `packages/ftdata/tests/test_generate.py`
질문마다 검색 근거를 한 번 잡고, `run_chat`(서빙과 동일)으로 K개 후보를 temp 다양성으로 생성하며 각 후보의 근거 점수를 매긴다. retriever·llm·judge_fn 주입.

Task 5: 빌드 CLI
**Files:**
- Create: `packages/ftdata/src/ftdata/cli.py`
- (테스트 없음 — 라이브 오케스트레이션. 순수 로직은 Task 2~4에서 검증됨)