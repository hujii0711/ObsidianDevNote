● ✅ Plan 1 + Documentation 정비 완료

  최종 커밋: e1927c6 (docs: enrich README + add M2 readiness checklist)

  추가된 내용

  README.md (81% rewrite)

  - Prerequisites — Python 3.12 winget 설치 명령 포함
  - Quickstart — fresh clone에서 부터의 7단계 (venv → install → CUDA torch → tests → smoke → FlagEmbedding patch)
  - Make targets 표 — 11개 타겟 모두 설명
  - Layout — 실제 파일 구조 명세
  - Running tests — fast/slow/E2E 분리 가이드
  - Try the retriever — 실행 예시
  - Known Issues — 6개 항목 (FlagEmbedding patch, Unsloth defer, bnb 0.49, pyarrow 24, Chroma Settings, make clean)

  docs/superpowers/m2-readiness.md (new)

  - Plan 1 Summary — 모듈 / 테스트 / acceptance signals
  - Empirical adaptations — plan 외 실증 변경사항 6개
  - Open issues from final review — 4개 Important + 6개 Minor
  - Required before Plan 2 checklist — A/B/C/D 4개 체크박스
  - Plan 2 preview + suggested first 3 tasks

  ---
  전체 Plan 1 최종 상태

  ┌──────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
  │    메트릭    │                                            값                                             │
  ├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ 커밋 수 (총) │ 17 (spec + plan + 14 tasks + 폴리싱)                                                      │
  ├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ 태그         │ m0-bootstrap, m1-rag-baseline                                                             │
  ├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ 테스트       │ 29 passed (27 fast + 2 slow)                                                              │
  ├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ 파일 추가    │ src/common (5), src/ingestion (3), src/rag (5), tests (6), scripts (1), docs (2)          │
  ├──────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ 의존성 핀    │ torch 2.4.1+cu124, transformers<4.50, bnb<0.49, pyarrow<24, FlagEmbedding 1.4.0 (patched) │
  └──────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘

  M2 진입 전 권장 사항 (m2-readiness.md에서)

  체크박스 4개 (Important):
  - A FlagEmbedding 패치 영구 해결
  - B 결합 VRAM (bge-m3 + Qwen2.5-3B) 검증 스크립트
  - C _make_chroma_client() 헬퍼 추출
  - D Retriever dense (id, score) → list[str] + RRF 공식 값 테스트

  이들을 처리하면 Plan 2 (M2 — Orchestrator + Gradio UI) brainstorming → plan → 실행으로 진입할 준비가 됩니다.

  ---
  이 세션의 작업은 여기서 자연스러운 마무리 지점입니다. 다음 세션에서:
  1. 위 A/B/C/D를 작은 fix 사이클로 처리 후
  2. superpowers:brainstorming으로 M2 설계 시작 (또는 같은 spec의 §4.4 Serving 섹션을 그대로 사용해 바로
  superpowers:writing-plans)
  3. Plan 2 실행 → M2 acceptance gate

  다른 질문/요청 있으시면 알려주세요.

✻ Worked for 2m 50s

> 현재 구성된 프로젝트에서 QLoRA 파인튜닝하는 모듈도 있니?

● Bash(cd "C:/ClaudeProject/PrivateLLM" && ls -la src/training/ src/eval/ src/serving/ 2>&1 && echo "---" && grep -A2
       "^train:" Makefile 2>&1)
  ⎿  src/eval/:
     total 4
     drwxr-xr-x 1 hujii 197609 0 May  2 22:55 .
     … +14 lines (ctrl+o to expand)

● 아직 없습니다 — src/training/은 빈 디렉토리

  현재 구성 상태:

  ┌────────────────┬─────────────────────────────────────────────┐
  │    디렉토리    │                    상태                     │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/common/    │ ✅ 구현됨 (schemas, paths, config, logging) │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/ingestion/ │ ✅ 구현됨 (fetch_statutes, chunk)           │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/rag/       │ ✅ 구현됨 (bm25, embed, index, retriever)   │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/training/  │ 🟡 빈 디렉토리 (Plan 3 = M3 예정)           │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/serving/   │ 🟡 빈 디렉토리 (Plan 2 = M2 예정)           │
  ├────────────────┼─────────────────────────────────────────────┤
  │ src/eval/      │ 🟡 빈 디렉토리 (Plan 4 = M4 예정)           │
  └────────────────┴─────────────────────────────────────────────┘

  Makefile에 make train 타겟은 placeholder로 있지만 실행하면 python -m src.training.train_qlora 시도 →
  ModuleNotFoundError.

  왜 아직 없는지

  처음 brainstorming 단계에서 사용자가 (a) 기술 검증형 PoC를 선택하고, MVP를 점진 슬라이스로 진행하기로 합의했습니다.
   그래서 5개 마일스톤으로 분할:

<font color="#ff0000">  - **M0+M1 (Plan 1, 완료) — 프로젝트 기반 + RAG 검색기 동작**</font>
<font color="#ff0000">  - **M2 (Plan 2, 다음) — Orchestrator + Gradio UI (base / RAG 모드)**</font>
<font color="#ff0000">  - **M3 (Plan 3) — QLoRA 파인튜닝 파이프라인 ← 이 부분**</font>
<font color="#ff0000">  - **M4 (Plan 4) — 4-way ablation 평가**</font>

  QLoRA 학습이 들어갈 위치 (spec §4.3 기준)

  Plan 3에서 src/training/에 다음 파일들이 추가될 예정:

  src/training/
  ├── __init__.py
  ├── prepare_dataset.py    # qa_seed → train/val split + chat template
  └── train_qlora.py        # Unsloth FastLanguageModel + LoRA r=16/α=32 + gradient checkpointing

  학습 결과는 models/adapters/qwen2.5-3b-civil-v{N}/에 저장 (어댑터 가중치 + tokenizer + training_config.yaml).

  QLoRA 학습으로 넘어가기 전 해결해야 할 이슈

  m2-readiness.md에 이미 기록된 대로:

  1. Unsloth 호환성 문제 — Unsloth 2026.4가 torch ≥ 2.5를 요구하는데 프로젝트는 torch 2.4.1+cu124에 핀. M3 진입 시
  다음 셋 중 하나 결정 필요:
    - torch를 2.5/2.6으로 업그레이드 (다른 deps 호환성 재검증 필요)
    - Unsloth 없이 vanilla peft + bitsandbytes로 QLoRA 직접 구현 (느리지만 간단)
    - WSL2 + Linux 환경으로 전환 (Unsloth 공식 지원)
  2. 합성 Q&A 데이터 — QLoRA 학습은 instruction-completion 쌍이 필요합니다. spec에서는 Claude API로 1,000~3,000쌍
  합성 예정 (src/ingestion/synth_qa.py도 아직 없음, M3에서 추가).

  진행 옵션

  - (A) Plan 2 (M2 Gradio + 오케스트레이터)부터 순차 진행 — 권장. RAG 베이스라인이 UI에서 동작하는 시연을 먼저 확보
  후 M3.
  - (B) M2를 건너뛰고 Plan 3 (M3 QLoRA)부터 진행 — 가능하지만 평가 시각화/오케스트레이터 없이 학습만 돌리게 됨.
  - (C) M3을 위해 새 spec/plan 작성 시작 — superpowers:brainstorming으로 시작 (M3에 한정한 세부 설계 — Unsloth vs
  vanilla 결정 등).

  어느 쪽으로 진행하시겠습니까?

✻ Baked for 32s

> 현재 상태 기억해주면 내일 다시 이어서 진행가능한가?

● 네, 메모리 시스템으로 저장해두면 내일 이어서 진행 가능합니다. 핵심 상태와 다음 단계를 메모리에 기록합니다.

  Listed 1 directory (ctrl+o to expand)

● Bash(mkdir -p "C:/Users/hujii/.claude/projects/C--ClaudeProject-PrivateLLM/memory" && ls -la
      "C:/Users/hujii/.claude/projects/C--ClaudeProject-PrivateLLM/memory/")
  ⎿  total 4
     drwxr-xr-x 1 hujii 197609 0 May  3 01:44 .
     drwxr-xr-x 1 hujii 197609 0 May  3 01:44 ..

  Wrote 5 memories (ctrl+o to expand)

● ✅ 메모리 저장 완료

  내일 새 세션에서도 이 프로젝트 작업을 이어갈 수 있습니다.

  저장된 메모리 4개

  ┌───────────┬─────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
  │   유형    │              파일               │                            내용 요약                            │
  ├───────────┼─────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ project   │ project_status.md               │ M0+M1 완료, M2 다음, 5개 마일스톤 구조                          │
  ├───────────┼─────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ project   │ project_env_pins.md             │ torch 2.4 / transformers <4.50 / bnb <0.49 / pyarrow <24 /      │
  │           │                                 │ FlagEmbedding venv patch / Unsloth 보류 사유                    │
  ├───────────┼─────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ reference │ reference_resume_doc.md         │ 새 세션 시작 시 docs/superpowers/m2-readiness.md 먼저 읽기      │
  ├───────────┼─────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ feedback  │ feedback_collaboration_style.md │ 한국어 응답, 단답 선택, β-모드 subagent 진행, 실용적 적응 OK    │
  └───────────┴─────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

  내일 새 세션에서

  다음과 같이 시작하시면 이어집니다:

  "Legal PrivateLLM 프로젝트 이어서 하자"

  또는 더 구체적으로:

  "M2 진입 준비 작업 시작 — m2-readiness의 A부터"

  저는 새 세션에서 자동으로 메모리를 읽고:
  1. docs/superpowers/m2-readiness.md를 먼저 확인 (resume document)
  2. 4개 Important 이슈 (A: FlagEmbedding patch, B: 결합 VRAM 검증, C: ChromaDB 팩토리, D: retriever 정리)부터 처리
  3. 한국어로 응답, 단답 선택지로 의사결정 요청, β 모드로 진행

  프로젝트 내부 문서도 함께 보존되므로 (git committed m2-readiness.md, README), 메모리가 stale해져도 정확한 현재
  상태로 회복 가능합니다.

  좋은 휴식 되세요. 내일 이어서 진행하겠습니다.