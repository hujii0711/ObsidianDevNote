
**Goal:** Plan 1이 구축한 Chroma 인덱스(`jeonse_deposit`) 위에, 질의를 검색해 근거를 주입하고 MLX Qwen2.5-7B로 상담형 답변(`[n]` 인용 포함)을 스트리밍 생성하는 FastAPI `/chat` 엔드포인트를 만든다. `curl`로 데모 가능한 RAG 베이스라인 챗 API가 완성 산출물

**Architecture:** uv 워크스페이스 모노레포. `packages/rag`(검색·프롬프트 조립·인용 매핑 — Plan 3 평가와 공유)와 `apps/api`(FastAPI + MLX 서빙)로 분리. 데이터 흐름: `질의 → rag.retrieve(bge-m3+Chroma) → rag.build_prompt → api.llm.stream(MLX) → rag.map_citations → SSE(answer 토큰 + sources)`. LLM과 임베더는 주입 가능하게 설계해 네트워크/GPU 없이 단위 테스트한다.

**Tech Stack:** Python 3.11, uv (workspace), FastAPI, uvicorn, sse-starlette, pydantic, chromadb, sentence-transformers(bge-m3), mlx-lm(Qwen2.5-7B-Instruct-4bit), pytest, httpx(TestClient).

```
privateLLM/
├── pyproject.toml                  # [신규] uv 워크스페이스 루트 (members: pipelines, packages/*, apps/*)
├── packages/
│   └── rag/
│       ├── pyproject.toml          # rag 패키지
│       ├── src/rag/
│       │   ├── __init__.py
│       │   ├── config.py           # RagConfig (chroma_dir/collection/model/k/threshold) — OC 불필요
│       │   ├── types.py            # Retrieved, Source 데이터클래스
│       │   ├── embedder.py         # bge-m3 질의 임베더 (encode_fn 주입)
│       │   ├── retriever.py        # Chroma top-k → list[Retrieved] + grounding 판정
│       │   ├── prompt.py           # 시스템 프롬프트 + 근거 번호 매김 + chat messages 조립
│       │   └── citations.py        # 답변의 [n] 파싱·검증 → Source 매핑(환각 인용 제거)
│       └── tests/
│           ├── conftest.py
│           ├── test_config.py
│           ├── test_retriever.py
│           ├── test_prompt.py
│           └── test_citations.py
└── apps/
    └── api/
        ├── pyproject.toml          # api 패키지 (rag 경로 의존)
        ├── src/api/
        │   ├── __init__.py
        │   ├── settings.py         # 환경 설정 (모델명, chroma 경로, 생성 파라미터)
        │   ├── schemas.py          # pydantic ChatRequest/Source/...
        │   ├── llm.py              # MLX Qwen 로더 + stream 생성 (LLM 프로토콜, 주입 가능)
        │   ├── pipeline.py         # RAG 오케스트레이션 (retrieve→prompt→generate→cite)
        │   └── main.py             # FastAPI app: POST /chat (SSE), GET /health
        └── tests/
            ├── conftest.py
            ├── test_pipeline.py
            └── test_chat_endpoint.py
```

루트 워크스페이스로 묶으면 `uv run --package rag pytest` / `uv run --package api ...` 형태로 실행. 기존 `pipelines`는 워크스페이스 멤버로 편입하되 코드 변경은 없다.


Task 1: RagConfig + 상수
**Files:**
- Create: `packages/rag/src/rag/config.py`
- Test: `packages/rag/tests/test_config.py`
Plan 1과 **반드시 동일한** 임베딩 모델·컬렉션·cosine 규약을 상수로 고정한다(불일치 시 검색이 깨짐).

Task 2: 타입 + 질의 임베더
**Files:**
- Create: `packages/rag/src/rag/types.py`
- Create: `packages/rag/src/rag/embedder.py`
- Test: `packages/rag/tests/test_embedder.py`

Task 3: Retriever (Chroma top-k + grounding 판정)
**Files:**
- Create: `packages/rag/src/rag/retriever.py`
- Test: `packages/rag/tests/test_retriever.py`

Task 4: 프롬프트 빌더 (상담형 + 번호 매긴 근거)
**Files:**
- Create: `packages/rag/src/rag/prompt.py`
- Test: `packages/rag/tests/test_prompt.py`

Task 5: 인용 매핑 (환각 인용 제거)
**Files:**
- Create: `packages/rag/src/rag/citations.py`
- Test: `packages/rag/tests/test_citations.py`
생성된 답변에서 `[n]`을 추출해 실제 검색 근거와 대조한다. 범위를 벗어난 인용(환각)은 제거하고, 실제 인용된 근거만 `Source` 리스트로 반환한다.

Task 6: api 패키지 스캐폴딩 + 설정 + 스키마
**Files:**
- Create: `apps/api/pyproject.toml`
- Create: `apps/api/src/api/__init__.py` (빈 파일)
- Create: `apps/api/src/api/settings.py`
- Create: `apps/api/src/api/schemas.py`
- Create: `apps/api/tests/conftest.py`
- Test: `apps/api/tests/test_schemas.py`

Task 7: LLM 서비스 (MLX Qwen, 주입 가능 스트리밍)
**Files:**
- Create: `apps/api/src/api/llm.py`
- Test: `apps/api/tests/test_llm.py`
`LLM` 프로토콜은 `stream(messages) -> Iterator[str]`. 실제 구현 `MlxLLM`은 mlx-lm으로 토큰을 스트리밍한다. 테스트는 가짜 LLM으로 한다(모델 불필요).

Task 8: RAG 파이프라인 오케스트레이션
**Files:**
- Create: `apps/api/src/api/pipeline.py`
- Test: `apps/api/tests/test_pipeline.py`
`run_chat`은 retrieve → (grounding 판정) → prompt → LLM stream → 누적된 답변에서 인용 정리. 답변 토큰을 스트리밍하고, 마지막에 정리된 `sources`를 만든다. retriever와 llm을 주입받아 테스트한다.

Task 9: FastAPI `/chat` (SSE) + `/health`
**Files:**
- Create: `apps/api/src/api/main.py`
- Test: `apps/api/tests/test_chat_endpoint.py`
`/chat`은 SSE로 토큰을 흘리고 마지막에 `done`(answer+sources) 이벤트를 보낸다. 테스트는 의존성 주입(앱 state의 retriever/llm을 가짜로 교체)으로 모델 없이 한다.

Task 10: CORS + 도메인 가드 마무리 + 라이브 스모크
**Files:**
- Modify: `apps/api/src/api/main.py` (CORS 미들웨어 추가)
- Test: `apps/api/tests/test_cors.py`
Next.js(Plan 2B)가 브라우저에서 호출하므로 CORS가 필요하다. 도메인 밖 질문은 이미 grounding 판정(Task 8)으로 자연 차단되므로 별도 분류기는 YAGNI.