# 학습 가이드 05: Python 비동기 / FastAPI / Streamlit

> 이 프로젝트의 서빙 레이어를 구현하기 위해 필요한 웹 기술 학습 가이드

---

## 1. Python 비동기 프로그래밍 (async/await)

### 1.1 왜 알아야 하는가?

```
동기 방식 (일반 Python):
  사용자 A 요청 → [모델 추론 30초] → 응답
  사용자 B 요청 → [A 끝날 때까지 대기...] → [모델 추론 30초] → 응답
  → B는 A의 추론이 끝날 때까지 60초 대기

비동기 방식 (이 프로젝트):
  사용자 A 요청 → [모델 추론 30초] → 응답
  사용자 B 요청 → [RAG 검색 1초] → [큐 대기] → [모델 추론 30초] → 응답
  → I/O 작업(DB, 벡터검색)은 병렬로 처리, CPU 작업(추론)만 순차
```

### 1.2 핵심 문법

```python
import asyncio

# 비동기 함수 정의
async def fetch_legal_docs(query: str) -> list[str]:
    # await으로 비동기 작업 대기
    results = await vector_db.search(query)
    return results

# 동기 함수를 비동기로 실행 (CPU 바운드)
async def run_inference(prompt: str) -> str:
    # asyncio.to_thread: 별도 스레드에서 실행
    result = await asyncio.to_thread(model.generate, prompt)
    return result

# 병렬 실행
async def process_request(query: str):
    # 두 작업을 동시에 실행
    docs, history = await asyncio.gather(
        fetch_legal_docs(query),
        get_chat_history(session_id),
    )
    return docs, history
```

### 1.3 이 프로젝트에서의 async 패턴

```python
# FastAPI 라우터 → 반드시 async def
@router.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    # RAG 검색 (I/O → 비동기)
    results = await retriever.search(request.message)
    
    # LLM 추론 (CPU → to_thread)
    answer = await asyncio.to_thread(llm_engine.generate, prompt)
    
    # DB 저장 (I/O → 비동기)
    await db.save_message(session_id, "assistant", answer)
    
    return ChatResponse(answer=answer)
```

### 학습 자료

```
├── Python 공식 asyncio 문서
│   → https://docs.python.org/3/library/asyncio.html
│
├── "Async IO in Python: A Complete Walkthrough" (Real Python)
│   → https://realpython.com/async-io-python/
│
└── "Python Concurrency" (asyncio vs threading vs multiprocessing)
    → 언제 어떤 동시성 패턴을 쓸지 이해
```

### 실습 과제

```python
import asyncio
import time

async def slow_io_task(name: str, seconds: float) -> str:
    """I/O 바운드 작업 시뮬레이션."""
    await asyncio.sleep(seconds)
    return f"{name} 완료 ({seconds}초)"

async def main():
    start = time.time()
    
    # 순차 실행 (3초)
    r1 = await slow_io_task("벡터검색", 1.0)
    r2 = await slow_io_task("DB조회", 1.0)
    r3 = await slow_io_task("리랭킹", 1.0)
    print(f"순차: {time.time() - start:.1f}초")
    
    start = time.time()
    
    # 병렬 실행 (1초)
    r1, r2, r3 = await asyncio.gather(
        slow_io_task("벡터검색", 1.0),
        slow_io_task("DB조회", 1.0),
        slow_io_task("리랭킹", 1.0),
    )
    print(f"병렬: {time.time() - start:.1f}초")

asyncio.run(main())
```

---

## 2. FastAPI

### 2.1 왜 FastAPI인가?

| 특징 | Flask | Django | **FastAPI** |
|------|-------|--------|-----------|
| async 지원 | 제한적 | 부분적 | **네이티브** |
| 타입 검증 | 수동 | Form 기반 | **Pydantic 자동** |
| API 문서 | 수동 | DRF로 가능 | **자동 (Swagger)** |
| 성능 | 보통 | 보통 | **최상위** |
| 학습 곡선 | 쉬움 | 높음 | 중간 |

### 2.2 핵심 개념

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Legal LLM API")

# Pydantic 스키마 = 자동 검증 + 자동 문서화
class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)
    use_rag: bool = True

class ChatResponse(BaseModel):
    answer: str
    disclaimer: str

# 라우터
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # request는 자동으로 타입 검증됨
    # message가 빈 문자열이면 422 에러 자동 반환
    return ChatResponse(
        answer="답변입니다.",
        disclaimer="면책 조항"
    )

# 서버 실행 후 http://localhost:8000/docs 에서 Swagger UI 확인
```

### 2.3 SSE (Server-Sent Events) 스트리밍

```python
from sse_starlette.sse import EventSourceResponse

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    async def event_generator():
        for token in model.generate_stream(prompt):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": "면책 조항"}
    
    return EventSourceResponse(event_generator())
```

**SSE vs WebSocket:**
```
SSE:       서버 → 클라이언트 (단방향)  ← 이 프로젝트 (LLM 응답 스트리밍)
WebSocket: 서버 ↔ 클라이언트 (양방향)  ← 채팅에 일반적이지만 여기선 과함
```

### 2.4 Lifespan (앱 생명주기)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행 (모델 로딩)
    await asyncio.to_thread(llm_engine.load)
    await retriever.initialize()
    
    yield  # 서버 실행 중
    
    # 서버 종료 시 실행 (정리)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

### 학습 자료

```
필수:
├── FastAPI 공식 튜토리얼
│   → https://fastapi.tiangolo.com/tutorial/
│   → "First Steps"부터 "Request Body"까지 필수
│
├── Pydantic V2 문서
│   → https://docs.pydantic.dev/latest/
│
└── SSE-Starlette 사용법
    → https://github.com/sysid/sse-starlette

권장:
└── FastAPI 공식 "Advanced" 섹션
    → Middleware, Dependencies, Background Tasks
```

---

## 3. Streamlit

### 3.1 왜 Streamlit인가?

```
React/Vue로 채팅 UI를 만들면:
  → 프론트엔드 빌드 환경 필요
  → JavaScript/TypeScript 코드 작성
  → API 통신 코드 작성
  → 개발 기간 1주일+

Streamlit으로 만들면:
  → Python 코드만으로 UI 구현
  → 채팅 컴포넌트 내장
  → 30줄이면 채팅 UI 완성
  → 개발 기간 수시간
```

### 3.2 핵심 패턴

```python
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="법률 상담 AI", layout="wide")

# 세션 상태 관리 (페이지 리렌더 간 데이터 유지)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = call_api(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 3.3 사이드바, 토글, 확장 패널

```python
# 사이드바
with st.sidebar:
    st.title("설정")
    use_rag = st.toggle("RAG 사용", value=True)
    if st.button("새 대화"):
        st.session_state.messages = []
        st.rerun()

# 확장 패널 (참고 자료 표시)
with st.expander("참고 자료"):
    st.markdown("**[법조문]** 민법 제750조")
```

### 학습 자료

```
├── Streamlit 공식 문서
│   → https://docs.streamlit.io/
│
├── "Build a chatbot" (Streamlit 공식 튜토리얼)
│   → https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
│
└── Streamlit Chat Elements API
    → st.chat_message, st.chat_input 사용법
```

---

## 4. Python 타입 힌트

### 4.1 이 프로젝트에서 필수인 이유

CLAUDE.md에서 타입 힌트를 필수로 지정했습니다. FastAPI + Pydantic은 타입 힌트를 기반으로 자동 검증과 문서화를 수행합니다.

```python
# 타입 힌트 기본
def search(query: str, top_k: int = 5) -> list[dict[str, str]]:
    ...

# Union 타입
def load_model(path: str | None = None) -> bool:
    ...

# 제네릭
from collections.abc import Generator, AsyncGenerator

def stream_tokens(prompt: str) -> Generator[str, None, None]:
    yield "토큰1"
    yield "토큰2"

async def async_stream() -> AsyncGenerator[str, None]:
    yield "토큰1"
```

### 학습 자료

```
├── Python 타입 힌트 공식 문서
│   → https://docs.python.org/3/library/typing.html
│
└── "Type hints cheat sheet" (mypy)
    → https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
```

---

## 5. 학습 체크리스트

| # | 주제 | 이해 수준 | 필수 여부 |
|---|------|----------|----------|
| 1 | async/await 기본 문법 | 실습 | 필수 |
| 2 | asyncio.to_thread (CPU 바운드 처리) | 실습 | 필수 |
| 3 | asyncio.gather (병렬 실행) | 실습 | 필수 |
| 4 | FastAPI 라우트 정의 (GET, POST) | 실습 | 필수 |
| 5 | Pydantic 스키마와 자동 검증 | 실습 | 필수 |
| 6 | SSE 스트리밍 응답 구현 | 실습 | 필수 |
| 7 | FastAPI Lifespan (startup/shutdown) | 개념 이해 | 필수 |
| 8 | Streamlit 채팅 UI 구현 | 실습 | 필수 |
| 9 | st.session_state 상태 관리 | 실습 | 필수 |
| 10 | Python 타입 힌트 (제네릭 포함) | 실습 | 필수 |
| 11 | FastAPI Middleware, Dependencies | 개념 이해 | 선택 |
| 12 | aiosqlite 비동기 DB 사용 | 실습 | 권장 |
