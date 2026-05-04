# 학습 가이드 07: Python 필수 문법 및 라이브러리

> 이 프로젝트의 소스 코드를 읽고 작성하기 위해 반드시 알아야 할 Python 문법과 라이브러리 총정리

---

## Part 1: 필수 문법

---

## 1. 타입 힌트 (Type Hints)

### 왜 필수인가?

이 프로젝트의 CLAUDE.md에서 **타입 힌트 필수, `Any` 사용 금지**로 규정했습니다.
FastAPI + Pydantic은 타입 힌트를 기반으로 요청 검증, 응답 직렬화, API 문서를 자동 생성합니다.

### 1.1 기본 타입

```python
# 변수
name: str = "민법"
article_num: int = 750
score: float = 0.95
is_loaded: bool = True

# 함수 인자와 반환값
def search(query: str, top_k: int = 5) -> list[str]:
    return ["결과1", "결과2"]

# None 허용 (Optional 대체)
def load_model(path: str | None = None) -> bool:
    if path is None:
        path = "default/path"
    return True
```

### 1.2 컬렉션 타입

```python
# Python 3.9+ 내장 타입 사용 (typing 모듈 불필요)
names: list[str] = ["민법", "형법"]
scores: dict[str, float] = {"민법": 0.95, "형법": 0.30}
unique_ids: set[int] = {1, 2, 3}
pair: tuple[str, int] = ("제750조", 750)

# 중첩
results: list[dict[str, str]] = [
    {"title": "민법 제750조", "content": "불법행위..."},
    {"title": "민법 제751조", "content": "재산 이외의 손해..."},
]
```

### 1.3 Union 타입

```python
# Python 3.10+ | 문법
def process(value: str | int) -> str:
    return str(value)

# None 허용 (가장 자주 사용하는 패턴)
def find_document(doc_id: str) -> dict[str, str] | None:
    if not found:
        return None
    return {"title": "...", "content": "..."}
```

### 1.4 Callable과 Generator

```python
from collections.abc import Callable, Generator, AsyncGenerator

# 콜백 함수 타입
def apply_filter(text: str, filter_fn: Callable[[str], str]) -> str:
    return filter_fn(text)

# 제너레이터 (스트리밍에서 사용)
def stream_tokens(prompt: str) -> Generator[str, None, None]:
    for token in ["답변", "입니다"]:
        yield token

# 비동기 제너레이터 (SSE 스트리밍)
async def async_stream() -> AsyncGenerator[str, None]:
    for token in ["답변", "입니다"]:
        yield token
```

### 1.5 TypeAlias와 복잡한 타입

```python
# 타입 별칭으로 가독성 향상
type SearchResult = dict[str, str]
type SearchResults = list[SearchResult]

# 또는 전통적 방식
SearchResult = dict[str, str]

def search(query: str) -> list[SearchResult]:
    return [{"title": "...", "content": "..."}]
```

---

## 2. 데이터클래스와 Pydantic

### 2.1 dataclass (설정 객체에 사용)

```python
from dataclasses import dataclass, field

# 이 프로젝트의 config.py에서 사용
@dataclass(frozen=True)  # frozen=True: 불변 객체
class ModelConfig:
    name: str = "EEVE-Korean-10.8B"
    max_tokens: int = 2048
    temperature: float = 0.3

config = ModelConfig()
print(config.name)       # "EEVE-Korean-10.8B"
# config.name = "other"  # frozen이므로 에러!

# 가변 기본값은 field(default_factory=...)
@dataclass
class TrainingConfig:
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
```

**dataclass vs 일반 클래스:**
```python
# 일반 클래스: __init__, __repr__, __eq__ 직접 구현 필요
class Config:
    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
    def __repr__(self):
        return f"Config(name={self.name}, port={self.port})"

# dataclass: 자동 생성
@dataclass
class Config:
    name: str
    port: int
    # __init__, __repr__, __eq__ 자동!
```

### 2.2 Pydantic BaseModel (API 스키마에 사용)

```python
from pydantic import BaseModel, Field

# 이 프로젝트의 schemas.py에서 사용
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., min_length=1, description="질문")
    use_rag: bool = Field(default=True)

# 자동 검증
request = ChatRequest(session_id="abc", message="질문입니다")  # OK
request = ChatRequest(session_id="abc", message="")  # 에러! min_length=1

# JSON ↔ 객체 자동 변환
json_data = '{"session_id": "abc", "message": "질문", "use_rag": true}'
request = ChatRequest.model_validate_json(json_data)

# 객체 → dict
data = request.model_dump()
# {"session_id": "abc", "message": "질문", "use_rag": True}
```

**dataclass vs Pydantic 사용 구분:**
```
dataclass  → 내부 설정 객체 (config.py)
             변환/검증 불필요, 단순 데이터 컨테이너
             
Pydantic   → 외부 입출력 (API 요청/응답)
             자동 검증, JSON 직렬화, API 문서 생성
```

---

## 3. 비동기 프로그래밍 (async/await)

### 3.1 기본 패턴

```python
import asyncio

# 비동기 함수 정의
async def fetch_data(url: str) -> dict[str, str]:
    # await: 비동기 작업이 끝날 때까지 대기
    # 대기하는 동안 다른 코루틴이 실행됨
    response = await some_async_call(url)
    return response

# 비동기 함수 호출
async def main():
    result = await fetch_data("http://...")
    print(result)

# 이벤트 루프 실행
asyncio.run(main())
```

### 3.2 동시 실행 패턴

```python
# gather: 여러 비동기 작업을 동시에 실행
async def process_request(query: str):
    # 벡터 검색과 DB 조회를 동시에!
    docs, history = await asyncio.gather(
        retriever.search(query),       # 1초
        db.get_history(session_id),    # 0.5초
    )
    # 총 1초 (순차면 1.5초)
    return docs, history
```

### 3.3 동기 함수를 비동기로 변환

```python
import asyncio

# CPU 바운드 작업 (모델 추론)은 별도 스레드에서 실행
async def generate_answer(prompt: str) -> str:
    # to_thread: 블로킹 함수를 별도 스레드에서 실행
    # 메인 이벤트 루프를 차단하지 않음
    result = await asyncio.to_thread(
        llm_engine.generate,  # 동기 함수
        prompt,               # 인자
    )
    return result
```

### 3.4 async for (비동기 이터레이션)

```python
# 비동기 제너레이터
async def stream_response():
    async for chunk in async_generator():
        yield chunk

# 사용
async for token in stream_response():
    print(token, end="")
```

### 3.5 async with (비동기 컨텍스트 매니저)

```python
import aiosqlite

# 비동기 DB 연결
async def get_history(session_id: str) -> list[dict[str, str]]:
    async with aiosqlite.connect("data.db") as conn:
        cursor = await conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
```

---

## 4. 컨텍스트 매니저

### 4.1 with 문

```python
# 파일 자동 닫기
with open("data.txt", encoding="utf-8") as f:
    content = f.read()
# f는 여기서 자동으로 닫힘

# DB 연결 자동 해제
with get_connection() as conn:
    conn.execute("INSERT INTO ...")
```

### 4.2 contextmanager 데코레이터

```python
from contextlib import contextmanager, asynccontextmanager

# 동기 컨텍스트 매니저
@contextmanager
def timer(name: str):
    import time
    start = time.time()
    yield  # with 블록 실행
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}초")

with timer("벡터 검색"):
    results = retriever.search("보증금 반환")

# 비동기 컨텍스트 매니저 (이 프로젝트의 lifespan)
@asynccontextmanager
async def lifespan(app):
    # startup
    await load_model()
    yield
    # shutdown
    print("종료")
```

---

## 5. 제너레이터와 이터레이터

### 5.1 yield (스트리밍의 핵심)

```python
# 일반 함수: 모든 결과를 메모리에 올린 후 반환
def get_all_tokens(prompt: str) -> list[str]:
    tokens = []
    for token in model.generate(prompt):
        tokens.append(token)
    return tokens  # 전체가 메모리에 있어야 함

# 제너레이터: 하나씩 생성하며 반환 (메모리 효율적)
def stream_tokens(prompt: str) -> Generator[str, None, None]:
    for token in model.generate(prompt):
        yield token  # 하나 반환 후 일시 정지, 다음 요청 시 재개

# 사용
for token in stream_tokens("민사소송이란?"):
    print(token, end="", flush=True)  # 실시간 출력
```

### 5.2 yield가 이 프로젝트에서 쓰이는 곳

```python
# 1. LLM 스트리밍 추론 (engine.py)
def generate_stream(self, prompt: str) -> Generator[str, None, None]:
    for response in stream_generate(model, tokenizer, prompt=prompt):
        yield response.text

# 2. SSE 이벤트 생성 (routes.py)
async def event_generator():
    for token in llm_engine.generate_stream(prompt):
        yield {"event": "token", "data": token}
    yield {"event": "done", "data": disclaimer}
```

---

## 6. 예외 처리

### 6.1 이 프로젝트의 예외 처리 원칙

```python
# CLAUDE.md 규칙: 외부 경계에서만 예외 처리

# ✅ 외부 경계: 파일 I/O
def load_documents(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("파일 없음: %s", path)
        raise
    except UnicodeDecodeError:
        logger.error("인코딩 오류: %s", path)
        raise
    return parse(text)

# ✅ 외부 경계: 모델 로딩
def load(self) -> None:
    try:
        from mlx_lm import load
        self._model, self._tokenizer = load(model_name)
    except ImportError:
        logger.warning("mlx_lm 없음, CPU 모드로 전환")
    except Exception:
        logger.exception("모델 로딩 실패")

# ❌ 내부 함수 간: 방어적 검증 금지
def build_prompt(query: str, context: str) -> str:
    # if not isinstance(query, str):  ← 이런 검증 하지 않음
    # if query is None:               ← 타입 힌트로 보장
    return f"질문: {query}\n참고: {context}"
```

### 6.2 커스텀 예외 (FastAPI)

```python
from fastapi import HTTPException, status

# 이 프로젝트의 exceptions.py
class ModelNotLoadedError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="모델이 로드되지 않았습니다.",
        )

# 사용
@router.post("/api/chat")
async def chat(request: ChatRequest):
    if not llm_engine.is_loaded:
        raise ModelNotLoadedError()  # 자동으로 503 응답
```

---

## 7. 패턴 매칭과 문자열 처리

### 7.1 정규표현식 (법률 문서 파싱에 필수)

```python
import re

# 법조문 번호 추출
pattern = re.compile(r'(제\d+조(?:의\d+)?(?:\([^)]+\))?)')
text = "제750조(불법행위의 내용) 고의 또는 과실로..."
match = pattern.search(text)
print(match.group(1))  # "제750조(불법행위의 내용)"

# 판례 사건번호 추출
case_pattern = re.compile(r'(\d{4}[가-힣]+\d+)')
text = "대법원 2023다12345 판결"
match = case_pattern.search(text)
print(match.group(1))  # "2023다12345"

# 문자열 치환 (개인정보 비식별화)
text = "원고 김철수(주민번호 900101-1234567)는..."
cleaned = re.sub(r'\d{6}-\d{7}', '[주민번호 삭제]', text)
# "원고 김철수(주민번호 [주민번호 삭제])는..."

# split으로 조문 분리
parts = pattern.split("제1조 내용 제2조 내용")
# ['', '제1조', ' 내용 ', '제2조', ' 내용']
```

### 7.2 f-string 포매팅

```python
year = 2026
article = 750
law_name = "민법"

# 기본
ref = f"{law_name} 제{article}조"  # "민법 제750조"

# 정렬, 패딩
wk = 5
formatted = f"{wk:02d}"  # "05" (2자리 zero-padding)

# 소수점
score = 0.9567
formatted = f"{score:.2f}"  # "0.96"

# 딕셔너리
meta = {"title": "민법", "type": "법조문"}
log = f"문서: {meta['title']} ({meta['type']})"
```

---

## 8. 리스트/딕셔너리 컴프리헨션

```python
# 리스트 컴프리헨션
scores = [0.9, 0.3, 0.7, 0.1, 0.8]
high_scores = [s for s in scores if s > 0.5]
# [0.9, 0.7, 0.8]

# 딕셔너리 컴프리헨션
docs = [{"title": "민법", "score": 0.9}, {"title": "형법", "score": 0.3}]
title_score = {d["title"]: d["score"] for d in docs}
# {"민법": 0.9, "형법": 0.3}

# 이 프로젝트에서의 실제 사용
sources = [
    SourceDocument(
        title=r.title,
        doc_type=r.doc_type,
        content_preview=r.content[:100],
    )
    for r in rag_results  # 검색 결과를 API 응답 객체로 변환
]
```

---

## 9. 싱글턴 패턴

```python
# 이 프로젝트의 engine.py에서 사용
# LLM 모델은 하나만 로딩해야 함 (48GB 메모리 제약)

class LLMEngine:
    _instance: "LLMEngine | None" = None

    def __new__(cls) -> "LLMEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._loaded = False
        return cls._instance

# 어디서 생성해도 같은 인스턴스
engine1 = LLMEngine()
engine2 = LLMEngine()
assert engine1 is engine2  # True

# 모듈 레벨 인스턴스로도 가능 (더 간단)
# engine.py 하단에:
llm_engine = LLMEngine()

# 다른 모듈에서:
from src.llm.engine import llm_engine
```

---

## 10. threading (동시 추론 제한)

```python
import threading

# 이 프로젝트: 동시 추론 1개로 제한 (M4 메모리 제약)
class LLMEngine:
    def __init__(self):
        self._inference_lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        # Lock: 한 번에 하나의 스레드만 추론 가능
        with self._inference_lock:
            return self._model.generate(prompt)
        # Lock 해제 → 다음 요청 처리

# _lock 패턴 (모델 로딩 시 race condition 방지)
_lock = threading.Lock()

def load(self):
    if self._loaded:
        return
    with _lock:               # 여러 스레드가 동시에 로딩 시도해도
        if self._loaded:      # double-check
            return
        self._model = load_model()
        self._loaded = True
```

---

## Part 2: 핵심 라이브러리

---

## 11. pathlib (파일 경로 처리)

```python
from pathlib import Path

# 경로 생성
data_dir = Path("data/raw")
file_path = data_dir / "민법.txt"  # / 연산자로 경로 결합

# 파일 읽기/쓰기
text = file_path.read_text(encoding="utf-8")
file_path.write_text("내용", encoding="utf-8")

# 디렉터리 탐색
for txt_file in data_dir.rglob("*.txt"):  # 재귀 검색
    print(txt_file.name)    # "민법.txt"
    print(txt_file.stem)    # "민법"
    print(txt_file.suffix)  # ".txt"
    print(txt_file.parent)  # Path("data/raw")

# 존재 여부 확인
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
```

---

## 12. json (데이터 직렬화)

```python
import json

# JSONL 읽기 (이 프로젝트의 학습 데이터 포맷)
entries: list[dict[str, str]] = []
with open("data/processed/train.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            entries.append(entry)

# JSONL 쓰기
with open("output.jsonl", "w", encoding="utf-8") as f:
    for entry in entries:
        # ensure_ascii=False: 한국어를 유니코드 이스케이프하지 않음
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# JSON 포맷팅 (디버깅용)
data = {"instruction": "질문", "output": "답변"}
print(json.dumps(data, ensure_ascii=False, indent=2))
```

---

## 13. logging (로깅)

```python
import logging

# 로거 생성 (모듈별)
logger = logging.getLogger(__name__)

# 기본 설정 (main.py에서 한 번)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 사용
logger.info("모델 로딩 시작: %s", model_name)    # 일반 정보
logger.warning("벡터DB 없음: %s", db_path)        # 경고
logger.error("모델 로딩 실패")                     # 에러
logger.exception("예외 발생")                      # 에러 + 스택트레이스

# 주의: 개인정보가 포함될 수 있는 사용자 질의는 로깅하지 않음!
# ❌ logger.info("사용자 질의: %s", user_message)
# ✅ logger.info("질의 처리 완료, session=%s", session_id)
```

---

## 14. yaml (설정 파일)

```python
import yaml
from pathlib import Path

# 읽기
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    # safe_load: 안전한 YAML 파싱 (임의 코드 실행 방지)

print(config["model"]["name"])      # "EEVE-Korean-10.8B"
print(config["rag"]["chunk_size"])  # 512

# 쓰기 (필요 시)
with open("output.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
```

---

## 15. argparse (CLI 도구)

```python
import argparse

# 이 프로젝트의 indexer.py, prepare_data.py에서 사용
def main():
    parser = argparse.ArgumentParser(description="RAG 인덱싱 도구")
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="원본 문서 디렉터리",
    )
    parser.add_argument(
        "--db-dir", default="data/vectordb",
        help="벡터DB 저장 디렉터리",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="인덱싱 배치 크기",
    )
    args = parser.parse_args()

    print(args.data_dir)    # "data/raw"
    print(args.batch_size)  # 100

# 실행: python -m src.rag.indexer --data-dir data/raw --db-dir data/vectordb
```

---

## 16. pytest (테스트)

### 16.1 기본 테스트

```python
# tests/test_prompt.py
from src.llm.prompt import build_chat_prompt

def test_prompt_with_context():
    prompt = build_chat_prompt(
        query="보증금 반환?",
        context="민법 제312조...",
        has_sources=True,
    )
    assert "보증금 반환?" in prompt
    assert "민법 제312조" in prompt

def test_prompt_without_context():
    prompt = build_chat_prompt(query="질문", context="", has_sources=False)
    assert "관련 자료를 찾지 못했습니다" in prompt
```

### 16.2 픽스처 (Fixture)

```python
import pytest
from pathlib import Path

# conftest.py에 공유 픽스처 정의
@pytest.fixture
def sample_document(tmp_path: Path) -> Path:
    """테스트용 임시 법률 문서 생성."""
    doc = tmp_path / "민법.txt"
    doc.write_text("제1조 이 법은 민사에 관한 기본법이다.", encoding="utf-8")
    return doc

# 테스트에서 픽스처 사용 (이름으로 자동 주입)
def test_load_document(sample_document: Path):
    text = sample_document.read_text(encoding="utf-8")
    assert "제1조" in text
```

### 16.3 비동기 테스트

```python
import pytest

# pytest-asyncio 사용
@pytest.mark.asyncio
async def test_async_search():
    results = await retriever.search("보증금 반환")
    assert len(results) > 0
```

### 16.4 모킹

```python
from unittest.mock import MagicMock, patch

# LLM 모킹 (실제 모델 로딩 없이 테스트)
@pytest.fixture
def mock_llm():
    with patch("src.api.routes.llm_engine") as mock:
        mock.is_loaded = True
        mock.generate.return_value = "테스트 답변"
        yield mock

async def test_chat_with_mock(mock_llm, async_client):
    response = await async_client.post("/api/chat", json={
        "session_id": "test",
        "message": "질문",
    })
    assert response.status_code == 200
    assert "테스트 답변" in response.json()["answer"]
```

### 16.5 실행

```bash
# 전체 테스트
pytest tests/ -v

# 특정 파일
pytest tests/test_api.py -v

# 특정 테스트
pytest tests/test_api.py::test_health_check -v

# 출력 표시
pytest tests/ -v -s
```

---

## 17. 자주 사용하는 내장 함수

```python
# enumerate: 인덱스와 값을 함께 순회
for i, doc in enumerate(documents):
    print(f"{i}: {doc['title']}")

# zip: 여러 리스트를 병렬 순회
for doc, score in zip(documents, scores):
    print(f"[{score:.2f}] {doc['title']}")

# sorted: 정렬 (리랭킹 결과 정렬에 사용)
results.sort(key=lambda r: r.score, reverse=True)

# any/all: 조건 검사
has_relevant = any(r.score > 0.5 for r in results)
all_valid = all(entry.get("output") for entry in data)

# isinstance: 타입 확인 (외부 경계에서만)
if not isinstance(raw_data, dict):
    raise ValueError("잘못된 데이터 형식")
```

---

## 학습 체크리스트

### 문법

| # | 주제 | 이 프로젝트에서 사용되는 곳 | 필수 |
|---|------|--------------------------|------|
| 1 | 타입 힌트 (기본 + 컬렉션 + Union) | 모든 파일 | 필수 |
| 2 | dataclass (frozen, field) | config.py | 필수 |
| 3 | Pydantic BaseModel + Field | schemas.py | 필수 |
| 4 | async/await + asyncio.to_thread | routes.py, retriever.py | 필수 |
| 5 | asyncio.gather (병렬 실행) | routes.py | 필수 |
| 6 | 컨텍스트 매니저 (with, asynccontextmanager) | main.py, database.py | 필수 |
| 7 | 제너레이터 (yield) | engine.py 스트리밍 | 필수 |
| 8 | 정규표현식 (re) | indexer.py 법조문 파싱 | 필수 |
| 9 | 리스트/딕셔너리 컴프리헨션 | 전체 | 필수 |
| 10 | 싱글턴 패턴 (__new__) | engine.py | 권장 |
| 11 | threading.Lock | engine.py 동시 추론 제한 | 권장 |
| 12 | 예외 처리 (try/except/raise) | 외부 경계 | 필수 |

### 라이브러리

| # | 라이브러리 | 이 프로젝트에서 사용되는 곳 | 필수 |
|---|-----------|--------------------------|------|
| 1 | pathlib (Path) | indexer.py, train_lora.py | 필수 |
| 2 | json (loads, dumps, JSONL) | prepare_data.py, routes.py | 필수 |
| 3 | logging | 모든 모듈 | 필수 |
| 4 | yaml (safe_load) | config.py | 필수 |
| 5 | argparse | indexer.py, prepare_data.py, train_lora.py | 필수 |
| 6 | pytest + pytest-asyncio | tests/ | 필수 |
| 7 | unittest.mock (patch, MagicMock) | tests/conftest.py | 필수 |
| 8 | subprocess | train_lora.py | 권장 |
| 9 | uuid | routes.py 세션 관리 | 권장 |
| 10 | random | prepare_data.py 데이터 분할 | 권장 |
