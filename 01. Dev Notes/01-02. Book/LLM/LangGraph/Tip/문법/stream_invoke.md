LangGraph에서 `graph = builder.compile()`로 만든 그래프는 실행 메서드 4개를 제공합니다. 앞의 asyncio 예제와 연결하면 이해가 쉽습니다.

## 한눈에 비교

| 메서드 | 동기/비동기 | 반환값 | 호출 방법 | 대응되는 예제 개념 |
|--------|------------|--------|-----------|------------------|
| `invoke` | 동기 | **최종 상태** (dict) | `graph.invoke(input)` | 일반 함수 (`01_sync`) |
| `ainvoke` | 비동기 | **최종 상태** (코루틴) | `await graph.ainvoke(input)` | 코루틴 (`01`) |
| `stream` | 동기 | **중간 결과를 하나씩** (제너레이터) | `for chunk in graph.stream(input)` | 제너레이터 (`04_sync`) |
| `astream` | 비동기 | **중간 결과를 하나씩** (비동기 제너레이터) | `async for chunk in graph.astream(input)` | 비동기 제너레이터 (`04`) |

```
invoke/ainvoke :  START ─→ node_a ─→ node_b ─→ END  ──→ 최종 결과 한 번 반환
stream/astream :  START ─→ node_a ─→ node_b ─→ END
                            ↓ chunk     ↓ chunk          ← 노드가 끝날 때마다 받음
```

## 예제 그래프

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    script: str
    title: str


async def write_script(state: State) -> dict:
    await asyncio.sleep(1)  # LLM 호출이라고 가정
    return {"script": f"{state['topic']}에 대한 쇼츠 대본"}


async def make_title(state: State) -> dict:
    await asyncio.sleep(0.5)
    return {"title": f"[쇼츠] {state['topic']}"}


builder = StateGraph(State)
builder.add_node("write_script", write_script)
builder.add_node("make_title", make_title)
builder.add_edge(START, "write_script")
builder.add_edge("write_script", "make_title")
builder.add_edge("make_title", END)
graph = builder.compile()
```

## 1. `invoke` / `ainvoke`: 끝까지 실행하고 최종 상태 반환

```python
# 동기
result = graph.invoke({"topic": "asyncio"})
print(result)
# {'topic': 'asyncio', 'script': 'asyncio에 대한 쇼츠 대본', 'title': '[쇼츠] asyncio'}

# 비동기
async def main():
    result = await graph.ainvoke({"topic": "asyncio"})

    # ainvoke의 진짜 장점: 여러 실행을 '동시에' 돌릴 수 있음 (03 예제의 gather)
    results = await asyncio.gather(
        graph.ainvoke({"topic": "A"}),
        graph.ainvoke({"topic": "B"}),
        graph.ainvoke({"topic": "C"}),
    )   # 약 1.5초 (순차 invoke 3번이면 약 4.5초)

asyncio.run(main())
```

- 중간 과정은 볼 수 없고 **최종 결과만** 필요할 때 씁니다 (배치 처리, 테스트, API 응답).
- 노드 함수가 `async def`라면 `ainvoke`/`astream`으로 실행해야 합니다.

## 2. `stream` / `astream`: 진행 상황을 하나씩 받기

```python
# 동기
for chunk in graph.stream({"topic": "asyncio"}):
    print(chunk)

# 비동기
async def main():
    async for chunk in graph.astream({"topic": "asyncio"}):
        print(chunk)
```

### `stream_mode`: 무엇을 흘려보낼지 선택

| `stream_mode` | 받는 값 | 용도 |
|---------------|---------|------|
| `"values"` | 각 단계 후의 **전체 상태** | 상태 변화를 전부 추적 |
| `"updates"` (stream 기본값) | 각 노드가 **반환한 변경분만** `{노드명: 반환값}` | 어떤 노드가 무엇을 했는지 진행률 표시 |
| `"messages"` | LLM 토큰 단위 `(message_chunk, metadata)` | 채팅 UI의 타이핑 효과 |
| `"custom"` | 노드 안에서 직접 보낸 값 | 진행률(%), 로그 등 임의 데이터 |
| `"debug"` | 실행 내부의 상세 이벤트 | 디버깅 |

```python
# "updates": 노드별 변경분
async for chunk in graph.astream({"topic": "asyncio"}, stream_mode="updates"):
    print(chunk)
# {'write_script': {'script': 'asyncio에 대한 쇼츠 대본'}}   ← 1초 후
# {'make_title': {'title': '[쇼츠] asyncio'}}               ← 0.5초 후

# "values": 매 단계의 전체 상태
async for state in graph.astream({"topic": "asyncio"}, stream_mode="values"):
    print(state)
# {'topic': 'asyncio'}
# {'topic': 'asyncio', 'script': '...'}
# {'topic': 'asyncio', 'script': '...', 'title': '...'}

# 여러 모드를 함께 쓰면 (mode, chunk) 튜플로 옴
async for mode, chunk in graph.astream(inp, stream_mode=["updates", "custom"]):
    ...
```

### LLM 토큰 스트리밍 (`"messages"`)

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("anthropic:claude-sonnet-5-5")

async def write_script(state: State, config) -> dict:   # config를 받아서
    msg = await llm.ainvoke(f"{state['topic']} 쇼츠 대본 써줘", config)  # 그대로 전달
    return {"script": msg.content}

async for token, meta in graph.astream(inp, stream_mode="messages"):
    if meta["langgraph_node"] == "write_script":
        print(token.content, end="", flush=True)
```

노드 안에서 `llm.ainvoke()`로 전체 응답을 받아도 LangGraph가 토큰을 가로채 스트림으로 내보냅니다.

### 직접 진행 상황 보내기 (`"custom"`)

```python
from langgraph.config import get_stream_writer

async def render_video(state: State) -> dict:
    writer = get_stream_writer()
    for p in (25, 50, 75, 100):
        await asyncio.sleep(0.2)
        writer({"progress": p})
    return {}

async for chunk in graph.astream(inp, stream_mode="custom"):
    print(chunk)   # {'progress': 25} ...
```

## 주의: Python 3.10 환경에서는

현재 환경이 **Python 3.10.7**입니다. Python 3.11 미만에서는 asyncio가 contextvar를 태스크로 자동 전파하지 않아서 비동기 실행 시 두 가지 제약이 있습니다.
- **`"messages"` 모드:** 노드에서 `config`를 받아 LLM 호출에 **직접 넘겨야** 토큰이 스트리밍됩니다. 위 예제처럼 `llm.ainvoke(..., config)`로 호출하세요.
- **`get_stream_writer()`:** 비동기 노드에서는 사용할 수 없습니다. 대신 노드 함수 인자로 `writer`를 받으세요 (`async def node(state, writer): ...`).

3.11 이상으로 올리면 두 제약 모두 사라집니다.

## 언제 무엇을 쓰나

| 상황 | 추천 |
|------|------|
| 스크립트나 배치에서 결과만 필요 | `invoke` |
| FastAPI 등 비동기 서버, 여러 요청을 동시 처리 | `ainvoke` |
| CLI에서 진행 상황 출력 | `stream(stream_mode="updates")` |
| 웹 UI에 토큰이나 진행률 실시간 전송 (SSE, 웹소켓) | `astream(stream_mode=["messages", "custom"])` |

**정리:** `invoke`와 `stream` 중 무엇을 쓸지는 **결과를 한 번에 받을지, 과정을 하나씩 받을지**로 정합니다. 앞에 `a`를 붙인 비동기 버전은 **이벤트 루프 위에서 여러 실행을 동시에 돌릴 수 있다**는 점이 다르며, `async for`와 `gather` 예제에서 본 개념이 그대로 적용됩니다.

예제 코드는 직접 실행해 보지 않았고, 사용법은 최신 LangGraph 기준으로 적었습니다. 원하시면 이 내용을 `example/06_langgraph_stream.py`로 만들어 실행해 볼게요. 설치된 langgraph 버전 확인이 필요합니다.
