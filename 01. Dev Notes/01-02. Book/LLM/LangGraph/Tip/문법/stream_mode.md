LangChain의 `agent.stream()`에서 `stream_mode`는 에이전트가 실행되는 동안 어떤 종류의 중간 결과를 전달받을 것인지 결정하는 옵션입니다.
대표적으로 `updates`, `messages`, `custom` 세 가지가 있습니다.
특히 LangGraph 기반의 LangChain `create_agent()`에서는 세 가지 모드의 용도가 명확하게 구분됩니다.

### 1. 세 가지 stream_mode 비교

| stream_mode | 목적              | 전달되는 데이터          |
| ----------- | --------------- | ----------------- |
| `updates`   | 에이전트 실행 단계 추적   | 각 노드의 상태 변경 결과    |
| `messages`  | LLM 응답 실시간 출력   | 토큰 단위 메시지 및 메타데이터 |
| `custom`    | 사용자 정의 진행 상황 출력 | 개발자가 직접 전달한 데이터   |

쉽게 구분하면 다음과 같습니다.
- `updates`: 에이전트가 지금 무슨 작업을 했는가?
- `messages`: LLM이 지금 무슨 말을 생성하고 있는가?
- `custom`: 개발자가 정의한 작업 진행 상황은 무엇인가?

### 2. updates 모드
기본적인 에이전트 실행 추적
에이전트 내부에서 노드가 실행되고 상태가 변경될 때마다 결과를 전달합니다.
질문에서 작성한 코드가 바로 이 방식입니다.

```python
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode="updates"
):
    print(chunk)
```

예상되는 실행 흐름은 다음과 같습니다.

(1)model 노드
LLM이 사용자 질문을 분석하고 도구 호출을 결정합니다.

(2) tools 노드
`get_weather("Seoul")` 실행

(3) model 노드
도구 실행 결과를 바탕으로 최종 답변 생성
실제 반환 데이터는 대략 다음과 같은 구조입니다.
```python
{
    "model": {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Seoul"},
                        "id": "call_123"
                    }
                ]
            )
        ]
    }
}
```

이후 tools 노드에서는 다음과 비슷한 결과가 나옵니다.
```python
{
    "tools": {
        "messages": [
            ToolMessage(
                content="The weather in Seoul is sunny!",
                name="get_weather"
            )
        ]
    }
}
```

마지막 model 노드에서는 다음과 같은 AIMessage가 반환됩니다.
```python
{
    "model": {
        "messages": [
            AIMessage(
                content="The weather in Seoul is sunny!"
            )
        ]
    }
}
```

※ 실제 데이터에는 메시지 ID, 실행 메타데이터 등이 추가될 수 있습니다.
핵심: `updates`는 LLM의 토큰을 하나씩 출력하는 것이 아니라 에이전트의 실행 단계별 상태 변경을 전달합니다.

### 3. messages 모드
LLM 토큰 실시간 스트리밍
LLM이 답변을 생성하는 과정을 실시간으로 받아볼 수 있습니다.
```python
for message, metadata in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode="messages"
):
    print(message.content, end="", flush=True)
```

예를 들어 LLM이 다음 문장을 생성한다고 가정하겠습니다.
```
The weather in Seoul is sunny!
```

실제 스트리밍에서는 다음과 같이 여러 청크로 나누어 전달될 수 있습니다.
```
The
 weather
 in
 Seoul
 is
 sunny!
```

여기서 중요한 것은 두 번째 반환값인 `metadata`입니다.
```python
for message, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Seoul?"}]},
    stream_mode="messages"
):
    print("Content:", message.content)
    print("Node:", metadata.get("langgraph_node"))
```

`metadata`에는 다음과 같은 정보가 포함될 수 있습니다.

| 항목               | 설명                   |
| ---------------- | -------------------- |
| `langgraph_node` | 메시지를 생성한 노드          |
| `tags`           | 실행 관련 태그             |
| `ls_*`           | LangSmith 관련 메타데이터 등 |

특히 `langgraph_node`를 활용하면 어떤 노드에서 생성한 메시지인지 구분할 수 있습니다.

주의할 점)
`messages` 모드에서는 도구 호출을 위한 AI 메시지도 전달될 수 있습니다. 따라서 단순히 `message.content`만 출력하면 사용자에게 보여줄 최종 답변 외의 데이터가 섞일 수 있습니다.

### 4. custom 모드
개발자가 직접 진행 상황 전달
`custom`은 개발자가 에이전트 내부에서 원하는 데이터를 직접 스트리밍하는 방식입니다.
예를 들어 다음과 같은 진행 상황을 전달할 수 있습니다.
- 파일 분석 시작
- 문서 10개 검색 완료
- 데이터베이스 조회 완료
- 전체 작업의 50% 완료

이를 위해 LangGraph의 `get_stream_writer()`를 사용합니다.
```python
from langgraph.config import get_stream_writer
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""

    writer = get_stream_writer()

    writer({
        "status": "started",
        "message": f"Getting weather for {city}"
    })

    # 실제 날씨 조회 작업
    result = f"The weather in {city} is sunny!"

    writer({
        "status": "completed",
        "message": f"Weather lookup completed for {city}"
    })

    return result
```

이제 에이전트에서 다음과 같이 실행합니다.
```python
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode="custom"
):
    print(chunk)
```

예상 출력:
```
{
    "status": "started",
    "message": "Getting weather for Seoul"
}

{
    "status": "completed",
    "message": "Weather lookup completed for Seoul"
}
```

`custom`은 LLM이 생성하는 메시지가 아니라 개발자가 직접 전달하는 데이터라는 점이 중요합니다.

### 5. 세 가지 모드 동시에 사용하기
실제 에이전트 서비스를 개발할 때는 여러 스트리밍 모드를 동시에 사용하는 것도 유용합니다.
```python
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode=["updates", "messages", "custom"]
):
    print(chunk)
```

이 경우 반환값은 다음과 같은 형태입니다.
```
("updates", {...})
("messages", (message_chunk, metadata))
("custom", {...})
```

즉, 첫 번째 값으로 어떤 스트리밍 모드에서 전달된 데이터인지 구분할 수 있습니다.

### 6. 실제 AI 에이전트 개발에서 어떤 모드를 사용해야 할까?

사용 목적에 따라 선택하면 됩니다.

| 개발 목적               | 추천 모드                 |
| ------------------- | --------------------- |
| 에이전트 내부 실행 흐름 디버깅   | `updates`             |
| ChatGPT처럼 답변 실시간 출력 | `messages`            |
| RAG 검색 진행 상황 표시     | `custom`              |
| 도구 실행 상태 표시         | `updates` 또는 `custom` |
| 복합적인 AI 에이전트 모니터링   | 세 가지 조합               |

제가 추천하는 것은 실제 서비스에서는 세 가지를 조합하는 방식입니다.
예를 들어 RAG 기반 AI 에이전트를 개발한다면 다음과 같이 활용할 수 있습니다.

RAG 에이전트 실행 예시

(1) 질문 분석
   updates
   LLM이 사용자 질문 분석
   
(2) 문서 검색
   custom
   `문서 검색 시작 → 100개 검색 → 상위 5개 추출`
   
(3) 답변 생성
   messages
   LLM이 답변을 토큰 단위로 생성
   
(4) 최종 응답
   updates
   에이전트 실행 완료 및 상태 변경

### 7. 추가로 알아두면 좋은 점
LangChain의 `create_agent()`는 LangGraph 기반이므로 `stream_mode`는 LangGraph의 스트리밍 기능을 따릅니다.

참고로 추가로 알아둘 만한 모드도 있습니다.
- `values`: 매번 업데이트된 전체 그래프 상태를 반환
- `checkpoints`: 체크포인트 정보 스트리밍
- `tasks`: 각 태스크의 시작 및 완료 정보
- `debug`: 디버깅에 필요한 상세 이벤트

공식 문서:
- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming?utm_source=chatgpt.com)
- [LangChain Streaming](https://docs.langchain.com/oss/python/langchain/streaming?utm_source=chatgpt.com)

정리하면: `updates`는 에이전트의 실행 흐름, `messages`는 LLM의 실시간 출력, `custom`은 개발자가 제어하는 진행 상황입니다.

다음 단계로는 `messages` 모드에서 `AIMessageChunk`, `tool_call_chunks`, `metadata`가 실제로 어떤 형태로 전달되는지 이해하면 LangGraph 스트리밍을 훨씬 깊이 있게 활용할 수 있습니다.

---
아래는 LangChain의 `create_agent()`와 LangGraph의 `get_stream_writer()`를 이용한 custom 스트리밍 기본 예제입니다.

도구가 실행되는 동안 개발자가 직접 진행 메시지를 전달하고, 메인 코드에서 이를 수신하는 구조입니다.

### 1. 전체 예제 코드

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.config import get_stream_writer

from edu.config.model import load_llm


# 1. Custom 스트리밍 메시지 전달
def emit(node: str, message: str):
    writer = get_stream_writer()
    writer({
        "node": node,
        "message": message
    })


# 2. 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""

    emit("weather", f"{city} 날씨 조회 시작")

    # 실제 날씨 조회 작업
    result = f"The weather in {city} is sunny!"

    emit("weather", f"{city} 날씨 조회 완료")

    return result


# 3. LLM 및 Agent 생성
model = load_llm()

agent = create_agent(
    model=model,
    tools=[get_weather]
)


# 4. Custom 스트리밍 실행
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode="custom"
):
    print(chunk)

```

### 2. 실행 결과

예상 출력은 다음과 같습니다.
```
{'node': 'weather', 'message': 'Seoul 날씨 조회 시작'}

{'node': 'weather', 'message': 'Seoul 날씨 조회 완료'}
```

여기서 LLM의 최종 답변은 출력되지 않습니다.
`custom`은 개발자가 직접 전달한 데이터만 수신하기 때문입니다.

### 3. 최종 답변도 함께 출력하려면?
`custom`과 `messages`를 동시에 사용하면 됩니다.

```python
for mode, chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seoul?"}
        ]
    },
    stream_mode=["custom", "messages"]
):
    if mode == "custom":
        print("\n[진행 상태]", chunk)

    elif mode == "messages":
        message, metadata = chunk

        if message.content:
            print(message.content, end="", flush=True)
```

이렇게 하면 다음 두 가지를 동시에 처리할 수 있습니다.
- `custom`: 도구 실행 및 작업 진행 상황 표시
- `messages`: LLM의 답변을 토큰 단위로 실시간 출력

### 4. 핵심 정리

| 구분                                   | 역할                   |
| ------------------------------------ | -------------------- |
| `get_stream_writer()`                | custom 스트림 writer 획득 |
| `writer({...})`                      | 사용자 정의 데이터 전송        |
| `stream_mode="custom"`               | custom 데이터 수신        |
| `stream_mode="messages"`             | LLM 토큰 수신            |
| `stream_mode=["custom", "messages"]` | 두 종류 동시 수신           |

참고: 위 예제는 그래프 내부에서 `emit()`을 호출하는 경우를 전제로 합니다. 그래프 외부에서도 실행하려면 앞서 작성한 것처럼 `RuntimeError` 예외 처리를 추가하면 됩니다.

LangGraph 기반 AI Agent를 개발할 때는 `custom`을 작업 상태 전달용 이벤트 인터페이스로 활용하는 것이 좋습니다.

---

## stream_mode의 chunk

`stream_mode`를 리스트로 주면 매 반복마다 `(mode, chunk)` 튜플이 나오고, `chunk`의 모양은 `mode`에 따라 달라집니다. (실행해 보지는 않았고, LangGraph 동작 기준 설명입니다.)

| `mode`       | `chunk`의 타입                    | 내용                                                                 |
| ------------ | ------------------------------ | ------------------------------------------------------------------ |
| `"custom"`   | `writer(...)`에 넘긴 값 그대로        | 이 코드에서는 `{"node": "weather", "message": "Seoul 날씨 조회 시작"}` 같은 dict |
| `"updates"`  | `{노드이름: 그 노드가 반환한 상태 변경}` dict | 노드 하나가 끝날 때마다 한 번                                                  |
| `"messages"` | `(message_chunk, metadata)` 튜플 | LLM이 토큰을 낼 때마다 한 번                                                 |
#### custom
[ex01.py:11](demo/stream_mode/ex01.py#L11)의 `writer({"node": node, "message": message})`에 넘긴 객체가 그대로 옵니다. 형식은 정해져 있지 않아서 문자열을 넘기면 문자열이 옵니다. 이 예제에서는 도구가 실행되는 동안 "조회 시작", "조회 완료" 두 번 나옵니다.

#### updates
노드가 끝난 뒤 상태에 더한 부분만 담깁니다. 전체 상태가 아닙니다. `create_agent`의 노드 이름은 `model`과 `tools`라서 대략 이런 순서로 나옵니다.

```python
{"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Seoul"}, ...}])]}}
{"tools": {"messages": [ToolMessage(content="The weather in Seoul is sunny!", name="get_weather", ...)]}}
{"model": {"messages": [AIMessage(content="서울 날씨는 맑습니다.")]}}
```

#### messages
[ex01.py:51](demo/stream_mode/ex01.py#L51)처럼 2개짜리 튜플입니다.

- **`message`**: 보통 `AIMessageChunk`이고, `content`에 토큰 조각이 들어 있습니다. 도구 호출을 만드는 중에는 `content`가 비어 있고 `tool_call_chunks`에 인자 조각이 들어 있습니다. 그래서 [ex01.py:53](demo/stream_mode/ex01.py#L53)의 `if message.content:` 검사가 필요합니다. 도구 노드가 끝나면 `ToolMessage`도 이 모드로 한 번 나옵니다.
- **`metadata`**: 어느 노드에서 나온 메시지인지 알려 주는 dict입니다. `langgraph_node`(`"model"` 또는 `"tools"`), `langgraph_step`, `ls_model_name` 같은 키가 있습니다. 최종 답변 토큰만 출력하려면 `metadata["langgraph_node"] == "model"`로 걸러내면 됩니다.

#### 참고
- `stream_mode`를 문자열 하나(`"updates"`)로 주면 `mode` 없이 `chunk`만 나옵니다. 리스트로 줄 때만 `(mode, chunk)` 튜플이 됩니다.
- 로컬 LLM 서버가 토큰 스트리밍을 지원하지 않으면 `messages`는 토큰 단위가 아니라 완성된 메시지 한 덩어리로 나올 수 있습니다.

실제 값을 확인하려면 루프 안에 `print(mode, type(chunk), chunk)`를 넣어 보면 됩니다. `./model_run.sh`로 LLM 서버가 떠 있어야 합니다.

---
## 실전 팁

### 적절한 스트리밍 모드 선택

상황에 따라 적절한 스트리밍 모드를 선택하면 더 나은 사용자 경험과 성능을 얻을 수 있습니다. 일반적으로 사용자 인터페이스에서는 `messages` 모드를, 백엔드 모니터링에서는 `updates` 모드를 사용합니다. 복잡한 워크플로우에서는 여러 모드를 조합하여 사용할 수도 있습니다.

| 상황          | 권장 모드            | 설명                       |
| :---------- | :--------------- | :----------------------- |
| 채팅 인터페이스    | messages         | 실시간 텍스트 출력으로 자연스러운 대화 경험 |
| 백엔드 작업 모니터링 | updates + custom | 노드 진행과 세부 작업 상태 동시 추적    |
| 디버깅/로깅      | updates          | 에이전트 실행 흐름 분석            |
