
## 한 줄 요약

- **`@tool`**: 함수를 LLM이 호출할 수 있는 **도구 정의**로 만드는 데코레이터
- **`ToolNode`**: 모델이 요청한 도구 호출을 **실제로 실행**하는 LangGraph의 **노드**

`@tool`은 "무엇을 할 수 있는가"를 정의하고, `ToolNode`는 "그것을 언제, 어떻게 실행하는가"를 담당합니다.

## @tool

함수에 이름, 설명(docstring), 인자 스키마를 붙여 `BaseTool` 객체로 바꿉니다. 모델은 이 스키마를 보고 어떤 도구를 어떤 인자로 호출할지 결정합니다.

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

# 단독으로도 호출 가능
print(get_weather.invoke({"city": "Seoul"}))

# 모델에 도구 스키마를 알려줌 (실행은 하지 않음)
model_with_tools = model.bind_tools([get_weather])
ai_msg = model_with_tools.invoke("서울 날씨 알려줘")
print(ai_msg.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Seoul'}, 'id': 'call_xxx'}]
```

여기서 `bind_tools`는 모델이 **호출 요청을 생성**할 뿐, 실제 함수 실행은 일어나지 않습니다.

## ToolNode

`AIMessage`의 `tool_calls`를 읽어서 해당 도구를 실행하고, 결과를 `ToolMessage`로 만들어 state에 추가하는 그래프 노드입니다. 여러 도구 호출이 있으면 병렬로 실행하고, 도구에서 발생한 예외 처리도 해 줍니다.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_weather])

result = tool_node.invoke({"messages": [ai_msg]})
print(result["messages"])
# [ToolMessage(content='서울: 맑음, 18도', tool_call_id='call_xxx')]
```

## 그래프에서 함께 쓰는 전형적인 구조

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

tools = [get_weather]
model_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))   # 도구 실행 노드

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # tool_calls 있으면 tools, 없으면 END
builder.add_edge("tools", "agent")

graph = builder.compile()
```

흐름은 `agent(모델이 호출 요청 생성)` → `tools(ToolNode가 실행)` → `agent(결과를 보고 답변)`입니다.

## 비교표

| 구분      | `@tool`                  | `ToolNode`              |
| ------- | ------------------------ | ----------------------- |
| 정체      | 도구 객체(`BaseTool`)        | LangGraph 노드            |
| 역할      | 도구의 이름, 설명, 인자 스키마 정의    | 모델의 `tool_calls`를 실제 실행 |
| 입력      | 함수 인자                    | `AIMessage`가 담긴 state   |
| 출력      | 함수 반환값                   | `ToolMessage` 리스트       |
| 모델과의 관계 | `bind_tools`로 모델에 스키마 전달 | 모델이 만든 호출 요청을 처리        |
| 필요한 곳   | 도구를 만들 때 항상              | 직접 그래프를 구성할 때           |

## 참고

- `create_agent`(또는 `create_react_agent`)를 쓰면 내부에서 `ToolNode`를 자동으로 만들어 주므로, `tools=[get_weather]`처럼 `@tool` 함수만 넘기면 됩니다. `ToolNode`를 직접 쓰는 경우는 LangGraph로 그래프를 손수 구성할 때입니다.
- `ToolNode`는 `@tool` 없이 일반 함수를 넘겨도 내부에서 도구로 변환해 주지만, 설명과 스키마를 명확히 하려면 `@tool`을 붙이는 편이 좋습니다.
- 임포트 경로(`langgraph.prebuilt`, `langchain.tools`)는 버전에 따라 달라질 수 있으니 설치된 버전을 확인해 보세요.

---

## 한 줄 요약

세 가지는 도구 호출 파이프라인의 **정의 → 전달 → 실행** 세 단계를 각각 맡습니다.

|단계|구성 요소|역할|
|---|---|---|
|1. 정의|`@tool`|함수를 이름, 설명, 인자 스키마가 있는 도구 객체로 변환|
|2. 전달|`llm.bind_tools()`|모델에 "이런 도구들이 있다"고 알려줌 (실행 X, 호출 **요청**만 생성)|
|3. 실행|`ToolNode`|모델의 `tool_calls`를 읽고 도구를 **실제 실행**, `ToolMessage` 반환|

(메서드 이름은 `bind_tool`이 아니라 `bind_tools`입니다.)

## 전체 흐름

```
@tool 로 도구 정의
      ↓
llm.bind_tools([tools])  → 모델이 스키마를 인식
      ↓
사용자 질문 → 모델이 AIMessage(tool_calls=[...]) 생성   ← 실행 아님, 요청일 뿐
      ↓
ToolNode 가 tool_calls 를 읽고 실제 함수 실행 → ToolMessage 생성
      ↓
ToolMessage 를 모델에 다시 전달 → 최종 답변 생성
```

## 1단계: @tool (정의)

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

@tool
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    return a + b

tools = [get_weather, add]
```

docstring과 타입 힌트가 모델이 보는 설명과 스키마가 되므로 꼭 작성해야 합니다.

## 2단계: llm.bind_tools (전달)

```python
from config.model import load_llm

llm = load_llm()
llm_with_tools = llm.bind_tools(tools)

ai_msg = llm_with_tools.invoke("서울 날씨 알려줘")
print(ai_msg.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Seoul'}, 'id': 'call_abc'}]
print(ai_msg.content)  # 보통 빈 문자열
```

이 시점에는 `get_weather`가 **아직 실행되지 않았습니다.** 모델이 "이 도구를 이 인자로 호출해 달라"는 요청을 만들었을 뿐입니다.

## 3단계: ToolNode (실행)

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

result = tool_node.invoke({"messages": [ai_msg]})
print(result["messages"])
# [ToolMessage(content='Seoul: 맑음, 18도', name='get_weather', tool_call_id='call_abc')]
```

`ToolNode`는 `tool_calls`에 적힌 이름으로 도구를 찾아 실행하고, 여러 호출은 병렬로 처리하며, `tool_call_id`를 맞춰 `ToolMessage`를 만들어 줍니다.

## 세 가지를 합친 완성 예제 (LangGraph)

```python
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage
from config.model import load_llm

# 1. 정의
@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

tools = [get_weather]

# 2. 전달
llm_with_tools = load_llm().bind_tools(tools)

def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 3. 실행
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # tool_calls 있으면 tools, 없으면 END
builder.add_edge("tools", "agent")                        # 결과를 들고 다시 모델로

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="서울 날씨 알려줘")]})
for msg in result["messages"]:
    msg.pretty_print()
```

출력되는 메시지 순서는 `HumanMessage` → `AIMessage(tool_calls)` → `ToolMessage` → `AIMessage(최종 답변)`입니다.

## 헷갈리기 쉬운 부분

- `bind_tools`만 하고 `ToolNode`가 없으면 모델이 호출 요청만 만들고 끝납니다. 도구는 실행되지 않습니다.
- `ToolNode`만 있고 `bind_tools`가 없으면 모델이 도구의 존재를 몰라서 `tool_calls`를 만들지 못합니다.
- `ToolNode`에 넣는 `tools`와 `bind_tools`에 넣는 `tools`는 **같은 리스트**를 써야 이름이 맞습니다.

## create_agent를 쓰면 한 번에 해결

2단계와 3단계, 그리고 위의 그래프 구성을 모두 자동으로 해 줍니다. `@tool`로 정의한 함수만 넘기면 됩니다.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],   # 내부에서 bind_tools + ToolNode 자동 구성
)

result = agent.invoke({"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]})
```

|방식|`@tool`|`bind_tools`|`ToolNode`|
|---|---|---|---|
|`create_agent` 사용|직접 작성|자동|자동|
|LangGraph 직접 구성|직접 작성|직접 호출|직접 추가|
|모델 단독 사용|직접 작성|직접 호출|없음 (결과를 직접 실행해야 함)|

임포트 경로와 일부 동작은 설치된 langchain, langgraph 버전에 따라 다를 수 있습니다.

---
아니요, **세 가지가 모두 있어야 하는 것은 아닙니다.** 도구를 "실행"하는 데 꼭 필요한 건 `@tool`(또는 호출 가능한 함수)뿐이고, <mark style="background:#fff88f">`bind_tools`와 `ToolNode`는 **모델이 도구 호출을 자동으로 결정하고 처리하게 만들 때** 필요합니다.</mark>

## 각 조합별 동작

|조합|도구 실행 여부|설명|
|---|---|---|
|`@tool`만|직접 호출하면 실행됨|`get_weather.invoke({"city": "Seoul"})` 처럼 사람이 직접 부를 때|
|`@tool` + `bind_tools`|실행 안 됨|모델이 호출 **요청**(`tool_calls`)만 생성|
|`@tool` + `ToolNode`|실행됨 (단, 요청이 있어야 함)|`tool_calls`가 담긴 `AIMessage`를 직접 만들어 주면 실행 가능. 모델이 요청을 만들지는 못함|
|`@tool` + `bind_tools` + `ToolNode`|모델이 요청 → 자동 실행|일반적인 에이전트 구조|

## 예제로 확인

**1. `@tool`만: 직접 호출하면 실행됨**

```python
@tool
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    return a + b

print(add.invoke({"a": 1, "b": 2}))  # 3
```

**2. `bind_tools`만: 요청만 생성, 실행 안 됨**

```python
llm_with_tools = llm.bind_tools([add])
ai_msg = llm_with_tools.invoke("1+2는?")
print(ai_msg.tool_calls)  # [{'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': '...'}]
# add는 실행되지 않음
```

**3. `bind_tools` + 수동 실행: ToolNode 없이도 가능**

```python
ai_msg = llm_with_tools.invoke("1+2는?")
for call in ai_msg.tool_calls:
    result = add.invoke(call["args"])   # 직접 실행
    print(result)  # 3
```

**4. `ToolNode`만: 모델 없이도 실행 가능**

```python
from langchain.messages import AIMessage
from langgraph.prebuilt import ToolNode

fake_msg = AIMessage(
    content="",
    tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
)
print(ToolNode([add]).invoke({"messages": [fake_msg]}))
# ToolMessage(content='3', ...)
```

## 정리

- **실행 주체는 `@tool`로 만든 함수 자체**입니다. `ToolNode`는 그것을 대신 호출해 주는 편의 장치입니다.
- **`bind_tools` + `ToolNode`가 함께 있어야 하는 경우**는 "모델이 알아서 도구를 고르고 자동으로 실행까지" 하는 에이전트 루프를 직접 만들 때입니다.
- `bind_tools` 없이 `ToolNode`만 쓰면 모델이 도구를 모르므로 호출 요청이 생기지 않고, `ToolNode` 없이 `bind_tools`만 쓰면 요청은 생기지만 실행은 직접 코드로 해야 합니다.
- `create_agent`를 쓰면 `bind_tools`와 `ToolNode`가 내부에서 자동 구성되므로 `@tool` 함수만 넘기면 됩니다.

---
맞습니다. **`@tool`만 사용하면 LLM은 관여하지 않습니다.**

`@tool`은 함수를 "도구 객체"로 포장할 뿐이라, 이 객체를 호출해도 그냥 파이썬 함수를 실행하는 것과 같습니다. 모델 호출도, 응답 생성도 일어나지 않습니다.

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

print(get_weather.invoke({"city": "Seoul"}))
# Seoul: 맑음, 18도   ← 함수가 반환한 값 그대로, LLM 없음
```

## LLM이 관여하는 시점

|구성|LLM 관여|결과|
|---|---|---|
|`@tool`만|없음|함수 반환값만 나옴|
|`@tool` + `bind_tools`|있음|모델이 "이 도구를 이 인자로 호출해줘"라는 `tool_calls` 생성|
|`@tool` + `bind_tools` + `ToolNode` + 모델 재호출|있음|도구 결과를 보고 모델이 **자연어 최종 답변** 생성|

## 정리

- LLM이 하는 일은 **어떤 도구를 어떤 인자로 호출할지 결정**하는 것과, **도구 결과를 바탕으로 자연어 답변을 만드는 것** 두 가지입니다.
- 둘 다 `bind_tools`(또는 `create_agent`)를 통해 모델에 도구가 연결되어 있어야 일어납니다.
- `@tool`만 쓴 경우 "서울 날씨는 맑고 18도입니다" 같은 문장형 응답이 아니라, 함수가 반환한 `"Seoul: 맑음, 18도"` 문자열이 그대로 나옵니다.

---
아니요, **세 가지를 각각 직접 쓰는 것이 필수는 아닙니다.** 다만 "모델이 도구를 골라 실행하고 그 결과로 응답"하려면 **세 가지 기능(정의, 전달, 실행)이 어떤 형태로든 갖춰져야** 합니다.

## 기능별 필수 여부

| 기능                | 필요한 이유                   | 대체 가능한 방법                                      |
| ----------------- | ------------------------ | ---------------------------------------------- |
| 정의 (`@tool`)      | 모델이 볼 이름, 설명, 스키마가 필요    | 일반 함수(타입 힌트+docstring), dict 스키마, Pydantic 클래스 |
| 전달 (`bind_tools`) | 모델이 도구의 존재를 알아야 함        | `create_agent(tools=[...])`가 대신 처리             |
| 실행 (`ToolNode`)   | `tool_calls`를 실제로 실행해야 함 | `for` 루프로 직접 `tool.invoke()` 실행                |

## 1. 가장 간단한 방법: create_agent (세 가지를 직접 안 써도 됨)

`@tool`만 직접 만들고, `bind_tools`와 `ToolNode`는 내부에서 자동 처리됩니다.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

agent = create_agent(model="openai:gpt-4.1", tools=[get_weather])

result = agent.invoke({"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]})
print(result["messages"][-1].content)  # 서울은 맑고 18도입니다.
```

## 2. ToolNode 없이 직접 구성 (bind_tools + 수동 실행)

```python
from langchain.messages import HumanMessage, ToolMessage

llm_with_tools = llm.bind_tools([get_weather])
tool_map = {"get_weather": get_weather}

messages = [HumanMessage(content="서울 날씨 알려줘")]

ai_msg = llm_with_tools.invoke(messages)          # 모델이 호출 요청 생성
messages.append(ai_msg)

for call in ai_msg.tool_calls:                    # ToolNode 대신 직접 실행
    result = tool_map[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = llm_with_tools.invoke(messages)           # 결과를 보고 최종 답변
print(final.content)
```

## 3. @tool 없이 일반 함수만 사용

`create_agent`는 일반 함수도 도구로 받아 주지만, docstring과 타입 힌트는 반드시 있어야 합니다.

```python
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"

agent = create_agent(model="openai:gpt-4.1", tools=[get_weather])
```

## 정리

- **정말 필수인 것**: 도구 정의(이름, 설명, 스키마), 모델에 도구 전달, 호출 요청의 실제 실행, 결과를 모델에 다시 전달.
- **`@tool`, `bind_tools`, `ToolNode`라는 특정 이름은 필수가 아닙니다.** 같은 일을 하는 다른 방법이 있습니다.
- 실무에서는 `create_agent`를 쓰는 것이 가장 간단하고, 그래프 흐름을 세밀하게 제어하고 싶을 때만 `bind_tools` + `ToolNode`를 직접 구성하면 됩니다.

버전에 따라 일반 함수 지원 여부나 임포트 경로가 다를 수 있으니 설치된 버전을 확인해 보세요.

---

## model.bind_tools

한 줄로 말하면, `llm_with_tools`는 도구 호출을 제안만 하는 모델이고 `agent`는 도구를 실제로 실행해 답까지 돌려주는 실행 루프입니다.

## `llm_with_tools`: 모델 한 번 호출

`bind_tools`는 모델에 "이런 도구가 있다"는 스키마만 알려줍니다. 호출하면 모델은 도구를 직접 실행하지 않고, `AIMessage.tool_calls`에 호출 요청만 담아 돌려줍니다.

```python
ai_msg = llm_with_tools.invoke("인사해줘")
ai_msg.tool_calls
# [{"name": "personalized_greeting", "args": {}, "id": "call_abc"}]
```

도구 실행, 결과를 `ToolMessage`로 되돌려주기, 모델 재호출 같은 나머지는 직접 짜야 합니다.

## `agent`: 루프까지 포함한 실행기

`create_agent`는 아래 과정을 알아서 반복합니다.

```
모델 호출 → tool_calls 있음? → 도구 실행 → 결과를 메시지에 추가 → 모델 재호출 → ... → tool_calls 없음 → 최종 답변
```

그 밖에 `context_schema`로 런타임 컨텍스트를 주입하고, `system_prompt`를 적용하고, 미들웨어(HITL 등)와 체크포인터(대화 상태 유지)를 붙일 수 있습니다.

## 지금 코드에서 중요한 차이

`personalized_greeting`은 `runtime: ToolRuntime[UserContext]`를 받습니다. 이 `runtime`은 에이전트가 도구를 실행할 때 `context`를 넣어 주입해 줍니다.

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "안녕!"}]},
    context=UserContext(user_name="철수"),
)
```

반면 `llm_with_tools`로 도구를 수동 실행하면 이 주입이 일어나지 않아 `runtime.context`를 쓸 수 없습니다. 그래서 `ToolRuntime`을 쓰는 도구는 사실상 에이전트(또는 LangGraph의 `ToolNode`)와 함께 써야 합니다.

## 쓰임새 정리

|`llm_with_tools`|`agent`|
|---|---|---|
|도구 실행|안 함 (요청만 생성)|자동 실행|
|반복 루프|없음 (1회 호출)|있음|
|`runtime`/`context` 주입|불가|가능|
|적합한 용도|분류, 라우팅, 구조화된 추출, 커스텀 그래프의 LLM 노드|일반적인 도구 사용 작업, HITL, 상태 유지|

- **`llm_with_tools`**: 모델의 판단만 필요하거나, LangGraph에서 호출 흐름을 직접 제어할 때 씁니다. 예를 들어 "이 질문에 어떤 도구가 필요한가"만 판단하게 할 수 있습니다.
- **`agent`**: 대부분의 도구 사용 시나리오에서 기본 선택입니다.

## 코드 관련 참고

- `bind_tools`는 도구를 리스트로 받으므로 `model.bind_tools([personalized_greeting], parallel_tool_calls=False)`로 써야 합니다. 지금처럼 단일 도구를 그대로 넘기면 오류가 날 수 있습니다.
- `parallel_tool_calls=False`는 한 번에 도구를 하나씩만 호출하게 하는 옵션이며, 이를 지원하는 모델(OpenAI 계열 등)에서만 동작합니다.
- `llm_with_tools`는 위 코드에서 `agent`와 연결되어 있지 않아서, 지금 상태로는 따로 쓰이지 않습니다.