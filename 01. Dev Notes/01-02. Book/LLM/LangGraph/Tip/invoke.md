
## 한눈에 비교

|호출|대상|도구 실행|입력|반환|
|---|---|---|---|---|
|`llm.invoke`|모델 단독|안 함 (요청만 생성)|메시지 리스트 또는 문자열|`AIMessage`|
|`agent.invoke`|`create_agent`로 만든 에이전트|자동|`{"messages": [...]}`|state dict|
|`graph.invoke`|직접 만든 `StateGraph`|직접 구성한 대로|`{"messages": [...]}`|state dict|

공통으로 쓸 도구와 모델부터 준비합니다.

```python
from langchain.tools import tool
from langchain.messages import HumanMessage
from config.model import load_llm


@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}: 맑음, 18도"


tools = [get_weather]
llm = load_llm()
```

## 1. llm.invoke (모델 단독)

입력은 문자열이나 메시지 리스트이고, 반환은 `AIMessage` 하나입니다.

```python
# 도구 없이: 일반 대화
response = llm.invoke("대한민국의 수도는 어디야?")
print(response.content)  # 서울입니다.

# 도구 바인딩: 호출 요청만 생성, 실행은 안 됨
llm_with_tools = llm.bind_tools(tools)
ai_msg = llm_with_tools.invoke("서울 날씨 알려줘")
print(ai_msg.content)      # 보통 빈 문자열
print(ai_msg.tool_calls)   # [{'name': 'get_weather', 'args': {'city': 'Seoul'}, 'id': '...'}]
```

## 2. agent.invoke (create_agent)

도구 선택, 실행, 최종 답변까지 한 번에 처리하고, 결과는 `messages`가 담긴 dict입니다.

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=tools)

result = agent.invoke({"messages": [HumanMessage(content="서울 날씨 알려줘")]})

print(result["messages"][-1].content)  # 최종 답변: 서울은 맑고 18도입니다.

for msg in result["messages"]:         # 전체 과정 확인
    msg.pretty_print()
```

## 3. graph.invoke (직접 만든 LangGraph)

`StateGraph`로 노드와 엣지를 직접 구성한 경우입니다. 입출력 형태는 agent와 같습니다.

```python
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="서울 날씨 알려줘")]})

print(result["messages"][-1].content)

for msg in result["messages"]:
    msg.pretty_print()
```

## 세 결과 비교

```python
# llm.invoke  → AIMessage 하나
type(response)            # AIMessage
response.content

# agent.invoke / graph.invoke → dict
type(result)              # dict
result["messages"]        # [HumanMessage, AIMessage(tool_calls), ToolMessage, AIMessage(최종)]
result["messages"][-1].content   # 최종 답변
```

## 정리

- `llm.invoke`는 **모델 한 번 호출**이라 도구를 실행하지 않고, 반환이 `AIMessage` 하나입니다.
- `agent.invoke`와 `graph.invoke`는 **루프를 돌며** 도구 실행과 최종 답변까지 처리하고, 반환은 전체 메시지가 담긴 state dict입니다.
- 에이전트와 직접 만든 그래프는 사실상 같은 구조이고, `create_agent`는 위 3번의 그래프 구성을 대신 해 주는 함수입니다.

임포트 경로는 설치된 langchain, langgraph 버전에 따라 다를 수 있습니다.

---

## invoke, stream

`invoke`는 그래프가 끝까지 실행된 뒤 **최종 상태 하나**를 한 번에 돌려주고, `stream`은 실행되는 **도중의 중간 결과를 하나씩** 흘려보내 줍니다. 둘 다 같은 그래프를 같은 방식으로 실행하며, 결과를 받는 시점과 형태만 다릅니다.

|       | `graph.invoke()` | `graph.stream()`      |
| ----- | ---------------- | --------------------- |
| 반환    | 최종 상태(dict) 1개   | 제너레이터 (여러 조각을 순차 반환)  |
| 반환 시점 | 모든 노드 실행이 끝난 후   | 노드가 끝날 때마다 즉시         |
| 용도    | 결과만 필요할 때        | 진행 상황 표시, 디버깅, 실시간 UI |

앞서 만든 예제로 비교해 보면 다음과 같습니다.

```python
# invoke: 끝난 뒤 최종 결과만
result = graph.invoke({"messages": [HumanMessage(content="안녕")]})
print(result["messages"])          # Human + AI 2개가 한 번에

# stream: 단계마다 하나씩
for chunk in graph.stream({"messages": [HumanMessage(content="안녕")]}):
    print(chunk)                   # {'chatbot': {'messages': [AIMessage(...)]}}
```

`stream`은 `stream_mode`에 따라 내용이 달라집니다.

- `"values"`: 각 단계 **직후의 전체 상태**입니다. 입력 상태(step 0)부터 나옵니다.
- `"updates"`: **노드가 반환한 변경분만** `{노드이름: 반환값}` 형태로 나옵니다. 기본값이 이 모드입니다.
- `"messages"`: LLM이 생성하는 **토큰 단위 출력**입니다. 챗봇의 타이핑 효과에 씁니다.
- `"debug"`: 실행 과정의 상세 이벤트입니다.

참고로 `invoke`의 최종 결과는 `stream_mode="values"`의 마지막 chunk와 같습니다. 그래서 중간 과정이 필요 없으면 `invoke`가 더 간단하고, 노드가 여러 개이거나 LLM 호출이 길어서 사용자에게 진행 상황을 보여줘야 하면 `stream`이 적합합니다. 비동기 환경에서는 각각 `ainvoke`, `astream`을 쓰면 됩니다.