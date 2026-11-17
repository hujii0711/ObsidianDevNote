
`MemorySaver`는 LangGraph에서 **에이전트의 실행 상태(State)를 메모리에 저장하고, 다음 실행에서 이전 상태를 복원할 수 있도록 해주는 Checkpointer(체크포인터)** 입니다.

쉽게 말하면 LangGraph 에이전트가 대화 도중 종료되거나 다음 요청을 받더라도, 이전 대화 내용을 기억하고 이어서 처리할 수 있도록 지원하는 기능입니다.

```
from langgraph.checkpoint.memory import MemorySaver
```

## **1. MemorySaver의 역할**

LangGraph에서 `MemorySaver`를 이해하려면 먼저 `State`, `Thread`, `Checkpoint` 세 가지 개념을 알아야 합니다.

|개념|설명|
|---|---|
|State|그래프 실행 중 관리하는 상태 데이터|
|Thread|하나의 대화 또는 작업 흐름을 구분하는 식별자|
|Checkpoint|특정 시점의 State를 저장한 스냅샷|

예를 들어 다음과 같은 대화를 생각해 보겠습니다.

**대화 흐름**

1. **첫 번째 요청**
    사용자: 내 이름은 홍길동이야.
    State 저장
    
2. **두 번째 요청**
    사용자: 내 이름이 뭐라고 했지?
    LangGraph가 이전 State를 복원합니다.
    
3. **응답**
    AI: 홍길동이라고 하셨습니다.

`MemorySaver`가 없으면 별도의 상태 관리가 없는 한 이전 실행의 대화 상태를 자동으로 이어갈 수 없습니다.

## **2. 기본 사용 방법**

가장 기본적인 사용 코드는 다음과 같습니다.

```
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class State(TypedDict):
    messages: str

def chatbot(state: State):
    return {
        "messages": state["messages"] + " 안녕하세요!"
    }

# 그래프 생성
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 메모리 체크포인터 생성
memory = MemorySaver()

# Checkpointer 등록
graph = builder.compile(checkpointer=memory)
```

여기서 핵심은 다음 코드입니다.

```
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

`compile()` 시점에 Checkpointer를 등록하면 그래프 실행 과정에서 State의 체크포인트를 저장하고 복원할 수 있습니다.

## **3. 핵심: thread_id를 사용한 대화 기억**

실제로 `MemorySaver`를 사용할 때 가장 중요한 부분은 `thread_id`입니다.

```
config = {
    "configurable": {
        "thread_id": "user-001"
    }
}
```

이 값이 동일해야 이전 대화의 State를 이어받을 수 있습니다.

실행 예제를 살펴보겠습니다.

```
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": state["messages"]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 동일한 대화방 ID
config = {"configurable": {"thread_id": "user-001"}}

# 첫 번째 대화
result1 = graph.invoke(
    {"messages": [HumanMessage(content="내 이름은 홍길동이야.")]},
    config
)

# 두 번째 대화
result2 = graph.invoke(
    {"messages": [HumanMessage(content="내 이름이 뭐야?")]},
    config
)

print(result2["messages"])
```

두 번째 실행에서도 첫 번째 대화의 메시지가 State에 남아 있습니다.

반대로 다음처럼 다른 `thread_id`를 사용하면 별도의 대화로 취급합니다.

```
config1 = {"configurable": {"thread_id": "user-001"}}
config2 = {"configurable": {"thread_id": "user-002"}}
```

즉, 여러 사용자의 대화를 분리할 수 있습니다.

## **4. MemorySaver의 저장 방식**

`MemorySaver`는 내부적으로 Python 프로세스의 RAM에 체크포인트를 저장합니다.

특징은 다음과 같습니다.

|항목|설명|
|---|---|
|저장 위치|Python 프로세스 메모리(RAM)|
|저장 단위|Thread별 Checkpoint|
|자동 저장|그래프 실행 단계별|
|프로세스 종료|저장 데이터 소멸|
|서버 재시작|이전 대화 복구 불가능|
|주요 용도|학습, 테스트, 프로토타입|

공식 문서에서도 InMemorySaver는 디버깅과 테스트 용도로 권장하고 있습니다.

## **5. MemorySaver와 다른 저장소 비교**

실제 AI 에이전트 서비스를 구축한다면 저장소 선택이 중요합니다.

|저장소|특징|추천 용도|
|---|---|---|
|MemorySaver|RAM 저장|학습 및 테스트|
|SqliteSaver|로컬 DB 저장|개인 프로젝트|
|PostgresSaver|PostgreSQL 저장|운영 서비스|
|AsyncPostgresSaver|비동기 PostgreSQL|비동기 운영 서비스|

## **6. MemorySaver와 장기 기억(Long-term Memory)의 차이**

이 부분은 특히 중요합니다.
LangGraph에는 크게 두 종류의 메모리가 있습니다.

**① Short-term Memory**
MemorySaver / Checkpointer
- 현재 대화의 메시지 기록
- 진행 중인 작업 상태
- 동일한 Thread 내 대화 유지

**② Long-term Memory**
Store
- 사용자별 선호 정보
- 장기적으로 기억할 사용자 정보
- 서로 다른 Thread 간 공유할 정보

예를 들어 다음과 같습니다.
- Short-term: "방금 검색한 주식 종목은 삼성전자야."
- Long-term: "이 사용자는 주식 분석 시 재무제표를 중요하게 생각해."

이 둘은 목적과 저장 방식이 다릅니다.

## **7. 실무적으로 추천하는 구조**

현재 학습 중인 LangGraph 기반 AI Agent를 실제 서비스로 발전시킨다면 다음 구조를 추천합니다.

**사용자 요청 (User Input)**

**LangGraph Agent**
StateGraph / Nodes / Tools

**Checkpointer**
대화 및 작업 상태 저장

**PostgreSQL**
Short-term Memory

**Store**
Long-term Memory

**정리하면 다음과 같습니다.**
- `MemorySaver`: LangGraph의 상태를 RAM에 저장하는 Checkpointer
- `thread_id`: 대화방을 구분하는 고유 식별자
- `Checkpoint`: 특정 시점의 그래프 상태 스냅샷
- `Store`: 서로 다른 대화에서도 활용할 수 있는 장기 기억 저장소

참고로 최신 LangGraph 문서에서는 `MemorySaver` 대신 `InMemorySaver`라는 이름을 사용하고 있으며, Python에서는 기존 `MemorySaver`가 하위 호환성을 위해 유지되는 별칭입니다.

**다음 학습 주제로는** `MemorySaver`**와** `InMemoryStore`**의 차이 및 실제 사용자별 장기 기억을 구현하는 방법을 이해하면 좋습니다.**