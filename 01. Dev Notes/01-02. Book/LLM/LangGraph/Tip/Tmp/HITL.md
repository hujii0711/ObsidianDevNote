
LangGraph의 Human-in-the-Loop(HITL) 기본 패턴은 `interrupt()`로 실행을 멈추고, `Command(resume=...)`로 재개하는 방식입니다. 체크포인터가 필수입니다.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    task: str
    approved: bool
    result: str


def propose(state: State):
    return {"task": "고객 3명에게 환불 메일 발송"}


def human_approval(state: State):
    # 여기서 실행이 멈추고, interrupt에 넘긴 값이 호출자에게 전달됨
    answer = interrupt({
        "question": "이 작업을 승인할까요?",
        "task": state["task"],
    })
    return {"approved": answer == "yes"}


def execute(state: State):
    if state["approved"]:
        return {"result": f"실행 완료: {state['task']}"}
    return {"result": "사용자가 거절함"}


builder = StateGraph(State)
builder.add_node("propose", propose)
builder.add_node("human_approval", human_approval)
builder.add_node("execute", execute)
builder.add_edge(START, "propose")
builder.add_edge("propose", "human_approval")
builder.add_edge("human_approval", "execute")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=MemorySaver())

# thread_id로 같은 실행을 식별
config = {"configurable": {"thread_id": "demo-1"}}

# 1) 첫 실행: interrupt에서 멈춤
result = graph.invoke({"task": "", "approved": False, "result": ""}, config)
print(result["__interrupt__"])  # 사람에게 보여줄 내용

# 2) 사람의 응답으로 재개
final = graph.invoke(Command(resume="yes"), config)
print(final["result"])  # 실행 완료: 고객 3명에게 환불 메일 발송
```

**핵심 포인트**
- `interrupt(value)`: 그래프를 일시 중지하고 `value`를 `__interrupt__`로 반환합니다.
- `Command(resume=값)`: 같은 `thread_id`로 호출하면 `interrupt()`의 반환값이 그 `값`이 되어 이어서 실행됩니다.
- 재개 시 `interrupt`가 있던 **노드는 처음부터 다시 실행**됩니다. 그래서 `interrupt` 앞에는 부작용(API 호출, DB 쓰기 등)을 두지 마세요.
- 운영 환경에서는 `MemorySaver` 대신 `SqliteSaver`/`PostgresSaver` 같은 영속 체크포인터를 사용하세요.

**응용: 승인 결과에 따라 분기하기**

```python
def human_approval(state: State) -> Command:
    answer = interrupt({"task": state["task"]})
    if answer == "yes":
        return Command(goto="execute", update={"approved": True})
    return Command(goto=END, update={"approved": False, "result": "거절됨"})
```

이때는 `add_edge("human_approval", "execute")` 엣지를 빼고 `Command`로 라우팅하면 됩니다.

설치: `pip install -U langgraph`

도구 호출 승인, 상태 수정 후 재개, 여러 interrupt 처리 같은 예제가 필요하면 말씀해 주세요.