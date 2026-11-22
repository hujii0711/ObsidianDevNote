
`langgraph_supervisor`의 `create_supervisor`를 쓰면 **이름 붙인 워커 에이전트 목록과 LLM만 넘겨서** 슈퍼바이저 그래프를 만들 수 있습니다. 슈퍼바이저가 요청을 분석해 적절한 워커에게 일을 넘기고, 워커가 끝나면 제어가 다시 슈퍼바이저로 돌아와 결과를 모아 답합니다.

## 1. 설치

```bash
pip install langgraph langgraph-supervisor langchain-openai
export OPENAI_API_KEY=sk-...
```

## 2. 기본 예제: 리서치 + 수학 에이전트

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

model = ChatOpenAI(model="gpt-4o", temperature=0)

# ── 1) 워커 에이전트의 도구 정의 ──────────────────────
def add(a: float, b: float) -> float:
    """두 수를 더한다."""
    return a + b

def multiply(a: float, b: float) -> float:
    """두 수를 곱한다."""
    return a * b

def web_search(query: str) -> str:
    """웹에서 정보를 검색한다."""
    return (
        "2024년 FAANG 직원 수: Meta 67,317 / Apple 164,000 / "
        "Amazon 1,551,000 / Netflix 14,000 / Google 181,269"
    )

# ── 2) 워커 에이전트 생성 (name은 필수!) ─────────────
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="너는 수학 전문가다. 계산은 반드시 도구를 사용하라.",
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="너는 리서처다. 계산은 하지 말고 검색만 수행하라.",
)

# ── 3) 슈퍼바이저 생성 ───────────────────────────────
workflow = create_supervisor(
    agents=[research_agent, math_agent],
    model=model,
    prompt=(
        "너는 팀을 관리하는 슈퍼바이저다.\n"
        "- 최신 정보/사실 조사는 research_expert에게\n"
        "- 계산은 math_expert에게 맡겨라.\n"
        "직접 작업하지 말고 반드시 위임한 뒤 결과를 종합해 답하라."
    ),
)

# ── 4) 컴파일 & 실행 ────────────────────────────────
app = workflow.compile()

result = app.invoke({
    "messages": [
        {"role": "user", "content": "2024년 FAANG 전체 직원 수 합계는?"}
    ]
})

for m in result["messages"]:
    m.pretty_print()
```

### 내부 동작

1. `create_supervisor`가 워커마다 `transfer_to_<agent_name>` 핸드오프 도구를 자동으로 만들어 슈퍼바이저 LLM에 붙입니다.
2. 슈퍼바이저가 이 도구를 호출하면 해당 워커 노드로 이동합니다.
3. 워커가 끝나면 `transfer_back_to_supervisor` 메시지와 함께 제어가 슈퍼바이저로 돌아옵니다.
4. 슈퍼바이저가 더 넘길 일이 없다고 판단하면 최종 답을 내고 종료합니다.

## 3. 주요 옵션

```python
workflow = create_supervisor(
    agents=[research_agent, math_agent],
    model=model,
    prompt="...",
    tools=[...],                    # 슈퍼바이저가 직접 쓸 도구(커스텀 핸드오프 포함)
    output_mode="last_message",     # "full_history"(기본): 워커의 모든 메시지 포함
                                    # "last_message": 워커의 최종 답변만 포함 → 토큰 절약
    add_handoff_back_messages=True, # 복귀 시 핸드오프 메시지 추가 여부
    supervisor_name="supervisor",
    parallel_tool_calls=False,      # True면 여러 워커에 동시에 위임 가능(모델 지원 시)
    include_agent_name="inline",    # 메시지에 에이전트 이름 표시(이름 필드 미지원 모델용)
)
```

## 4. 메모리(대화 유지) 추가

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

app = workflow.compile(
    checkpointer=InMemorySaver(),   # 단기 메모리 (thread 단위)
    store=InMemoryStore(),          # 장기 메모리
)

config = {"configurable": {"thread_id": "user-1"}}
app.invoke({"messages": [{"role": "user", "content": "Apple 직원 수는?"}]}, config)
app.invoke({"messages": [{"role": "user", "content": "거기에 2를 곱하면?"}]}, config)
```

## 5. 핸드오프 도구 커스터마이징

```python
from langgraph_supervisor import create_handoff_tool
from langgraph_supervisor.handoff import create_forward_message_tool

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    tools=[
        create_handoff_tool(
            agent_name="math_expert",
            name="assign_to_math_expert",
            description="수학 계산이 필요할 때 math_expert에게 작업을 맡긴다.",
        ),
        create_handoff_tool(
            agent_name="research_expert",
            name="assign_to_research_expert",
            description="정보 검색이 필요할 때 research_expert에게 작업을 맡긴다.",
        ),
        # 워커의 답변을 슈퍼바이저가 다시 요약하지 않고 그대로 전달
        create_forward_message_tool("supervisor"),
    ],
)
```

`create_forward_message_tool`을 쓰면 슈퍼바이저가 워커 답변을 다시 풀어 쓰는 과정을 건너뜁니다. 토큰이 절약되고, 바꿔 쓰다가 내용이 틀어지는 문제도 막을 수 있습니다.

## 6. 계층형(Hierarchical) 구조: 슈퍼바이저의 슈퍼바이저

컴파일된 슈퍼바이저도 하나의 에이전트처럼 다른 슈퍼바이저 밑에 넣을 수 있습니다.

```python
research_team = create_supervisor(
    [research_agent, math_agent],
    model=model,
    supervisor_name="research_supervisor",
).compile(name="research_team")

writing_team = create_supervisor(
    [writer_agent, editor_agent],
    model=model,
    supervisor_name="writing_supervisor",
).compile(name="writing_team")

top = create_supervisor(
    [research_team, writing_team],
    model=model,
    supervisor_name="top_level_supervisor",
).compile(name="top_level_supervisor")
```

## 7. 스트리밍으로 흐름 확인

```python
for chunk in app.stream(
    {"messages": [{"role": "user", "content": "FAANG 직원 수 합계는?"}]},
    subgraphs=True,   # 워커 내부 단계까지 출력
):
    print(chunk)
```

## 실무 팁과 최신 상황

- **워커의 `name`이 라우팅 키입니다.** 슈퍼바이저 프롬프트에 그 이름과 담당 업무를 분명히 써야 라우팅이 정확해집니다.
- 워커 프롬프트에는 "자기 일만 하라"는 제약을 넣어야 역할이 겹치지 않습니다.
- 워커가 많거나 대화가 길면 `output_mode="last_message"`로 컨텍스트를 줄이세요.
- **API 변경 주의:** LangChain v1부터는 `create_react_agent` 대신 `langchain.agents.create_agent`가 최신 API로 안내되고 있습니다. 위키독스 자료에서도 create_agent를 create_react_agent의 최신 대체 API로 소개합니다. 또 langgraph-supervisor와 create_react_agent는 이 패턴을 배우기에는 좋지만 LangChain이 더 새로운 방식으로 옮겨가고 있으니, 새 프로젝트라면 사용 전에 최신 문서를 확인하라는 지적도 있습니다. 공식 문서는 워커 에이전트를 도구로 감싸서 슈퍼바이저 에이전트에 붙이는 "tool-calling supervisor" 방식을 함께 권장하니, 새로 시작한다면 두 방식을 비교해 보세요.

원하시면 같은 예제를 `create_agent`와 tool-calling 방식으로 다시 짜 드릴 수 있습니다.