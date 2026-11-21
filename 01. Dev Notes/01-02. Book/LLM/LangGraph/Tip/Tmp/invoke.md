
`agent.invoke`와 `graph.invoke`는 같은 메서드라서, 쓸 수 있는 인자가 완전히 같습니다. `create_agent()`가 반환하는 것도 `CompiledStateGraph`이고, 둘 다 `Pregel.invoke`를 그대로 씁니다. 다른 점은 입력(`input`)에 어떤 State를 넣느냐뿐입니다.

아래는 이 PC에 설치된 langgraph 1.2.14 / langchain 1.4.4의 실제 시그니처와 docstring에서 확인한 내용입니다. 패키지는 옆 폴더의 `LangGraph/.venv`에 있었습니다.

## 시그니처
```python
invoke(
    input,                      # 필수: State 입력 또는 Command
    config=None,                # RunnableConfig
    *,
    context=None,
    stream_mode="values",
    print_mode=(),
    output_keys=None,
    interrupt_before=None,
    interrupt_after=None,
    durability=None,
    control=None,
    version="v1",
    **kwargs,
)
```

## 인자별 설명

| 인자 | 설명 | 예시 |
|---|---|---|
| `input` | 그래프 State 입력(dict), `Command`, 또는 `None` | `{"messages": [...]}`, `Command(resume=...)` |
| `config` | 실행 설정 (아래 표 참고) | `{"configurable": {"thread_id": "1"}}` |
| `context` | 실행 중에 바뀌지 않는 정적 컨텍스트. 도구나 노드에서 `runtime.context`로 꺼내 씁니다. `context_schema` 타입에 맞춰 넣습니다. | `context=Context(user_id="u1")` |
| `stream_mode` | 반환 형식. 기본값 `"values"`는 최종 State를 반환하고, 다른 값이면 청크 리스트를 반환합니다. | `"values"`, `"updates"`, `"messages"`, `"custom"`, `"checkpoints"`, `"tasks"`, `"debug"` |
| `print_mode` | `stream_mode`와 같은 값을 받지만 콘솔에 출력만 합니다. 디버깅용이고 반환값에는 영향이 없습니다. | `print_mode="updates"` |
| `output_keys` | 반환받을 State 키를 지정합니다. | `output_keys=["summary"]` |
| `interrupt_before` | 지정한 노드를 실행하기 직전에 멈춥니다. 정적 브레이크포인트이고 체크포인터가 필요합니다. | `["tools"]`, `"*"` |
| `interrupt_after` | 지정한 노드를 실행한 직후에 멈춥니다. | `["agent"]` |
| `durability` | 체크포인트 저장 시점 | `"sync"`: 다음 단계 전에 저장, `"async"`(기본): 다음 단계와 병렬로 저장, `"exit"`: 종료 시에만 저장 |
| `control` | 실행을 협조적으로 멈추게(drain) 요청하는 제어 객체. 고급 기능이라 일반적으로는 쓰지 않습니다. | |
| `version` | 스트리밍 출력 형식. `"v2"`는 `stream_mode`가 `"values"`가 아닐 때 `StreamPart` 타입 dict를 반환합니다. | `"v1"`(기본), `"v2"` |

## `config`(RunnableConfig)에 넣을 수 있는 키

| 키 | 설명 |
|---|---|
| `configurable` | `thread_id`(대화 구분), `checkpoint_id`(특정 시점부터 다시 실행), 사용자 정의 값 |
| `recursion_limit` | 최대 실행 단계 수. 기본값은 25이고, 넘으면 `GraphRecursionError`가 납니다. |
| `tags`, `metadata` | LangSmith 추적용 태그와 메타데이터 |
| `callbacks` | 콜백 핸들러 |
| `run_name`, `run_id` | 실행 이름과 ID |
| `max_concurrency` | 병렬로 실행할 노드의 최대 개수 |

## 입력(`input`)의 차이

| | `agent.invoke` (create_agent) | `graph.invoke` (직접 만든 StateGraph) |
|---|---|---|
| State | `AgentState`: `messages`가 핵심이고, `state_schema`로 확장할 수 있습니다. | 직접 정의한 State. 예: ex01_01의 `task`, `messages`, `summary` |
| 입력 예 | `{"messages": [{"role": "user", "content": "..."}]}` | `{"task": "hello.txt 작성..."}` |
| 결과 | `result["messages"]`, 응답 형식을 지정했다면 `result["structured_response"]` | 정의한 State 전체 |

## 종합 예시
```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "안녕"}]},
    config={
        "configurable": {"thread_id": "t1"},
        "recursion_limit": 50,
        "tags": ["study"],
    },
    context=Context(user_id="user_123"),
    durability="sync",
    print_mode="updates",   # 진행 과정을 콘솔에 출력
)
```

**참고:** 중간 과정을 실시간으로 받으려면 `invoke` 대신 `stream`을 쓰세요. `stream`은 위 인자에 더해 `subgraphs=True`(서브그래프 이벤트 포함)와 `debug`도 받습니다. 비동기 버전은 `ainvoke`와 `astream`입니다.