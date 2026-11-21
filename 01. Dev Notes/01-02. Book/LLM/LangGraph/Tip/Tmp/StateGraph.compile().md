`StateGraph.compile()`은 키워드 인자 8개를 받습니다. 이 PC에 설치된 langgraph 1.2.14의 시그니처와 docstring에서 확인했습니다.

## 시그니처
```python
graph = builder.compile(
    checkpointer=None,          # 위치 인자로도 전달 가능
    *,
    cache=None,
    store=None,
    interrupt_before=None,
    interrupt_after=None,
    debug=False,
    name=None,
    transformers=None,
)
```

## 인자별 설명

| 인자 | 타입 | 설명 |
|---|---|---|
| `checkpointer` | `BaseCheckpointSaver` / `None` / `True` / `False` | 단기 메모리. State를 단계마다 저장해서 일시정지, 재개, 리플레이(time travel)를 할 수 있게 합니다. interrupt와 HITL을 쓰려면 필수입니다. |
| `store` | `BaseStore` | 장기 메모리. 여러 thread가 공유하는 저장소이고, 노드나 도구에서 `runtime.store`로 접근합니다. ch03/ex02에서 쓴 `InMemoryStore`가 이것입니다. |
| `cache` | `BaseCache` | 노드 실행 결과를 캐시합니다. 입력이 같으면 노드를 다시 실행하지 않습니다. `add_node(..., cache_policy=CachePolicy(ttl=...))`와 함께 써야 효과가 있습니다. |
| `interrupt_before` | `list[str]` / `"*"` | 지정한 노드를 실행하기 직전에 항상 멈춥니다. 정적 브레이크포인트입니다. |
| `interrupt_after` | `list[str]` / `"*"` | 지정한 노드를 실행한 직후에 항상 멈춥니다. |
| `debug` | `bool` | 디버그 모드. 실행 과정을 자세히 출력합니다. |
| `name` | `str` | 컴파일된 그래프의 이름. 서브그래프로 쓸 때나 LangSmith 추적, 스트리밍 이벤트에서 표시됩니다. |
| `transformers` | `Sequence[...]` | `stream_events(version="v3")`에서 쓰는 스트림 변환기. 고급 기능입니다. |

## `checkpointer` 값에 따른 동작
타입이 `None | bool | BaseCheckpointSaver`라서, 값에 따라 의미가 달라집니다.

| 값 | 동작 |
|---|---|
| `InMemorySaver()` 등 | 그 저장소를 사용합니다. 최상위 그래프에서 쓰는 일반적인 방식입니다. |
| `None` (기본) | 서브그래프로 쓰이면 부모의 체크포인터를 상속하고, 단독으로 쓰면 저장하지 않습니다. ex01_01의 에이전트가 이 경우입니다. |
| `True` | 서브그래프가 부모의 체크포인터를 쓰되, 자기 State를 호출 간에 따로 누적합니다. 서브그래프에 독립적인 대화 기억이 필요할 때 씁니다. |
| `False` | 체크포인터를 쓰지도 상속하지도 않습니다. |

## `interrupt_before`/`after`: compile과 invoke의 차이
두 곳 모두에 있는 인자입니다.
- **`compile()`에 지정하면:** 그 그래프를 실행할 때마다 항상 멈춥니다. 그래프의 기본 설정이 됩니다.
- **`invoke()`에 지정하면:** 그 호출에서만 적용됩니다.

이 방식은 노드 단위로 무조건 멈추는 정적 브레이크포인트입니다. ex01_01의 HITL 미들웨어는 노드 안에서 `interrupt()`를 호출하는 동적 방식이라서, 조건(어떤 도구인지)에 따라 멈출지 말지를 정할 수 있습니다. 사람의 승인을 받는 용도라면 동적 방식을 권장합니다.

## 종합 예시
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

builder.add_node("summarize_result", summarize_result,
                 cache_policy=CachePolicy(ttl=60))   # 60초 동안 캐시

graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
    cache=InMemoryCache(),
    interrupt_before=["agent"],   # agent 노드 실행 전에 항상 멈춤
    name="hitl_file_graph",
)
```

## `create_agent()`와의 관계
`create_agent()`는 내부에서 `compile()`을 호출합니다. 그래서 `checkpointer`, `store`, `cache`, `interrupt_before`, `interrupt_after`, `debug`, `name`, `transformers`를 `create_agent()`에 바로 넘길 수 있습니다. ch03/ex02에서 `create_agent(..., store=store)`로 넘긴 것이 이 경로입니다.