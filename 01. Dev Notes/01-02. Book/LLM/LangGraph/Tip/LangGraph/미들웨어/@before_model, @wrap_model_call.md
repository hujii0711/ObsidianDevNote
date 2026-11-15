
둘 다 에이전트 미들웨어 훅이지만, `@before_model`은 모델 호출 직전에 상태를 손보는 훅이고 `@wrap_model_call`은 모델 호출 자체를 감싸서 제어하는 훅입니다.

## `@before_model`: 노드 스타일 훅

모델을 호출하기 직전에 실행됩니다. 현재 `state`를 읽고, 필요하면 상태 업데이트를 반환합니다.

```python
from langchain.agents.middleware import before_model, AgentState
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict | None:
    msgs = state["messages"]
    if len(msgs) > 10:
        # 상태 자체를 수정: 이후 단계와 체크포인트에도 반영됨
        return {"messages": [RemoveMessage(id=m.id) for m in msgs[:-10]]}
    return None
```

- 모델 호출 자체는 건드리지 못합니다.
- 반환한 값이 상태에 병합되어 영속됩니다.
- `{"jump_to": "end"}` 같은 흐름 제어도 가능합니다. 이때 `@before_model(can_jump_to=["end"])`처럼 선언해야 합니다.
- 그래프에서 독립된 노드로 실행됩니다.

## `@wrap_model_call`: 래퍼 스타일 훅

모델 호출을 감싸는 함수입니다. `handler`를 호출해야 실제 모델이 실행되므로, 호출 전후와 호출 여부를 모두 제어할 수 있습니다.

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def retry_and_fallback(request: ModelRequest, handler) -> ModelResponse:
    try:
        return handler(request)
    except Exception:
        # 이번 호출에만 다른 모델로 교체해서 재시도
        return handler(request.override(model=fallback_model))
```

- 모델, 도구 목록, 시스템 프롬프트, 메시지를 이번 호출에 한해 바꿀 수 있습니다. `request`는 불변이라 `request.override(...)`를 사용합니다.
- `handler`를 0번(캐시 히트로 단락), 1번(일반), 여러 번(재시도) 호출할 수 있습니다.
- 예외를 잡거나 응답을 후처리할 수 있습니다.
- 변경은 상태에 저장되지 않고 그 호출에만 적용됩니다.

## 비교

|          | `@before_model`                    | `@wrap_model_call`                    |
| -------- | ---------------------------------- | ------------------------------------- |
| 시그니처     | `(state, runtime) -> dict \| None` | `(request, handler) -> ModelResponse` |
| 대상       | 에이전트 **상태**                        | 모델 **호출(요청/응답)**                      |
| 변경의 지속성  | 상태에 반영(영속)                         | 해당 호출에만 적용(일시적)                       |
| 모델 호출 제어 | 불가                                 | 가능 (재시도, 스킵, 교체)                      |
| 예외 처리    | 불가                                 | 가능                                    |
| 흐름 제어    | `jump_to` 지원                       | 없음                                    |

## 선택 기준

- **`@before_model`**: 메시지 요약·트리밍, 입력 검증, 가드레일로 조기 종료, 호출 횟수 제한처럼 상태 자체를 바꾸거나 흐름을 끊을 때 씁니다.
- **`@wrap_model_call`**: 재시도, 모델 폴백, 동적 모델 선택, 응답 캐싱, 호출 단위 프롬프트나 도구 필터링, 지연 시간 측정처럼 호출을 감싸야 할 때 씁니다.

예를 들어 "대화가 길면 오래된 메시지를 지워서 상태를 줄인다"는 `before_model`이고, "이번 호출만 시스템 프롬프트를 바꾼다"나 "실패하면 다른 모델로 재시도한다"는 `wrap_model_call`입니다.

여러 미들웨어가 있으면 `before_*` 훅은 등록 순서대로, `wrap_*` 훅은 먼저 등록된 것이 바깥쪽에서 감싸는 양파 구조로 실행됩니다. 모델 호출 직후에 실행되는 짝으로 `@after_model`도 있고, 도구 호출을 감싸는 `@wrap_tool_call`도 있습니다.