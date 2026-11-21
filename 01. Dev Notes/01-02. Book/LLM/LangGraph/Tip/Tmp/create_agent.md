
`create_agent()`는 인자를 14개 받습니다. 에이전트 자체를 정의하는 인자 7개와, 내부의 `compile()`로 그대로 전달되는 인자 7개입니다. 이 PC에 설치된 langchain 1.4.4의 시그니처와 docstring에서 확인했습니다.

## 시그니처
```python
agent = create_agent(
    model,                  # 필수
    tools=None,
    *,
    system_prompt=None,
    middleware=(),
    response_format=None,
    state_schema=None,
    context_schema=None,
    # ↓ compile()로 전달되는 인자
    checkpointer=None,
    store=None,
    interrupt_before=None,
    interrupt_after=None,
    debug=False,
    name=None,
    cache=None,
    transformers=None,
)
```

## 1. 에이전트를 정의하는 인자

| 인자 | 타입 | 설명 |
|---|---|---|
| `model` (필수) | `str` 또는 `BaseChatModel` | 사용할 LLM. `"openai:gpt-5.5"` 같은 문자열이나 모델 객체를 넣습니다. ex01_01은 `load_llm()`이 만든 모델 객체를 넣습니다. |
| `tools` | 도구 리스트 | `@tool` 함수, 일반 함수, dict를 넣을 수 있습니다. `None`이나 빈 리스트면 도구 루프 없이 모델 노드만 만들어집니다. |
| `system_prompt` | `str` 또는 `SystemMessage` | 시스템 프롬프트. 모델을 호출할 때마다 메시지 목록 맨 앞에 붙습니다. |
| `middleware` | 미들웨어 리스트 | 에이전트 실행의 각 단계에 끼어들어 동작을 바꿉니다. 같은 미들웨어 인스턴스를 중복으로 넣으면 `AssertionError`가 납니다. |
| `response_format` | Pydantic 모델, `ToolStrategy`, `ProviderStrategy`, `AutoStrategy`, dict | 구조화된 출력. 결과는 `result["structured_response"]`에 담깁니다. Pydantic 모델만 넣으면 모델 능력에 맞는 전략이 자동으로 선택됩니다. |
| `state_schema` | `AgentState`를 확장한 TypedDict | State에 필드를 추가합니다. 공식 문서는 필드 확장을 미들웨어 쪽에서 하는 방식을 더 권장합니다. |
| `context_schema` | dataclass, TypedDict 등 | 실행 컨텍스트의 타입. `invoke(..., context=...)`로 넘기고, 도구에서 `runtime.context`로 꺼냅니다. ch03/ex02의 `Context`가 이것입니다. |

## 2. `compile()`로 전달되는 인자
의미는 지난번에 정리한 `compile()` 인자와 같습니다.

| 인자 | 요약 |
|---|---|
| `checkpointer` | 단기 메모리(thread 단위 대화 기록). interrupt와 HITL에 필요합니다. 서브그래프로 쓸 때는 비워 두면 부모의 체크포인터를 상속합니다. |
| `store` | 장기 메모리(thread끼리 공유) |
| `interrupt_before` / `interrupt_after` | 정적 브레이크포인트. 노드 이름 리스트를 받습니다. 에이전트의 기본 노드 이름은 `"model"`과 `"tools"`입니다. |
| `debug` | 노드 실행, State 업데이트, 전이 과정을 자세히 출력합니다. 미들웨어 동작을 디버깅할 때 유용합니다. |
| `name` | 그래프 이름. 서브그래프나 멀티 에이전트 구성에서 식별자로 쓰입니다. |
| `cache` | 그래프 실행 결과 캐시 |
| `transformers` | 스트림 변환기. 고급 기능입니다. |

## 기본 제공 미들웨어
`langchain.agents.middleware`에서 가져옵니다.

| 분류 | 미들웨어 |
|---|---|
| 사람 개입 | `HumanInTheLoopMiddleware` (ex01_01에서 사용) |
| 컨텍스트 관리 | `SummarizationMiddleware` (긴 대화 요약), `ContextEditingMiddleware` (오래된 도구 결과 정리) |
| 호출 제한 | `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` |
| 장애 대응 | `ModelRetryMiddleware`, `ModelFallbackMiddleware` (실패 시 다른 모델 사용), `ToolRetryMiddleware`, `ToolErrorMiddleware` |
| 보안 | `PIIMiddleware` (개인정보 탐지와 마스킹) |
| 도구 관련 | `LLMToolSelectorMiddleware` (도구가 많을 때 필요한 것만 선택), `LLMToolEmulator` (테스트용 도구 흉내), `ProviderToolSearchMiddleware` |
| 기타 | `TodoListMiddleware`, `ShellToolMiddleware`, `FilesystemFileSearchMiddleware` |
| 직접 만들기 | 데코레이터 `@before_model`, `@after_model`, `@before_agent`, `@after_agent`, `@wrap_model_call`, `@wrap_tool_call`, `@dynamic_prompt`. 또는 `AgentMiddleware`를 상속합니다. |

## 종합 예시
```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, SummarizationMiddleware, ModelCallLimitMiddleware,
)

class Answer(BaseModel):
    result: str
    files: list[str]

agent = create_agent(
    model=load_llm(),
    tools=[write_file, read_file, delete_file],
    system_prompt="너는 파일 관리 도우미야.",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"delete_file": True}),
        SummarizationMiddleware(model=load_llm(), trigger=("tokens", 4000)),  # 인자 형식은 버전마다 다를 수 있음
        ModelCallLimitMiddleware(run_limit=10),
    ],
    response_format=Answer,          # result["structured_response"]로 받음
    context_schema=Context,
    checkpointer=InMemorySaver(),
    name="file_agent",
)
```

`SummarizationMiddleware`와 `ModelCallLimitMiddleware`의 세부 인자 이름은 이번에 확인하지 않았습니다. 실제로 쓰기 전에 `help(SummarizationMiddleware)` 등으로 확인하세요.