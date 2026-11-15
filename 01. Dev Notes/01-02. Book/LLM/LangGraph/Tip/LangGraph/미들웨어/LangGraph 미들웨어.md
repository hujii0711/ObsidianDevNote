미들웨어는 **에이전트 루프(모델 호출 ↔ 도구 실행)의 정해진 지점에 끼워 넣는 코드**입니다. 프롬프트 바꾸기, 재시도, 사람 승인, 개인정보 가리기처럼 에이전트마다 반복되는 일을 루프를 고치지 않고 덧붙입니다.

> 기준 버전: 이 저장소 `.venv`의 `langchain 1.4.7`, `langgraph 1.2.18`. 아래 동작은 가짜 모델로 직접 실행해 확인했습니다(맨 아래 "확인 방법").

## 0. 먼저 알아 둘 점: 미들웨어는 `create_agent`의 기능입니다

- import 경로는 `langchain.agents.middleware`입니다. LangGraph의 `StateGraph`에는 미들웨어라는 개념이 없습니다.
- `create_agent(..., middleware=[...])`가 미들웨어를 **LangGraph 노드와 엣지로 바꿔서** 그래프를 만듭니다. 결과물은 보통의 `CompiledStateGraph`라서 `invoke`, `stream`, checkpointer, `interrupt`가 그대로 됩니다.
- 그래서 "LangGraph 미들웨어"라고 부르지만 정확히는 **LangGraph 위에 만든 LangChain 에이전트의 확장 장치**입니다.

| 직접 짠 `StateGraph` (예: [shorts_graph/graph.py](../../shorts_graph/graph.py)) | `create_agent` + 미들웨어 |
|---|---|
| 노드·엣지를 내가 전부 정한다 | 모델↔도구 루프는 정해져 있고, 훅만 끼운다 |
| 재시도: 노드의 `RetryPolicy` | 재시도: `ModelRetryMiddleware`, `ToolRetryMiddleware` |
| 사람 승인: 노드 안에서 `interrupt()` | 사람 승인: `HumanInTheLoopMiddleware` |
| 프롬프트 변경: 노드 코드 수정 | 프롬프트 변경: `@dynamic_prompt` |

## 1. 에이전트 루프와 훅 6개

```
            ┌──────────── 반복 ────────────┐
before_agent → before_model → [wrap_model_call → 모델] → after_model → [wrap_tool_call → 도구] ─┐
                    ▲                                                                       │
                    └───────────────────── 도구 결과를 들고 다시 모델로 ─────────────────────┘
                                         (도구 호출이 없으면) → after_agent → 끝
```

| 훅 | 언제 | 몇 번 | 받는 것 → 돌려주는 것 |
|---|---|---|---|
| `before_agent` | 호출 시작 시 | invoke당 1번 | `(state, runtime)` → 상태 변경 dict 또는 `None` |
| `before_model` | 모델 부르기 직전 | 모델 호출마다 | `(state, runtime)` → dict / `None` |
| `wrap_model_call` | 모델 호출을 감싼다 | 모델 호출마다 | `(request, handler)` → `ModelResponse` / `AIMessage` |
| `after_model` | 모델 응답 직후 | 모델 호출마다 | `(state, runtime)` → dict / `None` |
| `wrap_tool_call` | 도구 실행을 감싼다 | 도구 호출마다 | `(request, handler)` → `ToolMessage` / `Command` |
| `after_agent` | 끝나기 직전 | invoke당 1번 | `(state, runtime)` → dict / `None` |

훅은 두 종류로 나뉩니다.

- **노드형**(`before_*`, `after_*`): 그래프에 노드로 들어갑니다(이름은 `미들웨어이름.before_model` 꼴). 상태를 읽고 **바꿀 값만 dict로 돌려줍니다**. 노드 함수와 같습니다.
- **감싸기형**(`wrap_*`): 노드가 아니라 모델·도구 호출 자체를 감쌉니다. `handler(request)`를 부르면 실제 호출이 일어납니다. 안 부르면 건너뛰고(캐시), 여러 번 부르면 재시도이며, 부르기 전후로 요청과 응답을 고칠 수 있습니다.

비동기 버전은 앞에 `a`가 붙습니다(`abefore_model`, `awrap_model_call` ...). 동기 버전만 있는 미들웨어를 `ainvoke`로 써도 되지만, **`awrap_*`만 만들고 `invoke`(동기)로 부르면 `NotImplementedError`**가 납니다.

## 2. 만드는 방법 두 가지

### (1) 데코레이터: 훅 하나짜리

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_model_call, dynamic_prompt

@before_model
def log_count(state, runtime):          # 매개변수 이름은 꼭 state, runtime
    print("메시지 수:", len(state["messages"]))
    return None                          # 바꿀 상태가 없으면 None

@wrap_model_call
def retry(request, handler):
    for i in range(3):
        try:
            return handler(request)
        except Exception:
            if i == 2:
                raise

@dynamic_prompt                          # wrap_model_call의 지름길: 문자열만 돌려주면 시스템 프롬프트가 된다
def prompt(request):
    level = request.runtime.context.user_level
    return "쉽게 설명해." if level == "초보" else "전문가처럼 답해."

agent = create_agent(model, tools, middleware=[log_count, retry, prompt])
```

데코레이터 종류: `@before_agent`, `@before_model`, `@after_model`, `@after_agent`, `@wrap_model_call`, `@wrap_tool_call`, `@dynamic_prompt`.
선택 인자: `state_schema=`, `tools=`, `can_jump_to=`, `name=`.

> **주의:** 클래스든 데코레이터든 노드형 훅의 매개변수 이름은 `state`, `runtime`이어야 합니다. 이름으로 넘겨받기 때문에 `(self, s, r)`처럼 줄여 쓰면 `missing 1 required positional argument`가 납니다(직접 겪은 오류).

### (2) 클래스: 설정값이 있거나 훅 여러 개

```python
from langchain.agents.middleware import AgentMiddleware

class MessageLimit(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    def before_model(self, state, runtime):
        if len(state["messages"]) > self.max_messages:
            return {"messages": [AIMessage("대화가 너무 길어 종료합니다.")], "jump_to": "end"}
        return None
    before_model.__can_jump_to__ = ["end"]   # 또는 @hook_config(can_jump_to=["end"])
```

클래스 속성:

| 속성 | 용도 |
|---|---|
| `state_schema` | 에이전트 상태에 필드 추가 (아래 3-3) |
| `tools` | 이 미들웨어가 쓰는 도구를 에이전트에 자동 등록 (`TodoListMiddleware`의 `write_todos`가 이 방식) |
| `name` | 노드 이름. 같은 클래스를 두 번 넣으려면 `name`을 다르게 해야 한다 |
| `trace_policy` | LangSmith 추적에서 훅 입력/출력을 뺄지 정한다 (`TracePolicy(process_inputs=omit_payload)`) |

## 3. 핵심 기능

### 3-1. 실행 순서

`middleware=[A, B]`일 때 실제로 찍힌 순서입니다(도구 1번 → 모델 2번).

```
A.before_agent → B.before_agent
→ A.before_model → B.before_model
→ A.wrap_model> → B.wrap_model> → (모델) → B.wrap_model< → A.wrap_model<
→ B.after_model → A.after_model
→ A.wrap_tool> → B.wrap_tool> → (도구) → B.wrap_tool< → A.wrap_tool<
→ (다시 before_model부터 반복) ...
→ B.after_agent → A.after_agent
```

- `before_*`: 리스트 **앞에서부터**
- `after_*`: 리스트 **뒤에서부터** (역순)
- `wrap_*`: **앞이 바깥 껍질**. 양파처럼 A가 B를 감싼다

그래서 순서가 의미를 바꿉니다. 예를 들어 `[ModelFallback, ModelRetry]`이면 각 모델을 재시도한 뒤 다음 모델로 넘어가고, `[ModelRetry, ModelFallback]`이면 대체 모델까지 모두 실패한 뒤 처음부터 재시도합니다.

### 3-2. `ModelRequest` 고치기 (`wrap_model_call`)

`request`에는 이번 모델 호출의 재료가 다 들어 있습니다: `model`, `messages`(시스템 메시지 제외), `system_message`, `tools`, `tool_choice`, `response_format`, `model_settings`, `state`, `runtime`.
직접 대입하지 말고 **`request.override(...)`로 새 요청을 만들어** handler에 넘깁니다.

```python
@wrap_model_call
def pick(request, handler):
    # 대화가 길면 큰 모델로, 권한 없는 사용자에게는 위험한 도구 숨기기
    if len(request.messages) > 20:
        request = request.override(model=big_model)
    if request.runtime.context.role != "admin":
        request = request.override(tools=[t for t in request.tools if t.name != "delete"])
    return handler(request)
```

이 패턴 하나로 동적 모델 선택, 동적 도구 선택, 동적 출력 형식, 메시지 다듬기를 모두 합니다(edu/ch07 예제들).

응답 쪽에서 상태까지 바꾸고 싶으면 `ExtendedModelResponse(model_response=..., command=Command(update={...}))`를 돌려줍니다. 단 이 Command에서 `goto`, `resume`는 아직 안 됩니다.

### 3-3. 상태에 필드 추가 (`state_schema`)

```python
from typing_extensions import NotRequired
from langchain.agents.middleware import AgentState

class MyState(AgentState):
    blocked: NotRequired[bool]

@before_model(state_schema=MyState, can_jump_to=["end"])
def guard(state, runtime):
    if "금지" in state["messages"][-1].content:
        return {"blocked": True,
                "messages": [AIMessage("답할 수 없는 요청입니다.")],
                "jump_to": "end"}
```

실행 결과: 모델은 아예 불리지 않고 마지막 메시지가 "답할 수 없는 요청입니다.", 결과 dict에 `blocked: True`가 들어 있었습니다.
여러 미들웨어의 `state_schema`는 하나로 합쳐집니다. 입력·출력에서 숨기려면 `Annotated[..., PrivateStateAttr]`를 씁니다(`jump_to` 자체가 이렇게 숨겨진 필드입니다).

### 3-4. 흐름 건너뛰기 (`jump_to`)

노드형 훅이 `{"jump_to": "end" | "model" | "tools"}`를 돌려주면 그 지점으로 갑니다. **`can_jump_to`로 미리 선언해야** 그래프에 그 엣지가 생깁니다. 선언하지 않으면 함께 돌려준 메시지는 들어가지만 이동은 무시되고 모델이 그대로 불립니다(확인 결과: `['x', '차단', '모델 응답']`).

- `"end"`: 바로 종료. 입력 차단, 호출 한도 초과에 씁니다
- `"model"`: 모델을 다시 부릅니다. 출력이 검증에 실패하면 다시 쓰게 할 때 씁니다(무한 반복에 주의)
- `"tools"`: 도구 노드로 갑니다

### 3-5. `runtime`: context와 store

- `runtime.context`: `create_agent(context_schema=Ctx)`로 정의하고 `agent.invoke(..., context=Ctx(...))`로 넘기는 **호출마다 바뀌는 고정 설정**(사용자 ID, 권한, 등급). 상태가 아니므로 대화 기록에 남지 않습니다
- `runtime.store`: `create_agent(store=...)`의 장기 저장소. 세션을 넘는 사용자 선호를 읽습니다
- `wrap_*` 훅에서는 `request.runtime`, `request.state`로 접근합니다

## 4. 내장 미들웨어 (1.4.7 기준 16개)

### 대화 길이·컨텍스트 관리

| 이름 | 하는 일 | 주요 인자 |
|---|---|---|
| `SummarizationMiddleware` | 기준을 넘으면 오래된 메시지를 요약 1개로 바꾼다 | `model`, `trigger=("tokens", 4000)` · `("messages", 50)` · `("fraction", 0.8)` · `{"tokens":4000,"messages":10}`(AND) · 리스트(OR), `keep=("messages", 20)` |
| `ContextEditingMiddleware` | 오래된 **도구 결과**를 지워 토큰을 줄인다 | `edits=[ClearToolUsesEdit(...)]` |
| `LLMToolSelectorMiddleware` | 도구가 많을 때 작은 LLM으로 이번에 쓸 도구만 고른다 | `model`, `max_tools`, `always_include` |
| `ProviderToolSearchMiddleware` | 도구를 공급자(Anthropic 등)의 기본 도구 검색 뒤로 숨긴다 | `searchable_tools` |

### 안정성·비용

| 이름 | 하는 일 | 주요 인자 |
|---|---|---|
| `ModelRetryMiddleware` | 모델 호출 실패 시 지수 백오프로 재시도 | `max_retries=2`, `retry_on`, `on_failure="continue"`, `backoff_factor`, `initial_delay` |
| `ToolRetryMiddleware` | 도구 실패 재시도 | 위와 같음 + `tools=[...]`(대상 제한) |
| `ToolErrorMiddleware` | 도구 예외를 에러 `ToolMessage`로 바꿔 모델에 돌려준다(모델이 스스로 고칠 기회) | `on_error(exc, request) -> str \| None`(None이면 그대로 예외), `tools` |
| `ModelFallbackMiddleware` | 실패 시 다음 모델로 | `ModelFallbackMiddleware(first, second, ...)` |
| `ModelCallLimitMiddleware` | 모델 호출 횟수 제한 | `run_limit`(invoke 1번), `thread_limit`(대화 전체, checkpointer 필요), `exit_behavior="end" \| "error"` |
| `ToolCallLimitMiddleware` | 도구 호출 횟수 제한 | `tool_name`(없으면 전체), `run_limit`, `thread_limit`, `exit_behavior="continue"` |

`ModelCallLimitMiddleware(run_limit=1)`로 확인한 결과: 첫 모델 호출 → 도구 실행 → 두 번째 호출 대신 `"Model call limits exceeded: run limit (1/1)"` 메시지로 끝났습니다.

### 사람 개입·안전

| 이름 | 하는 일 | 주요 인자 |
|---|---|---|
| `HumanInTheLoopMiddleware` | 지정 도구 실행 전에 `interrupt`로 멈춘다 | `interrupt_on={"도구": True \| {"allowed_decisions": [...], "description": ...}}`, `interrupt_mode="batched" \| "per_call"` |
| `PIIMiddleware` | 이메일·카드번호 등 탐지 후 처리 | `pii_type`, `strategy="redact" \| "mask" \| "hash" \| "block"`, `detector`(정규식/함수), `apply_to_input/output/tool_results` |

### 작업 보조·개발

| 이름 | 하는 일 |
|---|---|
| `TodoListMiddleware` | `write_todos` 도구와 안내 프롬프트를 넣어 긴 작업을 계획하게 한다 |
| `LLMToolEmulator` | 도구를 실제로 실행하지 않고 LLM이 결과를 지어낸다(테스트·프로토타입용) |
| `FilesystemFileSearchMiddleware` | Glob/Grep 파일 검색 도구 추가 |
| `ShellToolMiddleware` | 세션 동안 유지되는 셸 도구 추가. `execution_policy`로 Host/Docker/Codex 샌드박스 선택 |

### HITL 결정 형식

checkpointer가 꼭 있어야 합니다(멈춘 상태를 저장해야 하므로).

```python
agent = create_agent(model, [send_email],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "send_email": {"allowed_decisions": ["approve", "edit", "reject"]}})],
    checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "t1"}}
r = agent.invoke({"messages": [...]}, cfg)
r["__interrupt__"][0].value["action_requests"]   # [{'name','args','description'}]

agent.invoke(Command(resume={"decisions": [
    {"type": "approve"},
    # {"type": "edit", "edited_action": {"name": "send_email", "args": {...}}},
    # {"type": "reject", "message": "보내지 마세요"},
    # {"type": "respond", "message": "사람이 도구 대신 답한 내용"},
]}), cfg)
```

- `batched`(기본): 모델 한 턴의 도구 호출들을 interrupt **1번**에 모읍니다. `decisions`는 `action_requests` 순서대로 줍니다
- `per_call`: 도구 호출마다 interrupt가 따로 생깁니다. `Command(resume={interrupt.id: 결정, ...})`로 답합니다
- `respond`: 도구를 실행하지 않고 사람의 답을 성공 결과로 돌려줍니다. "사용자에게 묻기" 같은 도구에 씁니다
- `edit`로 인자를 바꾸면, 모델에게 "사람이 호출을 바꿔서 실행했다"는 안내가 붙은 결과가 갑니다(확인 결과 `a=5,b=5` → `a=50,b=50`으로 바꿔 `100`)

## 5. 활용 패턴 정리

| 하고 싶은 일 | 쓸 것 | 이 저장소 예제 |
|---|---|---|
| 로그·측정 | `before_model`/`after_model`, 클래스 | [edu/ch06/ex03.py](../../edu/ch06/ex03.py) |
| 사용자별 프롬프트 | `@dynamic_prompt` + `runtime.context` | [edu/ch03/ex04.py](../../edu/ch03/ex04.py), [edu/ch07/ex03.py](../../edu/ch07/ex03.py) |
| store 기반 개인화 | `@dynamic_prompt` + `runtime.store` | [edu/ch07/ex02.py](../../edu/ch07/ex02.py) |
| 동적 모델 선택 | `wrap_model_call` + `override(model=)` | [edu/ch02/ex01.py](../../edu/ch02/ex01.py), [edu/ch06/ex06.py](../../edu/ch06/ex06.py) |
| 권한별 도구 제한 | `wrap_model_call` + `override(tools=)` | [edu/ch07/ex06.py](../../edu/ch07/ex06.py), [edu/ch07/ex07.py](../../edu/ch07/ex07.py) |
| 출력 형식 바꾸기 | `override(response_format=)` | [edu/ch07/ex10.py](../../edu/ch07/ex10.py) |
| 재시도 직접 구현 | `wrap_model_call` 안에서 handler 반복 | [edu/ch06/ex05.py](../../edu/ch06/ex05.py) |
| 입력 차단 | `before_agent` + `jump_to="end"` | [edu/ch08/ex04.py](../../edu/ch08/ex04.py), [edu/ch08/ex05.py](../../edu/ch08/ex05.py) |
| 최종 출력 검사 | `after_agent` | [edu/ch08/ex06.py](../../edu/ch08/ex06.py) |
| 개인정보 | `PIIMiddleware` | [edu/ch08/ex01.py](../../edu/ch08/ex01.py) ~ ex03 |
| 사람 승인 | `HumanInTheLoopMiddleware` | [edu/ch05/](../../edu/ch05/) |
| 여러 겹 방어 | 여러 개 조합 | [edu/ch08/ex07.py](../../edu/ch08/ex07.py), [edu/ch08/ex08.py](../../edu/ch08/ex08.py) |
| 실행 순서 실험 | 클래스 여러 개 | [edu/ch06/ex07.py](../../edu/ch06/ex07.py) |

edu에 아직 없는 것: `Summarization`, `ContextEditing`, `ModelRetry`/`ToolRetry`, `ToolError`, `ModelFallback`, `ModelCallLimit`/`ToolCallLimit`, `LLMToolSelector`, `TodoList`, `LLMToolEmulator`, HITL의 `per_call`/`respond`, `ExtendedModelResponse`.

### 도구 예외를 모델에게 돌려주기 (직접 구현)

```python
@wrap_tool_call
def safe(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"도구 오류: {e}",
                           tool_call_id=request.tool_call["id"], status="error")
```

확인 결과: 예외 대신 `ToolMessage('도구 오류: 터짐')`가 들어가고 모델이 다음 턴에 이어서 답했습니다. 같은 일을 내장 `ToolErrorMiddleware`가 합니다.

### 자주 하는 실수

1. 노드형 훅의 매개변수 이름을 `state, runtime`으로 쓰지 않음 → TypeError
2. `jump_to`를 돌려주면서 `can_jump_to` 선언을 빠뜨림 → 오류 없이 이동만 무시되어 모델이 그대로 불림
3. HITL이나 `thread_limit`을 쓰면서 checkpointer를 안 줌
4. `wrap_*`에서 `request`를 직접 대입함 → `override()`를 쓴다
5. 같은 미들웨어 클래스를 두 번 넣으면서 `name`을 안 바꿈 → `AssertionError: Please remove duplicate middleware instances.`
6. 순서를 신경 안 씀 → 재시도/대체 모델, PII 가리기/로그 순서에 따라 결과가 달라진다

## 6. 이 프로젝트에 적용한다면

`shorts_graph`의 큰 흐름은 직접 짠 `StateGraph`라 미들웨어가 들어갈 자리가 없습니다. 하지만 노드 **안에서** 쓰는 에이전트 두 개는 `create_agent`입니다.

- [shorts_graph/storyboard/nodes.py:42](../../shorts_graph/storyboard/nodes.py#L42): 장면 설명 에이전트
- [shorts_graph/translation/translate.py:46](../../shorts_graph/translation/translate.py#L46): 번역 에이전트

둘 다 도구가 없고(`tools=[]`) 로컬 LLM 한 번 호출이 전부라서 의미가 있는 건 이 정도입니다.

- `ModelRetryMiddleware(max_retries=2, on_failure="error")`: 로컬 mlx 서버가 잠깐 응답하지 않을 때 장면 하나 때문에 전체가 실패하지 않게 합니다. 지금은 첫 실패에 바로 `GraphError`가 납니다
- `wrap_model_call`로 장면 응답 형식 검사 후 재요청: 4bit 모델이 형식을 어길 때 씁니다. 다만 `describe_scene`의 파싱·재시도와 겹치지 않게 한 곳에서만 해야 합니다
- 대본 가드레일(`script/guards.py`)은 미들웨어로 옮기지 않는 편이 낫습니다. 가드레일은 에이전트 응답이 아니라 그래프 상태(대본)를 검사하고, 승인 interrupt와도 엮여 있기 때문입니다

## 확인 방법

LLM 서버 없이 `GenericFakeChatModel`(정해진 응답을 차례로 내는 가짜 모델)에 `bind_tools`만 덧붙여 1~4장 예제를 실행했습니다. 확인한 것은 실행 순서, `dynamic_prompt`+context, `jump_to`+`state_schema`, `wrap_tool_call` 예외 처리, `ModelCallLimit`, HITL edit 재개, 그래프 노드 이름입니다.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw): return self
model = Fake(messages=iter([AIMessage("", tool_calls=[...]), AIMessage("끝")]))
```

미들웨어를 넣은 그래프의 노드 이름은 `agent.get_graph().nodes`로 볼 수 있습니다: `model`, `tools`, `TraceA.before_model`, `guard.before_model` ...
