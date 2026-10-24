`@wrap_model_call`은 LangChain v1 미들웨어 데코레이터로, **모델 호출을 감싸서(wrap) 호출 전·후를 마음대로 제어**할 수 있게 해 줍니다. 앞서 쓴 `@dynamic_prompt`가 "시스템 프롬프트만 바꾸는" 특화 기능이라면, `@wrap_model_call`은 요청 전체를 수정하거나, 재시도하거나, 아예 호출을 가로챌 수 있는 범용 기능입니다.

## 1. 기본 구조

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def my_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 1) 호출 전: request 확인/수정
    print("모델 호출 전")

    response = handler(request)  # 2) 실제 모델 호출 (핵심)

    # 3) 호출 후: response 확인/수정
    print("모델 호출 후")
    return response
```

- `request`: 모델에 보낼 내용 (메시지, 시스템 프롬프트, 도구, 모델 등)
- `handler`: 다음 단계(실제 모델 호출)를 실행하는 함수. **호출하지 않으면 모델이 실행되지 않습니다.**
- 반환값: `ModelResponse` (모델 응답)

## 2. 주요 활용 예시

### (1) 요청 수정: 상황에 따라 모델 바꾸기

```python
from langchain.chat_models import init_chat_model

basic_model = init_chat_model("gpt-4o-mini")
advanced_model = init_chat_model("gpt-4o")


@wrap_model_call
def dynamic_model(request: ModelRequest, handler) -> ModelResponse:
    # 대화가 길어지면 고성능 모델 사용
    if len(request.messages) > 10:
        request = request.override(model=advanced_model)
    return handler(request)
```

`request.override(...)`로 일부 값만 바꾼 새 요청을 만들어 넘깁니다. 원본을 직접 수정하지 않는 방식입니다.

### (2) 재시도 / 오류 처리

```python
@wrap_model_call
def retry_on_error(request: ModelRequest, handler) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            print(f"시도 {attempt + 1} 실패: {e}")
            if attempt == 2:
                raise
```

`handler`를 여러 번 호출할 수 있다는 점이 `@dynamic_prompt`와의 가장 큰 차이입니다.

### (3) 로깅 / 시간 측정

```python
import time


@wrap_model_call
def timing(request: ModelRequest, handler) -> ModelResponse:
    start = time.perf_counter()
    response = handler(request)
    print(f"모델 호출 소요 시간: {time.perf_counter() - start:.2f}초")
    return response
```

### (4) 호출 가로채기 (캐시 등)

```python
cache = {}


@wrap_model_call
def simple_cache(request: ModelRequest, handler) -> ModelResponse:
    key = str(request.messages[-1].content)
    if key in cache:
        return cache[key]  # handler를 호출하지 않고 바로 반환
    response = handler(request)
    cache[key] = response
    return response
```

## 3. 에이전트에 적용

```python
agent = create_agent(
    model=model,
    middleware=[timing, retry_on_error, user_role_prompt],
    context_schema=Context,
)
```

미들웨어 리스트의 **앞에 있는 것이 바깥쪽**을 감쌉니다. 위 예시에서는 `timing` → `retry_on_error` → `user_role_prompt` 순으로 감싸므로, 시간 측정에는 재시도 시간까지 모두 포함됩니다.

## 4. 데코레이터 비교

|데코레이터|용도|
|---|---|
|`@dynamic_prompt`|시스템 프롬프트만 동적으로 생성|
|`@before_model` / `@after_model`|모델 호출 전/후에 상태(state)를 확인·수정|
|`@wrap_model_call`|모델 호출 자체를 감싸서 요청 수정, 재시도, 가로채기|
|`@wrap_tool_call`|도구 호출을 감싸서 제어 (모델 대신 도구 대상)|

## 참고

- 비동기 에이전트(`ainvoke`, `astream`)에서는 `async def`로 정의하고 `await handler(request)`를 사용해야 합니다. 동기 함수만 정의하면 비동기 호출 시 오류가 날 수 있습니다.
- 정확한 import 경로와 `request.override()`에서 바꿀 수 있는 필드는 LangChain 버전에 따라 다를 수 있으니, 설치된 버전의 문서를 확인하세요.