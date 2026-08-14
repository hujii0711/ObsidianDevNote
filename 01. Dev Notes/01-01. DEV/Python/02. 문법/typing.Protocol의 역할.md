
## 한 줄 요약

`Protocol`은 **"이런 메서드/속성을 가지고 있으면 어떤 타입이든 인정한다"**는 방식(구조적 타이핑, structural typing)으로 인터페이스를 정의하는 도구입니다.

```python
from typing import Protocol

class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...
```

## 1. 기존 방식(상속 기반, `ABC`)과의 차이

일반적으로 파이썬에서 "인터페이스"를 만들려면 `abc.ABC`를 상속받아 강제합니다.

```python
from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

class MyGPT(LLM):  # 반드시 상속받아야 함
    def generate(self, prompt: str) -> str:
        return "응답"
```

→ 이 방식은 **명시적 상속(nominal typing)**이 필수입니다. `LLM`을 상속하지 않으면 아무리 `generate` 메서드가 있어도 `LLM` 타입으로 인정되지 않습니다.

## 2. `Protocol`은 상속 없이도 인정됨 (구조적 타이핑)

```python
from typing import Protocol

class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...

# LLM을 상속받지 않았는데도...
class MyGPT:
    def generate(self, prompt: str) -> str:
        return "응답"

class ClaudeWrapper:
    def generate(self, prompt: str) -> str:
        return "다른 응답"

def call_llm(model: LLM) -> str:
    return model.generate("안녕")

call_llm(MyGPT())         # OK
call_llm(ClaudeWrapper())  # OK
```

→ `MyGPT`, `ClaudeWrapper`는 `LLM`을 전혀 상속받지 않았지만, **`generate(prompt: str) -> str` 메서드 시그니처만 맞으면** 타입 체커(mypy 등)가 `LLM` 타입으로 인정합니다.

이걸 "오리가 걷고, 오리처럼 꽥꽥거리면 오리로 취급한다"는 뜻에서 **덕 타이핑(duck typing)**이라고 부르고, `Protocol`은 이 덕 타이핑을 **타입 힌트 수준에서 공식적으로 지원**하는 도구입니다.

## 3. 왜 이 코드에서 `Protocol`을 썼을까?

```python
class LLM(Protocol):
```

이 이름을 보면, 아마 이런 상황일 가능성이 높습니다.

```python
class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...

# 실제 구현체들 (LLM을 상속하지 않음)
class OpenAIClient:
    def generate(self, prompt: str) -> str:
        # OpenAI API 호출
        ...

class AnthropicClient:
    def generate(self, prompt: str) -> str:
        # Anthropic API 호출
        ...

# 어떤 LLM 구현체든 상관없이 받을 수 있는 함수
def ask(model: LLM, question: str) -> str:
    return model.generate(question)

ask(OpenAIClient(), "질문")
ask(AnthropicClient(), "질문")
```

즉, "OpenAI든 Anthropic이든 로컬 모델이든, **`generate` 메서드만 구현되어 있으면 LLM으로 취급하겠다**"는 **느슨한 계약(loose contract)**을 정의한 것입니다.

## 4. 장점 정리

|특징|설명|
|---|---|
|**낮은 결합도**|구현 클래스가 `Protocol`을 몰라도(import 안 해도) 됨|
|**외부 라이브러리 대응**|이미 만들어진 외부 클래스(수정 불가)도 시그니처만 맞으면 타입으로 인정 가능|
|**테스트 용이**|Mock 객체도 메서드만 맞으면 바로 타입 체크 통과|
|**타입 체크 시점**|런타임이 아니라 mypy 같은 정적 타입 검사기에서 검증됨 (`...`는 실제 실행 안 됨, 시그니처 선언용)|

## 5. 주의할 점

- `Protocol`은 기본적으로 **정적 타입 검사(mypy)용**입니다. 런타임에서 `isinstance()` 체크를 하려면 `@runtime_checkable` 데코레이터가 필요합니다.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...

print(isinstance(OpenAIClient(), LLM))  # True (런타임 체크 가능)
```

- 다만 `runtime_checkable`을 쓰더라도 **메서드 이름 존재 여부만 확인**하지, 인자 타입이나 반환 타입까지 런타임에 검증하지는 않습니다.

---

정리하면, 앞서 다룬 `BaseModel`이 "데이터의 형태(값)를 검증"하는 도구라면, `Protocol`은 "객체가 어떤 행동(메서드)을 할 수 있는지"를 상속 없이 유연하게 검증하는 **인터페이스 정의 도구**라고 이해하시면 됩니다.