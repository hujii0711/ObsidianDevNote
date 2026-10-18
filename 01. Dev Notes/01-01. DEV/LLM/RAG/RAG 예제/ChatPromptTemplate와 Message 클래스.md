
핵심 차이는 `SystemMessage`, `HumanMessage`, `AIMessage`는 실제 메시지를 만드는 클래스이고, `ChatPromptTemplate`은 메시지를 생성하기 위한 템플릿이라는 점입니다.

웹 개발에 비유하면 `ChatPromptTemplate`은 변수 치환이 가능한 템플릿이고, `HumanMessage` 등은 실제 데이터가 채워진 메시지 객체라고 생각하면 이해하기 쉽습니다.

## 1. 개념 비교

|구분|메시지 클래스|`ChatPromptTemplate`|
|---|---|---|
|목적|실제 메시지 구성|메시지 생성 규칙 정의|
|변수 치환|직접 문자열 조합|`{변수명}` 사용 가능|
|대화 이력|메시지 객체를 직접 전달|템플릿에 메시지 이력을 삽입 가능|
|주요 메서드|`llm.invoke(messages)`|`format_messages()` 또는 `invoke()`|
|적합한 상황|메시지가 이미 정해져 있을 때|반복적으로 프롬프트를 생성할 때|

## 2. 같은 프롬프트를 두 가지 방법으로 작성하기

### 방법 A. 메시지 클래스를 직접 사용

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

question = "LangGraph란 무엇인가?"

messages = [
    SystemMessage(
        content="너는 AI 개발 전문가다. 한국어로 답변하라."
    ),
    HumanMessage(
        content=question
    ),
]

response = llm.invoke(messages)
print(response.content)
```

이 방식은 실행 시점에 실제 메시지를 직접 구성합니다.

### 방법 B. ChatPromptTemplate 사용

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 AI 개발 전문가다. 한국어로 답변하라."
    ),
    (
        "human",
        "{question}"
    ),
])

messages = prompt.format_messages(
    question="LangGraph란 무엇인가?"
)

response = llm.invoke(messages)
print(response.content)
```

이번에는 `"question"`이라는 변수를 사용했습니다. `format_messages()`가 변수에 실제 값을 대입하고, `SystemMessage`와 `HumanMessage` 등을 포함한 메시지 목록을 생성합니다.

두 예제는 결과적으로 비슷한 메시지를 LLM에 전달합니다.

## 3. ChatPromptTemplate의 진짜 장점: LCEL 체이닝

`ChatPromptTemplate`은 단순히 메시지를 편리하게 만드는 기능만 있는 것은 아닙니다. LangChain의 LCEL(LangChain Expression Language)과 결합해 프롬프트 생성부터 모델 실행까지 연결할 수 있습니다.

입력 변수
question = "LangGraph란 무엇인가?"

ChatPromptTemplate
변수를 메시지에 삽입

Chat Model
LLM 실행

AIMessage
모델의 응답

코드로는 다음처럼 표현할 수 있습니다.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 AI 개발 전문가다."),
    ("human", "{question}"),
])

chain = prompt | llm

response = chain.invoke({
    "question": "LangGraph란 무엇인가?"
})

print(response.content)
```

여기서 `|`는 단순한 논리 연산자가 아니라 LCEL의 파이프 연산자로, 앞 단계의 결과를 다음 단계의 입력으로 전달합니다.

이렇게 작성하면 중간에 `format_messages()`를 직접 호출하거나 메시지 목록을 별도로 관리할 필요가 없습니다.

## 4. 대화 이력이 필요하다면?

`ChatPromptTemplate`은 `MessagesPlaceholder`를 이용해 이전 대화 이력을 삽입할 수 있습니다.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 Python 전문 개발자다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm

response = chain.invoke({
    "history": [
        HumanMessage(content="Python 리스트가 무엇인가?"),
        AIMessage(content="여러 값을 순서대로 저장하는 자료구조다."),
    ],
    "question": "값을 추가하려면?"
})

print(response.content)
```

이렇게 하면 시스템 지침, 이전 대화 이력, 현재 질문을 명확하게 분리할 수 있습니다.

단, `MessagesPlaceholder`는 대화 이력을 자동으로 저장하는 기능이 아닙니다. 이력은 직접 전달하거나 LangGraph의 상태 관리 또는 별도 저장소를 이용해 관리해야 합니다.

## 5. 실무에서는 어떤 것을 선택해야 할까?

메시지 클래스를 직접 사용
메시지 내용을 동적으로 조립하거나, 도구 호출 결과 및 AI 응답을 직접 관리할 때 적합합니다. 특히 LangGraph의 노드 내부에서 메시지 목록을 조작할 때 유용합니다.

ChatPromptTemplate 사용
시스템 지침이 정해져 있고 사용자 질문, 검색 문서, 대화 이력 같은 변수를 반복적으로 넣어야 할 때 적합합니다. 프롬프트와 모델 실행을 LCEL로 연결하기에도 좋습니다.

특히 RAG를 구현한다면 `ChatPromptTemplate`이 편리합니다.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """
다음 문서를 근거로 질문에 답변하라.

[문서]
{context}
"""),
    ("human", "{question}"),
])

chain = prompt | llm
```

검색된 문서를 `context`에, 사용자의 질문을 `question`에 전달하면 됩니다.

기억할 점: 둘은 서로 대체 관계가 아닙니다. `ChatPromptTemplate`이 최종적으로 `SystemMessage`, `HumanMessage` 등의 메시지 객체를 만들어 주므로 함께 사용하는 것이 일반적입니다. LangGraph에서도 프롬프트 템플릿으로 메시지를 구성하고, `MessagesState`로 대화 상태를 관리하는 조합을 추천합니다.

---

LangChain의 `PromptTemplate`과 `ChatPromptTemplate`의 가장 큰 차이는 일반 텍스트 프롬프트를 만드느냐, 역할이 구분된 채팅 메시지들을 만드느냐입니다.

앞서 살펴본 `SystemMessage`, `HumanMessage`, `AIMessage`까지 함께 비교하면 세 가지의 관계를 명확하게 이해할 수 있습니다.

## 1. 네 가지 개념 비교

|구분|`PromptTemplate`|`ChatPromptTemplate`|메시지 클래스|
|---|---|---|---|
|결과물|문자열|메시지 목록|개별 메시지 객체|
|역할 구분|없음|있음|있음|
|변수 치환|지원|지원|직접 처리|
|주요 목적|텍스트 프롬프트 생성|채팅 프롬프트 생성|실제 메시지 구성|
|대표 사용처|일반 텍스트 LLM, 요약|채팅 모델, RAG, 에이전트|대화 이력, 도구 호출|

## 2. PromptTemplate: 하나의 문자열 생성

`PromptTemplate`은 변수에 값을 대입해 하나의 텍스트를 만듭니다.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
너는 {role} 전문가다.

다음 질문에 답변하라.
질문: {question}
"""
)

text = prompt.format(
    role="Python 개발",
    question="LangGraph란 무엇인가?"
)

print(text)
```

생성 결과는 다음과 같은 일반 문자열입니다.

```
너는 Python 개발 전문가다.

다음 질문에 답변하라.
질문: LangGraph란 무엇인가?
```

즉, 결과 타입은 `str`입니다.

```
print(type(text))  # <class 'str'>
```

이 문자열을 일반 텍스트 입력을 받는 LLM에 전달할 수 있습니다. 다만 `ChatPromptTemplate`과 마찬가지로 채팅 모델과 연결하는 체인을 만들 수도 있습니다.

## 3. ChatPromptTemplate: 역할별 메시지 목록 생성

이번에는 같은 내용을 채팅 메시지 형식으로 구성합니다.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 {role} 전문가다."),
    ("human", "{question}"),
])

messages = prompt.format_messages(
    role="Python 개발",
    question="LangGraph란 무엇인가?"
)

print(messages)
```

생성 결과의 개념적인 형태는 다음과 같습니다.

```
[
    SystemMessage(
        content="너는 Python 개발 전문가다."
    ),
    HumanMessage(
        content="LangGraph란 무엇인가?"
    )
]
```

즉, 결과는 일반 문자열이 아니라 메시지 객체의 목록입니다.

```python
print(type(messages))     # list
print(type(messages[0]))  # SystemMessage
print(type(messages[1]))  # HumanMessage
```

## 4. 실제 차이를 코드로 확인하기

### A. PromptTemplate
하나의 텍스트로 구성
"너는 Python 전문가다. 질문: LangGraph란 무엇인가?"
### B. ChatPromptTemplate
역할과 내용이 분리된 메시지로 구성

SystemMessage
너는 Python 전문가다.

HumanMessage
LangGraph란 무엇인가?

중요한 점은 `ChatPromptTemplate`이 단순히 문자열에 `system`, `human` 같은 글자를 붙이는 것이 아니라, 각 메시지의 역할을 구조적으로 구분한다는 것입니다.

## 5. LLM에 전달하는 방법

두 가지 모두 LCEL의 `|` 연산자로 모델과 연결할 수 있습니다.

### PromptTemplate 연결

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "다음 질문에 답변하라: {question}"
)

chain = prompt | llm

response = chain.invoke({
    "question": "LangGraph란 무엇인가?"
})

print(response.content)
```

### ChatPromptTemplate 연결

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 AI 개발 전문가다."),
    ("human", "{question}"),
])

chain = prompt | llm

response = chain.invoke({
    "question": "LangGraph란 무엇인가?"
})

print(response.content)
```

두 코드 모두 같은 방식으로 `invoke()`를 호출하지만, 모델에 전달되는 입력의 구조가 다릅니다.

|단계|PromptTemplate|ChatPromptTemplate|
|---|---|---|
|입력|변수 딕셔너리|변수 딕셔너리|
|프롬프트 생성 결과|문자열|메시지 목록|
|모델 호출|`invoke()`|`invoke()`|
|일반적인 응답|`AIMessage`|`AIMessage`|

채팅 모델을 사용할 때는 `PromptTemplate`의 문자열도 입력으로 처리될 수 있지만, 시스템 지침과 사용자 발화를 구분해야 한다면 `ChatPromptTemplate`이 더 명확합니다.

## 6. 그렇다면 언제 무엇을 써야 할까?

PromptTemplate을 선택할 때
- 일반 텍스트 생성
- 문서 요약용 문자열 구성
- 단일 텍스트 프롬프트를 사용하는 작업

ChatPromptTemplate을 선택할 때
- 시스템 지침과 사용자 질문 분리
- RAG 검색 문서를 프롬프트에 삽입
- 이전 대화 이력 삽입
- LangGraph 에이전트의 프롬프트 구성

메시지 클래스를 직접 사용할 때
- 대화 이력을 코드로 직접 조작
- `AIMessage` 응답을 이력에 추가
- 도구 호출과 도구 결과를 메시지로 관리

## 7. LangGraph와 RAG를 개발한다면 추천하는 방식

현재처럼 로컬 LLM과 RAG를 연결하고 LangGraph 에이전트를 구축하려는 경우에는 다음과 같이 구분하는 것이 좋습니다.

```
PromptTemplate
    └── 일반 문자열 프롬프트 생성

ChatPromptTemplate
    └── 시스템 지침 + 검색 문서 + 사용자 질문 구성

SystemMessage / HumanMessage / AIMessage
    └── 실제 대화 메시지 및 이력 관리

LangGraph MessagesState
    └── 그래프 실행 중 메시지 상태 관리
```

특히 `ChatPromptTemplate`은 프롬프트의 구조를 정의하고, `MessagesState`는 실행 과정에서 메시지 상태를 관리한다는 점을 구분하면 좋습니다.

한 줄 요약: `PromptTemplate`은 텍스트를 만들고, `ChatPromptTemplate`은 역할별 채팅 메시지를 만들며, `SystemMessage`와 `HumanMessage` 등은 그 메시지를 구성하는 실제 객체입니다.