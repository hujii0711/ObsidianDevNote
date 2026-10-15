LangChain 기준으로 설명드리면, 두 클래스의 핵심 차이는 **출력 형태**입니다.

## PromptTemplate

- 하나의 **문자열 프롬프트**를 만드는 템플릿입니다.
- 주로 텍스트 입력을 받는 LLM(completion 스타일 모델)에 사용합니다.
- 결과는 `StringPromptValue`(문자열)입니다.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘.")
prompt.format(topic="벡터 DB")
# '벡터 DB에 대해 한 문장으로 설명해줘.'
```

## ChatPromptTemplate

- **역할(role)이 있는 메시지 리스트**를 만드는 템플릿입니다.
- system / human / ai 메시지를 구분해 구성할 수 있어 채팅 모델(ChatOpenAI, ChatAnthropic 등)에 적합합니다.
- 결과는 `ChatPromptValue`(메시지 리스트)입니다.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    ("human", "{question}"),
])
prompt.invoke({"role": "데이터베이스", "question": "인덱스란?"})
# [SystemMessage(...), HumanMessage(...)]
```

## 한눈에 비교

|구분|PromptTemplate|ChatPromptTemplate|
|---|---|---|
|출력|단일 문자열|메시지 리스트|
|역할 구분|없음|system/human/ai 지원|
|대화 기록 삽입|불가(직접 문자열로 이어붙여야 함)|`MessagesPlaceholder`로 가능|
|적합한 모델|텍스트 LLM|채팅 모델|

## 선택 기준

- 요즘 대부분의 모델은 채팅 모델이므로 **ChatPromptTemplate이 사실상 기본 선택**입니다. system 지시문 분리, 대화 히스토리, few-shot 예시를 넣기 쉽습니다.
- 단순 문자열 조합이 필요하거나, 프롬프트를 다른 템플릿의 일부로 재사용할 때는 PromptTemplate이 간단합니다.
- 참고로 채팅 모델에 일반 문자열을 넘기면 자동으로 human 메시지 하나로 변환되어 동작하긴 하지만, 역할 구분의 이점은 얻지 못합니다.

원하시면 `MessagesPlaceholder`를 이용한 대화 히스토리 예시나 LCEL 체인(`prompt | llm | parser`) 구성도 보여드릴게요.

---
**메시지 객체의 리스트(`list[BaseMessage]`)** 형식입니다. 문자열이 아니라 역할이 구분된 메시지 4개가 순서대로 담겨 나옵니다.

```python
[
    SystemMessage(content="\n        너는 항상 밝은 말투로 대화하는 챗봇이야. 답변의 끝에 이모티콘을 붙여줘.\n        "),
    HumanMessage(content="\n                          오늘은 날씨가 어때?\n                          "),
    AIMessage(content="\n                         오늘은 날씨가 아주 좋아요!\n                         "),
    HumanMessage(content="오늘 너의 기분은 어때?"),
]
```

## 왜 이렇게 되는지

1. `SystemMessage + HumanMessage`: 메시지 객체끼리 `+`를 하면 `ChatPromptTemplate`이 만들어집니다.
2. `+ AIMessage`: 같은 템플릿에 메시지가 하나 더 추가됩니다.
3. `+ "{input}"`: 문자열을 더하면 **human 메시지 템플릿**으로 변환되어 추가됩니다. `{input}`이 변수 자리가 됩니다.
4. `.format_messages(input=...)`: 변수를 채워서 최종 메시지 리스트를 반환합니다. 앞의 세 메시지는 고정 내용이라 그대로 들어가고, 마지막 HumanMessage에만 `input` 값이 들어갑니다.

즉 `new_prompt`는 `ChatPromptTemplate`이고, `format_messages()`의 결과가 위 리스트입니다. `invoke()`를 쓰면 같은 리스트가 `ChatPromptValue`로 감싸져 나옵니다.

## 참고할 점

- **공백 문제**: 삼중따옴표(`"""`) 안의 들여쓰기와 줄바꿈이 `content`에 그대로 포함됩니다. 모델에 불필요한 공백이 전달되므로 `"..."` 한 줄 문자열로 쓰거나 `textwrap.dedent().strip()`을 쓰는 게 좋습니다.
- **import 경로**: `langchain.schema`는 구버전 경로입니다. 요즘은 `from langchain_core.messages import HumanMessage, SystemMessage, AIMessage`를 사용합니다.
- **같은 결과를 만드는 더 명시적인 방식**:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 항상 밝은 말투로 대화하는 챗봇이야. 답변의 끝에 이모티콘을 붙여줘."),
    ("human", "오늘은 날씨가 어때?"),
    ("ai", "오늘은 날씨가 아주 좋아요!"),
    ("human", "{input}"),
])
prompt.format_messages(input="오늘 너의 기분은 어때?")
```

이 리스트를 `ChatOpenAI().invoke(...)`에 그대로 넘기면 앞의 대화가 few-shot 예시처럼 작용해 밝은 말투와 이모티콘 스타일이 유지됩니다.