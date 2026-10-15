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