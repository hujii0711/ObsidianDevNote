
**Runnable**은 LangChain의 핵심 추상화로, "입력을 받아 출력을 내는 모든 구성 요소"가 따르는 **공통 인터페이스**입니다. 프롬프트 템플릿, 챗 모델, 출력 파서, 리트리버, 툴 등이 모두 Runnable이기 때문에 같은 방식으로 호출하고 서로 연결할 수 있습니다. `langchain_core.runnables`에 정의되어 있고, LCEL(LangChain Expression Language)의 기반이 됩니다.

## 1. 공통 실행 메서드

모든 Runnable은 다음 메서드를 기본으로 제공합니다.

| 메서드                              | 용도                            |
| -------------------------------- | ----------------------------- |
| `invoke(input)`                  | 입력 하나를 처리해 결과 반환              |
| `batch([inputs])`                | 여러 입력을 병렬로 처리                 |
| `stream(input)`                  | 결과를 조각(chunk) 단위로 스트리밍        |
| `ainvoke` / `abatch` / `astream` | 위 메서드의 비동기 버전                 |
| `astream_events`                 | 중간 단계 이벤트까지 스트리밍 (디버깅·UI에 유용) |

구현체가 `invoke`만 정의해도 나머지는 기본 구현으로 따라옵니다.

## 2. 파이프(`|`)로 조합하기

`|` 연산자로 Runnable을 이어 붙이면 앞의 출력이 다음의 입력이 되는 **RunnableSequence**가 만들어지고, 이것 역시 Runnable입니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘")
model = ChatAnthropic(model="claude-sonnet-4-5")
parser = StrOutputParser()

chain = prompt | model | parser

chain.invoke({"topic": "양자 컴퓨팅"})          # 단일 실행
chain.batch([{"topic": "RAG"}, {"topic": "LoRA"}])  # 배치 실행
for chunk in chain.stream({"topic": "트랜스포머"}):  # 스트리밍
    print(chunk, end="")
```

## 3. 자주 쓰는 Runnable 구성 요소

**RunnableParallel** — 같은 입력으로 여러 Runnable을 동시에 실행하고 결과를 dict로 묶습니다. 체인 안에서는 그냥 dict를 써도 자동 변환됩니다.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup = RunnableParallel(
    context=retriever,               # 검색 결과
    question=RunnablePassthrough(),  # 입력을 그대로 전달
)
rag_chain = setup | rag_prompt | model | parser
```

**RunnablePassthrough** — 입력을 그대로 통과시킵니다. `RunnablePassthrough.assign(key=...)`을 쓰면 기존 입력 dict에 새 키를 추가할 수 있습니다.

**RunnableLambda** — 일반 파이썬 함수를 Runnable로 감쌉니다. 체인 안에 함수를 직접 넣어도 자동으로 변환됩니다.

```python
from langchain_core.runnables import RunnableLambda
to_upper = RunnableLambda(lambda s: s.upper())
chain = prompt | model | parser | to_upper
```

**RunnableBranch** — 조건에 따라 다른 Runnable로 분기합니다. (단순 분기는 함수 안에서 Runnable을 반환하는 `RunnableLambda`로 처리하는 방식도 많이 씁니다.)

## 4. 동작을 덧입히는 메서드

Runnable을 감싸서 새 Runnable을 돌려주는 유틸리티들입니다.

- `.with_retry(stop_after_attempt=3)` — 실패 시 재시도
- `.with_fallbacks([backup_model])` — 실패 시 대체 Runnable 사용
- `.bind(stop=["\n"])` / `.bind_tools([...])` — 호출 인자를 미리 고정
- `.with_config(run_name=..., tags=[...])` — 콜백·추적용 설정
- `.configurable_fields(...)` / `.configurable_alternatives(...)` — 실행 시점에 파라미터나 구성 요소 교체
- `.with_structured_output(Schema)` — (챗 모델) 구조화된 출력

## 5. 왜 유용한가

- **일관성**: 무엇이든 `invoke`/`stream`/`batch`로 똑같이 다룰 수 있음
- **스트리밍·비동기·병렬 처리**를 별도 코드 없이 얻음
- **LangSmith 추적**이 체인의 각 단계별로 자동 기록됨
- **입출력 스키마**(`input_schema`, `output_schema`)를 제공해 LangServe 같은 배포 도구와 연동이 쉬움

참고로 최근 LangChain에서는 복잡한 에이전트·상태 기반 워크플로를 LangGraph로 만드는 것을 권장하지만, LangGraph의 노드와 그래프도 Runnable 인터페이스 위에서 동작하므로 Runnable 개념은 여전히 기본기입니다.

원하시면 RAG 체인을 Runnable로 처음부터 구성하는 예제나, `astream_events`로 중간 단계를 스트리밍하는 방법도 이어서 설명해 드릴게요.
