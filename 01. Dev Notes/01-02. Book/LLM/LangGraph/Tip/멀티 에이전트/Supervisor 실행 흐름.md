ex04.py를 실행했을 때 무엇이 어떤 순서로 도는지 정리했습니다. 가장 중요한 점은, **ex04를 실행하면 Supervisor보다 먼저 ex01과 ex02의 테스트가 돌아간다**는 것입니다. 코드만 읽고 정리했고 직접 실행하지는 않았습니다.

## 1단계: import하는 순간 ex01·ex02 전체가 실행됨

[ex04.py:16-17](edu/ch11/ex04.py#L16-L17)

```python
from edu.ch11.ex01 import research_agent
from edu.ch11.ex02 import math_agent
```

Python에서 모듈을 import하면 그 파일의 최상위 코드가 처음부터 끝까지 실행됩니다. `research_agent` 변수 하나만 가져오려 해도 ex01.py 전체가 돕니다.

**ex01.py가 import될 때**
1. `.env`를 로드하고 `TavilySearch(max_results=3)`를 만듭니다.
2. `web_search.invoke({"query": "2026년 대한민국 GDP"})`로 **Tavily에 실제로 검색**하고 결과 미리보기를 출력합니다.
3. `research_agent`를 만들고 "LLM 클라이언트 준비 완료"를 출력합니다.
4. `research_agent.stream("NYC의 현재 시장은 누구인가요?")`로 **LLM 호출과 검색을 실제로 실행**하고 과정을 출력합니다.

**ex02.py가 import될 때**
1. 수학 함수 5개를 정의하고 `math_agent`를 만듭니다. "LLM 클라이언트 준비 완료"가 다시 출력됩니다.
2. `math_agent.stream("(3 + 5) x 7은 얼마인가요?")`로 **LLM을 실제로 호출**하고 과정을 출력합니다.

그래서 ex04를 한 번 실행하면 Supervisor 질문 전에 **Tavily 검색 2번 이상과 에이전트 실행 2번**이 먼저 일어나고, 화면 위쪽은 ex01·ex02의 출력으로 채워집니다.

## 2단계: Supervisor 그래프 만들기

[ex04.py:22-46](edu/ch11/ex04.py#L22-L46)

1. `load_llm()`으로 Supervisor용 LLM을 만듭니다. "LLM 클라이언트 준비 완료"가 세 번째로 출력됩니다.
2. `create_supervisor()`가 내부적으로 다음을 합니다.
   - 두 에이전트의 `name`을 확인합니다(`research_agent`, `math_agent`).
   - 넘기기 도구 `transfer_to_research_agent`, `transfer_to_math_agent`를 만들어 Supervisor LLM에 연결합니다(`bind_tools`).
   - `prompt=`로 받은 지시문을 Supervisor의 시스템 메시지로 넣습니다. 지난번에 고친 부분입니다.
   - 노드 3개와 연결선으로 이루어진 설계도(StateGraph)를 만듭니다.
3. `.compile()`로 실행할 수 있는 그래프로 바꿉니다.
4. `print_graph_ascii(supervisor)`로 그래프 구조를 출력합니다.

만들어지는 그래프 구조:

```
START → supervisor ─┬─(transfer_to_research_agent)→ research_agent ─┐
                    ├─(transfer_to_math_agent)────→ math_agent ─────┤
                    └─(넘기기 도구를 안 부르면)───→ END              │
                         ▲                                          │
                         └──────────── 끝나면 항상 돌아옴 ──────────┘
```

## 3단계: 복합 질문 실행

[ex04.py:64-75](edu/ch11/ex04.py#L64-L75)

질문은 "2025년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요."입니다. 아래는 LLM이 의도대로 움직일 때의 흐름입니다.

| 순서 | 노드 | 하는 일 | 메시지 기록에 추가되는 것 |
|---|---|---|---|
| ① | supervisor | 질문을 읽고 숫자부터 찾기로 정함 | `transfer_to_research_agent` 호출과 "Successfully transferred to research_agent" |
| ② | research_agent | 안에서 LLM과 `web_search`를 여러 번 오가며 GDP를 찾음 | (`full_history`라서) 검색 호출, 검색 결과, 답변 **전부**와 `transfer_back_to_supervisor` |
| ③ | supervisor | 숫자가 모였으니 계산을 맡기기로 정함 | `transfer_to_math_agent` 호출 |
| ④ | math_agent | 기록에서 숫자를 읽고 `divide`, `percentage` 등으로 계산 | 계산 과정 전부와 `transfer_back_to_supervisor` |
| ⑤ | supervisor | 넘기기 도구를 부르지 않고 최종 답을 씀 | 최종 답변, 그 뒤 END |

LLM이 노드를 거치는 순서를 코드로 정해 두지 않았습니다. 매번 Supervisor LLM이 고릅니다. 그래서 ①에서 바로 답해 버리거나 research_agent를 두 번 부르는 일도 있을 수 있습니다.

## 4단계: 화면 출력 방식

```python
for chunk in supervisor.stream({...}):
    pretty_print_messages(chunk, last_message=True)
```

- `stream()`의 기본 모드는 `"updates"`입니다. **노드 하나가 끝날 때마다** `{"노드이름": {"messages": [...]}}` 형태의 chunk가 하나 나옵니다.
- `subgraphs=True`를 주지 않았으므로 research_agent 안에서 LLM과 도구가 오가는 과정은 따로 출력되지 않습니다. 노드가 끝난 뒤 한 번에 나옵니다.
- `last_message=True`라서 각 chunk의 마지막 메시지만 출력됩니다. 그래서 에이전트 노드에서는 내용 대신 `transfer_back_to_supervisor` ToolMessage만 보입니다. 파일 아래쪽 주석에 남아 있는 출력 예시가 이렇게 생긴 이유입니다.

## 5단계: 마지막 줄

```python
final_message_history = chunk.get("supervisor", {}).get("messages", [])
```

반복문이 끝나면 `chunk`에는 마지막 업데이트, 즉 ⑤의 supervisor 노드 결과가 남아 있습니다. 여기서 messages를 꺼냅니다. 변수에 담기만 하고 출력하지는 않으므로 화면에는 아무 변화가 없습니다.

## 실행 순서 한눈에 보기

```
ex04 실행
 ├─ import ex01 → Tavily 테스트 검색 → research_agent 생성 → "NYC 시장" 질문 실행
 ├─ import ex02 → math_agent 생성 → "(3+5)x7" 질문 실행
 ├─ create_supervisor().compile() → 그래프 ASCII 출력
 ├─ GDP 질문 stream
 │    supervisor → research_agent → supervisor → math_agent → supervisor → END
 └─ final_message_history 저장 (출력 없음)
```

## 눈에 띄는 점

1. **import할 때 테스트가 돌아갑니다.** ex01·ex02의 테스트 코드를 `if __name__ == "__main__":` 아래로 옮기면, ex04에서 import할 때는 에이전트만 만들어지고 테스트는 각 파일을 직접 실행할 때만 돕니다. 시간과 Tavily 사용량을 아낄 수 있습니다.
2. **질문 문구가 다릅니다.** 화면에 출력하는 질문([ex04.py:60](edu/ch11/ex04.py#L60))은 "2024년"인데, 실제로 보내는 질문([ex04.py:70](edu/ch11/ex04.py#L70))은 "2025년"입니다.
3. **최종 답을 볼 수 없습니다.** 맨 끝에 `print(final_message_history[-1].content)`를 추가하면 Supervisor의 최종 답변을 전체 문장으로 확인할 수 있습니다.

1번이나 3번을 고쳐 드릴까요?