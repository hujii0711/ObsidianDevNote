# Supervisor 패턴 이해하기 (`langgraph-supervisor`)

> 대상: LangGraph로 에이전트를 하나 만들어 본 적은 있지만, 여러 에이전트를 함께 쓰는 건 처음인 사람
> 예제 코드: [ex01.py](ex01.py)(Research Agent), [ex02.py](ex02.py)(Math Agent), [ex04.py](ex04.py)(Supervisor)
> 기준 버전: `langgraph-supervisor` 0.0.31

---

## 1. 한 줄 요약

**Supervisor 패턴은 "팀장 에이전트 하나가 질문을 보고, 알맞은 팀원 에이전트에게 일을 넘기고, 결과를 모아 답하는 구조"입니다.**
`langgraph-supervisor`는 이 구조를 `create_supervisor()` 함수 하나로 만들어 주는 라이브러리입니다.

---

## 2. 비유로 이해하기: 팀장과 팀원

회사에서 이런 요청을 받았다고 해 봅시다.

> "2025년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체의 몇 %인지 계산해 주세요."

혼자 다 하는 대신 팀으로 나눠서 처리합니다.

| 역할 | 실제 코드 | 하는 일 |
|---|---|---|
| 팀장 | **Supervisor** | 요청을 읽고 누구에게 맡길지 정한다. 직접 일하지 않는다. |
| 조사 담당 | **Research Agent** | 웹 검색으로 GDP 숫자를 찾는다. |
| 계산 담당 | **Math Agent** | 받은 숫자로 백분율을 계산한다. |

일은 이렇게 진행됩니다.

1. 팀장이 요청을 읽고 "먼저 숫자부터 찾아야겠다" → 조사 담당에게 넘김
2. 조사 담당이 GDP를 찾아서 팀장에게 보고
3. 팀장이 "이제 계산하면 되겠다" → 계산 담당에게 넘김
4. 계산 담당이 백분율을 구해서 팀장에게 보고
5. 팀장이 결과를 정리해서 사용자에게 최종 답변

**핵심은 팀원끼리 직접 일을 주고받지 않는다는 점입니다.** 모든 일은 팀장을 거쳐 갑니다.

---

## 3. 왜 에이전트를 나누나요?

에이전트 하나에 검색 도구와 계산 도구를 다 주면 안 될까요? 작은 예제라면 됩니다. 하지만 도구와 역할이 늘어나면 문제가 생깁니다.

- **도구가 많을수록 LLM이 헷갈립니다.** 도구 20개 중 하나를 고르는 것보다 3개 중 하나를 고르는 게 훨씬 정확합니다.
- **프롬프트가 길어집니다.** "검색할 땐 이렇게, 계산할 땐 저렇게…"를 한 프롬프트에 다 넣으면 지시가 서로 섞입니다.
- **고치기 어렵습니다.** 검색 동작만 바꾸고 싶어도 전체 에이전트를 건드려야 합니다.

에이전트를 나누면 각자 **좁은 역할, 적은 도구, 짧은 프롬프트**를 갖게 됩니다. Supervisor는 "누구에게 맡길지"만 고민하면 됩니다.

---

## 4. 구성 요소

```
                  ┌──────────────┐
   사용자 질문 ──▶ │  supervisor  │ ──▶ 최종 답변
                  └──────┬───────┘
              넘기기 ▲   │   ▲ 돌려주기
         ┌───────────┘   │   └───────────┐
         ▼               │               ▼
┌─────────────────┐      │      ┌─────────────────┐
│ research_agent  │      │      │   math_agent    │
│ (web_search 등) │      │      │ (add, divide 등)│
└─────────────────┘      │      └─────────────────┘
                         │
          모두 같은 messages(대화 기록)를 공유
```

### ① Supervisor (팀장)
- 그 자체로 LLM 에이전트입니다.
- 하지만 가진 도구는 **"다른 에이전트에게 넘기기"** 도구뿐입니다.

### ② Worker 에이전트 (팀원)
- `create_agent()`로 만든 평범한 에이전트입니다. ex01, ex02에서 만든 것을 그대로 씁니다.
- **반드시 `name`이 있어야 합니다.** 이 이름으로 넘기기 도구가 만들어지기 때문입니다.

### ③ 핸드오프(handoff) 도구 — 이 패턴의 핵심
"핸드오프"는 **일을 넘긴다**는 뜻입니다. `create_supervisor()`는 팀원 이름마다 도구를 하나씩 자동으로 만듭니다.

| 자동 생성되는 도구 | 누가 쓰나 | 의미 |
|---|---|---|
| `transfer_to_research_agent` | Supervisor | "조사 담당, 이거 맡아 줘" |
| `transfer_to_math_agent` | Supervisor | "계산 담당, 이거 맡아 줘" |
| `transfer_back_to_supervisor` | (자동) | "팀장님, 끝났습니다" |

Supervisor 입장에서 **"누구에게 맡길지 정하는 것"은 "어떤 도구를 부를지 정하는 것"과 똑같습니다.**
LLM은 원래 도구 고르기를 잘하므로, 라이브러리는 이 능력을 그대로 이용해 일을 넘깁니다.

### ④ 공유 메시지 기록
Supervisor와 모든 팀원은 같은 `messages` 목록을 봅니다. 조사 담당이 찾은 GDP 숫자가 기록에 남아 있기 때문에 계산 담당이 그 숫자를 읽고 계산할 수 있습니다.

---

## 5. 실제 실행 흐름 (ex04.py)

ex04.py를 실행하면 이런 로그가 나옵니다. 위에서 설명한 흐름과 그대로 맞아떨어집니다.

```
[노드 supervisor로부터 업데이트]
Name: transfer_to_research_agent        ← ① 팀장이 조사 담당에게 넘김
Successfully transferred to research_agent

[노드 research_agent로부터 업데이트]
Name: transfer_back_to_supervisor       ← ② 조사 끝, 팀장에게 돌아옴
Successfully transferred back to supervisor

[노드 supervisor로부터 업데이트]
Name: transfer_to_math_agent            ← ③ 팀장이 계산 담당에게 넘김
...
[노드 math_agent로부터 업데이트]
Name: transfer_back_to_supervisor       ← ④ 계산 끝, 팀장에게 돌아옴

[노드 supervisor로부터 업데이트]
(최종 답변)                              ← ⑤ 더 넘길 일이 없으면 종료
```

Supervisor가 **넘기기 도구를 부르지 않고 그냥 답하면** 그게 최종 답변이 되고 그래프가 끝납니다.

---

## 6. 코드는 3단계

### 1단계: 팀원 에이전트 만들기 (`name` 필수)

```python
from langchain.agents import create_agent

research_agent = create_agent(
    model=load_llm(),
    tools=[web_search, read_page],
    system_prompt="You are a research specialist agent. ...",
    name="research_agent",   # ← 이 이름으로 transfer_to_research_agent 도구가 생긴다
)

math_agent = create_agent(
    model=load_llm(),
    tools=[add, multiply, divide, power, percentage],
    system_prompt="You are a mathematics specialist agent. ...",
    name="math_agent",
)
```

### 2단계: Supervisor 만들기

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    agents=[research_agent, math_agent],   # 관리할 팀원들
    model=load_llm(),                      # 팀장이 쓸 LLM
    prompt=(                               # 팀장의 지시문 (주의: system_prompt가 아니라 prompt)
        "You are a supervisor managing two agents:\n"
        "- research_agent: web search, facts, statistics\n"
        "- math_agent: arithmetic, percentages\n"
        "Delegate to one agent at a time. Do not do any work yourself."
    ),
).compile()                                # ← compile()을 해야 실행할 수 있다
```

`create_supervisor()`는 **설계도(StateGraph)** 를 돌려줍니다. `.compile()`을 해야 실행 가능한 그래프가 됩니다.

### 3단계: 실행하기

```python
result = supervisor.invoke(
    {"messages": [{"role": "user", "content": "미국과 뉴욕주 GDP를 찾고 비율을 계산해 줘"}]}
)
print(result["messages"][-1].content)
```

---

## 7. 자주 쓰는 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `prompt` | 없음 | Supervisor의 지시문. **각 팀원이 무슨 일을 하는지 꼭 적어 주세요.** 팀장은 이걸 보고 누구에게 맡길지 정합니다. |
| `output_mode` | `"last_message"` | 팀원이 일을 마쳤을 때 대화 기록에 무엇을 남길지 (아래 설명) |
| `add_handoff_messages` | `True` | `transfer_to_...` 호출 기록을 대화에 남길지 |
| `add_handoff_back_messages` | `add_handoff_messages`와 같음 | `transfer_back_to_supervisor` 기록을 대화에 남길지 |
| `parallel_tool_calls` | `False` | 여러 팀원에게 동시에 넘길 수 있게 할지. 순서대로 처리하려면 `False` 그대로 둡니다. |
| `tools` | 없음 | 넘기기 도구 말고 Supervisor가 직접 쓸 도구를 추가할 때 |
| `supervisor_name` | `"supervisor"` | Supervisor 노드 이름 |
| `handoff_tool_prefix` | `"transfer_to_"` | 넘기기 도구 이름의 앞부분 |

### `output_mode`: 팀원의 보고서를 얼마나 자세히 남길까?

팀원 에이전트도 안에서 여러 번 도구를 부르고 생각합니다. 그 과정을 전부 남길지, 결론만 남길지 고르는 옵션입니다.

| 값 | 비유 | 대화 기록에 남는 것 | 장단점 |
|---|---|---|---|
| `"last_message"` (기본) | **결론만 보고** | 팀원의 마지막 답변 하나 | 기록이 짧아 LLM이 덜 헷갈림. 과정은 안 보임 |
| `"full_history"` | **작업 일지 전체 제출** | 팀원이 부른 도구, 검색 결과 등 전부 | 디버깅에 좋음. 기록이 길어져 토큰을 많이 씀 |

학습할 때는 `full_history`로 과정을 보고, 실제로 쓸 때는 `last_message`를 쓰는 게 일반적입니다.
특히 **로컬 4bit 모델은 긴 입력에 약하므로** `last_message`가 더 안정적입니다.

---

## 8. 라이브러리가 대신 해 주는 일

`create_supervisor()` 없이 직접 만들면 아래를 모두 손으로 작성해야 합니다.

- 팀원 이름마다 넘기기 도구 만들기
- 도구가 불리면 해당 팀원 노드로 이동하게 하기(`Command(goto=...)`)
- 팀원이 끝나면 Supervisor로 돌아오는 연결(edge) 만들기
- 돌아올 때 "끝났습니다" 메시지 붙이기
- 팀원의 출력을 `last_message` / `full_history` 방식으로 잘라 넣기
- Supervisor가 더 넘길 곳이 없으면 종료하기

라이브러리는 이 반복 작업을 감춰 줍니다. 대신 내부 동작이 보이지 않으므로, 흐름이 이상할 때는 `full_history`와 `stream()`으로 단계별 로그를 확인하는 게 좋습니다.

---

## 9. 초보자가 자주 하는 실수

### ❶ 팀원 에이전트에 `name`을 안 줌
```
ValueError: Please specify a name when you create your agent ...
```
`create_agent(..., name="math_agent")`처럼 이름을 꼭 줍니다. 이름은 서로 겹치면 안 됩니다.

### ❷ `system_prompt=`로 지시문을 넘김
`create_agent()`는 `system_prompt=`를 쓰지만, **`create_supervisor()`는 `prompt=`** 를 씁니다.
0.0.31 버전에서는 `system_prompt=`로 넘겨도 **에러 없이 조용히 무시됩니다.** 그러면 Supervisor가 팀원 설명 없이 동작해서 엉뚱한 팀원에게 넘기거나 직접 답해 버릴 수 있습니다.

### ❸ `.compile()`을 빼먹음
`create_supervisor()`의 결과는 설계도입니다. `.invoke()`/`.stream()`은 `.compile()` 뒤에만 쓸 수 있습니다.

### ❹ 지시문에 팀원 설명이 없음
Supervisor가 아는 건 도구 이름(`transfer_to_math_agent`)뿐입니다. 이름만으로는 무엇을 맡기면 되는지 정확히 알기 어려우므로, 프롬프트에 "누가 무엇을 하는지"를 적어 주세요.

### ❺ Supervisor가 직접 답해 버림
LLM이 "이 정도는 내가 알지" 하고 넘기지 않는 경우입니다. 프롬프트에 `"DO NOT perform any work yourself - always delegate to agents"` 같은 규칙을 넣습니다.

---

## 10. 정리

- **Supervisor 패턴** = 팀장 1명 + 전문 팀원 여러 명. 모든 일은 팀장을 거친다.
- **일을 넘기는 방법** = 도구 호출. 팀원 이름마다 `transfer_to_<이름>` 도구가 자동으로 생긴다.
- **팀원이 끝나면** 자동으로 팀장에게 돌아온다(`transfer_back_to_supervisor`).
- **코드 3단계** = 이름 있는 팀원 만들기 → `create_supervisor(agents, model, prompt)` → `.compile()` 후 실행
- **주의** = 지시문은 `prompt=`로, 팀원 설명은 프롬프트에 꼭.
