
멀티 에이전트 **supervisor(관리자) 패턴**에서, 다음에 어떤 에이전트가 일할지 LLM이 고르게 하는 프롬프트를 만드는 코드입니다.

## 전체 구조

```python
prompt = ChatPromptTemplate.from_messages([...]).partial(options=str(options_for_next))
```

메시지 3개를 순서대로 이어 붙인 프롬프트 템플릿을 만들고, `{options}` 변수를 미리 채워 둡니다.

## 구성 요소

**1. `("system", system_prompt)`**  
supervisor의 역할을 정의하는 시스템 메시지입니다. 보통 "너는 worker들(예: math_agent, research_agent)을 관리하는 supervisor다" 같은 내용이 들어갑니다.

**2. `MessagesPlaceholder(variable_name="messages")`**  
지금까지의 대화 기록이 들어갈 자리입니다. 실행할 때 `{"messages": [...]}`를 넘기면 사용자 질문과 각 에이전트의 응답이 이 위치에 펼쳐집니다.

**3. `("human", "Given the conversation above, ...")`**  
마지막에 붙이는 지시 메시지입니다. 해석하면 다음과 같습니다.

- "위 대화를 보고, 다음에 누가 행동해야 하는가? 아니면 FINISH인가?"
- "요청된 작업이 끝났다면 반드시 FINISH를 선택하라. 끝난 작업을 반복하지 마라."
- "설명 없이 옵션 중 정확히 하나만 답하라: `{options}`"

대화 맨 끝에 이 질문을 두는 이유는, 긴 대화에서도 LLM이 "지금 내가 해야 할 일은 다음 담당자 선택"이라는 점을 잊지 않게 하기 위해서입니다. "FINISH를 반드시 선택하라"와 "반복하지 마라"는 문장은 supervisor가 끝나지 않고 같은 에이전트를 계속 호출하는 무한 루프를 막는 안전장치입니다. "설명 없이 하나만"은 출력을 `math_agent` 같은 한 단어로 고정해서 코드에서 바로 라우팅에 쓰기 위한 것입니다.

**4. `.partial(options=str(options_for_next))`**  
템플릿의 `{options}` 변수를 지금 미리 채웁니다. `options_for_next`가 예를 들어 `["math_agent", "research_agent", "FINISH"]`라면, 프롬프트에는 `"['math_agent', 'research_agent', 'FINISH']"` 문자열이 들어갑니다. 이렇게 하면 이후에는 `messages`만 넘겨도 됩니다.

```python
chain = prompt | llm
chain.invoke({"messages": state["messages"]})  # options는 안 넘겨도 됨
```

## 참고 사항

- **`# noqa: ISC004`**: ruff(린터)의 경고를 끄는 주석입니다. ISC004는 "튜플 안에서 문자열 리터럴을 쉼표 없이 이어 붙임(암묵적 연결)"에 대한 규칙인데, 여기서는 의도적으로 긴 문자열을 여러 줄로 나눈 것이라 경고를 무시합니다. 프롬프트 동작과는 무관합니다.
- **중괄호 주의**: `ChatPromptTemplate`은 `{...}`를 변수로 해석합니다. `system_prompt`에 JSON 예시처럼 `{`가 들어 있으면 오류가 나므로 `{{`, `}}`로 이스케이프해야 합니다.
- **`str(options_for_next)`**: 리스트를 그대로 `str()`하면 대괄호와 따옴표가 포함된 형태로 들어갑니다. 깔끔하게 보이게 하려면 `", ".join(options_for_next)`를 쓰는 방법도 있습니다.
- **최신 방식**: 이 방식은 LLM 출력을 문자열로 받아 파싱하는 구조입니다. 모델이 지원한다면 `with_structured_output`으로 `Literal["math_agent", "research_agent", "FINISH"]` 같은 스키마를 강제하는 편이 더 안정적입니다.

`system_prompt`와 `options_for_next`가 어떻게 정의돼 있는지 보여주시면 전체 흐름까지 이어서 설명해 드릴 수 있습니다.