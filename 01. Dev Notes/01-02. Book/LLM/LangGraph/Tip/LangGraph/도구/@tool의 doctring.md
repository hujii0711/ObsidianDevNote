`@tool`(LangChain 기준)에서는 docstring이 **LLM이 읽는 도구 설명(description)** 이 됩니다. 모델은 이 설명만 보고 "이 도구를 언제, 어떻게 쓸지" 판단하므로 일반 docstring보다 훨씬 중요합니다.

```python
from langchain_core.tools import tool

```python
@tool(parse_docstring=True)
def get_weather(city: str, unit: str = "C") -> str:
    """도시의 현재 날씨를 조회한다.

    Args:
        city: 도시 이름 (예: "Seoul")
        unit: 온도 단위, "C" 또는 "F"
    """
    ...

print(get_weather.description)  # "도시의 현재 날씨를 조회한다."
print(get_weather.args)
# {'city': {'description': '도시 이름 (예: "Seoul")', 'type': 'string', ...},
#  'unit': {'description': '온도 단위, "C" 또는 "F"', ...}}
```

**영향 포인트**
- **도구 선택**: 첫 줄/본문이 `description`으로 모델에 전달돼서, 설명이 모호하면 도구를 안 부르거나 엉뚱하게 부릅니다.
- **필수 여부**: docstring이 없고 `description`도 따로 주지 않으면 `ValueError`가 납니다.
- **인자 설명**: `parse_docstring=True`로 하면 `Args:` 섹션(Google 스타일)이 파싱돼 각 인자의 설명이 스키마에 들어갑니다. 인자 이름·타입은 type hint에서 가져옵니다.
- **토큰 비용**: docstring 전체가 매 호출마다 프롬프트에 포함되니, 길고 장황하게 쓰면 비용이 늘어납니다.
- **직접 지정**: docstring 대신 `@tool(description="...")`처럼 명시할 수도 있습니다.

**작성 팁:** "무엇을 하는지 + 언제 쓰는지(쓰지 말아야 할 때 포함) + 인자 예시"를 간결하게 쓰세요.

다른 프레임워크(예: Claude Agent SDK, CrewAI)의 `@tool`을 말씀하신 거라면 알려주세요. 동작이 조금씩 다릅니다.

---
### 주석과 docstring

**주석(comment)** 은 `#`으로 시작하는 코드 설명이고, **docstring**은 함수·클래스·모듈 맨 앞에 `"""..."""`로 쓰는 문서화용 문자열입니다.

| | 주석 (`#`) | docstring (`"""`) |
|---|---|---|
| 대상 | 읽는 사람(개발자) | 사용자·도구·개발자 모두 |
| 실행 시 | 파서가 완전히 무시 | 문자열 객체로 저장됨 |
| 접근 | 불가 | `__doc__`, `help()`로 조회 가능 |
| 위치 | 어디든 가능 | 모듈/클래스/함수 첫 문장 |
| 용도 | "왜 이렇게 짰는지" 설명 | "무엇을 하는지, 인자·반환값" 설명 |

```python
def area(r):
    """원의 넓이를 반환한다.

    Args:
        r: 반지름
    Returns:
        원의 넓이 (float)
    """
    # 소수점 오차를 줄이기 위해 math.pi 대신 상수 사용
    return 3.14159 * r ** 2

print(area.__doc__)   # docstring 출력
help(area)            # docstring 기반 도움말
```

**정리하면:**
- **docstring**: 함수의 사용법(인터페이스)을 설명. Sphinx 같은 문서 자동 생성 도구와 IDE 툴팁이 이걸 읽습니다.
- **주석**: 코드 내부의 구현 이유나 주의사항을 설명. 실행에는 아무 영향이 없습니다.