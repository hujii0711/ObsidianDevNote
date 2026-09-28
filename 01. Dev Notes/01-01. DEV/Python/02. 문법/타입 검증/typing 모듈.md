
파이썬은 기본적으로 변수의 자료형을 선언할 필요가 없는 **동적 타이핑(Dynamic Typing)** 언어입니다. 하지만 코드가 커지면 "이 함수의 인자로 무엇이 들어와야 하지?", "이 함수는 어떤 값을 반환하지?" 헷갈리기 쉽습니다.
이를 해결하기 위해 파이썬 3.5(PEP 484)부터 도입된 모듈이 바로 `typing`입니다. 변수나 함수의 입출력에 타입 힌트(Type Hints)를 부여할 수 있게 도와줍니다.

### 1. `typing`을 사용하는 이유

- **가독성 향상:** 코드만 봐도 어떤 데이터 타입이 오고 가는지 명확히 알 수 있습니다.
- **IDE 지원 및 자동 완성:** VS Code, PyCharm 같은 편집기에서 타입을 미리 인지하여 오타나 잘못된 메서드 사용을 사전에 경고해 줍니다.
- **정적 분석 도구 활용:** `mypy` 같은 도구를 사용하면 실행(Runtime) 전에 타입 오류를 미리 찾아낼 수 있습니다.

### 2. 자주 사용하는 주요 타입들
`typing` 모듈에서 가장 흔하게 가져다 쓰는 대표적인 기능들입니다.

- **`List`, `Dict`, `Tuple`, `Set`**: 내부 요소의 타입까지 지정할 수 있습니다.  
    - 예: `List[str]` (문자열로 이루어진 리스트)
- **`Optional`**: 값이 들어올 수도 있고 `None`일 수도 있음을 나타냅니다. (예: `Optional[str]`은 `str` 또는 `None`)
- **`Union`**: 여러 타입 중 하나가 될 수 있음을 나타냅니다. (예: `Union[int, float]`는 정수 또는 실수)
- **`Callable`**: 함수 자체를 인자로 받을 때 사용합니다.
- **`Any`**: 어떤 타입이든 허용합니다. (타입 체크를 포기하는 것과 비슷하므로 꼭 필요한 곳에만 사용해야 합니다.)

#### 보충)

> `typing` 모듈: 보조 도구 상자

- **정의:** 타입 힌트를 작성하고 싶지만, 파이썬의 **기본 자료형(`list`, `dict` 등)만으로는 표현하기 까다로운 복잡한 타입들을 정의**하기 위해 파이썬 표준 라이브러리에 추가된 모듈입니다.
- **역할:** "그냥 리스트가 아니라, **정수들로만 이루어진 리스트**야"라는 걸 표현하려면 `List[int]` 같은 특별한 도구가 필요합니다. 이 도구들을 모아둔 곳이 바로 `typing` 모듈입니다.
- **주요 도구들:** `List`, `Dict`, `Union`, `Optional`, `Callable`, `Any` 등

### 3. 간단한 코드 예제

```python
from typing import List, Dict, Optional, Union

# 1. 리스트와 딕셔너리 타입 지정
def get_user_names(user_ids: List[int]) -> Dict[int, str]:
    # user_ids는 정수 리스트, 반환값은 (정수: 문자열) 딕셔너리
    return {uid: f"User_{uid}" for uid in user_ids}

# 2. Optional과 Union 사용
def process_data(data: Union[str, int], prefix: Optional[str] = None) -> str:
    if prefix:
        return f"{prefix}_{data}"
    return str(data)

# 사용 예시
result = get_user_names([1, 2, 3])
print(result)  # {1: 'User_1', 2: 'User_2', 3: 'User_3'}
```

### 💡 파이썬 버전별 참고 사항 (중요!)

파이썬 **3.9 버전 이후**부터는 `typing` 모듈을 일일이 임포트하지 않고도 파이썬 기본 자료형(`list`, `dict` 등)에 직접 대괄호를 쓸 수 있게 변경되었습니다.

- **구 방식 (과거):** `from typing import List` -> `x: List[int] = [1, 2]`
- **신 방식 (파이썬 3.9+ 권장):** `x: list[int] = [1, 2]`

다만, `Optional`, `Union`, `Callable` 같은 특수한 타입들은 여전히 `typing` 모듈을 임포트해서 사용하거나 파이썬 3.10+의 파이프 연산자(`int | None`) 등을 활용합니다.