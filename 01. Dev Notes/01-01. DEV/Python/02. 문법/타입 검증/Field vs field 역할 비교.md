
두 라이브러리 모두 클래스의 속성을 정의할 때 쓰지만, 목적과 기능이 다릅니다.

---

### dataclasses의 `field`

파이썬 표준 라이브러리입니다. **기본값 설정**이 주된 역할입니다.

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    tags: list = field(default_factory=list)  # 가변 기본값 설정
    age: int = field(default=0, repr=False)   # repr 출력 제외
```

리스트처럼 가변(mutable) 객체를 기본값으로 쓸 때 `field(default_factory=list)`를 써야 합니다. `tags: list = []`로 쓰면 모든 인스턴스가 같은 리스트를 공유하는 버그가 생기기 때문입니다.

주요 옵션은 `default`, `default_factory`, `repr`, `compare`, `init` 정도입니다.

---

### pydantic의 `Field`

외부 라이브러리입니다. 기본값 외에 **유효성 검사(validation)와 직렬화 메타데이터**까지 담당합니다.

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=0, le=150)           # 0 이상 150 이하
    email: str = Field(pattern=r".+@.+")     # 정규식 검사
    tags: list = Field(default_factory=list, description="태그 목록")
```

입력값이 조건을 벗어나면 자동으로 오류를 발생시킵니다.

---

### 핵심 차이 한눈에

|항목|`dataclasses.field`|`pydantic.Field`|
|---|---|---|
|목적|기본값·동작 제어|기본값 + 유효성 검사|
|유효성 검사|없음|있음 (타입, 범위, 패턴 등)|
|JSON 직렬화|없음|기본 지원|
|별칭(alias)|없음|`alias="user_name"`|
|설명 추가|없음|`description="..."`|
|표준 여부|표준 라이브러리|외부 라이브러리|

---

### 언제 뭘 쓰냐면

단순히 데이터를 묶어두는 구조체가 필요하면 `dataclasses`, API 요청/응답처럼 외부 데이터를 검증하고 직렬화까지 해야 하면 `pydantic`을 선택하면 됩니다.