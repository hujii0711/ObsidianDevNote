
## `field()` 역할

### 핵심 개념

`field()`는 dataclass의 **각 필드에 상세 옵션을 부여**하는 함수입니다.
단순 타입 힌트만으로는 설정할 수 없는 동작을 지정할 때 사용합니다.

---

### 주요 옵션 한눈에 보기

```python
from dataclasses import dataclass, field

@dataclass
class Example:
    # 1. default — 단순 기본값
    name: str = field(default="홍길동")

    # 2. default_factory — 호출 가능한 객체로 기본값 생성
    tags: list = field(default_factory=list)

    # 3. repr — __repr__ 출력에서 제외
    password: str = field(default="secret", repr=False)

    # 4. init — 생성자(__init__) 파라미터에서 제외
    created_at: str = field(default="2024", init=False)

    # 5. compare — 동등 비교(__eq__)에서 제외
    score: int = field(default=0, compare=False)
```

---

### 옵션별 설명

| 옵션                | 타입   | 설명                                 |
| ----------------- | ---- | ---------------------------------- |
| `default`         | 값    | 기본값 직접 지정                          |
| `default_factory` | 함수   | 인스턴스 생성마다 호출해서 기본값 생성              |
| `repr`            | bool | `print(obj)` 출력에 포함 여부 (기본 `True`) |
| `init`            | bool | `__init__` 파라미터 포함 여부 (기본 `True`)  |
| `compare`         | bool | ==, <  비교 시 포함 여부 (기본 `True`)      |
| `hash`            | bool | `__hash__` 계산에 포함 여부               |
| `metadata`        | dict | 필드에 임의 메타데이터 부착                    |

---

### `field()` 없이 vs 있을 때 비교

```python
# field() 없이 — 기본값만 지정 가능
@dataclass
class Simple:
    name: str = "홍길동"
    tags: list = []          # ❌ 오류! 가변 객체는 직접 기본값 불가


# field() 사용 — 세부 동작 제어 가능
@dataclass
class Advanced:
    name: str = field(default="홍길동", repr=True)
    tags: list = field(default_factory=list)   # ✅ 인스턴스마다 새 리스트
    password: str = field(default="1234", repr=False)  # print시 숨김
```

---

### 실제 동작 확인

```python
a = Advanced()
b = Advanced()

a.tags.append("python")

print(a.tags)  # ['python']
print(b.tags)  # []  ← 독립적인 리스트 (default_factory 덕분)

print(a)  # Advanced(name='홍길동', tags=['python'])
          # password는 repr=False 라 출력 안 됨
```

> **한 줄 요약:** `field()`는 기본값 설정을 넘어 **초기화·출력·비교 동작까지 세밀하게 제어**하는 도구입니다.