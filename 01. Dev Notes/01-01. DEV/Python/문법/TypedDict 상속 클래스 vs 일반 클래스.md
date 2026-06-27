
네, 근본적으로 다릅니다.

---

### 가장 큰 차이: 결과물의 정체

```python
# 일반 클래스
class NormalChunk:
    def __init__(self):
        self.id = "001"

obj = NormalChunk()
type(obj)  # <class 'NormalChunk'>  ← NormalChunk 인스턴스


# TypedDict 클래스
class Chunk(TypedDict):
    id: str

obj = Chunk(id="001")
type(obj)  # <class 'dict'>  ← 그냥 딕셔너리
```

---

### 차이점 비교

| 항목        | 일반 클래스    | TypedDict 클래스   |
| --------- | --------- | --------------- |
| 결과물       | 클래스 인스턴스  | 일반 `dict`       |
| 필드 접근     | `obj.id`  | `obj["id"]`     |
| 메서드 정의    | ✅ 가능      | ❌ 불가능           |
| 상속 확장     | ✅ 자유롭게    | ⚠️ TypedDict끼리만 |
| 런타임 타입 체크 | ✅ 동작      | ❌ 동작 안 함        |
| 목적        | 동작(로직) 정의 | 딕셔너리 구조 선언      |

---

### 메서드를 가질 수 없다

```python
class Chunk(TypedDict):
    id: str
    text: str

    def get_summary(self):  # ❌ 의미 없음, 사실상 무시됨
        return self.text[:100]
```

TypedDict 클래스에 메서드를 넣어도 런타임에서 제대로 동작하지 않습니다. **데이터 구조만 선언하는 용도**이기 때문입니다.

---

### TypedDict끼리는 상속 가능

```python
class BaseChunk(TypedDict):
    id: str
    text: str

class Chunk(BaseChunk):   # ✅ TypedDict 간 상속
    source_type: SourceType
    title: str

# Chunk는 id, text, source_type, title 모두 가짐
```

---

### 한 줄 요약

> 일반 클래스는 **"동작하는 객체"** 를 만들고, TypedDict 클래스는 **"딕셔너리의 생김새"** 를 선언할 뿐이다.