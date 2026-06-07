
`@dataclass`는 Python 3.7+에서 도입된 데코레이터로, 데이터를 저장하는 클래스를 간결하게 작성할 수 있게 해줍니다.

### 핵심 기능

`__init__`, `__repr__`, `__eq__` 등의 특수 메서드를 **자동으로 생성**해줍니다.

---

### 기본 예제

```python
from dataclasses import dataclass

# 일반 클래스 (boilerplate 많음)
class PersonOld:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, email={self.email})"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age


# @dataclass 사용 (훨씬 간결!)
@dataclass
class Person:
    name: str
    age: int
    email: str

p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice", 30, "alice@example.com")

print(p1)           # Person(name='Alice', age=30, email='alice@example.com')
print(p1 == p2)     # True (자동 생성된 __eq__)
```

---

### 주요 옵션

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)   # 불변 객체 (immutable)
class Point:
    x: float
    y: float

@dataclass(order=True)    # 비교 연산자 자동 생성 (<, >, <=, >=)
class Score:
    value: int
    name: str

# 기본값 설정
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    tags: list = field(default_factory=list)  # 가변 기본값은 field() 사용!
```

---

### 실전 예제 — 쇼핑 카트

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity


@dataclass
class Cart:
    owner: str
    items: List[Product] = field(default_factory=list)

    def add(self, product: Product):
        self.items.append(product)

    def grand_total(self) -> float:
        return sum(item.total() for item in self.items)


cart = Cart(owner="Bob")
cart.add(Product("사과", 1500, 3))
cart.add(Product("우유", 2800))

print(cart)
# Cart(owner='Bob', items=[Product(name='사과', price=1500, quantity=3), ...])

print(f"총 금액: {cart.grand_total():,}원")  # 총 금액: 7,300원
```

---

### 언제 사용하면 좋을까?

|상황|권장 여부|
|---|---|
|데이터 저장 목적의 클래스|✅ 적극 권장|
|API 응답/요청 모델링|✅ 적극 권장|
|복잡한 비즈니스 로직 포함|⚠️ 일반 클래스 고려|
|불변 데이터 (frozen=True)|✅ `namedtuple` 대체로 좋음|

> **Tip:** 더 강력한 기능(유효성 검사, JSON 직렬화 등)이 필요하다면 **Pydantic**의 `BaseModel`도 함께 살펴보세요.