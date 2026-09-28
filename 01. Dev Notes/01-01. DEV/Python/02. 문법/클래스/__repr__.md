
`__repr__`는 파이썬의 **매직 메서드(Special Method)** 중 하나로, **객체를 개발자 관점에서 표현하는 문자열을 반환**하는 메서드입니다.

쉽게 말하면 다음과 같습니다.

- `__repr__` → **개발자가 객체를 이해하고 디버깅하기 위한 표현**
    
- `__str__` → **사용자에게 보여주기 위한 표현**

---

## 1. 기본 사용법

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Kim", 30)

print(p)
```

출력

```
<__main__.Person object at 0x104d3c610>
```

별도로 `__repr__`를 구현하지 않으면 객체의 메모리 주소가 출력됩니다.

---

## 2. **repr** 구현하기

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age})"

p = Person("Kim", 30)

print(p)
```

출력

```
Person(name='Kim', age=30)
```

훨씬 이해하기 쉬운 형태가 됩니다.

---

## 3. 언제 호출될까?

### print()

```python
print(p)
```

`__str__`이 없으면 `__repr__`가 호출됩니다.

---

### 인터프리터에서 객체 입력

```python
>>> p
Person(name='Kim', age=30)
```

REPL에서는 항상 `__repr__`이 사용됩니다.

---

### 리스트 안의 객체

```python
people = [p]

print(people)
```

출력

```
[Person(name='Kim', age=30)]
```

리스트는 내부 요소를 출력할 때 각 요소의 `__repr__`을 호출합니다.

---

### 딕셔너리

```python
d = {
    "person": p
}

print(d)
```

출력

```
{'person': Person(name='Kim', age=30)}
```

---

## 4. __repr__와 **str** 차이

```python
class Person:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Person({self.name!r})"

    def __str__(self):
        return f"{self.name}"
```

```python
p = Person("Kim")

print(p)
```

출력

```
Kim
```

반면

```python
p
```

출력

```
Person('Kim')
```

즉,

|상황|호출되는 메서드|
|---|---|
|print(obj)|`__str__` 우선, 없으면 `__repr__`|
|REPL에서 객체 입력|`__repr__`|
|list 출력|`__repr__`|
|dict 출력|`__repr__`|
|set 출력|`__repr__`|

---

## 5. 공식적인 권장사항

파이썬 문서에서는 `__repr__`은 가능한 한 **객체를 다시 생성할 수 있는 형태**를 권장합니다.

예를 들어

```python
class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
```

출력

```
Point(3, 5)
```

가능하면

```python
p = eval(repr(point))
```

처럼 동일한 객체를 다시 만들 수 있는 표현이 이상적입니다. (물론 항상 가능한 것은 아닙니다.)

---

## 6. `!r`의 의미

아까 예제에서

```python
return f"Person(name={self.name!r}, age={self.age})"
```

의 `!r`은

```python
repr(self.name)
```

을 의미합니다.

예를 들어

```python
name = "Kim"

print(f"{name}")
```

출력

```
Kim
```

하지만

```python
print(f"{name!r}")
```

출력

```
'Kim'
```

문자열이라는 것이 명확하게 표현됩니다.

---

## 7. dataclass에서는 자동 생성

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Kim", 30)

print(p)
```

출력

```
Person(name='Kim', age=30)
```

`@dataclass`는 기본적으로 적절한 `__repr__`을 자동으로 생성해 주므로 별도로 구현할 필요가 없는 경우가 많습니다.

---

## 8. 실무에서 많이 사용하는 형태

실무에서는 대부분 다음과 같이 작성합니다.

```python
class User:

    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

    def __repr__(self):
        return (
            f"User(id={self.id}, "
            f"name={self.name!r}, "
            f"email={self.email!r})"
        )
```

디버깅 시

```python
users = [
    User(1, "Kim", "kim@test.com"),
    User(2, "Lee", "lee@test.com")
]

print(users)
```

출력

```
[
    User(id=1, name='Kim', email='kim@test.com'),
    User(id=2, name='Lee', email='lee@test.com')
]
```

처럼 객체의 핵심 상태를 한눈에 확인할 수 있어 디버깅이 매우 편리합니다.

### 정리

- `__repr__`은 **개발자용 객체 표현**을 정의하는 메서드입니다.
    
- `repr(obj)`나 REPL, 리스트·딕셔너리 출력 시 주로 사용됩니다.
    
- `__str__`이 없으면 `print(obj)`도 `__repr__`을 사용합니다.
    
- 가능한 한 객체를 명확히 식별할 수 있는 문자열을 반환하는 것이 좋으며, 가능하다면 객체를 다시 생성할 수 있는 형태를 권장합니다.
    
- `@dataclass`를 사용하면 `__repr__`이 자동으로 생성되어 대부분의 경우 별도 구현이 필요 없습니다.


---
##  `__str__` vs `__repr__` 차이점

파이썬에는 문자열로 변환하는 비슷한 메서드로 `__str__`도 있습니다. 이 둘은 목적이 다릅니다.

| **특징**    | **__str__**                    | **__repr__**                                           |
| --------- | ------------------------------ | ------------------------------------------------------ |
| **목적**    | **사용자(User)**가 보기 편한 예쁜 문자열 출력 | **개발자(Developer)**가 디버깅할 때 필요한 명확한 정보 제공               |
| **호출 시점** | `print()`, `str()` 호출 시        | 인터랙티브 콘솔(REPL)에서 객체 그냥 입력 시, `repr()` 호출 시             |
| **권장 형태** | 가독성 좋은 텍스트 (`"3학년 1반 홍길동"`)    | 객체 생성 코드 형태 (`"Student(grade=3, class=1, name='홍길동')"` |
|           |                                |                                                        |
|           |                                |                                                        |