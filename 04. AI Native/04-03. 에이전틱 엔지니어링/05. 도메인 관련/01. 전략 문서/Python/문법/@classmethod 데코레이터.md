
파이썬의 `@classmethod`는 클래스 내부에서 정의되는 데코레이터로, **인스턴스(객체)가 아닌 클래스 자체를 첫 번째 인자로 받는 메서드**를 만들 때 사용합니다.
쉽게 말해, 개별 데이터(인스턴스)를 다루는 게 아니라 **클래스 전체와 관련된 기능이나 데이터를 다룰 때** 아주 유용합니다.

## 1. @classmethod의 핵심 특징

- **`cls` 인자 사용:** 일반 메서드가 인스턴스 자신을 가리키는 `self`를 첫 번째 인자로 받는 것처럼, 데코레이터가 붙은 메서드는 클래스 자신을 가리키는 `cls`를 첫 번째 인자로 받습니다.
    
- **객체 생성 없이 호출 가능:** `클래스명.메서드명()` 형태로 객체를 따로 만들지 않고 바로 호출할 수 있습니다.
    
- **대체 생성자(Alternative Constructor) 역할:** 파이썬은 생성자(`__init__`)를 하나만 가질 수 있는데, `@classmethod`를 사용하면 다양한 방식으로 객체를 생성하는 서브 생성자를 만들 수 있습니다.
    

## 2. 기본적인 사용법 예시

가장 흔하게 쓰이는 "대체 생성자"의 예시를 통해 알아보겠습니다.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 클래스 메서드 정의
    @classmethod
    def from_birth_year(cls, name, birth_year):
        import datetime
        current_year = datetime.date.today().year
        age = current_year - birth_year
        # cls(name, age)는 결국 Person(name, age)를 호출하여 객체를 반환하는 것과 같습니다.
        return cls(name, age)

    def introduce(self):
        return f"안녕하세요, 제 이름은 {self.name}이고 {self.age}살입니다."


# 1. 일반적인 방식으로 객체 생성
p1 = Person("이몽룡", 25)
print(p1.introduce())

# 2. 클래스 메서드를 통해 태어난 연도로 객체 생성 (객체 생성 없이 바로 호출)
p2 = Person.from_birth_year("성춘향", 2000)
print(p2.introduce())
```

## 3. 꿀팁: `@staticmethod`와의 차이점

종종 `@staticmethod`(정적 메서드)와 헷갈려하시는 분들이 많습니다. 두 메서드 모두 객체 생성 없이 호출할 수 있다는 점은 같지만, 결정적인 차이가 있습니다.

|**구분**|**@classmethod**|**@staticmethod**|
|---|---|---|
|**첫 번째 인자**|`cls` (클래스 자신)|없음 (일반 함수와 동일)|
|**클래스 속성 접근**|가능 (`cls.속성명`으로 변경 및 접근 가능)|불가능|
|**주요 용도**|클래스 상태 변경, 대체 생성자 구현|클래스와 연관은 있지만 독립적인 유틸리티 함수 구현|
|**상속 시 동작**|상속받은 자식 클래스를 올바르게 가리킴 (`cls`가 자식 클래스가 됨)|부모 클래스에 고정되어 동작|

> **요약하자면:**
> 
> 메서드 내부에서 **"클래스의 속성에 접근해야 하거나, 상속 시 자식 클래스의 정보를 활용해야 한다"**면 `@classmethod`를, 클래스 정보가 전혀 필요 없고 단순 계산이나 기능만 수행한다면 `@staticmethod`를 사용하시면 됩니다.