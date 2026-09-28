
`@abstractmethod`는 "자식 클래스가 반드시 구현해야 하는 메서드"를 지정할 때 씁니다. `ABC`를 상속한 클래스에서만 동작합니다.

```python
from abc import ABC, abstractmethod


class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        """자식 클래스가 반드시 구현해야 함"""
        pass

    def introduce(self) -> str:
        # 일반 메서드는 그대로 상속됨
        return f"저는 {self.speak()} 소리를 내요"


class Dog(Animal):
    def speak(self) -> str:
        return "멍멍"


class Cat(Animal):
    def speak(self) -> str:
        return "야옹"


dog = Dog()
print(dog.introduce())  # 저는 멍멍 소리를 내요
print(Cat().speak())    # 야옹
```

## 동작 규칙

추상 클래스는 직접 인스턴스를 만들 수 없습니다.

```python
animal = Animal()
# TypeError: Can't instantiate abstract class Animal without an implementation for abstract method 'speak'
```

자식 클래스가 추상 메서드를 구현하지 않아도 마찬가지로 에러가 납니다.

```python
class Bird(Animal):
    pass  # speak 구현 안 함

Bird()
# TypeError: Can't instantiate abstract class Bird without an implementation for abstract method 'speak'
```

## 정리

- `ABC`를 상속하지 않으면 `@abstractmethod`가 있어도 강제되지 않습니다.
- 에러는 클래스를 정의할 때가 아니라 인스턴스를 만들 때 발생합니다.
- 추상 메서드에도 본문을 작성할 수 있어서 `super().speak()`로 공통 로직을 호출할 수 있습니다.
- `@property`와 함께 쓸 때는 `@property`를 바깥에, `@abstractmethod`를 안쪽에 둡니다.

```python
class Shape(ABC):
    @property
    @abstractmethod
    def area(self) -> float: ...
```