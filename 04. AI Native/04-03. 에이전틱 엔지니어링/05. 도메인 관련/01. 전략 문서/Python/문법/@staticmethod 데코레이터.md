
`@staticmethod`(정적 메서드)는 `@classmethod`와 마찬가지로 **인스턴스(객체)를 생성하지 않고도 클래스를 통해 바로 호출할 수 있는 메서드**입니다.

가장 큰 특징은 메서드를 정의할 때 `self`나 `cls` 같은 **첫 번째 인자를 받지 않는다**는 점입니다. 즉, 클래스나 인스턴스의 상태(데이터)에 전혀 관여하지 않고, 독립적으로 기능을 수행하는 함수를 클래스 안에 묶어두고 싶을 때 사용합니다.

## 1. @staticmethod의 핵심 특징

- **인자가 없음:** 일반 메서드의 `self`나 클래스 메서드의 `cls`처럼 자동으로 넘어오는 매개변수가 없습니다. 일반 함수와 똑같이 동작합니다.
    
- **독립적인 기능:** 클래스 변수나 인스턴스 변수를 수정하거나 읽을 필요가 없을 때 사용합니다.
    
- **네임스페이스(Namespace) 정리:** 클래스와 밀접한 관련이 있는 유틸리티(도움) 함수를 클래스 내부로 밀어 넣어 코드를 깔끔하게 정리하는 역할을 합니다.

## 2. 기본적인 사용법 예시

간단한 계산기 클래스를 통해 두 수의 합을 구하는 정적 메서드를 만들어 보겠습니다.

```python
class Calculator:
    @staticmethod
    def add(a, b):
        # self나 cls를 쓰지 않고, 전달받은 인자값으로만 동작합니다.
        return a + b

    @staticmethod
    def is_even(num):
        return num % 2 == 0

# 객체를 생성하지 않고 클래스 이름으로 바로 호출
result1 = Calculator.add(10, 20)
result2 = Calculator.is_even(7)

print(result1)  # 출력: 30
print(result2)  # 출력: False
```

이 예시에서 `add`나 `is_even` 함수는 `Calculator`라는 클래스의 데이터(상태)를 건드릴 필요가 없습니다. 다만 기능적으로 "계산기"에 속하는 게 자연스럽기 때문에 클래스 안에 `@staticmethod`로 묶어둔 것입니다.

## 3. 그림으로 보는 차이점 (`self` vs `cls` vs `staticmethod`)

메서드가 호출될 때 어디에 접근할 수 있는지를 비교하면 이해하기 쉽습니다.

- **인스턴스 메서드 (`self`):** 인스턴스 영역과 클래스 영역 모두에 접근 가능합니다.
    
- **클래스 메서드 (`cls`):** 클래스 영역에 접근할 수 있지만, 개별 인스턴스 영역에는 접근할 수 없습니다.
    
- **정적 메서드 (`@staticmethod`):** 인스턴스와 클래스 영역 그 어디에도 직접 접근할 수 없으며, 오직 매개변수로 넘어온 데이터로만 작동합니다.
    

## 4. 실무에서는 언제 쓸까요?

보통 데이터의 형식이 올바른지 검증하는 **유틸리티 함수**를 만들 때 자주 사용됩니다.

```python
class User:
    def __init__(self, email, password):
        self.email = email
        self.password = password

    # 이메일 형식이 올바른지 검증하는 정적 메서드
    @staticmethod
    def validate_email(email):
        return "@" in email and "." in email

# 객체를 만들기 전에 이메일 주소가 유효한지 먼저 체크할 수 있습니다.
email_input = "user@example.com"

if User.validate_email(email_input):
    user = User(email_input, "secure_password123")
    print("회원가입 성공!")
else:
    print("유효하지 않은 이메일 형식입니다.")
```

> **한 줄 요약:**
> 
> `@staticmethod`는 클래스나 인스턴스의 데이터를 쓰지 않으면서, 기능적으로 그 클래스와 연관된 **'단순 도구(Utility) 함수'**를 만들 때 사용하면 됩니다.