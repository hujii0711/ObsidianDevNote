
파이썬의 **람다(lambda) 함수**는 이름 없이 **한 줄로 간단하게 만드는 함수**예요. `def`로 함수를 만드는 것과 같은 역할을 하지만, 훨씬 짧고 간단한 문법을 씁니다.

## 1. 기본 문법

```python
lambda 매개변수1, 매개변수2, ... : 표현식
```

- `lambda` 키워드로 시작
- 콜론(`:`) 앞에는 **매개변수**
- 콜론(`:`) 뒤에는 **반환할 값(표현식)** — `return`을 따로 안 써도 자동으로 반환됨
- **딱 한 줄, 표현식 하나만** 쓸 수 있음 (if문, for문 등 여러 줄 로직은 불가)

## 2. 가장 기본적인 예제

### 일반 함수 vs 람다 함수 비교

```python
# 일반 함수로 정의
def add(a, b):
    return a + b

print(add(3, 5))
```

```python
# 람다 함수로 정의
add = lambda a, b: a + b

print(add(3, 5))
```

**출력 (둘 다 동일):**

```
8
```

### 어떻게 대응되는지 보기

|일반 함수 (`def`)|람다 함수 (`lambda`)|
|---|---|
|`def add(a, b):`|`lambda a, b:`|
|`return a + b`|`a + b`|
|함수 이름 `add`에 저장됨|변수 `add`에 저장됨|

`lambda a, b: a + b`에서:

- `a, b` → 매개변수 (일반 함수의 `(a, b)`와 같은 역할)
- `a + b` → 계산 결과를 **자동으로 반환** (별도의 `return` 불필요)

## 3. 여러 가지 람다 예제

```python
# 제곱 계산
square = lambda x: x ** 2
print(square(5))        # 25

# 두 수 중 큰 값 구하기
max_num = lambda a, b: a if a > b else b
print(max_num(3, 7))    # 7

# 짝수인지 판별
is_even = lambda x: x % 2 == 0
print(is_even(4))       # True
print(is_even(7))       # False

# 매개변수 없는 람다
greet = lambda: "안녕하세요"
print(greet())          # 안녕하세요
```

`max_num` 예제처럼 **조건식(삼항 연산자)**은 람다 안에서도 쓸 수 있어요. `a if a > b else b`는 "a가 b보다 크면 a, 아니면 b"라는 뜻입니다.

## 4. 람다는 왜 쓰나요? — 진짜 쓰임새

사실 람다는 `add = lambda a, b: a + b`처럼 **변수에 저장해서 쓰는 경우는 드물고**, 주로 **다른 함수의 인자로 즉석에서 넘길 때** 진가를 발휘해요. 대표적으로 `map()`, `filter()`, `sorted()`와 함께 자주 쓰입니다.

### (1) `sorted()`와 함께 — 정렬 기준 지정

```python
students = [("철수", 85), ("영희", 92), ("민수", 78)]

# 점수(두 번째 값) 기준으로 정렬
result = sorted(students, key=lambda x: x[1])
print(result)
```

**출력:**

```
[('민수', 78), ('철수', 85), ('영희', 92)]
```

`key=lambda x: x[1]`은 **"각 튜플에서 x[1](https://claude.ai/chat/%EC%A0%90%EC%88%98)을 기준으로 정렬해줘"**라는 뜻이에요. 이때 이 람다 함수를 위해 `def`로 따로 함수를 정의하기엔 너무 사소하죠? 이럴 때 람다가 딱 좋습니다.

### (2) `map()`과 함께 — 리스트의 모든 원소에 함수 적용

```python
numbers = [1, 2, 3, 4, 5]

# 모든 원소를 제곱하기
squared = list(map(lambda x: x ** 2, numbers))
print(squared)
```

**출력:**

```
[1, 4, 9, 16, 25]
```

### (3) `filter()`와 함께 — 조건에 맞는 값만 골라내기

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 짝수만 골라내기
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)
```

**출력:**

```
[2, 4, 6, 8, 10]
```

## 5. `def`로 바꿔서 비교해보기

`sorted()` 예제를 `def`로 바꾸면 이렇게 돼요:

```python
def get_score(x):
    return x[1]

result = sorted(students, key=get_score)
```

동작은 완전히 같지만, **이 함수가 딱 한 번, 정렬할 때만 쓰이는 사소한 용도**라면 굳이 이름까지 지어서 `def`로 만들 필요가 없죠. 이럴 때 람다로 **"이름 짓기 귀찮은 일회용 함수"**를 즉석에서 만들어 쓰는 거예요.

## 6. 정리 표

|특징|`def` 함수|`lambda` 함수|
|---|---|---|
|이름|있음|없음 (익명 함수)|
|코드 줄 수|여러 줄 가능|딱 한 줄(표현식 하나)만 가능|
|`return`|명시적으로 써야 함|자동으로 반환됨|
|주 사용처|재사용할 로직, 복잡한 로직|`map`, `filter`, `sorted` 등에 즉석으로 넘길 때|

## 7. 주의할 점

람다 안에서는 **여러 줄의 코드나 `if`문(조건문 단독), `for`문**은 쓸 수 없어요. 오직 **표현식 하나**만 가능합니다.

```python
# ❌ 이런 건 불가능 (여러 줄 로직)
# calc = lambda x: 
#     if x > 0:
#         return "양수"

# ✅ 조건 표현식(삼항연산자)은 가능
calc = lambda x: "양수" if x > 0 else "음수 또는 0"
print(calc(5))    # 양수
print(calc(-3))   # 음수 또는 0
```

정리하면, 람다는 **"이름 붙일 필요 없는, 짧고 간단한 함수를 즉석에서 만들 때"** 쓰는 문법이라고 기억하시면 돼요. 특히 `map`, `filter`, `sorted`의 `key` 인자처럼 **"함수를 값처럼 다른 함수에 바로 넘겨야 하는 상황"**에서 진가를 발휘합니다!