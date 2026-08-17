
파이썬 함수 인자의 전체 종류를 순서대로 정리해서 설명해드릴게요.

## 1. 일반 인자 (Positional arguments)

호출 시 순서대로 값이 전달되는 기본 인자입니다.

```python
def greet(name, age):
    print(f"{name}는 {age}살입니다.")

greet("철수", 20)
```

## 2. 기본값 인자 (Default arguments)

기본값을 지정해두면 호출 시 생략 가능합니다.

```python
def greet(name, age=20):
    print(f"{name}는 {age}살입니다.")

greet("철수")       # age=20 사용
greet("영희", 25)   # age=25로 덮어씀
```

## 3. 가변 위치 인자 (`*args`)

개수 제한 없이 위치 인자들을 **튜플**로 모아 받습니다.

```python
def add_all(*args):
    print(args)  # 튜플
    return sum(args)

add_all(1, 2, 3, 4)  # (1, 2, 3, 4) -> 10
```


## 4. 키워드 전용 인자 (Keyword-only arguments)

`*` 또는 `*args` **뒤에 위치**하여, 반드시 `키워드=값` 형태로만 전달해야 하는 인자입니다.

```python
def greet(name, *, age, city="서울"):
    print(f"{name}, {age}살, {city}")

greet("철수", age=20)              # OK
greet("철수", age=20, city="부산")  # OK
greet("철수", 20)                  # 에러!
```

#### 보충: 키워드 전용 구분자 (`*`)

`*` 하나만 단독으로 쓰면, 이름도 없고 값도 저장하지 않는 **구분 기호**입니다. "이 뒤에 오는 인자는 반드시 키워드로만 전달하라"는 표시만 합니다.

```python
def func(a, *, b):
    print(a, b)

func(1, b=2)   # OK
func(1, 2)     # 에러! b는 위치로 전달 불가
```

> ⚠️ `*` 뒤에는 반드시 최소 1개 이상의 인자가 와야 합니다. `def func(a, *):` 처럼 뒤에 아무것도 없으면 SyntaxError 발생.

**`*args` vs 단독 `*` 비교**

|표현|값 저장 여부|역할|
|---|---|---|
|`*args`|O (튜플로 수집)|남은 위치 인자를 모음 + 이후 키워드 전용 강제|
|`*`|X (수집 안 함)|이후 인자를 키워드 전용으로 강제만 함|
## 5. 가변 키워드 인자 (`**kwargs`)

개수 제한 없이 키워드 인자들을 **딕셔너리**로 모아 받습니다.

```python
def show_info(**kwargs):
    print(kwargs)  # 딕셔너리

show_info(name="철수", age=20, city="서울")
# {'name': '철수', 'age': 20, 'city': '서울'}
```

## 전체 순서 & 종합 예제

인자를 함께 쓸 때 정의 순서는 다음과 같이 **고정**되어 있습니다.

```
일반 인자 → 기본값 인자 → *args (또는 단독 *) → 키워드 전용 인자 → **kwargs
```

```python
def func(a, b=10, *args, c, d=20, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"c={c}, d={d}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, 5, c=100, e=999, f=888)
# a=1, b=2
# args=(3, 4, 5)
# c=100, d=20
# kwargs={'e': 999, 'f': 888}
```

### 요약표

|종류|문법|저장 형태|특징|
|---|---|---|---|
|일반 인자|`a`|-|순서대로 필수 전달|
|기본값 인자|`b=10`|-|생략 가능|
|가변 위치 인자|`*args`|튜플|개수 제한 없는 위치 값 수집|
|키워드 전용 구분자|`*`|- (수집 안 함)|이후 인자를 키워드 전용으로 강제만 함|
|키워드 전용 인자|`c` (★ 뒤에 위치)|-|반드시 `키워드=값`으로 전달|
|가변 키워드 인자|`**kwargs`|딕셔너리|개수 제한 없는 키워드 값 수집|

더 궁금한 부분이나 실습해보고 싶은 예제 있으면 말씀해주세요!