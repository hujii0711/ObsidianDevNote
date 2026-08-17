
파이썬에서 `for ... in` 문에 사용할 수 있는 객체들을 반복 가능한 객체(Iterable)라고 부릅니다. 쉽게 말해, 내부 요소들을 한 개씩 차례대로 꺼낼 수 있는 데이터 타입들을 의미합니다.

파이썬의 대표적인 기본 데이터 타입들 중 `for in`문과 함께 쓸 수 있는 것들을 정리해 드릴게요.

## 1. 대표적인 Iterable 데이터 타입들

### ① 리스트 (List) & 튜플 (Tuple)

순서가 있는 데이터의 집합으로, 가장 흔하게 `for`문과 함께 사용됩니다.

```python
# 리스트 순회
for item in [1, 2, 3]:
    print(item)

# 튜플 순회
for item in ("A", "B", "C"):
    print(item)
```

### ② 문자열 (String)

문자열도 하나의 문자(Character)들이 나열된 시퀀스(Sequence) 타입이므로, 한 글자씩 꺼내올 수 있습니다.

```python
for char in "Python":
    print(char)  # P, y, t, h, o, n이 한 줄씩 출력됨
```

### ③ 딕셔너리 (Dictionary)

딕셔너리를 그냥 `for in`에 넣으면 기본적으로 키(Key)를 꺼내옵니다. 메서드를 활용해 값이나 쌍을 꺼낼 수도 있습니다.

```python
my_dict = {"name": "Alice", "age": 25}

for key in my_dict:          # 키 순회
    print(key) 

for val in my_dict.values(): # 값 순회
    print(val)
```

### ④ 집합 (Set)

중복을 허용하지 않는 집합 타입도 순회가 가능합니다. 단, 집합은 **순서가 없기 때문에** 출력되는 순서가 매번 달라질 수 있습니다.

```python
for num in {1, 2, 3, 3, 3}:  # 중복은 제거됨
    print(num)
```

## 2. 자주 함께 쓰이는 내장 함수 및 Generator

데이터 타입 자체는 아니지만, `for in`문과 결합하여 반복을 만들어내는 특수한 객체들입니다.

- **`range()` 함수:** 특정 횟수만큼 반복하거나 연속된 숫자를 만들 때 필수적입니다.
```python
    for i in range(3): # 0, 1, 2
       print(i)
```

- **`enumerate()` 함수:** 순회할 때 요소뿐만 아니라 인덱스(몇 번째인지)를 함께 꺼내줍니다.

```python
    for idx, name in enumerate(["Kim", "Lee"]):
        print(f"{idx}번: {name}") # 0번: Kim, 1번: Lee
```
 
- **`zip()` 함수:** 여러 개의 리스트를 엮어서 동시에 하나씩 꺼낼 때 사용합니다.

```python
    for fruit, color in zip(["apple", "banana"], ["red", "yellow"]):
        print(f"{fruit}은 {color}")
```


## ⚠️ `for in`문을 사용할 수 없는 타입 (Non-iterable)

단일 값을 나타내는 데이터 타입들은 꺼낼 내부 요소가 없기 때문에 `for in`문에 넣으면 **`TypeError: '...' object is not iterable`** 에러가 발생합니다.

- **숫자형:** `int` (정수), `float` (실수)
- **논리형:** `bool` (`True`, `False`)
- **기타:** `None`

```python
# ❌ 에러 발생 예시
for i in 12345:  # TypeError: 'int' object is not iterable
    print(i)
```

> 숫자를 5번 반복하고 싶다면 `12345`가 아니라 `range(5)`처럼 iterable한 객체로 변환해 주어야 합니다.