
파이썬에서 `{}`는 문맥에 따라 **딕셔너리(dict)** 또는 **집합(set)**을 만드는 데 쓰입니다.

## 1. 딕셔너리 선언

`{}`만 단독으로 쓰면 빈 딕셔너리가 됩니다.

```python
# 빈 딕셔너리
d = {}
print(type(d))  # <class 'dict'>

# key: value 쌍으로 선언
person = {
    "name": "홍길동",
    "age": 30,
    "city": "서울"
}
```

### 사용법

```python
# 값 접근
print(person["name"])       # 홍길동
print(person.get("age"))    # 30
print(person.get("job", "없음"))  # 없는 키는 기본값 반환 → 없음

# 값 추가/수정
person["job"] = "개발자"
person["age"] = 31

# 값 삭제
del person["city"]

# 키 존재 확인
if "name" in person:
    print("name 키가 있습니다")

# 반복
for key, value in person.items():
    print(key, value)

for key in person.keys():
    print(key)

for value in person.values():
    print(value)
```

## 2. 집합(set) 선언

`{}`는 비어있으면 딕셔너리로 인식되므로, **빈 집합은 반드시 `set()`으로 만들어야 합니다.**

```python
s = {1, 2, 3}       # 집합
print(type(s))      # <class 'set'>

empty_set = set()   # 빈 집합 (❌ {} 아님, {}는 dict)
```

### 사용법

```python
s = {1, 2, 3}
s.add(4)          # 추가
s.remove(2)        # 삭제
print(3 in s)       # 포함 여부 확인

# 집합 연산
a = {1, 2, 3}
b = {2, 3, 4}
print(a | b)   # 합집합 {1, 2, 3, 4}
print(a & b)   # 교집합 {2, 3}
print(a - b)   # 차집합 {1}
```

## 3. 컴프리헨션으로도 생성 가능

```python
# 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 집합 컴프리헨션
unique_squares = {x**2 for x in range(-3, 4)}
# {0, 1, 4, 9}
```

## 핵심 정리

| 표현          | 결과         |
| ----------- | ---------- |
| `{}`        | 빈 **딕셔너리** |
| `{1, 2, 3}` | **집합**     |
| `{"a": 1}`  | **딕셔너리**   |
| `set()`     | 빈 **집합**   |

더 자세히 알고 싶은 부분(예: 중첩 딕셔너리, JSON 변환 등)이 있으면 말씀해 주세요!

---

파이썬에서 **set(집합)**을 만드는 방법은 여러 가지가 있습니다.

## 1. 중괄호 `{}`로 만들기 (원소가 있을 때만)

```python
s = {1, 2, 3}
print(type(s))  # <class 'set'>

s2 = {"apple", "banana", "cherry"}
```

⚠️ 주의: `{}`만 단독으로 쓰면 **빈 딕셔너리**가 됩니다. 빈 집합은 안 됩니다.

```python
empty = {}
print(type(empty))  # <class 'dict'>  ❌ 집합 아님
```

## 2. `set()` 함수로 만들기

빈 집합을 만들 때는 반드시 이 방법을 써야 합니다.

```python
empty_set = set()
print(type(empty_set))  # <class 'set'>
```

`set()`은 다른 자료형(리스트, 튜플, 문자열 등)을 집합으로 변환할 때도 사용합니다.

```python
s1 = set([1, 2, 2, 3, 3, 3])   # 리스트 → 집합, 중복 제거됨
print(s1)  # {1, 2, 3}

s2 = set((1, 2, 3))            # 튜플 → 집합
print(s2)  # {1, 2, 3}

s3 = set("hello")              # 문자열 → 집합 (문자 단위로 분해)
print(s3)  # {'h', 'e', 'l', 'o'}
```

## 3. 집합 컴프리헨션(Set Comprehension)

```python
squares = {x**2 for x in range(5)}
print(squares)  # {0, 1, 4, 9, 16}

evens = {x for x in range(10) if x % 2 == 0}
print(evens)  # {0, 2, 4, 6, 8}
```

## 4. `frozenset()` — 수정 불가능한 집합

일반 집합은 수정 가능하지만, `frozenset`은 한 번 만들면 변경할 수 없습니다 (딕셔너리 키나 다른 집합의 원소로 쓸 때 유용).

```python
fs = frozenset([1, 2, 3])
print(fs)  # frozenset({1, 2, 3})
# fs.add(4)  # ❌ 에러 발생! 수정 불가
```

## 핵심 정리

|방법|예시|결과|
|---|---|---|
|중괄호 (원소 있음)|`{1, 2, 3}`|집합|
|`set()`|`set()`|**빈 집합** (유일한 방법)|
|`set(iterable)`|`set([1,2,2])`|중복 제거된 집합|
|컴프리헨션|`{x for x in range(5)}`|집합|
|`frozenset()`|`frozenset([1,2])`|수정 불가 집합|

**참고:** set의 원소는 반드시 **해시 가능(hashable)**해야 하므로, 리스트나 딕셔너리처럼 변경 가능한 객체는 원소로 넣을 수 없습니다.

```python
# s = {[1, 2], 3}  # ❌ TypeError: unhashable type: 'list'
s = {(1, 2), 3}     # ✅ 튜플은 가능
```