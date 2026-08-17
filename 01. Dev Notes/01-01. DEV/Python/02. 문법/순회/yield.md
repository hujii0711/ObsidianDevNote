
## 기본 개념

`return`은 값을 **한 번** 반환하고 함수가 종료되지만,  
`yield`는 값을 **하나씩** 반환하고 함수가 **일시 정지** 상태로 유지됩니다.

```python
# return - 한 번에 모든 값 반환
def return_func():
    return [1, 2, 3]  # 리스트 전체를 메모리에 올림

# yield - 하나씩 순서대로 반환
def yield_func():
    yield 1  # 여기서 일시 정지
    yield 2  # 다음 호출 시 여기서 재개
    yield 3
```

---

## 실행 흐름 비교

```python
def generator():
    print("1번 시작")
    yield 10          # ① 반환 후 일시정지
    print("2번 시작")
    yield 20          # ② 반환 후 일시정지
    print("3번 시작")
    yield 30          # ③ 반환 후 일시정지
    print("끝")

gen = generator()     # 아직 실행 안 됨!

print(next(gen))      # "1번 시작" 출력 → 10 반환
print(next(gen))      # "2번 시작" 출력 → 20 반환
print(next(gen))      # "3번 시작" 출력 → 30 반환
# next(gen)           # StopIteration 예외 발생
```

```
출력:
1번 시작
10
2번 시작
20
3번 시작
30
```

---

## 제너레이터 순회

```python
def count_up(n):
    for i in range(n):
        yield i

# for문으로 순회 (가장 일반적)
for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4

# list로 변환
print(list(count_up(5)))  # [0, 1, 2, 3, 4]

# next()로 하나씩
gen = count_up(3)
print(next(gen))  # 0
print(next(gen))  # 1
```

---

## return vs yield 메모리 차이

```python
import sys

# return - 100만개 리스트를 한번에 메모리에 올림
def use_return(n):
    return [i * 2 for i in range(n)]

# yield - 하나씩 그때그때 생성
def use_yield(n):
    for i in range(n):
        yield i * 2

result_return = use_return(1_000_000)
result_yield  = use_yield(1_000_000)

print(sys.getsizeof(result_return))  # 8,448,728 bytes (약 8MB)
print(sys.getsizeof(result_yield))   # 104 bytes ← 제너레이터 객체만!
```

---

## yield from (중첩 제너레이터)

```python
def inner():
    yield 1
    yield 2

def outer():
    yield from inner()   # inner의 yield를 그대로 위임
    yield from [3, 4, 5] # 이터러블도 가능
    yield 6

print(list(outer()))  # [1, 2, 3, 4, 5, 6]
```


``` python
# 기존 방식 (`for` + `yield`)
def sub_generator():
    yield 1
    yield 2

def main_generator():
    # for 문으로 일일이 꺼내서 yield 해야 함
    for value in sub_generator():
        yield value

for item in main_generator():
    print(item)  # 출력: 1, 2
```


```python
# `yield from` 방식 (간결함)
# `yield from`을 사용하면 `for` 루프 없이 한 줄로 깔끔하게 축약할 수 있습니다.
def sub_generator():
    yield 1
    yield 2

def main_generator():
    # sub_generator()의 모든 yield를 그대로 전달함
    yield from sub_generator()

for item in main_generator():
    print(item)  # 출력: 1, 2
```


---

## async yield (비동기 제너레이터)

```python
import asyncio

async def async_generator():
    for i in range(5):
        await asyncio.sleep(1)  # 비동기 대기
        yield i                 # 값 반환

# 사용
async def main():
    async for value in async_generator():
        print(value)  # 1초마다 0, 1, 2, 3, 4 출력

asyncio.run(main())
```

> FastAPI SSE에서 사용한 패턴이 바로 이것!

---

## 실전 활용 패턴

```python
# 1. 대용량 파일 읽기
def read_large_file(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()  # 한 줄씩 처리 (메모리 절약)

# 2. 무한 수열
def fibonacci():
    a, b = 0, 1
    while True:       # 무한 루프지만 yield로 안전하게 사용
        yield a
        a, b = b, a + b

fib = fibonacci()
print([next(fib) for _ in range(8)])  # [0, 1, 1, 2, 3, 5, 8, 13]

# 3. 파이프라인 처리
def read_data():
    for i in range(10):
        yield i

def filter_even(data):
    for item in data:
        if item % 2 == 0:
            yield item

def multiply(data, factor):
    for item in data:
        yield item * factor

# 체이닝
result = multiply(filter_even(read_data()), 10)
print(list(result))  # [0, 20, 40, 60, 80]
```

---

## 핵심 정리

| 항목 | `return` | `yield` |
|------|----------|---------|
| 반환 횟수 | 1회 | 여러 번 |
| 함수 종료 | 즉시 종료 | 일시 정지 |
| 메모리 | 전체 저장 | 하나씩 생성 |
| 반환 타입 | 값 그대로 | 제너레이터 객체 |
| 사용 상황 | 일반 함수 | 대용량/스트리밍 |

**한 줄 요약:** `yield`는 **"지금 이 값 줄게, 나중에 다시 와"** 라고 실행을 멈추는 마법의 키워드!


---

## 파이썬의 yield는 스트림 처리와 유사한가?

개념적으로 상당히 유사합니다. 핵심을 짚어보겠습니다.

## 왜 비슷한가

`yield`를 사용하는 제너레이터(generator)는 **모든 데이터를 한 번에 메모리에 올리지 않고, 필요할 때마다 하나씩 값을 흘려보낸다**는 점에서 스트림 처리와 철학이 같습니다.

```python
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()  # 한 줄씩 "흘려보냄"

for line in read_large_file("huge.log"):
    process(line)
```

이 코드는 파일 전체를 리스트로 읽어들이지 않고, 한 줄씩 생성-소비를 반복합니다. Java의 Stream API, Node.js의 Readable Stream, Unix 파이프(`cmd1 | cmd2`)와 같은 지연 평가(lazy evaluation) + 순차적 데이터 흐름이라는 점에서 동일한 패턴입니다.

## 공통점

| 특징           | 스트림                         | `yield` (제너레이터) |
| ------------ | --------------------------- | --------------- |
| 지연 계산        | O (필요할 때 계산)                | O               |
| 메모리 효율       | O (전체를 안 들고 있음)             | O               |
| 파이프라인 구성     | O (`stream.map().filter()`) | O (제너레이터 체이닝)   |
| 무한 시퀀스 처리 가능 | O                           | O               |

제너레이터끼리 연결하면 실제로 파이프라인처럼 동작합니다:

```python
def numbers():
    n = 0
    while True:
        yield n
        n += 1

def evens(stream):
    for x in stream:
        if x % 2 == 0:
            yield x

def squared(stream):
    for x in stream:
        yield x ** 2

pipeline = squared(evens(numbers()))
for i, val in zip(range(5), pipeline):
    print(val)  # 0, 4, 16, 36, 64
```

이건 함수형 언어나 RxJava/RxPY 같은 리액티브 스트림 라이브러리에서 하는 `map`/`filter` 체이닝과 본질적으로 같은 구조입니다.

## 차이점 (구분해둘 부분)

- **Push vs Pull**: `yield`는 기본적으로 **pull 기반**입니다 — 소비자가 `next()`를 호출해야 값이 나옵니다. 반면 진짜 "스트림"(예: 이벤트 스트림, Reactive Streams)은 종종 **push 기반**으로, 생산자가 준비되는 대로 값을 밀어 넣습니다.
- **비동기 여부**: `yield`는 기본적으로 동기적입니다. 비동기 스트림을 다루려면 `async def` + `yield`로 만드는 **비동기 제너레이터**(`async for`로 순회)가 필요합니다.
- **백프레셔(backpressure)**: Node.js 스트림처럼 버퍼가 가득 찼을 때 생산 속도를 자동 조절하는 기능은 순수 제너레이터엔 없습니다. pull 기반이라 소비자가 속도를 자연스럽게 제어하긴 하지만, 명시적인 백프레셔 메커니즘은 아닙니다.

## 정리

`yield`는 파이썬에서 **스트림 처리를 구현하는 가장 기본적이고 가벼운 도구**라고 보면 정확합니다. 다만 "스트림"이라는 단어가 가리키는 더 큰 개념(비동기, push 기반, 백프레셔 등)의 일부만 다루는 것이니, 진짜 대규모 스트림 처리(예: Kafka consumer, 비동기 이벤트 스트림)를 만든다면 `asyncio`의 비동기 제너레이터나 관련 라이브러리(`aiostream` 등)로 확장해서 써야 합니다.