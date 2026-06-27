
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