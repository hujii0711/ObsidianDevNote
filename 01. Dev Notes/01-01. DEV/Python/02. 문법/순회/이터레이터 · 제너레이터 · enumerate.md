
셋은 **같은 층위의 개념이 아닙니다.** 이게 핵심입니다.

| | 정체 | 관계 |
|---|---|---|
| **이터레이터** | 규약(프로토콜) | 가장 큰 개념 |
| **제너레이터** | 이터레이터를 만드는 **가장 쉬운 방법** | 이터레이터의 부분집합 |
| **enumerate** | 이터레이터를 돌려주는 **내장 함수** | 이터레이터를 만드는 도구 중 하나 |

```
이터러블 (Iterable)  ─ for 문에 넣을 수 있는 모든 것
└─ 이터레이터 (Iterator)  ─ 스스로 next() 를 아는 것
   ├─ 제너레이터 (Generator)  ─ yield 로 만든 것
   ├─ enumerate 객체
   ├─ zip / map / filter 객체
   └─ list_iterator, file 객체 ...
```

실제로 확인한 결과입니다.

```python
from collections.abc import Iterable, Iterator, Generator

lst = [1, 2, 3]
it = iter(lst)
gen = (v for v in lst)
enu = enumerate(lst)
```

```
객체            Iterable  Iterator  Generator   type
list            True      False     False       list
iter(list)      True      True      False       list_iterator
generator       True      True      True        generator
enumerate       True      True      False       enumerate
```

---

## 1. 이터러블 vs 이터레이터 — 가장 많이 헷갈리는 지점

**리스트는 이터레이터가 아닙니다.** `for` 문에 넣을 수 있지만 `__next__` 가 없습니다.

```python
lst = [1, 2, 3]
hasattr(lst, "__next__")  # False
next(lst)  # TypeError
```

`for` 문이 내부에서 `iter(lst)` 를 호출해 **매번 새 이터레이터를 만들어 쓰기** 때문에 여러 번
순회할 수 있는 것입니다.

```python
iter(lst) is lst  # False  <- 매번 새 객체
iter(gen) is gen  # True   <- 자기 자신
iter(enu) is enu  # True
```

[../examples/yield_demo.py](../examples/yield_demo.py) 의 "실험 4"에서 리스트는 몇 번이든 다시
볼 수 있는데 제너레이터는 2차에 비어 있는 이유가 정확히 이 차이입니다. 리스트는 순회 상태를
자기가 들고 있지 않고, 제너레이터는 들고 있어서 소진되면 끝입니다.

| | 이터러블 | 이터레이터 |
|---|---|---|
| 필요한 메서드 | `__iter__` | `__iter__` + `__next__` |
| `iter(x) is x` | 아님 | **맞음** |
| 재사용 | 가능 | 불가 (일회용) |
| 상태 보유 | 없음 | 현재 위치를 기억 |
| 예 | list, dict, str, set | 제너레이터, enumerate, zip, 파일 객체 |

---

## 2. 이터레이터 vs 제너레이터 — 같은 결과, 다른 노동량

이터레이터는 **직접 구현할 수도** 있습니다. 아래 두 코드는 결과가 완전히 같습니다.

```python
class CountDown:  # 손으로 만든 이터레이터
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1


def countdown(n):  # 제너레이터 함수
    while n > 0:
        yield n
        n -= 1
```

```
손으로 만든 이터레이터: [3, 2, 1]
제너레이터 함수      : [3, 2, 1]
제너레이터 표현식    : [3, 2, 1]
```

**제너레이터는 이 보일러플레이트를 파이썬이 대신 써주는 문법 설탕**입니다. `yield` 가 함수 안에
하나라도 있으면 그 함수는 호출 시 몸통을 실행하지 않고 제너레이터 객체를 돌려줍니다 —
`yield_demo.py` 의 "실험 1"에서 호출 직후 아무것도 출력되지 않는 그 동작입니다.

수동 구현에서 신경 써야 할 것들을 전부 자동으로 해줍니다.

* 지역 변수와 실행 위치를 알아서 보존 (수동이면 `self.n` 같은 필드로 직접 관리)
* 끝나면 `StopIteration` 자동 발생
* `__iter__` 가 자기 자신을 반환

### 제너레이터에만 있는 것

```
제너레이터에만 있는 메서드: ['send', 'throw', 'close']
CountDown 에는          : []
```

`send()` 로 값을 **주입**할 수 있고(코루틴), `close()` 로 중간에서 정리할 수 있습니다.
그래서 "모든 제너레이터는 이터레이터지만, 모든 이터레이터가 제너레이터는 아니다"가
성립합니다.

### 두 가지 문법, 하나의 결과

```python
def squares(n):  # 제너레이터 함수
    for i in range(n):
        yield i * i


squares_expr = (i * i for i in range(n))  # 제너레이터 표현식
```

**완전히 같은 물건**입니다. 문법만 다릅니다.

### 언제 클래스로 직접 만드나

거의 없습니다. 다만 `len()` 을 지원해야 하거나, 여러 번 순회 가능해야 하거나, 순회 상태를
외부에서 조작해야 하면 클래스가 필요합니다. 이 프로젝트의
[iter_records](../src/pythontest/jsonl_to_json.py) 처럼 "한 번 흘려보내면 되는" 경우엔
제너레이터가 정답입니다.

---

## 3. enumerate — 이터레이터를 만들어 주는 내장 함수

`enumerate` 는 "이터러블을 받아 `(인덱스, 값)` 튜플을 흘려보내는 **이터레이터를 반환하는
함수**"입니다. 제너레이터는 아니고(C로 구현됨), 이터레이터입니다.

```python
list(enumerate(["a", "b"]))  # [(0, 'a'), (1, 'b')]
list(enumerate(["a", "b"], start=10))  # [(10, 'a'), (11, 'b')]
```

### 게으릅니다

리스트를 미리 만들지 않아서 **무한 제너레이터에도 붙일 수 있습니다.**

```python
def naturals():
    i = 0
    while True:
        i += 1
        yield i


for idx, val in enumerate(naturals(), start=1):
    if idx > 3:
        break
    print(idx, val)  # 1 1 / 2 2 / 3 3
```

`yield_demo.py` 의 무한 피보나치에도 그대로 쓸 수 있습니다.

### 역시 일회용입니다

```
enumerate 1차: [(0, 'a'), (1, 'b')]
enumerate 2차: []          <- 비었다
```

이터레이터니까 당연합니다. 제너레이터와 똑같은 성질입니다.

### `start` 는 값이 아니라 라벨입니다

이게 실무에서 헷갈리는 부분입니다. `start=10` 은 **10번째부터 시작하라**는 뜻이 아니라
**번호를 10부터 매기라**는 뜻입니다. 원소는 처음부터 다 나옵니다.

이 프로젝트에서 그 용법을 쓰고 있습니다.

```python
# jsonl_to_json.py — 파일 줄 번호는 1부터
for lineno, raw_line in enumerate(handle, start=1):
    ...
```

사람이 읽을 오류 메시지에 "2번째 줄"이라고 쓰려면 0이 아니라 1부터여야 하니까요.

---

## 실무에서의 선택 기준

```python
# 나쁨: 인덱스가 필요한데 range(len())
for i in range(len(items)):
    print(i, items[i])

# 좋음: enumerate
for i, item in enumerate(items):
    print(i, item)
```

```python
# 나쁨: 이미 이터레이터인데 카운터를 수동 관리
i = 0
for line in file:
    i += 1

# 좋음: enumerate 는 이터레이터에도 붙는다
for i, line in enumerate(file, start=1):
    ...
```

세 가지를 한 줄로 정리하면:

* **이터레이터** = "다음 값 하나 주세요"에 답할 수 있는 규약
* **제너레이터** = 그 규약을 `yield` 한 줄로 구현하는 문법
* **enumerate** = 아무 이터러블에 번호를 붙여 이터레이터로 감싸는 도구

---

## 이 저장소에서 확인해 보기

```powershell
uv run python examples/yield_demo.py
```

관련 코드:

* [examples/yield_demo.py](../examples/yield_demo.py) — yield 가 있을 때와 없을 때의 6가지 실험
* [src/pythontest/jsonl_to_json.py](../src/pythontest/jsonl_to_json.py) — `iter_records` 는
  제너레이터, `load` 는 그걸 `list()` 로 감싼 것
* [src/pythontest/xml_to_jsonl.py](../src/pythontest/xml_to_jsonl.py) — `iter_records` 가
  `iterparse` 스트리밍 위에 얹힌 제너레이터

---
### 보충

- iterable
for문과 같은 반복 구문에 적용할 수 있는 리스트와 같은 객체를 반복 가능(iterable) 객체라고 한다.

- iterator(이터레이터)
이터레이터는 next 함수 호출 시 계속 그 다음 값을 리턴하는 객체이다.
리스트는 반복 가능하지만 이터레이터는 아니다. 하지만 반복 가능하다면 iter 함수를 이용해 이터레이터로 만들 수 있다.
이터레이터의 값을 가져오는 가장 일반적인 방법은 for 문을 이용하는 것이다.
for 문을 이용하면 자동으로 값을 호출하므로 next 함수를 따로 쓸 필요도 없다. 다만 for 문으로 출력 후 다시 반복하더라도 더는 그 값을 가져오지 못한다. 즉, for 문이나 next로 그 값을 한 번 읽으면 그 값을 다시는 읽을 수 없다는 특징이 있다.

```python
a = [1, 2, 3]
ia = iter(a)
for i in ia:
	print(i)
```

- generator(제너레이터)
제너레이터는 이터레이터를 생성해주는 함수이다. 제너레이터로 생성한 객체는 이터레이터와 마찬가지로 next 함수 호출시 그 값을 차례대로 얻을 수 있다. 이때 제너레이터에서는 차례대로 결과를 반환하고자 return 대신 yield 키워드를 사용한다. return 대신 yield를 사용하는 함수는 제너레이터이다.