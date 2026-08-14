
## 한 줄 요약

**`yield` 는 함수를 끝내지 않고 일시정지시키는 문법이고, 제너레이터는 그 일시정지된 상태를 담아 두는 객체입니다.** `return` 이 "결과물을 다 만들어 한 번에" 준다면, `yield` 는 "하나씩 만들어 그때그때" 줍니다.

이 문서의 예제는 [`examples/yield_basics.py`](../examples/yield_basics.py) 와 [`examples/yield_demo.py`](../examples/yield_demo.py) 로 직접 실행해 볼 수 있습니다.

```powershell
python -X utf8 examples/yield_basics.py   # 선언과 호출 방법
python -X utf8 examples/yield_demo.py     # 리스트와의 차이를 눈으로 확인
```

한글이 깨지면 `-X utf8` 을 붙이세요.

---

## 1. 관계의 사슬

```
def 안에 yield 가 있다          →  그 함수는 '제너레이터 함수'
    ↓  (컴파일 시점에 결정)
제너레이터 함수를 호출한다       →  '제너레이터 객체'가 나온다 (몸통은 실행 안 됨)
    ↓
제너레이터 객체는 이터레이터다   →  for / list() / sum() 이 그대로 받아 준다
```

핵심은 **컴파일 시점에 결정된다**는 점입니다. 조건이 붙지 않습니다. 몸통에 `yield` 라는 토큰이 하나라도 있으면 — 그게 절대 실행되지 않는 `if False:` 안에 있어도 — 파이썬은 그 함수의 코드 객체에 `CO_GENERATOR` 플래그를 붙입니다. 그 순간부터 호출은 "실행"이 아니라 "제너레이터 생성"이 됩니다.

### 용어 정리

혼동하기 쉬운 세 가지입니다.

| 용어 | 무엇 |
| --- | --- |
| 제너레이터 **함수** | `yield` 가 든 `def`. 호출하면 객체를 준다 |
| 제너레이터 **객체** | 호출 결과. 프레임을 들고 있는 일시정지된 실행 상태 |
| 제너레이터 **표현식** | `(x*x for x in ...)`. 문법만 다른, 같은 제너레이터 객체 |

`jsonl_to_json.py` 의 `iter_records` 는 제너레이터 **함수**이고, `convert()` 안의 `records` 변수는 제너레이터 **객체**입니다. 그래서 반환 타입 힌트가 `Iterator[Any]` 입니다 — 리스트를 주는 게 아니라 "하나씩 꺼내 쓸 수 있는 것"을 준다는 선언입니다.

---

## 2. `yield` 가 정확히 하는 일

`return` 은 함수를 **끝냅니다**. `yield` 는 함수를 **일시정지**시킵니다. 그게 전부이자 핵심입니다.

일시정지하려면 하던 일을 어딘가 보관해야 합니다. 그 보관소가 제너레이터 객체입니다. 실제로 안을 들여다보면 이렇습니다.

```python
def countdown(n):
    total = 0
    while n > 0:
        total += n
        yield n
        n -= 1
```

```
생성 직후 : GEN_CREATED   | 지역변수: {'n': 3}
next ->  3
멈춘 상태 : GEN_SUSPENDED | 지역변수: {'n': 3, 'total': 3}
next ->  2
멈춘 상태 : GEN_SUSPENDED | 지역변수: {'n': 2, 'total': 5}
소진 후   : GEN_CLOSED    | 프레임: None
```

`total` 이 3 → 5 로 **누적되어 있습니다.** 함수가 처음부터 다시 도는 게 아니라, 멈췄던 자리에서 그 변수들을 그대로 들고 이어서 실행된다는 증거입니다. 보통 함수는 반환하면 지역변수가 사라지지만, 제너레이터는 실행 프레임(`gi_frame`)을 객체 안에 붙들고 있습니다. 그래서 `iter_records` 의 `lineno` 와 열린 파일 핸들이 `yield` 를 넘어서도 살아남습니다.

### 제너레이터 객체가 제공하는 것

```
__iter__  __next__  send  throw  close  gi_frame  gi_code  gi_running
```

`__iter__` 와 `__next__` 가 있으니 **이터레이터 프로토콜**을 만족합니다. `iter(g) is g` 가 `True` — 자기 자신을 반환합니다. `for` 문이 하는 일은 결국 `__next__()` 를 반복 호출하다 `StopIteration` 이 오면 멈추는 것이므로, 제너레이터는 별도 준비 없이 `for` / `list()` / `sum()` / 언패킹에 그대로 들어갑니다.

`send` / `throw` / `close` 는 밖에서 안으로 개입하는 통로입니다 — 멈춰 있는 함수에 값을 밀어넣거나, 예외를 던지거나, 정리시킵니다.

### 왜 이게 편한가

`yield` 없이 같은 이터레이터를 만들려면 클래스를 써야 합니다.

```python
class Countdown:                      def countdown(n):
    def __init__(self, n):                while n > 0:
        self.n = n                            yield n
    def __iter__(self):                       n -= 1
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1
```

왼쪽은 "어디까지 진행했는지"를 `self.n` 에 **손으로** 저장합니다. 상태가 복잡해질수록 이 부기가 폭발합니다. 오른쪽은 그 저장을 파이썬이 프레임 보관으로 대신해 줍니다.

**그래서 `yield` 의 진짜 의미는 "값을 하나 내보낸다"가 아니라 "이터레이터를 만드는 지긋지긋한 상태 관리를 컴파일러에게 떠넘긴다"입니다.** 값을 내보내는 건 그 부수 효과에 가깝습니다.

---

## 3. 선언과 호출

### 선언

```python
def greet():
    yield "안녕"
    yield "반가워"
    yield "잘 가"
```

`return` 이 없어도 됩니다. `yield` 만 있으면 제너레이터 함수입니다.

### 호출

```python
gen = greet()          # 몸통은 아직 한 줄도 실행되지 않는다
print(gen)             # <generator object greet at 0x...>

next(gen)              # '안녕'   <- 첫 yield 까지 실행하고 멈춘다
next(gen)              # '반가워'
next(gen)              # '잘 가'
next(gen)              # StopIteration  <- 끝났다는 신호
```

꺼내는 방법은 네 가지입니다.

| 방법 | 예 | 비고 |
| --- | --- | --- |
| `for` 문 | `for v in greet(): ...` | 가장 흔하다. `StopIteration` 을 알아서 처리 |
| `list()` | `list(greet())` | 전부 꺼내 리스트로 |
| 다른 함수에 전달 | `sum(...)`, `max(...)`, `"".join(...)` | 이터러블을 받는 함수면 무엇이든 |
| `next()` | `next(gen, "없음")` | 한 개씩 직접. 두 번째 인자는 소진 시 기본값 |

**주의:** 호출할 때마다 새 제너레이터가 생깁니다. `greet()` 를 네 번 적었다면 제너레이터도 네 개이지 하나를 재사용한 게 아닙니다.

### 조건부 `yield`

`yield` 는 루프 안, `if` 안, 어디에 놔도 됩니다. 실행이 그 줄에 닿을 때 하나 나갑니다. `yield` 하지 않고 `continue` 하면 그 항목은 그냥 사라집니다 — `iter_records` 가 빈 줄을 건너뛰는 방식이 정확히 이것입니다.

```python
def evens_only(numbers):
    for n in numbers:
        if n % 2 == 0:
            yield n          # 짝수일 때만 내보낸다
```

### `return` — 도중에 끝내기

제너레이터 안의 `return` 은 "값을 돌려주는" 게 아니라 "거기서 끝"이라는 뜻입니다. 굳이 값을 붙이면 `StopIteration.value` 에 실려 옵니다.

```python
def take_until_zero(numbers):
    for n in numbers:
        if n == 0:
            return "0을 만나 멈췄습니다"   # 여기서 종료
        yield n

list(take_until_zero([1, 2, 0, 3, 4]))   # [1, 2]  <- 0 뒤는 나오지 않는다
```

### `yield from` — 위임

```python
def outer():
    yield "["
    yield from inner()      # inner 가 내보내는 걸 그대로 흘려보낸다
    yield "]"
```

`yield from inner()` 는 `for value in inner(): yield value` 와 같은 뜻입니다.

---

## 4. 리스트와 무엇이 다른가

같은 일을 두 방식으로 쓰면 이렇습니다.

```python
# 리스트                              # yield
def make_list(n):                     def make_gen(n):
    result = []                           for i in range(1, n + 1):
    for i in range(1, n + 1):                 yield i * i
        result.append(i * i)
    return result
```

### (1) 실행 시점

`examples/yield_demo.py` 실험 1의 실제 출력입니다.

```
[리스트] 호출하는 순간 ...        [yield] 호출하는 순간 ...
    [만드는 중] 1                 -> 아무것도 출력되지 않았다
    [만드는 중] 2                    <generator object ...>
    [만드는 중] 3                 이제 하나씩 꺼내 본다:
  -> 결과: [1, 4, 9]                 [만드는 중] 1
  이제 하나씩 꺼내 본다:              [받음] 1
    [받음] 1                         [만드는 중] 2
    [받음] 4                         [받음] 4
    [받음] 9                         [만드는 중] 3
                                     [받음] 9
```

리스트는 만들기가 다 끝난 뒤 받기가 시작되고, `yield` 는 하나씩 번갈아 일어납니다.

### (2) 메모리

100만 개를 `tracemalloc` 으로 실측한 값입니다.

```
리스트로 모으기      :    38.57 MB
yield 로 흘려보내기  :     0.00 MB
```

리스트는 파싱된 객체 전부가 동시에 메모리에 올라가지만, `yield` 는 항상 **한 개 분량**만 올라갑니다. 입력이 10배 커져도 사용량은 그대로입니다. `jsonl_to_json` 모듈 docstring의 "입력이 커져도 메모리 사용량은 일정하다"가 이 얘기입니다.

### (3) 첫 결과까지 걸리는 시간

한 건에 0.2초 걸리는 작업 5건:

```
[리스트] 1번째 결과 도착 ... 1.01초   (전부 끝나야 나온다)
[yield ] 1번째 결과 도착 ... 0.20초   (바로 나오고 계속 흐른다)
```

`convert()` → `write_json()` 이 읽기와 쓰기를 번갈아 하므로, 표준 출력으로 뽑으면 결과가 즉시 흐르기 시작합니다. 표준 입력(`-`)으로 파이프를 연결할 때 특히 체감됩니다.

### (4) 한 번 쓰면 끝 — `yield` 의 대가

리스트는 몇 번이든 다시 순회할 수 있지만 제너레이터는 소모품입니다.

```python
gen = (v for v in [1, 2, 3])
list(gen)      # [1, 2, 3]
list(gen)      # []            <- 이미 다 써 버렸다
len(gen)       # TypeError: object of type 'generator' has no len()
```

그래서 `write_json` 은 `count` 를 손으로 세고, 리스트가 필요한 쪽을 위해 `load()` (= `list(iter_records(...))`) 를 따로 둡니다.

### (5) 예외가 터지는 시점

리스트 방식은 함수 호출 그 자리에서 예외가 납니다. `yield` 방식은 **깨진 줄에 도달했을 때** 나므로, `write_json` 이 이미 `[` 와 앞부분 레코드를 써 놓은 뒤에 터집니다. `_convert_to_file` 이 임시 파일에 먼저 쓰고 성공했을 때만 `replace()` 하는 이유입니다 — 잘린 JSON이 목적지에 남는 걸 막습니다.

### (6) 리스트로는 불가능한 것

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

for value in fibonacci():
    if value > 1000:
        break            # 필요한 만큼만 받고 멈춘다
```

리스트로 만들려면 무한 루프에 빠집니다.

---

## 5. 언제 무엇을 쓰나

| 상황 | 선택 |
| --- | --- |
| 데이터가 크거나 크기를 모름 | `yield` |
| 결과를 흘려보내며 처리 (스트리밍, 파이프) | `yield` |
| 무한 수열 | `yield` |
| 여러 번 순회하거나 `len()`, 인덱싱이 필요 | 리스트 `return` |
| 데이터가 작고 코드 단순함이 우선 | 리스트 `return` |

`jsonl_to_json` 은 둘 다 제공합니다 — 스트리밍이 필요하면 `iter_records()`, 편하게 쓰고 싶으면 `load()`.

---

## 참고: 같은 키워드, 다른 용도

`jsonl_to_json.py` 의 `_open_text` 에 있는 `yield` 는 값을 흘려보내는 게 아닙니다.

```python
@contextmanager
def _open_text(source):
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig") as handle:
            yield handle          # <- 여기가 with 블록 본문이 실행될 자리
    else:
        yield source
```

`@contextmanager` 와 짝을 이뤄 "**여기가 `with` 블록 본문이 실행될 자리**"를 표시하는 용도입니다. `yield` 앞은 진입 처리, 뒤는 정리 처리가 됩니다. 문법은 같지만 목적이 다릅니다.
