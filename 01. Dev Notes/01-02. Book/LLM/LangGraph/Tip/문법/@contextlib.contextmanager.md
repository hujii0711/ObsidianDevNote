
파이썬의 `@contextlib.contextmanager`는 **일반 함수를 컨텍스트 매니저(Context Manager)로 만들어 주는 데코레이터**입니다.
주로 `with` 구문에서 사용하는 리소스의 생성, 사용, 정리 과정을 간결하게 구현할 때 사용합니다.

### **1. 기본 개념**
일반적으로 파일을 열 때 다음과 같이 사용합니다.
```python
with open("test.txt", "w") as f:
    f.write("Hello Python")
```

여기서 `with`는 다음 작업을 자동으로 처리합니다.
1. 리소스 생성 및 초기화
2. 작업 수행
3. 리소스 정리 및 종료

`@contextlib.contextmanager`를 사용하면 이와 같은 동작을 직접 구현할 수 있습니다.

### **2. 기본 문법**
```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("1. 시작")
    yield
    print("3. 종료")

with my_context():
    print("2. 작업 수행")
```

실행 결과
```
1. 시작
2. 작업 수행
3. 종료
```

핵심은 `yield`입니다.

|구문|역할|
|---|---|
|`@contextmanager`|함수를 컨텍스트 매니저로 변환|
|`yield` 이전|리소스 초기화 및 사전 작업|
|`yield`|`with` 블록에 제어권 전달|
|`yield` 이후|`with` 블록 종료 후 정리 작업|

즉, `yield`를 기준으로 실행 흐름이 나뉩니다.

### **3. 실제 활용 예제**

#### **예제 1. 데이터베이스 연결 관리**
```python
from contextlib import contextmanager

@contextmanager
def db_connection():
    print("DB 연결")

    conn = {"connected": True}

    try:
        yield conn
    finally:
        conn["connected"] = False
        print("DB 연결 종료")


with db_connection() as conn:
    print("DB 작업 수행")
    print(conn)
```

실행 결과:
```
DB 연결
DB 작업 수행
{'connected': True}
DB 연결 종료
```

여기서 중요한 것은 `finally`입니다.
`with` 블록 내부에서 예외가 발생하더라도 `finally`가 실행되어 정리 작업을 수행합니다.

#### **예제 2. 실행 시간 측정**
AI 에이전트나 LangGraph 노드 실행 시간을 측정할 때도 유용합니다.

```python
import time
from contextlib import contextmanager

@contextmanager
def measure_time(name):
    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.3f}초")

with measure_time("LLM 실행"):
    time.sleep(1)

with measure_time("데이터 검색"):
    time.sleep(0.5)
```

실행 결과:
```
LLM 실행: 1.001초
데이터 검색: 0.500초
```

#### **예제 3. 예외 처리**
`yield` 이후 코드가 반드시 실행되는지 확인하는 예제입니다.
```python
from contextlib import contextmanager

@contextmanager
def transaction():
    print("트랜잭션 시작")

    try:
        yield
        print("트랜잭션 커밋")
    except Exception:
        print("트랜잭션 롤백")
        raise
    finally:
        print("트랜잭션 종료")


with transaction():
    print("데이터 저장")
    raise ValueError("저장 실패")
```

실행 결과:
```
트랜잭션 시작
데이터 저장
트랜잭션 롤백
트랜잭션 종료
Traceback ...
ValueError: 저장 실패
```

### **4.** `yield`**에 값을 전달하는 방법**

`yield`에 객체를 전달하면 `with ... as`에서 해당 값을 받을 수 있습니다.
```python
from contextlib import contextmanager

@contextmanager
def get_resource():
    resource = {
        "name": "MyResource",
        "status": "active"
    }

    try:
        yield resource
    finally:
        resource["status"] = "closed"
        print("리소스 정리")


with get_resource() as resource:
    print(resource["name"])
    print(resource["status"])
```

실행 결과:
```
MyResource
active
리소스 정리
```

### **5. 일반 Context Manager와 비교**
파이썬에서는 컨텍스트 매니저를 구현하는 방법이 크게 두 가지입니다.

**방법 1. 클래스 기반**
```python
class MyContext:

    def __enter__(self):
        print("시작")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("종료")

with MyContext():
    print("작업 수행")
```

**방법 2. 함수 기반**
```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("시작")

    try:
        yield
    finally:
        print("종료")


with my_context():
    print("작업 수행")
```

|구분|클래스 방식|contextmanager 방식|
|---|---|---|
|구현|`__enter__`, `__exit__`|`yield`|
|코드량|상대적으로 많음|간결함|
|복잡한 상태 관리|유리함|상대적으로 제한적|
|단순 리소스 관리|가능|매우 적합|

### **6. 알아두면 좋은 추가 문법**
`contextlib`에는 함께 알아두면 좋은 기능들이 있습니다.

|기능|설명|
|---|---|
|`@contextmanager`|함수 기반 컨텍스트 매니저 생성|
|`closing()`|객체의 `close()` 자동 호출|
|`suppress()`|지정한 예외 무시|
|`ExitStack`|여러 컨텍스트 매니저를 동적으로 관리|
|`nullcontext()`|아무 작업도 하지 않는 컨텍스트 매니저|

특히 `ExitStack`은 여러 리소스를 동적으로 관리해야 하는 에이전트 시스템에서 유용합니다.

### **7. 핵심적으로 기억할 부분**
`@contextlib.contextmanager`는 다음과 같은 패턴으로 이해하면 됩니다.

**초기화 코드**
DB 연결, 파일 열기, 타이머 시작

**yield**
with 블록으로 제어권 전달

**정리 코드**
DB 연결 종료, 파일 닫기, 타이머 종료

**추가로 중요한 특징 2가지**
- `yield`는 기본적으로 한 번만 사용해야 합니다.
- `yield` 이후의 정리 코드는 `try/finally`로 보호하는 것이 안전합니다.

---

LangGraph에서 노드 실행 전후의 상태 관리, 실행 시간 측정, 리소스 관리 등을 구현할 때 활용할 수 있는 문법입니다.
### `lock.py`에서 `@contextlib.contextmanager`를 쓰는 이유

`run_lock`이 하는 일은 "잠금을 잡고 → 그래프를 실행하고 → 무슨 일이 있어도 잠금을 푼다"입니다. 이렇게 앞뒤로 정리할 일이 있는 작업은 `with` 문으로 쓰는 게 파이썬의 기본 방식입니다. `@contextlib.contextmanager`를 붙이면 클래스를 따로 만들지 않고 함수 하나로 `with`에 쓸 도구를 만들 수 있습니다.

### `yield`를 기준으로 앞뒤가 나뉩니다
```python
@contextlib.contextmanager
def run_lock(out_root):
    with open(out_root / "run.lock", "a") as f:
        fcntl.flock(f, LOCK_EX | LOCK_NB)   # ① with에 들어갈 때: 잠금 잡기 (실패하면 LockBusy)
        try:
            yield                            # ② 여기서 멈추고 with 블록(그래프 실행)이 돈다
        finally:
            fcntl.flock(f, LOCK_UN)          # ③ with를 나갈 때: 잠금 풀기
```
- ① `yield` 앞은 `with`에 들어갈 때 실행됩니다.
- ② `yield`에서 함수가 멈추고, 그동안 `with` 블록 안의 코드가 실행됩니다.
- ③ `yield` 뒤는 `with` 블록이 끝날 때 실행됩니다.

### 이 파일에서 특히 중요한 점

**1. 실패해도 잠금이 반드시 풀립니다**
그래프 실행은 몇 분 걸리고, 도중에 LLM 서버가 꺼져 있거나 검색이 실패하는 등 예외가 날 수 있습니다. 예외가 `with` 블록에서 나도 `yield`를 감싼 `try/finally` 덕분에 ③이 항상 실행됩니다. 잠금이 풀리지 않으면 다음 실행이 계속 "이미 실행 중"(종료 코드 2)으로 막힙니다.

**2. 잠금을 못 잡으면 블록 자체가 실행되지 않습니다**
①에서 `LockBusy`가 나면 `yield`까지 가지 않습니다. 그래서 그래프는 시작되지 않고, 예외는 호출한 쪽으로 올라가 error 이벤트가 됩니다. 이렇게 웹에서 생성 중일 때 CLI로 또 실행하는 것을 막습니다.

**3. 파일 닫기도 함께 처리됩니다**
바깥의 `with open(...)` 덕분에 잠금 파일도 자동으로 닫힙니다. 덧붙여 `flock` 잠금은 파일이 닫히거나 프로세스가 죽어도 운영체제가 풀어 줍니다. 그래서 프로세스가 강제로 종료돼도 잠금이 영구히 남지 않습니다.

**4. 필요할 때만 잠금을 거는 데 잘 맞습니다**
[cli/main.py:253-255](shorts_graph/cli/main.py#L253-L255)에서는 실행 명령에만 잠금을 겁니다.
```python
with ExitStack() as stack:
    if args.command in ("start", "resume"):
        stack.enter_context(run_lock(out_root))
```
`start`·`resume`처럼 그래프를 실행하는 명령만 잠금을 잡고, `status`·`prefs` 같은 조회 명령은 잠금 없이 돕니다. `run_lock`이 컨텍스트 매니저라서 `ExitStack.enter_context`로 조건에 따라 붙일 수 있습니다. 붙인 잠금은 `ExitStack`이 끝날 때 함께 풀립니다.

### 데코레이터 없이 쓴다면
같은 기능을 클래스로 만들면 `__enter__`(잠금 잡기)와 `__exit__`(잠금 풀기)를 따로 구현하고, 열린 파일도 `self`에 보관해야 합니다. 데코레이터를 쓰면 이 과정이 함수 하나에 위에서 아래로 읽히는 순서대로 담깁니다.