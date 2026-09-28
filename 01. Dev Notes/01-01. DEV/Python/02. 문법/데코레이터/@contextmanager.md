## 핵심 장점

`@contextmanager`의 장점은 `__enter__`/`__exit__`를 가진 클래스를 만들 때 필요한 코드와 복잡함을 없애 준다는 것입니다.

## 1. 코드가 짧고 읽기 쉽다

같은 기능을 클래스와 비교하면 차이가 분명합니다.

```python
# 클래스 방식
class ChangeDir:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.original = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original)
        return False
```

```python
# @contextmanager 방식
@contextmanager
def change_dir(path):
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)
```

## 2. setup과 teardown이 한 함수 안에 모인다

클래스는 준비 코드(`__enter__`)와 정리 코드(`__exit__`)가 다른 메서드로 나뉘고, 둘이 공유할 값을 `self.xxx`에 저장해야 합니다. `@contextmanager`는 **지역 변수를 그대로 공유**하고 위에서 아래로 읽으면 흐름이 이해됩니다.

```python
@contextmanager
def timer(label):
    start = time.perf_counter()        # 준비
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.3f}초")   # 정리 (start를 바로 사용)
```

## 3. 예외 처리가 직관적이다

클래스의 `__exit__`는 `exc_type, exc_val, exc_tb` 세 인자와 반환값(`True`면 예외 삼킴)의 규칙을 알아야 합니다. `@contextmanager`는 **평범한 `try/except/finally`**로 처리합니다.

```python
@contextmanager
def transaction(conn):
    try:
        yield conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

## 4. 자원 정리가 보장된다

`finally`와 함께 쓰면 with 블록에서 예외가 나도 파일, 락, DB 연결 같은 자원이 반드시 정리됩니다. `open()`만 쓰고 `close()`를 빼먹거나 `lock.release()`를 놓치는 실수를 막아 줍니다.

```python
with acquire(lock):
    do_work()   # 예외가 나도 lock은 반드시 해제됨
```

## 5. 반복되는 패턴을 재사용 가능하게 묶는다

"시작 전에 뭔가 하고, 끝나면 되돌리기" 패턴을 함수 하나로 캡슐화해서 프로젝트 전체에서 재사용할 수 있습니다.

```python
with timer("전처리"):
    preprocess()

with temp_env("MODE", "test"):
    run_tests()

with change_dir("/tmp"):
    do_something()
```

## 6. 데코레이터로도 쓸 수 있다

`@contextmanager`로 만든 객체는 함수 데코레이터로도 사용할 수 있습니다.

```python
@timer("함수 실행")
def work():
    ...
```

## 7. 비동기로 확장하기 쉽다

`@asynccontextmanager`로 같은 방식의 비동기 컨텍스트 매니저를 만들 수 있습니다. FastAPI의 `lifespan`처럼 서버 시작과 종료 시 리소스를 관리하는 곳에서 자주 쓰입니다.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    model = load_model()   # 서버 시작 시 (앞에서 다룬 LLM, 임베딩 모델 로드에 활용 가능)
    yield
    del model              # 서버 종료 시
```

## 단점과 클래스가 나은 경우

|상황|추천|
|---|---|
|간단한 setup/teardown|`@contextmanager`|
|인스턴스 상태를 오래 유지하거나 여러 메서드를 제공|클래스|
|같은 객체를 여러 번 재사용 (재진입)|클래스|
|상속으로 동작을 확장|클래스|

- `@contextmanager`로 만든 객체는 **한 번만 사용**할 수 있습니다. 같은 객체를 다시 `with`에 쓰려면 함수를 다시 호출해야 합니다.
- `yield`를 반드시 한 번만 실행해야 하고, `try/finally`를 빼먹으면 예외 시 정리 코드가 실행되지 않습니다.

## 정리

간단한 준비/정리 로직이라면 클래스보다 `@contextmanager`가 적은 코드로 같은 안전성을 제공합니다. 상태 관리나 상속이 필요한 복잡한 경우에만 클래스로 넘어가면 됩니다.