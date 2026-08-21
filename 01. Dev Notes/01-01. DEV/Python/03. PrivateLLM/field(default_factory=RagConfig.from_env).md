
`rag: RagConfig = field(default_factory=RagConfig.from_env)` 코드를 하나씩 뜯어서 설명드리겠습니다.

## 왜 이렇게 써야 하는가

### 문제가 되는 잘못된 방법

```python
@dataclass
class Settings:
    rag: RagConfig = RagConfig.from_env()  # ❌ 위험한 코드
```

이렇게 쓰면 `RagConfig.from_env()`가 **클래스 정의 시점에 딱 한 번만 실행**됩니다. 그리고 그 결과로 만들어진 **동일한 RagConfig 객체 하나**가 모든 `Settings` 인스턴스의 기본값으로 공유됩니다.

```python
s1 = Settings()
s2 = Settings()

s1.rag.top_k = 10  # s1의 rag 설정을 바꿨을 뿐인데
print(s2.rag.top_k)  # 놀랍게도 10 출력! (같은 객체를 참조하므로)
```

이건 Python에서 `def func(x=[])`처럼 **가변(mutable) 객체를 기본값으로 직접 쓰면 안 되는 것**과 동일한 문제입니다. `RagConfig`는 `@dataclass`로 만든 가변 객체이기 때문에 이 함정에 걸립니다.

### 올바른 해결책: `field(default_factory=...)`

```python
rag: RagConfig = field(default_factory=RagConfig.from_env)
```

이렇게 쓰면:

1. **`RagConfig.from_env`를 함수 자체(참조)로 넘깁니다** — `RagConfig.from_env()`처럼 **괄호를 붙여 호출하지 않습니다**.
2. `Settings()`로 **인스턴스를 새로 만들 때마다** 파이썬이 `default_factory`에 등록된 함수를 **그때그때 호출**해서 새 `RagConfig` 객체를 만들어줍니다.
3. 결과적으로 인스턴스마다 **독립적인 RagConfig 객체**를 가지게 됩니다.

```python
s1 = Settings()
s2 = Settings()

s1.rag.top_k = 10
print(s2.rag.top_k)  # 6 (기본값 그대로, 서로 영향 없음) ✅
```

## 각 부분 문법 설명

|부분|의미|
|---|---|
|`rag:`|필드 이름|
|`RagConfig`|타입 힌트 (이 필드는 `RagConfig` 타입이어야 함을 명시)|
|`field(...)`|`dataclasses` 모듈이 제공하는 함수로, 필드의 세부 동작(기본값, 기본 팩토리 등)을 커스터마이징|
|`default_factory=RagConfig.from_env`|"기본값이 필요할 때 `RagConfig.from_env()`를 호출해서 만들어라"는 지시 (함수 참조만 전달, 호출 X)|

## 실행 흐름 정리

```python
# 1. Settings() 를 호출하면 (인자 없이)
settings = Settings()

# 2. 내부적으로 dataclass가 자동 생성한 __init__은 대략 이렇게 동작합니다:
def __init__(self, rag=None, mlx_model=..., max_tokens=768, temperature=0.2):
    if rag is None:
        rag = RagConfig.from_env()  # ← 여기서 실제로 호출됨 (매번 새로)
    self.rag = rag
    ...
```

즉, `RagConfig.from_env`라는 "함수 이름표"만 들고 있다가, **실제 인스턴스 생성 시점에** 그 함수를 실행해서 새 객체를 만드는 지연 실행(lazy evaluation) 방식입니다.

## 주석에 있는 실제 결과와 연결

주석에 적힌 예시를 보면:

```python
Settings(
    rag=RagConfig(chroma_dir=PosixPath('/Users/fujii0711/Claude/PrivateLLM/data/chroma'), 
                  collection='jeonse_deposit', 
                  model_name='BAAI/bge-m3', 
                  top_k=6, 
                  min_similarity=0.35),
    mlx_model='mlx-community/Qwen2.5-7B-Instruct-4bit',
    max_tokens=768,
    temperature=0.2
)
```

이는 `Settings()`를 호출했을 때 `default_factory=RagConfig.from_env`가 실행되어, 환경 변수 `CHROMA_DIR`이 설정되어 있으면 그 경로를, 없으면 기본 경로(`data/chroma`)를 담은 `RagConfig` 객체가 자동으로 만들어져 채워진 결과입니다.

## 요약

- **`field(default_factory=함수)`**: dataclass에서 가변 객체(리스트, 딕셔너리, 커스텀 클래스 인스턴스 등)를 기본값으로 안전하게 지정하는 표준 방법
- **핵심 규칙**: 함수를 **호출하지 말고 참조만** 넘긴다 (`RagConfig.from_env`, `RagConfig.from_env()` 아님)
- **효과**: 인스턴스 생성마다 독립된 새 객체가 만들어져, 인스턴스 간 상태 공유 버그를 방지