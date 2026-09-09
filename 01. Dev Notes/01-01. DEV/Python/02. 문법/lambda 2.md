이 코드에서 쓰인 람다 함수 문법을 하나씩 뜯어서 설명해드릴게요.

## 기본 구조

```python
lambda sample: { 'messages': [...] }
```

```python
train_dataset = dataset.map(
    lambda sample: 
    { 'messages' : [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample['instruction']},
        {"role": "assistant", "content": sample['output']}]
    }
)
```

람다는 익명 함수를 한 줄로 만드는 문법입니다. 일반 함수로 풀어 쓰면 이렇게 됩니다:

```python
def transform(sample):
    return {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['instruction']},
            {"role": "assistant", "content": sample['output']}
        ]
    }
```

즉:

- `lambda` → 함수를 정의하겠다는 키워드 (`def`와 같은 역할)
- `sample` → 매개변수 (파라미터). `def transform(sample):`의 `sample`과 동일
- `:` 뒤에 오는 부분 → 함수의 반환값 (return 문 없이 바로 표현식만 씀)

## 왜 이렇게 쓰는가

`dataset.map()`은 데이터셋의 각 행(row)에 함수를 적용시켜주는 메서드입니다. 여기서 `sample`은 데이터셋의 한 행을 나타내는 딕셔너리이고, `sample['instruction']`, `sample['output']`처럼 컬럼명으로 값을 꺼낼 수 있습니다.

```python
dataset.map(
    lambda sample: { 'messages': [...] }
)
```

이는 다음과 동시에 같은 의미입니다:

```python
def transform(sample):
    return { 'messages': [...] }

dataset.map(transform)
```

## 핵심 포인트

1. **람다는 표현식 하나만 가질 수 있다** — `if`, `for` 같은 문장(statement)은 못 쓰고, 조건 표현식(`a if 조건 else b`) 정도만 가능합니다.
2. **자동으로 return 된다** — `:` 뒤의 표현식 결과가 그대로 반환값이 됩니다. 여기선 딕셔너리 `{'messages': [...]}` 자체가 반환값이죠.
3. **딕셔너리를 반환하는 것뿐**, 람다 문법 자체는 단순합니다. 헷갈리는 건 사실 중괄호 `{}`가 두 가지 의미로 겹쳐 보이기 때문입니다:
    - 바깥 `{ 'messages': [...] }` → 딕셔너리 리터럴 (람다의 반환값)
    - 리스트 안의 `{"role": ..., "content": ...}` → 각각 별개의 딕셔너리 (대화 메시지 하나씩)

## 정리하면 흐름은 이렇습니다

```
dataset의 각 행(sample)
  → sample['instruction'], sample['output']을 꺼내서
  → system/user/assistant 3개짜리 메시지 리스트로 감싸고
  → {'messages': [...]} 형태의 새 컬럼으로 반환
```

즉 원래 `instruction`, `output` 컬럼을 chat 형식(`messages`)으로 변환하는 전처리 코드입니다. LLM 파인튜닝 시 자주 쓰는 패턴이에요.