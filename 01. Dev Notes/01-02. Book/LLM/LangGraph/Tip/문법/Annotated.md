
`Annotated[타입, 메타데이터]`는 타입에 부가 정보를 붙이는 파이썬 표준 문법(`typing.Annotated`)입니다. 파이썬 자체는 두 번째 인자를 무시하고, 그것을 읽는 라이브러리가 의미를 정합니다. LangGraph는 두 번째 인자를 **리듀서(reducer)**, 즉 "노드가 반환한 값을 기존 상태와 어떻게 합칠지 정하는 함수"로 읽습니다.

## 구조

```python
result: Annotated[str, operator.add]
#                 ───  ────────────
#                 타입   리듀서(합치는 함수)
```

- 첫 번째 인자 `str`: 이 필드의 실제 타입입니다. 타입 검사기는 `result`를 그냥 `str`로 봅니다.
- 두 번째 인자 `operator.add`: `operator.add(a, b)`는 `a + b`와 같습니다. LangGraph는 노드가 값을 반환할 때마다 `새 상태 = 리듀서(기존 값, 노드가 반환한 값)`으로 계산합니다.

## 필드별 동작

| 필드 | 선언 | 노드가 값을 반환하면 |
|---|---|---|
| `result` | `Annotated[str, operator.add]` | `기존 + 새값`, 문자열이 뒤에 이어 붙음 |
| `logs` | `Annotated[list, operator.add]` | `기존 + 새값`, 리스트가 뒤에 이어 붙음 |
| `txt` | `str` (리듀서 없음) | 새 값으로 덮어씀 |

예를 들어 노드 두 개가 차례로 실행되면 이렇게 됩니다.

```python
def node_a(state):
    return {"result": "A", "logs": ["a 실행"], "txt": "a"}

def node_b(state):
    return {"result": "B", "logs": ["b 실행"], "txt": "b"}

# 최종 상태
# result: "AB"                    ← "" + "A" + "B"
# logs:   ["a 실행", "b 실행"]     ← 리스트 연결
# txt:    "b"                     ← 마지막 값만 남음
```

## 주의할 점

- **`logs`는 리스트로 반환해야 합니다.** `{"logs": "a 실행"}`처럼 문자열을 반환하면 `list + str`이 되어 `TypeError`가 납니다.
- **노드는 추가할 부분만 반환합니다.** `state["logs"] + ["새 로그"]`처럼 전체를 반환하면 기존 내용이 한 번 더 붙어 중복됩니다.
- **병렬 실행에서는 리듀서가 필수입니다.** 같은 단계에서 두 노드가 `txt`처럼 리듀서 없는 필드를 동시에 쓰면 `InvalidUpdateError`가 납니다. 리듀서가 있는 필드는 두 값을 모두 합칩니다.
- **리듀서는 직접 만든 함수여도 됩니다.** `(기존값, 새값) -> 합친값` 형태면 되고, 예를 들어 `Annotated[int, max]`는 더 큰 값만 남깁니다. 메시지 목록에 쓰는 `add_messages`도 같은 방식의 리듀서입니다.