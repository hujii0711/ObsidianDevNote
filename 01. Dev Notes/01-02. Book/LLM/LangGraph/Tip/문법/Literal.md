
한 줄로 말하면, **"이 함수는 `Command`를 돌려주고, 그 `Command`가 보낼 수 있는 곳은 `"write_script"` 또는 `"approve_script"` 두 곳뿐"**이라는 타입 표시입니다. [script/nodes.py:82](shorts_graph/script/nodes.py#L82)에 있습니다.

### 1. `Literal`: 정확히 이 값들만

`Literal`은 파이썬 `typing` 모듈의 타입 표시입니다. 이 파일 맨 위에서 `from typing import Literal`로 가져옵니다.

```python
x: str                                    # 아무 문자열이나
x: Literal["write_script", "approve_script"]  # 이 두 문자열 중 하나만
```
- `str`은 "문자열이면 다 됨"이라는 뜻입니다.
- `Literal[...]`은 "괄호 안에 적은 **바로 그 값들** 중 하나"라는 뜻입니다. 쓸 수 있는 값을 열거해 두는 것과 같습니다.
- 대괄호 `[]`는 리스트가 아닙니다. 타입에 세부 정보를 붙이는 문법입니다. `list[str]`(문자열을 담은 리스트)과 같은 방식입니다.

### 2. `Command[...]`: Command인데 갈 곳이 정해진 것

`Command`는 LangGraph의 클래스입니다. 노드가 **"상태를 이렇게 바꾸고(update), 다음엔 여기로 가라(goto)"**를 한 번에 돌려줄 때 씁니다.

```python
return Command(goto="write_script", update={...})    # 102줄
return Command(goto="approve_script", update={...})  # 111줄
```
`Command[Literal["write_script", "approve_script"]]`는 **"`goto`에 들어가는 값이 이 두 이름 중 하나인 Command"**라는 뜻입니다. 함수 안의 두 `return`이 정확히 그 두 곳으로 갑니다.

전체를 풀어 쓰면 이렇습니다.
```python
def guard_script(state, runtime) -> Command[Literal["write_script", "approve_script"]]:
#                                    ↑ 돌려주는 것: Command
#                                            ↑ 그 Command가 갈 수 있는 곳: 이 두 노드
```

### 3. 실제로 무엇을 막아 주나

**파이썬은 실행 중에 이 표시를 검사하지 않습니다.** 타입 표시는 사람과 도구를 위한 안내일 뿐입니다. 예를 들어 `goto="write_scrpit"`처럼 오타를 내도, 실행해 보기 전까지 파이썬은 아무 말도 하지 않습니다. 이 표시가 도움이 되는 곳은 다음과 같습니다.
- **코드를 읽는 사람**: 함수 본문을 다 읽지 않아도 시그니처만 보고 "이 노드 다음은 대본 재작성이나 사람 확인"이라는 걸 압니다.
- **편집기와 타입 검사기**(VS Code의 Pylance, mypy 등): 오타나 목록에 없는 이름을 경고해 줄 수 있습니다.
- **LangGraph**: 노드 함수의 반환 타입이 `Command[Literal[...]]`이면 LangGraph가 이 표시를 읽어 갈 수 있는 곳을 알아냅니다.

### 4. 이 프로젝트에서는 `destinations`와 함께 씀

같은 정보가 [graph.py:87-91](shorts_graph/graph.py#L87-L91)에도 적혀 있습니다.
```python
graph.add_node(
    "guard_script",
    node_guard_script(deps),
    destinations=("write_script", "approve_script"),
)
```
`guard_script`에서 나가는 `add_edge`가 없으니, 그래프는 이 노드가 어디로 갈 수 있는지 다른 방법으로 알아야 합니다. 그래프 그림에 화살표를 그릴 때도 이 정보가 필요합니다. 반환 타입의 `Literal`이나 `destinations`가 그 역할을 하는데, 이 프로젝트는 `destinations`로 분명히 적어 두었습니다. 그래서 `Literal` 표시는 주로 **읽는 사람과 편집기를 위한 문서** 역할을 합니다. 갈 곳을 바꾸려면 **두 곳을 함께 고쳐야** 합니다.

같은 문법은 [research/nodes.py](shorts_graph/research/nodes.py)의 `route_after_input(...) -> Literal["search", "__end__"]`에도 있습니다. 이쪽은 `Command` 없이, 라우터 함수가 다음 노드 이름(문자열)을 바로 돌려주는 경우입니다.