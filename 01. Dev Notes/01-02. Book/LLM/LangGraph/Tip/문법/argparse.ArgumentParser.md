
선택하신 44번째 줄은 `argparse.ArgumentParser`를 **그대로 쓰지 않고 상속해서 `_Parser`를 만드는** 부분입니다. 이유는 두 가지로 나눠 볼 수 있습니다.

## 1. argparse를 쓰는 이유

이 CLI에는 하위 명령이 다섯 개(`start`, `resume`, `status`, `prefs`, `graph`) 있고, 명령마다 옵션이 다릅니다([main.py:79-127](shorts_graph/cli/main.py#L79-L127)). argparse는 표준 라이브러리라서 따로 설치할 것이 없고, 다음 일을 대신 해 줍니다.

- `--thread`가 빠졌는지 같은 필수 인자 검사, `--auto` 같은 플래그 처리
- `-- "검색어"`처럼 `-`로 시작하는 검색어도 위치 인자로 받기
- `--help` 출력 자동 생성. 여기에 `_EPILOG`(resume 입력 형식, 종료 코드)를 붙여 사용법 문서 역할도 합니다.

## 2. 상속해서 `error()`를 바꾸는 이유 (이 줄의 핵심)

argparse는 기본적으로 인자가 잘못되면 **stderr에 사용법을 찍고 `sys.exit(2)`로 프로그램을 끝냅니다**. 이 프로젝트에서는 이 동작이 두 가지 약속과 부딪힙니다.

1. **종료 코드 2가 겹칩니다.** 이 CLI에서 2는 "실행 잠금 실패(이미 실행 중인 작업이 있음)"입니다(`_EPILOG`의 종료 코드 표). 인자 오류도 2로 끝나면 웹이 "다른 작업이 돌고 있다"와 "인자를 잘못 줬다"를 구분할 수 없습니다. 주석의 `argparse 기본 동작(종료 코드 2)은 잠금 실패와 겹친다`가 이 얘기입니다.
2. **stdout의 JSON 이벤트 약속이 깨집니다.** 웹(`shorts_web`)은 stdout을 한 줄씩 JSON으로 파싱합니다. argparse가 그냥 종료해 버리면 `error` 이벤트가 하나도 나오지 않아서, 웹은 왜 실패했는지 알 수 없습니다.

그래서 `_Parser.error()`가 프로그램을 끝내는 대신 `_ArgError` 예외를 냅니다([main.py:44-49](shorts_graph/cli/main.py#L44-L49)). 그러면 `main()`이 이 예외를 `GraphError`와 같은 방식으로 받아서 처리합니다([main.py:299-301](shorts_graph/cli/main.py#L299-L301)).

```python
except (GraphError, _ArgError) as e:
    emit({"type": "error", "thread_id": thread_id, "message": str(e)})
    return 1
```

결과적으로 인자 오류도 다른 오류처럼 **`{"type":"error", ...}` 이벤트 한 줄 + 종료 코드 1**로 끝납니다.

`add_subparsers(..., parser_class=_Parser)`도 같은 이유로 들어가 있습니다. 이걸 빼면 `start`, `resume` 같은 하위 명령의 파서는 기본 `ArgumentParser`로 만들어지고, 예를 들어 `resume`에서 `--thread`를 빠뜨리면 다시 `exit(2)`로 끝나 버립니다.

참고로 `--help`는 `error()`를 거치지 않고 argparse가 직접 `sys.exit(0)`을 부르기 때문에, 도움말은 원래대로 출력되고 정상 종료합니다.

---
### `add_subparsers`

`add_subparsers`는 `git commit`, `git push`처럼 **하나의 프로그램 안에 하위 명령을 여러 개 두고, 명령마다 받는 인자를 따로 정할 때** 씁니다. `shorts_graph` CLI([main.py:85](shorts_graph/cli/main.py#L85))가 이 구조입니다. (지금 열려 있는 `demo/sse_cli/cli.py`에서는 쓰지 않습니다.)

## 명령마다 필요한 인자가 다릅니다

| 명령 | 받는 인자 |
|---|---|
| `start` | `query`(필수), `--auto`, `--thread-id` |
| `resume` | `--thread`(필수), `--input` |
| `status` | `--thread`(필수) |
| `prefs` | `--set` |
| `graph` | 없음 |

파서 하나에 옵션을 전부 넣으면 이렇게 됩니다.

- `--thread`를 필수로 둘 수 없습니다. `start`에는 필요 없기 때문에, "resume이면 `--thread`가 있어야 한다" 같은 검사를 직접 짜야 합니다.
- `status --auto`처럼 그 명령과 관계없는 옵션을 줘도 오류가 나지 않습니다.
- `--help`에 모든 옵션이 섞여 나와서 어떤 명령에 어떤 옵션을 쓰는지 알기 어렵습니다.

하위 파서를 쓰면 이런 검사를 argparse가 명령별로 해 줍니다. `resume`에 `--thread`가 빠졌거나 `status`에 `--auto`를 주면 오류로 거르고, `uv run -m shorts_graph resume --help`는 resume의 옵션만 보여 줍니다.

## 인자 세 개의 역할

```python
sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
```

- **`dest="command"`**: 사용자가 고른 명령 이름을 `args.command`에 넣습니다. 아래 코드가 이 값으로 할 일을 나눕니다([main.py:249-261](shorts_graph/cli/main.py#L249-L261)).
  ```python
  if args.command == "graph": ...
  if args.command == "start": ...
  if args.command == "resume": ...
  ```
- **`required=True`**: 명령 없이 `uv run -m shorts_graph`만 실행하면 오류로 처리합니다. 이게 없으면 `args.command`가 `None`인 채로 아래 코드로 넘어갑니다.
- **`parser_class=_Parser`**: 하위 파서도 `_Parser`로 만듭니다. 앞에서 설명한 것과 같은 이유입니다. 이게 빠지면 하위 명령의 인자 오류는 `error` 이벤트를 내지 못하고 종료 코드 2로 끝나는데, 2는 잠금 실패 코드와 겹칩니다.

---

선택하신 부분은 **최상위 파서를 만들고, 그 아래에 `start` 하위 명령과 `--auto` 옵션을 붙이는** 코드입니다. 각 값이 `--help` 화면 어디에 나오는지 같이 보면 이해가 쉽습니다.

## 1. 최상위 파서 `_Parser(...)`

```python
p = _Parser(
    prog="shorts_graph",
    description="뉴스 쇼츠 LangGraph 파이프라인 CLI (JSON 줄 이벤트 출력)",
    epilog=_EPILOG,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
```

| 인자 | 하는 일 |
|---|---|
| `prog` | 사용법 줄에 나오는 프로그램 이름입니다. 지정하지 않으면 argparse가 `sys.argv[0]`에서 가져오는데, `python -m`으로 실행하면 `__main__.py`처럼 엉뚱한 이름이 나옵니다. 그래서 직접 정해 둡니다. |
| `description` | `--help`에서 사용법 줄 바로 아래에 나오는 한 줄 설명입니다. |
| `epilog` | `--help` 맨 아래에 붙는 글입니다. 여기서는 `_EPILOG`(resume 입력 형식, 예시 흐름, 종료 코드)를 넣습니다. |
| `formatter_class` | argparse는 기본적으로 description과 epilog의 줄바꿈과 공백을 없애고 한 문단으로 다시 정렬합니다. 그렇게 되면 `_EPILOG`의 표 정렬이 다 깨집니다. `RawDescriptionHelpFormatter`를 쓰면 **쓴 그대로** 출력됩니다. |

`uv run -m shorts_graph --help`를 실행하면 이렇게 나옵니다.

```
usage: shorts_graph [-h] {start,resume,status,prefs,graph} ...    ← prog

뉴스 쇼츠 LangGraph 파이프라인 CLI (JSON 줄 이벤트 출력)            ← description

positional arguments:
  {start,resume,status,prefs,graph}
    start               새 실행을 시작한다                          ← add_parser의 help
    ...

resume --input 형식 (질문 종류별)                                   ← epilog (정렬 유지)
  pick_article   {"choice": N}                          N번째 기사를 고른다 (0부터)
```

## 2. 하위 명령 묶음 `add_subparsers(...)`

앞 답변에서 설명한 부분입니다. 고른 명령 이름을 `args.command`에 넣고(`dest`), 명령을 꼭 주게 하고(`required`), 하위 파서도 `_Parser`로 만듭니다(`parser_class`).

## 3. `start` 하위 명령 `sub.add_parser(...)`

```python
start = sub.add_parser(
    "start", help="새 실행을 시작한다", description="검색어로 새 실행을 시작한다"
)
```

`help`와 `description`은 비슷해 보이지만 나오는 곳이 다릅니다.

- **`help`**: 상위 도움말(`shorts_graph --help`)의 명령 목록에 나오는 짧은 설명입니다. 위 출력의 `start  새 실행을 시작한다`가 이 값입니다.
- **`description`**: `start`의 자체 도움말(`shorts_graph start --help`)에 나오는 설명입니다.

`add_parser`가 돌려준 `start`도 파서 객체라서, 여기에 `add_argument`로 `start` 전용 인자를 붙입니다.

## 4. `--auto` 옵션

```python
start.add_argument(
    "--auto", action="store_true", help="사람 확인 없이 끝까지 자동 실행"
)
```

- `action="store_true"`: 값 없이 켜고 끄는 플래그입니다. `--auto`를 주면 `args.auto == True`, 주지 않으면 `False`입니다. `--auto true`처럼 값을 쓰지 않습니다.
- `start`에만 붙였으므로 `status --auto`처럼 다른 명령에 쓰면 인자 오류로 걸러집니다.
- 이 값은 [main.py:192](shorts_graph/cli/main.py#L192)의 `initial_state(args.query, out_dir, auto=args.auto)`로 그래프 상태에 들어갑니다. 켜져 있으면 기사 선택과 대본 승인에서 interrupt로 멈추지 않고 끝까지 진행합니다.

`shorts_graph start --help` 결과입니다. 선택 범위 다음 줄에 있는 `--thread-id`와 `query`도 같이 나옵니다.

```
usage: shorts_graph start [-h] [--auto] [--thread-id THREAD_ID] query

검색어로 새 실행을 시작한다                       ← add_parser의 description

options:
  --auto                사람 확인 없이 끝까지 자동 실행
```