
`uv run`은 둘 다 같습니다. 프로젝트의 `.venv`를 맞춘 뒤 그 환경의 파이썬으로 명령을 실행합니다. 실제로 다른 점은 그 뒤에 무엇을 넘기느냐, 즉 **모듈 이름을 주느냐(`-m`), 파일 경로를 주느냐**입니다.

```bash
uv run -m shorts_graph start      # = uv run python -m shorts_graph start
uv run python shorts_graph/cli/main.py start
```

`uv run -m X`는 `uv run python -m X`를 줄여 쓴 것이고, 결과도 같습니다. 그러니 실제로 비교할 것은 `python -m 모듈`과 `python 파일.py`입니다.

### 차이점

| | `python -m shorts_graph` | `python shorts_graph/cli/main.py` |
|---|---|---|
| 넘기는 것 | 모듈 이름(점으로 구분) | 파일 경로(슬래시로 구분) |
| `sys.path[0]` | **현재 폴더**(저장소 루트) | **그 파일이 있는 폴더**(`shorts_graph/cli/`) |
| 패키지로 인식 | 됨(`__package__`가 정해짐) | 안 됨(그냥 단독 스크립트) |
| 패키지를 넘기면 | 그 안의 `__main__.py`를 실행 | 폴더는 실행할 수 없음 |
| 상대 import(`from .events import ...`) | 동작 | `attempted relative import with no known parent package` 오류 |

### 이 프로젝트에서 `-m`을 쓰는 이유

1. **절대 import**: [cli/main.py:24-28](shorts_graph/cli/main.py#L24-L28)은 `from shorts_graph.cli.events import ...`처럼 패키지 이름부터 import합니다. 저장소 루트에서 `-m`으로 실행하면 루트가 `sys.path`에 들어가니 `shorts_graph`를 찾을 수 있습니다. 파일 경로로 실행하면 `sys.path[0]`이 `shorts_graph/cli/`가 됩니다. 그러면 (프로젝트가 venv에 패키지로 설치되어 있지 않은 한) `ModuleNotFoundError: No module named 'shorts_graph'`가 납니다.
2. **진입점**: [shorts_graph/__main__.py](shorts_graph/__main__.py)는 `python -m shorts_graph`로 실행해야만 시작점이 됩니다. 이 파일이 `main()`을 부르고 그 반환값(0/1/2)을 종료 코드로 넘깁니다. 웹이 잠금 실패를 종료 코드 2로 알아보는 것도 이 경로를 거칩니다.
3. **실행 위치**: `-m`은 현재 폴더를 기준으로 모듈을 찾습니다. 그래서 `uv run -m shorts_graph`, `uv run -m demo.news_tts.news_to_shorts` 같은 명령은 **저장소 루트에서** 실행해야 합니다.

### 정리

- 패키지 안의 코드(`shorts_graph`, `demo.news_tts`처럼 import로 서로 엮인 코드)는 루트에서 `uv run -m 점.으로.구분한.이름`으로 실행합니다.
- import 없이 혼자 도는 파일 하나는 `uv run python foo.py`나 `uv run foo.py`로 실행해도 됩니다. `uv run foo.py`는 파일 맨 위에 적은 PEP 723 의존성 메타데이터도 읽습니다.

---

## uv run foo.py와 uv run python foo.py 차이

PEP 723 메타데이터가 **없는** 평범한 파일이라면 둘은 같습니다. 둘 다 프로젝트 `.venv`의 파이썬으로 `foo.py`를 실행합니다. 차이가 생기는 건 파일 맨 위에 아래와 같은 **인라인 의존성 메타데이터(PEP 723)**가 있을 때입니다.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "rich"]
# ///
import requests
...
```

### 차이점

| | `uv run foo.py` | `uv run python foo.py` |
|---|---|---|
| uv가 보는 것 | `foo.py`를 **스크립트로 보고** 내용을 읽음 | `python`이라는 **명령**을 실행할 뿐, `foo.py`는 파이썬에 넘기는 인자 |
| PEP 723 메타데이터 | 읽음: 적힌 의존성만 담은 **임시 격리 환경**을 만들어 실행. 프로젝트 의존성은 쓰지 않음 | 무시함: 주석으로만 취급 |
| `requires-python` | 메타데이터에 맞는 파이썬 버전을 골라 씀 | 프로젝트 venv의 파이썬 |
| 메타데이터가 없을 때 | 프로젝트 환경에서 실행 | 프로젝트 환경에서 실행(같음) |

### 예시

```bash
uv run foo.py           # requests, rich만 있는 임시 환경. 프로젝트 패키지는 없음
uv run python foo.py    # 프로젝트 .venv에서 실행. requests가 거기 없으면 ImportError
```

### 덧붙여

- `uv add --script foo.py requests`: 의존성을 파일의 메타데이터 블록에 적어 줍니다.
- `uv run --script foo`: 확장자가 `.py`가 아니어도 스크립트로 취급합니다.
- `uv run --with rich foo.py`: 메타데이터 없이 이번 실행에만 패키지를 더합니다.

이 저장소의 코드는 서로 import하는 패키지이고 의존성을 `pyproject.toml`로 관리하므로 메타데이터를 쓸 일이 거의 없습니다. 그래서 여기서는 `uv run foo.py`와 `uv run python foo.py`가 사실상 같습니다. 패키지 코드는 앞서 말한 대로 `uv run -m`으로 실행합니다. PEP 723은 프로젝트와 상관없이 혼자 도는 유틸 스크립트를 의존성까지 담아 파일 하나로 나눠 줄 때 씁니다.