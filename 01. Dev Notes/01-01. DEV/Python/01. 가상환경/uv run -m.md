실제로 두 방식을 테스트해서 차이를 보여드릴게요. `src/httpTest.py`라는 파일이 있다고 가정하고 실험해보겠습니다.

1. `python -m src.httpTest` 시도 (`__init__.py` 없는 상태)의외로 성공했네요. 최신 파이썬은 `__init__.py`가 없어도 **namespace package**로 인식합니다. 하지만 조건이 있어요 — **현재 작업 디렉토리(cwd)에서 실행해야** `src`를 찾을 수 있습니다.예상대로 실패했습니다. `-m`은 `sys.path` (기본적으로 현재 디렉토리 포함)에서 모듈을 **이름으로 찾기** 때문에, 위치가 어긋나면 못 찾아요.

2. `python src/main.py` (파일 경로 직접 실행)**파일 경로 방식은 어느 디렉토리에서든 경로만 정확하면 항상 작동**해요.

3. import 관계에서의 결정적 차이

이제 진짜 중요한 차이를 보여드릴게요 — 패키지 내부 상대/절대 import가 있을 때입니다.`-m` 방식은 성공적으로 `from src.helper import greet`를 처리했어요. `src`가 패키지 루트로 인식되기 때문입니다.**여기가 핵심입니다.** 똑같은 코드인데, 파일 경로로 직접 실행하면 `ModuleNotFoundError`가 발생해요. 이유는:

- `python src/httpTest.py`로 실행하면, 파이썬은 `sys.path[0]`을 `src/` 폴더 자체로 설정합니다 (파일이 있는 디렉토리)
- 그래서 `src`라는 이름의 패키지를 그 안에서 찾을 수 없음 (`src` 폴더 자기 자신을 상위 경로에서 찾으려 하니 실패)
- 반면 `-m src.httpTest`는 실행 위치(`/home/claude/mod_test`)를 `sys.path`에 넣고, 거기서부터 `src.httpTest`라는 **모듈 경로**를 찾기 때문에 `src` 패키지가 정상적으로 인식됨

## 핵심 차이 정리

|항목|`uv run python -m src.httpTest`|`uv run src/main.py` (또는 `uv run python src/main.py`)|
|---|---|---|
|실행 방식|**모듈 이름**으로 찾아서 실행|**파일 경로**를 직접 실행|
|실행 위치|프로젝트 루트(현재 디렉토리)에서 실행해야 함|경로만 맞으면 어디서든 실행 가능|
|`sys.path[0]`|현재 작업 디렉토리|실행되는 파일이 있는 폴더 (`src/`)|
|`from src.xxx import ...` (절대 import)|✅ 정상 동작|❌ `ModuleNotFoundError`|
|`.` 앞에 점 표기 필요|`src.httpTest`처럼 점(`.`)으로 구분|그냥 파일 경로 그대로|
|언제 쓰나|패키지 구조를 갖춘 프로젝트, `src.모듈` 형태 import가 있는 코드|독립적인 단일 스크립트, 패키지 import 관계가 없는 파일|

## 결론

- `uv run python -m src.httpTest`는 **"src를 패키지로 취급해서, 그 안의 httpTest 모듈을 실행해라"**는 뜻이라, 프로젝트 내부의 다른 모듈(`src.helper` 등)을 절대 경로로 import하는 코드가 있을 때 안전하게 작동합니다.
- `uv run src/main.py`는 그냥 **그 파일 하나를 스크립트로 돌리는 것**뿐이라, 그 파일이 같은 프로젝트의 다른 모듈을 `src.xxx` 형태로 import하려 하면 실패할 수 있습니다.

앞서 만든 `http_project`처럼 `[project.scripts]`로 콘솔 명령어를 등록해두면 이런 실행 방식 차이를 신경 쓸 필요 없이 `uv run check-pkg`로 항상 안정적으로 실행할 수 있다는 게, 실무에서 콘솔 스크립트를 등록하는 이유이기도 합니다.