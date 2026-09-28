
### `python -m pythontest.xml_to_jsonl` — "모듈로 실행한다"의 의미

### `-m`의 의미: "파일 경로"가 아니라 "임포트 이름"으로 실행

두 방식의 차이는 **파이썬이 실행할 코드를 어떻게 찾는가**입니다.

| 명령                                      | 찾는 방법                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `python src/pythontest/xml_to_jsonl.py` | **파일시스템 경로**를 직접 열어서 실행                                                 |
| `python -m pythontest.xml_to_jsonl`     | `import pythontest.xml_to_jsonl` 과 **똑같은 규칙**으로 `sys.path`를 뒤져서 찾은 뒤 실행 |

즉 `-m` 은 "`sys.path` 에서 `pythontest` 패키지를 찾고, 그 안의 `xml_to_jsonl` 모듈을 임포트해서, 그것을 최상위 프로그램으로 돌려라"는 뜻입니다.

점(`.`)은 디렉터리 구분자가 아니라 **패키지 경로 구분자**입니다. 그래서 `.py` 확장자를 붙이지 않으며, `python -m pythontest/xml_to_jsonl.py` 같은 표기는 동작하지 않습니다.

## 실행 과정 (`-m` 기준)

1. `sys.path` 에서 `pythontest` 패키지를 검색 → `src/pythontest/` 를 찾음
2. `pythontest/__init__.py` 를 먼저 **임포트해서 실행**
3. 그 안의 `xml_to_jsonl` 모듈을 로드하되, `__name__` 을 `"__main__"` 으로 설정
4. 따라서 `xml_to_jsonl.py` 의 `if __name__ == "__main__":` 블록이 참이 되어 `main()` 호출

**핵심은 3번**입니다. 모듈로 임포트되면서도 이름표만 `__main__` 으로 바뀝니다. 그래서 `if __name__ == "__main__"` 관용구가 두 실행 방식 모두에서 동작합니다.

`src/pythontest/xml_to_jsonl.py` 의 마지막 부분:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

## 실질적 차이 3가지

### 1. `sys.path[0]` 이 다릅니다 — 가장 중요한 차이

| 실행 방식 | `sys.path[0]` |
| --- | --- |
| 경로 실행 (`python 파일.py`) | **스크립트가 있는 디렉터리** (`src/pythontest/`) |
| 모듈 실행 (`python -m ...`) | **현재 작업 디렉터리** |

이 프로젝트는 src 레이아웃이라 이 차이가 결정적입니다.

`python src/pythontest/xml_to_jsonl.py` 로 실행하면 `sys.path[0]` 이 `src/pythontest/` 가 되어 **`pythontest` 패키지 자체를 찾지 못합니다.** 그 안에서 `from pythontest.something import ...` 같은 임포트가 있으면 `ModuleNotFoundError` 가 발생합니다.

### 2. `__init__.py` 가 실행됩니다

`-m` 은 패키지를 거쳐 가므로 `src/pythontest/__init__.py` 가 반드시 먼저 실행됩니다. 경로 실행은 이 단계를 건너뜁니다. 패키지 초기화 로직(로깅 설정, 버전 상수 등)이 있다면 결과가 달라집니다.

### 3. 상대 임포트가 가능합니다

`from . import foo` 같은 상대 임포트는 모듈이 패키지의 일부로 인식돼야 동작합니다. 경로 실행 시에는 패키지 소속이 아니므로 다음 에러가 납니다.

```
ImportError: attempted relative import with no known parent package
```

## 이 프로젝트에서 가능한 4가지 실행 방법

```powershell
uv run xml2jsonl                         # ① 콘솔 스크립트 ([project.scripts])
uv run python -m pythontest.xml_to_jsonl # ② 모듈 실행
uv run python -m pythontest              # ③ 패키지 실행 → __main__.py 실행
uv run pythontest                        # ④ 패키지 기본 진입점
```

`pyproject.toml` 의 진입점 정의:

```toml
[project.scripts]
pythontest = "pythontest:main"
xml2jsonl  = "pythontest.xml_to_jsonl:main"
jsonl2json = "pythontest.jsonl_to_json:main"
```

이 설정 덕분에 `.venv/Scripts/` 에 `xml2jsonl.exe`, `jsonl2json.exe`, `pythontest.exe` 래퍼가 생성됩니다.

### ②와 ③의 차이

같은 원리입니다. `-m` 에 **패키지 이름만** 주면 파이썬은 그 패키지 안의 `__main__.py` 를 찾아 실행합니다.

```python
# src/pythontest/__main__.py
from pythontest import main

if __name__ == "__main__":
    main()
```

`python -m http.server` 나 `python -m pytest` 가 동작하는 것도 정확히 이 규칙 덕분입니다.

## 어느 것을 쓸 것인가

- **일상 작업**: ① 콘솔 스크립트가 가장 짧고 편합니다.
- **디버깅 / CI**: ② `-m` 방식이 낫습니다. `.exe` 래퍼를 거치지 않고 **지금 활성화된 인터프리터가 확실히 실행**되므로, 어떤 파이썬이 도는지 모호하지 않습니다.
- **절대 피할 것**: `python src/pythontest/xml_to_jsonl.py` — src 레이아웃에서는 임포트가 깨집니다.

## 관련 문서

- [.venv 활성화란 정확히 무엇인가](venv-activation.md)
