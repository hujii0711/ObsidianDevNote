## `.venv` 활성화란 정확히 무엇인가

## 한 줄 요약

"`.venv` 활성화"란 **현재 셸의 환경변수(주로 `PATH`)를 바꿔서, `python` / `pip` 같은 명령이 시스템 파이썬이 아니라 `C:\PythonTest\.venv\` 안의 파이썬을 가리키게 만든 상태**를 말합니다. 그 이상도 이하도 아닙니다.

## 실제로 일어나는 일

`.venv\Scripts\Activate.ps1`을 실행하면 셸 프로세스에서 딱 세 가지가 바뀝니다.

| 변경 | 내용 |
| --- | --- |
| `PATH` 맨 앞에 `C:\PythonTest\.venv\Scripts` 추가 | `python`, `pip`, `pytest` 입력 시 이 폴더가 **먼저** 검색됨 |
| `VIRTUAL_ENV=C:\PythonTest\.venv` 설정 | 도구들이 "지금 가상환경 안"임을 인식하는 표식 |
| 프롬프트 앞에 `(pythontest)` 표시 | 사람 눈으로 확인하기 위한 장식일 뿐, 기능은 없음 |

프롬프트 문자열은 `.venv\pyvenv.cfg`의 `prompt = pythontest` 값에서 옵니다. 즉 이 프로젝트에서는 `(.venv)`가 아니라 `(pythontest)`로 표시됩니다.

원래 `PATH` 값은 `_OLD_VIRTUAL_PATH` 변수에 백업되고, `deactivate` 를 실행하면 그대로 복원되면서 `VIRTUAL_ENV` / `VIRTUAL_ENV_PROMPT` 는 제거됩니다.

## 왜 이것만으로 충분한가

`.venv\Scripts\python.exe` 가 실행되면, 그 exe는 자기 위치를 기준으로 `.venv\pyvenv.cfg` 를 찾아 읽고 **`sys.prefix` 를 `.venv` 로 설정**합니다. 그 결과 import 경로(`sys.path`)가 `.venv\Lib\site-packages` 를 향하게 되고, `pip install` 도 그곳에 설치됩니다.

즉 **격리를 수행하는 주체는 exe 자신**이고, 활성화는 "그 exe를 기본으로 부르게 해주는 편의 장치"에 불과합니다.

이 프로젝트의 `pyvenv.cfg` 내용:

```ini
home = C:\Users\hujii\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none
implementation = CPython
uv = 0.12.3
version_info = 3.13
include-system-site-packages = false
prompt = pythontest
```

`include-system-site-packages = false` 이므로 시스템 파이썬에 설치된 패키지는 전혀 보이지 않습니다. 완전 격리 상태입니다.

## 중요한 오해 3가지

1. **셸 세션 단위입니다.** 새 터미널 탭을 열면 다시 활성화해야 합니다. 프로젝트에 영구 설정되는 것이 아닙니다.
2. **디렉터리 이동과 무관합니다.** 활성화한 채로 `cd C:\OtherProject` 해도 여전히 `C:\PythonTest\.venv` 의 파이썬을 씁니다. 실수의 단골 원인입니다.
3. **필수가 아닙니다.** 활성화 없이도 `.venv\Scripts\python.exe -m pytest` 처럼 전체 경로로 직접 호출하면 완전히 동일하게 동작합니다.

## 이 프로젝트에서의 권장 사용법

`uv.lock` 과 `.python-version` 이 있는 uv 기반 프로젝트이므로, **활성화 자체가 거의 필요 없습니다.**

```powershell
uv run pytest        # uv가 .venv를 알아서 찾아 실행
uv sync              # 잠금파일대로 .venv 동기화
```

`uv run` 은 내부적으로 `.venv` 를 탐색해 그 안의 인터프리터로 실행하므로, 활성화 상태든 아니든 결과가 같습니다.

그래도 활성화하려면:

```powershell
.\.venv\Scripts\Activate.ps1     # PowerShell — 활성화
deactivate                        # 해제
```

PowerShell에서 실행 정책 오류가 발생하면 다음을 먼저 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

## 셸별 활성화 스크립트

`.venv\Scripts\` 에는 셸별 스크립트가 함께 들어 있습니다.

| 셸 | 스크립트 |
| --- | --- |
| PowerShell | `Activate.ps1` |
| cmd.exe | `activate.bat` |
| Git Bash / WSL | `activate` |
| fish | `activate.fish` |
| csh | `activate.csh` |
| nushell | `activate.nu` |
| xonsh | `activate.xsh` |

## 활성화 여부 확인 방법

```powershell
$env:VIRTUAL_ENV                 # 활성화되어 있으면 .venv 경로 출력
(Get-Command python).Source      # 실제로 어떤 python.exe가 잡히는지 확인
python -c "import sys; print(sys.prefix)"
```

`(Get-Command python).Source` 가 `C:\PythonTest\.venv\Scripts\python.exe` 를 가리키면 의도대로 활성화된 것입니다.

---

가상 환경(Virtual Environment)을 생성하고 활성화하지 않은 채 전역(Global) Python 환경에 패키지를 설치하고 프로젝트를 진행할 경우, 다음과 같은 실무적이고 기술적인 문제점들이 발생할 수 있습니다.

## 1. 패키지 버전 충돌 (가장 흔한 문제)

- **상황:** 프로젝트 A는 `requests 2.28.0` 버전을 필요로 하고, 나중에 진행할 프로젝트 B는 최신 기능 때문에 `requests 2.31.0` 이상을 요구하는 경우입니다.

- **문제점:** 전역 환경에는 하나의 패키지 버전만 설치할 수 있으므로, 프로젝트 B를 위해 버전을 업그레이드하는 순간 프로젝트 A에서 예기치 않은 오류(코드 호환성 문제)가 발생할 수 있습니다.

## 2. 전역 시스템 Python 환경 오염

- **상황:** 운영체제나 기본 개발 환경에 설치된 전역 Python 경로에 불필요한 서드파티 패키지들이 계속 쌓이게 됩니다.

- **문제점:** 시스템 전역 환경이 지저분해지며, 어떤 패키지가 어떤 프로젝트에 필요한 것인지 파악하기 어려워집니다. 심한 경우, 운영체제 내부 스크립트가 의존하는 기본 라이브러리와 충돌하여 시스템 전체의 Python 도구들이 오작동할 수 있습니다.

## 3. 프로젝트 의존성 관리 및 배포의 어려움 (`requirements.txt` 관리 불가)

- **상황:** 혼자 또는 팀 협업으로 프로젝트를 다른 PC나 서버(예: AWS, Docker 등)로 옮겨야 할 때입니다.

- **문제점:** 가상 환경을 쓰지 않으면 전역에 설치된 수십 개의 불필요한 라이브러리까지 함께 섞여 나오므로, 현재 프로젝트에 **실제로 필요한 패키지만 골라내어 `requirements.txt`로 추출하기가 매우 어려워집니다.** 이로 인해 배포 서버에서 "ModuleNotFoundError"가 발생할 확률이 급격히 높아집니다.

## 4. 권한(Permission) 문제

- **상황:** macOS, Linux 또는 관리자 권한이 제한된 회사 PC에서 작업할 때 발생합니다.

- **문제점:** 전역 Python 디렉토리(`site-packages`)에 접근하려면 관리자 권한(`sudo` 등)이 필요한 경우가 많습니다. 가상 환경을 사용하면 사용자 폴더 내부에 환경을 만들기 때문에 **root 권한 없이도 자유롭게 패키지를 설치하고 삭제**할 수 있습니다.

> **💡 요약:** 가상 환경은 **"프로젝트마다 독립된 방을 만들어주는 도구"**입니다. 이를 생략하면 당장은 편해 보이지만, 프로젝트가 늘어날수록 관리 지옥에 빠지기 쉽기 때문에 반드시 가상 환경을 습관화하는 것이 좋습니다.