#### 1. uv 설치
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
downloading uv 0.11.7 aarch64-apple-darwin
installing to /Users/fujii0711/.local/bin
  uv
  uvx
everything's installed!

To add $HOME/.local/bin to your PATH, either restart your shell or run:

    source $HOME/.local/bin/env (sh, bash, zsh)
    source $HOME/.local/bin/env.fish (fish)
    
- 터미널 재시작 없이 적용
```zsh
source ~/.zshrc
```

#### 2. 가상 환경 만들기

``` zsh
# 현재 폴더에 .venv 생성 (기본)
uv venv

# 이름 지정
uv venv myenv

# Python 버전 지정
uv venv --python 3.11

# 이름 + 버전 동시 지정
uv venv myenv --python 3.12

# 활성화
source .venv/bin/activate

# 의존성 설치
pip install requests # 또는 uv pip install

# 비활성화
deactivate
```

- 가상환경만 **덩그러니** 만들어줌
- `pyproject.toml` 없음 → 의존성 기록 안 됨
- 매번 `activate` 해야 함
- `requirements.txt`를 직접 관리해야 함
- 기존 `pip` + `venv` 방식과 거의 동일한 워크플로우

#### 3. 프로젝트 통합 워크플로우 생성
uv를 이용하여 프로젝트 전체를 관리하는 더 편한 방법이다.
```zsh
# 새 프로젝트 생성 (pyproject.toml 자동 생성)
uv init my-project
cd my-project

# 패키지 추가 (가상환경 자동 생성 + 설치 + 기록)
uv add requests pandas

# 스크립트 실행/activate 없이 바로 실행
uv run main.py

# 패키지 제거
uv remove requests
```

- `pyproject.toml`이 생겨서 의존성이 **자동으로 기록**됨
- `uv.lock`으로 버전이 **정확히 고정**됨
- `activate` 없이 `uv run`으로 실행 가능
- 다른 사람이 받았을 때 `uv sync` 한 방으로 환경 재현 가능

> 한눈에 비교

|                | uv venv     | uv init     |
| -------------- | ----------- | ----------- |
| 용도             | 가상환경만 필요할 때 | 프로젝트 전체 관리  |
| 의존성 기록         | 수동          | 자동          |
| activate 필요    | 매번          | uv run으로 대체 |
| 협업/배포          | 불편          | 편리          |
| pyproject.toml | X           | O           |

> 언제 뭘 쓰나?

- **빠르게 테스트**하거나 일회성 스크립트 → `uv venv`
- **제대로 된 프로젝트** 시작, 협업, 배포 예정 → `uv init`

---
### uv pip install requests와 uv add requests의 차이

두 명령어는 모두 `uv`를 통해 `requests` 라이브러리를 설치하는 명령어이지만, `uv`를 어떤 방식으로 활용하고 있느냐(프로젝트 관리 모드 vs 단순 pip 대체 모드)에 따라 큰 차이가 있습니다.

가장 핵심적인 차이는 **`pyproject.toml` 파일에 의존성을 자동으로 기록하느냐 아니냐**입니다.

## 🛠️ 한눈에 보는 차이점 요약

|**비교 항목**|**uv add requests (권장)**|**uv pip install requests**|
|---|---|---|
|**개념**|**현대적인 프로젝트/패키지 관리**|기존 `pip` 명령어를 고속으로 대체|
|**의존성 기록**|`pyproject.toml`에 **자동 기록**됨|기록되지 않음 (일회성 설치)|
|**가상환경 자동화**|가상환경이 없으면 **알아서 만들고** 설치함|가상환경이 활성화되어 있어야만 설치됨|
|**권장 용도**|일반적인 파이썬 프로젝트 개발 시|기존 `requirements.txt` 기반의 레거시 작업 시|

## 1. `uv add requests` 🚀 (새로운 표준 권장 방식)

이 명령어는 Cargo(Rust), npm(Node.js), Poetry(Python)와 같은 현대적인 패키지 매니저의 동작 방식을 따릅니다. **프로젝트 단위로 개발할 때 사용하는 것을 강력히 권장**합니다.

- **동작 방식:** 프로젝트 루트에 있는 `pyproject.toml` 파일의 `[project.dependencies]` 항목에 `requests`를 자동으로 추가하고, `uv.lock` 파일을 생성하여 버전을 고정합니다.
    
- **장점:** * 가상환경(`.venv`)이 없으면 `uv`가 알아서 가상환경까지 만든 후 패키지를 설치해 줍니다.
    
    - 나중에 다른 컴퓨터나 동료가 프로젝트를 이어받았을 때, 개별 설치할 필요 없이 `uv sync` 명령어 하나만으로 똑같은 개발 환경을 복제할 수 있습니다.
        

Bash

```
# 사용 예시 (새 프로젝트 시작 시)
uv init my-project
cd my-project
uv add requests      # pyproject.toml에 등록되며 자동 설치
```

## 2. `uv pip install requests` 🧊 (기존 pip 대체 방식)

이 명령어는 기존에 사용하던 전통적인 `pip install`을 `uv`가 가진 압도적인 속도로 똑같이 흉내 내는 **하위 호환성용 명령어**입니다.

- **동작 방식:** 현재 활성화된 가상환경 내부(`site-packages`)에 `requests` 패키지 파일만 빠르게 다운로드하여 설치합니다.
    
- **단점:** 프로젝트 설정 파일(`pyproject.toml`)에 기록이 남지 않기 때문에, 설치한 패키지 목록을 관리하려면 직접 `uv pip freeze > requirements.txt` 같은 명령어를 수동으로 실행해야 합니다.
    
- **주의점:** 가상환경이 활성화되어 있지 않으면 에러가 발생하거나 설치가 거부됩니다.
    

Bash

```
# 사용 예시 (기존 가상환경 방식)
uv venv
source .venv/bin/activate
uv pip install requests   # 가상환경 안에 단순 설치
```

## 💡 요약하자면 무엇을 써야 할까요?

- **새로운 파이썬 프로젝트를 만들거나 관리할 때:** 👉 항상 `uv add requests`를 사용하세요. 프로젝트 관리가 훨씬 깔끔하고 현대적입니다.
    
- **기존에 만들어진 `requirements.txt` 기반의 프로젝트에서 작업하거나, 테스트용으로 패키지만 빠르게 깔고 지우고 싶을 때:** 👉 `uv pip install requests`를 사용하시면 됩니다.
