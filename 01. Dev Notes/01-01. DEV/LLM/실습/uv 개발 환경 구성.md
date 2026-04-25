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