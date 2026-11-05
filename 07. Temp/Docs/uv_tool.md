`uv tool`은 **Python으로 만든 CLI 프로그램을 프로젝트와 상관없이 내 PC에 설치하는 uv 기능**입니다. 예전 `pipx`와 같은 역할입니다.

## 동작 방식
`uv tool install mflux`를 실행하면 다음과 같이 됩니다.
1. **전용 가상환경 생성:** mflux만을 위한 가상환경을 따로 만들고, 거기에 mflux와 의존성을 설치합니다. 위치는 `~/.local/share/uv/tools/mflux/`입니다.
2. **명령어 등록:** `mflux-generate` 같은 실행 파일을 `~/.local/bin`에 연결해서, 어느 폴더에서든 명령어로 쓸 수 있게 합니다.

그래서 **프로젝트의 `.venv`나 `pyproject.toml`은 전혀 바뀌지 않습니다**.

## `uv add`와 비교
| | `uv add mflux` | `uv tool install mflux` |
|---|---|---|
| 용도 | 프로젝트 코드에서 `import`해서 쓰는 **라이브러리** | 터미널에서 실행하는 **명령어 도구** |
| 설치 위치 | 프로젝트의 `.venv` | 도구 전용 가상환경 |
| `pyproject.toml` | 의존성으로 기록됨 | 바뀌지 않음 |
| 사용 범위 | 이 프로젝트 안에서만 | PC 어디서나 |
| 의존성 충돌 | 프로젝트 패키지들과 버전이 충돌할 수 있음 | 따로 설치되므로 충돌 없음 |

## 자주 쓰는 명령
```bash
uv tool install mflux      # 설치
uv tool list               # 설치된 도구 목록 (현재: 없음)
uv tool upgrade mflux      # 업데이트
uv tool uninstall mflux    # 삭제
uvx --from mflux mflux-generate ...   # 설치 없이 일회성 실행 (uv tool run과 같음)
```

설치한 뒤 `mflux-generate: command not found`가 나오면 `~/.local/bin`이 PATH에 없는 것입니다. 이때는 `uv tool update-shell`을 실행하고 터미널을 다시 여세요.

## 이 프로젝트에서 이렇게 정한 이유
계획 문서에서 mflux를 `uv tool`로 설치하도록 한 이유는 두 가지입니다.
- **의존성 분리:** mflux는 MLX, torch 계열 등 의존성이 무겁습니다. 이것들을 LangGraph 프로젝트의 의존성과 섞지 않으려는 것입니다.
- **호출 방식에 맞음:** 코드에서는 앞에서 말씀드린 1번 방식(`subprocess`로 CLI 호출)으로만 부르므로, 프로젝트에 라이브러리로 설치할 필요가 없습니다.

나중에 2번 방식(Python API를 직접 `import`)으로 바꾸려면 그때는 `uv add`로 프로젝트에 넣어야 합니다.