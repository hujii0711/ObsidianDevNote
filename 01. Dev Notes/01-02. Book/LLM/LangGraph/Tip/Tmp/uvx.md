
`uvx`는 `uv`에 들어 있는 명령으로, **Python CLI 도구를 설치하지 않고 바로 실행**합니다. `uv tool run`의 줄임말이고, Node.js의 `npx`나 `pipx run`과 비슷합니다.

## 동작 방식

```bash
uvx --from huggingface_hub hf download ...
#        └ 설치할 패키지   └ 실행할 명령
```

1. `huggingface_hub` 패키지를 프로젝트와 분리된 **임시 가상 환경**에 설치합니다.
2. 그 환경에서 `hf` 명령을 실행합니다.
3. 설치한 패키지는 uv 캐시(`~/.cache/uv`)에 남습니다. 그래서 다음에 실행할 때는 바로 시작합니다.

**`--from`이 필요한 이유:** 명령 이름(`hf`)과 패키지 이름(`huggingface_hub`)이 다르기 때문입니다. 둘이 같으면 생략합니다. 예를 들어 `uvx ruff check`는 `--from` 없이 됩니다.

## 비슷한 명령과 비교

| 명령 | 설치 위치 | 용도 |
|---|---|---|
| `uvx 도구` | 임시 환경(캐시) | 가끔 쓰는 도구를 한 번 실행할 때 |
| `uv tool install 도구` | `~/.local/bin`에 계속 남음 | 자주 쓰는 도구. 이 프로젝트의 `mflux`가 이 방식입니다 |
| `uv run ...` | 프로젝트의 `.venv` | 프로젝트 코드 실행. 예: `uv run pytest` |

## 이 프로젝트에 미치는 영향

`uvx`는 프로젝트의 `.venv`나 `pyproject.toml`, `uv.lock`을 **건드리지 않습니다.** 그래서 모델 다운로드용으로 `hf`를 실행해도 프로젝트 의존성은 바뀌지 않습니다.

`hf`를 자주 쓸 예정이면 `uv tool install huggingface_hub`로 한 번 설치해 두세요. 그 뒤로는 `hf download ...`만 입력하면 됩니다.