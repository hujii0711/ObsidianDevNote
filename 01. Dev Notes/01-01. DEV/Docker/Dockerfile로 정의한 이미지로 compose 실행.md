
Compose에서 서비스의 `image:` 대신 **`build:`**를 쓰면 됩니다. 이미지를 내려받는 대신, 지정한 폴더의 Dockerfile로 직접 만들어서 실행합니다.

## 1. 기본 형태

```yaml
services:
  api:
    build: ./api        # ./api 폴더의 Dockerfile로 빌드
```

짧은 문법은 **빌드 컨텍스트 경로**만 지정한 것이고, 그 폴더 안의 `Dockerfile`을 기본으로 찾습니다. 옵션이 필요하면 긴 문법을 씁니다.

```yaml
services:
  api:
    build:
      context: .                           # 빌드에 보낼 파일들의 기준 폴더
      dockerfile: services/api/Dockerfile  # context 기준 상대 경로
      args:                                # 빌드 시점 변수 (Dockerfile의 ARG)
        PYTHON_VERSION: "3.12"
      target: runtime                      # 멀티스테이지 중 특정 단계까지만 빌드
    image: rag-api:dev                     # (선택) 빌드 결과에 붙일 이름
```

|키|의미|
|---|---|
|`context`|Docker에 전달되는 파일 범위입니다. Dockerfile 안의 `COPY`는 **이 범위 안의 파일만** 가져올 수 있습니다.|
|`dockerfile`|Dockerfile 위치입니다. 이름이 다르거나 다른 폴더에 있을 때 지정합니다.|
|`args`|Dockerfile의 `ARG`에 넘길 값입니다.|
|`target`|멀티스테이지 Dockerfile에서 어느 단계까지 빌드할지 정합니다.|
|`image`|함께 쓰면 빌드 결과에 이 이름과 태그가 붙습니다. 없으면 `<프로젝트명>-<서비스명>` 같은 이름이 자동으로 붙습니다.|

## 2. 모노레포에서는 context 위치가 핵심

앞서 보여주신 Dockerfile은 `packages/rag`와 `services/rag-api`를 함께 `COPY`했습니다. 이런 구조는 컨텍스트가 **저장소 루트**여야 합니다.

```yaml
services:
  rag-api:
    build:
      context: .                              # 저장소 루트 (compose.yaml이 루트에 있을 때)
      dockerfile: services/rag-api/Dockerfile
```

컨텍스트가 `./services/rag-api`이면 `COPY packages/rag ...`는 컨텍스트 밖이라 **실패**합니다. `COPY ../x`도 안 됩니다. "file not found" 오류의 가장 흔한 원인입니다.

## 3. 빌드는 언제 일어나는가

|상황|동작|
|---|---|
|`docker compose up -d`|이미지가 없으면 빌드하고, **이미 있으면 재빌드하지 않습니다**|
|`docker compose up -d --build`|항상 다시 빌드한 뒤 시작|
|`docker compose build`|빌드만 수행 (특정 서비스만: `build api`)|
|`docker compose build --no-cache`|캐시 없이 처음부터 빌드|
|`docker compose build --pull`|베이스 이미지도 최신으로 다시 받아 빌드|

코드나 Dockerfile을 고쳤는데 반영이 안 된다면 대부분 `--build`를 빠뜨린 경우입니다. 항상 다시 빌드되게 하려면 서비스에 `pull_policy: build`를 지정할 수 있습니다.

## 4. Dockerfile 작성 시 알아둘 점

Python 예시입니다.

```dockerfile
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base
WORKDIR /app

# 1) 의존성 파일만 먼저 복사 → 코드가 바뀌어도 이 레이어는 캐시 재사용
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# 2) 소스는 나중에 복사
COPY src ./src

RUN useradd -m app
USER app
CMD ["python", "-m", "src.main"]
```

- **레이어 캐시**: 위에서 아래로 실행되며, 앞 레이어가 바뀌면 그 뒤는 전부 다시 실행됩니다. 자주 바뀌는 것(소스)을 아래에, 잘 안 바뀌는 것(의존성)을 위에 두세요.
- **`.dockerignore`**: 컨텍스트 폴더에 두고 `.git`, `.venv`, `__pycache__`, `node_modules`, `.env`, 데이터 폴더를 제외하세요. 빌드가 빨라지고 비밀 파일이 이미지에 들어가는 사고를 막습니다.
- **`ARG`와 `ENV`**: `ARG`는 빌드할 때만, `ENV`는 실행 컨테이너에도 남습니다. Compose의 `args:`는 `ARG`에 대응합니다.
- **멀티스테이지**: 빌드 도구는 앞 단계에서 쓰고 결과물만 최종 이미지로 복사하면 이미지가 작아집니다. `target: dev`, `target: prod`처럼 환경별로 다른 단계를 고를 수도 있습니다.

## 5. 비밀 값은 args에 넣지 마세요

`args`로 넘긴 값은 이미지 기록에 남을 수 있습니다. 토큰 같은 비밀은 **빌드 시크릿**으로 전달합니다.

```yaml
services:
  api:
    build:
      context: .
      secrets:
        - pip_token

secrets:
  pip_token:
    file: ./pip_token.txt
```

```dockerfile
RUN --mount=type=secret,id=pip_token \
    PIP_TOKEN=$(cat /run/secrets/pip_token) pip install ...
```

시크릿은 해당 `RUN` 동안만 마운트되고 이미지에는 남지 않습니다.

## 6. Apple Silicon(Mac)에서의 주의

- 기본은 호스트와 같은 **arm64**로 빌드되므로 그대로 두는 것이 가장 빠릅니다.
- `platform: linux/amd64`를 강제하면 에뮬레이션으로 느려지고, MongoDB처럼 AVX가 필요한 프로그램은 앞서 본 것처럼 실행 자체가 실패할 수 있습니다.
- 베이스 이미지와 pip 휠이 arm64를 지원하는지 확인하세요.

## 7. 개발 중 자동 재빌드

`develop.watch`의 `rebuild` 동작을 쓰면 특정 파일이 바뀔 때 이미지를 다시 빌드하고 서비스를 재생성합니다.

```yaml
services:
  api:
    build: ./api
    develop:
      watch:
        - path: ./api/src
          action: sync            # 소스는 컨테이너에 즉시 동기화
          target: /app/src
        - path: ./api/pyproject.toml
          action: rebuild         # 의존성이 바뀌면 재빌드
```

`docker compose up --watch`로 실행합니다.

## 8. 문제 해결

|증상|확인할 것|
|---|---|
|`COPY failed: file not found`|`context` 범위 밖의 파일을 복사하려는 것은 아닌지, `.dockerignore`가 제외하고 있지는 않은지|
|코드를 고쳤는데 반영 안 됨|`up -d --build`를 했는지|
|빌드가 매번 느림|의존성 `COPY`와 소스 `COPY` 순서, `.dockerignore` 유무|
|이미지가 너무 큼|멀티스테이지, `slim` 베이스, 불필요한 파일 제외|
|빌드 로그가 축약돼 원인을 못 찾겠음|`docker compose build --progress=plain`으로 전체 출력 확인|
|최종 설정 확인|`docker compose config`로 `context`와 `dockerfile` 경로가 의도대로인지 확인|

정리하면, **`image:` 대신 `build:`를 쓰고, `context`(파일 범위)와 `dockerfile`(위치)을 정확히 잡고, 코드가 바뀌면 `--build`로 다시 빌드**하면 됩니다. 저장소 구조(폴더 배치)를 알려주시면 `context`와 `dockerfile` 경로를 어떻게 잡을지 구체적으로 맞춰 드리겠습니다.