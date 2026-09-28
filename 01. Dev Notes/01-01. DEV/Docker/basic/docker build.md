
## docker build 기본 문법

```bash
docker build [옵션] <context 경로>
```

마지막 인자는 앞서 설명한 **context**입니다. 보통 현재 디렉터리를 뜻하는 `.`을 씁니다.

## 가장 일반적인 사용법

```bash
# 현재 디렉터리의 Dockerfile로 빌드, 이름:태그 지정
docker build -t my-app:1.0 .
```

- `-t` (`--tag`): 이미지 이름과 태그를 지정합니다. 태그를 생략하면 `latest`입니다.
- `.`: context 경로입니다.

## 자주 쓰는 옵션

|옵션|설명|예시|
|---|---|---|
|`-t`|이미지 이름:태그 지정 (여러 번 사용 가능)|`-t my-app:1.0 -t my-app:latest`|
|`-f`|Dockerfile 경로 지정 (기본값: context 내 `Dockerfile`)|`-f docker/Dockerfile.prod`|
|`--build-arg`|빌드 시점 변수 전달 (Dockerfile의 `ARG`)|`--build-arg NODE_ENV=production`|
|`--no-cache`|캐시를 사용하지 않고 처음부터 빌드|`--no-cache`|
|`--target`|멀티스테이지 빌드에서 특정 스테이지까지만 빌드|`--target builder`|
|`--platform`|대상 플랫폼 지정|`--platform linux/amd64`|
|`--progress`|빌드 로그 출력 방식|`--progress=plain`|

## 실무 예시

```bash
# 1) 기본 빌드
docker build -t my-app .

# 2) Dockerfile 위치가 다를 때 (모노레포)
docker build -f services/api/Dockerfile -t my-api .

# 3) 빌드 인자 전달
docker build --build-arg NODE_ENV=production -t my-app:prod .

# 4) 캐시 무시하고 재빌드
docker build --no-cache -t my-app .

# 5) Apple Silicon(M1/M2)에서 서버(amd64)용 이미지 빌드
docker build --platform linux/amd64 -t my-app .
```

## 빌드 후 확인 및 실행

```bash
docker images                          # 이미지 목록 확인
docker run --rm -p 3000:3000 my-app    # 실행 (ENTRYPOINT: node dist/main.js)
```

## 다른 방법과의 관계

- `docker compose build`: compose 파일의 `build` 설정(context, dockerfile 등)을 읽어 내부적으로 동일한 빌드를 수행합니다.
- `docker buildx build`: 최신 Docker에서 `docker build`는 기본적으로 BuildKit(buildx)을 사용하며, 멀티 플랫폼 빌드나 레지스트리 푸시(`--push`)가 필요할 때 `buildx`를 명시적으로 씁니다.

## 정리

대부분의 경우 아래 한 줄이면 충분합니다.

```bash
docker build -t 이미지명:태그 .
```

Dockerfile이 다른 위치에 있으면 `-f`, 환경별로 값을 바꿔야 하면 `--build-arg`를 추가하는 식으로 확장하면 됩니다.


---
---

`.`은 **명령어를 실행하는 현재 디렉터리**이지, "Dockerfile이 있는 위치"라는 의미는 아닙니다. 다만 결과적으로 Dockerfile을 찾는 기준이 되기 때문에 헷갈리기 쉽습니다.

## 동작 방식

1. `.`은 **context 경로**를 뜻합니다. (현재 디렉터리)
2. Docker는 기본적으로 **context 디렉터리 안에서 `Dockerfile`이라는 이름의 파일**을 찾습니다.

그래서 현재 디렉터리에 Dockerfile이 있으면 `docker build -t my-app .`이 동작하는 것입니다. Dockerfile의 위치가 `.`의 정의는 아니고, "context 안에 있는 Dockerfile을 기본으로 사용한다"는 규칙 때문입니다.

## 둘이 다를 수 있는 경우

```
project/
├── src/
├── docker/
│   └── Dockerfile
```

```bash
# project/ 에서 실행
docker build -t my-app -f docker/Dockerfile .
```

- **context**: `.` = `project/` (src/ 등 전체가 COPY 대상)
- **Dockerfile**: `docker/Dockerfile` (`-f`로 별도 지정)

Dockerfile은 `docker/` 안에 있지만 context는 `project/`입니다.

## 다른 디렉터리에서 실행하는 경우

```bash
# 어느 위치에서든 경로를 명시
docker build -t my-app /home/user/project
```

이때 context는 `/home/user/project`이고, Dockerfile도 기본적으로 그 안에서 찾습니다.

## 정리

|항목|기본값|
|---|---|
|`.` (context)|명령어를 실행한 현재 디렉터리|
|Dockerfile 위치|context 안의 `Dockerfile`|

**context와 Dockerfile 위치는 별개 개념**이며, 기본 동작에서는 같은 곳을 가리킬 뿐입니다. 다르게 하고 싶으면 `-f`(Dockerfile 경로)와 마지막 인자(context)를 각각 지정하면 됩니다.