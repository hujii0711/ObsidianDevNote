
## context의 역할

`context`는 **Docker 빌드 시 데몬(daemon)에 전달되는 파일들의 기준 디렉터리**입니다. 즉 "빌드에서 어디까지를 내 파일로 볼 것인가"를 정합니다.

```yaml
build:
  context: .
  dockerfile: Dockerfile
```

## 구체적으로 영향을 주는 것

1. **`COPY` / `ADD`의 기준 경로**: Dockerfile 안의 `COPY . .`에서 앞의 `.`은 Dockerfile 위치가 아니라 **context 디렉터리**를 가리킵니다.
2. **빌드에 전송되는 파일 범위**: context 디렉터리 전체가 빌드 데몬에 전송됩니다. 그래서 `.dockerignore`도 context 루트 기준으로 적용됩니다.
3. **context 밖의 파일은 접근 불가**: `COPY ../other/file .`처럼 context 상위 경로는 참조할 수 없습니다.

## 경로 해석 규칙

- `context`는 **docker-compose.yaml이 있는 위치 기준 상대경로**입니다.
- `dockerfile`은 **context 기준 상대경로**입니다. (compose 파일 기준이 아님)

## 예시: 모노레포 구조

```
project/
├── docker-compose.yaml
├── shared/
└── services/
    └── api/
        ├── Dockerfile
        └── src/
```

```yaml
# 1) context를 서비스 폴더로 지정
build:
  context: ./services/api
  dockerfile: Dockerfile
# → COPY . . 은 services/api 안의 파일만 복사. shared/는 접근 불가

# 2) context를 루트로 지정
build:
  context: .
  dockerfile: services/api/Dockerfile
# → shared/ 도 COPY 가능: COPY shared ./shared
```

`shared/` 같은 공통 코드를 함께 빌드해야 한다면 **context를 상위로 올리고 `dockerfile`에 경로를 지정**하는 방식을 씁니다.

## 주의사항

- **context가 클수록 빌드 시작이 느려집니다.** `node_modules`, `.git`, `dist` 등이 포함되면 전송량이 커지므로 `.dockerignore`로 제외하세요.
- `context`에는 Git 저장소 URL도 지정할 수 있습니다. (예: `https://github.com/user/repo.git#main`)
- 지정하지 않고 `build: .`처럼 축약해서 쓰면 context만 지정한 것이며, Dockerfile은 기본값(`Dockerfile`)을 사용합니다.

현재 프로젝트처럼 Dockerfile이 루트에 있고 단일 서비스라면 `context: .`이 가장 일반적인 설정입니다.