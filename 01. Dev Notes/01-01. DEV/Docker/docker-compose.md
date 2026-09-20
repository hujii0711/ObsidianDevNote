
Docker Compose의 현재 표준과 최신 기능을 확인하겠습니다.Docker Compose는 여러 컨테이너로 이루어진 애플리케이션을 YAML 파일 하나로 정의하고 한 번에 실행·관리하는 도구입니다. 아래는 공식 문서와 최신 자료를 조사해 정리한 학습 가이드입니다.

## 1. 먼저 정리할 배경 지식

**명령어 표기**: 지금은 `docker compose`(공백)가 표준입니다. 예전 자료의 `docker-compose`(하이픈)는 구버전(v1) 표기이므로 공백으로 바꿔 읽으면 대부분 그대로 통합니다.

**파일 이름과 규격**

- 기본 파일명은 `compose.yaml`(권장) 또는 `compose.yml`이고, 이전 버전과의 호환을 위해 `docker-compose.yaml/yml`도 지원됩니다.
- Compose Specification이 현재 권장되는 파일 형식이며, 예전의 2.x, 3.x 형식은 이 규격에 통합되었습니다.
- 최상위 `version` 필드는 더 이상 필요하지 않습니다. 예전 튜토리얼의 `version: '3.8'`은 지워도 됩니다.

**동작 방식**: Compose 파일로 애플리케이션의 서비스를 설정하고, Compose CLI로 그 설정에서 모든 서비스를 한 번에 생성·시작합니다.

## 2. 핵심 개념 3가지

애플리케이션의 컴퓨팅 구성요소는 **서비스**로 정의되고, 서비스끼리는 **네트워크**로 통신하며, 영속 데이터는 **볼륨**에 저장·공유합니다. 여기에 민감 정보용 `secrets`와 설정 파일용 `configs`가 추가로 있습니다.

서비스 정의에서 자주 쓰는 키는 다음과 같습니다.

|키|역할|
|---|---|
|`image` / `build`|기존 이미지를 쓸지, Dockerfile로 빌드할지|
|`command` / `entrypoint`|실행 명령 덮어쓰기|
|`environment` / `env_file`|환경변수|
|`ports`|호스트 포트 공개|
|`volumes`|데이터 마운트|
|`depends_on` / `healthcheck`|시작 순서, 준비 상태 판단|
|`restart`|재시작 정책 (`unless-stopped` 등)|
|`profiles`|선택적 실행 그룹|
|`mem_limit` 등|자원 제한|

### kubernetes 개념 대응표

Compose를 알고 있다면 Kubernetes 개념을 이렇게 매핑해서 이해할 수 있어요.

|Docker Compose|Kubernetes|
|---|---|
|`service`|Deployment (또는 StatefulSet) + Service|
|`ports`|Service (NodePort/LoadBalancer), Ingress|
|`environment`|ConfigMap, Secret|
|`volumes`|PersistentVolumeClaim|
|`depends_on`|직접 대응되는 것 없음 (readiness probe, init container로 처리)|
|`deploy.replicas`|Deployment의 `replicas`, HPA|
|서비스 이름으로 통신|Service 이름으로 통신 (DNS 방식이 비슷함)|

## 3. 필수 CLI 명령

|명령|설명|
|---|---|
|`docker compose up -d`|생성 및 백그라운드 시작 (설정이 바뀐 서비스는 재생성)|
|`docker compose up -d --build`|이미지를 다시 빌드하고 시작|
|`docker compose down`|컨테이너와 네트워크 삭제 (볼륨은 유지)|
|`docker compose down -v`|볼륨까지 삭제 (**데이터 소실 주의**)|
|`docker compose ps`|상태 확인 (healthy 여부 포함)|
|`docker compose logs -f 서비스`|로그 따라가기|
|`docker compose exec 서비스 sh`|실행 중인 컨테이너에 접속|
|`docker compose run --rm 서비스`|일회성 작업 실행 후 삭제|
|`docker compose config`|변수 치환과 파일 병합이 끝난 **최종 설정 출력**|
|`docker compose pull` / `build`|이미지 받기 / 빌드|

디버깅의 출발점은 `docker compose config`입니다. `.env` 값이 어떻게 들어갔는지, 여러 파일이 어떻게 합쳐졌는지 눈으로 확인할 수 있습니다.

## 4. 네트워킹

이 부분이 가장 많이 헷갈리는 곳입니다.

- `up`을 실행하면 `<프로젝트명>_default` 네트워크가 만들어지고 모든 서비스가 붙으며, 각 서비스 이름이 내부 DNS에 등록되어 서로 서비스 이름으로 접속할 수 있습니다. 즉 API 컨테이너에서 `mongo:27017`, `qdrant:6333`으로 접속하면 됩니다.
- 같은 Compose 네트워크 안에서는 모든 컨테이너 포트가 서로 열려 있으므로, `ports:`는 Docker 바깥(호스트)에서 접근할 때만 필요합니다. 컨테이너끼리만 통신한다면 `ports`를 쓰지 않는 편이 안전합니다.
- 호스트 IP를 지정하지 않으면 모든 인터페이스(0.0.0.0)에 바인딩되어 호스트 방화벽 규칙을 우회할 수 있습니다. 그래서 앞서 예시에서 `127.0.0.1:27017:27017`로 썼습니다.
- `HOST:CONTAINER`는 YAML이 60진수 실수로 해석하는 문제를 피하려면 항상 따옴표로 감싸야 합니다.
- 프로젝트 이름은 기본적으로 디렉터리 이름(소문자)이고, `-p` 옵션이나 최상위 `name:`으로 바꿀 수 있습니다. 네트워크와 볼륨 이름 앞에 이 이름이 붙습니다.
- 컨테이너 안의 `localhost`는 그 컨테이너 자신입니다. 다른 컨테이너는 서비스 이름으로, 호스트의 MLX 서버는 `host.docker.internal`로 접근합니다.

## 5. 볼륨

|종류|예|특징|
|---|---|---|
|named volume|`mongo_data:/data/db`|Docker가 관리, DB 데이터에 적합, Mac에서도 빠름|
|bind mount|`./src:/app/src`|호스트 경로 직접 연결, 소스 코드나 설정 파일용|
|tmpfs|`tmpfs: /tmp`|메모리에만 존재|

named volume은 최상위 `volumes:`에 선언해야 하고, 실제 이름은 `<프로젝트명>_<볼륨명>`이 됩니다. 읽기 전용 마운트는 `:ro`를 붙입니다.

## 6. 환경변수와 보간

`${MONGO_USER}` 같은 변수 치환을 **보간(interpolation)**이라고 합니다.

- `--env-file` 없이 실행하면 Compose는 프로젝트 디렉터리의 `.env` 파일을 찾아 보간에 사용합니다.
- 값을 컨테이너 안에 넣는 방법은 `environment`(직접 작성)와 `env_file`(파일로 주입) 두 가지입니다.
- `$$`로 쓰면 Compose가 파싱할 때가 아니라 컨테이너 안의 셸이 변수를 해석하게 됩니다. healthcheck 명령에서 자주 필요합니다.
- `.env` 파일 안의 보간은 Compose CLI의 기능이므로 `docker run --env-file`에서는 동작하지 않습니다.

여러 곳에서 같은 변수를 정의했을 때의 우선순위(높은 것부터)는 다음과 같습니다.

1. `docker compose run -e`로 준 값
2. `environment`나 `env_file`에서 셸 또는 `.env`로 보간된 값
3. Compose 파일의 `environment`에 직접 적은 값
4. `env_file` 속성의 값
5. 이미지의 `ENV` 지시어

`.env`는 `.gitignore`에 넣고, 대신 `.env.example`을 커밋하는 것이 관례입니다.

## 7. 시작 순서: `depends_on`과 `healthcheck`

초보자가 가장 많이 놓치는 부분입니다.

- 짧은 문법(`depends_on: [db]`)은 의존 서비스가 **시작**된 것만 보장하고, 준비 완료(healthy)까지 기다리지는 않습니다. DB 컨테이너가 떠 있어도 DB가 연결을 받을 준비가 안 됐을 수 있습니다.
- 준비 상태까지 기다리려면 `condition`을 쓰며, 값은 `service_started`, `service_healthy`, `service_completed_successfully` 중 하나입니다.
- `service_completed_successfully`는 의존 서비스가 정상 종료(0)할 때까지 기다립니다. DB 마이그레이션이나 초기화 작업 후에 앱을 띄울 때 유용합니다.
- `restart: true`를 두면 의존 서비스가 Compose 조작으로 재시작될 때 이 서비스도 함께 재시작됩니다.

```yaml
services:
  mongo:
    image: mongodb/mongodb-community-server:<태그>
    healthcheck:
      test: ["CMD-SHELL", "mongosh --quiet --eval \"db.adminCommand('ping')\""]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s

  api:
    build: ./api
    depends_on:
      mongo:
        condition: service_healthy
```

`healthcheck`의 `start_period`는 초기 기동 중 실패를 실패 횟수로 세지 않는 유예 시간입니다. 그래도 앱 쪽에는 **재시도 로직**을 두는 것이 안전합니다. 운영 중에 DB가 재시작될 수도 있기 때문입니다.

## 8. profiles: 선택적으로 실행할 서비스

- `--profile frontend up`으로 실행하면 해당 프로필의 서비스와, 프로필이 지정되지 않은 서비스가 함께 시작됩니다.
- 프로필은 여러 개를 켤 수 있고(`--profile a --profile b`), `COMPOSE_PROFILES` 환경변수로도 지정할 수 있습니다.

앞서 만든 인덱싱 파이프라인처럼 "평소에는 안 띄우고 필요할 때만 실행"하는 서비스에 `profiles: ["ingest"]`를 씁니다.

## 9. 여러 파일 조합 (환경별 분리)

|방법|용도|
|---|---|
|`-f a.yaml -f b.yaml` (병합)|dev/prod 등 환경별 덮어쓰기|
|`compose.override.yaml`|이름 규칙에 따라 자동 적용되는 개발용 오버라이드|
|`include:`|파일을 모듈처럼 가져오기 (Compose 2.20+)|
|`extends:`|특정 서비스 설정 상속|

`-f`를 여러 개 주면 뒤의 파일이 앞 파일의 값을 덮어쓰거나 새 값을 추가하며, 여러 파일을 쓸 때 경로는 첫 번째 파일 기준의 상대 경로입니다.

```bash
docker compose -f compose.yaml -f compose.prod.yaml up -d
```

## 10. 개발 워크플로우: build와 watch

이미지를 직접 만드는 서비스는 `build`를 씁니다.

```yaml
api:
  build:
    context: ./api
    dockerfile: Dockerfile
```

`develop.watch`는 Compose 2.22.0 이상에서 쓸 수 있는 선택 기능으로, 로컬 파일 변경에 따라 서비스를 자동 갱신하는 규칙을 정의합니다. 동작(action)은 다음과 같습니다.

- `sync`: 컨테이너는 그대로 두고 변경된 파일만 동기화
- `rebuild`: 이미지를 다시 빌드하고 서비스를 재생성
- `restart`: 컨테이너 재시작 (2.32.0 이상)

`docker compose watch` 또는 `docker compose up --watch`로 시작합니다. 소스는 `sync`, 의존성 파일(`pyproject.toml`, `uv.lock`)은 `rebuild`로 나누는 것이 정석입니다.

## 11. 자주 하는 실수

1. **`depends_on`만 쓰고 준비 상태를 기다리지 않음**: healthcheck와 `service_healthy`를 함께 쓰세요.
2. **DB에 `ports`를 습관처럼 공개**: 컨테이너끼리는 필요 없고, 필요해도 `127.0.0.1`로 묶으세요.
3. **컨테이너 안에서 `localhost`로 다른 서비스에 접속**: 서비스 이름을 쓰세요.
4. **`down -v`로 데이터 삭제**: 볼륨은 `down`만으로는 지워지지 않습니다.
5. **이미지 태그를 `latest`로 둠**: 재현성이 깨지므로 버전을 고정하세요.
6. **`.env`를 커밋**: 비밀번호가 유출됩니다.
7. **`container_name` 남발**: 이름이 고정되어 다른 프로젝트와 충돌하고 `--scale`도 못 씁니다. 꼭 필요하지 않다면 생략하세요.
8. **Dockerfile을 고쳤는데 반영이 안 됨**: `up -d --build`를 쓰거나 `build`를 먼저 실행하세요.
9. **YAML 들여쓰기 오류**: 탭이 아닌 공백을 쓰고, 이상하면 `docker compose config`로 검증하세요.
10. **Mac에서 bind mount로 DB 데이터 저장**: 느리고 불안정하므로 named volume을 쓰세요.

## 12. 추천 학습 순서 (우리 프로젝트로 실습)

|단계|실습|익히는 것|
|---|---|---|
|1|MongoDB 한 개 띄우기 → `down` 후 다시 `up` 해서 데이터가 남는지 확인|services, volumes, `.env`|
|2|Qdrant 추가, 다른 컨테이너에서 서비스 이름으로 접속 테스트|기본 네트워크, DNS|
|3|API 컨테이너를 `build`로 추가하고 healthcheck + `depends_on` 설정|시작 순서|
|4|인덱싱 파이프라인을 `profiles`로 분리, `docker compose --profile ingest run --rm` 실행|profiles, run|
|5|`compose.override.yaml`에 개발용 설정 분리, `watch` 적용|다중 파일, watch|
|6|`docker compose config`로 최종 설정 확인|디버깅|

각 단계마다 일부러 실패를 만들어 보면 이해가 빨라집니다. 예를 들어 healthcheck 없이 `depends_on`만 쓴 상태에서 API가 DB 연결에 실패하는 모습을 관찰해 보세요.

## 13. 참고 문서

- Compose 파일 레퍼런스: https://docs.docker.com/reference/compose-file/
- 서비스 속성 (ports, depends_on 등): https://docs.docker.com/reference/compose-file/services/
- 시작 순서: https://docs.docker.com/compose/how-tos/startup-order.md
- 환경변수 우선순위: https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/
- 네트워킹: https://docs.docker.com/compose/how-tos/networking.md
- CLI 레퍼런스: https://docs.docker.com/reference/cli/docker/compose/

이 외에 Compose Bridge(Compose 파일을 Kubernetes 매니페스트로 변환)나 OCI 레지스트리/Git에서 Compose 파일을 직접 가져오는 기능도 있지만, 기본기를 익힌 뒤에 봐도 충분합니다. 원하시면 앞서 논의한 MongoDB, Qdrant, API, 인덱서를 하나의 `compose.yaml`로 합친 완성본을 위 개념에 맞춰 만들어 드리겠습니다.

### 14. 작성 예시

```yaml
services:
  chroma:
    image: chromadb/chroma:<버전 고정>
    volumes:
      - chroma_data:/data        # 이미지 버전에 따라 경로가 다를 수 있음
    ports:
      - "127.0.0.1:8000:8000"    # 호스트 인덱싱 스크립트용, 외부 노출 금지
    mem_limit: 2g

  api:
    build: ./api
    environment:
      CHROMA_HOST: chroma
      LLM_BASE_URL: http://host.docker.internal:8080/v1
      EMBED_BASE_URL: http://host.docker.internal:8081/v1
    depends_on: [chroma]

volumes:
  chroma_data:
```

### 15. dockerfile로 이미지 빌드
