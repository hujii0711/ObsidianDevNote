# GitHub Actions로 자동화할 수 있는 작업들

## 🔨 CI/CD (빌드 & 배포)

**빌드 자동화**
- 코드 푸시 시 자동 빌드 (Node.js, Java, Python, Go 등)
- Docker 이미지 빌드 및 레지스트리 푸시 (DockerHub, ECR, GCR)
- 멀티 플랫폼 빌드 (linux/amd64, linux/arm64)

**배포 자동화**
- AWS (EC2, ECS, Lambda, S3, Elastic Beanstalk)
- GCP (Cloud Run, GKE, App Engine)
- Azure (App Service, AKS)
- Vercel, Netlify, Heroku 등 PaaS
- Kubernetes 클러스터 배포 (kubectl, Helm)
- SSH를 통한 서버 직접 배포

---

## 🧪 테스트 자동화

- 유닛 테스트 / 통합 테스트 / E2E 테스트 실행
- 테스트 커버리지 측정 및 리포트 생성 (Codecov, Coveralls)
- 매트릭스 빌드로 다중 환경 테스트 (Node 18/20/22, Python 3.10/3.11/3.12)
- 브라우저 크로스 테스트 (Playwright, Cypress)
- 성능 테스트 / 부하 테스트 (k6, Artillery)

---

## 🔍 코드 품질 & 보안

**코드 품질**
- Lint 자동 실행 (ESLint, Pylint, RuboCop)
- 코드 포맷 검사 (Prettier, Black, gofmt)
- 정적 분석 (SonarQube, CodeClimate)
- 타입 체크 (TypeScript, mypy)

**보안 스캔**
- 의존성 취약점 스캔 (Snyk, Dependabot, npm audit)
- SAST 정적 보안 분석 (CodeQL, Semgrep)
- 컨테이너 이미지 보안 스캔 (Trivy, Grype)
- 시크릿 노출 감지 (GitLeaks, TruffleHog)
- DAST 동적 보안 테스트 (OWASP ZAP)

---

## 📦 패키지 & 릴리즈 관리

- npm / PyPI / Maven / NuGet 패키지 자동 배포
- 시맨틱 버저닝 자동화 (semantic-release)
- GitHub Release 자동 생성
- CHANGELOG 자동 생성
- Git 태그 자동 생성

---

## 🔄 PR & 이슈 관리

- PR 생성 시 자동 라벨링
- 코드 리뷰어 자동 지정
- PR 사이즈 체크 (너무 큰 PR 경고)
- 이슈 자동 할당 및 라벨링
- 오래된 이슈/PR 자동 스테일 처리
- PR 머지 시 연관 이슈 자동 닫기
- 브랜치 보호 규칙 강제

---

## 📖 문서화

- API 문서 자동 생성 (Swagger, JSDoc, Sphinx)
- GitHub Pages 자동 배포
- README 배지 자동 업데이트
- 문서 사이트 빌드 (Docusaurus, MkDocs, GitBook)

---

## 🔔 알림 & 모니터링

- Slack / Teams / Discord 빌드 결과 알림
- 이메일 알림
- 배포 성공/실패 Webhook 전송
- PagerDuty 인시던트 자동 생성

---

## 🛠️ 인프라 & 데이터

- Terraform / Pulumi IaC 자동 apply
- 데이터베이스 마이그레이션 자동 실행
- 환경별 설정 파일 동기화
- 캐시 자동 무효화 (CDN Purge)
- 크론 잡 스케줄링 (정기 백업, 리포트 생성)

---

## 🤖 개발 생산성

- 의존성 자동 업데이트 PR 생성 (Dependabot, Renovate)
- 코드 자동 포맷 후 커밋
- 번역 파일 동기화
- 스크린샷 / 시각적 회귀 테스트 (Percy, Chromatic)
- Feature Flag 배포 연동

---

## 트리거 종류 요약

| 트리거 | 설명 |
|--------|------|
| `push` | 특정 브랜치에 코드 푸시 시 |
| `pull_request` | PR 생성/업데이트 시 |
| `schedule` | cron으로 주기적 실행 |
| `workflow_dispatch` | 수동 실행 |
| `release` | 릴리즈 생성 시 |
| `issue_comment` | 이슈/PR 코멘트 작성 시 |

팀의 기술 스택이나 특정 자동화하고 싶은 작업이 있으면 더 구체적인 예시 코드도 제공해드릴 수 있어요.

---
---
# GitHub Actions 트리거 종류 & 적절한 작업

## 📌 코드 관련 트리거

### `push`
코드가 특정 브랜치에 푸시될 때 발생

| 브랜치 | 적절한 작업 |
|--------|------------|
| `main` / `master` | 프로덕션 배포, Docker 이미지 빌드 & 푸시, 릴리즈 생성 |
| `develop` | 스테이징 배포, 통합 테스트 실행 |
| `feature/*` | 유닛 테스트, Lint, 타입 체크 |
| 모든 브랜치 | 코드 포맷 검사, 빌드 검증 |

```yaml
on:
  push:
    branches: [main, develop, 'feature/**']
    paths:
      - 'src/**'        # 특정 경로 변경 시만 실행
      - '!docs/**'      # docs 변경은 무시
```

---

### `pull_request`
PR이 생성/업데이트/머지될 때 발생

**적절한 작업:**
- ✅ 유닛 테스트 / 통합 테스트 실행
- ✅ Lint & 코드 포맷 검사
- ✅ 코드 커버리지 측정 및 PR 코멘트로 결과 게시
- ✅ 보안 취약점 스캔
- ✅ PR 사이즈 라벨 자동 부착 (small/medium/large)
- ✅ 리뷰어 자동 지정
- ✅ 미리보기(Preview) 환경 배포

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches: [main, develop]
```

---

### `pull_request_review`

PR 리뷰가 제출될 때 발생

**적절한 작업:**
- ✅ 승인(approved) 시 스테이징 자동 배포
- ✅ 변경 요청(changes_requested) 시 Slack 알림
- ✅ 특정 승인자 수 충족 시 자동 머지 트리거

```yaml
on:
  pull_request_review:
    types: [submitted, dismissed]
```

---

### `push` + 태그
태그가 푸시될 때 발생

**적절한 작업:**
- ✅ 공식 릴리즈 배포
- ✅ npm / PyPI 패키지 퍼블리시
- ✅ GitHub Release 자동 생성 + CHANGELOG 첨부
- ✅ 프로덕션 Docker 이미지 태깅 & 푸시

```yaml
on:
  push:
    tags:
      - 'v*.*.*'    # v1.0.0, v2.1.3 형태
```

---

## 📌 이슈 & 코멘트 트리거

### `issues`
이슈가 생성/수정/닫힐 때 발생

**적절한 작업:**
- ✅ 이슈 생성 시 자동 라벨링 (제목 키워드 기반)
- ✅ 담당자 자동 할당
- ✅ Jira / Linear 티켓 자동 생성 연동
- ✅ Slack 채널에 새 이슈 알림

```yaml
on:
  issues:
    types: [opened, labeled, closed, reopened]
```

---

### `issue_comment`
이슈 또는 PR에 코멘트가 작성될 때 발생

**적절한 작업:**
- ✅ `/deploy staging` 코멘트 → 스테이징 배포
- ✅ `/run tests` 코멘트 → 테스트 재실행
- ✅ `/approve` 코멘트 → 자동 머지
- ✅ 봇 명령어 기반 ChatOps 구현

```yaml
on:
  issue_comment:
    types: [created]

# 워크플로우 내에서 명령어 파싱
- if: contains(github.event.comment.body, '/deploy')
```

---

## 📌 스케줄 & 수동 트리거

### `schedule`
cron 표현식으로 주기적으로 실행

**적절한 작업:**

| 주기 | 적절한 작업 |
|------|------------|
| 매일 새벽 | DB 백업, 로그 정리, 리포트 생성 |
| 매주 월요일 | 의존성 업데이트 PR 생성, 주간 리포트 |
| 매시간 | 헬스체크, 모니터링 데이터 수집 |
| 매월 1일 | 비용 리포트, 인증서 만료 체크 |

```yaml
on:
  schedule:
    - cron: '0 2 * * *'      # 매일 새벽 2시
    - cron: '0 9 * * 1'      # 매주 월요일 오전 9시
    - cron: '0 0 1 * *'      # 매월 1일 자정
```

---

### `workflow_dispatch`
GitHub UI 또는 API에서 수동으로 실행

**적절한 작업:**
- ✅ 핫픽스 긴급 배포
- ✅ 특정 환경 선택 배포 (dev/staging/prod)
- ✅ 데이터 마이그레이션 수동 실행
- ✅ 캐시 초기화

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: '배포 환경'
        required: true
        type: choice
        options: [dev, staging, production]
      version:
        description: '배포 버전 (예: v1.2.3)'
        required: false
```

---

### `workflow_call`
다른 워크플로우에서 현재 워크플로우를 호출

**적절한 작업:**
- ✅ 공통 테스트 워크플로우 재사용
- ✅ 공통 배포 로직 모듈화
- ✅ 워크플로우 라이브러리 구성

```yaml
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
    secrets:
      deploy-key:
        required: true
```

---

## 📌 릴리즈 & 패키지 트리거

### `release`
GitHub Release가 생성/수정/퍼블리시될 때 발생

**적절한 작업:**
- ✅ 프로덕션 배포 실행
- ✅ 패키지 레지스트리 퍼블리시
- ✅ 릴리즈 노트 자동 생성
- ✅ CDN 캐시 퍼지

```yaml
on:
  release:
    types: [published, prereleased]
```

---

### `registry_package`
패키지가 GitHub Packages에 등록될 때 발생

**적절한 작업:**
- ✅ 새 패키지 버전 배포 자동 트리거
- ✅ 의존 서비스 자동 업데이트 알림

---

## 📌 외부 & 기타 트리거

### `repository_dispatch`
외부 시스템에서 API 호출로 트리거

**적절한 작업:**
- ✅ 외부 CI 시스템 연동 (Jenkins → GitHub Actions)
- ✅ 모니터링 알람 발생 시 자동 롤백
- ✅ 다른 레포 배포 완료 후 연계 배포

```yaml
on:
  repository_dispatch:
    types: [deploy-staging, rollback-production]
```

---

### `deployment` / `deployment_status`
배포 이벤트가 발생할 때

**적절한 작업:**
- ✅ 배포 성공 시 통합 테스트 자동 실행
- ✅ 배포 실패 시 자동 롤백 + 알림
- ✅ Slack / PagerDuty 알림 전송

---

### `status` / `check_run`
커밋 상태 또는 체크가 변경될 때

**적절한 작업:**
- ✅ 모든 체크 통과 시 자동 머지
- ✅ 체크 실패 시 담당자 알림

---

## 📊 트리거 선택 가이드 요약

```
코드 품질 검사     → push, pull_request
배포 (자동)        → push(main), release, tags
배포 (수동/긴급)   → workflow_dispatch
주기적 작업        → schedule
ChatOps / 명령어   → issue_comment
외부 시스템 연동   → repository_dispatch
워크플로우 재사용  → workflow_call
```

특정 기술 스택이나 시나리오에 맞는 실제 워크플로우 예시 코드가 필요하시면 말씀해 주세요! 🚀