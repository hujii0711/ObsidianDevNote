
가장 큰 목적은 **CLAUDE.md 분리**입니다.

공식 문서도 이렇게 설명합니다:

> "For larger projects, you can organize instructions into multiple files using the `.claude/rules/` directory. This keeps instructions modular and easier for teams to maintain."

---

### 기능 차이로 보면

||CLAUDE.md|`.claude/rules/`|
|---|---|---|
|핵심 목적|항상 필요한 핵심 규칙|CLAUDE.md 분리 + 선택적 로드|
|path 스코프|❌|✅ (선택사항)|
|토픽별 분리|하나의 파일|여러 파일로 분산 가능|

`paths` 지정은 **부가 기능**이고, 단순 분리만 해도 충분히 가치 있습니다.

---

### 실제로는 이 순서로 도입하게 됨

1. 처음엔 `CLAUDE.md` 하나로 시작
2. 내용이 늘어나면 → `rules/`로 토픽별 분리
3. 규모가 더 커지면 → `paths`로 스코프 제한까지 추가

결국 `rules/`는 **CLAUDE.md의 확장 공간**이라고 보면 가장 정확합니다.

---

## `.claude/rules/` 활용 가이드

### CLAUDE.md vs rules/ 차이부터 이해

||`CLAUDE.md`|`.claude/rules/`|
|---|---|---|
|로드 시점|세션 시작 시 **항상**|시작 시 전체 로드 OR 해당 파일 열 때만|
|용도|항상 필요한 핵심 규칙|상황별/파일별 선택적 규칙|
|크기|200줄 이하 권장|토픽별로 분리 가능|

핵심은 **"항상 필요한가, 아니면 특정 상황에서만 필요한가"**로 구분하면 됩니다.

---

### 유용한 활용 패턴

**1. 언어/파일 타입별 규칙** (path-scoped)

```yaml
# .claude/rules/typescript.md
---
paths:
  - "**/*.ts"
  - "**/*.tsx"
---
- strict mode 필수
- any 타입 사용 금지
- interface보다 type 선호
```

**2. 레이어별 규칙** (path-scoped)

```yaml
# .claude/rules/api.md
---
paths:
  - "src/api/**/*.ts"
---
- 모든 엔드포인트 입력값 검증 필수
- 표준 에러 응답 포맷 사용
- OpenAPI 주석 포함
```

```yaml
# .claude/rules/components.md
---
paths:
  - "src/components/**/*.tsx"
---
- props에 반드시 타입 명시
- 스타일은 CSS module만 사용
- 컴포넌트당 파일 하나
```

**3. 테스트 규칙**

```yaml
# .claude/rules/testing.md
---
paths:
  - "**/*.test.ts"
  - "**/*.spec.ts"
---
- describe/it 구조 필수
- mock은 파일 상단에 선언
- 각 테스트는 독립적으로 실행 가능해야 함
```

**4. 보안 규칙** (path 없이 항상 적용)

```
# .claude/rules/security.md
- 환경변수 직접 하드코딩 절대 금지
- SQL은 반드시 parameterized query 사용
- 사용자 입력값은 항상 sanitize
- 시크릿 관련 코드 변경 시 보안팀 리뷰 필요 명시
```

**5. 마이그레이션/레거시 주의사항**

```yaml
# .claude/rules/legacy.md
---
paths:
  - "src/legacy/**"
---
- 이 디렉토리는 리팩토링 금지
- 새 기능 추가 시 반드시 신규 모듈에 작성
- 여기서 import 추가할 때 팀장 승인 필요
```

---

### 구조 예시

```
.claude/
├── CLAUDE.md              # 핵심 공통 규칙 (항상 로드)
└── rules/
    ├── typescript.md      # TS 파일에만 적용
    ├── testing.md         # 테스트 파일에만 적용
    ├── api.md             # API 레이어에만 적용
    ├── security.md        # 항상 적용 (paths 없음)
    └── legacy.md          # 레거시 디렉토리에만 적용
```

---

### 판단 기준 요약

- **CLAUDE.md에 넣을 것** → 빌드 명령어, 프로젝트 전체 구조, 네이밍 컨벤션 등 항상 필요한 것
- **rules/에 넣을 것** → 특정 파일 타입이나 디렉토리에서만 의미 있는 규칙, 내용이 길어서 분리하고 싶은 것
- **skills에 넣을 것** → 특정 태스크를 수행할 때만 필요한 절차적 지침 (배포 절차, PR 작성법 등)

---

`paths` 없이도 충분히 유용합니다. 핵심은 **"CLAUDE.md가 너무 길어질 때 분리"** 입니다.

---

### paths 없는 rules/ — 단순 분리 목적

```
.claude/rules/
├── security.md      # 항상 적용, 보안 규칙만 모아둠
├── git-workflow.md  # 항상 적용, 커밋/PR 규칙
└── testing.md       # 항상 적용, 테스트 규칙
```

CLAUDE.md 하나에 200줄 넘게 우겨넣는 것보다, 토픽별로 파일을 나누는 것만으로도 **유지보수가 훨씬 편해집니다.** 팀에서 "보안 규칙 바꾸자"고 하면 `security.md`만 열면 되니까요.

---

### paths 지정 — 컨텍스트 절약 목적

`paths`의 진짜 가치는 **불필요한 규칙을 컨텍스트에서 제외**하는 것입니다.

예를 들어 레거시 디렉토리 주의사항을 `paths: ["src/legacy/**"]`로 걸어두면, 레거시 파일을 건드릴 때만 로드됩니다. 평소엔 컨텍스트를 차지하지 않습니다.

규칙이 많아질수록 이 차이가 커집니다.

---

### 정리

- **규칙이 적다면** → `paths` 없이 토픽별 분리만 해도 충분
- **규모가 커질수록** → `paths`로 스코프를 좁혀서 컨텍스트 효율화

결국 `rules/`의 핵심은 **CLAUDE.md 분리 + 선택적 로드** 두 가지이고, `paths`는 후자를 위한 옵션입니다.

---
## CLAUDE.md 예시 (세분화 전)

```markdown
# 프로젝트 개요
- E-commerce 플랫폼 (Next.js + FastAPI + PostgreSQL)
- 팀: 프론트 3명, 백엔드 2명

# 빌드 & 실행
- 프론트: `cd frontend && npm run dev`
- 백엔드: `cd backend && uvicorn main:app --reload`
- 테스트: `npm test` / `pytest`

# 코드 스타일
- TypeScript strict mode 필수
- any 타입 사용 금지
- 함수형 컴포넌트만 사용 (클래스 컴포넌트 금지)
- Python은 type hint 필수
- Black 포매터 사용

# 테스트
- 모든 API 엔드포인트는 테스트 필수
- 커버리지 80% 이상 유지
- 테스트 파일명: `*.test.ts` / `test_*.py`
- mock은 파일 상단에 선언

# API 설계
- RESTful 원칙 준수
- 응답 형식: `{ data, error, meta }` 통일
- 인증이 필요한 엔드포인트는 JWT 미들웨어 적용
- 입력값 검증 필수 (Pydantic 사용)

# 보안
- 환경변수 하드코딩 절대 금지
- SQL은 parameterized query만 사용
- 사용자 입력값 항상 sanitize
- 시크릿 관련 코드 변경 시 주석으로 SECURITY 태그 추가

# Git
- 커밋 메시지: `feat:` `fix:` `chore:` `docs:` 접두사 사용
- PR은 최소 1명 리뷰 후 머지
- main 브랜치 직접 push 금지
```

이걸 `rules/`로 세분화하면 👇

---

## 세분화 후 구조

```
.claude/
├── CLAUDE.md              # 핵심 공통 내용만
└── rules/
    ├── typescript.md      # TS 파일에만 적용
    ├── python.md          # Python 파일에만 적용
    ├── testing.md         # 테스트 파일에만 적용
    ├── api.md             # API 레이어에만 적용
    ├── security.md        # 항상 적용
    └── git.md             # 항상 적용
```

---

### CLAUDE.md (세분화 후 — 핵심만 남김)

```markdown
# 프로젝트 개요
- E-commerce 플랫폼 (Next.js + FastAPI + PostgreSQL)
- 팀: 프론트 3명, 백엔드 2명

# 빌드 & 실행
- 프론트: `cd frontend && npm run dev`
- 백엔드: `cd backend && uvicorn main:app --reload`
- 테스트: `npm test` / `pytest`
```

---

### `.claude/rules/typescript.md`

```markdown
---
paths:
  - "frontend/**/*.ts"
  - "frontend/**/*.tsx"
---
- strict mode 필수
- any 타입 사용 금지
- 함수형 컴포넌트만 사용 (클래스 컴포넌트 금지)
- props는 반드시 타입 명시
- 컴포넌트당 파일 하나
```

---

### `.claude/rules/python.md`

```markdown
---
paths:
  - "backend/**/*.py"
---
- type hint 필수
- Black 포매터 사용
- 함수 docstring 필수
```

---

### `.claude/rules/testing.md`

```markdown
---
paths:
  - "**/*.test.ts"
  - "**/*.spec.ts"
  - "**/test_*.py"
---
- 커버리지 80% 이상 유지
- mock은 파일 상단에 선언
- 각 테스트는 독립적으로 실행 가능해야 함
- describe/it 구조 필수 (TypeScript)
```

---

### `.claude/rules/api.md`

```markdown
---
paths:
  - "backend/api/**/*.py"
---
- RESTful 원칙 준수
- 응답 형식: `{ data, error, meta }` 통일
- 인증 필요 엔드포인트는 JWT 미들웨어 적용
- 입력값 검증 필수 (Pydantic 사용)
- 모든 엔드포인트 테스트 필수
```

---

### `.claude/rules/security.md` (paths 없음 — 항상 적용)

```markdown
- 환경변수 하드코딩 절대 금지
- SQL은 parameterized query만 사용
- 사용자 입력값 항상 sanitize
- 시크릿 관련 코드 변경 시 주석으로 SECURITY 태그 추가
```

---

### `.claude/rules/git.md` (paths 없음 — 항상 적용)

```markdown
- 커밋 메시지: `feat:` `fix:` `chore:` `docs:` 접두사 사용
- PR은 최소 1명 리뷰 후 머지
- main 브랜치 직접 push 금지
```

---

## 효과 비교

||세분화 전|세분화 후|
|---|---|---|
|CLAUDE.md 길이|~40줄|~8줄|
|TS 작업 시 로드|보안+Git+Python 규칙까지 전부|TS 규칙 + 보안 + Git만|
|Python 작업 시 로드|전부|Python 규칙 + 보안 + Git만|
|규칙 수정|한 파일에서 찾아야 함|해당 파일만 열면 됨|

규모가 작을 땐 CLAUDE.md 하나로 충분하고, **팀이 커지거나 규칙이 20줄을 넘어가기 시작할 때** `rules/`로 분리하는 것이 자연스러운 시점입니다.