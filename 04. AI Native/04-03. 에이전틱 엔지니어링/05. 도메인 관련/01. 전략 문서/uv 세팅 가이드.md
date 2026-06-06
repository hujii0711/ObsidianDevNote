
팀 프로젝트 수준에 딱 맞는 구조를 디렉터리 트리와 uv 세팅 가이드로 함께 보여드리겠습니다.

![[Pasted image 20260517124311.png]]
## uv 세팅 전체 가이드

### 1. 초기 세팅 (팀원 최초 1회)

```bash
# uv 설치 (curl 또는 pip 불필요)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 클론 후
cd rag-project/backend

# 가상환경 생성 + 의존성 설치 (한 번에)
uv sync
```

`uv sync`만 치면 `.venv/`가 자동 생성되고 `uv.lock`에 고정된 버전이 설치됩니다. `pip install -r requirements.txt` 와 `python -m venv` 두 단계가 하나로 합쳐진 셈입니다.

---

### 2. `pyproject.toml` 전체 예시

```toml
[project]
name = "rag-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "chromadb>=0.5.0",
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.13.0",
    "asyncpg>=0.29.0",
    "pydantic-settings>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.9",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",          # TestClient용
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "pre-commit>=3.7.0",
]

[tool.uv]
dev-dependencies = ["pytest", "ruff", "mypy", "pre-commit"]

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

### 3. 일상적인 팀 개발 명령어

```bash
# 패키지 추가 (uv.lock 자동 갱신)
uv add httpx

# 개발 전용 패키지 추가
uv add --dev pytest-cov

# 앱 실행 (가상환경 활성화 불필요)
uv run uvicorn app.main:app --reload

# 테스트 실행
uv run pytest

# 린트 + 포맷
uv run ruff check .
uv run ruff format .

# 타입 체크
uv run mypy app/

# 의존성 버전 일괄 업그레이드
uv lock --upgrade
uv sync
```

---

### 4. `Makefile` — 팀 표준 단축 명령

```makefile
.PHONY: install dev test lint format typecheck

install:
	cd backend && uv sync
	cd frontend && npm install

dev:
	docker compose up -d db chroma redis
	cd backend && uv run uvicorn app.main:app --reload &
	cd frontend && npm run dev

test:
	cd backend && uv run pytest --cov=app --cov-report=term-missing

lint:
	cd backend && uv run ruff check . && uv run mypy app/
	cd frontend && npm run lint

format:
	cd backend && uv run ruff format .
	cd frontend && npm run format

migrate:
	cd backend && uv run alembic upgrade head
```

---

### 5. `.pre-commit-config.yaml` — PR 전 자동 검사

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, fastapi]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
```

팀원 최초 세팅: `uv run pre-commit install`

---

### 6. `.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
    branches: [main, develop]

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Lint
        run: uv run ruff check .

      - name: Type check
        run: uv run mypy app/

      - name: Test
        run: uv run pytest --cov=app
        env:
          DATABASE_URL: sqlite+aiosqlite:///./test.db
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm run build
```

---

### 7. `.gitignore` 필수 항목

```gitignore
# uv 가상환경 (절대 커밋 금지)
backend/.venv/

# uv.lock은 반드시 커밋 (팀 전체 버전 고정)
# !backend/uv.lock  ← 주석 해제 불필요, 기본 추적됨

# 환경변수
backend/.env
frontend/.env.local

# Chroma 로컬 데이터
backend/chroma_db/

# 빌드 아티팩트
frontend/.next/
**/__pycache__/
```

---

### 프로토타입 → 팀 수준 주요 추가 사항 요약

|항목|추가 이유|
|---|---|
|`uv.lock` 커밋|팀 전원 동일 버전 보장|
|`pyproject.toml` 통합|pip + setup.cfg + requirements.txt 대체|
|`pre-commit`|PR 전 자동 lint/format 강제|
|`alembic/`|DB 스키마 변경 이력 공유|
|`tests/` + `conftest.py`|기능 추가 시 회귀 방지|
|`CONTRIBUTING.md`|브랜치 전략, 커밋 컨벤션 문서화|
|`docker-compose.override.yml`|개발자별 로컬 설정 분리|

`docker-compose.override.yml`이나 `alembic` 초기 설정 코드가 필요하면 말씀해 주세요!