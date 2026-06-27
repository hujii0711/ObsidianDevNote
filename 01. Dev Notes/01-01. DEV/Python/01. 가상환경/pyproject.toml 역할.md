
Python 프로젝트의 **설정을 한 곳에서 관리하는 통합 설정 파일**입니다 (PEP 517/518 표준).

---

### 핵심 역할

|역할|설명|
|---|---|
|**패키지 메타데이터**|프로젝트 이름, 버전, 설명, 라이선스|
|**의존성 관리**|필요한 패키지 목록 정의|
|**빌드 설정**|패키지 빌드 방식 지정|
|**도구 설정**|linter, formatter, 테스트 도구 설정 통합|

---

### 기본 구조 예시

```toml
[build-system]
requires = ["setuptools"]          # 빌드 도구 지정
build-backend = "setuptools.build_meta"

[project]
name = "my-project"                # 패키지 이름
version = "1.0.0"
description = "My Python project"
requires-python = ">=3.10"

dependencies = [                   # 런타임 의존성
    "fastapi>=0.100.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [                            # 개발용 의존성
    "pytest",
    "ruff",
]

[tool.ruff]                        # 도구별 설정
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

### 기존 파일들과의 관계

```
이전 방식                    pyproject.toml 로 통합
─────────────────────────────────────────────
setup.py              ──→   [project]
setup.cfg             ──→   [project]
requirements.txt      ──→   [project.dependencies]
.flake8 / .pylintrc   ──→   [tool.ruff] / [tool.pylint]
pytest.ini            ──→   [tool.pytest.ini_options]
```

한 파일로 프로젝트의 **빌드 + 의존성 + 도구 설정**을 모두 관리하는 현대적인 Python 표준입니다.


---

## 비교

|역할|`package.json` (Node.js)|`pyproject.toml` (Python)|
|---|---|---|
|프로젝트 메타데이터|`name`, `version`, `description`|`[project]` name, version|
|런타임 의존성|`dependencies`|`[project.dependencies]`|
|개발 의존성|`devDependencies`|`[project.optional-dependencies.dev]`|
|스크립트|`scripts`|`[project.scripts]`|
|빌드 설정|없음 (번들러별 설정)|`[build-system]`|
|도구 설정|일부 (`eslintConfig` 등)|`[tool.*]` 로 완전 통합|

---

### 나란히 비교

**package.json**

```json
{
  "name": "my-app",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.0"
  },
  "devDependencies": {
    "eslint": "^8.0.0"
  },
  "scripts": {
    "start": "node index.js"
  }
}
```

**pyproject.toml**

```toml
[project]
name = "my-app"
version = "1.0.0"
dependencies = [
    "fastapi>=0.100.0",
]

[project.optional-dependencies]
dev = ["ruff"]

[project.scripts]
start = "my_app:main"
```

---

### 차이점

- `package.json`은 **npm이 직접 실행**까지 담당하지만, `pyproject.toml`은 **선언만** 하고 실행은 `pip`, `uv`, `poetry` 등 별도 도구가 담당
- Python은 빌드 도구가 다양해(`setuptools`, `poetry`, `hatch`) `[build-system]`으로 명시가 필요

개념적으로는 **거의 같은 역할**이라고 봐도 무방합니다.