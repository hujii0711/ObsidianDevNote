

```yaml
name: CI Pipeline

on:
	push:
		branches: [ "main" ]
	pull_request:
		branches: [ "main" ]
		
jobs:
	backend-ci:
		name: Backend CI (Node.js)
		runs-on: ubuntu-latest
	steps:
	- uses: actions/checkout@v4
	- name: Set up Node.js
		uses: actions/setup-node@v4
		with:
		node-version: '20'
		cache: 'npm'
		cache-dependency-path: ./package-lock.json

	- name: Install dependencies
		run: |
		if [ -f package-lock.json ]; then npm ci; else npm install; fi
	
	- name: Run Tests
		run: npm test
```


현재 작성된 `.github/workflows/ci.yml` 파일의 핵심인 `jobs` 섹션을 한 줄씩 자세히 설명해 드리겠습니다.

---

### 1. 작업 정의 (Jobs)
```yaml
jobs:
  backend-ci:
```
*   **`jobs:`**: 이 워크플로우에서 실행될 하나 이상의 작업(Job)들을 모아놓은 상위 카테고리입니다.
*   **`backend-ci:`**: 이 작업의 **고유 식별자(ID)**입니다. 사용자가 직접 지은 이름이며, 나중에 다른 작업에서 이 작업을 참조할 때 사용합니다.

### 2. 실행 환경 설정
```yaml
    name: Backend CI (Node.js)
    runs-on: ubuntu-latest
```
*   **`name:`**: GitHub Actions 웹 화면에 표시될 작업의 **표시 이름**입니다.
*   **`runs-on: ubuntu-latest`**: 이 작업이 돌아갈 **가상 서버의 종류**를 정합니다. 여기서는 가장 최신 버전의 Ubuntu 리눅스 환경을 사용하겠다고 지정했습니다.

### 3. 상세 단계 (Steps)
이 작업 내부에서 순차적으로 실행될 단계들입니다.

#### (1) 코드 가져오기
```yaml
    steps:
    - uses: actions/checkout@v4
```
*   **`uses: actions/checkout@v4`**: GitHub에서 제공하는 기본 액션입니다. 이 가상 서버 안으로 **여러분의 소스 코드를 그대로 복사(Clone)**해오는 역할을 합니다. 이 단계가 없으면 소스 코드가 없는 빈 서버에서 작업을 시작하게 됩니다.

#### (2) Node.js 설치 및 설정
```yaml
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'
        cache-dependency-path: ./package-lock.json
```
*   **`uses: actions/setup-node@v4`**: 가상 서버에 **Node.js 환경을 구축**해주는 액션입니다.
*   **`node-version: '20'`**: 설치할 Node.js 버전을 20으로 지정합니다.
*   **`cache: 'npm'`**: 빌드 속도를 높이기 위해, 이미 한 번 다운로드한 라이브러리들을 기억해두는 **캐싱 기능**을 켭니다.
*   **`cache-dependency-path`**: 캐시의 기준이 되는 파일인 `package-lock.json`의 위치를 알려줍니다.

#### (3) 의존성 라이브러리 설치
```yaml
    - name: Install dependencies
      run: |
        if [ -f package-lock.json ]; then npm ci; else npm install; fi
```
*   **`run: |`**: 가상 서버의 터미널에서 직접 실행할 쉘 명령어를 작성합니다.
*   **`if [ -f package-lock.json ] ...`**: `package-lock.json` 파일이 있는지 확인합니다.
    *   있다면 **`npm ci`**: CI 환경에 최적화된(더 빠르고 정확한) 설치 명령어를 사용합니다.
    *   없다면 **`npm install`**: 일반적인 설치 명령어를 사용합니다.

#### (4) 테스트 실행
```yaml
    - name: Run Tests
      run: npm test
```
*   **`npm test`**: `package.json` 파일 안에 정의된 `test` 스크립트를 실행합니다.
*   이 단계의 결과가 **성공(exit code 0)**이면 GitHub Actions에 초록색 체크가 뜨고, **실패(그 외)**하면 빨간색 X가 뜨게 됩니다.

---

### 요약
이 코드는 **"리눅스 서버를 하나 빌려서 -> 소스 코드를 복사하고 -> Node.js를 설치한 뒤 -> 필요한 라이브러리를 깔고 -> 마지막으로 테스트를 돌려본다"**는 일련의 과정을 자동화한 시나리오입니다!_
