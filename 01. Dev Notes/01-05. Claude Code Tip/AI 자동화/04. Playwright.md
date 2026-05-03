Claude Code에서 Playwright E2E 테스트를 담당하는 서브에이전트를 구성하는 방법을 설명할게요.

---

## 전체 아키텍처

```
메인 Claude Code
├── 코드 수정 에이전트
├── 🎭 Playwright E2E 테스트 서브에이전트  ← 오늘 주제
└── PR 생성 에이전트
```

---

## 1. CLAUDE.md에 서브에이전트 역할 정의

```markdown
# CLAUDE.md

## 서브에이전트 설정

### E2E 테스트 에이전트 (playwright-agent)
코드 수정이 완료되면 자동으로 아래를 수행:

**트리거 조건:**
- UI 컴포넌트 수정 시
- API 엔드포인트 변경 시
- 인증 흐름 변경 시

**수행 작업:**
1. `npx playwright test` 전체 테스트 실행
2. 실패한 테스트 스크린샷 캡처
3. 실패 원인 분석 후 메인 에이전트에 보고
4. 필요시 테스트 코드 자동 업데이트

**규칙:**
- 테스트 실패 시 코드 수정 없이 보고만 할 것
- 스크린샷은 test-results/ 폴더에 저장
- 테스트 통과율 100% 달성 후 PR 허용
```

---

## 2. Playwright 설정

```bash
# 설치
npm init playwright@latest
```

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['line']  // Claude가 읽기 쉬운 출력
  ],
  
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'on-first-retry',
  },

  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'Mobile', use: { ...devices['iPhone 13'] } },
  ],

  // 테스트 전 dev 서버 자동 시작
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

---

## 3. 서브에이전트 실행 스크립트

```bash
#!/bin/bash
# scripts/playwright-agent.sh
# Claude Code가 이 스크립트를 서브에이전트로 호출

TARGET=$1  # 테스트할 대상 (auth, dashboard, all)
REPORT_FILE="test-results/agent-report.md"

echo "🎭 Playwright 서브에이전트 시작..."
echo "대상: $TARGET"

# 테스트 실행
if [ "$TARGET" = "all" ]; then
  npx playwright test --reporter=json 2>&1
else
  npx playwright test --grep "@$TARGET" --reporter=json 2>&1
fi

EXIT_CODE=$?

# 결과 분석 및 리포트 생성
node scripts/parse-playwright-results.js

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 모든 E2E 테스트 통과"
  echo "STATUS=PASS" >> $REPORT_FILE
else
  echo "❌ E2E 테스트 실패 - 메인 에이전트에 보고"
  echo "STATUS=FAIL" >> $REPORT_FILE
  cat $REPORT_FILE  # Claude가 읽을 수 있게 출력
  exit 1
fi
```

```javascript
// scripts/parse-playwright-results.js
// 테스트 결과를 Claude가 읽기 쉽게 변환

const fs = require('fs');
const results = JSON.parse(
  fs.readFileSync('test-results/results.json', 'utf-8')
);

const report = {
  total: results.stats.expected,
  passed: results.stats.expected - results.stats.unexpected,
  failed: results.stats.unexpected,
  failures: results.suites
    .flatMap(s => s.specs)
    .filter(spec => spec.ok === false)
    .map(spec => ({
      test: spec.title,
      error: spec.tests[0]?.results[0]?.error?.message,
      screenshot: spec.tests[0]?.results[0]?.attachments
        ?.find(a => a.name === 'screenshot')?.path
    }))
};

fs.writeFileSync(
  'test-results/agent-report.md',
  `# E2E 테스트 결과\n
- 전체: ${report.total}
- 통과: ${report.passed}  
- 실패: ${report.failed}

## 실패 목록
${report.failures.map(f => 
  `### ❌ ${f.test}\n에러: ${f.error}\n스크린샷: ${f.screenshot}`
).join('\n\n')}
`);
```

---

## 4. E2E 테스트 파일 구조

```
e2e/
├── auth/
│   ├── login.spec.ts        # 로그인 플로우
│   └── signup.spec.ts       # 회원가입 플로우
├── dashboard/
│   └── dashboard.spec.ts    # 대시보드
└── fixtures/
    └── test-data.ts         # 공통 테스트 데이터
```

```typescript
// e2e/auth/login.spec.ts
import { test, expect } from '@playwright/test';

test.describe('인증 플로우 @auth', () => {
  
  test('로그인 성공', async ({ page }) => {
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'password123');
    await page.click('[data-testid="login-btn"]');
    
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('잘못된 비밀번호', async ({ page }) => {
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'wrong');
    await page.click('[data-testid="login-btn"]');
    
    await expect(page.locator('[data-testid="error-msg"]'))
      .toContainText('비밀번호가 올바르지 않습니다');
  });
});
```

---

## 5. 메인 에이전트와 연동

````markdown
# .claude/commands/test-and-fix.md

## E2E 테스트 서브에이전트 워크플로우

코드 수정 완료 후 다음 순서로 실행:

### Step 1: 서브에이전트 실행
```bash
bash scripts/playwright-agent.sh all
````

### Step 2: 결과 분석

- `test-results/agent-report.md` 읽기
- 실패한 테스트 목록 확인

### Step 3: 판단

- PASS → PR 생성 진행
- FAIL → 실패 원인 분석 후 사용자에게 보고 (테스트 실패는 코드를 수정하지 말고 반드시 보고할 것)

````

---

## 6. GitHub Actions 연동

```yaml
# .github/workflows/e2e.yml
name: E2E Tests (Playwright Agent)

on:
  pull_request:
    branches: [main]

jobs:
  playwright-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          
      - run: npm ci
      - run: npx playwright install --with-deps chromium
      
      - name: E2E 서브에이전트 실행
        run: bash scripts/playwright-agent.sh all
        
      - name: 실패 리포트 업로드
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: |
            playwright-report/
            test-results/
````

---

## 전체 자동화 흐름

```
Claude Code 코드 수정
        ↓
Playwright 서브에이전트 자동 실행
        ↓
   테스트 통과? 
   ↙         ↘
PASS         FAIL
  ↓            ↓
PR 생성    스크린샷 + 에러 보고
              ↓
         사용자 확인 후 수정 결정
```

---

## 핵심 포인트

|요소|역할|
|---|---|
|`CLAUDE.md`|서브에이전트 트리거 조건 정의|
|`playwright-agent.sh`|실행 및 결과 수집|
|`agent-report.md`|Claude가 읽는 구조화된 결과|
|`data-testid` 속성|안정적인 셀렉터 (CSS 변경에 무관)|
|GitHub Actions|CI에서 동일하게 자동 실행|

**`data-testid` 속성을 컴포넌트에 미리 추가**해두면 Claude Code가 테스트 코드를 더 정확하게 생성할 수 있습니다.