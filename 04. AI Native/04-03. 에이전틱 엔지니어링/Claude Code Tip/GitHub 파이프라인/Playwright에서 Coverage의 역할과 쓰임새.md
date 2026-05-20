
## Coverage란?

테스트 실행 중 **어떤 코드가 실행됐는지 추적**하는 기능입니다.
"내 테스트가 실제로 코드의 몇 %를 검증하고 있는가"를 측정합니다.

---

## Playwright Coverage의 종류

### 1. JavaScript Coverage
```javascript
// 어떤 JS 코드가 실행됐는지 추적
await page.coverage.startJSCoverage();
// ... 테스트 실행 ...
const coverage = await page.coverage.stopJSCoverage();
```

### 2. CSS Coverage
```javascript
// 어떤 CSS 규칙이 실제로 적용됐는지 추적
await page.coverage.startCSSCoverage();
// ... 테스트 실행 ...
const coverage = await page.coverage.stopCSSCoverage();
```

---

## 기본 사용법

```javascript
import { test } from '@playwright/test';

test('coverage 기본 예시', async ({ page }) => {

  // 1. Coverage 수집 시작
  await page.coverage.startJSCoverage();
  await page.coverage.startCSSCoverage();

  // 2. 테스트 실행
  await page.goto('https://myapp.com');
  await page.click('#login-button');
  await page.fill('#username', 'testuser');

  // 3. Coverage 수집 종료
  const jsCoverage  = await page.coverage.stopJSCoverage();
  const cssCoverage = await page.coverage.stopCSSCoverage();

  // 4. 결과 확인
  for (const entry of jsCoverage) {
    console.log(`파일: ${entry.url}`);
    console.log(`사용된 바이트: ${entry.ranges.reduce((acc, r) => acc + r.end - r.start, 0)}`);
    console.log(`전체 바이트: ${entry.text.length}`);
  }
});
```

---

## Coverage 데이터 구조

```javascript
// jsCoverage 배열의 각 항목 구조
{
  url: "https://myapp.com/bundle.js",   // 파일 경로
  text: "...전체 소스코드...",           // 전체 코드 내용
  ranges: [                             // 실행된 범위
    { start: 0,   end: 100 },           // 0~100번째 문자 실행됨
    { start: 250, end: 400 },           // 250~400번째 실행됨
    // 101~249는 실행 안됨 → 미커버 구간
  ]
}
```

---

## 실전 활용 — 커버리지 % 계산

```javascript
test('커버리지 퍼센트 측정', async ({ page }) => {
  await page.coverage.startJSCoverage();

  await page.goto('/');
  await page.click('.main-button');

  const coverage = await page.coverage.stopJSCoverage();

  let totalBytes = 0;
  let usedBytes  = 0;

  for (const entry of coverage) {
    totalBytes += entry.text.length;
    for (const range of entry.ranges) {
      usedBytes += range.end - range.start;
    }
  }

  const percent = ((usedBytes / totalBytes) * 100).toFixed(2);
  console.log(`✅ JS 커버리지: ${percent}%`);
  console.log(`실행된 코드: ${usedBytes} bytes / 전체: ${totalBytes} bytes`);
});
```

---

## 실전 활용 — Istanbul/NYC 리포트 연동

가장 많이 쓰이는 패턴으로, **HTML 리포트**로 시각화합니다.

```bash
npm install -D istanbul-lib-coverage istanbul-lib-report istanbul-reports v8-to-istanbul
```

```javascript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  use: {
    // 모든 테스트에서 coverage 활성화
  },
  reporter: [['html'], ['json', { outputFile: 'coverage/results.json' }]],
});
```

```javascript
// tests/coverage.setup.ts
import { chromium } from '@playwright/test';
import v8ToIstanbul from 'v8-to-istanbul';
import fs from 'fs';

async function collectCoverage() {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.coverage.startJSCoverage();
  await page.goto('http://localhost:3000');

  // 테스트 액션들...

  const coverage = await page.coverage.stopJSCoverage();

  // v8 포맷 → Istanbul 포맷 변환
  for (const entry of coverage) {
    const converter = v8ToIstanbul('', 0, { source: entry.text });
    await converter.load();
    converter.applyCoverage(entry.ranges);

    const istanbulCoverage = converter.toIstanbul();
    fs.writeFileSync(
      `coverage/${Date.now()}.json`,
      JSON.stringify(istanbulCoverage)
    );
  }

  await browser.close();
}
```

---

## 실전 활용 — 미사용 코드 감지

```javascript
test('사용 안 되는 JS 파일 감지', async ({ page }) => {
  await page.coverage.startJSCoverage();
  await page.goto('/');

  const coverage = await page.coverage.stopJSCoverage();

  const unused = coverage
    .filter(entry => {
      const usedBytes = entry.ranges.reduce(
        (acc, r) => acc + r.end - r.start, 0
      );
      const ratio = usedBytes / entry.text.length;
      return ratio < 0.1; // 10% 미만 사용된 파일
    })
    .map(e => e.url);

  console.log('⚠️ 거의 사용 안 되는 파일:', unused);
});
```

---

## 실전 활용 — CSS 미사용 규칙 감지

```javascript
test('미사용 CSS 감지', async ({ page }) => {
  await page.coverage.startCSSCoverage();
  await page.goto('/');

  const cssCoverage = await page.coverage.stopCSSCoverage();

  let totalRules = 0;
  let unusedRules = 0;

  for (const entry of cssCoverage) {
    const used  = entry.ranges.reduce((acc, r) => acc + r.end - r.start, 0);
    const total = entry.text.length;
    const unused = total - used;

    totalRules  += total;
    unusedRules += unused;

    console.log(`${entry.url}`);
    console.log(`  미사용: ${((unused/total)*100).toFixed(1)}%`);
  }

  console.log(`\n전체 CSS 미사용 비율: ${((unusedRules/totalRules)*100).toFixed(1)}%`);
});
```

---

## Coverage 주요 쓰임새 정리

```
📊 테스트 품질 측정
   └── 테스트가 코드의 몇 %를 실행하는지 확인

🗑️ 데드코드 제거
   └── 한 번도 실행 안 된 함수/파일 식별

🎨 미사용 CSS 제거
   └── 번들 사이즈 최적화에 활용

🔗 CI/CD 품질 게이트
   └── 커버리지 80% 미만이면 빌드 실패 처리

📈 리포트 시각화
   └── Istanbul/NYC와 연동해 HTML 리포트 생성
```

---

## 주의사항

```
⚠️ Chromium 전용
   Firefox, WebKit에서는 Coverage API 미지원

⚠️ 번들된 코드 기준
   소스맵 없으면 원본 파일과 매핑이 어려움
   → v8-to-istanbul로 변환 필요

⚠️ 성능 오버헤드
   Coverage 수집 중 테스트 속도 약간 저하
   → CI에서만 활성화 권장
```

CI 연동이나 리포트 시각화 설정이 필요하시면 더 자세히 알려드릴게요! 😊