
## Coverage란?

테스트 실행 중 **어떤 코드가 실행됐는지 추적**하는 기능입니다. "내 테스트가 실제로 코드의 몇 %를 검증하고 있는가"를 측정합니다.

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

---
Playwright에서 커버리지(Coverage)를 측정하는 가장 핵심적인 목적은 "우리가 짠 테스트 코드가 실제 웹 애플리케이션의 소스 코드를 얼마나 구석구석 실행시켰는가"를 확인하기 위함입니다.

Playwright는 주로 E2E(End-to-End) 테스트나 컴포넌트 테스트를 수행하는데, 이때 커버리지를 측정하면 다음과 같은 실무적인 이점을 얻을 수 있습니다.

### 1. 테스트의 빈틈(죽은 코드) 발견

사용자가 화면에서 버튼을 누르고 페이지를 이동하는 시나리오를 테스트할 때, 실제 브라우저가 실행한 JavaScript와 CSS 코드의 백분율(%)이 나옵니다.

이를 통해 "사용자 시나리오 테스트를 다 돌렸는데도 한 번도 실행되지 않은 프론트엔드 코드 블록"이 어디인지 정확히 찾아내어 테스트 케이스를 보완할 수 있습니다.

### 2. 불필요한 코드(Dead Code) 제거 및 최적화

테스트를 완벽하게 돌렸음에도 불구하고 프로덕션 번들 파일에서 전혀 실행되지 않는 JS/CSS 코드가 많다면, 이는 UI 라이브러리에서 쓰이지 않는 기능이 포함되어 있거나 과거의 유산(Legacy) 코드일 확률이 높습니다. 이를 파악해 코드 성능을 최적화(Tree-shaking)하는 지표로 삼습니다.

### 3. 과도한 E2E 테스트 방지 (테스트 효율화)

E2E 테스트는 실행 비용이 크고 속도가 느립니다. 커버리지를 모니터링하면 중복되는 UI 테스트를 줄일 수 있습니다.

- _예: "A 기능을 검증하면서 이미 B 코드가 100% 실행되었다면, B를 위한 무거운 E2E 테스트를 굳이 또 만들지 않고 가벼운 단위(Unit) 테스트로 대체하자"는 의사결정이 가능해집니다._
    

## 🛠️ Playwright에서 커버리지를 측정하는 원리

Playwright는 Chromium 기반 브라우저가 제공하는 `V8 프로파일러(V8 Coverage)` API를 직접 제어할 수 있습니다.

JavaScript

```
// Playwright에서 커버리지를 시작하고 수집하는 기본 흐름
await page.coverage.startJSCoverage();
await page.goto('https://example.com');
// ... 사용자 행동 테스트 수행 ...
const coverage = await page.coverage.stopJSCoverage();
```

> ⚠️ **주의할 점 (실무 팁)**
> 
> Playwright가 수집하는 원본 데이터는 브라우저가 읽은 '빌드/압축된 파일(main.min.js)' 기준입니다. 따라서 우리가 작성한 원본 소스코드(TypeScript, React, Vue 등) 기준의 깔끔한 리포트를 보려면 **Babel 플러그인(`babel-plugin-istanbul`)**을 사용해 코드에 인스트루멘테이션(Instrumentation, 코드 전처리)을 수집하고 소스맵(Source Map)을 연동해야 익숙한 형식의 리포트를 얻을 수 있습니다.