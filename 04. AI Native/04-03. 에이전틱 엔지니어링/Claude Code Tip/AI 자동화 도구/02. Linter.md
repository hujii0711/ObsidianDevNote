
Claude Code에서 린터를 자동화에 통합하는 방법을 설명할게요.

---

## 1. CLAUDE.md에 린터 규칙 정의

```markdown
# CLAUDE.md

## 코드 품질 규칙
모든 코드 수정 후 반드시 다음을 실행할 것:

### 필수 체크
1. `npm run lint` - ESLint 검사
2. `npm run type-check` - TypeScript 타입 검사  
3. `npm run test` - 테스트 통과 확인

### 린터 실패 시
- 자동으로 에러 수정 후 재검사
- 수정 불가능한 경우 사용자에게 보고
- PR 생성 전 반드시 lint 통과 필수
```

---

## 2. package.json 스크립트 설정

```json
{
  "scripts": {
    "lint": "eslint . --ext .ts,.tsx,.js,.jsx",
    "lint:fix": "eslint . --ext .ts,.tsx,.js,.jsx --fix",
    "type-check": "tsc --noEmit",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    
    "pre-commit": "npm run lint && npm run type-check",
    "ci:check": "npm run lint && npm run type-check && npm run test"
  }
}
```

---

## 3. ESLint 설정 (`.eslintrc.json`)

```json
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react-hooks/recommended",
    "prettier"
  ],
  "rules": {
    "no-unused-vars": "error",
    "no-console": "warn",
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/explicit-function-return-type": "warn"
  }
}
```

---

## 4. Husky + lint-staged (Git Hook 자동화)

커밋 전 자동으로 린터가 실행되게 설정합니다.

```bash
# 설치
npm install -D husky lint-staged
npx husky init
```

```bash
# .husky/pre-commit
#!/bin/sh
npx lint-staged
```

```json
// package.json에 추가
{
  "lint-staged": {
    "*.{ts,tsx,js,jsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,css}": [
      "prettier --write"
    ]
  }
}
```

---

## 5. Claude Code 자동화 스크립트

### 린터 체크 포함 워크플로우

```bash
#!/bin/bash
# scripts/claude-workflow.sh

echo "🔍 린터 검사 시작..."

# ESLint 실행
npm run lint
LINT_EXIT=$?

if [ $LINT_EXIT -ne 0 ]; then
  echo "❌ ESLint 실패 - 자동 수정 시도..."
  npm run lint:fix
  
  # 재검사
  npm run lint
  if [ $? -ne 0 ]; then
    echo "🚨 자동 수정 불가 - 수동 확인 필요"
    exit 1
  fi
fi

# 타입 체크
npm run type-check
if [ $? -ne 0 ]; then
  echo "🚨 TypeScript 에러 발견"
  exit 1
fi

echo "✅ 모든 검사 통과 - PR 생성 가능"
```

---

## 6. GitHub Actions CI 설정

```yaml
# .github/workflows/lint.yml
name: Lint Check

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          
      - run: npm ci
      
      - name: ESLint
        run: npm run lint
        
      - name: TypeScript Check
        run: npm run type-check
        
      - name: Prettier Check
        run: npm run format:check
```

---

## 7. Claude Code에 린터 자동 실행 지시

### CLAUDE.md에 추가

```markdown
## 자동화 규칙 (반드시 준수)

### 코드 수정 후 체크리스트
- [ ] `npm run lint:fix` 실행
- [ ] `npm run type-check` 실행
- [ ] 에러 0개 확인 후 커밋
- [ ] PR 생성 전 `npm run ci:check` 통과

### Claude가 지켜야 할 규칙
1. 코드 파일 수정 시 항상 린터 실행
2. lint 에러는 즉시 수정 (나중에 하지 않음)
3. `any` 타입 사용 금지
4. console.log는 개발용 주석 처리
```

---

## 8. VS Code 설정 (선택사항)

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "eslint.validate": [
    "javascript",
    "typescript",
    "javascriptreact",
    "typescriptreact"
  ]
}
```

---

## 전체 자동화 흐름

```
Claude Code 코드 수정
       ↓
lint:fix 자동 실행  ←── CLAUDE.md 규칙
       ↓
type-check 실행
       ↓
git commit (Husky pre-commit hook)
       ↓
lint-staged 재검사
       ↓
git push → GitHub Actions CI
       ↓
PR 생성 가능 ✅
```

---

## 핵심 포인트

|단계|도구|역할|
|---|---|---|
|코드 작성 중|ESLint + Prettier|실시간 수정|
|커밋 전|Husky + lint-staged|자동 강제 검사|
|PR 전|Claude CLAUDE.md 규칙|AI가 자체 검사|
|PR 후|GitHub Actions|CI 최종 검증|

**CLAUDE.md에 린터 규칙을 명시하는 것**이 가장 중요합니다. Claude Code는 이 파일을 기반으로 자율적으로 린터를 실행하고 에러를 수정합니다.