Claude Code에서 GitHub 이슈 등록 → 리뷰 → 해결 → PR까지 자동화하는 방법을 설명할게요.

## 전체 흐름

```
이슈 등록 → worktree 생성 → 코드 수정 → PR 생성
```

---

## 1. CLAUDE.md 설정 (프로젝트 루트)

프로젝트에 자동화 워크플로우를 정의합니다.

```markdown
# CLAUDE.md

## GitHub 워크플로우 자동화

### 이슈 처리 프로세스
1. `gh issue create`로 이슈 등록
2. `git worktree add`로 격리된 작업 환경 생성
3. 해당 worktree에서 문제 해결
4. `gh pr create`로 PR 생성

### 명령어 규칙
- 브랜치명: `fix/issue-{번호}-{설명}` 또는 `feat/issue-{번호}-{설명}`
- 커밋: `fix: #{번호} 설명` 형식
```

---

## 2. 자동화 스크립트 (`scripts/auto-workflow.sh`)

```bash
#!/bin/bash
# 이슈 생성 → worktree → PR 자동화

ISSUE_TITLE="$1"
ISSUE_BODY="$2"
BASE_BRANCH="${3:-main}"

# 1. GitHub 이슈 등록
echo "📌 이슈 등록 중..."
ISSUE_URL=$(gh issue create \
  --title "$ISSUE_TITLE" \
  --body "$ISSUE_BODY" \
  --label "bug")

ISSUE_NUM=$(echo $ISSUE_URL | grep -o '[0-9]*$')
echo "✅ 이슈 #$ISSUE_NUM 생성: $ISSUE_URL"

# 2. 브랜치 및 worktree 생성
BRANCH="fix/issue-${ISSUE_NUM}-$(echo $ISSUE_TITLE | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
WORKTREE_PATH="../worktree-issue-${ISSUE_NUM}"

echo "🌿 Worktree 생성: $WORKTREE_PATH"
git worktree add -b "$BRANCH" "$WORKTREE_PATH" "$BASE_BRANCH"

echo ""
echo "👉 다음 단계:"
echo "  cd $WORKTREE_PATH"
echo "  # 코드 수정 후:"
echo "  bash scripts/create-pr.sh $ISSUE_NUM"
```

---

## 3. PR 생성 스크립트 (`scripts/create-pr.sh`)

```bash
#!/bin/bash
ISSUE_NUM="$1"

# 변경사항 커밋
git add -A
git commit -m "fix: #${ISSUE_NUM} $(gh issue view $ISSUE_NUM --json title -q .title)"

# 브랜치 push
git push origin HEAD

# PR 생성 (이슈 자동 연결)
gh pr create \
  --title "fix: #${ISSUE_NUM} $(gh issue view $ISSUE_NUM --json title -q .title)" \
  --body "## 변경 사항
$(git log origin/main..HEAD --oneline)

## 관련 이슈
Closes #${ISSUE_NUM}" \
  --draft=false
```

---

## 4. Claude Code에서 사용하는 방법

### 방법 A: 슬래시 커맨드로 자동화

Claude Code 터미널에서 직접 지시:

```
# 이슈 등록부터 PR까지 한번에
"GitHub에 '로그인 버그' 이슈 등록하고, worktree 만들어서
 auth.js 파일의 토큰 검증 로직 수정 후 PR까지 올려줘"
```

### 방법 B: `.claude/commands/` 커스텀 커맨드

```markdown
# .claude/commands/fix-issue.md

GitHub 이슈를 처리하는 전체 워크플로우:

1. `gh issue create --title "$ISSUE_TITLE" --body "$ISSUE_BODY"`로 이슈 등록
2. 이슈 번호 추출
3. `git worktree add -b fix/issue-$NUM ../worktree-$NUM main` 실행
4. worktree 디렉토리로 이동하여 코드 수정
5. 수정 완료 후 커밋 및 `gh pr create`로 PR 생성
6. worktree 정리: `git worktree remove ../worktree-$NUM`
```

---

## 5. 전체 자동 실행 예시

Claude Code에 이렇게 요청하면 됩니다:

```
"scripts/auto-workflow.sh를 사용해서:
- 이슈: '결제 모듈 NullPointerException'
- payment.js 237번째 줄 null 체크 추가
- 테스트 통과 확인 후 PR 생성"
```

Claude Code가 순서대로:

1. `gh issue create` → 이슈 #42 생성
2. `git worktree add ../worktree-42 -b fix/issue-42`
3. `worktree-42/`에서 코드 수정
4. `gh pr create --body "Closes #42"` → PR 자동 연결

---

## 핵심 팁

|항목|명령어|
|---|---|
|gh 인증|`gh auth login`|
|worktree 목록|`git worktree list`|
|worktree 정리|`git worktree remove <경로>`|
|이슈→PR 자동 닫기|PR 본문에 `Closes #번호` 포함|

`gh` CLI가 설치되어 있고 GitHub 인증이 완료되어 있어야 합니다. 추가로 궁금한 부분 있으면 말씀해주세요!