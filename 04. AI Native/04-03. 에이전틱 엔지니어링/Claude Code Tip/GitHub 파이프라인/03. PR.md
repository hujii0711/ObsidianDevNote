
### 1. PR 생성

```bash
gh pr create \
  --title "feat: 로그인 기능 구현" \
  --body "Closes #42" \
  --reviewer "리뷰어_github_아이디"
```

> 유용한 추가 옵션

상황에 따라 아래 옵션들을 덧붙이면 더욱 정교한 PR 요청이 가능합니다.

| **옵션**                | **설명**                                |
| --------------------- | ------------------------------------- |
| `--web`               | PR 생성 직후 브라우저에서 확인하고 싶을 때 사용          |
| `--draft`             | 아직 작업 중인 상태로 PR을 올리고 싶을 때 (Draft PR)  |
| `--base <branch>`     | 기본값(`main` 등)이 아닌 다른 브랜치로 머지 요청을 보낼 때 |
| `--reviewer <handle>` | 특정 팀원을 리뷰어로 지정할 때                     |
| `--assignee <handle>` | 담당자를 지정할 때 (보통 `@me`)                 |

### 2. PR 확인

```bash
# PR 목록 확인
gh pr list

# PR 상세 확인
gh pr view 1

# PR 상태 확인 (CI 통과 여부)
gh pr checks 1

# 현재 생성된 PR 웹으로 열기
gh pr view --web
```


### 3. 코드 리뷰 & 승인

```bash
# 리뷰어 입장에서 코드 확인
gh pr diff 1

# 승인
gh pr review 1 --approve --body "LGTM"

# 변경 요청
gh pr review 1 --request-changes --body "수정 필요한 내용"

# 코멘트만
gh pr review 1 --comment --body "확인했습니다"
```


### 4. PR 머지

```bash
# Squash 머지 (커밋 하나로 합쳐서 머지)
gh pr merge 1 --squash --delete-branch

# Merge commit
gh pr merge 1 --merge --delete-branch

# Rebase 머지
gh pr merge 1 --rebase --delete-branch
```

`--delete-branch`를 붙이면 머지 후 원격 브랜치가 자동 삭제됩니다.

#### ★ --body Close가 있는 것과 없는 것의 차이
GitHub에서 PR이나 커밋 메시지에 `Closes #42` (또는 `Fixes`, `Resolves` 등)와 같은 키워드를 사용하는 것은 **이슈 자동 연결 및 종료(Linked Issues)** 기능을 활용하기 위함입니다.
이 문구가 포함되었을 때와 없을 때의 차이점은 크게 세 가지 측면에서 나타납니다.

##### (1) 이슈의 자동 종료 (Automation)

가장 큰 기술적 차이점입니다.
- **문구가 있을 때 (`Closes #42`):** PR이 메인 브랜치에 **머지(Merge)되는 순간**, 해당 번호(#42)의 이슈가 자동으로 **Closed** 상태로 바뀝니다. 사람이 일일이 이슈를 찾아가서 닫을 필요가 없습니다.
    
- **문구가 없을 때:** PR이 머지되어도 이슈는 여전히 **Open** 상태로 남습니다. 개발자가 직접 이슈 페이지에 들어가서 수동으로 닫아줘야 합니다.
##### (2) UI 상의 시각적 연결 (Visual Linking)
GitHub 웹 화면에서의 표시 방식이 달라집니다.
- **문구가 있을 때:**
    - PR 우측 사이드바의 **Development** 섹션에 해당 이슈가 자동으로 링크됩니다.
    - 이슈 페이지에서도 "이 이슈는 이 PR에 의해 해결될 예정입니다"라는 상태가 표시됩니다.
    
- **문구가 없을 때:** PR과 이슈 사이에 명확한 관계가 표시되지 않습니다. 동료 개발자들이 "이 PR이 어떤 이슈를 해결하려는 건지" 한눈에 파악하기 어렵습니다.
##### (3) 프로젝트 관리 및 트래킹
워크플로우 효율성 측면에서의 차이입니다.
- **문구가 있을 때:** 프로젝트 보드(Project Board)를 사용 중이라면, PR 머지와 동시에 이슈가 'Done' 컬럼으로 자동 이동하도록 자동화 설정을 할 수 있습니다.
    
- **문구가 없을 때:** 모든 히스토리를 수동으로 관리해야 하므로, 규모가 큰 프로젝트에서는 해결된 이슈가 계속 Open으로 방치되는 '이슈 누락' 현상이 발생하기 쉽습니다.

##### 💡 요약 및 권장 사항

| **구분**    | **Closes #42 포함 시** | **미포함 시**            |
| --------- | ------------------- | -------------------- |
| **이슈 상태** | 머지 시 **자동 종료**      | 머지 후에도 **Open 유지**   |
| **연결성**   | PR-이슈 간 상호 링크 생성    | 링크 없음 (단순 텍스트 언급만 됨) |
| **효율성**   | 자동화로 생산성 향상         | 수동 관리 필요             |

**권장하는 사용법:**
현재 `gh` CLI로 PR을 날리실 때, 특정 이슈를 해결하기 위한 작업이라면 **반드시 포함**하는 것이 좋습니다. 만약 단순히 이슈를 언급만 하고 이슈를 닫고 싶지는 않다면 `Closes` 대신 `Ref #42` 또는 그냥 `#42`라고만 적으시면 자동 종료 없이 링크만 생성됩니다.

**사용 가능한 키워드:**
`close`, `closes`, `closed`, `fix`, `fixes`, `fixed`, `resolve`, `resolves`, `resolved` 모두 동일하게 자동 종료 기능을 수행합니다. 편하신 단어를 선택해서 사용하세요!

##### 💡 여러 이슈 동시에 닫기

```bash
gh pr create \
  --title "feat: 로그인 구현" \
  --body "Closes #42, Closes #43, Fixes #50"
```

##### 주의사항
`Closes #이슈번호` 키워드가 없으면 PR 머지 후에도 이슈가 **자동으로 닫히지 않습니다.** 이슈를 수동으로 따로 닫아야 합니다.


### 5. 마무리 정리

```bash
# 메인으로 복귀 후 최신화
cd /path/to/main-repo
git checkout main
git pull origin main

# 로컬 워크트리 제거
git worktree remove .claude/worktrees/feature-42

# 로컬 브랜치 제거
git branch -d feature/#42-login

# 이슈 자동 닫힘 확인
gh issue view 42
```

---
### 🎯 전체 흐름 요약

```
gh issue create        # 이슈 등록
  ↓
git worktree add       # 브랜치 + 워크트리 생성
  ↓
git commit & push      # 작업 후 push
  ↓
gh pr create           # PR 생성 (Closes #이슈번호)
  ↓
gh pr review --approve # 리뷰 & 승인
  ↓
gh pr merge            # 머지 → 이슈 자동 닫힘
  ↓
git worktree remove    # 워크트리 정리
```
