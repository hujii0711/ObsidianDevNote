
## `.claude/memory`의 실제 위치

Auto memory(자동 메모리)의 저장 위치는 `.claude/memory/`가 **아닙니다.** 실제 저장 위치는 `~/.claude/projects/<project-hash>/memory/`입니다. 즉 프로젝트 루트가 아닌 **사용자 홈 디렉토리** 아래에 저장됩니다.

---

## Auto memory란?

CLAUDE.md는 **개발자가 직접 작성**하는 지침이고, Auto memory는 **Claude가 스스로 작성**하는 학습 내용입니다. 빌드 명령어, 디버깅 인사이트, 코드 스타일 취향 등 Claude가 작업하면서 발견한 패턴을 저장합니다.

||CLAUDE.md|Auto memory|
|---|---|---|
|작성자|개발자|Claude 자동|
|내용|규칙·지침|학습·패턴|
|위치|프로젝트 루트|`~/.claude/projects/<hash>/memory/`|
|로드|전체|최대 200줄 또는 25KB|

---

## Agent memory는 다름

서브에이전트 전용 메모리는 별도입니다. `memory: project`로 설정된 에이전트는 `.claude/agent-memory/<agent-name>/`에 저장되어 팀과 공유되고, `memory: local`은 `.claude/agent-memory-local/<agent-name>/`에 저장되어 gitignore됩니다.

---

## 정리

- `.claude/memory/` — **공식 디렉토리 아님**, 존재하지 않음
- `~/.claude/projects/<hash>/memory/` — Auto memory 실제 저장 위치 (공식)
- `.claude/agent-memory/` — 서브에이전트 메모리 (공식)

`/memory` 명령어로 현재 저장된 auto memory를 확인하고 직접 편집할 수 있습니다.

---
## `/memory` 명령어가 유용한 상황

### 1. Auto memory 내용 감사 (가장 핵심)

많은 개발자들이 auto memory를 켜두되, 주기적으로 `/memory`를 열어 잘못 저장된 내용을 수정하거나 삭제하는 방식을 택합니다.

Claude가 자동으로 저장한 내용이 **틀렸거나 오래됐을 때** 직접 편집해서 바로잡을 수 있습니다.

---

### 2. 중요한 내용을 명시적으로 저장 요청

"always use pnpm, not npm"이나 "API 테스트는 로컬 Redis 인스턴스가 필요해"처럼 Claude에게 기억하라고 말하면 auto memory에 저장됩니다.

어렵게 찾아낸 버그 해결법이나 중요한 아키텍처 결정을 Claude가 놓쳤을 때, 직접 "remember that..."으로 저장을 지시한 뒤 `/memory`로 확인하는 흐름입니다.

---

### 3. auto memory 토글 on/off

CI 파이프라인이나 자동화 환경에서는 `/memory`에서 auto memory 토글을 끄거나 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` 환경변수로 비활성화합니다.

---

### 4. memory 타입별 관리

memory 파일은 frontmatter의 type으로 구분됩니다. `user`(사용자 성향), `feedback`(교정 내용), `project`(진행 중인 결정), `reference`(외부 리소스 포인터) 등으로 나뉘며, type이 언제 사용될지를 결정합니다.

---

### 실제 활용 흐름 예시

```
1. 세션 중 Claude가 npm을 계속 쓰자 "use pnpm always" 교정
   → Claude가 auto memory에 저장

2. 며칠 후 /memory 열어서 확인
   → 맞게 저장됐는지 검토

3. 오래된 항목 발견 (예: 이미 삭제된 환경변수)
   → 직접 편집해서 제거

4. CLAUDE.md로 승격시킬 내용 발견
   → memory에서 꺼내 CLAUDE.md에 정식으로 추가
```

---

### CLAUDE.md vs auto memory 구분 기준

CLAUDE.md는 프로젝트 자체를 설명(스택, 컨벤션, 배포)하고, memory는 작업 관계를 설명(취향, 진행 중인 결정, 비직관적인 제약)합니다.

결국 `/memory`는 **"Claude가 스스로 배운 것들을 사람이 검토하고 교정하는 창구"**입니다.