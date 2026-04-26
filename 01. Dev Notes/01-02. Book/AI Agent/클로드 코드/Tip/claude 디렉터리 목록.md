
## `.claude/` 디렉토리 전체 파일 목록

| 파일/디렉토리               | git 커밋 | 역할                                  |
| --------------------- | ------ | ----------------------------------- |
| `CLAUDE.md`           | ✓      | 매 세션 로드되는 핵심 지침                     |
| `rules/*.md`          | ✓      | 토픽별 규칙, path 스코프 가능                 |
| `settings.json`       | ✓      | 권한, hooks, 환경변수, 모델 기본값             |
| `settings.local.json` | ✗      | 개인 설정 오버라이드, 자동 gitignore           |
| `.mcp.json`           | ✓      | 팀 공유 MCP 서버 설정                      |
| `.worktreeinclude`    | ✓      | 새 worktree 생성 시 복사할 gitignore 파일 목록 |
| `skills/<n>/SKILL.md` | ✓      | `/name`으로 호출하거나 자동 호출되는 재사용 프롬프트    |
| `commands/*.md`       | ✓      | 단일 파일 프롬프트 (skills와 동일 메커니즘)        |
| `output-styles/*.md`  | ✓      | 커스텀 시스템 프롬프트 섹션                     |
| `agents/*.md`         | ✓      | 자체 프롬프트와 툴을 가진 서브에이전트 정의            |
| `agent-memory/<n>/`   | ✓      | 서브에이전트의 persistent memory           |

질문하신 4가지 외에 추가로 있는 것들은 **`settings.json`, `settings.local.json`, `.mcp.json`, `.worktreeinclude`, `output-styles/`, `agent-memory/`** 입니다.

---

### 주목할 만한 것들

**`output-styles/*.md`** — 처음 보셨을 수도 있는데, Claude의 응답 스타일(말투, 포맷 등)을 커스터마이즈하는 시스템 프롬프트 섹션입니다.

**`.worktreeinclude`** — git worktree를 많이 쓰는 경우 유용합니다. `.gitignore`된 파일(`.env` 등)을 새 worktree에 자동 복사해줍니다.

**`agent-memory/<n>/`** — 서브에이전트별 독립적인 메모리 디렉토리로, 에이전트가 세션 간 학습 내용을 유지합니다.

---

### 전체 구조 예시

```
.claude/
├── CLAUDE.md
├── settings.json
├── settings.local.json      # gitignore
├── .mcp.json
├── .worktreeinclude
├── rules/
│   ├── security.md
│   └── testing.md
├── skills/
│   └── deploy/
│       └── SKILL.md
├── commands/
│   └── pr-review.md
├── output-styles/
│   └── concise.md
├── agents/
│   └── reviewer.md
└── agent-memory/
    └── reviewer/
```

현재 세션에서 실제로 무엇이 로드됐는지 확인하려면 `/context`, `/memory`, `/agents`, `/skills`, `/hooks`, `/mcp` 명령어로 확인할 수 있습니다.

---
### .claude/docs/

`.claude/docs/`는 **공식적으로 문서화된 디렉토리가 아닙니다.** 그냥 마크다운 파일을 담는 폴더일 뿐이지만, 패턴 자체는 강력합니다.

---

### 어떻게 활용하는 패턴인가?

**Skills가 on-demand로 읽어오는 레퍼런스 문서 저장소**로 사용합니다.

예를 들어 `SKILL.md` 안에서 `@` import로 참조합니다:

```markdown
# .claude/skills/deploy/SKILL.md
배포 전 반드시 아래 문서를 참고하세요:
@../../docs/deploy-checklist.md
@../../docs/env-variables.md
```

```
.claude/
├── skills/
│   └── deploy/
│       └── SKILL.md        # "배포할 때 docs/deploy-checklist.md 참고"
└── docs/                   # ← 공식 디렉토리 아님, 커뮤니티 관행
    ├── deploy-checklist.md
    ├── architecture.md
    └── api-conventions.md
```

---

### CLAUDE.md에 직접 넣지 않는 이유

CLAUDE.md는 **매 세션 항상** 컨텍스트에 로드됩니다. 길고 상세한 레퍼런스 문서를 직접 넣으면 토큰 낭비입니다.

`.claude/docs/`에 두고 skill이 필요할 때만 `@` import로 끌어오면 **필요한 순간에만** 컨텍스트에 올라옵니다.

---

### 정리

||공식 여부|역할|
|---|---|---|
|`.claude/rules/`|✅ 공식|항상 또는 path 조건으로 로드되는 규칙|
|`.claude/docs/`|❌ 비공식|skill이 on-demand로 참조하는 레퍼런스 문서 모음|

결국 **명명 규칙은 자유**이고, `docs` 대신 `references`, `context`, `knowledge` 등 어떤 이름을 써도 동작은 같습니다. 커뮤니티에서 `.claude/docs/`라는 이름이 관행으로 자리잡은 것입니다.

---
###### .claude/commands, .claude/skills/, .claude/rules/, .claude/agents은 다른 이름으로 사용할 수 없는 공식 명칭인가?

네, **공식 명칭이고 변경 불가**입니다.

Claude Code가 이 경로를 하드코딩으로 인식하기 때문에 다른 이름을 쓰면 인식하지 못합니다.

---

### 공식 고정 경로 (변경 불가)

|디렉토리|이유|
|---|---|
|`.claude/commands/`|Claude Code가 `/명령어`로 자동 등록하는 경로|
|`.claude/skills/`|skill 자동 검색 경로|
|`.claude/rules/`|규칙 자동 로드 경로|
|`.claude/agents/`|서브에이전트 자동 등록 경로|

---

### `.claude/docs/`가 자유로운 이유

반면 `.claude/docs/`는 Claude Code가 **전혀 인식하지 않는** 경로입니다. 그냥 파일 시스템상의 폴더일 뿐이라 이름이 자유롭고, skill의 `@` import나 CLAUDE.md의 `@` import로 **명시적으로 참조**해야만 동작합니다.

```markdown
# .claude/skills/deploy/SKILL.md
@../../docs/checklist.md      ← 이렇게 직접 경로를 지정해야 함
@../../references/checklist.md  ← 폴더 이름이 뭐든 상관없음
@../../whatever/checklist.md    ← 동일하게 동작
```

---

### 한 줄 요약

> Claude Code가 **자동으로 스캔하는 경로**는 공식 명칭 고정, **사람이 명시적으로 참조하는 경로**는 이름 자유.