
> 1. **YAML Frontmatter** (에이전트가 언제 이 스킬을 쓸지 결정하는 인덱스 정보)
> 2. **Markdown Body** (에이전트가 수행해야 할 상세한 워크플로우 지침)

## YAML frontmatter 지원 필드 전체 목록

필수 필드는 `name`과 `description` 두 가지뿐이며, 나머지는 모두 선택입니다.

### 필수 필드

```yaml
name: security-reviewer        # 소문자 + 하이픈. 고유해야 함 (필수)
description: |                 # Claude가 위임 여부 결정에 사용 (필수)
  Use proactively after any code changes.
  Checks for security vulnerabilities and OWASP issues.
```

---

### 도구 제어

```yaml
tools: Read, Grep, Glob, Bash  # 허용 툴 목록. 생략 시 부모 세션 전체 상속
disallowedTools: Write, Edit   # 거부 툴 목록. tools와 충돌 시 이쪽이 우선
```

두 필드가 모두 설정된 경우 `disallowedTools`가 우선합니다. 양쪽에 모두 있는 툴은 제거됩니다.

---

### 모델 및 성능

```yaml
model: sonnet         # haiku | sonnet | opus | inherit | 전체 모델 ID
effort: medium        # low | medium | high | max  (max는 세션 전용)
maxTurns: 20          # 서브에이전트가 실행할 수 있는 최대 턴 수
```

`effort`는 Sonnet 4.6, Opus 4.6 같은 지원 모델에서 추론 깊이를 제어하며, `max`는 세션 전용 옵션입니다. `inherit`를 사용하면 부모 세션의 모델을 그대로 사용합니다.

---

### 권한 모드

```yaml
permissionMode: default
# default | acceptEdits | dontAsk | bypassPermissions | plan
```

각 값의 의미: `default` - 표준 권한 게이트 적용, `acceptEdits` - 파일 편집 자동 수락, `dontAsk` - 허용된 작업은 묻지 않음, `bypassPermissions` - 모든 권한 우회 (주의 필요), `plan` - 플랜 모드.

---

### 컨텍스트 및 메모리

```yaml
skills:               # 시작 시 주입할 스킬 목록 (전문 지식 사전 로딩)
  - security-audit
  - code-quality

memory: project       # user | project | local — 세션 간 지속 메모리
                      # user: ~/.claude/agent-memory/
                      # project: .claude/agent-memory/
```

---

### 실행 방식

```yaml
background: false     # true = 백그라운드 병렬 실행 (권한 사전 승인 필요)
isolation: worktree   # worktree = 임시 git worktree에서 독립 실행
initialPrompt: "Run git diff and review changes."  # 첫 턴 자동 제출 프롬프트
```

`cd`는 서브에이전트 내에서 Bash 호출 간 유지되지 않으므로, 진짜 작업 디렉토리 격리가 필요하면 `isolation: worktree`를 사용하세요.

---

### 외부 연동

```yaml
mcpServers:           # 이 서브에이전트에만 적용되는 MCP 서버
  - github
  - linear

hooks:                # 서브에이전트 라이프사이클에 스코프된 훅
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh"
```

훅은 서브에이전트의 라이프사이클에 스코프되어, 해당 서브에이전트가 활성화된 동안만 실행되고 완료되면 정리됩니다. `Stop` 훅은 자동으로 `SubagentStop` 이벤트로 변환됩니다.

---

### UI

```yaml
color: cyan           # UI에서 서브에이전트를 구별하기 위한 배경색
```

---

### 전체 예시

```yaml
---
name: security-reviewer
description: >
  Use proactively after any code changes.
  Scans for security vulnerabilities, OWASP issues, and secret leaks.
tools: Read, Grep, Glob
disallowedTools: Write, Edit, Bash
model: opus
effort: high
maxTurns: 30
permissionMode: default
skills:
  - security-audit
memory: project
background: false
isolation: worktree
color: red
---

You are a senior security engineer. Review code for...
```

> **참고:** 플러그인 서브에이전트는 보안상의 이유로 `hooks`, `mcpServers`, `permissionMode` 세 필드를 지원하지 않습니다. 이 필드가 필요하면 `.claude/agents/`에 직접 파일을 복사해서 사용하세요.