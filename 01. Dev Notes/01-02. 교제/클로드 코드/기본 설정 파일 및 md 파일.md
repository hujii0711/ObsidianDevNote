
## 클로드 코드

#### 1. 설정 파일

| 로컬 프로젝트 설정 | /.claude/settings.local.json |
| ---------- | ---------------------------- |
| 공유 프로젝트 설정 | /.claude/settings.json       |
| 사용자 설정     | ~/.claude/settings.json      |

#### 2. md 파일

|     | /project/.claude/CLAUDE.md                       |
| --- | ------------------------------------------------ |
|     | /project/.claude/docs/API_SCHEME.md              |
|     | /project/.claude/docs/ER_DIAGRAM.md              |
|     | /project/.claude/docs/SCREEN_DESIGN.md           |
|     | /project/.claude/docs/TEST_DEFINITION.md         |
|     | /project/.claude/agents/BIZ.md                   |
|     | /project/.claude/commands/CUSTOM_HOOK_COMMAND.md |
|     | /project/.claude/skills/SKILL.md                 |
@docs/SCREEN_DESIGN.md: md 파일 참조

---

##  오픈 클로

#### 1. 설정 파일

|     | ~/.openclaw/openclaw.db   |     |
| --- | ------------------------- | --- |
|     | ~/.openclaw/openclaw.json |     |
|     | ~/.openclaw/.env          |     |

#### 2. md 파일

|     | ~/.openclaw/workspace/SOUL.md             | 행동 철학, 말투, 원칙, 금지 사항    |
| --- | ----------------------------------------- | ----------------------- |
|     | ~/.openclaw/workspace/IDENTIFY.md         | 이름, 이모지, 역할 한 줄 소개      |
|     | ~/.openclaw/workspace/USER.md             | 사용자 정보(이름, 호칭, 타임존, 선호) |
|     | ~/.openclaw/workspace/skills/biz/SKILL.md |                         |
|     | ~/.openclaw/workspace/AGENT.md            |                         |
|     | ~/.openclaw/memory/MEMORY.md              |                         |
#### 3. 명령어
`openclaw onboard`
`openclaw --version`
`openclaw models set google/gemini-3-flash-preview`
`openclaw gateway restart`
`openclaw skills list`
