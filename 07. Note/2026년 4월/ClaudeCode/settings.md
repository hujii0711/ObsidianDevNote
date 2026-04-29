
#### 1. 설정 파일

| 로컬 프로젝트 설정 | /.claude/settings.local.json |
| ---------- | ---------------------------- |
| 공유 프로젝트 설정 | /.claude/settings.json       |
| 사용자 설정     | ~/.claude/settings.json      |
| mcp 설정     | .mcp.json                    |
| 엔터프라이즈 정책  | managed-settings.json        |

#### 2. md 파일

|               | project(~)/.claude/CLAUDE.md                                                    |
| ------------- | ------------------------------------------------------------------------------- |
|               | project/.claude/docs/API_SCHEME.md                                              |
|               | project/.claude/docs/ER_DIAGRAM.md                                              |
|               | project/.claude/docs/SCREEN_DESIGN.md                                           |
|               | project/.claude/docs/TEST_DEFINITION.md                                         |
| 서브 에이전트 특정 역할 | project(~)/.claude/<font color="#ff0000">agents</font>/BIZ.md                   |
| 커스텀 슬래시 명령    | project(~)/.claude/<font color="#ff0000">commands</font>/CUSTOM_HOOK_COMMAND.md |
| 스킬 설정 파일      | project(~)/.claude/<font color="#ff0000">skills</font>/SKILL.md                 |

#### 3. settings.json
- 확인 요청없이 자동으로 파일 수정할 수 있는 특정 명령어(ex. mv) 지정할 수 있다.