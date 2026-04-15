
#### 1. 설정 파일

| 로컬 프로젝트 설정 | /.claude/settings.local.json |
| ---------- | ---------------------------- |
| 공유 프로젝트 설정 | /.claude/settings.json       |
| 사용자 설정     | ~/.claude/settings.json      |
|            | .mcp.json                    |

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

#### 3. /project/.claude/CLAUDE.md
- 모델을 구성할 수 있다.
- 시스템 전체의 정의 파일처럼 작동한다.
- 프로젝트의 구조, 사용 기술, 코딩 규약 등 클로드 코드가 참조할 정보를 담는 역할을 한다.
- 기술 명세 예시, 필요한 사양, 기술 스택, 코딩 규약
- 클로드 코드의 메모리 역할을 하며 대화에서 계속 참조된다.
- 프로젝트 루트뿐만 아니라 하위 디렉터리에도 CLAUDE.md를 배치할 수 있다. (폴더 구조에 따라 재귀적으로 읽어 컨테스트에 반영한다.)
- 모든 클로드 코드 실행에 공통 규칙을 적용하려면, 홈 디렉터리 아래 .claude 폴더에 CLAUDE.md를 추가한다.
- CLAUDE.md 파일 내에서 @ 표기를 사용해 다른 파일을 참조할 수 있다. (자세한 명세는 @docs/TEST.md를 참고합니다.)
- 즉시 기억 내용을 추가하고 싶을 때는 '#' 명령으로 추가할 수 있다.
- 설계의 기본적인 내용은 최대한 고정하고, CLAUDE.md에 기록해 두는 것이 일관성을 유지하는데 도움이 된다.

### 4. settings.json
- 확인 요청없이 자동으로 파일 수정할 수 있는 특정 명령어(ex. mv) 지정할 수 있다.