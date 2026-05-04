
- `.claude/` 폴더는 **Claude Code의 핵심 제어 디렉터리**로, 프로젝트별 규칙·명령·권한·메모리 상태를 관리함
- `CLAUDE.md`는 Claude의 **행동 원칙과 프로젝트 규칙을 정의하는 중심 파일**로, 여러 계층의 설정을 병합해 적용함
- `commands/`, `skills/`, `agents/` 폴더는 각각 **사용자 정의 명령, 자동 워크플로, 전문 서브에이전트**를 구성해 협업 효율을 높임
- `settings.json`은 **명령 실행 권한과 파일 접근 범위**를 제어하며, `settings.local.json`으로 개인별 오버라이드가 가능함
- 전체 구조는 Claude에게 **프로젝트의 정체성과 규칙을 전달하는 프로토콜**로 작동해, 명확한 설정이 생산성과 협업 효율을 극대화함

---

## .claude/ 폴더 구조와 구성 요소

- `.claude/` 폴더는 **Claude Code의 동작을 제어하는 핵심 디렉터리**로, 프로젝트별 규칙·명령·권한·메모리 상태를 관리
- 프로젝트 루트의 폴더는 **팀 단위 설정**을 포함하며 Git에 커밋됨
- 홈 디렉터리(`~/.claude/`)의 폴더는 **개인 설정과 세션 기록**을 저장하며, 자동 메모리 및 개인 명령을 포함

- ### CLAUDE.md — Claude의 지침서
    
    - Claude Code 세션 시작 시 가장 먼저 읽는 파일로, **Claude의 행동 원칙과 프로젝트 규칙을 정의**
    - 프로젝트 루트의 `CLAUDE.md`는 팀 공통 규칙, `~/.claude/CLAUDE.md`는 **전역 개인 규칙**, 하위 폴더의 `CLAUDE.md`는 **폴더별 규칙** 담당
    - Claude는 여러 `CLAUDE.md` 파일을 **병합하여 적용**
    - 권장 내용은 빌드·테스트 명령, 주요 아키텍처 결정, 비직관적 제약사항, 네이밍·에러 처리 규칙 등
    - **200줄 이하 유지**가 권장되며, 과도한 길이는 Claude의 지침 준수율을 저하시킴
    
- ### CLAUDE.local.md — 개인별 오버라이드
    
    - 팀 공통 규칙과 별도로 **개인 선호를 반영**할 수 있는 파일
    - 프로젝트 루트에 `CLAUDE.local.md`를 생성하면 Claude가 이를 함께 읽음
    - `.gitignore`에 자동 포함되어 **저장소에 커밋되지 않음**
    
- ### rules/ 폴더 — 모듈형 규칙 관리
    
    - `CLAUDE.md`가 커질 경우 `.claude/rules/` 폴더로 분리해 관리
    - 각 규칙 파일은 **주제별로 분리**되어 유지보수가 용이
        - 예: `code-style.md`, `testing.md`, `api-conventions.md`, `security.md`
    - YAML 프런트매터의 `paths` 필드를 사용하면 **특정 경로에만 적용되는 규칙** 지정 가능
        - 예: `src/api/**/*.ts` 경로에만 API 규칙 적용
    - 경로 지정이 없는 규칙은 모든 세션에 항상 로드됨
    
- ### commands/ 폴더 — 사용자 정의 슬래시 명령
    
    - `.claude/commands/` 폴더의 각 Markdown 파일은 **슬래시 명령(/)** 으로 등록됨
        - 예: `review.md` → `/project:review`, `fix-issue.md` → `/project:fix-issue`
    - `!` 백틱 구문으로 **셸 명령 실행 결과를 Claude 프롬프트에 삽입** 가능
        - 예: `!git diff main...HEAD`
    - `$ARGUMENTS` 변수를 사용해 명령 실행 시 **인자 전달** 가능
        - 예: `/project:fix-issue 234` → GitHub 이슈 234 내용 자동 로드
    - 프로젝트 명령은 팀과 공유되며, 개인 명령은 `~/.claude/commands/`에 저장되어 **모든 프로젝트에서 사용 가능**
    
- ### skills/ 폴더 — 자동 실행 워크플로
    
    - **명령과 유사하지만 자동으로 트리거되는 워크플로**로 작동
    - Claude가 대화 내용을 분석해 **적절한 상황에서 자동 실행**
    - 각 스킬은 하위 폴더의 `SKILL.md` 파일로 정의되며, YAML 프런트매터로 **트리거 조건과 허용 도구** 지정
        - 예: `security-review` 스킬은 보안 관련 대화 시 자동 실행
    - 스킬 폴더에는 `DETAILED_GUIDE.md` 등 **보조 문서나 템플릿 파일** 포함 가능
    - 개인 스킬은 `~/.claude/skills/`에 저장되어 전역적으로 사용 가능
    
- ### agents/ 폴더 — 전문 서브에이전트
    
    - `.claude/agents/` 폴더에는 **특정 역할을 수행하는 서브에이전트(persona)** 정의
    - 각 에이전트는 독립된 시스템 프롬프트, 모델, 도구 접근 권한을 가짐
        - 예: `code-reviewer.md`, `security-auditor.md`
    - `tools` 필드로 접근 가능한 도구를 제한해 **보안 및 역할 분리** 실현
    - `model` 필드로 작업에 맞는 Claude 모델(예: Haiku, Sonnet, Opus) 선택 가능
    - Claude는 필요 시 해당 에이전트를 **별도 컨텍스트에서 실행**해 결과만 요약 보고
    
- ### settings.json — 권한 및 프로젝트 설정
    
    - `.claude/settings.json`은 Claude의 **명령 실행 권한과 파일 접근 범위**를 정의
    - `$schema` 필드는 VS Code 등에서 **자동 완성과 유효성 검사** 지원
    - `allow` 목록은 **자동 승인 명령**, `deny` 목록은 **완전 차단 명령** 지정
        - 예: 허용 — `Bash(npm run *)`, `Read`, `Write`, `Edit`
        - 차단 — `Bash(rm -rf *)`, `Bash(curl *)`, `.env` 파일 읽기
    - 목록에 없는 명령은 실행 전 **사용자 확인 요청**
    - 개인별 권한 변경은 `.claude/settings.local.json`에 저장되며 Git에 포함되지 않음
    
- ### ~/.claude/ 폴더 — 전역 설정 및 메모리
    
    - `~/.claude/CLAUDE.md`는 **모든 프로젝트에 공통 적용되는 개인 지침**
    - `~/.claude/projects/`는 **프로젝트별 세션 기록과 자동 메모리** 저장
        - Claude가 학습한 명령, 패턴, 구조적 통찰을 유지
        - `/memory` 명령으로 조회 및 수정 가능
    - `~/.claude/commands/`, `~/.claude/skills/`, `~/.claude/agents/`는 **전역 개인 명령·스킬·에이전트** 저장소
    
- ### 전체 구조 예시
    
    ```objectivec
    your-project/  
    ├── CLAUDE.md  
    ├── CLAUDE.local.md  
    └── .claude/  
        ├── settings.json  
        ├── settings.local.json  
        ├── commands/  
        ├── rules/  
        ├── skills/  
        └── agents/  
    ~/.claude/  
    ├── CLAUDE.md  
    ├── settings.json  
    ├── commands/  
    ├── skills/  
    ├── agents/  
    └── projects/  
    ```
    
- ### 초기 설정 단계
    
    - **1단계:** `/init` 명령으로 기본 `CLAUDE.md` 생성 후 핵심 내용만 남김
    - **2단계:** `.claude/settings.json` 작성, 실행 허용·차단 규칙 정의
    - #### 3단계:**자주 사용하는 워크플로(예: 코드 리뷰, 이슈 수정)에 맞는**명령 추가
        
        - **4단계:** `CLAUDE.md`가 커지면 `.claude/rules/`로 분리
        - **5단계:** `~/.claude/CLAUDE.md`에 개인 선호 규칙 추가

## 핵심 인사이트

- `.claude/` 폴더는 **Claude에게 프로젝트의 정체성과 규칙을 전달하는 프로토콜**
- `CLAUDE.md`가 가장 중요한 파일이며, 이를 명확히 정의할수록 **Claude의 생산성이 극대화**
- 나머지 구성요소는 이를 보완하는 최적화 계층으로, 점진적으로 확장 가능
- 명확한 설정은 **Claude의 수정 요청 감소와 효율적 협업**으로 이어짐

## 추가 논의

- `settings.json`의 **deny 리스트는 인간 사용 시 안전하지만**, 에이전트 모드에서는 Bash 접근으로 인해 추가 보호 필요
- OneCLI는 네트워크 수준에서 **자격 증명 토큰을 대체하는 프록시 계층**을 제공해 비밀정보 노출을 방지
- 향후 **에이전트 모드 전용 .claude 설정**(규칙·권한·스킬 분리)의 필요성 제기
- 최신 문서에 따르면 **명령(commands)** 과 **스킬(skills)** 이 통합되어, `.claude/commands/deploy.md`와 `.claude/skills/deploy/SKILL.md`가 동일하게 `/deploy` 명령을 생성하며, 스킬은 추가 기능(보조 파일, 자동 트리거 등)을 지원
