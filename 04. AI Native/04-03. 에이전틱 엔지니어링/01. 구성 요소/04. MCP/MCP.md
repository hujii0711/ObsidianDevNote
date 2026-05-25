
### 1. MCP 서버란

MCP 서버란 클로드 코드와 다른 애플리케이션 간 상호 작용을 중개하는 서버이다.  
MCP 서버를 구축하고 이를 통해 데이터를 주고 받아 LLM이 특정 정보를 획득하거나 애플리케이션을 제어할 수 있게 된다. 구체적으로 Git 작업, 데이터베이스 작업, 노션과의 연동, API 연동 등 다양한 작업이 가능하며, 클로드 코드의 기능을 대폭 확장할 수 있다.

### 2. 스코프 설정하기

- local  
    기본 설정이며, 현재 사용자만 클로드 코드를 실행한 디렉터리 하위에서만 사용한다.
    
- project  
    프로젝트 단위에서 MCP를 사용하고 설정을 공유할 때 사용한다.  
    .mcp.json에 설정이 기재되어, Git에서 관리될 경우 프로젝트의 전체 사용자에서 MCP 설정이 적용된다.
    
- user  
    사용자 홈 디렉터리의 ~/.claude에 설정이 저장되므로 해당 사용자의 모든 프로젝트에 적용된다.
    
- 앞으로 진행하는 프로젝트에서 이 MCP 서버를 항상 사용할 수 있게 되어 언제든지 Context7을 호출
    

```
claude mcp add context7 -s user -- npx -y @upstash/context7-mcp
```

### 3. 실전 MCP 예시

| MCP 서버              | 할 수 있는 것                    |
| ------------------- | --------------------------- |
| GitHub MCP          | PR 조회/생성/리뷰, 이슈 관리, 코드 검색   |
| Notion MCP          | 페이지 검색/생성, 데이터베이스 조회, 댓글 작성 |
| Slack MCP           | 메시지 전송, 채널 읽기, 알림           |
| Google Calendar MCP | 일정 관리, 미팅 스케줄링              |
- MCP 연결 플로우 예시
이슈 확인(GitHub) --> 코드 수정 --> PR 생성(GitHub) --> 진행 기록(Notion) --> 팀 알림(Slack)

### 4. Supabase MCP 연결 (선택사항, 강력 추천)

> **MCP (Model Context Protocol)**
	AI 에이전트가 Supabase에 직접 접근할 수 있게 해주는 연결 규격. 연결하면 에이전트가 DB 테이블 조회, SQL 실행, 로그 확인, TypeScript 타입 생성 등을 **자연어 명령**으로 해줌.

💡왜 쓰나요? 
MCP 없이는 에이전트한테 DB 구조를 일일이 설명해야 하지만, MCP를 연결하면 에이전트가 알아서 DB를 보고 코드를 짜줍니다. 특히 RLS 정책, 마이그레이션, 테이블 생성 작업이 훨씬 빨라짐.

- Supabase MCP가 제공하는 기능
    
    |기능 그룹|설명|
    |---|---|
    |**Database**|테이블 목록 조회, SQL 쿼리 실행, 마이그레이션 관리|
    |**Debugging**|서비스 로그 확인, 보안/성능 어드바이저|
    |**Development**|프로젝트 URL, API 키 조회, TypeScript 타입 자동 생성|
    |**Edge Functions**|엣지 펑션 목록 조회, 배포|
    |**Docs**|Supabase 공식 문서 검색|
    
- 설정 방법 (Antigravity / Cursor 기준)
    
    1. 프로젝트 루트에 `.cursor/mcp.json` 파일 생성 (Antigravity도 동일 경로):
    
    ```json
    {
      "mcpServers": {
        "supabase": {
          "url": "<https://mcp.supabase.com/mcp>"
        }
      }
    }
    ```
    
    2. 에디터 재시작 → MCP 클라이언트가 브라우저 로그인을 요청
    3. Supabase 계정으로 로그인하면 연결 완료
    4. 에이전트에게 "내 Supabase 테이블 목록 보여줘" 같은 자연어로 테스트
- 추천 설정 옵션
    
    URL 파라미터로 기능을 제한할 수 있습니다:
    
    ```
    <https://mcp.supabase.com/mcp?read_only=true&project_ref=[프로젝트ID]>
    ```
    
    - `read_only=true` — 읽기 전용 (실수로 데이터 변경 방지)
    - `project_ref=[ID]` — 특정 프로젝트만 접근 가능하게 제한

<aside> ⚠️

**보안 주의사항:**

- **프로덕션 DB에는 직접 연결하지 마세요** — 개발/테스트 환경에서만 사용
- 처음에는 `read_only=true`로 시작하는 것을 추천
- MCP 클라이언트의 **수동 도구 승인(manual tool approval)** 설정을 켜두세요

</aside>

<aside> 💡

**자세한 설정 가이드:** [supabase.com/docs/guides/getting-started/mcp](http://supabase.com/docs/guides/getting-started/mcp)

</aside>

### 5. MCP 토큰 모니터링

- MCP를 여러 개 연결하면 **도구 설명만으로도 토큰을 크게 소비**합니다. `/context`로 주기적으로 확인하고, 안 쓰는 MCP는 `/mcp`에서 비활성화하세요.
- **토큰 절약 팁**: Notion, Linear 같은 MCP는 도구 설명이 매우 큽니다. 자주 쓰는 기능만 골라서 **커스텀 MCP를 래핑**하면 토큰 절약 + 응답 품질 향상

### 6. 훅 이벤트 예

| 이벤트              | 내용                                                                                                                         | matcher                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| PreToolUse       | 도구 실행 전에 훅을 실행                                                                                                             | Task, Bash, Glob, Grep, Read, Edit, MultiEdit, Write, WebFetch, WebSearch 등  |
| PostToolUse      | 도구 실행 후에 훅을 실행                                                                                                             | 위와 같음                                                                        |
| Notification     | 클로드가 Bash 실행을 허용하거나 <br>대기 상태에서 알림을 전송했을때                                                                                  | 없음                                                                           |
| UserPromptSubmit | 사용자가 프롬프트를 입력했을때                                                                                                           | 없음                                                                           |
| Stop             | 메인 에이전트가 응답을 완료했을때<br>(단, 사용자가 강제로 중단했을때는 해당없음)                                                                            | 없음                                                                           |
| SubagentStop     | 클로드 코드 서브에이전트가 응답을 완료했을때                                                                                                   | 없음                                                                           |
| PreCompact       | 컴팩트가 호출됐을 때                                                                                                                | manual: /compact 호출 시<br>auto: 자동 컴팩트 시                                      |
| SessionEnd       | 세션이 종료됐을 때. reason 필드로 종료 이유를<br>표시하는 것이 가능<br>(예: clear, logout, prompt_input_exit, other)<br>이를 통해 파라미터를 받아 로깅 등을 할 수 있음 | 없음                                                                           |
| SessionStart     | 클로드 코드가 새로운 세션을 시작 또는 재개했을 때                                                                                               | startup: 처음 실행<br>resume: --resume,<br>--continue, /resume,<br>clear: /clear |

---
---

