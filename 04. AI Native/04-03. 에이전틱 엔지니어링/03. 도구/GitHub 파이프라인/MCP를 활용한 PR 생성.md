
MCP(Model Context Protocol)로 GitHub PR을 요청하면, 실제로는 **AI가 GitHub API를 대신 호출할 수 있게 해주는 "도구(tool)"들을 MCP 서버가 제공**하고, AI가 그 도구들을 순차적으로 호출하는 방식으로 작동합니다.

## 전체 구조

```
사용자 → AI(Claude 등) → MCP 서버 → GitHub API → 실제 GitHub 저장소
```

MCP 서버 자체가 "PR을 써주는" 것이 아니라, **AI가 사용할 수 있는 GitHub 관련 함수(도구) 목록을 제공**하는 역할을 합니다. AI는 이 도구들을 조합해서 작업을 수행합니다.

## MCP 서버가 제공하는 대표적인 도구들

GitHub MCP 서버(공식/커뮤니티 버전 모두)는 보통 이런 도구들을 노출합니다:

- `get_file_contents` — 특정 파일 내용 읽기
- `search_code` / `search_repositories` — 코드나 저장소 검색
- `create_branch` — 새 브랜치 생성
- `create_or_update_file` — 파일 생성/수정 (커밋 생성)
- `push_files` — 여러 파일을 한 번에 커밋/푸시
- `create_pull_request` — PR 생성
- `get_pull_request` / `list_pull_requests` — 기존 PR 조회
- `create_pull_request_review` / `add_comment` — 리뷰/코멘트 작성

## 실제 작업 흐름 예시

사용자가 "이 버그 고치고 PR 만들어줘"라고 하면, AI는 대략 이런 순서로 도구를 호출합니다:

1. **저장소 상태 파악** — `get_file_contents`로 관련 파일 읽기
2. **브랜치 생성** — `create_branch`로 `fix/bug-123` 같은 브랜치 생성
3. **코드 수정 및 커밋** — `create_or_update_file` 또는 `push_files`로 변경사항 커밋
4. **PR 생성** — `create_pull_request` 호출 (제목, 설명, base/head 브랜치 지정)
5. **결과 반환** — 생성된 PR URL을 AI가 사용자에게 알려줌

## 인증은 어떻게 되나

MCP 서버는 사용자가 미리 연결해둔 **GitHub 토큰(OAuth 또는 Personal Access Token)**을 사용해 API를 호출합니다. 즉, AI 자체가 GitHub 계정 정보를 갖는 게 아니라 MCP 서버가 인증을 대신 처리하고, AI는 "이 도구를 이 파라미터로 호출해줘"라는 요청만 보냅니다.

## 핵심 포인트

- MCP 서버 = **번역기/중개자** 역할 (AI의 요청 → GitHub API 호출 형식으로 변환)
- 실제 Git 작업(브랜치, 커밋, 푸시)은 GitHub API를 통해 이루어짐 (로컬 git 명령어를 쓰는 게 아님)
- AI는 여러 도구 호출을 **논리적 순서로 조합**해서 하나의 작업(PR 생성)을 완성

원하시면 실제로 GitHub MCP 서버를 연결해서 PR을 만들어드릴 수도 있어요. 혹시 지금 작업 중인 저장소가 있으신가요?