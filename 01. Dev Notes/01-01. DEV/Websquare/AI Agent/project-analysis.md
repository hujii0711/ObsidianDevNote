# websquare-ai-agent-qwen 프로젝트 분석

> 작성일: 2026-08-28
> 기준: `@qwen-code/qwen-code` v0.12.0 포크, `@qwen-code/api-server` v0.1.0

이 문서는 저장소 코드를 직접 읽어 확인한 내용만 담았다. 확인되지 않은 부분은 그렇게 명시했다.

## 목차

1. [프로젝트의 목적](#1-프로젝트의-목적)
2. [포크 계보와 전략](#2-포크-계보와-전략)
3. [에이전트 도구와 역할](#3-에이전트-도구와-역할)
4. [MCP 서버](#4-mcp-서버)
5. [지원 모델과 프로바이더](#5-지원-모델과-프로바이더)
6. [모델별 파라미터 테이블](#6-모델별-파라미터-테이블)
7. [파인튜닝 / RAG 사용 여부](#7-파인튜닝--rag-사용-여부)
8. [에이전트 루프 구현 방식](#8-에이전트-루프-구현-방식)
9. [운영 시 확인이 필요한 지점](#9-운영-시-확인이-필요한-지점)
10. [부록: 주요 파일 위치](#부록-주요-파일-위치)

---

## 1. 프로젝트의 목적

**인스웨이브(Inswave)의 웹스퀘어(WebSquare) 개발 환경에 AI 코딩 에이전트를 붙이기 위한 qwen-code 포크.**

`README.md` 첫 줄이 그대로 정의한다: "qwen-code 포크 프로젝트. api-server, cli, core 3개 패키지만 사용한다."

핵심 산출물은 **`packages/api-server`** — Express + Socket.IO 기반 WebSocket 서버(`src/server/index.ts`). 터미널 CLI로 쓰는 원본 qwen-code와 달리, 에이전트를 **서버로 띄워 두고 외부 클라이언트(웹 UI, Eclipse/IDE 플러그인)가 소켓으로 붙어 쓰는** 형태로 바꿔 놓았다.

### 소켓 이벤트

| 이벤트 | 역할 |
| --- | --- |
| `initialize_session` | 세션 초기화 (query string 방식도 지원) |
| `user_message` | 사용자 메시지 전달 |
| `tool_approval_request` / `tool_approval_response` | 도구 실행 승인 요청/응답 |
| `change_llm` / `get_llm_config` | 모델 전환 및 조회 |
| `search_files` | 파일 검색 |
| `set_global_dir` / `set_project_dir` | 설정 디렉터리 지정 |
| `cancel_request` | 진행 중 요청 취소 |
| `disconnect` | 세션 정리 |

접속 시 `workspacePath`, `deploySource`, `metadataPath`, `globalDirPath`, `projectDirPath`를 받아 세션마다 별도 작업공간을 구성한다. 즉 **멀티 클라이언트 / 멀티 프로젝트용 에이전트 백엔드**다. 기본 포트 3001, `/health` 엔드포인트 제공.

### 웹스퀘어 특화 지점

원본 대비 커스터마이징은 `packages/api-server/esbuild.config.js`의 빌드타임 소스 패치로 들어간다.

| 항목 | 내용 |
| --- | --- |
| 브랜딩 | 시스템 프롬프트를 `"You are AI Talk Plus, an interactive CLI agent developed by Inswave"`로 교체 |
| 저장 경로 | `~/.qwen` 대신 `.deepsquare` 디렉터리에 todos/chats/skills 저장 |
| 컨텍스트 파일 | `QWEN.md`가 아니라 **`DeepSquare.md`**, `AGENTS.md`를 읽도록 변경 |
| 쓰기 권한 제한 | `deploySource`(예: `WebContent`)와 `deepsquare` 하위 경로만 파일 쓰기 허용 |
| 스트리밍 | `functionCallStart` 이벤트를 추가해 도구 호출 시작을 UI에 실시간 전달 |

도메인 지식은 `packages/api-server/.test/DeepSquare.md`(1,169줄, 40KB)에 있다. **웹스퀘어 화면의 TestSquare E2E 테스트 XML을 자동 생성**하는 가이드로, `xmlns:w2="http://www.inswave.com/websquare"` 네임스페이스와 `@testsquare/test`(Playwright 래퍼) 기반 테스트케이스 골격을 정의한다.

### 요약

**웹스퀘어 개발자가 IDE나 웹 UI에서 AI 에이전트("AI Talk Plus")를 호출해 화면 소스 이해·수정과 TestSquare E2E 테스트 코드 생성을 자동화하도록, qwen-code를 소켓 서버로 재포장하고 인스웨이브 환경에 맞게 경로·권한·프롬프트를 바꾼 프로젝트.**

참고로 `packages/`에는 upstream의 `webui`, `sdk-java`, `sdk-typescript`, `vscode-ide-companion` 등이 남아 있지만 workspaces에는 `core`, `cli`, `api-server`만 등록돼 있어 빌드 대상이 아니다.

---

## 2. 포크 계보와 전략

### 계보

포크가 두 단계로 겹쳐 있다.

```
Google Gemini CLI
      ↓ 포크 (Alibaba / QwenLM)
qwen-code  ─ Qwen3-Coder 모델에 맞춘 파서 레벨 적응
      ↓ 포크 (Inswave)
websquare-ai-agent-qwen  ─ 웹스퀘어 환경 적응
```

`README.md`에 qwen-code 팀이 직접 밝혀 둔 문장이 있다: *"This project is based on Google Gemini CLI... Our main contribution focuses on parser-level adaptations to better support Qwen-Coder models."*

흔적은 라이선스 헤더에 남아 있다. `packages/core/src`의 TypeScript 파일 중 **272개가 `Copyright 2025 Google LLC`**, **121개가 `Copyright 2025 Qwen Team`**이다. core의 3분의 2는 아직 Google 원본 코드고, qwen-code가 그 위에 3분의 1을 덧댄 구조다.

### 이 포크가 직접 수정한 파일

README 두 번째 줄이 범위를 명시한다 — **단 두 파일**이다.

- **`package.json`** — `workspaces`를 `core`, `cli`, `api-server` 셋으로 축소, `prepare`를 비워 husky 훅 비활성화, `postinstall`에 `npm audit fix` 추가
- **`.npmrc`** — `audit-level=high`, `min-release-age=7d` (배포된 지 7일 안 된 패키지는 설치하지 않음 — 공급망 공격 방어)

### 빌드타임 패치 전략

보통 포크는 원본 소스를 직접 고친다. 이 프로젝트는 **core 소스를 건드리지 않는다.**

`packages/api-server/esbuild.config.js`가 대신한다. 주석에 의도가 적혀 있다: *"websquare-ai-agent-talk 프로젝트와 동일한 기능을 core 소스 수정 없이 구현."*

빌드할 때 `../core/src`의 파일들을 find/replace로 **일시적으로 패치 → 번들링 → 원복**한다.

| 패치 대상 (core 소스) | 패치 내용 |
| --- | --- |
| `config/storage.ts` | metadataPath 지원, `getMetadataQwenDir` / `getTodosDir` 메서드 |
| `config/config.ts` | `setMetadataPath`, `setGlobalConfigDir`, `getStorage` 메서드 추가 |
| `tools/todoWrite.ts` | `QWEN_HOME` 환경변수 우선 사용 |
| `skills/skill-manager.ts` | `globalConfigDir` 경로 우선 사용 |
| `core/turn.ts` | `GeminiEventType.ToolCallStart` 추가, `functionCallStart` 감지 |
| `core/anthropicContentGenerator/` | `content_block_start`에서 `functionCallStart` emit |
| `core/openaiContentGenerator/` | `StreamingToolCallParser`에 `isNewToolStart` 감지, converter에서 emit |
| `core/prompts.ts` | 시스템 프롬프트를 "AI Talk Plus" (Inswave)로 변경 |

**장점** — core를 깨끗하게 유지하면 upstream qwen-code의 새 버전을 충돌 없이 당겨올 수 있다. 포크 유지비 최소화.

**대가** — 패치가 문자열 매칭이라 **upstream에서 해당 코드 한 줄만 바뀌어도 조용히 실패한다.** README가 *"빌드 시 패치 실패 로그가 출력되면 해당 파일의 변경 사항을 확인하고 `esbuild.config.js`의 find/replace 문자열을 업데이트해야 한다"*고 경고하는 이유다. 업스트림 머지 때마다 확인해야 할 체크리스트인 셈이다.

---

## 3. 에이전트 도구와 역할

도구는 **core에 등록된 전체 세트**와 웹스퀘어 배포에서 **실제로 켜 놓은 축소 세트** 두 층으로 나뉜다.

### 3.1 core가 제공하는 전체 도구 (16종)

`packages/core/src/tools/tool-names.ts`와 `config.ts:1859-1907`의 등록 순서 기준.

| 분류 | 도구 | 역할 |
| --- | --- | --- |
| 탐색 | `list_directory`, `glob`, `grep_search` | 디렉터리 훑기, 파일명 패턴 검색, 내용 정규식 검색 (ripgrep 있으면 `RipGrepTool`로 대체) |
| 읽기 | `read_file`, `read_many_files` | 소스·이미지·PDF 읽기 |
| 쓰기 | `edit`, `write_file` | 부분 수정 / 파일 생성·전체 교체 |
| 실행 | `run_shell_command` | 셸 명령 실행 |
| 코드 인텔리전스 | `lsp` | LSP 연동 — 정의/구현/참조, hover, call hierarchy |
| 작업 관리 | `todo_write`, `exit_plan_mode` | 할 일 목록, 계획 모드 종료 (SDK 모드에선 `exit_plan_mode` 미등록) |
| 위임 | `task` | 서브에이전트 실행 |
| 확장 | `skill` | 스킬(패키지화된 절차) 호출 |
| 기억 | `save_memory` | 사용자 컨텍스트 영속화 |
| 웹 | `web_fetch`, `web_search` | URL 가져오기, 검색 (프로바이더 설정 시에만 등록) |
| 대화 | `ask_user_question` | 사용자에게 선택지 질의 |
| 외부 | MCP 도구 | `mcp-client-manager.ts`로 동적 등록 |

### 3.2 웹스퀘어 배포에서 허용한 도구 (8종)

`.deepsquare/settings.json`이 `tools.core`로 화이트리스트를 건다.

```
list_directory, read_file, read_many_files, edit, write_file, grep_search, glob, todo_write
```

**파일 읽기·검색·수정과 할 일 관리만** 남기고 `run_shell_command`, `web_fetch`, `web_search`, `task`, `skill`은 제외했다. 셸 실행을 뺀 것이 특히 의도적으로 보인다 — IDE에 붙은 원격 에이전트가 임의 명령을 돌리지 못하게 막는 구조다.

대신 외부 능력은 MCP로 붙인다.

### 3.3 안전장치

**경로 제한** (`qwen-server.ts:165-180`) — 쓰기 가능 경로를 세션마다 두 곳으로 못박는다.

- `{workspace}/{deploySource}` (예: `WebContent`)
- `{workspace}/deepsquare`

읽기는 워크스페이스 전체, 쓰기는 이 두 경로 밖으로 못 나간다.

**승인 플로우** — `approvalMode = "default"`로 고정돼, 도구가 `awaiting_approval` 상태가 되면 `tool_approval_request`를 클라이언트로 쏘고 `tool_approval_response`를 기다린다. 자동 승인이 아니라 **매 수정마다 IDE 쪽 사용자 확인**을 거친다.

**기타** — 텔레메트리·사용량 통계 비활성화, `security.folderTrust`는 MCP 검색이 막히지 않도록 비활성화.

### 3.4 역할 정의

`task` 도구가 부르는 서브에이전트는 `builtin-agents.ts`에 `general-purpose` 하나만 내장돼 있고, 나머지는 사용자가 `.deepsquare/agents/`에 정의하는 구조다. 다만 화이트리스트에 `task`가 없으므로 **현재 배포 설정에서는 서브에이전트 위임이 꺼져 있다.**

실제 역할 정의는 도구가 아니라 **컨텍스트 파일**이 담당한다. 컨텍스트 파일명을 `DeepSquare.md` / `AGENTS.md`로 바꿔 global·project 디렉터리에서 로드하는데, 예시로 들어 있는 `DeepSquare.md`가 그 역할 명세서다 — *"test-plan.md와 interface-metadata.json을 기반으로 웹스퀘어 화면의 TestSquare E2E 테스트 XML을 생성"*.

즉 **역할을 바꾸려면 코드가 아니라 `DeepSquare.md`와 `settings.json`을 갈아끼우면 된다.**

---

## 4. MCP 서버

> **중요**: MCP 서버 자체는 이 저장소에 없다. 저장소에는 호출하는 클라이언트 쪽 배선만 있다.

### 등록된 서버

`.deepsquare/settings.json`에 하나 있다.

```json
"websquare-mcp": {
  "command": "node",
  "args": ["./dist/websquare-mcp.mjs"],
  "cwd": "/Users/fujii0711/Develop/PaaS/src/deepsquare-mcp",
  "trust": true
}
```

- **stdio 방식** — `node ./dist/websquare-mcp.mjs`를 자식 프로세스로 띄우고 표준입출력으로 통신 (`url`/`httpUrl`이 아니므로 `StdioClientTransport` 경로)
- **`cwd`가 개발자 개인 macOS 경로**(`/Users/fujii0711/...`). 저장소에도 없고 다른 머신에도 없는 경로라 **현재 상태 그대로는 연결되지 않는다.** 개발자 로컬 설정이 커밋된 것으로 보인다
- **`trust: true`** — 이 서버가 내놓는 도구는 사용자 승인 없이 바로 실행된다. `approvalMode: "default"` 승인 플로우를 우회하는 유일한 통로
- 별도 저장소 이름은 `deepsquare-mcp`, 빌드 산출물이 `websquare-mcp.mjs`

### 이 서버가 맡는 역할 (정황 근거)

서버 구현이 없어 실제 도구 목록은 확인할 수 없다. 다만 어떤 역할을 메우려고 붙였는지는 좁혀진다.

에이전트에 허용된 core 도구는 파일 조작 8종뿐이다. 즉 **파일을 읽고 쓰는 것 외의 모든 능력은 MCP를 통해서만** 들어온다.

그리고 `DeepSquare.md` 7장이 요구하는 입력은 파일 시스템을 뒤져서 나오는 게 아니다.

- **`interface-metadata.json`** — 화면의 `collections`(DataList/DataMap 정의와 필드), `submissions`(API URL과 요청/응답 컬렉션), `popups`, `messages`. 웹스퀘어 XML 화면을 **파싱해서 구조화한 산출물**
- **`test-plan.md`** — 화면 ID, 시나리오 목록, 우선순위, 관련 API 엔드포인트

`screenId`(예: `BM002M01`)로 화면을 조회하고 데이터 모델·서브미션 구조를 뽑아내는 일 — 웹스퀘어 IDE/PaaS 플랫폼이 쥐고 있는 메타데이터다. `websquare-mcp`가 그 게이트웨이 역할을 하는 것으로 읽는 게 자연스럽다.

### 클라이언트 쪽 배선

1. **설정 병합** (`qwen-server.ts:244-247`) — `mcpServers`를 global/project 디렉터리의 `settings.json`에서 딥머지. 세션마다 다른 MCP 서버 세트를 붙일 수 있다
2. **folderTrust 강제 해제** (`qwen-server.ts:283-285`) — core의 `mcp-client-manager.ts:83`이 `isTrustedFolder()`가 false면 MCP 탐색을 통째로 건너뛰기 때문에, 서버 환경에서 걸리지 않도록 껐다

연결 뒤에는 core가 도구/프롬프트를 discovery해 레지스트리에 등록하고, 헬스체크 타이머로 죽으면 재연결한다.

---

## 5. 지원 모델과 프로바이더

Qwen 전용이 아니다. **모델은 런타임에 클라이언트가 지정하는 값**이다.

### 5.1 지원 프로바이더 (5종)

`contentGenerator.ts:55-61`의 `AuthType`:

| AuthType | 대상 |
| --- | --- |
| `qwen-oauth` | Qwen OAuth (무료 티어) |
| `openai` | OpenAI 및 **OpenAI 호환 API 전부** |
| `anthropic` | Claude |
| `gemini` | Google Gemini |
| `vertex-ai` | Google Vertex AI |

각각 `packages/core/src/core` 아래에 전용 content generator가 있다 — `openaiContentGenerator`, `anthropicContentGenerator`, `geminiContentGenerator`.

### 5.2 이 포크가 실제로 손본 경로

README 패치 표에서 `functionCallStart` 스트리밍 이벤트를 주입한 대상이 **두 곳**이다.

- `core/anthropicContentGenerator/` — `content_block_start`에서 emit
- `core/openaiContentGenerator/` — `StreamingToolCallParser`의 `isNewToolStart` 감지 후 emit

**Anthropic과 OpenAI 계열 둘 다 실사용 전제.** Gemini 쪽은 패치하지 않았다.

### 5.3 런타임 모델 전환

`qwen-server.ts:872`의 `change_llm` 핸들러가 소켓으로 모델을 갈아끼운다.

```
{ authType, apiKey, model, baseUrl, modelMaxTokens, maxOutputTokens, studioConfig }
```

`config.updateCredentials()`로 키·baseUrl·모델을 바꾸고, `authType`이 오면 `refreshAuth()`로 프로바이더 자체를 교체한다. 컨텍스트 윈도우(`contextWindowSize`)와 최대 출력 토큰도 함께 지정 가능. **세션 중 모델 전환이 가능한 구조.**

### 5.4 실제 배포 형태 — Studio 프록시

가장 특징적인 부분은 `packages/api-server/src/adapters/config-adapter.ts`다. Eclipse IDE가 넘기는 `StudioServerConfig`를 처리한다.

```
baseUrl, apiKey, model, authType, proworksBody, proworksLang, proworksTKey
```

주석에 설계 의도가 있다: *"The proworksTKey is used as the apiKey since the studio's admin server acts as a proxy and uses this token for authentication."*

즉 실제 운영에서 에이전트는 모델 벤더에 직접 붙지 않는다. **인스웨이브의 ProWorks/Studio 관리 서버가 프록시로 서고, 에이전트는 거기에 `proworksTKey`를 들고 붙는다.** 어떤 모델을 쓸지는 프록시 뒤에서 결정되고, 클라이언트는 `model` 문자열만 넘긴다.

이 방식이면 **개별 개발자 PC에 API 키를 뿌리지 않아도 되고**, 사내에서 모델 선택·과금·감사를 한 곳에서 통제할 수 있다.

### 5.5 기본값

`models.ts`의 기본 모델명은 `'coder-model'`이라는 **추상 이름**이다 (실제 모델 ID가 아님). 임베딩만 `text-embedding-v4`로 구체적이다. 기본값조차 특정 모델에 못박지 않고 프록시가 해석하도록 열어 둔 형태다.

---

## 6. 모델별 파라미터 테이블

코드베이스에는 모델별 **파라미터 수(70B 등)나 벤더 버전 문서가 없다.** 대신 모델명을 정규식으로 매칭해 **컨텍스트 윈도우 / 출력 토큰 / 입력 모달리티**를 결정하는 레지스트리가 있다.

### 6.1 컨텍스트 윈도우 (입력)

`tokenLimits.ts:80-140` — 위에서부터 첫 매칭이 이긴다.

| 패턴 | 컨텍스트 | 비고 |
| --- | --- | --- |
| `gemini-3*`, `gemini-*` | **1M** | 3.x·2.x·1.5 전부 동일 |
| `gpt-5*` | **400K** | 벤더 선언 십진값 |
| `gpt-*` | 128K | 4o, 4.1 등 폴백 |
| `o숫자*` | 200K | o3, o4-mini 등 |
| `claude-*` | **200K** | 전 모델 일괄 |
| `qwen3-coder-plus`, `qwen3-coder-flash`, `qwen3.5-plus`, `qwen-plus-latest`, `qwen-flash-latest`, `coder-model` | **1M** | 상용 API |
| `qwen3-max*`, `qwen3-coder-*`, `qwen*` | 256K | 오픈소스 변형·폴백 |
| `deepseek*` | 128K | |
| `glm-5*`, `glm-*` | 202,752 | 벤더 정확값 |
| `minimax-m2.5` | 1M | |
| `minimax-*` | 200K | |
| `kimi-*` | 256K | |
| `seed-oss*` | 512K | ByteDance |
| **미매칭 기본값** | **131,072 (128K)** | `DEFAULT_TOKEN_LIMIT` |

### 6.2 최대 출력 토큰

`tokenLimits.ts:147-181` — 별도 패턴 테이블.

| 패턴 | 출력 |
| --- | --- |
| `gemini-3*` | 64K |
| `gemini-*` | 8K |
| `gpt-5*`, `o숫자*` | **128K** |
| `gpt-*` | 16K |
| `claude-opus-4-6` | **128K** |
| `claude-sonnet-4-6` | 64K |
| `claude-*` | 64K |
| `qwen3.5*`, `coder-model`, `qwen3-max*` | 64K |
| `deepseek-reasoner` | 64K |
| `deepseek-chat` | 8K |
| `glm-5*`, `glm-4.7*` | 16K |
| `minimax-m2.5` | 64K |
| `kimi-k2.5` | 32K |
| **미매칭 기본값** | **8,192 (8K)** |

### 6.3 입력 모달리티

`modalityDefaults.ts:20-75` — 미매칭이면 **텍스트 전용**.

| 패턴 | image | pdf | audio | video |
| --- | :---: | :---: | :---: | :---: |
| `gemini-*` | O | O | O | O |
| `gpt-*`, `o숫자*` | O | | | |
| `claude-*` | O | O | | |
| `qwen3.5-plus`, `coder-model` | O | | | O |
| `qwen-vl-*`, `qwen3-vl-*` | O | | | O |
| `qwen3-coder-*`, `qwen*` | - | - | - | - |
| `glm-4.5v` | O | | | |
| `kimi-k2.5` | O | | | O |
| `deepseek*`, `minimax-*`, `glm-5*` | - | - | - | - |

### 6.4 버전 정규화 규칙

핵심은 `normalize()` (`tokenLimits.ts:35`)다. 매칭 전에 모델명에서 **버전·날짜 꼬리표를 떼어낸다.**

- 프로바이더 접두사 제거 (`openai/gpt-4o` → `gpt-4o`), 파이프·콜론 뒤만 사용
- `-preview` 제거
- 날짜(`-20250219`), `-v1.2`, `-latest`, `-exp`, 파라미터 크기(`-7b`, `-4x8b`) 제거
- 양자화 접미사 제거 (`-int4`, `-bf16`, `-q5`, `-8bit`)

**예외 두 가지** — 버전이 곧 정체성이라 날짜를 남긴다.

- `qwen-plus-latest`, `qwen-flash-latest`, `qwen-vl-max-latest`
- `kimi-k2-0905` 같은 `kimi-k2-4자리`

정규식 주석에 명시된 미묘한 지점: `gpt-4.1`의 `4.1`은 앞에 대시가 하나뿐이라 버전으로 잘리지 않는다.

### 6.5 샘플링 파라미터

모델별 기본값 테이블은 **없다.** `contentGenerator.ts:87-94`에 스키마만 정의돼 있고 전부 옵셔널이다.

```
top_p, top_k, repetition_penalty, presence_penalty, temperature, max_tokens
```

값은 벤더 기본값에 맡기거나, 클라이언트가 `change_llm`으로 넘긴다 — `modelMaxTokens` → `contextWindowSize`, `maxOutputTokens` → `samplingParams.max_tokens`.

### 6.6 주의

이 테이블들은 **자동 감지 폴백일 뿐이다.** `tokenLimit()` 주석이 못박는다: *"primarily used during config initialization to auto-detect... After initialization, code should use `contentGeneratorConfig.contextWindowSize` or `maxOutputTokens` directly."*

이 포크의 운영 형태에서는 특히 그렇다. ProWorks Studio 프록시가 넘기는 `model` 문자열이 위 패턴 중 어느 것에도 안 걸리면 **128K / 8K / 텍스트전용**으로 떨어진다. 프록시가 커스텀 모델명을 쓴다면 클라이언트가 `modelMaxTokens`·`maxOutputTokens`를 명시적으로 넘기는 편이 안전하다.

---

## 7. 파인튜닝 / RAG 사용 여부

**둘 다 사용하지 않는다.**

### 7.1 파인튜닝 — 흔적 없음

`fine-tun`, `finetun`, `lora`, `.safetensors`, 학습 스크립트, 데이터셋 준비 코드 — `packages/core/src`와 `packages/api-server` 전체에서 매칭 0건.

구조적으로도 불가능에 가깝다. 이 프로젝트는 **모델을 API로 호출하는 클라이언트**이고, 실제로는 ProWorks Studio 프록시 뒤의 모델을 쓴다. 모델 가중치를 만질 지점 자체가 없다.

### 7.2 RAG — 없음 (부분 재료만 존재)

`vectorstore`, `faiss`, `chromadb`, `pinecone`, `qdrant`, `cosine`, `retrieval` 검색 결과는 전부 오탐이었다.

- `getFolderStructure.ts:24` — "folder structure retrieval" (주석)
- `qwenContentGenerator.ts:117` — "token and endpoint retrieval" (OAuth 토큰)
- `prompts.ts` — "codebase exploration" (서브에이전트 설명)

**임베딩 API는 있지만 아무도 안 쓴다.** `baseLlmClient.ts:157`에 `generateEmbedding()`이 구현돼 있고 기본 모델도 `text-embedding-v4`로 잡혀 있다. 그런데 **저장소 전체에서 이 메서드를 호출하는 곳이 단 한 군데도 없다** — 정의부 한 줄이 유일한 매칭이다. 청킹도, 인덱싱도, 유사도 계산도, 벡터 저장소도 없다.

Gemini CLI에서 물려받은 미사용 코드로 보인다. 참고로 Anthropic 경로는 아예 `throw new Error('Anthropic does not support embeddings.')`이고, OpenAI 경로는 하드코딩된 `text-embedding-ada-002`를 쓴다.

### 7.3 대신 쓰는 방식 — 에이전틱 검색

RAG 자리를 **도구 기반 탐색**이 대신한다. 벡터로 유사 문서를 찾아 프롬프트에 끼워넣는 대신, 모델이 `glob` → `grep_search` → `read_file`을 스스로 호출해 필요한 파일을 찾아 읽는다. 허용 도구 8종이 정확히 그 조합이다.

컨텍스트 주입도 정적이다 — `DeepSquare.md` / `AGENTS.md`를 global·project 디렉터리에서 통째로 로드한다. 40KB짜리 `DeepSquare.md` 전문이 매 세션 들어가는 구조지, 질의별로 관련 조각을 뽑아오는 게 아니다.

### 7.4 눈에 띄는 설정

`.deepsquare/settings.json`에 `chatCompression.contextPercentageThreshold: 0`이 있다.

`chatCompressionService.ts:89-97`을 보면 `threshold <= 0`이면 즉시 `NOOP`을 반환한다 — **대화 히스토리 압축을 완전히 껐다**는 뜻이다.

즉 컨텍스트 관리 전략이 "검색해서 좁히기"도 "요약해서 줄이기"도 아니고, **1M 컨텍스트 윈도우에 전부 밀어넣고 버티기**다. `coder-model`의 1M 컨텍스트 설정과 맞물린다. 웹스퀘어 XML 화면 소스와 테스트 가이드를 통째로 넣어야 하는 작업 성격상 나온 선택으로 보이나, 긴 세션에서는 컨텍스트 한계에 그대로 부딪힐 수 있는 설정이기도 하다.

---

## 8. 에이전트 루프 구현 방식

**LangGraph는 물론 LangChain, LlamaIndex, AutoGen, CrewAI 어느 것도 쓰지 않는다.** `packages/core/package.json`에 에이전트 프레임워크 의존성이 하나도 없다. LLM 관련 의존성은 벤더 SDK 3개뿐이다.

```
@anthropic-ai/sdk, openai, @google/genai, @modelcontextprotocol/sdk
```

에이전트 루프는 **직접 구현**돼 있다. 핵심 파일이 약 4,900줄이다.

| 파일 | 줄 수 | 역할 |
| --- | ---: | --- |
| `core/coreToolScheduler.ts` | 1,406 | 도구 실행 상태 머신 |
| `subagents/subagent.ts` | 1,010 | 서브에이전트 실행기 |
| `core/client.ts` | 785 | 메인 에이전트 루프 |
| `core/geminiChat.ts` | 746 | 대화 히스토리 관리 |
| `tools/task.ts` | 570 | 위임 도구 |
| `core/turn.ts` | 418 | 턴 단위 이벤트 정의 |

### 8.1 에이전트 루프 — async generator 재귀

LangGraph의 그래프/노드/엣지 대신 **비동기 제너레이터의 재귀 호출**을 쓴다.

`client.ts:413`의 `async *sendMessageStream()`이 중심이다. 이벤트를 `yield`로 흘려보내다가, 대화를 이어가야 하면 **자기 자신을 `yield*`로 재귀 호출**한다.

```js
return yield* this.sendMessageStream(nextRequest, signal, prompt_id, options, boundedTurns - 1);
```

`boundedTurns - 1`이 재귀 깊이 제한이다 — LangGraph의 `recursion_limit`에 해당하는 역할을 인자 하나로 처리한다.

**계속할지 판단하는 지점이 두 곳이다.**

- 도구 호출이 남아 있으면 → 실행 후 결과를 넣고 재귀
- 도구 호출이 없으면 → `checkNextSpeaker()`로 **LLM에게 "다음 차례가 누구냐"를 물어보고**, `model`이면 `"Please continue."`를 넣어 재귀

### 8.2 상태 전달 — 이벤트 스트림

LangGraph의 `State` 객체 대신 **타입 태그가 붙은 이벤트 유니온**이 흐른다. `turn.ts:52`의 `GeminiEventType` 15종:

```
content, thought, tool_call_request, tool_call_response, tool_call_confirmation,
user_cancelled, error, chat_compressed, max_session_turns,
session_token_limit_exceeded, finished, loop_detected, citation, retry,
hook_system_message
```

이 포크가 여기에 **`ToolCallStart`를 추가**했다. 도구 호출이 시작되는 순간을 잡아 IDE UI에 실시간 표시하기 위해서다.

api-server는 이 이벤트를 그대로 소켓으로 중계한다 — 제너레이터가 곧 스트리밍 파이프라인이다.

### 8.3 도구 호출 — CoreToolScheduler

가장 큰 조각인 `coreToolScheduler.ts`(1,406줄)가 도구 실행 전체를 관리한다. 명시적 상태 머신이다.

```
validating → awaiting_approval → scheduled → executing → success / error / cancelled
```

`awaiting_approval` 상태에서 멈추면 api-server가 `tool_approval_request`를 클라이언트에 쏘고 응답을 기다린다. **human-in-the-loop이 프레임워크 기능이 아니라 상태 머신에 직접 박혀 있다.**

도구 자체는 `tool-registry.ts`에 등록되고, 스키마 검증은 `ajv`로 한다. MCP 도구는 `@modelcontextprotocol/sdk`로 discovery해서 같은 레지스트리에 합류시킨다.

### 8.4 에이전트 연계 — Task 도구 + Subagent

멀티에이전트도 프레임워크 없이 구현했다.

- `tools/task.ts` — `task` 도구. 모델이 호출하면 서브에이전트가 뜬다
- `subagents/subagent.ts` — 자체 컨텍스트·도구 세트·프롬프트를 가진 **독립 루프**를 돌리고 결과만 상위로 반환
- `subagents/subagent-manager.ts` — 정의 로드·검증
- 내장 에이전트는 `general-purpose` 하나뿐, 나머지는 사용자가 마크다운으로 정의

구조상 **부모-자식 위임(delegation)**이지 LangGraph 같은 임의 그래프가 아니다. 단, 웹스퀘어 배포 설정에서는 `task`가 화이트리스트에 없어 **이 기능이 꺼져 있다.**

### 8.5 프레임워크 없이 직접 만든 안전장치

| 기능 | 구현 |
| --- | --- |
| 무한루프 방지 | `loop_detected` 이벤트 + `LoopDetectionService` |
| 컨텍스트 압축 | `chatCompressionService.ts` (웹스퀘어에선 비활성) |
| 중단 | `AbortSignal` 전파 → `cancel_request` 소켓 이벤트 |
| 재시도 | `retry` 이벤트 + 백오프 |
| 훅 | `Stop` 훅이 종료를 막고 강제 계속시킬 수 있음 |
| 관측 | OpenTelemetry (웹스퀘어에선 비활성) |
| 동시성 | `async-mutex` |

### 8.6 왜 이렇게 했나

Gemini CLI 계보 자체가 **터미널 CLI 도구**로 출발했다. LangGraph는 Python 중심이고, 이쪽은 TypeScript로 단일 번들을 뽑아야 한다 — api-server가 esbuild로 `dist/server.js` 하나를 만드는 구조에서 무거운 프레임워크 의존성은 부담이다.

또 스트리밍 UX가 핵심 요구사항이다. 토큰 단위로 UI에 흘려보내면서 중간에 승인을 받고 취소도 되어야 하는데, async generator가 그 모델에 자연스럽게 맞는다.

### 8.7 정리

**LangGraph 없이, async generator 재귀 + 이벤트 스트림 + 도구 상태 머신으로 직접 구현한 에이전트 루프.** 그래프 선언 대신 재귀 호출, 상태 객체 대신 타입 태그 이벤트, 프레임워크 체크포인트 대신 자체 훅과 압축 서비스를 쓴다. 웹스퀘어 포크는 이 위에 `ToolCallStart` 이벤트 하나를 얹어 IDE 실시간 표시를 붙였을 뿐, 루프 자체는 upstream 그대로다.

---

## 9. 운영 시 확인이 필요한 지점

분석 과정에서 드러난, 배포 전에 확인이 필요한 항목들.

| # | 항목 | 위치 | 내용 |
| --- | --- | --- | --- |
| 1 | MCP `cwd`가 개인 경로 | `.deepsquare/settings.json` | `/Users/fujii0711/Develop/PaaS/src/deepsquare-mcp` — 배포 환경 경로로 교체 필요. 현재 상태로는 MCP 연결 실패 |
| 2 | MCP `trust: true` | 같은 파일 | 해당 서버 도구는 승인 없이 실행됨. 의도된 것인지 확인 |
| 3 | 컨텍스트 압축 비활성 | `chatCompression.contextPercentageThreshold: 0` | 긴 세션에서 컨텍스트 한계에 그대로 부딪힘 |
| 4 | 모델명 폴백 | `tokenLimits.ts` | 프록시가 커스텀 모델명을 쓰면 128K/8K/텍스트전용으로 떨어짐. `modelMaxTokens`·`maxOutputTokens` 명시 권장 |
| 5 | 빌드타임 패치 취약성 | `esbuild.config.js` | upstream 머지 시 8개 패치 지점 문자열 매칭 확인 필요 |
| 6 | 미사용 임베딩 코드 | `baseLlmClient.ts:157` | `generateEmbedding()` 호출처 없음 (동작상 문제는 아님) |

---

## 부록: 주요 파일 위치

| 목적 | 경로 |
| --- | --- |
| 소켓 서버 진입점 | `packages/api-server/src/server/index.ts` |
| 세션·설정·승인 처리 | `packages/api-server/src/server/qwen-server.ts` |
| Studio 프록시 어댑터 | `packages/api-server/src/adapters/config-adapter.ts` |
| 빌드타임 패치 | `packages/api-server/esbuild.config.js` |
| 배포 설정 (도구 화이트리스트, MCP) | `packages/api-server/.deepsquare/settings.json` |
| 역할 명세 (TestSquare 가이드) | `packages/api-server/.test/DeepSquare.md` |
| 에이전트 루프 | `packages/core/src/core/client.ts` |
| 도구 실행 상태 머신 | `packages/core/src/core/coreToolScheduler.ts` |
| 이벤트 타입 정의 | `packages/core/src/core/turn.ts` |
| 도구 이름 상수 | `packages/core/src/tools/tool-names.ts` |
| 모델별 토큰 한계 | `packages/core/src/core/tokenLimits.ts` |
| 모델별 모달리티 | `packages/core/src/core/modalityDefaults.ts` |
| 프로바이더 타입 | `packages/core/src/core/contentGenerator.ts` |
