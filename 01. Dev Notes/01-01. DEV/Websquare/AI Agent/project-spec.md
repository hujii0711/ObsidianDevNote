# 웹스퀘어 AI 에이전트 기술 스펙

> 기준일: 2026-08-28
> 베이스: `@qwen-code/qwen-code` 0.12.0 포크
> 산출물: `@qwen-code/api-server` 0.1.0
> 작업 디렉터리: `c:\websquare-ai-agent`

저장소 코드를 직접 읽어 확인한 사실만 담았다. 확인되지 않은 항목은 그렇게 표시했다.

## 상태 표기

| 표기 | 의미 |
| --- | --- |
| **활성** | 현재 코드에서 동작 |
| **비활성** | 코드는 있으나 꺼져 있음 |
| **미등록** | 참조되나 구현·등록 없음 |
| **확인필요** | 배포 전 조치 대상 |

## 목차

1. [정체와 계보](#1-정체와-계보)
2. [런타임 · 빌드](#2-런타임--빌드)
3. [패키지 구조](#3-패키지-구조)
4. [소켓 · HTTP API](#4-소켓--http-api)
5. [세션 · 경로 · 권한](#5-세션--경로--권한)
6. [파일 쓰기 경로](#6-파일-쓰기-경로)
7. [빌드타임 패치](#7-빌드타임-패치)
8. [설정 스펙](#8-설정-스펙)
9. [도구 스펙](#9-도구-스펙)
10. [모델 · 프로바이더](#10-모델--프로바이더)
11. [토큰 · 모달리티 레지스트리](#11-토큰--모달리티-레지스트리)
12. [프롬프트 · 컨텍스트](#12-프롬프트--컨텍스트)
13. [에이전트 루프](#13-에이전트-루프)
14. [MCP](#14-mcp)
15. [의존성](#15-의존성)
16. [비활성 인벤토리](#16-비활성-인벤토리)
17. [운영 리스크](#17-운영-리스크)

---

## 1. 정체와 계보

웹스퀘어 개발자가 IDE나 웹 UI에서 AI 에이전트를 호출해 화면 소스 이해·수정과 TestSquare E2E 테스트 코드 생성을 자동화하도록, `qwen-code`를 **소켓 서버로 재포장**하고 인스웨이브 환경에 맞게 경로·권한·프롬프트를 바꾼 프로젝트다. 에이전트 브랜드명은 **AI Talk Plus**.

| 항목 | 값 |
| --- | --- |
| 포크 단계 | 2단계 |
| 배포 형태 | WebSocket 서버 |
| 기본 포트 | 3001 |
| api-server 소스 | 1,484줄 / 4파일 |
| core .ts 파일 | 425개 |
| 빌드타임 패치 | 20건 / 9파일 |

### 계보

```
Google Gemini CLI
      ↓ 포크 (Alibaba / QwenLM)
qwen-code  — Qwen3-Coder 모델용 파서 레벨 적응
      ↓ 포크 (Inswave)
websquare-ai-agent-qwen  — 웹스퀘어 환경 적응
```

계보는 라이선스 헤더에 그대로 남아 있다. `packages/core/src`의 TypeScript 425개 파일 중 저작권 표기 분포는 다음과 같다.

| 저작권 표기 | 파일 수 | 의미 |
| --- | ---: | --- |
| `Copyright 2025 Google LLC` | 272 | Gemini CLI 원본 — core의 **약 64%** |
| `Copyright 2026 Google LLC` | 1 | 〃 |
| `Copyright 2025 Qwen` | 86 | qwen-code가 덧댄 층 — **약 32%** |
| `Copyright 2025 Qwen Team` | 32 | 〃 |
| `Copyright 2026 Qwen Team` | 16 | 〃 |
| `Copyright 2025 Qwen Code` | 3 | 〃 |
| (표기 없음) | 15 | — |

인스웨이브 저작권 표기는 core에 하나도 없다 — core 소스를 직접 수정하지 않았기 때문이다.

### 이 포크가 직접 수정한 저장소 루트 파일

- `package.json` — `workspaces`를 core·cli·api-server 셋으로 축소, `prepare`를 비워 husky 훅 비활성화, `postinstall`에 `npm audit fix --audit-level=none` 추가
- `.npmrc` — `audit-level=high`, `min-release-age=7d`(배포 7일 미만 패키지 설치 차단 — 공급망 공격 방어)

core에 대한 변경은 전부 **빌드타임 패치**로 처리한다([§7](#7-빌드타임-패치)). 원본을 건드리지 않으므로 upstream 새 버전을 충돌 없이 당겨올 수 있다.

---

## 2. 런타임 · 빌드

| 항목 | 값 |
| --- | --- |
| Node | `>=20.0.0` (`engines`) |
| 모듈 형식 | ESM (`"type": "module"`) |
| 라이선스 | Apache License 2.0 |
| 레지스트리 | `https://registry.npmjs.org`, `audit-level=high`, `min-release-age=7d` |
| 번들러 | esbuild `^0.24.0` |
| 개발 실행 | `tsx src/server/index.ts` |
| 타입체크 | `tsc --noEmit` (TypeScript `^5.3.3`) |

### esbuild 빌드 옵션

| 옵션 | 값 |
| --- | --- |
| `entryPoints` | `src/server/index.ts` |
| `outfile` | `dist/server.js` |
| `bundle` | `true` |
| `platform` / `target` | `node` / `node20` |
| `format` | `esm` |
| `packages` | `"bundle"` |
| `resolveExtensions` | `.ts .tsx .js .jsx .json` |
| `keepNames` | `true` |
| `minify` | `--production` 일 때만 |
| `sourcemap` | `external` (prod) / `inline` (dev) |
| `banner.js` | `createRequire` shim (ESM 호환) |

`packages: "bundle"` — `file:` 프로토콜 로컬 패키지를 포함한 모든 npm 의존성을 **단일 파일에 인라인**한다.

### npm 스크립트 (api-server)

| 스크립트 | 명령 |
| --- | --- |
| `build` | `node esbuild.config.js` |
| `build:prod` | `node esbuild.config.js --production` |
| `build:watch` | `node esbuild.config.js --watch` |
| `server` / `start` | `tsx src/server/index.ts` |
| `server:prod` | `node dist/server.js` |
| `typecheck` | `tsc --noEmit --project tsconfig.json` |

빌드 절차는 `cleanDist → applyPatches → esbuild → copyPackageJson → restorePatches` 순이며, `finally` 블록에서 **성공·실패 관계없이 core 원본을 복원**한다. CRLF 파일은 LF로 정규화해 매칭한 뒤 원래 줄바꿈으로 되돌려 저장한다.

---

## 3. 패키지 구조

`packages/`에는 10개 디렉터리가 있으나 `workspaces`에 등록된 것은 3개뿐이다. 나머지는 upstream 잔재로 **빌드 대상이 아니다.**

| 패키지 | 상태 | 역할 |
| --- | --- | --- |
| `core` | 활성 | 에이전트 엔진 — 루프, 도구, 프로바이더, MCP. `@qwen-code/qwen-code-core` 0.12.0 |
| `api-server` | 활성 | 이 포크의 산출물 — Express + Socket.IO 서버. 0.1.0 |
| `cli` | 비활성 | workspaces에 있으나 **api-server가 참조하지 않는다** (의존성에 없음) |
| `sdk-java` | 제외 | workspaces 미등록 — upstream 잔재 |
| `sdk-typescript` | 제외 | 〃 |
| `test-utils` | 제외 | 〃 |
| `vscode-ide-companion` | 제외 | 〃 |
| `web-templates` | 제외 | 〃 |
| `webui` | 제외 | 〃 |
| `zed-extension` | 제외 | 〃 |

### api-server 소스 전체 (4파일)

| 파일 | 줄 | 상태 | 역할 |
| --- | ---: | --- | --- |
| `server/qwen-server.ts` | 930 | 활성 | 세션 초기화, 설정 병합, 메시지 처리, 도구 승인, 모델 전환 |
| `server/index.ts` | 270 | 활성 | Express·Socket.IO 부트스트랩, 연결 핸들러, 디렉터리 설정 이벤트 |
| `services/client-file-system-service.ts` | 241 | 비활성 | 클라이언트 위임 쓰기([§6](#6-파일-쓰기-경로)) — 배선이 주석 처리됨 |
| `adapters/config-adapter.ts` | 43 | 활성 | ProWorks Studio 프록시 자격증명 어댑터 |

---

## 4. 소켓 · HTTP API

| 항목 | 값 |
| --- | --- |
| 전송 | Socket.IO `^4.7.2` over HTTP |
| 포트 | `process.env.PORT` 또는 `3001` |
| CORS | `origin: '*'`, `methods: ['GET','POST']` |
| 인증 | **없음** — `io.use()` 미들웨어·토큰 검증 없음 |
| HTTP 라우트 | `GET /health` → `{ status: "ok" }` — 유일한 엔드포인트 |
| 바디 파서 | `express.json()` |

### 세션 개시 경로 (2가지)

| 경로 | 방식 |
| --- | --- |
| A · 웹 UI | handshake query → 즉시 `initializeSession` |
| B · Eclipse/IDE | `initialize_session` 이벤트 → `initializeSession` |

핸드셰이크 쿼리 파라미터는 `workspacePath`, `deploySource`, `metadataPath` 세 개다. 이벤트 방식은 여기에 `globalDirPath`, `projectDirPath`가 추가되며, 값이 없으면 쿼리값으로 폴백한다.

### 수신 이벤트 (12)

| 이벤트 | 페이로드 / 동작 |
| --- | --- |
| `initialize_session` | `workspacePath`, `deploySource`, `metadataPath`, `globalDirPath`, `projectDirPath` — 세션 구성 |
| `set_global_dir` | `globalDirPath`, `reinitializeSession?` — 경로 resolve 후 `mkdirSync` 보장 |
| `set_project_dir` | `projectDirPath`, `reinitializeSession?` |
| `get_global_dir` | 현재 global 디렉터리 조회 |
| `get_project_dir` | 현재 project 디렉터리 조회 |
| `user_message` | 사용자 메시지 전달 → 에이전트 루프 기동 |
| `tool_approval_response` | 도구 실행 승인/거부 응답 |
| `change_llm` | `authType`, `apiKey`, `model`, `baseUrl`, `modelMaxTokens`, `maxOutputTokens`, `studioConfig` |
| `get_llm_config` | 현재 모델 설정 조회 |
| `search_files` | 파일 검색 요청 |
| `cancel_request` | 진행 중 요청 취소 → `AbortSignal` 전파 |
| `disconnect` | `qwenServer.cleanup(socket.id)` — 세션 정리 |

### 송신 이벤트 (29)

| 이벤트 | 출처 | 상태 | 용도 |
| --- | --- | --- | --- |
| `session_initialized` | index | 활성 | 세션 준비 완료 |
| `session_initialization_error` | qwen-server | 활성 | 초기화 실패 |
| `session_reinitialized` | index | 활성 | 디렉터리 변경 후 재초기화 완료 |
| `session_reinitialize_error` | index | 활성 | 재초기화 실패 |
| `global_dir_config` | index | 활성 | global 디렉터리 조회 응답 |
| `global_dir_updated` | index | 활성 | global 디렉터리 변경 통지 |
| `project_dir_config` | index | 활성 | project 디렉터리 조회 응답 |
| `project_dir_updated` | index | 활성 | project 디렉터리 변경 통지 |
| `llm_config` | index | 활성 | 현재 모델 설정 응답 |
| `llm_changed` | qwen-server | 활성 | 모델 전환 완료 |
| `llm_change_error` | qwen-server | 활성 | 모델 전환 실패 |
| `ai_message_start` | qwen-server | 활성 | 응답 스트림 시작 |
| `ai_message_chunk` | qwen-server | 활성 | 토큰 단위 스트리밍 |
| `ai_message_end` | qwen-server | 활성 | 응답 스트림 종료 |
| `ai_message_retry` | qwen-server | 활성 | 재시도 발생 |
| `ai_message_cancelled` | qwen-server | 활성 | 사용자 취소 반영 |
| `ai_error` | qwen-server | 활성 | 모델 호출 오류 |
| `tool_call_start` | qwen-server | 활성 | **이 포크가 추가한 이벤트** — 도구 호출 시작 실시간 표시 |
| `tool_call_request` | qwen-server | 활성 | 도구 실행 요청 |
| `tool_call_complete` | qwen-server | 활성 | 도구 실행 완료 |
| `tool_call_error` | qwen-server | 활성 | 도구 실행 오류 |
| `tool_generating` | qwen-server | 활성 | 도구 인자 생성 중 |
| `tool_approval_request` | qwen-server | 활성 | 승인 요청 — `awaiting_approval` 상태에서 발화 |
| `chat_compressed` | qwen-server | 활성 | 히스토리 압축 발생 (현 설정에선 미발화) |
| `at_command_resolved` | qwen-server | 활성 | `@` 파일 참조 해석 결과 |
| `file_search_results` | qwen-server | 활성 | 파일 검색 결과 |
| `apply_diff_request` | client-fs | 비활성 | 클라이언트에 diff 적용 요청([§6](#6-파일-쓰기-경로)) |
| `diff_context_info` | client-fs | 비활성 | diff 통계 (hunks/added/deleted/ratio) |
| `file_created_notification` | client-fs | 비활성 | 신규 파일 생성 통지 |

동적 이벤트 하나 더 — `apply_diff_response_{requestId}`를 `socket.once`로 수신한다(비활성).

---

## 5. 세션 · 경로 · 권한

### 설정 병합 순서

```
1. globalDirPath/settings.json
2. projectDirPath/settings.json   ← 우선
3. 서버 강제 오버라이드            ← 최종
```

`projectDirPath` 기본값은 `{workspace}/deepsquare`이며, `PROJECT_DIR_PATH` 환경변수로도 지정 가능하다. 세션마다 다른 설정·MCP 세트를 붙일 수 있는 **멀티 클라이언트 / 멀티 프로젝트** 구조다.

### 서버가 강제하는 오버라이드

| 설정 | 강제값 | 이유 |
| --- | --- | --- |
| `context.fileName` | `["DeepSquare.md","AGENTS.md"]` | 기본 `QWEN.md` 대체 (미설정 시에만) |
| `telemetry.enabled` | `false` | `installation_id` 파일 생성 방지 |
| `privacy.usageStatisticsEnabled` | `false` | 〃 |
| `security.folderTrust.enabled` | `false` | `isTrustedFolder()`가 false면 MCP 탐색이 통째로 스킵되므로 |
| `argv.promptInteractive` | `"true"` | 대화형 모드 고정 |
| `argv.approvalMode` | `"default"` | 도구 실행마다 사용자 승인 |
| authType 폴백 | `"openai"` | `security.auth.selectedType` 미지정 시 |

### 쓰기 허용 경로

세션마다 두 곳으로 못박는다. **읽기는 워크스페이스 전체, 쓰기는 이 두 경로 밖으로 못 나간다.**

```
{workspace}/{deploySource}   // 예: WebContent
{workspace}/deepsquare
```

### 메타데이터 저장 레이아웃

upstream의 `~/.qwen` 대신 `metadataPath` 하위 `.deepsquare`를 쓴다. `QWEN_HOME` 환경변수도 같은 경로로 설정된다.

```
{metadataPath}/.deepsquare/
├── todos/          // getTodosDir()
├── projects/       // getProjectDir() — sanitizeCwd 기반 프로젝트 ID
├── skills/         // skill-manager (현재 비어 있음)
├── settings.json
└── installation_id
```

---

## 6. 파일 쓰기 경로

> **현재 상태**: `ClientFileSystemService`의 배선이 `qwen-server.ts:356-359`에서 전부 주석 처리되어 있다. 현재는 core의 표준 `FileSystemService`가 디스크에 직접 쓴다. 아래는 241줄짜리 구현체가 **활성화되면** 동작할 설계다.

### 설계된 분기

| 상황 | 처리 | 결과 |
| --- | --- | --- |
| 읽기 | 항상 로컬 `fallback.readTextFile` | 서버에서 직접 읽음 |
| 신규 파일 (`ENOENT`) | 로컬 `fallback.writeTextFile` | `file_created_notification` 발화 |
| 기존 파일 수정 | unified diff 생성 → 클라이언트에 위임 | 서버는 디스크에 쓰지 않음 |
| 내용 동일 | CRLF 정규화 후 `trim()` 비교 | 중복 적용 스킵 (Eclipse가 이미 적용한 경우) |

### 기존 파일 수정 시퀀스

```
1. Diff.createPatch (context: 3, ignoreWhitespace)
2. diff_context_info 통계 발화
3. apply_diff_request + requestId
4. apply_diff_response_{requestId} 대기
5. resolve / reject — 타임아웃 30초
```

- **중복 쓰기 방지** — `pendingWrites: Map<filePath, Promise>`로 같은 파일의 진행 중 쓰기를 대기시킨다.
- **requestId 형식** — `apply-diff-{Date.now()}-{random36}`
- **diff 통계** — hunks, contextLines, addedLines, deletedLines, contextRatio, diffSize
- **인코딩** — `writeTextFile(path, content, { bom?, encoding? })`, `readTextFileWithInfo()`가 `{ content, encoding, bom }` 반환. 구체 인코딩 판별은 core에 위임한다.

---

## 7. 빌드타임 패치

core 소스를 **일시적으로 find/replace 패치 → 번들링 → 원복**한다. 대상 **9파일 20건** 전량이다.

| 대상 (core/src) | 건 | 패치 |
| --- | ---: | --- |
| `config/storage.ts` | 2 | metadataPath 필드 + `getMetadataQwenDir()` / `getTodosDir()` 추가 · `getProjectDir()`를 metadataPath 기반으로 |
| `config/config.ts` | 2 | `readonly storage` → 재할당 가능 · `setMetadataPath()`, `setGlobalConfigDir()`, `getGlobalConfigDir()`, `getStorage()` 추가 |
| `tools/todoWrite.ts` | 1 | `QWEN_HOME` 환경변수 우선 사용 |
| `skills/skill-manager.ts` | 1 | `globalConfigDir` 경로 우선 사용 |
| `core/turn.ts` | 4 | `GeminiEventType.ToolCallStart` 추가 · `ToolCallStartInfo` 타입 · 이벤트 union 확장 · `run()`에 `functionCallStart` 감지 |
| `core/anthropicContentGenerator/anthropicContentGenerator.ts` | 2 | `content_block_start`에서 `functionCallStart` yield · `buildGeminiChunk` 파라미터 확장 |
| `core/openaiContentGenerator/streamingToolCallParser.ts` | 6 | `isNewToolStart` 필드 추가 · `addChunk` 감지 · 4개 return 경로(success / repaired / error / incomplete)에 전파 |
| `core/openaiContentGenerator/converter.ts` | 1 | `addChunk` 반환값으로 `functionCallStart` emit |
| `core/prompts.ts` | 1 | 시스템 프롬프트 브랜딩 치환([§12](#12-프롬프트--컨텍스트)) |

20건 중 13건이 `ToolCallStart` 스트리밍 이벤트 하나를 배관하는 데 쓰인다.

> **구조적 취약점**: 패치 실패 시 `console.warn`만 출력하고 **빌드는 그대로 성공한다.** upstream에서 대상 코드가 한 글자만 바뀌어도 해당 기능이 조용히 빠진 번들이 나온다. 업스트림 머지 때마다 `[patch-core] Total patches applied: 20` 확인이 필요하다.

---

## 8. 설정 스펙

설정 디렉터리가 저장소에 셋 있다. 스키마 버전만 다르고 내용은 사실상 동일하다.

| 디렉터리 | `$version` | MCP | 비고 |
| --- | :---: | --- | --- |
| `.deepsquare/` | 2 | websquare-mcp | 실사용 설정. `debug/` 94개 로그, 빈 `skills/` |
| `.test/` | 3 | 없음 | `DeepSquare.md` 예제 동봉 |
| `.others/` | 2 | 없음 | 빈 `skills/` |

### settings.json 전 필드

| 키 | 값 | 효과 |
| --- | --- | --- |
| `tools.core` | 8종 배열 | 도구 화이트리스트([§9](#9-도구-스펙)) |
| `useSmartEdit` | `true` | 스마트 편집 모드 |
| `model.chatCompression.contextPercentageThreshold` | `0` | **압축 OFF** — `threshold <= 0`이면 `chatCompressionService.ts:95`에서 즉시 NOOP |
| `mcpServers.websquare-mcp` | stdio | [§14](#14-mcp) |
| `$version` | `2` / `3` | 스키마 버전 |

> **컨텍스트 전략**: 압축을 완전히 껐다는 건 컨텍스트 관리가 "검색해서 좁히기"도 "요약해서 줄이기"도 아니라 **1M 윈도우에 전부 밀어넣고 버티기**라는 뜻이다. 웹스퀘어 XML 화면 소스와 40KB 테스트 가이드를 통째로 넣어야 하는 작업 성격과 맞물린 선택으로 읽히나, 긴 세션에서는 컨텍스트 한계에 그대로 부딪힌다.

---

## 9. 도구 스펙

도구는 세 층으로 갈린다 — core가 **정의**한 것, core가 **등록**하는 것, 웹스퀘어 배포가 **허용**한 것.

### ToolNames 상수 (16)

| 도구 | 표시명 | 등록 | 허용 | 비고 |
| --- | --- | :---: | :---: | --- |
| `list_directory` | ListFiles | O | **O** | |
| `read_file` | ReadFile | O | **O** | |
| `grep_search` | Grep | O | **O** | ripgrep 있으면 `RipGrepTool`, 없으면 `GrepTool` — 이름은 동일 |
| `glob` | Glob | O | **O** | |
| `edit` | Edit | O | **O** | 레거시명 `replace` |
| `write_file` | WriteFile | O | **O** | |
| `todo_write` | TodoWrite | O | **O** | |
| `run_shell_command` | Shell | O | — | 의도적 제외 — 원격 에이전트의 임의 명령 실행 차단 |
| `task` | Task | O | — | 서브에이전트 위임 — **기능 OFF** |
| `skill` | Skill | O | — | skills 디렉터리도 비어 있음 |
| `save_memory` | SaveMemory | O | — | |
| `web_fetch` | WebFetch | O | — | |
| `web_search` | WebSearch | △ | — | 프로바이더 설정 시에만 등록 |
| `lsp` | Lsp | △ | — | 조건부 등록 |
| `ask_user_question` | AskUserQuestion | O | — | |
| `exit_plan_mode` | ExitPlanMode | △ | — | SDK 모드에선 미등록 |

> **화이트리스트 8종 중 1종은 무효**
>
> `settings.json`의 `tools.core`에 `read_many_files`가 들어 있으나, **`ReadManyFilesTool`이라는 도구는 존재하지 않는다.** `readManyFiles`는 `utils/readManyFiles.ts`의 내부 유틸 함수일 뿐이고 `config.ts:1859-1907`의 등록 목록에도 없다.
>
> `registerCoreTool`은 `isToolEnabled()`로 등록 시점에 거르는 구조라 목록에 없는 이름은 **조용히 무시**된다 — 에러도 경고도 없다. 따라서 에이전트가 실제로 쥐는 도구는 **8종이 아니라 7종**이다.

### 승인 플로우

```
validating → awaiting_approval → scheduled → executing → success / error / cancelled
                    ↓
            tool_approval_request  →  클라이언트
                    ↑
            tool_approval_response ←  사용자 확인
```

`coreToolScheduler.ts`(1,406줄)의 명시적 상태 머신이다. human-in-the-loop이 프레임워크 기능이 아니라 상태 머신에 직접 박혀 있다. 유일한 우회로는 MCP의 `trust: true`다.

---

## 10. 모델 · 프로바이더

Qwen 전용이 아니다. **모델은 런타임에 클라이언트가 지정하는 값**이다.

### AuthType (5)

| 값 | 상수 | 대상 | ToolCallStart 패치 |
| --- | --- | --- | :---: |
| `openai` | `USE_OPENAI` | OpenAI 및 **OpenAI 호환 API 전부** — 기본 폴백 | O |
| `anthropic` | `USE_ANTHROPIC` | Claude | O |
| `qwen-oauth` | `QWEN_OAUTH` | Qwen OAuth (무료 티어) | — |
| `gemini` | `USE_GEMINI` | Google Gemini | — |
| `vertex-ai` | `USE_VERTEX_AI` | Google Vertex AI | — |

패치가 OpenAI·Anthropic 두 경로에만 들어간 것은 **이 둘이 실사용 전제**라는 신호다.

### ProWorks Studio 프록시

`config-adapter.ts`가 Eclipse IDE가 넘기는 `StudioServerConfig`를 처리한다.

| 필드 | 타입 | 용도 |
| --- | --- | --- |
| `baseUrl` | string | 프록시 엔드포인트 |
| `apiKey` | string | 폴백 자격증명 |
| `model` | string | 모델 식별자 문자열 |
| `authType` | string | 프로바이더 선택 |
| `proworksTKey` | string? | **실효 apiKey** — 있으면 `apiKey`보다 우선 |
| `proworksBody` | string? | ProWorks 전달 파라미터 |
| `proworksLang` | string? | ProWorks 전달 파라미터 |

코드 주석이 설계 의도를 밝힌다 — *"The proworksTKey is used as the apiKey since the studio's admin server acts as a proxy and uses this token for authentication."* 즉 에이전트는 모델 벤더에 직접 붙지 않는다. **인스웨이브 ProWorks/Studio 관리 서버가 프록시로 서고, 에이전트는 거기에 `proworksTKey`를 들고 붙는다.** 개별 개발자 PC에 API 키를 뿌릴 필요가 없고, 모델 선택·과금·감사를 한 곳에서 통제할 수 있다.

적용은 `config.updateCredentials({ apiKey, baseUrl, model })` 한 번이다. `proworksBody`·`proworksLang`은 인터페이스에만 선언돼 있고 `applyStudioConfig()`에서 사용되지 않는다.

### 런타임 모델 전환

`change_llm` 핸들러(`qwen-server.ts:872`)가 세션 중 모델을 교체한다. `authType`이 오면 `refreshAuth()`로 프로바이더 자체를 갈아끼운다. 기본 모델명은 `coder-model`이라는 **추상 이름**으로, 실제 모델 ID가 아니라 프록시가 해석하도록 열어 둔 값이다.

---

## 11. 토큰 · 모달리티 레지스트리

모델별 파라미터 수나 벤더 버전 문서는 없다. 대신 모델명을 **정규식으로 매칭**해 한계값을 결정한다. 위에서부터 첫 매칭이 이긴다.

### 컨텍스트 윈도우 · 최대 출력

| 패턴 | 컨텍스트 | 출력 |
| --- | ---: | ---: |
| `gemini-3*` | 1M | 64K |
| `gemini-*` | 1M | 8K |
| `gpt-5*` | 400K | 128K |
| `o{숫자}*` | 200K | 128K |
| `gpt-*` | 128K | 16K |
| `claude-opus-4-6` | 200K | 128K |
| `claude-sonnet-4-6` | 200K | 64K |
| `claude-*` | 200K | 64K |
| `coder-model` | 1M | 64K |
| `qwen3-coder-plus` / `-flash` | 1M | — |
| `qwen3.5-plus` | 1M | 64K |
| `qwen-plus-latest` / `qwen-flash-latest` | 1M | — |
| `qwen3-max*` | 256K | 64K |
| `qwen3-coder-*` / `qwen*` | 256K | — |
| `deepseek-reasoner` | 128K | 64K |
| `deepseek-chat` | 128K | 8K |
| `glm-5*` | 202,752 | 16K |
| `glm-4.7*` | 202,752 | 16K |
| `glm-*` | 202,752 | — |
| `minimax-m2.5` | 1M | 64K |
| `minimax-*` | 200K | — |
| `kimi-k2.5` | 256K | 32K |
| `kimi-*` | 256K | — |
| `seed-oss*` | 512K | — |
| **미매칭 기본값** | **131,072** | **8,192** |

`tokenLimits.ts:80-181`. 출력 열의 `—`는 별도 패턴이 없어 기본값 8K로 떨어짐을 뜻한다.

### 입력 모달리티

| 패턴 | image | pdf | audio | video |
| --- | :---: | :---: | :---: | :---: |
| `gemini-*` | O | O | O | O |
| `gpt-*` / `o{숫자}*` | O | · | · | · |
| `claude-*` | O | O | · | · |
| `coder-model` / `qwen3.5-plus` | O | · | · | O |
| `qwen-vl-*` / `qwen3-vl-*` | O | · | · | O |
| `glm-4.5v` | O | · | · | · |
| `kimi-k2.5` | O | · | · | O |
| 그 외 · 미매칭 | · | · | · | · |

`modalityDefaults.ts:20-75`. 미매칭이면 텍스트 전용이다.

### 모델명 정규화

매칭 전 `normalize()`가 버전·날짜 꼬리표를 떼어낸다.

- 프로바이더 접두사 제거 — `openai/gpt-4o` → `gpt-4o`, 파이프·콜론 뒤만 사용
- `-preview`, 날짜(`-20250219`), `-v1.2`, `-latest`, `-exp` 제거
- 파라미터 크기(`-7b`, `-4x8b`) · 양자화 접미사(`-int4`, `-bf16`, `-q5`, `-8bit`) 제거
- **예외** — 버전이 곧 정체성인 `qwen-plus-latest`, `qwen-flash-latest`, `qwen-vl-max-latest`, `kimi-k2-{4자리}`는 유지
- `gpt-4.1`의 `4.1`은 앞 대시가 하나뿐이라 버전으로 잘리지 않는다

### 샘플링 파라미터

모델별 기본값 테이블은 **없다.** `contentGenerator.ts:87-94`에 스키마만 있고 전부 옵셔널이다 — `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`, `temperature`, `max_tokens`. 값은 벤더 기본값에 맡기거나 `change_llm`으로 넘긴다(`modelMaxTokens` → `contextWindowSize`, `maxOutputTokens` → `samplingParams.max_tokens`).

> **폴백 주의**: 이 테이블들은 **자동 감지 폴백일 뿐**이다. ProWorks 프록시가 넘기는 `model` 문자열이 어느 패턴에도 안 걸리면 **128K / 8K / 텍스트 전용**으로 떨어진다. 커스텀 모델명을 쓴다면 클라이언트가 `modelMaxTokens`·`maxOutputTokens`를 명시적으로 넘기는 편이 안전하다.

---

## 12. 프롬프트 · 컨텍스트

파인튜닝과 RAG는 쓰지 않는다. 프롬프트 엔지니어링만 실제로 작업돼 있고, 그 본체는 **컨텍스트 파일**이다. 상세 감사는 [prompt-rag-finetuning-audit.md](prompt-rag-finetuning-audit.md) 참조.

### 프롬프트 작업 파일 (3)

| 파일 | 크기 | 역할 |
| --- | --- | --- |
| `.test/DeepSquare.md` | 40,746 B / 1,169줄 | **도메인 지식 본체** — TestSquare E2E 테스트 XML 생성 가이드 11장 |
| `esbuild.config.js:517-525` | 1건 | 시스템 프롬프트 브랜딩 치환 |
| `server/qwen-server.ts:271-277` | — | 컨텍스트 파일명 교체 배선 |

### 시스템 프롬프트 변경 — 전량

```
find:    "You are Qwen Code, an interactive CLI agent developed by Alibaba Group"
replace: "You are AI Talk Plus, an interactive CLI agent developed by Inswave"
```

55,957 B짜리 `core/prompts.ts`에 가한 변경은 이 한 줄이 전부다. **웹스퀘어 도메인 지식은 시스템 프롬프트에 전혀 들어가지 않는다.**

### 주입 경로

```
{globalDirPath}/DeepSquare.md  ─┐
{projectDirPath}/DeepSquare.md ─┼→ MEMORY_DISCOVERY → user memory → 시스템 프롬프트 뒤 append
{...}/AGENTS.md                ─┘
                                  ↑ 매 세션, 질의와 무관하게 전문 삽입
```

### DeepSquare.md 목차

| 장 | 제목 |
| ---: | --- |
| 1 | 출력 형식: XML (JS가 아님) |
| 2 | 웹스퀘어(WebSquare)란 |
| 3 | 웹스퀘어 SPA의 ID 문제 — 문서가 "가장 중요"로 표기 |
| 4 | 웹스퀘어 컴포넌트별 렌더링 특성과 주의사항 |
| 5 | Node.js 환경 주의사항 |
| 6 | 공통 헬퍼 함수 (CDATA 안에 인라인 선언) |
| 7 | 입력 파일 설명 — `interface-metadata.json`, `test-plan.md` |
| 8 | XML 메타데이터 생성 규칙 |
| 9 | 시나리오별 코드 패턴 |
| 10 | 실전 예시: BM002M01 전체 XML |
| 11 | 생성 시 체크리스트 |

핵심 기술 사실 — 화면 URL 규약 `/websquare/websquare.html?w2xPath=...`, 네임스페이스 `xmlns:w2="http://www.inswave.com/websquare"`, `page.evaluate` 안에서 prefix를 붙여 `window.WebSquare`에 접근.

> **저장소본 ≠ 배포본**
>
> 저장소의 `DeepSquare.md`는 `.test/` 아래 예제다. 런타임 로그를 보면 실제 로드된 파일은 `.deepsquare/DeepSquare.md`이고 **크기가 20,654 B**로 저장소본 40KB와 다르다. **실 운영 프롬프트 본문은 이 체크아웃에 없다.**
>
> 같은 로그에 저장소에 없는 프롬프트 리소스가 더 보인다 — `WebSquareSchema.xml`, `websquare_docs/`. 당시 `dsFs` 오버레이가 주입했으나 현재 소스엔 `index.ts:225`의 `// dsFs wrapper removed — standard fs used` 주석만 남았다.

### RAG · 파인튜닝 — 부재 근거

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| 파인튜닝 | 없음 | 학습 스크립트·데이터셋(`.jsonl`)·가중치(`.safetensors`)·노트북·`requirements.txt` 매칭 0건 |
| 벡터 저장소 | 없음 | faiss·chroma·pinecone·qdrant·weaviate·milvus 의존성 0건 |
| 임베딩 | 미사용 | `baseLlmClient.ts:157`에 `generateEmbedding()`이 있으나 **프로덕션 호출부 0건** — 정의 1건 + 테스트뿐. 청킹·인덱싱·유사도 계산 없음 |
| 기본 임베딩 모델 | 미사용 | `text-embedding-v4`. Anthropic 경로는 `throw new Error('Anthropic does not support embeddings.')`, OpenAI 경로는 하드코딩 `text-embedding-ada-002` |
| 대체 방식 | 에이전틱 검색 | 벡터 유사도 대신 모델이 `glob` → `grep_search` → `read_file`을 스스로 호출. 허용 도구 조합이 정확히 이것 |

---

## 13. 에이전트 루프

LangGraph·LangChain·LlamaIndex·AutoGen·CrewAI **어느 것도 쓰지 않는다.** `packages/core/package.json`에 에이전트 프레임워크 의존성이 하나도 없다. 루프는 직접 구현돼 있고 핵심 파일이 약 4,365줄이다.

| 파일 | 줄 | 역할 |
| --- | ---: | --- |
| `core/coreToolScheduler.ts` | 1,406 | 도구 실행 상태 머신 |
| `subagents/subagent.ts` | 1,010 | 서브에이전트 실행기 |
| `core/client.ts` | 785 | 메인 에이전트 루프 |
| `core/geminiChat.ts` | 746 | 대화 히스토리 관리 |
| `core/turn.ts` | 418 | 턴 단위 이벤트 정의 |

### 제어 흐름 — async generator 재귀

그래프·노드·엣지 대신 **비동기 제너레이터의 재귀 호출**을 쓴다. `client.ts:413`의 `async *sendMessageStream()`이 중심이며, 이벤트를 `yield`로 흘리다가 대화를 이어야 하면 자기 자신을 `yield*`로 재귀 호출한다.

```js
return yield* this.sendMessageStream(nextRequest, signal, prompt_id, options, boundedTurns - 1);
```

`boundedTurns`가 재귀 깊이 제한 — LangGraph의 `recursion_limit`에 해당하는 역할을 인자 하나로 처리한다. 계속 여부 판단은 두 지점이다.

- 도구 호출이 남아 있으면 → 실행 후 결과를 넣고 재귀
- 도구 호출이 없으면 → `checkNextSpeaker()`로 **LLM에게 다음 차례를 물어보고**, `model`이면 `"Please continue."`를 넣어 재귀

### 상태 전달 — GeminiEventType (15 + 1)

```
content · thought · tool_call_request · tool_call_response · tool_call_confirmation
user_cancelled · error · chat_compressed · max_session_turns
session_token_limit_exceeded · finished · loop_detected · citation · retry
hook_system_message
+ ToolCallStart  ← 이 포크가 추가
```

상태 객체 대신 **타입 태그가 붙은 이벤트 유니온**이 흐른다. api-server는 이 스트림을 그대로 소켓으로 중계한다 — 제너레이터가 곧 스트리밍 파이프라인이다.

### 프레임워크 없이 만든 안전장치

| 기능 | 구현 | 웹스퀘어 배포 |
| --- | --- | --- |
| 무한루프 방지 | `LoopDetectionService` + `loop_detected` | 활성 |
| 중단 | `AbortSignal` → `cancel_request` | 활성 |
| 재시도 | `retry` 이벤트 + 백오프 | 활성 |
| 동시성 | `async-mutex` | 활성 |
| 훅 | `Stop` 훅이 종료를 막고 강제 계속 | 활성 |
| 스키마 검증 | `ajv` + `ajv-formats` | 활성 |
| 컨텍스트 압축 | `chatCompressionService.ts` | **비활성** |
| 관측 | OpenTelemetry | **비활성** |
| 서브에이전트 위임 | `task` 도구 + `subagent.ts` | **비활성** |

서브에이전트는 `builtin-agents.ts`에 `general-purpose` 하나만 내장돼 있고 나머지는 사용자가 마크다운으로 정의한다. 구조상 **부모-자식 위임**이지 임의 그래프가 아니다.

---

## 14. MCP

> **범위**: MCP 서버 구현체는 **이 저장소에 없다.** 저장소에는 호출하는 클라이언트 쪽 배선만 있다.

```json
"websquare-mcp": {
  "command": "node",
  "args": ["./dist/websquare-mcp.mjs"],
  "cwd": "/Users/fujii0711/Develop/PaaS/src/deepsquare-mcp",
  "trust": true
}
```

| 항목 | 값 | 함의 |
| --- | --- | --- |
| 전송 | stdio | `url`/`httpUrl`이 아니므로 `StdioClientTransport` 경로 — 자식 프로세스 표준입출력 |
| `cwd` | 개발자 개인 macOS 경로 | **확인필요** — 저장소에도 없고 다른 머신에도 없다. **현재 상태로는 연결 실패** |
| `trust` | `true` | **확인필요** — 이 서버의 도구는 **승인 없이 즉시 실행**. `approvalMode: "default"` 승인 플로우의 유일한 우회로 |
| SDK | `@modelcontextprotocol/sdk ^1.25.1` | 도구·프롬프트 discovery 후 동일 레지스트리 합류, 헬스체크 타이머로 재연결 |

### 역할 (정황 근거)

에이전트에 허용된 core 도구는 파일 조작 7종뿐이다. 즉 **파일을 읽고 쓰는 것 외의 모든 능력은 MCP를 통해서만** 들어온다. 그리고 `DeepSquare.md` 7장이 요구하는 입력은 파일시스템을 뒤져서 나오는 게 아니다.

- `interface-metadata.json` — 화면의 `collections`(DataList/DataMap 정의와 필드), `submissions`(API URL과 요청/응답 컬렉션), `popups`, `messages`. 웹스퀘어 XML 화면을 파싱해 구조화한 산출물
- `test-plan.md` — 화면 ID, 시나리오 목록, 우선순위, 관련 API 엔드포인트

`screenId`(예: `BM002M01`)로 화면을 조회해 데이터 모델·서브미션 구조를 뽑는 일은 웹스퀘어 IDE/PaaS 플랫폼이 쥔 메타데이터다. `websquare-mcp`가 그 게이트웨이 역할을 하는 것으로 읽는 게 자연스럽다. **다만 서버 구현이 없어 실제 도구 목록은 확인할 수 없다.**

---

## 15. 의존성

### api-server (런타임 5 · 개발 5)

| 패키지 | 버전 | 용도 |
| --- | --- | --- |
| `@qwen-code/qwen-code-core` | `file:../core` | 에이전트 엔진 |
| `express` | `^4.18.2` | HTTP 서버 |
| `socket.io` | `^4.7.2` | WebSocket |
| `diff` | `^7.0.0` | unified diff 생성([§6](#6-파일-쓰기-경로)) |
| `glob` | `^10.3.10` | 파일 검색 |
| `esbuild` · `tsx` · `typescript` · `@types/node` · `@types/express` | dev | 빌드·타입 |

### core (48개) — 분류

| 분류 | 패키지 |
| --- | --- |
| LLM 벤더 SDK | `@anthropic-ai/sdk ^0.36.1` · `openai 5.11.0` · `@google/genai 1.30.0` |
| 프로토콜 | `@modelcontextprotocol/sdk ^1.25.1` |
| 관측 | `@opentelemetry/*` 8종 (api, sdk-node, instrumentation-http, exporter × logs/metrics/trace × grpc/http) |
| 인증·네트워크 | `google-auth-library ^10.5.0` · `undici ^6.22.0` · `https-proxy-agent ^7.0.6` · `ws ^8.18.0` |
| 스키마·파싱 | `ajv ^8.17.1` · `ajv-formats ^3.0.0` · `jsonrepair ^3.13.0` · `@iarna/toml ^2.2.5` · `marked ^15.0.12` · `html-to-text ^9.0.5` |
| 파일·검색 | `glob ^10.5.0` · `fdir ^6.4.6` · `picomatch ^4.0.1` · `ignore ^7.0.0` · `chokidar ^4.0.3` · `fzf ^0.5.2` · `fast-levenshtein ^2.0.6` |
| 인코딩 | `chardet ^2.1.0` · `iconv-lite ^0.6.3` · `mime 4.0.7` · `fast-uri ^3.0.6` |
| 터미널·셸 | `@xterm/headless 5.5.0` · `shell-quote ^1.8.3` · `strip-ansi ^7.1.0` · `prompts ^2.4.2` · `open ^10.1.2` |
| 기타 | `async-mutex ^0.5.0` · `diff ^7.0.0` · `dotenv ^17.1.0` · `simple-git ^3.28.0` · `tar ^7.5.2` · `extract-zip ^2.0.1` · `mnemonist ^0.40.3` · `uuid ^9.0.1` |

에이전트 프레임워크(LangChain 계열) 의존성은 **0건**이다.

---

## 16. 비활성 인벤토리

이 코드베이스의 두드러진 특징 하나 — **선언돼 있으나 실제로는 동작하지 않는 것이 많다.** 스펙을 읽을 때 가장 오해하기 쉬운 지점이므로 따로 모았다.

| 대상 | 상태 | 왜 | 영향 |
| --- | --- | --- | --- |
| `read_many_files` | 미등록 | 화이트리스트에 있으나 도구 자체가 없음 — `readManyFiles`는 내부 유틸 함수 | 실효 도구 8종 → **7종** |
| `ClientFileSystemService` | 비활성 | `qwen-server.ts:356-359` 배선 주석 처리 | 241줄 사문화. 소켓 이벤트 3개도 함께 미발화 |
| `task` 도구 / 서브에이전트 | 비활성 | 화이트리스트 미포함 | 위임·병렬 처리 불가 |
| `skill` 도구 | 비활성 | 화이트리스트 미포함 + `skills/` 디렉터리 비어 있음 | 스킬 확장 경로 없음 |
| `chatCompression` | 비활성 | `contextPercentageThreshold: 0` → 즉시 NOOP | 긴 세션에서 컨텍스트 한계 직면 |
| `generateEmbedding()` | 미사용 | 프로덕션 호출부 0건 — Gemini CLI 유산 | 동작상 문제 없음 |
| OpenTelemetry | 비활성 | 서버가 `telemetry.enabled=false` 강제 | 운영 관측 지표 없음 |
| `dsFs` 오버레이 | 제거됨 | `index.ts:225` 주석만 잔존 | `WebSquareSchema.xml`·`websquare_docs/` 주입 경로 소실 |
| `packages/cli` | 미참조 | workspaces엔 있으나 api-server 의존성에 없음 | 빌드 시간만 소비 |
| `proworksBody` · `proworksLang` | 미사용 | 인터페이스 선언만, `applyStudioConfig()`에서 미참조 | 클라이언트가 넘겨도 무시됨 |
| `websquare-mcp` | **연결 실패** | `cwd`가 존재하지 않는 개인 경로 | **MCP 능력 전부 사용 불가** |

---

## 17. 운영 리스크

| # | 항목 | 위치 | 내용 |
| ---: | --- | --- | --- |
| 1 | **소켓 무인증** | `server/index.ts` | `io.use()` 미들웨어가 없고 CORS가 `origin:'*'`. 포트에 닿는 누구나 세션을 열고 워크스페이스를 읽고 허용 경로에 쓸 수 있다. 네트워크 계층에서 반드시 격리해야 한다 |
| 2 | MCP `cwd` 개인 경로 | `.deepsquare/settings.json` | `/Users/fujii0711/...` — 배포 환경 경로로 교체 필요. 현 상태로는 MCP 연결 실패 |
| 3 | MCP `trust: true` | `.deepsquare/settings.json` | 해당 서버 도구는 승인 없이 실행. 의도된 설정인지 확인 필요 |
| 4 | 패치 실패 무음 | `esbuild.config.js` | 실패해도 빌드 성공. upstream 머지마다 **20건 적용** 확인 필요 |
| 5 | 컨텍스트 압축 비활성 | `settings.json` | 긴 세션에서 한계에 그대로 부딪힘 |
| 6 | 모델명 폴백 | `tokenLimits.ts` | 프록시 커스텀 모델명은 128K/8K/텍스트전용으로 떨어짐. `modelMaxTokens`·`maxOutputTokens` 명시 권장 |
| 7 | 화이트리스트 오타 무음 | `config.ts:1846` | `read_many_files`처럼 없는 이름은 경고 없이 무시. 설정 오타를 잡아주지 않는다 |
| 8 | 운영 프롬프트 부재 | `.deepsquare/` | 실제 로드되는 20KB `DeepSquare.md`가 저장소에 없다. 형상관리 대상인지 확인 필요 |
| 9 | 디버그 로그 누적 | `.deepsquare/debug/` | 세션당 파일 1개, 94개 커밋돼 있음. 절대경로·환경정보 포함 — 정리·gitignore 검토 |

---

## 확인 범위

모든 수치와 인용은 `c:\websquare-ai-agent` 체크아웃을 직접 읽어 확인했다. 확인 범위를 벗어난 두 가지는 본문에 명시했다.

- `websquare-mcp` 서버 구현체 (별도 저장소 `deepsquare-mcp`)
- 실 운영 `DeepSquare.md` 본문 (20,654 B, 저장소에 없음)
