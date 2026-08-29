# 운영 리스크 추정

> 기준일: 2026-08-28
> 질문: 현재 프로젝트 시스템을 실제 운영에 투입했을 때 어떤 문제가 발생하는가?

저장소 코드를 직접 읽어 확인한 근거만 담았다. 확인 범위를 벗어난 항목(MCP 서버 구현, 실 배포 `DeepSquare.md`)은 해당 절에 명시했다.

## 요약

| # | 문제 | 심각도 | 조치 난이도 |
| ---: | --- | --- | --- |
| 1 | 인증 없음 + `0.0.0.0` 바인딩 + CORS `*` | **치명** | 낮음 (3줄) |
| 2 | `QWEN_HOME` 전역 오염 → 세션 간 데이터 섞임 | 높음 | 중간 |
| 3 | 압축 OFF → 긴 세션 붕괴 | 높음 | 낮음 (설정값) |
| 4 | 빌드 패치 20건의 조용한 실패 | 높음 | 낮음 (`exit 1`) |
| 5 | 해외 텔레메트리 송신 시도 | 중간 (보안 검토 리스크) | 낮음 |
| 6 | Ollama 폴백으로 인한 진단 불가 실패 | 중간 | 낮음 |
| 7–9 | 쓰기 경로 검사 우회 3종 | 중간 | 중간 |
| 10–12 | 개인 경로 · 미버전관리 프롬프트 · 죽은 설정 | 중간 (운영 대응 저해) | 낮음 |

**1 · 3 · 4번은 각각 몇 줄 수정으로 막을 수 있는데 영향이 크다. 여기부터 손대기를 권한다.**

---

## P0 — 배포 즉시 노출되는 문제

### 1. 소켓 서버에 인증이 없고 전체 인터페이스에 바인딩된다

`packages/api-server/src/server/index.ts:17-22`

```js
const io = new Server(httpServer, { cors: { origin: '*', methods: ['GET','POST'] } });
```

`packages/api-server/src/server/index.ts:262`

```js
httpServer.listen(PORT, () => { console.log(`server running on http://localhost:${PORT}`); });
```

**두 번째 인자가 콜백이라 호스트 지정이 없다 → `0.0.0.0` 바인딩이다.** 로그 문구만 "localhost"라 실제 노출 범위를 오해하기 쉽다. `io.use()` 인증 미들웨어도 없다(grep 0건).

무슨 일이 벌어지나:

- **같은 네트워크의 아무 호스트**가 3001 포트에 붙어 파일 읽기/쓰기 에이전트를 조종할 수 있다
- `origin: '*'` 때문에 **개발자가 방문한 임의의 웹페이지**도 `localhost:3001`로 소켓을 열 수 있다. DNS rebinding도 필요 없다
- `index.ts:33-35`에서 `workspacePath` / `deploySource` / `metadataPath`를 **클라이언트 핸드셰이크 쿼리에서 그대로 받는다.** 즉 접속자가 자기 샌드박스 경계를 스스로 정한다

`isWriteAllowedPath()`는 **모델**이 경계를 벗어나는 걸 막는 장치지, **접속자**를 막는 장치가 아니다. 사내망 배포라도 성립하지 않는 가정이다.

최소 조치 — `listen(PORT, '127.0.0.1')`, `io.use()`에 토큰 검증, `origin`을 Eclipse 플러그인 출처로 제한.

### 2. 동시 세션에서 `QWEN_HOME` 전역 오염

`packages/api-server/src/server/qwen-server.ts:327`

```ts
process.env["QWEN_HOME"] = path.join(resolvedMetadata, ".deepsquare");
```

세션 상태는 전부 `Map<socketId, ...>`로 격리돼 있는데(`qwen-server.ts:67-73`), **이것 하나만 프로세스 전역**이다. 그리고 `packages/api-server/esbuild.config.js:145-152`의 빌드 패치가 `todoWrite`를 이 값을 **호출 시점에** 읽도록 바꿔 놨다.

```js
name: "todoWrite: QWEN_HOME 환경변수 우선 사용",
const todoDir = process.env['QWEN_HOME'] ? path.join(process.env['QWEN_HOME'], TODO_SUBDIR) : ...
```

사용자 A와 B가 서로 다른 `metadataPath`로 동시 접속하면 **나중에 붙은 쪽이 앞 세션의 todo 저장 경로를 덮어쓴다.** A의 작업 목록이 B의 워크스페이스에 기록된다. 서버 프로세스 하나에 여러 IDE가 붙는 순간 재현되고, 로그만 봐서는 원인을 못 찾는 종류의 버그다.

한 명만 붙는다는 전제라면 안전하지만, 그렇다면 애초에 소켓 서버로 만들 이유가 약해진다.

---

## P1 — 운영 중 서서히 드러나는 문제

### 3. 압축이 꺼져 있어 긴 세션이 컨텍스트 한계에 그대로 부딪힌다

`packages/api-server/.deepsquare/settings.json`의 `contextPercentageThreshold: 0` → `packages/core/src/services/chatCompressionService.ts:95`에서 `threshold <= 0`이면 즉시 NOOP.

고정 15K 토큰(아래 참조) 위에 대화·도구 결과가 무한 누적된다. 웹스퀘어 XML 화면 소스는 한 파일이 수천 줄이라 `read_file` 몇 번이면 수만 토큰이다. **한 세션에서 화면 3~4개만 다뤄도 한계에 닿는다.**

증상은 "갑자기 답이 이상해지거나 요청이 실패한다"로 나타나고 사용자는 원인을 모른다. 압축이 꺼져 있으니 우아한 저하도 없다. 1M 컨텍스트 모델 전제로 보이는데, 모델을 바꾸거나 프록시가 컨텍스트를 줄이면 즉시 문제가 된다.

참고 — 요청 앞단 고정 분량

| 구성 | 크기 | 근거 |
| --- | ---: | --- |
| 기본 시스템 프롬프트 | 20,472 B | `core/__snapshots__/prompts.test.ts.snap:1326-1540` |
| userMemory (`DeepSquare.md`) | 20,654 B | 런타임 로그 `(Length: 20654)` — 194회 |
| 도구 선언 7개 | 19,278 B | 각 `super()` 블록 실측 |
| **합계** | **~60,400 B** | **≈ 15,000 토큰** |

도구 선언 중 `todo_write` 하나가 10,248 B로 53%를 차지한다(`packages/core/src/tools/todoWrite.ts:69`, `:435`).

### 4. 빌드타임 패치 20건이 조용히 실패한다

`packages/api-server/esbuild.config.js:543-548`

```js
const patched = content.replace(patch.find, patch.replace);
if (patched === content) {
  console.warn(`[patch-core] WARNING: Patch not applied — "${patch.name}"`);
}
```

**경고만 출력하고 빌드는 성공한다.** 패치는 전부 문자열 리터럴 매칭이라 upstream qwen-code를 머지하면 공백 하나 바뀌어도 깨진다.

깨졌을 때 나타나는 증상:

- 브랜딩 패치 실패 → 모델이 자기를 "Qwen Code, developed by Alibaba Group"이라고 소개
- `ToolCallStart` 이벤트 패치 실패 → IDE 실시간 도구 표시가 조용히 멈춤
- `QWEN_HOME` 패치 실패 → todo가 엉뚱한 곳에 저장

CI에서 빌드는 초록불인데 런타임 동작만 바뀐다. **패치 미적용을 빌드 실패로 승격하는 게 맞다**(`process.exit(1)`).

원복은 `finally`로 보장돼 있어(`esbuild.config.js:619-622`) 그 부분은 괜찮지만, `--watch` 모드에서 프로세스를 강제 종료하면 core 소스가 패치된 채 남는다.

### 5. 폐쇄망에서 외부 텔레메트리 시도가 계속 실패한다

디버그 로그에 **101건**:

```
[ERROR] [QWEN_LOGGER] RUM flush failed. AggregateError [ETIMEDOUT]
```

전송 대상은 `packages/core/src/telemetry/qwen-logger/qwen-logger.ts:69`의 `gb4w8c3ygj-default-sea.rum.aliyuncs.com` — 알리바바 싱가포르 RUM 엔드포인트다.

`qwen-server.ts:279-282`에 비활성화 코드가 있고 게이트는 `qwen-logger.ts:163`의 `getInstance()`에 있으니 **현재 코드에서는 막힐 가능성이 높다.** 다만 로그가 남아 있다는 건 최소한 한 시점에는 나갔다는 뜻이고, `installation_id` 파일도 생성돼 있다.

고객사 보안 검토에서 **"국내 코드가 해외로 나가는가"** 는 반드시 나오는 질문이다. 설정 의존이 아니라 빌드 시점에 엔드포인트 자체를 제거하는 패치를 넣는 편이 방어하기 쉽다. 아웃바운드 차단망에서는 매 flush마다 타임아웃 대기가 걸리는 부작용도 있다.

### 6. `studioConfig` 없이 붙으면 존재하지 않는 Ollama로 간다

`packages/api-server/src/server/qwen-server.ts:87-94`

```ts
if (!process.env["OPENAI_API_KEY"])  process.env["OPENAI_API_KEY"]  = "token-abc123";
if (!process.env["OPENAI_BASE_URL"]) process.env["OPENAI_BASE_URL"] = "http://localhost:11434/v1";
if (!process.env["OPENAI_MODEL"])    process.env["OPENAI_MODEL"]    = "qwen3-coder";
```

Eclipse가 `studioConfig`를 안 넘기거나 필드가 비면 로컬 11434(Ollama)로 붙는다. 운영 서버에 Ollama가 없으므로 **ECONNREFUSED**가 나는데, 사용자에게는 "AI가 응답하지 않음"으로만 보인다. 인증 설정 누락과 네트워크 장애를 구분할 수 없다. 기본값을 넣는 대신 명시적으로 실패시키는 게 낫다.

---

## P2 — 보안 · 정합성 세부

### 7. MCP 도구는 쓰기 경로 검사를 통과한다

`packages/api-server/src/server/qwen-server.ts:41`

```ts
const WRITE_TOOLS = ["edit", "write-file", "smart-edit", "write_file", "write"];
```

이름 부분 일치로 쓰기 도구를 판별한다. `websquare-mcp`가 제공하는 도구가 파일을 쓰더라도 이름이 이 목록에 안 걸리면 **경로 검사를 아예 안 거친다.** 게다가 `settings.json`에서 `trust: true`라 승인 프롬프트도 없다.

> 확인 범위 밖 — MCP 서버 소스가 이 체크아웃에 없어 실제 도구 목록을 확인하지 못했다. 확인이 필요한 지점이다.

### 8. `filePath`를 못 뽑으면 통과된다

`packages/api-server/src/server/qwen-server.ts:756`

```ts
if (filePath && !this.isWriteAllowedPath(socket.id, filePath)) { /* 차단 */ }
```

`filePath`가 `undefined`면 **조건이 거짓이라 그냥 허용**된다. `getFilePathFromToolArgs`(`:728`)는 `file_path` / `filePath` / `path` / `absolute_path` 4개만 본다. 현재 화이트리스트의 `edit`·`write_file`은 `file_path`를 쓰니 지금은 맞지만, **fail-open 구조**라 도구가 하나 추가되면 조용히 뚫린다. `undefined`면 차단이 맞다.

### 9. 심볼릭 링크를 해석하지 않는다

`qwen-server.ts:718-722`는 `path.resolve` 후 prefix 비교만 한다. `realpath`를 거치지 않으므로 허용 디렉터리 안의 심볼릭 링크가 바깥을 가리키면 통과한다. 웹스퀘어 프로젝트가 공유 라이브러리를 링크로 참조하는 구조라면 실제로 발생한다.

### 10. `settings.json`의 MCP `cwd`가 특정 개발자의 맥 절대경로다

```json
"cwd": "/Users/fujii0711/Develop/PaaS/src/deepsquare-mcp"
```

다른 개발자·CI·운영 서버 어디에도 이 경로는 없다. MCP 서버 기동이 실패하고, 실패 시 세션이 조용히 degrade되는지 에러가 나는지도 불명확하다. 디버그 로그의 경로는 또 다른 개발자(`/Users/kjhoon/...`)라 **이 파일이 공유 설정이 아니라 개인 환경 파일**임을 보여준다.

### 11. 실제 배포 프롬프트가 저장소에 없다

로그 기준 실제 로드되는 `DeepSquare.md`는 20,654 B(빌드에 따라 19,893 B)인데, 저장소의 `.test/DeepSquare.md`는 40,746 B다. **운영에서 모델이 실제로 보는 지시문이 버전 관리되지 않는다.** 답변 품질이 나빠졌을 때 무엇이 바뀌었는지 추적할 방법이 없고 롤백도 불가능하다. 운영 이슈 대응에서 가장 답답한 지점이 될 가능성이 높다.

### 12. 화이트리스트의 `read_many_files`는 존재하지 않는 도구다

`packages/core/src/tools/tool-names.ts`의 `ToolNames`에 없다. 설정이 무시되는데 아무도 모른다. 여러 파일을 한 번에 읽는 수단이 없어 `read_file` 반복 호출로 대체되고, 이것이 3번(컨텍스트 폭발)을 가속한다.

---

## 부기 — 스킬 활성화 조건

품질 개선 방안으로 Skill 시스템 활성화를 검토한다면, `settings.json`의 `tools.core`에 `skill`을 추가하는 것만으로는 부족하다. `qwen-server.ts:296-299`에서 `experimental.skills` 플래그도 함께 켜야 `argv.experimentalSkills`가 설정된다.

## 관련 문서

- [project-spec.md](project-spec.md) §17 — 운영 리스크 (요약판)
- [prompt-rag-finetuning-audit.md](prompt-rag-finetuning-audit.md) — 프롬프트 구성과 주입 경로
- [project-analysis.md](project-analysis.md) — 초기 분석
