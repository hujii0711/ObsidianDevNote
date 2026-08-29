# 파인튜닝 · RAG · 프롬프트 엔지니어링 감사

> 기준일: 2026-08-28
> 질문: 웹스퀘어 코드 관련 질문에 모델이 잘 답하도록 파인튜닝 / RAG / 프롬프트 엔지니어링 중 하나라도 작업한 흔적이 있는가?

**결론: 셋 중 프롬프트 엔지니어링만 있다. 파인튜닝과 RAG는 흔적이 없다.**

## 요약

| 기법 | 상태 | 한 줄 근거 |
| --- | --- | --- |
| 파인튜닝 | **없음** | 학습 스크립트·데이터셋·가중치 매칭 0건. 구조적으로도 API 클라이언트라 가중치를 만질 지점이 없음 |
| RAG | **없음** | 벡터DB 의존성 0건. 임베딩 API는 있으나 프로덕션 호출부 0건 |
| 프롬프트 엔지니어링 | **있음** | 컨텍스트 파일(`DeepSquare.md`) 주입 방식. 파일 3개에 작업이 몰려 있음 |

---

## 1. 파인튜닝 — 흔적 없음

다음을 전부 검색했고 매칭 0건이다.

- 학습 스크립트 (`*train*.py`)
- 데이터셋 (`*.jsonl`, `*dataset*`)
- 가중치 (`*.safetensors`)
- 노트북 (`*.ipynb`)
- Python 의존성 (`requirements.txt`)
- 키워드 — `fine-tun`, `finetun`, `lora`

구조적으로도 불가능에 가깝다. 이 프로젝트는 **모델을 API로 호출하는 클라이언트**이고, 실제 운영에서는 ProWorks Studio 프록시 뒤의 모델을 쓴다. 모델 가중치를 만질 지점 자체가 없다.

---

## 2. RAG — 없음 (재료만 남아 있음)

### 벡터 저장소 의존성 0건

faiss · chroma · pinecone · qdrant · weaviate · milvus — `package.json` 어디에도 없다.

### 임베딩 API는 존재하나 아무도 호출하지 않는다

- `packages/core/src/core/baseLlmClient.ts:157` — `generateEmbedding()` 구현체 존재
- 저장소 전체에서 이 메서드 매칭은 **정의부 1건 + 테스트 8건**뿐. **프로덕션 호출부 0건**
- 기본 임베딩 모델도 `text-embedding-v4`로 잡혀 있으나 사용처가 없다
- 청킹 · 인덱싱 · 유사도 계산 · 벡터 저장소 — 전부 없음

Gemini CLI에서 물려받은 미사용 코드로 보인다. 참고로 Anthropic 경로는 아예 `throw new Error('Anthropic does not support embeddings.')`이고, OpenAI 경로는 하드코딩된 `text-embedding-ada-002`를 쓴다.

### 검색에서 나온 오탐

`retrieval` 등의 키워드 매칭은 전부 무관한 문맥이었다.

- `getFolderStructure.ts:24` — "folder structure retrieval" (주석)
- `qwenContentGenerator.ts:117` — "token and endpoint retrieval" (OAuth 토큰)
- `prompts.ts` — "codebase exploration" (서브에이전트 설명)

### 대신 쓰는 방식 — 에이전틱 검색

RAG 자리를 **도구 기반 탐색**이 대신한다. 벡터로 유사 문서를 찾아 프롬프트에 끼워넣는 대신, 모델이 `glob` → `grep_search` → `read_file`을 스스로 호출해 필요한 파일을 찾아 읽는다. 허용 도구 조합이 정확히 이것이다.

컨텍스트 주입도 정적이다 — `DeepSquare.md` / `AGENTS.md`를 global·project 디렉터리에서 통째로 로드한다. 질의별로 관련 조각을 뽑아오는 게 아니다.

---

## 3. 프롬프트 엔지니어링 — 이것만 실제로 작업됨

프로젝트가 직접 프롬프트를 건드린 파일은 **3개뿐**이다. 나머지 프롬프트 파일은 전부 qwen-code 상류 원본 그대로다.

### 3.1 `packages/api-server/.test/DeepSquare.md` — 실질적인 본체

40,746 bytes / 1,169줄. 웹스퀘어 도메인 지식이 **사실상 전부 여기 있다.** 도메인 마커(WebSquare/Inswave/DeepSquare) 28건.

| 장 | 내용 |
| ---: | --- |
| 1 | 출력 형식: XML (JS가 아님) |
| 2 | 웹스퀘어(WebSquare)란 |
| 3 | 웹스퀘어 SPA의 ID 문제 — 문서가 "가장 중요"로 표기 |
| 4 | 컴포넌트별 렌더링 특성과 주의사항 |
| 5 | Node.js 환경 주의사항 |
| 6 | 공통 헬퍼 함수 (CDATA 인라인 선언) |
| 7 | 입력 파일 설명 — `interface-metadata.json`, `test-plan.md` |
| 8 | XML 메타데이터 생성 규칙 |
| 9 | 시나리오별 코드 패턴 |
| 10 | 실전 예시: BM002M01 전체 XML |
| 11 | 생성 시 체크리스트 |

성격은 **TestSquare E2E 테스트 XML 생성 가이드**다.

핵심 기술 규약:

- 화면 URL — `/websquare/websquare.html?w2xPath=/ui/BM/BM002M01.xml`
- 네임스페이스 — `xmlns:w2="http://www.inswave.com/websquare"`
- `page.evaluate` 안에서 **prefix를 붙여** `window.WebSquare`에 접근

### 3.2 `packages/api-server/esbuild.config.js:517-525` — 시스템 프롬프트 치환

빌드타임에 `core/prompts.ts`를 패치한다. **단 한 건이고, 내용은 브랜딩뿐이다.**

```js
file: join(CORE_DIR, "core/prompts.ts"),
patches: [{
  name: "System prompt Change",
  find:    "You are Qwen Code, an interactive CLI agent developed by Alibaba Group",
  replace: "You are AI Talk Plus, an interactive CLI agent developed by Inswave",
}]
```

이 파일의 전체 패치 20건 중 프롬프트 관련은 이것 하나이고, 나머지 19건은 전부 인프라(경로/스트리밍 이벤트)다.

### 3.3 `packages/api-server/src/server/qwen-server.ts` — 주입 배선

프롬프트 텍스트는 없고, `DeepSquare.md`가 **로드되게 만드는** 코드다.

| 위치 | 내용 |
| --- | --- |
| `:271-277` | 컨텍스트 파일명을 `QWEN.md` → `["DeepSquare.md", "AGENTS.md"]`로 교체 **(핵심)** |
| `:264` | project 디렉터리 기본값을 `{workspace}/deepsquare`로 |
| `:317-327` | `QWEN_HOME`을 `metadataPath/.deepsquare`로 |

### 주입 경로

```
{globalDirPath}/DeepSquare.md  ─┐
{projectDirPath}/DeepSquare.md ─┼→ MEMORY_DISCOVERY → user memory → 시스템 프롬프트 뒤 append
{...}/AGENTS.md                ─┘
                                  ↑ 매 세션, 질의와 무관하게 전문 삽입
```

---

## 4. 수정되지 않은 상류 원본

혹시 손댔는지 확인했으나 **도메인 마커 0건**으로 전부 원본이다.

| 파일 | 크기 | 역할 |
| --- | ---: | --- |
| `packages/core/src/core/prompts.ts` | 55,957 B | 기본 시스템 프롬프트 (3.2 패치의 대상) |
| `packages/core/src/subagents/builtin-agents.ts` | 4,020 B | 내장 서브에이전트 프롬프트 |
| `packages/core/src/utils/subagentGenerator.ts` | 6,907 B | 서브에이전트 생성 프롬프트 |
| `packages/core/src/tools/memoryTool.ts` | 19,703 B | 컨텍스트 파일 로딩 (기본값 `QWEN.md`) |

즉 **55KB짜리 기본 시스템 프롬프트는 브랜딩 한 줄 빼고 그대로 두고, 도메인 지식은 전부 `DeepSquare.md`로 분리**한 구조다.

역할을 바꾸려면 코드가 아니라 `DeepSquare.md`와 `settings.json`을 갈아끼우면 된다.

---

## 5. 주의점

### 5.1 컨텍스트 압축이 꺼져 있다

`.deepsquare/settings.json`의 `model.chatCompression.contextPercentageThreshold: 0` → `chatCompressionService.ts:95`에서 `threshold <= 0`이면 즉시 NOOP.

컨텍스트 관리 전략이 "검색해서 좁히기"도 "요약해서 줄이기"도 아니고 **1M 컨텍스트 윈도우에 전부 밀어넣고 버티기**다. `coder-model`의 1M 컨텍스트 설정과 맞물린다. 웹스퀘어 XML 화면 소스와 테스트 가이드를 통째로 넣어야 하는 작업 성격상 나온 선택으로 보이나, 긴 세션에서는 컨텍스트 한계에 그대로 부딪힌다.

### 5.2 저장소의 `DeepSquare.md`는 실제 배포본이 아니다

경로가 `.test/`이고, 런타임 로그(`packages/api-server/.deepsquare/debug/`)를 보면 실제로 읽힌 파일은 `.deepsquare/DeepSquare.md`이며 **크기가 20,654 bytes**로 저장소의 40KB와 다르다. **실 운영 프롬프트 본문은 이 체크아웃에 없다.**

### 5.3 문서 코퍼스를 얹으려던 흔적이 있고, 지금은 제거됐다

같은 디버그 로그(2026-04-01)에 저장소에 없는 프롬프트 리소스가 보인다.

```
[dsFs] Disk overlay applied: 560 files ...
  (replaced top-level regions: DeepSquare.md, WebSquareSchema.xml, debug,
   installation_id, settings.json, settings.json.orig, websquare_docs)
```

`dsFs` 가상 파일시스템 오버레이가 `WebSquareSchema.xml`과 `websquare_docs/` 디렉터리를 주입하고 있었다. 현재 소스에는 `packages/api-server/src/server/index.ts:225`의 주석만 남고 코드가 사라졌다.

```
// Handle user messages (dsFs wrapper removed — standard fs used)
```

RAG는 아니고 파일 오버레이 방식이었지만, 도메인 문서를 붙이려던 시도로 읽힌다.

### 5.4 MCP가 현재 연결되지 않는다

파일 조작 외의 능력은 `websquare-mcp` MCP 서버가 담당하는데, 이 서버 구현체는 저장소에 없고 `cwd`가 개발자 개인 macOS 경로(`/Users/fujii0711/Develop/PaaS/src/deepsquare-mcp`)로 커밋돼 있어 **현재 상태로는 연결되지 않는다.**

---

## 관련 문서

- [project-spec.md](project-spec.md) — 전체 기술 스펙 (프롬프트 항목은 §12)
- [project-analysis.md](project-analysis.md) — 초기 프로젝트 분석
