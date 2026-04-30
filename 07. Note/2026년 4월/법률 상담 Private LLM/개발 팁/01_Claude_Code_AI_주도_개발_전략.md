# Claude Code를 활용한 AI 주도 개발 전략

> Claude Code의 MCP, 스킬, 훅, 서브 에이전트를 최대한 활용하여 Private LLM 시스템을 효율적으로 구축하는 전략

---

## 1. 전체 전략 개요

### AI 주도 개발이란?

```
기존 개발:
  개발자가 코드 작성 → AI가 보조 (자동완성, 리뷰)

AI 주도 개발:
  AI가 설계/구현/테스트 주도 → 개발자가 의사결정/검증/피드백
  
Claude Code에서의 실현:
  ┌─────────────────────────────────────────────────┐
  │  CLAUDE.md (프로젝트 규칙)                         │
  │  + MCP (외부 시스템 연동)                           │ 
  │  + Skills (반복 작업 자동화)                        │
  │  + Hooks (이벤트 기반 자동 실행)                     │
  │  + Sub-agents (병렬 작업 분산)                     │
  │  ─────────────────────────────────────          │
  │  = 개발자는 "무엇을" 지시, AI가 "어떻게" 실행            │
  └─────────────────────────────────────────────────┘
```

### 이 프로젝트에 적용할 도구 맵

```
┌──────────────────────────────────────────────────────┐
│                   Claude Code                         │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐│
│  │  MCP    │  │ Skills  │  │  Hooks   │  │ Agents ││
│  │ Servers │  │ (명령)  │  │ (자동화) │  │ (병렬) ││
│  └────┬────┘  └────┬────┘  └────┬─────┘  └───┬────┘│
│       │            │            │             │      │
│  GitHub│       /test│      커밋 시│      RAG + │      │
│  Slack │       /deploy     자동 린트    LLM 병렬│      │
│  DB    │       /review     PR 시 자동   코드리뷰│      │
│  Fetch │       /data       테스트 실행          │      │
└──────────────────────────────────────────────────────┘
```

---

## 2. MCP (Model Context Protocol) 연동 전략

### 2.1 MCP란?

<font color="#ff0000">Claude Code가 외부 시스템의 데이터를 읽고 조작할 수 있게 해주는 프로토콜</font>입니다.
MCP 서버를 연결하면 Claude가 GitHub 이슈 확인, DB 조회, 웹 검색 등을 직접 수행합니다.

### 2.2 이 프로젝트에 권장하는 MCP 서버

#### (1) GitHub MCP Server — 필수

```json
// .mcp.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

**활용 시나리오:**

| 명령 | Claude가 하는 일 |
|------|-----------------|
| "이슈 목록 확인해줘" | GitHub Issues 조회, 우선순위 정리 |
| "이 버그 이슈에 대한 PR 만들어줘" | 이슈 읽기 → 코드 수정 → PR 생성 |
| "PR #12 리뷰해줘" | PR diff 분석, 코멘트 작성 |
| "v0.2.0 릴리스 노트 작성해줘" | 커밋 히스토리 분석 → Release 생성 |

#### (2) Filesystem MCP Server — 권장

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/LawLLM"]
    }
  }
}
```

**활용:** 프로젝트 외부의 법률 데이터 파일에 안전하게 접근

#### (3) SQLite MCP Server — 권장

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "data/legal_llm.db"]
    }
  }
}
```

**활용 시나리오:**

| 명령              | Claude가 하는 일            |
| --------------- | ----------------------- |
| "최근 피드백 분석해줘"   | feedback 테이블 조회 → 패턴 분석 |
| "별점 낮은 응답 보여줘"  | rating 기준 정렬 조회         |
| "오늘 상담 통계 요약해줘" | messages 테이블 집계         |
| "DB 스키마 확인해줘"   | 테이블 구조 조회               |

#### (4) Fetch MCP Server — 권장

```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
```

**활용 시나리오:**

| 명령                          | Claude가 하는 일                 |
| --------------------------- | ---------------------------- |
| "국가법령정보센터에서 민법 조문 가져와"      | API 호출 → 데이터 파싱              |
| "HuggingFace에서 이 모델 정보 확인해" | 모델 카드 조회                     |
| "MLX 최신 릴리스 노트 확인해"         | GitHub Release API 조회        |
| "우리 API 헬스체크 해줘"            | localhost:8000/api/health 호출 |

#### (5) Slack MCP Server — 선택 (팀 협업 시)

```json
{
  "mcpServers": {
    "slack": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "<your-token>"
      }
    }
  }
}
```

**활용:** 배포 알림, 에러 알림, 팀 질문 확인

#### (6) Memory MCP Server — 선택

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

**활용:** 프로젝트 의사결정 히스토리, 실험 결과 누적 기록

### 2.3 MCP 조합 시나리오

```
"이번 주 GitHub 이슈 중 버그를 확인하고, 관련 코드를 수정해서 PR 올려줘"

Claude 동작:
  1. [GitHub MCP] 이슈 목록 조회 → bug 라벨 필터
  2. [Read/Edit]  관련 소스 코드 분석 및 수정
  3. [Bash]       테스트 실행 확인
  4. [GitHub MCP] PR 생성 + 이슈 링크

"피드백 DB에서 별점 1~2점인 응답을 분석하고, 프롬프트를 개선해줘"

Claude 동작:
  1. [SQLite MCP] 낮은 평점 피드백 조회
  2. [Read]       현재 prompt.py 분석
  3. [Edit]       프롬프트 템플릿 개선
  4. [Bash]       테스트 실행
```

---

## 3. 커스텀 스킬 (Slash Commands) 전략

### 3.1 스킬이란?

`.claude/commands/` 디렉터리에 마크다운 파일로 정의하는 재사용 가능한 명령입니다.
`/명령이름`으로 실행하면 Claude가 해당 프롬프트를 따릅니다.

### 3.2 이 프로젝트에 필요한 커스텀 스킬

#### /test — 테스트 실행 및 분석

```markdown
<!-- .claude/commands/test.md -->
프로젝트의 테스트를 실행하고 결과를 분석해주세요.

1. `pytest tests/ -v --tb=short` 를 실행하세요.
2. 실패한 테스트가 있으면:
   - 실패 원인을 분석하세요
   - 코드 수정이 필요하면 수정하세요
   - 테스트 자체가 잘못된 경우 테스트를 수정하세요
   - 수정 후 다시 테스트를 실행하여 확인하세요
3. 전체 통과하면 커버리지 요약을 보고해주세요.

인자가 있으면 해당 파일만 테스트하세요: $ARGUMENTS
```

#### /lint — 린트 및 자동 수정

```markdown
<!-- .claude/commands/lint.md -->
프로젝트 코드의 린트와 포맷을 확인하고 수정해주세요.

1. `ruff check src/ tests/` 를 실행하세요.
2. 위반 사항이 있으면 `ruff check --fix src/ tests/` 로 자동 수정하세요.
3. `ruff format src/ tests/` 로 포맷을 맞추세요.
4. 자동 수정이 안 되는 항목은 직접 수정하세요.
5. 최종 결과를 보고해주세요.
```

#### /data-add — 학습 데이터 추가

```markdown
<!-- .claude/commands/data-add.md -->
새로운 법률 상담 Q&A 데이터를 학습 데이터에 추가해주세요.

1. 입력된 내용을 instruction-tuning 포맷(JSONL)으로 변환하세요:
   ```json
   {"instruction": "질문", "input": "", "output": "답변"}
   ```
2. 답변이 다음 기준을 충족하는지 확인하세요:
   - 법률 용어가 정확한가
   - 관련 법조문이 인용되어 있는가
   - 일반인이 이해할 수 있는 표현인가
3. `data/raw/qa/manual_qa.jsonl` 에 추가하세요.
4. 추가된 건수와 내용 요약을 보고해주세요.

추가할 Q&A: $ARGUMENTS
```

#### /rag-test — RAG 검색 품질 테스트

```markdown
<!-- .claude/commands/rag-test.md -->
RAG 검색 품질을 테스트해주세요.

1. API 서버가 실행 중인지 확인하세요 (localhost:8000/api/health).
2. 다음 테스트 질의를 실행하세요:
   - "전세 보증금 반환 절차"
   - "교통사고 손해배상 범위"
   - "이혼 시 재산분할 기준"
   - "소멸시효가 지난 채권"
3. 각 질의에 대해:
   - 검색된 참고 자료의 관련성을 평가하세요
   - 응답의 정확성을 평가하세요
   - 면책 조항이 포함되어 있는지 확인하세요
4. 결과를 표로 정리해주세요.

추가 테스트 질의: $ARGUMENTS
```

#### /deploy-check — 배포 전 점검

```markdown
<!-- .claude/commands/deploy-check.md -->
배포 전 최종 점검을 수행해주세요.

1. 린트 확인: `ruff check src/ tests/`
2. 포맷 확인: `ruff format --check src/ tests/`
3. 테스트 실행: `pytest tests/ -v`
4. Git 상태 확인: 커밋되지 않은 변경사항이 있는지
5. config.yaml 검증: 필수 설정값이 모두 있는지
6. .gitignore 확인: .env, data/, models/ 가 포함되어 있는지
7. 결과를 ✅/❌ 체크리스트로 보고해주세요.
```

#### /api-test — API 엔드포인트 테스트

```markdown
<!-- .claude/commands/api-test.md -->
실행 중인 API 서버의 모든 엔드포인트를 테스트해주세요.

1. GET /api/health → 상태 확인
2. POST /api/chat → 법률 질문 전송 및 응답 확인
3. POST /api/chat/stream → SSE 스트리밍 확인
4. GET /api/history/{session_id} → 이력 조회
5. POST /api/feedback → 피드백 전송

각 엔드포인트의 응답 코드, 응답 시간, 응답 내용 요약을 보고해주세요.
API 서버 주소: ${ARGUMENTS:-http://localhost:8000}
```

#### /experiment — 모델/파라미터 실험

```markdown
<!-- .claude/commands/experiment.md -->
$ARGUMENTS 에 대한 실험을 설계하고 실행해주세요.

1. 실험 목적과 가설을 정리하세요.
2. 현재 설정(config.yaml)에서 변경할 파라미터를 명시하세요.
3. 비교할 기준(baseline)을 기록하세요.
4. 실험을 실행하고 결과를 측정하세요.
5. 결과를 표로 정리하고, 결론 및 권장사항을 제시하세요.
6. 실험 결과를 info/ 디렉터리에 마크다운으로 저장하세요.

주의: config.yaml 원본은 수정하지 마세요. 실험용 설정은 별도 파일로 관리하세요.
```

### 3.3 스킬 디렉터리 구조

```
.claude/
├── commands/                 # 사용자 실행 스킬
│   ├── test.md
│   ├── lint.md
│   ├── data-add.md
│   ├── rag-test.md
│   ├── deploy-check.md
│   ├── api-test.md
│   └── experiment.md
├── agents/                   # 서브 에이전트 (아래 섹션)
│   ├── code-reviewer.md
│   ├── data-validator.md
│   └── rag-evaluator.md
└── CLAUDE.md                 # 프로젝트 규칙
```

---

## 4. 훅 (Hooks) 전략

### 4.1 훅이란?

Claude Code의 특정 이벤트(도구 실행 전후, 알림 등)에 자동으로 실행되는 셸 명령입니다.
`.claude/settings.json`에 설정합니다.

### 4.2 이 프로젝트에 권장하는 훅

#### (1) 파일 수정 후 자동 린트

```json
// .claude/settings.json
{
  "hooks": {
    "afterEdit": [
      {
        "matcher": "**/*.py",
        "command": "ruff check --fix $FILE && ruff format $FILE"
      }
    ]
  }
}
```

**효과:** Claude가 Python 파일을 수정할 때마다 자동으로 린트 + 포맷 적용

#### (2) 커밋 전 테스트 실행

```json
{
  "hooks": {
    "preCommit": [
      {
        "command": "cd /path/to/LawLLM && pytest tests/ -x -q --tb=line"
      }
    ]
  }
}
```

**효과:** Claude가 커밋하기 전에 테스트가 통과하는지 자동 확인. 실패 시 커밋 차단

#### (3) PR 생성 전 배포 점검

```json
{
  "hooks": {
    "prePush": [
      {
        "command": "ruff check src/ tests/ && ruff format --check src/ tests/"
      }
    ]
  }
}
```

#### (4) 프롬프트 제출 시 컨텍스트 주입

```json
{
  "hooks": {
    "onPromptSubmit": [
      {
        "matcher": ".*법률.*|.*판례.*|.*민사.*",
        "command": "echo '참고: 법률 관련 작업 시 CLAUDE.md의 면책 조항 규칙과 보안 규칙을 반드시 준수하세요.'"
      }
    ]
  }
}
```

### 4.3 훅 조합 전략

```
개발 중 자동화 흐름:

  Claude가 코드 수정
       │
       ▼
  [afterEdit 훅] 자동 ruff 린트/포맷
       │
       ▼
  Claude가 커밋 시도
       │
       ▼
  [preCommit 훅] pytest 자동 실행
       │
    Pass ──→ 커밋 성공
    Fail ──→ Claude가 에러 확인 → 코드 수정 → 재시도
```

---

## 5. 서브 에이전트 (Sub-agents) 전략

### 5.1 서브 에이전트란?

Claude Code 내에서 독립적인 작업을 수행하는 전문화된 에이전트입니다.
`.claude/agents/` 디렉터리에 마크다운으로 정의합니다.
메인 Claude가 복잡한 작업을 분할하여 서브 에이전트에게 위임합니다.

### 5.2 이 프로젝트에 권장하는 서브 에이전트

#### (1) Code Reviewer 에이전트

```markdown
<!-- .claude/agents/code-reviewer.md -->
---
name: code-reviewer
description: 코드 변경사항을 CLAUDE.md 규칙에 따라 리뷰하는 에이전트
---

당신은 이 프로젝트의 코드 리뷰어입니다.

## 리뷰 기준

### 필수 확인 사항
1. **타입 힌트**: 모든 함수에 인자/반환 타입이 있는가? `Any` 사용은 없는가?
2. **하드코딩**: 매직 넘버나 하드코딩된 경로가 없는가? config.yaml을 사용하는가?
3. **비동기**: FastAPI 핸들러가 `async def`인가? I/O 작업에 await을 쓰는가?
4. **보안**: 사용자 입력이 로그에 기록되지 않는가? 면책 조항이 포함되는가?
5. **프롬프트**: 인라인 프롬프트가 없는가? prompt.py에서만 관리하는가?

### 리뷰 형식
각 파일에 대해:
- ✅ 규칙 준수 항목
- ❌ 위반 항목 (수정 제안 포함)
- 💡 개선 제안 (선택)
```

#### (2) Data Validator 에이전트

```markdown
<!-- .claude/agents/data-validator.md -->
---
name: data-validator
description: 학습 데이터와 RAG 문서의 품질을 검증하는 에이전트
---

당신은 법률 학습 데이터 품질 검증 전문가입니다.

## 검증 항목

### Fine-tuning 데이터 (JSONL)
1. JSON 파싱 가능 여부
2. instruction, output 필드 존재 및 비어있지 않은지
3. instruction 길이: 10~500자
4. output 길이: 50~2000자
5. 한국어 인코딩 (UTF-8) 정상
6. 개인정보 포함 여부 (주민번호, 전화번호 패턴 검사)

### RAG 문서
1. 제목과 doc_type 메타데이터 존재
2. 법조문: 조항 번호가 정상적으로 파싱되는지
3. 판례: 사건번호 형식이 올바른지
4. 빈 문서나 너무 짧은 문서(10자 미만) 없는지

## 보고 형식
- 총 건수, 유효 건수, 오류 건수
- 오류 유형별 분류 (표)
- 수정 필요한 항목의 라인 번호
```

#### (3) RAG Evaluator 에이전트

```markdown
<!-- .claude/agents/rag-evaluator.md -->
---
name: rag-evaluator
description: RAG 검색 품질을 평가하고 개선점을 제안하는 에이전트
---

당신은 RAG 시스템 성능 평가 전문가입니다.

## 평가 방법

1. 테스트 질의 세트 (10개 이상)에 대해 검색 실행
2. 각 결과의 관련성을 3단계로 평가:
   - 🟢 높음: 질문에 직접 답할 수 있는 문서
   - 🟡 중간: 관련은 있지만 핵심이 아닌 문서
   - 🔴 낮음: 관련 없는 문서
3. 지표 산출:
   - Precision@3: 상위 3개 중 관련 문서 비율
   - 적중률: 정답 문서가 상위 5개에 포함되는 비율

## 개선 제안
- 검색 실패 사례의 원인 분석
- 청킹 전략 수정 제안
- 파라미터 조정 제안 (top_k, chunk_size 등)
```

### 5.3 서브 에이전트 활용 시나리오

#### 시나리오 1: 대규모 코드 변경 리뷰

```
사용자: "RAG 파이프라인 전체 리팩터링 후 리뷰해줘"

Claude 동작:
  1. [메인] 리팩터링 수행 (indexer.py, retriever.py, reranker.py)
  2. [서브: code-reviewer] 변경된 3개 파일 병렬 리뷰
  3. [메인] 리뷰 결과 종합 → 수정 필요 사항 반영
  4. [서브: code-reviewer] 수정 후 재리뷰
```

#### 시나리오 2: 데이터 추가 + 검증 + 인덱싱

```
사용자: "새 판례 데이터 1,000건을 추가하고 RAG에 반영해줘"

Claude 동작:
  1. [메인] 데이터 파일 확인
  2. [서브: data-validator] 1,000건 품질 검증 (병렬)
  3. [메인] 검증 통과 데이터만 data/raw/에 저장
  4. [메인] 인덱싱 실행
  5. [서브: rag-evaluator] 인덱싱 후 검색 품질 평가
  6. [메인] 결과 보고
```

#### 시나리오 3: 기능 개발 풀사이클

```
사용자: "채팅 이력 내보내기 API를 추가해줘"

Claude 동작:
  1. [메인] 기능 설계 및 구현
     - schemas.py에 ExportRequest/Response 추가
     - routes.py에 /api/export 엔드포인트 추가
     - database.py에 export 쿼리 추가
  2. [서브: code-reviewer] 새 코드 리뷰 (CLAUDE.md 규칙 준수)
  3. [메인] 리뷰 반영 수정
  4. [메인] 테스트 코드 작성 + 실행
  5. [메인] /deploy-check 스킬 실행
```

---

## 6. CLAUDE.md 최적화 전략

### 6.1 AI 주도 개발을 위한 CLAUDE.md 강화

기존 CLAUDE.md에 AI 주도 작업 패턴을 추가합니다:

```markdown
<!-- CLAUDE.md에 추가할 섹션 -->

## AI 주도 개발 규칙

### 코드 작성 후 자동 수행
- 새 함수 작성 시 → 대응하는 테스트도 함께 작성
- 파일 수정 시 → 기존 테스트가 통과하는지 확인
- API 엔드포인트 추가 시 → schemas.py, routes.py, test_api.py 모두 수정

### 작업 보고 형식
- 변경한 파일 목록과 변경 이유
- 실행한 테스트 결과
- 추가 작업이 필요한 사항

### 의사결정 필요 시
다음 상황에서는 코드를 작성하기 전에 사용자에게 질문하세요:
- 아키텍처 변경이 필요할 때
- CLAUDE.md 규칙과 충돌하는 요구사항일 때
- 2가지 이상의 합리적인 접근 방식이 있을 때
```

### 6.2 docs/ 디렉터리 활용

```
.claude/
├── CLAUDE.md                   # 프로젝트 핵심 규칙
└── docs/
    ├── API_SCHEMA.md           # API 스키마 상세 (Claude 참조용)
    ├── DATA_FORMAT.md          # 학습 데이터 포맷 명세
    ├── PROMPT_GUIDELINES.md    # 프롬프트 작성 가이드라인
    └── TESTING_POLICY.md       # 테스트 정책 상세
```

Claude는 `.claude/docs/` 하위 파일을 자동으로 컨텍스트에 포함합니다.
규칙이 길어지면 CLAUDE.md에서 분리하여 docs/에 배치합니다.

---

## 7. 워크플로 자동화 조합

### 7.1 일일 개발 사이클

```
아침 시작:
  /deploy-check              ← 현재 상태 점검
  "GitHub 이슈 확인해줘"      ← [GitHub MCP] 오늘 할 일 파악

개발 중:
  "이 기능 구현해줘"           ← Claude가 코드 작성
  [afterEdit 훅]              ← 자동 린트/포맷
  /test                       ← 테스트 확인
  [서브: code-reviewer]       ← 코드 리뷰

커밋/PR:
  /commit                     ← 커밋 (내장 스킬)
  [preCommit 훅]              ← 테스트 자동 실행
  "PR 만들어줘"                ← [GitHub MCP] PR 생성

하루 마무리:
  /rag-test                   ← RAG 품질 확인
  "오늘 피드백 요약해줘"        ← [SQLite MCP] 피드백 분석
```

### 7.2 주간 유지보수 사이클

```
주 1회:
  "이번 주 GitHub 이슈 정리해줘"          ← 이슈 트리아지
  "피드백 DB에서 개선점 분석해줘"          ← 서비스 품질 분석
  "새 판례 데이터 검증하고 인덱싱해줘"     ← 데이터 업데이트
  /experiment "chunk_size 256 vs 512"    ← 파라미터 실험
```

### 7.3 릴리스 사이클

```
릴리스 시:
  /deploy-check                          ← 최종 점검
  /test                                  ← 전체 테스트
  "v0.2.0 릴리스 노트 작성해줘"            ← [GitHub MCP]
  "main에 머지하고 태그 생성해줘"           ← 자동 배포 트리거
```

---

## 8. 실전 프롬프트 패턴

### 8.1 기능 개발 프롬프트

```
사용자: "사용자가 대화를 PDF로 내보낼 수 있는 기능을 추가해줘.
        API 엔드포인트와 Streamlit UI 버튼 모두 필요해."

→ Claude가 CLAUDE.md 규칙에 따라:
  1. schemas.py에 ExportRequest 추가
  2. routes.py에 async 엔드포인트 추가
  3. UI에 다운로드 버튼 추가
  4. 테스트 코드 작성
  5. /test 실행
```

### 8.2 버그 수정 프롬프트

```
사용자: "RAG 검색 결과가 빈 배열일 때 500 에러가 발생해.
        GitHub 이슈 #15 확인하고 수정해줘."

→ Claude 동작:
  1. [GitHub MCP] 이슈 #15 상세 확인
  2. 관련 코드 분석 (routes.py, retriever.py)
  3. 버그 수정
  4. 엣지 케이스 테스트 추가
  5. /test 확인 후 커밋
  6. [GitHub MCP] PR 생성 + 이슈 #15 링크
```

### 8.3 데이터 작업 프롬프트

```
사용자: "/data-add
        Q: 전세 계약 갱신 거절 사유는 무엇인가요?
        A: 주택임대차보호법 제6조의3에 따르면..."

→ Claude가 /data-add 스킬에 따라:
  1. JSONL 포맷 변환
  2. 품질 검증
  3. data/raw/qa/manual_qa.jsonl에 추가
  4. 결과 보고
```

---

## 9. 설정 파일 종합

### 9.1 .mcp.json (최종)

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<token>"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "data/legal_llm.db"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
```

### 9.2 .claude/settings.json (최종)

```json
{
  "hooks": {
    "afterEdit": [
      {
        "matcher": "**/*.py",
        "command": "ruff check --fix $FILE && ruff format $FILE"
      }
    ],
    "preCommit": [
      {
        "command": "pytest tests/ -x -q --tb=line"
      }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(ruff *)",
      "Bash(pytest *)",
      "Bash(uvicorn *)",
      "Bash(streamlit *)",
      "Bash(pip install *)",
      "Bash(git *)",
      "Bash(curl *)",
      "Bash(python -m src.*)"
    ]
  }
}
```

---

## 10. 도입 체크리스트

### Phase 1: 기본 설정 (1일)
- [ ] CLAUDE.md 프로젝트 루트에 배치
- [ ] `.claude/commands/` 에 커스텀 스킬 6개 생성
- [ ] `.claude/settings.json` 훅 설정
- [ ] `.mcp.json` MCP 서버 설정 (GitHub + SQLite + Fetch)
- [ ] MCP 서버 동작 테스트

### Phase 2: 서브 에이전트 (1일)
- [ ] `.claude/agents/` 에 서브 에이전트 3개 정의
- [ ] code-reviewer 에이전트 테스트
- [ ] data-validator 에이전트 테스트
- [ ] rag-evaluator 에이전트 테스트

### Phase 3: 워크플로 검증 (1일)
- [ ] /test 스킬 동작 확인
- [ ] /lint 스킬 동작 확인
- [ ] /deploy-check 스킬 동작 확인
- [ ] afterEdit 훅 동작 확인 (Python 파일 수정 시 자동 린트)
- [ ] preCommit 훅 동작 확인 (커밋 시 테스트 실행)
- [ ] 기능 개발 → 리뷰 → 테스트 → 커밋 풀사이클 1회 수행
