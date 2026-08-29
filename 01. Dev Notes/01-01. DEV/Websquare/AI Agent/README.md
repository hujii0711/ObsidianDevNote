# docs

`websquare-ai-agent-qwen` 저장소 분석 문서 모음. 모든 내용은 저장소 코드를 직접 읽어 확인했고, 확인 범위를 벗어난 항목은 각 문서에 명시했다.

## 문서 목록

| 문서 | 형식 | 내용 |
| --- | --- | --- |
| [project-spec.md](project-spec.md) | Markdown | **전체 기술 스펙** — 17개 절. 계보, 런타임·빌드, 소켓 API 전량, 빌드타임 패치, 도구·모델 레지스트리, 에이전트 루프, 비활성 인벤토리, 운영 리스크 |
| [project-spec.html](project-spec.html) | HTML | 위 스펙의 웹 문서판. 브라우저에서 바로 열리는 독립 실행 파일 |
| [prompt-rag-finetuning-audit.md](prompt-rag-finetuning-audit.md) | Markdown | 파인튜닝 / RAG / 프롬프트 엔지니어링 감사 — 무엇이 있고 무엇이 없는지, 프롬프트 작업 파일 3개 정리 |
| [project-analysis.md](project-analysis.md) | Markdown | 초기 프로젝트 분석 (선행 문서) |

## 읽는 순서

1. 처음이면 **project-spec.md** §1(정체와 계보)과 §16(비활성 인벤토리)
2. 프롬프트·모델 품질이 관심사면 **prompt-rag-finetuning-audit.md**
3. 배포를 앞두고 있으면 **project-spec.md** §17(운영 리스크)

## 핵심 요약

- **정체** — 인스웨이브 웹스퀘어 개발환경용 AI 코딩 에이전트("AI Talk Plus"). `qwen-code` 0.12.0을 소켓 서버로 재포장한 포크
- **계보** — Gemini CLI → qwen-code → 이 프로젝트. core 425개 `.ts` 중 Google 273 / Qwen 137
- **전략** — core 소스를 직접 수정하지 않고 **빌드타임 패치 20건**(9파일)으로 처리해 upstream 머지 비용을 낮춤
- **모델** — 벤더 고정이 아님. 실제로는 ProWorks Studio 프록시(`proworksTKey`) 뒤의 모델을 호출
- **프롬프트** — 파인튜닝·RAG 없음. `DeepSquare.md` 컨텍스트 파일 전문(全文) 주입 + 에이전틱 검색
- **주의** — 선언돼 있으나 동작하지 않는 항목이 11개 있다(§16). 특히 실효 도구는 8종이 아니라 **7종**이고, MCP는 `cwd`가 개인 경로라 **현재 연결되지 않는다**

## 문서 생성 메모

`project-spec.html`은 `project-spec.md`와 같은 내용을 담은 웹 문서다. 두 파일은 별개 산출물이므로 **한쪽을 고치면 다른 쪽도 함께 갱신**해야 한다.

HTML판의 구현 사항:

- 완결 문서 — `<!doctype html>`, `<html lang="ko">`, charset·viewport·description 메타, 최소 CSS 리셋 내장
- 테마 — `prefers-color-scheme` 자동 대응 + 우상단 수동 토글(시스템 → 다크 → 라이트). `localStorage` 접근은 `try/catch`로 감쌌다
- 폰트 — Google Fonts(Noto Serif KR / IBM Plex Sans KR / IBM Plex Mono). **오프라인에서는 폴백 글꼴로 렌더링**되며, 폴백 스택에 `Malgun Gothic` 등 Windows 기본 한글 폰트를 지정해 레이아웃은 유지된다
- 인쇄 — 테마 토글은 `@media print`에서 숨김
