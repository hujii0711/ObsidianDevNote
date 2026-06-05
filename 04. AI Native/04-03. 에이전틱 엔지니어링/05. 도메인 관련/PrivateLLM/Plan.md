# 계획 1 — 데이터 파이프라인 (수집 → 정제 → 청킹 → 색인) 구현 계획

## Task 0: 프로젝트 스캐폴딩
## Task 1: 청크 스키마 정의
## Task 2: 설정 모듈 (경로 + OC 키)
## Task 3: law.go.kr HTTP 클라이언트 (URL 빌드 + 호출)
## Task 4: 실제 fixture 캡처 (네트워크, 수동 실행)
## Task 5: 법령 파서 (XML → 조문 구조)
## Task 6: 판례 파서 (XML → 판시사항/판결요지)
## Task 7: 수집 오케스트레이션 → data/raw
## Task 8: 텍스트 정규화
## Task 9: 구조 인지 청킹 → data/chunks/chunks.jsonl
## Task 10: 임베딩 래퍼 (bge-m3)
## Task 11: Chroma 색인 구축
## Task 12: 검색 CLI (엔드투엔드 스모크 검증)

---
# 계획 2A — RAG 코어 + FastAPI `/chat` 백엔드 구현 계획
## Task 0: uv 워크스페이스 + rag 패키지 스캐폴딩
## Task 1: RagConfig + 상수
## Task 2: 타입 + 질의 임베더
## Task 3: Retriever (Chroma top-k + grounding 판정)
## Task 4: 프롬프트 빌더 (상담형 + 번호 매긴 근거)
## Task 5: 인용 매핑 (환각 인용 제거)
## Task 6: api 패키지 스캐폴딩 + 설정 + 스키마
## Task 7: LLM 서비스 (MLX Qwen, 주입 가능 스트리밍)
## Task 8: RAG 파이프라인 오케스트레이션
## Task 9: FastAPI `/chat` (SSE) + `/health`
## Task 10: CORS + 도메인 가드 마무리 + 라이브 스모크

---

# 계획 2B — Next.js 채팅 UI 구현 계획
## Task 0: Next.js 스캐폴딩 + Vitest + 워크스페이스 제외
## Task 1: 타입 + SSE 파서
## Task 2: 채팅 클라이언트 (fetch POST + 스트림)
## Task 3: useChat 훅 (상태 머신)
## Task 4: SourceCard 컴포넌트
## Task 5: MessageBubble 컴포넌트
## Task 6: ChatInput 컴포넌트
## Task 7: Chat 컨테이너 + 페이지 연결
## Task 8: 라이브 엔드투엔드 스모크 (api + web)

---

# 계획 3A — 평가 하니스 + 베이스라인 측정 구현 계획
## Task 0: packages/eval 스캐폴딩
## Task 1: EvalItem 스키마 + 로더
## Task 2: 큐레이션된 평가셋 (그라운드 트루스)
## Task 3: 검색 지표 (recall@k / hit@k)
## Task 4: 답변 형식·키워드 지표
## Task 5: LLM-as-judge 충실도 (주입 가능)
## Task 6: 평가 러너
## Task 7: 리포트 집계 + CLI
## Task 8: 라이브 베이스라인 측정

---

# 계획 3B — FT 데이터셋 빌더 (rejection-sampling distillation) 구현 계획

## Task 0: packages/ftdata 스캐폴딩
## Task 1: 질문 풀 (평가셋과 disjoint)
## Task 2: 형식 필터 (eval 지표 재사용)
## Task 3: chat-example 빌더 + train/valid 분할
## Task 4: 후보 생성기
## Task 5: 빌드 CLI
## Task 6: 라이브 데이터셋 빌드 + 검수

---

# 계획 3C — QLoRA 학습 + 어댑터 서빙 + A/B 측정 구현 계획
## Task 0: finetune 패키지 스캐폴딩
## Task 1: MlxLLM 어댑터 지원
## Task 2: eval.cli `--adapter` (생성=어댑터, judge=base)
## Task 3: 학습 커맨드 빌더
## Task 4: A/B 비교 모듈
## Task 5: 라이브 — 학습 + A/B 측정 + 결론

---
## 설계

## 1. 개요
## 2. 제약 조건
## 3. 기술 스택
## 4. 아키텍처
## 5. 데이터 파이프라인 & RAG
## 6. QLoRA 파인튜닝
## 7. 평가 (단계적 전략의 핵심)
## 8. 에러 처리 & 견고성
## 9. 테스트 (TDD 기조)
