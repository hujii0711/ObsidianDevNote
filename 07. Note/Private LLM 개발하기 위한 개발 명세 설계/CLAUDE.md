# CLAUDE.md — Private Legal LLM

## 프로젝트 개요

민사사건 법률 상담 특화 Private LLM 서비스.
RAG + Fine-tuning 혼용 아키텍처, MacBook Pro M4 48GB에서 로컬 실행.

---

## 기술 스택 (변경 금지)

| 레이어         | 기술                     | 비고                              |
| ----------- | ---------------------- | ------------------------------- |
| 언어          | Python 3.11+           | 타입 힌트 필수                        |
| 파운데이션 모델    | EEVE-Korean-10.8B (Q4) | 한국어 법률 도메인 특화                   |
| 추론 엔진       | MLX                    | Apple Silicon 전용, llama.cpp는 폴백 |
| Fine-tuning | MLX-LM (QLoRA)         | LoRA rank 64, alpha 128         |
| RAG 프레임워크   | LlamaIndex             | LangChain 사용 금지                 |
| 벡터 DB       | ChromaDB               | 로컬 전용, 외부 DB 금지                 |
| 임베딩 모델      | BGE-M3                 | 리랭킹은 KoSentenceBERT             |
| API 서버      | FastAPI + Uvicorn      | 동기 엔드포인트 금지, async 필수           |
| UI          | Streamlit              |                                 |
| DB          | SQLite                 | 대화 이력, 피드백 저장                   |
| 컨테이너        | Docker Compose         |                                 |

---

## 디렉터리 구조 규칙

```
private-legal-llm/
├── data/raw/              # 원본 데이터 (git 미추적)
├── data/processed/        # 전처리된 학습 데이터
├── data/vectordb/         # ChromaDB 저장소 (git 미추적)
├── models/                # 모델 파일 (git 미추적)
├── src/api/               # FastAPI 서버
├── src/rag/               # RAG 파이프라인
├── src/llm/               # LLM 추론 엔진
├── src/training/          # 파인튜닝 스크립트
├── ui/                    # Streamlit UI
├── tests/                 # 테스트 (src/ 구조 미러링)
└── scripts/               # 유틸리티 스크립트
```

- `src/` 하위에만 프로덕션 코드 작성
- 새 디렉터리 생성 시 반드시 `__init__.py` 포함
- `data/`, `models/`는 `.gitignore` 대상

---

## 코딩 규칙

### Python 스타일
- 포매터: `ruff format`, 린터: `ruff check`
- 타입 힌트 필수. `Any` 사용 금지
- docstring: 공개 함수/클래스에만 Google 스타일
- import 순서: stdlib > third-party > local (isort 규칙)

### 하드코딩 금지
- 모델 경로, 청킹 파라미터, API 포트 등 모든 설정값은 환경변수 또는 `config.yaml`로 관리
- 매직 넘버 금지. 상수로 정의하여 사용

### 비동기 원칙
- FastAPI 라우터 핸들러는 반드시 `async def`
- LLM 추론, 벡터 검색 등 I/O 바운드 작업은 비동기 처리
- CPU 바운드 작업(임베딩 생성 등)은 `asyncio.to_thread` 사용

### 에러 처리
- 외부 경계(API 입력, 파일 I/O, 모델 로딩)에서만 예외 처리
- 내부 함수 간 방어적 검증 금지
- 커스텀 예외는 `src/api/exceptions.py`에 집중

---

## 아키텍처 규칙

### RAG 파이프라인
- 청킹: 법조문은 조항 단위, 판례는 512 토큰 + 64 오버랩
- 검색: top-k=5 검색 후 KoSentenceBERT로 리랭킹, 최종 top-3 사용
- 임베딩 모델은 `src/rag/` 내에서만 로드. 다른 모듈에서 직접 접근 금지

### LLM 추론
- 모델 로딩은 `src/llm/engine.py`의 싱글턴으로 관리
- 프롬프트 템플릿은 `src/llm/prompt.py`에서 관리. 인라인 프롬프트 금지
- 스트리밍 응답 시 SSE(Server-Sent Events) 사용

### 혼용 전략 (RAG + Fine-tuning)
- 모든 응답은 RAG 검색 결과를 컨텍스트로 포함해야 함 (환각 방지)
- Fine-tuned 모델은 톤/형식/법률 용어 이해를 담당
- RAG 검색 결과가 없으면 "관련 자료를 찾지 못했습니다" 응답. 추측 금지

---

## 데이터 처리 규칙

### 전처리
- 원본 데이터는 `data/raw/`에 보존, 절대 수정하지 않음
- 전처리 스크립트는 멱등성(idempotent) 보장
- 전처리 결과는 `data/processed/`에 저장

### 학습 데이터 포맷
```json
{"instruction": "질문", "input": "추가 컨텍스트(선택)", "output": "답변"}
```
- JSONL 형식 사용 (한 줄에 하나의 JSON 객체)
- 한국어 인코딩: UTF-8 필수

---

## 테스트 규칙

### 필수 테스트 범위
- API 엔드포인트: 모든 라우트에 대해 정상/에러 케이스 테스트
- RAG 파이프라인: 인덱싱, 검색, 리랭킹 각 단계별 테스트
- 프롬프트 템플릿: 변수 치환 및 포맷 검증

### 테스트 원칙
- 프레임워크: `pytest` + `pytest-asyncio`
- LLM 추론 테스트는 모킹 허용 (실제 모델 로딩은 통합 테스트에서만)
- 벡터 DB 테스트는 임시 디렉터리에 실제 ChromaDB 인스턴스 사용 (모킹 금지)
- 테스트 파일명: `test_{모듈명}.py`
- 픽스처는 `conftest.py`에 집중

---

## 보안 및 개인정보

- 모든 데이터는 로컬에서만 처리. 외부 API 호출 금지 (모델 다운로드 제외)
- 사용자 질의에 포함된 개인정보는 로그에 기록하지 않음
- API 응답에 반드시 면책 조항 포함:
  `"본 답변은 참고용이며, 구체적 법률 조언은 변호사와 상담하시기 바랍니다."`
- `.env` 파일은 `.gitignore` 대상

---

## 하드웨어 제약 (M4 48GB)

- 모델은 반드시 Q4 양자화 버전 사용
- 배치 크기는 4 이하로 제한
- 동시 추론 요청은 1개로 제한 (큐잉 처리)
- 메모리 모니터링: 모델 로딩 후 가용 메모리 확인 로직 포함

---

## 커밋 규칙

```
<type>(<scope>): <subject>

type: feat, fix, refactor, test, docs, chore
scope: api, rag, llm, training, ui, data
```

예시: `feat(rag): 판례 문서 청킹 파이프라인 추가`

---

## 자주 사용하는 명령어

```bash
# 개발 서버 실행
uvicorn src.api.main:app --reload --port 8000

# UI 실행
streamlit run ui/app.py

# 테스트
pytest tests/ -v

# 린트
ruff check src/ && ruff format --check src/

# RAG 인덱싱
python -m src.rag.indexer --data-dir data/raw --db-dir data/vectordb

# Fine-tuning
python -m src.training.train_lora --config config.yaml
```
