# 대한민국헌법 RAG 시스템

  
대한민국헌법 PDF를 소스로 하는 로컬 RAG(검색 증강 생성) 시스템. 조문 단위의 정확한 인용과 다턴 대화를 제공한다.
  
## 대상 환경

- **개발/실행 머신**: MacBook Pro M4, 메모리 48GB, macOS

- **실행 형태**: 완전 로컬 (네트워크 불필요, API 비용 없음)

- **Python**: 3.11 이상

- **패키지 관리**: [uv](https://github.com/astral-sh/uv) (pyproject.toml + uv.lock)


## 기술 스택


| 계층 | 선택 | 비고 |

|------|------|------|

| LLM | Ollama + **Qwen2.5 14B** | 한국어 품질 우수, M4 48GB에서 여유롭게 구동 |

| 임베딩 | **BGE-M3** (BAAI/bge-m3) | 다국어·한국어 법률 문서 성능 양호, 로컬 실행 |

| 리랭커 | **BAAI/bge-reranker-v2-m3** | 다국어 크로스 인코더 |

| 벡터 DB | **ChromaDB** (persistent, 로컬 파일) | 설치·운용 단순 |

| 청크 분할 | LangChain `RecursiveCharacterTextSplitter` | 문단/문장 우선순위로 자연 경계 확보 |

| UI | **Streamlit** | 브라우저 기반 질의응답, 출처 표시 |

| 원문 로더 | `pypdf` 또는 `pdfplumber` | 한글 추출 품질 확인 후 결정 |

| 프레임워크 | LangChain (선택적) | 필요한 부분만 사용, 과도한 래핑 지양 |

  
## 저장소 구조

```

RAGTest/

├── CLAUDE.md # 본 문서

├── README.md # 사용자용 README (별도 생성)

├── pyproject.toml # uv 프로젝트 정의

├── uv.lock

├── .env.example # 환경변수 템플릿

├── 대한민국헌법.pdf # 원본 문서 (소스)

├── data/

│ ├── chroma/ # 벡터 DB persistent 디렉토리 (gitignore)

│ └── raw/ # 추출된 원문 캐시 (gitignore)

├── src/

│ ├── __init__.py

│ ├── config.py # 모델명, 경로, 하이퍼파라미터 중앙화

│ ├── loader.py # PDF → 텍스트 추출

│ ├── chunker.py # RecursiveCharacterTextSplitter 래퍼

│ ├── embedder.py # BGE-M3 로딩 및 임베딩

│ ├── indexer.py # Chroma 컬렉션 생성·upsert

│ ├── retriever.py # top-20 검색 + 리랭커로 top-5 재정렬

│ ├── llm.py # Ollama 클라이언트

│ ├── rag.py # 검색→프롬프트→생성 파이프라인 + 다턴 메모리

│ └── prompts.py # 시스템 프롬프트, 인용 포맷 템플릿

├── scripts/

│ ├── reindex.py # PDF 재파싱 및 벡터 DB 재구축

│ └── eval.py # 평가 세트로 리포트 생성

├── evaluation/

│ ├── testset.json # 사용자 작성 질문-정답(조문 번호) 세트

│ └── reports/ # 평가 리포트 출력 (gitignore)

├── app/

│ └── streamlit_app.py # Streamlit 엔트리포인트

└── tests/

└── test_retriever.py 등

```

  
## 핵심 동작

### 인덱싱 파이프라인 (`scripts/reindex.py`)

1. `대한민국헌법.pdf` 로드 → 원문 텍스트 추출

2. `RecursiveCharacterTextSplitter`로 청크 생성

- separators 우선순위: `["\n\n", "\n", ". ", " ", ""]`

- chunk_size/overlap은 `src/config.py`에서 조정 (초기값: 500 / 80)

3. 각 청크에 메타데이터 부여: `source`, `page`, 가능하면 `article`(조문 번호)

4. BGE-M3로 임베딩 → Chroma `constitution` 컬렉션에 upsert

### 질의 파이프라인 (`src/rag.py`)

1. 사용자 질문을 BGE-M3로 임베딩

2. Chroma에서 코사인 유사도 top-20 검색

3. BGE-reranker-v2-m3로 top-5 재정렬

4. 시스템 프롬프트 + 검색 결과 + 대화 이력을 Qwen2.5 14B에 전달

5. 응답과 함께 참조한 청크의 출처 메타데이터(조문 번호·페이지) 반환

6. 다턴 메모리는 최근 N턴(기본 5)만 유지하여 컨텍스트 폭주 방지

### 출처 인용

- 답변 본문에는 각 주장 뒤에 `[헌법 제OO조]` 형식 인라인 인용

- 답변 하단에 "참조 원문" 블록으로 실제 청크 원문 + 페이지 표시

- 인용 누락 시 모델이 "확인 불가"로 응답하도록 시스템 프롬프트 강제

## 주요 명령
```bash
# 초기 환경 구축

uv sync


# Ollama 모델 준비 (최초 1회)

ollama pull qwen2.5:14b

# 벡터 DB 구축 / 재구축

uv run python scripts/reindex.py

# Streamlit UI 실행

uv run streamlit run app/streamlit_app.py

# 평가 실행

uv run python scripts/eval.py --testset evaluation/testset.json

```

## 평가
- `evaluation/testset.json` 포맷 (사용자 직접 작성):

```json
[

{

"question": "대통령의 임기는?",

"expected_articles": ["제70조"],

"expected_answer_keywords": ["5년", "중임할 수 없다"]

}

]

```

- 리포트 지표: Recall@5 (정답 조문이 top-5에 포함되었는가), 키워드 포함률, 평균 응답 시간

- 리포트는 `evaluation/reports/YYYY-MM-DD_HHMM.md`로 저장

## 개발 규약

- **비밀정보**: API 키가 필요 없는 로컬 스택이므로 `.env`는 선택. 사용 시 `.env.example`로 템플릿화

- **설정값 중앙화**: 모델명·경로·top_k·청크 크기 등 모든 튜너블 값은 `src/config.py`에 모은다. 코드 여러 곳에 리터럴을 흩뿌리지 않는다

- **로그**: 인덱싱·검색 단계별로 `logging` 사용. print 지양

- **테스트**: retriever와 chunker의 골든 케이스 위주. LLM 호출 테스트는 비결정적이므로 평가 스크립트로 대체

- **의존성 추가**: `uv add <패키지>`로만 추가. `requirements.txt` 직접 편집 금지

## 미결정 사항 / 향후 과제

- PDF 추출 품질 확인 후 `pypdf` ↔ `pdfplumber` 확정

- 조문 번호 자동 태깅(`article` 메타데이터) 정규식 작성 필요

- 필요 시 BM25 하이브리드 검색으로 확장 (현재는 벡터+리랭커만)

- Streamlit UI의 세션별 대화 히스토리 저장/내보내기 기능은 v2