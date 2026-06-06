
> 수집 → 정제 → 청킹 → 색인

**Goal**: 주택임대차 보증금 반환 도메인의 법령·판례를 국가법령정보센터 OPEN API로 수집·정제·청킹하고
bge-m3로 임베딩해 Chroma 벡터 인덱스를 구축한다. CLI 질의로 검색이 동작함을 검증한다.

**Architecture**: Python(`uv`) 단일 패키지 `pipelines`. HTTP 클라이언트 → XML 파서 → 정규화 → 구조 인지 청킹 → bge-m3 임베딩 → Chroma 색인의 단방향 파이프라인. 외부 API 응답 형식의 불확실성은 "실제 호출로 fixture를 캡처한 뒤 그 fixture에 대해 TDD로 파서를 작성"하는 방식으로 제거한다.

**Tech Stack**: Python 3.11+, `uv`, `requests`, `lxml`, `sentence-transformers`(BAAI/bge-m3, MPS), `chromadb`, `pytest`.

```
pipelines/
├── pyproject.toml                 # uv 프로젝트 정의 + 의존성
├── .python-version                # 3.11
├── .env.example                   # LAW_API_OC 등 템플릿
├── README.md                      # 실행법
├── src/pipelines/
│   ├── __init__.py
│   ├── schema.py                  # Chunk TypedDict (스펙 5.3)
│   ├── config.py                  # 환경변수·경로 로딩
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── law_client.py          # law.go.kr HTTP 클라이언트(검색+본문 URL 빌드/호출)
│   │   ├── law_parser.py          # 법령 XML → 구조화 dict
│   │   ├── prec_parser.py         # 판례 XML → 구조화 dict
│   │   └── fetch_corpus.py        # 대상 법령·판례 수집 → data/raw/*.json
│   ├── clean/
│   │   ├── __init__.py
│   │   └── normalize.py           # 텍스트 정규화(공백·특수문자)
│   ├── chunk/
│   │   ├── __init__.py
│   │   └── chunker.py             # 구조 인지 청킹 → data/chunks/chunks.jsonl
│   ├── index/
│   │   ├── __init__.py
│   │   ├── embedder.py            # bge-m3 래퍼(주입 가능)
│   │   └── build_index.py         # 청크 임베딩 → Chroma
│   └── cli/
│       ├── __init__.py
│       └── query.py               # 검색 스모크 테스트 CLI
├── scripts/
│   └── capture_fixtures.py        # 실제 API 호출 → tests/fixtures 저장
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── fixtures/                  # 캡처된 실제 XML + 샘플 청크
    ├── test_config.py
    ├── test_law_client.py
    ├── test_law_parser.py
    ├── test_prec_parser.py
    ├── test_normalize.py
    ├── test_chunker.py
    ├── test_embedder.py
    └── test_build_index.py
```

---

uv run pytest -q
uv run python -m pipelines.ingest.fetch_corpus
uv run pytest
uv run pytest -m slow
uv run pytest tests/test_schema.py -v
cd pipelines && set -a && source .env && set +a
cd pipelines && uv run python -c "from lxml import etree; r=etree.parse('tests/fixtures/law_주택임대차보호법.xml').getroot(); print([e.tag for e in r.iter()][:40])"
uv run --package rag pytest -q

---

Task 1: 청크 스키마 정의
**Files:**
- Create: `pipelines/src/pipelines/schema.py`
- Test: `pipelines/tests/test_schema.py`
스펙 5.3의 청크 스키마를 코드로 고정한다. 이후 모든 task가 이 타입을 참조한다.

Task 2: 설정 모듈 (경로 + OC 키)
**Files:**
- Create: `pipelines/src/pipelines/config.py`
- Test: `pipelines/tests/test_config.py`

Task 3: law.go.kr HTTP 클라이언트 (URL 빌드 + 호출)
**Files:**
- Create: `pipelines/src/pipelines/ingest/__init__.py` (빈 파일)
- Create: `pipelines/src/pipelines/ingest/law_client.py`
- Test: `pipelines/tests/test_law_client.py`
URL 구성 규칙은 OPEN API 문서로 확정돼 있어 단위 테스트로 검증한다(네트워크 없이). 실제 HTTP 호출은 주입된 세션을 가짜로 대체해 테스트한다.

API 엔드포인트:
- 검색: `https://www.law.go.kr/DRF/lawSearch.do?OC={oc}&target={law|prec}&type=XML&query={q}&display={n}`
- 본문: `https://www.law.go.kr/DRF/lawService.do?OC={oc}&target={law|prec}&type=XML&{ID|MST}={id}`

Task 4: 실제 fixture 캡처 (네트워크, 수동 실행)
**Files:**
- Create: `pipelines/scripts/capture_fixtures.py`
- Output: `pipelines/tests/fixtures/law_주택임대차보호법.xml`, `prec_search.xml`, `prec_one.xml`
이후 파서 task들이 **추측이 아닌 실제 응답**에 대해 작성되도록, 대표 응답을 캡처해 저장한다.

Task 5: 법령 파서 (XML → 조문 구조)
**Files:**
- Create: `pipelines/src/pipelines/ingest/law_parser.py`
- Test: `pipelines/tests/test_law_parser.py`
캡처된 `law_주택임대차보호법.xml` 에 대해 조문 단위 추출을 검증한다.

Task 6: 판례 파서 (XML → 판시사항/판결요지)
**Files:**
- Create: `pipelines/src/pipelines/ingest/prec_parser.py`
- Test: `pipelines/tests/test_prec_parser.py`
캡처된 `prec_one.xml` 에 대해 검증한다. 판례 본문은 `판시사항`, `판결요지`, `판례내용`, `사건명`, `사건번호`, `선고일자`, `법원명` 요소를 가진다.

Task 7: 수집 오케스트레이션 → data/raw
**Files:**
- Create: `pipelines/src/pipelines/ingest/fetch_corpus.py`
- Test: `pipelines/tests/test_fetch_corpus.py`
대상 법령(주택임대차보호법, 민법) 본문과 보증금 반환 판례를 수집해 `data/raw/`에 구조화 JSON으로 저장한다. 네트워크 의존 부분(클라이언트)은 주입해 테스트한다.

Task 8: 텍스트 정규화
**Files:**
- Create: `pipelines/src/pipelines/clean/__init__.py` (빈 파일)
- Create: `pipelines/src/pipelines/clean/normalize.py`
- Test: `pipelines/tests/test_normalize.py`

Task 9: 구조 인지 청킹 → data/chunks/chunks.jsonl
**Files:**
- Create: `pipelines/src/pipelines/chunk/__init__.py` (빈 파일)
- Create: `pipelines/src/pipelines/chunk/chunker.py`
- Test: `pipelines/tests/test_chunker.py`
`data/raw/`의 법령·판례 JSON을 읽어 스펙 5.3 청크로 변환한다. 법령=조문 단위, 판례=판시사항·판결요지 단위.

Task 10: 임베딩 래퍼 (bge-m3)
**Files:**
- Create: `pipelines/src/pipelines/index/__init__.py` (빈 파일)
- Create: `pipelines/src/pipelines/index/embedder.py`
- Test: `pipelines/tests/test_embedder.py`
`Embedder`는 인코딩 함수를 주입받아 단위 테스트 가능하게 하고, 기본은 bge-m3(MPS)를 지연 로딩한다.

Task 11: Chroma 색인 구축
**Files:**
- Create: `pipelines/src/pipelines/index/build_index.py`
- Test: `pipelines/tests/test_build_index.py`
질의를 임베딩해 Chroma에서 top-k를 검색해 출력한다. 계획 2의 `packages/rag` 검색 로직의 원형이 된다.