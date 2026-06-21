원천 데이터를 수집.정제.청크화하여 Chroma 벡터DB에 인덱싱하는 ETL 단계
(ETL: 각 글자는 **Extract(추출)**, **Transform(변환)**, Load(적재)의 앞 글자를 딴 것으로, 쉽게 말해 "여기저기 흩어진 데이터를 가져와서, 쓸 만하게 맛있는 요리로 만든 뒤, 원하는 그릇에 담는 과정"이라고 이해)

##### ① Extract — `fetch_corpus.collect` ([fetch_corpus.py](https://file+.vscode-resource.vscode-cdn.net/Users/fujii0711/Claude/PrivateLLM/docs/export/pipelines/src/pipelines/ingest/fetch_corpus.py))

- 법령(`주택임대차보호법`,`민법`)·판례(보증금 반환 관련 3쿼리) 수집.
- **견고성 처리 두 가지**:
    - `_match_law_mst`: 검색이 가나다순이라 "민법" 검색 시 "**난민법**"이 먼저 나오는 문제 → **법령명 정확 일치** 항목을 골라 엉뚱한 법령 수집 방지(없으면 첫 결과 폴백).
    - 판례: `seen` set으로 **중복 제거**, `<PrecService>` 없는 빈 본문·`case_no` 없는 결과 스킵.
- `law_client`(HTTP) → `law_parser`/`prec_parser`(파싱) → `_write`(raw JSON). `source_url`·`prec_id` 부착.

##### ② Transform — `chunker.build_all` ([chunker.py](https://file+.vscode-resource.vscode-cdn.net/Users/fujii0711/Claude/PrivateLLM/docs/export/pipelines/src/pipelines/chunk/chunker.py))

- **구조 인지 청킹**(길이 기반 아님): 법령은 **조문 1개 = 청크 1개**(`law-{법}-{조}`), 판례는 **판시사항/판결요지 각각 청크**(`prec-{id}-{판시사항|판결요지}`).
- 각 청크 본문은 `clean.normalize`로 정규화, `schema.make_chunk`로 **source_type 검증**(`법령/판례/해설/상담사례`만 허용) 후 생성.
- ⚠️ `해설`·`상담사례`는 스키마에만 있고 현재 수집 안 함(메모리의 "3B에서 채울 여지").

##### ③ Load — `build_index.build_index` ([build_index.py](https://file+.vscode-resource.vscode-cdn.net/Users/fujii0711/Claude/PrivateLLM/docs/export/pipelines/src/pipelines/index/build_index.py))

- chunks.jsonl 로드 → 64개씩 배치 → `embedder.embed`(bge-m3, **normalize=True**) → Chroma `upsert`.
- 컬렉션 `jeonse_deposit`, **`hnsw:space=cosine`**. 메타데이터(source_type/title/ref/url/date)도 함께 저장 → rag가 검색 시 그대로 꺼냄.
- `upsert`라 **재실행 시 id 기준 갱신**(멱등) → 블루/그린 재색인의 기반.

pipelines  
 ┣ scripts  
 ┃ ┗ capture_fixtures.py  
 ┣ src  
 ┃ ┗ pipelines  
 ┃ ┃ ┣ chunk
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┗ chunker.py  (조문/판례 단위 청크 분할, ★구조 인지 청킹 → `chunks.jsonl`)
 ┃ ┃ ┣ clean  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┗ normalize.py  (텍스트 정규화, 공백·soft hyphen·nbsp·개행 정규화)
 ┃ ┃ ┣ cli  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┗ query.py  (인덱스 검색 확인용 CLI, 검색 스모크 테스트 CLI)
 ┃ ┃ ┣ index  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ build_index.py  (Chroma jeonse_deposit 컬렉션 빌드, ★청크 → Chroma upsert(cosine))
 ┃ ┃ ┃ ┗ embedder.py  (bge-m3 임베딩 래퍼(`encode_fn` 주입))
 ┃ ┃ ┣ ingest  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ fetch_corpus.py (★수집 오케스트레이션 → `data/raw/{law,prec}/*.json`)
 ┃ ┃ ┃ ┣ law_client.py (국가법령정보 DRF API HTTP 클라이언트(search/fetch))
 ┃ ┃ ┃ ┣ law_parser.py (법령 XML → `{law_name, articles[]}` (가지조문 처리))
 ┃ ┃ ┃ ┗ prec_parser.py (판례 XML → `{판시사항, 판결요지, ...}`)
 ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┣ config.py (경로·OC키 구성(`data_root`→raw/chunks/chroma))
 ┃ ┃ ┗ schema.py (`Chunk` TypedDict + `make_chunk`(source_type 검증))
 ┣ tests  
 ┃ ┣ fixtures  
 ┃ ┃ ┣ law_search.xml  
 ┃ ┃ ┣ law_주택임대차보호법.xml  
 ┃ ┃ ┣ prec_one.xml  
 ┃ ┃ ┗ prec_search.xml  
 ┃ ┣ __init__.py  
 ┃ ┣ conftest.py  
 ┃ ┣ test_build_index.py  
 ┃ ┣ test_chunker.py  
 ┃ ┣ test_config.py  
 ┃ ┣ test_embedder.py  
 ┃ ┣ test_fetch_corpus.py  
 ┃ ┣ test_law_client.py  
 ┃ ┣ test_law_parser.py  
 ┃ ┣ test_normalize.py  
 ┃ ┣ test_prec_parser.py  
 ┃ ┣ test_query.py  
 ┃ ┗ test_schema.py  
 ┣ .DS_Store  
 ┣ .env  
 ┣ .env.example  
 ┣ .python-version  
 ┣ README.md  
 ┗ pyproject.toml