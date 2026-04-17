# 05. RAG 파이프라인 구축

## 1. RAG 아키텍처 개요

```
사용자 질의
    │
    ▼
┌─────────────┐
│ Query 전처리 │  ← 질의 정규화, 법률 용어 확장
└──────┬──────┘
       ▼
┌─────────────┐
│  BGE-M3     │  ← 질의 임베딩 생성
│  Embedding  │
└──────┬──────┘
       ▼
┌─────────────┐
│  ChromaDB   │  ← 벡터 유사도 검색 (top-k=5)
│  검색       │
└──────┬──────┘
       ▼
┌──────────────────┐
│  KoSentenceBERT  │  ← 리랭킹 (top-k=5 → top-3)
│  리랭킹          │
└──────┬───────────┘
       ▼
┌─────────────┐
│ 컨텍스트    │  ← 최종 3개 문서를 LLM 프롬프트에 삽입
│ 구성        │
└─────────────┘
```

---

## 2. 인덱싱 (문서 → 벡터DB)

### 2.1 사전 준비

```bash
# 전처리된 문서가 data/raw/ 하위에 있는지 확인
ls data/raw/statutes/
ls data/raw/cases/
ls data/raw/glossary/
```

### 2.2 인덱싱 실행

```bash
python -m src.rag.indexer --data-dir data/raw --db-dir data/vectordb
```

**예상 소요 시간 (M4 기준):**

| 문서 규모 | 임베딩 시간 | 저장 시간 |
|----------|-----------|----------|
| 1,000건 | ~5분 | ~1분 |
| 5,000건 | ~25분 | ~3분 |
| 10,000건 | ~50분 | ~5분 |

### 2.3 청킹 전략 상세

#### 법조문 (조항 단위)

```
입력: 민법 전문 (1,118조)
청킹: 조항별 1개 청크
결과: ~1,118개 청크

장점: 검색 정확도 높음, 출처 명확
단점: 일부 조항이 매우 짧을 수 있음
```

#### 판례 (고정 크기)

```
입력: 판례 1건 (수천 자)
청킹: 512 토큰 단위, 64 토큰 오버랩
결과: 판례당 ~5-15개 청크

장점: 긴 판례도 검색 가능
단점: 문맥이 잘릴 수 있음 (오버랩으로 완화)
```

#### 용어 사전 (용어 단위)

```
입력: 용어-정의 쌍
청킹: 용어 1개 = 1개 청크
결과: 용어 수만큼 청크
```

### 2.4 인덱싱 결과 검증

```python
import chromadb

client = chromadb.PersistentClient(path="data/vectordb")
collection = client.get_collection("legal_documents")

print(f"총 인덱싱된 문서 수: {collection.count()}")

# 샘플 검색 테스트
results = collection.query(
    query_texts=["임대차 보증금 반환"],
    n_results=3,
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta.get('doc_type')}] {meta.get('title')}")
    print(f"  내용: {doc[:100]}...")
    print()
```

---

## 3. 검색 파라미터 튜닝

### 3.1 주요 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `top_k_retrieve` | 5 | 3-10 | 벡터 검색 후보 수 |
| `top_k_rerank` | 3 | 1-5 | 리랭킹 후 최종 선택 수 |
| `chunk_size` | 512 | 256-1024 | 청크 크기 (토큰) |
| `chunk_overlap` | 64 | 0-128 | 청크 오버랩 (토큰) |

### 3.2 튜닝 가이드

```
검색 정확도가 낮을 때:
  → top_k_retrieve 증가 (5 → 8)
  → chunk_size 감소 (512 → 256)

응답이 너무 느릴 때:
  → top_k_retrieve 감소 (5 → 3)
  → top_k_rerank 감소 (3 → 2)

컨텍스트가 너무 길어질 때:
  → top_k_rerank 감소 (3 → 2)
  → chunk_size 감소

관련 없는 문서가 포함될 때:
  → 리랭킹 score 임계값 추가 (예: score > 0.5만 사용)
```

### 3.3 검색 품질 평가 방법

```python
# 수동 평가 세트 구성
eval_queries = [
    {
        "query": "전세 보증금을 돌려받지 못하면 어떻게 하나요?",
        "expected_docs": ["주택임대차보호법", "민법 제312조"],
    },
    {
        "query": "교통사고 손해배상 범위는?",
        "expected_docs": ["민법 제750조", "손해배상 판례"],
    },
]

# 각 질의에 대해 검색 결과 확인
for item in eval_queries:
    results = retriever.search(item["query"])
    retrieved_titles = [r.title for r in results]
    hit = any(exp in " ".join(retrieved_titles) for exp in item["expected_docs"])
    print(f"{'✅' if hit else '❌'} {item['query'][:30]}... → {retrieved_titles}")
```

---

## 4. 인덱스 관리

### 4.1 인덱스 재구축

데이터 추가/수정 시 전체 재구축이 필요합니다:

```bash
# 기존 인덱스 삭제 후 재구축
rm -rf data/vectordb
python -m src.rag.indexer --data-dir data/raw --db-dir data/vectordb
```

### 4.2 인덱스 백업

```bash
# 인덱싱에 시간이 오래 걸리므로 백업 권장
cp -r data/vectordb data/vectordb_backup_$(date +%Y%m%d)
```

---

## 5. 체크리스트

- [ ] 전처리된 문서가 `data/raw/` 하위에 준비됨
- [ ] BGE-M3 임베딩 모델 다운로드 완료
- [ ] `python -m src.rag.indexer` 실행 성공
- [ ] ChromaDB 인덱스 문서 수 확인
- [ ] 샘플 질의 검색 테스트 통과 (3건 이상)
- [ ] 리랭킹 후 관련 문서가 상위에 위치하는지 확인
- [ ] 검색 응답 시간 측정 (목표: 1초 이내)
- [ ] config.yaml의 RAG 파라미터 최적값 확정
