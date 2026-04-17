# 학습 가이드 02: RAG (Retrieval-Augmented Generation) 심화

> 이 프로젝트의 핵심 아키텍처인 RAG의 원리와 최적화 방법을 깊이 이해하기 위한 가이드

---

## 1. RAG가 무엇이고 왜 필요한가

### 기본 LLM의 한계

```
사용자: "주택임대차보호법 제3조의3 내용이 뭔가요?"

LLM만 사용 (Fine-tuning만):
→ 학습 시점의 지식에 의존
→ 정확한 조문을 외우지 못하면 환각 발생
→ 법률 개정 사항 반영 불가

RAG + LLM:
→ 벡터DB에서 제3조의3 원문을 검색
→ 검색된 원문을 컨텍스트로 LLM에 전달
→ LLM이 원문을 기반으로 설명 생성
→ 항상 최신 법률 반영 가능 (인덱스 갱신만 하면 됨)
```

### RAG의 핵심 가치

| 가치 | 설명 |
|------|------|
| **정확성** | 실제 문서에 근거한 답변 |
| **최신성** | 인덱스 업데이트만으로 새 정보 반영 |
| **투명성** | 출처를 명시할 수 있음 |
| **비용 효율** | 모든 지식을 모델에 학습시킬 필요 없음 |

---

## 2. 임베딩 (Embedding)

### 2.1 벡터 임베딩이란?

```
텍스트 → [0.012, -0.034, 0.078, ..., 0.045]  (1024차원 벡터)

"임대차 보증금 반환" → [0.23, -0.15, ...]
"전세 보증금 돌려받기" → [0.22, -0.14, ...]   ← 의미가 비슷하면 벡터도 가까움
"형사 고소장 작성"    → [-0.31, 0.42, ...]    ← 의미가 다르면 벡터가 멀리 떨어짐
```

### 2.2 유사도 측정

| 방법 | 수식 | 특징 |
|------|------|------|
| **코사인 유사도** | cos(A, B) = A·B / (\|A\|\|B\|) | 방향 기반, 가장 일반적 |
| 유클리드 거리 | \|\|A - B\|\| | 거리 기반 |
| 내적 (Dot Product) | A · B | 크기 + 방향 |

### 2.3 한국어 임베딩 특수 고려사항

```
문제: "보증금 반환 청구" vs "보증금을 돌려달라는 소송"
→ 표현은 다르지만 의미는 같음
→ 좋은 임베딩 모델은 이 두 문장을 가깝게 배치

문제: "민법 제750조" vs "불법행위로 인한 손해배상"
→ 같은 법조문이지만 표현이 전혀 다름
→ 법률 도메인에서 특히 어려운 과제
→ BGE-M3의 다국어 + 긴 컨텍스트 능력이 중요한 이유
```

### 학습 자료

```
├── "What are Embeddings?" (Vicki Boykis)
│   → https://vickiboykis.com/what_are_embeddings/
│   → 임베딩 개념을 처음부터 설명하는 최고의 자료
│
├── "Sentence Transformers" 공식 문서
│   → https://www.sbert.net/
│
└── "Text Embeddings Visually Explained" (Jay Alammar)
    → https://jalammar.github.io/illustrated-word2vec/
```

### 실습 과제

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")

# 법률 문장 유사도 실험
queries = ["전세 보증금 반환 방법"]
docs = [
    "임대차 보증금을 돌려받기 위한 법적 절차",    # 관련 높음
    "주택임대차보호법 제3조의3 임차권등기명령",    # 관련 중간
    "형사 고소장 작성 시 주의사항",                # 관련 없음
]

q_emb = model.encode(queries)
d_emb = model.encode(docs)
sims = cosine_similarity(q_emb, d_emb)[0]

for doc, sim in sorted(zip(docs, sims), key=lambda x: -x[1]):
    print(f"[{sim:.4f}] {doc}")

# 질문: 결과가 기대와 일치하는가? 어떤 문서가 가장 높은 유사도를 보이는가?
```

---

## 3. 벡터 데이터베이스

### 3.1 왜 벡터 DB가 필요한가?

```
일반 DB:  SELECT * FROM docs WHERE title = '민법 제750조'  → 정확히 일치만 검색
벡터 DB:  "불법행위 손해배상 관련 조문" → 의미적으로 유사한 문서 검색
```

### 3.2 ChromaDB (이 프로젝트 사용)

| 특징 | 설명 |
|------|------|
| 경량 | SQLite 기반, 별도 서버 불필요 |
| 로컬 | 데이터가 로컬에만 저장 (프라이버시) |
| 간편 | Python 네이티브, 설치 간단 |
| 영속성 | `PersistentClient`로 디스크 저장 |

### 3.3 다른 벡터 DB와의 비교 (참고)

| DB | 특징 | 적합한 경우 |
|----|------|-----------|
| **ChromaDB** | 경량, 로컬, 간편 | 이 프로젝트 (로컬, 소규모) |
| Pinecone | 클라우드 관리형 | 대규모 프로덕션 |
| Weaviate | 다양한 모듈 | 멀티모달 검색 |
| Milvus | 고성능 분산 | 대규모 벡터 |
| FAISS | Meta 라이브러리 | 연구/커스텀 필요 시 |

### 학습 자료

```
├── ChromaDB 공식 문서
│   → https://docs.trychroma.com/
│
├── "Vector Database Explained" (Pinecone 블로그)
│   → 벡터 DB의 원리를 잘 설명
│
└── LlamaIndex VectorStore 가이드
    → https://docs.llamaindex.ai/en/stable/module_guides/storing/
```

---

## 4. 청킹 전략 (Chunking)

### 4.1 왜 청킹이 중요한가?

```
문제: 민법 전문은 수만 자 → 하나의 벡터로 임베딩하면 정보 손실
해결: 의미 있는 단위로 나누어(청킹) 각각 임베딩

나쁜 청킹:  문서 중간에서 기계적으로 자름 → "제750조 (불법행위) 고의 또는" | "과실로 인한 위법행위로..."
좋은 청킹:  의미 단위로 자름 → "제750조 (불법행위) 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다."
```

### 4.2 청킹 방법론

| 방법 | 설명 | 적용 대상 |
|------|------|----------|
| **고정 크기** | N 토큰 단위로 분할 + 오버랩 | 판례 (긴 문서) |
| **구분자 기반** | 문장/문단/조항 경계로 분할 | 법조문 (구조화된 문서) |
| **시맨틱** | 의미 변화 지점에서 분할 | 비정형 문서 |
| **재귀적** | 큰 단위 → 작은 단위로 점진 분할 | 범용 |

### 4.3 이 프로젝트의 청킹 전략

```
법조문:  조항 단위 분리 (제1조, 제2조, ...)
  → 각 조항이 하나의 독립적 법적 의미를 가짐
  → 검색 시 정확한 조항을 찾을 수 있음

판례:    512 토큰 + 64 토큰 오버랩
  → 판례는 길고 여러 쟁점을 포함
  → 오버랩으로 문맥 단절 최소화

용어:    용어 1개 = 1청크
  → 정의와 설명이 짧고 독립적
```

### 실습 과제

```python
# 청크 크기에 따른 검색 결과 차이 실험
# 같은 문서를 256, 512, 1024 토큰으로 청킹한 후
# 동일한 질의로 검색하여 결과 비교
```

---

## 5. 리랭킹 (Reranking)

### 5.1 왜 2단계 검색인가?

```
1단계 - 벡터 검색 (Bi-Encoder):
  장점: 매우 빠름 (사전 임베딩 + ANN 검색)
  단점: 질의-문서 상호작용 없이 독립적으로 임베딩

2단계 - 리랭킹 (Cross-Encoder):
  장점: 질의와 문서를 함께 입력하여 정교한 유사도 계산
  단점: 느림 (모든 후보에 대해 추론 필요)

결합: 벡터 검색으로 후보 5개 빠르게 선별 → 리랭킹으로 상위 3개 정밀 선택
```

### 5.2 Bi-Encoder vs Cross-Encoder

```
Bi-Encoder (벡터 검색):
  Query  → [Encoder] → q_vec ─┐
                                ├→ cosine_similarity
  Doc    → [Encoder] → d_vec ─┘

Cross-Encoder (리랭킹):
  [Query + Doc] → [Encoder] → relevance_score
  → 질의와 문서의 토큰 간 직접 어텐션 계산
  → 더 정확하지만 더 느림
```

### 학습 자료

```
├── "Retrieve & Re-Rank" (Sentence Transformers)
│   → https://www.sbert.net/examples/applications/retrieve_rerank/
│
└── "Understanding Reranking in RAG" (LlamaIndex 블로그)
```

---

## 6. RAG 평가 지표

### 6.1 검색 품질 지표

| 지표 | 설명 | 계산 방법 |
|------|------|----------|
| **Recall@K** | 상위 K개에 정답 문서가 포함된 비율 | 정답 포함 쿼리 수 / 전체 쿼리 수 |
| **MRR** (Mean Reciprocal Rank) | 정답 문서의 평균 역순위 | 1/순위의 평균 |
| **nDCG** | 순위 가중 관련도 점수 | 상위에 관련 문서가 올수록 높음 |

### 6.2 응답 품질 지표

| 지표 | 설명 |
|------|------|
| **Faithfulness** | 응답이 검색된 문서에 근거하는가? |
| **Relevance** | 응답이 질문과 관련 있는가? |
| **Completeness** | 필요한 정보를 빠짐없이 포함하는가? |

### 6.3 평가 도구

```
├── RAGAS (RAG 평가 프레임워크)
│   → https://docs.ragas.io/
│   → Faithfulness, Answer Relevancy 등 자동 평가
│
└── LlamaIndex Evaluation 모듈
    → https://docs.llamaindex.ai/en/stable/module_guides/evaluating/
```

---

## 7. 고급 RAG 패턴 (향후 적용 가능)

| 패턴 | 설명 | 효과 |
|------|------|------|
| **Query Expansion** | 질의를 변형/확장하여 여러 번 검색 | 재현율 향상 |
| **HyDE** | 가상 답변을 먼저 생성 후 그걸로 검색 | 질의-문서 갭 해소 |
| **Parent-Child Retrieval** | 작은 청크로 검색, 큰 청크로 컨텍스트 전달 | 검색 정확도 + 컨텍스트 보존 |
| **Self-RAG** | 모델이 스스로 검색 필요 여부를 판단 | 불필요한 검색 감소 |

---

## 8. 학습 체크리스트

| # | 주제 | 이해 수준 | 필수 여부 |
|---|------|----------|----------|
| 1 | RAG의 목적과 기본 LLM 대비 장점 | 개념 이해 | 필수 |
| 2 | 벡터 임베딩과 코사인 유사도 | 실습 | 필수 |
| 3 | 한국어 임베딩 특성과 BGE-M3 사용법 | 실습 | 필수 |
| 4 | ChromaDB 기본 CRUD 및 검색 | 실습 | 필수 |
| 5 | 청킹 전략별 장단점 | 개념 이해 | 필수 |
| 6 | Bi-Encoder vs Cross-Encoder 차이 | 개념 이해 | 필수 |
| 7 | Recall@K, MRR 등 검색 평가 지표 | 개념 이해 | 권장 |
| 8 | RAGAS 프레임워크 사용법 | 실습 | 선택 |
| 9 | HyDE, Parent-Child 등 고급 패턴 | 개념 이해 | 선택 |
