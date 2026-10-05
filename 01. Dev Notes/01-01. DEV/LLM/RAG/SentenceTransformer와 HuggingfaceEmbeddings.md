
둘은 **임베딩 모델 자체의 차이라기보다, 사용하는 인터페이스/추상화 계층의 차이**라고 보는 것이 정확합니다.

예를 들어 같은 `BAAI/bge-m3` 모델을 사용한다면:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
vector = model.encode("안녕하세요")
```

와

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)
vector = embeddings.embed_query("안녕하세요")
```

는 **결국 같은 Sentence Transformers 계열 모델을 사용해 임베딩을 생성할 수 있습니다.**

### 핵심 차이

|구분|`sentence_transformers`|`HuggingFaceEmbeddings`|
|---|---|---|
|성격|임베딩 모델을 직접 사용하는 라이브러리|LangChain용 Embeddings 래퍼|
|사용 목적|임베딩 자체를 직접 제어|LangChain RAG/VectorStore와 연결|
|모델|SentenceTransformer 모델|내부적으로 SentenceTransformer 사용 가능|
|`encode()`|직접 사용|내부적으로 호출|
|LangChain 호환|❌|✅|
|Chroma/Qdrant 등 연결|직접 구현|LangChain VectorStore와 바로 연결|
|세부 옵션 제어|많음|`model_kwargs`, `encode_kwargs`로 전달|
|성능|직접 사용이므로 약간 더 단순|거의 동일|

즉 구조적으로 보면 대략:

```text
sentence_transformers
        │
        │  직접 사용
        ▼
SentenceTransformer
        │
        ▼
embedding vector
```

반면 LangChain은:

```text
LangChain
   │
   ▼
HuggingFaceEmbeddings
   │
   ▼
SentenceTransformer
   │
   ▼
embedding vector
```

입니다.

### 중요한 점

예를 들어:

```python
HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={
        "normalize_embeddings": True
    }
)
```

는 내부적으로 Sentence Transformers의 `encode()` 계열 기능을 이용합니다.

따라서 **동일한 모델 + 동일한 encode 옵션**을 사용하면 결과 벡터는 기본적으로 동일한 방향의 임베딩이 나옵니다.

---

### 그런데 BGE-M3를 쓰신다면 한 가지 주의

현재 하시려는 RAG 구조에서 `BAAI/bge-m3`를 사용한다면 단순히 다음처럼 생각하면 됩니다.

```python
# 모델 직접 사용
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

vectors = model.encode(
    texts,
    normalize_embeddings=True
)
```

vs

```python
# LangChain
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={
        "normalize_embeddings": True
    }
)

vectors = embeddings.embed_documents(texts)
```

**임베딩 품질 자체를 높여주는 것은 `HuggingFaceEmbeddings`가 아닙니다.**

실제 품질을 결정하는 것은 주로:

```text
사용 모델
   ↓
BAAI/bge-m3
   ↓
텍스트 전처리
   ↓
chunking
   ↓
embedding 옵션
   ↓
vector DB
```

입니다.

그래서 사용자가 구축하고 있는 **LangGraph + Qdrant + RAG** 구조라면 저는 보통:

```python
HuggingFaceEmbeddings
```

를 추천합니다.

왜냐하면 이후에

```python
QdrantVectorStore
        ↓
retriever
        ↓
LangGraph
        ↓
LLM
```

으로 연결하기가 훨씬 편하기 때문입니다.

반대로 **임베딩 벤치마크를 하거나, 배치 임베딩 성능을 최대한 직접 제어하거나, LangChain을 사용하지 않는다면** `SentenceTransformer`를 직접 사용하는 편이 좋습니다.

원하시면 제가 이어서 **`SentenceTransformer("BAAI/bge-m3")`와 `HuggingFaceEmbeddings("BAAI/bge-m3")`가 실제로 내부에서 어떤 코드 경로를 거치는지**까지 보여드릴 수 있습니다.