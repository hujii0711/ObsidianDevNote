
정확히는 "답변"이 아니라 **검색된 원문 텍스트(문서 조각)**가 조립되는 거예요. 이 차이가 중요합니다.

**조립되는 건 "답"이 아니라 "근거 자료"**

```python
def build_prompt(query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    prompt = f"""다음 문서를 참고해서 질문에 답하세요.

[참고 문서]
{context}          ← 여기 들어가는 게 "답변"이 아니라 "원문 청크"

[질문]
{query}

[답변]"""
```

벡터DB에서 검색으로 찾아낸 건 **미리 만들어둔 답이 아니라, 인덱싱 단계에서 저장해둔 원본 문서 조각**이에요.

```python
docs = [
    "회사 연차는 입사 1년 후부터 매년 15일이 자동 부여됩니다.",   # ← 원문 그대로
    "출산휴가는 90일이며, 그 중 45일은 출산 후 사용해야 합니다.",
    "재택근무는 팀장 승인 하에 주 2회까지 가능합니다.",
]
```

**흐름을 다시 짚으면**

```
1. 사용자 질문: "재택근무는 얼마나 자주 가능한가요?"
   ↓
2. 벡터DB 검색 결과: "재택근무는 팀장 승인 하에 주 2회까지 가능합니다."  ← 이게 원문 청크
   ↓
3. 프롬프트 조립:
   [참고 문서]
   - 재택근무는 팀장 승인 하에 주 2회까지 가능합니다.
   
   [질문]
   재택근무는 얼마나 자주 가능한가요?
   ↓
4. LLM이 이 프롬프트를 읽고 "주 2회"라는 정보를 뽑아내서 
   자연스러운 문장으로 답변을 "생성"함:
   → "재택근무는 팀장 승인을 받으면 주 2회까지 가능합니다."
```

**즉 정리하면**

|단계|내용|
|---|---|
|벡터DB에 있는 것|원본 문서/청크 (사람이 쓴 그대로의 텍스트)|
|검색으로 찾아오는 것|질문과 유사한 원본 문서 조각 (답이 아님)|
|프롬프트에 삽입되는 것|그 원본 문서 조각 그대로|
|LLM이 생성하는 것|문서 조각을 읽고 질문에 맞게 재구성한 최종 답변|

만약 벡터DB에 "미리 만든 답변"을 저장해뒀다면 그건 RAG가 아니라 그냥 **FAQ 매칭 시스템**이 돼버려요. RAG의 핵심은 원문(근거)만 주고, **답을 만드는 건 LLM이 실시간으로 한다**는 데 있습니다. 그래야 질문 표현이 조금 달라져도(예: "재택 얼마나 자주 돼요?") 같은 문서를 찾아서 자연스럽게 답을 생성할 수 있어요.

---

아래는 전체 흐름을 실제로 동작하는 최소 예제로 구현한 코드예요. (Chroma + sentence-transformers + MLX 조합, 지금까지 얘기한 Mac 환경 기준)

```python
# ============================================
# 1. 임베딩 모델 & 벡터DB 준비
# ============================================
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("BAAI/bge-m3", device="mps")

client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_or_create_collection("my_docs")

# ============================================
# 2. (사전 작업) 문서 인덱싱 - 최초 1회만 실행
# ============================================
docs = [
    "회사 연차는 입사 1년 후부터 매년 15일이 자동 부여됩니다.",
    "출산휴가는 90일이며, 그 중 45일은 출산 후 사용해야 합니다.",
    "재택근무는 팀장 승인 하에 주 2회까지 가능합니다.",
]

doc_vectors = embedder.encode(docs).tolist()

collection.add(
    embeddings=doc_vectors,
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

# ============================================
# 3. 사용자 질문 → 벡터화 → 유사도 검색 (top-k)
# ============================================
def retrieve(query: str, k: int = 3):
    query_vector = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=k
    )

    retrieved_docs = results["documents"][0]     # 검색된 문서 리스트
    distances = results["distances"][0]           # 유사도 거리 (낮을수록 유사)

    return retrieved_docs, distances

# ============================================
# 4. 검색된 문서를 프롬프트에 삽입
# ============================================
def build_prompt(query: str, retrieved_docs: list[str]) -> str:
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    prompt = f"""다음 문서를 참고해서 질문에 답하세요. 문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

[참고 문서]
{context}

[질문]
{query}

[답변]"""
    return prompt

# ============================================
# 5. LLM으로 답변 생성 (MLX 기준)
# ============================================
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

def rag_answer(query: str) -> str:
    # (1) 검색
    retrieved_docs, distances = retrieve(query, k=3)

    # (2) 프롬프트 조립
    prompt = build_prompt(query, retrieved_docs)

    # (3) 채팅 템플릿 적용
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # (4) 생성
    answer = generate(
        model, tokenizer,
        prompt=formatted_prompt,
        max_tokens=512
    )
    return answer

# ============================================
# 실행
# ============================================
question = "재택근무는 얼마나 자주 가능한가요?"
answer = rag_answer(question)
print(answer)
```

**흐름 매핑**

|단계|코드|
|---|---|
|질문 벡터화|`embedder.encode([query])`|
|top-k 검색|`collection.query(..., n_results=k)`|
|프롬프트 삽입|`build_prompt()`|
|LLM 답변 생성|`generate(model, tokenizer, prompt=...)`|

**일반 환경(transformers)으로 바꾸려면** 5번 블록만 아래로 교체하면 됩니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def rag_answer(query: str) -> str:
    retrieved_docs, distances = retrieve(query, k=3)
    prompt = build_prompt(query, retrieved_docs)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("mps")

    outputs = model.generate(inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return answer
```

나머지(임베딩, 벡터DB, 검색 로직)는 완전히 동일해요. 실제로 돌려보고 싶으시면 패키지 설치 명령어(`pip install chromadb sentence-transformers mlx-lm`)도 안내해드릴 수 있어요.