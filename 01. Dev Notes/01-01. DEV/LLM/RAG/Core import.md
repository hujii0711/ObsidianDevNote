
```python
#config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
```

```python
# load
from langchain_community.document_loaders import (
	CSVLoader,
	DirectoryLoader,
	PyPDFLoader,
	TextLoader,
)
from langchain_core.documents import Document
```

```python
# chunking
from langchain_core.documents import Document
from langchain_text_splitters import (
	MarkdownHeaderTextSplitter,
	RecursiveCharacterTextSplitter,
	SentenceTransformersTokenTextSplitter,
)
RecursiveCharacterTextSplitter(
	chunk_size=CHUNK_SIZE,
	chunk_overlap=CHUNK_OVERLAP,
	separators=["\n\n", "\n", ". ", "다. ", " ", ""],
	length_function=len,
	add_start_index=True, # metadata["start_index"] 로 원문 위치가 남는다
).split_documents(docs)
```

```python
# embedding
from langchain_huggingface import HuggingFaceEmbeddings
vectors = embeddings.embed_documents(SAMPLES)
query_vector = embeddings.embed_query(SAMPLES[0])
```

```python
#index
from qdrant_client.models import Filter, FilterSelector
store.add_documents(chunks, ids=ids)
info = client.get_collection(COLLECTION)
```

```python
#retriever
from qdrant_client.models import FieldCondition, Filter, MatchValue
client.collection_exists(COLLECTION):
store.similarity_search(QUERY, k=3)
store.similarity_search_with_score(QUERY, k=3)
store.max_marginal_relevance_search(QUERY, k=3, fetch_k=8, lambda_mult=0.5)
retriever = store.as_retriever(search_kwargs={"k": 3})
retriever.invoke("판다스에서 결측치를 채우는 방법")
```

```python
#prompt, pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROMPT = ChatPromptTemplate.from_messages(
	[("system", SYSTEM), ("human", "문서:\n{context}\n\n질문: {question}")]
)

def get_llm() -> ChatHuggingFace:
	device = get_device()
	tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
	model = AutoModelForCausalLM.from_pretrained(
		LLM_MODEL,
		dtype=get_torch_dtype(),
	).to(device)
	
	text_generation = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		max_new_tokens=256,
		do_sample=False, # 근거 기반 답변이라 매번 같은 결과가 나오게 둔다
		return_full_text=False, # 프롬프트는 되돌려주지 않고 생성분만
	)
	
	# ChatHuggingFace 가 모델의 chat template 을 적용해 준다.
	return ChatHuggingFace(
		llm=HuggingFacePipeline(pipeline=text_generation, model_id=LLM_MODEL),
		tokenizer=tokenizer,
	)
	
def build_chain(retriever, llm):
	"""LCEL 체인. context 와 question 을 만들어 프롬프트에 넣고 LLM 에 넘긴다."""
	return (
		RunnableParallel(
		context=retriever | format_docs,
		question=RunnablePassthrough(),
		)
		| PROMPT
		| llm
		| StrOutputParser()
	)

retriever = get_vector_store(client).as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(question)
```
