
```
from .config import RagConfig  
from .embedder import QueryEmbedder  
from .types import Retrieved

class Retriever:  
	def __init__(self, config: RagConfig, encode_fn=None):  
			self._embedder = QueryEmbedder(  
			encode_fn=encode_fn,  
			model_name=config.model_name,  
		)  
		client = chromadb.PersistentClient(path=str(config.chroma_dir))  
		self._col = client.get_collection(config.collection)
	
	def retrieve(self, query: str) -> list[Retrieved]:  
		qvec = self._embedder.embed_query(query)  
		res = self._col.query(  
		query_embeddings=[qvec],  
		n_results=self._cfg.top_k,  
		include=["documents", "metadatas", "distances"],  
	)  
	def is_grounded(self, hits: list[Retrieved]) -> bool:
```
