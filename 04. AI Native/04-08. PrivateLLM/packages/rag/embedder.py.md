
```
from .config import MODEL_NAME

class QueryEmbedder:  
	def __init__(  
		self,
		encode_fn: Callable[[list[str]], list] | None = None,  
		model_name: str = MODEL_NAME,  
	):
	
	def _lazy(self, texts: list[str]) -> list:
	
	def embed_query(self, query: str) -> list:
```
