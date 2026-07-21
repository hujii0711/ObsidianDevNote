
```
from rag.citations import extract_sources, strip_invalid_citations  
from rag.prompt import build_messages

def _ensure_disclaimer(answer: str) -> str:

def run_chat(  
	query: str,
	*,
	retriever,
	llm,
	max_tokens: int = 768,  
	temperature: float = 0.3,  
) -> Iterator[dict]:  
	messages = build_messages(query, hits)  
	answer = _ensure_disclaimer(strip_invalid_citations(raw, hits))  
	sources = [  
		{  
			"n": s.n, 
			"title": s.title,
			"ref": s.ref,
			"url": s.url,
			"source_type": s.source_type,
		}  
		for s in extract_sources(answer, hits)  
	]
```