
```
from eval.answer_metrics import answer_metrics

def format_ok(answer: str, *, sources: list) -> bool:  
	m = answer_metrics(answer, sources=sources, must_mention=[])
```