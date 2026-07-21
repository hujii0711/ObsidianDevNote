
```
from api.pipeline import run_chat  
from eval.judge import groundedness_score  
from rag.types import Retrieved

@dataclass  
class Candidate:

def generate_candidates(question: str, *, retriever, llm, judge_fn: Callable[[str], str], k: int = 6, temperature: float = 0.7) -> tuple[list[Retrieved], list[Candidate]]:  
	for _ in range(k):  
	events = list(run_chat(question, retriever=retriever, llm=llm,  
	temperature=temperature))  
	grounded = groundedness_score(question=question, answer=done["answer"],  
        contexts=contexts, judge_fn=judge_fn)
```