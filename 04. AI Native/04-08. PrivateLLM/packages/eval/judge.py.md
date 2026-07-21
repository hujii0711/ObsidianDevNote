
```
def build_judge_prompt(question: str, answer: str, contexts: list[str]) -> str:  

def groundedness_score(*,  
	question: str, 
	answer: str,
	contexts: list[str],
	judge_fn: Callable[[str], str], 
) -> float:
```
