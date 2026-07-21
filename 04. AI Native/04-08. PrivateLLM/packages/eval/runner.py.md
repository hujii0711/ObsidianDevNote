
```
from api.pipeline import run_chat  
from .answer_metrics import answer_metrics  
from .dataset import EvalItem  
from .judge import groundedness_score  
from .retrieval_metrics import ref_hit

@dataclass  
class ItemResult:

def run_item(  
item: EvalItem,
*, 
retriever, 
llm,
judge_fn: Callable[[str], str],
top_k: int = 6,
) -> ItemResult:  
	run_chat  
	answer_metrics  
	ref_hit  
	groundedness_score
```