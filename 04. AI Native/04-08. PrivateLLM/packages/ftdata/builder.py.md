
```
from rag.prompt import build_messages  
from rag.types import Retrieved

def to_chat_example(question: str, hits: list[Retrieved], answer: str) -> dict:  
	messages = build_messages(question, hits)
	
def split_train_valid(examples: list[dict], valid_every: int = 10):

def write_jsonl(path: Path, rows: list[dict]) -> None:
```