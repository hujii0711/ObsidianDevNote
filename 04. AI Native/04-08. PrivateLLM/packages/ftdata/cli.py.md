
```
from api.llm import MlxLLM  
from api.settings import Settings  
from rag.retriever import Retriever  
  
from .builder import split_train_valid, to_chat_example, write_jsonl  
from .filter import format_ok  
from .generate import generate_candidates  
from .questions import load_questions

def _arg(name: str, default):

def main() -> None:  
	settings = Settings.from_env()  
	retriever = Retriever(settings.rag)  
	llm = MlxLLM(settings.mlx_model)  
	questions = load_questions()  
	hits, cands = generate_candidates(q, retriever=retriever, llm=llm,               judge_fn=judge_fn, k=k, temperature=0.7)  
	kept = [c for c in cands if format_ok(c.answer, sources=c.sources)  
	and c.grounded >= min_ground]  
	for c in kept:  
	examples.append(to_chat_example(q, hits, c.answer))  
	train, valid = split_train_valid(examples, valid_every=10)  
	write_jsonl(_OUT_DIR / "train.jsonl", train)  
	write_jsonl(_OUT_DIR / "valid.jsonl", valid)
```