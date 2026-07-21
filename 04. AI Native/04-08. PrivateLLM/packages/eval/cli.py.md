
```
from api.llm import MlxLLM  
from api.settings import Settings  
from rag.retriever import Retriever  
from .dataset import load_eval_set  
from .report import aggregate  
from .runner import run_item

def main() -> None:  
	settings = Settings.from_env()
	retriever = Retriever(settings.rag)
	gen_llm = MlxLLM(settings.mlx_model, adapter_path=adapter)  
	judge_llm = gen_llm if adapter is None else MlxLLM(settings.mlx_model)  
	items = load_eval_set(_EVAL_SET)
	res = run_item(item, retriever=retriever, llm=gen_llm, judge_fn=judge_fn,        top_k=settings.rag.top_k)  
	agg = aggregate(results)
```
