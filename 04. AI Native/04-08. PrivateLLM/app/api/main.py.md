
```
from .pipeline import run_chat  
from .schemas import ChatRequest  
from .settings import Settings

def create_app(  
		retriever=None, 
		llm=None,
		settings: Settings | None = None, 
) -> FastAPI:
	@app.get("/health")  
	def health():
	
	@app.post("/chat")  
	def chat(req: ChatRequest):
		retr = app.state.retriever or _build_retriever(app.state.settings)  
		model = app.state.llm or _build_llm(app.state.settings)  
		def event_gen():  
			for ev in run_chat(  
				req.message,  
				retriever=retr,  
				llm=model,  
				max_tokens=cfg.max_tokens,  
				temperature=cfg.temperature,  
			):

def _build_retriever(settings: Settings):  
	from rag.retriever import Retriever # 지연 import (무거운 패키지이므로)  
	if not hasattr(_build_retriever, "_cached"):  
		_build_retriever._cached = Retriever(settings.rag)  
	return _build_retriever._cached

def _build_llm(settings: Settings):  
	from .llm import MlxLLM
	if not hasattr(_build_llm, "_cached"):  
		_build_llm._cached = MlxLLM(settings.mlx_model)  
	return _build_llm._cached
```