
```python
from .pipeline import run_chat # RAG 파이프라인 오케스트레이터
from .schemas import ChatRequest # 요청 데이터 검증 모델
from .settings import Settings # 전역 설정

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
		# run_chat 의 이벤트 딕셔너리를 SSE 형식으로 변환하는 내부 제너레이터.
		def event_gen():  
			for ev in run_chat(  
				req.message,  
				retriever=retr,  
				llm=model,  
				max_tokens=cfg.max_tokens,  
				temperature=cfg.temperature,  
			):
# Retriever 인스턴스를 처음 한 번만 생성하고 캐시합니다.
# Retriever 는 벡터 DB 를 메모리에 로드하므로 생성 비용이 비쌉니다.
# 싱글턴으로 관리해 중복 생성을 방지합니다.
def _build_retriever(settings: Settings):  
	from rag.retriever import Retriever # 지연 import (무거운 패키지이므로)  
	if not hasattr(_build_retriever, "_cached"):  
		_build_retriever._cached = Retriever(settings.rag)  
	return _build_retriever._cached

# MlxLLM 인스턴스를 처음 한 번만 생성하고 캐시합니다.
# LLM 모델 로딩은 수 GB 의 파일을 메모리에 올리는 작업이므로 시간이 많이 걸립니다.
# 싱글턴으로 관리해 요청마다 재로딩하는 것을 방지합니다.
def _build_llm(settings: Settings):  
	from .llm import MlxLLM
	if not hasattr(_build_llm, "_cached"):  
		_build_llm._cached = MlxLLM(settings.mlx_model)  
	return _build_llm._cached
```