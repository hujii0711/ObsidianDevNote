
```
from .settings import MLX_MODEL

class LLM(Protocol):  
	def stream(  
		self,  
		messages: list[dict],
		*,
		max_tokens: int = 768,
		temperature: float = 0.3,
	) -> Iterator[str]:

class FakeLLM:  
	def __init__(self, tokens: list[str]):
	
	def stream(  
		self,  
		messages: list[dict],  
		*,  
		max_tokens: int = 768,  
		temperature: float = 0.3,  
	) -> Iterator[str]:

class MlxLLM:  
	def __init__(  
		self,  
		model_name: str = MLX_MODEL,
		adapter_path: str | None = None,
	):
	from mlx_lm import load
	loaded = load(model_name, adapter_path=adapter_path)
	
	def stream(  
		self,  
		messages: list[dict],  
		*,  
		max_tokens: int = 768,  
		temperature: float = 0.3,  
	) -> Iterator[str]:
		from mlx_lm import stream_generate
		from mlx_lm.sample_utils import make_sampler
		prompt = self._tokenizer.apply_chat_template(  
		messages, add_generation_prompt=True, tokenize=False  
		)  
		sampler = make_sampler(temp=temperature)  
		for resp in stream_generate(  
			self._model,  
			self._tokenizer,  
			prompt,  
			max_tokens=max_tokens,  
			sampler=sampler,  
		):
```
