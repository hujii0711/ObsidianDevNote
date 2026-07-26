
```python
from collections.abc import Iterator # 제너레이터·이터레이터의 타입 힌트에 사용
from typing import Protocol # 인터페이스(추상 계약)를 정의하는 도구
from .settings import MLX_MODEL # 기본 모델 이름 상수 import (상대 경로 import)

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
		from mlx_lm import stream_generate # 스트리밍 생성 함수
		from mlx_lm.sample_utils import make_sampler
		
		# apply_chat_template : 메시지 목록을 모델이 이해하는 형식으로 변환합니다.
		# add_generation_prompt=True : 모델이 답변을 시작하도록 특수 토큰을 추가합니다.
		# tokenize=False : 토큰 ID 배열이 아닌 문자열로 반환합니다.
		prompt = self._tokenizer.apply_chat_template(  
			messages, add_generation_prompt=True, tokenize=False  
		)
		# make_sampler(temp=temperature): 온도(temperature)에 따른 샘플링 전략 객체 생성
		# temperature=0.0 → 항상 확률이 가장 높은 토큰 선택 (greedy decoding)
		# temperature=1.0 → 확률 분포에서 무작위 샘플링
		sampler = make_sampler(temp=temperature)
		
		# stream_generate : 프롬프트에 이어지는 텍스트를 한 토큰씩 생성합니다.
		# 각 반복마다 resp 객체가 반환되며, resp.text 에 생성된 텍스트 조각이 담겨 있습니다.
		for resp in stream_generate(  
			self._model,  
			self._tokenizer,  
			prompt,  
			max_tokens=max_tokens,  
			sampler=sampler,  
		):
```
