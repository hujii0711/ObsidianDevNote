
```python
# embedder.py — 텍스트를 벡터(숫자 배열)로 변환하는 임베더 모듈

from collections.abc import Callable # 함수 타입 힌트에 사용 (Callable)
from .config import MODEL_NAME # 기본 임베딩 모델 이름 상수


class QueryEmbedder:

	def __init__(
		self,
		# Callable[[list[str]], list] : 문자열 리스트를 받아 리스트를 반환하는 함수 타입
		# | None : None 도 허용 (기본값이 None 이므로)
		encode_fn: Callable[[list[str]], list] | None = None,
		model_name: str = MODEL_NAME,
	):
		self._encode_fn = encode_fn # 텍스트 → 벡터 변환 함수. None 이면 SentenceTransformer 사용.
		self._model_name = model_name # 사용할 SentenceTransformer 모델 이름 (Hugging Face 경로)
		self._model = None # 모델은 아직 로드하지 않음 (지연 로딩)
	
	# SentenceTransformer 모델을 처음 필요한 시점에 로드하고 인코딩합니다.
	def _lazy(self, texts: list[str]) -> list:
		if self._model is None:
			# 모델이 아직 로드되지 않았으면 지금 로드합니다.
			import torch # GPU/MPS 장치 감지에 사용
			from sentence_transformers import SentenceTransformer
			
			# MPS 를 지원하지 않는 환경(Intel Mac, Linux)에서는 CPU 로 폴백합니다.
			device = "mps" if torch.backends.mps.is_available() else "cpu"
			
			# 모델을 지정한 장치에 로드합니다.
			self._model = SentenceTransformer(self._model_name, device=device)
			
			# normalize_embeddings=True : 벡터의 크기(norm)를 1로 정규화합니다.
			# 정규화된 벡터끼리의 내적(dot product)이 곧 코사인 유사도가 됩니다.
			# ChromaDB 가 코사인 거리를 사용하므로 반드시 정규화해야 합니다.
			vecs = self._model.encode(texts, normalize_embeddings=True)
			
			# numpy 배열을 파이썬 기본 list 로 변환합니다.
			# ChromaDB 는 파이썬 list 형식을 요구합니다.
			return [v.tolist() for v in vecs]
			
	# 단일 질문 문자열을 벡터로 변환합니다.
	# 내부적으로 encode_fn 또는 _lazy 를 호출합니다.
	def embed_query(self, query: str) -> list:

	"""
	Args:
		query: 임베딩할 질문 텍스트 (예: "보증금 반환 소송 절차가 어떻게 되나요?")
	
	Returns:
		질문을 나타내는 1024차원 float 벡터 (list[float])
	"""

		# 주입된 함수가 있으면 그것을, 없으면 _lazy(실제 모델)를 사용합니다.
		# `or` 연산자: 왼쪽이 falsy(None, 0, "" 등)이면 오른쪽을 사용합니다.
		fn = self._encode_fn or self._lazy
		
		# fn([query]) : 리스트로 감싸서 호출 (배치 처리 API)
		# [0] : 결과 리스트의 첫 번째(유일한) 벡터를 꺼냄
		return fn([query])[0]
```
