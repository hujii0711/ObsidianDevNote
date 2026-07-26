
```python
# config.py — RAG 시스템 전역 설정 모듈

# 【이 파일의 역할】
# 벡터 DB(ChromaDB) 접속 정보, 임베딩 모델 이름,
# 검색 파라미터(top_k, min_similarity) 등을 한 곳에서 관리합니다.

# 【중요】 색인(indexing) 단계에서 사용한 설정값과 반드시 일치해야 합니다.
# 색인할 때 사용한 모델과 다른 모델로 검색하면 벡터 공간이 달라져 검색 결과가 무의미해집니다.

import os # 환경 변수 읽기
from dataclasses import dataclass # 데이터 전용 클래스 선언 도구
from pathlib import Path # 운영체제에 독립적인 파일 경로 처리

# ──────────────────────────────────────────────────────────
# 상수(Constants) — 색인 파이프라인(Plan 1)과 반드시 동일해야 함
# ──────────────────────────────────────────────────────────
# Chroma 컬렉션(테이블) 이름.
# 색인 시 이 이름으로 문서를 저장했으므로, 검색 시에도 동일한 이름을 사용해야 합니다.
COLLECTION = "jeonse_deposit"

# 임베딩 모델 이름 (Hugging Face 식별자).
# BAAI/bge-m3 는 한국어·영어 등 다국어를 지원하는 1024차원 임베딩 모델입니다.
# 색인 시 사용한 모델과 다르면 검색이 정상 동작하지 않습니다.
MODEL_NAME = "BAAI/bge-m3"

# ──────────────────────────────────────────────────────────
# 경로 자동 계산
# ──────────────────────────────────────────────────────────
# Path(__file__) : 현재 파일(config.py)의 절대 경로
# .resolve() : 심볼릭 링크를 해소한 실제 절대 경로
# .parents[4] : 상위 4번째 디렉터리
_REPO_ROOT = Path(__file__).resolve().parents[4]

# ChromaDB가 벡터를 디스크에 저장하는 기본 경로
# 레포 루트 기준: data/chroma/
_DEFAULT_CHROMA = _REPO_ROOT / "data" / "chroma"

# RAG 파이프라인 동작을 제어하는 설정값 묶음.
@dataclass
class RagConfig:
	chroma_dir: Path = _DEFAULT_CHROMA # 벡터 DB 저장 경로 (기본값: data/chroma)
	collection: str = COLLECTION # 검색 대상 컬렉션 이름
	model_name: str = MODEL_NAME # 임베딩 모델 이름
	
	# 유사도 검색 결과에서 반환할 최대 문서 수
	# top_k=6 이면 가장 유사한 6개 문서를 가져와 프롬프트에 포함합니다.
	top_k: int = 6
	
	# 코사인 유사도(Cosine Similarity) 하한값
	# 코사인 유사도 범위: -1.0(완전 반대) ~ 0.0(무관) ~ 1.0(완전 동일)
	# 0.35 미만이면 질문과 관련성이 낮다고 판단해 답변을 거부합니다.
	min_similarity: float = 0.35

	# 환경 변수에서 설정을 읽어 RagConfig 인스턴스를 생성합니다.
	@classmethod
	def from_env(cls) -> "RagConfig"
		chroma = os.environ.get("CHROMA_DIR")
		return cls(
			# 환경 변수가 있으면 Path 객체로 변환, 없으면 기본 경로 사용
			chroma_dir=Path(chroma) if chroma else _DEFAULT_CHROMA
		)
```
