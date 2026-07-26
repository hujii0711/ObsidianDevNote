
```python
# types.py — RAG 파이프라인에서 사용하는 공유 데이터 타입 정의

# 【이 파일의 역할】
# 여러 모듈(retriever, citations, prompt 등)에서 공통으로 사용하는
# 데이터 구조(클래스)를 한 곳에 모아 정의합니다.

# 타입을 분리하면:
# - 순환 import(circular import) 문제를 방지할 수 있습니다.
# - 각 모듈이 "어떤 형태의 데이터를 받고 반환하는지" 명확해집니다.

from dataclasses import dataclass # 데이터클래스 데코레이터

# ChromaDB 에서 검색된 문서 1건의 데이터.
# retriever.retrieve(query) 가 반환하는 리스트의 각 원소입니다.
@dataclass
class Retrieved:
	id: str # ChromaDB 문서 고유 식별자
	text: str # 검색된 문서 본문(실제 법령·판례) (LLM에게 "근거"로 제공되는 내용, LLM 프롬프트에 포함됨))
	similarity: float # 코사인 유사도: 1.0 - ChromaDB distance (거리 → 유사도로 변환)
	source_type: str # 출처 유형 (예: "law" = 법령, "case" = 판례)
	title: str # 법조항명·판례명 등 출처 제목 (예: "주택임대차보호법 제3조의3")
	ref: str # 본문 인용에 쓰이는 짧은 참조 코드 (예: "제3조의3")
	url: str # 원문을 볼 수 있는 외부 링크 URL
	date: str # 법령 시행일 또는 판례 선고일 (문자열)
	
# LLM 답변에서 실제로 인용된 출처 정보.
# Retrieved 와의 차이:
# 	Retrieved : 검색 단계에서 가져온 "후보" 문서 (본문 텍스트 포함)
# 	Source : 최종 답변에서 실제로 인용된 출처만 추출한 것 (본문 제외)
# citations.extract_sources() 함수가 답변 텍스트의 [1], [2] 번호를
# 실제 Retrieved 와 매핑해 Source 목록을 만들어 줍니다.
@dataclass  
class Source:
	n: int # 답변 본문에 등장한 인용 번호 [n] — 답변 본문의 [1], [2] 와 1:1 대응
	title: str # 출처 제목
	ref: str # 짧은 참조 코드
	url: str # 원문 외부 링크
	source_type: str # 출처 유형
```
