
```python
# schemas.py — API 요청/응답 데이터 형식(Schema) 정의 모듈
# Pydantic 라이브러리를 사용해 데이터 모델을 선언합니다.

# 【Pydantic 이란?】
# 파이썬 타입 힌트를 기반으로 데이터 검증을 수행하는 라이브러리입니다.
# 잘못된 데이터가 들어오면 ValidationError 를 발생시켜 버그를 조기에 잡아줍니다.

# BaseModel : Pydantic 모델의 부모 클래스. 상속받으면 자동 검증 기능이 활성화됩니다.
# Field : 필드에 추가 제약 조건(최소 길이, 최대 길이, 설명 등)을 붙일 때 사용합니다.
from pydantic import BaseModel, Field

# ChatRequest — 채팅 엔드포인트(/chat)에 전달되는 요청 바디 (JSON 형식)
class ChatRequest(BaseModel):
	message: str = Field(min_length=1)

# SourceOut — /chat 응답에 포함되는 출처(인용 문헌) 형식
# LLM 답변에서 인용된 법령·판례 등의 출처 정보.
class SourceOut(BaseModel):
	n: int # 인용 번호 (답변 본문의 [1], [2] 와 대응)
	title: str # 법령명, 판례명 등 출처의 제목
	ref: str # 본문에서 사용한 참조 표기 (예: "[1]")
	url: str # 원문 링크 (법령 정보 사이트, 판례 DB 등)
	source_type: str # 출처 유형 (예: "law" = 법령, "case" = 판례)
```
