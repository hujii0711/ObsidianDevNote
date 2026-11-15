
- ~~PIL과 moviepy와 역할 차이~~
- ~~외부 서비스 MCP 간단한 연동 데모~~
- ~~web -> cli -> JSON 줄 이벤트 -> SSE 데모~~
- ~~장기 메모리 저장 및 조회 방법~~
- ~~sqllite에 저장하는 데이터~~
- ~~메모리: InMemory, sqllite~~

- 웹검색으로 토대로 LLM의 답변 보강
- 쿼리 Rewrite 모듈
- 테디노트 유투브 강의 랭그래프 소스 코드로 정리하여 edu에 주입 (11.16_1)
- HITL(review)를 적극적으로 활용
- 노드 실행후 사용자 승인에 의해 노드 분기하는 로직

---

### 추후
- LangGraph를 SDK 버전, SDK마저 없는 버전으로 구현해보기
- 시스템 프롬프트와 프롬트트를 모듈화
- LLM 메모리 절약 위한 설정 소스내 확인
- 동영상 생성 모델 연동
- Bash를 도구로 사용하는 에이전트

---
### 문법

- GraphError 역할외 파이썬 예외클래스 정의하여 사용하는 방법
- ~~`def __call__(self, *args, **kwargs):`~~
	~~`raise GraphError(OFFLINE_MESSAGE)`~~
	~~`def __getattr__(self, name: str):`~~
- ~~json 읽기 쓰기 파일읽기 파일쓰기~~
- ~~uv run -m 과 uv run python의 차이~~


---

### 데모
- checkpointer
- store
- 


---

### 역할별 에이전트
> 순차 방식
##### 1. 웹 검색 에이전트(research)
##### 2. 대본 생성 에이전트(script)

##### 3. 장면 분할 에이전트(storyboard)

##### 4. 장면 생성 에이전트(media)

- HITL
- Memory


---

### UI 편집
- 자막 대응하면 이미지 연결
- 자막 수정
- 그래프 내부 작업 현황 web에 출력