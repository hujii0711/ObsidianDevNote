
- 웹검색으로 토대로 LLM의 답변 보강
- 쿼리 Rewrite 모듈
- 테디노트 유투브 강의 랭그래프 소스 코드 확인
- 테디노트 유튜브 강의 목차 확보
- 다시 짚고 넘어갈 개념 정리
- 개발 문서 요약 및 통합
- HITL(review)를 적극적으로 활용
- UI 기획 필요
- 외부 서비스 MCP 간단한 연동 데모
- 그래프 내부 작업 현황 web에 출력
- web -> cli -> SSE 데모
- 장기 메모리 저장 및 조회 방법
- JSON 줄 이벤트
- 노드 실행후 사용자 승인에 의해 노드 분기하는 로직
- 메모리: InMemory, sqllite

---
GraphError 역할
`def __call__(self, *args, **kwargs):`
`raise GraphError(OFFLINE_MESSAGE)`
`def __getattr__(self, name: str):`

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
- 