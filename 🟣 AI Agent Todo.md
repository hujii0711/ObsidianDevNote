
- ~~PIL과 moviepy와 역할 차이~~
- ~~외부 서비스 MCP 간단한 연동 데모~~
- ~~web -> cli -> JSON 줄 이벤트 -> SSE 데모~~
- ~~장기 메모리 저장 및 조회 방법~~
- ~~sqllite에 저장하는 데이터~~
- ~~메모리: InMemory, sqllite~~
- ~~테디노트 Langchain, LangGraph 튜토리얼~~
- 웹검색으로 토대로 LLM의 답변 보강
- 쿼리 Rewrite 모듈
- ~~테디노트 유투브 강의 랭그래프 소스 코드로 정리하여 edu에 주입 (11.16_1)~~
- ~~HITL(review)를 적극적으로 활용~~
- ~~노드 실행후 사용자 승인에 의해 노드 분기하는 로직~~
- 테디노트 GoogleNews
- 테디노트 밋업 PDF 자료

---
### 고도화
- 프로젝트 삭제 기능 --> 생성파일은 유지
- 프로젝트 DB 관리
- OAuth2 로그인 사용자 DB 관리
- Mock 임시 데이터 제거
- 각 step시 다른 step 비활성화
- 완료 영상 목록 따로 메뉴로
- 이미지 편집에서 이미지 바꾸기(LLM 재수정, pixabay)

---
### 추후
- LangGraph를 SDK 버전, SDK마저 없는 버전으로 구현해보기
- 시스템 프롬프트와 프롬트트를 모듈화
- LLM 메모리 절약 위한 설정 소스내 확인
- 동영상 생성 모델 연동
- Bash를 도구로 사용하는 에이전트
- 장기메모리(store)에 저장해야할 사항과 checkpointer(새대화)에 저장해야할 사항
- 현재 페이지 코드를 기반으로 코드작성 및 수정
- 모델이 모르는 문제 웹검색으로 답변받는 그래프
- Youtube Data API 활용
- 테디노트 유틸성 메서드 별도 정리
- ~~이벤트 스트리밍: astream_events, 도구 진행 상황 표시~~
- 이미지 모델도 언어모델처럼 서버로 초기화
- 쇼츠 생성 중 작업 진행율 보여주기
- sqllite --> mongoDB
- Tavliy extract를 WebBaseLoader로 대체 검증(p.287)
- 모델 토큰 사용량 통계
- 동영상 이미지 각각 메뉴 다르게
- 별도 프로젝트 폴더로 구성
- aws로 서비스화
- 유저가 토큰 사용량 확인하는 페이지
- 스트리밍중 토큰 사용량 확인
- 유투브 음악 다운로드 프로그램
- Next.js BFF 방식 제거
- cli 방식 제거하고 fastapi 방식 사용시 문제되는 점 파악(동시 접속 처리등)
---
### 문법

- GraphError 역할외 파이썬 예외클래스 정의하여 사용하는 방법
- ~~`def __call__(self, *args, **kwargs):`~~
	~~`raise GraphError(OFFLINE_MESSAGE)`~~
	~~`def __getattr__(self, name: str):`~~
- ~~json 읽기 쓰기 파일읽기 파일쓰기~~
- ~~uv run -m 과 uv run python의 차이~~
- 파이썬 예외 처리


---
### UI 편집
- 자막 대응하면 이미지 연결
- 자막 수정
- 그래프 내부 작업 현황 web에 출력
