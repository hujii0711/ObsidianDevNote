LangGraph 필수 개념 목록입니다.

**핵심 구성 요소**
1. **State**: 그래프 전체가 공유하는 데이터 구조 (TypedDict, Pydantic 등으로 정의)
2. **Node**: State를 입력받아 갱신분을 반환하는 함수 (LLM 호출, 도구 실행 등 실제 작업 단위)
3. **Edge**: 노드 간 실행 흐름 연결
   - 일반 Edge: 고정된 다음 노드
   - Conditional Edge: State에 따라 다음 노드를 동적으로 결정
4. **START / END**: 그래프의 시작점과 종료점을 나타내는 특수 노드
5. **Graph / StateGraph**: 노드와 엣지를 조립하는 빌더. `compile()`로 실행 가능한 객체 생성

**상태 관리**
6. **Reducer**: State 필드를 덮어쓸지 누적할지 정하는 규칙 (예: `add_messages`, `operator.add`)
7. **Messages State**: 대화형 에이전트용 기본 State (`MessagesState`)
8. **Checkpointer**: 각 단계의 State를 저장 (MemorySaver, SQLite, Postgres 등)
9. **Thread**: `thread_id`로 구분되는 하나의 대화/실행 세션
10. **Store**: 스레드를 넘어 유지되는 장기 메모리

**제어 흐름**
11. **Command**: 노드에서 State 갱신과 이동(`goto`)을 동시에 지정
12. **Send**: 동적 병렬 실행 (map-reduce 패턴)
13. **Subgraph**: 그래프 안에 중첩된 그래프
14. **Recursion Limit**: 무한 루프 방지용 최대 실행 단계 수

**Human-in-the-loop / 운영**
15. **Interrupt**: 실행을 일시 중단하고 사람의 입력을 기다림 (`interrupt()`, 재개는 `Command(resume=...)`)
16. **Time Travel**: 체크포인트로 되돌아가 재실행하거나 State 수정
17. **Streaming**: 단계별 업데이트, 토큰, 이벤트 스트리밍 (`stream_mode`: values, updates, messages 등)
18. **Persistence / Durable Execution**: 실패 후 중단 지점부터 재개

**에이전트 관련**
19. **Tool Calling**: 모델이 도구를 호출하고 `ToolNode`가 실행
20. **ReAct 에이전트**: 추론 → 도구 호출 → 관찰 반복 패턴 (`create_react_agent`)
21. **멀티 에이전트 패턴**: Supervisor, Swarm, Hierarchical 등

**학습 순서 추천**
State → Node → Edge → Conditional Edge → Reducer → Checkpointer → Interrupt → Subgraph → 멀티 에이전트

원하시면 최소 예제 코드나 각 개념의 상세 설명을 이어서 정리해 드릴게요. 참고로 API 명칭(`create_react_agent` 등)은 버전에 따라 바뀔 수 있으니 공식 문서 확인을 권장합니다.