**핵심 원칙**: 그래프 정의(State·Node·Edge 조립)와 노드 내부 로직(프롬프트·도구·모델)을 분리하고, 그래프 객체를 한 곳에서 export 합니다.

```
my-agent/
├─ src/
│  └─ agent/
│     ├─ graph.py            # StateGraph 조립 + compile (export: graph)
│     ├─ state.py            # State, Reducer 정의
│     ├─ configuration.py    # 모델명, 프롬프트 등 런타임 설정
│     ├─ nodes/              # 노드 함수 (파일당 1~2개)
│     │  ├─ classify.py
│     │  ├─ retrieve.py
│     │  └─ generate.py
│     ├─ routing.py          # Conditional Edge 라우팅 함수
│     ├─ tools/              # 도구 정의
│     ├─ prompts/            # 프롬프트 템플릿
│     ├─ subgraphs/          # 서브그래프 (각각 graph.py 보유)
│     │  └─ research/
│     │     ├─ graph.py
│     │     └─ state.py
│     └─ utils/              # 모델 로더, 공통 헬퍼
├─ tests/
│  ├─ unit/                  # 노드 단위 테스트
│  ├─ integration/           # 그래프 전체 흐름
│  └─ evals/                 # 품질 평가 데이터셋
├─ langgraph.json            # 배포/서버 설정 (그래프 진입점 지정)
├─ pyproject.toml
├─ .env.example
└─ Dockerfile
```

**운영 시 지키면 좋은 규칙**
- **`langgraph.json`**: `"graphs": {"agent": "./src/agent/graph.py:graph"}` 형태로 진입점 지정. LangGraph Server/Studio가 이걸 읽음
- **State는 한 곳에**: `state.py`에서 정의하고 노드는 import만. 서브그래프는 자체 State를 두고 부모와 매핑
- **노드는 순수하게**: State 입력 → 갱신분 반환. 외부 의존성(LLM, DB)은 설정/주입으로 받아 테스트하기 쉽게
- **라우팅 함수 분리**: 조건 분기 로직은 `routing.py`에 모아 단위 테스트 가능하게
- **프롬프트 분리**: 코드 안에 하드코딩하지 말고 `prompts/`로
- **Checkpointer는 환경별로**: 로컬은 MemorySaver, 운영은 Postgres. LangGraph Server로 배포하면 자동 제공되므로 `compile()`에서 직접 지정하지 않음
- **설정 외부화**: `Configuration` 클래스로 모델·온도 등을 `config["configurable"]`로 주입
- **관측성**: LangSmith 트레이싱 환경 변수 설정, 평가용 데이터셋은 `tests/evals/`에 버전 관리

**규모별 가이드**
- 소규모: `graph.py` + `state.py` + `nodes.py` + `tools.py` 단일 파일 구조로 충분
- 중대형: 위 구조처럼 `nodes/` 분리, 서브그래프 도입
- 멀티 에이전트: 에이전트별 서브그래프를 `subgraphs/` 또는 `agents/`로 나누고 supervisor 그래프에서 조합

**Next.js와 함께 쓰는 경우**
프론트(Next.js)와 에이전트(LangGraph)는 별도 저장소 또는 모노레포(`apps/web`, `apps/agent`)로 분리하고, Next.js는 LangGraph Server API(또는 SDK)를 호출하는 방식이 일반적입니다.

Python/TypeScript 중 어느 쪽인지, 단일 에이전트인지 멀티 에이전트인지 알려주시면 구조를 더 구체적으로 맞춰 드릴게요.