 목차)
1. MCP 개요 및 설치
2. 기본 MCP 서버 생성 (FastMCP)
3. MCPAdapter 설정
4. Agent와 MCP 통합
5. ToolNode와 MCP 통합
6. 외부 MCP 서버에서 3rd Party 도구 사용하기

학습 목표)
- MCP의 개념과 아키텍처를 이해합니다
- `MCPAdapter`를 사용하여 단일/다중 MCP 서버에 연결하고 도구를 로드하는 방법을 학습합니다
- `create_agent` 및 `ToolNode`와 MCP를 통합하는 방법을 익힙니다
- 실전 예제를 통해 복잡한 에이전트를 구축합니다

LangGraph와 MCP(Model Context Protocol)를 통합하여 강력한 AI 에이전트를 구축하는 방법을 배웁니다. MCP는 AI 애플리케이션에서 도구(Tool)와 컨텍스트를 표준화된 방식으로 제공하는 오픈 프로토콜입니다. MCP를 활용하면 다양한 외부 서비스와 데이터를 일관된 인터페이스로 LLM에 연결할 수 있습니다.
LangChain 1.4부터 MCP 지원이 `langchain.mcp` 네임스페이스로 LangChain 본체에 내장되었습니다. 기존의 별도 패키지였던 `langchain-mcp-adapters`(`MultiServerMCPClient`)는 단일 `MCPAdapter` 클래스로 대체되었으며, 내부적으로 FastMCP 클라이언트를 사용합니다.

##### 1. MCP 기본 개념
MCP는 애플리케이션이 언어 모델에 도구와 컨텍스트를 제공하는 방법을 표준화한 오픈 프로토콜입니다. 이 프로토콜을 사용하면 다양한 서비스와 도구를 일관된 방식으로 LLM에 연결할 수 있습니다. 기존에는 각 도구마다 개별적인 연동 방식이 필요했지만, MCP를 통해 하나의 표준 인터페이스로 통합할 수 있게 되었습니다.

##### 2. 주요 특징
- **표준화된 도구 인터페이스**: 일관된 방식으로 도구를 정의하고 사용할 수 있습니다
- **다양한 전송 메커니즘**: stdio, Streamable HTTP 등 여러 통신 방식을 지원합니다
- **동적 도구 검색**: 런타임에 도구를 자동으로 검색하고 로드할 수 있습니다
- **확장 가능한 아키텍처**: 여러 서버를 동시에 연결하여 사용할 수 있습니다
##### 3. 설치
MCP를 사용하기 위해 필요한 패키지를 설치합니다. `langchain[mcp]` extra 를 설치하면 `langchain.mcp` 네임스페이스와 FastMCP 클라이언트가 함께 설치됩니다. (이 프로젝트의 `pyproject.toml` 에 이미 포함되어 있으므로 `uv sync` 만으로 준비됩니다.)
	`langchain[mcp]` 
	MCP 서버의 도구를 LangChain 에이전트에서 쓰려고 할 때 설치합니다. `[mcp]`는 extra 옵션으로,
	기본 `langchain`에 MCP 연동에 필요한 의존성(FastMCP 등)을 함께 설치합니다. 릴리스 노트에는
	`fastmcp>=4.0.0`이 필요하다고 적혀 있습니다.
	
```bash
uv add "langchain[mcp]"
```

> 참고: `langchain.mcp` 네임스페이스는 `langchain[mcp]>=1.4.0` 이 필요하며 현재 **beta** 상태입니다. import 시 프로세스당 1회 `LangChainBetaWarning` 이 출력되며, API 는 변경될 수 있습니다.

##### 4. 기본 MCP 서버 생성
MCP 서버는 도구를 제공하는 독립적인 프로세스입니다. FastMCP를 사용하면 Python으로 간단하게 MCP 서버를 만들 수 있습니다. MCP 서버는 클라이언트의 요청을 받아 도구를 실행하고 결과를 반환하는 역할을 수행합니다. 이 튜토리얼에서는 미리 준비된 MCP 서버들을 사용합니다.

제공되는 MCP 서버)
이 튜토리얼에서 사용하는 MCP 서버 파일들은 `server/` 디렉토리에 위치해 있습니다:

| 파일명                    | 설명                        | 전송 방식                     |
| ---------------------- | ------------------------- | ------------------------- |
| `mcp_server_local.py`  | 날씨 정보를 제공하는 로컬 서버         | stdio                     |
| `mcp_server_remote.py` | 현재 시간을 제공하는 원격 서버         | Streamable HTTP (포트 8002) |
| `mcp_server_rag.py`    | PDF 문서 검색 기능을 제공하는 RAG 서버 | stdio                     |

각 서버는 FastMCP 4.x 의 `FastMCP` 클래스를 사용하여 구현되어 있으며, `@mcp.tool` 데코레이터로 도구(Tool)를 정의하고 클라이언트 요청에 응답합니다.
```python
from fastmcp import FastMCP
mcp = FastMCP("Weather", instructions="날씨 정보를 제공하는 어시스턴트입니다.")

@mcp.tool
async def get_weather(location: str) -> str:
	"""지정된 위치의 현재 날씨 정보를 가져옵니다."""
	return f"It's always Sunny in {location}"

if __name__ == "__main__":
	mcp.run(transport="stdio") # HTTP 서버: mcp.run(transport="http", port=8002)
```

##### 5. MCPAdapter 설정
`MCPAdapter`는 MCP 서버에 연결하여 서버가 제공하는 도구들을 LangChain 도구로 변환해 주는 **비동기 컨텍스트 매니저**입니다. 전달하는 대상(target)의 형태에 따라 전송 방식이 자동으로 결정됩니다.

| target                                   | 전송 방식              |
| ---------------------------------------- | ------------------ |
| `Path("server.py")` (스크립트 경로)            | stdio — 서브프로세스로 실행 |
| `"http://.../mcp"` (URL 문자열)             | Streamable HTTP    |
| `FastMCP` 인스턴스                           | 인메모리 (테스트에 적합)     |
| `{"mcpServers": {...}}` (MCPConfig 딕셔너리) | 여러 서버를 하나의 어댑터로    |

지원하는 전송 방식)
- **stdio**: 클라이언트가 서버를 서브프로세스로 실행하고 표준 입출력을 통해 통신합니다. 로컬 개발에 적합합니다.
- **Streamable HTTP**: 서버가 독립적인 프로세스로 실행되어 HTTP 요청을 처리합니다. 원격 연결에 적합합니다.

연결 생명주기)
`async with MCPAdapter(target) as adapter:` 블록 안에서 `list_tools()` 로 도구를 검색합니다. 반환된 도구들은 클라이언트를 내부에 보관하므로, 컨텍스트가 종료된 뒤에도 에이전트에서 계속 호출할 수 있습니다 (도구 호출 시마다 세션을 열고 닫습니다).

아래 코드는 MCP 서버에 연결하여 도구를 가져오는 헬퍼 함수를 정의합니다.
```python
# MCPAdapter으로 대체된 MultiServerMCPClient
async def setup_mcp_client(server_config: dict):
	# MCP 클라이언트 생성
	client = MultiServerMCPClient(server_config)
	
	# 서버에 연결하여 도구 목록을 가져오기
	tools = await client.get_tools()
	
	# 로드된 도구 목록을 출력
	for tool in tools:
		print(f" - {tool.name}")
	return client, tools
```

```python
async def load_mcp_tools(target):
	"""MCPAdapter 로 MCP 서버에 연결하고 도구를 가져옵니다.
	Args:
		target: 스크립트 경로(Path), HTTP URL(str), FastMCP 인스턴스,
		또는 {"mcpServers": {...}} 형태의 MCPConfig 딕셔너리
	Returns:
		list: 로드된 LangChain 도구 목록
	"""

	# 어댑터 컨텍스트 안에서 도구를 검색합니다
	async with MCPAdapter(target) as adapter:
		tools = await adapter.list_tools()
		
	# 로드된 도구 목록을 출력합니다
	print(f"[MCP] {len(tools)}개의 도구가 로드되었습니다:")
		
	for tool in tools:
		print(f" - {tool.name}")
	return tools
```

##### 6. stdio 전송 방식 사용
stdio 전송 방식은 MCP 클라이언트가 서버를 서브프로세스로 직접 실행하여 표준 입출력(stdin/stdout)을 통해 통신하는 방식입니다. 별도의 서버 실행 없이 클라이언트가 자동으로 프로세스를 관리하므로 로컬 개발 환경에서 가장 편리하게 사용할 수 있습니다.
`MCPAdapter` 에 `Path` 객체를 전달하면 해당 스크립트를 stdio 서버로 실행합니다. (문자열 경로는 URL 로만 해석되므로 반드시 `Path` 를 사용해야 합니다.)
아래 코드는 날씨 MCP 서버를 stdio 방식으로 연결하고 도구를 로드합니다.
```python
# 날씨 서버를 stdio 방식으로 실행 (Path 객체 전달 --> stdio 서버로 실행)
weather_server = Path("server/mcp_server_local.py")

# MCP 도구 로드
tools = await load_mcp_tools(weather_server)
# [MCP] 1개의 도구가 로드되었습니다.
# - get_weather
```

아래 코드는 MCP 도구를 사용하는 에이전트를 생성합니다. `create_agent`는 LangChain v1에서 제공하는 에이전트 생성 함수로, LLM과 도구 목록을 전달하면 추론-행동 루프를 자동으로 구현합니다.
```python
# LLM 설정
# 다른 모델을 사용하려면 gpt-5.4-mini, claude-sonnet-5 등으로 변경 가능
llm = init_chat_model("gpt-5.4", temperature=0)

# 에이전트 생성: MCP 도구를 사용하는 에이전트
agent = create_agent(
	llm,
	tools,
	checkpointer=InMemorySaver(), # 대화 상태를 메모리에 저장
)
```

아래 코드는 생성된 에이전트를 사용하여 날씨 정보를 요청합니다. `astream_graph` 함수를 사용하면 에이전트의 실행 과정을 스트리밍으로 확인할 수 있습니다.
```python
# 스트리밍 헬퍼 함수와 UUID 생성 함수를 import합니다
from langchain_teddynote.messages import astream_graph, random_uuid
from langchain_core.runnables import RunnableConfig

# 대화 스레드 ID를 설정합니다
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 에이전트 실행: 날씨 정보 요청
response = await astream_graph(
	agent,
	inputs={"messages": [("human", "안녕하세요. 서울의 날씨를 알려주세요.")]},
	config=config,
)
```

##### 7. HTTP 전송 방식 사용
원격 서버나 HTTP 엔드포인트를 사용하는 경우 Streamable HTTP 전송 방식을 사용합니다. 이 방식은 서버가 별도의 프로세스로 실행 중이어야 합니다. stdio 방식과 달리 클라이언트가 서버를 직접 관리하지 않으므로, 사전에 서버가 실행 상태여야 연결이 가능합니다.

**사전 준비**: 아래 코드를 실행하기 전에 별도의 터미널에서 Remote MCP 서버를 먼저 구동해야 합니다.
```bash
uv run python server/mcp_server_remote.py
```

> 8002 포트가 이미 사용 중이라면 `fastmcp run server/mcp_server_remote.py:mcp --transport http --port 8012` 처럼 다른 포트로 실행하고 아래 URL 의 포트를 맞춰 주세요.

`MCPAdapter` 에 `http://` 또는 `https://` URL 문자열을 전달하면 Streamable HTTP 로 연결합니다.
아래 코드는 HTTP 기반 MCP 서버에 연결하는 예제입니다.
```python
# HTTP 기반 MCP 서버 URL (Streamable HTTP 엔드포인트)
time_server_url = "http://127.0.0.1:8002/mcp"

# HTTP 서버 도구 로드
http_tools = await load_mcp_tools(time_server_url)
```

아래 코드는 HTTP 전송 방식으로 연결된 MCP 도구를 사용하는 에이전트를 생성합니다.
```python
# LLM 설정
# 다른 모델을 사용하려면 gpt-5.4-mini, claude-sonnet-5 등으로 변경 가능
llm = init_chat_model("gpt-5.4", temperature=0)

# HTTP 도구를 사용하는 에이전트 생성
agent = create_agent(
	llm,
	http_tools,
	checkpointer=InMemorySaver(),
)
```

아래 코드는 HTTP 기반 MCP 에이전트를 실행하여 현재 시간을 요청합니다.
```python
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 에이전트 실행: 현재 시간 요청
response = await astream_graph(
	agent,
	inputs={"messages": [("human", "안녕하세요. 현재 시간을 알려주세요.")]},
	config=config,
)
```

##### 8. MCP Inspector 사용
MCP Inspector는 MCP 서버를 테스트하고 디버깅할 수 있는 웹 기반 도구입니다. 브라우저에서 서버의 도구 목록을 확인하고, 직접 도구를 호출하여 결과를 확인할 수 있습니다. 개발 과정에서 MCP 서버가 올바르게 동작하는지 빠르게 검증할 때 매우 유용합니다.

다음 명령어를 터미널에서 실행하면 MCP Inspector가 시작됩니다:
```bash
npx @modelcontextprotocol/inspector
```

##### 9. RAG MCP 서버 사용
RAG(Retrieval-Augmented Generation, 검색 증강 생성)는 외부 문서에서 관련 정보를 검색하여 LLM의 응답을 보강하는 기법입니다. MCP 서버를 통해 RAG 기능을 제공하면, 에이전트가 PDF 문서 등의 외부 데이터에서 필요한 정보를 검색하여 보다 정확한 답변을 생성할 수 있습니다.

아래 코드는 RAG 기능을 제공하는 MCP 서버에 연결하고 도구를 로드합니다.
```python
# RAG(검색 증강 생성) MCP 서버 (stdio)
# PDF 문서에서 정보를 검색하는 기능을 제공합니다
rag_server = Path("server/mcp_server_rag.py")

# RAG 도구 로드
rag_tools = await load_mcp_tools(rag_server)
```

아래 코드는 RAG 도구를 사용하는 에이전트를 생성합니다.
```python
# LLM 설정
# 다른 모델을 사용하려면 gpt-5.4-mini, claude-sonnet-5 등으로 변경 가능
llm = init_chat_model("gpt-5.4", temperature=0)

# RAG 도구를 사용하는 에이전트 생성
rag_agent = create_agent(
	llm,
	rag_tools,
	checkpointer=InMemorySaver(),
)
```

아래 코드는 RAG 에이전트를 실행하여 PDF 문서에서 삼성전자의 생성형 AI 관련 정보를 검색합니다.
```python
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# RAG 에이전트 실행: PDF 문서에서 정보 검색
_ = await astream_graph(
	rag_agent,
	inputs={
		"messages": [
			(
				"human",
				"삼성전자가 개발한 생성형 AI 의 이름은? mcp 서버를 사용해서 검색해주세요.",
			)
		]
	},
	config=config,
)
```

##### 10. Agent와 MCP 통합 (다중 서버)
React Agent는 추론(Reason)과 행동(Act)을 반복하는 ReAct 패턴을 구현합니다. LLM이 상황을 분석하고, 필요한 도구를 선택하여 호출하고, 결과를 바탕으로 다음 행동을 결정하는 과정을 자동으로 수행합니다. 이 패턴은 복잡한 작업을 여러 단계로 나누어 처리할 때 특히 효과적입니다.
MCP 도구와 함께 사용하면 다양한 외부 서비스에 접근할 수 있는 강력한 에이전트를 만들 수 있습니다. 여러 MCP 서버의 도구를 하나의 에이전트에 통합하면 복합적인 작업도 단일 에이전트로 처리할 수 있습니다.

다중 서버 구성 (MCPConfig))
여러 서버를 하나의 어댑터로 묶으려면 표준 `MCPConfig` 형식(`{"mcpServers": {...}}`)의 딕셔너리를 전달합니다. 각 항목의 전송 방식은 키로부터 자동 추론됩니다.
- `command` / `args` → stdio (서브프로세스 실행)
- `url` → Streamable HTTP

다중 서버 구성에서는 도구 이름 충돌을 막기 위해 **각 도구 이름 앞에 서버 이름이 접두사로 붙습니다** (예: `weather_get_weather`, `current_time_get_current_time`).

아래 코드는 MCP 도구를 사용하는 에이전트를 생성하는 헬퍼 함수를 정의합니다.
```python
async def create_mcp_agent(target):
	"""MCP 도구를 사용하는 에이전트를 생성합니다.
	이 함수는 주어진 target(경로, URL, MCPConfig 등)으로 MCP 서버에 연결하고,
	해당 도구들을 사용하는 에이전트를 반환합니다.
	
	Args:
		target: MCPAdapter 에 전달할 대상
		
	Returns:
		CompiledStateGraph: 컴파일된 에이전트
	"""
	
	# MCP 도구 가져오기
	tools = await load_mcp_tools(target)
	
	# LLM 설정
	# 다른 모델을 사용하려면 gpt-5.4-mini, claude-sonnet-5 등으로 변경 가능
	llm = init_chat_model("gpt-5.4", temperature=0)
	
	# 에이전트 생성
	agent = create_agent(
		llm,
		tools,
		checkpointer=InMemorySaver(),
	)
	return agent
```

아래 코드는 날씨(stdio)와 시간(HTTP) 두 개의 MCP 서버를 동시에 연결하여 다중 서버 에이전트를 생성합니다. 이렇게 하면 하나의 에이전트가 여러 서버의 도구를 모두 사용할 수 있게 됩니다.
```python
# 다중 MCP 서버 구성 (MCPConfig 형식): 날씨(stdio) + 시간(HTTP)
server_config = {
	"mcpServers": {
		"weather": {
			"command": "uv", # uv 패키지 매니저로 서버 스크립트 실행 (stdio)
			"args": ["run", "python", "server/mcp_server_local.py"],
		},
		"current_time": {
			"url": "http://127.0.0.1:8002/mcp", # Streamable HTTP 엔드포인트
		},
	}
}

# 다중 MCP 서버를 사용하는 에이전트 생성
agent = await create_mcp_agent(server_config)
```

아래 코드는 동일한 대화 스레드에서 연속으로 두 가지 질문을 실행합니다. 같은 `thread_id`를 사용하면 대화 컨텍스트가 유지되어 이전 대화 내용을 참조할 수 있습니다. 이를 통해 에이전트는 대화의 흐름을 이해하고 연속적인 질문에 자연스럽게 응답합니다.
```python
# 대화 스레드 설정 (상태 유지를 위해 동일한 thread_id 사용)
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 첫 번째 질문: 현재 시간
await astream_graph(
	agent,
	inputs={"messages": [("human", "현재 시간을 알려주세요")]},
	config=config,
)

# 두 번째 질문: 날씨 (같은 대화 스레드에서 연속 질문)
await astream_graph(
	agent,
	inputs={"messages": [("human", "현재 서울의 날씨도 알려주세요")]},
	config=config,
)
# 같은 대화 스레드를 사용했기 때문에 에이전트가 이전 대화 맥락을 유지하며, 두새의 서로 다른 MCP 서버의 도구를 상황에 맞게 사용합니다.
```

##### 11. ToolNode와 MCP 통합
`ToolNode`를 사용하면 LangGraph에서 더 세밀한 제어가 가능한 커스텀 워크플로우를 만들 수 있습니다. React Agent와 달리, 그래프의 각 노드를 직접 정의하고 연결할 수 있어 복잡한 로직을 구현하기에 적합합니다. 에이전트-도구 루프의 각 단계를 명시적으로 제어할 수 있다는 점이 가장 큰 장점입니다.

ToolNode의 특징)
- **세밀한 제어**: 각 노드의 동작을 직접 정의할 수 있습니다
- **유연한 워크플로우**: 조건부 분기, 병렬 처리 등 복잡한 흐름을 구현할 수 있습니다
- **확장성**: 추가 도구(예: Tavily 검색)를 쉽게 통합할 수 있습니다

아래 코드는 MCP 도구와 Tavily 검색 도구를 결합한 커스텀 워크플로우를 생성하는 함수를 정의합니다.
```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Dict, Any, TypedDict
from langchain_tavily import TavilySearch

class AgentState(TypedDict):
"""에이전트 상태 정의
Attributes:
	messages: 대화 메시지 목록. add_messages 리듀서로 메시지가 누적됩니다.
	context: 추가 컨텍스트 정보를 저장하는 딕셔너리 (선택적)
"""
	messages: Annotated[List[BaseMessage], add_messages]
	context: Dict[str, Any]

async def create_mcp_workflow(target):
"""MCP 도구를 사용하는 커스텀 워크플로우를 생성합니다.
이 함수는 MCP 도구와 Tavily 검색 도구를 결합하여
에이전트-도구 루프를 구현하는 그래프를 생성합니다.

Args:
	target: MCPAdapter 에 전달할 대상 (경로, URL, MCPConfig 등)

Returns:
	CompiledStateGraph: 컴파일된 워크플로우 그래프
"""
	# MCP 도구 로드
	tools = await load_mcp_tools(target)
	
	# Tavily 웹 검색 도구 추가
	tavily_tool = TavilySearch(max_results=2)
	tools.append(tavily_tool)
	
	# LLM 설정 및 도구 바인딩
	# 다른 모델을 사용하려면 gpt-5.4-mini, claude-sonnet-5 등으로 변경 가능
	llm = init_chat_model("gpt-5.4", temperature=0)
	llm_with_tools = llm.bind_tools(tools)
	
	# 워크플로우 그래프 생성
	workflow = StateGraph(AgentState)

	async def agent_node(state: AgentState):
		"""에이전트 노드: LLM을 호출하여 응답을 생성합니다"""
		response = await llm_with_tools.ainvoke(state["messages"])
		return {"messages": [response]}

	# ToolNode 생성: 도구 호출을 처리합니다
	tool_node = ToolNode(tools)
	
	# 그래프에 노드 추가
	workflow.add_node("agent", agent_node)
	workflow.add_node("tools", tool_node)
	
	# 엣지 정의: 시작 -> 에이전트
	workflow.add_edge(START, "agent")
	
	# 조건부 엣지: 에이전트 -> (도구 or 종료)
	# tools_condition은 도구 호출이 필요하면 "tools"로, 아니면 END로 라우팅합니다
	workflow.add_conditional_edges("agent", tools_condition)
	
	# 엣지 정의: 도구 -> 에이전트 (루프)
	workflow.add_edge("tools", "agent")
	
	# 체크포인터와 함께 그래프 컴파일
	return workflow.compile(checkpointer=InMemorySaver())
```

아래 코드는 날씨 서버와 시간 서버를 사용하는 MCP 워크플로우를 생성합니다. 이어서 컴파일된 그래프 구조를 시각화하여 확인합니다.
```python
# MCP 서버 구성 정의 (MCPConfig 형식)
server_config = {
	"mcpServers": {
		"weather": {
			"command": "uv",
			"args": ["run", "python", "server/mcp_server_local.py"],
		},
		"current_time": {
			"url": "http://127.0.0.1:8002/mcp",
		},
	}
}

# MCP 워크플로우 생성
mcp_app = await create_mcp_workflow(server_config)
```

아래 코드는 생성된 MCP 워크플로우를 실행하여 현재 시간을 조회합니다.
```python
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# MCP 워크플로우 실행: 현재 시간 조회
_ = await astream_graph(
	mcp_app,
	inputs={"messages": [("human", "현재 시간을 알려주세요")]},
	config=config,
)
```

아래 코드는 MCP 도구(시간 조회)와 Tavily 도구(뉴스 검색)를 조합한 복합 작업을 실행합니다. 에이전트가 시간을 먼저 조회한 후 해당 날짜의 뉴스를 검색하는 과정을 자동으로 수행합니다.

```python
# 복합 작업: 시간 조회 후 뉴스 검색 (Tavily 도구 사용)
_ = await astream_graph(
	mcp_app,
	inputs={
		"messages": [
			("human", "오늘 뉴스를 검색해주세요. 검색시 시간을 조회한 뒤 처리하세요.")
		]
	},
	config=config,
)
```

##### 12. 외부 MCP 서버에서 3rd Party 도구 사용하기

Context7 MCP 서버)
Context7은 최신 프로그래밍 언어 및 프레임워크 문서를 검색하고 제공하는 MCP 서버입니다. LangGraph, React, Python 등의 최신 공식 문서를 실시간으로 검색하여 최신 정보 기반의 코드 생성에 활용할 수 있습니다.
`npx`를 통해 직접 실행할 수 있으며, stdio 전송 방식으로 클라이언트와 통신합니다. Smithery AI와 같은 MCP 서버 레지스트리에서 다양한 3rd Party MCP 서버를 검색하여 동일한 방식으로 사용할 수 있습니다.

아래 코드는 Context7 MCP 서버를 포함한 다중 서버 구성을 설정하고 워크플로우를 생성합니다.
```python
# 다중 MCP 서버 구성 (로컬 + HTTP + Context7)
server_config = {
	"mcpServers": {
		# 로컬 날씨 서버 (stdio)
		"weather": {
			"command": "uv",
			"args": ["run", "python", "server/mcp_server_local.py"],
		},
		
		# 원격 시간 서버 (Streamable HTTP)
		"current_time": {
			"url": "http://127.0.0.1:8002/mcp",
		},
	
		# Context7 MCP 서버: 최신 문서 검색 (npx 로 실행, stdio)
		"context7": {
			"command": "npx",
			"args": ["-y", "@upstash/context7-mcp@latest"],
		},
	}
}

# 다중 서버를 사용하는 MCP 워크플로우 생성
mcp_app = await create_mcp_workflow(server_config)
```

아래 코드는 Context7 서버를 활용하여 최신 LangGraph 문서에서 ReAct Agent 관련 내용을 검색하고, 검색된 정보를 바탕으로 Tavily 검색을 수행하는 ReAct Agent 코드를 생성하는 복합 작업을 실행합니다.
```python
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# Context7 서버를 활용한 복합 작업:
# 1. 최신 LangGraph 문서에서 ReAct Agent 관련 내용 검색
# 2. 검색된 정보를 바탕으로 코드 생성
await astream_graph(
	mcp_app,
	inputs={
		"messages": [
			(
				"human",
				"최신 LangGraph 도큐먼트에서 ReAct Agent 관련 내용을 검색하세요. 그런 다음 Tavily 검색을 수행하는 ReAct Agent를 생성하세요.",
			)
		]
	},
	config=config,
)
```

##### 13. 정리
이번 튜토리얼에서는 LangChain 1.4 에 내장된 `langchain.mcp` 네임스페이스를 사용하여 MCP 서버의 도구를 LangGraph 에이전트에 연결하는 방법을 살펴보았습니다.

**핵심 내용:**
- MCP 서버는 FastMCP 4.x 의 `FastMCP` 클래스와 `@mcp.tool` 데코레이터로 간단히 작성할 수 있습니다.
- `MCPAdapter` 는 target 의 형태(`Path`, URL, `FastMCP` 인스턴스, `MCPConfig`)로 전송 방식을 자동 추론합니다.
- `async with MCPAdapter(target) as adapter:` 안에서 `list_tools()` 로 도구를 검색하며, 반환된 도구는 컨텍스트 종료 후에도 사용할 수 있습니다.
- 다중 서버 구성(`MCPConfig`)에서는 도구 이름에 `{서버명}_` 접두사가 자동으로 붙습니다.
- 로드된 MCP 도구는 `create_agent` 와 `ToolNode` 모두에서 일반 LangChain 도구와 동일하게 사용할 수 있습니다.
- MCP 도구의 오류(`isError=True`)는 `status="error"` 인 `ToolMessage` 로 모델에 전달되며, 전송 계층의 실패는 예외로 발생합니다.

