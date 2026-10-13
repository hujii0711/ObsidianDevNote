
Apple Silicon Mac에서 MLX 모델을 LangChain에 연결하는 가장 안정적인 방법은 **`mlx_lm.server`로 모델을 OpenAI 호환 서버로 띄우고, LangChain에서는 `ChatOpenAI`의 `base_url`만 로컬로 바꾸는 것**입니다. 체인 구조는 앞의 코드와 같습니다.

**1) 설치 및 서버 실행 (터미널)**

```bash
pip install -U mlx-lm langchain-core langchain-openai

# 처음 실행할 때 Hugging Face에서 모델을 내려받습니다 (약 20GB)
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
```

**2) LangChain 코드 (다른 터미널 또는 노트북에서 실행)**

```python
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# 1) 프롬프트
prompt = PromptTemplate.from_template(
    """
    너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수}가지 추천하고,
    그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아.
    <재료>
    {재료}
    """
)

# 2) 로컬 MLX 서버에 연결 (API 키는 아무 문자열이나 넣으면 됨)
llm = ChatOpenAI(
    model="mlx-community/Qwen3.6-35B-A3B-4bit",
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
)

# 3) Qwen 계열의 <think>...</think> 추론 부분 제거
def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# 4) 체인 연결
chain = prompt | llm | StrOutputParser() | RunnableLambda(strip_think)

# 5) 실행
result = chain.invoke({
    "개수": 3,
    "재료": "계란, 양파, 대파, 밥, 간장",
})
print(result)
```

**스트리밍으로 보고 싶다면** `strip_think`를 빼고 스트리밍하면 됩니다. 이 경우 추론 과정도 함께 출력됩니다.

```python
stream_chain = prompt | llm | StrOutputParser()
for chunk in stream_chain.stream({"개수": 2, "재료": "두부, 김치, 돼지고기"}):
    print(chunk, end="", flush=True)
```

**서버 없이 파이썬 안에서 직접 로드하는 방법**도 있습니다. `langchain_community`의 `ChatMLX`를 사용합니다.

```python
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX

pipe = MLXPipeline.from_model_id(
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    pipeline_kwargs={"max_tokens": 2048},
)
llm = ChatMLX(llm=pipe)

chain = prompt | llm | StrOutputParser() | RunnableLambda(strip_think)
```

다만 이 방법은 `mlx-lm` 버전이 바뀔 때 인자(`temp` 등)가 맞지 않아 오류가 나는 경우가 자주 있어서, 서버 방식을 더 권장합니다.

참고할 점이 몇 가지 있습니다.

- 35B MoE 4bit 모델은 가중치만 약 20GB이므로, 통합 메모리가 **32GB 이상**인 Mac이 사실상 필요합니다. 활성 파라미터는 약 3B라서 속도는 빠른 편입니다.
- 추론(think) 과정 없이 바로 답을 받고 싶다면, 프롬프트 끝에 `/no_think`를 붙이는 방법이 Qwen3 계열에서 지원되었습니다. 3.6에서도 동작하는지는 모델 카드에서 확인해 보세요.
- 서버 포트를 바꿨다면 `base_url`의 포트도 함께 바꿔야 합니다.