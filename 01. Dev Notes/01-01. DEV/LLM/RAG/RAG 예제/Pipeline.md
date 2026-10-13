
transformers의 `pipeline()`은 **"원시 입력 → 사람이 읽을 결과"까지의 추론 과정 전체를 한 번의 호출로 묶은 래퍼**입니다. 내부는 크게 **전처리 → 모델 추론 → 후처리** 세 단계로 이루어집니다.

### 구성 요소
`pipeline("text-generation", model=MODEL_ID)` 한 줄이 내부적으로 하는 일은 다음과 같습니다.

| 단계          | 하는 일                                                                                                                                | 직접 짜면 필요한 코드                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **로딩**      | 모델 설정(config), 가중치, 토크나이저를 Hub에서 받아 task에 맞는 모델 클래스로 로드                                                                             | `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`      |
| **디바이스 배치** | GPU/CPU 선택, dtype(bf16 등) 지정                                                                                                        | `model.to("cuda")`, `torch_dtype=...`                                        |
| **전처리**     | 텍스트를 토큰 ID 텐서로 변환. 메시지 리스트가 들어오면 chat template 적용, 패딩/잘라내기, attention mask 생성                                                       | `tokenizer.apply_chat_template(...)`, `tokenizer(text, return_tensors="pt")` |
| **추론**      | text-generation이면 `model.generate()`로 토큰을 하나씩 생성(샘플링, temperature, top_p, max_new_tokens 적용). 분류 task면 `model(**inputs)`로 logits 계산 | `model.generate(**inputs, ...)`                                              |
| **후처리**     | 토큰 ID를 텍스트로 디코딩, 특수 토큰 제거, 입력 프롬프트 부분 잘라내기(`return_full_text=False`)                                                                | `tokenizer.decode(out[0][len(input_ids):], skip_special_tokens=True)`        |
| **배치 처리**   | 입력 여러 개를 묶어 처리                                                                                                                      | DataLoader, 패딩 정렬                                                            |

### 직접 구현과 비교
```python
# pipeline 없이
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.8)
answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# pipeline 사용
pipe = pipeline("text-generation", model=MODEL_ID, torch_dtype=torch.bfloat16, device=0)
answer = pipe(messages, max_new_tokens=1024, return_full_text=False)[0]["generated_text"]
```

### task별 차이
pipeline은 task 이름에 따라 전처리/후처리 로직이 달라집니다.
- `text-generation`: 생성 후 텍스트로 디코딩
- `text-classification`: logits에 softmax를 적용해 `{label, score}` 반환
- `feature-extraction`: 임베딩 벡터 반환(RAG 임베딩용)
- `automatic-speech-recognition`: 오디오 전처리 후 텍스트 반환

### LangChain과의 관계
- `HuggingFacePipeline`은 이 pipeline 객체를 LangChain의 LLM 인터페이스(`invoke`, `stream` 등)로 감싼 것입니다.
- `ChatHuggingFace`는 그 위에서 LangChain 메시지를 chat template으로 바꿔 줍니다.

main.py에서 `apply_chat_template` → `generate` 순서로 직접 짠 코드가 바로 pipeline의 **전처리 + 추론** 부분을 MLX로 손수 구현한 것입니다. `mlx_lm.generate`가 토큰화와 디코딩까지 처리해 주기 때문에 후처리 코드는 따로 없습니다.


---

## MLX에서 HuggingFacePipeline 사용 못하는 이유


### 안 되는 이유
`HuggingFacePipeline`은 내부적으로 **transformers의 `pipeline()`(PyTorch)** 을 만듭니다. 반면 `mlx-community/Qwen3.6-35B-A3B-4bit`는 **MLX 전용 형식**으로 양자화된 가중치라서 transformers가 읽지 못합니다. 그래서 main05.py도 MLX 변환본이 아니라 원본 `Qwen/Qwen3.6-35B-A3B`를 받도록 되어 있습니다.

Mac에서 main05.py를 그대로 실행하면 PyTorch 경로로 돌아갑니다. 지금 코드는 CUDA가 없으면 CPU + float32를 쓰기 때문에 35B 모델이면 메모리만 약 140GB가 필요합니다. 사실상 실행이 불가능하고, MLX를 쓸 이유도 사라집니다.

### MLX를 LangChain에 붙이는 방법
**1. `langchain_community`의 `MLXPipeline` + `ChatMLX`**
`HuggingFacePipeline` + `ChatHuggingFace`와 같은 구조의 MLX 버전입니다.
```python
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX

pipe = MLXPipeline.from_model_id(
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    pipeline_kwargs={"max_tokens": 1024, "temp": 0.7, "top_p": 0.8},
)
llm = ChatMLX(llm=pipe)
chain = prompt | llm | StrOutputParser()
```
주의할 점이 두 가지 있습니다.
- `ChatMLX`도 `apply_chat_template`에 `enable_thinking=False`를 넘기지 않습니다. 그래서 main05.py처럼 `strip_think`로 `<think>` 블록을 지워야 합니다.
- community 래퍼는 `mlx_lm`의 API 변경(예: `temp` 인자 대신 `sampler`를 넘기는 방식)을 늦게 따라가는 경우가 있습니다. 설치된 버전에서 직접 실행해 확인해 보세요. (제가 아직 이 프로젝트에서 실행해 보지는 않았습니다.)

**2. `mlx_lm.server` + `ChatOpenAI`**
```bash
uv run mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
```
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="none", model="default")
```
MLX 모델을 OpenAI 호환 서버로 띄우는 방식입니다. 장점은 다음과 같습니다.
- 모델 로드를 한 번만 하면 됩니다.
- 스트리밍과 tool calling 같은 기능을 LangChain 표준 방식으로 쓸 수 있습니다.
- 백엔드를 바꿔도 체인 코드는 그대로 둘 수 있습니다.

**3. 지금처럼 `RunnableLambda`로 직접 감싸기 (main.py)**
코드가 가장 짧고 `enable_thinking=False`를 그대로 제어할 수 있습니다. 학습용이라면 지금 방식도 충분히 합리적입니다.

**추천:** 학습/MVP 단계라면 지금의 main.py 방식을 유지하세요. RAG 단계에서 스트리밍이나 여러 스크립트 간 모델 공유가 필요해지면 2번(`mlx_lm.server` + `ChatOpenAI`)으로 넘어가는 것이 좋습니다.