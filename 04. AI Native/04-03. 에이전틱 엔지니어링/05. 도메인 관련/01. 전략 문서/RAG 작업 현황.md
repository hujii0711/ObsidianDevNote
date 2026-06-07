### 1. Ollama를 터미널에서 한 줄로 자동 설치
```zsh
curl -fsSL https://ollama.com/install.sh | sh
```
이 명령어를 치면 맥북 내부에서 다음과 같은 작업이 자동으로 진행됩니다.

- Ollama 최신 버전을 다운로드합니다.
- 실행 파일을 적절한 위치(`/usr/local/bin` 등)에 배치합니다.
- 필요한 권한 설정을 마칩니다.
- (이미 설치되어 있다면) 최신 버전으로 업데이트합니다.
- http://localhost:11434

> ollama 앱 실행 = ollama serve

### 2. llama 3.1:8b 다운로드

`ollama pull llama3.1:8b`

- 모델을 **다운로드만** 함
- 대화창 안 열림, 그냥 종료됨
- 나중에 쓰려고 미리 받아둘 때 사용

`ollama run llama3.1:8b`

- 모델이 없으면 **자동으로 pull 먼저** 하고
- 다운로드 완료 후 **바로 대화 모드로 진입**
- 터미널에서 `>>>` 프롬프트가 뜨고 채팅 가능

`ollama run`은 `pull + 실행`을 한 번에 하는 것이라 보면 됩니다.

처음 쓸 때는 그냥 `ollama run llama3.1:8b` 하나면 충분하고, `pull`은 "지금 당장 쓰진 않지만 미리 받아두고 싶을 때" 쓰는 명령어입니다.

### 3.  소스 예제

```zsh
uv add torchvision torchaudio
uv add langchain langchain-huggingface langchain-ollama langchain-community langchain-chroma
uv add sentence-transformers
```

```python
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# 1. MPS 장치 확인 (Apple Silicon GPU 가속)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"현재 사용 중인 장치: {device}")

# 2. Embedding 모델 로드 (BGE-M3 - 다국어 성능 최강)
# M4의 성능을 활용하기 위해 성능이 높은 bge-m3 모델을 사용합니다.
model_name = "BAAI/bge-m3"
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
model_name=model_name,
model_kwargs={'device': device}, # MPS 가속 적용
encode_kwargs=encode_kwargs
)

# 3. Ollama - Llama 3 로드
# 48GB 메모리라면 llama3:8b는 매우 빠르고, llama3:70b도 시도해볼 만합니다.
# 처음에는 8b로 테스트해보세요.
llm = OllamaLLM(model="llama3.1:8b")

# 4. 벡터 저장소 설정 (테스트용 데이터)
data = [
"애플 M4 칩은 강력한 뉴럴 엔진을 탑재하여 AI 연산에 최적화되어 있습니다.",
"RAG 시스템은 로컬 LLM의 정보 부족 문제를 외부 문서 검색으로 해결합니다.",
"맥북 프로 M4 모델의 통합 메모리는 GPU와 CPU가 공유하여 데이터 전송 효율이 높습니다."
]
vectorstore = Chroma.from_texts(texts=data, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. 프롬프트 및 체인 구성
template = """가져온 정보를 바탕으로 질문에 답하세요. 모르는 내용은 지어내지 마세요.

문맥:
{context}
질문: {question}
답변:"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

# 6. 실행
question = "M4 맥북의 메모리 특징이 뭐야?"
print(f"\n질문: {question}")
print("-" * 30)
print(rag_chain.invoke(question))
```

### 4. 로컬 다운로드 경로

| 구성                     | 저장 위치                   |
| ---------------------- | ----------------------- |
| HuggingFace 모델 (소스 예제) | `~/.cache/huggingface/` |
| Torch 캐시 (소스 예제)       | `~/.cache/torch/`       |
| Chroma                 | (기본) 메모리 only           |
| Chroma (persist 시)     | 지정한 디렉토리                |
| Ollama 모델              | `~/.ollama/models`      |
| Ollama 앱               | /usr/local/bin/ollama   |

### 5. 허깅페이스 인증 방법
#### (1) os.environ["HF_TOKEN"] = "..."`

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```
##### 1) 특징
- **현재 프로세스(파이썬 실행 중)에서만 유효**
- 메모리에만 존재 (디스크 저장 ❌)
- 프로그램 종료하면 사라짐
##### 2) 동작 방식
- 내부적으로 라이브러리(`transformers`, `huggingface_hub`)가  
  → `HF_TOKEN` 환경변수를 읽어서 인증
##### 3) 장점
- 간단함
- 테스트/임시 실행에 적합
- 보안 측면에서 디스크에 안 남음
##### 4) 단점
- 매번 실행할 때 다시 설정해야 함
- 다른 툴(예: CLI)에서는 자동으로 안 씀

##### (2) from huggingface_hub import login

```python
from huggingface_hub import login
login(token="hf_xxx")
```

##### 1) 특징
- **토큰을 로컬에 저장 (영구 사용)**
- 여러 실행/프로그램에서 자동으로 사용됨
##### 2) 저장 위치 (macOS)

```bash
~/.cache/huggingface/token
```

또는

```bash
~/.huggingface/token
```

##### 3) 장점
- 한 번 로그인하면 계속 사용 가능
- CLI, Python, 다양한 라이브러리에서 공통 사용

##### 4) 단점
- 로컬에 토큰이 저장됨 (보안 관리 필요)
- 공유 환경에서는 주의

> 핵심 차이 한눈에

| 항목     | `os.environ` | `login()` |
| ------ | ------------ | --------- |
| 저장 위치  | 메모리          | 디스크       |
| 지속성    | 실행 중만        | 영구        |
| 범위     | 현재 프로세스      | 전체 환경     |
| 사용 편의성 | 매번 설정        | 1회 로그인    |
| 보안     | 상대적으로 안전     | 관리 필요     |

> 언제 뭐 쓰는 게 좋냐
###### `os.environ`
- CI/CD
- 테스트 코드
- 일회성 실행
- 보안 민감 환경
###### `login()`
- 로컬 개발 환경
- 반복 실행
- 개인 머신

> 실전 팁

둘을 같이 쓸 수도 있음:

```python
import os
from huggingface_hub import login

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
```

👉 이렇게 하면
- 환경변수 있으면 그걸 쓰고
- 없으면 기존 로그인 정보 사용 가능

> 한 줄 정리

👉 `os.environ` = “임시 인증”  
👉 `login()` = “로컬에 저장되는 영구 로그인”