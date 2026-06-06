## macOS HuggingFace 다운로드 방법

맥(Mac)에서 허깅페이스(Hugging Face)의 모델이나 데이터셋을 설치(다운로드)하는 방법은 크게 두 가지로 나뉩니다. **① 파이썬 코드를 사용해 프로젝트 내에서 자동으로 불러오기**, **② CLI(터미널 명령어)를 사용해 로컬 컴퓨터에 미리 다운로드하기**입니다.

### 1. 사전 준비

맥의 터미널(Terminal)을 열고, 필요한 허깅페이스 라이브러리를 먼저 설치합니다. 가상환경(venv)을 먼저 활성화한 후 설치하는 것을 권장합니다.

```bash
pip3 install huggingface_hub transformers datasets
```

- huggingface_hub를 설치해야 huggingface-cli 명령어를 사용할 수 있다.
- transformers를 설치해야 파이썬 코드에서 자동 다운로드할 수 있다.

---

### 2. 모델 다운로드

#### 방법 1 — Python 코드

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-3.1-8B")
```

#### 방법 2 — transformers 자동 다운로드(파이썬 코드로 자동 다운로드 (추천))

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 다운로드할 모델 저장소 이름
model_id = "meta-llama/Llama-3.1-8B" 

# 토크나이저와 모델 다운로드 및 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

#### 방법 3 — CLI
코드 실행 없이 모델 파일 자체를 맥에 파일 형태로 받아두고 싶을 때 사용합니다. `huggingface-cli`를 이용하면 편리합니다.

```bash
huggingface-cli download meta-llama/Llama-3.1-8B

# huggingface-cli를 이용해 특정 경로(--local-dir)에 모델 다운로드
huggingface-cli download gpt2 --local-dir ./my_gpt2_model
```
대용량 모델(LLM 등)을 받을 때는 파일이 너무 크기 때문에 심볼릭 링크(기본값) 대신 실제 파일로 다운로드하려면 뒤에 `--local-dir-use-symlinks False` 옵션을 붙여주는 것이 편리합니다.

#### 💡맥에서 다운로드된 파일 위치

경로를 직접 지정하지 않고 파이썬 코드로 기본 다운로드를 진행했다면, 허깅페이스는 맥의 홈 디렉토리 아래 숨겨진 캐시 폴더에 파일들을 저장합니다.

- **기본 저장 경로:** `~/.cache/huggingface/`
    
- 터미널에서 이동해 확인해보고 싶다면:
```
cd ~/.cache/huggingface/hub/
ls
```

만약 저장공간 문제 등으로 캐시 경로를 외장 하드나 다른 곳으로 바꾸고 싶다면 맥의 환경 변수(`~/.zshrc`)에 아래 줄을 추가해 주시면 됩니다.

```
export HF_HOME="/Volumes/외장하드이름/hf_cache"
```

---

### 3. 데이터셋 다운로드

#### 방법 1 — Python 코드
파이썬 스크립트나 주피터 노트북을 열고 `load_dataset` 함수를 사용하면 원하는 데이터셋을 맥으로 자동 다운로드(캐싱)할 수 있습니다.

```python
from datasets import load_dataset

# 허깅페이스에서 원하는 데이터셋 이름 입력 (예: 'squad')
dataset = load_dataset("squad")
dataset = load_dataset("json", data_files="data.json")  # 로컬 파일
```

##### 💡 특정 경로에 로컬 저장하고 싶을 때 (오프라인 사용)

```python
# 로컬 폴더로 저장
dataset.save_to_disk("./my_local_dataset")

# 나중에 인터넷 연결 없이 로컬에서 불러올 때
from datasets import load_from_disk
dataset = load_from_disk("./my_local_dataset")

# 다운로드된 데이터셋 확인
print(dataset)
```

#### 방법 2 — CLI
```bash
huggingface-cli download datasets/squad --repo-type dataset
```

---

### 4. 비공개 모델 (Gated Model) 접근

```bash
# 토큰 로그인 (최초 1회)
huggingface-cli login
# HF_TOKEN 입력
```

```python
# 코드에서 직접 토큰 지정
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B",
    token="hf_xxxxxxxxxxxx"
)
```

> Llama, Gemma 등 Gated 모델은 HuggingFace 사이트에서 **사전 접근 신청 필수**

---

### 5. 특정 파일만 다운로드

```python
from huggingface_hub import hf_hub_download

# 단일 파일만
hf_hub_download(
    repo_id="mistralai/Mistral-7B-v0.1",
    filename="config.json"
)
```

---

### 6. 다운로드 경로 지정

```python
snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.1",
    local_dir="~/models/mistral-7b"   # 원하는 경로 지정
)
```

---

### 요약

| 목적       | 방법                                           |
| -------- | -------------------------------------------- |
| 모델 전체    | `snapshot_download()` 또는 `from_pretrained()` |
| 모델 단일 파일 | `hf_hub_download()`                          |
| 데이터셋     | `load_dataset()`                             |
| 비공개 모델   | `huggingface-cli login` 후 동일하게 사용            |
| 저장 경로 변경 | `local_dir` 파라미터 또는 `HF_HOME` 환경변수           |