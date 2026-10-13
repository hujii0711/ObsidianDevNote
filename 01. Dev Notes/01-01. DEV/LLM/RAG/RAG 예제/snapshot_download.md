
`snapshot_download`는 Hugging Face Hub에서 모델 저장소 전체를 내 컴퓨터로 내려받고, **저장된 로컬 폴더 경로를 문자열로 돌려주는** 함수입니다.

### 동작 순서

```python
local_path = snapshot_download(repo_id="mlx-community/Qwen3.6-35B-A3B-4bit")
```

1. Hub에서 해당 저장소(`repo_id`)의 최신 커밋 정보를 확인합니다.
2. 저장소의 모든 파일을 받습니다. 가중치(`*.safetensors`), `config.json`, `tokenizer.json`, `tokenizer_config.json`(chat template 포함) 등이 여기에 들어갑니다.
3. 받은 파일을 HF 캐시 폴더에 저장합니다.
   ```
   ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<커밋해시>/
   ```
   Windows라면 `C:\Users\<사용자>\.cache\huggingface\hub\...` 아래입니다.
4. 이 `snapshots/<커밋해시>/` 경로를 반환합니다. 이 값이 `local_path`에 들어갑니다.

### 특징

- **이미 받았으면 다시 받지 않습니다.** 캐시에 같은 버전이 있으면 다운로드를 건너뛰고 경로만 돌려줍니다. 그래서 두 번째 실행부터는 빠릅니다. 버전 확인 요청도 보내지 않으려면 `HF_HUB_OFFLINE=1`을 설정하면 됩니다.
- **중간에 끊겨도 이어받습니다.** 받다가 중단되면 다음 실행 때 남은 부분부터 다시 받습니다.
- **파일 위쪽의 `HF_HUB_ENABLE_HF_TRANSFER=1` 설정과 한 세트입니다.** 이 설정이 있으면 `hf_transfer` 패키지(Rust로 만든 병렬 다운로더)를 써서 큰 모델을 빠르게 받습니다. `hf_transfer`가 설치되어 있지 않으면 에러가 날 수 있습니다.

### 다음 줄과의 관계

```python
local_path = snapshot_download(repo_id=MODEL_ID)   # 1) 디스크에 준비 (다운로드/캐시 확인)
model, tokenizer = load(local_path)                # 2) 디스크 → 메모리로 로드
```

사실 `mlx_lm.load(MODEL_ID)`처럼 repo ID를 바로 넘겨도, 내부에서 같은 방식으로 다운로드한 뒤 로드합니다. 그래서 이 줄이 꼭 필요한 것은 아닙니다. 다운로드 단계와 로드 단계를 나눠서 "모델 다운로드/캐시 확인"과 "모델 로드 중" 진행 상황을 따로 보여주려고 분리해 둔 것입니다.

### 자주 쓰는 옵션

```python
snapshot_download(
    repo_id=MODEL_ID,
    revision="main",                                  # 특정 브랜치/커밋 고정
    allow_patterns=["*.json", "*.safetensors"],       # 필요한 파일만 받기
    local_dir="./models/qwen",                        # 캐시 대신 지정 폴더에 저장
)
```

---

이 코드에서는 꼭 필요하지 않습니다. `load(MODEL_ID)` 한 줄로 충분합니다. 다만 쓰면 좋은 상황이 따로 있습니다.

### 이 스크립트에서 필요 없는 이유

```python
# 지금
local_path = snapshot_download(repo_id=MODEL_ID)
model, tokenizer = load(local_path)

# 이것만으로 같은 동작
model, tokenizer = load(MODEL_ID)
```

- `mlx_lm.load`와 transformers의 `from_pretrained`는 repo ID를 받으면 내부에서 다운로드와 캐시 확인을 알아서 합니다.
- 내부 다운로드는 **로딩에 필요한 파일만** 골라서 받습니다(`*.json`, `*.safetensors`, 토크나이저 파일 등). 반면 `snapshot_download`를 `allow_patterns` 없이 쓰면 저장소의 모든 파일을 받습니다. 저장소에 README 이미지, `.gguf`, 원본 체크포인트 같은 파일이 같이 올라와 있으면 필요 없는 용량까지 내려받게 됩니다. `mlx-community` 저장소는 대부분 필요한 파일만 있어서 실제 차이는 작습니다.

### 쓰는 게 좋은 경우

| 상황 | 이유 |
|---|---|
| 다운로드와 실행을 분리할 때 | Docker 빌드나 서버 배포 전에 모델을 미리 받아 두고, 실행할 때는 `HF_HUB_OFFLINE=1`로 네트워크 없이 로드 |
| 모델을 특정 폴더에 두고 싶을 때 | `local_dir="./models/qwen"`으로 캐시 대신 프로젝트 폴더에 저장 |
| 일부 파일만 받고 싶을 때 | `allow_patterns`로 원하는 파일만 받음 |
| 로컬 경로가 따로 필요할 때 | `mlx_lm.server --model <경로>` 같은 다른 도구에 경로를 넘기거나, 여러 프로그램이 같은 모델 파일을 함께 쓸 때 |
| 진행 상황을 단계별로 보여주고 싶을 때 | 지금 코드처럼 "다운로드"와 "로드"를 나눠서 출력 |

### 권장

- **학습용, 단일 스크립트(지금 RAGTest MVP):** `load(MODEL_ID)`만 쓰세요. 코드가 짧아지고 불필요한 파일도 받지 않습니다.
- **배포용, 오프라인 환경, 여러 도구가 모델을 공유할 때:** `snapshot_download`에 `allow_patterns`나 `local_dir`를 지정해서 쓰세요.

원하시면 `main.py`와 `main03.py`를 `load(MODEL_ID)` 방식으로 정리해 드릴게요.