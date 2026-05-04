# Fine-tuning Lab (MLX on Apple Silicon)

학습 목적으로 LLM 파인튜닝을 실습하기 위한 파이썬 개발 환경입니다. 이 문서는 이 저장소에서 작업하는 Claude Code에게 프로젝트의 전제·규약·명령을 알려주기 위한 지침서입니다.

---
## 1. 프로젝트 개요

- **목적**: 한국어 공개 데이터셋(KoAlpaca 등)으로 Instruction/Chat SFT를 직접 돌려보며 파인튜닝 파이프라인을 체득한다.

- **프레임워크**: [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)

- Apple Silicon에 최적화된 네이티브 ML 프레임워크. LoRA/QLoRA 지원, HuggingFace 체크포인트 변환 가능.

- **학습 방식**: LoRA 어댑터 학습 (`mlx_lm.lora`) 중심.

- **주요 산출물**: 학습된 LoRA 어댑터, TensorBoard 로그, 평가 리포트, 추론 스크립트.

---
## 2. 개발 환경 (전제)

  
| 항목 | 값 |

|---|---|

| 하드웨어 | MacBook Pro M4, 48GB 통합 메모리 |

| OS | macOS (Apple Silicon, arm64) |

| Python | 3.11 (Miniforge/Mambaforge) |

| 가상환경 | `conda` 환경 이름 `ftlab` |

| 프레임워크 | `mlx`, `mlx-lm` |

| 로깅 | TensorBoard (로컬 파일 기반) |

| 실행 위치 | 최종 실행은 Mac에서. (이 디렉토리는 Windows에서 명세 작성 단계)|


> **중요**: 이 프로젝트는 CUDA / bitsandbytes / flash-attn 경로를 **사용하지 않는다**. Apple Silicon 전용 구성이며, NVIDIA GPU 튜토리얼의 명령을 그대로 복사해 오면 안 된다.


---
## 3. 권장 디렉토리 구조
```
finetune-lab/

├── CLAUDE.md # 본 문서

├── README.md # 사람용 요약

├── environment.yml # conda 환경 정의

├── requirements.txt # pip 보조 패키지

├── Makefile # 자주 쓰는 명령 단축

├── configs/

│ └── qwen2_5_7b_koalpaca_lora.yaml # 학습 하이퍼파라미터

├── data/

│ ├── raw/ # 원본 다운로드

│ └── processed/ # chat template 적용된 JSONL

├── scripts/

│ ├── prepare_dataset.py # HF → MLX JSONL 변환

│ ├── convert_model.py # HF checkpoint → MLX 변환/양자화

│ ├── train_lora.py # mlx_lm.lora 래퍼

│ ├── eval_model.py # perplexity/샘플 생성 평가

│ └── chat.py # 학습된 어댑터로 대화 테스트

├── models/

│ ├── base/ # 변환된 베이스 모델 (gitignore)

│ └── adapters/ # 학습된 LoRA 어댑터

├── runs/ # TensorBoard 로그 (gitignore)

└── notebooks/ # 탐색·분석용

```

  

---
## 4. 초기 셋업 (Mac에서 최초 1회)

### 4.1. Miniforge 설치
```bash
# Apple Silicon 전용 conda 배포판

brew install --cask miniforge

conda init zsh
```

### 4.2. 환경 생성

```bash
conda env create -f environment.yml

conda activate ftlab
```

`environment.yml` 핵심 패키지:

- `python=3.11`

- `pip`, `ipykernel`

- pip 섹션: `mlx`, `mlx-lm`, `transformers`, `datasets`, `huggingface_hub`, `tensorboard`, `pyyaml`, `tqdm`

### 4.3. HuggingFace 로그인 (게이트 모델 사용 시)

```bash

huggingface-cli login

```

---

## 5. 기본 모델 / 데이터셋

### 모델 (3B~8B 중형, 48GB에 여유롭게 탑재 가능)

- **1차 권장**: `Qwen/Qwen2.5-7B-Instruct` — 한국어 성능 양호, 게이트 없음, MLX 변환 안정.

- **대안**: `meta-llama/Llama-3.1-8B-Instruct` (게이트 승인 필요), `google/gemma-2-9b-it`.

### 데이터셋

- **1차**: `beomi/KoAlpaca-v1.1a` — 한국어 Alpaca 스타일, 가장 보편적인 실습 데이터.

- **대안**: `HAERAE-HUB/kollm-converations`, `maywell/ko_wikidata_QA`.

### 변환 규약

- `scripts/prepare_dataset.py`는 HF 데이터셋을 다음 JSONL 포맷으로 변환한다 (mlx-lm `chat` 포맷):

```json

{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

```

- 출력 경로: `data/processed/{dataset}/{train,valid,test}.jsonl`

- train/valid/test = 90/5/5 기본 분할, 시드 고정(`--seed 42`).

---

## 6. 표준 파이프라인

### 6.1. 모델 변환/양자화 (최초 1회)

```bash
# HF → MLX, 4-bit 양자화 (48GB면 bf16도 가능하지만 학습 안정성 위해 q4 권장)

python scripts/convert_model.py \
--hf-path Qwen/Qwen2.5-7B-Instruct \
--mlx-path models/base/qwen2.5-7b-instruct-q4 \
--quantize --q-bits 4
```

### 6.2. 데이터 준비

```bash
python scripts/prepare_dataset.py \
--hf-name beomi/KoAlpaca-v1.1a \
--out data/processed/koalpaca \
--seed 42
```

  
### 6.3. LoRA 학습

```bash
python scripts/train_lora.py \
--config configs/qwen2_5_7b_koalpaca_lora.yaml
```

`configs/*.yaml` 은 **mlx_lm.lora 네이티브 스키마**를 그대로 사용한다 (`--config` 로 전달). 주요 키: `model`, `data`, `adapter_path`, `fine_tune_type`, `iters`, `batch_size`, `num_layers`, `learning_rate`, `val_batches`, `steps_per_eval`, `save_every`, `max_seq_length`, `grad_checkpoint`, `lora_parameters.{rank,scale,dropout,keys}`.

### 6.4. TensorBoard 모니터링

```bash
tensorboard --logdir runs/ --port 6006
# http://localhost:6006
```


### 6.5. 추론·채팅 테스트

```bash
python scripts/chat.py \
--model models/base/qwen2.5-7b-instruct-q4 \
--adapter models/adapters/qwen2.5-7b-koalpaca
```

  
### 6.6. 평가

```bash
python scripts/eval_model.py \
--model models/base/qwen2.5-7b-instruct-q4 \
--adapter models/adapters/qwen2.5-7b-koalpaca \
--data data/processed/koalpaca # test.jsonl 이 있는 디렉토리
```

  
---

## 7. 하이퍼파라미터 초깃값 (7B + 48GB 기준)
```yaml
iters: 1000

batch_size: 2

num_layers: 16 # 상위 16개 transformer block 에만 LoRA

learning_rate: 1.0e-4

max_seq_length: 2048

grad_checkpoint: true

val_batches: 25

steps_per_report: 10

steps_per_eval: 100

save_every: 200

lora_parameters:

rank: 8

scale: 20.0

dropout: 0.05

```

- 메모리 부족 시: `batch_size=1`, `max_seq_length=1024`, `num_layers=8`로 축소.

- 과적합 징후 시: `lora_parameters.dropout` ↑, `iters` ↓.

---

## 8. Claude에게 주는 작업 규약

Claude Code가 이 저장소에서 작업할 때 따를 것:

1. **플랫폼 전제**: 모든 명령은 **macOS + Apple Silicon** 기준으로 제시한다. CUDA/bitsandbytes/flash-attn 관련 제안 금지.

2. **프레임워크**: 기본은 MLX (`mlx-lm`). PyTorch+MPS 제안은 사용자가 명시적으로 요청할 때만.

3. **데이터 포맷**: 새 데이터셋 추가 시 `data/processed/*/*.jsonl`의 `messages` 포맷을 반드시 따른다.

4. **설정 파일**: 학습 하이퍼파라미터는 코드에 박지 말고 `configs/*.yaml`로 분리한다.

5. **경로 상수화**: 스크립트는 상대경로 기준으로 동작하게 하고, 절대경로 하드코딩 금지.

6. **시크릿**: HuggingFace 토큰 등은 `~/.huggingface` 또는 환경변수로만. 코드·설정에 직접 쓰지 않는다.

7. **gitignore 필수**: `models/base/`, `models/adapters/`, `runs/`, `data/raw/`, `data/processed/`, `*.safetensors`, `*.bin`.

8. **무거운 작업 확인**: 5분 이상 걸릴 학습/변환을 시작하기 전에는 사용자에게 실행 여부를 확인한다.

9. **불필요한 추상화 금지**: 학습 스크립트는 한 파일에 읽기 쉽게 둔다. 조기 추상화/프레임워크화 하지 않는다.

10. **한국어 응답**: 사용자가 한국어로 질문하면 한국어로 답한다. 코드 주석은 간결하게, 왜(why)만 쓴다.

---

## 9. 자주 쓰는 명령 치트시트

```bash

# 환경

conda activate ftlab

conda env update -f environment.yml --prune

  

# MLX 정상 동작 확인

python -c "import mlx.core as mx; print(mx.default_device(), mx.ones((3,3)))"

  

# 디스크 정리 (모델/로그)

du -sh models/ runs/ data/

  

# 어댑터 합치기 (배포용)

python -m mlx_lm.fuse \

--model models/base/qwen2.5-7b-instruct-q4 \

--adapter-path models/adapters/qwen2.5-7b-koalpaca \

--save-path models/fused/qwen2.5-7b-koalpaca

```

  
---

## 10. 학습 로드맵 (제안)

1. **Week 1**: 환경 셋업 → 0.5B 모델(Qwen2.5-0.5B)로 end-to-end 파이프라인 한 바퀴.

2. **Week 2**: 7B + KoAlpaca LoRA 학습 1회 성공, TensorBoard 해석.

3. **Week 3**: 하이퍼파라미터(rank/lr/layers) 변주 실험, 결과 비교.

4. **Week 4**: 자체 도메인 JSONL 준비 → 소규모 커스텀 SFT.

5. **Week 5+**: DPO 등 선호학습, MLX vs PyTorch+MPS 벤치마크 비교(선택).


---

## 11. 참고 링크

- MLX Examples (LoRA): https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm

- mlx-lm docs: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md

- KoAlpaca: https://github.com/Beomi/KoAlpaca

- HuggingFace Datasets: https://huggingface.co/docs/datasets