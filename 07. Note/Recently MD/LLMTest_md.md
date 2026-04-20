## Project Overview

학습용(education) 소규모 한국어 LoRA 파인튜닝 프로젝트. <font color="#ff0000">Apple Silicon(M4, 48GB)에서 **MLX-LM**으로 4-bit 양자화된 Llama 3.1 8B Instruct 위에 LoRA 어댑터를 학습한다.</font> 원래 명칭인 "QLoRA"는 bitsandbytes 기반 NF4+LoRA를 의미하지만, Apple Silicon에서는 bitsandbytes가 동작하지 않으므로 MLX 네이티브의 "4-bit 양자화 베이스 + LoRA"로 동등한 설계를 재현한다.

- **Purpose**: 파이프라인(데이터 준비 → 학습 → 어댑터 평가)을 직접 굴려보며 이해. 성능 경쟁 아님.

- **Status**: 개발 환경은 Windows에서 구축 중. 실제 학습·평가는 macOS M4에서 수행.

## Environment Split

| 단계 | 위치 | 가능한 작업 |

|---|---|---|

| Dev | Windows (현재 디렉토리) | 스크립트 작성, 스키마/파서 검증, 데이터 준비, lint |

| Train / Eval | macOS M4 48GB | MLX-LM 학습·평가·추론 (MLX는 Apple Silicon 전용) |

- MLX 관련 import(`mlx`, `mlx_lm`)를 포함하는 코드는 **학습·평가 스크립트 내부로만** 한정. `prepare_data.py` 같은 CPU-only 유틸은 Windows에서도 돌아가야 함.

- 의존성 관리는 **Anaconda**. `environment.yml`로 버전 매핑 (Windows/macOS에서 동일 이름 env를 쓰되 MLX 패키지는 macOS에서만 설치).

## Training Specification

| 항목 | 값 |

|---|---|

| Base model | `mlx-community/Llama-3.1-8B-Instruct-4bit` (pre-quantized, 게이트 없음) |

| Method | LoRA over 4-bit quantized base (MLX-native, QLoRA 등가) |

| Task | Instruction / Chat SFT (single-turn 우선, multi-turn 확장 가능) |

| Framework | `mlx-lm` (Apple MLX) |

| Language | 한국어 |

| Dataset size | 500–5,000 샘플 |

| Raw data format | Alpaca JSONL: `{"instruction","input","output"}` |

| Training data format | MLX chat format: `{"messages":[{"role":"user",...},{"role":"assistant",...}]}` |

| Hardware | MacBook Pro M4, 48GB unified memory |

| Eval metric | ROUGE-L + BLEU (sacrebleu, `tokenize="char"`) |

| Artifact | MLX LoRA adapter (`adapters.safetensors` + `adapter_config.json`) |

  
### LoRA defaults (starting point)

- rank 8, alpha 16, dropout 0.0

- `--num-layers 16` (상위 16개 트랜스포머 블록에만 적용)

- max seq length 2048, batch 1, grad accum 4–8

- iterations 600–1200 (데이터 규모에 따라 조정; 500건 × 3 epoch 상당이면 ≈ 600 iter @ grad-accum 8)

- learning rate 1e-5 ~ 1e-4, cosine schedule

  
값 변경 시 `configs/` YAML에 근거와 함께 기록.

## Planned Repository Layout

```

data/

raw/ 원본 파일(JSON/JSONL/CSV)

processed/ MLX chat JSONL — train.jsonl / valid.jsonl (파일명 고정)

eval/ 고정 평가 프롬프트 (학습 split과 분리)

configs/ train_*.yaml

scripts/

prepare_data.py Alpaca raw → MLX chat JSONL 변환·분할

train.py configs/*.yaml → mlx_lm.lora 래퍼

evaluate.py 어댑터 로드 → 생성 → ROUGE-L + char-BLEU

outputs/

adapters/ run-NNN/adapters.safetensors, adapter_config.json (비커밋)

notebooks/ 탐색·분석용 ipynb

environment.yml conda env 정의

```


MLX-LM은 데이터 디렉토리에 `train.jsonl` / `valid.jsonl`(/ 선택적 `test.jsonl`)을 기대하므로 파일명 고정.

## Commands
bash 기준(macOS zsh/bash, Windows Git Bash 공통). 명령 앞에 `cd`를 붙이지 않고 프로젝트 루트에서 실행.

```bash

# 최초 1회: conda env 생성
conda env create -f environment.yml
conda activate llmtest

# macOS에서 추가:
pip install mlx mlx-lm

# 데이터 준비 (Windows/macOS 모두 가능)
python scripts/prepare_data.py --in data/raw --out data/processed

# 학습 (macOS 전용)
python scripts/train.py --config configs/train_llama31_8b.yaml
  
# 평가 (macOS 전용)
python scripts/evaluate.py \
--model mlx-community/Llama-3.1-8B-Instruct-4bit \
--adapter outputs/adapters/run-001 \
--eval data/eval/prompts.jsonl

# 빠른 수동 점검(macOS)

mlx_lm.generate \
--model mlx-community/Llama-3.1-8B-Instruct-4bit \
--adapter-path outputs/adapters/run-001 \
--prompt "다음 문장을 공손한 어체로 바꿔주세요: 내일까지 자료 보내."
```

## Conventions

- **MLX import 격리**: `mlx` / `mlx_lm`을 쓰는 코드는 학습·평가 스크립트 안에만. 공용 유틸(`prepare_data.py` 등)에서는 절대 import 하지 않음 — Windows에서 깨짐.

- `train.py` / `prepare_data.py`는 `--seed` 인자를 받고 기본값 42. 최종 config/seed를 `outputs/adapters/run-NNN/resolved_config.yaml`로 저장.

- 실험 폴더명은 `run-NNN`. 같은 이름으로 덮어쓰지 않기.

- **데이터 흐름**: `data/raw/`는 입력 전용. `data/processed/`는 언제든 `prepare_data.py`로 재생성 가능하다는 전제 — 중간 수동편집 금지.

- **평가 세트 고정**: `data/eval/prompts.jsonl`은 학습 split과 분리. 내용 변경 시 파일명에 버전 태그(`prompts_v2.jsonl`).

- **비커밋**: `outputs/**`, `*.safetensors`, `*.gguf`, `data/raw/**`, `data/processed/**` (`.gitkeep`만 유지).

## Constraints to Remember

- **MLX는 Apple Silicon 전용**. `import mlx`는 Windows/Intel Mac에서 실패. 학습·평가 스크립트는 macOS에서만 실행 가능하다는 전제로 작성.

- **bitsandbytes / flash-attn / xformers는 쓰지 않음** — Apple Silicon 미지원. 의존성에 넣지 말 것. 4-bit 양자화는 이미 `mlx-community/*-4bit` 모델에 내장되어 있어 학습 시 추가 양자화 불필요.

- 48GB 통합메모리는 8B-4bit LoRA에 여유롭지만 시스템 공용이므로 학습 중 다른 메모리 무거운 앱은 종료 권장.

- `meta-llama/Llama-3.1-8B-Instruct` 원본은 HF 게이트 모델. **mlx-community 변환본은 게이트 없음** — 이 경로를 기본으로 사용.

- 분 단위 이상 소요되는 학습(`train.py`)·평가(`evaluate.py`) 명령은 사용자에게 먼저 확인 후 실행.