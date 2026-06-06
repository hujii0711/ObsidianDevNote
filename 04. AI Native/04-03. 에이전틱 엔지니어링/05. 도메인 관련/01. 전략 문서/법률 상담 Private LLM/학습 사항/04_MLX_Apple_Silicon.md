# 학습 가이드 04: MLX 및 Apple Silicon ML

> Apple Silicon(M4)에서 LLM을 실행하기 위한 MLX 프레임워크 학습 가이드

---

## 1. Apple Silicon의 ML 특성

### 1.1 통합 메모리 아키텍처 (UMA)

```
일반 PC (NVIDIA GPU):
  CPU RAM (32GB) ←─ PCIe ─→ GPU VRAM (24GB)
  → 모델이 VRAM에 들어가야 함
  → VRAM 초과 시 OOM (Out of Memory)

Apple Silicon (M4 48GB):
  ┌─────────────────────────┐
  │   통합 메모리 48GB       │ ← CPU와 GPU가 같은 메모리 공유
  │   CPU ←→ GPU 직접 접근   │ ← 데이터 복사 불필요
  └─────────────────────────┘
  → 48GB 전체를 ML에 활용 가능
  → NVIDIA 24GB GPU보다 더 큰 모델을 로드할 수 있음
```

### 1.2 왜 MLX인가?

| 프레임워크 | Apple Silicon 지원 | 성능 |
|-----------|-------------------|------|
| PyTorch (MPS) | 부분 지원, 일부 연산 CPU 폴백 | 보통 |
| **MLX** | **네이티브 최적화** | **최고** |
| llama.cpp (Metal) | Metal GPU 가속 | 우수 |
| ONNX Runtime | CoreML 변환 필요 | 보통 |

MLX는 Apple이 만든 프레임워크로 Apple Silicon에 최적화되어 있습니다.

---

## 2. MLX 핵심 개념

### 2.1 기본 사용법

```python
import mlx.core as mx

# 배열 생성 (NumPy와 유사)
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])

# 연산 (자동으로 GPU에서 실행)
c = a + b
print(c)  # array([5, 7, 9], dtype=float32)

# 행렬 곱
A = mx.random.normal((3, 4))
B = mx.random.normal((4, 5))
C = A @ B  # GPU에서 실행
```

### 2.2 Lazy Evaluation (지연 평가)

```python
# MLX는 즉시 계산하지 않음 (계산 그래프만 구성)
a = mx.array([1, 2, 3])
b = a + 1        # 아직 계산하지 않음
c = b * 2        # 아직 계산하지 않음

# 실제로 값이 필요할 때 계산
print(c)          # 이 시점에 한 번에 계산
mx.eval(c)        # 또는 명시적으로 평가

# 왜? 여러 연산을 한 번에 최적화하여 GPU 효율을 높이기 위해
```

### 2.3 PyTorch와의 차이

| 특성 | PyTorch | MLX |
|------|---------|-----|
| 평가 방식 | Eager (즉시) | Lazy (지연) |
| 디바이스 관리 | `.to('cuda')` 명시 필요 | 자동 (통합 메모리) |
| 메모리 모델 | CPU/GPU 분리 | 통합 |
| API | `torch.tensor` | `mx.array` (NumPy 유사) |

### 학습 자료

```
필수:
├── MLX 공식 문서
│   → https://ml-explore.github.io/mlx/build/html/index.html
│
├── MLX Examples (공식 예제)
│   → https://github.com/ml-explore/mlx-examples
│
└── "MLX vs PyTorch" 비교 글
    → Apple 개발자 블로그

권장:
└── Apple WWDC ML 세션
    → Metal, Core ML, MLX 관련 세션
```

---

## 3. mlx-lm (LLM 전용 도구)

### 3.1 mlx-lm이란?

MLX 위에 구축된 LLM 전용 라이브러리로, 다음 기능을 제공합니다:
- 모델 로딩 (HuggingFace 모델 자동 변환)
- 텍스트 생성 (generate, stream_generate)
- LoRA/QLoRA 파인튜닝
- 모델 양자화
- 어댑터 병합 (fuse)

### 3.2 핵심 함수

```python
from mlx_lm import load, generate, stream_generate

# 모델 로딩
model, tokenizer = load("mlx-community/EEVE-Korean-10.8B-4bit")

# 텍스트 생성 (일괄)
response = generate(
    model, tokenizer,
    prompt="민사소송이란?",
    max_tokens=200,
    temp=0.3,
)

# 스트리밍 생성 (토큰 단위)
for token in stream_generate(
    model, tokenizer,
    prompt="민사소송이란?",
    max_tokens=200,
    temp=0.3,
):
    print(token.text, end="", flush=True)
```

### 3.3 LoRA 학습 (CLI)

```bash
# 학습
mlx_lm.lora \
    --model mlx-community/EEVE-Korean-10.8B-4bit \
    --train \
    --data data/processed \
    --batch-size 4 \
    --lora-layers 16 \
    --iters 1000

# 어댑터 병합
mlx_lm.fuse \
    --model mlx-community/EEVE-Korean-10.8B-4bit \
    --adapter-path adapters \
    --save-path fused_model

# 양자화 (필요 시)
mlx_lm.convert \
    --hf-path original-model \
    --mlx-path mlx-model \
    -q  # 4비트 양자화
```

### 3.4 mlx-lm 학습 데이터 포맷

```
mlx_lm.lora가 기대하는 디렉터리 구조:
data/processed/
├── train.jsonl    # 필수
├── valid.jsonl    # 선택 (있으면 검증에 사용)
└── test.jsonl     # 선택

각 줄의 포맷:
{"text": "전체 대화 텍스트"}
또는
{"instruction": "질문", "input": "", "output": "답변"}
```

### 학습 자료

```
├── mlx-lm 공식 리포지토리
│   → https://github.com/ml-explore/mlx-examples/tree/main/llms
│
├── "Fine-Tuning with LoRA or QLoRA" (mlx-lm 문서)
│   → 학습 파라미터 상세 설명
│
└── mlx-community (HuggingFace)
    → https://huggingface.co/mlx-community
    → MLX 변환된 모델 목록
```

### 실습 과제

```python
# 1. MLX 기본 연산
import mlx.core as mx
a = mx.random.normal((100, 100))
b = mx.random.normal((100, 100))
c = a @ b
mx.eval(c)
print(f"행렬곱 결과 shape: {c.shape}")

# 2. 모델 로딩 및 생성 테스트
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/EEVE-Korean-10.8B-4bit")
print(generate(model, tokenizer, prompt="대한민국 수도는", max_tokens=50))

# 3. 추론 속도 측정
import time
start = time.time()
response = generate(model, tokenizer, prompt="민사소송 절차를 설명해주세요.", max_tokens=200)
elapsed = time.time() - start
token_count = len(tokenizer.encode(response))
print(f"속도: {token_count / elapsed:.1f} tokens/sec")
```

---

## 4. 성능 최적화 팁

### 4.1 M4에서의 추론 최적화

```
1. 양자화 사용: FP16 대신 Q4 → 속도 ~50% 향상
2. 불필요한 앱 종료: 메모리 압력이 높으면 스왑 발생 → 속도 급감
3. 전원 연결: 배터리 모드에서는 GPU 클럭 제한 가능
4. MLX 최신 버전 유지: Apple이 지속적으로 최적화 중
```

### 4.2 메모리 관리

```python
import mlx.core as mx

# 메모리 사용량 확인
print(f"Peak memory: {mx.metal.get_peak_memory() / 1e9:.2f} GB")
print(f"Active memory: {mx.metal.get_active_memory() / 1e9:.2f} GB")

# 캐시 정리
mx.metal.clear_cache()
```

---

## 5. 학습 체크리스트

| # | 주제 | 이해 수준 | 필수 여부 |
|---|------|----------|----------|
| 1 | Apple Silicon 통합 메모리 아키텍처 이해 | 개념 이해 | 필수 |
| 2 | MLX vs PyTorch 차이점 (Lazy Evaluation) | 개념 이해 | 필수 |
| 3 | mlx-lm으로 모델 로딩 및 생성 | 실습 | 필수 |
| 4 | mlx-lm stream_generate 사용법 | 실습 | 필수 |
| 5 | mlx-lm LoRA 학습 CLI 사용법 | 실습 | 필수 |
| 6 | mlx-lm 모델 병합 (fuse) | 실습 | 필수 |
| 7 | MLX 메모리 모니터링 | 실습 | 권장 |
| 8 | MLX 커스텀 모델 구현 | 심화 | 선택 |
