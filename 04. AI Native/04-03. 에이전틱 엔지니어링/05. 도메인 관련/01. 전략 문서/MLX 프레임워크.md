
MLX는 Apple 머신러닝 연구팀이 개발한, Apple Silicon에서의 머신러닝을 위한 배열(array) 프레임워크입니다. 2023년 말에 처음 공개되었으며, NumPy, PyTorch, JAX 등에서 영감을 받아 설계되었습니다.

## 핵심 특징

**1. 친숙한 API** Python API가 NumPy를 가깝게 따르며, C++ API도 제공됩니다. PyTorch나 JAX를 써본 사람이라면 학습 곡선이 매우 완만합니다. Swift 바인딩도 있어서 iOS/macOS 네이티브 앱 개발과도 연결됩니다.

**2. 통합 메모리 모델 (Unified Memory)** 가장 큰 차별점입니다. MLX 배열은 공유 메모리에 존재하며, 데이터 전송 없이 CPU나 GPU 등 지원되는 어떤 디바이스에서도 연산이 가능합니다. CUDA처럼 `tensor.to('cuda')` 같은 명시적 디바이스 이동이 필요 없습니다. Apple Silicon의 CPU/GPU/Neural Engine이 같은 메모리 공간을 공유하는 하드웨어 특성을 그대로 활용한 설계입니다.

**3. 지연 연산 (Lazy Computation)** MLX의 연산은 지연(lazy) 방식이며, 배열은 필요할 때만 실체화(materialize)됩니다. 이를 통해 계산 그래프를 최적화할 수 있습니다.

**4. 함수 변환 (Composable Function Transformations)** 자동 미분, 자동 벡터화, 계산 그래프 최적화를 위한 합성 가능한 함수 변환을 지원합니다 — JAX의 `grad`, `vmap`, `jit` 같은 개념과 유사합니다.

## 주요 활용 분야

- **LLM 추론 및 파인튜닝**: LLaMA, Mistral, Qwen 등의 모델을 Mac에서 직접 실행하고 LoRA 등으로 튜닝
- **이미지 생성**: Stable Diffusion
- **음성 인식**: Whisper (MLX Whisper 패키지 제공)
- **수치 컴퓨팅**: NumPy 대체로도 사용 가능

## 생태계 현황 (2026년 기준)

MLX는 빠르게 채택이 늘고 있습니다. Ollama가 Apple Silicon에서의 가장 빠른 실행을 위해 MLX 기반으로 전환했고, 최신 macOS에서는 M5 칩의 Neural Accelerator를 활용해 행렬 곱셈 연산을 가속합니다. M5에서 Qwen 14B 4비트 양자화 모델 같은 것도 노트북에서 실용적으로 돌릴 수 있는 수준이 되었습니다.

## 한계점

MLX는 여전히 발전 중이며, 주로 연구·실험·로컬 개발에 사용됩니다. macOS에서는 좋은 성능을 보이지만, 성숙한 프레임워크에 비하면 프로덕션 배포 옵션이 제한적입니다. 또한 Apple Silicon 전용이라 NVIDIA GPU 등 다른 하드웨어에서는 사용할 수 없습니다.

## 누가 쓰면 좋을까

- Mac(M1 이후)으로 LLM을 로컬에서 돌려보고 싶은 개발자
- 클라우드 없이 온디바이스 ML 워크플로우를 구축하려는 사람
- Apple 생태계 네이티브 앱에 ML을 통합하려는 iOS/macOS 개발자
- 16GB 같은 비교적 작은 통합 메모리에서도 모델을 효율적으로 돌리고 싶은 연구자

설치는 `pip install mlx`로 매우 간단하고, 언어모델용으로는 `mlx-lm` 패키지가 따로 있어서 Hugging Face 모델을 거의 바로 실행할 수 있습니다. 특정 부분(예: 코드 예제, PyTorch와의 비교, 파인튜닝 방법 등)을 더 깊이 알고 싶으시면 말씀해 주세요.