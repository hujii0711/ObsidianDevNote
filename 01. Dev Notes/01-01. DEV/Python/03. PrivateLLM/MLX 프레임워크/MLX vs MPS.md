
**MLX**와 **MPS**는 둘 다 Apple Silicon(M 시리즈 칩)을 활용하지만, 목적과 수준이 다릅니다.

---

### MLX

- **Apple이 만든 머신러닝 프레임워크** (2023년 말 오픈소스 공개)
- NumPy/PyTorch 스타일의 API로 모델 학습·추론을 직접 구현
- CPU와 GPU가 **unified memory를 공유** → 데이터 복사 없이 연산
- Python/C++ API 제공, LLM 추론에 최적화
- 예: `mlx.core`, `mlx-lm`으로 LLaMA 같은 모델 직접 실행

### MPS (Metal Performance Shaders)

- **Apple의 GPU 가속 백엔드** (Metal 프레임워크의 일부)
- PyTorch, TensorFlow 등 기존 프레임워크에서 `device="mps"`로 사용
- Apple GPU에서 연산을 수행하는 **하위 레이어**
- 개발자가 직접 쓰기보다는 프레임워크가 내부적으로 활용

---

### 핵심 차이 요약

|항목|MLX|MPS|
|---|---|---|
|성격|독립 ML 프레임워크|GPU 가속 백엔드|
|사용 방식|직접 API 호출|PyTorch 등 통해 간접 사용|
|대상|Apple Silicon 전용 ML|Apple GPU 연산 일반|
|Unified Memory|완전 활용|부분적 활용|
|생태계|Apple 독자 생태계|기존 PyTorch 생태계|
|LLM 최적화|매우 높음|보통|

---

### 한 줄 정리

> **MPS**는 PyTorch가 Mac GPU를 쓰게 해주는 **드라이버 역할**이고,  
> **MLX**는 Apple이 처음부터 Apple Silicon을 위해 설계한 **새로운 ML 프레임워크**입니다.

Mac에서 LLM을 돌린다면 MLX가 훨씬 빠르고 메모리 효율이 좋습니다.