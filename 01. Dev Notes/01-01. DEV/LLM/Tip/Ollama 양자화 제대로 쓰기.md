
#### 1. Ollama 양자화 방식(기본으로 사용하면 품질과 정확도를 손해보는 이유)
- 양자화: 모델 품질을 내려놓는 대신 VRAM 사용량을 줄여서 쓰는 것
- Ollama에서 지원 양자화 목록

| 이름     | 비트수 | 설명                             |
| ------ | --- | ------------------------------ |
| BF16   | 16  | 원본(양자화 아님)                     |
| MXFP8  | 8   | 최신 GPU 아키텍처에 최적화된 양자화          |
| Q8_0   | 8   | GGUF Legacy 양자화                |
| NVFP4  | 4   | NVIDIA Blackwell GPU에 최적화된 양자화 |
| Q4_K_M | 4   | GGUF K-Quant 양자화               |
VRAM이 넉넉하면 양자화한 것을 사용하면 손해라는 것이다.
-> 양자화가 굳이 필요 없다면 정밀도가 높은 모델로 변경하는 것이 좋다.

#### 2. Ollama에 원하는 양자화 타입이 없을 때의 해결 방법
HF에서 GGUF 모델을 다운로드 받아서 Ollama에 수동으로 직접 등록해서 서빙할 수 있는 기능을 지원하고 있다.

#### 3. Ollama에서 모델 양자화 타입 변경하기
HF에서 GGUF 모델을 다운로드 받아서 Modelfile에 설정하여 ollama에 등록한다.
```shell
> ollama create qwen3.5:9b-iq2_m -f ./Modelfile
> ollama show
> ollama run qwen3.5:9b-iq2_m
```

