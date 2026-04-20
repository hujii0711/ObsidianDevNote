## Unsloth란?

**Unsloth**는 LLM(대형 언어 모델)을 **빠르고 메모리 효율적으로 파인튜닝(Fine-tuning)** 할 수 있게 해주는 Python 라이브러리입니다.

---

## 핵심 특징

**속도 & 메모리**

- 기존 대비 **2~5배 빠른** 학습 속도
- VRAM 사용량 **최대 80% 절감**
- 저사양 GPU (예: RTX 3090, 4090)에서도 대형 모델 파인튜닝 가능

**지원 모델**

- Llama 3, Mistral, Gemma, Phi, Qwen 등 주요 오픈소스 LLM 지원

**기술 기반**

- **LoRA / QLoRA** 방식으로 파인튜닝 (전체 가중치가 아닌 일부만 학습)
- 커스텀 CUDA 커널로 연산 최적화

---

## 기존 방식과 비교

||기존 (HuggingFace)|Unsloth|
|---|---|---|
|속도|기준|2~5배 빠름|
|VRAM|기준|최대 80% 절감|
|코드 변경|-|최소화|
|정확도 손실|-|없음|

---

## 간단한 사용 예시

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,  # 4bit 양자화로 VRAM 절감
)

# LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
)
```

---

## 어떤 상황에 유용한가?

- 개인 GPU로 LLM을 파인튜닝하고 싶을 때
- Google Colab 무료 티어(T4 GPU)에서 학습할 때
- 빠른 프로토타이핑이 필요할 때

---

**한마디로:** 고사양 서버 없이도 LLM을 내 데이터로 학습시키고 싶을 때 가장 먼저 고려하는 라이브러리입니다.