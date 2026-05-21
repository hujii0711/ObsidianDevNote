• 확인된 로컬 사양은 MacBook Pro, Apple M4 Pro 14코어 CPU, 20코어 GPU, 통합 메모리 48GB, 디스크 여유 약 589GiB입니다. 이 정도면 4B~14B급 LoRA/QLoRA 실험은 현실적이고, 26B~31B급은 추론은 가능하지만 파인튜닝은 빡빡한 편입니다.

• Questions 1/1 answered
  • 파인튜닝하려는 주된 목적은 무엇에 가장 가깝나요?
    answer: 한국어 업무 챗봇 (Recommended)

• 확인된 PC 사양 기준 추천은 Qwen3-14B-Instruct를 MLX-LM으로 LoRA 파인튜닝입니다.

  ## 로컬 LLM 파인튜닝 모델 추천

  ### Summary

  - 현재 PC: MacBook Pro / Apple M4 Pro / 20코어 GPU / 통합 메모리 48GB / 여유 디스크 약 589GiB
  - 목적: 한국어 업무 챗봇
  - 1순위 추천: Qwen3-14B-Instruct
  - 먼저 검증할 모델: Qwen3-8B-Instruct
  - 방식: 전체 파인튜닝이 아니라 LoRA 또는 QLoRA

  ### Recommended Models

  - Qwen3-14B-Instruct
      - 한국어 포함 다국어, 업무 챗봇, 요약, 질의응답에 가장 균형적입니다.
      - 48GB 통합 메모리에서 LoRA 실험 대상으로 현실적인 상한선입니다.
      - Apache 2.0 계열이라 라이선스가 비교적 다루기 쉽습니다.
  - Qwen3-8B-Instruct
      - 첫 실험용으로 추천합니다.
      - 데이터셋 포맷, 학습 파이프라인, 평가 기준을 검증하기 좋고 속도도 훨씬 낫습니다.
      - 8B에서 결과가 괜찮으면 14B로 올리는 순서가 안전합니다.
  - Ministral 3 14B Instruct
      - 최신 Apache 2.0 오픈 모델 후보입니다.
      - Mistral은 3B, 8B, 14B 모델을 로컬/엣지 용도로 공개했고, 멀티모달/다국어를 강조합니다.
      - 다만 한국어 업무 챗봇 기준으로는 Qwen 쪽을 먼저 권합니다.
  - Gemma 4 E4B 또는 Gemma 4 26B A4B
      - Google 계열을 쓰고 싶거나 이미지/오디오까지 고려한다면 후보입니다.
      - 26B A4B는 4bit 추론 메모리는 약 15.6GB로 가능해 보이지만, 파인튜닝은 추가 메모리가 필요하므로 48GB Mac에서는 보수적으로 접근해야 합니다.

  ### Implementation Direction

  - Apple Silicon에서는 MLX-LM을 우선 사용합니다.
      - MLX-LM은 Apple Silicon에서 LLM 실행과 파인튜닝을 지원하고, LoRA 및 full fine-tuning, quantized model fine-tuning을 지원합니다.
  - 추천 순서:
      - Qwen3-8B-Instruct로 데이터/프롬프트/평가셋 검증
      - 결과가 좋으면 Qwen3-14B-Instruct로 LoRA 파인튜닝
      - 긴 문서 QA가 중요하면 컨텍스트 길이를 무리하게 키우기보다 RAG와 결합

  ### Avoid Locally
  - 이런 모델은 API로 써서 데이터 생성, 평가, distillation 용도로 활용하는 쪽이 맞습니다.

  - base 모델, 8B LoRA, 14B LoRA 답변을 비교
  - 평가 기준:
      - 한국어 자연스러움
      - 사내/업무 용어 준수
      - 모르면 모른다고 답하는지
      - 불필요한 장문/추측 감소
      - 응답 속도

  ### Assumptions

  - 목적은 한국어 텍스트 기반 업무 챗봇입니다.
  - 로컬에서 학습까지 수행하려는 기준입니다.
  - full fine-tuning은 제외하고 LoRA/QLoRA 중심으로 봅니다.


  참고한 최신 공식/주요 출처: Qwen3.5 공식 블로그 (https://qwen.ai/blog?id=qwen3.5), Qwen3.5 Hugging Face
  (https://huggingface.co/Qwen/Qwen3.5-397B-A17B), Mistral 3 발표 (https://mistral.ai/news/mistral-3), Gemma 4 문서
  (https://ai.google.dev/gemma/docs/core), MLX-LM (https://github.com/ml-explore/mlx-lm).
