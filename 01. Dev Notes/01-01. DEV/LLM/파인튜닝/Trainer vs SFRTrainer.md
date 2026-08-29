
**Trainer** (Hugging Face `transformers`)는 범용 학습 루프예요. 어떤 모델·태스크든 쓸 수 있지만, 그만큼 사용자가 직접 해야 할 일이 많아요:

- 데이터셋 토크나이즈/포맷팅을 직접 처리
- loss 계산 방식을 모델 forward에 맡김 (커스텀 loss는 직접 subclass)
- 텍스트 patching/packing 같은 LLM 특화 기능 없음

**SFTTrainer** (TRL 라이브러리)는 `Trainer`를 상속해서 만든, LLM 지도 미세조정(Supervised Fine-Tuning) 전용 클래스예요. 주요 차이점:

- **데이터 전처리 자동화**: `dataset_text_field`나 `formatting_func`만 지정하면 토크나이징을 알아서 처리
- **Sequence packing**: 짧은 샘플들을 이어붙여 GPU 활용도를 높이는 기능 내장
- **PEFT/LoRA 통합**: `peft_config`만 넘기면 LoRA 적용된 학습이 바로 가능
- **Chat 템플릿 지원**: 대화형 데이터셋 포맷팅이 쉬움
- **completion-only loss**: 프롬프트는 무시하고 응답 부분만 loss 계산하는 옵션 (`DataCollatorForCompletionOnlyLM`)

**요약**: 일반적인 모델 학습(분류, 회귀 등)이나 세밀한 커스텀 제어가 필요하면 `Trainer`, LLM을 instruction/chat 데이터로 지도 미세조정할 때는 `SFTTrainer`가 훨씬 편리해요.

TRL 라이브러리는 업데이트가 잦아서 파라미터명이나 구조가 최근에 바뀌었을 수 있어요. 혹시 특정 버전 기준으로 정확한 API를 확인하고 싶으신가요?