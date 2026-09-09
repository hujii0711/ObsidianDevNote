
`Trainer`(Hugging Face `transformers`)와 `SFTTrainer`(Hugging Face `trl`)의 차이를 정리해드릴게요.

## 1. 소속 라이브러리와 설계 목적

||`Trainer`|`SFTTrainer`|
|---|---|---|
|라이브러리|`transformers`|`trl` (Transformer Reinforcement Learning)|
|목적|범용 모델 학습 (분류, 회귀, LM 등 모든 태스크)|LLM의 지도 미세조정(Supervised Fine-Tuning)에 특화|
|관계|베이스 클래스|내부적으로 `Trainer`를 상속·확장한 클래스|

`SFTTrainer`는 `Trainer`를 그대로 상속받아서 만든 클래스입니다. 즉 `Trainer`가 가진 기능(옵티마이저, 스케줄러, 로깅, 체크포인트, 분산학습 등)은 그대로 다 쓸 수 있고, 그 위에 LLM SFT에 필요한 편의 기능이 추가된 형태입니다.

## 2. 데이터 전처리 방식의 차이 — 가장 큰 차이점

**`Trainer`**

- 입력 데이터가 이미 토크나이징되어 `input_ids`, `attention_mask`, `labels`까지 다 준비되어 있어야 합니다.
- 즉, 앞서 만든 `messages` 형식의 데이터를 직접 chat template 적용 → 토크나이징 → label masking(프롬프트 부분은 -100 처리 등)까지 사용자가 직접 구현해야 합니다.

```python
def preprocess(sample):
    text = tokenizer.apply_chat_template(sample['messages'], tokenize=False)
    tokenized = tokenizer(text, truncation=True, max_length=2048)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

train_dataset = train_dataset.map(preprocess)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,  # 이것도 직접 정의
)
```

**`SFTTrainer`**

- `messages` 컬럼(chat 형식)이나 `text` 컬럼을 가진 raw 데이터셋을 그대로 넣을 수 있습니다.
- 내부에서 자동으로 `apply_chat_template` 적용, 토크나이징, 패킹(packing), 라벨 마스킹을 처리해줍니다.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=train_dataset,  # {'messages': [...]} 형태 그대로 사용 가능
    processing_class=tokenizer,
)
```

→ 지금까지 만드신 `{'messages': [...]}` 형식 데이터셋은 사실 `SFTTrainer`에 바로 넣기 위한 포맷입니다.

## 3. LLM 특화 기능 (SFTTrainer만 지원)

- **Packing**: 여러 개의 짧은 샘플을 하나의 시퀀스로 이어붙여 GPU 활용도를 높이는 기능 (`packing=True`)
- **PEFT/LoRA 통합**: `peft_config`만 넘기면 LoRA/QLoRA 학습을 자동 적용
- **Completion-only loss**: 프롬프트(system/user) 부분은 loss 계산에서 제외하고 assistant 응답 부분에만 loss를 주는 `DataCollatorForCompletionOnlyLM` 같은 기능을 쉽게 적용 가능
- **Chat template 자동 처리**: 모델별 chat template을 자동으로 적용

## 4. 유연성 vs 편의성

- `Trainer`: 범용적이라 어떤 태스크든 커스터마이징이 자유롭지만, LLM SFT를 하려면 전처리/라벨링 로직을 다 직접 짜야 함
- `SFTTrainer`: LLM SFT에 최적화된 기본값들이 세팅되어 있어 빠르게 시작할 수 있지만, 세밀한 커스텀이 필요하면 내부 동작을 더 잘 알아야 함

## 요약

```
Trainer      : 범용 학습 엔진, 데이터 전처리는 100% 사용자 책임
SFTTrainer   : Trainer를 상속 + LLM SFT용 전처리/패킹/LoRA 자동화 wrapper
```

지금 진행 중이신 `messages` 형식 데이터 + system_prompt 구성 방식을 보면, `Trainer`보다는 `SFTTrainer`를 쓰시는 게 훨씬 적은 코드로 동일한 결과를 낼 수 있습니다.