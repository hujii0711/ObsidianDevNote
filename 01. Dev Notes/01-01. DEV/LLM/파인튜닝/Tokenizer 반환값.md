
```python
input = tokenizer(text, return_tensors="pt")
```


Hugging Face의 `transformers` 라이브러리에서 `input = tokenizer(text, return_tensors="pt")` (질문해주신 코드에서 `return_tensor`는 보통 `return_tensors="pt"` 복수형으로 사용됩니다)를 수행하고 나면, `input`은 파이토치(PyTorch) **텐서(Tensor)들이 담긴 딕셔너리(Dictionary) 객체**가 됩니다.

입력된 텍스트와 사용하는 모델(BERT, GPT 등)에 따라 세부적인 키와 크기는 달라지지만, 일반적으로 다음과 같은 구조와 값들을 포함하게 됩니다.

### 예상 출력 구조 (딕셔너리 형태)

```
{
    "input_ids": tensor([[  101,  2054,  2003,  1037,  3231,   102]]),
    "token_type_ids": tensor([[0, 0, 0, 0, 0, 0]]), # (BERT 등 일부 모델에만 존재)
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1]])
}

`input_ids`의 크기: `torch.Size([6])` 
`token_type_ids`의 크기: `torch.Size([6])`
`attention_mask`의 크기: `torch.Size([6])`
```

### 포함된 키(Key)별 상세 설명

1. **`input_ids` (가장 핵심)**
    - **설명:** 텍스트를 토큰화한 후, 각 토큰을 모델의 **사전(Vocabulary)에 해당하는 정수 ID로 변환한 텐서**입니다.
    - **형태(Shape):** `(배치 크기, 시퀀스 길이)` $\rightarrow$ 보통 문장 하나를 넣으면 `(1, 토큰 개수)` 형태가 됩니다.
    - **특징:** 모델에 따라 문장 시작을 알리는 특수 토큰(예: BERT의 `[CLS]` ID `101`)이나 끝을 알리는 토큰(`[SEP]`, `EOS` 등)이 자동으로 앞뒤에 붙을 수 있습니다.
    - 입력 ID는 입력 텍스트를 정수 인코딩으로 반환한 값을 의미한다. 이는 텍스트의 각 토큰을 고유한 정수로 변환해 모델에 입력으로 사용될 수 있도록 한다.

2. **`token_type_ids` (Segment IDs)**
    - **설명:** 문장이 두 개 이상 주어졌을 때(예: 질문과 답변 쌍, 문장 A와 문장 B), **어느 토큰이 첫 번째 문장 소속이고 어느 토큰이 두 번째 문장 소속인지** 구분해 주는 텐서입니다.
    - **값:** 첫 번째 문장 영역은 `0`, 두 번째 문장 영역은 `1`로 채워집니다. (단일 문장을 입력하거나 GPT 계열 등 일부 모델에서는 생성되지 않거나 사용되지 않습니다.)

3. **`attention_mask`**
    - **설명:** 모델이 어느 부분에 **주의(Attention)를 기울여야 하고, 어느 부분을 무시(Padding)해야 하는지** 알려주는 이진(Binary) 텐서입니다.
    - **형태(Shape):** `input_ids`와 동일한 크기 `(배치 크기, 시퀀스 길이)`
    - **값:** 유효한 토큰인 곳은 `1`, 문장을 맞추기 위해 빈자리를 채운 패딩(Padding) 토큰인 곳은 `0`이 들어갑니다. 패딩이 없는 단일 문장 처리 시에는 전부 `1`로 채워집니다.
    - 트랜스포머 인코더의 셀프 어텐션에 사용되는 마스크 값을 의미하며, 이는 모델이 어떤 토큰을 무시해야하는지를 지정하는 역할을 한다.

### `torch.Size([1, 10, 768])`

PyTorch에서 모델의 출력인 output.last_hidden_state.shape 통해서 형상 확인 --> `torch.Size([1, 10, 768])`은 주로 트랜스포머(Transformer) 기반의 언어 모델(예: BERT, GPT 등)에서 출력된 은닉 상태(Hidden States)나 임베딩 텐서의 형태(Shape)를 나타냅니다. 3차원 텐서로 구성되어 있으며, 각 차원의 의미는 다음과 같습니다.

- **첫 번째 차원 (`1` - Batch Size):** 배치 크기, 즉 한 번에 모델에 입력된 데이터(문장)의 개수입니다. 여기서는 1개의 샘플을 처리 중임을 의미합니다.

- **두 번째 차원 (`10` - Sequence Length):** 시퀀스 길이, 즉 문장이 몇 개의 토큰(단어 조각)으로 이루어져 있는지를 나타냅니다. 여기서는 총 10개의 토큰으로 구성된 입력입니다.

- **세 번째 차원 (`768` - Hidden Dimension):** 각 토큰이 가지는 벡터의 차원(은닉 크기)입니다. 예를 들어 **BERT-base** 모델은 각 토큰을 768차원의 벡터로 표현하므로 이 값이 고정되어 나타납니다.

즉, "1개의 문장 안에 10개의 토큰이 있고, 각 토큰이 768차원의 특징 벡터로 변환되어 있는 상태"를 의미합니다.

### 실제 코드 확인 예시

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello, world!", return_tensors="pt")

print(inputs)
```

**실행 결과 예시:**

```python
{
    'input_ids': tensor([[ 101, 7592, 1010, 2088,  999,  102]]), 
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
}
```

- 패딩: 각 입력 텍스트는 길이가 모두 다르기 때문에, 배치 처리를 위해 정수 인코딩과 어텐션 마스크를 동일한 길이로 맞춰야 한다. (토큰수에 맞게)
1) 모든 데이터를 모델의 최대 길이나 다른 정해진 길이로 패딩하는 방법
2) 배치 내에서 가장 긴 데이터의 길이로 패딩하는 방법