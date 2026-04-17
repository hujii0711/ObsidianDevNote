# 학습 가이드 01: LLM 기초 이론

> 이 프로젝트에서 사용하는 LLM(대규모 언어 모델)의 핵심 개념을 이해하기 위한 학습 가이드

---

## 1. Transformer 아키텍처

### 왜 알아야 하는가?
EEVE-Korean-10.8B를 포함한 모든 현대 LLM은 Transformer 기반입니다.
모델의 동작 원리를 이해해야 파인튜닝 파라미터 선택과 성능 튜닝을 올바르게 할 수 있습니다.

### 핵심 개념

| 개념 | 설명 | 이 프로젝트와의 관계 |
|------|------|---------------------|
| **Self-Attention** | 입력 시퀀스의 각 토큰이 다른 모든 토큰과의 관계를 계산 | LoRA가 수정하는 q_proj, k_proj, v_proj가 바로 이 어텐션 레이어 |
| **Multi-Head Attention** | 여러 어텐션 헤드가 서로 다른 관계 패턴을 학습 | lora_layers 수가 이 레이어 중 몇 개를 튜닝할지 결정 |
| **Feed-Forward Network** | 어텐션 출력을 비선형 변환 | 모델 파라미터의 대부분을 차지 |
| **Positional Encoding** | 토큰의 순서 정보를 제공 | RoPE(Rotary Position Embedding) 방식 사용 |
| **Layer Normalization** | 학습 안정화를 위한 정규화 | 양자화 시 정밀도에 영향 |

### 학습 자료

```
입문 (개념 이해):
├── "Attention Is All You Need" 논문 요약
│   → https://arxiv.org/abs/1706.03762
│   → 유튜브 "3Blue1Brown - Attention in transformers" 시청 권장
│
├── "Illustrated Transformer" (Jay Alammar)
│   → https://jalammar.github.io/illustrated-transformer/
│   → 시각적으로 Transformer 구조를 가장 잘 설명한 글
│
└── "nanoGPT" (Andrej Karpathy)
    → https://github.com/karpathy/nanoGPT
    → 코드로 직접 GPT를 구현하며 이해

심화 (선택):
├── "Formal Algorithms for Transformers" (Google)
│   → https://arxiv.org/abs/2207.09238
│
└── Hugging Face NLP Course - Chapter 1
    → https://huggingface.co/learn/nlp-course
```

### 실습 과제

```python
# Transformer의 핵심인 Self-Attention을 직접 구현해보기
import numpy as np

def self_attention(Q, K, V):
    """Scaled Dot-Product Attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)       # 유사도 계산
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)  # Softmax
    return weights @ V                      # 가중 합산

# Q, K, V가 무엇인지, 왜 스케일링하는지 이해하기
```

---

## 2. 토큰화 (Tokenization)

### 왜 알아야 하는가?
토큰화 방식이 한국어 성능과 청킹 전략에 직접 영향을 미칩니다.
"512 토큰"이 실제로 한국어에서 얼마나 되는지 알아야 RAG 청크 크기를 올바르게 설정할 수 있습니다.

### 핵심 개념

| 방식 | 설명 | 예시 |
|------|------|------|
| **BPE** (Byte Pair Encoding) | 빈도 기반으로 서브워드 병합 | GPT, Llama 사용 |
| **SentencePiece** | 언어 무관 토크나이저 | 다국어 모델에서 주로 사용 |
| **WordPiece** | BPE 변형, 확률 기반 | BERT 사용 |

### 한국어 토큰화 특성

```
영어: "The contract is terminated" → 5 토큰
한국어: "계약이 해지되었습니다" → 7~12 토큰 (토크나이저에 따라)

→ 같은 의미라도 한국어가 더 많은 토큰을 소비
→ max_tokens, chunk_size 설정 시 이를 고려해야 함
→ 한국어 특화 모델(EEVE)을 선택한 이유 중 하나
```

### 실습 과제

```python
# 실제 모델의 토크나이저로 한국어 토큰 수 확인
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

texts = [
    "임대차 보증금 반환 청구",
    "민사소송법 제1조에 따르면 법원은 소송절차가 공정하고 신속하게 진행되도록 노력하여야 한다.",
    "원고는 피고에 대하여 금 5,000만 원의 지급을 구합니다.",
]
for text in texts:
    tokens = tokenizer.encode(text)
    print(f"[{len(tokens)}토큰] {text}")
    print(f"  토큰 분리: {tokenizer.convert_ids_to_tokens(tokens)}")
```

---

## 3. 양자화 (Quantization)

### 왜 알아야 하는가?
M4 48GB에서 10.8B 모델을 돌리려면 양자화가 필수입니다.
양자화 수준에 따른 성능/속도 트레이드오프를 이해해야 합니다.

### 핵심 개념

```
원본 모델 (FP16, 16비트)
  10.8B × 2 bytes = ~21.6 GB  ← M4 48GB에서 학습 어려움

양자화 모델 (Q4, 4비트)
  10.8B × 0.5 bytes = ~5.4 GB + 오버헤드 ≈ ~8 GB  ← 학습 가능!
```

| 양자화 수준 | 비트 | 모델 크기 | 품질 손실 | 속도 |
|-----------|------|----------|----------|------|
| FP16 | 16bit | ~22 GB | 없음 | 기준 |
| Q8 | 8bit | ~11 GB | 거의 없음 | +20% |
| **Q4** | 4bit | **~8 GB** | 미미함 | **+50%** |
| Q2 | 2bit | ~4 GB | 눈에 띔 | +80% |

### 양자화 방식

| 방식 | 설명 |
|------|------|
| **GPTQ** | 학습 데이터 기반 양자화, 높은 품질 |
| **AWQ** | 활성화 기반 가중치 양자화 |
| **GGUF** | llama.cpp 전용 포맷 |
| **MLX 4bit** | Apple MLX 네이티브 양자화 (이 프로젝트 사용) |

### 학습 자료

```
├── "A Gentle Introduction to Quantization" (Hugging Face Blog)
│   → https://huggingface.co/blog/merve/quantization
│
├── "GPTQ vs AWQ vs GGUF" 비교 글
│   → 양자화 방식별 장단점 이해
│
└── MLX 공식 양자화 문서
    → https://ml-explore.github.io/mlx/build/html/python/nn.html#quantization
```

---

## 4. 프롬프트 엔지니어링

### 왜 알아야 하는가?
RAG 검색 결과를 모델에 전달하는 프롬프트 설계가 응답 품질을 크게 좌우합니다.

### 핵심 기법

| 기법 | 설명 | 이 프로젝트 적용 |
|------|------|-----------------|
| **System Prompt** | 모델의 역할과 행동 규칙 정의 | "민사사건 전문 법률 상담 AI" 역할 부여 |
| **Few-shot** | 예시를 포함하여 응답 형식 유도 | 법률 답변 형식 예시 제공 가능 |
| **Chain-of-Thought** | 단계적 추론 유도 | "법적 근거 → 절차 → 주의사항" 순서 |
| **RAG Prompt** | 검색된 문서를 컨텍스트로 삽입 | "다음 참고 자료에 근거하여 답변하세요" |
| **Guardrails** | 불확실한 경우 거부/면책 | "추측하지 마세요", 면책 조항 |

### 이 프로젝트의 프롬프트 구조

```
[System] 당신은 민사사건 전문 법률 상담 AI입니다. 규칙: ...
[Context] 참고 자료: {RAG 검색 결과}
[User] 질문: {사용자 질의}
[Assistant] 답변:
```

### 학습 자료

```
├── "Prompt Engineering Guide"
│   → https://www.promptingguide.ai/kr (한국어 버전)
│
├── Anthropic Prompt Engineering 가이드
│   → https://docs.anthropic.com/claude/docs/prompt-engineering
│
└── OpenAI Prompt Engineering 가이드
    → https://platform.openai.com/docs/guides/prompt-engineering
```

---

## 5. 디코딩 전략

### 왜 알아야 하는가?
`temperature`, `top_p` 같은 생성 파라미터가 답변의 일관성과 창의성을 결정합니다.
법률 상담에서는 일관적이고 정확한 답변이 중요하므로 적절한 설정이 필수입니다.

### 핵심 파라미터

| 파라미터 | 이 프로젝트 설정 | 설명 |
|---------|----------------|------|
| **temperature** | 0.3 (낮음) | 낮을수록 결정적(deterministic), 법률 답변에 적합 |
| **top_p** | 0.9 | 누적 확률 상위 90% 토큰에서만 샘플링 |
| **max_tokens** | 2048 | 최대 생성 토큰 수 |

```
temperature 0.0 → 항상 같은 답변 (greedy)
temperature 0.3 → 약간의 변화, 사실 기반 답변에 적합 ← 이 프로젝트
temperature 0.7 → 자연스러운 다양성
temperature 1.0+ → 창의적이지만 부정확할 수 있음
```

---

## 6. 환각 (Hallucination)

### 왜 알아야 하는가?
법률 상담에서 환각은 잘못된 법적 조언으로 이어질 수 있어 가장 위험한 문제입니다.
이 프로젝트에서 RAG를 사용하는 가장 큰 이유가 환각 방지입니다.

### 환각 유형

| 유형 | 예시 | 위험도 |
|------|------|--------|
| 사실 왜곡 | 존재하지 않는 "민법 제9999조" 인용 | 매우 높음 |
| 법률 혼동 | 형사법 규정을 민사에 적용 | 높음 |
| 절차 오류 | 잘못된 소송 절차 안내 | 높음 |
| 자신감 있는 오답 | 확신 있는 어조로 틀린 정보 제공 | 매우 높음 |

### 이 프로젝트의 환각 방지 전략

```
1. RAG 강제: 모든 답변에 검색된 문서 컨텍스트 포함
2. 출처 명시: 참조한 법조문/판례를 응답에 포함
3. 불확실성 표현: 검색 결과 없으면 "관련 자료를 찾지 못했습니다"
4. 면책 조항: 모든 응답에 자동 첨부
5. 낮은 temperature: 0.3으로 설정하여 추측 최소화
```

### 학습 자료

```
├── "Survey of Hallucination in Natural Language Generation"
│   → https://arxiv.org/abs/2202.03629
│
└── "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    → https://arxiv.org/abs/2005.11401 (RAG 원논문)
```

---

## 7. 학습 체크리스트

| # | 주제 | 이해 수준 | 필수 여부 |
|---|------|----------|----------|
| 1 | Transformer의 Self-Attention 동작 원리 | 개념 이해 | 필수 |
| 2 | Q, K, V 행렬의 역할 | 개념 이해 | 필수 (LoRA 대상 이해) |
| 3 | 한국어 토큰화와 토큰 수 감각 | 실습 | 필수 |
| 4 | 양자화 수준별 트레이드오프 (Q4 vs Q8) | 개념 이해 | 필수 |
| 5 | System/User/Assistant 프롬프트 설계 | 실습 | 필수 |
| 6 | temperature, top_p의 효과 | 실습 | 필수 |
| 7 | 환각의 유형과 방지 전략 | 개념 이해 | 필수 |
| 8 | Positional Encoding (RoPE) | 개념 이해 | 선택 |
| 9 | BPE 토큰화 알고리즘 상세 | 심화 | 선택 |
