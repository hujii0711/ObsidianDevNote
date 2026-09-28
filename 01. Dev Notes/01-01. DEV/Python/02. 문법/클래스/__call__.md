
**파이썬의 `__call__` 메서드**는 클래스의 인스턴스를 함수처럼 직접 호출할 수 있게 만들어주는 특별한(magic) 메서드입니다.

인스턴스 이름 뒤에 괄호 `()`를 붙여서 실행(`instance()`)하면, 파이썬 내부적으로 이 `__call__` 메서드가 자동으로 실행됩니다.

**기본 사용법 예시**
```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

# 인스턴스 생성 (초기 상태 설정)
double = Multiplier(2)

# 인스턴스를 함수처럼 호출 (실제로는 __call__이 실행됨)
result = double(5)  # double.__call__(5)와 동일
print(result)       # 출력: 10
```

**주요 활용 목적**

- **상태(State)를 기억하는 함수:** 일반 함수는 호출될 때마다 내부 변수가 초기화되지만, `__call__`을 가진 클래스 인스턴스는 객체 내부의 상태(`self` 변수)를 유지한 채 함수처럼 동작할 수 있습니다.

- **프레임워크 내부 구조 (PyTorch 등):** 파이토치(PyTorch)의 `nn.Module` 같은 클래스는 `__call__`을 활용해 모델 객체를 `model(x)` 형태로 곧바로 호출할 수 있게 구현되어 있습니다. 이 과정에서 입력 데이터 전처리나 연산 전후의 훅(Hook) 기능이 함께 처리된 뒤 내부의 `forward` 메서드로 연결됩니다.

---

Python에서 객체가 함수처럼 작동하는 원리인 `__call__` 메서드를 활용한 간단한 데모 코드입니다.

```python
class Greeter:
    def __init__(self, greeting_word):
        self.greeting_word = greeting_word

    def __call__(self, name):
        # 변수 뒤에 괄호()를 붙여 호출할 때 내부적으로 이 메서드가 실행됩니다.
        return f"{self.greeting_word}, {name}님!"

# 1. 객체를 생성하여 변수에 할당합니다 (Hugging Face의 pipeline()과 유사)
greeter = Greeter("안녕하세요")

# 2. 함수처럼 변수를 바로 호출합니다 (pipe(prompt)와 유사)
result = greeter("홍길동")
print(result)  
# 출력: 안녕하세요, 홍길동님!
```

`Greeter` 클래스 내부에 `__call__` 메서드가 정의되어 있기 때문에, `greeter`는 단순한 데이터 변수가 아니라 "호출 가능한 객체(Callable Object)"가 됩니다. Hugging Face의 `pipe` 역시 이와 같은 원리로 설계되어 있어서, 복잡한 내부 로직을 함수를 쓰듯 `pipe(prompt)` 형태로 간단하게 실행할 수 있는 것입니다.

```python
# 테스트 데이터를 불러옵니다.
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))

# 샘플 데이터 설정합니다.
prompt = pipe.tokenizer.apply_chat_template(
	eval_dataset[rand_idx]["messages"][:2],
	tokenize=False,
	add_generation_prompt=True
)

outputs = pipe(prompt,
	max_new_tokens=256,
	do_sample=False,
	temperature=0.1,
	top_k=50,
	top_p=0.1,
	eos_token_id=pipe.tokenizer.eos_token_id,
	pad_token_id=pipe.tokenizer.pad_token_id
)

print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}".replace("<|im_end|>", ""))
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
eval_dataset[rand_idx]['messages'][2]['content'].replace("<|im_end|>", "") == outputs[0]['generated_text'][len(prompt):].strip()
```
