
```
def build_lora_command(  
	*,
	model: str,
	data_dir: str,
	adapter_dir: str,
	iters: int = 300,
	batch_size: int = 1,
	num_layers: int = 8,
	learning_rate: float = 1e-5, # 학습률 (가중치 업데이트 크기, 1e-5 = 0.00001)  
) -> list[str]:
```
