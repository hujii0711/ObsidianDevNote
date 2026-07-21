
```
from .compare import compare_runs, load_summary, to_markdown

def main() -> None:  
	base = load_summary(_RUNS / "baseline.json")  
	qlora = load_summary(_RUNS / "qlora.json")  
	cmp = compare_runs(base, qlora)  
	print(to_markdown(cmp))
```
