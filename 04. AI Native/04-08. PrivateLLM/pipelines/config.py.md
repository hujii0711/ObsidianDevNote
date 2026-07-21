
```
@dataclass  
class Config:  
	@property  
	def raw_dir(self) -> Path:
	
	@property  
	def chunks_dir(self) -> Path:
	
	@property  
	def chroma_dir(self) -> Path:
	 
	def ensure_dirs(self) -> None:
	
	@classmethod  
	def from_env(cls) -> "Config":

```

