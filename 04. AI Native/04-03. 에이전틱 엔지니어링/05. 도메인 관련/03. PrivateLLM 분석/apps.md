```
                         main.py  ◀── (uvicorn api.main:app)
                        /   |   |   \
                       /    |   |    \
                schemas  settings  pipeline   llm
                            |         |   \     |
                       rag.config   rag.   rag. (mlx_lm)
                                    prompt citations
```

apps  
 ┣ api  
 ┃ ┣ src  
 ┃ ┃ ┗ api  
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ llm.py (`MlxLLM` — Qwen2.5-7B MLX 추론(`adapter_path`로 QLoRA 적용 가능))
 ┃ ┃ ┃ ┣ main.py (FastAPI 앱, `/health` · `/chat`(SSE) 엔드포인트)
 ┃ ┃ ┃ ┣ pipeline.py (`run_chat` — 검색→프롬프트→생성 전체 오케스트레이션(eval도 재사용))
 ┃ ┃ ┃ ┣ schemas.py (입출력 데이터 계약(pydantic))
 ┃ ┃ ┃ ┗ settings.py (구성값(모델·토큰·온도·RAG))
 ┃ ┗ pyproject.toml  
 ┣ web  
 ┃ ┣ app  
 ┃ ┃ ┣ favicon.ico  
 ┃ ┃ ┣ globals.css  
 ┃ ┃ ┣ layout.tsx  
 ┃ ┃ ┣ page.tsx  
 ┃ ┃ ┗ providers.tsx  
 ┃ ┣ components  
 ┃ ┃ ┣ Chat.test.tsx  
 ┃ ┃ ┣ Chat.tsx  
 ┃ ┃ ┣ ChatInput.test.tsx  
 ┃ ┃ ┣ ChatInput.tsx  
 ┃ ┃ ┣ MessageBubble.test.tsx  
 ┃ ┃ ┣ MessageBubble.tsx  
 ┃ ┃ ┣ SourceCard.test.tsx  
 ┃ ┃ ┗ SourceCard.tsx  
 ┃ ┣ hooks  
 ┃ ┃ ┣ useChat.test.tsx  
 ┃ ┃ ┗ useChat.ts  
 ┃ ┣ lib  
 ┃ ┃ ┣ chatClient.test.ts  
 ┃ ┃ ┣ chatClient.ts  
 ┃ ┃ ┣ sse.test.ts  
 ┃ ┃ ┣ sse.ts  
 ┃ ┃ ┗ types.ts 
 ┃ ┣ public  
 ┃ ┃ ┣ file.svg  
 ┃ ┃ ┣ globe.svg  
 ┃ ┃ ┣ next.svg  
 ┃ ┃ ┣ vercel.svg  
 ┃ ┃ ┗ window.svg  
 ┃ ┣ store  
 ┃ ┃ ┗ uiStore.ts  
 ┃ ┣ .DS_Store  
 ┃ ┣ .env.local  
 ┃ ┣ .env.local.example  
 ┃ ┣ .gitignore  
 ┃ ┣ AGENTS.md  
 ┃ ┣ CLAUDE.md  
 ┃ ┣ README.md  
 ┃ ┣ eslint.config.mjs  
 ┃ ┣ next-env.d.ts  
 ┃ ┣ next.config.ts  
 ┃ ┣ package-lock.json  
 ┃ ┣ package.json  
 ┃ ┣ postcss.config.mjs  
 ┃ ┣ tsconfig.json  
 ┃ ┣ tsconfig.tsbuildinfo  
 ┃ ┣ vitest.config.ts  
 ┃ ┗ vitest.setup.ts  
 ┗ .DS_Store