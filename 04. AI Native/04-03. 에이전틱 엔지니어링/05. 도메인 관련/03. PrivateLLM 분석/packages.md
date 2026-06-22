서빙과 평가가 같은 코드를 공유하도록 핵심 로직을 라이브러리로 분리한 영역이다.

packages  
 ┣ eval  
 ┃ ┣ src  
 ┃ ┃ ┗ eval  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ answer_metrics.py  (인용.구조.면책률)
 ┃ ┃ ┃ ┣ cli.py  
 ┃ ┃ ┃ ┣ dataset.py  
 ┃ ┃ ┃ ┣ judge.py  (LLM 채점)
 ┃ ┃ ┃ ┣ report.py  
 ┃ ┃ ┃ ┣ retrieval_metrics.py  
 ┃ ┃ ┃ ┗ runner.py  (`api.pipeline.run_chat` **재사용**(서빙=평가 동일 경로))
 ┃ ┣ tests  
 ┃ ┃ ┣ conftest.py  
 ┃ ┃ ┣ test_answer_metrics.py  
 ┃ ┃ ┣ test_dataset.py  
 ┃ ┃ ┣ test_eval_set_valid.py  
 ┃ ┃ ┣ test_judge.py  
 ┃ ┃ ┣ test_report.py  
 ┃ ┃ ┣ test_retrieval_metrics.py  
 ┃ ┃ ┗ test_runner.py  
 ┃ ┣ eval_set.jsonl  
 ┃ ┗ pyproject.toml  
 ┣ finetune  
 ┃ ┣ src  
 ┃ ┃ ┗ finetune  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ compare.py  
 ┃ ┃ ┃ ┣ compare_cli.py  
 ┃ ┃ ┃ ┗ train.py  
 ┃ ┣ tests  
 ┃ ┃ ┣ conftest.py  
 ┃ ┃ ┣ test_compare.py  
 ┃ ┃ ┗ test_train.py  
 ┃ ┗ pyproject.toml  
 ┣ ftdata
 ┃ ┣ src  
 ┃ ┃ ┗ ftdata  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ builder.py  
 ┃ ┃ ┃ ┣ cli.py  
 ┃ ┃ ┃ ┣ filter.py  
 ┃ ┃ ┃ ┣ generate.py  
 ┃ ┃ ┃ ┗ questions.py  
 ┃ ┣ tests  
 ┃ ┃ ┣ conftest.py  
 ┃ ┃ ┣ test_builder.py  
 ┃ ┃ ┣ test_filter.py  
 ┃ ┃ ┣ test_generate.py  
 ┃ ┃ ┗ test_questions.py  
 ┃ ┣ pyproject.toml  
 ┃ ┗ question_pool.jsonl  
 ┣ rag  
 ┃ ┣ src  
 ┃ ┃ ┗ rag  
 ┃ ┃ ┃ ┣ __init__.py  
 ┃ ┃ ┃ ┣ citations.py  (인용.출처 처리)
 ┃ ┃ ┃ ┣ config.py (RAG Config)
 ┃ ┃ ┃ ┣ embedder.py  (질의 임베딩) 
 ┃ ┃ ┃ ┣ prompt.py  (컨텍스트 주입 프롬프트(build_messages, 서빙.학습 입력 공통))
 ┃ ┃ ┃ ┣ retriever.py  (Chroma 검색)
 ┃ ┃ ┃ ┗ types.py 
 ┃ ┣ tests  
 ┃ ┃ ┣ conftest.py  
 ┃ ┃ ┣ test_citations.py  
 ┃ ┃ ┣ test_config.py  
 ┃ ┃ ┣ test_embedder.py  
 ┃ ┃ ┣ test_prompt.py  
 ┃ ┃ ┗ test_retriever.py  
 ┃ ┗ pyproject.toml  
 ┗ .DS_Store