
data  
 ┣ adapters  
 ┃ ┗ qlora(QLoRA 어댑터(`adapters.safetensors` + 100/200/300 체크포인트)) <Plan 3C에서 생성>
 ┃ ┃ ┣ 0000100_adapters.safetensors  
 ┃ ┃ ┣ 0000200_adapters.safetensors  
 ┃ ┃ ┣ 0000300_adapters.safetensors  
 ┃ ┃ ┣ adapter_config.json  
 ┃ ┃ ┗ adapters.safetensors  
 ┣ chroma  (벡터DB(12M, `jeonse_deposit`)) <Plan 1 index에서 생성>
 ┃ ┣ 4d3a75ce-d936-4d7e-af45-df6eaf3acadd  
 ┃ ┃ ┣ data_level0.bin  
 ┃ ┃ ┣ header.bin  
 ┃ ┃ ┣ index_metadata.pickle  
 ┃ ┃ ┣ length.bin  
 ┃ ┃ ┗ link_lists.bin  
 ┃ ┗ chroma.sqlite3  
 ┣ chunks  
 ┃ ┣ aaa.jsonl  
 ┃ ┗ chunks.jsonl  
 ┣ eval_runs  (`baseline.json` · `qlora.json` (A/B 측정 결과)) <Plan 3A/3C에서 생성>
 ┃ ┣ baseline.json  
 ┃ ┗ qlora.json  
 ┣ ft  (`train.jsonl`(45)·`valid.jsonl`(4)·`test.jsonl`·`stats.json`) <Plan 3B에서 생성>
 ┃ ┣ stats.json  
 ┃ ┣ test.jsonl  
 ┃ ┣ train.jsonl  
 ┃ ┗ valid.jsonl  
 ┣ raw  
 ┃ ┣ law (수집 원본 JSON(법령 2) <Plan 1 ingest에서 생성>
 ┃ ┃ ┣ 민법.json  
 ┃ ┃ ┗ 주택임대차보호법.json  
 ┃ ┗ prec (수집 원본 JSON(판례 5파일)) <Plan 1 chunk에서 생성>
 ┃ ┃ ┣ 197932.json  
 ┃ ┃ ┣ 206299.json  
 ┃ ┃ ┣ 207174.json  
 ┃ ┃ ┣ 207180.json  
 ┃ ┃ ┣ 219919.json  
 ┃ ┃ ┣ 226751.json  
 ┃ ┃ ┣ 227015.json  
 ┃ ┃ ┣ 233877.json  
 ┃ ┃ ┣ 238047.json  
 ┃ ┃ ┣ 238469.json  
 ┃ ┃ ┣ 238665.json  
 ┃ ┃ ┣ 238841.json  
 ┃ ┃ ┣ 238851.json  
 ┃ ┃ ┣ 239217.json  
 ┃ ┃ ┣ 239521.json  
 ┃ ┃ ┣ 241081.json  
 ┃ ┃ ┣ 241383.json  
 ┃ ┃ ┣ 241585.json  
 ┃ ┃ ┣ 605771.json  
 ┃ ┃ ┣ 606045.json  
 ┃ ┃ ┣ 606119.json  
 ┃ ┃ ┣ 606789.json  
 ┃ ┃ ┣ 607789.json  
 ┃ ┃ ┗ 612881.json  
 ┗ .DS_Store