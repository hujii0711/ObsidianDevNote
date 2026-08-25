
1. vocab_files_names
	단어 집합 파일의 이름과 경로를 포함하는 딕셔너리이다.
	ex) {"vocab_file": "path/to/vocab.txt"}과 같은 형식으로 경로를 지정한다.

2. pretrained_vocab_files_map
	사전 학습된 단어 집합 파일의 매핑을 포함하는 딕셔너리이다.
	키는 모델의 이름이나 버전을, 값은 각 파일의 이름과 경로를 포함한다.
	ex) {"bert-base-uncased": {"vocab_file": "path/to/vocab.txt"}}과 같은 형식으로 경로를 지정한다.

3. pretrained_init_configuration
	사전 학습된 토크나이저 구성을 포함하는 딕셔너리이다.
	키는 모델의 이름이나 버전을, 값은 해당 모델의 토크나이저 구성을 나타낸다.

4. max_model_input_sizes
	모델의 최대 입력 길이를 지정하는 딕셔너리이다.
	키는 모델의 이름이나 버전을, 값은 해당 모델의 최대 입력 길이를 나타내는 정숫값을 입력한다.
	만약 None 값이라면 입력 길이에 제한이 없음을 의미한다.

5. model_max_length
	토크나이저가 사용하는 모델의 최대 입력 길이를 지정한다.
	이 값을 기준으로 입력 시퀀스를 자르거나 패딩한다.

6. padding_side
	입력 시퀀스에 패딩을 적용할 위치를 지정한다.
	"left"인 경우 왼쪽에 패딩을 추가하고 "right"인 경우 오른쪽에 패딩을 추가한다.

7. truncation_side
	입력 시퀀스가 model_max_length를 초과할 때, 어느 쪽에서 자를지 결정한다.
	"left"인 경우 왼쪽에서 자르고, "right"인 경우 오른쪽에서 자른다.

8. model_input_names
	순전파에 입력되는 텐서들의 이름 목록을 설정한다.
	예를 들어, BERT 모델의 경우 ["input_ids", "attension_mask", "token_type_ids"]와 같이 지정된다.

9. bos_token
	시퀀스의 시작을 나타내는 BOS(Beginning Of Sequence) 토큰을 설정한다.

10. eos_token
	시퀀스의 끝을 나타내는 EOS 토큰을 설정한다.

11. unk_token
	단어 집합에 없는 토큰을 대체하는 UNK(Unknowm) 토큰을 설정한다.

12. sep_token
	두 개의 시퀀스를 구분하는 SEP 토큰을 설정한다.

13. pad_token
	시퀀스를 패딩할 때 사용하는 PAD 토큰을 설정한다.

14. cls_token
	시퀀스 전체를 분류하는 CLS(Classification) 토큰을 설정한다.

15. mask_token
	마스크된 언어 모델링 작업에서 마스킹된 토큰을 나타내는 Mask 토큰을 설정한다.

16. additional_special_tokens
	위에 나열된 특수 토큰 외에 추가로 필요한 특수 토큰 목록을 설정한다.