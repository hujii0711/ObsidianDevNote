
PreTrainedConfig는 허깅페이스의 사전 학습된 모델들의 구성을 정의하는 기본 클래스이다.
허깅페이스의 사전 학습된 모델들에서 공통으로 사용되는 기본 구성 클래스로, 모델의 구조와 하이퍼파라미터를 간편하게 정의하고 관리할 수 있다.

1. model_type
	모델의 유형을 나타내는 문자열이다.(ex. bert, gpt2등)

2. vocab_size
	모델의 어휘 사전 크기이다.
	모델이 인식할 수 있는 고유 토큰의 수를 결정한다.

3. hidden_size
	모델의 은닉 계층에 있는 노드의 수이다.
	이 값이 클수록 모델의 표현 능력이 높아진다.

4. num_attension_heads
	모델의 멀티 헤드 어텐션에서 사용되는 어텐션 헤드의 수이다.

5. num_hidden_layers
	모델의 트랜스포머 계층 수이다.
	계층의 수가 많을수록 모델의 표현 능력이 높아진다.

6. output_hidden_states
	모델이 모든 은닉 상태를 출력할지를 결정한다.

7. output_attensions
	모델이 모든 어텐션 값을 출력할지를 결정한다.

8. return_dict
	모델이 일반 튜플 대신 ModelOutput 객체를 반환할지를 결정한다.

9. is_encoder_decoder
	모델이 인코더-디코더 모델인지를 나타낸다.

10. is_decoder
	모델이 디코더 모델인지를 나타낸다.