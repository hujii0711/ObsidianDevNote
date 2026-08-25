

1. task
	수행하려는 과제를 설정한다.

2. model
	파이프라인에서 사용할 모델을 설정한다.
	모델의 이름이나 경로 또는 직접 PreTrainedModel(파이토치) 또는 TFPreTrainedModel(텐서플로) 객체를 전달할 수 있다.
	지정하지 않으면 task에 대한 기본 모델을 사용한다.

3. config
	모델 구성 파일의 이름이나 경로 또는 PreTrainedConfig 객체를 지정한다.
	지정하지 않으면 task에 대한 기본 모델 설정을 사용한다.

4. tokenizer
	파이프라인에서 사용할 토크나이저를 지정한다.
	토크나이저의 이름이나 경로 또는 PreTrainedTokenizer나 PreTrainedTokenizerFast 객체를 전달한다.
	지정하지 않으면 task에 대한 기본 토크나이저를 사용한다.

5. feature_extractor
	파이프라인에서 특징 추출기를 지정한다.
	특징 추출기의 이름이나 경로 또는 PreTrainedFeatrueExtractor 객체를 전달한다.
	지정하지 않으면 task에 대한 기본 특징 추출기를 사용한다.

6. image_processor
	파이프라인에서 사용할 이미지 프로세서를 지정한다.
	이미지 프로세서의 이름이나 경로 또는 BaseImageProcessor 객체를 전달한다.

7. framework
	파이프라인에서 사용할 딥러닝 프레임워크를 지정한다.
	pt, tf를 입력할 수 있다. 지정하는 않으면 설치된 프레임워크나 모델에 따라 자동으로 결정된다.

8. revision
	모델 허브에서 모델을 다운로드할 때 사용할 특정 버전을 지정한다.
	브랜치 이름, 태그 이름, 커밋 ID를 입력할 수 있다.

9. use_fast
	빠른 토크나이저(PreTrainedTokenizerFast)를 사용할지 여부를 지정한다.

10. token
	모델 허브에서 모델을 다운로드할 때 사용할 액세스 토큰을 지정한다.

11. device
	파이프라인에 할당될 장치를 설정한다.
	cpu, cuda:1, mps, GPU 번호 및 파이토피 장치 인스턴스를 입력할 수 있다.

12. device_map
	모델의 각 모듈을 어떤 장치에 불러올지 지정한다.
	auto로 설정하면 가장 최적화된 device_map이 자동으로 계산된다. device와 device_map을 동시에 사용하면 충돌이 발생할 수 있으므로 사용에 주의한다.

13. torch_dtype
	모델의 가중치를 불러올 때 사용할 정밀도(torch.float16, torch.bfloat16 등)을 지정한다.

14. trust_remote_code
	사용자 정의 모델링, 구성, 토크나이저 또는 파이프라인 파일의 코드를 신뢰할지 여부를 지정한다.
	코드를 읽고 신뢰할 수 있는 저장소에서만 True로 설정해야 한다.

15. model_kwargs
	모델을 생성할 때 추가로 전달할 매개변수를 딕셔너리 형태로 지정한다.

16. pipeline_class
	사용자 정의 파이프라인 클래스를 지정한다.