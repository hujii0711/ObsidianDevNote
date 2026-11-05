Whisper는 두 곳에 나뉘어 있습니다. 실행 코드는 프로젝트 가상환경에, 모델 파일은 홈 폴더의 캐시에 있습니다.

**1. 파이썬 패키지 (`openai-whisper`, 약 2MB)**
```
/Users/fujii0711/Study/AI_Agent_LangGraph/.venv/lib/python3.11/site-packages/whisper/
```
이 프로젝트의 uv 가상환경(`.venv`)에 설치되어 있습니다. pyproject.toml의 dependencies에 원래 들어 있던 패키지입니다.

**2. 모델 파일 (처음 실행할 때 자동으로 받음)**
```
~/.cache/whisper/
├── base.pt              (약 145MB)
└── large-v3-turbo.pt    (약 1.6GB)  ← --model turbo
```
캐시는 프로젝트 밖에 있어서 다른 프로젝트에서도 같은 모델을 다시 받지 않고 씁니다. 디스크를 비우려면 이 파일을 지우면 되고, 필요할 때 다시 받아집니다.

**저장 위치를 바꾸고 싶을 때**
`load_model`에 `download_root`를 주면 됩니다.
```python
whisper.load_model(model_name, device=device, download_root="whisper/models")
```

[transcribe.py](whisper/transcribe.py)의 import 순서가 바뀐 것은 포매터가 정리한 것입니다. 저장소의 `whisper/` 폴더 이름 때문에 `import whisper`를 이 프로젝트의 모듈로 보고 아래로 옮겼습니다. 위 출력에서 보듯 실제로는 `.venv`에 설치된 패키지가 불러와지므로 동작에는 문제가 없습니다.