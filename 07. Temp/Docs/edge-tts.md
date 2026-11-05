네, 됩니다. edge-tts로 [whisper/sample.txt](whisper/sample.txt)를 [whisper/sample_edge.mp3](whisper/sample_edge.mp3)로 만들었습니다. 이 파일을 다시 Whisper(`turbo`)로 변환해 보니 원문과 똑같은 문장이 나왔습니다.

```bash
uvx edge-tts --voice ko-KR-SunHiNeural -f whisper/sample.txt --write-media whisper/sample_edge.mp3
```

- **`uvx`**: 임시 환경에서 한 번만 실행하는 방식이라 pyproject.toml은 바뀌지 않았습니다. 파이썬 코드에서 쓰려면 `uv add edge-tts`로 설치하면 됩니다.
- **한국어 음성**: `ko-KR-SunHiNeural`(여성), `ko-KR-InJoonNeural`(남성), `ko-KR-HyunsuMultilingualNeural`(남성, 다국어) 중에서 고를 수 있습니다.
- **속도·음량·음높이**: `--rate=+20%`, `--volume=-10%`, `--pitch=+5Hz` 같은 옵션으로 조절합니다.
- **자막**: `--write-subtitles sample.srt`를 붙이면 자막 파일도 함께 만들어집니다.

**주의할 점**: edge-tts는 로컬 모델이 아닙니다. Microsoft Edge의 온라인 음성 합성 서비스를 쓰기 때문에 텍스트가 Microsoft 서버로 전송되고, 인터넷이 연결되어 있어야 합니다. Whisper처럼 완전히 로컬에서 돌리려면 macOS 기본 `say` 명령을 쓰면 됩니다. 앞에서 만든 `sample.mp3`가 `say`로 만든 것입니다. 음질은 edge-tts가 더 자연스럽습니다.

원하시면 edge-tts로 텍스트 파일을 음성으로 바꾸는 `tts.py`를 [transcribe.py](whisper/transcribe.py)와 같은 형태로 만들어 드릴까요?