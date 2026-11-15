
그래프가 실제로 실행되는 순서대로 정리했습니다. `media` 단계는 세 노드가 이어서 돌고, 시작점은 모두 [nodes.py](shorts_graph/media/nodes.py)입니다. 나머지 파일들은 이 노드들이 불러 쓰는 부품입니다.

```
storyboard(장면 나누기, media 밖)
   ↓
① scene_audio  ── tts.py
   ↓
② scene_images ── visuals.py → (Z-Image) → pixabay.py → images.py
   ↓
③ render       ── subtitles.py → render.py (→ images.py)
   ↓
save_memory(기록 저장, media 밖)
```
연결은 [graph.py:127-130](shorts_graph/graph.py#L127-L130)에 있습니다.

## ① scene_audio: 장면마다 음성 만들기
**[nodes.py:31](shorts_graph/media/nodes.py#L31) → [tts.py](shorts_graph/media/tts.py)**
1. 장면 나레이션을 `synthesize_sync()`(edge-tts)에 넘깁니다. `scenes/NN.mp3`가 만들어집니다.
2. TTS가 알려 주는 단어별 시작·끝 시간(`Word`)과 음성 길이(`duration`)를 `NN.words.json`에 저장합니다.
3. `NN.words.json`과 음성 파일이 이미 있는 장면은 건너뜁니다. 그래서 중간에 멈췄다 다시 시작해도 처음부터 만들지 않습니다.

## ② scene_images: 장면마다 이미지 만들기
**[nodes.py:76](shorts_graph/media/nodes.py#L76) → [visuals.py](shorts_graph/media/visuals.py)의 `get_scene_image()`**
1. **Z-Image**: `generate_zimage()`가 `image_prompt`로 AI 이미지를 생성합니다. mflux를 실행하며 장면당 약 1분이 걸립니다.
2. 실패하면 **Pixabay**로 넘어갑니다. [pixabay.py](shorts_graph/media/pixabay.py)의 `search_images()`가 `visual_query`로 사진을 검색하고 `download_image()`로 받습니다. 받은 사진은 [images.py](shorts_graph/media/images.py)의 `fit_to_vertical()`로 1080x1920에 맞춰 자릅니다.
3. 그것도 실패하면 **플레이스홀더**를 씁니다. [images.py](shorts_graph/media/images.py)의 `make_placeholder_image()`가 단색 배경에 흰 글씨 이미지를 그립니다.
4. 결과는 `.part.jpg`로 먼저 만든 뒤 `NN.jpg`로 이름을 바꿉니다. 이미 `NN.jpg`가 있으면 그대로 다시 씁니다.

## ③ render: 영상 합치기
**[nodes.py:106](shorts_graph/media/nodes.py#L106)**
1. **[subtitles.py](shorts_graph/media/subtitles.py)**
   - `words_to_cues()`가 장면별 단어 시간을 받아 14자 이하의 자막 덩어리(`Cue`)로 묶습니다.
   - 장면 길이를 차례로 더해서 영상 전체 기준의 시간으로 바꿉니다.
   - `format_srt()`가 결과를 `subtitles.srt`로 씁니다.
2. **[render.py](shorts_graph/media/render.py)의 `render_video()`** (moviepy 사용)
   - 장면 이미지를 각 음성 길이만큼 이어 붙여 영상을 만듭니다(`_scene_clip`).
   - 자막마다 `render_caption_image()`로 자막 이미지를 그려 해당 시간에 겹칩니다. 이때 [images.py](shorts_graph/media/images.py)의 `load_font`와 `wrap_text`를 씁니다.
   - 음성들을 이어 붙여 영상에 넣고 `final.part.mp4`로 저장합니다. 성공하면 `final.mp4`로 이름을 바꿉니다.

## 파일을 읽어 볼 순서
처음 보신다면, 다른 파일에 기대지 않는 부품부터 읽고 위로 올라가는 순서가 이해하기 쉽습니다.

**tts.py → subtitles.py → images.py → pixabay.py → visuals.py → render.py → nodes.py**

- tts.py, images.py, pixabay.py는 다른 media 파일을 쓰지 않습니다.
- subtitles.py는 tts.py의 `Word`를 씁니다.
- visuals.py는 images.py와 pixabay.py를, render.py는 images.py와 subtitles.py를 씁니다.
- nodes.py가 이것들을 모두 묶습니다.

참고로 `deps.tts`, `deps.image`가 실제로 어떤 함수인지는 [wiring.py](shorts_graph/wiring.py)의 `default_deps()`에서 정해집니다. 테스트에서는 이 자리에 가짜 함수가 들어갑니다.


네, 실제 실행(`start`, `resume`)에서는 `synthesize_sync`가 그대로 호출됩니다. 테스트와 오프라인 명령에서는 다른 함수가 호출됩니다.

## 실제 실행에서 연결되는 과정

1. **[wiring.py:43](shorts_graph/wiring.py#L43)**: `default_deps()`가 함수를 불러옵니다.
   ```python
   from shorts_graph.media.tts import synthesize_sync
   ```
2. **[wiring.py:78](shorts_graph/wiring.py#L78)**: `Deps`를 만들 때 그 함수를 `tts` 칸에 넣습니다.
   ```python
   return Deps(..., tts=synthesize_sync, ...)
   ```
   `synthesize_sync()`처럼 괄호를 붙이지 않았으므로 호출한 것이 아닙니다. 함수 자체를 값으로 넣은 것이라, 이제 `deps.tts`와 `synthesize_sync`는 같은 함수를 가리킵니다.
3. **[media/nodes.py:50](shorts_graph/media/nodes.py#L50)**: 노드에서 호출합니다.
   ```python
   tts = deps.tts(s["narration"], audio_path, prefs["voice"], prefs["rate"])
   ```
   결국 다음과 같은 호출입니다.
   ```python
   tts = synthesize_sync(s["narration"], audio_path, prefs["voice"], prefs["rate"])
   ```
   인자는 위치 순서대로 짝지어집니다.

   | 호출하는 쪽 | `synthesize_sync` 매개변수 |
   |---|---|
   | `s["narration"]` | `text` |
   | `audio_path` | `out_path` |
   | `prefs["voice"]` | `voice` |
   | `prefs["rate"]` | `rate` |

   반환값 `TtsResult`가 `tts` 변수에 담깁니다. 그 뒤 코드가 `tts.audio_path`, `tts.duration`, `tts.words`를 꺼내 씁니다.

## 상황별로 호출되는 함수

| 실행 상황 | `deps.tts(...)`가 실제로 부르는 것 |
|---|---|
| `uv run -m shorts_graph start` / `resume`, 웹에서 생성 | `synthesize_sync` (edge-tts로 진짜 음성 생성) |
| `uv run pytest` | `fake_tts` (무음 wav와 가짜 단어 시간) |
| `status` / `graph` / `prefs` 명령 | `_Offline` (호출 즉시 `GraphError`) |

확인해 보고 싶으시면 `synthesize_sync` 첫 줄에 `print("synthesize_sync 호출", file=sys.stderr)`를 잠깐 넣어 보세요. 실제 실행 때는 이 문구가 찍히고, 테스트 때는 찍히지 않습니다. stdout이 아니라 stderr로 찍어야 합니다. CLI의 stdout은 JSON 이벤트 전용이라 다른 출력이 섞이면 안 됩니다.


---

네, 실제 실행(`start`, `resume`)에서는 `synthesize_sync`가 그대로 호출됩니다. 테스트와 오프라인 명령에서는 다른 함수가 호출됩니다.

## 실제 실행에서 연결되는 과정

1. **[wiring.py:43](shorts_graph/wiring.py#L43)**: `default_deps()`가 함수를 불러옵니다.
   ```python
   from shorts_graph.media.tts import synthesize_sync
   ```
2. **[wiring.py:78](shorts_graph/wiring.py#L78)**: `Deps`를 만들 때 그 함수를 `tts` 칸에 넣습니다.
   ```python
   return Deps(..., tts=synthesize_sync, ...)
   ```
   `synthesize_sync()`처럼 괄호를 붙이지 않았으므로 호출한 것이 아닙니다. 함수 자체를 값으로 넣은 것이라, 이제 `deps.tts`와 `synthesize_sync`는 같은 함수를 가리킵니다.
3. **[media/nodes.py:50](shorts_graph/media/nodes.py#L50)**: 노드에서 호출합니다.
   ```python
   tts = deps.tts(s["narration"], audio_path, prefs["voice"], prefs["rate"])
   ```
   결국 다음과 같은 호출입니다.
   ```python
   tts = synthesize_sync(s["narration"], audio_path, prefs["voice"], prefs["rate"])
   ```
   인자는 위치 순서대로 짝지어집니다.


   | 호출하는 쪽 | `synthesize_sync` 매개변수 |
   |---|---|
   | `s["narration"]` | `text` |
   | `audio_path` | `out_path` |
   | `prefs["voice"]` | `voice` |
   | `prefs["rate"]` | `rate` |


   반환값 `TtsResult`가 `tts` 변수에 담깁니다. 그 뒤 코드가 `tts.audio_path`, `tts.duration`, `tts.words`를 꺼내 씁니다.

## 상황별로 호출되는 함수

| 실행 상황 | `deps.tts(...)`가 실제로 부르는 것 |
|---|---|
| `uv run -m shorts_graph start` / `resume`, 웹에서 생성 | `synthesize_sync` (edge-tts로 진짜 음성 생성) |
| `uv run pytest` | `fake_tts` (무음 wav와 가짜 단어 시간) |
| `status` / `graph` / `prefs` 명령 | `_Offline` (호출 즉시 `GraphError`) |

확인해 보고 싶으시면 `synthesize_sync` 첫 줄에 `print("synthesize_sync 호출", file=sys.stderr)`를 잠깐 넣어 보세요. 실제 실행 때는 이 문구가 찍히고, 테스트 때는 찍히지 않습니다. stdout이 아니라 stderr로 찍어야 합니다. CLI의 stdout은 JSON 이벤트 전용이라 다른 출력이 섞이면 안 됩니다.