# AI 유튜브 쇼츠 제작 에이전트 — 설계 문서

- 작성일: 2026-11-05
- 관련 문서: [쇼츠 제작과정.md](../../쇼츠%20제작과정.md) (제작 단계·도구 조사)

## 1. 의도 (왜 만드는가)

**LangGraph 학습용 프로젝트**다. 지금까지 ch01~ch21에서 따로 배운 기능을 하나의 실제 결과물로 묶어 보는 것이 목적이다. 다룰 기능은 `StateGraph`, 조건부 엣지, `Send` 병렬 실행, `interrupt` 사람 확인, 체크포인터, `create_agent` 도구 호출 에이전트, 구조화 출력이다.

결과물은 "주제 키워드를 넣으면 세로형 쇼츠 mp4 한 편이 나오는" 파이프라인이다. 실제 채널 운영은 목적이 아니다.

## 2. 목표 / 비목표 / 실패 조건

**목표**
- 주제 키워드 1개를 입력하면 30~60초짜리 9:16(1080×1920) 한국어 나레이션 쇼츠 mp4가 생성된다.
- 대본, 장면 JSON, 이미지, 음성, 자막, 메타데이터 같은 중간 산출물이 모두 파일로 남는다. 각 단계를 따로 확인하고 다시 실행할 수 있어야 한다.
- 위에 나열한 LangGraph 기능을 각각 최소 한 번 이상, 자연스러운 용도로 쓴다.
- 비용 0원. 로컬 모델과 무료 도구만 사용한다.

**비목표 (이번 범위 밖)**
- YouTube 자동 업로드와 성과 분석. 업로드 정보(metadata.json)만 만든다. 업로드는 확장 과제로 남긴다.
- 상용 수준의 영상 품질, 화려한 자막 애니메이션, AI 영상 생성.
- 운영 안정성. 재시도 정책, 큐, 동시 다발 실행은 다루지 않는다.

**실패 조건 (동작은 해도 실패로 보는 경우)**
- 최상위 흐름을 LLM이 라우팅하는 구조. 이번 프로젝트에서는 Supervisor 방식을 쓰지 않기로 했다.
- 유료 API에 의존하는 구조.
- 그래프 전체를 돌려야만 확인할 수 있는 블랙박스 구조. 학습용이므로 노드 단위로 이해하고 실행할 수 있어야 한다.

## 3. 제약

- 하드웨어: Apple M4 Pro, 메모리 48GB.
- LLM: 로컬 `mlx-community/Qwen3.6-35B-A3B-4bit`([model_run.sh](../../../model_run.sh), 포트 8080). 기존 [config/model.py](../../../config/model.py)의 `load_llm()`(`ChatOpenAI` 호환)을 그대로 재사용한다. 이 모델은 약 20GB 이상을 차지하므로, mflux 같은 무거운 로컬 모델과 동시에 띄울 때 메모리를 고려해야 한다.
- 이미 있는 것: langchain 1.x, langgraph, langchain-tavily(`TAVILY_API_KEY`가 [.env](../../../.env)에 있음), pillow.
- 없는 것: ffmpeg(시스템에 미설치), moviepy, edge-tts, mflux, Pexels API 키(2026-11-06 기준 무료 키 발급 일시중단).
- 언어: 나레이션과 자막 모두 한국어. 자막에는 한글 폰트가 필요하다(macOS 기본 `AppleSDGothicNeo` 등).
- 패키지 관리는 기존과 같이 `uv`와 [pyproject.toml](../../../pyproject.toml)을 사용한다.

## 4. 검토한 방식과 선택

| 방식 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **1. 고정 파이프라인 + 루프 (선택)** | `StateGraph`로 순서를 고정. 검수 루프는 조건부 엣지, 장면 처리는 `Send` 병렬, 사람 확인은 `interrupt` | 흐름이 결정적이라 디버깅이 쉽다. 로컬 모델 라우팅 실수가 없다. 배운 기능을 고루 쓴다 | "에이전트가 다음 담당을 고르는" 경험은 적다 |
| 2. Supervisor | 감독 에이전트가 역할 에이전트를 도구처럼 호출 | 멀티 에이전트다운 구조 | 순서가 사실상 고정이라 감독이 할 판단이 없다. 로컬 모델 라우팅 실수가 잦다 |
| 3. 계층형 팀 | 상위 감독 아래 기획팀·제작팀 서브그래프 | 계층 구조를 학습하기 좋다 | 규모에 비해 무겁다. 2의 단점이 두 겹으로 생긴다 |

**사용자가 1번을 선택했다.** 대신 "에이전트다운 판단"은 리서처와 비주얼 담당 노드 **내부**에서 `create_agent` 도구 호출 에이전트로 경험한다. Supervisor는 이 파이프라인 위에 얹는 확장 과제로 남긴다.

## 5. 설계

### 5.1 전체 흐름

```
START
  → researcher           (LLM 에이전트: Tavily로 주제 후보 3개와 근거 자료 수집)
  → pick_topic           (interrupt: 사람이 후보 중 하나를 고르거나 직접 입력)
  → writer               (LLM: 훅과 대본 작성)
  → reviewer             (LLM: 판정 + 피드백)
       ├─ 불합격이고 수정 횟수 < 3 → writer (피드백 반영해 다시 작성)
       └─ 합격 또는 수정 횟수 ≥ 3 → approve_script
  → approve_script       (interrupt: 사람이 대본 승인 / 직접 수정 / 재작성 요청)
  → storyboarder         (LLM 구조화 출력: 장면 목록)
  → [Send] scene_worker × N   (장면별 병렬: 화면 소스 확보 + TTS)
  → editor               (일반 노드: 자막 생성 + 렌더링)
  → publisher            (LLM: 제목·설명·해시태그 → metadata.json)
END
```

체크포인터는 `SqliteSaver`(`outputs/checkpoints.sqlite`)를 쓰고, 실행마다 `thread_id`를 둔다. *(설계자 결정)* `MemorySaver`는 프로세스가 끝나면 기록이 사라진다. Sqlite를 쓰면 `interrupt`에서 멈춘 실행을 다음 실행에서 이어 갈 수 있다. `get_state_history`로 이전 체크포인트에서 다시 실행하는(time travel) 학습도 할 수 있다. 의존성 `langgraph-checkpoint-sqlite`를 추가한다.

**사람 응답에 따른 분기** *(설계자 결정)*

| 지점 | 응답 | 다음 노드 |
|---|---|---|
| pick_topic | 후보 중 선택 | writer |
| pick_topic | 주제 직접 입력 | researcher로 돌아감 (입력한 주제 하나에 대해서만 근거 자료 검색) → 다시 pick_topic |
| approve_script | 승인 | storyboarder |
| approve_script | 재작성 요청 (코멘트 포함) | writer. 사람의 코멘트를 `review_feedback`으로 넘기고 `revision_count`는 0으로 초기화 → 다시 reviewer 루프 |
| approve_script | 직접 수정한 대본 | storyboarder. 사람이 최종 권한자이므로 reviewer를 거치지 않음 |

### 5.2 역할(노드)별 명세

| 노드 | 형태 | 입력 → 출력 | 비고 |
|---|---|---|---|
| researcher | `create_agent` + Tavily 도구 | 주제 키워드 → 후보 3개(제목, 한 줄 요약, 근거 URL·요점) | 학습 포인트: 도구 호출 에이전트 |
| pick_topic | `interrupt` | 후보 → 선택된 주제 | 학습 포인트: 사람 확인 |
| writer | LLM 구조화 출력 | 주제 + 근거 (+ 이전 피드백) → `hook`, `script` | 목표 분량: 한국어 약 250~350자 |
| reviewer | LLM 구조화 출력 | 대본 → `passed: bool`, `feedback: str` | 점검 항목: 분량, 첫 문장 훅, 근거와의 사실 일치. 학습 포인트: 조건부 엣지 루프 |
| approve_script | `interrupt` | 대본 → 승인 / 수정본 / 재작성 | |
| storyboarder | LLM 구조화 출력 | 대본 → 장면 5~8개 | 장면 필드: `index`, `narration`, `visual_query`(영어 검색어), `image_prompt`(영어) |
| scene_worker | `Send`로 장면마다 병렬 실행 | 장면 1개 → 이미지 경로, 음성 경로, 단어 타이밍, 실제 길이 | 내부 ⑤ 비주얼: `create_agent`가 장면 설명으로 mflux 프롬프트를 다듬어 생성(실패 시 placeholder). Pexels 키가 있으면 검색 결과를 먼저 보고 쓸지 판단. 내부 ⑥ 보이스: edge-tts (LLM 없음) |
| editor | 일반 함수 (LLM 없음) | 장면 결과 전체 → `subtitles.srt`, `final.mp4` | 장면 길이 = 음성 길이. 이미지에 느린 확대(Ken Burns) 효과, 하단에 큰 자막 |
| publisher | LLM 구조화 출력 | 대본 + 주제 → `metadata.json` (제목, 설명, 해시태그) | |

### 5.3 상태(State)

- 공통 상태: `keyword`, `candidates`, `topic`, `sources`, `hook`, `script`, `review_feedback`, `revision_count`, `scenes`, `scene_results`, `output_dir`, `metadata`
- `scene_results`는 `Send` 병렬 결과를 모아야 하므로 누적 리듀서를 쓴다. 병렬 실행이라 완료 순서가 섞이므로 editor에서 `index` 기준으로 정렬한다.
- *(설계자 결정)* storyboarder를 다시 실행하면 이전 결과가 중복으로 쌓인다. 이를 막기 위해 `operator.add` 대신 **초기화 신호를 받으면 비우는 커스텀 리듀서**를 쓴다. storyboarder가 실행될 때 `scene_results`를 비운다.

### 5.4 산출물 구조

```
outputs/<thread_id>/
  research.json   script.json   scenes.json
  scenes/01.jpg   scenes/01.mp3   scenes/01.words.json  ...
  subtitles.srt   final.mp4   metadata.json
```

`outputs/`는 git에 올리지 않는다.

### 5.4.1 코드 위치와 실행 (설계자 결정)

- 저장소 루트에 새 패키지 `shorts_agent/`를 만든다. 기존 `chNN/exNN.py` 연습 파일과 구분하기 위해서다. `config/`는 그대로 import해서 쓴다.
- 실행은 `uv run python -m shorts_agent "주제 키워드"`로 한다. `interrupt` 지점에서는 터미널 `input()`으로 사람의 응답을 받아 `Command(resume=...)`로 재개한다.
- 사전 조건: `./model_run.sh`로 로컬 LLM 서버가 떠 있어야 한다.

### 5.5 사용 도구 (조합 A, 전부 무료)

| 용도 | 도구 |
|---|---|
| LLM | 로컬 Qwen (`load_llm()`) |
| 검색 | Tavily |
| 이미지 | mflux `FLUX.1-schnell`(기본) → 실패 시 placeholder. Pexels API는 키가 있을 때만 쓰는 선택 기능(현재 키 발급 중단) |
| TTS | edge-tts (`ko-KR-SunHiNeural` 등). 단어 경계 이벤트를 자막 타이밍에 활용 |
| 렌더링 | moviepy 2.x + ffmpeg |

### 5.6 세부 결정 사항 *(설계자 결정, 검토 후 추가)*

**"단계별 재실행"의 의미**
- 각 노드는 `state → dict` 형태의 **평범한 함수**로 만든다. 테스트나 REPL에서 상태 dict만 넣으면 단독으로 호출할 수 있다.
- 각 노드는 자기 산출물을 `outputs/<thread_id>/`에 파일로 저장한다. 사람이 결과를 눈으로 확인하기 위한 용도다.
- 그래프 안에서 특정 단계부터 다시 돌리는 일은 Sqlite 체크포인터의 time travel로 처리한다.
- 파일에서 상태를 복원하는 `--from <노드>` CLI는 만들지 않는다.

**비주얼 담당의 판단 방식**
- 로컬 Qwen은 텍스트만 다루므로 이미지를 직접 볼 수 없다.
- (Pexels 키가 있을 때) Pexels 도구는 세로형(portrait) 결과만 걸러서 후보 목록(alt 텍스트, 크기, URL)을 돌려준다.
- 에이전트는 alt 텍스트가 장면 설명과 맞는지 보고 후보 하나를 고른다. 맞는 후보가 없으면 mflux 생성 도구를 호출한다.
- 비전 모델은 추가하지 않는다.

**메모리 (Qwen 약 20GB + mflux)**
- mflux 생성은 프로세스 안의 락으로 **한 번에 하나씩만** 실행한다. Pexels 검색과 TTS는 병렬 그대로 둔다.
- 모델은 `FLUX.1-schnell` 4bit 양자화 버전을 쓴다. 약 576×1024로 생성한 뒤 1080×1920으로 확대한다.
- mflux를 실행할 수 없는 환경이면 **플레이스홀더 이미지**(단색 배경 + 장면 설명 텍스트, Pillow)로 대체한다. 이 경우에도 파이프라인 실행은 성공으로 본다.

**Pexels API 키 (2026-11-06 기준 무료 키 발급 일시중단)**
- 현재 키를 발급받을 수 없으므로 **기본 이미지 소스는 mflux**다. mflux를 쓸 수 없으면 placeholder로 대체한다.
- 이미지 소스는 설정값 `IMAGE_SOURCE = mflux | pexels | placeholder`로 고른다(기본값 `mflux`).
- Pexels 도구는 **키가 있을 때만 에이전트에 등록되는 선택 기능**으로 구현한다. 나중에 `.env`에 `PEXELS_API_KEY`를 넣으면 켜진다.
- Pexels가 꺼져 있으면 비주얼 에이전트는 장면 설명을 바탕으로 mflux 프롬프트를 다듬어 생성 도구를 호출한다. 결과 파일이 없거나 생성에 실패하면 placeholder 도구를 호출한다.
- 테스트는 항상 `placeholder`로 실행하므로 키나 GPU가 없어도 돌아간다.

**영상 길이**
- 30~60초는 **목표값이지 강제 조건이 아니다.**
- reviewer가 글자 수(250~350자)로 1차 점검한다. editor는 최종 길이가 60초를 넘으면 경고만 출력한다.
- 쇼츠는 최대 3분까지 허용되므로 재작성 루프는 두지 않는다.

## 6. 구현 단계 (학습 순서)

1. **기획 파이프라인**: researcher → pick_topic → writer ⇄ reviewer → approve_script → storyboarder. 텍스트 산출물만 나온다. 미디어 도구 없이도 전부 돌아간다.
2. **제작 파이프라인**: scene_worker(`Send`) → editor. mp4가 나온다.
3. **마무리**: publisher와 그래프 시각화(`draw_ascii`/mermaid), 실행 스크립트.

각 단계가 끝날 때마다 실제로 실행되는 상태를 유지한다.

## 7. 빌더에게 맡기는 부분

- 패키지 내부 파일 분리 방식, 프롬프트 문구, 구조화 출력용 Pydantic 모델의 세부 필드명
- 자막 렌더링 방식(moviepy `TextClip`으로 할지, Pillow로 그려 이미지 클립으로 넣을지)과 폰트·크기·위치
- 장면 이미지를 9:16으로 자르거나 맞추는 방식
- 노드별 단위 테스트 범위(최소한 LLM이 없는 editor·자막 생성 로직은 테스트한다)
