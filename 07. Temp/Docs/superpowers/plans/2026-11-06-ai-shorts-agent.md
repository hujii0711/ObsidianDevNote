# AI 유튜브 쇼츠 제작 에이전트 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 주제 키워드 하나를 넣으면 9:16 한국어 나레이션 쇼츠 mp4와 중간 산출물이 `outputs/<thread_id>/`에 생성되는 LangGraph 파이프라인을 만든다.

**Architecture:** 최상위는 순서가 고정된 `StateGraph`다. 대본 검수는 조건부 엣지 루프로, 사람 확인은 `interrupt` + `Command(goto=...)`로, 장면 처리는 `Send` 병렬로 구현한다. 리서처와 비주얼 담당은 내부에서 `create_agent` 도구 호출 에이전트를 쓴다. 보이스·편집은 LLM이 없는 일반 함수다. 모든 노드는 `make_<name>_node(deps)` 팩토리로 만든다. 의존성(LLM, 에이전트, TTS)을 주입할 수 있어서 테스트에서는 가짜로 바꿔 끼운다.

**Tech Stack:** Python ≥3.11, uv, langgraph 1.2, langchain 1.4 (`create_agent`, `ToolStrategy`), langchain-tavily, langgraph-checkpoint-sqlite, edge-tts, moviepy 2.x(내장 imageio-ffmpeg), Pillow, httpx, mflux(CLI, `uv tool`로 별도 설치), pytest

**Spec:** [Docs/superpowers/specs/2026-11-05-ai-shorts-agent-design.md](../specs/2026-11-05-ai-shorts-agent-design.md)

## Global Constraints

- 유료 API 금지. 최상위 흐름을 LLM이 라우팅하지 않는다(Supervisor 금지).
- LLM은 기존 `config.model.load_llm()`을 쓴다. `shorts_agent` 모듈은 import 시점에 `config.model`을 불러오지 않는다(`default_deps()` 안에서만 지연 import).
- 구조화 출력은 `llm.with_structured_output(Schema, method="function_calling")`을 쓴다. 에이전트는 `create_agent(..., response_format=ToolStrategy(Schema))`를 쓴다. 로컬 mlx 서버에서 도구 호출 방식이 동작하는 것은 ch04 예제로 확인되어 있다.
- 산출물 경로: `outputs/<thread_id>/{research.json, script.json, scenes.json, scenes/NN.jpg, scenes/NN.mp3, scenes/NN.words.json, subtitles.srt, final.mp4, metadata.json}`. NN은 01부터 시작하는 두 자리 숫자다.
- 영상은 1080×1920, 30fps, 한국어 나레이션, 기본 음성 `ko-KR-SunHiNeural`, 자막 폰트 `/System/Library/Fonts/AppleSDGothicNeo.ttc`.
- 대본 분량은 공백을 제외한 250~350자다. 자동 재작성은 최대 3회(`MAX_REVISIONS = 3`). 장면은 5~8개를 목표로 프롬프트에 명시한다(강제하지 않음). 60초 초과는 경고만 한다.
- `IMAGE_SOURCE = mflux | pexels | placeholder`, 기본값 `mflux`. Pexels 도구는 `PEXELS_API_KEY`가 있을 때만 등록한다(2026-11-06 현재 무료 키 발급 중단).
- mflux는 FLUX.1-schnell 4bit, 576×1024로 생성한 뒤 1080×1920으로 맞춘다. 프로세스 안의 락으로 한 번에 하나만 실행한다. 실패하면 placeholder로 대체한다.
- 체크포인터는 실행 시 `SqliteSaver`(`outputs/checkpoints.sqlite`), 테스트 시 `InMemorySaver`를 쓴다.
- 상태에는 Pydantic 객체가 아니라 `model_dump()`한 dict만 넣는다(체크포인트 직렬화 안정성).
- 테스트는 네트워크·GPU·LLM 서버 없이 돌아가야 한다. 네트워크가 필요한 테스트는 `@pytest.mark.network`로 표시하고 기본 실행에서 제외한다.
- 작업 브랜치는 `feat/shorts-agent`다(main에 직접 커밋하지 않는다).

## Review Focus

1. **로컬 LLM이 구조화 출력에 실패**(잘못된 도구 호출, 스키마 불일치)하면 한 번 재시도한다. 그래도 실패하면 어느 노드에서 실패했는지 담긴 `NodeError`를 낸다. 조용히 빈 값으로 진행하지 않는다. → Task 2 `test_invoke_structured_retries_then_raises_node_error`
2. **사람 입력이 잘못된 경우**(빈 줄, 범위 밖 번호, 빈 수정본)에는 다시 묻는다. 그래프에 잘못된 값이 넘어가 죽지 않게 한다. → Task 5 `test_ask_topic_reprompts_on_invalid`, `test_ask_script_approval_reprompts`
3. **리서치 결과가 0건**(Tavily 실패, 결과 없음)이어도 사람이 주제를 직접 입력해 진행할 수 있다. 직접 입력한 주제의 검색도 0건이면 근거 없는 후보를 만들어 계속 진행한다. → Task 3 `test_single_mode_empty_falls_back`, Task 5 `test_ask_topic_without_candidates`
4. **스토리보드 이상**(인덱스 중복·누락, 빈 나레이션 장면)이 오면 빈 장면은 버리고 1..N으로 다시 번호를 매긴다. 장면이 0개면 `NodeError`를 낸다. → Task 4 `test_normalize_scenes_*`
5. **자막으로 쪼개기 어려운 나레이션**(최대 글자 수보다 긴 단어, 영어·숫자 혼합, TTS가 단어 경계를 주지 않는 경우)이 와도 자막이 빠지지 않는다. 긴 단어는 단독 자막이 되고, 단어 경계가 없으면 장면 전체가 자막 하나가 된다. → Task 6 `test_long_word_gets_own_cue`, `test_no_words_falls_back_to_whole_narration`

---

## 파일 구조

```
shorts_agent/
  __init__.py
  __main__.py          # CLI: 실행/재개, interrupt 응답 입력
  settings.py          # ShortsSettings (dataclass, from_env)
  state.py             # ShortsState, merge_scene_results, RESET_SCENE_RESULTS
  schemas.py           # Pydantic 구조화 출력 스키마
  files.py             # save_json, scene_path
  llm_utils.py         # invoke_structured, NodeError
  deps.py              # Deps dataclass, default_deps()
  graph.py             # build_graph, initial_state
  cli_prompts.py       # ask_topic, ask_script_approval
  nodes/
    __init__.py
    research.py        # build_research_agent, make_researcher_node
    human.py           # make_pick_topic_node, make_approve_script_node
    writing.py         # count_script_chars, make_writer_node, make_reviewer_node, route_after_review
    storyboard.py      # normalize_scenes, make_storyboarder_node, fan_out_scenes
    scene_worker.py    # make_visual_runner, make_scene_worker_node
    editor.py          # make_editor_node
    publisher.py       # normalize_metadata, make_publisher_node
  media/
    __init__.py
    subtitles.py       # Cue, words_to_cues, format_srt
    tts.py             # Word, TtsResult, parse_boundaries, synthesize_sync, audio_duration
    images.py          # make_placeholder_image, fit_to_vertical, build_mflux_command, generate_mflux_image, Pexels
    render.py          # render_caption_image, render_video
tests/
  fakes.py             # FakeLLM, FakeAgent, fake_tts, write_silence_wav
  test_*.py
```

---

### Task 1: 프로젝트 골격 · 설정 · 상태 · 스키마

**Files:**
- Modify: `pyproject.toml` (의존성, pytest 설정)
- Create: `.gitignore`, `shorts_agent/__init__.py`, `shorts_agent/settings.py`, `shorts_agent/state.py`, `shorts_agent/schemas.py`, `shorts_agent/files.py`, `tests/fakes.py`, `tests/test_state.py`

**Interfaces:**
- Produces:
  - `ShortsSettings` (frozen dataclass): `output_root: Path = BASE_DIR/"outputs"`, `image_source: str = "mflux"`, `pexels_api_key: str | None`, `tts_voice: str = "ko-KR-SunHiNeural"`, `font_path: str = "/System/Library/Fonts/AppleSDGothicNeo.ttc"`, `video_size: tuple[int,int] = (1080, 1920)`, `fps: int = 30`, `script_min_chars: int = 250`, `script_max_chars: int = 350`, `max_revisions: int = 3`, `target_max_seconds: float = 60.0`, `mflux_model: str = "schnell"`, `mflux_quantize: int = 4`, `mflux_steps: int = 4`, `mflux_size: tuple[int,int] = (576, 1024)`, `subtitle_max_chars: int = 14`. `ShortsSettings.from_env() -> ShortsSettings`는 `.env`의 `IMAGE_SOURCE`, `PEXELS_API_KEY`, `TTS_VOICE`, `MFLUX_MODEL`을 읽는다(없으면 기본값).
  - `state.py`: `RESET_SCENE_RESULTS = "__reset__"`, `merge_scene_results(left: list[dict] | None, right: list[dict] | str) -> list[dict]`, `ShortsState(TypedDict, total=False)`. 필드: `keyword: str`, `research_mode: Literal["explore","single"]`, `candidates: list[dict]`, `topic: dict`, `hook: str`, `script: str`, `review_passed: bool`, `review_feedback: str`, `revision_count: int`, `scenes: list[dict]`, `scene_results: Annotated[list[dict], merge_scene_results]`, `output_dir: str`, `srt_path: str`, `video_path: str`, `metadata: dict`.
  - `schemas.py` (Pydantic, 모든 필드에 한국어 `description`):
    - `Source(url: str, point: str)`, `TopicCandidate(title: str, summary: str, sources: list[Source] = [])`, `ResearchResult(candidates: list[TopicCandidate])`
    - `ScriptDraft(hook: str, script: str)`: `script`는 훅 문장으로 시작하는 전체 나레이션이다.
    - `Review(passed: bool, feedback: str)`
    - `Scene(index: int, narration: str, visual_query: str, image_prompt: str)`, `Storyboard(scenes: list[Scene])`
    - `VisualResult(image_path: str, source: Literal["mflux","pexels","placeholder"])`
    - `Metadata(title: str, description: str, hashtags: list[str])`
  - `files.py`: `save_json(output_dir: str | Path, name: str, data: dict | list) -> Path` (UTF-8, `ensure_ascii=False`, `indent=2`, 디렉터리 자동 생성), `scene_path(output_dir: str | Path, index: int, suffix: str) -> Path` → `<output_dir>/scenes/{index:02d}{suffix}` (scenes 디렉터리 자동 생성)
  - `tests/fakes.py`:
    - `FakeLLM(responses: list)`: `.with_structured_output(schema, **kw)`가 같은 객체를 돌려준다. `.invoke(messages)`는 `responses`에서 하나씩 꺼내 반환하고, 꺼낸 값이 Exception이면 raise한다. `.calls`에 받은 messages를 쌓는다.
    - `FakeAgent(responses: list)`: `.invoke(input) -> {"structured_response": next}`, `.calls`
    - `write_silence_wav(path: Path, seconds: float) -> Path` (표준 라이브러리 `wave`, 16kHz mono)
    - `fake_tts(text: str, out_path: Path) -> TtsResult`는 Task 7에서 추가한다.

- [ ] **Step 0: 브랜치 생성** — `git checkout -b feat/shorts-agent`
- [ ] **Step 1: 의존성 추가**

```bash
uv add langgraph-checkpoint-sqlite edge-tts "moviepy>=2" httpx
uv add --dev pytest
```

`pyproject.toml`에 다음을 추가한다.

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["network: 인터넷이 필요한 테스트"]
addopts = "-m 'not network'"
```

`.gitignore`: `outputs/`, `.pytest_cache/`

- [ ] **Step 2: 실패하는 테스트 작성** (`tests/test_state.py`)

```python
def test_merge_appends():
    assert merge_scene_results([{"index": 1}], [{"index": 2}]) == [{"index": 1}, {"index": 2}]

def test_merge_handles_none_left():
    assert merge_scene_results(None, [{"index": 1}]) == [{"index": 1}]

def test_merge_reset_clears():
    assert merge_scene_results([{"index": 1}], RESET_SCENE_RESULTS) == []

def test_settings_defaults():
    s = ShortsSettings()
    assert (s.image_source, s.max_revisions, s.video_size) == ("mflux", 3, (1080, 1920))
    assert (s.script_min_chars, s.script_max_chars) == (250, 350)

def test_scene_path_zero_pads(tmp_path):
    assert scene_path(tmp_path, 3, ".jpg") == tmp_path / "scenes" / "03.jpg"
    assert (tmp_path / "scenes").is_dir()

def test_save_json_keeps_korean(tmp_path):
    p = save_json(tmp_path, "x.json", {"t": "한국어"})
    assert "한국어" in p.read_text(encoding="utf-8")
```

- [ ] **Step 3: 실패 확인** — `uv run pytest tests/test_state.py -v` → ImportError로 FAIL
- [ ] **Step 4: Interfaces대로 구현**
- [ ] **Step 5: 통과 확인** — `uv run pytest -v` → 전부 PASS
- [ ] **Step 6: 커밋** — `git add pyproject.toml uv.lock .gitignore shorts_agent tests && git commit -m "feat(shorts): 프로젝트 골격, 설정, 상태, 스키마"`

---

### Task 2: 구조화 출력 헬퍼 · 의존성 묶음

**Files:**
- Create: `shorts_agent/llm_utils.py`, `shorts_agent/deps.py`, `tests/test_llm_utils.py`

**Interfaces:**
- Consumes: `ShortsSettings`(Task 1)
- Produces:
  - `class NodeError(RuntimeError)`
  - `invoke_structured(llm, schema: type[T], system: str, user: str, node: str, retries: int = 1) -> T`: `llm.with_structured_output(schema, method="function_calling").invoke([SystemMessage(system), HumanMessage(user)])`. 예외가 나거나 결과가 `schema` 인스턴스가 아니면 `retries`번 재시도한다. 그래도 실패하면 `NodeError(f"[{node}] 구조화 출력 실패: {원인}")`.
  - `@dataclass Deps`: `settings: ShortsSettings`, `llm: Any`, `research_agent: Any`, `visual_runner: Callable[[dict, Path], tuple[Path, str]]`, `tts: Callable[[str, Path], TtsResult]`
  - `default_deps(settings: ShortsSettings | None = None) -> Deps`: 여기서만 `config.model.load_llm`, `build_research_agent`, `make_visual_runner`, `synthesize_sync`를 import한다. 이 함수는 Task 9에서 완성한다. 이 Task에서는 시그니처와 llm·research_agent 부분만 만들고, 나머지는 Task 7·9에서 채운다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_invoke_structured_returns_schema_instance():
    llm = FakeLLM([Review(passed=True, feedback="")])
    assert invoke_structured(llm, Review, "s", "u", node="reviewer").passed is True

def test_invoke_structured_retries_once():
    llm = FakeLLM([ValueError("bad tool call"), Review(passed=False, feedback="x")])
    assert invoke_structured(llm, Review, "s", "u", node="reviewer").feedback == "x"
    assert len(llm.calls) == 2

def test_invoke_structured_retries_then_raises_node_error():
    llm = FakeLLM([ValueError("a"), ValueError("b")])
    with pytest.raises(NodeError, match=r"\[reviewer\]"):
        invoke_structured(llm, Review, "s", "u", node="reviewer")
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_llm_utils.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 구조화 출력 헬퍼와 Deps`

---

### Task 3: 리서처 노드 · 주제 선택(interrupt)

**Files:**
- Create: `shorts_agent/nodes/__init__.py`, `shorts_agent/nodes/research.py`, `shorts_agent/nodes/human.py`(pick_topic 부분), `tests/test_research.py`, `tests/test_pick_topic.py`

**Interfaces:**
- Consumes: `Deps`, `ResearchResult`, `TopicCandidate`, `save_json`
- Produces:
  - `build_research_agent(llm) -> agent`: `create_agent(model=llm, tools=[TavilySearch(max_results=5)], system_prompt=..., response_format=ToolStrategy(ResearchResult))`
  - `make_researcher_node(deps) -> Callable[[ShortsState], dict]`
    - explore 모드: `keyword`로 쇼츠 주제 후보 3개를 찾게 하고 앞의 3개만 남긴다.
    - single 모드: `keyword` 자체를 주제로 근거를 찾게 하고 앞의 1개만 남긴다. 결과가 0건이면 `TopicCandidate(title=keyword, summary="", sources=[])`로 대체한다.
    - 반환: `{"candidates": [dict...]}`. `research.json`에 저장한다.
  - `make_pick_topic_node(deps) -> Callable[[ShortsState], Command[Literal["writer","researcher"]]]`
    - `interrupt({"type": "pick_topic", "candidates": state["candidates"]})`
    - resume이 `{"choice": i}`이면 `Command(goto="writer", update={"topic": candidates[i], "review_feedback": "", "revision_count": 0})`
    - resume이 `{"custom_topic": t}`이면 `Command(goto="researcher", update={"keyword": t, "research_mode": "single"})`
    - 범위 밖 `choice`는 `ValueError`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def _cand(t): return TopicCandidate(title=t, summary="s", sources=[Source(url="u", point="p")])

def test_explore_keeps_three(tmp_path, deps_factory):
    deps = deps_factory(research_agent=FakeAgent([ResearchResult(candidates=[_cand(c) for c in "abcd"])]))
    out = make_researcher_node(deps)({"keyword": "AI", "research_mode": "explore", "output_dir": str(tmp_path)})
    assert [c["title"] for c in out["candidates"]] == ["a", "b", "c"]
    assert (tmp_path / "research.json").exists()

def test_single_mode_empty_falls_back(tmp_path, deps_factory):
    deps = deps_factory(research_agent=FakeAgent([ResearchResult(candidates=[])]))
    out = make_researcher_node(deps)({"keyword": "양자컴퓨터", "research_mode": "single", "output_dir": str(tmp_path)})
    assert out["candidates"] == [{"title": "양자컴퓨터", "summary": "", "sources": []}]
```

`test_pick_topic.py`: `InMemorySaver`를 붙인 2-노드 미니 그래프(pick_topic → writer/researcher 더미)로 다음을 검증한다.
- `{"choice": 1}`로 재개하면 `topic["title"] == "b"`이고 writer에 도달한다.
- `{"custom_topic": "새 주제"}`로 재개하면 researcher에 도달하고 `keyword == "새 주제"`, `research_mode == "single"`이다.

`deps_factory`는 `tests/conftest.py`에 fixture로 둔다. `Deps(settings=ShortsSettings(image_source="placeholder", output_root=tmp_path), llm=FakeLLM([]), research_agent=FakeAgent([]), visual_runner=..., tts=...)`를 만들되 키워드 인자로 덮어쓸 수 있게 한다.

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현** — pick_topic은 `add_node(..., destinations=("writer","researcher"))`로 등록할 수 있게 반환 타입 힌트를 단다.
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 리서처 노드와 주제 선택 interrupt`

---

### Task 4: 작가 · 검수자 루프 · 대본 승인 · 스토리보더

**Files:**
- Create: `shorts_agent/nodes/writing.py`, `shorts_agent/nodes/storyboard.py`, `tests/test_writing.py`, `tests/test_storyboard.py`, `tests/test_approve_script.py`
- Modify: `shorts_agent/nodes/human.py` (approve_script 추가)

**Interfaces:**
- Consumes: `invoke_structured`, `ScriptDraft`, `Review`, `Storyboard`, `Scene`, `RESET_SCENE_RESULTS`, `save_json`
- Produces:
  - `count_script_chars(script: str) -> int` = `len("".join(script.split()))`
  - `make_writer_node(deps)`: 주제·근거를 넣고, `review_feedback`이 있으면 이전 대본과 피드백도 함께 넣는다. 반환은 `{"hook", "script", "revision_count"}`. `revision_count`는 `review_feedback`이 비어 있지 않을 때만 +1한다.
  - `make_reviewer_node(deps)`:
    - 분량이 범위를 벗어나면 LLM을 호출하지 않고 `{"review_passed": False, "review_feedback": f"분량이 {n}자입니다. {min}~{max}자로 맞춰 주세요."}`를 반환한다.
    - 범위 안이면 LLM으로 훅 강도와 근거 일치를 점검한다. 주제에 sources가 없으면 프롬프트에 "사실 확인 생략"을 명시한다.
    - 합격이면 `review_feedback = ""`
    - 매번 `script.json`(hook, script, revision_count, review)을 저장한다.
  - `route_after_review(state, max_revisions: int = 3) -> Literal["writer","approve_script"]`: 합격이면 approve_script, `revision_count >= max_revisions`이면 approve_script, 그 외에는 writer
  - `make_approve_script_node(deps) -> Callable[..., Command[Literal["storyboarder","writer"]]]`
    - `interrupt({"type": "approve_script", "hook", "script", "review_passed", "review_feedback"})`
    - `{"action":"approve"}`이면 storyboarder로 간다.
    - `{"action":"edit","script":s}`이면 storyboarder로 가고 `update={"script": s}`. `script.json`도 다시 저장한다.
    - `{"action":"rewrite","comment":c}`이면 writer로 가고 `update={"review_feedback": c, "revision_count": 0}`
  - `normalize_scenes(scenes: list[Scene]) -> list[dict]`: `narration.strip()`이 빈 장면은 버리고 index를 1..N으로 다시 매긴다. 0개면 `NodeError("[storyboarder] 장면이 없습니다")`
  - `make_storyboarder_node(deps)`: 프롬프트에 "장면 5~8개, visual_query와 image_prompt는 영어"를 명시한다. 반환은 `{"scenes": [...], "scene_results": RESET_SCENE_RESULTS}`. `scenes.json`에 저장한다.
  - `fan_out_scenes(state) -> list[Send]` = `[Send("scene_worker", {"scene": s, "output_dir": state["output_dir"]}) for s in state["scenes"]]`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_count_ignores_whitespace():
    assert count_script_chars("가 나\n다") == 3

def test_reviewer_rejects_short_without_llm(deps_factory, tmp_path):
    llm = FakeLLM([])
    deps = deps_factory(llm=llm)
    out = make_reviewer_node(deps)({"script": "짧다", "hook": "h", "topic": {"sources": []}, "output_dir": str(tmp_path), "revision_count": 0})
    assert out["review_passed"] is False and "2자" in out["review_feedback"]
    assert llm.calls == []

def test_writer_increments_only_with_feedback(deps_factory, tmp_path):
    llm = FakeLLM([ScriptDraft(hook="h", script="s"), ScriptDraft(hook="h", script="s2")])
    node = make_writer_node(deps_factory(llm=llm))
    base = {"topic": {"title": "t", "summary": "", "sources": []}, "output_dir": str(tmp_path)}
    assert node({**base, "review_feedback": "", "revision_count": 0})["revision_count"] == 0
    assert node({**base, "review_feedback": "더 짧게", "revision_count": 0, "script": "s"})["revision_count"] == 1

@pytest.mark.parametrize("passed,count,expected", [
    (True, 0, "approve_script"), (False, 0, "writer"), (False, 2, "writer"), (False, 3, "approve_script")])
def test_route_after_review(passed, count, expected):
    assert route_after_review({"review_passed": passed, "revision_count": count}) == expected

def test_normalize_scenes_reindexes_and_drops_empty():
    scenes = [Scene(index=5, narration="a", visual_query="q", image_prompt="p"),
              Scene(index=5, narration="  ", visual_query="q", image_prompt="p"),
              Scene(index=9, narration="b", visual_query="q", image_prompt="p")]
    assert [(s["index"], s["narration"]) for s in normalize_scenes(scenes)] == [(1, "a"), (2, "b")]

def test_normalize_scenes_empty_raises():
    with pytest.raises(NodeError):
        normalize_scenes([])

def test_storyboarder_resets_scene_results(deps_factory, tmp_path): ...
    # 반환값 out["scene_results"] == RESET_SCENE_RESULTS 이고, scenes.json이 생성된다

def test_fan_out_one_send_per_scene():
    sends = fan_out_scenes({"scenes": [{"index": 1}, {"index": 2}], "output_dir": "o"})
    assert [s.arg["scene"]["index"] for s in sends] == [1, 2] and sends[0].node == "scene_worker"
```

`test_approve_script.py`는 Task 3과 같은 미니 그래프 방식으로 approve / edit / rewrite 세 가지 분기의 도착 노드와 상태 변화를 검증한다.

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 작가·검수 루프, 대본 승인, 스토리보더`

---

### Task 5: 기획 그래프 연결 · CLI (1단계 완성)

**Files:**
- Create: `shorts_agent/graph.py`, `shorts_agent/cli_prompts.py`, `shorts_agent/__main__.py`, `tests/test_cli_prompts.py`, `tests/test_graph_planning.py`

**Interfaces:**
- Consumes: Task 3·4의 모든 노드 팩토리, `route_after_review`
- Produces:
  - `initial_state(keyword: str, output_dir: Path) -> dict` = `{"keyword", "research_mode": "explore", "revision_count": 0, "review_feedback": "", "output_dir": str(output_dir)}`
  - `build_graph(deps: Deps, checkpointer)`: 컴파일된 그래프를 반환한다. 이 Task에서는 `storyboarder → END`까지만 연결한다. Task 11에서 제작 단계를 잇는다.
  - `ask_topic(payload: dict, input_fn=input, print_fn=print) -> dict`
    - 후보를 번호(1부터)와 함께 출력한다.
    - 범위 안의 숫자를 입력하면 `{"choice": n-1}`
    - 숫자가 아닌, 비어 있지 않은 텍스트를 입력하면 `{"custom_topic": text}`
    - 빈 줄이나 범위 밖 숫자는 다시 묻는다.
    - 후보가 없으면 "주제를 직접 입력하세요"만 묻는다.
  - `ask_script_approval(payload, input_fn=input, print_fn=print) -> dict`
    - 메뉴는 `1 승인 / 2 재작성 요청 / 3 직접 수정`
    - 2를 고르면 코멘트를 묻고, 빈 코멘트면 다시 묻는다.
    - 3을 고르면 여러 줄을 입력받는다. `.` 한 줄을 입력하면 종료하고, 빈 수정본이면 다시 묻는다.
    - 그 외 입력은 다시 묻는다.
  - `__main__.py`: `python -m shorts_agent "키워드" [--thread-id ID] [--resume ID] [--show-graph]`
    - `SqliteSaver.from_conn_string(str(output_root/"checkpoints.sqlite"))`를 컨텍스트 매니저로 연다.
    - 실행 루프: `result = graph.invoke(...)`를 한 뒤, `"__interrupt__"`가 있는 동안 `payload["type"]`에 맞는 ask 함수를 호출하고 `Command(resume=...)`로 다시 invoke한다.
    - `--resume`: `graph.get_state(config)`에 남은 interrupt가 있으면 그 payload로 응답 루프를 시작한다. 없으면 "이미 완료된 실행입니다"를 출력한다.
    - thread_id 기본값은 `datetime.now().strftime("%Y%m%d-%H%M%S")`
    - `--show-graph`이면 `graph.get_graph().draw_ascii()`를 출력한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def _inputs(*xs):
    it = iter(xs); return lambda _="": next(it)

def test_ask_topic_choice():
    assert ask_topic({"candidates": [{"title": "a", "summary": ""}]}, _inputs("1"), print) == {"choice": 0}

def test_ask_topic_reprompts_on_invalid():
    assert ask_topic({"candidates": [{"title": "a", "summary": ""}]}, _inputs("", "5", "1"), lambda *a: None) == {"choice": 0}

def test_ask_topic_custom_text():
    assert ask_topic({"candidates": [{"title": "a", "summary": ""}]}, _inputs("양자컴퓨터"), lambda *a: None) == {"custom_topic": "양자컴퓨터"}

def test_ask_topic_without_candidates():
    assert ask_topic({"candidates": []}, _inputs("", "새 주제"), lambda *a: None) == {"custom_topic": "새 주제"}

def test_ask_script_approval_reprompts():
    p = {"hook": "h", "script": "s", "review_passed": True, "review_feedback": ""}
    assert ask_script_approval(p, _inputs("9", "2", "", "더 재밌게"), lambda *a: None) == {"action": "rewrite", "comment": "더 재밌게"}
    assert ask_script_approval(p, _inputs("3", ".", "3", "새 대본", "."), lambda *a: None) == {"action": "edit", "script": "새 대본"}
```

`test_graph_planning.py`는 가짜 LLM과 에이전트로 전체 기획 흐름을 끝까지 돌린다.
- 준비: FakeAgent 후보 3개, FakeLLM 응답 순서 [짧은 대본 → (분량 불합격, LLM 호출 없음) → 정상 대본 → Review 합격 → Storyboard 2장면]. 정상 대본은 공백 제외 300자다.
- 실행: `InMemorySaver`를 붙여 `invoke` → `Command(resume={"choice": 0})` → `Command(resume={"action": "approve"})` 순서로 진행한다.
- 검증: 최종 상태의 `revision_count == 1`, `len(scenes) == 2`이고, `research.json`, `script.json`, `scenes.json`이 모두 존재한다.

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `uv run pytest -v` 전체 PASS
- [ ] **Step 5: 실제 실행 확인 (수동)**
  - `./model_run.sh`를 띄운 뒤 `uv run python -m shorts_agent "AI 뉴스" --show-graph`를 실행한다.
  - 주제를 고르고 대본을 승인하면 `outputs/<id>/scenes.json`이 생성되어야 한다.
  - 중간에 Ctrl+C로 종료한 뒤 `--resume <id>`로 이어지는지도 확인한다.
- [ ] **Step 6: 커밋** — `feat(shorts): 기획 파이프라인 그래프와 CLI`

---

### Task 6: 자막 모듈

**Files:**
- Create: `shorts_agent/media/__init__.py`, `shorts_agent/media/subtitles.py`, `tests/test_subtitles.py`

**Interfaces:**
- Consumes: `Word(text: str, start: float, end: float)` — Task 7의 `media/tts.py`에서 정의한다. 이 Task에서 `media/tts.py`에 `Word` dataclass만 먼저 만든다.
- Produces:
  - `@dataclass Cue(start: float, end: float, text: str)`
  - `words_to_cues(words: list[Word], narration: str, offset: float, duration: float, max_chars: int = 14) -> list[Cue]`
    - 연속된 단어를 공백으로 이어 붙이다가, 다음 단어를 붙이면 `max_chars`를 넘을 때 끊는다.
    - 각 자막의 시간은 첫 단어 start + offset부터 마지막 단어 end + offset까지다.
    - `max_chars`보다 긴 단어는 단독 자막이 된다.
    - `words`가 비어 있으면 `[Cue(offset, offset+duration, narration.strip())]`
  - `format_srt(cues: list[Cue]) -> str`: 번호는 1부터, 시간 형식은 `HH:MM:SS,mmm --> HH:MM:SS,mmm`, 자막 사이에 빈 줄을 둔다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
W = lambda t, s, e: Word(t, s, e)

def test_groups_by_max_chars():
    ws = [W("안녕하세요", 0.0, 0.5), W("오늘은", 0.5, 0.9), W("인공지능", 0.9, 1.4)]
    cues = words_to_cues(ws, "", offset=10.0, duration=2.0, max_chars=10)
    assert [(c.text, c.start, c.end) for c in cues] == [("안녕하세요 오늘은", 10.0, 10.9), ("인공지능", 10.9, 11.4)]

def test_long_word_gets_own_cue():
    cues = words_to_cues([W("초장문단어테스트입니다아아", 0, 1), W("끝", 1, 1.2)], "", 0, 2, max_chars=5)
    assert [c.text for c in cues] == ["초장문단어테스트입니다아아", "끝"]

def test_no_words_falls_back_to_whole_narration():
    assert words_to_cues([], " 전체 문장 ", offset=3.0, duration=4.5) == [Cue(3.0, 7.5, "전체 문장")]

def test_format_srt():
    assert format_srt([Cue(0, 1.5, "가"), Cue(61.25, 62, "나")]) == (
        "1\n00:00:00,000 --> 00:00:01,500\n가\n\n2\n00:01:01,250 --> 00:01:02,000\n나\n")
```

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 자막 묶기와 srt 출력`

---

### Task 7: TTS 모듈

**Files:**
- Modify: `shorts_agent/media/tts.py`, `tests/fakes.py` (`fake_tts` 추가)
- Create: `tests/test_tts.py`

**Interfaces:**
- Produces:
  - `@dataclass TtsResult(audio_path: Path, words: list[Word], duration: float)`
  - `parse_boundaries(events: list[dict]) -> list[Word]`: `type == "WordBoundary"`인 이벤트만 사용한다. offset과 duration은 100ns 단위이므로 1e7로 나눠 초로 바꾼다. start=offset, end=offset+duration.
  - `synthesize_sync(text: str, out_path: Path, voice: str = "ko-KR-SunHiNeural", rate: str = "+0%") -> TtsResult`
    - `edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")`의 `stream()`을 `asyncio.run`으로 돈다. audio 청크는 파일에 쓰고, 경계 이벤트는 모은다.
    - 결과로 `.words.json`을 `out_path.with_suffix(".words.json")`에 저장한다. 예: `scenes/01.mp3` → `scenes/01.words.json`
    - edge-tts 7.x부터는 단어 경계 이벤트를 받으려면 `boundary="WordBoundary"`를 명시해야 한다. 설치된 버전의 시그니처를 먼저 확인한다.
  - `audio_duration(path: Path) -> float`: `moviepy.AudioFileClip(path).duration`을 쓰고 clip을 닫는다.
  - `tests/fakes.py`의 `fake_tts(text, out_path) -> TtsResult`: `out_path.with_suffix(".wav")`에 2.0초 무음을 쓴다. 단어는 공백으로 나눈 뒤 2.0초를 균등하게 나눠 배정한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_parse_boundaries_converts_ticks():
    ev = [{"type": "WordBoundary", "offset": 1_000_000, "duration": 5_000_000, "text": "안녕"},
          {"type": "SentenceBoundary", "offset": 0, "duration": 1, "text": "x"}]
    assert parse_boundaries(ev) == [Word("안녕", 0.1, 0.6)]

def test_audio_duration_of_silence(tmp_path):
    assert audio_duration(write_silence_wav(tmp_path / "a.wav", 1.5)) == pytest.approx(1.5, abs=0.05)

@pytest.mark.network
def test_synthesize_korean(tmp_path):
    r = synthesize_sync("안녕하세요 반갑습니다", tmp_path / "01.mp3")
    assert r.audio_path.stat().st_size > 0 and len(r.words) >= 2 and r.duration > 0.5
    assert (tmp_path / "01.words.json").exists()
```

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `uv run pytest tests/test_tts.py -v`가 PASS(network 제외). `uv run pytest -m network tests/test_tts.py -v`도 PASS. 이 테스트는 인터넷이 필요하다.
- [ ] **Step 5: 커밋** — `feat(shorts): edge-tts 음성 합성과 단어 타이밍`

---

### Task 8: 이미지 모듈 (placeholder · mflux · Pexels)

**Files:**
- Create: `shorts_agent/media/images.py`, `tests/test_images.py`

**Interfaces:**
- Produces:
  - `make_placeholder_image(text: str, out_path: Path, size=(1080, 1920), font_path=...) -> Path`: 단색 배경에 줄바꿈한 텍스트를 넣어 JPEG로 저장한다.
  - `fit_to_vertical(src: Path, out_path: Path, size=(1080, 1920)) -> Path`: 비율을 유지한 채 화면을 꽉 채우도록 확대하고, 가운데를 잘라 JPEG로 저장한다.
  - `class MfluxUnavailable(RuntimeError)`
  - `build_mflux_command(prompt: str, out_path: Path, model: str, quantize: int, steps: int, width: int, height: int) -> list[str]`: `["mflux-generate", "--model", model, "--quantize", str(quantize), "--steps", str(steps), "--width", ..., "--height", ..., "--prompt", prompt, "--output", str(out_path)]`
    - mflux 버전마다 플래그가 다를 수 있다. 구현 전에 `mflux-generate --help`로 확인하고, 다르면 이 함수와 테스트를 함께 맞춘다.
  - `generate_mflux_image(prompt: str, out_path: Path, settings: ShortsSettings) -> Path`
    - 모듈 전역 `_MFLUX_LOCK = threading.Lock()` 안에서 실행한다.
    - `shutil.which("mflux-generate")`가 None이거나 종료 코드가 0이 아니면 `MfluxUnavailable`
    - 생성한 PNG는 `fit_to_vertical`로 `out_path`(jpg)에 맞춘다.
  - `@dataclass PexelsPhoto(id: int, alt: str, width: int, height: int, url: str)`
  - `parse_pexels_response(data: dict) -> list[PexelsPhoto]`: `width >= height`인 사진은 제외한다. url에는 `src["portrait"]`를 쓴다.
  - `search_pexels(query: str, api_key: str, per_page: int = 5) -> list[PexelsPhoto]`: httpx로 `GET https://api.pexels.com/v1/search?orientation=portrait`, `Authorization: <key>` 헤더
  - `download_to_vertical(url: str, out_path: Path) -> Path`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_placeholder_size(tmp_path):
    p = make_placeholder_image("장면 설명", tmp_path / "01.jpg", size=(108, 192))
    assert Image.open(p).size == (108, 192)

def test_fit_to_vertical_crops_landscape(tmp_path):
    Image.new("RGB", (400, 100), "red").save(tmp_path / "w.png")
    assert Image.open(fit_to_vertical(tmp_path / "w.png", tmp_path / "o.jpg", (90, 160))).size == (90, 160)

def test_build_mflux_command():
    cmd = build_mflux_command("a cat", Path("o.png"), "schnell", 4, 4, 576, 1024)
    assert cmd[0] == "mflux-generate" and cmd[cmd.index("--width") + 1] == "576" and cmd[cmd.index("--prompt") + 1] == "a cat"

def test_generate_raises_when_cli_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(MfluxUnavailable):
        generate_mflux_image("x", tmp_path / "01.jpg", ShortsSettings())

def test_parse_pexels_filters_landscape():
    data = {"photos": [
        {"id": 1, "alt": "cat", "width": 1000, "height": 2000, "src": {"portrait": "u1"}},
        {"id": 2, "alt": "dog", "width": 2000, "height": 1000, "src": {"portrait": "u2"}}]}
    assert parse_pexels_response(data) == [PexelsPhoto(1, "cat", 1000, 2000, "u1")]
```

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: mflux 설치 확인 (수동, 선택)**
  - `uv tool install mflux`로 별도 환경에 설치한다. 프로젝트의 mlx-lm과 의존성이 충돌하지 않게 하려는 것이다.
  - `mflux-generate --help`로 플래그를 대조한다.
  - 첫 실행 때 FLUX.1-schnell 가중치(수십 GB)를 내려받는다는 점을 사용자에게 알린다. 미리 양자화된 저장소를 쓰려면 `.env`에 `MFLUX_MODEL`을 설정한다.
- [ ] **Step 6: 커밋** — `feat(shorts): 이미지 소스(placeholder, mflux, Pexels)`

---

### Task 9: 비주얼 에이전트 · 장면 워커 노드

**Files:**
- Create: `shorts_agent/nodes/scene_worker.py`, `tests/test_scene_worker.py`
- Modify: `shorts_agent/deps.py` (`default_deps` 완성)

**Interfaces:**
- Consumes: Task 8 이미지 함수, `TtsResult`, `scene_path`, `VisualResult`, `save_json`
- Produces:
  - `build_image_tools(scene: dict, output_dir: Path, settings: ShortsSettings) -> list[BaseTool]`. 이미지 저장 경로는 `scene_path(..., index, ".jpg")`다.
    - 항상 등록: `generate_image(prompt: str) -> str`(성공 시 경로, 실패 시 `"ERROR: ..."` 문자열), `make_placeholder() -> str`
    - `settings.pexels_api_key`가 있을 때만 등록: `search_stock_photos(query: str) -> str`(후보 JSON: id, alt, size), `use_stock_photo(photo_id: int) -> str`
  - `make_visual_runner(settings, agent_factory: Callable[[list[BaseTool]], Any]) -> Callable[[dict, Path], tuple[Path, str]]` (두 번째 인자는 `output_dir`)
    - `image_source == "placeholder"`이면 에이전트 없이 바로 placeholder를 만들어 `(path, "placeholder")`를 반환한다.
    - 그 외에는 에이전트를 실행해 `VisualResult`를 받는다. 예외가 나면 placeholder로 대체한다.
  - `default_agent_factory(llm)`는 `create_agent(model=llm, tools=tools, system_prompt=..., response_format=ToolStrategy(VisualResult))`
    - 시스템 프롬프트: 장면 설명을 보고, 스톡 도구가 있으면 alt가 맞는 후보를 고른다. 없으면 `image_prompt`를 다듬어 `generate_image`를 호출한다. `ERROR`가 오면 `make_placeholder`를 호출한다.
  - `make_scene_worker_node(deps) -> Callable[[dict], dict]`
    - 입력: Send payload `{"scene", "output_dir"}`
    - 처리:
      1. `visual_runner`로 이미지를 받는다. 반환된 경로가 실제로 없으면 placeholder를 만든다.
      2. `deps.tts(narration, scene_path(..., ".mp3"))`로 음성을 만든다.
      3. 결과의 `duration`이 0 이하이면 `audio_duration`으로 다시 잰다.
    - 반환: `{"scene_results": [{"index", "narration", "image_path", "image_source", "audio_path", "words": [asdict(w)...], "duration"}]}`
  - `default_deps()` 완성:
    - `visual_runner = make_visual_runner(settings, default_agent_factory(llm))`
    - `tts = partial(synthesize_sync, voice=settings.tts_voice)`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
SCENE = {"index": 1, "narration": "고양이가 잔다", "visual_query": "cat", "image_prompt": "a sleeping cat"}

def test_tools_without_pexels_key(tmp_path):
    names = {t.name for t in build_image_tools(SCENE, tmp_path, ShortsSettings(pexels_api_key=None))}
    assert names == {"generate_image", "make_placeholder"}

def test_tools_with_pexels_key(tmp_path):
    names = {t.name for t in build_image_tools(SCENE, tmp_path, ShortsSettings(pexels_api_key="k"))}
    assert {"search_stock_photos", "use_stock_photo"} <= names

def test_placeholder_mode_skips_agent(tmp_path):
    runner = make_visual_runner(ShortsSettings(image_source="placeholder"), agent_factory=lambda tools: pytest.fail("agent called"))
    path, source = runner(SCENE, tmp_path)
    assert source == "placeholder" and path.exists()

def test_agent_error_falls_back_to_placeholder(tmp_path):
    runner = make_visual_runner(ShortsSettings(), agent_factory=lambda tools: FakeAgent([RuntimeError("boom")]))
    assert runner(SCENE, tmp_path)[1] == "placeholder"

def test_scene_worker_missing_image_falls_back(tmp_path, deps_factory):
    deps = deps_factory(visual_runner=lambda s, d: (d / "nope.jpg", "mflux"), tts=fake_tts)
    r = make_scene_worker_node(deps)({"scene": SCENE, "output_dir": str(tmp_path)})["scene_results"][0]
    assert r["image_source"] == "placeholder" and Path(r["image_path"]).exists()
    assert r["duration"] == pytest.approx(2.0) and r["words"][0]["text"] == "고양이가"
```

`FakeAgent`가 예외를 응답으로 받으면 raise하도록 Task 1의 fakes를 확장한다(FakeLLM과 같은 규칙).

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 비주얼 에이전트와 장면 워커`

---

### Task 10: 렌더러 · 편집자 노드

**Files:**
- Create: `shorts_agent/media/render.py`, `shorts_agent/nodes/editor.py`, `tests/test_render.py`, `tests/test_editor.py`

**Interfaces:**
- Consumes: `Cue`, `words_to_cues`, `format_srt`, `Word`, scene_results dict(Task 9)
- Produces:
  - `render_caption_image(text: str, width: int, font_path: str, font_size: int) -> PIL.Image.Image`: RGBA 이미지에 흰 글씨와 검은 외곽선(stroke)을 넣는다. 폭 안에서 줄바꿈한다.
  - `render_video(scenes: list[dict], cues: list[Cue], out_path: Path, size=(1080, 1920), fps=30, font_path=...) -> Path`
    - moviepy 2.x를 쓴다.
    - 장면마다 `ImageClip(image).with_duration(duration)`을 만들고, 1.0→1.08배로 서서히 확대한다(Ken Burns). 장면들은 이어 붙인다.
    - 오디오는 장면 음성을 순서대로 이어 붙인다.
    - 자막마다 `render_caption_image`로 만든 클립을 `start`~`end` 동안 화면 높이의 75% 지점에 가운데 정렬로 겹친다.
    - 인코딩은 `libx264` / `aac`로 한다.
    - 자막 렌더링은 moviepy `TextClip` 대신 Pillow를 쓴다. 한글 폰트 처리를 안정적으로 하기 위한 설계자 결정이다.
  - `make_editor_node(deps)`
    1. `scene_results`를 index로 정렬하고, duration을 누적해 장면별 offset을 구한다.
    2. 장면마다 `words_to_cues(..., max_chars=settings.subtitle_max_chars)`로 자막을 만든다.
    3. `subtitles.srt`를 저장하고, `final.mp4`를 렌더링한다.
    3-1. `render_video(scenes, cues, out_path, size=settings.video_size, fps=settings.fps, font_path=settings.font_path)` — 앞의 세 인자는 위치 인자로 넘긴다(테스트가 `a[2]`로 out_path를 받음).
    4. 총 길이가 `target_max_seconds`를 넘으면 `logging.getLogger("shorts_agent").warning(...)`을 남긴다.
    - 반환: `{"srt_path", "video_path"}`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def _scene(tmp_path, i, secs):
    img = make_placeholder_image(f"장면{i}", tmp_path / f"{i:02d}.jpg", size=(108, 192))
    wav = write_silence_wav(tmp_path / f"{i:02d}.wav", secs)
    return {"index": i, "narration": f"장면 {i}", "image_path": str(img), "image_source": "placeholder",
            "audio_path": str(wav), "words": [], "duration": secs}

def test_render_small_video(tmp_path):
    out = render_video([_scene(tmp_path, 1, 1.0), _scene(tmp_path, 2, 1.0)], [Cue(0, 1, "가나다")],
                       tmp_path / "final.mp4", size=(108, 192), fps=10)
    clip = VideoFileClip(str(out))
    assert clip.size == [108, 192] and clip.duration == pytest.approx(2.0, abs=0.2)
    clip.close()

def test_editor_sorts_and_writes_srt(tmp_path, deps_factory, monkeypatch):
    monkeypatch.setattr("shorts_agent.nodes.editor.render_video", lambda *a, **k: a[2])  # 렌더링 생략
    state = {"output_dir": str(tmp_path), "scene_results": [_scene(tmp_path, 2, 1.0), _scene(tmp_path, 1, 1.5)]}
    out = make_editor_node(deps_factory())(state)
    srt = Path(out["srt_path"]).read_text(encoding="utf-8")
    assert srt.index("장면 1") < srt.index("장면 2") and "00:00:01,500 --> 00:00:02,500" in srt

def test_editor_warns_over_target(tmp_path, deps_factory, monkeypatch, caplog):
    # 61초짜리 장면 하나 → "shorts_agent" 로거에 WARNING이 남는다
```

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(shorts): 자막 합성 렌더러와 편집자 노드`

---

### Task 11: 퍼블리셔 · 전체 그래프 연결 (2·3단계 완성)

**Files:**
- Create: `shorts_agent/nodes/publisher.py`, `tests/test_publisher.py`, `tests/test_graph_full.py`
- Modify: `shorts_agent/graph.py` (제작 단계 연결), `shorts_agent/__main__.py` (완료 시 결과 경로 출력)

**Interfaces:**
- Consumes: 모든 노드, `fan_out_scenes`
- Produces:
  - `normalize_metadata(m: Metadata) -> dict`: 제목을 100자로 자른다. 해시태그는 `#`을 떼고 앞뒤 공백을 지운 뒤, 중복을 제거하고 순서를 유지한다. `shorts`가 없으면 맨 앞에 넣는다. 마지막에 모두 `#`을 붙인다.
  - `make_publisher_node(deps)`: 대본과 주제로 `Metadata`를 만들고 `normalize_metadata`를 거친다. `metadata.json`에 저장하고 `{"metadata": ...}`를 반환한다.
  - `build_graph` 최종 연결: `storyboarder --fan_out_scenes--> scene_worker → editor → publisher → END`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_normalize_metadata():
    m = Metadata(title="가" * 120, description="d", hashtags=["#AI", "AI", " 뉴스 "])
    out = normalize_metadata(m)
    assert len(out["title"]) == 100 and out["hashtags"] == ["#shorts", "#AI", "#뉴스"]
```

`test_graph_full.py`: Task 5의 기획 흐름 픽스처에 다음을 더해 끝까지 돌린다.
- 추가 준비: Storyboard 2장면, `Metadata` 응답, `image_source="placeholder"`, `tts=fake_tts`
- `settings.video_size=(108, 192)`, `fps=10`으로 빠르게 렌더링한다. 렌더러는 settings의 크기와 fps를 쓴다.
- 검증:
  - 최종 상태에 `video_path`가 있고 파일이 존재한다.
  - `scene_results`가 2개다(중복 없음).
  - `metadata["hashtags"][0] == "#shorts"`
  - 산출물 목록 전체(`research.json` … `metadata.json`, `scenes/01.jpg`, `scenes/02.jpg`)가 존재한다.

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `uv run pytest -v` 전체 PASS
- [ ] **Step 5: 실제 실행 확인 (수동)**
  - `./model_run.sh`를 띄운 뒤 `IMAGE_SOURCE=placeholder uv run python -m shorts_agent "AI 뉴스"`로 끝까지 실행한다. edge-tts를 쓰므로 인터넷이 필요하다.
  - `outputs/<id>/final.mp4`가 1080×1920으로 재생되고, 음성과 자막이 맞는지 확인한다.
  - mflux를 설치했다면 `IMAGE_SOURCE` 없이(기본 mflux) 한 번 더 실행해 본다.
- [ ] **Step 6: 커밋** — `feat(shorts): 퍼블리셔와 전체 파이프라인 연결`
