
JSON 줄 이벤트는 이벤트 하나를 JSON 객체 하나로 만들어 한 줄에 하나씩 쓰는 출력 형식입니다. 일반적으로 JSON Lines(JSONL) 또는 NDJSON이라고 부릅니다. 전체가 하나의 JSON 배열이 아니라, 줄바꿈(`\n`)으로 끝나는 각 줄이 따로 완결된 JSON입니다.

```
{"type": "run", "thread_id": "20261114-..."}
{"type": "progress", "node": "search", "message": "후보 5건"}
{"type": "node", "node": "search"}
{"type": "interrupt", "thread_id": "20261114-...", "kind": "pick_article", "payload": {...}}
```

### 이 프로젝트에서 쓰는 이유
`shorts_graph` CLI는 그래프가 도는 몇 분 동안 진행 상황을 계속 알려야 합니다. 이 형식이면 웹 쪽은 줄이 하나 도착할 때마다 바로 `JSON.parse`해서 화면에 반영할 수 있습니다. 출력이 다 끝날 때까지 기다릴 필요가 없습니다. 출력이 중간에 끊겨 도착하면 줄바꿈이 올 때까지 모아 두었다가 처리합니다([run-cli.ts](shorts_web/lib/run-cli.ts)의 `LineSplitter`).

### 흐름
```
노드 ── progress.emit() ──┐
                          ▼
graph.stream(stream_mode=["updates","custom"])
                          ▼
cli/events.py  from_chunk / final_event  → dict
                          ▼
cli/main.py  emit(): json.dumps(event) + "\n" → stdout
                          ▼
shorts_web  stdout을 줄 단위로 나눠 JSON.parse → type 이름의 SSE 이벤트로 브라우저에 전달
```

### 이벤트 종류 (`type` 값)

| type | 만드는 곳 | 뜻 |
|---|---|---|
| `run` | `_stream` 시작 | 실행 시작, `thread_id` 알림 |
| `progress` | 노드의 `emit(...)` → [events.py:16](shorts_graph/cli/events.py#L16) | 진행 메시지 (예: "대본 작성 완료 (300자)") |
| `node` | [events.py:13-15](shorts_graph/cli/events.py#L13-L15) | 노드 하나가 끝남 |
| `interrupt` | `final_event` | 사람 입력을 기다리며 멈춤. `kind`는 `pick_article` 또는 `approve_script` |
| `done` | `final_event` | 그래프 종료. `status`, `message`, 영상이 있으면 `video_path` |
| `status` | `status_event` | `status` 명령의 결과(저장된 상태 조회) |
| `error` | CLI 오류 처리 | 실패 |

웹은 `interrupt`·`done`·`error` 중 하나가 나와야 실행이 정상적으로 끝났다고 봅니다([run-cli.ts](shorts_web/lib/run-cli.ts)의 `FINAL_TYPES`). 이 중 아무것도 없이 프로세스가 끝나면 비정상 종료로 처리합니다.

### 지켜야 할 규칙
웹이 stdout을 줄 단위로 파싱하므로 stdout에는 이 JSON 줄만 써야 합니다. 그래서 진행 로그나 `print`는 stderr로 보냅니다. 예를 들어 [progress.py](shorts_graph/core/progress.py)의 `emit`은 그래프 밖에서 불리면 stderr에 찍고, [wiring.py](shorts_graph/wiring.py)의 Pixabay 안내도 stderr로 나갑니다. stdout에 다른 글이 섞이면 그 줄은 이벤트로 해석되지 않습니다.

---

`print`가 웹까지 직접 보내는 것은 아닙니다. `print`는 stdout에 한 줄을 쓸 뿐이고, 그 stdout이 **서버로 이어지는 파이프**에 연결되어 있어서 서버가 그 줄을 읽고 브라우저로 넘겨 줍니다.

## 경로

```
cli.py  print(...)                         stdout에 JSON 한 줄
   │  (OS 파이프)
server.js  spawn() → child.stdout          서버가 그 줄을 받음
   │  readline "line"                       줄 단위로 끊음
   │  stdoutLineToEvent(line)               JSON 해석 → {event, data}
   │  res.write(encodeSse(event, data))     "event: progress\ndata: {...}\n\n"
   │  (끝나지 않고 열려 있는 HTTP 응답, text/event-stream)
index.html  EventSource addEventListener   화면에 한 줄 추가
```

1. **`spawn()`이 stdout을 파이프로 연결합니다** ([server.js:56](demo/sse_cli/server.js#L56)). 서버가 CLI를 자식 프로세스로 띄우면, 자식의 stdout은 터미널이 아니라 부모(Node)와 이어진 파이프가 됩니다. 그래서 `print`한 내용이 화면에 찍히지 않고 서버의 `child.stdout`으로 들어갑니다.
2. **줄 단위로 읽습니다** ([server.js:67](demo/sse_cli/server.js#L67)). `readline`은 청크 경계에서 잘린 줄과 한글(UTF-8 바이트)을 이어 붙여 한 줄씩 넘겨 줍니다. `print`가 끝에 붙이는 `\n`이 줄의 경계가 됩니다.
3. **JSON을 SSE 형식으로 바꿉니다.** [lib.js:40](demo/sse_cli/lib.js#L40)의 `stdoutLineToEvent`가 JSON을 해석하고, [lib.js:32](demo/sse_cli/lib.js#L32)의 `encodeSse`가 `event:`·`data:` 두 줄과 빈 줄로 만듭니다.
4. **응답을 끝내지 않고 이어 씁니다.** `Content-Type: text/event-stream`과 `flushHeaders()`로 응답을 먼저 열어 두고([server.js:37-41](demo/sse_cli/server.js#L37-L41)), 이벤트가 생길 때마다 `res.write`로 덧붙입니다.
5. **브라우저가 받습니다.** `EventSource`는 열린 응답에서 빈 줄이 나올 때마다 이벤트 하나를 꺼내, 이름(`progress` 등)에 맞는 리스너를 부릅니다([index.html:98](demo/sse_cli/index.html#L98)).

## `flush=True`가 필요한 이유

- 출력 대상이 터미널이면 파이썬은 줄마다 내보냅니다(줄 버퍼링).
- 출력 대상이 파이프면 약 8KB를 모았다가 한꺼번에 내보냅니다(블록 버퍼링).
- `count 5`의 이벤트 몇 줄은 8KB에 한참 못 미칩니다. 그래서 flush가 없으면 프로세스가 끝날 때까지 버퍼에 남아 있다가 마지막에 한꺼번에 도착합니다. 진행 표시가 실시간으로 움직이지 않게 됩니다.
- `flush=True`를 주면 줄마다 바로 파이프로 밀어냅니다.

## argparse 버전에서 주의할 점

`redirect_stdout(sys.stderr)` 안에서는 이 `print`가 **stderr로 갑니다**. 그러면 서버가 이 줄을 `stderrLineToEvent`로 받아 `log` 이벤트로 감싸 버리고, 진짜 이벤트(`progress` 등)로 인식하지 못합니다. 그래서 argparse 버전은 redirect하기 전에 `out = sys.stdout`을 잡아 두고 `out.write(...)` + `out.flush()`로 썼습니다. shorts_graph도 같은 방식입니다.

지금 작업 폴더에서는 `cli.py`가 커밋되지 않은 상태로 예전 `print` 버전으로 돌아가 있고, `cli_argparser.py`라는 새 파일이 있습니다. 비교하려고 나눠 두신 것이라면, 서버는 `demo.sse_cli.cli`를 실행하므로 지금은 예전 버전이 돌아갑니다.