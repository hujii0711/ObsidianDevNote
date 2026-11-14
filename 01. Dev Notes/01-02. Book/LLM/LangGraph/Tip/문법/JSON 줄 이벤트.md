
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