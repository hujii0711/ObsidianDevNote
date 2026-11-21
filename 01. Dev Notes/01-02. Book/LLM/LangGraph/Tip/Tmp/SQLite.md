
SQLite는 관계형 데이터베이스(RDBMS)입니다. 테이블, 행, 열 구조에 SQL로 질의하고, 기본키/외래키, JOIN, 인덱스, 뷰, 트리거, 트랜잭션(ACID)을 지원합니다. 주로 모바일 앱, 데스크톱 앱, 소규모 서비스, 로컬 캐시나 테스트용으로 적합하고, 동시 접속이 많은 대규모 서비스에는 PostgreSQL 같은 서버형이 낫습니다.

---

실행하면 `SqliteSaver`가 `checkpoint.db` 안에 테이블 두 개를 만듭니다. `checkpoints`에는 각 단계가 끝난 뒤의 상태 전체가, `writes`에는 노드가 쓴 값이 들어갑니다. 아래 내용은 지금 DB를 직접 조회하고 BLOB을 풀어 확인한 것입니다.

## 1. `checkpoints`: 단계별 상태 스냅숏

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `thread_id` | TEXT | 대화(실행) 구분 키. 지금 코드는 `"user-001"` |
| `checkpoint_ns` | TEXT | 서브그래프 구분용 이름. 최상위 그래프는 `''` |
| `checkpoint_id` | TEXT | 체크포인트 ID. 시간순으로 정렬됩니다 |
| `parent_checkpoint_id` | TEXT | 바로 앞 체크포인트. 이걸 따라가면 실행 이력이 됩니다 |
| `type` | TEXT | 직렬화 형식 (`msgpack`) |
| `checkpoint` | BLOB | 상태 본문 (아래 참고) |
| `metadata` | BLOB | `{"source", "step", "parents"}` (JSON) |

기본 키는 `(thread_id, checkpoint_ns, checkpoint_id)`입니다.

`checkpoint` BLOB을 풀면 이렇게 나옵니다.
```json
{
  "v": 4,
  "ts": "2026-11-21T11:06:24...",
  "id": "1f1e59b9-7311-...",
  "channel_values": { "message": "안녕하세요 처리 완료" },   // ← State 값
  "channel_versions": { "__start__": "...", "branch:to:process": "...", "message": "..." },
  "versions_seen":   { "__start__": {...}, "process": {...} },
  "updated_channels": ["message"]
}
```
- `channel_values`가 실제 `State`입니다. `State`에 `message: str`를 선언했기 때문에 `message` 채널이 생겼습니다.
- `channel_versions`와 `versions_seen`은 LangGraph가 다음에 어떤 노드를 실행할지 정할 때 씁니다.

## 2. `writes`: 노드가 쓴 값(중간 기록)

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `thread_id`, `checkpoint_ns`, `checkpoint_id` | TEXT | 어느 체크포인트 단계에서 쓴 값인지 |
| `task_id` | TEXT | 값을 쓴 작업(노드 실행) ID |
| `task_path` | TEXT | 작업 경로 |
| `idx` | INTEGER | 한 작업 안에서 쓴 순서. 에러는 `-1` |
| `channel` | TEXT | 값을 쓴 채널. `message`, `branch:to:process`, `__error__` 등 |
| `type`, `value` | TEXT, BLOB | 직렬화된 값 |

기본 키는 `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`입니다.

## 3. `invoke` 한 번에 쌓이는 행

한 번 실행하면 `checkpoints`에 3행이 추가됩니다.

| step | source | 시점 | `message` |
|---|---|---|---|
| n | `input` | 입력을 받은 직후 | (아직 없음) |
| n+1 | `loop` | `__start__`가 입력을 반영한 뒤 | `"안녕하세요"` |
| n+2 | `loop` | `process` 노드가 끝난 뒤 | `"안녕하세요 처리 완료"` |

`writes`에는 같은 실행에서 `message = "안녕하세요"`(`__start__`가 씀)와 `message = "안녕하세요 처리 완료"`(`process`가 씀)가 기록됩니다.

## 지금 DB에 들어 있는 것

`thread_id`가 매번 `"user-001"`로 같아서 실행할 때마다 기존 이력 뒤에 이어 붙습니다. 지금은 세 번 실행한 기록이 있습니다.
- **step -1~0:** 아까 `KeyError`로 실패한 실행입니다. `writes`에 `__error__` 행이 남아 있습니다.
- **step 1~3, 4~6:** 성공한 실행 두 번입니다.

같은 `thread_id`로 다시 실행하면 이전 상태를 이어받습니다. 새로 시작하려면 `thread_id`를 바꾸거나 `data/checkpoint.db`를 지우면 됩니다.

---
네, 테이블 구조는 정해져 있습니다. 다만 BLOB 안에 들어가는 내용은 우리가 선언한 `State`에 따라 달라집니다.

## 고정된 것: 테이블 구조

`checkpoints`와 `writes` 두 테이블은 우리 코드가 아니라 `langgraph-checkpoint-sqlite` 라이브러리(지금 설치된 버전은 3.1.3)가 만듭니다.
- 처음 체크포인트를 읽거나 쓸 때 `SqliteSaver.setup()`이 자동으로 호출됩니다.
- 이 함수가 `CREATE TABLE IF NOT EXISTS checkpoints ...`와 `writes ...`를 실행합니다. 같은 곳에서 `PRAGMA journal_mode=WAL`도 설정합니다.
- 테이블 이름, 컬럼, 기본 키는 이 SQL에 박혀 있어 설정으로 바꿀 수 없습니다.
- 라이브러리 버전이 올라가면 구조가 바뀔 수 있습니다. 실제로 `writes.task_path` 컬럼은 나중에 추가되어서, `setup()`에 기존 DB용 `ALTER TABLE writes ADD COLUMN task_path ...` 코드가 들어 있습니다.

그래서 이 DB를 SQL로 직접 읽는 코드를 짜면 라이브러리 내부 구조에 묶이게 됩니다. 상태는 `graph.get_state(config)`나 `graph.get_state_history(config)`로 읽는 편이 안전합니다.

## State에 따라 바뀌는 것: BLOB 내용

`checkpoint` BLOB 안의 `channel_values` 같은 값은 `StateGraph(State)`에 넘긴 `State` 선언을 그대로 따릅니다.
- 지금은 `message: str` 하나라서 `channel_values`가 `{"message": ...}`입니다.
- 예를 들어 `count: int`를 추가하면 `channel_values`에 `count`가 생기고, `writes.channel`에도 `count` 행이 기록됩니다.
- 노드나 엣지를 바꾸면 `branch:to:process` 같은 내부 채널 이름이 달라집니다.

정리하면, SQL 테이블은 라이브러리가 정한 대로 늘 같고, 그 안의 msgpack 데이터 구조는 `State`와 그래프 구성이 결정합니다.