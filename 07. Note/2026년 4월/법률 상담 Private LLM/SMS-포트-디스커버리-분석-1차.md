# SMS 포트 디스커버리 - 런처 기반 아키텍처 분석

> 작성일: 2026-04-11
> 주제: SMS 랜덤 포트 디스커버리 방안 분석 및 런처 기반 대안 제시

## 현재 런처 구조 분석

```
[vscode-launcher.bat] (사용자 더블클릭 진입점)
  └─ vscode-launcher-org.bat
       ├─ Step 1: 포터블 환경 디렉터리 준비
       ├─ Step 2: VSIX 익스텐션 설치 확인
       ├─ Step 3: 기존 SMS pid kill → 새 SMS spawn (PowerShell hidden)
       ├─ Step 4: server.pid 생성될 때까지 대기 (최대 100초)
       └─ Step 5: VS Code 실행 (--user-data-dir, --extensions-dir, ws.code-workspace)
```

## 🔑 핵심 통찰: 런처가 모든 것의 오케스트레이터

**디스커버리 서버가 필요한 이유는 "Extension이 SMS를 미리 알 수 없을 때"입니다.**
하지만 이 구조는 **런처가 SMS를 먼저 띄우고, 그 다음 VS Code를 띄웁니다.** 즉:

- ✅ 런처는 SMS의 포트를 **알 수 있음** (또는 알게 만들 수 있음)
- ✅ 런처는 VS Code를 **직접 띄움** → 정보 전달 가능
- ✅ 모든 라이프사이클이 **선형적**이며 race condition 없음

→ **디스커버리 서버는 과도한 추상화입니다.**

---

## 권장 방안 (우선순위 순)

### 🥇 방안 1: 런처가 런타임 구성 파일을 생성하고 VS Code에 경로 전달

가장 단순하고 강력한 방법.

#### 흐름

```
[Launcher]
  1. SMS 시작 (랜덤 포트로) → server.runtime.json 생성 대기
     {
       "pid": 12345,
       "port": 35421,
       "startedAt": "2026-04-11T10:00:00Z",
       "workspaceId": "abc123"
     }
  2. VS Code 실행:
     code.exe --user-data-dir ...
              ws.code-workspace
     ※ 환경변수 SET WS_RUNTIME_CONFIG=<path-to-server.runtime.json>
```

#### Extension에서 (웹워커 호환)

웹 Extension은 `process.env`에 직접 접근 못 하지만, **VS Code 자체가 환경변수를 받아 시작**되었으므로 다음 방법으로 해결:

**옵션 A**: 런처가 워크스페이스 내부의 고정 위치에 파일을 둠

```
[workspace]/.vscode/sms.runtime.json
```

```typescript
const ws = vscode.workspace.workspaceFolders![0];
const uri = vscode.Uri.joinPath(ws.uri, ".vscode/sms.runtime.json");
const bytes = await vscode.workspace.fs.readFile(uri);
const { port } = JSON.parse(new TextDecoder().decode(bytes));
```

**옵션 B**: VS Code 시작 시 settings로 주입

```bat
:: 런처에서 settings.json에 직접 주입 (jq 또는 simple sed)
echo {"websquare.smsPort": 35421} > "%USER_SETTINGS_DIR%\settings.json"
```

```typescript
const port = vscode.workspace.getConfiguration("websquare").get<number>("smsPort");
```

| 장점                                   | 단점                          |
| -------------------------------------- | ----------------------------- |
| 디스커버리 서버 불필요                 | 런처가 SMS 출력을 파싱해야 함 |
| 라이프사이클 선형, race condition 없음 | 멀티 인스턴스 시 추가 처리 필요 |
| 웹 Extension 100% 호환                 | -                             |
| SPOF 없음                              | -                             |

---

### 🥈 방안 2: 워크스페이스 스코프 PID/Port 파일 (멀티 인스턴스 대응)

현재 런처의 최대 문제는 **`taskkill /F /PID !OLD_PID!`로 항상 기존 SMS를 죽임**이라는 점입니다. 이는 두 번째 VS Code를 띄우면 첫 번째 SMS가 죽는 버그입니다.

#### 개선 흐름

```
[Launcher 호출 시마다]
  1. 워크스페이스 식별자 생성 (예: 워크스페이스 경로 SHA256 hash)
     예: workspaceId = "a3f5c2..."

  2. 워크스페이스별 격리된 런타임 폴더 사용:
     %BIN_PATH%\runtime\%workspaceId%\
       ├─ server.pid
       └─ server.runtime.json   ← { port: 35421, ... }

  3. 해당 워크스페이스의 SMS만 kill (다른 워크스페이스 영향 없음)

  4. SMS 시작 시 환경변수로 workspaceId 전달:
     SET WS_RUNTIME_DIR=%BIN_PATH%\runtime\%workspaceId%
     start node App.js

  5. SMS는 받은 디렉터리에 server.pid + server.runtime.json 작성

  6. VS Code 실행 시 Extension에 workspaceId 알림
     SET WEBSQUARE_WORKSPACE_ID=%workspaceId%
     start code.exe ...
```

#### 멀티 인스턴스 동작 시뮬레이션

```
[Launcher 1] workspaceId=A → SMS_A(port:35421) → VS Code (인식 A)
[Launcher 2] workspaceId=B → SMS_B(port:47823) → VS Code (인식 B)
                                    ↑
                            서로 영향 없음
```

| 장점               | 단점                |
| ---------------- | ----------------- |
| 멀티 워크스페이스 완벽 격리  | 리소스 N배 (SMS N개)   |
| 디스커버리 불필요        | 런타임 디렉터리 정리 로직 필요 |
| 자격증명/AI Agent 격리 | -                 |

---

### 🥉 방안 3: 런처가 SMS spawn + stdout 파싱

```bat
:: SMS가 시작되면 첫 줄에 "PORT:35421" 출력하도록 약속
for /f "tokens=2 delims=:" %%a in ('"%NODE_EXE%" "%SMS_APP_PATH%" --print-port') do (
    set "SMS_PORT=%%a"
    goto :got_port
)
:got_port
echo SMS_PORT = %SMS_PORT%

:: VS Code 시작 시 환경변수 또는 settings 주입
```

| 장점               | 단점                                          |
| ------------------ | --------------------------------------------- |
| 매우 단순          | bat 스크립트로 stdout 파싱 어려움 (PowerShell 권장) |
| 파일 I/O 불필요    | nodemon 환경 등 stdout 형식이 변하면 깨짐     |

---

### 🥉 방안 4: 디스커버리 서버 (제안한 방안)

| 장점                               | 단점                          |
| ---------------------------------- | ----------------------------- |
| Extension이 독립적으로 동작 (런처 없이도) | 추가 컴포넌트, SPOF           |
| 표준화된 패턴                      | 이 구조에서는 과도함          |

---

## 깊은 분석: 왜 디스커버리보다 방안 1, 2가 우월한가

### 1. 라이프사이클 단순성

```
디스커버리 방식:
[SMS 시작] → [디스커버리 등록] → [Extension 조회] → [실제 통신]
                  ↑ 여러 시점에서 동기화 필요

런처 주도 방식:
[SMS 시작] → [포트 캡쳐] → [VS Code 시작] → [Extension은 이미 알고 있음]
              ↑ 런처가 모든 것을 보장
```

### 2. Race Condition 방지

디스커버리는 본질적으로 비동기이며 다음 race condition이 발생 가능:

- SMS는 시작했지만 디스커버리 등록 전 → Extension 조회 실패
- Extension이 디스커버리 조회 후 SMS 죽음 → stale port
- 멀티 윈도우 동시 시작 → 디스커버리 등록 충돌

런처 주도 방식은 **순차 보장**이라 race condition 자체가 불가능.

### 3. Web Extension 호환성

| 방식                               | 웹 Extension 호환 |
| ---------------------------------- | ----------------- |
| 디스커버리 (HTTP)                  | O                 |
| **런처 → 워크스페이스 파일**       | **O**             |
| **런처 → settings 주입**           | **O**             |
| 런처 → ENV (process.env)           | X                 |

### 4. 디버깅 용이성

- 디스커버리: 디스커버리 로그, SMS 로그, Extension 로그 3곳 봐야 함
- 런처 주도: `server.runtime.json` 파일 1개로 현재 상태 즉시 확인

### 5. 격리 vs 공유의 명시적 선택

- 디스커버리: 단일 디스커버리에 N개 SMS 등록 → 어느 SMS가 어느 워크스페이스 것인지 매핑 로직 필요
- 워크스페이스 스코프 파일: **디렉터리 구조 자체가 격리** (`runtime/[workspaceId]/`)

---

## 🎯 최종 권장 아키텍처

**상황별 분기:**

### 시나리오 A: **단일 VS Code 사용** (가장 흔한 경우)

→ **방안 1 (런처가 단일 runtime 파일 생성)**

```bat
:: 런처
1. SMS 시작 (server.runtime.json 작성 대기)
2. VS Code 시작 (워크스페이스 폴더 안에 sms.runtime.json 심볼릭 링크 또는 복사)
```

### 시나리오 B: **멀티 VS Code 동시 사용 + 격리 필요**

→ **방안 2 (워크스페이스 스코프 N개 SMS)**

```bat
:: 런처 호출마다
1. workspaceId = hash(workspace_path)
2. runtime/[workspaceId]/server.runtime.json 격리
3. SMS, VS Code에 workspaceId 전달
```

### 시나리오 C: **SMS가 런처 외부에서 실행됨** (예: 시스템 서비스)

→ **디스커버리 서버 도입**

이 경우만 디스커버리가 진짜 필요.

---

## 결론

현재 런처 구조를 보면 **시나리오 A 또는 B**에 해당합니다. 즉 **디스커버리 서버 없이 더 단순하게 해결 가능**합니다.

**구체적 행동 계획:**

1. **단기**: 방안 1 적용
   - SMS가 `server.runtime.json` 생성 (`{ port, pid }`)
   - 런처가 이 파일을 워크스페이스의 `.vscode/sms.runtime.json`에 복사
   - Extension은 `vscode.workspace.fs.readFile()`로 읽음

2. **중기**: 멀티 인스턴스 지원이 필요해지면 방안 2로 확장
   - 워크스페이스 hash 기반 격리
   - 런처가 인스턴스별로 격리된 런타임 디렉터리 사용

3. **디스커버리**: 정말 필요한 시점이 오면(예: VS Code 외부 도구가 SMS에 접근해야 할 때) 그때 도입

이 접근이 디스커버리 서버보다 단순하고, 안정적이며, 디버깅하기 쉽고, 웹 Extension과 완벽히 호환됩니다.
