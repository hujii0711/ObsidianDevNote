# SMS 포트 디스커버리 - 웹워커 + 멀티 워크스페이스 제약 분석

> 작성일: 2026-04-11
> 주제: 웹워커 빌드 + 멀티 워크스페이스 제약을 반영한 SMS 포트 디스커버리 최종 권장 아키텍처

## 제약 조건 재정리

| 제약                                       | 영향                                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| **웹워커 타깃**                            | `fs`, `child_process`, `process.env` 사용 불가. `vscode.*` API만 사용             |
| **멀티 워크스페이스**                      | `ws.code-workspace` 단위로 N개 워크스페이스 동시 운용 가능                          |
| **포터블 VS Code Desktop**                 | `vscode.workspace.fs.readFile(vscode.Uri.file("절대경로"))` 가능                  |
| **런처가 SMS와 VS Code를 모두 spawn**      | 런처가 오케스트레이터                                                              |

## 멀티 워크스페이스 시나리오 정의

### 시나리오 분류

```
[Scenario A] ws.code-workspace 1개 → VS Code 1개  (단일 인스턴스)
[Scenario B] ws.code-workspace A   → VS Code 윈도우 1
              ws.code-workspace B   → VS Code 윈도우 2  (다른 워크스페이스 동시 운용)
[Scenario C] 같은 ws.code-workspace를 두 윈도우에서 열기  (VS Code가 보통 차단)
```

이 프로젝트는 **시나리오 A + B**가 일반적입니다.

---

## 권장 아키텍처: **워크스페이스 키 기반 런타임 레지스트리**

### 핵심 설계

```
[bin/]
  └─ runtime/                          ← 런처가 관리하는 런타임 레지스트리
      ├─ {ws-hash-A}/
      │   ├─ server.pid
      │   └─ server.runtime.json       ← { port, pid, wsFile, startedAt }
      ├─ {ws-hash-B}/
      │   ├─ server.pid
      │   └─ server.runtime.json
      └─ index.json                    ← { "ws-hash-A": "/path/to/ws-A.code-workspace", ... }
```

**워크스페이스 해시**: `ws.code-workspace` 절대경로의 SHA256 (또는 간단히 base64) → 워크스페이스 = 런타임 인스턴스 1:1 매핑

### 흐름

```
[1. 런처 vscode-launcher.bat 실행]
    │
    ├─ workspaceFile = "C:\paas-project\PaaS\src\workspace\ws.code-workspace"
    ├─ wsHash = SHA256(workspaceFile)  → "a3f5c2..."
    ├─ runtimeDir = "%BIN_PATH%\runtime\a3f5c2..."
    │
    ├─ 기존 runtime/{wsHash}/server.pid 있으면 → 그 SMS가 살아있는지 확인
    │     ├─ 살아있음 → 재사용 (kill 안 함) ★
    │     └─ 죽었음   → pid 삭제하고 새로 시작
    │
    ├─ SMS 시작:
    │     SET WS_HASH=a3f5c2...
    │     SET WS_RUNTIME_DIR=%BIN_PATH%\runtime\a3f5c2...
    │     SET WS_FILE=C:\...\ws.code-workspace
    │     start node App.js
    │     ↓
    │     SMS는 받은 디렉터리에 server.runtime.json 작성
    │     { port: 35421, pid: 12345, wsFile: "...", startedAt: "..." }
    │
    ├─ runtime/index.json 갱신:
    │     { "a3f5c2...": "C:\...\ws.code-workspace" }
    │
    └─ VS Code 시작:
          start code.exe --user-data-dir ... ws.code-workspace
          ※ 별도 정보 전달 불필요
            → Extension이 workspaceFile만 알면 wsHash를 재계산 가능
```

```
[2. Extension 활성화 (웹워커)]
    │
    ├─ wsFile = vscode.workspace.workspaceFile
    │     예: file:///c:/paas-project/PaaS/src/workspace/ws.code-workspace
    │
    ├─ wsHash = SHA256(wsFile.fsPath)  → "a3f5c2..."
    │     ↑ 웹워커에서도 SubtleCrypto API로 SHA256 계산 가능
    │
    ├─ runtime 디렉터리 경로 추론:
    │     런처가 미리 알려준 BIN_PATH 또는 워크스페이스 상대경로로 추정
    │     예: <wsFile>/../../bin/runtime/{wsHash}/server.runtime.json
    │
    ├─ vscode.workspace.fs.readFile(vscode.Uri.file(절대경로)):
    │     { port: 35421, pid: 12345, ... }
    │
    └─ 이후 모든 통신은 localhost:35421
```

### Extension 코드 (웹워커 호환)

```typescript
import * as vscode from "vscode";

async function getSmsPort(): Promise<number> {
  const wsFile = vscode.workspace.workspaceFile;
  if (!wsFile) throw new Error("워크스페이스 파일이 없음");

  // 1. wsFile 경로의 SHA256 계산 (웹워커에서 SubtleCrypto 사용)
  const encoder = new TextEncoder();
  const hashBuffer = await crypto.subtle.digest("SHA-256", encoder.encode(wsFile.fsPath));
  const wsHash = Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .substring(0, 16); // 16자만 사용

  // 2. runtime 파일 경로 (런처 약속 위치)
  // 예: ws.code-workspace 기준 ../../bin/runtime/{wsHash}/server.runtime.json
  const runtimeUri = vscode.Uri.joinPath(
    wsFile,
    "..", "..", "..",
    "bin", "runtime", wsHash, "server.runtime.json"
  );

  // 3. 파일 읽기 (vscode.workspace.fs는 웹워커에서도 동작)
  const bytes = await vscode.workspace.fs.readFile(runtimeUri);
  const config = JSON.parse(new TextDecoder().decode(bytes));
  return config.port;
}
```

### 런처 수정 (vscode-launcher-org.bat)

```bat
:: 워크스페이스 해시 계산 (PowerShell 활용)
for /f %%h in ('powershell -Command "[System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes('%WORKSPACE_PATH%'))).Replace('-','').Substring(0,16).ToLower()"') do set WS_HASH=%%h

set "RUNTIME_DIR=%BIN_PATH%\runtime\%WS_HASH%"
set "RUNTIME_PID=%RUNTIME_DIR%\server.pid"
set "RUNTIME_JSON=%RUNTIME_DIR%\server.runtime.json"

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

:: 기존 SMS가 살아있는지 확인 → 살아있으면 재사용
set "REUSE_SMS="
if exist "%RUNTIME_PID%" (
    set /p OLD_PID=<"%RUNTIME_PID%"
    tasklist /FI "PID eq !OLD_PID!" 2>nul | findstr /I "node.exe" >nul
    if !errorlevel! equ 0 (
        echo [SMS] 기존 인스턴스 재사용 PID=!OLD_PID!
        set "REUSE_SMS=1"
    ) else (
        del "%RUNTIME_PID%" >nul 2>&1
        del "%RUNTIME_JSON%" >nul 2>&1
    )
)

if not defined REUSE_SMS (
    echo [SMS] 새 인스턴스 시작 wsHash=%WS_HASH%
    powershell -Command ^
      "$env:WS_HASH='%WS_HASH%'; $env:WS_RUNTIME_DIR='%RUNTIME_DIR%'; $env:WS_FILE='%WORKSPACE_PATH%'; Start-Process '%NODE_EXE%' -ArgumentList '%SMS_APP_PATH%' -WindowStyle Hidden"

    :: server.runtime.json 생성 대기
    set WAIT_COUNT=0
    :WAIT_LOOP
    if not exist "%RUNTIME_JSON%" (
        set /a WAIT_COUNT+=1
        if !WAIT_COUNT! geq 30 ( goto END )
        timeout /t 1 /nobreak > nul
        goto WAIT_LOOP
    )
)

:: VS Code 시작
start "" "%VSCODE_EXE%" --extensions-dir "%USER_EXT_DIR%" --user-data-dir "%USER_DATA_DIR%" "%WORKSPACE_PATH%"
```

### SMS 수정 (App.ts)

```typescript
const runtimeDir = process.env.WS_RUNTIME_DIR || rootDir;
const wsHash = process.env.WS_HASH || "default";
const wsFile = process.env.WS_FILE || "";

const server = app.listen(0, async () => {  // 포트 0 → OS가 랜덤 할당
  const port = (server.address() as AddressInfo).port;

  const runtimePath = path.join(runtimeDir, "server.runtime.json");
  fs.writeFileSync(runtimePath, JSON.stringify({
    port,
    pid: process.pid,
    wsHash,
    wsFile,
    startedAt: new Date().toISOString(),
  }, null, 2));

  fs.writeFileSync(path.join(runtimeDir, "server.pid"), String(process.pid));
});
```

---

## 다른 방안과의 비교 (웹워커 + 멀티 워크스페이스 기준)

| 방안                               | 웹워커 호환 | 멀티 워크스페이스 격리  | SPOF   | 구현 복잡도 |
| -------------------------------- | ------ | ------------- | ------ | ------ |
| **워크스페이스 키 런타임 레지스트리** ⭐         | **O**  | **O** (완벽)    | 없음     | 중간     |
| 디스커버리 서버 + N SMS                 | O      | O             | 디스커버리  | 높음     |
| 공유 SMS + socketMapKey 격리         | O      | △ (글로벌 상태 충돌) | SMS 1개 | 낮음     |
| `.vscode/sms.port` (워크스페이스 폴더 내) | O      | X (어느 폴더?)    | 없음     | 낮음     |
| settings.json 주입                 | O      | △ (사용자 설정 오염) | 없음     | 중간     |

---

## 디스커버리 서버 vs 권장안: 결정적 차이

### 디스커버리 서버의 본질적 문제 (이 환경에서)

1. **고정 포트 충돌**: 런처가 이미 SMS의 포트 충돌을 처리하는데, 디스커버리는 또 다른 포트 충돌 지점 추가
2. **추가 프로세스**: SMS + 디스커버리 = 2개 프로세스 관리 (라이프사이클 동기화 필요)
3. **중복 정보**: 디스커버리는 결국 "워크스페이스 → 포트" 매핑인데, 이미 런처가 알고 있음
4. **웹워커에서는 fetch만 가능**: 디스커버리가 죽으면 복구 불가능

### 권장안의 우월한 점

1. **파일 시스템이 곧 디스커버리**: `runtime/{wsHash}/` 디렉터리 자체가 워크스페이스 인덱스
2. **자가 복구**: Extension이 활성화될 때마다 파일을 다시 읽음 → 런타임 변경 즉시 반영
3. **런처 없이도 동작 가능**: SMS가 약속된 위치에 파일만 쓰면 됨 (런처는 편의)
4. **감사/디버깅 용이**: `runtime/index.json`만 보면 현재 활성 인스턴스 전체 현황 파악
5. **고정 포트 의존성 0**: SMS는 OS가 할당한 랜덤 포트, 디스커버리 포트도 없음

---

## 추가 고려사항

### 1. 같은 워크스페이스 두 윈도우 (시나리오 C)

```
[Launcher 1] wsHash=A → SMS_A 시작 (port 35421)
[Launcher 2] wsHash=A → 기존 SMS_A 살아있음 감지 → 재사용 (kill 안 함)
                        → VS Code 윈도우 2 시작 (같은 SMS_A 공유)
```

→ 두 윈도우가 같은 SMS를 공유. `socketMapKey`로 클라이언트 식별 (현재 코드 그대로)

### 2. 고아 인스턴스 정리

런처 시작 시 `runtime/index.json`을 순회하며:

- pid가 죽은 인스턴스의 디렉터리 정리
- 일정 시간(예: 7일) 사용되지 않은 인스턴스 정리

### 3. SHA256 대신 단순 해시

bat에서 SHA256은 부담스러우면 단순한 base32 인코딩이나 hash trick 사용 가능. 일관성만 있으면 됨.

### 4. WSL/원격 환경 고려

`vscode.workspace.workspaceFile`이 `vscode-remote://` 스킴이면 다른 처리 필요. 현재는 로컬 데스크톱이라 무시 가능.

---

## 최종 결론

**디스커버리 서버는 이 환경에서 과도한 복잡도입니다.**

이유:

1. 런처가 이미 워크스페이스를 식별 가능
2. 웹워커 Extension은 `vscode.workspace.fs`로 절대경로 파일 읽기 가능 (Desktop)
3. 멀티 워크스페이스는 워크스페이스 해시로 자연스럽게 격리
4. 추가 프로세스/포트 없이 파일 시스템만으로 디스커버리 효과 달성

**권장: "워크스페이스 키 기반 런타임 레지스트리"** 패턴 채택.

구현 단계:

- **Phase 1**: SMS가 `server.runtime.json` 작성하도록 수정 (포트 0 → 랜덤 포트 캡쳐)
- **Phase 2**: 런처가 wsHash 기반 런타임 디렉터리 사용 (기존 SMS 재사용 로직 포함)
- **Phase 3**: Extension이 `vscode.workspace.fs`로 runtime 파일 읽기
- **Phase 4**: 고아 인스턴스 정리 + index.json 관리

이 접근이 디스커버리보다 단순하고, 안정적이며, 웹워커 + 멀티 워크스페이스 제약을 자연스럽게 충족합니다.
