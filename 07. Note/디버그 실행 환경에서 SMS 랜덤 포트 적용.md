
## 변경 요약

|파일|변경|
|---|---|
|[studio-message-server/.env](vscode-webview://1tk9858l3ui9jtn3vjirul74fd8dgc9m16qdd5g81kkqb2q1j78g/src/studio-message-server/.env)|`PORT=0` (OS 랜덤 포트)|
|[studio-message-server/src/App.ts](vscode-webview://1tk9858l3ui9jtn3vjirul74fd8dgc9m16qdd5g81kkqb2q1j78g/src/studio-message-server/src/App.ts)|`app.listen(0)` → 실제 포트 획득 → `server.runtime.json` 작성. 종료 시 파일 정리|
|[vscode-websquare-extension/.env](vscode-webview://1tk9858l3ui9jtn3vjirul74fd8dgc9m16qdd5g81kkqb2q1j78g/src/vscode-websquare-extension/.env)|`STUDIO_MESSAGE_SERVER_PORT=0` (런타임 주입)|
|[vscode-websquare-extension/src/common/environment/env.ts](vscode-webview://1tk9858l3ui9jtn3vjirul74fd8dgc9m16qdd5g81kkqb2q1j78g/src/vscode-websquare-extension/src/common/environment/env.ts)|`loadSmsRuntime()` 함수 추가 — `vscode.workspace.fs`로 `server.runtime.json` 읽어 포트 주입 (최대 30회 × 500ms 재시도)|
|[vscode-websquare-extension/src/App.ts](vscode-webview://1tk9858l3ui9jtn3vjirul74fd8dgc9m16qdd5g81kkqb2q1j78g/src/vscode-websquare-extension/src/App.ts)|`activate()` 최초 단계에서 `await loadSmsRuntime()` 호출|

## 동작 흐름 (디버그 모드)

```
F5 (Launch PaaS Development)
  ↓
preLaunchTask: start-server-and-build-extension
  ↓
kill-existing-server → build-studio-message-server → start-studio-message-server
  ↓
SMS 시작:
  - app.listen(0) → OS가 랜덤 포트 할당 (예: 53421)
  - server.runtime.json 작성: { port: 53421, pid: 12345, startedAt: "..." }
  ↓
Extension Host 시작:
  - activate() 호출
  - loadSmsRuntime():
    - workspaceFile 위치 기준 ../../studio-message-server/server.runtime.json 경로 계산
    - vscode.workspace.fs.readFile()로 읽기 (웹워커 호환)
    - env.STUDIO_MESSAGE_SERVER_PORT = "53421"
    - env.STUDIO_MESSAGE_SERVER_ORIGIN = "http://127.0.0.1:53421"
    - env.SERVER_SOCKET_ORIGIN = "ws://127.0.0.1:53421"
  ↓
이후 모든 통신은 53421 포트로 정상 동작
```

## 확장 경로

배포 환경(런처)에서는 `resolveRuntimeUri()`를 wsHash 기반으로 확장하면 됩니다 (1차 분석 문서의 Phase 2 참조).