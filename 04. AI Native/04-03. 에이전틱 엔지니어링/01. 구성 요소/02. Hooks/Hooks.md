![[Pasted image 20260419152551.png]]
### 1. Hook이란?

Claude Code의 **자동화 엔진**입니다. 이벤트가 발생하면 matcher가 조건을 검사하고, 조건이 맞으면 지정된 액션이 자동으로 실행됩니다.

🔄

**이벤트 → Hook 감지 → 액션 실행** 파일 저장, 도구 호출, 알림 수신 등의 이벤트 발생 → matcher가 조건 검사 (와일드카드 * 또는 특정 패턴) → lint 실행, format 적용, 알림 전송 등 자동 실행

### 2. Hook 만들기

![[Pasted image 20260419155649.png]]

- **방법 1 — Claude에게 요청 (추천):** "알림 Hook 만들어줘"라고 말하면 settings.json에 자동 추가
- **방법 2 — settings.json 직접 편집:** `~/.claude/settings.json` (개인용) 또는 `.claude/settings.json` (프로젝트용)

### 3. Hook JSON 구조

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "terminal-notifier -title 'Claude Code' -message '알림이 있습니다' && afplay /System/Library/Sounds/Ping.aiff &"
          }
        ]
      }
    ]
  }
}
```

**JSON 구조 분해:**

- **이벤트 타입** (`Notification`) — 언제 실행할지
- **matcher** (`*`) — 어떤 상황에 매칭할지 (와일드카드 = 전부)
- **hooks 배열** — 실행할 Hook 목록 (여러 개 가능)
- **command** — 실제 실행할 셸 명령어

### 4. 이벤트 타입

![[Pasted image 20260419160306.png]]

| 이벤트              | 타이밍         | 설명                     |
| ---------------- | ----------- | ---------------------- |
| **PreToolUse**   | 도구 호출 직전    | 입력값 검증, 승인/차단/수정 가능    |
| **PostToolUse**  | 도구 실행 직후    | 결과 검증, 후처리 (린트, 포맷팅 등) |
| **Notification** | 사용자 응답 대기 시 | 알림 전송, 로깅, 외부 서비스 연동   |
| **Stop**         | 에이전트 턴 종료 시 | 최종 정리, 보고서 생성, 상태 저장   |

⚠️ Hook 실행 중 Claude는 멈춰서 기다립니다. **timeout**을 꼭 설정하고, 무거운 작업은 백그라운드(`&`)로 돌리세요.

---
---

클로드 코드의 다양한 처리 시점에 맞춰 특정 명령어를 실행할 수 있다.

- 훅을 만드는 방법
(~)project/.claude/settings.json 안에 실행 타이밍(후킹 이벤트)과 실행하고 싶은 스크립트를 작성하면 된다.
이때 실행 타이밍은 클로드 코드 세션 시작 시점이나 종료 시점, 명령어 사용 시점 등이다.

- 클로드가 파일을 새로 생성하거나 편집할 때 포매터나 린터를 실행하는 훅 설정
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.file_path // .path // \"\"' | grep -E '\\.(ts|tsx|js|jsx|mjs|cjs)$' | xargs -I{} sh -c 'npx prettier --write \"{}\" 2>&1 && echo \"✅ Prettier: {}\"'",
            "timeout": 30,
            "statusMessage": "Prettier 포맷 중..."
          },
          {
            "type": "command",
            "command": "echo \"$CLAUDE_TOOL_INPUT\" | jq -r '.file_path // .path // \"\"' | grep -E '\\.(ts|tsx|js|jsx|mjs|cjs)$' | xargs -I{} sh -c 'npx eslint --fix \"{}\" 2>&1; exit 0'",
            "timeout": 60,
            "statusMessage": "ESLint 검사 중..."
          }
        ]
      }
    ]
  }
}
```

- 훅 테스트하기
```
@settings.json이 훅이 작동하는지 테스트하세요.
```

- 훅의 장점
확률적으로 정해지는 클로드의 출력 결과에 대해 확실하게(결정론적으로) 실행이나 출력 결과를 가져온다는 것이다. 따라서 클로드의 실행에 확실성을 부여하고 싶을 때는 훅을 적극적으로 활용하면 좋다.

- 훅 활용 예
1) 비밀 파일에 대한 접근을 차단하거나 설정 파일의 삭제를 방지할 수 있다.
2) 프롬프트에 특정 컨텍스트를 포함시켜 클로드가 기억해야 할 내용을 항상 상기 시킬 수 있다.
3) 클로드의 알림을 데스크탑이나 슬랙에 보내 대기 시간을 효율적으로 활용할 수 있다.

- Hook 4가지 유형

| Hook 유형      | 동작 시점              | 실전 예시              |
| ------------ | ------------------ | ------------------ |
| Pre-action   | 에이전트가 행동하기 전       | 코드 저장 전 lint 자동 실행 |
| Post-action  | 에이전트가 행동한 후        | 커밋 후 자동 테스트 실행     |
| Validation   | 에이전트 출력의 품질 검증     | 보안 취약점 스캔          |
| Notification | 특정 조건 충족 시 인간에게 알림 | 비용 임계치 초과 경고       |
- Claude Code Hook 실전 예시
1) 파일 변경시 --> 관련 테스트 자동 실행
2) package.json -->  의존성 충돌 검사
3) 보안 관련 파일 수정 시 --> 보안 리뷰 에이전트 자동 호출
4) 토큰 사용량 초과 시 --> 컨텍스트 압축 트리거

- Hook + Skill 시너지

| Hook(감지)     | Skill(행동)           |
| ------------ | ------------------- |
| .pptx 생성 감지  | pptx 베스트 프랙티스 자동 로드 |
| 보안 코드 변경 감지  | 보안 리뷰 체크리스트 자동 적용   |
| API 통합 코드 감지 | 해당 API 가이드 라인 자동 주입 |

### 훅(Hooks)과 안전장치

- **배경**: 아마존 2026년 3월 사고 - AI가 작성한 코드를 검토 없이 배포하여 6시간 다운타임 발생  
- **훅의 정의**: AI가 특정 행동을 하기 직전/직후에 자동 실행되는 스크립트  
- **4가지 실용적인 훅**:  
    1. **린트/테스트/빌드 훅**: 커밋 전 자동 실행  
    2. **PR 리뷰 훅**: 서브 에이전트를 통한 확증 편향 방지  
    3. **TDD 훅**: 테스트 미작성 시 파일 수정 차단  
    4. **서비스 장애 패턴 검증 훅**: 과거 장애 패턴을 스크립트화하여 자동 검증   
- 사람이 아닌 시스템에 의존하는 것이 핵심