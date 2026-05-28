
# Claude Code 하네스 엔지니어링의 피드백 루프와 GitHub Actions

## 결론부터

**GitHub Actions가 가장 적절한 유일한 방법은 아닙니다.**
피드백 루프의 **위치(어디서 발생하는가)** 에 따라 적합한 도구가 다릅니다.

---

## 하네스의 피드백 루프란?

Claude Code의 각 툴 사용은 결과를 반환하고, 그 결과가 다시 루프에 피드백되어 Claude의 다음 결정을 이끕니다. [Claude](https://code.claude.com/docs/en/how-claude-code-works) 이것이 피드백 루프의 핵심입니다.

에이전트 패턴의 최소 루프는 다음과 같습니다:
- 모델이 툴 호출 여부를 결정
- 코드가 모델이 요청한 것을 실행
- 결과를 다시 messages[]에 추가
- 루프 반복 [GitHub](https://github.com/shareAI-lab/learn-claude-code)

---

## 피드백 루프의 두 가지 레벨

### 1. 에이전트 내부 루프 (Hooks 기반)
Claude Code 실행 **세션 안에서** 실시간으로 동작

```
PreToolUse Hook → 툴 실행 → PostToolUse Hook → 결과 피드백
```

PreToolUse 훅이 exit code 2로 종료되는 것이 Claude Code에서 툴 호출을 무조건 차단하는 유일한 메커니즘입니다. CLAUDE.md의 지시는 컨텍스트나 모델 추론으로 재정의될 수 있지만, 훅은 우회할 수 없습니다. [DEV Community](https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j)

```bash
# PostToolUse Hook 피드백 루프 예시
# 테스트 실패 시 → 즉시 Claude에게 피드백
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "bash .claude/hooks/verify-tests.sh"
      }]
    }]
  }
}
```

### 2. 에이전트 외부 루프 (CI/CD 기반)
코드가 **레포지토리에 반영된 후** 동작 → 여기서 GitHub Actions 활용

---

## Anthropic이 말하는 피드백 루프 핵심 패턴

생성기(generator)와 평가기(evaluator)를 분리함으로써 생성기를 더 강한 출력으로 이끄는 피드백 루프를 만들 수 있습니다. 이를 위해 생성 에이전트와 평가 에이전트 모두에게 4가지 채점 기준을 제공했습니다: 디자인 품질, 독창성, 기술 실행력, 기능성. [Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)

즉 Anthropic이 강조하는 피드백 루프의 핵심은 **자기검증(Self-Verification)** 입니다.

LangChain의 세 가지 하네스 개선 중 하나는 에이전트가 작업 완료 전 스스로 작업을 검증하는 검증 미들웨어였습니다. Boris Cherny(Claude Code 창시자)는 검증이 "아마도 품질을 위한 가장 중요한 것"이라고 말합니다. [DEV Community](https://dev.to/shipwithaiio/the-complete-claude-code-harness-engineering-guide-5-layers-8-deep-dives-3d4j)

---

## GitHub Actions가 적합한 피드백 루프 시나리오

Claude Code Action v1은 PR 코멘트, 자동 CI 실패 수정, 코드 리뷰 게시 등을 위한 자동화된 GitHub Actions 에이전트로 실행될 수 있습니다. [Claude](https://code.claude.com/docs/en/github-actions)

```yaml
# PR 피드백 루프 예시
name: Claude Code Feedback Loop
on:
  pull_request:
  issue_comment:
    types: [created]

jobs:
  claude-review:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          # 테스트 실패 → Claude가 분석 → 수정 → 재검증
```

---

## 피드백 루프 위치별 적합한 도구

| 피드백 루프 위치      | 적합한 방법                             | GitHub Actions 필요? |
| -------------- | ---------------------------------- | ------------------ |
| 툴 실행 전/후 검증    | **Hooks** (PreToolUse/PostToolUse) | ❌                  |
| 자기검증 루프        | **멀티 에이전트** (생성+평가 분리)             | ❌                  |
| PR/커밋 후 CI 피드백 | **GitHub Actions**                 | ✅                  |
| 장기 세션 컨텍스트 관리  | **Context Reset + Compaction**     | ❌                  |
| 보안/가드레일 강제     | **Guardrail Engine (Go binary)**   | ❌                  |

---

## 실제 권장 구조 (조합)

```
[로컬 세션 내부]
PreToolUse Hook → 가드레일 검사
    ↓
Claude 실행 (생성 에이전트)
    ↓
PostToolUse Hook → 자기검증 (평가 에이전트)
    ↓
컨텍스트 압축 / 리셋

[레포지토리 반영 후]
GitHub Actions → 테스트/린트/배포
    ↓
실패 시 @claude 코멘트 → Claude Code가 자동 수정
    ↓
다시 CI 실행 (외부 피드백 루프)
```

---

## 요약

```
🔁 에이전트 내부 피드백  → Hooks + 멀티에이전트 (Self-Verification)
🔁 에이전트 외부 피드백  → GitHub Actions (CI/CD)
🔁 장기 세션 피드백      → Context Reset + Compaction

GitHub Actions는 외부 피드백 루프에서 강력하지만
내부 루프는 Hooks와 멀티에이전트 패턴이 더 적합합니다.
두 가지를 조합하는 것이 가장 완성도 높은 하네스입니다.
```

더 구체적으로 어느 레이어의 피드백 루프를 구축하려는지 알려주시면 맞춤 설계를 도와드릴게요! 😊