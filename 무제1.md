Claude Code CLI에서 `-p` (또는 `--print`) 플래그는 **비대화형(Non-interactive) 모드**, 즉 **'프린트 모드(Print Mode)'로 명령을 실행하고 즉시 종료할 때** 사용합니다.

기본적으로 `claude`를 실행하면 AI와 계속 대화할 수 있는 터미널 창(REPL 환경)이 열리지만, `-p`를 붙이면 단발성 명령을 처리한 후 결과만 화면에 출력(Print)하고 세션이 바로 종료됩니다.

주로 다음과 같은 3가지 상황에서 강력한 위력을 발휘합니다.

---

## 1. 리눅스 파이프(`|`) 및 다른 명령어와 조합할 때 (가장 중요)

Unix 철학에 맞게 다른 명령어의 출력 결과를 Claude에게 전달하여 분석하거나 가공할 때 사용합니다.

Bash

```
# 1. 최근 로그 뒷부분 200줄을 읽어서 에러가 있는지 분석 요청
tail -n 200 app.log | claude -p "이 로그에서 이상 징후나 에러가 있으면 찾아줘"

# 2. Git diff 내용을 넘겨서 보안 취약점 리뷰 요청
git diff main --name-only | claude -p "변경된 파일들의 보안상 취약점을 리뷰해줘"
```

## 2. CI/CD 파이프라인 및 스크립트 자동화에 활용할 때

GitHub Actions, GitLab CI/CD 스크립트나 로컬 자동화 셸 스크립트 내에서 Claude를 에이전트로 활용할 때 필수적입니다. 인간의 키보드 입력(인터랙션)을 기다리지 않아야 하기 때문입니다.

Bash

```
# CI 파이프라인 내에서 새로 추가된 번역 문자열을 자동 번역하고 결과를 받아볼 때
claude -p "새로 추가된 영어 문자열을 한국어로 번역해서 JSON 파일에 업데이트해줘"
```

## 3. 터미널에서 즉시 한 번만 묻고 답을 얻고 싶을 때

Claude Code의 `-p` 플래그에 대해 설명해 드릴게요.## `claude -p` 명령어 (`--print` 플래그)

`-p`는 `--print`의 단축키로, **비대화형(non-interactive) 모드**로 Claude Code를 실행하는 핵심 플래그입니다.

### 기본 동작

`--print` / `-p`는 쿼리를 보내고 대화형 세션을 시작하지 않고 바로 종료합니다. 결과를 stdout으로 출력하고 빠져나옵니다.

```bash
# 기본 사용법
claude -p "이 코드 설명해줘"

# 결과를 파일로 저장
claude -p "저장소 구조 설명해줘" > overview.txt
```

### 주요 활용법

**파이프라인 연결**

```bash
# 파일 내용을 파이프로 넘기기
cat error.log | claude -p "무슨 에러인지 설명하고 해결책 알려줘"

# Git diff 리뷰
git diff | claude -p "변경사항 검토하고 문제점 알려줘"
```

**스크립트 자동화**

```bash
# JSON 출력 + 최대 턴 수 제한
claude -p "보안 취약점 검토해줘" \
  --output-format json \
  --max-turns 5 \
  --permission-mode bypassPermissions
```

**여러 파일 일괄 처리**

```bash
for f in src/components/*.tsx; do
  claude -p "$f 파일에 접근성 속성 추가해줘"
done
```

### 함께 쓰면 유용한 옵션들

|옵션|설명|
|---|---|
|`--output-format json`|구조화된 JSON으로 출력 (비용, 소요시간 포함)|
|`--output-format text`|일반 텍스트 출력|
|`--max-turns N`|최대 반복 횟수 제한 (자동화 시 비용 제어)|
|`--permission-mode`|권한 모드 설정 (`default`, `acceptEdits`, `bypassPermissions`)|

### 핵심 용도

스크립팅에 필수적인 플래그로, Claude Code를 비대화형으로 실행하고 결과를 stdout으로 출력한 뒤 종료합니다. 쉘 파이프라인이나 cron 작업에 Claude Code를 연결해 자동화 파이프라인을 구성하는 데 사용됩니다.

간단히 말해, **터미널을 열지 않고 한 번만 실행하고 끝내고 싶을 때** 쓰는 플래그입니다. CI/CD, 배치 작업, 스크립트 자동화에 특히 유용합니다.

---
---

대화형 창을 띄우지 않고, 마치 `curl`이나 일반 CLI 도구를 쓰듯 특정 함수나 에러 메시지에 대한 피드백만 빠르게 화면에 출력하고 싶을 때 사용합니다.

Bash

```
# 특정 함수에 대한 빠른 설명 요구 후 종료
claude -p "auth.ts 파일에 있는 generateToken 함수가 어떻게 작동하는지 요약해줘"
```

---

### 💡 꿀팁: 이전 대화 이어가기 (`-c` 플래그와의 조합)

`-p` 모드는 한 번 실행하면 끝나지만, 바로 이전에 했던 대화 맥락을 이어서 프린트 모드로 명령을 내리고 싶다면 `-c` (`--continue`) 플래그와 함께 사용할 수 있습니다.

Bash

```
# 이전 세션의 컨텍스트를 유지한 채로, 테스트 실행 명령만 내리고 종료하기
claude -c -p "프로젝트의 전체 테스트 코드를 실행하고 실패 원인을 요약해줘"
```

Claude Code의 `-p` 플래그에 대해 설명해 드릴게요.## `claude -p` 명령어 (`--print` 플래그)

`-p`는 `--print`의 단축키로, **비대화형(non-interactive) 모드**로 Claude Code를 실행하는 핵심 플래그입니다.

### 기본 동작

`--print` / `-p`는 쿼리를 보내고 대화형 세션을 시작하지 않고 바로 종료합니다. 결과를 stdout으로 출력하고 빠져나옵니다.

```bash
# 기본 사용법
claude -p "이 코드 설명해줘"

# 결과를 파일로 저장
claude -p "저장소 구조 설명해줘" > overview.txt
```

### 주요 활용법

**파이프라인 연결**

```bash
# 파일 내용을 파이프로 넘기기
cat error.log | claude -p "무슨 에러인지 설명하고 해결책 알려줘"

# Git diff 리뷰
git diff | claude -p "변경사항 검토하고 문제점 알려줘"
```

**스크립트 자동화**

```bash
# JSON 출력 + 최대 턴 수 제한
claude -p "보안 취약점 검토해줘" \
  --output-format json \
  --max-turns 5 \
  --permission-mode bypassPermissions
```

**여러 파일 일괄 처리**

```bash
for f in src/components/*.tsx; do
  claude -p "$f 파일에 접근성 속성 추가해줘"
done
```

### 함께 쓰면 유용한 옵션들

|옵션|설명|
|---|---|
|`--output-format json`|구조화된 JSON으로 출력 (비용, 소요시간 포함)|
|`--output-format text`|일반 텍스트 출력|
|`--max-turns N`|최대 반복 횟수 제한 (자동화 시 비용 제어)|
|`--permission-mode`|권한 모드 설정 (`default`, `acceptEdits`, `bypassPermissions`)|

### 핵심 용도

스크립팅에 필수적인 플래그로, Claude Code를 비대화형으로 실행하고 결과를 stdout으로 출력한 뒤 종료합니다. 쉘 파이프라인이나 cron 작업에 Claude Code를 연결해 자동화 파이프라인을 구성하는 데 사용됩니다.

