

||`/rewind`|`/resume`|
|---|---|---|
|**범위**|**같은 세션 안에서** 시점을 되감기|**다른 세션(과거 대화)을** 다시 불러오기|
|**되돌리는 대상**|대화 히스토리 + 코드 파일 변경 (체크포인트 단위)|세션 전체(메시지, 컨텍스트)를 그대로 재개|
|**비유**|Ctrl+Z (특정 턴 시점으로 롤백)|어제 닫았던 대화를 다시 열기|

### `/rewind` — 세션 내부 되감기
Claude Code는 사용자가 프롬프트를 입력할 때마다 자동으로 **체크포인트**를 저장합니다. `/rewind`(또는 입력창이 비어있을 때 Esc Esc)를 실행하면 체크포인트 메뉴가 열리고, 다음 중 선택해서 되돌릴 수 있습니다.

- **대화만 되돌리기** (conversation only)
- **코드만 되돌리기** (code only)
- **둘 다 되돌리기**

Claude의 리팩터링이 잘못된 방향으로 갔을 때 유용합니다. 단순히 파일만 수동으로 되돌리면 Claude는 여전히 "잘못된 결정"을 컨텍스트에 기억하고 있어서 계속 그 위에 쌓아 올리는데, `/rewind`는 대화 기록 자체를 되돌려서 그 문제를 해결합니다.

**주의**: 체크포인트는 Claude Code의 파일 편집 도구로 변경된 파일만 추적합니다. `sed`, `mv`, `git checkout` 같은 Bash 명령으로 바뀐 파일은 추적되지 않아 되돌릴 수 없습니다.

```
VSCode Claude Code 확장에서는 CLI의 `/rewind` 명령어 자체가 그대로 있는 게 아니라, 새 창이 열리는 방식이 아니라 **메시지에 마우스를 올리면 나타나는 rewind 버튼**을 통해 되감기를 합니다.

옵션은 3가지입니다:

- **Fork conversation from here**: 해당 시점에서 새 대화 브랜치를 시작 (코드 변경사항은 유지)
- **Rewind code to here**: 대화 기록은 그대로 두고 파일 변경사항만 해당 시점으로 되돌림
- **Fork conversation and rewind code**: 둘 다 (새 브랜치 + 코드 되돌리기)
```

### `/resume` — 세션 간 이동
저장된 과거 세션(대화 전체)을 다시 불러옵니다. 세션 ID나 이름으로 지정하거나, 인자 없이 실행하면 최근 세션 목록에서 선택하는 피커가 뜹니다. 어제 하다 만 작업을 오늘 이어서 할 때 씁니다.

#### resume 보충

`/resume`는 **현재 작업 디렉터리에 속한 세션만**, 그리고 **<font color="#ff0000">최근 30일 이내 활동</font>한 것까지** 재개할 수 있다.

<이 환경의 실제 상태>
트랜스크립트는 `~/.claude/projects/<디렉터리-슬러그>/<session-id>.jsonl`로 저장된다.

```
c:\websquare-ai-agent  →  C:\Users\hujii\.claude\projects\c--websquare-ai-agent\
```

이 프로젝트에 남아 있는 세션은 **5개, 전부 오늘(2026-08-28)** 것이다.

|세션 ID|크기|마지막 활동|
|---|---|---|
|`ae974957…` (현재)|711 KB|17:33|
|`af052a2d…`|977 KB|17:17|
|`dc14c5a5…`|524 KB|17:17|
|`dd7b177f…`|416 KB|16:56|
|`038dedb5…`|78 KB|17:30|

전체 계정으로 보면 프로젝트 11개에 세션 83개가 있고, **가장 오래된 것이 2026-07-31 — 정확히 28일 전**이다. 30일 보관 정책과 맞아떨어진다. 그 이전 세션은 이미 삭제됐다.


<보관 기간을 정하는 값>
[settings.json](vscode-webview://0kn1upfq0pgov6joeqae9ant0ihtbuu2v197965giu348gn7f4ei/index.html?id=0dc32cfd-4710-4255-a2bd-a0f29fe4bd95&parentId=1&origin=19aecf67-0796-4945-bb0f-05ebbfffa526&swVersion=6&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&session=ae974957-e0da-4acf-9de3-cdba4d9f5181)에 `cleanupPeriodDays`가 없다 → **기본값 30일** 적용. 마지막 활동일 기준이므로, 오래 전에 시작했더라도 최근에 이어서 쓴 세션은 계속 남는다.

```json
{
  "model": "opus[1m]",
  "theme": "dark",
  ...
}
```

<두 가지 제약>
**1. 디렉터리가 다르면 안 보인다.** `/resume`은 `cwd` 슬러그로 폴더를 찾는다. `c:\websquare-ai-agent`가 아닌 곳에서 실행하면 이 5개는 목록에 안 나온다. 프로젝트를 옮기거나 이름을 바꾸면 슬러그가 달라져 기존 세션과 끊긴다.

**2. 30일이 지나면 파일 자체가 사라진다.** 복구 수단은 없다.

늘리려면
```json
"cleanupPeriodDays": 90
```

를 `settings.json`에 추가하면 된다. 다만 이미 삭제된 7월 31일 이전 세션은 돌아오지 않고, **지금부터** 적용된다. 세션 파일이 500KB~1MB씩 되므로 기간을 크게 늘리면 디스크 사용량도 그만큼 늘어난다.

참고로 세션 ID를 알면 목록 없이 바로 붙을 수 있다 — `claude --resume ae974957-e0da-4acf-9de3-cdba4d9f5181`. 직전 세션은 `claude --continue`.


### 언제 뭘 쓰나

1) rewind
Claude가 방금 잘못된 방향으로 코드를 고쳤을 때같은 세션 안에서 특정 시점으로 되돌리고 싶을 때 사용합니다.

/rewind 입력 (또는 입력창이 비어있을 때 Esc Esc)
되돌릴 체크포인트(턴) 선택
대화만 / 코드만 / 둘 다 되돌릴지 선택

2) resume
어제 하던 작업을 오늘 이어서 할 때다른 세션(과거 대화)을 그대로 다시 불러오고 싶을 때 사용합니다.
/resume 입력 후 목록에서 선택
또는 claude --resume <session-id>로 바로 진입
이름을 지정해뒀다면 --resume <name>으로도 접근 가능

한 줄로 요약하면: **`/rewind`는 "타임머신"(같은 세션 내 되감기), `/resume`는 "책갈피"(다른 날의 세션을 다시 펴기)**입니다.