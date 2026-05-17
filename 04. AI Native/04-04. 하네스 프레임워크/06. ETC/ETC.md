
### 1. Tip
- @: 특정 파일을 지정해 컨텍스트에 포함하여 클로드에 질문하거나 대화할 수 있다.
- 대화 모드에서 Tab 키를 누르는 것만으로 사고 확장 모드 기능이 추가된다.
- gh 명령어를 사용하면 GitHub의 원격 레포지터리를 만들거나 관리할 수 있고, 이슈 생성, 풀 리퀘스트 확인, 리뷰 GitHub Page에 배포 등도 브라우저를 열지 않고 터미널에서 바로 처리할 수 있다.
- 클로드 코드는 세션 단위로 대화를 기억한다. 클로드는 실행된 작업 디렉터리 단위로 세션을 기록하여, $HOME/.claude/projects 디렉터리에 저장한다.
- makefile: 프로그램의 빌드를 자동화하기 위한 설정 파일이다. 확장자 없는 Makefile에 bash 기반의 명령을 정의할 수 있다. init, dev, build, deploy 등의 명령 모음이다.
- 프로젝트 관리 명령어들은 클로드가 다른 세션에서 잊을 수 있으므로, CLAUDE.md 파일에 기록해두는 것이 좋다.
- 클로드 코드의 기능 확장 두가지 방법: MCP, 스킬
- 클로드 코드를 헤드리스 모드로 사용할 수 있다. 이는 대화 모드가 아니라 일회성 처리만 수행한다. 이 방식을 이용하면 대용량 파일을 분석하거 내용을 자동을 점검하게 할 수 있다.
- 복잡한 애플리케이션일수록 작업을 분해하는 반복 접근 방식을 사용하고, 그 과정에서 자기 회고와 개선을 유도하는 방식이 정확도가 높은 것으로 나타나고 있다. 그 대신 반복 접근 방식은 구현의 오버헤드가 크고, 작업을 얼마나 세분화할지 조정하는 것이 문제가 되는 것 보이고 있다.

### 2. 커스텀 슬래시
클로드 코드에 대한 프롬프트를 명령어로 만들 수 있다.
커스텀 슬래시 명령을 통해 여러 개의 지시를 하나로 묶어 슬래시 명령으로 미리 정의해둘 수 있다.
자주 반복하는 지시를 할 때 사용하면 유용하다.

- 커스텀 슬래시 명령 만드는 방법
커스텀 슬래시 명령은 project(~)/.claude/commands/CUSTOM_HOOK_COMMAND.md 파일을 생성하여 만든다.
- 커스텀 슬래시 명령에 인자 전달하기
커스텀 슬래시 명령에는 인자를 전달할 수 있으며, 프롬프트 내에서 $ARUMENTS라고 기술하면 명령을 호출할 때 특정 값을 제공할 수 있다.
ex.) pre-review 라는 슬래시 명령을 만들고 마크다운 파일에 다음과 같이 기재해둘 수 있다.
```markdown
풀 리퀘스트 #$ARUMENTS를 검토하세요.
```

```terminal
/pre-review 14
```

### 3. 나만의 슬래시 명령어 만들기

프로젝트 루트에 `.claude/commands/` 폴더를 만들고 `.md` 파일을 넣으면, 파일 이름이 곧 슬래시 명령어가 됩니다.

```
.claude/
  commands/
    review_script.md    →  /review_script
    deploy_staging.md   →  /deploy_staging
```

> 💡 [claude.md](http://claude.md)의 트리거 키워드와 비슷하지만, 슬래시 명령어는 **탭 자동완성**이 되고 Claude Code가 **구조적으로 인식**하기 때문에 더 안정적입니다.

- supabase
백엔드 통째로 제공 (인증 + DB + 스토리지). 무료 플랜 있음. 인증 붙이면서 DB도 같이 세팅됨.

- ! 접두사로 bash 명령 실행
Claude 프롬프트에서 `!`를 붙이면 대화를 끊지 않고 터미널 명령을 바로 실행할 수 있습니다.
```bash
!npm run build
!git status
!ls -la src/
```

> 안 쓰는 MCP는 `/mcp`에서 비활성화
### 4. 명령어 Quick Reference
- `/init` [claude.md](http://claude.md) 자동 생성
- `/clear` 컨텍스트 초기화 (새 작업 시작 시)
- `/compact` 컨텍스트 압축 (맥락 유지)
- `/context` 토큰 사용량 확인
- `/models` 모델 전환 (Opus / Sonnet / Haiku)
- `/resume` 이전 세션 복구
- `/mcp` MCP 관리
- `/export` 채팅 내보내기

### 5. 다른 AI에게 비평 받기

Claude와 작업하다가 막히면, `/export`로 대화를 내보내서 ChatGPT나 Gemini에게 보여주세요.

> "이 대화를 분석해서, Claude가 놓치고 있는 것이나 잘못된 접근이 있으면 지적해줘"  
> ![Pasted image 20260419134741.png](app://833c599d816d68a5dfe5a2b0c9c10cc381dd/Users/fujii0711/Documents/Obsidian/DevNote/06.%20Link%20Images/Pasted%20image%2020260419134741.png?1776574061785)

💡 **자동화 팁**: Custom Skill을 만들어 `/review-with-gpt` 같은 명령으로 이 과정을 원커맨드로 처리할 수 있습니다.

### 6. Escape
🛑 `Escape` **→ 즉시 중단**
잘못된 방향으로 가고 있다 싶으면 망설이지 말고 바로 Escape!
↩️ `Escape × 2` **→ 입력 삭제 / 복원**
- **텍스트가 있을 때** → 입력 내용 삭제
- **입력창이 비었을 때** → 이전 입력 복원

### 7. Thinking 로그 읽기
Claude가 생각하는 과정을 보여주는 **thinking 로그**를 무시하지 마세요.  
![Pasted image 20260419134700.png](app://833c599d816d68a5dfe5a2b0c9c10cc381dd/Users/fujii0711/Documents/Obsidian/DevNote/06.%20Link%20Images/Pasted%20image%2020260419134700.png?1776574020760)

🛑 Claude가 잘못된 가정을 하고 있다면, 그 순간 **Escape**로 중단하세요. 잘못된 가정 위에 쌓인 코드는 전부 쓸모없습니다. **초반에 잡는 게 핵심**입니다.

### 8. 멀티 인스턴스 운영
터미널 탭 여러 개를 열고 **각 탭에서 다른 Claude 인스턴스**를 돌리세요. 탭 이름을 바꿔두면 한눈에 관리 가능합니다.  
![Pasted image 20260419160707.png](app://833c599d816d68a5dfe5a2b0c9c10cc381dd/Users/fujii0711/Documents/Obsidian/DevNote/06.%20Link%20Images/Pasted%20image%2020260419160707.png?1776582427983)

```
탭 1: "Feature-Auth" — 인증 기능 개발
탭 2: "Bug-Fix" — 버그 수정
탭 3: "Refactor" — 리팩토링
```

### 9. 음성 입력 (/voice)
`/voice`로 음성 입력 모드 활성화. **Push-to-talk** (스페이스바) 방식이며 한국어 포함 20개 이상 언어를 지원합니다. 복잡한 요구사항을 키보드로 5분 걸릴 프롬프트를 말로 1분에 전달할 수 있습니다.

### 10. 컨텍스트 관리 체크리스트
- `/memory`로 Second Brain 구축 (개인 메모리)
- [CLAUDE.md](http://claude.md/)에는 참조만, 상세 내용은 별도 파일로 (Lazy Loading)
- **한 세션 = 한 피처** 원칙 준수
- `/context`로 토큰 사용량 주기적 확인
- 안 쓰는 MCP는 `/mcp`에서 비활성화
- Mermaid로 아키텍처 정리
- 무거운 데이터 처리는 스크립트로 오프로드

### 11. 워크플로우 체크리스트
- 큰 작업은 **Plan Mode**(`Shift+Tab`)부터
- **작은 변경 → 테스트 → 린트 → 커밋** 루프 유지
- Thinking 로그 확인, 잘못된 가정은 **Escape**로 즉시 중단
- 막히면 `/export` → 다른 AI에게 비평 받기
- 에러 로그는 **해석 없이 그대로** 붙여넣기
- `TODO.md`로 작업 연속성 유지
- 복잡한 프로젝트는 **WAT 프레임워크** 적용