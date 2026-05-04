GitHub CLI(`gh`)는 터미널에서 브라우저 없이 레포지토리 관리, PR 생성, 이슈 확인 등 GitHub의 핵심 기능을 수행할 수 있게 해주는 강력한 도구입니다. 운영체제별 설치 방법과 계정 연동 과정을 정리해 드립니다.

---

## 1. 운영체제별 설치 방법

### **macOS**

가장 간편한 방법은 Homebrew를 사용하는 것입니다.

Bash

```
brew install gh
```

### **Windows**

Windows 패키지 매니저인 `winget`이나 `scoop`을 사용하거나 공식 MSI 설치 파일을 이용할 수 있습니다.

PowerShell

```
# winget 사용 시
winget install --id GitHub.cli

# scoop 사용 시
scoop install gh
```

---

## 2. GitHub 계정 연동 (Authentication)

설치가 완료되었다면 터미널에서 아래 명령어를 입력하여 로그인을 진행합니다.

Bash

```
gh auth login
```

명령어를 실행하면 대화형 프롬프트가 나타납니다. 아래 순서대로 선택하세요.

1. **What account do you want to log into?** `GitHub.com` 선택
    
2. **What is your preferred protocol for Git operations?** `HTTPS` 또는 `SSH` 선택
    
    - _앞서 설정한 SSH 환경을 활용하고 싶다면 `SSH`를 추천합니다._
        
3. **Generate a new SSH key?** (SSH 선택 시) 기존 키가 있다면 `No`, 새로 만들려면 `Yes`
    
4. **How would you like to authenticate GitHub CLI?** `Login with a web browser` 선택
    
5. **화면에 나타난 8자리 One-time code를 복사한 뒤 Enter를 누릅니다.**
    
6. 브라우저 창이 열리면 코드를 입력하고 **Authorize github**를 클릭합니다.
    

---

## 3. 연동 확인 및 기본 설정

연동이 제대로 되었는지 확인하려면 다음 명령어를 입력합니다.

Bash

```
# 로그인 상태 확인
gh auth status

# 현재 계정의 레포지토리 목록 확인
gh repo list
```

### **유용한 초기 설정**

기본 에디터를 설정해두면 PR 메시지 등을 작성할 때 편리합니다. (예: VS Code로 설정)

Bash

```
gh config set editor "code --wait"
```

---

## 4. 자주 사용하는 명령어 한눈에 보기

|**기능**|**명령어**|
|---|---|
|**클론**|`gh repo clone <repository>`|
|**PR 생성**|`gh pr create --title "제목" --body "내용"`|
|**PR 목록**|`gh pr list`|
|**이슈 생성**|`gh issue create`|
|**레포지토리 브라우저로 열기**|`gh browse`|

GitHub CLI를 활용하면 브라우저를 왔다 갔다 하는 맥락 전환(Context Switching)을 줄일 수 있어 개발 생산성이 크게 향상됩니다.

혹시 현재 진행 중인 VS Code 익스텐션 개발 프로젝트에서 GitHub Actions나 이슈 관리 같은 특정 워크플로우에 CLI를 활용해보고 싶으신가요?