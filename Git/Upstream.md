Git에서 **Upstream**은 두 브랜치 사이의 **'부모-자식' 같은 연결 관계**를 의미합니다. 로컬 브랜치가 원격 저장소의 특정 브랜치를 추적(Tracking)하도록 설정되었을 때, 그 원격 브랜치를 **Upstream**이라고 부릅니다.

---

## 1. Upstream의 주요 역할

Upstream이 설정되어 있으면 Git은 로컬 브랜치와 원격 브랜치의 상태를 비교할 수 있게 됩니다.

- **상태 비교:** 로컬이 원격보다 몇 개의 커밋이 앞서 있는지(Ahead), 혹은 뒤처져 있는지(Behind) 알려줍니다. (`git status` 시 확인 가능)
    
- **명령어 단축:** `git push`나 `git pull`을 할 때, 대상(origin)과 브랜치명(main)을 매번 입력하지 않아도 자동으로 연결된 곳으로 데이터를 보냅니다.
    
- **협업 흐름 유지:** 여러 명이 같은 브랜치에서 작업할 때, 내 로컬 환경이 서버의 최신 상태를 항상 인지하도록 돕습니다.
    

---

## 2. 현재 프로젝트의 Upstream 체크 방법

현재 내 로컬 브랜치들이 각각 어떤 원격 브랜치와 연결되어 있는지 확인하는 가장 확실한 방법들은 다음과 같습니다.

### A. 자세한 브랜치 정보 확인 (가장 추천)

Bash

```
git branch -vv
```

이 명령어를 입력하면 로컬 브랜치 목록 옆에 대괄호`[]`로 묶인 **원격 브랜치 이름**과 **상태(Ahead/Behind)**가 나타납니다.

- 예: `main abc1234 [origin/main] Add new feature`
    
    - 로컬 `main`의 Upstream이 `origin/main`임을 의미합니다.
        

### B. 현재 브랜치의 Upstream만 확인

Bash

```
git rev-parse --abbrev-ref @{u}
```

현재 체크아웃된 브랜치에 설정된 Upstream 브랜치 이름만 깔끔하게 출력합니다. 설정된 것이 없다면 에러가 발생합니다.

### C. 원격 저장소 설정 전체 확인

Bash

```
git remote show origin
```

`origin`이라는 원격 저장소의 상세 정보를 보여주며, 어떤 로컬 브랜치가 어떤 원격 브랜치를 push/pull 하도록 설정되어 있는지 리스트로 보여줍니다.

---

## 3. Upstream 설정 및 변경 방법

만약 Upstream이 설정되어 있지 않거나 바꾸고 싶다면 아래 명령어를 사용합니다.

- **처음 설정할 때 (Push와 동시에):**
    
    Bash
    
    ```
    git push -u origin main
    ```
    
    `-u` (또는 `--set-upstream`) 옵션을 주면 이후부터는 `git push`만 입력해도 됩니다.
    
- **이미 존재하는 브랜치에 설정만 할 때:**
    
    Bash
    
    ```
    git branch --set-upstream-to=origin/main main
    ```
    

---

**💡 참고 (Open Source 협업 시):** 오픈소스 프로젝트에서는 내가 포크(Fork)한 저장소를 `origin`, 원래 원본 저장소를 `upstream`이라는 **이름(Alias)**으로 등록해서 사용하기도 합니다. 이때의 `upstream`은 관습적인 원격 저장소의 이름일 뿐, 위에서 설명한 브랜치 간의 추적 관계(Upstream Tracking)와는 용어가 겹치지만 의미상 약간의 차이가 있습니다.

---
---
`git checkout`으로 브랜치를 이동하는 것과 `upstream` 설정 사이의 관계를 정리해 드릴게요. 결론부터 말씀드리면, **`checkout`만으로는 새로운 브랜치에 자동으로 `upstream`이 설정되지 않는 경우가 많습니다.**

상황에 따라 어떻게 동작하는지 나누어 설명하겠습니다.

---

### 1. 이미 존재하고 `upstream`이 설정된 브랜치로 이동할 때

만약 예전에 이미 `git push -u origin 브랜치명`을 수행했던 브랜치로 `checkout` 하는 것이라면, **기존의 연결 관계가 유지**됩니다.

- 이 상태에서는 `git push`나 `git pull`을 인자 없이 실행해도 원래 연결된 원격 브랜치와 통신합니다.
    

### 2. 원격에만 있는 브랜치를 처음 가져올 때

Bash

```
git checkout 원격브랜치명
```

원격 저장소에는 있지만 내 로컬에는 없는 브랜치를 이 명령어로 가져오면, Git은 똑똑하게도 **로컬 브랜치를 생성함과 동시에 원격 브랜치를 `upstream`으로 자동 설정**합니다.

- 이때는 별도 설정 없이 바로 `git pull/push`가 가능합니다.
    

### 3. 로컬에서 새로 만든 브랜치에서 이동할 때 (주의)

Bash

```
git checkout -b 새로운브랜치명
```

이렇게 로컬에서 새로 만든 브랜치는 **`upstream`이 없는 상태**입니다. 이 상태에서 `git push`를 하면 Git은 "어디로 보낼지 모른다"며 에러를 냅니다.

---

### 💡 `checkout` 후 `upstream` 관계 확인 및 설정 흐름

브랜치를 이동한 직후에 아래 흐름대로 확인해 보세요.

1. **현재 브랜치의 연결 상태 확인**
    
    Bash
    
    ```
    git status
    ```
    
    - 만약 `Your branch is up to date with 'origin/브랜치명'.` 메시지가 **없다면** 연결되지 않은 것입니다.
        
2. **연결되지 않았다면 첫 Push 때 설정**
    
    Bash
    
    ```
    git push -u origin 현재브랜치명
    ```
    
    - 한 번 이렇게 `-u` 옵션으로 밀어 넣으면, 다음부터는 이 브랜치로 `checkout` 할 때마다 자동으로 `upstream` 관계가 활성화됩니다.
        

---

### ⚠️ 주의할 점: "브랜치명"만 바꾼다고 해결되지 않음

`checkout`은 단순히 내가 작업할 "장소"를 옮기는 것일 뿐입니다.

- **로컬 A 브랜치**가 **원격 A 브랜치**를 바라보게 하려면, 그 브랜치에 '체크인'한 상태에서 반드시 `upstream` 설정( `-u` 옵션 등)이 기록되어 있어야 합니다.
    
- 만약 `upstream`이 설정되지 않은 브랜치에서 `git pull`을 하면 "There is no tracking information for the current branch"라는 메시지를 보게 됩니다.
    

**요약하자면:**

- **원격에서 받아온 브랜치:** `checkout` 시 자동으로 `upstream` 연결됨.
    
- **내가 새로 만든 브랜치:** 첫 `push` 때 `-u`를 써야 나중에 `checkout` 해와도 `upstream` 관계가 유지됨.
    

