
Git을 사용할 때 가장 헷갈리는 부분이 바로 파일의 **상태(State)**와 **영역(Area)**입니다. 이 개념들을 명확히 구분하면 `git status` 메시지가 훨씬 쉽게 읽힐 거예요.

---

## 1. 파일의 4가지 상태 (File Status)

파일은 Git에 의해 관리되고 있는지(Tracked)와 수정되었는지에 따라 구분됩니다.

| **상태**           | **설명**                                      |
| ---------------- | ------------------------------------------- |
| **Untracked**    | Git이 아직 관리하지 않는 파일. 새로 만든 파일이 여기에 해당합니다.    |
| **Tracked**      | Git이 이미 알고 있는 파일. 수정 여부에 따라 아래 3가지로 나뉩니다.   |
| └ **Unmodified** | 마지막 커밋 이후로 수정되지 않은 상태.                      |
| └ **Modified**   | 파일이 수정되었지만, 아직 **Staging Area**에 올리지 않은 상태. |
| └ **Staged**     | 수정된 파일을 다음 커밋에 포함시키겠다고 찜해둔 상태.              |

---

## 2. Git의 3가지 영역 (Git Sections)

파일이 어디에 머물고 있는지를 나타내는 논리적 공간입니다.

### 🏠 Working Tree (Working Directory)

- **개념:** 현재 내가 실제로 코드를 짜고, 파일을 만들고, 수정하고 있는 **실제 폴더**입니다.
    
- **특징:** Git의 관리를 받지 않는 'Untracked' 파일과 수정 중인 'Modified' 파일들이 공존하는 곳입니다.
    

### 🚦 Staging Area (Index)

- **개념:** 커밋을 하기 전, **"이 파일들을 묶어서 스냅샷을 찍겠다"**고 준비하는 완충 지대입니다.
    
- **명령어:** `git add <file>`을 하면 파일이 이곳으로 이동(Staged)합니다.
    

### 📜 Git Directory (Repository)

- **개념:** 프로젝트의 메타데이터와 객체 데이터베이스가 저장되는 곳입니다. (보통 `.git` 폴더)
    
- **특징:** `git commit`을 하면 Staging Area에 있던 파일들이 하나의 버전으로 영구 저장됩니다.
    

---

## 3. 핵심 키워드 완벽 요약

- **Untracked vs Tracked**
    
    - **Untracked:** "넌 누구니?" (Git이 모르는 새 파일)
        
    - **Tracked:** "너 예전에 본 적 있어." (Git이 관리 중인 파일)
        
- **Uncommitted vs Committed**
    
    - **Uncommitted:** Working Tree나 Staging Area에 있는, 아직 '확정'되지 않은 수정 사항들.
        
    - **Committed:** `.git` 저장소에 안전하게 기록된 상태. (언제든 되돌릴 수 있음)
        
- **Staged vs Unstaged**
    
    - **Staged:** "다음 커밋 기차에 탈 준비 완료!" (`git add` 후)
        
    - **Unstaged (Modified):** "수정은 했는데 기차표는 아직 안 샀어." (수정 후 `git add` 전)
        

---

### 💡 꿀팁: 흐름으로 기억하세요!

1. **Working Tree**에서 열심히 코딩한다. (Untracked / Modified)
    
2. 의미 있는 작업이 끝나면 **Staging Area**에 올린다. (`git add` → Staged)
    
3. 준비된 파일들을 모아 **Repository**에 영구 저장한다. (`git commit` → Committed)
    

이제 `git status`를 입력했을 때 나오는 **"Changes to be committed"**(Staged 상태)와 **"Changes not staged for commit"**(Modified 상태)의 차이가 명확히 보이실 겁니다!

---

|               | Working Tree | Staging | Repository |
| ------------- | ------------ | ------- | ---------- |
| Untracked<br> | O            | X       | X          |
| Tracked       | X            | X       | O          |
| └ Unmodified  | O            | X       | X          |
| └ Modified    | O            | X       | X          |
| └ Staged      | X            | O       | X          |

|                    | Working Tree | Staging | Repository |
| ------------------ | ------------ | ------- | ---------- |
| Uncommitted<br>    | O            | O       | X          |
| Committed          | X            | X       | O          |
| Staged             | X            | O       | X          |
| Unstaged(Modified) | O            | X       | X          |

