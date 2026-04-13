## 최초 등록

#### 1. 로컬 저장소 초기화
```bash
git init
```

#### 2. 모든 파일 추가 (주의: .obsidian/workspace.json 등은 제외하는 것이 좋습니다)
```bash
git add .
```

#### 3. 첫 번째 커밋
```bash
git commit -m "Initial commit"
```
#### 4. 브랜치 이름을 main으로 변경
```bash
git branch -M main
```

#### 5. PAT 발급후 원격 저장소로 업로드

1)  **토큰 발급:** [GitHub Settings] > [Developer settings] > [Personal access tokens] > **[Tokens (classic)]**에서 토큰을 생성합니다. (`repo` 권한에 반드시 체크하세요.)

2) **원격 URL 수정:** 아래 명령어를 입력하여 URL에 토큰을 직접 포함시킵니다.
형식: git remote add origin https://<토큰>@github.com/<사용자명>/<레포지토리명>.git

```bash
git remote add origin https://<발급받은_토큰>@github.com/<사용자명>/<레포지토리명>.git

cf.)
git remote set-url origin https://<발급받은_토큰>@github.com/<사용자명>/<레포지토리명>.git
```

3.  다시 시도
```bash
git push -u origin main
```

## 이미 git repositoty 있는 상태에 새로운 PC에 이식


#### 1. 로컬 저장소 초기화
```bash
git init
```

#### 2. git remote

```bash
git remote add origin https://<발급받은_토큰>@github.com/<사용자명>/<레포지토리명>.git
```

#### 3. 참고

메시지를 보면 원격 저장소(`origin`)로부터 새로운 데이터들을 가져오는 것(fetch)까지는 성공했지만, 정작 내 로컬의 **master** 브랜치가 서버의 어떤 브랜치와 짝꿍인지 설정되어 있지 않아서 **"가져오긴 했는데, 어디에 합쳐줄까?"**라고 묻고 있는 상황입니다.

---

## 현재 상황 분석

1. **데이터 수신 성공:** `Receiving objects: 100%`를 통해 서버의 변경 사항은 내 PC로 넘어왔습니다.
    
2. **브랜치 연결 누락:** `There is no tracking information for the current branch`라는 문구는 현재 로컬의 `master` 브랜치가 서버의 `main` 브랜치(혹은 master)와 연결(Tracking)되어 있지 않다는 뜻입니다.
    
3. **병합 중단:** 연결 정보가 없으니 Git이 자동으로 합쳐주지 않고 작업을 멈춘 것입니다.
    

---

## 해결 방법

가장 깔끔한 해결책은 터미널에 제안된 대로 **로컬 master 브랜치를 서버의 main 브랜치와 연결**해 주는 것입니다. (요즘 GitHub은 기본 브랜치명이 `main`이므로, 서버에 `main`만 있다면 아래 명령어를 입력하세요.)

### 1. 브랜치 연결 설정 (추천)

이 명령어를 한 번 실행하면, 다음부터는 그냥 `git pull`만 입력해도 알아서 동작합니다.

Bash

```
git branch --set-upstream-to=origin/main master
```

_만약 서버의 브랜치 이름이 `master`라면 끝을 `origin/master master`로 바꿔주세요._

### 2. 연결 없이 이번만 당겨오기

연결 설정을 건너뛰고 지금 당장 합치고 싶다면 대상을 직접 지정합니다.

Bash

```
git pull origin main
```

---

## 💡 참고: GitHub의 브랜치명 변화

메시지에 `[new branch] main -> origin/main`이라고 뜨는 것을 보니, 원격 저장소의 기본 브랜치가 `main`으로 설정되어 있는 것 같습니다. 로컬 브랜치 이름이 `master`라면 이름이 서로 달라 생기는 혼선일 수 있으니, 위 1번 명령어로 관계를 맺어주시는 것이 가장 좋습니다.

연결 후에 다시 `git pull`을 해보시면 "Already up to date" 혹은 실제 파일이 업데이트되는 메시지를 보실 수 있을 거예요.