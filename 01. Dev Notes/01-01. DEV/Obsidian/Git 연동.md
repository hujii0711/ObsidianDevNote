
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