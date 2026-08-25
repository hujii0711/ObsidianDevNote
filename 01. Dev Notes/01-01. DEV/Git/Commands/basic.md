
git checkout -b <브랜치 이름>
git checkout -b alternate master
git checkout HEAD -- <파일 이름>: 작업 트리의 변경 사항 돌려 놓기
git checkout -m <기존 브랜치> <새로운 브랜치>: 새로운 브랜치가 존재하지 않은 경우 브랜치명 변경하기

git commit -a
git commit --amend
git commit -m "XXX" -amend: 마지막 커밋 고치기
git commit -m "XXX" -a: 수정되고 추적되는 모든 파일의 변경 사항 커밋하기
git commit -C HEAD -a --amend: 이전 커밋을 수정하고 커밋 메시지 재사용하기

git log -p: 변경 사항을 보여주는 패치와 함께 로그 표시하기
git log -2: 2개의 항목만 보이도록 로그 개수 제한하기
git log --word-diff
git log --stat
git log --name-only
git log --relative-date
git log --graph
git log --decorate -1
git log -1 HEAD~3: HEAD보다 세 개 이전의 커밋 로그 1개의 항목만 보기(HEAD^^, HEAD-1^^)
git log <시작 지점>..<끝 지점>
cf.) 시작 지점, 끝 지점: 커밋명, 브랜치명, 태그명

git push origin --all

git tag -l
git show-ref --tags

git rebase --continue
git rebase --skip
git rebase --abort
git rebase -i HEAD~~

git branch: 현재 브랜치 목록 보기
git branch -r: 원격 브랜치 목록 보기
git branch -a: 지역과 원격을 포함한 모든 브랜치 목록 보기
git branch <새로운 브랜치>: 현재 브랜치에서 새로운 브랜치 생성하기
git branch <새로운 브랜치> <브랜치를 생성할 위치>: 원격 브랜치에서 브랜치 생성하기
cf.) 브랜치를 생성할 위치: 다른 브랜치, 커밋명, 태그
git branch <생성할 브랜치명> <분기해 나올 브랜치명>
git branch -d <브랜치 이름>: 브랜치 생성과 체크아웃을 동시에 함
git branch -m <변경하려는 브랜치명> <새로 바꾸려는 브랜치명>

git config --global user.name "XXX"
git config --global user.email "XXX@XXX.com"
git config --global list
git config --global color.ui "auto"

git diff: 현재 작업 트리와 인덱스의 차이점 보기
git diff --cached: 인덱스와 저장소의 차이점 보기
git diff HEAD: 작업 트리와 저장소의 차이점 보기
git diff <시작지점>: 작업 트리와 특정 위치 간의 차이점 보기
cf.) 시작지점은 커밋명, 브랜치명, 태그명
git diff <시작지점> <끝지점>

git reset --hard HEAD^
git reset HEAD <파일>: 커밋되지 않고 스테이징된 변경 사항 재설정하기

git rebase -i HEAD~3

git merge --no-commit <브랜치>: 커밋하지 않고 합치기

git fetch: origin 저장소에서 합치지 않고 지역 브랜치로 변경 사항 가져오기
git fetch <원격 저장소>: 원격 저장소에서 합치지 않고 지역 브랜치로 변경 사항 가져오기



