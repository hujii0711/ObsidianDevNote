
GitHub Actions 워크플로우를 실행(트거)시키는 주요 이벤트들은 리포지토리 내부의 코드 변경, 외부 웹훅, 수동 실행, 일정 예약 등 다양한 상황에 따라 분류할 수 있습니다.

자주 사용되는 핵심 이벤트와 주요 이벤트 목록은 다음과 같습니다.

### 1. 가장 흔하게 쓰이는 핵심 이벤트

- **`push`**: 특정 브랜치에 코드를 푸시하거나 태그를 생성할 때 발생합니다. (예: `main` 브랜치에 푸시될 때 빌드 및 테스트 수행)

- **`pull_request`**: 풀 리퀘스트(PR)가 생성(`opened`), 업데이트(`synchronize`), 닫힘(`closed`) 등의 상태 변화가 생길 때 발생합니다.

- **`schedule`**: 리눅스 크론(Cron) 구문을 이용해 정해진 시간에 반복적으로 실행되도록 합니다. (예: 매일 자정에 보안 검사 수행)

- **`workflow_dispatch`**: GitHub 웹 UI의 Actions 탭에서 사용자가 직접 'Run workflow' 버튼을 눌러 **수동으로 실행**할 수 있게 합니다. 실행 시 입력값(`inputs`)을 받을 수도 있습니다.

### 2. 이슈 및 풀 리퀘스트 상호작용 이벤트

- **`issues`**: 이슈가 생성되거나, 편집되거나, 레이블이 달리거나, 닫히는 등의 활동이 있을 때 발생합니다.

- **`issue_comment`**: 이슈나 풀 리퀘스트에 댓글이 작성될 때 발생합니다. (봇 자동화나 특정 명령어 처리에 유용)

- **`pull_request_review` / `pull_request_review_comment`**: PR에 리뷰가 등록되거나 리뷰 댓글이 달릴 때 발생합니다.

- **`pull_request_target`**: `pull_request`와 유사하지만, 포크(Fork)된 저장소의 PR에 대해 안전하게 기본 브랜치의 권한과 시크릿을 참조해야 할 때 사용합니다.

### 3. 리포지토리 관리 및 배포 이벤트

- **`release`**: 새로운 릴리스가 발행(Published)되거나 수정·삭제될 때 발생합니다. (예: 빌드 결과물을 자동 업로드)

- **`create` / `delete`**: 새로운 브랜치나 태그가 생성되거나 삭제될 때 발생합니다.
  
- **`fork`**: 다른 사용자가 내 리포지토리를 포크(Fork)했을 때 발생합니다.

- **`star`**: 리포지토리에 스타(Star)가 추가되거나 취소될 때 발생합니다.

- **`deployment` / `deployment_status`**: 외부 API 등을 통해 배포가 요청되거나 배포 상태가 업데이트될 때 발생합니다.

### 4. 워크플로우 연동 이벤트

- **`workflow_run`**: 다른 워크플로우가 완료(`completed`)되었을 때 이를 연쇄적으로 트리거합니다. (예: 테스트 워크플로우가 성공적으로 끝난 뒤 배포 워크플로우 실행)

- **`workflow_call`**: 다른 워크플로우 내부에서 재사용 가능한 워크플로우(Reusable Workflow)로 호출될 때 작동합니다.

- **`repository_dispatch`**: GitHub 외부의 서드파티 서비스나 API 요청을 통해 웹훅(Repository Dispatch API)을 받아 커스텀 이벤트로 실행합니다.

### 💡YAML 설정 예시

워크플로우 파일(`on` 키워드)에서 이와 같은 이벤트들을 조합하거나 필터링(브랜치 지정 등)하여 활용할 수 있습니다.

YAML

```
on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '0 0 * * *'
  workflow_dispatch:
```