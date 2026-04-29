### 1. Devkit
- [ ] 서버버전에 codebuilder ctrl+F 기능 반영
- [ ] studioAPI.js push xml 비활성화시 안내 메시지 출력
- [ ] 2차 임베딩 색인하여 배포 또는 관리자 페이지에서 업데이트 하는 방안 마련
- [x] documents와 Websquare API 나눠서 색인하는 관리자 페이지 필요
- [x] devkit-server 일본어 서비스 동기화 [우선]
- [ ] 검색 상세페이지 랜더링 안됨, 동영상 경로 못찾음: vscode에서만 발생하는 현상


### 2. PaaS
- [ ] ==웹스퀘어 엔진 체크 및 등록 화면 개발 (안함)==
- [ ] settings theme 변경시 editor, panel 모두 즉시 반영되도록 조치
- [ ] ==SMS 크래시 발생시 서버 죽지 않게 PM2 적용 (안함)==
- [ ] ==SMS 로그 정보 쌓기 (안함)==
- [ ] ==spring boot 확장팩 자동 설치(내부망 설치 여부 확인) (안함)==
- [x] SMS 랜덤 포트 전환  [우선]
- [ ] vscode개발환경에서 xml 편집하고 wpack 돌릴수 있게 작업환경 개선 workspace 유동적 지정
- [ ] ai agent 로딩 지연 막기 [우선]
- [ ] 이미 띄워진 xml 탭에도 webviewTarget add
- [ ] 특정 PC의 사용자 jetty 서버 프로젝트 디렉터리 404 문제 있음
- [ ] w-pack 콘솔 사용자 vscode의 console에 출력되도록 변경 [우선]
- [x] Portable 빌드시 IWD build 포함
- [ ] ws.codeworkspace untracked하게
- [ ] 맥 빌드 starup.sh시 오류 있음
- [ ] ==vscode인지 이클립스인지 체크하는 공통함수 추가 (안함)==
- [x] production일때 랜덤 포트 경로 체크 [우선]
- [x] devkit 서버 버전 vscode에서 띄우기 [우선]
- [x] ws-ext에서 production인지 development인지 환경 변수 적용 [우선]
- [x] sms 랜덤 포트 전환시 운영에서는 같은 server.runtime.json을 바라보므로 문제가 방생함. Sms hash키를 가지고 있어야함 [우선]
- [x] vsix에 SMS와 AI_AGENT 포함하기 [우선]
- [x] JDK 21(LTS) 버전으로 내장하기 [우선]
- [ ] panel 리로드 기능


---
### 3. Tip
- [ ] docs와 CLAUDE.md가 젤 중요하다고 함(실밸개발자) -> lazy loading으로 다른 .md 참조할 수 있게 해야 함
- [ ] compact 40%로 설정하기(~/.claude/settings.json: CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "40")