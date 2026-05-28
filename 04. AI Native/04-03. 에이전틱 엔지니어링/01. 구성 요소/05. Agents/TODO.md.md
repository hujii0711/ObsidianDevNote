### 1. 로컬 TODO.md 활용
![Pasted image 20260419135326.png](app://833c599d816d68a5dfe5a2b0c9c10cc381dd/Users/fujii0711/Documents/Obsidian/DevNote/06.%20Link%20Images/Pasted%20image%2020260419135326.png?1776574406441)  
프로젝트 루트에 `TODO.md`를 만들어 Claude에게 태스크를 추적하게 하세요.

```markdown
- [ ] 결제 기능 구현 (Stripe 연동)
- [ ] 랜딩 페이지 CTA 수정
- [ ] 구독 시스템 설계
- [ ] 버그 #1: 로그인 리다이렉트 오류
- [ ] 버그 #2: 모바일 레이아웃 깨짐
```

![[Pasted image 20260419135507.png]]

> **실전 워크플로우:**
1. 하루 시작 — 할 일을 [TODO.md](http://todo.md/)에 체크리스트로 작성
2. Claude에게 **"[TODO.md](http://todo.md/) 읽고 첫 번째 항목부터 시작해"** 지시
3. **Agent Teams**로 여러 태스크를 **병렬 처리** 가능
4. 세션 종료 시 **"[TODO.md](http://todo.md/) 업데이트해줘"** → 진행 상황 자동 반영

💡여러 세션에 걸쳐 **작업의 연속성**을 유지하는 핵심 도구입니다.