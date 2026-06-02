#### RSC = React Server Components (리액트 서버 컴포넌트)

서버에서만 렌더링되는 React 컴포넌트입니다. 브라우저로 JavaScript 번들이 전송되지 않고, 서버에서 렌더링된 결과만 클라이언트로 보내집니다.

|구분|Server Component (RSC)|Client Component|
|---|---|---|
|실행 위치|서버에서만|서버(초기) + 브라우저|
|선언|기본값|파일 상단 `'use client'`|
|JS 번들|클라이언트로 전송 안 됨|전송됨|
|사용 가능|DB 직접 접근, 파일시스템, 비밀키|`useState`, `useEffect`, 이벤트 핸들러, 브라우저 API|
|사용 불가|`useState`, 이벤트 핸들러, `onClick` 등|서버 전용 리소스 직접 접근|
