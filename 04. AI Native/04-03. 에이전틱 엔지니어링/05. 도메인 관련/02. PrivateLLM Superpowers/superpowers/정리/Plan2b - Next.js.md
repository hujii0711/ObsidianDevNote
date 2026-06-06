
**Goal:** Plan 2A의 FastAPI `/chat` SSE 엔드포인트를 소비하는 Next.js 채팅 UI를 만든다. 사용자가 질문하면 답변 토큰이 실시간 스트리밍되고, 완료 시 출처 카드(`[n]` → 법령·판례 링크)와 면책 고지가 표시된다. 브라우저에서 데모 가능한 RAG 챗봇 완성이 산출물.

**Architecture:** `apps/web`(Next.js App Router + TypeScript + Tailwind). 핵심 로직은 순수·테스트 가능 단위로 분리: `lib/sse.ts`(SSE 프레임 파서), `lib/chatClient.ts`(fetch POST + 스트림 리더), `hooks/useChat.ts`(상태 머신). UI는 `components/*`(SourceCard, MessageBubble, ChatInput, Chat). 브라우저 EventSource는 GET만 지원하므로 **POST+SSE는 `fetch` + `ReadableStream` 리더**로 소비한다.

**Tech Stack:** Next.js 15(App Router), React 19, TypeScript, Tailwind CSS v4, Vitest + @testing-library/react + jsdom(단위·컴포넌트 테스트), npm.

```
apps/web/
├── package.json
├── tsconfig.json
├── next.config.ts
├── vitest.config.ts
├── vitest.setup.ts
├── postcss.config.mjs            # tailwind v4
├── .env.local.example            # NEXT_PUBLIC_API_BASE
├── app/
│   ├── layout.tsx
│   ├── page.tsx                  # Chat 컨테이너 마운트
│   └── globals.css               # tailwind import
├── lib/
│   ├── types.ts                  # ChatEvent, Source, Message
│   ├── sse.ts                    # SSEParser (순수, TDD)
│   └── chatClient.ts             # streamChat(message, handlers) (TDD, fetch mock)
├── hooks/
│   └── useChat.ts                # 상태 머신 (TDD, client mock)
└── components/
    ├── SourceCard.tsx            # 출처 카드 (TDD/RTL)
    ├── MessageBubble.tsx         # 메시지 버블 (TDD/RTL)
    ├── ChatInput.tsx             # 입력창 (TDD/RTL)
    └── Chat.tsx                  # 컨테이너: useChat + 리스트 + 입력 (TDD/RTL)
```


Task 1: 타입 + SSE 파서
**Files:**
- Create: `apps/web/lib/types.ts`
- Create: `apps/web/lib/sse.ts`
- Test: `apps/web/lib/sse.test.ts`
SSE 프레임(`data: <json>\n\n`)을 청크 경계와 무관하게 누적 파싱하는 순수 파서. 가장 중요한 테스트 가능 단위.

Task 2: 채팅 클라이언트 (fetch POST + 스트림)
**Files:**
- Create: `apps/web/lib/chatClient.ts`
- Test: `apps/web/lib/chatClient.test.ts`
`streamChat`은 `/chat`에 POST하고 응답 본문 스트림을 `SSEParser`로 파싱해 핸들러로 토큰/완료/에러를 전달한다.

Task 3: useChat 훅 (상태 머신)
**Files:**
- Create: `apps/web/hooks/useChat.ts`
- Test: `apps/web/hooks/useChat.test.tsx`
메시지 목록·전송·스트리밍 상태를 관리. `streamChat`을 주입 가능하게 해서 테스트한다.

Task 4: SourceCard 컴포넌트
**Files:**
- Create: `apps/web/components/SourceCard.tsx`
- Test: `apps/web/components/SourceCard.test.tsx`

Task 5: MessageBubble 컴포넌트
**Files:**
- Create: `apps/web/components/MessageBubble.tsx`
- Test: `apps/web/components/MessageBubble.test.tsx`
사용자/어시스턴트 메시지를 렌더. 어시스턴트 메시지는 답변 본문(줄바꿈 보존) + 스트리밍 표시 + 출처 카드 목록.

Task 6: ChatInput 컴포넌트
**Files:**
- Create: `apps/web/components/ChatInput.tsx`
- Test: `apps/web/components/ChatInput.test.tsx`

Task 7: Chat 컨테이너 + 페이지 연결
**Files:**
- Create: `apps/web/components/Chat.tsx`
- Test: `apps/web/components/Chat.test.tsx`
- Modify: `apps/web/app/page.tsx` (Chat 마운트)
- Modify: `apps/web/app/layout.tsx` (제목/언어)

Task 8: 라이브 엔드투엔드 스모크 (api + web)
**Files:** 없음(검증 전용). 코드 변경이 필요하면 별도 커밋.
실제 백엔드(MLX Qwen + Chroma)와 프론트엔드를 함께 띄워 브라우저에서 검증한다.