
# **Next.js에서 SSE(Server-Sent Events) 구현하기**

Next.js에서는 **Route Handler + Web Streams API + EventSource** 조합으로 SSE를 구현할 수 있습니다.
Next.js App Router 기준으로 설명합니다.
SSE는 서버에서 클라이언트로 데이터를 실시간 전송하는 단방향 통신 방식입니다. HTTP 연결을 유지하면서 서버가 이벤트를 지속적으로 전달합니다.

## **1. SSE 동작 구조**

```
Browser (Client)
    |
    | new EventSource('/api/events')
    |
    | HTTP GET /api/events
    v
Next.js Route Handler
    |
    | app/api/events/route.ts
    |
    | text/event-stream
    v
Browser EventSource
    |
    | onmessage 이벤트로 데이터 수신
    v
React UI 업데이트
```

주요 특징은 다음과 같습니다.

- 클라이언트 → 서버: 일반 HTTP 요청
- 서버 → 클라이언트: 지속적인 이벤트 스트리밍
- 연결이 끊어지면 EventSource가 자동 재연결
- HTTP 기반이므로 WebSocket보다 구현이 간단
- 기본적으로 서버에서 클라이언트 방향으로만 데이터 전달

## **2. 서버 구현 (Route Handler)**

파일 위치:
`app/api/events/route.ts`
```ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      let count = 0;

      const timer = setInterval(() => {
        count++;

        const data = {
          count,
          message: `SSE message ${count}`,
          timestamp: new Date().toISOString(),
        };

        // SSE 프로토콜 형식
        const message =
          `id: ${count}\n` +
          `event: message\n` +
          `data: ${JSON.stringify(data)}\n\n`;

        controller.enqueue(encoder.encode(message));

        if (count >= 10) {
          clearInterval(timer);
          controller.close();
        }
      }, 1000);
    },

    cancel() {
      // 연결 종료 시 리소스 정리
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
```

### **주요 API 및 설정**

| 항목                     | 설명                  |
| ---------------------- | ------------------- |
| `ReadableStream`       | 스트리밍 응답 생성          |
| `TextEncoder`          | 문자열을 Uint8Array로 변환 |
| `controller.enqueue()` | 클라이언트로 데이터 전송       |
| `controller.close()`   | 스트림 종료              |
| `text/event-stream`    | SSE 응답 MIME 타입      |
| `X-Accel-Buffering`    | Nginx 응답 버퍼링 방지     |
| `force-dynamic`        | 정적 렌더링 대신 동적 요청 처리  |

## **3. 클라이언트 구현 (EventSource)**

SSE 클라이언트는 반드시 Client Component에서 구현해야 합니다.

파일 위치:
`app/components/SSEClient.tsx`
```ts
"use client";

import { useEffect, useState } from "react";

interface SSEMessage {
  count: number;
  message: string;
  timestamp: string;
}

export default function SSEClient() {
  const [messages, setMessages] = useState<SSEMessage[]>([]);
  const [status, setStatus] = useState("연결 중");

  useEffect(() => {
    const eventSource = new EventSource("/api/events");

    eventSource.onopen = () => {
      setStatus("연결됨");
    };

    eventSource.addEventListener("message", (event) => {
      const data: SSEMessage = JSON.parse(event.data);

      setMessages((prev) => [...prev, data]);
    });

    eventSource.onerror = () => {
      setStatus("연결 오류 / 재연결 중");
    };

    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <div>
      <h2>SSE Status: {status}</h2>

      <ul>
        {messages.map((item) => (
          <li key={item.count}>
            {item.message} ({item.timestamp})
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### **페이지에서 사용하기**

```ts
import SSEClient from "./components/SSEClient";

export default function Page() {
  return (
    <main>
      <h1>SSE Example</h1>
      <SSEClient />
    </main>
  );
}
```

### **EventSource 주요 API**

|API|설명|
|---|---|
|`new EventSource(url)`|SSE 연결 생성|
|`onopen`|연결 성공|
|`onmessage`|기본 메시지 수신|
|`addEventListener()`|이벤트 이름별 수신|
|`onerror`|연결 오류|
|`close()`|연결 종료|
|`readyState`|연결 상태 확인|

연결 상태는 다음 세 가지입니다.

|상태|값|설명|
|---|---|---|
|`CONNECTING`|0|연결 중 또는 재연결 중|
|`OPEN`|1|연결됨|
|`CLOSED`|2|연결 종료|

## **4. SSE 메시지 프로토콜**
SSE는 단순히 JSON을 전송하는 것이 아니라 정해진 텍스트 프로토콜을 사용합니다.

### **서버 전송 데이터 예시**

```
id: 1001
event: message
data: {"message":"Hello"}

id: 1002
event: notification
data: {"message":"New notification"}

: heartbeat
```

### **각 필드의 의미**

|필드|설명|
|---|---|
|`id`|이벤트 식별자, 재연결 시 활용|
|`event`|이벤트 이름|
|`data`|실제 전달 데이터|
|`retry`|재연결 대기 시간(ms)|
|`:`|주석, heartbeat 용도로 활용|

**중요한 규칙**
하나의 이벤트는 반드시 빈 줄(`\n\n`)로 끝나야 합니다.

### **이벤트 이름별 수신 방법**

이벤트 이름이 `message`이면 `onmessage`로 받을 수 있고, `notification`이면 다음과 같이 받아야 합니다.

```ts
eventSource.addEventListener("notification", (event) => {
  console.log(JSON.parse(event.data));
});
```

## **5. 실무용 서버 코드 개선**

앞서 작성한 서버 예제에서 클라이언트 연결 종료 시 타이머를 정리하도록 보완해야 합니다.

다음은 연결 종료와 AbortSignal까지 처리하는 예제입니다.

`app/api/events/route.ts`

```ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const encoder = new TextEncoder();

export async function GET(request: Request) {
  let timer: ReturnType<typeof setInterval> | undefined;
  let count = 0;
  let closed = false;

  const stream = new ReadableStream({
    start(controller) {
      const cleanup = () => {
        if (closed) return;

        closed = true;

        if (timer) {
          clearInterval(timer);
        }

        request.signal.removeEventListener("abort", cleanup);
      };

      const send = (data: object) => {
        if (closed) return;

        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify(data)}\n\n`
          )
        );
      };

      request.signal.addEventListener("abort", cleanup);

      timer = setInterval(() => {
        count++;

        send({
          count,
          message: `Message ${count}`,
        });

        if (count >= 10) {
          cleanup();
          controller.close();
        }
      }, 1000);
    },

    cancel() {
      if (!closed) {
        closed = true;
        if (timer) clearInterval(timer);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
```

### **실제 운영 환경에서 고려할 사항**

1. **Heartbeat**: 일정 시간마다 `: heartbeat\n\n` 전송
2. **재연결**: `retry` 및 `Last-Event-ID` 기반 이벤트 복구
3. **인증**: 연결 요청에 대한 사용자 인증 및 권한 검증
4. **리소스 관리**: 클라이언트 연결 종료 시 구독 및 타이머 해제
5. **프록시**: Nginx 등의 응답 버퍼링 비활성화
6. **서버리스**: 플랫폼의 최대 실행 시간과 연결 유지 제한 확인

## **6. SSE와 WebSocket 비교**

|구분|SSE|WebSocket|
|---|---|---|
|통신 방향|단방향|양방향|
|프로토콜|HTTP|WebSocket|
|연결 유지|O|O|
|자동 재연결|기본 지원|직접 구현|
|데이터 형식|텍스트 기반|텍스트 및 바이너리|
|브라우저 API|EventSource|WebSocket|
|서버 구현|상대적으로 간단|상대적으로 복잡|

### **SSE가 적합한 경우**
- AI LLM 응답 실시간 스트리밍
- 실시간 알림
- 작업 진행률 표시
- 배치 작업 상태 모니터링
- 서버 로그 실시간 출력
- 주식 시세 및 데이터 업데이트

### **WebSocket이 적합한 경우**
- 실시간 채팅
- 온라인 게임
- 실시간 공동 편집
- 양방향 데이터 동기화
- 클라이언트와 서버 간 지속적인 상호작용

## **7. Next.js에서 SSE와 AI 스트리밍**

AI 애플리케이션에서는 SSE가 상당히 유용합니다.

예를 들어 Next.js에서 LangGraph 또는 Python 기반 AI 서버와 연동한다면 다음 구조로 구현할 수 있습니다.

### **전체 아키텍처**

```
Next.js Client
     |
     | EventSource 또는 fetch 기반 스트리밍
     |
     v
Next.js Route Handler
     |
     | /api/ai/stream
     | (BFF)
     |
     v
Python AI Backend
     |
     | LangGraph / FastAPI
     |
     v
LLM
```

각 계층의 역할은 다음과 같습니다.

|계층|역할|
|---|---|
|Next.js Client|SSE 연결 및 실시간 UI 업데이트|
|Route Handler|BFF 역할, 인증 및 요청 중계|
|FastAPI|Python AI 서버의 SSE 응답 생성|
|LangGraph|Agent 실행 및 이벤트 생성|
|LLM|토큰 생성 및 응답 생성|

### **LangGraph 이벤트 예시**

LangGraph에서 생성되는 다음과 같은 이벤트를 전달할 수 있습니다.

```
{"type":"token","content":"안녕하세요"}
{"type":"token","content":"반갑습니다"}
{"type":"tool_start","name":"search"}
{"type":"tool_end","name":"search"}
{"type":"complete"}
```

Next.js는 이 이벤트를 받아 React UI에 실시간으로 반영하면 됩니다.

### **EventSource와 Fetch 스트리밍의 차이**

기본 `EventSource`는 GET 요청만 지원하고 임의의 Authorization 헤더를 설정할 수 없습니다.

|구분|EventSource|Fetch + ReadableStream|
|---|---|---|
|HTTP 메서드|GET|GET, POST 등|
|커스텀 헤더|불가능|가능|
|자동 재연결|기본 지원|직접 구현|
|POST 요청|불가능|가능|
|JSON 요청 본문|불가능|가능|
|SSE 프로토콜|지원|직접 파싱 필요|

따라서 다음과 같이 선택할 수 있습니다.

- 일반적인 서버 알림: `EventSource`
- AI 질문을 POST로 전달하고 스트리밍 응답 받기: `fetch() + ReadableStream`
- 인증 토큰을 Authorization 헤더로 전달: `fetch() + ReadableStream`

## **8. 공식 문서**

- [Next.js Route Handlers](https://nextjs.org/docs/app/api-reference/file-conventions/route?utm_source=chatgpt.com)
- [Next.js Route Handler API Reference](https://nextjs.org/docs/app/api-reference/functions/next-response?utm_source=chatgpt.com)
- [MDN - Using Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events?utm_source=chatgpt.com)

## **9. 핵심 정리**

|항목|핵심 기술|
|---|---|
|서버 구현|Next.js Route Handler|
|스트림 생성|`ReadableStream`|
|데이터 인코딩|`TextEncoder`|
|데이터 전송|`controller.enqueue()`|
|클라이언트 연결|`EventSource`|
|React 연동|`useEffect`, `useState`|
|응답 타입|`text/event-stream`|
|연결 관리|`AbortSignal`, `close()`|

**추천 아키텍처**

Next.js + Python LangGraph 조합이라면 다음 구조를 추천합니다.

`LangGraph → FastAPI SSE → Next.js Route Handler → React UI`

이 구조를 사용하면 AI Agent의 토큰 스트리밍뿐 아니라 Tool 실행 시작, 종료, 진행 상황까지 실시간으로 표시할 수 있습니다.