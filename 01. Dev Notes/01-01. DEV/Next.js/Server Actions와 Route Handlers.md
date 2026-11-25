Next.js에서 **Server Actions와 Route Handlers를 분리해서 사용하는 이유는 서버 기능의 목적과 통신 방식이 다르기 때문**입니다.

간단히 정리하면 다음과 같습니다.
- **Server Actions:** 프론트엔드에서 서버의 함수를 직접 호출하는 방식으로 데이터 변경 작업에 적합합니다.
- **Route Handlers:** HTTP API를 제공하는 방식으로 외부 시스템과의 통신 및 REST API 구현에 적합합니다.

## **1. 구조적 차이**

### **1-1. Server Actions**
서버에서 실행할 함수를 정의하고 클라이언트에서 함수 호출 형태로 사용합니다.

## **Server Actions 구조**

**Client Component**
`createUser() 호출`

내부적으로 HTTP POST 요청
**Server Action**
`createUser(formData)`

**Database**
INSERT / UPDATE / DELETE

코드 예제:
```js
// app/actions/user.ts
"use server";

export async function createUser(formData: FormData) {
  const name = formData.get("name");

  await db.user.create({
    data: { name: String(name) }
  });

  return { success: true };
}
```

클라이언트에서 호출:
```js
"use client";

import { createUser } from "@/app/actions/user";

export default function UserForm() {
  return (
    <form action={createUser}>
      <input name="name" />
      <button type="submit">등록</button>
    </form>
  );
}
```

특징:
- HTTP API를 직접 작성하지 않아도 됩니다.
- 함수 호출 형태로 서버 기능을 사용할 수 있습니다.
- Form 처리에 편리합니다.
- 데이터 변경 후 캐시 재검증을 연결하기 쉽습니다.
- Next.js 애플리케이션 내부의 사용자 액션에 적합합니다.

### **1-2. Route Handlers**
HTTP 요청을 받아 응답하는 API 엔드포인트입니다.

## **Route Handlers 구조**

**Client / 외부 시스템**
`POST /api/users`

HTTP Request

**Route Handler**
`POST(request: Request)`

**Database**
INSERT / UPDATE / DELETE

HTTP Response (JSON)
**Client / 외부 시스템**
`{ success: true }`

코드 예제:
```js
// app/api/users/route.ts

import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const body = await request.json();

  const user = await db.user.create({
    data: {
      name: body.name
    }
  });

  return NextResponse.json({
    success: true,
    data: user
  });
}
```

호출하는 쪽:
```js
const response = await fetch("/api/users", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    name: "홍길동"
  })
});

const result = await response.json();
```

특징:
- 명확한 URL과 HTTP Method를 제공합니다.
- REST API 구현에 적합합니다.
- 다른 프론트엔드나 외부 시스템에서도 호출할 수 있습니다.
- HTTP Header, Status Code, Request/Response 등을 직접 제어할 수 있습니다.

## **2. Server Actions와 Route Handlers 비교**

|비교 항목|Server Actions|Route Handlers|
|---|---|---|
|기본 목적|서버 함수 실행|HTTP API 제공|
|호출 방식|함수 호출 형태|HTTP 요청|
|내부 통신|HTTP POST 기반|HTTP|
|URL 관리|프레임워크가 관리|직접 정의|
|HTTP Method|내부적으로 POST|GET, POST 등|
|데이터 변경|매우 적합|적합|
|조회 API|일반적인 조회에는 덜 적합|매우 적합|
|외부 시스템 연동|제한적|매우 적합|
|Form 처리|매우 편리|직접 구현|
|REST API|부적합|적합|
|응답 형식|직렬화 가능한 반환값|자유롭게 정의|

## **3. 왜 두 가지를 모두 제공할까?**

핵심은 **내부 애플리케이션 로직과 외부 시스템에 공개하는 API를 구분하기 위해서**입니다.

예를 들어 사용자 관리 시스템을 개발한다고 가정하겠습니다.

|기능|권장 방식|이유|
|---|---|---|
|사용자 등록 버튼|Server Actions|내부 UI의 데이터 변경|
|사용자 정보 수정|Server Actions|내부 UI의 데이터 변경|
|사용자 삭제|Server Actions|내부 UI의 데이터 변경|
|사용자 목록 API|Route Handlers|HTTP 기반 데이터 조회|
|외부 시스템 사용자 조회|Route Handlers|외부 접근 필요|
|모바일 앱 사용자 등록|Route Handlers|모바일 앱에서 HTTP 호출|
|외부 Webhook 수신|Route Handlers|HTTP 요청 수신|

## **4. 실무에서는 어떻게 설계하는 것이 좋은가?**

제가 추천하는 방식은 **비즈니스 로직을 별도로 분리하고 Server Actions와 Route Handlers가 공통 로직을 호출하도록 설계하는 것**입니다.

예를 들어 다음과 같이 구성할 수 있습니다.

```
src/
 ├── app/
 │    ├── users/
 │    │    └── page.tsx
 │    │
 │    ├── actions/
 │    │    └── user.ts
 │    │
 │    └── api/
 │         └── users/
 │              └── route.ts
 │
 ├── services/
 │    └── user.service.ts
 │
 ├── repositories/
 │    └── user.repository.ts
 │
 └── lib/
      └── db.ts
```

각 계층의 역할은 다음과 같습니다.

|계층|역할|
|---|---|
|Page / Component|사용자 화면|
|Server Actions|내부 UI의 서버 기능 호출|
|Route Handlers|외부 HTTP API 제공|
|Service|비즈니스 로직|
|Repository|데이터베이스 CRUD|
|DB|데이터 저장|

예를 들어 공통 Service:
```js
// services/user.service.ts

export async function createUser(name: string) {
  if (!name.trim()) {
    throw new Error("이름은 필수입니다.");
  }

  return db.user.create({
    data: { name }
  });
}
```

Server Actions:
```js
// app/actions/user.ts
"use server";

import { createUser } from "@/services/user.service";

export async function createUserAction(formData: FormData) {
  const name = String(formData.get("name") ?? "");

  return createUser(name);
}
```

Route Handlers:
```js
// app/api/users/route.ts

import { NextResponse } from "next/server";
import { createUser } from "@/services/user.service";

export async function POST(request: Request) {
  const body = await request.json();

  const user = await createUser(body.name);

  return NextResponse.json(user, { status: 201 });
}
```

이렇게 하면 동일한 비즈니스 로직을 내부 UI와 외부 API에서 재사용할 수 있습니다.

단, 실제 서비스에서는 두 진입점 모두에서 인증·권한·입력값 검증을 수행해야 합니다.

## **5. Node.js Express와 비교하면?**

기존 Express 개발 경험을 기준으로 이해하면 다음과 같습니다.

| Next.js          | Express 대응 개념               |
| ---------------- | --------------------------- |
| Server Actions   | 서버 기능을 호출하는 전용 액션 인터페이스     |
| Route Handlers   | Express Router / Controller |
| Service          | Service Layer               |
| Repository       | Repository / DAO            |
| Server Component | 서버에서 데이터를 조회하는 페이지 렌더링 로직   |

여기서 중요한 차이가 있습니다.

Express에서는 대부분의 서버 기능을 API 엔드포인트로 구현하지만, Next.js는 내부 UI에서 사용하는 서버 기능을 Server Actions로 직접 노출할 수 있다는 점입니다.

## **결론**

**두 가지를 무조건 분리해야 하는 것은 아닙니다.** 프로젝트의 요구사항에 따라 선택하면 됩니다.
- Next.js 내부 화면에서 발생하는 등록·수정·삭제 → Server Actions
- 외부 시스템과 통신하거나 REST API를 제공해야 하는 경우 → Route Handlers
- 양쪽에서 동일한 비즈니스 로직을 사용한다면 → Service 계층으로 공통화

특히 기존 Node.js 백엔드 개발 경험이 있다면 Route Handlers는 익숙한 REST API 방식으로 이해하고, Server Actions는 Next.js가 제공하는 서버 함수 호출 추상화로 이해하면 가장 정확합니다.