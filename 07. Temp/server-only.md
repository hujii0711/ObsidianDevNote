
### **Next.js의** `import 'server-only'`**란?**
Next.js에서 `import 'server-only'`는 **특정 모듈이 서버 환경에서만 사용되도록 강제하는 보호 장치**입니다.
즉, 서버 전용 코드가 실수로 Client Component에서 import되는 것을 방지합니다.
Next.js App Router에서는 Server Component와 Client Component가 공존하기 때문에 서버 전용 모듈을 명확하게 구분하는 것이 중요합니다.

## **1. 기본 사용법**
예를 들어 데이터베이스 접속이나 API Secret을 사용하는 모듈이 있다고 가정하겠습니다.

`lib/db.ts`
```ts
import 'server-only';

import { PrismaClient } from '@prisma/client';

export const db = new PrismaClient();
```

첫 번째 줄의 `import 'server-only'`가 핵심입니다.

이 모듈을 서버에서 import하는 것은 허용하지만, Client Component에서 직접 또는 간접적으로 import하면 빌드 단계에서 오류가 발생합니다.

## **2. 사용하지 않았을 때의 문제**

다음과 같은 구조를 가정해 보겠습니다.

```
app/
 ├── page.tsx
 ├── components/
 │    └── UserList.tsx
 └── lib/
      └── db.ts
```

`db.ts`

```ts
import { PrismaClient } from '@prisma/client';

export const db = new PrismaClient();
```

`UserList.tsx`

```ts
"use client";

import { db } from "../lib/db";

export default function UserList() {
  // ...
}
```

이 경우 서버 전용 데이터베이스 모듈이 클라이언트 번들에 포함될 수 있는 잘못된 의존성 구조가 됩니다.

실제로 Prisma와 같은 서버 전용 라이브러리는 클라이언트에서 사용할 수 없으며, 빌드 오류가 발생하거나 잘못된 코드 구조가 만들어질 수 있습니다.

## **3.** `server-only`**를 적용하면?**

`db.ts`에 다음 코드를 추가합니다.

```ts
import 'server-only';

import { PrismaClient } from '@prisma/client';

export const db = new PrismaClient();
```

이후 Client Component에서 해당 모듈을 import하면 Next.js가 오류를 발생시킵니다.

```ts
"use client";

import { db } from "../lib/db"; // 오류 발생
```

즉, 서버 전용 모듈의 클라이언트 영역 유입을 차단합니다.

## **4. Server Component에서는 사용 가능**

다음과 같이 Server Component에서 사용하는 것은 정상입니다.

`app/users/page.tsx`

```ts
import { db } from "@/lib/db";

export default async function UsersPage() {
  const users = await db.user.findMany();

  return (
    <div>
      {users.map((user) => (
        <div key={user.id}>
          {user.name}
        </div>
      ))}
    </div>
  );
}
```

Server Component는 기본적으로 서버에서 실행되므로 `server-only` 모듈을 사용할 수 있습니다.

## **5.** `server-only`**와** `client-only` **비교**

Next.js에서는 반대 역할을 하는 `client-only`도 제공합니다.

|구분|server-only|client-only|
|---|---|---|
|목적|서버 전용 모듈 보호|클라이언트 전용 모듈 보호|
|사용 위치|서버 전용 파일|클라이언트 전용 파일|
|반대 환경에서 import|오류|오류|
|주요 용도|DB, Secret, 서버 API|브라우저 API, DOM|

### **server-only 예제**

```ts
// lib/server-api.ts

import 'server-only';

export async function getSecretData() {
  const apiKey = process.env.API_SECRET;

  // 서버 API 호출
}
```

### **client-only 예제**

```ts
// lib/browser-storage.ts

import 'client-only';

export function getLocalData() {
  return localStorage.getItem('token');
}
```

## **6. 실제 프로젝트에서 사용하는 구조**

실무에서는 서버와 클라이언트 모듈을 명확하게 분리하는 것이 좋습니다.

```
src/
 ├── app/
 │    ├── users/
 │    │    └── page.tsx
 │    └── components/
 │         └── UserClient.tsx
 │
 ├── lib/
 │    ├── server/
 │    │    ├── db.ts
 │    │    ├── auth.ts
 │    │    └── api.ts
 │    │
 │    └── client/
 │         ├── storage.ts
 │         └── browser.ts
 │
 └── services/
      ├── user.server.ts
      └── user.client.ts
```

예를 들어 서버 전용 서비스는 다음과 같이 작성합니다.

`services/user.server.ts`

```ts
import 'server-only';

import { db } from '@/lib/server/db';

export async function getUsers() {
  return await db.user.findMany();
}
```

클라이언트에서 호출할 필요가 있다면 API Route Handler를 통해 접근하도록 구성합니다.

```
Client Component
      |
      | fetch('/api/users')
      v
Route Handler
      |
      v
Server Service
      |
      v
Database
```

## **7. 알아야 할 핵심 사항**

`import 'server-only'`는 다음과 같은 기능을 제공하지 않습니다.

- 서버에서만 실행하도록 코드를 자동 변환하는 기능이 아닙니다.
    
- 서버 API를 생성하는 기능이 아닙니다.
    
- 환경변수를 암호화하는 기능이 아닙니다.
    
- 인증이나 권한을 자동으로 검사하는 기능이 아닙니다.
    
- 일반적인 Node.js 모듈에서 동작을 강제하는 범용 보안 기능이 아닙니다.
    

Next.js의 모듈 경계 검사 기능이라고 이해하면 됩니다.

특히 `server-only`를 사용한다고 해서 `NEXT_PUBLIC_` 환경변수나 서버 데이터가 자동으로 보호되는 것은 아닙니다.

## **8. 핵심 정리**

|구분|설명|
|---|---|
|`import 'server-only'`|서버 전용 모듈임을 선언|
|목적|Client Component에서 서버 모듈 import 방지|
|대표 사용처|DB, API Secret, 인증, 서버 서비스|
|`import 'client-only'`|클라이언트 전용 모듈임을 선언|
|도입 목적|Server/Client 모듈 경계 명확화|

**결론:** Next.js App Router에서 `server-only`는 서버 전용 코드가 클라이언트 번들로 유입되는 것을 방지하기 위한 모듈 경계 보호 장치입니다.


---

## `import 'server-only'`란

**이 파일은 서버에서만 실행되어야 한다**는 표시입니다. Client Component(`'use client'`) 쪽에서 이 파일을 import하면 **빌드할 때 오류**가 나서, 서버 코드가 브라우저 번들에 섞여 들어가는 실수를 미리 막아 줍니다.

[memo-db.ts](demo_next/src/lib/server/memo-db.ts)에 이 표시가 필요한 이유는 이렇습니다.
- `node:sqlite`는 Node에만 있는 모듈이라 브라우저에서는 실행되지 않습니다.
- DB 접근 코드나 SQL 쿼리가 클라이언트 번들에 들어가면 안 됩니다. API 키를 쓰는 [backend-config.ts](demo_next/src/lib/server/backend-config.ts) 같은 파일은 비밀값이 새어 나갈 위험까지 막아 줍니다.

동작에 대해 알아 둘 점은 두 가지입니다.
- `package.json`에 `server-only`가 없는데도 동작합니다. Next.js가 이 import를 직접 처리하기 때문에 설치는 선택 사항입니다(`node_modules/next/dist/docs/01-app/01-getting-started/05-server-and-client-components.md` 584~602행). 반대로 클라이언트 전용 코드를 표시할 때는 `'client-only'`를 씁니다.
- 검사는 **import할 때 한 번만** 합니다. 실행 중에 접근을 막는 보안 장치는 아닙니다.

## 함께 살펴볼 것

**1. 타입만 가져오는 import는 괜찮습니다**
[todo-list.tsx:5](demo_next/src/app/(nextjs)/mutation/server-actions/todo-list.tsx#L5)는 Client Component인데 `import type { Memo } from '@/lib/server/memo-db'`를 씁니다. `import type`은 컴파일할 때 지워지므로 오류가 나지 않습니다. `type`을 빼고 `import { Memo }`로 쓰면 빌드 오류가 납니다.

**2. `server-only`와 `'use server'`는 다릅니다** (가장 헷갈리기 쉬운 부분)
- `server-only`: 이 모듈을 클라이언트가 **import하지 못하게** 막습니다.
- `'use server'`: 함수를 **외부에서 POST로 부를 수 있는 엔드포인트**로 공개합니다.

그래서 `actions.ts`의 서버 액션은 누구나 부를 수 있습니다. 인증·인가 검사는 액션 안에서 직접 해야 합니다. [auth/authorization/actions.ts](demo_next/src/app/(nextjs)/auth/authorization/actions.ts)가 `getCurrentUser()`와 `can()`으로 검사하는 것이 그 예입니다.

**3. 이 DAL에는 권한 검사가 없습니다**
`getMemo`, `updateMemo`, `deleteMemo`는 id만 받고 owner를 확인하지 않습니다. 누가 쓸 수 있는지는 호출하는 쪽이 매번 확인해야 합니다. 지금은 [rest/memos/[id]/route.ts:11-18](demo_next/src/app/api/rest/memos/[id]/route.ts#L11-L18)의 `findOwn`처럼 잘 지키고 있습니다. 다만 새 호출부를 만들면서 이 검사를 빠뜨리면, 남의 id만 넣어서 다른 사람의 메모를 볼 수 있게 됩니다(IDOR). Next 공식 문서(`data-security.md`)는 권한 검사를 DAL 안에 두라고 권합니다.

**4. `export const db`도 바깥으로 공개되어 있습니다** ([37행](demo_next/src/lib/server/memo-db.ts#L37))
이 함수들을 거치지 않고 raw SQL을 바로 실행할 수 있는 통로입니다. 밖에서 쓰는 곳이 없다면 `export`를 빼는 편이 계층을 나눈 의도에 맞습니다.

**5. 그 밖의 작은 점**
- **`globalThis` 캐시** ([10행](demo_next/src/lib/server/memo-db.ts#L10), [37행](demo_next/src/lib/server/memo-db.ts#L37)): dev 서버에서 코드를 고쳐 모듈이 다시 로드돼도 DB가 초기화되지 않게 하는 흔한 방법입니다.
- **LIKE 검색** ([42행](demo_next/src/lib/server/memo-db.ts#L42)): 자리표시자를 쓰고 있어 SQL Injection은 막혀 있습니다. 다만 사용자가 `%`나 `_`를 입력하면 와일드카드로 해석됩니다. 정확히 일치하는 검색이 필요하다면 이스케이프해야 합니다.
- **`done: number`**: SQLite에는 boolean 타입이 없어서 0과 1이 그대로 클라이언트까지 전달됩니다.
- **시간 기준이 섞여 있음**: 시드 데이터(`'2026-11-01 09:00:00'`)는 고정 문자열이고, `now()`는 UTC 기준이라 화면에 표시되는 시각의 기준이 다릅니다.

정리하면 `server-only`는 **번들 경계**를 지키는 장치이고, **접근 권한**은 서버 액션·Route Handler·DAL 안에서 따로 검사해야 합니다.