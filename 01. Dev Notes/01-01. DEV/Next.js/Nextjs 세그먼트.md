
세 가지 모두 **URL 경로의 일부를 변수로 받는 방법**입니다. 차이는 **몇 개의 세그먼트를 받느냐**에 있습니다. 세그먼트는 `/`로 나뉜 경로 조각 하나를 말합니다.

## 1. `[id]` 동적 세그먼트: 정확히 1개

```
app/blog/[slug]/page.tsx

보충)
요청 URL:   https://example.com/blog/1
렌더링 파일: app/blog/[slug]/page.tsx
params:      { slug: '1' }
```

|URL|매칭|params|
|---|---|---|
|`/blog/hello`|✅|`{ slug: 'hello' }`|
|`/blog`|❌|–|
|`/blog/a/b`|❌|–|

값은 **문자열 하나**입니다. 게시글 상세나 상품 상세처럼 "ID 하나로 페이지 하나"를 보여줄 때 씁니다.

```tsx
// Next.js 15+에서는 params가 Promise라서 await로 꺼내야 합니다
export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return <h1>{slug}</h1>;
}
```

## 2. `[...slug]` 캐치올 세그먼트: 1개 이상

```
app/docs/[...slug]/page.tsx
보충)
요청 URL:   https://example.com/blog/a/b/c
렌더링 파일: app/blog/[...slug]/page.tsx
params:      { slug: ['a', 'b', 'c'] }
```

| URL           | 매칭  | params                      |
| ------------- | --- | --------------------------- |
| `/docs/a`     | ✅   | `{ slug: ['a'] }`           |
| `/docs/a/b/c` | ✅   | `{ slug: ['a', 'b', 'c'] }` |
| `/docs`       | ❌   | –                           |

값은 **배열**이고, 그 뒤에 오는 경로를 깊이에 상관없이 전부 받습니다. 문서 사이트(`/docs/guide/routing/dynamic`)나 카테고리 계층처럼 깊이가 정해지지 않은 경로에 적합합니다. 다만 `/docs` 자체는 매칭되지 않으므로 `app/docs/page.tsx`를 따로 만들어야 합니다.

## 3. `[[...slug]]` 옵셔널 캐치올: 0개 이상

```
app/shop/[[...slug]]/page.tsx
```

|URL|매칭|params|
|---|---|---|
|`/shop`|✅|`{}` (slug는 `undefined`)|
|`/shop/clothes`|✅|`{ slug: ['clothes'] }`|
|`/shop/clothes/tops`|✅|`{ slug: ['clothes', 'tops'] }`|

캐치올과 같지만 **세그먼트가 없는 경우까지 포함**합니다. 파일 하나로 `/shop`과 그 하위 경로를 모두 처리할 수 있습니다.

```tsx
export default async function Page({ params }: { params: Promise<{ slug?: string[] }> }) {
  const { slug = [] } = await params;   // undefined 처리 필수
  return slug.length === 0 ? <ShopHome /> : <Category path={slug} />;
}
```

## 한눈에 비교

|문법|받는 세그먼트 수|타입|상위 경로(`/shop`) 매칭|
|---|---|---|---|
|`[id]`|정확히 1|`string`|❌|
|`[...slug]`|1 이상|`string[]`|❌|
|`[[...slug]]`|0 이상|`string[] \| undefined`|✅|

## 알아두면 좋은 규칙

- **정적 경로가 우선입니다.** `app/blog/new/page.tsx`와 `app/blog/[slug]/page.tsx`가 함께 있으면 `/blog/new`는 정적 경로로 갑니다.
- **충돌하는 조합이 있습니다.** `[[...slug]]`를 쓸 때 같은 위치에 `page.tsx`를 또 두면 `/shop` 경로가 겹쳐서 에러가 납니다. 같은 레벨에 `[...slug]`와 `[[...slug]]`를 함께 두는 것도 안 됩니다.
- **정적 생성(SSG)을 하려면** `generateStaticParams`에서 캐치올 값을 배열 형태로 반환합니다. 예: `[{ slug: ['a', 'b'] }]`
- **Pages Router**(`pages/` 디렉터리)도 문법은 같고, 값은 `useRouter().query`나 `getStaticProps`의 `params`로 받습니다.

정리하면, 상세 페이지 하나에는 `[id]`, 깊이가 자유로운 하위 경로에는 `[...slug]`, 상위 경로까지 한 파일로 처리하려면 `[[...slug]]`를 쓰면 됩니다.

---
`page.tsx`는 **화면(UI)**을 렌더링하는 파일이고, `route.ts`는 **API 응답(HTTP Response)**을 반환하는 파일입니다. 동적 세그먼트 문법(`[...nextauth]`)은 두 파일에서 똑같이 동작하고, 매칭된 요청을 어떻게 처리하느냐만 다릅니다.

## page.tsx와 route.ts 비교

|        | `page.tsx`<br>                | `route.ts` (Route Handler)                |
| ------ | ----------------------------- | ----------------------------------------- |
| 역할     | React 컴포넌트를 렌더링해 HTML 페이지를 반환 | HTTP 메서드별 함수를 실행해 JSON 등 응답을 반환           |
| export | `export default` 컴포넌트 하나      | `GET`, `POST`, `PUT`, `DELETE` 등 이름 있는 함수 |
| 호출 주체  | 주로 브라우저에서 사용자가 접속             | 주로 `fetch`, 폼 제출, OAuth 리다이렉트 같은 프로그램 요청  |
| 확장자    | `.tsx` (JSX 사용)               | `.ts` (JSX 불필요)                           |
| 레이아웃   | `layout.tsx` 적용됨              | 적용 안 됨                                    |

**같은 폴더에 둘을 함께 둘 수 없습니다.** 하나의 경로가 페이지이면서 동시에 API일 수는 없기 때문입니다.

## `src/app/api/auth/[...nextauth]/route.ts`가 하는 일

- `src/app`은 `app`과 똑같이 동작합니다. `src`도 URL에 포함되지 않습니다.
- 이 파일은 `/api/auth/` 아래의 **모든 경로**를 받습니다. 세그먼트가 1개 이상이어야 하므로 `/api/auth` 자체는 매칭되지 않습니다.

|요청 URL|params|
|---|---|
|`/api/auth/signin`|`{ nextauth: ['signin'] }`|
|`/api/auth/signout`|`{ nextauth: ['signout'] }`|
|`/api/auth/session`|`{ nextauth: ['session'] }`|
|`/api/auth/callback/google`|`{ nextauth: ['callback', 'google'] }`|

NextAuth는 로그인, 로그아웃, 세션 조회, OAuth 콜백 등 여러 엔드포인트가 필요합니다. 캐치올 하나로 이 요청을 전부 받은 다음, NextAuth가 `nextauth` 배열을 보고 내부에서 알맞은 처리로 분기합니다. 그래서 파일마다 라우트를 따로 만들 필요가 없습니다.

## 코드 형태

버전에 따라 보통 이렇게 생겼습니다.

```ts
// NextAuth v4
import NextAuth from "next-auth";
import { authOptions } from "@/lib/auth";

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

```ts
// Auth.js (NextAuth v5)
import { handlers } from "@/auth";
export const { GET, POST } = handlers;
```

`default` export가 없고 `GET`과 `POST`를 export한다는 점이 `page.tsx`와 가장 크게 다릅니다. 예를 들어 브라우저가 `/api/auth/callback/google`로 GET 요청을 보내면 export된 `GET` 함수가 실행됩니다. 로그인 폼처럼 POST로 보내는 요청은 `POST` 함수가 처리합니다.
