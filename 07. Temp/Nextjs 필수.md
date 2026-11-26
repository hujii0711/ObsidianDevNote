Next.js 필수 개념 목록입니다. (App Router 기준)

**라우팅**
1. **App Router**: `app/` 폴더 기반 파일 시스템 라우팅
2. **특수 파일**: `page.tsx`, `layout.tsx`, `loading.tsx`, `error.tsx`, `not-found.tsx`, `template.tsx`
3. **동적 라우트**: `[id]`, `[...slug]`, `[[...slug]]`
4. **Route Group / Parallel / Intercepting Routes**: `(group)`, `@slot`, `(.)`
5. **네비게이션**: `<Link>`, `useRouter`, `redirect`, `usePathname`, `useSearchParams`

**렌더링**
6. **Server Components**: 기본값. 서버에서 실행되어 번들 크기 감소, DB/API 직접 접근
7. **Client Components**: `"use client"` 선언, 상태·이벤트·브라우저 API 사용
8. **SSR / SSG / ISR / CSR**: 렌더링 시점별 전략
9. **Streaming & Suspense**: 페이지를 점진적으로 전송
10. **Partial Prerendering (PPR)**: 정적 셸 + 동적 부분 혼합

**데이터 처리**
11. **데이터 페칭**: Server Component에서 `async/await`로 `fetch` 사용
12. **캐싱**: 요청 메모이제이션, Data Cache, Full Route Cache, Router Cache
13. **재검증**: 시간 기반(`revalidate`), 온디맨드(`revalidatePath`, `revalidateTag`)
14. **Server Actions**: `"use server"` 함수로 폼 제출·데이터 변경 처리

**서버 기능**
15. **Route Handlers**: `route.ts`로 API 엔드포인트 구현
16. **Middleware** (`middleware.ts`): 요청 전 처리 (인증, 리다이렉트, 리라이트)
17. **쿠키/헤더**: `cookies()`, `headers()`

**최적화**
18. **`next/image`**: 이미지 자동 최적화
19. **`next/font`**: 폰트 최적화
20. **`next/script`**: 외부 스크립트 로딩 제어
21. **Metadata API**: SEO용 `metadata`, `generateMetadata`
22. **동적 import / lazy loading**

**설정·배포**
23. **`next.config.js`**: 빌드·런타임 설정
24. **환경 변수**: `.env`, `NEXT_PUBLIC_` 접두사 규칙
25. **스타일링**: CSS Modules, Tailwind, CSS-in-JS
26. **배포 방식**: Node 서버, 정적 export, Docker, Vercel, Amplify 등

**학습 순서 추천**
App Router/파일 구조 → Server vs Client Components → 데이터 페칭·캐싱 → Server Actions → Route Handlers → Middleware → 최적화 → 배포

Next.js 버전(특히 15 이후)에서 캐싱 기본값과 일부 API(`cookies()`, `params`가 비동기로 변경 등)가 바뀌었으니, 사용 중인 버전의 공식 문서를 함께 확인하세요.