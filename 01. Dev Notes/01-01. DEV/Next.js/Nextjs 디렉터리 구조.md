**핵심 원칙**: `app/`은 라우팅만 담당하고, 비즈니스 로직과 UI는 `features/` 같은 도메인 단위 폴더로 분리합니다.

```
my-app/
├─ src/
│  ├─ app/                  # 라우팅 전용 (얇게 유지)
│  │  ├─ (marketing)/       # Route Group
│  │  ├─ (dashboard)/
│  │  │  ├─ layout.tsx
│  │  │  └─ orders/page.tsx
│  │  ├─ api/               # Route Handlers
│  │  ├─ layout.tsx
│  │  ├─ error.tsx
│  │  └─ not-found.tsx
│  ├─ features/             # 도메인별 묶음
│  │  └─ orders/
│  │     ├─ components/
│  │     ├─ actions.ts      # Server Actions
│  │     ├─ queries.ts      # 데이터 조회
│  │     ├─ schemas.ts      # zod 검증
│  │     └─ types.ts
│  ├─ components/
│  │  ├─ ui/                # 버튼, 모달 등 공통 UI
│  │  └─ layout/            # 헤더, 푸터
│  ├─ lib/                  # 외부 연동 (db, auth, api client)
│  ├─ hooks/                # 공통 클라이언트 훅
│  ├─ config/               # 상수, 사이트 설정
│  ├─ styles/
│  └─ middleware.ts
├─ public/
├─ tests/                   # e2e (Playwright)
├─ .env.example
├─ next.config.js
└─ Dockerfile
```

**운영 시 지키면 좋은 규칙**
- **`src/` 사용**: 설정 파일과 소스를 분리
- **`app/`은 얇게**: `page.tsx`는 데이터 호출과 컴포넌트 조합만 하고, 로직은 `features/`로
- **Colocation**: 특정 페이지에서만 쓰는 건 `_components/`처럼 언더스코어 폴더로 같은 위치에 둠 (라우트로 노출 안 됨)
- **서버/클라이언트 경계 명확화**: DB 접근 코드는 `import "server-only"`를 붙여 클라이언트 번들 유입 차단
- **환경 변수 검증**: `env.ts`에서 zod로 검증해 배포 전에 누락 발견
- **import 방향**: `app → features → components/lib` 단방향만 허용 (순환 방지, ESLint로 강제 가능)
- **절대 경로**: `tsconfig`의 `paths`로 `@/`별칭 사용
- **Route Group**: `(auth)`, `(dashboard)`로 레이아웃을 URL 영향 없이 분리

**규모별 가이드**
- 소규모: `app/` + `components/` + `lib/`로 충분
- 중대형: 위 구조처럼 `features/` 도입
- 여러 앱·공유 패키지: Turborepo 모노레포 (`apps/web`, `packages/ui`, `packages/config`)

팀 규모나 백엔드를 Next.js 안에 둘지 별도 서버로 둘지 알려주시면 그에 맞춰 조정해 드릴게요.