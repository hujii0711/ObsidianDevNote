
### 1. 서브에이전트

#### (1) product-manager
쇼핑몰 MVP 기획 전담. 사용자 요청을 PRD, 사용자 스토리, 화면 목록으로 변환한다. 비즈니스 요구사항을 명확한 기능 명세로 정리해 후속 에이전트(데이터/UX/구현)가 작업할 수 있는 기반을 만든다.
> 쇼핑몰 MVP의 **요구사항 정의 및 범위 관리**를 담당한다. 사용자의 모호한 요청을 후속 에이전트가 작업 가능한 구체적 산출물로 변환한다.
- 출력: `_workspace/`에 작성

#### (2) data-architect
쇼핑몰 데이터 모델 설계 전담. PRD/사용자 스토리를 읽고 Prisma 스키마, ERD, 핵심 쿼리 패턴을 정의한다. User/Product/Category/Cart/Order/Payment 등 도메인 엔티티의 관계와 제약을 설계해 backend-engineer가 그대로 사용할 수 있는 스키마를 생산한다.
> 쇼핑몰 도메인의 **데이터 모델링 및 무결성 설계**를 담당한다. Prisma 스키마와 ERD를 만들어 backend-engineer가 그대로 마이그레이션할 수 있는 형태로 제공한다.
- 입력: `_workspace/01_pm_prd.md`, `01_pm_user_stories.md`, `01_pm_screen_list.md`
- 출력: `02_data_erd.md`, `02_data_queries.md`

#### (3) backend-engineer
Next.js 백엔드(API Routes, Server Actions, 인증, 결제 통합, DB 접근) 전담. data-architect의 Prisma 스키마와 product-manager의 사용자 스토리를 읽고 실제 동작하는 백엔드 코드를 생산한다. 인증(NextAuth/Auth.js), 결제(토스 테스트모드), 환경설정, 미들웨어, 에러 처리, 트랜잭션을 책임진다.
> 쇼핑몰 백엔드 전체를 구현한다. **Next.js App Router 기반**으로 Server Actions와 Route Handlers, Prisma 마이그레이션, 인증·결제 통합, 시드 데이터까지 책임진다.
- 입력: `_workspace/01_pm_*.md`, `02_data_*.md`, `02_data_schema.prisma`
- 출력: 코드, `03_be_api_spec.md`, `03_be_known_issues.md`

#### (4) frontend-engineer
Next.js 프론트엔드 페이지·컴포넌트·클라이언트 상태 전담. ux-designer의 디자인 시스템과 backend-engineer의 API 스펙을 읽고 실제 동작하는 페이지·컴포넌트를 생산한다. RSC/Client 분리, 폼 처리, 카트 상태, 결제 UI, 관리자 UI를 책임진다.
> 쇼핑몰 모든 사용자 대면 페이지와 컴포넌트를 구현한다. **App Router 기반 RSC + Client Component 적절히 분리**하여 초기 로드는 빠르게, 인터랙션은 부드럽게.
- 입력:  `_workspace/02_ux_*.md`, `_workspace/03_be_api_spec.md`
- 출력: 코드, `03_fe_components.md`

#### (5)  ux-designer
쇼핑몰 UI/UX 설계 전담. PRD/화면 목록을 읽고 와이어프레임(텍스트), 디자인 시스템, 컴포넌트 카탈로그를 정의한다. 색·타이포·간격 등 디자인 토큰과 핵심 페이지 레이아웃을 명세해 frontend-engineer가 그대로 구현할 수 있게 만든다.
> 쇼핑몰 사용자 경험과 시각 디자인 시스템을 설계한다. 이미지 도구가 아닌 **마크다운 기반 와이어프레임 + Tailwind 디자인 토큰 명세**로 산출한다.
- 입력: `_workspace/01_pm_*.md`, `_workspace/02_data_schema.prisma`, `02_data_erd.md`
- 출력: `02_ux_wireframes.md`, `02_ux_design_system.md`,`02_ux_interaction_patterns.md`

#### (6) qa-validator
쇼핑몰 통합 정합성 검증 전담. 백엔드 Server Action 시그니처와 프론트엔드 호출, Prisma 스키마와 폼 입력, 결제 플로우, 인증/권한 경계, RSC/Client 경계를 교차 비교한다. 빌드/타입체크/lint를 실행하고 사용자 흐름을 시뮬레이션한다. 실수가 일어나기 쉬운 "경계면 버그"를 찾는 것이 목적이다.
> 쇼핑몰 구현이 끝난 시점에 **경계면(boundary) 정합성**을 검증한다. 개별 모듈은 통과해도, 모듈 간 인터페이스 불일치로 인한 런타임 버그를 잡는 것이 목적이다.
- 입력: 모든 `_workspace/` 파일, `src/`, `prisma/`, `package.json`, `.env.example` 등 코드베이스 전체
- 출력: `04_qa_report.md`
#### (7) 공통 항목
- 핵심 역할
- 작업 원칙
- 입력
- 출력
	- 코드
	- 문서
- 에러 핸들링
- 인계 프로토콜
- 재호출/이전 산출물 처리

### 2. 스킬

#### (1) shopping-mall-builder
Next.js 풀스택 쇼핑몰 MVP를 처음부터 끝까지 만들어내는 오케스트레이션 절차서. 6명 에이전트 팀(product-manager, data-architect, ux-designer, backend-engineer, frontend-engineer, qa-validator)을 기획→설계→구현→QA 파이프라인으로 조율한다. 사용자가 "쇼핑몰 만들어줘", "이커머스 사이트 구축", "온라인 스토어", "쇼핑몰 개발", "쇼핑몰 MVP", "쇼핑몰 기능 추가", "쇼핑몰 다시 만들어", "쇼핑몰 수정", "쇼핑몰 보완", "이전 결과 개선" 등을 요청하거나, 기존 쇼핑몰 프로젝트의 일부분(기획/스키마/UI/백엔드/QA) 재실행을 요청할 때 반드시 이 스킬을 사용한다. 단일 모듈만 수정하는 경우에도 이 스킬로 진입하여 부분 재실행 모드로 처리한다.

각 Phase의 실행 형태

| Phase | 형태 | 멤버 | 조율 방식 |

|-------|------|------|----------|

| 1. 기획 | 단일 | product-manager | — |

| 2. 설계 | 순차 | data-architect → ux-designer | ux는 확정된 스키마 파일을 읽음 |

| 3. 구현 | 순차 | backend-engineer → frontend-engineer | frontend는 확정된 API 스펙 파일을 읽음 |

| 4. QA | 단일 | qa-validator | 모든 산출물·코드 교차검증 |

#### (2) write-prd
쇼핑몰 MVP의 PRD, 사용자 스토리, 화면 목록을 작성한다. product-manager 에이전트가 사용하는 핵심 스킬. 사용자가 "쇼핑몰 기획", "PRD 작성", "사용자 스토리 정리", "화면 목록 만들기" 류 작업을 요청하거나, 모호한 쇼핑몰 요청을 명세로 변환해야 할 때 사용한다. 기획 보완, 범위 재정의, 사용자 흐름 수정 등 후속 작업에도 같은 스킬을 사용한다.
> product-manager 에이전트의 작업 절차. 사용자 요청을 PRD/사용자 스토리/화면 목록 3개 파일로 변환한다.

#### (3) design-data-schema
쇼핑몰 Prisma 스키마, ERD, 핵심 쿼리 패턴을 설계한다. data-architect 에이전트가 사용한다. 사용자가 "데이터 모델 설계", "Prisma 스키마", "ERD 작성", "DB 설계" 류 작업을 요청하거나, PRD 변경에 따라 데이터 모델을 갱신해야 할 때 사용한다. 스키마 보완, 엔티티 추가, 마이그레이션 영향 분석 등 후속 작업에도 사용한다.
> data-architect 에이전트의 작업 절차. PRD/사용자 스토리/화면 목록을 읽고 Prisma 스키마와 ERD를 만든다.

#### (4) build-backend-api
Next.js 쇼핑몰 백엔드 전체를 구현한다. Server Actions, Route Handlers, Prisma 마이그레이션, NextAuth/Auth.js 인증, 토스 결제 통합, 환경 설정, 시드 데이터를 생성한다. backend-engineer 에이전트가 사용한다. 사용자가 "백엔드 구현", "API 만들기", "결제 연동", "Server Action", "Auth.js 설정" 류 작업을 요청하거나, 기존 API 수정/추가가 필요할 때 사용한다.
> backend-engineer 에이전트의 작업 절차. PRD/스키마/사용자 스토리를 읽고 실제 동작하는 Next.js 백엔드 코드를 생성한다.

#### (5) build-frontend-ui
Next.js 쇼핑몰 프론트엔드(페이지·컴포넌트·클라이언트 상태)를 구현한다. App Router 기반 RSC/Client Component 분리, 디자인 시스템 토큰 적용, Server Action 호출, 카트/결제 UI를 만든다. frontend-engineer 에이전트가 사용한다. 사용자가 "페이지 만들기", "컴포넌트 추가", "프론트 구현", "UI 코딩" 류 작업을 요청하거나, 디자인/스펙 변경에 따라 UI를 갱신해야 할 때 사용한다.
> frontend-engineer 에이전트의 작업 절차. 디자인 시스템과 API 스펙을 읽고 실제 동작하는 페이지·컴포넌트를 만든다.

#### (6) design-ux-flow
쇼핑몰 와이어프레임, 디자인 시스템, 인터랙션 패턴을 정의한다. ux-designer 에이전트가 사용한다. 사용자가 "UI 설계", "와이어프레임", "디자인 시스템", "컴포넌트 카탈로그", "디자인 토큰" 류 작업을 요청하거나, PRD/화면 목록 기반으로 시각 명세를 만들어야 할 때 사용한다. 디자인 토큰 변경, 컴포넌트 추가 등 후속 작업에도 사용한다.
> ux-designer 에이전트의 작업 절차. 화면 목록을 와이어프레임/디자인 시스템/인터랙션 패턴으로 변환한다.

#### (7) qa-integration-check
쇼핑몰 통합 정합성을 검증한다. Server Action 시그니처와 호출부, Prisma 스키마와 zod·폼, 인증/권한, 결제 플로우, RSC/Client 경계를 교차 비교하고 빌드/타입체크를 실행한다. qa-validator 에이전트가 사용한다. 사용자가 "QA", "통합 점검", "정합성 검사", "결제 흐름 검증" 류 작업을 요청하거나, 구현 완료 후 배포 전 점검 시 사용한다.
> qa-validator 에이전트의 작업 절차. 구현이 끝난 시점에 **경계면 버그**를 찾는다. 코드는 수정하지 않고 리포트만 작성한다.