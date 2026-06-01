
### 1. CLAUDE.md
- **목표:** Next.js 풀스택 쇼핑몰 MVP를 6명 전문 에이전트(기획→설계→구현→QA 파이프라인)로 자동 빌드.
- **조율 주체는 메인 Claude다.** `Agent` 도구를 가진 유일한 주체이므로, 메인 Claude가 각 Phase의 전문 서브에이전트를 직접 스폰한다. 서브에이전트는 격리 실행되어 서로 메시지를 주고받거나 다른 에이전트를 스폰할 수 없으므로, 모든 에이전트 간 조율은 `_workspace/` 파일 인계 + 순차 재스폰으로 이뤄진다. 절차의 단일 출처는 `shopping-mall-builder` 스킬이다.
- **에이전트 구성 (모두 서브에이전트, 메인 Claude가 스폰):*
	- 기획: `product-manager`
	- 설계: `data-architect` → `ux-designer` (순차, ux는 확정 스키마를 읽음)
	- 구현: `backend-engineer` → `frontend-engineer` (순차, fe는 확정 API 스펙을 읽음)
	- 검증: `qa-validator`
- **작업 디렉토리 규약:*
	- 코드: 프로젝트 루트 (`src/`, `prisma/`, `package.json` 등)
	- 중간 산출물: `_workspace/{phase}_{agent}_{artifact}.{ext}` (보존)
	- 이전 실행 백업: `_workspace_prev_{YYYYMMDD-HHMM}/`

### 2. 에이전트 구성 (모두 서브에이전트, 메인 Claude가 스폰)
#### (1) 에이전트 구성
- 기획(Phase1): `product-manager`
- 설계(Phase2): `data-architect` → `ux-designer` (순차, ux는 확정 스키마를 읽음)
- 구현(Phase3): `backend-engineer` → `frontend-engineer` (순차, fe는 확정 API 스펙을 읽음)
- 검증(Phase4): `qa-validator`
#### (2) 에이전트 입출력 문서

##### 1) product-manager (Phase1)
- 사용 스킬: write-prd
- 출력문서
```
_workspace/01_pm_prd.md
_workspace/01_pm_user_stories.md
_workspace/01_pm_screen_list.md
```

##### 2) data-architect (Phase2)
- 사용 스킬: design-data-schema
- 입력문서
```
_workspace/01_pm_*.md
```

- 출력문서
```
_workspace/02_data_schema.prisma
_workspace/02_data_erd.md
_workspace/02_data_queries.md
```

- 기타
	3개 파일 확인 후, ux-designer (확정된 스키마를 입력으로)

#### 3) ux-designer (Phase2)
- 사용 스킬: design-ux-flow
- 입력문서
```
_workspace/01_pm_*.md
_workspace/02_data_schema.prisma
```

- 출력문서
```
_workspace/02_ux_wireframes.md
_workspace/02_ux_design_system.md
_workspace/02_ux_interaction_patterns.md
```

- 기타
	ux 산출물에 '스키마 보완 요청'이 있으면 → data-architect를 1회 재스폰(해당 필드만 추가)한 뒤 다음 Phase로. 없으면 바로 진행. 완료 조건: 6개 02_* 파일 존재.

#### 4) backend-engineer (Phase3)
- 사용 스킬: build-backend-api
- 입력문서
```
_workspace/01_pm_*.md
_workspace/02_data_*.md
_workspace/02_data_schema.prisma
```

- 출력문서
```
_workspace/03_be_api_spec.md (API 명세)
_workspace/03_be_known_issues.md (이슈)
```

- 기타
	빌드/타입체크 통과 + 03_be_api_spec.md 존재 확인 후, frontend-engineer

#### 5) frontend-engineer (Phase3)
- 사용 스킬: build-frontend-ui
- 입력문서
```
_workspace/02_ux_*.md
_workspace/03_be_api_spec.md
backend가 작성한 실제 코드(src/actions/, src/lib/)
```

- 출력문서
```
_workspace/03_fe_components.md (구현 상태)
```

- 기타
	frontend 산출물에 'API 불일치'가 있으면 → backend-engineer를 1회 재스폰해 스펙/코드를 정렬한 뒤, 필요 시 frontend를 1회 더 스폰. 완료 조건: 빌드+타입체크 통과, package.json·prisma/schema.prisma·src/actions/·src/app/·src/components/·03_be_api_spec.md·03_be_known_issues.md·03_fe_components.md 존재.

#### 6) qa-validator (Phase4)
- 사용 스킬: qa-integration-check

- 출력문서
```
_workspace/04_qa_report.md
```

- 에러 핸들링
QA가 차단 이슈를 보고하면 → 권고에 따라 backend/frontend를 1회 재스폰해 수정 → qa-validator를 1회 재실행. (재시도 한도는 [에러 핸들링](file:///c%3A/ClaudeProject/KakaoHarness/KakaoHarness/.claude/skills/shopping-mall-builder/SKILL.md#%EC%97%90%EB%9F%AC-%ED%95%B8%EB%93%A4%EB%A7%81) 참고.)

### 3. 스킬
shopping-mall-builder
write-prd
design-data-schema
design-ux-flow
build-backend-api
build-frontend-ui
qa-integration-check
#### (1) write-prd
##### 1) Description
쇼핑몰 MVP의 PRD, 사용자 스토리, 화면 목록을 작성한다. product-manager 에이전트가 사용하는 핵심 스킬. 사용자가 "쇼핑몰 기획", "PRD 작성", "사용자 스토리 정리", "화면 목록 만들기" 류 작업을 요청하거나, 모호한 쇼핑몰 요청을 명세로 변환해야 할 때 사용
##### 2) 출력 위치
- `_workspace/01_pm_prd.md`: PRD
- `_workspace/01_pm_user_stories.md`: 사용자 스토리
- `_workspace/01_pm_screen_list.md`: 화면 목록

#### (2) shopping-mall-builder
##### 1) Description
Next.js 풀스택 쇼핑몰 MVP를 처음부터 끝까지 만들어내는 오케스트레이션 절차서. 6명 에이전트 팀을 기획→설계→구현→QA 파이프라인으로 조율한다. 

##### 2) 에이전트 간 조율 방법
① **파일 인계** — 앞 단계가 `_workspace/`에 산출물을 쓰고, 다음 단계가 그것을 읽는다.
② **순차 재실행** — 어떤 에이전트가 산출물에 "선행 단계에 X가 필요하다"는 요청을 남기면, 너가 그 선행 에이전트를 다시 스폰해 반영한 뒤 진행한다.

##### 3) Phase
Phase 0 — 컨텍스트 확인 및 모드 결정
Phase 1 — 기획 (단일)
Phase 2 — 설계 (순차: data → ux)
Phase 3 — 구현 (순차: backend → frontend)
Phase 4 — QA (단일)
Phase 5 — 결과 보고

##### 4) 산출물(데이터 전달) 프로토콜
- 파일 기반이 유일한 인계 수단이다. 모든 중간 산출물은 `_workspace/{phase}_{agent}_{artifact}.{ext}` 형식.
- 단일 에이전트 결과는 Agent 도구의 반환 메시지로도 받는다(요약·차단 이슈). 단, 다음 에이전트가 쓸 실제 데이터는 항상 파일에 있어야 한다.
- 후행→선행 요구는 후행 에이전트의 산출물 안 지정된 섹션(예: '스키마 보완 요청', 'API 불일치')에 남기게 하고, 너가 읽어서 선행 재스폰 여부를 결정한다.
- `_workspace/`는 사후 검증/감사를 위해 보존한다(사용자가 명시 삭제 요청 전까지).

##### 5) Phase 간 검증 (매 Phase 종료 시)
- 약속된 산출물 파일 존재를 Glob/Read로 확인. "에이전트가 완료라고 말함"을 신뢰하지 말고 실제 파일을 본다.
- 누락 시 동일 에이전트를 컨텍스트 추가하여 1회 재스폰.
- 2회째 실패 시 차단 이슈로 분류. QA만 남았다면 진행하고 최종 보고에 명시, 아니면 사용자에게 보고 후 판단.