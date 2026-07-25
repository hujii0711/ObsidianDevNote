코드를 다 찾았습니다. 초보자 눈높이에 맞춰 3단계로 설명드릴게요.

---

## 1. `m_pPlatformAPI`가 무엇인가?

[ScriptFunction.h:29-30](IztScript/ScriptFunction.h#L29-L30)에 선언되어 있습니다:

```cpp
private:
	IIztScreenAPI	*m_pScreenAPI;      // 화면(Screen) 하나에 대한 API
	IIztPlatformAPI	*m_pPlatformAPI;    // 플랫폼(메인 프레임) 전체에 대한 API
```

- `IIztPlatformAPI`라는 **인터페이스를 가리키는 포인터**입니다.
- 그런데 [ScriptFunction.h:11](IztScript/ScriptFunction.h#L11)을 보면 `interface IIztPlatformAPI;`라고 **이름만 미리 선언(전방 선언, forward declaration)**되어 있을 뿐, 실제 정의는 이 프로젝트(IztScript) 안에 없습니다. 즉, **"ShowPane 같은 함수들이 있다"는 약속(계약)만 알고, 실제 구현은 다른 모듈(플랫폼 본체 EXE ― 메인 프레임 프로그램)에 있습니다.**

### 누가, 언제 채워주나?

[ScriptFunction.h:45](IztScript/ScriptFunction.h#L45)를 보세요:

```cpp
void SetPlatform(IIztPlatformAPI *pApi){m_pPlatformAPI = pApi;};
```

프로그램이 시작될 때 **플랫폼 본체가 "내가 플랫폼 API 구현체야, 이 포인터 써"라고 넘겨주면**(주입, dependency injection) 그때부터 `m_pPlatformAPI`가 실제 객체를 가리키게 됩니다. IztScript DLL은 플랫폼의 내부 구조를 전혀 몰라도, 이 포인터 하나로 "창 보여줘/숨겨줘" 같은 요청을 플랫폼에 전달할 수 있습니다.

---

## 2. 호출하는 함수/화면에 따라 맥락이 달라지나?

**아니요, `m_pPlatformAPI` 자체는 달라지지 않습니다.** 위의 두 멤버를 비교하면 이해가 쉽습니다:

| 멤버               | 가리키는 대상                   | 화면마다 다른가?           |
| ---------------- | ------------------------- | ------------------- |
| `m_pScreenAPI`   | **자기 화면 하나** (SetAPI로 주입) | O 화면마다 다름           |
| `m_pPlatformAPI` | **플랫폼 메인 프레임 (단 하나)**     | X 모든 화면이 같은 객체를 가리킴 |

- 화면(Screen)이 열릴 때마다 그 화면 전용 `CScriptFunction` 인스턴스가 만들어지지만, 그 안의 `m_pPlatformAPI`는 **전부 동일한 플랫폼 객체 하나**를 가리킵니다.
- 그래서 A화면에서 `ShowPane("AlimiFrame", ...)`을 부르든 B화면에서 부르든, **도착지는 같은 메인 프레임**이고 결과도 같습니다. `ShowPane`이 조작하는 것(우측 확장 프레임, 알리미 프레임 등)이 애초에 특정 화면 소속이 아니라 **메인 창의 공용 패널**이기 때문입니다.
- 반대로 "이 화면의 컨트롤 값을 바꿔줘" 같은 화면 종속 작업은 `m_pScreenAPI` 쪽으로 갑니다. 이게 두 포인터를 나눠 놓은 이유입니다.

---

## 3. 실제 화면에서 ShowPane을 호출하는 예시

### 스크립트 → C++ 연결 고리

먼저 [ScriptFunction.cpp:514](IztScript/ScriptFunction.cpp#L514)에 이런 매크로가 있습니다:

```cpp
DISP_FUNCTION_ID(CScriptFunction, "ShowPane", dispidShowPane, ShowPane, VT_EMPTY, VTS_BSTR VTS_BOOL VTS_BOOL)
```

이것은 MFC의 **디스패치 맵**으로, "화면 스크립트에서 `ShowPane`이라는 이름이 호출되면 C++의 `CScriptFunction::ShowPane`을 실행하라"는 등록입니다. 덕분에 화면 개발자는 C++을 몰라도 스크립트에서 이 함수를 쓸 수 있습니다.

### 실제 화면 예시 ― 업무팩 시작 화면

[SYPK-BzPkStart.scn:1229-1237](Solution/System/Screen/SYPK-BzPkStart.scn#L1229-L1237) (DIY 버튼 클릭):

```javascript
function Diy_Btn_OnClick(){
	this.ShowPane("RightExtendFrame", false, true);        // 우측 확장 프레임 닫기
	this.ShowPane("RightExtendSecondFrame", false, true);  // 우측 두 번째 확장 프레임 닫기
	Popup("SYPK-DiyPopup","");                             // DIY 설정 팝업 호출
}
```

같은 파일 [2004-2007행](Solution/System/Screen/SYPK-BzPkStart.scn#L2004-L2007) (종료 버튼):

```javascript
function End_Btn_OnClick(){
	this.ShowPane("ScreenWork", false, true);   // 작업 영역 프레임 숨기기
}
```

### 전체 흐름 정리

```
[화면 스크립트]  Diy_Btn_OnClick()에서 this.ShowPane("RightExtendFrame", false, true)
      ↓  스크립트 엔진이 이름 "ShowPane"으로 디스패치 (DISP_FUNCTION_ID, dispid 257)
[IztScript DLL]  CScriptFunction::ShowPane(frameName, bShow, bRefresh)   ← 질문하신 그 함수
      ↓  포인터로 위임
[플랫폼 본체]    m_pPlatformAPI->ShowPane(...)  → 메인 프레임이 실제로 패널을 숨기고 갱신
```

즉 이 시나리오는: 사용자가 DIY 버튼을 누르면 → 팝업을 띄우기 전에 **오른쪽에 펼쳐져 있던 확장 패널들을 먼저 닫는** 동작입니다. `false`가 `bShow`(숨김), `true`가 `bRefresh`(즉시 화면 갱신)에 해당합니다.

---

## 요약

1. `m_pPlatformAPI` = 플랫폼 본체(메인 프레임)가 시작 시 `SetPlatform()`으로 꽂아주는 **플랫폼 API 구현체를 가리키는 포인터**. 정의는 이 프로젝트 밖에 있음.
2. 어느 함수/화면에서 호출하든 **같은 플랫폼 객체 하나**를 가리키므로 맥락이 바뀌지 않음. 화면별 맥락이 필요한 작업은 `m_pScreenAPI`가 담당.
3. 실전 예: SYPK-BzPkStart 화면의 DIY 버튼이 `this.ShowPane("RightExtendFrame", false, true)`로 우측 패널을 닫고 팝업을 띄움.


---

앞선 프레임 계층구조 분석과 연결하면 이 두 포인터가 왜 존재하는지 훨씬 명확해집니다. 다시 정리해 드릴게요.

# m_pScreenAPI vs m_pPlatformAPI ― 프레임 계층구조로 다시 보기

## 1. 선언부 복습

[ScriptFunction.h:29-30](IztScript/ScriptFunction.h#L29-L30):

```cpp
IIztScreenAPI	*m_pScreenAPI;      // ← "내 화면(.scn) 하나"를 조작하는 통로
IIztPlatformAPI	*m_pPlatformAPI;    // ← "메인 프레임(단말 전체)"을 조작하는 통로
```

둘 다 IztScript 프로젝트 안에는 **이름만 전방 선언**되어 있고([ScriptFunction.h:11-12](IztScript/ScriptFunction.h#L11-L12)) 실제 구현은 바깥(플랫폼 본체/화면 엔진)에 있습니다. 주입 시점도 파일에 그대로 보입니다:

```cpp
void SetPlatform(IIztPlatformAPI *pApi){...};              // 기동 시 1회 ― 플랫폼이 꽂아줌
void SetAPI(IIztScreenAPI *pApi, LPDISPATCH pTran){...};   // 화면이 열릴 때마다 ― 그 화면이 꽂아줌
```

## 2. 어제 분석한 화면 계층구조에 겹쳐 보기

지난 답변의 레이아웃 그림에 두 포인터의 "관할 구역"을 표시하면 이렇습니다:

```
┌─────────────────────────────────────────────┐
│ TOP (SYTopFrame)                            │ ◀─┐
├──────┬───────────────────────────┬──────────┤   │
│ LEFT │  중앙 MDI 작업영역          │  RIGHT   │   │ 이 바깥 골격 전체
│ 메뉴 │ ┌───────────────────────┐ │ 알리미/  │   │ (KBFrame.frm의 PANE들,
│ 트리 │ │ 업무화면 KAA00010000  ◀─┼─┼─ 임시   │   │  MDI 탭, 상태바...)
│      │ │  = m_pScreenAPI 관할   │ │  저장... │   │  = m_pPlatformAPI 관할
│      │ └───────────────────────┘ │          │   │
├──────┴───────────────────────────┴──────────┤   │
│ BOTTOM (SYStatusbar)                        │ ◀─┘
└─────────────────────────────────────────────┘
```

- **`m_pScreenAPI`** = 안쪽 액자 **하나**. 지금 스크립트가 돌고 있는 그 `.scn` 화면 자신입니다.
- **`m_pPlatformAPI`** = 바깥 골격 **전부**. `KBFrame.frm`이 만든 PANE들(AlimiFrame, RightExtendFrame...), MDI 탭, 상태바, 화면 열기/닫기 자체를 관리하는 메인 프레임입니다.

핵심 규칙: **PANE은 특정 화면의 소유물이 아니라 모든 화면이 공유하는 공용 시설**이므로, PANE을 건드리는 `ShowPane`은 반드시 `m_pPlatformAPI`로 가야 합니다. 반대로 "내 화면의 컨트롤·상태"는 `m_pScreenAPI`로 갑니다.

## 3. 개수와 수명이 다르다 (가장 중요한 차이)

| | `m_pScreenAPI` | `m_pPlatformAPI` |
|---|---|---|
| 가리키는 것 | 열려 있는 `.scn` 화면 **각각** | 메인 프레임 **단 하나** |
| 개수 | 화면 수만큼 (MDI 탭 5개면 5개) | 프로세스 전체에 1개 |
| 주입 시점 | 화면 열릴 때 `SetAPI()` | 기동 시 `SetPlatform()` |
| 수명 | 화면 닫히면 소멸 | 단말 종료까지 |
| 비유 | 각 방의 리모컨 | 건물 중앙 관제실 |

MDI라서 업무화면이 동시에 여러 개 떠 있다는 점(앞선 분석 참고)이 바로 두 포인터를 나눈 이유입니다. 화면마다 자기 전용 `CScriptFunction` 인스턴스가 생기고, 그 안의 `m_pScreenAPI`는 서로 다르지만 `m_pPlatformAPI`는 전부 같은 곳을 가리킵니다.

## 4. 실제 코드에서 갈리는 기준

[ScriptFunction.cpp](IztScript/ScriptFunction.cpp)의 실사용 예를 보면 기준이 일관됩니다:

**화면 소속 작업 → `m_pScreenAPI`** (331회 사용)
```cpp
m_pScreenAPI->Alert(message, title, ...);   // 이 화면 위에 경고창
m_pScreenAPI->GetScreenName();              // 내 화면 이름
m_pScreenAPI->GetID();                      // 내 화면번호
m_pScreenAPI->IsModal();                    // 내가 모달 팝업인가?
```

**프레임/단말 공용 작업 → `m_pPlatformAPI`** (484회 사용)
```cpp
m_pPlatformAPI->ShowPane(frameName, bShow, bRefresh);  // 공용 PANE 열고 닫기
m_pPlatformAPI->MessageColor(msg, MT_SLIDEPOPUP);      // 단말 공통 슬라이드 알림
m_pPlatformAPI->GetValue(_T("Link...."));              // 전역 설정값 조회
m_pPlatformAPI->SetScreenChange(ScreenNo, ...);        // 화면 전환 (화면 밖의 일!)
```

마지막 예가 재미있는데, "화면 전환"은 화면 자신이 할 수 없습니다 ― 자기를 닫고 남을 여는 건 **MDI를 관리하는 프레임의 권한**이라 `m_pPlatformAPI`로 갑니다. 이때 [ScriptFunction.cpp:1176](IztScript/ScriptFunction.cpp#L1176)처럼 둘을 조합하기도 합니다:

```cpp
m_pPlatformAPI->SetScreenChange(ScreenNo, m_pScreenAPI->GetScreenName());
//  ↑ 프레임에게 "화면 바꿔줘"          ↑ "지금 화면(나)의 이름은 이거야"
```

## 5. 전체 호출 흐름 한 장 정리

```
[.scn 스크립트]   this.ShowPane("AlimiFrame", true, true)
      ↓ DISP_FUNCTION_ID 디스패치 맵 (이름 → C++ 함수)
[IztScript.dll]   CScriptFunction::ShowPane(...)
      ↓ 대상이 공용 PANE이므로 →  m_pPlatformAPI 로 위임
[플랫폼 본체]     KBFrame의 AlimiFrame PANE을 표시 + 갱신
```

만약 같은 스크립트가 `Alert("...")`를 불렀다면 두 번째 단계에서 `m_pScreenAPI`로 갈라졌을 뿐, 구조는 동일합니다.

## 한 줄 요약

**`m_pScreenAPI`는 "지금 이 .scn 화면 한 장"(화면마다 하나씩), `m_pPlatformAPI`는 "KBFrame.frm이 만든 단말 골격 전체"(프로세스에 하나)** 를 가리키는 위임 통로이며, 조작 대상이 **내 화면 안이냐 / 공용 프레임이냐**가 두 포인터를 가르는 유일한 기준입니다.
