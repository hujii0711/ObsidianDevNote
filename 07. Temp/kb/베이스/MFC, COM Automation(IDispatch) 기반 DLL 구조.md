
이 구조는 **"C++로 비즈니스 로직을 짜고, 그걸 COM 인터페이스로 감싸서 다른 언어/프로세스에서도 쓸 수 있게 InProcess DLL로 배포"**하는 전형적인 Windows 레거시 아키텍처입니다. VB6, VBA(Excel/AutoCAD), ASP 클래식, 스크립트 언어(JScript/VBScript) 등에서 호출하기 위해 많이 쓰였습니다.

## 1. 전체 개념

```
[클라이언트]                          [DLL 서버]
VB6/VBA/스크립트   --- IDispatch --->   MFC Automation 객체
(late binding)                         (CCmdTarget 파생)
     |                                       |
   Excel VBA                            내부 C++ 로직
   AutoCAD VBA                          (DB, 파일, 알고리즘 등)
```

- **COM(Component Object Model)**: 언어 독립적으로 객체를 노출하는 바이너리 표준
- **IDispatch**: COM 인터페이스 중 하나로, "메서드/속성 이름을 문자열로 찾아 호출"할 수 있게 해주는 late-binding(늦은 바인딩) 메커니즘. VB, VBScript, JScript처럼 vtable을 직접 알 수 없는 언어에서 COM 객체를 쓰기 위해 필수
- **In-process server(DLL)**: 클라이언트 프로세스 안에 로드되어 함수 호출처럼 빠르게 동작 (EXE 기반 Out-of-process 서버보다 오버헤드가 적음)

## 2. MFC가 제공하는 것

MFC는 COM/OLE Automation을 직접 구현하는 번거로움(IUnknown, IDispatch, 타입 정보 관리 등)을 감싸주는 헬퍼 프레임워크입니다.

### 핵심 클래스

- **`CCmdTarget`**: Automation을 지원하는 MFC 클래스의 기반. `DECLARE_DYNCREATE`, `DECLARE_OLECREATE` 등을 붙여 COM 객체로 노출 가능
- **`COleDispatchDriver`**: 반대로 클라이언트 입장에서 다른 Automation 서버를 호출할 때 사용

### 매크로 기반 디스패치 맵

```cpp
class CMyAutoObj : public CCmdTarget
{
    DECLARE_DYNCREATE(CMyAutoObj)
    DECLARE_OLECREATE(CMyAutoObj)   // CLSID, ClassFactory 연결
    DECLARE_DISPATCH_MAP()          // IDispatch 메서드/속성 매핑
    DECLARE_INTERFACE_MAP()         // COM 인터페이스 매핑

public:
    long m_nValue;                  // Automation 속성
    long Calculate(long a, long b); // Automation 메서드
};

BEGIN_DISPATCH_MAP(CMyAutoObj, CCmdTarget)
    DISP_PROPERTY(CMyAutoObj, "Value", m_nValue, VT_I4)
    DISP_FUNCTION(CMyAutoObj, "Calculate", Calculate, VT_I4, VTS_I4 VTS_I4)
END_DISPATCH_MAP()

BEGIN_INTERFACE_MAP(CMyAutoObj, CCmdTarget)
    INTERFACE_PART(CMyAutoObj, IID_IMyAutoObj, Dispatch)
END_INTERFACE_MAP()
```

- `DISP_PROPERTY`/`DISP_FUNCTION`: 실제 C++ 멤버를 IDispatch의 `Invoke()`에서 호출 가능한 항목으로 등록
- 클라이언트는 `obj.Value = 10`, `obj.Calculate(1,2)` 같은 식으로 이름 기반 호출 → 내부적으로 `GetIDsOfNames()` + `Invoke()`로 처리됨

### ODL/IDL과 타입 라이브러리

- `.odl` 또는 `.idl` 파일에 인터페이스, 메서드 시그니처를 정의 → 빌드 시 `.tlb`(타입 라이브러리) 생성
- 타입 라이브러리가 있으면 **Dual Interface**(IDispatch + vtable 직접 호출)도 지원 가능해 C++/VB6 양쪽에서 조금 더 빠르게 바인딩(early binding) 가능
- 리소스에 `.tlb`를 embed하여 DLL 하나로 배포

## 3. DLL 진입점과 등록

### `DllMain` 대신 MFC 확장 진입점

```cpp
extern "C" int APIENTRY DllMain(HINSTANCE hInstance, DWORD dwReason, LPVOID lpReserved)
{
    if (dwReason == DLL_PROCESS_ATTACH)
        AfxWinInit(...);   // MFC 초기화
    ...
}

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, LPVOID* ppv)
{
    return AfxDllGetClassObject(rclsid, riid, ppv);
}

STDAPI DllCanUnloadNow(void)
{
    return AfxDllCanUnloadNow();
}

STDAPI DllRegisterServer(void)   // regsvr32.exe로 호출
{
    COleObjectFactory::UpdateRegistryAll();
    return S_OK;
}

STDAPI DllUnregisterServer(void)
{
    COleObjectFactory::UpdateRegistryAll(FALSE);
    return S_OK;
}
```

### 배포/등록 절차

1. `MyAutoServer.dll` 빌드
2. `regsvr32 MyAutoServer.dll` 실행 → 레지스트리에 CLSID, ProgID, 타입 라이브러리 경로 등록
    - `HKCR\CLSID\{GUID}\InprocServer32` = DLL 경로
    - `HKCR\MyAutoServer.MyAutoObj` = ProgID
3. 클라이언트(VB6, VBA 등)는 `CreateObject("MyAutoServer.MyAutoObj")` 또는 `new MyAutoServer.MyAutoObj`로 인스턴스화

## 4. 클라이언트 측 호출 흐름 (예: VBA)

```vb
Dim obj As Object
Set obj = CreateObject("MyAutoServer.MyAutoObj")
obj.Value = 10
Debug.Print obj.Calculate(1, 2)
```

내부적으로는:

1. `CoCreateInstance` → DLL 로드, ClassFactory로 객체 생성
2. `IDispatch::GetIDsOfNames("Calculate")` → DISPID 조회
3. `IDispatch::Invoke(dispid, ...)` → 파라미터를 `VARIANT`로 패킹해 전달
4. MFC 디스패치 맵이 이를 실제 C++ 함수 호출로 라우팅

## 5. 특징과 트레이드오프

**장점**

- 언어 독립적: VB6, VBA, 스크립트, C++(vtable) 등 다양한 클라이언트 지원
- In-process라 EXE 서버 대비 호출 오버헤드 적음
- 레거시 Office/AutoCAD/CAD 프로그램 등 자동화(매크로) 연동에 강함

**단점 / 현대적 고려사항**

- IDispatch 호출은 이름 조회 + VARIANT 마킹 비용 때문에 순수 vtable 호출보다 느림
- 레지스트리 등록(regsvr32) 필요 → xcopy 배포 어려움, DLL Hell 위험
- 32/64비트 프로세스 간 불일치 문제 (레지스트리도 WOW6432Node로 분리됨)
- .NET/COM Interop, gRPC, REST API 등으로 대체되는 추세이며 신규 프로젝트에서는 잘 선택되지 않음 (유지보수 목적의 레거시 시스템에서 주로 발견됨)

---

혹시 실제로 이런 구조의 코드를 새로 만들거나, 기존 레거시 DLL을 분석/디버깅하는 상황이신가요? 상황에 맞춰 더 구체적으로 (예: Dual Interface 설계, 타입 라이브러리 작성, VB6/닷넷 상호운용 이슈 등) 들어갈 수 있어요.