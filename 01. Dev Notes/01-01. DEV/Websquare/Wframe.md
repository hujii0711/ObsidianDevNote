
### 1. 속성
- frameModal
$p.openPopup() 메소드로 팝업창을 표시할때 WFrame 영역 내에서 표시할지 설정합니다.
$p.openPopup() 메소드의 frameModal 파라미터를 설정한 경우 동작하지 않습니다.

`true`: 팝업창이 WFrame 영역을 기준으로 표시됩니다.
`false (default)`: 팝업창이 웹브라우저 영역을 기준으로 표시됩니다.

- scope
WFrame의 `scope` 속성 기본값은 **true**입니다.

교육 자료에 따르면, 그 이유는 WFrame이 파일을 include하는 개념으로 부모 페이지와 하나의 영역 내에 구성되기 때문에, 부모 페이지와 자식 페이지에 동일한 id를 가진 컴포넌트가 있으면 id 중복 문제가 발생하는데, 이를 방지하기 위해 scope 기능을 사용한다고 합니다. scope 속성을 true로 사용하면 적용되며, default도 true라고 설명하고 있습니다.

참고로 알아두면 좋은 점)
1) scope="true"일 때는 WFrame 내부 컴포넌트의 실제 렌더링 id가 `wframeID + "_" + originalID` 형태로 자동 변경되어 부모/자식 페이지 간 id 충돌이 방지됩니다.
2) 다만 이 방식은 컴포넌트를 자바스크립트의 `with`문으로 감싸는 구조라, 성능이나 스코프 관리 측면에서 트레이드오프가 있으므로, 프로젝트 상황에 따라 명시적으로 `scope="false"`로 지정해 끄는 경우도 있습니다.

- scopeExternal
WFrame 컴포넌트에 연결된 WebSquare XML 페이지에서 사용된 외부 JS 에 Scope 적용 여부를 설정합니다.

`true`: WebSquare XML 페이지에서 사용된 외부 JS 에 Scope 를 적용합니다.
`false (default)`: WebSquare XML 페이지에서 사용된 외부 JS 에 Scope 를 적용하지 않습니다.


- scopeInherit
중첩된 Frame 구조일때 하위 Frame 에서 참조할 수 있는 상위 WFrame 의 범위를 설정합니다.
하위 Frame 에서 스크립트 실행 시 $p.main() 메소드의 Scope 반환 범위와 컴포넌트 참조 범위를 결정합니다.
IFrame 을 WFrame 으로 전환할때 IFrame 에 정의 되었던 전역 변수 접근을 용이하게 하고, 코드를 최대한 적게 바꿀 수 있습니다.

|Value|Description|
|---|---|
|`"none" (default)`|하위 Frame 에서 현재 WFrame 을 $p.main() 메소드로 접근할 수 없고, 컴포넌트도 직접 참조할 수 없습니다.|
|`"all"`|하위 Frame 에서 현재 WFrame 을 $p.main() 메소드로 접근할 수 있고, 컴포넌트도 직접 참조할 수 있습니다.|
|`"api"`|하위 Frame 에서 현재 WFrame 을 $p.main() 메소드로 접근할 수 있지만, 컴포넌트는 직접 참조할 수 없습니다.|
|`"component"`|하위 Frame 에서 현재 WFrame 을 $p.main() 메소드로 접근할 수 없지만, 컴포넌트는 직접 참조할 수 있습니다.|
|`"recursive"`|하위 Frame 에서 현재 WFrame 을 $p.main() 메소드로 접근할 수 없지만, 컴포넌트는 직접 참조할 수 있습니다.|

### 2. 자식 WFrame이 부모 Wframe의 Scope을 상속
WFrame을 포함하는 화면을 부모 WFrame의 소스(`src`) 화면으로 생성할 경우, 부모 WFrame이 포함한 WFrame의 소스(`src`) 화면에서 가장 가까운 부모의 WFrame의 객체 및 영역에 접근하도록 허용할 수 있습니다.

소스(`src`) 화면을 포함하는 부모 WFrame의 `scopeInherit` 옵션을 아래와 같이 설정하십시오.

- `"none"` (기본 값) : 부모 영역의 객체를 자동으로 참조하지 않고, `$p.main()` 함수는 (`$p.top()` 함수와 동일하게 실행되어) 최상단 페이지에 접근 가능.
![[Pasted image 20260712223743.png]]

- `"api"` : 부모 영역의 객체를 자동으로 참조하지 않지만, `$p.main()` 함수는 부모 영역에 접근 가능.
![[Pasted image 20260712223754.png]]

- `"all"` : 부모 영역의 객체를 자동으로 참조하고, `$p.main()` 함수도 부모 영역에 접근 가능.
![[Pasted image 20260712223811.png]]

- `"component"` : 부모 영역의 객체를 자동으로 참조하지만, `$p.main()` 함수는 (`$p.top()` 함수와 동일하게 실행되어) 최상단 페이지에 접근 가능.
 ![[Pasted image 20260712223846.png]]