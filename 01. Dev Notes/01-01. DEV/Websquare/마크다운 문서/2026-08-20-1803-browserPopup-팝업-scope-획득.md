
핵심: `browserPopup`은 `getPopupWindow()`가 **scope가 아니라 진짜 `window` 객체**를 돌려줍니다. 거기서 팝업 문서의 전역(`$p`, `scwin`, `$w`, `$c`)으로 들어가면 됩니다.

### 왜 그런가 (엔진 동작)

`$c.cm.openPopup` → [popup.base.xml:126](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L126) → `$c.win.openPopup` → 최종적으로 `$p.openPopup(...)`. 엔진이 타입을 정규화합니다:

```js
"wframePopup"  → type="litewindow", frameMode="wframe"
"browserPopup" → type="browser"                      // ← 우리 케이스
```

그리고 `getPopupWindow(id)` → `popupList[i].getWindow()` 인데:

```js
getWindow = function(){ return 1 == this.useWindowPopup() ? this.popupWin
                                                         : this.popupWin.frame.getWindow(); }
useWindowPopup = function(){ return this.options.type=="browser" || this.options.type=="window"; }
```

즉 **layer/wframe 팝업은 scope 유사 객체**(`.scwin`이 바로 먹힘)를 주지만, **browserPopup은 `popupWin`(raw `window`)** 을 줍니다. 코드베이스에서도 같은 객체를 `WebSquare.util.getPopup(popupId)?.popupWin` 으로 꺼내 `focus()`/`addEventListener('beforeunload')` 를 겁니다 ([popup.base.xml:240](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L240), [:322](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L322)).

팝업 문서는 `popup.html`이 자기 창에서 `WebSquare.startApplication()`을 직접 돌리므로([websquare/popup.js](WebTop/src/main/webapp/websquare/popup.js)), 그 창의 최상위 페이지 scope가 **window 전역으로 노출**됩니다 — 엔진이 `window.scwin = {}; window.scwin.$w = api; window.$p = window.scwin.$w` 형태로 만듭니다. 팝업이 부모를 `(opener||parent).$c.popup._closePopup(...)`으로 부르는 것과 정확히 대칭입니다 ([popup.base.xml:397~](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L397)).

### 실제 코드

```js
// 1) 팝업 식별 — openPopup에 넘긴 id는 org_id로 남고,
//    실제 id에는 scope_uuid('wt_cm') 프리픽스가 붙는다
var entry = $p.getAllPopupList().find(d => d.org_id === 'MY_POP');
if (!entry) return;

// 2) browserPopup → raw window 객체
var popWin = $p.getPopupWindow(entry.id);        // === WebSquare.util.getPopup(entry.id).popupWin
if (!popWin || popWin.closed) return;

// 3) 팝업의 scope 계층
var popScope = popWin.$p;      // page scope (= pageContainer_WEBTOP.xml)
var popScwin = popWin.scwin;   // pageContainer의 scwin
var popApi   = popWin.$w;      // 팝업 창의 WebSquare API
var popCm    = popWin.$c;      // 팝업 창의 공통모듈 (창마다 별도 인스턴스)

// 4) 실제 업무화면 scwin — frameContainer 한 단계 더 내려가야 한다
var bizScwin = popWin.$p.getComponentById('frameContainer').getObj('scwin');
```

`_openPopup`은 항상 `$c.consts.PAGE_CONTAINER`를 띄우고 업무화면을 `frameContainer`에 `setSrc` 하므로([popup.base.xml:242](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L242), [pageContainer_WEBTOP.xml:919](WebTop/src/main/webapp/webTop/layout/page/pageContainer_WEBTOP.xml#L919)), `popWin.scwin` ≠ 업무화면 scwin 입니다. MDI 창용 [`$c.cm.getWindowBizScwin()`](WebTop/src/main/webapp/webTop/cm/gcc/cm.xml#L1643)의 `pageScope.frameContainer.getObj('scwin')` 과 같은 구조입니다.

### 함정 3가지

**① `$c.popup.getPopupBizScwin()`은 browserPopup에서 무조건 `""`를 반환합니다.** 목록에서 window 팝업을 명시적으로 걸러냅니다 ([popup.base.xml:580](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L580)):

```js
arrPopList.filter(function(objPopWin) { return (objPopWin.constructor.name == "Object"); });
```
raw `window`는 `constructor.name === "Window"`라 탈락 → layer/wframe 전용 헬퍼입니다.

**② 타이밍.** `openPopup` 직후엔 `popWin.$p`가 `undefined`입니다(새 창에서 `startApplication()`이 비동기로 도는 중). `_openPopup` 자신도 `beforeunload` 붙일 때 `setTimeout(..., 1000)`을 씁니다. 폴링 헬퍼를 쓰세요:

```js
scwin.getBrowserPopupScope = async function(orgId, timeout) {
    var deadline = Date.now() + (timeout || 5000);
    while (Date.now() < deadline) {
        var e = ($p.getAllPopupList() || []).find(d => d.org_id === orgId);
        var w = e ? $p.getPopupWindow(e.id) : null;
        if (w && !w.closed && w.$p && w.$p.getComponentById('frameContainer')) return w.$p;
        await $c.util.wait($p, 100);
    }
    return null;
};
```
업무화면 scwin까지 필요하면 `frameContainer.getObj('scwin')`이 `undefined`가 아닐 때까지 한 조건 더 거세요(`setSrc` 완료 시점이 더 늦습니다).

**③ 중복 오픈 차단 로직과 겹칩니다.** window 팝업은 `window.windowWebTopPopup[options.id]`에 popupId를 캐싱해두고 이미 있으면 `focus()`만 하고 return 합니다 ([popup.base.xml:236~](WebTop/src/main/webapp/webTop/cm/gcc/base/popup.base.xml#L236)). 이 맵을 직접 읽어 popupId를 얻는 것도 가능하지만, 닫힘 감지가 `beforeunload` 리스너에 의존하므로 stale 엔트리가 남을 수 있습니다 — `getAllPopupList()` 경유가 더 안전합니다.

> 양방향으로 자주 호출할 거면 폴링보다, 팝업 쪽 pageContainer의 `onpageload`에서 `(opener||parent)`에 자기 scope를 등록해두는 핸드셰이크가 훨씬 안정적입니다. 이미 `closePopup`이 `opener || parent` 패턴을 쓰고 있으니 결이 맞습니다.
