
---

### 한 줄 요약

||dataview|dataviewjs|
|---|---|---|
|방식|**전용 쿼리 언어 (DQL)**|**JavaScript 코드**|
|난이도|쉬움|어려움|
|유연성|제한적|무제한|

---

### dataview — 전용 쿼리 언어 (DQL)

SQL과 비슷한 문법으로 노트를 조회합니다. 간단한 표, 목록, 달력을 빠르게 만들 때 적합합니다.

````
```dataview
TABLE date, tags, file.mtime AS "수정일"
FROM "Daily Notes"
WHERE date >= date(today) - dur(7 days)
SORT date DESC
LIMIT 10
```
````

**사용 가능한 명령어**

```
TABLE   → 표
LIST    → 목록
TASK    → 체크박스 목록
CALENDAR → 달력 점 표시

FROM    → 출처 (폴더, 태그)
WHERE   → 조건 필터
SORT    → 정렬
GROUP BY → 그룹화
LIMIT   → 개수 제한
FLATTEN → 배열 펼치기
```

**예시**

```sql
-- 이번 주 Daily Note 목록
TABLE date, day
FROM "Daily Notes"
WHERE week = "W15"
SORT date ASC

-- 태그별 그룹화
TABLE rows.file.link AS "노트"
FROM #업무
GROUP BY file.folder AS "폴더"
```

---

### dataviewjs — JavaScript

JavaScript를 그대로 사용합니다. 파일 내용 읽기, HTML 생성, 복잡한 연산 등 뭐든 가능합니다.

````
```dataviewjs
const pages = dv.pages('"Daily Notes"')
  .where(p => p.date)
  .sort(p => p.date, "desc");

dv.table(
  ["날짜", "노트"],
  pages.map(p => [p.date, p.file.link])
);
```
````

**dataview에서 안 되는 것들이 dataviewjs에서 가능**

```javascript
// ✅ 파일 내용 직접 읽기
const content = await app.vault.read(file);

// ✅ HTML 직접 생성
dv.el("div", "<b>굵은 텍스트</b>");

// ✅ 복잡한 연산 / 가공
const grouped = pages.groupBy(p => p.tags);

// ✅ 외부 API 호출
const res = await fetch("https://api.example.com");

// ✅ 달력 그리드, textarea 등 커스텀 UI
```

---

### 핵심 차이 비교

|항목|dataview|dataviewjs|
|---|---|---|
|문법|DQL (SQL 유사)|JavaScript|
|설정 필요|없음|Enable JavaScript Queries ✅|
|파일 내용 읽기|❌|✅ `app.vault.read()`|
|HTML 출력|❌|✅ `dv.el()`|
|커스텀 UI|❌|✅|
|복잡한 연산|제한적|✅ 무제한|
|frontmatter 조회|✅|✅|
|속도|빠름|상대적으로 느림|
|가독성|높음|낮음|

---

### 언제 뭘 쓸까

```
단순 조회, 목록, 표
→ dataview 로 충분
   예) 이번 달 노트 목록, 태그별 분류

파일 내용 파싱, 커스텀 UI, 복잡한 가공
→ dataviewjs 필요
   예) 달력 그리드, 노트 안의 표 읽기, textarea 입력
```

---

### 같은 결과, 두 가지 방법 비교

**Daily Notes 최근 7개 표로 보기**

````
```dataview
TABLE date, day
FROM "Daily Notes"
SORT date DESC
LIMIT 7
```
````

````
```dataviewjs
const pages = dv.pages('"Daily Notes"')
  .sort(p => p.date, "desc")
  .slice(0, 7);

dv.table(
  ["날짜", "요일"],
  pages.map(p => [p.date, p.day])
);
```
````

결과는 동일하지만 이런 단순한 경우엔 **dataview가 훨씬 간결**합니다. 반대로 이번에 만든 달력 그리드처럼 HTML을 직접 그려야 하는 경우엔 **dataviewjs가 유일한 선택**입니다.