```dataview
TABLE file.name AS "노트 이름", file.mtime AS "수정일"
FROM "03. Daily Note"
SORT file.mtime DESC
```
```dataviewjs

```
```dataviewjs
// 1. 노트 조회
const pages = dv.pages('"TodoList"')
  //.where(p => p.type === "monthly-calendar")
  //.sort(p => p["month-label"], "asc");

// 2. 헤더 정의
const headers = ["📅 연월", "📝 노트", "📅 생성일"];

// 3. 데이터 rows 생성
const rows = pages.map(p => [
  p["month-label"],   // frontmatter 값
  p.file.link,        // 노트 링크
  p.file.ctime        // 생성일
]);

// 4. 표 출력
dv.table(headers, rows);
```



| 일   | 월   | 화   | 수   | 목   | 금   | 토   |
| --- | --- | --- | --- | --- | --- | --- |
| 1   | 2   | 3   | 4   | 5   | 6   | 7   |
|     |     |     |     |     |     |     |
| 8   | 9   | 11  | 12  | 13  | 14  | 15  |
|     |     |     |     |     |     |     |
| 16  | 17  | 18  | 19  | 20  | 21  | 22  |
|     |     |     |     |     |     |     |
| 21  | 22  | 23  | 24  | 25  | 26  | 27  |
|     |     |     |     |     |     |     |
| 28  | 29  | 30  | 31  |     |     |     |
|     |     |     |     |     |     |     |
