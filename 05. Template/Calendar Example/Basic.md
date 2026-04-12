📌 page.file 내장 속성
page.file.name   // 파일명 (확장자 제외)
page.file.path   // 전체 경로
page.file.link   // [[링크]] 형태로 자동 변환
page.file.mtime  // 수정일
page.file.ctime  // 생성일
page.file.size   // 파일 크기 (bytes)
page.file.tags   // 태그 배열

---
## 실전 예시 — pages → table 흐름

// 1. 노트 조회
const pages = dv.pages('"Calendar"')
  .where(p => p.type === "monthly-calendar")
  .sort(p => p["month-label"], "asc");

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
