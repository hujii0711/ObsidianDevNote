# 📆 월간 달력 뷰 (Dataview)

> 이 노트는 Dataview로 월간 달력 노트들을 조회하는 뷰입니다.
> `Calendar/` 폴더 안의 월간 노트들을 자동으로 읽어 표로 표시합니다.

---

## 📋 월별 달력 목록

월별 달력 노트 전체를 요약해서 보여줍니다.

````dataview
TABLE
  month-label AS "📅 연월",
  length(rows) AS "📌 일정 수"
FROM "Calendar"
WHERE type = "monthly-calendar"
GROUP BY month-label
SORT month-label ASC
````

---

## 📌 전체 일정 표 (모든 달)

모든 월의 일정을 날짜순으로 표시합니다.

> **⚠️ 사용 방법:**  
> 아래는 `dataviewjs`를 이용해 각 월간 노트의 **인라인 테이블**을 파싱하는 방식입니다.  
> 각 월간 노트의 `주요 일정` 테이블을 읽어옵니다.

````dataviewjs
// Calendar 폴더의 monthly-calendar 타입 노트 전체 조회
const pages = dv.pages('"Calendar"')
  .where(p => p.type === "monthly-calendar")
  .sort(p => p["month-label"], "asc");

// 표 헤더
const headers = ["📅 날짜", "요일", "📝 제목", "🏷️ 카테고리", "✅ 완료", "📂 노트"];

// 각 노트에서 인라인 테이블 파싱
const rows = [];

for (let page of pages) {
  const file = app.vault.getAbstractFileByPath(page.file.path);
  const content = await app.vault.read(file);

  // 주요 일정 테이블 파싱 (| 날짜 | 요일 | 제목 | 카테고리 | 완료 | 형식)
  const lines = content.split("\n");
  let inTable = false;

  for (let line of lines) {
    // 헤더 구분선 skip
    if (line.startsWith("| 날짜") || line.startsWith("|---")) {
      inTable = true;
      continue;
    }
    // 테이블 끝
    if (inTable && !line.startsWith("|")) {
      inTable = false;
      continue;
    }
    // 데이터 행 파싱
    if (inTable && line.startsWith("|")) {
      const cols = line.split("|").map(c => c.trim()).filter(c => c !== "");
      if (cols.length >= 5) {
        rows.push([
          cols[0],        // 날짜
          cols[1],        // 요일
          cols[2],        // 제목
          cols[3],        // 카테고리
          cols[4],        // 완료
          page.file.link  // 노트 링크
        ]);
      }
    }
  }
}

// 날짜 오름차순 정렬
rows.sort((a, b) => a[0].localeCompare(b[0]));

dv.table(headers, rows);
````

---

## 🗓️ 특정 월 일정 보기

원하는 연월을 입력해서 해당 월만 필터링합니다.  
아래 `TARGET_MONTH` 값을 바꾸면 해당 월만 표시됩니다.

````dataviewjs
const TARGET_MONTH = "2025-01"; // ← 여기를 원하는 연월로 변경 (예: "2025-03")

const pages = dv.pages('"Calendar"')
  .where(p => p.type === "monthly-calendar" && p["month-label"] === TARGET_MONTH);

if (pages.length === 0) {
  dv.paragraph(`⚠️ **${TARGET_MONTH}** 에 해당하는 달력 노트가 없습니다.`);
} else {
  const headers = ["📅 날짜", "요일", "📝 제목", "🏷️ 카테고리", "✅ 완료"];
  const rows = [];

  for (let page of pages) {
    const file = app.vault.getAbstractFileByPath(page.file.path);
    const content = await app.vault.read(file);
    const lines = content.split("\n");
    let inTable = false;

    for (let line of lines) {
      if (line.startsWith("| 날짜") || line.startsWith("|---")) {
        inTable = true;
        continue;
      }
      if (inTable && !line.startsWith("|")) { inTable = false; continue; }
      if (inTable && line.startsWith("|")) {
        const cols = line.split("|").map(c => c.trim()).filter(c => c !== "");
        if (cols.length >= 5) {
          rows.push([cols[0], cols[1], cols[2], cols[3], cols[4]]);
        }
      }
    }
  }

  rows.sort((a, b) => a[0].localeCompare(b[0]));
  dv.table(headers, rows);
}
````

---

## 🏷️ 카테고리별 필터

카테고리(업무 / 개인 / 공휴일 등)별로 일정을 모아봅니다.

````dataviewjs
const TARGET_CATEGORY = "업무"; // ← "개인", "공휴일" 등으로 변경

const pages = dv.pages('"Calendar"')
  .where(p => p.type === "monthly-calendar")
  .sort(p => p["month-label"], "asc");

const headers = ["📅 날짜", "요일", "📝 제목", "✅ 완료", "📂 노트"];
const rows = [];

for (let page of pages) {
  const file = app.vault.getAbstractFileByPath(page.file.path);
  const content = await app.vault.read(file);
  const lines = content.split("\n");
  let inTable = false;

  for (let line of lines) {
    if (line.startsWith("| 날짜") || line.startsWith("|---")) {
      inTable = true;
      continue;
    }
    if (inTable && !line.startsWith("|")) { inTable = false; continue; }
    if (inTable && line.startsWith("|")) {
      const cols = line.split("|").map(c => c.trim()).filter(c => c !== "");
      if (cols.length >= 5 && cols[3] === TARGET_CATEGORY) {
        rows.push([cols[0], cols[1], cols[2], cols[4], page.file.link]);
      }
    }
  }
}

rows.sort((a, b) => a[0].localeCompare(b[0]));

if (rows.length === 0) {
  dv.paragraph(`⚠️ **${TARGET_CATEGORY}** 카테고리 일정이 없습니다.`);
} else {
  dv.table(headers, rows);
}
````
