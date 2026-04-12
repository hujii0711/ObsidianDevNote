# 📆 월간 달력 그리드

```dataviewjs
// ─── 설정 ───────────────────────────────
const YEAR  = 2025;
const MONTH = 1;   // 1~12
// ────────────────────────────────────────

const dayNames = ["일", "월", "화", "수", "목", "금", "토"];

// 1일의 요일, 말일 계산
const firstWeekday = new Date(YEAR, MONTH - 1, 1).getDay(); // 0=일
const lastDate     = new Date(YEAR, MONTH, 0).getDate();    // 말일

// 달력 셀 배열 생성 (빈칸 + 날짜)
// null = 빈칸, 숫자 = 날짜
const cells = [];
for (let i = 0; i < firstWeekday; i++) cells.push(null);
for (let d = 1; d <= lastDate; d++)    cells.push(d);
// 7의 배수로 맞추기
while (cells.length % 7 !== 0) cells.push(null);

// 주(row) 단위로 분할
const weeks = [];
for (let i = 0; i < cells.length; i += 7) {
  weeks.push(cells.slice(i, i + 7));
}

// 마크다운 표 문자열 생성
const header    = "| " + dayNames.join(" | ") + " |";
const separator = "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|";

const bodyRows = weeks.map(week => {
  const cols = week.map(d => d === null ? "   " : String(d));
  return "| " + cols.join(" | ") + " |";
});

const table = [header, separator, ...bodyRows].join("\n");

dv.paragraph(`### ${YEAR}년 ${MONTH}월`);
dv.paragraph(table);
```
