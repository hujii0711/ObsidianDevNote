---
year: 2026
month: 5
---

```dataviewjs
// ─── 월간 달력 그리드 ─────────────────────────────
const fm = dv.current().file.frontmatter || {};
const fname = dv.current().file.name;
const fnMatch = fname.match(/(\d{4})[-년\s]*\s*(\d{1,2})/);
const now = new Date();
const YEAR  = Number(fm.year)  || (fnMatch && Number(fnMatch[1])) || now.getFullYear();
const MONTH = Number(fm.month) || (fnMatch && Number(fnMatch[2])) || (now.getMonth() + 1);

const dayNames = ["일","월","화","수","목","금","토"];
const firstWeekday = new Date(YEAR, MONTH - 1, 1).getDay();
const lastDate = new Date(YEAR, MONTH, 0).getDate();
const today = new Date();

const cells = [];
for (let i = 0; i < firstWeekday; i++) cells.push(null);
for (let d = 1; d <= lastDate; d++) cells.push(d);
while (cells.length % 7 !== 0) cells.push(null);

const isDark = document.body.classList.contains("theme-dark");
const gh = isDark ? {
  border: "#30363d", bg: "#0d1117", headerBg: "#161b22",
  text: "#e6edf3", muted: "#7d8590", rowBorder: "#21262d",
  hover: "#161b22", todayBg: "#1f6feb", todayText: "#ffffff",
  sun: "#f85149", sat: "#58a6ff"
} : {
  border: "#d0d7de", bg: "#ffffff", headerBg: "#f6f8fa",
  text: "#1f2328", muted: "#656d76", rowBorder: "#eaeef2",
  hover: "#f6f8fa", todayBg: "#0969da", todayText: "#ffffff",
  sun: "#cf222e", sat: "#0969da"
};

const container = dv.el("div", "");
const style = document.createElement("style");
style.textContent = `
  .cal-wrap {
    font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
    border: 1px solid ${gh.border}; border-radius: 6px; overflow: hidden;
    background: ${gh.bg}; margin-bottom: 16px;
  }
  .cal-title {
    font-size: 14px; font-weight: 600;
    padding: 12px 16px; background: ${gh.headerBg};
    border-bottom: 1px solid ${gh.border}; color: ${gh.text};
  }
  .cal-table { width: 100%; border-collapse: collapse; table-layout: fixed; background: ${gh.bg}; }
  .cal-table th {
    background: ${gh.headerBg}; color: ${gh.muted};
    font-size: 12px; font-weight: 600;
    padding: 8px 0; text-align: center;
    border-bottom: 1px solid ${gh.border};
  }
  .cal-table th.sun { color: ${gh.sun}; }
  .cal-table th.sat { color: ${gh.sat}; }
  .cal-table td {
    height: 56px; text-align: center; vertical-align: middle;
    font-size: 14px; color: ${gh.text};
    border-right: 1px solid ${gh.rowBorder};
    border-bottom: 1px solid ${gh.rowBorder};
  }
  .cal-table td:last-child { border-right: none; }
  .cal-table tr:last-child td { border-bottom: none; }
  .cal-table td.sun { color: ${gh.sun}; }
  .cal-table td.sat { color: ${gh.sat}; }
  .cal-table td.empty { background: ${gh.headerBg}; }
  .cal-day-today {
    display: inline-block; min-width: 28px; padding: 4px 8px;
    background: ${gh.todayBg}; color: ${gh.todayText};
    border-radius: 999px; font-weight: 700;
  }
  @media (max-width: 600px) {
    .cal-table td { height: 44px; font-size: 13px; }
  }
`;
container.appendChild(style);

const wrap = document.createElement("div");
wrap.className = "cal-wrap";

const title = document.createElement("div");
title.className = "cal-title";
title.textContent = `📆 ${YEAR}년 ${MONTH}월`;
wrap.appendChild(title);

const table = document.createElement("table");
table.className = "cal-table";
const thead = document.createElement("thead");
const hrow = document.createElement("tr");
dayNames.forEach((n, i) => {
  const th = document.createElement("th");
  th.textContent = n;
  if (i === 0) th.className = "sun";
  if (i === 6) th.className = "sat";
  hrow.appendChild(th);
});
thead.appendChild(hrow);
table.appendChild(thead);

const tbody = document.createElement("tbody");
for (let i = 0; i < cells.length; i += 7) {
  const tr = document.createElement("tr");
  cells.slice(i, i + 7).forEach((d, idx) => {
    const td = document.createElement("td");
    if (d === null) {
      td.className = "empty";
    } else {
      if (idx === 0) td.className = "sun";
      else if (idx === 6) td.className = "sat";
      const isToday = today.getFullYear() === YEAR && today.getMonth() + 1 === MONTH && today.getDate() === d;
      if (isToday) {
        const span = document.createElement("span");
        span.className = "cal-day-today";
        span.textContent = d;
        td.appendChild(span);
      } else {
        td.textContent = d;
      }
    }
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
}
table.appendChild(tbody);
wrap.appendChild(table);
container.appendChild(wrap);
```

```dataviewjs
// ─── 설정 ─────────────────────────────────────────
const TARGET_SECTION = "## 3.  학습 계획";
const DAILY_FOLDER_TPL = "03. Daily Notes/{YEAR}년/{MONTH}월"; // {YEAR}, {MONTH} 치환
// ──────────────────────────────────────────────────

// ── 년/월 결정: frontmatter → 파일명 → 오늘 ──
const fm = dv.current().file.frontmatter || {};
const fname = dv.current().file.name;
const fnMatch = fname.match(/(\d{4})[-년\s]*\s*(\d{1,2})/);
const now = new Date();

const YEAR  = Number(fm.year)  || (fnMatch && Number(fnMatch[1])) || now.getFullYear();
const MONTH = Number(fm.month) || (fnMatch && Number(fnMatch[2])) || (now.getMonth() + 1);
const DAILY_FOLDER = DAILY_FOLDER_TPL.replace("{YEAR}", YEAR).replace("{MONTH}", MONTH);

const dayNames = ["일","월","화","수","목","금","토"];
const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const lastDate = new Date(YEAR, MONTH, 0).getDate();
const today    = new Date();
const monthStr = String(MONTH).padStart(2, "0");

// ── 1. 폴더 경로 직접 탐색 (dv.pages 우회) ──
// dv.pages()의 따옴표 이스케이프 문제를 피하기 위해
// app.vault.getFiles() 로 직접 파일 목록을 가져옴
const taskMap = {};

const allFiles = app.vault.getFiles();
const targetFiles = allFiles.filter(f => {
  // 폴더 경로가 일치하고 파일명이 YYYY-MM-DD 형식인 것만
  return f.path.startsWith(DAILY_FOLDER) &&
         f.name.startsWith(`${YEAR}-${monthStr}`) &&
         f.extension === "md";
});

// 디버그: 찾은 파일 수 확인 (문제 시 주석 해제)
// dv.paragraph(`찾은 파일 수: ${targetFiles.length}`);

for (const file of targetFiles) {
  const content = await app.vault.read(file);
  const lines   = content.split("\n");
  let inSection = false;
  const tasks   = [];

  for (const line of lines) {
    const t = line.trim();
    if (t === TARGET_SECTION.trim()) { inSection = true; continue; }
    if (inSection && t.startsWith("## ")) break;
    if (inSection && /^- \[[ xX]\]/.test(t)) {
      const checked = /^- \[[xX]\]/.test(t);
      const text    = t.replace(/^- \[[ xX]\]\s*/, "").trim();
      if (text) tasks.push({ text, checked });
    }
  }
  // 파일명에서 확장자 제거 (예: "2026-04-03(Sun)")
  taskMap[file.basename] = tasks;
}

// ── 2. 컨테이너 + 스타일 ──
const container = dv.el("div", "");

const isDark = document.body.classList.contains("theme-dark");
const gh = isDark ? {
  border: "#30363d", bg: "#0d1117", headerBg: "#161b22",
  text: "#e6edf3", muted: "#7d8590", rowBorder: "#21262d",
  hover: "#161b22", accent: "#2f81f7",
  sun: "#f85149", sat: "#58a6ff", today: "#3fb950"
} : {
  border: "#d0d7de", bg: "#ffffff", headerBg: "#f6f8fa",
  text: "#1f2328", muted: "#656d76", rowBorder: "#eaeef2",
  hover: "#f6f8fa", accent: "#0969da",
  sun: "#cf222e", sat: "#0969da", today: "#1a7f37"
};

const style = document.createElement("style");
style.textContent = `
  .ml-wrap {
    font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
    width: 100%;
    border: 1px solid ${gh.border};
    border-radius: 6px;
    overflow: hidden;
    background: ${gh.bg};
  }
  .ml-title {
    font-size: 14px; font-weight: 600;
    padding: 12px 16px;
    background: ${gh.headerBg};
    border-bottom: 1px solid ${gh.border};
    color: ${gh.text};
  }
  .debug-box {
    font-size: 12px; color: ${gh.muted};
    padding: 8px 16px;
    background: ${gh.bg};
    border-bottom: 1px solid ${gh.rowBorder};
  }
  .ml-table-wrap { width: 100%; overflow-x: auto; }
  .ml-table { width: 100%; border-collapse: collapse; background: ${gh.bg}; }
  .ml-table th {
    background: ${gh.headerBg};
    color: ${gh.muted};
    font-size: 12px; font-weight: 600;
    padding: 8px 12px;
    border-bottom: 1px solid ${gh.border};
    text-align: left;
  }
  .ml-table th:nth-child(1) { width: 110px; }
  .ml-table th:nth-child(2) { width: 170px; text-align: center; }
  .ml-table th:nth-child(3) { width: auto; }
  .ml-table td:nth-child(1) { vertical-align: middle; }
  .ml-table td:nth-child(2) { text-align: center; vertical-align: middle; }
  .ml-table td {
    padding: 12px 14px;
    border-bottom: 1px solid ${gh.rowBorder};
    vertical-align: top; font-size: 15px;
    color: ${gh.text};
  }
  .ml-table tbody tr:last-child td { border-bottom: none; }
  .ml-table tbody tr:hover { background: ${gh.hover}; }
  .date-cell { font-weight: 600; white-space: nowrap; color: ${gh.text}; font-size: 15px; }
  .date-cell.sun   { color: ${gh.sun}; }
  .date-cell.sat   { color: ${gh.sat}; }
  .date-cell.today { color: ${gh.today}; font-weight: 700; }
  .date-sub {
    font-size: 12px; color: ${gh.muted};
    font-weight: normal; display: block; margin-top: 3px;
  }
  .file-link {
    font-size: 14px; color: ${gh.accent};
    cursor: pointer; text-decoration: none; display: block;
    word-break: break-all;
  }
  .file-link:hover { text-decoration: underline; }
  .file-none { font-size: 14px; color: ${gh.muted}; }
  .task-list { margin: 0 0 0 -20px; padding: 0; list-style: none; }
  .task-list li {
    display: flex; align-items: center;
    gap: 4px; margin: 1px 0; line-height: 1.3;
  }
  .task-list li input { flex-shrink: 0; margin: 0; padding: 0; cursor: pointer; accent-color: ${gh.accent}; }
  .task-list li span  { font-size: 14px; color: ${gh.text}; word-break: break-word; padding: 0; }
  .task-list li.done span { text-decoration: line-through; color: ${gh.muted}; }
  .no-task { color: ${gh.muted}; font-size: 14px; }

  /* 모바일 반응형: 카드 레이아웃 */
  @media (max-width: 600px) {
    .ml-table thead { display: none; }
    .ml-table, .ml-table tbody, .ml-table tr, .ml-table td { display: block; width: 100%; box-sizing: border-box; }
    .ml-table tr {
      border-bottom: 1px solid ${gh.border};
      padding: 10px 12px;
    }
    .ml-table tbody tr:last-child { border-bottom: none; }
    .ml-table td {
      border: none;
      padding: 4px 0;
    }
    .ml-table td:first-child { padding-top: 0; }
    .date-cell { font-size: 14px; }
  }
`;
container.appendChild(style);

// ── 3. DOM 생성 ──
const wrap = document.createElement("div");
wrap.className = "ml-wrap";

// 제목
const title = document.createElement("div");
title.className = "ml-title";
title.textContent = `${YEAR}년 ${MONTH}월 일정`;
wrap.appendChild(title);

// 디버그 박스: 읽은 파일 경로 확인
const debug = document.createElement("div");
debug.className = "debug-box";
debug.textContent = targetFiles.length === 0
  ? `⚠️ 파일을 찾지 못했습니다. 폴더 경로를 확인하세요: "${DAILY_FOLDER}"`
  : `✅ ${targetFiles.length}개 파일 로드됨 (${DAILY_FOLDER})`;
wrap.appendChild(debug);

// 테이블
const table = document.createElement("table");
table.className = "ml-table";

const thead = document.createElement("thead");
const hrow  = document.createElement("tr");
["월일", "파일명", "주요 할일"].forEach(h => {
  const th = document.createElement("th");
  th.textContent = h;
  hrow.appendChild(th);
});
thead.appendChild(hrow);
table.appendChild(thead);

const tbody = document.createElement("tbody");

for (let d = 1; d <= lastDate; d++) {
  const dateObj = new Date(YEAR, MONTH - 1, d);
  const dow     = dateObj.getDay();
  const dayName = days[dow];
  const dateKey = `${YEAR}-${monthStr}-${String(d).padStart(2,"0")}(${dayName})`;
  const tasks   = taskMap[dateKey] || [];
  const hasFile = Object.prototype.hasOwnProperty.call(taskMap, dateKey);

  const isToday = (
    today.getFullYear() === YEAR &&
    today.getMonth() + 1 === MONTH &&
    today.getDate() === d
  );

  const tr = document.createElement("tr");

  // 날짜 셀
  const tdDate  = document.createElement("td");
  const dateDiv = document.createElement("div");
  dateDiv.className = "date-cell" +
    (isToday ? " today" : dow === 0 ? " sun" : dow === 6 ? " sat" : "");
  dateDiv.textContent = `${MONTH}월 ${d}일`;
  const sub = document.createElement("span");
  sub.className   = "date-sub";
  sub.textContent = `${dayNames[dow]}요일${isToday ? "  ◀ 오늘" : ""}`;
  dateDiv.appendChild(sub);
  tdDate.appendChild(dateDiv);

  // 파일명 셀
  const tdFile = document.createElement("td");
  if (hasFile) {
    const a    = document.createElement("a");
    a.className   = "file-link";
    a.textContent = dateKey;
    a.onclick     = () => app.workspace.openLinkText(dateKey, "", false);
    tdFile.appendChild(a);
  } else {
    const s    = document.createElement("span");
    s.className   = "file-none";
    s.textContent = "";
    tdFile.appendChild(s);
  }

  // 할일 셀
  const tdTask = document.createElement("td");
  if (tasks.length === 0) {
    const s    = document.createElement("span");
    s.className   = "no-task";
    s.textContent = "";
    tdTask.appendChild(s);
  } else {
    const ul = document.createElement("ul");
    ul.className = "task-list";
    tasks.forEach(t => {
      const li = document.createElement("li");
      if (t.checked) li.className = "done";
      const cb  = document.createElement("input");
      cb.type    = "checkbox";
      cb.checked = t.checked;
      const sp  = document.createElement("span");
      sp.textContent = t.text;
      li.appendChild(cb);
      li.appendChild(sp);
      ul.appendChild(li);
    });
    tdTask.appendChild(ul);
  }

  tr.appendChild(tdDate);
  tr.appendChild(tdFile);
  tr.appendChild(tdTask);
  tbody.appendChild(tr);
}

table.appendChild(tbody);
const tableWrap = document.createElement("div");
tableWrap.className = "ml-table-wrap";
tableWrap.appendChild(table);
wrap.appendChild(tableWrap);
container.appendChild(wrap);
```