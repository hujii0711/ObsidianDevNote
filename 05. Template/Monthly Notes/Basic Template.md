```dataviewjs
// ─── 설정 ─────────────────────────────────────────
const YEAR           = 2026;
const MONTH          = 4;
const DAILY_FOLDER   = "03. Daily Notes/2026년/4월";  // 실제 폴더 경로
const TARGET_SECTION = "## 3.  저녁 공부";
// ──────────────────────────────────────────────────

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

const style = document.createElement("style");
style.textContent = `
  .ml-wrap { font-family: var(--font-interface); width: 100%; }
  .ml-title {
    font-size: 1.1em; font-weight: bold;
    margin-bottom: 10px; text-align: center;
    color: var(--text-normal);
  }
  .ml-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  .ml-table th {
    background: var(--background-secondary);
    color: var(--text-muted);
    font-size: 0.85em; font-weight: 600;
    padding: 8px 10px;
    border: 1px solid var(--background-modifier-border);
    text-align: left; box-sizing: border-box;
  }
  .ml-table th:nth-child(1) { width: 100px; }
  .ml-table th:nth-child(2) { width: 130px; }
  .ml-table th:nth-child(3) { width: auto; }
  .ml-table th:nth-child(4) { width: 120px; }
  .ml-table td {
    padding: 8px 10px;
    border: 1px solid var(--background-modifier-border);
    vertical-align: top; font-size: 0.85em;
    color: var(--text-normal); box-sizing: border-box;
  }
  .ml-table tr:hover { background: var(--background-modifier-hover); }
  .date-cell { font-weight: 600; white-space: nowrap; }
  .date-cell.sun   { color: #e05252; }
  .date-cell.sat   { color: #5588e8; }
  .date-cell.today { color: var(--interactive-accent); font-weight: 700; }
  .date-sub {
    font-size: 0.80em; color: var(--text-faint);
    font-weight: normal; display: block; margin-top: 2px;
  }
  .file-link {
    font-size: 0.82em; color: var(--text-accent);
    cursor: pointer; text-decoration: none; display: block;
  }
  .file-link:hover { text-decoration: underline; }
  .file-none { font-size: 0.80em; color: var(--text-faint); }
  .task-list { margin: 0; padding: 0; list-style: none; }
  .task-list li {
    display: flex; align-items: flex-start;
    gap: 5px; margin: 3px 0; line-height: 1.4;
  }
  .task-list li input { flex-shrink: 0; margin-top: 2px; cursor: pointer; }
  .task-list li span  { font-size: 0.88em; color: var(--text-normal); word-break: break-all; }
  .task-list li.done span { text-decoration: line-through; color: var(--text-faint); }
  .no-task { color: var(--text-faint); font-size: 0.80em; }
  .debug-box {
    font-size: 0.78em; color: var(--text-faint);
    margin-bottom: 8px; padding: 4px 8px;
    border: 1px dashed var(--background-modifier-border);
    border-radius: 4px;
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
["월일", "파일명", "주요 할일", "비고"].forEach(h => {
  const th = document.createElement("th");
  th.textContent = h;
  hrow.appendChild(th);
});
thead.appendChild(hrow);
table.appendChild(thead);

const tbody = document.createElement("tbody");

for (let d = 1; d <= lastDate; d++) {
  const dayName = days[today.getDay()];
  const dateKey = `${YEAR}-${monthStr}-${String(d).padStart(2,"0")}(${dayName})`;
  const dateObj = new Date(YEAR, MONTH - 1, d);
  const dow     = dateObj.getDay();
  console.log("dateKey====", dateKey)
  const tasks   = taskMap[dateKey] || [];
  console.log("tasks====", tasks)
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
    s.textContent = "—";
    tdFile.appendChild(s);
  }

  // 할일 셀
  const tdTask = document.createElement("td");
  if (tasks.length === 0) {
    const s    = document.createElement("span");
    s.className   = "no-task";
    s.textContent = "—";
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

  // 비고 셀
  const tdMemo = document.createElement("td");

  tr.appendChild(tdDate);
  tr.appendChild(tdFile);
  tr.appendChild(tdTask);
  tr.appendChild(tdMemo);
  tbody.appendChild(tr);
}

table.appendChild(tbody);
wrap.appendChild(table);
container.appendChild(wrap);
```