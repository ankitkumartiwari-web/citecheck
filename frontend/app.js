// CiteCheck frontend — chat (per-paper sessions) / compare / review / translate
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const icons = () => { try { lucide.createIcons(); } catch (_) {} };

const VERDICT = {
  supported: ["b-green", "Supported"],
  partially_supported: ["b-amber", "Partial"],
  unsupported: ["b-red", "Unsupported"],
};
const ROLE = { BACKGROUND: "b-blue", METHOD: "b-violet", RESULT: "b-green", CONCLUSION: "b-amber" };
const STANCE = { supports: ["b-green", "Supports"], refutes: ["b-red", "Refutes"], neutral: ["b-ink", "Neutral"] };
const LANGS = ["Spanish", "French", "German", "Hindi", "Chinese (Simplified)", "Arabic", "Japanese", "Portuguese"];
const SUGGESTIONS = [
  "What problem does this paper solve?",
  "Summarize the key contributions.",
  "What method is proposed and how does it work?",
  "What are the main results and limitations?",
];
// Per-paper chat sessions (persisted server-side in the vector DB). Key "*" = all.
let chatsByPaper = {};
let chatCounts = {};
let activePaper = "*";
let currentPapers = [];
let thread = null;
let busy = false;
let selectedPaper = null;   // Peer Review dropdown
let trSelected = null;      // Translate dropdown
let trLang = null;          // Translate language
let compareExcluded = new Set();  // papers de-selected from Compare (default: all in)
let populateReview = null;
let populateTranslate = null;

// ---------- helpers ----------
function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
function badge(cls, label, plain) { return el("span", `badge ${cls}${plain ? " plain" : ""}`, label); }
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) { let m = `Request failed (${res.status})`; try { m = (await res.json()).detail || m; } catch (_) {} throw new Error(m); }
  return res.json();
}
function toast(msg, err = false) {
  const t = $("toast"); t.textContent = msg; t.classList.toggle("error", err); t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}
function md(t) { return DOMPurify.sanitize(marked.parse(t || "")); }
function download(name, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }

// Capture a DOM node (a result card or the whole thread) into a multi-page A4 PDF
// with a CiteCheck header and a page-numbered footer on every page. The content is
// sliced to fit between header and footer (no overlap), and renders any language.
async function exportNodePDF(node, filename, title) {
  if (!(window.jspdf && window.jspdf.jsPDF && window.html2canvas)) return toast("PDF library unavailable.", true);
  toast("Building PDF…");
  try {
    const src = await html2canvas(node, {
      scale: 2, backgroundColor: "#ffffff", useCORS: true,
      ignoreElements: (e) => e.classList && e.classList.contains("actions"),
    });
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ unit: "pt", format: "a4" });
    const pageW = pdf.internal.pageSize.getWidth(), pageH = pdf.internal.pageSize.getHeight();
    const M = 40, HEADER = 36, FOOTER = 26;
    const cTop = M + HEADER, cBottom = pageH - M - FOOTER, cW = pageW - M * 2, cH = cBottom - cTop;
    const ratio = cW / src.width;                    // px -> pt
    const sliceHpx = Math.floor(cH / ratio);         // source px per page
    const pages = Math.max(1, Math.ceil(src.height / sliceHpx));
    const dateStr = new Date().toLocaleDateString();

    for (let i = 0; i < pages; i++) {
      if (i) pdf.addPage();
      const sy = i * sliceHpx, hpx = Math.min(sliceHpx, src.height - sy);
      const tmp = document.createElement("canvas");
      tmp.width = src.width; tmp.height = hpx;
      tmp.getContext("2d").drawImage(src, 0, sy, src.width, hpx, 0, 0, src.width, hpx);
      pdf.addImage(tmp.toDataURL("image/png"), "PNG", M, cTop, cW, hpx * ratio);

      // Header
      pdf.setFont("helvetica", "bold").setFontSize(11).setTextColor(20, 21, 26);
      pdf.text("CiteCheck", M, M + 14);
      if (title) {
        pdf.setFont("helvetica", "normal").setFontSize(9).setTextColor(120, 120, 128);
        pdf.text(pdf.splitTextToSize(String(title), cW - 90)[0], pageW - M, M + 14, { align: "right" });
      }
      pdf.setDrawColor(227, 227, 222).line(M, M + HEADER - 10, pageW - M, M + HEADER - 10);

      // Footer
      pdf.setDrawColor(227, 227, 222).line(M, pageH - M - FOOTER + 12, pageW - M, pageH - M - FOOTER + 12);
      pdf.setFont("helvetica", "normal").setFontSize(8).setTextColor(150, 150, 150);
      pdf.text(`Generated ${dateStr}`, M, pageH - M);
      pdf.text(`Page ${i + 1} of ${pages}`, pageW - M, pageH - M, { align: "right" });
    }
    pdf.save(filename);
  } catch (e) { toast("PDF failed: " + e.message, true); }
}

function exportPDF(filename, title, body) {
  if (!(window.jspdf && window.jspdf.jsPDF)) {
    download(filename.replace(/\.pdf$/, ".txt"), title + "\n\n" + body);
    return toast("PDF library unavailable — exported TXT instead.", true);
  }
  const doc = new window.jspdf.jsPDF({ unit: "pt", format: "a4" });
  const M = 48, W = doc.internal.pageSize.getWidth() - M * 2, BOTTOM = doc.internal.pageSize.getHeight() - M;
  let y = M;
  doc.setFont("helvetica", "bold").setFontSize(14);
  for (const ln of doc.splitTextToSize(title, W)) { if (y > BOTTOM) { doc.addPage(); y = M; } doc.text(ln, M, y); y += 18; }
  y += 6;
  doc.setFont("helvetica", "normal").setFontSize(10.5);
  for (const para of body.split("\n")) {
    const lines = doc.splitTextToSize(para === "" ? " " : para, W);
    for (const ln of lines) { if (y > BOTTOM) { doc.addPage(); y = M; } doc.text(ln, M, y); y += 14; }
  }
  doc.save(filename);
}
function currentList() { return (chatsByPaper[activePaper] ||= []); }
function saveActive() { try { localStorage.setItem("citecheck_active", activePaper); } catch (_) {} }
function loadActive() { try { return localStorage.getItem("citecheck_active") || "*"; } catch (_) { return "*"; } }
async function switchPaper(key) {
  activePaper = key; setMode("chat");
  renderPapers(currentPapers);   // self-corrects activePaper to "*" if the paper is gone
  saveActive();
  try { const d = await api(`/api/chats?paper=${encodeURIComponent(activePaper)}`); chatsByPaper[activePaper] = d.items || []; } catch (_) {}
  renderThread();
}

// ---------- mode switching ----------
function setMode(mode) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  ["chat", "compare", "review", "translate"].forEach((m) => $(`view-${m}`).classList.toggle("hidden", m !== mode));
  $("modeLabel").textContent = { chat: "Chat", compare: "Compare", review: "Peer Review", translate: "Translate" }[mode];
  $("verifyWrap").style.display = mode === "chat" ? "" : "none";
  closeSidebar();
}

// ---------- export ----------
function exportText(item) {
  const lines = [`QUESTION: ${item.query}`, "", "ANSWER:", item.answer, ""];
  if (item.report) {
    const r = item.report;
    lines.push(`FACT-CHECK (${r.overall_grounded ? "GROUNDED" : "REVIEW NEEDED"}): ${r.summary}`, "");
    for (const c of r.claims) lines.push(`  [${c.verdict}] ${c.claim} (sources: ${c.supporting_ids.map((i) => `[${i}]`).join(", ") || "-"})`);
    lines.push("");
  }
  lines.push("SOURCES:");
  for (const c of item.chunks) lines.push(`  [${c.id}] ${c.page ? `${c.source}, p.${c.page}` : c.source}${c.role ? " — " + c.role : ""}`);
  return lines.join("\n");
}

// ---------- evidence graph ----------
function graphTip() {
  let t = $("graphTip");
  if (!t) { t = el("div", "graph-tip"); t.id = "graphTip"; document.body.appendChild(t); }
  return t;
}
function buildGraph(item) {
  const claims = (item.report && item.report.claims) || [];
  const sources = item.chunks || [];
  if (!claims.length || !sources.length) return null;
  const NH = 30, GAP = 16, PAD = 18, W = 560, leftX = PAD, leftW = 230, rightX = W - PAD - 150, rightW = 150;
  const rows = Math.max(claims.length, sources.length);
  const H = PAD * 2 + rows * NH + (rows - 1) * GAP;
  const cy = (i) => PAD + i * (NH + GAP) + NH / 2;
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%"); svg.setAttribute("height", H);
  const idx = {}; sources.forEach((s, i) => { idx[s.id] = i; });
  const vC = { supported: "#1f9d57", partially_supported: "#c47f12", unsupported: "#d23f4a" };
  claims.forEach((c, i) => (c.supporting_ids || []).forEach((sid) => {
    if (idx[sid] == null) return;
    const p = document.createElementNS(SVGNS, "path");
    const x1 = leftX + leftW, y1 = cy(i), x2 = rightX, y2 = cy(idx[sid]), mx = (x1 + x2) / 2;
    p.setAttribute("d", `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
    p.setAttribute("fill", "none"); p.setAttribute("stroke", vC[c.verdict] || "#9a9ea6");
    p.setAttribute("stroke-width", "1.6"); p.setAttribute("opacity", "0.75"); svg.appendChild(p);
  }));
  const tip = graphTip();
  const node = (x, y, w, label, stroke, tipText) => {
    const g = document.createElementNS(SVGNS, "g"); g.setAttribute("class", "gnode");
    const r = document.createElementNS(SVGNS, "rect");
    r.setAttribute("x", x); r.setAttribute("y", y - NH / 2); r.setAttribute("width", w); r.setAttribute("height", NH); r.setAttribute("rx", 7);
    if (stroke) { r.setAttribute("stroke", stroke); r.setAttribute("stroke-width", "1.6"); }
    const t = document.createElementNS(SVGNS, "text"); t.setAttribute("x", x + 9); t.setAttribute("y", y + 3.5); t.textContent = label;
    if (tipText) {
      g.style.cursor = "pointer";
      g.addEventListener("mouseenter", () => { tip.textContent = tipText; tip.classList.add("show"); });
      g.addEventListener("mousemove", (e) => {
        const tw = tip.offsetWidth || 320, th = tip.offsetHeight || 80;
        let x = e.clientX + 14, y = e.clientY + 14;
        if (x + tw > window.innerWidth - 8) x = e.clientX - tw - 14;
        if (y + th > window.innerHeight - 8) y = e.clientY - th - 14;
        tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
      });
      g.addEventListener("mouseleave", () => tip.classList.remove("show"));
    }
    g.append(r, t); svg.appendChild(g);
  };
  claims.forEach((c, i) => {
    const tipText = `Claim C${i + 1} · ${c.verdict.replace(/_/g, " ")}\n\n${c.claim}` + (c.rationale ? `\n\n${c.rationale}` : "");
    node(leftX, cy(i), leftW, `C${i + 1} · ${c.verdict.replace("partially_supported", "partial")}`, vC[c.verdict], tipText);
  });
  sources.forEach((s, i) => {
    const loc = s.page ? `${s.source}, p.${s.page}` : s.source;
    const snip = (s.text || "").slice(0, 220);
    node(rightX, cy(i), rightW, `[${s.id}] ${s.source.slice(0, 16)}`, null, `Source [${s.id}] · ${loc}\n\n${snip}${(s.text || "").length > 220 ? "…" : ""}`);
  });
  const wrap = el("div", "graph-wrap"); wrap.appendChild(svg); return wrap;
}

// ---------- chat rendering ----------
function ensureThread() { if (!thread) { $("empty").style.display = "none"; thread = el("div", "thread"); $("chat").appendChild(thread); } return thread; }
function scrollChat() { const c = $("chat"); c.scrollTop = c.scrollHeight; }

function addUser(text) {
  const m = el("div", "msg user"); m.appendChild(el("div", "label", "You"));
  const b = el("div", "bubble"); b.textContent = text; m.appendChild(b);
  ensureThread().appendChild(m); scrollChat();
}
function addTyping() {
  const m = el("div", "msg"); const card = el("div", "card"); card.appendChild(el("div", "label", "CiteCheck"));
  const t = el("div", "typing"); t.append(el("i"), el("i"), el("i")); card.appendChild(t);
  m.appendChild(card); ensureThread().appendChild(m); scrollChat(); return m;
}
function renderAnswer(node, item, idx) {
  node.className = "msg"; node.innerHTML = "";
  const card = el("div", "card"); card.appendChild(el("div", "label", "CiteCheck"));
  const ans = el("div", "answer"); ans.innerHTML = md(item.answer); card.appendChild(ans);

  if (item.report) {
    const r = item.report, sub = el("div", "sub"), head = el("div", "sub-head");
    head.appendChild(el("span", "label", "Fact-check"));
    head.appendChild(badge(r.overall_grounded ? "b-green" : "b-amber", r.overall_grounded ? "Grounded" : "Review needed"));
    sub.appendChild(head);
    if (r.summary) sub.appendChild(el("div", "fc-summary", r.summary));
    r.claims.forEach((c) => {
      const [cls, label] = VERDICT[c.verdict] || ["b-ink", c.verdict];
      const cl = el("div", "claim"), top = el("div", "claim-top");
      top.appendChild(badge(cls, label));
      top.appendChild(el("span", "cite", "sources: " + (c.supporting_ids.map((i) => `[${i}]`).join(" ") || "—")));
      cl.appendChild(top); cl.appendChild(el("div", "claim-text", c.claim));
      if (c.rationale) cl.appendChild(el("div", "claim-why", c.rationale));
      sub.appendChild(cl);
    });
    card.appendChild(sub);
  }
  if (item.chunks && item.chunks.length) {
    const sub = el("div", "sub"); sub.appendChild(el("div", "label", "Sources used"));
    item.chunks.forEach((c) => {
      const box = el("div", "src"), h = el("div", "src-head");
      h.appendChild(badge("b-ink", `[${c.id}]`, true));
      h.appendChild(el("span", "cite", c.page ? `${c.source}, p.${c.page}` : c.source));
      if (c.role) h.appendChild(badge(ROLE[c.role] || "b-ink", c.role));
      box.appendChild(h); box.appendChild(el("div", "src-text", c.text)); sub.appendChild(box);
    });
    card.appendChild(sub);
  }
  const actions = el("div", "actions");
  const gbtn = el("button", "linkbtn"); gbtn.innerHTML = '<i data-lucide="workflow"></i> Evidence graph';
  let g = null; gbtn.onclick = () => { if (g) { g.remove(); g = null; return; } g = buildGraph(item); if (!g) return toast("No claims/sources to graph."); card.appendChild(g); };
  const pdf = el("button", "linkbtn"); pdf.innerHTML = '<i data-lucide="file-down"></i> Export PDF';
  pdf.onclick = () => exportNodePDF(card, `citecheck_answer_${idx + 1}.pdf`, "Q: " + item.query, exportText(item));
  actions.append(gbtn, pdf); card.appendChild(actions);
  node.appendChild(card); icons(); scrollChat();
}
function appendExchange(item, idx) {
  addUser(item.query);
  const a = el("div", "msg"); ensureThread().appendChild(a); renderAnswer(a, item, idx);
}
function renderThread() {
  if (thread) { thread.remove(); thread = null; }
  const list = currentList();
  $("empty").style.display = list.length ? "none" : "";
  if (list.length) { ensureThread(); list.forEach((it, i) => appendExchange(it, i)); }
  scrollChat();
}

async function send(text) {
  text = (text ?? $("input").value).trim();
  if (!text || busy) return;
  busy = true; $("send").disabled = true; $("input").value = ""; autoresize();
  addUser(text);
  const typing = addTyping();
  try {
    const data = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, verify: $("verifyToggle").checked, sources: activePaper === "*" ? null : [activePaper] }),
    });
    const item = { query: text, answer: data.answer, report: data.report, chunks: data.chunks };
    currentList().push(item);  // server already persisted it to the vector DB
    renderAnswer(typing, item, currentList().length - 1);
    refreshStats();
  } catch (e) {
    typing.querySelector(".card").appendChild(el("div", "claim-why", "⛔ " + e.message));
    toast(e.message, true);
  } finally { busy = false; $("send").disabled = false; }
}

// ---------- selectable papers = per-paper chat sessions ----------
function renderPapers(papers) {
  const list = $("papersList"); list.innerHTML = "";
  if (activePaper !== "*" && !papers.includes(activePaper)) activePaper = "*";
  const pill = (key, label) => {
    const p = el("div", "pill" + (key === activePaper ? " sel" : "")); p.dataset.key = key; p.title = label;
    const tick = el("span", "tick"); tick.innerHTML = '<i data-lucide="check"></i>';
    p.append(tick, el("span", "pname", label));
    const n = chatCounts[key] || (chatsByPaper[key] || []).length; if (n) p.appendChild(el("span", "count", String(n)));
    p.onclick = () => switchPaper(key);
    return p;
  };
  list.appendChild(pill("*", "All papers"));
  papers.forEach((p) => list.appendChild(pill(p, p)));
  const h = $("scopeHint"); h.innerHTML = "";
  h.appendChild(el("span", "label", "Active · " + (activePaper === "*" ? "All papers" : activePaper)));
  icons();
}

// ---------- generic custom dropdown ----------
function setupDropdown(rootId, trigId, labelId, menuId, onSelect) {
  const root = $(rootId);
  $(trigId).onclick = (e) => { e.stopPropagation(); const open = root.classList.toggle("open"); $(trigId).setAttribute("aria-expanded", open ? "true" : "false"); };
  document.addEventListener("click", (e) => { if (!root.contains(e.target)) root.classList.remove("open"); });
  return function populate(items, getSel, setSel) {
    const menu = $(menuId); menu.innerHTML = "";
    if (!items.length) { $(labelId).textContent = "No papers yet"; menu.appendChild(el("div", "dd-empty", "Upload a PDF to begin")); setSel(null); return; }
    let cur = getSel(); if (!cur || !items.includes(cur)) { cur = items[0]; setSel(cur); }
    $(labelId).textContent = cur;
    items.forEach((p) => {
      const it = el("div", "dd-item" + (p === cur ? " selected" : ""), p);
      it.onclick = () => { setSel(p); $(labelId).textContent = p; menu.querySelectorAll(".dd-item").forEach((n) => n.classList.toggle("selected", n.textContent === p)); root.classList.remove("open"); if (onSelect) onSelect(p); };
      menu.appendChild(it);
    });
  };
}

// ---------- compare ----------
function renderComparePapers(papers) {
  const wrap = $("comparePapers"); if (!wrap) return;
  wrap.innerHTML = "";
  [...compareExcluded].forEach((p) => { if (!papers.includes(p)) compareExcluded.delete(p); });
  wrap.appendChild(el("div", "label", papers.length < 2
    ? `Upload at least 2 papers to compare (you have ${papers.length})`
    : "Papers to compare · click to toggle"));
  const chips = el("div", "lang-chips");
  papers.forEach((p) => {
    const c = el("button", "chip" + (compareExcluded.has(p) ? "" : " active"), p);
    c.onclick = () => { if (compareExcluded.has(p)) compareExcluded.delete(p); else compareExcluded.add(p); c.classList.toggle("active"); };
    chips.appendChild(c);
  });
  const add = el("button", "chip add", "＋ Add paper");
  add.onclick = () => $("fileInput").click();
  chips.appendChild(add);
  wrap.appendChild(chips);
}

async function runCompare() {
  const claim = $("compareInput").value.trim(); if (!claim) return;
  const sel = currentPapers.filter((p) => !compareExcluded.has(p));
  if (currentPapers.length < 2) return toast("Upload at least 2 papers to compare (sidebar → Upload PDFs).", true);
  if (sel.length < 2) return toast("Select at least 2 papers to compare.", true);
  const out = $("compareResult"); out.innerHTML = "";
  const ld = el("div", "card consensus"); const t = el("div", "typing"); t.append(el("i"), el("i"), el("i")); ld.appendChild(t); out.appendChild(ld);
  try {
    const d = await api("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim, papers: sel }) });
    out.innerHTML = "";
    const total = d.stances.length || 1, head = el("div", "consensus");
    head.appendChild(el("div", "label", "Claim")); head.appendChild(el("div", "claim-text", d.claim));
    const bar = el("div", "cbar"); const seg = (cls, n) => { const i = el("i", cls); i.style.width = (n / total * 100) + "%"; return i; };
    bar.append(seg("s", d.counts.supports), seg("r", d.counts.refutes), seg("n", d.counts.neutral)); head.appendChild(bar);
    const lg = el("div", "clegend");
    lg.append(el("span", null, `● ${d.counts.supports} support`), el("span", null, `● ${d.counts.refutes} refute`), el("span", null, `● ${d.counts.neutral} neutral`));
    head.appendChild(lg);
    if (d.consensus_summary) head.appendChild(el("p", "muted", d.consensus_summary));
    out.appendChild(head);
    d.stances.forEach((s) => {
      const [cls, label] = STANCE[s.stance] || ["b-ink", s.stance];
      const c = el("div", "card stance-card"), h = el("div", "src-head");
      h.appendChild(badge(cls, label)); h.appendChild(el("span", "cite", s.paper)); c.appendChild(h);
      if (s.evidence) c.appendChild(el("div", "quote", `“${s.evidence}”`));
      if (s.explanation) c.appendChild(el("div", "claim-why", s.explanation));
      out.appendChild(c);
    });
    const act = el("div", "actions"); const b = el("button", "linkbtn"); b.innerHTML = '<i data-lucide="file-down"></i> Export PDF';
    b.onclick = () => exportNodePDF(out, "citecheck_compare.pdf", "Cross-paper consensus"); act.appendChild(b); out.appendChild(act); icons();
    refreshStats();
  } catch (e) { out.innerHTML = ""; toast(e.message, true); }
}

// ---------- peer review ----------
async function runReview() {
  const paper = selectedPaper; if (!paper) return toast("Upload a paper first.");
  const out = $("reviewResult"); out.innerHTML = "";
  const ld = el("div", "card chair"); ld.appendChild(el("div", "label", "Convening reviewers…")); const t = el("div", "typing"); t.append(el("i"), el("i"), el("i")); ld.appendChild(t); out.appendChild(ld);
  try {
    const d = await api("/api/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paper }) });
    out.innerHTML = "";
    const ch = d.chair || {}, chair = el("div", "card chair"), h = el("div", "rev-head"), left = el("div");
    left.appendChild(el("div", "label", "Chair decision")); left.appendChild(badge("b-violet", ch.recommendation || "—")); h.appendChild(left);
    const sc = el("div", "score"); sc.innerHTML = `${ch.overall_score ?? "—"}<small>/10</small>`; h.appendChild(sc); chair.appendChild(h);
    if (ch.summary) chair.appendChild(el("p", "muted", ch.summary));
    const mk = (title, items) => { const c = el("div"); c.appendChild(el("div", "label", title)); const ul = el("ul", "list-tight"); (items || []).forEach((x) => ul.appendChild(el("li", null, x))); c.appendChild(ul); return c; };
    const cols = el("div", "cols"); cols.append(mk("Top strengths", ch.top_strengths), mk("Top concerns", ch.top_concerns)); chair.appendChild(cols); out.appendChild(chair);
    const grid = el("div", "review-grid");
    (d.reviewers || []).forEach((r) => {
      const c = el("div", "card"), hh = el("div", "rev-head"), l = el("div");
      l.appendChild(el("div", "label", r.lens)); l.appendChild(el("b", null, r.name)); hh.appendChild(l);
      hh.appendChild(badge("b-ink", `${r.score ?? "—"}/10`, true)); c.appendChild(hh);
      if (r.summary) c.appendChild(el("p", "muted", r.summary));
      c.appendChild(mk("Strengths", r.strengths)); c.appendChild(mk("Weaknesses", r.weaknesses)); grid.appendChild(c);
    });
    out.appendChild(grid);
    const act = el("div", "actions"); const b = el("button", "linkbtn"); b.innerHTML = '<i data-lucide="file-down"></i> Export PDF';
    b.onclick = () => exportNodePDF(out, "citecheck_review.pdf", "Peer review · " + paper); act.appendChild(b); out.appendChild(act); icons();
    refreshStats();
  } catch (e) { out.innerHTML = ""; toast(e.message, true); }
}

// ---------- translate ----------
function initLangChips() {
  const wrap = $("langChips");
  LANGS.forEach((l) => {
    const b = el("button", "chip", l);
    b.onclick = () => { trLang = l; $("langCustom").value = ""; document.querySelectorAll("#langChips .chip").forEach((c) => c.classList.toggle("active", c === b)); };
    wrap.appendChild(b);
  });
  $("langCustom").addEventListener("input", () => { if ($("langCustom").value.trim()) { trLang = $("langCustom").value.trim(); document.querySelectorAll("#langChips .chip").forEach((c) => c.classList.remove("active")); } });
}
async function runTranslate() {
  const paper = trSelected, lang = trLang || $("langCustom").value.trim();
  if (!paper) return toast("Select a paper.");
  if (!lang) return toast("Pick or type a language.");
  const out = $("translateResult"); out.innerHTML = "";
  const ld = el("div", "card"); ld.appendChild(el("div", "label", "Translating to " + lang + "…")); const t = el("div", "typing"); t.append(el("i"), el("i"), el("i")); ld.appendChild(t); out.appendChild(ld);
  try {
    const d = await api("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paper, language: lang }) });
    out.innerHTML = "";
    const card = el("div", "card"), h = el("div", "sub-head");
    h.appendChild(el("span", "label", lang)); h.appendChild(el("span", "cite", d.paper)); card.appendChild(h);
    const body = el("div", "answer"); body.innerHTML = md(d.text); card.appendChild(body);
    const actions = el("div", "actions"), dl = el("button", "linkbtn"); dl.innerHTML = '<i data-lucide="file-down"></i> Export PDF';
    dl.onclick = () => exportNodePDF(card, `${d.paper}.${lang}.pdf`, `${lang} — ${d.paper}`, d.text); actions.appendChild(dl); card.appendChild(actions);
    out.appendChild(card); icons(); refreshStats();
  } catch (e) { out.innerHTML = ""; toast(e.message, true); }
}

// ---------- stats / library ----------
async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("statChunks").textContent = s.chunks; $("statPapers").textContent = s.papers.length; $("statCalls").textContent = `${s.calls_used}/${s.calls_cap}`;
    $("meterFill").style.width = s.calls_cap ? Math.min(100, s.calls_used / s.calls_cap * 100) + "%" : "0%";
    currentPapers = s.papers;
    chatCounts = s.chat_counts || {};
    renderPapers(s.papers);
    renderComparePapers(s.papers);
    if (populateReview) populateReview(s.papers, () => selectedPaper, (v) => selectedPaper = v);
    if (populateTranslate) populateTranslate(s.papers, () => trSelected, (v) => trSelected = v);
  } catch (_) {}
}
async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData(); for (const f of files) fd.append("files", f);
  toast(`Ingesting ${files.length} file(s)…`);
  try { const d = await api("/api/ingest", { method: "POST", body: fd }); toast(`Indexed ${Object.values(d.results).reduce((a, b) => a + b, 0)} chunks.`); refreshStats(); }
  catch (e) { toast(e.message, true); }
}

// ---------- ui plumbing ----------
function autoresize() { const i = $("input"); i.style.height = "auto"; i.style.height = Math.min(i.scrollHeight, 180) + "px"; }
function openSidebar() { $("sidebar").classList.add("open"); $("backdrop").classList.add("show"); }
function closeSidebar() { $("sidebar").classList.remove("open"); $("backdrop").classList.remove("show"); }

// ---------- events ----------
document.querySelectorAll(".nav-item").forEach((b) => b.onclick = () => setMode(b.dataset.mode));
$("send").onclick = () => send();
$("input").addEventListener("input", autoresize);
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("compareBtn").onclick = runCompare;
$("compareInput").addEventListener("keydown", (e) => { if (e.key === "Enter") runCompare(); });
$("reviewBtn").onclick = runReview;
$("translateBtn").onclick = runTranslate;
$("fileInput").onchange = (e) => uploadFiles([...e.target.files]);
$("ingestFolder").onclick = async () => { toast("Ingesting folder…"); try { const d = await api("/api/ingest-folder", { method: "POST" }); toast(`Indexed ${d.total} chunks.`); refreshStats(); } catch (e) { toast(e.message, true); } };
$("clearIndex").onclick = async () => { if (!confirm("Clear the whole index?")) return; try { await api("/api/clear", { method: "POST" }); toast("Index cleared."); refreshStats(); } catch (e) { toast(e.message, true); } };
$("exportConv").onclick = () => {
  const list = currentList(); if (!thread || !list.length) return toast("Nothing to export in this chat.");
  const title = `CiteCheck chat — ${activePaper === "*" ? "All papers" : activePaper}`;
  const body = list.map((it, i) => `--- Exchange ${i + 1} ---\n${exportText(it)}`).join("\n\n");
  exportNodePDF(thread, "citecheck_chat.pdf", title, body);
};
$("menuBtn").onclick = openSidebar;
$("backdrop").onclick = closeSidebar;

// ---------- init ----------
SUGGESTIONS.forEach((s) => { const b = el("button", "suggestion", s); b.onclick = () => send(s); $("suggestions").appendChild(b); });
initLangChips();
populateReview = setupDropdown("reviewDropdown", "ddTrigger", "ddLabel", "ddMenu");
populateTranslate = setupDropdown("trDropdown", "trTrigger", "trLabel", "trMenu");
(async () => { await refreshStats(); await switchPaper(loadActive()); icons(); })();
