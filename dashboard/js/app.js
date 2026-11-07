/**
 * dashboard/js/app.js
 *
 * Shell: hash router, navigation, stay switcher, hour playback, keyboard
 * shortcuts, re-rendering.
 * Routes: #/overview #/patients #/map #/risk #/graph #/temporal
 *         #/performance #/settings   (#/graph?node=<feature> preselects a node)
 * Keys:   ← / →  previous / next hour (Shift: 6 h)    [ / ]  previous / next stay
 *         /      search stays
 */
(function () {
  "use strict";
  const PG = window.PG;
  const esc = PG.esc;

  const NAV = [
    { group: "Workspace", items: [["overview", "Overview"], ["patients", "Patient Explorer"], ["map", "Physiological Map"]] },
    { group: "Analysis", items: [["risk", "Risk Prediction"], ["graph", "Graph Explainability"], ["temporal", "Temporal Analysis"]] },
    { group: "Research", items: [["performance", "Model Performance"], ["settings", "Settings"]] },
  ];

  const $ = (id) => document.getElementById(id);
  const app = $("app"), page = $("page"), nav = $("nav");
  let route = "overview";

  nav.innerHTML = NAV.map((g) => `<li class="nav-label">${g.group}</li>` + g.items.map(([k, label]) =>
    `<li><a href="#/${k}" data-route="${k}">${PG.icon(k)}<span>${label}</span></a></li>`).join("")).join("");

  page.innerHTML = '<div id="caseHost" style="display:contents"></div><div id="pageBody" style="display:contents"></div>';
  const caseHost = $("caseHost"), body = $("pageBody");

  // ---------------------------------------------------------- sidebar data
  (function sidebarData() {
    const M = PG.DEMO_META;
    if (!M) return;
    const hours = PG.cases.reduce((a, c) => a + c.hours.length, 0);
    $("sidebarData").innerHTML = `<div class="sd-k">Dataset</div>
      <div class="sd-name">${esc(M.dataset)} ${esc(M.version)}</div>
      <div class="sd-stats"><span><b>${PG.cases.length}</b> stays</span><span><b>${PG.fmt.int(hours)}</b> hours</span></div>`;
  })();

  // ----------------------------------------------------------- stay switch
  PG.unitAbbr = function (c) {
    const u = (c.meta && c.meta.careunit) || "";
    const m = u.match(/\(([^)]+)\)/);
    if (m) return m[1].replace(/\/.*/, "").slice(0, 4);
    return u.split(/\s+/).filter(Boolean).map((w) => w[0]).join("").slice(0, 3).toUpperCase() || "ICU";
  };

  const splitRank = { test: 0, val: 1, train: 2 };
  const ordered = () => PG.cases.slice().sort((a, b) =>
    ((splitRank[a.meta && a.meta.split] ?? 3) - (splitRank[b.meta && b.meta.split] ?? 3)) || a.case_id.localeCompare(b.case_id));

  const pop = $("stayPop"), list = $("stayList"), search = $("staySearch"), stayBtn = $("stayBtn");
  let active = 0, shown = [];

  function renderStayButton() {
    const c = PG.currentCase();
    $("stayAvatar").textContent = PG.unitAbbr(c);
    $("stayLabel").textContent = `${c.case_id} · ${c.hours.length} h`;
    stayBtn.title = c.meta ? c.meta.careunit : c.label;
    const chip = $("sourceChip");
    chip.lastElementChild.textContent = PG.isDemo(c) ? "Real data · MIMIC-IV Demo" : "Real data · MIMIC-IV export";
  }

  function renderStayList() {
    const q = search.value.trim().toLowerCase();
    shown = ordered().filter((c) => !q || c.case_id.includes(q) || ((c.meta && c.meta.careunit) || "").toLowerCase().includes(q));
    active = Math.max(0, Math.min(active, shown.length - 1));
    let lastSplit = null;
    list.innerHTML = shown.length ? shown.map((c, i) => {
      const s = PG.caseStats(c), split = (c.meta && c.meta.split) || "export";
      const head = split !== lastSplit ? `<div class="sl-group">${split === "export" ? "Exported" : "Likely " + split + " split"}</div>` : "";
      lastSplit = split;
      return `${head}<div class="sl-item${i === active ? " active" : ""}${c.case_id === PG.state.caseId ? " current" : ""}" role="option" id="sl-${i}" data-i="${i}" aria-selected="${c.case_id === PG.state.caseId}">
        <span class="stay-avatar sm">${esc(PG.unitAbbr(c))}</span>
        <span class="sl-main"><b class="mono">${esc(c.case_id)}</b><span>${esc(c.meta ? c.meta.careunit : PG.sourceLabel(c))}</span></span>
        <span class="sl-meta"><span>${c.hours.length} h</span>${s.peak ? `<span class="sl-peak" title="Peak score"><i style="width:${(s.peak.score * 100).toFixed(0)}%"></i></span>` : '<span class="hint">no score</span>'}</span>
      </div>`;
    }).join("") : '<div class="sl-empty">No stays match.</div>';
    const el = $("sl-" + active);
    if (el) el.scrollIntoView({ block: "nearest" });
  }

  function openStays() {
    pop.hidden = false;
    stayBtn.setAttribute("aria-expanded", "true");
    search.value = "";
    active = Math.max(0, ordered().findIndex((c) => c.case_id === PG.state.caseId));
    renderStayList();
    search.focus();
  }
  function closeStays(focusBtn) {
    if (pop.hidden) return;
    pop.hidden = true;
    stayBtn.setAttribute("aria-expanded", "false");
    if (focusBtn) stayBtn.focus();
  }
  function stepStay(d) {
    const o = ordered(), i = o.findIndex((c) => c.case_id === PG.state.caseId);
    const n = o[(i + d + o.length) % o.length];
    if (n) PG.selectCase(n.case_id);
  }

  stayBtn.addEventListener("click", () => (pop.hidden ? openStays() : closeStays(true)));
  $("stayPrev").addEventListener("click", () => stepStay(-1));
  $("stayNext").addEventListener("click", () => stepStay(1));
  search.addEventListener("input", () => { active = 0; renderStayList(); });
  search.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(active + 1, shown.length - 1); renderStayList(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); renderStayList(); }
    else if (e.key === "Enter" && shown[active]) { e.preventDefault(); closeStays(true); PG.selectCase(shown[active].case_id); }
    else if (e.key === "Escape") { e.preventDefault(); closeStays(true); }
  });
  list.addEventListener("click", (e) => {
    const it = e.target.closest("[data-i]");
    if (!it) return;
    closeStays(true);
    PG.selectCase(shown[+it.dataset.i].case_id);
  });
  document.addEventListener("click", (e) => { if (!$("staySwitch").contains(e.target)) closeStays(false); });

  // ------------------------------------------------------------- playback
  let playTimer = null;
  function stopPlay() {
    if (!playTimer) return;
    clearInterval(playTimer); playTimer = null;
    const b = $("tlPlay"); if (b) b.classList.remove("playing");
  }
  function togglePlay() {
    if (playTimer) return stopPlay();
    const c = PG.currentCase(), H = c.hours.length, minT = Math.min(PG.MODEL.windowH - 1, H - 1);
    if (PG.state.t >= H - 1) PG.setState({ t: minT });
    $("tlPlay").classList.add("playing");
    playTimer = setInterval(() => {
      const t = PG.state.t + 1;
      if (t > H - 1) return stopPlay();
      PG.setState({ t });
    }, H > 200 ? 90 : 220);
  }
  function stepHour(d) {
    const c = PG.currentCase(), H = c.hours.length, minT = Math.min(PG.MODEL.windowH - 1, H - 1);
    PG.setState({ t: Math.max(minT, Math.min(H - 1, PG.state.t + d)) });
  }

  // -------------------------------------------------------------- render
  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    const [name, query] = raw.split("?");
    return { name: PG.PAGES[name] ? name : "overview", params: new URLSearchParams(query || "") };
  }

  function renderCaseHost() {
    const p = PG.PAGES[route], c = PG.currentCase();
    caseHost.innerHTML = p.caseBar ? PG.caseBar(c, PG.state.t) : "";
    if (!p.caseBar) return;
    PG.updateHero(c, PG.state.t);
    const slider = $("tSlider");
    slider.addEventListener("input", () => { stopPlay(); PG.setState({ t: +slider.value }); });
    $("tlPlay").addEventListener("click", togglePlay);
    $("tlBack").addEventListener("click", () => { stopPlay(); stepHour(-1); });
    $("tlFwd").addEventListener("click", () => { stopPlay(); stepHour(1); });
    caseHost.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => { stopPlay(); PG.setState({ t: +b.dataset.goto }); }));
  }

  function focusKey(el) {
    if (!el || !body.contains(el)) return null;
    for (const a of ["data-key", "data-select", "data-node", "data-edge", "data-var", "data-lens", "data-node-pick", "data-run", "data-route"]) {
      if (el.hasAttribute(a)) return `[${a}="${el.getAttribute(a)}"]`;
    }
    return el.id ? "#" + el.id : null;
  }

  function renderBody() {
    const sel = focusKey(document.activeElement);
    PG.PAGES[route].render(body);
    if (sel) { const el = body.querySelector(sel); if (el) el.focus({ preventScroll: true }); }
  }

  function renderAll(animate) {
    const p = PG.PAGES[route];
    $("pageTitle").textContent = p.title;
    $("crumb").textContent = "PhysioGraph · " + p.crumb;
    $("pageIcon").innerHTML = PG.icon(route);
    document.title = `${p.title} · PhysioGraph`;
    nav.querySelectorAll("a").forEach((a) => (a.dataset.route === route ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    renderStayButton();
    renderCaseHost();
    renderBody();
    if (animate && !PG.state.settings.reduceMotion) {
      page.classList.remove("enter"); void page.offsetWidth; page.classList.add("enter");
      clearTimeout(renderAll.t); renderAll.t = setTimeout(() => page.classList.remove("enter"), 900);
    }
  }

  PG.subscribe((state, patch) => {
    document.body.classList.toggle("reduce-motion", !!state.settings.reduceMotion);
    if ("caseId" in patch) { stopPlay(); return renderAll(true); }
    if ("settings" in patch) return renderAll(false);
    if ("t" in patch) PG.updateHero(PG.currentCase(), state.t);
    renderBody();
  });

  function onRoute() {
    const { name, params } = parseHash();
    stopPlay();
    route = name;
    const node = params.get("node");
    if (name === "graph" && node && PG.NODE[node]) Object.assign(PG.state, { graphNode: node, graphEdge: null });
    closeNav();
    closeStays(false);
    renderAll(true);
    window.scrollTo(0, 0);
  }

  // ------------------------------------------------------------ keyboard
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "select" || tag === "textarea" || e.target.isContentEditable;
    if (e.key === "Escape") { closeNav(); closeStays(true); return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); openStays(); }
    else if (e.key === "[") stepStay(-1);
    else if (e.key === "]") stepStay(1);
    else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && PG.PAGES[route].caseBar && !e.target.closest("[role=button], .marker, .gnode, .edge-g")) {
      e.preventDefault(); stopPlay(); stepHour((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 6 : 1));
    }
  });

  // ------------------------------------------------- mobile navigation drawer
  const menuBtn = $("menuBtn");
  function closeNav() { app.classList.remove("nav-open"); menuBtn.setAttribute("aria-expanded", "false"); }
  menuBtn.addEventListener("click", () => {
    const open = !app.classList.contains("nav-open");
    app.classList.toggle("nav-open", open);
    menuBtn.setAttribute("aria-expanded", String(open));
  });
  $("scrim").addEventListener("click", closeNav);

  // charts size to their container, so re-render on width changes
  let lastW = window.innerWidth, rt = null;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; renderBody(); } }, 150);
  });

  window.addEventListener("hashchange", onRoute);
  document.body.classList.toggle("reduce-motion", !!PG.state.settings.reduceMotion);
  onRoute();
})();
