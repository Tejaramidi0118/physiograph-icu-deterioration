/**
 * dashboard/js/pages/clinical.js
 *
 * Case-centred pages: Overview (body workspace), Physiological Map,
 * Patient Explorer.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);
  const f = PG.fmt;

  PG.PAGES = PG.PAGES || {};

  function bindSelect(root) {
    root.querySelectorAll("[data-select]").forEach((el) =>
      el.addEventListener("click", () => PG.setState({ selected: el.dataset.select })));
  }

  // --------------------------------------------------------------- overview
  PG.PAGES.overview = {
    title: "Overview",
    crumb: "Case workspace",
    caseBar: true,
    render(root) {
      const c = PG.currentCase(), t = PG.state.t, sel = PG.state.selected;
      const w = PG.windowRange(c, t);
      root.innerHTML = `
        <section class="workspace">
          <div class="card body-card">
            <div class="card-head">
              <div><h2>Physiological map</h2><div class="sub">The eight model variables at their anatomical reference sites. Select a marker for detail.</div></div>
              ${PG.statusLegend()}
            </div>
            <div class="body-stage" id="bodyStage"></div>
            <div class="body-foot">
              <span class="hint">Marker sites are display references. WBC, lactate and creatinine are blood laboratory values; the kidney marks creatinine’s reference organ only.</span>
            </div>
          </div>
          ${PG.detailPanel(c, t, sel)}
        </section>

        ${PG.metricCards(c, t, sel)}

        <section class="grid-2">
          <div class="card">
            <div class="card-head">
              <div><h3>Deterioration score</h3><div class="sub">8-hour target · ${f.hour(c, t)} · event window (t + 6, t + 8] h</div></div>
              <a href="#/risk" class="hint">Risk detail →</a>
            </div>
            <div class="card-body">
              ${PG.riskBlock(c, t)}
              <div id="miniHistory" style="margin-top:14px"></div>
              <p class="hint" style="margin-top:6px">A model-generated score for research review — not a diagnosis and not a clinical recommendation.</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head">
              <div><h3>Observation window</h3><div class="sub">Status of each model input, h ${c.hours[w.start]}–${c.hours[w.end]}</div></div>
              <a href="#/temporal" class="hint">Temporal analysis →</a>
            </div>
            <div class="card-body">
              ${PG.statusMatrix(c, w.start, w.end)}
              <div style="margin-top:12px">${PG.statusLegend()}</div>
            </div>
          </div>
        </section>`;

      PG.renderBody(root.querySelector("#bodyStage"), {
        case: c, t, selected: sel, mode: "interactive",
        onSelect: (key) => PG.setState({ selected: key }),
      });
      PG.mountDetailChart(root, c, t, sel);
      PG.mountRiskHistory(root.querySelector("#miniHistory"), c, t, { height: 120, onPick: (h) => PG.setState({ t: h }) });
      bindSelect(root);
    },
  };

  // ------------------------------------------------------- physiological map
  PG.PAGES.map = {
    title: "Physiological Map",
    crumb: "Variables and anatomy",
    caseBar: true,
    render(root) {
      const c = PG.currentCase(), t = PG.state.t, sel = PG.state.selected;
      const ranked = PG.NODES.slice().sort((a, b) => b.gradShare - a.gradShare);
      root.innerHTML = `
        <section class="grid-main-side">
          <div class="card body-card">
            <div class="card-head">
              <div><h2>Anatomical reference map</h2><div class="sub">All eight variables annotated at ${f.hour(c, t)}</div></div>
              ${PG.statusLegend()}
            </div>
            <div class="body-stage" id="bodyStage"></div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Measurement sites</h3><div class="sub">What each marker represents</div></div></div>
            <div class="card-body">
              <div class="ranklist">
                ${PG.NODES.map((n) => `
                  <div class="rank${sel === n.key ? " sel" : ""}" data-select="${n.key}" style="--mc:${PG.SYSTEMS[n.system].color};grid-template-columns:20px minmax(0,1fr)" role="button" tabindex="0">
                    <span class="i">${n.index}</span>
                    <span><span class="n"><i></i><b>${esc(n.label)}</b></span><span class="hint">${n.kind === "lab" ? "Laboratory · " : "Vital sign · "}${esc(n.site)}</span></span>
                  </div>`).join("")}
              </div>
              <hr class="divider">
              <p class="hint"><strong>Laboratory values are not organ measurements.</strong> WBC and lactate are systemic blood markers; serum creatinine is drawn from blood and only conceptually references the kidney.</p>
            </div>
          </div>
        </section>

        <div class="card">
          <div class="card-head"><div><h3>Variable registry</h3><div class="sub">Canonical feature names, MIMIC-IV items and pipeline parameters</div></div></div>
          <div class="card-body flush table-wrap">
            <table class="data">
              <thead><tr><th>Node</th><th>Variable</th><th>Feature key</th><th>MIMIC item</th><th>Unit</th><th>System</th><th class="r">Carry-forward</th><th class="r">Plausibility</th><th class="r">Missing after fill (test)</th><th class="r">Gradient share</th></tr></thead>
              <tbody>${PG.NODES.map((n) => `
                <tr class="clickable${sel === n.key ? " hl" : ""}" data-select="${n.key}">
                  <td class="mono">${n.index}</td><td>${esc(n.label)}</td><td class="mono">${n.key}</td>
                  <td class="mono">${n.itemid}<span class="sub">${n.source}</span></td><td>${esc(n.unit)}</td><td>${esc(PG.SYSTEMS[n.system].label)}</td>
                  <td class="r">${n.carryH} h</td><td class="r">${n.plaus[0]}–${n.plaus[1]}</td>
                  <td class="r">${n.missTestPct.toFixed(2)}%</td><td class="r">${(n.gradShare * 100).toFixed(1)}% <span class="sub">rank ${ranked.indexOf(n) + 1}</span></td>
                </tr>`).join("")}</tbody>
            </table>
          </div>
        </div>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Node ordering reconciliation</h3><div class="sub">Indices differ between files; every view here resolves nodes by feature name</div></div></div>
            <div class="card-body flush table-wrap">
              <table class="data">
                <thead><tr><th>Feature</th>${PG.ORDERINGS.map((o, i) => `<th class="r">${i === 0 ? "Training notebook" : "Graph module / rebuild"}</th>`).join("")}</tr></thead>
                <tbody>${PG.NODES.map((n) => {
                  const idx = PG.ORDERINGS.map((o) => o.order.indexOf(n.key));
                  const diff = idx[0] !== idx[1];
                  return `<tr${diff ? ' class="hl"' : ""}><td class="mono">${n.key}</td>${idx.map((i) => `<td class="r mono">${i}</td>`).join("")}</tr>`;
                }).join("")}</tbody>
              </table>
            </div>
            <div class="card-body"><p class="hint">Canonical: the executed training notebook (cell 4), whose order defines the 15 edge indices used by the trained model. Highlighted rows differ. Data columns are selected by name in both, so no values were mixed — only figures that copy indices between files would mislabel nodes 5–7.</p></div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Preprocessing pipeline</h3><div class="sub">Corrected forward-only pipeline · no backward interpolation</div></div></div>
            <div class="card-body flush"><div class="steps" style="grid-template-columns:repeat(2,minmax(0,1fr))">${PG.PIPELINE.map((s) => `<div class="step" style="border-right:1px solid var(--border)"><div class="k">${esc(s.step)}</div><div class="d">${esc(s.detail)}</div></div>`).join("")}</div></div>
          </div>
        </section>`;

      PG.renderBody(root.querySelector("#bodyStage"), {
        case: c, t, selected: sel, mode: "annotated",
        onSelect: (key) => PG.setState({ selected: key }),
      });
      bindSelect(root);
      root.querySelectorAll(".rank[data-select]").forEach((el) => el.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); el.click(); }
      }));
    },
  };

  // --------------------------------------------------------- patient explorer
  PG.PAGES.patients = {
    title: "Patient Explorer",
    crumb: "ICU stays in this session",
    caseBar: false,
    render(root) {
      const ex = (PG.state.explorer = PG.state.explorer || { q: "", sort: "labs", split: "all" });
      const M = PG.DEMO_META || {};
      const stats = PG.cases.map((c) => {
        const H = c.hours.length;
        const obs = Object.fromEntries(PG.NODES.map((n) => [n.key, c.variables[n.key].filter((r) => r.s === "observed").length / H]));
        const labs = ["wbc", "lactate", "creatinine"].reduce((a, k) => a + c.variables[k].filter((r) => r.s === "observed").length, 0);
        const best = c.predictions && c.predictions.length ? c.predictions.reduce((a, b) => (b.score > a.score ? b : a)) : null;
        return { c, H, obs, labs, best };
      });
      const splits = stats.reduce((a, s) => ((a[(s.c.meta && s.c.meta.split) || "export"] = (a[(s.c.meta && s.c.meta.split) || "export"] || 0) + 1), a), {});
      const totalH = stats.reduce((a, s) => a + s.H, 0);

      const rowsHtml = () => {
        const q = ex.q.trim().toLowerCase();
        const list = stats.filter((s) =>
          (!q || s.c.case_id.includes(q) || (s.c.meta && s.c.meta.careunit.toLowerCase().includes(q))) &&
          (ex.split === "all" || (s.c.meta && s.c.meta.split) === ex.split));
        const by = { labs: (a, b) => b.labs - a.labs, hours: (a, b) => b.H - a.H, score: (a, b) => ((b.best && b.best.score) || -1) - ((a.best && a.best.score) || -1), id: (a, b) => a.c.case_id.localeCompare(b.c.case_id) }[ex.sort];
        list.sort(by);
        if (!list.length) return `<tr><td colspan="7" class="hint">No stays match.</td></tr>`;
        return list.map(({ c, H, obs, best }) => `<tr class="clickable${c.case_id === PG.state.caseId ? " hl" : ""}" data-case="${esc(c.case_id)}">
          <td><b class="mono">${esc(c.case_id)}</b><span class="sub">${esc(c.meta ? c.meta.careunit : PG.sourceLabel(c))}</span></td>
          <td class="r">${H} h<span class="sub">${c.meta ? c.meta.los_days + " d LOS" : ""}</span></td>
          <td>${PG.splitTag(c) || PG.sourceTag(c)}</td>
          <td style="min-width:220px">
            <div class="obs-bars">${PG.NODES.map((n) => `<div title="${esc(n.label)}: ${(obs[n.key] * 100).toFixed(0)}% of hours observed"><div class="ob"><i style="height:${Math.max(2, obs[n.key] * 100)}%;background:${PG.SYSTEMS[n.system].color}"></i></div><span>${esc(n.short)}</span></div>`).join("")}</div>
          </td>
          <td class="r">${c.predictions ? c.predictions.length : 0}</td>
          <td class="r">${best ? `<b>${best.score.toFixed(3)}</b><span class="sub">at ${PG.fmt.hour(c, best.hour)}</span>` : '<span class="hint">&lt; 24 h</span>'}</td>
          <td class="r"><button class="btn" data-open="${esc(c.case_id)}">Open</button></td>
        </tr>`).join("");
      };

      root.innerHTML = `
        <section class="kpi-grid">
          <div class="card kpi"><div class="kpi-title">ICU stays</div><div class="kpi-vals"><div><div class="v">${PG.cases.length}</div></div></div><div class="prev">${f.int(totalH)} hourly rows</div></div>
          <div class="card kpi"><div class="kpi-title">Scored hours</div><div class="kpi-vals"><div><div class="v">${f.int(stats.reduce((a, s) => a + (s.c.predictions ? s.c.predictions.length : 0), 0))}</div></div></div><div class="prev">${stats.filter((s) => !s.best).length} stays shorter than 24 h have no score</div></div>
          <div class="card kpi"><div class="kpi-title">Likely split by admission date</div><div class="kpi-vals">${["train", "val", "test"].map((k) => `<div><div class="k">${k}</div><div class="v">${splits[k] || 0}</div></div>`).join("")}</div><div class="prev">Boundaries from data_exploration cell 92</div></div>
          <div class="card kpi"><div class="kpi-title">Source</div><div style="font-size:13px;font-weight:600">${esc(M.dataset || "MIMIC-IV")} ${esc(M.version || "")}</div><div class="prev">ODbL v1.0 · SHA-256 verified · built ${esc(M.built || "")}</div></div>
        </section>

        <div class="card">
          <div class="card-head">
            <div><h2>ICU stays</h2><div class="sub">Real stays from ${esc(M.dataset || "MIMIC-IV")}. Bars show the share of hours with a directly charted value.</div></div>
            <div class="explorer-tools">
              <input type="search" class="input" id="exQ" placeholder="Stay ID or care unit" value="${esc(ex.q)}" aria-label="Search stays">
              <select class="select" id="exSplit" aria-label="Split">${["all", "train", "val", "test"].map((k) => `<option value="${k}"${ex.split === k ? " selected" : ""}>${k === "all" ? "All splits" : "Likely " + k}</option>`).join("")}</select>
              <select class="select" id="exSort" aria-label="Sort">${[["labs", "Most lab observations"], ["hours", "Longest"], ["score", "Highest score"], ["id", "Stay ID"]].map(([k, l]) => `<option value="${k}"${ex.sort === k ? " selected" : ""}>${l}</option>`).join("")}</select>
            </div>
          </div>
          <div class="card-body flush table-wrap explorer-table">
            <table class="data">
              <thead><tr><th>Stay</th><th class="r">Length</th><th>Split</th><th>Observed share by variable</th><th class="r">Scored h</th><th class="r">Peak score</th><th></th></tr></thead>
              <tbody id="exRows">${rowsHtml()}</tbody>
            </table>
          </div>
          <div class="card-body"><p class="hint">Peak score is the highest hourly score from the pre-correction checkpoint with past-only inputs. Stays whose admission date falls in the training period may have been seen by the checkpoint.</p></div>
        </div>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Load an exported case</h3><div class="sub">physiograph.case/1 JSON produced by the export script</div></div></div>
            <div class="card-body">
              <label class="file-drop" id="drop" for="fileInput">
                <b>Drop a case file here</b> or choose a file<br>
                <span class="hint">Read in this tab only — nothing is uploaded or stored.</span>
              </label>
              <input type="file" id="fileInput" accept=".json,application/json" class="sr-only">
              <div id="loadMsg"></div>
              <hr class="divider">
              <p class="hint">Exported cases are de-identified MIMIC-IV data covered by the PhysioNet data use agreement. Keep files within your credentialed environment.</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Export a case from project data</h3><div class="sub">Reads the corrected <span class="mono">*_filled.parquet</span> and recovers observation status</div></div></div>
            <div class="card-body">
<pre class="code">python scripts/export_dashboard_case.py \\
  --data data/processed --split test \\
  --stay-id &lt;STAY_ID&gt; \\
  --predictions predictions/test_scores.csv \\
  --out dashboard/cases/&lt;STAY_ID&gt;.json</pre>
              <p class="hint" style="margin-top:10px"><strong>Status recovery.</strong> The pipeline keeps a <span class="mono">*_missing</span> flag per hour. A flagged hour within the feature’s carry-forward limit of the last observation is <em>carried</em>; beyond it, the value is the training median. Scores are included only from a predictions CSV (<span class="mono">stay_id, hour, score</span>) and are tagged as model output.</p>
            </div>
          </div>
        </section>`;

      const tbody = root.querySelector("#exRows");
      tbody.addEventListener("click", (ev) => {
        const tr = ev.target.closest("[data-case]");
        if (!tr) return;
        const open = ev.target.closest("[data-open]");
        PG.selectCase(tr.dataset.case);
        if (open) location.hash = "#/overview";
      });
      const refresh = () => (tbody.innerHTML = rowsHtml());
      root.querySelector("#exQ").addEventListener("input", (e) => { ex.q = e.target.value; refresh(); });
      root.querySelector("#exSplit").addEventListener("change", (e) => { ex.split = e.target.value; refresh(); });
      root.querySelector("#exSort").addEventListener("change", (e) => { ex.sort = e.target.value; refresh(); });

      const input = root.querySelector("#fileInput"), drop = root.querySelector("#drop"), msg = root.querySelector("#loadMsg");
      const load = (file) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const c = PG.validateCase(JSON.parse(reader.result));
            PG.addCase(c);
            location.hash = "#/overview";
          } catch (e) {
            msg.innerHTML = `<p class="load-error">Could not load ${esc(file.name)}: ${esc(e.message)}</p>`;
          }
        };
        reader.readAsText(file);
      };
      input.addEventListener("change", () => load(input.files[0]));
      drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
      drop.addEventListener("dragleave", () => drop.classList.remove("over"));
      drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); load(e.dataTransfer.files[0]); });
    },
  };
})();
