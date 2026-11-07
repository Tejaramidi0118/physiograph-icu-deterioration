/**
 * dashboard/js/components/panels.js
 *
 * Shared building blocks: case bar, variable detail panel, metric cards,
 * risk score block, observation-status matrix.
 * Each returns an HTML string; `mount*` functions draw charts after insertion.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);
  const f = PG.fmt;

  PG.ICONS = {
    overview: '<path d="M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z"/>',
    patients: '<path d="M4 6h16M4 12h16M4 18h10"/><circle cx="19" cy="18" r="2"/>',
    map: '<circle cx="12" cy="4.5" r="2.2"/><path d="M8 9.5c1.2-.8 2.5-1.1 4-1.1s2.8.3 4 1.1l1 6.5M8 9.5 7 16M10 9v12M14 9v12"/>',
    risk: '<path d="M3 17l5-6 4 3 5-7 4 4"/><path d="M3 21h18"/>',
    graph: '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><circle cx="12" cy="10" r="1.6"/><path d="M7 6h10M6.2 7.6l4.8 9M17.8 7.6 13 16.6M12 11.6V16"/>',
    temporal: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/>',
    performance: '<path d="M5 20V11M10 20V5M15 20v-7M20 20V9M3 20h18"/>',
    settings: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    alert: '<path d="M12 3.5 2.5 20h19L12 3.5z"/><path d="M12 10v4.5M12 17.2v.1"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.1"/>',
  };
  PG.icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${PG.ICONS[name]}</svg>`;

  PG.notice = (html, kind) => `<div class="notice${kind ? " " + kind : ""}">${PG.icon(kind === "info" ? "info" : "alert")}<div>${html}</div></div>`;

  PG.sourceTag = (c) => `<span class="tag real">${esc(PG.sourceLabel(c))}</span>`;

  PG.splitTag = function (c) {
    if (!c.meta || !c.meta.split) return "";
    return c.meta.split === "train"
      ? '<span class="tag unverified" title="Admission date falls inside the training period (data_exploration cell 92). The checkpoint may have seen this stay.">likely training split</span>'
      : `<span class="tag" title="Admission date falls inside the ${c.meta.split} period (data_exploration cell 92)">likely ${c.meta.split} split</span>`;
  };

  PG.scoreSourceTag = function (c) {
    if (c.prediction_source === "model") {
      const pre = c.score_profile === "pre-correction";
      return `<span class="tag ${pre ? "unverified" : "real"}">Model output${c.checkpoint ? " · " + esc(c.checkpoint) : ""}</span>`;
    }
    return '<span class="tag unverified">Score source not declared</span>';
  };

  // ------------------------------------------------------------- case hero
  /** Per-stay summary used by the hero and the stay switcher (cached). */
  PG.caseStats = function (c) {
    if (c._stats) return c._stats;
    const H = c.hours.length;
    const vit = ["resp_rate", "spo2", "heart_rate", "sbp"];
    const vitObs = vit.reduce((a, k) => a + c.variables[k].filter((r) => r.s === "observed").length, 0) / (H * vit.length);
    const labs = ["wbc", "lactate", "creatinine"].reduce((a, k) => a + c.variables[k].filter((r) => r.s === "observed").length, 0);
    const preds = c.predictions || [];
    const peak = preds.length ? preds.reduce((a, b) => (b.score > a.score ? b : a)) : null;
    return (c._stats = { H, vitObs, labs, scored: preds.length, peak });
  };

  /** 270° score ring with the active profile's thresholds as ticks. */
  PG.gauge = function (c, t) {
    const p = PG.prediction(c, t), th = PG.thresholdInfo(c), prof = PG.profile(c);
    const R = 46, C = 60, A0 = 135, SWEEP = 270;
    const pt = (deg, r = R) => { const a = (deg * Math.PI) / 180; return [C + r * Math.cos(a), C + r * Math.sin(a)]; };
    const arc = (from, to) => {
      const [x1, y1] = pt(A0 + SWEEP * from), [x2, y2] = pt(A0 + SWEEP * to);
      return `M${x1.toFixed(2)},${y1.toFixed(2)}A${R},${R} 0 ${SWEEP * (to - from) > 180 ? 1 : 0} 1 ${x2.toFixed(2)},${y2.toFixed(2)}`;
    };
    const ticks = prof.thresholds.map((x) => {
      const [a, b] = [pt(A0 + SWEEP * x.value, R - 9), pt(A0 + SWEEP * x.value, R + 9)];
      return `<line class="g-tick${x.value === th.value ? " on" : ""}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"><title>Threshold ${x.label}</title></line>`;
    }).join("");
    const above = p && p.score >= th.value;
    return `<svg class="gauge-svg" viewBox="0 0 120 120" role="img" aria-label="${p ? "Score " + p.score.toFixed(3) : "No score"} at ${f.hour(c, t)}">
        <defs><linearGradient id="g-grad" x1="0" x2="1"><stop offset="0" style="stop-color:var(--teal)"/><stop offset="1" style="stop-color:${above ? "var(--risk)" : "var(--accent)"}"/></linearGradient></defs>
        <path class="g-track" d="${arc(0, 1)}"/>
        ${p ? `<path class="g-val" d="${arc(0, Math.max(0.004, p.score))}" stroke="url(#g-grad)"/>` : ""}
        ${ticks}
        <text class="g-num" x="60" y="62" text-anchor="middle">${p ? p.score.toFixed(3) : "—"}</text>
        <text class="g-lbl" x="60" y="78" text-anchor="middle">${p ? "score" : "no score"}</text>
      </svg>
      <div class="gauge-cap">${p ? (above ? `<span class="tag risk">≥ ${th.label}</span>` : `<span class="tag">&lt; ${th.label}</span>`) : `<span class="tag">needs 24 h</span>`}<span class="hint">${f.hour(c, t)}</span></div>`;
  };

  /** Score history across the whole stay, drawn behind the hour slider. */
  function timelineSvg(c) {
    const H = c.hours.length, W = 1000, Ht = 56;
    const X = (h) => (H > 1 ? (h / (H - 1)) * W : 0);
    const preds = c.predictions || [];
    let d = "";
    if (preds.length) {
      d = preds.map((p, i) => `${i ? "L" : "M"}${X(p.hour).toFixed(1)},${(Ht - 4 - p.score * (Ht - 10)).toFixed(1)}`).join("");
      d = `<path class="tl-area" d="${d}L${X(preds[preds.length - 1].hour)},${Ht}L${X(preds[0].hour)},${Ht}Z"/><path class="tl-line" d="${d}"/>`;
    }
    const th = PG.thresholdInfo(c).value;
    const ty = (Ht - 4 - th * (Ht - 10)).toFixed(1);
    return `<svg class="tl-svg" viewBox="0 0 ${W} ${Ht}" preserveAspectRatio="none" aria-hidden="true">
      <defs><linearGradient id="tl-grad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" style="stop-color:var(--accent);stop-opacity:.28"/><stop offset="1" style="stop-color:var(--accent);stop-opacity:0"/></linearGradient></defs>
      ${preds.length ? `<line class="tl-th" x1="0" x2="${W}" y1="${ty}" y2="${ty}"/>` : ""}${d}</svg>`;
  }

  PG.caseBar = function (c, t) {
    const H = c.hours.length;
    const minT = Math.min(PG.MODEL.windowH - 1, H - 1);
    const s = PG.caseStats(c);
    return `<section class="hero" aria-label="Case context">
      <div class="hero-top">
        <div class="hero-id">
          <div class="hero-kicker"><span class="live-dot"></span>${esc(PG.sourceLabel(c))} · real ICU stay ${PG.splitTag(c)}</div>
          <h2 class="hero-title">Stay <span class="mono">${esc(c.case_id)}</span></h2>
          <div class="hero-sub">${c.meta ? esc(c.meta.careunit) + " · " + c.meta.los_days + " d ICU stay · " : ""}${H} hourly rows</div>
          <div class="hero-stats">
            <div class="hs"><span class="k">Vitals charted</span><span class="v">${(s.vitObs * 100).toFixed(0)}%</span></div>
            <div class="hs"><span class="k">Lab draws</span><span class="v">${s.labs}</span></div>
            <div class="hs"><span class="k">Scored hours</span><span class="v">${s.scored}</span></div>
            <div class="hs"><span class="k">Peak score</span><span class="v">${s.peak ? s.peak.score.toFixed(3) : "—"}</span>${s.peak ? `<button class="hs-link" data-goto="${s.peak.hour}">h ${c.hours[s.peak.hour]} ↗</button>` : ""}</div>
          </div>
        </div>
        <div class="hero-gauge" id="heroGauge">${PG.gauge(c, t)}</div>
      </div>
      <div class="hero-timeline">
        <div class="tl-controls">
          <button class="icon-btn" id="tlBack" aria-label="Previous hour" title="Previous hour (←)"><svg viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6"/></svg></button>
          <button class="icon-btn play" id="tlPlay" aria-label="Replay the stay hour by hour" title="Replay"><svg viewBox="0 0 24 24"><path class="i-play" d="M8 5.5v13l11-6.5z"/><path class="i-pause" d="M8 5h3v14H8zM13 5h3v14h-3z"/></svg></button>
          <button class="icon-btn" id="tlFwd" aria-label="Next hour" title="Next hour (→)"><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg></button>
        </div>
        <div class="tl-track">
          ${timelineSvg(c)}
          <div class="tl-band" id="tlBand"></div>
          <input type="range" id="tSlider" min="${minT}" max="${H - 1}" step="1" value="${t}" aria-label="Prediction hour" aria-valuetext="ICU hour ${c.hours[t]}">
          <div class="tl-axis"><span>h ${c.hours[0]}</span><span>score history · shaded = 24 h window</span><span>h ${c.hours[H - 1]}</span></div>
        </div>
        <div class="case-facts" id="caseFacts">${PG.caseFacts(c, t)}</div>
      </div>
      ${PG.provenanceStrip(c)}
    </section>`;
  };

  /** Keep the hero in sync with the prediction hour without rebuilding it. */
  PG.updateHero = function (c, t) {
    const el = (id) => document.getElementById(id);
    if (el("caseFacts")) el("caseFacts").innerHTML = PG.caseFacts(c, t);
    if (el("heroGauge")) el("heroGauge").innerHTML = PG.gauge(c, t);
    const slider = el("tSlider");
    if (slider) {
      if (+slider.value !== t) slider.value = t;
      slider.setAttribute("aria-valuetext", "ICU hour " + c.hours[t]);
    }
    const band = el("tlBand"), H = c.hours.length;
    if (band && H > 1) {
      const w = PG.windowRange(c, t);
      band.style.left = (w.start / (H - 1)) * 100 + "%";
      band.style.width = ((w.end - w.start) / (H - 1)) * 100 + "%";
    }
  };

  PG.provenanceStrip = function (c) {
    if (!PG.isDemo(c)) return "";
    const train = c.meta && c.meta.split === "train";
    return `<details class="prov">
      <summary>${PG.icon("info")}<span><b>Real data.</b> Scores from the pre-correction checkpoint, past hours only — research output, not a diagnosis.${train ? " Admitted in the training period." : ""}</span><span class="prov-more">Details</span></summary>
      <div class="prov-body">Values come from the MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL v1.0), extracted as in <span class="mono">data_exploration.ipynb</span> and filled forward only by <span class="mono">rebuild_filled_parquets.py</span>. Scores and attributions come from <span class="mono">models/best_physiograph_multi_horizon_1_1.pt</span> (epoch 8, the run recorded in <span class="mono">PhysioGraph_Final_Colab.ipynb</span>) using the notebook’s model code. They are not validated.${train ? " This stay’s admission date falls in the training period, so the checkpoint may have seen it." : ""}</div>
    </details>`;
  };

  PG.caseFacts = function (c, t) {
    const w = PG.windowRange(c, t);
    const hz = PG.MODEL.horizon;
    const stamp = f.stamp(c, t);
    return `
      <div class="fact"><div class="k">Prediction at</div><div class="v">${f.hour(c, t)}${stamp ? ` <span class="hint">${esc(stamp)}</span>` : ""}</div></div>
      <div class="fact"><div class="k">Observation window</div><div class="v">h ${c.hours[w.start]}–${c.hours[w.end]}${w.complete ? "" : ' <span class="tag unverified">incomplete</span>'}</div></div>
      <div class="fact"><div class="k">Target interval</div><div class="v">(h ${c.hours[t] + hz.from}, h ${c.hours[t] + hz.to}]</div></div>`;
  };

  // ------------------------------------------------------------- statuses
  PG.whenText = function (c, key, rec, t) {
    const node = PG.NODE[key];
    const stamp = f.stamp(c, t);
    if (rec.s === "observed") return `Charted at ${f.hour(c, t)}${stamp ? " · " + stamp : ""}`;
    if (rec.s === "carried") {
      const ostamp = f.stamp(c, rec.o);
      return `Last observed ${f.hour(c, rec.o)}${ostamp ? " · " + ostamp : ""} · carried ${c.hours[t] - c.hours[rec.o]} h of ${node.carryH} h limit`;
    }
    if (rec.s === "median") {
      const last = PG.lastObserved(c, key, t);
      return `No observation within the ${node.carryH} h carry-forward limit${last ? ` · last observed ${f.hour(c, last.h)}` : " · not observed in this stay"}. Model input is the training-set median.`;
    }
    return "No value exported for this hour.";
  };

  PG.shortWhen = function (c, rec, t) {
    if (rec.s === "observed") return `h ${c.hours[t]}`;
    if (rec.s === "carried") return `from h ${c.hours[rec.o]}`;
    if (rec.s === "median") return "imputed";
    return "—";
  };

  // --------------------------------------------------------- detail panel
  PG.detailPanel = function (c, t, key) {
    const n = PG.NODE[key];
    const rec = PG.record(c, key, t);
    const sys = PG.SYSTEMS[n.system];
    const imputed = rec.s === "median" || rec.s === "unavailable";
    const valTxt = rec.v === null ? "—" : f.value(key, rec.v);
    const av = PG.availability(c, key, t);
    const w = PG.windowRange(c, t);
    const ranked = PG.NODES.slice().sort((a, b) => b.gradShare - a.gradShare);
    const rank = ranked.findIndex((x) => x.key === key) + 1;
    const maxShare = ranked[0].gradShare;
    const outE = PG.EDGES.filter((e) => e.from === key);
    const inE = PG.EDGES.filter((e) => e.to === key);
    const chip = (e, dir) => `<span class="edge-chip ${e.type}" title="${esc(e.rationale)}">${dir === "out" ? "→ " + esc(PG.NODE[e.to].short) : esc(PG.NODE[e.from].short) + " →"} · ${esc(e.rationale)}</span>`;
    const pct = (k) => (av.total ? (av[k] / av.total) * 100 : 0);
    const caseAttr = PG.attributionAt(c, t);
    const caseVals = caseAttr ? PG.NODES.map((x) => caseAttr[x.key]) : [];
    const caseMax = caseAttr ? Math.max(...caseVals) : 1;
    const caseRank = caseAttr ? caseVals.filter((v) => v > caseAttr[key]).length + 1 : null;

    return `<aside class="detail card" style="--mc:${sys.color}" aria-live="polite">
      <div class="detail-head">
        <div class="detail-kicker">
          <span class="tag">Node ${n.index}</span>
          <span class="tag mono">${n.source} ${n.itemid}</span>
          <span class="tag">${esc(sys.label)}</span>
        </div>
        <h2>${esc(n.label)}</h2>
        <div class="detail-site">${n.kind === "lab" ? "Laboratory value" : "Vital sign"} · ${esc(n.site)}</div>
        <div class="detail-value${imputed ? " is-imputed" : ""}"><span class="v">${esc(valTxt)}</span><span class="u">${rec.v === null ? (rec.s === "median" ? "training median — value not shown" : "") : esc(n.unit)}</span></div>
        <div class="detail-status">
          <span class="st ${rec.s}">${PG.STATUS[rec.s].label}</span>
          <span class="when">${esc(PG.whenText(c, key, rec, t))}</span>
        </div>
      </div>
      <div class="detail-section">
        <h4><span>24-hour trend</span><span class="mono">h ${c.hours[w.start]}–${c.hours[w.end]}</span></h4>
        <div class="detail-chart" id="detailChart"></div>
      </div>
      <div class="detail-section">
        <h4><span>Data availability in window</span><span>${av.observed}/${av.total} observed</span></h4>
        <div class="avail" role="img" aria-label="${av.observed} observed, ${av.carried} carried forward, ${av.median} training median, ${av.unavailable} unavailable">
          <i class="observed" style="width:${pct("observed")}%"></i><i class="carried" style="width:${pct("carried")}%"></i><i class="median" style="width:${pct("median")}%"></i><i class="unavailable" style="width:${pct("unavailable")}%"></i>
        </div>
        <div class="avail-legend">
          <span class="st observed">${av.observed} observed</span>
          <span class="st carried">${av.carried} carried</span>
          <span class="st median">${av.median} median</span>
          ${av.unavailable ? `<span class="st unavailable">${av.unavailable} unavailable</span>` : ""}
        </div>
      </div>
      <div class="detail-section">
        <h4><span>Model attribution</span>${caseAttr ? `<span class="tag accent">this hour</span>` : `<span class="tag">cohort only</span>`}</h4>
        ${caseAttr ? `<div class="share-row">
            <div class="bar"><i style="width:${(caseAttr[key] / caseMax) * 100}%"></i></div>
            <span class="num"><b>${(caseAttr[key] * 100).toFixed(1)}%</b> · rank ${caseRank} of 8</span>
          </div>
          <p class="hint" style="margin-top:6px">Plain-gradient share of the score at ${f.hour(c, t)}, ${esc(c.checkpoint || "model")}.</p>`
        : `<p class="hint">${c.predictions && t < PG.MODEL.windowH - 1 ? "No score yet at this hour, so no attribution." : "No per-hour attribution in this case file."}</p>`}
        <div class="share-row" style="margin-top:10px">
          <div class="bar cohort"><i style="width:${(n.gradShare / maxShare) * 100}%"></i></div>
          <span class="num hint">${(n.gradShare * 100).toFixed(1)}% · rank ${rank}</span>
        </div>
        <p class="hint" style="margin-top:6px">Cohort reference: corrected run, ${PG.EXPLAIN.windows} positive test windows (briefing §6.2).</p>
        ${n.caveat ? `<div class="caveat" style="margin-top:8px">${esc(n.caveat)}</div>` : ""}
      </div>
      <div class="detail-section">
        <h4><span>Graph connections</span><a href="#/graph?node=${key}">Open graph</a></h4>
        <div class="edge-chips">${inE.map((e) => chip(e, "in")).join("")}${outE.map((e) => chip(e, "out")).join("")}</div>
      </div>
      <div class="detail-section">
        <h4><span>Pipeline</span></h4>
        <dl class="kv">
          <dt>Carry-forward limit</dt><dd>${n.carryH} h</dd>
          <dt>Plausibility range</dt><dd>${n.plaus[0]}–${n.plaus[1]} ${esc(n.unitShort)}</dd>
          <dt>Missing after fill (test)</dt><dd>${n.missTestPct.toFixed(2)}%</dd>
        </dl>
        <p class="hint" style="margin-top:8px">${esc(n.siteNote)}</p>
      </div>
    </aside>`;
  };

  PG.trendTooltip = function (c, key) {
    const n = PG.NODE[key];
    return (h) => {
      const r = PG.record(c, key, h);
      return `<div class="t">${f.hour(c, h)}</div>
        <div class="r"><span>${esc(n.short)}</span><b>${r.v === null ? "—" : f.value(key, r.v) + " " + esc(n.unitShort)}</b></div>
        <div class="r"><span>Status</span><b>${PG.STATUS[r.s].label}</b></div>
        ${r.s === "carried" ? `<div class="r"><span>Observed</span><b>${f.hour(c, r.o)}</b></div>` : ""}`;
    };
  };

  PG.mountDetailChart = function (root, c, t, key) {
    const el = root.querySelector("#detailChart");
    if (!el) return;
    const n = PG.NODE[key];
    const w = PG.windowRange(c, t);
    PG.chart(el, {
      height: 132,
      margin: { t: 16, r: 8, b: 24, l: 38 },
      x: { domain: [w.start, w.end], fmt: (h) => "h" + c.hours[h], ticks: [w.start, Math.round((w.start + w.end) / 2), w.end] },
      y: { label: n.unitShort, fmt: PG.tickFmt },
      series: [{ records: PG.windowRecords(c, key, t), color: PG.SYSTEMS[n.system].color, status: true }],
      tooltip: PG.trendTooltip(c, key),
      aria: `${n.label} over the observation window`,
    });
  };

  // --------------------------------------------------------- metric cards
  PG.metricCards = function (c, t, selected) {
    return `<section class="metrics" aria-label="Eight physiological variables">${PG.NODES.map((n) => {
      const rec = PG.record(c, n.key, t);
      const av = PG.availability(c, n.key, t);
      const color = PG.SYSTEMS[n.system].color;
      const imputed = rec.s === "median" || rec.s === "unavailable";
      const pct = (k) => (av.total ? (av[k] / av.total) * 100 : 0);
      return `<button class="metric" data-select="${n.key}" aria-pressed="${selected === n.key}" style="--mc:${color}">
        <div class="metric-top">
          <span class="metric-name">${n.kind === "lab" ? '<i class="lab-glyph"></i>' : '<i class="sys"></i>'}${esc(n.label)}</span>
          <span class="metric-id">${n.itemid}</span>
        </div>
        <div class="metric-mid">
          <span class="metric-val${imputed ? " is-imputed" : ""}"><span class="v${rec.v === null ? " txt" : ""}">${rec.v === null ? (rec.s === "median" ? "Median fill" : "—") : f.value(n.key, rec.v)}</span>${rec.v === null ? "" : `<span class="u">${esc(n.unitShort)}</span>`}</span>
          ${PG.sparkline(PG.windowRecords(c, n.key, t), { color })}
        </div>
        <div class="avail" title="${av.observed} observed · ${av.carried} carried · ${av.median} median of ${av.total} h"><i class="observed" style="width:${pct("observed")}%"></i><i class="carried" style="width:${pct("carried")}%"></i><i class="median" style="width:${pct("median")}%"></i></div>
        <div class="metric-foot"><span class="st ${rec.s}">${PG.STATUS[rec.s].label}</span><span class="when">${esc(PG.shortWhen(c, rec, t))} · ${av.observed}/${av.total} obs</span></div>
      </button>`;
    }).join("")}</section>`;
  };

  // ------------------------------------------------------------ risk block
  PG.riskBlock = function (c, t, opts = {}) {
    const p = PG.prediction(c, t);
    const th = PG.thresholdInfo(c);
    const prof = PG.profile(c);
    const big = opts.large ? "" : " sm";
    if (!p) {
      const reason = !c.predictions ? "This case file contains no scores." :
        t < PG.MODEL.windowH - 1 ? "No score before a complete 24-hour window is available." : "No score was exported for this hour.";
      return `<div class="risk-empty"><div class="risk-score"><span class="v${big}" style="color:var(--faint)">—</span></div><p class="hint" style="margin-top:8px">${esc(reason)}</p></div>`;
    }
    const above = p.score >= th.value;
    const ticks = prof.thresholds.map((x) => `<div class="scale-th${x.value === th.value ? "" : " secondary"}" style="left:${x.value * 100}%"><span>${x.label}</span></div>`).join("");
    return `<div class="risk-score"><span class="v${big}">${p.score.toFixed(3)}</span><span class="hint">model score · uncalibrated</span></div>
      <div class="scale" role="img" aria-label="Score ${p.score.toFixed(3)} on a 0 to 1 scale; operating threshold ${th.label}">
        <div class="scale-track"></div>${ticks}
        <div class="scale-pt${above ? " above" : ""}" style="left:${p.score * 100}%"></div>
      </div>
      <div class="scale-ends"><span>0</span><span>1</span></div>
      <div class="class-line">
        ${above ? `<span class="tag risk">At or above operating threshold ${th.label}</span>` : `<span class="tag">Below operating threshold ${th.label}</span>`}
        ${PG.scoreSourceTag(c)}
      </div>
      ${PG.state.settings.showRecalibrated ? `<p class="hint" style="margin-top:10px">Prior-shift recalibrated estimate at the full-population prevalence (${(prof.prevalence * 100).toFixed(3)}%): <strong class="num">${PG.recalibrate(p.score, c).toFixed(4)}</strong></p>` : ""}`;
  };

  PG.mountRiskHistory = function (el, c, t, opts = {}) {
    if (!el || !c.predictions || !c.predictions.length) {
      if (el) el.innerHTML = '<p class="hint">No score history in this case file.</p>';
      return;
    }
    const th = PG.thresholdInfo(c);
    const preds = c.predictions;
    const xs = preds.map((p) => p.hour);
    PG.chart(el, {
      height: opts.height || 150,
      margin: opts.margin || { t: 16, r: 12, b: 26, l: 38 },
      x: { domain: [Math.min(...xs), Math.max(...xs)], fmt: (h) => "h" + c.hours[h] },
      y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2) },
      hlines: PG.profile(c).thresholds.map((x) => ({ y: x.value, label: x.label, cls: x.value === th.value ? "hline" : "hline soft" })),
      vlines: [{ x: t, cls: "vline" }],
      series: [{
        records: preds.map((p) => ({ h: p.hour, v: p.score, cls: p.hour === t ? "sel" : "", r: p.hour === t ? 4.5 : 2 })),
        color: "var(--accent)", area: true, points: true,
      }],
      snap: xs,
      tooltip: (h) => {
        const p = PG.prediction(c, h);
        if (!p) return null;
        return `<div class="t">${f.hour(c, h)}</div><div class="r"><span>Score</span><b>${p.score.toFixed(3)}</b></div>` +
          (PG.state.settings.showRecalibrated ? `<div class="r"><span>Recalibrated</span><b>${PG.recalibrate(p.score, c).toFixed(4)}</b></div>` : "") +
          `<div class="r"><span>vs ${th.label}</span><b>${p.score >= th.value ? "at/above" : "below"}</b></div>`;
      },
      aria: "Score history",
    });
    if (opts.onPick) {
      el.querySelector(".hit").addEventListener("click", (ev) => {
        const svg = el.querySelector("svg"), r = svg.getBoundingClientRect();
        const W = svg.viewBox.baseVal.width, m = (opts.margin || { l: 38, r: 12 });
        const px = ((ev.clientX - r.left) / r.width) * W;
        const x0 = Math.min(...xs), x1 = Math.max(...xs);
        const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
        const near = xs.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a), xs[0]);
        opts.onPick(near);
      });
    }
  };

  // ---------------------------------------------------------------- matrix
  PG.statusMatrix = function (c, from, to, opts = {}) {
    const n = to - from + 1;
    const w = opts.window;
    let s = `<div class="matrix" role="table" aria-label="Observation status by hour">`;
    PG.NODES.forEach((node) => {
      s += `<span class="rl" data-select="${node.key}" role="rowheader" title="${esc(node.label)}">${esc(node.label)}</span><div class="cells" role="row" style="grid-template-columns:repeat(${n},minmax(0,1fr))">`;
      for (let h = from; h <= to; h++) {
        const r = PG.record(c, node.key, h);
        const out = w && (h < w.start || h > w.end);
        s += `<i class="cell ${r.s}${out ? " out" : ""}" title="${esc(node.short)} · ${f.hour(c, h)} · ${PG.STATUS[r.s].label}${r.v !== null ? " · " + f.value(node.key, r.v) : ""}"></i>`;
      }
      s += `</div>`;
    });
    s += `<div class="axis"><span>h ${c.hours[from]}</span><span>h ${c.hours[Math.round((from + to) / 2)]}</span><span>h ${c.hours[to]}</span></div></div>`;
    return s;
  };

  PG.statusLegend = () => `<div class="legend">
      <span class="st observed">Observed</span><span class="st carried">Carried forward</span><span class="st median">Training-median fill</span><span><i class="lab-glyph"></i>Laboratory value</span>
    </div>`;
})();
