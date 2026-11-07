/**
 * dashboard/js/pages/analysis.js
 *
 * Risk Prediction, Graph Explainability, Temporal Analysis.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);
  const f = PG.fmt;
  const R = PG.RESULTS, X = PG.EXPLAIN;

  PG.PAGES = PG.PAGES || {};

  // ------------------------------------------------------------ helpers
  function horizonSvg(c, t) {
    const W = 680, H = 118, l = 18, r = 18;
    const h0 = c.hours[t] - 24, h1 = c.hours[t] + 9;
    const x = (h) => l + ((h - h0) / (h1 - h0)) * (W - l - r);
    const T = c.hours[t], hz = PG.MODEL.horizon;
    const y = 40, bh = 26;
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Observation window and prediction horizon">
      <line class="hz-axis" x1="${l}" x2="${W - r}" y1="${y + bh + 10}" y2="${y + bh + 10}"/>
      <rect class="hz-win" x="${x(T - 23.5)}" y="${y}" width="${x(T + 0.5) - x(T - 23.5)}" height="${bh}" rx="4"/>
      <text class="hz-text strong" x="${(x(T - 23.5) + x(T + 0.5)) / 2}" y="${y + 17}" text-anchor="middle">24-hour observation window</text>
      <path class="hz-gap" d="M${x(T + 0.5)},${y + bh / 2}H${x(T + hz.from)}"/>
      <text class="hz-text muted" x="${(x(T + 0.5) + x(T + hz.from)) / 2}" y="${y + bh / 2 - 5}" text-anchor="middle">6 h</text>
      <rect class="hz-target" x="${x(T + hz.from)}" y="${y}" width="${x(T + hz.to) - x(T + hz.from)}" height="${bh}" rx="4"/>
      <text class="hz-text" x="${x(T + hz.to)}" y="${y - 6}" text-anchor="end">Event interval</text>
      <line class="hz-now" x1="${x(T + 0.5)}" x2="${x(T + 0.5)}" y1="${y - 14}" y2="${y + bh + 14}"/>
      <text class="hz-text strong" x="${x(T + 0.5) - 4}" y="${y - 18}" text-anchor="end">Prediction</text>
      ${[[T - 23, "h " + (T - 23)], [T, "h " + T], [T + hz.from, "h " + (T + hz.from)], [T + hz.to, "h " + (T + hz.to)]]
        .map(([h, s], i) => `<text class="hz-text muted" x="${x(i === 0 ? h - 0.5 : i === 1 ? h + 0.5 : h)}" y="${y + bh + 26}" text-anchor="${i === 0 ? "start" : i === 3 ? "start" : "middle"}">${s}</text>`).join("")}
    </svg>`;
  }

  /** Nodes × scored hours, cell opacity ∝ that hour's gradient share. */
  function attributionMatrix(c, t) {
    const hs = (c.predictions || []).map((p) => p.hour).filter((h) => PG.attributionAt(c, h));
    if (!hs.length) {
      return `<p class="hint">${c.predictions && c.predictions.length === 0 ? "This stay is shorter than 24 hours, so it has no scores and no attribution." : "No per-hour attribution in this case file."}</p>`;
    }
    let max = 0;
    hs.forEach((h) => PG.NODES.forEach((n) => (max = Math.max(max, c.attribution[h][n.key]))));
    let s = `<div class="matrix attr-matrix" role="table" aria-label="Gradient attribution share by hour">`;
    PG.NODES.forEach((n) => {
      s += `<span class="rl" role="rowheader">${esc(n.label)}</span><div class="cells" role="row" style="grid-template-columns:repeat(${hs.length},minmax(0,1fr))">`;
      hs.forEach((h) => {
        const v = c.attribution[h][n.key];
        s += `<i class="cell${h === t ? " now" : ""}" data-attr-h="${h}" style="background:${PG.SYSTEMS[n.system].color};opacity:${(0.06 + 0.94 * (v / max)).toFixed(3)}" title="${esc(n.short)} · ${f.hour(c, h)} · ${(v * 100).toFixed(1)}%"></i>`;
      });
      s += `</div>`;
    });
    const mid = hs[Math.floor(hs.length / 2)];
    s += `<div class="axis"><span>h ${c.hours[hs[0]]}</span><span>h ${c.hours[mid]}</span><span>h ${c.hours[hs[hs.length - 1]]}</span></div></div>`;
    const at = PG.attributionAt(c, t);
    if (at) {
      const top = PG.NODES.slice().sort((a, b) => at[b.key] - at[a.key]).slice(0, 3);
      s += `<p class="hint" style="margin-top:10px">At ${f.hour(c, t)} the largest shares are ${top.map((n) => `<strong>${esc(n.label)}</strong> ${(at[n.key] * 100).toFixed(1)}%${PG.record(c, n.key, t).s === "median" ? " (not observed: imputed input)" : ""}`).join(", ")}. Shares describe what the score is sensitive to, not clinical causes; HR and SBP nodes carry the shock-index blend.</p>`;
    }
    return s;
  }

  // -------------------------------------------------------- risk prediction
  PG.PAGES.risk = {
    title: "Risk Prediction",
    crumb: "8-hour deterioration score",
    caseBar: true,
    render(root) {
      const c = PG.currentCase(), t = PG.state.t;
      const th = PG.thresholdInfo(c);
      const prof = PG.profile(c);
      const pre = c.score_profile === "pre-correction";
      // Cohort operating statistics belong to the corrected run only.
      const op = pre ? null : R.windowOps.find((o) => o.th === th.value);
      const stayOp = pre ? null : R.stay.ops.find((o) => o.th === th.value);
      const nbBal = PG.NOTEBOOK_RUN.balanced[0];
      const offset = Math.log(prof.prevalence / (1 - prof.prevalence));
      // Share of all loaded scored hours from the same checkpoint at/above the threshold.
      const peers = PG.cases.filter((x) => x.score_profile === c.score_profile && x.predictions);
      const peerScores = peers.flatMap((x) => x.predictions.map((p) => p.score));
      const peerAbove = peerScores.length ? peerScores.filter((v) => v >= th.value).length / peerScores.length : null;
      root.innerHTML = `
        <section class="grid-main-side">
          <div class="card">
            <div class="card-head">
              <div><h2>Deterioration score at ${f.hour(c, t)}</h2><div class="sub">Primary target: deterioration event in (t + 6, t + 8] h · inputs: preceding 24 h</div></div>
            </div>
            <div class="card-body">
              <div style="max-width:560px">${PG.riskBlock(c, t, { large: true })}</div>
              <hr class="divider">
              <div class="eyebrow">Observation window and prediction horizon</div>
              <div class="horizon">${horizonSvg(c, t)}</div>
              <p class="hint">A label at hour t denotes an event in (t + 6, t + 8]. The score uses only hours up to t.</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Reading this score</h3></div></div>
            <div class="card-body">
              <p class="hint"><strong>Not a probability at ICU prevalence.</strong> The model was trained on a balanced 1:1 cohort, so raw scores overstate absolute risk. The prior-shift recalibration (offset ${offset.toFixed(4)}) maps scores to the ${(prof.prevalence * 100).toFixed(3)}% full-population base rate.</p>
              <hr class="divider">
              <p class="hint"><strong>Scored by: ${esc(prof.label)}.</strong> ${pre ? "The only checkpoint in the repository, recorded in PhysioGraph_Final_Colab.ipynb. It was trained on the pre-correction data, and here it receives past-only inputs, which differ from its training encoding. These scores are not validated." : "The corrected run described in the briefing."}</p>
              <hr class="divider">
              <p class="hint"><strong>Operating threshold ${th.label}.</strong> ${esc(th.note)}</p>
              ${pre && th.value === 0.5 ? `<dl class="kv" style="margin-top:10px">
                <dt>Balanced-test sensitivity</dt><dd>${nbBal.sens.toFixed(4)}</dd>
                <dt>Balanced-test specificity</dt><dd>${nbBal.spec.toFixed(4)}</dd>
              </dl>
              <p class="hint" style="margin-top:8px">As recorded by the notebook for this checkpoint on its balanced test cohort (50% prevalence, leaked pre-correction data). Not transferable to these stays.</p>` : ""}
              ${pre && peerAbove !== null ? `<p class="hint" style="margin-top:8px"><strong>Context:</strong> ${(peerAbove * 100).toFixed(1)}% of all ${f.int(peerScores.length)} scored hours across ${peers.length} loaded stays are at or above ${th.label}. The checkpoint’s own training encoding gives nearly the same share, so this reflects the checkpoint (balanced training, uncalibrated), not the past-only inputs. A high score here is not a meaningful alert.</p>` : ""}
              ${op ? `<dl class="kv" style="margin-top:10px">
                <dt>Window sensitivity</dt><dd>${op.sens.toFixed(3)}</dd>
                <dt>Window specificity</dt><dd>${op.spec.toFixed(3)}</dd>
                <dt>Window PPV</dt><dd>${op.ppv.toFixed(3)}</dd>
                <dt>Hours flagged</dt><dd>${op.flagged.toFixed(2)}%</dd>
                <dt>False alerts per true alert</dt><dd>${op.fa.toFixed(1)}</dd>
              </dl>
              <p class="hint" style="margin-top:8px">Full test population, window level, N = ${f.int(R.fullPopulation.n)} (prevalence 1.304%).</p>` : ""}
              ${stayOp ? `<p class="hint" style="margin-top:8px">Stay level at ${th.label}: ${stayOp.alerted}% of stays alerted, sensitivity ${stayOp.sens.toFixed(3)}, PPV ${stayOp.ppv.toFixed(3)} <span class="tag unverified">test sweep</span></p>` : ""}
              <hr class="divider">
              <p class="hint"><strong>Model-generated score, not a diagnosis.</strong> PhysioGraph is a retrospective research model and has not been clinically validated.</p>
            </div>
          </div>
        </section>

        <div class="card">
          <div class="card-head">
            <div><h3>Score history</h3><div class="sub">One score per hour once a 24-hour window exists. Select a point to move the prediction hour.</div></div>
            <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Score</span><span>— — Operating threshold</span></div>
          </div>
          <div class="card-body"><div id="history"></div></div>
        </div>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>How the score is produced</h3><div class="sub">${esc(PG.MODEL.input)} · channels: ${PG.MODEL.channels.map(esc).join(", ")}</div></div></div>
            <div class="card-body">
              <div class="arch">${PG.MODEL.architecture.map((a, i) => `<div class="arch-step"><span class="arch-dot">${i + 1}</span><div><div class="n">${esc(a.name)}</div><div class="d">${esc(a.detail)}</div></div></div>`).join("")}</div>
              <hr class="divider">
              <p class="hint"><strong>Inference blend.</strong> <span class="mono">${esc(PG.MODEL.blendFormula)}</span> with weights fixed a priori. Every reported metric scores this blend against the 8-hour label. Ablation: the 8-hour head alone gives the same AUROC (${R.ablations.multiHorizon.h8Only}) as the blend.</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Alerting behaviour in the test cohort</h3><div class="sub">At threshold 0.9059 · full population</div></div></div>
            <div class="card-body flush">
              <div class="stat-row">
                <div class="stat"><div class="k">Raw alarms</div><div class="v">${f.int(R.suppression.rawAlarms)}</div><div class="s">${R.suppression.rawPct}% of hours</div></div>
                <div class="stat"><div class="k">Alert episodes</div><div class="v">${f.int(R.suppression.episodes)}</div><div class="s">${R.suppression.refractoryH} h refractory · −${R.suppression.reductionPct}%</div></div>
                <div class="stat"><div class="k">Episode PPV</div><div class="v">${R.suppression.episodePPV.toFixed(3)}</div><div class="s">per episode</div></div>
                <div class="stat"><div class="k">Stays alerted</div><div class="v">${R.leadTime.pct}%</div><div class="s">${R.leadTime.alerted} of ${f.int(R.leadTime.of)} deteriorating</div></div>
              </div>
            </div>
            <div class="card-body">
              <p class="hint">Median lead relative to the first labelled window: <strong>${R.leadTime.medianH} h</strong> (IQR ${R.leadTime.iqr[0]} to +${R.leadTime.iqr[1]}). Because a label at t marks an event in (t + 6, t + 8], this corresponds to roughly 6 hours before the event. Stays never alerted (34.4%) are excluded, which biases the distribution optimistically.</p>
            </div>
          </div>
        </section>`;

      PG.mountRiskHistory(root.querySelector("#history"), c, t, { height: 240, margin: { t: 18, r: 16, b: 28, l: 42 }, onPick: (h) => PG.setState({ t: h }) });
    },
  };

  // ------------------------------------------------------ graph explainability
  function graphDetail() {
    const st = PG.state, lens = st.graphLens;
    if (lens === "attention") {
      return `<div class="detail-head" style="--mc:var(--teal)"><div class="detail-kicker"><span class="tag">Model attention</span></div><h2>Attention is not attribution</h2>
        <div class="detail-site">Organ attention pooling weights γ and cross-organ multihead attention</div></div>
        <div class="detail-section"><p class="hint">In the in-distribution permutation test, removing the nodes ranked highest by attention <strong>raised</strong> predicted risk at every k, tracking the random ranking. Gradient ranking was the only one that lowered it.</p>
        <dl class="kv" style="margin-top:10px">${X.rankCorrelation.map((r) => `<dt>${esc(r.comparison)}</dt><dd>ρ ${r.rho >= 0 ? "+" : ""}${r.rho.toFixed(3)}</dd>`).join("")}</dl></div>
        <div class="detail-section"><p class="hint">Per-node attention weights are therefore not drawn as importance. The graph shows structure only under this lens.</p></div>`;
    }
    if (st.graphEdge !== null && st.graphEdge !== undefined) {
      const e = PG.EDGES[st.graphEdge], a = PG.NODE[e.from], b = PG.NODE[e.to];
      const rev = PG.EDGES.findIndex((x) => x.from === e.to && x.to === e.from);
      return `<div class="detail-head" style="--mc:${e.type === "reflex" ? "var(--teal)" : "var(--accent)"}">
          <div class="detail-kicker"><span class="tag">Edge ${st.graphEdge + 1} of 15</span><span class="tag ${e.type === "reflex" ? "ok" : "accent"}">${e.type === "reflex" ? "Reciprocal reflex arc" : "Forward cascade"}</span></div>
          <h2>${esc(a.label)} → ${esc(b.label)}</h2>
          <div class="detail-site">${esc(e.rationale)}</div>
        </div>
        <div class="detail-section"><dl class="kv"><dt>Source node</dt><dd>${a.index} · ${esc(a.key)}</dd><dt>Target node</dt><dd>${b.index} · ${esc(b.key)}</dd><dt>Reverse edge</dt><dd>${rev >= 0 ? "yes · " + esc(PG.EDGES[rev].rationale) : "none"}</dd></dl></div>
        <div class="detail-section"><p class="hint"><strong>Structural prior.</strong> Fixed before training to encode a hypothesised physiological pathway. The GAT learns how to pass messages along it; per-case edge attention coefficients are not exported.</p>
        <p class="hint" style="margin-top:8px">The model using an edge does not establish clinical causation.</p></div>
        ${e.type === "reflex" ? `<div class="detail-section"><p class="hint">One of the five reciprocal arcs added after the M2 design review (10 → 15 edges), a documented deviation.</p></div>` : ""}`;
    }
    if (st.graphNode) {
      const n = PG.NODE[st.graphNode];
      const ranked = PG.NODES.slice().sort((a, b) => b.gradShare - a.gradShare);
      const edgeRow = (e) => {
        const i = PG.EDGES.indexOf(e);
        return `<button class="edge-chip ${e.type}" data-edge-pick="${i}" style="cursor:pointer;text-align:left">${esc(PG.NODE[e.from].short)} → ${esc(PG.NODE[e.to].short)} · ${esc(e.rationale)}</button>`;
      };
      return `<div class="detail-head" style="--mc:${PG.SYSTEMS[n.system].color}">
          <div class="detail-kicker"><span class="tag">Node ${n.index}</span><span class="tag">${esc(PG.SYSTEMS[n.system].label)}</span><span class="tag mono">${n.itemid}</span></div>
          <h2>${esc(n.label)}</h2><div class="detail-site">${n.kind === "lab" ? "Laboratory value" : "Vital sign"} · ${esc(n.key)}</div>
        </div>
        <div class="detail-section"><h4><span>Incoming · what drives it</span><span>${PG.EDGES.filter((e) => e.to === n.key).length}</span></h4><div class="edge-chips">${PG.EDGES.filter((e) => e.to === n.key).map(edgeRow).join("") || '<span class="hint">None</span>'}</div></div>
        <div class="detail-section"><h4><span>Outgoing · what it drives</span><span>${PG.EDGES.filter((e) => e.from === n.key).length}</span></h4><div class="edge-chips">${PG.EDGES.filter((e) => e.from === n.key).map(edgeRow).join("") || '<span class="hint">None</span>'}</div></div>
        <div class="detail-section"><h4><span>Gradient attribution</span><span class="tag accent">cohort</span></h4>
          <div class="share-row"><div class="bar"><i style="width:${(n.gradShare / ranked[0].gradShare) * 100}%"></i></div><span class="num"><b>${(n.gradShare * 100).toFixed(1)}%</b> · rank ${ranked.indexOf(n) + 1}</span></div>
          ${n.caveat ? `<div class="caveat" style="margin-top:10px">${esc(n.caveat)}</div>` : ""}
        </div>`;
    }
    return `<div class="detail-head"><div class="detail-kicker"><span class="tag">Physiological graph</span></div><h2>8 nodes · 15 directed edges</h2>
        <div class="detail-site">Defined before training from physiology, shared by every window</div></div>
      <div class="detail-section"><dl class="kv"><dt>Forward cascade edges</dt><dd>${PG.EDGES.filter((e) => e.type === "cascade").length}</dd><dt>Reciprocal reflex arcs</dt><dd>${PG.EDGES.filter((e) => e.type === "reflex").length}</dd><dt>Most connected</dt><dd>${(() => { const deg = (k) => PG.EDGES.filter((e) => e.from === k || e.to === k).length; const top = PG.NODES.slice().sort((a, b) => deg(b.key) - deg(a.key))[0]; return esc(top.label) + " · " + deg(top.key) + " edges"; })()}</dd></dl></div>
      <div class="detail-section"><p class="hint">Select a node to see what drives it and what it drives, or select an edge for its physiological rationale. Switch to <strong>Attribution · cohort</strong> or <strong>Attribution · this stay</strong> to size nodes by the model’s gradient sensitivity.</p></div>`;
  }

  function faithfulnessSvg(el) {
    const rows = X.faithfulness.rows;
    const W = Math.max(300, el.clientWidth), H = 220, m = { t: 14, r: 10, b: 30, l: 50 };
    const lo = -0.04, hi = 0.025;
    const Y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
    const gw = (W - m.l - m.r) / rows.length, bw = Math.min(22, gw / 4.5);
    let s = "";
    [-0.04, -0.02, 0, 0.02].forEach((v) => {
      s += `<line class="fb-grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="fb-tick" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v.toFixed(2)}</text>`;
    });
    rows.forEach((r, i) => {
      const cx = m.l + gw * i + gw / 2;
      ["gradient", "attention", "random"].forEach((k, j) => {
        const v = r[k], x = cx + (j - 1.5) * (bw + 4) + 2;
        const y = Math.min(Y(0), Y(v)), h = Math.abs(Y(v) - Y(0));
        s += `<rect class="fb-bar ${k}" x="${x}" y="${y}" width="${bw}" height="${Math.max(1, h)}" rx="2"><title>k = ${r.k} · ${k}: ${v > 0 ? "+" : ""}${v.toFixed(4)}</title></rect>`;
      });
      s += `<text class="fb-k" x="${cx}" y="${H - 10}" text-anchor="middle">k = ${r.k}</text>`;
    });
    s += `<line class="fb-zero" x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Change in mean predicted risk after permuting the top-k nodes by each ranking">${s}</svg>`;
  }

  PG.PAGES.graph = {
    title: "Graph Explainability",
    crumb: "Structure, attribution and attention",
    caseBar: false,
    render(root) {
      const st = PG.state, lens = st.graphLens;
      const c = PG.currentCase(), caseShares = PG.attributionAt(c, st.t);
      const cohortShares = Object.fromEntries(PG.NODES.map((n) => [n.key, n.gradShare]));
      const shares = lens === "attribution" ? cohortShares : lens === "case" ? caseShares : null;
      const ranked = PG.NODES.slice().sort((a, b) => b.gradShare - a.gradShare);
      const lensBtn = (k, label) => `<button data-lens="${k}" aria-pressed="${lens === k}">${label}</button>`;
      root.innerHTML = `
        <section class="grid-main-side">
          <div class="card">
            <div class="card-head">
              <div><h2>Physiological graph</h2><div class="sub">${lens === "attribution" ? "Node area ∝ plain-gradient sensitivity share · cohort, corrected run (256 positive test windows)" : lens === "case" ? (caseShares ? `Node area ∝ gradient share of the score for stay ${esc(c.case_id)} at ${f.hour(c, st.t)} · ${esc(c.checkpoint || "model")}` : `No score for stay ${esc(c.case_id)} at ${f.hour(c, st.t)}, so no attribution to draw`) : lens === "attention" ? "Structure only — attention weights are not shown as importance" : "Fixed prior structure used by the Graph Attention Network"}</div></div>
              <div class="seg" role="group" aria-label="Lens">${lensBtn("structure", "Structure")}${lensBtn("attribution", "Attribution · cohort")}${lensBtn("case", "Attribution · this stay")}${lensBtn("attention", "Model attention")}</div>
            </div>
            <div class="graph-stage" id="graphStage"></div>
            <div class="graph-legend">
              <span><i class="lg-line"></i>Forward cascade (10)</span>
              <span><i class="lg-line reflex"></i>Reciprocal reflex arc (5)</span>
              ${Object.values(PG.SYSTEMS).map((s) => `<span><i class="lg-sys" style="--mc:${s.color}"></i>${esc(s.label)}</span>`).join("")}
            </div>
          </div>
          <div class="card detail" id="graphDetail">${graphDetail()}</div>
        </section>

        <section class="grid-3 explain-cards">
          <div class="card"><div class="card-body">
            <h3><span class="k" style="background:var(--surface-3);color:var(--text-2)">S</span>Physiological structure</h3>
            <p class="hint" style="margin-top:8px">Which variables may exchange information. Set by the authors from physiology before training; it is an assumption, not a finding. Topology ablation is incomplete (1 of 4 variants), so no claim is made that this graph beats alternatives.</p>
          </div></div>
          <div class="card"><div class="card-body">
            <h3><span class="k" style="background:var(--accent-soft);color:var(--accent)">A</span>Model attribution</h3>
            <p class="hint" style="margin-top:8px">How sensitive the score is to each input: exact plain gradient ∂s/∂x of the blended logit. Validated for <em>faithfulness to the model</em> by in-distribution permutation — not for clinical correctness.</p>
          </div></div>
          <div class="card"><div class="card-body">
            <h3><span class="k" style="background:var(--teal-soft);color:var(--teal)">γ</span>Model attention</h3>
            <p class="hint" style="margin-top:8px">Internal pooling weights. They rank nodes no better than chance in the faithfulness test, so they are not used as explanation here.</p>
          </div></div>
        </section>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Gradient attribution ranking</h3><div class="sub">Normalised node sensitivity share · ${X.windows} positive test windows</div></div></div>
            <div class="card-body">
              <div class="ranklist">${ranked.map((n, i) => `
                <div class="rank${st.graphNode === n.key ? " sel" : ""}" data-node-pick="${n.key}" style="--mc:${PG.SYSTEMS[n.system].color}" role="button" tabindex="0">
                  <span class="i">${i + 1}</span><span class="n"><i></i>${esc(n.label)}</span>
                  <div class="bar"><i style="width:${(n.gradShare / ranked[0].gradShare) * 100}%"></i></div><span class="p">${(n.gradShare * 100).toFixed(1)}%</span>
                </div>`).join("")}</div>
              <hr class="divider">
              <p class="hint">Diffuse distribution: Gini ${X.gini}, ${X.maxMinRatio}× between largest and smallest. The top four form the shock/hypoperfusion axis. ICU elapsed time carries ${X.time.sharePct}% of total sensitivity — the model predicts from physiology, not time in unit.</p>
              <p class="hint" style="margin-top:8px">Ranked by plain gradient. Grad×input is not used: it scales with |x| and inflates the 94%-imputed temperature node (${(X.gradXInputTemperature * 100).toFixed(1)}% vs ${(PG.NODE.temperature.gradShare * 100).toFixed(1)}%).</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head">
              <div><h3>Faithfulness · in-distribution permutation</h3><div class="sub">Change in mean predicted risk (baseline ${X.faithfulness.baseline.toFixed(4)}) after permuting the top-k nodes</div></div>
              <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Gradient</span><span><span class="sw" style="background:var(--teal);opacity:.75"></span>Attention</span><span><span class="sw" style="background:var(--faint)"></span>Random</span></div>
            </div>
            <div class="card-body">
              <div class="fbars" id="fbars"></div>
              <p class="hint" style="margin-top:8px">A node’s 24-hour trace is swapped with the same node’s trace from another window, preserving its distribution and breaking only the patient link. Only the gradient ranking lowers risk at every k. k = 1–2 is the cleanest estimate; the random column’s rise measures off-pattern injection.</p>
            </div>
          </div>
        </section>`;

      PG.renderGraph(root.querySelector("#graphStage"), {
        shares,
        node: st.graphNode, edge: st.graphEdge,
        onNode: (k) => PG.setState({ graphNode: st.graphNode === k ? null : k, graphEdge: null }),
        onEdge: (i) => PG.setState({ graphEdge: st.graphEdge === i ? null : i, graphNode: null }),
      });
      faithfulnessSvg(root.querySelector("#fbars"));
      root.querySelectorAll("[data-lens]").forEach((b) => b.addEventListener("click", () => PG.setState({ graphLens: b.dataset.lens })));
      root.querySelectorAll("[data-node-pick]").forEach((b) => {
        b.addEventListener("click", () => PG.setState({ graphNode: b.dataset.nodePick, graphEdge: null }));
        b.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); b.click(); } });
      });
      root.querySelectorAll("[data-edge-pick]").forEach((b) => b.addEventListener("click", () => PG.setState({ graphEdge: +b.dataset.edgePick, graphNode: null })));
    },
  };

  // --------------------------------------------------------- temporal analysis
  PG.PAGES.temporal = {
    title: "Temporal Analysis",
    crumb: "Trajectories across the stay",
    caseBar: true,
    render(root) {
      const c = PG.currentCase(), t = PG.state.t, key = PG.state.temporalVar;
      const n = PG.NODE[key], H = c.hours.length, w = PG.windowRange(c, t);
      const av = PG.availability(c, key, t);
      root.innerHTML = `
        <div class="card">
          <div class="card-head">
            <div><h2>${esc(n.label)} <span class="hint" style="font-weight:400">· ${esc(n.unit)}</span></h2><div class="sub">Full stay, h ${c.hours[0]}–${c.hours[H - 1]} · shaded band is the 24-hour observation window ending at ${f.hour(c, t)}</div></div>
            ${PG.statusLegend()}
          </div>
          <div class="card-body">
            <div class="var-chips" role="group" aria-label="Variable">${PG.NODES.map((x) => `<button class="var-chip" data-var="${x.key}" aria-pressed="${x.key === key}" style="--mc:${PG.SYSTEMS[x.system].color}"><i></i>${esc(x.label)}</button>`).join("")}</div>
            <div id="mainChart" style="margin-top:14px"></div>
            <div class="hint" style="margin-top:4px">In window: ${av.observed} observed · ${av.carried} carried forward · ${av.median} training-median hours. Carried values are dashed; median-filled hours are gaps with a baseline tick.</div>
            <hr class="divider">
            <div class="eyebrow" style="margin-bottom:6px">Deterioration score · same time axis ${PG.scoreSourceTag(c)}</div>
            <div id="riskAligned"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-head"><div><h3>All variables · observation window</h3><div class="sub">h ${c.hours[w.start]}–${c.hours[w.end]} · select one to focus the chart above</div></div></div>
          <div class="minis">${PG.NODES.map((x) => {
            const rec = PG.record(c, x.key, t);
            return `<button class="mini" data-var="${x.key}" aria-pressed="${x.key === key}" style="--mc:${PG.SYSTEMS[x.system].color}">
              <span class="metric-name">${x.kind === "lab" ? '<i class="lab-glyph"></i>' : '<i class="sys"></i>'}${esc(x.label)}</span>
              <div class="mini-chart" data-mini="${x.key}"></div>
              <div class="metric-foot"><span class="st ${rec.s}">${PG.STATUS[rec.s].label}</span><span class="num">${rec.v === null ? "—" : f.value(x.key, rec.v) + " " + esc(x.unitShort)}</span></div>
            </button>`;
          }).join("")}</div>
        </div>

        <div class="card">
          <div class="card-head"><div><h3>Observation status across the stay</h3><div class="sub">Hours outside the current window are faded</div></div></div>
          <div class="card-body">${PG.statusMatrix(c, 0, H - 1, { window: w })}</div>
        </div>

        <div class="card">
          <div class="card-head"><div><h3>Gradient attribution over time</h3><div class="sub">Each node’s share of |∂score/∂input| at every scored hour (project method, run_attribution2.py) · ${esc(c.checkpoint || "model")}. Select a column to move the prediction hour.</div></div></div>
          <div class="card-body">${attributionMatrix(c, t)}</div>
        </div>`;

      const color = PG.SYSTEMS[n.system].color;
      PG.chart(root.querySelector("#mainChart"), {
        height: 260,
        margin: { t: 18, r: 16, b: 28, l: 46 },
        x: { domain: [0, H - 1], fmt: (h) => "h" + c.hours[h] },
        y: { label: n.unitShort, fmt: PG.tickFmt },
        vband: { from: w.start, to: w.end, label: "24 h window" },
        vlines: [{ x: t, cls: "vline" }],
        series: [{ records: c.hours.map((_, h) => Object.assign({ h }, PG.record(c, key, h))), color, status: true }],
        tooltip: PG.trendTooltip(c, key),
        aria: `${n.label} across the stay`,
      });

      const ra = root.querySelector("#riskAligned");
      if (c.predictions && c.predictions.length) {
        PG.chart(ra, {
          height: 120,
          margin: { t: 10, r: 16, b: 24, l: 46 },
          x: { domain: [0, H - 1], fmt: (h) => "h" + c.hours[h] },
          y: { domain: [0, 1], ticks: [0, 0.5, 1], fmt: (v) => v.toFixed(1) },
          hlines: [{ y: PG.thresholdInfo(c).value, cls: "hline" }],
          vband: { from: w.start, to: w.end },
          vlines: [{ x: t, cls: "vline" }],
          series: [{ records: c.predictions.map((p) => ({ h: p.hour, v: p.score })), color: "var(--accent)", area: true }],
          tooltip: (h) => { const p = PG.prediction(c, h); return p ? `<div class="t">${f.hour(c, h)}</div><div class="r"><span>Score</span><b>${p.score.toFixed(3)}</b></div>` : null; },
          aria: "Score history",
        });
      } else {
        ra.innerHTML = '<p class="hint">No scores in this case file.</p>';
      }

      root.querySelectorAll(".mini-chart").forEach((el) => {
        const x = PG.NODE[el.dataset.mini];
        PG.chart(el, {
          height: 74,
          margin: { t: 6, r: 4, b: 6, l: 30 },
          x: { domain: [w.start, w.end], ticks: [] },
          y: { fmt: PG.tickFmt, count: 2 },
          series: [{ records: PG.windowRecords(c, x.key, t), color: PG.SYSTEMS[x.system].color, status: true }],
          aria: x.label,
        });
      });
      root.querySelectorAll("[data-var]").forEach((b) => b.addEventListener("click", () => PG.setState({ temporalVar: b.dataset.var, selected: b.dataset.var })));
      root.querySelectorAll(".matrix [data-select]").forEach((b) => b.addEventListener("click", () => PG.setState({ temporalVar: b.dataset.select, selected: b.dataset.select })));
      root.querySelectorAll("[data-attr-h]").forEach((b) => b.addEventListener("click", () => PG.setState({ t: +b.dataset.attrH })));
    },
  };
})();
