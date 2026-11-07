/**
 * dashboard/js/pages/research.js
 *
 * Model Performance and Settings.
 * Only CURRENT results (briefing §5–§7) are shown; superseded figures are not
 * referenced anywhere in the interface.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});
  const esc = (s) => PG.esc(s);
  const f = PG.fmt;
  const R = PG.RESULTS;

  PG.PAGES = PG.PAGES || {};

  const n3 = (v) => (v === null || v === undefined ? "—" : v.toFixed(3));
  const thFmt = (v) => (v === 0.9059 ? "0.9059" : v.toFixed(2));
  const n4 = (v) => (v === null || v === undefined ? "—" : v.toFixed(4));

  function kpi(o) {
    return `<div class="card kpi">
      <div class="kpi-head"><div><div class="kpi-title">${o.title}</div><div class="kpi-unit">${o.unit}</div></div>${o.tag || ""}</div>
      <div class="kpi-vals">${o.vals.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}</div>
      <div class="prev">${o.prev}</div>
    </div>`;
  }

  PG.PAGES.performance = {
    title: "Model Performance",
    crumb: "Corrected pipeline results",
    caseBar: false,
    render(root) {
      const fp = R.fullPopulation, st = R.stay, cal = R.calibration;
      if (PG.state.perfRun === "notebook") return renderNotebookRun(root);
      const selTh = PG.SCORE_PROFILES.corrected.thresholds[PG.state.settings.thresholdIndex].value;
      const hb = (label, v, max, cls, ref) => `<div class="hbar"><span>${label}</span><div class="track"><i class="${cls || ""}" style="width:${(v / max) * 100}%"></i>${ref !== undefined ? `<span class="ref" style="left:${(ref / max) * 100}%" title="Non-informative null ${ref}"></span>` : ""}</div><span class="val">${n4(v)}</span></div>`;

      root.innerHTML = `
        ${runSwitch()}
        ${PG.notice("<strong>Read each metric against its own population.</strong> Window-level average precision at 1.304% prevalence and stay-level AP at 13.90% are not comparable to each other or to the balanced 50% cohort. Figures are the corrected forward-only results in PhysioGraph_PROJECT_BRIEFING.md §5–7. No executed notebook in the repository reproduces them; the recorded notebook run is shown separately.", "info")}

        <section class="kpi-grid">
          ${kpi({ title: "Balanced cohort", unit: "Protocol A · window level", vals: [["AUROC", "0.8225"], ["AP", "0.8389"]], prev: `Prevalence <b>50.00%</b> · N ${f.int(R.balanced.n)} windows` })}
          ${kpi({ title: "Full population", unit: "Protocol B · window level", vals: [["AUROC", n4(fp.auroc)], ["AP", n4(fp.ap)]], prev: `Prevalence <b>1.304%</b> · AP ${fp.lift}× base rate · N ${f.int(fp.n)}` })}
          ${kpi({ title: "Stay level", unit: "Max hourly score per stay", vals: [["AUROC", n4(st.auroc)], ["AP", n4(st.ap)]], prev: `Prevalence <b>13.90%</b> · AP 4.7× base rate · N ${f.int(st.n)} stays` })}
          ${kpi({ title: "Calibration", unit: "Full population · prior-shift recalibrated", vals: [["Brier", n4(cal.brier.recalibrated)], ["ECE", n4(cal.ece.recalibrated)]], prev: `Null-model Brier <b>${n4(cal.brier.nullModel)}</b> · uncalibrated ${n4(cal.brier.uncalibrated)}` })}
        </section>

        <div class="card">
          <div class="card-head"><div><h2>Matched comparison · balanced cohort</h2><div class="sub">Protocol A test, N = ${f.int(R.balanced.n)}, prevalence 50%, threshold 0.50 fixed a priori. Identical windows, labels and splits.</div></div></div>
          <div class="card-body flush table-wrap">
            <table class="data">
              <thead><tr><th>Model</th><th class="r">AUROC</th><th class="r">AP</th><th class="r">Accuracy</th><th class="r">F1</th><th class="r">Sensitivity</th><th class="r">Specificity</th></tr></thead>
              <tbody>${R.balanced.rows.map((r) => `
                <tr class="${r.ours ? "ours hl" : ""}">
                  <td>${esc(r.model)} ${r.unverified ? '<span class="tag unverified">unverified</span>' : ""}${r.note ? `<span class="sub">${esc(r.note)}</span>` : ""}</td>
                  <td class="r">${n4(r.auroc)}</td><td class="r">${n4(r.ap)}</td><td class="r">${n4(r.acc)}</td><td class="r">${n4(r.f1)}</td><td class="r">${n4(r.sens)}</td><td class="r">${n4(r.spec)}</td>
                </tr>`).join("")}</tbody>
            </table>
          </div>
          <div class="card-body"><p class="hint">Seed variance: two identical configurations differing only in seed scored ${R.ablations.seedSpread.a} and ${R.ablations.seedSpread.b} (spread ${R.ablations.seedSpread.spread}). Multi-seed confidence intervals are outstanding work. Dashes mark values that were not computed or were withheld.</p></div>
        </div>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Window-level operating points</h3><div class="sub">Full test population · prevalence 1.304%</div></div>
              <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Sensitivity</span><span><span class="sw" style="background:var(--teal)"></span>Specificity</span><span><span class="sw" style="background:var(--sys-cardio)"></span>PPV</span><span><span class="sw" style="background:var(--sys-systemic)"></span>F1</span></div>
            </div>
            <div class="card-body"><div id="opsChart"></div><p class="hint" style="margin-top:6px">0.9059 was selected on the validation set at its own prevalence. The vertical line marks the threshold chosen in Settings.</p></div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Calibration under prior shift</h3><div class="sub">Training prevalence 50% → test prevalence 1.304% · offset ${cal.offset}</div></div></div>
            <div class="card-body">
              <div class="hbar-group"><h4>Brier score · lower is better</h4>
                ${hb("Uncalibrated", cal.brier.uncalibrated, 0.2, "alt")}
                ${hb("Recalibrated", cal.brier.recalibrated, 0.2, "", cal.brier.nullModel)}
                ${hb("Null (base rate)", cal.brier.nullModel, 0.2, "alt")}
              </div>
              <div class="hbar-group"><h4>Expected calibration error</h4>
                ${hb("Uncalibrated", cal.ece.uncalibrated, 0.4, "alt")}
                ${hb("Recalibrated", cal.ece.recalibrated, 0.4, "teal")}
              </div>
              <p class="hint" style="margin-top:14px">The logit offset is a monotone map, so AUROC and AP are unchanged; only Brier, ECE and the meaning of a threshold change. Recalibrated Brier (0.0115) sits just below the non-informative null (0.0129). No reliability-diagram data was exported, so no calibration curve is drawn.</p>
            </div>
          </div>
        </section>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Window level</h3><div class="sub">N = ${f.int(fp.n)} windows · F1 at 0.9059 = ${fp.f1}</div></div></div>
            <div class="card-body flush table-wrap">
              <table class="data">
                <thead><tr><th>Threshold</th><th class="r">Sens.</th><th class="r">Spec.</th><th class="r">PPV</th><th class="r">F1</th><th class="r">Flagged</th><th class="r" title="False alerts per true alert">FA / TA</th></tr></thead>
                <tbody>${R.windowOps.map((o) => `<tr class="${o.th === selTh ? "hl" : ""}"><td class="mono">${thFmt(o.th)}${o.selected ? ' <span class="tag accent">validation-selected</span>' : ""}</td><td class="r">${n3(o.sens)}</td><td class="r">${n3(o.spec)}</td><td class="r">${n3(o.ppv)}</td><td class="r">${n3(o.f1)}</td><td class="r">${o.flagged.toFixed(2)}%</td><td class="r">${o.fa.toFixed(1)}</td></tr>`).join("")}</tbody>
              </table>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Stay level</h3><div class="sub">N = ${f.int(st.n)} stays · score = max hourly risk per stay</div></div><span class="tag unverified">test sweep · unverified</span></div>
            <div class="card-body flush table-wrap">
              <table class="data">
                <thead><tr><th>Threshold</th><th class="r">Stays alerted</th><th class="r">Sensitivity</th><th class="r">PPV</th></tr></thead>
                <tbody>${st.ops.map((o) => `<tr class="${o.th === selTh ? "hl" : ""}"><td class="mono">${thFmt(o.th)}</td><td class="r">${o.alerted.toFixed(1)}%</td><td class="r">${n3(o.sens)}</td><td class="r">${n3(o.ppv)}</td></tr>`).join("")}</tbody>
              </table>
            </div>
            <div class="card-body"><p class="hint">These thresholds are a sweep on test, not a validation-selected operating point; no single row should be reported as “the” operating point.</p></div>
          </div>
        </section>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Validation trace</h3><div class="sub">Balanced validation cohort · selection on validation AP</div></div>
              <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Val AP</span><span><span class="sw" style="background:var(--teal)"></span>Val AUROC</span></div>
            </div>
            <div class="card-body"><div id="traceChart"></div><p class="hint" style="margin-top:6px">Selected checkpoint: epoch ${PG.MODEL.selectedEpoch}, validation AP ${PG.MODEL.selectedValAP}. Early stopping fired at epoch 10.</p></div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Ablations and open items</h3></div></div>
            <div class="card-body">
              <dl class="kv">
                <dt>Multi-horizon heads</dt><dd>H8 alone ${R.ablations.multiHorizon.h8Only} = blend ${R.ablations.multiHorizon.blend} · null result</dd>
                <dt>Topology · clinical-15</dt><dd>AUROC ${R.ablations.topology.clinical15.auroc} · AP ${R.ablations.topology.clinical15.ap}</dd>
                <dt>Topology · other variants</dt><dd>${R.ablations.topology.remaining} of 4 not run</dd>
                <dt>Seed spread</dt><dd>${R.ablations.seedSpread.spread} AUROC</dd>
              </dl>
              <hr class="divider">
              <p class="hint"><strong>Not established:</strong> that the physiological graph outperforms alternative topologies; patient-level independence of splits (no <span class="mono">subject_id</span> join yet); multi-seed confidence intervals; MEWS/NEWS/qSOFA on corrected data; sequence-model baselines; external validation.</p>
            </div>
          </div>
        </section>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Evaluation cohorts</h3><div class="sub">Protocol A test is a strict subset of Protocol B test with identical positives</div></div></div>
            <div class="card-body flush table-wrap">
              <table class="data">
                <thead><tr><th>Protocol</th><th>Split</th><th class="r">Positive</th><th class="r">Negative</th><th class="r">Prevalence</th></tr></thead>
                <tbody>${R.cohorts.map((r) => `<tr><td>${esc(r.protocol)}</td><td>${r.split}${r.unit ? " · stays" : ""}</td><td class="r">${f.int(r.pos)}</td><td class="r">${f.int(r.neg)}</td><td class="r">${(r.prev * 100).toFixed(r.prev < 0.1 ? 3 : 2)}%</td></tr>`).join("")}</tbody>
              </table>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Result provenance</h3></div></div>
            <div class="card-body">
              <p class="hint">All figures on this page are from <span class="mono">PhysioGraph_PROJECT_BRIEFING.md</span> §5–§7 (corrected forward-only pipeline, 30 Oct 2026). Items marked unverified are labelled where they appear.</p>
              <hr class="divider">
              <p class="hint"><strong>Checkpoint in the repository differs from the documented one.</strong> <span class="mono">${esc(PG.PROVENANCE.inRepository.file)}</span> records epoch ${PG.PROVENANCE.inRepository.epoch} and validation AP ${PG.PROVENANCE.inRepository.valAP}; the corrected run selected epoch ${PG.PROVENANCE.documented.epoch} (AP ${PG.PROVENANCE.documented.valAP}). This interface never presents that file as the source of a score.</p>
              <hr class="divider">
              <p class="hint"><strong>Superseded files not used:</strong> ${PG.PROVENANCE.supersededFiles.map((x) => `<span class="mono">${esc(x)}</span>`).join(", ")}.</p>
            </div>
          </div>
        </section>`;

      // operating points
      const ops = R.windowOps;
      const series = [["sens", "var(--accent)"], ["spec", "var(--teal)"], ["ppv", "var(--sys-cardio)"], ["f1", "var(--sys-systemic)"]];
      PG.chart(root.querySelector("#opsChart"), {
        height: 240,
        margin: { t: 16, r: 16, b: 30, l: 40 },
        x: { domain: [0.48, 0.97], ticks: [0.5, 0.6, 0.7, 0.8, 0.9], fmt: (v) => v.toFixed(1), label: "threshold" },
        y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2) },
        vlines: [{ x: selTh, cls: "vline" }],
        series: series.map(([k, color]) => ({ records: ops.map((o) => ({ h: o.th, v: o[k], r: 2.6 })), color, points: true })),
        snap: ops.map((o) => o.th),
        tooltip: (x) => {
          const o = ops.find((q) => q.th === x);
          return `<div class="t">Threshold ${o.th}</div>` + series.map(([k]) => `<div class="r"><span>${{ sens: "Sensitivity", spec: "Specificity", ppv: "PPV", f1: "F1" }[k]}</span><b>${o[k].toFixed(3)}</b></div>`).join("") +
            `<div class="r"><span>Hours flagged</span><b>${o.flagged}%</b></div>`;
        },
        aria: "Window-level operating points",
      });

      bindRunSwitch(root);
      const tr = R.validationTrace;
      PG.chart(root.querySelector("#traceChart"), {
        height: 220,
        margin: { t: 16, r: 16, b: 30, l: 46 },
        x: { domain: [1, 10], ticks: tr.map((r) => r.epoch), label: "epoch" },
        y: { domain: [0.79, 0.85], ticks: [0.79, 0.81, 0.83, 0.85], fmt: (v) => v.toFixed(2) },
        vlines: [{ x: PG.MODEL.selectedEpoch, cls: "vline", label: "selected" }],
        series: [
          { records: tr.map((r) => ({ h: r.epoch, v: r.ap, r: r.best ? 4.5 : 2.6 })), color: "var(--accent)", points: true },
          { records: tr.map((r) => ({ h: r.epoch, v: r.auroc, r: r.best ? 4.5 : 2.6 })), color: "var(--teal)", points: true },
        ],
        snap: tr.map((r) => r.epoch),
        tooltip: (e) => {
          const r = tr.find((q) => q.epoch === e);
          return `<div class="t">Epoch ${e}${r.best ? " · selected" : ""}</div><div class="r"><span>Val AP</span><b>${r.ap.toFixed(4)}</b></div><div class="r"><span>Val AUROC</span><b>${r.auroc.toFixed(4)}</b></div><div class="r"><span>Train loss</span><b>${r.loss.toFixed(4)}</b></div><div class="r"><span>Val accuracy</span><b>${r.acc.toFixed(4)}</b></div>`;
        },
        aria: "Validation trace",
      });
    },
  };

  function runSwitch() {
    const r = PG.state.perfRun === "notebook" ? "notebook" : "corrected";
    return `<div class="run-switch"><div class="seg" role="group" aria-label="Training run">
      <button data-run="corrected" aria-pressed="${r === "corrected"}">Corrected run · briefing</button>
      <button data-run="notebook" aria-pressed="${r === "notebook"}">Pre-correction run · notebook</button>
    </div><span class="hint">${r === "corrected" ? "Primary results (epoch 6, corrected data)." : "What PhysioGraph_Final_Colab.ipynb recorded (epoch 8). The checkpoint in models/ is from this run."}</span></div>`;
  }

  function bindRunSwitch(root) {
    root.querySelectorAll("[data-run]").forEach((b) => b.addEventListener("click", () => PG.setState({ perfRun: b.dataset.run })));
  }

  function renderNotebookRun(root) {
    const NB = PG.NOTEBOOK_RUN, fp = NB.fullPopulation;
    const best = NB.trace.find((r) => r.best);
    const nTest = NB.cohorts[2].pos * 2;
    root.innerHTML = `
      ${runSwitch()}
      ${PG.notice(`<strong>Superseded results, shown for transparency.</strong> This is the run recorded in <span class="mono">${esc(NB.file)}</span>. It was trained and evaluated on the pre-correction data: bidirectional interpolation that leaks future values, IQR capping (lactate capped at 3.0 mmol/L) and missing values set to 0 before scaling. Its shock index also used z-score constants on MinMax data. The project briefing (§9) says these figures must not be quoted as results.`)}

      <section class="kpi-grid">
        ${kpi({ title: "Balanced test", unit: "Window level · threshold 0.50", vals: [["AUROC", n4(NB.balanced[0].auroc)], ["AP", n4(NB.balanced[0].ap)]], prev: `Prevalence <b>50.00%</b> · N ${f.int(nTest)} windows`, tag: '<span class="tag unverified">leaked data</span>' })}
        ${kpi({ title: "Full population", unit: "Window level", vals: [["AUROC", n4(fp.auroc)], ["AP", n4(fp.ap)]], prev: `Prevalence <b>${(fp.prevalence * 100).toFixed(3)}%</b> · ${fp.lift}× base rate · N ${f.int(fp.n)}`, tag: '<span class="tag unverified">leaked data</span>' })}
        ${kpi({ title: "Brier · uncalibrated", unit: "Full population", vals: [["Brier", n4(fp.brier)], ["F1 @ 0.35", n4(fp.f1)]], prev: "Printed as “Calibrated!” by a literal string; 0.35 was an arbitrary threshold" })}
        ${kpi({ title: "Checkpoint", unit: "Saved by cell 8 · matches models/*.pt", vals: [["Epoch", String(best.epoch)], ["Val AP", n4(best.ap)]], prev: `Stored threshold <b>${PG.PROVENANCE.inRepository.storedThreshold}</b> (F1-optimal on balanced validation)` })}
      </section>

      <section class="grid-2">
        <div class="card">
          <div class="card-head"><div><h3>Balanced test cohort · as recorded</h3><div class="sub">N = ${f.int(nTest)} windows, prevalence 50%</div></div></div>
          <div class="card-body flush table-wrap">
            <table class="data">
              <thead><tr><th>Model</th><th class="r">AUROC</th><th class="r">AP</th><th class="r">Acc.</th><th class="r">F1</th><th class="r">Sens.</th><th class="r">Spec.</th></tr></thead>
              <tbody>${NB.balanced.map((r) => `<tr class="${r.ours ? "ours hl" : ""}"><td>${esc(r.model)}${r.note ? `<span class="sub">${esc(r.note)}</span>` : ""}</td><td class="r">${n4(r.auroc)}</td><td class="r">${n4(r.ap)}</td><td class="r">${n4(r.acc)}</td><td class="r">${n4(r.f1)}</td><td class="r">${n4(r.sens)}</td><td class="r">${n4(r.spec)}</td></tr>`).join("")}</tbody>
            </table>
          </div>
          <div class="card-body"><p class="hint">${esc(NB.excluded)}</p></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>Validation trace · as recorded</h3><div class="sub">Balanced validation · checkpoint saved at the best validation AP</div></div>
            <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>Val AP</span><span><span class="sw" style="background:var(--teal)"></span>Val AUROC</span></div>
          </div>
          <div class="card-body"><div id="nbTrace"></div></div>
        </div>
      </section>

      <section class="grid-2">
        <div class="card">
          <div class="card-head"><div><h3>Cohorts · as recorded</h3><div class="sub">1:1 balanced windows on the 8-hour label</div></div></div>
          <div class="card-body flush table-wrap">
            <table class="data"><thead><tr><th>Split</th><th class="r">Positive</th><th class="r">Negative</th></tr></thead>
              <tbody>${NB.cohorts.map((r) => `<tr><td>${r.split}</td><td class="r">${f.int(r.pos)}</td><td class="r">${f.int(r.neg)}</td></tr>`).join("")}
              <tr><td>test · full population</td><td class="r">${f.int(fp.positives)}</td><td class="r">${f.int(fp.n - fp.positives)}</td></tr></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <div class="card-head"><div><h3>How this run differs from the corrected run</h3></div></div>
          <div class="card-body">
            <dl class="kv" style="grid-template-columns:1fr auto">
              <dt>Missing-value fill</dt><dd>ffill(3) + interpolate both ways</dd>
              <dt>Outliers</dt><dd>Train IQR caps</dd>
              <dt>Residual gaps</dt><dd>0 before MinMax</dd>
              <dt>Shock-index constants</dt><dd>z-score (wrong for MinMax)</dd>
              <dt>Train / val / test positives</dt><dd>53,278 / 11,206 / 10,984</dd>
              <dt>Corrected run positives</dt><dd>53,498 / 11,245 / 11,027</dd>
            </dl>
            <hr class="divider">
            <p class="hint">Sources: data_exploration.ipynb cells 89–102 (pipeline), PhysioGraph_Final_Colab.ipynb cells 5 and 9–13 (encoding and results), briefing §3 and §9.</p>
          </div>
        </div>
      </section>`;
    bindRunSwitch(root);
    const tr = NB.trace;
    PG.chart(root.querySelector("#nbTrace"), {
      height: 220,
      margin: { t: 16, r: 16, b: 30, l: 46 },
      x: { domain: [1, 10], ticks: tr.map((r) => r.epoch), label: "epoch" },
      y: { domain: [0.79, 0.85], ticks: [0.79, 0.81, 0.83, 0.85], fmt: (v) => v.toFixed(2) },
      vlines: [{ x: best.epoch, cls: "vline", label: "saved" }],
      series: [
        { records: tr.map((r) => ({ h: r.epoch, v: r.ap, r: r.best ? 4.5 : 2.6 })), color: "var(--accent)", points: true },
        { records: tr.map((r) => ({ h: r.epoch, v: r.auroc, r: r.best ? 4.5 : 2.6 })), color: "var(--teal)", points: true },
      ],
      snap: tr.map((r) => r.epoch),
      tooltip: (e) => {
        const r = tr.find((q) => q.epoch === e);
        return `<div class="t">Epoch ${e}${r.best ? " · saved" : ""}</div><div class="r"><span>Val AP</span><b>${r.ap.toFixed(4)}</b></div><div class="r"><span>Val AUROC</span><b>${r.auroc.toFixed(4)}</b></div><div class="r"><span>Train loss</span><b>${r.loss.toFixed(4)}</b></div>`;
      },
      aria: "Notebook validation trace",
    });
  }

  // ---------------------------------------------------------------- settings
  PG.PAGES.settings = {
    title: "Settings",
    crumb: "Display and data integrity",
    caseBar: false,
    render(root) {
      const s = PG.state.settings;
      root.innerHTML = `
        <div class="card">
          <div class="card-head"><div><h2>Display</h2><div class="sub">Saved in this browser only</div></div></div>
          <div class="card-body">
            <div class="settings-row">
              <div><div class="t">Operating threshold</div><div class="d">Used to classify scores and mark charts. A threshold belongs to the run it was selected on, so each checkpoint offers only its own documented thresholds: the first or the second option of the list below.</div></div>
              <div class="radio-list">${[0, 1].map((i) => `<label><input type="radio" name="th" value="${i}" ${i === s.thresholdIndex ? "checked" : ""}><span><b>${i === 0 ? "Primary" : "Secondary"}</b><br>${Object.values(PG.SCORE_PROFILES).map((pr) => `<span class="hint">${esc(pr.label)}: <b>${pr.thresholds[i].label}</b> — ${esc(pr.thresholds[i].note)}</span>`).join("<br>")}</span></label>`).join("")}</div>
            </div>
            <div class="settings-row">
              <div><div class="t">Show recalibrated estimate</div><div class="d">Alongside each raw score, show the prior-shift recalibrated value at the full-population prevalence of the run that produced the score (1.304% corrected, 1.299% pre-correction).</div></div>
              <label class="switch"><input type="checkbox" id="setRecal" ${s.showRecalibrated ? "checked" : ""}><span></span></label>
            </div>
            <div class="settings-row">
              <div><div class="t">Reduce motion</div><div class="d">Stop the selected-marker pulse. The system reduced-motion preference is always respected.</div></div>
              <label class="switch"><input type="checkbox" id="setMotion" ${s.reduceMotion ? "checked" : ""}><span></span></label>
            </div>
          </div>
        </div>

        <section class="grid-2">
          <div class="card">
            <div class="card-head"><div><h3>Data integrity</h3></div></div>
            <div class="card-body">
              <dl class="kv" style="grid-template-columns:1fr auto">
                <dt>Data</dt><dd>MIMIC-IV v3.1 · retrospective, de-identified</dd>
                <dt>Live monitoring</dt><dd>None</dd>
                <dt>Clinical validation</dt><dd>None</dd>
                <dt>Imputation</dt><dd>Forward-only · no backward fill</dd>
                <dt>Stays in session</dt><dd>${PG.cases.length} real (${PG.cases.filter(PG.isDemo).length} from MIMIC-IV Demo)</dd>
              </dl>
              <hr class="divider">
              <p class="hint">No synthetic or generated values are shown. Stays come from the open MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL v1.0; Johnson et al., 2023), processed by <span class="mono">scripts/build_demo_cases.py</span>. Exported full-MIMIC case files are held in memory for this tab only.</p>
            </div>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>About</h3></div></div>
            <div class="card-body">
              <p><b>${esc(PG.MODEL.name)}</b> — ${esc(PG.MODEL.title)}</p>
              <p class="hint" style="margin-top:6px">Spatio-temporal graph network (GAT, cross-organ attention, causal TCN, attention pooling, multi-horizon heads) for detecting ICU deterioration 6–8 hours ahead from 24 hours of eight physiological variables.</p>
              <hr class="divider">
              <p class="hint">Phase 2 case-review interface, version ${PG.VERSION}. A research prototype: not a medical device, not clinically validated, and not for patient care decisions.</p>
            </div>
          </div>
        </section>`;

      root.querySelectorAll('input[name="th"]').forEach((r) => r.addEventListener("change", () => {
        PG.state.settings.thresholdIndex = parseInt(r.value, 10); PG.saveSettings(); PG.setState({ settings: PG.state.settings });
      }));
      root.querySelector("#setRecal").addEventListener("change", (e) => {
        PG.state.settings.showRecalibrated = e.target.checked; PG.saveSettings(); PG.setState({ settings: PG.state.settings });
      });
      root.querySelector("#setMotion").addEventListener("change", (e) => {
        PG.state.settings.reduceMotion = e.target.checked; PG.saveSettings(); PG.setState({ settings: PG.state.settings });
      });
    },
  };
})();
