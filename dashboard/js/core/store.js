/**
 * dashboard/js/core/store.js
 *
 * Application state, case registry, case-file validation, settings and the
 * derived quantities every view shares (observation window, statuses, risk).
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});

  const STATUS = {
    observed:    { label: "Observed",       short: "Obs",     desc: "Charted at this hour" },
    carried:     { label: "Carried forward", short: "Carried", desc: "Last observation carried forward within the limit" },
    median:      { label: "Training median", short: "Median",  desc: "No observation within the carry-forward limit — model input is the training-set median" },
    unavailable: { label: "Unavailable",     short: "N/A",     desc: "No value exported for this hour" },
  };
  PG.STATUS = STATUS;

  // ------------------------------------------------------------- settings
  const SETTINGS_KEY = "physiograph.settings.v3";
  // thresholdIndex selects among the active checkpoint's documented thresholds.
  const DEFAULT_SETTINGS = { thresholdIndex: 0, showRecalibrated: true, reduceMotion: false };

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) return Object.assign({}, DEFAULT_SETTINGS, JSON.parse(raw));
    } catch (e) { /* storage unavailable */ }
    return Object.assign({}, DEFAULT_SETTINGS);
  }

  PG.saveSettings = function () {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(PG.state.settings)); } catch (e) { /* ignore */ }
  };

  // ------------------------------------------------------- real demo cases
  const CODE = { o: "observed", c: "carried", m: "median" };

  /** Expand one compact stay from mimic-demo-cases.js into a case object. */
  function expandDemo(r) {
    const t0 = Date.parse(r.t0 + "Z");
    const hours = [], timestamps = [];
    for (let i = 0; i < r.n; i++) {
      hours.push(i + r.offset);
      timestamps.push(new Date(t0 + i * 3600e3).toISOString().slice(0, 16));
    }
    const variables = {};
    for (const n of PG.NODES) {
      const src = r.vars[n.key];
      let last = null;
      variables[n.key] = src.v.map((v, i) => {
        const s = CODE[src.s[i]];
        if (s === "observed") last = i;
        return { v, s, o: last };
      });
    }
    const order = PG.DEMO_META.attr_node_order;
    const predictions = r.scores.map((score, j) => ({ hour: r.pred_start + j, score }));
    const attribution = {};
    r.attr.forEach((row, j) => {
      const shares = {};
      order.forEach((key, k) => (shares[key] = row[k] / 1000));
      attribution[r.pred_start + j] = shares;
    });
    return {
      schema: "physiograph.case/1",
      source: "mimic-iv-demo",
      case_id: String(r.stay_id),
      label: `Stay ${r.stay_id}`,
      description: `${r.careunit} · ICU length of stay ${r.los_days} d`,
      hours, timestamps, variables, predictions, attribution,
      prediction_source: "model",
      score_profile: "pre-correction",
      checkpoint: "pre-correction checkpoint · epoch 8",
      meta: { careunit: r.careunit, los_days: r.los_days, intime: r.intime, split: r.split },
    };
  }

  PG.cases = (PG.DEMO_RAW || []).map(expandDemo);

  // Open on a data-rich stay the checkpoint cannot have trained on: admitted in
  // the test period, at least 48 h, most observed laboratory values.
  const labObs = (c) => ["wbc", "lactate", "creatinine"].reduce((a, k) => a + c.variables[k].filter((r) => r.s === "observed").length, 0);
  const rankSplit = (c) => ({ test: 0, val: 1, train: 2 }[c.meta && c.meta.split] ?? 3);
  const initial = PG.cases.filter((c) => c.hours.length >= 48)
    .sort((a, b) => rankSplit(a) - rankSplit(b) || labObs(b) - labObs(a))[0] || PG.cases[0];

  // ---------------------------------------------------------------- state
  PG.state = {
    caseId: initial ? initial.case_id : null,
    t: initial ? initial.hours.length - 1 : 0,
    selected: "heart_rate",
    graphNode: null,
    graphEdge: null,
    graphLens: "structure",
    temporalVar: "heart_rate",
    settings: loadSettings(),
  };

  const listeners = new Set();
  PG.subscribe = (fn) => listeners.add(fn);
  PG.setState = function (patch) {
    Object.assign(PG.state, patch);
    listeners.forEach((fn) => fn(PG.state, patch));
  };

  PG.currentCase = () => PG.cases.find((c) => c.case_id === PG.state.caseId) || PG.cases[0];

  PG.selectCase = function (id) {
    const c = PG.cases.find((x) => x.case_id === id);
    if (!c) return;
    PG.setState({ caseId: id, t: c.hours.length - 1 });
  };

  // -------------------------------------------------------- derived values
  PG.windowRange = function (c, t) {
    const start = Math.max(0, t - PG.MODEL.windowH + 1);
    return { start, end: t, complete: t - start + 1 === PG.MODEL.windowH };
  };

  PG.record = function (c, key, t) {
    const arr = c.variables[key];
    const rec = arr && arr[t];
    if (!rec) return { v: null, s: "unavailable", o: null };
    return rec;
  };

  PG.windowRecords = function (c, key, t) {
    const { start, end } = PG.windowRange(c, t);
    const out = [];
    for (let h = start; h <= end; h++) out.push(Object.assign({ h }, PG.record(c, key, h)));
    return out;
  };

  PG.availability = function (c, key, t) {
    const counts = { observed: 0, carried: 0, median: 0, unavailable: 0 };
    PG.windowRecords(c, key, t).forEach((r) => counts[r.s]++);
    counts.total = counts.observed + counts.carried + counts.median + counts.unavailable;
    return counts;
  };

  PG.lastObserved = function (c, key, t) {
    for (let h = t; h >= 0; h--) {
      const r = PG.record(c, key, h);
      if (r.s === "observed") return { h, v: r.v };
    }
    return null;
  };

  PG.prediction = function (c, t) {
    if (!c.predictions) return null;
    return c.predictions.find((p) => p.hour === t) || null;
  };

  /** Thresholds and prevalence of the checkpoint that produced a case's scores. */
  PG.profile = (c) => PG.SCORE_PROFILES[(c && c.score_profile) || "corrected"] || PG.SCORE_PROFILES.corrected;

  /** Prior-shift recalibration from the balanced training prior to the profile's prevalence. */
  PG.recalibrate = function (p, c) {
    const pi = PG.profile(c).prevalence;
    const q = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    const z = Math.log(q / (1 - q)) + Math.log(pi / (1 - pi));
    return 1 / (1 + Math.exp(-z));
  };

  PG.thresholdInfo = function (c) {
    const list = PG.profile(c).thresholds;
    return list[PG.state.settings.thresholdIndex] || list[0];
  };

  PG.isDemo = (c) => c.source === "mimic-iv-demo";
  PG.sourceLabel = (c) => (PG.isDemo(c) ? "MIMIC-IV Demo v2.2" : "MIMIC-IV export");
  PG.attributionAt = (c, t) => (c.attribution && c.attribution[t]) || null;

  // ------------------------------------------------------------ formatting
  PG.fmt = {
    value(key, v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return Number(v).toFixed(PG.NODE[key].dp);
    },
    num(v, dp = 3) { return v === null || v === undefined ? "—" : Number(v).toFixed(dp); },
    pct(v, dp = 1) { return v === null || v === undefined ? "—" : (v * 100).toFixed(dp) + "%"; },
    int(v) { return Number(v).toLocaleString("en-US"); },
    hour(c, h) {
      return "ICU h " + c.hours[h];
    },
    stamp(c, h) {
      if (c.timestamps && c.timestamps[h]) return c.timestamps[h].replace("T", " ").slice(0, 16);
      return null;
    },
  };

  PG.esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  // -------------------------------------------------------- case loading
  /**
   * Validate a case file against physiograph.case/1. Returns a normalised
   * case or throws with a message suitable for display.
   */
  PG.validateCase = function (obj) {
    if (!obj || obj.schema !== "physiograph.case/1") throw new Error('Expected "schema": "physiograph.case/1".');
    if (!obj.case_id) throw new Error("Missing case_id.");
    if (!Array.isArray(obj.hours) || obj.hours.length === 0) throw new Error("Missing hours array.");
    const H = obj.hours.length;
    if (!obj.variables) throw new Error("Missing variables.");
    const variables = {};
    for (const n of PG.NODES) {
      const arr = obj.variables[n.key];
      if (!Array.isArray(arr)) throw new Error(`Missing variable "${n.key}".`);
      if (arr.length !== H) throw new Error(`"${n.key}" has ${arr.length} records; expected ${H}.`);
      variables[n.key] = arr.map((r, i) => {
        if (!r || !STATUS[r.s]) throw new Error(`"${n.key}"[${i}] has an unknown status.`);
        const v = r.v === null || r.v === undefined ? null : Number(r.v);
        return { v, s: r.s, o: r.o === undefined ? null : r.o };
      });
    }
    let predictions = null;
    if (Array.isArray(obj.predictions)) {
      predictions = obj.predictions
        .filter((p) => Number.isInteger(p.hour) && p.hour >= 0 && p.hour < H && typeof p.score === "number")
        .map((p) => ({ hour: p.hour, score: p.score }));
    }
    if (obj.source === "synthetic") throw new Error("Synthetic case files are not accepted; load a MIMIC-IV export.");
    const predSource = obj.prediction_source === "model" ? "model" : (predictions ? "unspecified" : null);
    let attribution = null;
    if (obj.attribution && typeof obj.attribution === "object") {
      attribution = {};
      for (const [h, shares] of Object.entries(obj.attribution)) {
        if (PG.NODES.every((n) => typeof shares[n.key] === "number")) attribution[+h] = shares;
      }
    }
    return {
      schema: obj.schema,
      source: "mimic-iv-export",
      attribution,
      score_profile: PG.SCORE_PROFILES[obj.score_profile] ? obj.score_profile : "corrected",
      case_id: String(obj.case_id),
      label: obj.label ? String(obj.label) : `ICU stay ${obj.case_id}`,
      description: obj.description ? String(obj.description) : "",
      hours: obj.hours.map(Number),
      timestamps: Array.isArray(obj.timestamps) && obj.timestamps.length === H ? obj.timestamps.map(String) : null,
      variables,
      predictions,
      prediction_source: predSource,
      checkpoint: obj.checkpoint ? String(obj.checkpoint) : null,
    };
  };

  PG.addCase = function (c) {
    const i = PG.cases.findIndex((x) => x.case_id === c.case_id);
    if (i >= 0) PG.cases[i] = c; else PG.cases.push(c);
    PG.selectCase(c.case_id);
  };
})();
