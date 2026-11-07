/**
 * dashboard/js/data/project.js
 *
 * Canonical project metadata and verified research results.
 *
 * Source-of-truth order (PhysioGraph_PROJECT_BRIEFING.md §0):
 *   executed code + recorded output  >  briefing  >  planning documents.
 *
 * Everything in this file is copied from the executed training notebook
 * (PhysioGraph_Final_Colab.ipynb, cell 4), scripts/rebuild_filled_parquets.py,
 * or the CURRENT results in briefing §3–§7. Nothing here is a placeholder.
 * Superseded numbers (briefing §9) are deliberately absent.
 *
 * NODE ORDERING
 * -------------
 * Three orderings exist in the repository:
 *   training notebook (executed model):  ... 5=wbc, 6=lactate, 7=creatinine
 *   src/graph/physio_graph.py + exploration notebook + rebuild script:
 *                                        ... 5=lactate, 6=creatinine, 7=wbc
 * Data columns are selected by NAME, so values were never mixed up, but edge
 * indices are only meaningful in the training-notebook order. This file keys
 * every node and edge by canonical feature name and stores the
 * training-notebook index as `index`, so no view relies on position.
 */
(function () {
  "use strict";
  const PG = (window.PG = window.PG || {});

  PG.VERSION = "2.0.0";

  PG.SYSTEMS = {
    pulmonary:      { label: "Pulmonary",          color: "var(--sys-pulm)" },
    cardiovascular: { label: "Cardiovascular",     color: "var(--sys-cardio)" },
    systemic:       { label: "Systemic / Thermal", color: "var(--sys-systemic)" },
    renal:          { label: "Renal / Metabolic",  color: "var(--sys-renal)" },
  };

  // Training-notebook ORGAN_SYSTEMS: Pulmonary [0,1], Cardiovascular [2,3],
  // Systemic_Thermal [4,5], Renal_Metabolic [6,7].
  // scale: MinMax inverse (value = v * range + min), briefing §3.2.
  // carryH / plaus: FILL_LIMITS / VALID_RANGES in rebuild_filled_parquets.py.
  // missTestPct: missing after forward fill, test split, briefing §3.5.
  // gradShare: plain-gradient node sensitivity share, briefing §6.2.
  PG.NODES = [
    { key: "resp_rate", index: 0, label: "Respiratory Rate", short: "RR", itemid: 220210,
      source: "chartevents", kind: "vital", unit: "breaths/min", unitShort: "/min", dp: 0,
      system: "pulmonary", carryH: 3, plaus: [4, 60], scale: { min: 4.0, range: 56.0 },
      missTestPct: 2.17, gradShare: 0.096, site: "Chest · lungs",
      siteNote: "Bedside respiratory rate, charted hourly." },
    { key: "spo2", index: 1, label: "SpO₂", short: "SpO₂", itemid: 220277,
      source: "chartevents", kind: "vital", unit: "%", unitShort: "%", dp: 0,
      system: "pulmonary", carryH: 3, plaus: [50, 100], scale: { min: 50.0, range: 50.0 },
      missTestPct: 1.89, gradShare: 0.147, site: "Upper chest · oxygenation",
      siteNote: "Peripheral oxygen saturation (pulse oximetry). Shown at the chest as the oxygenation reference." },
    { key: "heart_rate", index: 2, label: "Heart Rate", short: "HR", itemid: 220045,
      source: "chartevents", kind: "vital", unit: "bpm", unitShort: "bpm", dp: 0,
      system: "cardiovascular", carryH: 3, plaus: [20, 220], scale: { min: 20.0, range: 200.0 },
      missTestPct: 1.44, gradShare: 0.135, site: "Heart",
      siteNote: "Bedside heart rate, charted hourly.",
      caveat: "Model node 2 is a blend: 0.75 × heart rate + 0.25 × normalised shock index (HR/SBP). Attribution to this node is not attributable to heart rate alone." },
    { key: "sbp", index: 3, label: "Systolic Blood Pressure", short: "SBP", itemid: 220179,
      source: "chartevents", kind: "vital", unit: "mmHg", unitShort: "mmHg", dp: 0,
      system: "cardiovascular", carryH: 3, plaus: [40, 250], scale: { min: 40.0, range: 210.0 },
      missTestPct: 29.45, gradShare: 0.152, site: "Upper arm · cuff",
      siteNote: "Item 220179 is non-invasive cuff pressure. Arterial-line pressure (220050) was not extracted, which explains much of the missingness.",
      caveat: "Model node 3 is a blend: 0.75 × SBP − 0.25 × normalised shock index (HR/SBP). Attribution to this node is not attributable to blood pressure alone." },
    { key: "temperature", index: 4, label: "Temperature", short: "Temp", itemid: 223762,
      source: "chartevents", kind: "vital", unit: "°C", unitShort: "°C", dp: 1,
      system: "systemic", carryH: 6, plaus: [32, 43], scale: { min: 32.0, range: 10.7 },
      missTestPct: 93.87, gradShare: 0.110, site: "Forehead",
      siteNote: "Only the Celsius item (223762) was extracted; MIMIC-IV records most temperatures under the Fahrenheit item (223761).",
      caveat: "94% of temperature inputs are training-median imputation. Treat any temperature attribution with caution." },
    { key: "wbc", index: 5, label: "White Blood Cell Count", short: "WBC", itemid: 51301,
      source: "labevents", kind: "lab", unit: "K/µL", unitShort: "K/µL", dp: 1,
      system: "systemic", carryH: 24, plaus: [0.1, 100], scale: { min: 0.1, range: 99.7 },
      missTestPct: 11.17, gradShare: 0.094, site: "Blood sample · systemic",
      siteNote: "Laboratory value from a blood sample. It is a systemic marker, not measured at any organ." },
    { key: "lactate", index: 6, label: "Lactate", short: "Lac", itemid: 50813,
      source: "labevents", kind: "lab", unit: "mmol/L", unitShort: "mmol/L", dp: 1,
      system: "renal", carryH: 24, plaus: [0.1, 20], scale: { min: 0.1, range: 19.9 },
      missTestPct: 63.27, gradShare: 0.159, site: "Blood sample · metabolic",
      siteNote: "Laboratory value from a blood-gas sample. It is a systemic metabolic marker, not measured at any organ.",
      caveat: "Ranks first in gradient sensitivity despite 63% of test inputs being imputed." },
    { key: "creatinine", index: 7, label: "Creatinine", short: "Cr", itemid: 50912,
      source: "labevents", kind: "lab", unit: "mg/dL", unitShort: "mg/dL", dp: 2,
      system: "renal", carryH: 24, plaus: [0.1, 15], scale: { min: 0.1, range: 14.9 },
      missTestPct: 10.01, gradShare: 0.107, site: "Blood sample · kidney reference",
      siteNote: "Serum creatinine is a blood laboratory value. The kidney is shown only as the conceptual reference organ." },
  ];

  PG.NODE = Object.fromEntries(PG.NODES.map((n) => [n.key, n]));
  PG.NODE_BY_INDEX = Object.fromEntries(PG.NODES.map((n) => [n.index, n]));

  // Index of each canonical feature in each ordering found in the repository.
  PG.ORDERINGS = [
    { source: "PhysioGraph_Final_Colab.ipynb (cell 4) — executed model, canonical",
      order: ["resp_rate", "spo2", "heart_rate", "sbp", "temperature", "wbc", "lactate", "creatinine"] },
    { source: "scripts/rebuild_filled_parquets.py · src/graph/physio_graph.py · exploration notebook",
      order: ["resp_rate", "spo2", "heart_rate", "sbp", "temperature", "lactate", "creatinine", "wbc"] },
  ];

  // 15 directed edges, training-notebook order, keyed by canonical name.
  // Rationale strings follow the executed notebook's edge comments.
  PG.EDGES = [
    { from: "resp_rate",   to: "spo2",        type: "cascade", rationale: "Desaturation" },
    { from: "spo2",        to: "heart_rate",  type: "cascade", rationale: "Compensatory tachycardia" },
    { from: "heart_rate",  to: "sbp",         type: "cascade", rationale: "Perfusion" },
    { from: "sbp",         to: "creatinine",  type: "cascade", rationale: "Renal hypoperfusion · pre-renal AKI" },
    { from: "lactate",     to: "creatinine",  type: "cascade", rationale: "Acidosis compounds renal stress" },
    { from: "temperature", to: "wbc",         type: "cascade", rationale: "Systemic inflammatory response" },
    { from: "wbc",         to: "lactate",     type: "cascade", rationale: "Microcirculatory hypoperfusion" },
    { from: "resp_rate",   to: "lactate",     type: "cascade", rationale: "Anaerobic shift" },
    { from: "heart_rate",  to: "lactate",     type: "cascade", rationale: "Circulatory shock" },
    { from: "spo2",        to: "lactate",     type: "cascade", rationale: "Tissue hypoxia" },
    { from: "spo2",        to: "resp_rate",   type: "reflex",  rationale: "Carotid chemoreceptor reflex" },
    { from: "sbp",         to: "heart_rate",  type: "reflex",  rationale: "Baroreceptor reflex" },
    { from: "lactate",     to: "resp_rate",   type: "reflex",  rationale: "Kussmaul hyperventilation" },
    { from: "creatinine",  to: "sbp",         type: "reflex",  rationale: "Renin–angiotensin activation" },
    { from: "lactate",     to: "sbp",         type: "reflex",  rationale: "Distributive vasodilation" },
  ];

  PG.MODEL = {
    name: "PhysioGraph",
    title: "Explainable Disease Progression Modeling for Early ICU Deterioration Prediction",
    dataset: "MIMIC-IV v3.1",
    input: "24 h × 8 nodes × 4 channels",
    channels: ["value", "missing flag", "Δ1h", "Δ4h / 4"],
    windowH: 24,
    horizon: { from: 6, to: 8, label: "(t + 6, t + 8] h" },
    blend: { h8: 0.70, h6: 0.15, h12: 0.15 },
    blendFormula: "ŷ = σ(0.70·ℓ₈ + 0.15·ℓ₆ + 0.15·ℓ₁₂)",
    loss: "L₈ + 0.3·L₆ + 0.3·L₁₂ (BCEWithLogits)",
    trainedPrevalence: 0.5,
    architecture: [
      { name: "Graph Attention Network", detail: "2 layers · ℝ⁴ → ℝ²⁵⁶ (4 heads concat) → ℝ⁶⁴ · BatchNorm, GELU, residual, self-loops" },
      { name: "Cross-organ multihead attention", detail: "4 heads · LayerNorm residual" },
      { name: "Causal TCN", detail: "4 layers · k = 3 · dilations 1, 2, 4, 8 · receptive field 61 h > T = 24" },
      { name: "Temporal trajectory attention pooling", detail: "Pools over the 24-hour window" },
      { name: "Organ attention pooling", detail: "Pools over 8 nodes · weights γ retained" },
      { name: "Multi-horizon heads", detail: "65 → 32 → 1 · ℓ₈, ℓ₆, ℓ₁₂ · ICU elapsed time concatenated" },
    ],
    training: "AdamW · lr 3e-4 · wd 1e-4 · cosine T_max 10 · batch 256 · 10 epochs · selection on validation AP · seed 42",
    selectedEpoch: 6,
    selectedValAP: 0.8388,
  };

  // Documented operating thresholds, per checkpoint. A threshold belongs to the
  // run it was selected on, so scores are always classified with the
  // thresholds of the checkpoint that produced them.
  //   corrected      — briefing §5.3–5.6 (corrected pipeline, epoch 6)
  //   pre-correction — PhysioGraph_Final_Colab.ipynb recorded run (epoch 8),
  //                    the checkpoint present in models/
  PG.SCORE_PROFILES = {
    corrected: {
      label: "Corrected run (epoch 6)",
      prevalence: 0.01304,
      thresholds: [
        { value: 0.9059, label: "0.9059", note: "Selected on the validation set at its own prevalence (full population). Never tuned on test." },
        { value: 0.50, label: "0.50", note: "Fixed a priori for the balanced 1:1 protocol." },
      ],
    },
    "pre-correction": {
      label: "Pre-correction checkpoint (epoch 8)",
      prevalence: 0.01299,
      thresholds: [
        { value: 0.50, label: "0.50", note: "Fixed a priori; the threshold the notebook used for its balanced test metrics." },
        { value: 0.29, label: "0.29", note: "Stored in the checkpoint: F1-optimal on the balanced validation set (50% prevalence), not a full-population operating point." },
      ],
    },
  };
  PG.THRESHOLDS = PG.SCORE_PROFILES.corrected.thresholds;

  PG.CALIBRATION_OFFSET = -4.3268; // ln(π_te/(1−π_te)) − ln(π_tr/(1−π_tr)), briefing §5.5
  PG.TEST_PREVALENCE = 0.01304;

  PG.PIPELINE = [
    { step: "Source", detail: "MIMIC-IV v3.1 → 7,912,779 hourly rows · 94,438 ICU stays" },
    { step: "Split", detail: "Chronological by ICU intime, 70/15/15 by stay · stay overlap 0" },
    { step: "Grid", detail: "Complete hourly grid per stay · 8,234,792 rows" },
    { step: "Clip", detail: "Physiological plausibility ranges · artefacts → missing" },
    { step: "Flag", detail: "Missingness flags recorded before any filling" },
    { step: "Carry forward", detail: "Forward-only · vitals 3 h · temperature 6 h · labs 24 h" },
    { step: "Median", detail: "Residual gaps → training-set median (flag retained)" },
    { step: "Scale", detail: "MinMax [0, 1] fit on training split only" },
  ];

  // ---------------------------------------------------------------- results
  PG.RESULTS = {
    cohorts: [
      { protocol: "A · balanced 1:1", split: "train", pos: 53498, neg: 53498, prev: 0.5 },
      { protocol: "A · balanced 1:1", split: "val", pos: 11245, neg: 11245, prev: 0.5 },
      { protocol: "A · balanced 1:1", split: "test", pos: 11027, neg: 11027, prev: 0.5 },
      { protocol: "B · full population", split: "val", pos: 11245, neg: 824324, prev: 0.01346, n: 835569 },
      { protocol: "B · full population", split: "test", pos: 11027, neg: 834772, prev: 0.01304, n: 845799 },
      { protocol: "Stay level", split: "test", pos: 1287, neg: 7973, prev: 0.1390, n: 9260, unit: "stays" },
    ],
    validationTrace: [
      { epoch: 1, loss: 0.9374, auroc: 0.7951, ap: 0.8131, acc: 0.7130 },
      { epoch: 2, loss: 0.8801, auroc: 0.8054, ap: 0.8220, acc: 0.7140 },
      { epoch: 3, loss: 0.8652, auroc: 0.8106, ap: 0.8259, acc: 0.7090 },
      { epoch: 4, loss: 0.8510, auroc: 0.8168, ap: 0.8311, acc: 0.7335 },
      { epoch: 5, loss: 0.8411, auroc: 0.8137, ap: 0.8284, acc: 0.7130 },
      { epoch: 6, loss: 0.8316, auroc: 0.8249, ap: 0.8388, acc: 0.7422, best: true },
      { epoch: 7, loss: 0.8223, auroc: 0.8201, ap: 0.8349, acc: 0.7378 },
      { epoch: 8, loss: 0.8177, auroc: 0.8223, ap: 0.8367, acc: 0.7300 },
      { epoch: 9, loss: 0.8144, auroc: 0.8238, ap: 0.8384, acc: 0.7325 },
      { epoch: 10, loss: 0.8112, auroc: 0.8236, ap: 0.8380, acc: 0.7328 },
    ],
    balanced: {
      n: 22054, threshold: 0.5,
      rows: [
        { model: "Gradient-boosted trees (XGBoost)", auroc: 0.8105, ap: 0.8293, acc: 0.7331, f1: 0.7095, sens: null, spec: null,
          note: "68 engineered features from identical windows. The only strictly matched comparison." },
        { model: "PhysioGraph", auroc: 0.8225, ap: 0.8389, acc: 0.7381, f1: 0.7359, sens: 0.7296, spec: 0.7467, ours: true },
        { model: "Blend 0.60 / 0.40", auroc: 0.8314, ap: 0.8465, acc: 0.7496, f1: null, sens: 0.7149, spec: 0.7843, unverified: true,
          note: "Unverified whether α = 0.60 was fixed a priori. F1 omitted: its threshold was grid-searched on test labels." },
      ],
    },
    fullPopulation: { n: 845799, prevalence: 0.01304, auroc: 0.8224, ap: 0.1936, lift: 14.8, brier: 0.1832, f1: 0.2829, threshold: 0.9059 },
    calibration: {
      offset: -4.3268,
      brier: { uncalibrated: 0.1832, recalibrated: 0.0115, nullModel: 0.0129 },
      ece: { uncalibrated: 0.3701, recalibrated: 0.0024 },
    },
    windowOps: [
      { th: 0.50, sens: 0.730, spec: 0.746, ppv: 0.037, f1: 0.070, flagged: 26.06, fa: 26.4 },
      { th: 0.70, sens: 0.519, spec: 0.925, ppv: 0.083, f1: 0.144, flagged: 8.11, fa: 11.0 },
      { th: 0.80, sens: 0.407, spec: 0.967, ppv: 0.141, f1: 0.210, flagged: 3.76, fa: 6.1 },
      { th: 0.90, sens: 0.271, spec: 0.991, ppv: 0.294, f1: 0.282, flagged: 1.20, fa: 2.4 },
      { th: 0.9059, sens: 0.261, spec: 0.992, ppv: 0.309, f1: 0.283, flagged: 1.10, fa: 2.2, selected: true },
      { th: 0.95, sens: 0.170, spec: 0.997, ppv: 0.446, f1: 0.246, flagged: 0.50, fa: 1.2 },
    ],
    suppression: { rawAlarms: 9336, rawPct: 1.10, episodes: 2804, episodePct: 0.33, reductionPct: 70.0, episodePPV: 0.2518, refractoryH: 8 },
    stay: {
      n: 9260, prevalence: 0.1390, auroc: 0.8883, ap: 0.6509,
      ops: [
        { th: 0.50, alerted: 81.3, sens: 0.997, ppv: 0.170 },
        { th: 0.70, alerted: 55.6, sens: 0.957, ppv: 0.239 },
        { th: 0.80, alerted: 39.2, sens: 0.881, ppv: 0.312 },
        { th: 0.90, alerted: 18.4, sens: 0.674, ppv: 0.509 },
        { th: 0.95, alerted: 8.7, sens: 0.469, ppv: 0.744 },
        { th: 0.99, alerted: 2.0, sens: 0.132, ppv: 0.924 },
      ],
    },
    leadTime: { alerted: 844, of: 1287, pct: 65.6, medianH: -1.0, iqr: [-7.0, 17.3], fracPositive: 0.409, threshold: 0.9059 },
    ablations: {
      multiHorizon: { h8Only: 0.8279, blend: 0.8279 },
      topology: { clinical15: { auroc: 0.8279, ap: 0.8432, valAP: 0.8387 }, remaining: 3 },
      seedSpread: { a: 0.8225, b: 0.8279, spread: 0.0054 },
    },
  };

  PG.EXPLAIN = {
    method: "Plain gradient φᵢ,ₖ = ∂s/∂xᵢ,ₖ of the blended logit · one backward pass · float32",
    windows: 256,
    gini: 0.176,
    maxMinRatio: 1.69,
    time: { meanAbsGradHour: 0.001799, meanAbsGradNormHour: 0.1799, meanAbsGradX: 0.2466, sharePct: 0.095 },
    faithfulness: {
      baseline: 0.6690,
      rows: [
        { k: 1, gradient: -0.0363, attention: 0.0067, random: 0.0105 },
        { k: 2, gradient: -0.0330, attention: 0.0032, random: 0.0194 },
        { k: 3, gradient: -0.0226, attention: 0.0097, random: 0.0196 },
        { k: 4, gradient: -0.0063, attention: 0.0198, random: 0.0211 },
      ],
    },
    rankCorrelation: [
      { comparison: "grad×input vs attention", rho: -0.238, fracNeg: 0.76, p: "3.3e−23" },
      { comparison: "gradient vs attention", rho: 0.071, fracNeg: 0.39, p: "8.2e−04" },
    ],
    gradXInputTemperature: 0.240,
  };

  // What PhysioGraph_Final_Colab.ipynb actually printed (cells 9–13). This run
  // used the pre-correction data (bidirectional interpolation, IQR capping,
  // NaN→0 before scaling) and the z-score de-normalisation constants; the
  // briefing (§9) marks these figures as superseded. Shown only in a clearly
  // separated section of Model Performance.
  PG.NOTEBOOK_RUN = {
    file: "PhysioGraph_Final_Colab.ipynb",
    cohorts: [
      { split: "train", pos: 53278, neg: 53278 },
      { split: "val", pos: 11206, neg: 11206 },
      { split: "test", pos: 10984, neg: 10984 },
    ],
    trace: [
      { epoch: 1, loss: 0.9171, auroc: 0.8010, ap: 0.8197, acc: 0.7169 },
      { epoch: 2, loss: 0.8643, auroc: 0.8041, ap: 0.8244, acc: 0.7204 },
      { epoch: 3, loss: 0.8440, auroc: 0.8151, ap: 0.8329, acc: 0.7250 },
      { epoch: 4, loss: 0.8323, auroc: 0.8168, ap: 0.8350, acc: 0.7184 },
      { epoch: 5, loss: 0.8179, auroc: 0.8211, ap: 0.8392, acc: 0.7427 },
      { epoch: 6, loss: 0.8070, auroc: 0.8236, ap: 0.8403, acc: 0.7373 },
      { epoch: 7, loss: 0.7969, auroc: 0.8217, ap: 0.8396, acc: 0.7385 },
      { epoch: 8, loss: 0.7888, auroc: 0.8235, ap: 0.8413, acc: 0.7386, best: true },
      { epoch: 9, loss: 0.7820, auroc: 0.8222, ap: 0.8402, acc: 0.7387 },
      { epoch: 10, loss: 0.7784, auroc: 0.8214, ap: 0.8397, acc: 0.7357 },
    ],
    balanced: [
      { model: "PhysioGraph", auroc: 0.8256, ap: 0.8422, acc: 0.7340, f1: 0.6914, sens: 0.5958, spec: 0.8723, ours: true },
      { model: "XGBoost (68 features, 40,000 train windows)", auroc: 0.8105, ap: 0.8288, acc: 0.7346, f1: 0.7159, sens: null, spec: null },
      { model: "Ensemble 0.60 / 0.40", auroc: 0.8327, ap: 0.8478, acc: 0.7449, f1: 0.7540, sens: 0.6245, spec: 0.8653, note: "F1 at 0.33, a threshold grid-searched on the test labels." },
    ],
    fullPopulation: { n: 845689, positives: 10984, prevalence: 0.01299, auroc: 0.8264, ap: 0.1853, lift: 14.3, brier: 0.1075, f1: 0.0691, threshold: 0.35 },
    excluded: "The notebook's MEWS row (AUROC 76.10%) is a hard-coded literal, not a computed result, and is not shown.",
  };

  // Checkpoint actually present in models/ (inspected with torch.load). Its
  // epoch, validation AUPRC and threshold match the run recorded in
  // PhysioGraph_Final_Colab.ipynb, not the corrected run in the briefing.
  PG.PROVENANCE = {
    documented: { file: "best_physiograph_multi_horizon_1_1.pt", epoch: 6, valAP: 0.8388 },
    inRepository: { file: "models/best_physiograph_multi_horizon_1_1.pt", epoch: 8, valAP: 0.8413, storedThreshold: 0.29 },
    supersededFiles: [
      "benchmarks/physiograph_multi_horizon_benchmark_results.csv",
      "benchmarks/bootstrap_gnn_summary.json",
      "benchmarks/bootstrap_ensemble_summary.json",
      "scripts/export_filled_features.py (bidirectional interpolation)",
    ],
  };
})();
