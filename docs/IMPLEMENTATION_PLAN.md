# Implementation and experiment checklist

## Before training

- [ ] Verify split disjointness by `stay_id`.
- [ ] Verify event onset columns use ICU-relative hours from the same `intime` source.
- [ ] Check duplicate `(stay_id, hour)` rows.
- [ ] Check one-hour spacing and how preprocessing handles missing hours.
- [ ] Confirm missingness flags use `1 = missing`.
- [ ] Confirm scaler was fitted on train only.
- [ ] Run V2 label audit and compare counts to the audit report.
- [ ] Export V2 label tables for all splits.
- [ ] Run synthetic tests and one small data smoke test.

## Baselines

- [ ] Rerun corrected target-specific XGBoost sampler.
- [ ] Save the exact command, package versions, seed, and output CSV.
- [ ] Compare physiology model with time-only baseline.
- [ ] Train GNN on the same target, horizon, validity rules, and split.

## Controlled model comparisons

- [ ] XGBoost rolling statistics.
- [ ] Time-only baseline.
- [ ] Temporal-only neural baseline (no graph message passing).
- [ ] Current GAT → TCN → MLP with locked graph.
- [ ] Edge/graph ablation while keeping all other settings fixed.
- [ ] Missingness/recency ablation.

## Final evaluation

- [ ] Select model and operating threshold using validation only.
- [ ] Freeze configuration, seed strategy, and threshold.
- [ ] Evaluate test once.
- [ ] Report AUROC, AUPRC, prevalence, lift, sensitivity, specificity, precision, F1, Brier score, and confusion matrix as applicable.
- [ ] Use stay-clustered confidence intervals for window-level metrics.
- [ ] Distinguish window-level from stay-level results.
- [ ] Describe the sepsis proxy accurately and avoid causal claims about graph edges.
