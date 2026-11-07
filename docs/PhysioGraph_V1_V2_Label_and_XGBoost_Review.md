# PhysioGraph: V1 vs V2 Label Diagnostics and XGBoost Baseline Review

**Purpose:** Consolidate the previously supplied V1 diagnostic findings, the new V2 label audit, the full V2 XGBoost results, the smoke-test results, and a code review of `labels_v2.py` and `xgb_v2.py`.

**Evidence boundary:** Metrics below are reported from the user's pasted command output. Code-review findings are based on the uploaded scripts. No test-set performance is reported here. V2 validation results should be treated as promising exploratory results until the training-sampling issue described below is corrected and the final configuration is rerun.

---

## 1. Executive summary

1. **V1 defines a narrow target:** positive only in the single 6–8-hour-before-event slot. Its supplied validation predictions were weak: AUROC 0.5515, AUPRC 0.00763, prevalence 0.5865%, lift 1.30×. This was also inconsistent with an earlier training-console result (AUROC 0.6120, AUPRC 0.0127), so those two V1 evaluations should not be treated as reconciled.
2. **V1 has a cohort/target mismatch risk:** event stays with an event early in the ICU stay often cannot contribute a usable positive, and rows in the 0–6 hours before an event are negative under the single-slot definition.
3. **V2 defines a more actionable prediction horizon:** for a window ending at hour `h`, a selected event type is at risk only if it has not started by `h`; the label is positive if its onset is in `(h, h + H]`. V2 requires at least six hourly rows of history, does not truncate a stay after its first event, and drops outcomes that are not observable through the full horizon when no event occurs.
4. **The full-run XGBoost output reports substantial discrimination under V2.** For the primary H=8 horizon, `all4` reaches AUROC 0.7751 / AUPRC 0.1724; `nosepsis` reaches AUROC 0.8303 / AUPRC 0.2068; `sepsisonly` reaches AUROC 0.7192 / AUPRC 0.0556.
5. **The non-sepsis event set is the strongest of the three in this experiment.** At H=8, `nosepsis` is materially stronger than `all4`, while `sepsisonly` is much weaker. This suggests the sepsis proxy is not the main source of the reported XGBoost signal. It does not prove the proxy is invalid; it shows the prediction task and event composition matter.
6. **Physiology improves substantially over the time-only baseline in the reported results.** At H=8, `nosepsis` AUROC is 0.8303 compared with 0.6269 for `hour_in_icu` alone. This supports additional predictive signal from the measured features in this run.
7. **Important qualification:** `xgb_v2.py` constructs a single negative-sampling mask across all configurations and horizons, then applies weights based on whether a row is positive for *any* target. For a specific target, rows that are negative for that target but positive for another target can be retained with weight 1 instead of the inverse sampling-probability weight. This can bias the per-target training distribution. Fix and rerun before treating these metrics as final.
8. **The V2 scripts are not yet a drop-in replacement for the current GNN training pipeline.** `labels_v2.py` audits/computes labels in memory; `xgb_v2.py` trains directly on those labels. The existing GNN `train.py`/`PhysioDataset` would need a compatible target and validity-mask path before it can be fairly trained and compared on V2.

---

## 2. V1 baseline: what the previous diagnostics established

The prior V1 diagnostics reported:

- Validation prediction windows: 222,492 across 5,276 stays.
- Positive windows: 1,305; prevalence: 0.5865%.
- Recomputed AUROC: 0.5515.
- Recomputed AUPRC / average precision: 0.007627 (0.763%).
- AUPRC divided by prevalence: 1.30×.
- Earlier training log, not reconciled with the saved prediction file: AUROC 0.6120 and AUPRC 0.0127.

The V1 label audit showed the single-slot definition produces a 100% label rate in the `[6,8]` hours-to-first-event bin and 0% in the other bins, including the 0–6 hours immediately before an event. This means a window immediately preceding deterioration can be labeled negative if it is outside the designated two-hour slot.

The previous event-timing audit also showed substantial early-event burden. For example, among event stays, the first event was before hour 6 for many stays, while the V1 24-row eligibility requirement further restricted which stays could produce a usable positive. Those findings motivated V2.

**V1 interpretation:** the original target is unusually narrow and creates a potentially clinically awkward distinction between a window 7 hours before an event (positive) and a window 3 hours before it (negative). The metric discrepancy between the training log and saved predictions remains unresolved and should be documented rather than silently reconciled.

---

## 3. V2 label definition in `labels_v2.py`

### 3.1 Prediction unit and history

- A prediction is associated with each hourly row/window endpoint `h`.
- The feature window looks back at most 24 hours.
- At least `MIN_HIST = 6` hourly rows must be available; the earliest eligible endpoint is approximately hour 5 when indexing starts at hour 0.
- Stays are not cut off after the first event.

### 3.2 At-risk event definition

For each selected event type `X`, it is considered at risk at time `h` when its onset is missing or strictly later than `h`. If the event has already started at or before `h`, it represents the current state and is not a future target for that event type.

For horizon `H`, a row is positive if at least one selected event onset satisfies:

`h < onset <= h + H`

A row with no future event is retained only if observation continues through the complete horizon (`h + H <= last observed hour`). Otherwise, the outcome is censored/unknown and the row is excluded. Positive rows are retained because the future event is observed.

### 3.3 Event-set configurations

| Configuration | Event types | Intended diagnostic question |
|---|---|---|
| `all4` | Vasopressor, ventilation, death, sepsis proxy | Can the model predict the composite endpoint? |
| `nosepsis` | Vasopressor, ventilation, death | How predictable is the endpoint without the sepsis proxy? |
| `sepsisonly` | Sepsis proxy only | How much signal is present for the sepsis proxy task alone? |

### 3.4 Horizons

`H = 8` is designated as the primary horizon; 6, 12, and 24 hours are sensitivity analyses. The 24-hour number is the maximum **lookback**, not the prediction horizon.

---

## 4. V2 label audit results

Source: pasted output from `python labels_v2.py`, using `data/processed/checkpoints/val_normalized.parquet`.

- Stays: 14,164
- Hourly rows: 1,232,929
- Rows with at least six hours of history: 1,162,508

### 4.1 Cohort and prevalence

| Config | Horizon (h) | Valid windows | Positive windows | Prevalence | Stays with valid windows | Stays with ≥1 positive |
|---|---:|---:|---:|---:|---:|---:|
| all4 | 6 | 1,084,717 | 22,742 | 2.097% | 13,598 | 3,264 |
| all4 | 8 | 1,059,794 | 29,373 | 2.772% | 13,397 | 3,281 |
| all4 | 12 | 1,011,891 | 41,660 | 4.117% | 12,820 | 3,320 |
| all4 | 24 | 890,782 | 72,812 | 8.174% | 10,109 | 3,439 |
| nosepsis | 6 | 1,083,970 | 13,914 | 1.284% | 13,595 | 2,161 |
| nosepsis | 8 | 1,058,787 | 18,011 | 1.701% | 13,388 | 2,171 |
| nosepsis | 12 | 1,010,189 | 25,590 | 2.533% | 12,789 | 2,188 |
| nosepsis | 24 | 885,939 | 44,510 | 5.024% | 9,953 | 2,219 |
| sepsisonly | 6 | 595,167 | 9,191 | 1.544% | 11,616 | 1,723 |
| sepsisonly | 8 | 575,329 | 11,933 | 2.074% | 11,407 | 1,733 |
| sepsisonly | 12 | 537,309 | 17,117 | 3.186% | 10,850 | 1,764 |
| sepsisonly | 24 | 443,452 | 31,030 | 6.997% | 8,220 | 1,867 |

As expected, longer horizons increase the positive rate. The valid cohort shrinks with horizon because a negative label requires full follow-up through that horizon. `sepsisonly` has fewer valid windows because its at-risk condition is defined only by sepsis onset.

### 4.2 Event onsets before the first eligible window

The audit treats events that began before approximately hour 5 as already present at the first eligible endpoint, not as future labels for that same event type.

| Event type | Stays with event | Event before hour 5 | Percentage before hour 5 |
|---|---:|---:|---:|
| Vasopressor | 4,016 | 2,783 | 69.3% |
| Ventilation | 4,601 | 3,453 | 75.0% |
| Death | 1,709 | 59 | 3.5% |
| Sepsis proxy | 4,389 | 1,900 | 43.3% |

This supports the motivation for V2: early events are treated as already present at a prediction time rather than requiring the model to predict an event that has already occurred. Later deterioration in the same stay can still be represented for other event types that remain at risk.

### 4.3 Positive composition for `all4`, H=8

The audit explicitly notes that a row may count for more than one event type.

| Event type | Positive rows attributed to type |
|---|---:|
| Vasopressor | 6,290 |
| Ventilation | 5,982 |
| Death | 7,447 |
| Sepsis proxy | 11,933 |

These are overlapping attribution counts, not mutually exclusive classes; they must not be summed to infer the total number of unique positive rows.

---

## 5. XGBoost v2 method

The script constructs one feature vector per eligible hourly endpoint from the current and preceding rows, using a maximum 24-hour history. For each of eight physiological variables it computes:

- Last observed value
- Mean, standard deviation, minimum, and maximum of observed values
- Missing fraction
- Hours since the last observation, capped at the window length

It also includes `hour_in_icu`. Missing measurements are represented as NaN for XGBoost rather than being treated as observed values. The full model is compared with a time-only model using `hour_in_icu` alone.

Training negatives are subsampled to reduce memory use; the script intends to keep all positives and reweight sampled negatives by inverse keep probability. The validation split is not negative-subsampled. The script uses validation early stopping, so validation estimates have some model-selection optimism. The test split should remain untouched until configuration selection is complete.

---

## 6. Full-run XGBoost validation results

Source: full `python xgb_v2.py` output. These are the reported validation results, not test results.

### 6.1 All configurations and horizons

| Config | H (h) | Train positives | Estimated train windows | Validation windows | Validation positives | Prevalence | AUROC | AUPRC | Lift | Time-only AUROC | Time-only AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all4 | 6 | 107,796 | 5,077,699 | 1,084,717 | 22,742 | 2.097% | 0.7860 | 0.1583 | 7.55× | 0.6405 | 0.0337 |
| all4 | 8 | 139,038 | 4,960,294 | 1,059,794 | 29,373 | 2.772% | 0.7751 | 0.1724 | 6.22× | 0.6368 | 0.0431 |
| all4 | 12 | 196,558 | 4,738,360 | 1,011,891 | 4.117% | 0.7596 | 0.2009 | 4.88× | 0.6336 | 0.0618 |
| all4 | 24 | 342,352 | 4,172,392 | 890,782 | 72,812 | 8.174% | 0.7367 | 0.2696 | 3.30× | 0.6361 | 0.1201 |
| nosepsis | 6 | 66,653 | 5,074,326 | 1,083,970 | 13,914 | 1.284% | 0.8436 | 0.1959 | 15.26× | 0.6323 | 0.0206 |
| nosepsis | 8 | 86,131 | 4,955,606 | 1,058,787 | 18,011 | 1.701% | 0.8303 | 0.2068 | 12.15× | 0.6269 | 0.0264 |
| nosepsis | 12 | 122,016 | 4,730,231 | 1,010,189 | 25,590 | 2.533% | 0.8110 | 0.2321 | 9.16× | 0.6218 | 0.0379 |
| nosepsis | 24 | 211,407 | 4,150,139 | 885,939 | 44,510 | 5.024% | 0.7797 | 0.2834 | 5.64× | 0.6159 | 0.0722 |
| sepsisonly | 6 | 43,015 | 2,735,618 | 595,167 | 9,191 | 1.544% | 0.7296 | 0.0460 | 2.98× | 0.5604 | 0.0195 |
| sepsisonly | 8 | 55,781 | 2,641,723 | 575,329 | 11,933 | 2.074% | 0.7192 | 0.0556 | 2.68× | 0.5590 | 0.0256 |
| sepsisonly | 12 | 79,764 | 2,464,851 | 537,309 | 17,117 | 3.186% | 0.7056 | 0.0722 | 2.27× | 0.5535 | 0.0378 |
| sepsisonly | 24 | 144,305 | 2,023,508 | 443,452 | 31,030 | 6.997% | 0.6850 | 0.1205 | 1.72× | 0.5507 | 0.0798 |

### 6.2 Primary horizon H=8

| Config | Prevalence | AUROC | AUPRC | Lift | Time-only AUROC | AUROC 95% bootstrap CI | AUPRC 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| all4 | 2.772% | 0.7751 | 0.1724 | 6.22× | 0.6368 | [0.7674, 0.7823] | [0.1638, 0.1820] |
| nosepsis | 1.701% | 0.8303 | 0.2068 | 12.15× | 0.6269 | [0.8209, 0.8375] | [0.1915, 0.2216] |
| sepsisonly | 2.074% | 0.7192 | 0.0556 | 2.68× | 0.5590 | [0.6979, 0.7383] | [0.0485, 0.0640] |

The code requests stay-clustered bootstrap intervals for the primary horizon. The pasted full-run output gives the intervals shown above. The number of bootstrap replicates is controlled by `--n-boot` (default 100 in the script); report the actual invocation alongside the intervals in final documentation.

### 6.3 Interpretation by event configuration

- **`nosepsis` is strongest** across the tested horizons. At H=8, AUROC is 0.8303 and AUPRC is 0.2068 despite 1.70% prevalence.
- **`all4` is also learnable** in the reported run, but weaker than `nosepsis` at H=8.
- **`sepsisonly` is substantially weaker**, especially in AUPRC lift. At H=8 its lift is 2.68×, compared with 12.15× for `nosepsis`.
- **The physiology features add signal beyond admission time.** The H=8 AUROC gaps over the time-only model are approximately 0.138 for `all4`, 0.203 for `nosepsis`, and 0.160 for `sepsisonly`.
- **Longer horizons raise raw AUPRC while lowering lift and AUROC.** Because prevalence rises as H increases, raw AUPRC alone should not be used to choose the horizon. For this run, H=6 has the highest AUROC and lift within each configuration, while H=24 has the highest raw AUPRC.

These results support the existence of predictive signal for V2, especially for non-sepsis events, but are not clinical validation and do not establish that the labels are clinically correct.

---

## 7. Smoke-test results are not the full-run results

The earlier command used `--max-stays 1500 --horizons 8 --n-boot 20`. It reported:

| Config | Validation positives | Prevalence | AUROC | AUPRC | Lift | Time-only AUROC |
|---|---:|---:|---:|---:|---:|---:|
| all4 | 3,052 | 2.614% | 0.6842 | 0.0842 | 3.22× | 0.6412 |
| nosepsis | 1,862 | 1.597% | 0.7899 | 0.1293 | 8.10× | 0.6118 |
| sepsisonly | 1,241 | 1.868% | 0.6012 | 0.0307 | 1.64× | 0.5997 |

These results are based on the first 1,500 stays in each split, not a random sample. They are useful as a pipeline smoke test only and should not be compared as if they were a representative validation estimate. The full run supersedes them for reporting.

---

## 8. Feature importance for `all4`, H=8

The full-run XGBoost output reported the following top 12 features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `spo2_last` | 0.1132 |
| 2 | `lactate_last` | 0.0837 |
| 3 | `hour_in_icu` | 0.0545 |
| 4 | `lactate_hrs_since` | 0.0507 |
| 5 | `heart_rate_hrs_since` | 0.0490 |
| 6 | `spo2_hrs_since` | 0.0393 |
| 7 | `sbp_last` | 0.0356 |
| 8 | `spo2_std` | 0.0317 |
| 9 | `heart_rate_last` | 0.0313 |
| 10 | `sbp_miss_rate` | 0.0292 |
| 11 | `resp_rate_mean` | 0.0289 |
| 12 | `lactate_mean` | 0.0201 |

The most influential features are SpO2 and lactate summaries, followed by time in ICU and recency/missingness-derived features. These are model feature importances, not causal effects or evidence that an intervention on a feature would change risk.

---

## 9. Code review findings

### 9.1 `labels_v2.py`: definition and implementation

**What is implemented consistently with the documented intent:**

- Computes ICU-relative hours using `intime` and feature timestamps.
- Defines event sets and horizons explicitly.
- Treats an event with onset `<= h` as already started at the prediction time.
- Defines a positive for `h < onset <= h + H`.
- Keeps a negative only when full horizon follow-up is available.
- Requires at least six hourly rows of history and does not stop processing a stay after its first event.

**Items to verify before finalizing the target:**

1. Confirm the event-hour columns in `composite_events.csv` use the same ICU-relative time origin as `hours`. If event hours are absolute or admission-relative to a different reference, labels would be wrong.
2. Confirm that the source features are genuinely hourly and that duplicate or missing hour rows are handled as intended. `MIN_HIST` is implemented as six rows, not an explicit elapsed-time check.
3. The audit prints per-type positive attribution counts that overlap. These are not class counts and must not be summed to calculate unique positives.
4. The audit only prints statistics; it does not write a reusable label/validity table for the GNN. The actual training pipeline needs a deliberate integration step.

### 9.2 `xgb_v2.py`: important negative-sampling weighting issue

In `build_split`, the script creates a shared sampling mask across all configurations and horizons:

- `any_pos` is true if a row is positive for **any** configured target/horizon.
- Rows positive for any target are always retained.
- Other valid rows are retained with probability `keep_neg_frac`.
- The shared weight is 1 for `any_pos` rows and `1 / keep_neg_frac` for sampled rows that are negative for all targets.

Later, `fit_eval` trains a specific `(config, H)` target using that target's validity mask, but uses the shared weight. Consequently, a row can be negative for the current target while positive for another target; it is always retained and gets weight 1 instead of the inverse sampling probability that would be needed if it were sampled as a negative for the current target. This can distort the effective negative distribution for each target and affect the reported metrics.

**Recommended correction:** use target-specific negative sampling and weights for each `(config, H)` training task, or retain all valid training rows if feasible. For target-specific sampling, define the target's `valid` and `y` first; retain all target positives and sample only that target's valid negatives. Give sampled target negatives weight `1 / keep_neg_frac` and target positives weight 1. Make sure rows invalid for that target do not contribute to its training loss. Then rerun all configurations/horizons and regenerate confidence intervals.

This is a methodological concern identified from code inspection; the direction and size of its effect on the current metrics cannot be determined without rerunning.

### 9.3 Other implementation/experimental cautions

- **Smoke-test selection:** `--max-stays` takes the first N stays after sorting, not a random sample. Treat smoke-test metrics as debugging evidence only.
- **Validation early stopping:** the validation set is used for model selection, so validation metrics are somewhat optimistic. Use the untouched test split once configuration choices are fixed.
- **Bootstrap design:** the script resamples stays, which is preferable to treating correlated windows as independent. Record the number of bootstrap replicates and preserve the test set for final evaluation.
- **Feature construction:** rolling features are built per stay using padding so windows do not mix adjacent stays. Confirm the feature parquet's missingness convention (`1 = missing`) matches the script's `np.where(m == 1, np.nan, x)` assumption.
- **Potential cross-target sampling bias is the main issue to fix before relying on the reported full-run XGBoost metrics.**

---

## 10. Implications for PhysioGraph GNN

The V2 label definition is not a simple replacement for the existing V1 label parquet. It introduces, for each configuration/horizon:

- A different positive definition.
- A different valid-row mask because unknown/censored negatives are dropped.
- Different eligible-window counts by horizon and event set.
- Multiple experiments (`all4`, `nosepsis`, `sepsisonly`; H=6, 8, 12, 24).

The existing GNN pipeline should not be pointed at V2 features while continuing to use V1 label files. To make a fair comparison with XGBoost, build or export the same V2 labels and validity masks for train/validation/test, align them to the same feature-hour rows, and ensure the GNN only trains/evaluates on valid endpoints. Preserve stay-level splits and the same lookback/horizon definitions.

A fair comparison should use the same cohort and target rows for XGBoost and PhysioGraph, with the test set untouched during selection. Compare AUROC, AUPRC, lift over prevalence, calibration, and clinically relevant operating-point metrics; report uncertainty at the stay level.

---

## 11. Recommended next steps in order

1. **Fix target-specific negative sampling in `xgb_v2.py`.** Rerun the full validation experiments for `all4`, `nosepsis`, and `sepsisonly` at H=6, 8, 12, and 24.
2. **Save a reproducible experiment record:** command, code version, seed, feature files, event file, split definitions, negative sampling rate, number of bootstrap replicates, and generated result CSV.
3. **Freeze the target choice using validation only.** H=8 is the documented primary horizon; H=6/12/24 remain sensitivity analyses unless a predeclared reason changes the primary target.
4. **Integrate V2 labels and validity masks into the GNN dataset/training code.** Do not mix V1 labels with V2 model features or compare results from different eligible rows as though they were identical tasks.
5. **Run the final chosen configuration once on the untouched test split.** Report stay-clustered confidence intervals and calibration/threshold metrics.
6. **Then compare PhysioGraph with XGBoost on exactly the same valid endpoints.** A GNN should not be claimed better or worse if the target, cohort, preprocessing, or split differs.

---

## 12. Overall conclusion

V2 addresses a substantive weakness of V1 by defining a future-event horizon, distinguishing events already present from events yet to occur, allowing later deterioration after an earlier event, and excluding unobservable negative outcomes. The V2 audit produces far more usable windows and higher prevalence than V1. The reported full-run XGBoost results show strong apparent discrimination, particularly for the non-sepsis event set, and consistently outperform a time-only baseline.

However, these are validation results from an exploratory baseline. The shared negative-sampling mask/weight logic in `xgb_v2.py` is not target-specific and can bias training for individual configurations and horizons. Correct this and rerun before treating the exact reported metrics as final. Next, adapt the GNN to the same V2 labels, validity masks, and rows; only then is a direct model comparison defensible. Keep the test set untouched until those choices are fixed.
