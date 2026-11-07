#!/usr/bin/env python3
"""
mews_baseline.py
----------------
Tier 3: Implementation and Evaluation of the Modified Early Warning Score (MEWS) baseline.
Replaces fabricated/hardcoded literals with a genuine, clinically faithful rule-based model.

MEWS Scoring Rules:
---------------------------------------------------------------------------------------
Score                3          2          1          0          1          2          3
---------------------------------------------------------------------------------------
SBP (mmHg)          <=70      71-80      81-100    101-199       -        >=200        -
Heart Rate (bpm)     -         <40       41-50      51-100    101-110    111-129     >=130
Resp Rate (bpm)      -          <9         -         9-14      15-20      21-29      >=30
Temp (°C)            -         <35       35.0-38.4     -       38.5-38.9   >=39.0      -
AVPU (Neuro)         -          -          -         Alert     Voice       Pain      Unresp
---------------------------------------------------------------------------------------
Note on AVPU: Consciousness is unrecorded in standard continuous bedside vitals, so
AVPU is assigned 0 (Alert) uniformly, yielding a standardized 4-vital MEWS score (0-11).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def compute_mews_score(row: pd.Series) -> int:
    """Computes vital-sign-based MEWS score for a single observation."""
    score = 0

    # 1. Systolic Blood Pressure (mmHg)
    sbp = row.get("sbp", np.nan)
    if not np.isnan(sbp):
        if sbp <= 70:
            score += 3
        elif sbp <= 80:
            score += 2
        elif sbp <= 100:
            score += 1
        elif sbp >= 200:
            score += 2

    # 2. Heart Rate (bpm)
    hr = row.get("heart_rate", np.nan)
    if not np.isnan(hr):
        if hr < 40:
            score += 2
        elif hr <= 50:
            score += 1
        elif hr <= 100:
            score += 0
        elif hr <= 110:
            score += 1
        elif hr <= 129:
            score += 2
        elif hr >= 130:
            score += 3

    # 3. Respiratory Rate (breaths/min)
    rr = row.get("resp_rate", np.nan)
    if not np.isnan(rr):
        if rr < 9:
            score += 2
        elif rr <= 14:
            score += 0
        elif rr <= 20:
            score += 1
        elif rr <= 29:
            score += 2
        elif rr >= 30:
            score += 3

    # 4. Temperature (°C)
    temp = row.get("temperature", np.nan)
    if not np.isnan(temp):
        if temp > 50:
            temp = (temp - 32) * 5.0 / 9.0
        if temp < 35.0:
            score += 2
        elif temp <= 38.4:
            score += 0
        elif temp <= 38.9:
            score += 1
        elif temp >= 39.0:
            score += 2

    return score


def evaluate_cohort_mews(pred_df: pd.DataFrame, vitals_df: pd.DataFrame, cohort_name: str = "Protocol A"):
    print(f"\n[Evaluating MEWS on {cohort_name}]")
    print(f"Aligning {len(pred_df):,} prediction windows with physical bedside vitals...")

    # Filter vitals to relevant stays
    relevant_stays = pred_df["stay_id"].unique()
    v_sub = vitals_df[vitals_df["stay_id"].isin(relevant_stays)].copy()

    # Merge on stay_id and rounded hours
    pred_df = pred_df.copy()
    pred_df["h_round"] = pred_df["hour_in_icu"].round(2)
    v_sub["h_round"] = v_sub["hour_in_icu"].round(2)

    merged = pred_df.merge(
        v_sub[["stay_id", "h_round", "sbp", "heart_rate", "resp_rate", "temperature"]],
        on=["stay_id", "h_round"],
        how="inner"
    ).drop_duplicates(["stay_id", "h_round"])

    print(f"Successfully matched: {len(merged):,} / {len(pred_df):,} windows")
    if len(merged) == 0:
        print("❌ Could not match windows.")
        return

    # Compute MEWS scores
    print("Computing MEWS scores across all windows...")
    scores = merged.apply(compute_mews_score, axis=1).to_numpy()
    y_true = merged["true_label"].to_numpy()

    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    # Standard clinical alert trigger: MEWS >= 3 or MEWS >= 4
    for trigger in [3, 4]:
        pred_bin = (scores >= trigger).astype(int)
        acc = accuracy_score(y_true, pred_bin)
        tp = np.sum((pred_bin == 1) & (y_true == 1))
        fp = np.sum((pred_bin == 1) & (y_true == 0))
        tn = np.sum((pred_bin == 0) & (y_true == 0))
        fn = np.sum((pred_bin == 0) & (y_true == 1))

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        f1 = 2 * (ppv * sens) / max(ppv + sens, 1e-12)

        print(f"\n--- MEWS Benchmark ({cohort_name}) ---")
        print(f"Trigger Threshold : MEWS >= {trigger}")
        print(f"AUROC             : {auroc*100:.2f}%")
        print(f"AUPRC             : {auprc*100:.2f}%")
        print(f"Accuracy          : {acc*100:.2f}%")
        print(f"Sensitivity       : {sens*100:.2f}%")
        print(f"Specificity       : {spec*100:.2f}%")
        print(f"PPV               : {ppv*100:.2f}%")
        print(f"F1 Score          : {f1*100:.2f}%")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "data" / "processed" / "checkpoints"
    pred_dir = root_dir / "predictions"
    intime_path = root_dir / "data" / "processed" / "icu_intime.parquet"
    stats_path = checkpoints_dir / "scaler_statistics.json"

    pred_1_1_path = pred_dir / "test_predictions_multi_horizon_1_1.parquet"
    test_filled_path = checkpoints_dir / "test_filled.parquet"

    if not pred_1_1_path.exists() or not test_filled_path.exists() or not intime_path.exists():
        print(f"❌ Missing prerequisites for evaluation.")
        return

    print("[1/3] Loading physical vitals and inverse-scaling...")
    intime_df = pd.read_parquet(intime_path)
    with open(stats_path) as f:
        scaler_stats = json.load(f)

    # Load test_filled
    tf = pd.read_parquet(
        test_filled_path,
        columns=["stay_id", "hour", "sbp", "heart_rate", "resp_rate", "temperature"]
    )
    # De-standardise back to genuine physical units
    for col in ["sbp", "heart_rate", "resp_rate", "temperature"]:
        mean = scaler_stats[col]["mean"]
        scale = scaler_stats[col]["scale"]
        tf[col] = tf[col] * scale + mean

    # Compute hour_in_icu
    tf = tf.merge(intime_df, on="stay_id", how="left")
    tf["hour_in_icu"] = (pd.to_datetime(tf["hour"]) - tf["intime"]).dt.total_seconds() / 3600.0

    print("[2/3] Loading Protocol A (Balanced 1:1, N=21,968)...")
    pred_1_1 = pd.read_parquet(pred_1_1_path, columns=["stay_id", "hour_in_icu", "true_label"])
    evaluate_cohort_mews(pred_1_1, tf, cohort_name="Protocol A (Balanced 1:1, N=21,968)")


if __name__ == "__main__":
    main()
