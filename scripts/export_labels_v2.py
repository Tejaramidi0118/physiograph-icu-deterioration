#!/usr/bin/env python3
"""
export_labels_v2.py
-------------------
Reproduces:
  - data/processed/labels_v2/train_labels_v2.parquet
  - data/processed/labels_v2/val_labels_v2.parquet
  - data/processed/labels_v2/test_labels_v2.parquet

Origin:
  Formulated in docs/original_scripts/labels_v2_original.py to establish
  clinically sound deterioration labels free from admission state leakage.

Labeling Task Formulation:
  - Event configurations:
      • all4: vasopressor OR invasive ventilation OR death OR sepsis-proxy
      • nosepsis: vasopressor OR invasive ventilation OR death
      • sepsisonly: sepsis-proxy only
  - Prediction horizons: H ∈ {6, 8, 12, 24} hours ahead.
  - History Lookback constraint: Minimum 6 hours history (MIN_HIST = 6) before predicting.
  - Dynamic at-risk: An event already present at hour h is an existing admission state,
    not a future deterioration. Only at-risk events (onset > h) are valid targets.
  - Determinacy filter: A negative window is valid only if the stay remained observed
    for the entire forward horizon (h + H <= last_observed_hour).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

EVENT_COLS = {
    "vasopressor": "vasopressor_hour",
    "ventilation": "ventilation_hour",
    "death": "death_hour",
    "sepsis": "sepsis_hour",
}

CONFIGS = {
    "all4": ["vasopressor", "ventilation", "death", "sepsis"],
    "nosepsis": ["vasopressor", "ventilation", "death"],
    "sepsisonly": ["sepsis"],
}

HORIZONS = (6, 8, 12, 24)
MIN_HIST = 6


def load_intime(path: Path) -> pd.DataFrame:
    """Loads stay_id and intime admission timestamps."""
    icu = pd.read_parquet(path)
    icu = icu[["stay_id", "intime"]].copy()
    icu["stay_id"] = pd.to_numeric(icu["stay_id"]).astype("int64")
    icu["intime"] = pd.to_datetime(icu["intime"])
    return icu.drop_duplicates("stay_id")


def attach_hours(df: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    """Calculates hours elapsed since ICU admission."""
    df = df.merge(icu, on="stay_id", how="left")
    if df["intime"].isna().any():
        n = int(df["intime"].isna().sum())
        raise ValueError(f"{n} rows have no ICU admission timestamp.")
    df["hours"] = (pd.to_datetime(df["hour"]) - df["intime"]).dt.total_seconds() / 3600.0
    return df.drop(columns="intime")


def load_events(path: Path) -> pd.DataFrame:
    """Loads onset hour for each deterioration event per stay."""
    ev = pd.read_csv(path)
    out = pd.DataFrame({"stay_id": pd.to_numeric(ev["stay_id"]).astype("int64")})
    for name, col in EVENT_COLS.items():
        out[name] = pd.to_numeric(ev[col], errors="coerce") if col in ev.columns else np.nan
    return out.drop_duplicates("stay_id").set_index("stay_id")


def compute_labels(df: pd.DataFrame, events: pd.DataFrame, configs=None, horizons=HORIZONS, min_hist=MIN_HIST):
    """
    Computes rolling horizon deterioration labels and validity indicators.
    Returns:
      labels: dict[(config, H)] = (y: int8 array, valid: bool array)
    """
    configs = configs or CONFIGS
    stay = df["stay_id"].to_numpy()
    h = df["hours"].to_numpy(dtype=np.float64)
    grp = df.groupby("stay_id")
    pos = grp.cumcount().to_numpy()
    last_hour = grp["hours"].transform("max").to_numpy(dtype=np.float64)

    onset = {}
    for name in EVENT_COLS:
        onset[name] = events[name].reindex(stay).to_numpy(dtype=np.float64)

    # Minimum history lookback requirement (at least 6 hours observed)
    enough = pos >= (min_hist - 1)
    labels = {}

    for cfg, types in configs.items():
        pos_ev = {H: np.zeros(len(h), dtype=bool) for H in horizons}
        at_risk_any = np.zeros(len(h), dtype=bool)

        for t in types:
            o = onset[t]
            not_started = np.isnan(o) | (o > h)
            at_risk_any |= not_started
            has = ~np.isnan(o)
            for H in horizons:
                pos_ev[H] |= has & (o > h) & (o <= h + H)

        for H in horizons:
            y = pos_ev[H]
            determinate = y | ((h + H) <= last_hour)
            valid = enough & at_risk_any & determinate
            labels[(cfg, H)] = (y.astype(np.int8), valid)

    return labels


def export_split_labels(split: str, features_path: Path, intime_path: Path, events_path: Path, out_dir: Path):
    print(f"\n[Processing Split: {split.upper()}]")
    print(f"Reading feature split from: {features_path}")
    df = pd.read_parquet(features_path, columns=["stay_id", "hour"])
    df["stay_id"] = df["stay_id"].astype("int64")

    # Attach ICU-relative hours
    icu = load_intime(intime_path)
    df = attach_hours(df, icu)

    # Filter pre-admission measurements (hours >= 0) and sort chronologically
    df = df[df["hours"] >= 0].sort_values(["stay_id", "hours"]).reset_index(drop=True)
    print(f"In-ICU rows (hours >= 0): {len(df):,} across {df['stay_id'].nunique():,} stays")

    # Load composite events
    events = load_events(events_path)

    # Compute V2 labels
    labels = compute_labels(df, events)

    # Build output table
    out = pd.DataFrame({
        "stay_id": df["stay_id"],
        "hour": df["hour"],
        "hours": df["hours"]
    })

    for (cfg, H), (y, v) in labels.items():
        out[f"label_{cfg}_h{H}"] = y
        out[f"valid_{cfg}_h{H}"] = v

    out_file = out_dir / f"{split}_labels_v2.parquet"
    print(f"Saving {len(out):,} rows and {out.shape[1]} columns to {out_file}...")
    out.to_parquet(out_file, index=False)
    print(f"✅ Generated {out_file.name} ({out_file.stat().st_size / (1024*1024):.2f} MB)")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "processed"
    checkpoints_dir = data_dir / "checkpoints"
    out_dir = data_dir / "labels_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    intime_path = data_dir / "icu_intime.parquet"
    events_path = data_dir / "composite_events.csv"

    if not intime_path.exists() or not events_path.exists():
        print(f"❌ Missing prerequisites: {intime_path} or {events_path}")
        return

    splits = ["val", "test", "train"]
    for split in splits:
        # Check normalized checkpoint or filled checkpoint
        feat_path = checkpoints_dir / f"{split}_normalized.parquet"
        if not feat_path.exists():
            feat_path = checkpoints_dir / f"{split}_filled.parquet"
        
        if not feat_path.exists():
            print(f"⚠️ Feature file for {split} not found. Skipping.")
            continue

        export_split_labels(split, feat_path, intime_path, events_path, out_dir)


if __name__ == "__main__":
    main()

