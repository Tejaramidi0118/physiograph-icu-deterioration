#!/usr/bin/env python3
"""
export_filled_features.py
-------------------------
Reproduces:
  - data/processed/checkpoints/train_filled.parquet
  - data/processed/checkpoints/val_filled.parquet
  - data/processed/checkpoints/test_filled.parquet

Origin:
  Extracted directly from notebooks/data_exploration.ipynb (Cells 87–96).

Pipeline:
  1. Loads raw hourly bedside chart events and lab observations (graph_{split}_raw.parquet).
  2. Expands each ICU stay to a full continuous 1-hour grid from stay start to end.
  3. Creates binary missingness indicator columns: {feature}_missing.
  4. Imputes physiological vitals/labs using forward-fill (limit=3 hours) followed by
     bidirectional linear interpolation.
  5. Exports the filled splits to Parquet checkpoints.
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 8 Core Physiological Vitals and Lab Biomarkers
GRAPH_FEATURES = [
    "resp_rate",
    "spo2",
    "heart_rate",
    "sbp",
    "temperature",
    "lactate",
    "creatinine",
    "wbc"
]

def fill_per_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands each ICU stay to a continuous hourly frequency and applies
    forward-fill (limit 3h) followed by linear interpolation.
    Preserves true observation vs imputation via binary indicator flags.
    """
    parts = []
    
    # Process each stay independently
    for stay_id, group in tqdm(df.groupby("stay_id", sort=False), desc="Filling stays"):
        group = group.sort_values("hour").copy()

        # Generate full continuous 1-hour time index
        full_hours = pd.date_range(
            start=group["hour"].min(),
            end=group["hour"].max(),
            freq="h"
        )

        # Reindex to full hourly grid
        group = (
            group
            .set_index("hour")
            .reindex(full_hours)
            .rename_axis("hour")
            .reset_index()
        )

        group["stay_id"] = stay_id

        # Carry forward patient identifiers if present
        for id_col in ["subject_id", "hadm_id"]:
            if id_col in group.columns:
                group[id_col] = group[id_col].ffill().bfill()

        # Step A: Create binary missingness indicators (1 if originally missing / not charted)
        for feature in GRAPH_FEATURES:
            if feature in group.columns:
                group[f"{feature}_missing"] = group[feature].isna().astype("int8")

        # Step B: Project-specified imputation (Forward fill limit 3h, then linear interpolation)
        present_features = [f for f in GRAPH_FEATURES if f in group.columns]
        group[present_features] = (
            group[present_features]
            .ffill(limit=3)
            .interpolate(method="linear", limit_direction="both")
        )

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = root_dir / "data" / "processed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splits = ["val", "test", "train"]

    for split in splits:
        raw_path = checkpoint_dir / f"graph_{split}_raw.parquet"
        out_path = checkpoint_dir / f"{split}_filled.parquet"

        if not raw_path.exists():
            print(f"⚠️ Raw input {raw_path} not found. Skipping {split}.")
            continue

        print(f"\n[Processing Split: {split.upper()}]")
        print(f"Loading raw features from: {raw_path}")
        raw_df = pd.read_parquet(raw_path)
        print(f"Raw shape: {raw_df.shape} | Unique Stays: {raw_df['stay_id'].nunique():,}")

        print(f"Applying hourly grid expansion and fill_per_stay()...")
        filled_df = fill_per_stay(raw_df)
        print(f"Filled shape: {filled_df.shape} | Total Hourly Rows: {len(filled_df):,}")

        print(f"Exporting to: {out_path} ...")
        filled_df.to_parquet(out_path, index=False)
        print(f"✅ Successfully exported {out_path.name} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()

