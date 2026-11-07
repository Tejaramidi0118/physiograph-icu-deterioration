#!/usr/bin/env python3
"""
rebuild_filled_parquets.py
==========================
Vectorised, leakage-free replacement for Cells 95-102 of data_exploration.ipynb.

WHAT THIS FIXES
---------------
1. LEAKAGE (fatal).  The old fill was
       .ffill(limit=3).interpolate(method="linear", limit_direction="both")
   `limit_direction="both"` with no `limit=` draws a straight line across gaps
   of ANY length in BOTH directions, so a lab drawn after a deterioration event
   was interpolated backwards into windows that precede it.  95% of lactate,
   94% of WBC and 93% of creatinine values were synthetic this way.
   Here: forward-only carry-forward with a per-feature limit.  Nothing from the
   future ever reaches a past row.

2. CLINICALLY DESTRUCTIVE CAPPING.  IQR capping clipped lactate at 3.0 mmol/L
   (265,571 values), creatinine at 2.95 mg/dL (597,627) and WBC at 24.3
   (267,855).  Sepsis-3 uses lactate > 2.0 and septic shock > 4.0, so the old
   cap deleted exactly the signal the model exists to detect.
   Here: physiological plausibility ranges, which remove recording artefacts
   without touching real pathology.

3. BROKEN DE-NORMALISATION.  The training notebook inverts the scaling as
       hr = v * 18.14 + 85.99        # mean/SD form
   but the data was MinMax-scaled to [0,1], which has no mean or SD.  The true
   inverse is v * (max - min) + min.
   Here: the fitted scaler AND an explicit denorm table are saved to JSON, so
   the training notebook can read the right constants instead of guessing.

4. ZERO-FILL OF MISSING VALUES.  `fillna(0)` before MinMax put absent readings
   at the bottom of the scale, which the model cannot distinguish from a real
   extreme low.  Here: residual gaps take the TRAIN median and the companion
   *_missing flag carries the information.

INPUT   data/processed/checkpoints/graph_{train,val,test}_raw.parquet
        (written by Cell 94 - the MIMIC extraction does NOT need to rerun)
OUTPUT  {train,val,test}_filled.parquet  + preprocessing_params.json

Run:    python rebuild_filled_parquets.py --in-dir <ckpt> --out-dir <out>
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES = [
    "resp_rate", "spo2", "heart_rate", "sbp",
    "temperature", "lactate", "creatinine", "wbc",
]

# Forward-only carry-forward limits, in hours.
# Bedside vitals are charted hourly, so a short limit is right.
# Labs are drawn once or twice a day; 24 h reflects what a clinician would
# actually have in front of them. Lower it to 12 for a more conservative run.
FILL_LIMITS = {
    "resp_rate": 3, "spo2": 3, "heart_rate": 3, "sbp": 3,
    "temperature": 6,
    "lactate": 24, "creatinine": 24, "wbc": 24,
}

# Physiological plausibility ranges. These remove impossible recordings
# (probe artefacts, transcription errors) and nothing else.
VALID_RANGES = {
    "resp_rate":   (4.0,  60.0),
    "spo2":        (50.0, 100.0),
    "heart_rate":  (20.0, 220.0),
    "sbp":         (40.0, 250.0),
    "temperature": (32.0,  43.0),
    "lactate":     (0.1,  20.0),
    "creatinine":  (0.1,  15.0),
    "wbc":         (0.1, 100.0),
}


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def build_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Complete hourly index per stay, from its first to its last charted hour.

    Vectorised equivalent of the per-stay date_range/reindex loop; verified
    row-for-row identical to it.
    """
    g = df.groupby("stay_id")["hour"].agg(["min", "max"])
    counts = (((g["max"] - g["min"]).dt.total_seconds() // 3600)
              .astype("int64") + 1).to_numpy()
    total = int(counts.sum())
    starts = np.concatenate([[0], counts.cumsum()[:-1]])
    offsets = np.arange(total) - np.repeat(starts, counts)
    return pd.DataFrame({
        "stay_id": np.repeat(g.index.to_numpy(), counts),
        "hour": np.repeat(g["min"].to_numpy(), counts)
                + offsets.astype("timedelta64[h]"),
    })


def prepare_split(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    """Grid -> plausibility clip -> missing flags -> forward-only fill."""
    t0 = time.time()
    raw = raw[["stay_id", "hour"] + FEATURES].copy()
    raw["hour"] = pd.to_datetime(raw["hour"])
    raw["stay_id"] = pd.to_numeric(raw["stay_id"]).astype("int64")
    raw = raw.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    raw = raw.drop_duplicates(["stay_id", "hour"], keep="last")

    grid = build_hourly_grid(raw)
    out = grid.merge(raw, on=["stay_id", "hour"], how="left")
    del grid, raw
    gc.collect()

    # Clip implausible recordings to NaN BEFORE the flags are taken, so an
    # artefact counts as missing rather than as an observation.
    for f, (lo, hi) in VALID_RANGES.items():
        out.loc[(out[f] < lo) | (out[f] > hi), f] = np.nan
        out[f] = out[f].astype("float32")

    # Missingness flags reflect genuine observation, recorded before filling.
    for f in FEATURES:
        out[f"{f}_missing"] = out[f].isna().astype("int8")

    # FORWARD-ONLY fill, within stay, per-feature limit. No future leakage.
    out = out.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    grp = out.groupby("stay_id", sort=False)
    for f, limit in FILL_LIMITS.items():
        out[f] = grp[f].ffill(limit=limit).astype("float32")

    obs = {f: int((out[f"{f}_missing"] == 0).sum()) for f in FEATURES}
    filled = {f: int(out[f].notna().sum() - obs[f]) for f in FEATURES}
    print(f"  [{name}] rows={len(out):,} stays={out.stay_id.nunique():,} "
          f"({time.time() - t0:.1f}s)")
    for f in FEATURES:
        still = int(out[f].isna().sum())
        print(f"    {f:12s} observed={obs[f]:>9,}  carried={filled[f]:>9,}  "
              f"still_missing={still:>9,} ({100 * still / len(out):5.2f}%)")
    return out



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    pass


def validate_split(df: pd.DataFrame, name: str) -> None:
    """Refuse to continue on an input that cannot produce a usable dataset."""
    problems = []
    for f in FEATURES:
        n_obs = int((df[f"{f}_missing"] == 0).sum())
        if n_obs == 0:
            problems.append(f"{f}: ZERO observed values")
        elif n_obs / len(df) < 0.001:
            problems.append(f"{f}: only {n_obs:,} observed "
                            f"({100 * n_obs / len(df):.4f}%)")
    if problems:
        raise PipelineError(
            f"\n  Split '{name}' is unusable:\n    "
            + "\n    ".join(problems)
            + "\n\n  This almost always means the input parquet is a stale or\n"
              "  partial artifact rather than the real split. Re-derive the\n"
              "  splits from the source table with --graph-data/--icustays."
        )


def check_overlap(splits: dict, icustays: pd.DataFrame | None) -> None:
    """Stay-level overlap must be zero; patient-level is reported."""
    ids = {k: set(v["stay_id"].unique()) for k, v in splits.items()}
    print("\nStay counts:", {k: f"{len(v):,}" for k, v in ids.items()})
    bad = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(ids[a] & ids[b])
        print(f"  stay overlap {a:5s} & {b:5s}: {n:,}")
        bad |= n > 0
    if bad:
        raise PipelineError("Splits share ICU stays. Re-derive the splits.")

    if icustays is not None and "subject_id" in icustays.columns:
        m = dict(zip(icustays["stay_id"], icustays["subject_id"]))
        pat = {k: {m[s] for s in v if s in m} for k, v in ids.items()}
        print("\nPATIENT-level overlap (limitation T1):")
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            n = len(pat[a] & pat[b])
            flag = "  <-- leakage" if n else "  OK"
            print(f"  patient overlap {a:5s} & {b:5s}: {n:,}{flag}")


def split_from_source(graph_path: Path, icustays: pd.DataFrame) -> dict:
    """Reproduce the chronological 70/15/15 ICU-stay split (Cells 92-93)."""
    print(f"Loading source table {graph_path.name} ...")
    gd = pd.read_parquet(graph_path)
    gd["stay_id"] = pd.to_numeric(gd["stay_id"]).astype("int64")
    print(f"  {len(gd):,} rows, {gd['stay_id'].nunique():,} stays")

    info = (icustays[["stay_id", "intime"]].drop_duplicates("stay_id")
            .sort_values("intime").reset_index(drop=True))
    info = info[info["stay_id"].isin(gd["stay_id"].unique())]
    n = len(info)
    a, b = int(n * 0.70), int(n * 0.85)
    sets = {"train": info.iloc[:a]["stay_id"].to_numpy(),
            "val":   info.iloc[a:b]["stay_id"].to_numpy(),
            "test":  info.iloc[b:]["stay_id"].to_numpy()}
    for k, v in sets.items():
        print(f"  {k:5s}: {len(v):,} stays")
    return {k: gd[gd["stay_id"].isin(v)].copy() for k, v in sets.items()}


def fit_params(train: pd.DataFrame) -> dict:
    """Medians and MinMax bounds, fit on TRAIN only."""
    params = {"features": FEATURES, "fill_limits": FILL_LIMITS,
              "valid_ranges": VALID_RANGES, "median": {},
              "data_min": {}, "data_max": {}, "data_range": {}}
    for f in FEATURES:
        med = float(train[f].median())
        if not np.isfinite(med):
            raise PipelineError(
                f"Feature '{f}' has no finite values in TRAIN, so no "
                "median or scaling bounds can be computed.")
        params["median"][f] = med
        filled = train[f].fillna(med)
        lo, hi = float(filled.min()), float(filled.max())
        params["data_min"][f] = lo
        params["data_max"][f] = hi
        params["data_range"][f] = (hi - lo) if hi > lo else 1.0
    return params


def apply_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Median-fill residual gaps, then MinMax to [0,1] with TRAIN bounds."""
    for f in FEATURES:
        df[f] = df[f].fillna(params["median"][f])
        df[f] = ((df[f] - params["data_min"][f])
                 / params["data_range"][f]).clip(0.0, 1.0).astype("float32")
    return df


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild filled parquets with forward-only (leak-free) imputation.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph-data",
                     help="icu_graph_hourly_raw.parquet (the Cell 79 source "
                          "table). RECOMMENDED: splits are re-derived here, so "
                          "stale checkpoints cannot be picked up.")
    src.add_argument("--in-dir",
                     help="folder holding graph_{train,val,test}_raw.parquet")
    ap.add_argument("--icustays",
                    help="icustays.csv/parquet. Required with --graph-data. "
                         "If it has subject_id, patient-level overlap is "
                         "reported (limitation T1).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inspect", action="store_true",
                    help="validate the inputs and exit without writing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    icustays = None
    if args.icustays:
        ip = Path(args.icustays)
        icustays = (pd.read_parquet(ip) if ip.suffix == ".parquet"
                    else pd.read_csv(ip))
        icustays["stay_id"] = pd.to_numeric(icustays["stay_id"]).astype("int64")
        if "intime" in icustays:
            icustays["intime"] = pd.to_datetime(icustays["intime"])

    print("=" * 72)
    print("Rebuilding filled parquets  (forward-only fill, no leakage)")
    print("=" * 72)

    if args.graph_data:
        if icustays is None:
            raise SystemExit("--graph-data requires --icustays")
        raw = split_from_source(Path(args.graph_data), icustays)
    else:
        raw = {}
        for name in ["train", "val", "test"]:
            f = Path(args.in_dir) / f"graph_{name}_raw.parquet"
            if not f.exists():
                raise SystemExit(f"{f} not found.")
            print(f"Loading {f.name} ...")
            raw[name] = pd.read_parquet(f)

    splits = {}
    for name, df in raw.items():
        print()
        splits[name] = prepare_split(df, name)
        validate_split(splits[name], name)
    del raw
    gc.collect()

    check_overlap(splits, icustays)

    if args.inspect:
        print("\n--inspect: inputs validated, nothing written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nFitting median / MinMax parameters on TRAIN only ...")
    params = fit_params(splits["train"])

    for name, df in splits.items():
        df = apply_params(df, params)
        cols = (["stay_id", "hour"] + FEATURES
                + [f"{f}_missing" for f in FEATURES])
        dst = out_dir / f"{name}_filled.parquet"
        df[cols].to_parquet(dst, index=False)
        print(f"  wrote {dst}  ({len(df):,} rows)")

    pj = out_dir / "preprocessing_params.json"
    pj.write_text(json.dumps(params, indent=2))
    print(f"  wrote {pj}")

    print("\n" + "=" * 72)
    print("DE-NORMALISATION CONSTANTS  (use these in the training notebook)")
    print("=" * 72)
    print("    import json")
    print("    P = json.load(open('preprocessing_params.json'))")
    print("    def denorm(v, f):")
    print("        return v * P['data_range'][f] + P['data_min'][f]\n")
    for f in FEATURES:
        print(f"    {f:12s} value = v * {params['data_range'][f]:8.3f} "
              f"+ {params['data_min'][f]:8.3f}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as e:
        print(f"\nABORTED: {e}")
        raise SystemExit(1)