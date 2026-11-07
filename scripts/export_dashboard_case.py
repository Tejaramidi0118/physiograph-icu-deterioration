#!/usr/bin/env python3
"""
export_dashboard_case.py
========================
Export one ICU stay from the corrected filled parquets as a dashboard case
file (schema ``physiograph.case/1``), for loading in the PhysioGraph dashboard
(Patient Explorer -> "Load an exported case").

WHAT IT RECOVERS
----------------
``rebuild_filled_parquets.py`` stores MinMax-scaled values plus one
``<feature>_missing`` flag per hour, recorded BEFORE filling. Combined with the
forward-only fill limits in ``preprocessing_params.json`` this is enough to
recover the provenance of every model input exactly:

    flag == 0                                  -> "observed"
    flag == 1, within fill limit of last obs   -> "carried"  (o = source hour)
    flag == 1, otherwise                       -> "median"   (training median)

Values are de-normalised with ``v * data_range + data_min`` (the correct
MinMax inverse; briefing §3.2). Nothing is interpolated or filled backwards.

Scores are NOT computed here. They are attached only from a predictions CSV
you supply (columns ``stay_id, hour, score``), where ``hour`` is either the
elapsed-hour index of the exported grid (0 = first hour of the stay) or a
timestamp matching the parquet ``hour`` column. Such scores are tagged
``prediction_source: "model"`` with the checkpoint label you pass.

    python scripts/export_dashboard_case.py \\
        --data data/processed --split test --stay-id 30000153 \\
        --predictions predictions/test_scores.csv \\
        --checkpoint "best_physiograph_multi_horizon_1_1.pt (epoch 6)" \\
        --out dashboard/cases/30000153.json

The output contains de-identified MIMIC-IV data covered by the PhysioNet data
use agreement. Keep it inside your credentialed environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical feature names (column selection is always by name).
FEATURES = [
    "resp_rate", "spo2", "heart_rate", "sbp",
    "temperature", "wbc", "lactate", "creatinine",
]
DECIMALS = {"resp_rate": 1, "spo2": 1, "heart_rate": 1, "sbp": 1,
            "temperature": 2, "wbc": 2, "lactate": 2, "creatinine": 2}


class ExportError(RuntimeError):
    pass


def load_stay(path: Path, stay_id: int) -> pd.DataFrame:
    cols = ["stay_id", "hour"] + FEATURES + [f"{f}_missing" for f in FEATURES]
    try:
        df = pd.read_parquet(path, columns=cols, filters=[("stay_id", "==", stay_id)])
    except (ValueError, TypeError):
        df = pd.read_parquet(path, columns=cols)
        df = df[df["stay_id"] == stay_id]
    if df.empty:
        raise ExportError(f"stay_id {stay_id} not found in {path.name}")
    df = df.sort_values("hour").reset_index(drop=True)
    df["hour"] = pd.to_datetime(df["hour"])
    steps = df["hour"].diff().dropna().dt.total_seconds().unique()
    if len(steps) and not np.allclose(steps, 3600.0):
        raise ExportError("stay is not a strict 1-hour grid; was it built by rebuild_filled_parquets.py?")
    return df


def provenance(df: pd.DataFrame, params: dict) -> tuple[dict, dict]:
    """Per-feature [{v, s, o}] records plus consistency counters."""
    out, checks = {}, {}
    hours = np.arange(len(df))
    for f in FEATURES:
        lo, rng = params["data_min"][f], params["data_range"][f]
        limit = int(params["fill_limits"][f])
        med = float(params["median"][f])
        raw = df[f].to_numpy(dtype="float64") * rng + lo
        flag = df[f"{f}_missing"].to_numpy()
        tol = max(1e-3, rng * 1e-5)
        recs, last = [], None
        bad_carry = bad_median = 0
        for i in hours:
            v = round(float(raw[i]), DECIMALS[f])
            if flag[i] == 0:
                last = i
                recs.append({"v": v, "s": "observed", "o": int(i)})
            elif last is not None and i - last <= limit:
                bad_carry += abs(raw[i] - raw[last]) > tol
                recs.append({"v": v, "s": "carried", "o": int(last)})
            else:
                bad_median += abs(raw[i] - med) > tol
                recs.append({"v": v, "s": "median", "o": None if last is None else int(last)})
        out[f] = recs
        checks[f] = {"carried_mismatch": int(bad_carry), "median_mismatch": int(bad_median)}
    return out, checks


def load_predictions(path: Path, stay_id: int, df: pd.DataFrame) -> list[dict]:
    p = pd.read_csv(path)
    missing = {"stay_id", "hour", "score"} - set(p.columns)
    if missing:
        raise ExportError(f"predictions CSV lacks columns: {sorted(missing)}")
    p = p[p["stay_id"] == stay_id]
    if p.empty:
        return []
    if pd.api.types.is_numeric_dtype(p["hour"]):
        idx = p["hour"].astype(int)
    else:
        lookup = {ts: i for i, ts in enumerate(df["hour"])}
        idx = pd.to_datetime(p["hour"]).map(lookup)
    p = p.assign(idx=idx).dropna(subset=["idx"])
    p = p[(p["idx"] >= 0) & (p["idx"] < len(df))]
    return [{"hour": int(i), "score": round(float(s), 6)}
            for i, s in sorted(zip(p["idx"], p["score"]))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one ICU stay as a PhysioGraph dashboard case file.")
    ap.add_argument("--data", required=True, help="folder with {split}_filled.parquet and preprocessing_params.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--stay-id", type=int, required=True)
    ap.add_argument("--predictions", help="optional CSV with stay_id, hour, score")
    ap.add_argument("--checkpoint", help="label recorded with the scores, e.g. checkpoint file and epoch")
    ap.add_argument("--score-profile", default="corrected", choices=["corrected", "pre-correction"],
                    help="which run produced the scores; selects the documented thresholds in the dashboard")
    ap.add_argument("--label", help="display label (default: 'ICU stay <id>')")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    params = json.loads((data / "preprocessing_params.json").read_text())
    for key in ("data_min", "data_range", "median", "fill_limits"):
        if key not in params:
            raise ExportError(f"preprocessing_params.json lacks '{key}'; rebuild with rebuild_filled_parquets.py")

    df = load_stay(data / f"{args.split}_filled.parquet", args.stay_id)
    variables, checks = provenance(df, params)

    predictions = load_predictions(Path(args.predictions), args.stay_id, df) if args.predictions else None
    if args.predictions and not args.checkpoint:
        print("  note: --checkpoint not given; scores will be labelled without a checkpoint.")

    case = {
        "schema": "physiograph.case/1",
        "source": "mimic-iv-export",
        "case_id": str(args.stay_id),
        "label": args.label or f"ICU stay {args.stay_id}",
        "description": f"{args.split} split · exported from {args.split}_filled.parquet",
        "hours": list(range(len(df))),
        "timestamps": [ts.isoformat() for ts in df["hour"]],
        "variables": variables,
        "predictions": predictions if predictions else None,
        "prediction_source": "model" if predictions else None,
        "checkpoint": args.checkpoint if predictions else None,
        "score_profile": args.score_profile,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(case, separators=(",", ":")))

    print(f"Exported stay {args.stay_id}: {len(df)} hours -> {out}")
    for f in FEATURES:
        recs = variables[f]
        n = {s: sum(r["s"] == s for r in recs) for s in ("observed", "carried", "median")}
        c = checks[f]
        warn = "  <-- check params" if c["carried_mismatch"] or c["median_mismatch"] else ""
        print(f"  {f:12s} observed={n['observed']:4d} carried={n['carried']:4d} median={n['median']:4d}{warn}")
    if predictions is not None:
        print(f"  scores attached: {len(predictions)}")


if __name__ == "__main__":
    try:
        main()
    except ExportError as e:
        raise SystemExit(f"ABORTED: {e}")
