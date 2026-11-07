#!/usr/bin/env python3
"""
build_demo_cases.py
===================
Build the dashboard's patient cases from REAL ICU stays in the open-access
MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL v1.0; the same demo
dataset used by notebooks/01_explore_mimic.ipynb).

    python scripts/build_demo_cases.py \\
        --raw  data/raw/mimic-iv-demo \\
        --ckpt models/best_physiograph_multi_horizon_1_1.pt \\
        --notebook PhysioGraph_Final_Colab.ipynb \\
        --out  dashboard/js/data/mimic-demo-cases.js

WHAT EACH PART REPRODUCES
-------------------------
1. Extraction - notebooks/data_exploration.ipynb cells 35-38, 73-75:
   chartevents 220045/220210/220277/220179 (+ 223762 temperature), labevents
   50813/50912/51301 joined to the stay by hadm_id within [intime, outtime];
   charttime floored to the hour; hourly median per item.
2. Displayed values and observation status - scripts/rebuild_filled_parquets.py
   (the corrected pipeline): hourly grid, plausibility clip, missing flags taken
   before filling, FORWARD-ONLY carry-forward (vitals 3 h, temperature 6 h,
   labs 24 h). Residual gaps are training-median filled in the real pipeline;
   the training medians are not in the repository, so those hours are exported
   as status "median" with no value.
3. Scores - the checkpoint in models/ with the EXACT model code executed in
   PhysioGraph_Final_Colab.ipynb (cells 4 and 6 are exec'd from the notebook),
   the notebook's 4-channel encoding and shock-index injection (including its
   original constants), and the input scaling that produced its training data
   (data_exploration cells 98-102: train IQR caps, NaN -> 0, MinMax with min 0
   and max = upper cap). One deliberate change makes scoring causal: the
   executed fill was ffill(3) + linear interpolation in BOTH directions, which
   reads future values. Here each hour uses past observations only (ffill(3),
   then last observation carried forward). Scores are therefore not identical
   to what the notebook would produce, and the checkpoint is the
   pre-correction run (epoch 8). They are labelled as such in the dashboard.
4. Attribution - the project's plain-gradient method (run_attribution2.py):
   |d s / d x| of the blended logit summed over time and channels per node,
   normalised to shares, for every scored hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebuild_filled_parquets as rb  # noqa: E402  FEATURES, FILL_LIMITS, VALID_RANGES, build_hourly_grid

VITAL_ITEMS = {220045: "heart_rate", 220210: "resp_rate", 220277: "spo2", 220179: "sbp"}
TEMP_ITEM = 223762
LAB_ITEMS = {50813: "lactate", 50912: "creatinine", 51301: "wbc"}

# data_exploration.ipynb cell 98 output: TRAIN-derived IQR bounds of the run
# the checkpoint was trained on. Cell 102 then fills NaN with 0 and fits
# MinMax, so data_min = 0 and data_max = upper bound for every feature.
IQR_BOUNDS = {
    "resp_rate": (4.0, 36.0), "spo2": (89.0, 105.0),
    "heart_rate": (35.5, 135.5), "sbp": (58.421051, 177.298248),
    "temperature": (34.7, 39.5), "lactate": (-0.2, 3.0),
    "creatinine": (-0.65, 2.95), "wbc": (-2.1, 24.3),
}
LOOKBACK = 24
DECIMALS = {"resp_rate": 1, "spo2": 1, "heart_rate": 1, "sbp": 1,
            "temperature": 2, "wbc": 2, "lactate": 2, "creatinine": 2}

# Chronological split boundaries printed by data_exploration.ipynb cell 92.
TRAIN_END = pd.Timestamp("2169-10-25 23:26:00")
VAL_END = pd.Timestamp("2181-10-25 12:10:00")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------- extraction
def extract(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    icu = pd.read_csv(raw / "icu" / "icustays.csv.gz", parse_dates=["intime", "outtime"])

    ce = pd.read_csv(raw / "icu" / "chartevents.csv.gz",
                     usecols=["stay_id", "charttime", "itemid", "valuenum"])
    ce = ce[ce["itemid"].isin(list(VITAL_ITEMS) + [TEMP_ITEM])].dropna(subset=["valuenum"])
    ce["hour"] = pd.to_datetime(ce["charttime"]).dt.floor("h")

    vit = (ce[ce["itemid"].isin(VITAL_ITEMS)]
           .groupby(["stay_id", "hour", "itemid"])["valuenum"].median()
           .unstack("itemid").rename(columns=VITAL_ITEMS).reset_index())
    vit.columns.name = None
    temp = (ce[ce["itemid"] == TEMP_ITEM].groupby(["stay_id", "hour"])["valuenum"]
            .median().rename("temperature").reset_index())

    lab = pd.read_csv(raw / "hosp" / "labevents.csv.gz",
                      usecols=["hadm_id", "charttime", "itemid", "valuenum"])
    lab = lab[lab["itemid"].isin(LAB_ITEMS)].dropna(subset=["hadm_id", "charttime", "valuenum"]).copy()
    lab["hadm_id"] = lab["hadm_id"].astype("int64")
    lab["charttime"] = pd.to_datetime(lab["charttime"])
    lab["feature"] = lab["itemid"].map(LAB_ITEMS)
    lab = lab.merge(icu[["stay_id", "hadm_id", "intime", "outtime"]], on="hadm_id", how="inner")
    lab = lab[(lab["charttime"] >= lab["intime"]) & (lab["charttime"] <= lab["outtime"])]
    lab["hour"] = lab["charttime"].dt.floor("h")
    labs = lab.pivot_table(index=["stay_id", "hour"], columns="feature",
                           values="valuenum", aggfunc="median").reset_index()
    labs.columns.name = None

    # The hourly table is defined by the vital-sign rows (cell 75 left joins).
    df = vit.merge(temp, on=["stay_id", "hour"], how="left").merge(labs, on=["stay_id", "hour"], how="left")
    for f in rb.FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    return icu, df[["stay_id", "hour"] + rb.FEATURES]


# ---------------------------------------------------------- display pipeline
def display_records(filled: pd.DataFrame) -> dict:
    """Corrected-pipeline values and statuses for one stay (already sorted)."""
    out = {}
    for f in rb.FEATURES:
        vals, flags = filled[f].to_numpy(), filled[f"{f}_missing"].to_numpy()
        v_out, s_out = [], []
        for v, m in zip(vals, flags):
            if m == 0:
                v_out.append(round(float(v), DECIMALS[f])); s_out.append("o")
            elif not np.isnan(v):
                v_out.append(round(float(v), DECIMALS[f])); s_out.append("c")
            else:
                v_out.append(None); s_out.append("m")
        out[f] = {"v": v_out, "s": "".join(s_out)}
    return out


# ------------------------------------------------------------ model encoding
def model_inputs(grid_raw: pd.DataFrame, node_features: list[str]) -> np.ndarray:
    """Scaled values + missing flags in the encoding the checkpoint was trained on,
    using past observations only. Returns [H, 8, 2] (value, missing)."""
    vals, miss = [], []
    for f in node_features:
        x = grid_raw[f]
        miss.append(x.isna().to_numpy(dtype=np.float32))
        x = x.ffill(limit=3).ffill()          # causal stand-in for bidirectional interpolation
        lo, hi = IQR_BOUNDS[f]
        x = x.clip(lo, hi).fillna(0.0) / hi   # cells 98-102: cap, NaN -> 0, MinMax(min 0, max hi)
        vals.append(x.to_numpy(dtype=np.float32))
    return np.stack([np.stack(vals, 1), np.stack(miss, 1)], -1)


def colab_dense(vm: np.ndarray) -> np.ndarray:
    """PhysioGraph_Final_Colab.ipynb cell 5, verbatim arithmetic, single stay."""
    val_block = vm[:, :, 0].copy()
    mis_block = vm[:, :, 1].copy()
    hr_bpm = val_block[:, 2] * 18.14 + 85.99
    sbp_mmHg = np.clip(val_block[:, 3] * 21.66 + 119.84, 40.0, 250.0)
    shock_index = np.clip(hr_bpm / sbp_mmHg, 0.2, 3.0)
    norm_si = (shock_index - 0.70) / 0.25
    val_block[:, 2] = val_block[:, 2] * 0.75 + norm_si * 0.25
    val_block[:, 3] = val_block[:, 3] * 0.75 - norm_si * 0.25
    delta_1h = np.zeros_like(val_block)
    delta_4h = np.zeros_like(val_block)
    delta_1h[1:] = val_block[1:] - val_block[:-1]
    if len(val_block) > 4:
        delta_4h[4:] = (val_block[4:] - val_block[:-4]) / 4.0
    return np.stack([val_block, mis_block, delta_1h, delta_4h], axis=-1).astype(np.float32)


def load_model(notebook: Path, ckpt: Path):
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}
    if "EDGES_DIRECTED" not in src[4] or "class PhysioGraphMultiHorizonPro" not in src[6]:
        raise SystemExit("Notebook cells 4/6 are not the graph/model cells this script expects.")
    ns: dict = {"torch": torch}
    exec(compile(src[4], f"{notebook.name}:cell4", "exec"), ns)
    exec(compile(src[6], f"{notebook.name}:cell6", "exec"), ns)
    model = ns["PhysioGraphMultiHorizonPro"](edge_index=ns["EDGE_INDEX"], in_channels=4)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    meta = {k: (float(v) if isinstance(v, float) else v) for k, v in ck.items() if k != "model_state"}
    return model, ns["NODE_FEATURES"], ns["EDGES_DIRECTED"], meta


def score_stay(model, dense: np.ndarray, hours_in_icu: np.ndarray, batch: int = 64):
    H = len(dense)
    ends = list(range(LOOKBACK - 1, H))
    scores, shares = [], []
    for s in range(0, len(ends), batch):
        idx = ends[s:s + batch]
        xb = torch.from_numpy(np.stack([dense[e - LOOKBACK + 1:e + 1] for e in idx])).requires_grad_(True)
        hb = torch.tensor(hours_in_icu[idx], dtype=torch.float32)
        l8, l6, l12, _ = model(xb, hb)
        sc = 0.70 * l8 + 0.15 * l6 + 0.15 * l12
        (gx,) = torch.autograd.grad(sc.sum(), [xb])
        sens = gx.abs().sum(dim=(1, 3))                       # [B, 8], as run_attribution2.py
        shares.append((sens / sens.sum(1, keepdim=True)).detach().numpy())
        scores.append(torch.sigmoid(sc).detach().numpy())
    if not ends:
        return [], []
    return np.concatenate(scores).tolist(), np.concatenate(shares).tolist()


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", default="data/raw/mimic-iv-demo")
    ap.add_argument("--ckpt", default="models/best_physiograph_multi_horizon_1_1.pt")
    ap.add_argument("--notebook", default="PhysioGraph_Final_Colab.ipynb")
    ap.add_argument("--out", default="dashboard/js/data/mimic-demo-cases.js")
    args = ap.parse_args()

    raw, t0 = Path(args.raw), time.time()
    torch.manual_seed(0)
    torch.set_grad_enabled(True)

    icu, df = extract(raw)
    print(f"Extracted {len(df):,} hourly rows from {df['stay_id'].nunique()} stays")

    # Corrected pipeline (display): grid -> clip -> flags -> forward-only fill.
    filled = rb.prepare_split(df, "demo")

    # Pre-correction encoding (model input): same grid, no plausibility clip.
    df = df.sort_values(["stay_id", "hour"]).drop_duplicates(["stay_id", "hour"], keep="last")
    grid_raw = rb.build_hourly_grid(df).merge(df, on=["stay_id", "hour"], how="left")

    model, node_features, edges, ck_meta = load_model(Path(args.notebook), Path(args.ckpt))
    if [e for e in edges] != [(0, 1), (1, 2), (2, 3), (3, 7), (6, 7), (4, 5), (5, 6), (0, 6), (2, 6), (1, 6),
                              (1, 0), (3, 2), (6, 0), (7, 3), (6, 3)]:
        raise SystemExit("Notebook edge list differs from the documented 15 edges.")
    print(f"Loaded checkpoint: epoch {ck_meta.get('epoch')}, val AUPRC {ck_meta.get('best_val_auprc'):.4f}")

    icu = icu.set_index("stay_id")
    cases = []
    for sid, g in filled.groupby("stay_id", sort=False):
        g = g.sort_values("hour").reset_index(drop=True)
        gr = grid_raw[grid_raw["stay_id"] == sid].sort_values("hour").reset_index(drop=True)
        assert (g["hour"].to_numpy() == gr["hour"].to_numpy()).all()
        info = icu.loc[sid]
        first = g["hour"].iloc[0]
        offset = max(0, int(round((first - info["intime"]).total_seconds() / 3600)))
        hours_in_icu = ((g["hour"] - info["intime"]).dt.total_seconds() / 3600).to_numpy(dtype=np.float32)

        dense = colab_dense(model_inputs(gr, node_features))
        scores, shares = score_stay(model, dense, hours_in_icu)
        split = "train" if info["intime"] <= TRAIN_END else "val" if info["intime"] <= VAL_END else "test"

        cases.append({
            "stay_id": int(sid),
            "careunit": str(info["first_careunit"]),
            "los_days": round(float(info["los"]), 2),
            "intime": info["intime"].isoformat(),
            "t0": first.isoformat(),
            "offset": offset,
            "n": len(g),
            "split": split,
            "vars": display_records(g),
            "pred_start": LOOKBACK - 1,
            "scores": [round(s, 4) for s in scores],
            "attr": [[int(round(x * 1000)) for x in row] for row in shares],
        })
        print(f"  stay {sid}: {len(g):4d} h, {len(scores):4d} scores, likely {split}", flush=True)

    files = {f: sha256(raw / f) for f in ("icu/icustays.csv.gz", "icu/chartevents.csv.gz", "hosp/labevents.csv.gz")}
    meta = {
        "dataset": "MIMIC-IV Clinical Database Demo",
        "version": "2.2",
        "source": "https://physionet.org/content/mimic-iv-demo/2.2/",
        "license": "Open Data Commons Open Database License v1.0",
        "citation": "Johnson A, Bulgarelli L, Pollard T, Horng S, Celi LA, Mark R. MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet (2023). https://doi.org/10.13026/dp1f-ex47",
        "files_sha256": files,
        "built": time.strftime("%Y-%m-%d"),
        "stays": len(cases),
        "checkpoint": {"file": Path(args.ckpt).name, **ck_meta},
        "model_code": f"{Path(args.notebook).name} cells 4 and 6 (exec'd verbatim)",
        "attr_node_order": node_features,
        "split_boundaries": {"train_end": str(TRAIN_END), "val_end": str(VAL_END)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "/* Generated by scripts/build_demo_cases.py - do not edit.\n"
        " * Real ICU stays from the MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL v1.0).\n"
        " * Scores: pre-correction checkpoint, past-only inputs. See the script docstring. */\n"
        f"window.PG = window.PG || {{}};\nPG.DEMO_META = {json.dumps(meta)};\n"
        f"PG.DEMO_RAW = {json.dumps(cases, separators=(',', ':'))};\n",
        encoding="utf-8")
    print(f"\nWrote {len(cases)} stays -> {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
