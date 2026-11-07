"""
xgb_v2.py - is there ANY learnable signal under the v2 labels?

Before spending hours training the GNN on a new label definition, find
out in minutes whether a gradient-boosted model can predict it at all.

For every window (ending at hour h, using at most the last 24 h and at
least 6 h of history) it builds, per variable, from OBSERVED values only
(missing values are NaN, which XGBoost handles natively):
    last observed value (within the window), mean, std, min, max,
    fraction of the window that was missing, hours since last observation
plus hours-since-admission. It then trains one model per
(event set, horizon) and reports AUROC / AUPRC / lift on the validation
split, next to a TIME-ONLY baseline (hours since admission alone) so you
can see how much the physiology adds beyond "early in the stay is risky".

Notes
  * Training negatives are subsampled (--neg-frac) to save memory; every
    positive is kept and rows are re-weighted by 1/keep-probability, so
    the loss matches the true prevalence (no class weighting is applied).
  * Early stopping uses the validation split, so validation numbers are
    very slightly optimistic. Keep the TEST split untouched until the
    configuration is final.
  * Stay-clustered bootstrap CIs are computed only for --primary-h.

Smoke test (about a minute):
  python xgb_v2.py --max-stays 1500 --horizons 8 --n-boot 20
Full run:
  python xgb_v2.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from labels_v2 import (CONFIGS, HORIZONS, MIN_HIST, attach_hours, compute_labels,
                       load_events, load_intime)

FEATS = ["resp_rate", "spo2", "heart_rate", "sbp",
         "temperature", "lactate", "creatinine", "wbc"]
WINDOW = 24
PAD = WINDOW - 1
STATS = ["last", "mean", "std", "min", "max", "miss_rate", "hrs_since"]


# ----------------------------------------------------------------------
# window features
# ----------------------------------------------------------------------

def window_features(chunk):
    """
    chunk: DataFrame sorted by (stay_id, hours), whole stays.
    Returns X (n_rows x n_features, float32) and the feature names.
    Each stay is front-padded with 23 NaN rows so a global rolling(24)
    can never mix two stays.
    """
    n = len(chunk)
    stay = chunk["stay_id"].to_numpy()
    uniq, counts = np.unique(stay, return_counts=True)
    rank = np.repeat(np.arange(len(uniq)), counts)
    real_pos = np.arange(n) + PAD * (rank + 1)
    total = n + PAD * len(uniq)
    padded_stay = np.repeat(uniq, counts + PAD)
    p = np.arange(total, dtype=np.float64)

    cols, names = [], []
    for f in FEATS:
        x = chunk[f].to_numpy(dtype=np.float32)
        m = chunk[f + "_missing"].to_numpy()
        xm = np.where(m == 1, np.nan, x).astype(np.float32)

        px = np.full(total, np.nan, dtype=np.float32)
        px[real_pos] = xm
        pm = np.full(total, np.nan, dtype=np.float32)
        pm[real_pos] = m.astype(np.float32)

        sx = pd.Series(px)
        roll = sx.rolling(WINDOW, min_periods=1)
        mean = roll.mean().to_numpy()[real_pos]
        std = roll.std().to_numpy()[real_pos]
        mn = roll.min().to_numpy()[real_pos]
        mx = roll.max().to_numpy()[real_pos]
        miss = pd.Series(pm).rolling(WINDOW, min_periods=1).mean().to_numpy()[real_pos]

        last_val = sx.groupby(padded_stay).ffill().to_numpy()[real_pos]
        obs_pos = np.where(~np.isnan(px), p, np.nan)
        last_pos = pd.Series(obs_pos).groupby(padded_stay).ffill().to_numpy()[real_pos]
        hs = p[real_pos] - last_pos
        in_window = ~np.isnan(hs) & (hs < WINDOW)
        last_val = np.where(in_window, last_val, np.nan)
        hs = np.where(in_window, hs, float(WINDOW))

        for arr, s in zip((last_val, mean, std, mn, mx, miss, hs), STATS):
            cols.append(arr.astype(np.float32))
            names.append(f"{f}_{s}")

    cols.append(chunk["hours"].to_numpy(dtype=np.float32))
    names.append("hour_in_icu")
    return np.column_stack(cols).astype(np.float32), names


# ----------------------------------------------------------------------
# build a split
# ----------------------------------------------------------------------

def build_split(path, icu, events, configs, horizons, keep_neg_frac,
                chunk_stays, max_stays, seed, tag):
    t0 = time.time()
    cols = ["stay_id", "hour"] + FEATS + [f + "_missing" for f in FEATS]
    df = pd.read_parquet(path, columns=cols)
    df["stay_id"] = df["stay_id"].astype("int64")
    df = attach_hours(df, icu)
    df = df.sort_values(["stay_id", "hours"]).reset_index(drop=True)

    sid = df["stay_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    ends = np.r_[starts[1:], len(df)]
    if max_stays:
        starts, ends = starts[:max_stays], ends[:max_stays]
    n_stays = len(starts)
    print(f"[{tag}] {n_stays:,} stays, building features in chunks of {chunk_stays:,} stays")

    rng = np.random.default_rng(seed)
    keys = [(c, H) for c in configs for H in horizons]
    Xp, wp, sp = [], [], []
    Y = {k: [] for k in keys}
    V = {k: [] for k in keys}
    names = None

    for c0 in range(0, n_stays, chunk_stays):
        c1 = min(c0 + chunk_stays, n_stays)
        chunk = df.iloc[starts[c0]:ends[c1 - 1]].reset_index(drop=True)
        X, names = window_features(chunk)
        _, labs, _ = compute_labels(chunk, events, configs=configs, horizons=horizons)

        n = len(chunk)
        any_valid = np.zeros(n, dtype=bool)
        any_pos = np.zeros(n, dtype=bool)
        for k in keys:
            y, v = labs[k]
            any_valid |= v
            any_pos |= (v & (y == 1))

        if keep_neg_frac is None:
            keep = any_valid
            w = np.ones(n, dtype=np.float32)
        else:
            u = rng.random(n)
            keep = any_valid & (any_pos | (u < keep_neg_frac))
            w = np.where(any_pos, 1.0, 1.0 / keep_neg_frac).astype(np.float32)

        idx = np.flatnonzero(keep)
        Xp.append(X[idx])
        wp.append(w[idx])
        sp.append(chunk["stay_id"].to_numpy()[idx])
        for k in keys:
            y, v = labs[k]
            Y[k].append(y[idx])
            V[k].append(v[idx])
        print(f"  [{tag}] stays {c1:,}/{n_stays:,}  kept rows so far "
              f"{sum(len(a) for a in Xp):,}  ({time.time() - t0:.0f}s)")

    out = {
        "X": np.vstack(Xp), "w": np.concatenate(wp), "stay": np.concatenate(sp),
        "Y": {k: np.concatenate(Y[k]) for k in keys},
        "V": {k: np.concatenate(V[k]) for k in keys},
        "names": names,
    }
    del df
    return out


# ----------------------------------------------------------------------
# fit / evaluate
# ----------------------------------------------------------------------

def boot_ci(y, p, stays, n_boot, seed):
    groups = pd.DataFrame({"s": stays}).groupby("s").indices
    keys = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    a_l, p_l = [], []
    for _ in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in chosen])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a_l.append(roc_auc_score(yy, p[idx]))
        p_l.append(average_precision_score(yy, p[idx]))
    if not a_l:
        return (np.nan,) * 4
    return (np.percentile(a_l, 2.5), np.percentile(a_l, 97.5),
            np.percentile(p_l, 2.5), np.percentile(p_l, 97.5))


def fit_eval(tr, va, key, feat_idx, params, n_boot, seed):
    vtr, vva = tr["V"][key], va["V"][key]
    ytr = tr["Y"][key][vtr]
    yva = va["Y"][key][vva]
    if ytr.sum() < 30 or yva.sum() < 30:
        return None, None
    Xtr = tr["X"][vtr][:, feat_idx]
    wtr = tr["w"][vtr]
    Xva = va["X"][vva][:, feat_idx]

    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    p = model.predict_proba(Xva)[:, 1]

    prev = float(yva.mean())
    auroc = roc_auc_score(yva, p)
    auprc = average_precision_score(yva, p)
    res = {
        "train_positives": int(ytr.sum()),
        "train_windows_est": int(round(float(wtr.sum()))),
        "val_windows": int(len(yva)),
        "val_positives": int(yva.sum()),
        "val_prevalence_%": 100 * prev,
        "AUROC": auroc, "AUPRC": auprc, "lift": auprc / prev if prev > 0 else np.nan,
    }
    if n_boot > 0:
        lo_a, hi_a, lo_p, hi_p = boot_ci(yva, p, va["stay"][vva], n_boot, seed)
        res.update({"AUROC_lo": lo_a, "AUROC_hi": hi_a, "AUPRC_lo": lo_p, "AUPRC_hi": hi_p})
    return res, model


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-features", default="data/processed/checkpoints/train_normalized.parquet")
    ap.add_argument("--val-features", default="data/processed/checkpoints/val_normalized.parquet")
    ap.add_argument("--intime", default="data/processed/icu_intime.parquet")
    ap.add_argument("--events", default="data/processed/composite_events.csv")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--primary-h", type=int, default=8)
    ap.add_argument("--neg-frac", type=float, default=0.15)
    ap.add_argument("--chunk-stays", type=int, default=6000)
    ap.add_argument("--max-stays", type=int, default=None, help="smoke test: use only the first N stays per split")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/v2_xgb_results.csv")
    a = ap.parse_args()

    configs = {c: CONFIGS[c] for c in a.configs}
    horizons = tuple(a.horizons)

    icu = load_intime(a.intime)
    events = load_events(a.events)

    tr = build_split(a.train_features, icu, events, configs, horizons,
                     a.neg_frac, a.chunk_stays, a.max_stays, a.seed, "train")
    va = build_split(a.val_features, icu, events, configs, horizons,
                     None, a.chunk_stays, a.max_stays, a.seed + 1, "val")
    names = tr["names"]
    all_idx = list(range(len(names)))
    time_idx = [names.index("hour_in_icu")]

    full_params = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                       early_stopping_rounds=30, n_jobs=-1, random_state=a.seed)
    time_params = dict(full_params, n_estimators=100, max_depth=3, early_stopping_rounds=10)

    rows = []
    importance = None
    for cfg in configs:
        for H in horizons:
            key = (cfg, H)
            t0 = time.time()
            nb = a.n_boot if H == a.primary_h else 0
            res, model = fit_eval(tr, va, key, all_idx, full_params, nb, a.seed)
            if res is None:
                print(f"{cfg:11s} H={H:>2}: too few positives, skipped")
                continue
            base, _ = fit_eval(tr, va, key, time_idx, time_params, 0, a.seed)
            res["time_only_AUROC"] = base["AUROC"] if base else np.nan
            res["time_only_AUPRC"] = base["AUPRC"] if base else np.nan
            rows.append({"config": cfg, "H": H, **res})
            ci = ""
            if nb > 0:
                ci = (f"  AUROC CI [{res['AUROC_lo']:.3f},{res['AUROC_hi']:.3f}]"
                      f"  AUPRC CI [{res['AUPRC_lo']:.4f},{res['AUPRC_hi']:.4f}]")
            print(f"{cfg:11s} H={H:>2}: val pos={res['val_positives']:>6,} "
                  f"prev={res['val_prevalence_%']:.2f}%  AUROC={res['AUROC']:.4f} "
                  f"AUPRC={res['AUPRC']:.4f} lift={res['lift']:.2f}x | "
                  f"time-only AUROC={res['time_only_AUROC']:.4f} "
                  f"({time.time() - t0:.0f}s){ci}")
            if cfg == "all4" and H == a.primary_h:
                importance = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])[:12]

    table = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)

    print("\n" + "=" * 90)
    print("SUMMARY (validation split)")
    print("=" * 90)
    show = ["config", "H", "val_positives", "val_prevalence_%", "AUROC", "AUPRC", "lift",
            "time_only_AUROC", "time_only_AUPRC"]
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if importance:
        print("\nTop features, all4 / primary horizon:")
        for n_, v_ in importance:
            print(f"  {n_:24s} {v_:.4f}")
    print(f"\nSaved {a.out}")
    print("""
How to read this:
  * AUROC well above time_only_AUROC  -> the physiology carries real warning signal.
  * AUROC ~ time_only_AUROC           -> the 8 variables add little beyond 'when in the stay'.
  * Compare configs: if 'nosepsis' is about as predictable as 'all4', the sepsis proxy is not what
    drives the result; if only 'sepsisonly' is predictable, say so in the report.
  * Compare lift (AUPRC / prevalence), not raw AUPRC, across horizons.""")


if __name__ == "__main__":
    main()