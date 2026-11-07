"""
labels_v2.py - deterioration labels that define a learnable task.

WHAT WAS WRONG WITH v1 (measured on the validation split)
---------------------------------------------------------
  * 71% of event stays had their FIRST event within 6 h of ICU admission
    (median ventilation start 1.4 h, vasopressor 1.6 h). Those are
    admission states, not deterioration.
  * v1 kept only the first event and dropped every row after it, so a
    patient ventilated at admission could never contribute a later,
    real deterioration.
  * v1 labelled a single 2-hour slot (6-8 h ahead) positive, and the
    6 hours right before an event were labelled 0.
  * The 24-row rule removed ~80% of the remaining event stays.

v2 DEFINITION
-------------
For a window ending at hour h, and each event type X in the chosen set:
  * X is "at risk" at h if X has not started yet (no onset, or onset > h).
    Events already present at h are NOT a label - that is the patient's
    current state.
  * the window is POSITIVE for horizon H if some at-risk type has its
    onset in (h, h + H].
  * windows where nothing is left at risk are dropped (nothing to predict).
  * a window with no event is only kept if the stay was observed for the
    whole horizon (h + H <= last observed hour); otherwise the outcome is
    unknown and the window is dropped. A window followed by an event is
    always kept (the event is known even if the stay ends).
  * windows need at least MIN_HIST (6) hourly rows of history; the model
    looks back at most 24 h. Stays are NOT cut at their first event.

HORIZON vs WINDOW
-----------------
The 24 h figure is the LOOKBACK (how much history the model sees). The
HORIZON H is how far ahead we predict. The original plan was 6-8 h ahead,
so H=8 is the primary horizon (as "event within the next 8 h"); 6, 12 and
24 are reported as sensitivity checks.

EVENT SETS (so the sepsis question is answered by data, not by opinion)
-----------------------------------------------------------------------
  all4        vasopressor + ventilation + death + sepsis-proxy
  nosepsis    vasopressor + ventilation + death
  sepsisonly  sepsis-proxy only

USAGE (audit only - fast, no model):
  python labels_v2.py --features data/processed/checkpoints/val_normalized.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


# ----------------------------------------------------------------------
# loading helpers
# ----------------------------------------------------------------------

def load_intime(path):
    icu = pd.read_parquet(path)
    for c in ("stay_id", "intime"):
        if c not in icu.columns:
            raise ValueError(f"{path} is missing column '{c}'. Columns: {list(icu.columns)}")
    icu = icu[["stay_id", "intime"]].copy()
    icu["stay_id"] = pd.to_numeric(icu["stay_id"]).astype("int64")
    icu["intime"] = pd.to_datetime(icu["intime"])
    return icu.drop_duplicates("stay_id")


def attach_hours(df, icu):
    """ICU-relative hours, computed exactly as dataset.py does."""
    df = df.merge(icu, on="stay_id", how="left")
    if df["intime"].isna().any():
        n = int(df["intime"].isna().sum())
        raise ValueError(f"{n} rows have no ICU intime; check icu_intime.parquet.")
    df["hours"] = (pd.to_datetime(df["hour"]) - df["intime"]).dt.total_seconds() / 3600.0
    return df.drop(columns="intime")


def load_events(path):
    ev = pd.read_csv(path)
    if "stay_id" not in ev.columns:
        raise ValueError(f"{path} has no stay_id column.")
    out = pd.DataFrame({"stay_id": pd.to_numeric(ev["stay_id"]).astype("int64")})
    for name, col in EVENT_COLS.items():
        out[name] = pd.to_numeric(ev[col], errors="coerce") if col in ev.columns else np.nan
    return out.drop_duplicates("stay_id").set_index("stay_id")


# ----------------------------------------------------------------------
# label computation
# ----------------------------------------------------------------------

def compute_labels(df, events, configs=None, horizons=HORIZONS, min_hist=MIN_HIST):
    """
    df      : DataFrame sorted by (stay_id, hours), containing WHOLE stays,
              with columns stay_id and hours.
    events  : output of load_events().
    Returns (pos_in_stay, labels, onset) where
        labels[(config, H)] = (y int8 array, valid bool array)
        onset[type]         = onset hour per row (NaN if none)
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
    return pos, labels, onset


# ----------------------------------------------------------------------
# audit CLI
# ----------------------------------------------------------------------

def audit(features_path, intime_path, events_path):
    df = pd.read_parquet(features_path, columns=["stay_id", "hour"])
    df["stay_id"] = df["stay_id"].astype("int64")
    df = attach_hours(df, load_intime(intime_path))
    df = df.sort_values(["stay_id", "hours"]).reset_index(drop=True)
    ev = load_events(events_path)

    pos, labels, onset = compute_labels(df, ev)
    stays = df["stay_id"].to_numpy()

    print("=" * 80)
    print(f"v2 LABEL AUDIT  {features_path}")
    print("=" * 80)
    print(f"stays: {df['stay_id'].nunique():,}   hourly rows: {len(df):,}   "
          f"rows with >= {MIN_HIST}h history: {int((pos >= MIN_HIST - 1).sum()):,}")

    rows = []
    for (cfg, H), (y, v) in labels.items():
        rows.append({
            "config": cfg, "H": H,
            "valid_windows": int(v.sum()),
            "positives": int(y[v].sum()),
            "prevalence_%": 100 * float(y[v].mean()) if v.any() else np.nan,
            "stays_valid": int(pd.Series(stays[v]).nunique()),
            "stays_with_positive": int(pd.Series(stays[v & (y == 1)]).nunique()),
        })
    print("\nCohort and prevalence per configuration / horizon")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n(v1 for reference: ~1,335 usable positives from ~236,021 windows "
          "= 0.57% prevalence, ~600 stays with a positive.)")

    # which event types are 'already present' before the first window can end
    first_eligible_hour = float(MIN_HIST - 1)
    st = df.groupby("stay_id")["hours"].min()
    print(f"\nEvent onsets before the first eligible window (hour ~{first_eligible_hour:.0f}) "
          f"are treated as 'present at admission' and never become a label:")
    for t in EVENT_COLS:
        o = ev[t].reindex(st.index)
        has = o.notna()
        early = (o < first_eligible_hour) & has
        print(f"  {t:12s} stays with event: {int(has.sum()):6,}   "
              f"started before hour {first_eligible_hour:.0f}: {int(early.sum()):6,} "
              f"({100 * early.sum() / max(has.sum(), 1):.1f}%)")

    # composition of positives for the primary setting
    y, v = labels[("all4", 8)]
    h = df["hours"].to_numpy()
    print("\nComposition of positives for all4, H=8 (a row can count for >1 type):")
    for t in EVENT_COLS:
        o = onset[t]
        pt = v & (y == 1) & ~np.isnan(o) & (o > h) & (o <= h + 8)
        print(f"  {t:12s} {int(pt.sum()):8,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/processed/checkpoints/val_normalized.parquet")
    ap.add_argument("--intime", default="data/processed/icu_intime.parquet")
    ap.add_argument("--events", default="data/processed/composite_events.csv")
    a = ap.parse_args()
    audit(a.features, a.intime, a.events)