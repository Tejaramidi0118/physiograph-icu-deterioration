"""
src/evaluation/bootstrap.py

Stay-Clustered Non-Parametric Bootstrap for Statistical Significance:
Resamples distinct ICU stays (stay_id) with replacement across B=1,000 iterations.
Accounts for within-patient correlation across continuous prediction windows and derives
empirical 95% Confidence Intervals for AUROC and AUPRC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def run_stay_clustered_bootstrap(
    df,
    stay_col="stay_id",
    label_col="true_label",
    prob_col="ensemble_risk",
    n_bootstrap=1000,
    seed=42
):
    df_clean = df[[stay_col, label_col, prob_col]].dropna().copy()
    df_clean[label_col] = df_clean[label_col].astype(int)
    df_clean[prob_col] = df_clean[prob_col].astype(float)
    
    grouped = {
        sid: grp[[label_col, prob_col]].to_numpy(dtype=float)
        for sid, grp in df_clean.groupby(stay_col, sort=False)
    }
    stay_ids = np.array(list(grouped.keys()), dtype=object)
    n_stays = len(stay_ids)
    
    y_all = df_clean[label_col].to_numpy()
    p_all = df_clean[prob_col].to_numpy()
    
    point_estimates = {
        "auroc": float(roc_auc_score(y_all, p_all)),
        "auprc": float(average_precision_score(y_all, p_all))
    }
    
    rng = np.random.default_rng(seed)
    records = []
    
    for rep in range(n_bootstrap):
        sampled_ids = rng.choice(stay_ids, size=n_stays, replace=True)
        sample = np.concatenate([grouped[sid] for sid in sampled_ids], axis=0)
        y = sample[:, 0].astype(int)
        p = sample[:, 1]
        
        if len(np.unique(y)) != 2:
            continue
            
        records.append({
            "replicate": rep + 1,
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p))
        })
        
    boot_df = pd.DataFrame(records)
    
    summary = {
        "n_stays": int(n_stays),
        "n_windows": int(len(df_clean)),
        "point_estimates": point_estimates,
        "ci_95": {
            "auroc": {
                "lower": float(np.quantile(boot_df["auroc"], 0.025)),
                "upper": float(np.quantile(boot_df["auroc"], 0.975))
            },
            "auprc": {
                "lower": float(np.quantile(boot_df["auprc"], 0.025)),
                "upper": float(np.quantile(boot_df["auprc"], 0.975))
            }
        }
    }
    return summary, boot_df
