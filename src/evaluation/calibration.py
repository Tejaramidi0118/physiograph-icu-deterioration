"""
src/evaluation/calibration.py

Clinical Probability Recalibration & Reliability Metrics:
1. Saerens Bayesian Prior-Shift Recalibration:
   Transforms probabilities trained on 1:1 case-control balanced cohorts (prevalence = 50%)
   to match real-world unselected hospital population prevalence (e.g., 1.30%), preventing alarm fatigue.
2. Expected Calibration Error (ECE):
   Quantifies difference between predicted confidence and empirical event rate across quantile/uniform bins.
3. Brier Calibration Score:
   Strictly proper scoring rule measuring both discrimination and calibration accuracy.
"""
import numpy as np
from sklearn.metrics import brier_score_loss


def saerens_prior_shift(y_prob, p_train=0.50, p_real=0.01299):
    """
    Applies Saerens et al. (2002) Bayesian prior-shift recalibration formula:
    P_real(Y=1|x) = [ (p_real / p_train) * P_train ] / [ (p_real / p_train) * P_train + ((1 - p_real)/(1 - p_train)) * (1 - P_train) ]
    """
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    ratio_pos = p_real / p_train
    ratio_neg = (1.0 - p_real) / (1.0 - p_train)

    numerator = ratio_pos * p
    denominator = numerator + ratio_neg * (1.0 - p)
    return np.clip(numerator / denominator, 0.0, 1.0)


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) across n_bins.
    ECE = sum_k (|B_k| / N) * | acc(B_k) - conf(B_k) |
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_conf = np.mean(y_prob[in_bin])
            bin_acc = np.mean(y_true[in_bin])
            abs_err = np.abs(bin_acc - bin_conf)
            ece += (bin_count / n) * abs_err

            bin_stats.append({
                "bin": i,
                "range": (float(bin_lower), float(bin_upper)),
                "count": int(bin_count),
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
                "error": float(abs_err)
            })

    return float(ece), bin_stats


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Computes Brier score, ECE, and summary calibration metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = float(brier_score_loss(y_true, y_prob))
    ece, bin_stats = compute_ece(y_true, y_prob, n_bins=n_bins)

    return {
        "brier_score": brier,
        "ece": ece,
        "mean_predicted_risk": float(np.mean(y_prob)),
        "empirical_prevalence": float(np.mean(y_true)),
        "bin_stats": bin_stats
    }

