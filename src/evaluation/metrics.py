"""
src/evaluation/metrics.py

Clinical & Machine Learning Evaluation Metrics Engine:
Computes AUROC, AUPRC, Prevalence, AUPRC Lift, Brier Score, Accuracy, Balanced Accuracy,
F1 Score, Sensitivity, Specificity, and optimal decision threshold tuning.
"""
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)

def compute_all_metrics(y_true, y_pred_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    
    auroc = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall, precision)
    brier = brier_score_loss(y_true, y_pred_prob)
    prevalence = np.mean(y_true)
    auprc_lift = auprc / max(prevalence, 1e-6)
    
    preds = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "auprc_lift": float(auprc_lift),
        "brier": float(brier),
        "accuracy": float(acc),
        "balanced_acc": float(bal_acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "prevalence": float(prevalence),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def optimize_decision_threshold(y_true, y_pred_prob):
    best_th, best_f1 = 0.50, -1.0
    for th in np.linspace(0.10, 0.90, 81):
        preds = (np.asarray(y_pred_prob) >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th), float(best_f1)
