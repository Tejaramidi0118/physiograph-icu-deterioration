"""
src/evaluation/explainability.py

Clinical Explainability & Visual Evaluation Utilities:
- Comparative ROC & Precision-Recall curves.
- Dynamic ICU Patient Trajectory Tracking.
- Decision Curve Analysis (DCA Net Benefit).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def plot_comparative_roc_pr(test_labels, gnn_preds, xgb_preds, ensemble_preds, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    fpr_pg, tpr_pg, _ = roc_curve(test_labels, gnn_preds)
    fpr_xgb, tpr_xgb, _ = roc_curve(test_labels, xgb_preds)
    fpr_ens, tpr_ens, _ = roc_curve(test_labels, ensemble_preds)
    
    axes[0].plot(fpr_ens, tpr_ens, color="#e74c3c", lw=3.2, label="Clinical Ensemble 🏆")
    axes[0].plot(fpr_pg, tpr_pg, color="#2980b9", lw=2.4, label="PhysioGraph-Pro GNN")
    axes[0].plot(fpr_xgb, tpr_xgb, color="#f39c12", lw=2.0, linestyle="--", label="XGBoost")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Baseline (50%)")
    axes[0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    prec_pg, rec_pg, _ = precision_recall_curve(test_labels, gnn_preds)
    prec_xgb, rec_xgb, _ = precision_recall_curve(test_labels, xgb_preds)
    prec_ens, rec_ens, _ = precision_recall_curve(test_labels, ensemble_preds)
    
    axes[1].plot(rec_ens, prec_ens, color="#e74c3c", lw=3.2, label="Clinical Ensemble 🏆")
    axes[1].plot(rec_pg, prec_pg, color="#2980b9", lw=2.4, label="PhysioGraph-Pro GNN")
    axes[1].plot(rec_xgb, prec_xgb, color="#f39c12", lw=2.0, linestyle="--", label="XGBoost")
    axes[1].axhline(0.50, color="gray", linestyle=":", label="Balanced Baseline (50%)")
    axes[1].set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (PPV)", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    tn, fp, fn, tp = confusion_matrix(test_labels, (np.array(ensemble_preds) >= 0.5).astype(int), labels=[0, 1]).ravel()
    cm_data = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm_data / cm_data.sum(axis=1, keepdims=True) * 100.0
    axes[2].imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f"{cm_data[i, j]:,}\n({cm_norm[i, j]:.1f}%)", ha="center", va="center", fontsize=11, fontweight="bold",
                         color="white" if cm_norm[i, j] > 50 else "black")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Pred Stable", "Pred Crisis"])
    axes[2].set_yticklabels(["True Stable", "True Crisis"])
    axes[2].set_title("Ensemble Confusion Matrix", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_patient_trajectory(df_traj, stay_id, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_traj["hour"], df_traj["risk_pct"], color="#d62728", lw=2.8, marker="o", label="PhysioGraph Predicted Risk")
    ax.axhline(50.0, color="#e67e22", linestyle="--", lw=1.5, label="Clinical Action Threshold (50%)")
    ax.axhline(25.0, color="#f1c40f", linestyle=":", lw=1.5, label="Early Watchlist Alert (25%)")
    
    det_hours = df_traj[df_traj["label"] == 1]["hour"]
    if len(det_hours) > 0:
        min_det_h = det_hours.min()
        ax.axvspan(min_det_h, min_det_h + 8, color="#e74c3c", alpha=0.15, label="8-Hour Warning Window")
        ax.axvline(min_det_h, color="#c0392b", linestyle="-", lw=2, label=f"Earliest Advance Warning (Hr {int(min_det_h)})")
        
    ax.set_title(f"ICU Trajectory (Stay ID: {stay_id})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour in ICU", fontsize=10)
    ax.set_ylabel("Predicted Risk (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_decision_curve_analysis(y_true, ensemble_preds, gnn_preds, save_path=None):
    thresholds = np.linspace(0.10, 0.80, 50)
    net_benefits_ens, net_benefits_pg, net_benefits_all = [], [], []
    N_test = len(y_true)
    prevalence = np.mean(y_true)
    
    for th in thresholds:
        tp_e = np.sum((np.array(ensemble_preds) >= th) & (np.array(y_true) == 1))
        fp_e = np.sum((np.array(ensemble_preds) >= th) & (np.array(y_true) == 0))
        net_benefits_ens.append((tp_e / N_test) - (fp_e / N_test) * (th / max(1.0 - th, 1e-6)))
        
        tp_p = np.sum((np.array(gnn_preds) >= th) & (np.array(y_true) == 1))
        fp_p = np.sum((np.array(gnn_preds) >= th) & (np.array(y_true) == 0))
        net_benefits_pg.append((tp_p / N_test) - (fp_p / N_test) * (th / max(1.0 - th, 1e-6)))
        net_benefits_all.append(prevalence - (1.0 - prevalence) * (th / max(1.0 - th, 1e-6)))
        
    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds * 100, net_benefits_ens, color="#e74c3c", lw=3.2, label="Clinical Ensemble 🏆")
    plt.plot(thresholds * 100, net_benefits_pg, color="#2980b9", lw=2.4, label="PhysioGraph Alone")
    plt.plot(thresholds * 100, net_benefits_all, color="#7f8c8d", linestyle=":", lw=1.5, label="Treat All Strategy")
    plt.axhline(0, color="black", linestyle="-", lw=1.2, label="Treat None Strategy (Zero Benefit)")
    
    plt.title("Decision Curve Analysis (Clinical Net Benefit)", fontsize=12, fontweight="bold")
    plt.xlabel("Threshold Probability for Clinical Action (%)", fontsize=10)
    plt.ylabel("Net Benefit", fontsize=10)
    plt.xlim(10, 80)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
