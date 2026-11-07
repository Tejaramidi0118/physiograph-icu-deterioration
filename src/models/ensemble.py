"""
src/models/ensemble.py

Grandmaster Clinical Stacking Ensemble:
Combines continuous spatio-temporal representations from PhysioGraph with
non-linear tabular decision trees from XGBoost.
Formula: P_ensemble = alpha * P_GNN + (1 - alpha) * P_XGBoost
"""
import numpy as np

class ClinicalStackingEnsemble:
    def __init__(self, alpha=0.60):
        self.alpha = alpha

    def predict_proba(self, gnn_probs, xgb_probs):
        gnn_probs = np.asarray(gnn_probs)
        xgb_probs = np.asarray(xgb_probs)
        return self.alpha * gnn_probs + (1.0 - self.alpha) * xgb_probs

    def predict(self, gnn_probs, xgb_probs, threshold=0.50):
        ensemble_probs = self.predict_proba(gnn_probs, xgb_probs)
        return (ensemble_probs >= threshold).astype(int)
