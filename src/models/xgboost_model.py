"""
src/models/xgboost_model.py

Gradient-Boosted Decision Trees model trained on 70 clinical interaction biomarkers.
Provides the non-linear clinical cutoff baseline for stacking ensemble.
"""
from pathlib import Path
import xgboost as xgb

def train_xgboost_classifier(
    X_train, y_train,
    X_val, y_val,
    n_estimators=350,
    max_depth=5,
    learning_rate=0.04,
    use_gpu=False
):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.80,
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu"
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def save_xgboost_model(model, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

def load_xgboost_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model
