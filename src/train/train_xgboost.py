"""
src/train/train_xgboost.py

Tabular Clinical Interaction Pipeline & XGBoost Classifier Training Engine.
Extracts 70 non-linear biomarkers across the 24h timeline and fits an optimized
Gradient-Boosted Decision Trees model.
"""
from pathlib import Path
import numpy as np
import torch

from ..data.biomarkers import extract_clinical_interaction_features
from ..models.xgboost_model import train_xgboost_classifier, save_xgboost_model, load_xgboost_model
from ..evaluation.metrics import compute_all_metrics


def train_xgboost_pipeline(
    train_dataset,
    val_dataset,
    test_dataset=None,
    max_train_samples=40000,
    n_estimators=350,
    max_depth=5,
    learning_rate=0.04,
    use_gpu=None,
    output_model_path="./models/best_xgboost_multi_horizon.json"
):
    """
    Extracts 70 clinical interaction biomarkers from PyTorch datasets, trains an
    XGBoost classifier, evaluates it on validation/test cohorts, and saves the model.
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    print("=" * 80)
    print("STARTING TABULAR CLINICAL INTERACTION BIOMARKER EXTRACTION FOR XGBOOST")
    print("=" * 80)

    print("Extracting validation biomarkers...")
    X_val, y_val = extract_clinical_interaction_features(val_dataset)

    print(f"Extracting training biomarkers (max_samples={max_train_samples})...")
    X_train, y_train = extract_clinical_interaction_features(train_dataset, max_samples=max_train_samples)

    X_test, y_test = None, None
    if test_dataset is not None:
        print("Extracting test biomarkers...")
        X_test, y_test = extract_clinical_interaction_features(test_dataset)

    print(f"✓ Enhanced Tabular Features Extracted: {X_train.shape[1]} clinical biomarkers per patient window.")
    print(f"Training XGBoost Model (GPU={use_gpu}, Trees={n_estimators}, MaxDepth={max_depth})...")

    model = train_xgboost_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_gpu=use_gpu
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    m_val = compute_all_metrics(y_val, val_preds)
    print(f"✓ XGBoost Val AUROC: {m_val['auroc']*100:.2f}% | Val AUPRC: {m_val['auprc']*100:.2f}%")

    test_preds = None
    m_test = None
    if X_test is not None:
        test_preds = model.predict_proba(X_test)[:, 1]
        m_test = compute_all_metrics(y_test, test_preds)
        print("=" * 80)
        print("✓ ADVANCED XGBOOST MODEL EVALUATION (Balanced Test Cohort)")
        print("=" * 80)
        print(f"XGBoost Test AUROC     : {m_test['auroc']*100:.2f}%")
        print(f"XGBoost Test AUPRC     : {m_test['auprc']*100:.2f}%")
        print(f"XGBoost Test Accuracy  : {m_test['accuracy']*100:.2f}%")
        print(f"XGBoost Test F1 Score  : {m_test['f1']*100:.2f}%")
        print("=" * 80)

    if output_model_path:
        save_xgboost_model(model, output_model_path)
        print(f"✓ Saved Advanced XGBoost Model to: {output_model_path}")

    return model, test_preds, m_test

