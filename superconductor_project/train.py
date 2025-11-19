#!/usr/bin/env python3
"""
train.py

Reproducible training script for the Superconductor predictor.
Expected CSV: superconductor_project/data/materials.csv
The CSV must include a numeric "critical_temp" column and numeric descriptor columns.

Outputs saved to: superconductor_project/artifacts/
 - model_clf.joblib (Pipeline with scaler + classifier)   [if classifier trained]
 - model_reg.joblib (Pipeline with scaler + regressor)
 - kdtree.joblib
 - metadata.json
 - bg.npy (SHAP background)
"""
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.neighbors import KDTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error

# ----------------------------
# Configuration
# ----------------------------
SRC_CSV = Path("data/materials.csv")  # change if your CSV is elsewhere
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
RND = 42

# Reproducibility
random.seed(RND)
np.random.seed(RND)

# ----------------------------
# Data loader
# ----------------------------
def load_dataset(csv_path: Path = SRC_CSV):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    if "critical_temp" not in df.columns:
        raise KeyError("Required column 'critical_temp' not found in CSV")

    # regression target
    y_reg = df["critical_temp"].astype(float)

    # classification target: superconducting if Tc > 0 (adjust threshold if needed)
    THRESHOLD = 0.0
    y_class = (y_reg > THRESHOLD).astype(int)

    # feature columns: all except target
    feature_cols = [c for c in df.columns if c != "critical_temp"]

    # drop non-numeric feature columns if present (rare here)
    X = df[feature_cols].copy()
    # ensure numeric
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="raise")

    return X, y_class, y_reg, feature_cols


# ----------------------------
# Training function
# ----------------------------
def train_and_save():
    print("Loading dataset...")
    X, y_class, y_reg, feature_order = load_dataset()
    print(f"Loaded dataset: n_samples={len(X)}, n_features={X.shape[1]}")

    # split for evaluation
    X_train, X_val, yc_train, yc_val, yr_train, yr_val = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=RND, stratify=y_class
    )

    # scaler used inside pipelines
    scaler = StandardScaler()

    # ----------------------------
    # Classifier: handle class imbalance and safe CV
    # ----------------------------
    n_pos = int((yc_train == 1).sum())
    n_neg = int((yc_train == 0).sum())
    print(f"Training class counts (train split): pos={n_pos}, neg={n_neg}")

    if n_pos == 0:
        raise RuntimeError("No positive examples (critical_temp>0) in training data. Cannot train classifier.")

    # choose stratified splits safely: at most 5, at most number of positives
    n_splits = min(5, n_pos) if n_pos >= 2 else 2
    if n_splits < 2:
        n_splits = 2
    print(f"Using StratifiedKFold with n_splits={n_splits}")

    strat_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RND)

    clf_pipe = Pipeline([
        ("scaler", scaler),
        ("clf", RandomForestClassifier(random_state=RND, class_weight="balanced"))
    ])

    clf_params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10]
    }

    clf_grid = GridSearchCV(
        clf_pipe,
        param_grid=clf_params,
        cv=strat_cv,
        scoring="roc_auc",
        n_jobs=-1,
        error_score=np.nan,
        verbose=0,
    )

    print("Fitting classifier (GridSearchCV)...")
    clf_grid.fit(X_train, yc_train)
    best_clf = clf_grid.best_estimator_
    print("Best classifier params:", clf_grid.best_params_)

    # ----------------------------
    # Regressor: KFold CV
    # ----------------------------
    reg_pipe = Pipeline([
        ("scaler", scaler),
        ("reg", GradientBoostingRegressor(random_state=RND))
    ])

    reg_params = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [2, 3]
    }

    reg_cv = KFold(n_splits=5, shuffle=True, random_state=RND)
    reg_grid = GridSearchCV(
        reg_pipe,
        param_grid=reg_params,
        cv=reg_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        error_score=np.nan,
        verbose=0,
    )

    print("Fitting regressor (GridSearchCV)...")
    reg_grid.fit(X_train, yr_train)
    best_reg = reg_grid.best_estimator_
    print("Best regressor params:", reg_grid.best_params_)

    # ----------------------------
    # Evaluation on validation set
    # ----------------------------
    print("Evaluating on validation set...")
    try:
        y_proba_val = best_clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(yc_val, y_proba_val))
    except Exception as e:
        print("Classifier evaluation failed:", e)
        auc = None

    try:
        ypred_reg = best_reg.predict(X_val)
        rmse = float(mean_squared_error(yr_val, ypred_reg, squared=False))
    except Exception as e:
        print("Regressor evaluation failed:", e)
        rmse = None

    print(f"Validation metrics: ROC-AUC={auc}, RMSE={rmse}")

    # ----------------------------
    # Build KDTree for nearest-neighbor (on scaled full X)
    # ----------------------------
    print("Building KDTree on scaled descriptors...")
    # use scaler from classifier pipeline to scale the entire dataset (pipeline expected)
    if hasattr(best_clf, "named_steps") and "scaler" in best_clf.named_steps:
        X_scaled = best_clf.named_steps["scaler"].transform(X)
    else:
        # fallback: fit a fresh scaler on X
        tmp_scaler = StandardScaler().fit(X)
        X_scaled = tmp_scaler.transform(X)

    kdtree = KDTree(X_scaled, leaf_size=40)

    # compute OOD threshold: 95th percentile of nearest non-self distances
    dists, idxs = kdtree.query(X_scaled, k=2)
    min_nonself = dists[:, 1]
    ood_threshold = float(np.percentile(min_nonself, 95))
    print(f"OOD threshold (95th pct) = {ood_threshold:.6g}")

    # ----------------------------
    # Save artifacts
    # ----------------------------
    print("Saving artifacts to", ARTIFACT_DIR.resolve())
    joblib.dump(best_clf, ARTIFACT_DIR / "model_clf.joblib")
    joblib.dump(best_reg, ARTIFACT_DIR / "model_reg.joblib")
    joblib.dump(kdtree, ARTIFACT_DIR / "kdtree.joblib")

    metadata = {
        "feature_order": feature_order,
        "ood_threshold": ood_threshold,
        "n_samples": int(len(X)),
        "metrics": {"val_auc": auc, "val_rmse": rmse},
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    # SHAP background
    bg_size = min(100, X_scaled.shape[0])
    bg_idx = np.random.choice(X_scaled.shape[0], size=bg_size, replace=False)
    bg = X_scaled[bg_idx]
    np.save(ARTIFACT_DIR / "bg.npy", bg)
    print(f"Saved SHAP background (n={bg_size}) to artifacts/bg.npy")

    print("All artifacts saved. Done.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    train_and_save()

