# train.py
# Reproducible training script: load data → preprocess → tune → train → save artifacts

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.neighbors import KDTree
import random

# --------------------------
# Reproducibility seeds
# --------------------------
random.seed(42)
np.random.seed(42)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


# --------------------------
# Load your dataset here
# Replace with real loading code
# --------------------------
def load_dataset():
    df = pd.read_csv("data/materials.csv")     # MODIFY THIS according to your dataset
    feature_cols = [c for c in df.columns if c not in ["is_superconducting", "Tc"]]
    X = df[feature_cols]
    y_class = df["is_superconducting"]
    y_reg = df["Tc"]
    return X, y_class, y_reg, feature_cols


# --------------------------
# Train + save
# --------------------------
def main():

    X, y_class, y_reg, feature_order = load_dataset()

    X_train, X_val, yc_train, yc_val, yr_train, yr_val = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    # --------------------------
    # Build preprocessing + models
    # --------------------------

    scaler = StandardScaler()

    clf = RandomForestClassifier(random_state=42)
    reg = GradientBoostingRegressor(random_state=42)

    # Hyperparameter tuning example
    clf_params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10]
    }

    reg_params = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [2, 3]
    }

    clf_pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    reg_pipe = Pipeline([("scaler", scaler), ("reg", reg)])

    clf_grid = GridSearchCV(clf_pipe, clf_params, cv=3, scoring="roc_auc")
    reg_grid = GridSearchCV(reg_pipe, reg_params, cv=3, scoring="neg_mean_squared_error")

    clf_grid.fit(X_train, yc_train)
    reg_grid.fit(X_train, yr_train)

    best_clf = clf_grid.best_estimator_
    best_reg = reg_grid.best_estimator_

    # --------------------------
    # Evaluation
    # --------------------------
    proba = best_clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(yc_val, proba)

    pred_tc = best_reg.predict(X_val)
    rmse = mean_squared_error(yr_val, pred_tc, squared=False)

    print("AUC:", auc)
    print("RMSE:", rmse)

    # --------------------------
    # Nearest-neighbor index + OOD threshold
    # --------------------------
    X_scaled = best_clf.named_steps["scaler"].transform(X)
    tree = KDTree(X_scaled)

    dists, idxs = tree.query(X_scaled, k=2)
    ood_threshold = float(np.percentile(dists[:, 1], 95))

    # --------------------------
    # Save artifacts
    # --------------------------
    joblib.dump(best_clf, ARTIFACT_DIR / "model_clf.joblib")
    joblib.dump(best_reg, ARTIFACT_DIR / "model_reg.joblib")
    joblib.dump(tree, ARTIFACT_DIR / "kdtree.joblib")
    
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump({
            "feature_order": feature_order,
            "ood_threshold": ood_threshold,
            "metrics": {"auc": auc, "rmse": rmse}
        }, f, indent=2)

    print("Training complete. Artifacts saved in /artifacts.")


if __name__ == "__main__":
    main()
