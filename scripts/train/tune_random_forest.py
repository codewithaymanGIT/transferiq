"""
Small, bounded grid search over Random Forest hyperparameters, using the
same 5-fold CV methodology as train_baseline_models.py (KFold, same
preprocessing pipeline) for a fair, apples-to-apples comparison against
the current default (n_estimators=200, all else default).

Deliberately small (6 configs): with 185 training rows, an exhaustive
search would just be fitting noise. This is meant to check whether the
current defaults are reasonable, not to squeeze out a marginal win by
overfitting to this particular CV split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

from train_baseline_models import build_feature_matrix  # noqa: E402

CONFIGS = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},  # current default
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
]


def main() -> None:
    matrix_path = REPO_ROOT / "data" / "processed" / "training_matrix.parquet"
    df = pd.read_parquet(matrix_path)
    feature_df, numeric_cols = build_feature_matrix(df)
    y_log = df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)
    X = feature_df[numeric_cols + ["position"]]

    n_splits = max(2, min(5, len(df)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preprocess = ColumnTransformer(
        [("position_ohe", OneHotEncoder(handle_unknown="ignore"), ["position"])],
        remainder="passthrough",
    )

    print(f"Tuning on {len(df)} rows, {n_splits}-fold CV\n")
    print(f"{'Config':<55} {'Mean R2':>10} {'Mean MAE':>10}")
    results = []
    for cfg in CONFIGS:
        fold_r2, fold_mae = [], []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
            pipe = Pipeline(
                [("prep", preprocess), ("model", RandomForestRegressor(random_state=42, **cfg))]
            )
            pipe.fit(X_train, y_train)
            pred_log = pipe.predict(X_test)
            pred = np.expm1(pred_log)
            actual = np.expm1(y_test)
            fold_r2.append(r2_score(actual, pred))
            fold_mae.append(mean_absolute_error(actual, pred))
        mean_r2 = float(np.mean(fold_r2))
        mean_mae = float(np.mean(fold_mae))
        results.append((cfg, mean_r2, mean_mae))
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{label:<55} {mean_r2:>10.3f} {mean_mae:>10.2f}")

    best = max(results, key=lambda r: r[1])
    default_r2 = results[0][1]
    print(f"\nBest by mean R2: {best[0]} (R2={best[1]:.3f}, MAE={best[2]:.2f})")
    print(f"Current default: R2={default_r2:.3f}")
    if best[1] > default_r2 + 0.01:
        print("-> Meaningfully better than default. Worth adopting.")
    else:
        print("-> Not meaningfully better than the current default (within CV noise). Keeping defaults.")


if __name__ == "__main__":
    main()
