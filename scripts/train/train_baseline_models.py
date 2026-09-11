"""
First ML experiment: baseline -> Linear/Ridge -> Random Forest -> XGBoost,
on the real (if currently small) training matrix from build_training_matrix.py.

**Proof-of-concept caveat, load-bearing, not boilerplate:** with ~70 rows
(as of the first real run), these metrics have enormous variance and are
NOT representative of eventual model quality -- this script exists to
prove the full pipeline (leakage-safe features -> multiple real models ->
honestly-reported comparison) works end to end, not to produce a
deployable model. Re-run once more historical seasons are loaded.

All metrics printed/saved here come from actual sklearn/xgboost fits on
real data -- never hand-typed, per the project's "no fabricated metrics"
rule. If a run produces suspicious numbers, that's the model/data talking,
not something to "clean up" before reporting it.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]

_CANDIDATE_NUMERIC_FEATURES = [
    "minutes", "goals", "assists", "xg", "xa",
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
]

MIN_ROWS_FOR_TEST_SPLIT = 20
PROOF_OF_CONCEPT_ROW_THRESHOLD = 100  # below this, treat any comparison as illustrative only


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    median_ae: float
    r2: float
    n_train: int
    n_test: int


def usable_numeric_features(df: pd.DataFrame) -> list[str]:
    """Only keep candidate columns that exist and aren't entirely null --
    e.g. xg/xa are frequently all-null for a given FBref pull (see
    fbref_provider.py's documented limitation) and must be dropped rather
    than imputed."""
    return [c for c in _CANDIDATE_NUMERIC_FEATURES if c in df.columns and df[c].notna().any()]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = usable_numeric_features(df)
    feature_df = df[numeric_cols + ["position"]].copy()
    for col in numeric_cols:
        # Median imputation for sporadic nulls within an otherwise-usable
        # column (e.g. a handful of missing `assists`) -- documented, not
        # hidden, and only applied to columns that aren't entirely null.
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    return feature_df, numeric_cols


def _split(df: pd.DataFrame, y: pd.Series, numeric_cols: list[str]):
    n = len(df)
    if n < MIN_ROWS_FOR_TEST_SPLIT:
        logging.warning(
            "Only %d rows -- below the %d-row floor for a meaningful held-out test set. "
            "Splitting anyway (80/20) for illustration, but treat every metric below as "
            "highly unstable, not a real quality signal.",
            n, MIN_ROWS_FOR_TEST_SPLIT,
        )
    X = df[numeric_cols + ["position"]]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _evaluate(name: str, model, X_train, X_test, y_train_log, y_test_log) -> ModelResult:
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    # Evaluate in real EUR-millions space (expm1 of the log1p target) --
    # that's the unit a person reading this comparison actually cares about.
    y_test = np.expm1(y_test_log)
    pred = np.expm1(pred_log)
    return ModelResult(
        name=name,
        mae=mean_absolute_error(y_test, pred),
        rmse=root_mean_squared_error(y_test, pred),
        median_ae=median_absolute_error(y_test, pred),
        r2=r2_score(y_test, pred),
        n_train=len(X_train),
        n_test=len(X_test),
    )


def run_comparison(df: pd.DataFrame) -> list[ModelResult]:
    feature_df, numeric_cols = build_feature_matrix(df)
    y_log = df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)

    X_train, X_test, y_train, y_test = _split(feature_df, y_log, numeric_cols)

    preprocess = ColumnTransformer(
        [("position_ohe", OneHotEncoder(handle_unknown="ignore"), ["position"])],
        remainder="passthrough",
    )

    results = []
    results.append(_evaluate("Median baseline", DummyRegressor(strategy="median"), X_train, X_test, y_train, y_test))
    results.append(
        _evaluate(
            "Linear Regression",
            Pipeline([("prep", preprocess), ("model", LinearRegression())]),
            X_train, X_test, y_train, y_test,
        )
    )
    results.append(
        _evaluate(
            "Ridge",
            Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0))]),
            X_train, X_test, y_train, y_test,
        )
    )
    results.append(
        _evaluate(
            "Random Forest",
            Pipeline([("prep", preprocess), ("model", RandomForestRegressor(n_estimators=200, random_state=42))]),
            X_train, X_test, y_train, y_test,
        )
    )
    results.append(
        _evaluate(
            "XGBoost",
            Pipeline([("prep", preprocess), ("model", XGBRegressor(n_estimators=200, random_state=42, verbosity=0))]),
            X_train, X_test, y_train, y_test,
        )
    )
    return results


def write_comparison_report(
    results: list[ModelResult], n_total_rows: int, numeric_features: list[str], out_path: Path | None = None
) -> Path:
    lines = [
        "# Model comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`scripts/train/train_baseline_models.py` -- every number below comes from an "
        "actual sklearn/xgboost fit on real data, never hand-typed.",
        "",
    ]
    if n_total_rows < PROOF_OF_CONCEPT_ROW_THRESHOLD:
        lines += [
            "> **Proof-of-concept only.** This run used "
            f"**{n_total_rows} rows** -- far below what's needed for a trustworthy "
            "comparison. Treat every metric here as illustrating that the pipeline works, "
            "not as a real signal of model quality. Re-run after loading more historical "
            "seasons (see README).",
            "",
        ]
    lines += [
        f"Features used: {', '.join(numeric_features)}, position (one-hot).",
        "",
        "| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.mae:.2f} | {r.rmse:.2f} | {r.median_ae:.2f} | {r.r2:.3f} | {r.n_train} | {r.n_test} |"
        )
    lines.append("")

    out_path = out_path or (REPO_ROOT / "docs" / "model-comparison.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matrix_path = REPO_ROOT / "data" / "processed" / "training_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"{matrix_path} not found. Run: python scripts/train/build_training_matrix.py first"
        )

    df = pd.read_parquet(matrix_path)
    logging.info("Loaded %d rows from %s", len(df), matrix_path)

    numeric_cols = usable_numeric_features(df)
    logging.info("Usable numeric features (non-all-null): %s", numeric_cols)
    dropped = [c for c in _CANDIDATE_NUMERIC_FEATURES if c not in numeric_cols]
    if dropped:
        logging.info("Dropped (all-null for this data): %s", dropped)

    results = run_comparison(df)

    logging.info("")
    logging.info("%-20s %10s %10s %10s %8s", "Model", "MAE", "RMSE", "MedianAE", "R2")
    for r in results:
        logging.info("%-20s %10.2f %10.2f %10.2f %8.3f", r.name, r.mae, r.rmse, r.median_ae, r.r2)

    out_path = write_comparison_report(results, len(df), numeric_cols)
    logging.info("")
    logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
