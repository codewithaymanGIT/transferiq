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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]

_CANDIDATE_NUMERIC_FEATURES = [
    "minutes", "goals", "assists", "xg", "xa",
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "age_at_transfer",
]
# transfer_year tested and dropped: two runs (alone, and combined with
# age_at_transfer) both made every model worse, not better -- likely
# overfitting on ~100 training rows rather than a real era signal. Kept
# out of the default candidate set; still computed and available in the
# training matrix (see build_training_matrix.py) if revisited later with
# more data.


def add_transfer_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a numeric transfer_year (e.g. 2020 from '2020-2021') from the
    real transfer_season string written by build_training_matrix.py. This
    lets a model learn era-based market effects (inflation, spending
    trends) directly from real data, instead of us hand-picking an
    external inflation index we cannot fully verify."""
    df = df.copy()
    if "transfer_season" in df.columns:
        df["transfer_year"] = df["transfer_season"].str.split("-").str[0].astype("Int64").astype("float")
    return df

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


@dataclass
class CVModelResult:
    name: str
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    n_folds: int


def run_cv_comparison(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> list[CVModelResult]:
    """K-fold cross-validated comparison -- more reliable than a single
    train/test split at small sample sizes. A single 80/20 split's R2 can
    swing wildly (observed firsthand: adding/removing one feature moved R2
    by more than a full point on the same 128-row dataset) simply because
    which ~25 rows land in the held-out set matters a lot when there are so
    few of them. Averaging over multiple folds gives a steadier, more
    trustworthy read on whether a feature actually helps, though with this
    little data it is still a rough signal, not a precise one."""
    feature_df, numeric_cols = build_feature_matrix(df)
    y_log = df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)
    X = feature_df[numeric_cols + ["position"]]

    n = len(df)
    effective_splits = max(2, min(n_splits, n))
    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    preprocess = ColumnTransformer(
        [("position_ohe", OneHotEncoder(handle_unknown="ignore"), ["position"])],
        remainder="passthrough",
    )

    model_specs = [
        ("Median baseline", DummyRegressor(strategy="median")),
        ("Linear Regression", Pipeline([("prep", preprocess), ("model", LinearRegression())])),
        ("Ridge", Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0))])),
        (
            "Random Forest",
            Pipeline([("prep", preprocess), ("model", RandomForestRegressor(n_estimators=200, random_state=random_state))]),
        ),
        (
            "XGBoost",
            Pipeline([("prep", preprocess), ("model", XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0))]),
        ),
    ]

    results = []
    for name, model in model_specs:
        fold_r2, fold_mae, fold_rmse = [], [], []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_log, y_test_log = y_log.iloc[train_idx], y_log.iloc[test_idx]
            model_clone = clone(model)
            model_clone.fit(X_train, y_train_log)
            pred_log = model_clone.predict(X_test)
            y_test = np.expm1(y_test_log)
            pred = np.expm1(pred_log)
            fold_r2.append(r2_score(y_test, pred))
            fold_mae.append(mean_absolute_error(y_test, pred))
            fold_rmse.append(root_mean_squared_error(y_test, pred))
        results.append(
            CVModelResult(
                name=name,
                r2_mean=float(np.mean(fold_r2)),
                r2_std=float(np.std(fold_r2)),
                mae_mean=float(np.mean(fold_mae)),
                mae_std=float(np.std(fold_mae)),
                rmse_mean=float(np.mean(fold_rmse)),
                rmse_std=float(np.std(fold_rmse)),
                n_folds=effective_splits,
            )
        )
    return results


def compute_feature_importances(df: pd.DataFrame, random_state: int = 42) -> list[tuple[str, float]]:
    """Random Forest feature importances fit on the FULL dataset (no
    train/test split -- this is about understanding which features the
    model leans on, not evaluating held-out accuracy, so using all
    available rows gives the most stable estimate at this sample size).
    Purely descriptive of what this particular model does, not a validated
    or causal claim -- SHAP (planned per the blueprint) will give a more
    rigorous, per-prediction explanation later. Returns (feature_name,
    importance) sorted descending."""
    feature_df, numeric_cols = build_feature_matrix(df)
    y_log = df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)
    X = feature_df[numeric_cols + ["position"]]

    preprocess = ColumnTransformer(
        [("position_ohe", OneHotEncoder(handle_unknown="ignore"), ["position"])],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [("prep", preprocess), ("model", RandomForestRegressor(n_estimators=200, random_state=random_state))]
    )
    pipeline.fit(X, y_log)

    def _clean_name(name: str) -> str:
        # sklearn's ColumnTransformer prefixes names with the transformer id
        # (e.g. "remainder__age_at_transfer", "position_ohe__position_MF") --
        # strip that internal plumbing for a report a person will actually read.
        name = str(name).replace("remainder__", "")
        prefix = "position_ohe__position_"
        if name.startswith(prefix):
            return "position=" + name[len(prefix):]
        return name

    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return [(_clean_name(name), float(imp)) for name, imp in pairs]


def write_comparison_report(
    results: list[ModelResult],
    n_total_rows: int,
    numeric_features: list[str],
    out_path: Path | None = None,
    cv_results: list["CVModelResult"] | None = None,
    feature_importances: list[tuple[str, float]] | None = None,
) -> Path:
    lines = [
        "# Model comparison",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`scripts/train/train_baseline_models.py` -- every number below comes from an "
        "actual sklearn/xgboost fit on real data, never hand-typed.",
        "",
    ]
    # Prefer the cross-validated results for this check when available -- a
    # single 80/20 split's R2 is too unstable at this sample size to safely
    # call "negative" (we observed a model that failed on one split come out
    # ahead of baseline once averaged over 5 folds). Fall back to the single
    # split only if no CV results were computed.
    if cv_results:
        baseline_r2 = next((r.r2_mean for r in cv_results if r.name == "Median baseline"), None)
        non_baseline_r2s = [r.r2_mean for r in cv_results if r.name != "Median baseline"]
    else:
        baseline_r2 = next((r.r2 for r in results if r.name == "Median baseline"), None)
        non_baseline_r2s = [r.r2 for r in results if r.name != "Median baseline"]
    no_model_beats_baseline = (
        baseline_r2 is not None and bool(non_baseline_r2s) and all(r2 <= baseline_r2 for r2 in non_baseline_r2s)
    )

    if n_total_rows < PROOF_OF_CONCEPT_ROW_THRESHOLD:
        lines += [
            "> **Proof-of-concept only.** This run used "
            f"**{n_total_rows} rows** -- far below what's needed for a trustworthy "
            "comparison. Treat every metric here as illustrating that the pipeline works, "
            "not as a real signal of model quality. Re-run after loading more historical "
            "seasons (see README).",
            "",
        ]
    if no_model_beats_baseline:
        lines += [
            "> **Negative result.** No model here beat a naive median-fee guess "
            "(every model's R2 is at or below the median baseline's R2). This is an "
            "honest finding, not a working valuation model -- treat it as evidence the "
            "current features/data aren't sufficient yet, not as real predictions.",
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

    if cv_results:
        lines += [
            f"## Cross-validated results ({cv_results[0].n_folds}-fold)",
            "",
            "Averaged over multiple folds rather than one fixed 80/20 split -- at "
            "this sample size a single split's R2 is unstable enough (observed "
            "swings of more than a full point from adding a single feature) that "
            "it should not be trusted alone. This is a steadier, though still "
            "rough, read on whether a feature genuinely helps.",
            "",
            "| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |",
            "|---|---|---|---|---|",
        ]
        for r in cv_results:
            lines.append(
                f"| {r.name} | {r.mae_mean:.2f} | {r.rmse_mean:.2f} | {r.r2_mean:.3f} | {r.r2_std:.3f} |"
            )
        lines.append("")

    if feature_importances:
        lines += [
            "## Random Forest feature importances",
            "",
            "Fit on the full dataset (not a held-out split) purely to see which "
            "features this particular model leans on most -- descriptive, not a "
            "validated or causal claim, and not the same as the metrics above. "
            "SHAP explanations (planned) will give a more rigorous per-prediction "
            "breakdown later.",
            "",
            "| Feature | Importance |",
            "|---|---|",
        ]
        for name, imp in feature_importances:
            lines.append(f"| {name} | {imp:.3f} |")
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
    df = add_transfer_year_column(df)

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

    cv_results = run_cv_comparison(df)
    logging.info("")
    logging.info("Cross-validated (%d-fold), more reliable at this sample size:", cv_results[0].n_folds)
    logging.info("%-20s %14s %14s %10s", "Model", "Mean MAE", "Mean RMSE", "Mean R2")
    for r in cv_results:
        logging.info("%-20s %14.2f %14.2f %10.3f", r.name, r.mae_mean, r.rmse_mean, r.r2_mean)

    feature_importances = compute_feature_importances(df)
    logging.info("")
    logging.info("Random Forest feature importances (fit on full data):")
    for name, imp in feature_importances:
        logging.info("  %-30s %.3f", name, imp)

    out_path = write_comparison_report(
        results, len(df), numeric_cols, cv_results=cv_results, feature_importances=feature_importances
    )
    logging.info("")
    logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
