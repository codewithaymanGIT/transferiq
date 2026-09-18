"""
Walk-forward (out-of-time) validation, as distinct from the random 5-fold
CV used elsewhere in this project.

Random k-fold CV shuffles all 185 transfers across folds, which silently
assumes transfer fees are drawn from the same distribution regardless of
year -- but we already found real evidence against that (transfer_year
carries real signal, even though adding it directly as a feature made
the model worse, likely because 185 rows isn't enough to separate a
trend from noise). A model meant to predict FUTURE transfer fees should
be validated the way it will actually be used: trained only on the past,
tested only on the future. This is standard practice in model risk
management (close to what "out-of-time validation" means in SR 11-7
style model governance), not just an ML nicety.

Method: sort transfers by season chronologically. For each season from
the 3rd onward, train on every season strictly before it, test on that
season alone, and pool every held-out prediction across all folds into
one combined set for the headline metric (per-season R2 on 13-35 rows
is too noisy to trust individually -- see the per-season breakdown for
that caveat made explicit rather than hidden).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_predictions import _FEATURE_COLS, _build_pipeline  # noqa: E402


def _season_sort_key(label: str) -> int:
    return int(label.split("-")[0])


def main() -> None:
    matrix_path = REPO_ROOT / "data" / "processed" / "training_matrix.parquet"
    df = pd.read_parquet(matrix_path)
    seasons = sorted(df["transfer_season"].unique(), key=_season_sort_key)

    print(f"Walk-forward validation across {len(seasons)} seasons, {len(df)} total rows\n")
    print(f"{'Test season':<14} {'Train rows':>10} {'Test rows':>10} {'R2':>8} {'MAE':>8}")

    all_actual: list[float] = []
    all_pred: list[float] = []
    per_season_rows: list[dict] = []

    for i in range(2, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]

        train_df = df[df["transfer_season"].isin(train_seasons)].reset_index(drop=True)
        test_df = df[df["transfer_season"] == test_season].reset_index(drop=True)
        if test_df.empty or train_df.empty:
            continue

        feature_df = train_df[_FEATURE_COLS + ["position"]].copy()
        for col in _FEATURE_COLS:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        y_log = train_df["log_fee"].reset_index(drop=True)

        pipe = _build_pipeline()
        pipe.fit(feature_df, y_log)

        X_test = test_df[_FEATURE_COLS + ["position"]].copy()
        for col in _FEATURE_COLS:
            # Real anti-leakage discipline: impute test-fold nulls using
            # the TRAINING fold's median, never the test fold's own median
            # -- using test-set statistics to fill test-set gaps would
            # leak future information into a fold meant to simulate not
            # having it yet.
            X_test[col] = X_test[col].fillna(feature_df[col].median())

        pred_log = pipe.predict(X_test)
        pred = np.expm1(pred_log)
        actual = np.expm1(test_df["log_fee"].to_numpy())

        fold_r2 = r2_score(actual, pred) if len(actual) > 1 else float("nan")
        fold_mae = mean_absolute_error(actual, pred)
        print(f"{test_season:<14} {len(train_df):>10} {len(test_df):>10} {fold_r2:>8.3f} {fold_mae:>8.2f}")

        all_actual.extend(actual.tolist())
        all_pred.extend(pred.tolist())
        per_season_rows.append(
            {"test_season": test_season, "train_rows": len(train_df), "test_rows": len(test_df),
             "r2": round(float(fold_r2), 3) if not np.isnan(fold_r2) else None, "mae": round(float(fold_mae), 2)}
        )

    pooled_r2 = r2_score(all_actual, all_pred)
    pooled_mae = mean_absolute_error(all_actual, all_pred)
    print(f"\nPooled out-of-time R2 (all {len(all_actual)} held-out predictions combined): {pooled_r2:.3f}")
    print(f"Pooled out-of-time MAE: {pooled_mae:.2f}")

    report_lines = [
        "# Temporal (Walk-Forward) Validation",
        "",
        "Out-of-time validation, distinct from the random 5-fold CV reported in",
        "`model-comparison.md`. Trains only on seasons strictly before the test",
        "season, simulating how the model is actually used: predicting fees for",
        "transfers that have not happened yet, using only what was known before.",
        "",
        "## Per-season breakdown",
        "",
        "Early folds train on very little data (a handful of prior seasons) and",
        "each test season alone is small (13-35 rows) -- individual per-season R2",
        "swings a lot and should not be over-interpreted on its own. The pooled",
        "metric below, combining every held-out prediction across all folds, is",
        "the more reliable headline number.",
        "",
        "| Test season | Train rows | Test rows | R2 | MAE (EUR m) |",
        "|---|---|---|---|---|",
    ]
    for row in per_season_rows:
        r2_str = f"{row['r2']:.3f}" if row["r2"] is not None else "n/a (1 row)"
        report_lines.append(f"| {row['test_season']} | {row['train_rows']} | {row['test_rows']} | {r2_str} | {row['mae']:.2f} |")

    report_lines += [
        "",
        "## Pooled out-of-time result",
        "",
        f"- **R2: {pooled_r2:.3f}** across all {len(all_actual)} held-out predictions combined",
        f"- **MAE: {pooled_mae:.2f}m EUR**",
        "",
        "For comparison, the random 5-fold CV reported in `model-comparison.md`",
        f"gives R2=0.140 on the same 185-row dataset. The gap between the two",
        "(if any) is itself informative: a materially worse out-of-time result",
        "would mean the model is leaning on patterns that don't hold up when",
        "tested the way it will actually be used, which random shuffling can",
        "hide.",
        "",
        "No test-fold statistic (e.g. median for imputation) is ever computed",
        "from the test fold itself -- only from the training fold available at",
        "that point in time, to avoid leaking future information backward.",
    ]
    out_path = REPO_ROOT / "docs" / "temporal-validation.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
