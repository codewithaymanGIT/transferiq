"""
Tests for scripts/train/train_baseline_models.py. Uses synthetic data
(clearly labeled "Player N") sized and shaped like the real training
matrix (including all-null xg/xa, matching the real FBref pull's
documented limitation) -- this tests the pipeline mechanics, not model
quality, which is exactly what this script's own proof-of-concept caveat
says to expect from a small real run too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

from train_baseline_models import (  # noqa: E402
    MIN_ROWS_FOR_TEST_SPLIT,
    CVModelResult,
    ModelResult,
    build_feature_matrix,
    run_comparison,
    run_cv_comparison,
    usable_numeric_features,
    write_comparison_report,
)


def _synthetic_matrix(n: int = 70, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = rng.choice(["GK", "DF", "MF", "FW"], n)
    minutes = rng.integers(500, 3000, n)
    goals = rng.poisson(3, n)
    assists = rng.poisson(2, n)
    fee = np.clip(5 + 0.5 * goals + 0.01 * minutes / 90 + rng.normal(0, 5, n), 0.5, None)

    return pd.DataFrame(
        {
            "player_name": [f"Player {i}" for i in range(n)],
            "position": positions,
            "minutes": minutes,
            "goals": goals,
            "assists": assists,
            "xg": [None] * n,  # mirrors the real pull: all-null for this season
            "xa": [None] * n,
            "fee_eur_millions": fee,
            "log_fee": np.log1p(fee),
        }
    )


def test_usable_numeric_features_drops_all_null_columns():
    df = _synthetic_matrix()
    features = usable_numeric_features(df)
    assert "minutes" in features
    assert "goals" in features
    assert "assists" in features
    assert "xg" not in features  # all-null -- must be dropped, never imputed
    assert "xa" not in features


def test_build_feature_matrix_imputes_only_sporadic_nulls():
    df = _synthetic_matrix()
    df.loc[0, "assists"] = None  # one sporadic missing value, not all-null
    feature_df, numeric_cols = build_feature_matrix(df)
    assert feature_df["assists"].isna().sum() == 0  # imputed
    assert "assists" in numeric_cols  # still usable since not entirely null


def test_run_comparison_produces_five_real_models():
    df = _synthetic_matrix(n=70)
    results = run_comparison(df)
    names = [r.name for r in results]
    assert names == ["Median baseline", "Linear Regression", "Ridge", "Random Forest", "XGBoost"]
    for r in results:
        assert r.n_train + r.n_test == 70
        assert r.mae >= 0
        assert r.rmse >= 0
        # sanity: metrics must be real floats, not placeholders
        assert isinstance(r.mae, float) and not pd.isna(r.mae)


def test_run_comparison_warns_below_min_rows(caplog):
    import logging

    df = _synthetic_matrix(n=10)
    with caplog.at_level(logging.WARNING):
        run_comparison(df)
    assert any("below the" in msg for msg in caplog.messages)
    assert MIN_ROWS_FOR_TEST_SPLIT > 10  # sanity-check the threshold makes this test meaningful


def test_report_flags_proof_of_concept_for_small_n(tmp_path):
    df = _synthetic_matrix(n=70)
    results = run_comparison(df)
    out_path = write_comparison_report(results, len(df), usable_numeric_features(df), out_path=tmp_path / "report.md")
    content = out_path.read_text()
    assert "Proof-of-concept only" in content
    assert "70 rows" in content
    assert "xg" not in content.split("Features used:")[1].split("\n")[0]  # dropped column not listed as used


def test_report_omits_caveat_for_large_n(tmp_path):
    df = _synthetic_matrix(n=500)
    results = run_comparison(df)
    out_path = write_comparison_report(results, len(df), usable_numeric_features(df), out_path=tmp_path / "report.md")
    content = out_path.read_text()
    assert "Proof-of-concept only" not in content


def test_report_contains_real_numbers_not_placeholders(tmp_path):
    df = _synthetic_matrix(n=70)
    results = run_comparison(df)
    out_path = write_comparison_report(results, len(df), usable_numeric_features(df), out_path=tmp_path / "report.md")
    content = out_path.read_text()
    for r in results:
        assert f"{r.mae:.2f}" in content
        assert f"{r.r2:.3f}" in content


def test_report_flags_negative_result_when_no_model_beats_baseline(tmp_path):
    results = [
        ModelResult(name="Median baseline", mae=10.0, rmse=15.0, median_ae=8.0, r2=-0.05, n_train=100, n_test=25),
        ModelResult(name="Linear Regression", mae=12.0, rmse=17.0, median_ae=11.0, r2=-0.30, n_train=100, n_test=25),
        ModelResult(name="Ridge", mae=12.5, rmse=17.5, median_ae=11.5, r2=-0.35, n_train=100, n_test=25),
    ]
    out_path = write_comparison_report(results, 500, ["minutes", "goals"], out_path=tmp_path / "report.md")
    content = out_path.read_text()
    assert "Negative result" in content
    # n=500 is above the proof-of-concept threshold, isolating this as the negative-result trigger, not the row-count one
    assert "Proof-of-concept only" not in content


def test_report_omits_negative_result_when_a_model_beats_baseline(tmp_path):
    results = [
        ModelResult(name="Median baseline", mae=10.0, rmse=15.0, median_ae=8.0, r2=-0.05, n_train=100, n_test=25),
        ModelResult(name="Linear Regression", mae=7.0, rmse=9.0, median_ae=6.5, r2=0.17, n_train=100, n_test=25),
    ]
    out_path = write_comparison_report(results, 500, ["minutes", "goals"], out_path=tmp_path / "report.md")
    content = out_path.read_text()
    assert "Negative result" not in content


def test_run_cv_comparison_produces_five_real_models_with_fold_stats():
    df = _synthetic_matrix(n=70)
    results = run_cv_comparison(df, n_splits=5)
    assert len(results) == 5
    names = {r.name for r in results}
    assert names == {"Median baseline", "Linear Regression", "Ridge", "Random Forest", "XGBoost"}
    for r in results:
        assert r.n_folds == 5
        assert isinstance(r.r2_mean, float) and not pd.isna(r.r2_mean)
        assert isinstance(r.r2_std, float) and r.r2_std >= 0
        assert r.mae_mean >= 0
        assert r.rmse_mean >= 0


def test_run_cv_comparison_caps_folds_at_row_count():
    df = _synthetic_matrix(n=3)
    results = run_cv_comparison(df, n_splits=5)
    # Can't have more folds than rows -- must degrade gracefully, not crash.
    assert results[0].n_folds <= 3
    assert results[0].n_folds >= 2


def test_report_includes_cv_section_when_provided(tmp_path):
    df = _synthetic_matrix(n=70)
    results = run_comparison(df)
    cv_results = run_cv_comparison(df)
    out_path = write_comparison_report(
        results, len(df), usable_numeric_features(df), out_path=tmp_path / "report.md", cv_results=cv_results
    )
    content = out_path.read_text()
    assert "Cross-validated results" in content
    for r in cv_results:
        assert f"{r.r2_mean:.3f}" in content


def test_report_omits_cv_section_when_not_provided(tmp_path):
    df = _synthetic_matrix(n=70)
    results = run_comparison(df)
    out_path = write_comparison_report(
        results, len(df), usable_numeric_features(df), out_path=tmp_path / "report.md"
    )
    content = out_path.read_text()
    assert "Cross-validated results" not in content


def test_negative_result_caveat_uses_cv_results_when_available(tmp_path):
    # Single-split results all look bad, but CV shows one model genuinely
    # beating baseline -- the caveat must trust the CV numbers, not the
    # single split, since that's the whole point of adding CV.
    results = [
        ModelResult(name="Median baseline", mae=10.0, rmse=15.0, median_ae=8.0, r2=-0.05, n_train=100, n_test=25),
        ModelResult(name="Random Forest", mae=12.0, rmse=17.0, median_ae=11.0, r2=-0.50, n_train=100, n_test=25),
    ]
    cv_results = [
        CVModelResult(name="Median baseline", r2_mean=-0.08, r2_std=0.09, mae_mean=10.0, mae_std=1.0, rmse_mean=15.0, rmse_std=1.0, n_folds=5),
        CVModelResult(name="Random Forest", r2_mean=0.06, r2_std=0.27, mae_mean=9.0, mae_std=1.0, rmse_mean=14.0, rmse_std=1.0, n_folds=5),
    ]
    out_path = write_comparison_report(
        results, 500, ["minutes", "goals"], out_path=tmp_path / "report.md", cv_results=cv_results
    )
    content = out_path.read_text()
    assert "Negative result" not in content
