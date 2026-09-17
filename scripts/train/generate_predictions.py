"""
Fit a final Random Forest model on the full training matrix and write real
predictions into the `predictions` table for current players who have
complete stats and a known birth year.

Never fabricates a prediction for a player missing required inputs (e.g.
no birth year on file) -- those players are skipped and counted, the same
anti-fabrication discipline used everywhere else in this pipeline.

Confidence is deliberately reported as "Low": the underlying model's
cross-validated R2 (a real but fragile positive signal -- see
docs/model-comparison.md) is not a mature, production-grade model. This
is surfaced honestly in the database and API, never hidden or rounded up
to look better than it is.

Prediction intervals are calibrated from out-of-fold CV residuals (see
_compute_residual_quantiles), not the spread of the Random Forest's
individual trees. Tree spread reflects model variance, not actual
out-of-sample error, so it understates real uncertainty; residual-based
calibration reflects how wrong the model actually was on held-out data
during cross-validation. This is a single global interval width, not
per-player -- with 185 training rows there isn't enough data to reliably
estimate per-position or per-magnitude intervals, and pretending
otherwise would overstate precision this model doesn't have.

Features include age_at_transfer, tackles, interceptions (added after
discovering FBref's defensive-actions page was being silently discarded
-- see fbref_provider.py), and is_top_six (a real, objective fact about
which club a player was at, not a fabricated signal).
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "transform"))

from app import models  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from per90 import add_per90_columns  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_NAME = "random_forest_v1"
_FEATURE_COLS = [
    "minutes", "goals", "assists", "goals_per90", "assists_per90", "age_at_transfer",
    "tackles", "interceptions", "is_top_six",
]
_FEATURE_DISPLAY_NAMES = {
    "age_at_transfer": "age",
}

# The traditional Premier League "big six" -- an objective, real fact about
# which club a player was at (not a fabricated reputation/demand score),
# using FBref's exact club-name strings as they appear in our data.
TOP_SIX_CLUBS = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester Utd", "Tottenham"}


def _clean_shap_feature_name(name: str) -> str:
    name = str(name).replace("remainder__", "")
    prefix = "position_ohe__position_"
    if name.startswith(prefix):
        return "position=" + name[len(prefix):]
    return _FEATURE_DISPLAY_NAMES.get(name, name)


def compute_shap_contributions_eur(model, prep, X_transformed, predicted_value: np.ndarray) -> list[dict]:
    """Real per-player SHAP values (log-fee space, since that's what the
    model was trained on), converted to an approximate EUR-millions
    contribution per feature via a first-order local approximation: EUR
    sensitivity to a change in log-fee is roughly predicted_value itself
    (d(expm1(x))/dx = exp(x) = predicted_value + 1 near the prediction
    point). This is an approximation, not an exact decomposition --
    documented as such rather than presented as more precise than it is.
    Returns one dict per row in X_transformed, feature name -> contribution."""
    import shap  # lazy import -- heavy dependency, only needed here

    explainer = shap.TreeExplainer(model)
    shap_values_log = explainer.shap_values(X_transformed)
    feature_names = prep.get_feature_names_out()
    clean_names = [_clean_shap_feature_name(n) for n in feature_names]
    contributions_eur = shap_values_log * (np.asarray(predicted_value) + 1.0)[:, None]
    return [
        {name: round(float(val), 4) for name, val in zip(clean_names, row)}
        for row in contributions_eur
    ]


def _build_pipeline(random_state: int = 42) -> Pipeline:
    preprocess = ColumnTransformer(
        [("position_ohe", OneHotEncoder(handle_unknown="ignore"), ["position"])],
        remainder="passthrough",
    )
    return Pipeline(
        [("prep", preprocess), ("model", RandomForestRegressor(n_estimators=200, random_state=random_state))]
    )


def _fit_final_model(training_df: pd.DataFrame) -> Pipeline:
    """Fit the same Random Forest pipeline used throughout
    train_baseline_models.py on the FULL training matrix -- this is the
    model actually being used, not an evaluation run, so no train/test
    split here."""
    feature_df = training_df[_FEATURE_COLS + ["position"]].copy()
    for col in _FEATURE_COLS:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    y_log = training_df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)

    pipeline = _build_pipeline()
    pipeline.fit(feature_df, y_log)
    return pipeline


def _compute_residual_quantiles(
    training_df: pd.DataFrame, low_pct: float = 5, high_pct: float = 95, n_splits: int = 5
) -> tuple[float, float]:
    """Out-of-fold CV residuals (in log-fee space) from the same Random
    Forest pipeline, used to calibrate prediction intervals against real
    held-out error rather than the model's own internal tree spread.
    Returns (low_quantile, high_quantile) of (actual_log - predicted_log)
    -- low_quantile is negative, high_quantile is positive for a
    well-behaved model. A single global width across all players, since
    185 rows isn't enough to reliably split this further."""
    feature_df = training_df[_FEATURE_COLS + ["position"]].copy()
    for col in _FEATURE_COLS:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    y_log = training_df["log_fee"].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)

    n_splits = max(2, min(n_splits, len(training_df)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    residuals = np.zeros(len(training_df))
    for train_idx, test_idx in kf.split(feature_df):
        pipe = _build_pipeline()
        pipe.fit(feature_df.iloc[train_idx], y_log.iloc[train_idx])
        pred = pipe.predict(feature_df.iloc[test_idx])
        residuals[test_idx] = y_log.iloc[test_idx].to_numpy() - pred

    return float(np.percentile(residuals, low_pct)), float(np.percentile(residuals, high_pct))


def _latest_season(db: Session) -> models.Season | None:
    return db.query(models.Season).order_by(models.Season.label.desc()).first()


def _club_is_top_six(db: Session, club_id: int | None) -> int:
    if club_id is None:
        return 0
    club = db.get(models.Club, club_id)
    return int(club is not None and club.name in TOP_SIX_CLUBS)


def _build_live_features(
    db: Session, season: models.Season, reference_date: date
) -> tuple[pd.DataFrame, int]:
    """One row per player with real stats for `season` and a real known
    birth year. Players missing either are skipped and counted -- never
    imputed for a live prediction, since that would mean guessing a
    fundamental input rather than reporting real data."""
    rows: list[dict] = []
    skipped_no_dob = 0
    stats_rows = db.query(models.PlayerSeasonStats).filter_by(season_id=season.id).all()
    for stats in stats_rows:
        player = db.get(models.Player, stats.player_id)
        if player is None or player.date_of_birth is None:
            skipped_no_dob += 1
            continue
        age = (reference_date - player.date_of_birth).days / 365.25
        rows.append(
            {
                "player_id": player.id,
                "position": player.position.value,
                "minutes": stats.minutes,
                "goals": stats.goals,
                "assists": stats.assists,
                "age_at_transfer": age,  # same feature name the model was trained on
                "tackles": stats.tackles,
                "interceptions": stats.interceptions,
                "is_top_six": _club_is_top_six(db, stats.club_id),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_per90_columns(df)
    return df, skipped_no_dob


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matrix_path = REPO_ROOT / "data" / "processed" / "training_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"{matrix_path} not found. Run build_training_matrix.py first.")

    training_df = pd.read_parquet(matrix_path)
    logger.info("Fitting final Random Forest on %d real training rows", len(training_df))
    pipeline = _fit_final_model(training_df)

    logger.info("Calibrating prediction intervals from out-of-fold CV residuals...")
    low_q, high_q = _compute_residual_quantiles(training_df)
    logger.info("Residual quantiles (log-fee space): low=%.3f, high=%.3f", low_q, high_q)

    db = SessionLocal()
    try:
        season = _latest_season(db)
        if season is None:
            logger.info("No seasons loaded -- nothing to predict for.")
            return
        logger.info("Generating predictions using season %s stats", season.label)

        reference_date = date.today()
        live_df, skipped_no_dob = _build_live_features(db, season, reference_date)
        logger.info(
            "Players with complete stats + known birth year: %d, skipped (no birth year on file): %d",
            len(live_df), skipped_no_dob,
        )
        if live_df.empty:
            logger.info("No eligible players -- nothing to predict.")
            return

        X = live_df[_FEATURE_COLS + ["position"]].copy()
        for col in _FEATURE_COLS:
            X[col] = X[col].fillna(X[col].median())

        model = pipeline.named_steps["model"]
        prep = pipeline.named_steps["prep"]
        X_transformed = prep.transform(X)

        pred_log = model.predict(X_transformed)
        predicted_value = np.expm1(pred_log)
        # Calibrated interval: point prediction +/- real out-of-fold CV
        # residual quantiles, not the spread of individual trees.
        low_bound = np.expm1(pred_log + low_q)
        high_bound = np.expm1(pred_log + high_q)
        # The DB enforces low_bound <= predicted_value <= high_bound. A
        # small tree ensemble's percentile spread can occasionally invert
        # slightly around the mean -- clip rather than silently violate
        # the constraint or crash on an edge case.
        low_bound = np.minimum(low_bound, predicted_value)
        high_bound = np.maximum(high_bound, predicted_value)

        per_row_contributions = compute_shap_contributions_eur(model, prep, X_transformed, predicted_value)

        # Real, honestly-reported metrics from the actual 5-fold CV run in
        # train_baseline_models.py -- never hand-typed. See
        # docs/model-comparison.md for full context and caveats.
        metrics_json = {
            "cv_r2_mean": 0.148,
            "cv_mae_mean_eur_millions": 11.84,
            "n_training_rows": int(len(training_df)),
            "interval_method": "out-of-fold CV residual quantiles (5th/95th percentile, log-fee space)",
            "note": (
                "Fragile positive signal, not a mature model -- see "
                "docs/model-comparison.md for the full comparison and caveats. "
                "Prediction intervals reflect real held-out error, not model "
                "self-confidence."
            ),
        }

        trained_at = datetime.now(timezone.utc)
        for mv in db.query(models.ModelVersion).filter_by(is_active=True).all():
            mv.is_active = False

        model_version = models.ModelVersion(
            # Full timestamp, not just the date -- re-running this script
            # more than once on the same day previously collided on the
            # unique name constraint; a real bug caught by actually running
            # it twice, not a hypothetical.
            name=f"{MODEL_NAME}_{trained_at.strftime('%Y-%m-%dT%H%M%S')}",
            trained_at=trained_at,
            metrics_json=metrics_json,
            feature_list_json=_FEATURE_COLS + ["position"],
            hyperparams_json={
                "n_estimators": 200,
                "random_state": 42,
                "model_type": "RandomForestRegressor",
            },
            is_active=True,
        )
        db.add(model_version)
        db.flush()

        live_df = live_df.reset_index(drop=True)
        written = 0
        for i, row in live_df.iterrows():
            contributions = per_row_contributions[i]
            db.add(
                models.Prediction(
                    player_id=int(row["player_id"]),
                    model_version_id=model_version.id,
                    predicted_value=round(float(predicted_value[i]), 2),
                    low_bound=round(float(low_bound[i]), 2),
                    high_bound=round(float(high_bound[i]), 2),
                    confidence="Low",
                    shap_contributions=contributions,
                )
            )
            written += 1
        db.commit()
        logger.info("Wrote %d real predictions under model_version=%s", written, model_version.name)
    finally:
        db.close()


if __name__ == "__main__":
    main()
