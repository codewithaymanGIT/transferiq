"""
Tests for scripts/train/generate_predictions.py's core logic: feature
building (with honest skip-when-missing-DOB behavior) and model fitting.
Uses an in-memory SQLite DB, same pattern as tests/ml/test_load_core.py.
The full main() orchestration (real Postgres SessionLocal, commit,
logging) is intentionally not exercised here -- it's thin wiring around
the tested logic below, and is verified instead by the real run against
Postgres documented in docs/model-comparison.md and the live API/frontend.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "transform"))

from app import models  # noqa: E402
from generate_predictions import _build_live_features, _fit_final_model, _latest_season  # noqa: E402


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    Session_ = sessionmaker(bind=engine)
    session = Session_()
    yield session
    session.close()


def _synthetic_training_matrix(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = rng.choice(["GK", "DF", "MF", "FW"], n)
    minutes = rng.integers(500, 3000, n)
    goals = rng.poisson(3, n)
    assists = rng.poisson(2, n)
    ages = rng.uniform(18, 35, n)
    fee = np.clip(10 - 0.3 * ages + 0.5 * goals + rng.normal(0, 3, n), 0.5, None)
    return pd.DataFrame(
        {
            "player_name": [f"Player {i}" for i in range(n)],
            "position": positions,
            "minutes": minutes,
            "goals": goals,
            "assists": assists,
            "goals_per90": goals / (minutes / 90),
            "assists_per90": assists / (minutes / 90),
            "age_at_transfer": ages,
            "log_fee": np.log1p(fee),
        }
    )


def test_fit_final_model_produces_working_pipeline_on_full_data():
    df = _synthetic_training_matrix(n=30)
    pipeline = _fit_final_model(df)
    # Fit on the FULL dataset -- no held-out split -- so predicting on the
    # same rows it was trained on should work without error and produce
    # real, finite numbers.
    X = df[["minutes", "goals", "assists", "goals_per90", "assists_per90", "age_at_transfer", "position"]]
    preds = pipeline.predict(X)
    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))


def _make_season(db, label: str) -> models.Season:
    start_year = int(label.split("-")[0])
    season = models.Season(
        label=label, start_date=date(start_year, 8, 1), end_date=date(start_year + 1, 5, 31)
    )
    db.add(season)
    db.flush()
    return season


def test_latest_season_picks_highest_label(db_session):
    _make_season(db_session, "2020-2021")
    _make_season(db_session, "2022-2023")
    _make_season(db_session, "2021-2022")
    db_session.commit()
    latest = _latest_season(db_session)
    assert latest.label == "2022-2023"


def _make_player(db, name: str, dob: date | None) -> models.Player:
    player = models.Player(name=name, position=models.PositionGroup.FW, date_of_birth=dob)
    db.add(player)
    db.flush()
    return player


def test_build_live_features_skips_players_without_birth_year(db_session):
    season = _make_season(db_session, "2024-2025")

    with_dob = _make_player(db_session, "Has DOB", date(1998, 5, 12))
    without_dob = _make_player(db_session, "No DOB", None)
    db_session.add_all(
        [
            models.PlayerSeasonStats(player_id=with_dob.id, season_id=season.id, minutes=2000, goals=10, assists=5),
            models.PlayerSeasonStats(player_id=without_dob.id, season_id=season.id, minutes=1500, goals=5, assists=2),
        ]
    )
    db_session.commit()

    live_df, skipped = _build_live_features(db_session, season, reference_date=date(2026, 1, 1))
    assert skipped == 1
    assert len(live_df) == 1
    assert live_df.iloc[0]["player_id"] == with_dob.id


def test_build_live_features_computes_real_age_from_real_dob(db_session):
    season = _make_season(db_session, "2024-2025")

    player = _make_player(db_session, "Known Age", date(2000, 1, 1))
    db_session.add(
        models.PlayerSeasonStats(player_id=player.id, season_id=season.id, minutes=2000, goals=10, assists=5)
    )
    db_session.commit()

    live_df, _ = _build_live_features(db_session, season, reference_date=date(2026, 1, 1))
    # Exactly 26 years old on the reference date -- not fabricated, computed
    # directly from a real stored birth date.
    assert abs(live_df.iloc[0]["age_at_transfer"] - 26.0) < 0.01


def test_build_live_features_returns_empty_for_no_stats(db_session):
    season = _make_season(db_session, "2024-2025")
    db_session.commit()

    live_df, skipped = _build_live_features(db_session, season, reference_date=date(2026, 1, 1))
    assert live_df.empty
    assert skipped == 0
