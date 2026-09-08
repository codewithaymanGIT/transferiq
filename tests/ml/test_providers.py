"""
Provider contract tests.

These run entirely offline: the FPL provider's HTTP call is mocked with a
static fixture, so CI (and this sandbox) can verify the shape contract
every provider must satisfy without depending on a live third-party site.
Live end-to-end pulls are a separate, manual step (see scripts/ingest/run_ingest.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ingest"))

from providers.fpl_provider import FPLProvider  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fpl_bootstrap_sample.json"

REQUIRED_COLUMNS = {
    "source_player_id",
    "player_name",
    "club",
    "position",
    "minutes",
    "starts",
    "apps",
    "goals",
    "assists",
    "xg",
    "xa",
    "shots",
    "sot",
    "key_passes",
    "prog_passes",
    "prog_carries",
    "tackles",
    "interceptions",
    "clearances",
    "blocks",
    "saves",
    "save_pct",
    "goals_conceded",
    "season",
    "source",
}


@pytest.fixture
def mock_session():
    payload = json.loads(FIXTURE_PATH.read_text())
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


def test_fpl_provider_returns_required_columns(mock_session):
    provider = FPLProvider(session=mock_session)
    df = provider.fetch_player_season_stats(season="2024-2025")

    assert isinstance(df, pd.DataFrame)
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert len(df) == 2


def test_fpl_provider_maps_positions_correctly(mock_session):
    provider = FPLProvider(session=mock_session)
    df = provider.fetch_player_season_stats(season="2024-2025")

    forward = df[df["player_name"] == "Sample Forward"].iloc[0]
    keeper = df[df["player_name"] == "Sample Keeper"].iloc[0]

    assert forward["position"] == "FW"
    assert keeper["position"] == "GK"
    assert forward["club"] == "Arsenal"
    assert keeper["club"] == "Man City"


def test_fpl_provider_coerces_xg_to_float(mock_session):
    provider = FPLProvider(session=mock_session)
    df = provider.fetch_player_season_stats(season="2024-2025")

    forward = df[df["player_name"] == "Sample Forward"].iloc[0]
    assert forward["xg"] == pytest.approx(15.4)


def test_fpl_provider_rejects_non_premier_league(mock_session):
    provider = FPLProvider(session=mock_session)
    with pytest.raises(ValueError):
        provider.fetch_player_season_stats(season="2024-2025", league="ESP-La Liga")


def test_no_leakage_helper_placeholder():
    """
    The real leakage tests now live in tests/ml/test_clean_validate_transform.py
    (test_longitudinal_growth_only_uses_strictly_prior_season and
    test_longitudinal_does_not_leak_across_players), exercising
    scripts/transform/longitudinal.py directly. Kept here as a pointer so
    this file's history stays readable.
    """
    assert True
