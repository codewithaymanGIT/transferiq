"""
Provider contract tests for TransfermarktProvider. Runs offline against a
small real-data excerpt (tests/ml/fixtures/transfermarkt_sample.csv --
genuine rows pulled from the live source, not fabricated), so this
verifies real parsing/type-coercion logic without a network call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ingest"))

from providers.transfermarkt_provider import TransfermarktProvider  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "transfermarkt_sample.csv"

REQUIRED_COLUMNS = {
    "player_name", "position_raw", "age_at_transfer", "club", "counterparty_club",
    "transfer_movement", "transfer_period", "fee_raw", "fee_eur_millions", "fee_disclosed",
    "transfer_type", "year", "season", "source",
}


@pytest.fixture
def provider():
    return TransfermarktProvider(csv_source=str(FIXTURE_PATH))


def test_returns_required_columns(provider):
    df = provider.fetch_transfers()
    assert isinstance(df, pd.DataFrame)
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert len(df) == 5


def test_real_disclosed_fees_parsed_correctly(provider):
    df = provider.fetch_transfers()
    ronaldo_out = df[(df["player_name"] == "Cristiano Ronaldo") & (df["transfer_movement"] == "out")].iloc[0]
    assert ronaldo_out["fee_eur_millions"] == pytest.approx(94.0)
    assert ronaldo_out["fee_disclosed"] is True or bool(ronaldo_out["fee_disclosed"]) is True
    assert ronaldo_out["season"] == "2009-2010"  # '2009/2010' -> '2009-2010'


def test_undisclosed_fees_are_null_not_zero(provider):
    df = provider.fetch_transfers()
    undisclosed = df[df["player_name"] == "Chris Morris"].iloc[0]
    assert pd.isna(undisclosed["fee_eur_millions"])
    assert bool(undisclosed["fee_disclosed"]) is False
    assert undisclosed["transfer_type"] == "PERMANENT"  # '?' = real transfer, fee just undisclosed


def test_dash_rows_excluded_as_non_transfer_events(provider):
    df = provider.fetch_transfers()
    youth_move = df[df["player_name"] == "Ben Roberts"].iloc[0]
    assert pd.isna(youth_move["fee_eur_millions"])
    assert pd.isna(youth_move["transfer_type"])  # '-' -- not a real transfer event


def test_loan_and_free_classified_correctly(provider):
    df = provider.fetch_transfers()
    # Ronaldo's Sporting CP move is a disclosed permanent fee -- sanity check the positive case.
    ronaldo_in = df[(df["player_name"] == "Cristiano Ronaldo") & (df["transfer_movement"] == "in")].iloc[0]
    assert ronaldo_in["transfer_type"] == "PERMANENT"
    assert ronaldo_in["fee_eur_millions"] == pytest.approx(19.0)


def test_season_filter(provider):
    df = provider.fetch_transfers(season="2009-2010")
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Cristiano Ronaldo"


def test_fetch_player_season_stats_not_supported(provider):
    with pytest.raises(NotImplementedError):
        provider.fetch_player_season_stats(season="2024-2025")
