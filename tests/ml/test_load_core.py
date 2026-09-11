"""
Tests for scripts/load/load_core.py -- the DataFrame-to-Postgres upsert
logic. Uses an in-memory SQLite DB (same pattern as tests/backend), since
the ORM models were already verified SQLite-compatible there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "load"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "clean"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validate"))

from app import models  # noqa: E402
from clean_player_stats import clean_player_season_stats  # noqa: E402
from load_core import load_dataframe, load_transfers, primary_position  # noqa: E402
from validate_player_stats import validate_player_season_stats  # noqa: E402


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    Session_ = sessionmaker(bind=engine)
    session = Session_()
    yield session
    session.close()


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["Bukayo Saka", "David Raya", "Erling Haaland"],
            "club": ["Arsenal", "Arsenal", "Manchester City"],
            "position": ["FW,MF", "GK", "FW"],
            "minutes": [1729, 3420, 2900],
            "starts": [19, 38, 32],
            "apps": [20, 38, 33],
            "goals": [6, 0, 27],
            "assists": [10, 0, 5],
            "xg": [None, None, None],
            "xa": [None, None, None],
            "season": ["2024-2025"] * 3,
            "source": ["FBref"] * 3,
        }
    )


def test_primary_position_handles_multi_eligible():
    assert primary_position("FW,MF") == "FW"
    assert primary_position("GK") == "GK"
    assert primary_position(None) is None
    assert primary_position("STRIKER") is None  # not a recognized code -- never guessed


def test_load_creates_players_clubs_and_stats(db_session: Session):
    df = _sample_df()
    result = load_dataframe(db_session, df, season="2024-2025")
    db_session.commit()

    assert result.players_created == 3
    assert result.players_updated == 0
    assert result.stats_rows_written == 3

    players = db_session.query(models.Player).all()
    assert {p.name for p in players} == {"Bukayo Saka", "David Raya", "Erling Haaland"}

    saka = db_session.query(models.Player).filter_by(name="Bukayo Saka").first()
    assert saka.position == models.PositionGroup.FW  # took the first of "FW,MF"

    clubs = {c.name for c in db_session.query(models.Club).all()}
    assert clubs == {"Arsenal", "Manchester City"}

    stats = db_session.query(models.PlayerSeasonStats).filter_by(player_id=saka.id).first()
    assert stats.minutes == 1729
    assert stats.goals == 6
    assert stats.assists == 10


def test_load_is_idempotent_on_rerun(db_session: Session):
    df = _sample_df()
    load_dataframe(db_session, df, season="2024-2025")
    db_session.commit()

    # Re-run with updated stats for the same players/season -- should UPDATE, not duplicate.
    df2 = _sample_df()
    df2.loc[df2["player_name"] == "Bukayo Saka", "goals"] = 9  # stats changed since last pull
    result2 = load_dataframe(db_session, df2, season="2024-2025")
    db_session.commit()

    assert result2.players_created == 0
    assert result2.players_updated == 3

    n_players = db_session.query(models.Player).count()
    n_stats_rows = db_session.query(models.PlayerSeasonStats).count()
    assert n_players == 3  # no duplicates
    assert n_stats_rows == 3  # no duplicates

    saka_stats = (
        db_session.query(models.PlayerSeasonStats)
        .join(models.Player)
        .filter(models.Player.name == "Bukayo Saka")
        .first()
    )
    assert saka_stats.goals == 9  # updated, not a second row


def test_load_skips_rows_with_missing_identity(db_session: Session):
    df = _sample_df()
    df.loc[0, "player_name"] = None
    result = load_dataframe(db_session, df, season="2024-2025")
    db_session.commit()

    assert result.rows_skipped_missing_identity == 1
    assert result.stats_rows_written == 2


def test_load_skips_rows_with_unparseable_position(db_session: Session):
    df = _sample_df()
    df.loc[0, "position"] = "UNKNOWN"
    result = load_dataframe(db_session, df, season="2024-2025")
    db_session.commit()

    assert "Bukayo Saka" in result.rows_skipped_unparseable_position
    assert result.stats_rows_written == 2


def test_full_pipeline_clean_validate_load(db_session: Session):
    """End-to-end: raw-shaped df -> clean -> validate -> load, matching what
    load_to_postgres.py actually does against a real parquet file."""
    raw = pd.DataFrame(
        {
            "player_name": ["Declan Rice", "Declan Rice"],  # exact duplicate row, as real pulls can have
            "club": ["Arsenal", "Arsenal"],
            "position": ["MF", "MF"],
            "minutes": [2825, 2825],
            "goals": [4, 4],
            "assists": [7, None],  # second row more complete... actually less; first wins on non-null count tie
            "season": ["2024-2025", "2024-2025"],
            "source": ["FBref", "FBref"],
        }
    )
    cleaned = clean_player_season_stats(raw)
    assert len(cleaned) == 1  # deduped

    report = validate_player_season_stats(cleaned)
    assert report.is_clean

    result = load_dataframe(db_session, cleaned, season="2024-2025")
    db_session.commit()
    assert result.players_created == 1
    assert result.stats_rows_written == 1


# --------------------------------------------------------------- transfers

def _sample_transfers_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["Bukayo Saka", "Bukayo Saka", "Ghost Player", "Youth Signing"],
            "club": ["Arsenal", "Southampton", "Arsenal", "Arsenal"],
            "counterparty_club": ["Southampton", "Arsenal", "Some Club", None],
            "transfer_movement": ["in", "out", "in", "in"],
            "transfer_period": ["Summer", "Summer", "Summer", "Summer"],
            "fee_eur_millions": [None, None, 10.0, None],
            "fee_disclosed": [False, False, True, False],
            "transfer_type": ["FREE", "FREE", "PERMANENT", None],  # last row: '-' -> non-transfer, excluded
            "season": ["2019-2020", "2019-2020", "2019-2020", "2019-2020"],
            "source": ["Transfermarkt (via ewenme/transfers)"] * 4,
        }
    )


def test_load_transfers_matches_existing_players_only(db_session: Session):
    # Seed one real player (as load_dataframe would from an FBref pull) -- "Ghost Player" is NOT seeded.
    db_session.add(models.Player(name="Bukayo Saka", position=models.PositionGroup.FW))
    db_session.commit()

    df = _sample_transfers_df()
    result = load_transfers(db_session, df, source_name="Transfermarkt", source_url="https://example.com")
    db_session.commit()

    # Only the 'in' row for Saka (matched player, real transfer_type) should load.
    # "Ghost Player" in-row is skipped (not in players table); "Youth Signing" is skipped ('-' -> non-transfer).
    assert result.transfers_created == 1
    assert result.rows_skipped_non_transfer == 1
    assert "Ghost Player" in result.rows_skipped_unmatched_player

    transfer = db_session.query(models.Transfer).first()
    assert transfer.transfer_type == models.TransferType.FREE
    assert transfer.fee_amount is None


def test_load_transfers_preserves_real_disclosed_fee(db_session: Session):
    db_session.add(models.Player(name="Ghost Player", position=models.PositionGroup.MF))
    db_session.commit()

    df = _sample_transfers_df()
    load_transfers(db_session, df, source_name="Transfermarkt", source_url="https://example.com")
    db_session.commit()

    transfer = (
        db_session.query(models.Transfer)
        .join(models.Player)
        .filter(models.Player.name == "Ghost Player")
        .first()
    )
    assert transfer.transfer_type == models.TransferType.PERMANENT
    assert transfer.fee_amount == pytest.approx(10.0)
    assert transfer.fee_disclosed is True


def test_load_transfers_is_idempotent_on_rerun(db_session: Session):
    db_session.add(models.Player(name="Bukayo Saka", position=models.PositionGroup.FW))
    db_session.commit()

    df = _sample_transfers_df()
    first = load_transfers(db_session, df, source_name="Transfermarkt", source_url="https://example.com")
    db_session.commit()
    assert first.transfers_created == 1

    second = load_transfers(db_session, df, source_name="Transfermarkt", source_url="https://example.com")
    db_session.commit()
    assert second.transfers_created == 0
    assert second.transfers_already_loaded == 1

    n_transfers = db_session.query(models.Transfer).count()
    assert n_transfers == 1  # not duplicated
