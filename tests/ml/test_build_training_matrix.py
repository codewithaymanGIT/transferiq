"""
Tests for scripts/train/build_training_matrix.py -- especially that the
leakage rule (transfer in season S can only use S-1 stats) is enforced by
construction, not just documented.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

from app import models  # noqa: E402
from build_training_matrix import _prior_season_label, assemble_training_matrix  # noqa: E402


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    Session_ = sessionmaker(bind=engine)
    session = Session_()
    yield session
    session.close()


def _season(db, label, start_year):
    s = models.Season(label=label, start_date=datetime.date(start_year, 8, 1), end_date=datetime.date(start_year + 1, 5, 31))
    db.add(s)
    db.flush()
    return s


def test_prior_season_label():
    assert _prior_season_label("2022-2023") == "2021-2022"
    assert _prior_season_label("2020-2021") == "2019-2020"


def test_matches_transfer_to_correct_prior_season_only(db_session: Session):
    player = models.Player(name="Test Player", position=models.PositionGroup.FW)
    db_session.add(player)
    db_session.flush()

    s_2021 = _season(db_session, "2021-2022", 2021)
    s_2022 = _season(db_session, "2022-2023", 2022)

    # Stats in the CORRECT prior season (2021-2022) -- should be matched.
    prior_stats = models.PlayerSeasonStats(
        player_id=player.id, season_id=s_2021.id, minutes=2500, goals=15, assists=5
    )
    # Stats in the SAME season as the transfer (2022-2023) -- must NEVER be used as a feature
    # for a transfer that happened in that season (that would be leakage).
    same_season_stats = models.PlayerSeasonStats(
        player_id=player.id, season_id=s_2022.id, minutes=2800, goals=25, assists=10  # much better -- if leaked, would be obviously wrong
    )
    db_session.add_all([prior_stats, same_season_stats])
    db_session.flush()

    transfer = models.Transfer(
        player_id=player.id,
        transfer_date=datetime.date(2022, 8, 1),
        fee_amount=50.0,
        fee_currency=models.FeeCurrency.EUR,
        fee_disclosed=True,
        transfer_type=models.TransferType.PERMANENT,
        season_id=s_2022.id,
    )
    db_session.add(transfer)
    db_session.commit()

    result = assemble_training_matrix(db_session)
    assert result.matched == 1
    row = result.df.iloc[0]
    assert row["stats_season"] == "2021-2022"
    assert row["goals"] == 15  # the PRIOR season's goals, not the same-season 25


def test_skips_transfer_with_no_prior_season_stats(db_session: Session):
    player = models.Player(name="No History Player", position=models.PositionGroup.MF)
    db_session.add(player)
    db_session.flush()
    s_2022 = _season(db_session, "2022-2023", 2022)

    transfer = models.Transfer(
        player_id=player.id,
        transfer_date=datetime.date(2022, 8, 1),
        fee_amount=20.0,
        fee_currency=models.FeeCurrency.EUR,
        fee_disclosed=True,
        transfer_type=models.TransferType.PERMANENT,
        season_id=s_2022.id,
    )
    db_session.add(transfer)
    db_session.commit()

    result = assemble_training_matrix(db_session)
    assert result.matched == 0
    assert result.transfers_considered == 1
    assert len(result.skipped_no_prior_season_stats) == 1


def test_excludes_undisclosed_and_loan_transfers(db_session: Session):
    player = models.Player(name="Loan Player", position=models.PositionGroup.DF)
    db_session.add(player)
    db_session.flush()
    s_2021 = _season(db_session, "2021-2022", 2021)
    s_2022 = _season(db_session, "2022-2023", 2022)
    db_session.add(models.PlayerSeasonStats(player_id=player.id, season_id=s_2021.id, minutes=2000, goals=2, assists=1))
    db_session.flush()

    loan = models.Transfer(
        player_id=player.id, transfer_date=datetime.date(2022, 8, 1), fee_amount=None,
        fee_currency=models.FeeCurrency.EUR, fee_disclosed=False,
        transfer_type=models.TransferType.LOAN, season_id=s_2022.id,
    )
    undisclosed = models.Transfer(
        player_id=player.id, transfer_date=datetime.date(2022, 8, 1), fee_amount=None,
        fee_currency=models.FeeCurrency.EUR, fee_disclosed=False,
        transfer_type=models.TransferType.PERMANENT, season_id=s_2022.id,
    )
    db_session.add_all([loan, undisclosed])
    db_session.commit()

    result = assemble_training_matrix(db_session)
    assert result.matched == 0  # neither the loan nor the undisclosed-fee transfer qualifies


def test_log_fee_computed_correctly(db_session: Session):
    import math

    player = models.Player(name="Fee Player", position=models.PositionGroup.FW)
    db_session.add(player)
    db_session.flush()
    s_2021 = _season(db_session, "2021-2022", 2021)
    s_2022 = _season(db_session, "2022-2023", 2022)
    db_session.add(models.PlayerSeasonStats(player_id=player.id, season_id=s_2021.id, minutes=2500, goals=10, assists=5))
    db_session.flush()
    db_session.add(
        models.Transfer(
            player_id=player.id, transfer_date=datetime.date(2022, 8, 1), fee_amount=80.0,
            fee_currency=models.FeeCurrency.EUR, fee_disclosed=True,
            transfer_type=models.TransferType.PERMANENT, season_id=s_2022.id,
        )
    )
    db_session.commit()

    result = assemble_training_matrix(db_session)
    row = result.df.iloc[0]
    assert row["fee_eur_millions"] == pytest.approx(80.0)
    assert row["log_fee"] == pytest.approx(math.log1p(80.0))
