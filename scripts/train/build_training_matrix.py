"""
Assemble the ML training matrix: real transfer fees joined against the
performance stats known *before* each transfer happened.

Leakage rule (per docs/phase-0-blueprint.md section 14): a transfer in
season S can only be matched against player_season_stats from season S-1
(the season immediately prior) -- never the same season or later, since
that information wasn't available at transfer time. This is enforced by
construction here, not just documented: `_prior_season_label` computes S-1
from the transfer's season and the join looks up exactly that season.

Only PERMANENT transfers with a disclosed fee are usable as the
regression target (per the blueprint's target-definition discussion) --
loans, frees, and undisclosed fees are excluded from `y`, not imputed.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "transform"))

from app import models  # noqa: E402
from per90 import add_per90_columns  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class MatrixResult:
    df: pd.DataFrame
    transfers_considered: int = 0
    matched: int = 0
    skipped_no_prior_season_stats: list[str] = field(default_factory=list)


def _prior_season_label(season_label: str) -> str:
    """'2022-2023' -> '2021-2022'."""
    start, end = (int(y) for y in season_label.split("-"))
    return f"{start - 1}-{end - 1}"


def assemble_training_matrix(db: Session) -> MatrixResult:
    transfers = (
        db.query(models.Transfer)
        .filter(
            models.Transfer.transfer_type == models.TransferType.PERMANENT,
            models.Transfer.fee_disclosed.is_(True),
            models.Transfer.fee_amount.isnot(None),
        )
        .all()
    )

    rows: list[dict] = []
    skipped: list[str] = []

    for transfer in transfers:
        player = db.get(models.Player, transfer.player_id)
        season = db.get(models.Season, transfer.season_id) if transfer.season_id else None
        if player is None or season is None:
            skipped.append(f"transfer#{transfer.id}: missing player/season link")
            continue

        prior_label = _prior_season_label(season.label)
        prior_season = db.query(models.Season).filter_by(label=prior_label).first()
        if prior_season is None:
            skipped.append(f"{player.name} ({season.label}): prior season {prior_label} not loaded")
            continue

        stats = (
            db.query(models.PlayerSeasonStats)
            .filter_by(player_id=player.id, season_id=prior_season.id)
            .first()
        )
        if stats is None:
            skipped.append(f"{player.name}: no {prior_label} stats loaded (transfer in {season.label})")
            continue

        age_at_transfer = None
        if player.date_of_birth is not None and transfer.transfer_date is not None:
            # Real age in years at the (approximate) transfer date, computed from
            # a real birth year sourced from FBref (see fbref_provider.py) -- not
            # backfilled or guessed when missing, per the project's anti-
            # fabrication rule. Age is a well-documented major driver of transfer
            # valuations that goals/assists/minutes alone don't capture.
            age_at_transfer = (transfer.transfer_date - player.date_of_birth).days / 365.25

        rows.append(
            {
                "player_name": player.name,
                "position": player.position.value,
                "transfer_season": season.label,
                "stats_season": prior_label,
                "age_at_transfer": age_at_transfer,
                "minutes": stats.minutes,
                "goals": stats.goals,
                "assists": stats.assists,
                "xg": float(stats.xg) if stats.xg is not None else None,
                "xa": float(stats.xa) if stats.xa is not None else None,
                "fee_eur_millions": float(transfer.fee_amount),
                "log_fee": float(np.log1p(float(transfer.fee_amount))),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_per90_columns(df)

    return MatrixResult(
        df=df,
        transfers_considered=len(transfers),
        matched=len(rows),
        skipped_no_prior_season_stats=skipped,
    )


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.db import SessionLocal

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db = SessionLocal()
    try:
        result = assemble_training_matrix(db)
    finally:
        db.close()

    logger.info(
        "Transfers considered (disclosed permanent fee): %d\nMatched to prior-season stats: %d\nSkipped: %d",
        result.transfers_considered,
        result.matched,
        len(result.skipped_no_prior_season_stats),
    )
    if result.matched == 0:
        logger.info("No matched rows -- nothing to train on yet. See skip reasons below (first 15):")
        for reason in result.skipped_no_prior_season_stats[:15]:
            logger.info("  - %s", reason)
        return

    out_path = REPO_ROOT / "data" / "processed" / "training_matrix.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(result.df), out_path)


if __name__ == "__main__":
    main()
