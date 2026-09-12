"""
Core load logic: a cleaned/validated player-season DataFrame -> the real
schema (competitions, seasons, clubs, players, player_season_stats).

Deliberately split from the CLI/file-reading wrapper (load_to_postgres.py)
so this can be unit-tested against an in-memory SQLite session without a
real Postgres instance or a real parquet file on disk.

Idempotent: re-running against the same season updates existing rows
(matched on player name + club/season) rather than duplicating them --
important since ingest will be re-run periodically as new data comes in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)

_VALID_POSITIONS = {"GK", "DF", "MF", "FW"}


@dataclass
class LoadResult:
    players_created: int = 0
    players_updated: int = 0
    stats_rows_written: int = 0
    rows_skipped_missing_identity: int = 0
    rows_skipped_unparseable_position: list[str] = field(default_factory=list)


def primary_position(raw: str | None) -> str | None:
    """
    FBref lists multi-eligible positions like 'FW,MF' -- the schema's
    `position` column is a single required value, so take the first listed
    position as primary rather than guessing which is "real". Returns None
    (not a fabricated default) if nothing recognizable is found.
    """
    if not raw or pd.isna(raw):
        return None
    first = str(raw).split(",")[0].strip().upper()
    return first if first in _VALID_POSITIONS else None


def season_bounds(season_label: str) -> tuple[date, date]:
    start_year, end_year = season_label.split("-")
    return date(int(start_year), 8, 1), date(int(end_year), 5, 31)


def get_or_create_competition(db: Session, name: str) -> models.Competition:
    comp = db.query(models.Competition).filter_by(name=name).first()
    if comp is None:
        comp = models.Competition(name=name, tier=1, country="England")
        db.add(comp)
        db.flush()
    return comp


def get_or_create_season(db: Session, label: str) -> models.Season:
    season = db.query(models.Season).filter_by(label=label).first()
    if season is None:
        start, end = season_bounds(label)
        season = models.Season(label=label, start_date=start, end_date=end)
        db.add(season)
        db.flush()
    return season


def get_or_create_club(db: Session, name: str, competition_id: int) -> models.Club:
    club = db.query(models.Club).filter_by(name=name, competition_id=competition_id).first()
    if club is None:
        club = models.Club(name=name, competition_id=competition_id, country="England")
        db.add(club)
        db.flush()
    return club


def get_or_create_data_source(db: Session, name: str, url: str) -> models.DataSource:
    source = db.query(models.DataSource).filter_by(name=name).first()
    if source is None:
        source = models.DataSource(name=name, url=url)
        db.add(source)
        db.flush()
    return source


def _birth_year_to_date(birth_year) -> "date | None":
    """FBref's 'born' column gives only a birth YEAR, not an exact date.
    Approximate with Jan 1 of that year -- documented as approximate, same
    pattern as _approximate_transfer_date below. Good enough for computing
    age-at-transfer to the nearest year, not for exact-birthday precision."""
    if birth_year is None or pd.isna(birth_year):
        return None
    return date(int(birth_year), 1, 1)


def _upsert_player(
    db: Session,
    name: str,
    position: str,
    club_id: int,
    birth_year=None,
    nationality: str | None = None,
) -> tuple[models.Player, bool]:
    """Returns (player, was_created)."""
    player = db.query(models.Player).filter_by(name=name).first()
    was_created = player is None
    dob = _birth_year_to_date(birth_year)
    clean_nationality = nationality if (nationality is not None and pd.notna(nationality)) else None
    if player is None:
        player = models.Player(
            name=name,
            position=models.PositionGroup(position),
            current_club_id=club_id,
            date_of_birth=dob,
            nationality=clean_nationality,
        )
        db.add(player)
        db.flush()
    else:
        # Players move clubs season to season -- keep this current with the latest pull.
        player.current_club_id = club_id
        player.position = models.PositionGroup(position)
        # Birth year/nationality are static real-world facts -- only fill if
        # we don't already have them; never overwrite a previously-set real
        # value with a possibly-missing one from an older/different pull.
        if player.date_of_birth is None and dob is not None:
            player.date_of_birth = dob
        if player.nationality is None and clean_nationality is not None:
            player.nationality = clean_nationality
    return player, was_created


def _upsert_season_stats(
    db: Session, player_id: int, club_id: int, season_id: int, row: pd.Series, source_id: int
) -> None:
    existing = (
        db.query(models.PlayerSeasonStats)
        .filter_by(player_id=player_id, season_id=season_id, club_id=club_id)
        .first()
    )
    fields = dict(
        minutes=int(row["minutes"]) if pd.notna(row.get("minutes")) else 0,
        goals=int(row["goals"]) if pd.notna(row.get("goals")) else None,
        assists=int(row["assists"]) if pd.notna(row.get("assists")) else None,
        xg=float(row["xg"]) if pd.notna(row.get("xg")) else None,
        xa=float(row["xa"]) if pd.notna(row.get("xa")) else None,
        source_id=source_id,
    )
    if existing is None:
        db.add(models.PlayerSeasonStats(player_id=player_id, club_id=club_id, season_id=season_id, **fields))
    else:
        for key, value in fields.items():
            setattr(existing, key, value)


@dataclass
class TransferLoadResult:
    transfers_created: int = 0
    transfers_already_loaded: int = 0
    rows_skipped_non_transfer: int = 0  # '-' rows: not a real transfer event
    rows_skipped_unmatched_player: list[str] = field(default_factory=list)


def _approximate_transfer_date(season: str, period: str) -> date:
    """
    The source data gives a season + window (Summer/Winter), not an exact
    date. Approximate: Aug 1 for the summer window, Jan 1 for winter --
    documented here as an approximation, not treated as precise.
    """
    start_year = int(season.split("-")[0])
    if str(period).strip().lower() == "winter":
        return date(start_year + 1, 1, 1)
    return date(start_year, 8, 1)


def load_transfers(db: Session, df: pd.DataFrame, source_name: str, source_url: str) -> TransferLoadResult:
    """
    Load real transfer records into the `transfers` table. Only rows for
    the 'in' direction are loaded (a player joining a PL club) to avoid
    double-counting the same real-world transfer that also appears as an
    'out' row for the selling PL club (or is simply absent if the buyer
    is outside the dataset's league scope).

    Players not already present in the `players` table (matched on exact
    name -- most transfer-era historical players won't overlap with the
    current FBref pull) are skipped and counted, never fabricated.
    """
    result = TransferLoadResult()
    source = get_or_create_data_source(db, source_name, source_url)
    comp = get_or_create_competition(db, "Premier League")

    in_rows = df[df["transfer_movement"] == "in"]
    for _, row in in_rows.iterrows():
        if pd.isna(row.get("transfer_type")):
            result.rows_skipped_non_transfer += 1
            continue

        player = db.query(models.Player).filter_by(name=row["player_name"]).first()
        if player is None:
            result.rows_skipped_unmatched_player.append(str(row["player_name"]))
            continue

        to_club = get_or_create_club(db, row["club"], comp.id)
        from_club = None
        if pd.notna(row.get("counterparty_club")):
            from_club = get_or_create_club(db, row["counterparty_club"], comp.id)

        season_row = get_or_create_season(db, row["season"]) if pd.notna(row.get("season")) else None
        transfer_date = _approximate_transfer_date(row["season"], row["transfer_period"])

        # Dedup key: same player, same approximate date, same destination club --
        # good enough to make re-running this script idempotent without a DB-level
        # unique constraint (the real-world event is unique on these in practice).
        existing = (
            db.query(models.Transfer)
            .filter_by(player_id=player.id, transfer_date=transfer_date, to_club_id=to_club.id)
            .first()
        )
        if existing is not None:
            result.transfers_already_loaded += 1
            continue

        db.add(
            models.Transfer(
                player_id=player.id,
                from_club_id=from_club.id if from_club else None,
                to_club_id=to_club.id,
                transfer_date=transfer_date,
                fee_amount=float(row["fee_eur_millions"]) if pd.notna(row.get("fee_eur_millions")) else None,
                fee_currency=models.FeeCurrency.EUR,
                fee_disclosed=bool(row.get("fee_disclosed", False)),
                transfer_type=models.TransferType(row["transfer_type"]),
                season_id=season_row.id if season_row else None,
                source_id=source.id,
            )
        )
        result.transfers_created += 1

    return result


def load_dataframe(
    db: Session,
    df: pd.DataFrame,
    season: str,
    source_name: str = "FBref",
    source_url: str = "https://fbref.com",
) -> LoadResult:
    """
    Load an already-cleaned player-season DataFrame into the schema.
    Caller is responsible for running it through clean_player_season_stats
    and validate_player_season_stats first -- this function assumes the
    minimum required columns (player_name, club, position, minutes,
    season) are present and does not re-validate them.
    """
    result = LoadResult()
    comp = get_or_create_competition(db, "Premier League")
    season_row = get_or_create_season(db, season)
    source = get_or_create_data_source(db, source_name, source_url)

    for _, row in df.iterrows():
        if pd.isna(row.get("player_name")) or pd.isna(row.get("club")):
            result.rows_skipped_missing_identity += 1
            continue

        position = primary_position(row.get("position"))
        if position is None:
            result.rows_skipped_unparseable_position.append(str(row.get("player_name")))
            continue

        club = get_or_create_club(db, row["club"], comp.id)
        player, created = _upsert_player(
            db,
            row["player_name"],
            position,
            club.id,
            birth_year=row.get("birth_year"),
            nationality=row.get("nationality"),
        )
        result.players_created += int(created)
        result.players_updated += int(not created)

        _upsert_season_stats(db, player.id, club.id, season_row.id, row, source.id)
        result.stats_rows_written += 1

    return result
