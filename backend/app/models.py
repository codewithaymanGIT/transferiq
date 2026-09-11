"""
SQLAlchemy 2.0 ORM models -- the Python-side mirror of database/schema.sql.

Kept as the single source of truth for Alembic migrations
(`alembic revision --autogenerate`). If you change a table here, update
database/schema.sql to match (or regenerate it from this file) so the two
never drift apart.
"""
from __future__ import annotations

import enum
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PositionGroup(str, enum.Enum):
    GK = "GK"
    DF = "DF"
    MF = "MF"
    FW = "FW"


class PreferredFoot(str, enum.Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BOTH = "BOTH"


class TransferType(str, enum.Enum):
    PERMANENT = "PERMANENT"
    LOAN = "LOAN"
    FREE = "FREE"
    LOAN_WITH_OPTION = "LOAN_WITH_OPTION"


class FeeCurrency(str, enum.Enum):
    EUR = "EUR"
    GBP = "GBP"
    USD = "USD"


# Every native Postgres ENUM type below MUST be named to exactly match the
# `CREATE TYPE ... AS ENUM` names in database/schema.sql. Without an
# explicit `name=`, SQLAlchemy derives one from the Python class name
# (e.g. FeeCurrency -> "feecurrency") which does NOT match schema.sql's
# "fee_currency" -- single-row inserts can silently dodge this mismatch
# (Postgres infers the type from the target column with no cast needed),
# but SQLAlchemy's bulk/"insertmany" fast path adds an explicit `::name`
# cast and fails with "type ... does not exist" the moment more than one
# row of that type is flushed together. Pin every one explicitly so this
# can't resurface for a column that happens to load one-row-at-a-time today.
POSITION_GROUP_ENUM = Enum(PositionGroup, name="position_group")
PREFERRED_FOOT_ENUM = Enum(PreferredFoot, name="preferred_foot")
TRANSFER_TYPE_ENUM = Enum(TransferType, name="transfer_type")
FEE_CURRENCY_ENUM = Enum(FeeCurrency, name="fee_currency")


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    tier: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)
    country: Mapped[str | None] = mapped_column(String, nullable=True)

    clubs: Mapped[list["Club"]] = relationship(back_populates="competition")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (
        CheckConstraint("end_date > start_date", name="seasons_valid_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)


class Club(Base):
    __tablename__ = "clubs"
    __table_args__ = (UniqueConstraint("name", "competition_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"))
    country: Mapped[str | None] = mapped_column(String, nullable=True)

    competition: Mapped[Competition | None] = relationship(back_populates="clubs")


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    license_note: Mapped[str | None] = mapped_column(String, nullable=True)
    reliability_note: Mapped[str | None] = mapped_column(String, nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (
        CheckConstraint("height_cm IS NULL OR height_cm BETWEEN 140 AND 220", name="players_height_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_player_id: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    nationality: Mapped[str | None] = mapped_column(String, nullable=True)
    position: Mapped[PositionGroup] = mapped_column(POSITION_GROUP_ENUM, nullable=False, index=True)
    foot: Mapped[PreferredFoot | None] = mapped_column(PREFERRED_FOOT_ENUM, nullable=True)
    height_cm: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    current_club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    season_stats: Mapped[list["PlayerSeasonStats"]] = relationship(back_populates="player")
    transfers: Mapped[list["Transfer"]] = relationship(back_populates="player")


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (UniqueConstraint("player_id", "season_id", "club_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"))
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), nullable=False, index=True)

    minutes: Mapped[int] = mapped_column(default=0, nullable=False)
    starts: Mapped[int | None]
    apps: Mapped[int | None]

    goals: Mapped[int | None]
    assists: Mapped[int | None]
    xg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    xa: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    shots: Mapped[int | None]
    shots_on_target: Mapped[int | None]

    key_passes: Mapped[int | None]
    prog_passes: Mapped[int | None]
    prog_carries: Mapped[int | None]
    pass_completion_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))

    tackles: Mapped[int | None]
    interceptions: Mapped[int | None]
    clearances: Mapped[int | None]
    blocks: Mapped[int | None]
    aerial_duel_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))

    saves: Mapped[int | None]
    save_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    goals_conceded: Mapped[int | None]
    psxg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))

    source_id: Mapped[int | None] = mapped_column(ForeignKey("data_sources.id"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    player: Mapped[Player] = relationship(back_populates="season_stats")


class Transfer(Base):
    __tablename__ = "transfers"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    from_club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"))
    to_club_id: Mapped[int | None] = mapped_column(ForeignKey("clubs.id"))
    transfer_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    fee_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    fee_currency: Mapped[FeeCurrency | None] = mapped_column(FEE_CURRENCY_ENUM)
    fee_disclosed: Mapped[bool] = mapped_column(default=True, nullable=False)
    transfer_type: Mapped[TransferType] = mapped_column(TRANSFER_TYPE_ENUM, nullable=False)
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"))
    source_id: Mapped[int | None] = mapped_column(ForeignKey("data_sources.id"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    player: Mapped[Player] = relationship(back_populates="transfers")


class MarketValue(Base):
    """External benchmark valuation -- NEVER the model's own prediction (see Prediction)."""

    __tablename__ = "market_values"
    __table_args__ = (UniqueConstraint("player_id", "valuation_date", "source_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    valuation_date: Mapped[date] = mapped_column(Date, nullable=False)
    value_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[FeeCurrency] = mapped_column(FEE_CURRENCY_ENUM, default=FeeCurrency.EUR, nullable=False)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("data_sources.id"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    feature_list_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    hyperparams_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)


class Prediction(Base):
    """This project's own model output -- kept structurally separate from MarketValue."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "low_bound <= predicted_value AND predicted_value <= high_bound",
            name="predictions_bounds_valid",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False, index=True)
    predicted_value: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    low_bound: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    high_bound: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Watchlist(Base):
    __tablename__ = "watchlists"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
