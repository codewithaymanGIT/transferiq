"""Pydantic schemas -- the API's public contract. Kept separate from the
ORM models (app/models.py) on purpose: the DB shape and the API shape are
allowed to diverge (e.g. we never expose password_hash, we reshape
predictions into a friendlier envelope, etc.)."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PositionLiteral = Literal["GK", "DF", "MF", "FW"]


class ClubOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    date_of_birth: date | None
    nationality: str | None
    position: PositionLiteral
    current_club_id: int | None
    club_name: str | None = None
    last_season_label: str | None = None


class PlayerListResponse(BaseModel):
    items: list[PlayerOut]
    total: int
    page: int
    page_size: int


class ValuationOut(BaseModel):
    """Response for GET /api/v1/players/{id}/valuation."""

    player_id: int
    model_version: str
    predicted_value: Decimal
    low_bound: Decimal
    high_bound: Decimal
    confidence: Literal["High", "Medium", "Low"]
    benchmark_value: Decimal | None = Field(
        default=None, description="External market-value benchmark, if available -- never the model's own output."
    )
    benchmark_difference_pct: float | None = None
    shap_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Feature -> signed £ contribution to the prediction, from the real SHAP explainer.",
    )


class ErrorResponse(BaseModel):
    detail: str


class TopValuationOut(BaseModel):
    """One row in the top-valuations leaderboard -- reuses the same real
    Prediction rows already served by /players/{id}/valuation, just
    re-sorted and joined with player identity for a ranked view."""
    player_id: int
    player_name: str
    position: PositionLiteral
    club_name: str | None
    predicted_value: str
    confidence: str
    top_driver_feature: str | None
    top_driver_value: float | None


class TopValuationsResponse(BaseModel):
    items: list[TopValuationOut]
    total: int
    page: int
    page_size: int
