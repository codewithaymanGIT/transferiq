"""
TransferIQ API.

Endpoints implemented in this slice: health check, player listing/detail,
and valuation lookup. Valuation deliberately returns 404 (not a fabricated
number) when no `Prediction` row exists yet for a player -- there is no
trained model in this phase, so the honest response is "not available",
never a made-up figure.
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models, schemas
from .db import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transferiq")

app = FastAPI(
    title="TransferIQ API",
    description="Football market-value intelligence: player data, "
    "gradient-boosted valuation predictions, and SHAP-based explanations.",
    version="0.1.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/players", response_model=schemas.PlayerListResponse, tags=["players"])
def list_players(
    db: Session = Depends(get_db),
    position: schemas.PositionLiteral | None = Query(default=None),
    club_id: int | None = Query(default=None),
    q: str | None = Query(default=None, description="Case-insensitive substring match on player name."),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> schemas.PlayerListResponse:
    stmt = select(models.Player)
    count_stmt = select(func.count()).select_from(models.Player)

    if position is not None:
        stmt = stmt.where(models.Player.position == position)
        count_stmt = count_stmt.where(models.Player.position == position)
    if club_id is not None:
        stmt = stmt.where(models.Player.current_club_id == club_id)
        count_stmt = count_stmt.where(models.Player.current_club_id == club_id)
    if q:
        name_filter = models.Player.name.ilike(f"%{q}%")
        stmt = stmt.where(name_filter)
        count_stmt = count_stmt.where(name_filter)

    total = db.scalar(count_stmt) or 0
    stmt = stmt.order_by(models.Player.name).offset((page - 1) * page_size).limit(page_size)
    rows = db.scalars(stmt).all()

    return schemas.PlayerListResponse(
        items=[schemas.PlayerOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/api/v1/players/{player_id}", response_model=schemas.PlayerOut, tags=["players"])
def get_player(player_id: int, db: Session = Depends(get_db)) -> schemas.PlayerOut:
    player = db.get(models.Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return schemas.PlayerOut.model_validate(player)


@app.get(
    "/api/v1/players/{player_id}/valuation",
    response_model=schemas.ValuationOut,
    tags=["players"],
    responses={404: {"model": schemas.ErrorResponse}},
)
def get_valuation(player_id: int, db: Session = Depends(get_db)) -> schemas.ValuationOut:
    player = db.get(models.Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    latest_pred = db.scalar(
        select(models.Prediction)
        .where(models.Prediction.player_id == player_id)
        .order_by(models.Prediction.predicted_at.desc())
    )
    if latest_pred is None:
        # Honest response: no trained model has scored this player yet.
        # Never fabricate a placeholder valuation.
        raise HTTPException(
            status_code=404,
            detail=f"No model valuation available yet for player {player_id}.",
        )

    model_version = db.get(models.ModelVersion, latest_pred.model_version_id)

    latest_benchmark = db.scalar(
        select(models.MarketValue)
        .where(models.MarketValue.player_id == player_id)
        .order_by(models.MarketValue.valuation_date.desc())
    )

    benchmark_value = latest_benchmark.value_amount if latest_benchmark else None
    benchmark_diff_pct = None
    if benchmark_value and benchmark_value > 0:
        benchmark_diff_pct = float((latest_pred.predicted_value - benchmark_value) / benchmark_value * 100)

    return schemas.ValuationOut(
        player_id=player_id,
        model_version=model_version.name if model_version else "unknown",
        predicted_value=latest_pred.predicted_value,
        low_bound=latest_pred.low_bound,
        high_bound=latest_pred.high_bound,
        confidence=latest_pred.confidence,  # type: ignore[arg-type]
        benchmark_value=benchmark_value,
        benchmark_difference_pct=benchmark_diff_pct,
        shap_contributions=latest_pred.shap_contributions or {},
    )
