"""
API tests for the player/valuation endpoints. Uses an in-memory SQLite DB
via FastAPI's dependency override so these run fast, offline, and in CI --
Postgres-specific behavior (real ENUM/JSONB types) is instead covered by
integration tests that run against docker-compose's Postgres service
(see tests/backend/README.md, added when that harness lands in Phase 8).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app import models  # noqa: E402
from app.db import get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    models.Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()


@pytest.fixture()
def client(db_session: Session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed_player(db: Session, **overrides) -> models.Player:
    defaults = dict(name="Sample Winger", position=models.PositionGroup.FW)
    defaults.update(overrides)
    player = models.Player(**defaults)
    db.add(player)
    db.commit()
    db.refresh(player)
    return player


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_players_empty(client: TestClient):
    resp = client.get("/api/v1/players")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_players_returns_seeded_rows(client: TestClient, db_session: Session):
    _seed_player(db_session, name="Bukayo Saka", position=models.PositionGroup.MF)
    _seed_player(db_session, name="Erling Haaland", position=models.PositionGroup.FW)

    resp = client.get("/api/v1/players")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    names = {p["name"] for p in body["items"]}
    assert names == {"Bukayo Saka", "Erling Haaland"}


def test_list_players_filters_by_position(client: TestClient, db_session: Session):
    _seed_player(db_session, name="Bukayo Saka", position=models.PositionGroup.MF)
    _seed_player(db_session, name="Erling Haaland", position=models.PositionGroup.FW)

    resp = client.get("/api/v1/players", params={"position": "FW"})
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "Erling Haaland"


def test_get_player_404_for_missing_id(client: TestClient):
    resp = client.get("/api/v1/players/9999")
    assert resp.status_code == 404


def test_valuation_404_when_no_prediction_exists(client: TestClient, db_session: Session):
    """Core anti-fabrication contract: no Prediction row -> 404, not a made-up number."""
    player = _seed_player(db_session)

    resp = client.get(f"/api/v1/players/{player.id}/valuation")
    assert resp.status_code == 404
    assert "No model valuation available" in resp.json()["detail"]


def test_valuation_returns_real_prediction_and_benchmark_diff(client: TestClient, db_session: Session):
    player = _seed_player(db_session, name="Declan Rice")

    model_version = models.ModelVersion(
        name="xgb_quantile_v1",
        trained_at=datetime.datetime.now(datetime.timezone.utc),
        metrics_json={"mae": 4.1},
        feature_list_json={"features": ["xg", "age"]},
        hyperparams_json={"n_estimators": 300},
        is_active=True,
    )
    db_session.add(model_version)
    db_session.flush()

    db_session.add(
        models.MarketValue(
            player_id=player.id,
            valuation_date=datetime.date(2025, 1, 1),
            value_amount=70_000_000,
            currency=models.FeeCurrency.EUR,
        )
    )
    db_session.add(
        models.Prediction(
            player_id=player.id,
            model_version_id=model_version.id,
            predicted_value=82_400_000,
            low_bound=72_000_000,
            high_bound=94_000_000,
            confidence="High",
        )
    )
    db_session.commit()

    resp = client.get(f"/api/v1/players/{player.id}/valuation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "xgb_quantile_v1"
    assert float(body["predicted_value"]) == 82_400_000
    assert float(body["benchmark_value"]) == 70_000_000
    # (82.4M - 70M) / 70M * 100 = ~17.71%
    assert body["benchmark_difference_pct"] == pytest.approx(17.71, abs=0.01)


def test_valuation_bounds_are_never_inverted(client: TestClient, db_session: Session):
    """Regression guard for the predictions_bounds_valid CHECK constraint."""
    player = _seed_player(db_session)
    model_version = models.ModelVersion(
        name="xgb_v1",
        trained_at=datetime.datetime.now(datetime.timezone.utc),
        metrics_json={},
        feature_list_json={},
        hyperparams_json={},
    )
    db_session.add(model_version)
    db_session.flush()

    bad_prediction = models.Prediction(
        player_id=player.id,
        model_version_id=model_version.id,
        predicted_value=1,
        low_bound=100,
        high_bound=200,
        confidence="Low",
    )
    db_session.add(bad_prediction)
    with pytest.raises(Exception):
        db_session.commit()
