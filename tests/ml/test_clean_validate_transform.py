"""
Tests for scripts/clean, scripts/validate, scripts/transform.

All data here is synthetic and clearly labeled as such (player names like
"Player A") -- this exercises the pipeline logic, not a claim about any
real player's stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "clean"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validate"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "transform"))

from clean_player_stats import clean_player_season_stats  # noqa: E402
from name_matching import name_similarity, normalize_name  # noqa: E402
from validate_player_stats import validate_player_season_stats  # noqa: E402
from per90 import add_per90_columns  # noqa: E402
from longitudinal import add_age_features, build_longitudinal_features  # noqa: E402


# ---------------------------------------------------------------- cleaning

def test_normalize_name_strips_accents_and_case():
    assert normalize_name("Ödegaard") == "odegaard"
    assert normalize_name("N'Golo Kanté") == "n golo kante"
    assert normalize_name("  Bukayo   Saka ") == "bukayo saka"


def test_name_similarity_handles_partial_names():
    assert name_similarity("Bukayo Saka", "Saka") == pytest.approx(0.5)
    assert name_similarity("Bukayo Saka", "Bukayo Saka") == 1.0
    assert name_similarity("Bukayo Saka", "Erling Haaland") == 0.0


def test_clean_coerces_negative_values_to_missing_not_zero():
    df = pd.DataFrame(
        {
            "player_name": ["Player A"],
            "season": ["2024-2025"],
            "club": ["Test FC"],
            "minutes": [-5],  # corrupt input
            "goals": [3],
        }
    )
    cleaned = clean_player_season_stats(df)
    assert pd.isna(cleaned.loc[0, "minutes"])  # NOT coerced to 0


def test_clean_dedupes_keeping_more_complete_row():
    df = pd.DataFrame(
        {
            "player_name": ["Player A", "Player A"],
            "season": ["2024-2025", "2024-2025"],
            "club": ["Test FC", "Test FC"],
            "minutes": [2000, 2000],
            "goals": [None, 10],  # second row more complete
            "assists": [5, 5],
        }
    )
    cleaned = clean_player_season_stats(df)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "goals"] == 10


def test_clean_flags_goals_with_zero_minutes():
    df = pd.DataFrame(
        {
            "player_name": ["Player A"],
            "season": ["2024-2025"],
            "club": ["Test FC"],
            "minutes": [0],
            "goals": [2],
        }
    )
    cleaned = clean_player_season_stats(df)
    assert "goals_with_zero_minutes" in cleaned.loc[0, "_clean_flags"]


# --------------------------------------------------------------- validation

def test_validation_passes_clean_data():
    df = pd.DataFrame(
        {
            "player_name": ["Player A"],
            "name_key": ["player a"],
            "position": ["FW"],
            "minutes": [2450],
            "goals": [18],
            "season": ["2024-2025"],
            "source": ["test"],
        }
    )
    report = validate_player_season_stats(df)
    assert report.is_clean


def test_validation_catches_out_of_range_minutes():
    df = pd.DataFrame(
        {
            "player_name": ["Player A"],
            "name_key": ["player a"],
            "position": ["FW"],
            "minutes": [50000],  # clearly a unit/parsing error
            "season": ["2024-2025"],
            "source": ["test"],
        }
    )
    report = validate_player_season_stats(df)
    assert not report.is_clean
    assert any(item["column"] == "minutes" for item in report.out_of_range)


def test_validation_catches_invalid_position():
    df = pd.DataFrame(
        {
            "player_name": ["Player A"],
            "name_key": ["player a"],
            "position": ["STRIKER"],  # not in {GK, DF, MF, FW}
            "minutes": [1000],
            "season": ["2024-2025"],
            "source": ["test"],
        }
    )
    report = validate_player_season_stats(df)
    assert not report.is_clean
    assert len(report.invalid_positions) == 1


def test_validation_reports_missing_columns():
    df = pd.DataFrame({"player_name": ["Player A"]})
    report = validate_player_season_stats(df)
    assert not report.is_clean
    assert "position" in report.missing_columns


# --------------------------------------------------------------- per-90

def test_per90_high_minutes_matches_raw_rate_closely():
    df = pd.DataFrame(
        {
            "position": ["FW", "FW", "FW"],
            "minutes": [2700, 2700, 2700],
            "goals": [27, 18, 9],  # 0.9, 0.6, 0.3 per 90
        }
    )
    result = add_per90_columns(df, shrinkage_prior_minutes=450)
    # With 2700 minutes >> 450-minute prior, shrinkage should barely move the raw rate.
    assert result.loc[0, "goals_per90"] == pytest.approx(0.9, abs=0.05)


def test_per90_low_minutes_shrinks_toward_positional_mean():
    df = pd.DataFrame(
        {
            "position": ["FW", "FW", "FW"],
            "minutes": [2700, 2700, 90],  # third player: one match, one goal
            "goals": [9, 9, 1],  # qualified players average 0.3 goals/90
        }
    )
    result = add_per90_columns(df, shrinkage_prior_minutes=450)
    low_minutes_rate = result.loc[2, "goals_per90"]
    # Naive rate would be 1 goal / 1 match = 1.0 per90 -- shrinkage must pull
    # this DOWN toward the ~0.3 positional mean, not leave it at the raw rate.
    assert low_minutes_rate < 1.0
    assert low_minutes_rate == pytest.approx(0.3, abs=0.15)


def test_per90_zero_minutes_is_missing_not_zero():
    df = pd.DataFrame({"position": ["FW"], "minutes": [0], "goals": [0]})
    result = add_per90_columns(df)
    assert pd.isna(result.loc[0, "goals_per90"])


# --------------------------------------------------------------- longitudinal

def test_longitudinal_growth_only_uses_strictly_prior_season():
    df = pd.DataFrame(
        {
            "name_key": ["player a", "player a", "player a"],
            "season": ["2022-2023", "2023-2024", "2024-2025"],
            "goals_per90": [0.2, 0.4, 0.6],
        }
    )
    season_ends = {
        "2022-2023": pd.Timestamp("2023-05-31"),
        "2023-2024": pd.Timestamp("2024-05-31"),
        "2024-2025": pd.Timestamp("2025-05-31"),
    }
    result = build_longitudinal_features(df, season_ends)

    row_2024 = result[result["season"] == "2024-2025"].iloc[0]
    assert row_2024["prev_season_goals_per90"] == pytest.approx(0.4)
    assert row_2024["growth_goals_per90"] == pytest.approx(0.2)

    row_first = result[result["season"] == "2022-2023"].iloc[0]
    assert pd.isna(row_first["prev_season_goals_per90"])


def test_longitudinal_raises_on_unknown_season():
    df = pd.DataFrame({"name_key": ["player a"], "season": ["1999-2000"], "goals_per90": [0.1]})
    with pytest.raises(ValueError, match="1999-2000"):
        build_longitudinal_features(df, season_end_dates={})


def test_longitudinal_does_not_leak_across_players():
    """Two different players must never see each other's 'previous season'."""
    df = pd.DataFrame(
        {
            "name_key": ["player a", "player b"],
            "season": ["2024-2025", "2024-2025"],
            "goals_per90": [0.5, 0.9],
        }
    )
    season_ends = {"2024-2025": pd.Timestamp("2025-05-31")}
    result = build_longitudinal_features(df, season_ends)
    assert result["prev_season_goals_per90"].isna().all()


def test_age_features_nonlinear_and_bucketed():
    df = pd.DataFrame(
        {
            "date_of_birth": ["2003-09-21"],  # ~21 at the given as-of date
            "_season_end": ["2025-05-31"],
        }
    )
    result = add_age_features(df)
    assert result.loc[0, "age"] == pytest.approx(21.7, abs=0.1)
    assert result.loc[0, "age_squared"] == pytest.approx(21.7**2, abs=5)
    assert result.loc[0, "age_bucket"] == "21-23"
