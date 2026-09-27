"""Tests for combining mid-season club stints (scripts/transform/season_stats.py)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "transform"))

from season_stats import combine_stints  # noqa: E402


def stint(club_id, minutes, goals=0, assists=0, tackles=None, interceptions=None, xg=None, xa=None):
    return SimpleNamespace(club_id=club_id, minutes=minutes, goals=goals, assists=assists,
                           tackles=tackles, interceptions=interceptions, xg=xg, xa=xa)


def test_single_stint_passes_through():
    out = combine_stints([stint(7, 1800, goals=5, tackles=20)])
    assert out["minutes"] == 1800 and out["goals"] == 5 and out["tackles"] == 20
    assert out["club_id"] == 7 and out["n_stints"] == 1


def test_two_stints_are_summed_and_primary_club_has_most_minutes():
    out = combine_stints([stint(1, 600, goals=2, assists=1), stint(2, 1500, goals=6, assists=3)])
    assert out["minutes"] == 2100 and out["goals"] == 8 and out["assists"] == 4
    assert out["club_id"] == 2 and out["n_stints"] == 2


def test_missing_values_stay_missing_not_zero():
    out = combine_stints([stint(1, 900), stint(2, 900)])
    assert out["tackles"] is None and out["xg"] is None


def test_partially_missing_values_sum_what_exists():
    out = combine_stints([stint(1, 900, tackles=10), stint(2, 900, tackles=None)])
    assert out["tackles"] == 10


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        combine_stints([])
