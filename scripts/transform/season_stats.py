"""
Combine a player's stats rows for one season into a single record.

A player who changes club mid-season has one PlayerSeasonStats row per
club. Both the training matrix and live predictions must see the whole
season, not one arbitrary stint, so counting stats are summed across
stints. The club used for club-level features (is_top_six) is the stint
with the most minutes -- an approximation, since stats rows carry no
dates to tell which stint came last.
"""
from __future__ import annotations

from collections.abc import Iterable

COUNT_FIELDS = ("minutes", "goals", "assists", "tackles", "interceptions")


def _sum_or_none(values: Iterable) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def combine_stints(stints: list) -> dict:
    """`stints` are PlayerSeasonStats-like objects for ONE player and ONE
    season. Returns summed counting stats, xg/xa (summed, None if absent
    everywhere), the primary club_id and the number of stints combined."""
    if not stints:
        raise ValueError("combine_stints needs at least one stats row")
    combined = {f: _sum_or_none(getattr(s, f) for s in stints) for f in COUNT_FIELDS}
    for f in ("xg", "xa"):
        combined[f] = _sum_or_none(
            float(getattr(s, f)) if getattr(s, f, None) is not None else None for s in stints
        )
    primary = max(stints, key=lambda s: s.minutes or 0)
    combined["club_id"] = primary.club_id
    combined["n_stints"] = len(stints)
    return combined
