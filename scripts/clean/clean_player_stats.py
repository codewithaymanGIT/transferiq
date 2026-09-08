"""
Clean a raw player-season-stats DataFrame (the shape every provider in
scripts/ingest/providers returns) into a form ready for validation and
loading. Deliberately conservative: this step never invents a value for
a missing stat -- it only standardizes types and identity, and flags rows
that need manual attention.
"""
from __future__ import annotations

import logging

import pandas as pd

from name_matching import normalize_name

logger = logging.getLogger(__name__)

# Columns that should be non-negative integers when present; anything
# outside this after coercion becomes NaN (never silently clamped to 0,
# which would be indistinguishable from a genuine zero).
_NONNEGATIVE_INT_COLUMNS = [
    "minutes", "starts", "apps", "goals", "assists", "shots", "sot",
    "key_passes", "prog_passes", "prog_carries", "tackles", "interceptions",
    "clearances", "blocks", "saves", "goals_conceded",
]

_FLOAT_COLUMNS = ["xg", "xa"]


def clean_player_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with:
    - a `name_key` column (normalized join key, see name_matching.py)
    - integer/float columns coerced, invalid values -> NaN (not 0)
    - exact duplicate (name_key, season, club) rows collapsed, keeping the
      row with the most non-null fields (a naive but transparent tie-break)
    - a `_clean_flags` column listing anything worth manual review
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["name_key"] = out["player_name"].map(normalize_name)

    for col in _NONNEGATIVE_INT_COLUMNS:
        if col not in out.columns:
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        invalid_negative = coerced < 0
        coerced = coerced.where(~invalid_negative, other=pd.NA)
        out[col] = coerced.astype("Int64")

    for col in _FLOAT_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["_clean_flags"] = out.apply(_row_flags, axis=1)

    dedup_keys = [k for k in ("name_key", "season", "club") if k in out.columns]
    if dedup_keys:
        out["_non_null_count"] = out.notna().sum(axis=1)
        out = (
            out.sort_values("_non_null_count", ascending=False)
            .drop_duplicates(subset=dedup_keys, keep="first")
            .drop(columns="_non_null_count")
            .reset_index(drop=True)
        )

    n_flagged = (out["_clean_flags"].str.len() > 0).sum()
    if n_flagged:
        logger.info("clean_player_season_stats: %d/%d rows flagged for review", n_flagged, len(out))

    return out


def _row_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []

    minutes = row.get("minutes")
    if pd.isna(minutes):
        flags.append("missing_minutes")
    elif minutes == 0 and (row.get("goals") or 0) not in (0, None):
        flags.append("goals_with_zero_minutes")

    if pd.isna(row.get("name_key")) or row.get("name_key") == "":
        flags.append("unresolvable_name")

    goals = row.get("goals")
    xg = row.get("xg")
    if pd.notna(goals) and pd.notna(xg) and xg == 0 and goals and goals > 0:
        # Not necessarily wrong (xG=0 with goals happens, e.g. own-goal-adjacent
        # oddities), but worth a human glance rather than silent trust.
        flags.append("goals_without_xg")

    return flags
