"""
Longitudinal + age features -- built with an explicit leakage guard.

The core rule (blueprint section 14): when building features "as of" a
given season/valuation date, only seasons that ended strictly before that
date may be used. `build_longitudinal_features` takes an explicit
`as_of_season_end` per row rather than assuming "the row before this one
in the DataFrame" is safe, because a naive shift() over an unsorted or
multi-source DataFrame is exactly how leakage sneaks in.
"""
from __future__ import annotations

import pandas as pd


def build_longitudinal_features(
    player_season_df: pd.DataFrame,
    season_end_dates: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """
    player_season_df: one row per (player, season), must include
        'name_key', 'season', and the per-90 columns to build trends from.
    season_end_dates: maps season label -> end date, used to sort strictly
        chronologically per player (never trust row order).

    Adds, for every `<stat>_per90` column already present:
        prev_season_<stat>_per90   -- value from the immediately prior season
        growth_<stat>_per90        -- current - previous (NaN if no prior season)
    """
    df = player_season_df.copy()
    df["_season_end"] = df["season"].map(season_end_dates)
    if df["_season_end"].isna().any():
        unknown = sorted(df.loc[df["_season_end"].isna(), "season"].unique())
        raise ValueError(f"season_end_dates missing entries for: {unknown}")

    df = df.sort_values(["name_key", "_season_end"])

    per90_cols = [c for c in df.columns if c.endswith("_per90")]
    for col in per90_cols:
        prev_col = f"prev_season_{col}"
        growth_col = f"growth_{col}"
        # groupby+shift only ever looks at a player's OWN strictly-earlier
        # row, because we sorted by _season_end above -- this is the
        # leakage guard in practice, not just documentation.
        df[prev_col] = df.groupby("name_key")[col].shift(1)
        df[growth_col] = df[col] - df[prev_col]

    return df.drop(columns="_season_end")


def add_age_features(df: pd.DataFrame, as_of_date_col: str = "_season_end", dob_col: str = "date_of_birth") -> pd.DataFrame:
    """
    Nonlinear age features: raw age, age^2, and a coarse bucket. Left for
    tree models to interact with position rather than hand-coding an
    age x position formula (per blueprint section 9).
    """
    out = df.copy()
    dob = pd.to_datetime(out[dob_col], errors="coerce")
    as_of = pd.to_datetime(out[as_of_date_col], errors="coerce")
    age_years = (as_of - dob).dt.days / 365.25

    out["age"] = age_years
    out["age_squared"] = age_years**2
    out["age_bucket"] = pd.cut(
        age_years,
        bins=[0, 21, 24, 28, 32, 100],
        labels=["u21", "21-23", "24-27", "28-31", "32plus"],
    )
    return out
