"""
Per-90 normalization. A player with 90 minutes and 1 goal must not look
statistically equivalent to a player with 3,000 minutes and 30 goals
(both are "1.0 goals per 90") -- below a minimum-minutes threshold, shrink
the per-90 rate toward the positional mean using simple empirical-Bayes
shrinkage, weighted by how little data backs the raw rate.
"""
from __future__ import annotations

import pandas as pd

MIN_MINUTES_THRESHOLD = 450  # ~5 full matches; below this, raw per-90 is noisy
_PER90_SOURCE_COLUMNS = [
    "goals", "assists", "xg", "xa", "shots", "sot", "key_passes",
    "prog_passes", "prog_carries", "tackles", "interceptions", "clearances", "blocks",
]


def add_per90_columns(
    df: pd.DataFrame,
    shrinkage_prior_minutes: int = MIN_MINUTES_THRESHOLD,
) -> pd.DataFrame:
    """
    Adds `<stat>_per90` columns using empirical-Bayes shrinkage:

        shrunk_per90 = (raw_total + prior_rate * k) / (minutes/90 + k)

    where `prior_rate` is the positional-mean per-90 rate (computed from
    players *above* the minutes threshold, so noisy low-minute rows never
    contaminate the prior they're being shrunk toward) and `k` is the
    shrinkage strength in "90-minute units" (`shrinkage_prior_minutes / 90`).

    Players with 0 minutes get NaN, not a division-by-zero or fabricated 0.
    """
    out = df.copy()
    minutes_90 = out["minutes"].astype(float) / 90.0
    k = shrinkage_prior_minutes / 90.0

    above_threshold = out["minutes"] >= shrinkage_prior_minutes

    for col in _PER90_SOURCE_COLUMNS:
        if col not in out.columns:
            continue
        total = pd.to_numeric(out[col], errors="coerce")

        # Positional-mean prior rate, computed per position from qualified players only.
        qualified = out.loc[above_threshold]
        qualified_total = pd.to_numeric(qualified[col], errors="coerce")
        qualified_minutes_90 = qualified["minutes"].astype(float) / 90.0
        prior_by_position = (
            (qualified_total / qualified_minutes_90.replace(0, pd.NA))
            .groupby(qualified["position"])
            .mean()
        )

        prior_rate = out["position"].map(prior_by_position)
        # Global fallback for positions with no qualified players in this slice.
        global_prior = (qualified_total / qualified_minutes_90.replace(0, pd.NA)).mean()
        prior_rate = prior_rate.fillna(global_prior)

        shrunk = (total + prior_rate * k) / (minutes_90 + k)

        # Zero-minute players: no meaningful rate, not a fabricated 0.
        shrunk = shrunk.where(out["minutes"] > 0, other=pd.NA)
        out[f"{col}_per90"] = shrunk

    out["qualifies_min_minutes"] = above_threshold
    return out
