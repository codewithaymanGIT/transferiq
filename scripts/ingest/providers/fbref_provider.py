"""
FBref provider.

FBref (fbref.com) is the deepest free source for advanced stats (xG, xA,
progressive actions) but has no official API. It sits behind Cloudflare's
bot protection (TLS fingerprinting, JS challenges) -- a plain `requests`
call with a browser User-Agent header is NOT enough to get past this, and
attempting to hand-roll a bypass is exactly the kind of fragile, constantly-
breaking code this project's provider abstraction is meant to avoid.

Instead this provider delegates to `soccerdata` (https://github.com/probberechts/soccerdata),
a maintained, open-source package built specifically to keep working
against FBref's anti-bot measures as they change, while still respecting
the same "1 request per few seconds" politeness norm internally.
"""

from __future__ import annotations

import pandas as pd

from .base import FootballDataProvider, ProviderInfo


def _to_soccerdata_season(season: str) -> str:
    """'2024-2025' -> '2425' (soccerdata's compact season format)."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


class FBrefProvider(FootballDataProvider):
    info = ProviderInfo(
        name="FBref",
        url="https://fbref.com",
        license_note="Public site, no official API/ToS grant for bulk redistribution. "
        "Accessed via the soccerdata package, which caches responses locally by "
        "default -- don't disable that cache and re-pull repeatedly.",
        reliability_note="High depth (xG, xA, progressive actions) but structure can "
        "change between seasons -- treat column presence as season-dependent, not guaranteed. "
        "FBref's Cloudflare protection means occasional pulls may still fail even via "
        "soccerdata; retry after a pause rather than looping tightly.",
    )

    def fetch_player_season_stats(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        if league != "ENG-Premier League":
            raise ValueError("FBrefProvider (as configured) only targets the Premier League")

        try:
            import soccerdata as sd
        except ImportError as exc:
            raise ImportError(
                "FBrefProvider requires the 'soccerdata' package. "
                "Install it with: pip install soccerdata"
            ) from exc

        sd_season = _to_soccerdata_season(season)
        fbref = sd.FBref(leagues=league, seasons=sd_season)
        raw = fbref.read_player_season_stats(stat_type="standard").reset_index()

        # soccerdata returns FBref's standard-stats table as a two-level
        # column MultiIndex, e.g. ('Playing Time', 'Min'), ('Performance',
        # 'Gls') -- verified directly against a live pull rather than
        # guessed, since flattening naively collides ('Performance', 'Gls')
        # with ('Per 90 Minutes', 'Gls'). xG/xA columns are dropped entirely
        # by soccerdata when empty for the pulled season (a real FBref data-
        # availability gap, not a bug here) -- handled as genuinely missing,
        # never backfilled with a guess.
        def col(top: str, sub: str = "") -> pd.Series | None:
            key = (top, sub)
            return raw[key] if key in raw.columns else None

        df = pd.DataFrame(
            {
                "player_name": col("player"),
                "club": col("team"),
                "position": col("pos"),
                "minutes": _numeric(col("Playing Time", "Min")),
                "starts": _numeric(col("Playing Time", "Starts")),
                "apps": _numeric(col("Playing Time", "MP")),
                "goals": _numeric(col("Performance", "Gls")),
                "assists": _numeric(col("Performance", "Ast")),
                "xg": _numeric(col("Expected", "xG")),
                "xa": _numeric(col("Expected", "xAG")),
                "source_player_id": None,  # FBref player-page slug; fetch separately if needed for joins
                "season": season,
                "source": self.info.name,
            }
        )
        return df


def _numeric(series):
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")

