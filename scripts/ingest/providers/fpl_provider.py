"""
Official Fantasy Premier League API provider.

This is the *most reliable* source in the stack: it's Premier League's own
public product, requires no API key, and has been stable for years. It only
covers the current season and current PL players, so it's used as (a) a
cross-check against FBref for current-season stats and (b) the fallback
source if FBref is unreachable or rate-limited.

Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/
No authentication. Be a reasonable citizen: this script makes exactly one
request per run.
"""

from __future__ import annotations

import pandas as pd
import requests

from .base import FootballDataProvider, ProviderInfo

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL's numeric element_type -> our position vocabulary
_POSITION_MAP = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}


class FPLProvider(FootballDataProvider):
    info = ProviderInfo(
        name="Fantasy Premier League (official API)",
        url=BOOTSTRAP_URL,
        license_note="Official Premier League public product, no auth required. "
        "No formal open-data license published; treat as look-but-don't-redistribute-raw-dumps.",
        reliability_note="Very high -- official, stable, current season only.",
    )

    def __init__(self, session: requests.Session | None = None, timeout: int = 15):
        self._session = session or requests.Session()
        self._timeout = timeout

    def fetch_player_season_stats(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        if league != "ENG-Premier League":
            raise ValueError("FPLProvider only covers the Premier League")

        resp = self._session.get(BOOTSTRAP_URL, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()

        teams_by_id = {t["id"]: t["name"] for t in payload["teams"]}
        rows = []
        for p in payload["elements"]:
            minutes = p.get("minutes", 0) or 0
            rows.append(
                {
                    "source_player_id": p["id"],
                    "player_name": f"{p['first_name']} {p['second_name']}".strip(),
                    "club": teams_by_id.get(p["team"]),
                    "position": _POSITION_MAP.get(p["element_type"]),
                    "minutes": minutes,
                    "starts": p.get("starts"),
                    "apps": None,  # not directly exposed; derive from starts+subs downstream if needed
                    "goals": p.get("goals_scored"),
                    "assists": p.get("assists"),
                    "xg": _to_float(p.get("expected_goals")),
                    "xa": _to_float(p.get("expected_assists")),
                    "shots": None,  # not exposed by this endpoint
                    "sot": None,
                    "key_passes": None,
                    "prog_passes": None,
                    "prog_carries": None,
                    "tackles": p.get("tackles"),
                    "interceptions": None,
                    "clearances": None,
                    "blocks": None,
                    "saves": p.get("saves"),
                    "save_pct": None,
                    "goals_conceded": p.get("goals_conceded"),
                    "season": season,
                    "source": self.info.name,
                }
            )
        return pd.DataFrame(rows)


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
