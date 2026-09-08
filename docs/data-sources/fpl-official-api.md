# Data Source: Fantasy Premier League Official API

- **URL:** https://fantasy.premierleague.com/api/bootstrap-static/
- **Data available:** All current-season PL players — price, form, ICT index, ownership %, per-gameweek totals, minutes, goals, assists, xG, xA, saves, goals conceded. Current season only, no historical seasons.
- **Access method:** Single unauthenticated GET request (`scripts/ingest/providers/fpl_provider.py`). One request per ingest run — this endpoint returns the full dataset in one payload, no pagination needed.
- **License / usage restriction:** Official Premier League public product. No formal open-data license published; treated as look/analyze, not bulk-redistribute.
- **Update frequency:** Near real-time during the season (updates after each gameweek).
- **Reliability:** Very high — official, stable, has been publicly used by the FPL community for years without a formal API contract breaking.
- **Fallback:** N/A — this is itself the fallback for FBref. If this is ever unavailable, fall back to FBref's current-season table.
- **Last synced:** not yet run in production — pending a machine with unrestricted network access.
