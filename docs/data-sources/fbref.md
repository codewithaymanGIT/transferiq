# Data Source: FBref

- **URL:** https://fbref.com
- **Data available:** Standard, shooting, passing, defensive, possession, and goalkeeping stats per player per season; match logs; xG/xA for the Big 5 leagues only.
- **Access method:** `pandas.read_html` against the public "Standard Stats" table for a given season (`scripts/ingest/providers/fbref_provider.py`). No API key required; no official API exists.
- **License / usage restriction:** No published open-data license. FBref documents a scraping restriction of **1 request per 3 seconds**, enforced in code via `FBrefProvider._throttle()`. Raw pulls are cached to `data/raw/fbref/` and not re-scraped on every run. Not redistributed in bulk outside this project.
- **Update frequency:** Site updates 24–48h after matches; this project pulls once per season for training, not live.
- **Reliability:** High stat depth, but table structure/column names can shift between seasons — the provider treats missing columns as `None`, never fabricates values.
- **Fallback:** `FPLProvider` (official, current season only) for cross-checking; `soccerdata` Python package as an alternative access layer if direct scraping breaks.
- **Last synced:** not yet run in production — pending a machine with unrestricted network access (see README "Running ingestion").
