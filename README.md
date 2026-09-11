# TransferIQ

Premier League player market-value intelligence platform, per the full architecture
in `docs/phase-0-blueprint.md`. This README tracks what's actually built and verified
working, not just planned.

**Status: data pipeline, database, backend, and frontend are real and running
end-to-end, with real player data pulled from live sources and loaded into Postgres.
The ML valuation model itself (Phase 6/7) hasn't been built yet.**

**Anti-fabrication contract, enforced by both the schema and the API:** `market_values`
(external benchmark) and `predictions` (this project's own model output) are separate
tables that are never conflated, and `GET /players/{id}/valuation` returns **404**, not
a placeholder number, until a real `Prediction` row exists.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
docker compose up --build          # postgres + backend + frontend
# API docs:  http://localhost:8000/docs
# Frontend:  http://localhost:3001
```

## Project layout

```
scripts/
  ingest/providers/    # FootballDataProvider ABC + FPL / FBref implementations
  ingest/run_ingest.py # CLI: pull one season from one provider, write a raw snapshot
  clean/                # name normalization, type coercion, dedup (never fabricates missing values)
  validate/             # schema/range checks -> ValidationReport, before anything hits Postgres
  transform/            # per-90 shrinkage stats, leakage-safe longitudinal/age features
  load/                 # DataFrame -> Postgres upsert, idempotent

backend/app/            # FastAPI + SQLAlchemy ORM (models.py mirrors database/schema.sql)
frontend/                # Next.js, dark theme, honest empty/error states
database/schema.sql      # authoritative Postgres DDL
docs/data-sources/        # one file per source: license, reliability, fallback
data/raw/                  # raw pulls land here, one subfolder per provider (gitignored)
tests/                      # ml/, backend/ -- 35 tests, all run offline
```

## Data sources (all verified against live pulls)

- **FPL official API** -- current player prices/form, official, no auth needed.
- **FBref** (via the `soccerdata` package) -- deep stats (minutes, goals, assists, xG
  where available). FBref sits behind Cloudflare bot protection; `soccerdata` handles
  this with real browser automation (`seleniumbase`), which on some Windows machines
  needs a Defender/antivirus exclusion for its driver folder. See `docs/data-sources/fbref.md`.
  **Currently loaded: 2024-25 season only.**
- **Transfermarkt** (via `ewenme/transfers`, a maintained CSV mirror) -- real historical
  transfer fees, 1992/93 through 2022/23. Already pulled: 23,675 records, 9,040 with a
  disclosed fee. See `docs/data-sources/transfermarkt.md` for the important caveat: this
  does **not** cover the current season, so it doesn't season-match the FBref pull above yet.

```bash
python scripts/ingest/run_ingest.py --provider fpl --season 2024-2025
python scripts/ingest/run_ingest.py --provider fbref --season 2024-2025
python scripts/ingest/run_ingest.py --provider transfermarkt   # full history in one pull, no --season
```

Each run writes `data/raw/<provider>/<season>_<utc-timestamp>.parquet` and never
overwrites a prior snapshot.

## Loading into Postgres

```bash
python scripts/load/load_to_postgres.py --season 2024-2025
```

Reads the latest FBref snapshot, runs it through clean + validate, then upserts
players/clubs/season-stats. Idempotent -- re-running updates existing rows (matched
on player name + club/season) instead of duplicating them. Multi-position players
(e.g. FBref's "FW,MF") take the first listed position as primary. Rows with an
unparseable position or missing identity are skipped and counted, never silently lost.

## Frontend

Dark charcoal-navy base, muted gold accent reserved for £ figures, `Space Grotesk`
headings / `IBM Plex Sans` body / `IBM Plex Mono` for all numeric data. Player profile
pages show "no model valuation yet" rather than a placeholder number when no
`Prediction` row exists for that player.

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
```

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v      # 35 tests, fully offline
```

Covers: provider shape contracts (static fixture), cleaning/validation/transform logic
(synthetic data), the FastAPI backend (in-memory SQLite, including the "no fabricated
valuation" 404 path), and the Postgres loader (in-memory SQLite, including idempotency).

## Next: the ML model (Phase 6/7)

**Real status as of the first full run:** the leakage-safe join between transfer fees
and prior-season stats currently produces **70 matched training rows** (out of 735
candidate disclosed-permanent transfers) -- limited by how many historical FBref seasons
have been loaded. That's genuinely too small for a trustworthy model, but the full
pipeline is real and working end to end:

```bash
python scripts/train/build_training_matrix.py     # assembles data/processed/training_matrix.parquet
python scripts/train/train_baseline_models.py      # median -> Linear -> Ridge -> RF -> XGBoost, writes docs/model-comparison.md
```

`train_baseline_models.py` auto-drops any feature column that's entirely null for the
current pull (e.g. `xg`/`xa`, which FBref doesn't return for every season pulled so far
-- never imputed), and flags the report itself as proof-of-concept when the sample is
below ~100 rows, rather than presenting small-sample metrics as if they meant something.

**To get a real, trustworthy model:** pull more historical FBref seasons (same command
as above, different `--season`, e.g. 2016-17 through 2018-19) to widen the season overlap
with the transfer-fee data, then re-run both scripts above. More rows, not more model
complexity, is the actual bottleneck right now.
