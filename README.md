# TransferIQ — Progress: Phases 1 & 3 (skipping ahead to a runnable backend)

This is the working slice of the platform described in `docs/phase-0-blueprint.md`
(the full architecture doc). Phases 1 (data acquisition) and 3 (database + API) are
now real, tested code: a provider abstraction, a normalized Postgres schema, a
SQLAlchemy ORM layer, and a FastAPI backend with endpoints backed by real queries —
`docker compose up` brings all three services online.

**Anti-fabrication contract, enforced by both the schema and the API:** `market_values`
(external benchmark) and `predictions` (this project's own model output) are separate
tables that are never conflated, and `GET /players/{id}/valuation` returns **404**, not
a placeholder number, until a real `Prediction` row exists — there's no trained model
yet (that's Phase 6/7), so returning anything else would be exactly the kind of
fabricated data the project brief rules out.

## What's here

```
scripts/ingest/
  providers/
    base.py            # FootballDataProvider ABC every source implements
    fpl_provider.py     # Official Fantasy Premier League API (no auth, most stable)
    fbref_provider.py   # FBref advanced stats (rate-limited: 1 req / 3s)
  run_ingest.py          # CLI entry point: pulls one season, writes a raw snapshot

tests/ml/
  fixtures/fpl_bootstrap_sample.json   # static fixture, no live network needed
  test_providers.py                     # contract tests for every provider

docs/data-sources/       # one file per source: license, reliability, fallback
data/raw/                # raw snapshots land here, one subfolder per provider
```

## Why a provider abstraction

Every provider implements the same interface (`fetch_player_season_stats`, optionally
`fetch_transfers` / `fetch_market_values`) and returns a DataFrame with the same
column contract. Nothing downstream (cleaning, feature engineering, training) imports
`fbref_provider` or `fpl_provider` directly — it depends only on `FootballDataProvider`.
That means adding a new source later, or swapping FBref for a paid API, changes one file.

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ml/test_providers.py -v
```

These run fully offline against a static fixture — they prove the shape contract,
not live connectivity.

## Running a real ingest

**Note:** this repo was scaffolded in a sandboxed environment whose network access
is restricted to package registries — it cannot reach `fbref.com` or
`fantasy.premierleague.com`. Run the commands below from a normal machine:

```bash
python scripts/ingest/run_ingest.py --provider fpl --season 2024-2025
python scripts/ingest/run_ingest.py --provider fbref --season 2024-2025
```

Each run writes `data/raw/<provider>/<season>_<utc-timestamp>.parquet` and never
overwrites a prior snapshot, so a bad pull doesn't destroy known-good data.

## Cleaning / validation / transform (Phase 2 & 4)

```
scripts/clean/name_matching.py        # normalize_name + name_similarity -- the join-key logic
scripts/clean/clean_player_stats.py    # type coercion, dedup, review flags (never fabricates missing values)
scripts/validate/validate_player_stats.py  # schema + range checks -> ValidationReport, before anything hits Postgres
scripts/transform/per90.py             # empirical-Bayes shrinkage per-90 stats (low-minute players pulled toward the positional mean, not left noisy)
scripts/transform/longitudinal.py      # prior-season growth features + nonlinear age features, with an explicit chronological-sort leakage guard
```

Run just these: `pytest tests/ml/test_clean_validate_transform.py -v` (16 tests, synthetic fixture data only).

## Frontend (pulled forward)

```
frontend/app/layout.tsx          # dark theme shell, Space Grotesk + IBM Plex Sans/Mono
frontend/app/page.tsx            # dashboard: aggregate stats, honest empty/error states
frontend/app/players/page.tsx    # player list with position filter
frontend/app/players/[id]/page.tsx  # profile page; shows "no model valuation yet" instead of a fake number
frontend/lib/api.ts              # typed client for the FastAPI backend
```

Design: dark charcoal-navy base (not pure black), muted gold accent reserved for money
figures, `Space Grotesk` headings / `IBM Plex Sans` body / `IBM Plex Mono` for all numeric
data so £ values and stats read like a financial terminal rather than generic SaaS cards.

**Verified in this environment:** `npx tsc --noEmit` passes clean, and `npm run build`
succeeds end-to-end (all 3 routes compile) once fonts can be fetched from Google Fonts —
this sandbox's network can't reach `fonts.googleapis.com`, so that step was verified
separately with local fonts substituted, then reverted. It'll build normally for you.

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
# or, full stack:
docker compose up --build   # now works end-to-end: postgres + backend + frontend
```

## Database (Phase 3)

```
database/schema.sql        # authoritative Postgres DDL (source for the ER diagram)
backend/app/models.py       # SQLAlchemy 2.0 ORM mirror of schema.sql
backend/app/db.py           # engine/session, reads DATABASE_URL
backend/app/schemas.py      # Pydantic API contracts (deliberately separate from the ORM shape)
backend/app/main.py         # FastAPI app: /health, /api/v1/players, /players/{id}, /players/{id}/valuation
```

## Backend (Phase 8, pulled forward)

```bash
pip install -r requirements.txt
pytest tests/backend/test_players_api.py -v   # 8 tests, all offline against in-memory SQLite

# full stack, real Postgres:
docker compose up --build
# API docs: http://localhost:8000/docs
```

## Running the tests (everything, offline)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

`tests/ml/test_providers.py` proves the provider shape contract against a static
fixture; `tests/backend/test_players_api.py` proves the API against an in-memory
Postgres-shaped SQLite DB, including the "no fabricated valuation" 404 path and the
`predictions_bounds_valid` constraint.

## Running a real ingest

**Note:** this repo was scaffolded in a sandboxed environment whose network access
is restricted to package registries — it cannot reach `fbref.com`,
`fantasy.premierleague.com`, or the Transfermarkt dataset's Cloudflare R2 host.
Run the commands below from a normal machine (or widen this environment's allowed
network domains in settings if you want me to run them here):

```bash
python scripts/ingest/run_ingest.py --provider fpl --season 2024-2025
python scripts/ingest/run_ingest.py --provider fbref --season 2024-2025
```

Each run writes `data/raw/<provider>/<season>_<utc-timestamp>.parquet` and never
overwrites a prior snapshot, so a bad pull doesn't destroy known-good data.

For transfer fees / market-value benchmarks, pull `dcaribou/transfermarkt-datasets`
(see `docs/data-sources/`) — it's distributed via DVC/Kaggle/data.world rather than
plain files in the GitHub repo, so it also needs to be fetched from a machine with
normal internet access.

## Next: Phase 5 / Phase 6

Feature-engineering assembly (join per-90 + longitudinal + transfer/market-value
targets into one training matrix, per docs/ml-methodology.md's target-definition
split) and the first ML experiments (baseline -> Ridge -> Random Forest -> XGBoost),
once real data has been pulled via the ingest scripts above on an unrestricted
machine and loaded into Postgres.
