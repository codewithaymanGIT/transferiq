# TransferIQ — Premier League Market Intelligence Platform
### Phase 0: Technical Blueprint

> *"Football Market Intelligence"* — a gradient-boosted regression valuation engine trained on historical performance and transfer-market data, validated temporally and explained with SHAP.

---

## 1. Final Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind | SSR/ISR for fast dashboards, strong typing, huge ecosystem |
| Charts | Recharts (primary) + D3 (value-trajectory/percentile viz only) | Recharts covers 90% of needs with less code; D3 only where it earns its complexity |
| Animation | Framer Motion, used sparingly (page transitions, chart entrance) | |
| Backend | Python 3.11 + FastAPI | Async, Pydantic validation, auto OpenAPI docs, native ML ecosystem |
| Database | PostgreSQL 16 | Relational integrity for players/transfers/stats; window functions for per-90 & trend features |
| ORM/Migrations | SQLAlchemy 2.0 + Alembic | Typed models, versioned schema |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, SciPy | Free, open-source, industry-standard |
| Caching | In-process TTL cache (`cachetools`) locally; Redis optional if deployed | Avoid infra you don't need on localhost |
| Auth | FastAPI + `python-jose` (JWT) + `passlib` (bcrypt) | Optional layer, not the centerpiece |
| Testing | pytest (backend/ML), Vitest + React Testing Library (frontend), Playwright (E2E) | |
| Containerization | Docker + docker-compose (frontend, backend, postgres) | One-command local startup |
| CI | GitHub Actions (lint, test, build on PR) | Signals professional practice to recruiters |

---

## 2. System Architecture

```
┌────────────────────┐
│   Next.js Frontend  │  (dashboard, player profiles, scouting, comparisons)
└─────────┬───────────┘
          │ REST (JSON)
┌─────────▼───────────┐
│   FastAPI Backend    │  (auth, prediction serving, rankings, search)
│  ┌────────────────┐  │
│  │ Prediction Svc  │──┼──> loads serialized model (joblib/ONNX) + SHAP explainer
│  └────────────────┘  │
└─────────┬────────────┘
          │ SQLAlchemy
┌─────────▼───────────┐
│     PostgreSQL        │  (players, stats, transfers, predictions, users)
└─────────┬────────────┘
          │
┌─────────▼───────────┐
│   ETL / ML Pipeline   │  (offline, scheduled/manual — scripts/, not request-time)
│  ingest → clean →     │
│  validate → feature-  │
│  engineer → train →   │
│  evaluate → serialize │
└───────────────────────┘
```

Key principle: **training is offline**; the API only loads a frozen model artifact + feature-engineering function at request time. This separation is itself a resume point (offline batch ML vs. online serving).

---

## 3. Data Architecture

```
Raw Sources → data/raw/ (immutable, timestamped snapshots)
            → scripts/ingest/    (pull from APIs/scrapers, write raw)
            → scripts/clean/     (dedupe, type coercion, name normalization)
            → scripts/validate/  (schema + range checks, Great Expectations-style asserts)
            → scripts/transform/ (per-90, growth features, joins)
            → PostgreSQL (canonical store)
            → scripts/train/     (build training matrix with point-in-time correctness)
            → models/ (versioned .joblib + metadata.json)
```

Every raw pull is saved to disk before transformation, so the pipeline is replayable without re-hitting external sources — critical since scraped sources are unstable.

---

## 4. Database Schema (Core Tables)

```sql
players(id, fbref_id, name, dob, nationality, position, foot, height_cm, created_at)
clubs(id, name, league_id, country)
competitions(id, name, tier)
seasons(id, label, start_date, end_date)  -- e.g. "2023-2024"

player_season_stats(
  id, player_id, club_id, season_id,
  minutes, starts, apps,
  goals, assists, xg, xa, shots, sot,
  key_passes, prog_passes, prog_carries,
  tackles, interceptions, clearances, blocks,
  -- goalkeeper-specific, nullable for outfield:
  saves, save_pct, goals_conceded, psxg,
  source, ingested_at
)

transfers(
  id, player_id, from_club_id, to_club_id,
  transfer_date, fee_amount, fee_currency, fee_disclosed,
  transfer_type,  -- 'permanent' | 'loan' | 'free' | 'loan_with_option'
  season_id, source
)

market_values(id, player_id, valuation_date, value_amount, source)  -- benchmark, NOT the model's own prediction
predictions(id, player_id, model_version_id, predicted_value, low_bound, high_bound, confidence, predicted_at)
model_versions(id, name, trained_at, metrics_json, feature_list_json, hyperparams_json)
data_sources(id, name, url, license_note, last_synced_at, reliability_note)

users(id, email, password_hash, created_at)
watchlists(id, user_id, name)
watchlist_players(watchlist_id, player_id)
player_comparisons(id, user_id, player_ids[], created_at)
```

Constraints: FKs everywhere, `CHECK (minutes >= 0)`, unique `(player_id, season_id, club_id)` on stats, indexes on `player_id`, `season_id`, and a composite on `(position, season_id)` for ranking queries. Full ER diagram to be generated with `dbdiagram.io`/`schemaspy` in Phase 3 and committed to `docs/`.

**Critical distinction enforced at the schema level:** `market_values` (external benchmark) and `predictions` (this project's model output) are separate tables — they must never be conflated, per the project's own requirement.

---

## 5. Data-Source Options (researched)

| Source | Data | Format | License/Access | Reliability | Fallback |
|---|---|---|---|---|---|
| **FBref (via FBR API / fbrapi.com)** | Player & team standard + advanced stats (xG, xA, progressive actions), match logs, multi-season | REST, JSON | Public site; FBR API is a community wrapper — rate-limited (**1 request/3s**), no official ToS grant, so treat as "use respectfully, don't redistribute raw dumps" | High for stats depth; scraping-adjacent risk if FBref changes structure | Cache every pull to `data/raw/`; `soccerdata` Python package as an alternative access layer |
| **Understat** | xG/xA/shot-level data, Big-5 leagues | Scraped JSON embedded in pages | No official API; same caveats as FBref | Good for xG cross-checks | Use only supplementary, not sole source |
| **Fantasy Premier League official API** (`fantasy.premierleague.com/api/`) | ~700 current PL players: price, form, ICT index, per-gameweek history, ownership | Official public JSON API, **no auth required** | Genuinely public, stable, official Premier League product | Very high | N/A — this is already a fallback-grade source |
| **football-data.co.uk** | Historical match results, odds, 20+ leagues | Free CSV downloads | Explicitly free, public domain style | High, very stable | Good for club-context features |
| **Transfermarkt (community datasets, e.g. `dcaribou/transfermarkt-datasets` on GitHub)** | Historical transfer fees, market-value benchmarks, 10 leagues, ~30 seasons | Pre-scraped CSV/Parquet, refreshed periodically | Community-maintained; Transfermarkt's own ToS restricts scraping of the live site, so **use the pre-built dataset rather than scraping Transfermarkt directly** | Medium-high, depends on maintainer | If stale, document the cutoff date explicitly rather than re-scraping |
| **football-data.org** | Fixtures, competitions, basic standings | Official REST API, free tier with API key | Official, clear ToS, free-tier rate limits | High | Good for competitions/seasons metadata |

**Decision:** Build the training dataset primarily from (a) FBref-derived performance stats for features and (b) the Transfermarkt community dataset for transfer-fee targets and benchmark market values, joined on player name + DOB + club with a fuzzy-matching + manual-review step (name mismatches are the #1 practical failure mode in this kind of project — document this explicitly in `docs/`).

Every source gets a `data_sources` row and a short markdown file in `docs/data-sources/` recording exactly what was pulled, when, and under what constraint (e.g., "FBref: respect 1 req/3s, cached snapshot dated X, not redistributed").

**Data-provider abstraction:** an `ingest/providers/base.py` `Protocol`/ABC (`fetch_player_stats(season) -> DataFrame`) implemented per source, so swapping FBref for `soccerdata` or a paid API later doesn't touch downstream code.

---

## 6. Feature List (by group)

- **Identity/context:** age, age², position (one-hot), foot, height, club, league-tier of previous club
- **Attacking (per-90):** goals, assists, xG, xA, shots, SoT%, npxG, touches in box
- **Creativity (per-90):** key passes, prog. passes, prog. carries, crosses
- **Defensive (per-90):** tackles, interceptions, clearances, blocks, aerial win%
- **Goalkeeping (separate feature set):** save%, goals prevented (goals conceded − PSxG), clean sheet rate
- **Availability:** minutes/max-available-minutes, starts ratio, appearances
- **Longitudinal:** prior-season stats, 1-2-3-season deltas, career trend slope
- **Age × performance / age × position interactions**
- **Team context:** club's league-position, club's goal difference relative to league average
- **Engineered composites:** goal contribution (G+A), expected contribution (xG+xA), performance index (z-scored weighted blend, documented weighting rationale)

Every feature gets one line in `docs/ml-methodology.md` justifying inclusion — features without a stated rationale are cut (per the project's own "no meaningless features" rule).

---

## 7. Feature Engineering Methodology

1. Compute per-90 stats only above a **minimum-minutes threshold** (e.g., 450 mins/season); below threshold, apply empirical-Bayes shrinkage toward the positional mean rather than using raw small-sample per-90 values.
2. Build longitudinal features using only **strictly prior** seasons relative to the transfer date (see leakage prevention, §9).
3. Nonlinear age handling: raw age, age², and age-bucket dummy, left for the tree models to interact with position; linear models get explicit age×position interaction terms.
4. Log-transform the skewed target (see §8) and compare against raw-fee models empirically rather than assuming.

---

## 8. ML Target Definition

Two candidate targets, evaluated empirically, not assumed:

- **Target A:** `log(transfer_fee + 1)` for permanent transfers with disclosed fees.
- **Target B:** `market_value_benchmark` (from Transfermarkt-style dataset) for the larger set of players who haven't transferred recently — used for the "current valuation" use case shown on player profiles.

Free transfers, loans, and undisclosed fees are **excluded from Target A's training rows** (flagged, not imputed) but retained in the database for transparency. The production "Estimated Market Value" shown to users is modeled primarily on Target B (benchmark market value), since that's what most players have, with Target A used as a secondary model to compute the "Model vs. actual transfer fee" validation story in the ML report — this is the single biggest methodological decision and gets its own subsection in `docs/ml-methodology.md`.

---

## 9. Candidate Models & Validation Methodology

Models: median baseline → Linear Regression → Ridge/Lasso → Random Forest → Gradient Boosting → XGBoost → LightGBM/CatBoost.

**Temporal split** (adjust to actual data coverage):
```
Train:      2018–2023 seasons
Validation: 2024 season
Test:       2025–2026 seasons
```
No random k-fold on the full dataset — that would leak future information into training, since a player's 2024 value is correlated with their 2025 value. Group-aware temporal CV (`sklearn.model_selection.TimeSeriesSplit` adapted per-player) used for hyperparameter search within the training window only.

**Leakage checklist enforced in code review:**
- No feature computed using data dated after the row's `valuation_date`/`transfer_date`.
- Longitudinal features explicitly windowed (`WHERE season.end_date < transfer.transfer_date`).
- A unit test (`tests/ml/test_no_leakage.py`) that asserts max feature timestamp < target timestamp for every training row.

---

## 10. Evaluation Metrics & Explainability/Uncertainty Methodology

- MAE, RMSE, R², Median Absolute Error as primary metrics; MAPE reported but flagged as unreliable near low-fee players (discussed explicitly, not hidden).
- Model comparison table populated only from actual experiment runs, logged via a small `scripts/evaluate/report.py` that writes `docs/model-comparison.md` from real metrics — never hand-typed numbers.
- **Explainability:** SHAP `TreeExplainer` on the winning model; global importance (`docs/ml-methodology.md`) + per-player local explanation surfaced via `GET /api/players/{id}/valuation` and rendered as a waterfall chart on the player profile.
- **Uncertainty:** start with **quantile regression** (LightGBM/CatBoost native quantile loss, or `GradientBoostingRegressor(loss="quantile")`) for a defensible, well-understood interval; document as a stretch goal to compare against **conformal prediction** (`mapie` library) for coverage guarantees. State plainly that intervals reflect model uncertainty on historical patterns, not true market unpredictability (agent negotiations, release clauses, etc. — §42 of the spec).

---

## 11. Backend API Design

RESTful, versioned under `/api/v1`, Pydantic schemas for every request/response, structured logging, rate limiting on `/api/predict` and `/api/scouting/search`. Endpoints as specified in the brief (`/players`, `/players/{id}/valuation`, `/players/{id}/similar`, `/rankings/*`, `/predict`, `/scouting/search`, `/model/*`), each backed by a service layer (not fat route handlers) so ML logic is testable independent of FastAPI.

---

## 12. Frontend Architecture

Route structure mirrors the product surfaces: `/`, `/players`, `/players/[id]`, `/compare`, `/scouting`, `/rankings/[slug]`. Data fetching via server components + a typed API client generated from the FastAPI OpenAPI schema (`openapi-typescript`) — this keeps frontend/backend contracts in sync and is itself a nice resume line. Skeleton loading, empty, and error states as first-class components, not afterthoughts.

---

## 13. UI/UX Design System

Dark-first palette, one accent color (avoid "gradient soup"), a type scale with a distinct display font for large value numbers (e.g., £82.4M) vs. a workhorse font for tables. Design tokens (`frontend-design` skill will be consulted at implementation time) drive spacing/radius/shadow consistently rather than ad hoc Tailwind classes per component.

---

## 14. GitHub Repository Structure

As specified in the brief (`frontend/`, `backend/`, `ml/`, `data/`, `scripts/`, `notebooks/`, `tests/`, `docs/`, `database/`, `docker/`), with `docs/data-sources/`, `docs/ml-methodology.md`, `docs/model-comparison.md`, `docs/er-diagram.png` as committed artifacts, not just claims in the README.

---

## 15. Testing Strategy

- ML: feature-generation unit tests, leakage test, prediction-schema test, "model loads and predicts on a golden fixture" smoke test.
- Backend: pytest + `httpx.AsyncClient` for API tests, validation-error tests, auth tests.
- Frontend: component tests for the valuation card, SHAP waterfall, and comparison table; a handful of Playwright E2E flows (search → player profile → compare).

---

## 16. Docker/Local-Development Strategy

`docker-compose.yml` with three services (`frontend`, `backend`, `postgres`), a `Makefile` or `just` recipe for `make up`, `make seed`, `make train`. `.env.example` for both frontend and backend. Model artifacts mounted as a volume so retraining doesn't require rebuilding the backend image.

---

## 17. Development Phases

Phase 1 Data acquisition → Phase 2 Pipeline → Phase 3 Database → Phase 4 EDA → Phase 5 Feature engineering → Phase 6 ML experimentation → Phase 7 Final model → Phase 8 FastAPI backend → Phase 9 Frontend → Phase 10 Advanced analytics (scouting, similarity, comparisons) → Phase 11 Testing → Phase 12 Docker/documentation. Each phase should end with a working, demoable slice — not a dangling half-built layer.

---

## 18. Risks & Limitations

- **Source fragility:** FBref/Understat have no formal API guarantee; mitigated by the provider abstraction and cached raw snapshots.
- **Name-matching across sources:** the biggest practical time sink; budget real time for a fuzzy-match + manual-review step.
- **Small sample for expensive transfers:** the model will be least reliable at the extreme high end (£80M+ deals), which is exactly where error analysis (§41 of the brief) should focus.
- **Unmeasurable value drivers:** agent leverage, release clauses, club finances — explicitly out of scope, stated in the UI copy itself, not just the docs.

## 19. What Should NOT Be Implemented

- No LLM/generic AI anywhere in the valuation path.
- No natural-language search backed by an LLM in v1 — deterministic parsing first (§33 of the brief).
- No premature Redis/Kubernetes/microservices — the local-first Docker Compose setup is the right scale.
- No fabricated metrics, SHAP values, or player stats anywhere, including placeholder UI states (use clearly labeled "—" or skeletons instead).

## 20. What Makes This Resume-Worthy

A coherent, end-to-end system where a recruiter can trace one data point from a FBref scrape through a Postgres row, through a documented feature-engineering step, into a temporally-validated XGBoost model, out through a SHAP explanation, and onto a rendered player-profile card — with tests and Docker proving it's reproducible, not just a notebook. That full-stack + ML + data-engineering coherence, defensible in an interview, is the actual differentiator versus a typical portfolio project.

---

### Next step
Phase 1 (data acquisition): stand up the provider abstraction, pull one season of FBref stats + the Transfermarkt dataset, and get a raw snapshot committed to `data/raw/` before any cleaning code is written. Say the word and I'll start there.
