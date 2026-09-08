-- TransferIQ database schema
-- PostgreSQL 16+. Applied via Alembic migrations in practice (see
-- backend/app/models.py + alembic/); this file is the human-readable
-- reference copy and the source of truth for the ER diagram.

CREATE TYPE position_group AS ENUM ('GK', 'DF', 'MF', 'FW');
CREATE TYPE preferred_foot AS ENUM ('LEFT', 'RIGHT', 'BOTH');
CREATE TYPE transfer_type AS ENUM ('PERMANENT', 'LOAN', 'FREE', 'LOAN_WITH_OPTION');
CREATE TYPE fee_currency AS ENUM ('EUR', 'GBP', 'USD');

-- ============================================================
-- Reference / dimension tables
-- ============================================================

CREATE TABLE competitions (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    tier            SMALLINT NOT NULL DEFAULT 1,
    country         TEXT
);

CREATE TABLE seasons (
    id              SERIAL PRIMARY KEY,
    label           TEXT NOT NULL UNIQUE,      -- e.g. '2024-2025'
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    CONSTRAINT seasons_valid_range CHECK (end_date > start_date)
);

CREATE TABLE clubs (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    competition_id  INTEGER REFERENCES competitions(id),
    country         TEXT,
    UNIQUE (name, competition_id)
);

CREATE TABLE data_sources (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    url                 TEXT,
    license_note        TEXT,
    reliability_note    TEXT,
    last_synced_at      TIMESTAMPTZ
);

-- ============================================================
-- Players and their attributes
-- ============================================================

CREATE TABLE players (
    id              SERIAL PRIMARY KEY,
    source_player_id TEXT,                      -- id in the primary source system, for joining
    name            TEXT NOT NULL,
    date_of_birth   DATE,
    nationality     TEXT,
    position        position_group NOT NULL,
    foot            preferred_foot,
    height_cm       SMALLINT CHECK (height_cm IS NULL OR height_cm BETWEEN 140 AND 220),
    current_club_id INTEGER REFERENCES clubs(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_players_name ON players (name);
CREATE INDEX idx_players_position ON players (position);

-- ============================================================
-- Performance data
-- ============================================================

CREATE TABLE player_season_stats (
    id                  BIGSERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    club_id             INTEGER REFERENCES clubs(id),
    season_id           INTEGER NOT NULL REFERENCES seasons(id),

    minutes             INTEGER NOT NULL DEFAULT 0 CHECK (minutes >= 0),
    starts              INTEGER CHECK (starts IS NULL OR starts >= 0),
    apps                INTEGER CHECK (apps IS NULL OR apps >= 0),

    -- attacking
    goals               INTEGER CHECK (goals IS NULL OR goals >= 0),
    assists             INTEGER CHECK (assists IS NULL OR assists >= 0),
    xg                  NUMERIC(6,2),
    xa                  NUMERIC(6,2),
    shots               INTEGER,
    shots_on_target     INTEGER,

    -- creativity / possession
    key_passes          INTEGER,
    prog_passes         INTEGER,
    prog_carries        INTEGER,
    pass_completion_pct NUMERIC(5,2),

    -- defensive
    tackles             INTEGER,
    interceptions       INTEGER,
    clearances          INTEGER,
    blocks              INTEGER,
    aerial_duel_pct     NUMERIC(5,2),

    -- goalkeeping (nullable for outfield players)
    saves               INTEGER,
    save_pct            NUMERIC(5,2),
    goals_conceded      INTEGER,
    psxg                NUMERIC(6,2),

    source_id           INTEGER REFERENCES data_sources(id),
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (player_id, season_id, club_id)
);
CREATE INDEX idx_pss_player ON player_season_stats (player_id);
CREATE INDEX idx_pss_season ON player_season_stats (season_id);
CREATE INDEX idx_pss_position_season ON player_season_stats (season_id, player_id);

-- ============================================================
-- Transfer market data
-- ============================================================

CREATE TABLE transfers (
    id                  BIGSERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    from_club_id        INTEGER REFERENCES clubs(id),
    to_club_id          INTEGER REFERENCES clubs(id),
    transfer_date       DATE NOT NULL,
    fee_amount          NUMERIC(12,2),           -- NULL when undisclosed
    fee_currency        fee_currency,
    fee_disclosed       BOOLEAN NOT NULL DEFAULT TRUE,
    transfer_type       transfer_type NOT NULL,
    season_id           INTEGER REFERENCES seasons(id),
    source_id           INTEGER REFERENCES data_sources(id),
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_transfers_player ON transfers (player_id);
CREATE INDEX idx_transfers_date ON transfers (transfer_date);

-- Benchmark market values from an external source (e.g. Transfermarkt).
-- Deliberately a *separate* table from `predictions` -- the model's own
-- output must never be conflated with this external benchmark.
CREATE TABLE market_values (
    id                  BIGSERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    valuation_date      DATE NOT NULL,
    value_amount        NUMERIC(12,2) NOT NULL,
    currency            fee_currency NOT NULL DEFAULT 'EUR',
    source_id           INTEGER REFERENCES data_sources(id),
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (player_id, valuation_date, source_id)
);
CREATE INDEX idx_mv_player ON market_values (player_id);

-- ============================================================
-- Modeling artifacts
-- ============================================================

CREATE TABLE model_versions (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,   -- e.g. 'xgb_quantile_v3'
    trained_at          TIMESTAMPTZ NOT NULL,
    metrics_json        JSONB NOT NULL,
    feature_list_json   JSONB NOT NULL,
    hyperparams_json    JSONB NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE predictions (
    id                  BIGSERIAL PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    model_version_id    INTEGER NOT NULL REFERENCES model_versions(id),
    predicted_value     NUMERIC(12,2) NOT NULL,
    low_bound           NUMERIC(12,2) NOT NULL,
    high_bound          NUMERIC(12,2) NOT NULL,
    confidence          TEXT NOT NULL,           -- 'High' | 'Medium' | 'Low' -- derived from interval width, see ml-methodology.md
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT predictions_bounds_valid CHECK (low_bound <= predicted_value AND predicted_value <= high_bound)
);
CREATE INDEX idx_predictions_player ON predictions (player_id);
CREATE INDEX idx_predictions_model ON predictions (model_version_id);

-- ============================================================
-- User-facing features (optional layer, not the centerpiece)
-- ============================================================

CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE watchlists (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, name)
);

CREATE TABLE watchlist_players (
    watchlist_id    INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    player_id       INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (watchlist_id, player_id)
);

CREATE TABLE player_comparisons (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE player_comparison_players (
    comparison_id   INTEGER NOT NULL REFERENCES player_comparisons(id) ON DELETE CASCADE,
    player_id       INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    PRIMARY KEY (comparison_id, player_id)
);
