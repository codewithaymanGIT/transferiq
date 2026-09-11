"""
Load the most recent FBref data pull into Postgres.

Usage:
    python scripts/load/load_to_postgres.py --season 2024-2025

Reads data/raw/fbref/<season>_*.parquet (the latest snapshot if several
exist), runs it through the same clean + validate pipeline covered by
tests/ml/test_clean_validate_transform.py, then upserts into the real
schema via load_core.load_dataframe. Aborts without writing anything if
validation finds missing required columns or unrecognizable positions
across the board -- a partial, silently-wrong load is worse than a loud
failure here.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "clean"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validate"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "load"))

from app.db import SessionLocal  # noqa: E402
from clean_player_stats import clean_player_season_stats  # noqa: E402
from validate_player_stats import validate_player_season_stats  # noqa: E402
from load_core import load_dataframe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("load_to_postgres")


def find_latest_snapshot(provider: str, season: str) -> Path:
    pattern = str(REPO_ROOT / "data" / "raw" / provider / f"{season}_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {provider} data found for {season}. "
            f"Run: python scripts/ingest/run_ingest.py --provider {provider} --season {season}"
        )
    return Path(files[-1])


def main(season: str) -> None:
    snapshot = find_latest_snapshot("fbref", season)
    logger.info("Reading %s", snapshot)
    df = pd.read_parquet(snapshot)
    logger.info("Raw rows: %d", len(df))

    df = clean_player_season_stats(df)
    n_flagged = (df["_clean_flags"].str.len() > 0).sum()
    if n_flagged:
        logger.info("%d rows flagged during cleaning (see _clean_flags column)", n_flagged)

    report = validate_player_season_stats(df)
    logger.info(report.summary())
    if report.missing_columns:
        raise ValueError(f"Required columns missing, aborting load: {report.missing_columns}")

    db = SessionLocal()
    try:
        result = load_dataframe(db, df, season=season, source_name="FBref", source_url="https://fbref.com")
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    logger.info(
        "Load complete -- players created: %d, players updated: %d, "
        "season-stat rows written: %d, skipped (missing identity): %d, "
        "skipped (unparseable position): %d",
        result.players_created,
        result.players_updated,
        result.stats_rows_written,
        result.rows_skipped_missing_identity,
        len(result.rows_skipped_unparseable_position),
    )
    if result.rows_skipped_unparseable_position:
        logger.info(
            "Players skipped for unparseable position (first 10): %s",
            result.rows_skipped_unparseable_position[:10],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2024-2025")
    args = parser.parse_args()
    main(args.season)
