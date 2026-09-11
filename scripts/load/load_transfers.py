"""
Load the transfer-fee data into Postgres.

Usage:
    python scripts/load/load_transfers.py

Reads the latest data/raw/transfermarkt/transfers_*.parquet, loads real
transfer records for players already present in the `players` table
(i.e. run load_to_postgres.py for the relevant seasons FIRST -- players
not yet loaded are skipped and counted, never fabricated).
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "load"))

from app.db import SessionLocal  # noqa: E402
from load_core import load_transfers  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("load_transfers")


def find_latest_snapshot() -> Path:
    pattern = str(REPO_ROOT / "data" / "raw" / "transfermarkt" / "transfers_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            "No transfer data found. Run: python scripts/ingest/run_ingest.py --provider transfermarkt"
        )
    return Path(files[-1])


def main() -> None:
    snapshot = find_latest_snapshot()
    logger.info("Reading %s", snapshot)
    df = pd.read_parquet(snapshot)
    logger.info("Raw rows: %d", len(df))

    db = SessionLocal()
    try:
        result = load_transfers(
            db, df, source_name="Transfermarkt (via ewenme/transfers)", source_url="https://github.com/ewenme/transfers"
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    logger.info(
        "Load complete -- transfers created: %d, already loaded (skipped, idempotent): %d, "
        "skipped (non-transfer '-' rows): %d, skipped (player not yet in DB): %d",
        result.transfers_created,
        result.transfers_already_loaded,
        result.rows_skipped_non_transfer,
        len(result.rows_skipped_unmatched_player),
    )
    if result.rows_skipped_unmatched_player:
        sample = result.rows_skipped_unmatched_player[:10]
        logger.info(
            "Sample unmatched players (%d total) -- these need a matching FBref season pulled/loaded first: %s",
            len(result.rows_skipped_unmatched_player),
            sample,
        )


if __name__ == "__main__":
    main()
