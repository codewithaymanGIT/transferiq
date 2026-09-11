"""
Ingest entry point.

Usage:
    python scripts/ingest/run_ingest.py --season 2024-2025 --provider fpl
    python scripts/ingest/run_ingest.py --season 2024-2025 --provider fbref

Every pull is written to data/raw/<provider>/<season>_<utc-timestamp>.parquet
BEFORE any cleaning happens, so a bad/partial pull never corrupts prior
snapshots and the pipeline can be replayed offline.

Note: this sandbox's network access does not reach fbref.com or
fantasy.premierleague.com. Run this script on a machine with normal
internet access. Against the same interface, tests/ml/test_providers.py
proves the downstream shape contract using a static fixture instead of a
live call.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from providers.base import FootballDataProvider
from providers.fbref_provider import FBrefProvider
from providers.fpl_provider import FPLProvider
from providers.transfermarkt_provider import TransfermarktProvider

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

_STATS_PROVIDERS: dict[str, FootballDataProvider] = {
    "fpl": FPLProvider(),
    "fbref": FBrefProvider(),
}
_TRANSFER_PROVIDERS: dict[str, FootballDataProvider] = {
    "transfermarkt": TransfermarktProvider(),
}


def run_stats(provider_key: str, season: str) -> Path:
    provider = _STATS_PROVIDERS[provider_key]
    df = provider.fetch_player_season_stats(season=season)

    out_dir = RAW_DIR / provider_key
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{season}_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[{provider.info.name}] wrote {len(df)} rows -> {out_path}")
    return out_path


def run_transfers(provider_key: str) -> Path:
    """Transfer providers pull their full available history in one call
    (efficient for a source distributed as one CSV) rather than one
    request per season."""
    provider = _TRANSFER_PROVIDERS[provider_key]
    df = provider.fetch_transfers(season=None)

    out_dir = RAW_DIR / provider_key
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"transfers_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[{provider.info.name}] wrote {len(df)} transfer records -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull data from one provider.")
    parser.add_argument("--provider", choices=sorted(_STATS_PROVIDERS) + sorted(_TRANSFER_PROVIDERS), required=True)
    parser.add_argument("--season", help='e.g. "2024-2025" -- required for stats providers, ignored for transfer providers')
    args = parser.parse_args()

    if args.provider in _TRANSFER_PROVIDERS:
        run_transfers(args.provider)
    else:
        if not args.season:
            parser.error(f"--season is required for provider '{args.provider}'")
        run_stats(args.provider, args.season)
