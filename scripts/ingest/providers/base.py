"""
Data-provider abstraction.

Every external football-data source (FBref, the official FPL API, a
Transfermarkt-derived dataset, football-data.co.uk, ...) implements this
interface. Downstream cleaning/transform/training code depends only on
these methods and the DataFrame shapes they return -- never on a specific
source. That means swapping FBref for `soccerdata`, or adding a paid API
later, touches only one new provider file.

Design notes
------------
- Providers do network I/O (or read a local file) and return raw,
  minimally-typed pandas DataFrames. They do NOT clean, validate, or
  engineer features -- that happens in scripts/clean and scripts/transform.
- Every provider call should be paired with a raw snapshot written to
  data/raw/<source>/<what>_<timestamp>.parquet by the caller (see
  ingest/run_ingest.py), so the pipeline is replayable without re-hitting
  a fragile external source.
- Providers must be polite: respect documented rate limits internally
  (e.g. FBref: max 1 request / 3 seconds) rather than relying on the
  caller to throttle.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProviderInfo:
    """Static metadata surfaced to the data_sources table (see database/schema.sql)."""

    name: str
    url: str
    license_note: str
    reliability_note: str


class FootballDataProvider(abc.ABC):
    """Interface every ingestion source must implement."""

    info: ProviderInfo

    @abc.abstractmethod
    def fetch_player_season_stats(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        """
        Return one row per player for the given season.

        Required columns (nullable where genuinely unavailable, never
        fabricated): player_name, club, position, minutes, starts, apps,
        goals, assists, xg, xa, shots, sot, key_passes, prog_passes,
        prog_carries, tackles, interceptions, clearances, blocks,
        saves, save_pct, goals_conceded, source_player_id.
        """
        raise NotImplementedError

    def fetch_transfers(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        """
        Optional: return transfer records (player_name, from_club, to_club,
        transfer_date, fee_amount, fee_currency, transfer_type). Not every
        provider has this -- default is "not supported" rather than a
        fabricated empty guess.
        """
        raise NotImplementedError(f"{self.info.name} does not provide transfer data")

    def fetch_market_values(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        """Optional: return benchmark market values (player_name, valuation_date, value_amount, currency)."""
        raise NotImplementedError(f"{self.info.name} does not provide market-value benchmarks")
