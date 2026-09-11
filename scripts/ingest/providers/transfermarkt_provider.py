"""
Transfer-fee provider.

Source: github.com/ewenme/transfers -- a maintained, plain-CSV mirror of
Transfermarkt transfer records (scraped in accordance with Transfermarkt's
terms of use, per that repo's own README), covering the Premier League
from the 1992/93 season. This is deliberately chosen over
dcaribou/transfermarkt-datasets: that dataset's `data/prep/` files are
distributed via DVC/Kaggle/R2 rather than plain files in the git repo, so
it can't be pulled with a simple HTTP GET the way this one can.

Known, documented limitation: as of the last check, this dataset's
Premier League file covers seasons through 2022/23 -- NOT the current
season. Any training pipeline using this as a target must match features
to the SAME season as each transfer record (see scripts/train/), not
naively join against current-season stats.
"""
from __future__ import annotations

import pandas as pd

from .base import FootballDataProvider, ProviderInfo

_CSV_URL = "https://raw.githubusercontent.com/ewenme/transfers/master/data/premier-league.csv"


class TransfermarktProvider(FootballDataProvider):
    info = ProviderInfo(
        name="Transfermarkt (via ewenme/transfers)",
        url="https://github.com/ewenme/transfers",
        license_note="Scraped from Transfermarkt in accordance with Transfermarkt's terms of "
        "use, per the source repo's README. Redistributed as a maintained CSV mirror, not "
        "re-scraped here. Attribute Transfermarkt as the ultimate data source.",
        reliability_note="Covers Premier League transfers from 1992/93 through 2022/23 as of "
        "the last check -- does NOT include the current season. ~38% of rows have a real "
        "numeric fee (fee_cleaned); the rest are loans, frees, or undisclosed and must be "
        "excluded from a fee-regression target, not imputed.",
    )

    def __init__(self, csv_source: str = _CSV_URL) -> None:
        """`csv_source` is injectable (a local file path works too) so tests
        can run offline against a small fixture instead of the live URL."""
        self.csv_source = csv_source

    def fetch_player_season_stats(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        raise NotImplementedError(f"{self.info.name} does not provide performance stats")

    def fetch_transfers(self, season: str | None = None, league: str = "ENG-Premier League") -> pd.DataFrame:
        """
        Returns transfer records. This provider's underlying CSV holds the
        full 1992/93-onward history in one file, so `season` filters the
        already-fetched data rather than triggering a separate request --
        pass None to get everything.
        """
        raw = pd.read_csv(self.csv_source)

        df = pd.DataFrame(
            {
                "player_name": raw["player_name"],
                "position_raw": raw["position"],
                "age_at_transfer": pd.to_numeric(raw["age"], errors="coerce"),
                "club": raw["club_name"],
                "counterparty_club": raw["club_involved_name"],
                "transfer_movement": raw["transfer_movement"],  # 'in' | 'out'
                "transfer_period": raw["transfer_period"],  # 'Summer' | 'Winter'
                "fee_raw": raw["fee"],
                "fee_eur_millions": pd.to_numeric(raw["fee_cleaned"], errors="coerce"),
                "fee_disclosed": pd.to_numeric(raw["fee_cleaned"], errors="coerce").notna(),
                "transfer_type": raw["fee"].map(_classify_transfer_type),
                "year": pd.to_numeric(raw["year"], errors="coerce"),
                "season": raw["season"].str.replace("/", "-", regex=False),  # '2022/2023' -> '2022-2023'
                "source": self.info.name,
            }
        )
        if season is not None:
            df = df[df["season"] == season].reset_index(drop=True)
        return df


def _classify_transfer_type(fee_raw: object) -> str | None:
    """
    Maps the source's free-text `fee` field to one of this project's
    transfer_type enum values. Returns None (not a fabricated guess) for
    '-' rows, which represent non-transfer bookkeeping entries (e.g.
    releases, unattached signings) rather than an actual transfer event --
    these are excluded from the loaded `transfers` table entirely.
    '?' is kept as PERMANENT with an undisclosed fee, since on Transfermarkt
    that specifically denotes a real paid transfer whose amount wasn't
    reported -- distinct from '-'.
    """
    if pd.isna(fee_raw):
        return None
    text = str(fee_raw).strip().lower()
    if text == "-":
        return None
    if text.startswith("loan") or text.startswith("end of loan"):
        return "LOAN"
    if "free" in text:
        return "FREE"
    return "PERMANENT"
