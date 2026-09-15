"""
FBref provider.

FBref (fbref.com) is the deepest free source for advanced stats (xG, xA,
progressive actions) but has no official API. It sits behind Cloudflare's
bot protection (TLS fingerprinting, JS challenges) -- a plain `requests`
call with a browser User-Agent header is NOT enough to get past this, and
attempting to hand-roll a bypass is exactly the kind of fragile, constantly-
breaking code this project's provider abstraction is meant to avoid.

Instead this provider delegates to `soccerdata` (https://github.com/probberechts/soccerdata),
a maintained, open-source package built specifically to keep working
against FBref's anti-bot measures as they change, while still respecting
the same "1 request per few seconds" politeness norm internally.
"""

from __future__ import annotations

import io

import pandas as pd

from .base import FootballDataProvider, ProviderInfo


def _to_soccerdata_season(season: str) -> str:
    """'2024-2025' -> '2425' (soccerdata's compact season format)."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


class FBrefProvider(FootballDataProvider):
    info = ProviderInfo(
        name="FBref",
        url="https://fbref.com",
        license_note="Public site, no official API/ToS grant for bulk redistribution. "
        "Accessed via the soccerdata package, which caches responses locally by "
        "default -- don't disable that cache and re-pull repeatedly.",
        reliability_note="High depth (xG, xA, progressive actions) but structure can "
        "change between seasons -- treat column presence as season-dependent, not guaranteed. "
        "FBref's Cloudflare protection means occasional pulls may still fail even via "
        "soccerdata; retry after a pause rather than looping tightly.",
    )

    def fetch_player_season_stats(self, season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
        if league != "ENG-Premier League":
            raise ValueError("FBrefProvider (as configured) only targets the Premier League")

        try:
            import soccerdata as sd
        except ImportError as exc:
            raise ImportError(
                "FBrefProvider requires the 'soccerdata' package. "
                "Install it with: pip install soccerdata"
            ) from exc

        sd_season = _to_soccerdata_season(season)
        fbref = sd.FBref(leagues=league, seasons=sd_season)
        raw = fbref.read_player_season_stats(stat_type="standard").reset_index()
        defense = _fetch_defense_stats(fbref)

        # soccerdata returns FBref's standard-stats table as a two-level
        # column MultiIndex, e.g. ('Playing Time', 'Min'), ('Performance',
        # 'Gls') -- verified directly against a live pull rather than
        # guessed, since flattening naively collides ('Performance', 'Gls')
        # with ('Per 90 Minutes', 'Gls'). xG/xA columns are dropped entirely
        # by soccerdata when empty for the pulled season (a real FBref data-
        # availability gap, not a bug here) -- handled as genuinely missing,
        # never backfilled with a guess.
        def col(top: str, sub: str = "") -> pd.Series | None:
            key = (top, sub)
            return raw[key] if key in raw.columns else None

        df = pd.DataFrame(
            {
                "player_name": col("player"),
                "club": col("team"),
                "position": col("pos"),
                "nationality": col("nation"),
                # FBref's "age" column is text like "27-045" (years-days) as of the
                # scrape date, not the season; we keep raw age separately from the
                # more reliable "born" (birth year, e.g. 1996), which is what
                # age-at-transfer should actually be computed from downstream.
                "age_raw": col("age"),
                "birth_year": _numeric(col("born")),
                "minutes": _numeric(col("Playing Time", "Min")),
                "starts": _numeric(col("Playing Time", "Starts")),
                "apps": _numeric(col("Playing Time", "MP")),
                "goals": _numeric(col("Performance", "Gls")),
                "assists": _numeric(col("Performance", "Ast")),
                "xg": _numeric(col("Expected", "xG")),
                "xa": _numeric(col("Expected", "xAG")),
                "source_player_id": None,  # FBref player-page slug; fetch separately if needed for joins
                "season": season,
                "source": self.info.name,
            }
        )
        if defense is not None and not defense.empty:
            # Join on player_name AND club, not just player_name -- a
            # player_name-only join created a Cartesian product for
            # mid-season transfers who appear twice (once per club) in both
            # tables, confirmed by finding real duplicated rows for players
            # like "Axel Disasi" after a name-only join.
            df = df.merge(defense, on=["player_name", "club"], how="left")
        else:
            df["tackles_won"] = None
            df["interceptions"] = None
        return df


def _numeric(series):
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def _fetch_defense_stats(fbref) -> pd.DataFrame | None:
    """FBref's defensive-actions page (tackles won, interceptions) is not
    exposed by soccerdata's read_player_season_stats -- only
    standard/keeper/shooting/playing_time/misc are supported there, even
    though the underlying page-fetch mechanism (fbref.get, which handles
    the Cloudflare bypass and local caching) is generic. This reuses that
    same mechanism directly with FBref's real 'defense' page/table-id
    convention, verified against a live pull rather than guessed.

    Note: FBref's raw 'Tkl' (tackle attempts), 'Blocks', and 'Clearances'
    columns are consistently blank in this table for outfield players as
    pulled here -- only 'tackles_won' (TklW) and 'interceptions' (Int) are
    reliably populated, so only those two are returned. Never backfilled
    with a guess for the blank columns.
    """
    try:
        from lxml import etree, html as lxml_html
    except ImportError:
        return None

    rows = []
    seasons = fbref.read_seasons()
    for (lkey, skey), season in seasons.iterrows():
        filepath = fbref.data_dir / f"players_{lkey}_{skey}_defense.html"
        url = (
            "https://fbref.com"
            + "/".join(season.url.split("/")[:-1])
            + "/defense/"
            + season.url.split("/")[-1]
        )
        try:
            reader = fbref.get(url, filepath)
            tree = lxml_html.parse(reader)
            (el,) = tree.xpath("//comment()[contains(.,'div_stats_defense')]")
            parser = etree.HTMLParser(recover=True)
            (html_table,) = etree.fromstring(el.text, parser).xpath(
                "//table[contains(@id, 'stats_defense')]"
            )
            # pd.read_html() in the installed pandas version (3.0.5) raises
            # FileNotFoundError on this valid HTML, misinterpreting the
            # string as a file path -- confirmed via two independent fix
            # attempts (decoding to str, wrapping in io.StringIO) that both
            # failed identically. Parsing the table manually via lxml's own
            # API sidesteps that bug entirely, and is arguably more robust
            # anyway since it keys off FBref's stable data-stat attributes
            # rather than display header text.
            table_rows = []
            # .//tbody/tr (not .//tr) found zero rows -- confirmed via an
            # actual isolated test returning None. etree.fromstring() on
            # this extracted table fragment (not a full document) doesn't
            # synthesize an implicit <tbody> the way a browser's HTML5
            # parser would, so tbody-anchored rows never match.
            for tr in html_table.xpath(".//tr"):
                tr_class = tr.get("class") or ""
                if "thead" in tr_class:
                    continue  # repeated mid-table header row
                row = {}
                for cell in tr.xpath(".//th | .//td"):
                    stat = cell.get("data-stat")
                    if stat:
                        # .text_content() only exists on lxml.html.HtmlElement,
                        # not the plain lxml.etree._Element returned by
                        # etree.fromstring() -- confirmed via an actual
                        # AttributeError, not guessed. itertext() works on both.
                        row[stat] = "".join(cell.itertext()).strip()
                if row.get("player") and row["player"] != "Player":
                    table_rows.append(row)
        except Exception:
            # Defensive-stats page unavailable for this pull -- degrade to
            # no defensive data rather than failing the whole ingest.
            continue

        if not table_rows:
            continue
        raw = pd.DataFrame(table_rows)
        rows.append(
            pd.DataFrame(
                {
                    "player_name": raw["player"],
                    # Real bug caught by inspecting actual output: mid-season
                    # transfers appear twice (once per club) in both this
                    # table and the standard-stats table. Merging on
                    # player_name alone created a Cartesian product (2x2=4
                    # rows for such players instead of 2) -- confirmed by
                    # finding real duplicated names like "Axel Disasi" and
                    # "James Ward-Prowse" with doubled row counts. Including
                    # club as a second join key fixes this.
                    "club": raw.get("team"),
                    "tackles_won": pd.to_numeric(raw.get("tackles_won"), errors="coerce"),
                    "interceptions": pd.to_numeric(raw.get("interceptions"), errors="coerce"),
                }
            )
        )
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

