# Transfermarkt (via ewenme/transfers)

**Data available:** Premier League transfer records, 1992/93 season onward. Player name,
position, age at transfer, both clubs, transfer direction, window (summer/winter), and
fee (raw + cleaned to EUR millions where disclosed).

**API/dataset format:** plain CSV, fetched via a single HTTP GET
(`raw.githubusercontent.com/ewenme/transfers/master/data/premier-league.csv`). No auth,
no rate limit beyond normal politeness.

**License/usage restrictions:** the source repo states data was scraped from
Transfermarkt "in accordance with their terms of use." Redistributed here as a maintained
CSV mirror, not re-scraped. Attribute Transfermarkt as the ultimate source of the
underlying data in anything derived from it.

**Update frequency:** the source repo has automated summer/winter update workflows, but
as verified on 2026-09-09 the Premier League file's most recent season present is
**2022/23** -- it does not yet include 2023-24 or 2024-25. Re-check before relying on
this for a "current" target.

**Reliability:** high for the fields it has. Of 23,675 total transfer records, only
~38% (9,040) have a disclosed numeric fee -- the rest are loans, free transfers, or
undisclosed fees, and are excluded from the fee-regression training target rather than
imputed (see `scripts/ingest/providers/transfermarkt_provider.py`, `fee_disclosed` column).

**Fallback:** `dcaribou/transfermarkt-datasets` has broader coverage (more leagues,
market-value benchmarks in addition to transfer fees) but is distributed via DVC/Kaggle/
Cloudflare R2 rather than plain files in the git repo, so it needs a separate pull
outside this project's HTTP-only ingest scripts.

**Known limitation for this project specifically:** this dataset's season coverage
(through 2022/23) does not overlap with the FBref performance data pulled so far
(2024-25 only). Building a genuine performance -> fee model requires performance stats
from the *same* seasons as the fees being predicted -- i.e. pulling FBref data for
2018-19 through 2022-23 as well, not just the current season.
