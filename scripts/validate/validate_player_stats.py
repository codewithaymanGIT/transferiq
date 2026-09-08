"""
Validation pass: schema + range checks on cleaned player-season-stats data,
run before anything is loaded into Postgres. Raises `ValidationReport`
issues rather than silently dropping or fixing rows -- a human decides what
to do with genuinely bad data; this module's job is only to surface it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {
    "player_name", "name_key", "position", "minutes", "season", "source",
}

# (column, min, max) -- deliberately generous bounds; the goal is catching
# corrupted/misparsed data (e.g. minutes = 34200 from a unit mixup), not
# enforcing "realistic" performance, which the model should learn on its own.
_RANGE_CHECKS: list[tuple[str, float, float]] = [
    ("minutes", 0, 3420),          # max possible minutes in a 38-game PL season (+ some allowance)
    ("goals", 0, 60),
    ("assists", 0, 40),
    ("xg", 0, 60),
    ("xa", 0, 40),
    ("save_pct", 0, 100),
    ("pass_completion_pct", 0, 100),
    ("aerial_duel_pct", 0, 100),
]

_VALID_POSITIONS = {"GK", "DF", "MF", "FW"}


@dataclass
class ValidationReport:
    n_rows: int
    missing_columns: list[str] = field(default_factory=list)
    out_of_range: list[dict] = field(default_factory=list)
    invalid_positions: list[dict] = field(default_factory=list)
    duplicate_keys: list[dict] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.missing_columns or self.out_of_range or self.invalid_positions or self.duplicate_keys)

    def summary(self) -> str:
        lines = [f"Validated {self.n_rows} rows."]
        if self.missing_columns:
            lines.append(f"  MISSING COLUMNS: {self.missing_columns}")
        if self.out_of_range:
            lines.append(f"  OUT-OF-RANGE VALUES: {len(self.out_of_range)}")
        if self.invalid_positions:
            lines.append(f"  INVALID POSITIONS: {len(self.invalid_positions)}")
        if self.duplicate_keys:
            lines.append(f"  DUPLICATE (name_key, season, club) KEYS: {len(self.duplicate_keys)}")
        if self.is_clean:
            lines.append("  OK -- no issues found.")
        return "\n".join(lines)


def validate_player_season_stats(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(n_rows=len(df))

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        report.missing_columns = missing
        # Can't safely run row-level checks without the required columns.
        return report

    for col, lo, hi in _RANGE_CHECKS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = df[(values.notna()) & ((values < lo) | (values > hi))]
        for _, row in bad.iterrows():
            report.out_of_range.append(
                {"name_key": row.get("name_key"), "column": col, "value": row.get(col)}
            )

    bad_positions = df[~df["position"].isin(_VALID_POSITIONS)]
    for _, row in bad_positions.iterrows():
        report.invalid_positions.append({"name_key": row.get("name_key"), "position": row.get("position")})

    dedup_keys = [k for k in ("name_key", "season", "club") if k in df.columns]
    if dedup_keys:
        dupes = df[df.duplicated(subset=dedup_keys, keep=False)]
        for _, row in dupes.iterrows():
            report.duplicate_keys.append({k: row.get(k) for k in dedup_keys})

    return report
