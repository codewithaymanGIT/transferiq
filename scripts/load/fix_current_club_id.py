"""
One-time corrective script: recompute current_club_id for EVERY player
from their real, actual most-recent season's stats, fixing any player
whose current_club_id was incorrectly reverted backward in time by the
bug fixed in load_core.py's _upsert_player (backfilling an older season
after a newer one was already loaded used to unconditionally overwrite
current_club_id -- see that function's docstring for the full story).

Safe to re-run any time; it only ever sets current_club_id to whatever
a player's real most-recent PlayerSeasonStats row says, nothing more.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app import models  # noqa: E402
from app.db import SessionLocal  # noqa: E402


def main() -> None:
    db = SessionLocal()
    try:
        players = db.query(models.Player).all()
        fixed = 0
        for player in players:
            latest = (
                db.query(models.PlayerSeasonStats.club_id, models.Season.label)
                .join(models.Season, models.Season.id == models.PlayerSeasonStats.season_id)
                .filter(models.PlayerSeasonStats.player_id == player.id)
                .order_by(models.Season.label.desc())
                .first()
            )
            if latest is None:
                continue
            correct_club_id = latest[0]
            if player.current_club_id != correct_club_id:
                player.current_club_id = correct_club_id
                fixed += 1
        db.commit()
        print(f"Checked {len(players)} players, corrected current_club_id on {fixed}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
