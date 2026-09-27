"""
Snapshot the running API into frontend/data/demo.json for the read-only
public demo (NEXT_PUBLIC_DEMO_MODE=1). Every record is copied from the live
API response, never generated, so the demo shows exactly what the full
stack serves.

Includes: every current-season player (the API's default list, with birth
dates removed), each player's valuation (or null where the API returns
404), and the full ranking from the active model version.

Usage (backend running via docker compose):
    python scripts/docs/export_demo_data.py
    python scripts/docs/export_demo_data.py --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "frontend" / "data" / "demo.json"


def get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as res:
            return json.loads(res.read())
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def fetch_all(api: str, path: str) -> list[dict]:
    items, page = [], 1
    while True:
        sep = "&" if "?" in path else "?"
        body = get(f"{api}{path}{sep}page={page}&page_size=100")
        items += body["items"]
        if len(items) >= body["total"] or not body["items"]:
            return items
        page += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    api = parser.parse_args().api.rstrip("/")

    if get(f"{api}/health") is None:
        raise SystemExit(f"Backend not reachable at {api}. Start it with `docker compose up`.")

    players = fetch_all(api, "/api/v1/players")
    # Data minimisation: the demo never displays birth dates, so they are not
    # published. Names, clubs, positions and nationality are what the pages show.
    for p in players:
        p["date_of_birth"] = None
    top = fetch_all(api, "/api/v1/predictions/top")
    valuations = {}
    for p in players:
        valuations[str(p["id"])] = get(f"{api}/api/v1/players/{p['id']}/valuation")

    valued = sum(v is not None for v in valuations.values())
    snapshot = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "players": players,
        "valuations": valuations,
        "top": top,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Players: {len(players)}, with valuation: {valued}, ranked: {len(top)}")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
