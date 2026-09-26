"""
Capture README screenshots of the running app into docs/screenshots/.

Players are chosen from the live API, never mocked:
  - valued player: the top row of /api/v1/predictions/top (or --valued-id)
  - unvalued player: the first player whose /valuation returns 404
    (or --unvalued-id)

Prerequisites:
    docker compose up --build        # postgres + backend + frontend, data loaded
    python scripts/train/generate_predictions.py
    pip install playwright           # browser: `python -m playwright install chromium`

Usage:
    python scripts/docs/take_screenshots.py
    python scripts/docs/take_screenshots.py --frontend http://localhost:3001 --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "screenshots"
DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}


def _get(url: str) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=15) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, None


def pick_valued_player(api: str) -> int:
    status, body = _get(f"{api}/api/v1/predictions/top?page=1&page_size=1")
    if status != 200 or not body or not body["items"]:
        raise SystemExit("No predictions in the database. Run scripts/train/generate_predictions.py first.")
    return int(body["items"][0]["player_id"])


def pick_unvalued_player(api: str, max_pages: int = 40) -> int:
    for include_inactive in ("false", "true"):
        for page in range(1, max_pages + 1):
            status, body = _get(
                f"{api}/api/v1/players?page={page}&page_size=100&include_inactive={include_inactive}"
            )
            if status != 200 or not body or not body["items"]:
                break
            for player in body["items"]:
                v_status, _ = _get(f"{api}/api/v1/players/{player['id']}/valuation")
                if v_status == 404:
                    return int(player["id"])
    raise SystemExit("Every player has a valuation; pass --unvalued-id to choose one explicitly.")


def shoot(page, url: str, path: Path, full_page: bool = False) -> None:
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(500)  # let web fonts settle
    page.screenshot(path=str(path), full_page=full_page)
    print(f"Wrote {path}  <- {url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", default="http://localhost:3001")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--valued-id", type=int)
    parser.add_argument("--unvalued-id", type=int)
    parser.add_argument("--out", type=Path, default=OUT, help="Output folder (default docs/screenshots)")
    args = parser.parse_args()

    status, _ = _get(f"{args.api}/health")
    if status != 200:
        raise SystemExit(f"Backend not reachable at {args.api}/health. Start it with `docker compose up`.")

    valued = args.valued_id or pick_valued_player(args.api)
    unvalued = args.unvalued_id or pick_unvalued_player(args.api)
    print(f"Valued player id={valued}, unvalued player id={unvalued}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fe = args.frontend.rstrip("/")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        desktop = browser.new_context(viewport=DESKTOP, color_scheme="dark")
        page = desktop.new_page()
        shoot(page, f"{fe}/", out / "dashboard.png")
        shoot(page, f"{fe}/players/{valued}", out / "player_valued.png")
        shoot(page, f"{fe}/players/{unvalued}", out / "player_no_valuation.png")
        # No dedicated model/metrics page exists; Rankings is the model-output view.
        shoot(page, f"{fe}/rankings", out / "rankings.png")
        desktop.close()

        mobile = browser.new_context(viewport=MOBILE, color_scheme="dark", device_scale_factor=1, is_mobile=True)
        page = mobile.new_page()
        shoot(page, f"{fe}/players/{valued}", out / "mobile_player.png")
        mobile.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
