"""Screenshot the Kalendarz tab with real data.

Assumes uvicorn is up on :8765 against the example portfolio AND a fresh
monitoring snapshot has been generated against the example data (so the
calendar has reports to show).

The caller is responsible for sandboxing the user's real
`reports/monitoring/` before generating the example snapshot, and
restoring it afterward — this script only captures.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).parent / "web-gui-calendar-screenshot.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1100},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Pozycje", timeout=20_000)
        page.click("text=Kalendarz")
        page.wait_for_selector("text=Raporty kwartalne", timeout=10_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
