"""Screenshot the Watchlist tab. Run with uvicorn up on :8765."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).parent / "web-gui-watchlist-screenshot.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Pozycje", timeout=20_000)
        page.click("text=Watchlist")
        page.wait_for_selector("text=Tickery które obserwujesz", timeout=10_000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
