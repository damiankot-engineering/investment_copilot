"""Screenshot the Rebalancing tab with a computed plan. Run with uvicorn up on :8765."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).parent / "web-gui-rebalance-screenshot.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1050},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Pozycje", timeout=20_000)
        page.click("text=Rebalancing")
        page.wait_for_selector("text=Przelicz plan", timeout=10_000)
        page.click("text=Przelicz plan")
        # Plan summary appears once the server responds.
        page.wait_for_selector("text=Reszta gotówki", timeout=15_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
