"""Screenshot the Analiza AI tab after running an analysis.

Run while uvicorn is up with `portfolio.example.yaml`:
    .venv/Scripts/python docs/_capture_analysis_screenshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).parent / "web-gui-analysis-screenshot.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1400},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Pozycje", timeout=20_000)

        # Switch to Analiza AI tab.
        page.click("text=Analiza AI")
        page.wait_for_selector("text=Analiza AI jeszcze nie uruchomiona", timeout=5_000)

        # Trigger analysis. Wait for the metrics card to render — that's the
        # last thing to appear in the new layout.
        page.click("button:has-text('Uruchom analizę'):not(:has(svg.lucide-loader))")
        page.wait_for_selector("text=Metryki ilościowe", timeout=60_000)
        page.wait_for_selector("text=Top korelacje", timeout=10_000)
        page.wait_for_timeout(1500)  # let animations settle

        page.screenshot(path=str(OUT), full_page=True)
        browser.close()

    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
