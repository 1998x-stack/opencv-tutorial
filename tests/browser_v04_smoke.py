"""Optional Chromium DOM + backend bridge smoke. Direct browser-to-HTTP is not certified.

Run: python tests/browser_v04_smoke.py (requires playwright and /usr/bin/chromium)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright


def main() -> None:
    port = 19044
    app = subprocess.Popen([sys.executable, "-m", "opencv_tutorial", "serve",
                            "--port", str(port)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=.4):
                    break
            except OSError:
                time.sleep(.12)
        else:
            raise RuntimeError("Local API is not serving")
        root = Path(__file__).resolve().parents[1]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True,
                                                 executable_path="/usr/bin/chromium",
                                                 args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            failures = []
            page.on("pageerror", lambda exc: failures.append(str(exc)))

            def bridge(path, opts=None):
                opts = opts or {}
                raw = opts.get("body")
                request = Request(f"http://127.0.0.1:{port}{path}",
                                  data=raw.encode() if raw is not None else None,
                                  headers=opts.get("headers", {}),
                                  method=opts.get("method", "GET"))
                try:
                    with urlopen(request, timeout=30) as response:
                        return {"status": response.status, "body": json.load(response)}
                except HTTPError as exc:
                    return {"status": exc.code, "body": json.load(exc)}

            page.expose_function("bridgeHTTP", bridge)
            html = (root / "web/index.html").read_text()
            for script in ("app.js", "intermediate.js", "advanced.js"):
                html = html.replace(f'<script defer src="/{script}"></script>', "")
            page.set_content(html)
            page.add_style_tag(content=(root / "web/style.css").read_text())
            page.evaluate("""(() => { window.fetch = async (url,opts={}) => {
              const r=await window.bridgeHTTP(url,opts);
              return {ok:r.status>=200&&r.status<300,status:r.status,json:async()=>r.body};
            }; return null; })()""")
            for script in ("app.js", "intermediate.js", "advanced.js"):
                page.add_script_tag(content=(root / "web" / script).read_text())
            page.wait_for_function("document.querySelector('#advancedStatus').textContent.includes('Sample loaded')")
            page.locator("#advancedRun").click()
            page.wait_for_function("document.querySelector('#advancedMetrics').textContent.includes('median_depth_m')")
            assert 'median_depth_m' in page.locator('#advancedMetrics').inner_text()
            assert page.locator('#advancedOutput').get_attribute('src').startswith('data:image')
            page.locator('#geometryLoad').click()
            page.wait_for_function("document.querySelector('#geometryMetrics').textContent.includes('rotation_error_deg')")
            assert 'translation_scale' in page.locator('#geometryMetrics').inner_text()
            page.locator('#advancedFx').fill('-1')
            page.locator('#advancedRun').click()
            page.wait_for_function("document.querySelector('#advancedStatus').textContent.includes('rejected')")
            assert not page.locator('#advancedOutput').get_attribute('src')
            page.locator('#advancedMode').select_option('vo')
            assert 'relative pose' in page.locator('#advancedRun').inner_text().lower()
            page.locator('#advanced').screenshot(path="/mnt/data/opencv-v04-browser.png")
            assert not failures, failures
            print("v0.4 bridge Chromium: stereo + geometry + negative input + VO mode PASS")
            browser.close()
    finally:
        app.terminate()
        try:
            app.wait(timeout=3)
        except subprocess.TimeoutExpired:
            app.kill()


if __name__ == "__main__":
    main()
