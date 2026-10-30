"""Optional bridged browser check. Requires playwright + locally installed chromium.

Run directly: python tests/browser_v03_smoke.py
Not included in CI because Chromium is not a declared dev dependency.
"""
from __future__ import annotations

import subprocess
import sys
import time
from urllib.request import Request, urlopen
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    port = 18767
    server = subprocess.Popen([sys.executable, "-m", "opencv_tutorial", "serve",
                               "--port", str(port)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=.4):
                    break
            except OSError:
                time.sleep(.15)
        else:
            raise RuntimeError("Local server could not start")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium",
                                        args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1366, "height": 860})
            failures = []
            page.on("pageerror", lambda error: failures.append(str(error)))
            # Browser JS requests are deliberately bridged through Python in this
            # restricted environment. This is *not* unmodified browser-network E2E.
            def fetch_in_python(url, options=None):
                options = options or {}
                body = options.get("body")
                body = body.encode() if body is not None else None
                request = Request(f"http://127.0.0.1:{port}{url}", data=body,
                                  headers=options.get("headers", {}),
                                  method=options.get("method", "GET"))
                with urlopen(request, timeout=15) as response:
                    return {"status": response.status,
                            "payload": __import__('json').load(response)}
            page.expose_function("bridgeFetch", fetch_in_python)
            html = (Path(__file__).resolve().parents[1] / "web/index.html").read_text()
            html = html.replace('<script defer src="/app.js"></script>', '')
            html = html.replace('<script defer src="/intermediate.js"></script>', '')
            page.set_content(html)
            page.evaluate("""(() => { window.fetch = async (url, opts = {}) => {
                const response = await window.bridgeFetch(url, opts);
                return {ok: response.status >= 200 && response.status < 300,
                    status: response.status, json: async () => response.payload};
            }; return null; })()""")
            for script in ("app.js", "intermediate.js"):
                page.add_script_tag(content=(Path(__file__).resolve().parents[1] /
                                             "web" / script).read_text())
            page.get_by_role("button", name="Stitch panorama").click()
            page.wait_for_function("document.querySelector('#intermediateMetrics').textContent.includes('inliers')")
            assert "candidate_matches" in page.locator("#intermediateMetrics").inner_text()
            page.locator("#pairMode").select_option("flow")
            page.get_by_role("button", name="Estimate optical flow").click()
            page.wait_for_function("document.querySelector('#intermediateMetrics').textContent.includes('median_dx')")
            assert '"tracked"' in page.locator("#intermediateMetrics").inner_text()
            page.get_by_role("button", name="Load synthetic sequence").click()
            page.wait_for_function("document.querySelector('#trackingEvents').textContent.includes('tracks') || document.querySelector('#trackingEvents').textContent.includes('id')")
            page.locator("#trackingFrame").fill("6")
            assert "Frame 7 / 8" in page.locator("#trackingLabel").inner_text()
            page.screenshot(path="/mnt/data/opencv-v03-browser.png", full_page=True)
            assert not failures, failures
            print("Browser bridge: panorama + sparse flow + tracking demo PASS")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
