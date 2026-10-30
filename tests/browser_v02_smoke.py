"""Optional real-Chromium E2E smoke. Run with: python tests/browser_v02_smoke.py."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "opencv_tutorial", "serve",
                             "--port", str(port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        time.sleep(1.0)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium",
                                        args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1350, "height": 1000})
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            # This runtime blocks Chromium localhost traffic. Bridge fetch through
            # Python HTTP so real server behavior and actual DOM interactions run.
            def bridge(path, options):
                payload = options.get("body")
                url = f"http://127.0.0.1:{port}{path}"
                request = Request(url, data=payload.encode() if payload else None,
                                  headers=options.get("headers", {}),
                                  method=options.get("method", "GET"))
                with urlopen(request, timeout=20) as response:
                    return {"status": response.status, "body": response.read().decode()}

            page.expose_function("bridgeToLab", bridge)
            html = (ROOT / "web/index.html").read_text(encoding="utf-8")
            html = html.replace('<script defer src="/app.js"></script>', '')
            html = html.replace('<link rel="stylesheet" href="/style.css">', '')
            page.set_content(html)
            page.add_style_tag(content=(ROOT / "web/style.css").read_text(encoding="utf-8"))
            page.evaluate('''(() => { window.fetch = async (url, options={}) => {
              const res = await window.bridgeToLab(url, options);
              return new Response(res.body, {status: res.status,
                headers: {'Content-Type':'application/json'}});
            }; return 0; })()''')
            page.add_script_tag(content=(ROOT / "web/app.js").read_text(encoding="utf-8"))
            page.evaluate("init()")
            page.locator("#processedImage").wait_for(state="visible")
            page.wait_for_function("document.querySelectorAll('.stage').length >= 3")
            print("base stages:", page.locator(".stage").count())
            page.locator("#playButton").click()
            page.wait_for_timeout(1050)
            print("stage after playback:", page.locator("#playStageLabel").inner_text())
            page.locator("#playButton").click()
            page.locator("#algorithmList button").nth(1).click()
            page.wait_for_function("document.querySelector('#status').textContent.includes('Finished')")
            page.locator("#studyButton").click()
            page.wait_for_function("document.querySelectorAll('#studyResults .candidate').length === 4", timeout=20000)
            print("study cards:", page.locator("#studyResults .candidate").count())
            print("study status:", page.locator("#studyStatus").inner_text())
            page.screenshot(path="/mnt/data/opencv-v02-browser.png", full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
